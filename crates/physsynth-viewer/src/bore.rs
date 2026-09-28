//! The acoustic bore with a radiating bell — `serialize.py`'s bore builder, the first wind model.
//!
//! The field is PRESSURE along a tube. The structural fact the scene turns on: the bell's loss is
//! BOOKED — `energy = acoustic + radiated` — so a lossless tube with a radiating end still
//! conserves, and the energy panel plots the split (acoustic falling, radiated rising, sum flat).
//! The viscous `sigma` is not booked, so it is pinned at zero and never exposed; `R/Z0` is the
//! loss control.
//!
//! Three panels read the OPERATOR rather than a render: the discrete resonances (a generalized
//! `eigsh` on the free pressure nodes), the dispersion-versus-lambda curve (eighteen such solves,
//! no time-stepping), and the one-bounce reflection against `r = (R - Z0)/(R + Z0)`.

use physsynth_analysis::{modal, spectrum};
use physsynth_core::bore::{self as bore, Bore, End};
use physsynth_core::eigs::eigsh_shift_invert;
use physsynth_core::fmt::py_float;
use physsynth_core::pyfloat::scalar_pow;
use physsynth_core::sparse::Csr;
use serde_json::{json, Value};

use crate::contact::band_spectrum;
use crate::energy::{energy_block, EnergyOpts};
use crate::py::{
    b64f32, finite_list, float_to_int, int, max_abs, num, round_int, round_nd, IntError,
};
use crate::reed::{C0_AIR, RHO0_AIR};
use crate::string::{construction, int_at_least_one, FRAMES_PER_PERIOD, MAX_FRAMES, SPEED_MAX};
use crate::{fnum, resample_normalize, Refusal, AUDIO_FS};

/// Grid range.
pub const BORE_N_MIN: i64 = 32;
/// See [`BORE_N_MIN`].
pub const BORE_N_MAX: i64 = 256;
/// Longest audio, seconds.
pub const BORE_AUDIO_MAX: f64 = 1.5;
/// NOT the shared animation cap: at `N = 256` a 2 s window alone is over the step budget, and a
/// frame-count ceiling is not a cost ceiling.
pub const BORE_ANIM_MAX: f64 = 0.1;
/// Steps across the render AND the reflection run.
pub const BORE_WORK_MAX: i64 = 300_000;
/// Pinned: lambda is an operator claim here, not a render knob.
pub const BORE_LAMBDA: f64 = 1.0;
/// Bore radius (m): scales absolute energy and `Z0` only.
pub const BORE_RADIUS: f64 = 0.008;
/// Pressure bump amplitude (Pa); the model is linear, so this is pure scale.
pub const BORE_AMP: f64 = 1e-3;
/// Odd/even gate, set at the SHORTEST allowed duration (the ratio is a window property).
pub const BORE_ODD_EVEN_GATE: f64 = 1e3;
/// The bell's `R/Z0` range.
pub const BORE_R_RATIO_MIN: f64 = 1e-4;
/// See [`BORE_R_RATIO_MIN`].
pub const BORE_R_RATIO_MAX: f64 = 30.0;
/// Above this `R/Z0` no standing wave forms: labelled, never failed.
pub const BORE_RESONANT_RATIO: f64 = 0.05;
/// The lambdas of the dispersion curve.
pub const BORE_LAMBDA_CURVE: [f64; 9] = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.94, 0.97, 1.0];

fn pf(x: f64) -> String {
    py_float(x)
}

/// Python's `f"{x:.3e}"`: Rust writes `3.162e-4`, Python `3.162e-04` (a sign and two digits).
fn sci3(x: f64) -> String {
    let s = format!("{x:.3e}");
    match s.split_once('e') {
        Some((m, e)) => {
            let (sign, digits) = match e.strip_prefix('-') {
                Some(d) => ('-', d),
                None => ('+', e),
            };
            format!("{m}e{sign}{digits:0>2}")
        }
        None => s,
    }
}

/// `int(float)` refused the way the reference's uncaught `ValueError`/`OverflowError` was.
fn int_of(x: f64) -> Result<i64, Refusal> {
    match float_to_int(x) {
        Ok(i) => Ok(i),
        Err(IntError::NaN) => Err(Refusal::Construction(
            "cannot convert float NaN to integer".into(),
        )),
        Err(_) => Err(Refusal::Construction(
            "cannot convert float infinity to integer".into(),
        )),
    }
}

/// A bore with the reference's defaults for `rho0` and `c0`.
fn make_bore(l: f64, fs: f64, n: usize, right: End, r_bell: f64) -> Result<Bore, Refusal> {
    let p = bore::Params::new(
        l,
        fs,
        n,
        BORE_RADIUS,
        Some((End::Closed, right)),
        0.0,
        r_bell,
        RHO0_AIR,
        C0_AIR,
    )
    .map_err(construction)?;
    Ok(Bore::new(p))
}

/// `_bore_bump`: the suite's Gaussian pressure bump.
fn bump(x: &[f64], l: f64, center_frac: f64, width_frac: f64) -> Vec<f64> {
    let (c, w) = (center_frac * l, width_frac * l);
    x.iter()
        .map(|&xi| {
            let d = xi - c;
            BORE_AMP * (-(d * d) / (2.0 * w * w)).exp()
        })
        .collect()
}

/// `A[dof][:, dof]` for a sparse `A`.
fn submatrix(a: &Csr, dof: &[usize]) -> Csr {
    let mut pos = vec![usize::MAX; a.ncols()];
    for (k, &d) in dof.iter().enumerate() {
        pos[d] = k;
    }
    let rows: Vec<Vec<(usize, f64)>> = dof
        .iter()
        .map(|&i| {
            (a.indptr()[i]..a.indptr()[i + 1])
                .filter_map(|p| {
                    let j = a.indices()[p];
                    (pos[j] != usize::MAX).then(|| (pos[j], a.data()[p]))
                })
                .collect()
        })
        .collect();
    Csr::from_rows(dof.len(), dof.len(), rows)
}

/// `_bore_eigenfrequencies`: the lowest discrete resonances (Hz), from the OPERATOR.
pub fn eigenfrequencies(b: &Bore, n_modes: usize) -> Result<Vec<f64>, Refusal> {
    let p = b.params();
    let dof = p.dof();
    let (lop, cmat) = p.pressure_operator();
    let (lfree, cfree) = (submatrix(&lop, &dof), submatrix(&cmat, &dof));
    let n_open = usize::from(p.open_left()) + usize::from(p.open_right());
    let err = |e: physsynth_core::eigs::EigsError| Refusal::Internal(e.to_string());
    let w2: Vec<f64> = if n_open == 0 {
        // a closed tube has an omega = 0 mode: shift below it and drop it
        let shift = -1e-3 * scalar_pow(std::f64::consts::PI * p.c0 / p.l, 2.0);
        let v = eigsh_shift_invert(&lfree, Some(&cfree), shift, n_modes + 1).map_err(err)?;
        v.values[1..=n_modes].to_vec()
    } else {
        eigsh_shift_invert(&lfree, Some(&cfree), 0.0, n_modes)
            .map_err(err)?
            .values
    };
    Ok(w2
        .iter()
        .map(|&w| modal::discrete_bore_eigenfrequency(w, p.k))
        .collect())
}

/// What `_build_bore` reports beside the bore.
struct Info {
    l: f64,
    n: usize,
    fs: f64,
    lam: f64,
    z0: f64,
    ratio: f64,
    radiating: bool,
    r_bell: f64,
    resonant: bool,
}

/// `_build_bore`.
fn build(p: &Value) -> Result<(Bore, Info), Refusal> {
    let l = fnum(p, "L", 0.5)?;
    let n = int_of(fnum(p, "N", 128.0)?)?;
    let ratio_exp = fnum(p, "bell_ratio_exp", -3.5)?;
    let dv = p.get("domain").cloned().unwrap_or(json!("radiating"));
    let domain = crate::py::py_str(&dv);
    if domain != "radiating" && domain != "open" {
        return Err(Refusal::Param(format!(
            "bore end must be 'radiating' or 'open', got {}.",
            crate::py::repr_str(&domain)
        )));
    }
    if !(0.1..=2.0).contains(&l) {
        return Err(Refusal::Param(format!(
            "L must be in [0.1, 2.0] m, got {}.",
            pf(l)
        )));
    }
    if !(BORE_N_MIN..=BORE_N_MAX).contains(&n) {
        return Err(Refusal::Param(format!(
            "N must be in [{BORE_N_MIN}, {BORE_N_MAX}] for the bore, got {n}."
        )));
    }
    let ratio = scalar_pow(10.0, ratio_exp);
    if !(BORE_R_RATIO_MIN..=BORE_R_RATIO_MAX).contains(&ratio) {
        return Err(Refusal::Param(format!(
            "the bell's R/Z0 must be in [{}, {}], got {} (bell_ratio_exp = {}).",
            pf(BORE_R_RATIO_MIN),
            pf(BORE_R_RATIO_MAX),
            sci3(ratio),
            pf(ratio_exp)
        )));
    }
    let radiating = domain == "radiating";
    let z0 = RHO0_AIR * C0_AIR / (std::f64::consts::PI * BORE_RADIUS * BORE_RADIUS);
    let fs = C0_AIR / (BORE_LAMBDA * (l / n as f64));
    let r_bell = if radiating { ratio * z0 } else { 0.0 };
    let right = if radiating { End::Radiating } else { End::Open };
    let b = make_bore(l, fs, n as usize, right, r_bell)?;
    let info = Info {
        l,
        n: n as usize,
        fs,
        lam: b.params().lam,
        z0,
        ratio,
        radiating,
        r_bell: b.params().r_bell,
        resonant: !radiating || ratio <= BORE_RESONANT_RATIO,
    };
    Ok((b, info))
}

/// `_bore_reflection_block`: one bounce off the bell against `r = (R - Z0)/(R + Z0)`.
fn reflection_block(info: &Info) -> Result<Value, Refusal> {
    // `np.logspace(-4, 1.5, 79)` is `10 ** linspace(-4, 1.5, 79)` — not geomspace, which
    // overwrites its ends — and 1.0 is inserted EXACTLY: the anechoic null is the curve's point.
    let mut ratios = Vec::with_capacity(80);
    let (a, b) = (-4.0f64, 1.5f64);
    let step = (b - a) / 78.0;
    for i in 0..79 {
        let y = if i == 78 { b } else { a + i as f64 * step };
        ratios.push(scalar_pow(10.0, y));
    }
    ratios.push(1.0);
    ratios.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
    let curve: Vec<f64> = ratios
        .iter()
        .map(|&r| {
            let q = (r - 1.0) / (r + 1.0);
            0.5 * (1.0 - q * q)
        })
        .collect();
    let mut block = serde_json::Map::new();
    block.insert("radiating".into(), json!(info.radiating));
    block.insert(
        "curve".into(),
        json!({"ratio": finite_list(&ratios, Some(5)), "shed": finite_list(&curve, Some(6))}),
    );
    if !info.radiating {
        block.insert(
            "note".into(),
            json!("the ideal open end reflects perfectly (r = -1) - no radiation to measure."),
        );
        return Ok(Value::Object(block));
    }
    let mut bb = make_bore(info.l, info.fs, info.n, End::Radiating, info.r_bell)?;
    let x = bb.params().grid();
    let p0 = bump(&x, info.l, 0.5, 0.04);
    bb.set_state(&p0, &vec![0.0; info.n]);
    let e0 = bb.energy();
    for _ in 0..info.n {
        bb.step(None);
    }
    let shed = if e0 > 0.0 {
        bb.radiated_energy() / e0
    } else {
        f64::NAN
    };
    let r = (info.r_bell - info.z0) / (info.r_bell + info.z0);
    let oracle = 0.5 * (1.0 - r * r);
    let err = (shed - oracle).abs();
    block.insert("ratio".into(), num(info.ratio));
    block.insert("r".into(), num(round_nd(r, 6)));
    block.insert("oracle".into(), num(round_nd(oracle, 6)));
    block.insert("measured".into(), num(round_nd(shed, 6)));
    block.insert("abs_error".into(), num(err));
    block.insert("tol".into(), num(1e-9));
    block.insert("pass".into(), json!(err < 1e-9));
    block.insert("anechoic".into(), json!((info.ratio - 1.0).abs() < 1e-9));
    block.insert("steps".into(), int(info.n as i64));
    Ok(Value::Object(block))
}

/// `_bore_dispersion_block`: cents versus lambda, from the eigenvalue oracle alone.
fn dispersion_block(l: f64) -> Result<Value, Refusal> {
    let (n_modes, n_coarse, n_fine) = (5usize, 64usize, 128usize);
    let cont = modal::bore_resonance_frequencies(C0_AIR, l, n_modes, "closed-open")
        .expect("a known boundary");
    let worst = |lam: f64, n: usize| -> Result<f64, Refusal> {
        let b = make_bore(l, C0_AIR / (lam * (l / n as f64)), n, End::Open, 0.0)?;
        let f = eigenfrequencies(&b, n_modes)?;
        Ok(f.iter()
            .zip(&cont)
            .map(|(&a, &c)| modal::cents(a, c).abs())
            .fold(f64::NEG_INFINITY, f64::max))
    };
    let mut coarse = Vec::new();
    let mut fine = Vec::new();
    for &lam in &BORE_LAMBDA_CURVE {
        coarse.push(worst(lam, n_coarse)?);
    }
    for &lam in &BORE_LAMBDA_CURVE {
        fine.push(worst(lam, n_fine)?);
    }
    let order: Vec<Value> = coarse
        .iter()
        .zip(&fine)
        .map(|(&c, &f)| {
            if f > 0.0 {
                num(round_nd(c / f, 3))
            } else {
                Value::Null
            }
        })
        .collect();
    Ok(json!({
        "lambda": BORE_LAMBDA_CURVE.iter().map(|&x| num(round_nd(x, 4))).collect::<Vec<_>>(),
        "coarse": finite_list(&coarse, Some(5)),
        "fine": finite_list(&fine, Some(5)),
        "order": order,
        "n_coarse": int(n_coarse as i64),
        "n_fine": int(n_fine as i64),
        "n_modes": int(n_modes as i64),
    }))
}

/// `_bore_signature_block`: odd harmonics only, and the partials against BOTH oracles.
fn signature_block(pickup: &[f64], fs: f64, info: &Info) -> Result<Value, Refusal> {
    let n_partials = 5;
    let cont: Vec<f64> =
        modal::bore_resonance_frequencies(C0_AIR, info.l, n_partials, "closed-open")
            .expect("a known boundary")
            .into_iter()
            .filter(|&f| f < 0.45 * fs)
            .collect();
    let ideal = make_bore(info.l, fs, info.n, End::Open, 0.0)?;
    let mut eig = eigenfrequencies(&ideal, n_partials)?;
    eig.truncate(cont.len());
    let measured = spectrum::measure_partials_near(pickup, fs, &cont, None);

    let f1 = cont[0];
    let s = spectrum::magnitude_spectrum(pickup, fs, 2);
    let df = s.freqs[1] - s.freqs[0];
    let peak_near = |f: f64| -> f64 {
        let i = round_int(f / df);
        let lo = (i - 2).max(1) as usize;
        let hi = ((i + 3).max(0) as usize).min(s.mag.len());
        s.mag[lo..hi]
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max)
    };
    let odd: Vec<f64> = (1..6).map(|n| peak_near((2 * n - 1) as f64 * f1)).collect();
    let even: Vec<f64> = (1..6).map(|n| peak_near((2 * n) as f64 * f1)).collect();
    let max_even = even.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let min_odd = odd.iter().copied().fold(f64::INFINITY, f64::min);
    let ratio = if max_even > 0.0 {
        min_odd / max_even
    } else {
        f64::INFINITY
    };

    let f_max = (0.45 * fs).min(14.0 * f1);
    let (sf, sm, snorm) = band_spectrum(pickup, fs, f_max, 240);
    let norm = if snorm == 0.0 { 1.0 } else { snorm };
    let sm: Vec<f64> = sm.iter().map(|v| v / norm).collect();
    let cents = |a: &[f64], b: &[f64]| -> Vec<f64> {
        a.iter().zip(b).map(|(&x, &y)| modal::cents(x, y)).collect()
    };
    Ok(json!({
        "kind": "bore",
        "f1": num(round_nd(f1, 3)),
        "applies": info.resonant,
        "odd_even": {
            "ratio": num(ratio), "gate": num(BORE_ODD_EVEN_GATE),
            "pass": ratio > BORE_ODD_EVEN_GATE,
        },
        "partials": {
            "continuum": finite_list(&cont, Some(4)),
            "eigen": finite_list(&eig, Some(4)),
            "measured": finite_list(&measured, Some(4)),
            "cents_vs_continuum": finite_list(&cents(&measured, &cont), Some(4)),
            "cents_vs_eigen": finite_list(&cents(&measured, &eig), Some(4)),
            "eigen_vs_continuum": finite_list(&cents(&eig, &cont), Some(5)),
        },
        "spectrum": {
            "f_max": num(round_nd(f_max, 1)),
            "f": finite_list(&sf, Some(3)),
            "mag": finite_list(&sm, None),
        },
    }))
}

/// `_build_payload_bore`.
pub fn build_payload(p: &Value) -> Result<Value, Refusal> {
    let playback_speed = fnum(p, "playback_speed", 0.02)?;
    let pickup_frac = fnum(p, "pickup_position", 0.1)?;
    let audio_dur = fnum(p, "audio_duration", 0.5)?;
    let anim_win = fnum(p, "animation_window", 0.03)?;
    let fpp = int_at_least_one(fnum(p, "frames_per_period", FRAMES_PER_PERIOD)?)?;

    if !(0.0 < playback_speed && playback_speed <= SPEED_MAX) {
        return Err(Refusal::Param(format!(
            "playback_speed must be in (0, {}], got {}.",
            pf(SPEED_MAX),
            pf(playback_speed)
        )));
    }
    if !(0.0 < pickup_frac && pickup_frac < 1.0) {
        return Err(Refusal::Param(format!(
            "pickup_position must be in (0, 1), got {}.",
            pf(pickup_frac)
        )));
    }
    if !(0.0 < audio_dur && audio_dur <= BORE_AUDIO_MAX) {
        return Err(Refusal::Param(format!(
            "audio_duration must be in (0, {}] s, got {}.",
            pf(BORE_AUDIO_MAX),
            pf(audio_dur)
        )));
    }
    if !(0.0 < anim_win && anim_win <= BORE_ANIM_MAX) {
        return Err(Refusal::Param(format!(
            "animation_window must be in (0, {}] s, got {}.",
            pf(BORE_ANIM_MAX),
            pf(anim_win)
        )));
    }

    let (mut b, info) = build(p)?;
    let (fs, l, n) = (info.fs, info.l, info.n as i64);
    let f1 = C0_AIR / (4.0 * l);
    let n_steps = round_int(audio_dur * fs).max(1);
    let n_refl = if info.radiating { n } else { 0 };
    if n_steps + n_refl > BORE_WORK_MAX {
        return Err(Refusal::Param(format!(
            "this configuration needs {} steps across the render and the reflection run (budget \
             {BORE_WORK_MAX}). At lambda = 1, fs = c0*N/L — so N buys the sample rate, not just \
             the grid. Shorten audio_duration or lower N.",
            n_steps + n_refl
        )));
    }

    let pickup_idx = round_int(pickup_frac * n as f64).max(0).min(n - 1) as usize;
    // Pace on the TRANSIT, not on f1: the picture's claim is the bounce, four per period.
    let mut anim_stride = round_int((fs / (C0_AIR / l)) / fpp as f64).max(1);
    let n_anim = n_steps.min(anim_stride.max(round_int(anim_win * fs)));
    if n_anim / anim_stride > MAX_FRAMES {
        anim_stride = ((n_anim as f64 / MAX_FRAMES as f64).ceil() as i64).max(1);
    }

    let x = b.params().grid();
    b.set_state(&bump(&x, l, 0.12, 0.06), &vec![0.0; info.n]);
    // -- one run: pickup, both energy channels, frames, the running envelope
    let nodes = info.n + 1;
    let (mut e_ac, mut e_rad, mut pickup) = (Vec::new(), Vec::new(), Vec::new());
    let mut env = vec![0.0f64; nodes];
    let (mut frames, mut frame_steps, mut frame_rad) = (Vec::new(), Vec::new(), Vec::new());
    let sample =
        |b: &Bore, ea: &mut Vec<f64>, er: &mut Vec<f64>, pu: &mut Vec<f64>, env: &mut Vec<f64>| {
            ea.push(b.acoustic_energy());
            er.push(b.radiated_energy());
            pu.push(b.displacement_at(pickup_idx));
            for (e, v) in env.iter_mut().zip(b.p()) {
                let a = v.abs();
                // `np.maximum`: NaN-propagating
                if a > *e || a.is_nan() {
                    *e = a;
                }
            }
        };
    sample(&b, &mut e_ac, &mut e_rad, &mut pickup, &mut env);
    frames.push(b.p().to_vec());
    frame_steps.push(0usize);
    frame_rad.push(b.radiated_energy());
    for i in 1..=n_steps as usize {
        b.step(None);
        sample(&b, &mut e_ac, &mut e_rad, &mut pickup, &mut env);
        if i <= n_anim as usize && i % anim_stride as usize == 0 {
            frames.push(b.p().to_vec());
            frame_steps.push(i);
            frame_rad.push(b.radiated_energy());
        }
    }
    if !e_ac.iter().all(|v| v.is_finite()) {
        return Err(Refusal::Param(
            "simulation produced non-finite energy (instability) — adjust parameters.".into(),
        ));
    }

    let flat: Vec<f64> = frames.iter().flatten().copied().collect();
    let field_amp = if flat.is_empty() { 0.0 } else { max_abs(&flat) };
    let (audio48, peak) = resample_normalize(&pickup, fs);
    // THE VERDICT RIDES ON THE TOTAL; the split rides alongside for the panel.
    let e_total: Vec<f64> = e_ac.iter().zip(&e_rad).map(|(a, r)| a + r).collect();
    let time: Vec<f64> = (0..e_total.len()).map(|i| i as f64 / fs).collect();
    let e0 = if e_total[0] == 0.0 { 1.0 } else { e_total[0] };
    let rad_frames: Vec<f64> = frame_rad.iter().map(|v| v / e0).collect();
    let times: Vec<f64> = frame_steps.iter().map(|&i| i as f64 / fs).collect();
    let ends = json!(["closed", if info.radiating { "radiating" } else { "open" }]);

    Ok(json!({
        "model": "bore",
        "fs_sim": num(round_nd(fs, 3)),
        "lambda": num(round_nd(info.lam, 6)),
        "grid": {"x": finite_list(&x, Some(6)), "envelope": finite_list(&env, None)},
        "frames": {
            "b64": b64f32(&flat),
            "n_frames": int(frames.len() as i64),
            "width": int(nodes as i64),
            "dims": int(1),
        },
        "frame_times": finite_list(&times, Some(6)),
        "radiated_frames": finite_list(&rad_frames, Some(6)),
        "anim_dt": num(anim_stride as f64 / fs),
        "playback_speed": num(playback_speed),
        "field_amp": num(field_amp),
        "audio": {
            "b64": b64f32(&audio48),
            "fs": num(AUDIO_FS),
            "peak": num(peak),
            "n": int(audio48.len() as i64),
        },
        "energy": energy_block(&time, &e_total, true, 0.0, EnergyOpts {
            split: Some(vec![("acoustic", &e_ac), ("radiated", &e_rad)]),
            ..EnergyOpts::default()
        }),
        "meta": {
            "c": num(round_nd(C0_AIR, 3)),
            "f1": num(round_nd(f1, 3)),
            "num_steps": int(n_steps),
            "n_frames": int(frames.len() as i64),
            "probe_x": num(round_nd(x[pickup_idx], 4)),
            "ends": ends,
            "radiating": info.radiating,
            "r_ratio": if info.radiating { num(info.ratio) } else { Value::Null },
            "Z0": num(info.z0),
            "R_bell": num(info.r_bell),
            "radiated_frac": num(round_nd(e_rad[e_rad.len() - 1] / e0, 6)),
            "transit": num(round_nd(l / C0_AIR, 8)),
            "frames_per_transit": num(round_nd((fs * l / C0_AIR) / anim_stride as f64, 2)),
            "reflection": reflection_block(&info)?,
            "dispersion": dispersion_block(l)?,
            "spectrum": signature_block(&pickup, fs, &info)?,
        },
    }))
}
