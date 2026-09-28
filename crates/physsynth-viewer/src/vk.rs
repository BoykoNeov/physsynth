//! The von Kármán nonlinear plate — `serialize.py`'s vk builder (model #6: the gong and the cymbal).
//!
//! Two coupled fields and a fixed-point iteration per step, and NO analytic modal oracle: energy
//! conservation is the correctness test, and it holds only at the fixed point, so the verdict is
//! gated on convergence. The pitch HARDENS with amplitude, so the spectrum panel's marker lines are
//! the LINEAR (w -> 0) modes and the real peaks sit above them; the hardened fundamental is read by
//! zero-crossing spacing, and only on the supported gong — a free-edge crash is a mode wash.

use physsynth_core::exciter::raised_cosine_2d;
use physsynth_core::fmt::py_float;
use physsynth_core::plate::{self as pl, Boundary, VkParams, VkPlate, VkSpec};
use physsynth_core::pyfloat::scalar_pow;
use serde_json::{json, Value};

use crate::energy::{energy_block, EnergyOpts};
use crate::horizon::{horizon_none, HORIZON_NONLINEAR};
use crate::membrane::decimate_field_mask;
use crate::plate::{discrete_eigenfreqs, horizon_block};
use crate::py::{
    as_bool, b64f32, b64u8, finite_list, int, max_abs, np_mean, num, py_int, py_repr, py_str,
    repr_str, round_int, round_nd,
};
use crate::string::{
    construction, int_at_least_one, ANIM_WIN_MAX, FRAMES_PER_PERIOD, MAX_FRAMES, N_MIN, SPEED_MAX,
};
use crate::tension::pooled_spectrum;
use crate::{fnum, resample_normalize, Refusal, AUDIO_FS};

/// Grid ceiling.
pub const VK_N_MAX: i64 = 32;
/// Live-node ceiling (a free plate is every-node-live).
pub const VK_NLIVE_MAX: usize = 1_600;
/// Longest audio, seconds.
pub const VK_AUDIO_MAX: f64 = 1.0;
/// Oversample around the nonlinearity: the sample rate is the time control, in this range.
pub const VK_FS_MIN: f64 = 8_000.0;
/// See [`VK_FS_MIN`].
pub const VK_FS_MAX: f64 = 96_000.0;
/// Picard safety cap, and the worst-case cost multiplier.
pub const VK_COUPLE_MAX_ITER: i64 = 50;
/// `n_live x steps x couple_max_iter`.
pub const VK_WORK_MAX: f64 = 2.0e9;
/// Strike amplitude in thickness units — the hardening knob.
pub const VK_WOVERE_MAX: f64 = 6.0;
/// Linear eigenmodes marked on the spectrum panel.
pub const N_VK_MODES: usize = 6;

fn pf(x: f64) -> String {
    py_float(x)
}

/// Python's `min` over a sequence.
fn py_min(vals: &[f64]) -> f64 {
    let mut m = vals[0];
    for &v in &vals[1..] {
        if v < m {
            m = v;
        }
    }
    m
}

/// `_build_vk`.
fn build(p: &Value) -> Result<(VkPlate, f64, Boundary, bool), Refusal> {
    let bv = p.get("domain").cloned().unwrap_or(json!("supported"));
    let boundary = py_str(&bv);
    if boundary != "supported" && boundary != "free" {
        return Err(Refusal::Param(format!(
            "boundary must be 'supported' or 'free', got {}.",
            repr_str(&boundary)
        )));
    }
    let young = fnum(p, "E", 2.0e11)?;
    let e = fnum(p, "e", 1.0e-3)?;
    let nu = fnum(p, "nu", 0.3)?;
    let rho = fnum(p, "rho", 7800.0)?;
    let lx = fnum(p, "Lx", 0.4)?;
    let ly = fnum(p, "Ly", 0.4)?;
    let fs = fnum(p, "fs", 32_000.0)?;
    let sigma = fnum(p, "sigma", 0.0)?;
    let nonlinear = as_bool(p.get("nonlinear"), true);
    let nv = p.get("N").cloned().unwrap_or(json!(20));
    let n = py_int(&nv)
        .map_err(|_| Refusal::Param(format!("N must be an integer, got {}.", py_repr(&nv))))?;

    if !(N_MIN..=VK_N_MAX).contains(&n) {
        return Err(Refusal::Param(format!(
            "N must be in [{N_MIN}, {VK_N_MAX}] for the von Kármán plate, got {n}."
        )));
    }
    if py_min(&[young, e, rho, lx, ly]) <= 0.0 {
        return Err(Refusal::Param(
            "E, e, rho, Lx, Ly must all be positive.".into(),
        ));
    }
    if sigma < 0.0 {
        return Err(Refusal::Param(format!(
            "sigma (loss) must be >= 0, got {}.",
            pf(sigma)
        )));
    }
    if !(VK_FS_MIN..=VK_FS_MAX).contains(&fs) {
        return Err(Refusal::Param(format!(
            "fs must be in [{VK_FS_MIN:.0}, {VK_FS_MAX:.0}] Hz, got {}.",
            pf(fs)
        )));
    }
    let b = if boundary == "supported" {
        Boundary::Supported
    } else {
        Boundary::Free
    };
    let spec = VkSpec {
        lx,
        ly,
        young,
        thickness: e,
        nu,
        rho,
        fs,
        n,
        sigma,
        boundary: Some(b),
        nonlinear,
        couple_max_iter: VK_COUPLE_MAX_ITER,
        ..VkSpec::default()
    };
    let params = VkParams::new(&spec).map_err(construction)?;
    if params.lin.n_live > VK_NLIVE_MAX {
        return Err(Refusal::Param(format!(
            "this plate has {} live nodes (> {VK_NLIVE_MAX}); reduce N or the aspect ratio.",
            params.lin.n_live
        )));
    }
    Ok((VkPlate::new(params), fs, b, nonlinear))
}

/// `_vk_strike`: the (1,1) mode on the gong (a pure mode is what makes the hardened fundamental
/// read cleanly off zero-crossings), a broad raised-cosine crash on the cymbal; zero off the mask.
fn strike(
    vk: &VkPlate,
    boundary: Boundary,
    amplitude: f64,
    centre: (f64, f64),
    wc: f64,
) -> Result<Vec<f64>, Refusal> {
    let lin = &vk.p.lin;
    let mut field: Vec<f64> = match boundary {
        Boundary::Supported => lin
            .x
            .iter()
            .zip(&lin.y)
            .map(|(&x, &y)| {
                amplitude
                    * (std::f64::consts::PI * x / lin.lx).sin()
                    * (std::f64::consts::PI * y / lin.ly).sin()
            })
            .collect(),
        Boundary::Free => {
            raised_cosine_2d(&lin.x, &lin.y, centre, wc, amplitude).map_err(construction)?
        }
    };
    for (v, &alive) in field.iter_mut().zip(lin.mask.flags()) {
        if !alive {
            *v = 0.0;
        }
    }
    Ok(field)
}

/// One run's record: energy, optional pickup and snapshots, and the Picard convergence record.
struct Run {
    time: Vec<f64>,
    energy: Vec<f64>,
    pickup: Vec<f64>,
    snapshots: Vec<(usize, Vec<f64>)>,
    convergence: Value,
}

/// `_run_vk`: `simulate`, plus the convergence stats the energy verdict is gated on.
fn run(
    vk: &mut VkPlate,
    num_steps: usize,
    pickup: Option<usize>,
    snapshot_stride: usize,
) -> Result<Run, Refusal> {
    let n = num_steps + 1;
    let mut r = Run {
        // `np.arange(n) * vk.k` — not `i / fs`, which can differ in the last bit
        time: (0..n).map(|i| i as f64 * vk.p.lin.k).collect(),
        energy: Vec::with_capacity(n),
        pickup: Vec::new(),
        snapshots: Vec::new(),
        convergence: Value::Null,
    };
    let (mut n_not_conv, mut worst, mut max_iters) = (0i64, 0.0f64, 0usize);
    r.energy.push(vk.energy());
    if let Some(i) = pickup {
        r.pickup.push(vk.u[i]);
    }
    if snapshot_stride > 0 {
        r.snapshots.push((0, vk.state()));
    }
    for i in 1..n {
        vk.step(None)
            .map_err(|e| Refusal::Internal(e.to_string()))?;
        if !vk.converged {
            n_not_conv += 1;
        }
        // Python's `max(worst, x)`: keeps `worst` unless `x` is strictly larger.
        if vk.last_residual > worst {
            worst = vk.last_residual;
        }
        max_iters = max_iters.max(vk.n_iters);
        r.energy.push(vk.energy());
        if let Some(j) = pickup {
            r.pickup.push(vk.u[j]);
        }
        if snapshot_stride > 0 && i % snapshot_stride == 0 {
            r.snapshots.push((i, vk.state()));
        }
    }
    r.convergence = json!({
        "all_converged": n_not_conv == 0,
        "n_not_converged": int(n_not_conv),
        "worst_residual": num(worst),
        "max_iters": int(max_iters as i64),
        "couple_tol": num(vk.p.couple_tol),
    });
    Ok(r)
}

/// `_zero_cross_fundamental`: `fs / (2 mean(diff(zero crossings)))`, robust to the hardening.
fn zero_cross_fundamental(sig: &[f64], fs: f64) -> f64 {
    let m = np_mean(sig);
    let neg: Vec<bool> = sig.iter().map(|v| (v - m).is_sign_negative()).collect();
    let zc: Vec<usize> = (0..neg.len().saturating_sub(1))
        .filter(|&i| neg[i] != neg[i + 1])
        .collect();
    if zc.len() < 3 {
        return f64::NAN;
    }
    let d: Vec<f64> = zc.windows(2).map(|w| (w[1] - w[0]) as f64).collect();
    fs / (2.0 * np_mean(&d))
}

/// `_vk_spectrum_block`: the pooled spectrum, the LINEAR markers and the hardening shift.
fn spectrum_block(pickup: &[f64], fs: f64, f_lin: &[f64], f0: f64) -> Value {
    if f_lin.is_empty() {
        return Value::Null;
    }
    let f1_lin = f_lin[0];
    let mut fmax = f_lin.iter().copied().fold(f64::NEG_INFINITY, f64::max) * 1.6;
    if f0.is_finite() && f0 > 0.0 {
        fmax = fmax.max(f0 * 1.6);
    }
    let Some((f_ds, m_ds)) = pooled_spectrum(pickup, fs, fmax) else {
        return Value::Null;
    };
    let shift = (f0.is_finite() && f0 > 0.0 && f1_lin > 0.0).then(|| 100.0 * (f0 / f1_lin - 1.0));
    json!({
        "kind": "vk",
        "freq": finite_list(&f_ds, Some(3)),
        "mag": finite_list(&m_ds, Some(5)),
        "fmax": num(round_nd(fmax, 3)),
        "modes_linear": finite_list(f_lin, Some(4)),
        "f1_linear": num(round_nd(f1_lin, 4)),
        "f0_detected": if f0.is_finite() { num(round_nd(f0, 4)) } else { Value::Null },
        "shift_pct": shift.map_or(Value::Null, |s| num(round_nd(s, 2))),
    })
}

/// `_build_payload_vk`.
pub fn build_payload(p: &Value) -> Result<Value, Refusal> {
    let audio_dur = fnum(p, "audio_duration", 0.5)?;
    let anim_win = fnum(p, "animation_window", 0.02)?;
    let playback_speed = fnum(p, "playback_speed", 0.02)?;
    let w_over_e = fnum(p, "w_over_e", 2.0)?;
    let pluck_fx = fnum(p, "pluck_x", 0.5)?;
    let pluck_fy = fnum(p, "pluck_y", 0.5)?;
    let pluck_wfrac = fnum(p, "pluck_width", 0.28)?;
    let pickup_fx = fnum(p, "pickup_x", 0.47)?;
    let pickup_fy = fnum(p, "pickup_y", 0.53)?;
    let fpp = int_at_least_one(fnum(p, "frames_per_period", FRAMES_PER_PERIOD)?)?;

    if !(0.0 < audio_dur && audio_dur <= VK_AUDIO_MAX) {
        return Err(Refusal::Param(format!(
            "audio_duration must be in (0, {}] s, got {}.",
            pf(VK_AUDIO_MAX),
            pf(audio_dur)
        )));
    }
    if !(0.0 < anim_win && anim_win <= ANIM_WIN_MAX) {
        return Err(Refusal::Param(format!(
            "animation_window must be in (0, {}] s, got {}.",
            pf(ANIM_WIN_MAX),
            pf(anim_win)
        )));
    }
    if !(0.0 < playback_speed && playback_speed <= SPEED_MAX) {
        return Err(Refusal::Param(format!(
            "playback_speed must be in (0, {}], got {}.",
            pf(SPEED_MAX),
            pf(playback_speed)
        )));
    }
    if !(0.0 < w_over_e && w_over_e <= VK_WOVERE_MAX) {
        return Err(Refusal::Param(format!(
            "w_over_e must be in (0, {}], got {}.",
            pf(VK_WOVERE_MAX),
            pf(w_over_e)
        )));
    }
    for (name, v) in [
        ("pluck_x", pluck_fx),
        ("pluck_y", pluck_fy),
        ("pickup_x", pickup_fx),
        ("pickup_y", pickup_fy),
    ] {
        if !(0.0 < v && v < 1.0) {
            return Err(Refusal::Param(format!(
                "{name} must be in (0, 1), got {}.",
                pf(v)
            )));
        }
    }
    if !(0.0 < pluck_wfrac && pluck_wfrac <= 1.0) {
        return Err(Refusal::Param(format!(
            "pluck_width must be in (0, 1], got {}.",
            pf(pluck_wfrac)
        )));
    }

    let (mut vk, fs, boundary, nonlinear) = build(p)?;
    let (lx, ly) = (vk.p.lin.lx, vk.p.lin.ly);

    // Picard-aware budget: worst-case sweeps every step.
    let n_audio = round_int(audio_dur * fs).max(1);
    let n_anim_est = round_int(anim_win * fs).max(1);
    let iters = if nonlinear {
        vk.p.couple_max_iter as i64
    } else {
        1
    };
    let work = vk.p.lin.n_live as i64 * (n_audio + n_anim_est) * iters;
    if work as f64 > VK_WORK_MAX {
        return Err(Refusal::Param(format!(
            "this configuration needs up to ~{:.0}M coupled node-solves (over the ~{:.0}M \
             budget); reduce N, lower fs, or shorten the audio.",
            work as f64 / 1e6,
            VK_WORK_MAX / 1e6
        )));
    }

    // The LINEAR (w -> 0) modes: the markers the hardened peaks sit above.
    let f_lin = discrete_eigenfreqs(&vk.p.lin, N_VK_MODES)?;
    let f1_lin = if f_lin.is_empty() {
        vk.p.lin.kappa / (2.0 * scalar_pow(py_min(&[lx, ly]), 2.0))
    } else {
        f_lin[0]
    };

    // -- the audio run: a strike of w_over_e thicknesses
    let amplitude = w_over_e * vk.p.thickness;
    let wc = pluck_wfrac * py_min(&[lx, ly]);
    let centre = (pluck_fx * lx, pluck_fy * ly);
    let excite = |vk: &mut VkPlate| -> Result<(), Refusal> {
        let field = strike(vk, boundary, amplitude, centre, wc)?;
        let u0 = vk.p.to_live(&field);
        let v0 = vec![0.0; u0.len()];
        vk.set_state(&u0, &v0)
            .map_err(|e| Refusal::Internal(e.to_string()))
    };
    excite(&mut vk)?;
    let pickup_idx = pl::pickup_index_at(pickup_fx * lx, pickup_fy * ly, &vk.p.lin);
    let audio = run(&mut vk, n_audio as usize, Some(pickup_idx), 0)?;
    if !audio.pickup.iter().all(|v| v.is_finite()) {
        return Err(Refusal::Param(
            "simulation produced non-finite output (instability) — adjust parameters.".into(),
        ));
    }
    // The hardened fundamental: only honest for the gong's clean mode.
    let f0 = if boundary == Boundary::Supported {
        zero_cross_fundamental(&audio.pickup, fs)
    } else {
        f64::NAN
    };

    // -- the animation run: a fresh plate, a fundamental-resolving stride
    let mut anim = build(p)?.0;
    excite(&mut anim)?;
    let mut anim_stride = round_int((fs / f1_lin.max(1.0)) / fpp as f64).max(1);
    let n_anim = anim_stride.max(round_int(anim_win * fs));
    if n_anim / anim_stride > MAX_FRAMES {
        anim_stride = ((n_anim as f64 / MAX_FRAMES as f64).ceil() as i64).max(1);
    }
    let anim_run = run(&mut anim, n_anim as usize, None, anim_stride as usize)?;
    let frames_full: Vec<Vec<f64>> = anim_run.snapshots.iter().map(|(_, s)| s.clone()).collect();
    let frame_times: Vec<f64> = anim_run
        .snapshots
        .iter()
        .map(|(i, _)| *i as f64 / fs)
        .collect();
    let lin = &vk.p.lin;
    let (ny, nx) = (lin.mask.nrows(), lin.mask.ncols());
    let (frames_dec, mask_dec, ny_dec, nx_dec) =
        decimate_field_mask(&frames_full, lin.mask.flags(), ny, nx);
    let nf = frames_full.len();
    let field_amp = if frames_dec.is_empty() {
        0.0
    } else {
        max_abs(&frames_dec)
    };
    let (audio48, peak) = resample_normalize(&audio.pickup, fs);
    let of = "the von Karman plate";
    let horizon = if nonlinear {
        horizon_none(HORIZON_NONLINEAR, Some(of))
    } else {
        horizon_block(lin, of)
    };
    let sigma = lin.sigma;
    Ok(json!({
        "model": "vk",
        "horizon": horizon,
        "boundary": if boundary == Boundary::Supported { "supported" } else { "free" },
        "nonlinear": nonlinear,
        "fs_sim": num(round_nd(fs, 3)),
        "grid": {
            "dims": int(2), "nx": int(nx_dec as i64), "ny": int(ny_dec as i64),
            "extent_x": num(round_nd(lx, 6)), "extent_y": num(round_nd(ly, 6)),
            "domain": "rectangle",
        },
        "frames": {
            "b64": b64f32(&frames_dec),
            "n_frames": int(nf as i64), "nx": int(nx_dec as i64), "ny": int(ny_dec as i64),
            "width": int(nx_dec as i64), "dims": int(2),
        },
        "mask": {"b64": b64u8(&mask_dec), "nx": int(nx_dec as i64), "ny": int(ny_dec as i64)},
        "frame_times": finite_list(&frame_times, Some(6)),
        "anim_dt": num(anim_stride as f64 / fs),
        "playback_speed": num(playback_speed),
        "field_amp": num(field_amp),
        "audio": {
            "b64": b64f32(&audio48),
            "fs": num(AUDIO_FS),
            "peak": num(peak),
            "n": int(audio48.len() as i64),
        },
        "energy": energy_block(&audio.time, &audio.energy, sigma == 0.0, 2.0 * sigma,
                               EnergyOpts {
                                   convergence: nonlinear.then(|| audio.convergence.clone()),
                                   ..EnergyOpts::default()
                               }),
        "meta": {
            "kappa": num(round_nd(lin.kappa, 4)),
            "e": num(vk.p.thickness),
            "f1": num(round_nd(f1_lin, 3)),
            "num_steps": int(n_audio),
            "n_frames": int(nf as i64),
            "spectrum": spectrum_block(&audio.pickup, fs, &f_lin, f0),
        },
    }))
}
