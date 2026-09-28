//! A string on a bridge spring to a lumped modal body, read out to the far field —
//! `serialize.py`'s `_build_payload_body` (viewer batch 12).
//!
//! The first scene of the body stage. Its money panel is the three-way exchange
//! `E_string + E_body + E_conn`: the string's energy alone is NOT conserved once coupled (it sloshes
//! into the body), and the total is. The second panel is the radiated-pressure spectrum of the raw
//! volume acceleration, whose shape is exactly distance-invariant, with two honest read-outs: an
//! `omega²` consistency ratio that is a SANITY check (near-tautological by FFT linearity), and the
//! terminus fundamental gliding from `c/4L` towards `c/2L` as the spring stiffens.

use std::f64::consts::PI;

use physsynth_analysis::spectrum::{hann, rfft_mag, rfftfreq};
use physsynth_core::body::{self as body, ModalBody};
use physsynth_core::connection::StringBodyBridge;
use physsynth_core::exciter::triangular_pluck;
use physsynth_core::fmt::py_float;
use physsynth_core::pyfloat::scalar_pow;
use physsynth_core::radiation::{AirParams, AirRadiation};
use physsynth_core::string_ideal::{self as ideal, Boundary, IdealString};
use serde_json::{json, Value};

use crate::energy::{energy_block, EnergyOpts};
use crate::py::{
    b64f32, commas, finite_list, int, linspace_idx, max_abs, np_mean, num, py_int, py_repr,
    round_int, round_nd,
};
use crate::radbody::{BODY_BODY_FREQS, BODY_BODY_MASS};
use crate::reed::{C0_AIR, RHO0_AIR};
use crate::string::SPEED_MAX;
use crate::string::{construction, int_at_least_one, FRAMES_PER_PERIOD, MAX_FRAMES, N_MIN};
use crate::tension::N_SPEC_POINTS;
use crate::{fnum, resample_normalize, Refusal, AUDIO_FS};

/// Grid ceiling: each step is one leapfrog and one body step.
pub const BODY_N_MAX: i64 = 160;
/// Below 1 required: the spring pushes the Nyquist mode unstable at 1.
pub const BODY_LAM_DEFAULT: f64 = 0.9;
/// The bridge spring (N/m); the exact guard trips ~21.5k at the default rig.
pub const BODY_K_DEFAULT: f64 = 8000.0;
/// Body loss off by default: the conservation headline first.
pub const BODY_SIGMA_BODY_DEFAULT: f64 = 0.0;
/// Body-loss ceiling.
pub const BODY_SIGMA_BODY_MAX: f64 = 80.0;
/// Listening radius (m) for the 1/r read-out.
pub const BODY_DISTANCE_DEFAULT: f64 = 1.0;
/// Listening-radius ceiling (m).
pub const BODY_DISTANCE_MAX: f64 = 8.0;
/// Pluck amplitude (m).
pub const BODY_AMP_DEFAULT: f64 = 1e-3;
/// Longest audio run, seconds.
pub const BODY_AUDIO_MAX: f64 = 3.0;
/// Steps of the single instrumented run.
pub const BODY_WORK_MAX: i64 = 200_000;
/// Seconds of string animation (the body is lumped — no shape to draw).
pub const BODY_ANIM_WIN: f64 = 0.06;
/// Seconds of the exchange trace plotted (~9 sloshes).
pub const BODY_EXCHANGE_WINDOW: f64 = 0.4;
/// Decimated exchange-trace length.
pub const BODY_TRACE_POINTS: usize = 600;

fn pf(x: f64) -> String {
    py_float(x)
}

/// The rig and what the payload needs to know about it.
struct Rig {
    bridge: StringBodyBridge<ModalBody>,
    c: f64,
    l: f64,
    fs: f64,
    lam: f64,
    k: f64,
    sigma_body: f64,
}

/// `_build_body_bridge`: a fixed/free string on a `ModalBody` through a spring.
fn build(p: &Value) -> Result<Rig, Refusal> {
    let l = fnum(p, "L", 1.0)?;
    let t = fnum(p, "T", 200.0)?;
    let rho = fnum(p, "rho", 0.005)?;
    let lam = fnum(p, "lambda", BODY_LAM_DEFAULT)?;
    let k = fnum(p, "bridge_stiffness", BODY_K_DEFAULT)?;
    let sigma_body = fnum(p, "sigma_body", BODY_SIGMA_BODY_DEFAULT)?;
    let nv = p.get("N").cloned().unwrap_or(json!(100));
    let n = py_int(&nv)
        .map_err(|_| Refusal::Param(format!("N must be an integer, got {}.", py_repr(&nv))))?;
    if !(N_MIN..=BODY_N_MAX).contains(&n) {
        return Err(Refusal::Param(format!(
            "N must be in [{N_MIN}, {BODY_N_MAX}] for the body bridge, got {n}."
        )));
    }
    let mut mn = l;
    for v in [t, rho] {
        if v < mn {
            mn = v;
        }
    }
    if mn <= 0.0 {
        return Err(Refusal::Param("L, T, rho must all be positive.".into()));
    }
    if !(0.0 < lam && lam < 1.0) {
        return Err(Refusal::Param(format!(
            "lambda must be in (0, 1), got {}: the string's Nyquist mode is marginal at lambda = 1 \
             and the bridge spring pushes it unstable, so the coupled system needs headroom below \
             it.",
            pf(lam)
        )));
    }
    if k < 0.0 {
        return Err(Refusal::Param(format!(
            "bridge_stiffness must be >= 0, got {}.",
            pf(k)
        )));
    }
    if !(0.0..=BODY_SIGMA_BODY_MAX).contains(&sigma_body) {
        return Err(Refusal::Param(format!(
            "sigma_body must be in [0, {}], got {}.",
            pf(BODY_SIGMA_BODY_MAX),
            pf(sigma_body)
        )));
    }
    let c = (t / rho).sqrt();
    let fs = c * n as f64 / (l * lam);
    let sp = ideal::Params::new(
        l,
        t,
        rho,
        fs,
        n,
        0.0,
        Some((Boundary::Fixed, Boundary::Free)),
    )
    .map_err(construction)?;
    let string = IdealString::new(sp);
    let lam_s = string.params().lam;
    let m = BODY_BODY_FREQS.len();
    let bp = body::Params::new(
        BODY_BODY_FREQS.to_vec(),
        fs,
        vec![sigma_body; m],
        vec![BODY_BODY_MASS; m],
        vec![1.0; m],
        None,
    )
    .map_err(construction)?;
    // The exact coupled stability guard fires HERE if K exceeds the ceiling.
    let bridge = StringBodyBridge::new(string, ModalBody::new(bp), k).map_err(construction)?;
    Ok(Rig {
        bridge,
        c,
        l,
        fs,
        lam: lam_s,
        k,
        sigma_body,
    })
}

/// `np.abs(np.fft.rfft((sig - sig.mean()) * np.hanning(n)))` and its frequency axis.
pub fn windowed_mag(sig: &[f64], fs: f64, remove_mean: bool) -> (Vec<f64>, Vec<f64>) {
    let n = sig.len();
    let mean = if remove_mean { np_mean(sig) } else { 0.0 };
    let w = hann(n);
    let x: Vec<f64> = if remove_mean {
        sig.iter().zip(&w).map(|(s, wi)| (s - mean) * wi).collect()
    } else {
        sig.iter().zip(&w).map(|(s, wi)| s * wi).collect()
    };
    (rfftfreq(n, 1.0 / fs), rfft_mag(&x))
}

/// `_body_pooled_spectrum`: the band `[0, f_max]`, max-pooled to `n_points`, normalized.
pub fn pooled_spectrum(sig: &[f64], fs: f64, f_max: f64, n_points: usize) -> (Vec<f64>, Vec<f64>) {
    let (freqs, mag) = windowed_mag(sig, fs, true);
    let (mut f, mut m): (Vec<f64>, Vec<f64>) = freqs
        .iter()
        .zip(&mag)
        .filter(|(&fr, _)| fr <= f_max)
        .map(|(&fr, &mg)| (fr, mg))
        .unzip();
    if f.len() > n_points {
        let edges = linspace_idx(f.len(), n_points + 1);
        let (mut fo, mut mo) = (Vec::new(), Vec::new());
        for w in edges.windows(2) {
            let (a, b) = (w[0], w[1]);
            if b > a {
                fo.push(np_mean(&f[a..b]));
                mo.push(m[a..b].iter().copied().fold(f64::NEG_INFINITY, f64::max));
            }
        }
        f = fo;
        m = mo;
    }
    let top = if m.is_empty() {
        1.0
    } else {
        m.iter().copied().fold(f64::NEG_INFINITY, f64::max)
    };
    if top > 0.0 {
        for v in &mut m {
            *v /= top;
        }
    }
    (f, m)
}

/// `np.median` of a non-empty list: the middle value, or the mean of the middle two.
pub fn median(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = s.len();
    if n % 2 == 1 {
        s[n / 2]
    } else {
        // `np.median` averages the two middle values with `np.mean`: (a + b) / 2.
        (s[n / 2 - 1] + s[n / 2]) / 2.0
    }
}

/// `_body_omega2_consistency`: median of `|Q''| / |W_b| / (2 pi f)²` over the strong peaks.
pub(crate) fn omega2_consistency(qaccel: &[f64], wb: &[f64], fs: f64, f_max: f64) -> f64 {
    let (freqs, pqa) = windowed_mag(qaccel, fs, true);
    let (_, pwb) = windowed_mag(wb, fs, true);
    let band_max = freqs
        .iter()
        .zip(&pqa)
        .filter(|(&f, _)| f <= f_max)
        .map(|(_, &m)| m)
        .fold(None, |a: Option<f64>, m| Some(a.map_or(m, |x| x.max(m))));
    let Some(top) = band_max else {
        return f64::NAN;
    };
    if top <= 0.0 {
        return f64::NAN;
    }
    let thresh = 0.05 * top;
    let mut ratios = Vec::new();
    for i in 1..pqa.len().saturating_sub(1) {
        if freqs[i] > f_max {
            break;
        }
        let is_peak = pqa[i] >= pqa[i - 1] && pqa[i] > pqa[i + 1];
        if is_peak && pqa[i] > thresh && pwb[i] > 0.0 && freqs[i] > 0.0 {
            let w2 = scalar_pow(2.0 * PI * freqs[i], 2.0);
            ratios.push(pqa[i] / pwb[i] / w2);
        }
    }
    if ratios.is_empty() {
        f64::NAN
    } else {
        median(&ratios)
    }
}

/// `_body_terminus_f1`: the coupled fundamental from the free-end pickup, band from `c/4L..c/2L`.
pub(crate) fn terminus_f1(u_end: &[f64], fs: f64, c: f64, l: f64) -> f64 {
    let n = u_end.len();
    let (freqs, spec) = windowed_mag(u_end, fs, true);
    let (lo, hi) = (0.5 * c / (4.0 * l), 1.3 * c / (2.0 * l));
    let band: Vec<usize> = (0..freqs.len())
        .filter(|&i| freqs[i] > lo && freqs[i] < hi)
        .collect();
    if band.is_empty() {
        return f64::NAN;
    }
    // `np.argmax` over the band: the FIRST maximum.
    let mut j = band[0];
    for &i in &band {
        if spec[i] > spec[j] {
            j = i;
        }
    }
    let off = if 0 < j && j < spec.len() - 1 {
        let (a, b, cc) = (spec[j - 1], spec[j], spec[j + 1]);
        let denom = a - 2.0 * b + cc;
        if denom != 0.0 {
            0.5 * (a - cc) / denom
        } else {
            0.0
        }
    } else {
        0.0
    };
    (j as f64 + off) * fs / n as f64
}

/// `np.max` / `np.min` of a non-empty slice, NaN-propagating.
pub(crate) fn np_extreme(a: &[f64], max: bool) -> f64 {
    let mut m = if max {
        f64::NEG_INFINITY
    } else {
        f64::INFINITY
    };
    for &v in a {
        if v.is_nan() {
            return f64::NAN;
        }
        if (max && v > m) || (!max && v < m) {
            m = v;
        }
    }
    m
}

/// A read-out that may have nothing to measure: the number, or `null` when it is not finite.
pub(crate) fn opt_finite(x: f64) -> Value {
    if x.is_finite() {
        num(x)
    } else {
        Value::Null
    }
}

/// `_build_payload_body`.
pub fn build_payload(p: &Value) -> Result<Value, Refusal> {
    let playback_speed = fnum(p, "playback_speed", 0.02)?;
    let pluck_frac = fnum(p, "pluck_position", 0.3)?;
    let amplitude = fnum(p, "amplitude", BODY_AMP_DEFAULT)?;
    let audio_dur = fnum(p, "audio_duration", 2.0)?;
    let distance = fnum(p, "distance", BODY_DISTANCE_DEFAULT)?;
    let fpp = int_at_least_one(fnum(p, "frames_per_period", FRAMES_PER_PERIOD)?)?;
    if !(0.0 < playback_speed && playback_speed <= SPEED_MAX) {
        return Err(Refusal::Param(format!(
            "playback_speed must be in (0, {}], got {}.",
            pf(SPEED_MAX),
            pf(playback_speed)
        )));
    }
    if !(0.0 < pluck_frac && pluck_frac < 1.0) {
        return Err(Refusal::Param(format!(
            "pluck_position must be in (0, 1), got {}.",
            pf(pluck_frac)
        )));
    }
    if !(0.0 < audio_dur && audio_dur <= BODY_AUDIO_MAX) {
        return Err(Refusal::Param(format!(
            "audio_duration must be in (0, {}] s, got {}.",
            pf(BODY_AUDIO_MAX),
            pf(audio_dur)
        )));
    }
    if !(0.0 < distance && distance <= BODY_DISTANCE_MAX) {
        return Err(Refusal::Param(format!(
            "distance must be in (0, {}] m, got {}.",
            pf(BODY_DISTANCE_MAX),
            pf(distance)
        )));
    }

    let mut rig = build(p)?;
    let (c, l, fs, lam) = (rig.c, rig.l, rig.fs, rig.lam);
    let f1_base = c / (2.0 * l);
    let n_steps = round_int(audio_dur * fs).max(1);
    if n_steps > BODY_WORK_MAX {
        return Err(Refusal::Param(format!(
            "work budget exceeded ({} steps > {}). Lower the audio duration, N, or the tension.",
            commas(n_steps),
            commas(BODY_WORK_MAX)
        )));
    }
    let mut anim_stride = round_int((fs / f1_base) / fpp as f64).max(1);
    let frame_until = anim_stride.max(round_int(BODY_ANIM_WIN * fs));
    if frame_until / anim_stride > MAX_FRAMES {
        anim_stride = ((frame_until as f64 / MAX_FRAMES as f64).ceil() as i64).max(1);
    }
    let x = rig.bridge.string().params().grid();
    let u0 = triangular_pluck(&x, l, pluck_frac * l, amplitude).map_err(construction)?;
    rig.bridge.string_mut().set_state(&u0, &vec![0.0; u0.len()]);
    let air_p = AirParams::new(fs, distance, RHO0_AIR, C0_AIR, true).map_err(construction)?;
    let (gain, latency, residual) = (
        air_p.gain,
        air_p.latency_samples,
        air_p.retardation_residual,
    );
    let mut air = AirRadiation::new(air_p);

    // -- one instrumented run: the split energies, the bridge, the free end, and the read-out
    let n = n_steps as usize;
    let mut ch: [Vec<f64>; 8] = Default::default(); // E, string, body, conn, wb, u_end, Q'', p
    let (mut frames, mut frame_steps): (Vec<Vec<f64>>, Vec<usize>) = (Vec::new(), Vec::new());
    let k_spring = rig.bridge.stiffness();
    let mut sample = |b: &StringBodyBridge<ModalBody>, ch: &mut [Vec<f64>; 8]| {
        ch[0].push(b.energy());
        ch[1].push(b.string().energy());
        ch[2].push(b.body().energy());
        ch[3].push(0.5 * k_spring * b.stretch(false) * b.stretch(true));
        ch[4].push(b.body().bridge_displacement());
        let u = &b.string().u;
        ch[5].push(u[u.len() - 1]);
        // The raw Q'' drives the spectrum panel; the retarded far field is the audio only.
        ch[6].push(b.pressure());
        ch[7].push(air.process(b.pressure()));
    };
    sample(&rig.bridge, &mut ch);
    if frame_until >= 1 {
        frames.push(rig.bridge.string().u.clone());
        frame_steps.push(0);
    }
    for i in 1..=n {
        let _ = rig.bridge.step(&mut ());
        sample(&rig.bridge, &mut ch);
        if i as i64 <= frame_until && i as i64 % anim_stride == 0 {
            frames.push(rig.bridge.string().u.clone());
            frame_steps.push(i);
        }
    }
    let [total, e_string, e_body, e_conn, wb, u_end, qaccel, pressure] = ch;
    if !total.iter().all(|v| v.is_finite()) {
        return Err(Refusal::Param(
            "simulation produced non-finite energy (instability) — adjust parameters.".into(),
        ));
    }

    // -- the money panel: the E_string <-> E_body exchange
    let frac = |a: &[f64]| {
        a.iter()
            .zip(&total)
            .map(|(x, t)| x / t)
            .collect::<Vec<f64>>()
    };
    let (es, eb, ec, ones) = (frac(&e_string), frac(&e_body), frac(&e_conn), frac(&total));
    let n_exch = n.min(round_int(BODY_EXCHANGE_WINDOW * fs).max(1) as usize);
    let eidx = linspace_idx(n_exch, (n_exch + 1).min(BODY_TRACE_POINTS));
    let pick = |a: &[f64]| eidx.iter().map(|&i| a[i]).collect::<Vec<f64>>();
    let t_ex: Vec<f64> = eidx.iter().map(|&i| i as f64 / fs).collect();
    let e0 = total[0];
    let quarter = (n / 4).max(1);
    // `np.argmax`: the FIRST maximum.
    let mut first = 0;
    for i in 0..quarter {
        if e_body[i] > e_body[first] {
            first = i;
        }
    }
    let exchange = json!({
        "kind": "body",
        "time": finite_list(&t_ex, Some(6)),
        "e_string_frac": finite_list(&pick(&es), None),
        "e_body_frac": finite_list(&pick(&eb), None),
        "e_conn_frac": finite_list(&pick(&ec), None),
        "total_frac": finite_list(&pick(&ones), None),
        "window": num(round_nd(n_exch as f64 / fs, 4)),
        "body_frac_peak": num(round_nd(np_extreme(&eb, true), 4)),
        "string_frac_min": num(round_nd(np_extreme(&es, false), 4)),
        "string_frac_max": num(round_nd(np_extreme(&es, true), 4)),
        "first_peak_ms": num(round_nd(first as f64 / fs * 1000.0, 1)),
        "total_drift": num(if e0 != 0.0 {
            (np_extreme(&total, true) - np_extreme(&total, false)) / e0.abs()
        } else {
            f64::NAN
        }),
        "K": num(round_nd(rig.k, 1)),
    });

    // -- the radiated-pressure spectrum and the honest read-outs
    let f_max = (0.45 * fs).min((10.0 * f1_base).max(1500.0));
    let (sf, sm) = pooled_spectrum(&qaccel, fs, f_max, N_SPEC_POINTS);
    let spectrum = json!({
        "kind": "body",
        "f_max": num(round_nd(f_max, 1)),
        "f": finite_list(&sf, Some(3)),
        "mag": finite_list(&sm, None),
        "body_modes": finite_list(&BODY_BODY_FREQS, Some(1)),
        // `null` when there is nothing to measure. The reference shipped NaN here, which its own
        // strict JSON writer refused: at `bridge_stiffness = 0` — the slider's decoupling anchor —
        // the body never moves, the ratio is 0/0, and the Python server died on the request. The
        // front-end already draws a null as "—" (`app.js`), so null is what was meant.
        "omega2_consistency": opt_finite(round_nd(omega2_consistency(&qaccel, &wb, fs, f_max), 3)),
        "terminus_f1": opt_finite(round_nd(terminus_f1(&u_end, fs, c, l), 2)),
        "f1_free": num(round_nd(c / (4.0 * l), 2)),
        "f1_clamped": num(round_nd(c / (2.0 * l), 2)),
        "distance": num(round_nd(distance, 3)),
        "gain": num(gain),
        "gain_times_r": num(round_nd(gain * distance, 6)),
        "latency_ms": num(round_nd(latency as f64 / fs * 1000.0, 3)),
        "retardation_residual": num(round_nd(residual, 4)),
    });

    let flat: Vec<f64> = frames.iter().flatten().copied().collect();
    let n_frames = frames.len();
    let field_amp = if flat.is_empty() { 0.0 } else { max_abs(&flat) };
    let (audio48, peak) = resample_normalize(&pressure, fs);
    let time: Vec<f64> = (0..total.len()).map(|i| i as f64 / fs).collect();
    let times: Vec<f64> = frame_steps.iter().map(|&i| i as f64 / fs).collect();
    Ok(json!({
        "model": "body",
        "fs_sim": num(round_nd(fs, 3)),
        "lambda": num(round_nd(lam, 6)),
        "grid": {"x": finite_list(&x, Some(6))},
        "frames": {
            "b64": b64f32(&flat),
            "n_frames": int(n_frames as i64),
            "width": int(if n_frames > 0 { frames[0].len() as i64 } else { 0 }),
            "dims": int(1),
        },
        "frame_times": finite_list(&times, Some(6)),
        "anim_dt": num(anim_stride as f64 / fs),
        "playback_speed": num(playback_speed),
        "field_amp": num(field_amp),
        "audio": {
            "b64": b64f32(&audio48),
            "fs": num(AUDIO_FS),
            "peak": num(peak),
            "n": int(audio48.len() as i64),
        },
        "energy": energy_block(&time, &total, rig.sigma_body == 0.0, 0.0, EnergyOpts {
            no_decay_oracle: true,
            ..EnergyOpts::default()
        }),
        "meta": {
            "c": num(round_nd(c, 3)),
            "f1": num(round_nd(f1_base, 3)),
            "num_steps": int(n_steps),
            "n_frames": int(n_frames as i64),
            "probe_x": num(round_nd(l, 4)),
            "exchange": exchange,
            "spectrum": spectrum,
        },
    }))
}
