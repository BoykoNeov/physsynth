//! A string on a bridge to a body the AIR pushes back on — `serialize.py`'s
//! `_build_payload_radbody` (viewer batch 15).
//!
//! The lumped body of the `body` scene wrapped in a [`RadiatedBody`]: a constant radiation
//! resistance `R` loads it, and the energy it hands the far field is BOOKED, so the total
//! `E_string + E_body + E_conn + int P_rad` still conserves. The money panel is those four channels;
//! the second is the `t50`-versus-`R` sweep, which shows that radiation has an OPTIMUM — more air
//! is worse past it, because the air chokes the body it drains. That optimum is physics, not the
//! timestep: the curve does not move under a 4x refinement.
//!
//! `R = 0` decouples the air and reproduces the bare body bit-for-bit.

use std::f64::consts::PI;

use physsynth_core::body::{self as body, ModalBody};
use physsynth_core::connection::StringBodyBridge;
use physsynth_core::exciter::triangular_pluck;
use physsynth_core::fmt::py_float;
use physsynth_core::radiation::{
    monopole_radiation_resistance, AirParams, AirRadiation, RadiatedBody,
};
use physsynth_core::string_ideal::{self as ideal, Boundary, IdealString};
use serde_json::{json, Map, Value};

use crate::energy::{energy_block, EnergyOpts};
use crate::py::{
    b64f32, commas, finite_list, geomspace, int, linspace_idx, max_abs, num, opt_num, py_int,
    py_repr, round_int, round_nd,
};
use crate::reed::{C0_AIR, RHO0_AIR};
use crate::string::SPEED_MAX;
use crate::string::{construction, int_at_least_one, FRAMES_PER_PERIOD, MAX_FRAMES, N_MIN};
use crate::{fnum, resample_normalize, Refusal, AUDIO_FS};

/// The body's modes (Hz) — the `body` scene's guitar-ish off-harmonic set.
pub const BODY_BODY_FREQS: [f64; 4] = [110.0, 196.0, 261.0, 440.0];
/// Modal mass: about the string's `rho L`, so the body genuinely loads the string.
pub const BODY_BODY_MASS: f64 = 0.02;

/// Grid ceiling; the per-step cost is N-independent, so the budget is steps alone.
pub const RADBODY_N_MAX: i64 = 160;
/// Below 1 required: the spring pushes the Nyquist mode unstable at 1.
pub const RADBODY_LAM_DEFAULT: f64 = 0.9;
/// The `body` scene's spring, so `R = 0` reproduces it.
pub const RADBODY_K_DEFAULT: f64 = 8000.0;
/// The exact guard trips ~21.5k at N = 100, so the sweep's N always builds.
pub const RADBODY_K_MAX: f64 = 19000.0;
/// The monopole `R_a` at the first body mode (110 Hz).
pub const RADBODY_R_DEFAULT: f64 = 133.0;
/// Reaches the basin (2-10), the physical value (133) and `R = 0`.
pub const RADBODY_R_MAX: f64 = 300.0;
/// Body loss off: the headline is conservation THROUGH radiation.
pub const RADBODY_SIGMA_BODY_DEFAULT: f64 = 0.0;
/// Body-loss ceiling.
pub const RADBODY_SIGMA_BODY_MAX: f64 = 80.0;
/// Listening radius (m) for the far-field read-out.
pub const RADBODY_DISTANCE_DEFAULT: f64 = 1.0;
/// Listening-radius ceiling (m).
pub const RADBODY_DISTANCE_MAX: f64 = 8.0;
/// Pluck amplitude (m).
pub const RADBODY_AMP_DEFAULT: f64 = 1e-3;
/// Longest audio run, seconds.
pub const RADBODY_AUDIO_MAX: f64 = 3.0;
/// Steps of the single instrumented run.
pub const RADBODY_WORK_MAX: i64 = 200_000;
/// Seconds of string animation (the body is lumped).
pub const RADBODY_ANIM_WIN: f64 = 0.06;
/// Seconds of the exchange trace plotted.
pub const RADBODY_EXCHANGE_WINDOW: f64 = 0.4;
/// Decimated length of the exchange trace.
pub const RADBODY_TRACE_POINTS: usize = 600;
/// FIXED sweep grid: the curve is N-independent, and 100 clears the K guard.
pub const RADBODY_SWEEP_N: i64 = 100;
/// Points on the sweep.
pub const RADBODY_SWEEP_POINTS: i64 = 18;
/// A decade below the basin ...
pub const RADBODY_SWEEP_R_MIN: f64 = 0.3;
/// ... to two decades above it (log-spaced).
pub const RADBODY_SWEEP_R_MAX: f64 = 300.0;
/// Seconds per point; beyond this the point is CENSORED, never guessed.
pub const RADBODY_SWEEP_CAP: f64 = 0.8;
/// Hard stop: censor the remaining points rather than hang.
pub const RADBODY_SWEEP_WORK_MAX: i64 = 200_000;

fn pf(x: f64) -> String {
    py_float(x)
}

/// The rig and what the payload needs to know about it.
pub struct Rig {
    pub bridge: StringBodyBridge<RadiatedBody>,
    pub c: f64,
    pub l: f64,
    pub fs: f64,
    pub lam: f64,
    pub k: f64,
    pub r: f64,
    pub sigma_body: f64,
}

/// `_build_radbody_bridge`: the overrides exist for the sweep, which pins N, R and sigma.
pub fn build(
    p: &Value,
    n_override: Option<i64>,
    r_override: Option<f64>,
    sigma_override: Option<f64>,
) -> Result<Rig, Refusal> {
    let l = fnum(p, "L", 1.0)?;
    let t = fnum(p, "T", 200.0)?;
    let rho = fnum(p, "rho", 0.005)?;
    let lam = fnum(p, "lambda", RADBODY_LAM_DEFAULT)?;
    let k = fnum(p, "bridge_stiffness", RADBODY_K_DEFAULT)?;
    let r = match r_override {
        Some(v) => v,
        None => fnum(p, "radiation_R", RADBODY_R_DEFAULT)?,
    };
    let sigma_body = match sigma_override {
        Some(v) => v,
        None => fnum(p, "sigma_body", RADBODY_SIGMA_BODY_DEFAULT)?,
    };
    let n = match n_override {
        Some(n) => n,
        None => {
            let v = p.get("N").cloned().unwrap_or(json!(100));
            py_int(&v).map_err(|_| {
                Refusal::Param(format!("N must be an integer, got {}.", py_repr(&v)))
            })?
        }
    };
    if !(N_MIN..=RADBODY_N_MAX).contains(&n) {
        return Err(Refusal::Param(format!(
            "N must be in [{N_MIN}, {RADBODY_N_MAX}] for the radiating body, got {n}."
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
    if !(0.0..=RADBODY_K_MAX).contains(&k) {
        return Err(Refusal::Param(format!(
            "bridge_stiffness must be in [0, {}], got {}.",
            pf(RADBODY_K_MAX),
            pf(k)
        )));
    }
    if !(0.0..=RADBODY_R_MAX).contains(&r) {
        return Err(Refusal::Param(format!(
            "radiation_R must be in [0, {}] Pa·s/m³, got {}. R = 0 decouples the air \
             (bit-identical to the read-out-only body); the load is unconditionally passive, so \
             this is a legibility range, not a stability one.",
            pf(RADBODY_R_MAX),
            pf(r)
        )));
    }
    if !(0.0..=RADBODY_SIGMA_BODY_MAX).contains(&sigma_body) {
        return Err(Refusal::Param(format!(
            "sigma_body must be in [0, {}], got {}.",
            pf(RADBODY_SIGMA_BODY_MAX),
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
    let loaded = RadiatedBody::new(ModalBody::new(bp), r).map_err(construction)?;
    // The exact coupled stability guard fires HERE if K exceeds the ceiling.
    let bridge = StringBodyBridge::new(string, loaded, k).map_err(construction)?;
    Ok(Rig {
        bridge,
        c,
        l,
        fs,
        lam: lam_s,
        k,
        r,
        sigma_body,
    })
}

/// Pluck the rig's string from rest.
fn pluck(rig: &mut Rig, frac: f64, amplitude: f64) -> Result<(), Refusal> {
    let x = rig.bridge.string().params().grid();
    let u0 = triangular_pluck(&x, rig.l, frac * rig.l, amplitude).map_err(construction)?;
    let v0 = vec![0.0; u0.len()];
    rig.bridge.string_mut().set_state(&u0, &v0);
    Ok(())
}

/// `_radbody_t50`: steps until the radiated channel holds half the pluck's energy; NaN if censored.
pub fn t50(p: &Value, r: f64, cap_seconds: f64, n: i64) -> Result<(f64, i64), Refusal> {
    let mut rig = build(p, Some(n), Some(r), Some(0.0))?;
    let pluck_frac = fnum(p, "pluck_position", 0.3)?;
    let amplitude = fnum(p, "amplitude", RADBODY_AMP_DEFAULT)?;
    pluck(&mut rig, pluck_frac, amplitude)?;
    let half = 0.5 * rig.bridge.energy();
    let cap_steps = round_int(cap_seconds * rig.fs).max(1);
    for i in 1..=cap_steps {
        let _ = rig.bridge.step(&mut ());
        if rig.bridge.body().radiated_energy >= half {
            return Ok((i as f64 / rig.fs, i));
        }
    }
    Ok((f64::NAN, cap_steps))
}

/// `_radbody_sweep`: the `t50`-vs-R reference curve and the operating point measured the same way.
///
/// `work_max` is [`RADBODY_SWEEP_WORK_MAX`] on the shipped path; a test lowers it to reach the
/// censoring, which the shipped range cannot.
pub fn sweep(p: &Value, r_now: f64, k_now: f64, work_max: i64) -> Result<Value, Refusal> {
    let pv = p
        .get("sweep_points")
        .cloned()
        .unwrap_or(json!(RADBODY_SWEEP_POINTS));
    let points = py_int(&pv).map_err(|_| {
        Refusal::Param(format!(
            "sweep_points must be an integer, got {}.",
            py_repr(&pv)
        ))
    })?;
    let cap = fnum(p, "sweep_cap", RADBODY_SWEEP_CAP)?;
    if !(2..=40).contains(&points) {
        return Err(Refusal::Param(format!(
            "sweep_points must be in [2, 40], got {points}."
        )));
    }
    if !(0.05..=1.5).contains(&cap) {
        return Err(Refusal::Param(format!(
            "sweep_cap must be in [0.05, 1.5] s, got {}.",
            pf(cap)
        )));
    }
    let grid = geomspace(RADBODY_SWEEP_R_MIN, RADBODY_SWEEP_R_MAX, points as usize);
    let mut out = Map::new();
    out.insert("r".into(), finite_list(&grid, Some(4)));
    out.insert("cap_ms".into(), num(round_nd(cap * 1000.0, 1)));
    out.insert("sweep_n".into(), int(RADBODY_SWEEP_N));
    out.insert("r_now".into(), num(round_nd(r_now, 3)));
    if k_now <= 0.0 {
        for (k, v) in [
            ("t50_ms", json!([])),
            ("skipped", json!(true)),
            ("truncated", json!(false)),
            ("steps", int(0)),
            ("t50_now_ms", Value::Null),
            ("best_r", Value::Null),
            ("best_t50_ms", Value::Null),
            ("basin", Value::Null),
            ("n_censored", int(0)),
            (
                "note",
                json!("no bridge coupling (K = 0): the body never moves, nothing radiates"),
            ),
        ] {
            out.insert(k.into(), v);
        }
        return Ok(Value::Object(out));
    }

    let (mut t50s, mut spent, mut truncated) = (Vec::with_capacity(grid.len()), 0i64, false);
    for &r in &grid {
        if spent >= work_max {
            truncated = true;
            t50s.push(f64::NAN);
            continue;
        }
        let (t, steps) = t50(p, r, cap, RADBODY_SWEEP_N)?;
        spent += steps;
        t50s.push(t);
    }
    let finite: Vec<bool> = t50s.iter().map(|v| v.is_finite()).collect();
    // `np.argmin(np.where(finite, t50, inf))`, first on ties.
    let best = (0..t50s.len())
        .filter(|&i| finite[i])
        .fold(None, |b: Option<usize>, i| match b {
            Some(j) if t50s[j] <= t50s[i] => Some(j),
            _ => Some(i),
        });
    let basin = best.map(|b| {
        let near: Vec<usize> = (0..t50s.len())
            .filter(|&i| finite[i] && t50s[i] <= 1.25 * t50s[b])
            .collect();
        json!([
            num(round_nd(grid[near[0]], 3)),
            num(round_nd(grid[near[near.len() - 1]], 3)),
        ])
    });
    let mut t50_now = f64::NAN;
    if r_now > 0.0 && spent < work_max {
        let (t, steps) = t50(p, r_now, cap, RADBODY_SWEEP_N)?;
        t50_now = t;
        spent += steps;
    }
    let ms: Vec<f64> = t50s.iter().map(|v| v * 1000.0).collect();
    out.insert("t50_ms".into(), finite_list(&ms, Some(3)));
    out.insert("skipped".into(), json!(false));
    out.insert("truncated".into(), json!(truncated));
    out.insert("steps".into(), int(spent));
    out.insert(
        "n_censored".into(),
        int(finite.iter().filter(|&&f| !f).count() as i64),
    );
    out.insert("best_r".into(), opt_num(best.map(|b| round_nd(grid[b], 3))));
    out.insert(
        "best_t50_ms".into(),
        opt_num(best.map(|b| round_nd(t50s[b] * 1000.0, 3))),
    );
    out.insert("basin".into(), basin.unwrap_or(Value::Null));
    out.insert(
        "t50_now_ms".into(),
        if t50_now.is_finite() {
            num(round_nd(t50_now * 1000.0, 3))
        } else {
            Value::Null
        },
    );
    Ok(Value::Object(out))
}

/// `np.max` / `np.min` of a non-empty slice, NaN-propagating.
fn np_extreme(a: &[f64], max: bool) -> f64 {
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

/// `_build_payload_radbody`.
pub fn build_payload(p: &Value) -> Result<Value, Refusal> {
    build_payload_with(p, RADBODY_SWEEP_WORK_MAX)
}

/// [`build_payload`] with the sweep's work budget as an argument.
pub fn build_payload_with(p: &Value, sweep_work_max: i64) -> Result<Value, Refusal> {
    let playback_speed = fnum(p, "playback_speed", 0.02)?;
    let pluck_frac = fnum(p, "pluck_position", 0.3)?;
    let amplitude = fnum(p, "amplitude", RADBODY_AMP_DEFAULT)?;
    let audio_dur = fnum(p, "audio_duration", 2.0)?;
    let distance = fnum(p, "distance", RADBODY_DISTANCE_DEFAULT)?;
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
    if !(0.0 < audio_dur && audio_dur <= RADBODY_AUDIO_MAX) {
        return Err(Refusal::Param(format!(
            "audio_duration must be in (0, {}] s, got {}.",
            pf(RADBODY_AUDIO_MAX),
            pf(audio_dur)
        )));
    }
    if !(0.0 < distance && distance <= RADBODY_DISTANCE_MAX) {
        return Err(Refusal::Param(format!(
            "distance must be in (0, {}] m, got {}.",
            pf(RADBODY_DISTANCE_MAX),
            pf(distance)
        )));
    }

    let mut rig = build(p, None, None, None)?;
    let (c, l, fs, lam, r) = (rig.c, rig.l, rig.fs, rig.lam, rig.r);
    let f1_base = c / (2.0 * l);
    let n_steps = round_int(audio_dur * fs).max(1);
    if n_steps > RADBODY_WORK_MAX {
        return Err(Refusal::Param(format!(
            "work budget exceeded ({} steps > {}). Lower the audio duration, N, or the tension.",
            commas(n_steps),
            commas(RADBODY_WORK_MAX)
        )));
    }
    let mut anim_stride = round_int((fs / f1_base) / fpp as f64).max(1);
    let frame_until = anim_stride.max(round_int(RADBODY_ANIM_WIN * fs));
    if frame_until / anim_stride > MAX_FRAMES {
        anim_stride = ((frame_until as f64 / MAX_FRAMES as f64).ceil() as i64).max(1);
    }
    pluck(&mut rig, pluck_frac, amplitude)?;
    let air_p = AirParams::new(fs, distance, RHO0_AIR, C0_AIR, true).map_err(construction)?;
    let (gain, latency) = (air_p.gain, air_p.latency_samples);
    let mut air = AirRadiation::new(air_p);

    // -- one instrumented run: the four booked channels and the far-field read-out
    let n = n_steps as usize;
    let (mut total, mut e_string, mut e_body, mut e_conn, mut rad, mut pressure) = (
        Vec::with_capacity(n + 1),
        Vec::with_capacity(n + 1),
        Vec::with_capacity(n + 1),
        Vec::with_capacity(n + 1),
        Vec::with_capacity(n + 1),
        Vec::with_capacity(n + 1),
    );
    let (mut frames, mut frame_steps): (Vec<Vec<f64>>, Vec<usize>) = (Vec::new(), Vec::new());
    let k_spring = rig.bridge.stiffness();
    let mut sample = |b: &StringBodyBridge<RadiatedBody>| {
        total.push(b.energy());
        e_string.push(b.string().energy());
        // The BARE body: the loaded body's own energy already folds the radiated channel in.
        e_body.push(b.body().body().energy());
        e_conn.push(0.5 * k_spring * b.stretch(false) * b.stretch(true));
        rad.push(b.body().radiated_energy);
        pressure.push(air.process(b.pressure()));
    };
    sample(&rig.bridge);
    if frame_until >= 1 {
        frames.push(rig.bridge.string().u.clone());
        frame_steps.push(0);
    }
    for i in 1..=n {
        let _ = rig.bridge.step(&mut ());
        sample(&rig.bridge);
        if i as i64 <= frame_until && i as i64 % anim_stride == 0 {
            frames.push(rig.bridge.string().u.clone());
            frame_steps.push(i);
        }
    }
    if !total.iter().all(|v| v.is_finite()) {
        return Err(Refusal::Param(
            "simulation produced non-finite energy (instability) — adjust parameters.".into(),
        ));
    }

    // -- the money panel: the FOUR booked channels, as fractions of the booked total
    let frac = |a: &[f64]| {
        a.iter()
            .zip(&total)
            .map(|(x, t)| x / t)
            .collect::<Vec<f64>>()
    };
    let (es, eb, ec, er) = (frac(&e_string), frac(&e_body), frac(&e_conn), frac(&rad));
    let ones = frac(&total);
    let n_exch = n.min(round_int(RADBODY_EXCHANGE_WINDOW * fs).max(1) as usize);
    let eidx = linspace_idx(n_exch, (n_exch + 1).min(RADBODY_TRACE_POINTS));
    let e0 = total[0];
    let pick = |a: &[f64]| eidx.iter().map(|&i| a[i]).collect::<Vec<f64>>();
    let t_ex: Vec<f64> = eidx.iter().map(|&i| i as f64 / fs).collect();
    let last = |a: &[f64]| a[a.len() - 1];
    let exchange = json!({
        "kind": "radbody",
        "time": finite_list(&t_ex, Some(6)),
        "e_string_frac": finite_list(&pick(&es), None),
        "e_body_frac": finite_list(&pick(&eb), None),
        "e_conn_frac": finite_list(&pick(&ec), None),
        "e_rad_frac": finite_list(&pick(&er), None),
        "total_frac": finite_list(&pick(&ones), None),
        "window": num(round_nd(n_exch as f64 / fs, 4)),
        "rad_frac_end": num(round_nd(last(&er), 4)),
        "mech_frac_end": num(round_nd(1.0 - last(&er), 4)),
        "rad_frac_window": num(round_nd(er[eidx[eidx.len() - 1]], 4)),
        "body_frac_peak": num(round_nd(np_extreme(&eb, true), 4)),
        "string_frac_min": num(round_nd(np_extreme(&es, false), 4)),
        "string_frac_max": num(round_nd(np_extreme(&es, true), 4)),
        "total_drift": num(if e0 != 0.0 {
            (np_extreme(&total, true) - np_extreme(&total, false)) / e0.abs()
        } else {
            f64::NAN
        }),
        "K": num(round_nd(rig.k, 1)),
        "R": num(round_nd(r, 3)),
    });

    // -- the second panel: the t50-vs-R reference curve
    let sweep = sweep(p, r, rig.k, sweep_work_max)?;
    let bp = rig.bridge.body().body().params();
    let (a1, m1) = (bp.a[0], bp.m[0]);
    let w1 = 2.0 * PI * BODY_BODY_FREQS[0];
    let mrr = |omega: f64| monopole_radiation_resistance(omega, RHO0_AIR, C0_AIR);
    let r_a_unit = mrr(1.0);
    let f_match = if r > 0.0 {
        (r / r_a_unit).sqrt() / (2.0 * PI)
    } else {
        0.0
    };
    let r_phys: Vec<f64> = BODY_BODY_FREQS.iter().map(|&f| mrr(2.0 * PI * f)).collect();
    let spectrum = json!({
        "kind": "radbody",
        "sweep": sweep,
        "R": num(round_nd(r, 3)),
        "sigma_rad": num(round_nd(a1 * a1 * r / (2.0 * m1), 4)),
        "sigma_ratio": num(round_nd(a1 * a1 * r / (2.0 * m1) / w1, 5)),
        "f_match": num(round_nd(f_match, 2)),
        "body_modes": finite_list(&BODY_BODY_FREQS, Some(1)),
        "r_phys": finite_list(&r_phys, Some(1)),
        "r_phys_f1": num(round_nd(mrr(2.0 * PI * BODY_BODY_FREQS[0]), 1)),
        "distance": num(round_nd(distance, 3)),
        "gain": num(gain),
        "gain_times_r": num(round_nd(gain * distance, 6)),
        "latency_ms": num(round_nd(latency as f64 / fs * 1000.0, 3)),
    });

    let flat: Vec<f64> = frames.iter().flatten().copied().collect();
    let n_frames = frames.len();
    let field_amp = if flat.is_empty() { 0.0 } else { max_abs(&flat) };
    let (audio48, peak) = resample_normalize(&pressure, fs);
    let time: Vec<f64> = (0..total.len()).map(|i| i as f64 / fs).collect();
    let times: Vec<f64> = frame_steps.iter().map(|&i| i as f64 / fs).collect();
    Ok(json!({
        "model": "radbody",
        "fs_sim": num(round_nd(fs, 3)),
        "lambda": num(round_nd(lam, 6)),
        "grid": {"x": finite_list(&rig.bridge.string().params().grid(), Some(6))},
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
        // R > 0 is a booked CHANNEL, not a loss: the gate stays sigma_body alone.
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
