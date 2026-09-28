//! A string on a bridge to a body loaded by the air's IMPEDANCE — `serialize.py`'s
//! `_build_payload_airload` (viewer batch 17).
//!
//! `radbody`'s constant resistance becomes a [`RationalAirLoad`]: a resistance in parallel with the
//! radiation mass `M_a`, the exact acoustic load of a pulsating sphere. Two consequences: the air
//! now STORES as well as radiates (a fifth booked channel), and the damping becomes frequency-
//! dependent, which a constant `R` cannot do.
//!
//! The money test is the LEDGER RESIDUAL, not the drift: `load.energy()` is stored PLUS radiated
//! while the bare body's is neither, so a copy-paste slip can double-count a channel with the drift
//! still green. The second panel is a CONTROLLED single-mode sweep at a pinned 48 kHz: measured
//! `alpha(f)` and pitch drop against the weak-loading closed form. `air_corner = 0` (`M_a = inf`)
//! with weight 1 reproduces `radbody` bit for bit.

use std::f64::consts::PI;

use physsynth_core::body::{self as body, ModalBody};
use physsynth_core::connection::StringBodyBridge;
use physsynth_core::exciter::triangular_pluck;
use physsynth_core::fmt::py_float;
use physsynth_core::radiation::{
    monopole_radiation_resistance, AirParams, AirRadiation, LoadParams, RationalAirLoad,
    ReactiveRadiatedBody,
};
use physsynth_core::string_ideal::{self as ideal, Boundary, IdealString};
use serde_json::{json, Map, Value};

use crate::energy::LOSSLESS_TOL;
use crate::energy::{energy_block, lstsq_slope, EnergyOpts};
use crate::py::{
    b64f32, commas, finite_list, float_to_int, geomspace, int, linspace_idx, max_abs, num, opt_num,
    py_int, py_repr, round_int, round_nd,
};
use crate::radbody::{BODY_BODY_FREQS, BODY_BODY_MASS};
use crate::reed::{C0_AIR, RHO0_AIR};
use crate::string::SPEED_MAX;
use crate::string::{construction, int_at_least_one, FRAMES_PER_PERIOD, MAX_FRAMES, N_MIN};
use crate::{fnum, resample_normalize, Refusal, AUDIO_FS};

/// Grid ceiling: the per-step cost is flat, so the budget is steps alone.
pub const AIRLOAD_N_MAX: i64 = 160;
/// Below 1 required, as for the other bridge scenes.
pub const AIRLOAD_LAM_DEFAULT: f64 = 0.9;
/// The bridge spring, unchanged from `body` and `radbody`.
pub const AIRLOAD_K_DEFAULT: f64 = 8000.0;
/// The coupled guard is load-independent (measured), so `radbody`'s cap carries over.
pub const AIRLOAD_K_MAX: f64 = 19000.0;
/// `rho0 c0 / (4 pi a²)` at `a = 5 cm`: the SATURATED plane-wave value.
pub const AIRLOAD_R_DEFAULT: f64 = 13146.4;
/// Reaches the 5 cm sphere and `radbody`'s 133.
pub const AIRLOAD_R_MAX: f64 = 20000.0;
/// `c0 / (2 pi a)` at `a = 5 cm`; 0 means `M_a = inf`, the flat load.
pub const AIRLOAD_CORNER_DEFAULT: f64 = 1091.8;
/// Corner ceiling (Hz).
pub const AIRLOAD_CORNER_MAX: f64 = 3000.0;
/// The impedance match: best drain AND best slosh AND loudest.
pub const AIRLOAD_WEIGHT_DEFAULT: f64 = 0.05;
/// Weight floor.
pub const AIRLOAD_WEIGHT_MIN: f64 = 0.005;
/// Weight ceiling; 1.0 is `ModalBody`'s default, i.e. `radbody`'s rig.
pub const AIRLOAD_WEIGHT_MAX: f64 = 1.0;
/// Body loss off: the headline is conservation THROUGH radiation.
pub const AIRLOAD_SIGMA_BODY_DEFAULT: f64 = 0.0;
/// Body-loss ceiling.
pub const AIRLOAD_SIGMA_BODY_MAX: f64 = 80.0;
/// Listening radius (m).
pub const AIRLOAD_DISTANCE_DEFAULT: f64 = 1.0;
/// Listening-radius ceiling (m).
pub const AIRLOAD_DISTANCE_MAX: f64 = 8.0;
/// Pluck amplitude (m).
pub const AIRLOAD_AMP_DEFAULT: f64 = 1e-3;
/// Longest audio run, seconds.
pub const AIRLOAD_AUDIO_MAX: f64 = 3.0;
/// Steps of the single instrumented run.
pub const AIRLOAD_WORK_MAX: i64 = 200_000;
/// Seconds of string animation.
pub const AIRLOAD_ANIM_WIN: f64 = 0.06;
/// Seconds of the exchange trace plotted.
pub const AIRLOAD_EXCHANGE_WINDOW: f64 = 0.4;
/// Decimated exchange-trace length.
pub const AIRLOAD_TRACE_POINTS: usize = 600;
/// PINNED: the oracle error's floor is then the formula, not the scheme.
pub const AIRLOAD_SWEEP_FS: f64 = 48_000.0;
/// Points on the sweep.
pub const AIRLOAD_SWEEP_POINTS: i64 = 12;
/// The rig's first body mode ...
pub const AIRLOAD_SWEEP_F_MIN: f64 = 110.0;
/// ... to four octaves above it.
pub const AIRLOAD_SWEEP_F_MAX: f64 = 1760.0;
/// Cycles per sweep point; run length is not the accuracy limiter (measured).
pub const AIRLOAD_SWEEP_CYCLES: f64 = 12.0;
/// A guard only: 12 cycles at 48 kHz binds below ~96 Hz.
pub const AIRLOAD_SWEEP_STEP_CAP: f64 = 6_000.0;
/// Hard stop: censor the remaining points rather than hang.
pub const AIRLOAD_SWEEP_WORK_MAX: i64 = 200_000;
/// `alpha / omega` beyond which the ORACLE overlay is labelled, not hidden.
pub const AIRLOAD_WEAK_LOADING_MAX: f64 = 0.02;

fn pf(x: f64) -> String {
    py_float(x)
}

/// `_airload_mass`: `M_a` from the corner `f_c = R / (2 pi M_a)`; `inf` at `f_c = 0` or `R = 0`.
pub fn airload_mass(r: f64, corner: f64) -> f64 {
    if corner <= 0.0 || r <= 0.0 {
        f64::INFINITY
    } else {
        r / (2.0 * PI * corner)
    }
}

/// `_airload_sphere_readout`: the radius `R` implies and the radius the corner implies.
fn sphere_readout(r: f64, corner: f64) -> Value {
    if r <= 0.0 {
        return json!({"a_from_R": null, "a_from_corner": null, "rel_gap": null,
                      "consistent": false, "corner_sphere": null});
    }
    let a_r = (RHO0_AIR * C0_AIR / (4.0 * PI * r)).sqrt();
    let corner_sphere = (r * C0_AIR / (RHO0_AIR * PI)).sqrt();
    if corner <= 0.0 {
        return json!({
            "a_from_R": num(round_nd(a_r, 5)), "a_from_corner": null, "rel_gap": null,
            "consistent": false, "corner_sphere": num(round_nd(corner_sphere, 2)),
        });
    }
    let a_t = C0_AIR / (2.0 * PI * corner);
    let gap = (a_r - a_t).abs() / a_t;
    json!({
        "a_from_R": num(round_nd(a_r, 5)),
        "a_from_corner": num(round_nd(a_t, 5)),
        "rel_gap": num(round_nd(gap, 6)),
        "consistent": gap < 0.01,
        "corner_sphere": num(round_nd(corner_sphere, 2)),
    })
}

fn load(fs: f64, r: f64, m_a: f64) -> Result<RationalAirLoad, Refusal> {
    Ok(RationalAirLoad::new(
        LoadParams::new(fs, r, m_a, RHO0_AIR, C0_AIR).map_err(construction)?,
    ))
}

/// The rig and what the payload needs to know about it.
pub struct Rig {
    pub bridge: StringBodyBridge<ReactiveRadiatedBody>,
    pub c: f64,
    pub l: f64,
    pub fs: f64,
    pub lam: f64,
    pub k: f64,
    pub r: f64,
    pub corner: f64,
    pub m_a: f64,
    pub weight: f64,
    pub sigma_body: f64,
}

/// `_build_airload_bridge`.
pub fn build(p: &Value) -> Result<Rig, Refusal> {
    let l = fnum(p, "L", 1.0)?;
    let t = fnum(p, "T", 200.0)?;
    let rho = fnum(p, "rho", 0.005)?;
    let lam = fnum(p, "lambda", AIRLOAD_LAM_DEFAULT)?;
    let k = fnum(p, "bridge_stiffness", AIRLOAD_K_DEFAULT)?;
    let r = fnum(p, "radiation_R", AIRLOAD_R_DEFAULT)?;
    let corner = fnum(p, "air_corner", AIRLOAD_CORNER_DEFAULT)?;
    let weight = fnum(p, "radiation_weight", AIRLOAD_WEIGHT_DEFAULT)?;
    let sigma_body = fnum(p, "sigma_body", AIRLOAD_SIGMA_BODY_DEFAULT)?;
    let nv = p.get("N").cloned().unwrap_or(json!(100));
    let n = py_int(&nv)
        .map_err(|_| Refusal::Param(format!("N must be an integer, got {}.", py_repr(&nv))))?;
    if !(N_MIN..=AIRLOAD_N_MAX).contains(&n) {
        return Err(Refusal::Param(format!(
            "N must be in [{N_MIN}, {AIRLOAD_N_MAX}] for the air load, got {n}."
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
    if !(0.0..=AIRLOAD_K_MAX).contains(&k) {
        return Err(Refusal::Param(format!(
            "bridge_stiffness must be in [0, {}], got {}.",
            pf(AIRLOAD_K_MAX),
            pf(k)
        )));
    }
    if !(0.0..=AIRLOAD_R_MAX).contains(&r) {
        return Err(Refusal::Param(format!(
            "radiation_R must be in [0, {}] Pa·s/m³, got {}. This is the SATURATED (plane-wave) \
             resistance rho0*c0/S, not batch 15's compact-source value at one frequency — the \
             same name, a different limit of the same formula.",
            pf(AIRLOAD_R_MAX),
            pf(r)
        )));
    }
    if !(0.0..=AIRLOAD_CORNER_MAX).contains(&corner) {
        return Err(Refusal::Param(format!(
            "air_corner must be in [0, {}] Hz, got {}. 0 means M_a = inf: the purely resistive \
             load, bit-identical to the constant-R body.",
            pf(AIRLOAD_CORNER_MAX),
            pf(corner)
        )));
    }
    if !(AIRLOAD_WEIGHT_MIN..=AIRLOAD_WEIGHT_MAX).contains(&weight) {
        return Err(Refusal::Param(format!(
            "radiation_weight must be in [{}, {}], got {}. It is the body's volume-velocity \
             coupling a_i, not a volume control.",
            pf(AIRLOAD_WEIGHT_MIN),
            pf(AIRLOAD_WEIGHT_MAX),
            pf(weight)
        )));
    }
    if !(0.0..=AIRLOAD_SIGMA_BODY_MAX).contains(&sigma_body) {
        return Err(Refusal::Param(format!(
            "sigma_body must be in [0, {}], got {}.",
            pf(AIRLOAD_SIGMA_BODY_MAX),
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
        Some(vec![weight; m]),
    )
    .map_err(construction)?;
    let m_a = airload_mass(r, corner);
    let loaded =
        ReactiveRadiatedBody::new(ModalBody::new(bp), load(fs, r, m_a)?).map_err(construction)?;
    let bridge = StringBodyBridge::new(string, loaded, k).map_err(construction)?;
    Ok(Rig {
        bridge,
        c,
        l,
        fs,
        lam: lam_s,
        k,
        r,
        corner,
        m_a,
        weight,
        sigma_body,
    })
}

/// `_airload_measure_mode`: one mode, body alone, free decay -> `(f_measured, alpha)`.
pub fn measure_mode(
    f0: f64,
    r: f64,
    m_a: f64,
    weight: f64,
    mass: f64,
    fs: f64,
    steps: usize,
) -> Result<(f64, f64), Refusal> {
    let bp = body::Params::new(
        vec![f0],
        fs,
        vec![0.0],
        vec![mass],
        vec![1.0],
        Some(vec![weight]),
    )
    .map_err(construction)?;
    let mut loaded =
        ReactiveRadiatedBody::new(ModalBody::new(bp), load(fs, r, m_a)?).map_err(construction)?;
    loaded.set_state(&[1e-3], &[0.0]);
    let mut q = Vec::with_capacity(steps);
    for _ in 0..steps {
        loaded.step(0.0);
        q.push(loaded.body().q()[0]);
    }
    if !q.iter().all(|v| v.is_finite()) {
        return Ok((f64::NAN, f64::NAN));
    }
    let crossings: Vec<usize> = (0..q.len().saturating_sub(1))
        .filter(|&i| q[i].is_sign_negative() != q[i + 1].is_sign_negative())
        .collect();
    if crossings.len() < 3 {
        return Ok((f64::NAN, f64::NAN));
    }
    let f_meas = 0.5 * fs * (crossings.len() - 1) as f64
        / (crossings[crossings.len() - 1] - crossings[0]) as f64;
    let env: Vec<f64> = q.iter().map(|v| v.abs()).collect();
    let peaks: Vec<usize> = (1..env.len().saturating_sub(1))
        .filter(|&i| env[i] > env[i - 1] && env[i] >= env[i + 1] && env[i] > 0.0)
        .collect();
    if peaks.len() < 3 {
        return Ok((f_meas, f64::NAN));
    }
    let t: Vec<f64> = peaks.iter().map(|&i| i as f64 / fs).collect();
    let y: Vec<f64> = peaks.iter().map(|&i| env[i].ln()).collect();
    Ok((f_meas, -lstsq_slope(&t, &y)))
}

/// `_airload_sweep`: measured `alpha(f)` and pitch drop against the closed form, per mode.
///
/// `work_max` is [`AIRLOAD_SWEEP_WORK_MAX`] on the shipped path; a test lowers it.
pub fn sweep(p: &Value, rig: &Rig, work_max: i64) -> Result<Value, Refusal> {
    let (r, weight, m_a) = (rig.r, rig.weight, rig.m_a);
    let mass = BODY_BODY_MASS;
    let pv = p
        .get("sweep_points")
        .cloned()
        .unwrap_or(json!(AIRLOAD_SWEEP_POINTS));
    let points = py_int(&pv).map_err(|_| {
        Refusal::Param(format!(
            "sweep_points must be an integer, got {}.",
            py_repr(&pv)
        ))
    })?;
    if !(2..=40).contains(&points) {
        return Err(Refusal::Param(format!(
            "sweep_points must be in [2, 40], got {points}."
        )));
    }
    let cycles = match float_to_int(fnum(p, "sweep_cycles", AIRLOAD_SWEEP_CYCLES)?) {
        Ok(c) => c.max(2),
        Err(crate::py::IntError::NaN) => {
            return Err(Refusal::Construction(
                "cannot convert float NaN to integer".into(),
            ))
        }
        Err(_) => {
            return Err(Refusal::Construction(
                "cannot convert float infinity to integer".into(),
            ))
        }
    };
    let grid = geomspace(AIRLOAD_SWEEP_F_MIN, AIRLOAD_SWEEP_F_MAX, points as usize);
    let fs = AIRLOAD_SWEEP_FS;
    let reference = load(fs, r, m_a)?;
    let r_flat = reference.impedance(2.0 * PI * BODY_BODY_FREQS[0]).0;
    let alpha_flat = weight * weight * r_flat / (2.0 * mass);

    let mut out = Map::new();
    out.insert("f".into(), finite_list(&grid, Some(3)));
    out.insert("fs".into(), num(fs));
    out.insert("cycles".into(), int(cycles));
    out.insert("r_flat".into(), num(round_nd(r_flat, 3)));
    out.insert("alpha_flat".into(), num(round_nd(alpha_flat, 5)));
    out.insert("weak_max".into(), num(AIRLOAD_WEAK_LOADING_MAX));
    if r <= 0.0 {
        for (k, v) in [
            ("skipped", json!(true)),
            ("truncated", json!(false)),
            ("steps", int(0)),
            ("alpha", json!([])),
            ("alpha_oracle", json!([])),
            ("df_meas", json!([])),
            ("df_oracle", json!([])),
            ("alpha_over_omega", json!([])),
            ("n_censored", int(0)),
            ("span", Value::Null),
            ("flat_ratio_top", Value::Null),
            ("resid_coef", Value::Null),
            (
                "note",
                json!("radiation_R = 0: the air is decoupled, nothing radiates, no curve"),
            ),
        ] {
            out.insert(k.into(), v);
        }
        return Ok(Value::Object(out));
    }

    let (mut alpha_m, mut alpha_o, mut df_m, mut df_o, mut a_over_w) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let (mut spent, mut truncated) = (0i64, false);
    for &f0 in &grid {
        if spent >= work_max {
            truncated = true;
            for v in [
                &mut alpha_m,
                &mut alpha_o,
                &mut df_m,
                &mut df_o,
                &mut a_over_w,
            ] {
                v.push(f64::NAN);
            }
            continue;
        }
        let steps = (cycles as f64 * fs / f0).min(AIRLOAD_SWEEP_STEP_CAP) as usize;
        let (f_load, al) = measure_mode(f0, r, m_a, weight, mass, fs, steps)?;
        let (f_free, _) = measure_mode(f0, 0.0, m_a, weight, mass, fs, steps)?;
        spent += 2 * steps as i64;
        alpha_m.push(al);
        // against the SCHEME's own unloaded frequency, never the nominal f0
        df_m.push(if f_free > 0.0 {
            (f_load - f_free) / f_free * 100.0
        } else {
            f64::NAN
        });
        let w0 = 2.0 * PI * f0;
        match reference.loaded_mode(w0, weight, mass, 50, 1e-14) {
            Ok((w_eff, a_pred)) => {
                alpha_o.push(a_pred);
                df_o.push((w_eff / (2.0 * PI) - f0) / f0 * 100.0);
                a_over_w.push(a_pred / (2.0 * PI * f0));
            }
            Err(_) => {
                // outside the formula's range: a fact about the OVERLAY, not the measurement
                alpha_o.push(f64::NAN);
                df_o.push(f64::NAN);
                a_over_w.push(if al.is_finite() {
                    al / (2.0 * PI * f0)
                } else {
                    f64::NAN
                });
            }
        }
    }
    let good: Vec<bool> = (0..alpha_m.len())
        .map(|i| alpha_m[i].is_finite() && alpha_o[i].is_finite() && alpha_o[i] > 0.0)
        .collect();
    let coef = if (0..good.len())
        .filter(|&i| good[i] && a_over_w[i] > 0.0)
        .count()
        >= 2
    {
        let c2: Vec<f64> = (0..good.len())
            .filter(|&i| good[i])
            .map(|i| {
                let rel = (alpha_m[i] - alpha_o[i]).abs() / alpha_o[i];
                rel / (a_over_w[i] * a_over_w[i])
            })
            .collect();
        let (mn, mx) = c2
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &v| {
                if v.is_nan() {
                    (f64::NAN, f64::NAN)
                } else {
                    (a.min(v), b.max(v))
                }
            });
        json!([num(round_nd(mn, 2)), num(round_nd(mx, 2))])
    } else {
        Value::Null
    };
    let finite_a: Vec<f64> = alpha_m
        .iter()
        .copied()
        .filter(|v| v.is_finite() && *v > 0.0)
        .collect();
    let (fa_max, fa_min) = (
        finite_a.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        finite_a.iter().copied().fold(f64::INFINITY, f64::min),
    );
    for (k, v) in [
        ("skipped", json!(false)),
        ("truncated", json!(truncated)),
        ("steps", int(spent)),
        ("alpha", finite_list(&alpha_m, Some(5))),
        ("alpha_oracle", finite_list(&alpha_o, Some(5))),
        ("df_meas", finite_list(&df_m, Some(4))),
        ("df_oracle", finite_list(&df_o, Some(4))),
        ("alpha_over_omega", finite_list(&a_over_w, Some(6))),
        (
            "n_censored",
            int(alpha_m.iter().filter(|v| !v.is_finite()).count() as i64),
        ),
        (
            "n_weak",
            int(a_over_w
                .iter()
                .filter(|v| v.is_finite() && **v <= AIRLOAD_WEAK_LOADING_MAX)
                .count() as i64),
        ),
        (
            "span",
            opt_num((finite_a.len() >= 2).then(|| round_nd(fa_max / fa_min, 2))),
        ),
        (
            "flat_ratio_top",
            opt_num(
                (!finite_a.is_empty() && alpha_flat > 0.0)
                    .then(|| round_nd(fa_max / alpha_flat, 2)),
            ),
        ),
        ("resid_coef", coef),
    ] {
        out.insert(k.into(), v);
    }
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

/// `_build_payload_airload`.
pub fn build_payload(p: &Value) -> Result<Value, Refusal> {
    build_payload_with(p, AIRLOAD_SWEEP_WORK_MAX)
}

/// [`build_payload`] with the sweep's work budget as an argument.
pub fn build_payload_with(p: &Value, sweep_work_max: i64) -> Result<Value, Refusal> {
    let playback_speed = fnum(p, "playback_speed", 0.02)?;
    let pluck_frac = fnum(p, "pluck_position", 0.3)?;
    let amplitude = fnum(p, "amplitude", AIRLOAD_AMP_DEFAULT)?;
    let audio_dur = fnum(p, "audio_duration", 2.0)?;
    let distance = fnum(p, "distance", AIRLOAD_DISTANCE_DEFAULT)?;
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
    if !(0.0 < audio_dur && audio_dur <= AIRLOAD_AUDIO_MAX) {
        return Err(Refusal::Param(format!(
            "audio_duration must be in (0, {}] s, got {}.",
            pf(AIRLOAD_AUDIO_MAX),
            pf(audio_dur)
        )));
    }
    if !(0.0 < distance && distance <= AIRLOAD_DISTANCE_MAX) {
        return Err(Refusal::Param(format!(
            "distance must be in (0, {}] m, got {}.",
            pf(AIRLOAD_DISTANCE_MAX),
            pf(distance)
        )));
    }

    let mut rig = build(p)?;
    let (c, l, fs, lam, r, corner, weight) =
        (rig.c, rig.l, rig.fs, rig.lam, rig.r, rig.corner, rig.weight);
    let f1_base = c / (2.0 * l);
    let n_steps = round_int(audio_dur * fs).max(1);
    if n_steps > AIRLOAD_WORK_MAX {
        return Err(Refusal::Param(format!(
            "work budget exceeded ({} steps > {}). Lower the audio duration, N, or the tension.",
            commas(n_steps),
            commas(AIRLOAD_WORK_MAX)
        )));
    }
    let mut anim_stride = round_int((fs / f1_base) / fpp as f64).max(1);
    let frame_until = anim_stride.max(round_int(AIRLOAD_ANIM_WIN * fs));
    if frame_until / anim_stride > MAX_FRAMES {
        anim_stride = ((frame_until as f64 / MAX_FRAMES as f64).ceil() as i64).max(1);
    }
    let x = rig.bridge.string().params().grid();
    let u0 = triangular_pluck(&x, l, pluck_frac * l, amplitude).map_err(construction)?;
    rig.bridge.string_mut().set_state(&u0, &vec![0.0; u0.len()]);
    let air_p = AirParams::new(fs, distance, RHO0_AIR, C0_AIR, true).map_err(construction)?;
    let (gain, latency) = (air_p.gain, air_p.latency_samples);
    let mut air = AirRadiation::new(air_p);

    // -- one instrumented run: FIVE booked channels and the read-out
    let n = n_steps as usize;
    let mut ch: [Vec<f64>; 7] = Default::default(); // total, string, body, conn, stored, rad, p
    let (mut frames, mut frame_steps): (Vec<Vec<f64>>, Vec<usize>) = (Vec::new(), Vec::new());
    let k_spring = rig.bridge.stiffness();
    let mut sample = |b: &StringBodyBridge<ReactiveRadiatedBody>, ch: &mut [Vec<f64>; 7]| {
        ch[0].push(b.energy());
        ch[1].push(b.string().energy());
        // The BARE body: the wrapper's energy already folds BOTH air channels in.
        ch[2].push(b.body().body().energy());
        ch[3].push(0.5 * k_spring * b.stretch(false) * b.stretch(true));
        ch[4].push(b.body().load().stored_energy());
        ch[5].push(b.body().load().radiated_energy);
        ch[6].push(air.process(b.pressure()));
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
    let [total, e_string, e_body, e_conn, stored, rad, pressure] = ch;
    if !total.iter().all(|v| v.is_finite()) {
        return Err(Refusal::Param(
            "simulation produced non-finite energy (instability) — adjust parameters.".into(),
        ));
    }

    // -- the money panel: FIVE channels, and the residual that guards them
    let e0 = total[0];
    let resid: Vec<f64> = (0..total.len())
        .map(|i| (total[i] - (e_string[i] + e_body[i] + e_conn[i] + stored[i] + rad[i])).abs())
        .collect();
    let ledger = if e0 != 0.0 {
        np_extreme(&resid, true) / e0.abs()
    } else {
        f64::NAN
    };
    let frac = |a: &[f64]| {
        a.iter()
            .zip(&total)
            .map(|(x, t)| x / t)
            .collect::<Vec<f64>>()
    };
    let (es, eb, ec, est, er, ones) = (
        frac(&e_string),
        frac(&e_body),
        frac(&e_conn),
        frac(&stored),
        frac(&rad),
        frac(&total),
    );
    let n_exch = n.min(round_int(AIRLOAD_EXCHANGE_WINDOW * fs).max(1) as usize);
    let eidx = linspace_idx(n_exch, (n_exch + 1).min(AIRLOAD_TRACE_POINTS));
    let pick = |a: &[f64]| eidx.iter().map(|&i| a[i]).collect::<Vec<f64>>();
    let t_ex: Vec<f64> = eidx.iter().map(|&i| i as f64 / fs).collect();
    let last = |a: &[f64]| a[a.len() - 1];
    let exchange = json!({
        "kind": "airload",
        "time": finite_list(&t_ex, Some(6)),
        "e_string_frac": finite_list(&pick(&es), None),
        "e_body_frac": finite_list(&pick(&eb), None),
        "e_conn_frac": finite_list(&pick(&ec), None),
        "e_stored_frac": finite_list(&pick(&est), None),
        "e_rad_frac": finite_list(&pick(&er), None),
        "total_frac": finite_list(&pick(&ones), None),
        "window": num(round_nd(n_exch as f64 / fs, 4)),
        "rad_frac_end": num(round_nd(last(&er), 4)),
        "mech_frac_end": num(round_nd(1.0 - last(&er), 4)),
        "rad_frac_window": num(round_nd(er[eidx[eidx.len() - 1]], 4)),
        "body_frac_peak": num(round_nd(np_extreme(&eb, true), 4)),
        "stored_frac_peak": num(round_nd(np_extreme(&est, true), 5)),
        "stored_frac_window_peak": num(round_nd(np_extreme(&pick(&est), true), 5)),
        "string_frac_min": num(round_nd(np_extreme(&es, false), 4)),
        "conn_frac_min": num(round_nd(np_extreme(&ec, false), 5)),
        "total_drift": num(if e0 != 0.0 {
            (np_extreme(&total, true) - np_extreme(&total, false)) / e0.abs()
        } else {
            f64::NAN
        }),
        "ledger_residual": num(ledger),
        "ledger_tol": num(LOSSLESS_TOL),
        "ledger_pass": ledger < LOSSLESS_TOL,
        "K": num(round_nd(rig.k, 1)),
        "R": num(round_nd(r, 3)),
        "corner": num(round_nd(corner, 3)),
        "weight": num(round_nd(weight, 4)),
    });

    // -- the second panel: the per-mode curve the constant-R load cannot bend
    let sweep = sweep(p, &rig, sweep_work_max)?;
    let reference = load(fs, r, rig.m_a)?;
    let modes_w: Vec<f64> = BODY_BODY_FREQS.iter().map(|&f| 2.0 * PI * f).collect();
    let re_z: Vec<f64> = modes_w.iter().map(|&w| reference.impedance(w).0).collect();
    let im_z: Vec<f64> = modes_w.iter().map(|&w| reference.impedance(w).1).collect();
    let compact: Vec<f64> = modes_w
        .iter()
        .map(|&w| monopole_radiation_resistance(w, RHO0_AIR, C0_AIR))
        .collect();
    let spectrum = json!({
        "kind": "airload",
        "sweep": sweep,
        "R": num(round_nd(r, 3)),
        "corner": num(round_nd(corner, 3)),
        "M_a": if rig.m_a.is_finite() { num(rig.m_a) } else { Value::Null },
        "weight": num(round_nd(weight, 4)),
        "flat": corner <= 0.0,
        "sphere": sphere_readout(r, corner),
        "body_modes": finite_list(&BODY_BODY_FREQS, Some(1)),
        "re_z_modes": finite_list(&re_z, Some(2)),
        "im_z_modes": finite_list(&im_z, Some(2)),
        "r_compact_modes": finite_list(&compact, Some(1)),
        "distance": num(round_nd(distance, 3)),
        "gain": num(gain),
        "latency_ms": num(round_nd(latency as f64 / fs * 1000.0, 3)),
    });

    let flat: Vec<f64> = frames.iter().flatten().copied().collect();
    let n_frames = frames.len();
    let field_amp = if flat.is_empty() { 0.0 } else { max_abs(&flat) };
    let pressure_peak = if pressure.is_empty() {
        0.0
    } else {
        max_abs(&pressure)
    };
    let (audio48, peak) = resample_normalize(&pressure, fs);
    let time: Vec<f64> = (0..total.len()).map(|i| i as f64 / fs).collect();
    let times: Vec<f64> = frame_steps.iter().map(|&i| i as f64 / fs).collect();
    Ok(json!({
        "model": "airload",
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
        // Both air channels are BOOKED, so a radiating run still conserves: sigma_body gates.
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
            "pressure_peak": num(round_nd(pressure_peak, 6)),
            "exchange": exchange,
            "spectrum": spectrum,
        },
    }))
}
