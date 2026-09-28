//! The geometrically-exact string (model #10) — `serialize.py`'s `_build_payload_geometric` and its
//! four regimes.
//!
//! A viz-first model, for a reason that is physics: its longitudinal wave is ~22x faster than the
//! transverse one, and resolving it (`lam_long <= 1`, the model's central trap) forces a sample rate
//! at which a second of sound is minutes of compute. Four regimes, four claims:
//!
//! * **planar** — a bit-exact straight line (`max|w| == 0.0`, the `w -> -w` symmetry), and the
//!   honesty gate every whirl growth ratio rests on;
//! * **rotating** — a true circle, seeded from the converged rotating-wave BVP: an exact solution
//!   of the scheme, whose oracle is its own roundness and a longitudinal field that holds still;
//! * **whirl** — the Mathieu tongue: out-of-plane growth only inside `0 < frac < 1/2`, with the
//!   energy conserved THROUGH the growth;
//! * **phantom** — the bridge force `EA v_x(0)` carries combination tones `f2 - f1`, `2 f1`,
//!   `f1 + f2`, `2 f2` that a scalar tension cannot, discriminated only where the partials are
//!   stretched far enough (a one-sided defect gate).

use std::f64::consts::PI;

use physsynth_analysis::rotating_wave::{
    rotating_wave_history, solve_rotating_wave, BvpParams, CONTINUATION_STEPS_DEFAULT,
    NEWTON_MAXITER_DEFAULT, NEWTON_TOL_DEFAULT,
};
use physsynth_analysis::{damping, dispersion, duffing, modal, spectrum};
use physsynth_core::fmt::{py_float, py_general};
use physsynth_core::pyfloat::scalar_pow;
use physsynth_core::string_geometric::{self as geo, GeometricString};
use serde_json::{json, Map, Value};

use crate::energy::{energy_block, EnergyOpts};
use crate::py::{
    as_bool, b64f32, commas, dot, finite_list, int, linspace_idx, max_abs, np_mean, num, opt_num,
    py_int, py_repr, py_str, repr_str, round_int, round_nd,
};
use crate::string::{construction, int_at_least_one, FRAMES_PER_PERIOD, MAX_FRAMES, SPEED_MAX};
use crate::tension::pool_band;
use crate::{fnum, resample_normalize, Refusal, AUDIO_FS};

/// The four regimes, selected by `domain`.
pub const GEOM_REGIMES: [&str; 4] = ["planar", "rotating", "whirl", "phantom"];
/// A HARD cap here: above 1 the longitudinal wave is under-resolved and fails silently.
pub const GEOM_LAM_LONG_MAX: f64 = 1.0;
/// The shipped `lam_long`.
pub const GEOM_LAM_LONG_DEFAULT: f64 = 0.9;
/// Grid ceiling: every step is a vector Newton solve over three coupled fields.
pub const GEOM_N_MAX: i64 = 32;
/// Steps (the product of N, the window and lam_long is the cost).
pub const GEOM_WORK_MAX: i64 = 6_000;
/// The whirl's `dT/T0`; its planar-breakup coordinate is model #9's.
pub const GEOM_DT_DEFAULT: f64 = 1.5;
/// Past this the DRIVEN polarization stops being single-mode.
pub const GEOM_DT_MAX: f64 = 2.2;
/// The tongue coordinate `frac = delta / (eps A²)`; unstable in (0, 1/2), peak at 1/4.
pub const GEOM_TONGUE_DEFAULT: f64 = 0.25;
/// The slider runs past 1/2 on purpose: watching the growth die is the point.
pub const GEOM_TONGUE_MAX: f64 = 1.0;
/// Out-of-plane seed as a fraction of the driven amplitude.
pub const GEOM_SEED_FRAC: f64 = 1e-3;
/// Planar / rotating amplitude.
pub const GEOM_AMP_DEFAULT: f64 = 4e-3;
/// Revolutions drawn in the rotating regime.
pub const GEOM_ROTATING_PERIODS: f64 = 2.0;
/// Seconds of the planar run.
pub const GEOM_PLANAR_WINDOW: f64 = 0.02;
/// `(u, w)` trail points shipped.
pub const GEOM_ORBIT_POINTS: usize = 1_500;
/// Decimated whirl-envelope length.
pub const GEOM_ENV_POINTS: usize = 400;
/// Growth = max over the last 1/8 against the first 1/8.
pub const GEOM_GROWTH_FRAC: usize = 8;
/// Seconds of bridge force measured in the phantom regime — the test rig's own window.
pub const GEOM_PHANTOM_WINDOW: f64 = 0.10;
/// The phantom regime's amplitude.
pub const GEOM_PHANTOM_AMP_DEFAULT: f64 = 1.5e-3;
/// Hz of defect `f2 - 2 f1` below which the panel labels instead of scoring (one-sided).
pub const GEOM_PHANTOM_DEFECT_MIN: f64 = 3.0;
/// Display band as a multiple of `f1`.
pub const GEOM_PHANTOM_BAND: f64 = 4.8;
/// Half-width (Hz) of the zoom strip around the `f1` / `f2 - f1` pair.
pub const GEOM_PHANTOM_ZOOM_HZ: f64 = 26.0;
/// Zero-pad factor for the DRAWN traces only; the detector keeps the rig's 2x.
pub const GEOM_PHANTOM_DISPLAY_PAD: usize = 32;
/// Steps: the phantom window is fixed physics, so it has its own, larger budget.
pub const GEOM_PHANTOM_WORK_MAX: i64 = 16_500;

fn regime(p: &Value) -> Result<String, Refusal> {
    let r = p
        .get("domain")
        .map_or_else(|| "rotating".to_owned(), py_str);
    if GEOM_REGIMES.contains(&r.as_str()) {
        Ok(r)
    } else {
        Err(Refusal::Param(format!(
            "regime must be one of ('planar', 'rotating', 'whirl', 'phantom'), got {}.",
            repr_str(&r)
        )))
    }
}

/// `_geom_long_kinetic`: kinetic energy of the longitudinal field alone.
fn long_kinetic(res: &GeometricString) -> f64 {
    let n = res.v.len();
    let dt_v: Vec<f64> = (1..n - 1)
        .map(|i| (res.v[i] - res.v_prev[i]) / res.p.k)
        .collect();
    0.5 * res.p.rho * res.p.h * dot(&dt_v, &dt_v)
}

/// `_sliding_max`: a centred sliding-window maximum — the envelope of an oscillation.
pub fn sliding_max(a: &[f64], win: usize) -> Vec<f64> {
    let win = win.max(1).min(a.len());
    if win <= 1 {
        return a.to_vec();
    }
    let env: Vec<f64> = a
        .windows(win)
        .map(|w| {
            let mut m = f64::NEG_INFINITY;
            for &v in w {
                if v.is_nan() {
                    return f64::NAN;
                }
                if v > m {
                    m = v;
                }
            }
            m
        })
        .collect();
    let lead = (win - 1) / 2;
    let tail = a.len() as i64 - env.len() as i64 - lead as i64;
    let mut out = vec![env[0]; lead];
    out.extend_from_slice(&env);
    out.extend(std::iter::repeat_n(
        env[env.len() - 1],
        tail.max(0) as usize,
    ));
    out.truncate(a.len());
    out
}

/// The whirl recipe — `_geom_tongue`.
pub struct Tongue {
    pub amplitude: f64,
    pub kappa_w: f64,
    pub omega: f64,
    pub predicted_rate: f64,
    pub omega0_sq: f64,
}

/// `_geom_tongue`: amplitude, `kappa_w`, the driven frequency and the predicted Mathieu rate.
#[allow(clippy::too_many_arguments)]
pub fn tongue(
    c: f64,
    ea: f64,
    t: f64,
    rho: f64,
    l: f64,
    n: i64,
    dt_over_t0: f64,
    frac: f64,
) -> Result<Tongue, Refusal> {
    let p2 = damping::spatial_eigenvalue_p2(n, l / n as f64, 1);
    let (omega0_sq, eps) =
        duffing::kc_mode_coefficients(c, 0.0, ea - t, rho, p2, l).map_err(Refusal::Construction)?;
    let amplitude = (dt_over_t0 * omega0_sq / eps).sqrt();
    let kappa_w = if frac > 0.0 {
        (frac * eps * scalar_pow(amplitude, 2.0) / scalar_pow(p2, 2.0)).sqrt()
    } else {
        0.0
    };
    let ea2 = eps * scalar_pow(amplitude, 2.0);
    let omega = (omega0_sq + 0.75 * ea2).sqrt();
    let q_m = ea2 / (4.0 * scalar_pow(omega, 2.0));
    let sigma = (frac * ea2 - ea2 / 4.0) / scalar_pow(omega, 2.0);
    let rate = (omega / 2.0)
        * (scalar_pow(q_m, 2.0) - scalar_pow(sigma, 2.0))
            .max(0.0)
            .sqrt();
    Ok(Tongue {
        amplitude,
        kappa_w,
        omega,
        predicted_rate: rate,
        omega0_sq,
    })
}

/// `_build_geometric`: `fs` is driven by `lam_long`, not by `lambda` — the reverse of every other
/// string, on purpose.
fn build(p: &Value, kappa: f64, kappa_w: f64) -> Result<GeometricString, Refusal> {
    let l = fnum(p, "L", 1.0)?;
    let t = fnum(p, "T", 200.0)?;
    let rho = fnum(p, "rho", 0.005)?;
    let ea = fnum(p, "EA", 1.0e5)?;
    let lam_long = fnum(p, "lam_long", GEOM_LAM_LONG_DEFAULT)?;
    let theta = fnum(p, "theta", 0.28)?;
    let n_val = p.get("N").cloned().unwrap_or(json!(16));
    let n = py_int(&n_val)
        .map_err(|_| Refusal::Param(format!("N must be an integer, got {}.", py_repr(&n_val))))?;
    if n > GEOM_N_MAX {
        return Err(Refusal::Param(format!(
            "N must be <= {GEOM_N_MAX} for the geometrically-exact string (got {n}): each step a \
             vector Newton solve over three coupled fields, and fs rides N."
        )));
    }
    if !(0.0 < lam_long && lam_long <= GEOM_LAM_LONG_MAX) {
        return Err(Refusal::Param(format!(
            "lam_long must be in (0, {}], got {}. This is the model's central trap: above 1 the \
             longitudinal wave is under-resolved, and because the theta-scheme is unconditionally \
             stable it fails SILENTLY — no CFL error, just quiet nonsense that stops conserving.",
            py_float(GEOM_LAM_LONG_MAX),
            py_float(lam_long)
        )));
    }
    if ea <= 0.0 {
        return Err(Refusal::Param(format!(
            "EA must be positive, got {}.",
            py_float(ea)
        )));
    }
    let c_long = (ea / rho).sqrt();
    let fs = c_long * n as f64 / (l * lam_long);
    let sigma0 = fnum(p, "sigma0", 0.0)?;
    let sigma1 = fnum(p, "sigma1", 0.0)?;
    let params = geo::Params::new(
        l,
        t,
        rho,
        fs,
        n,
        ea,
        kappa,
        Some(kappa_w),
        sigma0,
        sigma1,
        None,
        None,
        theta,
        true,
        geo::NEWTON_TOL_DEFAULT,
        geo::NEWTON_MAXITER_DEFAULT,
        false,
    )
    .map_err(construction)?;
    Ok(GeometricString::new(params))
}

/// Per-step telemetry.
struct Run {
    e: Vec<f64>,
    u_probe: Vec<f64>,
    w_probe: Vec<f64>,
    u_max: Vec<f64>,
    w_max: Vec<f64>,
    long_kin: f64,
    /// Each frame is `[u, w, v]` stacked.
    frames: Vec<Vec<f64>>,
    frame_steps: Vec<usize>,
    bridge: Vec<f64>,
    q1: Vec<f64>,
    q2: Vec<f64>,
}

/// `_run_geometric`: step once, capturing every panel's telemetry.
fn run(
    res: &mut GeometricString,
    n_steps: usize,
    probe: usize,
    anim_stride: usize,
    modal: Option<(&[f64], &[f64])>,
    track_long_kin: bool,
) -> Result<Run, Refusal> {
    let mut r = Run {
        e: Vec::with_capacity(n_steps + 1),
        u_probe: Vec::with_capacity(n_steps + 1),
        w_probe: Vec::with_capacity(n_steps + 1),
        u_max: Vec::with_capacity(n_steps + 1),
        w_max: Vec::with_capacity(n_steps + 1),
        long_kin: 0.0,
        frames: Vec::new(),
        frame_steps: Vec::new(),
        bridge: Vec::new(),
        q1: Vec::new(),
        q2: Vec::new(),
    };
    let norms = modal.map(|(s1, s2)| (dot(s1, s1), dot(s2, s2)));
    let sample = |res: &GeometricString, r: &mut Run| {
        r.e.push(res.energy());
        r.u_probe.push(res.u[probe]);
        r.w_probe.push(res.w[probe]);
        r.u_max.push(max_abs(&res.u));
        r.w_max.push(max_abs(&res.w));
        if let (Some((s1, s2)), Some((d1, d2))) = (modal, norms) {
            // `_bridge_force`: EA v_x(0) = EA v[1] / h.
            r.bridge.push(res.p.ea * res.v[1] / res.p.h);
            r.q1.push(dot(&res.u, s1) / d1);
            r.q2.push(dot(&res.u, s2) / d2);
        }
    };
    let cap = |res: &GeometricString, r: &mut Run, i: usize| {
        let mut f = res.u.clone();
        f.extend_from_slice(&res.w);
        f.extend_from_slice(&res.v);
        r.frames.push(f);
        r.frame_steps.push(i);
    };
    sample(res, &mut r);
    cap(res, &mut r, 0);
    for i in 1..=n_steps {
        // The binding raised a ValueError here, which the reference's handler reported as a
        // construction error.
        res.step().map_err(construction)?;
        sample(res, &mut r);
        if track_long_kin {
            let lk = long_kinetic(res);
            if lk > r.long_kin {
                r.long_kin = lk;
            }
        }
        if i % anim_stride == 0 {
            cap(res, &mut r, i);
        }
    }
    if !r.e.iter().all(|v| v.is_finite()) {
        return Err(Refusal::Param(
            "simulation produced non-finite energy (instability) — adjust parameters.".into(),
        ));
    }
    Ok(r)
}

/// `np.max` of a slice, NaN-propagating; an empty slice is the reference's `ValueError`.
fn np_max(a: &[f64]) -> Result<f64, Refusal> {
    if a.is_empty() {
        return Err(Refusal::Construction(
            "zero-size array to reduction operation maximum which has no identity".into(),
        ));
    }
    let mut m = f64::NEG_INFINITY;
    for &v in a {
        if v.is_nan() {
            return Ok(f64::NAN);
        }
        if v > m {
            m = v;
        }
    }
    Ok(m)
}

/// `_geom_orbit_block`: the `(u, w)` trail at the probe node.
fn orbit_block(r: &Run, n_frames: usize) -> Value {
    let n = r.u_probe.len();
    let idx = linspace_idx(n - 1, n.min(GEOM_ORBIT_POINTS));
    let pick = |a: &[f64]| idx.iter().map(|&i| a[i]).collect::<Vec<f64>>();
    json!({
        "u": b64f32(&pick(&r.u_probe)),
        "w": b64f32(&pick(&r.w_probe)),
        "n": int(idx.len() as i64),
        "per_frame": num(idx.len() as f64 / n_frames.max(1) as f64),
    })
}

/// The whirl's tongue fields, shipped inside its panel.
struct WhirlInfo {
    frac: f64,
    dt_over_t0: f64,
    tg: Tongue,
    seed_velocity: bool,
}

/// `_geom_whirl_block`.
fn whirl_block(r: &Run, fs: f64, f_osc: f64, w: &WhirlInfo) -> Result<Value, Refusal> {
    let wm = &r.w_max;
    let env = sliding_max(wm, round_int(fs / f_osc.max(1e-9)).max(1) as usize);
    let idx = linspace_idx(env.len() - 1, env.len().min(GEOM_ENV_POINTS));
    let eighth = (wm.len() / GEOM_GROWTH_FRAC).max(1);
    let first = np_max(&wm[..eighth])?;
    let last = np_max(&wm[wm.len() - eighth..])?;
    let growth = if first > 0.0 { last / first } else { 0.0 };
    let q = (wm.len() / 4).max(1);
    let mut quarters = [0.0; 4];
    for (i, qv) in quarters.iter_mut().enumerate() {
        let lo = (i * q).min(wm.len());
        let hi = ((i + 1) * q).min(wm.len());
        *qv = np_max(&wm[lo..hi])?;
    }
    let t_total = (wm.len() - 1) as f64 / fs;
    let measured = if quarters[2] > 0.0 && quarters[3] > 0.0 {
        Some(4.0 * (quarters[3] / quarters[2]).ln() / t_total)
    } else {
        None
    };
    let predicted = w.tg.predicted_rate;
    let u_top = np_max(&r.u_max)?;
    let t: Vec<f64> = idx.iter().map(|&i| i as f64 / fs).collect();
    let e: Vec<f64> = idx.iter().map(|&i| env[i]).collect();
    let mut m = Map::new();
    m.insert("kind".into(), json!("whirl"));
    m.insert("time".into(), finite_list(&t, Some(6)));
    m.insert("envelope".into(), finite_list(&e, None));
    m.insert("growth".into(), num(growth));
    m.insert("w_over_u".into(), num(last / u_top.max(1e-300)));
    m.insert("seeded".into(), Value::Bool(first > 0.0));
    m.insert("measured_rate".into(), opt_num(measured));
    m.insert(
        "rate_ratio".into(),
        match measured {
            Some(mr) if predicted > 0.0 => num(mr / predicted),
            _ => Value::Null,
        },
    );
    m.insert("tongue_position".into(), num(round_nd(w.frac, 4)));
    m.insert("dt_over_t0".into(), num(round_nd(w.dt_over_t0, 4)));
    m.insert("amplitude".into(), num(w.tg.amplitude));
    m.insert("kappa_w".into(), num(round_nd(w.tg.kappa_w, 3)));
    m.insert("predicted_rate".into(), num(predicted));
    m.insert("seed_velocity".into(), Value::Bool(w.seed_velocity));
    m.insert("degenerate".into(), Value::Bool(w.frac == 0.0));
    m.insert(
        "in_tongue".into(),
        Value::Bool(0.0 < w.frac && w.frac < 0.5),
    );
    m.insert("peak_at".into(), num(0.25));
    Ok(Value::Object(m))
}

/// `_geom_phantom_dt_over_t0`: the static tension excess of the two-mode IC.
fn phantom_dt_over_t0(res: &GeometricString, amplitude: f64) -> f64 {
    let n = res.p.n as i64;
    let h = res.p.l / res.p.n as f64;
    let p21 = damping::spatial_eigenvalue_p2(n, h, 1);
    let p22 = damping::spatial_eigenvalue_p2(n, h, 2);
    (res.p.ea - res.p.t) * scalar_pow(amplitude, 2.0) * (p21 + p22) / (4.0 * res.p.t)
}

/// Index of the smallest `|a[i] - target|` — `np.argmin`, first on ties.
fn argmin_dist(a: &[f64], target: f64) -> usize {
    let mut best = 0;
    for i in 1..a.len() {
        if (a[i] - target).abs() < (a[best] - target).abs() {
            best = i;
        }
    }
    best
}

/// `_geom_phantom_block`: the bridge spectrum, the four combination tones, the defect verdict.
fn phantom_block(r: &Run, res: &GeometricString) -> Result<Value, Refusal> {
    let fs = res.p.fs;
    let (bridge, q1, q2) = (&r.bridge[1..], &r.q1[1..], &r.q2[1..]); // drop the IC sample
    let first_peak = |sig: &[f64]| -> Result<f64, Refusal> {
        spectrum::detect_peaks(sig, fs, 1, 10.0, None)
            .first()
            .copied()
            .ok_or_else(|| Refusal::Internal("no partial detected in a modal projection".into()))
    };
    let f1 = first_peak(q1)?;
    let f2 = first_peak(q2)?;
    let combos = [
        ("f2-f1", f2 - f1),
        ("2f1", 2.0 * f1),
        ("f1+f2", f1 + f2),
        ("2f2", 2.0 * f2),
    ];
    let defect = f2 - 2.0 * f1;
    let f_long1 = res.p.c_long / (2.0 * res.p.l);

    let peaks: Vec<f64> = spectrum::detect_peaks(bridge, fs, 40, 10.0, None)
        .into_iter()
        .filter(|&pk| pk < 0.9 * f_long1)
        .collect();
    let s = spectrum::magnitude_spectrum(bridge, fs, 2);
    let mags: Vec<f64> = peaks
        .iter()
        .map(|&pk| s.mag[argmin_dist(&s.freqs, pk)])
        .collect();

    let (mut combo_err, mut dominance) = (None, None);
    if peaks.len() >= 4 {
        // `np.argsort(mags)[::-1][:4]`: the four largest magnitudes.
        let mut order: Vec<usize> = (0..mags.len()).collect();
        order.sort_by(|&a, &b| {
            mags[a]
                .partial_cmp(&mags[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut strongest: Vec<f64> = order.iter().rev().take(4).map(|&i| peaks[i]).collect();
        strongest.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mut cv: Vec<f64> = combos.iter().map(|c| c.1).collect();
        cv.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        combo_err = Some(
            strongest
                .iter()
                .zip(&cv)
                .map(|(a, b)| (a - b).abs())
                .fold(f64::NEG_INFINITY, f64::max),
        );
        let is_combo: Vec<bool> = peaks
            .iter()
            .map(|pk| combos.iter().any(|c| (pk - c.1).abs() < 1.0))
            .collect();
        if is_combo.iter().any(|&b| b) && is_combo.iter().any(|&b| !b) {
            let min_c = (0..peaks.len())
                .filter(|&i| is_combo[i])
                .map(|i| mags[i])
                .fold(f64::INFINITY, f64::min);
            let max_n = (0..peaks.len())
                .filter(|&i| !is_combo[i])
                .map(|i| mags[i])
                .fold(f64::NEG_INFINITY, f64::max);
            dominance = Some(min_c / max_n);
        }
    }
    let nearest = |target: f64| -> Option<f64> {
        (!peaks.is_empty()).then(|| peaks[argmin_dist(&peaks, target)])
    };
    let (near_f1, near_f2) = (nearest(f1), nearest(f2));
    let (peak_diff, peak_2f1) = (nearest(combos[0].1), nearest(combos[1].1));
    let disp = json!([
        opt_num(peak_diff.map(|v| (v - f1).abs())),
        opt_num(peak_2f1.map(|v| (v - f2).abs())),
    ]);

    let bridge_max = if bridge.is_empty() {
        0.0
    } else {
        max_abs(bridge)
    };
    let linear = bridge_max == 0.0;
    let resolved = defect >= GEOM_PHANTOM_DEFECT_MIN && !linear && peaks.len() >= 4;
    let d = spectrum::magnitude_spectrum(bridge, fs, GEOM_PHANTOM_DISPLAY_PAD);
    let wide = pool_band(&d.freqs, &d.mag, 0.0, GEOM_PHANTOM_BAND * f1);
    let z_lo = (f1.min(combos[0].1) - GEOM_PHANTOM_ZOOM_HZ).max(0.0);
    let z_hi = f1.max(combos[0].1) + GEOM_PHANTOM_ZOOM_HZ;
    let zoom = pool_band(&d.freqs, &d.mag, z_lo, z_hi);
    let ladder = dispersion::stiff_dispersion_frequencies(
        res.p.c,
        res.p.l,
        res.p.n as i64,
        res.p.kappa_u,
        res.p.k,
        res.p.theta,
        &[1, 2, 3, 4, 5],
    );
    let band = |v: &Option<(Vec<f64>, Vec<f64>)>, which: usize, nd: usize| match v {
        Some((f, m)) => finite_list(if which == 0 { f } else { m }, Some(nd)),
        None => json!([]),
    };
    let mut combos_m = Map::new();
    for (k, v) in combos {
        combos_m.insert(k.into(), num(v));
    }
    Ok(json!({
        "kind": "phantom",
        "f1": num(f1),
        "f2": num(f2),
        "defect": num(defect),
        "combos": combos_m,
        "ladder": finite_list(&ladder, Some(4)),
        "peaks": finite_list(&peaks, Some(4)),
        "combo_err": opt_num(combo_err),
        "dominance": opt_num(dominance),
        "nearest_to_f1": opt_num(near_f1),
        "nearest_to_f2": opt_num(near_f2),
        "displacements": disp,
        "resolved": resolved,
        "linear": linear,
        "bridge_max": num(bridge_max),
        "n_peaks": int(peaks.len() as i64),
        "defect_min": num(GEOM_PHANTOM_DEFECT_MIN),
        "kappa": num(round_nd(res.p.kappa_u, 3)),
        "f_long1": num(round_nd(f_long1, 1)),
        "band": [num(0.0), num(round_nd(GEOM_PHANTOM_BAND * f1, 3))],
        "zoom": [num(round_nd(z_lo, 3)), num(round_nd(z_hi, 3))],
        "wide_freq": band(&wide, 0, 3),
        "wide_mag": band(&wide, 1, 6),
        "zoom_freq": band(&zoom, 0, 3),
        "zoom_mag": band(&zoom, 1, 6),
    }))
}

/// `_geom_audio_block`: only the phantom regime has audio, and only 0.1 s of bridge force.
fn audio_block(r: &Run, res: &GeometricString, regime: &str) -> (Value, String) {
    if regime != "phantom" {
        return (
            Value::Null,
            format!(
                "viz-only: c_long/c = {:.0}x, so resolving the longitudinal wave forces fs = \
                 {:.0} kHz and one second of sound would be ~10 minutes of compute.",
                (res.p.ea / res.p.t).sqrt(),
                res.p.fs / 1000.0
            ),
        );
    }
    let (audio48, peak) = resample_normalize(&r.bridge[1..], res.p.fs);
    (
        json!({
            "b64": b64f32(&audio48),
            "fs": num(AUDIO_FS),
            "peak": num(peak),
            "n": int(audio48.len() as i64),
        }),
        format!(
            "the bridge force EA·v_x(0) — the piano's radiating channel — {}s of it (peak {} N). \
             A blip, not a note: fs = {:.0} kHz here, so a second would be ~10 minutes of \
             compute. What you hear is mostly the longitudinal startup transient at ~{:.0} Hz \
             (v = 0 is not the longitudinal equilibrium); the phantoms are ~4.5x lower, where the \
             panel looks.",
            py_float(GEOM_PHANTOM_WINDOW),
            py_general(peak, 3),
            res.p.fs / 1000.0,
            res.p.c_long / (2.0 * res.p.l)
        ),
    )
}

/// `sin(pi x / L)` on the grid — `np.sin(np.pi * res.x / res.L)`.
fn sine(x: &[f64], l: f64, factor: f64) -> Vec<f64> {
    x.iter().map(|&xi| (factor * xi / l).sin()).collect()
}

/// `_build_payload_geometric`.
pub fn build_payload(p: &Value) -> Result<Value, Refusal> {
    let regime = regime(p)?;
    let playback_speed = fnum(p, "playback_speed", 0.02)?;
    let probe_frac = fnum(p, "pickup_position", 0.25)?;
    let fpp = int_at_least_one(fnum(p, "frames_per_period", FRAMES_PER_PERIOD)?)?;
    if !(0.0 < playback_speed && playback_speed <= SPEED_MAX) {
        return Err(Refusal::Param(format!(
            "playback_speed must be in (0, {}], got {}.",
            py_float(SPEED_MAX),
            py_float(playback_speed)
        )));
    }
    if !(0.0 < probe_frac && probe_frac < 1.0) {
        return Err(Refusal::Param(format!(
            "pickup_position must be in (0, 1), got {}.",
            py_float(probe_frac)
        )));
    }
    let mut kappa = fnum(p, "kappa", 0.0)?;
    let (mut whirl_in, mut amplitude) = (None, 0.0);
    match regime.as_str() {
        "whirl" => {
            let dt = fnum(p, "dt_over_t0", GEOM_DT_DEFAULT)?;
            let frac = fnum(p, "tongue_position", GEOM_TONGUE_DEFAULT)?;
            if !(0.0 < dt && dt <= GEOM_DT_MAX) {
                return Err(Refusal::Param(format!(
                    "dT/T0 must be in (0, {}], got {dt:.3}. Above that the DRIVEN polarization \
                     stops being single-mode (model #9's own planar parametric breakup), and the \
                     tongue oracle silently stops describing the run.",
                    py_float(GEOM_DT_MAX)
                )));
            }
            if !(0.0..=GEOM_TONGUE_MAX).contains(&frac) {
                return Err(Refusal::Param(format!(
                    "tongue_position must be in [0, {}], got {}.",
                    py_float(GEOM_TONGUE_MAX),
                    py_float(frac)
                )));
            }
            kappa = 0.0;
            whirl_in = Some((dt, frac));
        }
        other => {
            let default = if other == "phantom" {
                GEOM_PHANTOM_AMP_DEFAULT
            } else {
                GEOM_AMP_DEFAULT
            };
            amplitude = fnum(p, "amplitude", default)?;
            if !(0.0 < amplitude && amplitude <= 0.05) {
                return Err(Refusal::Param(format!(
                    "amplitude must be in (0, 0.05] m, got {}.",
                    py_float(amplitude)
                )));
            }
        }
    }

    let zeros = |n: usize| vec![0.0; n];
    let mut whirl_info = None;
    let mut diag = Map::new();
    let mut modal: Option<(Vec<f64>, Vec<f64>)> = None;
    let (mut res, f_osc, n_steps) = match regime.as_str() {
        "whirl" => {
            let (dt, frac) = whirl_in.expect("set above");
            let probe_res = build(p, 0.0, 0.0)?; // cheap: only to read c/L/N back
            let pp = &probe_res.p;
            let tg = tongue(pp.c, pp.ea, pp.t, pp.rho, pp.l, pp.n as i64, dt, frac)?;
            let amp = tg.amplitude;
            let mut res = build(p, 0.0, tg.kappa_w)?;
            let x = res.p.grid();
            let shape = sine(&x, res.p.l, PI);
            let seed = fnum(p, "seed_frac", GEOM_SEED_FRAC)?;
            let seed_velocity = as_bool(p.get("seed_velocity"), false);
            let nn = shape.len();
            let u0: Vec<f64> = shape.iter().map(|s| amp * s).collect();
            if seed_velocity {
                let k = seed * amp * tg.omega0_sq.sqrt();
                let w_dot: Vec<f64> = shape.iter().map(|s| k * s).collect();
                res.set_state(&u0, &zeros(nn), &zeros(nn), &[zeros(nn), w_dot, zeros(nn)]);
            } else {
                let k = seed * amp;
                let w0: Vec<f64> = shape.iter().map(|s| k * s).collect();
                res.set_state(&u0, &w0, &zeros(nn), &[zeros(nn), zeros(nn), zeros(nn)]);
            }
            let f_osc = tg.omega / (2.0 * PI);
            let n_steps = round_int(fnum(p, "animation_window", 0.06)? * res.p.fs).max(1);
            whirl_info = Some(WhirlInfo {
                frac,
                dt_over_t0: dt,
                tg,
                seed_velocity,
            });
            (res, f_osc, n_steps)
        }
        "rotating" => {
            let mut res = build(p, kappa, kappa)?; // a helix needs a DEGENERATE string
            let wave = solve_rotating_wave(&BvpParams {
                l: res.p.l,
                t: res.p.t,
                rho: res.p.rho,
                ea: res.p.ea,
                fs: res.p.fs,
                n_cells: res.p.n,
                theta: res.p.theta,
                amplitude,
                mode: 1,
                kappa,
                time_discrete: true,
                continuation_steps: CONTINUATION_STEPS_DEFAULT,
                tol: NEWTON_TOL_DEFAULT,
                maxiter: NEWTON_MAXITER_DEFAULT,
            })
            .map_err(Refusal::Construction)?;
            // Assign the history DIRECTLY: set_state's y^{-1} is a Taylor start, whose O(k³)
            // error lands in the longitudinal field and ruins the claim by ten orders.
            let (u0, w0, v0, up, wp, vp) =
                rotating_wave_history(&wave, res.p.fs).map_err(Refusal::Construction)?;
            (res.u, res.w, res.v) = (u0, w0, v0);
            (res.u_prev, res.w_prev, res.v_prev) = (up, wp, vp);
            let f_osc = wave.frequency;
            let n_steps = round_int(GEOM_ROTATING_PERIODS * res.p.fs / f_osc).max(1);
            diag.insert("bvp_frequency".into(), num(round_nd(f_osc, 4)));
            diag.insert("bvp_iterations".into(), int(wave.iterations as i64));
            diag.insert("bvp_converged".into(), Value::Bool(wave.converged));
            (res, f_osc, n_steps)
        }
        "phantom" => {
            let mut res = build(p, kappa, kappa)?;
            let x = res.p.grid();
            let sin1 = sine(&x, res.p.l, PI);
            let sin2 = sine(&x, res.p.l, 2.0 * PI);
            let nn = x.len();
            let u0: Vec<f64> = sin1
                .iter()
                .zip(&sin2)
                .map(|(a, b)| amplitude * (a + b))
                .collect();
            res.set_state(
                &u0,
                &zeros(nn),
                &zeros(nn),
                &[zeros(nn), zeros(nn), zeros(nn)],
            );
            let dt = phantom_dt_over_t0(&res, amplitude);
            if dt > GEOM_DT_MAX {
                return Err(Refusal::Param(format!(
                    "dT/T0 = {dt:.2} exceeds {} at A = {amplitude:.4} m: the two-mode motion \
                     stops being two modes (model #9's planar parametric breakup), and the \
                     partials f1, f2 the phantoms are measured against stop being clean single \
                     peaks. Lower the amplitude or EA.",
                    py_float(GEOM_DT_MAX)
                )));
            }
            let f_osc = dispersion::stiff_dispersion_frequencies(
                res.p.c,
                res.p.l,
                res.p.n as i64,
                res.p.kappa_u,
                res.p.k,
                res.p.theta,
                &[1],
            )[0];
            let n_steps = round_int(GEOM_PHANTOM_WINDOW * res.p.fs).max(1);
            diag.insert("dt_over_t0".into(), num(round_nd(dt, 4)));
            modal = Some((sin1, sin2));
            (res, f_osc, n_steps)
        }
        _ => {
            let mut res = build(p, kappa, kappa)?;
            let x = res.p.grid();
            let nn = x.len();
            let u0: Vec<f64> = sine(&x, res.p.l, PI)
                .iter()
                .map(|s| amplitude * s)
                .collect();
            res.set_state(
                &u0,
                &zeros(nn),
                &zeros(nn),
                &[zeros(nn), zeros(nn), zeros(nn)],
            );
            let f_osc = modal::stiff_harmonic_frequencies(res.p.c, res.p.l, kappa, 1)[0];
            let n_steps =
                round_int(fnum(p, "animation_window", GEOM_PLANAR_WINDOW)? * res.p.fs).max(1);
            (res, f_osc, n_steps)
        }
    };

    let work_max = if regime == "phantom" {
        GEOM_PHANTOM_WORK_MAX
    } else {
        GEOM_WORK_MAX
    };
    if n_steps > work_max {
        let tail = if regime == "phantom" {
            "The phantom window is fixed physics, so lower N or lam_long."
        } else {
            "Lower N or the animation window."
        };
        return Err(Refusal::Param(format!(
            "work budget exceeded ({} steps > {}): every step is a vector Newton solve over three \
             coupled fields, and fs is forced ~22x higher than a transverse-only string's by the \
             longitudinal wave (lam_long <= 1). {tail}",
            commas(n_steps),
            commas(work_max)
        )));
    }

    let nn = res.p.n as i64;
    let probe = round_int(probe_frac * nn as f64).max(1).min(nn - 1) as usize;
    let mut anim_stride = round_int((res.p.fs / f_osc) / fpp as f64).max(1);
    if n_steps / anim_stride > MAX_FRAMES {
        anim_stride = ((n_steps as f64 / MAX_FRAMES as f64).ceil() as i64).max(1);
    }
    let r = run(
        &mut res,
        n_steps as usize,
        probe,
        anim_stride as usize,
        modal.as_ref().map(|(a, b)| (a.as_slice(), b.as_slice())),
        regime == "rotating",
    )?;

    let flat: Vec<f64> = r.frames.iter().flatten().copied().collect();
    let n_frames = r.frames.len();
    let field_amp = if flat.is_empty() { 0.0 } else { max_abs(&flat) };
    let e0 = r.e[0];
    let fs = res.p.fs;
    let time: Vec<f64> = (0..r.e.len()).map(|i| i as f64 / fs).collect();
    // `sigma0 == 0 and sigma1 == 0`, short-circuit and all.
    let sigma_zero = fnum(p, "sigma0", 0.0)? == 0.0 && fnum(p, "sigma1", 0.0)? == 0.0;

    let spectrum_block = match regime.as_str() {
        "whirl" => whirl_block(&r, fs, f_osc, whirl_info.as_ref().expect("set above"))?,
        "phantom" => {
            let mut b = phantom_block(&r, &res)?;
            if let Value::Object(m) = &mut b {
                m.extend(diag);
            }
            b
        }
        "rotating" => {
            let rr: Vec<f64> = r
                .u_probe
                .iter()
                .zip(&r.w_probe)
                .map(|(u, w)| u.hypot(*w))
                .collect();
            let mean = np_mean(&rr);
            let (mx, mn) = (
                rr.iter().copied().fold(f64::NEG_INFINITY, f64::max),
                rr.iter().copied().fold(f64::INFINITY, f64::min),
            );
            let mut m = Map::new();
            m.insert("kind".into(), json!("rotating"));
            m.insert(
                "roundness".into(),
                num(if mean > 0.0 { (mx - mn) / mean } else { 0.0 }),
            );
            m.insert(
                "long_kin_over_e".into(),
                num(if e0 > 0.0 { r.long_kin / e0 } else { 0.0 }),
            );
            m.extend(diag);
            Value::Object(m)
        }
        _ => {
            let mw = np_max(&r.w_max)?;
            json!({"kind": "planar", "max_w": num(mw), "exact_zero": mw == 0.0})
        }
    };
    let (audio, audio_note) = audio_block(&r, &res, &regime);
    let times: Vec<f64> = r.frame_steps.iter().map(|&i| i as f64 / fs).collect();
    let oracle = 2.0 * fnum(p, "sigma0", 0.0)?;
    Ok(json!({
        "model": "geometric",
        "regime": regime,
        "fs_sim": num(round_nd(fs, 3)),
        "lambda": num(round_nd(res.p.lam, 6)),
        "lam_long": num(round_nd(res.p.lam_long, 6)),
        "grid": {"x": finite_list(&res.p.grid(), Some(6))},
        "frames": {
            "b64": b64f32(&flat),
            "n_frames": int(n_frames as i64),
            "width": int(if n_frames > 0 { res.p.nodes() as i64 } else { 0 }),
            "fields": ["u", "w", "v"],
            "dims": int(1),
        },
        "frame_times": finite_list(&times, Some(6)),
        "anim_dt": num(anim_stride as f64 / fs),
        "playback_speed": num(playback_speed),
        "field_amp": num(field_amp),
        "orbit": orbit_block(&r, n_frames),
        "audio": audio,
        "audio_note": audio_note,
        "energy": energy_block(&time, &r.e, sigma_zero, oracle, EnergyOpts::default()),
        "meta": {
            "c": num(round_nd(res.p.c, 3)),
            "c_long": num(round_nd((res.p.ea / res.p.rho).sqrt(), 1)),
            "f1": num(round_nd(f_osc, 3)),
            "num_steps": int(n_steps),
            "n_frames": int(n_frames as i64),
            "probe_x": num(round_nd(res.p.grid()[probe], 4)),
            "spectrum": spectrum_block,
        },
    }))
}
