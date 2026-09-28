//! The tension-modulated string (model #9) — both of its regimes, `serialize.py`'s
//! `_build_payload_tension` and `_build_payload_parametric`.
//!
//! **duffing** (the default) is the Kirchhoff–Carrier hardening demo. Its headline is the
//! *amplitude shift* `omega(A) - omega(A -> 0)`, not an absolute frequency: both measurements carry
//! the theta-scheme's linear dispersion error and their difference cancels it. The shift is
//! measured on two dedicated **lossless, single-mode** runs — never the audio run, which is lossy on
//! purpose and so chirps downward — and scored against the exact Duffing closed form. Two honesty
//! gates: *purity* (above a threshold the single mode parametrically disintegrates, the Duffing
//! reduction stops describing the motion, and the shift reads `null`) and *convergence* of the
//! per-step tension root-find, folded into the energy verdict.
//!
//! **parametric** is that disintegration given a panel of its own: the tension pumps at
//! `2 omega_m`, neighbouring modes sit in Mathieu tongues, and above a threshold they grow until the
//! mode breaks up. The verdict is the ordinary lossless drift, and what makes it a claim is what it
//! survives — the drift stays at ~1e-13 *through* the breakup. `sigma` is forced to zero here, on
//! purpose: loss would decay the amplitude back through the threshold mid-panel.
//!
//! The run lengths the reference's tests lower with `monkeypatch` (`TENSION_MEASURE_PERIODS`,
//! `PARAM_SWEEP_WORK_MAX`) are arguments of the public helpers here, so a test lowers them by
//! passing a smaller number — the shipped path passes the constant.

use std::f64::consts::PI;

use physsynth_analysis::{damping, duffing, modal, spectrum};
use physsynth_core::engine::simulate;
use physsynth_core::pyfloat::scalar_pow;
use physsynth_core::string_nonlinear::TensionModulatedString;
use serde_json::{json, Map, Value};

use crate::energy::{energy_block, EnergyOpts};
use crate::py::{
    commas, dot, finite_list, int, linspace_idx, max_abs, norm, np_mean, num, opt_num, py_float,
    py_int, py_repr, py_str, repr_str, round_int, round_nd,
};
use crate::string::{build_resonator, int_at_least_one, Frames, ANIM_WIN_MAX, MAX_FRAMES};
use crate::string::{FRAMES_PER_PERIOD, SPEED_MAX};
use crate::{fnum, resample_normalize, Refusal, AUDIO_FS};

/// `EA / T = 500` at the default tension; real strings sit at 150–600.
pub const TENSION_EA_DEFAULT: f64 = 1.0e5;
/// Axial-stiffness ceiling.
pub const TENSION_EA_MAX: f64 = 2.0e5;
/// Amplitude IS this model's independent variable: 0.02 gives a plainly visible, sub-breakup shift.
pub const TENSION_AMP_DEFAULT: f64 = 0.02;
/// Slider bound only; the load-bearing guard is [`TENSION_DT_MAX`].
pub const TENSION_AMP_MAX: f64 = 0.06;
/// Fixed absolute reference amplitude (a ratio of `A` would court zero-crossing noise).
pub const TENSION_REF_AMP: f64 = 1e-5;
/// Fundamental periods per measurement run; the residual at 12 is the scheme's own O(h²)+O(k²).
pub const TENSION_MEASURE_PERIODS: f64 = 12.0;
/// Purity gate: sub-threshold runs sit at ~1e-11 off-mode, broken-up ones above 1e-3.
pub const TENSION_OFFMODE_MAX: f64 = 1e-6;
/// THE guard: mode 1 breaks up for `dT/T0` in (4.44, 6.05]; bounding `dT/T0`, not amplitude.
pub const TENSION_DT_MAX: f64 = 4.45;
/// Grid ceiling: every step is a banded re-solve inside a root-find.
pub const TENSION_N_MAX: i64 = 256;
/// Longest audio run, seconds.
pub const TENSION_AUDIO_MAX: f64 = 3.0;
/// Total steps: audio + both measurement runs + animation.
pub const TENSION_WORK_MAX: i64 = 60_000;

/// The two regimes of the `tension` key, selected by `domain`.
pub const PARAM_REGIMES: [&str; 2] = ["duffing", "parametric"];
/// The documented driven mode (mode 1 has no lower partner and does not break up).
pub const PARAM_M_DEFAULT: f64 = 3.0;
/// Highest drivable mode.
pub const PARAM_M_MAX: i64 = 8;
/// Seed size relative to the driven amplitude; the growth rate is seed-independent.
pub const PARAM_SEED_REL: f64 = 1e-6;
/// Modes the seed spreads over (the driven one removed), N-independent by construction.
pub const PARAM_SEED_MODES: usize = 24;
/// `np.random.default_rng(12345).standard_normal(25)` under NumPy 2.4.6 — the reference's seed
/// coefficients, frozen here rather than regenerated: the point of a fixed seed is that the
/// perturbation is the same everywhere, and a Rust PCG64 + ziggurat would be a second copy of
/// NumPy's internals to keep in step. Index 0 is drawn and never used, as in the original.
pub const PARAM_SEED_COEF: [f64; 25] = [
    -1.4238250364546312,
    1.2637284581291104,
    -0.8706617379590857,
    -0.2591732349343976,
    -0.07534330701052097,
    -0.740884652085609,
    -1.3677927017829434,
    0.6488928021930399,
    0.361058113054895,
    -1.95286306301219,
    2.347409654378852,
    0.9684969057519236,
    -0.7593871804245066,
    0.9021982742122517,
    -0.46695317332055025,
    -0.06068951873702798,
    0.7888443445192008,
    -1.2566681331396765,
    0.5758575143959287,
    1.3989789947237192,
    1.3222980607327857,
    -0.29969851529910546,
    0.9029193414250598,
    -1.6215827341822058,
    -0.15818926067687128,
];
/// Above-threshold default (the m = 3 edge is ~2).
pub const PARAM_DT_ABOVE_DEFAULT: f64 = 3.0;
/// Below-threshold default — still 42 % nonlinear, which is the point.
pub const PARAM_DT_BELOW_DEFAULT: f64 = 1.5;
/// Ceiling on either `dT/T0`.
pub const PARAM_DT_MAX: f64 = 14.0;
/// Claim-run length, in periods of mode m (never steps).
pub const PARAM_CLAIM_PERIODS: f64 = 40.0;
/// Off-mode content is sampled every this many steps.
pub const PARAM_SAMPLE: usize = 4;
/// The sweep's fixed grid.
pub const PARAM_SWEEP_N: i64 = 100;
/// The sweep's `dT/T0` points.
pub const PARAM_SWEEP_DTS: [f64; 8] = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0];
/// Sweep cap, periods of mode m.
pub const PARAM_SWEEP_CAP_PERIODS: f64 = 60.0;
/// Saturation is judged chunk to chunk, in periods of mode m.
pub const PARAM_SWEEP_CHUNK_PERIODS: i64 = 15;
/// A chunk beating the previous by < 10 % counts as saturated.
pub const PARAM_SAT_RATIO: f64 = 1.10;
/// Level > 100x the seed floor = the mode was destabilized.
pub const PARAM_UNSTABLE_FACTOR: f64 = 100.0;
/// Floor under a NAMED cascade partner: one printable tick of the readout (0.1 %).
pub const PARAM_PARTNER_FLOOR: f64 = 1e-3;
/// Grid ceiling for this regime.
pub const PARAM_N_MAX: i64 = 200;
/// Steps in the two claim runs.
pub const PARAM_WORK_MAX: i64 = 30_000;
/// Steps in the sweep, which is the cost driver.
pub const PARAM_SWEEP_WORK_MAX: i64 = 60_000;

fn pf(x: f64) -> String {
    physsynth_core::fmt::py_float(x)
}

/// `_tension_regime`.
pub fn regime(p: &Value) -> Result<String, Refusal> {
    let r = p.get("domain").map_or_else(|| "duffing".to_owned(), py_str);
    if PARAM_REGIMES.contains(&r.as_str()) {
        Ok(r)
    } else {
        Err(Refusal::Param(format!(
            "regime must be one of ('duffing', 'parametric'), got {}.",
            repr_str(&r)
        )))
    }
}

/// The `tension` key: either regime.
pub fn build_payload(p: &Value) -> Result<Value, Refusal> {
    if regime(p)? == "parametric" {
        build_payload_parametric(p)
    } else {
        build_payload_duffing(p)
    }
}

/// `sin(m pi x / L)` on the grid — `m * np.pi * np.arange(N + 1) / N`, in that order.
pub fn mode_shape(n: usize, m: usize) -> Vec<f64> {
    let mpi = m as f64 * PI;
    (0..=n)
        .map(|i| mpi * i as f64 / n as f64)
        .map(f64::sin)
        .collect()
}

/// `_mode1_shape`: `np.pi * np.arange(N + 1) / N` — one factor fewer than [`mode_shape`], so it
/// is spelled separately rather than as `mode_shape(n, 1)` (which multiplies by `1.0` first; the
/// same value, but this is the reference's expression).
fn mode1_shape(n: usize) -> Vec<f64> {
    (0..=n)
        .map(|i| PI * i as f64 / n as f64)
        .map(f64::sin)
        .collect()
}

/// Peak tension excess `dT/T0 = EA A² p²/(4T)` of a single mode — exact, no stepping.
pub fn dt_over_t(ea: f64, amplitude: f64, p2: f64, t: f64) -> f64 {
    ea * scalar_pow(amplitude, 2.0) * p2 / (4.0 * t)
}

/// Steps in one measurement run: `periods` linear fundamental periods, at least 64.
pub fn measure_steps(fs: f64, l: f64, c: f64, periods: f64) -> usize {
    round_int(periods * fs * 2.0 * l / c).max(64) as usize
}

/// Fundamental (Hz) from the mean zero-crossing spacing, each crossing linearly interpolated.
pub fn interp_zero_cross_frequency(sig: &[f64], fs: f64) -> f64 {
    let mean = np_mean(sig);
    let s: Vec<f64> = sig.iter().map(|v| v - mean).collect();
    let idx: Vec<usize> = (0..s.len().saturating_sub(1))
        .filter(|&i| s[i].is_sign_negative() != s[i + 1].is_sign_negative())
        .collect();
    if idx.len() < 3 {
        return f64::NAN;
    }
    let t: Vec<f64> = idx
        .iter()
        .map(|&i| {
            let (a, b) = (s[i], s[i + 1]);
            let span = a - b;
            let frac = if span != 0.0 { a / span } else { 0.0 };
            (i as f64 + frac) / fs
        })
        .collect();
    let dt: Vec<f64> = t.windows(2).map(|w| w[1] - w[0]).collect();
    1.0 / (2.0 * np_mean(&dt))
}

/// One lossless, short mode-1 measurement run.
pub struct Mode1Run {
    pub f: f64,
    pub off_mode: f64,
    pub dt_over_t: f64,
    pub not_converged: f64,
}

/// `_measure_tension_mode1`: frequency, worst off-mode fraction and peak `dT/T0`.
///
/// Purity is against the FIXED `||u_0||`, never the instantaneous `||u||`, which passes through
/// zero twice a period where roundoff would read as a spurious 1.0.
pub fn measure_mode1(p: &Value, amplitude: f64, periods: f64) -> Result<Mode1Run, Refusal> {
    let mut q = p.clone();
    q["sigma0"] = json!(0.0);
    q["sigma1"] = json!(0.0);
    let b = build_resonator(&q)?;
    let (fs, l, c) = (b.fs, b.l, b.c);
    let mut res = b.res.into_tension();
    let shape = mode1_shape(res.p.n);
    let u0: Vec<f64> = shape.iter().map(|s| amplitude * s).collect();
    res.set_state(&u0, &vec![0.0; u0.len()]);
    let scale = norm(&u0);
    let denom = dot(&shape, &shape);
    let n_steps = measure_steps(fs, l, c, periods);

    let mut qs = Vec::with_capacity(n_steps);
    let (mut worst_off, mut peak_dt) = (0.0f64, 0.0f64);
    let mut resid = vec![0.0; shape.len()];
    for _ in 0..n_steps {
        let qi = dot(&res.u, &shape) / denom;
        qs.push(qi);
        for ((r, u), s) in resid.iter_mut().zip(&res.u).zip(&shape) {
            *r = u - qi * s;
        }
        let off = norm(&resid) / scale;
        if off > worst_off {
            worst_off = off;
        }
        let d = res.tension() / res.p.t - 1.0;
        if d > peak_dt {
            peak_dt = d;
        }
        res.step().map_err(|e| Refusal::Internal(e.to_string()))?;
    }
    Ok(Mode1Run {
        f: interp_zero_cross_frequency(&qs, fs),
        off_mode: worst_off,
        dt_over_t: peak_dt,
        not_converged: res.n_not_converged as f64,
    })
}

/// `_pool_band`: slice `[fmin, fmax]`, max-pool to ~[`N_SPEC_POINTS`], normalize to 0..1.
pub fn pool_band(freqs: &[f64], mag: &[f64], fmin: f64, fmax: f64) -> Option<(Vec<f64>, Vec<f64>)> {
    let (f, m): (Vec<f64>, Vec<f64>) = freqs
        .iter()
        .zip(mag)
        .filter(|(&fr, _)| fr >= fmin && fr <= fmax)
        .map(|(&fr, &mg)| (fr, mg))
        .unzip();
    if m.is_empty() {
        return None;
    }
    let npts = N_SPEC_POINTS.min(m.len());
    let edges = linspace_idx(m.len(), npts + 1);
    let mut f_ds = Vec::with_capacity(npts);
    let mut m_ds = Vec::with_capacity(npts);
    for i in 0..npts {
        let lo = edges[i];
        let hi = (edges[i] + 1).max(edges[i + 1]);
        f_ds.push(np_mean(&f[lo..hi]));
        m_ds.push(m[lo..hi].iter().copied().fold(f64::NEG_INFINITY, f64::max));
    }
    let mmax = m_ds.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if mmax > 0.0 {
        for v in &mut m_ds {
            *v /= mmax;
        }
    }
    Some((f_ds, m_ds))
}

/// Decimated magnitude-spectrum length for the plot.
pub const N_SPEC_POINTS: usize = 420;

/// `_pooled_spectrum`: the magnitude spectrum over `[0, fmax]`, pooled.
pub fn pooled_spectrum(pickup: &[f64], fs: f64, fmax: f64) -> Option<(Vec<f64>, Vec<f64>)> {
    let s = spectrum::magnitude_spectrum(pickup, fs, 2);
    pool_band(&s.freqs, &s.mag, 0.0, fmax)
}

/// `_tension_spectrum_block`: the measured amplitude shift against the exact Duffing closed form.
pub fn spectrum_block(
    p: &Value,
    pickup: &[f64],
    fs: f64,
    w0sq: f64,
    eps: f64,
    amplitude: f64,
    periods: f64,
) -> Result<Value, Refusal> {
    let f_lin = w0sq.sqrt() / (2.0 * PI);
    let shift_oracle = duffing::duffing_frequency_shift(amplitude, w0sq, eps)
        .map_err(Refusal::Construction)?
        / (2.0 * PI);

    let run_a = measure_mode1(p, amplitude, periods)?;
    let run_ref = measure_mode1(p, TENSION_REF_AMP, periods)?;
    let off = if run_ref.off_mode > run_a.off_mode {
        run_ref.off_mode
    } else {
        run_a.off_mode
    };
    let pure = off < TENSION_OFFMODE_MAX;
    let measured = run_a.f.is_finite() && run_ref.f.is_finite();

    let (mut shift_meas, mut rel_err) = (None, None);
    if pure && measured {
        let s = run_a.f - run_ref.f;
        shift_meas = Some(s);
        if shift_oracle > 0.0 {
            rel_err = Some((s - shift_oracle).abs() / shift_oracle);
        }
    }
    let fmax = (4.0 * f_lin).max(1.25 * (f_lin + shift_oracle));
    let (freq, mag) = match pooled_spectrum(pickup, fs, fmax) {
        None => (json!([]), json!([])),
        Some((f, m)) => (finite_list(&f, Some(3)), finite_list(&m, Some(5))),
    };
    let rnd = |x: f64, nd: usize| num(round_nd(x, nd));
    Ok(json!({
        "kind": "tension",
        "freq": freq,
        "mag": mag,
        "fmax": rnd(fmax, 3),
        "f_linear": rnd(f_lin, 4),
        "f_hardened": if measured { rnd(run_a.f, 4) } else { Value::Null },
        "f_reference": if measured { rnd(run_ref.f, 4) } else { Value::Null },
        "shift_measured": shift_meas.map_or(Value::Null, |s| rnd(s, 4)),
        "shift_oracle": rnd(shift_oracle, 4),
        "shift_rel_error": opt_num(rel_err),
        "shift_cents": match shift_meas {
            Some(s) if f_lin + s > 0.0 => rnd(modal::cents(f_lin + s, f_lin), 3),
            _ => Value::Null,
        },
        "dT_over_T": rnd(run_a.dt_over_t, 4),
        "purity": {"off_mode": num(run_a.off_mode), "tol": num(TENSION_OFFMODE_MAX), "pure": pure},
    }))
}

/// Read `N` the way both regimes pre-check it: `int(p.get("N", 128))`, refused as `param`.
fn requested_n(p: &Value) -> Result<i64, Refusal> {
    let v = p.get("N").cloned().unwrap_or(json!(128));
    py_int(&v).map_err(|_| Refusal::Param(format!("N must be an integer, got {}.", py_repr(&v))))
}

/// The per-step tension root-find's convergence, folded into the energy verdict.
fn convergence(res: &TensionModulatedString, detail_tail: &str) -> Value {
    let n_bad = res.n_not_converged;
    let tol = physsynth_core::fmt::py_exp(res.p.tension_tol, 0);
    json!({
        "all_converged": n_bad == 0,
        "n_not_converged": int(n_bad as i64),
        "tension_tol": num(res.p.tension_tol),
        "bracket_expansions": int(res.bracket_expansions as i64),
        "detail": format!(
            "tension root-find did not converge: {n_bad} step(s), tol {tol}\n{detail_tail}"
        ),
        "note": format!("  ·  tension solve converged (tol {tol})"),
    })
}

/// `_build_payload_tension` — the Duffing regime.
pub fn build_payload_duffing(p: &Value) -> Result<Value, Refusal> {
    build_payload_duffing_with(p, TENSION_MEASURE_PERIODS)
}

/// [`build_payload_duffing`] with the measurement-run length as an argument.
pub fn build_payload_duffing_with(p: &Value, periods: f64) -> Result<Value, Refusal> {
    let audio_dur = fnum(p, "audio_duration", 2.0)?;
    let anim_win = fnum(p, "animation_window", 0.06)?;
    let playback_speed = fnum(p, "playback_speed", 0.02)?;
    let amplitude = fnum(p, "amplitude", TENSION_AMP_DEFAULT)?;
    let pickup_frac = fnum(p, "pickup_position", 0.1)?;
    let fpp = int_at_least_one(fnum(p, "frames_per_period", FRAMES_PER_PERIOD)?)?;

    if !(0.0 < audio_dur && audio_dur <= TENSION_AUDIO_MAX) {
        return Err(Refusal::Param(format!(
            "audio_duration must be in (0, {}] s, got {}.",
            pf(TENSION_AUDIO_MAX),
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
    if !(0.0 < amplitude && amplitude <= TENSION_AMP_MAX) {
        return Err(Refusal::Param(format!(
            "amplitude must be in (0, {}] m, got {}.",
            pf(TENSION_AMP_MAX),
            pf(amplitude)
        )));
    }
    if !(0.0 < pickup_frac && pickup_frac < 1.0) {
        return Err(Refusal::Param(format!(
            "pickup_position must be in (0, 1), got {}.",
            pf(pickup_frac)
        )));
    }
    let n_req = requested_n(p)?;
    if n_req > TENSION_N_MAX {
        return Err(Refusal::Param(format!(
            "N must be <= {TENSION_N_MAX} for the tension string (got {n_req}): every step runs a \
             tension root-find, so a fine grid is far costlier here than on a linear string."
        )));
    }

    let b = build_resonator(p)?;
    let (c, l, n, fs) = (b.c, b.l, b.n, b.fs);
    let (sigma_zero, oracle_2sigma) = (b.sigma_zero, b.oracle_2sigma);
    let mut res = b.res.into_tension();
    let pickup_idx = round_int(pickup_frac * n as f64).max(1).min(n - 1) as usize;

    // Mode 1's Duffing coefficients on the DISCRETE grid — the eigenvalue the stepper rings at.
    let p2 = damping::spatial_eigenvalue_p2(n, res.p.h, 1);
    let (w0sq, eps) = duffing::kc_mode_coefficients(c, res.p.kappa, res.p.ea, res.p.rho, p2, l)
        .map_err(Refusal::Construction)?;
    let f_lin = w0sq.sqrt() / (2.0 * PI);
    let f_hard_est = f_lin
        + duffing::duffing_frequency_shift(amplitude, w0sq, eps).map_err(Refusal::Construction)?
            / (2.0 * PI);

    let dt = dt_over_t(res.p.ea, amplitude, p2, res.p.t);
    if dt > TENSION_DT_MAX {
        return Err(Refusal::Param(format!(
            "dT/T0 = {dt:.2} exceeds {} — above this the single mode is parametrically unstable \
             and breaks up into its neighbours, so the Duffing shift stops describing the motion. \
             (The breakup is real, energy-conserving physics; it wants a panel of its own.) Lower \
             the amplitude or EA.",
            pf(TENSION_DT_MAX)
        )));
    }

    let n_measure = measure_steps(fs, l, c, periods) as i64;
    let n_audio_req = round_int(audio_dur * fs).max(1);
    let work = n_audio_req + 2 * n_measure + round_int(anim_win * fs);
    if work > TENSION_WORK_MAX {
        return Err(Refusal::Param(format!(
            "work budget exceeded ({} steps > {}): every step runs a tension root-find (~176 µs \
             at N=128). Lower N or audio_duration.",
            commas(work),
            commas(TENSION_WORK_MAX)
        )));
    }

    let shape = mode1_shape(res.p.n);
    let u0: Vec<f64> = shape.iter().map(|s| amplitude * s).collect();
    res.set_state(&u0, &vec![0.0; u0.len()]);
    // At the IC, which is the PEAK: all of E is potential and the stretch is maximal.
    let e0 = res.energy();
    let nl_fraction = if e0 > 0.0 {
        res.nonlinear_energy() / e0
    } else {
        0.0
    };

    let n_audio = n_audio_req;
    let audio_run =
        simulate(&mut res, n_audio as usize, Some(pickup_idx), 0).map_err(Refusal::Internal)?;
    let pickup = audio_run.output.as_deref().expect("a pickup was requested");
    if !pickup.iter().all(|v| v.is_finite()) {
        return Err(Refusal::Param(
            "simulation produced non-finite output (instability) — adjust parameters.".into(),
        ));
    }

    // The stride rides the HARDENED estimate: a linear-f1 stride would under-resolve the motion.
    let mut anim = build_resonator(p)?.res.into_tension();
    let u0a: Vec<f64> = mode1_shape(anim.p.n)
        .iter()
        .map(|s| amplitude * s)
        .collect();
    anim.set_state(&u0a, &vec![0.0; u0a.len()]);
    let mut anim_stride = round_int((fs / f_hard_est) / fpp as f64).max(1);
    let n_anim = anim_stride.max(round_int(anim_win * fs));
    if n_anim / anim_stride > MAX_FRAMES {
        anim_stride = ((n_anim as f64 / MAX_FRAMES as f64).ceil() as i64).max(1);
    }
    let anim_run = simulate(&mut anim, n_anim as usize, None, anim_stride as usize)
        .map_err(Refusal::Internal)?;
    let frames = Frames::of(&anim_run.snapshots, 0);

    let (audio48, peak) = resample_normalize(pickup, fs);
    let conv = convergence(&res, "energy verdict N/A — lower the amplitude or EA");
    let energy = energy_block(
        &audio_run.time,
        &audio_run.energy,
        sigma_zero,
        oracle_2sigma,
        EnergyOpts {
            convergence: Some(conv),
            ..EnergyOpts::default()
        },
    );
    let spectrum = spectrum_block(p, pickup, fs, w0sq, eps, amplitude, periods)?;

    Ok(json!({
        "model": "tension",
        "fs_sim": num(round_nd(fs, 3)),
        "lambda": num(round_nd(res.p.lam, 6)),
        "grid": {"x": finite_list(&res.p.grid(), Some(6))},
        "frames": frames.json(),
        "frame_times": frames.times(fs),
        "anim_dt": num(anim_stride as f64 / fs),
        "playback_speed": num(playback_speed),
        "field_amp": num(frames.amp()),
        "audio": {
            "b64": crate::py::b64f32(&audio48),
            "fs": num(AUDIO_FS),
            "peak": num(peak),
            "n": int(audio48.len() as i64),
        },
        "energy": energy,
        "meta": {
            "c": num(round_nd(c, 3)),
            "f1": num(round_nd(f_lin, 3)),
            "num_steps": int(n_audio),
            "n_frames": int(frames.n as i64),
            "spectrum": spectrum,
            "EA_over_T": num(round_nd(res.p.ea_over_t, 3)),
            "nonlinear_fraction": num(round_nd(nl_fraction, 6)),
        },
    }))
}

// == the parametric regime ========================================================================

/// `_parametric_seed`: unit-norm off-mode seed over modes 1..=24, mode `m` removed.
pub fn parametric_seed(n: usize, m: usize) -> Vec<f64> {
    let mut v = vec![0.0; n + 1];
    for (mm, &coef) in PARAM_SEED_COEF
        .iter()
        .enumerate()
        .take(PARAM_SEED_MODES + 1)
        .skip(1)
    {
        if mm != m {
            let shape = mode_shape(n, mm);
            for (vi, s) in v.iter_mut().zip(&shape) {
                *vi += coef * s;
            }
        }
    }
    let nv = norm(&v);
    v.iter().map(|x| x / nv).collect()
}

/// `_param_mode_frequency`: `(f_m, p2)` of mode `m` on the discrete grid.
pub fn mode_frequency(res: &TensionModulatedString, m: usize) -> (f64, f64) {
    let p2 = damping::spatial_eigenvalue_p2(res.p.n as i64, res.p.h, m as i64);
    let w0sq = scalar_pow(res.p.c, 2.0) * p2 + scalar_pow(res.p.kappa, 2.0) * scalar_pow(p2, 2.0);
    (w0sq.sqrt() / (2.0 * PI), p2)
}

/// `_param_amplitude`: invert `dT/T0 = EA A² p2/(4T)`.
pub fn amplitude_for(res: &TensionModulatedString, dt: f64, p2: f64) -> f64 {
    (4.0 * res.p.t * dt / (res.p.ea * p2)).sqrt()
}

/// `_off_fraction`: off-mode content of `u` against the FIXED `scale = ||u_0||`.
fn off_fraction(u: &[f64], shape: &[f64], denom: f64, scale: f64) -> f64 {
    let q = dot(u, shape) / denom;
    let r: Vec<f64> = u.iter().zip(shape).map(|(ui, si)| ui - q * si).collect();
    norm(&r) / scale
}

/// The seeded single-mode start: `A shape + seed_rel A seed`.
fn seeded_start(n: usize, m: usize, amplitude: f64, seed_rel: f64) -> (Vec<f64>, Vec<f64>) {
    let shape = mode_shape(n, m);
    let mut u0: Vec<f64> = shape.iter().map(|s| amplitude * s).collect();
    if seed_rel > 0.0 {
        let seed = parametric_seed(n, m);
        let k = seed_rel * amplitude;
        for (u, s) in u0.iter_mut().zip(&seed) {
            *u += k * s;
        }
    }
    (shape, u0)
}

/// One claim run: energy, off-mode trace, per-mode running peaks, frames and pickup.
pub struct ParamRun {
    pub energy: Vec<f64>,
    pub off: Vec<f64>,
    pub off_t: Vec<f64>,
    pub pickup: Option<Vec<f64>>,
    pub snapshots: Vec<(usize, Vec<f64>)>,
    pub amplitude: f64,
    /// Running max of `|phi_m . u| / |phi_m|²` from `modal_from` on, divided by the amplitude.
    pub modal_peak: Vec<f64>,
    pub n_modes: usize,
    pub nl_fraction: f64,
    /// `seed_rel A / ||u_0||` — what "did it grow?" is decided against.
    pub seed_floor: f64,
}

/// `_run_parametric`.
#[allow(clippy::too_many_arguments)]
pub fn run_parametric(
    res: &mut TensionModulatedString,
    m: usize,
    amplitude: f64,
    n_steps: usize,
    seed_rel: f64,
    pickup_index: Option<usize>,
    snapshot_stride: usize,
    modal_from: usize,
) -> Result<ParamRun, Refusal> {
    let n = res.p.n;
    let (shape, u0) = seeded_start(n, m, amplitude, seed_rel);
    res.set_state(&u0, &vec![0.0; u0.len()]);
    let scaled: Vec<f64> = shape.iter().map(|s| amplitude * s).collect();
    let scale = norm(&scaled);
    let denom = dot(&shape, &shape);
    let e_ic = res.energy();
    let nl_fraction = if e_ic > 0.0 {
        res.nonlinear_energy() / e_ic
    } else {
        0.0
    };

    let n_modes = (n / 2).min(40);
    let phi: Vec<Vec<f64>> = (1..=n_modes).map(|mm| mode_shape(n, mm)).collect();
    let phi_norm: Vec<f64> = phi.iter().map(|r| dot(r, r)).collect();
    let mut modal_peak = vec![0.0; n_modes];

    let mut energy = Vec::with_capacity(n_steps + 1);
    let mut pickup = pickup_index.map(|_| Vec::with_capacity(n_steps + 1));
    let n_off = n_steps / PARAM_SAMPLE + 1;
    let mut off = Vec::with_capacity(n_off);
    let mut off_t = Vec::with_capacity(n_off);
    let mut snaps = Vec::new();

    energy.push(res.energy());
    if let (Some(pk), Some(i)) = (pickup.as_mut(), pickup_index) {
        pk.push(res.u[i]);
    }
    if snapshot_stride != 0 {
        snaps.push((0, res.u.clone()));
    }
    off.push(off_fraction(&res.u, &shape, denom, scale));
    off_t.push(0.0);

    for i in 1..=n_steps {
        res.step().map_err(|e| Refusal::Internal(e.to_string()))?;
        energy.push(res.energy());
        if let (Some(pk), Some(j)) = (pickup.as_mut(), pickup_index) {
            pk.push(res.u[j]);
        }
        if snapshot_stride != 0 && i % snapshot_stride == 0 {
            snaps.push((i, res.u.clone()));
        }
        if i % PARAM_SAMPLE == 0 {
            if off.len() < n_off {
                off.push(off_fraction(&res.u, &shape, denom, scale));
                off_t.push(i as f64 / res.p.fs);
            }
            if i >= modal_from {
                for (k, row) in phi.iter().enumerate() {
                    let v = dot(row, &res.u).abs() / phi_norm[k];
                    // `np.maximum`: NaN-propagating.
                    if v > modal_peak[k] || v.is_nan() {
                        modal_peak[k] = v;
                    }
                }
            }
        }
    }
    Ok(ParamRun {
        energy,
        off,
        off_t,
        pickup,
        snapshots: snaps,
        amplitude,
        modal_peak: modal_peak.iter().map(|v| v / amplitude).collect(),
        n_modes,
        nl_fraction,
        seed_floor: if seed_rel > 0.0 {
            seed_rel * amplitude / scale
        } else {
            0.0
        },
    })
}

/// One sweep point's outcome.
pub struct Saturation {
    pub level: f64,
    pub floor: f64,
    pub growth: f64,
    pub unstable: bool,
    pub saturated: bool,
    pub steps: usize,
    pub periods: f64,
    pub drift: f64,
    pub n_not_converged: usize,
}

/// `_param_saturation_run`: grow the seed until the off-mode level saturates, or stop at the cap.
pub fn saturation_run(
    res: &mut TensionModulatedString,
    m: usize,
    amplitude: f64,
    cap_periods: f64,
    chunk_periods: f64,
    f_m: f64,
    seed_rel: f64,
) -> Result<Saturation, Refusal> {
    let n = res.p.n;
    let (shape, u0) = seeded_start(n, m, amplitude, seed_rel);
    res.set_state(&u0, &vec![0.0; u0.len()]);
    let scaled: Vec<f64> = shape.iter().map(|s| amplitude * s).collect();
    let scale = norm(&scaled);
    let denom = dot(&shape, &shape);
    let steps_per_period = res.p.fs / f_m;
    let chunk = round_int(chunk_periods * steps_per_period).max(8) as usize;
    let cap = (round_int(cap_periods * steps_per_period).max(chunk as i64)) as usize;

    let e0 = res.energy();
    let mut chunk_max: Vec<f64> = Vec::new();
    let (mut cur, mut saturated, mut i) = (0.0f64, false, 0usize);
    while i < cap {
        res.step().map_err(|e| Refusal::Internal(e.to_string()))?;
        i += 1;
        if i % PARAM_SAMPLE == 0 {
            let o = off_fraction(&res.u, &shape, denom, scale);
            if o > cur {
                cur = o;
            }
        }
        if i % chunk == 0 {
            chunk_max.push(cur);
            cur = 0.0;
            let k = chunk_max.len();
            if k >= 2 && chunk_max[k - 1] <= PARAM_SAT_RATIO * chunk_max[k - 2] {
                saturated = true;
                break;
            }
        }
    }
    // Python's `max(list)`: the first element unless a later one is strictly greater.
    let level = chunk_max
        .iter()
        .copied()
        .reduce(|a, b| if b > a { b } else { a })
        .unwrap_or(cur);
    let floor = seed_rel * amplitude / scale;
    Ok(Saturation {
        level,
        floor,
        growth: if floor > 0.0 { level / floor } else { 0.0 },
        unstable: level > PARAM_UNSTABLE_FACTOR * floor,
        saturated,
        steps: i,
        periods: i as f64 / steps_per_period,
        drift: if e0 > 0.0 {
            (res.energy() - e0).abs() / e0
        } else {
            0.0
        },
        n_not_converged: res.n_not_converged,
    })
}

/// `_param_sweep`: the saturated off-mode level against `dT/T0`, at fixed N and sigma = 0.
///
/// `work_max` is [`PARAM_SWEEP_WORK_MAX`] on the shipped path; a test lowers it to reach the
/// truncation labelling, which cannot fire in the shipped range.
pub fn sweep(
    base: &Value,
    m: usize,
    dts: &[f64],
    cap_periods: f64,
    chunk_periods: i64,
    work_max: i64,
) -> Result<Value, Refusal> {
    let mut points = Vec::new();
    let (mut spent, mut truncated) = (0i64, false);
    let mut sweep_p = base.clone();
    sweep_p["N"] = json!(PARAM_SWEEP_N);
    sweep_p["sigma0"] = json!(0.0);
    sweep_p["sigma1"] = json!(0.0);

    // (dt, unstable, saturated) of every point that ran
    let mut done: Vec<(f64, bool, bool)> = Vec::new();
    for &dt in dts {
        if spent >= work_max {
            truncated = true;
            points.push(json!({"dt": num(dt), "level": null, "truncated": true}));
            continue;
        }
        let mut res = build_resonator(&sweep_p)?.res.into_tension();
        let (f_m, p2) = mode_frequency(&res, m);
        let amp = amplitude_for(&res, dt, p2);
        let r = saturation_run(
            &mut res,
            m,
            amp,
            cap_periods,
            chunk_periods as f64,
            f_m,
            PARAM_SEED_REL,
        )?;
        spent += r.steps as i64;
        done.push((dt, r.unstable, r.saturated));
        points.push(json!({
            "dt": num(dt),
            "amplitude": num(round_nd(amp, 6)),
            "level": num(round_nd(r.level, 6)),
            "floor": num(r.floor),
            "growth": num(round_nd(r.growth, 1)),
            "unstable": r.unstable,
            "saturated": r.saturated,
            "periods": num(round_nd(r.periods, 1)),
            "drift": num(r.drift),
            "n_not_converged": int(r.n_not_converged as i64),
            "truncated": false,
        }));
    }
    let unstable: Vec<f64> = done.iter().filter(|d| d.1).map(|d| d.0).collect();
    let stable: Vec<f64> = done.iter().filter(|d| !d.1).map(|d| d.0).collect();
    // Python's `min`/`max` over a sequence: first unless a later one is strictly smaller/greater.
    let py_min = |v: &[f64]| v.iter().copied().reduce(|a, b| if b < a { b } else { a });
    let py_max = |v: &[f64]| v.iter().copied().reduce(|a, b| if b > a { b } else { a });
    let min_unstable = py_min(&unstable);
    let below: Vec<f64> = stable
        .iter()
        .copied()
        .filter(|&d| min_unstable.is_none_or(|u| d < u))
        .collect();
    Ok(json!({
        "points": points,
        "N": int(PARAM_SWEEP_N),
        "seed_rel": num(PARAM_SEED_REL),
        "cap_periods": num(cap_periods),
        "chunk_periods": int(chunk_periods),
        "edge_lo": opt_num(py_max(&below)),
        "edge_hi": opt_num(min_unstable),
        "truncated": truncated,
        "steps": int(spent),
        "n_unstable": int(unstable.len() as i64),
        "n_unsaturated": int(done.iter().filter(|d| !d.2).count() as i64),
    }))
}

/// `_param_sweep_dts`: the sweep grid, or a hidden `sweep_points` override (the test suite's).
pub fn sweep_dts(p: &Value) -> Result<Vec<f64>, Refusal> {
    let raw = match p.get("sweep_points") {
        None | Some(Value::Null) => return Ok(PARAM_SWEEP_DTS.to_vec()),
        Some(r) => r,
    };
    let bad_type = || {
        Refusal::Param(format!(
            "sweep_points must be a list of numbers, got {}.",
            py_repr(raw)
        ))
    };
    // `tuple(float(v) for v in raw)` — whatever Python can iterate: a list's items, a dict's
    // keys, a string's characters.
    let items: Vec<Value> = match raw {
        Value::Array(a) => a.clone(),
        Value::Object(m) => m.keys().map(|k| json!(k)).collect(),
        Value::String(s) => s.chars().map(|c| json!(c.to_string())).collect(),
        _ => return Err(bad_type()),
    };
    let dts: Vec<f64> = items
        .iter()
        .map(py_float)
        .collect::<Option<Vec<f64>>>()
        .ok_or_else(bad_type)?;
    if dts.is_empty() || dts.iter().any(|&d| !(0.0 < d && d <= PARAM_DT_MAX)) {
        return Err(Refusal::Param(format!(
            "sweep_points must all be in (0, {}], got {}.",
            pf(PARAM_DT_MAX),
            py_repr(raw)
        )));
    }
    Ok(dts)
}

/// `_param_envelope`: sliding MAX over one period of the driven mode, edge-padded.
pub fn envelope(off: &[f64], win: usize) -> Vec<f64> {
    if win < 2 || off.len() < win {
        return off.to_vec();
    }
    let pad = win / 2;
    let mut padded = vec![off[0]; pad];
    padded.extend_from_slice(off);
    padded.extend(std::iter::repeat_n(off[off.len() - 1], win - 1 - pad));
    padded
        .windows(win)
        .map(|w| {
            // `.max(axis=1)`: NaN-propagating.
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
        .collect()
}

/// `_param_trace`: one claim run's off-mode story, and the index it was decimated on.
fn trace(run: &ParamRun, fs: f64, f_m: f64) -> (Map<String, Value>, Vec<usize>, bool) {
    let max_points = 500usize;
    let off = &run.off;
    let win = round_int((fs / f_m) / PARAM_SAMPLE as f64).max(2) as usize;
    let env = envelope(off, win);
    let step = ((off.len() as f64 / max_points as f64).ceil() as usize).max(1);
    let idx: Vec<usize> = (0..off.len()).step_by(step).collect();
    let floor = run.seed_floor;
    let level = if env.is_empty() {
        0.0
    } else {
        max_abs_signed(&env)
    };
    let unstable = level > PARAM_UNSTABLE_FACTOR * floor;
    let mut m = Map::new();
    let pick = |a: &[f64]| idx.iter().map(|&i| a[i]).collect::<Vec<f64>>();
    m.insert("t".into(), finite_list(&pick(&run.off_t), Some(6)));
    // NOT rounded: a log axis spanning 1e-7..1e0, where rounding deletes the flat line.
    m.insert("off".into(), finite_list(&pick(off), None));
    m.insert("env".into(), finite_list(&pick(&env), None));
    m.insert("floor".into(), num(floor));
    m.insert("level".into(), num(level));
    m.insert(
        "growth".into(),
        num(if floor > 0.0 {
            round_nd(level / floor, 1)
        } else {
            0.0
        }),
    );
    m.insert("unstable".into(), Value::Bool(unstable));
    m.insert("nl_fraction".into(), num(round_nd(run.nl_fraction, 6)));
    (m, idx, unstable)
}

/// `np.max` of a non-empty array (the envelope is non-negative, so this is its plain max).
fn max_abs_signed(a: &[f64]) -> f64 {
    let mut m = f64::NEG_INFINITY;
    for &v in a {
        if v.is_nan() {
            return f64::NAN;
        }
        if v > m {
            m = v;
        }
    }
    m
}

/// `_param_cascade`: where the energy landed, as a max over the run's second half.
fn cascade(run: &ParamRun, m: usize) -> Result<Value, Refusal> {
    let peak = &run.modal_peak;
    let n_modes = run.n_modes;
    if m > n_modes {
        // The reference indexed past the array here and the request died with a 500.
        return Err(Refusal::Internal(format!(
            "mode_number {m} is above the {n_modes} modes tracked at this N; raise N."
        )));
    }
    let lo = (2 * m + 8).max(n_modes / 2).min(n_modes - 1);
    let grid_scale = if lo < n_modes {
        max_abs_signed(&peak[lo..])
    } else {
        0.0
    };
    let bar = (100.0 * grid_scale).max(PARAM_PARTNER_FLOOR);
    let mut others: Vec<(usize, f64)> = (1..=n_modes)
        .filter(|&mm| mm != m && peak[mm - 1] > bar)
        .map(|mm| (mm, peak[mm - 1]))
        .collect();
    // `sorted(..., key=lambda kv: -kv[1])`: stable, descending.
    others.sort_by(|a, b| {
        (-a.1)
            .partial_cmp(&-b.1)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    others.truncate(4);
    let best = others.first().map_or(0.0, |t| t.1);
    Ok(json!({
        "modes": (1..=n_modes).map(|mm| int(mm as i64)).collect::<Vec<_>>(),
        "peak": finite_list(peak, Some(9)),
        "driven": int(m as i64),
        "driven_peak": num(round_nd(peak[m - 1], 6)),
        "grid_scale": num(grid_scale),
        "grid_from": int(lo as i64 + 1),
        "top": others
            .iter()
            .map(|(mm, a)| json!([int(*mm as i64), num(round_nd(*a, 6))]))
            .collect::<Vec<_>>(),
        "over_grid": if !others.is_empty() && grid_scale > 0.0 {
            num(round_nd(best / grid_scale, 1))
        } else {
            Value::Null
        },
    }))
}

/// `_param_block`: the regime's second panel, assembled.
#[allow(clippy::too_many_arguments)]
fn param_block(
    fs: f64,
    m: usize,
    f_m: f64,
    periods: f64,
    above: &ParamRun,
    below: &ParamRun,
    dt_above: f64,
    dt_below: f64,
    sweep: Value,
) -> Result<Value, Refusal> {
    let (tr_above, idx_above, un_above) = trace(above, fs, f_m);
    let (tr_below, _, un_below) = trace(below, fs, f_m);
    // The UNSTABLE run's drift on the same axes as its growth; the headline is the full array's.
    let e = &above.energy;
    let e0 = e[0];
    let drift: Vec<f64> = idx_above
        .iter()
        .map(|&i| {
            let j = (i * PARAM_SAMPLE).min(e.len() - 1);
            if e0 != 0.0 {
                (e[j] - e0).abs() / e0.abs()
            } else {
                e[j].abs()
            }
        })
        .collect();
    let straddles = un_above && !un_below;
    let nl_gap = (round_nd(above.nl_fraction, 6) - round_nd(below.nl_fraction, 6)).abs();
    Ok(json!({
        "kind": "parametric",
        "m": int(m as i64),
        "f_m": num(round_nd(f_m, 3)),
        "periods": num(round_nd(periods, 1)),
        "seed_rel": num(PARAM_SEED_REL),
        "seed_modes": int(PARAM_SEED_MODES as i64),
        "dt_above": num(round_nd(dt_above, 4)),
        "dt_below": num(round_nd(dt_below, 4)),
        "amp_above": num(round_nd(above.amplitude, 6)),
        "amp_below": num(round_nd(below.amplitude, 6)),
        "above": tr_above,
        "below": tr_below,
        "drift_trace": finite_list(&drift, None),
        "cascade": cascade(above, m)?,
        "sweep": sweep,
        "straddles": straddles,
        "nl_gap": num(round_nd(nl_gap, 6)),
        "note": if straddles {
            "both runs are equally nonlinear — the tongue is the difference"
        } else {
            "the pair does not straddle this mode's threshold — see the sweep for where it is"
        },
    }))
}

/// `_build_payload_parametric`.
pub fn build_payload_parametric(p: &Value) -> Result<Value, Refusal> {
    build_payload_parametric_with(p, PARAM_SWEEP_WORK_MAX)
}

/// [`build_payload_parametric`] with the sweep's work budget as an argument.
pub fn build_payload_parametric_with(p: &Value, sweep_work_max: i64) -> Result<Value, Refusal> {
    let playback_speed = fnum(p, "playback_speed", 0.02)?;
    let pickup_frac = fnum(p, "pickup_position", 0.1)?;
    let fpp = int_at_least_one(fnum(p, "frames_per_period", FRAMES_PER_PERIOD)?)?;
    let dt_above = fnum(p, "dt_over_t0", PARAM_DT_ABOVE_DEFAULT)?;
    let dt_below = fnum(p, "dt_below", PARAM_DT_BELOW_DEFAULT)?;
    let periods = fnum(p, "claim_periods", PARAM_CLAIM_PERIODS)?;

    // `try: int(_fnum(...)) except (TypeError, ValueError)` — and `ParamError` IS a `ValueError`,
    // so an unreadable value gets THIS message, not `_fnum`'s own "must be a number".
    let m = match fnum(p, "mode_number", PARAM_M_DEFAULT)
        .ok()
        .and_then(|x| crate::py::float_to_int(x).ok())
    {
        Some(m) => m,
        None => {
            let shown = p.get("mode_number").map_or("None".to_owned(), py_repr);
            return Err(Refusal::Param(format!(
                "mode_number must be an integer, got {shown}."
            )));
        }
    };
    if !(1..=PARAM_M_MAX).contains(&m) {
        return Err(Refusal::Param(format!(
            "mode_number must be in [1, {PARAM_M_MAX}], got {m}."
        )));
    }
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
    for (name, val) in [("dt_over_t0", dt_above), ("dt_below", dt_below)] {
        if !(0.0 < val && val <= PARAM_DT_MAX) {
            return Err(Refusal::Param(format!(
                "{name} must be in (0, {}], got {}.",
                pf(PARAM_DT_MAX),
                pf(val)
            )));
        }
    }
    if !(1.0..=200.0).contains(&periods) {
        return Err(Refusal::Param(format!(
            "claim_periods must be in [1, 200], got {}.",
            pf(periods)
        )));
    }
    let n_req = requested_n(p)?;
    if n_req > PARAM_N_MAX {
        return Err(Refusal::Param(format!(
            "N must be <= {PARAM_N_MAX} for the parametric regime (got {n_req}): every step runs \
             a tension root-find (~250 µs), and this regime runs two claim runs plus a sweep."
        )));
    }
    let m = m as usize;

    // sigma is FORCED to 0, not defaulted: under loss the instability self-extinguishes mid-panel.
    let mut base = p.clone();
    base["sigma0"] = json!(0.0);
    base["sigma1"] = json!(0.0);
    let b = build_resonator(&base)?;
    let c = b.c;
    let mut res = b.res.into_tension();
    if res.p.ea <= 0.0 {
        return Err(Refusal::Param(
            "EA must be > 0 in the parametric regime: with EA = 0 the string is model #3 exactly, \
             the tension never modulates, and there is no parametric pump to be unstable to."
                .into(),
        ));
    }
    let fs = res.p.fs;
    let n = res.p.n as i64;
    let (f_m, p2) = mode_frequency(&res, m);
    let amp_above = amplitude_for(&res, dt_above, p2);
    let amp_below = amplitude_for(&res, dt_below, p2);
    let pickup_idx = round_int(pickup_frac * n as f64).max(1).min(n - 1) as usize;

    let n_steps = round_int(periods * fs / f_m).max(16);
    if 2 * n_steps > PARAM_WORK_MAX {
        return Err(Refusal::Param(format!(
            "work budget exceeded ({} steps > {} for the two claim runs): every step runs a \
             tension root-find. Lower N, claim_periods, or the mode number (a higher mode needs \
             more steps per unit time).",
            commas(2 * n_steps),
            commas(PARAM_WORK_MAX)
        )));
    }
    let n_steps = n_steps as usize;
    let mut stride = round_int((fs / f_m) / fpp as f64).max(1);
    if n_steps as i64 / stride > MAX_FRAMES {
        stride = ((n_steps as f64 / MAX_FRAMES as f64).ceil() as i64).max(1);
    }

    // ONE above-threshold run yields the verdict, the trace, the cascade, the frames and the audio:
    // the claim window IS the animation window here.
    let above = run_parametric(
        &mut res,
        m,
        amp_above,
        n_steps,
        PARAM_SEED_REL,
        Some(pickup_idx),
        stride as usize,
        n_steps / 2,
    )?;
    let mut res_below = build_resonator(&base)?.res.into_tension();
    let below = run_parametric(
        &mut res_below,
        m,
        amp_below,
        n_steps,
        PARAM_SEED_REL,
        None,
        0,
        0,
    )?;

    let dts = sweep_dts(p)?;
    let cap = fnum(p, "sweep_cap", PARAM_SWEEP_CAP_PERIODS)?;
    let sweep = sweep(
        &base,
        m,
        &dts,
        cap,
        PARAM_SWEEP_CHUNK_PERIODS,
        sweep_work_max,
    )?;

    let frames = Frames::of(&above.snapshots, 0);
    let pickup = above.pickup.as_deref().expect("a pickup was requested");
    let (audio48, peak) = resample_normalize(pickup, fs);
    let time: Vec<f64> = (0..above.energy.len()).map(|i| i as f64 / fs).collect();
    let conv = convergence(
        &res,
        "verdict N/A — the breakup cannot be told from a failed solve",
    );
    let energy = energy_block(
        &time,
        &above.energy,
        true,
        0.0,
        EnergyOpts {
            convergence: Some(conv),
            ..EnergyOpts::default()
        },
    );
    let spectrum = param_block(
        fs, m, f_m, periods, &above, &below, dt_above, dt_below, sweep,
    )?;

    Ok(json!({
        "model": "tension",
        "regime": "parametric",
        "fs_sim": num(round_nd(fs, 3)),
        "lambda": num(round_nd(res.p.lam, 6)),
        "grid": {"x": finite_list(&res.p.grid(), Some(6))},
        "frames": frames.json(),
        "frame_times": frames.times(fs),
        "anim_dt": num(stride as f64 / fs),
        "playback_speed": num(playback_speed),
        "field_amp": num(if frames.flat.is_empty() { 0.0 } else { max_abs(&frames.flat) }),
        "audio": {
            "b64": crate::py::b64f32(&audio48),
            "fs": num(AUDIO_FS),
            "peak": num(peak),
            "n": int(audio48.len() as i64),
        },
        "energy": energy,
        "meta": {
            "c": num(round_nd(c, 3)),
            "f1": num(round_nd(f_m / m as f64, 3)),
            "num_steps": int(n_steps as i64),
            "n_frames": int(frames.n as i64),
            "EA_over_T": num(round_nd(res.p.ea_over_t, 3)),
            "nonlinear_fraction": num(round_nd(above.nl_fraction, 6)),
            "spectrum": spectrum,
        },
    }))
}
