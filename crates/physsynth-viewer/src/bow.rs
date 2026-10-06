//! The bowed string — `serialize.py`'s `_build_payload_bow`, `_run_bow` and `_bow_stickslip_block`.
//!
//! The viewer's first *actively driven* model, and the reason the energy panel has a third verdict:
//! a bow pumps energy in through a nonlinear friction curve, so neither "conserved" nor "decays" is
//! true of it. What replaces them is the balance `E - E0 == bow_work - loss` (see
//! [`crate::energy::balance_verdict`]).
//!
//! The second panel is the stick-slip trace, and its oracle — the slip fraction equals `beta`, the
//! bow's fractional position — is **claimed only inside Schelleng's window**. Outside it the motion
//! legitimately stops being Helmholtz (it crushes, or goes raucous), which is real physics, so the
//! panel labels the regime instead of scoring it.
//!
//! One run, not two: the animation shows the *settled* motion, which the audio run already passes
//! through, so the frames are captured from its tail. A second run would double the cost of a model
//! whose every step is a friction root-find, and the work budget would not see it.

use physsynth_analysis::{modal, spectrum};
use physsynth_core::bow::BowedString;
use physsynth_core::engine::SimResult;
use serde_json::{json, Map, Value};

use crate::energy::{energy_block, EnergyOpts};
use crate::horizon::string_block;
use crate::py::{commas, finite_list, int, np_mean, num, py_int, py_repr, round_int, round_nd};
use crate::string::{
    build_resonator, damped_info, int_at_least_one, Frames, ANIM_WIN_MAX, FRAMES_PER_PERIOD,
    MAX_FRAMES, SPEED_MAX,
};
use crate::{fnum, resample_normalize, Refusal, AUDIO_FS};

/// `beta = x_bow / L`; the slip fraction of the period is ~ beta.
pub const BOW_POSITION_DEFAULT: f64 = 0.13;
/// Bow speed, m/s.
pub const BOW_V_DEFAULT: f64 = 0.1;
/// Peak of the friction curve, N.
pub const BOW_FORCE_DEFAULT: f64 = 1.0;
/// Friction-curve sharpness, s²/m².
pub const BOW_SHARPNESS_DEFAULT: f64 = 60.0;
/// Loss is ON by default: it lets the note settle to a limit cycle instead of growing forever.
pub const BOW_SIGMA0_DEFAULT: f64 = 0.5;
/// High-partial damping, which keeps the corner clean (one slip per period).
pub const BOW_SIGMA1_DEFAULT: f64 = 0.05;
/// Force ceiling.
pub const BOW_FORCE_MAX: f64 = 5.0;
/// Bow-speed ceiling.
pub const BOW_V_MAX: f64 = 1.0;
/// Sharpness ceiling.
pub const BOW_SHARPNESS_MAX: f64 = 200.0;
/// Lossless balance bar (the retired `tests/test_bow_energy.py`'s; now `bow_harness.rs`'s).
pub const BOW_BALANCE_TOL: f64 = crate::energy::BOW_BALANCE_TOL;
/// Clean one-slip-per-period window for the stick-slip verdict.
pub const BOW_SLIPS_LO: f64 = 0.85;
/// Clean one-slip-per-period window for the stick-slip verdict.
pub const BOW_SLIPS_HI: f64 = 1.25;
/// `|slip_fraction - beta|` bar.
pub const BOW_SLIP_MATCH_TOL: f64 = 0.05;
/// The statistics tail: the settled last 40 % of the run (a rate needs a long window).
pub const BOW_TAIL_FRAC: f64 = 0.4;
/// Periods drawn in the trace — a plot, not the statistics window.
pub const BOW_TRACE_PERIODS: f64 = 3.0;
/// Grid ceiling: every step is a friction root-find.
pub const BOW_N_MAX: i64 = 256;
/// Longest audio run, seconds.
pub const BOW_AUDIO_MAX: f64 = 3.0;
/// Total steps.
pub const BOW_WORK_MAX: i64 = 60_000;

fn pf(x: f64) -> String {
    physsynth_core::fmt::py_float(x)
}

/// `_run_bow`: step the bow, also recording its cumulative work and `v_rel` every step.
///
/// Frames are captured from `snapshot_from` on, every `snapshot_stride` steps after it.
pub fn run_bow(
    bow: &mut BowedString,
    num_steps: usize,
    pickup_index: Option<usize>,
    snapshot_stride: usize,
    snapshot_from: usize,
) -> Result<(SimResult, Vec<f64>, Vec<f64>), Refusal> {
    if num_steps < 1 {
        return Err(Refusal::Construction("num_steps must be >= 1.".into()));
    }
    let n = num_steps + 1;
    let mut energy = Vec::with_capacity(n);
    let mut work = Vec::with_capacity(n);
    let mut v_rel = Vec::with_capacity(n);
    let mut output = pickup_index.map(|_| Vec::with_capacity(n));
    let mut snapshots = Vec::new();
    let snap = |i: usize| {
        snapshot_stride != 0 && i >= snapshot_from && (i - snapshot_from) % snapshot_stride == 0
    };

    energy.push(bow.energy());
    work.push(bow.s.bow_work);
    v_rel.push(bow.s.v_rel);
    if let (Some(o), Some(j)) = (output.as_mut(), pickup_index) {
        o.push(bow.string.u[j]);
    }
    if snap(0) {
        snapshots.push((0, bow.string.u.clone()));
    }
    for i in 1..n {
        bow.step().map_err(|e| Refusal::Internal(e.to_string()))?;
        energy.push(bow.energy());
        work.push(bow.s.bow_work);
        v_rel.push(bow.s.v_rel);
        if let (Some(o), Some(j)) = (output.as_mut(), pickup_index) {
            o.push(bow.string.u[j]);
        }
        if snap(i) {
            snapshots.push((i, bow.string.u.clone()));
        }
    }
    let k = bow.p.k;
    Ok((
        SimResult {
            time: (0..n).map(|i| i as f64 * k).collect(),
            energy,
            output,
            fs: 1.0 / k,
            snapshots,
        },
        work,
        v_rel,
    ))
}

/// `_bow_stickslip_block`: the `v_rel` trace, and `slip_fraction == beta` inside the window only.
pub fn stickslip_block(
    v_rel: &[f64],
    pickup: &[f64],
    fs: f64,
    v_bow: f64,
    beta: f64,
    f1: f64,
) -> Value {
    let len = v_rel.len();
    let n_tail = (round_int(BOW_TAIL_FRAC * len as f64).max(4) as usize).min(len);
    let tail = &v_rel[len - n_tail..];
    let slipping: Vec<bool> = tail.iter().map(|v| v.abs() >= 0.5 * v_bow).collect();
    let slip_fraction = slipping.iter().filter(|&&s| s).count() as f64 / slipping.len() as f64;
    let onsets = slipping.windows(2).filter(|w| !w[0] && w[1]).count();
    let n_periods = tail.len() as f64 * f1 / fs;
    let slips_per_period = if n_periods > 0.0 {
        onsets as f64 / n_periods
    } else {
        0.0
    };
    let helmholtz = BOW_SLIPS_LO < slips_per_period && slips_per_period < BOW_SLIPS_HI;

    let n_show = (round_int(BOW_TRACE_PERIODS * fs / f1).max(8) as usize).min(tail.len());
    let shown = &tail[tail.len() - n_show..];
    let step = (shown.len() / crate::energy::N_ENERGY_POINTS).max(1);
    let trace: Vec<f64> = shown.iter().step_by(step).copied().collect();

    let mut block = Map::new();
    block.insert("kind".into(), json!("bow"));
    block.insert("v_rel".into(), finite_list(&trace, None));
    block.insert("dt".into(), num(step as f64 / fs));
    block.insert("v_bow".into(), num(v_bow));
    block.insert("stick_threshold".into(), num(0.5 * v_bow));
    block.insert("beta".into(), num(beta));
    block.insert("slip_fraction".into(), num(round_nd(slip_fraction, 4)));
    block.insert(
        "slips_per_period".into(),
        num(round_nd(slips_per_period, 3)),
    );
    block.insert("helmholtz".into(), Value::Bool(helmholtz));
    block.insert("slip_tol".into(), num(BOW_SLIP_MATCH_TOL));
    if helmholtz {
        block.insert(
            "slip_matches_beta".into(),
            Value::Bool((slip_fraction - beta).abs() < BOW_SLIP_MATCH_TOL),
        );
        block.insert("slip_error".into(), num(round_nd(slip_fraction - beta, 4)));
        // The bow does NOT set the pitch — the string does.
        let seg = &pickup[pickup.len() - n_tail..];
        let mean = np_mean(seg);
        let sig: Vec<f64> = seg.iter().map(|v| v - mean).collect();
        let detected = spectrum::measure_partials_near(&sig, fs, &[f1], Some(0.15 * f1))[0];
        block.insert(
            "f_detected".into(),
            if detected.is_finite() {
                num(round_nd(detected, 3))
            } else {
                Value::Null
            },
        );
        block.insert("f1".into(), num(round_nd(f1, 3)));
        block.insert(
            "pitch_cents".into(),
            if detected.is_finite() && detected > 0.0 {
                num(round_nd(1200.0 * (detected / f1).log2(), 2))
            } else {
                Value::Null
            },
        );
    } else {
        block.insert("slip_matches_beta".into(), Value::Null);
        let never = slip_fraction > 0.95;
        block.insert(
            "regime".into(),
            json!(if never { "never_sticks" } else { "multi_slip" }),
        );
        let what = if never {
            "the string never sticks to the bow (slip fraction ≈ 1), so no stick-slip cycle forms"
                .to_owned()
        } else {
            format!("{slips_per_period:.1} slips per period (clean bowing is 1)")
        };
        block.insert(
            "note".into(),
            json!(format!(
                "outside the Helmholtz window — {what}. Real physics, not a solver failure: \
                 Schelleng's playable force window has a floor and a ceiling, and both narrow as \
                 the bow moves off the bridge. The slip = beta oracle only describes one-slip \
                 motion, so it is not scored here."
            )),
        );
    }
    Value::Object(block)
}

/// `_build_payload_bow`.
pub fn build_payload(p: &Value) -> Result<Value, Refusal> {
    let audio_dur = fnum(p, "audio_duration", 2.0)?;
    let anim_win = fnum(p, "animation_window", 0.06)?;
    let playback_speed = fnum(p, "playback_speed", 0.02)?;
    let pickup_frac = fnum(p, "pickup_position", 0.33)?;
    let fpp = int_at_least_one(fnum(p, "frames_per_period", FRAMES_PER_PERIOD)?)?;

    if !(0.0 < audio_dur && audio_dur <= BOW_AUDIO_MAX) {
        return Err(Refusal::Param(format!(
            "audio_duration must be in (0, {}] s, got {}.",
            pf(BOW_AUDIO_MAX),
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
    if !(0.0 < pickup_frac && pickup_frac < 1.0) {
        return Err(Refusal::Param(format!(
            "pickup_position must be in (0, 1), got {}.",
            pf(pickup_frac)
        )));
    }
    // `int(p.get("N", 100))` — the pre-check's default is 100 where the constructor's is 128. The
    // reference's own inconsistency, kept: with N absent the check passes either way.
    let n_val = p.get("N").cloned().unwrap_or(json!(100));
    let n_req = py_int(&n_val)
        .map_err(|_| Refusal::Param(format!("N must be an integer, got {}.", py_repr(&n_val))))?;
    if n_req > BOW_N_MAX {
        return Err(Refusal::Param(format!(
            "N must be <= {BOW_N_MAX} for the bowed string (got {n_req}): every step runs a \
             friction root-find, so a fine grid is far costlier here than on a linear string."
        )));
    }

    let b = build_resonator(p)?;
    let (c, l, n, fs) = (b.c, b.l, b.n, b.fs);
    let (sigma_zero, oracle_2sigma) = (b.sigma_zero, b.oracle_2sigma);
    let mut bow = b.res.into_bow();
    let pickup_idx = round_int(pickup_frac * n as f64).max(1).min(n - 1) as usize;
    // The string's own fundamental — the pitch the bow locks to.
    let kappa = fnum(p, "kappa", 0.0)?;
    let f1 = modal::stiff_harmonic_frequencies(c, l, kappa, 1)[0];

    let n_audio = round_int(audio_dur * fs).max(1);
    if n_audio > BOW_WORK_MAX {
        return Err(Refusal::Param(format!(
            "work budget exceeded ({} steps > {}): every step runs a friction root-find. Lower N \
             or audio_duration.",
            commas(n_audio),
            commas(BOW_WORK_MAX)
        )));
    }
    let mut anim_stride = round_int((fs / f1) / fpp as f64).max(1);
    let n_anim = n_audio.min(anim_stride.max(round_int(anim_win * fs)));
    if n_anim / anim_stride > MAX_FRAMES {
        anim_stride = ((n_anim as f64 / MAX_FRAMES as f64).ceil() as i64).max(1);
    }
    // Animate the SETTLED motion: the bow starts from rest, so the opening frames are near-flat.
    let n_settle = (n_audio - n_anim).max(0) as usize;
    let (audio_run, work, v_rel) = run_bow(
        &mut bow,
        n_audio as usize,
        Some(pickup_idx),
        anim_stride as usize,
        n_settle,
    )?;
    let pickup = audio_run.output.as_deref().expect("a pickup was requested");
    if !pickup.iter().all(|v| v.is_finite()) {
        return Err(Refusal::Param(
            "simulation produced non-finite output (instability) — adjust parameters.".into(),
        ));
    }
    // Relative to the window start, so the animation clock begins at 0 like every other model's.
    let frames = Frames::of(&audio_run.snapshots, n_settle);
    let (audio48, peak) = resample_normalize(pickup, fs);

    let energy = energy_block(
        &audio_run.time,
        &audio_run.energy,
        sigma_zero,
        oracle_2sigma,
        EnergyOpts {
            balance_work: Some(&work),
            ..EnergyOpts::default()
        },
    );
    Ok(json!({
        "model": "bow",
        "horizon": string_block(&damped_info(&bow.string), "the bowed string"),
        "fs_sim": num(round_nd(fs, 3)),
        "lambda": num(round_nd(bow.string.p.lam, 6)),
        "grid": {"x": finite_list(&bow.string.p.grid(), Some(6))},
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
            "f1": num(round_nd(f1, 3)),
            "num_steps": int(n_audio),
            "n_frames": int(frames.n as i64),
            "spectrum": stickslip_block(&v_rel, pickup, fs, bow.p.v_bow, bow.beta, f1),
            "bow_x": num(round_nd(bow.x_bow, 4)),
            "beta": num(round_nd(bow.beta, 4)),
            // Reported, never asserted: above 1 the friction curve is multivalued, which is the
            // regime of real sustained bowing, not a stability limit.
            "helmholtz_number": num(round_nd(bow.p.helmholtz_number, 3)),
            "fallbacks": int(bow.s.fallbacks as i64),
        },
    }))
}
