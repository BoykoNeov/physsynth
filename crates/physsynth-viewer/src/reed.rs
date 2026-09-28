//! The dynamic single reed on the bore — `serialize.py`'s `_build_payload_reed` and its panels.
//!
//! A self-oscillating exciter, the acoustic dual of the bow. Its energy panel is the balance with
//! **measured** loss channels (`jet`, `reed damping`), so the residual can genuinely fail in every
//! regime (see [`crate::energy::balance_verdict`]). Its second panel is the odd-harmonic signature
//! off the mouthpiece with the far-field caveat measured beside it, plus a blowing-threshold sweep
//! and a pitch-leverage pair run at a FIXED `N`, so the headline threshold does not move when the
//! user drags `N`.
//!
//! The reference memoized the sweep (`_REED_SWEEP_MEMO`) because in Python it cost ~3.5 s. Here it
//! is well under a second, so it is recomputed every time: the memo's key was a documented trap
//! (a key missing an input returns stale numbers), and a cache that is not needed cannot be keyed
//! wrong.

use std::f64::consts::PI;

use physsynth_analysis::spectrum;
use physsynth_core::bore::{self as bore, Bore, End};
use physsynth_core::fmt::{py_exp, py_float};
use physsynth_core::pyfloat::scalar_pow;
use physsynth_core::reed::{self as reed, ReedBore};
use serde_json::{json, Map, Value};

use crate::energy::{energy_block, EnergyOpts};
use crate::py::{
    b64f32, finite_list, float_to_int, int, max_abs, np_mean, num, py_str, repr_str, round_int,
    round_nd,
};
use crate::string::{construction, int_at_least_one, FRAMES_PER_PERIOD, MAX_FRAMES, SPEED_MAX};
use crate::{fnum, resample_normalize, Refusal, AUDIO_FS};

/// Air density (kg/m³) — `physsynth.core.bore.RHO0_AIR`.
pub const RHO0_AIR: f64 = 1.2041;
/// Speed of sound (m/s) — `physsynth.core.bore.C0_AIR`.
pub const C0_AIR: f64 = 343.0;
/// The bell's `R/Z0` range (shared with the bore scene).
pub const BORE_R_RATIO_MIN: f64 = 1e-4;
/// The bell's `R/Z0` range (shared with the bore scene).
pub const BORE_R_RATIO_MAX: f64 = 30.0;

/// The reed's `N` range.
pub const REED_N_MIN: i64 = 32;
/// The reed's `N` range.
pub const REED_N_MAX: i64 = 256;
/// Longest audio run, seconds.
pub const REED_AUDIO_MAX: f64 = 1.0;
/// Longest animation window, seconds.
pub const REED_ANIM_MAX: f64 = 0.1;
/// Bounds the RENDER only; the sweeps are separate.
pub const REED_WORK_MAX: i64 = 300_000;
/// Pinned: at `lambda = 1`, `fs = c0 N / L`.
pub const REED_LAMBDA: f64 = 1.0;
/// Bore radius (m).
pub const REED_RADIUS: f64 = 0.008;
/// Pinned: the balance is linear in the residual, so a loose solve degrades the headline.
pub const REED_NEWTON_TOL: f64 = 1e-10;
/// The settled tail = the last 40 % of a run.
pub const REED_SETTLE_FRAC: f64 = 0.4;
/// The sweeps' fixed grid (the threshold is N-invariant to the 4th digit).
pub const REED_SWEEP_N: usize = 64;
/// Seconds per sweep point: near the onset the reed critically slows down.
pub const REED_SWEEP_SECS: f64 = 0.8;
/// The sweep's `gamma` points; `1/3` is on the grid EXACTLY.
pub const REED_SWEEP_GAMMAS: [f64; 8] = [0.20, 0.30, 1.0 / 3.0, 0.355, 0.372, 0.45, 0.51, 0.61];
/// AC rms / `p_closing` above which the reed speaks.
pub const REED_SPEAK_GATE: f64 = 0.02;
/// Seconds per pitch-leverage run.
pub const REED_PITCH_SECS: f64 = 0.6;
/// The two reed frequencies of the pitch-leverage pair.
pub const REED_PITCH_FREEDS: [f64; 2] = [2000.0, 3000.0];

/// What `_build_reed` reports beside the reed.
pub struct Info {
    pub l: f64,
    pub n: usize,
    pub fs: f64,
    pub lam: f64,
    pub z0: f64,
    pub gamma: f64,
    pub f_reed: f64,
    pub q_reed: f64,
    pub p_mouth: f64,
    pub p_closing: f64,
    pub h0: f64,
    pub ratio: f64,
    pub radiating: bool,
    pub r_bell: f64,
}

/// A `Bore` with the reference's defaults for `rho0` and `c0`.
fn make_bore(l: f64, fs: f64, n: usize, right: End, r_bell: f64) -> Result<Bore, Refusal> {
    let p = bore::Params::new(
        l,
        fs,
        n,
        REED_RADIUS,
        Some((End::Closed, right)),
        0.0,
        r_bell,
        RHO0_AIR,
        C0_AIR,
    )
    .map_err(construction)?;
    Ok(Bore::new(p))
}

/// `_reed_at_gamma`: a `ReedBore` blown at `gamma`, `p_mouth` derived from its own `p_closing`.
fn reed_at_gamma(bore: Bore, gamma: f64, f_reed: f64, q_reed: f64) -> Result<ReedBore, Refusal> {
    let (mu, h0) = (0.03, 4.0e-4);
    let p_closing = mu * scalar_pow(2.0 * PI * f_reed, 2.0) * h0;
    // `ReedBore`'s own defaults: Sr = 1.5e-4, width = 1.5e-2, newton_maxiter = 60.
    let params = reed::Params::new(
        bore.params(),
        f_reed,
        q_reed,
        mu,
        1.5e-4,
        1.5e-2,
        h0,
        REED_NEWTON_TOL,
        60,
    )
    .map_err(construction)?;
    Ok(ReedBore::new(params, bore, gamma * p_closing))
}

fn pf(x: f64) -> String {
    py_float(x)
}

/// `_build_reed`.
fn build(p: &Value) -> Result<(ReedBore, Info), Refusal> {
    let l = fnum(p, "L", 0.5)?;
    let n = float_to_int(fnum(p, "N", 128.0)?).map_err(|e| match e {
        crate::py::IntError::NaN => {
            Refusal::Construction("cannot convert float NaN to integer".into())
        }
        _ => Refusal::Construction("cannot convert float infinity to integer".into()),
    })?;
    let gamma = fnum(p, "gamma", 0.51)?;
    let f_reed = fnum(p, "f_reed", 2500.0)?;
    let q_reed = fnum(p, "q_reed", 4.0)?;
    let ratio_exp = fnum(p, "bell_ratio_exp", -2.6)?;
    let domain = p
        .get("domain")
        .map_or_else(|| "radiating".to_owned(), py_str);

    if domain != "radiating" && domain != "open" {
        return Err(Refusal::Param(format!(
            "reed bore end must be 'radiating' or 'open', got {}.",
            repr_str(&domain)
        )));
    }
    if !(0.1..=2.0).contains(&l) {
        return Err(Refusal::Param(format!(
            "L must be in [0.1, 2.0] m, got {}.",
            pf(l)
        )));
    }
    if !(REED_N_MIN..=REED_N_MAX).contains(&n) {
        return Err(Refusal::Param(format!(
            "N must be in [{REED_N_MIN}, {REED_N_MAX}] for the reed, got {n}."
        )));
    }
    if !(0.02..=1.2).contains(&gamma) {
        return Err(Refusal::Param(format!(
            "gamma = p_mouth/p_closing must be in [0.02, 1.2], got {}. The note speaks above \
             gamma ~ 0.36; at gamma >= 1 the reed is held statically shut.",
            pf(gamma)
        )));
    }
    if !(800.0..=4000.0).contains(&f_reed) {
        return Err(Refusal::Param(format!(
            "f_reed must be in [800, 4000] Hz, got {}.",
            pf(f_reed)
        )));
    }
    if !(0.5..=20.0).contains(&q_reed) {
        return Err(Refusal::Param(format!(
            "q_reed must be in [0.5, 20], got {}.",
            pf(q_reed)
        )));
    }
    let ratio = scalar_pow(10.0, ratio_exp);
    if !(BORE_R_RATIO_MIN..=BORE_R_RATIO_MAX).contains(&ratio) {
        return Err(Refusal::Param(format!(
            "the bell's R/Z0 must be in [{}, {}], got {}.",
            pf(BORE_R_RATIO_MIN),
            pf(BORE_R_RATIO_MAX),
            py_exp(ratio, 3)
        )));
    }

    let radiating = domain == "radiating";
    let n = n as usize;
    let z0 = RHO0_AIR * C0_AIR / (PI * REED_RADIUS * REED_RADIUS);
    let fs = C0_AIR / (REED_LAMBDA * (l / n as f64));
    let right = if radiating { End::Radiating } else { End::Open };
    let bore = make_bore(l, fs, n, right, if radiating { ratio * z0 } else { 0.0 })?;
    let (lam, r_bell) = (bore.params().lam, bore.params().r_bell);
    let rb = reed_at_gamma(bore, gamma, f_reed, q_reed)?;
    let info = Info {
        l,
        n,
        fs,
        lam,
        z0,
        gamma,
        f_reed,
        q_reed,
        p_mouth: rb.state().p_mouth,
        p_closing: rb.params().p_closing,
        h0: rb.params().h0,
        ratio,
        radiating,
        r_bell,
    };
    Ok((rb, info))
}

/// Per-step telemetry: every balance channel, the opening, and the field.
struct Run {
    e: Vec<f64>,
    e_rad: Vec<f64>,
    mouth: Vec<f64>,
    jet: Vec<f64>,
    damp: Vec<f64>,
    pickup: Vec<f64>,
    far: Vec<f64>,
    opening: Vec<f64>,
    env: Vec<f64>,
    frames: Vec<Vec<f64>>,
    frame_steps: Vec<usize>,
    frame_open: Vec<f64>,
    frame_rad: Vec<f64>,
}

/// `_run_reed`: one run, the animation window captured out of its TAIL.
fn run(
    rb: &mut ReedBore,
    n_steps: usize,
    anim_stride: usize,
    frame_from: usize,
) -> Result<Run, Refusal> {
    let width = rb.bore().params().nodes();
    let mut r = Run {
        e: Vec::with_capacity(n_steps + 1),
        e_rad: Vec::with_capacity(n_steps + 1),
        mouth: Vec::with_capacity(n_steps + 1),
        jet: Vec::with_capacity(n_steps + 1),
        damp: Vec::with_capacity(n_steps + 1),
        pickup: Vec::with_capacity(n_steps + 1),
        far: Vec::with_capacity(n_steps + 1),
        opening: Vec::with_capacity(n_steps + 1),
        env: vec![0.0; width],
        frames: Vec::new(),
        frame_steps: Vec::new(),
        frame_open: Vec::new(),
        frame_rad: Vec::new(),
    };
    let sample = |rb: &ReedBore, r: &mut Run, i: usize| {
        let s = rb.state();
        r.e.push(rb.energy());
        r.e_rad.push(rb.bore().radiated_energy());
        r.mouth.push(s.mouth_work);
        r.jet.push(s.jet_loss);
        r.damp.push(s.reed_damp_work);
        r.pickup.push(rb.mouthpiece_pressure());
        r.far.push(rb.bore().radiated_pressure());
        r.opening.push(s.reed_opening(rb.params()));
        if i >= frame_from {
            for (e, p) in r.env.iter_mut().zip(rb.bore().p()) {
                // `np.maximum`: NaN-propagating.
                let a = p.abs();
                if a > *e || a.is_nan() {
                    *e = a;
                }
            }
        }
    };
    sample(rb, &mut r, 0);
    for i in 1..=n_steps {
        rb.step().map_err(|e| Refusal::Internal(format!("{e:?}")))?;
        sample(rb, &mut r, i);
        if i >= frame_from && (i - frame_from) % anim_stride == 0 {
            r.frames.push(rb.bore().p().to_vec());
            r.frame_steps.push(i);
            r.frame_open.push(r.opening[i]);
            r.frame_rad.push(r.e_rad[i]);
        }
    }
    if !r.e.iter().all(|v| v.is_finite()) {
        return Err(Refusal::Param(
            "simulation produced non-finite energy (instability) — adjust parameters.".into(),
        ));
    }
    Ok(r)
}

/// `sqrt(mean((x - mean(x))**2))` — the AC rms.
fn ac_rms(x: &[f64]) -> f64 {
    let m = np_mean(x);
    let sq: Vec<f64> = x.iter().map(|v| (v - m) * (v - m)).collect();
    np_mean(&sq).sqrt()
}

/// `_reed_ac_level`: the settled-tail AC rms of the mouthpiece pressure over `p_closing`.
fn ac_level(rb: &mut ReedBore, secs: f64) -> Result<f64, Refusal> {
    let n = ((secs / rb.params().k) as usize).max(2);
    let mut sig = Vec::with_capacity(n);
    for _ in 0..n {
        rb.step().map_err(|e| Refusal::Internal(format!("{e:?}")))?;
        sig.push(rb.mouthpiece_pressure());
    }
    let from = ((1.0 - REED_SETTLE_FRAC) * n as f64) as usize;
    Ok(ac_rms(&sig[from..]) / rb.params().p_closing)
}

/// `_reed_sweep_block`: the blowing threshold and the pitch leverage, both at a FIXED N.
fn sweep_block(info: &Info) -> Result<Value, Refusal> {
    let right = if info.radiating {
        End::Radiating
    } else {
        End::Open
    };
    let fs = C0_AIR / (info.l / REED_SWEEP_N as f64);
    let fresh = |gamma: f64, fr: f64| -> Result<ReedBore, Refusal> {
        let b = make_bore(info.l, fs, REED_SWEEP_N, right, info.r_bell)?;
        reed_at_gamma(b, gamma, fr, info.q_reed)
    };
    let mut levels = Vec::with_capacity(REED_SWEEP_GAMMAS.len());
    for g in REED_SWEEP_GAMMAS {
        let mut rb = fresh(g, info.f_reed)?;
        levels.push(ac_level(&mut rb, REED_SWEEP_SECS)?);
    }
    let speaks: Vec<bool> = levels.iter().map(|&v| v > REED_SPEAK_GATE).collect();
    let (mut lo, mut hi) = (None, None);
    for (g, &sp) in REED_SWEEP_GAMMAS.iter().zip(&speaks) {
        if !sp {
            lo = Some(*g);
        } else if hi.is_none() {
            hi = Some(*g);
        }
    }

    let f1 = C0_AIR / (4.0 * info.l);
    let mut pitches = Vec::with_capacity(2);
    for fr in REED_PITCH_FREEDS {
        let mut r = fresh(0.51, fr)?;
        let k = r.params().k;
        let n = ((REED_PITCH_SECS / k) as usize).max(4);
        let mut sig = Vec::with_capacity(n);
        for _ in 0..n {
            r.step().map_err(|e| Refusal::Internal(format!("{e:?}")))?;
            sig.push(r.bore().displacement_at(1));
        }
        let tail = &sig[n / 2..];
        let m = np_mean(tail);
        let centred: Vec<f64> = tail.iter().map(|v| v - m).collect();
        pitches.push(spectrum::measure_partials_near(&centred, 1.0 / k, &[f1], None)[0]);
    }
    let leverage = if pitches.iter().all(|p| p.is_finite()) && pitches[0] > 0.0 {
        json!({
            "f_reed": REED_PITCH_FREEDS.iter().map(|&f| num(round_nd(f, 1))).collect::<Vec<_>>(),
            "reed_change_pct": num(round_nd(
                100.0 * (REED_PITCH_FREEDS[1] / REED_PITCH_FREEDS[0] - 1.0), 1)),
            "f0": pitches.iter().map(|&f| num(round_nd(f, 3))).collect::<Vec<_>>(),
            "pitch_change_pct": num(round_nd(100.0 * (pitches[1] / pitches[0] - 1.0), 2)),
            "cents": pitches
                .iter()
                .map(|&f| num(round_nd(1200.0 * (f / f1).log2(), 1)))
                .collect::<Vec<_>>(),
        })
    } else {
        Value::Null
    };
    Ok(json!({
        "sweep_N": int(REED_SWEEP_N as i64),
        "sweep_secs": num(REED_SWEEP_SECS),
        "gamma": REED_SWEEP_GAMMAS.iter().map(|&g| num(g)).collect::<Vec<_>>(),
        "level": levels.iter().map(|&v| num(round_nd(v, 6))).collect::<Vec<_>>(),
        "speaks": speaks,
        "gate": num(REED_SPEAK_GATE),
        "bracket": [lo.map_or(Value::Null, num), hi.map_or(Value::Null, num)],
        "pitch": leverage,
        "f1": num(round_nd(f1, 3)),
        "r_ratio": if info.radiating { num(info.ratio) } else { Value::Null },
    }))
}

/// `_reed_beating_block`: closure statistics over the settled tail — duty first.
fn beating_block(opening: &[f64], fs: f64, f1: f64) -> Value {
    let from = ((1.0 - REED_SETTLE_FRAC) * opening.len() as f64) as usize;
    let tail = &opening[from..];
    if tail.len() < 4 || f1 <= 0.0 {
        return json!({"beats": false, "duty": num(0.0), "per_period": null});
    }
    let period = round_int(fs / f1).max(1) as usize;
    let shut: Vec<bool> = tail.iter().map(|&v| v <= 0.0).collect();
    let duty = shut.iter().filter(|&&s| s).count() as f64 / shut.len() as f64;
    let gap = ((0.10 * period as f64) as usize).max(1);
    let (mut episodes, mut i) = (0usize, 0usize);
    while i < shut.len() {
        if shut[i] {
            episodes += 1;
            let mut j = i;
            while j < shut.len() {
                if shut[j] {
                    j += 1;
                    continue;
                }
                let mut run_open = 0;
                while j + run_open < shut.len() && !shut[j + run_open] {
                    run_open += 1;
                }
                if run_open >= gap {
                    break;
                }
                j += run_open;
            }
            i = j;
        } else {
            i += 1;
        }
    }
    let periods = tail.len() as f64 / period as f64;
    let min_open = tail.iter().copied().fold(f64::INFINITY, f64::min);
    json!({
        "beats": duty > 0.0,
        "duty": num(round_nd(duty, 4)),
        "per_period": if periods > 0.0 {
            num(round_nd(episodes as f64 / periods, 3))
        } else {
            Value::Null
        },
        "min_opening": num(min_open),
    })
}

/// `_reed_signature_block`: odd-harmonic dominance, with the far-field caveat beside it.
fn signature_block(pickup: &[f64], far: &[f64], fs: f64, f1: f64, spoke: bool) -> Value {
    let mut out = Map::new();
    out.insert("kind".into(), json!("reed"));
    out.insert("applies".into(), Value::Bool(spoke));
    if !spoke {
        out.insert("note".into(), json!("below threshold — no tone to analyse"));
        return Value::Object(out);
    }
    let tail = &pickup[pickup.len() / 2..];
    let m = np_mean(tail);
    let ac: Vec<f64> = tail.iter().map(|v| v - m).collect();
    let s = spectrum::magnitude_spectrum(&ac, fs, 2);
    let df = s.freqs[1] - s.freqs[0];
    let peak = |f: f64| -> f64 {
        let i = round_int(f / df);
        let lo = (i - 3).max(1) as usize;
        let hi = ((i + 4).max(0) as usize).min(s.mag.len());
        s.mag[lo..hi]
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max)
    };
    let n_show = ((6.5 * f1 / df) as usize).max(8).min(s.freqs.len());
    out.insert("freq".into(), finite_list(&s.freqs[1..n_show], Some(3)));
    out.insert("mag".into(), finite_list(&s.mag[1..n_show], Some(6)));
    out.insert(
        "markers".into(),
        json!((1..=5)
            .map(|k| num(round_nd(k as f64 * f1, 3)))
            .collect::<Vec<_>>()),
    );
    out.insert(
        "odd_even".into(),
        num(round_nd(peak(f1) / peak(2.0 * f1).max(1e-30), 1)),
    );
    out.insert(
        "third_second".into(),
        num(round_nd(peak(3.0 * f1) / peak(2.0 * f1).max(1e-30), 2)),
    );
    let sq: Vec<f64> = ac.iter().map(|v| v * v).collect();
    let rms = np_mean(&sq).sqrt();
    let ac_peak = max_abs(&ac);
    out.insert(
        "crest".into(),
        if rms > 0.0 {
            num(round_nd(ac_peak / rms, 3))
        } else {
            Value::Null
        },
    );
    let ftail = &far[far.len() / 2..];
    let fm = np_mean(ftail);
    let fac: Vec<f64> = ftail.iter().map(|v| v - fm).collect();
    let fsq: Vec<f64> = fac.iter().map(|v| v * v).collect();
    let frms = np_mean(&fsq).sqrt();
    let fpeak = max_abs(&fac);
    if frms > 0.0 && fpeak > 1e-12 * ac_peak.max(1e-30) {
        out.insert(
            "far_field".into(),
            json!({
                "peak": num(round_nd(max_abs(ftail), 4)),
                "crest": num(round_nd(fpeak / frms, 3)),
                "quieter_by": num(round_nd(ac_peak / fpeak, 1)),
            }),
        );
    } else {
        out.insert("far_field".into(), Value::Null);
        out.insert(
            "far_note".into(),
            json!(
                "an ideal open end radiates nothing — there is no far field to compare the \
                 mouthpiece with"
            ),
        );
    }
    Value::Object(out)
}

/// `_build_payload_reed`.
pub fn build_payload(p: &Value) -> Result<Value, Refusal> {
    let playback_speed = fnum(p, "playback_speed", 0.02)?;
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
    if !(0.0 < audio_dur && audio_dur <= REED_AUDIO_MAX) {
        return Err(Refusal::Param(format!(
            "audio_duration must be in (0, {}] s, got {}.",
            pf(REED_AUDIO_MAX),
            pf(audio_dur)
        )));
    }
    if !(0.0 < anim_win && anim_win <= REED_ANIM_MAX) {
        return Err(Refusal::Param(format!(
            "animation_window must be in (0, {}] s, got {}.",
            pf(REED_ANIM_MAX),
            pf(anim_win)
        )));
    }

    let (mut rb, info) = build(p)?;
    let (fs, l) = (info.fs, info.l);
    let f1 = C0_AIR / (4.0 * l);
    let n_steps = round_int(audio_dur * fs).max(1);
    if n_steps > REED_WORK_MAX {
        return Err(Refusal::Param(format!(
            "this render needs {n_steps} steps (budget {REED_WORK_MAX}). At lambda = 1, fs = \
             c0*N/L — so N buys the sample rate, not just the grid. Shorten audio_duration or \
             lower N."
        )));
    }
    // Pace on the TRANSIT, not f1: f1 aliases the travelling pressure step.
    let mut anim_stride = round_int((fs / (C0_AIR / l)) / fpp as f64).max(1);
    let n_anim = n_steps.min(anim_stride.max(round_int(anim_win * fs)));
    if n_anim / anim_stride > MAX_FRAMES {
        anim_stride = ((n_anim as f64 / MAX_FRAMES as f64).ceil() as i64).max(1);
    }
    let frame_from = (n_steps - n_anim).max(0) as usize;
    let r = run(&mut rb, n_steps as usize, anim_stride as usize, frame_from)?;

    let flat: Vec<f64> = r.frames.iter().flatten().copied().collect();
    let n_frames = r.frames.len();
    let field_amp = if flat.is_empty() { 0.0 } else { max_abs(&flat) };
    let (audio48, peak) = resample_normalize(&r.pickup, fs);
    let time: Vec<f64> = (0..r.e.len()).map(|i| i as f64 / fs).collect();

    let tail0 = ((1.0 - REED_SETTLE_FRAC) * r.pickup.len() as f64) as usize;
    let ac_level = ac_rms(&r.pickup[tail0..]) / info.p_closing;
    let spoke = ac_level > REED_SPEAK_GATE;
    let stored: Vec<f64> = r.e.iter().zip(&r.e_rad).map(|(e, x)| e - x).collect();
    let last = |v: &[f64]| v[v.len() - 1];
    let breath = if last(&r.mouth) == 0.0 {
        1.0
    } else {
        last(&r.mouth)
    };
    let e_last = if last(&r.e) == 0.0 { 1.0 } else { last(&r.e) };
    let times: Vec<f64> = r.frame_steps.iter().map(|&i| i as f64 / fs).collect();
    let rad_frames: Vec<f64> = r.frame_rad.iter().map(|v| v / e_last).collect();
    let env_last = last(&r.env);

    let energy = energy_block(
        &time,
        &r.e,
        true,
        0.0,
        EnergyOpts {
            balance_work: Some(&r.mouth),
            measured_loss: Some(vec![("jet", &r.jet), ("reed damping", &r.damp)]),
            ..EnergyOpts::default()
        },
    );
    let sweep = sweep_block(&info)?;
    Ok(json!({
        "model": "reed",
        "fs_sim": num(round_nd(fs, 3)),
        "lambda": num(round_nd(info.lam, 6)),
        "grid": {
            "x": finite_list(&rb.bore().params().grid(), Some(6)),
            "envelope": finite_list(&r.env, None),
        },
        "frames": {
            "b64": b64f32(&flat),
            "n_frames": int(n_frames as i64),
            "width": int(if n_frames > 0 { r.frames[0].len() as i64 } else { 0 }),
            "dims": int(1),
        },
        "frame_times": finite_list(&times, Some(6)),
        "opening_frames": finite_list(&r.frame_open, Some(9)),
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
        "energy": energy,
        "meta": {
            "c": num(round_nd(C0_AIR, 3)),
            "f1": num(round_nd(f1, 3)),
            "num_steps": int(n_steps),
            "n_frames": int(n_frames as i64),
            "ends": ["reed", if info.radiating { "radiating" } else { "open" }],
            "radiating": info.radiating,
            "r_ratio": if info.radiating { num(info.ratio) } else { Value::Null },
            "Z0": num(info.z0),
            "gamma": num(round_nd(info.gamma, 4)),
            "p_mouth": num(round_nd(info.p_mouth, 2)),
            "p_closing": num(round_nd(info.p_closing, 2)),
            "H0": num(info.h0),
            "f_reed": num(round_nd(info.f_reed, 1)),
            "q_reed": num(round_nd(info.q_reed, 3)),
            "transit": num(round_nd(l / C0_AIR, 8)),
            "frames_per_transit": num(round_nd((fs * l / C0_AIR) / anim_stride as f64, 2)),
            "fallbacks": int(rb.state().fallbacks as i64),
            "budget": {
                "mouth_work": num(last(&r.mouth)),
                "jet_frac": num(round_nd(last(&r.jet) / breath, 4)),
                "damping_frac": num(round_nd(last(&r.damp) / breath, 4)),
                "radiated_frac": num(round_nd(last(&r.e_rad) / breath, 4)),
                "stored_frac": num(round_nd((last(&stored) - stored[0]) / breath, 4)),
            },
            "speaks": spoke,
            "ac_level": num(round_nd(ac_level, 6)),
            "envelope_ratio": num(round_nd(r.env[0] / env_last.max(1e-12), 2)),
            "beating": beating_block(&r.opening, fs, f1),
            "spectrum": signature_block(&r.pickup, &r.far, fs, f1, spoke),
            "sweep": sweep,
        },
    }))
}
