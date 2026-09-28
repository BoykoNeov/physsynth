//! Sympathetic strings — two fixed/free strings sharing ONE bridge point on a modal body.
//! `serialize.py`'s `_build_payload_sympathetic` and its three regimes.
//!
//! A closed, undriven, linear-leapfrog system, so conservation and passivity are automatic from its
//! structure — they pass even with a flipped coupling sign. They are table stakes; the claim lives
//! in the second panel, and the three regimes carry three claims:
//!
//! * **normal** — the antisymmetric start `u_B = -u_A` keeps the bridge EXACTLY still (bit-exact
//!   `w_b == 0`), shown against its symmetric contrast, which swings it. A flat zero alone would
//!   read as "broken", so both traces ship.
//! * **transfer** — pluck one string; a tuned neighbour drains most of its energy, a detuned one
//!   barely responds.
//! * **weinreich** — the piano unison's two-stage decay over a LOSSY body: strike one string and
//!   the symmetric mode dies fast (the prompt) while the antisymmetric one lingers (the
//!   aftersound). Its contrast is striking both, the pure symmetric mode: one slope, no floor.
//!
//! J is fixed at 2: the validated oracles are two-string.

use physsynth_core::body::{self as body, ModalBody};
use physsynth_core::connection::SympatheticStrings;
use physsynth_core::exciter::triangular_pluck;
use physsynth_core::pyfloat::scalar_pow;
use physsynth_core::string_ideal::{self as ideal, Boundary, IdealString};
use serde_json::{json, Value};

use crate::energy::{energy_block, lstsq_slope, EnergyOpts};
use crate::py::{
    b64f32, commas, finite_list, int, linspace_idx, max_abs, np_mean, num, py_int, py_repr, py_str,
    repr_str, round_int, round_nd, uniform_filter1d_nearest,
};
use crate::string::SPEED_MAX;
use crate::string::{construction, int_at_least_one, FRAMES_PER_PERIOD, MAX_FRAMES, N_MIN};
use crate::{fnum, resample_normalize, Refusal, AUDIO_FS};

/// The three regimes, selected by `domain`.
pub const SYMP_REGIMES: [&str; 3] = ["normal", "transfer", "weinreich"];
/// Fixed: the validated oracles are two-string.
pub const SYMP_J: usize = 2;
/// Grid ceiling: each step is two leapfrogs and a body step.
pub const SYMP_N_MAX: i64 = 160;
/// Below 1 REQUIRED: the bridge spring pushes the Nyquist mode unstable at 1.
pub const SYMP_LAM_DEFAULT: f64 = 0.9;
/// The normal-mode bridge.
pub const SYMP_K_DEFAULT: f64 = 8000.0;
/// A SOFTER spring is frequency-selective: the transfer bridge.
pub const SYMP_K_TRANSFER: f64 = 1500.0;
/// The two-stage-decay bridge.
pub const SYMP_K_WEINREICH: f64 = 6000.0;
/// Body loss default: a visible prompt and a long aftersound in ~2 s.
pub const SYMP_SIGMA_BODY_DEFAULT: f64 = 20.0;
/// Heavier only shortens the prompt to an invisible cliff.
pub const SYMP_SIGMA_BODY_MAX: f64 = 80.0;
/// A piano unison is mistuned by a few cents, so weinreich needs a FINE detune range.
pub const SYMP_WEINREICH_DETUNE_MAX: f64 = 0.4;
/// The body's off-harmonic modes (Hz).
pub const SYMP_BODY_FREQS: [f64; 5] = [137.0, 213.0, 330.0, 471.0, 620.0];
/// About the string's `rho L`, so the body genuinely reacts.
pub const SYMP_BODY_MASS: f64 = 0.02;
/// Pluck amplitude; the bridge-stillness claim is scale-invariant.
pub const SYMP_AMP: f64 = 1e-3;
/// Seconds of animation for the normal mode.
pub const SYMP_NORMAL_ANIM_WIN: f64 = 0.06;
/// Longest audio run, seconds.
pub const SYMP_AUDIO_MAX: f64 = 3.0;
/// Decimated length of the traces.
pub const SYMP_TRACE_POINTS: usize = 600;
/// Seconds of `w_b` actually plotted: long enough for the swings, short enough not to alias.
pub const SYMP_TRACE_WINDOW: f64 = 0.10;
/// Total steps (the normal and weinreich regimes run twice).
pub const SYMP_WORK_MAX: i64 = 130_000;

fn pf(x: f64) -> String {
    physsynth_core::fmt::py_float(x)
}

/// `_symp_regime`.
fn regime(p: &Value) -> Result<String, Refusal> {
    let r = p.get("domain").map_or_else(|| "normal".to_owned(), py_str);
    if SYMP_REGIMES.contains(&r.as_str()) {
        Ok(r)
    } else {
        Err(Refusal::Param(format!(
            "regime must be one of ('normal', 'transfer', 'weinreich'), got {}.",
            repr_str(&r)
        )))
    }
}

/// What `_build_sympathetic` returns besides the system.
struct Rig {
    symp: SympatheticStrings,
    c0: f64,
    l: f64,
    n: i64,
    fs: f64,
    lam: f64,
}

/// `_build_sympathetic`: two strings (the second detuned DOWN) on a shared modal body.
fn build(p: &Value, detune: f64, k_default: f64, sigma_body: f64) -> Result<Rig, Refusal> {
    let l = fnum(p, "L", 1.0)?;
    let t = fnum(p, "T", 200.0)?;
    let rho = fnum(p, "rho", 0.005)?;
    let lam = fnum(p, "lambda", SYMP_LAM_DEFAULT)?;
    let k = fnum(p, "K", k_default)?;
    let n_val = p.get("N").cloned().unwrap_or(json!(100));
    let n = py_int(&n_val)
        .map_err(|_| Refusal::Param(format!("N must be an integer, got {}.", py_repr(&n_val))))?;
    if !(N_MIN..=SYMP_N_MAX).contains(&n) {
        return Err(Refusal::Param(format!(
            "N must be in [{N_MIN}, {SYMP_N_MAX}] for sympathetic strings, got {n}."
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
            "bridge stiffness K must be >= 0, got {}.",
            pf(k)
        )));
    }

    let c0 = (t / rho).sqrt();
    let fs = c0 * n as f64 / (l * lam);
    // f ~ sqrt(T), so a semitone is a factor 2^(1/6) in tension.
    let tensions = [t, t * scalar_pow(2.0, -detune / 6.0)];
    let mut strings = Vec::with_capacity(SYMP_J);
    for tj in tensions {
        let sp = ideal::Params::new(
            l,
            tj,
            rho,
            fs,
            n,
            0.0,
            Some((Boundary::Fixed, Boundary::Free)),
        )
        .map_err(construction)?;
        strings.push(IdealString::new(sp));
    }
    let m = SYMP_BODY_FREQS.len();
    let bp = body::Params::new(
        SYMP_BODY_FREQS.to_vec(),
        fs,
        vec![sigma_body; m],
        vec![SYMP_BODY_MASS; m],
        vec![1.0; m],
        None,
    )
    .map_err(construction)?;
    let lam0 = strings[0].params().lam;
    let symp =
        SympatheticStrings::new(strings, ModalBody::new(bp), vec![k, k]).map_err(construction)?;
    Ok(Rig {
        symp,
        c0,
        l,
        n,
        fs,
        lam: lam0,
    })
}

/// Per-step telemetry of one run.
struct Run {
    e: Vec<f64>,
    wb: Vec<f64>,
    e_body: Vec<f64>,
    e_str: [Vec<f64>; SYMP_J],
    pickup: Vec<f64>,
    /// Each frame is the J fields stacked, string 0 first.
    frames: Vec<Vec<f64>>,
    frame_steps: Vec<usize>,
}

/// `_run_sympathetic`: step once, capturing every panel's telemetry in one pass.
fn run(
    symp: &mut SympatheticStrings,
    n_steps: usize,
    pickup_idx: usize,
    anim_stride: usize,
    frame_until: usize,
) -> Result<Run, Refusal> {
    let mut r = Run {
        e: Vec::with_capacity(n_steps + 1),
        wb: Vec::with_capacity(n_steps + 1),
        e_body: Vec::with_capacity(n_steps + 1),
        e_str: [
            Vec::with_capacity(n_steps + 1),
            Vec::with_capacity(n_steps + 1),
        ],
        pickup: Vec::with_capacity(n_steps + 1),
        frames: Vec::new(),
        frame_steps: Vec::new(),
    };
    let sample = |symp: &SympatheticStrings, r: &mut Run| {
        r.e.push(symp.energy());
        r.wb.push(symp.bridge_displacement(false));
        r.e_body.push(symp.body().energy());
        for (j, es) in r.e_str.iter_mut().enumerate() {
            es.push(symp.string_energy(j));
        }
        r.pickup.push(symp.strings()[0].u[pickup_idx]);
    };
    let cap = |symp: &SympatheticStrings, r: &mut Run, i: usize| {
        let mut f = Vec::new();
        for s in symp.strings() {
            f.extend_from_slice(&s.u);
        }
        r.frames.push(f);
        r.frame_steps.push(i);
    };
    sample(symp, &mut r);
    if frame_until >= 1 {
        cap(symp, &mut r, 0);
    }
    for i in 1..=n_steps {
        symp.step();
        sample(symp, &mut r);
        if i <= frame_until && i % anim_stride == 0 {
            cap(symp, &mut r, i);
        }
    }
    if !r.e.iter().all(|v| v.is_finite()) {
        return Err(Refusal::Param(
            "simulation produced non-finite energy (instability) — adjust parameters.".into(),
        ));
    }
    Ok(r)
}

/// `_symp_finish`: the payload common to all regimes.
#[allow(clippy::too_many_arguments)]
fn finish(
    regime: &str,
    frames_run: &Run,
    energy_e: &[f64],
    pickup: &[f64],
    field_labels: [&str; 2],
    grid_x: &[f64],
    fs: f64,
    lam: f64,
    anim_stride: i64,
    c0: f64,
    f1: f64,
    n_steps: i64,
    probe_x: f64,
    spectrum: Value,
    playback_speed: f64,
    sigma_zero: bool,
    decay_oracle: bool,
) -> Value {
    let flat: Vec<f64> = frames_run.frames.iter().flatten().copied().collect();
    let field_amp = if flat.is_empty() { 0.0 } else { max_abs(&flat) };
    let time: Vec<f64> = (0..energy_e.len()).map(|i| i as f64 / fs).collect();
    let (audio48, peak) = resample_normalize(pickup, fs);
    let times: Vec<f64> = frames_run
        .frame_steps
        .iter()
        .map(|&i| i as f64 / fs)
        .collect();
    let n_frames = frames_run.frames.len();
    json!({
        "model": "sympathetic",
        "regime": regime,
        "fs_sim": num(round_nd(fs, 3)),
        "lambda": num(round_nd(lam, 6)),
        "grid": {"x": finite_list(grid_x, Some(6))},
        "frames": {
            "b64": b64f32(&flat),
            "n_frames": int(n_frames as i64),
            "width": int(if n_frames > 0 { grid_x.len() as i64 } else { 0 }),
            "fields": ["string A", "string B"],
            "field_labels": field_labels,
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
        "energy": energy_block(&time, energy_e, sigma_zero, 0.0, EnergyOpts {
            no_decay_oracle: !decay_oracle,
            ..EnergyOpts::default()
        }),
        "meta": {
            "c": num(round_nd(c0, 3)),
            "f1": num(round_nd(f1, 3)),
            "num_steps": int(n_steps),
            "n_frames": int(n_frames as i64),
            "probe_x": num(round_nd(probe_x, 4)),
            "spectrum": spectrum,
        },
    })
}

/// The shared front half of every regime's builder.
struct Common {
    playback_speed: f64,
    pickup_frac: f64,
    pluck_frac: f64,
    audio_dur: f64,
    fpp: i64,
}

/// `_build_payload_sympathetic`.
pub fn build_payload(p: &Value) -> Result<Value, Refusal> {
    let regime = regime(p)?;
    let c = Common {
        playback_speed: fnum(p, "playback_speed", 0.02)?,
        pickup_frac: fnum(p, "pickup_position", 0.1)?,
        pluck_frac: fnum(p, "pluck_position", 0.3)?,
        audio_dur: fnum(p, "audio_duration", 2.0)?,
        fpp: int_at_least_one(fnum(p, "frames_per_period", FRAMES_PER_PERIOD)?)?,
    };
    if !(0.0 < c.playback_speed && c.playback_speed <= SPEED_MAX) {
        return Err(Refusal::Param(format!(
            "playback_speed must be in (0, {}], got {}.",
            pf(SPEED_MAX),
            pf(c.playback_speed)
        )));
    }
    if !(0.0 < c.pickup_frac && c.pickup_frac < 1.0) {
        return Err(Refusal::Param(format!(
            "pickup_position must be in (0, 1), got {}.",
            pf(c.pickup_frac)
        )));
    }
    if !(0.0 < c.pluck_frac && c.pluck_frac < 1.0) {
        return Err(Refusal::Param(format!(
            "pluck_position must be in (0, 1), got {}.",
            pf(c.pluck_frac)
        )));
    }
    if !(0.0 < c.audio_dur && c.audio_dur <= SYMP_AUDIO_MAX) {
        return Err(Refusal::Param(format!(
            "audio_duration must be in (0, {}] s, got {}.",
            pf(SYMP_AUDIO_MAX),
            pf(c.audio_dur)
        )));
    }
    match regime.as_str() {
        "transfer" => transfer(p, &c),
        "weinreich" => weinreich(p, &c),
        _ => normal(p, &c),
    }
}

fn pickup_index(frac: f64, n: i64) -> usize {
    round_int(frac * n as f64).max(1).min(n - 1) as usize
}

fn stride_for(fs: f64, f1: f64, fpp: i64, frames_over: i64) -> i64 {
    let mut s = round_int((fs / f1) / fpp as f64).max(1);
    if frames_over / s > MAX_FRAMES {
        s = ((frames_over as f64 / MAX_FRAMES as f64).ceil() as i64).max(1);
    }
    s
}

fn pick(a: &[f64], idx: &[usize]) -> Vec<f64> {
    idx.iter().map(|&i| a[i]).collect()
}

fn times_of(idx: &[usize], fs: f64) -> Value {
    let t: Vec<f64> = idx.iter().map(|&i| i as f64 / fs).collect();
    finite_list(&t, Some(6))
}

/// Python's `max(list)` over a non-empty float array (`np.max`, NaN-propagating).
fn np_max(a: &[f64]) -> f64 {
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

/// `_symp_normal`: run BOTH starts and plot both bridge traces.
fn normal(p: &Value, c: &Common) -> Result<Value, Refusal> {
    let k = fnum(p, "K", SYMP_K_DEFAULT)?;
    let mut anti = build(p, 0.0, SYMP_K_DEFAULT, 0.0)?;
    let mut sym = build(p, 0.0, SYMP_K_DEFAULT, 0.0)?;
    let (c0, l, n, fs, lam) = (anti.c0, anti.l, anti.n, anti.fs, anti.lam);
    let f1 = c0 / (2.0 * l);
    let pickup_idx = pickup_index(c.pickup_frac, n);
    let x = anti.symp.strings()[0].params().grid();
    let pluck = triangular_pluck(&x, l, c.pluck_frac * l, SYMP_AMP).map_err(construction)?;
    let minus: Vec<f64> = pluck.iter().map(|v| -v).collect(); // exact negation: w_b == 0 bit-exact
    let zeros = vec![0.0; pluck.len()];

    let n_steps = round_int(c.audio_dur * fs).max(1);
    if 2 * n_steps > SYMP_WORK_MAX {
        return Err(Refusal::Param(format!(
            "work budget exceeded ({} steps > {}): the normal-mode regime runs twice \
             (antisymmetric + its symmetric contrast). Lower the audio duration, N, or the \
             tension.",
            commas(2 * n_steps),
            commas(SYMP_WORK_MAX)
        )));
    }
    let frame_until = round_int(SYMP_NORMAL_ANIM_WIN * fs).max(1);
    let anim_stride = stride_for(fs, f1, c.fpp, frame_until);

    anti.symp.string_mut(0).set_state(&pluck, &zeros);
    anti.symp.string_mut(1).set_state(&minus, &zeros);
    let run_a = run(
        &mut anti.symp,
        n_steps as usize,
        pickup_idx,
        anim_stride as usize,
        frame_until as usize,
    )?;
    sym.symp.string_mut(0).set_state(&pluck, &zeros);
    sym.symp.string_mut(1).set_state(&pluck, &zeros);
    let run_s = run(&mut sym.symp, n_steps as usize, pickup_idx, 1, 0)?;

    let n_trace = n_steps.min(round_int(SYMP_TRACE_WINDOW * fs).max(1)) as usize;
    let idx = linspace_idx(n_trace, (n_trace + 1).min(SYMP_TRACE_POINTS));
    let e0 = run_a.e[0];
    let anti_max = max_abs(&run_a.wb);
    let sym_max = max_abs(&run_s.wb);
    let spectrum = json!({
        "kind": "sympathetic",
        "regime": "normal",
        "time": times_of(&idx, fs),
        "wb_anti": finite_list(&pick(&run_a.wb, &idx), None),
        "wb_sym": finite_list(&pick(&run_s.wb, &idx), None),
        "anti_max": num(anti_max),
        "sym_max": num(sym_max),
        "anti_exact_zero": anti_max == 0.0,
        "body_frac_anti": num(if e0 > 0.0 { np_max(&run_a.e_body) / e0 } else { 0.0 }),
        "body_frac_sym": num(if e0 > 0.0 { np_max(&run_s.e_body) / e0 } else { 0.0 }),
        "K": num(round_nd(k, 1)),
    });
    Ok(finish(
        "normal",
        &run_a,
        &run_a.e,
        &run_a.pickup,
        ["string A — plucked +", "string B — antiphase −"],
        &x,
        fs,
        lam,
        anim_stride,
        c0,
        f1,
        n_steps,
        x[pickup_idx],
        spectrum,
        c.playback_speed,
        true,
        true,
    ))
}

/// `_symp_transfer`: pluck string A; a tuned neighbour drains it, a detuned one does not.
fn transfer(p: &Value, c: &Common) -> Result<Value, Refusal> {
    let k = fnum(p, "K", SYMP_K_TRANSFER)?;
    let detune = fnum(p, "detune", 0.0)?;
    if !(0.0..=12.0).contains(&detune) {
        return Err(Refusal::Param(format!(
            "detune must be in [0, 12] semitones, got {}.",
            pf(detune)
        )));
    }
    let mut rig = build(p, detune, SYMP_K_TRANSFER, 0.0)?;
    let (c0, l, n, fs, lam) = (rig.c0, rig.l, rig.n, rig.fs, rig.lam);
    let f1 = c0 / (2.0 * l);
    let pickup_idx = pickup_index(c.pickup_frac, n);
    let x = rig.symp.strings()[0].params().grid();

    let n_steps = round_int(c.audio_dur * fs).max(1);
    if n_steps > SYMP_WORK_MAX {
        return Err(Refusal::Param(format!(
            "work budget exceeded ({} steps > {}). Lower the audio duration, N, or the tension.",
            commas(n_steps),
            commas(SYMP_WORK_MAX)
        )));
    }
    let anim_stride = stride_for(fs, f1, c.fpp, n_steps);
    let pluck = triangular_pluck(&x, l, c.pluck_frac * l, SYMP_AMP).map_err(construction)?;
    rig.symp
        .string_mut(0)
        .set_state(&pluck, &vec![0.0; pluck.len()]);
    let r = run(
        &mut rig.symp,
        n_steps as usize,
        pickup_idx,
        anim_stride as usize,
        n_steps as usize,
    )?;

    let idx = linspace_idx(
        n_steps as usize,
        (n_steps as usize + 1).min(SYMP_TRACE_POINTS),
    );
    let frac0: Vec<f64> = r.e_str[0].iter().zip(&r.e).map(|(a, e)| a / e).collect();
    let frac1: Vec<f64> = r.e_str[1].iter().zip(&r.e).map(|(a, e)| a / e).collect();
    let spectrum = json!({
        "kind": "sympathetic",
        "regime": "transfer",
        "time": times_of(&idx, fs),
        "frac0": finite_list(&pick(&frac0, &idx), None),
        "frac1": finite_list(&pick(&frac1, &idx), None),
        "peak_neighbour": num(np_max(&frac1)),
        "detune": num(round_nd(detune, 2)),
        "tuned": detune < 0.05,
        "K": num(round_nd(k, 1)),
    });
    Ok(finish(
        "transfer",
        &r,
        &r.e,
        &r.pickup,
        ["string A — plucked", "string B — sympathetic"],
        &x,
        fs,
        lam,
        anim_stride,
        c0,
        f1,
        n_steps,
        x[pickup_idx],
        spectrum,
        c.playback_speed,
        true,
        true,
    ))
}

/// `_neg_log_slope`: `-d/dt log e` over the samples above `1e-9 max(e)`; 0 when too short.
fn neg_log_slope(t: &[f64], e: &[f64]) -> f64 {
    let thresh = if e.is_empty() { 1.0 } else { np_max(e) * 1e-9 };
    let (tt, ll): (Vec<f64>, Vec<f64>) = t
        .iter()
        .zip(e)
        .filter(|(_, &ev)| ev > thresh)
        .map(|(&tv, &ev)| (tv, ev.ln()))
        .unzip();
    if tt.len() < 2 {
        return 0.0;
    }
    -lstsq_slope(&tt, &ll)
}

/// `_weinreich_rates`: `(prompt_rate, aftersound_rate, floor)` from the string-energy envelope.
fn weinreich_rates(t: &[f64], env: &[f64]) -> (f64, f64, f64) {
    let n = env.len();
    if n == 0 {
        return (0.0, 0.0, 0.0);
    }
    let floor = np_mean(&env[(0.80 * n as f64) as usize..]);
    let a = (0.60 * n as f64) as usize;
    let aftersound = neg_log_slope(&t[a..], &env[a..]);
    let excess: Vec<f64> = env.iter().map(|v| v - floor).collect();
    let (ts, es): (Vec<f64>, Vec<f64>) = t
        .iter()
        .zip(&excess)
        .filter(|(_, &x)| x > 0.03)
        .map(|(&tv, &x)| (tv, x))
        .unzip();
    let prompt = if ts.len() > 3 {
        neg_log_slope(&ts, &es)
    } else {
        0.0
    };
    (prompt, aftersound, floor)
}

/// `_symp_weinreich`: strike ONE of two near-unison strings over a lossy bridge.
fn weinreich(p: &Value, c: &Common) -> Result<Value, Refusal> {
    let k = fnum(p, "K", SYMP_K_WEINREICH)?;
    let sigma_body = fnum(p, "sigma_body", SYMP_SIGMA_BODY_DEFAULT)?;
    let detune = fnum(p, "detune", 0.0)?;
    if !(0.0..=SYMP_SIGMA_BODY_MAX).contains(&sigma_body) {
        return Err(Refusal::Param(format!(
            "body loss must be in [0, {}], got {}.",
            pf(SYMP_SIGMA_BODY_MAX),
            pf(sigma_body)
        )));
    }
    if !(0.0..=SYMP_WEINREICH_DETUNE_MAX).contains(&detune) {
        return Err(Refusal::Param(format!(
            "detune must be in [0, {}] semitones for the weinreich regime (a piano unison is \
             mistuned by a few cents), got {}.",
            pf(SYMP_WEINREICH_DETUNE_MAX),
            pf(detune)
        )));
    }
    let mut one = build(p, detune, SYMP_K_WEINREICH, sigma_body)?;
    let mut both = build(p, detune, SYMP_K_WEINREICH, sigma_body)?;
    let (c0, l, n, fs, lam) = (one.c0, one.l, one.n, one.fs, one.lam);
    let f1 = c0 / (2.0 * l);
    let pickup_idx = pickup_index(c.pickup_frac, n);
    let x = one.symp.strings()[0].params().grid();
    let pluck = triangular_pluck(&x, l, c.pluck_frac * l, SYMP_AMP).map_err(construction)?;
    let zeros = vec![0.0; pluck.len()];

    let n_steps = round_int(c.audio_dur * fs).max(1);
    if 2 * n_steps > SYMP_WORK_MAX {
        return Err(Refusal::Param(format!(
            "work budget exceeded ({} steps > {}): the weinreich regime runs twice (strike-one + \
             its strike-both contrast). Lower the audio duration, N, or the tension.",
            commas(2 * n_steps),
            commas(SYMP_WORK_MAX)
        )));
    }
    let anim_stride = stride_for(fs, f1, c.fpp, n_steps);

    one.symp.string_mut(0).set_state(&pluck, &zeros);
    let run_one = run(
        &mut one.symp,
        n_steps as usize,
        pickup_idx,
        anim_stride as usize,
        n_steps as usize,
    )?;
    both.symp.string_mut(0).set_state(&pluck, &zeros);
    both.symp.string_mut(1).set_state(&pluck, &zeros);
    let run_both = run(&mut both.symp, n_steps as usize, pickup_idx, 1, 0)?;

    let t_full: Vec<f64> = (0..=n_steps as usize).map(|i| i as f64 / fs).collect();
    let win = round_int(fs / f1).max(1) as usize;
    let sum2 = |r: &Run| -> Vec<f64> {
        r.e_str[0]
            .iter()
            .zip(&r.e_str[1])
            .map(|(a, b)| a + b)
            .collect()
    };
    let (e_one, e_both) = (sum2(&run_one), sum2(&run_both));
    let norm0 = |e: &[f64]| -> Vec<f64> { e.iter().map(|v| v / e[0]).collect() };
    let env_one = uniform_filter1d_nearest(&norm0(&e_one), win);
    let env_both = uniform_filter1d_nearest(&norm0(&e_both), win);
    let (prompt, aftersound, floor_one) = weinreich_rates(&t_full, &env_one);

    let idx = linspace_idx(
        n_steps as usize,
        (n_steps as usize + 1).min(SYMP_TRACE_POINTS),
    );
    let spectrum = json!({
        "kind": "sympathetic",
        "regime": "weinreich",
        "time": times_of(&idx, fs),
        "env_one": finite_list(&pick(&env_one, &idx), None),
        "env_both": finite_list(&pick(&env_both, &idx), None),
        "prompt_rate": num(round_nd(prompt, 3)),
        "aftersound_rate": num(round_nd(aftersound, 4)),
        "floor_one": num(round_nd(floor_one, 4)),
        "both_final": num(round_nd(env_both[env_both.len() - 1], 4)),
        "sigma_body": num(round_nd(sigma_body, 2)),
        "detune": num(round_nd(detune, 3)),
        "K": num(round_nd(k, 1)),
        "sigma_zero": sigma_body == 0.0,
    });
    Ok(finish(
        "weinreich",
        &run_one,
        &run_one.e,
        &run_one.pickup,
        ["string A — struck", "string B — silent → sympathetic"],
        &x,
        fs,
        lam,
        anim_stride,
        c0,
        f1,
        n_steps,
        x[pickup_idx],
        spectrum,
        c.playback_speed,
        sigma_body == 0.0,
        false,
    ))
}
