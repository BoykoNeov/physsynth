//! The linear string family — `ideal`, `stiff`, `damped` — `_build_resonator` and
//! `_build_payload_string`.
//!
//! Two runs per request, decoupled on purpose (the viewer plan's catches #1 and #2): an **audio**
//! run over the full duration that records only the pickup, and a short **animation** run on a
//! fresh resonator that records snapshots at a stride resolving the fundamental — about
//! [`FRAMES_PER_PERIOD`] per period, played back in slow motion. A wall-clock 60 fps stride would
//! alias the wiggle it exists to show.

use physsynth_analysis::{modal, spectrum};
use physsynth_core::engine::{simulate, Resonator};
use physsynth_core::exciter::triangular_pluck;
use physsynth_core::string_damped::{self as damped, DampedStiffString};
use physsynth_core::string_ideal::{self as ideal, Boundary, IdealString};
use physsynth_core::string_stiff::{self as stiff, StiffString};
use serde_json::{json, Value};

use crate::energy::{energy_block, EnergyOpts};
use crate::horizon::{string_block, StringInfo, StringScheme};
use crate::py::{
    b64f32, finite_list, float_to_int, int, max_abs, num, py_int, py_repr, round_int, round_nd,
    IntError,
};
use crate::{fnum, model_of, resample_normalize, Refusal, AUDIO_FS};

/// Animation temporal resolution (frames per fundamental period).
pub const FRAMES_PER_PERIOD: f64 = 12.0;
/// Partials overlaid on the spectrum.
pub const N_PARTIALS: usize = 12;
/// Payload guard: the frame-count ceiling.
pub const MAX_FRAMES: i64 = 1500;
/// The string's `N` range.
pub const N_MIN: i64 = 2;
/// The string's `N` range.
pub const N_MAX: i64 = 2000;
/// Longest audio run, seconds.
pub const AUDIO_DUR_MAX: f64 = 10.0;
/// Longest animation window, seconds.
pub const ANIM_WIN_MAX: f64 = 2.0;
/// Fastest playback speed.
pub const SPEED_MAX: f64 = 8.0;

/// One of the three linear strings.
pub enum StringRes {
    Ideal(IdealString),
    Stiff(StiffString),
    Damped(DampedStiffString),
}

impl StringRes {
    fn resonator(&mut self) -> &mut dyn Resonator {
        match self {
            StringRes::Ideal(r) => r,
            StringRes::Stiff(r) => r,
            StringRes::Damped(r) => r,
        }
    }

    /// Node positions `x`.
    pub fn grid(&self) -> Vec<f64> {
        match self {
            StringRes::Ideal(r) => r.params().grid(),
            StringRes::Stiff(r) => r.p.grid(),
            StringRes::Damped(r) => r.p.grid(),
        }
    }

    /// The Courant number `c k / h`.
    pub fn lam(&self) -> f64 {
        match self {
            StringRes::Ideal(r) => r.params().lam,
            StringRes::Stiff(r) => r.p.lam,
            StringRes::Damped(r) => r.p.lam,
        }
    }

    /// Displace from rest with zero velocity — `set_state(u0)`.
    fn set_displacement(&mut self, u0: &[f64]) {
        let v0 = vec![0.0; u0.len()];
        match self {
            StringRes::Ideal(r) => r.set_state(u0, &v0),
            StringRes::Stiff(r) => r.set_state(u0, &v0),
            StringRes::Damped(r) => r.set_state(u0, &v0),
        }
    }

    /// What the resolution read-out needs.
    pub fn horizon_info(&self) -> StringInfo {
        match self {
            StringRes::Ideal(r) => {
                let p = r.params();
                StringInfo {
                    pinned: p.bc_left == Boundary::Fixed && p.bc_right == Boundary::Fixed,
                    n: p.n,
                    c: p.c,
                    l: p.l,
                    fs: p.fs,
                    scheme: StringScheme::Explicit { lam: p.lam },
                }
            }
            // Both theta strings are simply supported at both ends — `supported` is a pinned end.
            StringRes::Stiff(r) => StringInfo {
                pinned: true,
                n: r.p.n,
                c: r.p.c,
                l: r.p.l,
                fs: r.p.fs,
                scheme: StringScheme::Theta {
                    kappa: r.p.kappa,
                    theta: r.p.theta,
                    k: r.p.k,
                },
            },
            StringRes::Damped(r) => StringInfo {
                pinned: true,
                n: r.p.n,
                c: r.p.c,
                l: r.p.l,
                fs: r.p.fs,
                scheme: StringScheme::Theta {
                    kappa: r.p.kappa,
                    theta: r.p.theta,
                    k: r.p.k,
                },
            },
        }
    }
}

/// `_Built`: the resonator plus the derived scalars the payload needs.
pub struct Built {
    pub res: StringRes,
    pub c: f64,
    pub l: f64,
    pub n: i64,
    pub fs: f64,
    pub sigma_zero: bool,
    pub oracle_2sigma: f64,
}

/// `min(a, b, c)` as Python's builtin does it — the first element unless a later one is `<` it,
/// so a NaN in first position is returned rather than skipped.
fn py_min3(a: f64, b: f64, c: f64) -> f64 {
    let mut m = a;
    if b < m {
        m = b;
    }
    if c < m {
        m = c;
    }
    m
}

/// A core constructor's refusal, surfaced as `serialize.py`'s `except ValueError` did.
fn construction<E: std::fmt::Display>(e: E) -> Refusal {
    Refusal::Construction(e.to_string())
}

/// Construct a fresh string from params. The check order is the reference's, because a request
/// that is wrong in two ways must be refused for the same one.
pub fn build_resonator(p: &Value) -> Result<Built, Refusal> {
    let model = model_of(p);
    let l = fnum(p, "L", 1.0)?;
    let t = fnum(p, "T", 200.0)?;
    let rho = fnum(p, "rho", 0.005)?;
    let lam = fnum(p, "lambda", 1.0)?;
    let n_val = p.get("N").cloned().unwrap_or(json!(128));
    let n = match py_int(&n_val) {
        Ok(n) => n,
        Err(_) => {
            return Err(Refusal::Param(format!(
                "N must be an integer, got {}.",
                py_repr(&n_val)
            )))
        }
    };
    if !(N_MIN..=N_MAX).contains(&n) {
        return Err(Refusal::Param(format!(
            "N must be in [{N_MIN}, {N_MAX}], got {n}."
        )));
    }
    if py_min3(l, t, rho) <= 0.0 {
        return Err(Refusal::Param("L, T, rho must all be positive.".into()));
    }
    if lam <= 0.0 {
        return Err(Refusal::Param(format!(
            "lambda must be > 0, got {}.",
            physsynth_core::fmt::py_float(lam)
        )));
    }

    let c = (t / rho).sqrt();
    let fs = c * n as f64 / (l * lam);

    match model.as_str() {
        "ideal" => {
            let sigma = fnum(p, "sigma", 0.0)?;
            let params = ideal::Params::new(
                l,
                t,
                rho,
                fs,
                n,
                sigma,
                Some((Boundary::Fixed, Boundary::Fixed)),
            )
            .map_err(construction)?;
            Ok(Built {
                res: StringRes::Ideal(IdealString::new(params)),
                c,
                l,
                n,
                fs,
                sigma_zero: sigma == 0.0,
                oracle_2sigma: 2.0 * sigma,
            })
        }
        "stiff" => {
            let sigma = fnum(p, "sigma", 0.0)?;
            let kappa = fnum(p, "kappa", 0.0)?;
            let theta = fnum(p, "theta", 0.28)?;
            let params = stiff::Params::new(l, t, rho, fs, n, kappa, sigma, theta, true)
                .map_err(construction)?;
            Ok(Built {
                res: StringRes::Stiff(StiffString::new(params)),
                c,
                l,
                n,
                fs,
                sigma_zero: sigma == 0.0,
                oracle_2sigma: 2.0 * sigma,
            })
        }
        "damped" => {
            let s0 = fnum(p, "sigma0", 0.0)?;
            let s1 = fnum(p, "sigma1", 0.0)?;
            let kappa = fnum(p, "kappa", 0.0)?;
            let theta = fnum(p, "theta", 0.28)?;
            let params = damped::Params::new(l, t, rho, fs, n, kappa, s0, s1, theta, true)
                .map_err(construction)?;
            Ok(Built {
                res: StringRes::Damped(DampedStiffString::new(params)),
                c,
                l,
                n,
                fs,
                // The frequency-INDEPENDENT base rate; sigma1 adds a per-mode term on top.
                sigma_zero: s0 == 0.0 && s1 == 0.0,
                oracle_2sigma: 2.0 * s0,
            })
        }
        other => Err(Refusal::Param(format!(
            "unknown model {} (expected 'ideal' | 'stiff' | 'damped' | 'tension' | 'bow').",
            crate::py::repr_str(other)
        ))),
    }
}

/// `max(1, int(x))` for a float parameter — the reference's `int()` refusals included.
///
/// `int(nan)` raises `ValueError`, which `serialize.py`'s outer `except ValueError` reported as a
/// *construction* error, so that is the kind it keeps here.
pub fn int_at_least_one(x: f64) -> Result<i64, Refusal> {
    match float_to_int(x) {
        Ok(i) => Ok(i.max(1)),
        Err(IntError::NaN) => Err(Refusal::Construction(
            "cannot convert float NaN to integer".into(),
        )),
        Err(_) => Err(Refusal::Construction(
            "cannot convert float infinity to integer".into(),
        )),
    }
}

/// Detected-vs-analytic partials (cents error), on the full-rate pickup.
fn partials_block(pickup: &[f64], fs: f64, model: &str, p: &Value, c: f64, l: f64) -> Value {
    let analytic: Vec<f64> = if model == "stiff" || model == "damped" {
        // Already parsed once by the resonator build, so it cannot refuse here.
        let kappa = fnum(p, "kappa", 0.0).unwrap_or(0.0);
        modal::stiff_harmonic_frequencies(c, l, kappa, N_PARTIALS)
    } else {
        modal::harmonic_frequencies(c, l, N_PARTIALS)
    };
    let analytic: Vec<f64> = analytic
        .into_iter()
        .filter(|&f| f < 0.95 * (fs / 2.0))
        .collect();
    if analytic.is_empty() {
        return Value::Null;
    }
    let detected = spectrum::measure_partials_near(pickup, fs, &analytic, None);
    let cents: Vec<f64> = detected
        .iter()
        .zip(&analytic)
        .map(|(&d, &a)| modal::cents(d, a))
        .collect();
    json!({
        "analytic": finite_list(&analytic, Some(4)),
        "detected": finite_list(&detected, Some(4)),
        "cents": finite_list(&cents, Some(4)),
    })
}

/// `_build_payload_string`.
pub fn build_payload(p: &Value) -> Result<Value, Refusal> {
    let model = model_of(p);
    let audio_dur = fnum(p, "audio_duration", 2.0)?;
    let anim_win = fnum(p, "animation_window", 0.06)?;
    let playback_speed = fnum(p, "playback_speed", 0.02)?;
    let amplitude = fnum(p, "amplitude", 1e-3)?;
    let pluck_frac = fnum(p, "pluck_position", 0.3)?;
    let pickup_frac = fnum(p, "pickup_position", 0.1)?;
    let fpp = int_at_least_one(fnum(p, "frames_per_period", FRAMES_PER_PERIOD)?)?;

    let pf = physsynth_core::fmt::py_float;
    if !(0.0 < audio_dur && audio_dur <= AUDIO_DUR_MAX) {
        return Err(Refusal::Param(format!(
            "audio_duration must be in (0, {}] s, got {}.",
            pf(AUDIO_DUR_MAX),
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
    if !(0.0 < pluck_frac && pluck_frac < 1.0) {
        return Err(Refusal::Param(format!(
            "pluck_position must be in (0, 1), got {}.",
            pf(pluck_frac)
        )));
    }
    if !(0.0 < pickup_frac && pickup_frac < 1.0) {
        return Err(Refusal::Param(format!(
            "pickup_position must be in (0, 1), got {}.",
            pf(pickup_frac)
        )));
    }

    let mut b = build_resonator(p)?;
    let (c, l, n, fs) = (b.c, b.l, b.n, b.fs);
    let f1_base = c / (2.0 * l);
    let pickup_idx = round_int(pickup_frac * n as f64).max(1).min(n - 1) as usize;

    // -- audio run: full duration, pickup, no snapshots
    let x = b.res.grid();
    let pluck = triangular_pluck(&x, l, pluck_frac * l, amplitude).map_err(construction)?;
    b.res.set_displacement(&pluck);
    let n_audio = round_int(audio_dur * fs).max(1);
    let audio_run =
        simulate(b.res.resonator(), n_audio as usize, Some(pickup_idx), 0).map_err(construction)?;
    let pickup = audio_run.output.as_deref().expect("a pickup was requested");
    if !pickup.iter().all(|v| v.is_finite()) {
        return Err(Refusal::Param(
            "simulation produced non-finite output (instability) — adjust parameters.".into(),
        ));
    }

    // -- animation run: fresh resonator, short window, fundamental-resolving stride
    let mut anim = build_resonator(p)?.res;
    anim.set_displacement(&pluck);
    let mut anim_stride = round_int((fs / f1_base) / fpp as f64).max(1);
    let n_anim = anim_stride.max(round_int(anim_win * fs));
    if n_anim / anim_stride > MAX_FRAMES {
        anim_stride = ((n_anim as f64 / MAX_FRAMES as f64).ceil() as i64).max(1);
    }
    let anim_run = simulate(
        anim.resonator(),
        n_anim as usize,
        None,
        anim_stride as usize,
    )
    .map_err(construction)?;
    let n_frames = anim_run.snapshots.len();
    let width = anim_run.snapshots.first().map_or(0, |s| s.1.len());
    let flat: Vec<f64> = anim_run
        .snapshots
        .iter()
        .flat_map(|(_, s)| s.iter().copied())
        .collect();
    let frame_times: Vec<f64> = anim_run
        .snapshots
        .iter()
        .map(|(i, _)| *i as f64 / fs)
        .collect();

    // -- audio + assembly
    let (audio48, peak) = resample_normalize(pickup, fs);
    let field_amp = if flat.is_empty() { 0.0 } else { max_abs(&flat) };

    let energy = energy_block(
        &audio_run.time,
        &audio_run.energy,
        b.sigma_zero,
        b.oracle_2sigma,
        EnergyOpts::default(),
    );
    let horizon = string_block(&b.res.horizon_info(), &format!("the {model} string"));
    Ok(json!({
        "model": model,
        "fs_sim": num(round_nd(fs, 3)),
        "lambda": num(round_nd(b.res.lam(), 6)),
        "grid": {"x": finite_list(&x, Some(6))},
        "frames": {
            "b64": b64f32(&flat),
            "n_frames": int(n_frames as i64),
            "width": int(width as i64),
            "dims": int(1),
        },
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
        "energy": energy,
        "horizon": horizon,
        "meta": {
            "c": num(round_nd(c, 3)),
            "f1": num(round_nd(f1_base, 3)),
            "num_steps": int(n_audio),
            "n_frames": int(n_frames as i64),
            "partials": partials_block(pickup, fs, &model, p, c, l),
        },
    }))
}
