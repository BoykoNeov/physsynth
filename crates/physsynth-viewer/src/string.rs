//! The string family's constructor, `_build_resonator` — all five keys it builds (`ideal`,
//! `stiff`, `damped`, `tension`, `bow`) — and the linear strings' payload, `_build_payload_string`.
//! The tension string's and the bow's payloads are in their own modules.
//!
//! Two runs per request, decoupled on purpose (the viewer plan's catches #1 and #2): an **audio**
//! run over the full duration that records only the pickup, and a short **animation** run on a
//! fresh resonator that records snapshots at a stride resolving the fundamental — about
//! [`FRAMES_PER_PERIOD`] per period, played back in slow motion. A wall-clock 60 fps stride would
//! alias the wiggle it exists to show.

use physsynth_analysis::{modal, spectrum};
use physsynth_core::bow::BowedString;
use physsynth_core::engine::{simulate, Resonator};
use physsynth_core::exciter::triangular_pluck;
use physsynth_core::string_damped::{self as damped, DampedStiffString};
use physsynth_core::string_ideal::{self as ideal, Boundary, IdealString};
use physsynth_core::string_nonlinear::{self as nonlinear, TensionModulatedString};
use physsynth_core::string_stiff::{self as stiff, StiffString};
use physsynth_core::string_tuning::{
    ideal_mode_frequency, mode_frequency, partial_drift, stretched_partial, string_material,
    tune_damped_string, tune_flexible_string, tune_ideal_string, tune_tension_string,
    FlexibleTuning, GaugeTuning, IdealTuning, StringMaterial, Targets, Tuning, HORIZON_CENTS,
};
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

/// Whatever `_build_resonator` built.
pub enum StringRes {
    Ideal(IdealString),
    Stiff(StiffString),
    Damped(DampedStiffString),
    Tension(TensionModulatedString),
    Bow(Box<BowedString>),
}

impl StringRes {
    /// The resonator, for the engine.
    pub fn resonator(&mut self) -> &mut dyn Resonator {
        match self {
            StringRes::Ideal(r) => r,
            StringRes::Stiff(r) => r,
            StringRes::Damped(r) => r,
            StringRes::Tension(r) => r,
            StringRes::Bow(r) => r.as_mut(),
        }
    }

    /// Node positions `x` (the bowed string's own).
    pub fn grid(&self) -> Vec<f64> {
        match self {
            StringRes::Ideal(r) => r.params().grid(),
            StringRes::Stiff(r) => r.p.grid(),
            StringRes::Damped(r) => r.p.grid(),
            StringRes::Tension(r) => r.p.grid(),
            StringRes::Bow(r) => r.string.p.grid(),
        }
    }

    /// The Courant number `c k / h`.
    pub fn lam(&self) -> f64 {
        match self {
            StringRes::Ideal(r) => r.params().lam,
            StringRes::Stiff(r) => r.p.lam,
            StringRes::Damped(r) => r.p.lam,
            StringRes::Tension(r) => r.p.lam,
            StringRes::Bow(r) => r.string.p.lam,
        }
    }

    /// Displace from rest with zero velocity — `set_state(u0)`.
    pub fn set_displacement(&mut self, u0: &[f64]) {
        let v0 = vec![0.0; u0.len()];
        match self {
            StringRes::Ideal(r) => r.set_state(u0, &v0),
            StringRes::Stiff(r) => r.set_state(u0, &v0),
            StringRes::Damped(r) => r.set_state(u0, &v0),
            StringRes::Tension(r) => r.set_state(u0, &v0),
            StringRes::Bow(r) => r.string.set_state(u0, &v0),
        }
    }

    /// The tension string, for the builders that asked for one.
    pub fn into_tension(self) -> TensionModulatedString {
        match self {
            StringRes::Tension(r) => r,
            _ => unreachable!("built with model = tension"),
        }
    }

    /// The bowed string, for the builder that asked for one.
    pub fn into_bow(self) -> BowedString {
        match self {
            StringRes::Bow(r) => *r,
            _ => unreachable!("built with model = bow"),
        }
    }

    /// What the resolution read-out needs.
    ///
    /// # Panics
    /// On the tension string, which has no linear modal reference and is never asked.
    pub fn horizon_info(&self) -> StringInfo {
        match self {
            StringRes::Tension(_) => unreachable!("the tension string's read-out is a refusal"),
            StringRes::Bow(r) => damped_info(&r.string),
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
            StringRes::Damped(r) => damped_info(r),
        }
    }
}

/// The bow's read-out is its string's: an exciter inherits the dispersion of what it drives.
pub fn damped_info(r: &DampedStiffString) -> StringInfo {
    StringInfo {
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
    }
}

/// Snapshots flattened for the payload: the float32 buffer, its shape, and each frame's step.
pub struct Frames {
    pub flat: Vec<f64>,
    pub n: usize,
    pub width: usize,
    /// Step index of each frame, relative to `offset` (so the animation clock starts at 0).
    pub steps: Vec<f64>,
}

impl Frames {
    /// Flatten `(step, state)` snapshots, reporting steps relative to `offset`.
    pub fn of(snaps: &[(usize, Vec<f64>)], offset: usize) -> Frames {
        Frames {
            flat: snaps.iter().flat_map(|(_, s)| s.iter().copied()).collect(),
            n: snaps.len(),
            width: snaps.first().map_or(0, |s| s.1.len()),
            steps: snaps.iter().map(|(i, _)| (*i - offset) as f64).collect(),
        }
    }

    /// The `frames` object of a 1-D payload.
    pub fn json(&self) -> Value {
        json!({
            "b64": b64f32(&self.flat),
            "n_frames": int(self.n as i64),
            "width": int(self.width as i64),
            "dims": int(1),
        })
    }

    /// `frame_steps / fs`, rounded to 6 places — the `frame_times` list.
    pub fn times(&self, fs: f64) -> Value {
        let t: Vec<f64> = self.steps.iter().map(|s| s / fs).collect();
        finite_list(&t, Some(6))
    }

    /// `np.max(np.abs(frames))`, or 0 with no frames — `field_amp`.
    pub fn amp(&self) -> f64 {
        if self.flat.is_empty() {
            0.0
        } else {
            max_abs(&self.flat)
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
    /// The musical targets and their mapping, when the string was built from musical controls
    /// (`build_musical`); `None` on every physics path.
    pub tuning: Option<Musical>,
}

/// What a string built from musical targets carries into its payload's `tuning` block.
pub struct Musical {
    /// What the mapping gave.
    pub map: Mapped,
    /// What was asked. The ideal and flexible strings take a pitch and one ring time; their
    /// targets carry `inharmonicity = 0` (the harmonic law) and both ring times equal.
    pub targets: Targets,
    /// The theta the string was built with, which the drift read-out needs.
    pub theta: f64,
    /// The tension string's material and gauge; `None` for the damped string and the bow.
    pub gauge: Option<Gauge>,
}

/// Which mapping built the string (string mapping plan §1).
pub enum Mapped {
    /// The damped string's, the bow's and (gentle limit) the tension string's: two anchors.
    Damped(Tuning),
    /// The ideal string at `lambda = 1`: the pitch and every partial's ring time exact.
    Ideal(IdealTuning),
    /// The barrier pages' flexible string: mode 1 exact.
    Flexible(FlexibleTuning),
}

/// The tension string's material, diameter (m) and what they gave.
pub struct Gauge {
    pub material: &'static StringMaterial,
    pub diameter: f64,
    pub g: GaugeTuning,
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

/// The musical domain's sample-rate range (Hz). In this domain `fs` is an input — the pitch fixes
/// `c`, so the Courant number becomes a read-out — and the cap is the same cost guard `lambda`'s
/// floor is on the physics path.
pub const MUSICAL_FS_MIN: f64 = 8000.0;
/// See [`MUSICAL_FS_MIN`].
pub const MUSICAL_FS_MAX: f64 = 192_000.0;

/// The bow's musical sample-rate default. Its page caps a run at 60,000 friction steps
/// ([`crate::bow::BOW_WORK_MAX`]), which holds the page's 2 s default run at 22.05 kHz and not at
/// 44.1 kHz (string mapping plan §9).
pub const BOW_MUSICAL_FS_DEFAULT: f64 = 22050.0;
/// The tension string's diameter range in musical mode (mm): a plain string from a thin treble to
/// a thick unwound bass.
pub const DIAMETER_MM_MIN: f64 = 0.1;
/// See [`DIAMETER_MM_MIN`].
pub const DIAMETER_MM_MAX: f64 = 1.5;

/// True when a string request asks for musical controls.
///
/// The damped string names them on its `domain` select (the first musical page); the bow and the
/// tension string on `controls`, because the tension string's `domain` is already its regime
/// (string mapping plan §7). Keyed on the select, not on the presence of `f0`: the front-end
/// sends every slider it has to every model, so a musical key's presence says nothing about what
/// the user chose.
pub fn is_musical(p: &Value) -> bool {
    let key = match model_of(p).as_str() {
        "damped" => "domain",
        "bow" | "tension" | "ideal" | "jawari" | "juari" | "fret" => "controls",
        _ => return false,
    };
    p.get(key).and_then(Value::as_str) == Some("musical")
}

/// The tension string's material and gauge from the request (`string_material`, `diameter_mm`).
fn gauge_of(
    p: &Value,
    l: f64,
    fs: f64,
    n: i64,
    theta: f64,
    targets: &Targets,
) -> Result<Gauge, Refusal> {
    let name = p
        .get("string_material")
        .map_or_else(|| "steel".to_owned(), crate::py::py_str);
    let material = string_material(&name).ok_or_else(|| {
        Refusal::Param(format!(
            "string_material must be one of steel, nylon, gut; got {}.",
            crate::py::repr_str(&name)
        ))
    })?;
    let mm = fnum(p, "diameter_mm", 0.5)?;
    if !(DIAMETER_MM_MIN..=DIAMETER_MM_MAX).contains(&mm) {
        return Err(Refusal::Param(format!(
            "diameter_mm must be in [{DIAMETER_MM_MIN}, {DIAMETER_MM_MAX}] mm, got {}.",
            physsynth_core::fmt::py_float(mm)
        )));
    }
    let diameter = mm * 1e-3;
    let g = tune_tension_string(material, diameter, l, fs, n as usize, theta, targets)
        .map_err(|e| Refusal::Param(e.to_string()))?;
    Ok(Gauge {
        material,
        diameter,
        g,
    })
}

/// A string built from musical targets: the damped string (`docs/dev/parameter-mapping-plan.md`
/// §6), the bow's string and the tension string (`docs/dev/string-mapping-plan.md`).
///
/// `T`, `kappa`, `sigma0` and `sigma1` come from [`tune_damped_string`] — through
/// [`tune_tension_string`] for the tension string, whose `rho` and `EA` come from its gauge. The
/// request's own values of those keys and of `lambda` are ignored, though the shared prefix of
/// [`build_resonator`] still parsed and range-checked them. `lossless` zeroes both mapped losses,
/// for the tension page's single-mode measurements (its physics path zeroes the keys instead).
fn build_musical(
    p: &Value,
    model: &str,
    l: f64,
    rho: f64,
    n: i64,
    lossless: bool,
) -> Result<Built, Refusal> {
    let theta = fnum(p, "theta", 0.28)?;
    let targets = Targets {
        f0: fnum(p, "f0", 110.0)?,
        inharmonicity: fnum(p, "inharmonicity", 1e-4)?,
        t60_low: fnum(p, "t60_low", 3.0)?,
        t60_high: fnum(p, "t60_high", 0.8)?,
        f_ref: fnum(p, "f_ref", 2000.0)?,
    };
    let fs_default = if model == "bow" {
        BOW_MUSICAL_FS_DEFAULT
    } else {
        44100.0
    };
    let fs = fnum(p, "sample_rate", fs_default)?;
    if !(MUSICAL_FS_MIN..=MUSICAL_FS_MAX).contains(&fs) {
        return Err(Refusal::Param(format!(
            "sample_rate must be in [{}, {}] Hz, got {}.",
            MUSICAL_FS_MIN,
            MUSICAL_FS_MAX,
            physsynth_core::fmt::py_float(fs)
        )));
    }
    let gauge = if model == "tension" {
        Some(gauge_of(p, l, fs, n, theta, &targets)?)
    } else {
        None
    };
    let (rho, tu) = match &gauge {
        Some(g) => (g.g.rho, g.g.tuning),
        None => (
            rho,
            tune_damped_string(l, rho, fs, n as usize, theta, &targets)
                .map_err(|e| Refusal::Param(e.to_string()))?,
        ),
    };
    let (s0, s1) = if lossless {
        (0.0, 0.0)
    } else {
        (tu.sigma0, tu.sigma1)
    };
    let res = match (model, &gauge) {
        ("tension", Some(g)) => {
            let ea = g.g.ea;
            if ea > crate::tension::TENSION_EA_MAX {
                return Err(Refusal::Param(format!(
                    "this gauge gives EA = {ea:.0} N, above the page's {:.0} N; choose a \
                     thinner string or a softer material.",
                    crate::tension::TENSION_EA_MAX
                )));
            }
            let params = nonlinear::Params::new(
                l,
                tu.t,
                rho,
                fs,
                n,
                tu.kappa,
                ea,
                s0,
                s1,
                theta,
                nonlinear::TENSION_TOL_DEFAULT,
                true,
            )
            .map_err(construction)?;
            StringRes::Tension(TensionModulatedString::new(params))
        }
        _ => {
            let params = damped::Params::new(l, tu.t, rho, fs, n, tu.kappa, s0, s1, theta, true)
                .map_err(construction)?;
            let string = DampedStiffString::new(params);
            if model == "bow" {
                let settings = bow_settings(p)?;
                StringRes::Bow(Box::new(bow_on(p, string, settings)?))
            } else {
                StringRes::Damped(string)
            }
        }
    };
    Ok(Built {
        res,
        c: tu.c,
        l,
        n,
        fs,
        sigma_zero: s0 == 0.0 && s1 == 0.0,
        oracle_2sigma: 2.0 * s0,
        tuning: Some(Musical {
            map: Mapped::Damped(tu),
            targets,
            theta,
            gauge,
        }),
    })
}

/// The pitch and the one ring time the ideal and flexible strings take (`f0`, `t60_low`), as
/// targets on the harmonic law.
fn one_ring_targets(p: &Value) -> Result<Targets, Refusal> {
    let f0 = fnum(p, "f0", 110.0)?;
    let t60 = fnum(p, "t60_low", 3.0)?;
    Ok(Targets {
        f0,
        inharmonicity: 0.0,
        t60_low: t60,
        t60_high: t60,
        f_ref: f0,
    })
}

/// The ideal string from a pitch and a ring time at `lambda = 1` (string mapping plan §2, §13):
/// the sample rate follows the pitch, the human's call. The request's `T`, `sigma` and `lambda`
/// are ignored, though the shared prefix of [`build_resonator`] still parsed and checked them.
fn build_musical_ideal(p: &Value, l: f64, rho: f64, n: i64) -> Result<Built, Refusal> {
    let targets = one_ring_targets(p)?;
    let tu = tune_ideal_string(l, rho, n as usize, targets.f0, targets.t60_low)
        .map_err(|e| Refusal::Param(e.to_string()))?;
    let params = ideal::Params::new(
        l,
        tu.t,
        rho,
        tu.fs,
        n,
        tu.sigma,
        Some((Boundary::Fixed, Boundary::Fixed)),
    )
    .map_err(construction)?;
    Ok(Built {
        res: StringRes::Ideal(IdealString::new(params)),
        c: tu.c,
        l,
        n,
        fs: tu.fs,
        sigma_zero: tu.sigma == 0.0,
        oracle_2sigma: 2.0 * tu.sigma,
        tuning: Some(Musical {
            map: Mapped::Ideal(tu),
            targets,
            theta: 0.0,
            gauge: None,
        }),
    })
}

/// The barrier pages' flexible string from a pitch, a ring time and the sample-rate slider
/// (string mapping plan §13) — `(tension, sigma0, fs, wave speed, the musical record)`. The page
/// builds its own string from them; its `lambda` becomes a read-out, refused at 1 or above as the
/// physics path refuses it (the contact solve needs headroom below the Nyquist mode).
pub(crate) fn flexible_musical(
    p: &Value,
    l: f64,
    rho: f64,
    n: i64,
    theta: f64,
) -> Result<(f64, f64, f64, f64, Musical), Refusal> {
    let targets = one_ring_targets(p)?;
    let fs = fnum(p, "sample_rate", 44100.0)?;
    if !(MUSICAL_FS_MIN..=MUSICAL_FS_MAX).contains(&fs) {
        return Err(Refusal::Param(format!(
            "sample_rate must be in [{}, {}] Hz, got {}.",
            MUSICAL_FS_MIN,
            MUSICAL_FS_MAX,
            physsynth_core::fmt::py_float(fs)
        )));
    }
    let tu = tune_flexible_string(l, rho, fs, n as usize, theta, targets.f0, targets.t60_low)
        .map_err(|e| Refusal::Param(e.to_string()))?;
    if tu.lam >= 1.0 {
        return Err(Refusal::Param(format!(
            "this pitch on N = {n} needs lambda = {:.2} at {} Hz, and the contact solve needs \
             lambda < 1 (headroom below the string's Nyquist mode): raise the sample rate above \
             {:.0} Hz, lower N, or lower the pitch.",
            tu.lam,
            physsynth_core::fmt::py_float(fs),
            (fs * tu.lam).ceil()
        )));
    }
    let musical = Musical {
        map: Mapped::Flexible(tu),
        targets,
        theta,
        gauge: None,
    };
    Ok((tu.t, tu.sigma0, fs, tu.c, musical))
}

/// The bow's force, speed and friction sharpness, range-checked — `(force, v_bow, sharpness)`.
fn bow_settings(p: &Value) -> Result<(f64, f64, f64), Refusal> {
    use crate::bow as b;
    let force = fnum(p, "force", b::BOW_FORCE_DEFAULT)?;
    let v_bow = fnum(p, "v_bow", b::BOW_V_DEFAULT)?;
    let sharp = fnum(p, "sharpness", b::BOW_SHARPNESS_DEFAULT)?;
    let pf = physsynth_core::fmt::py_float;
    if !(0.0..=b::BOW_FORCE_MAX).contains(&force) {
        return Err(Refusal::Param(format!(
            "force must be in [0, {}] N, got {}.",
            pf(b::BOW_FORCE_MAX),
            pf(force)
        )));
    }
    if !(0.0 < v_bow && v_bow <= b::BOW_V_MAX) {
        return Err(Refusal::Param(format!(
            "v_bow must be in (0, {}] m/s, got {}.",
            pf(b::BOW_V_MAX),
            pf(v_bow)
        )));
    }
    if !(0.0 < sharp && sharp <= b::BOW_SHARPNESS_MAX) {
        return Err(Refusal::Param(format!(
            "sharpness must be in (0, {}], got {}.",
            pf(b::BOW_SHARPNESS_MAX),
            pf(sharp)
        )));
    }
    Ok((force, v_bow, sharp))
}

/// A bow on `string` at the request's position, with `settings` from [`bow_settings`].
fn bow_on(
    p: &Value,
    string: DampedStiffString,
    (force, v_bow, sharp): (f64, f64, f64),
) -> Result<BowedString, Refusal> {
    let bow_position = fnum(p, "bow_position", crate::bow::BOW_POSITION_DEFAULT)?;
    // `BowedString`'s own defaults for the Newton solve: tol 1e-13, 60 iterations.
    BowedString::new(string, bow_position, v_bow, force, sharp, 1e-13, 60).map_err(construction)
}

/// The tension string's stress warnings, in plain words — the human's "warn, keep running" for a
/// fit used outside its data and for a string past its breaking stress (string mapping plan §12).
fn stress_warnings(g: &Gauge) -> Vec<String> {
    let mpa = g.g.stress / 1e6;
    let mut out = Vec::new();
    if let physsynth_core::string_tuning::Modulus::StressFit {
        measured_mpa: (lo, hi),
        ..
    } = g.material.modulus
    {
        if g.g.modulus_extrapolated {
            out.push(format!(
                "this {} string runs at {mpa:.0} MPa, outside the {lo:.0}–{hi:.0} MPa its \
                 stiffness was measured at, so E = {:.1} GPa is an extrapolation of the fit.",
                g.material.name,
                g.g.youngs / 1e9
            ));
        }
    }
    if g.g.past_breaking {
        out.push(format!(
            "a real {} string would break at this pitch: {mpa:.0} MPa is past its {:.0} MPa ({}). \
             The model still sounds; a shorter string or a lower pitch is a real one.",
            g.material.name, g.material.breaking_mpa, g.material.breaking_source
        ));
    }
    out
}

/// The `tuning` block of a musical payload: what was asked, what it maps to, and how far the
/// partials the mapping does not pin drift (the human's "allow and warn", parameter-mapping plan
/// §3). The tension string adds its gauge.
pub(crate) fn tuning_block(b: &Built) -> Option<Value> {
    Some(tuning_block_for(b.tuning.as_ref()?, b.l, b.n, b.fs))
}

/// The partials an ideal or flexible read-out judges against `m f0`: up to the 16th, below the
/// scheme's own top mode.
const HARMONIC_READOUT_PARTIALS: usize = 16;

/// How far the partials of an ideal or flexible string sit from `m f0`: the highest partial up to
/// which every one is within [`HORIZON_CENTS`], and the worst signed drift among partials
/// `2..=`[`HARMONIC_READOUT_PARTIALS`] — `(horizon, worst_cents, worst_mode)`.
fn harmonic_drift(f0: f64, n: i64, freq: impl Fn(usize) -> Option<f64>) -> (usize, f64, usize) {
    let (mut horizon, mut worst, mut worst_mode) = (0usize, 0.0f64, 0usize);
    for m in 1..n.max(1) as usize {
        let Some(f) = freq(m) else { break };
        let c = 1200.0 * (f / (m as f64 * f0)).log2();
        if horizon == m - 1 && c.abs() <= HORIZON_CENTS {
            horizon = m;
        }
        if (2..=HARMONIC_READOUT_PARTIALS).contains(&m) && c.abs() > worst.abs() {
            worst = c;
            worst_mode = m;
        }
    }
    (horizon, worst, worst_mode)
}

/// The ideal or flexible string's `tuning` block (string mapping plan §13): one ring time, the
/// harmonic law as the reference, and which way the partials above `f0` lean.
fn harmonic_tuning_block(m: &Musical, l: f64, n: i64, fs: f64) -> Value {
    let f0 = m.targets.f0;
    let reach = HARMONIC_READOUT_PARTIALS.min((n - 1).max(1) as usize);
    let (kind, t, sigma, lam, (horizon, worst, worst_mode), why) = match &m.map {
        Mapped::Ideal(tu) => (
            "ideal",
            tu.t,
            tu.sigma,
            1.0,
            harmonic_drift(f0, n, |k| {
                ideal_mode_frequency(n as usize, tu.fs, tu.sigma, k)
            }),
            "a heavily damped string's low partials are pulled down further than its high ones \
             — physics, not the grid; a longer ring time brings them in.",
        ),
        Mapped::Flexible(tu) => (
            "flexible",
            tu.t,
            tu.sigma0,
            tu.lam,
            harmonic_drift(f0, n, |k| {
                mode_frequency(l, n as usize, fs, m.theta, tu.c, 0.0, tu.sigma0, 0.0, k)
            }),
            "the grid flattens the partials above the fundamental; a finer grid (N) or a higher \
             sample rate brings them in.",
        ),
        Mapped::Damped(_) => unreachable!("the damped mappings have their own block"),
    };
    let warn = horizon < reach;
    let message = if warn {
        format!(
            "the partials above f0 run {} of m·f0, up to {worst:+.1} cents (partial {worst_mode}); \
             in tune only up to partial {horizon}: {why}",
            if worst > 0.0 { "sharp" } else { "flat" }
        )
    } else {
        format!("every partial up to {reach} is within {HORIZON_CENTS} cents of m·f0.")
    };
    json!({
        "kind": kind,
        "f0": num(f0),
        "t60": num(m.targets.t60_low),
        "T": num(t),
        "sigma0": num(sigma),
        "fs": num(fs),
        "lambda": num(lam),
        "horizon": int(horizon as i64),
        "worst_drift_cents": num(round_nd(worst, 2)),
        "worst_drift_mode": int(worst_mode as i64),
        "warn": warn,
        "message": message,
    })
}

/// [`tuning_block`] for a musical record outside a [`Built`] — the barrier pages build their own
/// strings (`contact.rs`).
pub(crate) fn tuning_block_for(m: &Musical, l: f64, n: i64, fs: f64) -> Value {
    let tu = match &m.map {
        Mapped::Damped(tu) => tu,
        _ => return harmonic_tuning_block(m, l, n, fs),
    };
    let tg = &m.targets;
    let d = partial_drift(l, n as usize, fs, m.theta, tu, tg);
    let warn = d.warn(tu.anchor);
    let message = if warn {
        format!(
            "partials between f0 and partial {} drift off the target law, up to {:.1} cents \
             (partial {}); in tune only up to partial {}. A finer grid (N) or a higher sample \
             rate brings them in.",
            tu.anchor, d.worst_between, d.worst_mode, d.horizon
        )
    } else {
        format!(
            "every partial up to the anchor is within {} cents of the target law.",
            physsynth_core::string_tuning::HORIZON_CENTS
        )
    };
    let mut block = json!({
        "f0": num(tg.f0),
        "inharmonicity": num(tg.inharmonicity),
        "t60_low": num(tg.t60_low),
        "t60_high": num(tg.t60_high),
        "f_ref": num(tg.f_ref),
        "anchor": int(tu.anchor as i64),
        "f_anchor": num(round_nd(stretched_partial(tg.f0, tg.inharmonicity, tu.anchor), 3)),
        "T": num(tu.t),
        "kappa": num(tu.kappa),
        "sigma0": num(tu.sigma0),
        "sigma1": num(tu.sigma1),
        "horizon": int(d.horizon as i64),
        "worst_drift_cents": num(round_nd(d.worst_between, 2)),
        "worst_drift_mode": int(d.worst_mode as i64),
        "warn": warn,
        "message": message,
    });
    if let Some(g) = &m.gauge {
        block["gauge"] = json!({
            "material": g.material.name,
            "diameter_mm": num(round_nd(g.diameter * 1e3, 4)),
            "rho": num(g.g.rho),
            "EA": num(g.g.ea),
            "E_gpa": num(round_nd(g.g.youngs / 1e9, 4)),
            "stress_mpa": num(round_nd(g.g.stress / 1e6, 3)),
            "gauge_kappa": num(g.g.gauge_kappa),
            "gauge_inharmonicity": num(g.g.gauge_inharmonicity),
            "source": g.material.source,
            "breaking_mpa": num(g.material.breaking_mpa),
            "modulus_extrapolated": g.g.modulus_extrapolated,
            "past_breaking": g.g.past_breaking,
            "warnings": stress_warnings(g),
        });
    }
    block
}

/// A core constructor's refusal, surfaced as `serialize.py`'s `except ValueError` did.
pub(crate) fn construction<E: std::fmt::Display>(e: E) -> Refusal {
    Refusal::Construction(e.to_string())
}

/// Construct a fresh string from params. The check order is the reference's, because a request
/// that is wrong in two ways must be refused for the same one.
pub fn build_resonator(p: &Value) -> Result<Built, Refusal> {
    build_resonator_with(p, false)
}

/// [`build_resonator`] with both losses zeroed on a musical build — the tension page's lossless
/// single-mode measurements. Its physics path zeroes the `sigma` keys itself, so this changes
/// nothing there.
pub fn build_resonator_lossless(p: &Value) -> Result<Built, Refusal> {
    build_resonator_with(p, true)
}

fn build_resonator_with(p: &Value, lossless: bool) -> Result<Built, Refusal> {
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
        "ideal" if is_musical(p) => build_musical_ideal(p, l, rho, n),
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
                tuning: None,
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
                tuning: None,
            })
        }
        "damped" | "bow" | "tension" if is_musical(p) => {
            build_musical(p, &model, l, rho, n, lossless)
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
                tuning: None,
            })
        }
        "tension" => {
            let s0 = fnum(p, "sigma0", 0.0)?;
            let s1 = fnum(p, "sigma1", 0.0)?;
            let ea = fnum(p, "EA", crate::tension::TENSION_EA_DEFAULT)?;
            if !(0.0..=crate::tension::TENSION_EA_MAX).contains(&ea) {
                return Err(Refusal::Param(format!(
                    "EA must be in [0, {:.0}] N, got {}.",
                    crate::tension::TENSION_EA_MAX,
                    physsynth_core::fmt::py_float(ea)
                )));
            }
            let kappa = fnum(p, "kappa", 0.0)?;
            let theta = fnum(p, "theta", 0.28)?;
            let params = nonlinear::Params::new(
                l,
                t,
                rho,
                fs,
                n,
                kappa,
                ea,
                s0,
                s1,
                theta,
                nonlinear::TENSION_TOL_DEFAULT,
                true,
            )
            .map_err(construction)?;
            Ok(Built {
                res: StringRes::Tension(TensionModulatedString::new(params)),
                c,
                l,
                n,
                fs,
                sigma_zero: s0 == 0.0 && s1 == 0.0,
                oracle_2sigma: 2.0 * s0,
                tuning: None,
            })
        }
        "bow" => {
            use crate::bow as b;
            let s0 = fnum(p, "sigma0", b::BOW_SIGMA0_DEFAULT)?;
            let s1 = fnum(p, "sigma1", b::BOW_SIGMA1_DEFAULT)?;
            let settings = bow_settings(p)?;
            let kappa = fnum(p, "kappa", 0.0)?;
            let theta = fnum(p, "theta", 0.28)?;
            let sp = damped::Params::new(l, t, rho, fs, n, kappa, s0, s1, theta, true)
                .map_err(construction)?;
            let bowed = bow_on(p, DampedStiffString::new(sp), settings)?;
            Ok(Built {
                res: StringRes::Bow(Box::new(bowed)),
                c,
                l,
                n,
                fs,
                sigma_zero: s0 == 0.0 && s1 == 0.0,
                oracle_2sigma: 2.0 * s0,
                tuning: None,
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
fn partials_block(
    pickup: &[f64],
    fs: f64,
    model: &str,
    p: &Value,
    c: f64,
    l: f64,
    targets: Option<&Targets>,
) -> Value {
    let analytic: Vec<f64> = if let Some(tg) = targets {
        // The musical domain: the reference is the law the user asked for, not a kappa slider.
        (1..=N_PARTIALS)
            .map(|m| stretched_partial(tg.f0, tg.inharmonicity, m))
            .collect()
    } else if model == "stiff" || model == "damped" {
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
    // The musical domain's fundamental is its target; c / 2L ignores the stiffness it maps to.
    let f1_base = b.tuning.as_ref().map_or(c / (2.0 * l), |m| m.targets.f0);
    let pickup_idx = round_int(pickup_frac * n as f64).max(1).min(n - 1) as usize;

    // -- audio run: full duration, pickup, no snapshots
    let x = b.res.grid();
    let pluck = triangular_pluck(&x, l, pluck_frac * l, amplitude).map_err(construction)?;
    b.res.set_displacement(&pluck);
    let n_audio = round_int(audio_dur * fs).max(1);
    let audio_run = simulate(b.res.resonator(), n_audio as usize, Some(pickup_idx), 0)
        .map_err(Refusal::Internal)?;
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
    .map_err(Refusal::Internal)?;
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
    let mut payload = json!({
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
            "partials": partials_block(
                pickup,
                fs,
                &model,
                p,
                c,
                l,
                b.tuning.as_ref().map(|m| &m.targets),
            ),
        },
    });
    if let Some(t) = tuning_block(&b) {
        payload["tuning"] = t;
    }
    Ok(payload)
}
