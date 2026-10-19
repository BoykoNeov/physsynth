//! Musical controls for the damped stiff string (model #3) — `docs/dev/parameter-mapping-plan.md`.
//!
//! Raw physics (tension, `kappa`, `sigma0`, `sigma1`) is not a musician's interface
//! (`docs/dev/scientific-hurdles.md` §12). This module maps five musical targets — the pitch of
//! the fundamental, an inharmonicity, the ring time of the fundamental, the ring time high up, and
//! where "high up" is — onto the four coefficients [`crate::string_damped::Params::new`] takes.
//! It is an optional helper: `Params::new` and the raw coefficients are untouched, and the mapping
//! refuses only what model #3 cannot represent.
//!
//! # Corrected to the scheme, not to the continuum
//!
//! The targets are met by the *simulated* string, not by a continuum string the scheme then
//! detunes. Mode `m` (`sin(m pi x / L)`, an exact discrete eigenvector under the supported
//! boundary) obeys the two-step recurrence `a z^2 + b z + cc = 0` with, writing `X = k^2 Q_m`,
//! `Y = k sigma_m`, `Q_m = c^2 p_m^2 + kappa^2 p_m^4`, `sigma_m = sigma0 + sigma1 p_m^2` and `p_m^2`
//! the discrete eigenvalue of `-delta_xx`:
//!
//! ```text
//! a  = 1 + theta X + Y
//! b  = -2 + (1 - 2 theta) X
//! cc = 1 + theta X - Y
//! ```
//!
//! A mode that sounds at `f` and rings for `T60` (amplitude down 60 dB) is the root pair
//! `z = r e^{+-i phi}`, `phi = 2 pi f k`, `r = 10^(-3k/T60)`. The product and the sum of the roots,
//! `cc/a = r^2` and `-b/a = 2 r cos(phi)`, are **linear** in `X` and `Y`:
//!
//! ```text
//! theta (1 - r^2) X - (1 + r^2) Y                 = r^2 - 1
//! -[(1 - 2 theta) + 2 theta r cos(phi)] X - 2 r cos(phi) Y = 2 r cos(phi) - 2
//! ```
//!
//! so one `(f, T60)` target fixes one mode's `(Q_m, sigma_m)` exactly. Two anchors — mode 1 and
//! the mode `M` nearest the reference frequency — give `(c^2, kappa^2)` and `(sigma0, sigma1)` by
//! two more 2x2 solves. No iteration, and the loss's own pull on the pitch is inside the same two
//! equations. At `r = 1` they reduce to the theta-scheme's dispersion relation
//! `4 s = X (1 - 4 theta s)`, `s = sin^2(phi/2)`.
//!
//! # What "exact" covers
//!
//! **Two modes, not all of them.** Modes 1 and `M` land on their targets to rounding; every other
//! mode lands where the scheme puts it, off the stretched law by the scheme's own dispersion — the
//! pitch error of hurdle §15 and the rate suppression of §4, which are one factor and cannot be
//! removed by choosing coefficients. [`partial_drift`] measures that residual so a caller can show
//! it. The human's call (plan §3): a string whose in-between partials drift past
//! [`HORIZON_CENTS`] is built anyway and the drift reported, not refused.
//!
//! A consequence worth knowing: an inharmonicity of `0` does **not** give `kappa = 0`. The grid and
//! the theta average both flatten high modes, so a perfectly harmonic anchor needs a little
//! stiffness to hold it up. The returned `kappa` is the scheme's, not the textbook's.
//!
//! The textbook mapping — `c = 2 L f0 / sqrt(1 + B)`, `kappa = sqrt(B) c L / pi`, and
//! `physsynth-analysis`'s continuum `loss_coefficients_from_t60` — is what this improves on. It is
//! close on the fundamental and wrong at the anchor; the plan's §7 has the numbers.
//!
//! # The tension string: a material and a gauge (`docs/dev/string-mapping-plan.md` §5, §9)
//!
//! Model #9 is model #3 plus an axial stiffness `EA`, and at `EA = 0` it *is* model #3 bit for bit;
//! a gentle pluck feels no `EA` at all. So [`tune_tension_string`] is an adapter: a material and a
//! diameter give the linear density and `EA`, and [`tune_damped_string`] does the rest. The pitch
//! met is the **settled** note — a hard pluck starts sharp and glides onto it, by an amount the
//! gauge sets. The inharmonicity stays a free target (the human's call, plan §8.4); the bending
//! stiffness the gauge alone implies is returned beside the mapped `kappa`, not imposed.

use crate::string_nonlinear::string_coefficients_from_material;
use std::f64::consts::{LN_10, PI};

/// The drift, in cents, inside which a partial counts as in tune for [`partial_drift`].
pub const HORIZON_CENTS: f64 = 5.0;

/// What a musician sets.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Targets {
    /// Pitch of the fundamental (Hz).
    pub f0: f64,
    /// Inharmonicity `B` of the stretched partial law [`stretched_partial`]; `0` is harmonic.
    pub inharmonicity: f64,
    /// Ring time of the fundamental: seconds for its amplitude to fall 60 dB.
    pub t60_low: f64,
    /// Ring time at the anchor mode (the partial nearest `f_ref`), in seconds.
    pub t60_high: f64,
    /// Where "high" is measured (Hz); picks the anchor mode. Must lie above `f0`.
    pub f_ref: f64,
}

/// The coefficients that meet the targets, plus what a caller shows alongside them.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tuning {
    /// Tension (N), `c^2 rho`.
    pub t: f64,
    /// Stiffness coefficient (m^2/s).
    pub kappa: f64,
    /// Frequency-independent loss (1/s).
    pub sigma0: f64,
    /// Frequency-dependent loss (m^2/s).
    pub sigma1: f64,
    /// Wave speed `sqrt(T / rho)` (m/s).
    pub c: f64,
    /// Courant number `c k / h` — a read-out; model #3 is stable at any value for `theta >= 1/4`.
    pub lam: f64,
    /// The anchor mode `M`, the second mode met exactly.
    pub anchor: usize,
}

/// Why a set of targets cannot be met by model #3.
#[derive(Debug, Clone, PartialEq)]
pub enum TuningError {
    /// An input was zero, negative or not finite. Names the input.
    NonPositive(&'static str),
    /// The inharmonicity is not finite, or makes the stretched law imaginary at a mode used.
    BadInharmonicity,
    /// `theta` outside `(0, 1]`.
    BadTheta(f64),
    /// Fewer than three segments: there is no second mode to anchor on.
    TooFewSegments,
    /// `f_ref` at or below `f0` — one anchor cannot separate two unknowns.
    RefNotAboveFundamental,
    /// No positive stiffness sounds at this mode's target at this `fs` and `theta`.
    Unreachable {
        /// The mode whose target the scheme cannot produce.
        mode: usize,
    },
    /// The targets need `c^2 <= 0`.
    NoWaveSpeed,
    /// The targets need `kappa^2 < 0`: the anchor sits below what a flexible string gives.
    ShrinkingPartials,
    /// The targets need a negative loss: ring time rising with frequency.
    RisingDecay {
        /// The `sigma0` the targets imply.
        sigma0: f64,
        /// The `sigma1` the targets imply.
        sigma1: f64,
    },
}

impl std::fmt::Display for TuningError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TuningError::NonPositive(name) => write!(f, "{name} must be positive and finite."),
            TuningError::BadInharmonicity => write!(
                f,
                "inharmonicity must be finite and keep 1 + B m^2 > 0 at the modes used."
            ),
            TuningError::BadTheta(t) => write!(f, "theta must be in (0, 1], got {t}."),
            TuningError::TooFewSegments => {
                write!(
                    f,
                    "N must be >= 3 (need a second mode to anchor the high ring time)."
                )
            }
            TuningError::RefNotAboveFundamental => {
                write!(f, "the reference frequency must lie above the fundamental.")
            }
            TuningError::Unreachable { mode } => write!(
                f,
                "mode {mode}'s target pitch is beyond what this grid and sample rate can produce; \
                 raise fs or lower the pitch."
            ),
            TuningError::NoWaveSpeed => {
                write!(
                    f,
                    "these targets need a non-positive tension; no string sounds them."
                )
            }
            TuningError::ShrinkingPartials => write!(
                f,
                "these targets need negative stiffness (partials closer than a flexible string \
                 gives); raise the inharmonicity."
            ),
            TuningError::RisingDecay { sigma0, sigma1 } => write!(
                f,
                "these targets need negative loss (sigma0 = {sigma0:.4e}, sigma1 = {sigma1:.4e}); \
                 the high ring time must not exceed what the low one allows."
            ),
        }
    }
}

impl std::error::Error for TuningError {}

/// The stretched partial law the targets are written in: `f_m = m f0 sqrt((1 + B m^2)/(1 + B))`.
///
/// `f_1 = f0` exactly. This is the continuum stiff string's law with `B` its inharmonicity.
pub fn stretched_partial(f0: f64, inharmonicity: f64, m: usize) -> f64 {
    let m = m as f64;
    m * f0 * ((1.0 + inharmonicity * m * m) / (1.0 + inharmonicity)).sqrt()
}

/// Discrete eigenvalue `p^2 = (4/h^2) sin^2(m pi / 2N)` of `-delta_xx` for mode `m`.
fn mode_p2(l: f64, n: usize, m: usize) -> f64 {
    let h = l / n as f64;
    let s = (m as f64 * PI / (2.0 * n as f64)).sin();
    4.0 * s * s / (h * h)
}

/// The anchor: the mode in `2..=n-1` whose stretched partial lies nearest `f_ref` (lower on a tie).
pub fn anchor_mode(f0: f64, inharmonicity: f64, f_ref: f64, n: usize) -> usize {
    let mut best = 2;
    let mut best_d = f64::INFINITY;
    for m in 2..n {
        let d = (stretched_partial(f0, inharmonicity, m) - f_ref).abs();
        if d < best_d {
            best = m;
            best_d = d;
        }
    }
    best
}

/// One mode's `(X, Y) = (k^2 Q_m, k sigma_m)` from its target `(f, T60)` — module header.
///
/// `r^2 - 1` and `r cos(phi) - 1` are formed without cancellation (`exp_m1`, a half-angle sine):
/// at a low pitch and a long ring time both are ~1e-5, so the plain spelling would lose about
/// five of the sixteen digits. That is a precision choice, **not a pinned one**: planting the
/// plain `r * r - 1` (2026-10-18) turned nothing red, because the ~1e-11 it costs sits below
/// every round-trip bar's measurement floor (`tests/string_tuning.rs`).
fn mode_xy(f: f64, t60: f64, k: f64, theta: f64, mode: usize) -> Result<(f64, f64), TuningError> {
    let phi = 2.0 * PI * f * k;
    let ln_r = -3.0 * k * LN_10 / t60;
    let r = ln_r.exp();
    let r2m1 = (2.0 * ln_r).exp_m1();
    let half = (0.5 * phi).sin();
    let rc = r * phi.cos();
    // r cos(phi) - 1 = r (cos(phi) - 1) + (r - 1)
    let rcm1 = -2.0 * r * half * half + ln_r.exp_m1();

    let (a11, a12, b1) = (-theta * r2m1, -(2.0 + r2m1), r2m1);
    let (a21, a22, b2) = (
        -((1.0 - 2.0 * theta) + 2.0 * theta * rc),
        -2.0 * rc,
        2.0 * rcm1,
    );
    let det = a11 * a22 - a12 * a21;
    let x = (b1 * a22 - a12 * b2) / det;
    let y = (a11 * b2 - a21 * b1) / det;
    if !(x > 0.0 && x.is_finite() && y.is_finite()) {
        return Err(TuningError::Unreachable { mode });
    }
    Ok((x, y))
}

/// Map musical targets to model #3's coefficients, met exactly at mode 1 and the anchor.
///
/// `l`, `rho`, `fs`, `n` and `theta` are the string's own inputs, unchanged; the result's `t`,
/// `kappa`, `sigma0`, `sigma1` complete a [`crate::string_damped::Params::new`] call.
pub fn tune_damped_string(
    l: f64,
    rho: f64,
    fs: f64,
    n: usize,
    theta: f64,
    targets: &Targets,
) -> Result<Tuning, TuningError> {
    let positive = [
        ("L", l),
        ("rho", rho),
        ("fs", fs),
        ("f0", targets.f0),
        ("t60_low", targets.t60_low),
        ("t60_high", targets.t60_high),
        ("f_ref", targets.f_ref),
    ];
    for (name, v) in positive {
        if !(v > 0.0 && v.is_finite()) {
            return Err(TuningError::NonPositive(name));
        }
    }
    if !(theta > 0.0 && theta <= 1.0) {
        return Err(TuningError::BadTheta(theta));
    }
    if n < 3 {
        return Err(TuningError::TooFewSegments);
    }
    if targets.f_ref <= targets.f0 {
        return Err(TuningError::RefNotAboveFundamental);
    }
    let b = targets.inharmonicity;
    let anchor = anchor_mode(targets.f0, b, targets.f_ref, n);
    let ma = anchor as f64;
    if !(b.is_finite() && 1.0 + b > 0.0 && 1.0 + b * ma * ma > 0.0) {
        return Err(TuningError::BadInharmonicity);
    }

    let k = 1.0 / fs;
    let f_anchor = stretched_partial(targets.f0, b, anchor);
    let (x1, y1) = mode_xy(targets.f0, targets.t60_low, k, theta, 1)?;
    let (xm, ym) = mode_xy(f_anchor, targets.t60_high, k, theta, anchor)?;
    let (q1, qm) = (x1 / (k * k), xm / (k * k));
    let (s1, sm) = (y1 / k, ym / k);

    // Q_m = c^2 p_m^2 + kappa^2 p_m^4 and sigma_m = sigma0 + sigma1 p_m^2 at the two anchors.
    let (p1, pm) = (mode_p2(l, n, 1), mode_p2(l, n, anchor));
    let det = p1 * pm * (pm - p1);
    let c2 = (q1 * pm * pm - qm * p1 * p1) / det;
    let kappa2 = (qm * p1 - q1 * pm) / det;
    let sigma1 = (sm - s1) / (pm - p1);
    let sigma0 = s1 - sigma1 * p1;

    if c2.is_nan() || c2 <= 0.0 {
        return Err(TuningError::NoWaveSpeed);
    }
    if kappa2 < 0.0 {
        return Err(TuningError::ShrinkingPartials);
    }
    if sigma0 < 0.0 || sigma1 < 0.0 {
        return Err(TuningError::RisingDecay { sigma0, sigma1 });
    }
    let c = c2.sqrt();
    Ok(Tuning {
        t: c2 * rho,
        kappa: kappa2.sqrt(),
        sigma0,
        sigma1,
        c,
        lam: c * k * n as f64 / l,
        anchor,
    })
}

/// The pitch (Hz) model #3 gives mode `m` — the angle of its damped root pair — or `None` when
/// the mode does not oscillate (overdamped, or past the scheme's highest frequency).
#[allow(clippy::too_many_arguments)]
pub fn mode_frequency(
    l: f64,
    n: usize,
    fs: f64,
    theta: f64,
    c: f64,
    kappa: f64,
    sigma0: f64,
    sigma1: f64,
    m: usize,
) -> Option<f64> {
    let k = 1.0 / fs;
    let p2 = mode_p2(l, n, m);
    let x = k * k * (c * c * p2 + kappa * kappa * p2 * p2);
    let y = k * (sigma0 + sigma1 * p2);
    let base = 1.0 + theta * x;
    let (a, b, cc) = (base + y, -2.0 + (1.0 - 2.0 * theta) * x, base - y);
    if cc <= 0.0 {
        return None;
    }
    let cos_phi = -b / (2.0 * (a * cc).sqrt());
    if cos_phi.is_nan() || cos_phi.abs() >= 1.0 {
        return None;
    }
    Some(cos_phi.acos() / (2.0 * PI * k))
}

/// How far the partials the mapping does *not* pin drift from the stretched law.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Drift {
    /// The highest mode `h` such that every mode `1..=h` is within [`HORIZON_CENTS`].
    pub horizon: usize,
    /// The largest `|drift|` (cents) among the modes strictly between 1 and the anchor; `0` when
    /// there are none.
    pub worst_between: f64,
    /// The mode where `worst_between` occurs (`0` when there is none).
    pub worst_mode: usize,
}

impl Drift {
    /// True when some partial below the anchor is out of tune — the human's "warn" case.
    pub fn warn(&self, anchor: usize) -> bool {
        self.horizon < anchor
    }
}

/// Measure [`Drift`] for a tuning from the scheme's own per-mode pitch ([`mode_frequency`]).
pub fn partial_drift(
    l: f64,
    n: usize,
    fs: f64,
    theta: f64,
    tuning: &Tuning,
    targets: &Targets,
) -> Drift {
    let mut horizon = 0;
    let mut worst_between = 0.0f64;
    let mut worst_mode = 0;
    for m in 1..n {
        let f = mode_frequency(
            l,
            n,
            fs,
            theta,
            tuning.c,
            tuning.kappa,
            tuning.sigma0,
            tuning.sigma1,
            m,
        );
        let Some(f) = f else { break };
        let target = stretched_partial(targets.f0, targets.inharmonicity, m);
        let cents = (1200.0 * (f / target).log2()).abs();
        if horizon == m - 1 && cents <= HORIZON_CENTS {
            horizon = m;
        }
        if m > 1 && m < tuning.anchor && cents > worst_between {
            worst_between = cents;
            worst_mode = m;
        }
    }
    Drift {
        horizon,
        worst_between,
        worst_mode,
    }
}

// == the tension string: a material and a gauge ==================================================

/// How a string material's Young's modulus is known.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Modulus {
    /// One value (Pa), independent of how hard the string is pulled.
    Fixed(f64),
    /// Nylon's stress-dependent fit (Lynch-Aird & Woodhouse 2017, Eq. 19): `E = a + b σ + c ρ` GPa
    /// with `σ` the string's stress in MPa and `ρ` its density in kg/m³.
    StressFit {
        /// Constant term (GPa).
        a: f64,
        /// Per MPa of stress (GPa/MPa).
        b: f64,
        /// Per kg/m³ of density (GPa m³/kg).
        c: f64,
        /// The stress range (MPa) the fit's data covers; outside it the fit is an extrapolation,
        /// which the human chose to warn about, not refuse (string mapping plan §12).
        measured_mpa: (f64, f64),
    },
}

impl Modulus {
    /// Young's modulus (Pa) at stress `stress` (Pa) for a material of density `density` (kg/m³).
    pub fn at(&self, stress: f64, density: f64) -> f64 {
        match *self {
            Modulus::Fixed(e) => e,
            Modulus::StressFit { a, b, c, .. } => (a + b * (stress * 1e-6) + c * density) * 1e9,
        }
    }

    /// True when `stress` (Pa) lies outside the range a fit was measured over. A fixed modulus is
    /// never an extrapolation.
    pub fn extrapolated(&self, stress: f64) -> bool {
        match *self {
            Modulus::Fixed(_) => false,
            Modulus::StressFit {
                measured_mpa: (lo, hi),
                ..
            } => !(lo..=hi).contains(&(stress * 1e-6)),
        }
    }
}

/// A string material a caller can name.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StringMaterial {
    /// The name a caller selects it by.
    pub name: &'static str,
    /// Volumetric density (kg/m³).
    pub density: f64,
    /// Young's modulus.
    pub modulus: Modulus,
    /// Where the values come from.
    pub source: &'static str,
    /// The stress (MPa) past which a real string of this material breaks — a warning, never a
    /// refusal: a model string does not snap (string mapping plan §12).
    pub breaking_mpa: f64,
    /// Where the breaking stress comes from.
    pub breaking_source: &'static str,
}

/// Lynch-Aird & Woodhouse, "Mechanical Properties of Nylon Harp Strings", *Materials* 10 (2017)
/// 497. Table 1 gives the unstretched densities (1067–1159 kg/m³ in two groups); Eq. 19 the
/// tension-modulation modulus of the lower-density group, `r² = 0.983`. One string's modulus runs
/// from about 4 to over 10 GPa across its stress range, so no fixed value would serve. Its own
/// tension-modulation example spans 30 MPa (about 3 GPa) to 220 MPa (nearly 11 GPa), the range
/// [`Modulus::StressFit::measured_mpa`] carries.
const NYLON_SRC: &str = "Lynch-Aird & Woodhouse, Materials 10 (2017) 497: Table 1, Eq. 19";

/// Lynch-Aird & Woodhouse, "The sound of a harp string" (euphonics.org, 2023), Fig. 1: the top gut
/// harp strings run at about 275 MPa, "very close to their breaking stress limit", and "the
/// breaking stress limit for nylon strings is fairly similar to that for gut". So about 280 MPa
/// for both — read from a plot, an approximate figure.
const HARP_BREAKING_SRC: &str =
    "Lynch-Aird & Woodhouse, The sound of a harp string (2023), Fig. 1: about 280 MPa";

/// The materials a caller can name (plan §9). Plain, unwound strings only: a wound string's mass
/// sits in its winding and its `EA` in its core, which one density and one modulus cannot carry.
pub const STRING_MATERIALS: &[StringMaterial] = &[
    // MakeItFrom's ASTM A228 music-wire page (cold-worked): 190 GPa, 7.8 g/cm³, ultimate tensile
    // strength 2,450 MPa.
    StringMaterial {
        name: "steel",
        density: 7800.0,
        modulus: Modulus::Fixed(190.0e9),
        source: "MakeItFrom, ASTM A228 music wire",
        breaking_mpa: 2450.0,
        breaking_source: "MakeItFrom, ASTM A228 music wire: ultimate tensile strength 2,450 MPa",
    },
    // The lower-density group of Table 1 (1072–1097 kg/m³), whose fit Eq. 19 is.
    StringMaterial {
        name: "nylon",
        density: 1080.0,
        modulus: Modulus::StressFit {
            a: 32.0,
            b: 0.0353,
            c: -0.0269,
            measured_mpa: (30.0, 220.0),
        },
        source: NYLON_SRC,
        breaking_mpa: 280.0,
        breaking_source: HARP_BREAKING_SRC,
    },
    // Lynch-Aird & Woodhouse, "Comparison of Mechanical Properties of Natural Gut and Synthetic
    // Polymer Harp Strings", Materials 11 (2018) 2160: Table 1's gut densities are 1320–1325
    // kg/m³, and gut's modulus is flat in stress, inside the 5.5–6.5 GPa the paper cites.
    StringMaterial {
        name: "gut",
        density: 1322.0,
        modulus: Modulus::Fixed(6.0e9),
        source: "Lynch-Aird & Woodhouse, Materials 11 (2018) 2160: Table 1, and its cited range",
        breaking_mpa: 280.0,
        breaking_source: HARP_BREAKING_SRC,
    },
];

/// Look a material up by name.
pub fn string_material(name: &str) -> Option<&'static StringMaterial> {
    STRING_MATERIALS.iter().find(|m| m.name == name)
}

/// The tension string's coefficients from a material, a gauge and the musical targets.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GaugeTuning {
    /// What [`tune_damped_string`] gave: `T`, `kappa`, `sigma0`, `sigma1`, the anchor.
    pub tuning: Tuning,
    /// Linear density (kg/m), `density * pi r^2`.
    pub rho: f64,
    /// Axial stiffness (N), `E pi r^2`.
    pub ea: f64,
    /// Young's modulus used (Pa) — for nylon, its value at [`GaugeTuning::stress`].
    pub youngs: f64,
    /// The string's stress `T / (pi r^2)` (Pa).
    pub stress: f64,
    /// The bending stiffness the gauge alone implies, `sqrt(E I / rho)` (m²/s) — a read-out.
    pub gauge_kappa: f64,
    /// The inharmonicity that bending stiffness gives, `pi^2 kappa^2 / (c^2 L^2)` — a read-out.
    pub gauge_inharmonicity: f64,
    /// True when [`GaugeTuning::youngs`] came from a fit outside the stress it was measured over
    /// — a warning, not a refusal (string mapping plan §12).
    pub modulus_extrapolated: bool,
    /// True when the tuned stress is past the material's breaking stress — a warning, not a
    /// refusal: a model string does not snap (plan §12).
    pub past_breaking: bool,
}

/// Map a material, a diameter and musical targets onto model #9 (module header).
///
/// `material` comes from [`STRING_MATERIALS`] or is built by the caller; `diameter` is in metres.
/// The result's `rho`, `ea` and `tuning`'s `t`, `kappa`, `sigma0`, `sigma1` complete a
/// [`crate::string_nonlinear::Params::new`] call.
#[allow(clippy::too_many_arguments)]
pub fn tune_tension_string(
    material: &StringMaterial,
    diameter: f64,
    l: f64,
    fs: f64,
    n: usize,
    theta: f64,
    targets: &Targets,
) -> Result<GaugeTuning, TuningError> {
    if !(diameter > 0.0 && diameter.is_finite()) {
        return Err(TuningError::NonPositive("diameter"));
    }
    if !(material.density > 0.0 && material.density.is_finite()) {
        return Err(TuningError::NonPositive("density"));
    }
    let radius = 0.5 * diameter;
    let area = PI * radius * radius;
    let rho = material.density * area;
    let tuning = tune_damped_string(l, rho, fs, n, theta, targets)?;
    let stress = tuning.t / area;
    let youngs = material.modulus.at(stress, material.density);
    if !(youngs > 0.0 && youngs.is_finite()) {
        return Err(TuningError::NonPositive("Young's modulus"));
    }
    let co = string_coefficients_from_material(youngs, radius, material.density, tuning.t)
        .map_err(|_| TuningError::NonPositive("Young's modulus"))?;
    Ok(GaugeTuning {
        tuning,
        rho,
        ea: co.ea,
        youngs,
        stress,
        gauge_kappa: co.kappa,
        gauge_inharmonicity: PI * PI * co.kappa * co.kappa / (tuning.c * tuning.c * l * l),
        modulus_extrapolated: material.modulus.extrapolated(stress),
        past_breaking: stress * 1e-6 > material.breaking_mpa,
    })
}

// == the ideal string, at lambda = 1 ==============================================================

/// The ideal string's coefficients from a pitch and a ring time (string mapping plan §2, §13).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IdealTuning {
    /// Tension (N), `rho c^2`.
    pub t: f64,
    /// The loss coefficient (1/s).
    pub sigma: f64,
    /// The sample rate the string runs at (Hz): it follows the pitch, so that `lambda = 1`.
    pub fs: f64,
    /// Wave speed (m/s), `L / (N k)`.
    pub c: f64,
}

/// Map a pitch and a ring time onto model #1 at `lambda = 1` — the human's call (plan §8.2): the
/// grid stays at its exact setting and the sample rate follows the pitch.
///
/// The explicit scheme's loss is the centred `1 +- sigma k` of model #3, so mode `m` obeys the
/// damped string's recurrence at `theta = 0`. At `lambda = 1`, `k^2 c^2 p_m^2 = 4 sin^2(m pi/2N)`,
/// and the root pair gives
///
/// ```text
/// r^2 = (1 - sigma k) / (1 + sigma k)                (the same for EVERY mode)
/// cos(phi_m) = cos(m pi / N) cosh(a k),   a = 3 ln 10 / T60,   sigma k = tanh(a k)
/// ```
///
/// So the ring time is met by every partial exactly, and the fundamental's pitch fixes the time
/// step `k` through `cos(2 pi f0 k) = cos(pi / N) cosh(a k)` — one equation in one unknown, solved
/// by bisection on `(0, 1/(2 f0 N)]`, where it always has a root (the left side starts above the
/// right and ends below it). Then `fs = 1/k`, `c = L/(N k)`, `T = rho c^2`, `sigma = tanh(a k)/k`.
///
/// Lossless, every partial is exactly harmonic. With loss the fundamental is pinned and the
/// partials above it run **sharp** of `m f0`: a damped oscillator's frequency falls by a larger
/// share at a low pitch than at a high one, which is physics, not the scheme (plan §13). The
/// residual is what [`ideal_mode_frequency`] reports.
pub fn tune_ideal_string(
    l: f64,
    rho: f64,
    n: usize,
    f0: f64,
    t60: f64,
) -> Result<IdealTuning, TuningError> {
    for (name, v) in [("L", l), ("rho", rho), ("f0", f0), ("t60", t60)] {
        if !(v > 0.0 && v.is_finite()) {
            return Err(TuningError::NonPositive(name));
        }
    }
    if n < 2 {
        return Err(TuningError::TooFewSegments);
    }
    let a = 3.0 * LN_10 / t60;
    let half = (PI / (2.0 * n as f64)).sin();
    let (s2, c1) = (half * half, 1.0 - 2.0 * half * half);
    // cos(2 pi f0 k) - cos(pi/N) cosh(a k), spelled through half-angles so that near cos = 1 the
    // difference keeps its digits: 1 - cos x = 2 sin^2(x/2), cosh y - 1 = 2 sinh^2(y/2).
    let g = |k: f64| {
        let sp = (PI * f0 * k).sin();
        let sh = (0.5 * a * k).sinh();
        2.0 * s2 - 2.0 * sp * sp - 2.0 * c1 * sh * sh
    };
    let (mut lo, mut hi) = (0.0f64, 1.0 / (2.0 * f0 * n as f64));
    loop {
        let mid = 0.5 * (lo + hi);
        if mid <= lo || mid >= hi {
            break;
        }
        if g(mid) > 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let k = 0.5 * (lo + hi);
    let c = l / (n as f64 * k);
    Ok(IdealTuning {
        t: rho * c * c,
        sigma: (a * k).tanh() / k,
        fs: 1.0 / k,
        c,
    })
}

/// The pitch (Hz) model #1 gives mode `m` at `lambda = 1` with loss `sigma` — or `None` past the
/// modes that oscillate.
pub fn ideal_mode_frequency(n: usize, fs: f64, sigma: f64, m: usize) -> Option<f64> {
    let k = 1.0 / fs;
    let y = sigma * k;
    // cos(phi) = cos(m pi/N) / sqrt(1 - y^2): from cc/a = r^2 and -b/a = 2 r cos(phi).
    let cos_phi = (m as f64 * PI / n as f64).cos() / (1.0 - y * y).sqrt();
    if cos_phi.abs() >= 1.0 {
        return None;
    }
    Some(cos_phi.acos() / (2.0 * PI * k))
}

// == the flexible string of the contact pages: mode 1 only ========================================

/// A flexible (`kappa = 0`), singly damped (`sigma1 = 0`) theta-scheme string from a pitch and a
/// ring time — the string the barrier pages (jawari, juari, fret) build (plan §1).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FlexibleTuning {
    /// Tension (N).
    pub t: f64,
    /// The loss coefficient (1/s).
    pub sigma0: f64,
    /// Wave speed (m/s).
    pub c: f64,
    /// Courant number `c k / h` — a read-out.
    pub lam: f64,
}

/// Map a pitch and a ring time onto a flexible model #3 string, exact at mode 1.
///
/// One mode, two unknowns (`c^2`, `sigma0`), and the same 2x2 per-mode solve as
/// [`tune_damped_string`]: mode 1's `(X, Y)` from its `(f0, T60)`, then `c^2 = X / (k^2 p_1^2)` and
/// `sigma0 = Y / k`. Every partial shares `sigma0` but not its ring time — the theta average
/// lengthens it up the series (hurdle §4) — and the partials above mode 1 fall flat of `m f0` by
/// the scheme's dispersion; [`mode_frequency`] reports both.
pub fn tune_flexible_string(
    l: f64,
    rho: f64,
    fs: f64,
    n: usize,
    theta: f64,
    f0: f64,
    t60: f64,
) -> Result<FlexibleTuning, TuningError> {
    for (name, v) in [("L", l), ("rho", rho), ("fs", fs), ("f0", f0), ("t60", t60)] {
        if !(v > 0.0 && v.is_finite()) {
            return Err(TuningError::NonPositive(name));
        }
    }
    if !(theta > 0.0 && theta <= 1.0) {
        return Err(TuningError::BadTheta(theta));
    }
    if n < 2 {
        return Err(TuningError::TooFewSegments);
    }
    let k = 1.0 / fs;
    let (x, y) = mode_xy(f0, t60, k, theta, 1)?;
    let c2 = x / (k * k * mode_p2(l, n, 1));
    let c = c2.sqrt();
    Ok(FlexibleTuning {
        t: rho * c2,
        sigma0: y / k,
        c,
        lam: c * k * n as f64 / l,
    })
}

// == the stiff string: one loss, an inharmonicity =================================================

/// Map a pitch, an inharmonicity and ONE ring time onto model #2 (the stiff string, one `sigma`)
/// — string mapping plan §3, §14.
///
/// Model #2 is model #3 at `sigma1 = 0`: the same theta-scheme matrix `(1 + sigma k) I - theta k^2
/// L`. So mode 1's `(f0, T60)` gives `(X_1, Y)` by [`tune_damped_string`]'s 2x2, and `Y = sigma k`
/// is then the same for EVERY mode. At the anchor `M` only the pitch is a target, and with `Y`
/// known, `cos(phi) = (2 - (1 - 2 theta) X) / (2 sqrt((1 + theta X)^2 - Y^2))` is one equation in
/// `X_M`. Squared, it is the quadratic
///
/// ```text
/// (beta^2 - 4 C theta^2) X^2 - (4 beta + 8 C theta) X + 4 (sin^2 phi + C Y^2) = 0,
/// beta = 1 - 2 theta,   C = cos^2 phi
/// ```
///
/// and a root is the anchor's only if `X > 0` and `2 - beta X` has the sign of `cos(phi)`.
/// **Exactly one ever qualifies.** Lossless, `cos(phi)` falls strictly as `X` rises (its derivative
/// is `-1 / (2 (1 + theta X)^2)`), so each pitch has one stiffness and the squaring's other root
/// belongs to the mirrored branch; the probe found one valid root in all 1,686 representable
/// configurations of 1,728 and two in none (plan §14). Were two ever to qualify, the mapping
/// refuses rather than guesses.
///
/// The returned [`Tuning`] has `sigma1 = 0`, so [`mode_frequency`] and [`partial_drift`] read it as
/// model #2. The anchor's ring time is not a target — one `sigma` leaves it to the scheme, which
/// lengthens it up the series (hurdle §4) — and [`stiff_mode_t60`] reports it.
#[allow(clippy::too_many_arguments)]
pub fn tune_stiff_string(
    l: f64,
    rho: f64,
    fs: f64,
    n: usize,
    theta: f64,
    f0: f64,
    inharmonicity: f64,
    t60: f64,
    f_ref: f64,
) -> Result<Tuning, TuningError> {
    for (name, v) in [
        ("L", l),
        ("rho", rho),
        ("fs", fs),
        ("f0", f0),
        ("t60", t60),
        ("f_ref", f_ref),
    ] {
        if !(v > 0.0 && v.is_finite()) {
            return Err(TuningError::NonPositive(name));
        }
    }
    if !(theta > 0.0 && theta <= 1.0) {
        return Err(TuningError::BadTheta(theta));
    }
    if n < 3 {
        return Err(TuningError::TooFewSegments);
    }
    if f_ref <= f0 {
        return Err(TuningError::RefNotAboveFundamental);
    }
    let b = inharmonicity;
    let anchor = anchor_mode(f0, b, f_ref, n);
    let ma = anchor as f64;
    if !(b.is_finite() && 1.0 + b > 0.0 && 1.0 + b * ma * ma > 0.0) {
        return Err(TuningError::BadInharmonicity);
    }

    let k = 1.0 / fs;
    let (x1, y) = mode_xy(f0, t60, k, theta, 1)?;
    let phi = 2.0 * PI * stretched_partial(f0, b, anchor) * k;
    let (s, cs) = (phi.sin(), phi.cos());
    let c2 = cs * cs;
    let beta = 1.0 - 2.0 * theta;
    let qa = beta * beta - 4.0 * c2 * theta * theta;
    let qb = -(4.0 * beta + 8.0 * c2 * theta);
    // 4 - 4 C (1 - Y^2), spelled as 4 (sin^2 phi + C Y^2) so a low anchor keeps its digits.
    let qc = 4.0 * (s * s + c2 * y * y);
    let disc = qb * qb - 4.0 * qa * qc;
    if disc < 0.0 {
        return Err(TuningError::Unreachable { mode: anchor });
    }
    // The cancellation-free pair: q = -(qb + sign(qb) sqrt(disc)) / 2, roots q / qa and qc / q.
    let q = -0.5 * (qb + qb.signum() * disc.sqrt());
    let valid: Vec<f64> = [q / qa, qc / q]
        .into_iter()
        .filter(|&x| x > 0.0 && x.is_finite() && (2.0 - beta * x).signum() == cs.signum())
        .collect();
    let &[xm] = valid.as_slice() else {
        return Err(TuningError::Unreachable { mode: anchor });
    };

    let (q1, qm) = (x1 / (k * k), xm / (k * k));
    let (p1, pm) = (mode_p2(l, n, 1), mode_p2(l, n, anchor));
    let det = p1 * pm * (pm - p1);
    let c2w = (q1 * pm * pm - qm * p1 * p1) / det;
    let kappa2 = (qm * p1 - q1 * pm) / det;
    if c2w.is_nan() || c2w <= 0.0 {
        return Err(TuningError::NoWaveSpeed);
    }
    if kappa2 < 0.0 {
        return Err(TuningError::ShrinkingPartials);
    }
    let c = c2w.sqrt();
    Ok(Tuning {
        t: c2w * rho,
        kappa: kappa2.sqrt(),
        sigma0: y / k,
        sigma1: 0.0,
        c,
        lam: c * k * n as f64 / l,
        anchor,
    })
}

/// The ring time (s) model #2 gives mode `m` — what one `sigma` leaves the anchor (a read-out).
///
/// `r^2 = cc / a = (1 + theta X - Y) / (1 + theta X + Y)` for that mode, and a 60 dB fall takes
/// `-6 k ln 10 / ln(r^2)`.
pub fn stiff_mode_t60(l: f64, n: usize, fs: f64, theta: f64, tuning: &Tuning, m: usize) -> f64 {
    let k = 1.0 / fs;
    let p2 = mode_p2(l, n, m);
    let x = k * k * (tuning.c * tuning.c * p2 + tuning.kappa * tuning.kappa * p2 * p2);
    let y = k * (tuning.sigma0 + tuning.sigma1 * p2);
    let u = 1.0 + theta * x;
    let ln_r2 = (-2.0 * y / (u + y)).ln_1p();
    -6.0 * k * LN_10 / ln_r2
}
