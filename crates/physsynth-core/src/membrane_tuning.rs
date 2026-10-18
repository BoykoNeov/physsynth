//! Musical controls for the membrane — `docs/dev/membrane-mapping-plan.md`.
//!
//! The third piece of `docs/dev/scientific-hurdles.md` §12, after [`crate::string_tuning`] and
//! [`crate::plate_tuning`]: a musician picks a head, a size and either a tension or a pitch, plus
//! two ring times, and this module returns a [`crate::membrane::Params`] that meets them. Like the
//! other mappings it is an optional helper — the raw constructor is untouched.
//!
//! # Heads
//!
//! [`HEADS`] lists clear polyester-film heads by ply construction, each giving an areal density
//! `ρ = ρ_film · t` (kg/m²). The density moves no pitch — only `c = sqrt(T/ρ)` does — but it
//! scales every energy, every force a mallet feels and every load the air puts on the head. A wrong
//! `ρ` therefore leaves every partial right; only an energy or a strike can see it.
//!
//! # The mapping: corrected to the scheme
//!
//! [`tune_membrane`] meets its targets in the *simulated* membrane. Each mode `m` (eigenvalue
//! `μ_m` of `−L`) obeys the explicit scheme's two-step recurrence, the frequency-dependent loss
//! included (`crate::membrane`'s header):
//!
//! ```text
//! (1 + S) z² + (−2 + X + Y) z + (1 − S − Y) = 0,   S = σ₀k,  X = c²k²μ,  Y = 2σ₁kμ
//! ```
//!
//! and a target `(f, T60)` is the root pair `r e^{±iφ}`, `φ = 2πfk`, `r = 10^(−3k/T60)`:
//!
//! ```text
//! S (1 + r²) + Y = 1 − r²              (ring time — no X in it)
//! X = 2 − Y − 2r cos φ (1 + S)         (pitch — the loss lowers it)
//! ```
//!
//! **The two ring times alone fix both losses**, by a 2×2 in `(σ₀, σ₁)` over mode 1 and the
//! anchor, whichever of tension and pitch is set. Set by pitch, mode 1's pitch equation then gives
//! `c²` *with* the losses in; set by tension, every pitch is a read-out. One wave speed, so one free
//! pitch: the anchor's pitch is whatever the shape makes it, and only its ring time is a target.
//!
//! **Exact on both shapes.** The loss operator is the stiffness operator `−L` itself, so every
//! mode — the staircased circle's included — keeps its own recurrence, and mode 1 and the anchor
//! land on their targets to rounding (plan §9: `1e-8` cents, `5e-12` in ring time). The textbook
//! mapping (`c` from `j₀₁`) is 10–25 cents flat on the round head at the viewer's grids.

use crate::eigs::{eigsh_shift_invert, EigsError};
use crate::membrane::{Domain, ParamError, Params};
use std::f64::consts::{LN_10, PI};

/// A drumhead's construction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Head {
    /// The name a caller selects it by.
    pub name: &'static str,
    /// Each ply's film thickness (mil, 1/1000 inch).
    pub plies_mil: &'static [f64],
    /// Where the construction comes from.
    pub source: &'static str,
}

/// Polyester (PET) film density, kg/m³ — DuPont's *Mylar* polyester film typical-properties sheet,
/// 1.390 g/cm³ by ASTM D1505 (DuPont Teijin Films' current sheets give 1.395–1.40 by gauge). The
/// head makers' product pages say "film" without naming it; PET is the material assumed.
pub const FILM_DENSITY: f64 = 1390.0;

/// One mil in metres.
pub const MIL: f64 = 25.4e-6;

/// Remo's product pages (remo.com, accessed 2026-10-18) give each construction: Ambassador Hazy
/// Snare Side "1-ply of 3-mil film", Diplomat Clear "1-ply of 7.5-mil film", Ambassador Clear
/// "1 ply 10-mil Clear film", Emperor Clear "two-plies of 7-mil Clear film". Clear heads only: a
/// coating, a muffling ring or the collar's glue adds mass this table does not carry.
const REMO: &str = "Remo product pages (ply count, film thickness); DuPont Mylar density";

/// The heads a caller can name.
pub const HEADS: &[Head] = &[
    Head {
        name: "single-ply 3 mil",
        plies_mil: &[3.0],
        source: REMO,
    },
    Head {
        name: "single-ply 7.5 mil",
        plies_mil: &[7.5],
        source: REMO,
    },
    Head {
        name: "single-ply 10 mil",
        plies_mil: &[10.0],
        source: REMO,
    },
    Head {
        name: "two-ply 7 + 7 mil",
        plies_mil: &[7.0, 7.0],
        source: REMO,
    },
];

impl Head {
    /// Total film thickness (m).
    pub fn thickness(&self) -> f64 {
        self.plies_mil.iter().sum::<f64>() * MIL
    }

    /// Areal density `ρ_film · t` (kg/m²) — what the membrane's `rho` is.
    pub fn areal_density(&self) -> f64 {
        FILM_DENSITY * self.thickness()
    }
}

/// The head called `name`, if there is one.
pub fn head(name: &str) -> Option<&'static Head> {
    HEADS.iter().find(|h| h.name == name)
}

// -- the mapping ----------------------------------------------------------------------------------

/// The shape, size, grid and sample rate — everything about a membrane that is not tuned.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Geometry {
    /// Which shape.
    pub domain: Domain,
    /// Rectangle width (m); ignored for a circle.
    pub lx: f64,
    /// Rectangle height (m); ignored for a circle.
    pub ly: f64,
    /// Disk radius (m); ignored for a rectangle.
    pub radius: f64,
    /// Grid segments, as [`Params::new`] takes them.
    pub n: i64,
    /// Sample rate (Hz).
    pub fs: f64,
}

/// Which of tension and pitch the musician sets; the other is read out.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SetBy {
    /// Tension per unit length (N/m); the fundamental's pitch is read out.
    Tension(f64),
    /// The fundamental's pitch (Hz); the tension is worked out.
    Pitch(f64),
}

/// What a musician sets, beyond the head and the geometry.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Targets {
    /// Tension or pitch.
    pub set_by: SetBy,
    /// Ring time of the lowest mode: seconds for its amplitude to fall 60 dB.
    pub t60_low: f64,
    /// Ring time of the anchor — the mode nearest `f_ref` — in seconds.
    pub t60_high: f64,
    /// Where "high" is measured (Hz); picks the anchor. Must lie above the fundamental.
    pub f_ref: f64,
}

/// The membrane that meets the targets, and what a caller shows beside it.
#[derive(Debug, Clone)]
pub struct MembraneTuning {
    /// The membrane, both losses set — ready for [`crate::membrane::Membrane::new`].
    pub params: Params,
    /// Tension (N/m) — the input when set by tension, worked out when set by pitch.
    pub tension: f64,
    /// Frequency-independent loss (1/s).
    pub sigma0: f64,
    /// Frequency-dependent loss (m²/s).
    pub sigma1: f64,
    /// The lowest mode's frequency (Hz) in the scheme, loss included — the target when set by
    /// pitch, the read-out when set by tension.
    pub f0: f64,
    /// The anchor's frequency (Hz) in the scheme, loss included.
    pub anchor_frequency: f64,
    /// The relative gap from the anchor to the nearest mode at another frequency. Both ring within a
    /// hair of each other inside a cluster; the anchor is exact either way.
    pub anchor_margin: f64,
    /// The lowest mode's eigenvalue of `−L` (1/m²).
    pub fundamental_mu: f64,
    /// The anchor's eigenvalue of `−L` (1/m²).
    pub anchor_mu: f64,
}

/// Why a set of targets was refused.
#[derive(Debug, Clone, PartialEq)]
pub enum TuningError {
    /// A target or the density was not a positive finite number; names it.
    NonPositive(&'static str),
    /// The membrane itself would not build — its geometry, or the CFL bound (with the loss) at
    /// the tuned values.
    Membrane(ParamError),
    /// The circle's modes could not be computed.
    Modes(EigsError),
    /// The frequency is beyond what this scheme can sound at this `fs`.
    OutOfReach {
        /// What was asked (Hz).
        f: f64,
        /// Which target.
        what: &'static str,
    },
    /// `f_ref` picks the fundamental itself, so one anchor would have to fix two losses.
    AnchorIsTheFundamental {
        /// The reference asked for (Hz).
        f_ref: f64,
        /// The fundamental (Hz).
        f0: f64,
    },
    /// The equations had no solution, or a non-positive tension.
    Singular,
    /// The high ring time needs a negative loss. Fixing `t60_low` bounds `t60_high` on both sides:
    /// the shortest comes with `σ₀ = 0`, the longest with `σ₁ = 0`.
    HighRingTimeOutOfRange {
        /// The high ring time asked for (s).
        asked: f64,
        /// The shortest these settings allow (s), at `σ₀ = 0`.
        shortest: f64,
        /// The longest (s), at `σ₁ = 0`.
        longest: f64,
    },
}

impl std::fmt::Display for TuningError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TuningError::NonPositive(what) => write!(f, "{what} must be positive and finite."),
            TuningError::Membrane(e) => write!(f, "{e}"),
            TuningError::Modes(e) => write!(f, "the membrane's modes could not be found: {e:?}"),
            TuningError::OutOfReach { f: hz, what } => write!(
                f,
                "{what} = {hz} Hz is beyond what this grid and sample rate can sound; raise the \
                 sample rate or lower the target."
            ),
            TuningError::AnchorIsTheFundamental { f_ref, f0 } => write!(
                f,
                "f_ref = {f_ref} Hz picks the fundamental ({f0} Hz) itself; put it above the \
                 fundamental so the high ring time has a partial of its own."
            ),
            TuningError::Singular => write!(f, "these targets have no membrane that meets them."),
            TuningError::HighRingTimeOutOfRange {
                asked,
                shortest,
                longest,
            } => write!(
                f,
                "t60_high = {asked} s is out of reach: with this fundamental's ring time the anchor \
                 can ring between {shortest:.4} s and {longest:.4} s (a passive loss that rises \
                 with frequency)."
            ),
        }
    }
}

impl std::error::Error for TuningError {}

/// `r² − 1` for a ring time, through `exp_m1` (`r` is a hair below 1).
fn r2_minus_1(t60: f64, k: f64) -> f64 {
    (-6.0 * LN_10 * k / t60).exp_m1()
}

/// The `X` a lossless mode at `f` needs, `4 sin²(πfk)`; `None` past Nyquist.
fn lossless_x(f: f64, k: f64) -> Option<f64> {
    (f > 0.0 && f * k < 0.5).then(|| 4.0 * (PI * f * k).sin().powi(2))
}

/// The lossless frequency (Hz) of a mode with `X`; `None` past the scheme's reach.
fn lossless_f(x: f64, k: f64) -> Option<f64> {
    (x > 0.0 && x < 4.0).then(|| (0.5 * x.sqrt()).asin() / (PI * k))
}

/// The frequency (Hz) of the recurrence's root pair at `(S, X, Y)`; `None` for a non-oscillating
/// one.
pub fn mode_frequency(s: f64, x: f64, y: f64, k: f64) -> Option<f64> {
    let (a, b, cc) = (1.0 + s, -2.0 + x + y, 1.0 - s - y);
    let disc = 4.0 * a * cc - b * b;
    (disc > 0.0).then(|| disc.sqrt().atan2(-b) / (2.0 * PI * k))
}

/// The ring time (s) of the recurrence's root pair at `(S, Y)`, from `|z|² = (1 − S − Y)/(1 + S)`.
pub fn mode_t60(s: f64, y: f64, k: f64) -> f64 {
    let ln_r2 = ((-2.0 * s - y) / (1.0 + s)).ln_1p();
    -6.0 * LN_10 * k / ln_r2
}

/// `−L`'s eigenvalues near `shift` (circle), ascending.
fn circle_mus(p: &Params, shift: f64, nev: usize) -> Result<Vec<f64>, TuningError> {
    let nev = nev.min(p.n_live() - 1);
    let pairs =
        eigsh_shift_invert(&p.l.scaled(-1.0), None, shift, nev).map_err(TuningError::Modes)?;
    Ok(pairs.values)
}

/// Rectangle: closed form, `(4/h²)(sin²(pπ/2N) + sin²(qπ/2N_y))` over the interior nodes — every
/// mode, no solve. Ascending, degenerate pairs listed twice.
fn rectangle_mus(p: &Params) -> Vec<f64> {
    let (rows, cols) = p.shape();
    let (nx, ny) = (cols - 1, rows - 1);
    let h2 = p.h * p.h;
    let mut out = Vec::with_capacity((nx - 1) * (ny - 1));
    for pm in 1..nx {
        let d = 4.0 / h2 * (pm as f64 * PI / (2.0 * nx as f64)).sin().powi(2);
        for qm in 1..ny {
            out.push(d + 4.0 / h2 * (qm as f64 * PI / (2.0 * ny as f64)).sin().powi(2));
        }
    }
    out.sort_by(f64::total_cmp);
    out
}

/// How many modes around the anchor's eigenvalue the circle's solve returns.
const ANCHOR_CANDIDATES: usize = 10;

/// A shift safely below the circle's lowest eigenvalue `(j₀₁/a)²`.
fn circle_low_shift(p: &Params) -> f64 {
    let r = p.radius.expect("a circle");
    -1e-3 * (2.4 / r).powi(2)
}

/// The lowest `count` eigenvalues of `−L` (1/m²), ascending, a degenerate pair listed twice. For a
/// caller reporting what the mapping does to the partials it does not pin: each one's pitch and
/// ring time are [`mode_frequency`] and [`mode_t60`] at `S = σ₀k`, `X = c²k²μ`, `Y = 2σ₁kμ`.
///
/// # Errors
/// [`TuningError::Modes`] when the circle's solve fails.
pub fn lowest_modes(p: &Params, count: usize) -> Result<Vec<f64>, TuningError> {
    match p.domain {
        Domain::Rectangle => Ok(rectangle_mus(p).into_iter().take(count).collect()),
        Domain::Circle => circle_mus(p, circle_low_shift(p), count),
    }
}

/// The membrane described by `geometry`, its areal density `rho` (kg/m²), at the tension `t`.
fn build(g: &Geometry, t: f64, rho: f64, sigma0: f64) -> Result<Params, ParamError> {
    let (lx, ly, radius) = match g.domain {
        Domain::Rectangle => (Some(g.lx), Some(g.ly), None),
        Domain::Circle => (None, None, Some(g.radius)),
    };
    Params::new(Some(g.domain), t, rho, g.fs, g.n, lx, ly, radius, sigma0)
}

/// Map `targets` onto the membrane `geometry` describes, with areal density `rho` (kg/m²) — a
/// [`Head::areal_density`], or any custom value.
///
/// # Errors
/// Any [`TuningError`]; the tuned membrane past the loss-tightened CFL bound comes back as
/// [`TuningError::Membrane`] with [`ParamError::LossCflViolated`] (or the plain `CflViolated`).
pub fn tune_membrane(
    geometry: &Geometry,
    rho: f64,
    targets: &Targets,
) -> Result<MembraneTuning, TuningError> {
    let positive = |v: f64| v > 0.0 && v.is_finite();
    match targets.set_by {
        SetBy::Tension(t) if !positive(t) => return Err(TuningError::NonPositive("tension")),
        SetBy::Pitch(f) if !positive(f) => return Err(TuningError::NonPositive("f0")),
        _ => {}
    }
    for (v, what) in [
        (rho, "rho"),
        (targets.t60_low, "t60_low"),
        (targets.t60_high, "t60_high"),
        (targets.f_ref, "f_ref"),
    ] {
        if !positive(v) {
            return Err(TuningError::NonPositive(what));
        }
    }

    // The modes depend on the shape and the grid, not on the tension: build at a tension that puts
    // the Courant number at 0.1, comfortably legal, and read them there.
    let n_seg = geometry.n.max(2) as f64;
    let span = match geometry.domain {
        Domain::Rectangle => geometry.lx,
        Domain::Circle => 2.0 * geometry.radius,
    };
    let c_probe = 0.1 * (span / n_seg) * geometry.fs;
    let probe =
        build(geometry, rho * c_probe * c_probe, rho, 0.0).map_err(TuningError::Membrane)?;
    let k = probe.k;
    let rect = match probe.domain {
        Domain::Rectangle => Some(rectangle_mus(&probe)),
        Domain::Circle => None,
    };
    let mu1 = match &rect {
        Some(all) => all[0],
        None => circle_mus(&probe, circle_low_shift(&probe), 3)?[0],
    };

    // A lossless wave speed, to choose the anchor on the LOSSLESS spectrum — so the choice does not
    // depend on the losses it then helps fix.
    let c2_lossless = match targets.set_by {
        SetBy::Tension(t) => t / rho,
        SetBy::Pitch(f0) => {
            lossless_x(f0, k).ok_or(TuningError::OutOfReach { f: f0, what: "f0" })? / (k * k * mu1)
        }
    };
    let f_of = |mu: f64| lossless_f(c2_lossless * k * k * mu, k);
    let f0_lossless = f_of(mu1).ok_or(TuningError::OutOfReach {
        f: f64::INFINITY,
        what: "the fundamental",
    })?;
    let x_ref = lossless_x(targets.f_ref, k).ok_or(TuningError::OutOfReach {
        f: targets.f_ref,
        what: "f_ref",
    })?;
    let near = match rect {
        Some(all) => all,
        None => circle_mus(&probe, x_ref / (c2_lossless * k * k), ANCHOR_CANDIDATES)?,
    };
    let mut best: Option<(f64, f64)> = None;
    for &mu in near.iter().filter(|&&mu| mu > mu1 * (1.0 + 1e-9)) {
        if let Some(fm) = f_of(mu) {
            if best.is_none_or(|(_, fb)| (fm - targets.f_ref).abs() < (fb - targets.f_ref).abs()) {
                best = Some((mu, fm));
            }
        }
    }
    let (mu_m, fa) = best.ok_or(TuningError::OutOfReach {
        f: targets.f_ref,
        what: "f_ref",
    })?;
    if (targets.f_ref - f0_lossless).abs() <= (targets.f_ref - fa).abs() {
        return Err(TuningError::AnchorIsTheFundamental {
            f_ref: targets.f_ref,
            f0: f0_lossless,
        });
    }
    let anchor_margin = near
        .iter()
        .filter_map(|&mu| f_of(mu))
        .filter(|&f| (f - fa).abs() > 1e-9 * fa)
        .fold(f64::INFINITY, |m, f| m.min((f - fa).abs() / fa));

    // Both losses from the two ring-time equations, S (2 + e) + 2kμ σ₁ = −e, e = r² − 1.
    let (e1, em) = (
        r2_minus_1(targets.t60_low, k),
        r2_minus_1(targets.t60_high, k),
    );
    let (a11, a12, b1) = (2.0 + e1, 2.0 * k * mu1, -e1);
    let (a21, a22, b2) = (2.0 + em, 2.0 * k * mu_m, -em);
    let det = a11 * a22 - a12 * a21;
    let s = (b1 * a22 - a12 * b2) / det;
    let sigma1 = (a11 * b2 - b1 * a21) / det;
    let sigma0 = s / k;
    if !(det != 0.0 && sigma0.is_finite() && sigma1.is_finite()) {
        return Err(TuningError::Singular);
    }
    if sigma0 < 0.0 || sigma1 < 0.0 {
        // The ends of the range: all of mode 1's loss in σ₁ (S = 0), or none of it (Y₁ = 0).
        let y1_all = -e1;
        return Err(TuningError::HighRingTimeOutOfRange {
            asked: targets.t60_high,
            shortest: mode_t60(0.0, y1_all * mu_m / mu1, k),
            longest: mode_t60(-e1 / (2.0 + e1), 0.0, k),
        });
    }
    let (y1, ym) = (2.0 * sigma1 * k * mu1, 2.0 * sigma1 * k * mu_m);

    // The wave speed: from mode 1's pitch equation with the losses in, or from the tension.
    let c2 = match targets.set_by {
        SetBy::Tension(t) => t / rho,
        SetBy::Pitch(f0) => {
            let r1 = (-3.0 * LN_10 * k / targets.t60_low).exp();
            let x1 = 2.0 - y1 - 2.0 * r1 * (2.0 * PI * f0 * k).cos() * (1.0 + s);
            if !(x1 > 0.0 && x1.is_finite()) {
                return Err(TuningError::Singular);
            }
            x1 / (k * k * mu1)
        }
    };
    let tension = rho * c2;
    let params = build(geometry, tension, rho, sigma0)
        .and_then(|p| p.with_sigma1(sigma1))
        .map_err(TuningError::Membrane)?;
    let (x1, xm) = (c2 * k * k * mu1, c2 * k * k * mu_m);
    Ok(MembraneTuning {
        params,
        tension,
        sigma0,
        sigma1,
        f0: mode_frequency(s, x1, y1, k).ok_or(TuningError::Singular)?,
        anchor_frequency: mode_frequency(s, xm, ym, k).ok_or(TuningError::Singular)?,
        anchor_margin,
        fundamental_mu: mu1,
        anchor_mu: mu_m,
    })
}
