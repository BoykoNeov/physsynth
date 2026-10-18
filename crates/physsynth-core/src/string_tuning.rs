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
