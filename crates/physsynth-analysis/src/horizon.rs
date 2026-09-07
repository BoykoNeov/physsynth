//! The resolution horizon — where a scheme's answer stops being in tune.
//!
//! Promoted out of `tests/helpers.py` (`docs/dev/resolution-horizon-plan.md` §6), which is the
//! first module in this crate that was never a transcription of a deleted Python body: it was
//! *written* in the test folder, used by five test files, and moved here because it is an
//! instrument like every other member of this crate rather than a fixture.
//!
//! # What §6 predicted, and what promoting it actually cost
//!
//! Two of that bullet's three obstacles were not obstacles.
//!
//! * It says `scipy.optimize.brentq` "would be a dependency decision in the analysis crate, whose
//!   allowlist is deliberately narrow". It is not one. [`crate::root`] is a line-for-line
//!   transcription of SciPy's `brentq.c`, `#[path]`-included into this crate rather than taken as
//!   a Cargo edge — the argument is written out at the top of `lib.rs`. `ALLOWED` in
//!   `tests/deps.rs` stays empty and no line of `Cargo.toml` moves.
//! * It says satisfying `tests/test_analysis_frozen.py`'s derived guard "is impossible — there is
//!   no Python implementation left to freeze against". That was true of the modules deleted in
//!   plan §44 and false of these seven functions, whose Python bodies were live in
//!   `tests/helpers.py` right up to the commit that replaced them. They were frozen first, the
//!   same move §44 made, so the enforcement contract covering the other 62 fixtures is untouched.
//!
//! The prose that justifies each of these — the measured constants, the corner argument, the
//! isotropy caveats — lives in full in `physsynth/analysis/horizon.py`, which is what a caller
//! reaches through `help()`. What is here is the mathematical statement and a pointer, because a
//! reader of this file wants to check the arithmetic and a reader of that one wants to know
//! whether the number is safe to quote.

use crate::root::brentq;
use std::f64::consts::PI;

/// `scipy.optimize.brentq`'s defaults, which the Python original relied on by not passing them.
///
/// Spelled out rather than inherited: the two root finds have to take the *same* path for the
/// frozen comparison to be about the mathematics instead of about a stopping rule, and a default
/// that lives in another project is not something this crate can promise. `modal.rs` carries the
/// same pair for the same reason.
const SCIPY_XTOL: f64 = 2e-12;
const SCIPY_RTOL: f64 = 8.881_784_197_001_252e-16;

/// Signed per-mode pitch error in cents — negative means the scheme is flat.
///
/// `1200 log₂(f_discrete / f_continuum)`, elementwise. Cents because every threshold worth arguing
/// about is perceptual, and because it puts the θ-scheme's rate suppression `S` and its pitch
/// error in the same units through `cents = 600 log₂ S`.
pub fn pitch_error_cents(f_discrete: &[f64], f_continuum: &[f64]) -> Result<Vec<f64>, String> {
    if f_discrete.len() != f_continuum.len() {
        return Err(format!(
            "shape mismatch: discrete ({},) against continuum ({},)",
            f_discrete.len(),
            f_continuum.len()
        ));
    }
    Ok(f_discrete
        .iter()
        .zip(f_continuum.iter())
        .map(|(&d, &c)| 1200.0 * (d / c).log2())
        .collect())
}

/// `(horizon, monotone)` — how many *leading* modes are within `cents` of the continuum.
///
/// The count is a leading prefix, not "the last mode that happens to be inside", and those differ
/// exactly when the error curve is not monotone. So the predicate travels with the number: a
/// caller that sees `monotone == false` knows the integer is hiding something. Never collapse the
/// pair back to the integer without reading the flag.
pub fn pitch_horizon(
    f_discrete: &[f64],
    f_continuum: &[f64],
    cents: f64,
) -> Result<(usize, bool), String> {
    // `cents <= 0.0 || is_nan()` rather than `!(cents > 0.0)`: the same predicate over every
    // double, and the negation clippy objects to. NaN is refused rather than accepted, which
    // is the whole reason the test is written around the positive comparison.
    if cents <= 0.0 || cents.is_nan() {
        return Err(format!("cents bound must be positive, got {cents}."));
    }
    let err: Vec<f64> = pitch_error_cents(f_discrete, f_continuum)?
        .into_iter()
        .map(f64::abs)
        .collect();
    let horizon = err.iter().position(|&e| e > cents).unwrap_or(err.len());
    // The same `-1e-12` slack the Python carried: a flat stretch of the error curve must not read
    // as non-monotone because two equal values differed in the last bit.
    let monotone = err.windows(2).all(|w| w[1] - w[0] >= -1e-12);
    Ok((horizon, monotone))
}

/// `m*/N` — the closed-form space floor, as a fraction of the grid, with no fixture in it.
///
/// The discrete axis eigenvalue is `(2/h) sin(m π h / 2L)` against the continuum `m π / L`, so the
/// ratio is `sinc(u)` with `u = m π / 2N`. `power` is how many factors of that the model's
/// *frequency* carries and is read off its dispersion relation: 1 for a string (`ω ~ c p`), 2 for
/// a plate or beam (`ω ~ κ p²`). Hence the identity
/// `sinc_horizon_fraction(c, 2) == sinc_horizon_fraction(c/2, 1)` — a plate resolves the same
/// share of its grid as a string given half the cents budget.
///
/// Independent of `c`, `L`, `N`, `k` and `fs`, which is the claim.
pub fn sinc_horizon_fraction(cents: f64, power: i64) -> Result<f64, String> {
    // `cents <= 0.0 || is_nan()` rather than `!(cents > 0.0)`: the same predicate over every
    // double, and the negation clippy objects to. NaN is refused rather than accepted, which
    // is the whole reason the test is written around the positive comparison.
    if cents <= 0.0 || cents.is_nan() {
        return Err(format!("cents bound must be positive, got {cents}."));
    }
    if power < 1 {
        return Err(format!(
            "power must be a positive number of sinc factors, got {power}."
        ));
    }
    let target = 2.0_f64.powf(-cents / 1200.0);
    let f = |z: f64| (z.sin() / z).powi(power as i32) - target;
    let u = brentq(f, 1e-12, PI / 2.0, SCIPY_XTOL, SCIPY_RTOL, 100).map_err(|e| {
        format!("the sinc horizon root find did not converge for cents={cents}, power={power}: {e}")
    })?;
    Ok(2.0 * u / PI)
}

/// The leading `count` `(m, n)` index pairs of one 2-D mode family.
///
/// A *family* is a sequence along which the pitch error is monotone, which is what makes
/// [`pitch_horizon`]'s leading-prefix reading mean anything. `"axial"` is `(m, 1)`, `"axial_y"` is
/// its transpose `(1, n)` — separate because a grain destroys their degeneracy — and `"diagonal"`
/// is `(m, m)`.
///
/// Index-side only, on purpose: the caller still builds its own discrete and continuum frequencies
/// from its own fixture. Assumes a square domain.
pub fn mode_family(kind: &str, count: i64) -> Result<Vec<(i64, i64)>, String> {
    if count < 1 {
        return Err(format!(
            "a family needs at least one mode, got count={count}."
        ));
    }
    match kind {
        "axial" => Ok((1..=count).map(|m| (m, 1)).collect()),
        "axial_y" => Ok((1..=count).map(|n| (1, n)).collect()),
        "diagonal" => Ok((1..=count).map(|m| (m, m)).collect()),
        _ => Err(format!(
            "unknown mode family '{kind}'; expected 'axial', 'axial_y' or 'diagonal'."
        )),
    }
}

/// Every `(m, n)` with `1 <= m, n <= m_max`, ordered by continuum frequency (`m² + n²`).
///
/// A block is the 2-D shape a "the first few modes are in tune" claim actually asserts, and it is
/// not a family: the error is not monotone along it, so a prefix over it is not a horizon. What
/// makes it readable is that its worst mode is a *corner* — [`block_weight`] dips at
/// `n* = m·√(√2 − 1) ≈ 0.6436 m`, strictly inside `(0, m)`, so the maximum over a block never sits
/// inside it. (On the *integer* grid that minimum is only reachable from `m = 3` up: at `m = 2` the
/// minimiser is 1.287 and the nearest index below it is the edge. The corner argument is about the
/// maximum and is untouched by that.) *Which* corner is a property of the scheme and not of the
/// block; ask [`cancellation_courant`].
///
/// The ordering key is isotropic. A grained plate orders by `g_x a² + 2 g_h a b + g_y b²`, so
/// there the returned *set* is still the block and the order is no longer its spectrum.
pub fn mode_block(m_max: i64) -> Result<Vec<(i64, i64)>, String> {
    if m_max < 1 {
        return Err(format!(
            "a block needs at least one mode per axis, got m_max={m_max}."
        ));
    }
    let mut modes: Vec<(i64, i64)> = (1..=m_max)
        .flat_map(|m| (1..=m_max).map(move |n| (m, n)))
        .collect();
    // Ties broken by `(m, n)`, matching Python's `sorted(key=lambda mn: (m² + n², m, n))`. The
    // tie-break is not cosmetic: `(1, 2)` and `(2, 1)` are degenerate on a square, so without it
    // the order would depend on the sort's stability rather than on a written-down rule.
    modes.sort_by_key(|&(m, n)| (m * m + n * n, m, n));
    Ok(modes)
}

/// The Courant number at which an explicit scheme's mode `(m, n)` is exactly in tune.
///
/// An explicit leapfrog has two pitch errors of opposite sign — the spatial operator droops the
/// frequency, the time discretisation sharpens it — and to leading order in `1/N` they combine as
/// `1 + (a²/6)(λ² ρ² − w)` with `a = π/2N`, `ρ² = m² + n²` and `w` the [`block_weight`]. So they
/// cancel at `λ² = (m⁴ + n⁴)/(m² + n²)²`, which is what this returns.
///
/// Three consequences, none of them a fixture: it is `1/√2` on the diagonal for every `m`, which
/// *is* the 2-D CFL ceiling; hence `λ ≤ 1/√2 ≤ cancellation_courant(m, n)` for every mode, so on a
/// stable membrane no mode is ever sharp; and it rises toward 1 along the axial family, which is
/// above the ceiling and therefore unreachable.
///
/// `n = 0` spells the 1-D degenerate case — no second axis, so `ρ² = m²` and `w = m²` — and the
/// formula returns exactly `1.0` for every `m`. That is the 1-D CFL limit, and it is why an ideal
/// string at `λ = 1` resolves its whole grid while a membrane at its own ceiling resolves one
/// family: in 1-D every mode attains the stability limit at once, in 2-D only the diagonal does.
///
/// Leading order in `1/N²`, so a *measured* crossing approaches this rather than sitting on it.
/// The diagonal value is the exception and is exact at every `N`.
pub fn cancellation_courant(m: i64, n: i64) -> Result<f64, String> {
    if m < 1 {
        return Err(format!("a mode index starts at 1, got m={m}."));
    }
    if n < 0 {
        return Err(format!(
            "got n={n}; use n = 0 for the 1-D case with no second axis."
        ));
    }
    let (mf, nf) = (m as f64, n as f64);
    let rho2 = mf * mf + nf * nf;
    Ok((mf.powi(4) + nf.powi(4)).sqrt() / rho2)
}

/// `w(m, n) = (m⁴ + n⁴)/(m² + n²)` — the space droop's weight, shared by both schemes.
///
/// The only thing a mode's spatial pitch error depends on, up to a factor set by the grid: a
/// plate's frequency ratio is `1 − a² w/3` and a membrane's is its square root, `1 − a² w/6`.
/// Square root is monotone, so the two models order a block identically and the plate's corner
/// argument transfers to the membrane unchanged — in space. What breaks it is the term the
/// implicit plate does not have ([`cancellation_courant`]).
pub fn block_weight(m: i64, n: i64) -> Result<f64, String> {
    if m < 1 || n < 1 {
        return Err(format!("a mode index starts at 1, got ({m}, {n})."));
    }
    let (mf, nf) = (m as f64, n as f64);
    Ok((mf.powi(4) + nf.powi(4)) / (mf * mf + nf * nf))
}
