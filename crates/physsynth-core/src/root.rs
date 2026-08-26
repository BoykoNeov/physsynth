//! Scalar root-finding — Brent's method, transcribed from SciPy's `Zeros/brentq.c`.
//!
//! # Why this is here rather than a dependency, and why it is a transcription rather than a
//! rewrite
//!
//! `crates/physsynth-core/tests/deps.rs` keeps this crate's dependency list **empty**, and the
//! migration plan (§2.2) says the first numeric crate arrives as a deliberate edit with its reason
//! written next to it. `reed` does not supply that reason, because the requirement here is not
//! "a root-finder" — it is *this* root-finder. The Python original calls
//! `scipy.optimize.brentq(..., xtol=1e-13, rtol=8.9e-16)` as the guaranteed fallback when its
//! safeguarded Newton stalls on the `sqrt` cusp, and the plan's acceptance runs were taken with
//! whatever that call returns. A different-but-equally-good Brent would move the numbers.
//!
//! **And the fallback is not hypothetical.** Measured on 2026-08-26 over the configurations the
//! reed's own tests build: it fires 4-5 times per 4,000 steps in the flagship
//! (`p_mouth = 1500 Pa`) case, 13 times at `p_mouth = 1800`, and **219 times** on a coarse
//! `N = 40` grid. So the choice was between transcribing this ~90-line C function and dropping the
//! reed out of the bit-identical bucket entirely.
//!
//! The transcription was checked before it was relied on: implemented in Python first and run
//! against `scipy.optimize.brentq` on the reed's own residuals over **248 real calls**, the two
//! returned bit-identical roots every time — not close, equal. That measurement is what makes
//! `tests/test_rust_parity_reed.py` able to assert `array_equal` rather than a tolerance.
//!
//! # What is deliberately *not* reproduced
//!
//! SciPy's Python wrapper rejects `rtol` below `4 * eps` before it ever reaches the C code. That
//! check belongs to the caller and is not repeated here; `reed` passes `8.9e-16`, which clears it.

/// How a [`brentq`] call ended.
#[derive(Debug, Clone, PartialEq)]
pub enum RootError {
    /// `f(a)` and `f(b)` have the same sign — there is no bracketed root. SciPy raises
    /// `ValueError: f(a) and f(b) must have different signs`.
    SameSign,
    /// The iteration limit was reached. SciPy raises `RuntimeError: Failed to converge after N
    /// iterations.`; carries `N`.
    NotConverged(usize),
}

impl std::fmt::Display for RootError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RootError::SameSign => write!(f, "f(a) and f(b) must have different signs"),
            RootError::NotConverged(n) => {
                write!(f, "Failed to converge after {n} iterations.")
            }
        }
    }
}

impl std::error::Error for RootError {}

/// SciPy's default iteration limit.
pub const DEFAULT_MAXITER: usize = 100;

/// Find a root of `f` in the bracket `[xa, xb]` by Brent's method.
///
/// A line-for-line transcription of SciPy's `brentq.c`, including the details that look
/// incidental and are not: the sign test is on `signbit` (so `-0.0` counts as negative), the
/// tolerance is `(xtol + rtol * |xcur|) / 2` recomputed each iteration, the interpolation/
/// extrapolation branch is chosen by `xpre == xblk`, and a step shorter than `delta` is replaced
/// by `delta` in the direction of the bisection rather than being taken as is.
///
/// Returns the root, or [`RootError`] for the two ways SciPy refuses.
pub fn brentq<F>(
    mut f: F,
    xa: f64,
    xb: f64,
    xtol: f64,
    rtol: f64,
    maxiter: usize,
) -> Result<f64, RootError>
where
    F: FnMut(f64) -> f64,
{
    let mut xpre = xa;
    let mut xcur = xb;
    let mut xblk = 0.0;
    let mut fblk = 0.0;
    let mut spre = 0.0;
    let mut scur = 0.0;

    let mut fpre = f(xpre);
    let mut fcur = f(xcur);
    if fpre == 0.0 {
        return Ok(xpre);
    }
    if fcur == 0.0 {
        return Ok(xcur);
    }
    if fpre.is_sign_negative() == fcur.is_sign_negative() {
        return Err(RootError::SameSign);
    }

    for _ in 0..maxiter {
        if fpre != 0.0 && fcur != 0.0 && (fpre.is_sign_negative() != fcur.is_sign_negative()) {
            xblk = xpre;
            fblk = fpre;
            spre = xcur - xpre;
            scur = spre;
        }
        if fblk.abs() < fcur.abs() {
            xpre = xcur;
            xcur = xblk;
            xblk = xpre;

            fpre = fcur;
            fcur = fblk;
            fblk = fpre;
        }

        let delta = (xtol + rtol * xcur.abs()) / 2.0;
        let sbis = (xblk - xcur) / 2.0;
        if fcur == 0.0 || sbis.abs() < delta {
            return Ok(xcur);
        }

        if spre.abs() > delta && fcur.abs() < fpre.abs() {
            let stry = if xpre == xblk {
                // interpolate
                -fcur * (xcur - xpre) / (fcur - fpre)
            } else {
                // extrapolate
                let dpre = (fpre - fcur) / (xpre - xcur);
                let dblk = (fblk - fcur) / (xblk - xcur);
                -fcur * (fblk * dblk - fpre * dpre) / (dblk * dpre * (fblk - fpre))
            };
            if 2.0 * stry.abs() < spre.abs().min(3.0 * sbis.abs() - delta) {
                // good short step
                spre = scur;
                scur = stry;
            } else {
                // bisect
                spre = sbis;
                scur = sbis;
            }
        } else {
            // bisect
            spre = sbis;
            scur = sbis;
        }

        xpre = xcur;
        fpre = fcur;
        if scur.abs() > delta {
            xcur += scur;
        } else {
            xcur += if sbis > 0.0 { delta } else { -delta };
        }

        fcur = f(xcur);
    }
    Err(RootError::NotConverged(maxiter))
}
