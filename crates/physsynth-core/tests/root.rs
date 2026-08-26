//! Bars for the Brent transcription in `root`.
//!
//! This is the one piece of the crate that is a **transcription of somebody else's code** rather
//! than of somebody else's formula, so what it owes is different: not "does it find a root" — any
//! bisection does that — but "does it find *SciPy's* root". `reed.py` calls
//! `scipy.optimize.brentq` as its guaranteed fallback and the project's acceptance runs were taken
//! with whatever that call returns, so a different-but-equally-good Brent moves the numbers.
//!
//! That claim is checked where it can be: `tests/test_rust_parity_reed.py` runs both
//! implementations of the whole clarinet side by side and requires the trajectories to stay
//! bit-identical through thousands of steps in which the fallback fires — including a coarse-grid
//! case where it fires 270 times and must land on the **same steps**. It was also checked before
//! any of this was written: the same transcription, implemented in Python and run against SciPy on
//! the reed's own residuals over 248 real calls, returned bit-identical roots every time.
//!
//! What is left for this file is the behaviour a caller can reason about locally: the refusals, the
//! tolerance contract, and the awkward shapes (a root at an endpoint, a cusp, a flat region) that
//! a hand transcription is most likely to get wrong.

use physsynth_core::root::{brentq, RootError, DEFAULT_MAXITER};

const XTOL: f64 = 1e-13;
const RTOL: f64 = 8.9e-16;

fn solve<F: FnMut(f64) -> f64>(f: F, a: f64, b: f64) -> f64 {
    brentq(f, a, b, XTOL, RTOL, DEFAULT_MAXITER).expect("this bracket has a root")
}

#[test]
fn a_bracket_without_a_sign_change_is_refused() {
    let err = brentq(|x| x * x + 1.0, -1.0, 1.0, XTOL, RTOL, DEFAULT_MAXITER).unwrap_err();
    assert_eq!(err, RootError::SameSign);
    assert_eq!(err.to_string(), "f(a) and f(b) must have different signs");
}

#[test]
fn an_exhausted_iteration_budget_is_refused_rather_than_silently_returned() {
    // One iteration is not enough for anything but a lucky guess; the point is that it *says so*
    // rather than handing back a half-converged number.
    let err = brentq(|x| x - 0.37, 0.0, 1.0, 1e-300, 1e-300, 1).unwrap_err();
    assert_eq!(err, RootError::NotConverged(1));
    assert_eq!(err.to_string(), "Failed to converge after 1 iterations.");
}

#[test]
fn an_exact_root_at_either_endpoint_is_returned_as_is() {
    // SciPy checks both endpoints for an exact zero *before* the sign test, so a bracket whose
    // ends are `[root, something_positive]` returns rather than being refused for same-sign.
    assert_eq!(solve(|x| x - 2.0, 2.0, 5.0), 2.0);
    assert_eq!(solve(|x| x - 5.0, 2.0, 5.0), 5.0);
}

#[test]
fn the_sign_test_is_on_signbit_so_negative_zero_counts_as_negative() {
    // `signbit(-0.0)` is true and `-0.0 < 0.0` is false — the two spellings disagree here, and the
    // C original uses `signbit`. A function that reaches exactly `-0.0` at one end is bracketed
    // against a positive value, not refused.
    let r = brentq(
        |x| if x <= 0.0 { -0.0 } else { x },
        -1.0,
        1.0,
        XTOL,
        RTOL,
        DEFAULT_MAXITER,
    );
    assert_eq!(
        r,
        Ok(-1.0),
        "an endpoint evaluating to -0.0 is an exact root: -0.0 == 0.0"
    );
}

#[test]
fn it_finds_the_roots_of_well_behaved_functions() {
    for (name, root, f) in [
        (
            "linear",
            0.37,
            Box::new(|x: f64| x - 0.37) as Box<dyn Fn(f64) -> f64>,
        ),
        ("cubic", 1.0, Box::new(|x: f64| x * x * x - 1.0)),
        ("exp", 0.0, Box::new(|x: f64| x.exp() - 1.0)),
        ("sin", std::f64::consts::PI, Box::new(|x: f64| x.sin())),
    ] {
        let got = solve(&*f, root - 1.3, root + 1.7);
        assert!(
            (got - root).abs() <= XTOL + RTOL * root.abs() + 1e-14,
            "{name}: got {got}, want {root}"
        );
    }
}

#[test]
fn it_handles_the_sqrt_cusp_that_defeats_newton() {
    // The shape the reed's fallback exists for: infinite slope at the origin, where a Newton step
    // overshoots forever. Brent bisects its way in regardless.
    let f = |x: f64| x.abs().sqrt().copysign(x) + 0.5 * x - 0.25;
    let got = solve(f, -10.0, 10.0);
    assert!(f(got).abs() < 1e-12, "residual {:.3e} at {got}", f(got));
}

#[test]
fn a_flat_region_next_to_the_root_still_converges() {
    // Interpolation is useless where the function is flat (`fcur - fpre` underflows), so this
    // exercises the bisection fallback inside the iteration rather than the fallback around it.
    let f = |x: f64| if x < 0.5 { -1e-300 } else { x - 0.5 };
    let got = solve(f, 0.0, 1.0);
    assert!((got - 0.5).abs() < 1e-9, "got {got}");
}

#[test]
fn the_returned_root_respects_the_tolerance_it_was_given() {
    // `delta = (xtol + rtol |xcur|) / 2`, and the loop stops when the half-bracket is inside it.
    // A loose `xtol` must therefore stop early and *stay* within it — not accidentally converge.
    let root = 12345.6789;
    for &xtol in &[1e-13, 1e-6, 1e-2] {
        let got = brentq(|x| x - root, 0.0, 20000.0, xtol, RTOL, DEFAULT_MAXITER).unwrap();
        assert!(
            (got - root).abs() <= xtol + RTOL * root.abs(),
            "xtol {xtol:.1e}: |{got} - {root}| exceeds it"
        );
    }
}
