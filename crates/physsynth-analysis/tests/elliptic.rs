//! The elliptic functions' own bars — identities, degenerate limits, and published digits.
//!
//! Same discipline as `bessel.rs`: a parity test proves this agrees with SciPy, and these prove it
//! is right. The Jacobian functions are especially worth checking against their identities rather
//! than against a table, because the AGM descent is a loop whose output looks plausible for a wide
//! range of wrong loop bounds — `sn² + cn² = 1` holds trivially by construction (both come from one
//! `φ`), but `dn² + m·sn² = 1` and the addition-free period relation do not.

use physsynth_analysis::elliptic::{ellipj, ellipk};
use std::f64::consts::PI;

#[test]
fn ellipk_hits_its_known_values() {
    assert_eq!(ellipk(0.0), PI / 2.0, "K(0) = pi/2 exactly");
    // K(1/2) = Gamma(1/4)^2 / (4 sqrt(pi)) = 1.8540746773013719...
    assert!((ellipk(0.5) - 1.854_074_677_301_372).abs() < 1e-15);
    // Legendre's relation in its simplest form: K(m) is strictly increasing on [0, 1).
    let mut prev = ellipk(0.0);
    for i in 1..200 {
        let k = ellipk(i as f64 / 200.0);
        assert!(k > prev, "K is not increasing at m = {}", i as f64 / 200.0);
        prev = k;
    }
    assert!(ellipk(1.0).is_infinite(), "K(1) diverges");
    assert!(ellipk(1.5).is_nan(), "K(m) is undefined above 1");
    // Negative m is ordinary and SciPy computes it: K(-1) = 1.3110287771460598.
    assert!((ellipk(-1.0) - 1.311_028_777_146_059_8).abs() < 1e-15);
}

#[test]
fn ellipj_is_hyperbolic_on_the_separatrix() {
    // m = 1 is where the AGM descent degenerates and the functions stop being periodic. No caller
    // in this project reaches it, which is the reason to pin it: an unvisited edge is an untested
    // one, and the loop's own answer there is plausible and wrong.
    for &u in &[0.0, 1.0, -2.5] {
        let (sn, cn, dn) = ellipj(u, 1.0);
        assert_eq!(sn, u.tanh());
        assert_eq!(cn, 1.0 / u.cosh());
        assert_eq!(dn, 1.0 / u.cosh());
    }
}

#[test]
fn ellipj_degenerates_to_the_circular_functions_exactly() {
    // m = 0 terminates the Landen sequence before its first step, so this is not "close to cos" --
    // it IS cos, and `tests/test_tension_string.py` leans on that at atol 1e-14.
    for &u in &[-3.5, -0.25, 0.0, 0.25, 1.0, 7.75] {
        let (sn, cn, dn) = ellipj(u, 0.0);
        assert_eq!(sn, u.sin(), "sn(u, 0) = sin u");
        assert_eq!(cn, u.cos(), "cn(u, 0) = cos u");
        assert_eq!(dn, 1.0, "dn(u, 0) = 1");
    }
}

#[test]
fn ellipj_satisfies_both_pythagorean_identities() {
    for i in 0..40 {
        let m = i as f64 / 40.0 * 0.98;
        for j in -30..=30 {
            let u = j as f64 * 0.7;
            let (sn, cn, dn) = ellipj(u, m);
            assert!(
                (sn * sn + cn * cn - 1.0).abs() < 1e-14,
                "sn^2 + cn^2 != 1 at u={u}, m={m}"
            );
            assert!(
                (dn * dn + m * sn * sn - 1.0).abs() < 1e-14,
                "dn^2 + m sn^2 != 1 at u={u}, m={m}"
            );
        }
    }
}

#[test]
fn ellipj_has_the_period_the_quarter_period_predicts() {
    // cn has period 4K(m) and cn(K) = 0 -- the two facts that tie `ellipj` to `ellipk`, which are
    // otherwise unrelated pieces of code in this module.
    for i in 0..20 {
        let m = i as f64 / 20.0 * 0.95;
        let k = ellipk(m);
        let (_, cn_k, _) = ellipj(k, m);
        assert!(cn_k.abs() < 1e-13, "cn(K, {m}) = {cn_k}, expected 0");
        for &u in &[0.3, 1.1, 2.9] {
            let (sn0, cn0, _) = ellipj(u, m);
            let (sn4, cn4, _) = ellipj(u + 4.0 * k, m);
            assert!((sn0 - sn4).abs() < 1e-12, "sn not 4K-periodic at m={m}");
            assert!((cn0 - cn4).abs() < 1e-12, "cn not 4K-periodic at m={m}");
        }
    }
}

#[test]
fn ellipj_starts_at_a_maximum_which_is_what_makes_it_the_duffing_solution() {
    // q(0) = A, q'(0) = 0 is the initial condition `duffing_displacement` promises, and it comes
    // entirely from cn(0) = 1 with a vanishing derivative. cn' = -sn*dn, and sn(0) = 0.
    for i in 0..10 {
        let m = i as f64 / 10.0 * 0.9;
        let (sn, cn, dn) = ellipj(0.0, m);
        assert_eq!(cn, 1.0);
        assert_eq!(sn, 0.0);
        assert_eq!(dn, 1.0);
    }
}

#[test]
fn ellipj_solves_the_pendulum_ode_it_is_the_solution_of() {
    // d/du cn = -sn dn, checked by a central difference. This is the one property that would catch
    // a Landen descent that converged to a *different* function of the right shape.
    let h = 1e-6;
    for &m in &[0.1, 0.45, 0.8] {
        for &u in &[0.4, 1.7, 3.3] {
            let (sn, _, dn) = ellipj(u, m);
            let (_, cn_p, _) = ellipj(u + h, m);
            let (_, cn_m, _) = ellipj(u - h, m);
            let d = (cn_p - cn_m) / (2.0 * h);
            assert!(
                (d + sn * dn).abs() < 1e-9,
                "cn' = -sn dn fails at u={u}, m={m}: {d} vs {}",
                -sn * dn
            );
        }
    }
}
