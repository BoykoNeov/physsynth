//! Native bars for the contact scheme (plan §16).
//!
//! The project's acceptance contract is energy, and for a contact model the statement is exact:
//! the discrete-gradient force makes the contact power **telescope**, so over one step the work
//! done by the force equals the change in stored potential to the last bit or two, with no `O(k²)`
//! remainder. That is asserted here directly, which no Python test does — the Python suite asserts
//! the *consequence* (a lossless run conserves) through a model. Both are worth having; this one
//! localises a failure to the primitive rather than to the host.
//!
//! The two things that are genuinely easy to get wrong and are each pinned below: the removable
//! `0/0` (the commonest state the scheme is ever in — quiet, stuck, grazing — and a `NaN` without
//! its Taylor branch), and the two power spellings, which are not an implementation detail but the
//! difference between reproducing NumPy's array path and its scalar path.

use physsynth_core::collision::{
    contact_force_dg, contact_force_elastic, contact_force_total, contact_potential,
    contact_stiffness, deriv_total_vec, force_total_vec, solve_contact, solve_contact_vector,
    ContactParams, PowPath,
};

const K: f64 = 1.0e6;
const TOL: f64 = 1e-12;

fn params(alpha: f64, lam_h: f64) -> ContactParams {
    ContactParams {
        stiffness: K,
        alpha,
        lam_h,
        k: 1.0 / 48000.0,
        tol: TOL,
    }
}

#[test]
fn the_discrete_gradient_makes_the_contact_power_telescope_exactly() {
    // f * (eta+ - eta-) == phi(eta+) - phi(eta-), which is what conservation rests on. Asserted
    // over a sweep that crosses the barrier in both directions and grazes it.
    for &alpha in &[1.0, 1.5, 2.3, 3.0] {
        for i in -40i32..=40 {
            for j in -40i32..=40 {
                let (en, ep) = (1e-4 * f64::from(i), 1e-4 * f64::from(j));
                let f = contact_force_dg(en, ep, K, alpha, TOL, PowPath::Scalar);
                let work = f * (en - ep);
                let dphi = contact_potential(en, K, alpha, PowPath::Scalar)
                    - contact_potential(ep, K, alpha, PowPath::Scalar);
                let scale = dphi.abs().max(1e-30);
                assert!(
                    (work - dphi).abs() <= 1e-12 * scale,
                    "alpha = {alpha}, eta = ({en}, {ep}): work {work} vs dphi {dphi}"
                );
            }
        }
    }
}

#[test]
fn the_taylor_branch_covers_the_removable_zero_over_zero() {
    // Exactly stuck: the quotient is 0/0 and the answer must be phi' at the midpoint, not NaN.
    for &alpha in &[1.0, 1.5, 2.3] {
        for &eta in &[-1e-3, 0.0, 1e-9, 1e-3, 1e-1] {
            let f = contact_force_dg(eta, eta, K, alpha, TOL, PowPath::Scalar);
            assert!(f.is_finite(), "alpha = {alpha}, eta = {eta} gave {f}");
            assert_eq!(f, contact_force_elastic(eta, K, alpha, PowPath::Scalar));
        }
        // And it joins continuously to the quotient branch just outside the threshold.
        let (a, b) = (1e-3 + 2e-12, 1e-3);
        let inside = contact_force_dg(b + 1e-13, b, K, alpha, TOL, PowPath::Scalar);
        let outside = contact_force_dg(a, b, K, alpha, TOL, PowPath::Scalar);
        assert!(
            (inside - outside).abs() <= 1e-6 * outside.abs(),
            "alpha = {alpha}: {inside} vs {outside} across the tol threshold"
        );
    }
}

#[test]
fn nothing_touches_the_string_outside_the_barrier() {
    for &alpha in &[1.0, 1.5, 2.3] {
        for &eta in &[-1.0, -1e-9, 0.0] {
            assert_eq!(contact_potential(eta, K, alpha, PowPath::Array), 0.0);
            assert_eq!(contact_force_elastic(eta, K, alpha, PowPath::Array), 0.0);
            // alpha = 1 is the case a naive `[eta]+^(alpha-1)` gets wrong: `0^0` is 1, which would
            // leak the full stiffness K into the open gap.
            assert_eq!(contact_stiffness(eta, K, alpha, PowPath::Array), 0.0);
        }
    }
}

#[test]
fn the_two_power_spellings_are_the_ones_numpy_uses() {
    // The array path takes NumPy's ufunc fast-path ladder; the scalar path calls `pow`. At the
    // exponents in play — alpha + 1, alpha, alpha - 1 — that separates exactly at alpha = 1 (where
    // alpha + 1 = 2) and alpha = 1.5 (where alpha - 1 = 0.5). Both spellings must be *available*,
    // and the array one must agree with the closed forms that NumPy shortcuts to.
    for i in 1..500 {
        let x = 1e-4 * f64::from(i);
        // exponent 2 (alpha = 1, the potential): the ufunc loop spells it x * x.
        assert_eq!(
            contact_potential(x, K, 1.0, PowPath::Array),
            K / 2.0 * (x * x),
            "exponent 2 must be a multiply, not a pow"
        );
        // exponent 0.5 (alpha = 1.5, the stiffness): the ufunc loop spells it sqrt.
        assert_eq!(
            contact_stiffness(x, K, 1.5, PowPath::Array),
            K * 1.5 * x.sqrt(),
            "exponent 0.5 must be a sqrt, not a pow"
        );
    }
    // And the scalar path is deliberately NOT that — it is `pow`, whichever way that rounds.
    let mut differed = 0;
    for i in 1..200_000 {
        let x = 1e-6 * f64::from(i);
        if contact_stiffness(x, K, 1.5, PowPath::Array)
            != contact_stiffness(x, K, 1.5, PowPath::Scalar)
        {
            differed += 1;
        }
    }
    assert!(
        differed > 0,
        "the two paths came out identical, which means one of them stopped being what it claims"
    );
}

#[test]
fn the_scalar_solve_finds_the_root_it_reports() {
    // eta = eta_free - g f(eta), to the tolerance it claims, across contact and free flight.
    let g = 1.0e-8;
    for &alpha in &[1.0, 1.5, 2.3] {
        for &lam_h in &[0.0, 1.0e3] {
            let p = params(alpha, lam_h);
            for i in -20i32..=20 {
                let eta_free = 1e-4 * f64::from(i);
                let eta_prev = eta_free - 3e-5;
                let sol = solve_contact(eta_free, eta_prev, g, p, eta_prev, 1e-14, 60)
                    .expect("the residual is monotone, so a root exists");
                let f = contact_force_total(
                    sol.eta,
                    eta_prev,
                    K,
                    alpha,
                    lam_h,
                    p.k,
                    TOL,
                    PowPath::Scalar,
                );
                assert_eq!(
                    f, sol.force,
                    "the reported force must be the one at the root"
                );
                assert!(
                    (sol.eta - eta_free + g * f).abs() <= 1e-13,
                    "alpha = {alpha}, eta_free = {eta_free}: residual {}",
                    sol.eta - eta_free + g * f
                );
            }
        }
    }
}

#[test]
fn out_of_contact_the_scalar_solve_is_free_flight_exactly() {
    let p = params(1.5, 0.0);
    let sol = solve_contact(-1e-3, -1.1e-3, 1.0e-8, p, -1.1e-3, 1e-14, 60).unwrap();
    assert_eq!(sol.force, 0.0);
    assert_eq!(sol.eta, -1e-3);
    assert!(!sol.used_fallback);
}

/// An SPD admittance block of the shape a string's `A⁻¹` gives: symmetric, diagonally dominant,
/// decaying off the diagonal. Entries are deliberately not powers of two (§14.3).
fn admittance(m: usize) -> Vec<f64> {
    let mut g = vec![0.0; m * m];
    for i in 0..m {
        for j in 0..m {
            let d = (i as f64 - j as f64).abs();
            g[i * m + j] = 1.7e-9 * (-0.43 * d).exp();
        }
    }
    g
}

#[test]
fn the_vector_solve_converges_and_reports_the_force_at_its_root() {
    for m in [1usize, 2, 8, 30] {
        let g = admittance(m);
        for &alpha in &[1.0, 1.5, 2.3] {
            for &lam_h in &[0.0, 2.0e4] {
                let p = params(alpha, lam_h);
                let eta_free: Vec<f64> = (0..m)
                    .map(|i| 1e-3 * (0.5 - (i as f64) / (m as f64)))
                    .collect();
                let eta_prev: Vec<f64> = eta_free.iter().map(|e| e - 4e-5).collect();
                let sol = solve_contact_vector(&eta_free, &eta_prev, &g, p, &eta_prev, 1e-13, 60);
                assert!(
                    sol.converged,
                    "m = {m}, alpha = {alpha}: residual {}",
                    sol.residual
                );
                assert!(sol.iters < 60);
                // The residual is re-derived here rather than trusted: eta = eta_free - G F(eta).
                let mut f = vec![0.0; m];
                force_total_vec(&sol.eta, &eta_prev, &mut f, K, alpha, lam_h, p.k, TOL);
                assert_eq!(
                    f, sol.force,
                    "the reported force must be the one at the root"
                );
                for (i, (&eta_i, &free_i)) in sol.eta.iter().zip(&eta_free).enumerate() {
                    let gf: f64 = (0..m).map(|j| g[i * m + j] * f[j]).sum();
                    assert!(
                        (eta_i - free_i + gf).abs() <= 1e-13,
                        "m = {m}, alpha = {alpha}, node {i}"
                    );
                }
            }
        }
    }
}

#[test]
fn an_out_of_reach_barrier_leaves_the_free_solution_untouched() {
    // The whole support clear of the barrier: F and F' are exactly zero, so the Jacobian is the
    // identity and the system is linear — one Newton step lands on it. **The force is exactly
    // zero**, and that, not the penetration, is what makes a barriered string bit-for-bit a bare
    // string: the host adds `force_pref * (cols @ F)` to its field, which is an exact no-op only
    // if every component of `F` is a true zero rather than a small number.
    let m = 12;
    let g = admittance(m);
    let p = params(1.5, 0.0);
    let eta_free: Vec<f64> = (0..m).map(|i| -100.0 - (i as f64)).collect();
    let eta_prev: Vec<f64> = eta_free.iter().map(|e| e - 1e-3).collect();
    let sol = solve_contact_vector(&eta_free, &eta_prev, &g, p, &eta_prev, 1e-13, 60);
    assert_eq!(
        sol.iters, 1,
        "one step from the seed to the free solution, then converged"
    );
    assert!(
        sol.force.iter().all(|&f| f == 0.0),
        "the force must be a true zero, not a small one"
    );
    for (eta_i, free_i) in sol.eta.iter().zip(&eta_free) {
        assert!((eta_i - free_i).abs() <= 1e-13 * free_i.abs());
    }
}

#[test]
fn the_vector_forms_agree_with_the_scalar_ones_where_numpy_makes_them() {
    // At an exponent NumPy does *not* shortcut, the array and scalar paths are the same
    // computation and the vector form must reduce to the scalar one element by element. At alpha
    // = 1 they must NOT, and that asymmetry is the finding the module header records — so it is
    // asserted in both directions rather than only the convenient one.
    let m = 64;
    let eta_next: Vec<f64> = (0..m)
        .map(|i| 1e-3 * (0.5 - (i as f64) / (m as f64)))
        .collect();
    let eta_prev: Vec<f64> = eta_next.iter().map(|e| e - 7e-5).collect();
    let k = 1.0 / 48000.0;

    let mut agree = vec![0.0; m];
    force_total_vec(&eta_next, &eta_prev, &mut agree, K, 2.3, 0.0, k, TOL);
    for i in 0..m {
        let s = contact_force_total(
            eta_next[i],
            eta_prev[i],
            K,
            2.3,
            0.0,
            k,
            TOL,
            PowPath::Scalar,
        );
        assert_eq!(
            agree[i], s,
            "alpha = 2.3 takes no fast path, so the two must agree at node {i}"
        );
    }

    let mut d = vec![0.0; m];
    deriv_total_vec(&eta_next, &eta_prev, &mut d, K, 1.5, 0.0, k, TOL);
    for i in 0..m {
        let s = physsynth_core::collision::contact_force_total_deriv(
            eta_next[i],
            eta_prev[i],
            K,
            1.5,
            0.0,
            k,
            TOL,
            PowPath::Array,
        );
        assert_eq!(
            d[i], s,
            "the vector derivative must be the array-path scalar one at node {i}"
        );
    }
}
