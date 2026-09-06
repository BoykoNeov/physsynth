//! The matrix-free GMRES the von Karman plate's Newton step solves with (plan §5 Part 2).
//!
//! Asserted here, against **hand-built** matrices whose solution is known in closed form, before
//! `plate.rs` is allowed to call it. This is Part 1's discipline applied one level down: "the
//! plate converged" is a weak bar for a linear solver, because a Newton iteration that overshoots
//! and is caught by the line search still ends up at the root. These systems have no line search
//! and no second chance.
//!
//! Plan §7 trap 4 — "GMRES restarts change the iterate; pin it and record it rather than tuning it
//! per fixture" — is discharged by [`the_restart_length_changes_the_cost_and_not_the_answer`],
//! which is the precise form of that worry: the restart is allowed to move the *count*, and it is
//! not allowed to move the answer.

use physsynth_core::krylov::gmres;

/// Never returned by any operator here — GMRES is generic over the error and these cannot fail.
#[derive(Debug, PartialEq)]
struct Never;

/// `A x` for a dense row-major `n x n`.
fn matvec(a: &[f64], n: usize, x: &[f64]) -> Vec<f64> {
    (0..n)
        .map(|i| (0..n).map(|j| a[i * n + j] * x[j]).sum())
        .collect()
}

fn l2(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn rel_err(got: &[f64], want: &[f64]) -> f64 {
    let d: Vec<f64> = (0..got.len()).map(|i| got[i] - want[i]).collect();
    l2(&d) / l2(want)
}

/// A deterministic xorshift64 stream in `[-0.5, 0.5)` — the same on every platform.
fn noise(n: usize, salt: u64) -> Vec<f64> {
    let mut s = 0x2545_f491_4f6c_dd1d ^ salt.wrapping_mul(0x9e37_79b9_7f4a_7c15);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 11) as f64) / 9_007_199_254_740_992.0 - 0.5
        })
        .collect()
}

/// A well-conditioned **non-symmetric** `n x n`: a strong diagonal plus deterministic noise.
///
/// Non-symmetric on purpose. The plate's `J = I - c A^-1 K` is not symmetric — `A^-1 K` is a
/// product of two symmetric operators, which is not — so a solver asserted only on an SPD system
/// would be asserted on a case the caller never presents. Conjugate gradients would pass that and
/// fail here; GMRES is in the batch precisely because it does not need the symmetry.
fn nonsymmetric(n: usize, salt: u64) -> Vec<f64> {
    let mut a = noise(n * n, salt);
    for i in 0..n {
        a[i * n + i] += 4.0 + i as f64 * 0.1;
    }
    a
}

/// Near the plate's own operator: `I` plus a small non-symmetric perturbation.
///
/// This is the shape GMRES actually meets (`J = I - c A^-1 K`, 2.4%-12.4% from `I` on Part 1's
/// fixtures), and it converges in a handful of products because the spectrum is clustered at 1.
fn near_identity(n: usize, scale: f64, salt: u64) -> Vec<f64> {
    let mut a: Vec<f64> = noise(n * n, salt).iter().map(|v| scale * v).collect();
    for i in 0..n {
        a[i * n + i] += 1.0;
    }
    a
}

const TIGHT: f64 = 1e-13;

#[test]
fn gmres_solves_a_nonsymmetric_system_to_the_solution_it_was_built_from() {
    // `b` is formed as `A x_true`, so the answer is known exactly rather than compared against a
    // second solver that could share a mistake.
    for n in [4usize, 10, 25] {
        let a = nonsymmetric(n, 7);
        let x_true = noise(n, 11);
        let b = matvec(&a, n, &x_true);
        let r = gmres::<Never, _>(|d| Ok(matvec(&a, n, d)), &b, n + 5, 200, TIGHT)
            .expect("this operator cannot fail");
        assert!(r.converged, "n={n}: GMRES did not reach {TIGHT:e}");
        let err = rel_err(&r.x, &x_true);
        assert!(err < 1e-11, "n={n}: relative error {err:.3e}");
        // Krylov's own guarantee: with a full-length restart, no more products than the dimension.
        assert!(
            r.products <= n,
            "n={n}: {} products for an n-dimensional space",
            r.products
        );
    }
}

#[test]
fn the_restart_length_changes_the_cost_and_not_the_answer() {
    // Plan §7 trap 4, in its exact form. A restart shorter than the dimension throws away the
    // Arnoldi basis and starts again, which is a genuinely different sequence of iterates -- so
    // the product count must move, and the converged answer must not.
    let n = 25usize;
    let a = nonsymmetric(n, 3);
    let x_true = noise(n, 5);
    let b = matvec(&a, n, &x_true);

    let mut answers: Vec<(usize, Vec<f64>, usize)> = Vec::new();
    for restart in [3usize, 8, 30] {
        let r = gmres::<Never, _>(|d| Ok(matvec(&a, n, d)), &b, restart, 4000, TIGHT)
            .expect("this operator cannot fail");
        assert!(r.converged, "restart={restart}: did not converge");
        let err = rel_err(&r.x, &x_true);
        assert!(err < 1e-10, "restart={restart}: relative error {err:.3e}");
        answers.push((restart, r.x, r.products));
    }

    // The cost moves: a short restart forgets, and pays for it.
    assert!(
        answers[0].2 > answers[2].2,
        "restart 3 spent {} products and restart 30 spent {} -- the restart is not being felt, \
         so this test is not measuring what it says",
        answers[0].2,
        answers[2].2
    );
    // The answer does not, to a hair above the tolerance each was solved to.
    for i in 1..answers.len() {
        let d = rel_err(&answers[i].1, &answers[0].1);
        assert!(
            d < 1e-10,
            "restart {} and restart {} disagree by {d:.3e}",
            answers[0].0,
            answers[i].0
        );
    }
}

#[test]
fn the_identity_is_solved_by_one_product() {
    // At zero amplitude the plate's Jacobian **is** the identity (plan §1), and a Newton step
    // there must not cost a Krylov subspace. One product proves the residual is already zero.
    let n = 12usize;
    let b = noise(n, 2);
    let r = gmres::<Never, _>(|d| Ok(d.to_vec()), &b, 30, 100, TIGHT)
        .expect("this operator cannot fail");
    assert!(r.converged);
    assert_eq!(r.products, 1, "the identity took {} products", r.products);
    assert!(rel_err(&r.x, &b) < 1e-15, "x should be b exactly");
}

#[test]
fn a_near_identity_operator_converges_in_a_handful_of_products() {
    // The regime the plate is actually in. This is not a performance test -- it is the claim that
    // makes the unpreconditioned choice defensible (plan §1): the preconditioner is already inside
    // `J`, so a clustered spectrum is what GMRES is handed, and the cost is small and *bounded*.
    let n = 40usize;
    // Measured at this dimension and `TIGHT`: 8, 13, 25 products for spectra spread 0.018, 0.092
    // and 0.367 around 1. The bars are those numbers with room, and the *shape* is the claim --
    // the cost tracks how far the operator is from `I` and stays under the dimension, which is
    // what "the preconditioner is already inside `J`" has to mean if it means anything.
    let mut counts: Vec<usize> = Vec::new();
    for (scale, bar) in [(0.01f64, 12usize), (0.05, 18), (0.2, 32)] {
        let a = near_identity(n, scale, 13);
        let x_true = noise(n, 17);
        let b = matvec(&a, n, &x_true);
        let r = gmres::<Never, _>(|d| Ok(matvec(&a, n, d)), &b, 30, 200, TIGHT)
            .expect("this operator cannot fail");
        assert!(r.converged, "scale={scale}: did not converge");
        assert!(rel_err(&r.x, &x_true) < 1e-10);
        assert!(
            r.products <= bar && r.products < n,
            "scale={scale}: {} products (bar {bar}, dimension {n})",
            r.products
        );
        counts.push(r.products);
    }
    assert!(
        counts[0] < counts[1] && counts[1] < counts[2],
        "the cost {counts:?} does not track the distance from the identity, so the bars above          are three unrelated numbers rather than one claim"
    );
}

#[test]
fn a_zero_right_hand_side_costs_no_product_at_all() {
    // A converged Newton step hands GMRES `-G(w)` with `G` at the tolerance; an exactly zero one
    // must not buy a Krylov space to discover that zero solves it.
    let n = 9usize;
    let b = vec![0.0; n];
    let mut calls = 0usize;
    let r = gmres::<Never, _>(
        |d| {
            calls += 1;
            Ok(d.to_vec())
        },
        &b,
        30,
        100,
        TIGHT,
    )
    .expect("this operator cannot fail");
    assert_eq!(calls, 0, "the operator was applied to a zero system");
    assert_eq!(r.products, 0);
    assert!(r.converged && r.x.iter().all(|&v| v == 0.0));
}

#[test]
fn the_product_cap_stops_it_and_it_says_so() {
    // The contract the Newton driver relies on: a hard operator does not loop forever and does not
    // silently return an approximation labelled converged. `converged=false` with the residual it
    // actually reached is what lets the caller record a stall instead of believing one.
    let n = 30usize;
    // The cyclic shift `A e_j = e_{j+1}`: orthogonal, perfectly conditioned, and the textbook
    // matrix on which GMRES makes **no** progress at all until iteration `n`. Chosen over an
    // ill-conditioned one on purpose -- a cap that bites here is the cap biting, not the
    // arithmetic falling apart, so the second half of this test is a fair control.
    let mut a = vec![0.0f64; n * n];
    for i in 0..n {
        a[i * n + (i + n - 1) % n] = 1.0;
    }
    let b = noise(n, 23);
    let r = gmres::<Never, _>(|d| Ok(matvec(&a, n, d)), &b, 30, 5, 1e-14)
        .expect("this operator cannot fail");
    assert_eq!(r.products, 5, "the cap was not respected");
    assert!(!r.converged, "a capped solve must not report convergence");
    assert!(
        r.residual.is_finite() && r.residual > 1e-14,
        "the reported residual {:.3e} should be the one it actually reached",
        r.residual
    );

    // And with the cap lifted it does converge, so the failure above is the cap and not the solver.
    let full = gmres::<Never, _>(|d| Ok(matvec(&a, n, d)), &b, n + 2, 500, 1e-12)
        .expect("this operator cannot fail");
    assert!(full.converged, "residual {:.3e}", full.residual);
    let ax = matvec(&a, n, &full.x);
    assert!(rel_err(&ax, &b) < 1e-10);
}

#[test]
fn the_reported_residual_is_the_true_one() {
    // The Givens recursion tracks `||b - A x||` without forming it. If that bookkeeping were wrong
    // the solver would stop early or late and every test above would still pass, because they all
    // check the *answer* against a known `x_true` at a tolerance the recursion chose. This is the
    // one test that closes the loop by forming the residual explicitly.
    let n = 20usize;
    let a = nonsymmetric(n, 31);
    let b = noise(n, 37);
    for rtol in [1e-2f64, 1e-6, 1e-12] {
        let r = gmres::<Never, _>(|d| Ok(matvec(&a, n, d)), &b, 6, 500, rtol)
            .expect("this operator cannot fail");
        assert!(r.converged, "rtol={rtol:e}: did not converge");
        let ax = matvec(&a, n, &r.x);
        let resid: Vec<f64> = (0..n).map(|i| b[i] - ax[i]).collect();
        let true_rel = l2(&resid) / l2(&b);
        assert!(
            true_rel <= 5.0 * rtol,
            "rtol={rtol:e}: reported {:.3e}, true {true_rel:.3e}",
            r.residual
        );
        // And the recursion's own number is the true one, not merely below the bar. Loose because
        // a restarted run rebuilds `g[0]` from a freshly formed residual, so the two agree to the
        // conditioning of the basis rather than to rounding.
        assert!(
            (r.residual - true_rel).abs() <= 1e-6 * rtol.max(1e-12) + 1e-12 * true_rel.max(1.0),
            "rtol={rtol:e}: recursion says {:.3e}, truth is {true_rel:.3e}",
            r.residual
        );
    }
}
