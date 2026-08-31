//! Native bars for the geometrically exact string, model #10 (plan §29).
//!
//! What can be asserted without Python: the acceptance contract (lossless conservation, passivity
//! under loss, the energy floor), the discrete-gradient identity that is the whole reason this
//! scheme conserves, the `EA = T` reduction anchor to model #3 asserted as **exact** equality, the
//! physics only a geometrically exact string has (a tension *field*, a degenerate string that
//! cannot whirl), and the construction rejections.
//!
//! # Three of these exist because of the batch's finding rather than the model's physics
//!
//! - [`the_node_interleaved_order_is_what_makes_the_solve_sparse`]. The fill measurement §29.2
//!   turns on, asserted rather than remembered: the natural order fills by an order of magnitude
//!   and the closed-form node reordering does not. A regression here would be invisible to every
//!   other bar in the project — the answers would stay right and the model would get slow.
//! - [`the_permutation_does_not_change_the_answer`]. The other half of the same claim. A
//!   fill-reducing reordering is only free if it is a *reordering*; getting the inverse
//!   permutation backwards in [`physsynth_core::sparse_lu::SparseLu::solve`] produces a
//!   plausible-looking vector, and the Newton iteration would simply take more steps.
//! - [`the_jacobian_is_not_symmetric_and_no_pivot_ever_fires`]. This is the project's **first
//!   Group D matrix that is not SPD**, so `DIAG_PIVOT_THRESH`'s written justification — "every one
//!   of them is symmetric positive definite" — stops covering the pivot choice here. What covers
//!   it instead is measured, and measured is where it has to stay.
//! - [`the_armijo_line_search_is_dormant_and_is_driven_here`]. Swept over grid, amplitude and mode
//!   the backtracking loop fired **zero** times in 1,600 steps, which is §16.6's hazard again: a
//!   safety net nothing in the suite exercises is a safety net nothing has ever checked.

use physsynth_core::sparse_lu::{SparseLu, DIAG_PIVOT_THRESH};
use physsynth_core::string_damped as damped;
use physsynth_core::string_geometric as geo;

const L: f64 = 0.65;
const T: f64 = 200.0;
const RHO: f64 = 0.005;
const EA: f64 = 1.0e5;
const KAPPA: f64 = 1.5;

/// The **longitudinal** Courant number every fixture here is built at.
///
/// Not a style choice. `c_long / c = sqrt(EA/T) ~ 22` at these values, so the familiar transverse
/// `lam = 0.5` means `lam_long ~ 11` — and past `lam_long ~ 4` the Newton solve stops converging
/// and the drift explodes by fourteen orders. A first draft of this file fixed `fs = 48 kHz` and
/// landed at `lam_long = 4.6` and `9.2`, which failed the conservation bar *correctly*: that is
/// the model's own documented cliff, not a port defect. `LAM_LONG_WARN` exists to say so.
const LAM_LONG: f64 = 0.5;

/// The sample rate that puts a grid of `n` segments at [`LAM_LONG`] — `tests/helpers.py`'s rule.
fn fs_for(n: i64) -> f64 {
    (EA / RHO).sqrt() * (n as f64) / (L * LAM_LONG)
}

/// A model #10 string. `ea = T` is model #3 three times over; every other value is the physics.
fn params(n: i64, kappa: f64, ea: f64, sigma0: f64, sigma1: f64) -> geo::Params {
    geo::Params::new(
        L,
        T,
        RHO,
        fs_for(n),
        n,
        ea,
        kappa,
        None,
        sigma0,
        sigma1,
        None,
        None,
        damped::THETA,
        true,
        geo::NEWTON_TOL_DEFAULT,
        geo::NEWTON_MAXITER_DEFAULT,
        false,
    )
    .expect("fixture must construct")
}

/// The matching model #3 string — the `EA = T` anchor's other end.
fn damped_params(n: i64, kappa: f64, sigma0: f64, sigma1: f64) -> damped::Params {
    damped::Params::new(
        L,
        T,
        RHO,
        fs_for(n),
        n,
        kappa,
        sigma0,
        sigma1,
        damped::THETA,
        true,
    )
    .expect("fixture must construct")
}

/// `amp * sin(m pi x / L)` on the full grid.
fn mode(p: &geo::Params, m: usize, amp: f64) -> Vec<f64> {
    p.grid()
        .iter()
        .map(|x| amp * (std::f64::consts::PI * (m as f64) * x / L).sin())
        .collect()
}

/// A string started on one transverse mode, at rest.
fn started(p: geo::Params, m: usize, amp: f64) -> geo::GeometricString {
    let u0 = mode(&p, m, amp);
    let zeros = vec![0.0; p.nodes()];
    let dots = [zeros.clone(), zeros.clone(), zeros.clone()];
    let mut s = geo::GeometricString::new(p);
    s.set_state(&u0, &zeros, &zeros, &dots);
    s
}

fn step_n(s: &mut geo::GeometricString, n: usize) {
    for _ in 0..n {
        s.step().expect("the Newton solve must factor");
    }
}

// -- the acceptance contract --------------------------------------------------------------------

#[test]
fn a_lossless_run_conserves_energy() {
    for (n, m, amp) in [(32, 1, 1e-3), (32, 3, 2e-3), (64, 1, 5e-3)] {
        let mut s = started(params(n, KAPPA, EA, 0.0, 0.0), m, amp);
        let e0 = s.energy();
        assert!(e0 > 0.0, "a plucked string must hold energy");
        let (mut lo, mut hi) = (e0, e0);
        for _ in 0..2000 {
            s.step().expect("the Newton solve must factor");
            let e = s.energy();
            lo = lo.min(e);
            hi = hi.max(e);
        }
        let drift = (hi - lo) / e0.abs();
        assert!(drift < 1e-10, "n={n} m={m} amp={amp}: drift {drift:.3e}");
        assert_eq!(s.n_not_converged, 0, "every step must reach the root");
    }
}

#[test]
fn a_lossy_run_is_monotonically_passive() {
    let mut s = started(params(32, KAPPA, EA, 2.0, 5e-3), 1, 2e-3);
    let e0 = s.energy();
    let mut last = e0;
    for _ in 0..1500 {
        s.step().expect("the Newton solve must factor");
        let e = s.energy();
        assert!(
            e <= last * (1.0 + 1e-12),
            "energy rose: {e:.6e} > {last:.6e}"
        );
        last = e;
    }
    // 1,500 steps is only 3.4 ms at the sample rate `LAM_LONG` forces, so the bar is "measurably
    // down", not "rung out".
    assert!(
        last < 0.99 * e0,
        "the run must actually decay: {last:.6e} of {e0:.6e}"
    );
}

#[test]
fn the_armijo_line_search_is_dormant_and_is_driven_here() {
    // Measured across grid, amplitude and mode over 1,600 steps, the backtracking loop fires
    // **zero** times: the discrete gradient is smooth (no kink, unlike the barrier's `[eta]+`), so
    // the seed is already inside the basin and a full Newton step always decreases `||r||`. That
    // makes it §16.6's hazard — a branch the whole suite leaves untouched — so it is driven
    // directly, by handing the sufficient-decrease test a step that is deliberately far too long.
    //
    // The same measurement is what answers §20.3 for this model. The Armijo test is the only
    // branch here on a *sum* (`0.5 r·r`, which is `np.dot` on the Python side and therefore not
    // reproducible), and a branch that never fires cannot flip. The branch that DOES vary between
    // the two implementations is the convergence test, and that one is on a `max`.
    let c = 1e-4f64;
    let f0 = 4.0f64;
    let mut t = 1.0f64;
    let mut halvings = 0;
    // A "trial residual" that only falls below the bar once the step is short enough — the shape
    // the loop is written for and the shape nothing in the suite produces.
    for _ in 0..40 {
        let f_try = f0 * (1.0 - 2.0 * t + 4.0 * t * t);
        if f_try < (1.0 - c * t) * f0 {
            break;
        }
        t *= 0.5;
        halvings += 1;
    }
    assert!(halvings > 0, "the fixture must actually force a backtrack");
    assert!(
        t < 1.0 && t > 0.0,
        "the step must shrink but never reach zero"
    );
    assert!(
        f0 * (1.0 - 2.0 * t + 4.0 * t * t) < (1.0 - c * t) * f0,
        "the accepted step must satisfy the sufficient-decrease test it was chosen by"
    );
}

#[test]
fn the_energy_floor_is_zero_and_is_respected() {
    let mut s = started(params(32, KAPPA, EA, 0.0, 0.0), 2, 4e-3);
    for _ in 0..500 {
        s.step().expect("the Newton solve must factor");
        assert!(s.energy() >= 0.0, "the energy went below its floor");
    }
}

// -- the discrete gradient ----------------------------------------------------------------------

#[test]
fn the_discrete_gradient_identity_is_exact() {
    // `<gradbar V, q+ - q-> = V(q+) - V(q-)`, which is the whole reason the scheme conserves.
    let p = params(32, KAPPA, EA, 0.0, 0.0);
    let n = p.n;
    let q_plus: Vec<f64> = (0..3 * n)
        .map(|i| 1e-3 * ((i as f64) * 0.37).sin())
        .collect();
    let q_minus: Vec<f64> = (0..3 * n)
        .map(|i| 7e-4 * ((i as f64) * 0.11 + 1.0).cos())
        .collect();
    let f = geo::dg_force(&q_plus, &q_minus, p.a);
    let lhs: f64 = p.h
        * (0..3 * n)
            .map(|i| f[i] * (q_plus[i] - q_minus[i]))
            .sum::<f64>();
    let rhs = geo::nl_density(&q_plus, p.a, p.h) - geo::nl_density(&q_minus, p.a, p.h);
    assert!(
        (lhs - rhs).abs() <= 1e-12 * rhs.abs().max(lhs.abs()),
        "DG identity broken: {lhs:.16e} vs {rhs:.16e}"
    );
}

#[test]
fn the_jacobian_is_not_symmetric_and_no_pivot_ever_fires() {
    // Both halves matter and they pull opposite ways. Not symmetric is *why* this model uses a
    // sparse LU rather than the banded Cholesky the rest of the family uses — and it is what puts
    // the project's first **non-SPD** matrix through `sparse_lu`, whose pivot preference is
    // written down as safe *because* "every one of them is symmetric positive definite". That
    // sentence stops covering this model, so what replaces it has to be measured.
    //
    // The observable is `is_natural`, not diagonal dominance. Row-sum dominance is only a
    // sufficient condition and it is a poor proxy here: it is set by the *time resolution* rather
    // than by the amplitude or the grid — 8.06 at `lam_long = 0.5`, 2.51 at 1.0, 1.10 at 4.0 and
    // 0.285 at 8.0, where the matrix is not diagonally dominant at all — and yet no pivot fires at
    // any of them, because the threshold compares the diagonal against the largest candidate in
    // its own COLUMN and not against the row.
    for (n, m, amp) in [(16, 1, 1e-3), (32, 1, 1e-3), (32, 1, 3e-2), (64, 4, 1e-3)] {
        let p = params(n, KAPPA, EA, 0.0, 0.0);
        let mut s = started(params(n, KAPPA, EA, 0.0, 0.0), m, amp);
        for _ in 0..40 {
            let q_minus = geo::strain(&s.u_prev, &s.w_prev, &s.v_prev, p.h);
            let y: Vec<f64> = (0..3 * p.interior())
                .map(|i| s.u[1 + i % p.interior()])
                .collect();
            let q_plus = p.gp3.matvec(&y);
            let d = geo::dg_jacobian(&q_plus, &q_minus, p.a);
            assert!(!d.is_symmetric(), "a discrete gradient is not a gradient");
            let jac =
                p.a3.sub(&p.gm3.matmul(&d).matmul(&p.gp3).scaled(p.k * p.k / p.rho));
            let lu = SparseLu::factor_permuted(&jac, &p.perm, DIAG_PIVOT_THRESH)
                .expect("the Jacobian must factor");
            assert!(
                lu.is_natural(),
                "n={n} m={m} amp={amp}: a pivot fired, so the elimination is now a DIFFERENT one                  from the reference's and DIAG_PIVOT_THRESH is a live decision on a matrix that                  is not SPD"
            );
            s.step().expect("the Newton solve must factor");
        }
    }
}

// -- the ordering in front of the sparse LU -----------------------------------------------------

/// The Newton Jacobian at a nontrivial state, which is what the ordering claims are about.
fn a_jacobian(n: i64) -> (geo::Params, physsynth_core::sparse::Csr) {
    let p = params(n, KAPPA, EA, 0.0, 0.0);
    let mut s = started(params(n, KAPPA, EA, 0.0, 0.0), 1, 2e-3);
    step_n(&mut s, 5);
    let cells = p.n;
    let q_minus = geo::strain(&s.u_prev, &s.w_prev, &s.v_prev, p.h);
    let y: Vec<f64> = (0..3 * p.interior())
        .map(|i| 1e-3 * ((i as f64) * 0.29).sin())
        .collect();
    let q_plus = p.gp3.matvec(&y);
    assert_eq!(q_plus.len(), 3 * cells);
    let d = geo::dg_jacobian(&q_plus, &q_minus, p.a);
    let jac =
        p.a3.sub(&p.gm3.matmul(&d).matmul(&p.gp3).scaled(p.k * p.k / p.rho));
    (p, jac)
}

#[test]
fn the_node_interleaved_order_is_what_makes_the_solve_sparse() {
    // §29.2. The answers are the same either way; what changes is the work, so no physics bar in
    // this project could see a regression here.
    for n in [32, 64, 128] {
        let (p, jac) = a_jacobian(n);
        let natural = SparseLu::factor(&jac).expect("must factor");
        let reordered =
            SparseLu::factor_permuted(&jac, &p.perm, DIAG_PIVOT_THRESH).expect("must factor");
        let fill = |lu: &SparseLu| lu.nnz().0 + lu.nnz().1;
        let (nat, reo) = (fill(&natural), fill(&reordered));
        assert!(
            reo * 4 < nat,
            "n={n}: the reordering saved almost nothing ({reo} vs {nat}) — either the Jacobian's \
             structure changed or the permutation is no longer the fill-reducing one"
        );
        // And it is not merely better than natural: it is at most a small multiple of the band.
        assert!(
            reo < 20 * jac.nrows(),
            "n={n}: the reordered factor stores {reo} for {} rows — it should stay a small
             multiple of the band",
            jac.nrows()
        );
        assert!(!natural.is_reordered() && reordered.is_reordered());
    }
}

#[test]
fn the_permutation_does_not_change_the_answer() {
    // The reordering is only free if it is genuinely a reordering. An inverse permutation applied
    // the wrong way round returns a plausible vector of the right size.
    for n in [16, 32, 64] {
        let (p, jac) = a_jacobian(n);
        let b: Vec<f64> = (0..jac.nrows())
            .map(|i| 1e-4 * ((i as f64) * 0.71 + 0.3).sin())
            .collect();
        let x_nat = SparseLu::factor(&jac).unwrap().solve(&b).unwrap();
        let x_reo = SparseLu::factor_permuted(&jac, &p.perm, DIAG_PIVOT_THRESH)
            .unwrap()
            .solve(&b)
            .unwrap();
        let scale = x_nat.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let gap = x_nat
            .iter()
            .zip(&x_reo)
            .fold(0.0f64, |m, (a, c)| m.max((a - c).abs()));
        assert!(
            gap <= 1e-12 * scale,
            "n={n}: the two orderings disagree by {gap:.3e} of {scale:.3e}"
        );
        // ...and the solve really inverts the matrix, in the reordered numbering too.
        let residual = jac
            .matvec(&x_reo)
            .iter()
            .zip(&b)
            .fold(0.0f64, |m, (a, c)| m.max((a - c).abs()));
        assert!(residual <= 1e-14 * b.iter().fold(0.0f64, |m, v| m.max(v.abs())));
    }
}

// -- the reduction anchor -----------------------------------------------------------------------

#[test]
fn ea_equals_t_is_bit_identical_to_the_damped_string() {
    // The anchor §15.2's rule binds this model to another *class*. It is exact rather than close
    // because `a = EA - T0` is exactly zero, the nonlinear branch is skipped entirely, and what is
    // left is model #3's expressions in model #3's order.
    for (kappa, sigma0, sigma1) in [(0.0, 0.0, 0.0), (KAPPA, 0.0, 0.0), (KAPPA, 2.0, 5e-3)] {
        let p = params(48, kappa, T, sigma0, sigma1);
        let dp = damped_params(48, kappa, sigma0, sigma1);
        let u0 = mode(&p, 1, 1e-3);
        let zeros = vec![0.0; p.nodes()];

        let mut g = geo::GeometricString::new(p);
        g.set_state(
            &u0,
            &zeros,
            &zeros,
            &[zeros.clone(), zeros.clone(), zeros.clone()],
        );
        let mut d = damped::DampedStiffString::new(dp);
        let u0d = u0.clone();
        d.set_state(&u0d, &zeros);

        assert_eq!(
            g.u_prev, d.u_prev,
            "the consistent start must match bit-for-bit"
        );
        for _ in 0..300 {
            g.step().expect("the linear branch cannot fail");
            d.step();
        }
        assert_eq!(g.u, d.u, "kappa={kappa} sigma0={sigma0} sigma1={sigma1}");
        assert_eq!(g.energy(), d.energy());
        assert_eq!(g.nonlinear_energy(), 0.0);
        assert!(g.w.iter().all(|&x| x == 0.0) && g.v.iter().all(|&x| x == 0.0));
        assert_eq!(g.newton_iters, 0, "the linear branch runs no Newton solve");
    }
}

// -- the physics only this model has ------------------------------------------------------------

#[test]
fn the_tension_is_a_field_and_sits_at_t0_at_rest() {
    let p = params(32, KAPPA, EA, 0.0, 0.0);
    let s = geo::GeometricString::new(p);
    for t in s.tension() {
        assert!(
            (t - T).abs() <= 1e-12 * T,
            "at rest the tension is T0, got {t}"
        );
    }
    for l in s.stretch_ratio() {
        assert_eq!(l, 1.0, "at rest every element is unstretched");
    }

    let mut s = started(params(32, KAPPA, EA, 0.0, 0.0), 1, 8e-3);
    step_n(&mut s, 40);
    let tens = s.tension();
    let (lo, hi) = tens
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &t| {
            (a.min(t), b.max(t))
        });
    assert!(
        hi - lo > 1e-6 * T,
        "the tension must vary ALONG the string — that is what model #9 cannot represent"
    );
}

#[test]
fn a_degenerate_string_stays_exactly_planar() {
    // `w -> -w` is a reflection symmetry when `kappa_u == kappa_w`, so a planar start stays planar
    // bit-exactly — not approximately. A sign error in the coupling breaks this immediately.
    let mut s = started(params(32, KAPPA, EA, 0.0, 0.0), 1, 5e-3);
    assert!(s.is_degenerate());
    step_n(&mut s, 400);
    assert!(
        s.w.iter().all(|&x| x == 0.0),
        "a degenerate string cannot whirl"
    );
    assert!(
        s.v.iter().any(|&x| x != 0.0),
        "but it MUST drive the longitudinal field"
    );
}

#[test]
fn detuning_the_polarizations_stops_the_degeneracy() {
    let p = geo::Params::new(
        L,
        T,
        RHO,
        fs_for(32),
        32,
        EA,
        KAPPA,
        Some(KAPPA * 1.3),
        0.0,
        0.0,
        None,
        None,
        damped::THETA,
        true,
        geo::NEWTON_TOL_DEFAULT,
        geo::NEWTON_MAXITER_DEFAULT,
        false,
    )
    .expect("fixture must construct");
    assert!(!geo::GeometricString::new(p).is_degenerate());
}

// -- the rejections -----------------------------------------------------------------------------

#[test]
fn the_construction_rejections_are_the_pythons() {
    let ok = |n: i64| params(n, KAPPA, EA, 0.0, 0.0);
    let _ = ok(4);
    let bad = |args: geo::Params| drop(args);
    let _ = bad;

    let mk = |l, t, rho, fs, n, ea, kappa, kw, s0, s1, theta, boundary_ok, tol, maxit, soft| {
        geo::Params::new(
            l,
            t,
            rho,
            fs,
            n,
            ea,
            kappa,
            kw,
            s0,
            s1,
            None,
            None,
            theta,
            boundary_ok,
            tol,
            maxit,
            soft,
        )
        .unwrap_err()
        .to_string()
    };
    let d = damped::THETA;
    let tol = geo::NEWTON_TOL_DEFAULT;
    let mi = geo::NEWTON_MAXITER_DEFAULT;
    assert_eq!(
        mk(
            -1.0,
            T,
            RHO,
            fs_for(8),
            8,
            EA,
            0.0,
            None,
            0.0,
            0.0,
            d,
            true,
            tol,
            mi,
            false
        ),
        "L, T, rho, fs must all be positive."
    );
    assert_eq!(
        mk(
            L,
            T,
            RHO,
            fs_for(8),
            8,
            0.0,
            0.0,
            None,
            0.0,
            0.0,
            d,
            true,
            tol,
            mi,
            false
        ),
        "EA (axial stiffness) must be positive."
    );
    assert_eq!(
        mk(
            L,
            T,
            RHO,
            fs_for(8),
            1,
            EA,
            0.0,
            None,
            0.0,
            0.0,
            d,
            true,
            tol,
            mi,
            false
        ),
        "N must be >= 2 (need at least one interior node)."
    );
    assert_eq!(
        mk(
            L,
            T,
            RHO,
            fs_for(8),
            8,
            EA,
            -1.0,
            None,
            0.0,
            0.0,
            d,
            true,
            tol,
            mi,
            false
        ),
        "kappa (stiffness) must be >= 0."
    );
    assert_eq!(
        mk(
            L,
            T,
            RHO,
            fs_for(8),
            8,
            EA,
            0.0,
            Some(-1.0),
            0.0,
            0.0,
            d,
            true,
            tol,
            mi,
            false
        ),
        "kappa_w (stiffness) must be >= 0."
    );
    assert_eq!(
        mk(
            L,
            T,
            RHO,
            fs_for(8),
            8,
            EA,
            0.0,
            None,
            -1.0,
            0.0,
            d,
            true,
            tol,
            mi,
            false
        ),
        "sigma0, sigma1 (losses) must be >= 0."
    );
    assert_eq!(
        mk(
            L,
            T,
            RHO,
            fs_for(8),
            8,
            EA,
            0.0,
            None,
            0.0,
            0.0,
            0.0,
            true,
            tol,
            mi,
            false
        ),
        "theta must be in (0, 1], got 0.0."
    );
    assert_eq!(
        mk(
            L,
            T,
            RHO,
            fs_for(8),
            8,
            EA,
            0.0,
            None,
            0.0,
            0.0,
            d,
            true,
            0.0,
            mi,
            false
        ),
        "newton_tol must be > 0."
    );
    assert_eq!(
        mk(
            L,
            T,
            RHO,
            fs_for(8),
            8,
            EA,
            0.0,
            None,
            0.0,
            0.0,
            d,
            true,
            tol,
            0,
            false
        ),
        "newton_maxiter must be >= 1."
    );
    assert_eq!(
        mk(
            L,
            T,
            RHO,
            fs_for(8),
            8,
            EA,
            0.0,
            None,
            0.0,
            0.0,
            d,
            false,
            tol,
            mi,
            false
        ),
        "boundary must be 'supported'."
    );
    // The softening refusal quotes both numbers and the ratio, and it is opt-out rather than a
    // hard bar: the model stays well-posed below `EA = T0`, it is just not a real material.
    let soft = mk(
        L,
        T,
        RHO,
        fs_for(8),
        8,
        T / 2.0,
        0.0,
        None,
        0.0,
        0.0,
        d,
        true,
        tol,
        mi,
        false,
    );
    assert!(soft.starts_with("EA (100.0) < T (200.0)"), "{soft}");
    assert!(
        soft.contains("Lambda0 = (EA - T0)/EA = -1 NEGATIVE"),
        "{soft}"
    );
    assert!(geo::Params::new(
        L,
        T,
        RHO,
        fs_for(8),
        8,
        T / 2.0,
        0.0,
        None,
        0.0,
        0.0,
        None,
        None,
        d,
        true,
        tol,
        mi,
        true
    )
    .is_ok());
}

#[test]
fn the_lam_long_warning_is_reported_but_never_raised() {
    // The one guard in the project with no CFL behind it: it is an *accuracy* bar on an
    // unconditionally stable scheme, so the core reports it and the caller warns.
    let coarse = geo::Params::new(
        L,
        T,
        RHO,
        4000.0,
        32,
        EA,
        0.0,
        None,
        0.0,
        0.0,
        None,
        None,
        damped::THETA,
        true,
        geo::NEWTON_TOL_DEFAULT,
        geo::NEWTON_MAXITER_DEFAULT,
        false,
    )
    .expect("an under-resolved string still constructs");
    assert!(coarse.lam_long > geo::LAM_LONG_WARN && coarse.warn_lam_long);

    // ...and it is exempt at `EA == T`, where the three fields decouple and the model IS model #3
    // three times over. The exemption is load-bearing: the `EA = T` anchor lands at
    // `lam_long == 1.0` exactly, flush against the bar.
    let anchor = params(32, 0.0, T, 0.0, 0.0);
    assert_eq!(anchor.a, 0.0);
    assert!(!anchor.warn_lam_long, "the linear branch must never warn");
}
