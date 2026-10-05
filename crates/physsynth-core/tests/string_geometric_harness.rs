//! The geometrically exact string's validation harness, first half — model #10, carried from
//! `tests/test_geometric_energy.py`, `tests/test_geometric_polarization.py` and
//! `tests/test_geometric_limits.py` (retirement plan §35) and run at the retired Python helper
//! `make_geometric_string`'s parameters: `L = 1`, `T = 200`, `rho = 0.005` (so `c = 200 m/s`),
//! `kappa = 2`, `theta = 0.28`, `EA = 1e5` (`EA/T = 500`), `newton_tol = 1e-15`, and the sample
//! rate set from the **longitudinal** Courant number, `fs = sqrt(EA/rho) N / (L lam_long)` with
//! `lam_long = 0.5`, unless a bar passes the transverse `lam` instead.
//!
//! **A separate file from `string_geometric.rs`**, for §32's reason: that file runs its own
//! fixture (`L = 0.65`, `kappa = 1.5`) and holds the bars about the sparse LU's ordering, which
//! are about the port rather than the physics.
//!
//! What the three retired files asserted, in the order this file keeps them:
//!
//! - **The scheme** — the discrete-gradient identity the conservation is a corollary of, its
//!   negative control (`Λ(mean)` is *not* a discrete gradient), the summation-by-parts pair, the
//!   Newton Jacobian against finite differences, the `EA = T` anchor to model #3 bit for bit, the
//!   drift, the floor, passivity, the linear limit and the guards.
//! - **The two polarizations** — the nonlinearity sees `u` and `w` only through `r² = u_x² + w_x²`,
//!   so a planar start stays planar to the bit, a 90° rotation is bit-exact and any rotation is
//!   exact to round-off. Energy cannot see a defect here; these bars can.
//! - **The limits** — in `string_geometric_long.rs`, with every other long simulation the three
//!   files ran (the three linear waves, the softening sweep, the under-resolved band). That file is
//!   optimised-only in CI (the human's call, §35.7); this one runs in both profiles, because the
//!   exact structural bars here are the ones most worth an unfolded-arithmetic run.
//!
//! Two Python tests have **no analogue**:
//!
//! - `test_apply_Ainv_raises_for_a_reason_that_is_not_model_9s` asserted that a call raises; the
//!   native [`GeometricString`] has no `apply_ainv`, so the wrong call does not compile (§16's "an
//!   absence becomes a type").
//! - `test_set_state_velocity_arguments_are_keyword_only` guarded a Python name clash (`v0` is a
//!   displacement here and a velocity in models #1–#9); natively the velocities are a separate
//!   typed argument, `dots: &[Vec<f64>; 3]`, so model #3's `set_state(u0, v0)` does not compile.
//!
//! A third, `test_non_convergence_warns_and_is_counted`, is carried as its **condition** only: the
//! two warnings' prose lives in the binding and goes with it, and what the core reports —
//! `Params::warn_lam_long` and the Newton report's `converged` — is asserted below in every
//! configuration the Python built.

mod geometric_fixture;

use geometric_fixture::*;
use physsynth_core::ops::second_difference_matrix;
use physsynth_core::string_damped as damped;
use physsynth_core::string_geometric::{self as geo, GeometricString, ParamError, Params};
use std::f64::consts::PI;

/// `<gradbar V, q+ - q->` against `V(q+) - V(q-)`, as a relative error.
fn dg_identity_error(p: &Params, force: &[f64], q_plus: &[f64], q_minus: &[f64]) -> f64 {
    let lhs = p.h
        * (0..q_plus.len())
            .map(|i| force[i] * (q_plus[i] - q_minus[i]))
            .sum::<f64>();
    let rhs = geo::nl_density(q_plus, p.a, p.h) - geo::nl_density(q_minus, p.a, p.h);
    (lhs - rhs).abs() / rhs.abs()
}

// == the discrete gradient: the identity the whole scheme rests on ================================

#[test]
fn the_discrete_gradient_telescopes_at_five_decades_of_strain() {
    // Carried from `test_discrete_gradient_telescopes_exactly`. Energy conservation is a corollary
    // of this identity plus SBP, so it is tested directly, across the strains where the naive
    // midpoint below is most and least wrong.
    let p = Geo::new(32).params().unwrap();
    let mut reached_other_branch = false;
    for (k, scale) in [0.5, 0.1, 1e-2, 1e-3, 1e-4].into_iter().enumerate() {
        let q_plus = draw(11, 6 * k * p.n, p.n, scale);
        let q_minus = draw(11, (6 * k + 3) * p.n, p.n, scale);
        for q in [&q_plus, &q_minus] {
            reached_other_branch |= geo::stretch_terms(q).denom.iter().any(|&d| d <= 1.0);
        }
        let f = geo::dg_force(&q_plus, &q_minus, p.a);
        let err = dg_identity_error(&p, &f, &q_plus, &q_minus);
        println!("strain {scale:e}: DG identity error {err:.3e}");
        assert!(
            err <= 1e-12,
            "DG identity failed at strain {scale}: {err:.3e}"
        );
    }
    // The draw is only worth its breadth if it reaches the cancellation-free branch too.
    assert!(
        reached_other_branch,
        "no cell took the `denom <= 1` branch of the stretch terms"
    );
}

#[test]
fn the_naive_midpoint_misses_the_identity_by_percent() {
    // Carried from `test_naive_midpoint_gradient_is_not_a_discrete_gradient`: `Λ(mean)` — the
    // tempting midpoint — has the right limit, glide and spectrum, and fails only the energy gate.
    let p = Geo::new(32).params().unwrap();
    let n = p.n;
    let mut worst = 0.0f64;
    for (k, scale) in [0.5, 0.1, 1e-2, 1e-3].into_iter().enumerate() {
        let q_plus = draw(11, 6 * k * n, n, scale);
        let q_minus = draw(11, (6 * k + 3) * n, n, scale);
        let q_bar: Vec<f64> = q_plus
            .iter()
            .zip(&q_minus)
            .map(|(a, b)| 0.5 * (a + b))
            .collect();
        let lam = geo::stretch_ratio(&q_bar);
        let mut naive = vec![0.0; 3 * n];
        for i in 0..n {
            let chi = 1.0 - 1.0 / lam[i]; // Λ(mean): the WRONG one
            naive[i] = p.a * chi * q_bar[i];
            naive[n + i] = p.a * chi * q_bar[n + i];
            naive[2 * n + i] = p.a * (chi * (1.0 + q_bar[2 * n + i]) - q_bar[2 * n + i]);
        }
        let err = dg_identity_error(&p, &naive, &q_plus, &q_minus);
        println!("strain {scale:e}: naive midpoint error {err:.3e}");
        worst = worst.max(err);
    }
    assert!(
        worst > 1e-2,
        "the naive midpoint should MISS by percent, worst {worst:.3e}"
    );
}

#[test]
fn at_equal_states_the_discrete_gradient_is_the_continuum_gradient() {
    // Carried from `test_dg_at_equal_states_is_the_continuum_gradient` — which is why `set_state`
    // can reuse it for the consistent start. `np.allclose(..., rtol=1e-11, atol=0.0)`: per entry.
    let p = Geo::new(32).params().unwrap();
    let n = p.n;
    let q = draw(3, 0, n, 0.05);
    let lam = geo::stretch_ratio(&q);
    let got = geo::dg_force(&q, &q, p.a);
    let mut worst = 0.0f64;
    for i in 0..n {
        let chi = 1.0 - 1.0 / lam[i];
        let expected = [
            p.a * chi * q[i],
            p.a * chi * q[n + i],
            p.a * (chi * (1.0 + q[2 * n + i]) - q[2 * n + i]),
        ];
        for (f, e) in expected.iter().enumerate() {
            let gap = (got[f * n + i] - e).abs();
            assert!(
                gap <= 1e-11 * e.abs(),
                "cell {i} field {f}: {} vs {e}",
                got[f * n + i]
            );
            worst = worst.max(gap / e.abs());
        }
    }
    println!("worst relative gap {worst:.3e}");
}

#[test]
fn the_sbp_pair_is_an_exact_adjoint_and_composes_to_d2() {
    // Carried from `test_sbp_adjoint_pair`: `δ_x− = −(δ_x+)ᵀ` exactly, and `δ_x− δ_x+ = D2`. The
    // pair is what turns the DG identity into an energy statement.
    let p = Geo::new(16).params().unwrap();
    let (gp, gm) = (&p.gp, &p.gm);
    assert_eq!((gm.nrows(), gm.ncols()), (gp.ncols(), gp.nrows()));
    for i in 0..gm.nrows() {
        for j in 0..gm.ncols() {
            assert!(gm.get(i, j) == -gp.get(j, i), "Gm[{i},{j}] != -Gp[{j},{i}]");
        }
    }
    let d2 = second_difference_matrix(p.n, p.h);
    let gmgp = gm.matmul(gp);
    let scale = (0..d2.nrows())
        .flat_map(|i| (0..d2.ncols()).map(move |j| (i, j)))
        .fold(0.0f64, |m, (i, j)| m.max(d2.get(i, j).abs()));
    let mut gap = 0.0f64;
    for i in 0..d2.nrows() {
        for j in 0..d2.ncols() {
            gap = gap.max((gmgp.get(i, j) - d2.get(i, j)).abs());
        }
    }
    println!("|Gm Gp - D2| = {gap:.3e} of {scale:.3e}");
    assert!(gap <= 1e-9 * scale);
}

#[test]
fn the_newton_jacobian_matches_finite_differences_and_is_not_symmetric() {
    // Carried from `test_dg_jacobian_matches_finite_differences_and_is_not_symmetric`. Asserting
    // the asymmetry (not merely tolerating it) pins why the solve is a sparse LU and not the banded
    // Cholesky every other string in the family uses.
    let p = Geo::new(12).params().unwrap();
    let n = p.n;
    let q_plus = draw(5, 0, n, 0.05);
    let q_minus = draw(5, 3 * n, n, 0.05);
    let jac = geo::dg_jacobian(&q_plus, &q_minus, p.a);
    let eps = 1e-6;
    let mut fd = vec![vec![0.0; 3 * n]; 3 * n];
    for i in 0..3 * n {
        let mut up = q_plus.clone();
        let mut dn = q_plus.clone();
        up[i] += eps;
        dn[i] -= eps;
        let (fp, fm) = (
            geo::dg_force(&up, &q_minus, p.a),
            geo::dg_force(&dn, &q_minus, p.a),
        );
        for r in 0..3 * n {
            fd[r][i] = (fp[r] - fm[r]) / (2.0 * eps);
        }
    }
    let (mut gap, mut fd_max, mut jac_max, mut asym) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for (r, fd_row) in fd.iter().enumerate() {
        for (col, &fd_rc) in fd_row.iter().enumerate() {
            gap = gap.max((jac.get(r, col) - fd_rc).abs());
            fd_max = fd_max.max(fd_rc.abs());
            jac_max = jac_max.max(jac.get(r, col).abs());
            asym = asym.max((jac.get(r, col) - jac.get(col, r)).abs());
        }
    }
    println!(
        "FD gap {:.3e}, asymmetry {:.3e} (relative)",
        gap / fd_max,
        asym / jac_max
    );
    assert!(
        gap < 1e-6 * fd_max,
        "the Jacobian is not the derivative of the DG force"
    );
    assert!(
        asym > 0.1 * jac_max,
        "expected a NON-symmetric discrete gradient"
    );
}

// == the regression anchor
// =========================================================================

#[test]
fn ea_equals_t_is_model_three_bit_for_bit_from_a_pluck() {
    // Carried from `test_EA_equals_T_is_bit_identical_to_damped_string` (three loss sets). `a =
    // EA - T0` is exactly zero, the nonlinear branch is skipped, and what is left is model #3's
    // expressions in model #3's order. A pluck, at the Python's `N = 64`, `fs = 12800` — the
    // pre-existing native anchor runs a single mode at another fixture.
    for (kappa, sigma0, sigma1) in [(0.0, 0.0, 0.0), (KAPPA, 0.0, 0.0), (KAPPA, 2.0, 5e-3)] {
        let cfg = Geo {
            fs: Some(12800.0),
            kappa,
            ea: T,
            sigma0,
            sigma1,
            ..Geo::new(64)
        };
        let mut g = cfg.build();
        let dp = damped::Params::new(L, T, RHO, 12800.0, 64, kappa, sigma0, sigma1, THETA, true)
            .expect("the damped twin must construct");
        let mut d = damped::DampedStiffString::new(dp);
        let ic = pluck_ic(&g, 1e-3);
        start_u(&mut g, &ic);
        d.set_state(&ic, &zeros(&g));
        assert_eq!(
            g.u_prev, d.u_prev,
            "the consistent start must match bit-for-bit"
        );

        for _ in 0..300 {
            step(&mut g);
            d.step();
        }
        let label = format!("kappa={kappa} sigma0={sigma0} sigma1={sigma1}");
        assert_eq!(g.u, d.u, "{label}");
        assert_eq!(g.energy(), d.energy(), "{label}");
        assert_eq!(g.nonlinear_energy(), 0.0);
        // ...and the two spare fields never woke up.
        assert_eq!(max_abs(&g.w), 0.0);
        assert_eq!(max_abs(&g.v), 0.0);
    }
}

#[test]
fn at_ea_equals_t_the_three_fields_are_three_decoupled_linear_strings() {
    // Carried from `test_EA_equals_T_makes_all_three_fields_linear_and_decoupled`.
    let mut s = Geo {
        fs: Some(12800.0),
        kappa: 0.0,
        ea: T,
        ..Geo::new(64)
    }
    .build();
    let (u0, w0, v0) = (
        mode_ic(&s, 1, 1e-3),
        mode_ic(&s, 2, 1e-3),
        mode_ic(&s, 3, 1e-3),
    );
    start(&mut s, &u0, &w0, &v0);
    let e0 = s.energy();
    step_n(&mut s, 400);
    let drift = rel_drift(s.energy(), e0);
    println!("drift {drift:.3e}");
    assert!(drift < 1e-12);
    assert_eq!(s.nonlinear_energy(), 0.0);
}

// == energy: drift, the floor, passivity
// ===========================================================

#[test]
fn a_single_mode_conserves_energy_with_the_nonlinearity_engaged() {
    // Carried from `test_lossless_drift_single_mode_with_nonlinear_fraction_reported`. The second
    // half is the point: a nonlinearity bug hides at small amplitude, where the test silently
    // re-runs the linear scheme and passes.
    let mut s = Geo::new(64).build();
    let u0 = mode_ic(&s, 1, 4e-3);
    start_u(&mut s, &u0);
    let e0 = s.energy();
    let (mut peak_nl, mut peak_long, mut worst) = (0.0f64, 0.0f64, 0.0f64);
    for _ in 0..600 {
        step(&mut s);
        peak_nl = peak_nl.max(s.nonlinear_energy().abs() / e0.abs());
        peak_long = peak_long.max(s.longitudinal_energy() / e0.abs());
        worst = worst.max(rel_drift(s.energy(), e0));
    }
    let drift = rel_drift(s.energy(), e0);
    println!("drift {drift:.3e} (worst {worst:.3e}), nl {peak_nl:.3e}, long {peak_long:.3e}");
    assert_eq!(s.n_not_converged, 0);
    assert!(drift < DRIFT_GATE, "drift {drift:.3e}");
    assert!(
        peak_nl > 1e-3,
        "nonlinearity barely engaged ({peak_nl:.2e}) — secretly linear"
    );
    assert!(
        peak_long > 1e-3,
        "longitudinal field barely moved ({peak_long:.2e})"
    );
}

#[test]
fn a_broadband_pluck_conserves_energy_through_the_local_tension_field() {
    // Carried from `test_lossless_drift_from_a_plucked_broadband_ic`: a single mode keeps the
    // strain nearly uniform and never exercises the local tension that is the whole difference
    // between model #10 and model #9. If only one energy test could exist, it would be this one.
    let mut s = Geo::new(64).build();
    let u0 = pluck_ic(&s, 3e-3);
    start_u(&mut s, &u0);
    let e0 = s.energy();
    let mut worst = 0.0f64;
    for _ in 0..800 {
        step(&mut s);
        worst = worst.max(rel_drift(s.energy(), e0));
    }
    let drift = rel_drift(s.energy(), e0);
    let spread = ptp(&s.stretch_ratio());
    println!("drift {drift:.3e} (worst {worst:.3e}), stretch spread {spread:.3e}");
    assert_eq!(s.n_not_converged, 0);
    assert!(drift < DRIFT_GATE, "drift {drift:.3e}");
    assert!(
        spread > 1e-6,
        "stretch field is uniform ({spread:.2e}) — KC would have sufficed"
    );
}

#[test]
fn a_broadband_pluck_conserves_energy_at_every_theta() {
    // NOT carried from Python: planted in §35.4. Every Python bar here built at the default
    // `theta = 0.28`, so `theta` hard-coded at 0.28 in the step's right-hand side, in the update
    // matrix or in the energy passed the whole workspace — §33's and §34's finding a third time,
    // and their guard: the general-case start (a pluck) at both ends of the stable range.
    for theta in [0.25, 0.28, 0.5] {
        let mut s = Geo {
            theta,
            ..Geo::new(64)
        }
        .build();
        let u0 = pluck_ic(&s, 3e-3);
        start_u(&mut s, &u0);
        let e0 = s.energy();
        let mut worst = 0.0f64;
        for _ in 0..800 {
            step(&mut s);
            let d = rel_drift(s.energy(), e0);
            worst = if d.is_nan() { f64::NAN } else { worst.max(d) };
        }
        println!("theta {theta}: worst drift {worst:.3e}");
        assert_eq!(s.n_not_converged, 0, "theta {theta}");
        assert!(worst < DRIFT_GATE, "theta {theta}: drift {worst:.3e}");
    }
}

#[test]
fn the_drift_follows_the_newton_tolerance() {
    // Carried from `test_drift_is_controlled_by_the_newton_tolerance` — the self-certification
    // that the drift belongs to the solve and not to the scheme. Not proportional: Newton is
    // quadratic, so the drift is a step function of the iteration count. Assert the control.
    let drift_at = |tol: f64| {
        let mut s = Geo {
            newton_tol: tol,
            ..Geo::new(32)
        }
        .build();
        let u0 = mode_ic(&s, 1, 8e-3);
        start_u(&mut s, &u0);
        let e0 = s.energy();
        step_n(&mut s, 400);
        (rel_drift(s.energy(), e0), s.total_newton_iters)
    };
    let (loose, loose_iters) = drift_at(1e-4);
    let (tight, tight_iters) = drift_at(1e-15);
    println!("tol 1e-4: {loose:.3e} ({loose_iters} iters); 1e-15: {tight:.3e} ({tight_iters})");
    assert!(
        loose > 1e6 * tight,
        "loosening the solve must loosen the drift"
    );
    assert!(tight < DRIFT_GATE);
}

#[test]
fn the_energy_floor_is_zero_and_a_string_at_rest_sits_on_it() {
    // Carried from `test_energy_floor_is_zero_and_respected`. The floor is written here as `0.0`
    // (the binding's `energy_floor` is that literal; §29.3), and it is tighter than the free
    // string's `−L T0²/(2 EA)`, which clamped ends cannot reach (next test).
    let floor = 0.0;
    let loose = -L * T * T / (2.0 * EA);
    assert!(
        floor > loose,
        "the free-string floor is the loose one; clamped is tighter"
    );

    let mut s = Geo::new(32).build();
    let u0 = pluck_ic(&s, 3e-3);
    start_u(&mut s, &u0);
    let mut lowest = f64::INFINITY;
    for _ in 0..400 {
        step(&mut s);
        let e = s.energy();
        assert!(e >= floor, "energy {e:.3e} below the floor");
        lowest = lowest.min(e);
    }
    println!("lowest energy {lowest:.3e}");
    // ...and the floor has teeth: a string at rest sits *at* it, not above it.
    let mut at_rest = Geo::new(32).build();
    let z = zeros(&at_rest);
    start_u(&mut at_rest, &z);
    assert_eq!(at_rest.energy(), 0.0);
}

#[test]
fn the_relaxed_state_is_unreachable_with_clamped_ends() {
    // Carried from `test_the_relaxed_state_is_inadmissible_with_clamped_ends`: `−L T0²/(2 EA)` is
    // the energy of a string shrunk to its natural length everywhere, which needs `v(L) − v(0) =
    // −L T0/EA ≠ 0`. The tension is read off the model's own field here, not recomputed.
    let p = Geo::new(32).params().unwrap();
    let relaxed = (EA - T) / EA; // Λ with zero tension
    let mut q = vec![0.0; 3 * p.n];
    for v in q[2 * p.n..].iter_mut() {
        *v = relaxed - 1.0;
    }
    for t in geo::tension(&q, p.ea, p.a) {
        assert!(
            t.abs() <= 1e-9,
            "the tension must vanish at the relaxed state, got {t:.3e}"
        );
    }
    let shortening: f64 = q[2 * p.n..].iter().sum::<f64>() * p.h;
    let expected = -L * T / EA;
    assert!(
        (shortening - expected).abs() <= 1e-12 * expected.abs(),
        "{shortening:e}"
    );
    let excess = geo::nl_density(&q, p.a, p.h);
    assert!(
        excess.abs() <= 1e-12,
        "the excess vanishes there, got {excess:.3e}"
    );
}

#[test]
fn losses_make_the_energy_monotone_with_all_six_terms() {
    // Carried from `test_passivity_is_monotone_with_all_six_losses`. None of the six loss terms
    // enters `E`; each is dissipative by SBP, so passivity is automatic and cheap to check.
    let mut s = Geo {
        sigma0: 2.0,
        sigma1: 5e-3,
        sigma0_long: Some(1.0),
        sigma1_long: Some(2e-3),
        ..Geo::new(64)
    }
    .build();
    let u0 = pluck_ic(&s, 3e-3);
    start_u(&mut s, &u0);
    let e0 = s.energy();
    let (mut last, mut worst) = (e0, f64::NEG_INFINITY);
    for _ in 0..600 {
        step(&mut s);
        let e = s.energy();
        assert!(e - last <= 0.0, "energy increased by {:.3e}", e - last);
        worst = worst.max(e - last);
        last = e;
    }
    println!("worst one-step change {worst:.3e}; E/E0 = {:.6}", last / e0);
    assert!(last < e0);
}

#[test]
fn the_longitudinal_losses_inherit_the_transverse_ones_unless_set() {
    // Carried from `test_longitudinal_loss_defaults_to_the_transverse_values`: the constructor
    // makes no silent physics claim. Real strings damp longitudinal motion less, but that is a
    // setting to opt into.
    let inherited = Geo {
        sigma0: 2.0,
        sigma1: 5e-3,
        ..Geo::new(64)
    }
    .params()
    .unwrap();
    assert_eq!((inherited.sigma0_long, inherited.sigma1_long), (2.0, 5e-3));
    let opted_in = Geo {
        sigma0: 2.0,
        sigma1: 5e-3,
        sigma0_long: Some(0.0),
        sigma1_long: Some(0.0),
        ..Geo::new(64)
    }
    .params()
    .unwrap();
    assert_eq!((opted_in.sigma0_long, opted_in.sigma1_long), (0.0, 0.0));
}

#[test]
fn each_field_takes_its_own_losses() {
    // NOT carried from Python: planted in §35.4, the human's call. Routing the transverse losses
    // to the longitudinal field (or the reverse) passed the whole workspace: passivity holds
    // either way, and the inheritance test reads only the parameters. At `EA = T` the three fields
    // are decoupled linear strings, so a field whose own losses are zero must conserve its own
    // energy exactly while the damped one decays — in both directions.
    for (sideways, lengthways) in [((2.0, 5e-3), (0.0, 0.0)), ((0.0, 0.0), (2.0, 5e-3))] {
        let mut s = Geo {
            fs: Some(12800.0),
            ea: T,
            sigma0: sideways.0,
            sigma1: sideways.1,
            sigma0_long: Some(lengthways.0),
            sigma1_long: Some(lengthways.1),
            ..Geo::new(64)
        }
        .build();
        let (z, u0, v0) = (zeros(&s), mode_ic(&s, 1, 1e-3), mode_ic(&s, 3, 1e-3));
        start(&mut s, &u0, &z, &v0);
        let long0 = s.longitudinal_energy();
        let side0 = s.energy() - long0;
        step_n(&mut s, 400);
        let long = s.longitudinal_energy();
        let side = s.energy() - long;
        let (dl, ds) = (rel_drift(long, long0), rel_drift(side, side0));
        println!("sideways losses {sideways:?}: lengthways change {dl:.3e}, sideways {ds:.3e}");
        let (lossless, lossy) = if sideways.0 > 0.0 { (dl, ds) } else { (ds, dl) };
        assert!(
            lossless < 1e-12,
            "the undamped field lost {lossless:.3e} of its energy"
        );
        assert!(
            lossy > 1e-3,
            "the damped field kept its energy ({lossy:.3e})"
        );
    }
}

// == the linear limit
// ==============================================================================

#[test]
fn both_polarizations_share_one_wave_speed() {
    // Carried from `test_both_polarizations_share_one_wave_speed`: the tension is isotropic by
    // construction, and only `kappa` can distinguish the polarizations.
    let p = Geo {
        kappa: 1.0,
        kappa_w: Some(3.0),
        ..Geo::new(64)
    }
    .params()
    .unwrap();
    assert_eq!(p.c, c());
    assert_eq!((p.kappa_u, p.kappa_w), (1.0, 3.0));
    assert!(!GeometricString::new(p).is_degenerate());
    assert!(Geo {
        kappa: 1.0,
        ..Geo::new(64)
    }
    .build()
    .is_degenerate());
}

// == the start-up
// ==================================================================================

#[test]
fn released_from_rest_the_first_step_is_time_symmetric_in_every_field() {
    // NOT carried from Python: planted in §35.4. Dropping the 1/2 in the Taylor start was seen
    // only by the two `EA = T` twin anchors, and dropping the start's NONLINEAR force was seen by
    // nothing in the workspace. Released from rest, the motion is even in time; the scheme is
    // time-reversible (the discrete gradient is symmetric in its two levels), so a start on the
    // even solution makes `f^1 = f^{-1}` up to the Taylor start's own O(k^4) error. Measured
    // against the step's size: the transverse field (the 1/2) and the longitudinal one, whose ONLY
    // acceleration at t = 0 is the nonlinear force, so leaving that force out of the start puts
    // the whole of `v^1` into the asymmetry.
    //
    // The 1e-2 bound is THIS fixture's (mode 1 at `N = 64`, `lam_long = 0.5`, where `k omega` is
    // small): the Taylor start's own error grows like `(k omega)^2`, and a correct start on mode 3
    // at `N = 32` already reads 4.7e-2. Do not reuse the bound elsewhere. Planted, the dropped 1/2
    // read 2.4e4 (u) and 1.5e3 (v); the dropped nonlinear force leaves `v^{-1} = 0`, which the
    // first assertion below refuses (its asymmetry would be exactly 1).
    let mut s = Geo::new(64).build();
    let u0 = mode_ic(&s, 1, 4e-3);
    start_u(&mut s, &u0);
    let (u_back, v_back) = (s.u_prev.clone(), s.v_prev.clone());
    assert!(
        max_abs(&v_back) > 0.0,
        "the start must give v an acceleration"
    );
    step(&mut s);
    let gap = |a: &[f64], b: &[f64]| {
        a.iter()
            .zip(b)
            .fold(0.0f64, |m, (x, y)| m.max((x - y).abs()))
    };
    let u_asym = gap(&s.u, &u_back) / gap(&s.u, &u0);
    let v_asym = gap(&s.v, &v_back) / max_abs(&s.v);
    println!("asymmetry: u {u_asym:.3e}, v {v_asym:.3e}");
    assert!(
        u_asym < 1e-2,
        "the transverse start is not the even solution: {u_asym:.3e}"
    );
    assert!(
        v_asym < 1e-2,
        "the longitudinal start is not the even solution: {v_asym:.3e}"
    );
}

// == guards
// ========================================================================================

#[test]
fn a_softening_string_is_refused_by_default_and_built_on_request() {
    // Carried from `test_softening_EA_is_rejected_by_default_and_permitted_on_request`: refused for
    // a MATERIALS reason — `Λ0 = (EA − T0)/EA < 0` is an unstretched length below zero — and not a
    // stability one (next test).
    assert_eq!(
        class_default(T * 0.5).params().unwrap_err(),
        ParamError::Softening(T * 0.5, T)
    );
    let soft = Geo {
        allow_softening: true,
        ..class_default(T * 0.5)
    }
    .params()
    .unwrap();
    assert!(soft.allow_softening);
    assert!(soft.a < 0.0);
    assert!(
        (soft.ea - soft.t) / soft.ea < 0.0,
        "the natural length is negative"
    );
    // EA = T is the anchor and must stay constructible.
    assert_eq!(class_default(T).params().unwrap().a, 0.0);
}

#[test]
fn the_pythons_twelve_refusals_are_the_right_variants() {
    // Carried from `test_construction_rejects_nonphysical_parameters` (twelve cases), at the
    // class defaults the Python built. Each asserts the VARIANT and the fragment the Python
    // matched: the regexes overlapped ("EA" also matches the softening message, "kappa" the
    // `kappa_w` one), so the variant is the sharper claim (§32.4 D).
    let base = || Geo {
        ea: 1e5,
        ..class_default(1e5)
    };
    let with_t = |t: f64| {
        let g = base();
        Params::new(
            L,
            t,
            RHO,
            g.fs(),
            g.n,
            g.ea,
            g.kappa,
            None,
            0.0,
            0.0,
            None,
            None,
            THETA,
            true,
            NEWTON_TOL,
            NEWTON_MAXITER,
            false,
        )
    };
    let with_boundary = |ok: bool| {
        let g = base();
        Params::new(
            L,
            T,
            RHO,
            g.fs(),
            g.n,
            g.ea,
            g.kappa,
            None,
            0.0,
            0.0,
            None,
            None,
            THETA,
            ok,
            NEWTON_TOL,
            NEWTON_MAXITER,
            false,
        )
    };
    let cases: Vec<(Result<Params, ParamError>, ParamError, &str)> = vec![
        (
            Geo { ea: 0.0, ..base() }.params(),
            ParamError::NonPositiveEa,
            "EA",
        ),
        (
            Geo { ea: -1.0, ..base() }.params(),
            ParamError::NonPositiveEa,
            "EA",
        ),
        (
            Geo { n: 1, ..base() }.params(),
            ParamError::TooFewSegments,
            "N must be",
        ),
        (
            Geo {
                kappa: -1.0,
                ..base()
            }
            .params(),
            ParamError::NegativeKappa,
            "kappa",
        ),
        (
            Geo {
                kappa_w: Some(-1.0),
                ..base()
            }
            .params(),
            ParamError::NegativeKappaW,
            "kappa_w",
        ),
        (
            Geo {
                sigma0: -1.0,
                ..base()
            }
            .params(),
            ParamError::NegativeSigma,
            "sigma0",
        ),
        (
            Geo {
                sigma0_long: Some(-1.0),
                ..base()
            }
            .params(),
            ParamError::NegativeSigmaLong,
            "sigma0_long",
        ),
        (
            Geo {
                theta: 0.0,
                ..base()
            }
            .params(),
            ParamError::BadTheta(0.0),
            "theta",
        ),
        (
            Geo {
                theta: 1.5,
                ..base()
            }
            .params(),
            ParamError::BadTheta(1.5),
            "theta",
        ),
        (
            Geo {
                newton_tol: 0.0,
                ..base()
            }
            .params(),
            ParamError::BadNewtonTol,
            "newton_tol",
        ),
        (with_boundary(false), ParamError::BadBoundary, "boundary"),
        (
            with_t(-1.0),
            ParamError::NonPositive,
            "must all be positive",
        ),
    ];
    for (got, want, fragment) in cases {
        let err = got.expect_err("must refuse");
        assert_eq!(err, want);
        assert!(
            err.to_string().contains(fragment),
            "{err} lacks {fragment:?}"
        );
    }
}

#[test]
fn a_stalled_newton_solve_is_reported_and_counted() {
    // Carried from `test_non_convergence_warns_and_is_counted`. The warning is the binding's; what
    // the core owes it is the condition — a report with `converged == false` on the stalled step,
    // and the counter. Driven far past any sane amplitude at `lam_long ~ 45` with three iterations.
    let cfg = Geo {
        lam: Some(2.0),
        newton_maxiter: 3,
        ..Geo::new(32)
    };
    let mut s = cfg.build();
    assert!(
        s.p.warn_lam_long,
        "lam = 2 is lam_long ~ 45: the construction is flagged"
    );
    let u0 = mode_ic(&s, 1, 0.35);
    start_u(&mut s, &u0);
    let mut stalled_reports = 0;
    for _ in 0..30 {
        let report = s
            .step()
            .expect("the Jacobian must factor")
            .expect("a nonlinear step");
        if !report.converged {
            stalled_reports += 1;
            assert!(
                report.residual > report.tol_abs,
                "a stall is a residual above its bar"
            );
        }
    }
    println!("{} of 30 steps stalled", s.n_not_converged);
    assert!(s.n_not_converged > 0);
    assert_eq!(stalled_reports, s.n_not_converged);
    assert!(!s.converged);
}

#[test]
fn the_reported_courant_numbers_expose_the_longitudinal_tax() {
    // Carried from `test_reported_courant_numbers_expose_the_longitudinal_tax`: `lam_long =
    // sqrt(EA/T) lam`, ~22x `lam` at a real `EA/T`. Stable is not accurate.
    let p = Geo {
        lam: Some(1.0),
        ..Geo::new(64)
    }
    .params()
    .unwrap();
    assert!(p.warn_lam_long);
    assert!((p.lam - 1.0).abs() <= 1e-6);
    let ratio = (EA / T).sqrt();
    assert!((p.lam_long - ratio).abs() <= 1e-12 * ratio);
    assert!(p.lam_long > 20.0);
    let c_long = (EA / RHO).sqrt();
    assert!((p.c_long - c_long).abs() <= 1e-6 * c_long);
    assert!((p.ea_over_t - EA / T).abs() <= 1e-6 * (EA / T));
}

#[test]
fn an_under_resolved_longitudinal_field_is_flagged_because_nothing_else_will() {
    // Carried from `test_under_resolved_longitudinal_field_warns_because_nothing_else_will`. The
    // transverse `lam = 0.5` a reader of models #1-#9 reaches for first is exactly the trap. The
    // bar's value is written here (§29.3), and the model's constant checked against it.
    let bar = 1.0;
    assert_eq!(geo::LAM_LONG_WARN, bar);
    let bad = Geo {
        lam: Some(0.5),
        ..Geo::new(32)
    }
    .params()
    .unwrap();
    assert!(bad.warn_lam_long);
    assert!(
        bad.lam_long > 4.0,
        "lam = 0.5 should land deep in the failing regime"
    );
    let good = Geo {
        lam_long: Some(0.5),
        ..Geo::new(32)
    }
    .params()
    .unwrap();
    assert!(!good.warn_lam_long, "a resolved build is silent");
    assert!((good.lam_long - 0.5).abs() <= 1e-12 * 0.5);
    assert!(good.lam_long <= bar);
}

#[test]
fn the_linear_anchor_is_never_flagged_however_coarse() {
    // Carried from `test_the_linear_anchor_never_warns_however_coarse_the_longitudinal_field`. At
    // `a == 0` the fields decouple and there is no Newton solve to stall. The anchor lands at
    // `lam_long == 1.0` exactly, flush against the bar, so the exemption is load-bearing.
    let anchor = Geo {
        lam: Some(1.0),
        ea: T,
        ..Geo::new(32)
    }
    .params()
    .unwrap();
    let deep = Geo {
        lam: Some(50.0),
        ea: T,
        ..Geo::new(32)
    }
    .params()
    .unwrap();
    assert!(!anchor.warn_lam_long && !deep.warn_lam_long);
    assert_eq!((anchor.a, deep.a), (0.0, 0.0));
    assert!((anchor.lam_long - 1.0).abs() <= 1e-12);
    assert!(
        deep.lam_long > 4.0,
        "well past the bar, and still silent because it is linear"
    );
    // The exemption is on `a`, not on `lam_long`: the same coarse build is flagged once EA != T.
    let near = Geo {
        lam: Some(50.0),
        ea: T * 1.000001,
        ..Geo::new(32)
    }
    .params()
    .unwrap();
    assert!(near.warn_lam_long);
}

#[test]
fn the_tension_is_a_field_not_a_scalar() {
    // Carried from `test_tension_is_a_field_not_a_scalar`: `T(x) = EA Λ(x) − (EA − T0)`, exactly
    // `T0` at rest. The Python's `np.allclose(..., rtol=1e-12)` kept `atol = 1e-8`; at rest the
    // value is exact, so it is asserted exact.
    let mut s = Geo::new(64).build();
    assert!(
        s.tension().iter().all(|&t| t == T),
        "at rest the tension is T0 exactly"
    );
    assert!(s.stretch_ratio().iter().all(|&l| l == 1.0));
    let u0 = pluck_ic(&s, 4e-3);
    start_u(&mut s, &u0);
    step_n(&mut s, 300);
    let tension = s.tension();
    println!("tension spread {:.3e} T0", ptp(&tension) / T);
    assert!(
        ptp(&tension) > 1e-3 * T,
        "tension should vary along a plucked string"
    );
    assert_eq!(tension.len(), 64);
}

// == the two polarizations
// =========================================================================

#[test]
fn a_planar_pluck_stays_planar_to_the_bit() {
    // Carried from `test_planar_ic_stays_bit_exactly_planar`. Every `w` term is multiplied by `w`,
    // so a `w` that starts at zero can never acquire a bit. If this ever reads 1e-30, some term is
    // not routed through `r²` and the isotropy is fake.
    let mut s = Geo::new(64).build();
    let u0 = pluck_ic(&s, 4e-3);
    start_u(&mut s, &u0);
    step_n(&mut s, 400);
    assert_eq!(max_abs(&s.w), 0.0);
    assert_eq!(max_abs(&s.w_prev), 0.0);
    // ...while the in-plane and longitudinal fields are emphatically alive.
    assert!(max_abs(&s.u) > 1e-4);
    assert!(max_abs(&s.v) > 0.0);
}

#[test]
fn planar_stays_planar_when_the_polarizations_are_detuned() {
    // Carried from `test_planar_subspace_invariance_survives_detuning`: whirling must be seeded,
    // never leaked, or a threshold measurement would be measuring the leak.
    let mut s = Geo {
        kappa_w: Some(0.5),
        ..Geo::new(64)
    }
    .build();
    assert!(!s.is_degenerate());
    let u0 = mode_ic(&s, 1, 5e-3);
    start_u(&mut s, &u0);
    step_n(&mut s, 400);
    assert_eq!(max_abs(&s.w), 0.0);
}

#[test]
fn the_other_polarization_is_equally_invariant() {
    // Carried from `test_the_other_polarization_is_equally_invariant`: `u` is not privileged.
    let mut s = Geo::new(64).build();
    let (z, w0) = (zeros(&s), pluck_ic(&s, 4e-3));
    start(&mut s, &z, &w0, &z);
    step_n(&mut s, 400);
    assert_eq!(max_abs(&s.u), 0.0);
    assert!(max_abs(&s.w) > 1e-4);
}

#[test]
fn a_ninety_degree_rotation_is_bit_exact() {
    // Carried from `test_ninety_degree_swap_is_bit_exact`: at `kappa_u == kappa_w` the `u` and `w`
    // rows are literally the same expression, so this is a claim about SHARED code.
    let (mut a, mut b) = (Geo::new(48).build(), Geo::new(48).build());
    let (z, ic) = (zeros(&a), pluck_ic(&a, 4e-3));
    start(&mut a, &ic, &z, &z);
    start(&mut b, &z, &ic, &z);
    for _ in 0..300 {
        step(&mut a);
        step(&mut b);
    }
    assert_eq!(a.u, b.w);
    assert_eq!(a.w, b.u);
    assert_eq!(a.v, b.v);
    assert_eq!(a.energy(), b.energy());
}

/// Run a reference and a rotated twin from the same pluck; return the rotated run's worst gaps
/// `(u, w, v)` relative to the pluck's height, and both energies.
fn rotated_pair(angle: f64, kappa: f64, kappa_w: f64) -> ([f64; 3], f64, f64) {
    let cfg = Geo {
        kappa,
        kappa_w: Some(kappa_w),
        ..Geo::new(48)
    };
    let (mut r, mut rot) = (cfg.build(), cfg.build());
    let (z, ic) = (zeros(&r), pluck_ic(&r, 4e-3));
    let (ca, sa) = (angle.cos(), angle.sin());
    let cu: Vec<f64> = ic.iter().map(|x| ca * x).collect();
    let cw: Vec<f64> = ic.iter().map(|x| sa * x).collect();
    start(&mut r, &ic, &z, &z);
    start(&mut rot, &cu, &cw, &z);
    for _ in 0..250 {
        step(&mut r);
        step(&mut rot);
    }
    let scale = max_abs(&ic);
    let gap = |got: &[f64], want: Vec<f64>| {
        got.iter()
            .zip(&want)
            .fold(0.0f64, |m, (g, w)| m.max((g - w).abs()))
            / scale
    };
    let eu: Vec<f64> = (0..r.u.len()).map(|i| ca * r.u[i] - sa * r.w[i]).collect();
    let ew: Vec<f64> = (0..r.u.len()).map(|i| sa * r.u[i] + ca * r.w[i]).collect();
    let gaps = [gap(&rot.u, eu), gap(&rot.w, ew), gap(&rot.v, r.v.clone())];
    (gaps, rot.energy(), r.energy())
}

#[test]
fn an_arbitrary_rotation_commutes_with_the_dynamics() {
    // Carried from `test_arbitrary_rotation_commutes_with_the_dynamics` (four angles). An arbitrary
    // angle genuinely exercises `r² = u_x² + w_x²`; only round-off separates the rotated run from
    // the rotation of the unrotated one. The Python's `pytest.approx(rel=1e-11)` on the energy also
    // admitted `abs = 1e-12` J, about 1e-10 of this 1e-2 J energy; the relative bar alone is
    // carried.
    for angle in [0.3, PI / 4.0, 1.1, 2.7] {
        let (gaps, e_rot, e_ref) = rotated_pair(angle, KAPPA, KAPPA);
        let e_gap = (e_rot - e_ref).abs() / e_ref.abs();
        println!("angle {angle}: gaps {gaps:?}, energy {e_gap:.3e}");
        for g in gaps {
            assert!(g < 1e-11, "angle {angle}: {gaps:?}");
        }
        assert!(e_gap <= 1e-11, "angle {angle}: energy {e_gap:.3e}");
    }
}

#[test]
fn detuning_breaks_rotation_only_through_the_linear_operator() {
    // Carried from
    // `test_rotation_invariance_is_broken_by_detuning_but_isotropy_of_the_nonlinearity_is_not`:
    // `kappa_u != kappa_w` must break rotational invariance (that is what buys whirling), and the
    // same run with the bending matched must not — so the anisotropy did not leak into the discrete
    // gradient.
    let detuned = rotated_pair(0.7, KAPPA, 0.5 * KAPPA).0[0];
    let isotropic = rotated_pair(0.7, KAPPA, KAPPA).0[0];
    println!("detuned {detuned:.4e}, isotropic {isotropic:.3e}");
    assert!(
        detuned > 1e-6,
        "kappa_u != kappa_w must break rotational invariance"
    );
    assert!(
        isotropic < 1e-11,
        "with kappa_u == kappa_w the model must stay isotropic"
    );
}

#[test]
fn the_discrete_gradient_is_equivariant_under_rotation() {
    // Carried from `test_nonlinearity_depends_on_the_polarizations_only_through_r_squared`: a test
    // on the operator itself, so a failure points at the DG rather than at the scheme around it.
    let p = Geo::new(24).params().unwrap();
    let n = p.n;
    let q_plus = draw(17, 0, n, 0.05);
    let q_minus = draw(17, 3 * n, n, 0.05);
    let (ca, sa) = (0.9f64.cos(), 0.9f64.sin());
    let rotate = |q: &[f64]| {
        let mut out = q.to_vec();
        for i in 0..n {
            out[i] = ca * q[i] - sa * q[n + i];
            out[n + i] = sa * q[i] + ca * q[n + i];
        }
        out
    };
    let got = geo::dg_force(&rotate(&q_plus), &rotate(&q_minus), p.a);
    let expected = rotate(&geo::dg_force(&q_plus, &q_minus, p.a));
    let gap = got
        .iter()
        .zip(&expected)
        .fold(0.0f64, |m, (g, e)| m.max((g - e).abs()));
    println!("equivariance gap {:.3e}", gap / max_abs(&expected));
    assert!(gap < 1e-13 * max_abs(&expected));
    // ...and the stored energy is a scalar under the same rotation.
    let (v0, v1) = (
        geo::nl_density(&q_plus, p.a, p.h),
        geo::nl_density(&rotate(&q_plus), p.a, p.h),
    );
    assert!((v1 - v0).abs() <= 1e-12 * v0.abs(), "{v1:e} vs {v0:e}");
}

#[test]
fn a_circular_mode_is_a_different_motion_with_twice_the_energy() {
    // Carried from `test_circular_and_planar_are_different_motions_and_not_equal_energy`. At equal
    // amplitude a circular mode runs BOTH polarizations at full amplitude: 2x the planar energy.
    let amp = 4e-3;
    let mut planar = Geo::new(48).build();
    let u0 = mode_ic(&planar, 1, amp);
    start_u(&mut planar, &u0);

    let mut circ = Geo::new(48).build();
    let shape = mode_ic(&circ, 1, amp);
    let omega = 2.0 * PI * circ.p.c / (2.0 * circ.p.l);
    let z = zeros(&circ);
    let w_dot: Vec<f64> = shape.iter().map(|x| omega * x).collect();
    circ.set_state(&shape, &z, &z, &[z.clone(), w_dot, z.clone()]); // u = A cos, w = A sin
    for _ in 0..300 {
        step(&mut planar);
        step(&mut circ);
    }
    let ratio = circ.energy() / planar.energy();
    println!("energy ratio {ratio:.6}");
    assert!(
        max_abs(&circ.w) > 1e-4,
        "the circular run must actually leave the plane"
    );
    assert_eq!(max_abs(&planar.w), 0.0);
    assert_eq!((circ.n_not_converged, planar.n_not_converged), (0, 0));
    assert!(
        (ratio - 2.0).abs() <= 0.05 * 2.0,
        "2x the planar energy, not 1x: {ratio}"
    );
}
