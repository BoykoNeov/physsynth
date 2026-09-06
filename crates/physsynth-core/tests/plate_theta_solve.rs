//! The coupled step against a **supplied** operator — `docs/dev/air-box-vk-newton-plan.md` Part 1.
//!
//! `VkCoupledStep` used to reach its theta-scheme factorization as `p.lin.lu`. It now reaches it
//! through [`ThetaSolve`], so the air box's surface seam can point the *same* step at the loaded
//! matrix SciPy factored for it instead of carrying a second copy of the sweep.
//!
//! Two claims, and the second is the one that would otherwise pass unnoticed:
//!
//! 1. handing back the plate's own operator reproduces the shipped step **bit for bit**, on both
//!    boundary arms and under both iterations, so the indirection costs nothing;
//! 2. a substituted operator moves the sweep **and** the Jacobian-vector product. Routing only the
//!    sweep would leave Newton solving the right residual with the wrong tangent — a preconditioner
//!    mismatch, which converges to the correct answer, passes every energy bar, and is simply
//!    slower. Nothing else in the suite could see it.

use physsynth_core::plate::{
    vk_step, vk_step_with, Boundary, CoupleMethod, ThetaSolve, VkCoupledStep, VkParams, VkPlate,
    VkSpec,
};
use physsynth_core::sparse_lu::SparseLuError;

/// A struck plate — `tests/plate.rs`'s own loud fixture, and it has to be loud for a reason.
///
/// `J = I` exactly at zero amplitude and *numerically* at small amplitude, so a quiet plate cannot
/// tell a routed Jacobian from an unrouted one: the correction `c A^-1(...)` falls below the ulp of
/// the `d` it is subtracted from and both readings return `d` unchanged. The free edge reaches that
/// point sooner than the supported one, because its `couple_factor` carries an extra `h^2` — at
/// `w = 6e` with an 8 cm strike the free arm is already vacuous while the supported arm still
/// resolves. These are `JAC_CASES`' loud parameters, where `the_jacobian_is_far_from_the_identity`
/// already asserts the margin exists.
fn struck(boundary: Boundary, method: CoupleMethod, amp: f64) -> VkPlate {
    let mut vk = VkPlate::new(
        VkParams::new(&VkSpec {
            lx: 0.4,
            ly: 0.4,
            young: 2.0e11,
            thickness: 1e-3,
            nu: 0.3,
            rho: 7860.0,
            fs: 48_000.0,
            n: 20,
            boundary: Some(boundary),
            couple_max_iter: 200,
            couple_method: Some(method),
            ..VkSpec::default()
        })
        .expect("a valid plate"),
    );
    let p = &vk.p.lin;
    let (cx, cy) = (0.5 * p.lx, 0.5 * p.ly);
    let a = amp * vk.p.thickness;
    let w = 0.08;
    let u0: Vec<f64> = p
        .mask
        .flags()
        .iter()
        .enumerate()
        .filter(|(_, &alive)| alive)
        .map(|(idx, _)| {
            let (dx, dy) = (p.x[idx] - cx, p.y[idx] - cy);
            a * (-((dx * dx + dy * dy) / (w * w))).exp()
        })
        .collect();
    let zero = vec![0.0; p.n_live];
    vk.set_state(&u0, &zero).expect("the Airy solve factors");
    vk
}

/// A normalised pseudo-random direction — `tests/plate.rs`'s `jac_direction`, xorshift and all.
///
/// Copied rather than shared because each file under `tests/` is its own crate, and copied
/// deliberately from *that* one so the two Jacobian tests probe the same directions.
fn direction(n: usize) -> Vec<f64> {
    let mut s = 0x2545_f491_4f6c_dd1d_u64 ^ 0x9e37_79b9_7f4a_7c15_u64;
    let mut d: Vec<f64> = (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 11) as f64) / 9_007_199_254_740_992.0 - 0.5
        })
        .collect();
    let norm = d.iter().map(|v| v * v).sum::<f64>().sqrt();
    for v in d.iter_mut() {
        *v /= norm;
    }
    d
}

/// An operator that is the plate's own, scaled — deliberately wrong, and wrong by a knowable factor.
///
/// Not a random matrix: `scale * A^-1` is still a perfectly usable *preconditioner*, which is the
/// point. A Newton whose tangent used this would still converge; only its cost would move. That is
/// exactly the failure the second test exists to make visible.
struct Scaled<'a> {
    inner: &'a dyn ThetaSolve,
    scale: f64,
}

impl ThetaSolve for Scaled<'_> {
    fn solve(&self, b: &[f64]) -> Result<Vec<f64>, SparseLuError> {
        Ok(self
            .inner
            .solve(b)?
            .into_iter()
            .map(|v| self.scale * v)
            .collect())
    }
}

/// Supplying the plate's own factorization is the shipped step, to the last bit, on every arm.
///
/// The right-hand side is taken from the context the model builds for itself, because that is what
/// the seam does: it assembles its own (the room's two load terms are in it) and hands it over
/// already formed. What this asserts is that `with_rhs` + `vk_step_with` is the same *function* as
/// `vk_step` once the two inputs it lets a caller replace are the ones `vk_step` would have used.
#[test]
fn the_plates_own_operator_through_the_trait_is_the_shipped_step() {
    for boundary in [Boundary::Supported, Boundary::Free] {
        for method in [CoupleMethod::Picard, CoupleMethod::Newton] {
            let vk = struck(boundary, method, 16.0);
            let p = &vk.p;
            let (u, u_prev) = (vk.u.clone(), vk.u_prev.clone());
            let (f, f_prev) = (vk.f.clone(), vk.f_prev.clone());

            let shipped = vk_step(&u, &u_prev, &f, &f_prev, None, p).expect("a solvable step");

            let own = VkCoupledStep::new(&u, &u_prev, &f_prev, None, p);
            let ctx = VkCoupledStep::with_rhs(own.rhs_lin.clone(), &u_prev, &f_prev, p, &p.lin.lu);
            let routed = vk_step_with(&ctx, &u, &u_prev, &f).expect("a solvable step");

            assert_eq!(
                shipped, routed,
                "{boundary:?}/{method:?}: routing through ThetaSolve moved the step"
            );
            // The free edge's extra h^2 lives in `couple_factor` and in nothing else, so a
            // constructor that rebuilt it would show up here rather than as a factor-of-h^2 in the
            // physics three parts later.
            assert_eq!(ctx.couple_factor, own.couple_factor);
        }
    }
}

/// A substituted operator must move BOTH readings of the map, not just the sweep.
///
/// `sweep_from` is the residual's half; `jacobian_vector` is the tangent's. They are separate call
/// sites and it is entirely possible to route one and forget the other — the result converges, so
/// no energy bar and no convergence bar can see it.
#[test]
fn a_substituted_operator_moves_the_sweep_and_the_jacobian_alike() {
    for boundary in [Boundary::Supported, Boundary::Free] {
        let vk = struck(boundary, CoupleMethod::Picard, 16.0);
        let p = &vk.p;
        let own = VkCoupledStep::new(&vk.u, &vk.u_prev, &vk.f_prev, None, p);
        let wrong = Scaled {
            inner: &p.lin.lu,
            scale: 2.0,
        };
        let ctx = VkCoupledStep::with_rhs(own.rhs_lin.clone(), &vk.u_prev, &vk.f_prev, p, &wrong);

        let seed = VkCoupledStep::seed(&vk.u, &vk.u_prev);
        let av = own.averages(&seed).expect("the Airy solve factors");

        // The sweep IS the operator's output, so doubling it is exact -- no cancellation anywhere
        // between the solve and the assertion, and `2.0 * x` is representable whenever `x` is.
        let sweep_own = own.sweep_from(&av).expect("a solvable sweep");
        let sweep_sub = ctx.sweep_from(&av).expect("a solvable sweep");
        assert!(
            sweep_own.iter().any(|v| *v != 0.0),
            "{boundary:?}: the fixture's sweep is zero, so this test would pass on anything"
        );
        for i in 0..sweep_own.len() {
            assert_eq!(
                sweep_sub[i],
                2.0 * sweep_own[i],
                "{boundary:?}: the sweep did not go through the supplied operator at {i}"
            );
        }

        // `tests/plate.rs`'s own direction, and it has to be this one rather than anything
        // smooth. The bracket differentiates twice, so a direction that is *linear* in the index
        // is very nearly in its kernel: a ramp produced a coupling so small that `d - J d`
        // cancelled to exactly zero on the free edge and the test passed on nothing. A normalised
        // pseudo-random vector has curvature everywhere.
        let d = direction(seed.len());
        let jv_own = own.jacobian_vector(&av, &d).expect("a solvable product");
        let jv_sub = ctx.jacobian_vector(&av, &d).expect("a solvable product");
        assert!(
            jv_own.iter().zip(&jv_sub).any(|(a, b)| (a - b).abs() > 0.0),
            "{boundary:?}: the JACOBIAN still used the plate's own operator — Newton would solve \
             the right residual with the wrong tangent, converge, and pass every bar in the suite"
        );

        // And the difference is the scale's: `J d = d - c A^-1 (...)`, so doubling the solve
        // doubles the correction and leaves the `d` alone. A test that only asserted "something
        // moved" would also pass on a typo that routed the wrong vector into the product.
        //
        // The bar is relative rather than exact, and that is the arithmetic's doing, not a
        // hedge: recovering the correction as `d - Jd` is a difference of two nearly equal
        // numbers, so it carries the cancellation of whatever `c A^-1(...)/d` happens to be. The
        // ratio itself is clean; only its reconstruction is not. So the bar is absolute against
        // the LARGEST correction rather than relative per component: the cancellation is set by
        // the size of what was subtracted away, which is the same for every component, and a
        // per-component relative bar would demand more precision of a small entry than the
        // subtraction can carry.
        let corr_own: Vec<f64> = (0..d.len()).map(|i| d[i] - jv_own[i]).collect();
        let corr_sub: Vec<f64> = (0..d.len()).map(|i| d[i] - jv_sub[i]).collect();
        let biggest = corr_own.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        assert!(
            biggest > 0.0,
            "{boundary:?}: the fixture's Jacobian correction is zero -- nothing to compare"
        );
        for i in 0..d.len() {
            let gap = (corr_sub[i] - 2.0 * corr_own[i]).abs();
            assert!(
                gap <= 1e-9 * biggest,
                "{boundary:?}: component {i} moved by {} rather than by the operator's scale of 2",
                corr_sub[i] / corr_own[i]
            );
        }
    }
}
