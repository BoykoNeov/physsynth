//! Model #7g — the mallet on a gong. The nested solve's native bars.
//!
//! An outer chord on the contact force wrapped around a full von Karman plate solve per
//! evaluation. `crates/physsynth-core/src/mallet.rs`'s gong section derives the algorithm and
//! `docs/dev/mallet-gong-plan.md` records what these bars measure.
//!
//! What is asserted here and not from Python: everything that needs the plate's *pre-step*
//! buffers in hand (the exact outer tangent), the cost ratio against a bare step taken from the
//! identical state, and the two failure attributions.

use physsynth_core::collision::ContactError;
use physsynth_core::mallet::{
    vk_drive_point_tangent, MalletPlate, MalletVkPlate, ParamError, PlateParams, VkContactError,
    VkPlateParams,
};
use physsynth_core::plate::{
    vk_step, Boundary, Params, Plate, PlateSpec, VkParams, VkPlate, VkSpec,
};

/// The shipped gong: a 0.4 m square 1 mm steel plate, simply supported, 361 live nodes.
///
/// The material and geometry are `tests/test_vk_energy.py`'s, so the nonlinear plate this model
/// drives is the one every existing von Karman bar was measured on.
fn gong_spec() -> VkSpec {
    VkSpec {
        lx: 0.4,
        ly: 0.4,
        young: 2.0e11,
        thickness: 1.0e-3,
        nu: 0.3,
        rho: 7800.0,
        fs: 48_000.0,
        n: 20,
        ..VkSpec::default()
    }
}

/// The shipped mallet: a 50 g head on piano-ish felt, struck off every low mode's symmetry axis.
const MASS: f64 = 0.05;
const STIFFNESS: f64 = 5.0e4;
const ALPHA: f64 = 2.3;
/// The shipped outer tolerance, and it is chosen **above the inner solve's floor** rather than as
/// tight as the arithmetic allows -- see `vk_plate_step`. At 1e-14 the hardest strike here plateaus
/// at 5.1e-14 and spends the cap on one step in eight thousand.
const OUTER_TOL: f64 = 1e-13;

fn mallet_params(p: &VkParams, mass: f64, alpha: f64, v0: f64, outer_tol: f64) -> VkPlateParams {
    VkPlateParams::new(
        p, mass, STIFFNESS, alpha, 0.0, 0.12, 0.16, v0, 0.0, 1e-12, 1e-14, 60, outer_tol, 20,
    )
    .expect("a valid mallet")
}

fn gong(spec: &VkSpec, mass: f64, v0: f64) -> MalletVkPlate {
    let p = VkParams::new(spec).expect("a valid gong");
    let params = mallet_params(&p, mass, ALPHA, v0, OUTER_TOL);
    MalletVkPlate::new(params, VkPlate::new(p), 0.0, v0)
}

/// One step's context: `(u, u_prev, F_prev)` as they stood going in, and the `w` it accepted.
///
/// What `vk_drive_point_tangent` needs, and the reason it needs it is the point: `J` has to be
/// evaluated inside a *specific* step, so a caller cannot reconstruct the context after the plate
/// has rolled its buffers.
type StepContext = ((Vec<f64>, Vec<f64>, Vec<f64>), Vec<f64>);

fn peak(v: &[f64]) -> f64 {
    v.iter().fold(0.0f64, |a, &b| a.max(b.abs()))
}

/// A run's energy range and how far the plate ended up translated.
fn ring(m: &mut MalletVkPlate, steps: usize) -> (f64, f64) {
    let e0 = m.energy();
    let (mut lo, mut hi) = (e0, e0);
    for _ in 0..steps {
        m.step().expect("a gong step");
        let e = m.energy();
        lo = lo.min(e);
        hi = hi.max(e);
    }
    let rigid = m.plate.u.iter().sum::<f64>() / m.plate.u.len() as f64;
    ((hi - lo) / e0.abs(), rigid)
}

// -- the anchor: switch the coupling off and this IS model #7p -------------------------------------

#[test]
fn a_linear_gong_is_the_linear_mallet_plate_and_the_outer_loop_knows_it() {
    // With `nonlinear = false` the plate is affine in `f_ext`, so `Psi(f) = u_free - g_s f` and
    // the chord's effective free node is `u_free` **whatever the trial force was**. The outer
    // loop's first residual is therefore exactly zero and it exits at one iteration -- which is
    // the structural half of this test and the one a sign error in `f_ext = -f e_node` cannot
    // survive: a flipped sign leaves `u_eff` a function of `f` and the count goes to two or more.
    let spec = VkSpec {
        nonlinear: false,
        ..gong_spec()
    };
    let vk = VkParams::new(&spec).expect("a valid plate");
    let mut g = MalletVkPlate::new(
        mallet_params(&vk, MASS, ALPHA, 6.0, OUTER_TOL),
        VkPlate::new(vk.clone()),
        0.0,
        6.0,
    );

    let twin = Params::new(&PlateSpec {
        lx: spec.lx,
        ly: vk.lin.ly,
        kappa: vk.lin.kappa,
        rho: vk.rho_s,
        fs: spec.fs,
        n: spec.n,
        theta: vk.lin.theta,
        boundary: Some(Boundary::Supported),
        nu: Some(spec.nu),
        ..PlateSpec::default()
    })
    .expect("a valid twin");
    let lin_params = PlateParams::new(
        &twin, MASS, STIFFNESS, ALPHA, 0.0, 0.12, 0.16, 0.0, 1e-12, 1e-14, 60,
    )
    .expect("a valid mallet");

    // The frozen tangent is the SAME NUMBER, to the bit: both are built from `A^-1 e_node` scaled
    // by the same `k^2 / force_denominator`, and the gong reads the nonlinear plate's stored
    // `force_denominator` where the linear one calls `Params::force_denominator()`.
    assert_eq!(g.params().g_s, lin_params.g_s);
    assert_eq!(g.params().g_h, lin_params.g_h);
    assert_eq!(g.params().node, lin_params.node);

    let mut lin = MalletPlate::new(lin_params, Plate::new(twin), 0.0, 6.0);
    let mut worst_field = 0.0f64;
    let mut worst_force = 0.0f64;
    let mut ever_bitwise = false;
    for _ in 0..400 {
        g.step().expect("a gong step");
        lin.step().expect("a plate step");
        assert_eq!(
            g.last.as_ref().expect("stepped").n_outer,
            1,
            "an affine plate makes the chord's effective free node independent of the trial force"
        );
        let scale = peak(&lin.plate.u).max(1e-300);
        let d = g
            .plate
            .u
            .iter()
            .zip(lin.plate.u.iter())
            .fold(0.0f64, |a, (&x, &y)| a.max((x - y).abs()));
        worst_field = worst_field.max(d / scale);
        let f_lin = lin.state().contact_force;
        worst_force =
            worst_force.max((g.state().contact_force - f_lin).abs() / f_lin.abs().max(1.0));
        ever_bitwise |= g.plate.u == lin.plate.u;
    }
    // Measured 2.08e-14 / 2.07e-14 over 400 steps.
    assert!(worst_field < 1e-13, "field {worst_field:e}");
    assert!(worst_force < 1e-13, "force {worst_force:e}");
    // And NOT to the bit, for the plate section's own reason one level out: this model puts the
    // force in the right-hand side before the solve, `MalletPlate` adds `influence * f` after it,
    // and a sparse LU back-substitution is not a linear map over doubles. Asserted rather than
    // merely expected, because a bitwise agreement here would mean one of them stopped doing what
    // its header says it does.
    assert!(
        !ever_bitwise,
        "the two routes rounded identically on some step -- one of them is no longer solving the \
         way its documentation says"
    );
}

// -- the miss, which is exact ----------------------------------------------------------------------

#[test]
fn a_mallet_that_never_lands_leaves_the_gong_bit_identical() {
    // Structural, not a floating-point coincidence: a zero force short-circuits to the force-free
    // advance and returns it *unmodified*. Driving the plate with a zero force VECTOR would add
    // `+0.0` to every right-hand-side entry instead, which is the identity for every double
    // except `-0.0`.
    let spec = gong_spec();
    let p = VkParams::new(&spec).expect("a gong");
    let mut m = MalletVkPlate::new(
        VkPlateParams::new(
            &p, MASS, STIFFNESS, ALPHA, 0.0, 0.12, 0.16, 1.0, 0.01, 1e-12, 1e-14, 60, OUTER_TOL, 20,
        )
        .expect("a mallet"),
        VkPlate::new(p.clone()),
        0.01,
        -1.0, // receding: it never reaches the plate
    );
    let mut bare = VkPlate::new(p);
    // A real deflection, so the bare plate is doing nonlinear work while the mallet is away.
    let ic: Vec<f64> = (0..m.plate.p.lin.n_live)
        .map(|i| 2e-3 * ((i % 7) as f64 - 3.0) / 3.0)
        .collect();
    let zero = vec![0.0; ic.len()];
    m.plate.set_state(&ic, &zero).expect("a start");
    bare.set_state(&ic, &zero).expect("a start");
    for _ in 0..200 {
        m.step().expect("a gong step");
        bare.step(None).expect("a bare step");
        assert_eq!(m.state().contact_force, 0.0);
        assert_eq!(m.last.as_ref().expect("stepped").n_outer, 0);
        assert_eq!(
            m.plate.u, bare.u,
            "a miss is not free of the plate's history"
        );
        assert_eq!(m.plate.f, bare.f);
    }
}

// -- the money test: the discrete gradient still telescopes through a nested solve -----------------

#[test]
fn energy_is_conserved_through_the_nested_solve() {
    // The plate is implicit and the force enters its right-hand side, so dotting the theta scheme
    // with `w^{n+1} - w^{n-1}` gives the same discrete power `f . delta_t. u` the membrane gives
    // -- the von Karman coupling telescopes on its own and adds nothing to the contact's ledger.
    // What is new is that the identity now holds only as well as the OUTER loop converges, which
    // is why `outer_tol` and not `newton_tol` is the tolerance this bar is a function of.
    for (mass, v0) in [(0.05, 3.0), (0.05, 6.0), (0.02, 6.0), (0.05, 12.0)] {
        let mut m = gong(&gong_spec(), mass, v0);
        let e0 = m.energy();
        let (mut lo, mut hi) = (e0, e0);
        let mut membrane_share = 0.0f64;
        for _ in 0..2000 {
            m.step().expect("a gong step");
            let l = m.last.as_ref().expect("stepped");
            assert!(l.outer_converged && l.inner_converged);
            let e = m.energy();
            lo = lo.min(e);
            hi = hi.max(e);
            membrane_share = membrane_share.max(m.plate.membrane_energy() / e0.abs());
        }
        let drift = (hi - lo) / e0.abs();
        // Measured 4.1e-13 to 6.0e-13 -- the same order as model #7p's linear plate, so the
        // nested solve costs nothing in fidelity. The project's contract is 1e-10.
        assert!(drift < 1e-11, "M={mass} v0={v0} drift {drift:e}");
        // ... and the run has to be nonlinear, or the bar is re-testing the linear theta scheme.
        assert!(
            membrane_share > 0.01,
            "M={mass} v0={v0} membrane share {membrane_share:e} -- this strike never left the \
             linear regime, so the conservation above is model #5's and not the gong's"
        );
    }
}

#[test]
fn loss_and_hysteresis_each_make_the_total_monotone() {
    for (sigma, lam_h) in [(2.0, 0.0), (0.0, 5.0)] {
        let spec = VkSpec {
            sigma,
            ..gong_spec()
        };
        let p = VkParams::new(&spec).expect("a gong");
        let params = VkPlateParams::new(
            &p, MASS, STIFFNESS, ALPHA, lam_h, 0.12, 0.16, 6.0, 0.0, 1e-12, 1e-14, 60, OUTER_TOL,
            20,
        )
        .expect("a mallet");
        let mut m = MalletVkPlate::new(params, VkPlate::new(p), 0.0, 6.0);
        let mut prev = m.energy();
        let mut worst = 0.0f64;
        for _ in 0..1500 {
            m.step().expect("a gong step");
            let e = m.energy();
            worst = worst.max((e - prev) / prev.abs());
            prev = e;
        }
        // Measured 1.2e-15 / 4.8e-15.
        assert!(
            worst < 1e-13,
            "sigma={sigma} lam_h={lam_h} rose by {worst:e}"
        );
    }
}

// -- the batch's headline claim: what the nested solve actually costs ------------------------------

#[test]
fn the_nested_solve_costs_between_two_and_three_bare_steps() {
    // The denominator is a **bare** `VkPlate` step taken from the mallet's own state at that
    // moment -- what this plate would have cost with no mallet on it, at the amplitude the mallet
    // actually drove it to. Anything else compares two different plates.
    //
    // `docs/dev/mallet-plate-plan.md` section 6 and HANDOFF section 14.1 both predicted "roughly
    // ten to a hundred plate solves per timestep". This is that prediction's test, and it fails
    // it by an order of magnitude in the cheap direction.
    for (mass, v0) in [(0.02, 3.0), (0.05, 6.0), (0.2, 6.0)] {
        let mut m = gong(&gong_spec(), mass, v0);
        let (mut mine, mut bare, mut in_contact, mut contact_solves) = (0usize, 0usize, 0usize, 0);
        for _ in 0..2000 {
            let b = vk_step(
                &m.plate.u,
                &m.plate.u_prev,
                &m.plate.f,
                &m.plate.f_prev,
                None,
                &m.plate.p,
            )
            .expect("a bare step");
            m.step().expect("a gong step");
            let l = m.last.as_ref().expect("stepped");
            bare += b.n_solves;
            mine += l.n_solves;
            if l.n_outer > 0 {
                in_contact += b.n_solves;
                contact_solves += l.n_solves;
            }
        }
        let whole = mine as f64 / bare as f64;
        let during = contact_solves as f64 / in_contact as f64;
        // Measured 1.9 / 2.0 / 2.3 over the whole run, 2.4-2.9 during contact. A miss costs
        // exactly one bare step, which is what pulls the whole-run figure below the contact one.
        assert!(
            (1.5..3.0).contains(&whole),
            "M={mass} v0={v0}: whole-run ratio {whole:.3}"
        );
        assert!(
            (2.0..4.0).contains(&during),
            "M={mass} v0={v0}: in-contact ratio {during:.3}"
        );
        assert!(whole < during, "a miss must cost less than a hit");
    }
}

// -- the instrument that explains the cost --------------------------------------------------------

#[test]
fn the_outer_tangent_is_closed_form_and_the_mallet_enters_it_only_through_g() {
    // The plan this batch closes says a gong has "no closed-form derivative for the outer
    // iteration, a secant or a bracket instead". It has one. Differentiating the fixed point
    // gives `dw/df = -J^-1 influence` for exactly the `J` the Newton batch already asserted
    // against a finite difference, so `g_exact = [J^-1 influence]_node + g_h` and the chord's
    // contraction is bounded by `|1 - g_exact/g|`.
    //
    // The claim under test is sharper than "a heavy mallet converges more slowly", which is true
    // but is not the mechanism: the NUMERATOR `|g - g_exact|` is a property of the plate's
    // deflection and the mallet does not appear in it at all. Same plate state, five masses.
    let spec = gong_spec();
    let mut driver = gong(&spec, MASS, 6.0);
    let mut best: (f64, Option<StepContext>) = (0.0, None);
    for _ in 0..600 {
        let before = (
            driver.plate.u.clone(),
            driver.plate.u_prev.clone(),
            driver.plate.f_prev.clone(),
        );
        driver.step().expect("a gong step");
        let f = driver.state().contact_force;
        if f > best.0 {
            best = (f, Some((before, driver.plate.u.clone())));
        }
    }
    let (force, ctx) = (best.0, best.1.expect("the mallet landed"));
    let ((u, u_prev, f_prev), w) = ctx;
    let p = VkParams::new(&spec).expect("a gong");
    let mut f_ext = vec![0.0; p.lin.n_live];
    f_ext[driver.params().node] = -force;

    let mut numerators = Vec::new();
    let mut bounds = Vec::new();
    for mass in [0.005, 0.02, 0.05, 0.2, 1.0] {
        let par = mallet_params(&p, mass, ALPHA, 6.0, OUTER_TOL);
        let (g_exact, response, products) =
            vk_drive_point_tangent(&u, &u_prev, &f_prev, Some(&f_ext), &w, &par, &p)
                .expect("a tangent");
        assert!(products > 0, "a GMRES that spent nothing solved nothing");
        // The plate-only half, `[J^-1 influence]_node`, against the chord's plate-only half `g_s`.
        // Taken from `response` rather than from `g_exact - g_h`: adding `g_h` on and subtracting
        // it back off is not the identity over doubles when the two differ by four orders of
        // magnitude, which is exactly what a 1 kg mallet is -- and that arithmetic, not the
        // physics, is what a bitwise assertion on `g - g_exact` would be measuring.
        numerators.push((par.g_s - response).abs());
        bounds.push(((1.0 - g_exact / par.g).abs(), par.g));
    }
    // Bitwise: the numerator is `[(I - J^-1) influence]_node`, in which no mallet quantity appears.
    for n in &numerators {
        assert_eq!(
            *n, numerators[0],
            "the numerator is not a plate-only quantity"
        );
    }
    // So the whole mass dependence is `1/g`, and it saturates: `g -> g_s` as `M -> infinity`.
    for (bound, g) in &bounds {
        let predicted = numerators[0] / g;
        assert!(
            (bound - predicted).abs() <= 1e-12 * predicted,
            "bound {bound:e} is not numerator/g = {predicted:e}"
        );
    }
    let (light, heavy) = (bounds[0].0, bounds[4].0);
    assert!(heavy > light, "a heavier mallet must contract more slowly");
    // Measured 3.29e-3 -> 5.39e-3 across a 200x mass range: heavier IS worse, by a bounded factor
    // of `g(M->0)/g_s`, not by an amount that grows with the mass.
    assert!(
        heavy / light < 2.0,
        "the 1/g factor is bounded, ratio {}",
        heavy / light
    );
    assert!(
        heavy < 0.05,
        "and every bound is far inside a contraction: {heavy:e}"
    );
}

#[test]
fn the_outer_loop_takes_the_handful_of_iterations_the_bound_predicts() {
    for (mass, v0) in [(0.02, 1.0), (0.05, 6.0), (0.2, 6.0)] {
        let mut m = gong(&gong_spec(), mass, v0);
        let (mut hits, mut total, mut worst) = (0usize, 0usize, 0usize);
        for _ in 0..1500 {
            m.step().expect("a gong step");
            let l = m.last.as_ref().expect("stepped");
            if l.n_outer > 0 {
                hits += 1;
                total += l.n_outer;
                worst = worst.max(l.n_outer);
            }
        }
        let mean = total as f64 / hits as f64;
        // Measured: mean 1.4-1.9, max 2 at the shipped `outer_tol = 1e-14`. The cap is 20.
        assert!(worst <= 4, "M={mass} v0={v0}: {worst} outer iterations");
        assert!(mean < 2.5, "M={mass} v0={v0}: mean {mean:.2}");
    }
}

// -- the free branch, where the READ-OUT and not the scheme sets the bar ---------------------------

#[test]
fn the_free_branch_read_out_error_is_quadratic_in_the_rigid_drift() {
    // A point strike feeds the free plate's `{1, x, y}` rigid nullspace and the gong translates
    // away for ever after -- physics, and model #7p attributed the resulting energy read-out error
    // to `Plate::energy()`'s bending form rather than to the mallet. It carries over here
    // unchanged, and the von Karman term adds nothing to it because the Monge-Ampere bracket
    // annihilates the nullspace too.
    //
    // It is only *visible* at the shipped `outer_tol`. At the 1e-12 this model was first built
    // with, the outer loop's own error was larger than the cancellation error and the free branch
    // measured no worse than the supported one -- tightening a solver tolerance is what uncovered
    // a defect in a read-out.
    let spec = VkSpec {
        boundary: Some(Boundary::Free),
        ..gong_spec()
    };
    let mut ratios = Vec::new();
    for steps in [2000usize, 4000, 8000] {
        let (drift, rigid) = ring(&mut gong(&spec, MASS, 6.0), steps);
        ratios.push(drift / (rigid * rigid));
    }
    // Constant to within a factor of 1.1 across a 4x window and a 5x rigid displacement: measured
    // 7.3e-8 / 6.9e-8 / 7.0e-8.
    let (min, max) = ratios
        .iter()
        .fold((f64::MAX, 0.0f64), |(a, b), &r| (a.min(r), b.max(r)));
    assert!(max / min < 1.3, "drift/rigid^2 ranged {min:e}..{max:e}");

    // And the supported control on the identical mallet, which is what makes this an attribution
    // rather than an excuse.
    let (supported, _) = ring(&mut gong(&gong_spec(), MASS, 6.0), 8000);
    let (free, _) = ring(&mut gong(&spec, MASS, 6.0), 8000);
    // Measured 5.9e-13 supported against 3.3e-10 free at 8000 steps -- a factor of 550, and the
    // free branch's number bounds a read-out, not the scheme.
    assert!(supported < 1e-11, "supported control {supported:e}");
    assert!(free < 1e-8, "free read-out {free:e}");
    assert!(
        free > 50.0 * supported,
        "the free branch stopped being the one with the rigid mode in it"
    );
}

// -- refusals -------------------------------------------------------------------------------------

#[test]
fn the_gong_refuses_what_every_mallet_refuses_plus_its_own_two() {
    let p = VkParams::new(&gong_spec()).expect("a gong");
    // `(mass, K, alpha, hysteresis, gap, outer_tol, outer_max_iter, expected)` -- one field per
    // check, in the order the checks run, so a row reads as the call it stands for.
    type Case = (f64, f64, f64, f64, f64, f64, usize, ParamError);
    let cases: [Case; 7] = [
        (
            0.0,
            STIFFNESS,
            ALPHA,
            0.0,
            0.0,
            1e-14,
            20,
            ParamError::NonPositiveMass,
        ),
        (
            MASS,
            0.0,
            ALPHA,
            0.0,
            0.0,
            1e-14,
            20,
            ParamError::NonPositiveStiffness,
        ),
        (
            MASS,
            STIFFNESS,
            0.5,
            0.0,
            0.0,
            1e-14,
            20,
            ParamError::AlphaTooSmall,
        ),
        (
            MASS,
            STIFFNESS,
            ALPHA,
            -1.0,
            0.0,
            1e-14,
            20,
            ParamError::NegativeHysteresis,
        ),
        (
            MASS,
            STIFFNESS,
            ALPHA,
            0.0,
            -1.0,
            1e-14,
            20,
            ParamError::NegativeGap,
        ),
        (
            MASS,
            STIFFNESS,
            ALPHA,
            0.0,
            0.0,
            0.0,
            20,
            ParamError::NonPositiveOuterTol,
        ),
        (
            MASS,
            STIFFNESS,
            ALPHA,
            0.0,
            0.0,
            1e-14,
            0,
            ParamError::TooFewOuterIters,
        ),
    ];
    for (mass, k, alpha, lam, gap, otol, omax, want) in cases {
        let got = VkPlateParams::new(
            &p, mass, k, alpha, lam, 0.12, 0.16, 6.0, gap, 1e-12, 1e-14, 60, otol, omax,
        )
        .expect_err("must refuse");
        assert_eq!(got, want);
    }
    // The five shared checks run FIRST and in the shared order, so a call that is wrong in more
    // than one way reports what `MalletMembrane` would report -- the outer loop's two are after.
    let both = VkPlateParams::new(
        &p, 0.0, STIFFNESS, ALPHA, 0.0, 0.12, 0.16, 6.0, 0.0, 1e-12, 1e-14, 60, 0.0, 0,
    )
    .expect_err("must refuse");
    assert_eq!(both, ParamError::NonPositiveMass);
}

#[test]
fn a_failure_of_the_plates_own_iteration_is_not_reported_as_a_contact_bug() {
    // Measured while mapping the boundary: at a low sample rate and a hard strike the Picard loop
    // stops converging, and the very next thing that happens is that the contact solve scans its
    // bracket six times and finds no sign change. `ContactError::NoRoot`'s own text calls that
    // "impossible for the monotone convex-potential force" -- and it is. The force is fine; the
    // field the residual was built from is the output of an iteration that did not finish.
    let spec = VkSpec {
        fs: 8_000.0,
        couple_max_iter: 6,
        ..gong_spec()
    };
    let p = VkParams::new(&spec).expect("a gong");
    let params = VkPlateParams::new(
        &p, 0.05, STIFFNESS, ALPHA, 0.0, 0.12, 0.16, 30.0, 0.0, 1e-12, 1e-14, 60, OUTER_TOL, 20,
    )
    .expect("a mallet");
    let mut m = MalletVkPlate::new(params, VkPlate::new(p), 0.0, 30.0);
    let mut failure = None;
    for _ in 0..400 {
        if let Err(e) = m.step() {
            failure = Some(e);
            break;
        }
    }
    let e = failure.expect("this rig is chosen to break the inner solve");
    match &e {
        VkContactError::Contact {
            source,
            inner_converged,
        } => {
            assert_eq!(*source, ContactError::NoRoot);
            assert!(
                !inner_converged,
                "the inner solve converged, so this really is a contact failure and the fixture \
                 has stopped exercising what it was chosen for"
            );
            let text = e.to_string();
            assert!(
                text.contains("the plate's own iteration did not converge"),
                "{text}"
            );
            assert!(text.contains("couple_max_iter"), "{text}");
        }
        other => panic!("expected an attributed contact failure, got {other:?}"),
    }
}

// -- what the outer loop's cost is, and is not, a function of --------------------------------------

#[test]
fn the_outer_cost_answers_to_its_own_tolerance_and_not_to_the_plates() {
    // The prediction going in was that the plate's `couple_tol` would floor the outer residual,
    // because every `Psi(f)` carries the inner solve's error. It does not -- see `vk_plate_step`.
    // The inner error is a deterministic bias rather than noise, so the chord simply converges to
    // the perturbed map's own fixed point.
    //
    // Both halves are asserted, because the negative one is the surprising one.
    let strike = 12.0;
    let ask = |couple_tol: f64, outer_tol: f64| {
        let spec = VkSpec {
            couple_tol,
            ..gong_spec()
        };
        let p = VkParams::new(&spec).expect("a gong");
        let params = VkPlateParams::new(
            &p, MASS, STIFFNESS, ALPHA, 0.0, 0.12, 0.16, strike, 0.0, 1e-12, 1e-14, 60, outer_tol,
            20,
        )
        .expect("a mallet");
        let mut m = MalletVkPlate::new(params, VkPlate::new(p), 0.0, strike);
        let (mut hits, mut total, mut stalls, mut worst) = (0usize, 0usize, 0usize, 0usize);
        for _ in 0..2000 {
            m.step().expect("a gong step");
            let l = m.last.as_ref().expect("stepped");
            if l.n_outer > 0 {
                hits += 1;
                total += l.n_outer;
                stalls += usize::from(l.outer_stalled);
                worst = worst.max(l.n_outer);
            }
        }
        (total as f64 / hits as f64, worst, stalls, hits)
    };

    // Four orders of magnitude of the PLATE's tolerance, at a fixed outer ask: measured 2.59,
    // 2.60, 2.59, 2.60. The outer loop does not notice.
    let mut means = Vec::new();
    for couple_tol in [1e-9, 1e-11, 1e-13, 1e-15] {
        let (mean, worst, _, _) = ask(couple_tol, 1e-16);
        means.push(mean);
        assert!(
            worst <= 6,
            "couple_tol={couple_tol:e}: {worst} outer iterations"
        );
    }
    let (lo, hi) = means
        .iter()
        .fold((f64::MAX, 0.0f64), |(a, b), &m| (a.min(m), b.max(m)));
    assert!(
        hi - lo < 0.1,
        "the outer count moved with couple_tol: {means:?} -- the predicted floor would do this"
    );

    // And three decades of the OUTER tolerance, which is what it does answer to. Geometric
    // contraction at a few parts in a thousand buys three decades for under one extra iteration:
    // measured 1.89 at 1e-13 against 2.59 at 1e-16.
    let (loose, loose_worst, loose_stalls, hits) = ask(1e-13, 1e-13);
    let (tight, _, _, _) = ask(1e-13, 1e-16);
    assert!(
        tight > loose,
        "a tighter ask must cost more: {loose:.2} -> {tight:.2}"
    );
    assert!(
        tight - loose < 1.5,
        "three decades cost {:.2} iterations",
        tight - loose
    );
    // The shipped default is the one at which nothing stalls.
    assert_eq!(
        loose_stalls, 0,
        "the shipped `outer_tol` stalls on {loose_stalls}/{hits} steps"
    );
    assert_eq!(loose_worst, 2);
}

#[test]
fn a_step_that_stops_contracting_stops_early_instead_of_spending_the_cap() {
    // One in-contact step in about nine hundred plateaus near 5e-14 and stays there. Without the
    // stagnation exit that step took all twenty of its outer iterations -- twenty full plate
    // solves to discover that the nineteenth was no better than the second.
    let strike = 12.0;
    let p = VkParams::new(&gong_spec()).expect("a gong");
    let params = VkPlateParams::new(
        &p, MASS, STIFFNESS, ALPHA, 0.0, 0.12, 0.16, strike, 0.0, 1e-12, 1e-14, 60, 1e-15, 20,
    )
    .expect("a mallet");
    let mut m = MalletVkPlate::new(params, VkPlate::new(p), 0.0, strike);
    let (mut stalls, mut hits, mut worst_n) = (0usize, 0usize, 0usize);
    for _ in 0..2000 {
        m.step().expect("a gong step");
        let l = m.last.as_ref().expect("stepped");
        if l.n_outer > 0 {
            hits += 1;
            worst_n = worst_n.max(l.n_outer);
            if l.outer_stalled {
                stalls += 1;
                assert!(!l.outer_converged, "a stall is not a convergence");
                assert!(
                    l.outer_residual > 1e-15 && l.outer_residual < 1e-12,
                    "the plateau moved: {:e}",
                    l.outer_residual
                );
            }
        }
    }
    // Measured: exactly one stall in 911 in-contact steps, and a worst count of 4 rather than 20.
    assert!(
        stalls >= 1,
        "this rig is chosen because one step stalls on it"
    );
    assert!(
        (stalls as f64) < 0.01 * hits as f64,
        "{stalls} of {hits} stalled -- that is a floor, not a one-off, and the header says there          is no floor"
    );
    assert!(
        worst_n <= 6,
        "the stagnation exit did not fire: {worst_n} outer iterations"
    );
}
