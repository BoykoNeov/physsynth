//! Native bars for the mallet (plan §17) — model #7's acceptance contract without an interpreter.
//!
//! The Python suite's four criteria are reproduced here in the same order they are argued there:
//! the standalone rig's **closed form** (contact time `pi sqrt(M/K)` and exact velocity reversal),
//! **conservation** at any exponent, the discrete gradient's removable `0/0`, and **passivity**
//! once the felt is hysteretic. Then the coupled model, where the head carries energy away and
//! there is no analytic contact time, so the bar is conservation of the *total*.
//!
//! One test here has no Python counterpart and is the reason this file exists at all: the
//! `** 2` pin. Every admittance in this model is a squaring, and if LLVM folds `powf(x, 2.0)` into
//! `x * x` the port is wrong in a way no physics bar can see — a last-bit force scale conserves
//! energy perfectly and simply is not the Python model any more. It is the one assertion in the
//! file that would fail on a "harmless" cleanup, and the only one that has to be read in a release
//! build to mean anything (plan §17.2).
//!
//! Everything from `the plate model` down is model #7p (2026-09-06) and has no Python counterpart
//! at all — it was never transcribed, so these are its *only* bars outside the Python suite.

use physsynth_core::mallet::{
    scalar_pow, MalletMembrane, MalletPlate, MalletWall, ParamError, Params, PlateParams,
    WallParams,
};
use physsynth_core::membrane::{self, Domain, Membrane};
// Qualified rather than glob-imported: `plate::Params` and `mallet::Params` are two different
// things and this file names both.
use physsynth_core::plate;
use std::f64::consts::PI;

const M: f64 = 0.02; // kg — the project's default mallet
const K: f64 = 5.0e4; // N/m^alpha — the project's default felt
const ETA_TOL: f64 = 1e-12;
const NEWTON_TOL: f64 = 1e-14;
const MAXITER: usize = 60;

fn wall(alpha: f64, lam_h: f64, fs: f64, v0: f64) -> MalletWall {
    let p = WallParams::new(
        M, K, fs, alpha, lam_h, 0.0, 0.0, ETA_TOL, NEWTON_TOL, MAXITER,
    )
    .expect("valid wall rig");
    MalletWall::new(p, 0.0, v0)
}

/// A square drumhead at `lam = 0.5`, which oversamples the stiff felt — the Python helper's
/// default, and the reason the coupled conservation test is not measuring aliasing instead.
fn drumhead(n: i64, sigma: f64) -> Membrane {
    let t: f64 = 100.0;
    let rho: f64 = 0.26;
    let c = (t / rho).sqrt();
    let h = 1.0 / n as f64;
    let fs = c / (0.5 * h);
    Membrane::new(
        membrane::Params::new(
            Some(Domain::Rectangle),
            t,
            rho,
            fs,
            n,
            Some(1.0),
            Some(1.0),
            None,
            sigma,
        )
        .expect("valid drumhead"),
    )
}

fn mallet(n: i64, sigma: f64, alpha: f64, lam_h: f64, v0: f64) -> MalletMembrane {
    let mem = drumhead(n, sigma);
    let p = Params::new(
        mem.params(),
        M,
        K,
        alpha,
        lam_h,
        0.5,
        0.5,
        0.0,
        ETA_TOL,
        NEWTON_TOL,
        MAXITER,
    )
    .expect("valid mallet");
    MalletMembrane::new(p, mem, 0.0, v0)
}

// -- the pin that no physics bar can see ----------------------------------------------------------

#[test]
fn squaring_goes_through_pow_and_not_through_a_multiply() {
    // The claim is STRUCTURAL, not a witness value, and that distinction cost a red CI run one
    // batch ago: `tests/collision.rs` asserted that `pow` and the ufunc ladder's spelling
    // *disagree somewhere*, and the optimiser folded its literal exponent so that both arms became
    // the same code. See that file, and plan §17.2. How often the two spellings differ is in any
    // case a property of the C library rather than of this code; what is testable everywhere is
    // whether this function reaches `pow` at all.
    //
    // `black_box` hides the exponent from LLVM on the right-hand side, which is the whole
    // apparatus: at a visible literal the optimiser rewrites `powf(x, 2.0)` into `x * x`, so if
    // `scalar_pow`'s `#[inline(never)]` were ever dropped the two sides would part company —
    // wherever the libm makes that observable, which is exactly where it matters.
    for i in 1..20_000 {
        let x = 1e-6 * f64::from(i);
        assert_eq!(
            scalar_pow(x, 2.0),
            x.powf(std::hint::black_box(2.0)),
            "scalar_pow(x, 2.0) must be the C library's pow"
        );
    }
    // And the admittance built from it inherits that, which is what actually reaches the physics.
    let m = 1.0 / 0.005419571191933292;
    let p = WallParams::new(m, K, 1.0, 1.0, 0.0, 0.0, 0.0, ETA_TOL, NEWTON_TOL, MAXITER)
        .expect("valid rig");
    assert_eq!(p.g, 1.0_f64.powf(std::hint::black_box(2.0)) / m);
}

// -- criterion 1: the closed form -----------------------------------------------------------------

#[test]
fn the_linear_felt_reproduces_the_half_period_and_reverses_the_velocity() {
    let fs = 96_000.0;
    let v0 = 2.0;
    let mut rig = wall(1.0, 0.0, fs, v0);
    let k = rig.params().k;

    let steps = 700;
    let mut eta = vec![rig.state().penetration];
    for _ in 0..steps {
        rig.step().expect("contact solve converged");
        eta.push(rig.state().penetration);
    }

    // Interpolated zero-crossings of the penetration: entry (-> +) and exit (-> -).
    let up = (0..steps)
        .find(|&i| eta[i] <= 0.0 && eta[i + 1] > 0.0)
        .expect("the mass never made contact");
    let dn = (0..steps)
        .find(|&i| eta[i] > 0.0 && eta[i + 1] <= 0.0)
        .expect("the mass never left");
    let t_in = (up as f64 + eta[up] / (eta[up] - eta[up + 1])) * k;
    let t_out = (dn as f64 + eta[dn] / (eta[dn] - eta[dn + 1])) * k;
    let theory = PI * (M / K).sqrt();
    let rel = ((t_out - t_in) - theory).abs() / theory;
    assert!(
        rel < 5e-3,
        "contact time off by {rel:.2e} (theory {theory:.3e} s)"
    );

    // Restitution exactly 1: the elastic felt returns every joule it took.
    let v_exit = rig.velocity();
    assert!(v_exit > 0.0, "the mass did not rebound");
    assert!(
        (v_exit - v0).abs() / v0 < 1e-9,
        "restitution is not 1: v_exit = {v_exit:.9} against v0 = {v0}"
    );
}

// -- criterion 2: conservation, standalone and coupled ---------------------------------------------

#[test]
fn the_standalone_rig_conserves_at_every_exponent() {
    for &alpha in &[1.0, 2.0, 2.3, 3.0] {
        let mut rig = wall(alpha, 0.0, 96_000.0, 2.5);
        let e0 = rig.energy();
        let mut worst: f64 = 0.0;
        let mut deepest: f64 = 0.0;
        for _ in 0..800 {
            rig.step().expect("contact solve converged");
            worst = worst.max((rig.energy() - e0).abs());
            deepest = deepest.max(rig.state().penetration);
        }
        assert!(
            deepest > 0.0,
            "the mass never made contact (alpha = {alpha})"
        );
        let drift = worst / e0.abs();
        assert!(drift < 1e-11, "energy drift {drift:.2e} at alpha = {alpha}");
    }
}

#[test]
fn the_coupled_model_conserves_the_total_energy() {
    // The head is lossless and the felt elastic, so H = membrane + mallet KE + contact PE is a
    // constant — including across the moment the felt separates and the mallet flies off, which is
    // where a mis-signed force injection would show up as a step rather than a drift.
    let mut mal = mallet(24, 0.0, 2.3, 0.0, 3.0);
    let e0 = mal.energy();
    let mut worst: f64 = 0.0;
    let mut deepest: f64 = 0.0;
    for _ in 0..1_200 {
        mal.step().expect("contact solve converged");
        worst = worst.max((mal.energy() - e0).abs());
        deepest = deepest.max(mal.state().penetration);
    }
    assert!(deepest > 0.0, "the mallet never struck the head");
    let drift = worst / e0.abs();
    assert!(drift < 1e-10, "total energy drift {drift:.2e} (bar 1e-10)");
}

#[test]
fn the_head_actually_takes_the_strike() {
    // A guard against the model conserving perfectly while doing nothing: the drumhead must end up
    // holding a real share of the mallet's kinetic energy, and the mallet must lose it.
    let mut mal = mallet(24, 0.0, 2.3, 0.0, 3.0);
    let ke0 = 0.5 * M * 3.0 * 3.0;
    let mut head_peak: f64 = 0.0;
    for _ in 0..1_200 {
        mal.step().expect("contact solve converged");
        head_peak = head_peak.max(mal.membrane.energy());
    }
    assert!(
        head_peak / ke0 > 0.1,
        "the head took only {:.1}% of the strike",
        100.0 * head_peak / ke0
    );
    assert!(
        mal.mallet_velocity() > 0.0,
        "the mallet never rebounded (velocity {:.3} m/s)",
        mal.mallet_velocity()
    );
}

// -- criterion 3: passivity ------------------------------------------------------------------------

#[test]
fn the_hysteretic_felt_is_strictly_dissipative() {
    let v0 = 2.0;
    let mut rig = wall(1.5, 5.0e3, 96_000.0, v0);
    let mut prev = rig.energy();
    let e0 = prev;
    for _ in 0..900 {
        rig.step().expect("contact solve converged");
        let now = rig.energy();
        assert!(
            now - prev <= 1e-9 * e0,
            "hysteresis added {:.2e} J at step {}",
            now - prev,
            rig.state().n
        );
        prev = now;
    }
    let v_exit = rig.velocity();
    assert!(
        v_exit > 0.0 && v_exit < v0,
        "restitution not in (0, 1): {v_exit:.4} against {v0}"
    );
}

#[test]
fn a_lossy_head_makes_the_coupled_total_monotone() {
    let mut mal = mallet(24, 2.0, 2.3, 0.0, 3.0);
    let mut prev = mal.energy();
    let e0 = prev;
    for _ in 0..600 {
        mal.step().expect("contact solve converged");
        let now = mal.energy();
        assert!(
            now - prev <= 1e-12 * e0,
            "the lossy run gained {:.2e} J",
            now - prev
        );
        prev = now;
    }
    assert!(prev < e0, "a lossy run did not lose anything");
}

// -- the shell's own seams --------------------------------------------------------------------------

#[test]
fn the_strike_snaps_to_a_live_node_and_reports_where_it_landed() {
    let mem = drumhead(24, 0.0);
    let p = Params::new(
        mem.params(),
        M,
        K,
        2.3,
        0.0,
        0.5,
        0.5,
        0.0,
        ETA_TOL,
        NEWTON_TOL,
        MAXITER,
    )
    .expect("valid mallet");
    assert!(p.node < mem.params().n_live());
    // The snapped coordinates are the node's own, not the request's, and on a 24-interval square
    // the centre is a grid point — so they land exactly on it.
    assert!((p.x_strike - 0.5).abs() < mem.params().h);
    assert!((p.y_strike - 0.5).abs() < mem.params().h);
    // The two admittances are separate quantities and their sum is what the residual uses.
    assert_eq!(p.g, p.g_s + p.g_h);
    assert!(p.g_s > 0.0 && p.g_h > 0.0);
}

#[test]
fn a_gap_delays_the_strike_rather_than_moving_it() {
    // With the mallet parked well above the head, the first steps are pure flight: the head stays
    // exactly at rest and the mallet moves by k*v0 each step.
    let mem = drumhead(16, 0.0);
    let k = mem.params().k;
    let v0 = 3.0;
    let gap = 100.0 * k * v0;
    let p = Params::new(
        mem.params(),
        M,
        K,
        2.3,
        0.0,
        0.5,
        0.5,
        gap,
        ETA_TOL,
        NEWTON_TOL,
        MAXITER,
    )
    .expect("valid mallet");
    let mut mal = MalletMembrane::new(p, mem, gap, v0);
    for _ in 0..50 {
        mal.step().expect("contact solve converged");
        assert_eq!(
            mal.state().contact_force,
            0.0,
            "force applied out of contact"
        );
        assert!(!mal.state().in_contact);
    }
    let flown = gap - mal.state().z_h;
    assert!(
        (flown - 50.0 * k * v0).abs() / (50.0 * k * v0) < 1e-12,
        "the force-free flight is not exact: {flown:.6e}"
    );
    assert!(
        mal.membrane.u.iter().all(|&v| v == 0.0),
        "the head moved before contact"
    );
}

#[test]
fn the_two_seedings_are_the_same_motion_at_a_zero_wall() {
    // `at_wall` and `at_strike` are written differently on purpose (see the doc comment). At
    // wall = 0 they must nonetheless describe the same mallet, which is what makes the difference
    // a transcription choice rather than a physics one.
    let a = physsynth_core::mallet::State::at_strike(1e-3, 2.0, 1.0 / 96_000.0, 0.0);
    let b = physsynth_core::mallet::State::at_wall(0.0, 1e-3, 2.0, 1.0 / 96_000.0);
    assert_eq!(a.z_h, b.z_h);
    assert_eq!(a.z_h_prev, b.z_h_prev);
}

// -- refusals ---------------------------------------------------------------------------------------

#[test]
fn the_scalar_checks_fire_in_the_originals_order() {
    let mem = drumhead(8, 0.0);
    let build = |mass, stiffness, alpha, lam_h, gap| {
        Params::new(
            mem.params(),
            mass,
            stiffness,
            alpha,
            lam_h,
            0.5,
            0.5,
            gap,
            ETA_TOL,
            NEWTON_TOL,
            MAXITER,
        )
        .unwrap_err()
    };
    assert_eq!(build(0.0, K, 2.3, 0.0, 0.0), ParamError::NonPositiveMass);
    assert_eq!(
        build(M, 0.0, 2.3, 0.0, 0.0),
        ParamError::NonPositiveStiffness
    );
    assert_eq!(build(M, K, 0.5, 0.0, 0.0), ParamError::AlphaTooSmall);
    assert_eq!(build(M, K, 2.3, -1.0, 0.0), ParamError::NegativeHysteresis);
    assert_eq!(build(M, K, 2.3, 0.0, -1e-3), ParamError::NegativeGap);
    // A call wrong in two ways reports the earlier check, which is what makes the order testable.
    assert_eq!(build(0.0, 0.0, 2.3, 0.0, 0.0), ParamError::NonPositiveMass);

    let wall_err = |mass, stiffness, alpha, lam_h, gap| {
        WallParams::new(
            mass, stiffness, 96_000.0, alpha, lam_h, 0.0, gap, ETA_TOL, NEWTON_TOL, MAXITER,
        )
        .unwrap_err()
    };
    assert_eq!(
        wall_err(-1.0, K, 1.0, 0.0, 0.0),
        ParamError::NonPositiveMass
    );
    assert_eq!(wall_err(M, K, 1.0, 0.0, -1.0), ParamError::NegativeGap);
}

#[test]
fn the_refusal_messages_are_the_pythons() {
    assert_eq!(
        ParamError::NonPositiveMass.to_string(),
        "mallet mass must be > 0."
    );
    assert_eq!(
        ParamError::NonPositiveStiffness.to_string(),
        "felt stiffness K must be > 0."
    );
    assert_eq!(
        ParamError::AlphaTooSmall.to_string(),
        "felt exponent alpha must be >= 1."
    );
    assert_eq!(
        ParamError::NegativeHysteresis.to_string(),
        "hysteresis lambda_h must be >= 0."
    );
    assert_eq!(
        ParamError::NegativeGap.to_string(),
        "initial gap must be >= 0."
    );
}

// == the plate model ==============================================================================
//
// Model #7p's acceptance contract, and the first part of this file with no Python twin to lean on.
// Four things are asserted, in the order the design argues them:
//
// 1. the **superposition identity** — the whole reason a mallet can strike an implicit resonator at
//    all — as a claim about every node and not just the struck one;
// 2. **conservation** of plate + mallet KE + contact PE on both boundary branches;
// 3. **passivity** once the plate is lossy or the felt is hysteretic;
// 4. the **miss**, which is the one place the identity is exact rather than merely tight.

const PLATE_L: f64 = 0.4; // m — square, the plate suite's own geometry
const PLATE_FS: f64 = 20_000.0; // Hz — 39.7 steps per felt half-period, comfortably resolved

fn plate_spec(boundary: plate::Boundary, n: i64, sigma: f64) -> plate::PlateSpec {
    plate::PlateSpec {
        lx: PLATE_L,
        ly: PLATE_L,
        kappa: 1.0,
        rho: 2.0,
        fs: PLATE_FS,
        n,
        sigma,
        boundary: Some(boundary),
        domain: Some(plate::Domain::Rectangle),
        ..plate::PlateSpec::default()
    }
}

fn plate_params(boundary: plate::Boundary, n: i64, sigma: f64) -> plate::Params {
    plate::Params::new(&plate_spec(boundary, n, sigma)).expect("a valid plate")
}

/// A mallet's parameters against `pl`, struck a little off centre so no mode is nulled by accident.
fn plate_mallet_params(pl: &plate::Params, alpha: f64, lam_h: f64) -> PlateParams {
    PlateParams::new(
        pl,
        M,
        K,
        alpha,
        lam_h,
        0.3 * PLATE_L,
        0.4 * PLATE_L,
        0.0,
        ETA_TOL,
        NEWTON_TOL,
        MAXITER,
    )
    .expect("valid plate mallet")
}

fn struck_plate(
    boundary: plate::Boundary,
    n: i64,
    sigma: f64,
    alpha: f64,
    lam_h: f64,
    v0: f64,
) -> MalletPlate {
    let pl = plate_params(boundary, n, sigma);
    let p = plate_mallet_params(&pl, alpha, lam_h);
    MalletPlate::new(p, plate::Plate::new(pl), 0.0, v0)
}

/// A centred Gaussian bump over the live nodes — a generic non-rest state to test an identity from.
fn plate_bump(p: &plate::Params, amp: f64) -> Vec<f64> {
    let (cx, cy) = (0.5 * p.lx, 0.5 * p.ly);
    let s = 0.15 * p.lx;
    p.mask
        .flags()
        .iter()
        .enumerate()
        .filter(|(_, &alive)| alive)
        .map(|(idx, _)| {
            let (dx, dy) = (p.x[idx] - cx, p.y[idx] - cy);
            amp * (-(dx * dx + dy * dy) / (s * s)).exp()
        })
        .collect()
}

// -- the identity the whole model rests on --------------------------------------------------------

#[test]
fn the_influence_column_reproduces_a_forced_step_at_every_node() {
    // The claim: stepping force-free and then adding the influence column is the same plate as
    // stepping with the force in the right-hand side. It is asserted over the WHOLE field, because
    // a column that is right at the strike node and wrong elsewhere would satisfy the drive-point
    // check, pass every energy bar (the wrong field would simply be a different, self-consistent
    // trajectory) and sound wrong.
    //
    // Not `assert_eq!`: a sparse LU back-substitution is not a linear map over doubles, so
    // `solve(rhs) + c*solve(e)` and `solve(rhs + c*e)` round differently. The identity is exact in
    // exact arithmetic and tight-but-not-bitwise in this one, and the tolerance below says which.
    for boundary in [plate::Boundary::Supported, plate::Boundary::Free] {
        let pl = plate_params(boundary, 14, 0.0);
        let p = plate_mallet_params(&pl, 2.3, 0.0);
        let force = 7.5; // N — an arbitrary nonzero, sign included by the injection convention

        // A generic state: a bump, advanced a few steps so `u` and `u_prev` genuinely differ.
        let mut forced = plate::Plate::new(pl.clone());
        forced.set_state(&plate_bump(&pl, 1e-3), &vec![0.0; pl.n_live]);
        for _ in 0..5 {
            forced.step(None);
        }
        let mut split = forced.clone();

        let mut f_ext = vec![0.0; pl.n_live];
        f_ext[p.node] = -force;
        forced.step(Some(&f_ext));

        split.step(None);
        physsynth_core::mallet::plate_inject(&mut split.u, &mut split.accel, &p, force);

        let scale = forced.u.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        assert!(scale > 0.0, "the reference field is identically zero");
        for i in 0..pl.n_live {
            assert!(
                (forced.u[i] - split.u[i]).abs() <= 1e-14 * scale,
                "{boundary:?}: node {i} disagrees by {:.3e} (field scale {scale:.3e})",
                forced.u[i] - split.u[i]
            );
            // The acceleration is `(u^{n+1} - 2u^n + u^{n-1}) / k^2`, and only the first term
            // moved — so a displacement error `e` arrives here as `e / k^2`, while the
            // acceleration's own scale is set by the *second* difference, which at audio
            // frequencies is a factor `(omega k)^2` smaller than the displacement. Normalising by
            // `max|accel|` would therefore demand that the accelerations agree three orders better
            // than the displacements they are built from. The claim made instead is the honest
            // one: multiplied back by `k^2`, these came from displacements agreeing to the same
            // `1e-14`.
            assert!(
                (forced.accel[i] - split.accel[i]).abs() * pl.k * pl.k <= 1e-14 * scale,
                "{boundary:?}: accel at node {i} disagrees by {:.3e} m/s^2",
                forced.accel[i] - split.accel[i]
            );
        }
    }
}

#[test]
fn the_driving_point_admittance_is_the_columns_own_entry() {
    // `g_s` is READ from the column rather than computed a second way, so this is a structural
    // assertion and can be exact. The measurement beside it is the one that could fail: a unit
    // force must move the struck node by `g_s` metres, one step later, from rest.
    for boundary in [plate::Boundary::Supported, plate::Boundary::Free] {
        let pl = plate_params(boundary, 14, 0.0);
        let p = plate_mallet_params(&pl, 2.3, 0.0);
        assert_eq!(p.g_s, p.influence[p.node]);
        assert!(
            p.g_s > 0.0,
            "an SPD plate cannot have a negative admittance"
        );

        let mut pl_obj = plate::Plate::new(pl.clone());
        let mut f_ext = vec![0.0; pl.n_live];
        f_ext[p.node] = 1.0;
        pl_obj.step(Some(&f_ext));
        assert!(
            (pl_obj.u[p.node] - p.g_s).abs() <= 1e-14 * p.g_s,
            "{boundary:?}: one newton moved the strike node {:.6e}, admittance says {:.6e}",
            pl_obj.u[p.node],
            p.g_s
        );
    }
}

#[test]
fn the_strike_snaps_to_a_live_node_and_reports_where() {
    // The live/full-grid mixup this model's `live_coords` exists to prevent: `pickup_index_at`
    // counts live nodes, `Params::x` is indexed by the full grid. A confusion between them reports
    // a plausible-looking point, so the assertion is that the reported point is the one the plate
    // itself would report for that live index.
    let pl = plate_params(plate::Boundary::Supported, 14, 0.0);
    let p = plate_mallet_params(&pl, 2.3, 0.0);
    assert_eq!(
        p.node,
        physsynth_core::plate::pickup_index_at(p.x_strike, p.y_strike, &pl)
    );
    let h = PLATE_L / 14.0;
    assert!(
        (p.x_strike - 0.3 * PLATE_L).abs() <= h,
        "x snapped further than one cell"
    );
    assert!(
        (p.y_strike - 0.4 * PLATE_L).abs() <= h,
        "y snapped further than one cell"
    );
}

// -- conservation ---------------------------------------------------------------------------------

#[test]
fn a_struck_lossless_plate_conserves_the_total() {
    // The money test, and the reason the plate needed the discrete-gradient force rather than
    // `phi'` at a point: plate + mallet kinetic + averaged contact potential is a constant of the
    // scheme, on both branches, at any felt exponent.
    for boundary in [plate::Boundary::Supported, plate::Boundary::Free] {
        for alpha in [1.0, 2.3] {
            let mut m = struck_plate(boundary, 12, 0.0, alpha, 0.0, 3.0);
            let e0 = m.energy();
            let mut worst = 0.0f64;
            for _ in 0..1500 {
                m.step().expect("the contact solve converges");
                worst = worst.max((m.energy() - e0).abs() / e0.abs());
            }
            assert!(
                worst < 1e-10,
                "{boundary:?} alpha={alpha}: relative drift {worst:.3e} over 1500 steps"
            );
            assert!(m.state().n == 1500);
        }
    }
}

#[test]
fn the_felt_actually_engages() {
    // Without this, every conservation bar above is satisfied by a mallet that sailed past: it
    // conserves perfectly because nothing happened. The strike must store contact potential and the
    // mallet must leave slower than it arrived in one direction and reversed in the other.
    let mut m = struck_plate(plate::Boundary::Supported, 12, 0.0, 2.3, 0.0, 3.0);
    let mut peak_force = 0.0f64;
    let mut contacts = 0usize;
    for _ in 0..1500 {
        m.step().expect("the contact solve converges");
        peak_force = peak_force.max(m.state().contact_force.abs());
        contacts += usize::from(m.state().in_contact);
    }
    assert!(
        peak_force > 1.0,
        "peak contact force was only {peak_force:.3e} N"
    );
    assert!(
        contacts > 10,
        "the mallet was in contact for {contacts} steps"
    );
    assert!(
        m.mallet_velocity() > 0.0,
        "the mallet did not rebound: velocity {:.3e} m/s",
        m.mallet_velocity()
    );
}

// -- passivity -------------------------------------------------------------------------------------

#[test]
fn a_lossy_plate_or_a_hysteretic_felt_only_ever_loses_energy() {
    // `lam_h = 5e3` is the Python suite's hysteretic felt. A smaller one is still passive and
    // still monotone, but it removes so little (8 removes 0.03% over this window) that the closing
    // assertion would be measuring the tolerance rather than the loss.
    for (sigma, lam_h) in [(2.0, 0.0), (0.0, 5.0e3), (2.0, 5.0e3)] {
        let mut m = struck_plate(plate::Boundary::Supported, 12, sigma, 2.3, lam_h, 3.0);
        let e0 = m.energy();
        let mut prev = e0;
        for step in 0..1200 {
            m.step().expect("the contact solve converges");
            let now = m.energy();
            assert!(
                now <= prev + 1e-12 * prev.abs().max(1.0),
                "sigma={sigma} lam_h={lam_h}: energy rose at step {step} ({prev:.9e} -> {now:.9e})"
            );
            prev = now;
        }
        assert!(
            prev < 0.99 * e0,
            "sigma={sigma} lam_h={lam_h}: the run lost nothing measurable ({e0:.6e} -> {prev:.6e})"
        );
    }
}

// -- the miss ---------------------------------------------------------------------------------------

#[test]
fn a_mallet_that_never_touches_leaves_the_plate_bit_identical() {
    // `f == 0.0` makes every increment a signed zero and `x - (+-0.0) == x` for finite `x`, so the
    // superposition correction is EXACTLY a no-op when there is no contact — which is what lets a
    // plate carrying a mallet it never meets be compared to a bare one with `==` rather than a
    // tolerance. This is the `K = 0` analog the membrane model asserts, and it is also the guard
    // that the injection never touches a node it should not.
    let pl = plate_params(plate::Boundary::Supported, 12, 0.0);
    let seed = plate_bump(&pl, 1e-3);
    let zeros = vec![0.0; pl.n_live];

    let mut bare = plate::Plate::new(pl.clone());
    bare.set_state(&seed, &zeros);

    let p = PlateParams::new(
        &pl,
        M,
        K,
        2.3,
        0.0,
        0.3 * PLATE_L,
        0.4 * PLATE_L,
        1.0,
        ETA_TOL,
        NEWTON_TOL,
        MAXITER,
    )
    .expect("valid plate mallet");
    let mut held = MalletPlate::new(p, plate::Plate::new(pl.clone()), 1.0, -1.0);
    held.plate.set_state(&seed, &zeros);

    for _ in 0..400 {
        bare.step(None);
        held.step().expect("no contact to solve");
    }
    assert!(
        !held.state().in_contact,
        "the mallet was supposed to fly away"
    );
    assert_eq!(
        bare.u, held.plate.u,
        "the untouched plate must be bit-identical"
    );
    assert_eq!(bare.accel, held.plate.accel, "and so must its acceleration");
}

// -- the free plate recoils --------------------------------------------------------------------------

#[test]
fn a_struck_free_plate_carries_away_net_momentum() {
    // Physics, asserted here so a reader watching a struck cymbal drift does not file it as a bug.
    //
    // The claim is exact, not qualitative. On the free branch `K 1 = 0`, so projecting the step
    // onto `1^T W` kills the stiffness term entirely and leaves `m^{n+1} = 2 m^n - m^{n-1}` for
    // `m = 1^T W u` — the mass-weighted mean is a **linear function of the step index** once the
    // contact has ended, exactly. The supported plate has `1^T B != 0` and its mean therefore
    // oscillates, which is what the boundary holding the plate means numerically.
    let weighted_mean = |m: &MalletPlate| -> f64 {
        let pl = &m.plate.p;
        let w: Vec<f64> = if pl.w.is_empty() {
            vec![1.0; pl.n_live]
        } else {
            pl.w.clone()
        };
        let total: f64 = w.iter().sum();
        (0..pl.n_live).map(|i| w[i] * m.plate.u[i]).sum::<f64>() / total
    };

    let mut free = struck_plate(plate::Boundary::Free, 12, 0.0, 2.3, 0.0, 3.0);
    let mut supported = struck_plate(plate::Boundary::Supported, 12, 0.0, 2.3, 0.0, 3.0);
    let (mut free_marks, mut sup_marks) = (Vec::new(), Vec::new());
    for step in 1..=2000 {
        free.step().expect("the contact solve converges");
        supported.step().expect("the contact solve converges");
        // Three marks, evenly spaced and all well after the ~40-step contact.
        if [1000, 1500, 2000].contains(&step) {
            free_marks.push(weighted_mean(&free));
            sup_marks.push(weighted_mean(&supported));
        }
    }

    // Equal spacing in time, so a constant drift means equal spacing in displacement.
    let free_first = free_marks[1] - free_marks[0];
    let free_second = free_marks[2] - free_marks[1];
    assert!(
        free_first.abs() > 1e-6,
        "the free plate took no net momentum: mean moved {free_first:.3e} m in 500 steps"
    );
    assert!(
        (free_second - free_first).abs() <= 1e-9 * free_first.abs(),
        "the free plate's drift is not constant: {free_first:.9e} then {free_second:.9e}"
    );

    // The supported plate's mean is an oscillation, so its two secants do not agree at all.
    let (sup_first, sup_second) = (sup_marks[1] - sup_marks[0], sup_marks[2] - sup_marks[1]);
    assert!(
        (sup_second - sup_first).abs() > 1e-3 * sup_first.abs().max(sup_second.abs()),
        "a supported plate has no rigid mode to drift along: {sup_first:.9e}, {sup_second:.9e}"
    );
}

// -- refusals ----------------------------------------------------------------------------------------

#[test]
fn the_plate_mallet_refuses_what_the_membrane_mallet_refuses() {
    let pl = plate_params(plate::Boundary::Supported, 10, 0.0);
    let build = |mass, stiffness, alpha, lam_h, gap| {
        PlateParams::new(
            &pl, mass, stiffness, alpha, lam_h, 0.2, 0.2, gap, ETA_TOL, NEWTON_TOL, MAXITER,
        )
        .unwrap_err()
    };
    assert_eq!(build(0.0, K, 2.3, 0.0, 0.0), ParamError::NonPositiveMass);
    assert_eq!(
        build(M, 0.0, 2.3, 0.0, 0.0),
        ParamError::NonPositiveStiffness
    );
    assert_eq!(build(M, K, 0.5, 0.0, 0.0), ParamError::AlphaTooSmall);
    assert_eq!(build(M, K, 2.3, -1.0, 0.0), ParamError::NegativeHysteresis);
    assert_eq!(build(M, K, 2.3, 0.0, -1e-3), ParamError::NegativeGap);
    // Same order as the other two models: a call wrong in two ways reports the earlier check.
    assert_eq!(build(0.0, 0.0, 2.3, 0.0, 0.0), ParamError::NonPositiveMass);
}
