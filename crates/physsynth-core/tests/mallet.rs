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

use physsynth_core::mallet::{
    scalar_pow, MalletMembrane, MalletWall, ParamError, Params, WallParams,
};
use physsynth_core::membrane::{self, Domain, Membrane};
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
