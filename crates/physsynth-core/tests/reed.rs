//! Native acceptance bars for `reed` — the same claims `tests/test_reed_*.py` make about the
//! Python original, asserted here against the Rust one without a Python interpreter in the way.
//!
//! The reed is an **active** exciter: the mouth does work, so the contract is not conservation but
//! an exact **balance**,
//!
//! ```text
//! E^n - E^0 = mouth_work - jet_loss - reed_damp_work
//! ```
//!
//! with every dissipation channel sign-definite. That is the money test and it is the first bar
//! below.
//!
//! **But balance alone is not sufficient**, and the project already learned why: a dead reed — one
//! with the coupling sign wrong — rings down quietly and balances its books perfectly the whole
//! way. So the independent oracle is the *signature*: self-sustained oscillation above a blowing
//! threshold, locked to the bore, and failure to speak below it. Both are here.
//!
//! Two bars have no counterpart on any earlier model in this crate:
//!
//! - `the_bracketed_fallback_actually_fires` — because the whole reason `crate::root` exists is
//!   that it does, and a test suite that never took the branch would leave it unasserted.
//! - `the_stall_test_is_nan_true` — the original spells it `if not (abs(r_new) < abs(r))`, and the
//!   inverted spelling `>=` differs exactly when a residual is NaN.

use physsynth_core::bore::{self, Bore, End};
use physsynth_core::reed::{self, ParamError, Params, ReedBore, State};

const RHO0: f64 = 1.2041;
const C0: f64 = 343.0;
const L: f64 = 0.6;

/// A clarinet air column at Courant number `lam`, as `tests/helpers.make_reed_bore` builds it.
fn bore_params(n: usize, lam: f64, bc: (End, End), r_bell: f64, sigma: f64) -> bore::Params {
    let h = L / (n as f64);
    bore::Params::new(
        L,
        C0 / (lam * h),
        n,
        0.008,
        Some(bc),
        sigma,
        r_bell,
        RHO0,
        C0,
    )
    .expect("bore parameters should be accepted")
}

/// The default clarinet reed: ~2.5 kHz, heavily lip-damped, `p_closing ~ 3 kPa`.
fn reed_params(bp: &bore::Params) -> Params {
    Params::new(bp, 2500.0, 4.0, 0.03, 1.5e-4, 1.5e-2, 4.0e-4, 1e-10, 60)
        .expect("reed parameters should be accepted")
}

/// A whole clarinet, blown at `p_mouth`.
fn clarinet(n: usize, lam: f64, bc: (End, End), r_bell: f64, p_mouth: f64) -> ReedBore {
    let bp = bore_params(n, lam, bc, r_bell, 0.0);
    let rp = reed_params(&bp);
    ReedBore::new(rp, Bore::new(bp), p_mouth)
}

// -- construction ------------------------------------------------------------------------------

#[test]
fn non_physical_parameters_are_rejected_in_the_originals_order() {
    let bp = bore_params(60, 1.0, (End::Closed, End::Open), 0.0, 0.0);
    let build = |f_reed, q_reed, mu, sr, width, h0, maxiter| {
        Params::new(&bp, f_reed, q_reed, mu, sr, width, h0, 1e-10, maxiter)
    };
    for bad in [
        build(0.0, 4.0, 0.03, 1.5e-4, 1.5e-2, 4.0e-4, 60),
        build(2500.0, 0.0, 0.03, 1.5e-4, 1.5e-2, 4.0e-4, 60),
        build(2500.0, 4.0, 0.0, 1.5e-4, 1.5e-2, 4.0e-4, 60),
        build(2500.0, 4.0, 0.03, 0.0, 1.5e-2, 4.0e-4, 60),
        build(2500.0, 4.0, 0.03, 1.5e-4, 0.0, 4.0e-4, 60),
        build(2500.0, 4.0, 0.03, 1.5e-4, 1.5e-2, 0.0, 60),
    ] {
        assert_eq!(bad.unwrap_err(), ParamError::NonPositiveScalar);
    }
    assert_eq!(
        build(2500.0, 4.0, 0.03, 1.5e-4, 1.5e-2, 4.0e-4, 0).unwrap_err(),
        ParamError::BadMaxIter
    );
    // A non-physical scalar wins over a bad iteration count — the original's order.
    assert_eq!(
        build(0.0, 4.0, 0.03, 1.5e-4, 1.5e-2, 4.0e-4, 0).unwrap_err(),
        ParamError::NonPositiveScalar
    );
}

#[test]
fn the_reed_refuses_a_bore_whose_mouthpiece_end_is_not_closed() {
    for (bc, name) in [
        ((End::Open, End::Open), "open"),
        ((End::Radiating, End::Open), "radiating"),
    ] {
        let bp = bore_params(60, 1.0, bc, 650.0, 0.0);
        let err =
            Params::new(&bp, 2500.0, 4.0, 0.03, 1.5e-4, 1.5e-2, 4.0e-4, 1e-10, 60).unwrap_err();
        assert_eq!(err, ParamError::MouthpieceNotClosed(name));
        assert!(err.to_string().contains("must be 'closed'"));
        assert!(err.to_string().contains(&format!("got '{name}'")));
    }
}

#[test]
fn a_reed_too_stiff_for_the_timestep_is_rejected() {
    // `wr k >= 2` — a very low sample rate (coarse grid) with a high reed resonance.
    let bp = bore_params(4, 1.0, (End::Closed, End::Open), 0.0, 0.0);
    let err = Params::new(&bp, 2500.0, 4.0, 0.03, 1.5e-4, 1.5e-2, 4.0e-4, 1e-10, 60).unwrap_err();
    assert!(matches!(err, ParamError::CflViolated(_)));
    assert!(err.to_string().starts_with("reed CFL violated: wr*k = "));
    assert!(err.to_string().contains(">= 2 (reed too stiff"));
}

#[test]
fn the_node_zero_compliance_is_not_the_bores_own_and_that_is_deliberate() {
    // The plan's §12.8 finding, asserted so a later "tidy-up" fails loudly. The bore spells the
    // compliance `rho0 * c0**2`; the reed spells it `rho0 * c0 * c0`, from the bore's *public*
    // geometry. They differ by one ulp on most tube/grid combinations, and the reed's number is
    // the one the acceptance runs were taken with.
    let mut differed = 0;
    let mut compared = 0;
    for n in [17usize, 31, 40, 60, 97, 128, 200] {
        for lam in [1.0, 0.93, 0.87, 0.71] {
            let bp = bore_params(n, lam, (End::Closed, End::Open), 0.0, 0.0);
            let rp = reed_params(&bp);
            compared += 1;
            if rp.p_pref0 != bp.p_pref[0] {
                differed += 1;
                let rel = (rp.p_pref0 - bp.p_pref[0]).abs() / bp.p_pref[0].abs();
                assert!(
                    rel < 1e-15,
                    "the divergence is one ulp, not a bug: {rel:.3e}"
                );
            }
        }
    }
    assert!(
        differed > 0,
        "the two spellings agreed everywhere — this test is no longer watching anything"
    );
    assert!(compared > differed || differed == compared);
}

// -- the Bernoulli jet ---------------------------------------------------------------------------

#[test]
fn the_jet_is_a_passive_resistor_and_a_shut_reed_passes_no_air() {
    for &dp in &[-5000.0, -1.0, -1e-12, 0.0, 1e-12, 1.0, 5000.0] {
        let u = reed::bernoulli_flow(dp, 4.0e-4, 1.5e-2, RHO0);
        assert!(
            dp * u >= 0.0,
            "the jet must dissipate, not supply, at dp = {dp}"
        );
    }
    for &opening in &[0.0, -1e-9, -1.0] {
        assert_eq!(reed::bernoulli_flow(3000.0, opening, 1.5e-2, RHO0), 0.0);
    }
}

#[test]
fn the_jet_is_odd_in_the_pressure_drop() {
    for &dp in &[1e-9, 1.0, 137.5, 5000.0] {
        let plus = reed::bernoulli_flow(dp, 4.0e-4, 1.5e-2, RHO0);
        let minus = reed::bernoulli_flow(-dp, 4.0e-4, 1.5e-2, RHO0);
        assert_eq!(plus, -minus);
    }
}

// -- the scalar solve ----------------------------------------------------------------------------

#[test]
fn the_residual_is_strictly_decreasing_so_the_root_is_unique() {
    let bp = bore_params(60, 1.0, (End::Closed, End::Open), 0.0, 0.0);
    let rp = reed_params(&bp);
    let opening = 4.0e-4;
    let mut previous = f64::INFINITY;
    let mut dp = -8000.0;
    while dp <= 8000.0 {
        let r = reed::residual(dp, opening, 12.0, 1500.0, &rp);
        assert!(r < previous, "residual rose at dp = {dp}");
        previous = r;
        dp += 25.0;
    }
}

#[test]
fn the_bracketed_root_and_newton_find_the_same_root() {
    let bp = bore_params(60, 1.0, (End::Closed, End::Open), 0.0, 0.0);
    let rp = reed_params(&bp);
    let opening = 4.0e-4;
    for &c_const in &[-500.0, 0.0, 12.0, 5000.0] {
        let (newton, used) = reed::solve_dp(opening, c_const, 1500.0, 1500.0, 0, &rp).unwrap();
        let bracketed = reed::bracketed_root(opening, c_const, 1500.0, 0, &rp).unwrap();
        let r_newton = reed::residual(newton, opening, c_const, 1500.0, &rp);
        let r_brack = reed::residual(bracketed, opening, c_const, 1500.0, &rp);
        assert!(
            r_newton.abs() <= rp.newton_tol,
            "Newton did not converge: {r_newton:.3e}"
        );
        assert!(
            r_brack.abs() <= 1e-8,
            "Brent did not converge: {r_brack:.3e}"
        );
        // Both land on the same root to well inside the tolerance, whichever branch was used.
        assert!(
            (newton - bracketed).abs() <= 1e-9 * newton.abs().max(1.0),
            "used={used}"
        );
    }
}

#[test]
fn the_bracketed_fallback_actually_fires() {
    // The reason `crate::root` exists. Measured on the Python original: 4-5 fallbacks per 4,000
    // steps at the flagship blowing pressure, and hundreds on a coarse grid. If this ever reads
    // zero, `brentq` has stopped being exercised and the parity claim about it is unasserted.
    let mut coarse = clarinet(40, 1.0, (End::Closed, End::Radiating), 650.0, 1500.0);
    for _ in 0..2000 {
        coarse
            .step()
            .expect("the monotone residual always brackets");
    }
    assert!(
        coarse.state().fallbacks > 0,
        "the Newton path never stalled — the Brent fallback is untested"
    );
}

#[test]
fn the_stall_test_is_nan_true() {
    // `if not (abs(r_new) < abs(r))` is TRUE for a NaN residual; the inverted spelling `>=` is
    // FALSE, and would accept a NaN Newton step and keep iterating on it. This pins the spelling
    // by driving the solver with a NaN `c_const`, which makes every residual NaN: the loop must
    // break out to the bracket on the first iteration rather than spinning.
    let bp = bore_params(60, 1.0, (End::Closed, End::Open), 0.0, 0.0);
    let rp = reed_params(&bp);
    // The bracket cannot succeed on NaN either, so the call must *refuse* rather than hang or
    // silently return a NaN root.
    let outcome = reed::solve_dp(4.0e-4, f64::NAN, 1500.0, 1500.0, 7, &rp);
    let err = outcome.expect_err("a NaN residual has no root and must be refused");
    assert_eq!(err.step, 7);
    assert!(err.to_string().contains("failed to bracket at step 7"));
}

// -- the energy balance (the money test) ---------------------------------------------------------

#[test]
fn the_lossless_balance_is_exact() {
    // `E^n - E^0 == mouth_work - jet_loss - reed_damp_work`, to machine precision, on a lossless
    // bore. This is what the two-field coupling being *centered* buys, and it is exact only up to
    // the scalar residual — so it verifies the solve converged as well as that the physics is
    // right (the per-step error is linear in that residual).
    for &p_mouth in &[1000.0, 1500.0, 1800.0] {
        let mut c = clarinet(100, 1.0, (End::Closed, End::Open), 0.0, p_mouth);
        let e0 = c.energy();
        for _ in 0..3000 {
            c.step().unwrap();
        }
        let scale = c
            .state()
            .mouth_work
            .abs()
            .max(c.state().jet_loss.abs())
            .max((c.energy() - e0).abs())
            .max(1e-30);
        let rel = c.balance_error(e0).abs() / scale;
        assert!(
            rel < 1e-11,
            "balance error {rel:.3e} at p_mouth = {p_mouth}"
        );
    }
}

#[test]
fn the_balance_holds_with_a_radiating_bell() {
    // `energy()` already carries the bore's `radiated_energy`, so the bell needs no extra term.
    let mut c = clarinet(100, 1.0, (End::Closed, End::Radiating), 650.0, 1500.0);
    let e0 = c.energy();
    for _ in 0..3000 {
        c.step().unwrap();
    }
    let scale = c.state().mouth_work.abs().max(1e-30);
    assert!(c.balance_error(e0).abs() / scale < 1e-11);
    assert!(
        c.bore().radiated_energy() > 0.0,
        "the bell should have sounded"
    );
}

#[test]
fn every_dissipation_channel_is_sign_definite() {
    let mut c = clarinet(100, 1.0, (End::Closed, End::Radiating), 650.0, 1500.0);
    let mut jet = 0.0;
    let mut damp = 0.0;
    for _ in 0..2000 {
        c.step().unwrap();
        let s = c.state();
        assert!(s.jet_loss >= jet, "the Bernoulli jet supplied energy");
        assert!(s.reed_damp_work >= damp, "the reed damping supplied energy");
        jet = s.jet_loss;
        damp = s.reed_damp_work;
    }
    assert!(jet > 0.0 && damp > 0.0, "neither channel did anything");
}

// -- the signature (the independent oracle) -------------------------------------------------------

/// Peak-to-peak mouthpiece pressure over the last `tail` steps of a run — the "did it speak" read.
fn speaks(c: &mut ReedBore, steps: usize, tail: usize) -> f64 {
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for i in 0..steps {
        c.step().unwrap();
        if i >= steps - tail {
            let p = c.mouthpiece_pressure();
            lo = lo.min(p);
            hi = hi.max(p);
        }
    }
    hi - lo
}

#[test]
fn the_note_speaks_above_threshold_and_fails_below_it() {
    // The bar that a dead (wrong-sign) reed cannot pass, which is why balance alone is not enough:
    // a reed that merely rings down balances its books perfectly the whole way.
    let mut quiet = clarinet(100, 1.0, (End::Closed, End::Radiating), 650.0, 300.0);
    let mut loud = clarinet(100, 1.0, (End::Closed, End::Radiating), 650.0, 1500.0);
    let quiet_pp = speaks(&mut quiet, 6000, 2000);
    let loud_pp = speaks(&mut loud, 6000, 2000);

    assert!(
        loud_pp > 100.0,
        "the note did not speak above threshold: {loud_pp:.3} Pa peak-to-peak"
    );
    assert!(
        quiet_pp < 0.01 * loud_pp,
        "the note spoke below threshold: {quiet_pp:.3} vs {loud_pp:.3} Pa"
    );
}

#[test]
fn blowing_harder_gives_a_louder_note() {
    let mut soft = clarinet(100, 1.0, (End::Closed, End::Radiating), 650.0, 1100.0);
    let mut hard = clarinet(100, 1.0, (End::Closed, End::Radiating), 650.0, 1800.0);
    assert!(speaks(&mut hard, 6000, 2000) > speaks(&mut soft, 6000, 2000));
}

#[test]
fn a_silent_mouth_leaves_the_reed_at_rest() {
    // `p_mouth = 0` is not a special case anywhere in the code, so this is a real test of the
    // whole chain rather than of a guard: no breath, no flow, no motion, nothing booked.
    let mut c = clarinet(60, 1.0, (End::Closed, End::Open), 0.0, 0.0);
    for _ in 0..500 {
        c.step().unwrap();
    }
    let s = c.state();
    assert_eq!(s.y, 0.0);
    assert_eq!(s.y_prev, 0.0);
    assert_eq!(s.dp, 0.0);
    assert_eq!(s.flow, 0.0);
    assert_eq!(s.mouth_work, 0.0);
    assert_eq!(s.jet_loss, 0.0);
    assert_eq!(s.reed_damp_work, 0.0);
    assert_eq!(c.bore().energy(), 0.0);
}

// -- the reed's own state --------------------------------------------------------------------------

#[test]
fn the_opening_clamps_shut_and_the_gamma_scale_is_the_closing_pressure() {
    let bp = bore_params(60, 1.0, (End::Closed, End::Open), 0.0, 0.0);
    let rp = reed_params(&bp);
    let mut s = State::at_rest(rp.p_closing);
    assert_eq!(s.reed_opening(&rp), rp.h0);
    assert_eq!(s.gamma(&rp), 1.0);

    s.y = -rp.h0;
    assert_eq!(s.reed_opening(&rp), 0.0);
    s.y = -2.0 * rp.h0;
    assert_eq!(
        s.reed_opening(&rp),
        0.0,
        "the clamp is max(H0 + y, 0), not |H0 + y|"
    );
    s.y = 0.5 * rp.h0;
    assert_eq!(s.reed_opening(&rp), rp.h0 + 0.5 * rp.h0);
}

#[test]
fn the_reed_energy_is_positive_definite_below_its_cfl() {
    // The cross-time potential `1/2 Mr wr^2 y^n y^{n-1}` is positive-definite only while
    // `wr k < 2` — the same two-time-level trick, and the same condition, as the string's.
    let mut c = clarinet(100, 1.0, (End::Closed, End::Radiating), 650.0, 1500.0);
    for _ in 0..4000 {
        c.step().unwrap();
        assert!(
            c.state().reed_energy(c.params()) >= 0.0,
            "the reed's stored energy went negative"
        );
    }
}

#[test]
fn the_mouth_pressure_can_be_changed_between_steps() {
    // The original documents this as how an attack is played, so it is interface and not accident.
    let mut c = clarinet(100, 1.0, (End::Closed, End::Radiating), 650.0, 0.0);
    for _ in 0..1500 {
        c.step().unwrap();
    }
    assert_eq!(c.state().mouth_work, 0.0, "a silent mouth did work");
    c.state_mut().p_mouth = 1500.0;
    assert_eq!(c.state().gamma(c.params()), 1500.0 / c.params().p_closing);
    for _ in 0..4000 {
        c.step().unwrap();
    }
    assert!(
        c.state().mouth_work > 0.0,
        "the breath did no work after the attack"
    );
}
