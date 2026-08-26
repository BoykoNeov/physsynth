//! Native acceptance bars for `radiation` — the air node's three tiers, asserted without a Python
//! interpreter in the way.
//!
//! The project's contract (`CLAUDE.md`) is closed-form physics, not agreement with a reference
//! implementation, so these are the same claims `tests/test_radiation.py` makes: the monopole gain
//! and its exact inverse-distance law, an integer-sample delay that preserves amplitude, and — the
//! money bar for both loaded bodies — the energy identity that says the air's books balance.
//!
//! Three of them exist because a *port* can break them while every other bar stays green:
//!
//! * **`R = 0` is bit-identical to a bare body, and `M_a = inf` is bit-identical to the
//!   constant-`R` load.** Both are exact reductions in the original, both survive only if the two
//!   loaded bodies share one copy of the rank-1 precomputes and the operation order inside
//!   `solve` is left alone. They compare Rust against Rust, so they stay meaningful after the
//!   Python side is deleted.
//! * **The delay line's length is round-half-to-even.** `2.5 -> 2`, not 3.
//! * **`stored_energy` is zero, not NaN, for the constant-`R` load** — the `inf * 0` the original
//!   special-cases.

use physsynth_core::body::{ModalBody, Params as BodyParams};
use physsynth_core::radiation::{
    self, AirError, AirParams, AirRadiation, LoadError, LoadParams, LoadedModeError, RadiatedBody,
    RationalAirLoad, ReactiveRadiatedBody, C0_AIR, RHO0_AIR,
};

const FS: f64 = 48000.0;
/// The four off-harmonic modes `tests/helpers.py` uses for a guitar-ish body.
const FREQS: [f64; 4] = [110.0, 196.0, 261.0, 440.0];

fn body(sigma: f64) -> ModalBody {
    let n = FREQS.len();
    ModalBody::new(
        BodyParams::new(
            FREQS.to_vec(),
            FS,
            vec![sigma; n],
            vec![1.0; n],
            vec![1.0; n],
            None,
        )
        .expect("the reference body is well posed"),
    )
}

fn air(distance: f64, retarded: bool) -> AirRadiation {
    AirRadiation::new(
        AirParams::new(FS, distance, RHO0_AIR, C0_AIR, retarded).expect("well-posed air"),
    )
}

fn plucked(sigma: f64) -> Vec<f64> {
    let _ = sigma;
    vec![1e-3, -8e-4, 6e-4, 4e-4]
}

// -- tier 1: the read-out ------------------------------------------------------------------------

#[test]
fn the_monopole_gain_is_the_free_space_greens_function() {
    let r = 2.5;
    let a = air(r, false);
    let expect = RHO0_AIR / (4.0 * std::f64::consts::PI * r);
    assert!((a.params().gain - expect).abs() <= 1e-18);
}

#[test]
fn pressure_falls_off_exactly_as_one_over_r() {
    let q = 3.25;
    let p1 = air(1.0, false).process(q);
    let p2 = air(2.0, false).process(q);
    let p4 = air(4.0, false).process(q);
    // Doubling the distance halves the pressure, to the last bit the gain allows.
    assert!((p1 / p2 - 2.0).abs() < 1e-14);
    assert!((p1 / p4 - 4.0).abs() < 1e-14);
}

#[test]
fn the_retardation_is_an_exact_amplitude_preserving_sample_delay() {
    // r / c0 = 0.01 s -> exactly 480 samples at 343 m/s.
    let mut a = air(3.43, true);
    let delay = a.params().latency_samples;
    assert_eq!(delay, 480);
    let gain = a.params().gain;
    let out: Vec<f64> = (0..delay + 5)
        .map(|i| a.process(if i == 0 { 1.0 } else { 0.0 }))
        .collect();
    assert_eq!(out.iter().filter(|v| **v != 0.0).count(), 1);
    assert_eq!(out[delay], gain);
}

#[test]
fn the_delay_length_rounds_halves_to_even() {
    // THE trap of this batch, and no energy bar can see it. `float.__round__` is half-to-even;
    // C's `round` (and Rust's) is half-away-from-zero, which would make these 1 and 3.
    assert_eq!(radiation::py_round(0.5), 0.0);
    assert_eq!(radiation::py_round(1.5), 2.0);
    assert_eq!(radiation::py_round(2.5), 2.0);
    assert_eq!(radiation::py_round(-0.5), -0.0);
    assert_eq!(radiation::py_round(3.5), 4.0);
    // ...and reached through the constructor: fs = 2, c0 = 1, r = 1.25 -> 2.5 samples -> 2.
    let p = AirParams::new(2.0, 1.25, RHO0_AIR, 1.0, true).unwrap();
    assert_eq!(p.latency_samples, 2);
    assert_eq!(p.retardation_residual, 0.5);
}

#[test]
fn an_unretarded_read_out_has_no_delay_line_at_all() {
    let p = AirParams::new(FS, 5.0, RHO0_AIR, C0_AIR, false).unwrap();
    assert_eq!(p.latency_samples, 0);
    assert_eq!(p.retardation_residual, 0.0);
}

#[test]
fn reset_empties_the_delay_line() {
    let mut a = air(3.43, true);
    for _ in 0..50 {
        a.process(1.0);
    }
    a.reset();
    assert!(a.buf().iter().all(|v| *v == 0.0));
    assert_eq!(a.idx(), 0);
    assert_eq!(a.n(), 0);
    assert_eq!(a.process(1.0), 0.0);
}

#[test]
fn the_read_out_refuses_every_non_physical_parameter() {
    assert_eq!(
        AirParams::new(0.0, 1.0, RHO0_AIR, C0_AIR, true),
        Err(AirError::NonPositiveFs)
    );
    assert_eq!(
        AirParams::new(FS, 0.0, RHO0_AIR, C0_AIR, true),
        Err(AirError::NonPositiveDistance)
    );
    assert_eq!(
        AirParams::new(FS, 1.0, 0.0, C0_AIR, true),
        Err(AirError::NonPositiveRho0)
    );
    assert_eq!(
        AirParams::new(FS, 1.0, RHO0_AIR, 0.0, true),
        Err(AirError::NonPositiveC0)
    );
}

// -- tier 2: the constant-R load -----------------------------------------------------------------

#[test]
fn the_energy_channel_is_conserved_for_a_lossless_body() {
    // THE bar: E_body + integral P_rad dt is constant, and the radiated channel accounts for
    // everything the body sheds.
    let mut loaded = RadiatedBody::new(body(0.0), 2000.0).unwrap();
    loaded.set_state(&plucked(0.0), &[0.0; 4]);
    let e0 = loaded.energy();
    for _ in 0..4000 {
        loaded.step(0.0);
        assert!((loaded.energy() - e0).abs() <= 1e-10 * e0.abs());
    }
    // ...and the air really did take some of it.
    assert!(loaded.radiated_energy > 0.1 * e0);
    assert!(loaded.body().energy() < 0.9 * e0);
}

#[test]
fn the_load_is_passive_at_an_absurd_resistance() {
    // Unconditionally passive: 1 + R G >= 1 for any R >= 0, so there is no CFL to violate.
    let mut loaded = RadiatedBody::new(body(0.0), 1e9).unwrap();
    loaded.set_state(&plucked(0.0), &[0.0; 4]);
    let e0 = loaded.energy();
    let mut previous = e0;
    for _ in 0..2000 {
        loaded.step(0.0);
        let e = loaded.body().energy();
        assert!(e <= previous + 1e-12 * e0.abs());
        previous = e;
        assert!(loaded.radiated_energy >= 0.0);
    }
}

#[test]
fn zero_resistance_is_bit_identical_to_a_bare_body() {
    // An exact reduction in the original, and the one that says the rank-1 correction is inert
    // when it should be — including the `_accel` rewrite, which `pressure()` reads.
    let mut loaded = RadiatedBody::new(body(1.5), 0.0).unwrap();
    let mut bare = body(1.5);
    loaded.set_state(&plucked(0.0), &[0.0; 4]);
    bare.set_state(&plucked(0.0), &[0.0; 4]);
    for _ in 0..500 {
        loaded.step(0.0);
        bare.step(0.0);
        assert_eq!(loaded.body().q(), bare.q());
        assert_eq!(loaded.pressure(), bare.pressure());
    }
    assert_eq!(loaded.radiated_energy, 0.0);
}

#[test]
fn a_negative_resistance_is_refused() {
    assert_eq!(
        RadiatedBody::new(body(0.0), -1.0).unwrap_err(),
        "radiation resistance R must be >= 0."
    );
}

// -- tier 3: the rational impedance --------------------------------------------------------------

#[test]
fn the_sphere_constructor_is_impedance_consistent() {
    let a = 0.05;
    let p = LoadParams::from_sphere(FS, a, RHO0_AIR, C0_AIR).unwrap();
    // tau = a / c0, and the (R, M_a) pair is recognised as sphere-consistent.
    assert!((p.tau - a / C0_AIR).abs() <= 1e-14 * (a / C0_AIR));
    let radius = p.sphere_radius.expect("a sphere load has a radius");
    assert!((radius - a).abs() <= 1e-12 * a);
}

#[test]
fn an_arbitrary_pair_has_no_radius_and_refuses_the_far_field() {
    let load = RationalAirLoad::new(LoadParams::new(FS, 2000.0, 0.2, RHO0_AIR, C0_AIR).unwrap());
    assert!(load.params().sphere_radius.is_none());
    assert!(load.far_field_pressure(1.0, None).is_err());
}

#[test]
fn the_impedance_brackets_are_the_two_closed_form_anchors() {
    let a = 0.05;
    let load = RationalAirLoad::new(LoadParams::from_sphere(FS, a, RHO0_AIR, C0_AIR).unwrap());
    // ka -> 0: Re Z -> the free-space monopole resistance.
    let w_low = 2.0 * std::f64::consts::PI * 1.0;
    let re_low = load.impedance(w_low).0;
    let mono = radiation::monopole_radiation_resistance(w_low, RHO0_AIR, C0_AIR);
    assert!((re_low / mono - 1.0).abs() < 1e-6);
    // ka -> oo: Re Z -> R = rho0 c0 / S, the plane-wave saturation. The approach is from below
    // and second order in 1/(omega tau) -- at omega = 1e9 that is 4.7e-11, so 1e-9 is the honest
    // bar rather than a machine-precision one.
    let re_high = load.impedance(1e9).0;
    assert!(re_high < load.params().r);
    assert!((re_high / load.params().r - 1.0).abs() < 1e-9);
    // Im Z peaks at ka = 1, i.e. omega tau = 1.
    let w_corner = 1.0 / load.params().tau;
    let im_corner = load.impedance(w_corner).1;
    assert!(im_corner > load.impedance(0.5 * w_corner).1);
    assert!(im_corner > load.impedance(2.0 * w_corner).1);
}

#[test]
fn a_constant_r_load_has_no_reactance_at_any_frequency() {
    let load =
        RationalAirLoad::new(LoadParams::new(FS, 2000.0, f64::INFINITY, RHO0_AIR, C0_AIR).unwrap());
    for w in [1.0, 1e3, 1e5] {
        assert_eq!(load.impedance(w), (2000.0, 0.0));
        assert_eq!(load.impedance_discrete(w), (2000.0, 0.0));
    }
    // The `inf * 0` the original special-cases: zero stored energy, not NaN.
    assert_eq!(load.stored_energy(), 0.0);
}

#[test]
fn a_decoupled_load_has_no_impedance_at_all() {
    let load = RationalAirLoad::new(LoadParams::new(FS, 0.0, 0.2, RHO0_AIR, C0_AIR).unwrap());
    assert_eq!(load.impedance(1e4), (0.0, 0.0));
}

#[test]
fn the_load_refuses_every_non_physical_parameter() {
    assert_eq!(
        LoadParams::new(0.0, 1.0, 1.0, RHO0_AIR, C0_AIR),
        Err(LoadError::NonPositiveFs)
    );
    assert_eq!(
        LoadParams::new(FS, -1.0, 1.0, RHO0_AIR, C0_AIR),
        Err(LoadError::NegativeR)
    );
    assert_eq!(
        LoadParams::new(FS, 1.0, 0.0, RHO0_AIR, C0_AIR),
        Err(LoadError::NonPositiveMass)
    );
    // NaN too: the original writes `not (M_a > 0.0)` exactly so this is caught.
    assert_eq!(
        LoadParams::new(FS, 1.0, f64::NAN, RHO0_AIR, C0_AIR),
        Err(LoadError::NonPositiveMass)
    );
    assert_eq!(
        LoadParams::new(FS, 1.0, 1.0, 0.0, C0_AIR),
        Err(LoadError::NonPositiveMedium)
    );
    assert_eq!(
        LoadParams::from_sphere(FS, 0.0, RHO0_AIR, C0_AIR),
        Err(LoadError::NonPositiveRadius)
    );
}

#[test]
fn loaded_mode_reduces_to_the_constant_r_answer_when_there_is_no_mass() {
    // Im Z = 0 -> no added mass -> no pitch shift, and alpha collapses to a^2 R / (2 m).
    let load =
        RationalAirLoad::new(LoadParams::new(FS, 2000.0, f64::INFINITY, RHO0_AIR, C0_AIR).unwrap());
    let w0 = 2.0 * std::f64::consts::PI * 300.0;
    let (w_eff, alpha) = load.loaded_mode(w0, 0.02, 0.02, 50, 1e-14).unwrap();
    assert!((w_eff - w0).abs() <= 1e-14 * w0);
    let expect = 0.02 * 0.02 * 2000.0 / (2.0 * 0.02);
    assert!((alpha - expect).abs() <= 1e-14 * expect);
}

#[test]
fn loaded_mode_refuses_rather_than_returning_the_last_iterate() {
    let load = RationalAirLoad::new(LoadParams::from_sphere(FS, 0.05, RHO0_AIR, C0_AIR).unwrap());
    assert_eq!(
        load.loaded_mode(1.0, 0.02, 0.0, 50, 1e-14),
        Err(LoadedModeError::NonPositiveMass)
    );
    assert_eq!(
        load.loaded_mode(0.0, 0.02, 0.02, 50, 1e-14),
        Err(LoadedModeError::NonPositiveOmega0)
    );
    // One iteration is never enough for a genuinely loaded mode, so the cap fires.
    let err = load.loaded_mode(2.0 * std::f64::consts::PI * 110.0, 0.5, 0.02, 1, 1e-14);
    assert!(matches!(err, Err(LoadedModeError::NotConverged { .. })));
    // ...and the message quotes a last step of exactly zero, which is the ORIGINAL's arithmetic
    // faithfully transcribed rather than a slip here. See `RationalAirLoad::loaded_mode`.
    assert!(err
        .unwrap_err()
        .to_string()
        .contains("step 0.000e+00 > tol"));
}

// -- tier 3: the loaded body ---------------------------------------------------------------------

#[test]
fn the_three_way_energy_identity_holds_for_a_lossless_body() {
    // E_body + stored (radiation mass) + radiated = const. The stored term is what is new here:
    // this air gives back as well as taking, so neither half alone is monotone.
    let load = RationalAirLoad::new(LoadParams::new(FS, 1500.0, 0.03, RHO0_AIR, C0_AIR).unwrap());
    let mut loaded = ReactiveRadiatedBody::new(body(0.0), load).unwrap();
    loaded.set_state(&plucked(0.0), &[0.0; 4]);
    let e0 = loaded.energy();
    for _ in 0..4000 {
        loaded.step(0.0);
        assert!((loaded.energy() - e0).abs() <= 1e-10 * e0.abs());
    }
    assert!(loaded.load().stored_energy() >= 0.0);
    assert!(loaded.load().radiated_energy > 0.0);
}

#[test]
fn an_infinite_radiation_mass_is_bit_identical_to_the_constant_r_load() {
    // The second exact reduction, and the reason both classes call one `rank_one`. Anything that
    // reassociated `solve` — or recomputed `_G` a second time — would break this and nothing else.
    let r = 2000.0;
    let load =
        RationalAirLoad::new(LoadParams::new(FS, r, f64::INFINITY, RHO0_AIR, C0_AIR).unwrap());
    let mut batch3 = ReactiveRadiatedBody::new(body(0.0), load).unwrap();
    let mut batch2 = RadiatedBody::new(body(0.0), r).unwrap();
    batch3.set_state(&plucked(0.0), &[0.0; 4]);
    batch2.set_state(&plucked(0.0), &[0.0; 4]);
    for _ in 0..1000 {
        batch3.step(0.0);
        batch2.step(0.0);
        assert_eq!(batch3.body().q(), batch2.body().q());
        assert_eq!(batch3.load().radiated_energy, batch2.radiated_energy);
    }
    // The auxiliary state never moved: k p / inf is exactly 0.0.
    assert_eq!(batch3.load().u_l, 0.0);
}

#[test]
fn zero_resistance_is_bit_identical_to_a_bare_body_here_too() {
    let load = RationalAirLoad::new(LoadParams::new(FS, 0.0, 0.2, RHO0_AIR, C0_AIR).unwrap());
    let mut loaded = ReactiveRadiatedBody::new(body(1.5), load).unwrap();
    let mut bare = body(1.5);
    loaded.set_state(&plucked(0.0), &[0.0; 4]);
    bare.set_state(&plucked(0.0), &[0.0; 4]);
    for _ in 0..500 {
        loaded.step(0.0);
        bare.step(0.0);
        assert_eq!(loaded.body().q(), bare.q());
    }
}

#[test]
fn a_mismatched_timestep_is_refused_and_names_both_rates() {
    let load =
        RationalAirLoad::new(LoadParams::new(44100.0, 1000.0, 0.2, RHO0_AIR, C0_AIR).unwrap());
    let err = ReactiveRadiatedBody::new(body(0.0), load).unwrap_err();
    assert!(err.starts_with("load fs (44100.0) must match the body's (48000.0)"));
}

#[test]
fn the_far_field_power_balances_the_booked_radiated_power() {
    // The calibration bar: a sphere of radius r around the source must carry exactly the power the
    // load booked as gone. Exact at every ka because S |Z|^2 / (rho0 c0) == Re Z identically.
    let a = 0.05;
    let load = RationalAirLoad::new(LoadParams::from_sphere(FS, a, RHO0_AIR, C0_AIR).unwrap());
    let mut loaded = ReactiveRadiatedBody::new(body(0.0), load).unwrap();
    loaded.set_state(&plucked(0.0), &[0.0; 4]);
    let r = 1.5;
    let k = 1.0 / FS;
    let mut crossed = 0.0;
    for _ in 0..4000 {
        loaded.step(0.0);
        let p_far = loaded.load().far_field_pressure(r, None).unwrap();
        crossed += k * 4.0 * std::f64::consts::PI * r * r * p_far * p_far / (RHO0_AIR * C0_AIR);
    }
    let booked = loaded.load().radiated_energy;
    assert!(
        (crossed - booked).abs() <= 1e-9 * booked,
        "far-field {crossed:e} vs booked {booked:e}"
    );
}
