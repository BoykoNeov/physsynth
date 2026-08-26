//! Native acceptance bars for `body` — the same claims `tests/test_body.py` makes about the
//! Python original, asserted here against the Rust one without a Python interpreter in the way.
//!
//! The project's contract (`CLAUDE.md`) is that correctness is asserted against closed-form
//! physics, not against a reference implementation. These are that: a lossless bank conserves its
//! discrete energy, a lossy one decreases monotonically, a single mode oscillates at the frequency
//! the leapfrog's dispersion relation says it should, and the construction-time refusals fire.
//!
//! One bar here has no counterpart in the membrane's file, and it is the interesting one:
//! **a body whose `q` is corrected from outside must still report the acceleration of the step it
//! actually took.** Three modules do that correction (see the module header), and none of them is
//! reachable from a body test in `tests/`.

use physsynth_core::body::{self, ModalBody, ParamError, Params};

/// The `1.0` default for a broadcast scalar, spelled once.
fn filled(value: f64, n: usize) -> Vec<f64> {
    vec![value; n]
}

fn params(freqs: &[f64], fs: f64, sigma: f64) -> Params {
    let n = freqs.len();
    Params::new(
        freqs.to_vec(),
        fs,
        filled(sigma, n),
        filled(1.0, n),
        filled(1.0, n),
        None,
    )
    .expect("parameters should be accepted")
}

// -- construction ------------------------------------------------------------------------------

#[test]
fn an_empty_bank_is_rejected() {
    let err = Params::new(vec![], 48000.0, vec![], vec![], vec![], None).unwrap_err();
    assert_eq!(err, ParamError::EmptyFreqs);
    assert_eq!(
        err.to_string(),
        "freqs must be a 1-D array with at least one mode."
    );
}

#[test]
fn non_physical_parameters_are_rejected_in_the_originals_order() {
    // A negative frequency beats a bad fs to the report, because the original checks it first.
    let err = Params::new(
        vec![100.0, -5.0],
        -1.0,
        filled(0.0, 2),
        filled(1.0, 2),
        filled(1.0, 2),
        None,
    )
    .unwrap_err();
    assert_eq!(err, ParamError::NonPositiveFreq);

    let err = Params::new(vec![100.0], 0.0, vec![0.0], vec![1.0], vec![1.0], None).unwrap_err();
    assert_eq!(err, ParamError::NonPositiveFs);

    let err = Params::new(
        vec![100.0],
        48000.0,
        vec![-1e-9],
        vec![1.0],
        vec![1.0],
        None,
    )
    .unwrap_err();
    assert_eq!(err, ParamError::NegativeSigma);

    let err = Params::new(vec![100.0], 48000.0, vec![0.0], vec![0.0], vec![1.0], None).unwrap_err();
    assert_eq!(err, ParamError::NonPositiveMass);
}

#[test]
fn a_mode_above_the_modal_cfl_is_rejected_and_the_message_names_the_worst_one() {
    // 20 kHz at 48 kHz gives omega*k = 2.618 > 2. Mode 1 is the offender and also the argmax.
    let err = Params::new(
        vec![100.0, 20000.0],
        48000.0,
        filled(0.0, 2),
        filled(1.0, 2),
        filled(1.0, 2),
        None,
    )
    .unwrap_err();
    let text = err.to_string();
    assert!(text.starts_with("CFL violated: omega*k = "), "{text}");
    assert!(text.contains("for mode 1"), "{text}");
    assert!(text.contains("f = 20000.000 Hz at fs = 48000.0"), "{text}");
}

#[test]
fn the_reported_mode_is_the_largest_cfl_number_not_the_first_offender() {
    // Both violate; mode 0 is first in index order and mode 1 is the argmax. The original reports
    // the argmax, and a port that reported the first offender would pass every physics bar.
    let err = Params::new(
        vec![18000.0, 22000.0],
        48000.0,
        filled(0.0, 2),
        filled(1.0, 2),
        filled(1.0, 2),
        None,
    )
    .unwrap_err();
    assert!(err.to_string().contains("for mode 1"), "{err}");
}

#[test]
fn radiation_defaults_to_phi_as_a_copy_not_a_share() {
    let p = Params::new(vec![220.0], 48000.0, vec![0.0], vec![1.0], vec![0.7], None).unwrap();
    assert_eq!(p.a, p.phi);
    // Given explicitly, it is kept as given.
    let p = Params::new(
        vec![220.0],
        48000.0,
        vec![0.0],
        vec![1.0],
        vec![0.7],
        Some(vec![1e-3]),
    )
    .unwrap();
    assert_eq!(p.a, vec![1e-3]);
    assert_eq!(p.phi, vec![0.7]);
}

// -- energy ------------------------------------------------------------------------------------

#[test]
fn a_lossless_bank_conserves_its_energy() {
    let p = params(&[220.0, 337.0, 512.5], 48000.0, 0.0);
    let mut body = ModalBody::new(p);
    body.set_state(&[1e-3, -4e-4, 2e-4], &[0.0, 0.0, 0.0]);

    let e0 = body.energy();
    assert!(e0 > 0.0, "a displaced body should hold energy");
    for _ in 0..20_000 {
        body.step(0.0);
        let rel = ((body.energy() - e0) / e0).abs();
        assert!(rel < 1e-10, "energy drifted by {rel:e}");
    }
}

#[test]
fn a_lossy_bank_decreases_monotonically() {
    let p = params(&[220.0, 337.0], 48000.0, 3.0);
    let mut body = ModalBody::new(p);
    body.set_state(&[1e-3, 5e-4], &[0.0, 0.0]);

    let mut previous = body.energy();
    for _ in 0..5_000 {
        body.step(0.0);
        let e = body.energy();
        assert!(e <= previous, "energy rose from {previous:e} to {e:e}");
        previous = e;
    }
    assert!(previous < body.energy() * 1e3);
}

#[test]
fn a_driven_bank_gains_energy_and_a_zero_force_is_the_undriven_step() {
    let p = params(&[220.0], 48000.0, 0.0);
    let mut driven = ModalBody::new(p.clone());
    let mut free = ModalBody::new(p);
    driven.set_state(&[0.0], &[0.0]);
    free.set_state(&[0.0], &[0.0]);

    for _ in 0..100 {
        driven.step(1.0);
        free.step(0.0);
    }
    assert!(driven.energy() > 0.0, "a forced body must be moving");
    assert_eq!(
        free.energy(),
        0.0,
        "an unforced body at rest must stay there"
    );
    assert_eq!(free.q(), &[0.0]);
}

// -- the oracle --------------------------------------------------------------------------------

#[test]
fn a_single_lossless_mode_follows_the_leapfrog_dispersion_relation() {
    // The explicit leapfrog turns omega into the discrete omega_d where
    //   sin(omega_d k / 2) = omega k / 2,
    // so a free mode's zero crossings are spaced by pi / omega_d, not pi / omega.
    let fs = 48000.0;
    let f = 440.0;
    let p = params(&[f], fs, 0.0);
    let k = p.k;
    let omega = p.omega[0];
    let mut body = ModalBody::new(p);
    body.set_state(&[1.0], &[0.0]);

    let omega_d = 2.0 * (omega * k / 2.0).asin() / k;

    // Count zero crossings over a whole number of cycles and read the period back.
    let steps = 48_000usize;
    let mut previous = body.q()[0];
    let mut crossings = 0usize;
    let mut last_crossing = 0.0f64;
    let mut first_crossing = f64::NAN;
    for i in 1..=steps {
        body.step(0.0);
        let current = body.q()[0];
        if previous.signum() != current.signum() {
            // Linear interpolation between the two samples straddling zero.
            let frac = previous / (previous - current);
            let t = ((i - 1) as f64 + frac) * k;
            if crossings == 0 {
                first_crossing = t;
            }
            last_crossing = t;
            crossings += 1;
        }
        previous = current;
    }
    assert!(crossings > 100, "expected many crossings, saw {crossings}");
    let half_period = (last_crossing - first_crossing) / (crossings - 1) as f64;
    let measured = std::f64::consts::PI / half_period;
    let rel = ((measured - omega_d) / omega_d).abs();
    assert!(
        rel < 1e-6,
        "measured omega {measured} vs discrete {omega_d} (continuum {omega}), rel {rel:e}"
    );
    // And the discrete frequency is genuinely above the continuum one, so the test is not
    // vacuously passing against an oracle that happens to equal it.
    assert!(omega_d > omega);
}

// -- the read-outs and the thing only a client can break ---------------------------------------

#[test]
fn the_bridge_read_outs_are_the_definitions() {
    let n = 3;
    let p = Params::new(
        vec![220.0, 337.0, 512.5],
        48000.0,
        filled(0.0, n),
        vec![0.02, 0.03, 0.05],
        vec![1.0, -0.5, 0.25],
        None,
    )
    .unwrap();
    let mut b = ModalBody::new(p);
    b.set_state(&[1e-3, -4e-4, 2e-4], &[0.0; 3]);
    b.step(0.0);

    let p = b.params();
    let expect_w: f64 = (0..3).map(|i| p.phi[i] * b.q()[i]).sum();
    assert!((b.bridge_displacement() - expect_w).abs() <= 1e-18 + expect_w.abs() * 1e-15);

    let expect_v: f64 = (0..3)
        .map(|i| p.phi[i] * ((b.q()[i] - b.q_prev()[i]) / p.k))
        .sum();
    assert!((b.bridge_velocity() - expect_v).abs() <= 1e-18 + expect_v.abs() * 1e-15);
}

#[test]
fn pressure_before_the_first_step_is_the_free_response_not_zero() {
    // `set_state` seeds `accel` with `-omega^2 q0`. A port that zeroed it instead would read a
    // silent zero out of `pressure()` until the first step — and every energy bar would stay green.
    let p = params(&[220.0], 48000.0, 0.0);
    let omega = p.omega[0];
    let mut b = ModalBody::new(p);
    b.set_state(&[1e-3], &[0.0]);
    let expected = -omega * omega * 1e-3;
    assert!((b.pressure() - expected).abs() <= expected.abs() * 1e-15);
}

#[test]
fn the_acceleration_carries_an_external_force() {
    // Reconstructing q'' = -omega^2 q - 2 sigma q' would drop the bridge force entirely. This is
    // the difference, made visible: a forced step's acceleration is not the free-response one.
    let p = params(&[220.0], 48000.0, 0.0);
    let omega = p.omega[0];
    let mut b = ModalBody::new(p);
    b.set_state(&[0.0], &[0.0]);
    b.step(7.0);
    let reconstructed = -omega * omega * b.q()[0];
    assert!(
        (b.pressure() - reconstructed).abs() > 1e3 * reconstructed.abs().max(1e-30),
        "the true acceleration should be dominated by the force, not the restoring term"
    );
}

#[test]
fn a_rank_one_correction_from_outside_is_visible_to_the_read_outs() {
    // What `RadiatedBody`, the rational air load and `RoomLoadedBody` all do: step, then correct
    // `q` and rewrite `accel` from the corrected second difference. The point of the test is that
    // the state is reachable and that the read-outs follow it — this is the contract the binding
    // has to preserve, and no `tests/test_body.py` case exercises it.
    let p = params(&[220.0, 337.0], 48000.0, 0.0);
    let k = p.k;
    let mut b = ModalBody::new(p);
    b.set_state(&[1e-3, 5e-4], &[0.0, 0.0]);

    let q_nm1: Vec<f64> = b.q_prev().to_vec();
    b.step(0.0);
    let before = b.pressure();

    let correction = [1e-6, -2e-6];
    let q_prev_snapshot: Vec<f64> = b.q_prev().to_vec();
    for (i, c) in correction.iter().enumerate() {
        b.q_mut()[i] -= c;
    }
    let corrected: Vec<f64> = (0..2)
        .map(|i| ((b.q()[i] - 2.0 * q_prev_snapshot[i]) + q_nm1[i]) / (k * k))
        .collect();
    b.accel_mut().copy_from_slice(&corrected);

    assert_ne!(b.pressure(), before, "the correction must reach pressure()");
    let expect: f64 = (0..2).map(|i| b.params().a[i] * corrected[i]).sum();
    assert!((b.pressure() - expect).abs() <= expect.abs() * 1e-14);
}

// -- kernels vs the owning struct ---------------------------------------------------------------

#[test]
fn the_free_functions_and_the_struct_agree() {
    // The binding calls the kernels directly (its buffers are NumPy arrays), so the two paths must
    // not be allowed to drift apart.
    let p = params(&[220.0, 337.0], 48000.0, 1.5);
    let mut b = ModalBody::new(p.clone());
    b.set_state(&[1e-3, 5e-4], &[0.1, -0.2]);

    let (mut q_prev, mut accel) = body::initial_state(&[1e-3, 5e-4], &[0.1, -0.2], &p);
    let mut q = vec![1e-3, 5e-4];
    assert_eq!(q_prev, b.q_prev());
    assert_eq!(accel, b.accel(), "set_state must seed the same accel");

    for _ in 0..500 {
        b.step(0.3);
        let mut next = vec![0.0; 2];
        body::step_into(&q, &q_prev, 0.3, &mut next, &mut accel, &p);
        q_prev = q;
        q = next;
        assert_eq!(q, b.q(), "the two step paths diverged");
        assert_eq!(q_prev, b.q_prev());
    }
    assert_eq!(body::energy(&q, &q_prev, &p), b.energy());
    assert_eq!(body::pressure(&accel, &p), b.pressure());
}
