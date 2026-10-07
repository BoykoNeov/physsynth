//! Native acceptance bars for `body`, asserted without a Python interpreter in the way.
//! `tests/test_body.py` made the same claims about the Python original until retirement plan §43
//! carried the last of them here (the section at the bottom) and retired it.
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

    let e0 = body.energy();
    let mut previous = e0;
    for _ in 0..5_000 {
        body.step(0.0);
        let e = body.energy();
        assert!(e <= previous, "energy rose from {previous:e} to {e:e}");
        previous = e;
    }
    // The bank actually lost energy. (This line used to compare `previous` with itself, which is
    // the same number, and asserted nothing -- retirement plan §43.)
    assert!(previous < e0, "a lossy bank kept all {e0:e} of its energy");
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
    // Computed here, not read off `p`: a bar about a constant must not import it (plan §29.3).
    let k = 1.0 / fs;
    let omega = 2.0 * std::f64::consts::PI * f;
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
    // has to preserve, and no case of the retired `tests/test_body.py` exercised it.
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

// -- carried from `tests/test_body.py` (retirement plan §43) -------------------------------------
//
// The Python file's own rig: four modes at 110 / 196 / 261 / 440 Hz at 48 kHz (`make_body()`'s
// defaults), every mode displaced by a distinct `1e-3 (1 + 0.1 i)` and released from rest, so all
// four carry energy. The bars above run at three modes and unit masses; these keep the Python's
// figures, and two of them see defects nothing above can (the masses, and the loss rate).

const PY_FREQS: [f64; 4] = [110.0, 196.0, 261.0, 440.0];
const PY_FS: f64 = 48000.0;

/// `make_body(sigmas=sigma, masses=mass)` displaced by `_excite`.
fn python_rig(sigma: f64, mass: f64) -> ModalBody {
    let n = PY_FREQS.len();
    let p = Params::new(
        PY_FREQS.to_vec(),
        PY_FS,
        filled(sigma, n),
        filled(mass, n),
        filled(1.0, n),
        None,
    )
    .unwrap();
    let mut b = ModalBody::new(p);
    // `amplitude * (1.0 + 0.1 * np.arange(M))`, in that order.
    let q0: Vec<f64> = (0..n).map(|i| 1e-3 * (1.0 + 0.1 * i as f64)).collect();
    b.set_state(&q0, &vec![0.0; n]);
    b
}

#[test]
fn the_python_rig_conserves_its_energy_and_never_reaches_zero() {
    // `test_body_energy_conserved` (20,000 steps) and `test_body_energy_strictly_positive` (its
    // first 5,000), as one run. Measured drift 6.8e-14 against the 1e-10 bar. A NaN fails both
    // comparisons, so a run that blew up cannot report a clean drift.
    let mut body = python_rig(0.0, 1.0);
    let e0 = body.energy();
    assert!(e0 > 0.0, "the displaced body holds no energy: {e0:e}");
    for n in 1..=20_000 {
        body.step(0.0);
        let e = body.energy();
        assert!(e > 0.0, "energy {e:e} at step {n} is not strictly positive");
        let rel = ((e - e0) / e0).abs();
        assert!(rel < 1e-10, "body drift {rel:e} at step {n}");
    }
}

#[test]
fn the_energy_is_linear_in_the_modal_masses() {
    // Every other bar in this file runs at unit masses, where an energy that forgot `m` (or
    // squared it) is indistinguishable from the right one -- and the unforced step never reads `m`
    // at all, so such an energy would also conserve perfectly. Doubling every mass at a fixed state
    // must double the energy.
    let e1 = python_rig(0.0, 1.0).energy();
    let e2 = python_rig(0.0, 2.0).energy();
    let ratio = e2 / e1;
    assert!(
        (ratio - 2.0).abs() <= 2.0 * 1e-12,
        "energy ratio {ratio} for doubled masses, want 2"
    );
}

#[test]
fn the_python_rig_is_passive_at_sigma_8() {
    // `test_body_passivity_monotonic`. The tolerance is the Python's `1e-12 e0` and it is needed:
    // at this rig the largest step UP is +1.8e-15 (rounding on a 9.5 J total), so the strict
    // `e <= previous` of `a_lossy_bank_decreases_monotonically` would go red here.
    let mut body = python_rig(8.0, 1.0);
    let e0 = body.energy();
    let mut previous = e0;
    for n in 1..=20_000 {
        body.step(0.0);
        let e = body.energy();
        let up = e - previous;
        assert!(up <= 1e-12 * e0, "energy rose by {up:e} at step {n}");
        previous = e;
    }
    // Not part of the Python's claim, but what makes it non-vacuous: the run did lose its energy
    // (0.0121 J of 9.54 is left).
    assert!(
        previous < 0.01 * e0,
        "a sigma = 8 body kept {previous:e} of {e0:e}"
    );
}

#[test]
fn a_single_damped_mode_decays_at_twice_sigma() {
    // `test_body_decay_rate_matches_2sigma`: E(t) ~ E0 exp(-2 sigma t) over one second. The
    // Python's 2% bar; the measured error is 2.5e-5, the cross-time energy's own wobble.
    let (sigma, secs, fs) = (6.0, 1.0, 48000.0);
    let p = Params::new(vec![220.0], fs, vec![sigma], vec![1.0], vec![1.0], None).unwrap();
    let mut body = ModalBody::new(p);
    body.set_state(&[1e-3], &[0.0]);
    let e0 = body.energy();
    for _ in 0..(secs * fs) as usize {
        body.step(0.0);
    }
    let measured = body.energy() / e0;
    let expected = (-2.0 * sigma * secs).exp();
    let rel = (measured.ln() - expected.ln()).abs() / expected.ln().abs();
    assert!(
        rel < 0.02,
        "decay rate off by {:.3}% (got {measured:e}, want {expected:e})",
        100.0 * rel
    );
    // And at 0.1%, still 40x the measured error (added in §43, the human's call). The 2% bar
    // passes a loss coefficient read 1% high -- the decay is then 1% fast and the only witness
    // was the Windows-exact viewer freeze.
    assert!(
        rel < 1e-3,
        "decay rate off by {:.4}%: the loss is not 2 sigma",
        100.0 * rel
    );
}

#[test]
fn a_launched_lossless_mode_is_the_exact_discrete_sine() {
    // The start-up bar's velocity half (plan §30.4), not in the Python file: every start there,
    // and every one above, is from rest, so a start-up that flipped the sign of `k v0` in
    // `q^{-1} = q0 - k v0 - 1/2 k^2 omega^2 q0` passed the whole workspace (§43). Launched from
    // q0 = 0 with velocity V, the centred velocity of the first step is V exactly
    // (q^{-1} = -kV, q^1 = kV), and the mode is then the discrete sine
    // kV sin(Omega n k) / sin(Omega k).
    let (f, fs, v) = (261.63, 48000.0, 0.37);
    let p = Params::new(vec![f], fs, vec![0.0], vec![1.0], vec![1.0], None).unwrap();
    let mut body = ModalBody::new(p);
    body.set_state(&[0.0], &[v]);
    let q_m1 = body.q_prev()[0];
    body.step(0.0);
    let k = 1.0 / fs;
    let centred = (body.q()[0] - q_m1) / (2.0 * k);
    assert!(
        (centred - v).abs() <= 1e-14 * v,
        "launched at {v}, the first step's centred velocity is {centred}"
    );

    let omega = 2.0 * std::f64::consts::PI * f;
    let big_omega = 2.0 * (0.5 * omega * k).asin() / k;
    let amp = k * v / (big_omega * k).sin();
    let mut worst = (body.q()[0] - amp * (big_omega * k).sin()).abs();
    for n in 2..=4000usize {
        body.step(0.0);
        let err = (body.q()[0] - amp * (big_omega * n as f64 * k).sin()).abs();
        assert!(!err.is_nan(), "NaN at step {n}");
        worst = worst.max(err);
    }
    // The amplitude is v / Omega ~ 2.3e-4, below the cosine bar's 1e-3, so its absolute 1e-14 is
    // kept as it stands. Measured: 2.5e-17 here, 1.5e-16 relative on the centred velocity.
    assert!(
        worst < 1e-14,
        "a launched mode is not the exact discrete sine: worst {worst:e}"
    );
}

#[test]
fn a_single_lossless_mode_is_the_exact_discrete_cosine() {
    // `test_body_single_mode_is_exact_discrete_cosine`: with the consistent (v0 = 0) start, one
    // lossless mode reproduces q0 cos(Omega n k) to machine precision, with
    // Omega = (2/k) asin(omega k / 2).
    // Sharper than the zero-crossing bar above in two ways: a start-up that drops the 1/2 in q^{-1}
    // shifts the PHASE without touching the period, and omega and k are computed here from f and
    // fs rather than read off the model, so a wrong constant there cannot move the oracle with it.
    let (f, fs, q0) = (261.63, 48000.0, 1e-3);
    let p = Params::new(vec![f], fs, vec![0.0], vec![1.0], vec![1.0], None).unwrap();
    let mut body = ModalBody::new(p);
    body.set_state(&[q0], &[0.0]);

    // `discrete_sho_frequency(f, 1/fs)`, spelled as the helper spelled it.
    let k = 1.0 / fs;
    let omega = 2.0 * std::f64::consts::PI * f;
    let fd = (0.5 * omega * k).asin() / (std::f64::consts::PI * k);
    let omega_d = 2.0 * std::f64::consts::PI * fd;

    // Snapshots 0..=4000, step 0 included, as `simulate(snapshot_stride=1)` records them.
    let mut worst = (body.q()[0] - q0).abs();
    for n in 1..=4000usize {
        body.step(0.0);
        let exact = q0 * (omega_d * n as f64 / fs).cos();
        let err = (body.q()[0] - exact).abs();
        // `f64::max` drops a NaN, where `np.max` propagates it: check before folding.
        assert!(!err.is_nan(), "NaN at step {n}");
        worst = worst.max(err);
    }
    assert!(
        worst < 1e-14,
        "single mode is not the exact discrete cosine: worst {worst:e}"
    );
}
