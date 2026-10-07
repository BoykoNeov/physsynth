//! Native acceptance bars for `radiation` — the air node's three tiers, asserted without a Python
//! interpreter in the way.
//!
//! The project's contract (`CLAUDE.md`) is closed-form physics, not agreement with a reference
//! implementation. These bars started as a subset of `tests/test_radiation.py`'s claims and, since
//! retirement plan §45 (phase C, carrying batch 23), are the whole of them: that file is deleted,
//! and every claim it made is carried here — the monopole gain and its exact inverse-distance law,
//! an integer-sample delay that preserves amplitude, the energy identity that says the air's books
//! balance, the measured impedance against the pre-warped closed form, the loaded mode against
//! both parts of `Z_a`, and the far-field calibration. The Python figures were recorded first
//! (`W:\temp\claude\batch23\record.txt`); §45.1 compares them.
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
//!
//! # Rigs that are not at unit weight and mass
//!
//! [`body`] is `tests/helpers.py::make_body`'s rig: every mode at mass 1 and radiation weight 1.
//! There, `a^2 -> a` in `G`, a dropped `/m`, and a dropped `a` in the volume velocity are all the
//! identity. The Python's own rigs at mass 0.02 (the bit-identity and dense-solve bars) and at
//! weight 0.02 (the loaded-mode bars) are carried at those values for exactly that reason.
//!
//! # Air written in the test
//!
//! [`RHO0`] and [`C0`] are literals, and [`the_air_constants_are_standard_air_in_both_crates`]
//! pins the crate's own constants to them (§29.3: a bar about a constant must not import it).

use physsynth_core::body::{ModalBody, Params as BodyParams};
use physsynth_core::dense::{lu_factor, lu_solve};
use physsynth_core::radiation::{
    self, c_div, AirError, AirParams, AirRadiation, FarFieldError, LoadError, LoadParams,
    LoadedModeError, RadiatedBody, RationalAirLoad, ReactiveRadiatedBody, C0_AIR, RHO0_AIR,
};
use std::f64::consts::PI;

const FS: f64 = 48000.0;
/// The four off-harmonic modes `tests/helpers.py` used for a guitar-ish body.
const FREQS: [f64; 4] = [110.0, 196.0, 261.0, 440.0];
/// Standard air, written here rather than imported (module header).
const RHO0: f64 = 1.2041;
const C0: f64 = 343.0;
/// `tests/helpers.py::SPHERE_RADIUS_DEFAULT` — a 5 cm pulsating sphere (`ka = 1` near 1.1 kHz).
const SPHERE: f64 = 0.05;

fn body(sigma: f64) -> ModalBody {
    body_at(FS, sigma, 1.0)
}

/// [`FREQS`] at sample rate `fs`, one `sigma` and one modal mass for every mode, `phi = a = 1`.
fn body_at(fs: f64, sigma: f64, mass: f64) -> ModalBody {
    let n = FREQS.len();
    ModalBody::new(
        BodyParams::new(
            FREQS.to_vec(),
            fs,
            vec![sigma; n],
            vec![mass; n],
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

fn sphere(fs: f64, radius: f64) -> RationalAirLoad {
    RationalAirLoad::new(LoadParams::from_sphere(fs, radius, RHO0, C0).expect("a real sphere"))
}

fn pair(fs: f64, r: f64, m_a: f64) -> RationalAirLoad {
    RationalAirLoad::new(LoadParams::new(fs, r, m_a, RHO0, C0).expect("a well-posed load"))
}

fn reactive(sigma: f64, r: f64, m_a: f64) -> ReactiveRadiatedBody {
    ReactiveRadiatedBody::new(body(sigma), pair(FS, r, m_a)).expect("matching timesteps")
}

fn plucked(sigma: f64) -> Vec<f64> {
    let _ = sigma;
    vec![1e-3, -8e-4, 6e-4, 4e-4]
}

/// `max |x|` that propagates NaN — `fold(_, f64::max)` would drop it (§29).
fn nan_max(xs: impl IntoIterator<Item = f64>) -> f64 {
    let mut m = 0.0f64;
    for x in xs {
        if x.is_nan() {
            return f64::NAN;
        }
        m = m.max(x);
    }
    m
}

/// `Σ a_i ((q⁺ − 2q) + q⁻) / k²` — the body's volume acceleration from three recorded states,
/// formed outside the model. What `pressure()` must report when it carries the load.
fn volume_accel(a: &[f64], q_next: &[f64], q: &[f64], q_prev: &[f64], k: f64) -> f64 {
    let mut acc = 0.0;
    for i in 0..a.len() {
        acc += a[i] * (((q_next[i] - 2.0 * q[i]) + q_prev[i]) / (k * k));
    }
    acc
}

// -- the air itself -----------------------------------------------------------------------------

#[test]
fn the_air_constants_are_standard_air_in_both_crates() {
    // Carried from the deleted Python shim, whose `RHO0_AIR = 1.2041` / `C0_AIR = 343.0` were the
    // last literals outside this crate that the crate's constants were checked against. The
    // analysis crate keeps its own copy (it cannot depend on this one); the two must agree.
    assert_eq!(RHO0_AIR, 1.2041);
    assert_eq!(C0_AIR, 343.0);
    assert_eq!(RHO0_AIR, physsynth_analysis::radiation::RHO0_AIR);
    assert_eq!(C0_AIR, physsynth_analysis::radiation::C0_AIR);
}

// -- a medium that is not air (§45.4, the human's call) --------------------------------------------
//
// Every caller in the workspace builds these types with standard air, so a read that ignored the
// density or sound speed it was handed and used 1.2041 / 343 instead was the identity at every call
// site: sixteen such reads passed the whole workspace. These bars build the three tiers at a
// medium written here, and assert every value that reads either constant.

/// A medium that is not air, written in the test.
const RHO0_X: f64 = 0.9;
const C0_X: f64 = 380.0;

#[test]
fn the_read_out_honours_the_medium_it_is_given() {
    // r = 2 m: 252.6 samples of travel at 380 m/s, where standard air would give 279.9.
    let r = 2.0;
    let p = AirParams::new(FS, r, RHO0_X, C0_X, true).unwrap();
    assert_eq!(p.gain, RHO0_X / (4.0 * PI * r));
    assert_eq!(p.retardation_seconds, r / C0_X);
    assert_eq!(p.latency_samples, 253);
    assert_eq!((p.rho0, p.c0), (RHO0_X, C0_X));
    let omega = 2.0 * PI * 200.0;
    let mono = radiation::monopole_radiation_resistance(omega, RHO0_X, C0_X);
    let want = RHO0_X * omega * omega / (4.0 * PI * C0_X);
    assert!((mono / want - 1.0).abs() <= 1e-14, "{mono} vs {want}");
}

#[test]
fn the_sphere_load_honours_the_medium_it_is_given() {
    // R = rho0 c0 / S, M_a = rho0 / (4 pi a), tau = a / c0 — and the pair must still be recognised
    // as this sphere, which is a test that reads both constants again.
    let a = SPHERE;
    let p = LoadParams::from_sphere(FS, a, RHO0_X, C0_X).unwrap();
    let close = |x: f64, y: f64| (x / y - 1.0).abs() <= 1e-14;
    assert!(close(p.r, RHO0_X * C0_X / (4.0 * PI * a * a)), "R {}", p.r);
    assert!(close(p.m_a, RHO0_X / (4.0 * PI * a)), "M_a {}", p.m_a);
    assert!(close(p.tau, a / C0_X), "tau {}", p.tau);
    assert_eq!((p.rho0, p.c0), (RHO0_X, C0_X));
    let radius = p
        .sphere_radius
        .expect("the medium's own sphere is recognised as one");
    assert!((radius / a - 1.0).abs() <= 1e-12, "{radius}");
    let area = p.sphere_area.expect("and has an area");
    assert!((area / (4.0 * PI * a * a) - 1.0).abs() <= 1e-12, "{area}");
    // A pair built directly keeps the medium it was given too.
    let q = LoadParams::new(FS, 2000.0, 0.2, RHO0_X, C0_X).unwrap();
    assert_eq!((q.rho0, q.c0), (RHO0_X, C0_X));
}

// -- restarting a loaded body (§45.4, the human's call) --------------------------------------------

#[test]
fn restarting_a_loaded_body_empties_its_air_channels() {
    // Nothing else restarts a body that has already run, so a `set_state` that kept the previous
    // run's radiated energy passed the whole workspace: the motion is unaffected, but the conserved
    // total would carry the old number.
    let q0 = [1e-3, -8e-4, 6e-4, 4e-4];
    let mut constant = RadiatedBody::new(body(0.0), 2000.0).unwrap();
    let mut rational = reactive(0.0, 2000.0, 0.2);
    constant.set_state(&q0, &[0.0; 4]);
    rational.set_state(&q0, &[0.0; 4]);
    for _ in 0..200 {
        constant.step(0.0);
        rational.step(0.0);
    }
    assert!(constant.radiated_energy > 0.0 && constant.volume_velocity != 0.0);
    assert!(rational.load().radiated_energy > 0.0 && rational.load().u_l != 0.0);
    for restart in 0..2 {
        if restart == 0 {
            constant.set_state(&q0, &[0.0; 4]);
            rational.set_state(&q0, &[0.0; 4]);
        } else {
            for _ in 0..50 {
                constant.step(0.0);
                rational.step(0.0);
            }
            constant.reset();
            rational.reset();
        }
        assert_eq!(constant.radiated_energy, 0.0);
        assert_eq!(constant.volume_velocity, 0.0);
        assert_eq!(constant.n(), 0);
        assert_eq!(constant.energy(), constant.body().energy());
        let load = rational.load();
        assert_eq!(
            (load.radiated_energy, load.u_l, load.stored_energy()),
            (0.0, 0.0, 0.0)
        );
        assert_eq!(
            (load.volume_velocity, load.pressure_load, load.n()),
            (0.0, 0.0, 0)
        );
        assert_eq!(rational.n(), 0);
        assert_eq!(rational.energy(), rational.body().energy());
    }
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
fn the_gain_is_exact_and_the_read_out_is_the_gain_times_the_input() {
    // Carried from `test_far_field_gain_is_exact`: `p_far = rho0 Q'' / (4 pi r)` with nothing
    // else in the path, so the gain and every output are exact — equality, not a tolerance.
    let r = 2.0;
    let mut a = air(r, false);
    let gain = RHO0 / (4.0 * PI * r);
    assert_eq!(a.params().gain, gain);
    for qdd in [0.0, 1.0, -3.5, 42.0, -1e-4, 7.0] {
        assert_eq!(a.process(qdd), gain * qdd);
    }
}

#[test]
fn a_prescribed_volume_velocity_radiates_the_closed_form_monopole_amplitude() {
    // Carried from `test_monopole_amplitude_from_volume_velocity`: `U = U0 sin(wt)` gives
    // `Q'' = U0 w cos(wt)` and so `|p| = rho0 w U0 / (4 pi r)`. The first sample is cos(0) = 1,
    // so the sampled peak is the analytic one (the Python measured a relative gap of exactly 0).
    let (r, f, u0) = (1.5, 220.0, 3e-4);
    let mut a = air(r, false);
    let omega = 2.0 * PI * f;
    let n = (4.0 * FS / f) as usize; // a few periods
    assert_eq!(n, 872);
    let peak = nan_max((0..n).map(|i| {
        let t = i as f64 / FS;
        a.process(u0 * omega * (omega * t).cos()).abs()
    }));
    let expect = RHO0 * omega * u0 / (4.0 * PI * r);
    assert!((peak / expect - 1.0).abs() <= 1e-6, "{peak} vs {expect}");
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
    // The quantisation error is at most half a sample (here it is exactly zero).
    assert!(a.params().retardation_residual.abs() <= 0.5);
    let gain = a.params().gain;
    let out: Vec<f64> = (0..delay + 5)
        .map(|i| a.process(if i == 0 { 1.0 } else { 0.0 }))
        .collect();
    assert_eq!(out.iter().filter(|v| **v != 0.0).count(), 1);
    assert_eq!(out[delay], gain);
}

#[test]
fn a_wavefront_in_transit_is_silence() {
    // Carried from `test_wavefront_in_transit_is_silence`: a constant drive is heard as nothing
    // for exactly `latency_samples`, then as the gain.
    let mut a = air(3.43, true);
    let lat = a.params().latency_samples;
    for i in 0..lat {
        assert_eq!(
            a.process(1.0),
            0.0,
            "sample {i} heard a wavefront still in transit"
        );
    }
    assert_eq!(a.process(1.0), a.params().gain);
}

/// `tests/test_airbox_freefield.py::test_the_lumped_tier_agrees_with_the_same_closed_form` — the
/// room's free-field bar (`src/airbox.rs`) fits the box to `p = rho0 Qdd(t - r/c0) / (4 pi r)`; this
/// is the other half of that cross-tier claim. Fed a pulse's volume acceleration, the integer-sample
/// delay line — a completely different construction — emits the same closed form, so the box and
/// the lumped tier agree with each other via a law neither of them defines.
#[test]
fn the_retarded_read_out_emits_the_closed_form_monopole_of_a_pulse() {
    let (fs, r) = (40000.0, 0.4);
    let mut a = AirRadiation::new(AirParams::new(fs, r, RHO0_AIR, C0_AIR, true).unwrap());
    let (gain, lat) = (a.params().gain, a.params().latency_samples);
    assert_eq!(lat, 47); // r / c0 = 46.65 samples
                         // `tests/helpers.py::gaussian_pulse(fs, 1400)`'s derivative: sigma = 1/(2 pi f0), centred 4
                         // sigma in, amplitude 1e-3.
    let sigma = 1.0 / (2.0 * std::f64::consts::PI * 1400.0);
    let t0 = 4.0 * sigma;
    let qdot = |t: f64| {
        -1e-3 * (t - t0) / (sigma * sigma) * (-((t - t0) * (t - t0)) / (2.0 * sigma * sigma)).exp()
    };
    let out: Vec<f64> = (0..400).map(|n| a.process(qdot(n as f64 / fs))).collect();
    let expect: Vec<f64> = (0..400)
        .map(|n| gain * qdot(n as f64 / fs - lat as f64 / fs))
        .collect();
    let scale = expect.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    // The wavefront-in-transit prologue is documented silence, while the closed form already has
    // the pulse's leading tail arriving — so the comparison starts at the latency.
    for n in lat..400 {
        let err = (out[n] - expect[n]).abs();
        assert!(err < 1e-14 * scale, "sample {n}: {:e}", err / scale);
    }
    assert!(out[..lat].iter().all(|&v| v == 0.0));
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
    // ...and its first output is already the far field (`test_retarded_false_has_no_delay`).
    let mut a = AirRadiation::new(p);
    assert_eq!(a.process(2.0), a.params().gain * 2.0);
}

#[test]
fn the_read_out_is_linear() {
    // Carried from `test_linearity`: three read-outs, inputs a, b and a + b. The Python's bar was
    // `atol = 1e-15`; the outputs are `gain * x`, so the gap is one rounding of each sum.
    let a_in = [1.0, -2.0, 0.5, 3.0, -0.25];
    let b_in = [0.3, 0.3, -1.0, 2.0, 4.0];
    let (mut ra, mut rb, mut rab) = (air(1.0, false), air(1.0, false), air(1.0, false));
    for i in 0..5 {
        let (pa, pb) = (ra.process(a_in[i]), rb.process(b_in[i]));
        let pab = rab.process(a_in[i] + b_in[i]);
        assert!((pab - (pa + pb)).abs() <= 1e-15, "sample {i}");
    }
}

#[test]
fn a_real_modal_body_radiates_its_own_volume_acceleration() {
    // Carried from `test_radiates_a_real_modal_body`. The Python compared `radiate(body)` with
    // `gain * body.pressure()` — the same number twice, `radiate` being `process(pressure())`.
    // The claim underneath is that what the air reads IS the body's volume acceleration, so that
    // is asserted against the second difference of the recorded states, formed here.
    let mut b = body(0.0);
    b.set_state(&[1e-3; 4], &[0.0; 4]);
    let mut a = air(1.0, false);
    let (k, w) = (b.params().k, b.params().a.clone());
    let mut peak = 0.0f64;
    for n in 0..2000 {
        let q_nm1 = b.q_prev().to_vec();
        let q_n = b.q().to_vec();
        b.step(0.0);
        let want = volume_accel(&w, b.q(), &q_n, &q_nm1, k);
        let p = a.process(b.pressure());
        assert_eq!(p, a.params().gain * want, "step {n}");
        peak = peak.max(p.abs());
    }
    assert!(peak > 0.0, "the body genuinely radiates");
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
    for fs in [0.0, -48000.0] {
        assert_eq!(
            AirParams::new(fs, 1.0, RHO0_AIR, C0_AIR, true),
            Err(AirError::NonPositiveFs)
        );
    }
    for r in [0.0, -1.0] {
        assert_eq!(
            AirParams::new(FS, r, RHO0_AIR, C0_AIR, true),
            Err(AirError::NonPositiveDistance)
        );
    }
    assert_eq!(
        AirParams::new(FS, 1.0, 0.0, C0_AIR, true),
        Err(AirError::NonPositiveRho0)
    );
    assert_eq!(
        AirParams::new(FS, 1.0, RHO0_AIR, 0.0, true),
        Err(AirError::NonPositiveC0)
    );
    assert_eq!(
        AirError::NonPositiveDistance.to_string(),
        "distance (listening radius r) must be positive."
    );
}

// -- tier 2: the closed-form resistances ---------------------------------------------------------

#[test]
fn the_monopole_resistance_is_the_free_space_value() {
    // Carried from `test_monopole_resistance_is_the_free_space_value`, in acoustic units.
    let omega = 2.0 * PI * 200.0;
    let r = radiation::monopole_radiation_resistance(omega, RHO0, C0);
    let want = RHO0 * omega * omega / (4.0 * PI * C0);
    assert!((r / want - 1.0).abs() <= 1e-14, "{r} vs {want}");
}

#[test]
fn the_baffled_piston_is_twice_the_free_space_monopole_in_the_rayleigh_limit() {
    // Carried from `test_piston_rayleigh_limit_is_twice_the_free_space_monopole`, as the link
    // between the two crates' functions: the analysis crate's own bar checks the piston against
    // the closed form, this one against THIS crate's monopole. ka ~ 9e-5.
    use physsynth_analysis::radiation::piston_radiation_resistance;
    let (omega, a) = (2.0 * PI * 5.0, 1e-3);
    let piston = piston_radiation_resistance(omega, a, RHO0, C0);
    let mono = radiation::monopole_radiation_resistance(omega, RHO0, C0);
    assert!((piston / (2.0 * mono) - 1.0).abs() <= 1e-6);
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
fn the_body_bleeds_entirely_into_the_radiated_channel() {
    // Carried from `test_body_energy_bleeds_entirely_into_the_radiated_channel`: lossless modes
    // at R = 3000, 6000 steps. The body has genuinely rung down, and every joule it lost is booked.
    let mut loaded = RadiatedBody::new(body(0.0), 3000.0).unwrap();
    loaded.set_state(&[1e-3; 4], &[0.0; 4]);
    let e_body0 = loaded.body().energy();
    for _ in 0..6000 {
        loaded.step(0.0);
    }
    let e_body = loaded.body().energy();
    assert!(e_body < 0.2 * e_body0, "{e_body} of {e_body0} left");
    let shed = e_body0 - e_body;
    assert!((loaded.radiated_energy / shed - 1.0).abs() <= 1e-10);
}

#[test]
fn the_radiation_load_is_passive() {
    // Carried from `test_radiation_load_is_passive`, at the Python's R = 2000: the body sheds
    // energy every step and the far field gains every step. A cross-time ripple of 1e-12 of the
    // start energy is allowed, as there.
    let mut loaded = RadiatedBody::new(body(0.0), 2000.0).unwrap();
    loaded.set_state(&[1e-3, 5e-4, -7e-4, 2e-4], &[0.0; 4]);
    let (mut body_e, mut rad_e) = (Vec::new(), Vec::new());
    for _ in 0..3000 {
        loaded.step(0.0);
        body_e.push(loaded.body().energy());
        rad_e.push(loaded.radiated_energy);
    }
    let tol = 1e-12 * body_e[0];
    for n in 1..body_e.len() {
        assert!(body_e[n] - body_e[n - 1] <= tol, "body gained at step {n}");
        assert!(
            rad_e[n] - rad_e[n - 1] >= -tol,
            "the far field gave back at step {n}"
        );
    }
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
fn the_total_is_conserved_at_an_enormous_resistance() {
    // Carried from `test_unconditionally_stable_at_enormous_R`: R = 1e12, far beyond anything
    // physical. Finite every step, the total still conserved, and the body critically over-damped
    // (its energy below the start) rather than growing.
    let mut loaded = RadiatedBody::new(body(0.0), 1e12).unwrap();
    loaded.set_state(&[1e-3; 4], &[0.0; 4]);
    let e0 = loaded.energy();
    for n in 0..2000 {
        loaded.step(0.0);
        assert!(loaded.body().q().iter().all(|v| v.is_finite()), "step {n}");
    }
    assert!((loaded.energy() - e0).abs() / e0 < 1e-10);
    assert!(loaded.body().energy() < e0);
}

#[test]
fn zero_resistance_is_bit_identical_to_a_bare_body() {
    // An exact reduction in the original, and the one that says the rank-1 correction is inert
    // when it should be — including the `_accel` rewrite, which `pressure()` reads. Run at the
    // crate's lossy unit rig AND at the Python's lossless mass-0.02 rig, where a dropped `/m` in
    // the correction is not the identity.
    for (sigma, mass) in [(1.5, 1.0), (0.0, 0.02)] {
        let mut loaded = RadiatedBody::new(body_at(FS, sigma, mass), 0.0).unwrap();
        let mut bare = body_at(FS, sigma, mass);
        let q0 = [1e-3, -5e-4, 3e-4, 8e-4];
        loaded.set_state(&q0, &[0.0; 4]);
        bare.set_state(&q0, &[0.0; 4]);
        for _ in 0..500 {
            loaded.step(0.0);
            bare.step(0.0);
            assert_eq!(loaded.body().q(), bare.q());
            assert_eq!(loaded.body().q_prev(), bare.q_prev());
            assert_eq!(loaded.pressure(), bare.pressure());
        }
        assert_eq!(loaded.radiated_energy, 0.0);
    }
}

#[test]
fn the_loaded_pressure_is_the_corrected_volume_acceleration() {
    // Carried from `test_loaded_body_radiates_through_the_air`, whose comment says `pressure()`
    // reflects the corrected (post-load) acceleration — and whose assertion compared
    // `radiate(loaded)` with `gain * loaded.pressure()`, the same number twice. The claim is
    // asserted here instead: the read-out equals the second difference of the CORRECTED states,
    // formed outside the model, for both loaded bodies. A step that forgot to refresh `accel`
    // after the rank-1 correction would report the free body's acceleration.
    let mut constant = RadiatedBody::new(body(0.0), 2000.0).unwrap();
    constant.set_state(&[1e-3; 4], &[0.0; 4]);
    let mut rational = reactive(0.0, 2000.0, 0.2);
    rational.set_state(&[1e-3; 4], &[0.0; 4]);
    let k = 1.0 / FS;
    let w = vec![1.0; 4];
    let mut a = air(1.0, false);
    for n in 0..1000 {
        let (c_nm1, c_n) = (
            constant.body().q_prev().to_vec(),
            constant.body().q().to_vec(),
        );
        let (r_nm1, r_n) = (
            rational.body().q_prev().to_vec(),
            rational.body().q().to_vec(),
        );
        constant.step(0.0);
        rational.step(0.0);
        let want_c = volume_accel(&w, constant.body().q(), &c_n, &c_nm1, k);
        let want_r = volume_accel(&w, rational.body().q(), &r_n, &r_nm1, k);
        assert!(
            (constant.pressure() - want_c).abs() <= 1e-12 * want_c.abs().max(1.0),
            "constant-R load, step {n}: {} vs {want_c}",
            constant.pressure()
        );
        assert!(
            (rational.pressure() - want_r).abs() <= 1e-12 * want_r.abs().max(1.0),
            "rational load, step {n}: {} vs {want_r}",
            rational.pressure()
        );
        assert_eq!(
            a.process(constant.pressure()),
            a.params().gain * constant.pressure()
        );
    }
}

/// One step of the dense coupled implicit solve the rank-1 path must equal:
///
/// ```text
/// [diag(1 + sigma k) + (k R/2)(a/m) a^T] q^{n+1}
///     = free_rhs + (k R/2)(a . q^{n-1})(a/m) + k^2 R u_l (a/m)
/// ```
///
/// with `free_rhs = 2 q^n - (1 - sigma k) q^{n-1} - k^2 omega^2 q^n` (the body's numerator). Solved
/// with `dense::lu_factor`/`lu_solve` — partial-pivoted Gaussian elimination, an algorithm that
/// shares nothing with the rank-1 update. `u_l = 0` for the constant-`R` load.
fn dense_coupled_step(b: &ModalBody, r: f64, u_l: f64) -> Vec<f64> {
    let p = b.params();
    let (k, n) = (p.k, p.n_modes());
    let (q, q_nm1) = (b.q(), b.q_prev());
    let a_dot_qnm1: f64 = (0..n).map(|i| p.a[i] * q_nm1[i]).sum();
    let mut mat = vec![0.0; n * n];
    let mut rhs = vec![0.0; n];
    for i in 0..n {
        let sk = p.sigma[i] * p.k;
        let am = p.a[i] / p.m[i];
        for j in 0..n {
            mat[i * n + j] = 0.5 * k * r * am * p.a[j];
        }
        mat[i * n + i] += 1.0 + sk;
        let free = 2.0 * q[i] - (1.0 - sk) * q_nm1[i] - k * k * p.omega[i] * p.omega[i] * q[i];
        rhs[i] = free + 0.5 * k * r * a_dot_qnm1 * am + k * k * r * u_l * am;
    }
    let lu = lu_factor(mat, n).expect("a square system");
    lu_solve(&lu, &rhs).expect("a nonsingular system")
}

#[test]
fn a_lossy_body_matches_the_exact_dense_coupled_solve() {
    // Carried from `test_lossy_body_matches_the_exact_dense_coupled_solve`. The `(1 + sigma k)` in
    // `G` and in the correction is invisible at sigma = 0, so every lossless bar leaves it free; a
    // wrong-but-consistent factor still conserves and still decays. This is the discriminating
    // check, at the Python's lossy, launched, mass-0.02 rig.
    let mut loaded = RadiatedBody::new(body_at(FS, 3.0, 0.02), 1500.0).unwrap();
    loaded.set_state(&[1e-3, -6e-4, 4e-4, 7e-4], &[0.2, -0.1, 0.05, 0.0]);
    let mut worst = 0.0f64;
    for _ in 0..20 {
        let reference = dense_coupled_step(loaded.body(), 1500.0, 0.0);
        loaded.step(0.0);
        worst = worst.max(nan_max(
            (0..4).map(|i| (loaded.body().q()[i] - reference[i]).abs()),
        ));
    }
    println!("worst gap to the dense solve: {worst:e}");
    assert!(worst <= 1e-13, "{worst:e}");
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
fn the_spheres_impedance_is_the_closed_form_pulsating_monopole() {
    // Carried from `test_from_sphere_impedance_is_the_closed_form_monopole`:
    // R = rho0 c0 / S, M_a = rho0 / (4 pi a), tau = a / c0, and
    // Z_a = (rho0 c0 / S) (j ka) / (1 + j ka) at three frequencies across ka = 1.
    let a = SPHERE;
    let load = sphere(FS, a);
    let p = load.params();
    let s = 4.0 * PI * a * a;
    let close = |x: f64, y: f64, rel: f64| (x / y - 1.0).abs() <= rel;
    assert!(close(p.r, RHO0 * C0 / s, 1e-14));
    assert!(close(p.m_a, RHO0 / (4.0 * PI * a), 1e-14));
    assert!(close(p.tau, a / C0, 1e-14));
    for f in [50.0, 400.0, 3000.0] {
        let omega = 2.0 * PI * f;
        let ka = omega * a / C0;
        let want = c_div((0.0, RHO0 * C0 / s * ka), (1.0, ka));
        let z = load.impedance(omega);
        let gap = ((z.0 - want.0).powi(2) + (z.1 - want.1).powi(2)).sqrt();
        let size = (want.0 * want.0 + want.1 * want.1).sqrt();
        assert!(gap <= 1e-13 * size, "{f} Hz: {z:?} vs {want:?}");
    }
}

#[test]
fn an_arbitrary_pair_has_no_radius_and_refuses_the_far_field() {
    let load = RationalAirLoad::new(LoadParams::new(FS, 2000.0, 0.2, RHO0_AIR, C0_AIR).unwrap());
    assert!(load.params().sphere_radius.is_none());
    assert_eq!(
        load.far_field_pressure(1.0, None),
        Err(FarFieldError::NotSphereConsistent)
    );
}

#[test]
fn a_sphere_serves_the_far_field_and_refuses_a_zero_distance() {
    // Carried from the second half of `test_far_field_refuses_an_inconsistent_load_but_serves_a_
    // sphere`: after one driven step, p_far(r) = (a / r) p_load, and r = 0 is refused.
    let mut load = sphere(FS, SPHERE);
    let radius = load
        .params()
        .sphere_radius
        .expect("a sphere load has a radius");
    assert!((radius / SPHERE - 1.0).abs() <= 1e-12);
    load.step(1e-3, 0.0);
    assert_ne!(load.pressure_load, 0.0);
    let far = load.far_field_pressure(2.0, None).unwrap();
    let want = SPHERE / 2.0 * load.pressure_load;
    assert!((far / want - 1.0).abs() <= 1e-14, "{far} vs {want}");
    assert_eq!(
        load.far_field_pressure(0.0, None),
        Err(FarFieldError::NonPositiveDistance)
    );
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
fn the_low_frequency_limit_is_exactly_the_monopole_over_one_plus_ka_squared() {
    // Carried from `test_low_frequency_limit_is_the_batch2_monopole_resistance`. Not only the
    // limit: Re Z_a = R_mono / (1 + (ka)^2) exactly, so the two tiers' resistances meet with a
    // KNOWN gap, below 2 (ka)^2, which shrinks as ka does.
    let load = sphere(FS, SPHERE);
    let mut gaps = Vec::new();
    for f in [80.0, 20.0, 5.0] {
        let omega = 2.0 * PI * f;
        let ka = omega * SPHERE / C0;
        let re = load.impedance(omega).0;
        let mono = radiation::monopole_radiation_resistance(omega, RHO0, C0);
        assert!(
            (re / (mono / (1.0 + ka * ka)) - 1.0).abs() <= 1e-13,
            "{f} Hz"
        );
        let gap = (re / mono - 1.0).abs();
        assert!(gap <= 2.0 * ka * ka, "{f} Hz: {gap:e}");
        gaps.push(gap);
    }
    assert!(gaps[0] > gaps[1] && gaps[1] > gaps[2], "{gaps:?}");
}

#[test]
fn the_high_frequency_limit_saturates_at_the_plane_wave_resistance() {
    // Carried from `test_high_frequency_limit_saturates_at_the_plane_wave_resistance`, at 1 MHz:
    // Re Z = R (ka)^2 / (1 + (ka)^2) exactly, approaching rho0 c0 / S from below as 1/(ka)^2,
    // and the reactance dying as R / ka.
    let load = sphere(FS, SPHERE);
    let plane = RHO0 * C0 / (4.0 * PI * SPHERE * SPHERE);
    let omega = 2.0 * PI * 1e6;
    let ka = omega * SPHERE / C0;
    let z = load.impedance(omega);
    assert!((z.0 / (plane / (1.0 + 1.0 / (ka * ka))) - 1.0).abs() <= 1e-13);
    assert!((z.0 / plane - 1.0).abs() <= 2.0 / (ka * ka));
    assert!((z.1 / (plane / ka) - 1.0).abs() <= 1e-5);
}

#[test]
fn a_constant_r_load_has_no_reactance_at_any_frequency() {
    let load =
        RationalAirLoad::new(LoadParams::new(FS, 2000.0, f64::INFINITY, RHO0_AIR, C0_AIR).unwrap());
    for w in [
        1.0,
        1e3,
        1e5,
        2.0 * PI * 10.0,
        2.0 * PI * 1000.0,
        2.0 * PI * 10000.0,
    ] {
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

/// Drive a load standalone at prescribed volume velocity `cos(omega n k)` and return `p / U` at
/// the drive frequency — the retired `_drive_and_measure_impedance`. `G = 0` is the prescribed
/// drive (a rigid piston whose `U` is imposed whatever the load does); `omega` must sit on a bin
/// of the `n_window`-sample window, and `warmup` lets the first-order transient die.
fn drive_and_measure(load: &mut RationalAirLoad, omega: f64, n_window: usize) -> (f64, f64) {
    let k = load.params().k;
    let mut n = 0usize;
    for _ in 0..8000 {
        load.step((omega * n as f64 * k).cos(), 0.0);
        n += 1;
    }
    let (mut pn, mut un) = ((0.0, 0.0), (0.0, 0.0));
    for i in 0..n_window {
        let u = (omega * n as f64 * k).cos();
        let (p, _) = load.step(u, 0.0);
        n += 1;
        // Single-bin DFT, `exp(-j omega k i)`.
        let th = omega * k * i as f64;
        pn = (pn.0 + p * th.cos(), pn.1 - p * th.sin());
        un = (un.0 + u * th.cos(), un.1 - u * th.sin());
    }
    c_div(pn, un)
}

#[test]
fn the_measured_impedance_is_the_prewarped_closed_form() {
    // Carried from `test_impedance_sweep_matches_the_prewarped_closed_form` — the spectral money
    // test. The trapezoid on the inertance IS the bilinear transform, so the scheme realises Z_a at
    // s = (2j/k) tan(omega k / 2); driven at prescribed U, the measured p/U is that value to
    // machine precision, magnitude AND phase, at four bins from 150 Hz to 16 kHz.
    let n_window = 4096;
    let mut load = sphere(FS, SPHERE);
    for bin in [13, 137, 613, 1361] {
        let omega = 2.0 * PI * FS * bin as f64 / n_window as f64;
        load.reset();
        let z = drive_and_measure(&mut load, omega, n_window);
        let want = load.impedance_discrete(omega);
        let gap = ((z.0 - want.0).powi(2) + (z.1 - want.1).powi(2)).sqrt();
        let rel = gap / (want.0 * want.0 + want.1 * want.1).sqrt();
        println!("bin {bin}: {z:?} vs {want:?}, {rel:e}");
        assert!(rel <= 1e-11, "bin {bin}: {rel:e}");
    }
}

#[test]
fn the_prewarping_gap_is_second_order_and_its_size_is_known() {
    // Carried from `test_the_prewarping_gap_is_the_honest_discretisation_error`. The gap between
    // the scheme's impedance and the physics is the bilinear warp, O((omega k)^2): halving k
    // quarters it, and 2 kHz at 48 kHz costs 0.27%.
    let omega = 2.0 * PI * 2000.0;
    let gaps: Vec<f64> = [48000.0, 96000.0, 192000.0]
        .iter()
        .map(|&fs| {
            let load = sphere(fs, SPHERE);
            let (zd, zc) = (load.impedance_discrete(omega), load.impedance(omega));
            let gap = ((zd.0 - zc.0).powi(2) + (zd.1 - zc.1).powi(2)).sqrt();
            gap / (zc.0 * zc.0 + zc.1 * zc.1).sqrt()
        })
        .collect();
    for w in gaps.windows(2) {
        let ratio = w[0] / w[1];
        assert!(3.5 < ratio && ratio < 4.5, "{gaps:?}");
    }
    assert!(2.0e-3 < gaps[0] && gaps[0] < 3.5e-3, "{gaps:?}");
}

#[test]
fn the_discrete_impedance_converges_to_the_continuous_one() {
    // Carried from `test_the_discrete_impedance_converges_to_the_continuous_one`: at 4 MHz the
    // warp at 1 kHz is (omega k)^2 / 12 ~ 2e-7, so the two coincide to 1e-6.
    let fine = sphere(4.0e6, SPHERE);
    let omega = 2.0 * PI * 1000.0;
    let (zd, zc) = (fine.impedance_discrete(omega), fine.impedance(omega));
    let gap = ((zd.0 - zc.0).powi(2) + (zd.1 - zc.1).powi(2)).sqrt();
    assert!(gap <= 1e-6 * (zc.0 * zc.0 + zc.1 * zc.1).sqrt());
}

#[test]
fn the_load_refuses_every_non_physical_parameter() {
    for fs in [0.0, -48000.0] {
        assert_eq!(
            LoadParams::new(fs, 1.0, 1.0, RHO0_AIR, C0_AIR),
            Err(LoadError::NonPositiveFs)
        );
    }
    assert_eq!(
        LoadParams::new(FS, -1.0, 1.0, RHO0_AIR, C0_AIR),
        Err(LoadError::NegativeR)
    );
    for m_a in [0.0, -1.0] {
        assert_eq!(
            LoadParams::new(FS, 1.0, m_a, RHO0_AIR, C0_AIR),
            Err(LoadError::NonPositiveMass)
        );
    }
    // NaN too: the original writes `not (M_a > 0.0)` exactly so this is caught.
    assert_eq!(
        LoadParams::new(FS, 1.0, f64::NAN, RHO0_AIR, C0_AIR),
        Err(LoadError::NonPositiveMass)
    );
    // Both halves of the medium check.
    assert_eq!(
        LoadParams::new(FS, 1.0, 1.0, 0.0, C0_AIR),
        Err(LoadError::NonPositiveMedium)
    );
    assert_eq!(
        LoadParams::new(FS, 1.0, 1.0, RHO0_AIR, -1.0),
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
    for mass in [0.0, -1.0] {
        assert_eq!(
            load.loaded_mode(1.0, 0.02, mass, 50, 1e-14),
            Err(LoadedModeError::NonPositiveMass)
        );
    }
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

#[test]
fn loaded_mode_is_refused_starved_and_converges_on_the_budget_its_callers_use() {
    // Carried from `test_loaded_mode_refuses_to_return_an_unconverged_iterate`, at its rig: the
    // 440 Hz mode at weight 0.02, mass 0.02. Starved at two iterations it refuses (it needs seven);
    // at 50 iterations and tol 1e-14 — the budget the binding's signature defaulted to and the
    // viewer's `airload` passes as literals; Rust has no default — it converges, below omega0.
    let load = sphere(FS, SPHERE);
    let w0 = 2.0 * PI * 440.0;
    let err = load.loaded_mode(w0, 0.02, 0.02, 2, 1e-14).unwrap_err();
    assert!(matches!(
        err,
        LoadedModeError::NotConverged { iterations: 2, .. }
    ));
    assert!(err
        .to_string()
        .starts_with("loaded_mode did not converge in 2 iterations"));
    let (w, alpha) = load.loaded_mode(w0, 0.02, 0.02, 50, 1e-14).unwrap();
    assert!(0.0 < w && w < w0 && alpha > 0.0, "{w} {alpha}");
}

// -- tier 3: the loaded body ---------------------------------------------------------------------

#[test]
fn the_three_way_energy_identity_holds_for_a_lossless_body() {
    // E_body + stored (radiation mass) + radiated = const. The stored term is what is new here:
    // this air gives back as well as taking, so neither half alone is monotone. The crate's
    // (1500, 0.03) plus the Python's four pairs — the last is the 5 cm sphere's own.
    for (r, m_a) in [
        (1500.0, 0.03),
        (2000.0, 0.2),
        (2000.0, 0.02),
        (5e4, 1.0),
        (13146.0, 1.916),
    ] {
        let mut loaded = reactive(0.0, r, m_a);
        loaded.set_state(&plucked(0.0), &[0.0; 4]);
        let e0 = loaded.energy();
        let mut peak = 0.0f64;
        for _ in 0..4000 {
            loaded.step(0.0);
            peak = nan_max([peak, (loaded.energy() - e0).abs() / e0]);
        }
        assert!(peak < 1e-10, "(R, M_a) = ({r}, {m_a}): {peak:e}");
        assert!(loaded.load().radiated_energy > 0.0);
        assert!(loaded.load().stored_energy() > 0.0);
    }
}

#[test]
fn the_air_stores_as_well_as_dissipates() {
    // Carried from `test_the_air_stores_as_well_as_dissipates`: the stored term rises AND falls,
    // repeatedly (the Python measured 1,507 rises in 2,999 steps), not as a rounding ripple.
    let mut loaded = reactive(0.0, 2000.0, 0.02);
    loaded.set_state(&[1e-3; 4], &[0.0; 4]);
    let mut stored = Vec::new();
    for _ in 0..3000 {
        loaded.step(0.0);
        stored.push(loaded.load().stored_energy());
    }
    let rises = stored.windows(2).filter(|w| w[1] > w[0]).count();
    let falls = stored.windows(2).filter(|w| w[1] < w[0]).count();
    assert!(rises > 0 && falls > 0);
    assert!(
        rises as f64 > 0.2 * (stored.len() - 1) as f64,
        "{rises} rises"
    );
}

#[test]
fn the_reactive_load_is_passive() {
    // Carried from `test_reactive_load_is_passive`: body + stored air sheds every step, and the
    // far field only gains, to 1e-12 of the start.
    let mut loaded = reactive(0.0, 2000.0, 0.05);
    loaded.set_state(&[1e-3, 5e-4, -7e-4, 2e-4], &[0.0; 4]);
    let (mut total, mut rad) = (Vec::new(), Vec::new());
    for _ in 0..3000 {
        loaded.step(0.0);
        total.push(loaded.body().energy() + loaded.load().stored_energy());
        rad.push(loaded.load().radiated_energy);
    }
    let tol = 1e-12 * total[0];
    for n in 1..total.len() {
        assert!(
            total[n] - total[n - 1] <= tol,
            "body + air gained at step {n}"
        );
        assert!(
            rad[n] - rad[n - 1] >= -tol,
            "the far field gave back at step {n}"
        );
    }
}

#[test]
fn the_reactive_load_is_unconditionally_stable() {
    // Carried from `test_unconditionally_stable_at_extreme_parameters`: 1 + R_eff G >= 1 for every
    // R >= 0, M_a > 0, k — no CFL, nothing to tune.
    for (r, m_a) in [(1e12, 1e-9), (1e12, 1e9), (1e-9, 1e-9)] {
        let mut loaded = reactive(0.0, r, m_a);
        loaded.set_state(&[1e-3; 4], &[0.0; 4]);
        let e0 = loaded.energy();
        for n in 0..2000 {
            loaded.step(0.0);
            assert!(
                loaded.body().q().iter().all(|v| v.is_finite()),
                "({r}, {m_a}) step {n}"
            );
        }
        assert!((loaded.energy() - e0).abs() / e0 < 1e-10, "({r}, {m_a})");
        assert!(loaded.body().energy() <= e0 * (1.0 + 1e-10), "({r}, {m_a})");
    }
}

#[test]
fn an_infinite_radiation_mass_is_bit_identical_to_the_constant_r_load() {
    // The second exact reduction, and the reason both classes call one `rank_one`. Anything that
    // reassociated `solve` — or recomputed `_G` a second time — would break this and nothing else.
    // At the crate's unit rig and at the Python's mass-0.02 rig, every read-out compared.
    let r = 2000.0;
    for mass in [1.0, 0.02] {
        let mut batch3 =
            ReactiveRadiatedBody::new(body_at(FS, 0.0, mass), pair(FS, r, f64::INFINITY)).unwrap();
        let mut batch2 = RadiatedBody::new(body_at(FS, 0.0, mass), r).unwrap();
        let q0 = [1e-3, -5e-4, 3e-4, 8e-4];
        batch3.set_state(&q0, &[0.0; 4]);
        batch2.set_state(&q0, &[0.0; 4]);
        for _ in 0..1000 {
            batch3.step(0.0);
            batch2.step(0.0);
            assert_eq!(batch3.body().q(), batch2.body().q());
            assert_eq!(batch3.body().q_prev(), batch2.body().q_prev());
            assert_eq!(batch3.pressure(), batch2.pressure());
            assert_eq!(batch3.load().radiated_energy, batch2.radiated_energy);
            assert_eq!(batch3.energy(), batch2.energy());
        }
        // The auxiliary state never moved: k p / inf is exactly 0.0, and inf * 0 is special-cased.
        assert_eq!(batch3.load().u_l, 0.0);
        assert_eq!(batch3.load().stored_energy(), 0.0);
    }
}

#[test]
fn zero_resistance_is_bit_identical_to_a_bare_body_here_too() {
    for (sigma, mass) in [(1.5, 1.0), (0.0, 0.02)] {
        let mut loaded =
            ReactiveRadiatedBody::new(body_at(FS, sigma, mass), pair(FS, 0.0, 0.2)).unwrap();
        let mut bare = body_at(FS, sigma, mass);
        let q0 = [1e-3, -5e-4, 3e-4, 8e-4];
        loaded.set_state(&q0, &[0.0; 4]);
        bare.set_state(&q0, &[0.0; 4]);
        for _ in 0..500 {
            loaded.step(0.0);
            bare.step(0.0);
            assert_eq!(loaded.body().q(), bare.q());
            assert_eq!(loaded.pressure(), bare.pressure());
        }
        // No radiated energy, no stored energy.
        assert_eq!(loaded.energy(), bare.energy());
    }
}

#[test]
fn a_lossy_reactive_body_matches_the_exact_dense_coupled_solve_at_two_rates() {
    // Carried from `test_lossy_reactive_body_matches_the_exact_dense_coupled_solve`. Two factors
    // no lossless single-rate bar can pin: the `(1 + sigma k)` in G and the correction, and
    // R_eff = R / (1 + k R / (2 M_a))'s dependence on k. At M_a = 0.03 R_eff is 34-44% below R.
    for fs in [48000.0, 32000.0] {
        let load = pair(fs, 1500.0, 0.03);
        assert!(load.params().r_eff < 0.75 * load.params().r);
        let mut loaded = ReactiveRadiatedBody::new(body_at(fs, 3.0, 0.02), load).unwrap();
        loaded.set_state(&[1e-3, -6e-4, 4e-4, 7e-4], &[0.2, -0.1, 0.05, 0.0]);
        let r_eff = loaded.load().params().r_eff;
        let mut worst = 0.0f64;
        for _ in 0..20 {
            let reference = dense_coupled_step(loaded.body(), r_eff, loaded.load().u_l);
            loaded.step(0.0);
            worst = worst.max(nan_max(
                (0..4).map(|i| (loaded.body().q()[i] - reference[i]).abs()),
            ));
        }
        println!("fs {fs}: worst gap to the dense solve {worst:e}");
        assert!(worst <= 1e-13, "fs {fs}: {worst:e}");
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
fn reset_clears_the_auxiliary_state_and_both_channels() {
    // Carried from `test_reset_clears_the_auxiliary_state_and_the_channels`.
    let mut load = sphere(FS, SPHERE);
    let k = load.params().k;
    for n in 0..200 {
        load.step((2.0 * PI * 300.0 * n as f64 * k).cos(), 0.0);
    }
    assert!(load.u_l != 0.0 && load.radiated_energy > 0.0);
    load.reset();
    assert_eq!(load.u_l, 0.0);
    assert_eq!(load.radiated_energy, 0.0);
    assert_eq!(load.energy(), 0.0);
    assert_eq!(load.n(), 0);
}

// -- tier 3: the far field -----------------------------------------------------------------------

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

#[test]
fn the_far_field_power_balances_a_driven_sphere_at_steady_state() {
    // Carried from `test_far_field_power_balances_the_booked_radiated_power`, at its rig: the 5 cm
    // sphere driven standalone at ~1.6 kHz (bin 137 of 4096) past its transient, a listening
    // sphere at 3 m. The balance is exact at every ka, so it pins the geometry constants (a / r,
    // S, the 4 pi) that no energy identity can see.
    let (n_window, r) = (4096usize, 3.0);
    let mut load = sphere(FS, SPHERE);
    let omega = 2.0 * PI * FS * 137.0 / n_window as f64;
    let k = load.params().k;
    let mut n = 0usize;
    for _ in 0..8000 {
        load.step((omega * n as f64 * k).cos(), 0.0);
        n += 1;
    }
    let e0 = load.radiated_energy;
    let mut far = 0.0;
    for _ in 0..n_window {
        load.step((omega * n as f64 * k).cos(), 0.0);
        let p_far = load.far_field_pressure(r, None).unwrap();
        far += 4.0 * PI * r * r * p_far * p_far / (RHO0 * C0) * k;
        n += 1;
    }
    let booked = load.radiated_energy - e0;
    println!("far field {far:?} vs booked {booked:?}");
    assert!((far / booked - 1.0).abs() <= 1e-12, "{far} vs {booked}");
}

#[test]
fn the_compact_read_out_overstates_a_finite_spheres_far_field_by_one_plus_ka_squared() {
    // Carried from `test_the_compact_read_out_overstates_the_far_field_of_a_finite_sphere`. The
    // compact read-out is the a -> 0 limit; a 15 cm sphere's far field carries an extra
    // 1 / (1 + j ka), so at ~797 Hz (ka ~ 2.2) the two mean squares differ by 1 + (ka)^2 ~ 5.8.
    // Not a bug in either node: they are different limits.
    let (n_window, a, r) = (4096usize, 0.15, 3.0);
    let mut load = sphere(FS, a);
    let mut compact = air(r, false);
    let omega = 2.0 * PI * FS * 68.0 / n_window as f64;
    let ka = omega * a / C0;
    let k = load.params().k;
    let mut n = 0usize;
    for _ in 0..8000 {
        load.step((omega * n as f64 * k).cos(), 0.0);
        n += 1;
    }
    let (mut sphere_ms, mut compact_ms) = (0.0, 0.0);
    for _ in 0..n_window {
        load.step((omega * n as f64 * k).cos(), 0.0);
        // The same monopole read out compactly: Q'' = dU/dt exactly.
        let c = compact.process(-omega * (omega * n as f64 * k).sin());
        let s = load.far_field_pressure(r, None).unwrap();
        sphere_ms += s * s;
        compact_ms += c * c;
        n += 1;
    }
    let ratio = compact_ms / sphere_ms;
    println!("ratio {ratio} vs 1 + (ka)^2 = {}", 1.0 + ka * ka);
    assert!((ratio / (1.0 + ka * ka) - 1.0).abs() <= 0.02);
}

// -- tier 3: the loaded mode, measured -----------------------------------------------------------

/// What [`measure_single_mode`] reads off one loaded mode.
struct ModeRun {
    /// Frequency from the zero crossings (Hz).
    f: f64,
    /// Decay rate from the envelope peaks (1/s).
    alpha: f64,
    /// The last displacement — to tell a trajectory difference from a fit difference.
    q_last: f64,
}

/// The retired `_measure_single_mode`: excite ONE mode of a loaded body (`q0 = 1e-3`, lossless),
/// run 0.6 s, and read its frequency from the zero crossings and its decay rate from a straight
/// line through the log of the *envelope peaks*.
///
/// The slope is `np.polyfit(t[peaks], log(env[peaks]), 1)[0]` written as the centred normal
/// equation — not NumPy's algorithm (a scaled-Vandermonde least squares), so agreement with it is
/// a tolerance, certified in [`the_decay_fit_reproduces_numpys_polyfit`].
fn measure_single_mode(f0: f64, load: RationalAirLoad, weight: f64, mass: f64) -> ModeRun {
    let b = ModalBody::new(
        BodyParams::new(
            vec![f0],
            FS,
            vec![0.0],
            vec![mass],
            vec![1.0],
            Some(vec![weight]),
        )
        .expect("a well-posed single mode"),
    );
    let mut loaded = ReactiveRadiatedBody::new(b, load).expect("matching timesteps");
    loaded.set_state(&[1e-3], &[0.0]);
    let steps = (0.6 * FS) as usize;
    let q: Vec<f64> = (0..steps)
        .map(|_| {
            loaded.step(0.0);
            loaded.body().q()[0]
        })
        .collect();
    // `np.signbit` — the sign bit, so -0.0 counts as negative.
    let crossings: Vec<usize> = (0..steps - 1)
        .filter(|&i| q[i].is_sign_negative() != q[i + 1].is_sign_negative())
        .collect();
    let (c0, c1) = (crossings[0], crossings[crossings.len() - 1]);
    let f = 0.5 * FS * (crossings.len() - 1) as f64 / (c1 - c0) as f64;
    let env: Vec<f64> = q.iter().map(|v| v.abs()).collect();
    let peaks: Vec<usize> = (1..steps - 1)
        .filter(|&i| env[i] > env[i - 1] && env[i] >= env[i + 1])
        .collect();
    let xs: Vec<f64> = peaks.iter().map(|&i| i as f64 / FS).collect();
    let ys: Vec<f64> = peaks.iter().map(|&i| env[i].ln()).collect();
    let len = xs.len() as f64;
    let (xm, ym) = (xs.iter().sum::<f64>() / len, ys.iter().sum::<f64>() / len);
    let (mut sxy, mut sxx) = (0.0, 0.0);
    for i in 0..xs.len() {
        sxy += (xs[i] - xm) * (ys[i] - ym);
        sxx += (xs[i] - xm) * (xs[i] - xm);
    }
    ModeRun {
        f,
        alpha: -(sxy / sxx),
        q_last: q[steps - 1],
    }
}

/// `(f0, f_meas, alpha, q_last)` as the retired Python measured them with `np.polyfit`, recorded
/// 2026-10-07 with the wheel freshly reinstalled (`W:\temp\claude\batch23\record.txt`). The first
/// six are the 5 cm sphere at weight 0.02 / mass 0.02; the last four the flat control
/// (`R = Re Z_a(110 Hz)`, `M_a = inf`).
const NUMPY_MODES: [(f64, f64, f64, f64); 10] = [
    (
        110.0,
        107.97237915881983,
        1.2271086987067534,
        9.655156998093963e-05,
    ),
    (
        196.0,
        192.45519838226065,
        3.8242178342370923,
        -9.957807780057588e-05,
    ),
    (
        261.0,
        256.38527385343446,
        6.642873502644783,
        9.228846210739158e-06,
    ),
    (
        440.0,
        432.8992269656661,
        17.37028268197991,
        -7.483805290551975e-10,
    ),
    (
        220.0,
        216.05024424284719,
        4.784637688370171,
        -3.8315560165107274e-05,
    ),
    (
        880.0,
        870.1939790029896,
        50.41563334522875,
        5.0177224996017765e-17,
    ),
    (
        110.0,
        110.00314894510339,
        1.3210458309592772,
        0.00045265483839767787,
    ),
    (
        220.0,
        220.00697107005925,
        1.3210424399783058,
        0.0004524853931000492,
    ),
    (
        440.0,
        440.0681952611252,
        1.3210443298585097,
        0.00044086869140177553,
    ),
    (
        880.0,
        880.4891606448027,
        1.321040994445677,
        -0.00011890758294187742,
    ),
];

/// The flat control's resistance: the sphere's `Re Z_a` at 110 Hz, so the two loads agree on the
/// fundamental's damping and on nothing else.
fn flat_control() -> RationalAirLoad {
    let r_const = sphere(FS, SPHERE).impedance(2.0 * PI * 110.0).0;
    pair(FS, r_const, f64::INFINITY)
}

#[test]
fn the_decay_fit_reproduces_numpys_polyfit() {
    // NOT a physics bar: the certificate for [`measure_single_mode`]'s fit, the transcription of
    // the one NumPy number the loaded-mode bars retired. The binding assembled its own step, so the
    // trajectory is compared too (`q_last`, and the crossing-count frequency, which is exact
    // integer arithmetic on it); the fit is then compared on the same data.
    let mut worst = 0.0f64;
    for (i, &(f0, f_np, alpha_np, q_np)) in NUMPY_MODES.iter().enumerate() {
        let load = if i < 6 {
            sphere(FS, SPHERE)
        } else {
            flat_control()
        };
        let run = measure_single_mode(f0, load, 0.02, 0.02);
        let rel = (run.alpha / alpha_np - 1.0).abs();
        println!(
            "{f0} Hz: q_last {:?} vs {q_np:?}, f {:?} vs {f_np:?}, alpha rel {rel:e}",
            run.q_last, run.f
        );
        assert!(
            (run.q_last - q_np).abs() <= 1e-12 * 1e-3,
            "{f0} Hz: trajectory differs"
        );
        assert!(
            (run.f / f_np - 1.0).abs() <= 1e-15,
            "{f0} Hz: crossings differ"
        );
        worst = worst.max(rel);
        assert!(rel <= 1e-9, "{f0} Hz: fit off NumPy's by {rel:e}");
    }
    println!("worst alpha gap to polyfit: {worst:e}");
}

#[test]
fn a_loaded_mode_matches_the_closed_form_in_both_parts_of_z() {
    // Carried from `test_a_loaded_mode_matches_the_closed_form_in_both_parts_of_Z` — THE physics
    // claim of the rational tier, and it needs both parts of the impedance: Im Z_a is an added mass
    // (the pitch DROPS), Re Z_a damps at the shifted frequency, alpha = a^2 Re Z_a / (2 m_eff).
    for f0 in [110.0, 196.0, 261.0, 440.0] {
        let load = sphere(FS, SPHERE);
        let (w_pred, a_pred) = load
            .loaded_mode(2.0 * PI * f0, 0.02, 0.02, 50, 1e-14)
            .unwrap();
        let run = measure_single_mode(f0, load, 0.02, 0.02);
        let f_pred = w_pred / (2.0 * PI);
        assert!(
            (run.f / f_pred - 1.0).abs() <= 1e-3,
            "{f0}: f {} vs {f_pred}",
            run.f
        );
        assert!(
            (run.alpha / a_pred - 1.0).abs() <= 0.02,
            "{f0}: alpha {} vs {a_pred}",
            run.alpha
        );
        assert!(run.f < 0.995 * f0, "{f0}: the air did not flatten it");
        assert!(
            run.alpha / (2.0 * PI * run.f) < 0.02,
            "{f0}: not weakly loaded"
        );
    }
}

#[test]
fn higher_partials_radiate_better_and_die_first() {
    // Carried from `test_higher_partials_radiate_better_and_die_first`: over three octaves the
    // decay rate climbs the way Re Z_a(omega) says, monotone and by more than 20x.
    let freqs = [110.0, 220.0, 440.0, 880.0];
    let alpha: Vec<f64> = freqs
        .iter()
        .map(|&f| measure_single_mode(f, sphere(FS, SPHERE), 0.02, 0.02).alpha)
        .collect();
    for w in alpha.windows(2) {
        assert!(w[1] > w[0], "{alpha:?}");
    }
    assert!(alpha[3] / alpha[0] > 20.0, "{alpha:?}");
    let load = sphere(FS, SPHERE);
    for (i, &f) in freqs.iter().enumerate() {
        let predicted = load
            .loaded_mode(2.0 * PI * f, 0.02, 0.02, 50, 1e-14)
            .unwrap()
            .1;
        // `np.allclose(rtol=0.02)` keeps atol = 1e-8, which is 1e-8 of a rate near 1 /s: nothing.
        assert!(
            (alpha[i] - predicted).abs() <= 0.02 * predicted.abs(),
            "{f} Hz"
        );
    }
}

#[test]
fn a_constant_r_load_cannot_bend_with_frequency() {
    // Carried from `test_a_constant_R_load_cannot_bend_with_frequency` — the negative control. The
    // constant-R load matched at the fundamental damps EVERY mode at that rate, shifts no pitch
    // (Im Z = 0), and under-damps the top mode by more than 10x.
    let freqs = [110.0, 220.0, 440.0, 880.0];
    let runs: Vec<ModeRun> = freqs
        .iter()
        .map(|&f| measure_single_mode(f, flat_control(), 0.02, 0.02))
        .collect();
    for (run, &f) in runs.iter().zip(&freqs) {
        assert!(
            (run.alpha - runs[0].alpha).abs() <= 0.02 * runs[0].alpha.abs(),
            "{f} Hz"
        );
        assert!(
            (run.f / f - 1.0).abs() <= 2e-3,
            "{f} Hz: pitch moved to {}",
            run.f
        );
    }
    let top_true = sphere(FS, SPHERE)
        .loaded_mode(2.0 * PI * 880.0, 0.02, 0.02, 50, 1e-14)
        .unwrap()
        .1;
    assert!(
        runs[3].alpha < 0.1 * top_true,
        "{} vs {top_true}",
        runs[3].alpha
    );
}
