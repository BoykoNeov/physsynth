//! `engine::simulate` — the driver's bookkeeping, asserted against a resonator whose every number
//! is known, and then against a real string's energy bar.

use physsynth_core::engine::{simulate, Resonator, SimResult};
use physsynth_core::string_ideal::{Boundary, IdealString, Params};

/// A counter pretending to be a resonator: state `[n, 2n]`, energy `n`, timestep 0.25.
struct Counter {
    n: usize,
}

impl Resonator for Counter {
    fn step(&mut self) -> Result<(), String> {
        if self.n == 99 {
            return Err("the counter refuses step 100".to_owned());
        }
        self.n += 1;
        Ok(())
    }
    fn energy(&self) -> f64 {
        self.n as f64
    }
    fn state(&self) -> Vec<f64> {
        vec![self.n as f64, 2.0 * self.n as f64]
    }
    fn displacement_at(&self, index: usize) -> f64 {
        self.state()[index]
    }
    fn timestep(&self) -> f64 {
        0.25
    }
}

#[test]
fn arrays_include_the_initial_state_and_are_num_steps_plus_one_long() {
    let r = simulate(&mut Counter { n: 0 }, 5, Some(1), 0).unwrap();
    assert_eq!(r.energy, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    assert_eq!(r.output.unwrap(), vec![0.0, 2.0, 4.0, 6.0, 8.0, 10.0]);
    assert_eq!(r.time, vec![0.0, 0.25, 0.5, 0.75, 1.0, 1.25]);
    assert_eq!(r.fs, 4.0);
    assert!(r.snapshots.is_empty());
}

#[test]
fn snapshots_start_at_zero_and_land_on_multiples_of_the_stride() {
    let r = simulate(&mut Counter { n: 0 }, 7, None, 3).unwrap();
    let steps: Vec<usize> = r.snapshots.iter().map(|s| s.0).collect();
    assert_eq!(steps, vec![0, 3, 6]);
    assert_eq!(r.snapshots[2].1, vec![6.0, 12.0]); // a copy of the state AT that step
    assert!(r.output.is_none());
}

#[test]
fn a_failed_step_stops_the_run_with_the_models_own_message() {
    let err = simulate(&mut Counter { n: 0 }, 200, None, 0).unwrap_err();
    assert_eq!(err, "the counter refuses step 100");
}

#[test]
fn zero_steps_is_refused_with_the_originals_message() {
    let err = simulate(&mut Counter { n: 0 }, 0, None, 0).unwrap_err();
    assert_eq!(err, "num_steps must be >= 1.");
}

fn result(energy: Vec<f64>) -> SimResult {
    SimResult {
        time: vec![0.0; energy.len()],
        energy,
        output: None,
        fs: 1.0,
        snapshots: Vec::new(),
    }
}

#[test]
fn energy_drift_is_relative_and_falls_back_to_absolute_at_zero() {
    assert_eq!(result(vec![2.0, 2.5, 1.0]).energy_drift(), 0.5);
    assert_eq!(result(vec![0.0, -3.0, 1.0]).energy_drift(), 3.0);
}

#[test]
fn energy_drift_propagates_a_nan_rather_than_reporting_a_clean_run() {
    // `fold(f64::max)` would skip the NaN and report 0.0 — a blown-up run passing the bar.
    assert!(result(vec![1.0, f64::NAN, 1.0]).energy_drift().is_nan());
}

#[test]
fn a_lossless_ideal_string_conserves_through_the_driver() {
    let p = Params::new(
        1.0,
        200.0,
        0.005,
        12_800.0,
        64,
        0.0,
        Some((Boundary::Fixed, Boundary::Fixed)),
    )
    .unwrap();
    let mut s = IdealString::new(p.clone());
    let x = p.grid();
    let u0: Vec<f64> = x
        .iter()
        .map(|&xv| if xv <= 0.3 { xv / 0.3 } else { (1.0 - xv) / 0.7 } * 1e-3)
        .collect();
    s.set_displacement(&u0);
    let r = simulate(&mut s, 3_000, Some(6), 0).unwrap();
    assert!(r.energy_drift() < 1e-10, "{}", r.energy_drift());
    assert_eq!(r.fs, 12_800.0);
}
