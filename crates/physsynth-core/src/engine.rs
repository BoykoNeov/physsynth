//! The time-stepping driver — `physsynth/core/engine.py`'s `simulate`, native.
//!
//! The Python original is thirty lines and a `Protocol`: step a resonator `num_steps` times,
//! recording `energy()` every step (always — it is the primary bug detector), an optional pickup
//! signal, and an optional snapshot of the full state every `snapshot_stride` steps. It lived in
//! `core/` because it is part of the `exciter -> resonator -> output` abstraction rather than of
//! any client, and it is here for the same reason: the viewer is its first native caller
//! (`docs/dev/python-retirement-plan.md` §23), and the physics bars that still drive it from Python
//! will be its next.
//!
//! The arrays include the initial state at index 0, so they are `num_steps + 1` long — the
//! original's convention, kept so that a decimation index computed against one is valid against
//! the other.

use crate::string_damped::DampedStiffString;
use crate::string_ideal::IdealString;
use crate::string_stiff::StiffString;

/// What [`simulate`] needs from a resonator — `engine.py`'s `Resonator` protocol.
///
/// `state` returns an owned copy, as the Python property did (`resonator.state` is a fresh array
/// per read), so a snapshot can never alias the live field.
pub trait Resonator {
    /// Advance one timestep.
    fn step(&mut self);
    /// Discrete energy `E^n` (Joules).
    fn energy(&self) -> f64;
    /// A copy of the displacement field.
    fn state(&self) -> Vec<f64>;
    /// Displacement at grid node `index`.
    fn displacement_at(&self, index: usize) -> f64;
    /// The timestep `k = 1 / fs` (s).
    fn timestep(&self) -> f64;
}

/// Captured diagnostics from a run.
#[derive(Debug, Clone)]
pub struct SimResult {
    /// `i * k` for `i` in `0..=num_steps`, seconds.
    pub time: Vec<f64>,
    /// `E^n` at each step.
    pub energy: Vec<f64>,
    /// The pickup displacement, if one was requested.
    pub output: Option<Vec<f64>>,
    /// `1 / k` — the resonator's rate, not whatever the caller asked for.
    pub fs: f64,
    /// `(step, state)` every `snapshot_stride` steps, starting with step 0.
    pub snapshots: Vec<(usize, Vec<f64>)>,
}

impl SimResult {
    /// `max|E^n - E^0| / |E^0|` — the lossless-conservation figure of merit.
    ///
    /// When `E^0 == 0` it is `max|E^n|` instead, as in the original. A NaN anywhere in the trace
    /// makes the result NaN — `np.max` propagates it, and a `fold(f64::max)` would silently drop it
    /// and report a clean drift for a run that blew up.
    pub fn energy_drift(&self) -> f64 {
        let e0 = self.energy[0];
        if e0 == 0.0 {
            return nan_max(self.energy.iter().map(|e| e.abs()));
        }
        nan_max(self.energy.iter().map(|e| (e - e0).abs())) / e0.abs()
    }
}

/// `np.max` over a non-empty sequence: NaN-propagating, unlike `f64::max`.
fn nan_max(it: impl Iterator<Item = f64>) -> f64 {
    let mut m = f64::NEG_INFINITY;
    for v in it {
        if v.is_nan() {
            return f64::NAN;
        }
        if v > m {
            m = v;
        }
    }
    m
}

/// Run `resonator` for `num_steps` steps, capturing energy (always) and optionally a pickup signal
/// and periodic state snapshots (`snapshot_stride == 0` means none).
///
/// Refuses `num_steps < 1` with the original's message.
pub fn simulate<R: Resonator + ?Sized>(
    resonator: &mut R,
    num_steps: usize,
    pickup_index: Option<usize>,
    snapshot_stride: usize,
) -> Result<SimResult, String> {
    if num_steps < 1 {
        return Err("num_steps must be >= 1.".to_owned());
    }
    let n = num_steps + 1;
    let mut energy = Vec::with_capacity(n);
    let mut output = pickup_index.map(|_| Vec::with_capacity(n));
    let mut snapshots = Vec::new();

    energy.push(resonator.energy());
    if let (Some(out), Some(i)) = (output.as_mut(), pickup_index) {
        out.push(resonator.displacement_at(i));
    }
    if snapshot_stride != 0 {
        snapshots.push((0, resonator.state()));
    }
    for i in 1..n {
        resonator.step();
        energy.push(resonator.energy());
        if let (Some(out), Some(j)) = (output.as_mut(), pickup_index) {
            out.push(resonator.displacement_at(j));
        }
        if snapshot_stride != 0 && i % snapshot_stride == 0 {
            snapshots.push((i, resonator.state()));
        }
    }

    let k = resonator.timestep();
    Ok(SimResult {
        time: (0..n).map(|i| i as f64 * k).collect(),
        energy,
        output,
        fs: 1.0 / k,
        snapshots,
    })
}

impl Resonator for IdealString {
    fn step(&mut self) {
        IdealString::step(self);
    }
    fn energy(&self) -> f64 {
        IdealString::energy(self)
    }
    fn state(&self) -> Vec<f64> {
        self.u.clone()
    }
    fn displacement_at(&self, index: usize) -> f64 {
        IdealString::displacement_at(self, index)
    }
    fn timestep(&self) -> f64 {
        self.params().k
    }
}

impl Resonator for StiffString {
    fn step(&mut self) {
        StiffString::step(self);
    }
    fn energy(&self) -> f64 {
        StiffString::energy(self)
    }
    fn state(&self) -> Vec<f64> {
        self.u.clone()
    }
    fn displacement_at(&self, index: usize) -> f64 {
        self.u[index]
    }
    fn timestep(&self) -> f64 {
        self.p.k
    }
}

impl Resonator for DampedStiffString {
    fn step(&mut self) {
        DampedStiffString::step(self);
    }
    fn energy(&self) -> f64 {
        DampedStiffString::energy(self)
    }
    fn state(&self) -> Vec<f64> {
        self.u.clone()
    }
    fn displacement_at(&self, index: usize) -> f64 {
        self.u[index]
    }
    fn timestep(&self) -> f64 {
        self.p.k
    }
}
