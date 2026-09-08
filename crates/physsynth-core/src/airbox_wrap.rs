//! The resonators a room loads — `airbox.py`'s wrapper tier, natively homed.
//!
//! The retirement's second batch opens this module with one class, [`RoomLoadedBody`]: a
//! [`crate::body::ModalBody`] radiating into an [`crate::airbox::AirBox`] through a
//! [`crate::airbox_port::RoomPort`]. The six *grid* wrappers (the plate, the von Kármán plate and
//! the membrane, baffled and suspended) are still only in the binding, together with the
//! distributed ports they stand on; they land in later batches and belong here when they do.
//!
//! # Why a wrapper exists at all
//!
//! The load's unknown is the body's *next* state, so a room cannot simply be stepped alongside a
//! body: the two have to be solved together, once per step. For a modal body that solve is
//! **rank one** — the port reads a single scalar volume velocity and returns a single scalar
//! pressure — so the whole coupling is a scalar division and a rank-one correction of `q^{n+1}`,
//! and no factorization is needed. That is what makes this the cheap wrapper and the right one to
//! open the module with.
//!
//! # The ledger, and why the scene total has no coupling term in it
//!
//! [`RoomLoadedBody::energy`] is `E_body + integral pbar U dt` and it is an **override**, not a
//! delegation: the delegated number is the body's total *without* its coupling channel. The room
//! books the same terminal from its own post-closure pressure, with the opposite sign, so
//!
//! ```text
//! sum_j inst_j.energy() + room.energy()
//! ```
//!
//! contains no coupling term at all and a drift in it is unambiguous evidence of a bug rather than
//! of accounting. It is *not* monotone per instrument — the port's near-field reactance hands
//! energy back every cycle — and it is necessary rather than sufficient: the conserved total is
//! structurally blind to a wrong `R_room`, because each side's identity telescopes against
//! whatever pressure *it* used. The detector for that is the cross-ledger residual
//! `radiated == injected`, which is a different measurement and not implied by the drift.

use crate::airbox::AirBox;
use crate::airbox_port::{PortError, RoomPort};
use crate::body::ModalBody;
use crate::fmt::py_float;
use crate::reduce;

/// A rejection from a room-loaded resonator.
#[derive(Debug, Clone, PartialEq)]
pub enum WrapError {
    /// The resonator and the room were built at different sample rates. Carries both rates.
    RateMismatch {
        /// What the caller calls the resonator, as the message names it.
        label: &'static str,
        /// The resonator's sample rate (Hz).
        model_fs: f64,
        /// The room's sample rate (Hz).
        room_fs: f64,
    },
    /// The port refused. Carries its reason.
    Port(PortError),
}

impl std::fmt::Display for WrapError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WrapError::RateMismatch {
                label,
                model_fs,
                room_fs,
            } => write!(
                f,
                "sample-rate mismatch: {label} fs = {} but room fs = {}. The port's solve is a                  single timestep shared by both, so they must agree exactly.",
                py_float(*model_fs),
                py_float(*room_fs),
            ),
            WrapError::Port(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for WrapError {}

impl From<PortError> for WrapError {
    fn from(e: PortError) -> WrapError {
        WrapError::Port(e)
    }
}

/// `_require_same_rate`'s predicate — the **room's** timestep is the reference of the comparison,
/// not the larger of the two, so it is asymmetric exactly as the reference's is.
fn same_rate(model_k: f64, room_k: f64) -> bool {
    (model_k - room_k).abs() <= 1e-12 * room_k.abs()
}

/// `a . b` — the modal read of the body's velocity, and the one `np.dot` in this crate that
/// **reaches the timestep**.
///
/// The crate's three other `dot` copies (`beam`, `plate`, `string_stiff`) are read-outs, and each
/// one says so before spelling itself `s += x * y`. This one is not: `u_free` becomes the volume
/// velocity the room is injected with and the pressure the body is corrected by, so a last bit
/// here is a different trajectory rather than a different number on a report.
///
/// `np.dot` on a 1-D float64 pair is BLAS `ddot`, and the accumulation is **fused** — the multiply
/// and the add round once, not twice. `s += x * y` does not reproduce it and `x.mul_add(y, s)`
/// does. That is a measurement rather than a construction, and it was made both ways: with the
/// plain loop, a two-instrument scene diverges from the reference at the **second step**, by one
/// ulp in the ball port's volume velocity, and 200 steps later 661 of 693 pressure nodes differ at
/// 1.9e-12 — the room amplifying one ulp, with the body's own state still bit-identical
/// throughout. With `mul_add`, every one of 34 quantities over 200 steps agrees to the bit on two
/// fixtures: two lossless modes in a rigid room, and five damped modes in a room with two lossy
/// walls. `f64::mul_add` is single-rounded by definition, so what this spelling makes portable is
/// the *native* answer; that it is also `ddot`'s answer is what the one-time check measured, at
/// two mode counts, before the reference went.
fn dot(a: &[f64], b: &[f64]) -> f64 {
    let mut s = 0.0;
    for (x, y) in a.iter().zip(b.iter()) {
        s = x.mul_add(*y, s);
    }
    s
}

/// A [`ModalBody`] loaded by a room through a lumped [`RoomPort`].
///
/// One step is: read the port's open-circuit pressure, advance the body **force-free**, solve the
/// scalar terminal, correct `q^{n+1}` by rank one, refresh the acceleration from the *corrected*
/// second difference, and queue the volume velocity. The caller steps the room afterwards, once,
/// when every instrument sharing it has solved.
///
/// The room is passed at each call rather than owned, which is what lets two instruments share one
/// — see [`RoomPort`].
#[derive(Debug, Clone)]
pub struct RoomLoadedBody {
    /// The resonator. Public because the bridges above this tier read the body's modal surface
    /// directly and Rust has no attribute fallback to delegate through; see the module header of
    /// `crates/physsynth-py/src/airbox_wrap.rs` for what the reference got for free here.
    pub body: ModalBody,
    /// The terminal into the room.
    pub port: RoomPort,
    k: f64,
    g: f64,
    corr: Vec<f64>,
    radiated_energy: f64,
    volume_velocity: f64,
    port_pressure: f64,
    n: usize,
}

impl RoomLoadedBody {
    /// Mount `body` in `room` at `at` (m), through a point port or a ball of radius `radius`.
    pub fn new(
        body: ModalBody,
        room: &mut AirBox,
        at: [f64; 3],
        radius: Option<f64>,
    ) -> Result<RoomLoadedBody, WrapError> {
        let k = body.params().k;
        if !same_rate(k, room.p.k) {
            return Err(WrapError::RateMismatch {
                label: "body",
                model_fs: body.params().fs,
                room_fs: room.p.fs,
            });
        }
        let port = RoomPort::new(room, at, radius)?;

        // Rank-1 precomputes. The `(1 + sigma_i k)` carries the body's implicit damping
        // denominator into the load, and the reduction is `np.sum` — it reaches the timestep, so
        // it goes through `reduce::sum` rather than a plain fold.
        let p = body.params();
        let one_plus_sk: Vec<f64> = p.sigma.iter().map(|s| 1.0 + s * k).collect();
        let terms: Vec<f64> = (0..p.a.len())
            .map(|i| p.a[i] * p.a[i] / (p.m[i] * one_plus_sk[i]))
            .collect();
        let g = 0.5 * k * reduce::sum(&terms);
        let corr: Vec<f64> = (0..p.a.len())
            .map(|i| k * k * p.a[i] / (p.m[i] * one_plus_sk[i]))
            .collect();

        Ok(RoomLoadedBody {
            body,
            port,
            k,
            g,
            corr,
            radiated_energy: 0.0,
            volume_velocity: 0.0,
            port_pressure: 0.0,
            n: 0,
        })
    }

    /// The timestep (s).
    pub fn k(&self) -> f64 {
        self.k
    }

    /// The lumped modal admittance `G` the port sees looking into the body.
    pub fn g(&self) -> f64 {
        self.g
    }

    /// The rank-one correction vector applied to `q^{n+1}` per unit terminal pressure.
    pub fn corr(&self) -> &[f64] {
        &self.corr
    }

    /// Cumulative energy handed to the room through this terminal (Joules).
    pub fn radiated_energy(&self) -> f64 {
        self.radiated_energy
    }

    /// The volume velocity of the most recent step (m^3/s).
    pub fn volume_velocity(&self) -> f64 {
        self.volume_velocity
    }

    /// The terminal pressure of the most recent step (Pa).
    pub fn port_pressure(&self) -> f64 {
        self.port_pressure
    }

    /// Completed steps.
    pub fn n(&self) -> usize {
        self.n
    }

    /// Advance one step: read the port, advance the body, solve the load, queue the injection.
    pub fn step(&mut self, room: &mut AirBox, force: f64) -> Result<(), WrapError> {
        self.port.require_ready(room)?; // before mutating anything
        let pbar_free = self.port.free_pressure(room); // read u^{n+1/2}, BEFORE room.step()
        let q_nm1 = self.body.q_prev().to_vec(); // before step() rolls the history

        self.body.step(force); // commit the force-free next state

        let q_free = self.body.q().to_vec();
        let delta: Vec<f64> = (0..q_nm1.len()).map(|i| q_free[i] - q_nm1[i]).collect();
        let u_free = dot(&self.body.params().a, &delta) / (2.0 * self.k);

        let r_room = self.port.r_room();
        let u = (u_free - self.g * pbar_free) / (1.0 + self.g * r_room);
        let pbar = pbar_free + r_room * u;

        // Rank-1 correction of q^{n+1}, then refresh q'' from the *corrected* second difference so
        // pressure() carries the load.
        let q_next: Vec<f64> = (0..self.corr.len())
            .map(|i| q_free[i] - pbar * self.corr[i])
            .collect();
        let prev = self.body.q_prev().to_vec(); // q^n, after the roll
        let k2 = self.k * self.k;
        let accel: Vec<f64> = (0..q_next.len())
            .map(|i| (q_next[i] - 2.0 * prev[i] + q_nm1[i]) / k2)
            .collect();
        *self.body.q_mut() = q_next;
        *self.body.accel_mut() = accel;

        self.port.inject(room, u)?;
        self.radiated_energy += self.k * pbar * u;
        self.volume_velocity = u;
        self.port_pressure = pbar;
        self.n += 1;
        Ok(())
    }

    /// Total discrete energy `E_body + integral pbar U dt` (Joules) — an override, not a
    /// delegation, because the delegated number is the total *without* its coupling channel.
    pub fn energy(&self) -> f64 {
        self.body.energy() + self.radiated_energy
    }

    /// Radiated pressure read-out `sum_i a_i q_i''`, reflecting the room load.
    pub fn pressure(&self) -> f64 {
        self.body.pressure()
    }

    /// Set the body's initial modal state and reset this port's coupling ledger to zero.
    pub fn set_state(&mut self, q0: &[f64], v0: &[f64]) {
        self.body.set_state(q0, v0);
        self.reset_books();
    }

    /// Zero the body state and the coupling ledger — reuse on a new run.
    pub fn reset(&mut self) {
        let m = self.body.params().n_modes();
        let zeros = vec![0.0; m];
        self.body.set_state(&zeros, &zeros);
        self.reset_books();
    }

    /// Clear the coupling ledger and unstick the port, leaving the body's state alone.
    pub fn reset_books(&mut self) {
        self.radiated_energy = 0.0;
        self.volume_velocity = 0.0;
        self.port_pressure = 0.0;
        self.port.reset();
        self.n = 0;
    }
}
