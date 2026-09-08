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
use crate::airbox_port::{InteriorSurfacePort, PortError, RoomPort, Spreading, SurfacePort};
use crate::body::ModalBody;
use crate::fmt::py_float;
use crate::membrane::{self, Membrane};
use crate::plate::{self, Boundary, Plate};
use crate::reduce;
use crate::sparse::Csr;
use crate::sparse_lu::{SparseLu, SparseLuError};

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
    /// The loaded system matrix could not be factored. Carries the solver's reason.
    ///
    /// `A_loaded = A + (k / 2 rho_s) T^T R T` is SPD added to SPD, so this is unreachable for any
    /// port this crate builds — it is here because [`SparseLu::factor`] is fallible and swallowing
    /// its verdict would turn a wrong matrix into a wrong trajectory.
    Factor(SparseLuError),
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
            WrapError::Factor(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for WrapError {}

impl From<PortError> for WrapError {
    fn from(e: PortError) -> WrapError {
        WrapError::Port(e)
    }
}

impl From<SparseLuError> for WrapError {
    fn from(e: SparseLuError) -> WrapError {
        WrapError::Factor(e)
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

// -- the grid tier ------------------------------------------------------------------------------
//
// The reference's six `RoomLoaded*` / `RoomSuspended*` classes, and the three `_*Surface` adapters
// underneath them, land here. The linear pair -- a `Plate` and a `Membrane` -- is this batch; the
// von Karman seam is the next one.
//
// # Six classes, two types
//
// The reference has six wrappers because Python has no generics: `RoomLoadedPlate`,
// `RoomSuspendedPlate` and the four others are one body of arithmetic with two enum arms in it
// (which face the surface is mounted on, and therefore which port it drives) and one attribute
// name that differs (`plate` vs `membrane`). The binding says so in its own comment -- "the
// reference's six wrappers are one class with two enum arms". Here that is expressible: the tier
// is [`RoomGrid<S>`], generic over the seam, with the tier itself carried by [`GridPort`]. So the
// six become `RoomGrid<PlateSeam>` and `RoomGrid<MembraneSeam>`, each with a `baffled` and a
// `suspended` constructor.
//
// This is not a simplification of the physics and it loses no distinction the reference made:
// `test_airbox_vk.py` anchors the two tiers against each other with `array_equal`, which is only
// meaningful because they *are* one transcription. Six copies would have been six chances to drift.
//
// # Why a wrapper exists at all, one tier up from `RoomLoadedBody`
//
// The lumped port reads one scalar volume velocity and returns one scalar pressure, so its coupling
// is a rank-one correction and needs no factorization. A distributed port reads a *vector* -- one
// pressure per air node under the surface -- so the load is a matrix, `(k / 2 rho_s) T^T R T`, and
// the only way to keep `radiated == injected` an identity rather than an approximation is to put
// that matrix **inside** the system matrix and factor the sum once:
//
// ```text
// A_loaded = A_bare + (k / 2 rho_s) T^T R T
// ```
//
// SPD plus PSD is SPD, so it factors. That single decision is what makes this tier cost a
// factorization the bare models never had -- model #4 in particular is a *pure explicit* update
// that suddenly needs a sparse LU -- and it is why the load matrix, not the pressure, is the object
// this tier is organised around.
//
// # The assembly, and the one thing about it with no referent after phase F
//
// `A_loaded` is `a_bare.add(&load.scaled(load_scale))` and nothing else. The reference spelled it
// `(a_bare + load_matrix * load_scale).tocsc()` followed by `eliminate_zeros()`, and the plan's
// section 16 is the measurement that those are the same matrix: [`Csr::add`] builds through
// `Csr::from_rows`, which drops an entry whose value is exactly zero, so the elimination is
// *inside* the add -- and at every fixture reachable through the reference there was nothing to
// drop anyway, because `T` carries no explicit zero and `T^T R T` is a Gram matrix over a positive
// `R`. `nnz_growth` and `lu_nnz` are therefore the same quantities they always were, not
// redefinitions.

/// The seam between a room port and the grid resonator it loads.
///
/// The reference's `_PlateSurface` / `_MembraneSurface` adapters, as one contract. The load-bearing
/// member is [`GridSeam::u_prev`]: it is a **live read** the caller must take before
/// [`GridSeam::commit`] rolls it away, which is why [`RoomGrid::prepare`] copies it out rather than
/// reading it again later in the step.
pub trait GridSeam {
    /// What the wrapper calls this resonator, as a rate-mismatch message names it.
    fn label(&self) -> &'static str;

    /// The timestep (s).
    fn k(&self) -> f64;

    /// The sample rate (Hz) -- read only to report a mismatch.
    fn fs(&self) -> f64;

    /// The number of unknowns, which is the length of every vector in this trait.
    fn n_live(&self) -> usize;

    /// Live-node positions (m), in the model's own C-order over its mask.
    ///
    /// `np.column_stack((X[mask], Y[mask]))` -- this ordering is what makes `T`'s columns line up
    /// with the state vector by construction rather than by a lookup.
    fn coords(&self) -> Vec<[f64; 2]>;

    /// Per-node areas (m^2) the port spreads onto the air, in the same order as [`Self::coords`].
    fn areas(&self) -> &[f64];

    /// The areal density (kg/m^2) an external nodal force divides by -- the load's `rho_s`.
    fn denominator(&self) -> f64;

    /// The **unloaded** system matrix.
    fn a_bare(&self) -> Csr;

    /// `u^{n-1}` -- read once per step, *before* [`Self::commit`].
    fn u_prev(&self) -> &[f64];

    /// The model's own right-hand side, force included, with none of the room's terms in it.
    fn rhs(&self, f_ext: Option<&[f64]>) -> Vec<f64>;

    /// Roll `u^{n-1} <- u^n <- u^{n+1}` and refresh whatever the model caches from it.
    fn commit(&mut self, u_next: Vec<f64>);

    /// The model's own discrete energy (Joules), without the coupling channel.
    fn energy(&self) -> f64;

    /// Set the initial displacement and velocity over the live nodes.
    fn set_state(&mut self, u0: &[f64], v0: &[f64]);
}

/// `rhs + k^2 f_ext / rho_s` -- the external-force path, added the way a model's own step adds it.
///
/// The plate's copy of this line is *inside* [`plate::step_rhs`] and cannot be hoisted out without
/// changing that function's shape, so the equality of the two is asserted by a native bar rather
/// than by construction. The membrane has no `f_ext` path of its own at all, which is why this
/// arithmetic has nothing in model #4 to be bit-identical *to* and has to be driven deliberately.
fn add_f_ext(rhs: Vec<f64>, f_ext: Option<&[f64]>, k2: f64, denominator: f64) -> Vec<f64> {
    let Some(f) = f_ext else { return rhs };
    assert_eq!(f.len(), rhs.len(), "f_ext must have n_live entries");
    (0..rhs.len())
        .map(|i| rhs[i] + k2 * f[i] / denominator)
        .collect()
}

/// The live-node coordinates of a masked 2-D model, C-order over the mask.
fn masked_coords(x: &[f64], y: &[f64], mask: &crate::ops2d::Mask) -> Vec<[f64; 2]> {
    let mut coords = Vec::with_capacity(mask.n_live());
    for j in 0..mask.nrows() {
        for i in 0..mask.ncols() {
            if mask.at(j, i) {
                coords.push([x[j * mask.ncols() + i], y[j * mask.ncols() + i]]);
            }
        }
    }
    coords
}

/// The seam on the [`Plate`] side -- model #5, and both of its boundary branches.
#[derive(Debug, Clone)]
pub struct PlateSeam {
    /// The resonator. Public because a caller reaches its state directly, and Rust has no
    /// attribute fallback to delegate through.
    pub plate: Plate,
    areas: Vec<f64>,
    denominator: f64,
}

impl PlateSeam {
    /// Wrap `plate`.
    ///
    /// The two branches differ in what a node's area *is*: simply supported, every live node
    /// carries the square cell `h^2` and the force denominator is `rho h^2`, which is exactly what
    /// `Plate::step` divides `f_ext` by. Free, the areas are the lumped quadrature weights `W`
    /// (`h^2`, `h^2/2`, `h^2/4` -- no dead rim), and because `W` sits *inside* `A` and is divided
    /// out by the solve, the denominator is the areal density alone.
    pub fn new(plate: Plate) -> PlateSeam {
        let areas = match plate.p.boundary {
            Boundary::Supported => vec![plate.p.h * plate.p.h; plate.p.n_live],
            Boundary::Free => plate.p.w.clone(),
        };
        let denominator = plate.p.force_denominator();
        PlateSeam {
            plate,
            areas,
            denominator,
        }
    }
}

impl GridSeam for PlateSeam {
    fn label(&self) -> &'static str {
        "plate"
    }

    fn k(&self) -> f64 {
        self.plate.p.k
    }

    fn fs(&self) -> f64 {
        self.plate.p.fs
    }

    fn n_live(&self) -> usize {
        self.plate.p.n_live
    }

    fn coords(&self) -> Vec<[f64; 2]> {
        masked_coords(&self.plate.p.x, &self.plate.p.y, &self.plate.p.mask)
    }

    fn areas(&self) -> &[f64] {
        &self.areas
    }

    fn denominator(&self) -> f64 {
        self.denominator
    }

    /// `(1 + sigma k) I + theta k^2 kappa^2 B` supported, the same with `W` for `I` and `K` for
    /// `B` free -- the reference's own operand order, which is what keeps a von Karman plate with
    /// its nonlinearity switched off a byte-exact reduction rather than a second transcription.
    fn a_bare(&self) -> Csr {
        let p = &self.plate.p;
        let sk = p.sigma * p.k;
        let coeff = p.theta * p.k * p.k * p.kappa * p.kappa;
        let left = match p.boundary {
            Boundary::Supported => Csr::identity(p.n_live),
            Boundary::Free => p.mass.clone().expect("the free branch assembles W"),
        };
        left.scaled(1.0 + sk).add(&p.stiffness.scaled(coeff))
    }

    fn u_prev(&self) -> &[f64] {
        &self.plate.u_prev
    }

    /// [`plate::step_rhs`] itself -- the model's own numerator, force path included.
    fn rhs(&self, f_ext: Option<&[f64]>) -> Vec<f64> {
        plate::step_rhs(&self.plate.u, &self.plate.u_prev, f_ext, &self.plate.p)
    }

    /// Roll the history and refresh the acceleration cache `pressure()` reads.
    ///
    /// The second difference is taken from the *committed* field, and the load was inside the
    /// solve, so the cache carries the air load with no post-solve refresh.
    fn commit(&mut self, u_next: Vec<f64>) {
        let k2 = self.plate.p.k * self.plate.p.k;
        self.plate.accel = (0..u_next.len())
            .map(|i| (u_next[i] - 2.0 * self.plate.u[i] + self.plate.u_prev[i]) / k2)
            .collect();
        self.plate.u_prev = std::mem::replace(&mut self.plate.u, u_next);
        self.plate.n += 1;
    }

    fn energy(&self) -> f64 {
        self.plate.energy()
    }

    fn set_state(&mut self, u0: &[f64], v0: &[f64]) {
        self.plate.set_state(u0, v0);
    }
}

/// The seam on the [`Membrane`] side -- model #4, and three differences from the plate.
///
/// * **The mass is uniform.** Every live node carries `rho h^2`; a membrane's rim is clamped and
///   dead, so there is no lumped `W` and no free-boundary branch, and the load's `(k / 2 rho h^2)`
///   is a plain scaling.
/// * **There is no acceleration cache**, so [`GridSeam::commit`] is the two-level roll and nothing
///   else -- the same gap that leaves `Membrane` without a `pressure()` read-out.
/// * **`rhs`'s force term has no counterpart in the model.** `Plate::step` has its own `f_ext`
///   path; `Membrane::step` takes no force at all. So that term is arithmetic with nothing in
///   model #4 to be bit-identical *to*, and a bar has to drive it deliberately -- a run that only
///   ever passes `None` compares the shared half twice and never reaches this one.
#[derive(Debug, Clone)]
pub struct MembraneSeam {
    /// The resonator.
    pub membrane: Membrane,
    areas: Vec<f64>,
    denominator: f64,
}

impl MembraneSeam {
    /// Wrap `membrane`.
    pub fn new(membrane: Membrane) -> MembraneSeam {
        let p = membrane.params();
        let n_live = p.n_live();
        // `(rho * h) * h` and `h * h` rather than a squaring call: both are the reference's own
        // left-to-right fold, and `powi`/`powf` is a different number in the last bit.
        let areas = vec![p.h * p.h; n_live];
        let denominator = p.rho * p.h * p.h;
        MembraneSeam {
            membrane,
            areas,
            denominator,
        }
    }
}

impl GridSeam for MembraneSeam {
    fn label(&self) -> &'static str {
        "membrane"
    }

    fn k(&self) -> f64 {
        self.membrane.params().k
    }

    fn fs(&self) -> f64 {
        self.membrane.params().fs
    }

    fn n_live(&self) -> usize {
        self.membrane.params().n_live()
    }

    /// The **live** nodes only, which is the moving surface: a membrane's rim is clamped and dead,
    /// so the radiating surface is one cell inside the nominal boundary.
    fn coords(&self) -> Vec<[f64; 2]> {
        let p = self.membrane.params();
        masked_coords(&p.x, &p.y, &p.mask)
    }

    fn areas(&self) -> &[f64] {
        &self.areas
    }

    fn denominator(&self) -> f64 {
        self.denominator
    }

    /// `(1 + sigma k) I`, and its emptiness is the tier's main design fact.
    ///
    /// Model #4 is a pure *explicit* update whose only implicit content is the damping denominator.
    /// The air load's unknown is `u^{n+1}`, so putting the load in `A` buys `radiated == injected`
    /// as an identity at the price of a factorization the model never had.
    fn a_bare(&self) -> Csr {
        let p = self.membrane.params();
        let sk = p.sigma * p.k;
        Csr::identity(p.n_live()).scaled(1.0 + sk)
    }

    fn u_prev(&self) -> &[f64] {
        &self.membrane.u_prev
    }

    /// [`membrane::step_numerator_into`] plus the force path.
    ///
    /// The numerator is the model's own, shared rather than copied, so a zero air load and no
    /// force reduce to the bare membrane exactly.
    fn rhs(&self, f_ext: Option<&[f64]>) -> Vec<f64> {
        let p = self.membrane.params();
        let mut base = vec![0.0; p.n_live()];
        membrane::step_numerator_into(&self.membrane.u, &self.membrane.u_prev, &mut base, p);
        let k = p.k;
        add_f_ext(base, f_ext, k * k, self.denominator)
    }

    fn commit(&mut self, u_next: Vec<f64>) {
        self.membrane.u_prev = std::mem::replace(&mut self.membrane.u, u_next);
        self.membrane.n_steps += 1;
    }

    fn energy(&self) -> f64 {
        self.membrane.energy()
    }

    fn set_state(&mut self, u0: &[f64], v0: &[f64]) {
        self.membrane.set_state(u0, v0);
    }
}

/// Which tier a grid wrapper is, carried by the port it holds.
///
/// The reference splits this across six class names; here it is two arms, and every difference
/// between the tiers lives in this enum's `match`es:
///
/// * **baffled** -- the surface is flush in a wall. The port reads one open-circuit pressure per
///   air node, the ledger is `surface_pressure`, and the closure is `pbar = pbar_free + R q`.
/// * **suspended** -- the surface hangs on an interior velocity plane and radiates from both
///   faces. The port reads a *pair* of node planes and the wrapper works with the **jump** across
///   them, the ledger is `pressure_jump`, and the closure carries the two loaded faces:
///   `d_pbar = d_free + (2 R) q`.
///
/// The doubling looks as if it should also appear in the load matrix and does not, because it is
/// already there: an [`InteriorSurfacePort`]'s `R` is exactly half a [`SurfacePort`]'s (a face in
/// the interior sees half the node weight of a face on a wall), so `2 R` interior and `R` baffled
/// are the same number and the two tiers' load matrices are bit-identical. A bar that compares them
/// is therefore not a discriminating test -- see `the_two_tiers_load_the_surface_identically`.
#[derive(Debug, Clone)]
pub enum GridPort {
    /// Mounted flush in a room face.
    Baffled(SurfacePort),
    /// Hung on an interior plane, radiating from both faces.
    Suspended(InteriorSurfacePort),
}

impl GridPort {
    /// How many numbers this tier's ledgers carry: nodes on a face, **faces** on a plane.
    pub fn ledger_len(&self) -> usize {
        match self {
            GridPort::Baffled(p) => p.node_count(),
            GridPort::Suspended(p) => p.face_count(),
        }
    }

    /// The spreading operator, live surface nodes to air nodes.
    pub fn t(&self) -> &Csr {
        match self {
            GridPort::Baffled(p) => p.t(),
            GridPort::Suspended(p) => p.t(),
        }
    }

    /// The per-terminal radiation resistance.
    pub fn r(&self) -> &[f64] {
        match self {
            GridPort::Baffled(p) => p.r(),
            GridPort::Suspended(p) => p.r(),
        }
    }

    /// `T^T R T` before the `k / 2 rho_s` scaling.
    pub fn load_matrix(&self) -> &Csr {
        match self {
            GridPort::Baffled(p) => p.load_matrix(),
            GridPort::Suspended(p) => p.load_matrix(),
        }
    }

    /// Refuse if this port already has an injection queued for the room's next step.
    pub fn require_ready(&self, room: &AirBox) -> Result<(), PortError> {
        match self {
            GridPort::Baffled(p) => p.require_ready(room),
            GridPort::Suspended(p) => p.require_ready(room),
        }
    }

    /// The open-circuit pressure the wrapper works with: the node values baffled, the **jump**
    /// `hi - lo` suspended.
    pub fn free_pressure(&self, room: &AirBox) -> Vec<f64> {
        match self {
            GridPort::Baffled(p) => p.free_pressure(room),
            GridPort::Suspended(p) => {
                let (lo, hi) = p.free_pressure(room);
                (0..hi.len()).map(|i| hi[i] - lo[i]).collect()
            }
        }
    }

    /// Queue this step's volume velocity.
    pub fn inject(&mut self, room: &mut AirBox, q: &[f64]) -> Result<(), PortError> {
        match self {
            GridPort::Baffled(p) => p.inject(room, q),
            GridPort::Suspended(p) => p.inject(room, q),
        }
    }

    /// Clear the pending mark -- reuse on a new run.
    pub fn reset(&mut self) {
        match self {
            GridPort::Baffled(p) => p.reset(),
            GridPort::Suspended(p) => p.reset(),
        }
    }
}

/// What [`RoomGrid::prepare`] read that the rest of the step needs -- one step's room half.
///
/// Every field is read **once per step**, before anything is mutated, and none of them depends on
/// `u^{n+1}`. That is what lets a client solve the resonator many times between the two phases: the
/// mallet's outer chord does exactly that, and it is a property of this struct rather than a happy
/// accident.
#[derive(Debug, Clone)]
pub struct RoomStepHalf {
    /// The tier's open-circuit pressure: one value per node baffled, the **jump** suspended.
    free: Vec<f64>,
    /// `u^{n-1}`, taken before `commit` rolls it away.
    u_prev: Vec<f64>,
    /// `T^T pbar_free` -- the known open-circuit term.
    load: Vec<f64>,
    /// `load_matrix @ u^{n-1}` -- the `u^{n-1}` half of the centred velocity, whose `u^{n+1}` half
    /// is already inside the factorization.
    carry: Vec<f64>,
}

impl RoomStepHalf {
    /// The tier's open-circuit pressure over the port's terminals.
    pub fn free(&self) -> &[f64] {
        &self.free
    }

    /// The `u^{n-1}` this half was assembled from.
    pub fn u_prev(&self) -> &[f64] {
        &self.u_prev
    }
}

/// A grid resonator loaded by a room through a distributed port.
///
/// One step is: read the port's open-circuit pressure, assemble the loaded right-hand side, solve
/// the **loaded** system once, commit, then close the terminal and queue the injection. The caller
/// steps the room afterwards, once, when every instrument sharing it has solved.
///
/// The room is passed at each call rather than owned, which is what lets two instruments share one.
#[derive(Debug, Clone)]
pub struct RoomGrid<S: GridSeam> {
    /// The seam, and through it the resonator.
    pub seam: S,
    /// The terminal into the room.
    pub port: GridPort,
    k: f64,
    denominator: f64,
    load_scale: f64,
    t_transpose: Csr,
    nnz_growth: f64,
    lu_loaded: SparseLu,
    lu_nnz: usize,
    radiated_energy: f64,
    nodal_volume_velocity: Vec<f64>,
    pbar: Vec<f64>,
    volume_velocity: f64,
    n: usize,
}

impl<S: GridSeam> RoomGrid<S> {
    /// Mount `seam`'s resonator flush in `face` -- the baffled tier.
    pub fn baffled(
        seam: S,
        room: &mut AirBox,
        face: &str,
        origin: Option<(f64, f64)>,
        spreading: Spreading,
    ) -> Result<RoomGrid<S>, WrapError> {
        Self::require_same_rate(&seam, room)?;
        let coords = seam.coords();
        let port = SurfacePort::new(room, face, &coords, seam.areas(), origin, spreading)?;
        Self::build(seam, GridPort::Baffled(port))
    }

    /// Hang `seam`'s resonator on the `index`-th velocity face plane normal to `plane` -- the
    /// suspended tier, radiating from both sides.
    pub fn suspended(
        seam: S,
        room: &mut AirBox,
        plane: &str,
        index: i64,
        origin: Option<(f64, f64)>,
        spreading: Spreading,
    ) -> Result<RoomGrid<S>, WrapError> {
        Self::require_same_rate(&seam, room)?;
        let coords = seam.coords();
        let port =
            InteriorSurfacePort::new(room, plane, index, &coords, seam.areas(), origin, spreading)?;
        Self::build(seam, GridPort::Suspended(port))
    }

    /// The rate check, run **before** the port claims anything, so a refused wrapper leaves the
    /// room exactly as it found it.
    fn require_same_rate(seam: &S, room: &AirBox) -> Result<(), WrapError> {
        if same_rate(seam.k(), room.p.k) {
            return Ok(());
        }
        Err(WrapError::RateMismatch {
            label: seam.label(),
            model_fs: seam.fs(),
            room_fs: room.p.fs,
        })
    }

    /// The shared constructor: the loaded matrix, its factorization and empty ledgers.
    ///
    /// `A_loaded = A_bare + (k / 2 rho_s) T^T R T`, SPD added to PSD, factored **once**. See the
    /// module's grid-tier header for why the sum is spelled as one `add` and why that reproduces
    /// the reference's `+` followed by `eliminate_zeros()` exactly.
    fn build(seam: S, port: GridPort) -> Result<RoomGrid<S>, WrapError> {
        let k = seam.k();
        let denominator = seam.denominator();
        let a_bare = seam.a_bare();
        let load_scale = 0.5 * k / denominator;
        let a_loaded = a_bare.add(&port.load_matrix().scaled(load_scale));
        let nnz_growth = a_loaded.nnz() as f64 / a_bare.nnz() as f64;
        let lu_loaded = SparseLu::factor(&a_loaded)?;
        let (l_nnz, u_nnz) = lu_loaded.nnz();
        let n_ledger = port.ledger_len();
        let t_transpose = port.t().transpose();
        Ok(RoomGrid {
            seam,
            port,
            k,
            denominator,
            load_scale,
            t_transpose,
            nnz_growth,
            lu_loaded,
            lu_nnz: l_nnz + u_nnz,
            radiated_energy: 0.0,
            nodal_volume_velocity: vec![0.0; n_ledger],
            pbar: vec![0.0; n_ledger],
            volume_velocity: 0.0,
            n: 0,
        })
    }

    /// Rebuild and refactor `A_loaded` from the port **as it stands now**.
    ///
    /// The reference got this for free: a test that wanted a deliberately wrong coupling wrote a
    /// new `R` and a new load matrix onto the port and then assigned `inst._lu_loaded = splu(...)`,
    /// re-deriving the assembly by hand. A native port is a value and the factorization is
    /// private, so the same experiments go through here instead -- which is *better* than the
    /// reference's route, because a hand-written re-derivation is a second spelling of `build`'s
    /// assembly and could drift from it, and this cannot.
    ///
    /// Only a bar that is deliberately breaking the coupling has any reason to call it.
    ///
    /// # Errors
    /// [`WrapError::Factor`] if the mutated load no longer factors.
    pub fn refactor(&mut self) -> Result<(), WrapError> {
        let a_bare = self.seam.a_bare();
        let a_loaded = a_bare.add(&self.port.load_matrix().scaled(self.load_scale));
        self.nnz_growth = a_loaded.nnz() as f64 / a_bare.nnz() as f64;
        self.lu_loaded = SparseLu::factor(&a_loaded)?;
        let (l_nnz, u_nnz) = self.lu_loaded.nnz();
        self.lu_nnz = l_nnz + u_nnz;
        self.t_transpose = self.port.t().transpose();
        Ok(())
    }

    /// The timestep (s).
    pub fn k(&self) -> f64 {
        self.k
    }

    /// The areal density the load divides by (kg/m^2).
    pub fn denominator(&self) -> f64 {
        self.denominator
    }

    /// `k / 2 rho_s` -- what `T^T R T` is scaled by inside `A_loaded`.
    pub fn load_scale(&self) -> f64 {
        self.load_scale
    }

    /// `nnz(A_loaded) / nnz(A_bare)` -- how much denser the air load made the system.
    pub fn nnz_growth(&self) -> f64 {
        self.nnz_growth
    }

    /// `nnz(L) + nnz(U)` of the loaded factorization -- the fill the load actually cost.
    pub fn lu_nnz(&self) -> usize {
        self.lu_nnz
    }

    /// The factorization of the loaded system.
    pub fn lu_loaded(&self) -> &SparseLu {
        &self.lu_loaded
    }

    /// Cumulative energy handed to the room through this surface (Joules).
    pub fn radiated_energy(&self) -> f64 {
        self.radiated_energy
    }

    /// Overwrite the coupling ledger -- for a client that books the channel itself.
    pub fn set_radiated_energy(&mut self, value: f64) {
        self.radiated_energy = value;
    }

    /// The per-terminal volume velocity of the most recent step (m^3/s).
    pub fn nodal_volume_velocity(&self) -> &[f64] {
        &self.nodal_volume_velocity
    }

    /// The closed terminal pressure of the most recent step (Pa) -- the node pressure baffled, the
    /// **jump** across the surface suspended.
    pub fn pbar(&self) -> &[f64] {
        &self.pbar
    }

    /// The net volume velocity of the most recent step (m^3/s).
    pub fn volume_velocity(&self) -> f64 {
        self.volume_velocity
    }

    /// Completed steps.
    pub fn n(&self) -> usize {
        self.n
    }

    /// Advance the step counter without stepping -- a client driving the seam itself keeps the
    /// wrapper's clock honest through this.
    pub fn set_n(&mut self, value: usize) {
        self.n = value;
    }

    /// The room half of one step, assembled **once**: the port read and the fixed load terms.
    ///
    /// Nothing here depends on `u^{n+1}`. `free_pressure()` is the *open-circuit* pressure -- what
    /// the room would do with the surface held still -- so it is a function of time-`n` room state;
    /// and the carry term is the `u^{n-1}` half of the centred velocity, whose `u^{n+1}` half is
    /// already inside the factorization. So a caller may solve the resonator as many times as it
    /// likes against this right-hand side, varying only its own `f_ext`, and every trial sees the
    /// same room.
    ///
    /// `require_ready` runs first and before anything is mutated, and the port is read **before**
    /// the caller steps the room.
    pub fn prepare(&self, room: &AirBox) -> Result<RoomStepHalf, WrapError> {
        self.port.require_ready(room)?; // before mutating anything
        let free = self.port.free_pressure(room); // read u^{n+1/2}, BEFORE room.step()
        let u_prev = self.seam.u_prev().to_vec(); // ONCE, and before commit()
        let load = self.t_transpose.matvec(&free);
        let carry = self.port.load_matrix().matvec(&u_prev);
        Ok(RoomStepHalf {
            free,
            u_prev,
            load,
            carry,
        })
    }

    /// The seam's own right-hand side plus the room's two load terms.
    ///
    /// The two terms are carried separately rather than pre-summed because `base + (-a + b)` and
    /// `(base - a) + b` are not the same double, and a client assembling this once per outer trial
    /// has to get the same one `step` does.
    pub fn loaded_rhs(&self, half: &RoomStepHalf, f_ext: Option<&[f64]>) -> Vec<f64> {
        let base = self.seam.rhs(f_ext);
        let k2 = self.k * self.k;
        (0..base.len())
            .map(|i| {
                base[i] - k2 * half.load[i] / self.denominator + self.load_scale * half.carry[i]
            })
            .collect()
    }

    /// The rest of the step, from the **committed** field: the volume velocity, the injection and
    /// the ledgers.
    ///
    /// `u_next` must be the field that was committed, not a trial. Handed a trial instead, every
    /// bar around this function still passes -- the port is injected exactly once, the radiated
    /// energy advances by a plausible amount -- and the room simply receives a volume velocity the
    /// resonator never had.
    pub fn finish(
        &mut self,
        room: &mut AirBox,
        half: &RoomStepHalf,
        u_next: &[f64],
    ) -> Result<(), WrapError> {
        let two_k = 2.0 * self.k;
        let vel: Vec<f64> = (0..u_next.len())
            .map(|i| (u_next[i] - half.u_prev[i]) / two_k)
            .collect();
        let q = self.port.t().matvec(&vel);
        let r = self.port.r();
        // Baffled: pbar = pbar_free + R q. Suspended: d_pbar = d_free + 2 R q, and the 2 is the two
        // loaded faces -- the operand order is the reference's, `(2 R) q` rather than `2 (R q)`.
        let pbar: Vec<f64> = match self.port {
            GridPort::Baffled(_) => (0..q.len()).map(|i| half.free[i] + r[i] * q[i]).collect(),
            GridPort::Suspended(_) => (0..q.len())
                .map(|i| half.free[i] + (2.0 * r[i]) * q[i])
                .collect(),
        };
        self.port.inject(room, &q)?;
        // A **read-out**, so `plate::dot`'s plain `s += x * y` and not this module's `mul_add`
        // one. Two reasons, and neither is a preference. The ledger reaches nothing: over 200
        // steps of all four linear scenes the resonator's state, the room's whole pressure field,
        // `pbar` and the per-node volume velocity are bit-identical to the reference while this
        // number is not, which is what "does not feed back" looks like when it is measured rather
        // than argued. And the identity it exists for is `radiated == injected`, whose other side
        // the room books with exactly this spelling -- matching the room matters more than
        // matching a `ddot` that is about to stop existing.
        self.radiated_energy += self.k * plate::dot(&pbar, &q);
        self.volume_velocity = reduce::sum(&q);
        self.nodal_volume_velocity = q;
        self.pbar = pbar;
        self.n += 1;
        Ok(())
    }

    /// Advance one step: read the port, solve the **loaded** system, commit, queue the injection.
    ///
    /// This is `prepare` -> one solve -> `commit` -> `finish`, and it is written that way rather
    /// than inline so that a client which needs *many* solves between the ends -- the mallet's
    /// outer chord -- can have them without a second copy of the room's bookkeeping.
    pub fn step(&mut self, room: &mut AirBox, f_ext: Option<&[f64]>) -> Result<(), WrapError> {
        let half = self.prepare(room)?;
        let rhs = self.loaded_rhs(&half, f_ext);
        let u_next = self.lu_loaded.solve(&rhs)?;
        // The load was IN the solve, so the seam's own caches carry it with no post-solve refresh.
        self.seam.commit(u_next.clone());
        self.finish(room, &half, &u_next)
    }

    /// Total discrete energy `E_model + integral pbar . q dt` (Joules) -- an explicit override, not
    /// a delegation, because the delegated number is the total *without* its coupling channel.
    pub fn energy(&self) -> f64 {
        self.seam.energy() + self.radiated_energy
    }

    /// Set the resonator's initial state and reset this port's coupling ledger to zero.
    pub fn set_state(&mut self, u0: &[f64], v0: &[f64]) {
        self.seam.set_state(u0, v0);
        self.reset_books();
    }

    /// Zero the resonator's state and the coupling ledger -- reuse on a new run.
    pub fn reset(&mut self) {
        let n_live = self.seam.n_live();
        let zero = vec![0.0; n_live];
        self.seam.set_state(&zero, &zero);
        self.reset_books();
    }

    /// Zero the coupling ledger and unstick the port, leaving the resonator's state alone.
    pub fn reset_books(&mut self) {
        let n_ledger = self.port.ledger_len();
        self.radiated_energy = 0.0;
        self.nodal_volume_velocity = vec![0.0; n_ledger];
        self.pbar = vec![0.0; n_ledger];
        self.volume_velocity = 0.0;
        self.port.reset();
        self.n = 0;
    }
}

impl RoomGrid<PlateSeam> {
    /// The plate's **monopole** read-out, reflecting the load.
    ///
    /// Right for free, because the load was inside the solve and the acceleration cache carries it
    /// with no post-solve refresh. There is deliberately no counterpart on `RoomGrid<MembraneSeam>`:
    /// model #4 caches no acceleration, so it has no monopole read-out to reflect.
    pub fn pressure(&self) -> f64 {
        self.seam.plate.pressure()
    }
}
