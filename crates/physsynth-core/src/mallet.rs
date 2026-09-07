//! Mallet-membrane collision — model #7, and the batch that finishes Phase 2.
//!
//! `docs/dev/rust-migration-plan.md` §17. The Python original is `physsynth/core/mallet.py`; its
//! class docstrings are the reference for the physics and the sign convention, and they are not
//! repeated here.
//!
//! # What is left to port, and what therefore any divergence means
//!
//! Both of this model's hard parts landed in earlier batches: the drumhead is
//! [`crate::membrane`] (Phase 2 batch 1, proved bit-identical over 2,000 fed-back steps) and the
//! contact root-find is [`crate::collision::solve_contact`] (Phase 3 batch 2, proved bit-identical
//! over 18,000 configurations and through the mallet's own five fixtures). What remains is the
//! **shell**: a two-line force-free flight integrator, two force-injection sites, and the
//! admittances that scale the force. So a divergence after this batch is the shell's — the plan
//! said so before the code was written, which is the only kind of prediction worth making.
//!
//! # The shell's one piece of arithmetic that can go wrong, and it is `** 2`
//!
//! Every constant here is built from a squaring:
//!
//! ```text
//! g_s = k**2 / (rho h**2 (1 + sigma k))     g_h = k**2 / M     KE = 0.5 M ((z - z')/k)**2
//! ```
//!
//! §16.2's finding applies one level out from where it was found. Those are **Python floats**, so
//! `**` is `float.__pow__`, which is the C library's `pow` — *not* `x * x`. The two are different
//! doubles: measured here on 2026-08-27, `float(x) ** 2.0` and `x * x` disagree in **225 of
//! 400,000** samples drawn from the range these quantities actually occupy. `g_s` and `g_h`
//! multiply the contact force at every timestep, so writing the obvious `k * k` would put a
//! last-bit error on the state of every step of every run.
//!
//! [`scalar_pow`] therefore exists, and its `#[inline(never)]` is load-bearing rather than
//! stylistic: LLVM rewrites `powf(x, 2.0)` into `x * x` when the exponent is a visible constant,
//! which is exactly the transformation this module must not have. `tests/mallet.rs` pins it with a
//! witness value rather than trusting the attribute.
//!
//! # Two models, because the oracle lives in the smaller one
//!
//! A **third** model joined them on 2026-09-06 — [`PlateParams`] / [`MalletPlate`], the same
//! mallet against an implicit Kirchhoff plate. It has no Python original and its own section
//! header, below, is where its one new idea (the drive-point influence column) is derived.
//!
//! [`WallParams`] is the same scheme with the resonator deleted (`g_s = 0`, the surface fixed), and
//! it is where model #7's closed form lives: a mass on a fixed linear spring is a half-period of
//! `sqrt(K/M)` with exact velocity reversal. It owns no field and performs no reduction, so unlike
//! the coupled model it is expected to be bit-identical in *every* observable, its energy included.

use crate::collision::{self, contact_potential, ContactError, ContactParams, PowPath};
use crate::krylov;
use crate::membrane;
use crate::plate;
use crate::sparse_lu::SparseLuError;

/// `x ** e` as CPython's `float.__pow__` spells it — re-exported from [`crate::pyfloat`].
///
/// Lived here first, and moved out when the theta-scheme strings needed the same guard: the
/// `#[inline(never)]` that keeps a literal `2.0` from being folded into `x * x` is not a property
/// of the mallet, it is a property of standing in for `float.__pow__` at all. The name stays
/// because this module's own tests and header speak of it.
pub use crate::pyfloat::scalar_pow;

/// Why a mallet was refused at construction.
///
/// The order of the variants is the order the Python original checks them in, and a call that is
/// wrong in more than one way must report the same fault Python would.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParamError {
    /// `mass <= 0`.
    NonPositiveMass,
    /// `stiffness <= 0`.
    NonPositiveStiffness,
    /// `alpha < 1`.
    AlphaTooSmall,
    /// `hysteresis < 0`.
    NegativeHysteresis,
    /// `gap < 0`.
    NegativeGap,
    /// `outer_tol <= 0` -- the gong's outer iteration only.
    ///
    /// After the five shared checks, and only reachable from [`VkPlateParams::new`]: the two
    /// linear models have no outer loop, so a `ParamError` they can return is still exactly the
    /// five the Python original checks.
    NonPositiveOuterTol,
    /// `outer_max_iter < 1` -- the gong's outer iteration only.
    TooFewOuterIters,
}

impl std::fmt::Display for ParamError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParamError::NonPositiveMass => write!(f, "mallet mass must be > 0."),
            ParamError::NonPositiveStiffness => write!(f, "felt stiffness K must be > 0."),
            ParamError::AlphaTooSmall => write!(f, "felt exponent alpha must be >= 1."),
            ParamError::NegativeHysteresis => write!(f, "hysteresis lambda_h must be >= 0."),
            ParamError::NegativeGap => write!(f, "initial gap must be >= 0."),
            ParamError::NonPositiveOuterTol => write!(f, "outer_tol must be > 0."),
            ParamError::TooFewOuterIters => write!(f, "outer_max_iter must be >= 1."),
        }
    }
}

impl std::error::Error for ParamError {}

/// The five scalar checks both models share, in the original's order.
///
/// Public because the binding has to run them **before** it can decide whether it was handed a
/// membrane it can drive: the original checks the scalars first, so a call that is both massless
/// and holding the wrong drumhead must report the mass.
pub fn check_common(
    mass: f64,
    stiffness: f64,
    alpha: f64,
    hysteresis: f64,
    gap: f64,
) -> Result<(), ParamError> {
    if mass <= 0.0 {
        return Err(ParamError::NonPositiveMass);
    }
    if stiffness <= 0.0 {
        return Err(ParamError::NonPositiveStiffness);
    }
    if alpha < 1.0 {
        return Err(ParamError::AlphaTooSmall);
    }
    if hysteresis < 0.0 {
        return Err(ParamError::NegativeHysteresis);
    }
    if gap < 0.0 {
        return Err(ParamError::NegativeGap);
    }
    Ok(())
}

// -- the coupled model ----------------------------------------------------------------------------

/// A mallet's fixed parameters, including the two admittances and the snapped strike node.
#[derive(Debug, Clone, PartialEq)]
pub struct Params {
    /// Mallet mass `M` (kg).
    pub mass: f64,
    /// Felt stiffness `K` (N/m^alpha).
    pub stiffness: f64,
    /// Felt exponent `alpha >= 1`.
    pub alpha: f64,
    /// Hunt-Crossley damping `lam_h >= 0`.
    pub lam_h: f64,
    /// Timestep, taken from the membrane.
    pub k: f64,
    /// Discrete-gradient Taylor-branch threshold.
    pub eta_tol: f64,
    /// Scalar-solve tolerance.
    pub newton_tol: f64,
    /// Scalar-solve iteration cap.
    pub newton_maxiter: usize,
    /// Flat live-node index of the strike.
    pub node: usize,
    /// Physical x of the snapped strike node (m).
    pub x_strike: f64,
    /// Physical y of the snapped strike node (m).
    pub y_strike: f64,
    /// Membrane driving-point admittance `k^2 / (rho h^2 (1 + sigma k))`.
    pub g_s: f64,
    /// Mallet admittance `k^2 / M`.
    pub g_h: f64,
    /// Their sum — what the scalar residual actually uses.
    pub g: f64,
    /// `sqrt(K/M) / 2pi` (Hz), the `alpha = 1` contact-frequency reference.
    pub contact_frequency: f64,
    /// `pi sqrt(M/K) / k` — steps per felt half-period. Under 8 the original warns.
    pub steps_per_contact: f64,
}

impl Params {
    /// Validate and derive, against the membrane the mallet will strike.
    ///
    /// The check order is the original's, and the strike snapping happens *after* all five scalar
    /// checks — so a call that is both massless and off the grid reports the mass.
    // Eleven arguments. This mirrors `MalletMembrane.__init__` minus its state half, and the
    // grouping that would please clippy would be a different function than the one being ported.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        mem: &membrane::Params,
        mass: f64,
        stiffness: f64,
        alpha: f64,
        hysteresis: f64,
        strike_x: f64,
        strike_y: f64,
        gap: f64,
        eta_tol: f64,
        newton_tol: f64,
        newton_maxiter: usize,
    ) -> Result<Params, ParamError> {
        check_common(mass, stiffness, alpha, hysteresis, gap)?;

        let node = mem.pickup_index_at(strike_x, strike_y);
        let x_strike = mem.to_live(&mem.x)[node];
        let y_strike = mem.to_live(&mem.y)[node];

        // The membrane is explicit, so a node force reaches only that node next step -> the
        // driving-point admittance is the bare local nodal mass, no A^-1 solve. `(1 + sigma k)`
        // carries the loss factor the force-free step already applied.
        let sk = mem.sigma * mem.k;
        let g_s = scalar_pow(mem.k, 2.0) / (mem.rho * scalar_pow(mem.h, 2.0) * (1.0 + sk));
        let g_h = scalar_pow(mem.k, 2.0) / mass;

        Ok(Params {
            mass,
            stiffness,
            alpha,
            lam_h: hysteresis,
            k: mem.k,
            eta_tol,
            newton_tol,
            newton_maxiter,
            node,
            x_strike,
            y_strike,
            g_s,
            g_h,
            g: g_s + g_h,
            contact_frequency: (stiffness / mass).sqrt() / (2.0 * std::f64::consts::PI),
            steps_per_contact: std::f64::consts::PI * (mass / stiffness).sqrt() / mem.k,
        })
    }

    /// The bundle every evaluation inside the scalar solve takes.
    pub fn contact(&self) -> ContactParams {
        ContactParams {
            stiffness: self.stiffness,
            alpha: self.alpha,
            lam_h: self.lam_h,
            k: self.k,
            tol: self.eta_tol,
        }
    }
}

/// The mallet's own state — everything the membrane does not already hold.
///
/// Shared by both models: the wall rig differs only in how it is seeded and what it calls a
/// surface, not in what it remembers.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct State {
    /// Mallet position `z_H^n`.
    pub z_h: f64,
    /// Mallet position `z_H^{n-1}`.
    pub z_h_prev: f64,
    /// The impact speed it was launched at (m/s, `> 0` = into the surface). Reported, not used.
    pub strike_velocity: f64,
    /// Penetration `eta^n`.
    pub penetration: f64,
    /// The force applied at the last step (N).
    pub contact_force: f64,
    /// Whether `eta^n > 0`.
    pub in_contact: bool,
    /// How many steps used the bracketed fallback rather than Newton.
    pub fallbacks: usize,
    /// Completed steps.
    pub n: usize,
}

impl State {
    /// Seed the coupled model: `gap` above the head at rest, moving in at `strike_velocity`.
    ///
    /// The pre-contact flight is force-free, so `z^{-1} = z^0 + k v` is exact (the acceleration is
    /// zero) rather than an approximation. `u_node` is the head's displacement at the strike node,
    /// which is `0` in every fixture the project uses but is read rather than assumed.
    pub fn at_strike(gap: f64, strike_velocity: f64, k: f64, u_node: f64) -> State {
        let z_h = gap;
        let penetration = u_node - z_h;
        State {
            z_h,
            z_h_prev: gap + k * strike_velocity,
            strike_velocity,
            penetration,
            contact_force: 0.0,
            in_contact: penetration > 0.0,
            fallbacks: 0,
            n: 0,
        }
    }

    /// Seed the wall rig. Deliberately *not* spelled like [`State::at_strike`]: the original
    /// writes `z_H = wall + gap` and then `z_H_prev = z_H + k v`, where the coupled model writes
    /// `gap + k v` from the argument. Same value when `wall = 0`, different expressions, and a
    /// port that "tidied" them into one would be transcribing a third model.
    pub fn at_wall(wall: f64, gap: f64, strike_velocity: f64, k: f64) -> State {
        let z_h = wall + gap;
        let penetration = wall - z_h;
        State {
            z_h,
            z_h_prev: z_h + k * strike_velocity,
            strike_velocity,
            penetration,
            contact_force: 0.0,
            in_contact: penetration > 0.0,
            fallbacks: 0,
            n: 0,
        }
    }

    /// Mallet velocity `delta_t- z_H` (m/s): negative into the surface, positive after rebound.
    pub fn velocity(&self, k: f64) -> f64 {
        (self.z_h - self.z_h_prev) / k
    }

    /// The averaged contact potential `0.5 (phi(eta^n) + phi(eta^{n-1}))` (J) — the form that
    /// telescopes with the discrete-gradient force.
    fn contact_pe(&self, surface_now: f64, surface_prev: f64, stiffness: f64, alpha: f64) -> f64 {
        let eta_n = surface_now - self.z_h;
        let eta_nm1 = surface_prev - self.z_h_prev;
        0.5 * (contact_potential(eta_n, stiffness, alpha, PowPath::Scalar)
            + contact_potential(eta_nm1, stiffness, alpha, PowPath::Scalar))
    }

    /// Commit a solved contact. `z_free` is the force-free flight position for this step.
    fn commit(&mut self, z_free: f64, g_h: f64, sol: collision::ContactSolution) {
        self.z_h_prev = self.z_h;
        self.z_h = z_free + g_h * sol.force;
        self.penetration = sol.eta;
        self.contact_force = sol.force;
        self.in_contact = sol.eta > 0.0;
        self.fallbacks += usize::from(sol.used_fallback);
        self.n += 1;
    }
}

/// `eta^{n-1}`, which must be read **before** the membrane steps.
///
/// The membrane's `step()` rolls `u_prev <- u^n`, so this quantity does not exist any more once it
/// has run. The original reads it on the first line of `step()` for exactly this reason.
pub fn eta_prev(u_prev_node: f64, s: &State) -> f64 {
    u_prev_node - s.z_h_prev
}

/// The force-free flight `z_free^{n+1} = 2 z^n - z^{n-1}`.
pub fn free_flight(s: &State) -> f64 {
    2.0 * s.z_h - s.z_h_prev
}

/// Solve the contact for one step and commit the mallet half of it.
///
/// `u_free` is the strike node *after* the membrane's force-free advance and `eta_prev` the value
/// [`eta_prev`] returned before it. Returns the corrected node value the caller must write back
/// into the membrane — the head is pushed by `-f` through its own admittance, which is exact
/// because the membrane is explicit.
pub fn resolve(
    u_free: f64,
    eta_prev: f64,
    z_free: f64,
    p: &Params,
    s: &mut State,
) -> Result<f64, ContactError> {
    let eta_free = u_free - z_free;
    let sol = collision::solve_contact(
        eta_free,
        eta_prev,
        p.g,
        p.contact(),
        s.penetration,
        p.newton_tol,
        p.newton_maxiter,
    )?;
    let u_corrected = u_free - p.g_s * sol.force;
    s.commit(z_free, p.g_h, sol);
    Ok(u_corrected)
}

/// Total discrete energy `H^n` (J): membrane + mallet kinetic + averaged contact potential.
///
/// `mem_energy` is the membrane's own `energy()`, and it is the one term here that is a
/// **reduction** — which is why the coupled model's energy is the single observable this port does
/// not claim bit-identity for (§14.2, and §16.3 attributed it by elimination against the wall rig).
pub fn energy(u_node: f64, u_prev_node: f64, mem_energy: f64, p: &Params, s: &State) -> f64 {
    let ke = 0.5 * p.mass * scalar_pow(s.velocity(p.k), 2.0);
    let pe = s.contact_pe(u_node, u_prev_node, p.stiffness, p.alpha);
    mem_energy + ke + pe
}

// -- the standalone wall rig ----------------------------------------------------------------------

/// The mass-vs-fixed-wall rig's parameters — the scheme with the resonator removed.
#[derive(Debug, Clone, PartialEq)]
pub struct WallParams {
    /// Mallet mass `M` (kg).
    pub mass: f64,
    /// Felt stiffness `K` (N/m^alpha).
    pub stiffness: f64,
    /// Felt exponent `alpha >= 1`.
    pub alpha: f64,
    /// Hunt-Crossley damping `lam_h >= 0`.
    pub lam_h: f64,
    /// The wall's position in the common `+z` frame.
    pub wall: f64,
    /// Timestep `1/fs`.
    pub k: f64,
    /// Discrete-gradient Taylor-branch threshold.
    pub eta_tol: f64,
    /// Scalar-solve tolerance.
    pub newton_tol: f64,
    /// Scalar-solve iteration cap.
    pub newton_maxiter: usize,
    /// The only admittance: `k^2 / M`. The wall is rigid, so `g_s = 0`.
    pub g: f64,
}

impl WallParams {
    /// Validate and derive. `fs` is **not** checked, because the original does not check it — the
    /// membrane validates its own sample rate and this rig has no membrane to borrow one from.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        mass: f64,
        stiffness: f64,
        fs: f64,
        alpha: f64,
        hysteresis: f64,
        wall: f64,
        gap: f64,
        eta_tol: f64,
        newton_tol: f64,
        newton_maxiter: usize,
    ) -> Result<WallParams, ParamError> {
        check_common(mass, stiffness, alpha, hysteresis, gap)?;
        let k = 1.0 / fs;
        Ok(WallParams {
            mass,
            stiffness,
            alpha,
            lam_h: hysteresis,
            wall,
            k,
            eta_tol,
            newton_tol,
            newton_maxiter,
            g: scalar_pow(k, 2.0) / mass,
        })
    }

    /// The bundle every evaluation inside the scalar solve takes.
    pub fn contact(&self) -> ContactParams {
        ContactParams {
            stiffness: self.stiffness,
            alpha: self.alpha,
            lam_h: self.lam_h,
            k: self.k,
            tol: self.eta_tol,
        }
    }
}

/// Advance the wall rig one step: force-free flight, scalar contact solve, exact force inject.
pub fn wall_step(p: &WallParams, s: &mut State) -> Result<(), ContactError> {
    let eta_prev = p.wall - s.z_h_prev;
    let z_free = free_flight(s);
    let eta_free = p.wall - z_free;
    let sol = collision::solve_contact(
        eta_free,
        eta_prev,
        p.g,
        p.contact(),
        s.penetration,
        p.newton_tol,
        p.newton_maxiter,
    )?;
    s.commit(z_free, p.g, sol);
    Ok(())
}

/// The wall rig's total energy `0.5 M (delta_t- z)^2 + 0.5 (phi(eta^n) + phi(eta^{n-1}))` (J).
///
/// No field and no reduction, which is what makes this the model that can be compared to the bit.
pub fn wall_energy(p: &WallParams, s: &State) -> f64 {
    let ke = 0.5 * p.mass * scalar_pow(s.velocity(p.k), 2.0);
    ke + s.contact_pe(p.wall, p.wall, p.stiffness, p.alpha)
}

// -- the native owning views ----------------------------------------------------------------------

/// A membrane struck by a mallet — the Rust caller's view, and what `cargo test` exercises.
///
/// The Python binding does **not** wrap this, for the reason every model in this crate has the
/// same split: the binding must hold the caller's own `Membrane` object so that `mal.membrane` is
/// the object that was passed in, and `tests/test_mallet_energy.py` reads the head's field through
/// exactly that handle.
#[derive(Debug, Clone)]
pub struct MalletMembrane {
    params: Params,
    state: State,
    /// The drumhead.
    pub membrane: membrane::Membrane,
}

impl MalletMembrane {
    /// Build from validated parameters and a membrane at rest.
    pub fn new(
        params: Params,
        membrane: membrane::Membrane,
        gap: f64,
        strike_velocity: f64,
    ) -> MalletMembrane {
        let u_node = membrane.u[params.node];
        let state = State::at_strike(gap, strike_velocity, params.k, u_node);
        MalletMembrane {
            params,
            state,
            membrane,
        }
    }

    /// The parameter set this mallet was built from.
    pub fn params(&self) -> &Params {
        &self.params
    }

    /// The mallet's own state.
    pub fn state(&self) -> &State {
        &self.state
    }

    /// Advance one step: force-free advance, scalar contact solve, exact force inject.
    pub fn step(&mut self) -> Result<(), ContactError> {
        let i = self.params.node;
        let eta_prev = eta_prev(self.membrane.u_prev[i], &self.state);
        self.membrane.step();
        let u_free = self.membrane.u[i];
        let z_free = free_flight(&self.state);
        self.membrane.u[i] = resolve(u_free, eta_prev, z_free, &self.params, &mut self.state)?;
        Ok(())
    }

    /// Total discrete energy `H^n` (J).
    pub fn energy(&self) -> f64 {
        let i = self.params.node;
        energy(
            self.membrane.u[i],
            self.membrane.u_prev[i],
            self.membrane.energy(),
            &self.params,
            &self.state,
        )
    }

    /// Mallet velocity `delta_t- z_H` (m/s).
    pub fn mallet_velocity(&self) -> f64 {
        self.state.velocity(self.params.k)
    }
}

/// The standalone mass-vs-fixed-wall rig — the Rust caller's view.
#[derive(Debug, Clone)]
pub struct MalletWall {
    params: WallParams,
    state: State,
}

impl MalletWall {
    /// Build from validated parameters.
    pub fn new(params: WallParams, gap: f64, strike_velocity: f64) -> MalletWall {
        let state = State::at_wall(params.wall, gap, strike_velocity, params.k);
        MalletWall { params, state }
    }

    /// The parameter set this rig was built from.
    pub fn params(&self) -> &WallParams {
        &self.params
    }

    /// The mallet's own state.
    pub fn state(&self) -> &State {
        &self.state
    }

    /// Advance one step.
    pub fn step(&mut self) -> Result<(), ContactError> {
        wall_step(&self.params, &mut self.state)
    }

    /// Total energy (J) — conserved at `lam_h = 0`, monotone decreasing above it.
    pub fn energy(&self) -> f64 {
        wall_energy(&self.params, &self.state)
    }

    /// Mallet velocity: `-strike_velocity` inbound, positive after rebound.
    pub fn velocity(&self) -> f64 {
        self.state.velocity(self.params.k)
    }
}

// == the plate model ==============================================================================
//
// Model #7p: the same mallet, striking a **Kirchhoff plate** instead of a drumhead. New in the
// batch of 2026-09-06, and the first thing in this file with no Python original — everything above
// was transcribed, this was written.
//
// # One thing is new, and it is the admittance
//
// The mallet's whole scheme is the scalar collapse: both bodies are linear in the contact force
// except through the contact itself, so one scalar equation in `eta^{n+1}` closes the step. That
// argument needs exactly one number from the resonator — how far the struck node moves per Newton
// of applied force — and for the membrane that number is free, because the membrane is *explicit*
// and a nodal force reaches only its own node by the next step: `g_s = k^2 / (rho h^2 (1 + sigma
// k))`, the bare local nodal mass.
//
// The plate is **implicit**. Its step solves `A u^{n+1} = rhs` with `A = (1 + sigma k) W + theta
// k^2 kappa^2 K`, so a force at one node reaches *every* node next step and there is no local
// admittance to read off. What replaces it is the **influence column**
//
// ```text
// influence = (k^2 / force_den) A^-1 e_node ,        g_s = influence[node]
// ```
//
// one back-substitution, done once at construction and reused every step. This is the bow's
// manoeuvre against the implicit stiff string (a precomputed `A^-1 e_i` baked into a vector), and
// the hammer plan named it as this model's route four batches before it was built.
//
// # Why superposition is exact here, and where it stops being exact
//
// `Plate::step` is affine in `f_ext`, so stepping force-free and adding `influence * f` afterwards
// is *algebraically identical* to putting `f` in the right-hand side before the solve:
// `A (u_free + d) = rhs_0 + A d = rhs_0 + (k^2 f / force_den) e_node`. The plate's own `step_rhs`
// comment says a post-solve correction is invalid because `A` couples all nodes — and that is right
// about a **one-node** correction, which is what the membrane does. Correcting along the whole
// column is the same thing done properly.
//
// In floating point the identity holds to about `1e-15` relative and **not to the bit**, because a
// sparse LU back-substitution is not a linear map over doubles: `solve(rhs_0) + c * solve(e)` and
// `solve(rhs_0 + c*e)` round differently. So the guard on this is a tight *relative* claim over the
// whole field, never an `array_equal`. A **miss** is the exception and is exact: `f == 0.0` makes
// every increment a signed zero and `x - (+-0.0) == x` for finite `x`, so a mallet that never
// touches the plate leaves a trajectory bit-identical to the bare plate's.
//
// # The energy identity carries over unchanged
//
// Dotting the theta-scheme with `u^{n+1} - u^{n-1}` gives the same discrete power a force does on
// the membrane, `f . delta_t. u`, so the discrete-gradient telescoping is untouched:
// `E_plate + 1/2 M (delta_t- z)^2 + 1/2 (phi(eta^n) + phi(eta^{n-1}))` is constant at `sigma = 0,
// lam_h = 0` and monotone decreasing above either. The theta-scheme is unconditionally stable for
// `theta >= 1/4`, so unlike the membrane the plate imposes no CFL on the strike — what still has to
// be resolved is the *felt*, whose half-period is `pi sqrt(M/K)` and whose `steps_per_contact`
// warning is the same one the membrane model raises.
//
// # What this model is NOT: the gong
//
// It takes [`plate::Plate`] and refuses the von Karman plate. That is not an omission to fill in
// later with a cast: the nonlinear plate's step is a *nonlinear* solve, so it is not affine in
// `f_ext` and the influence column does not exist. A mallet on a gong needs a nested iteration — an
// outer scalar contact solve wrapped around a full Picard or Newton plate solve per evaluation —
// which is a different algorithm with a different cost, not a different collaborator type.
//
// **That model is now [`MalletVkPlate`], below, and building it corrected two of the sentences
// above.** The algorithm really is different and the influence column really does not exist. But
// this section used to go on to price it at "roughly ten to a hundred plate solves per timestep,
// with no closed-form derivative for the outer Newton": the measured cost is **1.9 to 2.3** bare
// steps, and the outer derivative is closed-form. Both are corrected where they are derived, in
// the gong section's own header.
//
// # A free plate recoils
//
// On the `free` branch the plate has the `{1, x, y}` rigid nullspace, and a strike puts net impulse
// into it: the plate translates away at constant velocity for ever after. That is physics (an
// unsupported cymbal recoils), the energy is conserved through it, and it is named here because a
// reader watching a struck free plate drift off screen will otherwise file it as a bug.

/// The physical `(x, y)` of live node `node`.
///
/// Walks the mask exactly as [`plate::pickup_index_at`] walks it, and for the same reason it must:
/// `pickup_index_at` returns a **live** index while `plate::Params::x` and `y` are **full-grid**
/// arrays. Sharing the traversal makes the mixup unrepresentable rather than merely tested for — it
/// would otherwise report a plausible strike point and build a wrong influence column.
fn live_coords(p: &plate::Params, node: usize) -> (f64, f64) {
    let mut live = 0usize;
    for (idx, &alive) in p.mask.flags().iter().enumerate() {
        if !alive {
            continue;
        }
        if live == node {
            return (p.x[idx], p.y[idx]);
        }
        live += 1;
    }
    unreachable!(
        "live node {node} is beyond the plate's {} live nodes",
        p.n_live
    )
}

/// A mallet's fixed parameters against a plate, including the drive-point influence column.
#[derive(Debug, Clone, PartialEq)]
pub struct PlateParams {
    /// Mallet mass `M` (kg).
    pub mass: f64,
    /// Felt stiffness `K` (N/m^alpha).
    pub stiffness: f64,
    /// Felt exponent `alpha >= 1`.
    pub alpha: f64,
    /// Hunt-Crossley damping `lam_h >= 0`.
    pub lam_h: f64,
    /// Timestep, taken from the plate.
    pub k: f64,
    /// Discrete-gradient Taylor-branch threshold.
    pub eta_tol: f64,
    /// Scalar-solve tolerance.
    pub newton_tol: f64,
    /// Scalar-solve iteration cap.
    pub newton_maxiter: usize,
    /// Flat live-node index of the strike.
    pub node: usize,
    /// Physical x of the snapped strike node (m).
    pub x_strike: f64,
    /// Physical y of the snapped strike node (m).
    pub y_strike: f64,
    /// `(k^2 / force_den) A^-1 e_node` over the live nodes — how the whole field moves per Newton
    /// of force at the strike, one step later. Dense, because `A^-1` is.
    pub influence: Vec<f64>,
    /// Plate driving-point admittance — `influence[node]`, and *read from it* rather than computed
    /// a second way, so the number the contact solves against and the field the force is spread
    /// over cannot disagree.
    pub g_s: f64,
    /// Mallet admittance `k^2 / M`.
    pub g_h: f64,
    /// Their sum — what the scalar residual actually uses.
    pub g: f64,
    /// `sqrt(K/M) / 2pi` (Hz), the `alpha = 1` contact-frequency reference.
    pub contact_frequency: f64,
    /// `pi sqrt(M/K) / k` — steps per felt half-period. Under 8 the model warns.
    pub steps_per_contact: f64,
}

impl PlateParams {
    /// Validate and derive, against the plate the mallet will strike.
    ///
    /// The five scalar checks run first and in [`check_common`]'s order, so a call that is wrong in
    /// more than one way reports what the membrane model would report.
    ///
    /// # Errors
    /// Any of [`ParamError`].
    ///
    /// # Panics
    /// If the plate's factorization cannot back-substitute — which cannot happen for a `Plate` that
    /// was constructed, since `A` is SPD and was factored then. Spelled as `Plate::step` spells it.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        pl: &plate::Params,
        mass: f64,
        stiffness: f64,
        alpha: f64,
        hysteresis: f64,
        strike_x: f64,
        strike_y: f64,
        gap: f64,
        eta_tol: f64,
        newton_tol: f64,
        newton_maxiter: usize,
    ) -> Result<PlateParams, ParamError> {
        check_common(mass, stiffness, alpha, hysteresis, gap)?;

        let node = plate::pickup_index_at(strike_x, strike_y, pl);
        let (x_strike, y_strike) = live_coords(pl, node);

        // The influence column: one back-substitution against the plate's own factorization, so it
        // describes the operator the step actually applies rather than a re-derived twin. `k * k`
        // and not `scalar_pow(k, 2.0)` on purpose — this scale has to be the one
        // `plate::step_rhs` divides `f_ext` by, and that spells its `k^2` as a multiply. `g_h`
        // below is the mallet's own constant and keeps the `float.__pow__` spelling the other two
        // constructors use, so that a mallet's admittance is one number however it was built.
        let mut e = vec![0.0; pl.n_live];
        e[node] = 1.0;
        let column = pl.lu.solve(&e).expect("A is SPD and was factored");
        let scale = pl.k * pl.k / pl.force_denominator();
        let influence: Vec<f64> = column.iter().map(|&c| scale * c).collect();

        let g_s = influence[node];
        let g_h = scalar_pow(pl.k, 2.0) / mass;

        Ok(PlateParams {
            mass,
            stiffness,
            alpha,
            lam_h: hysteresis,
            k: pl.k,
            eta_tol,
            newton_tol,
            newton_maxiter,
            node,
            x_strike,
            y_strike,
            influence,
            g_s,
            g_h,
            g: g_s + g_h,
            contact_frequency: (stiffness / mass).sqrt() / (2.0 * std::f64::consts::PI),
            steps_per_contact: std::f64::consts::PI * (mass / stiffness).sqrt() / pl.k,
        })
    }

    /// The bundle every evaluation inside the scalar solve takes.
    pub fn contact(&self) -> ContactParams {
        ContactParams {
            stiffness: self.stiffness,
            alpha: self.alpha,
            lam_h: self.lam_h,
            k: self.k,
            tol: self.eta_tol,
        }
    }
}

/// Solve the contact for one step and commit the mallet half of it. Returns the contact force `f`.
///
/// Unlike [`resolve`], which hands back a corrected node value, this returns the **force**: the
/// plate's half of the correction is a whole-field operation and belongs to [`plate_inject`], which
/// the caller runs against the plate's own buffers.
///
/// # Errors
/// [`ContactError`] if the scalar contact solve finds no root.
pub fn plate_resolve(
    u_free_node: f64,
    eta_prev: f64,
    z_free: f64,
    p: &PlateParams,
    s: &mut State,
) -> Result<f64, ContactError> {
    let eta_free = u_free_node - z_free;
    let sol = collision::solve_contact(
        eta_free,
        eta_prev,
        p.g,
        p.contact(),
        s.penetration,
        p.newton_tol,
        p.newton_maxiter,
    )?;
    let force = sol.force;
    s.commit(z_free, p.g_h, sol);
    Ok(force)
}

/// The displacement node `i` takes from a contact force `force` — the one spelling of it.
///
/// Private, and the two public injectors below both go through it, because the whole point of
/// splitting them is that they can be called from separate borrows: two copies of this expression
/// could drift apart and nothing would notice until the acceleration disagreed with the field it
/// is supposed to be the second difference of.
#[inline]
fn injected_at(p: &PlateParams, force: f64, i: usize) -> f64 {
    p.influence[i] * force
}

/// Apply the plate's half of the contact to the **displacement** field: `-force` at the strike
/// node, spread along the influence column, into a field that has already advanced force-free.
///
/// # Panics
/// If `u` and the influence column are not the same length.
pub fn plate_inject_u(u: &mut [f64], p: &PlateParams, force: f64) {
    assert!(
        u.len() == p.influence.len(),
        "plate state and influence column must agree in length"
    );
    for (i, node) in u.iter_mut().enumerate() {
        *node -= injected_at(p, force, i);
    }
}

/// Apply the matching correction to the **acceleration**.
///
/// The plate defines `accel` as the actual second difference `(u^{n+1} - 2u^n + u^{n-1}) / k^2`,
/// and only the first term moved — so this is the same increment over `k^2`. Leaving it stale
/// would make `pressure()`, the radiated read-out, report the *unstruck* plate: a silent wrong
/// answer in exactly the observable a listener cares about.
///
/// # Panics
/// If `accel` and the influence column are not the same length.
pub fn plate_inject_accel(accel: &mut [f64], p: &PlateParams, force: f64) {
    assert!(
        accel.len() == p.influence.len(),
        "plate state and influence column must agree in length"
    );
    let k2 = p.k * p.k;
    for (i, node) in accel.iter_mut().enumerate() {
        *node -= injected_at(p, force, i) / k2;
    }
}

/// Both halves at once — the Rust caller's path, where `u` and `accel` are two distinct `Vec`s and
/// the borrow checker says so.
///
/// The binding cannot use this and calls the two separately, on purpose: there its buffers are
/// Python-owned arrays reached through `readwrite()`, both have public setters, and
/// `plate.u = a; plate._accel = a` makes them the **same object** — two simultaneous mutable
/// borrows of one NumPy buffer, which is a panic across the FFI boundary rather than an error.
/// Sequential borrows make the question not arise.
pub fn plate_inject(u: &mut [f64], accel: &mut [f64], p: &PlateParams, force: f64) {
    plate_inject_u(u, p, force);
    plate_inject_accel(accel, p, force);
}

/// Total discrete energy `H^n` (J): plate + mallet kinetic + averaged contact potential.
///
/// The plate term is a **reduction** (three `potential_form` calls over the whole field), which is
/// what makes this the one observable on this model a bit-exact claim cannot be made about — the
/// same split the membrane model has, for the same reason.
pub fn plate_total_energy(
    u_node: f64,
    u_prev_node: f64,
    plate_energy: f64,
    p: &PlateParams,
    s: &State,
) -> f64 {
    let ke = 0.5 * p.mass * scalar_pow(s.velocity(p.k), 2.0);
    let pe = s.contact_pe(u_node, u_prev_node, p.stiffness, p.alpha);
    plate_energy + ke + pe
}

/// A plate struck by a mallet — the Rust caller's view, and what `cargo test` exercises.
///
/// Splits from the binding for the reason every model in this crate splits: the binding has to hold
/// the caller's own plate object so that `mal.plate` **is** the object that was passed in.
///
/// The influence column is a snapshot of the plate's factorization, taken at construction. So is
/// the plate's own `lu`, and `Plate::set_B` deliberately does not rebuild it — which means the two
/// go stale together and stay consistent with each other. Assigning a different operator to a plate
/// a mallet is already holding gives an inconsistent plate, exactly as it does without one.
#[derive(Debug, Clone)]
pub struct MalletPlate {
    params: PlateParams,
    state: State,
    /// The plate.
    pub plate: plate::Plate,
}

impl MalletPlate {
    /// Build from validated parameters and a plate.
    pub fn new(
        params: PlateParams,
        plate: plate::Plate,
        gap: f64,
        strike_velocity: f64,
    ) -> MalletPlate {
        let u_node = plate.u[params.node];
        let state = State::at_strike(gap, strike_velocity, params.k, u_node);
        MalletPlate {
            params,
            state,
            plate,
        }
    }

    /// The parameter set this mallet was built from.
    pub fn params(&self) -> &PlateParams {
        &self.params
    }

    /// The mallet's own state.
    pub fn state(&self) -> &State {
        &self.state
    }

    /// Advance one step: force-free advance, scalar contact solve, force spread along the column.
    ///
    /// # Errors
    /// [`ContactError`] if the scalar contact solve finds no root.
    pub fn step(&mut self) -> Result<(), ContactError> {
        let i = self.params.node;
        // `eta^{n-1}` must be read BEFORE the plate steps: `Plate::step` rebinds `u_prev` to what
        // `u` was, so this quantity stops existing one line later.
        let eta_prev = eta_prev(self.plate.u_prev[i], &self.state);
        self.plate.step(None);
        let u_free = self.plate.u[i];
        let z_free = free_flight(&self.state);
        let force = plate_resolve(u_free, eta_prev, z_free, &self.params, &mut self.state)?;
        plate_inject(
            &mut self.plate.u,
            &mut self.plate.accel,
            &self.params,
            force,
        );
        Ok(())
    }

    /// Total discrete energy `H^n` (J).
    pub fn energy(&self) -> f64 {
        let i = self.params.node;
        plate_total_energy(
            self.plate.u[i],
            self.plate.u_prev[i],
            self.plate.energy(),
            &self.params,
            &self.state,
        )
    }

    /// Mallet velocity `delta_t- z_H` (m/s).
    pub fn mallet_velocity(&self) -> f64 {
        self.state.velocity(self.params.k)
    }
}

// -- model #7g: the mallet on a gong (a nonlinear plate) -------------------------------------------
//
// The batch `docs/dev/mallet-plate-plan.md` §6 deferred, and the second half of HANDOFF §14.1's
// "a batch wanting the gong impression still needs a mallet, not a budget".
//
// # Why [`MalletPlate`] cannot simply take a [`plate::VkPlate`]
//
// The linear plate's step is **affine** in `f_ext`, which is the entire basis of the scalar
// collapse above: the whole field's response to a unit drive-point force is one constant vector,
// the influence column, and the number the contact equation needs is one entry of it. The von
// Karman step is a *nonlinear* solve (Picard sweeps, or Newton), so the response to a force is not
// a linear map and no such column exists. `MalletPlate` refuses a `VKPlate` and says so.
//
// # What replaces it: an outer iteration whose frozen tangent is the LINEAR plate's
//
// Write `Psi(f)` for the strike node after a **full** von Karman step driven by `-f e_node`. The
// step to solve is
//
// ```text
// eta^{n+1} = Psi(f) - z_free - g_h f ,        f = phi'(eta^{n+1}, eta^n)
// ```
//
// one scalar equation whose every evaluation costs a complete nonlinear plate solve. The
// iteration shipped here is a **chord**, and its frozen tangent is the linear plate's own
// drive-point admittance `g_s = influence[node]` -- the exact quantity `MalletPlate` is built on,
// computed from the same `A^-1 e_node` back-substitution against `VkParams::lin`'s factorization.
// One outer iteration is:
//
// ```text
// u_eff = Psi(f_j) + g_s f_j          the "effective free node": where a LINEAR plate with this
//                                     tangent would have had to start to land on Psi(f_j)
// f_{j+1} = solve_contact(u_eff - z_free, eta^n, g = g_s + g_h, ...)
// ```
//
// and `solve_contact` is model #7's, unchanged and uncast -- the whole point of the manoeuvre is
// that the proven contact root-find is reused rather than re-derived inside a nonlinear solver.
//
// # It degenerates to [`MalletPlate`], exactly, and that is the anchor
//
// With `nonlinear = false`, `Psi(f) = u_free - g_s f` identically, so `u_eff = u_free` **whatever
// `f_j` is**: the outer loop's first residual is zero and it exits at one iteration having solved
// precisely the contact equation `MalletPlate` solves. A sign error in the `f_ext = -f e_node`
// convention therefore goes red against a shipped model rather than becoming a plausible different
// trajectory. It is **not** bit-identical, for the plate section's superposition reason one level
// out: this model puts the force in the right-hand side *before* the solve while `MalletPlate`
// adds `influence * f` *after* it, and `solve(rhs_0 + c e)` and `solve(rhs_0) + c solve(e)` round
// differently.
//
// # The contraction factor, in closed form, and why a HEAVY mallet is the hard case
//
// Differentiating the fixed point `w = A^-1(rhs_lin(f) + c l(w_bar, F_bar))` gives
// `dG/df = +influence` against `dG/dw = J`, `J` being exactly the operator
// [`plate::VkCoupledStep::jacobian_vector`] applies. So
//
// ```text
// dw/df = -J^-1 influence      and      d eta^{n+1}/df = -( [J^-1 influence]_node + g_h )
// ```
//
// **The outer tangent is closed-form.** `mallet-plate-plan.md` section 6 and HANDOFF section 14.1
// both say a gong has "no closed-form derivative for the outer iteration, a secant or a bracket
// instead"; that is wrong, and the correction is this crate's own Jacobian-vector product,
// asserted against a finite difference in the Newton batch's Part 1. Writing
// `g_exact = [J^-1 influence]_node + g_h` for the true tangent and `g = g_s + g_h` for the
// chord's, the outer map contracts at
//
// ```text
// |df_{j+1}/df_j| = |phi'' / (1 + g phi'')| * |g - g_exact|  <=  |1 - g_exact / g|
// ```
//
// Two things follow that are worth stating before they are measured. `g - g_exact` is a deviation
// of `J` from `I` *seen along a localized column*, which is not the deviation the Newton batch's
// Part 1 measured along smooth global fixtures -- a margin measured at one fixture is a claim
// about one fixture. And the bound divides by `g = g_s + g_h`, so shrinking `g_h = k^2/M` by
// making the mallet **heavier** makes the chord converge *slower*. [`vk_drive_point_tangent`]
// measures `g_exact` so that the observed iteration count has an instrument behind it rather than
// a story.
//
// # The inner tolerance is a floor under the outer one
//
// Every `Psi(f)` carries the inner solve's `O(couple_tol)` error, so the outer residual cannot be
// driven below roughly `couple_tol * ||w|| / g`. The energy bar of this model is therefore set by
// `couple_tol` -- the *plate's* tolerance -- and not by `newton_tol`, which is the inverse of the
// self-certification model #7p has. Both halves are asserted.
//
// The inner solves are **cold-seeded**, every one of them: `vk_step` seeds its own iteration from
// `2 w^n - w^{n-1}` and this driver does not reach past it. Warm-starting the inner solve from the
// previous outer iterate would make its exit point depend on the outer history, so `Psi` would
// stop being a function of `f` and the outer loop would chase a moving target. The speedup is
// real and is reported; it is not shipped.
//
// # What does not carry over from model #7p
//
// * **Rectangles only.** `VkParams::new` builds its linear half with `Domain::Rectangle`
//   hardcoded, so the linear model's "the outline is free" is a property of the *linear* mallet.
//   No circle, no guitar outline.
// * **No acceleration field, so no `pressure()`.** [`plate::VkPlate`] carries `u`, `u_prev` and
//   the two stress-function caches and nothing else, so the stale-`accel` trap cannot arise here
//   -- and neither can the radiated read-out.
// * **The free branch's energy read-out is still a read-out bar.** A point strike feeds the
//   `{1, x, y}` rigid nullspace exactly as it does on the linear plate, and the bending form's
//   cancellation error is still quadratic in the rigid drift. That attribution carries over
//   unchanged; the membrane term adds nothing to it, since the bracket annihilates the nullspace.

/// A mallet's fixed parameters against a **nonlinear** plate.
///
/// Deliberately a separate type from [`PlateParams`] rather than a reuse of it: the fields overlap
/// but `g_s` means something different here. There it is the plate's drive-point admittance and
/// the contact equation is exact in it; here it is a *frozen tangent* that an outer iteration
/// corrects away, and the truth is [`vk_drive_point_tangent`]'s `g_exact`. One struct with one
/// `g_s` whose meaning depended on the collaborator is the shape that makes a reader believe the
/// wrong thing.
#[derive(Debug, Clone, PartialEq)]
pub struct VkPlateParams {
    /// Mallet mass `M` (kg).
    pub mass: f64,
    /// Felt stiffness `K` (N/m^alpha).
    pub stiffness: f64,
    /// Felt exponent `alpha >= 1`.
    pub alpha: f64,
    /// Hunt-Crossley damping `lam_h >= 0`.
    pub lam_h: f64,
    /// Timestep, taken from the plate.
    pub k: f64,
    /// Discrete-gradient Taylor-branch threshold.
    pub eta_tol: f64,
    /// Scalar-solve tolerance.
    pub newton_tol: f64,
    /// Scalar-solve iteration cap.
    pub newton_maxiter: usize,
    /// Flat live-node index of the strike.
    pub node: usize,
    /// Physical x of the snapped strike node (m).
    pub x_strike: f64,
    /// Physical y of the snapped strike node (m).
    pub y_strike: f64,
    /// `(k^2 / force_den) A^-1 e_node` over the live nodes -- the **linear** plate's influence
    /// column.
    ///
    /// The step uses only `influence[node]`. The rest of it is kept because it is the right-hand
    /// side of the exact outer tangent: `g_exact - g_h` is `J^-1` applied to *this vector*, and a
    /// measurement that rebuilt the column would be measuring a different one.
    pub influence: Vec<f64>,
    /// The chord's frozen tangent -- `influence[node]`, the linear plate's drive-point admittance.
    pub g_s: f64,
    /// Mallet admittance `k^2 / M`.
    pub g_h: f64,
    /// Their sum -- what each outer iteration's scalar contact solve uses.
    pub g: f64,
    /// `sqrt(K/M) / 2pi` (Hz), the `alpha = 1` contact-frequency reference.
    pub contact_frequency: f64,
    /// `pi sqrt(M/K) / k` -- steps per felt half-period. Under 8 the model warns.
    pub steps_per_contact: f64,
    /// Relative tolerance on the outer iteration's force increment.
    pub outer_tol: f64,
    /// Cap on outer iterations. Hitting it is *recorded*, never absorbed.
    pub outer_max_iter: usize,
    /// The force the outer residual is measured against: `M v0 / k`, the impulse scale of the
    /// strike itself.
    ///
    /// **Not** `|f_j|`. The contact force passes through zero at the start and the end of every
    /// contact, so a residual normalised by the current force would demand more and more absolute
    /// accuracy exactly where there is least force to be accurate about, and the first and last
    /// step of every contact would take the cap.
    pub force_scale: f64,
}

impl VkPlateParams {
    /// Validate and derive, against the nonlinear plate the mallet will strike.
    ///
    /// The five scalar checks run first and in [`check_common`]'s order, as both other
    /// constructors do.
    ///
    /// # Errors
    /// Any of [`ParamError`].
    ///
    /// # Panics
    /// If the linear half's factorization cannot back-substitute -- impossible for a `VkParams`
    /// that was constructed, since `A` is SPD and was factored then.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        vk: &plate::VkParams,
        mass: f64,
        stiffness: f64,
        alpha: f64,
        hysteresis: f64,
        strike_x: f64,
        strike_y: f64,
        strike_velocity: f64,
        gap: f64,
        eta_tol: f64,
        newton_tol: f64,
        newton_maxiter: usize,
        outer_tol: f64,
        outer_max_iter: usize,
    ) -> Result<VkPlateParams, ParamError> {
        check_common(mass, stiffness, alpha, hysteresis, gap)?;
        if outer_tol <= 0.0 {
            return Err(ParamError::NonPositiveOuterTol);
        }
        if outer_max_iter < 1 {
            return Err(ParamError::TooFewOuterIters);
        }

        let node = plate::pickup_index_at(strike_x, strike_y, &vk.lin);
        let (x_strike, y_strike) = live_coords(&vk.lin, node);

        // The linear plate's influence column, built through the *nonlinear* plate's own
        // `force_denominator` field rather than `lin.force_denominator()`: the two are equal by
        // construction, and the one that matters is the number `VkCoupledStep::new` divides
        // `f_ext` by, because that is the force this column has to describe. `k * k` and not
        // `scalar_pow(k, 2.0)` for the same reason `PlateParams` spells it that way -- it must be
        // the double the right-hand side is scaled by. `g_h` keeps `float.__pow__` because it is
        // the mallet's own constant and must equal what every other mallet computes for the same
        // `M` and `k`.
        let mut e = vec![0.0; vk.lin.n_live];
        e[node] = 1.0;
        let column = vk.lin.lu.solve(&e).expect("A is SPD and was factored");
        let scale = vk.lin.k * vk.lin.k / vk.force_denominator;
        let influence: Vec<f64> = column.iter().map(|&c| scale * c).collect();

        let g_s = influence[node];
        let g_h = scalar_pow(vk.lin.k, 2.0) / mass;

        Ok(VkPlateParams {
            mass,
            stiffness,
            alpha,
            lam_h: hysteresis,
            k: vk.lin.k,
            eta_tol,
            newton_tol,
            newton_maxiter,
            node,
            x_strike,
            y_strike,
            influence,
            g_s,
            g_h,
            g: g_s + g_h,
            contact_frequency: (stiffness / mass).sqrt() / (2.0 * std::f64::consts::PI),
            steps_per_contact: std::f64::consts::PI * (mass / stiffness).sqrt() / vk.lin.k,
            outer_tol,
            outer_max_iter,
            force_scale: (mass * strike_velocity.abs() / vk.lin.k).max(1e-30),
        })
    }

    /// Point the frozen chord at a **different** influence column — the room's.
    ///
    /// [`VkPlateParams::new`] builds the column from the plate's own factorization, which is the
    /// operator the step inverts when the plate stands alone. Load the plate with a room and it is
    /// not: `A_loaded = A + (k / 2 rho_s) T^T R T` is what the coupled step actually back-substitutes
    /// against, and its drive-point column is a different vector.
    ///
    /// This exists as one method rather than three field writes because `influence`, `g_s` and `g`
    /// are one fact spelled three ways, and a caller that set the column and forgot `g` would get a
    /// chord whose tangent belongs to a different operator than its residual — convergent, correct,
    /// and slower for a reason nothing measures.
    ///
    /// **What this does not change is the answer.** `solve_contact` solves `eta = eta_free - g f`
    /// while the chord feeds it `eta_free = w_node(f) + g_s f - z_free`, so at the fixed point the
    /// two `g_s` terms cancel and the committed force and field are whatever the plate and the
    /// contact law say they are. What moves is the rate: measured on the shipped room fixture, a
    /// chord frozen on the bare column contracts by 2.9e-04 a pass and takes about five iterations
    /// where the loaded one takes one (`docs/dev/mallet-vk-room-plan.md` §2.2).
    ///
    /// # Panics
    /// If `column` is not one entry per live node.
    pub fn retarget_column(&mut self, column: &[f64]) {
        assert_eq!(
            column.len(),
            self.influence.len(),
            "the influence column must have one entry per live node"
        );
        self.influence = column.to_vec();
        self.g_s = self.influence[self.node];
        self.g = self.g_s + self.g_h;
    }

    /// The bundle every evaluation inside a scalar contact solve takes.
    pub fn contact(&self) -> ContactParams {
        ContactParams {
            stiffness: self.stiffness,
            alpha: self.alpha,
            lam_h: self.lam_h,
            k: self.k,
            tol: self.eta_tol,
        }
    }
}

/// Why a gong step failed. Two solvers are nested, so there are two ways -- and the outer one
/// **reports the inner one's failure in its own words**, which is why the first variant carries a
/// verdict that is not about itself.
///
/// Measured while mapping this model's boundary: at a low sample rate and a hard strike, the plate
/// solve stops converging and the very next thing that happens is that the scalar contact solve
/// scans its bracket six times and finds no sign change -- and `ContactError::NoRoot`'s own text
/// says that "should be impossible for the monotone convex-potential force". It is. The force is
/// still monotone and still convex; what is not is `Psi`, because the field the residual was built
/// from is the output of an iteration that did not finish. Handing that message to a caller
/// unqualified sends them to look for a bug in the contact model, which is the one place the fault
/// is not.
///
/// So `Contact` carries `inner_converged`, and the message says which solver actually gave up.
#[derive(Debug, Clone, PartialEq)]
pub enum VkContactError {
    /// The scalar contact solve found no root.
    Contact {
        /// What the contact solve itself said.
        source: ContactError,
        /// Whether every plate solve this step reached `couple_tol`. When this is `false` the
        /// contact solve is the symptom and the plate's iteration is the cause.
        inner_converged: bool,
    },
    /// One of the plate's factorizations could not back-substitute.
    Solve(SparseLuError),
}

impl std::fmt::Display for VkContactError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VkContactError::Contact {
                source,
                inner_converged: true,
            } => write!(f, "{source}"),
            VkContactError::Contact {
                source,
                inner_converged: false,
            } => write!(
                f,
                "{source} -- but the plate's own iteration did not converge on this step either,                  and that is the likelier cause: the contact residual is built from a field the                  nonlinear solve had not finished computing. Raise couple_max_iter, loosen                  couple_tol, raise the sample rate, or strike less hard."
            ),
            VkContactError::Solve(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for VkContactError {}

impl From<SparseLuError> for VkContactError {
    fn from(e: SparseLuError) -> Self {
        VkContactError::Solve(e)
    }
}

/// One gong step's full accounting -- both solvers, and the cost that answers "how much dearer?".
#[derive(Debug, Clone, PartialEq)]
pub struct VkContactStep {
    /// The accepted displacement over the live nodes.
    pub u: Vec<f64>,
    /// The accepted stress-function cache, `None` on the linear path (which does not roll it).
    pub f_full: Option<Vec<f64>>,
    /// The contact solution the accepted field was driven by.
    pub sol: collision::ContactSolution,
    /// Outer chord iterations. **Zero on a miss** -- the force-free advance is then the answer.
    pub n_outer: usize,
    /// Did the outer force increment fall below `outer_tol * force_scale`?
    pub outer_converged: bool,
    /// Did the loop stop because the increment **stopped shrinking**, short of the tolerance?
    ///
    /// About one in-contact step in nine hundred stops contracting short of the tolerance (see
    /// [`vk_plate_step`] -- it is *not* the inner tolerance's floor, which was predicted and does
    /// not exist), and every further iteration then re-solves the whole plate for nothing. The
    /// same shape `collision::solve_contact` uses to leave its own Newton, and spelled as a
    /// negation for the same reason: a `NaN` residual compares false against everything and must
    /// exit rather than iterate to the cap.
    pub outer_stalled: bool,
    /// That relative increment at exit.
    pub outer_residual: f64,
    /// Did **every** inner plate solve reach `couple_tol`?
    pub inner_converged: bool,
    /// Inner iterations summed over every plate solve this step, the force-free one included.
    pub inner_iters: usize,
    /// Back-substitutions summed the same way -- the portable cost number.
    ///
    /// Its denominator is a **bare** `VkPlate` step at the same state, whose `n_solves` is what
    /// `vk_step` reports. That ratio, not wall-clock, is what "ten to a hundred times dearer"
    /// has to be checked against.
    pub n_solves: usize,
    /// Inner plate solves this step abandoned to Newton, summed the same way -- see
    /// [`plate::VkStep::n_fallbacks`].
    ///
    /// Summed rather than named because a contact step contains *several* coupled solves and a
    /// method label would have no referent across them, while "how many needed rescuing" still
    /// does. Non-zero only under [`plate::CoupleMethod::Auto`], and its ceiling is
    /// `n_outer + 1` -- one per plate solve, the force-free advance included.
    pub n_fallbacks: usize,
}

/// Advance the gong one step: force-free solve, then the chord, then commit.
///
/// `u`, `u_prev`, `f_cache` and `f_prev` are the plate's four buffers as they stand at time `n`;
/// the caller writes back what comes out. `s` is committed in place, exactly once, and only after
/// the outer loop has settled.
///
/// # The inner tolerance does NOT floor the outer one, and that was predicted the other way
///
/// Every `Psi(f)` carries the plate solve's own `O(couple_tol)` error, so the outer residual
/// looks like it must plateau around `couple_tol * ||w|| / (g * force_scale)` -- 9.7e-14 on the
/// shipped gong, which would put it above a `1e-14` request. **It does not happen.** Measured at
/// `outer_tol = 1e-16`, the loop reaches that tolerance in a mean of 2.6 iterations, and it does
/// so at `couple_tol = 1e-9` exactly as readily as at `1e-15`: four orders of magnitude of the
/// plate's own tolerance change the outer iteration count in the second decimal place.
///
/// The reason the floor is not there is that the inner error is a **bias, not noise**. The inner
/// solve is cold-seeded from `2 w^n - w^{n-1}` and is otherwise deterministic, so a given `f`
/// gives the same `Psi(f)` every time; the chord then converges to the perturbed map's own fixed
/// point, to whatever precision it is asked for. A floor would need `Psi` to be *discontinuous*
/// in `f` -- the shape the tension string found when a root-find branched on a reduction -- and
/// it is not: sweeping the trial force by parts in `1e12` across the worst step in a run leaves
/// the inner sweep count pinned at 5 and moves `Psi` smoothly.
///
/// What is real is much smaller. About one in-contact step in nine hundred stops contracting
/// around `5e-14` and stays there, and *that* one is not attributable to either inner tolerance:
/// tightening `couple_tol` removes it, tightening `newton_tol` removes it, and loosening
/// `newton_tol` a thousandfold does not make it worse -- every one of those perturbs the
/// trajectory, and the step moves with it. So it is a single failure to contract rather than a
/// property, and the stagnation exit exists to stop paying for it: without one, that step spent
/// all twenty of its outer iterations; with one, it spends four.
///
/// # The miss is structural, not asserted
///
/// A contact solve that returns `force == 0.0` short-circuits to the force-free advance **and
/// returns it unmodified**, so a mallet that never touches the gong leaves a trajectory
/// bit-identical to the bare plate's. That is a property of the control flow rather than of
/// floating point: driving the plate with a zero force vector would instead add `+0.0` into every
/// right-hand-side entry, which is the identity for every double except `-0.0`.
///
/// # Errors
/// [`VkContactError`] from either solver.
pub fn vk_plate_step(
    u: &[f64],
    u_prev: &[f64],
    f_cache: &[f64],
    f_prev: &[f64],
    p: &VkPlateParams,
    vk: &plate::VkParams,
    s: &mut State,
) -> Result<VkContactStep, VkContactError> {
    vk_plate_step_with(u_prev, p, vk, s, |f_ext| {
        plate::vk_step(u, u_prev, f_cache, f_prev, f_ext, vk)
    })
}

/// The same step, with the plate advance supplied by the caller — the air box's entry point.
///
/// [`vk_plate_step`] *is* this function with `trial` closed over [`plate::vk_step`], so the bare
/// gong's arithmetic is this one's by construction rather than by transcription.
///
/// `trial(f_ext)` must advance the plate **one step from the same time-`n` state every time**: the
/// chord below calls it once with `None` for the force-free advance and then once per outer
/// iteration with a force vector, and a `trial` that rolled any history between calls would have
/// every iteration after the first solving a different step. That is the same constraint the model
/// itself is under, which is why `vk_plate_step` reaches the plate's four buffers directly.
///
/// The room's client closes over a [`plate::VkCoupledStep::with_rhs`] built on the loaded
/// factorization, so the two live load terms — invariant across the chord — are assembled once
/// outside and only the strike node's entry varies here.
///
/// `u` is not an argument: every use of the time-`n` field is inside `trial`, and the one thing
/// this function still reads off the buffers itself is `u_prev[node]`, for the `eta^{n-1}` that
/// must be taken before anything is rolled.
///
/// # Errors
/// [`VkContactError`] from either solver.
pub fn vk_plate_step_with<T>(
    u_prev: &[f64],
    p: &VkPlateParams,
    vk: &plate::VkParams,
    s: &mut State,
    trial: T,
) -> Result<VkContactStep, VkContactError>
where
    T: Fn(Option<&[f64]>) -> Result<plate::VkStep, SparseLuError>,
{
    let i = p.node;
    // `eta^{n-1}` off the incoming buffers, before anything is rolled -- the same read every other
    // mallet makes on its first line, for the same reason.
    let eta_prev_n = eta_prev(u_prev[i], s);
    let z_free = free_flight(s);

    // The force-free advance. It is the outer iteration's seed *and* the miss's answer, which is
    // why it is one solve rather than a continuation from the previous step's force: a seed taken
    // from history could not give the miss a bit-identical trajectory.
    let free = trial(None)?;
    let mut n_solves = free.n_solves;
    let mut inner_iters = free.n_iters;
    let mut inner_converged = free.converged;
    let mut n_fallbacks = free.n_fallbacks;

    // `inner_converged` is threaded through the closure because a contact failure has to be able
    // to say whether the plate solve that produced its input had finished -- see `VkContactError`.
    let solve = |eta_free: f64, seed: f64, inner_ok: bool| {
        collision::solve_contact(
            eta_free,
            eta_prev_n,
            p.g,
            p.contact(),
            seed,
            p.newton_tol,
            p.newton_maxiter,
        )
        .map_err(|source| VkContactError::Contact {
            source,
            inner_converged: inner_ok,
        })
    };

    // The linear prediction -- model #7p's whole answer, and this model's first guess.
    let mut sol = solve(free.u[i] - z_free, s.penetration, inner_converged)?;
    if sol.force == 0.0 {
        s.commit(z_free, p.g_h, sol);
        return Ok(VkContactStep {
            u: free.u,
            f_full: free.f,
            sol,
            n_outer: 0,
            outer_converged: true,
            outer_stalled: false,
            outer_residual: 0.0,
            inner_converged,
            inner_iters,
            n_solves,
            n_fallbacks,
        });
    }

    let mut f_ext = vec![0.0; vk.lin.n_live];
    let mut accepted: Option<(plate::VkStep, collision::ContactSolution)> = None;
    let mut n_outer = 0usize;
    let mut outer_converged = false;
    let mut outer_stalled = false;
    let mut outer_residual = f64::NAN;
    let mut prev_residual = f64::INFINITY;

    for _ in 0..p.outer_max_iter {
        n_outer += 1;
        let force = sol.force;
        f_ext[i] = -force;
        let advanced = trial(Some(&f_ext))?;
        n_solves += advanced.n_solves;
        inner_iters += advanced.n_iters;
        inner_converged &= advanced.converged;
        n_fallbacks += advanced.n_fallbacks;

        // Where a linear plate carrying the chord's tangent would have had to start to land on the
        // node the nonlinear one actually reached. With `nonlinear = false` the two terms cancel
        // exactly and this is `u_free`, whatever `force` was.
        let u_eff = advanced.u[i] + p.g_s * force;
        let next = solve(u_eff - z_free, s.penetration, inner_converged)?;
        outer_residual = (next.force - force).abs() / p.force_scale;
        // The pair committed is the field and the contact solution that DROVE it -- a fixed-point
        // iteration verifies its input by finding that the map barely moves it, and it is the
        // input that the plate, the mallet and the discrete gradient all shared.
        accepted = Some((advanced, sol));
        sol = next;
        if outer_residual <= p.outer_tol {
            outer_converged = true;
            break;
        }
        // Negated on purpose, so a NaN residual leaves rather than iterating to the cap.
        #[allow(clippy::neg_cmp_op_on_partial_ord)]
        if !(outer_residual < prev_residual) {
            outer_stalled = true;
            break;
        }
        prev_residual = outer_residual;
    }

    let (step, driving) = accepted.expect("outer_max_iter >= 1, so the loop body ran");
    s.commit(z_free, p.g_h, driving);
    Ok(VkContactStep {
        u: step.u,
        f_full: step.f,
        sol: driving,
        n_outer,
        outer_converged,
        outer_stalled,
        outer_residual,
        inner_converged,
        inner_iters,
        n_solves,
        n_fallbacks,
    })
}

/// The **exact** outer tangent `g_exact = [J^-1 influence]_node + g_h`, by one GMRES solve.
///
/// `-g_exact` is `d eta^{n+1} / df`, the derivative the shipped chord approximates by `-g`. It
/// exists in closed form, which is what makes the plan's "no closed-form derivative for the outer
/// iteration" wrong; it costs a Krylov solve against
/// [`plate::VkCoupledStep::jacobian_vector`] and so is an instrument rather than a step.
///
/// `w` must be the **accepted** iterate of the step whose context `u`, `u_prev`, `f_prev` and
/// `f_ext` describe: `J` is evaluated at the root, which is where the contraction factor is a
/// property of the problem rather than of where the iteration happened to be.
///
/// Returns `(g_exact, response, gmres_products)`. `response` is `[J^-1 influence]_node` on its
/// own -- the **plate-only** half, in which no mallet quantity appears at all, so it is the one a
/// claim about mass-independence can be made bitwise about. `g_exact` is `response + g_h`, and
/// recovering `response` by subtracting `g_h` back off it does *not* return the same double when
/// the two differ by orders of magnitude, which is exactly the case a heavy mallet is. The
/// product count is there because it is the cost of the Newton this chord is not.
///
/// # Errors
/// If either factorization cannot back-substitute.
pub fn vk_drive_point_tangent(
    u: &[f64],
    u_prev: &[f64],
    f_prev: &[f64],
    f_ext: Option<&[f64]>,
    w: &[f64],
    p: &VkPlateParams,
    vk: &plate::VkParams,
) -> Result<(f64, f64, usize), SparseLuError> {
    vk_drive_point_tangent_with(u, u_prev, f_prev, f_ext, w, p, vk, None, &p.influence)
}

/// The same tangent, against a **supplied** operator and influence column — the room's instrument.
///
/// Both overrides are needed and neither is optional in a room, which is the point of splitting
/// this out rather than letting the bare version be used everywhere:
///
/// * the **operator**, because `jacobian_vector` inverts the theta-scheme matrix and in a room that
///   is `A_loaded`, not `A`. A tangent taken against the bare operator while the residual belongs
///   to the loaded one is the previous batch's §3 hazard exactly — convergent, correct, silently
///   describing a different problem;
/// * the **column**, because this is the only consumer that reads `influence` as a *vector*. The
///   plan's §2.1 measures the loaded and bare columns as 2.9e-04 apart at the drive point and 414%
///   apart at the far end, so the scalar `g_s` is nearly indifferent to the choice and this GMRES
///   right-hand side is not.
///
/// `rhs_lin` is deliberately not overridable here: the tangent reads only the state (through
/// `averages`) and the operator (through `jacobian_vector`), never the right-hand side, so the
/// room's two load terms cannot reach this number and there is nothing for a `with_rhs` to carry.
///
/// # Errors
/// If either factorization cannot back-substitute.
#[allow(clippy::too_many_arguments)]
pub fn vk_drive_point_tangent_with(
    u: &[f64],
    u_prev: &[f64],
    f_prev: &[f64],
    f_ext: Option<&[f64]>,
    w: &[f64],
    p: &VkPlateParams,
    vk: &plate::VkParams,
    theta: Option<&dyn plate::ThetaSolve>,
    influence: &[f64],
) -> Result<(f64, f64, usize), SparseLuError> {
    let mut ctx = plate::VkCoupledStep::new(u, u_prev, f_prev, f_ext, vk);
    if let Some(t) = theta {
        ctx.theta = t;
    }
    let av = ctx.averages(w)?;
    let mut products = 0usize;
    let sol = krylov::gmres(
        |d| {
            products += 1;
            ctx.jacobian_vector(&av, d)
        },
        influence,
        plate::NEWTON_GMRES_RESTART,
        plate::NEWTON_GMRES_MAX_PRODUCTS,
        1e-12,
    )?;
    let response = sol.x[p.node];
    Ok((response + p.g_h, response, products))
}

/// Total discrete energy `H^n` (J): the gong's own total, plus mallet kinetic, plus the averaged
/// contact potential.
///
/// `vk_energy` is [`plate::VkPlate::energy`] -- linear theta-scheme energy plus the half-step
/// membrane energy. The contact term reads the **field**, not the stored penetration, so what this
/// certifies is the consistency of the committed force with the committed displacement: exactly
/// the quantity the outer iteration drives to `outer_tol`.
pub fn vk_plate_total_energy(
    u_node: f64,
    u_prev_node: f64,
    vk_energy: f64,
    p: &VkPlateParams,
    s: &State,
) -> f64 {
    let ke = 0.5 * p.mass * scalar_pow(s.velocity(p.k), 2.0);
    let pe = s.contact_pe(u_node, u_prev_node, p.stiffness, p.alpha);
    vk_energy + ke + pe
}

/// A gong struck by a mallet -- the Rust caller's view, and what `cargo test` exercises.
///
/// Splits from the binding for the reason every model in this crate splits: the binding must hold
/// the caller's own `VKPlate` object so that `mal.plate` **is** what was passed in.
#[derive(Debug, Clone)]
pub struct MalletVkPlate {
    params: VkPlateParams,
    state: State,
    /// The gong.
    pub plate: plate::VkPlate,
    /// The last step's accounting.
    pub last: Option<VkContactStep>,
}

impl MalletVkPlate {
    /// Build from validated parameters and a nonlinear plate.
    pub fn new(
        params: VkPlateParams,
        plate: plate::VkPlate,
        gap: f64,
        strike_velocity: f64,
    ) -> MalletVkPlate {
        let u_node = plate.u[params.node];
        let state = State::at_strike(gap, strike_velocity, params.k, u_node);
        MalletVkPlate {
            params,
            state,
            plate,
            last: None,
        }
    }

    /// The parameter set this mallet was built from.
    pub fn params(&self) -> &VkPlateParams {
        &self.params
    }

    /// The mallet's own state.
    pub fn state(&self) -> &State {
        &self.state
    }

    /// Advance one step: force-free solve, chord, commit.
    ///
    /// # Errors
    /// [`VkContactError`] from either solver.
    pub fn step(&mut self) -> Result<(), VkContactError> {
        let out = vk_plate_step(
            &self.plate.u,
            &self.plate.u_prev,
            &self.plate.f,
            &self.plate.f_prev,
            &self.params,
            &self.plate.p,
            &mut self.state,
        )?;
        self.plate.u_prev = std::mem::replace(&mut self.plate.u, out.u.clone());
        if let Some(f_new) = out.f_full.clone() {
            self.plate.f_prev = std::mem::replace(&mut self.plate.f, f_new);
        }
        self.plate.n += 1;
        self.plate.n_iters = out.inner_iters;
        self.plate.converged = out.inner_converged;
        self.plate.n_solves = out.n_solves;
        self.last = Some(out);
        Ok(())
    }

    /// Total discrete energy `H^n` (J).
    pub fn energy(&self) -> f64 {
        let i = self.params.node;
        vk_plate_total_energy(
            self.plate.u[i],
            self.plate.u_prev[i],
            self.plate.energy(),
            &self.params,
            &self.state,
        )
    }

    /// Mallet velocity `delta_t- z_H` (m/s).
    pub fn mallet_velocity(&self) -> f64 {
        self.state.velocity(self.params.k)
    }
}
