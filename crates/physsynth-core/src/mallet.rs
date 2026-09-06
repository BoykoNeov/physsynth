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
use crate::membrane;
use crate::plate;

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
}

impl std::fmt::Display for ParamError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParamError::NonPositiveMass => write!(f, "mallet mass must be > 0."),
            ParamError::NonPositiveStiffness => write!(f, "felt stiffness K must be > 0."),
            ParamError::AlphaTooSmall => write!(f, "felt exponent alpha must be >= 1."),
            ParamError::NegativeHysteresis => write!(f, "hysteresis lambda_h must be >= 0."),
            ParamError::NegativeGap => write!(f, "initial gap must be >= 0."),
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
    for i in 0..u.len() {
        u[i] -= injected_at(p, force, i);
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
    for i in 0..accel.len() {
        accel[i] -= injected_at(p, force, i) / k2;
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
