//! Far-field acoustic radiation — the "air" node of `exciter -> resonator -> body/radiation`.
//!
//! Port of `physsynth/core/radiation.py`, in the three tiers the original builds in that order: a
//! read-out ([`AirRadiation`]), a constant-resistance load ([`RadiatedBody`]), and the exact
//! first-order rational impedance ([`RationalAirLoad`] driving [`ReactiveRadiatedBody`]). That
//! module's docstrings were the physics and this header documented only what the translation had
//! to decide. They were cut to a shim by deletions 2–4 (`6c23a45`) and the shim went with the last
//! of the Python (retirement plan §51), so they are carried from `6c23a45^`: the module docstring
//! below, under "The physics", and the derivations in the class and method docstrings on the
//! native items of those names, each verbatim save for its cross-references and for references to
//! Python attributes, which now name the quantity instead.
//!
//! # The physics
//!
//! A vibrating body pushes air; at a listening distance `r` a microphone hears a **radiated
//! pressure**. This module is the first, lumped tier of that air node (the human's scope decision:
//! *lumped passive impedance*, built read-out-first). It turns the body's monopole source strength
//! — its net **volume acceleration** `Q'' = d²/dt² ∫ w dA` (already exposed project-wide as the
//! bodies' `pressure()` read-out) — into the far-field pressure of a **simple source (monopole)**
//! in free space:
//!
//! ```text
//! p_far(t, r) = rho0 / (4 pi r) * Q''(t - r / c0) .
//! ```
//!
//! That is the exact free-space Green's-function solution for a compact source whose size is small
//! against the wavelength: the pressure is the volume *acceleration* scaled by `rho0 / (4 pi r)`
//! and retarded by the travel time `r / c0` (the wavefront takes `r / c0` seconds to reach the
//! listener). [`AirRadiation`] itself carries no new state on the body and **no back-reaction**: it
//! is a pure output transform, so the body radiates "for free" and its own energy balance is
//! untouched — hence it deliberately has no `energy()` (it stores and dissipates none). The passive
//! **radiation load** (the air *damping* the body, with radiated power `P_rad = R U²` accounted as
//! an energy channel and a closed-form `R_a` oracle) is the separate [`RadiatedBody`] below — batch
//! 2, the back-reaction.
//!
//! **Why this is still validatable without energy** (the project rule is closed-form checks, not
//! ears): a prescribed sinusoidal volume velocity `U(t) = U0 sin(omega t)` has volume acceleration
//! `Q'' = U0 omega cos(omega t)`, so the monopole radiates a tone of amplitude `|p_far| = rho0
//! omega U0 / (4 pi r)` — an exact amplitude/phase oracle — and the pressure falls off as `1 / r`
//! (the inverse-distance law) with a pure `r / c0` delay. Those are the tests.
//!
//! The retardation is applied as an **integer-sample** delay line (`round(r / c0 * fs)` samples),
//! so the amplitude is preserved *exactly* (a fractional/interpolating delay would act as a mild
//! low-pass and blur the amplitude oracle); the sub-sample residual is exposed as the retardation
//! residual for honesty. Turn retardation off for the pure instantaneous gain (the delay is
//! inaudible latency for a single source, but matters once several sources are summed at one
//! listener).
//!
//! Headless: no dependencies.
//!
//! # This is where bit-identity leaves the toolkit, and it is not a solver that took it
//!
//! The plan's §13.8 predicted the first divergence would come from a *solver*, from pivoting, in
//! Phases 3-6. It arrives here instead, in a Group A model with no matrix in it, and the cause is
//! a **reduction that feeds back into state**:
//!
//! ```text
//! u_free = float(np.dot(b.a, b.q - q_nm1)) / (2.0 * self.k)
//! ```
//!
//! `body.pressure()` has had exactly this reduction since batch 2 and was held to the plan's
//! Group A target — because it is a *read-out*, so its last ulp reaches a test and nothing else.
//! Here the same arithmetic decides `q^{n+1}`, so the ulp compounds. Measured on this machine
//! (numpy 2.4.6, scipy-openblas 0.3.31 `DYNAMIC_ARCH`, SkylakeX kernel), `np.dot` on contiguous
//! doubles is a **single-accumulator sequential FMA loop below 16 terms** — 1000/1000 agreement
//! with fma-per-term, 0/1000 with a plain multiply-then-add — and vectorised at 16 and above. So
//! "match the library" would mean matching a CPU dispatch rather than a piece of code: the same
//! wheel on a Haswell runner picks a different kernel, and a bit-identity assertion would go red
//! on CI while staying green here. Rust therefore sums plainly, left to right, and the parity
//! test holds the loaded state to Group A (1e-13) rather than to zero.
//!
//! Two reductions in the same file are *not* affected, and keeping the split sharp is the point:
//!
//! * `_G` uses `np.sum`, whose pairwise summation is plain left-to-right for **7 terms or fewer**
//!   (measured: 0/2000 mismatches at 1-7, roughly half at 8+, where numpy switches to eight
//!   partial accumulators). Every body in this repo has 1-5 modes, so `_G` and `_corr` come out
//!   bit-identical in practice and the Group A residual is attributable to `u_free` alone.
//! * [`AirRadiation::process`] contains no reduction at all — a scalar multiply and a delay line —
//!   so that tier stays bit-identical and the parity test asserts it exactly.
//!
//! # Three things that are not arithmetic and would each be invisible to an energy bar
//!
//! * **`int(round(x))` is round-half-to-even**, where Rust's `f64::round` is half-away-from-zero.
//!   The delay line's length is the only place it matters, and a one-sample error there passes
//!   every energy, passivity and modal test in the suite. [`py_round`] transcribes CPython's
//!   `float.__round__`, halfway case and all.
//! * **`np.isclose(a, b, rtol, atol=0.0)` is asymmetric** — the tolerance scales on the *second*
//!   argument. [`ReactiveRadiatedBody`] uses it to compare the load's timestep against the body's.
//! * **Complex division is CPython's, not the textbook formula.** [`c_div`] is Smith's algorithm
//!   as `_Py_c_quot` spells it; measured 20000/20000 agreement with `complex.__truediv__` over
//!   the `(R, omega, tau)` ranges this model reaches.
//!
//! # `piston_radiation_resistance` is deliberately absent
//!
//! It is the one function in the file that needs a Bessel `J1`, and `scipy.special.j1` is Cephes.
//! Reproducing it means either transcribing some forty-five rational-approximation coefficients
//! or inventing a series/asymptotic split and owning its accuracy analysis — neither belongs
//! inside a load batch, and the plan already has a phase for exactly this (Phase 7, "Bessel roots,
//! closed-form eigenfrequencies"). It is a stateless helper with no coupling to the four types
//! here, so the Python swap leaves that one name unswapped: the half-a-module manoeuvre
//! `operators2d` established in §11.2.1.

use std::f64::consts::PI;

use crate::body::{self, Params as BodyParams};
use crate::fmt::{py_exp, py_float};

/// Ambient air density at ~20 °C, 1 atm (kg/m^3).
pub const RHO0_AIR: f64 = 1.2041;
/// Speed of sound in ambient air (m/s).
pub const C0_AIR: f64 = 343.0;

/// Free-space acoustic radiation resistance of a compact monopole, `rho0 omega^2 / (4 pi c0)`.
///
/// `R_a(omega) = rho0 omega^2 / (4 pi c0)`. This is the resistive part of the free-space monopole
/// load: a body radiating a volume velocity `U` (m³/s) into free space hands the far field `P_rad =
/// R_a U^2` watts. It is the exact passive dual of the batch-1 free-space (`4 pi`) read-out — same
/// geometry, same `rho0`/`c0`. Acoustic (per-volume-velocity) units, **not** the mechanical `rho0 c
/// pi a^2` resistance (which multiplies a *surface* velocity and is `S^2` larger). Note the `∝
/// omega^2` frequency dependence: the constant-`R` [`RadiatedBody`] evaluates this at one reference
/// frequency; true per-mode spectral shaping is a later, frequency-dependent batch.
pub fn monopole_radiation_resistance(omega: f64, rho0: f64, c0: f64) -> f64 {
    rho0 * omega * omega / (4.0 * PI * c0)
}

/// `round(x)` as CPython's `float.__round__(None)` does it — nearest, **halves to even**.
///
/// Transcribed from `Objects/floatobject.c`: take C `round` (half away from zero), and if the
/// argument sat exactly halfway, replace it with `2.0 * round(x / 2.0)`. Rust's `f64::round` is
/// the C one, so only the halfway correction has to be added.
pub fn py_round(x: f64) -> f64 {
    let rounded = x.round();
    if (x - rounded).abs() == 0.5 {
        2.0 * (x / 2.0).round()
    } else {
        rounded
    }
}

/// A complex number as `(re, im)`. The crate depends on nothing, so there is no `Complex` type —
/// the same reasoning that produced the hand-rolled CSR in `sparse`.
pub type C64 = (f64, f64);

/// `_Py_c_prod`: `(ar br - ai bi, ar bi + ai br)`.
pub fn c_mul(a: C64, b: C64) -> C64 {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}

/// `_Py_c_quot`: Smith's algorithm, scaling by whichever denominator component is larger.
///
/// Not the textbook `a conj(b) / |b|^2`. CPython has divided this way for decades and the two
/// disagree in the last ulp; measured 20000/20000 agreement with `complex.__truediv__` on the
/// values this module forms.
pub fn c_div(a: C64, b: C64) -> C64 {
    if b.0.abs() >= b.1.abs() {
        let ratio = b.1 / b.0;
        let denom = b.0 + b.1 * ratio;
        ((a.0 + a.1 * ratio) / denom, (a.1 - a.0 * ratio) / denom)
    } else {
        let ratio = b.0 / b.1;
        let denom = b.0 * ratio + b.1;
        ((a.0 * ratio + a.1) / denom, (a.1 * ratio - a.0) / denom)
    }
}

// =================================================================================================
// Tier 1 — the read-out
// =================================================================================================

/// A construction-time rejection of [`AirParams`]. `Display` is the Python message verbatim.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AirError {
    /// `fs <= 0`.
    NonPositiveFs,
    /// `distance <= 0`.
    NonPositiveDistance,
    /// `rho0 <= 0`.
    NonPositiveRho0,
    /// `c0 <= 0`.
    NonPositiveC0,
}

impl std::fmt::Display for AirError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AirError::NonPositiveFs => write!(f, "fs must be positive."),
            AirError::NonPositiveDistance => {
                write!(f, "distance (listening radius r) must be positive.")
            }
            AirError::NonPositiveRho0 => write!(f, "rho0 (medium density) must be positive."),
            AirError::NonPositiveC0 => write!(f, "c0 (speed of sound) must be positive."),
        }
    }
}

impl std::error::Error for AirError {}

/// Validated parameters of the monopole read-out, with every derived constant.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AirParams {
    /// Sample rate (Hz).
    pub fs: f64,
    /// Timestep `1 / fs` (s).
    pub k: f64,
    /// Listening radius `r` (m).
    pub distance: f64,
    /// Ambient density (kg/m^3).
    pub rho0: f64,
    /// Sound speed (m/s).
    pub c0: f64,
    /// Whether the travel time is applied as a delay.
    pub retarded: bool,
    /// Monopole far-field gain `rho0 / (4 pi r)`.
    pub gain: f64,
    /// Travel time `r / c0` (s).
    pub retardation_seconds: f64,
    /// The travel time quantised to whole samples (0 when `retarded` is false).
    pub latency_samples: usize,
    /// Signed sub-sample rounding error, in samples. Diagnostic only.
    pub retardation_residual: f64,
}

impl AirParams {
    /// Validate and derive. The rejections run in the original's order.
    pub fn new(
        fs: f64,
        distance: f64,
        rho0: f64,
        c0: f64,
        retarded: bool,
    ) -> Result<AirParams, AirError> {
        if fs <= 0.0 {
            return Err(AirError::NonPositiveFs);
        }
        if distance <= 0.0 {
            return Err(AirError::NonPositiveDistance);
        }
        if rho0 <= 0.0 {
            return Err(AirError::NonPositiveRho0);
        }
        if c0 <= 0.0 {
            return Err(AirError::NonPositiveC0);
        }
        let k = 1.0 / fs;
        let gain = rho0 / (4.0 * PI * distance);
        let retardation_seconds = distance / c0;
        let delay_exact = retardation_seconds * fs;
        // `int(round(...))` — halves to EVEN. A negative can never arise here because all four
        // scalars are positive; the `as usize` cast saturates rather than wrapping, and the only
        // way to reach that is a delay longer than the address space.
        let latency_samples = if retarded {
            py_round(delay_exact) as usize
        } else {
            0
        };
        let retardation_residual = if retarded {
            delay_exact - latency_samples as f64
        } else {
            0.0
        };
        Ok(AirParams {
            fs,
            k,
            distance,
            rho0,
            c0,
            retarded,
            gain,
            retardation_seconds,
            latency_samples,
            retardation_residual,
        })
    }
}

/// Volume acceleration in, far-field pressure out — a gain and an integer-sample delay line.
///
/// No reduction anywhere in the path, so this tier is bit-identical to the original and the
/// parity test says so exactly rather than to a tolerance.
#[derive(Debug, Clone)]
pub struct AirRadiation {
    p: AirParams,
    buf: Vec<f64>,
    idx: usize,
    n: usize,
}

impl AirRadiation {
    /// Build from validated parameters, with the delay line zero-filled (silence in transit).
    pub fn new(p: AirParams) -> AirRadiation {
        AirRadiation {
            buf: vec![0.0; p.latency_samples],
            p,
            idx: 0,
            n: 0,
        }
    }

    /// The parameters and every derived constant.
    pub fn params(&self) -> &AirParams {
        &self.p
    }

    /// The delay line, in the original's `_buf` layout (the write cursor is [`AirRadiation::idx`]).
    pub fn buf(&self) -> &[f64] {
        &self.buf
    }

    /// The write cursor into [`AirRadiation::buf`].
    pub fn idx(&self) -> usize {
        self.idx
    }

    /// Samples processed.
    pub fn n(&self) -> usize {
        self.n
    }

    /// One volume-acceleration sample `Q''` to the far-field pressure `p_far` (Pa).
    pub fn process(&mut self, volume_accel: f64) -> f64 {
        let p = self.p.gain * volume_accel;
        self.n += 1;
        if self.p.latency_samples == 0 {
            return p;
        }
        let out = self.buf[self.idx];
        self.buf[self.idx] = p;
        self.idx = (self.idx + 1) % self.buf.len();
        out
    }

    /// Clear the delay line and the sample counter — reuse on a new run.
    pub fn reset(&mut self) {
        self.buf.iter_mut().for_each(|v| *v = 0.0);
        self.idx = 0;
        self.n = 0;
    }
}

// =================================================================================================
// The rank-1 dashpot, shared by both loaded bodies
// =================================================================================================

/// The scalar driving-point factor `G` and the per-mode correction prefactor, in one place.
///
/// ```text
/// G       = (k/2) sum_i a_i^2 / (m_i (1 + sigma_i k))
/// corr_i  = k^2 a_i / (m_i (1 + sigma_i k))
/// ```
///
/// Both loaded bodies call this rather than each computing its own, because two of the suite's
/// bit-identity claims — `R = 0` is a bare body, `M_a = inf` is [`RadiatedBody`] — compare one
/// implementation against *itself*, and a one-ulp drift between two copies of this expression
/// would break them without moving any physics bar.
///
/// The sum is left to right, which is what `np.sum` does for up to seven terms; module header.
pub fn rank_one(p: &BodyParams) -> (f64, Vec<f64>) {
    let kk = p.k * p.k;
    let mut acc = 0.0;
    let mut corr = vec![0.0; p.n_modes()];
    for (i, c) in corr.iter_mut().enumerate() {
        let denom = p.m[i] * (1.0 + p.sigma[i] * p.k);
        acc += p.a[i] * p.a[i] / denom;
        *c = (kk * p.a[i]) / denom;
    }
    (0.5 * p.k * acc, corr)
}

/// The force-free centered volume velocity `a^T (q~^{n+1} - q^{n-1}) / (2k)`.
///
/// **This is the reduction that costs the batch its bit-identity.** The original spells it
/// `np.dot`, which fuses and (past sixteen terms) vectorises; summed plainly here. See the module
/// header for why matching OpenBLAS would be matching a CPU rather than a piece of code.
pub fn free_volume_velocity(q: &[f64], q_nm1: &[f64], p: &BodyParams) -> f64 {
    let mut acc = 0.0;
    for i in 0..p.n_modes() {
        acc += p.a[i] * (q[i] - q_nm1[i]);
    }
    acc / (2.0 * p.k)
}

/// `out = q - scale * corr` — the rank-1 correction of `q^{n+1}`.
pub fn correct_into(q: &[f64], scale: f64, corr: &[f64], out: &mut [f64]) {
    for i in 0..q.len() {
        out[i] = q[i] - scale * corr[i];
    }
}

/// `out = (q - 2 q_prev + q_nm1) / k^2` — `q''` from the *corrected* second difference.
///
/// The original recomputes this after every rank-1 correction so that `pressure()` carries the
/// load; the grouping is NumPy's, left to right.
pub fn refresh_accel(q: &[f64], q_prev: &[f64], q_nm1: &[f64], k: f64, out: &mut [f64]) {
    let k2 = k * k;
    for i in 0..q.len() {
        out[i] = ((q[i] - 2.0 * q_prev[i]) + q_nm1[i]) / k2;
    }
}

// =================================================================================================
// Tier 2 — the constant-resistance load
// =================================================================================================

/// A modal body loaded by a constant radiation resistance — the passive back-reaction.
///
/// (Until retirement plan §51 the Python binding did not use this owning struct: it drove the
/// free functions above against Python-owned arrays, as everywhere else in this crate.)
///
/// Batch 1 read the body's monopole out "for free"; here the air pushes back. The radiation
/// presents a resistance `R` (acoustic, Pa·s/m³) to the body's net **volume velocity** `U = sum_i
/// a_i q_i'` (the same weights `a_i` that read the volume acceleration also couple the uniform
/// back-pressure `R U` into each mode, by reciprocity), so the air removes power `P_rad = R U^2`
/// and hands it to the far field. That lost energy is tracked as the radiated energy (`= integral
/// P_rad dt`), giving the passivity identity
///
/// ```text
/// E_body^n  +  integral_0^n P_rad dt  =  const      (lossless body, any R >= 0),
/// ```
///
/// the radiation analogue of the `sigma`-damping energy channel — a closed-form check, not ears.
/// `R = 0` is **bit-identical** to a bare [`crate::body::ModalBody`].
///
/// **Why this is a rank-1 *implicit* dashpot (and why that matters).** The load force on mode `i`
/// is `-a_i R U` with `U` the **centered** volume velocity `a^T (q^{n+1} - q^{n-1}) / (2k)`,
/// centered (implicit), exactly like the body's own `2 sigma q'` term, which is what makes the
/// discrete dissipation `k R U^2` provably non-negative (an explicit one-sided `U` goes unstable at
/// large `R`). Because the coupling is a single scalar `U`, the implicit solve collapses to one
/// scalar (Sherman–Morrison): take the force-free step `q~^{n+1} = ModalBody::step(force)`, read
/// its free centered velocity `U_free`, then
///
/// ```text
/// U = U_free / (1 + R G) ,      G = (k/2) sum_i a_i^2 / (m_i (1 + sigma_i k)) ,
/// ```
///
/// and correct each mode `q_i^{n+1} = q~_i - [k^2 a_i / (m_i (1 + sigma_i k))] R U`. The corrected
/// state's centered volume velocity is exactly this `U` (`U_free - R U G = U`), so the energy
/// increment `k R U^2` telescopes to machine precision. Since `1 + R G >= 1` for any `R >= 0` the
/// solve is never singular: the load is **unconditionally** passive — no CFL, no stability guard
/// (unlike the bridge springs, whose rank-1 block can go *negative*). Mirrors the bow's rank-1
/// scalar solve, but linear (no Newton).
///
/// Being a drop-in for [`crate::body::ModalBody`] (it delegates every read accessor), it also slots
/// straight into a [`crate::connection::StringBodyBridge`] as the body, so the full instrument
/// chain conserves `E_string + E_body + E_conn + integral P_rad` with no edit to it.
#[derive(Debug, Clone)]
pub struct RadiatedBody {
    body: body::ModalBody,
    r: f64,
    g: f64,
    corr: Vec<f64>,
    /// `integral P_rad dt` — the energy handed to the far field.
    pub radiated_energy: f64,
    /// The last centered volume velocity `U^n`.
    pub volume_velocity: f64,
    n: usize,
}

impl RadiatedBody {
    /// Load `body` with acoustic resistance `r >= 0`. `Err` carries the Python message.
    pub fn new(body: body::ModalBody, r: f64) -> Result<RadiatedBody, &'static str> {
        if r < 0.0 {
            return Err("radiation resistance R must be >= 0.");
        }
        let (g, corr) = rank_one(body.params());
        Ok(RadiatedBody {
            body,
            r,
            g,
            corr,
            radiated_energy: 0.0,
            volume_velocity: 0.0,
            n: 0,
        })
    }

    /// The body being loaded.
    pub fn body(&self) -> &body::ModalBody {
        &self.body
    }

    /// The body being loaded, mutably.
    pub fn body_mut(&mut self) -> &mut body::ModalBody {
        &mut self.body
    }

    /// The radiation resistance `R`.
    pub fn r(&self) -> f64 {
        self.r
    }

    /// The scalar driving-point factor `G`.
    pub fn g(&self) -> f64 {
        self.g
    }

    /// Steps taken.
    pub fn n(&self) -> usize {
        self.n
    }

    /// Advance one step: force-free body advance, scalar solve, rank-1 correction.
    pub fn step(&mut self, force: f64) {
        let q_nm1 = self.body.q_prev().to_vec();
        self.body.step(force);
        let n_modes = self.body.params().n_modes();
        let k = self.body.params().k;
        let u_free = free_volume_velocity(self.body.q(), &q_nm1, self.body.params());
        let u = u_free / (1.0 + self.r * self.g);
        let mut next = vec![0.0; n_modes];
        correct_into(self.body.q(), self.r * u, &self.corr, &mut next);
        self.body.q_mut().copy_from_slice(&next);
        let mut accel = vec![0.0; n_modes];
        refresh_accel(self.body.q(), self.body.q_prev(), &q_nm1, k, &mut accel);
        self.body.accel_mut().copy_from_slice(&accel);
        self.radiated_energy += k * self.r * u * u;
        self.volume_velocity = u;
        self.n += 1;
    }

    /// `E_body + integral P_rad dt` (Joules) — the conserved total, not `body.energy()`.
    pub fn energy(&self) -> f64 {
        self.body.energy() + self.radiated_energy
    }

    /// Radiated pressure read-out, carrying the load.
    pub fn pressure(&self) -> f64 {
        self.body.pressure()
    }

    /// Set the body's modal state and reset the radiated channel.
    pub fn set_state(&mut self, q0: &[f64], v0: &[f64]) {
        self.body.set_state(q0, v0);
        self.radiated_energy = 0.0;
        self.volume_velocity = 0.0;
        self.n = 0;
    }

    /// Zero the body state and the radiated channel.
    pub fn reset(&mut self) {
        let zeros = vec![0.0; self.body.params().n_modes()];
        self.set_state(&zeros, &zeros);
    }
}

// =================================================================================================
// Tier 3 — the exact first-order rational impedance
// =================================================================================================

/// A construction-time rejection of [`LoadParams`]. `Display` is the Python message verbatim.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadError {
    /// `fs <= 0`.
    NonPositiveFs,
    /// `R < 0`.
    NegativeR,
    /// `M_a <= 0`, or NaN. `+inf` is allowed and means the constant-`R` load.
    NonPositiveMass,
    /// `rho0 <= 0` or `c0 <= 0`.
    NonPositiveMedium,
    /// `from_sphere` with a non-positive radius.
    NonPositiveRadius,
}

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoadError::NonPositiveFs => write!(f, "fs must be positive."),
            LoadError::NegativeR => write!(f, "radiation resistance R must be >= 0."),
            LoadError::NonPositiveMass => write!(
                f,
                "radiation mass M_a must be positive (inf = constant-R load)."
            ),
            LoadError::NonPositiveMedium => write!(f, "rho0 and c0 must be positive."),
            LoadError::NonPositiveRadius => write!(f, "sphere radius must be positive."),
        }
    }
}

impl std::error::Error for LoadError {}

/// Validated parameters of the rational air load, with every derived constant.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LoadParams {
    /// Sample rate (Hz).
    pub fs: f64,
    /// Timestep `1 / fs` (s).
    pub k: f64,
    /// Acoustic radiation resistance (Pa·s/m^3).
    pub r: f64,
    /// Acoustic radiation mass (kg/m^4); `+inf` is the constant-`R` load.
    pub m_a: f64,
    /// Ambient density (kg/m^3) — geometry only, never the time step.
    pub rho0: f64,
    /// Sound speed (m/s) — geometry only, never the time step.
    pub c0: f64,
    /// The trapezoid's effective resistance `R / (1 + k R / (2 M_a))`.
    pub r_eff: f64,
    /// Relaxation time `M_a / R`; `+inf` when either `R = 0` or the mass is infinite.
    pub tau: f64,
    /// Equivalent pulsating-sphere radius, when the `(R, M_a)` pair is sphere-consistent.
    pub sphere_radius: Option<f64>,
    /// Equivalent sphere area, on the same condition.
    pub sphere_area: Option<f64>,
}

impl LoadParams {
    /// Validate and derive. The rejections run in the original's order.
    pub fn new(fs: f64, r: f64, m_a: f64, rho0: f64, c0: f64) -> Result<LoadParams, LoadError> {
        if fs <= 0.0 {
            return Err(LoadError::NonPositiveFs);
        }
        if r < 0.0 {
            return Err(LoadError::NegativeR);
        }
        // The original writes `not (M_a > 0.0)`, which catches zero, negatives AND NaN while
        // letting `+inf` through. Spelled out here because a bare `m_a <= 0.0` would accept NaN,
        // and `!(m_a > 0.0)` — the literal transcription — is a clippy lint.
        if m_a.is_nan() || m_a <= 0.0 {
            return Err(LoadError::NonPositiveMass);
        }
        if rho0 <= 0.0 || c0 <= 0.0 {
            return Err(LoadError::NonPositiveMedium);
        }
        let k = 1.0 / fs;
        // `M_a = inf` makes `k R / (2 inf)` exactly 0.0, so `R_eff = R` bit for bit — which is
        // what collapses this whole tier onto tier 2. The grouping is the original's.
        let r_eff = r / (1.0 + k * r / (2.0 * m_a));
        let tau = if r == 0.0 || m_a.is_infinite() {
            f64::INFINITY
        } else {
            m_a / r
        };
        let (mut sphere_radius, mut sphere_area) = (None, None);
        if r > 0.0 && m_a.is_finite() {
            let a_eq = c0 * m_a / r;
            let s_eq = rho0 * c0 / r;
            if (4.0 * PI * a_eq * a_eq - s_eq).abs() <= 1e-9 * s_eq {
                sphere_radius = Some(a_eq);
                sphere_area = Some(s_eq);
            }
        }
        Ok(LoadParams {
            fs,
            k,
            r,
            m_a,
            rho0,
            c0,
            r_eff,
            tau,
            sphere_radius,
            sphere_area,
        })
    }

    /// The physically consistent pulsating sphere: `R = rho0 c0 / (4 pi a^2)`, `M_a = rho0/(4 pi a)`.
    pub fn from_sphere(fs: f64, radius: f64, rho0: f64, c0: f64) -> Result<LoadParams, LoadError> {
        if radius <= 0.0 {
            return Err(LoadError::NonPositiveRadius);
        }
        let area = 4.0 * PI * radius * radius;
        LoadParams::new(fs, rho0 * c0 / area, rho0 / (4.0 * PI * radius), rho0, c0)
    }
}

/// Why [`RationalAirLoad::far_field_pressure`] refused. `Display` is the Python message verbatim.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FarFieldError {
    /// The `(R, M_a)` pair has no radius, so there is no surface pressure to scale.
    NotSphereConsistent,
    /// `distance <= 0`.
    NonPositiveDistance,
}

impl std::fmt::Display for FarFieldError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FarFieldError::NotSphereConsistent => write!(
                f,
                "far_field_pressure needs a sphere-consistent load (4 pi a^2 == rho0 c0 / R); \
                 build it with RationalAirLoad.from_sphere(...)."
            ),
            FarFieldError::NonPositiveDistance => write!(f, "distance must be positive."),
        }
    }
}

impl std::error::Error for FarFieldError {}

/// Why [`RationalAirLoad::loaded_mode`] refused. `Display` is the Python message verbatim.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LoadedModeError {
    /// `mass <= 0`.
    NonPositiveMass,
    /// `omega0 <= 0`.
    NonPositiveOmega0,
    /// The fixed point did not reach `tol` within the iteration cap.
    NotConverged {
        /// The cap that was hit.
        iterations: usize,
        /// The "last relative step". Always `0.0` — see [`RationalAirLoad::loaded_mode`].
        last: f64,
        /// The tolerance that was asked for.
        tol: f64,
    },
}

impl std::fmt::Display for LoadedModeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoadedModeError::NonPositiveMass => write!(f, "mass must be positive."),
            LoadedModeError::NonPositiveOmega0 => write!(f, "omega0 must be positive."),
            LoadedModeError::NotConverged {
                iterations,
                last,
                tol,
            } => write!(
                f,
                "loaded_mode did not converge in {iterations} iterations (last relative step \
                 {} > tol {}); the added mass is comparable to the modal mass, which is outside \
                 this weak-loading formula's range.",
                py_exp(*last, 3),
                py_exp(*tol, 1)
            ),
        }
    }
}

impl std::error::Error for LoadedModeError {}

/// The air as a first-order positive-real impedance: a resistance in parallel with an inertance.
///
/// Batch 2's [`RadiatedBody`] loads a body with a single constant resistance `R`; real air is
/// frequency-dependent (`R_a(omega) = rho0 omega^2 / (4 pi c0)` for a free-space monopole), so a
/// constant over-damps the low modes and under-damps the high ones. The fix needs **no filter
/// approximation**: the exact acoustic radiation impedance of a **pulsating sphere** of radius `a`
/// is already a first-order rational function of `j omega`,
///
/// ```text
/// Z_a(j omega) = (rho0 c0 / S) * j k a / (1 + j k a) ,    S = 4 pi a^2 ,  k = omega / c0 ,
/// ```
///
/// which is exactly a resistance in **parallel** with an acoustic inertance:
///
/// ```text
/// Z_a = R * j omega tau / (1 + j omega tau) ,
/// R = rho0 c0 / S ,   M_a = rho0 / (4 pi a) ,   tau = M_a / R = a / c0 .
/// ```
///
/// `M_a` is the classic **radiation mass** — air dragged along but not radiated. The circuit
/// reading *is* the numerical scheme: the volume velocity splits, `U = U_R + U_L`, with `p = R
/// U_R = M_a dU_L/dt`. The resistor branch is radiated (lost to the far field), the inertance
/// branch is stored (returned later). That split makes the frequency dependence **passive by
/// construction** — the network is the proof, so there is no positive-realness check to pass and no
/// stability guard to tune.
///
/// Two closed-form anchors bracket it, and the first one is batch 2's own helper:
///
/// | limit | `Re Z_a` | anchor |
/// |---|---|---|
/// | `ka -> 0` | `rho0 omega^2 / (4 pi c0)` | exactly [`monopole_radiation_resistance`] |
/// | `ka -> oo` | `rho0 c0 / S` | plane-wave (fully loaded) saturation |
///
/// so this class does not supersede the batch-2 helper — it *contains* it as its low-frequency
/// limit. `Im Z_a` peaks at `ka = 1`.
///
/// **Parameters are the effective coefficients** `(R, M_a)`, independently, **not** a radius.
/// Besides being the general first-order load (reusable for a fitted horn/piston termination), it
/// is the only parameterisation that can reach `M_a -> infinity` at fixed `R` — the
/// constant-resistance load, i.e. batch 2 exactly. A sphere ties `M_a` to `R` and cannot.
/// [`LoadParams::from_sphere`] is the physically-consistent constructor for those who want the
/// radius.
///
/// **Discretisation: trapezoid on the inertance, which is the bilinear transform.** With `L^- =
/// U_L^{n-1/2}` known,
///
/// ```text
/// M_a (U_L^{n+1/2} - U_L^{n-1/2}) / k = p^n ,   U_L^n = (U_L^{n+1/2} + U_L^{n-1/2}) / 2 ,
/// p^n = R (U^n - U_L^n)   ==>   p = R_eff (U - L^-) ,  R_eff = R / (1 + k R / (2 M_a)) .
/// ```
///
/// So on the body's side the whole batch is `R -> R_eff` plus a known offset `L^-` — batch 2's
/// scalar Sherman-Morrison solve is untouched in shape. Because `R_eff` lies in `[0, R]` for every
/// `M_a > 0`, `1 + R_eff G >= 1` still holds: the solve is never singular, so the load is
/// **unconditionally passive** at any `R`, any `M_a`, any `k`. No CFL, no guard.
///
/// The exact discrete impedance is `Z_a` evaluated at the **pre-warped** frequency `s = (2j / k)
/// tan(omega k / 2)` ([`RationalAirLoad::impedance_discrete`]), not at `j omega` — that is the
/// bilinear transform's signature, and comparing a measured sweep to [`RationalAirLoad::impedance`]
/// instead shows a growing `O((omega k)^2)` error that looks like a scheme bug but is not.
///
/// **The radiation mass.** Acoustic radiation mass / inertance (kg/m⁴), `> 0`. Default `inf` — the
/// constant-resistance load, **bit-identical** to [`RadiatedBody`] (in IEEE arithmetic `k R / (2
/// inf) = 0` and `p / inf = 0`, so `R_eff = R` exactly and the auxiliary state stays exactly zero).
#[derive(Debug, Clone)]
pub struct RationalAirLoad {
    p: LoadParams,
    /// The inertance branch's velocity `U_L^{n-1/2}` — the one auxiliary state.
    pub u_l: f64,
    /// `integral R U_R^2 dt` — energy handed to the far field.
    pub radiated_energy: f64,
    /// Last centered total volume velocity `U^n`.
    pub volume_velocity: f64,
    /// Last load pressure `p^n`.
    pub pressure_load: f64,
    n: usize,
}

impl RationalAirLoad {
    /// Build from validated parameters, at rest.
    pub fn new(p: LoadParams) -> RationalAirLoad {
        RationalAirLoad {
            p,
            u_l: 0.0,
            radiated_energy: 0.0,
            volume_velocity: 0.0,
            pressure_load: 0.0,
            n: 0,
        }
    }

    /// The parameters and every derived constant.
    pub fn params(&self) -> &LoadParams {
        &self.p
    }

    /// Steps taken.
    pub fn n(&self) -> usize {
        self.n
    }

    /// Load pressure `p^n` and centered volume velocity `U^n`, *without* committing.
    ///
    /// Forming `u*` first and multiplying by `R_eff` — rather than any algebraically equal
    /// alternative — is what keeps `M_a = inf` bit-identical to [`RadiatedBody`].
    ///
    /// `u_free` is the body's force-free centered volume velocity and `G` its scalar driving-point
    /// factor `(k/2) sum_i a_i^2 / (m_i (1 + sigma_i k))`; the coupled scalar
    ///
    /// ```text
    /// u* = (u_free - L^-) / (1 + R_eff G) ,   p = R_eff u* ,   U = u* + L^-
    /// ```
    ///
    /// is exact (the last identity because `u*(1 + R_eff G) = u_free - L^-`). `G = 0` is the
    /// **prescribed-velocity drive** — a rigid piston moving at `U = u_free` regardless of the load
    /// — which is how the impedance oracle drives this object standalone.
    pub fn solve(&self, u_free: f64, g: f64) -> (f64, f64) {
        let u_star = (u_free - self.u_l) / (1.0 + self.p.r_eff * g);
        let p = self.p.r_eff * u_star;
        (p, u_star + self.u_l)
    }

    /// Advance the auxiliary state on an accepted `(p, U)` and book the energy split.
    ///
    /// The inertance branch takes `U_L` (stored, returned later); the resistor branch takes `U_R =
    /// U - U_L` (radiated, gone). The trapezoid makes the stored increment exactly `k p U_L^n` and
    /// the dissipated one exactly `k R U_R^2 >= 0`, and together they equal the `k p U^n` the body
    /// sheds — so the three-way identity telescopes to machine precision.
    pub fn commit(&mut self, p: f64, u: f64) {
        let u_l_mid = self.u_l + 0.5 * self.p.k * p / self.p.m_a;
        let u_r = u - u_l_mid;
        self.u_l += self.p.k * p / self.p.m_a;
        self.radiated_energy += self.p.k * self.p.r * u_r * u_r;
        self.volume_velocity = u;
        self.pressure_load = p;
        self.n += 1;
    }

    /// [`RationalAirLoad::solve`] then [`RationalAirLoad::commit`] — the standalone driven form.
    pub fn step(&mut self, u_free: f64, g: f64) -> (f64, f64) {
        let (p, u) = self.solve(u_free, g);
        self.commit(p, u);
        (p, u)
    }

    /// Kinetic energy of the radiation mass, `1/2 M_a (U_L^{n+1/2})^2`.
    ///
    /// Exactly zero for `M_a = inf`, where the auxiliary state is identically zero and the product
    /// would otherwise be `inf * 0 = NaN`.
    pub fn stored_energy(&self) -> f64 {
        if self.p.m_a.is_infinite() {
            return 0.0;
        }
        0.5 * self.p.m_a * self.u_l * self.u_l
    }

    /// The air's whole share: stored plus radiated.
    ///
    /// Unlike batch 2's purely dissipative channel this air can also *give back*, so the stored
    /// term is genuinely new structure — a wrong reactance shows up as drift in the identity
    /// `E_body + air.energy() = const`.
    pub fn energy(&self) -> f64 {
        self.stored_energy() + self.radiated_energy
    }

    /// Continuous acoustic impedance `Z_a(j omega) = R j omega tau / (1 + j omega tau)`.
    ///
    /// The physics. `Re Z -> rho0 omega^2 / (4 pi c0)` as `omega tau -> 0` (agreeing with
    /// [`monopole_radiation_resistance`] for a sphere-consistent load) and `-> R` as `omega tau ->
    /// inf`. Constant-`R` loads (`M_a = inf`) return `R` at every frequency — which is precisely
    /// batch 2's approximation, stated.
    pub fn impedance(&self, omega: f64) -> C64 {
        if self.p.tau.is_infinite() {
            return (self.saturated(), 0.0);
        }
        self.z_at((0.0, omega))
    }

    /// The *scheme's* impedance: `Z_a` at the pre-warped `s = (2j / k) tan(omega k / 2)`.
    ///
    /// Trapezoid is the bilinear transform, so this — not [`RationalAirLoad::impedance`] — is what
    /// a measured sweep matches to machine precision.
    pub fn impedance_discrete(&self, omega: f64) -> C64 {
        if self.p.tau.is_infinite() {
            return (self.saturated(), 0.0);
        }
        let s = (0.0, 2.0 * (0.5 * omega * self.p.k).tan() / self.p.k);
        self.z_at(s)
    }

    /// The frequency-independent value both impedance read-outs return when `tau` is infinite:
    /// `R` for a constant-`R` load, and `0` for the decoupled `R = 0` case.
    fn saturated(&self) -> f64 {
        if self.p.m_a.is_infinite() {
            self.p.r
        } else {
            0.0
        }
    }

    /// `R s tau / (1 + s tau)`, associated and divided exactly as CPython would.
    fn z_at(&self, s: C64) -> C64 {
        let num = c_mul(c_mul((self.p.r, 0.0), s), (self.p.tau, 0.0));
        let st = c_mul(s, (self.p.tau, 0.0));
        c_div(num, (1.0 + st.0, st.1))
    }

    /// Closed-form `(omega_eff, alpha)` of one weakly loaded mode — both parts of `Z_a`.
    ///
    /// A fixed point, iterated to `tol`, which **refuses** rather than returning the last iterate.
    ///
    /// The refusal's message quotes a "last relative step" that is **always exactly zero**. That
    /// is not a translation slip: the original's `for/else` reads `abs(w_next - w) / w_next` after
    /// the loop body has already assigned `w = w_next`, so the difference is identically `0.0`.
    /// Transcribed as it stands — nothing in the repo matches on the text (`web/serialize.py`,
    /// the one caller, catches the `ValueError` and censors the point), and quietly making a
    /// message more informative during a port is how a port stops being a port. It is a cosmetic
    /// bug in the Python original and belongs in a fix there, applied to both sides at once.
    ///
    /// A single mode `m q'' + m omega0^2 q = -a p` driven against this load, with `p = Z_a U` and
    /// `U = a q'`, becomes (at `q' = j omega q`)
    ///
    /// ```text
    /// [m + a^2 Im Z_a(omega) / omega] q''  +  a^2 Re Z_a(omega) q'  +  m omega0^2 q  =  0 ,
    /// ```
    ///
    /// so the air does **two** things and the batch needs both to be right:
    ///
    /// * the **reactance** is an *added mass* `m_add = a^2 Im Z_a / omega` — the body gets heavier
    ///   and its pitch drops (a constant-`R` load cannot: its `Im Z` is zero);
    /// * the **resistance** damps it at `alpha = a^2 Re Z_a / (2 m_eff)`, evaluated at the
    ///   *shifted* frequency.
    ///
    /// Both depend on the frequency they shift, so this solves the fixed point `omega_eff = omega0
    /// sqrt(m / m_eff(omega_eff))` by iteration — to `tol`, and it **raises** rather than silently
    /// returning the last iterate if that is not reached.
    ///
    /// This is a weak-loading result: valid while `alpha << omega`, with a residual second order in
    /// that ratio (~1% at `alpha / omega ~ 1%`). It is a *modeling* oracle, independent of the
    /// time-stepping scheme — it reads only `impedance`.
    pub fn loaded_mode(
        &self,
        omega0: f64,
        weight: f64,
        mass: f64,
        iterations: usize,
        tol: f64,
    ) -> Result<(f64, f64), LoadedModeError> {
        if mass <= 0.0 {
            return Err(LoadedModeError::NonPositiveMass);
        }
        if omega0 <= 0.0 {
            return Err(LoadedModeError::NonPositiveOmega0);
        }
        let w0 = omega0;
        let mut w = w0;
        let a2 = weight * weight;
        let mut converged = false;
        for _ in 0..iterations {
            let w_next = w0 * (mass / (mass + a2 * self.impedance(w).1 / w)).sqrt();
            if (w_next - w).abs() <= tol * w_next {
                w = w_next;
                converged = true;
                break;
            }
            w = w_next;
        }
        if !converged {
            return Err(LoadedModeError::NotConverged {
                iterations,
                last: 0.0,
                tol,
            });
        }
        let z = self.impedance(w);
        let m_eff = mass + a2 * z.1 / w;
        Ok((w, a2 * z.0 / (2.0 * m_eff)))
    }

    /// Far-field pressure at `r` from the sphere's own surface pressure, `(a / r) p_load`.
    ///
    /// `p_load` of `None` reads the last committed load pressure.
    ///
    /// **Not** the batch-1 read-out, and the difference is a trap worth stating. [`AirRadiation`]
    /// is the `a -> 0` compact-source limit `rho0 Q'' / (4 pi r)`; a *finite* sphere additionally
    /// low-passes the far field by `1 / (1 + j k a)`, so a naive power balance against batch 1
    /// misses by `1 / (1 + (ka)^2)` — which reads as a bug and is not one. With this form the
    /// balance is **exact at every `ka`**, because `S |Z_a|^2 / (rho0 c0) == Re Z_a` identically:
    ///
    /// ```text
    /// 4 pi r^2 <p_far^2> / (rho0 c0)  ==  <R U_R^2>   =   the booked radiated power.
    /// ```
    ///
    /// The pure travel delay `(r - a) / c0` is *not* applied (it is latency, not level — hand the
    /// result to [`AirRadiation`] if you want the delay line). Requires a sphere-consistent load: a
    /// general `(R, M_a)` pair has no radius, and this read-out is the one place where that
    /// interpretation is genuinely needed, so it refuses rather than guessing.
    pub fn far_field_pressure(
        &self,
        distance: f64,
        p_load: Option<f64>,
    ) -> Result<f64, FarFieldError> {
        let Some(a) = self.p.sphere_radius else {
            return Err(FarFieldError::NotSphereConsistent);
        };
        if distance <= 0.0 {
            return Err(FarFieldError::NonPositiveDistance);
        }
        Ok(a / distance * p_load.unwrap_or(self.pressure_load))
    }

    /// Zero the auxiliary state, the radiated channel and the counters.
    pub fn reset(&mut self) {
        self.u_l = 0.0;
        self.radiated_energy = 0.0;
        self.volume_velocity = 0.0;
        self.pressure_load = 0.0;
        self.n = 0;
    }
}

/// `np.isclose(a, b, rtol=rtol, atol=0.0)` — **asymmetric**: the tolerance scales on `b`.
pub fn isclose(a: f64, b: f64, rtol: f64) -> bool {
    (a - b).abs() <= rtol * b.abs()
}

/// The message `ReactiveRadiatedBody` refuses a mismatched timestep with.
///
/// Split out of the constructor so the binding can raise it without owning a native body.
pub fn timestep_mismatch(load_fs: f64, body_k: f64) -> String {
    format!(
        "load fs ({}) must match the body's ({}) — the trapezoid's R_eff and the body's \
         centered velocity share one timestep.",
        py_float(load_fs),
        py_float(1.0 / body_k)
    )
}

/// A modal body loaded by the frequency-dependent radiation impedance.
///
/// Batch 2's [`RadiatedBody`] is the purely **resistive** load; what is new here is the
/// **reactance** — the radiation mass — which is what turns one number into `Z_a(omega)`. The
/// audible consequence is the one a constant `R` cannot produce: high partials radiate better and
/// die first, at the per-mode rate `alpha_i = a_i^2 Re Z_a(omega_i) / (2 m_i)`.
///
/// Structurally it is batch 2 with the load pressure generalised from `p = R U` to `p = R_eff (U -
/// L^-)` (see [`RationalAirLoad`]), so the same rank-1 scalar Sherman-Morrison solve carries it:
///
/// ```text
/// q~^{n+1} = ModalBody::step(force)                    (force-free advance)
/// U_free   = a^T (q~^{n+1} - q^{n-1}) / (2 k)         (its centered volume velocity)
/// p, U     = load.solve(U_free, G) ,   G = (k/2) sum_i a_i^2 / (m_i (1 + sigma_i k))
/// q^{n+1}  = q~^{n+1} - p * [k^2 a_i / (m_i (1 + sigma_i k))]
/// ```
///
/// and the energy identity gains a **stored** term alongside the dissipated one:
///
/// ```text
/// E_body  +  1/2 M_a (U_L^{n+1/2})^2  +  integral R U_R^2 dt  =  const     (lossless modes).
/// ```
///
/// Passivity is unconditional (`1 + R_eff G >= 1` for every `R >= 0`, `M_a > 0`, `k`), so there is
/// still no CFL and no guard. Two exact reductions pin it: `R = 0` is bit-identical to a bare
/// [`crate::body::ModalBody`], and `M_a = inf` is bit-identical to [`RadiatedBody`] at the same
/// `R`. Like batch 2 it delegates every read accessor, so it drops straight into a
/// [`crate::connection::StringBodyBridge`] as the body with no edit to the bridge.
#[derive(Debug, Clone)]
pub struct ReactiveRadiatedBody {
    body: body::ModalBody,
    load: RationalAirLoad,
    g: f64,
    corr: Vec<f64>,
    n: usize,
}

impl ReactiveRadiatedBody {
    /// Couple `body` to `load`. `Err` carries the Python message.
    pub fn new(
        body: body::ModalBody,
        load: RationalAirLoad,
    ) -> Result<ReactiveRadiatedBody, String> {
        let bk = body.params().k;
        if !isclose(load.params().k, bk, 1e-12) {
            return Err(timestep_mismatch(load.params().fs, bk));
        }
        let (g, corr) = rank_one(body.params());
        Ok(ReactiveRadiatedBody {
            body,
            load,
            g,
            corr,
            n: 0,
        })
    }

    /// The body being loaded.
    pub fn body(&self) -> &body::ModalBody {
        &self.body
    }

    /// The body being loaded, mutably.
    pub fn body_mut(&mut self) -> &mut body::ModalBody {
        &mut self.body
    }

    /// The air impedance.
    pub fn load(&self) -> &RationalAirLoad {
        &self.load
    }

    /// The scalar driving-point factor `G`.
    pub fn g(&self) -> f64 {
        self.g
    }

    /// Steps taken.
    pub fn n(&self) -> usize {
        self.n
    }

    /// Advance one step: force-free body advance, scalar load solve, rank-1 correction.
    pub fn step(&mut self, force: f64) {
        let q_nm1 = self.body.q_prev().to_vec();
        self.body.step(force);
        let n_modes = self.body.params().n_modes();
        let k = self.body.params().k;
        let u_free = free_volume_velocity(self.body.q(), &q_nm1, self.body.params());
        let (p_load, u) = self.load.solve(u_free, self.g);
        let mut next = vec![0.0; n_modes];
        correct_into(self.body.q(), p_load, &self.corr, &mut next);
        self.body.q_mut().copy_from_slice(&next);
        let mut accel = vec![0.0; n_modes];
        refresh_accel(self.body.q(), self.body.q_prev(), &q_nm1, k, &mut accel);
        self.body.accel_mut().copy_from_slice(&accel);
        self.load.commit(p_load, u);
        self.n += 1;
    }

    /// `E_body + E_air`, the air term being stored plus radiated.
    pub fn energy(&self) -> f64 {
        self.body.energy() + self.load.energy()
    }

    /// Monopole read-out carrying the load.
    pub fn pressure(&self) -> f64 {
        self.body.pressure()
    }

    /// Set the body's modal state and reset the air.
    pub fn set_state(&mut self, q0: &[f64], v0: &[f64]) {
        self.body.set_state(q0, v0);
        self.load.reset();
        self.n = 0;
    }

    /// Zero the body state and the air.
    pub fn reset(&mut self) {
        let zeros = vec![0.0; self.body.params().n_modes()];
        self.set_state(&zeros, &zeros);
    }
}
