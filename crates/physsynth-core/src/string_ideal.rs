//! Ideal (lossless or linearly damped) transverse string — explicit FDTD scheme.
//!
//! Port of `physsynth/core/string_ideal.py`, HANDOFF §4.2–§4.3:
//!
//! ```text
//! u_tt = c^2 u_xx - 2 sigma u_t,      c = sqrt(T / rho)
//! ```
//!
//! with the explicit second-order scheme
//!
//! ```text
//! u[l, n+1] = 2 u[l,n] - u[l,n-1] + lambda^2 (u[l+1,n] - 2 u[l,n] + u[l-1,n]),
//! lambda = c k / h   (Courant number; lambda <= 1 required for stability).
//! ```
//!
//! The defining feature is [`energy`], which uses the **cross-time** potential term
//!
//! ```text
//! E^n = rho [ 1/2 ||delta_t- u^n||_w^2  +  (c^2/2) <delta_x+ u^n, delta_x+ u^{n-1}> ]
//! ```
//!
//! (the strain energy is a product of the gradient at steps n and n-1). This two-time-level form
//! is what makes the discrete energy conserved to machine precision for a lossless run; the
//! intuitive same-time form `||delta_x+ u^n||^2` drifts at ~1e-3. Do not "simplify" it.
//!
//! # Why the arithmetic is written out longhand
//!
//! Every expression below reproduces the *operation order* of the NumPy original, not merely its
//! algebra. Floating-point addition is not associative, so `(a - b) + c` and `a - (b - c)` are
//! different functions; writing them in NumPy's order is what makes the two implementations agree
//! bit-for-bit on the elementwise kernels ([`step_into`], [`second_diff`], [`initial_previous`])
//! rather than merely to a tolerance. The reductions inside [`energy`] cannot match bit-for-bit —
//! `np.dot` goes through BLAS, which accumulates in an order no portable loop reproduces — and
//! are held to the plan's ~1e-13 Group A agreement target instead. Where an expression looks
//! gratuitously parenthesised, that is why.

use crate::ops;

/// Which condition holds at one end of the string.
///
/// Both conserve energy in the lossless case: `Fixed` because the boundary velocity is zero,
/// `Free` because the reflected stencil is the summation-by-parts-consistent one for `u_x = 0`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Boundary {
    /// Dirichlet, `u = 0` at the end.
    Fixed,
    /// Neumann, `u_x = 0`, via a reflected stencil.
    Free,
}

impl Boundary {
    /// Parse the Python spelling. `None` for anything else — the caller owns the error message,
    /// because it is the one holding the object the user actually passed.
    pub fn parse(name: &str) -> Option<Boundary> {
        match name {
            "fixed" => Some(Boundary::Fixed),
            "free" => Some(Boundary::Free),
            _ => None,
        }
    }
}

/// A construction-time rejection. Every variant's `Display` is the Python original's message
/// verbatim, because `tests/test_stability.py` matches on the text.
#[derive(Debug, Clone, PartialEq)]
pub enum ParamError {
    /// One of `L`, `T`, `rho`, `fs` was not positive.
    NonPositive,
    /// Fewer than two spatial segments — no interior node to step.
    TooFewSegments,
    /// Negative loss coefficient.
    NegativeSigma,
    /// The boundary spec did not name a [`Boundary`]. The caller formats the message.
    BadBoundary,
    /// `lambda = c k / h > 1`: the explicit scheme would be unstable. Carries the offending value.
    CflViolated(f64),
}

impl std::fmt::Display for ParamError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParamError::NonPositive => write!(f, "L, T, rho, fs must all be positive."),
            ParamError::TooFewSegments => {
                write!(f, "N must be >= 2 (need at least one interior node).")
            }
            ParamError::NegativeSigma => write!(f, "sigma (loss) must be >= 0."),
            ParamError::BadBoundary => write!(f, "each boundary end must be 'fixed' or 'free'."),
            ParamError::CflViolated(lam) => write!(
                f,
                "CFL violated: lambda = c*k/h = {lam:.6} > 1. \
                 Reduce fs, refine the grid (increase N), or lower the wave speed."
            ),
        }
    }
}

impl std::error::Error for ParamError {}

/// Courant numbers above 1 are forbidden for the explicit scheme; allow a hair of floating-point
/// slack so that a requested `lambda == 1` is not spuriously rejected.
const LAMBDA_TOL: f64 = 1e-12;

/// The validated, immutable parameter set: everything about a string that is not its state.
///
/// Construction is the only place a string can be rejected, which is the project's rule for
/// explicit schemes — an unstable `lambda` is an error, never a run that quietly overflows.
#[derive(Debug, Clone, PartialEq)]
pub struct Params {
    /// Length (m).
    pub l: f64,
    /// Tension (N).
    pub t: f64,
    /// Linear density (kg/m).
    pub rho: f64,
    /// Sample rate (Hz).
    pub fs: f64,
    /// Number of spatial segments; the grid has `n + 1` nodes.
    pub n: usize,
    /// Loss coefficient for the `-2 sigma u_t` term.
    pub sigma: f64,
    /// Condition at `x = 0`.
    pub bc_left: Boundary,
    /// Condition at `x = L`.
    pub bc_right: Boundary,
    /// Wave speed `sqrt(T / rho)` (m/s).
    pub c: f64,
    /// Grid spacing `L / N` (m).
    pub h: f64,
    /// Timestep `1 / fs` (s).
    pub k: f64,
    /// Courant number `c k / h`.
    pub lam: f64,
}

impl Params {
    /// Validate and derive.
    ///
    /// `bc` is `None` when the caller could not make sense of the boundary spec it was handed;
    /// passing it in that shape (rather than rejecting earlier) is what keeps the *order* of the
    /// checks identical to Python's, so a call with two different faults reports the same one.
    ///
    /// `n` is taken as `i64` so that `N = 1` and `N = -3` are both rejected by the documented
    /// "N must be >= 2" path rather than by a cast.
    pub fn new(
        l: f64,
        t: f64,
        rho: f64,
        fs: f64,
        n: i64,
        sigma: f64,
        bc: Option<(Boundary, Boundary)>,
    ) -> Result<Params, ParamError> {
        if l <= 0.0 || t <= 0.0 || rho <= 0.0 || fs <= 0.0 {
            return Err(ParamError::NonPositive);
        }
        if n < 2 {
            return Err(ParamError::TooFewSegments);
        }
        if sigma < 0.0 {
            return Err(ParamError::NegativeSigma);
        }
        let (bc_left, bc_right) = bc.ok_or(ParamError::BadBoundary)?;

        let n = n as usize;
        let c = (t / rho).sqrt();
        let h = l / (n as f64);
        let k = 1.0 / fs;
        let lam = c * k / h;

        if lam > 1.0 + LAMBDA_TOL {
            return Err(ParamError::CflViolated(lam));
        }

        Ok(Params {
            l,
            t,
            rho,
            fs,
            n,
            sigma,
            bc_left,
            bc_right,
            c,
            h,
            k,
            lam,
        })
    }

    /// Number of grid nodes, `N + 1`.
    pub fn nodes(&self) -> usize {
        self.n + 1
    }

    /// Node positions — `np.linspace(0.0, L, N + 1)`, reproduced exactly.
    ///
    /// NumPy computes `i * step` with `step = L / N` and then *overwrites* the last entry with the
    /// endpoint, so `x[N]` is `L` exactly rather than `N * (L / N)`. The two differ in the last
    /// bit for most lengths, and `x` reaches the analysis layer, so the overwrite is reproduced.
    pub fn grid(&self) -> Vec<f64> {
        let step = self.l / (self.n as f64);
        let mut x: Vec<f64> = (0..self.nodes()).map(|i| (i as f64) * step).collect();
        x[self.n] = self.l;
        x
    }

    /// Trapezoidal node weights for the (kinetic) inner product: `h/2` at the two boundary nodes,
    /// `h` in the interior.
    ///
    /// This is the summation-by-parts-consistent weighting that keeps the free-boundary energy
    /// exact; for fixed ends the boundary velocity is zero so it is equivalent to uniform
    /// weighting. It is *not* equivalent when an end is free, which is the whole reason it exists.
    pub fn node_weights(&self) -> Vec<f64> {
        let mut w = vec![self.h; self.nodes()];
        w[0] = 0.5 * self.h;
        w[self.n] = 0.5 * self.h;
        w
    }
}

// -- kernels ---------------------------------------------------------------------------------
//
// Free functions over slices. They hold no state and allocate only what they return, so the
// caller decides where the buffers live — the native `IdealString` below puts them in `Vec`s,
// while the Python binding has to put them in NumPy arrays. One copy of the physics, two owners.

/// `u[l+1] - 2u[l] + u[l-1]` over the whole grid, with the boundary stencil.
///
/// Fixed ends contribute 0 at the boundary nodes (they are held clamped); free ends use the
/// reflected (Neumann) stencil `2(u[1]-u[0])` and `2(u[N-1]-u[N])`.
///
/// # Panics
/// If `u` has fewer than three nodes (i.e. `N < 2`), which [`Params::new`] already rejects.
pub fn second_diff(u: &[f64], bc_left: Boundary, bc_right: Boundary) -> Vec<f64> {
    let n = u.len();
    assert!(n >= 3, "second_diff needs at least 3 nodes (N >= 2)");
    let mut s = vec![0.0; n];
    for l in 1..n - 1 {
        // NumPy evaluates `u[2:] - 2.0 * u[1:-1] + u[:-2]` left to right.
        s[l] = (u[l + 1] - 2.0 * u[l]) + u[l - 1];
    }
    s[0] = match bc_left {
        Boundary::Free => 2.0 * (u[1] - u[0]),
        Boundary::Fixed => 0.0,
    };
    s[n - 1] = match bc_right {
        Boundary::Free => 2.0 * (u[n - 2] - u[n - 1]),
        Boundary::Fixed => 0.0,
    };
    s
}

/// Clamp the fixed ends of `u` in place. Free ends are unknowns and are left alone.
pub fn apply_boundary(u: &mut [f64], bc_left: Boundary, bc_right: Boundary) {
    let n = u.len();
    if bc_left == Boundary::Fixed {
        u[0] = 0.0;
    }
    if bc_right == Boundary::Fixed {
        u[n - 1] = 0.0;
    }
}

/// One timestep: write `u^{n+1}` into `out` from `u^n` and `u^{n-1}`. Boundaries applied.
///
/// # Panics
/// If the three slices do not all have `p.nodes()` elements.
pub fn step_into(u: &[f64], u_prev: &[f64], out: &mut [f64], p: &Params) {
    let n = p.nodes();
    assert_eq!(u.len(), n, "u must have N+1 elements");
    assert_eq!(u_prev.len(), n, "u_prev must have N+1 elements");
    assert_eq!(out.len(), n, "out must have N+1 elements");

    let sk = p.sigma * p.k;
    let lam2 = p.lam * p.lam;
    let one_minus_sk = 1.0 - sk;
    let one_plus_sk = 1.0 + sk;
    let s = second_diff(u, p.bc_left, p.bc_right);

    for l in 0..n {
        let stencil = lam2 * s[l];
        out[l] = ((2.0 * u[l] - one_minus_sk * u_prev[l]) + stencil) / one_plus_sk;
    }
    apply_boundary(out, p.bc_left, p.bc_right);
}

/// The consistent second-order start: `u^{-1} = u^0 - k v^0 + 1/2 * stencil(u^0)`.
///
/// `u0` must already have had [`apply_boundary`] run on it — the Python original clamps the
/// initial displacement *before* taking its stencil, and that stencil is what this returns half
/// of. The result has the boundary applied on the way out as well.
///
/// This start is what makes a single-mode initial condition oscillate as a clean discrete cosine
/// (no spurious first-step transient) and zero initial velocity exact.
///
/// # Panics
/// If `u0` or `v0` do not have `p.nodes()` elements.
pub fn initial_previous(u0: &[f64], v0: &[f64], p: &Params) -> Vec<f64> {
    let n = p.nodes();
    assert_eq!(u0.len(), n, "u0 must have N+1 elements");
    assert_eq!(v0.len(), n, "v0 must have N+1 elements");

    let lam2 = p.lam * p.lam;
    let s = second_diff(u0, p.bc_left, p.bc_right);
    let mut prev: Vec<f64> = (0..n)
        .map(|l| (u0[l] - p.k * v0[l]) + 0.5 * (lam2 * s[l]))
        .collect();
    apply_boundary(&mut prev, p.bc_left, p.bc_right);
    prev
}

/// Discrete energy `E^n` (Joules) using the cross-time potential term.
///
/// For a lossless run this is conserved to machine precision; for `sigma > 0` it decreases
/// monotonically (passivity). `w` comes from [`Params::node_weights`].
///
/// # Panics
/// If the slices do not all have `p.nodes()` elements.
pub fn energy(u: &[f64], u_prev: &[f64], w: &[f64], p: &Params) -> f64 {
    let n = p.nodes();
    assert_eq!(u.len(), n, "u must have N+1 elements");
    assert_eq!(u_prev.len(), n, "u_prev must have N+1 elements");
    assert_eq!(w.len(), n, "w must have N+1 elements");

    // `0.5 * np.dot(w, dt_u * dt_u)` — the square is elementwise and happens before the reduction.
    let mut acc = 0.0;
    for l in 0..n {
        let dt_u = (u[l] - u_prev[l]) / p.k;
        acc += w[l] * (dt_u * dt_u);
    }
    let kinetic = 0.5 * acc;

    let gx_now = ops::delta_x_forward(u, p.h);
    let gx_prev = ops::delta_x_forward(u_prev, p.h);
    let potential = 0.5 * p.c * p.c * ops::inner(&gx_now, &gx_prev, p.h);

    p.rho * (kinetic + potential)
}

// -- the native owning struct ----------------------------------------------------------------

/// A discretized ideal string resonator, owning its own state.
///
/// This is the Rust-side caller's view and what `cargo test` exercises. The Python binding does
/// **not** wrap it: its buffers have to be Python objects so that a reference held across a step
/// stays valid, so it owns NumPy arrays and calls the kernels above directly. Both go through the
/// same kernels, so there is one copy of the physics and two copies of the buffer bookkeeping.
#[derive(Debug, Clone)]
pub struct IdealString {
    params: Params,
    w: Vec<f64>,
    /// Current displacement field `u^n`.
    pub u: Vec<f64>,
    /// Previous displacement field `u^{n-1}`.
    pub u_prev: Vec<f64>,
    /// Number of completed steps.
    pub n_steps: usize,
}

impl IdealString {
    /// Build from validated parameters. Both state buffers start at rest.
    pub fn new(params: Params) -> IdealString {
        let nodes = params.nodes();
        let w = params.node_weights();
        IdealString {
            params,
            w,
            u: vec![0.0; nodes],
            u_prev: vec![0.0; nodes],
            n_steps: 0,
        }
    }

    /// The parameter set this string was built from.
    pub fn params(&self) -> &Params {
        &self.params
    }

    /// Set the initial displacement and velocity, resetting the step count.
    ///
    /// # Panics
    /// If `u0` or `v0` do not have `N + 1` elements.
    pub fn set_state(&mut self, u0: &[f64], v0: &[f64]) {
        let p = &self.params;
        assert_eq!(u0.len(), p.nodes(), "u0 must have N+1 elements");
        let mut u = u0.to_vec();
        apply_boundary(&mut u, p.bc_left, p.bc_right);
        self.u_prev = initial_previous(&u, v0, p);
        self.u = u;
        self.n_steps = 0;
    }

    /// Set the initial displacement with zero initial velocity.
    ///
    /// # Panics
    /// If `u0` does not have `N + 1` elements.
    pub fn set_displacement(&mut self, u0: &[f64]) {
        let v0 = vec![0.0; self.params.nodes()];
        self.set_state(u0, &v0);
    }

    /// Advance one timestep, rolling the history.
    pub fn step(&mut self) {
        let mut next = vec![0.0; self.params.nodes()];
        step_into(&self.u, &self.u_prev, &mut next, &self.params);
        std::mem::swap(&mut self.u_prev, &mut self.u);
        self.u = next;
        self.n_steps += 1;
    }

    /// Discrete energy `E^n` (Joules) — the primary bug detector.
    pub fn energy(&self) -> f64 {
        energy(&self.u, &self.u_prev, &self.w, &self.params)
    }

    /// Displacement at grid node `index` — a pickup for spectral analysis.
    ///
    /// # Panics
    /// If `index` is past the last node.
    pub fn displacement_at(&self, index: usize) -> f64 {
        self.u[index]
    }
}
