//! Stiff transverse string — implicit theta-scheme FDTD, simply supported (model #2).
//!
//! Port of `physsynth/core/string_stiff.py`, HANDOFF §5:
//!
//! ```text
//! u_tt = c^2 u_xx - kappa^2 u_xxxx - 2 sigma u_t,   c^2 = T/rho,  kappa^2 = E I / rho
//! ```
//!
//! The whole spatial operator `L = c^2 delta_xx - kappa^2 delta_xxxx` is time-averaged with a theta
//! weight, so each step is one back-substitution against a constant pentadiagonal SPD matrix
//!
//! ```text
//! A = (1 + sigma k) I - theta k^2 L
//! ```
//!
//! factored once at construction by [`crate::banded`]. There is no CFL limit: `theta >= 1/4` is
//! unconditionally stable, so `lam` is reported and never rejected. The Python module's docstring
//! is the reference for the physics, the energy form and why the biharmonic block is built as
//! `(delta_xx)^2`.
//!
//! # Why this is a separate module from [`crate::string_damped`], which is nearly the same code
//!
//! Model #3 is model #2 plus one loss term, and its Python file is very nearly a copy of this one.
//! Backing both Python classes with a single superset here would be less code and would **make a
//! test stop testing**: `tests/test_damped_string.py` anchors the two together by asserting that a
//! damped string with `sigma1 = 0` is `array_equal` to a stiff one, energy trace included, over
//! 1,500 steps. That anchor is a real comparison of two independent transcriptions in Python; if
//! both sides resolved to one implementation here it would be vacuously true under the flag, and a
//! guard that silently covers nothing is a failure mode this migration has already met once
//! (§17.6). So the duplication is deliberate, and the anchor stays a detector.
//!
//! # Where the two evaluation-order hazards live
//!
//! Both are §18's, and both are answered on the Python side rather than here:
//!
//! - **The matvec.** `L` is built by subtracting SciPy's `biharmonic_matrix`, which comes back with
//!   *descending* column indices; a CSR matvec sums each row in stored order, so the two spellings
//!   differ in essentially every vector. [`crate::sparse::Csr`] is canonical, and
//!   `physsynth/core/portable.py` sorts the Python side to meet it.
//! - **The reduction.** [`energy`] sums left to right, which `np.dot` does not. The same module
//!   supplies the Python spelling that does.
//!
//! Every expression below otherwise reproduces the NumPy original's *operation order*, not merely
//! its algebra — see [`crate::string_ideal`]'s header for why the parenthesisation is not
//! gratuitous. The one thing that is not an addition order is `** 2`, which is
//! [`crate::pyfloat::scalar_pow`] and not a multiply.

use crate::banded::{self, BandedError};
use crate::fmt::py_float;
use crate::ops;
use crate::pyfloat::scalar_pow;
use crate::sparse::Csr;

/// Time-averaging weight: a hair above the minimal-dispersion `1/4`, so the energy keeps a small
/// positivity margin. `physsynth.core.string_stiff.THETA_DEFAULT`.
pub const THETA_DEFAULT: f64 = 0.28;

/// A construction-time rejection. Every variant's `Display` is the Python original's message
/// verbatim, because `tests/test_stability.py` matches on the text.
#[derive(Debug, Clone, PartialEq)]
pub enum ParamError {
    /// One of `L`, `T`, `rho`, `fs` was not positive.
    NonPositive,
    /// Fewer than two spatial segments — no interior node to solve for.
    TooFewSegments,
    /// Negative stiffness coefficient.
    NegativeKappa,
    /// Negative loss coefficient.
    NegativeSigma,
    /// `theta` outside `(0, 1]`. Carries the offending value, which the message quotes.
    BadTheta(f64),
    /// The boundary spec was not `"supported"`. The caller formats the message, because it quotes
    /// the object the user passed and only the caller can `repr()` it.
    BadBoundary,
    /// `A` was not positive definite — cannot happen for `theta > 0` and the checks above, but the
    /// factor reports it rather than the constructor assuming.
    NotFactorable(BandedError),
}

impl std::fmt::Display for ParamError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParamError::NonPositive => write!(f, "L, T, rho, fs must all be positive."),
            ParamError::TooFewSegments => {
                write!(f, "N must be >= 2 (need at least one interior node).")
            }
            ParamError::NegativeKappa => write!(f, "kappa (stiffness) must be >= 0."),
            ParamError::NegativeSigma => write!(f, "sigma (loss) must be >= 0."),
            ParamError::BadTheta(t) => {
                write!(f, "theta must be in (0, 1], got {}.", py_float(*t))
            }
            ParamError::BadBoundary => write!(f, "boundary must be 'supported'."),
            ParamError::NotFactorable(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for ParamError {}

/// The validated parameter set plus the two things that are constant in time: the interior
/// operator `L` and the Cholesky factor of `A`.
///
/// Both are built **once**. Rebuilding `L` per access would pass every physics bar and make the
/// flagged run slower than the Python one — the finding `membrane` recorded in §11.4.
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
    /// Number of spatial segments; the grid has `n + 1` nodes and `n - 1` unknowns.
    pub n: usize,
    /// Stiffness coefficient `sqrt(E I / rho)` (m^2/s).
    pub kappa: f64,
    /// Frequency-independent loss coefficient.
    pub sigma: f64,
    /// Time-averaging weight in `(0, 1]`.
    pub theta: f64,
    /// Wave speed `sqrt(T / rho)` (m/s).
    pub c: f64,
    /// Grid spacing `L / N` (m).
    pub h: f64,
    /// Timestep `1 / fs` (s).
    pub k: f64,
    /// Courant number `c k / h` — reported only; the scheme is unconditionally stable.
    pub lam: f64,
    /// Inharmonicity `pi^2 kappa^2 / (c^2 L^2)`.
    pub b: f64,
    /// Interior spatial operator, `(n - 1) x (n - 1)`, canonical CSR.
    pub op_l: Csr,
    /// Upper-banded Cholesky factor of `A`, two superdiagonals, row-major `3 x (n - 1)`.
    pub chol: Vec<f64>,
}

impl Params {
    /// Validate, derive, assemble `L`, and factor `A`.
    ///
    /// `boundary_ok` is `false` when the caller could not make sense of the boundary spec it was
    /// handed; passing it in that shape keeps the *order* of the checks identical to Python's, so
    /// a call that is wrong in two ways reports the same fault.
    ///
    /// `n` is taken as `i64` so that `N = 1` and `N = -3` are both rejected by the documented
    /// "N must be >= 2" path rather than by a cast.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        l: f64,
        t: f64,
        rho: f64,
        fs: f64,
        n: i64,
        kappa: f64,
        sigma: f64,
        theta: f64,
        boundary_ok: bool,
    ) -> Result<Params, ParamError> {
        if l <= 0.0 || t <= 0.0 || rho <= 0.0 || fs <= 0.0 {
            return Err(ParamError::NonPositive);
        }
        if n < 2 {
            return Err(ParamError::TooFewSegments);
        }
        if kappa < 0.0 {
            return Err(ParamError::NegativeKappa);
        }
        if sigma < 0.0 {
            return Err(ParamError::NegativeSigma);
        }
        if !(theta > 0.0 && theta <= 1.0) {
            return Err(ParamError::BadTheta(theta));
        }
        if !boundary_ok {
            return Err(ParamError::BadBoundary);
        }

        let n = n as usize;
        let c = (t / rho).sqrt();
        let h = l / (n as f64);
        let k = 1.0 / fs;
        let lam = c * k / h;
        // `(np.pi ** 2) * kappa ** 2 / (c ** 2 * L ** 2)`, with Python's precedence: `**` binds
        // tighter than `*` and `/`, and every one of the four is `float.__pow__`, i.e. `pow`.
        let b = (scalar_pow(std::f64::consts::PI, 2.0) * scalar_pow(kappa, 2.0))
            / (scalar_pow(c, 2.0) * scalar_pow(l, 2.0));

        let op_l = interior_operator(n, h, c, kappa);
        let ab = update_matrix_bands(&op_l, sigma * k, theta * scalar_pow(k, 2.0));
        let chol =
            banded::cholesky_banded_upper(ab, 2, n - 1).map_err(ParamError::NotFactorable)?;

        Ok(Params {
            l,
            t,
            rho,
            fs,
            n,
            kappa,
            sigma,
            theta,
            c,
            h,
            k,
            lam,
            b,
            op_l,
            chol,
        })
    }

    /// Number of grid nodes, `N + 1`.
    pub fn nodes(&self) -> usize {
        self.n + 1
    }

    /// Number of interior unknowns, `N - 1`.
    pub fn interior(&self) -> usize {
        self.n - 1
    }

    /// Node positions — `np.linspace(0.0, L, N + 1)`, reproduced exactly (the endpoint is
    /// *overwritten* rather than computed, which differs in the last bit for most lengths).
    pub fn grid(&self) -> Vec<f64> {
        let step = self.l / (self.n as f64);
        let mut x: Vec<f64> = (0..self.nodes()).map(|i| (i as f64) * step).collect();
        x[self.n] = self.l;
        x
    }
}

// -- construction ------------------------------------------------------------------------------

/// `L = c^2 delta_xx - kappa^2 delta_xxxx` on the interior, canonical CSR.
///
/// `kappa == 0.0` skips the biharmonic block entirely, exactly as the Python guard does — which is
/// what makes `kappa = 0` a *structurally* different matrix rather than one with zero entries.
pub fn interior_operator(n: usize, h: f64, c: f64, kappa: f64) -> Csr {
    let l = ops::second_difference_matrix(n, h).scaled(scalar_pow(c, 2.0));
    if kappa != 0.0 {
        return l.sub(&ops::biharmonic_matrix(n, h).scaled(scalar_pow(kappa, 2.0)));
    }
    l
}

/// The three upper bands of `A = (1 + sigma k) I - theta_k2 L`, as `cholesky_banded` wants them:
/// row-major `3 x (n_int)` with `ab[2]` the diagonal, `ab[1]` the first superdiagonal from index 1,
/// and `ab[0]` the second from index 2.
///
/// Only the diagonals of `A` are ever read, so `A` is never assembled. That is not a shortcut past
/// SciPy's behaviour: `csr.diagonal(d)` picks stored entries by column and is independent of the
/// order they are stored in, which is why sorting `L` (§18) cannot move `ab` — measured over 288
/// parameter combinations, zero of the three bands changed.
pub fn update_matrix_bands(op_l: &Csr, sigma_k: f64, theta_k2: f64) -> Vec<f64> {
    let n_int = op_l.nrows();
    let mut ab = vec![0.0; 3 * n_int];
    for i in 0..n_int {
        ab[2 * n_int + i] = (1.0 + sigma_k) - theta_k2 * op_l.get(i, i);
        if i >= 1 {
            ab[n_int + i] = 0.0 - theta_k2 * op_l.get(i - 1, i);
        }
        if i >= 2 {
            ab[i] = 0.0 - theta_k2 * op_l.get(i - 2, i);
        }
    }
    ab
}

// -- kernels -----------------------------------------------------------------------------------
//
// Free functions over slices, so the native struct below and the Python binding can put the state
// wherever each of them must.

/// `L u` on the full grid, zeros at the two clamped nodes — `_apply_L`.
pub fn apply_l_full(u_full: &[f64], p: &Params) -> Vec<f64> {
    let mut out = vec![0.0; u_full.len()];
    let interior = p.op_l.matvec(&u_full[1..u_full.len() - 1]);
    out[1..u_full.len() - 1].copy_from_slice(&interior);
    out
}

/// The consistent second-order start `u^{-1} = u^0 - k v^0 + 1/2 k^2 L u^0`, ends clamped.
///
/// `u0` is clamped **before** `L u0` is formed, as Python does it.
pub fn initial_previous(u0: &mut [f64], v0: &[f64], p: &Params) -> Vec<f64> {
    let last = u0.len() - 1;
    u0[0] = 0.0;
    u0[last] = 0.0;
    let lu0 = apply_l_full(u0, p);
    let half_k2 = 0.5 * scalar_pow(p.k, 2.0);
    let mut prev: Vec<f64> = (0..u0.len())
        .map(|i| (u0[i] - p.k * v0[i]) + half_k2 * lu0[i])
        .collect();
    prev[0] = 0.0;
    prev[last] = 0.0;
    prev
}

/// One timestep's right-hand side on the interior.
///
/// ```text
/// rhs = 2 u + (1 - 2 theta) k^2 (L u) - u_prev + theta k^2 (L u_prev) + sigma k u_prev
/// ```
///
/// evaluated left to right, which is the order NumPy's chained `+`/`-` produces.
pub fn step_rhs(u: &[f64], u_prev: &[f64], p: &Params) -> Vec<f64> {
    let sk = p.sigma * p.k;
    let k2 = scalar_pow(p.k, 2.0);
    let lu = p.op_l.matvec(&u[1..u.len() - 1]);
    let lu_prev = p.op_l.matvec(&u_prev[1..u_prev.len() - 1]);
    let a = (1.0 - 2.0 * p.theta) * k2;
    let b = p.theta * k2;
    (0..p.interior())
        .map(|i| {
            let un = u[i + 1];
            let up = u_prev[i + 1];
            (((2.0 * un + a * lu[i]) - up) + b * lu_prev[i]) + sk * up
        })
        .collect()
}

/// Advance one step into `next` (full grid, ends left at zero).
pub fn step_into(u: &[f64], u_prev: &[f64], next: &mut [f64], p: &Params) {
    let rhs = step_rhs(u, u_prev, p);
    let sol = banded::cho_solve_banded_upper(&p.chol, 2, p.interior(), &rhs)
        .expect("the factor and the right-hand side are shaped by construction");
    next.fill(0.0);
    next[1..u.len() - 1].copy_from_slice(&sol);
}

/// `P(f, g) = <-L f, g> = -h (L f) . g` on interior vectors — the potential bilinear form.
///
/// The reduction is left to right; see the module header.
pub fn potential_form(f: &[f64], g: &[f64], p: &Params) -> f64 {
    -p.h * dot(&p.op_l.matvec(f), g)
}

/// A left-to-right inner product — `physsynth.core.portable.dot`.
///
/// # Panics
/// If the two slices differ in length.
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "dot length mismatch");
    let mut acc = 0.0;
    for i in 0..a.len() {
        acc += a[i] * b[i];
    }
    acc
}

/// Discrete energy `E^n` (Joules) for the implicit theta-scheme.
///
/// ```text
/// E^n = rho [ 1/2 ||delta_t- u||^2 + (theta/2)(P_nn + P_pp) + (1/2 - theta) P_np ]
/// ```
///
/// Conserved to machine precision in a lossless run; monotone decreasing when lossy.
pub fn energy(u: &[f64], u_prev: &[f64], p: &Params) -> f64 {
    let n = u.len();
    let un = &u[1..n - 1];
    let up = &u_prev[1..n - 1];
    let dt_u: Vec<f64> = (0..un.len()).map(|i| (un[i] - up[i]) / p.k).collect();
    let kinetic = 0.5 * p.h * dot(&dt_u, &dt_u);

    let p_nn = potential_form(un, un, p);
    let p_pp = potential_form(up, up, p);
    let p_np = potential_form(un, up, p);
    let potential = 0.5 * p.theta * (p_nn + p_pp) + (0.5 - p.theta) * p_np;
    p.rho * (kinetic + potential)
}

// -- the native owning struct ------------------------------------------------------------------

/// A stiff string with its own buffers — for Rust callers and for `cargo test`.
#[derive(Debug, Clone)]
pub struct StiffString {
    /// Parameters, operator and factor.
    pub p: Params,
    /// Current displacement `u^n` on the full grid.
    pub u: Vec<f64>,
    /// Previous displacement `u^{n-1}` on the full grid.
    pub u_prev: Vec<f64>,
    /// Completed steps.
    pub n: usize,
}

impl StiffString {
    /// A string at rest.
    pub fn new(p: Params) -> Self {
        let nodes = p.nodes();
        StiffString {
            p,
            u: vec![0.0; nodes],
            u_prev: vec![0.0; nodes],
            n: 0,
        }
    }

    /// Set the initial displacement and velocity (both full-grid).
    pub fn set_state(&mut self, u0: &[f64], v0: &[f64]) {
        let mut u = u0.to_vec();
        self.u_prev = initial_previous(&mut u, v0, &self.p);
        self.u = u;
        self.n = 0;
    }

    /// Advance one timestep, rolling the history.
    pub fn step(&mut self) {
        let mut next = vec![0.0; self.p.nodes()];
        step_into(&self.u, &self.u_prev, &mut next, &self.p);
        std::mem::swap(&mut self.u_prev, &mut self.u);
        self.u = next;
        self.n += 1;
    }

    /// Discrete energy `E^n` (Joules).
    pub fn energy(&self) -> f64 {
        energy(&self.u, &self.u_prev, &self.p)
    }
}
