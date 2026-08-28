//! Free-free Euler–Bernoulli beam — implicit theta-scheme FDTD (model #5b-pre).
//!
//! Port of `physsynth/core/beam.py`, HANDOFF §5 row 5. The Python module's docstring is the
//! reference for the physics: the energy-first operator pair `(K, W)` from
//! [`crate::ops::free_beam_stiffness`], the free-end closure that comes from the `h/2` mass cells
//! rather than from hand-written stencil rows, the theta time-average that removes the `kappa² k² /
//! h⁴` CFL, and the energy form the update and [`energy`] share exactly.
//!
//! ```text
//! W delta_tt u = -kappa^2 K (theta u^{n+1} + (1 - 2 theta) u^n + theta u^{n-1}) - 2 sigma W delta_t. u
//! A = (1 + sigma k) W + theta k^2 kappa^2 K        (SPD, because W is)
//! ```
//!
//! # Why this file is Phase 4 and what it was sent to find out
//!
//! `beam` is the smallest member of Group D — the sparse-LU models — at 254 Python lines with one
//! `splu`, and plan §4.1 chose it as the place to test whether Rust could reproduce SciPy's sparse
//! LU **bit for bit** by linking SuperLU itself. It cannot, and the reason that decides it is not
//! one of the three §4.1 named; [`crate::sparse_lu`]'s header carries the measurement. What
//! follows from it for this module is one sentence: **the beam's agreement with the Python
//! original is a tolerance, not an equality, from the very first step**, and the tolerance is not
//! small by this project's standards.
//!
//! # The agreement regime, which is the sixth and the first set by a boundary condition
//!
//! `K`'s nullspace is exactly the rigid-body space `{1, x}` — that is what "free-free" means, and
//! §5b built the whole model to have it. A per-step difference between two solvers therefore has a
//! component that the scheme does not restore, because along `{1, x}` the beam is a **free
//! particle**: a velocity error is integrated once into a displacement error and then again by
//! every subsequent step. Measured against SciPy's `splu` over 20,000 steps at `N = 32`, splitting
//! the difference in the `W`-inner product:
//!
//! ```text
//! steps      total      rigid    elastic
//!     1    2.2e-16    3.7e-17    1.9e-16
//!   100    8.2e-14    8.8e-14    3.4e-14
//!  1000    6.8e-12    6.9e-12    6.0e-14
//! 20000    3.3e-09    3.3e-09    1.4e-12
//! ```
//!
//! The rigid part grows like `t²` and swamps everything from about a hundred steps on. The elastic
//! part does not: it goes up by a factor of 40 while the step count goes up by 200, which is the
//! random walk §18.6 measured on the theta strings and is what a *linear* scheme does with a
//! perturbation it can restore. So a parity bar on this model must be read against that split, or
//! against the energy — never against `max|du|/amp`, which will read as a failure that is not. The
//! previous five regimes were set by nonlinearity (§16.5), by its persistence (§17.5), by
//! linearity (§18.6), by amplitude (§19.5) and by an attractor (§20.5); this one is set by a
//! *boundary condition*, and every free-edge model in Phases 5 and 6 inherits it.
//!
//! # Two evaluation-order decisions, both taken by declining `portable.py`
//!
//! `portable.py`'s scope note names `beam` as out of scope "because no anchor binds it to
//! anything", and §19.2's rule says that has to be re-taken the moment the model ports. Re-taken,
//! and the answer is the same both times:
//!
//! - **The matvec order.** `K @ u` runs every timestep, and SciPy's sparse product can leave
//!   descending column indices (§18.2). It buys nothing here, because under the flag the Python
//!   beam gets its `K` from [`crate::ops::free_beam_stiffness`] too — the same canonical matrix
//!   this module builds — so the two sides already sum in the same order.
//! - **The energy reduction.** [`energy`] sums left to right and `np.dot` does not. Also nothing:
//!   the two trajectories differ at 1e-9 for the solver reason above, so matching a reduction's
//!   last bit downstream of that is §23.3's "nothing is *available* where a coarser divergence
//!   sits upstream", exactly.
//!
//! Every expression below still reproduces the NumPy original's *operation order* rather than
//! merely its algebra, for the reason [`crate::string_ideal`]'s header gives: a reassociation that
//! is invisible today is a divergence the day something tightens.

use crate::fmt::py_float;
use crate::ops::free_beam_stiffness;
use crate::sparse::Csr;
use crate::sparse_lu::{SparseLu, SparseLuError};

/// Time-averaging weight — a hair above the minimal `1/4`, the project-wide theta inherited from
/// the stiff string. `physsynth.core.beam.THETA_DEFAULT`.
pub const THETA_DEFAULT: f64 = 0.28;

/// A construction-time rejection. Every variant's `Display` is the Python original's message
/// verbatim, because `tests/test_beam_stability.py` matches on the text.
#[derive(Debug, Clone, PartialEq)]
pub enum ParamError {
    /// One of `L`, `rho`, `fs` was not positive.
    NonPositive,
    /// Non-positive stiffness coefficient. Unlike the stiff string, `kappa = 0` is *degenerate*
    /// here (`u_tt = 0`), not merely a string without stiffness, so it is refused rather than
    /// special-cased.
    NonPositiveKappa,
    /// Fewer than four spatial segments.
    TooFewSegments,
    /// Negative loss coefficient.
    NegativeSigma,
    /// `theta` outside `(0, 1]`. Carries the offending value, which the message quotes.
    BadTheta(f64),
    /// The boundary spec was not `"free"`. The caller formats the message, because it quotes the
    /// object the user passed and only the caller can `repr()` it.
    BadBoundary,
    /// `A` could not be factored. Cannot happen for the checks above — `A` is SPD — but the
    /// factorization reports it rather than the constructor assuming.
    NotFactorable(SparseLuError),
}

impl std::fmt::Display for ParamError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParamError::NonPositive => write!(f, "L, rho, fs must all be positive."),
            ParamError::NonPositiveKappa => write!(f, "kappa (stiffness) must be positive."),
            ParamError::TooFewSegments => write!(
                f,
                "N must be >= 4 (need a few interior nodes for the free-free modes)."
            ),
            ParamError::NegativeSigma => write!(f, "sigma (loss) must be >= 0."),
            ParamError::BadTheta(t) => {
                write!(f, "theta must be in (0, 1], got {}.", py_float(*t))
            }
            ParamError::BadBoundary => write!(f, "boundary must be 'free'."),
            ParamError::NotFactorable(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for ParamError {}

/// The validated parameter set plus the three things that are constant in time: the stiffness `K`,
/// the lumped mass `w`, and the factorization of `A`.
///
/// All three are built **once**. Rebuilding any of them per access would pass every physics bar
/// and make the flagged run slower than the Python one — §11.4's finding, and the factorization
/// is the most expensive instance of it the migration has met.
#[derive(Debug, Clone)]
pub struct Params {
    /// Length (m).
    pub l: f64,
    /// Linear density `rho A` (kg/m).
    pub rho: f64,
    /// Sample rate (Hz).
    pub fs: f64,
    /// Number of spatial segments; every one of the `n + 1` nodes is an unknown.
    pub n: usize,
    /// Stiffness coefficient `sqrt(E I / (rho A))` (m^2/s).
    pub kappa: f64,
    /// Frequency-independent loss coefficient.
    pub sigma: f64,
    /// Time-averaging weight in `(0, 1]`.
    pub theta: f64,
    /// Grid spacing `L / N` (m).
    pub h: f64,
    /// Timestep `1 / fs` (s).
    pub k: f64,
    /// Beam "Courant" number `kappa k / h^2` — reported only; the implicit scheme has no limit.
    pub mu: f64,
    /// Symmetric PSD bending stiffness, `(n + 1) x (n + 1)`, canonical CSR.
    pub stiffness: Csr,
    /// Diagonal trapezoidal mass, `(n + 1) x (n + 1)`, canonical CSR.
    pub mass: Csr,
    /// `W`'s diagonal — the lumped weights, `h` inside and `h/2` at the two free ends.
    pub w: Vec<f64>,
    /// Factorization of `A = (1 + sigma k) W + theta k^2 kappa^2 K`.
    pub lu: SparseLu,
}

impl Params {
    /// Validate, derive, assemble `(K, W)`, and factor `A`.
    ///
    /// `boundary_ok` is `false` when the caller could not make sense of the boundary spec it was
    /// handed; passing it in that shape keeps the *order* of the checks identical to Python's.
    ///
    /// `n` is taken as `i64` so that `N = 2` and `N = -3` are both rejected by the documented
    /// "N must be >= 4" path rather than by a cast.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        l: f64,
        rho: f64,
        fs: f64,
        n: i64,
        kappa: f64,
        sigma: f64,
        theta: f64,
        boundary_ok: bool,
    ) -> Result<Params, ParamError> {
        // `min(L, rho, fs) <= 0` in Python, i.e. one test over the three.
        if l.min(rho).min(fs) <= 0.0 {
            return Err(ParamError::NonPositive);
        }
        if kappa <= 0.0 {
            return Err(ParamError::NonPositiveKappa);
        }
        if n < 4 {
            return Err(ParamError::TooFewSegments);
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
        let h = l / (n as f64);
        let k = 1.0 / fs;
        let mu = kappa * k / (h * h);

        let (stiffness, mass) = free_beam_stiffness(n, h);
        let w: Vec<f64> = (0..=n).map(|i| mass.get(i, i)).collect();

        let a = update_matrix(&stiffness, &mass, sigma * k, theta * k * k * kappa * kappa);
        let lu = SparseLu::factor(&a).map_err(ParamError::NotFactorable)?;

        Ok(Params {
            l,
            rho,
            fs,
            n,
            kappa,
            sigma,
            theta,
            h,
            k,
            mu,
            stiffness,
            mass,
            w,
            lu,
        })
    }

    /// Number of grid nodes, `N + 1` — every one of them an unknown.
    pub fn nodes(&self) -> usize {
        self.n + 1
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

/// `A = (1 + sigma k) W + theta k^2 kappa^2 K`.
///
/// Spelled as a subtraction of the negated stiffness because [`Csr`] carries `sub` and not `add`,
/// and the two are the same doubles here: negation is exact, `a - (-b)` computes `a + b`, and the
/// zero-dropping that follows is the behaviour of SciPy's `+` kernel as well.
pub fn update_matrix(stiffness: &Csr, mass: &Csr, sigma_k: f64, theta_k2_kappa2: f64) -> Csr {
    mass.scaled(1.0 + sigma_k)
        .sub(&stiffness.scaled(-theta_k2_kappa2))
}

// -- the scheme --------------------------------------------------------------------------------

/// `u^{-1} = u^0 - k v^0 + 1/2 k^2 a^0` with `a^0 = -kappa^2 W^-1 K u^0`.
///
/// The consistent second-order start: a single eigenmode oscillates as a clean discrete cosine and
/// a zero initial velocity is exact to second order. Nothing is zeroed at the ends — there are no
/// clamped nodes.
pub fn initial_previous(u0: &[f64], v0: &[f64], p: &Params) -> Vec<f64> {
    let ku0 = p.stiffness.matvec(u0);
    let kappa2 = p.kappa * p.kappa;
    let half_k2 = 0.5 * p.k * p.k;
    (0..p.nodes())
        .map(|i| {
            let accel = -kappa2 * ku0[i] / p.w[i];
            u0[i] - p.k * v0[i] + half_k2 * accel
        })
        .collect()
}

/// The step's right-hand side, in NumPy's summation order (left to right over the four terms).
pub fn step_rhs(u: &[f64], u_prev: &[f64], p: &Params) -> Vec<f64> {
    let sk = p.sigma * p.k;
    let k2 = p.k * p.k;
    let kappa2 = p.kappa * p.kappa;
    let ku = p.stiffness.matvec(u);
    let ku_prev = p.stiffness.matvec(u_prev);
    // `(1.0 - 2.0 * theta) * k2` and `theta * k2` are scalar-times-scalar in Python before the
    // array multiply, because `float * float * ndarray` associates to the left.
    let c_now = (1.0 - 2.0 * p.theta) * k2;
    let c_prev = p.theta * k2;
    (0..p.nodes())
        .map(|i| {
            let lop_u = -kappa2 * ku[i];
            let lop_prev = -kappa2 * ku_prev[i];
            p.w[i] * (2.0 * u[i] - u_prev[i])
                + c_now * lop_u
                + c_prev * lop_prev
                + sk * (p.w[i] * u_prev[i])
        })
        .collect()
}

/// One timestep: build the right-hand side and back-substitute.
///
/// # Panics
/// If the factorization refuses the right-hand side, which it cannot do for a state of the right
/// length — `Params::new` has already produced the factors.
pub fn step_into(u: &[f64], u_prev: &[f64], next: &mut [f64], p: &Params) {
    let rhs = step_rhs(u, u_prev, p);
    let solved =
        p.lu.solve(&rhs)
            .expect("the factored system matches the state length");
    next.copy_from_slice(&solved);
}

// -- diagnostics -------------------------------------------------------------------------------

/// `sum(a[i] * b[i])` left to right — the spelling `np.dot` does not use.
///
/// Kept as a named function rather than an iterator chain so that the summation order is the
/// visible thing about it. See the module header for why matching BLAS here was declined.
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    let mut acc = 0.0;
    for i in 0..a.len() {
        acc += a[i] * b[i];
    }
    acc
}

/// The potential bilinear form `P(f, g) = kappa^2 (K f) . g`, non-negative because `K` is PSD.
pub fn potential_form(f: &[f64], g: &[f64], p: &Params) -> f64 {
    let kf = p.stiffness.matvec(f);
    p.kappa * p.kappa * dot(&kf, g)
}

/// Discrete energy `E^n` (Joules) — conserved to machine precision when `sigma = 0`, monotone
/// decreasing otherwise.
///
/// Evaluated through the *same* `K` and `w` as the update, which is what makes `E^{n+1} = E^n` an
/// algebraic identity rather than an approximation.
pub fn energy(u: &[f64], u_prev: &[f64], p: &Params) -> f64 {
    let dt_u: Vec<f64> = (0..u.len()).map(|i| (u[i] - u_prev[i]) / p.k).collect();
    let weighted: Vec<f64> = (0..u.len()).map(|i| p.w[i] * dt_u[i]).collect();
    let kinetic = 0.5 * dot(&dt_u, &weighted);

    let p_nn = potential_form(u, u, p);
    let p_pp = potential_form(u_prev, u_prev, p);
    let p_np = potential_form(u, u_prev, p);
    let potential = 0.5 * p.theta * (p_nn + p_pp) + (0.5 - p.theta) * p_np;
    p.rho * (kinetic + potential)
}

// -- the model ---------------------------------------------------------------------------------

/// A discretized free-free beam, owning its two history buffers.
///
/// The Python binding does **not** use this type — it keeps the state in NumPy arrays for the
/// reason Phase 0 recorded (a Rust `Vec` view handed to Python is a use-after-free that reads
/// plausibly) and calls the free functions above. This exists so the native tests can drive a
/// beam without a Python interpreter.
#[derive(Debug, Clone)]
pub struct FreeBeam {
    /// The validated parameters, the operators and the factorization.
    pub p: Params,
    /// Current displacement field `u^n`.
    pub u: Vec<f64>,
    /// Previous displacement field `u^{n-1}`.
    pub u_prev: Vec<f64>,
    /// Completed steps.
    pub n: usize,
}

impl FreeBeam {
    /// A beam at rest.
    pub fn new(p: Params) -> Self {
        let nodes = p.nodes();
        FreeBeam {
            p,
            u: vec![0.0; nodes],
            u_prev: vec![0.0; nodes],
            n: 0,
        }
    }

    /// Set the initial displacement and velocity, and rebuild `u^{-1}`.
    pub fn set_state(&mut self, u0: &[f64], v0: &[f64]) {
        self.u_prev = initial_previous(u0, v0, &self.p);
        self.u.copy_from_slice(u0);
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

    /// Displacement at grid node `index` — a pickup for spectral analysis.
    pub fn displacement_at(&self, index: usize) -> f64 {
        self.u[index]
    }
}
