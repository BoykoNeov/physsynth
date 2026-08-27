//! Damped stiff string — theta-scheme FDTD with frequency-dependent loss (model #3).
//!
//! Port of `physsynth/core/string_damped.py`, HANDOFF §5:
//!
//! ```text
//! u_tt = c^2 u_xx - kappa^2 u_xxxx - 2 sigma0 u_t + 2 sigma1 u_txx
//! ```
//!
//! Model #2 plus the `+2 sigma1 u_txx` term, which damps a mode at a rate proportional to its
//! wavenumber squared — the ordering real strings have, and the cure for model #2's *backwards*
//! high-frequency under-damping. It costs exactly one extra block in the update matrix
//!
//! ```text
//! A = (1 + sigma0 k) I - theta k^2 L - sigma1 k D2
//! ```
//!
//! and one extra right-hand-side term. The stored energy is model #2's, unchanged: both losses
//! enter only its *rate of change*, and both are dissipative by summation by parts, so passivity is
//! unconditional. The Python module's docstring is the reference for the physics.
//!
//! # This is deliberately a near-copy of [`crate::string_stiff`]
//!
//! See that module's header. `sigma1 = 0` here must be `array_equal` to a stiff string over a
//! 1,500-step run including its energy trace, and that anchor is only a *test* while the two are
//! two transcriptions. Merging them would make it vacuous. The duplication buys a detector.
//!
//! # What model #3 has that model #2 does not: a second consumer of the factor
//!
//! [`apply_ainv`] exposes the action of `A^{-1}` so a coupled element can precompute this string's
//! one-step driving-point admittance — `bow`, `collision::BarrierString` and the bridges in
//! `connection` all do. It is the same factor [`step_into`] uses, which is the point: a coupled
//! model's force correction is exact only if it is the *same* solve.

use crate::banded::{self, BandedError};
use crate::fmt::py_float;
use crate::ops;
use crate::pyfloat::scalar_pow;
use crate::sparse::Csr;
use crate::string_stiff::{dot, THETA_DEFAULT};

/// Re-exported so a caller does not have to know that model #3 borrows model #2's default.
pub const THETA: f64 = THETA_DEFAULT;

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
    /// Negative frequency-independent loss.
    NegativeSigma0,
    /// Negative frequency-dependent loss.
    NegativeSigma1,
    /// `theta` outside `(0, 1]`. Carries the offending value, which the message quotes.
    BadTheta(f64),
    /// The boundary spec was not `"supported"`; the caller formats the message.
    BadBoundary,
    /// `A` was not positive definite.
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
            ParamError::NegativeSigma0 => {
                write!(f, "sigma0 (frequency-independent loss) must be >= 0.")
            }
            ParamError::NegativeSigma1 => {
                write!(f, "sigma1 (frequency-dependent loss) must be >= 0.")
            }
            ParamError::BadTheta(t) => {
                write!(f, "theta must be in (0, 1], got {}.", py_float(*t))
            }
            ParamError::BadBoundary => write!(f, "boundary must be 'supported'."),
            ParamError::NotFactorable(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for ParamError {}

/// The validated parameter set plus the three time-constant objects: `L`, `D2` and the factor.
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
    /// Frequency-independent loss.
    pub sigma0: f64,
    /// Frequency-dependent loss.
    pub sigma1: f64,
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
    /// Interior spatial operator `L`, canonical CSR.
    pub op_l: Csr,
    /// Second-difference matrix, kept separately for the `sigma1` term. Canonical CSR.
    pub op_d2: Csr,
    /// Upper-banded Cholesky factor of `A`, two superdiagonals, row-major `3 x (n - 1)`.
    pub chol: Vec<f64>,
}

impl Params {
    /// Validate, derive, assemble `L` and `D2`, and factor `A`.
    ///
    /// `boundary_ok` carries a boundary spec the caller could not parse, so the *order* of the
    /// checks stays Python's; `n` is `i64` so `N = 1` and `N = -3` take the same documented path.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        l: f64,
        t: f64,
        rho: f64,
        fs: f64,
        n: i64,
        kappa: f64,
        sigma0: f64,
        sigma1: f64,
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
        if sigma0 < 0.0 {
            return Err(ParamError::NegativeSigma0);
        }
        if sigma1 < 0.0 {
            return Err(ParamError::NegativeSigma1);
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
        let b = (scalar_pow(std::f64::consts::PI, 2.0) * scalar_pow(kappa, 2.0))
            / (scalar_pow(c, 2.0) * scalar_pow(l, 2.0));

        let op_d2 = ops::second_difference_matrix(n, h);
        let mut op_l = op_d2.scaled(scalar_pow(c, 2.0));
        if kappa != 0.0 {
            op_l = op_l.sub(&ops::biharmonic_matrix(n, h).scaled(scalar_pow(kappa, 2.0)));
        }

        let ab = update_matrix_bands(
            &op_l,
            &op_d2,
            sigma0 * k,
            theta * scalar_pow(k, 2.0),
            sigma1 * k,
            sigma1 != 0.0,
        );
        let chol =
            banded::cholesky_banded_upper(ab, 2, n - 1).map_err(ParamError::NotFactorable)?;

        Ok(Params {
            l,
            t,
            rho,
            fs,
            n,
            kappa,
            sigma0,
            sigma1,
            theta,
            c,
            h,
            k,
            lam,
            b,
            op_l,
            op_d2,
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

    /// Node positions — `np.linspace(0.0, L, N + 1)`, endpoint overwritten as NumPy does.
    pub fn grid(&self) -> Vec<f64> {
        let step = self.l / (self.n as f64);
        let mut x: Vec<f64> = (0..self.nodes()).map(|i| (i as f64) * step).collect();
        x[self.n] = self.l;
        x
    }
}

/// The three upper bands of `A = (1 + sigma0 k) I - theta_k2 L - sigma1_k D2`.
///
/// `with_sigma1` reproduces the Python guard: at `sigma1 == 0` the term is *skipped*, not added as
/// a zero, which is what keeps model #3 at `sigma1 = 0` bit-identical to model #2. Only the
/// diagonals of `A` are read, so `A` is never assembled — `csr.diagonal(d)` picks by column and is
/// independent of stored order, which is why §18's sort cannot move these bands.
pub fn update_matrix_bands(
    op_l: &Csr,
    op_d2: &Csr,
    sigma0_k: f64,
    theta_k2: f64,
    sigma1_k: f64,
    with_sigma1: bool,
) -> Vec<f64> {
    let n_int = op_l.nrows();
    let mut ab = vec![0.0; 3 * n_int];
    let band = |i: usize, j: usize, base: f64| -> f64 {
        let v = base - theta_k2 * op_l.get(i, j);
        if with_sigma1 {
            v - sigma1_k * op_d2.get(i, j)
        } else {
            v
        }
    };
    for i in 0..n_int {
        ab[2 * n_int + i] = band(i, i, 1.0 + sigma0_k);
        if i >= 1 {
            ab[n_int + i] = band(i - 1, i, 0.0);
        }
        if i >= 2 {
            ab[i] = band(i - 2, i, 0.0);
        }
    }
    ab
}

// -- kernels -----------------------------------------------------------------------------------

/// `L u` on the full grid, zeros at the two clamped nodes — `_apply_L`.
pub fn apply_l_full(u_full: &[f64], p: &Params) -> Vec<f64> {
    let mut out = vec![0.0; u_full.len()];
    let interior = p.op_l.matvec(&u_full[1..u_full.len() - 1]);
    out[1..u_full.len() - 1].copy_from_slice(&interior);
    out
}

/// The consistent second-order start `u^{-1} = u^0 - k v^0 + 1/2 k^2 L u^0`, ends clamped.
///
/// The start is the **lossless** one in both models: neither loss term appears, so a single
/// eigenmode still opens as a clean discrete cosine and the first few steps of a damped run
/// deviate slightly from the asymptotic decay. That is the Python behaviour, deliberately.
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
/// rhs = 2 u + (1 - 2 theta) k^2 (L u) - u_prev + theta k^2 (L u_prev) + sigma0 k u_prev
///       [ - sigma1 k (D2 u_prev), only when sigma1 != 0 ]
/// ```
///
/// The `sigma1` term is a **separate** subtraction over the whole vector, after the five-term sum,
/// exactly as the Python `rhs = rhs - ...` line is — not folded into the per-node expression, which
/// would be a different rounding.
pub fn step_rhs(u: &[f64], u_prev: &[f64], p: &Params) -> Vec<f64> {
    let s0k = p.sigma0 * p.k;
    let k2 = scalar_pow(p.k, 2.0);
    let lu = p.op_l.matvec(&u[1..u.len() - 1]);
    let lu_prev = p.op_l.matvec(&u_prev[1..u_prev.len() - 1]);
    let a = (1.0 - 2.0 * p.theta) * k2;
    let b = p.theta * k2;
    let mut rhs: Vec<f64> = (0..p.interior())
        .map(|i| {
            let un = u[i + 1];
            let up = u_prev[i + 1];
            (((2.0 * un + a * lu[i]) - up) + b * lu_prev[i]) + s0k * up
        })
        .collect();
    if p.sigma1 != 0.0 {
        let s1k = p.sigma1 * p.k;
        let d2_up = p.op_d2.matvec(&u_prev[1..u_prev.len() - 1]);
        for (i, r) in rhs.iter_mut().enumerate() {
            *r -= s1k * d2_up[i];
        }
    }
    rhs
}

/// Solve `A x = rhs` on the interior with this string's factor — `apply_Ainv`.
pub fn apply_ainv(rhs: &[f64], p: &Params) -> Vec<f64> {
    banded::cho_solve_banded_upper(&p.chol, 2, p.interior(), rhs)
        .expect("the factor and the right-hand side are shaped by construction")
}

/// Advance one step into `next` (full grid, ends left at zero).
pub fn step_into(u: &[f64], u_prev: &[f64], next: &mut [f64], p: &Params) {
    let sol = apply_ainv(&step_rhs(u, u_prev, p), p);
    next.fill(0.0);
    next[1..u.len() - 1].copy_from_slice(&sol);
}

/// `P(f, g) = <-L f, g> = -h (L f) . g` on interior vectors, reduced left to right.
pub fn potential_form(f: &[f64], g: &[f64], p: &Params) -> f64 {
    -p.h * dot(&p.op_l.matvec(f), g)
}

/// Discrete energy `E^n` (Joules) — model #2's form, unchanged.
///
/// The loss terms never enter the stored energy, only its rate of change.
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

/// A damped stiff string with its own buffers — for Rust callers and for `cargo test`.
#[derive(Debug, Clone)]
pub struct DampedStiffString {
    /// Parameters, operators and factor.
    pub p: Params,
    /// Current displacement `u^n` on the full grid.
    pub u: Vec<f64>,
    /// Previous displacement `u^{n-1}` on the full grid.
    pub u_prev: Vec<f64>,
    /// Completed steps.
    pub n: usize,
}

impl DampedStiffString {
    /// A string at rest.
    pub fn new(p: Params) -> Self {
        let nodes = p.nodes();
        DampedStiffString {
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
