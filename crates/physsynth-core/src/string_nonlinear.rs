//! Tension-modulated string — the Kirchhoff–Carrier nonlinearity (model #9).
//!
//! Port of `physsynth/core/string_nonlinear.py`. Model #3 plus a state-dependent tension:
//!
//! ```text
//! rho u_tt = T(t) u_xx - rho kappa^2 u_xxxx - 2 rho sigma0 u_t + 2 rho sigma1 u_txx
//! T(t) = T0 + (EA/2L) I,   I = integral u_x^2 dx        ("the stretch")
//! ```
//!
//! Displacing a string stretches it, which raises its tension, which raises its pitch — so a hard
//! pluck starts sharp and glides down. The Python module's docstring is the reference for the
//! physics: why the nonlinear excess alone is averaged at `theta = 1/2` (so that `EA = 0` stays
//! bit-for-bit model #3), why the discrete-gradient tension collapses to a plain midpoint with no
//! `0/0` branch, and why single-mode motion is *dynamically* unstable above `dT/T0 ~ 3`.
//!
//! # What makes this model different from every string ported before it
//!
//! `A = A0 - beta D2` depends on the tension, so the factorization is **not** reusable: each step
//! is a scalar root-find for the tension excess `dT`, and every residual evaluation costs one
//! banded refactor plus one solve. Two consequences the earlier string batches never had to face:
//!
//! - **The factor is inside a loop, not at construction.** [`crate::banded`] is called ~4.4 times
//!   per step rather than once per string — measured, and only that low because [`solve_tension`]
//!   reproduces the Python original's per-call memo (see [`update_for`]; without it a step takes
//!   roughly twice as many).
//! - **A last bit in the residual can change an *integer*.** [`crate::root::brentq`]'s iterate
//!   sequence is a function of the residual's exact value, so an implementation that differs by one
//!   ulp takes a different number of iterations. Measured on the Python side before this port was
//!   written: swapping `_stretch`'s reduction from BLAS `ddot` to a left-to-right sum changed the
//!   per-step evaluation count on **1,400 of 5,000 steps**. That is why `physsynth/core/portable.py`
//!   now covers the stretch as well — §19.2 of the migration plan — and it is the reason this model
//!   can be compared to the bit at all.
//!
//! # The reduction that had to move, and why it is this batch's business
//!
//! When `portable.py` was written (§18.2) it deliberately left `_stretch` on `np.dot`, on the
//! grounds that the stretch is on the *update* path and "is compared to nothing". Porting this
//! model made that false, which is exactly §18.3's rule — *a decision justified by "nothing
//! downstream depends on this" expires the moment something downstream ports* — arriving one batch
//! after it was written down. The reduction is now `portable.dot` on both sides.
//!
//! # What is deliberately absent
//!
//! `string_coefficients_from_material` and its `StringCoefficients` named tuple stay in Python.
//! They are a construction-time *modelling oracle* — six floats derived from a material and a
//! radius, never touched again — so they are not on any trajectory, and reproducing a named
//! tuple's tuple protocol through PyO3 would buy nothing measurable. §11.2.1's rule cuts this way:
//! port the function group that is on the hot path, and name the half that is not.

use crate::banded::{self, BandedError};
use crate::fmt::py_float;
use crate::ops;
use crate::pyfloat::scalar_pow;
use crate::root::{self, RootError};
use crate::sparse::Csr;
use crate::string_damped::update_matrix_bands;
use crate::string_stiff::{dot, THETA_DEFAULT};

/// Re-exported so a caller does not have to know model #9 borrows model #2's default.
pub const THETA: f64 = THETA_DEFAULT;

/// Relative tolerance of the scalar tension solve — `TENSION_TOL_DEFAULT`.
pub const TENSION_TOL_DEFAULT: f64 = 1e-13;

/// Cap on doubling the tension bracket — `MAX_BRACKET_EXPANSIONS`.
pub const MAX_BRACKET_EXPANSIONS: usize = 40;

/// `brentq`'s relative tolerance, as `string_nonlinear.py` passes it. Clears SciPy's own
/// `4 * eps` floor, which its Python wrapper checks and this transcription deliberately does not.
pub const BRENTQ_RTOL: f64 = 8.9e-16;

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
    /// Negative axial stiffness — the nonlinearity cannot soften.
    NegativeEa,
    /// Negative frequency-independent loss.
    NegativeSigma0,
    /// Negative frequency-dependent loss.
    NegativeSigma1,
    /// `theta` outside `(0, 1]`. Carries the offending value, which the message quotes.
    BadTheta(f64),
    /// `tension_tol` was not positive.
    BadTensionTol,
    /// The boundary spec was not `"supported"`; the caller formats the message, which quotes the
    /// offending value with Python's `repr`.
    BadBoundary,
    /// `A0` was not positive definite.
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
            ParamError::NegativeEa => write!(f, "EA (axial stiffness) must be >= 0."),
            ParamError::NegativeSigma0 => {
                write!(f, "sigma0 (frequency-independent loss) must be >= 0.")
            }
            ParamError::NegativeSigma1 => {
                write!(f, "sigma1 (frequency-dependent loss) must be >= 0.")
            }
            ParamError::BadTheta(t) => {
                write!(f, "theta must be in (0, 1], got {}.", py_float(*t))
            }
            ParamError::BadTensionTol => write!(f, "tension_tol must be > 0."),
            ParamError::BadBoundary => write!(f, "boundary must be 'supported'."),
            ParamError::NotFactorable(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for ParamError {}

/// The validated parameter set plus the time-constant operators and the `dT = 0` factor.
#[derive(Debug, Clone, PartialEq)]
pub struct Params {
    /// Length (m).
    pub l: f64,
    /// **Rest** tension (N).
    pub t: f64,
    /// Linear density (kg/m).
    pub rho: f64,
    /// Sample rate (Hz).
    pub fs: f64,
    /// Number of spatial segments; the grid has `n + 1` nodes and `n - 1` unknowns.
    pub n: usize,
    /// Stiffness coefficient `sqrt(E I / rho)` (m^2/s).
    pub kappa: f64,
    /// Axial stiffness (N) — the nonlinearity. `0` is model #3, bit-for-bit.
    pub ea: f64,
    /// Frequency-independent loss.
    pub sigma0: f64,
    /// Frequency-dependent loss.
    pub sigma1: f64,
    /// Time-averaging weight of the **linear** operator, in `(0, 1]`.
    pub theta: f64,
    /// Relative tolerance of the scalar tension solve.
    pub tension_tol: f64,
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
    /// The governing nonlinearity ratio `EA / T0`.
    pub ea_over_t: f64,
    /// Interior spatial operator `L`, canonical CSR.
    pub op_l: Csr,
    /// Second-difference matrix, canonical CSR — the `sigma1` term *and* the tension term.
    pub op_d2: Csr,
    /// Upper bands of `A0` (model #3's matrix), row-major `3 x (n - 1)`.
    pub ab0: Vec<f64>,
    /// Upper bands of `D2`, the same shape — what `beta` scales onto `ab0` each residual.
    pub ab_d2: Vec<f64>,
    /// Cholesky factor of `A0` alone: the `EA = 0` path, and model #3's factor exactly.
    pub chol0: Vec<f64>,
}

impl Params {
    /// Validate, derive, assemble `L` and `D2`, and factor `A0`.
    ///
    /// The check order is Python's, which is what makes a doubly-invalid parameter set produce the
    /// *same* message on both sides; `n` is `i64` so `N = 1` and `N = -3` take one documented path.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        l: f64,
        t: f64,
        rho: f64,
        fs: f64,
        n: i64,
        kappa: f64,
        ea: f64,
        sigma0: f64,
        sigma1: f64,
        theta: f64,
        tension_tol: f64,
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
        if ea < 0.0 {
            return Err(ParamError::NegativeEa);
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
        if tension_tol <= 0.0 {
            return Err(ParamError::BadTensionTol);
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
        let ea_over_t = ea / t;

        let op_d2 = ops::second_difference_matrix(n, h);
        let mut op_l = op_d2.scaled(scalar_pow(c, 2.0));
        if kappa != 0.0 {
            op_l = op_l.sub(&ops::biharmonic_matrix(n, h).scaled(scalar_pow(kappa, 2.0)));
        }

        let ab0 = update_matrix_bands(
            &op_l,
            &op_d2,
            sigma0 * k,
            theta * scalar_pow(k, 2.0),
            sigma1 * k,
            sigma1 != 0.0,
        );
        let ab_d2 = bands_of(&op_d2);
        let chol0 = banded::cholesky_banded_upper(ab0.clone(), 2, n - 1)
            .map_err(ParamError::NotFactorable)?;

        Ok(Params {
            l,
            t,
            rho,
            fs,
            n,
            kappa,
            ea,
            sigma0,
            sigma1,
            theta,
            tension_tol,
            c,
            h,
            k,
            lam,
            b,
            ea_over_t,
            op_l,
            op_d2,
            ab0,
            ab_d2,
            chol0,
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

/// The three upper bands of a symmetric pentadiagonal operator — `_banded`.
///
/// `csr.diagonal(d)` picks by *column*, so this is independent of stored index order and §18's
/// sort provably cannot move it. `D2` is tridiagonal, so the second superdiagonal is all zeros;
/// the row exists because `A` is pentadiagonal and the two are subtracted elementwise.
pub fn bands_of(m: &Csr) -> Vec<f64> {
    let n = m.nrows();
    let mut ab = vec![0.0; 3 * n];
    for i in 0..n {
        ab[2 * n + i] = m.get(i, i);
        if i >= 1 {
            ab[n + i] = m.get(i - 1, i);
        }
        if i >= 2 {
            ab[i] = m.get(i - 2, i);
        }
    }
    ab
}

// -- kernels -----------------------------------------------------------------------------------

/// Stretch `I = h ||delta_x+ u||^2` on the **full** grid, ends included — `_stretch`.
///
/// The `h` is `I = h sum ((u_{j+1} - u_j)/h)^2 = sum (du)^2 / h`; dropping it looks exactly like a
/// mis-scaled `EA` and passes every qualitative test.
///
/// The reduction is left to right, which is `portable.dot` and *not* `np.dot`. See the module
/// header: this one is on the update path, and matching it is what keeps brentq's iteration count
/// equal on the two sides.
pub fn stretch(u_full: &[f64], p: &Params) -> f64 {
    let du: Vec<f64> = (1..u_full.len())
        .map(|i| u_full[i] - u_full[i - 1])
        .collect();
    dot(&du, &du) / p.h
}

/// Stretch of an interior-only vector — `_stretch_int`. The clamped ends contribute their own
/// slopes, and those two `** 2` are Python float powers rather than multiplies (§17.3).
///
/// **The association is `(dot + u_0^2) + u_last^2`, left to right, and it is load-bearing.**
/// Grouping the two end terms first is a different sum, and it was the batch's one real porting
/// error — caught by `delta_tension` and *not* by the trajectory, which stayed bit-identical
/// through it because `beta = k^2 dT / (2 rho)` is ~1e-9 here and swallows a last bit of `dT`
/// long before it reaches a band entry. §19.4.
pub fn stretch_int(u_int: &[f64], p: &Params) -> f64 {
    let du: Vec<f64> = (1..u_int.len()).map(|i| u_int[i] - u_int[i - 1]).collect();
    let last = u_int[u_int.len() - 1];
    ((dot(&du, &du) + scalar_pow(u_int[0], 2.0)) + scalar_pow(last, 2.0)) / p.h
}

/// `L u` on the full grid, zeros at the two clamped nodes — `_apply_L`.
pub fn apply_l_full(u_full: &[f64], p: &Params) -> Vec<f64> {
    let mut out = vec![0.0; u_full.len()];
    let interior = p.op_l.matvec(&u_full[1..u_full.len() - 1]);
    out[1..u_full.len() - 1].copy_from_slice(&interior);
    out
}

/// `D2 u` on the full grid, zeros at the two clamped nodes — `_apply_D2`.
pub fn apply_d2_full(u_full: &[f64], p: &Params) -> Vec<f64> {
    let mut out = vec![0.0; u_full.len()];
    let interior = p.op_d2.matvec(&u_full[1..u_full.len() - 1]);
    out[1..u_full.len() - 1].copy_from_slice(&interior);
    out
}

/// The consistent second-order start, **including the nonlinear tension at `t = 0`**.
///
/// ```text
/// u^{-1} = u^0 - k v^0 + 1/2 k^2 [ L u^0 + (dT_0 / rho) D2 u^0 ],  dT_0 = (EA/2L) I(u^0)
/// ```
///
/// so a single eigenmode opens as a clean discrete Duffing cosine. At `EA = 0` the bracket's second
/// term is **skipped**, not added as a zero — which is what makes the start bit-identical to model
/// #3's rather than merely equal to it.
pub fn initial_previous(u0: &mut [f64], v0: &[f64], p: &Params) -> Vec<f64> {
    let last = u0.len() - 1;
    u0[0] = 0.0;
    u0[last] = 0.0;
    let mut accel = apply_l_full(u0, p);
    if p.ea != 0.0 {
        let d_t0 = (p.ea / (2.0 * p.l)) * stretch(u0, p);
        let d2 = apply_d2_full(u0, p);
        for (a, d) in accel.iter_mut().zip(d2.iter()) {
            *a += (d_t0 / p.rho) * d;
        }
    }
    let half_k2 = 0.5 * scalar_pow(p.k, 2.0);
    let mut prev: Vec<f64> = (0..u0.len())
        .map(|i| (u0[i] - p.k * v0[i]) + half_k2 * accel[i])
        .collect();
    prev[0] = 0.0;
    prev[last] = 0.0;
    prev
}

/// Model #3's right-hand side on the interior, expression for expression — `rhs0`.
///
/// This is `string_damped::step_rhs`'s arithmetic, written out again rather than called, because
/// the two modules are deliberate near-copies and merging them would make the `EA = 0` anchor
/// compare an implementation against itself (§18.7).
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

/// One candidate tension's update: `(A0 - beta D2) u+ = rhs0 + beta D2 u-` — `u_next_for`.
///
/// The Python original memoizes this on `dT`, and [`solve_tension`] reproduces the memo. It is
/// **performance only** — the same `dT` refactors to the same bits either way — but it is not the
/// rounding error it first looks like: measured on the flagship fixture, a step takes **4.4**
/// banded solves with the memo and roughly twice that without, because `brentq`'s first two
/// evaluations land exactly on the bracket ends the caller has already solved for, and its last
/// one lands on the root the caller is about to solve for again.
pub fn update_for(
    d_t: f64,
    rhs0: &[f64],
    d2_up: &[f64],
    p: &Params,
) -> Result<Vec<f64>, BandedError> {
    let beta = scalar_pow(p.k, 2.0) * d_t / (2.0 * p.rho);
    let ab: Vec<f64> = (0..p.ab0.len())
        .map(|i| p.ab0[i] - beta * p.ab_d2[i])
        .collect();
    let chol = banded::cholesky_banded_upper(ab, 2, p.interior())?;
    let rhs: Vec<f64> = (0..rhs0.len()).map(|i| rhs0[i] + beta * d2_up[i]).collect();
    banded::cho_solve_banded_upper(&chol, 2, p.interior(), &rhs)
}

/// How a step's tension solve ended. The three telemetry fields are *public* on the Python class,
/// and two of them are integers — so they are compared for equality, not for closeness.
#[derive(Debug, Clone, PartialEq)]
pub struct TensionSolve {
    /// `u^{n+1}` on the interior.
    pub u_next: Vec<f64>,
    /// The tension excess `dT` (N) that was applied.
    pub delta_tension: f64,
    /// Whether a bracket was found. `false` means the doubling loop hit its cap.
    pub converged: bool,
    /// How many times the bracket was doubled this step.
    pub expansions: usize,
}

/// Why a tension solve could not be completed at all.
#[derive(Debug, Clone, PartialEq)]
pub enum TensionError {
    /// A candidate `A(dT)` was not positive definite. Cannot happen for `dT >= 0` in exact
    /// arithmetic (`-D2` is SPD), so it is a refusal rather than a documented outcome.
    Banded(BandedError),
    /// `brentq` refused or ran out of iterations. The bracket is established before it is called,
    /// so `SameSign` is unreachable and `NotConverged` would be a 100-iteration Brent.
    Root(RootError),
}

impl std::fmt::Display for TensionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TensionError::Banded(e) => write!(f, "{e}"),
            TensionError::Root(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for TensionError {}

/// The scalar root-find for the tension excess — `_solve_tension`.
///
/// Solves `resid(dT) = dT - (EA/4L)(I^{n+1}(dT) + I^{n-1}) = 0`.
///
/// **A bracket always exists and doubling always finds it.** `resid(0) <= 0` because stretches are
/// non-negative, and `resid -> +inf` as `dT -> inf` because `u+ -> -u-` there, so `I^{n+1}` is
/// bounded by `I^{n-1}` while the linear `dT` is not. `I^{n+1}` is **not** monotone in `dT`, which
/// is why the seed takes `max(I^{n+1}(0), I^{n-1})` rather than the tempting `I^{n+1}(0)`.
///
/// The unknown handed to `brentq` is normalised to `s = dT / dT_hi` in `[0, 1]`, so `tension_tol`
/// is a unit-free relative bar.
pub fn solve_tension(
    rhs0: &[f64],
    u_prev: &[f64],
    p: &Params,
) -> Result<TensionSolve, TensionError> {
    let i_prev = stretch(u_prev, p);
    let coeff = p.ea / (4.0 * p.l);
    let d2_up = p.op_d2.matvec(&u_prev[1..u_prev.len() - 1]);

    // The Python original's `cache: dict[float, ...]`, which is per-call and holds a handful of
    // entries — so a linear scan over a `Vec` is the same structure without a float-keyed hash.
    //
    // `to_bits` is NOT quite a Python dict's key comparison: a dict hashes `-0.0` and `0.0` to the
    // same slot and this does not. It cannot matter here — every `dT` reaching this closure is
    // `0.0`, `d_t_hi > 0`, or `s * d_t_hi` for `s` in `[0, 1]`, so `-0.0` never arrives — and
    // `to_bits` is the honest spelling of "the same double", where `==` would fold the two. The
    // difference is noted rather than relied on.
    let mut memo: Vec<(u64, Vec<f64>)> = Vec::new();
    let solve = |d_t: f64, memo: &mut Vec<(u64, Vec<f64>)>| -> Result<Vec<f64>, TensionError> {
        let key = d_t.to_bits();
        if let Some((_, u)) = memo.iter().find(|(k, _)| *k == key) {
            return Ok(u.clone());
        }
        let u = update_for(d_t, rhs0, &d2_up, p).map_err(TensionError::Banded)?;
        memo.push((key, u.clone()));
        Ok(u)
    };
    let resid = |d_t: f64, memo: &mut Vec<(u64, Vec<f64>)>| -> Result<f64, TensionError> {
        Ok(d_t - coeff * (stretch_int(&solve(d_t, memo)?, p) + i_prev))
    };

    let i_free = stretch_int(&solve(0.0, &mut memo)?, p);
    let mut d_t_hi = coeff * (i_free.max(i_prev) + i_prev);
    if d_t_hi <= 0.0 {
        // The string is exactly at rest: no stretch, so no modulation and nothing to solve.
        return Ok(TensionSolve {
            u_next: solve(0.0, &mut memo)?,
            delta_tension: 0.0,
            converged: true,
            expansions: 0,
        });
    }

    let mut expansions = 0;
    while resid(d_t_hi, &mut memo)? < 0.0 && expansions < MAX_BRACKET_EXPANSIONS {
        d_t_hi *= 2.0;
        expansions += 1;
    }
    if resid(d_t_hi, &mut memo)? < 0.0 {
        // The caller warns and records; this side reports the failure and hands back the state the
        // Python original hands back, so that a non-converged run is still comparable.
        return Ok(TensionSolve {
            u_next: solve(d_t_hi, &mut memo)?,
            delta_tension: d_t_hi,
            converged: false,
            expansions,
        });
    }

    // `brentq` cannot carry a `Result` through its `FnMut`, and the only error `resid` can raise is
    // a non-positive-definite `A`, which `dT >= 0` forbids. It is captured rather than swallowed.
    let mut failure: Option<TensionError> = None;
    let root_s = root::brentq(
        |s| match resid(s * d_t_hi, &mut memo) {
            Ok(v) => v,
            Err(e) => {
                failure.get_or_insert(e);
                f64::NAN
            }
        },
        0.0,
        1.0,
        p.tension_tol,
        BRENTQ_RTOL,
        root::DEFAULT_MAXITER,
    );
    if let Some(e) = failure {
        return Err(e);
    }
    let root_s = root_s.map_err(TensionError::Root)?;
    let delta_tension = root_s * d_t_hi;
    Ok(TensionSolve {
        u_next: solve(delta_tension, &mut memo)?,
        delta_tension,
        converged: true,
        expansions,
    })
}

/// `P(f, g) = <-L f, g> = -h (L f) . g` on interior vectors, reduced left to right.
pub fn potential_form(f: &[f64], g: &[f64], p: &Params) -> f64 {
    -p.h * dot(&p.op_l.matvec(f), g)
}

/// Model #3's energy form, unchanged — `_linear_energy`.
pub fn linear_energy(u: &[f64], u_prev: &[f64], p: &Params) -> f64 {
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

/// The stretch (membrane) part of `E^n` alone (J) — `(EA/16L)((I^n)^2 + (I^{n-1})^2)`.
///
/// The **two-time half-average** is what telescopes against the scheme's nonlinear power; a
/// single-level `V(I^n)` is a two-step invariant and oscillates spuriously (model #6's lesson).
/// Exactly `0.0` at `EA = 0`, by an early return rather than by arithmetic.
pub fn nonlinear_energy(u: &[f64], u_prev: &[f64], p: &Params) -> f64 {
    if p.ea == 0.0 {
        return 0.0;
    }
    let i_n = stretch(u, p);
    let i_p = stretch(u_prev, p);
    (p.ea / (16.0 * p.l)) * (scalar_pow(i_n, 2.0) + scalar_pow(i_p, 2.0))
}

/// Discrete energy `E^n` (Joules) — model #3's energy plus the nonlinear stretch term.
pub fn energy(u: &[f64], u_prev: &[f64], p: &Params) -> f64 {
    linear_energy(u, u_prev, p) + nonlinear_energy(u, u_prev, p)
}

// -- the native owning struct ------------------------------------------------------------------

/// A tension-modulated string with its own buffers — for Rust callers and for `cargo test`.
#[derive(Debug, Clone)]
pub struct TensionModulatedString {
    /// Parameters, operators and the `dT = 0` factor.
    pub p: Params,
    /// Current displacement `u^n` on the full grid.
    pub u: Vec<f64>,
    /// Previous displacement `u^{n-1}` on the full grid.
    pub u_prev: Vec<f64>,
    /// Completed steps.
    pub n: usize,
    /// Tension excess `dT` (N) applied by the most recent step.
    pub delta_tension: f64,
    /// Whether the most recent tension solve converged.
    pub converged: bool,
    /// Cumulative bracket doublings — **diagnostic, not an error**.
    pub bracket_expansions: usize,
    /// Cumulative steps whose tension solve failed. Never render such a run as physics.
    pub n_not_converged: usize,
}

impl TensionModulatedString {
    /// A string at rest.
    pub fn new(p: Params) -> Self {
        let nodes = p.nodes();
        TensionModulatedString {
            p,
            u: vec![0.0; nodes],
            u_prev: vec![0.0; nodes],
            n: 0,
            delta_tension: 0.0,
            converged: true,
            bracket_expansions: 0,
            n_not_converged: 0,
        }
    }

    /// Set the initial displacement and velocity (both full-grid).
    pub fn set_state(&mut self, u0: &[f64], v0: &[f64]) {
        let mut u = u0.to_vec();
        self.u_prev = initial_previous(&mut u, v0, &self.p);
        self.u = u;
        self.n = 0;
        self.delta_tension = 0.0;
        self.converged = true;
    }

    /// Advance one timestep, rolling the history.
    ///
    /// Returns `Ok(true)` on a converged step and `Ok(false)` when the bracket search hit its cap —
    /// which is where the Python original emits a `RuntimeWarning` naming the step. The state is
    /// still advanced in that case, deliberately: a run that has to be discarded is more useful
    /// comparable than truncated.
    pub fn step(&mut self) -> Result<bool, TensionError> {
        let rhs0 = step_rhs(&self.u, &self.u_prev, &self.p);
        let (sol, d_t, converged, expansions) = if self.p.ea == 0.0 {
            let sol = banded::cho_solve_banded_upper(&self.p.chol0, 2, self.p.interior(), &rhs0)
                .map_err(TensionError::Banded)?;
            (sol, 0.0, true, 0)
        } else {
            let s = solve_tension(&rhs0, &self.u_prev, &self.p)?;
            (s.u_next, s.delta_tension, s.converged, s.expansions)
        };
        self.delta_tension = d_t;
        self.converged = converged;
        self.bracket_expansions += expansions;
        if !converged {
            self.n_not_converged += 1;
        }

        let mut next = vec![0.0; self.p.nodes()];
        next[1..self.p.nodes() - 1].copy_from_slice(&sol);
        std::mem::swap(&mut self.u_prev, &mut self.u);
        self.u = next;
        self.n += 1;
        Ok(converged)
    }

    /// Current stretch `I^n` (m).
    pub fn stretch(&self) -> f64 {
        stretch(&self.u, &self.p)
    }

    /// Current total tension `T0 + (EA/2L) I^n` (N). Always `>= T0` — hardening only.
    pub fn tension(&self) -> f64 {
        self.p.t + (self.p.ea / (2.0 * self.p.l)) * self.stretch()
    }

    /// Discrete energy `E^n` (Joules).
    pub fn energy(&self) -> f64 {
        energy(&self.u, &self.u_prev, &self.p)
    }

    /// The stretch part of `E^n` alone (J).
    pub fn nonlinear_energy(&self) -> f64 {
        nonlinear_energy(&self.u, &self.u_prev, &self.p)
    }
}
