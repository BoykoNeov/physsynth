//! Contact & collision primitives, and the two solves built on them (models #7 and #8).
//!
//! `docs/dev/rust-migration-plan.md` §16. The Python original, `physsynth/core/collision.py`, is
//! the shared home of the energy-conserving contact scheme: the mallet (`mallet.py`, model #7)
//! uses its **scalar** solve, the distributed barrier (`BarrierString`, model #8) its **vector**
//! one. This module ports the primitives, both solves, and — since §23 — the barrier model itself.
//!
//! `BarrierString` was deferred at §16 because it wraps a `DampedStiffString`, which was Python
//! then. That host landed in §18 and the deferral expired without anyone being told; §20.11 caught
//! it, and [`BarrierString`] here is what closes Phase 3.
//!
//! # The physics, in one paragraph
//!
//! With penetration `eta` (positive in contact), the felt stores potential density
//! `phi(eta) = K/(a+1) [eta]+^(a+1)`. Evaluating `phi'` pointwise drifts the energy at `O(k^2)`;
//! the **discrete gradient** `f = (phi(eta+) - phi(eta-)) / (eta+ - eta-)` makes the contact power
//! telescope exactly. That quotient is a genuine `0/0` whenever the penetration is not moving —
//! quiet, stuck, or grazing — so the removable singularity is handled by a midpoint Taylor branch,
//! and that branch is mandatory rather than defensive: without it the scheme returns `NaN` in the
//! commonest state it is ever in.
//!
//! # The finding this port had to be built around
//!
//! **A NumPy array and a NumPy scalar do not compute the same power.**
//!
//! `contact_potential` and its neighbours are written once in Python and called two ways — with a
//! float (the mallet's scalar solve) and with an array (the barrier's vector solve). They are
//! *not* the same computation. NumPy's float64 `power` **ufunc loop** carries a fast-path ladder
//! for the exponents `-1, 0, 0.5, 1, 2`, spelling `x**0.5` as `sqrt` and `x**2` as `x*x`; the
//! **scalar** path takes no such shortcut and calls the C library's `pow`. Measured on 2026-08-27
//! over 200,000 realistic penetrations, `x**0.5` and `pow(x, 0.5)` disagree in 94 of them and
//! `x**2` and `pow(x, 2)` in 53 — always by one ulp, and always with the shortcut being the *more*
//! accurate of the two.
//!
//! Two consequences, neither cosmetic:
//!
//! * The exponents in play are `a+1`, `a`, `a-1`. So the divergence lands exactly on the two
//!   configurations the project uses most: **`a = 1`**, the closed-form-oracle case (`a+1 = 2`),
//!   and **`a = 1.5`**, the barrier default (`a-1 = 0.5`).
//! * `_force_total_vec`'s Python docstring claims it is "numerically identical to calling
//!   `contact_force_total` per component". It is not, and the claim predates this port rather than
//!   being introduced by it: measured over 200,000 pairs at `a = 1`, the vector and scalar force
//!   paths disagree in 174 of them, by up to `1.5e-12` relative; the derivative paths disagree in
//!   86 of 100,000, by up to `5.6e-12`.
//!
//! So [`PowPath`] is a parameter of every primitive here, and each caller picks it by which Python
//! path it stands in for. Repairing the inconsistency is a physics change — it moves both models'
//! trajectories — and belongs to whoever retires the Python side, not to a port whose whole job is
//! to be indistinguishable from it.
//!
//! # Where bit-identity ends here, and it is not the LU
//!
//! See [`crate::dense`]: the vector solve's `G @ F(eta)` is a dense BLAS matvec feeding back into
//! the next Newton iterate, which is §14.2's construction exactly. The scalar solve contains no
//! reduction at all and *is* expected to be bit-identical; so is the vector solve at `m = 1`, where
//! the matvec is a single multiply. Those two cases are this batch's cause-separator: if either
//! diverges, the transcription is wrong and no story about BLAS applies to it.
//!
//! # The barrier shell adds a SECOND matvec, and at two contact nodes the state cannot see it
//!
//! §23. [`apply`] injects the contact force through `u[1..-1] += force_pref * (cols_mat @ F)` — a
//! second dense matvec, on the update path, and one nothing compared across the languages before
//! the model itself ported. Measured 2026-08-27 against the left-to-right row sum written here,
//! over the parity file's fixtures and 2,000 steps: identical at `m = 1` (0 rows — the sum is one
//! product), and at `m = 2` it **differs in 1,291 of 158,000 rows** while the trajectory stays
//! bit-identical.
//!
//! **That is a mechanism, not a coincidence, and the distinction is the finding.** A *two*-term
//! sum can only be reordered into a different double if its two terms **cancel** — and where they
//! cancel the sum is tiny, so the correction is tiny. At every one of those 1,291 rows the
//! correction is at most `9.3e-13` of `u` (`7e-12` at the linear exponent the parity test uses),
//! which makes one of *its* ulps worth about `1e-11` of one of `u`'s: it cannot survive the
//! addition, and none of them does. The error of a two-term reduction is correlated with its own
//! smallness.
//!
//! The control is the same code at 79 terms, where that correlation is gone. There the matvec
//! differs on 14,746 rows, the correction is an ordinary size where it does (median `1.2e-4` of
//! `u`), and roughly one difference in `1/1.2e-4` crosses a rounding boundary: 7 reach the state
//! over 2,000 steps and 30 over 6,000, which is what the ratio predicts to within a factor of two.
//! So **the exactness at two nodes is a statement about the length of the sum** — the general
//! version of §16's "`m = 1` is the cause-separator", one term further along and with a reason
//! attached.
//!
//! At `m = 79` the regime does not change, because the *solve's* matvec was already spending the
//! bit-identity there: the shell contributes `<= 1.9e-14` of peak at 500 steps, against a `1e-13`
//! bar and a measured window of 1,175-1,584 steps that the solve sets.
//!
//! That is also why `portable.py` was **not** extended here. Its manoeuvre — move the Python side
//! to a spelling both languages can express — was considered and rejected on evidence: it would
//! change a shipped model's reference numbers (and the viewer's fret and jawari output) to buy
//! exactness that measurement shows is already there where it is provable, and that no spelling
//! can buy where the solve has already spent it.

use crate::dense;
use crate::pyfloat::scalar_pow;
use crate::root::{brentq, RootError};
use crate::string_damped;

/// Which of NumPy's two power spellings a primitive should reproduce.
///
/// See the module header. This is not a performance knob — the two produce different doubles, and
/// which one is correct depends only on whether the Python function being stood in for was handed
/// an array or a scalar.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PowPath {
    /// The float64 `power` ufunc loop's fast-path ladder: `-1 -> 1/x`, `0 -> 1`, `0.5 -> sqrt`,
    /// `1 -> x`, `2 -> x*x`, everything else `pow`.
    Array,
    /// NumPy's scalar path and CPython's `float.__pow__` alike: the C library's `pow`, always.
    Scalar,
}

impl PowPath {
    /// `x ** e` down whichever of the two paths this is.
    ///
    /// Public because `mallet` stands in for a `float.__pow__` outside this module and must reach
    /// the same `pow` — see `mallet::scalar_pow`, which wraps it in an `#[inline(never)]` so that
    /// a literal exponent cannot be constant-folded into `x * x` on the way.
    #[inline]
    pub fn pow(self, x: f64, e: f64) -> f64 {
        match self {
            PowPath::Scalar => x.powf(e),
            PowPath::Array => {
                if e == -1.0 {
                    1.0 / x
                } else if e == 0.0 {
                    1.0
                } else if e == 0.5 {
                    x.sqrt()
                } else if e == 1.0 {
                    x
                } else if e == 2.0 {
                    x * x
                } else {
                    x.powf(e)
                }
            }
        }
    }
}

/// One-sided felt/barrier potential `phi(eta) = K/(a+1) [eta]+^(a+1)`. Zero for `eta <= 0`.
#[inline]
pub fn contact_potential(eta: f64, stiffness: f64, alpha: f64, path: PowPath) -> f64 {
    let ep = eta.max(0.0);
    stiffness / (alpha + 1.0) * path.pow(ep, alpha + 1.0)
}

/// Elastic contact force `phi'(eta) = K [eta]+^a` (`>= 0`). Zero for `eta <= 0`.
#[inline]
pub fn contact_force_elastic(eta: f64, stiffness: f64, alpha: f64, path: PowPath) -> f64 {
    let ep = eta.max(0.0);
    stiffness * path.pow(ep, alpha)
}

/// Contact stiffness `phi''(eta) = K a [eta]+^(a-1)`.
///
/// The `ep > 0` guard is the original's `np.where`, and it is load-bearing at `a = 1`, where
/// `[eta]+^0` is `1` even at `eta = 0` and would otherwise leak a stiffness into the open gap.
#[inline]
pub fn contact_stiffness(eta: f64, stiffness: f64, alpha: f64, path: PowPath) -> f64 {
    let ep = eta.max(0.0);
    if ep > 0.0 {
        stiffness * alpha * path.pow(ep, alpha - 1.0)
    } else {
        0.0
    }
}

/// Energy-conserving **discrete-gradient** contact force.
#[inline]
pub fn contact_force_dg(
    eta_next: f64,
    eta_prev: f64,
    stiffness: f64,
    alpha: f64,
    tol: f64,
    path: PowPath,
) -> f64 {
    let da = eta_next - eta_prev;
    if da.abs() < tol {
        return contact_force_elastic(0.5 * (eta_next + eta_prev), stiffness, alpha, path);
    }
    (contact_potential(eta_next, stiffness, alpha, path)
        - contact_potential(eta_prev, stiffness, alpha, path))
        / da
}

/// `d/d eta+` of [`contact_force_dg`].
#[inline]
pub fn contact_force_dg_deriv(
    eta_next: f64,
    eta_prev: f64,
    stiffness: f64,
    alpha: f64,
    tol: f64,
    path: PowPath,
) -> f64 {
    let da = eta_next - eta_prev;
    if da.abs() < tol {
        return 0.5 * contact_stiffness(0.5 * (eta_next + eta_prev), stiffness, alpha, path);
    }
    let fe = contact_force_elastic(eta_next, stiffness, alpha, path);
    let phi_next = contact_potential(eta_next, stiffness, alpha, path);
    let phi_prev = contact_potential(eta_prev, stiffness, alpha, path);
    (fe * da - (phi_next - phi_prev)) / (da * da)
}

/// Hunt-Crossley/Stulov hysteretic force `lam_h [eta]+^a delta_t eta`; exactly zero at `lam_h = 0`.
#[inline]
pub fn contact_force_hyst(
    eta_next: f64,
    eta_prev: f64,
    alpha: f64,
    lam_h: f64,
    k: f64,
    path: PowPath,
) -> f64 {
    if lam_h == 0.0 {
        return 0.0;
    }
    let mid = (0.5 * (eta_next + eta_prev)).max(0.0);
    let w = if mid > 0.0 { path.pow(mid, alpha) } else { 0.0 };
    lam_h * w * (eta_next - eta_prev) / (2.0 * k)
}

/// `d/d eta+` of [`contact_force_hyst`].
#[inline]
pub fn contact_force_hyst_deriv(
    eta_next: f64,
    eta_prev: f64,
    alpha: f64,
    lam_h: f64,
    k: f64,
    path: PowPath,
) -> f64 {
    if lam_h == 0.0 {
        return 0.0;
    }
    let mid = (0.5 * (eta_next + eta_prev)).max(0.0);
    if mid <= 0.0 {
        return 0.0;
    }
    let w = path.pow(mid, alpha);
    let wp = 0.5 * alpha * path.pow(mid, alpha - 1.0);
    lam_h / (2.0 * k) * (wp * (eta_next - eta_prev) + w)
}

/// Total felt/barrier force: elastic discrete gradient plus hysteretic damping.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn contact_force_total(
    eta_next: f64,
    eta_prev: f64,
    stiffness: f64,
    alpha: f64,
    lam_h: f64,
    k: f64,
    tol: f64,
    path: PowPath,
) -> f64 {
    contact_force_dg(eta_next, eta_prev, stiffness, alpha, tol, path)
        + contact_force_hyst(eta_next, eta_prev, alpha, lam_h, k, path)
}

/// `d/d eta+` of [`contact_force_total`].
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn contact_force_total_deriv(
    eta_next: f64,
    eta_prev: f64,
    stiffness: f64,
    alpha: f64,
    lam_h: f64,
    k: f64,
    tol: f64,
    path: PowPath,
) -> f64 {
    contact_force_dg_deriv(eta_next, eta_prev, stiffness, alpha, tol, path)
        + contact_force_hyst_deriv(eta_next, eta_prev, alpha, lam_h, k, path)
}

// -- the vector forms ---------------------------------------------------------------------------
//
// These are NOT loops over the scalar functions above, even though they compute the same formula.
// The original writes them as separate NumPy expressions, so they take the array power path (see
// the module header), and the branch structure differs too: `np.where` evaluates both arms and
// selects, which matters where one arm would divide by zero. `safe` is the original's guard for
// exactly that, reproduced rather than replaced by an `if`, so that a NaN arriving in `da`
// propagates the way it does in Python.

/// Elementwise total force on a vector of penetrations — `_force_total_vec`.
#[allow(clippy::too_many_arguments)]
pub fn force_total_vec(
    eta_next: &[f64],
    eta_prev: &[f64],
    out: &mut [f64],
    stiffness: f64,
    alpha: f64,
    lam_h: f64,
    k: f64,
    tol: f64,
) {
    let p = PowPath::Array;
    for i in 0..out.len() {
        let (en, ep) = (eta_next[i], eta_prev[i]);
        let da = en - ep;
        let small = da.abs() < tol;
        let safe = if small { 1.0 } else { da };
        let mut dg = if small {
            contact_force_elastic(0.5 * (en + ep), stiffness, alpha, p)
        } else {
            (contact_potential(en, stiffness, alpha, p)
                - contact_potential(ep, stiffness, alpha, p))
                / safe
        };
        if lam_h != 0.0 {
            let mid = (0.5 * (en + ep)).max(0.0);
            let w = if mid > 0.0 { p.pow(mid, alpha) } else { 0.0 };
            dg += lam_h * w * da / (2.0 * k);
        }
        out[i] = dg;
    }
}

/// Elementwise `dF/d eta+` on a vector — `_deriv_total_vec`, the Jacobian's diagonal block.
#[allow(clippy::too_many_arguments)]
pub fn deriv_total_vec(
    eta_next: &[f64],
    eta_prev: &[f64],
    out: &mut [f64],
    stiffness: f64,
    alpha: f64,
    lam_h: f64,
    k: f64,
    tol: f64,
) {
    let p = PowPath::Array;
    for i in 0..out.len() {
        let (en, ep) = (eta_next[i], eta_prev[i]);
        let da = en - ep;
        let small = da.abs() < tol;
        let safe = if small { 1.0 } else { da };
        let fe = contact_force_elastic(en, stiffness, alpha, p);
        let dphi =
            contact_potential(en, stiffness, alpha, p) - contact_potential(ep, stiffness, alpha, p);
        let mut d = if small {
            0.5 * contact_stiffness(0.5 * (en + ep), stiffness, alpha, p)
        } else {
            (fe * da - dphi) / (safe * safe)
        };
        if lam_h != 0.0 {
            let mid = (0.5 * (en + ep)).max(0.0);
            let pos = mid > 0.0;
            let w = if pos { p.pow(mid, alpha) } else { 0.0 };
            let wp = if pos {
                0.5 * alpha * p.pow(mid, alpha - 1.0)
            } else {
                0.0
            };
            d += lam_h / (2.0 * k) * (wp * da + w);
        }
        out[i] = d;
    }
}

/// The parameters shared by every evaluation inside one contact solve.
#[derive(Debug, Clone, Copy)]
pub struct ContactParams {
    /// Contact stiffness `K`.
    pub stiffness: f64,
    /// Contact exponent `a`.
    pub alpha: f64,
    /// Hunt-Crossley damping `lam_h`.
    pub lam_h: f64,
    /// Timestep `k`.
    pub k: f64,
    /// Discrete-gradient Taylor-branch threshold.
    pub tol: f64,
}

// -- the scalar solve (the mallet's) --------------------------------------------------------------

/// Why a scalar contact solve failed outright.
#[derive(Debug, Clone, PartialEq)]
pub enum ContactError {
    /// The bracket scan widened six times without finding a sign change. The original raises a
    /// `RuntimeError` with this text and calls it impossible for a monotone convex force.
    NoRoot,
    /// `brentq` itself refused; carries its own reason.
    Root(RootError),
}

impl std::fmt::Display for ContactError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ContactError::NoRoot => write!(
                f,
                "contact residual has no root in the bracket (should be impossible for the \
                 monotone convex-potential force)."
            ),
            ContactError::Root(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for ContactError {}

/// The scalar solve's answer: penetration, the force applied at it, and whether the bracketed
/// fallback was used.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ContactSolution {
    pub eta: f64,
    pub force: f64,
    pub used_fallback: bool,
}

/// `np.linspace(start, stop, num)`, transcribed.
///
/// Not `start + i * (stop - start) / (num - 1)`: NumPy forms the step once, multiplies, adds the
/// start, and then **overwrites the last element with `stop`** — so the endpoint is exact even
/// where the arithmetic would not have been. The bracket scan below compares neighbouring signs,
/// so an endpoint differing in the last bit is a different bracket and, one `brentq` later, a
/// different root.
///
/// Public because [`crate::bow`] runs the *same* scan-and-bracket algorithm over its own residual
/// and needs the same grid. The crate open-codes this NumPy spelling in half a dozen other places
/// (each model's `grid()`, `membrane::linspace_from_zero`, `ops2d::grid_coords`) and that stays as
/// it is — those are one-off coordinate axes. This one is shared because a second copy would be a
/// second thing to keep in step at the exact point where drifting changes which roots exist.
pub fn linspace(start: f64, stop: f64, num: usize) -> Vec<f64> {
    let div = (num - 1) as f64;
    let step = (stop - start) / div;
    let mut y: Vec<f64> = (0..num).map(|i| (i as f64) * step + start).collect();
    if num > 1 {
        y[num - 1] = stop;
    }
    y
}

/// Solve the **scalar** contact equation `eta = eta_free - g f(eta)` for `eta^{n+1}`.
///
/// Safeguarded Newton seeded from the previous step's penetration (continuation), falling back to
/// a scanned bracket plus `brentq`, picking the root nearest the seed. **The fallback is not
/// hypothetical** — measured on 2026-08-27 over the mallet's own fixtures it fires once per 3,000
/// steps in the flagship case and eight times at `a = 1`, which is why the scan is transcribed at
/// its exact grid size rather than approximated.
pub fn solve_contact(
    eta_free: f64,
    eta_prev: f64,
    g: f64,
    p: ContactParams,
    seed: f64,
    newton_tol: f64,
    maxiter: usize,
) -> Result<ContactSolution, ContactError> {
    let resid = |eta: f64| -> (f64, f64) {
        let f = contact_force_total(
            eta,
            eta_prev,
            p.stiffness,
            p.alpha,
            p.lam_h,
            p.k,
            p.tol,
            PowPath::Scalar,
        );
        (eta - eta_free + g * f, f)
    };

    let mut eta = seed;
    let (mut r, mut f) = resid(eta);
    for _ in 0..maxiter {
        if r.abs() <= newton_tol {
            return Ok(ContactSolution {
                eta,
                force: f,
                used_fallback: false,
            });
        }
        let rp = 1.0
            + g * contact_force_total_deriv(
                eta,
                eta_prev,
                p.stiffness,
                p.alpha,
                p.lam_h,
                p.k,
                p.tol,
                PowPath::Scalar,
            );
        if rp.abs() < 1e-30 {
            break;
        }
        let eta_new = eta - r / rp;
        let (r_new, f_new) = resid(eta_new);
        // `not (|r_new| < |r|)` in the original — written as a negation so that a NaN residual,
        // which compares false against everything, breaks out to the bracket instead of being
        // accepted as progress. Kept in that spelling for the same reason, which is also why
        // clippy's `>=` rewrite is refused here rather than applied.
        #[allow(clippy::neg_cmp_op_on_partial_ord)]
        if !(r_new.abs() < r.abs()) {
            break;
        }
        eta = eta_new;
        r = r_new;
        f = f_new;
    }
    if r.abs() <= newton_tol {
        return Ok(ContactSolution {
            eta,
            force: f,
            used_fallback: false,
        });
    }

    // Bracketed fallback: scan a band around eta_free for sign changes, brentq each, nearest seed.
    let f_free = contact_force_total(
        eta_free,
        eta_prev,
        p.stiffness,
        p.alpha,
        p.lam_h,
        p.k,
        p.tol,
        PowPath::Scalar,
    );
    let mut span = (g * f_free).abs() + (eta_free - eta_prev).abs() + 1e-12;
    for _ in 0..6 {
        let vs = linspace(eta_free - span, eta_free + span, 1025);
        let rs: Vec<f64> = vs.iter().map(|&v| resid(v).0).collect();
        let mut best: Option<f64> = None;
        let mut best_dist = f64::INFINITY;
        for j in 0..(vs.len() - 1) {
            if rs[j] * rs[j + 1] < 0.0 {
                let root = brentq(|e| resid(e).0, vs[j], vs[j + 1], 1e-15, 8.9e-16, 100)
                    .map_err(ContactError::Root)?;
                // `np.argmin` keeps the FIRST minimum, so a later root wins only on a strict
                // improvement.
                let d = (root - seed).abs();
                if d < best_dist {
                    best_dist = d;
                    best = Some(root);
                }
            }
        }
        if let Some(eta_b) = best {
            return Ok(ContactSolution {
                eta: eta_b,
                force: resid(eta_b).1,
                used_fallback: true,
            });
        }
        span *= 10.0;
    }
    Err(ContactError::NoRoot)
}

// -- the vector solve (the barrier's) -------------------------------------------------------------

/// The vector solve's answer.
#[derive(Debug, Clone)]
pub struct VectorContactSolution {
    /// Penetration on the contact support.
    pub eta: Vec<f64>,
    /// The force density applied at it — exact only *at* the root, which is what the energy
    /// balance depends on.
    pub force: Vec<f64>,
    /// Completed Newton steps, or `maxiter` if the cap was reached.
    pub iters: usize,
    /// `max|r|` at the returned `eta`. The caller warns on this rather than failing, because a
    /// long render finishing with a drifting energy beats one that does not finish at all.
    pub residual: f64,
    /// Whether `residual <= newton_tol`.
    pub converged: bool,
}

/// Solve the **vector** contact system `eta = eta_free - G F(eta)` over the contact nodes.
///
/// `g_mat` is the dense `m x m` SPD admittance block, row-major. Damped Newton with an Armijo
/// line-search on `0.5 ||r||^2`; `J = I + G diag(F')` has `lambda_min >= 1`, so the root is unique
/// and Newton converges globally.
///
/// Two details of the original are reproduced deliberately, because a "sensible" version of either
/// would move the trajectory:
///
/// * **The Armijo test is a branch over two reductions.** `r . r` is a BLAS `ddot` in Python and a
///   plain left-to-right sum here, so the two sides of `<` can differ in the last bit — and a flip
///   does not perturb the answer by an ulp, it halves the step and changes the iterate by `O(1)`.
///   This is §13.3's "a branch choice is part of the trajectory" with a reduction behind it, and
///   it is the sharpest thing in this batch.
/// * **The line search has no failure exit.** If all 40 backtracks are rejected the loop simply
///   ends with `t = 2^-40` and the step is taken anyway. That is the reference behaviour.
#[allow(clippy::too_many_arguments)]
pub fn solve_contact_vector(
    eta_free: &[f64],
    eta_prev: &[f64],
    g_mat: &[f64],
    p: ContactParams,
    seed: &[f64],
    newton_tol: f64,
    maxiter: usize,
) -> VectorContactSolution {
    let m = eta_free.len();
    let mut work = vec![0.0f64; m];
    let mut eta: Vec<f64> = seed.to_vec();
    let mut r = vec![0.0f64; m];
    let mut eta_try = vec![0.0f64; m];
    let mut r_try = vec![0.0f64; m];
    let mut fp = vec![0.0f64; m];
    let mut jac = vec![0.0f64; m * m];
    let mut neg_r = vec![0.0f64; m];

    // `G @ F`, one row at a time, summed left to right. NumPy dispatches this to `dgemv`; see the
    // module header for why matching it is not on the table.
    let residual_into = |eta: &[f64], work: &mut [f64], out: &mut [f64]| {
        force_total_vec(
            eta,
            eta_prev,
            work,
            p.stiffness,
            p.alpha,
            p.lam_h,
            p.k,
            p.tol,
        );
        for i in 0..m {
            let row = &g_mat[i * m..i * m + m];
            let mut acc = 0.0;
            for j in 0..m {
                acc += row[j] * work[j];
            }
            out[i] = eta[i] - eta_free[i] + acc;
        }
    };
    // `np.max(np.abs(r))` — and NaN *propagates* there, where `f64::max` would quietly discard it
    // and report a converged solve on a diverged state. The one comparison this feeds is the
    // convergence test, so getting it wrong is the difference between a warning and silence.
    let max_abs = |v: &[f64]| {
        let mut best = f64::NEG_INFINITY;
        for &x in v {
            let a = x.abs();
            if a.is_nan() {
                return f64::NAN;
            }
            if a > best {
                best = a;
            }
        }
        best
    };
    let sq_norm = |v: &[f64]| {
        let mut a = 0.0;
        for &x in v {
            a += x * x;
        }
        a
    };

    residual_into(&eta, &mut work, &mut r);
    for it in 0..maxiter {
        let rmax = max_abs(&r);
        if rmax <= newton_tol {
            force_total_vec(
                &eta,
                eta_prev,
                &mut work,
                p.stiffness,
                p.alpha,
                p.lam_h,
                p.k,
                p.tol,
            );
            return VectorContactSolution {
                eta,
                force: work,
                iters: it,
                residual: rmax,
                converged: true,
            };
        }
        deriv_total_vec(
            &eta,
            eta_prev,
            &mut fp,
            p.stiffness,
            p.alpha,
            p.lam_h,
            p.k,
            p.tol,
        );
        // jac = I + G * fp[newaxis, :] — column j of G scaled by F'_j.
        for i in 0..m {
            for j in 0..m {
                let base = if i == j { 1.0 } else { 0.0 };
                jac[i * m + j] = base + g_mat[i * m + j] * fp[j];
            }
        }
        for i in 0..m {
            neg_r[i] = -r[i];
        }
        // `lu_factor` consumes its matrix the way LAPACK does (in place), so the allocation is
        // handed over and taken back rather than cloned — every entry is rewritten above next
        // iteration anyway.
        let fac = dense::lu_factor(std::mem::take(&mut jac), m)
            .expect("the Jacobian is square by construction");
        let delta = dense::lu_solve(&fac, &neg_r).expect("the right-hand side matches the factor");
        jac = fac.lu;

        let f0 = 0.5 * sq_norm(&r);
        let mut t = 1.0;
        for _ls in 0..40 {
            for i in 0..m {
                eta_try[i] = eta[i] + t * delta[i];
            }
            residual_into(&eta_try, &mut work, &mut r_try);
            if 0.5 * sq_norm(&r_try) < (1.0 - 1e-4 * t) * f0 {
                break;
            }
            t *= 0.5;
        }
        for i in 0..m {
            eta[i] += t * delta[i];
        }
        residual_into(&eta, &mut work, &mut r);
    }
    let rmax = max_abs(&r);
    force_total_vec(
        &eta,
        eta_prev,
        &mut work,
        p.stiffness,
        p.alpha,
        p.lam_h,
        p.k,
        p.tol,
    );
    VectorContactSolution {
        eta,
        force: work,
        iters: maxiter,
        residual: rmax,
        converged: rmax <= newton_tol,
    }
}

// -- the distributed barrier (model #8) -----------------------------------------------------------

/// Why a barrier string was refused at construction.
///
/// The order of the variants is the order `BarrierString.__init__` checks them in; a call that is
/// wrong in more than one way must report the same fault Python would.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BarrierError {
    /// `stiffness <= 0`.
    NonPositiveStiffness,
    /// `alpha < 1`.
    AlphaTooSmall,
    /// `hysteresis < 0`.
    NegativeHysteresis,
    /// No interior node has a finite barrier height.
    EmptySupport,
}

impl std::fmt::Display for BarrierError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BarrierError::NonPositiveStiffness => write!(f, "contact stiffness K must be > 0."),
            BarrierError::AlphaTooSmall => write!(f, "contact exponent alpha must be >= 1."),
            BarrierError::NegativeHysteresis => write!(f, "hysteresis lambda_h must be >= 0."),
            BarrierError::EmptySupport => write!(
                f,
                "barrier has no finite interior node -> empty contact support."
            ),
        }
    }
}

impl std::error::Error for BarrierError {}

/// Everything a barrier string derives from its string once, at construction.
///
/// The admittance block is the expensive half: `m` banded solves against the string's factor,
/// which is why it is built here and never again. `g_mat` and `force_pref` are `pub` and meant to
/// be written — `tests/test_collision_modal.py` doubles both to move the fixed point, which is the
/// negative control for the coupling magnitude.
#[derive(Debug, Clone)]
pub struct BarrierParams {
    /// Stiffness, exponent, hysteresis, timestep and Taylor threshold, as the solve wants them.
    pub contact: ContactParams,
    /// Newton convergence tolerance on `max|r|`.
    pub newton_tol: f64,
    /// Newton iteration cap.
    pub newton_maxiter: usize,
    /// `k^2 / rho` — a force **density** prefactor, not `k^2/(rho h)`.
    pub force_pref: f64,
    /// The string's grid spacing, for the barrier's potential energy `h * sum phi`.
    pub h: f64,
    /// `N + 1`, the full grid.
    pub nodes: usize,
    /// Grid node indices carrying a finite barrier, ascending — the contact support.
    pub support: Vec<usize>,
    /// Barrier heights on the support, same order.
    pub b: Vec<f64>,
    /// The support's indices into the *interior* array (`support[j] - 1`).
    pub int_idx: Vec<usize>,
    /// `A^{-1} e_support[j]` as column `j`, row-major `(N-1) x m` — the rank-`m` correction.
    pub cols_mat: Vec<f64>,
    /// `G = force_pref * (A^{-1})_{support,support}`, row-major `m x m`.
    pub g_mat: Vec<f64>,
}

impl BarrierParams {
    /// Validate, pick the support, and build the admittance block.
    ///
    /// `barrier_full` is the profile already broadcast onto the `N + 1` grid — the binding does
    /// that, because "a scalar is a flat rail" is a NumPy broadcast in the original and its
    /// failure text belongs to NumPy rather than here.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        sp: &string_damped::Params,
        barrier_full: &[f64],
        stiffness: f64,
        alpha: f64,
        hysteresis: f64,
        eta_tol: f64,
        newton_tol: f64,
        newton_maxiter: i64,
    ) -> Result<Self, BarrierError> {
        if stiffness <= 0.0 {
            return Err(BarrierError::NonPositiveStiffness);
        }
        if alpha < 1.0 {
            return Err(BarrierError::AlphaTooSmall);
        }
        if hysteresis < 0.0 {
            return Err(BarrierError::NegativeHysteresis);
        }

        let nodes = sp.nodes();
        // The support is the *interior* nodes 1..N-1 whose barrier height is finite. `is_finite`
        // is `np.isfinite`: a NaN is excluded as well as an infinity, which is what makes `-inf`
        // the spelling for "no barrier here" and a point fret one finite entry.
        let support: Vec<usize> = (1..nodes - 1)
            .filter(|&i| barrier_full[i].is_finite())
            .collect();
        if support.is_empty() {
            return Err(BarrierError::EmptySupport);
        }
        let b: Vec<f64> = support.iter().map(|&i| barrier_full[i]).collect();
        let int_idx: Vec<usize> = support.iter().map(|&i| i - 1).collect();

        // `k ** 2`, not `k * k`. The original writes `string.k ** 2 / string.rho`, which is
        // `float.__pow__` and therefore the C library's `pow` — a different double from the
        // multiply in 79 of 200,007 samples (§16.2). `bow.py` writes `self.k * self.k` at the
        // same spot, which is why that model's port uses a multiply and this one must not.
        let force_pref = scalar_pow(sp.k, 2.0) / sp.rho;

        let interior = sp.interior();
        let m = support.len();
        let mut cols_mat = vec![0.0f64; interior * m];
        for (j, &node) in support.iter().enumerate() {
            let mut e = vec![0.0f64; interior];
            e[node - 1] = 1.0;
            let col = string_damped::apply_ainv(&e, sp);
            for i in 0..interior {
                cols_mat[i * m + j] = col[i];
            }
        }
        let mut g_mat = vec![0.0f64; m * m];
        for (a, &row) in int_idx.iter().enumerate() {
            for bcol in 0..m {
                g_mat[a * m + bcol] = force_pref * cols_mat[row * m + bcol];
            }
        }

        Ok(BarrierParams {
            contact: ContactParams {
                stiffness,
                alpha,
                lam_h: hysteresis,
                k: sp.k,
                tol: eta_tol,
            },
            newton_tol,
            // `int(newton_maxiter)` reaches a `range()` in the original, so a negative cap means
            // "no Newton iterations", not an error — the mallet's binding makes the same reading.
            newton_maxiter: newton_maxiter.max(0) as usize,
            force_pref,
            h: sp.h,
            nodes,
            support,
            b,
            int_idx,
            cols_mat,
            g_mat,
        })
    }

    /// How many nodes are in the contact support.
    pub fn support_len(&self) -> usize {
        self.support.len()
    }
}

/// The barrier's own state — the string holds the field, this holds the contact.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct BarrierState {
    /// Penetration `eta^n` on the support (positive in contact).
    pub penetration: Vec<f64>,
    /// Contact force density on the support for the last step (N/m).
    pub contact_force: Vec<f64>,
    /// Newton iterations the last step's solve took.
    pub newton_iters: usize,
    /// Completed steps.
    pub n: usize,
}

/// `b - u[support]` — the penetration the barrier reads off a displacement field.
pub fn penetration_of(p: &BarrierParams, u_full: &[f64], out: &mut [f64]) {
    for (j, &node) in p.support.iter().enumerate() {
        out[j] = p.b[j] - u_full[node];
    }
}

/// Solve the contact and apply the rank-`m` correction to `u_full`, in place.
///
/// `eta_free` is the penetration after the string's force-free advance; `eta_prev` is the
/// penetration at `u^{n-1}`, which must have been read *before* that advance rolled the history.
/// The seed is the previous step's penetration — continuation, and the reason a converged solve
/// usually costs two or three iterations rather than ten.
///
/// The correction is written as `u[i] = u[i] + force_pref * acc_i`, one row at a time and summed
/// left to right, which is not `cols_mat @ f`'s order. See the module header for the measurement
/// that says where that is visible and where it is not.
///
/// Returns the solve's residual and whether it converged, so the caller can raise the original's
/// warning. Nothing here decides that: a stall is a `UserWarning` in Python and the frame it is
/// attributed to is a property of the binding, not of the arithmetic.
pub fn apply(
    u_full: &mut [f64],
    eta_free: &[f64],
    eta_prev: &[f64],
    s: &mut BarrierState,
    p: &BarrierParams,
) -> (f64, bool) {
    let m = p.support.len();
    let sol = solve_contact_vector(
        eta_free,
        eta_prev,
        &p.g_mat,
        p.contact,
        &s.penetration,
        p.newton_tol,
        p.newton_maxiter,
    );

    let interior = p.nodes - 2;
    for i in 0..interior {
        let row = &p.cols_mat[i * m..i * m + m];
        // `zip` rather than an index, and the order is the whole point: this accumulates strictly
        // left to right, which is what §23.2's measurement is a measurement OF. Anything that
        // reassociated it would be a third spelling of the same sum.
        let mut acc = 0.0;
        for (&a, &f) in row.iter().zip(sol.force.iter()) {
            acc += a * f;
        }
        u_full[1 + i] += p.force_pref * acc;
    }

    s.penetration = sol.eta;
    s.contact_force = sol.force;
    s.newton_iters = sol.iters;
    s.n += 1;
    (sol.residual, sol.converged)
}

/// The barrier's stored potential energy, `h * sum_j 1/2 (phi(eta^n_j) + phi(eta^{n-1}_j))`.
///
/// The **two-time average** is the form that telescopes with the discrete-gradient force, so this
/// is what makes the coupled energy conserve rather than merely stay bounded. Both sums accumulate
/// left to right; `np.sum` is pairwise above eight elements, so at the 79-node rail this is a
/// different last bit from the original's — measured at 236 of 2,000 steps, on a read-out that
/// reaches no state.
pub fn barrier_energy(p: &BarrierParams, u_full: &[f64], u_prev_full: &[f64]) -> f64 {
    let path = PowPath::Array;
    let mut sum_n = 0.0;
    let mut sum_p = 0.0;
    for (j, &node) in p.support.iter().enumerate() {
        sum_n += contact_potential(
            p.b[j] - u_full[node],
            p.contact.stiffness,
            p.contact.alpha,
            path,
        );
        sum_p += contact_potential(
            p.b[j] - u_prev_full[node],
            p.contact.stiffness,
            p.contact.alpha,
            path,
        );
    }
    0.5 * p.h * (sum_n + sum_p)
}

// -- the native owning struct ------------------------------------------------------------------

/// A damped stiff string vibrating against a one-sided distributed barrier — model #8.
///
/// For Rust callers and for `cargo test`; the binding holds a Python-owned string instead, for the
/// reason `bow`'s does.
#[derive(Debug, Clone)]
pub struct BarrierString {
    /// The resonator.
    pub string: string_damped::DampedStiffString,
    /// The barrier's constants and admittance block.
    pub p: BarrierParams,
    /// The barrier's contact state.
    pub s: BarrierState,
}

impl BarrierString {
    /// Build a barrier on `string`, solving the `m` admittance columns once.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        string: string_damped::DampedStiffString,
        barrier_full: &[f64],
        stiffness: f64,
        alpha: f64,
        hysteresis: f64,
        eta_tol: f64,
        newton_tol: f64,
        newton_maxiter: i64,
    ) -> Result<Self, BarrierError> {
        let p = BarrierParams::new(
            &string.p,
            barrier_full,
            stiffness,
            alpha,
            hysteresis,
            eta_tol,
            newton_tol,
            newton_maxiter,
        )?;
        let m = p.support_len();
        let mut s = BarrierState {
            penetration: vec![0.0; m],
            contact_force: vec![0.0; m],
            newton_iters: 0,
            n: 0,
        };
        penetration_of(&p, &string.u, &mut s.penetration);
        Ok(BarrierString { string, p, s })
    }

    /// Set the string's state, then refresh the continuation seed.
    ///
    /// `contact_force` and `newton_iters` are deliberately *not* cleared: the original does not
    /// clear them either, and a fixture that reads a force before its first step must read the
    /// same stale value on both sides.
    pub fn set_state(&mut self, u0: &[f64], v0: &[f64]) {
        self.string.set_state(u0, v0);
        penetration_of(&self.p, &self.string.u, &mut self.s.penetration);
        self.s.n = 0;
    }

    /// Advance one step: force-free string advance, vector contact solve, exact force inject.
    pub fn step(&mut self) -> (f64, bool) {
        let m = self.p.support_len();
        let mut eta_prev = vec![0.0; m];
        let mut eta_free = vec![0.0; m];
        penetration_of(&self.p, &self.string.u_prev, &mut eta_prev);
        self.string.step();
        penetration_of(&self.p, &self.string.u, &mut eta_free);
        apply(
            &mut self.string.u,
            &eta_free,
            &eta_prev,
            &mut self.s,
            &self.p,
        )
    }

    /// Total discrete energy: the string's plus the barrier's stored potential.
    pub fn energy(&self) -> f64 {
        self.string.energy() + barrier_energy(&self.p, &self.string.u, &self.string.u_prev)
    }

    /// Which support nodes are currently in contact.
    pub fn contact_mask(&self) -> Vec<bool> {
        self.s.penetration.iter().map(|&e| e > 0.0).collect()
    }
}
