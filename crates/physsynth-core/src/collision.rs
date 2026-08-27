//! Contact & collision primitives, and the two solves built on them (models #7 and #8).
//!
//! `docs/dev/rust-migration-plan.md` §16. The Python original, `physsynth/core/collision.py`, is
//! the shared home of the energy-conserving contact scheme: the mallet (`mallet.py`, model #7)
//! uses its **scalar** solve, the distributed barrier (`BarrierString`, model #8) its **vector**
//! one. This module ports the primitives and both solves. `BarrierString` itself does not port in
//! this batch — it wraps a `DampedStiffString`, which is still Python, so porting it would mean
//! building a model on a model that has not moved.
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

use crate::dense;
use crate::root::{brentq, RootError};

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
fn linspace(start: f64, stop: f64, num: usize) -> Vec<f64> {
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
