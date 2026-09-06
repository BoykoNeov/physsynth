//! Matrix-free restarted GMRES — the linear solver behind the von Karman plate's Newton step.
//!
//! Written here rather than inside `plate.rs` for the same reason [`crate::root`] is not inside
//! `string_nonlinear`: the plate should hand this a **closure** and a right-hand side and get a
//! correction back, so that what is asserted about the solver is asserted against a matrix whose
//! answer is known in closed form. `tests/krylov.rs` does exactly that, on hand-built systems,
//! before `plate.rs` is allowed to call it (plan §5 Part 2, and Part 1's discipline).
//!
//! # No preconditioner, and that is a fact about the operator rather than a shortcut
//!
//! The plate's Newton system is `J = I - c A^-1 K`, and `A^-1` — the theta-scheme's prefactored
//! solve — is already **inside** it. At zero amplitude `J = I` exactly, and on the fixtures Part 1
//! measured it sits 2.4%-12.4% away. So the obvious preconditioner is present by construction and
//! the batch needs no second factorization. `docs/dev/vk-newton-plan.md` §1.
//!
//! # This is a reduction, and there is nothing to be bit-identical to
//!
//! Every inner product below decides the iterate, so the arithmetic order is part of the
//! trajectory in exactly the sense §19.2 of the findings means. That is normally a hazard because
//! a Python twin computes the same sum through BLAS; here there is no twin — Newton is new physics
//! and this project's rule is that new physics is Rust-first (migration plan §6). The order is
//! therefore *chosen* (modified Gram-Schmidt, left to right) and pinned by these tests, not
//! matched against anything.

/// `a . b`, left to right — see the module header on why the order is ours to choose.
fn dot(a: &[f64], b: &[f64]) -> f64 {
    let mut s = 0.0;
    for (x, y) in a.iter().zip(b.iter()) {
        s += x * y;
    }
    s
}

/// `sqrt(x . x)`.
fn norm2(x: &[f64]) -> f64 {
    dot(x, x).sqrt()
}

/// What one [`gmres`] call did.
///
/// `products` is the count of operator applications, which is the only cost worth reporting: for
/// the plate each one is an Airy back-substitution plus a theta-scheme back-substitution, and
/// everything else in here is `O(n)` vector work.
#[derive(Debug, Clone, PartialEq)]
pub struct GmresReport {
    /// The approximate solution of `A x = b`, from a **zero** initial guess.
    pub x: Vec<f64>,
    /// Operator applications spent.
    pub products: usize,
    /// Did the relative residual reach `rtol`?
    pub converged: bool,
    /// `||b - A x|| / ||b||` at exit, from the Givens recursion.
    pub residual: f64,
}

/// Restarted GMRES(`restart`) on `A x = b`, `A` given only as a closure.
///
/// Modified Gram-Schmidt Arnoldi with Givens rotations, starting from `x = 0` — which is the
/// natural guess for a Newton correction, and which is why the first pass spends no product on
/// forming its own residual.
///
/// `restart` is a **pin, not a tuning knob** (plan §7 trap 4): it changes the iteration count and
/// the intermediate iterates, so a per-fixture choice would make the convergence map a map of the
/// choice. `tests/krylov.rs` asserts that it does not change the *converged* answer.
///
/// Stops when `||b - A x|| <= rtol * ||b||`, when `max_products` operator applications have been
/// spent, or on a happy breakdown (the Krylov space became invariant, so the answer is exact).
///
/// # Errors
/// Whatever the operator returns — for the plate, a factorization that cannot back-substitute.
pub fn gmres<E, F>(
    mut apply: F,
    b: &[f64],
    restart: usize,
    max_products: usize,
    rtol: f64,
) -> Result<GmresReport, E>
where
    F: FnMut(&[f64]) -> Result<Vec<f64>, E>,
{
    let n = b.len();
    let m = restart.max(1);
    let bnorm = norm2(b);
    let mut x = vec![0.0; n];
    // An exactly zero right-hand side is exactly solved by zero, and spending a product to
    // discover that would make a converged Newton step cost one more solve than it needs.
    if bnorm == 0.0 {
        return Ok(GmresReport {
            x,
            products: 0,
            converged: true,
            residual: 0.0,
        });
    }
    let target = rtol * bnorm;
    let mut products = 0usize;
    let mut resid = bnorm;
    let mut first = true;

    while resid > target && products < max_products {
        // `r = b - A x`. On the first pass `x` is still zero, so `r` is `b` and no product is due.
        let r: Vec<f64> = if first {
            first = false;
            b.to_vec()
        } else {
            let ax = apply(&x)?;
            products += 1;
            (0..n).map(|i| b[i] - ax[i]).collect()
        };
        let beta = norm2(&r);
        resid = beta;
        if beta <= target {
            break;
        }

        let mut v: Vec<Vec<f64>> = Vec::with_capacity(m + 1);
        v.push(r.iter().map(|&t| t / beta).collect());
        // `h[i][j]`, the Hessenberg matrix, overwritten in place by the rotations.
        let mut h = vec![vec![0.0f64; m]; m + 1];
        let (mut cs, mut sn) = (vec![0.0f64; m], vec![0.0f64; m]);
        let mut g = vec![0.0f64; m + 1];
        g[0] = beta;
        let mut built = 0usize;

        for j in 0..m {
            if products >= max_products {
                break;
            }
            let mut w = apply(&v[j])?;
            products += 1;
            // Modified Gram-Schmidt: subtract each projection as it is formed, so the next inner
            // product sees the already-orthogonalised vector. Classical MGS, not re-orthogonalised.
            for i in 0..=j {
                let hij = dot(&w, &v[i]);
                h[i][j] = hij;
                for l in 0..n {
                    w[l] -= hij * v[i][l];
                }
            }
            let hnext = norm2(&w);
            h[j + 1][j] = hnext;

            // Rotate the new column by the rotations already chosen, then choose this column's.
            for i in 0..j {
                let (a, c) = (h[i][j], h[i + 1][j]);
                h[i][j] = cs[i] * a + sn[i] * c;
                h[i + 1][j] = -sn[i] * a + cs[i] * c;
            }
            let (dj, ej) = (h[j][j], h[j + 1][j]);
            let denom = dj.hypot(ej);
            if denom == 0.0 {
                cs[j] = 1.0;
                sn[j] = 0.0;
            } else {
                cs[j] = dj / denom;
                sn[j] = ej / denom;
            }
            h[j][j] = cs[j] * dj + sn[j] * ej;
            h[j + 1][j] = 0.0;
            g[j + 1] = -sn[j] * g[j];
            // `*=` rather than `g[j] = cs[j] * g[j]` at clippy's asking; IEEE multiplication is
            // commutative, so this is the same double. The ORDER of the two lines is not
            // negotiable -- the first reads `g[j]` before the second overwrites it.
            g[j] *= cs[j];
            built = j + 1;
            resid = g[j + 1].abs();

            if hnext > 0.0 {
                v.push(w.iter().map(|&t| t / hnext).collect());
            }
            // A zero sub-diagonal is a *happy* breakdown: the Krylov space is invariant and the
            // least-squares solution over it is the exact one, so stopping here loses nothing.
            if resid <= target || hnext == 0.0 {
                break;
            }
        }

        if built == 0 {
            break;
        }
        // Back-substitute the `built x built` upper triangle, then correct `x` in the Arnoldi basis.
        let mut y = vec![0.0f64; built];
        for i in (0..built).rev() {
            let mut s = g[i];
            for (j, yj) in y.iter().enumerate().skip(i + 1) {
                s -= h[i][j] * yj;
            }
            y[i] = if h[i][i] == 0.0 { 0.0 } else { s / h[i][i] };
        }
        for (i, yi) in y.iter().enumerate() {
            for l in 0..n {
                x[l] += yi * v[i][l];
            }
        }
    }

    Ok(GmresReport {
        converged: resid <= target,
        residual: resid / bnorm,
        x,
        products,
    })
}
