//! Symmetric dense eigenvalues — Householder tridiagonalization plus implicit-shift QL.
//!
//! Written for `connection`'s stability guard, which needs `lambda_max` of the coupled leapfrog
//! operator before a `SympatheticStrings` is allowed to exist. The Python original got that from
//! `np.linalg.eigvals` — LAPACK `dgeev` on the *unsymmetric* matrix — and this crate's
//! `[dependencies]` is empty and stays empty (`tests/deps.rs`), so the routine has to be here.
//!
//! # Why the symmetric routine is enough, which is a fact about the operator
//!
//! `A = M^-1 K` for the coupled system: `K` is the stiffness (string second differences, the
//! bridge springs, the modal restoring terms) and `M` the diagonal mass — `rho h` per interior
//! string node, `rho h / 2` at the free end that carries the spring, `m_i` per body mode. A
//! product of a symmetric matrix with a positive diagonal inverse is *similar* to the symmetric
//! `M^1/2 A M^-1/2`, so its spectrum is real, and the physics adds that it is non-negative: `K` is
//! a stiffness. Measured on a two-string fixture before this module was written, the symmetrized
//! matrix is symmetric to 3.6e-18 relative (i.e. to roundoff in the scaling multiply itself) and
//! every eigenvalue is real and positive. `connection::SympatheticStrings` therefore builds the
//! symmetrized matrix and calls [`symmetric_eigenvalues`]; `tests/connection.rs` asserts the
//! symmetry as a bar in its own right, because it is a claim about the discretization — the
//! coupled operator is self-adjoint in the energy inner product — and not an implementation
//! detail. If it ever stopped holding, this route would be wrong rather than merely inaccurate,
//! which is why it is asserted rather than assumed.
//!
//! An unsymmetric solver (Hessenberg plus Francis QR) would be three times this code and would
//! answer a question the operator does not pose.
//!
//! # Why not power iteration, which is what the guard's one number would seem to need
//!
//! The guard reads `lambda_max` alone, and the top of this spectrum is *clustered*: on the same
//! two-string fixture `lambda_2 / lambda_1 = 0.9969`, and two identical strings make the top
//! eigenvalue exactly degenerate by symmetry. Power iteration converges as that ratio to the
//! power of the iteration count — thousands of matrix-vector products for the last digits, with a
//! convergence test that has to be written on the Rayleigh quotient rather than the vector or a
//! degenerate pair never settles. The full tridiagonal reduction is `O(n^3)` once, at
//! construction, on `n = sum_j N_j + M` (204 for the suite's two-string fixture, 84 for the
//! reduced one used in the eigen bars): deterministic, no convergence criterion to get wrong, and
//! it returns the whole spectrum so a test can check it against a closed form rather than only its
//! largest entry.
//!
//! # Provenance
//!
//! The reduction is Householder's, in the EISPACK `tred2`/`tql1` arrangement that Numerical
//! Recipes §11.2-11.3 also uses: reduce to tridiagonal from the last row upward, then run QL with
//! Wilkinson shifts on the tridiagonal, deflating from the bottom. [`symmetric_eigenvalues`] does
//! not accumulate eigenvectors; [`symmetric_eigen`] does (EISPACK `tred2`/`tql2`), for the
//! Rayleigh-Ritz step of [`crate::eigs`]. The vector bookkeeping only writes the upper triangle
//! and rotates a separate matrix, so the eigenvalues the two routines return are the SAME doubles
//! — `tests/eig.rs` asserts that, since it is what lets the vector path inherit the value path's
//! bars. The arithmetic is ours to order (no NumPy twin), so it is written in the plain
//! left-to-right way this crate uses everywhere a reduction is not transcribed.

/// Why an eigenvalue computation was refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EigError {
    /// The data length is not `n * n`.
    BadShape,
    /// The QL sweep hit its iteration cap while deflating the given eigenvalue.
    ///
    /// Thirty sweeps per eigenvalue is EISPACK's cap and is not reachable by a well-formed
    /// symmetric matrix; a matrix carrying a NaN is the way it happens in practice.
    NotConverged(usize),
    /// A generalized problem's diagonal mass had a non-positive (or NaN) entry at this index.
    MassNotPositive(usize),
}

impl std::fmt::Display for EigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EigError::BadShape => write!(f, "matrix data must have exactly n * n entries."),
            EigError::NotConverged(i) => write!(
                f,
                "the QL iteration did not converge for eigenvalue {i} within {MAX_QL_SWEEPS} \
                 sweeps; the matrix is not a finite symmetric one."
            ),
            EigError::MassNotPositive(i) => write!(
                f,
                "the diagonal mass matrix must be positive definite; entry {i} is not positive."
            ),
        }
    }
}

impl std::error::Error for EigError {}

/// EISPACK's sweep cap, per eigenvalue.
const MAX_QL_SWEEPS: usize = 30;

/// Eigenvalues of a real symmetric matrix, ascending.
///
/// `a` is row-major `n * n`. Only the lower triangle and the diagonal are read, so a caller that
/// knows its matrix is symmetric need not have made the two halves agree to the last bit.
///
/// # Errors
/// [`EigError::BadShape`] if `a.len() != n * n`; [`EigError::NotConverged`] if the QL sweep hits
/// its cap, which a finite symmetric matrix does not do.
pub fn symmetric_eigenvalues(a: &[f64], n: usize) -> Result<Vec<f64>, EigError> {
    if a.len() != n * n {
        return Err(EigError::BadShape);
    }
    if n == 0 {
        return Ok(Vec::new());
    }
    let mut m = a.to_vec();
    let (mut d, mut e) = tridiagonalize(&mut m, n, false);
    ql(&mut d, &mut e, n, None)?;
    d.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
    Ok(d)
}

/// The largest eigenvalue of a real symmetric matrix.
///
/// A named wrapper because that is the whole of what a stability guard wants, and because
/// `.last()` on a sorted vector is the kind of line that reads as an accident at a call site.
///
/// # Errors
/// As [`symmetric_eigenvalues`], plus [`EigError::BadShape`] for `n == 0`, which has no largest
/// eigenvalue.
pub fn symmetric_max_eigenvalue(a: &[f64], n: usize) -> Result<f64, EigError> {
    if n == 0 {
        return Err(EigError::BadShape);
    }
    let values = symmetric_eigenvalues(a, n)?;
    Ok(values[n - 1])
}

/// Eigenvalues and eigenvectors of a real symmetric matrix, eigenvalues ascending.
///
/// Returns `(values, vectors)` with `vectors` row-major `n * n` and eigenvector `j` in COLUMN `j`,
/// orthonormal. Each vector's sign is whatever the QL rotations leave, which is arbitrary; so is
/// the basis inside a repeated eigenvalue. A caller that ships a vector has to be invariant to
/// both (retirement plan §23.11).
///
/// # Errors
/// As [`symmetric_eigenvalues`].
pub fn symmetric_eigen(a: &[f64], n: usize) -> Result<(Vec<f64>, Vec<f64>), EigError> {
    if a.len() != n * n {
        return Err(EigError::BadShape);
    }
    if n == 0 {
        return Ok((Vec::new(), Vec::new()));
    }
    let mut z = a.to_vec();
    let (mut d, mut e) = tridiagonalize(&mut z, n, true);
    ql(&mut d, &mut e, n, Some(&mut z))?;
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&x, &y| d[x].partial_cmp(&d[y]).unwrap_or(std::cmp::Ordering::Equal));
    let values: Vec<f64> = order.iter().map(|&j| d[j]).collect();
    let mut vectors = vec![0.0; n * n];
    for (jn, &jo) in order.iter().enumerate() {
        for i in 0..n {
            vectors[i * n + jn] = z[i * n + jo];
        }
    }
    Ok((values, vectors))
}

/// Eigenpairs of `A x = lambda D x` for a symmetric `A` and a positive DIAGONAL `D`, ascending.
///
/// Reduced to the symmetric standard problem `S = D^-1/2 A D^-1/2` and mapped back with
/// `x = D^-1/2 y`, so the vectors come out `D`-orthonormal (`V^T D V = I`) — the normalization
/// `scipy.linalg.eigh(A, B)` returns and the one a caller that PROJECTS under the mass needs.
/// Row-major `vectors`, eigenvector `j` in column `j`; sign and the basis inside a repeated
/// eigenvalue are arbitrary, as for [`symmetric_eigen`].
///
/// # Errors
/// [`EigError::BadShape`] if `a.len() != n * n` or `d.len() != n`;
/// [`EigError::MassNotPositive`] for a non-positive or NaN mass; otherwise as [`symmetric_eigen`].
pub fn generalized_eigen_diag(
    a: &[f64],
    d: &[f64],
    n: usize,
) -> Result<(Vec<f64>, Vec<f64>), EigError> {
    if a.len() != n * n || d.len() != n {
        return Err(EigError::BadShape);
    }
    if let Some(i) = d.iter().position(|&v| v <= 0.0 || v.is_nan()) {
        return Err(EigError::MassNotPositive(i));
    }
    let inv_sqrt: Vec<f64> = d.iter().map(|v| 1.0 / v.sqrt()).collect();
    let mut s = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            s[i * n + j] = a[i * n + j] * inv_sqrt[i] * inv_sqrt[j];
        }
    }
    let (values, mut vectors) = symmetric_eigen(&s, n)?;
    for i in 0..n {
        for j in 0..n {
            vectors[i * n + j] *= inv_sqrt[i];
        }
    }
    Ok((values, vectors))
}

/// Householder reduction to tridiagonal form, in place; returns `(diagonal, subdiagonal)`.
///
/// `e[0]` is unused and returned as zero, which is the arrangement the QL step below expects.
/// With `vectors` false the transformations are applied to the trailing submatrix and then
/// discarded. With `vectors` true (`tred2`) each Householder vector is also kept in the upper
/// triangle, and `a` comes back holding the accumulated orthogonal transform — the starting
/// point the QL rotations are applied to.
fn tridiagonalize(a: &mut [f64], n: usize, vectors: bool) -> (Vec<f64>, Vec<f64>) {
    let mut d = vec![0.0; n];
    let mut e = vec![0.0; n];
    let at = |i: usize, j: usize| i * n + j;

    for i in (1..n).rev() {
        let l = i - 1;
        let mut h = 0.0;
        if l > 0 {
            let mut scale = 0.0;
            for k in 0..=l {
                scale += a[at(i, k)].abs();
            }
            if scale == 0.0 {
                // Already zero across the row: nothing to reflect away.
                e[i] = a[at(i, l)];
            } else {
                for k in 0..=l {
                    a[at(i, k)] /= scale;
                    h += a[at(i, k)] * a[at(i, k)];
                }
                let mut f = a[at(i, l)];
                let g = if f >= 0.0 { -h.sqrt() } else { h.sqrt() };
                e[i] = scale * g;
                h -= f * g;
                a[at(i, l)] = f - g;
                f = 0.0;
                for j in 0..=l {
                    if vectors {
                        // The upper triangle: never read by the reduction, which reads the lower.
                        a[at(j, i)] = a[at(i, j)] / h;
                    }
                    let mut g = 0.0;
                    for k in 0..=j {
                        g += a[at(j, k)] * a[at(i, k)];
                    }
                    for k in (j + 1)..=l {
                        g += a[at(k, j)] * a[at(i, k)];
                    }
                    e[j] = g / h;
                    f += e[j] * a[at(i, j)];
                }
                let hh = f / (h + h);
                for j in 0..=l {
                    let f = a[at(i, j)];
                    let g = e[j] - hh * f;
                    e[j] = g;
                    for k in 0..=j {
                        a[at(j, k)] -= f * e[k] + g * a[at(i, k)];
                    }
                }
            }
        } else {
            e[i] = a[at(i, l)];
        }
        d[i] = h;
    }

    e[0] = 0.0;
    if !vectors {
        for i in 0..n {
            d[i] = a[at(i, i)];
        }
        return (d, e);
    }
    // Accumulate the transform. `d[i]` still holds that step's `h`, and a zero `h` means the step
    // reflected nothing, so there is nothing to apply.
    d[0] = 0.0;
    for i in 0..n {
        if d[i] != 0.0 {
            for j in 0..i {
                let mut g = 0.0;
                for k in 0..i {
                    g += a[at(i, k)] * a[at(k, j)];
                }
                for k in 0..i {
                    a[at(k, j)] -= g * a[at(k, i)];
                }
            }
        }
        d[i] = a[at(i, i)];
        a[at(i, i)] = 1.0;
        for j in 0..i {
            a[at(j, i)] = 0.0;
            a[at(i, j)] = 0.0;
        }
    }
    (d, e)
}

/// `sqrt(x^2 + y^2)` without spurious overflow — the QL step's one guarded expression.
fn hypot2(x: f64, y: f64) -> f64 {
    let ax = x.abs();
    let ay = y.abs();
    if ax > ay {
        let r = ay / ax;
        ax * (1.0 + r * r).sqrt()
    } else if ay == 0.0 {
        0.0
    } else {
        let r = ax / ay;
        ay * (1.0 + r * r).sqrt()
    }
}

/// `|a|` with the sign of `b` — Fortran's `SIGN`, which the shift formula is written in terms of.
fn sign(a: f64, b: f64) -> f64 {
    if b >= 0.0 {
        a.abs()
    } else {
        -a.abs()
    }
}

/// QL with implicit Wilkinson shifts on the tridiagonal `(d, e)`.
///
/// `d` comes back holding the eigenvalues in no particular order; the caller sorts. When `z` is
/// given (row-major `n * n`), every rotation is applied to its columns as well (`tql2`); the
/// rotations never read `z` into `d` or `e`, so the eigenvalues are the same with or without it.
fn ql(d: &mut [f64], e: &mut [f64], n: usize, mut z: Option<&mut [f64]>) -> Result<(), EigError> {
    if n == 1 {
        return Ok(());
    }
    // Shift the subdiagonal down one, so `e[i]` sits beside `d[i]`.
    for i in 1..n {
        e[i - 1] = e[i];
    }
    e[n - 1] = 0.0;

    for l in 0..n {
        let mut iter = 0usize;
        loop {
            // Look for a single small subdiagonal element to split the matrix at.
            let mut m = l;
            while m < n - 1 {
                let dd = d[m].abs() + d[m + 1].abs();
                if e[m].abs() <= f64::EPSILON * dd {
                    break;
                }
                m += 1;
            }
            if m == l {
                break;
            }
            iter += 1;
            if iter > MAX_QL_SWEEPS {
                return Err(EigError::NotConverged(l));
            }

            let mut g = (d[l + 1] - d[l]) / (2.0 * e[l]);
            let mut r = hypot2(g, 1.0);
            g = (d[m] - d[l]) + e[l] / (g + sign(r, g));
            let mut s = 1.0;
            let mut c = 1.0;
            let mut p = 0.0;
            let mut underflowed = false;

            for i in (l..m).rev() {
                let f = s * e[i];
                let b = c * e[i];
                r = hypot2(f, g);
                e[i + 1] = r;
                if r == 0.0 {
                    // The element has underflowed to zero: deflate and restart the sweep.
                    d[i + 1] -= p;
                    e[m] = 0.0;
                    underflowed = true;
                    break;
                }
                s = f / r;
                c = g / r;
                g = d[i + 1] - p;
                r = (d[i] - g) * s + 2.0 * c * b;
                p = s * r;
                d[i + 1] = g + p;
                g = c * r - b;
                if let Some(z) = z.as_deref_mut() {
                    for k in 0..n {
                        let f = z[k * n + i + 1];
                        z[k * n + i + 1] = s * z[k * n + i] + c * f;
                        z[k * n + i] = c * z[k * n + i] - s * f;
                    }
                }
            }
            if underflowed {
                continue;
            }
            d[l] -= p;
            e[l] = g;
            e[m] = 0.0;
        }
    }
    Ok(())
}
