//! Dense LU with partial pivoting — the Group C solver, and the only one in the project.
//!
//! `docs/dev/rust-migration-plan.md` §4 puts exactly one file in Group C: `collision`, whose
//! vector contact solve factors an `m × m` Newton Jacobian *every iteration of every timestep*.
//! The Python original calls `scipy.linalg.lu_factor` / `lu_solve`, i.e. LAPACK `dgetrf` /
//! `dgetrs`, and it calls them rather than `numpy.linalg.solve` on purpose: the plan records a
//! measured NumPy 2.4 threaded-BLAS cliff above ~100×100 (0.05 ms → 250 ms) that SciPy's path
//! does not hit. Nothing here may quietly reintroduce that, which a plain scalar transcription
//! cannot — it has no threading to get wrong.
//!
//! # Why this is not chasing bit-identity, and why that is settled before the first line
//!
//! §15.3 spent a batch establishing that OpenBLAS's `DTBSV` admits no scalar recipe. The same
//! question could be asked of `dgetrf`, and it is not worth asking, because the answer no longer
//! decides anything: `collision`'s Newton iteration is driven by `G @ F(η)`, a **dense BLAS matvec
//! whose result feeds back into the next iterate**. That is precisely the construction §14.2 named
//! as the end of bit-identity — `dgemv` fuses its multiply-add and OpenBLAS picks the kernel by
//! CPU. A perfectly reproduced factorization downstream of an irreproducible matvec buys nothing.
//!
//! So this module is written the plain way — right-looking, row-pivoted on the largest magnitude,
//! summed left to right — and `collision`'s agreement with SciPy is a tolerance from the start.
//! One thing it does keep from LAPACK is the *pivot choice*, which is a discrete decision rather
//! than a rounding: picking a different pivot row is a different elimination, not a different last
//! bit, and it would separate the two trajectories by far more than the arithmetic does.
//!
//! # Layout
//!
//! `a` is row-major `n × n`, which is what a C-contiguous NumPy array hands over. `lu` holds `L`
//! (unit diagonal, implicit) below the diagonal and `U` on and above it — LAPACK's packing, and
//! SciPy's `lu_factor` returns the same. `piv[i]` is the row swapped with row `i` at step `i`,
//! 0-based, which is SciPy's spelling of `ipiv` rather than LAPACK's 1-based one.

/// Why a dense solve was refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DenseError {
    /// The array's shape does not describe a square matrix, or a right-hand side does not match.
    BadShape,
}

impl std::fmt::Display for DenseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DenseError::BadShape => write!(f, "expected a square (n, n) matrix with n >= 1"),
        }
    }
}

/// The result of [`lu_factor`]: the packed factors, the pivots, and LAPACK's `info`.
#[derive(Debug, Clone)]
pub struct Lu {
    /// `L\U` packed row-major, `n × n`.
    pub lu: Vec<f64>,
    /// `piv[i]` is the row exchanged with row `i`, 0-based.
    pub piv: Vec<usize>,
    /// Order of the system.
    pub n: usize,
    /// `0`, or the 1-based index of the first exactly-zero diagonal of `U`.
    ///
    /// LAPACK returns this in `info` and SciPy turns it into a `LinAlgWarning` rather than an
    /// exception — the factorization is still handed back and the solve produces infinities. That
    /// is reproduced rather than upgraded to an error, because `collision`'s Jacobian
    /// `I + G·diag(F')` has every eigenvalue `>= 1` and so cannot reach it; a refusal here would
    /// be untestable code guarding an impossible case.
    pub info: usize,
}

/// `scipy.linalg.lu_factor(a)` — LAPACK `dgetrf`, unblocked.
pub fn lu_factor(mut a: Vec<f64>, n: usize) -> Result<Lu, DenseError> {
    if n == 0 || a.len() != n * n {
        return Err(DenseError::BadShape);
    }
    let mut piv = vec![0usize; n];
    let mut info = 0usize;

    for j in 0..n {
        // IDAMAX over the sub-column: strictly-greater keeps the FIRST maximum, which is the
        // reference kernel's choice and the reason two equal candidates do not pivot differently.
        let mut p = j;
        let mut best = a[j * n + j].abs();
        for i in (j + 1)..n {
            let v = a[i * n + j].abs();
            if v > best {
                best = v;
                p = i;
            }
        }
        piv[j] = p;
        if p != j {
            for c in 0..n {
                a.swap(j * n + c, p * n + c);
            }
        }

        let ajj = a[j * n + j];
        if ajj == 0.0 {
            if info == 0 {
                info = j + 1;
            }
            continue; // dgetf2 leaves the trailing update undone for a zero pivot.
        }
        // DSCAL by the reciprocal when it is safe, exactly as `dgetf2` does: the reference guards
        // on `|ajj| >= sfmin` and divides otherwise, and the two spellings are not the same in
        // binary floating point (§15.3 measured 19/120 against 120/120 on the banded factor).
        if ajj.abs() >= f64::MIN_POSITIVE {
            let inv = 1.0 / ajj;
            for i in (j + 1)..n {
                a[i * n + j] *= inv;
            }
        } else {
            for i in (j + 1)..n {
                a[i * n + j] /= ajj;
            }
        }
        // Rank-1 update of the trailing submatrix (DGER).
        for i in (j + 1)..n {
            let lij = a[i * n + j];
            if lij != 0.0 {
                for c in (j + 1)..n {
                    a[i * n + c] -= lij * a[j * n + c];
                }
            }
        }
    }
    Ok(Lu {
        lu: a,
        piv,
        n,
        info,
    })
}

/// `scipy.linalg.lu_solve((lu, piv), b)` for one right-hand side — LAPACK `dgetrs`, `TRANS = 'N'`.
pub fn lu_solve(f: &Lu, b: &[f64]) -> Result<Vec<f64>, DenseError> {
    let n = f.n;
    if b.len() != n || f.lu.len() != n * n || f.piv.len() != n {
        return Err(DenseError::BadShape);
    }
    let mut x = b.to_vec();

    // Apply the row interchanges, in factorization order (DLASWP forward).
    for i in 0..n {
        let p = f.piv[i];
        if p != i {
            x.swap(i, p);
        }
    }
    // Forward substitution, L unit lower triangular. Both loops walk the row slice paired with
    // the solution prefix rather than indexing, which keeps the left-to-right summation order the
    // reference kernel has — the order is the part that matters, not the spelling.
    for i in 0..n {
        let mut t = x[i];
        for (lij, xj) in f.lu[i * n..i * n + i].iter().zip(&x[..i]) {
            t -= lij * xj;
        }
        x[i] = t;
    }
    // Back substitution, U upper triangular.
    for i in (0..n).rev() {
        let mut t = x[i];
        for (uij, xj) in f.lu[i * n + i + 1..i * n + n].iter().zip(&x[i + 1..]) {
            t -= uij * xj;
        }
        x[i] = t / f.lu[i * n + i];
    }
    Ok(x)
}
