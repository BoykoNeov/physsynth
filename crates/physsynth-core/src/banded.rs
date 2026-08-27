//! Banded Cholesky — the shared solver behind the θ-scheme string family.
//!
//! `docs/dev/rust-migration-plan.md` §15. This module is the Phase 3 analogue of Phase 1's `ops`:
//! **not a model, but what four models are built out of.** `string_stiff`, `string_damped`,
//! `string_nonlinear` and `string_geometric` all factor a pentadiagonal SPD update matrix once (or,
//! in `string_nonlinear`'s case, once per distinct tension increment) and back-substitute every
//! timestep. Porting the solver before any of its callers is what keeps the family's four
//! bit-identity reduction anchors alive — see §15.2, which is the finding that forced this shape.
//!
//! # What is transcribed, and what deliberately is not
//!
//! SciPy dispatches `cholesky_banded` to LAPACK `dpbtrf` and `cho_solve_banded` to `dpbtrs`. At
//! `kd = 2` the blocked path in `dpbtrf` is never taken (`NB > KD`), so the factor is the
//! unblocked `DPBTF2`, and `dpbtrs` is two `DTBSV` calls. Both are transcribed here in their
//! reference form, with one detail that is *not* a detail:
//!
//! * `DPBTF2` scales the off-diagonal column through `DSCAL`, which forms the reciprocal **once**
//!   and multiplies. `x * (1/ajj)` is not `x / ajj` in binary floating point, and measured against
//!   OpenBLAS on 2026-08-27 the reciprocal spelling agreed on 120/120 of this family's matrices
//!   where the division spelling agreed on 19/120. That is the reference algorithm's own
//!   behaviour, not a property of a kernel, so it is transcribed.
//!
//! What is **not** transcribed is the fused multiply-add. With `fma` in the rank-1 update the
//! factor reproduces OpenBLAS exactly (120/120 against 82/120 without), but fusing is a property
//! of the `DSYR` kernel that `DYNAMIC_ARCH` picks at run time — asserting it would be a claim
//! about a CPU rather than about a port, which is the line §14.2 drew and this module keeps.
//!
//! The solve is not reproducible at all, and that was measured rather than assumed: per-element
//! candidate elimination over {forward, reverse} × {plain, fused} × {divide, reciprocal} on a
//! 15-element system left *disjoint* candidate sets — element 2 admitted only the dividing forms,
//! element 5 only the fused ones, element 9 only the forward order, element 14 excluded it — and
//! element 7 admitted nothing at all. No scalar recurrence produces OpenBLAS's `DTBSV`; it is a
//! blocked kernel. So this module does not chase it. It sums plainly, left to right, and the
//! family's agreement with LAPACK becomes a tolerance rather than an identity (§15.3).
//!
//! # Layout
//!
//! `ab` is row-major `(kd + 1) × n`, which is what a C-contiguous NumPy array of that shape hands
//! over, and it is *upper* banded storage: `ab[kd + i - j][j] == A[i][j]` for `j - kd <= i <= j`.
//! Row `kd` is the diagonal. Only the upper form is implemented — every call site in the project
//! passes `lower=False`, and a lower variant nothing calls would be untested code.

/// Why a factorization was refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BandedError {
    /// `A` is not positive definite; the 1-based index is the leading minor that failed, which is
    /// the number LAPACK returns in `info` and SciPy quotes back in its `LinAlgError`.
    NotPositiveDefinite(usize),
    /// The array's shape does not describe a band: zero columns, or a length that is not a
    /// multiple of `n`.
    BadShape,
}

impl std::fmt::Display for BandedError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BandedError::NotPositiveDefinite(i) => {
                write!(f, "{i}-th leading minor not positive definite")
            }
            BandedError::BadShape => write!(f, "ab must be a (kd + 1, n) band with n >= 1"),
        }
    }
}

/// Index into row-major `(kd + 1) × n` storage.
#[inline]
fn at(n: usize, row: usize, col: usize) -> usize {
    row * n + col
}

/// `A = U^T U`, upper banded — LAPACK `DPBTF2`, `UPLO = 'U'`.
///
/// `ab` is consumed by value and returned factored, mirroring `cholesky_banded`'s contract of
/// handing back a new array of the same shape rather than aliasing the input.
pub fn cholesky_banded_upper(
    mut ab: Vec<f64>,
    kd: usize,
    n: usize,
) -> Result<Vec<f64>, BandedError> {
    if n == 0 || ab.len() != (kd + 1) * n {
        return Err(BandedError::BadShape);
    }
    for j in 0..n {
        let mut ajj = ab[at(n, kd, j)];
        // `!(ajj > 0.0)`, not `ajj <= 0.0`: a NaN diagonal compares false against both, and it is
        // the negated form that then refuses. DPBTF2 tests `AJJ.LE.ZERO` on a value it has already
        // guaranteed non-NaN; here the caller may not have, so the refusal has to catch it.
        #[allow(clippy::neg_cmp_op_on_partial_ord)]
        if !(ajj > 0.0) {
            return Err(BandedError::NotPositiveDefinite(j + 1));
        }
        ajj = ajj.sqrt();
        ab[at(n, kd, j)] = ajj;

        // The trailing part of this column that still fits inside the band.
        let kn = kd.min(n - 1 - j);
        if kn == 0 {
            continue;
        }

        // DSCAL(kn, 1/ajj, ...) — reciprocal formed ONCE, then multiplied. See the header.
        let inv = 1.0 / ajj;
        for i in 0..kn {
            ab[at(n, kd - 1 - i, j + 1 + i)] *= inv;
        }

        // DSYR('U', kn, -1, x, ...) — the rank-1 update of the trailing block, where
        // `x[i] == A[j][j + 1 + i]` is the column just scaled. `alpha` multiplies `x[q]` first,
        // which is the order the reference kernel writes and therefore the order of the rounding.
        //
        // The read locations (row `kd - 1 - i`) and the write locations (row `kd + p - q`, with
        // `p <= q`) can never coincide, so this reads its own output only where LAPACK does too.
        for q in 0..kn {
            // `-1.0 * x` rather than `-x`, because DSYR's `alpha` multiplies `x[q]` first and the
            // rounding of this batch's transcription is the point of it. Negation is exact for a
            // finite double so the two agree — but the spelling is the reference kernel's.
            #[allow(clippy::neg_multiply)]
            let temp = -1.0 * ab[at(n, kd - 1 - q, j + 1 + q)];
            for p in 0..=q {
                let xp = ab[at(n, kd - 1 - p, j + 1 + p)];
                ab[at(n, kd + p - q, j + 1 + q)] += xp * temp;
            }
        }
    }
    Ok(ab)
}

/// `x := inv(U^T) x` — LAPACK `DTBSV('U', 'T', 'N')`, `incx = 1`.
fn tbsv_upper_transpose(chol: &[f64], kd: usize, n: usize, x: &mut [f64]) {
    for j in 0..n {
        let lo = j.saturating_sub(kd);
        let mut t = x[j];
        for i in lo..j {
            t -= chol[at(n, kd - (j - i), j)] * x[i];
        }
        x[j] = t / chol[at(n, kd, j)];
    }
}

/// `x := inv(U) x` — LAPACK `DTBSV('U', 'N', 'N')`, `incx = 1`.
///
/// The `x[j] != 0` guard is the reference kernel's, kept rather than simplified away: skipping the
/// update leaves a stored `-0.0` alone where subtracting `t * a` would flip its sign, and a signed
/// zero in the state is the kind of difference that survives to a comparison and explains nothing.
fn tbsv_upper_notranspose(chol: &[f64], kd: usize, n: usize, x: &mut [f64]) {
    for j in (0..n).rev() {
        if x[j] != 0.0 {
            x[j] /= chol[at(n, kd, j)];
            let t = x[j];
            let lo = j.saturating_sub(kd);
            for i in (lo..j).rev() {
                x[i] -= t * chol[at(n, kd - (j - i), j)];
            }
        }
    }
}

/// Solve `A x = b` given `A`'s upper banded Cholesky factor — LAPACK `DPBTRS`, `UPLO = 'U'`.
pub fn cho_solve_banded_upper(
    chol: &[f64],
    kd: usize,
    n: usize,
    b: &[f64],
) -> Result<Vec<f64>, BandedError> {
    if n == 0 || chol.len() != (kd + 1) * n || b.len() != n {
        return Err(BandedError::BadShape);
    }
    let mut x = b.to_vec();
    tbsv_upper_transpose(chol, kd, n, &mut x);
    tbsv_upper_notranspose(chol, kd, n, &mut x);
    Ok(x)
}
