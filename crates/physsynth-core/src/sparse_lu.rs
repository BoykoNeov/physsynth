//! Sparse LU with partial pivoting — the Group D solver, and the last solver class the migration
//! had not touched.
//!
//! `docs/dev/rust-migration-plan.md` §4 puts six files in Group D: `beam`, `operators2d`, `plate`,
//! `connection`, `string_geometric` and `airbox`, all of which factor once at construction and
//! back-substitute every timestep. The Python originals call `scipy.sparse.linalg.splu`, which is
//! SuperLU.
//!
//! # §4.1's hypothesis was tested on `beam` and it fails — measured, §24.2
//!
//! §4.1 proposed linking SuperLU itself so Group D could be held to the same bit-identity as
//! Groups A–C, and named three things that would have to match: the column ordering
//! (`permc_spec`), the pivot threshold (`diag_pivot_thresh`) and equilibration. All three were
//! measured on the beam's own matrix, and two of them are **non-issues**:
//!
//! * the ordering is a closed form in `n` for this pentadiagonal family, verified over twelve grid
//!   sizes (identity except that the two pairs at `n-5, n-4` and `n-3, n-2` are exchanged);
//! * SciPy calls `gstrf`, the factorization routine, not the `gssvx` driver, so **no
//!   equilibration happens at all** — `Equil=False` and `Equil=True` produce identical factors;
//! * the pivot threshold **is** a real obstacle, but only past a grid size the first two fixtures
//!   did not reach: SuperLU takes the diagonal up to `N = 48` and starts swapping rows at
//!   `N = 64`, filling `U` as it goes. This module declines that pivot on purpose — see
//!   [`DIAG_PIVOT_THRESH`] — so above the transition the two eliminations differ by a *discrete*
//!   decision and not only by a rounding.
//!
//! What does decide the answer is a fourth thing, which §4.1 did not list: SuperLU is
//! **supernodal**, and its panel/supernode blocking changes the arithmetic. `relax` and
//! `panel_size` visibly change the factors, and — the measurement that settles it — handed
//! SuperLU's *own* factors, a longhand column-oriented triangular solve still disagrees with
//! `lu.solve` in about 20 % of entries at ~4e-16. The blocking depends on how SciPy *built* its
//! copy of SuperLU (its `relax`/`panel_size` defaults, whether it compiled against an external
//! BLAS, its vendored patch level), so linking the library would buy a claim about a build rather
//! than about a library — §22.1's shape one layer down. The human's call (2026-08-28) is Group D's
//! fallback, which §4.1 had already named as survivable: **tolerance-level agreement, quantified**.
//!
//! So this module is written the plain way and makes no claim to reproduce SuperLU's digits. It
//! parts company with the reference on one discrete decision and does so deliberately: it prefers
//! the **diagonal** pivot, which is unconditionally stable on the SPD matrices Group D actually
//! holds and costs no fill, where the reference's strict threshold pivots and fills above
//! `N = 48`. `dense.rs` warns that a pivot choice is a different elimination rather than a
//! different last bit, and that warning is honoured by making the divergence a stated,
//! measured decision instead of an accident — see `tests/test_rust_parity_beam.py`, which prices
//! it (the rigid-body divergence at 5,000 steps grows about 20x between `N = 48` and `N = 96`,
//! and the energies still agree to ~1e-12).
//!
//! # The algorithm
//!
//! Left-looking Gilbert–Peierls: column `k` of `L` and `U` is obtained by solving `L x = A(:,k)`
//! against the columns already computed, and the *pattern* of that solve is the set of nodes
//! reachable from the nonzeros of `A(:,k)` in the graph of `L`, found by depth-first search. That
//! is what makes it sparse rather than dense-with-zeros: the work is proportional to the
//! nonzeros of the factors, not to `n²`.
//!
//! The column order is **natural**. SuperLU's COLAMD is not reproduced, because after the finding
//! above there is nothing left for it to buy: it exists to reduce fill, and every Group D matrix
//! in this project is a banded FDTD operator whose natural order already has none to speak of
//! (measured on the beam: `nnz(L) = nnz(U) = 3n`, which is the band itself). If a later model
//! makes fill the constraint, an ordering goes in front of this, not inside it.

use crate::sparse::Csr;

/// How far below the largest candidate the diagonal may sit and still be taken as the pivot.
///
/// This is SuperLU's own `diag_pivot_thresh` knob, and the value is where the Rust side and SciPy
/// deliberately differ. SciPy's default is `1.0` — strict partial pivoting — and the beam's own
/// matrix stops being diagonally largest between `N = 48` and `N = 64`, as the stiffness term
/// outgrows the mass. Past that point strict pivoting starts swapping rows and filling `U`, and it
/// does so on **both** sides: the reference stores 773 entries in `U` at `N = 200` where the band
/// is 600, and this module ordered naturally and pivoting strictly stores 772.
///
/// Preferring the diagonal removes both at once and is safe on exactly the matrices Group D holds:
/// every one of them is symmetric positive definite (`beam`, `plate`, `connection`, `airbox` all
/// build `(mass) + (positive coefficient)·(PSD stiffness)`), and for an SPD matrix elimination
/// without any pivoting is unconditionally stable — that is what makes Cholesky legal. The
/// threshold rather than a flat "always take the diagonal" is the guard for the day something
/// unsymmetric arrives: `0.1` is SuperLU's own documented alternative to `1.0`, and the beam's
/// diagonal clears it by a wide margin at every size tested.
pub const DIAG_PIVOT_THRESH: f64 = 0.1;

/// Why a sparse factorization or solve was refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SparseLuError {
    /// The matrix is not square, or is empty.
    BadShape,
    /// A right-hand side whose length is not the order of the system.
    BadRhs,
    /// Column `.0` had no admissible pivot — the matrix is (numerically) singular.
    Singular(usize),
}

impl std::fmt::Display for SparseLuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SparseLuError::BadShape => write!(f, "expected a square (n, n) matrix with n >= 1"),
            SparseLuError::BadRhs => write!(f, "right-hand side length does not match the matrix"),
            SparseLuError::Singular(k) => write!(f, "matrix is singular: no pivot in column {k}"),
        }
    }
}

/// Compressed-sparse-column storage, built column by column as the factorization proceeds.
#[derive(Debug, Clone, Default)]
struct CscBuilder {
    indptr: Vec<usize>,
    indices: Vec<usize>,
    data: Vec<f64>,
}

impl CscBuilder {
    fn with_capacity(n: usize, nnz_hint: usize) -> Self {
        let mut indptr = Vec::with_capacity(n + 1);
        indptr.push(0);
        Self {
            indptr,
            indices: Vec::with_capacity(nnz_hint),
            data: Vec::with_capacity(nnz_hint),
        }
    }

    fn push(&mut self, i: usize, x: f64) {
        self.indices.push(i);
        self.data.push(x);
    }

    fn end_column(&mut self) {
        self.indptr.push(self.indices.len());
    }
}

/// One triangular factor laid out for the per-timestep solve: each column's off-diagonal entries
/// sorted by row and grouped into runs of consecutive rows, values stored contiguously.
///
/// The CSC form drags an 8-byte row index along with every 8-byte value, and its scattered writes
/// cannot be vectorised; a run is a slice, so the axpy over it can be. Sorting a column cannot
/// change a bit: within one column every entry updates a DIFFERENT row, exactly once, so the
/// order inside the column is free, and the columns are still visited in the same order. No gap
/// is padded with an explicit zero — `-0.0 - 0.0 * xj` is `+0.0` when `xj < 0`, and `0 * inf` is
/// NaN — so the runs carry the stored pattern and nothing else.
#[derive(Debug, Clone, Default)]
struct Runs {
    /// Runs of column `j` are `col[j]..col[j + 1]`.
    col: Vec<usize>,
    /// First row of each run.
    start: Vec<usize>,
    /// Length of each run.
    len: Vec<usize>,
    /// Offset of each run's first value in `data`.
    off: Vec<usize>,
    data: Vec<f64>,
}

impl Runs {
    /// From a CSC factor in its final numbering, leaving out each column's LAST stored entry
    /// when `drop_last` (U's diagonal, which the solve divides by rather than eliminates with).
    fn from_csc(c: &CscBuilder, drop_last: bool) -> Self {
        let n = c.indptr.len() - 1;
        let mut r = Runs {
            col: Vec::with_capacity(n + 1),
            data: Vec::with_capacity(c.data.len()),
            ..Runs::default()
        };
        r.col.push(0);
        let mut entries: Vec<(usize, f64)> = Vec::new();
        for j in 0..n {
            let end = c.indptr[j + 1] - usize::from(drop_last);
            entries.clear();
            entries.extend((c.indptr[j]..end).map(|p| (c.indices[p], c.data[p])));
            entries.sort_unstable_by_key(|&(i, _)| i);
            for (k, &(i, v)) in entries.iter().enumerate() {
                if k == 0 || i != entries[k - 1].0 + 1 {
                    r.start.push(i);
                    r.len.push(0);
                    r.off.push(r.data.len());
                }
                *r.len.last_mut().expect("a run was opened") += 1;
                r.data.push(v);
            }
            r.col.push(r.start.len());
        }
        r
    }

    /// `x[i] -= a_ij * xj` over column `j`'s stored entries — the scatter both sweeps share.
    #[inline]
    fn eliminate(&self, j: usize, xj: f64, x: &mut [f64]) {
        for q in self.col[j]..self.col[j + 1] {
            let (s, len, off) = (self.start[q], self.len[q], self.off[q]);
            for (xi, &a) in x[s..s + len].iter_mut().zip(&self.data[off..off + len]) {
                *xi -= a * xj;
            }
        }
    }
}

/// The factors of a sparse `A = P⁻¹ L U`, ready for repeated back-substitution.
///
/// `L` is unit-lower-triangular in the *permuted* row numbering and `U` is upper-triangular;
/// `pinv[i]` is the position that original row `i` takes in that numbering. `beam` and every other
/// Group D model factors once at construction and calls [`SparseLu::solve`] once per timestep, so
/// the split matters: the DFS and the fill live entirely in [`SparseLu::factor`].
#[derive(Debug, Clone)]
pub struct SparseLu {
    n: usize,
    l: CscBuilder,
    u: CscBuilder,
    /// `l` and `u` again, laid out for [`SparseLu::solve`]; the CSC copies are what the
    /// elimination built and what [`SparseLu::nnz`] counts.
    l_runs: Runs,
    u_runs: Runs,
    /// U's diagonal, the last entry of each of its columns.
    u_diag: Vec<f64>,
    /// `pinv[original_row] = permuted_row`.
    pinv: Vec<usize>,
    /// The fill-reducing reordering the matrix was factored in, `q[factored] = caller`, or `None`
    /// for the natural order. See [`SparseLu::factor_permuted`].
    q: Vec<usize>,
    reordered: bool,
}

/// The column-compressed form of a [`Csr`], which is what the elimination walks.
fn to_csc(a: &Csr) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    let (n, m) = (a.nrows(), a.ncols());
    let mut counts = vec![0usize; m + 1];
    for &j in a.indices() {
        counts[j + 1] += 1;
    }
    for j in 0..m {
        counts[j + 1] += counts[j];
    }
    let indptr = counts.clone();
    let mut indices = vec![0usize; a.nnz()];
    let mut data = vec![0.0f64; a.nnz()];
    let mut next = indptr.clone();
    for i in 0..n {
        for p in a.indptr()[i]..a.indptr()[i + 1] {
            let j = a.indices()[p];
            indices[next[j]] = i;
            data[next[j]] = a.data()[p];
            next[j] += 1;
        }
    }
    (indptr, indices, data)
}

impl SparseLu {
    /// Order of the factored system.
    pub fn n(&self) -> usize {
        self.n
    }

    /// Nonzeros in `L` (unit diagonal excluded) and in `U`.
    pub fn nnz(&self) -> (usize, usize) {
        (self.l.indices.len(), self.u.indices.len())
    }

    /// Whether the row permutation is the identity — i.e. no pivot ever fired.
    ///
    /// A pivot is a *discrete* decision rather than a rounding, so this is the one thing about the
    /// elimination that the Python reference and this module have to agree on exactly, and it is
    /// checkable on either side (SciPy reports it as `perm_r == perm_c`).
    pub fn is_natural(&self) -> bool {
        self.pinv.iter().enumerate().all(|(i, &p)| i == p)
    }

    /// Factor `a` in the natural column order at the default pivot threshold.
    pub fn factor(a: &Csr) -> Result<Self, SparseLuError> {
        Self::factor_with_thresh(a, DIAG_PIVOT_THRESH)
    }

    /// Factor `a` in the natural column order, taking the diagonal as pivot whenever it is at
    /// least `thresh` times the largest candidate. `thresh = 1.0` is strict partial pivoting;
    /// `thresh = 0.0` always takes the diagonal. See [`DIAG_PIVOT_THRESH`].
    pub fn factor_with_thresh(a: &Csr, thresh: f64) -> Result<Self, SparseLuError> {
        let n = a.nrows();
        if n == 0 || a.ncols() != n {
            return Err(SparseLuError::BadShape);
        }
        let (ap, ai, ax) = to_csc(a);

        let nnz_hint = a.nnz() + n;
        let mut l = CscBuilder::with_capacity(n, nnz_hint);
        let mut u = CscBuilder::with_capacity(n, nnz_hint);
        let mut pinv = vec![usize::MAX; n];

        // Workspace for the sparse triangular solve: `x` is the dense accumulator (only the
        // reachable entries are ever touched or read), `xi[top..n]` the reachable rows in
        // topological order, and `stack`/`next` the explicit DFS frames.
        let mut x = vec![0.0f64; n];
        let mut xi = vec![0usize; n];
        let mut marked = vec![false; n];
        let mut stack = vec![0usize; n];
        let mut next = vec![0usize; n];

        for k in 0..n {
            let top = spsolve_column(
                &l,
                &pinv,
                &ap,
                &ai,
                &ax,
                k,
                &mut x,
                &mut xi,
                &mut marked,
                &mut stack,
                &mut next,
            );

            // Split the solved column: rows already pivotal are U, the rest are pivot candidates.
            // The pivot is the largest magnitude among the candidates -- and, exactly as in
            // `dense.rs`, a strictly-greater test keeps the FIRST maximum so that two equal
            // candidates cannot pivot differently between runs.
            let mut ipiv = usize::MAX;
            let mut best = 0.0f64;
            let mut diag = 0.0f64;
            for &i in &xi[top..n] {
                if pinv[i] == usize::MAX {
                    let mag = x[i].abs();
                    if i == k {
                        diag = mag;
                    }
                    if ipiv == usize::MAX || mag > best {
                        best = mag;
                        ipiv = i;
                    }
                }
            }
            if ipiv == usize::MAX || best == 0.0 {
                return Err(SparseLuError::Singular(k));
            }
            if diag > 0.0 && diag >= thresh * best {
                ipiv = k; // the diagonal is acceptable: take it, as the reference does
            }
            let pivot = x[ipiv];
            pinv[ipiv] = k;

            // U gets the pivotal rows in their permuted numbering, then the pivot on the diagonal;
            // L gets the rest, scaled by the reciprocal ONCE. That spelling is not cosmetic: §15.3
            // measured it as the difference between agreeing with a reference factor in 82 of 120
            // cases and in none of them, and it is what every library does.
            let recip = 1.0 / pivot;
            for &i in &xi[top..n] {
                if pinv[i] != usize::MAX && pinv[i] != k {
                    u.push(pinv[i], x[i]);
                }
            }
            u.push(k, pivot);
            u.end_column();
            for &i in &xi[top..n] {
                if pinv[i] == usize::MAX {
                    l.push(i, x[i] * recip);
                }
            }
            l.end_column();
            for p in top..n {
                x[xi[p]] = 0.0; // every reached row, not just L's -- U's are live too
            }
        }

        // Rewrite L's row indices into the final permuted numbering. They could not be written
        // that way as they were produced: at step `k` the rows below the pivot have no permuted
        // number yet -- that is exactly what makes them candidates.
        for i in l.indices.iter_mut() {
            *i = pinv[*i];
        }

        let u_diag = (0..n).map(|j| u.data[u.indptr[j + 1] - 1]).collect();
        Ok(Self {
            n,
            l_runs: Runs::from_csc(&l, false),
            u_runs: Runs::from_csc(&u, true),
            u_diag,
            l,
            u,
            pinv,
            q: (0..n).collect(),
            reordered: false,
        })
    }

    /// Factor `a` after the symmetric reordering `q` (`q[factored_index] = caller_index`), which
    /// the caller chooses to reduce fill. [`SparseLu::solve`] takes and returns vectors in the
    /// **caller's** numbering, so the reordering is invisible from the outside.
    ///
    /// # Why this exists — §29.2, and it is the first Group D matrix that needs it
    ///
    /// §24 wrote that the column order here is natural because "every Group D matrix in this
    /// project is a banded FDTD operator whose natural order already has none [no fill] to speak
    /// of", and added: *if a later model makes fill the constraint, an ordering goes in front of
    /// this, not inside it.* `string_geometric` is that model, and it is not a near miss.
    ///
    /// Its Newton Jacobian is `3(N-1)` unknowns stacked **by field** — all of `u`, then all of
    /// `w`, then all of `v` — while the discrete-gradient force couples the three fields *at the
    /// same cell*. So in the natural order every coupling sits `N-1` columns off the diagonal, and
    /// the elimination fills the whole envelope between. Measured at `N = 128` (`n = 381`):
    /// SuperLU's COLAMD stores 2,788 nonzeros in `L + U`, this module in the natural order stores
    /// **33,895**, and factoring costs 2,068 µs against SciPy's 156. Reordered by *node* —
    /// `(u_i, w_i, v_i)` together — it stores **2,645**, fewer than COLAMD's, and costs 58 µs.
    /// The reordering is a closed form in `N`, so no ordering heuristic is needed or wanted; that
    /// is §24's own finding about the beam's permutation being a closed form, arriving on the
    /// other side of the ledger.
    ///
    /// The permutation is applied to rows and columns alike, so the diagonal is still the
    /// diagonal and [`DIAG_PIVOT_THRESH`]'s preference means what it meant before.
    pub fn factor_permuted(a: &Csr, q: &[usize], thresh: f64) -> Result<Self, SparseLuError> {
        if a.nrows() != a.ncols() || q.len() != a.nrows() {
            return Err(SparseLuError::BadShape);
        }
        let mut lu = Self::factor_with_thresh(&a.permute_symmetric(q), thresh)?;
        lu.q = q.to_vec();
        lu.reordered = true;
        Ok(lu)
    }

    /// Whether a fill-reducing reordering was supplied to [`SparseLu::factor_permuted`].
    pub fn is_reordered(&self) -> bool {
        self.reordered
    }

    /// Solve `A x = b` by back-substitution through the stored factors.
    ///
    /// This is the per-timestep call. It is column-oriented (an axpy down each column) rather than
    /// row-oriented (a dot product across each row) because the factors are stored by column, and
    /// because a column sweep is the order a reader can follow from the code — there is no
    /// reduction here whose associativity is in question.
    ///
    /// It walks the factors as [`Runs`], which is what makes it fast (1.75x measured on the
    /// viewer's default 60 x 60 plate, 0.56 -> 0.32 ms, where the solve is ~80% of the implicit
    /// step) and changes no bit: every `x[i]` receives the same
    /// subtractions in the same column order as the CSC loop it replaced. `tests` below holds
    /// that loop as the reference and compares the two bit for bit.
    pub fn solve(&self, b: &[f64]) -> Result<Vec<f64>, SparseLuError> {
        if b.len() != self.n {
            return Err(SparseLuError::BadRhs);
        }
        let n = self.n;
        let mut x = vec![0.0f64; n];
        // Two permutations compose here and they are different things: `q` is the caller's
        // fill-reducing reordering, fixed before any arithmetic, and `pinv` is the pivoting the
        // elimination chose inside it.
        for i in 0..n {
            x[self.pinv[i]] = b[self.q[i]];
        }

        // Forward: L is unit lower triangular, its diagonal implicit and absent from storage.
        for j in 0..n {
            let xj = x[j];
            if xj != 0.0 {
                self.l_runs.eliminate(j, xj, &mut x);
            }
        }
        // Back: U's diagonal is the LAST entry pushed in each column, by construction above.
        for j in (0..n).rev() {
            x[j] /= self.u_diag[j];
            let xj = x[j];
            if xj != 0.0 {
                self.u_runs.eliminate(j, xj, &mut x);
            }
        }
        if self.reordered {
            let mut out = vec![0.0f64; n];
            for i in 0..n {
                out[self.q[i]] = x[i];
            }
            return Ok(out);
        }
        Ok(x)
    }
}

/// Solve `L x = A(:, k)` for the already-computed columns of `L`, returning `top` such that
/// `xi[top..n]` is the pattern of `x` in topological order.
///
/// Gilbert–Peierls' half: the pattern is the set of nodes reachable from the nonzeros of `A(:, k)`
/// in the directed graph of `L`, and a depth-first search emits it in reverse topological order,
/// which is the order the substitution needs.
#[allow(clippy::too_many_arguments)]
fn spsolve_column(
    l: &CscBuilder,
    pinv: &[usize],
    ap: &[usize],
    ai: &[usize],
    ax: &[f64],
    k: usize,
    x: &mut [f64],
    xi: &mut [usize],
    marked: &mut [bool],
    stack: &mut [usize],
    next: &mut [usize],
) -> usize {
    let n = x.len();
    let mut top = n;
    for p in ap[k]..ap[k + 1] {
        let i = ai[p];
        if !marked[i] {
            top = dfs(i, l, pinv, top, xi, marked, stack, next);
        }
        x[i] = ax[p];
    }
    for p in top..n {
        marked[xi[p]] = false;
    }

    // xi[top..n] is in topological order, so each x[j] is final by the time it is used.
    for &i in &xi[top..n] {
        let j = pinv[i];
        if j == usize::MAX {
            continue; // not yet pivotal: no column of L to eliminate against
        }
        let xj = x[i];
        if xj != 0.0 {
            for q in l.indptr[j]..l.indptr[j + 1] {
                x[l.indices[q]] -= l.data[q] * xj;
            }
        }
    }
    top
}

/// Iterative depth-first search from `root` through the graph of `L`, pushing finished nodes onto
/// `xi` from the back. Iterative rather than recursive so that a large room cannot overflow the
/// stack at Phase 6.
#[allow(clippy::too_many_arguments)]
fn dfs(
    root: usize,
    l: &CscBuilder,
    pinv: &[usize],
    mut top: usize,
    xi: &mut [usize],
    marked: &mut [bool],
    stack: &mut [usize],
    next: &mut [usize],
) -> usize {
    let mut head = 0usize;
    stack[0] = root;
    marked[root] = true;
    next[0] = match pinv[root] {
        usize::MAX => usize::MAX, // no column of L: the node has no children
        j => l.indptr[j],
    };

    loop {
        let i = stack[head];
        let j = pinv[i];
        let mut descended = false;
        if j != usize::MAX {
            while next[head] < l.indptr[j + 1] {
                let child = l.indices[next[head]];
                next[head] += 1;
                if !marked[child] {
                    marked[child] = true;
                    head += 1;
                    stack[head] = child;
                    next[head] = match pinv[child] {
                        usize::MAX => usize::MAX,
                        cj => l.indptr[cj],
                    };
                    descended = true;
                    break;
                }
            }
        }
        if !descended {
            top -= 1;
            xi[top] = i;
            if head == 0 {
                return top;
            }
            head -= 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The CSC loop [`SparseLu::solve`] ran before it walked [`Runs`], kept verbatim as the
    /// reference the runs must reproduce to the bit.
    fn reference_solve(lu: &SparseLu, b: &[f64]) -> Vec<f64> {
        let n = lu.n;
        let mut x = vec![0.0f64; n];
        for i in 0..n {
            x[lu.pinv[i]] = b[lu.q[i]];
        }
        for j in 0..n {
            let xj = x[j];
            if xj != 0.0 {
                for p in lu.l.indptr[j]..lu.l.indptr[j + 1] {
                    x[lu.l.indices[p]] -= lu.l.data[p] * xj;
                }
            }
        }
        for j in (0..n).rev() {
            let end = lu.u.indptr[j + 1];
            x[j] /= lu.u.data[end - 1];
            let xj = x[j];
            if xj != 0.0 {
                for p in lu.u.indptr[j]..(end - 1) {
                    x[lu.u.indices[p]] -= lu.u.data[p] * xj;
                }
            }
        }
        if lu.reordered {
            let mut out = vec![0.0f64; n];
            for i in 0..n {
                out[lu.q[i]] = x[i];
            }
            return out;
        }
        x
    }

    /// Right-hand sides that reach every branch of the sweep: both signs, exact `+0.0` and `-0.0`
    /// (so the zero-skip fires and a sign of zero has to survive), mostly-zero vectors, a single
    /// spike, and magnitudes spread over many decades.
    fn rhs_family(n: usize) -> Vec<Vec<f64>> {
        let mut s = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        let mut out = Vec::new();
        for kind in 0..6 {
            let v: Vec<f64> = (0..n)
                .map(|_| {
                    let r = next();
                    let mag = (r >> 11) as f64 / (1u64 << 53) as f64;
                    let signed = if r & 1 == 0 { mag } else { -mag };
                    match kind {
                        0 => signed,
                        1 => signed * 10f64.powi((r % 40) as i32 - 20),
                        2 if r % 5 == 0 => signed,
                        2 => 0.0,
                        3 if r % 3 == 0 => -0.0,
                        3 if r % 3 == 1 => 0.0,
                        3 => signed,
                        4 if r % 97 == 0 => signed,
                        4 => -0.0,
                        _ => signed * 1e-300,
                    }
                })
                .collect();
            out.push(v);
        }
        let mut spike = vec![0.0; n];
        spike[n / 2] = -1.0;
        out.push(spike);
        out
    }

    fn assert_bitwise(label: &str, lu: &SparseLu) {
        for (k, b) in rhs_family(lu.n).iter().enumerate() {
            let got = lu.solve(b).expect("a valid rhs");
            let want = reference_solve(lu, b);
            let differ = got
                .iter()
                .zip(&want)
                .filter(|(g, w)| g.to_bits() != w.to_bits())
                .count();
            assert_eq!(
                differ, 0,
                "{label}, rhs #{k}: {differ} of {} entries differ",
                lu.n
            );
        }
    }

    /// The 1-D second difference with Dirichlet ends, `n` interior nodes.
    fn d2(n: usize) -> Csr {
        let rows = (0..n)
            .map(|i| {
                let mut r = vec![(i, -2.0)];
                if i > 0 {
                    r.push((i - 1, 1.0));
                }
                if i + 1 < n {
                    r.push((i + 1, 1.0));
                }
                r
            })
            .collect();
        Csr::from_rows(n, n, rows)
    }

    /// `I + c B` with `B = L L` — the shape of the supported plate's implicit system (a 13-point
    /// biharmonic), or of the beam's (a pentadiagonal) when `ny == 1`. This file is also compiled
    /// into `physsynth-analysis`, which has no plate or beam, so the operators are built here.
    fn implicit_system(nx: usize, ny: usize, c: f64) -> Csr {
        let lap = if ny == 1 {
            d2(nx)
        } else {
            Csr::identity(ny)
                .kron(&d2(nx))
                .add(&d2(ny).kron(&Csr::identity(nx)))
        };
        Csr::identity(nx * ny).add(&lap.matmul(&lap).scaled(c))
    }

    #[test]
    fn the_runs_solve_matches_the_csc_loop_bit_for_bit() {
        let plate = SparseLu::factor(&implicit_system(31, 23, 0.37)).expect("SPD");
        let beam = SparseLu::factor(&implicit_system(120, 1, 2.9)).expect("SPD");
        assert!(
            plate.l_runs.start.len() < plate.nnz().0,
            "runs, not singletons"
        );
        assert_bitwise("plate-shaped", &plate);
        assert_bitwise("beam-shaped", &beam);

        // A shuffled symmetric reordering: the runs then hold gaps, and `q` is not the identity.
        let a = implicit_system(17, 13, 0.21);
        let n = a.nrows();
        let mut q: Vec<usize> = (0..n).collect();
        let mut s = 12345u64;
        for i in (1..n).rev() {
            s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            q.swap(i, (s >> 33) as usize % (i + 1));
        }
        let reordered = SparseLu::factor_permuted(&a, &q, DIAG_PIVOT_THRESH).expect("SPD");
        assert!(reordered.is_reordered());
        let gaps = reordered.l_runs.start.len() > reordered.l_runs.col.len() - 1;
        assert!(
            gaps,
            "a shuffled order must leave some column in more than one run"
        );
        assert_bitwise("reordered plate-shaped", &reordered);

        // Strict pivoting on an unsymmetric matrix, so `pinv` is not the identity either.
        let m = 40;
        let rows: Vec<Vec<(usize, f64)>> = (0..m)
            .map(|i| {
                vec![
                    (i, 0.01 + i as f64 * 1e-3),
                    ((i + 1) % m, 1.0 + (i % 7) as f64),
                    ((i + 5) % m, -0.5 - (i % 3) as f64),
                ]
            })
            .collect();
        let unsym = Csr::from_rows(m, m, rows);
        let pivoted = SparseLu::factor_with_thresh(&unsym, 1.0).expect("nonsingular");
        assert!(!pivoted.is_natural(), "the fixture must actually pivot");
        assert_bitwise("pivoted unsymmetric", &pivoted);
    }
}
