//! A minimal compressed-sparse-row matrix — just enough to *assemble* the operators.
//!
//! # Why this is hand-written rather than a dependency
//!
//! `crates/physsynth-core/tests/deps.rs` keeps the core's dependency list **empty**, and the
//! migration plan (§2.2) says the first numeric crate arrives as a deliberate edit with its reason
//! written next to it. Phase 1 does not supply that reason: the three operator builders only ever
//! *construct* matrices — transpose, multiply, scale — and never solve with one. The constraint
//! that should actually choose a sparse library is a **solver** constraint, and the plan puts it at
//! Phase 3 (banded Cholesky) and Phase 4 (the SuperLU hypothesis, §4.1), where it can be measured.
//! Pulling in `faer` or `nalgebra-sparse` now to get a `matmul` would fix the interchange type
//! before the requirement that ought to pick it exists.
//!
//! So: ~200 lines, no dependencies, and the day a solver lands this type becomes the thing that
//! converts *into* whatever that solver wants. That is a smaller commitment than the reverse.
//!
//! # Canonical form, and the one place this deliberately differs from SciPy
//!
//! Every matrix this module produces is **canonical**: column indices strictly ascending within
//! each row, no duplicates, no explicit zeros. SciPy is not. Measured 2026-08-26,
//! `biharmonic_matrix` comes back from `(d2 @ d2).tocsr()` with `has_sorted_indices == False` and
//! its columns in *descending* order — an artifact of SciPy's SMMP kernel, whose output list is a
//! stack. `free_beam_stiffness`, which reaches its product through a transpose, comes back sorted.
//!
//! Reproducing that split would mean reimplementing a SciPy internal *and* pinning the port to a
//! detail SciPy is free to change in a point release — a red gate on an upgrade, for a non-bug.
//! Both spellings describe the same matrix, and every consumer in this repo treats them as such:
//! nothing downstream reads `.data` or `.indices`, the matrices are only ever used as operators.
//! `tests/test_rust_parity_operators.py` therefore canonicalises the SciPy side before comparing,
//! and says so.
//!
//! What is **not** relaxed is the arithmetic. Products are accumulated in ascending order of the
//! contracted index, which is what SciPy's kernel does, and the resulting `data` is asserted equal
//! to SciPy's bit-for-bit.

/// Why a raw CSR triple could not be taken verbatim — see
/// [`Csr::from_arrays_preserving_order`].
///
/// A `Result` rather than the panicking asserts the other constructors use, because this is the
/// one constructor whose input comes from *outside* the crate: the binding hands it a SciPy
/// matrix a caller assigned, and a caller's malformed matrix must be a raise and not an abort.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RawCsrError {
    /// `indptr` was not `nrows + 1` long.
    IndptrLength { expected: usize, got: usize },
    /// `indptr` is not a run of row starts: it decreased here, started above zero, or its last
    /// entry disagreed with `data.len()`.
    IndptrNotRowStarts { row: usize },
    /// `indices` and `data` disagreed on how many values are stored.
    LengthMismatch { indices: usize, data: usize },
    /// A column index was at least `ncols`.
    ColumnOutOfRange { column: usize, ncols: usize },
    /// A row stored the same column twice — [`Csr::from_rows`] rejects that too, and for the
    /// same reason: a stencil written twice is a bug, not a matrix to sum.
    DuplicateColumn { row: usize, column: usize },
}

impl std::fmt::Display for RawCsrError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RawCsrError::IndptrLength { expected, got } => {
                write!(f, "indptr must have {expected} entries, got {got}.")
            }
            RawCsrError::IndptrNotRowStarts { row } => {
                write!(f, "indptr is not a run of row starts (at row {row}).")
            }
            RawCsrError::LengthMismatch { indices, data } => {
                write!(f, "indices has {indices} entries and data has {data}.")
            }
            RawCsrError::ColumnOutOfRange { column, ncols } => {
                write!(f, "column {column} out of range for {ncols} columns.")
            }
            RawCsrError::DuplicateColumn { row, column } => {
                write!(f, "row {row} stores column {column} twice.")
            }
        }
    }
}

/// A sparse matrix in canonical compressed-sparse-row form.
#[derive(Clone, Debug, PartialEq)]
pub struct Csr {
    nrows: usize,
    ncols: usize,
    /// Row starts; length `nrows + 1`, with `indptr[nrows] == data.len()`.
    indptr: Vec<usize>,
    /// Column index per stored value; ascending within each row.
    indices: Vec<usize>,
    data: Vec<f64>,
}

impl Csr {
    /// Build from per-row `(column, value)` lists.
    ///
    /// Each row is sorted by column and exact zeros are dropped, so the result is canonical
    /// whatever order the caller supplied. Duplicate columns within a row are a caller bug and
    /// panic rather than being summed — none of the builders here produce one, and silently
    /// accepting them would hide a stencil written twice.
    ///
    /// # Panics
    /// If `rows` is the wrong length, a column index is out of range, or a row repeats a column.
    pub fn from_rows(nrows: usize, ncols: usize, rows: Vec<Vec<(usize, f64)>>) -> Self {
        assert_eq!(rows.len(), nrows, "expected one entry list per row");
        let mut indptr = Vec::with_capacity(nrows + 1);
        let mut indices = Vec::new();
        let mut data = Vec::new();
        indptr.push(0);
        for mut row in rows {
            row.sort_by_key(|&(j, _)| j);
            let mut last: Option<usize> = None;
            for (j, v) in row {
                assert!(j < ncols, "column {j} out of range for {ncols} columns");
                assert!(last != Some(j), "column {j} repeated in a row");
                last = Some(j);
                if v != 0.0 {
                    indices.push(j);
                    data.push(v);
                }
            }
            indptr.push(data.len());
        }
        Self {
            nrows,
            ncols,
            indptr,
            indices,
            data,
        }
    }

    /// Build from per-row `(column, value)` lists, **keeping** entries whose value is exactly zero.
    ///
    /// [`Csr::from_rows`] drops them, which is right for every matrix this crate assembles from a
    /// stencil: a structural zero and a dropped one are the same operator and the sparser storage
    /// is free. It is wrong for a matrix that has to be handed back to SciPy and compared, because
    /// SciPy's `coo_matrix(...).tocsr()` and its sparse product both keep a stored `0.0` and
    /// `nnz` is asserted exactly. `airbox_port` builds two such matrices — `T`, whose reference
    /// deliberately keeps the entries of a zero-**area** surface node so that the `T = 0` reduction
    /// to a bare resonator stays exercisable, and the load matrix built from it.
    ///
    /// # Panics
    /// As [`Csr::from_rows`].
    pub fn from_rows_keeping_zeros(
        nrows: usize,
        ncols: usize,
        rows: Vec<Vec<(usize, f64)>>,
    ) -> Self {
        assert_eq!(rows.len(), nrows, "expected one entry list per row");
        let mut indptr = Vec::with_capacity(nrows + 1);
        let mut indices = Vec::new();
        let mut data = Vec::new();
        indptr.push(0);
        for mut row in rows {
            row.sort_by_key(|&(j, _)| j);
            let mut last: Option<usize> = None;
            for (j, v) in row {
                assert!(j < ncols, "column {j} out of range for {ncols} columns");
                assert!(last != Some(j), "column {j} repeated in a row");
                last = Some(j);
                indices.push(j);
                data.push(v);
            }
            indptr.push(data.len());
        }
        Self {
            nrows,
            ncols,
            indptr,
            indices,
            data,
        }
    }

    /// Build from raw CSR arrays, keeping each row **exactly as the caller stored it** —
    /// including a column order that does not ascend.
    ///
    /// Every other constructor here sorts, and that is the point of them: this crate's whole
    /// ordering story is that a canonical row is the one order both languages can express, so an
    /// assembled operator is canonical by construction and a matvec over it is a reproducible
    /// sum. This constructor exists for the single case where the *non*-canonical order is the
    /// subject rather than an accident. `tests/test_plate_modal.py` steps a plate on the
    /// pre-2026-08-28 biharmonic — the same numbers, in the order SciPy's sparse product happens
    /// to emit them — to assert that the canonical sort moved a summation order and not a value.
    /// Sorting here would hand that test the shipped operator twice and it would pass while
    /// comparing nothing.
    ///
    /// **The result may therefore break the invariant the rest of this type documents.**
    /// [`Csr::matvec`], [`Csr::get`], [`Csr::scaled`], [`Csr::nnz`] and the accessors do not care
    /// about row order; the three merge kernels ([`Csr::add`], [`Csr::sub`], [`Csr::matmul`]) do,
    /// and `debug_assert` it rather than trusting their callers. Ask
    /// [`Csr::has_sorted_indices`] of any matrix you did not assemble yourself.
    ///
    /// # Errors
    /// Any of [`RawCsrError`]. Structural zeros are **kept** — a caller handing over an existing
    /// matrix is handing over its `nnz` too.
    pub fn from_arrays_preserving_order(
        nrows: usize,
        ncols: usize,
        indptr: Vec<usize>,
        indices: Vec<usize>,
        data: Vec<f64>,
    ) -> Result<Self, RawCsrError> {
        if indptr.len() != nrows + 1 {
            return Err(RawCsrError::IndptrLength {
                expected: nrows + 1,
                got: indptr.len(),
            });
        }
        if indices.len() != data.len() {
            return Err(RawCsrError::LengthMismatch {
                indices: indices.len(),
                data: data.len(),
            });
        }
        if indptr[0] != 0 {
            return Err(RawCsrError::IndptrNotRowStarts { row: 0 });
        }
        for i in 0..nrows {
            if indptr[i + 1] < indptr[i] || indptr[i + 1] > data.len() {
                return Err(RawCsrError::IndptrNotRowStarts { row: i });
            }
        }
        if indptr[nrows] != data.len() {
            return Err(RawCsrError::IndptrNotRowStarts { row: nrows });
        }
        for i in 0..nrows {
            let row = &indices[indptr[i]..indptr[i + 1]];
            for &j in row {
                if j >= ncols {
                    return Err(RawCsrError::ColumnOutOfRange { column: j, ncols });
                }
            }
            let mut seen = row.to_vec();
            seen.sort_unstable();
            if let Some(w) = seen.windows(2).find(|w| w[0] == w[1]) {
                return Err(RawCsrError::DuplicateColumn {
                    row: i,
                    column: w[0],
                });
            }
        }
        Ok(Self {
            nrows,
            ncols,
            indptr,
            indices,
            data,
        })
    }

    /// Square diagonal matrix with the given entries.
    pub fn diagonal(d: &[f64]) -> Self {
        let n = d.len();
        Self::from_rows(
            n,
            n,
            d.iter().enumerate().map(|(i, &v)| vec![(i, v)]).collect(),
        )
    }

    /// The `n x n` identity — `diagonal` at `1.0`, named because that is how the call sites read.
    pub fn identity(n: usize) -> Self {
        Self::diagonal(&vec![1.0; n])
    }

    pub fn nrows(&self) -> usize {
        self.nrows
    }

    pub fn ncols(&self) -> usize {
        self.ncols
    }

    pub fn nnz(&self) -> usize {
        self.data.len()
    }

    pub fn indptr(&self) -> &[usize] {
        &self.indptr
    }

    pub fn indices(&self) -> &[usize] {
        &self.indices
    }

    pub fn data(&self) -> &[f64] {
        &self.data
    }

    /// True if every row's columns ascend — SciPy's attribute of the same name, computed.
    ///
    /// Always true of a matrix this crate assembled, so asking it of one is a way of saying where
    /// the matrix came from. The one source of a `false` is
    /// [`Csr::from_arrays_preserving_order`].
    pub fn has_sorted_indices(&self) -> bool {
        (0..self.nrows).all(|i| {
            self.indices[self.indptr[i]..self.indptr[i + 1]]
                .windows(2)
                .all(|w| w[0] < w[1])
        })
    }

    /// `self * s`, elementwise on the stored values.
    ///
    /// Structure is preserved even if a product underflows to zero, matching SciPy's
    /// `_mul_scalar`, which multiplies `.data` and leaves the sparsity pattern alone.
    pub fn scaled(&self, s: f64) -> Self {
        Self {
            nrows: self.nrows,
            ncols: self.ncols,
            indptr: self.indptr.clone(),
            indices: self.indices.clone(),
            data: self.data.iter().map(|v| v * s).collect(),
        }
    }

    /// `self - other`, over the union of the two sparsity patterns.
    ///
    /// SciPy's `csr - csr` computes `a - b` at every position either operand occupies, treating a
    /// missing entry as `0.0`, and drops results that are exactly zero. Both are reproduced: the
    /// arithmetic because `0.0 - b` is what a missing left entry contributes, the zero-dropping
    /// because `nnz` is compared against SciPy's in the parity tests.
    ///
    /// SciPy picks between two kernels here — a canonical merge when both operands have sorted,
    /// duplicate-free rows and a linked-list merge otherwise — and the two disagree on the *order*
    /// of the output row, not on its contents. This one always produces canonical order, which is
    /// the whole point: see `physsynth/core/portable.py` for why the Python side is sorted to meet
    /// it rather than the other way round.
    ///
    /// # Panics
    /// If the shapes disagree.
    pub fn sub(&self, other: &Csr) -> Self {
        debug_assert!(
            self.has_sorted_indices() && other.has_sorted_indices(),
            "sub merges two rows by walking both in ascending column order, so neither operand may be stored out of order"
        );
        assert_eq!(
            (self.nrows, self.ncols),
            (other.nrows, other.ncols),
            "sub shape mismatch: ({}x{}) - ({}x{})",
            self.nrows,
            self.ncols,
            other.nrows,
            other.ncols
        );
        let mut rows: Vec<Vec<(usize, f64)>> = Vec::with_capacity(self.nrows);
        for i in 0..self.nrows {
            let (mut p, p_end) = (self.indptr[i], self.indptr[i + 1]);
            let (mut q, q_end) = (other.indptr[i], other.indptr[i + 1]);
            let mut row: Vec<(usize, f64)> = Vec::with_capacity((p_end - p) + (q_end - q));
            while p < p_end || q < q_end {
                let ja = if p < p_end {
                    self.indices[p]
                } else {
                    usize::MAX
                };
                let jb = if q < q_end {
                    other.indices[q]
                } else {
                    usize::MAX
                };
                match ja.cmp(&jb) {
                    std::cmp::Ordering::Less => {
                        row.push((ja, self.data[p]));
                        p += 1;
                    }
                    std::cmp::Ordering::Greater => {
                        row.push((jb, 0.0 - other.data[q]));
                        q += 1;
                    }
                    std::cmp::Ordering::Equal => {
                        row.push((ja, self.data[p] - other.data[q]));
                        p += 1;
                        q += 1;
                    }
                }
            }
            rows.push(row);
        }
        // `from_rows` drops the exact zeros, which is SciPy's kernel's behaviour too.
        Self::from_rows(self.nrows, self.ncols, rows)
    }

    /// `self + other`, over the union of the two sparsity patterns — the mirror of `sub`.
    ///
    /// Same contract, and for the same reason: SciPy computes `a + b` at every position either
    /// operand occupies, treats a missing entry as `0.0`, and drops a result that is exactly zero.
    /// The output is canonical whatever order the operands arrived in, which is the property
    /// `physsynth/core/portable.py` sorts the Python side to meet.
    ///
    /// A two-term sum needs no note about association — but the *chain* of them does. The plate's
    /// operators are three- and four-term sums, and Python evaluates `a + b + c` as `(a + b) + c`;
    /// a caller folding them in another order gets a different matrix in the last bit. Every call
    /// site here spells the association out.
    ///
    /// # Panics
    /// If the shapes disagree.
    pub fn add(&self, other: &Csr) -> Self {
        debug_assert!(
            self.has_sorted_indices() && other.has_sorted_indices(),
            "add merges two rows by walking both in ascending column order, so neither operand may be stored out of order"
        );
        assert_eq!(
            (self.nrows, self.ncols),
            (other.nrows, other.ncols),
            "add shape mismatch: ({}x{}) + ({}x{})",
            self.nrows,
            self.ncols,
            other.nrows,
            other.ncols
        );
        let mut rows: Vec<Vec<(usize, f64)>> = Vec::with_capacity(self.nrows);
        for i in 0..self.nrows {
            let (mut p, p_end) = (self.indptr[i], self.indptr[i + 1]);
            let (mut q, q_end) = (other.indptr[i], other.indptr[i + 1]);
            let mut row: Vec<(usize, f64)> = Vec::with_capacity((p_end - p) + (q_end - q));
            while p < p_end || q < q_end {
                let ja = if p < p_end {
                    self.indices[p]
                } else {
                    usize::MAX
                };
                let jb = if q < q_end {
                    other.indices[q]
                } else {
                    usize::MAX
                };
                match ja.cmp(&jb) {
                    std::cmp::Ordering::Less => {
                        row.push((ja, self.data[p]));
                        p += 1;
                    }
                    std::cmp::Ordering::Greater => {
                        row.push((jb, other.data[q]));
                        q += 1;
                    }
                    std::cmp::Ordering::Equal => {
                        row.push((ja, self.data[p] + other.data[q]));
                        p += 1;
                        q += 1;
                    }
                }
            }
            rows.push(row);
        }
        Self::from_rows(self.nrows, self.ncols, rows)
    }

    /// The Kronecker product `self ⊗ other`.
    ///
    /// Block `(i, j)` of the result is `self[i, j] * other`, so entry
    /// `(i * other.nrows + p, j * other.ncols + q)` is `self[i, j] * other[p, q]`. Each output
    /// entry is a **single product** — no reduction, so nothing here depends on an order and the
    /// result is bit-identical to `scipy.sparse.kron(a, b, format="csr")`, which was measured
    /// canonical for every operand this module builds.
    ///
    /// Used to lift the 1-D differences onto the grid, C-order throughout: `kron(iy, dx)`
    /// differentiates along `x` (the *inner* factor) and `kron(dy, ix)` along `y`.
    pub fn kron(&self, other: &Csr) -> Self {
        let nrows = self.nrows * other.nrows;
        let ncols = self.ncols * other.ncols;
        let mut rows: Vec<Vec<(usize, f64)>> = Vec::with_capacity(nrows);
        for i in 0..self.nrows {
            for p in 0..other.nrows {
                let mut row = Vec::new();
                for a in self.indptr[i]..self.indptr[i + 1] {
                    let j = self.indices[a];
                    let va = self.data[a];
                    for b in other.indptr[p]..other.indptr[p + 1] {
                        row.push((j * other.ncols + other.indices[b], va * other.data[b]));
                    }
                }
                rows.push(row);
            }
        }
        Self::from_rows(nrows, ncols, rows)
    }

    /// The column restriction `self[:, keep]` — the columns whose flag is `true`, in order.
    ///
    /// SciPy spells this `a.tocsc()[:, mask]`; the two agree on values trivially (nothing is
    /// computed) and on **order** for a reason worth stating, because order is what this whole
    /// group of functions turns on: dropping columns is a *monotone* renumbering, so a row that
    /// arrived ascending leaves ascending. That matters because the restricted operator is then
    /// transposed and used as the left factor of a Gram product, where the stored order of its
    /// rows *is* the contraction order (plan §27.2).
    ///
    /// # Panics
    /// If `keep` does not have one flag per column.
    pub fn select_columns(&self, keep: &[bool]) -> Self {
        assert_eq!(
            keep.len(),
            self.ncols,
            "select_columns() needs one flag per column, got {} for {} columns",
            keep.len(),
            self.ncols
        );
        let mut renumber = vec![usize::MAX; self.ncols];
        let mut ncols = 0usize;
        for (j, &k) in keep.iter().enumerate() {
            if k {
                renumber[j] = ncols;
                ncols += 1;
            }
        }
        let rows = (0..self.nrows)
            .map(|i| {
                (self.indptr[i]..self.indptr[i + 1])
                    .filter(|&p| keep[self.indices[p]])
                    .map(|p| (renumber[self.indices[p]], self.data[p]))
                    .collect()
            })
            .collect();
        Self::from_rows(self.nrows, ncols, rows)
    }

    /// The transpose, by counting sort — which lands each output row's columns in ascending order
    /// for free, so the result is canonical without a second pass.
    pub fn transpose(&self) -> Self {
        let mut counts = vec![0usize; self.ncols + 1];
        for &j in &self.indices {
            counts[j + 1] += 1;
        }
        for i in 0..self.ncols {
            counts[i + 1] += counts[i];
        }
        let indptr = counts.clone();
        let mut indices = vec![0usize; self.data.len()];
        let mut data = vec![0.0f64; self.data.len()];
        let mut next = counts;
        for i in 0..self.nrows {
            for p in self.indptr[i]..self.indptr[i + 1] {
                let j = self.indices[p];
                let dst = next[j];
                indices[dst] = i;
                data[dst] = self.data[p];
                next[j] = dst + 1;
            }
        }
        Self {
            nrows: self.ncols,
            ncols: self.nrows,
            indptr,
            indices,
            data,
        }
    }

    /// `self @ other`.
    ///
    /// The accumulation order is the load-bearing part. For output row `i` the contracted index
    /// `j` runs over row `i` of `self` in **ascending** order, and within each `j` the column `k`
    /// runs over row `j` of `other` ascending. That is what SciPy's SMMP kernel does, and floating
    /// point makes it part of the definition rather than an implementation detail: `(p + 4p) + p`
    /// and `(p + p) + 4p` are different numbers in general. Checked against SciPy at six grid
    /// sizes, the resulting `data` is bit-identical — which is why the parity test asserts equality
    /// rather than a tolerance.
    ///
    /// Entries that cancel to exactly zero are dropped, as SciPy's kernel drops them, so the two
    /// sides agree on `nnz` as well as on the values.
    ///
    /// # Panics
    /// If the inner dimensions disagree.
    pub fn matmul(&self, other: &Csr) -> Self {
        debug_assert!(
            self.has_sorted_indices() && other.has_sorted_indices(),
            "matmul accumulates in ascending order of the contracted index, which is a claim about the STORED order of the left operand's rows"
        );
        assert_eq!(
            self.ncols, other.nrows,
            "matmul shape mismatch: ({}x{}) @ ({}x{})",
            self.nrows, self.ncols, other.nrows, other.ncols
        );
        let ncols = other.ncols;
        let mut sums = vec![0.0f64; ncols];
        let mut seen = vec![false; ncols];
        let mut touched: Vec<usize> = Vec::new();
        let mut rows: Vec<Vec<(usize, f64)>> = Vec::with_capacity(self.nrows);
        for i in 0..self.nrows {
            touched.clear();
            for p in self.indptr[i]..self.indptr[i + 1] {
                let j = self.indices[p];
                let v = self.data[p];
                for q in other.indptr[j]..other.indptr[j + 1] {
                    let k = other.indices[q];
                    sums[k] += v * other.data[q];
                    if !seen[k] {
                        seen[k] = true;
                        touched.push(k);
                    }
                }
            }
            touched.sort_unstable();
            let mut row = Vec::with_capacity(touched.len());
            for &k in &touched {
                if sums[k] != 0.0 {
                    row.push((k, sums[k]));
                }
                sums[k] = 0.0;
                seen[k] = false;
            }
            rows.push(row);
        }
        Self::from_rows(self.nrows, ncols, rows)
    }

    /// `self @ v` — for the native tests, which check an operator by what it does to a vector.
    ///
    /// # Panics
    /// If `v` does not have `ncols` entries.
    pub fn matvec(&self, v: &[f64]) -> Vec<f64> {
        assert_eq!(v.len(), self.ncols, "matvec length mismatch");
        (0..self.nrows)
            .map(|i| {
                let mut acc = 0.0;
                for p in self.indptr[i]..self.indptr[i + 1] {
                    acc += self.data[p] * v[self.indices[p]];
                }
                acc
            })
            .collect()
    }

    /// The stored value at `(i, j)`, or `0.0` — a test convenience, linear in the row's width.
    pub fn get(&self, i: usize, j: usize) -> f64 {
        for p in self.indptr[i]..self.indptr[i + 1] {
            if self.indices[p] == j {
                return self.data[p];
            }
        }
        0.0
    }

    /// True if the matrix equals its own transpose exactly — structure and values, no tolerance.
    pub fn is_symmetric(&self) -> bool {
        self.nrows == self.ncols && *self == self.transpose()
    }

    /// Block-diagonal stacking — `scipy.sparse.block_diag`.
    ///
    /// `string_geometric` builds three of these (`A3`, `Gp3`, `Gm3`), and each is the same block
    /// repeated or three siblings side by side. The blocks keep their own row order, so a matvec
    /// against the result sums each output row over that block's entries alone and no reduction
    /// crosses a block — which is why permuting the *global* unknown order (see
    /// [`Csr::permute_symmetric`]) cannot move a single sum.
    pub fn block_diag(blocks: &[&Csr]) -> Self {
        let nrows: usize = blocks.iter().map(|b| b.nrows()).sum();
        let ncols: usize = blocks.iter().map(|b| b.ncols()).sum();
        let mut rows: Vec<Vec<(usize, f64)>> = Vec::with_capacity(nrows);
        let mut col_off = 0usize;
        for b in blocks {
            for i in 0..b.nrows() {
                rows.push(
                    (b.indptr()[i]..b.indptr()[i + 1])
                        .map(|p| (b.indices()[p] + col_off, b.data()[p]))
                        .collect(),
                );
            }
            col_off += b.ncols();
        }
        Self::from_rows(nrows, ncols, rows)
    }

    /// The symmetric permutation `A[q][:, q]`, where `q[new] = old`.
    ///
    /// Used only as a **fill-reducing reordering in front of the sparse LU** (§29.2): the diagonal
    /// stays the diagonal, so the solver's diagonal-preferring pivot means the same thing on the
    /// permuted matrix as on the original.
    ///
    /// # Panics
    /// If `q` is not a permutation of `0..n` on a square matrix.
    pub fn permute_symmetric(&self, q: &[usize]) -> Self {
        let n = self.nrows;
        assert_eq!(self.ncols, n, "permute_symmetric needs a square matrix");
        assert_eq!(q.len(), n, "the permutation must have one entry per row");
        let mut qinv = vec![usize::MAX; n];
        for (new, &old) in q.iter().enumerate() {
            assert!(old < n, "permutation entry {old} out of range for {n} rows");
            assert!(qinv[old] == usize::MAX, "permutation entry {old} repeated");
            qinv[old] = new;
        }
        let rows = q
            .iter()
            .map(|&old| {
                (self.indptr[old]..self.indptr[old + 1])
                    .map(|p| (qinv[self.indices[p]], self.data[p]))
                    .collect()
            })
            .collect();
        Self::from_rows(n, n, rows)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The `[1, -2, 1]` tridiagonal, unscaled — a stand-in for the real operator.
    fn tri(n: usize) -> Csr {
        Csr::from_rows(
            n,
            n,
            (0..n)
                .map(|i| {
                    let mut row = vec![(i, -2.0)];
                    if i > 0 {
                        row.push((i - 1, 1.0));
                    }
                    if i + 1 < n {
                        row.push((i + 1, 1.0));
                    }
                    row
                })
                .collect(),
        )
    }

    #[test]
    fn from_rows_sorts_and_drops_exact_zeros() {
        let m = Csr::from_rows(1, 4, vec![vec![(3, 1.0), (1, 0.0), (0, 2.0)]]);
        assert_eq!(m.indices(), &[0, 3]);
        assert_eq!(m.data(), &[2.0, 1.0]);
        assert_eq!(m.indptr(), &[0, 2]);
    }

    #[test]
    fn transpose_is_an_involution() {
        let m = Csr::from_rows(2, 3, vec![vec![(2, 5.0), (0, 1.0)], vec![(1, -3.0)]]);
        let t = m.transpose();
        assert_eq!((t.nrows(), t.ncols()), (3, 2));
        assert_eq!(t.get(2, 0), 5.0);
        assert_eq!(t.get(1, 1), -3.0);
        assert_eq!(t.transpose(), m);
    }

    #[test]
    fn matmul_agrees_with_the_dense_definition() {
        let a = tri(6);
        let b = a.matmul(&a);
        for i in 0..6 {
            for j in 0..6 {
                let dense: f64 = (0..6).map(|k| a.get(i, k) * a.get(k, j)).sum();
                assert_eq!(b.get(i, j), dense, "entry ({i},{j})");
            }
        }
    }

    #[test]
    fn matmul_drops_entries_that_cancel_exactly() {
        // `[[1, 1]] @ [[1], [-1]] = [[0]]` — SciPy's kernel emits nothing for that entry, and
        // neither does this one. None of the three operator builders exercises the branch, so it
        // is exercised here rather than being an untested line that first matters at Phase 5.
        let a = Csr::from_rows(1, 2, vec![vec![(0, 1.0), (1, 1.0)]]);
        let b = Csr::from_rows(2, 1, vec![vec![(0, 1.0)], vec![(0, -1.0)]]);
        let c = a.matmul(&b);
        assert_eq!(c.nnz(), 0);
        assert_eq!(c.get(0, 0), 0.0);
    }

    #[test]
    fn a_gram_product_is_symmetric_exactly() {
        let a = Csr::from_rows(
            2,
            3,
            vec![vec![(0, 1.0), (1, -2.0)], vec![(1, 1.0), (2, 4.0)]],
        );
        assert!(a.transpose().matmul(&a).is_symmetric());
    }

    #[test]
    fn scaling_preserves_structure() {
        let m = tri(4);
        let s = m.scaled(0.5);
        assert_eq!(s.indptr(), m.indptr());
        assert_eq!(s.indices(), m.indices());
        assert_eq!(s.get(1, 1), -1.0);
    }

    #[test]
    fn addition_is_over_the_union_and_drops_exact_cancellations() {
        let a = Csr::from_rows(2, 3, vec![vec![(0, 1.0), (2, 3.0)], vec![(1, 5.0)]]);
        let b = Csr::from_rows(2, 3, vec![vec![(1, 2.0), (2, -3.0)], vec![(0, 7.0)]]);
        let s = a.add(&b);
        // Row 0: 1 from a alone, 2 from b alone, and 3 + (-3) which cancels and is not stored --
        // SciPy's kernel drops it too, and the parity tests compare nnz.
        assert_eq!(s.indices(), &[0, 1, 0, 1]);
        assert_eq!(s.data(), &[1.0, 2.0, 7.0, 5.0]);
        assert_eq!(s.nnz(), 4);
    }

    #[test]
    fn addition_agrees_with_subtracting_the_negation() {
        let a = tri(6);
        let b = a.scaled(0.375);
        assert_eq!(a.add(&b).data(), a.sub(&b.scaled(-1.0)).data());
    }

    #[test]
    fn the_kronecker_product_places_scaled_blocks() {
        let a = Csr::from_rows(2, 2, vec![vec![(0, 2.0), (1, 3.0)], vec![(1, 5.0)]]);
        let b = Csr::from_rows(2, 2, vec![vec![(0, 7.0)], vec![(0, 11.0), (1, 13.0)]]);
        let k = a.kron(&b);
        assert_eq!((k.nrows(), k.ncols()), (4, 4));
        for i in 0..2 {
            for j in 0..2 {
                for p in 0..2 {
                    for q in 0..2 {
                        assert_eq!(
                            k.get(i * 2 + p, j * 2 + q),
                            a.get(i, j) * b.get(p, q),
                            "block ({i},{j}) entry ({p},{q})"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn the_identity_is_the_multiplicative_one() {
        let a = tri(5);
        assert_eq!(Csr::identity(5).matmul(&a).data(), a.data());
        assert_eq!(a.matmul(&Csr::identity(5)).data(), a.data());
        assert_eq!(Csr::identity(3).kron(&a).nnz(), 3 * a.nnz());
    }

    #[test]
    fn matvec_matches_get() {
        let m = tri(5);
        let v = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = m.matvec(&v);
        for (i, yi) in y.iter().enumerate() {
            let expect: f64 = (0..5).map(|j| m.get(i, j) * v[j]).sum();
            assert_eq!(*yi, expect);
        }
    }

    // -- from_arrays_preserving_order: the one constructor that does not sort ------------------

    #[test]
    fn a_row_stored_out_of_order_comes_back_out_of_order() {
        let m =
            Csr::from_arrays_preserving_order(1, 3, vec![0, 3], vec![2, 0, 1], vec![1.0, 2.0, 3.0])
                .expect("a well-formed triple");
        assert_eq!(
            m.indices(),
            &[2, 0, 1],
            "the constructor sorted, which is its one job not to"
        );
        assert_eq!(m.data(), &[1.0, 2.0, 3.0]);
        assert!(!m.has_sorted_indices());
        // Same operator either way: `get` finds a column wherever it is stored.
        assert_eq!((m.get(0, 0), m.get(0, 1), m.get(0, 2)), (2.0, 3.0, 1.0));
    }

    #[test]
    fn stored_order_is_the_summation_order_and_it_is_visible_in_the_last_bit() {
        // The whole reason the constructor exists: two matrices holding the SAME numbers whose
        // matvec differs, because 1.0 + t + t loses both small terms while t + t + 1.0 keeps them.
        let t = 1e-16;
        let unsorted =
            Csr::from_arrays_preserving_order(1, 3, vec![0, 3], vec![2, 0, 1], vec![1.0, t, t])
                .expect("a well-formed triple");
        let sorted = Csr::from_rows(1, 3, vec![vec![(0, t), (1, t), (2, 1.0)]]);
        assert!(sorted.has_sorted_indices() && !unsorted.has_sorted_indices());
        let v = [1.0, 1.0, 1.0];
        assert_eq!(unsorted.matvec(&v)[0], 1.0);
        assert_ne!(
            sorted.matvec(&v)[0],
            unsorted.matvec(&v)[0],
            "the two orders agree, so this fixture no longer exercises what it is here for"
        );
        // ... and the difference is exactly the last bit, not a value.
        assert_eq!(sorted.matvec(&v)[0], 1.0 + f64::EPSILON);
    }

    #[test]
    fn structural_zeros_are_kept_where_from_rows_would_drop_them() {
        // A caller handing over an existing matrix is handing over its `nnz`, so a stored 0.0
        // stays stored — the opposite of `from_rows`, and `nnz` is compared against SciPy's.
        let m = Csr::from_arrays_preserving_order(1, 2, vec![0, 2], vec![0, 1], vec![0.0, 1.0])
            .expect("a well-formed triple");
        assert_eq!(m.nnz(), 2);
        assert_eq!(
            Csr::from_rows(1, 2, vec![vec![(0, 0.0), (1, 1.0)]]).nnz(),
            1
        );
    }

    #[test]
    fn a_malformed_triple_is_an_error_and_not_a_panic() {
        use RawCsrError::*;
        /// The rejection, then the `indptr`, `indices` and `data` that earn it.
        type Malformed = (RawCsrError, Vec<usize>, Vec<usize>, Vec<f64>);
        let cases: Vec<Malformed> = vec![
            (
                IndptrLength {
                    expected: 2,
                    got: 3,
                },
                vec![0, 1, 1],
                vec![0],
                vec![1.0],
            ),
            (
                LengthMismatch {
                    indices: 2,
                    data: 1,
                },
                vec![0, 1],
                vec![0, 1],
                vec![1.0],
            ),
            (
                IndptrNotRowStarts { row: 0 },
                vec![0, 2],
                vec![0],
                vec![1.0],
            ),
            (
                ColumnOutOfRange {
                    column: 7,
                    ncols: 3,
                },
                vec![0, 1],
                vec![7],
                vec![1.0],
            ),
            (
                DuplicateColumn { row: 0, column: 1 },
                vec![0, 2],
                vec![1, 1],
                vec![1.0, 2.0],
            ),
        ];
        for (want, indptr, indices, data) in cases {
            let got = Csr::from_arrays_preserving_order(1, 3, indptr, indices, data);
            assert_eq!(got.err(), Some(want), "wrong rejection for {want}");
        }
    }

    #[test]
    fn an_unsorted_row_is_still_a_matrix_the_accessors_agree_about() {
        // The order-agnostic half of the surface, asserted so the doc comment's list is checked
        // rather than believed: everything here must read the same as the sorted twin.
        let t = 1e-16;
        let unsorted =
            Csr::from_arrays_preserving_order(1, 3, vec![0, 3], vec![2, 0, 1], vec![1.0, t, t])
                .expect("a well-formed triple");
        let sorted = Csr::from_rows(1, 3, vec![vec![(0, t), (1, t), (2, 1.0)]]);
        assert_eq!(unsorted.nnz(), sorted.nnz());
        assert_eq!(
            (unsorted.nrows(), unsorted.ncols()),
            (sorted.nrows(), sorted.ncols())
        );
        for j in 0..3 {
            assert_eq!(unsorted.get(0, j), sorted.get(0, j));
        }
        let s = unsorted.scaled(2.0);
        assert_eq!(s.indices(), unsorted.indices(), "scaling reordered a row");
        assert!(!s.has_sorted_indices());
    }
}
