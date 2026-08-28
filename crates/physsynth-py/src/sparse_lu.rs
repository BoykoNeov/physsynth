//! The binding over `physsynth_core::sparse_lu` — the Group D solver (plan §24).
//!
//! Shaped like `scipy.sparse.linalg.splu`'s return value rather than like a model: a factorization
//! object with one `solve` method, and no state of the caller's held here.
//!
//! # Why this is exposed at all, when `beam` never calls it from Python
//!
//! The Rust beam factors internally and this class is not on its path. It exists so
//! `tests/test_rust_parity_beam.py` can put the *Python* beam on the *Rust* solver for the length
//! of a block, and so make a comparison in which the solver is held constant and only the model
//! varies. That is `test_rust_parity_strings.py`'s `shared_solver()` manoeuvre, and it is worth
//! more here than it was there: without it, every difference between the two beams is confounded
//! with the SuperLU gap §24.2 measured, and a genuine porting error would be invisible under it —
//! which is §19.4's finding (a real bug that the trajectory could not see) waiting to happen.
//!
//! The matrix arrives as CSR triplets rather than as a SciPy object, so nothing here imports
//! SciPy and the class stays usable from a bare interpreter.

use numpy::{PyArray1, PyReadonlyArray1};
use physsynth_core::sparse::Csr;
use physsynth_core::sparse_lu::{SparseLu, SparseLuError};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn lu_err(e: SparseLuError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// A factored sparse matrix — `scipy.sparse.linalg.splu`'s object, from Rust.
#[pyclass(name = "SparseLu", module = "physsynth_rs")]
pub struct PySparseLu {
    lu: SparseLu,
}

impl PySparseLu {
    /// Wrap an already-factored matrix — what `Plate._lu` hands out.
    ///
    /// Private by name and public by use, for a third time (§12.2): `test_plate_connection.py`
    /// reads `plate._lu.solve(...)` to check that a bridge's coupling force reached the
    /// acceleration, and `airbox.py` reassembles the same matrix rather than reaching in. Cloning
    /// copies the factors; it does not factor again.
    pub(crate) fn from_core(lu: SparseLu) -> Self {
        PySparseLu { lu }
    }
}

#[pymethods]
impl PySparseLu {
    /// Factor a square matrix given as CSR triplets (`data`, `indices`, `indptr`).
    #[new]
    fn new(
        data: PyReadonlyArray1<'_, f64>,
        indices: PyReadonlyArray1<'_, i32>,
        indptr: PyReadonlyArray1<'_, i32>,
        n: usize,
    ) -> PyResult<Self> {
        let data = data
            .as_slice()
            .map_err(|_| PyValueError::new_err("data must be a contiguous 1-D float64 array."))?;
        let indices = indices
            .as_slice()
            .map_err(|_| PyValueError::new_err("indices must be a contiguous 1-D int32 array."))?;
        let indptr = indptr
            .as_slice()
            .map_err(|_| PyValueError::new_err("indptr must be a contiguous 1-D int32 array."))?;
        if indptr.len() != n + 1 {
            return Err(PyValueError::new_err(format!(
                "indptr must have {} entries for an order-{n} matrix, got {}.",
                n + 1,
                indptr.len()
            )));
        }
        let rows: Vec<Vec<(usize, f64)>> = (0..n)
            .map(|i| {
                (indptr[i] as usize..indptr[i + 1] as usize)
                    .map(|p| (indices[p] as usize, data[p]))
                    .collect()
            })
            .collect();
        let lu = SparseLu::factor(&Csr::from_rows(n, n, rows)).map_err(lu_err)?;
        Ok(PySparseLu { lu })
    }

    /// `A x = b` — the per-timestep back-substitution, matching `splu(...).solve`.
    fn solve<'py>(
        &self,
        py: Python<'py>,
        b: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Py<PyArray1<f64>>> {
        let b = b
            .as_slice()
            .map_err(|_| PyValueError::new_err("b must be a contiguous 1-D float64 array."))?;
        let x = self.lu.solve(b).map_err(lu_err)?;
        Ok(PyArray1::from_vec(py, x).unbind())
    }

    /// Whether the row permutation is the identity — SciPy's `perm_r == perm_c`, from this side.
    #[getter]
    fn is_natural(&self) -> bool {
        self.lu.is_natural()
    }

    /// `(nnz(L), nnz(U))`, `L`'s unit diagonal excluded. The fill, for a test to look at.
    #[getter]
    fn nnz(&self) -> (usize, usize) {
        self.lu.nnz()
    }
}
