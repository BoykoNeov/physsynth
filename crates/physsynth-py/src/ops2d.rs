//! The binding over `physsynth_core::ops2d` — the builder half of `operators2d.py`.
//!
//! Same shape as Phase 1's operator binding: pointwise things return fresh NumPy arrays, and the
//! one matrix comes back as **CSR triplets**, `(data, indices, indptr, shape)`, which the shim at
//! the bottom of `physsynth/core/operators2d.py` rebuilds into a `scipy.sparse.csr_matrix`. The
//! core never learns what SciPy is; the modules that `from .operators2d import ...` never learn
//! what Rust is.
//!
//! `laplacian_from_mask` returns a **pair** — the matrix and the index map — so its binding
//! returns a pair too, with only the first half in triplet form. That asymmetry is the honest
//! shape of the original and is better than inventing a wrapper object for it.

use crate::shape::{as_bool_field, as_f64_field, shape_repr, to_2d_bool, to_2d_f64};
use crate::{csr_triplets, CsrTriplets};
use physsynth_core::ops2d::{self, Mask};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Read a 2-D boolean mask argument into the core's `Mask`.
fn mask_arg(py: Python<'_>, obj: &Bound<'_, PyAny>, name: &str) -> PyResult<Mask> {
    let (shape, flags) = as_bool_field(py, obj, name)?;
    if shape.len() != 2 {
        return Err(PyValueError::new_err(format!(
            "{name} must be a 2-D boolean array, got shape {}.",
            shape_repr(&shape)
        )));
    }
    Ok(Mask::new(shape[0], shape[1], flags))
}

/// Square grid of `N+1` nodes per axis over `[-half_extent, half_extent]^2`; returns `(X, Y, h)`.
#[pyfunction]
#[pyo3(name = "grid_coords")]
pub fn py_grid_coords(
    py: Python<'_>,
    N: i64,
    half_extent: f64,
) -> PyResult<(Py<PyAny>, Py<PyAny>, f64)> {
    if N < 1 {
        return Err(PyValueError::new_err(format!(
            "N must be >= 1 (need at least one cell); got {N}."
        )));
    }
    let n = N as usize;
    let (x, y, h) = ops2d::grid_coords(n, half_extent);
    Ok((
        to_2d_f64(py, x, n + 1, n + 1)?,
        to_2d_f64(py, y, n + 1, n + 1)?,
        h,
    ))
}

/// Live-node mask for a rectangle: every interior node of an `(Ny+1) x (Nx+1)` grid.
#[pyfunction]
#[pyo3(name = "rectangle_mask")]
pub fn py_rectangle_mask(py: Python<'_>, Nx: i64, Ny: i64) -> PyResult<Py<PyAny>> {
    if Nx < 0 || Ny < 0 {
        return Err(PyValueError::new_err(
            "Nx and Ny must be non-negative cell counts.",
        ));
    }
    let m = ops2d::rectangle_mask(Nx as usize, Ny as usize);
    let (nrows, ncols) = (m.nrows(), m.ncols());
    to_2d_bool(py, m.flags().to_vec(), nrows, ncols)
}

/// Live-node mask for a disk of `radius` centred at the origin on the grid `(X, Y)`.
#[pyfunction]
#[pyo3(name = "disk_mask")]
pub fn py_disk_mask(
    py: Python<'_>,
    X: &Bound<'_, PyAny>,
    Y: &Bound<'_, PyAny>,
    radius: f64,
) -> PyResult<Py<PyAny>> {
    let (xshape, xs) = as_f64_field(py, X, "X")?;
    let (yshape, ys) = as_f64_field(py, Y, "Y")?;
    if xshape.len() != 2 || xshape != yshape {
        return Err(PyValueError::new_err(format!(
            "X and Y must be 2-D arrays of the same shape; got {} and {}.",
            shape_repr(&xshape),
            shape_repr(&yshape)
        )));
    }
    let m = ops2d::disk_mask(&xs, &ys, radius, xshape[0], xshape[1]);
    to_2d_bool(py, m.flags().to_vec(), xshape[0], xshape[1])
}

/// Symmetric 5-point Laplacian on the live nodes, as `(csr_triplets, index_map)`.
#[pyfunction]
#[pyo3(name = "laplacian_from_mask_csr")]
pub fn py_laplacian_from_mask(
    py: Python<'_>,
    mask: &Bound<'_, PyAny>,
    h: f64,
) -> PyResult<(CsrTriplets, Py<PyAny>)> {
    let m = mask_arg(py, mask, "mask")?;
    let (nrows, ncols) = (m.nrows(), m.ncols());
    let (l, index_map) = ops2d::laplacian_from_mask(&m, h);
    Ok((
        csr_triplets(py, &l)?,
        crate::shape::to_2d_i64(py, index_map, nrows, ncols)?,
    ))
}

/// Scatter a flat live-node vector back onto the full 2-D grid (zeros at dead nodes).
#[pyfunction]
#[pyo3(name = "embed")]
pub fn py_embed(
    py: Python<'_>,
    values: &Bound<'_, PyAny>,
    index_map: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let (vshape, vals) = as_f64_field(py, values, "values")?;
    if vshape.len() != 1 {
        return Err(PyValueError::new_err(format!(
            "values must be a 1-D live-node vector, got shape {}.",
            shape_repr(&vshape)
        )));
    }
    // The map arrives as float only because it shares the reader; the entries are whole numbers.
    let (mshape, map_f) = as_f64_field(py, index_map, "index_map")?;
    if mshape.len() != 2 {
        return Err(PyValueError::new_err(format!(
            "index_map must be 2-D, got shape {}.",
            shape_repr(&mshape)
        )));
    }
    let map: Vec<i64> = map_f.iter().map(|&v| v as i64).collect();
    let n_live = vals.len() as i64;
    if map.iter().any(|&p| p >= n_live) {
        return Err(PyValueError::new_err(
            "index_map names a live index that values does not have.",
        ));
    }
    let field = ops2d::embed(&vals, &map);
    to_2d_f64(py, field, mshape[0], mshape[1])
}

/// Discrete 2-D inner product `<f, g> = h^2 * sum f g`.
#[pyfunction]
#[pyo3(name = "inner2d")]
pub fn py_inner2d(
    py: Python<'_>,
    f: &Bound<'_, PyAny>,
    g: &Bound<'_, PyAny>,
    h: f64,
) -> PyResult<f64> {
    let (_, a) = as_f64_field(py, f, "f")?;
    let (_, b) = as_f64_field(py, g, "g")?;
    if a.len() != b.len() {
        return Err(PyValueError::new_err(format!(
            "inner2d() operands must have equal size; got {} and {}.",
            a.len(),
            b.len()
        )));
    }
    Ok(ops2d::inner2d(&a, &b, h))
}

/// Squared discrete 2-D norm `||f||^2 = <f, f>`.
#[pyfunction]
#[pyo3(name = "norm2_2d")]
pub fn py_norm2_2d(py: Python<'_>, f: &Bound<'_, PyAny>, h: f64) -> PyResult<f64> {
    let (_, a) = as_f64_field(py, f, "f")?;
    Ok(ops2d::norm2_2d(&a, h))
}
