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

use crate::shape::{as_bool_field, as_f64_field, shape_repr, to_2d_bool, to_2d_f64, to_2d_i64};
use crate::string_stiff::csr_object;
use crate::{csr_triplets, CsrTriplets};
use numpy::PyArray1;
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

/// Un-normalised half-width profile of the guitar outline at `t = y/L` in `[0, 1]`.
///
/// Vectorised over `t` like the original, and — like the original since Phase 5 — evaluated one
/// point at a time so that both languages reach the same libm.
#[pyfunction]
#[pyo3(name = "guitar_half_width", signature = (t, waist=0.42, asym=0.30))]
pub fn py_guitar_half_width(
    py: Python<'_>,
    t: &Bound<'_, PyAny>,
    waist: f64,
    asym: f64,
) -> PyResult<Py<PyAny>> {
    let (shape, ts) = as_f64_field(py, t, "t")?;
    let w: Vec<f64> = ts
        .iter()
        .map(|&tv| ops2d::guitar_half_width(tv, waist, asym))
        .collect();
    crate::shape::to_shaped_f64(py, w, &shape)
}

/// Factor taking the half-width profile to a maximum of `width/2`.
#[pyfunction]
#[pyo3(name = "guitar_scale")]
pub fn py_guitar_scale(width: f64, waist: f64, asym: f64) -> f64 {
    ops2d::guitar_scale(width, waist, asym)
}

/// Live-node mask for a guitar-shaped outline.
#[pyfunction]
#[pyo3(name = "guitar_mask", signature = (X, Y, length, width, waist=0.42, asym=0.30))]
#[allow(clippy::too_many_arguments)]
pub fn py_guitar_mask(
    py: Python<'_>,
    X: &Bound<'_, PyAny>,
    Y: &Bound<'_, PyAny>,
    length: f64,
    width: f64,
    waist: f64,
    asym: f64,
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
    // The three refusals are the original's, message for message: a guitar with no length is not
    // a degenerate guitar, and an `asym` past 2 turns the profile negative and folds the outline
    // inside out rather than tilting it.
    if length <= 0.0 || width <= 0.0 {
        return Err(PyValueError::new_err("length and width must be positive."));
    }
    if !(0.0..1.0).contains(&waist) {
        return Err(PyValueError::new_err(format!(
            "waist must lie in [0, 1); got {}.",
            physsynth_core::fmt::py_float(waist)
        )));
    }
    if asym.abs() >= 2.0 {
        return Err(PyValueError::new_err(format!(
            "|asym| must be < 2 (the profile would go negative); got {}.",
            physsynth_core::fmt::py_float(asym)
        )));
    }
    let m = ops2d::guitar_mask(&xs, &ys, length, width, waist, asym, xshape[0], xshape[1]);
    to_2d_bool(py, m.flags().to_vec(), xshape[0], xshape[1])
}

/// Area of the *true* guitar outline (fine midpoint quadrature).
#[pyfunction]
#[pyo3(name = "guitar_area", signature = (length, width, waist=0.42, asym=0.30))]
pub fn py_guitar_area(length: f64, width: f64, waist: f64, asym: f64) -> f64 {
    ops2d::guitar_area(length, width, waist, asym)
}

/// Cells of the dual grid whose four corner nodes are all live.
#[pyfunction]
#[pyo3(name = "live_cells")]
pub fn py_live_cells(py: Python<'_>, mask: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let m = mask_arg(py, mask, "mask")?;
    let (nrows, ncols) = (m.nrows(), m.ncols());
    to_2d_bool(
        py,
        ops2d::live_cells(&m),
        nrows.saturating_sub(1),
        ncols.saturating_sub(1),
    )
}

/// Number of live cells (0..4) touching each node.
#[pyfunction]
#[pyo3(name = "cells_per_node")]
pub fn py_cells_per_node(py: Python<'_>, mask: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let m = mask_arg(py, mask, "mask")?;
    let (nrows, ncols) = (m.nrows(), m.ncols());
    crate::shape::to_2d_i64(py, ops2d::cells_per_node(&m), nrows, ncols)
}

/// Drop live nodes that touch no live cell, to a fixed point; returns `(mask, n_dropped)`.
#[pyfunction]
#[pyo3(name = "prune_to_area_carrying")]
pub fn py_prune_to_area_carrying(
    py: Python<'_>,
    mask: &Bound<'_, PyAny>,
) -> PyResult<(Py<PyAny>, usize)> {
    let m = mask_arg(py, mask, "mask")?;
    let (nrows, ncols) = (m.nrows(), m.ncols());
    let (pruned, dropped) = ops2d::prune_to_area_carrying(&m);
    Ok((
        to_2d_bool(py, pruned.flags().to_vec(), nrows, ncols)?,
        dropped,
    ))
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

/// `B = L @ L` on the live nodes of `mask`; returns `(triplets, index_map)`.
#[pyfunction]
#[pyo3(name = "biharmonic_from_mask_csr")]
pub fn py_biharmonic_from_mask(
    py: Python<'_>,
    mask: &Bound<'_, PyAny>,
    h: f64,
) -> PyResult<(CsrTriplets, Py<PyAny>)> {
    let m = mask_arg(py, mask, "mask")?;
    let (nrows, ncols) = (m.nrows(), m.ncols());
    let (b, index_map) = ops2d::biharmonic_from_mask(&m, h);
    Ok((
        csr_triplets(py, &b)?,
        crate::shape::to_2d_i64(py, index_map, nrows, ncols)?,
    ))
}

/// The `n_int x n_int` interior Dirichlet second difference; returns triplets.
#[pyfunction]
#[pyo3(name = "dirichlet_interior_d2_1d_csr")]
pub fn py_dirichlet_interior_d2_1d(py: Python<'_>, n_int: i64, h: f64) -> PyResult<CsrTriplets> {
    if n_int < 0 {
        return Err(PyValueError::new_err(format!(
            "n_int must be >= 0, got {n_int}."
        )));
    }
    csr_triplets(py, &ops2d::dirichlet_interior_d2_1d(n_int as usize, h))
}

/// The orthotropic simply-supported bending operator; returns `(triplets, index_map)`.
#[pyfunction]
#[pyo3(name = "orthotropic_biharmonic_csr")]
pub fn py_orthotropic_biharmonic(
    py: Python<'_>,
    Nx: i64,
    Ny: i64,
    h: f64,
    grain_x: f64,
    grain_cross: f64,
    grain_y: f64,
) -> PyResult<(CsrTriplets, Py<PyAny>)> {
    if Nx < 2 || Ny < 2 {
        return Err(PyValueError::new_err(format!(
            "Nx and Ny must both be >= 2, got ({Nx}, {Ny})."
        )));
    }
    let (nx, ny) = (Nx as usize, Ny as usize);
    let (b, index_map) = ops2d::orthotropic_biharmonic(nx, ny, h, grain_x, grain_cross, grain_y);
    Ok((
        csr_triplets(py, &b)?,
        crate::shape::to_2d_i64(py, index_map, ny + 1, nx + 1)?,
    ))
}

/// Validate `nu` exactly where the reference does — only where it supplies a missing half.
fn check_nu(nu: f64, grain_coupling: Option<f64>, grain_torsion: Option<f64>) -> PyResult<()> {
    if (grain_coupling.is_none() || grain_torsion.is_none()) && !(-1.0 < nu && nu < 0.5) {
        return Err(PyValueError::new_err(format!(
            "nu (Poisson's ratio) must be in (-1, 1/2), got {}.",
            physsynth_core::fmt::py_float(nu)
        )));
    }
    Ok(())
}

/// The free-edge stiffness on an arbitrary outline; returns `(K, W, index_map)`.
#[pyfunction]
#[pyo3(name = "free_plate_stiffness_from_mask_csr")]
#[allow(clippy::too_many_arguments)]
pub fn py_free_plate_stiffness_from_mask(
    py: Python<'_>,
    mask: &Bound<'_, PyAny>,
    h: f64,
    nu: f64,
    grain_x: f64,
    grain_y: f64,
    grain_coupling: Option<f64>,
    grain_torsion: Option<f64>,
) -> PyResult<(CsrTriplets, CsrTriplets, Py<PyAny>)> {
    let m = mask_arg(py, mask, "mask")?;
    check_nu(nu, grain_coupling, grain_torsion)?;
    if m.n_live() < 1 {
        return Err(PyValueError::new_err("the mask has no live nodes."));
    }
    let (nrows, ncols) = (m.nrows(), m.ncols());
    let (k, w, index_map) = ops2d::free_plate_stiffness_from_mask(
        &m,
        h,
        nu,
        grain_x,
        grain_y,
        grain_coupling,
        grain_torsion,
    );
    Ok((
        csr_triplets(py, &k)?,
        csr_triplets(py, &w)?,
        crate::shape::to_2d_i64(py, index_map, nrows, ncols)?,
    ))
}

/// The free-edge stiffness on a full bounding box; returns `(K, W, index_map)`.
#[pyfunction]
#[pyo3(name = "free_plate_stiffness_csr")]
#[allow(clippy::too_many_arguments)]
pub fn py_free_plate_stiffness(
    py: Python<'_>,
    Nx: i64,
    Ny: i64,
    h: f64,
    nu: f64,
    grain_x: f64,
    grain_y: f64,
    grain_coupling: Option<f64>,
    grain_torsion: Option<f64>,
) -> PyResult<(CsrTriplets, CsrTriplets, Py<PyAny>)> {
    if Nx < 2 || Ny < 2 {
        return Err(PyValueError::new_err(
            "Nx, Ny must be >= 2 (need at least one interior node per axis).",
        ));
    }
    check_nu(nu, grain_coupling, grain_torsion)?;
    let (nx, ny) = (Nx as usize, Ny as usize);
    let (k, w, index_map) = ops2d::free_plate_stiffness(
        nx,
        ny,
        h,
        nu,
        grain_x,
        grain_y,
        grain_coupling,
        grain_torsion,
    );
    Ok((
        csr_triplets(py, &k)?,
        csr_triplets(py, &w)?,
        crate::shape::to_2d_i64(py, index_map, ny + 1, nx + 1)?,
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

// --- the nonlinear plate ------------------------------------------------------------------------
//
// Two classes and the five 1-D differences they are built from. The differences come back as CSR
// triplets like every other matrix here; the classes are `pyclass`es, because the reference's
// clients hold them (`plate.VKPlate` keeps a bracket and a solver for the life of the model) and
// rebuilding four `kron`s per bracket call would be paying construction cost per timestep.
//
// `Sxx`, `Syy`, `Dxy`, `Acell` and `Bf` are built as SciPy objects **once**, in the constructor,
// for the same reason `beam`'s `K` and `W` are: they are documented attributes that tests hand
// straight to dense solvers, not conveniences for the parity file.

/// Read a full-grid vector argument of the expected length, whatever shape it arrived in.
///
/// The reference writes `np.asarray(a, dtype=float).ravel()`, so a `(ny+1, nx+1)` array and a flat
/// one are the same argument. Only the total length is a contract.
fn full_grid_arg(
    py: Python<'_>,
    obj: &Bound<'_, PyAny>,
    name: &str,
    n_nodes: usize,
) -> PyResult<Vec<f64>> {
    let (shape, values) = as_f64_field(py, obj, name)?;
    if values.len() != n_nodes {
        return Err(PyValueError::new_err(format!(
            "{name} must have {n_nodes} entries (the full node grid), got shape {}.",
            shape_repr(&shape)
        )));
    }
    Ok(values)
}

/// `Nx, Ny must be >= 2 ...` / `h ... must be positive.` — the two refusals both classes share.
fn vk_grid_args(nx: i64, ny: i64, h: f64) -> PyResult<(usize, usize)> {
    if nx < 2 || ny < 2 {
        return Err(PyValueError::new_err(
            "Nx, Ny must be >= 2 (need at least one interior node per axis).",
        ));
    }
    if h <= 0.0 {
        return Err(PyValueError::new_err("h (grid spacing) must be positive."));
    }
    Ok((nx as usize, ny as usize))
}

/// `(n+1) x (n+1)` collocated second difference with zero end rows — `_collocated_d2_1d`.
#[pyfunction]
#[pyo3(name = "collocated_d2_1d_csr")]
pub fn py_collocated_d2_1d(py: Python<'_>, N: i64, h: f64) -> PyResult<CsrTriplets> {
    if N < 1 {
        return Err(PyValueError::new_err("N must be >= 1."));
    }
    csr_triplets(py, &ops2d::collocated_d2_1d(N as usize, h))
}

/// `N x (N+1)` forward first difference on the cell midpoints — `_forward_d1_1d`.
#[pyfunction]
#[pyo3(name = "forward_d1_1d_csr")]
pub fn py_forward_d1_1d(py: Python<'_>, N: i64, h: f64) -> PyResult<CsrTriplets> {
    if N < 1 {
        return Err(PyValueError::new_err("N must be >= 1."));
    }
    csr_triplets(py, &ops2d::forward_d1_1d(N as usize, h))
}

/// `(N+1) x (N+1)` ordinary tridiagonal second difference — `_centered_d2_1d`.
#[pyfunction]
#[pyo3(name = "centered_d2_1d_csr")]
pub fn py_centered_d2_1d(py: Python<'_>, N: i64, h: f64) -> PyResult<CsrTriplets> {
    if N < 1 {
        return Err(PyValueError::new_err("N must be >= 1."));
    }
    csr_triplets(py, &ops2d::centered_d2_1d(N as usize, h))
}

/// `(N+1) x (N+1)` second difference with the clamped ghost mirror — `_clamped_d2_1d`.
#[pyfunction]
#[pyo3(name = "clamped_d2_1d_csr")]
pub fn py_clamped_d2_1d(py: Python<'_>, N: i64, h: f64) -> PyResult<CsrTriplets> {
    if N < 2 {
        return Err(PyValueError::new_err("N must be >= 2."));
    }
    csr_triplets(py, &ops2d::clamped_d2_1d(N as usize, h))
}

/// `N x (N+1)` node-to-cell average — `_avg_d1_1d`. Carries no `h`.
#[pyfunction]
#[pyo3(name = "avg_d1_1d_csr")]
pub fn py_avg_d1_1d(py: Python<'_>, N: i64) -> PyResult<CsrTriplets> {
    if N < 1 {
        return Err(PyValueError::new_err("N must be >= 1."));
    }
    csr_triplets(py, &ops2d::avg_d1_1d(N as usize))
}

/// The discrete von Karman bracket — the Rust implementation, wearing the Python interface.
///
/// Attribute-for-attribute and method-for-method compatible with
/// `physsynth.core.operators2d.VonKarmanBracket`; the docstring on that class is the reference.
#[pyclass(name = "VonKarmanBracket", module = "physsynth_rs")]
pub struct PyVonKarmanBracket {
    inner: ops2d::VonKarmanBracket,
    sxx: Py<PyAny>,
    syy: Py<PyAny>,
    dxy: Py<PyAny>,
    acell: Py<PyAny>,
}

#[pymethods]
impl PyVonKarmanBracket {
    #[new]
    fn new(py: Python<'_>, Nx: i64, Ny: i64, h: f64) -> PyResult<Self> {
        let (nx, ny) = vk_grid_args(Nx, Ny, h)?;
        let inner = ops2d::VonKarmanBracket::new(nx, ny, h);
        Ok(PyVonKarmanBracket {
            sxx: csr_object(py, inner.sxx())?,
            syy: csr_object(py, inner.syy())?,
            dxy: csr_object(py, inner.dxy())?,
            acell: csr_object(py, inner.acell())?,
            inner,
        })
    }

    /// The nodal field `l(a, b)` as a flat full-grid vector.
    fn __call__(
        &self,
        py: Python<'_>,
        a: &Bound<'_, PyAny>,
        b: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyArray1<f64>>> {
        let n = self.inner.n_nodes();
        let av = full_grid_arg(py, a, "a", n)?;
        let bv = full_grid_arg(py, b, "b", n)?;
        Ok(PyArray1::from_vec(py, self.inner.eval(&av, &bv)).unbind())
    }

    /// The trilinear form `T(a, b, c) = <l(a, b), c>`.
    fn trilinear(
        &self,
        py: Python<'_>,
        a: &Bound<'_, PyAny>,
        b: &Bound<'_, PyAny>,
        c: &Bound<'_, PyAny>,
    ) -> PyResult<f64> {
        let n = self.inner.n_nodes();
        let av = full_grid_arg(py, a, "a", n)?;
        let bv = full_grid_arg(py, b, "b", n)?;
        let cv = full_grid_arg(py, c, "c", n)?;
        Ok(self.inner.trilinear(&av, &bv, &cv))
    }

    #[getter]
    fn Nx(&self) -> usize {
        self.inner.nx()
    }

    #[getter]
    fn Ny(&self) -> usize {
        self.inner.ny()
    }

    #[getter]
    fn h(&self) -> f64 {
        self.inner.h()
    }

    #[getter]
    fn n_nodes(&self) -> usize {
        self.inner.n_nodes()
    }

    /// `d_xx` lifted onto the grid, as the `csr_matrix` the original holds. Built once.
    #[getter]
    fn Sxx(&self, py: Python<'_>) -> Py<PyAny> {
        self.sxx.clone_ref(py)
    }

    /// `d_yy` lifted onto the grid, as the `csr_matrix` the original holds. Built once.
    #[getter]
    fn Syy(&self, py: Python<'_>) -> Py<PyAny> {
        self.syy.clone_ref(py)
    }

    /// The cell-centred twist, as the `csr_matrix` the original holds. Built once.
    #[getter]
    fn Dxy(&self, py: Python<'_>) -> Py<PyAny> {
        self.dxy.clone_ref(py)
    }

    /// The node-to-cell corner average, as the `csr_matrix` the original holds. Built once.
    #[getter]
    fn Acell(&self, py: Python<'_>) -> Py<PyAny> {
        self.acell.clone_ref(py)
    }
}

/// The clamped Airy-stress solve — the Rust implementation, wearing the Python interface.
///
/// Attribute-for-attribute and method-for-method compatible with
/// `physsynth.core.operators2d.AiryStressSolver`; the docstring on that class is the reference.
///
/// `Bf` comes back as a `csc_matrix`, which is the format the original stores and the one
/// `scipy.sparse.linalg.splu` wants — a caller that re-factors it should not pay a conversion and
/// a caller that compares formats should not see a difference.
#[pyclass(name = "AiryStressSolver", module = "physsynth_rs")]
pub struct PyAiryStressSolver {
    inner: ops2d::AiryStressSolver,
    bf: Py<PyAny>,
    mask: Py<PyAny>,
    index_map: Py<PyAny>,
}

#[pymethods]
impl PyAiryStressSolver {
    #[new]
    fn new(py: Python<'_>, Nx: i64, Ny: i64, h: f64) -> PyResult<Self> {
        let (nx, ny) = vk_grid_args(Nx, Ny, h)?;
        let inner = ops2d::AiryStressSolver::new(nx, ny, h)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let (nrows, ncols) = (inner.mask().nrows(), inner.mask().ncols());
        let bf = csr_object(py, inner.bf())?
            .bind(py)
            .call_method0("tocsc")?
            .unbind();
        Ok(PyAiryStressSolver {
            bf,
            mask: to_2d_bool(py, inner.mask().flags().to_vec(), nrows, ncols)?,
            index_map: to_2d_i64(py, inner.index_map().to_vec(), nrows, ncols)?,
            inner,
        })
    }

    /// Solve for `F` (full-grid in, full-grid out, rim held at zero).
    fn solve(&self, py: Python<'_>, source: &Bound<'_, PyAny>) -> PyResult<Py<PyArray1<f64>>> {
        let s = full_grid_arg(py, source, "source", self.inner.n_nodes())?;
        let f = self
            .inner
            .solve(&s)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyArray1::from_vec(py, f).unbind())
    }

    /// Discrete `||lap F||^2 = F^T B_F F` for a full-grid `F`.
    fn laplacian_norm_sq(&self, py: Python<'_>, F: &Bound<'_, PyAny>) -> PyResult<f64> {
        let f = full_grid_arg(py, F, "F", self.inner.n_nodes())?;
        Ok(self.inner.laplacian_norm_sq(&f))
    }

    #[getter]
    fn Nx(&self) -> usize {
        self.inner.nx()
    }

    #[getter]
    fn Ny(&self) -> usize {
        self.inner.ny()
    }

    #[getter]
    fn h(&self) -> f64 {
        self.inner.h()
    }

    #[getter]
    fn n_nodes(&self) -> usize {
        self.inner.n_nodes()
    }

    #[getter]
    fn n_interior(&self) -> usize {
        self.inner.n_interior()
    }

    /// The interior mask, as the 2-D boolean array the original holds.
    #[getter]
    fn mask(&self, py: Python<'_>) -> Py<PyAny> {
        self.mask.clone_ref(py)
    }

    /// Full-grid node to interior index, `-1` on the rim; the 2-D array the original holds.
    #[getter]
    fn index_map(&self, py: Python<'_>) -> Py<PyAny> {
        self.index_map.clone_ref(py)
    }

    /// The assembled SPD operator, as the `csc_matrix` the original holds. Built once.
    #[getter]
    fn Bf(&self, py: Python<'_>) -> Py<PyAny> {
        self.bf.clone_ref(py)
    }
}
