//! The binding over `physsynth_core::exciter` — the initial-condition shapes.
//!
//! Three free functions, each returning a fresh NumPy array of the caller's grid shape. Nothing
//! here holds state, so there is no buffer-ownership question: the caller gets an array it owns
//! outright and hands it to `set_state`.
//!
//! `raised_cosine_2d` is the one that takes 2-D input, and it preserves the caller's shape rather
//! than flattening — the original returns `np.zeros_like(X)`, and a caller who passes a meshgrid
//! expects a meshgrid back.

use crate::shape::{as_f64_field, shape_repr, to_2d_f64};
use numpy::PyArray1;
use physsynth_core::exciter;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn exciter_err(e: exciter::ExciterError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// Triangular plucked-string initial displacement (zero at both ends).
#[pyfunction]
#[pyo3(name = "triangular_pluck")]
#[pyo3(signature = (x, L, position, amplitude=1.0))]
pub fn py_triangular_pluck(
    py: Python<'_>,
    x: &Bound<'_, PyAny>,
    L: f64,
    position: f64,
    amplitude: f64,
) -> PyResult<Py<PyAny>> {
    let (_, xs) = as_f64_field(py, x, "x")?;
    let out = exciter::triangular_pluck(&xs, L, position, amplitude).map_err(exciter_err)?;
    Ok(PyArray1::from_vec(py, out).into_any().unbind())
}

/// Smooth (`C^1`) raised-cosine displacement hump, zero outside `[center-width, center+width]`.
#[pyfunction]
#[pyo3(name = "raised_cosine")]
#[pyo3(signature = (x, L, center, width, amplitude=1.0))]
pub fn py_raised_cosine(
    py: Python<'_>,
    x: &Bound<'_, PyAny>,
    L: f64,
    center: f64,
    width: f64,
    amplitude: f64,
) -> PyResult<Py<PyAny>> {
    let (_, xs) = as_f64_field(py, x, "x")?;
    let out = exciter::raised_cosine(&xs, L, center, width, amplitude).map_err(exciter_err)?;
    Ok(PyArray1::from_vec(py, out).into_any().unbind())
}

/// Smooth (`C^1`) radial raised-cosine hump on a 2-D grid, zero outside radius `width`.
#[pyfunction]
#[pyo3(name = "raised_cosine_2d")]
#[pyo3(signature = (X, Y, center, width, amplitude=1.0))]
pub fn py_raised_cosine_2d(
    py: Python<'_>,
    X: &Bound<'_, PyAny>,
    Y: &Bound<'_, PyAny>,
    center: (f64, f64),
    width: f64,
    amplitude: f64,
) -> PyResult<Py<PyAny>> {
    let (xshape, xs) = as_f64_field(py, X, "X")?;
    let (yshape, ys) = as_f64_field(py, Y, "Y")?;
    if xshape != yshape {
        return Err(PyValueError::new_err(format!(
            "X and Y must have the same shape; got {} and {}.",
            shape_repr(&xshape),
            shape_repr(&yshape)
        )));
    }
    let out = exciter::raised_cosine_2d(&xs, &ys, center, width, amplitude).map_err(exciter_err)?;
    match xshape.as_slice() {
        [nrows, ncols] => to_2d_f64(py, out, *nrows, *ncols),
        _ => Ok(PyArray1::from_vec(py, out).into_any().unbind()),
    }
}
