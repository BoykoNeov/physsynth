//! Shared helpers for the 2-D half of the binding: reading NumPy arrays of unknown rank, and
//! spelling a shape the way Python's `repr` of a tuple does.
//!
//! Phase 0 and Phase 1 only ever handed 1-D arrays across the boundary, so `lib.rs` could get away
//! with `PyReadonlyArray1` and a length. Phase 2's membrane cannot: `u0` is accepted **either** as
//! a full 2-D field or as a flat live-node vector, and which one it is decides what happens next.
//! So arrays arrive here as `PyArrayDyn` and the rank is a value, not an assumption.
//!
//! The shape formatting is not cosmetic. `physsynth/core/membrane.py` puts `mask.shape` and
//! `u0.shape` into its `ValueError` text, and the project's convention (established in Phase 0) is
//! that a ported rejection reproduces the original's message verbatim, because the suite matches
//! on it. `(15,)` — with the trailing comma — is what Python prints for a 1-tuple, and `(5, 7)`
//! with the space is what it prints for a 2-tuple.

use numpy::{PyArrayDyn, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// `repr()` of a shape tuple, Python's spelling: `(15,)`, `(5, 7)`, `()`.
pub fn shape_repr(shape: &[usize]) -> String {
    match shape {
        [] => "()".to_owned(),
        [n] => format!("({n},)"),
        _ => {
            let inner: Vec<String> = shape.iter().map(|n| n.to_string()).collect();
            format!("({})", inner.join(", "))
        }
    }
}

/// `np.ascontiguousarray(np.asarray(obj, dtype=float))`, returned as `(shape, values)`.
///
/// Going through NumPy rather than a direct downcast is what makes a nested *list* an acceptable
/// field, which the original accepts. The values come back row-major (C-order), which is the
/// ordering every flat index in `physsynth-core::ops2d` means.
pub fn as_f64_field(
    py: Python<'_>,
    obj: &Bound<'_, PyAny>,
    name: &str,
) -> PyResult<(Vec<usize>, Vec<f64>)> {
    let np = py.import("numpy")?;
    let arr = np.call_method1("asarray", (obj, np.getattr("float64")?))?;
    let arr = np.call_method1("ascontiguousarray", (arr,))?;
    let arr: Bound<'_, PyArrayDyn<f64>> = arr
        .cast_into()
        .map_err(|_| PyValueError::new_err(format!("{name} must be an array of floats.")))?;
    let ro = arr.readonly();
    let shape = ro.shape().to_vec();
    let values = ro
        .as_slice()
        .map_err(|_| PyValueError::new_err(format!("{name} must be contiguous.")))?
        .to_vec();
    Ok((shape, values))
}

/// The same, for a boolean mask.
pub fn as_bool_field(
    py: Python<'_>,
    obj: &Bound<'_, PyAny>,
    name: &str,
) -> PyResult<(Vec<usize>, Vec<bool>)> {
    let np = py.import("numpy")?;
    let arr = np.call_method1("asarray", (obj, np.getattr("bool_")?))?;
    let arr = np.call_method1("ascontiguousarray", (arr,))?;
    let arr: Bound<'_, PyArrayDyn<bool>> = arr
        .cast_into()
        .map_err(|_| PyValueError::new_err(format!("{name} must be a boolean array.")))?;
    let ro = arr.readonly();
    let shape = ro.shape().to_vec();
    let values = ro
        .as_slice()
        .map_err(|_| PyValueError::new_err(format!("{name} must be contiguous.")))?
        .to_vec();
    Ok((shape, values))
}

/// A flat row-major `Vec<f64>` as a fresh 2-D NumPy array of the given shape.
pub fn to_2d_f64(
    py: Python<'_>,
    values: Vec<f64>,
    nrows: usize,
    ncols: usize,
) -> PyResult<Py<PyAny>> {
    let flat = numpy::PyArray1::from_vec(py, values);
    Ok(flat.reshape([nrows, ncols])?.into_any().unbind())
}

/// A flat row-major `Vec<bool>` as a fresh 2-D NumPy array of the given shape.
pub fn to_2d_bool(
    py: Python<'_>,
    values: Vec<bool>,
    nrows: usize,
    ncols: usize,
) -> PyResult<Py<PyAny>> {
    let flat = numpy::PyArray1::from_vec(py, values);
    Ok(flat.reshape([nrows, ncols])?.into_any().unbind())
}

/// A flat row-major `Vec<i64>` as a fresh 2-D NumPy array of the given shape.
pub fn to_2d_i64(
    py: Python<'_>,
    values: Vec<i64>,
    nrows: usize,
    ncols: usize,
) -> PyResult<Py<PyAny>> {
    let flat = numpy::PyArray1::from_vec(py, values);
    Ok(flat.reshape([nrows, ncols])?.into_any().unbind())
}
