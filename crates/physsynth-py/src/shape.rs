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
    // The shape is read from `asarray`, NOT from the contiguous copy below: `ascontiguousarray`
    // promotes a **0-d** array to shape `(1,)`, and `guitar_half_width` is vectorised over
    // whatever it is handed, including a bare float. The original returns `()` there.
    let shape: Vec<usize> = arr.getattr("shape")?.extract()?;
    let arr = np.call_method1("ascontiguousarray", (arr,))?;
    let arr: Bound<'_, PyArrayDyn<f64>> = arr
        .cast_into()
        .map_err(|_| PyValueError::new_err(format!("{name} must be an array of floats.")))?;
    let ro = arr.readonly();
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

/// A flat row-major `Vec<f64>` as a fresh NumPy array of arbitrary rank.
///
/// `guitar_half_width` is vectorised over whatever shape it is handed — a scalar, a row of
/// midpoints, a whole meshgrid — and the original returns the shape it was given. Reproducing that
/// is the difference between a drop-in swap and one that works until somebody passes a 1-D array.
pub fn to_shaped_f64(py: Python<'_>, values: Vec<f64>, shape: &[usize]) -> PyResult<Py<PyAny>> {
    // Built through `ArrayD` rather than by reshaping a 1-D array, because rank **zero** is a real
    // case here and the two spellings disagree about it: `PyArray1::reshape(vec![])` comes back
    // shape `(1,)`, while `np.asarray(0.3)` and therefore the original's return value are `()`.
    // A scalar `t` is exactly what `plate._depth_inside_outline` can hand this.
    use numpy::ndarray::{ArrayD, IxDyn};
    let arr = ArrayD::from_shape_vec(IxDyn(shape), values)
        .map_err(|e| PyValueError::new_err(format!("cannot shape the result: {e}")))?;
    Ok(numpy::PyArray::from_owned_array(py, arr)
        .into_any()
        .unbind())
}
