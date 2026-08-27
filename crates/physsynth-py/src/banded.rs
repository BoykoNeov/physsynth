//! The binding over `physsynth_core::banded` — the shared banded Cholesky (plan §15).
//!
//! Shaped like Phase 1's operator binding rather than like a model: free functions, fresh NumPy
//! arrays out, no state held here. The shim at the bottom of `physsynth/core/banded.py` is what
//! four still-Python models actually import, so the swap reaches `string_stiff`, `string_damped`,
//! `string_nonlinear` and `string_geometric` at once without an edit inside any of them beyond the
//! import line.
//!
//! # Two deliberate narrowings, both because the alternative would be untested code
//!
//! * **Upper storage only.** Every `cholesky_banded` call in the project passes `lower=False`.
//! * **One right-hand side.** Every `cho_solve_banded` call passes a 1-D `rhs`; SciPy would accept
//!   a matrix. A 2-D path here would be a guess at a column order that nothing exercises, so the
//!   shim raises instead — loudly, which is the property that matters.
//!
//! # The exception type is not a detail
//!
//! `cholesky_banded` raises `scipy.linalg.LinAlgError` on a non-SPD band, and `LinAlgError` is not
//! a `ValueError` subclass. Reporting the refusal as a `ValueError` here would be a silent change
//! to what a caller can catch, so the refusal comes back as its own exception type and the Python
//! shim re-raises it as the `LinAlgError` the original promises. Nothing in the repo catches it
//! today — that was checked, not assumed — but "nothing catches it yet" is a fact about the
//! clients, not a licence to change the contract.

use crate::shape::{as_f64_field, shape_repr};
use numpy::{PyArray1, PyArrayMethods};
use physsynth_core::banded::{self, BandedError};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

pyo3::create_exception!(
    physsynth_rs,
    NotPositiveDefinite,
    pyo3::exceptions::PyException,
    "A banded Cholesky was refused because the matrix is not positive definite."
);

fn banded_err(e: BandedError) -> PyErr {
    match e {
        BandedError::NotPositiveDefinite(_) => NotPositiveDefinite::new_err(e.to_string()),
        BandedError::BadShape => PyValueError::new_err(e.to_string()),
    }
}

/// `numpy.asarray_chkfinite`'s refusal, which is what SciPy raises for `check_finite=True`.
///
/// Done here rather than in the Python shim, and not only for tidiness: a `np.isfinite(a).all()`
/// in the shim walks the array a second time and measured away the whole of this port's speed
/// advantage (the primitive is ~3x faster than SciPy's call; with the extra pass the *model* came
/// out 4% slower). Here it is one pass over data already being copied. It must also happen
/// BEFORE the factorization, or a NaN diagonal would come back as `NotPositiveDefinite` — the
/// right refusal for the wrong reason, and a different exception type than SciPy's.
fn check_finite(values: &[f64]) -> PyResult<()> {
    if values.iter().any(|v| !v.is_finite()) {
        return Err(PyValueError::new_err("array must not contain infs or NaNs"));
    }
    Ok(())
}

/// Read an `(kd + 1, n)` band argument as `(kd, n, values)` in row-major order.
fn band_arg(py: Python<'_>, obj: &Bound<'_, PyAny>, name: &str) -> PyResult<(usize, usize, Vec<f64>)> {
    let (shape, values) = as_f64_field(py, obj, name)?;
    if shape.len() != 2 || shape[0] == 0 || shape[1] == 0 {
        return Err(PyValueError::new_err(format!(
            "{name} must be a 2-D (kd + 1, n) band with kd + 1 >= 1 and n >= 1, got shape {}.",
            shape_repr(&shape)
        )));
    }
    check_finite(&values)?;
    Ok((shape[0] - 1, shape[1], values))
}

/// `scipy.linalg.cholesky_banded(ab, lower=False)`.
#[pyfunction]
#[pyo3(name = "cholesky_banded_upper")]
pub fn py_cholesky_banded_upper(py: Python<'_>, ab: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let (kd, n, values) = band_arg(py, ab, "ab")?;
    let factored = banded::cholesky_banded_upper(values, kd, n).map_err(banded_err)?;
    let flat = PyArray1::from_vec(py, factored);
    Ok(flat.reshape([kd + 1, n])?.into_any().unbind())
}

/// `scipy.linalg.cho_solve_banded((cb, False), b)` for a single right-hand side.
#[pyfunction]
#[pyo3(name = "cho_solve_banded_upper")]
pub fn py_cho_solve_banded_upper(
    py: Python<'_>,
    cb: &Bound<'_, PyAny>,
    b: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let (kd, n, factored) = band_arg(py, cb, "cb")?;
    let (shape, rhs) = as_f64_field(py, b, "b")?;
    check_finite(&rhs)?;
    if shape.len() != 1 {
        return Err(PyValueError::new_err(format!(
            "b must be a 1-D right-hand side, got shape {}. Every call site in this project \
             passes one column; a matrix path here would be untested.",
            shape_repr(&shape)
        )));
    }
    if shape[0] != n {
        return Err(PyValueError::new_err(format!(
            "shapes of cb and b are not compatible: cb has {n} columns, b has {}.",
            shape[0]
        )));
    }
    let x = banded::cho_solve_banded_upper(&factored, kd, n, &rhs).map_err(banded_err)?;
    Ok(PyArray1::from_vec(py, x).into_any().unbind())
}
