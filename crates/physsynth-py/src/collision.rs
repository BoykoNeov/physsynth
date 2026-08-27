//! The binding over `physsynth_core::collision` and `::dense` — the contact leg (plan §16).
//!
//! Shaped like the banded binding rather than like a model: free functions, no state held here.
//! The shim at the bottom of `physsynth/core/collision.py` rebinds the module's own names to
//! these, which reaches `mallet.py` too — it re-exports the same primitives — so one flag swings
//! both contact models without an edit inside either.
//!
//! # The one place this binding is not a thin wrapper
//!
//! The primitives are *vectorized* in Python: `contact_potential(eta, K, alpha)` accepts a float
//! or an array, and — as `physsynth_core::collision`'s header measures — **those two are not the
//! same computation**, because NumPy's power ufunc loop carries fast paths its scalar path does
//! not. So the dispatch on argument rank here is not a convenience: it selects which arithmetic to
//! reproduce. A scalar in takes `PowPath::Scalar` and comes back a Python float; an array in takes
//! `PowPath::Array` and comes back an array of the same shape.
//!
//! # `G` is borrowed, not copied
//!
//! `solve_contact_vector` is called once per timestep with the same `m x m` admittance block, and
//! `BarrierString` — still Python this batch — holds it. Reading it through `PyReadonlyArray2`
//! borrows the NumPy buffer directly instead of copying `m^2` doubles per step, which for the
//! default barrier fixture is 6,241 of them. `as_f64_field`, which the rest of the binding uses,
//! would have copied.

use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use physsynth_core::collision::{self as cc, ContactParams, PowPath};
use physsynth_core::dense;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyTuple;

/// Read an argument that is either a float or a float64 array, keeping which one it was.
///
/// `None` for the shape means it arrived as a scalar. That distinction survives all the way to the
/// power spelling, so it cannot be flattened away here.
fn scalar_or_array(
    py: Python<'_>,
    obj: &Bound<'_, PyAny>,
    name: &str,
) -> PyResult<(Option<Vec<usize>>, Vec<f64>)> {
    if let Ok(v) = obj.extract::<f64>() {
        return Ok((None, vec![v]));
    }
    let np = py.import("numpy")?;
    let arr = np
        .call_method1("asarray", (obj,))?
        .call_method1("astype", ("float64",))?;
    let arr = np.call_method1("ascontiguousarray", (arr,))?;
    let dynarr: Bound<'_, numpy::PyArrayDyn<f64>> = arr
        .cast_into()
        .map_err(|_| PyValueError::new_err(format!("{name} must be real-valued")))?;
    let shape = dynarr.shape().to_vec();
    let values = dynarr.to_vec()?;
    Ok((Some(shape), values))
}

/// Hand back a float for a scalar call and an array of the original shape for an array call.
fn scalar_or_array_out(
    py: Python<'_>,
    shape: Option<Vec<usize>>,
    values: Vec<f64>,
) -> PyResult<Py<PyAny>> {
    match shape {
        None => Ok(values[0].into_pyobject(py)?.into_any().unbind()),
        Some(shape) => {
            let flat = PyArray1::from_vec(py, values);
            Ok(flat.reshape(shape)?.into_any().unbind())
        }
    }
}

fn path_for(shape: &Option<Vec<usize>>) -> PowPath {
    match shape {
        None => PowPath::Scalar,
        Some(_) => PowPath::Array,
    }
}

/// `contact_potential(eta, K, alpha)`.
#[pyfunction]
#[pyo3(name = "contact_potential", signature = (eta, K, alpha))]
#[allow(non_snake_case)]
pub fn py_contact_potential(
    py: Python<'_>,
    eta: &Bound<'_, PyAny>,
    K: f64,
    alpha: f64,
) -> PyResult<Py<PyAny>> {
    let (shape, values) = scalar_or_array(py, eta, "eta")?;
    let p = path_for(&shape);
    let out = values
        .into_iter()
        .map(|e| cc::contact_potential(e, K, alpha, p))
        .collect();
    scalar_or_array_out(py, shape, out)
}

/// `contact_force_elastic(eta, K, alpha)`.
#[pyfunction]
#[pyo3(name = "contact_force_elastic", signature = (eta, K, alpha))]
#[allow(non_snake_case)]
pub fn py_contact_force_elastic(
    py: Python<'_>,
    eta: &Bound<'_, PyAny>,
    K: f64,
    alpha: f64,
) -> PyResult<Py<PyAny>> {
    let (shape, values) = scalar_or_array(py, eta, "eta")?;
    let p = path_for(&shape);
    let out = values
        .into_iter()
        .map(|e| cc::contact_force_elastic(e, K, alpha, p))
        .collect();
    scalar_or_array_out(py, shape, out)
}

/// `contact_stiffness(eta, K, alpha)`.
#[pyfunction]
#[pyo3(name = "contact_stiffness", signature = (eta, K, alpha))]
#[allow(non_snake_case)]
pub fn py_contact_stiffness(
    py: Python<'_>,
    eta: &Bound<'_, PyAny>,
    K: f64,
    alpha: f64,
) -> PyResult<Py<PyAny>> {
    let (shape, values) = scalar_or_array(py, eta, "eta")?;
    let p = path_for(&shape);
    let out = values
        .into_iter()
        .map(|e| cc::contact_stiffness(e, K, alpha, p))
        .collect();
    scalar_or_array_out(py, shape, out)
}

/// `contact_force_dg(eta_next, eta_prev, K, alpha, tol)` — scalar only, as the original declares.
#[pyfunction]
#[pyo3(name = "contact_force_dg", signature = (eta_next, eta_prev, K, alpha, tol))]
#[allow(non_snake_case)]
pub fn py_contact_force_dg(
    eta_next: f64,
    eta_prev: f64,
    K: f64,
    alpha: f64,
    tol: f64,
) -> PyResult<f64> {
    Ok(cc::contact_force_dg(
        eta_next,
        eta_prev,
        K,
        alpha,
        tol,
        PowPath::Scalar,
    ))
}

/// `contact_force_total(eta_next, eta_prev, K, alpha, lam_h, k, tol)` — scalar only.
#[pyfunction]
#[pyo3(name = "contact_force_total", signature = (eta_next, eta_prev, K, alpha, lam_h, k, tol))]
#[allow(non_snake_case)]
pub fn py_contact_force_total(
    eta_next: f64,
    eta_prev: f64,
    K: f64,
    alpha: f64,
    lam_h: f64,
    k: f64,
    tol: f64,
) -> PyResult<f64> {
    Ok(cc::contact_force_total(
        eta_next,
        eta_prev,
        K,
        alpha,
        lam_h,
        k,
        tol,
        PowPath::Scalar,
    ))
}

/// `_contact_force_total_deriv` — private in Python, exposed here for the parity tests.
#[pyfunction]
#[pyo3(name = "contact_force_total_deriv", signature = (eta_next, eta_prev, K, alpha, lam_h, k, tol))]
#[allow(non_snake_case)]
pub fn py_contact_force_total_deriv(
    eta_next: f64,
    eta_prev: f64,
    K: f64,
    alpha: f64,
    lam_h: f64,
    k: f64,
    tol: f64,
) -> PyResult<f64> {
    Ok(cc::contact_force_total_deriv(
        eta_next,
        eta_prev,
        K,
        alpha,
        lam_h,
        k,
        tol,
        PowPath::Scalar,
    ))
}

/// `_force_total_vec(eta_next, eta_prev, K, alpha, lam_h, k, tol)`.
#[pyfunction]
#[pyo3(name = "force_total_vec", signature = (eta_next, eta_prev, K, alpha, lam_h, k, tol))]
#[allow(non_snake_case, clippy::too_many_arguments)]
pub fn py_force_total_vec(
    py: Python<'_>,
    eta_next: PyReadonlyArray1<'_, f64>,
    eta_prev: PyReadonlyArray1<'_, f64>,
    K: f64,
    alpha: f64,
    lam_h: f64,
    k: f64,
    tol: f64,
) -> PyResult<Py<PyAny>> {
    let (en, ep) = (
        vec_arg(&eta_next, "eta_next")?,
        vec_arg(&eta_prev, "eta_prev")?,
    );
    same_len(en.len(), ep.len())?;
    let mut out = vec![0.0; en.len()];
    cc::force_total_vec(en, ep, &mut out, K, alpha, lam_h, k, tol);
    Ok(PyArray1::from_vec(py, out).into_any().unbind())
}

/// `_deriv_total_vec(eta_next, eta_prev, K, alpha, lam_h, k, tol)`.
#[pyfunction]
#[pyo3(name = "deriv_total_vec", signature = (eta_next, eta_prev, K, alpha, lam_h, k, tol))]
#[allow(non_snake_case, clippy::too_many_arguments)]
pub fn py_deriv_total_vec(
    py: Python<'_>,
    eta_next: PyReadonlyArray1<'_, f64>,
    eta_prev: PyReadonlyArray1<'_, f64>,
    K: f64,
    alpha: f64,
    lam_h: f64,
    k: f64,
    tol: f64,
) -> PyResult<Py<PyAny>> {
    let (en, ep) = (
        vec_arg(&eta_next, "eta_next")?,
        vec_arg(&eta_prev, "eta_prev")?,
    );
    same_len(en.len(), ep.len())?;
    let mut out = vec![0.0; en.len()];
    cc::deriv_total_vec(en, ep, &mut out, K, alpha, lam_h, k, tol);
    Ok(PyArray1::from_vec(py, out).into_any().unbind())
}

fn vec_arg<'a>(ro: &'a PyReadonlyArray1<'_, f64>, name: &str) -> PyResult<&'a [f64]> {
    ro.as_slice().map_err(|_| {
        PyValueError::new_err(format!("{name} must be a contiguous 1-D float64 array"))
    })
}

fn same_len(a: usize, b: usize) -> PyResult<()> {
    if a != b {
        return Err(PyValueError::new_err(format!(
            "eta_next and eta_prev must have the same length, got {a} and {b}."
        )));
    }
    Ok(())
}

/// `solve_contact(...)` — returns `(eta, f, used_fallback)`, the original's tuple.
#[pyfunction]
#[pyo3(
    name = "solve_contact",
    signature = (eta_free, eta_prev, g, K, alpha, lam_h, k, *, tol, seed, newton_tol = 1e-14, maxiter = 60)
)]
#[allow(non_snake_case, clippy::too_many_arguments)]
pub fn py_solve_contact(
    py: Python<'_>,
    eta_free: f64,
    eta_prev: f64,
    g: f64,
    K: f64,
    alpha: f64,
    lam_h: f64,
    k: f64,
    tol: f64,
    seed: f64,
    newton_tol: f64,
    maxiter: usize,
) -> PyResult<Py<PyAny>> {
    let p = ContactParams {
        stiffness: K,
        alpha,
        lam_h,
        k,
        tol,
    };
    let sol = cc::solve_contact(eta_free, eta_prev, g, p, seed, newton_tol, maxiter)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(PyTuple::new(
        py,
        [
            sol.eta.into_pyobject(py)?.into_any(),
            sol.force.into_pyobject(py)?.into_any(),
            sol.used_fallback.into_pyobject(py)?.to_owned().into_any(),
        ],
    )?
    .into_any()
    .unbind())
}

/// `solve_contact_vector(...)` — returns `(eta, F, iters)` plus the two diagnostics the Python
/// side needs to reproduce its warning: the final residual, and whether it converged.
///
/// The warning is raised in the shim rather than here. It is a `UserWarning` with text the tests
/// match on, and `warnings.warn(..., stacklevel=2)` has to be issued from the Python frame that
/// the original issued it from for the stack level to mean the same thing.
#[pyfunction]
#[pyo3(
    name = "solve_contact_vector",
    signature = (eta_free, eta_prev, G, K, alpha, lam_h, k, *, tol, seed, newton_tol = 1e-13, maxiter = 60)
)]
#[allow(non_snake_case, clippy::too_many_arguments)]
pub fn py_solve_contact_vector(
    py: Python<'_>,
    eta_free: PyReadonlyArray1<'_, f64>,
    eta_prev: PyReadonlyArray1<'_, f64>,
    G: PyReadonlyArray2<'_, f64>,
    K: f64,
    alpha: f64,
    lam_h: f64,
    k: f64,
    tol: f64,
    seed: PyReadonlyArray1<'_, f64>,
    newton_tol: f64,
    maxiter: usize,
) -> PyResult<Py<PyAny>> {
    let free = vec_arg(&eta_free, "eta_free")?;
    let prev = vec_arg(&eta_prev, "eta_prev")?;
    let seed = vec_arg(&seed, "seed")?;
    let m = free.len();
    // An empty support is refused rather than trivially "converged": `np.max` on an empty array
    // raises in the original, and a solve that succeeds having done nothing is the worse failure
    // of the two. `BarrierString` rejects this at construction, so nothing reaches it today.
    if m == 0 {
        return Err(PyValueError::new_err(
            "the contact support is empty; there is nothing to solve",
        ));
    }
    if prev.len() != m || seed.len() != m {
        return Err(PyValueError::new_err(format!(
            "eta_free, eta_prev and seed must all have length {m}, got {}, {} and {}.",
            free.len(),
            prev.len(),
            seed.len()
        )));
    }
    let gshape = G.shape();
    if gshape != [m, m] {
        return Err(PyValueError::new_err(format!(
            "G must be the ({m}, {m}) admittance block, got ({}, {}).",
            gshape[0], gshape[1]
        )));
    }
    let g = G
        .as_slice()
        .map_err(|_| PyValueError::new_err("G must be a C-contiguous float64 array"))?;

    let p = ContactParams {
        stiffness: K,
        alpha,
        lam_h,
        k,
        tol,
    };
    let sol = cc::solve_contact_vector(free, prev, g, p, seed, newton_tol, maxiter);
    Ok(PyTuple::new(
        py,
        [
            PyArray1::from_vec(py, sol.eta).into_any(),
            PyArray1::from_vec(py, sol.force).into_any(),
            sol.iters.into_pyobject(py)?.into_any(),
            sol.residual.into_pyobject(py)?.into_any(),
            sol.converged.into_pyobject(py)?.to_owned().into_any(),
        ],
    )?
    .into_any()
    .unbind())
}

/// `scipy.linalg.lu_factor(a)` — exposed so the parity tests can compare the factorization on its
/// own rather than only through the solve that uses it.
#[pyfunction]
#[pyo3(name = "lu_factor")]
pub fn py_lu_factor(py: Python<'_>, a: PyReadonlyArray2<'_, f64>) -> PyResult<Py<PyAny>> {
    let shape = a.shape();
    if shape[0] != shape[1] || shape[0] == 0 {
        return Err(PyValueError::new_err(format!(
            "a must be a square (n, n) matrix with n >= 1, got ({}, {}).",
            shape[0], shape[1]
        )));
    }
    let n = shape[0];
    let values = a
        .as_slice()
        .map_err(|_| PyValueError::new_err("a must be a C-contiguous float64 array"))?
        .to_vec();
    let f = dense::lu_factor(values, n).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let piv: Vec<i64> = f.piv.iter().map(|&p| p as i64).collect();
    let lu = PyArray1::from_vec(py, f.lu).reshape([n, n])?;
    Ok(PyTuple::new(
        py,
        [
            lu.into_any(),
            numpy::PyArray1::from_vec(py, piv).into_any(),
            f.info.into_pyobject(py)?.into_any(),
        ],
    )?
    .into_any()
    .unbind())
}

/// `scipy.linalg.lu_solve((lu, piv), b)` for one right-hand side.
#[pyfunction]
#[pyo3(name = "lu_solve")]
pub fn py_lu_solve(
    py: Python<'_>,
    lu: PyReadonlyArray2<'_, f64>,
    piv: PyReadonlyArray1<'_, i64>,
    b: PyReadonlyArray1<'_, f64>,
) -> PyResult<Py<PyAny>> {
    let shape = lu.shape();
    if shape[0] != shape[1] || shape[0] == 0 {
        return Err(PyValueError::new_err(format!(
            "lu must be a square (n, n) factor with n >= 1, got ({}, {}).",
            shape[0], shape[1]
        )));
    }
    let n = shape[0];
    let values = lu
        .as_slice()
        .map_err(|_| PyValueError::new_err("lu must be a C-contiguous float64 array"))?
        .to_vec();
    let pivs = piv
        .as_slice()
        .map_err(|_| PyValueError::new_err("piv must be a contiguous int64 array"))?;
    let rhs = b
        .as_slice()
        .map_err(|_| PyValueError::new_err("b must be a contiguous float64 array"))?;
    if pivs.len() != n || rhs.len() != n {
        return Err(PyValueError::new_err(format!(
            "piv and b must both have length {n}, got {} and {}.",
            pivs.len(),
            rhs.len()
        )));
    }
    let f = dense::Lu {
        lu: values,
        piv: pivs.iter().map(|&p| p as usize).collect(),
        n,
        info: 0,
    };
    let x = dense::lu_solve(&f, rhs).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(PyArray1::from_vec(py, x).into_any().unbind())
}
