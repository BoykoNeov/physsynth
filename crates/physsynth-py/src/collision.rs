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
//! `solve_contact_vector` is called once per timestep with the same `m x m` admittance block.
//! Reading it through `PyReadonlyArray2` borrows the NumPy buffer directly instead of copying
//! `m^2` doubles per step, which for the default barrier fixture is 6,241 of them. `as_f64_field`,
//! which the rest of the binding uses, would have copied. (The free function keeps that path for
//! callers that hold `G` in Python; [`PyBarrierString`] below owns its block in Rust and never
//! crosses the boundary with it at all.)
//!
//! # And since §23 this module is not only free functions
//!
//! [`PyBarrierString`] is the phase's last model. It follows `bow`'s shape — it holds the
//! `Py<PyDampedStiffString>` the caller passed rather than a copy, refuses a pure-Python string,
//! and takes one borrow per step. What it adds is the **underscored half of an interface**
//! (§12.2): `_G` and `_force_pref` are settable, because a test doubles both to move the model's
//! fixed point, and `_b`/`_support` are read by the viewer to draw the rail.

use crate::as_1d_f64;
use crate::string_damped::PyDampedStiffString;
use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use physsynth_core::collision::{self as cc, ContactParams, PowPath};
use physsynth_core::dense;
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use std::ffi::CString;

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

// -- the barrier string (model #8), the last model of Phase 3 -------------------------------------

/// Broadcast the `barrier=` argument onto the `N + 1` grid, the way `np.broadcast_to` does.
///
/// A scalar becomes a flat rail; an `(N+1,)` array passes through; anything else is the
/// `ValueError` NumPy itself would raise, quoted rather than paraphrased — the original never
/// reaches its own shape check, because the broadcast fails first.
fn barrier_profile(py: Python<'_>, obj: &Bound<'_, PyAny>, nodes: usize) -> PyResult<Vec<f64>> {
    if let Ok(v) = obj.extract::<f64>() {
        return Ok(vec![v; nodes]);
    }
    let np = py.import("numpy")?;
    let arr = np.call_method1("asarray", (obj, np.getattr("float64")?))?;
    let arr = np.call_method1("ascontiguousarray", (arr,))?;
    let dynarr: Bound<'_, numpy::PyArrayDyn<f64>> = arr
        .cast_into()
        .map_err(|_| PyValueError::new_err("barrier must be real-valued"))?;
    let shape = dynarr.shape().to_vec();
    if shape.len() != 1 {
        return Err(PyValueError::new_err(
            "input operand has more dimensions than allowed by the axis remapping",
        ));
    }
    if shape[0] != nodes {
        return Err(PyValueError::new_err(format!(
            "operands could not be broadcast together with remapped shapes \
             [original->remapped]: ({},)  and requested shape ({},)",
            shape[0], nodes
        )));
    }
    Ok(dynarr.to_vec()?)
}

/// Raise the original's non-convergence `UserWarning`.
///
/// The message is byte-for-byte the Python one. What cannot be byte-for-byte is where it points:
/// the original issues it from `solve_contact_vector` with `stacklevel=2`, which names
/// `BarrierString.step` — a frame that no longer exists once the model is Rust. `1` here names the
/// Python code that called `step()`, which is the nearest true statement about who to blame. §23.6.
fn warn_not_converged(
    py: Python<'_>,
    maxiter: usize,
    residual: f64,
    newton_tol: f64,
) -> PyResult<()> {
    let msg = format!(
        "vector contact solve did not converge in {maxiter} iterations \
         (residual {} > {}); energy may drift. Raise newton_maxiter or oversample the contact.",
        physsynth_core::fmt::py_exp(residual, 2),
        physsynth_core::fmt::py_exp(newton_tol, 1),
    );
    let msg = CString::new(msg).map_err(|_| PyValueError::new_err("warning text had a NUL"))?;
    let category = py.get_type::<pyo3::exceptions::PyUserWarning>();
    PyErr::warn(py, category.as_any(), &msg, 1)
}

/// Translate a core refusal into the `ValueError` the Python original raises.
fn barrier_err(e: cc::BarrierError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// A damped stiff string vibrating against a one-sided distributed barrier — model #8, in Rust.
///
/// Attribute-for-attribute and method-for-method compatible with
/// `physsynth.core.collision.BarrierString`; the docstring on that class is the reference.
#[pyclass(name = "BarrierString", module = "physsynth_rs")]
pub struct PyBarrierString {
    p: cc::BarrierParams,
    s: cc::BarrierState,
    string: Py<PyDampedStiffString>,
}

impl PyBarrierString {
    /// `b - u[support]` off whichever of the string's two history levels `prev` selects.
    fn gather(&self, py: Python<'_>, prev: bool) -> PyResult<Vec<f64>> {
        let handle = self.string.bind(py).borrow();
        let mut out = vec![0.0; self.p.support_len()];
        let p = &self.p;
        if prev {
            handle.with_u_prev_ref(py, |u| cc::penetration_of(p, u, &mut out))?;
        } else {
            handle.with_u_ref(py, |u| cc::penetration_of(p, u, &mut out))?;
        }
        Ok(out)
    }
}

#[pymethods]
impl PyBarrierString {
    #[new]
    #[pyo3(signature = (
        *, string, barrier, stiffness, alpha=1.5, hysteresis=0.0, eta_tol=1e-12,
        newton_tol=1e-13, newton_maxiter=60
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        string: &Bound<'_, PyAny>,
        barrier: &Bound<'_, PyAny>,
        stiffness: f64,
        alpha: f64,
        hysteresis: f64,
        eta_tol: f64,
        newton_tol: f64,
        newton_maxiter: i64,
    ) -> PyResult<Self> {
        let handle: Py<PyDampedStiffString> = string
            .clone()
            .cast_into::<PyDampedStiffString>()
            .map_err(|_| {
                PyTypeError::new_err(
                    "the Rust BarrierString needs a Rust DampedStiffString \
                     (physsynth_rs.DampedStiffString). Got something else -- most likely the \
                     pure-Python `string_damped.DampedStiffStringPy`, whose banded factor this \
                     class cannot reach without crossing back into the interpreter for every \
                     admittance column. Build the string from the same implementation as the \
                     barrier.",
                )
            })?
            .unbind();

        let (p, mut s) = {
            let sref = handle.bind(py).borrow();
            let sp = sref.params();
            let profile = barrier_profile(py, barrier, sp.nodes())?;
            let p = cc::BarrierParams::new(
                sp,
                &profile,
                stiffness,
                alpha,
                hysteresis,
                eta_tol,
                newton_tol,
                newton_maxiter,
            )
            .map_err(barrier_err)?;
            let m = p.support_len();
            let mut s = cc::BarrierState {
                penetration: vec![0.0; m],
                contact_force: vec![0.0; m],
                newton_iters: 0,
                n: 0,
            };
            let pref = &p;
            sref.with_u_ref(py, |u| cc::penetration_of(pref, u, &mut s.penetration))?;
            (p, s)
        };
        s.contact_force = vec![0.0; p.support_len()];

        Ok(PyBarrierString {
            p,
            s,
            string: handle,
        })
    }

    // -- parameters ------------------------------------------------------------------------

    /// The resonator — the very object the caller passed in.
    #[getter]
    fn string(&self, py: Python<'_>) -> Py<PyDampedStiffString> {
        self.string.clone_ref(py)
    }
    #[getter]
    fn k(&self) -> f64 {
        self.p.contact.k
    }
    #[allow(non_snake_case)]
    #[getter]
    fn K(&self) -> f64 {
        self.p.contact.stiffness
    }
    #[getter]
    fn alpha(&self) -> f64 {
        self.p.contact.alpha
    }
    #[getter]
    fn lam_h(&self) -> f64 {
        self.p.contact.lam_h
    }
    #[getter]
    fn eta_tol(&self) -> f64 {
        self.p.contact.tol
    }
    #[getter]
    fn newton_tol(&self) -> f64 {
        self.p.newton_tol
    }
    #[getter]
    fn newton_maxiter(&self) -> usize {
        self.p.newton_maxiter
    }

    // -- the underscored half of the interface, which is not private (§12.2) -----------------
    //
    // `tests/test_collision_modal.py` reads `_b`, `_support` and `_G[0, 0]` and *writes* `_G` and
    // `_force_pref` — doubling both is the negative control that proves the coupling magnitude
    // gate has teeth. `web/serialize.py` reads `_b` and `_support` to draw the rail. So all five
    // are part of the interface, and two of them are settable.

    /// Grid node indices carrying a finite barrier — the contact support.
    #[getter]
    fn _support(&self, py: Python<'_>) -> Py<PyArray1<i64>> {
        let v: Vec<i64> = self.p.support.iter().map(|&i| i as i64).collect();
        PyArray1::from_vec(py, v).unbind()
    }
    /// Barrier heights on the support.
    #[getter]
    fn _b(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        PyArray1::from_slice(py, &self.p.b).unbind()
    }
    /// Settable, and this one is not symmetry: `tests/test_jawari.py` flattens a curved bridge to
    /// a rail at its own crest height, which is how "the wrap edge travels" is compared against a
    /// contact that cannot travel. Writing `_b` deliberately does NOT rebuild `G` or the admittance
    /// columns — neither does the Python original, and it is right not to: the support is chosen by
    /// which heights are *finite*, and a rewrite that keeps them finite leaves the string's
    /// admittance untouched.
    #[setter]
    fn set__b(&mut self, py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.p.b = as_1d_f64(py, value, "_b", self.p.support_len())?;
        Ok(())
    }
    /// The support's indices into the interior array.
    #[getter]
    fn _int_idx(&self, py: Python<'_>) -> Py<PyArray1<i64>> {
        let v: Vec<i64> = self.p.int_idx.iter().map(|&i| i as i64).collect();
        PyArray1::from_vec(py, v).unbind()
    }
    /// The admittance columns, `(N-1) x m`.
    #[getter]
    fn _cols_mat(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let rows = self.p.nodes - 2;
        let flat = PyArray1::from_slice(py, &self.p.cols_mat);
        Ok(flat
            .reshape([rows, self.p.support_len()])?
            .into_any()
            .unbind())
    }
    /// The driving-point admittance block on the support, `m x m`.
    #[allow(non_snake_case)]
    #[getter]
    fn _G(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let m = self.p.support_len();
        let flat = PyArray1::from_slice(py, &self.p.g_mat);
        Ok(flat.reshape([m, m])?.into_any().unbind())
    }
    #[allow(non_snake_case)]
    #[setter]
    fn set__G(&mut self, py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let m = self.p.support_len();
        let np = py.import("numpy")?;
        let arr = np.call_method1("asarray", (value, np.getattr("float64")?))?;
        let arr = np.call_method1("ascontiguousarray", (arr,))?;
        let arr: Bound<'_, numpy::PyArray2<f64>> = arr
            .cast_into()
            .map_err(|_| PyValueError::new_err("_G must be a 2-D float64 array."))?;
        let ro = arr.readonly();
        if ro.shape() != [m, m] {
            return Err(PyValueError::new_err(format!(
                "_G must be the ({m}, {m}) admittance block, got ({}, {}).",
                ro.shape()[0],
                ro.shape()[1]
            )));
        }
        self.p.g_mat = ro
            .as_slice()
            .map_err(|_| PyValueError::new_err("_G must be a C-contiguous float64 array."))?
            .to_vec();
        Ok(())
    }
    /// `k^2 / rho`, the force-density prefactor.
    #[getter]
    fn _force_pref(&self) -> f64 {
        self.p.force_pref
    }
    #[setter]
    fn set__force_pref(&mut self, value: f64) {
        self.p.force_pref = value;
    }

    // -- per-step observables ----------------------------------------------------------------

    /// Penetration on the support, `eta^n`.
    #[getter]
    fn penetration(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        PyArray1::from_slice(py, &self.s.penetration).unbind()
    }
    /// Settable, because `tests/test_collision_modal.py` seats the model at a static equilibrium
    /// by hand and has to reset the continuation seed to match.
    #[setter]
    fn set_penetration(&mut self, py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.s.penetration = as_1d_f64(py, value, "penetration", self.p.support_len())?;
        Ok(())
    }
    /// Contact force density on the support for the last step.
    #[getter]
    fn contact_force(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        PyArray1::from_slice(py, &self.s.contact_force).unbind()
    }
    #[getter]
    fn newton_iters(&self) -> usize {
        self.s.newton_iters
    }
    #[getter]
    fn n(&self) -> usize {
        self.s.n
    }

    // -- initial conditions ------------------------------------------------------------------

    /// Set the string's state (delegating), then refresh the continuation seed.
    #[pyo3(signature = (u0, v0=None))]
    fn set_state(
        &mut self,
        py: Python<'_>,
        u0: &Bound<'_, PyAny>,
        v0: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        {
            let handle = self.string.clone_ref(py);
            let mut sref = handle.bind(py).borrow_mut();
            sref.set_state(py, u0, v0)?;
        }
        self.s.penetration = self.gather(py, false)?;
        self.s.n = 0;
        Ok(())
    }

    // -- time stepping -----------------------------------------------------------------------

    /// Advance one step: force-free string advance, vector contact solve, exact force inject.
    ///
    /// One borrow, like the bow's and unlike the reed's (§13.2): nothing here re-enters the
    /// interpreter mid-step, so the mutable borrow of the string spans the whole update. The
    /// warning is raised *after* it is dropped, because a `UserWarning` can be promoted to an
    /// exception by the caller's filters and unwinding through a live `borrow_mut` is a panic.
    fn step(&mut self, py: Python<'_>) -> PyResult<()> {
        let handle = self.string.clone_ref(py);
        let m = self.p.support_len();
        let mut eta_prev = vec![0.0; m];
        let mut eta_free = vec![0.0; m];

        let (residual, converged) = {
            let mut sref = handle.bind(py).borrow_mut();
            // `u^{n-1}` on the support must be read BEFORE the advance: `step()` rolls `u_prev`
            // to what `u` was, so this quantity stops existing one line later.
            {
                let p = &self.p;
                sref.with_u_prev_ref(py, |u| cc::penetration_of(p, u, &mut eta_prev))?;
            }
            sref.step(py)?;
            {
                let p = &self.p;
                sref.with_u_ref(py, |u| cc::penetration_of(p, u, &mut eta_free))?;
            }
            let p = &self.p;
            let s = &mut self.s;
            sref.with_u_mut(py, |u| cc::apply(u, &eta_free, &eta_prev, s, p))?
        };

        if !converged {
            warn_not_converged(py, self.p.newton_maxiter, residual, self.p.newton_tol)?;
        }
        Ok(())
    }

    // -- diagnostics -------------------------------------------------------------------------

    /// Total discrete energy `E^n` (J): string energy plus the averaged barrier potential.
    fn energy(&self, py: Python<'_>) -> PyResult<f64> {
        let sref = self.string.bind(py).borrow();
        let e_string = sref.energy(py)?;
        let p = &self.p;
        let pe = sref.with_u_ref(py, |u| {
            sref.with_u_prev_ref(py, |up| cc::barrier_energy(p, u, up))
        })??;
        Ok(e_string + pe)
    }

    /// The string displacement field `u^n` (a copy, for animation snapshots).
    #[getter]
    fn state(&self, py: Python<'_>) -> PyResult<Py<PyArray1<f64>>> {
        self.string.bind(py).borrow().state(py)
    }

    /// String pickup at grid node `index`.
    fn displacement_at(&self, py: Python<'_>, index: i64) -> PyResult<f64> {
        self.string.bind(py).borrow().displacement_at(py, index)
    }

    /// Which support nodes are currently in contact (`eta > 0`).
    fn contact_mask(&self, py: Python<'_>) -> Py<PyArray1<bool>> {
        let v: Vec<bool> = self.s.penetration.iter().map(|&e| e > 0.0).collect();
        PyArray1::from_vec(py, v).unbind()
    }
}
