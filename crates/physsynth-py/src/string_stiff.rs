//! The binding over `physsynth_core::string_stiff` — model #2 wearing the Python interface.
//!
//! Buffer ownership follows §9.3 and every model since: `u` and `u_prev` are **Python-owned**
//! `PyArray1`s, rebound (not overwritten) by `step`, and settable. That is not a nicety here — a
//! stiff string is a resonator other models drive, and `bow` does `string.u += ...` while
//! `collision::BarrierString` does `s.u[1:-1] = s.u[1:-1] + ...`. Both are in-place writes
//! *through* the attribute, and both only reach the model because the array is the real one.
//!
//! `x`, `_L` and `_chol` are immutable after construction and handed back by `clone_ref`. `_L` is
//! a `scipy.sparse.csr_matrix` built **once**: nothing in the tree multiplies by it from Python
//! today, but a getter that rebuilt it per access is the trap `membrane` recorded in §11.4 and
//! costs nothing to avoid.
//!
//! # `_L` and `_chol` are exposed on purpose
//!
//! They are private names in Python and no client reads them. They are here so
//! `tests/test_rust_parity_stiff_string.py` can compare the *operator* and the *factor* rather
//! than only the trajectory they produce — §16.4's lesson, where a fixture chosen for the physics
//! left the solver untested. Comparing `_L`'s `indices` is also the only direct check that §18's
//! canonical-order decision landed on both sides.

use crate::{as_1d_f64, csr_triplets, state_slice};
use numpy::{PyArray1, PyArrayMethods, PyUntypedArrayMethods};
use physsynth_core::string_stiff as core;
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyString};

/// Translate a core rejection into the `ValueError` the Python original raises.
///
/// `BadBoundary` never reaches here: its message quotes the object the caller passed, which only
/// the caller can `repr()`, so it is formatted at the call site.
pub(crate) fn param_err(e: core::ParamError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// `boundary must be 'supported', got {boundary!r}.` — the message Python raises.
pub(crate) fn bad_boundary(boundary: &Bound<'_, PyAny>) -> PyErr {
    let shown = boundary
        .repr()
        .map(|r| r.to_string_lossy().into_owned())
        .unwrap_or_else(|_| "<unrepresentable>".to_owned());
    PyValueError::new_err(format!("boundary must be 'supported', got {shown}."))
}

/// Is this the one boundary the theta-scheme strings implement?
pub(crate) fn boundary_ok(boundary: &Bound<'_, PyAny>) -> bool {
    boundary
        .cast::<PyString>()
        .ok()
        .and_then(|s| s.to_cow().ok().map(|c| c == "supported"))
        .unwrap_or(false)
}

/// Build the `scipy.sparse.csr_matrix` a `_L` / `_D2` getter hands back.
pub(crate) fn csr_object(py: Python<'_>, m: &physsynth_core::sparse::Csr) -> PyResult<Py<PyAny>> {
    let scipy = py.import("scipy.sparse")?;
    let (data, indices, indptr, shape) = csr_triplets(py, m)?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("shape", shape)?;
    Ok(scipy
        .call_method("csr_matrix", ((data, indices, indptr),), Some(&kwargs))?
        .unbind())
}

/// `np.broadcast_to(np.asarray(v0, float), (nodes,))`: a scalar fills, a full-length array is
/// taken as is, anything else is the caller's mistake.
pub(crate) fn velocity_arg(
    py: Python<'_>,
    v0: Option<&Bound<'_, PyAny>>,
    nodes: usize,
) -> PyResult<Vec<f64>> {
    match v0 {
        None => Ok(vec![0.0; nodes]),
        Some(obj) => match obj.extract::<f64>() {
            Ok(scalar) => Ok(vec![scalar; nodes]),
            Err(_) => as_1d_f64(py, obj, "v0", nodes),
        },
    }
}

/// A discretized stiff string resonator — the Rust implementation, wearing the Python interface.
///
/// Attribute-for-attribute and method-for-method compatible with
/// `physsynth.core.string_stiff.StiffString`; the docstring on that class is the reference.
#[pyclass(name = "StiffString", module = "physsynth_rs")]
pub struct PyStiffString {
    p: core::Params,
    boundary: Py<PyAny>,
    x: Py<PyArray1<f64>>,
    op_l: Py<PyAny>,
    chol: Py<PyArray1<f64>>,
    u: Py<PyArray1<f64>>,
    u_prev: Py<PyArray1<f64>>,
    n: usize,
}

impl PyStiffString {
    /// Rebind `self.u` to `values`, returning the array object that was there before.
    fn swap_u(&mut self, py: Python<'_>, values: Vec<f64>) -> Py<PyArray1<f64>> {
        let fresh = PyArray1::from_vec(py, values).unbind();
        std::mem::replace(&mut self.u, fresh)
    }

    /// Validate an array being assigned to `.u` / `.u_prev` and take ownership of it.
    fn adopt_state(&self, value: &Bound<'_, PyAny>, name: &str) -> PyResult<Py<PyArray1<f64>>> {
        let arr: Bound<'_, PyArray1<f64>> = value.clone().cast_into().map_err(|_| {
            PyValueError::new_err(format!("{name} must be a 1-D float64 numpy array."))
        })?;
        let ro = arr.readonly();
        if ro.len() != self.p.nodes() {
            return Err(PyValueError::new_err(format!(
                "{name} must have shape ({},), got ({},).",
                self.p.nodes(),
                ro.len()
            )));
        }
        state_slice(&ro, name)?;
        Ok(arr.unbind())
    }
}

#[pymethods]
impl PyStiffString {
    // Nine keyword arguments plus the GIL token: this signature IS `StiffString.__init__`, and
    // every call site in `tests/` and `web/serialize.py` spells it out.
    #[allow(clippy::too_many_arguments)]
    #[new]
    #[pyo3(signature = (*, L, T, rho, fs, N, kappa=0.0, sigma=0.0,
                        theta=core::THETA_DEFAULT, boundary=None))]
    fn new(
        py: Python<'_>,
        L: f64,
        T: f64,
        rho: f64,
        fs: f64,
        N: i64,
        kappa: f64,
        sigma: f64,
        theta: f64,
        boundary: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let boundary = match boundary {
            Some(b) => b,
            None => PyString::new(py, "supported").into_any(),
        };
        let ok = boundary_ok(&boundary);
        let p =
            core::Params::new(L, T, rho, fs, N, kappa, sigma, theta, ok).map_err(|e| match e {
                core::ParamError::BadBoundary => bad_boundary(&boundary),
                other => param_err(other),
            })?;

        let x = PyArray1::from_vec(py, p.grid()).unbind();
        let op_l = csr_object(py, &p.op_l)?;
        let chol = PyArray1::from_slice(py, &p.chol).unbind();
        let nodes = p.nodes();
        Ok(PyStiffString {
            p,
            boundary: boundary.unbind(),
            x,
            op_l,
            chol,
            u: PyArray1::from_vec(py, vec![0.0; nodes]).unbind(),
            u_prev: PyArray1::from_vec(py, vec![0.0; nodes]).unbind(),
            n: 0,
        })
    }

    // -- parameters ------------------------------------------------------------------------

    #[getter]
    fn L(&self) -> f64 {
        self.p.l
    }
    #[getter]
    fn T(&self) -> f64 {
        self.p.t
    }
    #[getter]
    fn rho(&self) -> f64 {
        self.p.rho
    }
    #[getter]
    fn fs(&self) -> f64 {
        self.p.fs
    }
    #[getter]
    fn N(&self) -> usize {
        self.p.n
    }
    #[getter]
    fn kappa(&self) -> f64 {
        self.p.kappa
    }
    #[getter]
    fn sigma(&self) -> f64 {
        self.p.sigma
    }
    #[getter]
    fn theta(&self) -> f64 {
        self.p.theta
    }
    #[getter]
    fn c(&self) -> f64 {
        self.p.c
    }
    #[getter]
    fn h(&self) -> f64 {
        self.p.h
    }
    #[getter]
    fn k(&self) -> f64 {
        self.p.k
    }
    #[getter]
    fn lam(&self) -> f64 {
        self.p.lam
    }
    #[getter]
    fn B(&self) -> f64 {
        self.p.b
    }
    #[getter]
    fn boundary(&self, py: Python<'_>) -> Py<PyAny> {
        self.boundary.clone_ref(py)
    }
    #[getter]
    fn x(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        self.x.clone_ref(py)
    }

    /// The interior operator `L`, as the `csr_matrix` the original holds. Built once.
    #[getter]
    fn _L(&self, py: Python<'_>) -> Py<PyAny> {
        self.op_l.clone_ref(py)
    }

    /// The banded Cholesky factor of `A`, shaped `(3, N - 1)` as `cholesky_banded` returns it.
    #[getter]
    fn _chol(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let flat = self.chol.bind(py);
        Ok(flat.reshape([3, self.p.interior()])?.into_any().unbind())
    }

    // -- state -----------------------------------------------------------------------------

    /// Current displacement field `u^n` — the live array, writable in place.
    #[getter]
    fn u(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        self.u.clone_ref(py)
    }
    #[setter]
    fn set_u(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.u = self.adopt_state(value, "u")?;
        Ok(())
    }

    /// Previous displacement field `u^{n-1}` — after a step this *is* the object `.u` was.
    #[getter]
    fn u_prev(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        self.u_prev.clone_ref(py)
    }
    #[setter]
    fn set_u_prev(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.u_prev = self.adopt_state(value, "u_prev")?;
        Ok(())
    }

    #[getter]
    fn n(&self) -> usize {
        self.n
    }
    #[setter]
    fn set_n(&mut self, value: usize) {
        self.n = value;
    }

    /// Current displacement field (a copy, safe to mutate/store for plotting).
    #[getter]
    fn state(&self, py: Python<'_>) -> PyResult<Py<PyArray1<f64>>> {
        let bound = self.u.bind(py);
        let ro = bound.readonly();
        Ok(PyArray1::from_slice(py, state_slice(&ro, "u")?).unbind())
    }

    // -- initial conditions ----------------------------------------------------------------

    /// Set the initial displacement (and optional velocity), ends clamped.
    #[pyo3(signature = (u0, v0=None))]
    fn set_state(
        &mut self,
        py: Python<'_>,
        u0: &Bound<'_, PyAny>,
        v0: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let nodes = self.p.nodes();
        let mut u = as_1d_f64(py, u0, "u0", nodes)?;
        let v = velocity_arg(py, v0, nodes)?;
        let prev = core::initial_previous(&mut u, &v, &self.p);
        self.u_prev = PyArray1::from_vec(py, prev).unbind();
        self.swap_u(py, u);
        self.n = 0;
        Ok(())
    }

    // -- time stepping ---------------------------------------------------------------------

    /// Advance one timestep via the banded back-substitution, rolling the history.
    fn step(&mut self, py: Python<'_>) -> PyResult<()> {
        let mut next = vec![0.0; self.p.nodes()];
        {
            let u_bound = self.u.bind(py);
            let up_bound = self.u_prev.bind(py);
            let u_ro = u_bound.readonly();
            let up_ro = up_bound.readonly();
            core::step_into(
                state_slice(&u_ro, "u")?,
                state_slice(&up_ro, "u_prev")?,
                &mut next,
                &self.p,
            );
        }
        self.u_prev = self.swap_u(py, next);
        self.n += 1;
        Ok(())
    }

    // -- diagnostics -----------------------------------------------------------------------

    /// Discrete energy `E^n` (Joules) for the implicit theta-scheme.
    fn energy(&self, py: Python<'_>) -> PyResult<f64> {
        let u_bound = self.u.bind(py);
        let up_bound = self.u_prev.bind(py);
        let u_ro = u_bound.readonly();
        let up_ro = up_bound.readonly();
        Ok(core::energy(
            state_slice(&u_ro, "u")?,
            state_slice(&up_ro, "u_prev")?,
            &self.p,
        ))
    }

    /// Displacement at grid node `index` — a pickup for spectral analysis.
    ///
    /// Negative indices count from the end, as they do on the NumPy array this replaces.
    fn displacement_at(&self, py: Python<'_>, index: i64) -> PyResult<f64> {
        let bound = self.u.bind(py);
        let ro = bound.readonly();
        let s = state_slice(&ro, "u")?;
        node_value(s, index)
    }
}

/// `float(u[index])` with Python's negative-index rule and Python's `IndexError`.
pub(crate) fn node_value(u: &[f64], index: i64) -> PyResult<f64> {
    let nodes = u.len() as i64;
    let idx = if index < 0 { index + nodes } else { index };
    if idx < 0 || idx >= nodes {
        return Err(PyIndexError::new_err(format!(
            "index {index} is out of bounds for a grid of {nodes} nodes"
        )));
    }
    Ok(u[idx as usize])
}
