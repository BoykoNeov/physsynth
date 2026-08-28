//! The binding over `physsynth_core::beam` — model #5b-pre wearing the Python interface.
//!
//! Buffer ownership follows §9.3 and every model since: `u` and `u_prev` are **Python-owned**
//! `PyArray1`s, rebound (not overwritten) by `step`, and settable.
//!
//! `x`, `K` and `W` are immutable after construction and handed back by `clone_ref`. `K` and `W`
//! are `scipy.sparse.csr_matrix` objects built **once**, which here is not merely the §11.4
//! precaution it was for the membrane: `tests/helpers.beam_low_eigenfrequencies` hands both of
//! them straight to a generalized `eigsh`, and `tests/test_beam_energy.py` does it again for the
//! eigenvectors. A getter that rebuilt them per access would be paying for a shift-invert twice.
//!
//! # `K` and `W` are public names, and that is the whole eigen-oracle
//!
//! Unlike `string_stiff`'s `_L`, these are not exposed for the parity test's benefit — they are
//! the documented interface. The beam's closed-form modal oracle (`cos(βL)·cosh(βL) = 1`) is
//! checked by solving `K φ = μ W φ`, so a beam whose `K` did not come back as a real SciPy matrix
//! would fail the model's most important test for a reason having nothing to do with the physics.

use crate::string_stiff::{csr_object, node_value, velocity_arg};
use crate::{as_1d_f64, state_slice};
use numpy::{PyArray1, PyArrayMethods, PyUntypedArrayMethods};
use physsynth_core::beam as core;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyString;

/// Translate a core rejection into the `ValueError` the Python original raises.
fn param_err(e: core::ParamError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// `boundary must be 'free', got {boundary!r}.` — the message Python raises.
fn bad_boundary(boundary: &Bound<'_, PyAny>) -> PyErr {
    let shown = boundary
        .repr()
        .map(|r| r.to_string_lossy().into_owned())
        .unwrap_or_else(|_| "<unrepresentable>".to_owned());
    PyValueError::new_err(format!("boundary must be 'free', got {shown}."))
}

/// A discretized free-free beam — the Rust implementation, wearing the Python interface.
///
/// Attribute-for-attribute and method-for-method compatible with
/// `physsynth.core.beam.FreeBeam`; the docstring on that class is the reference.
#[pyclass(name = "FreeBeam", module = "physsynth_rs")]
pub struct PyFreeBeam {
    p: core::Params,
    boundary: Py<PyAny>,
    x: Py<PyArray1<f64>>,
    stiffness: Py<PyAny>,
    mass: Py<PyAny>,
    w: Py<PyArray1<f64>>,
    u: Py<PyArray1<f64>>,
    u_prev: Py<PyArray1<f64>>,
    n: usize,
}

impl PyFreeBeam {
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
impl PyFreeBeam {
    // Eight keyword arguments plus the GIL token: this signature IS `FreeBeam.__init__`, and every
    // call site in `tests/` spells it out.
    #[allow(clippy::too_many_arguments)]
    #[new]
    #[pyo3(signature = (*, L, rho, fs, N, kappa, sigma=0.0,
                        theta=core::THETA_DEFAULT, boundary=None::<Py<PyAny>>))]
    fn new(
        py: Python<'_>,
        L: f64,
        rho: f64,
        fs: f64,
        N: i64,
        kappa: f64,
        sigma: f64,
        theta: f64,
        boundary: Option<Option<Py<PyAny>>>,
    ) -> PyResult<Self> {
        // `Option<Option<_>>` so that an OMITTED `boundary` and an explicit `boundary=None` are
        // distinguishable -- a plain `Option` collapses them and silently ACCEPTS `boundary=None`,
        // which the Python original rejects with a message quoting `None` (§24.7). The arm order
        // is the surprising part and is pinned by a test: PyO3 wraps the DEFAULT expression, so
        // `Some(None)` is "argument omitted" and a bare `None` is the caller's literal `None`.
        let boundary = match boundary {
            Some(None) => PyString::new(py, "free").into_any(),
            None => py.None().into_bound(py),
            Some(Some(b)) => b.into_bound(py),
        };
        let ok = boundary
            .cast::<PyString>()
            .ok()
            .and_then(|s| s.to_cow().ok().map(|c| c == "free"))
            .unwrap_or(false);
        let p = core::Params::new(L, rho, fs, N, kappa, sigma, theta, ok).map_err(|e| match e {
            core::ParamError::BadBoundary => bad_boundary(&boundary),
            other => param_err(other),
        })?;

        let x = PyArray1::from_vec(py, p.grid()).unbind();
        let stiffness = csr_object(py, &p.stiffness)?;
        let mass = csr_object(py, &p.mass)?;
        let w = PyArray1::from_slice(py, &p.w).unbind();
        let nodes = p.nodes();
        Ok(PyFreeBeam {
            p,
            boundary: boundary.unbind(),
            x,
            stiffness,
            mass,
            w,
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
    fn h(&self) -> f64 {
        self.p.h
    }
    #[getter]
    fn k(&self) -> f64 {
        self.p.k
    }
    #[getter]
    fn mu(&self) -> f64 {
        self.p.mu
    }
    #[getter]
    fn boundary(&self, py: Python<'_>) -> Py<PyAny> {
        self.boundary.clone_ref(py)
    }
    #[getter]
    fn x(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        self.x.clone_ref(py)
    }

    /// The symmetric PSD bending stiffness, as the `csr_matrix` the original holds. Built once.
    #[getter]
    fn K(&self, py: Python<'_>) -> Py<PyAny> {
        self.stiffness.clone_ref(py)
    }

    /// The diagonal trapezoidal mass, as the `csr_matrix` the original holds. Built once.
    #[getter]
    fn W(&self, py: Python<'_>) -> Py<PyAny> {
        self.mass.clone_ref(py)
    }

    /// `W.diagonal()` — the lumped mass weights, `h` inside and `h/2` at the two free ends.
    #[getter]
    fn w(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        self.w.clone_ref(py)
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

    /// Set the initial displacement (and optional velocity). Nothing is clamped — both ends are
    /// free unknowns.
    #[pyo3(signature = (u0, v0=None))]
    fn set_state(
        &mut self,
        py: Python<'_>,
        u0: &Bound<'_, PyAny>,
        v0: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let nodes = self.p.nodes();
        let u = as_1d_f64(py, u0, "u0", nodes)?;
        let v = velocity_arg(py, v0, nodes)?;
        let prev = core::initial_previous(&u, &v, &self.p);
        self.u_prev = PyArray1::from_vec(py, prev).unbind();
        self.swap_u(py, u);
        self.n = 0;
        Ok(())
    }

    // -- time stepping ---------------------------------------------------------------------

    /// Advance one timestep via the prefactored sparse solve, rolling the history.
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
