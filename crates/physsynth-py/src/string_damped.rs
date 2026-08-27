//! The binding over `physsynth_core::string_damped` — model #3 wearing the Python interface.
//!
//! Everything the stiff string's binding says applies here; that module also owns the helpers both
//! share (`boundary_ok`, `csr_object`, `velocity_arg`, `node_value`), because they are about the
//! *Python interface* the two classes have in common rather than about either model's physics.
//! The two cores stay separate on purpose — see `physsynth_core::string_stiff`'s header.
//!
//! # The one method model #2 does not have
//!
//! [`PyDampedStiffString::apply_Ainv`] exposes the action of the update matrix's inverse, and it is
//! the interface three coupled models reach through: `bow` precomputes a driving-point admittance
//! with it, `collision::BarrierString` builds a whole admittance *block* out of `N - 1` calls to
//! it, and `connection`'s bridges do the same. Those callers are still Python and stay so this
//! batch, which is the ordinary state of a migration — a ported model waits on its clients (§1.2).

use crate::string_stiff::{bad_boundary, boundary_ok, csr_object, node_value, velocity_arg};
use crate::{as_1d_f64, state_slice};
use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1, PyUntypedArrayMethods};
use physsynth_core::string_damped as core;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyString;

/// A discretized damped stiff string — the Rust implementation, wearing the Python interface.
///
/// Attribute-for-attribute and method-for-method compatible with
/// `physsynth.core.string_damped.DampedStiffString`; the docstring on that class is the reference.
#[pyclass(name = "DampedStiffString", module = "physsynth_rs")]
pub struct PyDampedStiffString {
    p: core::Params,
    boundary: Py<PyAny>,
    x: Py<PyArray1<f64>>,
    op_l: Py<PyAny>,
    op_d2: Py<PyAny>,
    chol: Py<PyArray1<f64>>,
    u: Py<PyArray1<f64>>,
    u_prev: Py<PyArray1<f64>>,
    n: usize,
}

impl PyDampedStiffString {
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

    /// The validated parameter block — what a model built on this string derives from.
    pub(crate) fn params(&self) -> &core::Params {
        &self.p
    }

    /// `u[i]` — one node of the current field.
    pub(crate) fn u_at(&self, py: Python<'_>, i: usize) -> PyResult<f64> {
        let bound = self.u.bind(py);
        let ro = bound.readonly();
        Ok(state_slice(&ro, "u")?[i])
    }

    /// `u_prev[i]` — one node of the previous field.
    ///
    /// Only meaningful *before* `step()`, which rebinds `u_prev` to what `u` was. The bow reads it
    /// on its first line for exactly that reason.
    pub(crate) fn u_prev_at(&self, py: Python<'_>, i: usize) -> PyResult<f64> {
        let bound = self.u_prev.bind(py);
        let ro = bound.readonly();
        Ok(state_slice(&ro, "u_prev")?[i])
    }

    /// Run `f` over the live `u` buffer, in place.
    ///
    /// In place, not a rebind: `bow` applies its rank-1 force correction through this, and a caller
    /// holding `.u` from before the step must see it — the property `connection.py` already
    /// depends on for the bridge force.
    pub(crate) fn with_u_mut<R>(
        &self,
        py: Python<'_>,
        f: impl FnOnce(&mut [f64]) -> R,
    ) -> PyResult<R> {
        let bound = self.u.bind(py);
        let mut rw = bound.readwrite();
        let s = rw
            .as_slice_mut()
            .map_err(|_| PyValueError::new_err("u must be a contiguous 1-D float64 array."))?;
        Ok(f(s))
    }
}

#[pymethods]
impl PyDampedStiffString {
    #[allow(clippy::too_many_arguments)]
    #[new]
    #[pyo3(signature = (*, L, T, rho, fs, N, kappa=0.0, sigma0=0.0, sigma1=0.0,
                        theta=physsynth_core::string_stiff::THETA_DEFAULT, boundary=None))]
    fn new(
        py: Python<'_>,
        L: f64,
        T: f64,
        rho: f64,
        fs: f64,
        N: i64,
        kappa: f64,
        sigma0: f64,
        sigma1: f64,
        theta: f64,
        boundary: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let boundary = match boundary {
            Some(b) => b,
            None => PyString::new(py, "supported").into_any(),
        };
        let ok = boundary_ok(&boundary);
        let p =
            core::Params::new(L, T, rho, fs, N, kappa, sigma0, sigma1, theta, ok).map_err(|e| {
                match e {
                    core::ParamError::BadBoundary => bad_boundary(&boundary),
                    other => PyValueError::new_err(other.to_string()),
                }
            })?;

        let x = PyArray1::from_vec(py, p.grid()).unbind();
        let op_l = csr_object(py, &p.op_l)?;
        let op_d2 = csr_object(py, &p.op_d2)?;
        let chol = PyArray1::from_slice(py, &p.chol).unbind();
        let nodes = p.nodes();
        Ok(PyDampedStiffString {
            p,
            boundary: boundary.unbind(),
            x,
            op_l,
            op_d2,
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
    fn sigma0(&self) -> f64 {
        self.p.sigma0
    }
    #[getter]
    fn sigma1(&self) -> f64 {
        self.p.sigma1
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

    /// The second-difference matrix, kept separately for the `sigma1` term.
    #[getter]
    fn _D2(&self, py: Python<'_>) -> Py<PyAny> {
        self.op_d2.clone_ref(py)
    }

    /// The banded Cholesky factor of `A`, shaped `(3, N - 1)`.
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

    /// Previous displacement field `u^{n-1}`.
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
    pub(crate) fn state(&self, py: Python<'_>) -> PyResult<Py<PyArray1<f64>>> {
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
    pub(crate) fn step(&mut self, py: Python<'_>) -> PyResult<()> {
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

    /// Discrete energy `E^n` (Joules) — model #2's form, unchanged.
    pub(crate) fn energy(&self, py: Python<'_>) -> PyResult<f64> {
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
    pub(crate) fn displacement_at(&self, py: Python<'_>, index: i64) -> PyResult<f64> {
        let bound = self.u.bind(py);
        let ro = bound.readonly();
        let s = state_slice(&ro, "u")?;
        node_value(s, index)
    }

    /// Solve `A x = rhs_int` for the interior unknowns — the same factor `step` uses.
    ///
    /// The name is Python's, capital and all: `bow`, `collision` and `connection` call it.
    #[allow(non_snake_case)]
    fn apply_Ainv(
        &self,
        py: Python<'_>,
        rhs_int: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<Py<PyArray1<f64>>> {
        let want = self.p.interior();
        let s = state_slice(&rhs_int, "rhs_int")?;
        if s.len() != want {
            return Err(PyValueError::new_err(format!(
                "rhs_int must have shape ({},), got ({},).",
                want,
                s.len()
            )));
        }
        Ok(PyArray1::from_vec(py, core::apply_ainv(s, &self.p)).unbind())
    }
}
