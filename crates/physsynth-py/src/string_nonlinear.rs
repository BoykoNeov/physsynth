//! The binding over `physsynth_core::string_nonlinear` — model #9 wearing the Python interface.
//!
//! The two earlier theta-scheme strings' bindings are the template; this one differs in three
//! places, and every one of them is a thing `cargo test` cannot see.
//!
//! 1. **Four public telemetry attributes**, two of them integers: `delta_tension`, `converged`,
//!    `bracket_expansions`, `n_not_converged`. `tests/test_tension_string.py` reads them, and
//!    `bracket_expansions` is *cumulative and settable* on the Python object, so it is a getter
//!    and a setter rather than a read-out.
//! 2. **`apply_Ainv` raises.** Every other string in the family implements it; this one refuses,
//!    because `A` moves with the tension and a constant driving-point admittance does not exist.
//!    The message is the Python original's, verbatim, because the test matches on `"time-varying"`
//!    — and because `bow`, `collision` and `connection` would otherwise get a panic where they
//!    expect a clean `NotImplementedError`.
//! 3. **A `RuntimeWarning` on a failed bracket**, emitted through Python's own `warnings` machinery
//!    so a `pytest.warns` (or a `-W error`) sees it. It quotes the step number, which is `self.n`
//!    *before* the increment — the state the Python original is in when it warns.
//!
//! `string_coefficients_from_material` is deliberately not bound; the core module's header says
//! why.

use crate::string_stiff::{bad_boundary, boundary_ok, csr_object, node_value, velocity_arg};
use crate::{as_1d_f64, state_slice};
use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1, PyUntypedArrayMethods};
use physsynth_core::string_nonlinear as core;
use pyo3::exceptions::{PyNotImplementedError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyString, PyTuple};

/// The refusal `apply_Ainv` raises — the Python original's text, wrapped the same way.
const AINV_REFUSAL: &str = concat!(
    "TensionModulatedString has a time-varying update matrix (A depends on the tension), ",
    "so a constant driving-point admittance A^-1 e_i does not exist. Coupling an exciter ",
    "here requires a joint solve -- see docs/dev/tension-modulated-string-plan.md."
);

/// A Kirchhoff–Carrier tension-modulated stiff string — the Rust implementation, wearing the
/// Python interface.
///
/// Attribute-for-attribute and method-for-method compatible with
/// `physsynth.core.string_nonlinear.TensionModulatedString`; that class's docstring is the
/// reference.
#[pyclass(name = "TensionModulatedString", module = "physsynth_rs")]
pub struct PyTensionModulatedString {
    p: core::Params,
    boundary: Py<PyAny>,
    x: Py<PyArray1<f64>>,
    op_l: Py<PyAny>,
    op_d2: Py<PyAny>,
    chol0: Py<PyArray1<f64>>,
    u: Py<PyArray1<f64>>,
    u_prev: Py<PyArray1<f64>>,
    n: usize,
    delta_tension: f64,
    converged: bool,
    bracket_expansions: usize,
    n_not_converged: usize,
}

impl PyTensionModulatedString {
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

    /// Copy both state arrays out so the core can be handed plain slices.
    fn state_pair(&self, py: Python<'_>) -> PyResult<(Vec<f64>, Vec<f64>)> {
        let u_bound = self.u.bind(py);
        let up_bound = self.u_prev.bind(py);
        let u_ro = u_bound.readonly();
        let up_ro = up_bound.readonly();
        Ok((
            state_slice(&u_ro, "u")?.to_vec(),
            state_slice(&up_ro, "u_prev")?.to_vec(),
        ))
    }
}

#[pymethods]
impl PyTensionModulatedString {
    #[allow(clippy::too_many_arguments)]
    #[new]
    #[pyo3(signature = (*, L, T, rho, fs, N, kappa=0.0, EA=0.0, sigma0=0.0, sigma1=0.0,
                        theta=physsynth_core::string_stiff::THETA_DEFAULT,
                        boundary=None,
                        tension_tol=physsynth_core::string_nonlinear::TENSION_TOL_DEFAULT))]
    fn new(
        py: Python<'_>,
        L: f64,
        T: f64,
        rho: f64,
        fs: f64,
        N: i64,
        kappa: f64,
        EA: f64,
        sigma0: f64,
        sigma1: f64,
        theta: f64,
        boundary: Option<Bound<'_, PyAny>>,
        tension_tol: f64,
    ) -> PyResult<Self> {
        let boundary = match boundary {
            Some(b) => b,
            None => PyString::new(py, "supported").into_any(),
        };
        let ok = boundary_ok(&boundary);
        let p = core::Params::new(
            L,
            T,
            rho,
            fs,
            N,
            kappa,
            EA,
            sigma0,
            sigma1,
            theta,
            tension_tol,
            ok,
        )
        .map_err(|e| match e {
            core::ParamError::BadBoundary => bad_boundary(&boundary),
            other => PyValueError::new_err(other.to_string()),
        })?;

        let x = PyArray1::from_vec(py, p.grid()).unbind();
        let op_l = csr_object(py, &p.op_l)?;
        let op_d2 = csr_object(py, &p.op_d2)?;
        let chol0 = PyArray1::from_slice(py, &p.chol0).unbind();
        let nodes = p.nodes();
        Ok(PyTensionModulatedString {
            p,
            boundary: boundary.unbind(),
            x,
            op_l,
            op_d2,
            chol0,
            u: PyArray1::from_vec(py, vec![0.0; nodes]).unbind(),
            u_prev: PyArray1::from_vec(py, vec![0.0; nodes]).unbind(),
            n: 0,
            delta_tension: 0.0,
            converged: true,
            bracket_expansions: 0,
            n_not_converged: 0,
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
    fn EA(&self) -> f64 {
        self.p.ea
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
    fn tension_tol(&self) -> f64 {
        self.p.tension_tol
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
    fn EA_over_T(&self) -> f64 {
        self.p.ea_over_t
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

    /// The second-difference matrix — the `sigma1` term *and* the tension term.
    #[getter]
    fn _D2(&self, py: Python<'_>) -> Py<PyAny> {
        self.op_d2.clone_ref(py)
    }

    /// The `dT = 0` banded Cholesky factor, shaped `(3, N - 1)` — model #3's factor exactly.
    #[getter]
    fn _chol0(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let flat = self.chol0.bind(py);
        Ok(flat.reshape([3, self.p.interior()])?.into_any().unbind())
    }

    /// The upper bands of `A0`, shaped `(3, N - 1)`.
    #[getter]
    fn _ab0(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let flat = PyArray1::from_slice(py, &self.p.ab0);
        Ok(flat.reshape([3, self.p.interior()])?.into_any().unbind())
    }

    /// The upper bands of `D2`, shaped `(3, N - 1)` — what `beta` scales onto `_ab0`.
    #[getter]
    fn _ab_D2(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let flat = PyArray1::from_slice(py, &self.p.ab_d2);
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

    // -- solver telemetry ------------------------------------------------------------------
    //
    // All four are plain public attributes on the Python class, so all four get a setter as well
    // as a getter: a test or a diagnostic script is free to zero a counter mid-run.

    #[getter]
    fn delta_tension(&self) -> f64 {
        self.delta_tension
    }
    #[setter]
    fn set_delta_tension(&mut self, value: f64) {
        self.delta_tension = value;
    }

    #[getter]
    fn converged(&self) -> bool {
        self.converged
    }
    #[setter]
    fn set_converged(&mut self, value: bool) {
        self.converged = value;
    }

    #[getter]
    fn bracket_expansions(&self) -> usize {
        self.bracket_expansions
    }
    #[setter]
    fn set_bracket_expansions(&mut self, value: usize) {
        self.bracket_expansions = value;
    }

    #[getter]
    fn n_not_converged(&self) -> usize {
        self.n_not_converged
    }
    #[setter]
    fn set_n_not_converged(&mut self, value: usize) {
        self.n_not_converged = value;
    }

    // -- diagnostics -----------------------------------------------------------------------

    /// Current displacement field (a copy, safe to mutate/store for plotting).
    #[getter]
    fn state(&self, py: Python<'_>) -> PyResult<Py<PyArray1<f64>>> {
        let bound = self.u.bind(py);
        let ro = bound.readonly();
        Ok(PyArray1::from_slice(py, state_slice(&ro, "u")?).unbind())
    }

    /// Current stretch `I^n = h ||delta_x+ u^n||^2` (m) — what modulates the tension.
    #[getter]
    fn stretch(&self, py: Python<'_>) -> PyResult<f64> {
        let bound = self.u.bind(py);
        let ro = bound.readonly();
        Ok(core::stretch(state_slice(&ro, "u")?, &self.p))
    }

    /// Current total tension `T0 + (EA/2L) I^n` (N). Always `>= T0` — hardening only.
    #[getter]
    fn tension(&self, py: Python<'_>) -> PyResult<f64> {
        Ok(self.p.t + (self.p.ea / (2.0 * self.p.l)) * self.stretch(py)?)
    }

    // -- initial conditions ----------------------------------------------------------------

    /// Set the initial displacement (and optional velocity), ends clamped.
    ///
    /// The start carries the **nonlinear** tension at `t = 0`, so a single eigenmode opens as a
    /// clean discrete Duffing cosine; at `EA = 0` that term is skipped and the start is model #3's.
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
        self.delta_tension = 0.0;
        self.converged = true;
        Ok(())
    }

    // -- time stepping ---------------------------------------------------------------------

    /// Advance one timestep, rolling the history.
    ///
    /// `EA = 0` is one back-substitution against the prefactored model-#3 matrix — the identical
    /// code path, hence bit-identical results. Otherwise a scalar `brentq` root-find for `dT`,
    /// each residual costing one banded refactor and one solve.
    fn step(&mut self, py: Python<'_>) -> PyResult<()> {
        let (u, u_prev) = self.state_pair(py)?;
        let rhs0 = core::step_rhs(&u, &u_prev, &self.p);
        let (sol, d_t, converged, expansions) = if self.p.ea == 0.0 {
            let sol = physsynth_core::banded::cho_solve_banded_upper(
                &self.p.chol0,
                2,
                self.p.interior(),
                &rhs0,
            )
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            (sol, 0.0, true, 0)
        } else {
            let s = core::solve_tension(&rhs0, &u_prev, &self.p)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            (s.u_next, s.delta_tension, s.converged, s.expansions)
        };

        self.bracket_expansions += expansions;
        if !converged {
            // `stacklevel=3` in the original, which puts the warning at the caller of `step`.
            // The step number quoted is `self.n` *before* the increment below.
            self.n_not_converged += 1;
            let msg = format!(
                "Tension solve failed to bracket a root at step {} after {} doublings. \
                 Do not treat this run as physics.",
                self.n,
                core::MAX_BRACKET_EXPANSIONS
            );
            let warnings = py.import("warnings")?;
            let category = py.get_type::<pyo3::exceptions::PyRuntimeWarning>();
            warnings.call_method1(
                "warn",
                PyTuple::new(py, [msg.into_pyobject(py)?.into_any(), category.into_any()])?,
            )?;
        }
        self.delta_tension = d_t;
        self.converged = converged;

        let mut next = vec![0.0; self.p.nodes()];
        next[1..self.p.nodes() - 1].copy_from_slice(&sol);
        self.u_prev = self.swap_u(py, next);
        self.n += 1;
        Ok(())
    }

    // -- energy ----------------------------------------------------------------------------

    /// Discrete energy `E^n` (Joules) — model #3's energy plus the nonlinear stretch term.
    fn energy(&self, py: Python<'_>) -> PyResult<f64> {
        let (u, up) = self.state_pair(py)?;
        Ok(core::energy(&u, &up, &self.p))
    }

    /// The stretch (membrane) part of `E^n` alone (J) — `0` iff `EA = 0`.
    fn nonlinear_energy(&self, py: Python<'_>) -> PyResult<f64> {
        let (u, up) = self.state_pair(py)?;
        Ok(core::nonlinear_energy(&u, &up, &self.p))
    }

    /// Displacement at grid node `index` — a pickup for spectral analysis.
    fn displacement_at(&self, py: Python<'_>, index: i64) -> PyResult<f64> {
        let bound = self.u.bind(py);
        let ro = bound.readonly();
        let s = state_slice(&ro, "u")?;
        node_value(s, index)
    }

    /// Not available on this model — **the update matrix is time-varying**.
    ///
    /// The one method the rest of the theta-scheme family implements and this one refuses. See the
    /// module header: it has to be a clean `NotImplementedError`, not a panic, because three
    /// coupled models call it on whatever string they are handed.
    #[allow(non_snake_case)]
    fn apply_Ainv(&self, _rhs_int: PyReadonlyArray1<'_, f64>) -> PyResult<Py<PyArray1<f64>>> {
        Err(PyNotImplementedError::new_err(AINV_REFUSAL))
    }
}
