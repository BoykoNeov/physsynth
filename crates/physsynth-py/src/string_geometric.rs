//! The binding over `physsynth_core::string_geometric` — model #10 wearing the Python interface.
//!
//! # Six settable state arrays, not two
//!
//! Every earlier string exposes `u` and `u_prev`. This one has three fields, and **all six arrays
//! are assigned to from outside**: `tests/helpers.py::seed_rotating_wave` writes the exact
//! two-level helix history rather than going through `set_state` (whose Taylor start would seed an
//! `O(k^3)` error the helix immediately sheds into the longitudinal field — ten orders on the
//! claim being measured), and `tests/test_geometric_rotating_wave.py`,
//! `web/serialize.py::_build_payload_geometric` and `scripts/diagnose_geometric_string.py` do the
//! same. So `u`, `w`, `v`, `u_prev`, `w_prev`, `v_prev`, `n` and `converged` all take setters, and
//! `step` rebinds them the way the original does: after a step `u_prev` **is** the object `u` was.
//!
//! # The private names, and §12.2 for the n-th time
//!
//! `tests/test_geometric_energy.py` calls `_dg_force`, `_dg_jacobian`, `_nl_density`,
//! `_stretch_ratio` and reads `_a`, `_Gp` and `_Gm`; `tests/test_geometric_rotating_wave.py` calls
//! `_dg_jacobian` to cross-check the analysis module's Hessian. A leading underscore is not a
//! statement about the interface, so every one of them is a method or a getter here — including
//! the two that hand back **SciPy** matrices, because `s._Gm @ s._Gp` and `.toarray()` are what
//! the assertions are written in.
//!
//! `_dg_force` and `_dg_jacobian` take `(3, N)` arrays of *arbitrary* strains, not the model's own
//! state: the finite-difference check perturbs one entry at a time and the DG-identity check feeds
//! two unrelated levels. They are therefore pure functions of their arguments here too.
//!
//! # The two warnings
//!
//! Both are Python's, raised from this layer through `PyErr::warn` so `pytest.warns` and `-W
//! error` see them: the `lam_long` accuracy bar at construction (`stacklevel = 1` here for the
//! reason `mallet.rs` gives — a Rust `__new__` has no Python frame of its own, so "my caller" is
//! one level nearer) and the Newton stall per step.

use std::ffi::CString;

use numpy::{PyArray1, PyArray2, PyArrayMethods, PyUntypedArrayMethods};
use physsynth_core::string_geometric as core;
use pyo3::exceptions::{PyIndexError, PyNotImplementedError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyTuple;

use crate::shape::{as_f64_field, shape_repr};
use crate::string_stiff::{bad_boundary, boundary_ok, csr_object};

/// `warnings.warn(..., RuntimeWarning)` with the original's text.
fn warn_runtime(py: Python<'_>, msg: String) -> PyResult<()> {
    let category = py.get_type::<pyo3::exceptions::PyRuntimeWarning>();
    let msg = CString::new(msg).map_err(|_| PyValueError::new_err("warning text had a NUL"))?;
    PyErr::warn(py, category.as_any(), &msg, 1)
}

/// A geometrically exact stiff string — the Rust implementation, wearing the Python interface.
///
/// Attribute-for-attribute and method-for-method compatible with
/// `physsynth.core.string_geometric.GeometricString`; the docstring on that class is the reference.
#[pyclass(name = "GeometricString", module = "physsynth_rs")]
pub struct PyGeometricString {
    p: core::Params,
    boundary: Py<PyAny>,
    x: Py<PyArray1<f64>>,
    d2: Py<PyAny>,
    l_u: Py<PyAny>,
    l_w: Py<PyAny>,
    l_v: Py<PyAny>,
    a_u: Py<PyAny>,
    a_w: Py<PyAny>,
    a_v: Py<PyAny>,
    a3: Py<PyAny>,
    gp: Py<PyAny>,
    gm: Py<PyAny>,
    gp3: Py<PyAny>,
    gm3: Py<PyAny>,
    chol_u: Py<PyArray2<f64>>,
    chol_w: Py<PyArray2<f64>>,
    chol_v: Py<PyArray2<f64>>,
    u: Py<PyArray1<f64>>,
    w: Py<PyArray1<f64>>,
    v: Py<PyArray1<f64>>,
    u_prev: Py<PyArray1<f64>>,
    w_prev: Py<PyArray1<f64>>,
    v_prev: Py<PyArray1<f64>>,
    n: usize,
    converged: bool,
    newton_iters: usize,
    total_newton_iters: usize,
    n_not_converged: usize,
}

impl PyGeometricString {
    /// Validate an array being assigned to one of the six state fields and take ownership of it.
    fn adopt(&self, value: &Bound<'_, PyAny>, name: &str) -> PyResult<Py<PyArray1<f64>>> {
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
        Ok(arr.unbind())
    }

    /// The six state arrays as slices, in the order the core functions take them.
    fn fields(&self, py: Python<'_>) -> PyResult<[Vec<f64>; 6]> {
        let read = |a: &Py<PyArray1<f64>>, name: &str| -> PyResult<Vec<f64>> {
            let bound = a.bind(py);
            let ro = bound.readonly();
            Ok(ro
                .as_slice()
                .map_err(|_| {
                    PyValueError::new_err(format!("{name} must be a contiguous 1-D float64 array."))
                })?
                .to_vec())
        };
        Ok([
            read(&self.u, "u")?,
            read(&self.w, "w")?,
            read(&self.v, "v")?,
            read(&self.u_prev, "u_prev")?,
            read(&self.w_prev, "w_prev")?,
            read(&self.v_prev, "v_prev")?,
        ])
    }

    /// `_as_field`: a scalar fills the full grid, a `(N + 1,)` array is copied, anything else is a
    /// `ValueError` quoting the shape Python's `repr` would print.
    fn as_field(&self, py: Python<'_>, value: &Bound<'_, PyAny>, name: &str) -> PyResult<Vec<f64>> {
        let (shape, values) = as_f64_field(py, value, name)?;
        if shape.is_empty() {
            return Ok(vec![values[0]; self.p.nodes()]);
        }
        if shape != vec![self.p.nodes()] {
            return Err(PyValueError::new_err(format!(
                "{name} must have shape ({},), got {}.",
                self.p.nodes(),
                shape_repr(&shape)
            )));
        }
        Ok(values)
    }
}

/// Read a `(3, N)` strain array — the argument `_dg_force`, `_dg_jacobian`, `_nl_density` and
/// `_stretch_ratio` all take — as one field-major `Vec`.
fn as_strain(py: Python<'_>, obj: &Bound<'_, PyAny>, n: usize, name: &str) -> PyResult<Vec<f64>> {
    let (shape, values) = as_f64_field(py, obj, name)?;
    if shape != vec![3, n] {
        return Err(PyValueError::new_err(format!(
            "{name} must have shape (3, {n}), got {}.",
            shape_repr(&shape)
        )));
    }
    Ok(values)
}

/// A banded Cholesky factor back to Python with `scipy.linalg.cholesky_banded`'s **shape**.
///
/// `(3, n)`, not a flat `3n`. The first draft returned the flat buffer the core stores and every
/// value in it was right, which is §25.7's lesson once more: a shape is part of the interface, and
/// `np.array_equal` is the only thing in the suite that would have noticed.
fn band_array(py: Python<'_>, band: &[f64]) -> Py<PyArray2<f64>> {
    let n = band.len() / 3;
    let rows: Vec<Vec<f64>> = (0..3).map(|r| band[r * n..(r + 1) * n].to_vec()).collect();
    PyArray2::from_vec2(py, &rows)
        .expect("three rows of equal length")
        .unbind()
}

/// A `(3, N)` array back to Python, laid out the way `np.stack` lays one out.
fn strain_array(py: Python<'_>, values: &[f64], n: usize) -> Py<PyArray2<f64>> {
    let rows: Vec<Vec<f64>> = (0..3)
        .map(|f| values[f * n..(f + 1) * n].to_vec())
        .collect();
    PyArray2::from_vec2(py, &rows)
        .expect("three rows of equal length")
        .unbind()
}

#[pymethods]
impl PyGeometricString {
    #[allow(clippy::too_many_arguments)]
    #[new]
    #[pyo3(signature = (*, L, T, rho, fs, N, EA, kappa=0.0, kappa_w=None, sigma0=0.0, sigma1=0.0,
                        sigma0_long=None, sigma1_long=None,
                        theta=physsynth_core::string_stiff::THETA_DEFAULT,
                        boundary=None::<Py<PyAny>>, newton_tol=core::NEWTON_TOL_DEFAULT,
                        newton_maxiter=core::NEWTON_MAXITER_DEFAULT, allow_softening=false))]
    fn new(
        py: Python<'_>,
        #[allow(non_snake_case)] L: f64,
        #[allow(non_snake_case)] T: f64,
        rho: f64,
        fs: f64,
        #[allow(non_snake_case)] N: i64,
        #[allow(non_snake_case)] EA: f64,
        kappa: f64,
        kappa_w: Option<f64>,
        sigma0: f64,
        sigma1: f64,
        sigma0_long: Option<f64>,
        sigma1_long: Option<f64>,
        theta: f64,
        // `Option<Option<_>>` so that an OMITTED `boundary` and an explicit `boundary=None` stay
        // distinguishable — §24.7, where a plain `Option` silently built the default.
        boundary: Option<Option<Py<PyAny>>>,
        newton_tol: f64,
        newton_maxiter: i64,
        allow_softening: bool,
    ) -> PyResult<Self> {
        // The arm order is §24.7's finding and is pinned by a test: PyO3 wraps the DEFAULT
        // expression, so `Some(None)` is "argument omitted" and a bare `None` is the caller's
        // literal `None`, which the original rejects with a message quoting it.
        let boundary = match boundary {
            Some(None) => pyo3::types::PyString::new(py, "supported").into_any(),
            None => py.None().into_bound(py),
            Some(Some(b)) => b.into_bound(py),
        };
        let ok = boundary_ok(&boundary);
        let p = core::Params::new(
            L,
            T,
            rho,
            fs,
            N,
            EA,
            kappa,
            kappa_w,
            sigma0,
            sigma1,
            sigma0_long,
            sigma1_long,
            theta,
            ok,
            newton_tol,
            newton_maxiter,
            allow_softening,
        )
        .map_err(|e| match e {
            core::ParamError::BadBoundary => bad_boundary(&boundary),
            other => PyValueError::new_err(other.to_string()),
        })?;

        if p.warn_lam_long {
            warn_runtime(
                py,
                format!(
                    "lam_long = {:.2} > {}: the longitudinal field advances {:.1} cells per \
                     timestep and is under-resolved in time. The scheme is unconditionally STABLE \
                     here, so no CFL is violated and nothing else will warn -- but stable is not \
                     accurate: past lam_long ~ 4 the Newton solve stops converging and energy \
                     drift explodes (measured 1e+3 .. 1e+5). c_long/c = sqrt(EA/T) = {:.1}, so a \
                     transverse lam of {} buys this. Raise fs (or lower EA) until lam_long <= 1.",
                    p.lam_long,
                    physsynth_core::fmt::py_float(core::LAM_LONG_WARN),
                    p.lam_long,
                    p.ea_over_t.sqrt(),
                    physsynth_core::fmt::py_general(p.lam, 3),
                ),
            )?;
        }

        let nodes = p.nodes();
        let zeros = || PyArray1::from_vec(py, vec![0.0; nodes]).unbind();
        Ok(PyGeometricString {
            x: PyArray1::from_vec(py, p.grid()).unbind(),
            d2: csr_object(py, &p.d2)?,
            l_u: csr_object(py, &p.l_u)?,
            l_w: csr_object(py, &p.l_w)?,
            l_v: csr_object(py, &p.l_v)?,
            a_u: csr_object(py, &p.a_u)?,
            a_w: csr_object(py, &p.a_w)?,
            a_v: csr_object(py, &p.a_v)?,
            a3: csr_object(py, &p.a3)?,
            gp: csr_object(py, &p.gp)?,
            gm: csr_object(py, &p.gm)?,
            gp3: csr_object(py, &p.gp3)?,
            gm3: csr_object(py, &p.gm3)?,
            chol_u: band_array(py, &p.chol_u),
            chol_w: band_array(py, &p.chol_w),
            chol_v: band_array(py, &p.chol_v),
            u: zeros(),
            w: zeros(),
            v: zeros(),
            u_prev: zeros(),
            w_prev: zeros(),
            v_prev: zeros(),
            n: 0,
            converged: true,
            newton_iters: 0,
            total_newton_iters: 0,
            n_not_converged: 0,
            boundary: boundary.unbind(),
            p,
        })
    }

    // -- parameters ---------------------------------------------------------------------------

    #[getter]
    #[allow(non_snake_case)]
    fn L(&self) -> f64 {
        self.p.l
    }
    #[getter]
    #[allow(non_snake_case)]
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
    #[allow(non_snake_case)]
    fn N(&self) -> usize {
        self.p.n
    }
    #[getter]
    #[allow(non_snake_case)]
    fn EA(&self) -> f64 {
        self.p.ea
    }
    #[getter]
    fn kappa(&self) -> f64 {
        self.p.kappa
    }
    #[getter]
    fn kappa_u(&self) -> f64 {
        self.p.kappa_u
    }
    #[getter]
    fn kappa_w(&self) -> f64 {
        self.p.kappa_w
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
    fn sigma0_long(&self) -> f64 {
        self.p.sigma0_long
    }
    #[getter]
    fn sigma1_long(&self) -> f64 {
        self.p.sigma1_long
    }
    #[getter]
    fn theta(&self) -> f64 {
        self.p.theta
    }
    #[getter]
    fn boundary(&self, py: Python<'_>) -> Py<PyAny> {
        self.boundary.clone_ref(py)
    }
    #[getter]
    fn newton_tol(&self) -> f64 {
        self.p.newton_tol
    }
    #[getter]
    fn newton_maxiter(&self) -> usize {
        self.p.newton_maxiter
    }
    #[getter]
    fn allow_softening(&self) -> bool {
        self.p.allow_softening
    }
    #[getter]
    fn c(&self) -> f64 {
        self.p.c
    }
    #[getter]
    fn c_long(&self) -> f64 {
        self.p.c_long
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
    fn lam_long(&self) -> f64 {
        self.p.lam_long
    }
    #[getter]
    #[allow(non_snake_case)]
    fn B(&self) -> f64 {
        self.p.b
    }
    #[getter]
    #[allow(non_snake_case)]
    fn EA_over_T(&self) -> f64 {
        self.p.ea_over_t
    }
    #[getter]
    fn x(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        self.x.clone_ref(py)
    }

    /// `a = EA - T0`, **the** nonlinearity coefficient — private by name, read by the tests.
    #[getter]
    fn _a(&self) -> f64 {
        self.p.a
    }

    // -- the constant operators, as SciPy matrices ---------------------------------------------

    #[getter]
    #[allow(non_snake_case)]
    fn _D2(&self, py: Python<'_>) -> Py<PyAny> {
        self.d2.clone_ref(py)
    }
    #[getter]
    #[allow(non_snake_case)]
    fn _L_u(&self, py: Python<'_>) -> Py<PyAny> {
        self.l_u.clone_ref(py)
    }
    #[getter]
    #[allow(non_snake_case)]
    fn _L_w(&self, py: Python<'_>) -> Py<PyAny> {
        self.l_w.clone_ref(py)
    }
    #[getter]
    #[allow(non_snake_case)]
    fn _L_v(&self, py: Python<'_>) -> Py<PyAny> {
        self.l_v.clone_ref(py)
    }
    #[getter]
    #[allow(non_snake_case)]
    fn _A_u(&self, py: Python<'_>) -> Py<PyAny> {
        self.a_u.clone_ref(py)
    }
    #[getter]
    #[allow(non_snake_case)]
    fn _A_w(&self, py: Python<'_>) -> Py<PyAny> {
        self.a_w.clone_ref(py)
    }
    #[getter]
    #[allow(non_snake_case)]
    fn _A_v(&self, py: Python<'_>) -> Py<PyAny> {
        self.a_v.clone_ref(py)
    }
    #[getter]
    #[allow(non_snake_case)]
    fn _A3(&self, py: Python<'_>) -> Py<PyAny> {
        self.a3.clone_ref(py)
    }
    #[getter]
    #[allow(non_snake_case)]
    fn _Gp(&self, py: Python<'_>) -> Py<PyAny> {
        self.gp.clone_ref(py)
    }
    #[getter]
    #[allow(non_snake_case)]
    fn _Gm(&self, py: Python<'_>) -> Py<PyAny> {
        self.gm.clone_ref(py)
    }
    #[getter]
    #[allow(non_snake_case)]
    fn _Gp3(&self, py: Python<'_>) -> Py<PyAny> {
        self.gp3.clone_ref(py)
    }
    #[getter]
    #[allow(non_snake_case)]
    fn _Gm3(&self, py: Python<'_>) -> Py<PyAny> {
        self.gm3.clone_ref(py)
    }
    #[getter]
    fn _chol_u(&self, py: Python<'_>) -> Py<PyArray2<f64>> {
        self.chol_u.clone_ref(py)
    }
    #[getter]
    fn _chol_w(&self, py: Python<'_>) -> Py<PyArray2<f64>> {
        self.chol_w.clone_ref(py)
    }
    #[getter]
    fn _chol_v(&self, py: Python<'_>) -> Py<PyArray2<f64>> {
        self.chol_v.clone_ref(py)
    }

    // -- state ----------------------------------------------------------------------------------

    #[getter]
    fn u(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        self.u.clone_ref(py)
    }
    #[setter]
    fn set_u(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.u = self.adopt(value, "u")?;
        Ok(())
    }
    #[getter]
    fn w(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        self.w.clone_ref(py)
    }
    #[setter]
    fn set_w(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.w = self.adopt(value, "w")?;
        Ok(())
    }
    #[getter]
    fn v(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        self.v.clone_ref(py)
    }
    #[setter]
    fn set_v(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.v = self.adopt(value, "v")?;
        Ok(())
    }
    #[getter]
    fn u_prev(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        self.u_prev.clone_ref(py)
    }
    #[setter]
    fn set_u_prev(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.u_prev = self.adopt(value, "u_prev")?;
        Ok(())
    }
    #[getter]
    fn w_prev(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        self.w_prev.clone_ref(py)
    }
    #[setter]
    fn set_w_prev(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.w_prev = self.adopt(value, "w_prev")?;
        Ok(())
    }
    #[getter]
    fn v_prev(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        self.v_prev.clone_ref(py)
    }
    #[setter]
    fn set_v_prev(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.v_prev = self.adopt(value, "v_prev")?;
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
    #[getter]
    fn converged(&self) -> bool {
        self.converged
    }
    #[setter]
    fn set_converged(&mut self, value: bool) {
        self.converged = value;
    }
    #[getter]
    fn newton_iters(&self) -> usize {
        self.newton_iters
    }
    #[getter]
    fn total_newton_iters(&self) -> usize {
        self.total_newton_iters
    }
    #[getter]
    fn n_not_converged(&self) -> usize {
        self.n_not_converged
    }

    /// The three displacement fields as a `GeometricState` — copies, safe to mutate.
    ///
    /// The named tuple is the Python module's own: a Rust class rebound under the flag does not
    /// take `GeometricState` with it, and callers unpack it by name.
    #[getter]
    fn state(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let cls = py
            .import("physsynth.core.string_geometric")?
            .getattr("GeometricState")?;
        let copy = |a: &Py<PyArray1<f64>>| -> PyResult<Py<PyAny>> {
            Ok(a.bind(py).call_method0("copy")?.unbind())
        };
        Ok(cls
            .call1((copy(&self.u)?, copy(&self.w)?, copy(&self.v)?))?
            .unbind())
    }

    /// Whether the two polarizations are exactly degenerate.
    #[getter]
    fn is_degenerate(&self) -> bool {
        self.p.kappa_u == self.p.kappa_w
    }

    /// Lower bound on [`Self::energy`] — zero, for the reason in the Python docstring.
    #[getter]
    fn energy_floor(&self) -> f64 {
        0.0
    }

    /// Per-cell stretch ratio `Lambda^n`.
    #[getter]
    fn stretch_ratio(&self, py: Python<'_>) -> PyResult<Py<PyArray1<f64>>> {
        let [u, w, v, ..] = self.fields(py)?;
        let q = core::strain(&u, &w, &v, self.p.h);
        Ok(PyArray1::from_vec(py, core::stretch_ratio(&q)).unbind())
    }

    /// Per-cell axial tension `T(Lambda) = EA Lambda - (EA - T0)`.
    #[getter]
    fn tension(&self, py: Python<'_>) -> PyResult<Py<PyArray1<f64>>> {
        let [u, w, v, ..] = self.fields(py)?;
        let q = core::strain(&u, &w, &v, self.p.h);
        Ok(PyArray1::from_vec(py, core::tension(&q, self.p.ea, self.p.a)).unbind())
    }

    // -- initial conditions ---------------------------------------------------------------------

    #[pyo3(signature = (u0=None, w0=None, v0=None, *, u_dot=None, w_dot=None, v_dot=None))]
    #[allow(clippy::too_many_arguments)]
    fn set_state(
        &mut self,
        py: Python<'_>,
        u0: Option<&Bound<'_, PyAny>>,
        w0: Option<&Bound<'_, PyAny>>,
        v0: Option<&Bound<'_, PyAny>>,
        u_dot: Option<&Bound<'_, PyAny>>,
        w_dot: Option<&Bound<'_, PyAny>>,
        v_dot: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let nodes = self.p.nodes();
        let field = |o: Option<&Bound<'_, PyAny>>, name: &str| -> PyResult<Vec<f64>> {
            match o {
                None => Ok(vec![0.0; nodes]),
                Some(obj) => self.as_field(py, obj, name),
            }
        };
        let mut u = field(u0, "u0")?;
        let mut w = field(w0, "w0")?;
        let mut v = field(v0, "v0")?;
        let mut dots = [
            field(u_dot, "u_dot")?,
            field(w_dot, "w_dot")?,
            field(v_dot, "v_dot")?,
        ];
        let last = nodes - 1;
        for d in dots.iter_mut() {
            d[0] = 0.0;
            d[last] = 0.0;
        }
        let [up, wp, vp] = core::initial_previous(&mut u, &mut w, &mut v, &dots, &self.p);
        self.u = PyArray1::from_vec(py, u).unbind();
        self.w = PyArray1::from_vec(py, w).unbind();
        self.v = PyArray1::from_vec(py, v).unbind();
        self.u_prev = PyArray1::from_vec(py, up).unbind();
        self.w_prev = PyArray1::from_vec(py, wp).unbind();
        self.v_prev = PyArray1::from_vec(py, vp).unbind();
        self.n = 0;
        self.converged = true;
        self.newton_iters = 0;
        Ok(())
    }

    // -- time stepping --------------------------------------------------------------------------

    /// Advance one timestep (rolls the history).
    fn step(&mut self, py: Python<'_>) -> PyResult<()> {
        let [u, w, v, up, wp, vp] = self.fields(py)?;
        let p = &self.p;
        let rhs = [
            core::step_rhs(&u, &up, &p.l_u, p.sigma0, p.sigma1, p),
            core::step_rhs(&w, &wp, &p.l_w, p.sigma0, p.sigma1, p),
            core::step_rhs(&v, &vp, &p.l_v, p.sigma0_long, p.sigma1_long, p),
        ];
        let n_int = p.interior();

        let (interiors, stall) = if p.a == 0.0 {
            self.converged = true;
            self.newton_iters = 0;
            let mut y = Vec::with_capacity(3 * n_int);
            for (r, chol) in rhs.iter().zip([&p.chol_u, &p.chol_w, &p.chol_v]) {
                y.extend(
                    physsynth_core::banded::cho_solve_banded_upper(chol, 2, n_int, r)
                        .map_err(|e| PyValueError::new_err(e.to_string()))?,
                );
            }
            (y, None)
        } else {
            let rep = core::solve_newton(&rhs, &up, &wp, &vp, p)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            self.converged = rep.converged;
            self.newton_iters = rep.iters;
            self.total_newton_iters += rep.iters;
            let stall = if rep.converged {
                None
            } else {
                self.n_not_converged += 1;
                Some((rep.residual, rep.tol_abs))
            };
            (rep.y, stall)
        };

        // Rebind, so that after the step `u_prev` **is** the object `u` was — the original's
        // `self.u_prev, ... = self.u, ...` followed by a fresh `rolled`.
        let nodes = self.p.nodes();
        let fresh = |f: usize| -> Py<PyArray1<f64>> {
            let mut full = vec![0.0; nodes];
            full[1..nodes - 1].copy_from_slice(&interiors[f * n_int..(f + 1) * n_int]);
            PyArray1::from_vec(py, full).unbind()
        };
        self.u_prev = std::mem::replace(&mut self.u, fresh(0));
        self.w_prev = std::mem::replace(&mut self.w, fresh(1));
        self.v_prev = std::mem::replace(&mut self.v, fresh(2));
        let step_index = self.n;
        self.n += 1;

        // The warning comes last, after the state is consistent, because `-W error` turns it into
        // an exception and a half-stepped model would be worse than a raised one.
        if let Some((residual, tol_abs)) = stall {
            warn_runtime(
                py,
                format!(
                    "Geometric string Newton solve did not converge at step {} in {} iterations \
                     (residual {} > {}); energy may drift. The DG force is exact only *at* the \
                     root. Raise newton_maxiter or oversample. Do not treat this run as physics.",
                    step_index,
                    self.p.newton_maxiter,
                    physsynth_core::fmt::py_exp(residual, 2),
                    physsynth_core::fmt::py_exp(tol_abs, 1),
                ),
            )?;
        }
        Ok(())
    }

    // -- diagnostics ----------------------------------------------------------------------------

    /// Discrete energy `E^n` (J).
    fn energy(&self, py: Python<'_>) -> PyResult<f64> {
        let [u, w, v, up, wp, vp] = self.fields(py)?;
        Ok(core::linear_energy(&u, &w, &v, &up, &wp, &vp, &self.p)
            + core::nonlinear_energy(&u, &w, &v, &up, &wp, &vp, &self.p))
    }

    /// The nonlinear excess part of `E^n` alone (J).
    fn nonlinear_energy(&self, py: Python<'_>) -> PyResult<f64> {
        let [u, w, v, up, wp, vp] = self.fields(py)?;
        Ok(core::nonlinear_energy(&u, &w, &v, &up, &wp, &vp, &self.p))
    }

    /// The `v` field's kinetic + linear-potential energy alone (J).
    fn longitudinal_energy(&self, py: Python<'_>) -> PyResult<f64> {
        let [_, _, v, _, _, vp] = self.fields(py)?;
        Ok(core::longitudinal_energy(&v, &vp, &self.p))
    }

    /// Transverse `u` displacement at grid node `index` — the pickup.
    fn displacement_at(&self, py: Python<'_>, index: isize) -> PyResult<f64> {
        let bound = self.u.bind(py);
        let ro = bound.readonly();
        let s = ro
            .as_slice()
            .map_err(|_| PyValueError::new_err("u must be a contiguous 1-D float64 array."))?;
        // `IndexError`, not `ValueError`: the original is `float(self.u[index])` and the exception
        // a caller catches is NumPy's. Reproducing a rejection means reproducing its TYPE as well
        // as its text — the same rule `test_rust_parity_banded.py` keeps for `LinAlgError`.
        let i = if index < 0 {
            let back = (-index) as usize;
            if back > s.len() {
                return Err(PyIndexError::new_err(format!(
                    "index {index} is out of bounds for axis 0 with size {}",
                    s.len()
                )));
            }
            s.len() - back
        } else {
            let i = index as usize;
            if i >= s.len() {
                return Err(PyIndexError::new_err(format!(
                    "index {index} is out of bounds for axis 0 with size {}",
                    s.len()
                )));
            }
            i
        };
        Ok(s[i])
    }

    /// Not available on this model — **the one-step response is state-dependent**.
    ///
    /// The argument is untyped on purpose: the original raises before looking at it, so a caller
    /// who passes a list must get `NotImplementedError` and not a `TypeError` from the extraction.
    #[allow(unused_variables)]
    #[pyo3(name = "apply_Ainv")]
    fn apply_ainv(&self, rhs_int: &Bound<'_, PyAny>) -> PyResult<()> {
        Err(PyNotImplementedError::new_err(
            "GeometricString's one-step admittance is state-dependent: A3 is constant, but the \
             implicit discrete-gradient force makes the true response the inverse of the Newton \
             Jacobian, not of A3. Coupling an exciter here requires a joint solve -- see \
             docs/dev/geometrically-exact-string-plan.md.",
        ))
    }

    // -- the private kernels the tests reach for ------------------------------------------------

    /// Cell strains `q = (u_x, w_x, v_x)` as `(3, N)` from full-grid fields.
    fn _strain(
        &self,
        py: Python<'_>,
        u: &Bound<'_, PyAny>,
        w: &Bound<'_, PyAny>,
        v: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyArray2<f64>>> {
        let (_, u) = as_f64_field(py, u, "u")?;
        let (_, w) = as_f64_field(py, w, "w")?;
        let (_, v) = as_f64_field(py, v, "v")?;
        let q = core::strain(&u, &w, &v, self.p.h);
        Ok(strain_array(py, &q, u.len() - 1))
    }

    /// `Lambda` per cell from a `(3, N)` strain array — a static method on the original.
    #[staticmethod]
    fn _stretch_ratio(py: Python<'_>, q: &Bound<'_, PyAny>) -> PyResult<Py<PyArray1<f64>>> {
        let (shape, values) = as_f64_field(py, q, "q")?;
        if shape.len() != 2 || shape[0] != 3 {
            return Err(PyValueError::new_err(format!(
                "q must have shape (3, N), got {}.",
                shape_repr(&shape)
            )));
        }
        Ok(PyArray1::from_vec(py, core::stretch_ratio(&values)).unbind())
    }

    /// `(Lambda, Lambda-1, Lambda-(1+v_x), r^2, Lambda+1+v_x)` — a static method on the original.
    #[staticmethod]
    fn _stretch_terms<'py>(
        py: Python<'py>,
        q: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyTuple>> {
        let (shape, values) = as_f64_field(py, q, "q")?;
        if shape.len() != 2 || shape[0] != 3 {
            return Err(PyValueError::new_err(format!(
                "q must have shape (3, N), got {}.",
                shape_repr(&shape)
            )));
        }
        let t = core::stretch_terms(&values);
        PyTuple::new(
            py,
            [t.lam, t.lam_m1, t.d, t.r2, t.denom]
                .into_iter()
                .map(|v| PyArray1::from_vec(py, v)),
        )
    }

    /// The exact discrete gradient `gradbar V_nl` per cell, `(3, N)`.
    fn _dg_force(
        &self,
        py: Python<'_>,
        q_plus: &Bound<'_, PyAny>,
        q_minus: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyArray2<f64>>> {
        let n = self.p.n;
        let qp = as_strain(py, q_plus, n, "q_plus")?;
        let qm = as_strain(py, q_minus, n, "q_minus")?;
        Ok(strain_array(py, &core::dg_force(&qp, &qm, self.p.a), n))
    }

    /// `d(gradbar V_nl)/d q+` as a `3N x 3N` SciPy matrix of diagonal blocks.
    fn _dg_jacobian(
        &self,
        py: Python<'_>,
        q_plus: &Bound<'_, PyAny>,
        q_minus: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let n = self.p.n;
        let qp = as_strain(py, q_plus, n, "q_plus")?;
        let qm = as_strain(py, q_minus, n, "q_minus")?;
        csr_object(py, &core::dg_jacobian(&qp, &qm, self.p.a))
    }

    /// `h sum_c V_nl(q_c)` (J) — the nonlinear excess density.
    fn _nl_density(&self, py: Python<'_>, q: &Bound<'_, PyAny>) -> PyResult<f64> {
        let q = as_strain(py, q, self.p.n, "q")?;
        Ok(core::nl_density(&q, self.p.a, self.p.h))
    }
}
