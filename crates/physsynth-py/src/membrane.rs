//! The binding over `physsynth_core::membrane` — model #4 wearing the Python interface.
//!
//! # Two classes of buffer, and the line between them is the whole design
//!
//! §9.3 of the migration plan established that mutable state must live in **Python-owned NumPy
//! arrays**, because `step()` rebinds rather than overwrites and real callers depend on both
//! halves of that (a reference held across a step stays a valid snapshot; a write *through* `.u`
//! reaches the model). The membrane is the first model to hold arrays that are *not* mutable
//! state, so the rule needs its second half stated:
//!
//! - **`u`, `u_prev`** — rebound every step. Python-owned `PyArray1`, fresh array per step,
//!   settable, exactly as `PyIdealString`. `airbox._MembraneSurface.commit` assigns straight into
//!   both, so this is load-bearing here and not a hypothetical.
//! - **`X`, `Y`, `mask`, `index_map`, `L`** — immutable after construction. Built **once** in the
//!   constructor and handed back by `clone_ref`.
//! - **`state`** — a fresh 2-D array per call, because it is `embed`, which is a copy in the
//!   original too.
//!
//! The immutable half is not a micro-optimisation. `airbox._MembraneSurface.rhs` evaluates
//! `m.L @ m.u` **every timestep**, so a `L` getter that rebuilt a `csr_matrix` per access would
//! assemble a sparse matrix inside the inner loop of the heaviest tests in the suite — passing
//! every physics bar while making the flagged run mysteriously slower than the Python one. Build
//! once, hand back a reference.
//!
//! # This is the first place the binding calls SciPy
//!
//! `L` is a `scipy.sparse.csr_matrix` on the instance, not a function return, so Phase 1's trick
//! of handing back triplets and rebuilding in a Python shim does not apply — there is no call to
//! wrap. The constructor therefore imports `scipy.sparse` and builds the object itself.
//!
//! That is a new fact about this crate and worth naming: **`physsynth-py` is a SciPy client.**
//! It does not weaken the portability rule, which is about `physsynth-core` — that crate's
//! dependency list is still empty and `crates/physsynth-core/tests/deps.rs` still enforces it.
//! The binding is temporary by construction (plan §1) and exists precisely to speak Python's
//! numeric dialect; the day it is deleted, so is this import.

use crate::shape::{as_f64_field, shape_repr, to_2d_bool, to_2d_f64, to_2d_i64};
use crate::{csr_triplets, state_slice};
use numpy::{PyArray1, PyArrayMethods, PyUntypedArrayMethods};
use physsynth_core::membrane as core;
use physsynth_core::ops2d;
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// A discretized membrane resonator — the Rust implementation, wearing the Python interface.
///
/// Attribute-for-attribute and method-for-method compatible with
/// `physsynth.core.membrane.Membrane`; the docstring on that class is the reference.
#[pyclass(name = "Membrane", module = "physsynth_rs")]
pub struct PyMembrane {
    p: core::Params,
    x: Py<PyAny>,
    y: Py<PyAny>,
    mask: Py<PyAny>,
    index_map: Py<PyAny>,
    l: Py<PyAny>,
    u: Py<PyArray1<f64>>,
    u_prev: Py<PyArray1<f64>>,
    n: usize,
}

impl PyMembrane {
    /// Rebind `self.u` to `values`, returning the array object that was there before.
    fn swap_u(&mut self, py: Python<'_>, values: Vec<f64>) -> Py<PyArray1<f64>> {
        let fresh = PyArray1::from_vec(py, values).unbind();
        std::mem::replace(&mut self.u, fresh)
    }

    /// Validate an array being assigned to `.u` or `.u_prev` and take ownership of it.
    ///
    /// As on the string: Python would accept any object, this accepts any contiguous 1-D float64
    /// array of the right length and rejects the rest loudly. A migration wants a wrong assignment
    /// to fail at the assignment, not three models downstream.
    fn adopt_state(&self, value: &Bound<'_, PyAny>, name: &str) -> PyResult<Py<PyArray1<f64>>> {
        let arr: Bound<'_, PyArray1<f64>> = value.clone().cast_into().map_err(|_| {
            PyValueError::new_err(format!("{name} must be a 1-D float64 numpy array."))
        })?;
        let ro = arr.readonly();
        if ro.len() != self.p.n_live() {
            return Err(PyValueError::new_err(format!(
                "{name} must have shape ({},), got ({},).",
                self.p.n_live(),
                ro.len()
            )));
        }
        state_slice(&ro, name)?;
        Ok(arr.unbind())
    }

    /// The mask's shape, as Python prints it.
    fn mask_shape_repr(&self) -> String {
        let (nrows, ncols) = self.p.shape();
        shape_repr(&[nrows, ncols])
    }

    /// Reduce a `u0`/`v0` argument to a live-node vector: a full 2-D field is selected through the
    /// mask, a live-length vector is taken as is, anything else is the caller's mistake.
    fn to_live_arg(&self, shape: &[usize], values: Vec<f64>, name: &str) -> PyResult<Vec<f64>> {
        let (nrows, ncols) = self.p.shape();
        if shape == [nrows, ncols] {
            return Ok(self.p.to_live(&values));
        }
        if shape == [self.p.n_live()] {
            return Ok(values);
        }
        Err(PyValueError::new_err(format!(
            "{name} must have shape {} (full field) or {} (live), got {}.",
            self.mask_shape_repr(),
            shape_repr(&[self.p.n_live()]),
            shape_repr(shape)
        )))
    }
}

#[pymethods]
impl PyMembrane {
    // Nine keyword arguments plus the GIL token, over Clippy's limit of seven. The shape is not
    // negotiable: this signature IS `Membrane.__init__`, and every call site in `tests/` and
    // `web/serialize.py` spells it out.
    #[allow(clippy::too_many_arguments)]
    #[new]
    #[pyo3(signature = (*, domain, T, rho, fs, N, Lx=None, Ly=None, radius=None, sigma=0.0))]
    fn new(
        py: Python<'_>,
        domain: &Bound<'_, PyAny>,
        T: f64,
        rho: f64,
        fs: f64,
        N: i64,
        Lx: Option<f64>,
        Ly: Option<f64>,
        radius: Option<f64>,
        sigma: f64,
    ) -> PyResult<Self> {
        let parsed = domain
            .extract::<String>()
            .ok()
            .and_then(|s| core::Domain::parse(&s));
        let p = core::Params::new(parsed, T, rho, fs, N, Lx, Ly, radius, sigma).map_err(
            |e| match e {
                core::ParamError::BadDomain => {
                    let shown = domain
                        .repr()
                        .map(|r| r.to_string_lossy().into_owned())
                        .unwrap_or_else(|_| "<unrepresentable>".to_owned());
                    PyValueError::new_err(format!(
                        "domain must be 'rectangle' or 'circle', got {shown}."
                    ))
                }
                other => PyValueError::new_err(other.to_string()),
            },
        )?;

        let (nrows, ncols) = p.shape();
        let x = to_2d_f64(py, p.x.clone(), nrows, ncols)?;
        let y = to_2d_f64(py, p.y.clone(), nrows, ncols)?;
        let mask = to_2d_bool(py, p.mask.flags().to_vec(), nrows, ncols)?;
        let index_map = to_2d_i64(py, p.index_map.clone(), nrows, ncols)?;

        // Built once — see the module docstring. `airbox` does `L @ u` every step.
        let scipy = py.import("scipy.sparse")?;
        let (data, indices, indptr, shape) = csr_triplets(py, &p.l)?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("shape", shape)?;
        let l = scipy
            .call_method("csr_matrix", ((data, indices, indptr),), Some(&kwargs))?
            .unbind();

        let n_live = p.n_live();
        Ok(PyMembrane {
            p,
            x,
            y,
            mask,
            index_map,
            l,
            u: PyArray1::from_vec(py, vec![0.0; n_live]).unbind(),
            u_prev: PyArray1::from_vec(py, vec![0.0; n_live]).unbind(),
            n: 0,
        })
    }

    // -- parameters ------------------------------------------------------------------------

    #[getter]
    fn domain(&self) -> &'static str {
        match self.p.domain {
            core::Domain::Rectangle => "rectangle",
            core::Domain::Circle => "circle",
        }
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
    fn sigma(&self) -> f64 {
        self.p.sigma
    }
    #[getter]
    fn c(&self) -> f64 {
        self.p.c
    }
    #[getter]
    fn k(&self) -> f64 {
        self.p.k
    }
    #[getter]
    fn h(&self) -> f64 {
        self.p.h
    }
    #[getter]
    fn lam(&self) -> f64 {
        self.p.lam
    }
    /// `None` on a circle, as the original sets it.
    #[getter]
    fn Lx(&self) -> Option<f64> {
        self.p.lx
    }
    /// The **snapped** height — an integer number of square cells.
    #[getter]
    fn Ly(&self) -> Option<f64> {
        self.p.ly
    }
    /// `None` on a rectangle.
    #[getter]
    fn radius(&self) -> Option<f64> {
        self.p.radius
    }
    #[getter]
    fn n_live(&self) -> usize {
        self.p.n_live()
    }

    // -- the immutable grid objects, built once ---------------------------------------------

    #[getter]
    fn X(&self, py: Python<'_>) -> Py<PyAny> {
        self.x.clone_ref(py)
    }
    #[getter]
    fn Y(&self, py: Python<'_>) -> Py<PyAny> {
        self.y.clone_ref(py)
    }
    #[getter]
    fn mask(&self, py: Python<'_>) -> Py<PyAny> {
        self.mask.clone_ref(py)
    }
    #[getter]
    fn index_map(&self, py: Python<'_>) -> Py<PyAny> {
        self.index_map.clone_ref(py)
    }
    /// The masked 5-point Laplacian as a `scipy.sparse.csr_matrix`.
    #[getter]
    fn L(&self, py: Python<'_>) -> Py<PyAny> {
        self.l.clone_ref(py)
    }

    // -- state -----------------------------------------------------------------------------

    /// Current live-node displacement `u^n` — the live array, writable in place.
    #[getter]
    fn u(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        self.u.clone_ref(py)
    }
    #[setter]
    fn set_u(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.u = self.adopt_state(value, "u")?;
        Ok(())
    }

    /// Previous live-node displacement `u^{n-1}` — after a step this *is* the object `.u` was.
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

    /// Current displacement field `u^n` as a full 2-D array (dead nodes are 0).
    #[getter]
    fn state(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let bound = self.u.bind(py);
        let ro = bound.readonly();
        let field = ops2d::embed(state_slice(&ro, "u")?, &self.p.index_map);
        let (nrows, ncols) = self.p.shape();
        to_2d_f64(py, field, nrows, ncols)
    }

    // -- initial conditions ----------------------------------------------------------------

    /// Select the live-node values from a full 2-D `field`.
    fn to_live(&self, py: Python<'_>, field: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let (shape, values) = as_f64_field(py, field, "field")?;
        let (nrows, ncols) = self.p.shape();
        if shape != [nrows, ncols] {
            return Err(PyValueError::new_err(format!(
                "field must have shape {}, got {}.",
                self.mask_shape_repr(),
                shape_repr(&shape)
            )));
        }
        Ok(PyArray1::from_vec(py, self.p.to_live(&values))
            .into_any()
            .unbind())
    }

    /// Set the initial displacement (and optional velocity).
    ///
    /// `u0` may be a full 2-D field or a flat live-node vector; `v0` may additionally be a scalar.
    /// Uses the consistent second-order start `u^{-1} = u^0 - k v^0 + 1/2 c^2 k^2 L u^0`.
    #[pyo3(signature = (u0, v0=None))]
    fn set_state(
        &mut self,
        py: Python<'_>,
        u0: &Bound<'_, PyAny>,
        v0: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let (ushape, uvals) = as_f64_field(py, u0, "u0")?;
        let u = self.to_live_arg(&ushape, uvals, "u0")?;

        let v = match v0 {
            None => vec![0.0; self.p.n_live()],
            Some(obj) => match obj.extract::<f64>() {
                // `np.isscalar(v0) or np.asarray(v0).shape == ()` — a scalar fills.
                Ok(scalar) => vec![scalar; self.p.n_live()],
                Err(_) => {
                    let (vshape, vvals) = as_f64_field(py, obj, "v0")?;
                    self.to_live_arg(&vshape, vvals, "v0")?
                }
            },
        };

        let prev = core::initial_previous(&u, &v, &self.p);
        self.u_prev = PyArray1::from_vec(py, prev).unbind();
        self.swap_u(py, u);
        self.n = 0;
        Ok(())
    }

    // -- time stepping ---------------------------------------------------------------------

    /// Advance one timestep (rolls the history).
    fn step(&mut self, py: Python<'_>) -> PyResult<()> {
        let mut next = vec![0.0; self.p.n_live()];
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

    /// Discrete energy `E^n` (Joules) using the cross-time potential term.
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

    /// Displacement at flat live-node `index` — a pickup for spectral analysis.
    ///
    /// Negative indices count from the end, as they do on the NumPy array this replaces.
    fn displacement_at(&self, py: Python<'_>, index: i64) -> PyResult<f64> {
        let n_live = self.p.n_live() as i64;
        let idx = if index < 0 { index + n_live } else { index };
        if idx < 0 || idx >= n_live {
            return Err(PyIndexError::new_err(format!(
                "index {index} is out of bounds for {n_live} live nodes"
            )));
        }
        let bound = self.u.bind(py);
        let ro = bound.readonly();
        Ok(state_slice(&ro, "u")?[idx as usize])
    }

    /// Flat live-node index nearest the physical point `(x, y)`.
    fn pickup_index_at(&self, x: f64, y: f64) -> usize {
        self.p.pickup_index_at(x, y)
    }
}
