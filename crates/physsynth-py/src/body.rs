//! The binding over `physsynth_core::body` — the modal body wearing the Python interface.
//!
//! # Three state buffers, not two, and the third one has an underscore in its name
//!
//! §9.3's rule is that anything a client can write must live in a Python-owned NumPy array. The
//! string and the membrane had two such buffers each (`u`, `u_prev`). This model has **three**,
//! and the extra one is `_accel` — spelled private in the original and treated as public by three
//! separate modules:
//!
//! ```text
//! radiation.RadiatedBody.step     b.q = b.q - (R*u) * corr ;  b._accel = (...)/k^2
//! radiation.<rational air load>   b.q = b.q - p * corr     ;  b._accel = (...)/k^2
//! airbox.RoomLoadedBody.step      b.q = b.q - pbar * corr  ;  b._accel = (...)/k^2
//! ```
//!
//! All three run that correction **once per timestep**, and all three *rebind* `q` (they assign a
//! whole new array, they do not write into the old one). So `q`, `q_prev` and `_accel` are all
//! gettable and settable here, and a setter that quietly copied instead of adopting would leave
//! the wrapper's correction on the floor while every energy bar stayed green.
//!
//! That is the finding this model contributes to the migration: **a leading underscore is not a
//! statement about the interface**, it is a statement about intent, and the two came apart here
//! long before Rust was involved. Phase 0 found `connection.py` *reading* the string's private
//! names; this is the same discovery one step worse.
//!
//! # No SciPy, no geometry, no immutable half
//!
//! Unlike the membrane there is nothing here to build once and hand back by reference — the eight
//! parameter vectors are small, immutable and only read at construction by the wrappers
//! (`connection.py` computes `beta_b = k^2 sum(phi^2/m)` from `phi` and `m` exactly once). They
//! are handed back as fresh arrays, which is what the original's plain attributes do not do — see
//! `param_array`.

use crate::state_slice;
use numpy::{PyArray1, PyArrayMethods, PyUntypedArrayMethods};
use physsynth_core::body as core;
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;

/// A modal body — the Rust implementation, wearing the Python interface.
///
/// Attribute-for-attribute and method-for-method compatible with
/// `physsynth.core.body.ModalBody`; the docstring on that class is the reference.
#[pyclass(name = "ModalBody", module = "physsynth_rs")]
pub struct PyModalBody {
    p: core::Params,
    q: Py<PyArray1<f64>>,
    q_prev: Py<PyArray1<f64>>,
    accel: Py<PyArray1<f64>>,
    n: usize,
}

/// `np.broadcast_to(np.asarray(obj, dtype=float), (m,))`, with NumPy's refusals reproduced.
///
/// A scalar (or a length-1 array) fills; a length-`m` array is taken as is; anything else is the
/// error NumPy would have raised, spelled the way NumPy spells it — the suite does not match on
/// this text today, but a migration that quietly accepted a length-2 `sigmas` for a 3-mode bank
/// would be building a *different body*, and that is exactly the class of divergence no energy
/// test can see.
fn broadcast_to(obj: &Bound<'_, PyAny>, m: usize, name: &str) -> PyResult<Vec<f64>> {
    if let Ok(scalar) = obj.extract::<f64>() {
        return Ok(vec![scalar; m]);
    }
    let values = as_flat_f64(obj, name)?;
    match values.len() {
        1 => Ok(vec![values[0]; m]),
        len if len == m => Ok(values),
        len => Err(PyValueError::new_err(format!(
            "operands could not be broadcast together with remapped shapes \
             [original->remapped]: ({len},)  and requested shape ({m},)"
        ))),
    }
}

/// `np.atleast_1d(np.asarray(obj, dtype=float))`, refusing anything that is not 1-D.
///
/// Goes through NumPy rather than a direct downcast so that a plain Python list is acceptable —
/// `tests/test_airbox_port.py` passes `freqs=[220.0]`, and `web/serialize.py` passes lists too.
fn as_flat_f64(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<Vec<f64>> {
    let py = obj.py();
    let np = py.import("numpy")?;
    let arr = np.call_method1("asarray", (obj, np.getattr("float64")?))?;
    let arr = np.call_method1("atleast_1d", (arr,))?;
    let arr = np.call_method1("ascontiguousarray", (arr,))?;
    let arr: Bound<'_, PyArray1<f64>> = arr.cast_into().map_err(|_| {
        PyValueError::new_err(format!(
            "{name} must be a 1-D array with at least one mode."
        ))
    })?;
    let ro = arr.readonly();
    Ok(state_slice(&ro, name)?.to_vec())
}

impl PyModalBody {
    /// Validate an array being assigned to one of the three state buffers and take ownership.
    ///
    /// As on the string and the membrane: Python would accept any object; this accepts any
    /// contiguous 1-D float64 array of the right length and rejects the rest loudly. **Adopting**
    /// rather than copying is the point — `RoomLoadedBody` assigns a freshly computed array and
    /// then reads `.q` back on the next step.
    fn adopt_state(&self, value: &Bound<'_, PyAny>, name: &str) -> PyResult<Py<PyArray1<f64>>> {
        let arr: Bound<'_, PyArray1<f64>> = value.clone().cast_into().map_err(|_| {
            PyValueError::new_err(format!("{name} must be a 1-D float64 numpy array."))
        })?;
        let ro = arr.readonly();
        if ro.len() != self.p.n_modes() {
            return Err(PyValueError::new_err(format!(
                "{name} must have shape ({},), got ({},).",
                self.p.n_modes(),
                ro.len()
            )));
        }
        state_slice(&ro, name)?;
        Ok(arr.unbind())
    }

    /// A parameter vector, as a fresh NumPy array.
    fn param_array(&self, py: Python<'_>, values: &[f64]) -> Py<PyArray1<f64>> {
        PyArray1::from_slice(py, values).unbind()
    }
}

#[pymethods]
impl PyModalBody {
    #[new]
    #[pyo3(signature = (*, freqs, fs, sigmas=None, masses=None, phi=None, radiation=None))]
    fn new(
        py: Python<'_>,
        freqs: &Bound<'_, PyAny>,
        fs: f64,
        sigmas: Option<&Bound<'_, PyAny>>,
        masses: Option<&Bound<'_, PyAny>>,
        phi: Option<&Bound<'_, PyAny>>,
        radiation: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let f = as_flat_f64(freqs, "freqs")?;
        // The original checks emptiness before it broadcasts anything, and a zero-mode bank would
        // make every broadcast below vacuously succeed.
        if f.is_empty() {
            return Err(PyValueError::new_err(
                core::ParamError::EmptyFreqs.to_string(),
            ));
        }
        let m = f.len();

        let sigma = match sigmas {
            None => vec![0.0; m],
            Some(o) => broadcast_to(o, m, "sigmas")?,
        };
        let mass = match masses {
            None => vec![1.0; m],
            Some(o) => broadcast_to(o, m, "masses")?,
        };
        let ph = match phi {
            None => vec![1.0; m],
            Some(o) => broadcast_to(o, m, "phi")?,
        };
        let rad = match radiation {
            None => None,
            Some(o) if o.is_none() => None,
            Some(o) => Some(broadcast_to(o, m, "radiation")?),
        };

        let p = core::Params::new(f, fs, sigma, mass, ph, rad)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        Ok(PyModalBody {
            q: PyArray1::from_vec(py, vec![0.0; m]).unbind(),
            q_prev: PyArray1::from_vec(py, vec![0.0; m]).unbind(),
            accel: PyArray1::from_vec(py, vec![0.0; m]).unbind(),
            p,
            n: 0,
        })
    }

    // -- parameters --------------------------------------------------------------------------

    #[getter]
    fn freqs(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        self.param_array(py, &self.p.freqs)
    }

    #[getter]
    fn fs(&self) -> f64 {
        self.p.fs
    }

    #[getter]
    fn k(&self) -> f64 {
        self.p.k
    }

    #[getter]
    fn sigma(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        self.param_array(py, &self.p.sigma)
    }

    #[getter]
    fn m(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        self.param_array(py, &self.p.m)
    }

    #[getter]
    fn phi(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        self.param_array(py, &self.p.phi)
    }

    #[getter]
    fn a(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        self.param_array(py, &self.p.a)
    }

    #[getter]
    fn omega(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        self.param_array(py, &self.p.omega)
    }

    #[getter]
    fn omega_k(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        self.param_array(py, &self.p.omega_k)
    }

    #[getter]
    #[allow(non_snake_case)]
    fn M(&self) -> usize {
        self.p.n_modes()
    }

    // -- state -------------------------------------------------------------------------------

    /// Current modal displacement `q^n` — the live array, writable in place *and* rebindable.
    #[getter]
    fn q(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        self.q.clone_ref(py)
    }
    #[setter]
    fn set_q(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.q = self.adopt_state(value, "q")?;
        Ok(())
    }

    /// Previous modal displacement `q^{n-1}` — after a step this *is* the object `.q` was.
    #[getter]
    fn q_prev(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        self.q_prev.clone_ref(py)
    }
    #[setter]
    fn set_q_prev(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.q_prev = self.adopt_state(value, "q_prev")?;
        Ok(())
    }

    /// Modal acceleration `q''` of the most recent step.
    ///
    /// Named with a leading underscore because the original is, and settable because three
    /// modules assign to it once per timestep. See the module header.
    #[getter]
    fn _accel(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        self.accel.clone_ref(py)
    }
    #[setter]
    fn set__accel(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.accel = self.adopt_state(value, "_accel")?;
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

    /// Current modal-displacement vector `q^n` (a copy).
    #[getter]
    fn state(&self, py: Python<'_>) -> PyResult<Py<PyArray1<f64>>> {
        let bound = self.q.bind(py);
        let ro = bound.readonly();
        Ok(PyArray1::from_slice(py, state_slice(&ro, "q")?).unbind())
    }

    // -- initial conditions ------------------------------------------------------------------

    /// Set the initial modal displacement (and optional modal velocity).
    ///
    /// Uses the consistent second-order start `q^{-1} = q^0 - k v^0 - 1/2 k^2 omega^2 q^0`, and
    /// seeds `_accel` with the lossless free response `-omega^2 q^0` — not zero, because
    /// `pressure()` is readable before the first step.
    ///
    /// The original spells the default `v0=0.0`; PyO3 cannot express a float default for
    /// an object parameter, so it is `None` here and means the same zero. The difference is
    /// visible only to `inspect.signature`, which nothing in this repo reads.
    #[pyo3(signature = (q0, v0=None))]
    fn set_state(
        &mut self,
        py: Python<'_>,
        q0: &Bound<'_, PyAny>,
        v0: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let m = self.p.n_modes();
        let q = broadcast_to(q0, m, "q0")?;
        let v = match v0 {
            None => vec![0.0; m],
            Some(o) => broadcast_to(o, m, "v0")?,
        };
        let (prev, accel) = core::initial_state(&q, &v, &self.p);
        self.q = PyArray1::from_vec(py, q).unbind();
        self.q_prev = PyArray1::from_vec(py, prev).unbind();
        self.accel = PyArray1::from_vec(py, accel).unbind();
        self.n = 0;
        Ok(())
    }

    // -- time stepping -----------------------------------------------------------------------

    /// Advance one timestep under an optional scalar bridge `force` (default 0).
    #[pyo3(signature = (force=0.0))]
    fn step(&mut self, py: Python<'_>, force: f64) -> PyResult<()> {
        let m = self.p.n_modes();
        let mut next = vec![0.0; m];
        let mut accel = vec![0.0; m];
        {
            let q_bound = self.q.bind(py);
            let qp_bound = self.q_prev.bind(py);
            let q_ro = q_bound.readonly();
            let qp_ro = qp_bound.readonly();
            core::step_into(
                state_slice(&q_ro, "q")?,
                state_slice(&qp_ro, "q_prev")?,
                force,
                &mut next,
                &mut accel,
                &self.p,
            );
        }
        // `self.q_prev = self.q` hands the *same object* over, exactly as the original does, so a
        // caller holding a reference to `.q` across a step finds it under `.q_prev` afterwards.
        let fresh = PyArray1::from_vec(py, next).unbind();
        self.q_prev = std::mem::replace(&mut self.q, fresh);
        self.accel = PyArray1::from_vec(py, accel).unbind();
        self.n += 1;
        Ok(())
    }

    // -- diagnostics -------------------------------------------------------------------------

    /// Discrete modal energy `E^n` (Joules), cross-time potential.
    fn energy(&self, py: Python<'_>) -> PyResult<f64> {
        let q_bound = self.q.bind(py);
        let qp_bound = self.q_prev.bind(py);
        let q_ro = q_bound.readonly();
        let qp_ro = qp_bound.readonly();
        Ok(core::energy(
            state_slice(&q_ro, "q")?,
            state_slice(&qp_ro, "q_prev")?,
            &self.p,
        ))
    }

    /// Physical driving-point (bridge) displacement `w_b = sum_i phi_i q_i^n`.
    fn bridge_displacement(&self, py: Python<'_>) -> PyResult<f64> {
        let bound = self.q.bind(py);
        let ro = bound.readonly();
        Ok(core::bridge_displacement(state_slice(&ro, "q")?, &self.p))
    }

    /// Driving-point velocity `sum_i phi_i (delta_t- q_i)`.
    fn bridge_velocity(&self, py: Python<'_>) -> PyResult<f64> {
        let q_bound = self.q.bind(py);
        let qp_bound = self.q_prev.bind(py);
        let q_ro = q_bound.readonly();
        let qp_ro = qp_bound.readonly();
        Ok(core::bridge_velocity(
            state_slice(&q_ro, "q")?,
            state_slice(&qp_ro, "q_prev")?,
            &self.p,
        ))
    }

    /// Radiated pressure read-out `p = sum_i a_i q_i''`.
    ///
    /// Reads `_accel` — including whatever a body-loading wrapper last wrote there.
    fn pressure(&self, py: Python<'_>) -> PyResult<f64> {
        let bound = self.accel.bind(py);
        let ro = bound.readonly();
        Ok(core::pressure(state_slice(&ro, "_accel")?, &self.p))
    }

    /// Modal coordinate `q_index` — lets `engine.simulate` tap a single mode.
    ///
    /// Negative indices count from the end, as they do on the NumPy array this replaces.
    fn displacement_at(&self, py: Python<'_>, index: i64) -> PyResult<f64> {
        let m = self.p.n_modes() as i64;
        let idx = if index < 0 { index + m } else { index };
        if idx < 0 || idx >= m {
            return Err(PyIndexError::new_err(format!(
                "index {index} is out of bounds for {m} modes"
            )));
        }
        let bound = self.q.bind(py);
        let ro = bound.readonly();
        Ok(state_slice(&ro, "q")?[idx as usize])
    }
}
