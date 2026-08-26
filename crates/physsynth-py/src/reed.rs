//! The binding over `physsynth_core::reed` — the clarinet wearing the Python interface.
//!
//! # This is the seam the whole batch was held for, and here it stops being a seam
//!
//! `bore::PyBore::step` accepts an arbitrary Python callable and hands it a live view of the
//! in-progress pressure field, because `tests/test_reed_stability.py` passes its own
//! `lambda p: None` and the capability is therefore not the reed's private channel (plan §12.8).
//! That path costs a Python call per timestep.
//!
//! `PyReedBore` does not take it. It **requires a `PyBore`** — the Rust air column — extracted
//! natively, and injects through `PyBore::step_native`, a Rust closure. So the clarinet's hot loop
//! crosses the language boundary exactly once per `step()` call rather than twice, and the reed's
//! scalar solve, its Bernoulli jet and its Brent fallback all run without touching the
//! interpreter. Under `PHYSSYNTH_RS=1` both swaps fire together, so `tests/helpers.py` and
//! `web/serialize.py` hand it a Rust bore without knowing they did.
//!
//! Handed a *Python* bore it raises `TypeError` rather than silently falling back, because a
//! silent fallback would be a Rust reed reporting Rust while driving a Python air column — the
//! same class of green-and-meaningless run the plan's swap guard exists to prevent.
//!
//! # `.bore` is the object the caller passed, not a copy
//!
//! `tests/test_reed_energy.py` and `web/serialize.py` both reach through `reed.bore` and call
//! `energy()`, `set_state()` and `pressure()` on it. So this type holds a `Py<PyBore>` handle
//! rather than owning a `physsynth_core::reed::ReedBore`, and the free functions in the core
//! module — `inject`, `commit` — are what it actually calls. Same reasoning as every other model
//! here: the native owning struct is for `cargo test`, the binding owns the Python object graph.
//!
//! # `p_mouth` is state, not a parameter
//!
//! The original documents mutating it between steps as how an attack is played, and nothing
//! derived from it is cached, so it is settable and lives in the core's `State`.

use numpy::PyArray1;
use physsynth_core::reed as core;
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;

use crate::bore::PyBore;

/// Quasi-static Bernoulli volume flow through the reed channel.
///
/// A module-level function in the original (`reed.bernoulli_flow`), and exported as one here.
#[pyfunction]
#[pyo3(name = "bernoulli_flow")]
pub fn py_bernoulli_flow(dp: f64, opening: f64, width: f64, rho: f64) -> f64 {
    core::bernoulli_flow(dp, opening, width, rho)
}

/// A bore blown through a dynamic single reed — the Rust implementation, wearing the Python
/// interface.
///
/// Attribute-for-attribute and method-for-method compatible with
/// `physsynth.core.reed.ReedBore`; the docstring on that class is the reference.
#[pyclass(name = "ReedBore", module = "physsynth_rs")]
pub struct PyReedBore {
    p: core::Params,
    s: core::State,
    bore: Py<PyBore>,
}

#[pymethods]
impl PyReedBore {
    #[new]
    #[pyo3(signature = (
        *, bore, p_mouth, f_reed=2500.0, q_reed=4.0, mu=0.03, Sr=1.5e-4, width=1.5e-2,
        H0=4.0e-4, newton_tol=1e-10, newton_maxiter=60
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        bore: &Bound<'_, PyAny>,
        p_mouth: f64,
        f_reed: f64,
        q_reed: f64,
        mu: f64,
        Sr: f64,
        width: f64,
        H0: f64,
        newton_tol: f64,
        newton_maxiter: i64,
    ) -> PyResult<Self> {
        // The original's first two checks touch nothing on the bore, so they run first here too —
        // a call that is wrong in both ways must report the same fault as Python's.
        if f_reed <= 0.0 || q_reed <= 0.0 || mu <= 0.0 || Sr <= 0.0 || width <= 0.0 || H0 <= 0.0 {
            return Err(PyValueError::new_err(
                core::ParamError::NonPositiveScalar.to_string(),
            ));
        }
        if newton_maxiter < 1 {
            return Err(PyValueError::new_err(
                core::ParamError::BadMaxIter.to_string(),
            ));
        }

        let handle: Py<PyBore> = bore
            .clone()
            .cast_into::<PyBore>()
            .map_err(|_| {
                PyTypeError::new_err(
                "the Rust ReedBore needs a Rust Bore (physsynth_rs.Bore). Got something else -- \
                 most likely the pure-Python `bore.BorePy`, which this class cannot drive without \
                 crossing back into the interpreter every timestep. Build the bore from the same \
                 implementation as the reed.",
            )
            })?
            .unbind();

        let params = {
            let bore_ref = handle.bind(py).borrow();
            core::Params::new(
                &bore_ref.params,
                f_reed,
                q_reed,
                mu,
                Sr,
                width,
                H0,
                newton_tol,
                newton_maxiter,
            )
            .map_err(|e| PyValueError::new_err(e.to_string()))?
        };

        Ok(PyReedBore {
            s: core::State::at_rest(p_mouth),
            p: params,
            bore: handle,
        })
    }

    // -- parameters --------------------------------------------------------------------------

    /// The air column — the very object the caller passed in.
    #[getter]
    fn bore(&self, py: Python<'_>) -> Py<PyBore> {
        self.bore.clone_ref(py)
    }
    #[getter]
    fn k(&self) -> f64 {
        self.p.k
    }
    #[getter]
    fn f_reed(&self) -> f64 {
        self.p.f_reed
    }
    #[getter]
    fn q_reed(&self) -> f64 {
        self.p.q_reed
    }
    #[getter]
    fn mu(&self) -> f64 {
        self.p.mu
    }
    #[getter]
    fn Sr(&self) -> f64 {
        self.p.sr
    }
    #[getter]
    fn width(&self) -> f64 {
        self.p.width
    }
    #[getter]
    fn H0(&self) -> f64 {
        self.p.h0
    }
    #[getter]
    fn rho(&self) -> f64 {
        self.p.rho
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
    fn wr(&self) -> f64 {
        self.p.wr
    }
    #[getter]
    fn g(&self) -> f64 {
        self.p.g
    }
    #[getter]
    fn Mr(&self) -> f64 {
        self.p.mr
    }
    #[getter]
    fn p_closing(&self) -> f64 {
        self.p.p_closing
    }

    // -- state -------------------------------------------------------------------------------

    /// Steady mouth pressure (Pa). Settable — mutating it between steps is how an attack is played.
    #[getter]
    fn p_mouth(&self) -> f64 {
        self.s.p_mouth
    }
    #[setter]
    fn set_p_mouth(&mut self, value: f64) {
        self.s.p_mouth = value;
    }

    #[getter]
    fn y(&self) -> f64 {
        self.s.y
    }
    #[setter]
    fn set_y(&mut self, value: f64) {
        self.s.y = value;
    }
    #[getter]
    fn y_prev(&self) -> f64 {
        self.s.y_prev
    }
    #[setter]
    fn set_y_prev(&mut self, value: f64) {
        self.s.y_prev = value;
    }

    #[getter]
    fn dp(&self) -> f64 {
        self.s.dp
    }
    #[getter]
    fn reed_velocity(&self) -> f64 {
        self.s.reed_velocity
    }
    #[getter]
    fn flow(&self) -> f64 {
        self.s.flow
    }
    #[getter]
    fn jet_flow(&self) -> f64 {
        self.s.jet_flow
    }
    #[getter]
    fn mouth_work(&self) -> f64 {
        self.s.mouth_work
    }
    #[getter]
    fn jet_loss(&self) -> f64 {
        self.s.jet_loss
    }
    #[getter]
    fn reed_damp_work(&self) -> f64 {
        self.s.reed_damp_work
    }
    #[getter]
    fn fallbacks(&self) -> usize {
        self.s.fallbacks
    }
    #[getter]
    fn n(&self) -> usize {
        self.s.n
    }

    /// The bore pressure field (the vibrating air column, for animation snapshots).
    #[getter]
    fn state(&self, py: Python<'_>) -> PyResult<Py<PyArray1<f64>>> {
        let bore = self.bore.bind(py).borrow();
        Ok(PyArray1::from_vec(py, bore.pressure_vec(py)?).unbind())
    }

    // -- time stepping -----------------------------------------------------------------------

    /// Advance one step: the bore's leapfrog with the reed injecting at node 0 (implicit scalar
    /// solve inside the hook), then commit the reed state and book the energy channels.
    fn step(&mut self, py: Python<'_>) -> PyResult<()> {
        // Take the handle first so nothing borrows `self.bore` while the two disjoint field
        // borrows below are live.
        let handle = self.bore.clone_ref(py);
        let mut bore = handle.bind(py).borrow_mut();
        // `p0^n`, read before the bore has committed anything — the original reads it inside the
        // hook, where it is the same value for the same reason.
        let p_old = bore.pressure_node(py, 0)?;

        let params = &self.p;
        let state = &mut self.s;
        bore.step_native(py, |p_next: &mut [f64]| {
            core::inject(p_next, p_old, params, state)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })?;
        drop(bore);

        core::commit(&self.p, &mut self.s);
        Ok(())
    }

    // -- diagnostics -------------------------------------------------------------------------

    /// Current clamped channel opening `H^+ = max(H0 + y, 0)` (m).
    fn reed_opening(&self) -> f64 {
        self.s.reed_opening(&self.p)
    }

    /// Stored reed mechanical energy (J), with the cross-time potential.
    fn reed_energy(&self) -> f64 {
        self.s.reed_energy(&self.p)
    }

    /// Total stored energy `E_bore + E_reed` (J). **Not** conserved — the mouth is active.
    fn energy(&self, py: Python<'_>) -> PyResult<f64> {
        let bore = self.bore.bind(py).borrow();
        Ok(bore.energy(py)? + self.s.reed_energy(&self.p))
    }

    /// Bore pressure at node `index` — an interior microphone for spectral analysis.
    fn displacement_at(&self, py: Python<'_>, index: i64) -> PyResult<f64> {
        let bore = self.bore.bind(py).borrow();
        bore.displacement_at(py, index)
    }

    /// Pressure at the mouthpiece node `p0` (Pa) — the natural playing signal.
    fn mouthpiece_pressure(&self, py: Python<'_>) -> PyResult<f64> {
        let bore = self.bore.bind(py).borrow();
        bore.pressure_node(py, 0)
    }

    /// Far-field read-out of the bore's radiating bell; `0` with no radiating end.
    fn pressure(&self, py: Python<'_>) -> PyResult<f64> {
        let bore = self.bore.bind(py).borrow();
        Ok(bore.pressure())
    }

    /// Dimensionless blowing pressure `gamma = p_mouth / p_closing` — the clarinet control.
    #[getter]
    fn gamma(&self) -> f64 {
        self.s.gamma(&self.p)
    }
}
