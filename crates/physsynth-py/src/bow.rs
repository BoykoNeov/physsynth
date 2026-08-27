//! The binding over `physsynth_core::bow` — the bowed string wearing the Python interface.
//!
//! # The shape, and how little of it is new
//!
//! This is the third model in the project that holds another Rust pyclass and corrects it once per
//! step — after `PyReedBore` and `PyMalletMembrane` — and it follows both:
//!
//! * **`.string` is the object the caller passed, not a copy.** `tests/test_bow_energy.py` calls
//!   `bow.string.set_state(...)` and `bow.string.energy()`, `tests/test_bow_modal.py` reads
//!   `bow.string.fs` and `.N`, and `web/serialize.py` reads `.lam` and `.x`. So this type holds a
//!   `Py<PyDampedStiffString>` and drives the free functions in the core module.
//! * **It requires a Rust `DampedStiffString`.** Handed the pure-Python `DampedStiffStringPy` it
//!   raises `TypeError` rather than falling back — a Rust bow reporting Rust while bowing a Python
//!   string is the green-and-meaningless run the swap guard exists to prevent. Under
//!   `PHYSSYNTH_RS=1` both swaps fire together, so `tests/helpers.py` and `web/serialize.py` hand
//!   it a Rust string without knowing they did.
//! * **The borrow is one phase**, like the mallet's and unlike the reed's (§13.2). The bow lets the
//!   string advance force-free and *then* applies its rank-1 correction, so `step()` takes one
//!   `borrow_mut()` and never re-enters the interpreter.
//!
//! # `u += ...` is written in place here, and that is the same thing Python does
//!
//! The original writes `self.string.u += self._force_pref * f_B * self._a_full`. That is an
//! in-place `__iadd__` on the array object followed by an assignment of *the same object* back
//! through the property — so a caller holding `.u` from before the step sees the correction. This
//! binding writes through the live buffer directly, which has exactly that property; rebinding to a
//! fresh array would not.
//!
//! # Every underscore-named attribute is exposed
//!
//! `tests/test_bow_stability.py` reads `bow._g` to re-derive `helmholtz_number`. §12.2's rule
//! applies — a leading underscore is not a statement about the interface — so `_g`, `_a_vec`,
//! `_a_full` and `_force_pref` are all getters rather than a judgement call about which of them
//! somebody might mean.

use numpy::PyArray1;
use physsynth_core::bow as core;
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;

use crate::string_damped::PyDampedStiffString;

/// `friction_smooth` — the smooth single-hump friction characteristic, as a module-level function.
#[pyfunction]
#[pyo3(name = "friction_smooth")]
pub fn py_friction_smooth(v_rel: f64, force: f64, sharpness: f64) -> f64 {
    core::friction_smooth(v_rel, force, sharpness)
}

/// `friction_smooth_deriv` — its derivative.
#[pyfunction]
#[pyo3(name = "friction_smooth_deriv")]
pub fn py_friction_smooth_deriv(v_rel: f64, force: f64, sharpness: f64) -> f64 {
    core::friction_smooth_deriv(v_rel, force, sharpness)
}

/// A damped stiff string driven by a bow — the Rust implementation, wearing the Python interface.
///
/// Attribute-for-attribute and method-for-method compatible with `physsynth.core.bow.BowedString`;
/// the docstring on that class is the reference.
#[pyclass(name = "BowedString", module = "physsynth_rs")]
pub struct PyBowedString {
    p: core::Params,
    s: core::State,
    string: Py<PyDampedStiffString>,
    a_full: Vec<f64>,
    a_vec: Vec<f64>,
    x_bow: f64,
    beta: f64,
    l: f64,
}

#[pymethods]
impl PyBowedString {
    #[new]
    #[pyo3(signature = (
        *, string, bow_position, v_bow, force, sharpness=100.0, newton_tol=1e-13,
        newton_maxiter=60
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        string: &Bound<'_, PyAny>,
        bow_position: f64,
        v_bow: f64,
        force: f64,
        sharpness: f64,
        newton_tol: f64,
        newton_maxiter: i64,
    ) -> PyResult<Self> {
        let handle: Py<PyDampedStiffString> = string
            .clone()
            .cast_into::<PyDampedStiffString>()
            .map_err(|_| {
                PyTypeError::new_err(
                    "the Rust BowedString needs a Rust DampedStiffString \
                     (physsynth_rs.DampedStiffString). Got something else -- most likely the \
                     pure-Python `string_damped.DampedStiffStringPy`, which this class cannot bow \
                     without crossing back into the interpreter every timestep. Build the string \
                     from the same implementation as the bow.",
                )
            })?
            .unbind();

        let (p, a_full, a_vec, x_bow, beta, l) = {
            let sref = handle.bind(py).borrow();
            let sp = sref.params();
            let p = core::Params::new(
                bow_position,
                v_bow,
                force,
                sharpness,
                newton_tol,
                newton_maxiter,
                sp.l,
                sp.h,
                sp.n,
            )
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
            let (a_full, a_i) = core::admittance(sp, p.node);
            let a_vec = a_full[1..sp.nodes() - 1].to_vec();
            let p = p.with_admittance(sp.k, sp.rho, sp.h, a_i);
            let x_bow = sp.grid()[p.node];
            (p, a_full, a_vec, x_bow, x_bow / sp.l, sp.l)
        };

        Ok(PyBowedString {
            p,
            s: core::State::default(),
            string: handle,
            a_full,
            a_vec,
            x_bow,
            beta,
            l,
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
        self.p.k
    }
    #[getter]
    fn v_bow(&self) -> f64 {
        self.p.v_bow
    }
    #[getter]
    fn force(&self) -> f64 {
        self.p.force
    }
    #[getter]
    fn sharpness(&self) -> f64 {
        self.p.sharpness
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
    fn node(&self) -> usize {
        self.p.node
    }
    #[getter]
    fn x_bow(&self) -> f64 {
        self.x_bow
    }
    #[allow(non_snake_case)]
    #[getter]
    fn L(&self) -> f64 {
        self.l
    }
    #[getter]
    fn beta(&self) -> f64 {
        self.beta
    }
    #[getter]
    fn helmholtz_number(&self) -> f64 {
        self.p.helmholtz_number
    }

    /// `g = k a_i / (2 rho h)` — read by `tests/test_bow_stability.py`, see the module header.
    #[getter]
    fn _g(&self) -> f64 {
        self.p.g
    }
    #[getter]
    fn _force_pref(&self) -> f64 {
        self.p.force_pref
    }
    /// `a = A^{-1} e_i` on the interior, length `N - 1`.
    #[getter]
    fn _a_vec(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        PyArray1::from_slice(py, &self.a_vec).unbind()
    }
    /// The same, embedded on the full grid.
    #[getter]
    fn _a_full(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        PyArray1::from_slice(py, &self.a_full).unbind()
    }

    // -- per-step observables ----------------------------------------------------------------

    #[getter]
    fn v_rel(&self) -> f64 {
        self.s.v_rel
    }
    #[getter]
    fn bow_force(&self) -> f64 {
        self.s.bow_force
    }
    #[getter]
    fn bow_power(&self) -> f64 {
        self.s.bow_power
    }
    #[getter]
    fn bow_work(&self) -> f64 {
        self.s.bow_work
    }
    #[getter]
    fn fallbacks(&self) -> usize {
        self.s.fallbacks
    }
    #[getter]
    fn n(&self) -> usize {
        self.s.n
    }

    // -- time stepping -----------------------------------------------------------------------

    /// Advance one step: force-free string advance, scalar friction solve, rank-1 correction.
    fn step(&mut self, py: Python<'_>) -> PyResult<()> {
        self.step_reporting(py).map(|_| ())
    }

    /// `step()`, returning `(newton_residual_evaluations, used_fallback)`.
    ///
    /// **Not part of the Python model's interface** and deliberately so: §19.11 asked for the
    /// Newton iteration count and the fallback branch to be *compared* rather than assumed, and a
    /// count nothing can read cannot be compared. `tests/test_rust_parity_bow.py` is the only
    /// caller; the Python side reaches the same number by patching `_residual`, counting calls and
    /// muting the bracket -- which is why the count is of evaluations rather than of accepted
    /// steps (see `physsynth_core::bow::FrictionSolution`).
    fn step_reporting(&mut self, py: Python<'_>) -> PyResult<(usize, bool)> {
        // Take the handle first so nothing borrows `self.string` while the mutable borrow is live.
        let handle = self.string.clone_ref(py);
        let mut sref = handle.bind(py).borrow_mut();
        let i = self.p.node;

        // `u^{n-1}_i` must be read BEFORE the step: the string's `step()` rebinds `u_prev` to what
        // `u` was, so this quantity stops existing one line later.
        let u_prev_i = sref.u_prev_at(py, i)?;
        sref.step(py)?;
        let u_i = sref.u_at(py, i)?;
        let vf = core::v_free(u_i, u_prev_i, self.p.k, self.p.v_bow);

        let a_full = &self.a_full;
        let s = &mut self.s;
        let p = &self.p;
        let outcome = sref.with_u_mut(py, |u| core::apply(vf, u, a_full, s, p))?;
        let sol = outcome.map_err(|e| solve_err(e, self.s.n))?;
        Ok((sol.newton_evals, sol.used_fallback))
    }

    // -- diagnostics -------------------------------------------------------------------------

    /// Discrete string energy `E^n` (J). The bow stores none — assert the *balance*.
    fn energy(&self, py: Python<'_>) -> PyResult<f64> {
        self.string.bind(py).borrow().energy(py)
    }

    /// The string displacement field (a copy, for animation snapshots).
    #[getter]
    fn state(&self, py: Python<'_>) -> PyResult<Py<PyArray1<f64>>> {
        self.string.bind(py).borrow().state(py)
    }

    /// String pickup at grid node `index`.
    fn displacement_at(&self, py: Python<'_>, index: i64) -> PyResult<f64> {
        self.string.bind(py).borrow().displacement_at(py, index)
    }

    /// String transverse velocity at the bow node for the last step (m/s).
    fn bow_velocity(&self) -> f64 {
        self.s.v_rel + self.p.v_bow
    }
}

/// The friction solve's loud backstop, mapped to the original's `RuntimeError`.
///
/// The message is built here rather than in the core because it quotes the step count, which lives
/// on the state rather than on the error.
fn solve_err(e: core::BowError, n: usize) -> PyErr {
    match e {
        core::BowError::NoRoot => PyRuntimeError::new_err(format!(
            "bow friction residual has no root in the bracket at step {n} \
             (should be impossible for the bounded smooth friction curve)."
        )),
        other => PyRuntimeError::new_err(other.to_string()),
    }
}
