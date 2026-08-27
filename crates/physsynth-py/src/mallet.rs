//! The binding over `physsynth_core::mallet` — model #7 wearing the Python interface.
//!
//! # The batch that finishes Phase 2, and the shape it inherits
//!
//! Everything hard about this model was ported before it: the drumhead in Phase 2 batch 1, the
//! contact root-find in Phase 3 batch 2. What is here is the shell. Two consequences for this
//! file specifically:
//!
//! * **`.membrane` is the object the caller passed, not a copy.** `tests/test_mallet_energy.py`
//!   reaches through it for `u`, `X`, `Y`, `mask`, `index_map` and `energy()`, and
//!   `web/serialize.py` calls `pickup_index_at` and `energy()` on it every step of the audio run.
//!   So this type holds a `Py<PyMembrane>` handle and drives the free functions in the core
//!   module, exactly as `PyReedBore` holds its `Py<PyBore>`.
//! * **It requires a Rust `Membrane`.** Handed the pure-Python `MembranePy` it raises
//!   `TypeError` rather than falling back, for the reason the reed gives at greater length: a
//!   silent fallback is a Rust mallet reporting Rust while striking a Python drumhead, which is
//!   the green-and-meaningless run the whole swap guard exists to prevent. Under `PHYSSYNTH_RS=1`
//!   both swaps fire together, so `tests/helpers.py` and `web/serialize.py` hand it a Rust
//!   membrane without knowing they did.
//!
//! # The borrow is one phase here, unlike the reed's
//!
//! §13.2 established that a `&mut self` pymethod cannot hand control back to Python and still be
//! read — the reed pays for that with `step_native` and a Rust closure, because it injects
//! *inside* the bore's leapfrog. The mallet does not: it lets the membrane advance force-free and
//! *then* corrects one node. So `step()` takes a single `borrow_mut()`, reads, steps, reads,
//! solves and writes, and never re-enters the interpreter. The hazard is worth naming even though
//! it does not bite, because the shape that avoids it is a property of this model rather than a
//! precaution anyone took.
//!
//! # The under-resolved-contact warning, and why `stacklevel` changes number
//!
//! The original warns from `__init__` with `stacklevel=2`, meaning "my caller". A Rust `__new__`
//! pushes **no Python frame**, so the caller is already at level 1 — the same frame, reached by a
//! different count. That is the one place this file's spelling deliberately differs from the
//! Python source, and it is the mirror image of §16.8's split: `collision` moved its warning *out*
//! of Rust because a shim frame existed to host it, and this one stays in Rust because no such
//! frame does.

use physsynth_core::mallet as core;
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyAny;
use std::ffi::CString;

use crate::membrane::PyMembrane;

/// `newton_maxiter` as Python would use it: the original passes it straight to a `range()`, so a
/// negative value means "no Newton iterations", not an error.
fn maxiter_of(n: i64) -> usize {
    n.max(0) as usize
}

/// Raise the original's under-resolved-contact `UserWarning`.
///
/// See the module header for the `stacklevel` arithmetic. The message is byte-for-byte the
/// Python one; `{:.1}` and `{:.1f}` agree because both languages round the exact decimal
/// expansion half-to-even.
fn warn_under_resolved(py: Python<'_>, steps_per_contact: f64) -> PyResult<()> {
    let msg = format!(
        "stiff contact under-resolved: ~{steps_per_contact:.1} steps per half-period (want >= 8). \
         Raise fs or lower K/increase M to avoid aliasing the strike."
    );
    let msg = CString::new(msg).map_err(|_| PyValueError::new_err("warning text had a NUL"))?;
    let category = py.get_type::<pyo3::exceptions::PyUserWarning>();
    PyErr::warn(py, category.as_any(), &msg, 1)
}

/// Translate a core refusal into the `ValueError` the Python original raises.
fn param_err(e: core::ParamError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// Translate a contact-solve failure into the `RuntimeError` the Python original raises.
fn solve_err(e: physsynth_core::collision::ContactError) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

/// A membrane struck by a lumped-mass mallet — the Rust implementation, wearing the Python
/// interface.
///
/// Attribute-for-attribute and method-for-method compatible with
/// `physsynth.core.mallet.MalletMembrane`; the docstring on that class is the reference.
#[pyclass(name = "MalletMembrane", module = "physsynth_rs")]
pub struct PyMalletMembrane {
    p: core::Params,
    s: core::State,
    membrane: Py<PyMembrane>,
}

#[pymethods]
impl PyMalletMembrane {
    // Twelve keyword arguments plus the GIL token. This signature IS
    // `MalletMembrane.__init__` — `tests/helpers.py` and `web/serialize.py` both spell it out —
    // so bundling them to please clippy would be a different Python API.
    #[allow(clippy::too_many_arguments)]
    #[new]
    #[pyo3(signature = (
        *, membrane, mass, stiffness, alpha=2.3, hysteresis=0.0, strike_x, strike_y,
        strike_velocity, gap=0.0, eta_tol=1e-12, newton_tol=1e-14, newton_maxiter=60
    ))]
    fn new(
        py: Python<'_>,
        membrane: &Bound<'_, PyAny>,
        mass: f64,
        stiffness: f64,
        alpha: f64,
        hysteresis: f64,
        strike_x: f64,
        strike_y: f64,
        strike_velocity: f64,
        gap: f64,
        eta_tol: f64,
        newton_tol: f64,
        newton_maxiter: i64,
    ) -> PyResult<Self> {
        // The original's five scalar checks touch nothing on the membrane, so they run before the
        // cast: a call that is both massless and holding a drumhead this class cannot drive must
        // report the mass, the way Python's would.
        core::check_common(mass, stiffness, alpha, hysteresis, gap).map_err(param_err)?;

        let handle: Py<PyMembrane> = membrane
            .clone()
            .cast_into::<PyMembrane>()
            .map_err(|_| {
                PyTypeError::new_err(
                    "the Rust MalletMembrane needs a Rust Membrane (physsynth_rs.Membrane). Got \
                     something else -- most likely the pure-Python `membrane.MembranePy`, which \
                     this class cannot strike without crossing back into the interpreter every \
                     timestep. Build the drumhead from the same implementation as the mallet.",
                )
            })?
            .unbind();

        let params = {
            let mem = handle.bind(py).borrow();
            core::Params::new(
                mem.params(),
                mass,
                stiffness,
                alpha,
                hysteresis,
                strike_x,
                strike_y,
                gap,
                eta_tol,
                newton_tol,
                maxiter_of(newton_maxiter),
            )
            .map_err(param_err)?
        };

        if params.steps_per_contact < 8.0 {
            warn_under_resolved(py, params.steps_per_contact)?;
        }

        let u_node = handle.bind(py).borrow().u_at(py, params.node)?;
        let s = core::State::at_strike(gap, strike_velocity, params.k, u_node);
        Ok(PyMalletMembrane {
            p: params,
            s,
            membrane: handle,
        })
    }

    // -- parameters --------------------------------------------------------------------------

    /// The drumhead — the very object the caller passed in.
    #[getter]
    fn membrane(&self, py: Python<'_>) -> Py<PyMembrane> {
        self.membrane.clone_ref(py)
    }
    #[getter]
    fn k(&self) -> f64 {
        self.p.k
    }
    #[getter]
    fn M(&self) -> f64 {
        self.p.mass
    }
    #[getter]
    fn K(&self) -> f64 {
        self.p.stiffness
    }
    #[getter]
    fn alpha(&self) -> f64 {
        self.p.alpha
    }
    #[getter]
    fn lam_h(&self) -> f64 {
        self.p.lam_h
    }
    #[getter]
    fn eta_tol(&self) -> f64 {
        self.p.eta_tol
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
    fn x_strike(&self) -> f64 {
        self.p.x_strike
    }
    #[getter]
    fn y_strike(&self) -> f64 {
        self.p.y_strike
    }
    #[getter]
    fn contact_frequency(&self) -> f64 {
        self.p.contact_frequency
    }
    #[getter]
    fn strike_velocity(&self) -> f64 {
        self.s.strike_velocity
    }

    // The three admittances. §12.2's scar — a leading underscore is not a statement about the
    // interface — says to expose them rather than assume nobody reads them; nothing in `tests/`
    // or `web/` does today, but `_accel` did not either until three modules assigned to it.
    #[getter]
    fn _g_s(&self) -> f64 {
        self.p.g_s
    }
    #[getter]
    fn _g_h(&self) -> f64 {
        self.p.g_h
    }
    #[getter]
    fn _g(&self) -> f64 {
        self.p.g
    }

    // -- state -------------------------------------------------------------------------------

    /// Mallet position `z_H^n`. Settable — placing the mallet by hand is the one state edit that
    /// has an obvious meaning.
    #[getter]
    fn z_H(&self) -> f64 {
        self.s.z_h
    }
    #[setter]
    fn set_z_H(&mut self, value: f64) {
        self.s.z_h = value;
    }
    #[getter]
    fn z_H_prev(&self) -> f64 {
        self.s.z_h_prev
    }
    #[setter]
    fn set_z_H_prev(&mut self, value: f64) {
        self.s.z_h_prev = value;
    }

    #[getter]
    fn penetration(&self) -> f64 {
        self.s.penetration
    }
    #[getter]
    fn contact_force(&self) -> f64 {
        self.s.contact_force
    }
    #[getter]
    fn in_contact(&self) -> bool {
        self.s.in_contact
    }
    #[getter]
    fn fallbacks(&self) -> usize {
        self.s.fallbacks
    }
    #[getter]
    fn n(&self) -> usize {
        self.s.n
    }

    /// The membrane displacement field (full 2-D array, for animation snapshots).
    #[getter]
    fn state(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.membrane.bind(py).borrow().state(py)
    }

    // -- time stepping -----------------------------------------------------------------------

    /// Advance one step: force-free advance, scalar contact solve, exact force inject.
    fn step(&mut self, py: Python<'_>) -> PyResult<()> {
        // Take the handle first so nothing borrows `self.membrane` while the mutable borrow below
        // is live.
        let handle = self.membrane.clone_ref(py);
        let mut mem = handle.bind(py).borrow_mut();
        let i = self.p.node;

        // `eta^{n-1}` must be read BEFORE the step: the membrane's `step()` rebinds `u_prev` to
        // what `u` was, so this quantity stops existing one line later.
        let eta_prev = core::eta_prev(mem.u_prev_at(py, i)?, &self.s);
        mem.step(py)?;
        let u_free = mem.u_at(py, i)?;
        let z_free = core::free_flight(&self.s);
        let u_corrected =
            core::resolve(u_free, eta_prev, z_free, &self.p, &mut self.s).map_err(solve_err)?;
        mem.set_u_at(py, i, u_corrected)
    }

    // -- diagnostics -------------------------------------------------------------------------

    /// Total discrete energy `H^n` (J): membrane + mallet KE + averaged contact PE.
    ///
    /// The membrane term is a **reduction**, which makes this the one observable on this class
    /// that a port cannot claim to the bit (§14.2). `MalletWall`, which owns no field, can — and
    /// that contrast is what attributes the difference rather than assuming it.
    fn energy(&self, py: Python<'_>) -> PyResult<f64> {
        let mem = self.membrane.bind(py).borrow();
        let i = self.p.node;
        Ok(core::energy(
            mem.u_at(py, i)?,
            mem.u_prev_at(py, i)?,
            mem.energy(py)?,
            &self.p,
            &self.s,
        ))
    }

    /// Membrane pickup at flat live-node `index` — for spectral analysis of the tone.
    fn displacement_at(&self, py: Python<'_>, index: i64) -> PyResult<f64> {
        self.membrane.bind(py).borrow().displacement_at(py, index)
    }

    /// Mallet velocity `delta_t- z_H` (m/s): negative into the head, positive after rebound.
    fn mallet_velocity(&self) -> f64 {
        self.s.velocity(self.p.k)
    }
}

/// A lumped mass in one-sided contact with a fixed rigid wall — the Rust implementation, wearing
/// the Python interface.
///
/// Attribute-for-attribute and method-for-method compatible with
/// `physsynth.core.mallet.MalletWall`; the docstring on that class is the reference.
#[pyclass(name = "MalletWall", module = "physsynth_rs")]
pub struct PyMalletWall {
    p: core::WallParams,
    s: core::State,
}

#[pymethods]
impl PyMalletWall {
    #[allow(clippy::too_many_arguments)]
    #[new]
    #[pyo3(signature = (
        *, mass, stiffness, fs, alpha=1.0, hysteresis=0.0, wall_position=0.0, strike_velocity,
        gap=0.0, eta_tol=1e-12, newton_tol=1e-14, newton_maxiter=60
    ))]
    fn new(
        mass: f64,
        stiffness: f64,
        fs: f64,
        alpha: f64,
        hysteresis: f64,
        wall_position: f64,
        strike_velocity: f64,
        gap: f64,
        eta_tol: f64,
        newton_tol: f64,
        newton_maxiter: i64,
    ) -> PyResult<Self> {
        let p = core::WallParams::new(
            mass,
            stiffness,
            fs,
            alpha,
            hysteresis,
            wall_position,
            gap,
            eta_tol,
            newton_tol,
            maxiter_of(newton_maxiter),
        )
        .map_err(param_err)?;
        let s = core::State::at_wall(p.wall, gap, strike_velocity, p.k);
        Ok(PyMalletWall { p, s })
    }

    // -- parameters --------------------------------------------------------------------------

    #[getter]
    fn k(&self) -> f64 {
        self.p.k
    }
    #[getter]
    fn M(&self) -> f64 {
        self.p.mass
    }
    #[getter]
    fn K(&self) -> f64 {
        self.p.stiffness
    }
    #[getter]
    fn alpha(&self) -> f64 {
        self.p.alpha
    }
    #[getter]
    fn lam_h(&self) -> f64 {
        self.p.lam_h
    }
    #[getter]
    fn wall(&self) -> f64 {
        self.p.wall
    }
    #[getter]
    fn eta_tol(&self) -> f64 {
        self.p.eta_tol
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
    fn strike_velocity(&self) -> f64 {
        self.s.strike_velocity
    }
    #[getter]
    fn _g(&self) -> f64 {
        self.p.g
    }

    // -- state -------------------------------------------------------------------------------

    #[getter]
    fn z_H(&self) -> f64 {
        self.s.z_h
    }
    #[setter]
    fn set_z_H(&mut self, value: f64) {
        self.s.z_h = value;
    }
    #[getter]
    fn z_H_prev(&self) -> f64 {
        self.s.z_h_prev
    }
    #[setter]
    fn set_z_H_prev(&mut self, value: f64) {
        self.s.z_h_prev = value;
    }

    #[getter]
    fn penetration(&self) -> f64 {
        self.s.penetration
    }
    #[getter]
    fn contact_force(&self) -> f64 {
        self.s.contact_force
    }
    #[getter]
    fn in_contact(&self) -> bool {
        self.s.in_contact
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

    /// Advance one step: force-free mallet flight, scalar contact solve, exact force inject.
    fn step(&mut self) -> PyResult<()> {
        core::wall_step(&self.p, &mut self.s).map_err(solve_err)
    }

    // -- diagnostics -------------------------------------------------------------------------

    /// Total energy `0.5 M (delta_t- z_H)^2 + 0.5 (phi(eta^n) + phi(eta^{n-1}))` (J).
    fn energy(&self) -> f64 {
        core::wall_energy(&self.p, &self.s)
    }

    /// Mallet velocity: `-strike_velocity` inbound, positive after rebound.
    fn velocity(&self) -> f64 {
        self.s.velocity(self.p.k)
    }
}
