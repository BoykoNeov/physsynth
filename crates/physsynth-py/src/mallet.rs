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

use numpy::PyArray1;
use physsynth_core::mallet as core;
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyAny;
use std::ffi::CString;

use crate::membrane::PyMembrane;
use crate::plate::{PyPlate, PyVKPlate};

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

/// A plate struck by a lumped-mass mallet — the Rust implementation, wearing the Python interface.
///
/// The reference for the physics is the `physsynth_core::mallet` module header's plate section;
/// `physsynth.core.mallet.MalletPlate` re-exports this name.
///
/// Holds a `Py<PyPlate>` handle rather than a copy, for the same reason `PyMalletMembrane` holds
/// its drumhead: `mal.plate` has to **be** the object that was passed in, because every caller
/// reads the field, the mask and the energy back through it.
#[pyclass(name = "MalletPlate", module = "physsynth_rs")]
pub struct PyMalletPlate {
    p: core::PlateParams,
    s: core::State,
    plate: Py<PyPlate>,
}

#[pymethods]
impl PyMalletPlate {
    // Twelve keyword arguments plus the GIL token — `MalletMembrane`'s signature with `membrane`
    // replaced by `plate`, deliberately, so that swapping which body is struck is a one-word edit
    // at every call site.
    #[allow(clippy::too_many_arguments)]
    #[new]
    #[pyo3(signature = (
        *, plate, mass, stiffness, alpha=2.3, hysteresis=0.0, strike_x, strike_y,
        strike_velocity, gap=0.0, eta_tol=1e-12, newton_tol=1e-14, newton_maxiter=60
    ))]
    fn new(
        py: Python<'_>,
        plate: &Bound<'_, PyAny>,
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
        // The five scalar checks touch nothing on the plate, so they run before the cast: a call
        // that is both massless and holding the wrong body reports the mass, the way the membrane
        // model's does.
        core::check_common(mass, stiffness, alpha, hysteresis, gap).map_err(param_err)?;

        let handle: Py<PyPlate> = plate
            .clone()
            .cast_into::<PyPlate>()
            .map_err(|_| {
                PyTypeError::new_err(
                    "MalletPlate strikes a linear Plate (physsynth_rs.Plate). Got something else \
                     -- if it is a VKPlate, use MalletVKPlate. That is not a missing cast but a \
                     different model: the von Karman step is nonlinear, so it is not affine in \
                     f_ext and the drive-point influence column this mallet is built on does not \
                     exist. A gong needs an outer contact solve wrapped around the plate's own \
                     iteration, and MalletVKPlate is that model.",
                )
            })?
            .unbind();

        let params = {
            let pl = handle.bind(py).borrow();
            core::PlateParams::new(
                pl.params(),
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
        Ok(PyMalletPlate {
            p: params,
            s,
            plate: handle,
        })
    }

    // -- parameters --------------------------------------------------------------------------

    /// The plate — the very object the caller passed in.
    #[getter]
    fn plate(&self, py: Python<'_>) -> Py<PyPlate> {
        self.plate.clone_ref(py)
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
    fn steps_per_contact(&self) -> f64 {
        self.p.steps_per_contact
    }
    #[getter]
    fn strike_velocity(&self) -> f64 {
        self.s.strike_velocity
    }

    // The three admittances and the column they come from. Exposed rather than assumed private —
    // §12.2's scar, and here the column is the batch's whole new idea, so a test that wants to
    // check the superposition identity directly must be able to see it. A **copy** each call: the
    // column is a construction-time constant and handing out a view of it would let a caller
    // retune the mallet by writing into an array it happened to keep.
    #[getter]
    fn _influence<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.p.influence)
    }
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

    /// Mallet position `z_H^n`. Settable, as the membrane model's is.
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

    /// The plate displacement field (full 2-D array, dead nodes zero) — for animation snapshots.
    #[getter]
    fn state(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.plate.bind(py).borrow().state(py)
    }

    // -- time stepping -----------------------------------------------------------------------

    /// Advance one step: force-free solve, scalar contact solve, force spread along the column.
    fn step(&mut self, py: Python<'_>) -> PyResult<()> {
        // Take the handle first so nothing borrows `self.plate` while the mutable borrow is live.
        let handle = self.plate.clone_ref(py);
        let mut pl = handle.bind(py).borrow_mut();
        let i = self.p.node;

        // `eta^{n-1}` must be read BEFORE the step: the plate's `step()` rebinds `u_prev` to what
        // `u` was, so this quantity stops existing one line later.
        let eta_prev = core::eta_prev(pl.u_prev_at(py, i)?, &self.s);
        pl.step(py, None)?;
        let u_free = pl.u_at(py, i)?;
        let z_free = core::free_flight(&self.s);
        let force = core::plate_resolve(u_free, eta_prev, z_free, &self.p, &mut self.s)
            .map_err(solve_err)?;
        // Two calls, two borrows, taken one after the other -- see `PyPlate::with_u_mut` for why
        // holding both at once is not safe here even though the native path holds both.
        pl.with_u_mut(py, |u| core::plate_inject_u(u, &self.p, force))?;
        pl.with_accel_mut(py, |accel| core::plate_inject_accel(accel, &self.p, force))
    }

    // -- diagnostics -------------------------------------------------------------------------

    /// Total discrete energy `H^n` (J): plate + mallet KE + averaged contact PE.
    fn energy(&self, py: Python<'_>) -> PyResult<f64> {
        let pl = self.plate.bind(py).borrow();
        let i = self.p.node;
        Ok(core::plate_total_energy(
            pl.u_at(py, i)?,
            pl.u_prev_at(py, i)?,
            pl.energy(py)?,
            &self.p,
            &self.s,
        ))
    }

    /// Plate pickup at flat live-node `index` — for spectral analysis of the tone.
    fn displacement_at(&self, py: Python<'_>, index: i64) -> PyResult<f64> {
        self.plate.bind(py).borrow().displacement_at(py, index)
    }

    /// The plate's radiated-pressure read-out, through the corrected acceleration.
    fn pressure(&self, py: Python<'_>) -> PyResult<f64> {
        self.plate.bind(py).borrow().pressure(py)
    }

    /// Mallet velocity `delta_t- z_H` (m/s): negative into the plate, positive after rebound.
    fn mallet_velocity(&self) -> f64 {
        self.s.velocity(self.p.k)
    }
}

/// A mallet striking a **von Karman** plate — model #7g, the gong.
///
/// The reference for the algorithm is the gong section of `physsynth_core::mallet`'s module
/// header; `physsynth.core.mallet.MalletVKPlate` re-exports this name.
///
/// Holds a `Py<PyVKPlate>` handle rather than a copy, for the same reason both other coupled
/// mallets do: `mal.plate` has to **be** the object that was passed in.
///
/// # Why this cannot drive the plate through `VKPlate.step`
///
/// The outer iteration steps the plate several times from the *same* time-`n` state and commits
/// exactly one of those, so it reaches the four buffers directly and writes back once. Calling
/// `step()` per trial would roll `u_prev` on the first one and every trial after it would be
/// solving a different step.
#[pyclass(name = "MalletVKPlate", module = "physsynth_rs")]
pub struct PyMalletVKPlate {
    p: core::VkPlateParams,
    s: core::State,
    plate: Py<PyVKPlate>,
    last: Option<core::VkContactStep>,
}

/// Map a nested-solve failure to the Python exception each half would raise on its own.
///
/// A contact failure is a `RuntimeError` because that is what the scalar solve has always raised;
/// a factorization failure is a `ValueError` because that is what `VKPlate.step` raises. The
/// **message** is where the attribution lives — see `VkContactError`.
fn gong_err(e: core::VkContactError) -> PyErr {
    match e {
        core::VkContactError::Contact { .. } => PyRuntimeError::new_err(e.to_string()),
        core::VkContactError::Solve(_) => PyValueError::new_err(e.to_string()),
    }
}

#[pymethods]
impl PyMalletVKPlate {
    // `MalletPlate`'s signature with `plate` taking a `VKPlate` and two arguments added for the
    // outer loop, deliberately in that shape: swapping a linear soundboard for a gong is then a
    // one-word edit at the call site plus whatever the outer loop needs.
    #[allow(clippy::too_many_arguments)]
    #[new]
    #[pyo3(signature = (
        *, plate, mass, stiffness, alpha=2.3, hysteresis=0.0, strike_x, strike_y,
        strike_velocity, gap=0.0, eta_tol=1e-12, newton_tol=1e-14, newton_maxiter=60,
        outer_tol=1e-13, outer_max_iter=20
    ))]
    fn new(
        py: Python<'_>,
        plate: &Bound<'_, PyAny>,
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
        outer_tol: f64,
        outer_max_iter: i64,
    ) -> PyResult<Self> {
        core::check_common(mass, stiffness, alpha, hysteresis, gap).map_err(param_err)?;

        let handle: Py<PyVKPlate> = plate
            .clone()
            .cast_into::<PyVKPlate>()
            .map_err(|_| {
                PyTypeError::new_err(
                    "MalletVKPlate strikes a nonlinear VKPlate (physsynth_rs.VKPlate). Got \
                     something else -- if it is a linear Plate, use MalletPlate instead: an \
                     affine step has an exact drive-point influence column and needs no outer \
                     iteration, so driving it through this model would pay for a nested solve \
                     that converges on its first pass.",
                )
            })?
            .unbind();

        let params = {
            let pl = handle.bind(py).borrow();
            core::VkPlateParams::new(
                pl.vk_params(),
                mass,
                stiffness,
                alpha,
                hysteresis,
                strike_x,
                strike_y,
                strike_velocity,
                gap,
                eta_tol,
                newton_tol,
                maxiter_of(newton_maxiter),
                outer_tol,
                maxiter_of(outer_max_iter),
            )
            .map_err(param_err)?
        };

        if params.steps_per_contact < 8.0 {
            warn_under_resolved(py, params.steps_per_contact)?;
        }

        let u_node = handle.bind(py).borrow().u_at(py, params.node)?;
        let s = core::State::at_strike(gap, strike_velocity, params.k, u_node);
        Ok(PyMalletVKPlate {
            p: params,
            s,
            plate: handle,
            last: None,
        })
    }

    // -- parameters --------------------------------------------------------------------------

    /// The gong — the very object the caller passed in.
    #[getter]
    fn plate(&self, py: Python<'_>) -> Py<PyVKPlate> {
        self.plate.clone_ref(py)
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
    fn outer_tol(&self) -> f64 {
        self.p.outer_tol
    }
    #[getter]
    fn outer_max_iter(&self) -> usize {
        self.p.outer_max_iter
    }
    /// `M v0 / k` — the force the outer residual is measured against, never `|f|`.
    #[getter]
    fn force_scale(&self) -> f64 {
        self.p.force_scale
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
    fn steps_per_contact(&self) -> f64 {
        self.p.steps_per_contact
    }
    #[getter]
    fn strike_velocity(&self) -> f64 {
        self.s.strike_velocity
    }

    /// The **linear** plate's influence column, and the three admittances read off it.
    ///
    /// `_g_s` is the chord's *frozen tangent*, not this plate's drive-point admittance — the true
    /// one is `_drive_point_tangent`'s `g_exact`, which is a function of the plate's deflection.
    /// A copy each call, as `MalletPlate`'s is.
    #[getter]
    fn _influence<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.p.influence)
    }
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

    /// The gong's displacement field (full 2-D array, rim zero) — for animation snapshots.
    #[getter]
    fn state(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.plate.bind(py).borrow().state(py)
    }

    // -- the nested solve's own telemetry ------------------------------------------------------
    //
    // Everything below is `nan`/`0` before the first step. These are the numbers the batch's cost
    // claim is made of, so they are shipped attributes rather than something a test recomputes.

    /// Outer chord iterations the last step took. **Zero on a miss.**
    #[getter]
    fn n_outer(&self) -> usize {
        self.last.as_ref().map_or(0, |l| l.n_outer)
    }
    /// Did the outer force increment fall below `outer_tol * force_scale`?
    #[getter]
    fn outer_converged(&self) -> bool {
        self.last.as_ref().is_none_or(|l| l.outer_converged)
    }
    /// Did the outer loop stop because the increment stopped shrinking, short of `outer_tol`?
    ///
    /// Its floor is the plate's own `couple_tol`, so asking for more than the inner solve supplies
    /// buys nothing — see the core module's `vk_plate_step`. Recorded, never fatal.
    #[getter]
    fn outer_stalled(&self) -> bool {
        self.last.as_ref().is_some_and(|l| l.outer_stalled)
    }
    /// That relative increment at exit — `nan` before the first step.
    #[getter]
    fn outer_residual(&self) -> f64 {
        self.last.as_ref().map_or(f64::NAN, |l| l.outer_residual)
    }
    /// Did **every** plate solve of the last step reach the plate's own `couple_tol`?
    #[getter]
    fn inner_converged(&self) -> bool {
        self.last.as_ref().is_none_or(|l| l.inner_converged)
    }
    /// Plate iterations summed over the last step's solves, the force-free one included.
    #[getter]
    fn inner_iters(&self) -> usize {
        self.last.as_ref().map_or(0, |l| l.inner_iters)
    }
    /// Back-substitutions the last step spent, summed the same way.
    ///
    /// The portable cost number. Its denominator is a **bare** `VKPlate` step from the same state,
    /// whose own `n_solves` the plate reports.
    #[getter]
    fn n_solves(&self) -> usize {
        self.last.as_ref().map_or(0, |l| l.n_solves)
    }

    // -- time stepping -----------------------------------------------------------------------

    /// Advance one step: force-free plate solve, outer chord, one commit.
    fn step(&mut self, py: Python<'_>) -> PyResult<()> {
        let handle = self.plate.clone_ref(py);
        let mut pl = handle.bind(py).borrow_mut();
        let (u, u_prev, f_cache, f_prev) = pl.state_buffers(py)?;
        let out = core::vk_plate_step(
            &u,
            &u_prev,
            &f_cache,
            &f_prev,
            &self.p,
            pl.vk_params(),
            &mut self.s,
        )
        .map_err(gong_err)?;
        pl.commit(
            py,
            out.u.clone(),
            out.f_full.clone(),
            out.inner_iters,
            out.inner_converged,
            out.outer_residual,
            out.n_solves,
        );
        self.last = Some(out);
        Ok(())
    }

    // -- diagnostics -------------------------------------------------------------------------

    /// Total discrete energy `H^n` (J): gong + mallet KE + averaged contact PE.
    fn energy(&self, py: Python<'_>) -> PyResult<f64> {
        let pl = self.plate.bind(py).borrow();
        let i = self.p.node;
        Ok(core::vk_plate_total_energy(
            pl.u_at(py, i)?,
            pl.u_prev_at(py, i)?,
            pl.energy(py)?,
            &self.p,
            &self.s,
        ))
    }

    /// Gong pickup at flat live-node `index` — for spectral analysis of the tone.
    fn displacement_at(&self, py: Python<'_>, index: i64) -> PyResult<f64> {
        self.plate.bind(py).borrow().displacement_at(py, index)
    }

    /// Mallet velocity `delta_t- z_H` (m/s): negative into the gong, positive after rebound.
    fn mallet_velocity(&self) -> f64 {
        self.s.velocity(self.p.k)
    }

    /// The **exact** outer tangent `g_exact = [J^-1 influence]_node + g_h`, and the Krylov
    /// products it cost. Returns `(g_exact, products)`.
    ///
    /// Returns `(g_exact, response, products)`: `response` is `[J^-1 influence]_node` alone, the
    /// **plate-only** half that carries no mallet quantity, and `g_exact = response + _g_h`.
    ///
    /// An instrument, not a step: `-g_exact` is the derivative `d eta / df` that the shipped chord
    /// approximates by `-_g`, and `|1 - g_exact/_g|` bounds the outer contraction factor. It costs
    /// a GMRES solve, which is why the chord ships instead of a Newton that would use it.
    ///
    /// Every argument is the caller's, because `J` has to be evaluated in the context of a
    /// specific step: `u`, `u_prev` and `F_prev` as they stood at time `n`, `force` the contact
    /// force that step was driven by, and `w` the displacement it accepted. Read the first three
    /// off `mal.plate` **before** stepping and `w` off it after.
    #[pyo3(signature = (*, u, u_prev, F_prev, force, w))]
    #[allow(non_snake_case)]
    fn _drive_point_tangent(
        &self,
        py: Python<'_>,
        u: &Bound<'_, PyAny>,
        u_prev: &Bound<'_, PyAny>,
        F_prev: &Bound<'_, PyAny>,
        force: f64,
        w: &Bound<'_, PyAny>,
    ) -> PyResult<(f64, f64, usize)> {
        let pl = self.plate.bind(py).borrow();
        let vk = pl.vk_params();
        let n_live = self.p.influence.len();
        let n_nodes = vk.n_nodes;
        let u = crate::as_1d_f64(py, u, "u", n_live)?;
        let u_prev = crate::as_1d_f64(py, u_prev, "u_prev", n_live)?;
        let f_prev = crate::as_1d_f64(py, F_prev, "F_prev", n_nodes)?;
        let w = crate::as_1d_f64(py, w, "w", n_live)?;
        let mut f_ext = vec![0.0; n_live];
        f_ext[self.p.node] = -force;
        core::vk_drive_point_tangent(&u, &u_prev, &f_prev, Some(&f_ext), &w, &self.p, vk)
            .map_err(|e| PyValueError::new_err(e.to_string()))
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
