//! The binding over `physsynth_core::radiation` — the air node wearing the Python interface.
//!
//! # The two loaded bodies require a Rust `ModalBody`, and that is the reed's rule, not a new one
//!
//! `PyReedBore` refuses a pure-Python bore rather than falling back, because a silent fallback
//! would be a Rust reed reporting Rust while driving a Python air column. The same argument
//! applies here and the same `TypeError` is raised: both wrappers extract a [`PyModalBody`] and
//! refuse anything else. Under `PHYSSYNTH_RS=1` the two swaps fire together, so `tests/helpers.py`
//! and `web/serialize.py` hand these classes a Rust body without knowing they did.
//!
//! [`PyAirRadiation::radiate`] is the deliberate exception. It duck-types on `pressure()` and its
//! callers hand it a `Bore`, a `StringBodyBridge` and a `Plate` as well as a body — none of which
//! are ported, several of which never will be by this route — so it takes any object at all.
//!
//! # `__getattr__`, and why a pyclass can wear it
//!
//! `RadiatedBody` is a drop-in wherever a bare `ModalBody` is expected: `web/serialize.py` hands
//! one to `StringBodyBridge` as the body, and `connection.py` — still Python, and Phase 5 — then
//! reads `M`, `m`, `omega`, `phi`, `q_prev`, `bridge_displacement`, `pressure` and `step` off it.
//! The original serves those through `__getattr__`; so does this, and the semantics carry over
//! unchanged because Python consults `__getattr__` only after normal lookup fails, and a
//! pyclass's own getters *are* normal lookup. Measured surface: those ten names, and **nothing
//! outside this module writes an attribute on any of the four types** — which matters, because a
//! pyclass has no instance `__dict__`, so a write Python would have silently accepted would
//! raise here. That check is the `_accel` finding (plan §12.2) run one batch later and coming
//! back clean.
//!
//! # What is *not* here
//!
//! `piston_radiation_resistance` — the Bessel one. See the core module's header: it is Phase 7's
//! problem, and `physsynth/core/radiation.py` keeps its own copy while the other five names swap.

use numpy::PyArray1;
use physsynth_core::radiation as core;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyComplex;

use crate::body::PyModalBody;

/// Free-space acoustic radiation resistance of a compact monopole (Pa·s/m³).
#[pyfunction]
#[pyo3(name = "monopole_radiation_resistance", signature = (omega, *, rho0=core::RHO0_AIR, c0=core::C0_AIR))]
pub fn py_monopole_radiation_resistance(omega: f64, rho0: f64, c0: f64) -> f64 {
    core::monopole_radiation_resistance(omega, rho0, c0)
}

/// Extract the Rust modal body, refusing a Python one loudly. See the module header.
fn require_body(body: &Bound<'_, PyAny>) -> PyResult<Py<PyModalBody>> {
    body.extract().map_err(|_| {
        PyTypeError::new_err(
            "body must be a physsynth_rs.ModalBody. The object passed is not one — most likely \
             the pure-Python `body.ModalBodyPy`, which this class cannot load without stepping \
             back into the interpreter every sample. Build the body from `physsynth.core.body` \
             with PHYSSYNTH_RS set, or use `radiation.RadiatedBodyPy`.",
        )
    })
}

// =================================================================================================
// Tier 1 — the read-out
// =================================================================================================

/// Free-space monopole radiation: body volume acceleration -> far-field pressure at `r`.
///
/// Attribute-for-attribute compatible with `physsynth.core.radiation.AirRadiation`.
#[pyclass(name = "AirRadiation", module = "physsynth_rs")]
pub struct PyAirRadiation {
    inner: core::AirRadiation,
}

#[pymethods]
impl PyAirRadiation {
    #[new]
    #[pyo3(signature = (*, fs, distance=1.0, rho0=core::RHO0_AIR, c0=core::C0_AIR, retarded=true))]
    fn new(fs: f64, distance: f64, rho0: f64, c0: f64, retarded: bool) -> PyResult<Self> {
        let p = core::AirParams::new(fs, distance, rho0, c0, retarded)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyAirRadiation {
            inner: core::AirRadiation::new(p),
        })
    }

    #[getter]
    fn fs(&self) -> f64 {
        self.inner.params().fs
    }
    #[getter]
    fn k(&self) -> f64 {
        self.inner.params().k
    }
    #[getter]
    fn distance(&self) -> f64 {
        self.inner.params().distance
    }
    #[getter]
    fn rho0(&self) -> f64 {
        self.inner.params().rho0
    }
    #[getter]
    fn c0(&self) -> f64 {
        self.inner.params().c0
    }
    #[getter]
    fn retarded(&self) -> bool {
        self.inner.params().retarded
    }
    #[getter]
    fn gain(&self) -> f64 {
        self.inner.params().gain
    }
    #[getter]
    fn retardation_seconds(&self) -> f64 {
        self.inner.params().retardation_seconds
    }
    #[getter]
    fn latency_samples(&self) -> usize {
        self.inner.params().latency_samples
    }
    #[getter]
    fn retardation_residual(&self) -> f64 {
        self.inner.params().retardation_residual
    }
    #[getter]
    fn n(&self) -> usize {
        self.inner.n()
    }

    /// The delay line, as a fresh array.
    ///
    /// Private in the original and read there by `tests/test_radiation.py`, which asserts it is
    /// all zeros after `reset()`. Nothing writes it, so unlike the body's three state buffers this
    /// one does not have to be a Python-owned array — a copy is enough and cannot be
    /// accidentally aliased.
    #[getter]
    fn _buf(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        PyArray1::from_slice(py, self.inner.buf()).unbind()
    }

    #[getter]
    fn _idx(&self) -> usize {
        self.inner.idx()
    }

    /// Map one volume-acceleration sample `Q''` to the far-field pressure `p_far` (Pa).
    fn process(&mut self, volume_accel: f64) -> f64 {
        self.inner.process(volume_accel)
    }

    /// Read `source.pressure()` and [`PyAirRadiation::process`] it.
    ///
    /// Takes any object with a `pressure()`, on purpose — the callers pass bores, bridges and
    /// plates as well as bodies.
    fn radiate(&mut self, source: &Bound<'_, PyAny>) -> PyResult<f64> {
        let q = source.call_method0("pressure")?.extract::<f64>()?;
        Ok(self.inner.process(q))
    }

    /// Clear the delay line and the sample counter.
    fn reset(&mut self) {
        self.inner.reset();
    }
}

// =================================================================================================
// The rank-1 dashpot the two loaded bodies share
// =================================================================================================

/// The state a loaded body keeps besides its body: the rank-1 precomputes and the step counter.
struct RankOne {
    body: Py<PyModalBody>,
    k: f64,
    g: f64,
    corr: Vec<f64>,
    n: usize,
}

impl RankOne {
    fn new(py: Python<'_>, body: Py<PyModalBody>) -> RankOne {
        let (k, g, corr) = {
            let b = body.bind(py).borrow();
            let p = b.core_params();
            let (g, corr) = core::rank_one(p);
            (p.k, g, corr)
        };
        RankOne {
            body,
            k,
            g,
            corr,
            n: 0,
        }
    }

    /// The force-free advance and the free centered volume velocity, before the load is applied.
    ///
    /// Returns `(u_free, q_nm1)` — the caller needs `q^{n-1}` again to refresh the acceleration.
    fn advance(&self, py: Python<'_>, force: f64) -> PyResult<(f64, Vec<f64>)> {
        let q_nm1 = self.body.bind(py).borrow().q_prev_vec(py)?;
        self.body.bind(py).borrow_mut().step(py, force)?;
        let b = self.body.bind(py).borrow();
        let q = b.q_vec(py)?;
        Ok((
            core::free_volume_velocity(&q, &q_nm1, b.core_params()),
            q_nm1,
        ))
    }

    /// Apply the rank-1 correction `q -= scale * corr` and rewrite `_accel` from the corrected
    /// second difference — the two assignments `radiation.py` spells out in Python.
    fn correct(&self, py: Python<'_>, scale: f64, q_nm1: &[f64]) -> PyResult<()> {
        let mut b = self.body.bind(py).borrow_mut();
        let q_tilde = b.q_vec(py)?;
        let q_prev = b.q_prev_vec(py)?;
        let mut q = vec![0.0; q_tilde.len()];
        core::correct_into(&q_tilde, scale, &self.corr, &mut q);
        let mut accel = vec![0.0; q.len()];
        core::refresh_accel(&q, &q_prev, q_nm1, self.k, &mut accel);
        b.adopt_corrected(py, q, accel);
        Ok(())
    }

    fn corr_array(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        PyArray1::from_slice(py, &self.corr).unbind()
    }
}

// =================================================================================================
// Tier 2 — the constant-resistance load
// =================================================================================================

/// A `ModalBody` loaded by its own radiation resistance — the passive back-reaction.
///
/// Attribute-for-attribute and method-for-method compatible with
/// `physsynth.core.radiation.RadiatedBody`, `__getattr__` delegation included.
#[pyclass(name = "RadiatedBody", module = "physsynth_rs")]
pub struct PyRadiatedBody {
    rank: RankOne,
    R: f64,
    radiated_energy: f64,
    volume_velocity: f64,
}

#[pymethods]
impl PyRadiatedBody {
    #[new]
    #[pyo3(signature = (*, body, R))]
    fn new(py: Python<'_>, body: &Bound<'_, PyAny>, R: f64) -> PyResult<Self> {
        if R < 0.0 {
            return Err(PyValueError::new_err(
                "radiation resistance R must be >= 0.",
            ));
        }
        Ok(PyRadiatedBody {
            rank: RankOne::new(py, require_body(body)?),
            R,
            radiated_energy: 0.0,
            volume_velocity: 0.0,
        })
    }

    /// Delegate anything this class does not define to the body it loads. See the module header.
    fn __getattr__(&self, py: Python<'_>, name: &str) -> PyResult<Py<PyAny>> {
        Ok(self.rank.body.bind(py).getattr(name)?.unbind())
    }

    #[getter]
    fn body(&self, py: Python<'_>) -> Py<PyModalBody> {
        self.rank.body.clone_ref(py)
    }
    #[getter]
    fn R(&self) -> f64 {
        self.R
    }
    #[getter]
    fn k(&self) -> f64 {
        self.rank.k
    }
    #[getter]
    fn radiated_energy(&self) -> f64 {
        self.radiated_energy
    }
    #[getter]
    fn volume_velocity(&self) -> f64 {
        self.volume_velocity
    }
    #[getter]
    fn n(&self) -> usize {
        self.rank.n
    }
    /// The scalar driving-point factor `G`. Private in the original; exposed for the parity test.
    #[getter]
    fn _G(&self) -> f64 {
        self.rank.g
    }
    /// The per-mode correction prefactor. Private in the original; exposed for the parity test.
    #[getter]
    fn _corr(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        self.rank.corr_array(py)
    }

    /// Advance one step: force-free body advance, scalar solve, rank-1 correction.
    #[pyo3(signature = (force=0.0))]
    fn step(&mut self, py: Python<'_>, force: f64) -> PyResult<()> {
        let (u_free, q_nm1) = self.rank.advance(py, force)?;
        let u = u_free / (1.0 + self.R * self.rank.g);
        self.rank.correct(py, self.R * u, &q_nm1)?;
        self.radiated_energy += self.rank.k * self.R * u * u;
        self.volume_velocity = u;
        self.rank.n += 1;
        Ok(())
    }

    /// `E_body + integral P_rad dt` (Joules) — the conserved total.
    fn energy(&self, py: Python<'_>) -> PyResult<f64> {
        Ok(self.rank.body.bind(py).borrow().energy(py)? + self.radiated_energy)
    }

    /// Radiated pressure read-out, carrying the load.
    fn pressure(&self, py: Python<'_>) -> PyResult<f64> {
        self.rank.body.bind(py).borrow().pressure(py)
    }

    /// Set the body's initial modal state and reset the radiated-energy channel.
    #[pyo3(signature = (q0, v0=None))]
    fn set_state(
        &mut self,
        py: Python<'_>,
        q0: &Bound<'_, PyAny>,
        v0: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        self.rank.body.bind(py).borrow_mut().set_state(py, q0, v0)?;
        self.radiated_energy = 0.0;
        self.volume_velocity = 0.0;
        self.rank.n = 0;
        Ok(())
    }

    /// Zero the body state and the radiated-energy channel.
    fn reset(&mut self, py: Python<'_>) -> PyResult<()> {
        let zero = 0.0f64.into_pyobject(py)?;
        self.rank
            .body
            .bind(py)
            .borrow_mut()
            .set_state(py, zero.as_any(), None)?;
        self.radiated_energy = 0.0;
        self.volume_velocity = 0.0;
        self.rank.n = 0;
        Ok(())
    }
}

// =================================================================================================
// Tier 3 — the exact first-order rational impedance
// =================================================================================================

/// The air as a first-order positive-real impedance — resistance *and* radiation mass.
///
/// Attribute-for-attribute and method-for-method compatible with
/// `physsynth.core.radiation.RationalAirLoad`.
#[pyclass(name = "RationalAirLoad", module = "physsynth_rs")]
pub struct PyRationalAirLoad {
    inner: core::RationalAirLoad,
}

/// `Z_a` as a Python `complex`.
fn to_complex(py: Python<'_>, z: core::C64) -> Py<PyComplex> {
    PyComplex::from_doubles(py, z.0, z.1).unbind()
}

#[pymethods]
impl PyRationalAirLoad {
    #[new]
    #[pyo3(signature = (*, fs, R, M_a=f64::INFINITY, rho0=core::RHO0_AIR, c0=core::C0_AIR))]
    fn new(fs: f64, R: f64, M_a: f64, rho0: f64, c0: f64) -> PyResult<Self> {
        let p = core::LoadParams::new(fs, R, M_a, rho0, c0)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyRationalAirLoad {
            inner: core::RationalAirLoad::new(p),
        })
    }

    /// The physically consistent pulsating sphere of radius `a` — the exact monopole load.
    #[classmethod]
    #[pyo3(signature = (*, fs, radius, rho0=core::RHO0_AIR, c0=core::C0_AIR))]
    fn from_sphere(
        _cls: &Bound<'_, pyo3::types::PyType>,
        fs: f64,
        radius: f64,
        rho0: f64,
        c0: f64,
    ) -> PyResult<Self> {
        let p = core::LoadParams::from_sphere(fs, radius, rho0, c0)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyRationalAirLoad {
            inner: core::RationalAirLoad::new(p),
        })
    }

    #[getter]
    fn fs(&self) -> f64 {
        self.inner.params().fs
    }
    #[getter]
    fn k(&self) -> f64 {
        self.inner.params().k
    }
    #[getter]
    fn R(&self) -> f64 {
        self.inner.params().r
    }
    #[getter]
    fn M_a(&self) -> f64 {
        self.inner.params().m_a
    }
    #[getter]
    fn rho0(&self) -> f64 {
        self.inner.params().rho0
    }
    #[getter]
    fn c0(&self) -> f64 {
        self.inner.params().c0
    }
    #[getter]
    fn R_eff(&self) -> f64 {
        self.inner.params().r_eff
    }
    #[getter]
    fn tau(&self) -> f64 {
        self.inner.params().tau
    }
    /// The equivalent sphere radius, or `None` when the `(R, M_a)` pair is not sphere-consistent.
    #[getter]
    fn sphere_radius(&self) -> Option<f64> {
        self.inner.params().sphere_radius
    }
    #[getter]
    fn sphere_area(&self) -> Option<f64> {
        self.inner.params().sphere_area
    }
    #[getter]
    fn u_l(&self) -> f64 {
        self.inner.u_l
    }
    #[getter]
    fn radiated_energy(&self) -> f64 {
        self.inner.radiated_energy
    }
    #[getter]
    fn volume_velocity(&self) -> f64 {
        self.inner.volume_velocity
    }
    #[getter]
    fn pressure_load(&self) -> f64 {
        self.inner.pressure_load
    }
    #[getter]
    fn n(&self) -> usize {
        self.inner.n()
    }

    /// Load pressure `p^n` and centered volume velocity `U^n` — *without* committing.
    fn solve(&self, u_free: f64, G: f64) -> (f64, f64) {
        self.inner.solve(u_free, G)
    }

    /// Advance the auxiliary state on the accepted `(p, U)` and book the energy split.
    fn commit(&mut self, p: f64, u: f64) {
        self.inner.commit(p, u);
    }

    /// [`PyRationalAirLoad::solve`] then [`PyRationalAirLoad::commit`] — the driven standalone form.
    #[pyo3(signature = (u_free, G=0.0))]
    fn step(&mut self, u_free: f64, G: f64) -> (f64, f64) {
        self.inner.step(u_free, G)
    }

    /// Kinetic energy of the radiation mass (Joules); exactly zero for the constant-`R` load.
    fn stored_energy(&self) -> f64 {
        self.inner.stored_energy()
    }

    /// The air's whole share: stored plus radiated.
    fn energy(&self) -> f64 {
        self.inner.energy()
    }

    /// Continuous acoustic impedance `Z_a(j omega)`.
    fn impedance(&self, py: Python<'_>, omega: f64) -> Py<PyComplex> {
        to_complex(py, self.inner.impedance(omega))
    }

    /// The scheme's impedance: `Z_a` at the pre-warped `s = (2j / k) tan(omega k / 2)`.
    fn impedance_discrete(&self, py: Python<'_>, omega: f64) -> Py<PyComplex> {
        to_complex(py, self.inner.impedance_discrete(omega))
    }

    /// Closed-form `(omega_eff, alpha)` of one weakly loaded mode — both parts of `Z_a`.
    #[pyo3(signature = (omega0, *, weight, mass, iterations=50, tol=1e-14))]
    fn loaded_mode(
        &self,
        omega0: f64,
        weight: f64,
        mass: f64,
        iterations: usize,
        tol: f64,
    ) -> PyResult<(f64, f64)> {
        self.inner
            .loaded_mode(omega0, weight, mass, iterations, tol)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Far-field pressure at `r` from the sphere's own surface pressure, `(a / r) p_load`.
    #[pyo3(signature = (distance, p_load=None))]
    fn far_field_pressure(&self, distance: f64, p_load: Option<f64>) -> PyResult<f64> {
        self.inner
            .far_field_pressure(distance, p_load)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Zero the auxiliary state, the radiated channel and the counters.
    fn reset(&mut self) {
        self.inner.reset();
    }
}

/// A `ModalBody` loaded by a frequency-dependent radiation impedance.
///
/// Attribute-for-attribute and method-for-method compatible with
/// `physsynth.core.radiation.ReactiveRadiatedBody`, `__getattr__` delegation included.
#[pyclass(name = "ReactiveRadiatedBody", module = "physsynth_rs")]
pub struct PyReactiveRadiatedBody {
    rank: RankOne,
    load: Py<PyRationalAirLoad>,
}

#[pymethods]
impl PyReactiveRadiatedBody {
    #[new]
    #[pyo3(signature = (*, body, load))]
    fn new(py: Python<'_>, body: &Bound<'_, PyAny>, load: &Bound<'_, PyAny>) -> PyResult<Self> {
        let body = require_body(body)?;
        let load: Py<PyRationalAirLoad> = load.extract().map_err(|_| {
            PyTypeError::new_err(
                "load must be a physsynth_rs.RationalAirLoad — the pure-Python \
                 `radiation.RationalAirLoadPy` cannot drive this class.",
            )
        })?;
        let body_k = body.bind(py).borrow().core_params().k;
        let load_p = *load.bind(py).borrow().inner.params();
        // `np.isclose(load.k, body.k, rtol=1e-12, atol=0.0)` — asymmetric; the tolerance scales
        // on the SECOND argument, which is the body's.
        if !core::isclose(load_p.k, body_k, 1e-12) {
            return Err(PyValueError::new_err(core::timestep_mismatch(
                load_p.fs, body_k,
            )));
        }
        Ok(PyReactiveRadiatedBody {
            rank: RankOne::new(py, body),
            load,
        })
    }

    /// Delegate anything this class does not define to the body it loads.
    fn __getattr__(&self, py: Python<'_>, name: &str) -> PyResult<Py<PyAny>> {
        Ok(self.rank.body.bind(py).getattr(name)?.unbind())
    }

    #[getter]
    fn body(&self, py: Python<'_>) -> Py<PyModalBody> {
        self.rank.body.clone_ref(py)
    }
    /// The air impedance — the object the caller passed, not a copy.
    #[getter]
    fn load(&self, py: Python<'_>) -> Py<PyRationalAirLoad> {
        self.load.clone_ref(py)
    }
    #[getter]
    fn k(&self) -> f64 {
        self.rank.k
    }
    #[getter]
    fn n(&self) -> usize {
        self.rank.n
    }
    #[getter]
    fn _G(&self) -> f64 {
        self.rank.g
    }
    #[getter]
    fn _corr(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        self.rank.corr_array(py)
    }

    /// Energy handed to the far field — the load's dissipated share. A property in the original.
    #[getter]
    fn radiated_energy(&self, py: Python<'_>) -> f64 {
        self.load.bind(py).borrow().inner.radiated_energy
    }

    /// Last centered total volume velocity `U^n`. A property in the original.
    #[getter]
    fn volume_velocity(&self, py: Python<'_>) -> f64 {
        self.load.bind(py).borrow().inner.volume_velocity
    }

    /// Advance one step: force-free body advance, scalar load solve, rank-1 correction.
    #[pyo3(signature = (force=0.0))]
    fn step(&mut self, py: Python<'_>, force: f64) -> PyResult<()> {
        let (u_free, q_nm1) = self.rank.advance(py, force)?;
        // Solve on a shared borrow, correct the body, then commit on a fresh mutable borrow —
        // never holding both at once. Nothing here re-enters the interpreter, so this is only
        // bookkeeping, not the two-phase dance `reed` needs (plan §13.2).
        let (p_load, u) = self.load.bind(py).borrow().inner.solve(u_free, self.rank.g);
        self.rank.correct(py, p_load, &q_nm1)?;
        self.load.bind(py).borrow_mut().inner.commit(p_load, u);
        self.rank.n += 1;
        Ok(())
    }

    /// `E_body + E_air` (Joules), the air term being stored plus radiated.
    fn energy(&self, py: Python<'_>) -> PyResult<f64> {
        let body = self.rank.body.bind(py).borrow().energy(py)?;
        Ok(body + self.load.bind(py).borrow().inner.energy())
    }

    /// Monopole read-out carrying the load.
    fn pressure(&self, py: Python<'_>) -> PyResult<f64> {
        self.rank.body.bind(py).borrow().pressure(py)
    }

    /// Far-field pressure at `r` from the finite sphere.
    fn far_field_pressure(&self, py: Python<'_>, distance: f64) -> PyResult<f64> {
        self.load
            .bind(py)
            .borrow()
            .inner
            .far_field_pressure(distance, None)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Set the body's initial modal state and reset the air.
    #[pyo3(signature = (q0, v0=None))]
    fn set_state(
        &mut self,
        py: Python<'_>,
        q0: &Bound<'_, PyAny>,
        v0: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        self.rank.body.bind(py).borrow_mut().set_state(py, q0, v0)?;
        self.load.bind(py).borrow_mut().inner.reset();
        self.rank.n = 0;
        Ok(())
    }

    /// Zero the body state and the air's auxiliary state and channels.
    fn reset(&mut self, py: Python<'_>) -> PyResult<()> {
        let zero = 0.0f64.into_pyobject(py)?;
        self.rank
            .body
            .bind(py)
            .borrow_mut()
            .set_state(py, zero.as_any(), None)?;
        self.load.bind(py).borrow_mut().inner.reset();
        self.rank.n = 0;
        Ok(())
    }
}
