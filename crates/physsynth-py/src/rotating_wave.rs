//! The binding over `physsynth_analysis::rotating_wave` — the geometric string's Tier B oracle.
//!
//! Like `spectrum`, this module *consumes* and *produces* rather than stepping, so plan §14.2's
//! question — "does this value reach the next timestep?" — is answered no everywhere and there is
//! no buffer-ownership question. Unlike `spectrum`, it returns a **structure**, and that is the one
//! interesting decision here.
//!
//! # Why a tuple and not a `#[pyclass]`
//!
//! The Python `RotatingWave` is a `NamedTuple` with fourteen fields. §33.2's trap — a `#[getter]`
//! with no `#[setter]` silently makes a plain attribute read-only — is the reason to check before
//! choosing, and the check is a grep for *assignment* against the class being ported (§34.6): over
//! `tests/`, `scripts/` and `web/serialize.py` nothing assigns to a `RotatingWave` field and
//! nothing calls `_replace`, which is what a NamedTuple's immutability would have forced anyway.
//!
//! So a `#[pyclass]` would be safe. It is still the wrong shape, because the Python side must keep
//! constructing the *real* `RotatingWave` NamedTuple: `web/serialize.py` and the tests read it as a
//! tuple, and a `#[pyclass]` is not one. Returning the fields lets the Python wrapper build the
//! genuine NamedTuple, so every consumer keeps the type it already had and the swap is invisible.
//! The two extra values on the end are not fields of it — they carry what the wrapper needs to
//! raise the non-convergence warning with the same text and the same step number.

use numpy::PyArray1;
use physsynth_analysis::rotating_wave as rw;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::shape::as_f64_field;

/// The four arrays, then every scalar in one nested tuple.
///
/// Split in two because PyO3 implements `IntoPyObject` for tuples up to twelve elements and this
/// carries fifteen. The split is along the only line that means anything — arrays the caller owns
/// against plain numbers — rather than at the twelfth field, so a reader of the Python side sees a
/// grouping and not an arity workaround.
type WaveTuple = (Py<PyAny>, Py<PyAny>, Py<PyAny>, Py<PyAny>, WaveScalars);

/// `(u0, w0, v0, u_prev, w_prev, v_prev)` — the two-level history, as six owned arrays.
type HistoryTuple = (
    Py<PyAny>,
    Py<PyAny>,
    Py<PyAny>,
    Py<PyAny>,
    Py<PyAny>,
    Py<PyAny>,
);

/// `(Omega, frequency, s, amplitude, mode, shape_residual, iterations, converged, time_discrete,
/// failed_step, failed_amplitude)` — the last two are not `RotatingWave` fields; they carry what
/// the Python wrapper needs to raise the non-convergence warning with the original's own text.
type WaveScalars = (
    f64,
    f64,
    f64,
    f64,
    usize,
    f64,
    usize,
    bool,
    bool,
    usize,
    f64,
);

/// Solve the rotating-wave BVP for `(phi, psi, Omega)` by amplitude continuation.
///
/// Every validation error the original raises as `ValueError` is returned as one here, with the
/// same message, because `tests/test_geometric_rotating_wave.py` matches on the text.
#[pyfunction]
#[pyo3(name = "rotating_wave_solve")]
#[pyo3(signature = (
    l, t, rho, ea, fs, n, theta, amplitude, mode, kappa, time_discrete,
    continuation_steps, tol, maxiter
))]
#[allow(clippy::too_many_arguments)]
pub fn py_solve_rotating_wave(
    py: Python<'_>,
    l: f64,
    t: f64,
    rho: f64,
    ea: f64,
    fs: f64,
    n: i64,
    theta: f64,
    amplitude: f64,
    mode: i64,
    kappa: f64,
    time_discrete: bool,
    continuation_steps: i64,
    tol: f64,
    maxiter: i64,
) -> PyResult<WaveTuple> {
    // `N`, `mode`, `continuation_steps` and `maxiter` cross as signed integers so that a negative
    // value reaches the original's own message instead of becoming a PyO3 TypeError. The Python
    // tests assert on those messages, so the conversion has to fail the same way the original does.
    if n < 2 {
        return Err(PyValueError::new_err("N must be >= 2."));
    }
    if continuation_steps < 1 {
        return Err(PyValueError::new_err("continuation_steps must be >= 1."));
    }
    if maxiter < 1 {
        return Err(PyValueError::new_err("maxiter must be >= 1."));
    }
    if mode < 1 {
        return Err(PyValueError::new_err(format!(
            "mode must be in 1 .. {}, got {mode}.",
            n - 1
        )));
    }
    let params = rw::BvpParams {
        l,
        t,
        rho,
        ea,
        fs,
        n_cells: n as usize,
        theta,
        amplitude,
        mode: mode as usize,
        kappa,
        time_discrete,
        continuation_steps: continuation_steps as usize,
        tol,
        maxiter: maxiter as usize,
    };
    let w = rw::solve_rotating_wave(&params).map_err(PyValueError::new_err)?;
    Ok((
        PyArray1::from_vec(py, w.phi).into_any().unbind(),
        PyArray1::from_vec(py, w.psi).into_any().unbind(),
        PyArray1::from_vec(py, w.stretch_ratio).into_any().unbind(),
        PyArray1::from_vec(py, w.tension).into_any().unbind(),
        (
            w.omega,
            w.frequency,
            w.s,
            w.amplitude,
            w.mode,
            w.shape_residual,
            w.iterations,
            w.converged,
            w.time_discrete,
            w.failed_step,
            w.failed_amplitude,
        ),
    ))
}

/// The exact two-level history seeding the helix: `(u0, w0, v0, u_prev, w_prev, v_prev)`.
///
/// Takes `phi`, `psi` and `Omega` rather than a wave object, because the Python side holds the real
/// NamedTuple and there is nothing to be gained by shipping it back across the boundary.
#[pyfunction]
#[pyo3(name = "rotating_wave_history")]
pub fn py_rotating_wave_history(
    py: Python<'_>,
    phi: &Bound<'_, PyAny>,
    psi: &Bound<'_, PyAny>,
    omega: f64,
    fs: f64,
) -> PyResult<HistoryTuple> {
    let (_, phi_v) = as_f64_field(py, phi, "phi")?;
    let (_, psi_v) = as_f64_field(py, psi, "psi")?;
    let wave = rw::RotatingWave {
        phi: phi_v,
        psi: psi_v,
        omega,
        frequency: 0.0,
        s: 0.0,
        amplitude: 0.0,
        mode: 1,
        stretch_ratio: Vec::new(),
        tension: Vec::new(),
        shape_residual: 0.0,
        iterations: 0,
        converged: true,
        time_discrete: true,
        failed_step: 0,
        failed_amplitude: 0.0,
    };
    let (u0, w0, v0, up, wp, vp) =
        rw::rotating_wave_history(&wave, fs).map_err(PyValueError::new_err)?;
    Ok((
        PyArray1::from_vec(py, u0).into_any().unbind(),
        PyArray1::from_vec(py, w0).into_any().unbind(),
        PyArray1::from_vec(py, v0).into_any().unbind(),
        PyArray1::from_vec(py, up).into_any().unbind(),
        PyArray1::from_vec(py, wp).into_any().unbind(),
        PyArray1::from_vec(py, vp).into_any().unbind(),
    ))
}

/// `(H_pp, H_pz, H_zz)`: the Hessian of `V_nl` on the planar strain slice, per cell.
#[pyfunction]
#[pyo3(name = "rotating_wave_planar_hessian_cells")]
pub fn py_planar_hessian_cells(
    py: Python<'_>,
    p: &Bound<'_, PyAny>,
    z: &Bound<'_, PyAny>,
    a: f64,
) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>)> {
    let (_, pv) = as_f64_field(py, p, "p")?;
    let (_, zv) = as_f64_field(py, z, "z")?;
    if pv.len() != zv.len() {
        return Err(PyValueError::new_err(
            "p and z must have the same length (one entry per cell).",
        ));
    }
    let (h_pp, h_pz, h_zz) = rw::planar_hessian_cells(&pv, &zv, a);
    Ok((
        PyArray1::from_vec(py, h_pp).into_any().unbind(),
        PyArray1::from_vec(py, h_pz).into_any().unbind(),
        PyArray1::from_vec(py, h_zz).into_any().unbind(),
    ))
}

/// Kirchhoff–Carrier's `Omega = sqrt(omega0^2 + eps R^2)` for a **circular** mode (rad/s).
#[pyfunction]
#[pyo3(name = "rotating_wave_kc_circular_frequency")]
pub fn py_kc_circular_frequency(omega0_sq: f64, eps: f64, amplitude: f64) -> PyResult<f64> {
    rw::kc_circular_frequency(omega0_sq, eps, amplitude).map_err(PyValueError::new_err)
}
