//! The binding over `physsynth_analysis::spectrum` — the partial detector.
//!
//! Four free functions and no state, so there is no buffer-ownership question anywhere here: this
//! module *consumes* a trajectory and returns fresh arrays the caller owns. That is what makes an
//! analysis module a different porting problem from a model — plan §14.2 asks of every ported
//! value "does this reach the next timestep?", and here the answer is no, everywhere. Nothing
//! computed below is ever fed back into a simulation.
//!
//! `parabolic_refine` is exposed under its private-looking name because that is what it is called
//! on the Python side and what `tests/test_spectrum_detector.py` imports directly. The guard inside
//! it is the reason this module was worth porting first (plan §35.3), so the test that exercises
//! the guard has to be able to reach the Rust one.

use numpy::PyArray1;
use physsynth_analysis::spectrum;
use pyo3::prelude::*;

use crate::shape::as_f64_field;

/// `(freqs, magnitude, nfft)` of the DC-removed, Hann-windowed signal.
#[pyfunction]
#[pyo3(name = "spectrum_magnitude_spectrum")]
#[pyo3(signature = (signal, fs, zero_pad_factor=2))]
pub fn py_magnitude_spectrum(
    py: Python<'_>,
    signal: &Bound<'_, PyAny>,
    fs: f64,
    zero_pad_factor: usize,
) -> PyResult<(Py<PyAny>, Py<PyAny>, usize)> {
    let (_, sig) = as_f64_field(py, signal, "signal")?;
    let out = spectrum::magnitude_spectrum(&sig, fs, zero_pad_factor);
    Ok((
        PyArray1::from_vec(py, out.freqs).into_any().unbind(),
        PyArray1::from_vec(py, out.mag).into_any().unbind(),
        out.nfft,
    ))
}

/// Sub-bin frequency (Hz) of the peak at bin `i`, or its bin centre if `i` is not a local maximum.
///
/// `i` is taken as a signed integer and bounds-checked here rather than in the core, because the
/// original's guard is `if i <= 0 or i >= len(mag) - 1` — a *negative* index is one of the cases it
/// declines, and a `usize` parameter would turn that into a Python-level TypeError instead of the
/// bin centre the caller is entitled to.
#[pyfunction]
#[pyo3(name = "spectrum_parabolic_refine")]
pub fn py_parabolic_refine(
    py: Python<'_>,
    mag: &Bound<'_, PyAny>,
    i: i64,
    fs: f64,
    nfft: usize,
) -> PyResult<f64> {
    let (_, m) = as_f64_field(py, mag, "mag")?;
    if i <= 0 || i as usize + 1 >= m.len() {
        return Ok(i as f64 * fs / nfft as f64);
    }
    Ok(spectrum::parabolic_refine(&m, i as usize, fs, nfft))
}

/// Measure the partial frequencies nearest each value in `expected`; NaN where the window is empty.
#[pyfunction]
#[pyo3(name = "spectrum_measure_partials_near")]
#[pyo3(signature = (signal, fs, expected, search_hz=None))]
pub fn py_measure_partials_near(
    py: Python<'_>,
    signal: &Bound<'_, PyAny>,
    fs: f64,
    expected: &Bound<'_, PyAny>,
    search_hz: Option<f64>,
) -> PyResult<Py<PyAny>> {
    let (_, sig) = as_f64_field(py, signal, "signal")?;
    let (_, exp) = as_f64_field(py, expected, "expected")?;
    let out = spectrum::measure_partials_near(&sig, fs, &exp, search_hz);
    Ok(PyArray1::from_vec(py, out).into_any().unbind())
}

/// Blindly detect the `n_peaks` strongest spectral peaks above `f_min`, ascending in Hz.
#[pyfunction]
#[pyo3(name = "spectrum_detect_peaks")]
#[pyo3(signature = (signal, fs, n_peaks, f_min=1.0, min_separation_hz=None))]
pub fn py_detect_peaks(
    py: Python<'_>,
    signal: &Bound<'_, PyAny>,
    fs: f64,
    n_peaks: usize,
    f_min: f64,
    min_separation_hz: Option<f64>,
) -> PyResult<Py<PyAny>> {
    let (_, sig) = as_f64_field(py, signal, "signal")?;
    let out = spectrum::detect_peaks(&sig, fs, n_peaks, f_min, min_separation_hz);
    Ok(PyArray1::from_vec(py, out).into_any().unbind())
}
