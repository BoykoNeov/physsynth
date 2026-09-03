//! The binding over `physsynth_analysis`'s oracles — `modal`, `damping`, `dispersion`, `duffing`.
//!
//! Same shape as `spectrum.rs` next door and for the same reason: these are free functions with no
//! state, so plan §14.2's question ("does this value reach the next timestep?") is answered *no*
//! everywhere and there is no buffer-ownership problem to have. Everything returns a fresh array
//! the caller owns.
//!
//! # The one design decision here: elementwise shape stays in Python
//!
//! Fourteen of these functions are written in NumPy as "whatever you give me, elementwise" —
//! `np.asarray(Λ, dtype=float)` and then a ufunc, which returns a scalar for a scalar, a 1-D array
//! for a 1-D array and a 2-D array for a mesh. Reproducing that dispatch in Rust would mean
//! reproducing NumPy's broadcasting, and it would get the 0-d case wrong in the way plan §28
//! already recorded: `ascontiguousarray` promotes a 0-d array to shape `(1,)`, so a function that
//! should hand back a scalar hands back a one-element array and every caller that does arithmetic
//! with it keeps working — silently, with a different type.
//!
//! So the binding takes flat `f64` slices and returns flat `f64` vectors, and the swap footer in
//! `physsynth/analysis/modal.py` ravels on the way in and reshapes on the way out. The shape logic
//! is three lines of Python that are *the same three lines* for every one of the fourteen, which is
//! a much smaller surface than fourteen Rust signatures each guessing at a shape.
//!
//! `cents` is the one that takes two arrays, and it is broadcast on the Python side before it gets
//! here — `np.broadcast_arrays` first, then one flat pair. That keeps the whole expression
//! `1200·log₂(f/f_ref)` on this side rather than splitting the division out, which would have made
//! the port a port of half a function.

use numpy::PyArray1;
use physsynth_analysis::{damping, dispersion, duffing, modal};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::shape::as_f64_field;

/// Turn the analysis crate's `Result<T, String>` refusals into the `ValueError` Python expects.
fn val<T>(r: Result<T, String>) -> PyResult<T> {
    r.map_err(PyValueError::new_err)
}

/// A freshly allocated NumPy array the caller owns.
fn arr(py: Python<'_>, v: Vec<f64>) -> Py<PyAny> {
    PyArray1::from_vec(py, v).into_any().unbind()
}

/// Read a Python sequence of integers as `Vec<i64>` — mode numbers, not field data.
fn as_i64s(seq: &Bound<'_, PyAny>) -> PyResult<Vec<i64>> {
    seq.extract()
}

// -- modal: the scalar oracles ---------------------------------------------------------------

#[pyfunction]
#[pyo3(name = "modal_harmonic_frequencies")]
pub fn py_harmonic_frequencies(py: Python<'_>, c: f64, l: f64, n_partials: usize) -> Py<PyAny> {
    arr(py, modal::harmonic_frequencies(c, l, n_partials))
}

#[pyfunction]
#[pyo3(name = "modal_mode_shape")]
pub fn py_mode_shape(py: Python<'_>, x: &Bound<'_, PyAny>, l: f64, m: i64) -> PyResult<Py<PyAny>> {
    let (_, xs) = as_f64_field(py, x, "x")?;
    Ok(arr(py, modal::mode_shape(&xs, l, m)))
}

#[pyfunction]
#[pyo3(name = "modal_discrete_mode_frequency")]
pub fn py_discrete_mode_frequency(c: f64, l: f64, n: i64, lam: f64, m: i64) -> f64 {
    modal::discrete_mode_frequency(c, l, n, lam, m)
}

#[pyfunction]
#[pyo3(name = "modal_inharmonicity_b")]
pub fn py_inharmonicity_b(c: f64, l: f64, kappa: f64) -> f64 {
    modal::inharmonicity_b(c, l, kappa)
}

#[pyfunction]
#[pyo3(name = "modal_stiff_harmonic_frequencies")]
pub fn py_stiff_harmonic_frequencies(
    py: Python<'_>,
    c: f64,
    l: f64,
    kappa: f64,
    n_partials: usize,
) -> Py<PyAny> {
    arr(
        py,
        modal::stiff_harmonic_frequencies(c, l, kappa, n_partials),
    )
}

#[pyfunction]
#[pyo3(name = "modal_discrete_stiff_mode_frequency")]
#[allow(clippy::too_many_arguments)]
pub fn py_discrete_stiff_mode_frequency(
    c: f64,
    l: f64,
    n: i64,
    kappa: f64,
    k: f64,
    m: i64,
    theta: f64,
) -> f64 {
    modal::discrete_stiff_mode_frequency(c, l, n, kappa, k, m, theta)
}

/// `1200·log₂(f/f_ref)` over an already-broadcast pair of flat arrays.
#[pyfunction]
#[pyo3(name = "modal_cents")]
pub fn py_cents(
    py: Python<'_>,
    f: &Bound<'_, PyAny>,
    f_ref: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let (_, a) = as_f64_field(py, f, "f")?;
    let (_, b) = as_f64_field(py, f_ref, "f_ref")?;
    Ok(arr(
        py,
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| modal::cents(x, y))
            .collect(),
    ))
}

#[pyfunction]
#[pyo3(name = "modal_rectangular_membrane_freqs")]
pub fn py_rectangular_membrane_freqs(
    py: Python<'_>,
    c: f64,
    lx: f64,
    ly: f64,
    modes: Vec<(i64, i64)>,
) -> Py<PyAny> {
    arr(py, modal::rectangular_membrane_freqs(c, lx, ly, &modes))
}

#[pyfunction]
#[pyo3(name = "modal_rectangular_mode_field")]
pub fn py_rectangular_mode_field(
    py: Python<'_>,
    x: &Bound<'_, PyAny>,
    y: &Bound<'_, PyAny>,
    lx: f64,
    ly: f64,
    m: i64,
    n: i64,
) -> PyResult<Py<PyAny>> {
    let (_, xs) = as_f64_field(py, x, "X")?;
    let (_, ys) = as_f64_field(py, y, "Y")?;
    Ok(arr(
        py,
        modal::rectangular_mode_field(&xs, &ys, lx, ly, m, n),
    ))
}

#[pyfunction]
#[pyo3(name = "modal_rectangular_discrete_eigenvalues")]
pub fn py_rectangular_discrete_eigenvalues(
    py: Python<'_>,
    h: f64,
    nx: i64,
    ny: i64,
    modes: Vec<(i64, i64)>,
) -> Py<PyAny> {
    arr(
        py,
        modal::rectangular_discrete_eigenvalues(h, nx, ny, &modes),
    )
}

/// `(m, n, frequency, degeneracy)` per mode — a list of tuples, as the original returns.
#[pyfunction]
#[pyo3(name = "modal_circular_membrane_freqs")]
pub fn py_circular_membrane_freqs(
    c: f64,
    a: f64,
    n_modes: usize,
    m_max: u32,
    n_max: usize,
) -> Vec<(u32, usize, f64, u32)> {
    modal::circular_membrane_freqs(c, a, n_modes, m_max, n_max)
        .into_iter()
        .map(|e| (e.m, e.n, e.freq, e.degeneracy))
        .collect()
}

#[pyfunction]
#[pyo3(name = "modal_discrete_membrane_eigenfrequency")]
pub fn py_discrete_membrane_eigenfrequency(
    py: Python<'_>,
    lambda: &Bound<'_, PyAny>,
    c: f64,
    k: f64,
) -> PyResult<Py<PyAny>> {
    let (_, v) = as_f64_field(py, lambda, "Lambda")?;
    Ok(arr(
        py,
        v.iter()
            .map(|&x| modal::discrete_membrane_eigenfrequency(x, c, k))
            .collect(),
    ))
}

#[pyfunction]
#[pyo3(name = "modal_rectangular_plate_freqs")]
pub fn py_rectangular_plate_freqs(
    py: Python<'_>,
    kappa: f64,
    lx: f64,
    ly: f64,
    modes: Vec<(i64, i64)>,
) -> Py<PyAny> {
    arr(py, modal::rectangular_plate_freqs(kappa, lx, ly, &modes))
}

#[pyfunction]
#[pyo3(name = "modal_discrete_plate_eigenfrequency")]
pub fn py_discrete_plate_eigenfrequency(
    py: Python<'_>,
    lambda_lap: &Bound<'_, PyAny>,
    kappa: f64,
    k: f64,
    theta: f64,
) -> PyResult<Py<PyAny>> {
    let (_, v) = as_f64_field(py, lambda_lap, "Lambda_lap")?;
    Ok(arr(
        py,
        v.iter()
            .map(|&x| modal::discrete_plate_eigenfrequency(x, kappa, k, theta))
            .collect(),
    ))
}

#[pyfunction]
#[pyo3(name = "modal_orthotropic_plate_freqs")]
#[allow(clippy::too_many_arguments)]
pub fn py_orthotropic_plate_freqs(
    py: Python<'_>,
    kappa: f64,
    lx: f64,
    ly: f64,
    modes: Vec<(i64, i64)>,
    grain_x: f64,
    grain_cross: f64,
    grain_y: f64,
) -> PyResult<Py<PyAny>> {
    let v = val(modal::orthotropic_plate_freqs(
        kappa,
        lx,
        ly,
        &modes,
        grain_x,
        grain_cross,
        grain_y,
    ))?;
    Ok(arr(py, v))
}

#[pyfunction]
#[pyo3(name = "modal_discrete_orthotropic_plate_eigenfrequency")]
#[allow(clippy::too_many_arguments)]
pub fn py_discrete_orthotropic_plate_eigenfrequency(
    py: Python<'_>,
    lam_x: &Bound<'_, PyAny>,
    lam_y: &Bound<'_, PyAny>,
    kappa: f64,
    k: f64,
    theta: f64,
    grain_x: f64,
    grain_cross: f64,
    grain_y: f64,
) -> PyResult<Py<PyAny>> {
    let (_, xs) = as_f64_field(py, lam_x, "lam_x")?;
    let (_, ys) = as_f64_field(py, lam_y, "lam_y")?;
    let mut out = Vec::with_capacity(xs.len());
    for (&x, &y) in xs.iter().zip(ys.iter()) {
        out.push(val(modal::discrete_orthotropic_plate_eigenfrequency(
            x,
            y,
            kappa,
            k,
            theta,
            grain_x,
            grain_cross,
            grain_y,
        ))?);
    }
    Ok(arr(py, out))
}

#[pyfunction]
#[pyo3(name = "modal_dirichlet_axis_eigenvalue")]
pub fn py_dirichlet_axis_eigenvalue(
    py: Python<'_>,
    m: &Bound<'_, PyAny>,
    l: f64,
    h: f64,
) -> PyResult<Py<PyAny>> {
    let (_, v) = as_f64_field(py, m, "m")?;
    Ok(arr(
        py,
        v.iter()
            .map(|&x| modal::dirichlet_axis_eigenvalue(x, l, h))
            .collect(),
    ))
}

#[pyfunction]
#[pyo3(name = "modal_free_free_beam_beta_l")]
pub fn py_free_free_beam_beta_l(py: Python<'_>, n_modes: i64) -> PyResult<Py<PyAny>> {
    // A negative `n_modes` is `n_modes < 1`, which the original refuses — taking it as `usize`
    // here would raise a TypeError instead and change which exception a caller sees.
    if n_modes < 1 {
        return Err(PyValueError::new_err("n_modes must be >= 1."));
    }
    Ok(arr(
        py,
        val(modal::free_free_beam_beta_l(n_modes as usize))?,
    ))
}

#[pyfunction]
#[pyo3(name = "modal_free_free_beam_freqs")]
pub fn py_free_free_beam_freqs(
    py: Python<'_>,
    kappa: f64,
    l: f64,
    n_modes: i64,
) -> PyResult<Py<PyAny>> {
    if n_modes < 1 {
        return Err(PyValueError::new_err("n_modes must be >= 1."));
    }
    Ok(arr(
        py,
        val(modal::free_free_beam_freqs(kappa, l, n_modes as usize))?,
    ))
}

#[pyfunction]
#[pyo3(name = "modal_free_plate_ffff_square_lambdas")]
pub fn py_free_plate_ffff_square_lambdas(py: Python<'_>) -> Py<PyAny> {
    arr(py, modal::free_plate_ffff_square_lambdas())
}

#[pyfunction]
#[pyo3(name = "modal_free_plate_freq_from_lambda")]
pub fn py_free_plate_freq_from_lambda(
    py: Python<'_>,
    lam: &Bound<'_, PyAny>,
    kappa: f64,
    a: f64,
) -> PyResult<Py<PyAny>> {
    let (_, v) = as_f64_field(py, lam, "lam")?;
    Ok(arr(
        py,
        v.iter()
            .map(|&x| modal::free_plate_freq_from_lambda(x, kappa, a))
            .collect(),
    ))
}

#[pyfunction]
#[pyo3(name = "modal_free_plate_twist_bound")]
pub fn py_free_plate_twist_bound(kappa: f64, a: f64, b: f64, grain_torsion: f64) -> PyResult<f64> {
    val(modal::free_plate_twist_bound(kappa, a, b, grain_torsion))
}

#[pyfunction]
#[pyo3(name = "modal_free_circular_plate_lambda_roots")]
pub fn py_free_circular_plate_lambda_roots(
    py: Python<'_>,
    nu: f64,
    n: i32,
    lam_max: f64,
    scan: usize,
) -> PyResult<Py<PyAny>> {
    Ok(arr(
        py,
        val(modal::free_circular_plate_lambda_roots(
            nu, n, lam_max, scan,
        ))?,
    ))
}

#[pyfunction]
#[pyo3(name = "modal_free_circular_plate_lambdas")]
pub fn py_free_circular_plate_lambdas(
    py: Python<'_>,
    nu: f64,
    n_modes: i64,
    n_max: u32,
) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
    if n_modes < 1 {
        return Err(PyValueError::new_err(format!(
            "n_modes must be >= 1, got {n_modes}"
        )));
    }
    let (lam, nodal) = val(modal::free_circular_plate_lambdas(
        nu,
        n_modes as usize,
        n_max,
    ))?;
    Ok((
        arr(py, lam),
        PyArray1::from_vec(py, nodal).into_any().unbind(),
    ))
}

#[pyfunction]
#[pyo3(name = "modal_free_circular_plate_saddle_bound")]
pub fn py_free_circular_plate_saddle_bound(nu: f64) -> PyResult<f64> {
    val(modal::free_circular_plate_saddle_bound(nu))
}

#[pyfunction]
#[pyo3(name = "modal_free_plate_coupling_form")]
pub fn py_free_plate_coupling_form(grain_coupling: f64, h: f64, nx: i64, ny: i64) -> PyResult<f64> {
    val(modal::free_plate_coupling_form(grain_coupling, h, nx, ny))
}

#[pyfunction]
#[pyo3(name = "modal_bore_resonance_frequencies")]
pub fn py_bore_resonance_frequencies(
    py: Python<'_>,
    c0: f64,
    l: f64,
    n_partials: usize,
    boundary: &str,
) -> PyResult<Py<PyAny>> {
    Ok(arr(
        py,
        val(modal::bore_resonance_frequencies(
            c0, l, n_partials, boundary,
        ))?,
    ))
}

#[pyfunction]
#[pyo3(name = "modal_discrete_bore_eigenfrequency")]
pub fn py_discrete_bore_eigenfrequency(
    py: Python<'_>,
    omega2: &Bound<'_, PyAny>,
    k: f64,
) -> PyResult<Py<PyAny>> {
    let (_, v) = as_f64_field(py, omega2, "omega2")?;
    Ok(arr(
        py,
        v.iter()
            .map(|&x| modal::discrete_bore_eigenfrequency(x, k))
            .collect(),
    ))
}

#[pyfunction]
#[pyo3(name = "modal_discrete_beam_eigenfrequency")]
pub fn py_discrete_beam_eigenfrequency(
    py: Python<'_>,
    mu: &Bound<'_, PyAny>,
    kappa: f64,
    k: f64,
    theta: f64,
) -> PyResult<Py<PyAny>> {
    let (_, v) = as_f64_field(py, mu, "mu")?;
    Ok(arr(
        py,
        v.iter()
            .map(|&x| modal::discrete_beam_eigenfrequency(x, kappa, k, theta))
            .collect(),
    ))
}

// -- damping ---------------------------------------------------------------------------------

#[pyfunction]
#[pyo3(name = "damping_spatial_eigenvalue_p2")]
pub fn py_spatial_eigenvalue_p2(n: i64, h: f64, m: i64) -> f64 {
    damping::spatial_eigenvalue_p2(n, h, m)
}

#[pyfunction]
#[pyo3(name = "damping_modal_loss_rate_continuum")]
pub fn py_modal_loss_rate_continuum(
    c: f64,
    l: f64,
    kappa: f64,
    sigma0: f64,
    sigma1: f64,
    m: i64,
) -> f64 {
    damping::modal_loss_rate_continuum(c, l, kappa, sigma0, sigma1, m)
}

#[pyfunction]
#[pyo3(name = "damping_discrete_damped_mode_decay")]
#[allow(clippy::too_many_arguments)]
pub fn py_discrete_damped_mode_decay(
    c: f64,
    l: f64,
    n: i64,
    kappa: f64,
    k: f64,
    theta: f64,
    sigma0: f64,
    sigma1: f64,
    m: i64,
) -> f64 {
    damping::discrete_damped_mode_decay(c, l, n, kappa, k, theta, sigma0, sigma1, m)
}

#[pyfunction]
#[pyo3(name = "damping_discrete_damped_mode_rate")]
#[allow(clippy::too_many_arguments)]
pub fn py_discrete_damped_mode_rate(
    c: f64,
    l: f64,
    n: i64,
    kappa: f64,
    k: f64,
    theta: f64,
    sigma0: f64,
    sigma1: f64,
    m: i64,
) -> f64 {
    damping::discrete_damped_mode_rate(c, l, n, kappa, k, theta, sigma0, sigma1, m)
}

#[pyfunction]
#[pyo3(name = "damping_discrete_damped_mode_is_underdamped")]
#[allow(clippy::too_many_arguments)]
pub fn py_discrete_damped_mode_is_underdamped(
    c: f64,
    l: f64,
    n: i64,
    kappa: f64,
    k: f64,
    theta: f64,
    sigma0: f64,
    sigma1: f64,
    m: i64,
) -> bool {
    damping::discrete_damped_mode_is_underdamped(c, l, n, kappa, k, theta, sigma0, sigma1, m)
}

#[pyfunction]
#[pyo3(name = "damping_loss_coefficients_from_t60")]
#[allow(clippy::too_many_arguments)]
pub fn py_loss_coefficients_from_t60(
    c: f64,
    l: f64,
    kappa: f64,
    f1: f64,
    t60_1: f64,
    f2: f64,
    t60_2: f64,
) -> PyResult<(f64, f64)> {
    val(damping::loss_coefficients_from_t60(
        c, l, kappa, f1, t60_1, f2, t60_2,
    ))
}

// -- dispersion ------------------------------------------------------------------------------

#[pyfunction]
#[pyo3(name = "dispersion_frequencies")]
pub fn py_dispersion_frequencies(
    py: Python<'_>,
    c: f64,
    l: f64,
    n: i64,
    lam: f64,
    modes: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let m = as_i64s(modes)?;
    Ok(arr(
        py,
        dispersion::dispersion_frequencies(c, l, n, lam, &m),
    ))
}

#[pyfunction]
#[pyo3(name = "dispersion_stiff_frequencies")]
#[allow(clippy::too_many_arguments)]
pub fn py_stiff_dispersion_frequencies(
    py: Python<'_>,
    c: f64,
    l: f64,
    n: i64,
    kappa: f64,
    k: f64,
    theta: f64,
    modes: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let m = as_i64s(modes)?;
    Ok(arr(
        py,
        dispersion::stiff_dispersion_frequencies(c, l, n, kappa, k, theta, &m),
    ))
}

#[pyfunction]
#[pyo3(name = "dispersion_phase_velocity")]
pub fn py_phase_velocity(
    py: Python<'_>,
    f: &Bound<'_, PyAny>,
    l: f64,
    modes: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let (_, fs) = as_f64_field(py, f, "f")?;
    let m = as_i64s(modes)?;
    Ok(arr(py, dispersion::phase_velocity(&fs, l, &m)))
}

// -- duffing ---------------------------------------------------------------------------------

#[pyfunction]
#[pyo3(name = "duffing_kc_mode_coefficients")]
pub fn py_kc_mode_coefficients(
    c: f64,
    kappa: f64,
    ea: f64,
    rho: f64,
    p2: f64,
    l: f64,
) -> PyResult<(f64, f64)> {
    val(duffing::kc_mode_coefficients(c, kappa, ea, rho, p2, l))
}

#[pyfunction]
#[pyo3(name = "duffing_kc_mode_stretch")]
pub fn py_kc_mode_stretch(amplitude: f64, p2: f64, l: f64) -> f64 {
    duffing::kc_mode_stretch(amplitude, p2, l)
}

#[pyfunction]
#[pyo3(name = "duffing_elliptic_parameter")]
pub fn py_duffing_elliptic_parameter(amplitude: f64, omega0_sq: f64, eps: f64) -> PyResult<f64> {
    val(duffing::duffing_elliptic_parameter(
        amplitude, omega0_sq, eps,
    ))
}

#[pyfunction]
#[pyo3(name = "duffing_frequency")]
pub fn py_duffing_frequency(amplitude: f64, omega0_sq: f64, eps: f64) -> PyResult<f64> {
    val(duffing::duffing_frequency(amplitude, omega0_sq, eps))
}

#[pyfunction]
#[pyo3(name = "duffing_frequency_shift")]
pub fn py_duffing_frequency_shift(amplitude: f64, omega0_sq: f64, eps: f64) -> PyResult<f64> {
    val(duffing::duffing_frequency_shift(amplitude, omega0_sq, eps))
}

#[pyfunction]
#[pyo3(name = "duffing_displacement")]
pub fn py_duffing_displacement(
    py: Python<'_>,
    t: &Bound<'_, PyAny>,
    amplitude: f64,
    omega0_sq: f64,
    eps: f64,
) -> PyResult<Py<PyAny>> {
    let (_, ts) = as_f64_field(py, t, "t")?;
    Ok(arr(
        py,
        val(duffing::duffing_displacement(
            &ts, amplitude, omega0_sq, eps,
        ))?,
    ))
}

#[pyfunction]
#[pyo3(name = "duffing_frequency_expansion")]
pub fn py_duffing_frequency_expansion(amplitude: f64, omega0_sq: f64, eps: f64) -> PyResult<f64> {
    val(duffing::duffing_frequency_expansion(
        amplitude, omega0_sq, eps,
    ))
}
