//! The Duffing oracle — `physsynth/analysis/duffing.py`, transcribed.
//!
//! The **exact** solution of `q'' + ω₀²q + εq³ = 0`, which is what makes the tension-modulated
//! string (model #9) and the geometrically exact string (model #10) testable against something
//! other than themselves. `duffing_displacement` returns the whole waveform rather than just the
//! frequency, and that is the point of it: a scheme can land the right period with the wrong wave
//! shape, and only comparing displacement at a fixed time catches both.
//!
//! The elliptic functions it stands on are in [`crate::elliptic`], which carries the accuracy
//! story. The short version: `ellipk` agrees with SciPy to 3.5e-16 relative, and `ellipj` is
//! bit-identical at small argument and drifts as `1e-15·|u|` — inherent to the problem, not to the
//! implementation, and six orders below the `O(h²)` errors the convergence tests that consume it
//! are measuring.
//!
//! One degenerate case is exact rather than approximate and a test pins it: at `ε = 0` the
//! parameter `m` is zero, `cn(u, 0) = cos u`, and `duffing_displacement` reduces to
//! `A cos(ω₀t)` — which `tests/test_tension_string.py` compares against at `atol = 1e-14`.

use crate::elliptic::{ellipj, ellipk};

/// `(ω₀², ε)` for one mode of a Kirchhoff–Carrier string from its physical parameters.
///
/// `ω₀² = c²p² + κ²p⁴` is the linear part, `ε = (EA/4ρ)p⁴` the cubic stiffening that tension
/// modulation produces. `p²` is the eigenvalue of `-δ_xx` on the mode.
pub fn kc_mode_coefficients(
    c: f64,
    kappa: f64,
    ea: f64,
    rho: f64,
    p2: f64,
    l: f64,
) -> Result<(f64, f64), String> {
    if rho.min(l) <= 0.0 {
        return Err("rho and L must be positive.".to_string());
    }
    if ea < 0.0 {
        return Err("EA (axial stiffness) must be >= 0.".to_string());
    }
    if p2 < 0.0 {
        return Err("p2 (eigenvalue of -delta_xx) must be >= 0.".to_string());
    }
    let omega0_sq = c.powi(2) * p2 + kappa.powi(2) * p2.powi(2);
    let eps = (ea / (4.0 * rho)) * p2.powi(2);
    Ok((omega0_sq, eps))
}

/// The modal stretch `A²p²L/2` a single mode at amplitude `A` produces.
pub fn kc_mode_stretch(amplitude: f64, p2: f64, l: f64) -> f64 {
    amplitude.powi(2) * p2 * l / 2.0
}

/// The elliptic **parameter** `m = εA²/(2(ω₀² + εA²))` — SciPy's convention, `m = k²`.
pub fn duffing_elliptic_parameter(amplitude: f64, omega0_sq: f64, eps: f64) -> Result<f64, String> {
    let denom = omega0_sq + eps * amplitude.powi(2);
    if denom <= 0.0 {
        return Err(format!("omega0^2 + eps A^2 must be positive, got {denom}"));
    }
    Ok(eps * amplitude.powi(2) / (2.0 * denom))
}

/// The exact nonlinear angular frequency `π√(ω₀² + εA²) / (2K(m))`.
pub fn duffing_frequency(amplitude: f64, omega0_sq: f64, eps: f64) -> Result<f64, String> {
    if omega0_sq <= 0.0 {
        return Err(format!("omega0_sq must be positive, got {omega0_sq}"));
    }
    let m = duffing_elliptic_parameter(amplitude, omega0_sq, eps)?;
    Ok(std::f64::consts::PI * (omega0_sq + eps * amplitude.powi(2)).sqrt() / (2.0 * ellipk(m)))
}

/// The amplitude-dependent frequency shift `ω(A) - ω₀`.
pub fn duffing_frequency_shift(amplitude: f64, omega0_sq: f64, eps: f64) -> Result<f64, String> {
    Ok(duffing_frequency(amplitude, omega0_sq, eps)? - omega0_sq.sqrt())
}

/// The exact displacement `q(t) = A·cn(Ωt, m)`, `Ω = √(ω₀² + εA²)`, `q(0) = A`, `q'(0) = 0`.
pub fn duffing_displacement(
    t: &[f64],
    amplitude: f64,
    omega0_sq: f64,
    eps: f64,
) -> Result<Vec<f64>, String> {
    if omega0_sq <= 0.0 {
        return Err(format!("omega0_sq must be positive, got {omega0_sq}"));
    }
    let m = duffing_elliptic_parameter(amplitude, omega0_sq, eps)?;
    let omega_big = (omega0_sq + eps * amplitude.powi(2)).sqrt();
    Ok(t.iter()
        .map(|&ti| {
            let (_sn, cn, _dn) = ellipj(omega_big * ti, m);
            amplitude * cn
        })
        .collect())
}

/// The first-order Lindstedt–Poincaré frequency `ω₀(1 + 3εA²/8ω₀²)` — a cross-check, not an oracle.
pub fn duffing_frequency_expansion(
    amplitude: f64,
    omega0_sq: f64,
    eps: f64,
) -> Result<f64, String> {
    if omega0_sq <= 0.0 {
        return Err(format!("omega0_sq must be positive, got {omega0_sq}"));
    }
    let omega0 = omega0_sq.sqrt();
    Ok(omega0 * (1.0 + 3.0 * eps * amplitude.powi(2) / (8.0 * omega0_sq)))
}
