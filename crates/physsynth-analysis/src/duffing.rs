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
//! `A cos(ω₀t)` — which `tests/oracles.rs` compares against at `atol = 1e-14` (carried from the
//! retired `tests/test_tension_string.py`, retirement plan §34).
//!
//! The Python module's docstrings were the physics, and this header documented only what the
//! translation had to decide. The Python bodies went at deletion 7 (`b1d37c8`) and the package at
//! retirement plan §48, so the derivations are carried here — the module docstring under "The
//! physics", each function's on the item of the same name — verbatim from `b1d37c8^`, save for
//! cross-references respelled natively and SciPy plumbing dated (retirement plan §53).
//!
//! # The physics
//!
//! Duffing oracles for the tension-modulated string (model #9, Kirchhoff-Carrier).
//!
//! The Kirchhoff-Carrier string carries a tension that depends on the *stretch* — a scalar
//! functional of the whole state:
//!
//! ```text
//! rho u_tt = (T0 + (EA/2L) I) u_xx - rho kappa^2 u_xxxx,     I = integral of u_x^2 dx.
//! ```
//!
//! **Why this model has a closed-form oracle at all** (model #6, the other nonlinear model, has
//! none): `sin(m pi x / L)` is an *exact* eigenvector of both `delta_xx` and the simply-supported
//! biharmonic `(delta_xx)^2`, and the stretch `I` depends on the state **only through the modal
//! amplitude squared**. So a single-mode state maps to a single-mode state, and the whole PDE —
//! discrete grid and all — collapses **exactly** onto a 1-DOF Duffing oscillator:
//!
//! ```text
//! q'' + omega0^2 q + eps q^3 = 0,    omega0^2 = c^2 p^2 + kappa^2 p^4,   eps = (EA/4 rho) p^4
//! ```
//!
//! with the **discrete** eigenvalue `p^2` ([`crate::damping::spatial_eigenvalue_p2`]) when
//! validating the scheme, or the continuum `beta^2 = (m pi/L)^2` for the continuum limit.
//!
//! **Scope of the reduction (do not over-read it).** It is exact *in exact arithmetic*, and exact
//! per step at any amplitude. It is **not** a claim that a real single-mode motion persists: above
//! a pump threshold (`dT/T0 ~ 3`) that motion is **parametrically unstable**, roundoff-seeded
//! perturbations in neighbouring modes grow exponentially, and the state stops being single-mode —
//! so these oracles stop applying, even though the scheme's energy conservation does not care (see
//! `physsynth_core::string_nonlinear`). Use them sub-threshold, or over short runs.
//!
//! The hardening Duffing oscillator has an **exact** solution `q(t) = A cn(Omega t, m)` in Jacobi
//! elliptic functions, hence an exact frequency in complete elliptic integrals —
//! [`duffing_frequency`]. That is the closed-form nonlinear oracle for model #9.
//!
//! **How to use it honestly** (the trap): a frequency measured from a simulation at one sample
//! rate also carries the linear theta-scheme's *temporal* dispersion error, which will dominate
//! the nonlinear shift being measured. Do **not** compare a single-`fs` measurement to
//! [`duffing_frequency`] and expect cents-level agreement. Either refine (`O(h^2)`), or tighter,
//! compare the **amplitude shift** `omega(A) - omega(A -> 0)`, where the common linear dispersion
//! error cancels. [`duffing_frequency_shift`] is that quantity.
//!
//! Independent of the core (an oracle must not depend on the code it validates; in the Python,
//! pure NumPy/SciPy). See `docs/dev/tension-modulated-string-plan.md`.

use crate::elliptic::{ellipj, ellipk};

/// Duffing coefficients `(omega0^2, eps)` of a single Kirchhoff-Carrier mode.
///
/// Substituting `u = q(t) s` (`s` the mode shape, `-delta_xx s = p^2 s`) into the KC string gives
/// `q'' = -(c^2 p^2 + kappa^2 p^4) q - (EA/4 rho) p^4 q^3` — exactly a Duffing oscillator. The
/// cubic coefficient is where the `L` cancels: the stretch of a unit-amplitude mode is
/// `I = p^2 L / 2` ([`kc_mode_stretch`]), and the tension term `(EA/2L) I` eats the `L`.
///
/// * `c`, `kappa`, `rho` — wave speed (m/s), stiffness `sqrt(EI/rho)` (m^2/s), linear density
///   (kg/m).
/// * `ea` — axial stiffness (N). `0` -> `eps = 0` -> the linear stiff string.
/// * `p2` — the mode's eigenvalue `p^2` of `-delta_xx`. Pass the **discrete**
///   `(4/h^2) sin^2(m pi / 2N)` to validate a scheme, or the continuum `(m pi / L)^2` for the
///   continuum limit.
/// * `l` — length (m). Accepted for signature symmetry and validation only — it cancels
///   analytically.
///
/// Returns `omega0^2` (rad^2/s^2), the linear angular frequency squared; and `eps`
/// (1/(m^2 s^2)), the cubic (hardening) coefficient. `eps >= 0` always: transverse motion can
/// only *stretch* a string, so tension only ever rises.
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

/// Stretch `I = integral u_x^2 dx` of a single mode of amplitude `q`: `I = q^2 p^2 L / 2`.
///
/// Exact both in the continuum (`integral cos^2 = L/2`) and on the discrete grid: for a discrete
/// sine `s_j = sin(m pi j / N)`, summation-by-parts gives `I = h ||delta_x+ u||^2 =
/// -h <D2 u, u> = p^2 h sum s_j^2 q^2` and `h sum_{j=1..N-1} s_j^2 = h N / 2 = L / 2` exactly.
///
/// This is the identity that makes the tension a function of `q^2` alone — the reason the
/// single-mode ansatz survives the nonlinearity.
pub fn kc_mode_stretch(amplitude: f64, p2: f64, l: f64) -> f64 {
    amplitude.powi(2) * p2 * l / 2.0
}

/// Elliptic parameter `m = eps A^2 / (2 (omega0^2 + eps A^2))` of the exact `cn` solution.
///
/// For a **hardening** spring (`eps >= 0`) this lies in `[0, 1/2)` — strictly bounded away from
/// the `m -> 1` singularity of `K(m)`. So [`duffing_frequency`] is well-conditioned at *any*
/// amplitude, however large. (Softening springs, `eps < 0`, do run into that singularity — they
/// have a finite escape amplitude — but the KC string cannot soften.)
///
/// Note `m` is the *parameter* convention, `m = k²` (SciPy's `ellipk` took `m`, not the modulus
/// `k = sqrt(m)`, and [`crate::elliptic`] keeps that convention).
pub fn duffing_elliptic_parameter(amplitude: f64, omega0_sq: f64, eps: f64) -> Result<f64, String> {
    let denom = omega0_sq + eps * amplitude.powi(2);
    if denom <= 0.0 {
        return Err(format!("omega0^2 + eps A^2 must be positive, got {denom}"));
    }
    Ok(eps * amplitude.powi(2) / (2.0 * denom))
}

/// **Exact** angular frequency (rad/s) of `q'' + omega0^2 q + eps q^3 = 0`, `q(0)=A`.
///
/// The solution is `q(t) = A cn(Omega t, m)` with `Omega = sqrt(omega0^2 + eps A^2)` and `m` from
/// [`duffing_elliptic_parameter`]; the `cn` period is `4 K(m) / Omega`, so
///
/// ```text
/// omega(A) = pi sqrt(omega0^2 + eps A^2) / (2 K(m)).
/// ```
///
/// This is *exact*, not a perturbation result — valid at any amplitude (see
/// [`duffing_elliptic_parameter`]). As `eps -> 0`: `m -> 0`, `K(0) = pi/2`, and
/// `omega -> omega0` — the linear limit.
pub fn duffing_frequency(amplitude: f64, omega0_sq: f64, eps: f64) -> Result<f64, String> {
    if omega0_sq <= 0.0 {
        return Err(format!("omega0_sq must be positive, got {omega0_sq}"));
    }
    let m = duffing_elliptic_parameter(amplitude, omega0_sq, eps)?;
    Ok(std::f64::consts::PI * (omega0_sq + eps * amplitude.powi(2)).sqrt() / (2.0 * ellipk(m)))
}

/// Nonlinear **amplitude shift** `omega(A) - omega0` (rad/s) — the dispersion-cancelling one.
///
/// The quantity to test a scheme against. A simulated `omega(A)` carries the scheme's linear
/// dispersion error; so does the simulated `omega(A -> 0)`. Their **difference** cancels it to
/// leading order, isolating the nonlinear physics this model exists to produce. Always `>= 0`
/// (hardening).
pub fn duffing_frequency_shift(amplitude: f64, omega0_sq: f64, eps: f64) -> Result<f64, String> {
    Ok(duffing_frequency(amplitude, omega0_sq, eps)? - omega0_sq.sqrt())
}

/// **Exact** solution `q(t) = A cn(Omega t, m)` of `q'' + omega0^2 q + eps q^3 = 0`.
///
/// Initial conditions `q(0) = A`, `q'(0) = 0` (`cn(0) = 1`, and `cn` starts at a maximum).
/// `Omega = sqrt(omega0^2 + eps A^2)`; `m` from [`duffing_elliptic_parameter`].
///
/// The whole **waveform**, not just the frequency — a strictly stronger oracle, and the one a
/// convergence study wants: a scheme can land the right period with the wrong wave shape (the
/// Duffing `cn` is visibly non-sinusoidal at large `m`), and comparing displacement at a fixed
/// time catches both. `eps = 0` degenerates to `A cos(omega0 t)` (`cn(u, 0) = cos u`).
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

/// First-order small-amplitude frequency `omega ~ omega0 (1 + 3 eps A^2 / (8 omega0^2))`.
///
/// The textbook Lindstedt-Poincare / averaging result. Kept as an independent cross-check on
/// [`duffing_frequency`] (they must agree as `A -> 0`, and visibly diverge when
/// `eps A^2 / omega0^2` is no longer small) — **not** as a test oracle; the elliptic form is exact
/// and is what tests compare against.
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
