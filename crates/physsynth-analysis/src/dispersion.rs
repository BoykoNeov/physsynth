//! Numerical-dispersion curves — `physsynth/analysis/dispersion.py`, transcribed.
//!
//! Seventy-seven lines in the original and three functions, two of which are loops over
//! [`crate::modal`]. It is here as its own module rather than folded in for the same reason it is
//! its own file in Python: it answers a different question. `modal` says where mode `m` is;
//! `dispersion` asks how that answer *bends* as `m` climbs, which is the artifact a wave scheme
//! below `λ = 1` has and an exact one does not.
//!
//! The Python module's docstrings were the physics, and this header documented only what the
//! translation had to decide. The Python bodies went at deletion 7 (`b1d37c8`) and the package at
//! retirement plan §48, so the derivations are carried here — the module docstring under "The
//! physics", each function's on the item of the same name — verbatim from `b1d37c8^`, save for
//! cross-references respelled natively (retirement plan §53).
//!
//! # The physics
//!
//! Numerical-dispersion analysis for the explicit string scheme (HANDOFF §6 test 5, §7).
//!
//! The explicit scheme propagates spatial mode `m` at the discrete frequency
//!
//! ```text
//! f_m = arcsin( lambda * sin(m*pi/(2N)) ) / (pi*k),     k = lambda*L/(c*N)
//! ```
//!
//! (see [`crate::modal::discrete_mode_frequency`]). At `lambda = 1` this equals the continuum
//! `m*c/(2L)` for *every* mode — zero numerical dispersion, the d'Alembert-exact case. For
//! `lambda < 1` it lies below the continuum value and the gap widens with `m`: high partials
//! travel too slowly.
//!
//! HANDOFF §6.5 asks for this as a *phase velocity vs frequency* curve. A standing mode `m` is two
//! counter-propagating waves of wavenumber `k_m = m*pi/L` and angular frequency
//! `omega = 2*pi*f_m`, so its phase velocity is
//!
//! ```text
//! v_p(m) = omega / k_m = 2*L*f_m / m,
//! ```
//!
//! which equals `c` for the continuum and droops below `c` once `lambda < 1`. [`phase_velocity`]
//! returns it for measured *or* oracle frequencies; dividing by `c` gives the dimensionless curve
//! that is flat at `1` for `lambda = 1` and sags for `lambda < 1`.
//!
//! Independent of the core (an oracle must not depend on the code it validates; in the Python,
//! pure NumPy).
//!
//! # History: the import-time hazard the Python module was checked against
//!
//! While both languages ran, `dispersion.py` opened with an early `from .modal import …`, the
//! shape that defeats a swap done by rebinding module globals. It was safe because `modal.py`'s
//! swap ran inside `modal`'s own module body, so the names were already the Rust ones when copied,
//! and the retired `tests/test_rust_parity_analysis.py` pinned that. The flag went at retirement
//! plan §21 and the Python at §51; natively there is one implementation and nothing to swap.

use crate::modal::{discrete_mode_frequency, discrete_stiff_mode_frequency};

/// Discrete-scheme frequencies (Hz) for each mode in `modes` — the vectorised oracle.
///
/// A vectorised companion to [`crate::modal::discrete_mode_frequency`]; the closed form lives
/// there so there is a single source of truth for the dispersion relation.
pub fn dispersion_frequencies(c: f64, l: f64, n: i64, lam: f64, modes: &[i64]) -> Vec<f64> {
    modes
        .iter()
        .map(|&m| discrete_mode_frequency(c, l, n, lam, m))
        .collect()
}

/// Discrete stiff-string frequencies (Hz) for each mode — the vectorised stiff oracle.
///
/// Wraps [`crate::modal::discrete_stiff_mode_frequency`] (the single source of truth). Bending
/// adds a `kappa^2 p^4` term to `Q` that stiffens high partials, so the phase velocity rises with
/// mode (the partials sharpen) while numerical dispersion drags it the other way — the net is the
/// measured dispersion curve.
///
/// (The Python docstring opened that sentence with "Unlike the ideal case the curve droops
/// *more* steeply with mode number", which contradicts the rest of it, and is dropped here. The
/// rise is what is true and what `physsynth-core`'s `tests/string_stiff_harness.rs` asserts:
/// `v_p / c` above 1 and increasing with mode at `λ = 0.8`.)
pub fn stiff_dispersion_frequencies(
    c: f64,
    l: f64,
    n: i64,
    kappa: f64,
    k: f64,
    theta: f64,
    modes: &[i64],
) -> Vec<f64> {
    modes
        .iter()
        .map(|&m| discrete_stiff_mode_frequency(c, l, n, kappa, k, m, theta))
        .collect()
}

/// Modal phase velocity `v_p = 2*L*f/m` (m/s) for frequencies `f` of modes `modes`.
///
/// Accepts measured or analytic frequencies. For the continuum `f = m*c/(2L)` it returns `c` for
/// every mode; numerical dispersion (`lambda < 1`) makes it droop below `c` as `m` grows.
pub fn phase_velocity(f: &[f64], l: f64, modes: &[i64]) -> Vec<f64> {
    f.iter()
        .zip(modes.iter())
        .map(|(&fi, &m)| 2.0 * l * fi / m as f64)
        .collect()
}
