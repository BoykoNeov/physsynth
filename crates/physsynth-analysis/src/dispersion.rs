//! Numerical-dispersion curves — `physsynth/analysis/dispersion.py`, transcribed.
//!
//! Seventy-seven lines in the original and three functions, two of which are loops over
//! [`crate::modal`]. It is here as its own module rather than folded in for the same reason it is
//! its own file in Python: it answers a different question. `modal` says where mode `m` is;
//! `dispersion` asks how that answer *bends* as `m` climbs, which is the artifact a wave scheme
//! below `λ = 1` has and an exact one does not.
//!
//! # The import-time hazard this module was checked against
//!
//! `dispersion.py` opens `from .modal import discrete_mode_frequency, discrete_stiff_mode_frequency`
//! — an early binding, which is exactly the shape that defeats a swap done by rebinding module
//! globals. It does not defeat this one, and the reason is worth writing down because the
//! conclusion is "safe" and a reader will otherwise have to re-derive it: `modal.py`'s swap footer
//! runs inside `modal`'s own module body, so by the time `dispersion`'s `from` statement executes,
//! the names it copies are already the Rust ones. The hazard would be live only if the swap were a
//! monkeypatch applied after import — which is what §32.7 says never to do, and is why the
//! collaborators there are looked up as module globals deliberately. `tests/test_rust_parity_analysis.py`
//! pins it directly rather than leaving it as an argument.

use crate::modal::{discrete_mode_frequency, discrete_stiff_mode_frequency};

/// Discrete frequencies of the listed modes under the explicit scheme.
pub fn dispersion_frequencies(c: f64, l: f64, n: i64, lam: f64, modes: &[i64]) -> Vec<f64> {
    modes
        .iter()
        .map(|&m| discrete_mode_frequency(c, l, n, lam, m))
        .collect()
}

/// Discrete frequencies of the listed modes under the implicit θ-scheme stiff string.
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

/// Modal phase velocity `2Lf/m` — flat at `c` for a dispersionless scheme, drooping otherwise.
pub fn phase_velocity(f: &[f64], l: f64, modes: &[i64]) -> Vec<f64> {
    f.iter()
        .zip(modes.iter())
        .map(|(&fi, &m)| 2.0 * l * fi / m as f64)
        .collect()
}
