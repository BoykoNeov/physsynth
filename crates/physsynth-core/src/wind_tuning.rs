//! Musical controls for the acoustic bore — `docs/dev/wind-mapping-plan.md`.
//!
//! The bore page takes a tube length and the bell's resistance as a ratio `R/Z0`; a musician sets
//! a pitch and a ring time. This module maps one onto the other. It is an optional helper:
//! [`crate::bore::Params::new`] and its raw arguments are untouched.
//!
//! # Exact at every partial, by a special case
//!
//! The mapping is for a closed–radiating tube at `lambda = 1` (the bore page's only setting), and
//! there it is exact, in pitch and in ring time, at **every** partial. At `lambda = 1` the
//! staggered scheme carries a travelling wave without dispersion; the closed half-cell reflects it
//! with `+1`; and the bell's centred 1×1 solve ([`crate::bore::radiate_node`]) reflects it with
//! exactly
//!
//! ```text
//! rho = (R - Z0) / (R + Z0)
//! ```
//!
//! at every frequency. So one round trip, `2N` steps, multiplies the whole state by `-|rho|`: mode
//! `m` sounds at `(2m - 1) c0 / 4L` and every mode rings for
//!
//! ```text
//! T60 = (2L / c0) ln(1000) / (-ln|rho|)
//! ```
//!
//! The plan's §2 measured both on the one-step map's eigenvalues to `1e-9`. Inverting:
//!
//! ```text
//! L = c0 / (4 f0),     R/Z0 = tanh(3 ln10 L / (c0 T60))
//! ```
//!
//! # The branch
//!
//! `rho` and `1/rho` have the same modulus, so a ring time has two bells: `R < Z0` (`rho < 0`, the
//! open-like end) and `R > Z0` (`rho > 0`). The second makes the bell a closed end — the tube then
//! sounds a whole series an octave up, even harmonics included — so it is not the instrument asked
//! for. `tanh < 1` picks `R < Z0` by construction.
//!
//! Without a bell (the ideal open end) the tube loses nothing and rings forever; the pitch alone is
//! mapped there, the human's call (plan §3).

use std::f64::consts::LN_10;

/// What [`tune_bore`] gives.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoreTuning {
    /// Tube length (m), `c0 / 4 f0`.
    pub l: f64,
    /// The bell's `R / Z0`, in `(0, 1)` — `None` when no ring time was asked for (the open end).
    pub r_ratio: Option<f64>,
}

/// Why a pitch or a ring time cannot be mapped.
#[derive(Debug, Clone, PartialEq)]
pub enum WindTuningError {
    /// An input was zero, negative or not finite. Names the input.
    NonPositive(&'static str),
}

impl std::fmt::Display for WindTuningError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WindTuningError::NonPositive(name) => {
                write!(f, "{name} must be positive and finite.")
            }
        }
    }
}

impl std::error::Error for WindTuningError {}

/// Map a pitch and, with a bell, a ring time onto a closed–radiating bore at `lambda = 1`.
///
/// `t60` is the seconds for every partial's amplitude to fall 60 dB; `None` maps the pitch alone.
///
/// # Errors
/// [`WindTuningError::NonPositive`] when `f0`, `c0` or a given `t60` is not positive and finite.
pub fn tune_bore(f0: f64, t60: Option<f64>, c0: f64) -> Result<BoreTuning, WindTuningError> {
    for (name, v) in [("f0", Some(f0)), ("c0", Some(c0)), ("t60", t60)] {
        if let Some(v) = v {
            if !(v > 0.0 && v.is_finite()) {
                return Err(WindTuningError::NonPositive(name));
            }
        }
    }
    let l = c0 / (4.0 * f0);
    Ok(BoreTuning {
        l,
        r_ratio: t60.map(|t| (3.0 * LN_10 * l / (c0 * t)).tanh()),
    })
}

/// Partial `m` (from 1) of a closed–open or closed–radiating bore at `lambda = 1`, in Hz.
pub fn bore_partial(m: usize, l: f64, c0: f64) -> f64 {
    (2 * m - 1) as f64 * c0 / (4.0 * l)
}

/// The ring time (s) every partial of a closed–radiating bore at `lambda = 1` has with the bell at
/// `R/Z0 = r_ratio` — [`tune_bore`]'s inverse on its branch.
///
/// `-ln|rho| = 2 atanh(x)` with `x = min(r_ratio, 1/r_ratio)`, so a bell past `Z0` is read on the
/// other branch: the same ring time, but a closed end and another series. Infinite at `r_ratio = 0`
/// (nothing leaves), zero at `r_ratio = 1` (the anechoic bell sends nothing back).
pub fn bore_t60(l: f64, r_ratio: f64, c0: f64) -> f64 {
    let x = r_ratio.min(1.0 / r_ratio);
    3.0 * LN_10 * l / (c0 * x.atanh())
}
