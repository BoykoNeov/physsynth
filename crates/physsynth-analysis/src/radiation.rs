//! The one radiation oracle that needed a Bessel function — plan §14's parked call, paid.
//!
//! `physsynth/core/radiation.py` ported in Phase 2 batch 4 with a hole in it: four classes and the
//! monopole helper swapped, and `piston_radiation_resistance` did not, because it is the only name
//! in that file needing a Bessel `J1` and the plan already had a phase for special functions. This
//! is that phase, and this is the hole closed. §14's own words: "a file's Bessel call does not drag
//! the file into Phase 7; it drags the call."
//!
//! # Why it lives in the *analysis* crate while its Python stays in `core/`
//!
//! Two constraints that look like they conflict and do not:
//!
//! * `physsynth-core`'s dependency list is empty and must stay empty (plan §2.2, the portability
//!   contract), so it cannot reach [`crate::bessel`] — a core→analysis edge would also be a cycle,
//!   since `modal` reaches core's Brent transcription the other way.
//! * `CLAUDE.md` is explicit that **no `core/` module goes behind the analysis flag**. So
//!   `piston_radiation_resistance` keeps swapping on `PHYSSYNTH_RS` with the rest of its file.
//!
//! Both hold at once because the crate a function is *implemented* in and the flag its Python name
//! is *swapped* by are different questions. `physsynth-py` depends on both crates, so the binding
//! can expose an analysis-crate function under a core-flagged name without either crate depending
//! on the other. That is the whole trick, and it is written here because the arrangement reads like
//! a flag violation until the sentence above is in front of you.
//!
//! # What makes swapping it under the model flag safe
//!
//! There is a real question underneath: `tests/test_bore_radiation.py` uses this resistance to
//! check a bore's reflection, so under `PHYSSYNTH_RS=1` a Rust bore would be checked against a
//! Rust-computed `R` — the shared-misreading shape the two-flag rule exists to prevent.
//!
//! It is safe here, and by measurement rather than by argument: the same flagged run executes
//! `test_radiation.py::test_piston_resistance_matches_bessel_formula_away_from_the_limit`, which
//! builds its expectation from `scipy.special.j1` **inside the test body** and requires agreement
//! to `rel = 1e-12` at `ka ≈ 1.8`. The ruler is therefore checked against an unmoved reference in
//! the very run that uses it. Observed agreement there: the transcription and Cephes differ by
//! 7.9e-16 relative at that argument, four orders inside the bar.

//! # A defect this function has always had, reproduced rather than fixed
//!
//! The `ka < 1e-8` branch below exists because `1 - J1(2ka)/ka` is a genuine `0/0` as `ka → 0`. The
//! guard is in the right place and its threshold is about three decades too small: just *above* it,
//! the direct form subtracts two numbers agreeing to sixteen digits, and with SciPy's own `j1`
//! doing the work the shipped Python is **544% wrong at `ka = 1e-8`**, 2.3% at 1e-7, and does not
//! reach 1e-6 accuracy until around `ka = 1e-5`.
//!
//! This transcription reproduces it, threshold and all, because changing a shipped physics number
//! inside a porting batch is not a port. It is registered in `docs/dev/scientific-hurdles.md` §14
//! with the two-term series and the crossover that would fix it, for the human's call. No caller is
//! in the band — the suite's two real call sites are at `ka = 9.2e-5` and `ka = 1.83`, where the two
//! implementations agree to 5.3e-8 and exactly.

use crate::bessel::j1;
use std::f64::consts::PI;

/// Ambient air density (kg/m³) and the speed of sound (m/s) — the same two numbers `core` uses.
///
/// Repeated rather than imported for the reason the module header gives: this crate cannot depend
/// on `physsynth-core`. They are literals in both places and a native bar asserts they agree, so
/// the duplication cannot drift silently.
pub const RHO0_AIR: f64 = 1.2041;
/// Speed of sound in air (m/s).
pub const C0_AIR: f64 = 343.0;

/// Baffled circular-piston (half-space) **acoustic** radiation resistance, Pa·s/m³.
///
/// `R_a(ka) = (ρ₀c₀/S)[1 - J₁(2ka)/(ka)]` with `S = πa²` and `k = ω/c₀`. As `ka → 0` the bracket
/// tends to `(ka)²/2`, which is a genuine `0/0` in the direct form, so the series is used below
/// `ka = 1e-8` — the same branch and the same threshold as the original, because a branch choice is
/// part of the trajectory (plan §17) even when both sides of it are smooth.
pub fn piston_radiation_resistance(omega: f64, radius: f64, rho0: f64, c0: f64) -> f64 {
    let ka = omega * radius / c0;
    let s = PI * radius * radius;
    let bracket = if ka < 1e-8 {
        0.5 * ka * ka
    } else {
        1.0 - j1(2.0 * ka) / ka
    };
    rho0 * c0 / s * bracket
}
