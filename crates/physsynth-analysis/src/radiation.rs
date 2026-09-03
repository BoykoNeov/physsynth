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

//! # A defect the port found, and the fix that followed — hurdles §14, closed 2026-09-03
//!
//! The small-`ka` branch exists because `1 - J1(2ka)/ka` is a genuine `0/0` as `ka -> 0`. It is
//! *also* catastrophic cancellation long before that, and the original threshold of `1e-8` sat
//! three decades below where the direct form becomes usable: just above it the subtraction removes
//! sixteen digits, and with SciPy's own `j1` doing the work the shipped Python was **544% wrong at
//! `ka = 1e-8`**, 2.3% at 1e-7, and did not reach 1e-6 accuracy until around `ka = 1e-5`.
//!
//! **How it was found is the reusable part.** A native bar asserted that the two branches meet at
//! the threshold. They did not, by a factor of six — and the disagreement was in the *Python* all
//! along. No parity test could have found it: a parity test compares two implementations of the
//! same mistake, and both sides were computing the same cancellation faithfully.
//!
//! The port shipped it unchanged first, because changing a physics number inside a porting batch is
//! not a port; the fix landed separately, on both sides in one commit, once the human had called it.
//!
//! **The fix.** Three terms of the bracket's own Taylor series, `(ka)^2/2 - (ka)^4/12 +
//! (ka)^6/144`, in Horner form, below `ka = 3e-2`. Measured against a 60-digit reference over
//! `ka` in `[1e-10, 10]`, the worst relative error of the whole function goes from **5.24** to
//! **6.7e-13**, and the branches agree to 7e-13 across the seam so there is no step.
//!
//! The threshold was measured rather than derived, and the two answers differ: an algebraic
//! crossover estimate for the *one-term* series said `ka ~ 7.2e-4` giving 8.6e-8, and the measured
//! optimum was `2e-4` giving 1.3e-8 — off by 3.6x in the threshold. Plan §36.2's "measure the
//! margin first" turns out to apply to fixes and not only to ports.
//!
//! **What the fix does and does not make exact, measured rather than assumed.** The series is
//! `+ - * /` only — no `powi`, no library call — so IEEE-754 pins it: over 3,000 values below the
//! cutoff the two languages are **bit-identical, 0 differing**, and the parity file asserts that as
//! equality. The *direct* branch is not and cannot be, because it runs through two different `J1`
//! implementations (Cephes on one side, Miller recurrence on the other): 1,444 of 3,000 values
//! above the cutoff differ, worst 9.8e-13 relative.
//!
//! That figure is larger than the ~1e-16 the two `J1`s themselves differ by, and the factor is the
//! point of the threshold. At `ka = 3e-2` the bracket is `4.5e-4`, so the subtraction still
//! amplifies a last bit by about 2,200x. Moving the cutoff *down* would hand more of the domain to
//! a branch that magnifies disagreement; moving it up would hand more to a truncated series. 3e-2
//! is where those two costs cross, and the first draft of this comment claimed the whole function
//! was bit-identical — it is not, and the measurement is what said so.

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

/// Below this `ka`, [`piston_radiation_resistance`] uses its series rather than `J1`.
///
/// Chosen by measurement against a 60-digit reference (hurdles §14): with the three-term series the
/// whole function's worst relative error over `ka` in `[1e-10, 10]` is 6.7e-13 here, against
/// 7.9e-13 at `2e-2`, 2.8e-12 at `4e-2` and 5.0e-12 at `1e-2`. It sits deliberately *past* the
/// direct form's own noisy region (~5e-12 around `ka = 3e-3 .. 1e-2`) rather than at the crossover,
/// which is what buys the last order of magnitude.
///
/// Must equal `PISTON_SERIES_CUTOFF_KA` in `physsynth/core/radiation.py`; a parity test compares
/// the two functions across the seam and would go red if they drifted apart.
pub const PISTON_SERIES_CUTOFF_KA: f64 = 3e-2;

/// Baffled circular-piston (half-space) **acoustic** radiation resistance, Pa*s/m^3.
///
/// `R_a(ka) = (rho0*c0/S)[1 - J1(2ka)/(ka)]` with `S = pi*a^2` and `k = omega/c0`. Below
/// [`PISTON_SERIES_CUTOFF_KA`] the bracket is evaluated as three terms of its own Taylor series in
/// Horner form, `(ka)^2 * (1/2 - (ka)^2 * (1/12 - (ka)^2/144))`, because the direct form is a `0/0`
/// at the origin and catastrophic cancellation for three decades above it — see the module header.
///
/// The spelling is deliberate: `+ - * /` and nothing else. `ka.powi(4)` would be repeated
/// multiplication here and `ka ** 4` a `pow` call in Python, and plan §12 measured those
/// disagreeing in 1,400 of 3,998 cases. Written this way, both languages round identically.
pub fn piston_radiation_resistance(omega: f64, radius: f64, rho0: f64, c0: f64) -> f64 {
    let ka = omega * radius / c0;
    let s = PI * radius * radius;
    let ka2 = ka * ka;
    let bracket = if ka < PISTON_SERIES_CUTOFF_KA {
        ka2 * (0.5 - ka2 * (1.0 / 12.0 - ka2 / 144.0))
    } else {
        1.0 - j1(2.0 * ka) / ka
    };
    rho0 * c0 / s * bracket
}
