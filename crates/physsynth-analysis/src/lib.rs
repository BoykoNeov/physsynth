//! Measurement — the Rust edition of `physsynth/analysis`.
//!
//! The sibling of `physsynth-core`, and deliberately not part of it. The core *produces*
//! trajectories; this crate *measures* them, and the two are separated here for the same reason
//! they are separated in the Python package: an oracle that lived inside the thing it checks would
//! be checking itself. That separation has a sharper form in this migration than it does in the
//! original, and it is worth stating where someone will find it.
//!
//! # The second flag, and why one would not do
//!
//! `PHYSSYNTH_RS=1` swaps every model in `physsynth/core` for its Rust twin and runs the existing
//! Python suite against the result. That run is the migration's acceptance gate, and what makes it
//! worth anything is that the *instrument* does not move: a Rust string is measured by the same
//! Python detector, against the same analytic oracle, as the Python string was.
//!
//! Put this crate behind the same flag and that stops being true — the model and the ruler would
//! both be Rust, and a shared misreading would cancel. Plan §7 scheduled `analysis/` late for
//! exactly this reason and §35.3 re-planned the order without re-taking the argument. So the swap
//! here reads a *second* variable, `PHYSSYNTH_RS_ANALYSIS`, and the gate keeps its meaning:
//!
//! * `PHYSSYNTH_RS=1` alone — Rust models, Python instrument. The three-shard harness job.
//! * both set — Rust models, Rust instrument. One extra CI step over the detector's dependents,
//!   which is where this crate gets exercised through real clients rather than through fixtures.
//! * neither — the untouched baseline the acceptance numbers came from.
//!
//! Nothing in `physsynth/core` imports `physsynth/analysis` (checked, not assumed), so the two
//! flags are genuinely independent rather than merely written that way.

pub mod spectrum;
