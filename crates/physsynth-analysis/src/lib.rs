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

//! # Where the shared numerics come from, and why `root` is a `#[path]` include
//!
//! `modal.rs` needs Brent's method — the free-free beam's frequency equation and the free circular
//! plate's determinant are both root-finds, and both are `scipy.optimize.brentq` in the original.
//! `physsynth-core/src/root.rs` already transcribes that function line for line from SciPy's
//! `brentq.c`, for the reed. There were three ways to reach it and the choice is worth recording,
//! because two of them are wrong in ways that are not obvious:
//!
//! * **Take `physsynth-core` as a Cargo dependency.** This is the one to refuse. It inverts the
//!   argument the crate split exists to make: the oracle would then be built out of the model
//!   crate's parts, and "an oracle that lived inside the thing it checks would be checking itself"
//!   stops being a property of the layout and becomes a promise about which parts got reused. It
//!   also goes red in `tests/deps.rs`, which is name-based over the resolve graph and does not care
//!   that the name is a workspace sibling — correctly, since the whole point of that list is that
//!   *any* edge is a reviewed edit.
//! * **Copy the file.** Two transcriptions of one C function, free to drift, with nothing in the
//!   repo noticing when they do. `tests/deps.rs` opens by defending a deliberate near-copy, so the
//!   precedent exists — but that file duplicates ~30 lines of query plumbing, and this would
//!   duplicate a numerical method whose whole justification is that it reproduces SciPy *exactly*.
//! * **Include the source.** One file, one copy, compiled into both crates. `cargo metadata` sees
//!   no edge because there is none: this is not a dependency, it is the same text.
//!
//! The third is what is here. The cost is that the coupling is invisible in `Cargo.toml` and lives
//! only in the attribute below, which is why it is written up rather than left as a line — a reader
//! looking for why `analysis` compiles a core file will look here first.
//!
//! The module is `pub` so the native tests can reach it, and it carries its own header explaining
//! why a Brent transcription and not a crate. Nothing else crosses the two crates.
#[path = "../../physsynth-core/src/root.rs"]
pub mod root;

pub mod bessel;
pub mod damping;
pub mod dispersion;
pub mod duffing;
pub mod elliptic;
pub mod modal;
pub mod spectrum;
