//! Headless DSP core — the Rust edition of `physsynth/core`.
//!
//! Same non-negotiable as the Python original (`CLAUDE.md` #4): no I/O, no graphics, no Python.
//! Here that rule is a *crate* property rather than a convention — this crate depends on nothing,
//! which `tests/deps.rs` checks against `cargo metadata`.
//!
//! **This crate is a migration in progress.** `docs/dev/rust-migration-plan.md` is the order of
//! work; only what a completed phase has ported lives here. Phase 0 is `string_ideal` plus the two
//! operators it calls; Phase 1 completes `ops` — the remaining pointwise differences and the three
//! sparse builders — and brings in `sparse`, a hand-written CSR type whose reason for existing
//! (rather than being a dependency) is written down in that module. Phase 2 begins the explicit
//! models: `exciter`, `membrane`, and `ops2d` — the *builder* half of `operators2d.py`, which the
//! plan files under Group D for a solver the membrane never calls (see that module's header).
//! Its second batch adds `body`, the smallest resonator in the project — and the one whose
//! clients write its state rather than only reading it; its third the wind leg, `bore` and
//! `reed`; its fourth `radiation`, the air node in three tiers, which is where the migration's
//! bit-identity claim runs out (a BLAS reduction that feeds back into state — see that module).
//! Phase 3 opens with `banded`, which is not a model either: it is the banded Cholesky four
//! theta-scheme string models share, ported ahead of all four because the suite chains them
//! together with bit-identity reduction anchors that only survive if they change solver at once.
//! Its second batch is `collision` — the contact primitives and both contact solves, plus `dense`,
//! the project's one dense LU — which is where the migration's last remaining bit-identity claim
//! is retired for a *model*: the vector solve's admittance matvec feeds back into the next Newton
//! iterate. Its scalar half, which contains no reduction at all, still matches to the bit. Its
//! third batch brings the first two models out of the theta-scheme string chain, `string_stiff`
//! and `string_damped` — and the thing that had to move before either could is not in this crate
//! at all: two *evaluation orders* SciPy chose and no portable implementation reproduces, one in a
//! reduction and one in a matrix's column order. Both were answered on the Python side, in
//! `physsynth/core/portable.py`; `sparse::Csr::sub` and `pyfloat` are what this side needed. Its
//! fourth batch is `string_nonlinear`, the first model here whose matrix changes every step: the
//! banded factor moves inside a scalar root-find, so `banded` and `root` meet for the first time
//! and a last bit in a reduction becomes a different *iteration count* rather than a different
//! last bit. Its fifth is `bow`, the project's first continuous nonlinear *exciter* — a shell over
//! machinery four earlier batches already ported, whose one new piece of arithmetic is a residual
//! the original spells twice, because the array path has a scalar hoisted out of it and the scalar
//! path does not.
//!
//! # The shape every resonator here shares
//!
//! Each model splits into three pieces, and the split is what lets the same physics serve both a
//! native Rust caller and the temporary Python binding without being written twice:
//!
//! - **Parameters** — a validated, immutable struct (`string_ideal::Params`). All the derived
//!   quantities (`c`, `h`, `k`, `lam`) and every construction-time rejection live here.
//! - **Kernels** — free functions over `&[f64]` slices. They own no state and allocate only what
//!   they return, so the caller decides where the buffers live. This is the piece the binding
//!   reaches for, because *its* buffers have to be Python objects (see `physsynth-py`).
//! - **A native owning struct** (`string_ideal::IdealString`) — parameters plus `Vec<f64>` state,
//!   for Rust callers and for `cargo test`. It is a thin buffer-management shell over the kernels.
//!
//! # Energy is the primary bug detector
//!
//! Every resonator exposes an `energy()` that is conserved to machine precision in a lossless run
//! and monotonically decreasing when lossy. That is the project's acceptance contract, and it is
//! asserted natively in `tests/` here as well as through the Python harness.

pub mod banded;
pub mod beam;
pub mod body;
pub mod bore;
pub mod bow;
pub mod collision;
pub mod dense;
pub mod exciter;
pub mod fmt;
pub mod mallet;
pub mod membrane;
pub mod ops;
pub mod ops2d;
pub mod pyfloat;
pub mod radiation;
pub mod reed;
pub mod root;
pub mod sparse;
pub mod sparse_lu;
pub mod string_damped;
pub mod string_ideal;
pub mod string_nonlinear;
pub mod string_stiff;
