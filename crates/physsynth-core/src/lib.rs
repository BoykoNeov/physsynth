//! Headless DSP core — the Rust edition of `physsynth/core`.
//!
//! Same non-negotiable as the Python original (`CLAUDE.md` #4): no I/O, no graphics, no Python.
//! Here that rule is a *crate* property rather than a convention — this crate depends on nothing,
//! which `tests/deps.rs` checks against `cargo metadata`.
//!
//! **This crate is a migration in progress.** `docs/dev/rust-migration-plan.md` is the order of
//! work; only what a completed phase has ported lives here. Phase 0 is `string_ideal` plus the two
//! operators it calls.
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

pub mod ops;
pub mod string_ideal;
