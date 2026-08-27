//! Arithmetic on a *Python* float, where Python and Rust spell the same operation differently.
//!
//! Distinct from [`crate::fmt`], which is about how a float is *printed*. This is about what it
//! computes. There is one entry so far and it exists because of a compiler transformation rather
//! than a language difference.
//!
//! # `x ** 2` is not `x * x`, and only an opaque call keeps them apart
//!
//! CPython's `float.__pow__` is the C library's `pow` for every exponent, including `2.0`.
//! Measured 2026-08-27 over 400,000 samples from the range this project's constants occupy, `pow`
//! and a multiply disagree in **225** of them (§17.3). Nothing here is a physics difference — it
//! is a last bit — but the constants involved multiply a force or scale an operator at every
//! timestep, so writing `x * x` would put that last bit on the state of every step of every run
//! while conserving energy perfectly. No bar in this repo could catch it.
//!
//! LLVM rewrites `powf(x, 2.0)` into `x * x` whenever the exponent is a visible constant, which is
//! exactly the transformation that must not happen here — and it happens only in `--release`, so a
//! `cargo test` in debug will not show it (§17.2, which arrived as a red CI run). The
//! `#[inline(never)]` on [`scalar_pow`] is therefore load-bearing rather than stylistic: it keeps
//! the exponent a runtime value across the call boundary. Native tests that pin this must be run
//! in **both** profiles.
//!
//! Only exponent `2.0` is folded, measured on this machine: a literal `powf(x, 3.0)` or
//! `powf(x, 4.0)` reaches the real `pow` and needs nothing.

use crate::collision::PowPath;

/// `x ** e` as CPython's `float.__pow__` spells it: the C library's `pow`, for any exponent.
///
/// **The `#[inline(never)]` is the point of the function** — see the module header.
#[inline(never)]
pub fn scalar_pow(x: f64, e: f64) -> f64 {
    PowPath::Scalar.pow(x, e)
}
