---
name: rust-phase2-radiation-state
description: "Rust migration Phase 2 batch 4 (radiation) — bit-identity ends at a BLAS reduction that feeds back into state, not at a solver; and the test suite is blind to it because every fixture weight is 1.0"
metadata: 
  node_type: memory
  type: project
  originSessionId: 03ad9b9e-a456-46d6-a8eb-7bf51c6b448c
  modified: 2026-08-26T19:12:40.126Z
---

Phase 2 batch 4 of the Rust migration (2026-08-26): `physsynth/core/radiation.py` — the air node's
three tiers (far-field read-out, constant-`R` load, rational impedance) — ported to
`crates/physsynth-core/src/radiation.rs` + `crates/physsynth-py/src/radiation.rs`. Plan §14.
Continues [[rust-phase2-wind-state]]; the model it loads is [[rust-phase2-body-state]].

**The headline retires a prediction the first five batches were built on.** Batch 1 said the first
thing to break bit-identity would be "a solver, not a step"; §13.8 narrowed that to pivoting in
Phases 3-6. Both wrong. It broke here, in a Group A model with no matrix in it, at one line:
`u_free = float(np.dot(b.a, b.q - q_nm1)) / (2.0 * self.k)`. `body.pressure()` has had the
*identical* reduction since batch 2 and nobody minded, because it is a **read-out** — its last bit
reaches an assertion and stops. Here it decides `q^{n+1}`. So the question a model owes is not
"does it solve something" but **"does a reduction feed back into state?"**

It cannot be matched, and "use the same algorithm" is the obvious wrong answer. Measured: `np.dot`
on contiguous doubles is OpenBLAS, which **fuses the multiply-add** — a single-accumulator
sequential `fma` loop below 16 terms (1000/1000 agreement) and vectorised at 16+. `DYNAMIC_ARCH`
picks the kernel at run time, so a bit-identity assertion built on it passes here and fails on a
runner with a different CPU. **It would be a claim about a CPU, not about a port.**

**The part that changes how to read every future batch: the suite cannot see this.** A fused
multiply-add differs from a rounded one only when the **product** rounds. `tests/helpers.py`
builds every body with `phi=1.0` (so `a_i = 1`) and the parity suite's five-mode case uses
`[1, -0.5, 0.25, -0.125, 0.0625]` — all powers of two. Under both, the loaded state is
**bit-identical over 20,000 steps**, so the CI step is green with a divergence of exactly zero and
that number measures the *tests*, not the port. Only `radiation=0.02` makes it differ at all.
**Any batch comparing a reduction needs a fixture whose coefficients are not exactly
representable**, or its bit-identity result means less than it looks like it means.

**Group A is a SHORT-run bar, and this is the batch that needed the qualifier** (§4 already said
so; four batches never tested it). Weighted nine-mode body, state difference as a fraction of the
run's amplitude: 1.4e-14 at 2,000 steps, 3.4e-13 at 20,000. A fed-back reduction's error **grows
with run length** where a read-out's saturates. Both lengths are asserted, at their own bars.

**A metric trap next to it made the first measurement read 1e-7.** The body decays four orders of
magnitude, so an element-wise relative difference divides a frozen ~2e-17 absolute error by a
vanishing signal. **Normalise a decaying trajectory by its amplitude, never pointwise** — same for
read-outs (peak-normalised energy/pressure: 2.0e-15 / 5.4e-15).

Three non-arithmetic traps, each of which passes every physics bar this project owns:
- **`int(round(x))` is round-half-to-EVEN**; Rust's `f64::round` is half-away-from-zero. Only the
  delay line's length depends on it, and a one-sample error is invisible to energy, passivity,
  modal frequency and convergence order alike.
- **`np.isclose(a, b, rtol, atol=0.0)` is asymmetric** — the tolerance scales on the *second* arg.
- **Complex division is CPython's Smith's algorithm**, not `a conj(b)/|b|^2`; 20000/20000 match.

Two more worth keeping:
- **`piston_radiation_resistance` did not port** — it needs a Bessel `J1` (scipy's is Cephes), so
  the module ports in halves and that name waits for Phase 7, the analytic-oracle phase. The rule:
  a file's Bessel call drags *that function* to Phase 7, not the file. Same manoeuvre as
  `operators2d`'s solver half (§11.2.1).
- **A pyclass has no instance `__dict__`**, so an attribute *write* Python accepted silently would
  now raise. Grepped the clients (the [[rust-phase2-body-state]] `_accel` check, one batch on):
  eight private names are **read**, none written — so `_buf`/`_idx` could stay plain Rust `Vec`,
  the first buffer in this migration that did not have to cross by reference.
- **A transcribed refusal message contains a bug and was transcribed anyway.** `loaded_mode`'s
  `for/else` reports `abs(w_next - w) / w_next` after assigning `w = w_next`, so its "last
  relative step" is always exactly `0.000e+00`. Fixing it during a port would make the two sides
  incomparable; it is recorded as a fix owed to both at once.

`engine` is deliberately NOT in this batch and the reason is design, not scheduling: `simulate` is
the *loop*, not the step, so porting it adds four boundary crossings per iteration where the §11.6
speed win (per-step overhead, not arithmetic) does not apply — it would be **slower** until its
callees are Rust.
