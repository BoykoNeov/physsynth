---
name: rust-airbox-native-bars
description: "Unit 6's blocker was never real — the \"zero native bars\" audit read a directory, not the runner; plus two exactness claims of one identity that need different assertions"
metadata: 
  node_type: memory
  type: project
  originSessionId: 29392b97-3e94-434a-b224-fe2c3350d010
  modified: 2026-09-03T19:14:28.087Z
---

The deletion audit (plan §39.3) filed unit 6 (`airbox`) under **zero native bars** and §45.8 handed
that on as the last real blocker. It was wrong: `crates/physsynth-core/src/airbox.rs` and
`airbox_port.rs` carry **19 `#[test]`s in `mod tests` blocks inside `src/`**, which `cargo test
--workspace` runs exactly like the files under `tests/`. The audit's measurement was
`ls crates/physsynth-core/tests/` — a directory listing standing in for a question about the
runner. `beam.rs` really did have zero, so §45's blocker was real; the mistake is unit 6's alone.

The batch (plan §46, 2026-09-03) closed the gap instead of writing the bars from scratch: **18 new
tests**, 15 in `airbox.rs` and 3 in `airbox_port.rs`, `physsynth-core --lib` 53 → 71.

**Why:** the correction matters beyond one unit — a capability inventory has to ask what the test
*runner* runs, not what a directory holds, and the same class of mistake produced [[rust-deletion-beam-state]]'s
neighbouring finding that a module header is a claim nothing checks.

**How to apply:** three measured things worth carrying forward.

- **One algebraic identity, two exactness regimes.** The grid-diagonal mode at `λ = 1/√3` is exact
  by cancellation of an arcsine against a sine. On a *cube* it lands at 1.1e-16–3.3e-16 (it runs
  through `pow`, `sqrt`, `asin`, `sin`), so the bar is 1e-14; the **corner** mode of a *non-cube* is
  exactly `0.0`, because every axis reaches `sin(π/2)`. Assert each at what it is — the tolerance
  twice throws away a real exactness, the equality twice is a CPU-dependence bug waiting.
- **"Linear growth" is a claim about differences.** Windowed peaks normalised by the *seed* and
  asserted against `[1,2,3,4]` passed at 60–80% of threshold: secular growth is `a + b·i` and
  window 1 already contains some, so the integers state `a`, a property of the initial condition.
  Assert the differences are equal, plus a separate "and it did grow".
- **A wrapper is not the claim; the arithmetic it wraps is.** `R_room` was measured differentially
  off the room (0 to 2.2e-16 against a 1e-12 bar) with **no `RoomPort` object at all** — which is
  the class a `physsynth-core` test provably cannot reach. Native-writable is a question about the
  kernels, not about the tier's public class.

Next is the deletion itself — see [[rust-deletion-phase-state]] for the guard surface (~20 names for
this unit) and the `splu` / Rust-reads-Python-namespace constraints.
