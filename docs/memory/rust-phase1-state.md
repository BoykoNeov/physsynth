---
name: rust-phase1-state
description: "Phase 1 of the Rust migration is BUILT (2026-08-26) — the core's dependency list stays EMPTY (hand-rolled CSR), the assembled matrices come out BIT-IDENTICAL to SciPy including nnz/indices, and the one flag now swings five still-Python models at once"
metadata: 
  node_type: memory
  type: project
  originSessionId: badff900-5024-422d-8f19-15b06875c6a8
  modified: 2026-08-26T13:32:29.538Z
---

Phase 1 of the Rust migration is **built** (2026-08-26): all of `physsynth/core/operators.py` —
the four pointwise differences, `inner`/`norm2`, and the three sparse builders
(`second_difference_matrix`, `biharmonic_matrix`, `free_beam_stiffness`) — now has a Rust twin
behind `PHYSSYNTH_RS`. Follows [[rust-phase0-state]]; the order of work is [[rust-migration-state]].

**The blast radius changed character, and that is the phase's main structural fact.** Phase 0
swapped one model. `operators` is not a model — it is what five models are *built out of*.
`string_stiff`, `string_damped`, `string_nonlinear`, `string_geometric` and `beam` all do
`from .operators import ...`, so flipping the one flag runs five still-Python models on Rust-built
matrices with zero edits in any of them. The viewer does **not** import operators directly (checked,
not assumed) — its exposure is entirely through those five models.

**Measured:** 42 native `cargo test` (19 at end of Phase 0) · 70 new parity tests · the whole
suite **2,026 passed both ways** (flag on and off) · matrices bit-identical to SciPy in
`data`/`indices`/`indptr`/`nnz`. Wall clock 432.6 s flagged vs 397.9 s default — **one run each,
uncontrolled, and not evidence of anything**: nothing in this phase touched a timestepping inner
loop, since the matrices are assembled once at construction. [[ci-runner-variance]] applies.

**Four things worth not rediscovering:**

- **The dependency list stayed empty.** A hand-written ~200-line CSR type (`crates/physsynth-core/src/sparse.rs`)
  rather than `faer`/`nalgebra-sparse`, because Phase 1 only ever *constructs* matrices. The thing
  that should choose a sparse library is a **solver** constraint, and that does not exist until
  Phase 3/4 — taking a crate now would fix the interchange type before the requirement that picks it.
- **The matrices are bit-identical to SciPy**, `data`/`indices`/`indptr`/`nnz`, at eight grid sizes
  and on a non-power-of-two grid. That needed one thing to be right: products accumulate in
  **ascending order of the contracted index**, which is what SciPy's SMMP kernel does. Measured
  separately: for *these* structures ascending vs descending gives the same answer anyway, so the
  exactness is not balanced on the order alone.
- **SciPy's own output is not canonical, and we deliberately do not copy that.** `biharmonic_matrix`
  comes back with `has_sorted_indices == False` and columns in *descending* order (SMMP's output
  list is a stack); `free_beam_stiffness`, which goes through a transpose, comes back sorted. Rust
  is canonical in both cases and the parity test canonicalises the SciPy side. Reproducing the split
  would pin the port to a SciPy internal — a red gate on a point release, for a non-bug.
- **`h ** 4` is not `h*h*h*h`.** Python's `**` calls libm `pow` (correctly rounded); three chained
  multiplications round three times. Measured over `h = 1/N`, `N = 2..3999`: they disagree in
  **1400** of 3998 cases (`(h*h)*(h*h)`: 1934). So `delta_xxxx` says `powf(4.0)` in Rust, and it is
  the one kernel whose exactness rests on two libms agreeing rather than on IEEE-754 alone — hence a
  sweep of `N` in the parity test rather than a single size.

**A scar from writing the native bars:** `K @ 1 == 0` is **not** exact, and the first draft asserted
it was. `D2 @ 1` cancels exactly, but `K` is an *assembled* matrix — applying it sums a row of
already-rounded entries instead of re-deriving the cancellation. ~1e-15 relative, so the honest bar
is relative to what the same operator does to a field that genuinely bends.

**The success-condition trap repeated itself.** `tests/test_operators.py` is four tests and touches
none of `delta_xxxx` or the three builders. Naming it as the phase's gate would have been green and
meaningless — the same failure Phase 0's §5 correction records. Generalised: **a module's tests are
not in the file named after it, and neither are its clients'.** Look them up every phase.
