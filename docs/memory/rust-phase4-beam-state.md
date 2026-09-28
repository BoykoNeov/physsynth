---
name: rust-phase4-beam-state
description: "Rust migration Phase 4 (`beam`, 2026-08-28) — §4.1's SuperLU hypothesis TESTED AND FAILED, Group D runs on measured tolerance; a sixth agreement regime set by a BOUNDARY CONDITION; and the pivot obstacle was invisible on the two fixtures first tried"
metadata: 
  node_type: memory
  type: project
  originSessionId: 81239f80-7a00-4cb4-9b59-7e7d04269f5f
  modified: 2026-08-28T08:00:22.464Z
---

**PHASE 4 IS COMPLETE.** `beam` ported (`crates/physsynth-core/src/{beam,sparse_lu}.rs`,
`crates/physsynth-py/src/{beam,sparse_lu}.rs`, `tests/test_rust_parity_beam.py`). Plan §24.

**The batch existed to answer a question, and the answer is no.** §4.1 proposed linking SuperLU so
the six sparse-LU models could keep bit-identity. Measured on the beam's own matrix:

- **column ordering** — a closed form in `n` (identity except two pairs near the tail exchanged),
  verified at 17 grid sizes. Not a barrier, and not reproduced, because it buys nothing.
- **equilibration** — never runs. SciPy calls `_superlu.gstrf` (the factorization); equilibration
  lives in the `gssvx` **driver**. `Equil=True` and `Equil=False` give bit-identical factors.
- **pivot threshold** — **REAL, and only above `N = 48`.** The stiffness grows like `h⁻³` against
  the mass's `h`, so the matrix stops being diagonally largest as the grid refines; SuperLU pivots
  from `N = 64` and fills `U` (773 entries at `N = 200` vs a band of 600). The first probe used
  `N = 8` and `N = 32` and concluded "pivoting is moot" — [[rust-phase3-collision-state]]'s blind
  fixture arriving in the **measurement**. Rule: *parametrise over the grid before concluding
  anything about a solver.*
- **what actually decides it** — SuperLU is **supernodal**. Handed its *own* factors, a longhand
  triangular solve still differs in ~20% of entries at ~4e-16. That is a property of how SciPy was
  **built**, so linking buys a claim about a build — [[numpy-libm-cpu-dispatch]] one layer down.

The human's call (2026-08-28): **tolerance-level agreement, quantified.** No C dependency; the
Cargo dep list stays empty.

**The sixth agreement regime, and the first set by a BOUNDARY CONDITION.** Free-free means `K`'s
nullspace is exactly `{1, x}`, so along it the beam is a **free particle** and a per-step solver
difference integrates twice. At `N = 32` over 20,000 steps: rigid part 3.7e-17 → 3.3e-9 (`t²`),
elastic part random-walks to 1.4e-12, energy stays inside 7.2e-12. Damping attenuates without
removing (2.1e-10 at `σ = 100`). **So a parity bar on any free-edge model reads the rigid/elastic
split or the energy, never `max|du|/amp`** — inherited by the free plate at Phase 5.

**The port itself is exact, and proving it is a two-line patch.** `physsynth_rs.SparseLu` wears
`splu`'s interface, so the parity test drives the *Python* beam through the *Rust* factorization:
bit-identical over 2,000 steps at four fixtures, `energy()` differing only at the `np.dot`
reduction. Without that the whole comparison is confounded — a real bug would sit two orders under
the solver gap ([[rust-phase3-tension-state]]'s finding waiting to happen).

**Two bugs found in passing, both pre-existing and both invisible to every earlier test.**
1. **An omitted keyword and an explicit `None` were the same argument in all six bindings.** PyO3
   collapses them, so `boundary=None` silently built the *default* where Python raises. Fixed with
   `Option<Option<_>>` — and the arm order is inverted from the obvious guess (PyO3 wraps the
   **default expression**, so `Some(None)` = omitted and bare `None` = the caller's literal). The
   first attempt had them backwards and inverted the behaviour instead of fixing it.
2. **A swallowed line continuation** in `ci.yml` — the opposite sign of the literal-`\n` bug, on
   three pre-existing steps. Harmless there (pytest args just join) but fatal between two
   *commands*. Guarded by a 120-char cap on `run:` lines.

**The ARPACK chore was wider and sharper than the plan said.** §7 called it "four oracles,
harmless"; it was **twenty `eigsh` sites**, and the ones that matter return **eigenvectors** fed to
`set_state` — an initial condition, not a last digit. The two exactly-degenerate rigid modes came
back **1e-1 apart** run to run. `tests/helpers.arpack_v0` pins a seeded PCG64 start vector (no
transcendental, no symmetry); `test_stability.py` guards it by walking the **AST** rather than
listing call sites. Deliberately not done: the six `eigsh` calls in `web/serialize.py` (Phase 8).

**What Phase 5 inherits:** `sparse_lu.rs` is built, so `operators2d`, `plate`, `connection` and
`string_geometric` need no new solver; what they *do* need is `portable.canonical`, written down in
advance by [[rust-phase3-strings-state]]. See [[rust-migration-state]].
