---
name: vk-newton-part2-state
description: "VK Newton Part 2 — Newton behind couple_method; Picard's \"increment\" IS a residual, one field costs two, and Newton clears all three walls"
metadata: 
  node_type: memory
  type: project
  originSessionId: 1ab138c5-02fb-4f8e-b8cc-880794a8d2e9
  modified: 2026-09-06T07:46:59.820Z
---

Part 2 of the von Kármán Newton batch (`docs/dev/vk-newton-plan.md` §11), landed 2026-09-06.
`couple_method` on `VkSpec`/`VkParams` and the `VKPlate` constructor, default `"picard"`; a new
`crates/physsynth-core/src/krylov.rs` (matrix-free restarted GMRES, no dependency); `vk_newton` in
`plate.rs`. No shipped number moves. Builds on [[vk-newton-part1-state]] and
[[vk-newton-part0-state]].

**Picard's `last_residual` looks like an increment and IS a residual.** Its `diff` is
`sweep(w_j) - w_j = -G(w_j)`, so the shipped loop already stops on `‖G‖/‖w‖`. Newton's step
`-J⁻¹G` is a *different* quantity, and the two agree only while `J` is near `I` — 2.4–12.4% on the
gate fixtures and **unbounded** in the region the batch exists for. Stopping Newton on its step
norm would have made the gate ("same root to `couple_tol`") uncheckable while reading as checked.
So Newton stops on the same residual ratio, which is free (it is the next iteration's right-hand
side) and keeps Part 0's `couple_outcome` classifier meaningful.

**A new diagnostics field costs TWO edits, so budget one.** §9.5's fork — `VkPlate::step` and
`PyVKPlate::step` each hand-copy the step's diagnostics — is still there. `VkStep` gained exactly
one field, `n_solves` (back-substitutions), because `n_iters` is not comparable across the methods:
a Picard sweep is 2 solves, a Newton iteration is 2 + 2/Krylov product + 2/line-search trial, so
counting iterations would show Newton ahead 20× on a wash. Everything else worth knowing
(line-search halvings, Krylov products, inner stalls) lives on `VkNewtonReport`, which native tests
read and nothing mirrors.

**The line search does not fire — and the number that matters is where it starts to.** Zero
halvings on all six gate fixtures and all three wall fixtures. Pushing the seed: 1×/10×/40× take 0
halvings, **200× takes 6**, 10⁴× takes 33 — all converge. GMRES starts hitting its product cap at
40×, well before the search fires. Both halves are asserted (does *not* fire at 40×, does at 200×).

**Newton converges on all three of Part 0's walls** — 4–5 iterations where Picard is expansive or
capped at 400 — and 300-step runs conserve energy to 7e-13/1.7e-12/2.8e-13 with every step
converged. That is a property of the **iteration**, not a claim on the territory: energy is
conserved by *any* root of the discrete-gradient equation, so it is not independent evidence the
root resolves the physics. Trap 3's refined-`k` check is Part 3's.

**The airbox room scene cannot run under Newton at all.** `_VKPlateSurface.solve`
(`crates/physsynth-py/src/airbox_wrap.rs`) runs its own Picard loop in Python-object arithmetic
against the room-**loaded** factorization and never consults `couple_method`. Part 5's gong-in-a-room
needs wrapper-tier work; the gong on a string and the mallet on the gong go through `VKPlate.step`
and are fine. See [[air-box-state]].

**How to apply:** the Newton driver reads only `couple_tol`/`couple_max_iter` and goes through
`VkCoupledStep` — it forms no `h`, `k` or `rho_s`, which is the whole guard for the free edge's
extra `h²` (Part 1 measured that a finite difference is structurally blind to `couple_factor`).
Keep it that way. GMRES is asserted in `crates/physsynth-core/tests/krylov.rs` against hand-built
matrices, not against the plate — "the plate converged" is a weak bar for a linear solver.
