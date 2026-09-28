---
name: vk-newton-part1-state
description: "VK Newton Part 1 — the Jacobian shipped and asserted; an FD check cannot see the wrong MAP, and its power is a function of how far J sits from I"
metadata: 
  node_type: memory
  type: project
  originSessionId: ee3d446c-a418-44e1-8929-f8a3686ff42f
  modified: 2026-09-06T04:33:03.023Z
---

Part 1 of the von Kármán Newton plan (`docs/dev/vk-newton-plan.md` §5, result in §10), landed
2026-09-06. `VkCoupledStep` in `crates/physsynth-core/src/plate.rs` now holds one step's
sweep-invariant parts and offers `sweep` / `residual` / `jacobian_vector`; four native bars in
`crates/physsynth-core/tests/plate.rs`. Follows [[vk-newton-part0-state]]. No flag, no binding
change, no shipped number moved.

**A finite difference verifies a derivative — it cannot verify that the derivative is of the right
map.** If the Picard sweep and the residual are two transcriptions that drift apart, `J` becomes
the exact derivative of a function the loop is not iterating, and *every* FD check still passes.
So the loop body was **moved** into `sweep`, and `residual` is a wrapper over it — never the
reverse, because recovering the sweep as `w − G(w)` is exact only under Sterbenz and would move
the shipped trajectory. The two claims (refactor is bit-identical; derivative is right) are
independent and were established separately, the first by a 300-row before/after dump.

**The FD bar's discriminating power is a function of `‖Jd − d‖/‖d‖`, and that had to be measured.**
At zero amplitude `J = I` exactly, so a stub returning `d` is a *good approximation* at low
amplitude. Measured: taking `F'` at `w̄` instead of the raw `W` — a real, plausible, ~2× error —
reads 1e-4…1e-2 on loud fixtures but **6.3e-8 at margin 5.9e-3**, i.e. green under a 1e-7 bar on a
fixture nobody would flag as weak. The margin floor is therefore placed between two measurements
(gated fixtures 0.024–0.124; vacuity demonstrated at 0.0059), not at a round number.

**A "free boundary fixture" is only one if the excitation reaches the boundary.** A 3–8 cm Gaussian
centred on a 40 cm plate is `exp(−44)` at the rim, so a free plate (441 nodes) and a supported one
(361 nodes) ran the *same interior arithmetic* — margins agreeing to **twelve digits**. Nothing in
the suite says so: energy bar, FD bar and margin all pass identically either way. The tell was the
coincidence itself. Fixed by striking off-centre (35–38% of peak on the rim).

**`couple_factor` is structurally shared between `sweep` and `jacobian_vector`, and that sharing —
not the FD bar — is the guard for the free edge's extra `h²`.** FD is blind to it by construction.
Mutating it away *was* caught, but as the physics blowing up (map diverges, point overflows), never
as a derivative disagreement. Do not write "the FD bar catches the h²"; it is false.

**Bars measured, then loosened.** The check that the model's own step is a root of the residual read
3.4e-14…7.5e-14 against `couple_tol = 1e-13` — 1.3× headroom, not enough for two sparse
back-substitutions on another machine, so the bar is `10·couple_tol`. Same argument as CLAUDE.md's
refusal to tighten the `1e-10` energy bar. FD measured 9.5e-12…1.1e-10 against 1e-7; linearity
8.9e-17…1.4e-16 against 1e-13. The FD step is chosen from a measured V-curve (`G` is exactly cubic,
so truncation is pure `ε²`): floor at `ε = 1e-5·‖w‖`, two decades of slack either side.

**Six mutants, run.** FD caught all six and is the workhorse; the margin floor caught the two that
leave `J` closest to `I`; the linearity identity caught one (every other mutant is still exactly
linear in `d` — it is a structural check, not a transcription one); the root check caught only the
mutant that touched `sweep`, which is its job.

**The probe scripts in `M:\claud_projects\temp\vk-newton\` mutate the core in place** and restore
from memory at exit — a killed run leaves a mutated `plate.rs` on disk, one variant changing the
free plate's default path by 2500×. `git diff` after running one; the script's "restored" line is
not evidence. See [[destructive-undo-discipline]].
