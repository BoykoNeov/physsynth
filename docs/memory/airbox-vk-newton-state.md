---
name: airbox-vk-newton-state
description: "The room seam's solver batch — the seam's own Picard copy is gone, the wall is the BARE plate's, and the payoff has no single number because the boundary moves with the grid"
metadata: 
  node_type: memory
  type: project
  originSessionId: a5662165-251c-4cf7-894e-b3d6bbee2376
  modified: 2026-09-06T15:03:50.046Z
---

2026-09-06. `docs/dev/air-box-vk-newton-plan.md`. Chosen by the human over four alternatives when
asked "what is next" after model #7g. The blockage was one method: `_VKPlateSurface.solve` carried a
hand transcription of the Picard sweep, so no airbox von Kármán scene could run under Newton
whatever `couple_method` said. It now drives the model's own kernel against the loaded
factorization, via a `ThetaSolve` trait in the core.

Five things worth carrying forward.

**The air load does NOT move the wall.** The natural hypothesis — the room makes the iteration
harder — is false. Room and bare transition cell for cell (capped at 3.5e, two capped at 4e, blown
up on step zero at 4.5e). So the scene was never iteration-bound *by the room*; it was denied a fix
that already existed. Measure this before scoping any "the coupling makes it worse" batch.

**A hand transcription can be measured, and then deleted.** Driving the seam with the plate's own
unloaded factorization reproduced `VKPlate.step` bit for bit — both boundaries, forced and unforced,
in `w` and in `F`. That measurement is what made deleting the copy a refactor rather than a
re-derivation. The `f_ext` arm was added only after the advisor pointed out that the table claimed
the whole seam while measuring half of it.

**The payoff has NO single number — the wall moves with the GRID as well as with `fs`.** Picard's
last good amplitude is 4e at N=20 and 4.5e at N=8; Newton's is 6e and 10e. So the gain is 1.6x on
one grid and 1.9x on the other, both below the 2–4x [[vk-newton-part3-state]] mapped at 48 kHz. This
cost a red test: the payoff test was first written at the plan's `4.5e`, which is past the wall on
the grid the plan measured and comfortably inside it on the grid the test file builds. **A test
whose control is "the old way fails here" must assert that control on the fixture in front of it.**

**A hand-picked witness is a hand-picked answer — second occurrence.** The routing guard needed
three tries. Recovering the Jacobian's correction as `d - Jd` cancels (bar had to become absolute
against the largest correction); a quiet fixture makes `J = I` numerically so nothing is
distinguishable (had to use the loud one); and the killer — **a direction linear in the index is
nearly in the Monge–Ampère bracket's kernel**, because the bracket differentiates twice. The
smooth, obviously-nonzero witness gave exactly 0.0 on the free edge. The rule: *a witness for an
operator must not lie near that operator's kernel.* See [[guitar-plate-state]] for the other
direction of the same mistake.

**The seam was misreporting its own failure, independently of Newton.** It never wrote
`residual_ratio`, and `couple_outcome` tests the ratio *positively*, so a NaN filed **every**
non-converged room step as `expansive` — "no cap will ever help" about steps a bigger cap does fix.
`n_solves` sat at 0 for the life of a room-driven plate, and it is the only axis on which Picard and
Newton may honestly be compared. Both are written now, from one `VkStep`, through
`PyVKPlate::record_iteration`.

**Still missing, and named:** the mallet on a gong **in a room**. Its outer chord freezes the
*linear* plate's drive-point column from `A^-1 e_node`; against the loaded `A` that is a different
column, and whether it should be recomputed is a physics question with its own energy bars. Part 1
gives that composition the primitive; it does not perform it. See [[mallet-gong-state]].

Related: [[vk-newton-state]], [[von-karman-plate-state]], [[air-box-state]], [[vk-newton-part5-state]].
