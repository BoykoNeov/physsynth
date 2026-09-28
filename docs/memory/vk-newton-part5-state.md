---
name: vk-newton-part5-state
description: "VK Newton Part 5 — the gong on a string is no longer iteration-bound; two of the three scenes were misfiled (one never built), and a CENSORED grid produced a clean false null result"
metadata:
  node_type: memory
  type: project
  originSessionId: 8d72a366-f87e-40e3-b252-a42a1f72c712
  modified: 2026-09-06T09:49:30.133Z
---

Part 5 of the von Kármán Newton batch (`docs/dev/vk-newton-plan.md` §14), landed 2026-09-06.
Measurement only, by the human's call: run the one scene that can run, report the other two.
`M:\claud_projects\temp\vk-newton\part5_scene.py`, `part5_scene2.py`. Follows
[[vk-newton-part3-state]].

**Two of the three "scenes bounded by the iteration" were not scenes.** The gong in a room is
blocked at the wrapper tier (known). **The mallet on the gong was never built** —
`PyMalletMembrane` casts its collaborator to a `PyMembrane` and refuses anything else; there is no
`MalletPlate` anywhere. The phrase came from a plan that *recommended* a mallet for a future batch
and became, by quotation, a list of existing compositions. Same failure as
[[vk-newton-part4-state]], twice in one batch: **a list of scenes is worth a grep before it is
worth a plan**, and the check here was one `#[pyclass]` signature.

**A CENSORED grid is not a measurement — and one that makes two rows EQUAL is worse than one that
makes them differ.** The first pluck grid stopped at 50 mm, censoring three of six rows, and because
best-effort Picard and Newton were both censored on the gong it read as a clean "the entire gain is
the sweep cap" null result. Extending it: gong 30 mm (shipped cap 50) → 70 mm (best effort) →
**≥300 mm** (Newton); cymbal 20 → 20 → **≥300**. So Newton is ≥4.3× and ≥15× over an honest
baseline. [[vk-newton-part3-state]] had *invented* a censored column for exactly this and the trap
was walked into one section later anyway.

**No Newton tax on a scene, though the map found one on the bare model.** Worst solves per step:
20/20/20, 32/32/28, 50/50/38. §13.3's 0.55–0.85× Picard advantage does not appear, because the
plate's per-step work sits next to a string's. **A cost result from a bare-model map does not
transfer to a scene.**

**The curvature axis is about the DEFLECTION, not the drive.** §13.6 predicted narrow excitation →
unresolved territory, and this scene has the narrowest drive in the project (a point force at one
bridge node) — yet it holds the scheme's second order (ratios 3.5–4.3) out to ~5 ms at every pluck
measured, decorrelating after. §13.6's fixtures set the plate's *state* to a narrow Gaussian, so the
deflection itself was narrow; here the deflection is a mode pattern and is smooth. The drive is what
a reader can see in the code; the deflection is what matters.

**The bound moved off the solver.** At the top of the grid the gong reaches **154×** and the cymbal
**546×** their thickness in **four to six** Newton iterations, conserving to the 1e-10 bar. Von
Kármán is a moderate-rotation theory, so the *model* leaves its range long before the solver
struggles: §5's "bounded by the iteration rather than by the physics" is retired for this scene.
The resolution check covers plucks to 50 mm, not 300 — the claim stops where the measurement does.

**A free-boundary cross-check arrived unasked:** supported and free rows are identical to four
digits at 1.35 and 2.70 ms and separate at 5.40 ms — [[vk-newton-part1-state]] §10.4's phenomenon
(until the response reaches the rim the two are the same interior problem), here dating this plate's
rim arrival at ~5 ms.

**Shipping a capability creates FALSE INFERENCES in docs that predate it.** `HANDOFF.md`'s room
bullet says coarsening the air grid breaks the plate's fixed point — still true, and a reader who
has met `couple_method` elsewhere in the file will now assume Newton fixes it. It does not: that
seam runs its own Picard loop. The bullet had to say so explicitly. Sweeping for a *falsified*
claim (Part 4) is not the same sweep as looking for claims a new feature makes newly misleading.

**How to apply:** the wrapper work for the room scene is deferred to **its own batch** — making
`_VKPlateSurface.solve` consult `couple_method` means Newton against the room-**loaded**
factorization, which `vk_newton` cannot reach through `VkCoupledStep` today. A mallet-on-a-plate
composition is a model batch with its own energy bars, and is now filed as *missing* rather than as
*bounded*. See [[air-box-state]] and [[string-vk-bridge-state]].
