---
name: guitar-plate-viewer-state
description: "Viewer batch 20 (2026-08-26) — model #5g on screen. The claim is that the waist SWAPS the fundamental and the detector is a mirror PARITY reading exactly ±1 (a true crossing, not avoided). The outline is a DOMAIN of the plate, not a model id. Two traps: the display decimation can draw TWO guitars while the plate is one (249/140,349 REACHABLE, every detector green), and the shipped strike cannot EXPRESS the claim (centre line ⇒ exactly zero). The waist slider is QUANTISED."
metadata: 
  node_type: memory
  type: project
  originSessionId: 9b54963e-d6cb-4a18-9490-a5453e466625
  modified: 2026-08-26T10:06:59.606Z
---

Viewer batch 20 surfaces model #5g ([[guitar-plate-state]]) — see
`docs/dev/guitar-plate-viewer-plan.md` and web-viewer-plan.md's batch 20. What to carry forward:

**The claim, and why it was cheap.** Deepening the waist does not just detune the plate — past a
critical waist the *fundamental changes shape*: a long bender below, the twist above. Neither
precedent implied it ([[orthotropic-plate-state]] detuned without reordering,
[[free-plate-orthotropic-state]] reordered). The outline is `|x| < W(y)`, mirror-symmetric whatever
the sliders do, so every mode is exactly even or exactly odd across the width, the two families
cannot couple, and **the crossing is a TRUE crossing — nothing can open a gap**. So the detector is
one scalar (the mode's overlap with its own mirror image) that reads **+1.00000 or −1.00000**, never
between: no shape tracking, no threshold, no eigenvector matching across a near-degenerate pair.
Exactly one flip in all 12 configurations tried. Crossing is grid-converged (0.240→0.260 over
N = 24…76) but **not a constant** — it moves with elongation (0.18 at aspect 1.08 → 0.57 at 1.68).

**Do NOT quote `min f2/f1` as evidence.** It is non-monotone in N and it is not a gap: refining the
scan six times returned the *identical* residual, because the mask stops changing. The crossing is an
**interval between two adjacent reachable waists**, and that interval's width is the quantisation.

**The waist slider is QUANTISED** — the outline is a staircase, so it only changes the plate when a
node crosses the rim. **50 distinct plates at N = 16, widest dead band 0.107** (a tenth of the
travel doing nothing); 675 at N = 80. Say it on the panel or a drag that does nothing reads as a
broken viewer.

**Trap 1 — the picture can split where the plate does not.** The display decimation point-samples,
which is benign on a *convex* outline (that is why the drumhead never found it in six batches) and
not on a concave waist: **249 of 140,349 configurations reachable from the viewer's own sliders
render as two disconnected lobes** while the solver has one plate (0 under pooling). Energy, nullspace and audio are all green
through it — only the picture is wrong, and two lobes read as a design. Fix is a **proof**: max-pool the display mask (live
iff any solver node in the block is), which cannot disconnect a connected set. Left the membrane's
disk on the old path deliberately — pooling never splits it either, so there is no bug, and changing
it would perturb a shipped rendering for nothing.

**Trap 2 — the shipped strike could not express the claim.** A strike on the centre line has
*exactly zero* overlap with every odd mode, and past the crossing the fundamental IS odd. At the
plate's own default strike x the fundamental was already 12× weaker than the second partial; one
slider step to the centre and it is 1.7e-14. **And no strike point serves both branches** — near the
waist shows the bender, in a bout shows the twist. So the compromise is *reported* (a bar track of
the strike's overlap at every waist) rather than hidden, plus a red warning that fires **before** a
render is paid for. Same family as [[orthotropic-plate-state]]'s square-plate blind spot and
[[vkroom-viewer-state]]'s drive_index: a correct claim measured on a configuration that cannot
express it.

**Structure: the outline is a DOMAIN of the plate, not a new model id.** A new model id cannot show
`Lx`/`Ly` at all (they are gated by the secondary select), and the plate's select already carries
supported/free — so the third value is the guitar, and the three options are exactly the three
plates that exist (a supported guitar is refused by the core). `payload.model` stays `"plate"` and
the rectangle payload is **bit-identical across 20 configurations**, measured against HEAD.
**Repairing an invariant was part of the price**: the slider re-range gate was keyed on the *model*,
so adding the first `plate:` regime would silently have started resetting sliders on a
supported↔free switch, which the plan explicitly forbids. A mechanism keyed on the model cannot
express an invariant stated about a regime — and the first new regime is where you find out.

**A rectangle law was being drawn on a plate that is not a rectangle.** The free plate's continuum
reference (the Leissa square anchor) is gated on the plate being *near-square* — a test of the
BOUNDING BOX, when what matters is whether the reference APPLIES. A guitar with a square bounding
box passed that gate and was scored against the rectangle's anchor: **799 cents off**, on the one
panel whose job is to report distance from theory. Nothing failed. Found by sweeping the reachable
geometry grid for clean refusals, not by any test — **walking the parameter space is cheap and it
is how this surfaced.**

**Two things were argued and then measured the other way.** The plan's gate claimed #5g needed a
new frontend capability (masked field rendering) — it did not; that shipped with the membrane. And
the build record first guessed the display split was unreachable once the waist was capped, on a
hunt run at *one* body width; widening it to the sliders' range broke that immediately. **A guard
that is only argued for is a guard that is being talked out of.** Check what the frontend actually
does, and hunt across the whole reachable range, before pricing a batch on memory of either.

**The survey that found the mislabelled oracle found a second thing, and it is not a viewer bug.**
The same `log2` warning also came from `measure_partials_near` returning **-502 Hz** — the partial
detector every model test leans on. Recorded in [[spectrum-detector-guard]]; the transferable half
is that walking the parameter space produced two independent defects in one warning, and neither had
a test looking for it.
