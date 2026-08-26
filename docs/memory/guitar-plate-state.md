---
name: guitar-plate-state
description: "Model #5g, the plate's outline stops being a rectangle (2026-08-26) — the mask is not the outline, and the staircase error is a domain-size error the disk cannot test"
metadata: 
  node_type: memory
  type: project
  originSessionId: 1a9a1d31-894b-4037-a3c8-bee6efdd638d
  modified: 2026-08-26T07:38:32.043Z
---

Model **#5g**, built 2026-08-26, `docs/dev/guitar-plate-plan.md`. Discharges HANDOFF §12B's
non-rectangular-outline refusal — the last named plate refusal apart from orthotropic von Kármán and
the clamped rim. No new PDE: the same bending operator, ledger, θ-scheme and four grain constants,
assembled on a **mask** instead of a bounding box.

**The mask is not the outline.** A curved rim staircases into one-node spikes that touch no complete
quadrature cell, so their trapezoidal area weight is exactly `0` — the free plate's *mass* matrix
goes singular and the factorisation fails outright. Two such nodes are enough; the default guitar
makes 2–4 at every grid. Pruning them is a **silent geometry change**: the rule is topological, so on
a coarse grid with a deep waist it fires *mid-plate*, and energy, nullspace and spectrum all survive
that intact. Asserted geometrically instead — every dropped node within one `h` of the rim, met at a
measured **0.70–0.75 `h`** on the shipped path (the probe's wider 0.53–0.99 was an artefact: it built
the outline on the *nominal* length, `Plate` snaps `Ly` to whole cells first). The depth is
**exposed and asserted**, not merely compared — a bar that raises only on violation is never
observed on a grid that passes, so a sign error in it leaves every test green. That was one of
**three tests that passed without testing their own claim**, all found by an advisor pass *after*
the push: the rim test was vacuous, the pinch test's regex accepted either refusal (so the
connectivity path could never run), and the shipped `Plate(domain="circle")` was anchored only by
the two geometry-blind tiers because the oracle test drove a locally-built mask instead of the
constructor.

**The headline, and the shape that could refute it.** The staircase error is largely a **domain-size**
error rather than an operator error: frequency error tracks the mask's *area deficit*
mode-independently, so a staircased plate is a well-modelled slightly *smaller* plate, and dividing
the deficit out lifts O(h) toward O(h^1.5). Measured on a **disk** first, worth 6–15×. The disk is
exactly the shape that **cannot test it** — uniformly convex, so its area deficit and its mean
distance-from-the-rim are the same number. On the guitar (concave waist beside convex bouts) it
survives at only **3.7–7.4× and without the monotonicity**. Quote the range, never a factor. Related:
[[free-plate-orthotropic-state]], [[membrane-state]].

**Only one tier can falsify anything here,** which is the family's standing rule again
([[air-box-state]]). Energy is geometry-blind. The nullspace `{1,x,y}` is necessary, not sufficient.
And the **supported branch cannot fail**: `B = L @ L` already took any mask and `eig(L²) = eig(L)²`
makes a supported curved plate the membrane's spectrum *squared* — shipped as a **refusal at
construction**, not a model surface. So the weight sits on a **derived** free-*circular*-plate
frequency equation (no closed form exists for a guitar), self-checked three ways: rigid-body roots,
every root returning `λ⁴` in the plate's own Rayleigh quotient, and a Bessel-free saddle bound
`sqrt(48(1-ν))` overshooting by 8.18% — the disk's version of #5of's twist bound.

**Two scars, both of which looked like assembly bugs and were not.** Comparing a paired discrete
spectrum against an oracle listing each degenerate pair *once* reads as 26–42% spurious modes that
get **worse** under refinement. And `ω = κk²` makes the literature's frequency parameter `λ²`, so
checking a root against `λ²` instead of `λ⁴` makes every correct root look 2.3× off.

**`free_plate_stiffness` collapsed into the masked routine**, but the gate failed first: a
bit-identity measured at one grid is not measured. Across 7 grids × 4 ν × 3 grain splits it came back
`7.3e-12` because the Kronecker twist is a **product of two forward differences** — coefficient
`(1/h)·(1/h)`, not `1/(h·h)`. Those differ only when `h` is not exactly representable, so it failed
on exactly one grid (`h = 0.05`) for every ν and every split, and on none of the other six. Rule:
spell a coefficient the way the operator is *built*, not the way it is written. Fixed → `0.0` on all
84 cases.

**Surfaced in the viewer the same day** — [[guitar-plate-viewer-state]] (batch 20). It answered this model's open question (the waist **reorders**, and the detector is a mirror parity that reads exactly ±1) and found two things this plan did not: the viewer's *display* decimation can draw the outline as **two disconnected guitars** while the solver has one plate, and the plate's default strike sits close enough to the centre line that the odd family — the fundamental past the crossing — is barely excited.
