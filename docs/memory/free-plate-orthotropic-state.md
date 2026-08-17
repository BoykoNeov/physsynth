---
name: free-plate-orthotropic-state
description: "Model #5of free-edge orthotropic plate — the free edge needs FOUR constants, sees the split the supported plate cannot, and the ledger cannot referee any of it"
metadata: 
  node_type: memory
  type: project
  originSessionId: f81ca37b-dd53-43be-b96a-ef50c9a3da2e
  modified: 2026-08-17T14:22:54.806Z
---

Model **#5of**, 2026-08-17: the **free** plate gets a grain (`docs/dev/orthotropic-free-plate-plan.md`).
Follows [[orthotropic-plate-state]] (#5o, supported branch) and [[free-plate-state]] (#5b), and was
named by #5o as its own next step. Four bending constants where #5o needed three.

**The load-bearing asymmetry.** Merging the coupling and torsional rigidities into one cross term
`H = D_1 + 2 D_xy` takes two integrations by parts whose boundary terms vanish only on a *pinned*
rim. A free rim keeps them — its corner force is *pure* torsion. So `grain_coupling` and
`grain_torsion` are separate arguments on the free branch, and the contrast is measured on both
sides in one file: **two splits of the same `H` are the same supported plate BIT-IDENTICALLY (a
printed `0.0`) and different free plates by 6.5x in the fundamental.** The 6.5x is a mode *crossing*
plus a detuning, not pure detuning — the fundamental's identity changes across the sweep.

**Contrast with #5o that generalises: the grain DOES reorder modes here.** #5o's surviving headline
was that the cross term detunes selectively but never changes mode order. On a free edge the low
spectrum is a race between a twist mode governed by the torsional rigidity *alone* and a cross-grain
bender governed by `D_y`. Crossing measured at `g_y/g_xy = 1.025`; real spruce sits only 12.6% above
it. So "the free plate's fundamental is the twist" (a shipped #5b sentence) is a fact about
*isotropic* free plates.

**Validation had to be invented — four probes, one per constant.** No closed-form spectrum, and the
external orthotropic-FFFF tables (Narita; Leissa ch. 11) are paywalled, so nothing was entered from
memory. Derived instead:
- `g_xy`: the centred saddle's Rayleigh quotient — provably blind to the other three, numerator
  `4 g_xy ab` in closed form, and a one-sided bound `omega_1 <= 24 sqrt(D_xy/rho_s)/(ab)`. Its
  overshoot is **nearly material-independent** (5.27% isotropic, 5.55% spruce), which is what lets
  one bar cover both arms. This is why luthiers read the shear modulus off a tapped plate.
- `g_x`/`g_y`: an **exact** reduction to the shipped 1-D free beam on transverse-independent fields —
  but only at **zero coupling**. With coupling it breaks, and that is *anticlastic curvature*: the
  field keeps **exactly** the beam's energy while ceasing to be an eigenvector (a shape change, not
  an energy change), and the eigenvalue shift is downward for a strip and upward for a square.
  Anyone who "fixes" that residual has deleted the physics.
- `g_1`: the `(x², y²)` bilinear probe, exact discrete value `4 g_1 h²(Nx-1)(Ny-1)`. It is the
  **ONLY** detector of the four that responds to the coupling rigidity at all — so the least
  glamorous one is load-bearing.

**Three scars worth carrying past this model:**
1. **A superseded argument that is still validated is still a live constraint.** The operator
   validated `nu` against the isotropic `(-1, 1/2)` even when a split superseded it, and so rejected
   a legitimate orthotropic plate whose implied `nu_yx` was 0.70 — inside this branch's own guard,
   because an orthotropic sheet is bounded by `nu_xy nu_yx < 1`, a *different* admissible set.
2. **"Exact in exact arithmetic" is not exact in doubles** when the terms are products of `h²` with
   `1/h⁴`: two closed forms predicted at machine precision landed at 2.5e-14. Bars are set at
   measured magnitudes, and each test also asserts that the no-cancellation build is the closer of
   the two, so a future regression fails instead of passing at a loose bar.
3. **A guard can be provably sufficient and measurably conservative, and that is the shippable
   kind.** `g_1² < g_x g_y` with `g_xy > 0` is sufficient on any grid (pointwise argument) but the
   *discrete* operator survives 4–20% past it, with the margin shrinking under refinement — so it is
   the sharp *continuum* condition. Rejection is one-sided by design; #5o's supported guard was
   sharp, so the two are not the same kind of object. Separately, `g_xy = 0` is **degenerate, not
   stiff**: the saddle joins the rigid-body nullspace as a 4th zero mode.

**And the family rule holds again, from a new angle: the energy ledger cannot referee any of it.**
Three visibly different plates — spruce, its split swapped, all of `H` taken as torsion — conserve
at 1.5–2.1e-13 with fundamentals 5.68, 3.79, 5.99. Any symmetric `K` conserves exactly.

**One code path** here (unlike #5o's supported branch, which keeps a separate isotropic path): the
`nu`-derived split reproduces the pre-grain assembly **bit-exactly on all 7 survey grids x 4 `nu`
values**, because the four coefficients multiply the same four matrix products in the same order at
`1.0`, `1.0`, `nu` and an exactly-representable halving.

Still refused, deliberately: **orthotropic von Kármán** (four-constant in-plane compliance tensor, no
closed-form oracle — this batch is its prerequisite, sharing the strain-energy assembly) and a
**guitar-shaped outline** (the membrane batch's staircasing, in a 4th-order operator). Neither is in
the viewer — see [[web-viewer-state]], whose model list is closed.
