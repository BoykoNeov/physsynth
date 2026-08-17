---
name: orthotropic-plate-state
description: "Model #5o, the plate with a grain (wood) — shipped 2026-08-17; both planned headlines died and the replacements are better"
metadata: 
  node_type: memory
  type: project
  originSessionId: e5d36289-0cbc-43ea-94e5-248bbd339ba6
  modified: 2026-08-17T14:23:24.738Z
---

Model **#5o**: the simply-supported plate (#5) gets three bending stiffnesses instead of one —
along the grain, across it, and a cross term — i.e. wood instead of metal. Shipped 2026-08-17.
`operators2d.orthotropic_biharmonic`, three `grain_*` keywords on `Plate`,
`plate.grain_ratios_from_material`, three oracles in `analysis/modal.py`,
`tests/test_plate_orthotropic.py` (21 tests). Plan: `docs/dev/orthotropic-plate-plan.md`.

**Why this batch and not a bell or a mass-spring network.** The selection rule is the project's own
acceptance contract, and it is worth reusing: **(a)** switching the new term off must reproduce a
shipped model, and **(b)** the new model must have a *closed-form* oracle, not a convergence rate.
Orthotropy was the only shortlist candidate with both. A curved shell is more exciting and fails
(b) — shallow-shell formulas are themselves approximations. Thermoviscous bore loss passes both and
is the standing runner-up.

**Why it is cheap.** `B = L @ L` on a rectangle *is* `Dxx² + 2 Dxx Dyy + Dyy²`, so orthotropy is the
same three matrix products with three coefficients instead of one — no new stencil. And
`sin·sin` is already an exact eigenvector of `Dxx` and `Dyy` **separately**, so it survives
orthotropy exactly and carries the closed-form oracle with it.

## The trap: the factor of 2

`H = D_1 + 2 D_xy`, and **the other 2 is in the PDE, on `H`**, not inside it. The two rival
packagings the literature invites land at **0.30x** and **0.65x** of the correct cross term at
nu=0.3 — and all three produce a perfectly stable, exactly energy-conserving, wrong plate. See
[[unphysical-params-are-a-feature]]: the core API takes three dimensionless ratios (so a test can
dial the guard boundary), and `grain_ratios_from_material` is the optional realism layer.

Stability is no longer free: the isotropic `L²` is definite because it is a square, but with three
coefficients it is a **condition**, `g_h > -sqrt(g_x g_y)`, measured sharp to 2% and rejected at
construction.

## Both planned headlines died; the replacements are the batch's value

1. **Planned: the cross term reorders the modes.** It does not — across the whole range between
   solid spruce (0.567 of the "stretched isotropic" value) and isotropic material (exactly 1.0),
   all sixteen low modes stay in the **same order**, at two resolutions. **What replaced it: a
   selective detuner.** Modes shift 1.3%–29% over that sweep (22x spread), and the mechanism is
   that the cross term `2 g_h λx λy` matters **where the direct stiffness is weakest** — a mode
   bending along the grain has a huge `g_x λx²` that swamps it (+2.3%), one bending across it only
   has `g_y λy²` which is 13.75x smaller (+29.0%). Push the cross term to 2x stretched-isotropic
   (past any plank) and it *does* reorder — so "no reordering" is a statement about the range, not
   about the term being inert, and the test asserts both halves.
2. **Planned: a bridge point may not notice the grain.** Wrong as posed (frequencies move by over
   an octave). **What replaced it, and it is the one a coupled chain needs: the grain is in the
   partial series and not in the level.** Against an isotropic plate matched on the fundamental,
   mode ratios move **37%** while the single-node RMS lands at 1.116/0.809/0.873/0.916/0.834 across
   five pluck/pickup geometries — it **straddles 1**, so that spread is geometry, not grain. Judge
   a body by how loud its terminus rings and **you cannot tell wood from metal**. The first draft
   quoted *one* of those five as "the level moves 11.6%"; it measured where the pluck landed.

## Method notes that generalise

- **The energy ledger is blind again — fourth or fifth time in this project.** All three deliberate
  mis-wirings (grain swapped, factor of 2 dropped, `D_1` alone) conserve to machine precision,
  because a wrong-but-consistent coefficient is carried self-consistently through every step — the
  same shape as the `rho_v`/`rho_s` slip in [[air-box-state]]. Near enough a rule now: **a new
  coefficient needs an oracle, not a ledger.**
- **…and the oracle has a blind spot of its own, which the diagnose script found.** On a **square**
  plate, **every diagonal mode `(m,m)` is invariant under the grain running 90 degrees wrong** —
  swapping `g_x`/`g_y` while `λ_x = λ_y` leaves `g_x λx² + g_y λy²` untouched. So **checking a
  square plate's fundamental (or its first few diagonal modes) passes a plate whose grain runs the
  wrong way.** Ways out: an **off-diagonal** mode ((2,1)/(1,2) swap outright, 126% apart), or a
  **non-square** plate (fundamental alone moves 62%). Invariance is exact in real arithmetic but
  0 or **2.2e-16** in floating point — the two groupings round differently — so assert a tight
  bound, not `array_equal` (which (2,2) alone failed).
- So the batch ends with **three detectors and three different blind spots**: the ledger sees no
  wrong coefficient; the oracle sees no swapped grain *if asked only about diagonal modes on a
  square plate*; and a square plate cannot tell a material asymmetry from a shape one at all.
- **Close a trap, don't document it.** `grain_ratios_from_material` takes a **volume** density and
  `Plate` takes an **areal** one. v1 returned a bare 4-tuple and warned in the docstring — wrong
  shape of fix, because passing the volume density through leaves **every frequency right** (kappa
  carries them) and **every energy wrong by the thickness** (333× for 3 mm), which neither the
  ledger nor the modal oracle can see. Now returns a named `GrainSpec` carrying `rho_s`, so the
  mistake is unavailable and a stale 4-way unpack raises loudly.
- **A test can be a tautology.** The anisotropic-damping test first computed both decay rates from
  the formula it was claiming. Rewritten as a measurement with the isotropic plate as a control in
  the same rig: isotropic (4,1) and (1,4) decay at **5.857 / 5.857** (identical to six figures, as
  degeneracy requires), grained at **6.024 / 7.751** — 29% apart. The control is what makes it a
  measurement.
- **The isotropic collapse is grid-dependent** — bit-exact on `Lx = 1` grids, 1.7–2.4e-16 on
  others, depending on whether `1/h²` lands exactly. So the default deliberately stays on the
  **untouched `L @ L` line**, and the test that pins it runs on a grid **chosen because the two
  assemblies differ there** — on a grid where they coincide it would pass vacuously.
- **Two rig errors that looked like physics, one of them made TWICE.** (a) A twin built at matched
  `mu` instead of matched `fs` — `make_plate` solves fs from kappa, so the twin silently got a
  different timestep and frequencies disagreed 3%, reading as a modelling bug. Caught, fixed,
  documented — **then made again** in a second test (12.0 vs 20.5 kHz), where it happened not to
  move the numbers, which was luck. **A helper that derives one quantity from another will keep
  producing this until the comparison asserts the shared quantity** — so both tests now
  `assert a.fs == b.fs`. (b) Comparing along-grain against across-grain curvature on a **non-square**
  plate, where the shape asymmetry cancels most of the material one (1.6× instead of 2.95×).

## Refused, with reasons

- **Free boundary** — **DONE 2026-08-17 as model #5of, [[free-plate-orthotropic-state]].** It did
  need the coupling and torsional rigidities separately, and it overturned this batch's ordering
  result on the other boundary: the grain **does** reorder a *free* plate's modes. Note also that
  the supported branch's blindness to the split is now measured, and it is **bit-identical**, not
  merely small. A guitar-shaped top still needs a non-rectangular outline.
- **Orthotropic von Kármán (#6)** — needs a four-constant in-plane compliance tensor in the Airy
  solver and has no closed-form modal oracle, i.e. it fails the criterion that selected this batch.
  Take it *after* the free-edge orthotropic plate, which builds the machinery it needs.
