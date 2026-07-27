---
name: jawari-state
description: "Jawari / buzzing bridge (sitar & tanpura curved-bridge timbre) — a CONFIGURATION of model #8 (BarrierString), zero new core code; curved parabolic barrier at the termination; discriminating signatures = sustained shimmer + travelling wrap; money gates carried onto the curved profile"
metadata:
  node_type: memory
  type: project
  originSessionId: b935be8a-02a4-41e8-bf76-d89011a9f00f
---

**Jawari / buzzing bridge** (built & green 2026-07-10, full suite **723**). The sitar & tanpura
curved-bridge timbre. **NOT new core physics — a configuration of model #8** ([[barrier-collision-state]]
`BarrierString`, string vs one-sided distributed nonlinear barrier). #8's docstring/plan already
named the jawari as its target; this batch **realizes + validates** it and writes **zero** core
code (advisor: don't manufacture a "model #9"). Suite **723** (pre-jawari 712 = 693 barrier + 19
sympathetic; +11 jawari). Files: `make_jawari_string` + `jawari_barrier` in
`tests/helpers.py`, `tests/test_jawari.py` (11), `scripts/diagnose_jawari.py`, `docs/dev/jawari-plan.md`.

**Profile** = downward-opening parabola tangent to the rest line at the fixed termination, `-inf`
off-support beyond the bridge span: `b(x) = -clearance - depth·(x/d)²` on `0<x≤d=width_frac·L`.
Crest (nearest the string) at the termination side; curves away by `depth` at the far edge.
Defaults `width_frac=0.15`, `depth=1mm`, `K=2e6`, `N=100` (support ≈15 nodes — **resolves the wrap,
well under the NumPy Windows dense-solve BLAS cliff at n≥100**), `clearance=0` (grazing).
`clearance<0` preloads crest above rest → **whole span contacts at rest** = the static-oracle case.

**THE one geometry lesson (tuning, cost me the first prototype):** `depth` must be **comparable to
the near-termination downswing** (`≈amp·π·width_frac`), NOT larger. Too deep → string only grazes
the crest → acts like a point contact, no wrap, no shimmer (my first try depth=6mm: a flat rail at
matched clearance OUT-brightened it). At `depth≈1mm` (≈ downswing) the string wraps a wide span and
the departure point travels.

**Why NOT the fret-buzz already tested in #8's `test_collision_signature.py`:** that's a flat
rail / point fret = localised **intermittent** slap. The jawari is the physical opposite — a
**persistent travelling wrap** on a curve → **sustained** highs. New tests must SEPARATE curve from
flat, not re-run intermittency. Two discriminating signatures:
1. **Sustained shimmer** — clean-string midpoint pickup stays near the fundamental; jawari's curved
   contact re-injects highs each downswing → late-window centroid ~3× the clean string's, hasn't
   collapsed to its own fundamental. The **late/early ratio is fragile** (depends on decay/window —
   wobbles 0.9–1.3 across σ1/steps); the **absolute late-window elevation (jl > 2.5·cl) is the
   robust gate**. Sim is deterministic so a fixed config is non-flaky.
2. **Travelling departure point** — furthest-in-contact node (wrap edge) std ≈4.8 on the curve vs
   ≈2.0 on a flat rail at matched min-clearance (flat pins at the far edge; curve sweeps 0..14).

**Two machine-precision money gates (carried over from #8, re-exercised on the CURVED, persistently-
wrapping regime #8's flat-rail/point tests never touched):**
- **Lossless energy conservation through the sustained curved wrap** (σ=λ_h=0, α∈{1,1.5,2}): drift
  **~7e-13**, no solver stall — vector discrete-gradient telescoping handles many-node persistent
  contact.
- **Static-equilibrium magnitude oracle on a curved α=1 profile:** crest preloaded → whole span in
  gentle contact → DG hits no-warp Taylor branch → discrete fixed point == continuous augmented
  eq. `S u*=(K/ρ)b` (`S=-L+(K/ρ)diag(mask)`, mask=bridge support, `b`=the *curved* heights) to
  **~2e-16**. Pins magnitude for the curved profile specifically (a flat-only test wouldn't).
  Negative control: double `_G` & `_force_pref` → drift blows up >1e4×.

Diagnostic `diagnose_jawari.py` (4 panels): the wrap (string conforms to curve, zoom on bridge) /
sustained shimmer (jawari early≈late spectrum full to 4kHz vs clean = fundamental only) / travelling
departure point (curve sweeps, rail pinned) / energy partition (total flat 7e-13, string↔bridge-PE
exchange). Browser-verified figure.

**Natural follow-ons (kept out this batch, advisor):** tanpura **cotton thread (juari)** = one more
barrier node at a chosen position (obvious next); **jawari + [[sympathetic-strings-state]]** =
full sitar/tanpura instrument; sub-grid contact (wrap edge hops node-to-node = #8's documented
limitation). See [[barrier-collision-state]], [[mallet-collision-state]], [[commit-push-at-batch-end]],
[[respect-ruff-line-length]].
