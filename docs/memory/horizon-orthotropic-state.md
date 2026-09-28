---
name: horizon-orthotropic-state
description: "The grained plate's resolution horizon (2026-09-07) — ONE floor in mode index, per-direction only in hertz; the isotropic direction bar was passing on integer rounding; the block corner rule breaks at exactly -1/m_max²"
metadata: 
  node_type: memory
  type: project
  originSessionId: af7e088f-6163-4e9b-be5f-74a8a0ff116a
  modified: 2026-09-07T08:17:24.412Z
---

The orthotropic plate's resolution horizon, measured 2026-09-07 (`docs/dev/resolution-horizon-plan.md`
§9, tests in `M:\claud_projects\physical synthesis\tests\test_resolution_horizon.py`). Closes the
last inventory row that had a stated shape and no numbers; see [[plate-mode-family-split-state]]
for §8, the isotropic 2-D split it builds on, and [[orthotropic-plate-state]] for the model.

**The prediction was half wrong, in the same direction as §8's.** §5 said the horizon would be
"per-direction like the membrane's". In **mode index** it is not — all three families sit on the
isotropic plate's `sinc(u)²` floor. Only in **hertz** is it per-direction: the two axial families
sit `√(g_x/g_y)` apart at the same index (3.7× for spruce), approached from below like `1/m²`.
So a horizon quoted as a mode count is grain-independent and the same horizon in hertz is not.

**The diagonal droop is `sinc(u)²` for ANY grain, exactly (<1e-14).** On a square domain `(m,m)`
carries the same `u` on both axes, so every grain weight multiplies the discrete and the continuum
stiffness alike and divides back out. That makes it a **pin against a mis-wired grain**, not a
discovery about orthotropy — it is what a cross term applied once instead of twice would break.
It survives at `grain_cross = -0.9`, where the continuum stiffness has nearly cancelled itself.

**The real cost: the direction bar was passing on ROUNDING.** §8's isotropic plate test asserts
`horizon <= predicted` on the argument that a timestep can only cost modes. An axial family's
*space* floor is already a hair **above** `sinc²` (its undrooped cross-axis term dilutes the
droop), so it wants to sit above the closed form at `k → 0`. On an isotropic plate the excess is
under a third of a mode and the integer floor absorbs it; a grain doubles it on the soft axis and
six of 48 fixtures cross, by up to 0.304 modes. The sign of the axial deviation is **the sign of
`grain_cross`** (positive = above the floor = every real wood).

**Two block conditions, both now in `mode_block`'s docstring:**
- the corner rule (`W(m,m) = m²` for any grain, so the diagonal corner is the block's worst mode)
  holds for every real wood and breaks at **exactly `-1/m_max²`** — exact to bisection, and it
  **tightens as the block grows**, so a large enough block breaks it for any negative cross term.
- `mode_block`'s **ordering key is the isotropic frequency** `m²+n²`. On a grained plate the
  returned *set* is right and the *order* is not its spectrum.

**A block's worst error is grain-blind only as `k → 0`.** The time droop scales with the modal
stiffness, which *is* the grain. The first bar on that return was picked at a `2×2` block (2%) and
**failed at `4×4`** (7.8%) — the "a margin measured at one fixture is a claim about one fixture"
trap, met again. Asserted as a **collapse** instead: spread ÷ `(m_max/N)²` = 45.19 ± 2% over 20
fixtures, so the grain's share is set by how much of the *grid* the block occupies.

**The section's own first assertions were fixture bars** (amended the same day, second commit).
The crossing sweep kept a *private copy* of the grid list it swept and then asserted its own
results were non-empty, so every literal in its docstring described that copy. Sharing the lists as
module constants and widening them 4 grids -> 9 immediately found the rest: at `N=80`, 1 cent, the
closed form is 2.120 and the crossing 1.926, so a 0.19-mode deficit reads as a **whole mode lost**
and `|horizon - predicted| <= 1` fails. **The fix was to stop comparing through an integer**: the
crossing is now solved as a REAL mode index (`brentq`, the eigenvalue formula extends to real `m`),
which made the claims *stronger* — the diagonal crossing **IS** the closed form to 1e-12 over 108
fixtures, and `pitch_horizon`'s integer is exactly `floor(crossing)`, 324/324. A rate is asserted
where it exists (`gap*N` settles only past `N~2000`), not over the shipped grids.

`mode_family` gained `"axial_y"` → `(1,n)`; it had no spelling at all even though its docstring
told callers to ask for the two axial families separately. No `"axial_x"` alias — one concept, one
spelling. All of this is `tests/helpers.py`, so the frozen-analysis guard is untouched
([[analysis-freeze-state]]); promotion into `physsynth/analysis/` is still the open follow-on.
