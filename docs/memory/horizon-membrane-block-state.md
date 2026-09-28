---
name: horizon-membrane-block-state
description: "The membrane's block (2026-09-07) — the CFL ceiling IS the spectrum's minimum cancellation Courant number, so no mode is ever sharp; a block's worst corner FLIPS below the ceiling, inverting the plate's rule"
metadata: 
  node_type: memory
  type: project
  originSessionId: 5288abdc-de93-415e-ac89-b7838eb5fff4
  modified: 2026-09-07T08:51:07.269Z
---

`docs/dev/resolution-horizon-plan.md` §10 — the row deferred twice by §8.9 and §9.8, built and
closed. Commit `486dffc`.

**The deferral's reason was wrong, for the third time in this plan.** Both bullets said a membrane
needs its own corner argument "because its weight is `√`-ed". The weight `(m⁴+n⁴)/(m²+n²)` is one
function both models read (`block_weight` in `tests/helpers.py`), and a monotone square root cannot
reorder a block — so in space the plate's rule transfers unchanged. §8.4 and §9.8's own amendment
had the same shape: **a claim about a 2-D spectrum reasoned from the shape of the error rather than
from which scheme produced it.**

**What actually breaks it is the term the implicit plate does not have.** The explicit leapfrog's
time error is *sharp*: `ω_disc/ω_cont = 1 + (a²/6)(λ²ρ² − w)`, `a = π/2N`. So every mode has its own
cancellation Courant number `λ_cancel = √(m⁴+n⁴)/(m²+n²)` (`cancellation_courant`), and:

- **the 2-D CFL ceiling IS the minimum of `λ_cancel` over the whole spectrum** — `t²+(1−t)²` with
  `t = m²/ρ²` is minimised at `t = 1/2`, the diagonal. The "magic" Courant number is not a
  coincidence landing on the stability bound: the stability bound and the best attainable tuning
  are the same worst-case over the same spectrum. **1-D is the same formula with the second axis
  dropped** (`cancellation_courant(m, 0) == 1.0` exactly), which is the mechanism under §4's
  "the string resolves 255 of 255" — in 1-D every mode attains the limit at once;
- **no mode is ever sharp on a stable membrane** (`λ ≤ 1/√2 ≤ λ_cancel` always), so there is no
  "the errors average out" reading available;
- **a block's worst corner FLIPS at `λ = 1/√(M²+1)`** — the two corners' errors equated collapse to
  `(M²−1)[λ² − 1/(M²+1)] = 0`. That threshold is *below* the ceiling for every `M ≥ 2` and falls
  like `1/M`, so at every Courant number a membrane is actually run at the worst mode is an **axial**
  corner. The plate's diagonal-corner rule is **inverted**, not weakened. `mode_block`'s docstring
  was stating the plate's answer as the general licence and is amended — which corner is worst is a
  property of the **scheme**, not of the block.

**§4.1's hand-measured "about 12% of the grid" is a closed form**: at the ceiling the axial family
resolves exactly `√2 × sinc_horizon_fraction(cents, 1) × N` (the ratio `√(48/24)` is exact in the
leading order). Measured `4, 7, 15, 30, 60` at `N = 32…512`, never above the bound.

## The reusable lessons

- **Exact and asymptotic claims must be asserted differently, and the diagonal is the exact one.**
  At the ceiling `λ√S = sin(u)` and the scheme's own `arcsin` undoes it — an **identity at every
  N**, not a limit. Its measured "residual" is `brentq`'s tolerance against an increasingly flat
  function and *grows* with the grid, so a convergence-rate assertion on it asserts the root finder.
  Off-diagonal residuals fall like `1/N²` (64.1× across an 8× grid step). Same distinction as §8.2.
- **A "the worst mode is a corner" claim needs a measured companion, not just the expansion.** The
  integer sweep maximises the leading-order proxy; the licence for the whole block reading deserved
  the model asked too, across a λ range spanning the flip so both corners are seen to win.
- **Knowing which corner is worst does not yet license a block horizon.** `pitch_horizon` returns a
  prefix, and a prefix over a non-monotone list is meaningless (§8.6) — so the corner's *family*
  must be shown monotone. Easy to skip, and without it the batch proves a corner and can quote no
  number.
- **Derived bars go beside the hand-picked ones, never instead of** (§7.5). "Essentially exact"
  (`≥ 0.9(N−1)`) became `== N−1`; `< 0.25(N−1)` became the `√2` closed form within one mode.
- The analytic frequency path (no model built) is **not** a refactor of the model-built helper: the
  round-trip through `fs = c/(λh)` is a different sequence of roundings and moves the last bits.
  Two paths, with the seam asserted — see [[resolution-horizon-state]] and [[membrane-state]].

**Not done:** no rectangular-membrane pitch band exists to derive (§7.3's only membrane row is the
circular head against Bessel, limited by the staircased domain); the circular membrane stays in the
"no horizon yet" group; everything here is `σ = 0`; promotion of the primitives out of `tests/`
remains open, and this batch adds two more (both pure arithmetic, no new dependency).
