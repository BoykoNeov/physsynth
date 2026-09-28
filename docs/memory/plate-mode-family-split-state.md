---
name: plate-mode-family-split-state
description: "Splitting the plate's 2-D spectrum by mode family (2026-09-07) — the blocked band was derivable after all, and the refusal's stated reason was BACKWARDS: the two families are identical in index and differ by exactly 2 in pitch"
metadata: 
  node_type: memory
  type: project
  originSessionId: 76c0a932-2af1-4049-aae0-3f756cb974ef
  modified: 2026-09-07T05:22:17.336Z
---

[[horizon-bands-audit-state]]'s one left-open item, built 2026-09-07 on the human's call:
`docs/dev/resolution-horizon-plan.md` **§8** (the record),
`M:\claud_projects\physical synthesis\tests\helpers.py` (`sinc_horizon_fraction`, `mode_family`,
`mode_block`), `M:\claud_projects\physical synthesis\tests\test_resolution_horizon.py` (32 → **79**
tests, still under a second), `M:\claud_projects\physical synthesis\tests\test_plate_modal.py`
(the derived band). Probes: `W:\temp\claude\plate-family\`.

**The refusal was right and its stated reason was wrong.** §7.4 predicted the plate's two mode
families would sit "a factor of nine apart", by analogy with [[resolution-horizon-state]]'s
membrane. They are **identical** — same integer horizon at every `N` and every bound, as
`k → 0` (asserted over four decades of `μ`, not at one convenient value; the closest the
shared horizon comes to an integer boundary is 0.044 cents, 4% of the bound). The factor of
nine is not a property of 2-D spectra at all; it is the *explicit* scheme's cancellation at its
Courant ceiling, and the implicit plate has no ceiling to cancel at, so both families just sit on
the same `sinc²` space floor. **Do not carry a finding across scheme families by analogy — the
mechanism, not the dimensionality, decides.**

**What actually blocks a prefix over a mixed 2-D list: it is NOT MONOTONE.** Sorted by frequency,
`(3,1)` is *lower* in pitch than `(2,3)` and *further* out of tune, because the droop weight
`w(m,n) = (m⁴+n⁴)/(m²+n²)` orders differently from `m²+n²`. `pitch_horizon` still returns an
integer over such a list (4, the right answer by luck at that fixture); `monotone=False` is the
only thing saying it means nothing. This is the flag's designed case arriving in the wild.

**Five things worth carrying:**

1. **A band is usually a BLOCK, not a family — and a block IS readable, through its corner.**
   "The first few modes are in tune" asserts over `{1..M}²`. `w` has an interior minimum in `n` at
   `n ≈ 0.644 m`, so its max over a block sits at a **corner**, and the *diagonal* corner beats the
   axial one for every `M ≥ 2`. So a block's horizon is its diagonal family's horizon, which is a
   family and does have a prefix. Assert that licence, don't assume it.
2. **The plate's space floor is the string's at HALF the cents budget** — exactly, not
   approximately: `sinc(u)² = 2^(−c/1200)` *is* `sinc(u) = 2^(−(c/2)/1200)`. **5.925% of the grid**
   at 5 cents against a string's 8.378%. That is what "twice the droop" means as a number, and the
   diagonal family's eigenvalue ratio really is `sinc(u)²` to **2.2e-16** (the axial one only
   approaches it, closing like `1/N²`). `power` in `sinc_horizon_fraction` is read off the
   dispersion relation — 1 where `ω ~ p`, 2 where `ω ~ p²` — never chosen.
3. **A horizon in MODE INDEX is family-independent; a horizon in HERTZ is not.** At the same
   frequency an axial mode is **exactly twice** as flat (`w` is `ρ⁴` on the axis and `ρ⁴/2` on the
   diagonal). So "in tune to 5 cents up to 2 kHz" is a claim about a *direction* — on the axis the
   same 5 cents arrives at `1/√2` of that frequency. This is the sentence a future viewer read-out
   has to get right.
4. **A finite timestep breaks the family tie, toward the AXIAL family.** The index agreement is a
   `k → 0` claim. At index `m` the axial mode sits at half the diagonal's frequency and takes half
   the time droop, so at `N=512, μ=2, 25 cents` it is 54 against 44. Measure the ratio at the `μ`
   you intend to quote it at — at `μ→0` the pitch factor is 1.99, at `μ=2` it collapses to 1.26.
5. **The derived band came out EXACTLY SATURATED**, the one place [[horizon-bands-audit-state]]'s
   "every literal was conservative" needed amending: horizon 2, band 2, and the `3×3` block is 1.41
   cents out. Zero headroom by construction — so the `assert horizon >= floor` line cannot be
   raised even by one mode.

**Two design rules obeyed, both from the previous batch's scars.** The helpers are **index-side
only** (they return `(m,n)` pairs and nothing else) so the caller keeps its own geometry — §7.7's
`spatial_operator_horizon` hazard. And the assertion **window comes from the closed form**
(`ceil(2 · fraction · N)`), not a hand-picked count, so a finer grid cannot quietly turn
`horizon < window` into `horizon == window` — commit `aabe966`'s lesson, applied before it could
happen. The design deliberately *not* built: partitioning a mixed spectrum by rays through the index
origin. It is exact and useless — every axial mode becomes its own singleton family.

Related: [[horizon-bands-audit-state]], [[resolution-horizon-state]], [[plate-state]],
[[membrane-state]], [[orthotropic-plate-state]], [[theta-loss-lock-state]].
