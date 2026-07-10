# Jawari / buzzing bridge — the sitar & tanpura curved-bridge timbre

> **Status: BUILT & GREEN (2026-07-10).** No new core code. `make_jawari_string` + `jawari_barrier`
> in `tests/helpers.py`, `tests/test_jawari.py` (11 tests), `scripts/diagnose_jawari.py`. Full suite
> **723** green, ruff clean. Gates: lossless conservation **drift ~7e-13** through the sustained
> curved wrap (α∈{1,1.5,2}); the **static-equilibrium magnitude oracle carried to a curved α=1
> profile** holds the closed-form `S u* = (K/ρ)b` to **~2e-16** with a doubled-coupling negative
> control that blows the drift up >1e4×; jawari-specific signatures — buzz brightens the tone ~2.7×
> over a clean string, brightness is **sustained** (late-window centroid ~3× the clean string's,
> hasn't collapsed to the fundamental), and the **wrap edge travels** ~2× wider on the curve than on
> a flat rail at matched clearance.

## This is a configuration of model #8, not new physics

The jawari is a **curved barrier at the string termination** driving the already-built model #8
(`BarrierString`, `core/collision.py` — a string against a one-sided distributed nonlinear barrier
`b(x)`, energy-conserving discrete-gradient contact, vector Newton solve). Model #8's docstring and
plan already name "the tanpura/sitar *jawari* bridge" as its target. So this batch **realizes and
validates** that claim; it deliberately writes **zero** new core code. The honest framing (advisor):
don't manufacture a "model #9" to justify the work — the deliverable is the profile, the
discriminating validation, and the diagnostic.

## The bridge profile

A downward-opening parabola tangent to the rest line at the fixed termination, `-inf` (out of
support) beyond the bridge span (`jawari_barrier`):

```
b(x) = -clearance - depth·(x/d)²     for  0 < x ≤ d = width_frac·L ,   -inf  elsewhere
```

The crest (nearest the string) sits at the termination side; the surface curves away by `depth` at
the far edge. Defaults: `width_frac = 0.15`, `depth = 1 mm`, `K = 2·10⁶` (stiff wood/bone),
`N = 100` (support ≈ 15 nodes — resolves the wrap, **well under the dense-solve BLAS cliff at
n≥100**), `clearance = 0` (grazing crest). `clearance < 0` preloads the crest above rest so the
**whole** span contacts at rest — the static-oracle case.

**The one geometry lesson (tuning):** `depth` must be **comparable to the near-termination
downswing**, not larger. The downswing at the bridge's far edge is `≈ amp·π·width_frac`
(≈ 3.8 mm for `amp = 8 mm`); with `depth ≫ that` the string only grazes the crest and the bridge
acts like a point contact (no wrap, no shimmer — the first prototype failed exactly this way, the
flat rail out-brightened it). With `depth ≈ 1 mm` the string wraps a wide span and the departure
point travels.

## Why the jawari is *not* the fret buzz already tested

Model #8's `test_collision_signature.py` tests a **flat rail / point fret**: localised,
**intermittent** slap. The jawari is the physical opposite — a **persistent, travelling wrap** on a
curve → **sustained** high-partial energy. So the new tests must *separate the curve from the flat
rail*, not re-run the intermittency signatures. The two discriminating signatures:

1. **Sustained brightness (shimmer).** A clean string's midpoint pickup stays near the fundamental
   throughout; the jawari's curved contact re-injects highs on every downswing, so its late-window
   spectral centroid is ~3× the clean string's and has not collapsed toward its own fundamental. The
   late/early ratio is a fragile secondary metric (depends on decay/window) — the **absolute
   late-window elevation** is the robust gate.
2. **Travelling contact point.** The furthest-in-contact node (the wrap/departure edge) sweeps the
   whole bridge on the curve (std ≈ 4.8 nodes) but stays a pinned cluster at the far edge on a flat
   rail at the same minimum clearance (std ≈ 2.0). The spread is the discriminator — both buzz, only
   the curve *travels*.

## The two machine-precision money gates (carried over from #8, re-exercised on the curve)

- **Lossless energy conservation through the sustained curved wrap** (σ = λ_h = 0): the earlier
  tests used a flat rail / a point fret, which never put many nodes in persistent simultaneous
  contact. Drift stays ~7e-13 — the vector discrete-gradient telescoping handles the jawari regime.
- **Static-equilibrium magnitude oracle on a *curved* α=1 profile.** With the crest preloaded above
  rest the whole curved span is in gentle contact, so the discrete gradient hits its no-warp Taylor
  branch and the scheme's discrete fixed point equals the continuous augmented equilibrium
  `S u* = (K/ρ)b` (mask = the bridge support, `b` = the *curved* heights) to ~2e-16. This pins the
  coupling magnitude for the curved profile specifically — a flat-only test would not. Negative
  control: doubling `G` and `force_pref` moves the true fixed point → drift blows up >1e4×.

## Known limitation & the natural follow-on

- **Sub-grid contact.** The travelling departure point hops node-to-node on the grid (model #8's
  documented limitation). Fine for the signature; a sub-grid contact interpolation would smooth it.
- **The tanpura cotton thread (*juari*).** The tunable point contact (a thread laid on the bridge)
  that gives the tanpura its extra buzz is just **one more barrier node** at a chosen position —
  the obvious next deliverable, kept out of this batch (advisor).
- **Jawari + sympathetics.** Composing this curved-bridge string with `SympatheticStrings` (the
  linear coupled-string family) is where the full sitar/tanpura instrument lives.
