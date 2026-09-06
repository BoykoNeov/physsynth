# The resolution horizon — where each scheme stops being in tune

Opened 2026-09-06 on the human's call, directly out of `docs/dev/theta-loss-compensation-plan.md`.
That probe was supposed to justify a loss fix; what it actually found was that **the pitch error
bounds every accuracy claim this project makes about a mode**, and that no model had that boundary
written down. This is the boundary, measured, per family.

Code: `tests/helpers.py` (`pitch_error_cents`, `pitch_horizon`, `spatial_operator_horizon`) and
`tests/test_resolution_horizon.py` (32 tests). Probe scripts:
`M:\claud_projects\temp\theta-loss\probe_horizon2.py`, `probe_inventory.py`, `probe_checks.py`.

---

## 1. The definition, and the one decision inside it

**The horizon is the number of leading modes whose discrete frequency is within a stated number of
cents of the continuum.** Cents rather than relative frequency because every threshold worth
arguing about is perceptual, and because it makes the previous batch's finding expressible in the
same unit: the θ-scheme's decay-rate suppression `S` and its pitch error are one quantity,
`cents = 600 log₂(S)`.

The decision is **prefix, not last-inside**. "How many modes from the first are all in tune" and
"the highest mode that happens to be in tune" differ exactly when the error curve is not monotone,
and this project has already been bitten by a non-monotone boundary once (`vk-newton-plan.md` §13,
the amplitude boundary that is not monotone in one cell). `pitch_horizon` therefore returns
**`(horizon, monotone)`** and a caller that collapses the pair back to the integer without reading
the flag is throwing away the thing that says the integer is hiding something — the same move as
`VKPlate` reporting *which* failure it had rather than only that it failed.

The bound defaults to **5 cents**, roughly a sustained tone's pitch JND. It is an argument, not a
law; every table below is reproducible at 25 and 100 cents and the shape does not change.

---

## 2. Two mechanisms, and a reader must not assume the first is universal

This is the batch's structural finding, and the first draft of it was wrong. The horizon is **not**
`min(time floor, space floor)`. Whether the time error and the space error add or cancel depends on
their *signs*, and the two scheme families here have opposite answers.

| | time factor | space factor | together |
|---|---|---|---|
| **implicit θ-scheme** | `1/√(1 + θk²Q)` — **flat** | `sinc(u)` — **flat** | they **compound**; a hard floor |
| **explicit leapfrog** | `sin(ωk/2) = λ sin(u)` — **sharp** | `sinc(u)` — flat | they **cancel** at one λ |

`u = mπ/2N` throughout. The consequence is that "refine the timestep" is good advice for one family
and actively wrong for the other, which is the finding most likely to change what somebody does.

---

## 3. Family 1 — the implicit θ-scheme: there is a floor, and it arrives early

**Stiff string, damped string, both plates, the beam, and the VK plate's linear part.**

Refining `k` at fixed `N` buys modes and then stops. Measured on the suite's own damped string at
`N = 256`, 5 cents:

| λ | fs (Hz) | κ = 0 | κ = 2 | κ = 8 |
|---|---|---|---|---|
| 2 | 25,600 | 6 | 6 | 5 |
| 1 | 51,200 | 11 | 11 | 8 |
| 0.5 | 102,400 | 17 | 15 | 11 |
| 0.125 | 409,600 | 21 | 18 | 15 |
| 0.03125 | 1,638,400 | 21 | 19 | 15 |
| **space floor (`k → 0`)** | — | **21** | **19** | **15** |

A 64× increase in sample rate past the canonical `λ = 1` roughly doubles the horizon and then buys
nothing at all. **At `λ = 1` the string resolves about 4% of its own grid to 5 cents** — eleven
partials out of 255.

### 3.1 The floor is a fraction of the GRID, and it has a closed form

The space floor contains no `k`, no `c`, no `L` and no `fs`. It is the second difference's own
eigenvalue error, so the frequency ratio is `sinc(u)` and the horizon solves

```
sin(u)/u = 2^(−cents/1200),          m*/N = 2u/π
```

| bound | `m*/N` (closed form) | measured at N = 512 |
|---|---|---|
| 5 cents | 0.083779 | 0.0820 |
| 25 cents | 0.187120 | 0.1855 |
| 100 cents | 0.372613 | — |

`test_the_wave_space_floor_matches_its_closed_form` asserts this to within the integer
quantisation, over five grid sizes and three bounds. It is the only bar in the file that checks the
measurement against something outside itself, and it is analytic rather than recorded.

**Stiffness lowers the fraction and keeps lowering it under refinement** — 0.0820 → 0.0664 → 0.0586
for `κ = 0, 2, 8` at `N = 512` — because the frequency carries `√(c²p² + κ²p⁴)` and the quartic term
has twice the droop. At fixed `L` a finer grid admits higher `p²`, where that term dominates, so the
resolved *share* of a stiff string shrinks as you refine it. Both directions are asserted; neither
number is.

### 3.2 The plate is space-limited at every sample rate the suite uses

A plate's frequency is proportional to the Laplacian eigenvalue rather than its square root, so it
carries **twice** a string's droop. Simply-supported rectangle, diagonal modes, 5 cents:

| N | μ = 2 (fs) | m\* | μ = 0.05 (fs) | m\* | space floor |
|---|---|---|---|---|---|
| 16 | 2,560 | 0 | 102,400 | 0 | 0 |
| 32 | 10,240 | 1 | 409,600 | 1 | 1 |
| 64 | 40,960 | 3 | 1,638,400 | 3 | 3 |
| 128 | 163,840 | 6 | 6,553,600 | 7 | 7 |

**A 40× change of sample rate moves it by at most one mode.** `μ` is not the knob.

The `N = 16` zero deserves its actual number rather than a bare "nothing is in tune": at that
fixture the **fundamental is 9.6 cents flat** and the (2,2) mode is **83 cents flat**. So the claim
survives a 10-cent bound with exactly one mode passing, and it is not a threshold artifact.

The free-free beam is the same fourth-power story and adds a row rather than a finding: space floor
0, 2, 6, 13 as `N` goes 32 → 64 → 128 → 256, with `μ` again irrelevant.

---

## 4. Family 2 — the explicit leapfrog: the errors cancel, and in 2-D only along the diagonal

**Ideal string, membrane.**

At `λ = 1` the ideal string resolves **255 of 255** modes — more than **twelve times** its own space
floor of 21. That is the proof that the two families are genuinely different: an implicit scheme can
never pass the space floor, and this one passes it by an order of magnitude, because the time error
is sharp exactly where the space error is flat.

Below `λ = 1` the cancellation degrades, and **refining the timestep makes it worse**:

| λ | 1.0 | 0.99 | 0.95 | 0.9 | 0.75 | 0.5 |
|---|---|---|---|---|---|---|
| m\* (N = 256) | 255 | 131 | 66 | 48 | 32 | 24 |

A user who lowers the Courant number "for safety" is buying nothing and paying pitch accuracy for
it — the opposite of the advice that is correct for family 1.

### 4.1 The check that stopped a wrong headline: the 2-D cancellation is DIAGONAL-ONLY

The membrane at the 2-D Courant ceiling `λ = 1/√2` looks, on the diagonal, like the string at
`λ = 1`. It is not a property of the model. Measured at `N = 128`, 5 cents:

| λ | fs (Hz) | (m,m) diagonal | (m,1) axial | (m,2) |
|---|---|---|---|---|
| **0.7071** (ceiling) | 36,204 | **127** | **15** | 15 |
| 0.7 (suite default) | 36,571 | 65 | 15 | 15 |
| 0.6 | 42,667 | 20 | 13 | 13 |
| 0.5 | 51,200 | 15 | 12 | 12 |
| 0.25 | 102,400 | 11 | 11 | 11 |
| 0.125 | 204,800 | 10 | 10 | 10 |

The ceiling buys the diagonal family the entire grid and the axial family **nothing**. Reading only
the diagonal would have produced "the membrane is in tune at the CFL ceiling", which is a claim
about one mode family — this project's oldest recurring scar (a margin measured at one fixture is a
claim about one fixture) in a new place. The honest statement is that the membrane's *usable*
horizon is the axial one, about 12% of the grid, at every Courant number.

Both families converge to the same 1-D space floor (10 at `N = 128`) as `λ → 0`, which is the
sentence that ties the two families together: **the space floor is where every scheme ends up, and
what distinguishes a family is only whether a Courant number exists that beats it.**

---

## 5. The inventory — what has a horizon, and what does not yet

| model | family | horizon | note |
|---|---|---|---|
| ideal string | explicit | **measured** | exact at `λ = 1`; collapses below it |
| stiff / damped string | θ | **measured** | floor ≈ 8% of grid (κ=0), 6% (κ=8) |
| rectangular membrane | explicit | **measured** | diagonal-only cancellation; axial ≈ 12% |
| plate, simply supported | θ | **measured** | space-limited; ≈ 5% of grid |
| free-free beam | θ | **measured** | same fourth-power story as the plate |
| orthotropic plate | θ | *not measured* | same family and mechanism; a grain rotates the eigenvalues, so the horizon is per-direction like the membrane's — the shape of the answer is known, the numbers are not |
| circular membrane, guitar plate | — | **no horizon yet** | the error is the **staircased domain**, not dispersion: the continuum reference is a frequency for a *different shape*, so a cents comparison mixes two errors. Needs a geometry-convergence study, not this primitive |
| free plate (FFFF) | θ | **no horizon yet** | the continuum reference is a table of tabulated λ values (Narita/Leissa), not a formula over all modes, so a horizon exists only over the tabulated set |
| VK plate, tension string, geometric string | — | **no horizon yet** | no linear modal oracle; the horizon is a **refinement** horizon. One point is already measured: `vk-newton-plan.md` §13 finds a 3 cm strike's 48 kHz solution 37% away from its 96 kHz one at 83 µs, and that the curvature axis setting Picard's wall also sets this horizon |
| bore | explicit | **no horizon yet** | a reflection oracle exists, but the question is the Webster area function's resolution, which is a different comparison |
| air box | explicit | partly, elsewhere | §8 already records the defective corner mode at the 3-D CFL ceiling, and the lattice light cone is in the viewer's batch-18 record. 3-D dispersion is direction-dependent for the same reason the membrane's is |
| bow, reed, mallet, collision, barrier | — | inherited | exciters, not resonators; they inherit the horizon of whatever they drive |

---

## 6. Deliberately not done, and the named follow-ons

* **No existing test's assertion range was touched.** Several suites pin modal frequencies over a
  hand-picked band (`[1..16]` in the damped string's, and its siblings elsewhere) and those ranges
  are now *derivable* from the horizon. Rewriting them while introducing the primitive that computes
  them would make any failure ambiguous. That is a separate, mechanical, individually verifiable
  batch and it is the obvious next one.
* **The primitive stayed in `tests/helpers.py` and did not go into `physsynth/analysis/`.** A new
  public analysis name trips `test_analysis_frozen.py`'s derived guard, and satisfying it is
  impossible — there is no Python implementation left to freeze against. Widening that guard to
  accommodate the first thing that does not fit would change the enforcement contract for 62
  fixtures in order to add one function. Promotion is a deliberate follow-on that has to answer the
  freeze question on its own terms: a native bar in `crates/physsynth-analysis/tests/` plus a stated
  guard amendment. Note that it would also make `scipy.optimize.brentq` a question — fine in a test,
  a dependency decision in the analysis crate, whose allowlist is deliberately narrow.
* **No user-facing surface.** A viewer read-out saying "this configuration is trustworthy to 2 kHz"
  is the obvious product of this, and it needs the promotion above first.
* **The four "no horizon yet" rows were not attempted.** Each is a refinement or geometry study in
  its own right; the inventory names the reason rather than leaving a blank.
