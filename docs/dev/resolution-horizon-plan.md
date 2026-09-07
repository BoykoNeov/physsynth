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
  **Done 2026-09-07 — and this bullet was wrong twice.** It was not mechanical, it was not
  "several suites", and the example it names is not a pitch band at all. §7 is the audit.
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

---

## 7. The audit of the hand-picked bands — 2026-09-07

§6 promised a mechanical sweep converting every hand-picked assertion band into a derived one.
The sweep does not exist, because **most of those bands are not bounded by pitch**. What the batch
actually produced is a limiter for each one, written into the test beside the number.

### 7.1 §6 named the wrong example, and deriving it would have made the test worse

`test_damped_string.py::test_sigma1_makes_high_partials_die_faster` asserts over `[1..16]`, and §6
cited it as the exemplar. It is a **decay-rate** band: the claim is that the per-mode damping rate
*rises* with `σ₁ > 0` and *falls* without it, and the test's own comment already names its limiter —
"the rate turns over past ~m=32". That turnover is a numerator `~p²` against a θ denominator `~p⁴`;
it is not the pitch horizon, which at that fixture (`N = 128`, `λ = 1`) is **5** modes at 5 cents
and **2** at one cent. Deriving the band from `pitch_horizon` would have misattributed the limiter
*and shrunk the range from 16 to 5* — a green test covering less, which is the failure mode §6 was
itself worried about.

The number is also a caution about reading the wrong row of §3: the **10** that looks like the
answer here is that string's *space floor*, what it would resolve as `k → 0`. The fixture runs at
`λ = 1`, well short of it. **A floor is not a horizon**, and confusing the two overstates a
fixture by a factor of two.

And the band is not merely un-derivable, it is *asserting past* where the scheme is in tune: mode
16 of that fixture is **45 cents flat** and mode 32 — the turnover the test's comment names — is
**240 cents flat**, two semitones. That is not a defect in the test, because a decay-rate
monotonicity claim does not need the mode to be at the right pitch. It is the sharpest illustration
of §7.2's point: what a band is allowed to reach depends entirely on what is being claimed about
the modes inside it.

The generalisable point: §4's finding that the θ-scheme's rate error and pitch error are **one
factor and its square root** makes the two quantities easy to conflate. They are locked together
*for a single mode*; they are not the same *band*, because a rate claim can be about monotonicity
where a pitch claim is about absolute error.

### 7.2 There are three kinds of reference, and only one of them has a horizon

This is what collapses the sweep. A band's limiter follows from what the test compares against:

* **The continuum**, with a scheme that is *not* exact — the discrete answer really is flat, and
  the band is the pitch horizon. **Two tests in the whole suite.**
* **The scheme's own discrete oracle** — the dispersion is already inside the reference, so there
  is no continuum in the comparison to be flat against and the horizon is irrelevant *by
  construction*. The limiter is detectability: whether the n-th partial is separable from the noise
  floor of a pluck whose amplitudes fall like `1/n²`. **Four rows of §7.3's table, eight tests**
  — the last row bundles four single-mode FFT sanity checks that share this limiter.
* **The continuum of a different shape** — the circular membrane and the guitar outline, where the
  staircased boundary dominates and a cents reading mixes two errors. §5 already refused these.

And a fourth exclusion that turns out to be the largest: **`λ = 1` disqualifies more tests than
dispersion does.** §2's cancellation means the explicit family at the magic Courant number has *no
horizon to read* — every mode up to the grid is in tune — so the ideal string's and the bore's
continuum comparisons are exact by construction and their bands are about something else entirely.

### 7.3 The table

Thirteen rows covering **sixteen tests** — the last row bundles four single-mode FFT checks, and
the beam's row carries two bounds.

| test | band | reference | limiter | derived |
|---|---|---|---|---|
| `test_stiff_string.py::test_discrete_oracle_converges_to_continuum_stretched_law` | 10 -> **48** | continuum stretched law | **pitch horizon** | **yes** |
| `test_beam_modal.py::test_modal_frequencies_match_closed_form` | 1 @ 0.5c -> **2**; 4 @ 2c -> **6** | `cos βL cosh βL = 1` | **pitch horizon** | **yes** |
| `test_plate_modal.py::test_low_modes_within_one_cent` | 4 `(m,n)` pairs | continuum plate | pitch — but **two mode families in one list** | **blocked**, §7.4 |
| `test_modal.py::test_partials_within_one_cent_at_lambda_one` | 10 | continuum harmonics | scheme **exact** at `λ=1`; detectability | no |
| `test_bore_modal.py::test_open_open_full_series_present` | 4 | continuum | exact at `λ=1`; the claim is *presence* of the evens | no |
| `test_bore_modal.py::test_discrete_equals_continuum_at_lambda_one` | 6 | continuum | exactness **is** the subject | no |
| `test_stiff_string.py::test_partials_match_discrete_oracle` | 8 | own discrete oracle | detectability | no |
| `test_damped_string.py::test_partials_unmoved_by_light_damping` | 8 | own discrete oracle | detectability | no |
| `test_bore_modal.py::test_oracle_tracks_measured_spectrum` | 5 | own discrete oracle | detectability | no |
| `test_plate_modal.py::test_low_spectrum_via_eigsh_matches_oracle` | 6 | own discrete oracle | detectability + `eigsh` cost | no |
| `test_membrane_modal.py::test_circle_low_spectrum_tracks_bessel` | 8 @ 20c | Bessel continuum | **staircased domain**, not dispersion | no |
| `test_damped_string.py::test_sigma1_makes_high_partials_die_faster` | `[1..16]` | — (decay rates) | rate turnover at `~m=32` | no, §7.1 |
| four single-mode FFT sanity checks (beam, plate, free plate, orthotropic plate) | 1 | own oracle | one mode by design | no |

**Every literal was conservative, never over-claiming.** 10 against a real 48, 4 against 6, 1
against 2. The audit found no band asserting past where its scheme is in tune — which is the
reassuring half of the answer, and worth stating because the batch was scoped on the suspicion
that some band somewhere was.

### 7.4 The 2-D plate is the one genuine candidate that could not be built

`test_plate_modal.py::test_low_modes_within_one_cent` compares the discrete plate against the
continuum over `[(1,1), (2,1), (1,2), (2,2)]` — a real pitch band. `pitch_horizon` counts a leading
**prefix of one mode family**, and those four are two families at once: `(2,1)`/`(1,2)` are axial,
`(2,2)` is diagonal. §4 measured those families **a factor of nine apart** on the membrane at the
same Courant number, so a prefix over a mixed, sorted list is meaningless. Splitting the 2-D
spectrum by family is its own piece of work and is the honest successor to this batch.

### 7.5 A derived band without a floor asserts *less* than the literal it replaced

The trap, and the reason both rewrites keep their old number. If the horizon is measured from the
same two arrays the test then asserts over, `assert max(err[:horizon]) < bound` is a **tautology** —
it passes for any horizon at all, including 1. The information lives entirely in the comparison
against the old literal:

```python
horizon, monotone = pitch_horizon(measured, oracle, bound)
assert monotone                      # the prefix is not hiding a mode
assert horizon < window              # the number is measured, not truncated by the window
assert horizon >= floor              # <- the only line that can fail on a regression
```

Without the floor, a wrong "fix" to a dispersion oracle would shorten the band and **hide inside
it**, staying green on fewer modes. The floor is also what answers §6's own objection that
rewriting a range while introducing the primitive makes a failure ambiguous: a failure is now
either "the horizon shrank below what this fixture used to claim" or "the window truncated it",
and those are different messages.

`window < horizon` matters for the same reason from the other side: `pitch_horizon` returns the
array length when nothing is outside the bound, so a horizon equal to the window is a **lower
bound wearing a measurement's clothes**. Both rewrites assert it strictly inside.

### 7.6 What the two rewrites measure

* **Stiff string**, `N = 4000`, `fs = 800 kHz`: the discrete oracle tracks the continuum stretched
  law to a cent for **48** partials. Its space floor at that grid is **108**, so this fixture is
  **time**-limited and refining `fs` would buy more — the θ family behaving exactly as §3 says.
* **Free-free beam**, `N = 200`, `μ = 0.5`: **2** modes inside 0.5 cents, **6** inside 2 cents,
  both monotone. The fundamental is 0.18 cents flat and mode 6 is 1.95 cents flat.

Neither number is frozen anywhere. They are in comments as *what was measured on 2026-09-07*, and
the tests assert only the floors — the same discipline `test_resolution_horizon.py` states in its
own module docstring: a horizon recorded for one fixture is a claim about that fixture.

### 7.7 `spatial_operator_horizon` answers about one string, whatever you hand it

A tooling hazard found while scoping. The helper takes `(N, kappa, cents)` and looks general, but
it hardcodes `L_DEFAULT` and `wave_speed()` and builds a 1-D **Dirichlet-axis** second difference.
Called for the beam it would have returned a number about the *string* — silently, since nothing
in the signature carries a geometry or a boundary condition. Both rewrites therefore call
`pitch_horizon` on the two arrays the test already builds, and `spatial_operator_horizon` is used
only where the fixture genuinely is that string (the stiff string's floor of 108, §7.6).
