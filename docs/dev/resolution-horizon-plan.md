# The resolution horizon — where each scheme stops being in tune

Opened 2026-09-06 on the human's call, directly out of `docs/dev/theta-loss-compensation-plan.md`.
That probe was supposed to justify a loss fix; what it actually found was that **the pitch error
bounds every accuracy claim this project makes about a mode**, and that no model had that boundary
written down. This is the boundary, measured, per family.

Code: `physsynth/analysis/horizon.py` over `crates/physsynth-analysis/src/horizon.rs`
(`pitch_error_cents`, `pitch_horizon`, `sinc_horizon_fraction`, `mode_family`, `mode_block`,
`cancellation_courant`, `block_weight`), with 27 native bars in
`crates/physsynth-analysis/tests/horizon.rs`; `spatial_operator_horizon` alone stays in
`tests/helpers.py` (§11). Tests: `tests/test_resolution_horizon.py` (48 functions, 188 collected). Probe scripts:
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

**Since §8 the plate's floor has the same closed form the string's does**, and "twice the droop"
is exact rather than a hand-wave: the plate's frequency is linear in the Laplacian eigenvalue, so
its droop is `sinc(u)²` where the string's is `sinc(u)`, and the horizon solves
`sinc(u)² = 2^(−cents/1200)`. That is the *same equation* as the string's at **half the cents
budget** — `5.925%` of the grid at 5 cents against a string's `8.378%`, a factor of `√2` in the
small-`u` limit. The ≈5% quoted above was measured at a fixture that also carries a timestep.

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
| rectangular membrane | explicit | **measured** | §10. Every mode has a Courant number at which its two errors cancel, `√(m⁴+n⁴)/(m²+n²)`, and the 2-D CFL ceiling **is** the minimum of that over the spectrum — attained by the diagonal family and nothing else, which is why the cancellation is diagonal-only and why no mode is ever sharp. The axial family at the ceiling resolves exactly `√2` times the string's space floor (`0.1185 N` at 5 cents, §4.1's "about 12%" derived). A **block**'s worst mode is a corner, but *which* corner flips at `λ = 1/√(M²+1)` — below the ceiling for every block, so at every Courant number a membrane is run at the plate's diagonal-corner rule is **inverted**, not weakened |
| plate, simply supported | θ | **measured** | space-limited; **5.925% of grid** at 5 cents, closed form in §8; as `k → 0` the two mode families agree in index (a finite timestep breaks the tie toward the axial one) and differ by exactly 2 in pitch |
| free-free beam | θ | **measured** | same fourth-power story as the plate |
| orthotropic plate | θ | **measured** | §9. In **mode index** there is one floor and it is the isotropic plate's — `sinc(u)²`, exactly on the diagonal for *any* grain, within a mode on both axial families for every grain tried. In **hertz** it is per-direction, because the same index is a different frequency on each axis (`√(g_x/g_y)` apart, 3.7× for spruce). `axial` did split in two, but into two *deviations from one floor* rather than two horizons — and the soft one crosses the closed form, so `horizon ≤ predicted` is an isotropic bar |
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
* **The primitive stayed in `tests/helpers.py` and did not go into `physsynth/analysis/`.**
  **Done 2026-09-07 — §11**, and *both* of this bullet's reasons were wrong.
  `scipy.optimize.brentq` is not a dependency decision: the analysis crate already
  `#[path]`-includes `physsynth-core/src/root.rs`, a transcription of SciPy's `brentq.c`, and its
  allowlist stayed empty. And freezing was not impossible — that generalised from the modules
  *deleted* in rust-migration §44, where no Python survived, whereas these seven were *promoted*
  and their Python bodies were live until the commit that replaced them. They were recorded first,
  so no guard amendment was needed and the contract over the other 62 fixtures is untouched. What
  this bullet got right is that a native bar is wanted as well, and there are 27 of them.
* **No user-facing surface.** A viewer read-out saying "this configuration is trustworthy to 2 kHz"
  is the obvious product of this. The promotion it was waiting on is done (§11), so this is now the
  next thing in the sequence rather than a thing behind another thing.
  **Done 2026-09-07 — §12.** Two things this bullet did not see: the claim needs *two* readings
  because hertz is family-dependent and a mode index is not, and most of the viewer's models have
  to refuse rather than answer.
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
| `test_plate_modal.py::test_low_modes_within_one_cent` | 4 `(m,n)` pairs = the `2×2` block | continuum plate | pitch — a **block**, read through its diagonal corner | **yes**, §8 |
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

**Amended by §8: the plate's is not conservative, it is exactly saturated.** Its band is the
`2×2` index block and its measured horizon is `2`; the `3×3` block is 1.41 cents out. So that one
literal had *zero* headroom, and it is the only derived band here whose floor cannot be raised by
even one mode. Still not over-claiming — but the blanket sentence above was written before the
number could be measured.

### 7.4 The 2-D plate is the one genuine candidate that could not be built

`test_plate_modal.py::test_low_modes_within_one_cent` compares the discrete plate against the
continuum over `[(1,1), (2,1), (1,2), (2,2)]` — a real pitch band. `pitch_horizon` counts a leading
**prefix of one mode family**, and those four are two families at once: `(2,1)`/`(1,2)` are axial,
`(2,2)` is diagonal. §4 measured those families **a factor of nine apart** on the membrane at the
same Courant number, so a prefix over a mixed, sorted list is meaningless. Splitting the 2-D
spectrum by family is its own piece of work and is the honest successor to this batch.

**Done 2026-09-07 — §8. The refusal was right and its stated reason was wrong.** A prefix over
that sorted list really is meaningless, but not because the families are far apart: on the plate
they are **identical** in mode index, at every grid and every bound. The factor of nine is a
property of the *explicit* family's cancellation at a Courant ceiling, and the implicit plate has
no ceiling to cancel at. What actually breaks the prefix is that the frequency-sorted union is
**not monotone in the error at all** — `(3,1)` is lower in pitch than `(2,3)` and further out of
tune.

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


---

## 8. Splitting the 2-D spectrum by mode family — 2026-09-07

§7.4's blocked row, built. The deliverable is two index-side helpers, seven new claims about the
plate, and one derived band; the surprise is that **§7.4 predicted the answer and got it backwards**.

### 8.1 The primitive is index-side, because §7.7 said what happens otherwise

`mode_family(kind, count)` returns `[(m, n)]` for `"axial"` (`(m,1)`) and `"diagonal"` (`(m,m)`);
`mode_block(m_max)` returns the whole `{1..M}²` index block ordered by continuum frequency. Both
return **mode numbers and nothing else**. The caller still builds its own discrete and continuum
frequencies from its own fixture, which is exactly the shape §7.7 found `spatial_operator_horizon`
had got wrong — a helper that takes `(N, mu)` and returns "the horizon" hides the geometry and the
boundary condition in its body and will answer about the wrong model without saying so.

The design that was *not* built: partitioning a mixed spectrum into rays through the index origin.
It is exact — every `(m,n)` lies on one ray — and useless, because it makes every axial mode
`(2,1)`, `(3,1)`, `(4,1)` its own singleton family. "Split by family" is satisfied by enumerating
families, not by partitioning a list.

`sinc_horizon_fraction(cents, power)` moved out of the test file into `helpers.py` and gained the
`power` argument. `power` is read off the dispersion relation, not chosen: 1 where the frequency
goes like `p` (string, membrane), 2 where it goes like `p²` (plate, beam).

### 8.2 The diagonal family IS the string's droop squared — exactly

`Λ_disc / Λ_cont` along `(m,m)` is `sinc(u)²` to **2.2e-16**, because both axes carry the same `u`
and the weights cancel. The axial family only approaches it (`2.0e-4` at `N = 64`, `1.2e-5` at
`N = 256`, closing like `1/N²`) because `(m,1)` carries an undrooped `p₁²` in its numerator. That
is the mechanism behind everything below, and it is an identity, not a measurement.

### 8.3 The closed form, and it is the string's at half the cents budget

`sinc(u)² = 2^(−c/1200)` **is** `sinc(u) = 2^(−(c/2)/1200)`. So:

| bound | plate `m*/N` | = string at | string `m*/N` |
|---|---|---|---|
| 1 cent | 0.026500 | 0.5 cents | 0.037476 |
| 5 cents | 0.059250 | 2.5 cents | 0.083779 |
| 25 cents | 0.132409 | 12.5 cents | 0.187120 |

Asserted as an identity to `1e-12`, and the measured floor is asserted against it over
`N ∈ {64,128,256,512} × {1,5,25} cents × {axial, diagonal}` — 24 fixtures, within the one-mode
integer quantisation, and **never above** it, since a timestep can only cost modes.

### 8.4 §7.4's prediction, and why it was backwards

> §4 measured those families a factor of nine apart on the membrane at the same Courant number

They are **identical** on the plate — same integer horizon at every `N` and every bound, in the
`k → 0` limit. The membrane's factor of nine is not a property of 2-D spectra; it is the explicit
scheme's *cancellation*, which exists only at `λ = 1/√2` and only along the diagonal. The implicit
plate has no magic Courant number, so both families simply sit on the same `sinc²` floor.

A finite timestep **breaks the tie, toward the axial family**: at index `m` the axial mode sits at
about half the diagonal's frequency and so takes about half the time droop. At `N = 512`, `μ = 2`,
25 cents the split is 44 diagonal against 54 axial. So "the plate's horizon" quoted from one
family at a working sample rate is wrong by a few modes, in a known direction.

### 8.5 Where the families DO differ: a factor of exactly two, in pitch

The droop weight is `w(m,n) = (m⁴ + n⁴)/(m² + n²)`. At a fixed continuum frequency `ρ² = m² + n²`
that is maximal on the axis (`ρ⁴`) and minimal on the diagonal (`ρ⁴/2`) — **exactly two**,
approached from below as the axial mode's `n = 1` becomes negligible (1.968 at `m = 8`, 1.993 at
16, 1.998 at 24).

Which makes the unit matter. **A horizon quoted as a mode index is family-independent; a horizon
quoted in hertz is not.** "This plate is in tune to 5 cents up to 2 kHz" is a claim about the
diagonal; on the axis the same 5 cents arrives at `1/√2` of that frequency. That is the sentence a
future viewer read-out (§6's third bullet) has to get right.

### 8.6 The sorted union is not monotone, which is what actually blocked §7.4

Sorted by frequency, the plate's spectrum interleaves families and the error **falls** as pitch
rises: `(3,1)` at `ρ² = 10` is 1.27 cents out where `(2,3)` at `ρ² = 13` is 1.16, because `w`
orders differently from `ρ²`. `pitch_horizon` still returns an integer over such a list — 4 at one
cent on the rewritten test's own fixture, which is the right answer by luck — and `monotone=False` is the only thing that says the
integer means nothing. This is the case §1 built the flag for, arriving in the wild.

### 8.7 What makes a block readable: its worst mode is its diagonal corner

A "the first few modes are in tune" claim asserts over a **block**, not a family. `w` has an
interior minimum in `n` at `n = m√(√2 − 1) ≈ 0.644 m`, so its maximum over a block sits at a
corner; and `w(M,M) = M²` beats `w(M,1) = (M⁴+1)/(M²+1)` for every `M ≥ 2` (checked arithmetically
to `M = 199`, and measured for `M = 2..8` at two grids). **So a block's horizon is its diagonal
family's horizon**, which is a family and does have a prefix. That is the licence the rewritten
test rests on, and it is asserted rather than assumed.

### 8.8 The band, derived — and it is exactly saturated

`test_plate_modal.py::test_low_modes_within_one_cent`, `N = 96`, `μ = 0.5`, 1 cent. Both families'
horizons are **2**; the `2×2` block is in tune (worst mode `(2,2)`, 0.62 cents) and the `3×3`
block is not (`(3,3)`, 1.41 cents). So the four modes the test always asserted are exactly the
block the horizon licenses — no headroom in either direction, which is why §7.3's "every literal
was conservative" needed the amendment.

Following §7.5, the rewrite keeps the original `max(err) < 1.0` line and adds `monotone`,
`horizon < window` and `horizon >= 2`, the last being the only line a regression can fail. The
**window comes from the closed form** — `ceil(2 · fraction · N)` = 6 — rather than a hand-picked
count, so a finer grid cannot quietly turn `horizon < window` into `horizon == window`; that is
commit `aabe966`'s lesson (a widened window putting unchecked modes inside a boolean) applied
before it could happen.

### 8.9 Not done

* **The orthotropic plate. Done 2026-09-07 — §9**, and the prediction inside this bullet was
  half right: `(m,1)` and `(1,n)` do stop being degenerate and `axial` did become two
  measurements. What they are two of is not two horizons. Both sit on the isotropic plate's
  `sinc²` floor, and what splits is how far each *deviates* from it.
* **The membrane's block. Done 2026-09-07 — §10**, and the reason in this bullet is wrong: `w` is the
  *same* function for both models and a monotone square root cannot reorder a block, so in space the
  plate's rule transfers unchanged. What breaks it is the explicit scheme's *sharp* time error, and
  it inverts the rule rather than weakening it.
* **Promotion out of `tests/`. Done 2026-09-07 — §11**, and the `brentq` worry in this bullet was
  unfounded: the analysis crate already compiles a transcription of SciPy's own. Original text:
  `sinc_horizon_fraction` now brings a `brentq`
  with it, which is free in a test and a dependency decision in the analysis crate.


---

## 9. The plate with a grain — 2026-09-07

§5's inventory row predicted "the horizon is per-direction like the membrane's", and §8.9 predicted
that `axial` would become two measurements. The second is right and the first is right only about
the *units*: in mode index the grained plate has **one** floor and it is the isotropic plate's,
with no grain anywhere in the formula. What the grain actually costs is something §3 never had to
state, because on an isotropic plate it hides under the integer quantisation.

The model is `test_plate_orthotropic.py`'s: a square supported plate whose bending stiffness is
three ratios rather than one, with a closed-form continuum oracle
(`f ∝ √(g_x a² + 2 g_h ab + g_y b²)`, `a = (m/L_x)²`) and a matching discrete one built from the
two per-axis Dirichlet eigenvalues. Four fixtures throughout: **isotropic** (all ratios 1),
**spruce** (`1, 0.153, 0.0727`), **wild** (`11, 2.5, 0.9` — no material), and **near-guard**
(`1, -0.9, 1`), which sits just inside the constraint `g_h > -√(g_x g_y)` and is the only fixture
here that is not a plausible plate. It earns its place by being the only one that falsifies
anything: it flips two signs, and both flips are findings.

### 9.1 The primitive gained a family, and the missing name is deliberate

`mode_family` could return `(m,1)` and `(m,m)` and had no way to ask for `(1,n)` at all — its
docstring already told callers that on `Lx ≠ Ly` the two axial families are separate measurements
and to "ask for each separately", which was not possible. It is now `"axial_y"`. There is no
`"axial_x"` alias: one concept, one spelling, and every existing caller means the x-family.

### 9.2 The diagonal droop is `sinc(u)²` for *any* grain — exactly

On a square domain the diagonal mode `(m,m)` carries the same `u` on both axes, so every grain
weight multiplies the discrete and the continuum modal stiffness by the same factor and divides
straight back out. `q_disc/q_cont` is `sinc(u)⁴` whatever `(g_x, g_h, g_y)` are, so the frequency
ratio is `sinc(u)²` — §8.2's isotropic identity, to the last bit, measured at `< 1e-14` over four
grains and two grids.

This is a **pin, not a discovery**: it is exactly what a grain wired into the wrong axis, or a
cross term applied once instead of twice, would break. It survives even at `g_h = -0.9`, where the
continuum stiffness has nearly cancelled itself and a `0/0` would be a fair worry.

### 9.3 Where the grain does reach: the axial families, and the sign is the cross term's

`(m,1)` carries a drooped `λ_x` against an **undrooped** `λ_y`, so the weights no longer cancel
and the family only *approaches* `sinc²`. The sign of what is left is the sign of `g_h`, because
the cross term is the only one mixing a drooped axis with an undrooped one:

* `g_h > 0` — the family sits **above** `sinc²`: less flat, in tune slightly further than the
  isotropic closed form predicts. This is every real wood, and the isotropic plate too.
* `g_h < 0` — it sits **below**, and the closed form is conservative there.

The magnitude splits the two axes by an order of magnitude under a real grain: at `N = 512` the
deviation is `4.8e-7` along spruce's stiff axis against `6.7e-6` across it. That is what §8.9 meant
by "`axial` becomes two measurements" — two deviations from one floor, not two floors.

### 9.4 The floor is the isotropic closed form — which stops being an *upper* bound

`sinc_horizon_fraction(cents, 2) · N` was derived with no grain in it, and it is the grained
plate's floor too. The first version of this claim was "within one mode over 48 fixtures", which
turned out to be two mistakes at once — a bar on an *integer*, and a fixture list private to the
test that recorded it. Both are fixed and the claim is stronger for it:

* the **diagonal** crossing, solved for as a real mode index rather than counted, **is** the closed
  form — to `1e-12`, over 108 fixtures (4 grains × 3 bounds × 9 grids), for stiffness ratios
  spanning 150×;
* the **axial** crossings sit within **half a mode** of it everywhere (`−0.199` to `+0.399`), with
  the sign of `g_h`, closing like `1/N`. That rate is asymptotic: `gap · N` reaches its per-grain
  limit (18.9 isotropic, 39.7 spruce, 52.4 wild, −17.0 near-guard) only past `N ≈ 2000`, so it is
  measured where it exists rather than over the shipped grids, where the wild plate is still 20%
  short of its own limit.

**The integer reading is `floor` of that crossing** — 324 of 324 fixtures, no exceptions — so its
distance from the closed form is a real gap plus an artefact of where the prediction falls between
integers, and the artefact can dominate. At `N = 80` and one cent the prediction is **2.120** and
the near-guard plate's crossing is **1.926**: a 0.19-mode deficit reads as a whole mode lost, and
`|horizon − predicted| = 1.12`. The original four grids passed a `≤ 1` bar only because none of
them put a prediction that close above an integer. That hazard is now its own test rather than
something the fixture list steps around, and it is available to **any** horizon test in this
project that reads an integer.

The positive end is new. §8's plate test also asserts `horizon ≤ predicted`, on the argument that
a timestep can only cost modes and never buy them. That argument is sound and it is not the whole
story: by §9.3 an axial family's *space* floor is already a hair above `sinc²`, so its horizon
wants to sit above the closed form with no timestep involved at all. On an isotropic plate the
excess is under a third of a mode and the integer floor absorbs it — **the existing direction bar
passes on quantisation, not on its stated mechanism.** A grain roughly doubles the excess on the
soft axis and at coarse grids it clears the integer: **eight of the 108** cross, all of them the
`(1,n)` family of a positively-grained plate, at `N = 48, 64, 128`, by up to 0.304 modes. All of
them coarse, which is the mechanism — the gap closes like `1/N` while the prediction grows with
`N`, so refining always eventually buries it. So `horizon ≤ predicted` is a claim about an
isotropic plate specifically, and the grained plate is asserted with the symmetric bar.

### 9.5 Index versus hertz — where §5's "per-direction" is right

Both axial families have the same horizon in their own mode index, and index `m` is not the same
frequency on the two axes: `f(m,1)/f(1,m) → √(g_x/g_y)`, approached **from below** like `1/m²`
because the cross term still contributes at low index (`gap·m²` is constant to 0.5% over
`m = 10..80`). Spruce's is 3.709, and the residual is worth stating perceptually rather than as a
magnitude: the two axes are still 2.1 (spruce) and 2.8 (wild) cents from the limit at `m = 40`,
and inside a cent by `m = 80`. So **a horizon quoted as a mode count is grain-independent and
the same horizon quoted in hertz is not** — spruce's stiff axis is trustworthy 3.7× further up the
spectrum than its soft one. That is the sentence a viewer read-out would have to get right.

### 9.6 The block: the corner rule holds for every wood, and breaks at exactly `−1/m_max²`

§8.7 reads a block through its diagonal corner. With a grain the small-`u` error weight becomes

    W(m,n) = [g_x m⁶ + g_h m²n²(m²+n²) + g_y n⁶] / [g_x m⁴ + 2 g_h m²n² + g_y n⁴]

which collapses to §8.7's `(m⁴+n⁴)/(m²+n²)` at equal weights. `W(m,m) = m²` **exactly, for any
grain** — the same cancellation as §9.2 — while the axial corner stays below it. The grain does
narrow the margin: over a `3×3` block the stiff-axis corner climbs from 8.200 (isotropic) to 8.862
(spruce) against the corner's 9.

It fails for a **negative** cross term, and the threshold is not "negative": it is `−1/m_max²`,
exact to bisection at every block size measured (`−0.25, −1/9, −0.0625, −1/36, −0.015625` for
`m_max = 2,3,4,6,8`). It **tightens as the block grows**, so a large enough block breaks the rule
for any negative cross term at all, while a `2×2` tolerates `−0.25`. At `g_h = −0.9` the worst
mode of a `6×6` is `(4,6)`, 28% worse than the corner — reading that block through its corner
would understate its error by a quarter. `mode_block`'s docstring carries the condition now, along
with the warning that its **ordering key is the isotropic frequency** and is not a grained plate's
spectrum: on a grained plate the returned *set* is right and the *order* is not.

### 9.7 Grain-blind in space, not at a timestep — and the first bar was picked at one fixture

Combining §9.2 and §9.6: where the corner rule holds, a block's worst mode is its diagonal corner
and that corner's droop is grain-independent, so **a block's worst space error is the same number
on a grained plate as on an isotropic one**, bit for bit. `test_plate_modal.py`'s derived `2×2`
band therefore transfers to the orthotropic plate unchanged, including §8.8's exact saturation.

Only as `k → 0`. The time droop is `1/√(1 + θk²Q)` and `Q` is the modal stiffness, which *is* the
grain, so at a working timestep the grain returns through the other mechanism, ordered by
stiffness: the wild plate flattest, spruce sharpest. The first version of that test bounded the
return at 5%, measured at a `2×2` block where it is 2% — and it failed at `4×4`, where it is 7.8%.
This is the same "a margin measured at one fixture is a claim about one fixture" the migration kept
paying for, met again here. The honest form is a **collapse**: over 20 fixtures the relative spread
divided by `(m_max/N)²` is **45.19 ± 2%**, rising toward that limit as the grid refines. The grain's
share of a block's stepped error is set by how much of the *grid* the block occupies and by nothing
else, and the constant belongs to the fixture set rather than to the physics, so the test asserts
the collapse and not the number.

### 9.8 Not done

* **No band was derived here**, because the orthotropic module has none to derive: §7.3's audit
  files its only pitch assertion as a single-mode FFT check, and its two block-shaped tests
  (`test_the_cross_term_detunes_selectively_without_reordering_anything`,
  `test_the_grain_is_in_the_partial_series_and_not_in_the_level`) compare a grained *discrete*
  spectrum against an isotropic *discrete* one, so the limiter is detectability and not pitch —
  the "own discrete oracle" row of that table.
* **The free orthotropic plate (#5of) is untouched.** It is in §5's "no horizon yet" group for the
  free plate's reason — its reference is a set of probes rather than a formula over all modes —
  and the grain does not change that.
* **The membrane's block. Done 2026-09-07 — §10**, and this bullet repeated §8.9's wrong reason:
  the weight is not "the plate's, not a membrane's", it is one function both models read.
* **Promotion out of `tests/`. Done 2026-09-07 — §11.** Note that the continuous crossing
  introduced here brings a second `brentq` with it, so the dependency question §6 raises now has
  two call sites rather than one.

**Amended the same day, before the section was a day old.** Two of its first assertions were
fixture bars rather than claims — the `± 1` mode agreement (fixed above) and a magnitude bound on
the hertz residual at a single `m`. Both had the shape §9.7 records being caught by, in the same
batch that wrote §9.7. The general lesson is not "measure more fixtures": it is that a **sweep
whose fixture list lives inside the test that records its results cannot notice when the list is
the reason**. The lists are shared module constants now, and widening them from four grids to nine
is what surfaced the `N = 80` case.

---

## 10. The membrane's block — 2026-09-07

§8.9 and §9.8 deferred this row twice, with the same sentence both times:

> **The membrane's block.** §4 measured its families; nobody has asked a block question of it.
> The corner argument in §8.7 is the plate's weight `w`; a membrane's is `√`-ed and needs its own.

The row is built, and **the reason it gives is wrong**. The square root is real and it is not what
needs its own argument: `w` is the *same function* for both models, and a monotone square root
cannot reorder a block, so in space the plate's corner rule transfers to the membrane unchanged.
What breaks it is the term the implicit plate does not have — and it does not weaken the rule, it
**inverts** it, across the entire range of Courant numbers anyone runs a membrane at.

This is the third time a deferred prediction in this plan came out backwards (§8.4, §9.8's own
amendment, now this), and the three have the same shape: a claim about a *2-D spectrum* was
reasoned from the shape of the error rather than from which scheme produced it.

### 10.1 The one expansion the whole section rests on

For the explicit leapfrog, with `a = π/2N`, `ρ² = m² + n²` and `w = (m⁴+n⁴)/ρ²`:

```
ω_disc / ω_cont  =  1 + (a²/6)(λ²ρ² − w)  +  O(a⁴)
```

Two terms of **opposite sign**: `−w` is the spatial droop and `+λ²ρ²` is the time error, which on
an explicit scheme is *sharp*. The implicit plate has no `+` term at all — both of its errors
flatten — and that single structural difference is everything below.

The plate's own weight is the same `w`, entering as `1 − a²w/3` where the membrane has
`1 − a²w/6`, because a plate's frequency carries the operator's eigenvalue and a membrane's
carries its square root. `block_weight` in `physsynth/analysis/horizon.py` is that one shared function (§11).

### 10.2 The membrane's space floor is the STRING's, not half of it

On the diagonal, where §8.2 makes the plate's droop exactly `sinc(u)²`, the membrane's is exactly
`sinc(u)` — the string's, asserted to `1e-14`. So `sinc_horizon_fraction(cents, power=1)`, and
`power` is read off `ω ~ c p` rather than chosen. The two models' space-only cents are a factor of
two apart, exactly, for the same reason §8.3's plate is the string at half the cents.

**In space the plate's corner rule therefore transfers with nothing to prove.** The block's
heaviest mode is `(M, M)` for every `M` (checked to `M = 64` on integers), because it is the same
`argmax` of the same `w`.

### 10.3 Every mode has its own cancellation Courant number

Setting the bracket to zero gives a closed form with no grid in it:

```
λ_cancel(m, n) = √(m⁴ + n⁴) / (m² + n²)
```

Measured as the zero crossing of the signed pitch error, and it converges like `1/N²`: for `(7,4)`
the residual runs `1.58e-4 → 3.94e-5 → 9.84e-6 → 2.46e-6` over `N = 64…512`, a factor of **64.1**
across an 8× grid step.

**The diagonal is the exception and is exact at every `N`.** There `λ√S = (1/√2)√(2sin²u) = sin u`
and the scheme's own `arcsin` undoes it, so the residual is already at the floor and does not fall
with the grid — it is `brentq`'s tolerance against an increasingly flat function. Asserting a
convergence rate on the diagonal would be asserting the root finder, which is why it has its own
test. This is the §8.2 distinction (identity versus limit) arriving in a second place.

### 10.4 The CFL ceiling IS the spectrum's minimum cancellation number

`λ_cancel²` is `t² + (1−t)²` with `t = m²/ρ²`, minimised at `t = 1/2` — the diagonal — where it is
exactly `1/2`. So:

* `λ_cancel ≥ 1/√2` for **every** mode, with equality **iff `m = n`**. Asserted over integers with
  no dispersion relation evaluated at all;
* therefore on a stable membrane (`λ ≤ 1/√2`) **no mode is ever sharp**. Every mode is flat, or —
  the diagonal, at the ceiling — exact. There is no configuration in which one family runs sharp
  and another flat, so no "the errors average out" reading is available;
* and the "magic Courant number" is not a coincidence that happens to land on the stability bound.
  The stability bound and the best attainable tuning are the same worst-case over the spectrum,
  and they are the same number for that reason.

**The string is the same formula with the second axis dropped.** `λ_cancel(m, 0) = √(m⁴)/m² = 1`
for every `m` — the 1-D CFL limit — so in 1-D the entire spectrum cancels at one Courant number and
the string resolves essentially its whole grid, while in 2-D only the diagonal attains the minimum.
That is the mechanism under §4's headline, which until now was a measurement with no reason beneath
it. `cancellation_courant(m, 0)` spells it, and returns exactly `1.0`.

### 10.5 The block's corner FLIPS, at `1/√(M²+1)`

Set the two corners' errors equal. With `λ² ≤ 1/2` both are flat, so absolute values compare
directly, and:

```
(M⁴+1)/(M²+1) − λ²(M²+1)  =  M²(1 − 2λ²)
        ⟺  (M² − 1)[λ² − 1/(M² + 1)]  =  0
```

So for every `M ≥ 2`:

| | below `λ* = 1/√(M²+1)` | above it |
|---|---|---|
| worst mode of the `M×M` block | **diagonal** corner `(M,M)` — the plate's answer | **axial** corner `(M,1)` |

Measured against the closed form at `N = 512`, `M = 2…12`, and it converges like `1/N²` (the `M=8`
residual falls by ~250 over a 16× grid step). `M = 1` is the degenerate row the factor `(M²−1)`
removes: a `1×1` block has one mode and both corners are it.

### 10.6 Which is why the plate's rule is inverted, not weakened

`1/√(M²+1) < 1/√2` for every `M ≥ 2`, and it **falls like `1/M`**. So the window in which a
membrane's block behaves like a plate's shrinks as the block grows and never contains a Courant
number anyone would choose: at the ceiling, at the suite's default `0.7`, and at `0.6`, `0.5` and
`0.45`, the worst mode of every block from `2×2` to `12×12` is an **axial** corner, on every grid
measured.

Two things this section is careful about:

* **"an axial corner", never a particular one.** `(M,1)` and `(1,M)` are degenerate to the bit on
  a square (asserted as `==`, not a tolerance). `mode_block` sorts by `(ρ², m, n)`, so `argmax`
  returns `(1,M)` — a tiebreak, not a result, and a test that pinned it would be asserting the
  sort.
* **the worst mode is always a corner**, never interior. The two-corner algebra above never looks
  at `(M, n)` for `1 < n < M`, where the time term — monotone in `n` — could in principle move the
  weight's interior minimum far enough to win. It does not, for any block to `M = 120` at 48
  Courant numbers, asserted on integers — and with a measured companion that asks the model instead,
  on two grids across a sweep that spans the flip so it sees both corners win. Every other closed
  form here is either checked against a convergence rate or backed by a measurement, and this one is
  the licence the whole block reading rests on.

### 10.7 The step that turns a corner into a number

Knowing which corner is worst licenses a block reading **only** if that corner's family has a
leading prefix to read — `pitch_horizon` returns a prefix, and a prefix over a non-monotone list is
meaningless (§8.6). The membrane's axial family is monotone at every Courant number in the stable
range: at the ceiling its error goes like `(M²−1)²/(2(M²+1))`, which increases. Asserted through
`pitch_horizon`'s own `monotone` flag *and* strictly, at five Courant numbers.

Without this the section would prove which corner is worst and still be unable to quote a block
horizon at all.

### 10.8 §4.1's "about 12% of the grid", derived

The two hand-picked bars in `test_the_membranes_cancellation_is_DIAGONAL_ONLY` are now closed
forms, and the originals are kept beside them (§7.5 — a derived bar with no floor under it asserts
less than the literal it replaced):

* `diagonal >= 0.9 (N−1)` becomes `diagonal == N−1`. "Essentially exact" was exact: the
  cancellation is an identity, so every mode of the family is in tune and the horizon *is* the
  family. Measured `31/31, 63/63, 127/127, 255/255, 511/511`;
* `axial < 0.25 (N−1)` becomes `√2 · sinc_horizon_fraction(cents, 1) · N`, within one mode and
  never above. The axial family at the ceiling resolves **exactly `√2` times the share of the grid
  the string's space floor allows** — leading-order in both, and the ratio `√(48/24)` is exact.
  At 5 cents that is `0.1185 N` against the string's `0.0838 N`, which is where §4.1's
  hand-measured "about 12%" comes from. Measured `4, 7, 15, 30, 60` at `N = 32…512` against floors
  `4, 7, 15, 30, 60`.

A second closed form exists and was not used: solving `K(M²−1)²/(2(M²+1)) = c` exactly for `M²`
with `K = (1200/ln2)(π/2N)²/6` is a quadratic and tracks the measurement to a fraction of a mode at
1 and 5 cents. It drifts one mode high at 25 cents, where the leading-order expansion is being
asked about modes a quarter of the way up the grid. The `√2` form was preferred because it reuses
the shipped primitive and states the relationship rather than a number.

### 10.9 What this does to `mode_block`

Its docstring stated the plate's rule as the general licence — "its worst mode is its **diagonal
corner**". A membrane caller reading that at the CFL ceiling would name the exactly wrong mode, so
it is amended in the same commit: the general claim is that the worst mode is *a corner*, and
**which** corner is a property of the scheme, not of the block.

### 10.10 Not done

* **No band was derived here.** §7.3's audit lists no rectangular-membrane pitch band to derive:
  its one membrane row is the circular drum head against Bessel, whose limiter is the staircased
  domain rather than dispersion. The block rule is now available for the first such band that gets
  written.
* **The circular membrane is untouched**, and stays in §5's "no horizon yet" group for the reason
  that row already gives — its reference is a frequency for a different shape.
* **Damping is not in any of this.** Every statement here is `σ = 0`; a lossy membrane's modes are
  shifted by the loss as well, and §4 never separated the two either.
* **Promotion out of `tests/`. Done 2026-09-07 — §11**, and both of the added helpers went with
  the rest. This batch adds two more
  closed-form helpers (`cancellation_courant`, `block_weight`) to the pile that would move, and
  neither of them brings a new dependency — they are arithmetic.

---

## 11. The promotion — 2026-09-07

The horizon primitives stop being test-folder fixtures and become part of the instrument. §6, §8.9,
§9.8 and §10.10 all named this as the follow-on and §6 gave two reasons it would be hard. Neither
was a reason, and that is the useful content of this section.

### 11.1 The dependency question was already answered, three inches from the call site

§6 says `scipy.optimize.brentq` "would be a dependency decision in the analysis crate, whose
allowlist is deliberately narrow". It is not one, and the answer was written down months earlier in
a place a keyword search for `brentq` finds immediately: `crates/physsynth-analysis/src/lib.rs`
opens with a header arguing out three ways to reach Brent's method from this crate, rejects taking
`physsynth-core` as a Cargo dependency (it inverts the crate split and goes red in `deps.rs`) and
rejects copying the file (two transcriptions free to drift), and settles on `#[path]`-including
`physsynth-core/src/root.rs` — a line-for-line transcription of SciPy's `brentq.c`, already
compiled into this crate for `modal.rs`'s two root finds.

So `sinc_horizon_fraction` calls `crate::root::brentq` and `ALLOWED: &[&str] = &[]` in
`crates/physsynth-analysis/tests/deps.rs` stays empty. Not one line of `Cargo.toml` moved.

**The transferable point is about how the obstacle was stated.** "Would be a dependency decision"
described a *function name*, and dependencies are not properties of function names — they are
properties of what a crate already compiles. The question that would have dissolved this in one
grep is "does this crate already have a root finder?", and it does.

### 11.2 The freeze question was answered by not generalising

§6 says satisfying `tests/test_analysis_frozen.py`'s derived guard "is impossible — there is no
Python implementation left to freeze against", and calls widening that guard for one function a bad
trade for the 62 fixtures it protects. Both halves of that are sound *reasoning* over a false
premise.

The premise came from `docs/dev/rust-migration-plan.md` §44, where six analysis modules were
**deleted**: their Python bodies went, so nothing can ever recompute what they said, and
`tests/analysis_frozen_values.py` exists because that was the last chance to record it. These seven
functions are not in that situation. They were **promoted**, and a promotion has a live Python body
on one side of it — `tests/helpers.py`'s, right up to the commit that replaced it with the shim.

`scripts/freeze_horizon.py` is what walked through that window. Twelve cases, recorded from the
Python and compared against the Rust in the same pass, in the batch that had both. The guard was
widened afterwards for an unrelated and better reason (§11.5), not to accommodate this.

**The transferable point:** a constraint inherited from a *deletion* does not automatically hold
for a *move*. "There is no second implementation" was a fact about §44's subject, not a property of
this repository, and it stopped being true of anything the moment the question was about a function
that still existed.

### 11.3 What the measurement said

Every float case recorded a gap of **exactly 0.0** against the Python — including
`sinc_horizon_fraction`, where two hazards stack and neither could be reasoned away in advance: the
Python called `brentq` with SciPy's *implicit* defaults (`xtol=2e-12`, `rtol≈8.88e-16`), so the
Rust had to be handed them explicitly or the two would be different root finds; and the objective
evaluates `sin`, which NumPy computes with its own CPU-dispatched routine rather than the platform
libm (rust-migration ledger #28). Both were measured rather than assumed, and both came back clean.

That is a statement about *this machine*, which is exactly why the frozen bar stays a `1e-13`
tolerance and not an equality. Ledger #28 is the standing reminder, and 2026-09-07's mallet-gong CI
failure — nineteen red runs on an `== 0.0` that held on Windows and read 3.5e-13 on Linux — is the
recent one.

### 11.4 Five rows record a string, and it is the strong arm rather than a hole

`pitch_horizon` returns `(int, bool)`; `mode_family` and `mode_block` return lists of integer
pairs. There is no float in any of those answers, so the tolerance arm of the frozen test compares
nothing and the entire comparison is the exact one on `ints` and `structure`.

`test_every_case_carries_a_measured_gap_rather_than_a_reason_it_could_not_be_measured` used to
reject *any* non-float in that slot, which was right when every string there meant "the two sides
could not be compared". It now separates the two meanings, and adds a check the original did not
have: a row claiming "no floats" must really have none **and** must carry integers, so the
exemption cannot become a way to freeze a row that asserts nothing at all.

Do not read those five rows as bit-identity between two float computations. Nothing float was
compared.

### 11.5 The guard stopped being a hand-written list

`ANALYSIS_MODULES` was a tuple of six module names. Adding `"horizon"` to it by hand would have
restated the hole one number higher — the *eighth* module would arrive unfrozen and unnoticed for
exactly the same reason the seventh nearly did. It now reads the package with
`pkgutil.iter_modules`, so the claim is "no module in `physsynth.analysis` has a public function
without a frozen case".

Same move, and the same reasoning, as rust-migration ledger #67: a guard that would be *edited* to
stay correct should be *derived* instead. Widening the list by hand is how a derived-looking guard
quietly becomes a maintained one.

### 11.6 A native bar caught a docstring overclaim

27 bars in `crates/physsynth-analysis/tests/horizon.rs`, all identities out of the docstrings rather
than fixtures, because a frozen record cannot catch an error the Python made too (rust-migration
§37.11 is the precedent: a native bar found a 544% defect in the free circular plate's oracle that
both implementations had always shared).

One of them failed on the first run, and it was the docstring that was wrong. `mode_block` says the
error weight "has an interior minimum in `n`", which is what makes the corner argument work.
Minimising `(m⁴ + t²)/(m² + t)` over `t = n²` gives `t* = m²(√2 − 1)`, so the minimum sits at
`n* = m·√(√2 − 1) ≈ 0.6436 m` — strictly inside `(0, m)`, as claimed, **for the continuous
function**. On the integer grid it is only reachable from `m = 3` up: at `m = 2` the minimiser is
1.287 and the nearest index below it is `n = 1`, which is the block's own edge.

The corner argument is untouched, because it is a statement about the **maximum** and the maximum is
still always a corner (asserted separately, `m_max = 2..12`). But the bar now pins the closed-form
position rather than merely "somewhere inside", which is a stronger claim than the one that failed,
and it records the `m = 2` exception where a reader checking the docstring will hit it.

### 11.7 What did NOT move, and why that is the whole design

`spatial_operator_horizon` stayed in `tests/helpers.py`. §7.7 already said why and this batch is the
first time it cost anything to act on: it takes `(N, kappa)` and returns "the horizon", which reads
like a general answer and is not one — it hardcodes `L_DEFAULT`, `wave_speed()` and a 1-D Dirichlet
second difference, so it silently answers about the canonical string whatever the caller had in
mind. In the test folder that is a fixture and it is honest. In the library it would be a lie with a
general-sounding name.

The contrast with `mode_family` is the design rule this module is built on: `mode_family` returns
*index pairs only* and makes the caller build its own frequencies from its own geometry, which is
why it can live in a library at all. A promoted helper must not carry a fixture inside it.

There are **no re-exports** in `tests/helpers.py`. The four caller files import from
`physsynth.analysis.horizon` directly. A re-export would be an unused import kept alive only by
`__all__` — the `ruff --fix` hazard of rust-migration ledger #66 — and, worse, would leave two
routes to one name with nothing saying which is canonical.

### 11.8 Not done

* **The viewer read-out.** "This configuration is trustworthy to 2 kHz" is what all of this was for,
  and it is now unblocked: `web/serialize.py` can import `physsynth.analysis.horizon` like any other
  oracle. It is a separate batch and needs a decision about *which* horizon a mixed model reports.
  **Done 2026-09-07 — §12**, and the decision is the worst member of the chain, with any member
  that has no horizon refusing for the whole scene.
* **The four "no horizon yet" rows** of §5 are still not attempted, unchanged from §6.
* **Nothing was promoted that did not already exist.** This batch moved seven functions and wrote no
  new physics; §11.6's finding is a correction to prose, not to a formula.

---

## 12. The viewer read-out — 2026-09-07

"This configuration is trustworthy to 2 kHz" is what §6, §11.8 and the batch before them all said
this was for. It exists now: every payload `web/serialize.py` returns carries a `horizon` block, and
the viewer draws it as a strip under the transport — a headline frequency, the mode that ends the
claim, and a 1 / 5 / 25 cent selector that retunes the sentence with no round trip.

Code: `web/serialize.py` (`_horizon_report`, `_horizon_string_block`, `_horizon_grid2d`, the two
gates and `HORIZON_ABSENT`), `web/static/app.js` (`drawHorizon`), `web/static/index.html`,
`web/static/style.css`. Tests: 19 functions in `tests/test_web_backend.py`. The headless dev harness
`scripts/verify_web_headless.py` now reads the strip and fails a case whose strip is hidden or
empty. No physics was written: every number comes from `physsynth.analysis.horizon` and
`physsynth.analysis.modal`, which is what §11 was for.

### 12.1 The open decision: a mixed scene reports its WORST member, and refuses if any member has none

§11.8 left this open — "which horizon a mixed model reports". The answer is the worst member of the
chain, named, with two consequences that do most of the work:

* a **modal** component has no spatial discretisation error at all, so it never limits anything;
  only grid resonators can;
* a scene containing **any** member with no horizon is a refusal *as a whole*. Quoting the string's
  number for a string-plus-nonlinear-plate scene and hoping a caveat carries is how a read-out
  becomes a lie.

In practice the second rule is what decides most of the viewer, because the coupled scenes are
coupled *to* the things that have no reference.

### 12.2 The block is a two-value union, and both arms ship

```
{"kind": "prefix", "scheme": …, "dims": 1|2, "of": "the damped string", "n_modes": 127,
 "f_max": …, "nyquist": …, "default_cents": 5.0,
 "bands": [{"cents": 5.0, "modes": 5, "hz": 502.1, "limited_by": "partial 6",
            "limit_hz": 603.1, "limit_cents": -5.34, "saturated": false,
            "index": 5, "family": "harmonic", "family_tied": false, "families": [...],
            "monotone": true}, …]}

{"kind": "none", "reason": "this domain is staircased onto the grid, so the error being measured
                            is the SHAPE, not the scheme: …"}
```

Every model key produces one arm or the other. A builder that can compute a horizon sets the key
itself; everything else falls through `simulate_to_payload` to `HORIZON_ABSENT`, whose entries are
the plan's own §5 refusals in prose. A model in neither table would get a generic sentence, so
`test_horizon_every_model_the_viewer_OFFERS_is_classified` derives the population **from the
`<select>` in `index.html`** and asserts the two tables partition it — the same move §11.5 made on
`ANALYSIS_MODULES`, for the same reason: a hand-written tuple restates the hole one number higher.

### 12.3 Two readings, because §8.5 and §9.5 each end with a sentence about this read-out

Those sections say a horizon **in hertz is family-dependent and in mode index is not** — √2 between
a plate's two families, 3.7× between spruce's two axes. A user asks the question in hertz, so both
are shipped:

* **the frequency ceiling** — the highest continuum frequency with nothing out of tune at or below
  it, computed as a *first failure* over the exhaustive sorted spectrum. Deliberately **not**
  `pitch_horizon`: §8.6 says a prefix over the sorted 2-D union is meaningless, and this reading
  does not need one — "no mode below 361 Hz is out by more than 5 cents" is true whether or not the
  error curve is monotone;
* **the block index** — `pitch_horizon`'s leading prefix along the corner families, which are the
  sequences a prefix *does* mean something along. Reported with `monotone` rather than collapsed to
  the integer (§1).

They differ by a lot and that is the point: the same membrane is "8 modes below 361 Hz" and "every
mode with m, n ≤ 3", and at 25 cents the two are 22 and 5. Quoting the first as though it licensed
"the first 22 modes" would overstate the block by a factor of four.

### 12.4 All three corner families are read, and a tie is reported as a tie

`mode_block`'s docstring says the worst mode of a block is *a* corner and **which** corner is a
property of the scheme — diagonal for an implicit plate (§8.7), axial for an explicit membrane at
any runnable Courant number (§10.5). Rather than branch on the regime, the read-out asks all three
families (`axial`, `axial_y`, `diagonal`) and takes the minimum, which is right in both regimes and
in the one the plan has not characterised.

Ties are then unavoidable and are shipped as ties. §10.6 warns that `(M,1)` and `(1,M)` are
degenerate to the bit on a square, so naming one is reporting a sort's tiebreak as a result — and in
the viewer's range there is a coarser tie as well: three families whose prefixes land on the same
small integer, which is what a plate at `N = 40` does at every bound. So the payload carries every
family's own index and a `family_tied` flag, and the page says "no single worst corner at this
index — axial (1, n) and axial (m, 1) end together" instead of picking one. On a **non-square**
rectangle the tie breaks for a real reason and the strip names the loser: `Ly = 0.8` puts the
`(1, n)` family two modes short of `(m, 1)`, which is §9.5's index-versus-hertz point arriving
through geometry rather than through a grain.

### 12.5 The trap: a horizon read off a display array is a fact about the array

The spectrum panels ship `N_PARTIALS = 12` and `N_PLATE_MODES = 6` frequencies, and they were
already sitting in every payload builder. Handing either to `pitch_horizon` returns at most 12 or 6
— a number that moves with `N` in exactly the wrong way, plateauing at the panel's length. Every
mode set here is built from the scheme's own dispersion relation over the **whole** resolvable
range instead (`1..N-1`, or the full `(m, n)` grid), the way `tests/test_resolution_horizon.py`
builds its own. It costs nothing: these are closed forms, no eigensolve, 1.1 ms for a 1999-mode
string and 16.7 ms for a 98×98 grid against renders of seconds.

`test_horizon_is_built_from_the_SCHEME_and_not_from_the_display_arrays` is the guard, and it asserts
`n_modes == N - 1` over three grids rather than a single literal.

The read-out was also checked once against a number recorded **outside** it: §3's table says the
canonical damped string at `N = 256`, `λ = 1`, `κ = 0` resolves **11** partials at 5 cents, and the
viewer's block for `{"model": "damped", "N": 256, "lambda": 1.0}` says 11 — with the viewer's loss
defaults on, which is the second half of the check, since this read-out ignores the damping shift
(second order in `σ/ω`, ~1e-4 against a 5-cent bound of 2.9e-3). That agreement is recorded here
rather than asserted in a test on purpose: §5's "nothing is a frozen integer" applies to the viewer
too, and the tests pin family behaviour instead. Anyone doubting it can reproduce it in one call.

### 12.6 `horizon == 0` is live, and `hz` is `None` rather than a number

The plan already recorded a plate whose *fundamental* is 9.6 cents flat, so a viewer that assumed a
non-empty prefix would have been wrong on a shipped default: `_vk_params(nonlinear=False)`, the
viewer's own linear von Kármán fixture, is 7.4 cents out at `(1,1)` and has **no** horizon at 1 or 5
cents — and one mode at 25. `hz` is `None` there: `0.0` would read as a measurement and NaN would be
refused by `server.py`'s `allow_nan=False` on the way out. The page has its own sentence for it
("Nothing here is in tune to 5¢ — even the lowest mode …"), and the index line is suppressed rather
than printing "every mode with m, n ≤ 0", which is a sentence about the empty set.

### 12.7 The gate keys on `(model, domain)`, and one model key covers three plates

`plate` is a rectangle, a free plate and a guitar outline; `membrane` is a rectangle and a disk;
`vk` is a nonlinear plate and — with the checkbox off — a simply-supported Kirchhoff one. A block
keyed on the model alone would confidently quote the rectangle's number for the staircased guitar,
which is §5's staircase refusal happening inside a single select. So the gates read the resonator:
`domain`, `boundary`, `grain_is_isotropic` and the `nonlinear` flag each route to a refusal with its
own reason, and only the rectangle reaches `_horizon_grid2d`.

The 1-D gate is the same shape and its discriminator was a surprise: the θ-family strings report
`boundary == "supported"` and the ideal string reports `"fixed"`, both of which are the sine series
the modal oracles are derived for, while every bridge-coupled scene builds
`IdealString(boundary=("fixed", "free"))`. **The refusal that matters is not the spelling, it is the
free end** — and it is a physics refusal rather than a missing-oracle one: that string's continuum
partials are shifted by the bridge the scene exists to show, so a cents comparison against the
uncoupled harmonic series would report the coupling as a discretisation error. Two errors in one
number, the same reason §5 refuses a staircase.

### 12.8 What the viewer now says, by model

| models | read-out |
|---|---|
| ideal / stiff / damped string, bow, jawari, juari, fret | measured; the exciters and contacts inherit the string they drive (§5's "inherited" row) |
| membrane and mallet on a **rectangle** | measured, both readings — the mallet inherits the membrane's *gate* as well as its horizon, so a struck **disk** routes to the staircase refusal by the same call |
| plate, **supported rectangle** | measured, both readings |
| von Kármán plate with the nonlinearity **off** | measured — it is a Kirchhoff plate again |
| membrane/mallet on a **disk**, plate on the **guitar outline** | staircase refusal |
| plate, **free** | tabulated-reference refusal |
| von Kármán (on), tension string, geometric string | nonlinear refusal — the horizon they have is a *refinement* horizon |
| bore, reed | Webster-area refusal |
| sympathetic, body, platebody, radbody, airload | bridge-coupled refusal (§12.7) |
| airbox, vkroom | 3-D room: direction-dependent, recorded elsewhere as a light cone |

### 12.9 What it costs the suite

19 tests, **6.57 s** standalone against the file's 152.6 s locally (`--durations` puts none of them
in its slowest 25; the heaviest is 1.97 s, the three plate renders in the display-array guard). That
is ~4% of the slowest file in the suite, well inside the runner variance the three-shard gate
already tolerates, and `tests/test_shard_partition.py` passes unchanged —
`scripts/shard_costs.json` is built from **CI** durations rather than a local run, so it is
deliberately not regenerated here.

### 12.10 Not done

* **The four "no horizon yet" rows** of §5 are still not attempted, unchanged from §6 and §11.8 —
  and this batch is the first time their absence is *visible to a user*, which is an argument for
  doing them rather than against.
* **A bridge-coupled string has no horizon of its own here.** The refusal is honest but it is not
  the last word: the coupled system has modes, and a horizon against *them* is a real measurement
  that would need a coupled modal oracle. It is a physics batch, not a viewer one.
* **The cents bound is not in the deep link.** It is a display control, so a link cannot carry it;
  three bounds ship in every payload and the fourth someone wants would need a round trip.
* **The strip is not on the canvas.** Nothing marks the horizon *on* the spectrum panel, where a
  vertical line at `hz` would say the same thing in the place the partials are already drawn.
