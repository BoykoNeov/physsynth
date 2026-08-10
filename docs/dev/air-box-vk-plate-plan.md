# The gong in the room — the von Kármán plate as a suspended and baffled surface (air-box batch 6)

Model #6, the nonlinear plate, is the last resonator the air box has not taken. Batch 5 named its
extension point in advance (`air-box-membrane-plan.md` §3) so this batch would not rediscover it:
the load is linear in `w^{n+1}` and independent of the Airy stress `F`, so it folds into `A` exactly
once and the Picard loop is untouched — but the seam #6 needs is a **loop hook**, not the single
`rhs()` batch 5 built, because the loaded solve must sit *inside* the fixed-point iteration.

That prediction is confirmed below and it is the only part of batch 5's handover that survived
contact. Everything batch 5 said about the *headline* — that #6 would inherit its `c/c₀` threshold
shape, a size-free half plus a large-surface half — is killed in §0.

---

## 0. What the probe killed before this document was written

The family's standing pattern is that plan claims die on measurement — five in batch 3, four in
batch 4, five in batch 5, every one because the prototype measured the effect where it was hidden.
This batch's plan-stage probes killed **three claims of mine before the document asserted them**, so
they are recorded here as the reason the headline is what it is rather than what it nearly was.

*The three probes named below (`feasibility.py`, `probe_kappa_eff.py`, `probe_cascade.py`) are
plan-stage throwaways and are deliberately not committed — every number they produced is reproduced
in this section, and the ones worth keeping become `scripts/diagnose_airbox_vk.py` in §9.*

### 0.1 DEAD — "von Kármán stiffening moves the coincidence frequency"

The attractive size-free claim, and the one that would have inherited batch 5's shape exactly. The
reasoning: von Kármán stretching adds an effective tension, so the bending wave goes *faster*, so it
reaches `c₀` at a lower wavenumber, so **coincidence moves down with amplitude** — a relation
derivable from the plate alone, with the room only confirming the consequence.

The arithmetic even looked clean. A wave of amplitude `A` and wavenumber `ks` carries stretching
strain `A² ks²/4`, so `T_eff = Y_mem A² ks²/4`, and the membrane term of the dispersion relation is
`(T_eff/ρ_s) ks² = Y_mem A² ks⁴/(4 ρ_s)` — the *same* `ks⁴` as bending. So it predicts a
**wavenumber-independent** stiffening, `κ_eff² = κ² + Y_mem A²/(4 ρ_s)`, i.e. every mode's frequency
scales by one factor and coincidence moves as `1/κ_eff`. At `w = 3e` that is **5.06×**.

**Measured on the bare plate (`probe_kappa_eff.py`, the canonical 0.4 m × 1 mm steel sheet of
`tests/test_vk_free.py`), it is 1.18–1.41×.** The ansatz is wrong by a factor of ~4, and the repo's
own `test_vk_free.py::test_amplitude_pitch_glide_free` already contained the refutation — it asserts
only `>1.15×` by `w ≈ 3e`, which is what a plate that stiffens weakly looks like. A free plate
**relieves** the stretching: the edges are free to move in-plane and the Airy solve redistributes the
stress, so the fully-restrained uniform-strain estimate does not apply.

| `w/e` | mode 3 | mode 6 | mode 10 | spread | ansatz |
|---|---|---|---|---|---|
| 0.01 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 0.5 | 1.012 | 1.009 | 1.013 | 1.004 | **1.297** |
| 1.0 | 1.054 | 1.044 | 1.060 | 1.016 | **1.931** |
| 2.0 | 1.206 | 1.112 | 1.168 | 1.085 | **3.453** |
| 3.0 | 1.406 | 1.184 | (—) | — | **5.057** |

The one thing the ansatz got right is that the stiffening is nearly **mode-independent** — spread
1.004/1.016/1.085 across three modes spanning `ks = 7.4 … 16.0`. That is worth keeping as a small
true statement; it is not a headline.

Consequence: coincidence moves from 12220 Hz to about 10465 Hz at `w = 2e` — a **14% shift, not a
5× one**. Which kills the next claim too.

### 0.2 DEAD — "the nonlinearity brings coincidence into affordable reach"

This was the rescue for §0.1's cost problem and it dies with it. Resolving coincidence for this
plate needs `h_air ≤ 5.61 mm` (batch 5's ≥5 air cells per structural wave), hence `fs ≥ 105.8 kHz`
from the room's `λ ≤ 1/√3`, hence a 2.5 m room at 447³ ≈ **89 M air nodes (~3.6 GB)** and about
**5.25 h for half a second**. A 14% downward shift in `f_c` does not touch that. **A coincidence
claim is out of scope for this batch on cost, and the number is stated here so the deferral is a
budget decision rather than an omission.**

### 0.3 DEAD (as a *frequency* claim) — "a cymbal cascades up through coincidence"

Physically real and the reason a cymbal shimmers, but not measurable here: the cascade would have to
be resolved at its **top** wavenumber, which is §0.2's bill again. `probe_cascade.py` also shows the
spectral-peak observable itself failing at `w/e = 3` — the peak tracker on mode 10 reads 50.6 Hz,
*below* that mode's own linear frequency, because the field has gone broadband and "the" frequency
has stopped existing. **A spectral peak is not an observable for a cascading plate**, which is worth
carrying into any later batch that tries.

### 0.4 WHAT SURVIVED, and it is better than what died

The surviving claim needs no coincidence, no fine air grid, and no high frequencies — and it comes
with a control built into the same run. Measured with **no room at all**, projecting the velocity
field onto the mass-orthonormalised free-plate modes and watching the modal energy shares drift:

| strike `w/e` | modal-share drift, first → last window (0.5 s) | Picard peak sweeps |
|---|---|---|
| 0.05 (effectively linear) | **0.0026** | 3 |
| 1.0 | 0.1830 | 5 |
| 3.0 | **0.4086** | 7 |

At small amplitude the modal distribution is **frozen** — drift 0.26%, which is the linear plate's
defining property: a struck linear plate's shape-content is fixed for all time, so its radiation
pattern is fixed for all time too. At `w = 3e`, **41% of the modal energy has moved to different
modes within half a second**, at fixed geometry, with no change of excitation.

Batch 3's headline is that a surface radiates by the **shape** of its motion, not by its net volume
displacement. Put those together and the batch has its claim:

> **A loud plate's radiation is time-varying at fixed geometry, and a quiet one's is not.** No
> `R(ω)` — no constant `R`, no rational impedance, nothing in `radiation.py` — can state this: a
> scalar-per-frequency load has *one* pattern per frequency and cannot change it during a single
> strike. This is the first radiating object in the repo whose acoustic character is a function of
> **how hard it was hit**.

The monopole read-out is measured to be blind to it, as batch 3 would predict: net volume
displacement runs 9.69e-8, 1.94e-6, 5.82e-6 m³ across the three amplitudes — i.e. very nearly
**linear in the strike amplitude** (1 : 20 : 60 against amplitudes 1 : 20 : 60). The lumped tier
sees a louder plate and nothing else.

### 0.5 The `ka` gap — closed by probe, because otherwise it would have killed §0.4 in Commit C

The drift above was measured on the canonical 0.4 m plate, whose modal centroid sits at **`ka ≈
0.25`** — *acoustically compact*, where batch 5's lesson applies verbatim (a compact source is quiet
because it is compact) and a pattern claim has nothing to bite on. The band this plan originally
costed, `ka = 3`, was **not the band the drift was measured in**. That is precisely the mismatch that
killed five claims in batch 3, and it would have surfaced only in Commit C — as 41% drift producing
no measurable change in anything, with every ledger green.

`probe_ka_window.py` sweeps plate geometry × strike width for a cell with **both** drift ≥ 0.20 and
centroid `ka` ≥ 0.80 (`ka` scales as `e/L`, so smaller or thicker raises it; a narrower strike raises
the centroid). Five geometries × two strike widths, `w/e = 3`, free edge, 0.125 s at 96 kHz:

| `Lx` | `e` | strike frac | drift | centroid | `ka` centroid | `ka` p90 | peak sweeps | |
|---|---|---|---|---|---|---|---|---|
| 0.40 | 1.0 mm | 0.20 | 0.504 | 100 Hz | 0.37 | 0.68 | 5 | compact |
| 0.40 | 1.0 mm | 0.08 | 0.293 | 112 Hz | 0.41 | 0.68 | 9 | compact |
| 0.20 | 1.0 mm | 0.20 | 0.446 | 389 Hz | 0.71 | 1.35 | 10 | compact |
| 0.20 | 1.0 mm | 0.08 | 0.337 | 428 Hz | 0.78 | 1.35 | 36 | marginal |
| 0.20 | 2.0 mm | 0.08 | — | — | — | — | 50 | **diverged** |
| **0.10** | **1.0 mm** | **0.20** | **0.328** | **1660 Hz** | **1.52** | **2.70** | **29** | **PASS** |
| 0.10 | 2.0 mm | 0.20 | 0.304 | 1560 Hz | 1.43 | 2.70 | 50 | pass, at the cap |
| 0.10 | 1.0/2.0 | 0.08 | — | — | — | — | 50 | **diverged** |

**The build cell is `Lx = Ly = 0.10 m`, `e = 1 mm`, broad strike (width = 0.20 `Lx`), `w/e = 3`** —
drift 0.328 with the modal centroid at `ka = 1.52` and the 90th-percentile mode at `ka = 2.70`,
i.e. squarely inside batch 4's validated directivity band `ka = 0.8…2.8`.

**Two findings that come free with it, and both constrain the build:**

1. **The narrow strike is unusable — the knob that raises `ka` breaks convergence.** Every
   `0.08`-width cell at `L ≤ 0.20` hit the `couple_max_iter = 50` cap and produced NaN. This is the
   `VKPlate` docstring's own warning ("the strong-cascade regime `w ≫ e` may not converge —
   qualitative, not a gate") arriving as a *hard* limit on the experiment design. The broad strike
   is not a stylistic choice; it is the only one that runs.
2. **Picard sweeps go 5 → 10 → 29 → 50 across the grid, and §4's cost model assumed 3–7.** The build
   cell costs **29 sweeps**, ~4–6× the plan's first estimate, and the 2 mm variant sits *at the cap*
   — converged on the last step but not on every step, which makes it unusable for a conservation
   claim. **`n_iters` and `converged` must be asserted per step, not sampled at the end.**

**And one methodological note for §7.6:** drift is a function of the observation window, so the
window is part of the claim. Fix a canonical duration and state it — the juari viewer batch's lesson
(a settled quantity needs a canonical duration decoupled from anything else) applying to a
*transient* one for the first time.

---

## 1. Why — what a nonlinear plate says that a linear one cannot

### 1.1 The first *object* whose radiation depends on amplitude

Every radiating thing in this repo so far is linear in its excitation: strike it twice as hard and
every acoustic observable doubles. Ratios — radiated fraction, directivity, dipole/baffled — are
amplitude-**invariant** by construction. Model #6 breaks that, and it is the only model here that
can: the coupling is quadratic, so the *shape* of the motion evolves, and shape is exactly what
`SurfacePort` was built to make audible.

So the batch's ratios become **functions of a strike**, which is a new axis for the family and the
reason the claim is a ratio-of-ratios (batch 2's lesson: a ratio survives a small room, a magnitude
does not).

### 1.2 What that makes this batch, beside "the next model"

Three things the family has not had:

* a radiating object with a **built-in control in the same code path** — `nonlinear=False` is
  bit-identically the linear plate, so "frozen vs drifting" is one class and one flag, not two rigs;
* the first coupling where the **iteration tolerance** is an error source alongside the load, which
  makes the money test a two-parameter claim (§7.3) — and, if §7.3's prediction holds, exposes an
  asymmetry between the family's three detectors that the previous three batches could not see;
* the completion of §12H's model list: every resonator in `physsynth/core/` that can be a surface
  now can be one **in** the room.

---

## 2. What must change in `airbox.py` — and nothing else

Batch 5 extracted the seam (`_PlateSurface` / `_MembraneSurface`) precisely so this batch would be a
third adapter rather than a fifth and sixth copy of the load arithmetic. Verified against the source:

### 2.1 The adapter is `_PlateSurface` with one substitution — and the substitution is a trap

`VKPlate` carries `Plate`'s entire linear vocabulary under the same names: `theta`, `kappa`, `B`,
`K`, `W`, `wdiag`, `h`, `n_live`, `mask`, `index_map`, `X`, `Y`, `boundary`, `u`, `u_prev`, `sigma`,
`k`. `_VKPlateSurface` is therefore `_PlateSurface` with **`plate.rho` → `vk.rho_s`** and nothing
else:

* supported: `denominator = rho_s * h * h`, `a_bare = (1 + σk) I + θ k² κ² B`
* free: `denominator = rho_s` (bare — `W` is inside `A` and the solve divides it out),
  `a_bare = (1 + σk) W + θ k² κ² K`

**The trap: `VKPlate` has no attribute `rho`.** It has `rho_v` (volumetric, kg/m³) and `rho_s`
(areal, kg/m²), and they differ by the thickness `e` — a factor of **1000** for the canonical 1 mm
plate. Writing `rho_v` is silently wrong: the load simply becomes 1000× too weak, every ledger still
telescopes against the pressure it used, and **nothing green turns red**. This is `_PlateSurface`'s
own documented failure class ("plausible, and nothing green turns red") arriving through a different
door, and §7.1's bit-identical regression is what catches it.

### 2.2 `commit()` has no `_accel` to refresh, and that is a real difference

`_PlateSurface.commit` writes `p._accel` because `Plate.pressure()` reads it. **`VKPlate` has no
`_accel` and no `pressure()`** — its diagnostics are `state`, `stress_field`, `linear_energy`,
`membrane_energy`, `energy`. So `_VKPlateSurface.commit` is `_MembraneSurface.commit`'s shape (roll
the history) plus one thing neither predecessor has: **roll the cached Airy fields** `F_prev ← F`,
`F ← F^{n+1}`. That is why the loop hook must hand `F^{n+1}` back (§5.2) rather than stash it.

The consequence for the wrapper: it must **not** define `pressure()`. `RoomLoadedPlate` has one;
`_RoomLoadedMembraneMixin` deliberately does not, with a comment saying why. Batch 6 follows the
membrane, and for the same reason — the model has no such read-out.

### 2.3 Nothing in the port, the cut, the footprint check or the ledger changes

`SurfacePort`, `InteriorSurfacePort`, `AirBox.add_cut`, `_check_footprint` (span-wise since batch 5)
and the `injected` ledger are all untouched. A VK plate is a rectangle on a grid plane, which is the
case they were built for and the case batch 5's fix reduces to by construction.

---

## 3. Scope — and what is deliberately deferred

**In:** `RoomLoadedVKPlate` (baffled, flush in a wall) and `RoomSuspendedVKPlate` (hanging, two
faces, cut), for **both** boundary branches (`supported` = the gong, `free` = the cymbal). The
`_VKPlateSurface` adapter and the seam's fourth member, the loop hook. Oracles per §7.

**Out — string→VK-plate bridge composition, and this one is measured rather than assumed.**
`StringPlateBridge` cannot take a `VKPlate` today, for three independent reasons found by reading
`connection.py`:

1. `_stability_margin` reads **`p.rho`** (lines 381/385) to build the plate's `G0` block — the
   attribute §2.1 says does not exist on `VKPlate`.
2. `step()` calls **`plate.step(f_ext=...)`**, and `VKPlate.step()` takes no `f_ext` at all.
3. `pressure()` delegates to **`plate.pressure()`**, which `VKPlate` does not have.

None of those is hard to fix and none of them belongs in this batch: the guard is a *linear* 2-DOF
estimate and what it means for a plate whose stiffness is amplitude-dependent is a genuine question,
not a plumbing detail. `StringVKPlateBridge` is its own batch. **This is stated in the plan so that
`RoomLoadedVKPlate.__getattr__` is free of `RoomLoadedPlate`'s "NOTHING here may shadow a name the
bridge reads" constraint** — a constraint that exists only because a bridge composes with it.

**In, and with a counterpart this time: the `f_ext` path.** Batch 5 flagged that
`_MembraneSurface.rhs`'s `k² f_ext / ρh²` term was *new* arithmetic with nothing to be bit-identical
to, and pinned it with a static-deflection oracle instead. Batch 6 is better off: `VKPlate.step()`
takes no `f_ext` either, **but `RoomLoadedPlate` does**, and §7.1's regression runs the two against
each other. So the term is wired *and* gets a byte-exact counterpart — the first time this seam
member has had one.

**Also out:** a coincidence or cascade claim (§0.2, §0.3 — costed, deferred on budget); a `pressure()`
read-out for `VKPlate`; the viewer (needs a new claim *and* a new model per the web-viewer plan's own
rule — this batch supplies the claim, so a batch-19 surfacing becomes askable, not automatic); and
PML, non-rectangular rooms, moving ports and viscothermal absorption, which stay on §12H's deferred
list untouched.

---

## 4. The feasibility window — computed up front, not discovered mid-sweep

The room sets `fs` (3-D CFL runs the wrong way; cost `~h⁻⁴`), and this is the first batch where a
*third* multiplier applies: the loaded back-substitution runs **per Picard sweep**, not per step. And
§0.5 measured that multiplier to be much larger than this section first assumed: **29 sweeps at the
build cell**, not the 3–7 seen on the big soft plate — on top of batch 3's measured
loaded-factorization fill growth of 1.55×/3.50×/5.29×.

Because §0.5 moved the build cell to a **0.10 m** plate, the room shrinks with it: a 1.5 m cube gives
`r ≈ 0.5 m` for the directivity arc, which is 10 plate-spans away. Windows from `feasibility.py`,
re-computed for the build cell (≥5 air cells per structural wave at the **p90** mode, ≥8 plate points
per structural wave):

| target | `f_top` | `ka` | `h_air` | `fs` | room | air nodes | est. wall clock |
|---|---|---|---|---|---|---|---|
| coincidence (§0.2) | 12220 Hz | 44.8 | 5.61 mm | 105.8 kHz | 2.5 m | 89.3 M (3.6 GB) | ~5.25 h / 0.5 s |
| **the build cell** | 2950 Hz | 2.70 | 11.4 mm | 52 kHz | 1.5 m | 2.3 M (92 MB) | **~2–3 min / 0.2 s** |

The build cell is two orders of magnitude cheaper, which is why §0.4's claim is the one being made.
It sits **below** batch 5's large-surface knee — deliberately: this batch makes no threshold claim,
so batch 5's `ka = 8` requirement does not bind. What *does* bind is batch 4's directivity band
(`ka = 0.8…2.8`), and the build cell covers it at both the centroid (1.52) and the p90 (2.70).

**The caveat this section used to carry is now closed rather than flagged** — see §0.5. The original
draft costed `ka = 3` while the probe had measured drift at `ka ≈ 0.25`, which is the batch-3 failure
mode exactly. §7.6 still reports the `ka` its numbers were taken at, but now as a confirmation
instead of a hedge.

---

## 5. The discrete scheme

### 5.1 The load folds into `A` once, and the Picard loop is untouched

The air load is `f_load = -Tᵀ p̄_free - Tᵀ R T (w^{n+1} - w^{n-1}) / 2k`. It is linear in `w^{n+1}`
and **independent of `F`**, which is the orthogonality that makes this batch tractable at all. So,
exactly as batches 3–5:

```
    A_loaded = a_bare + (k / 2 rho_s) T^T R T          <- SPD, splu ONCE
    rhs_room = - k^2 T^T pbar_free / rho_s  +  (k / 2 rho_s) (T^T R T) w^{n-1}
```

and for the suspended tier the load matrix and the `p̄` term double, exactly as batch 4.

**`rhs_room` is sweep-invariant, and that is why the hook is cheap.** `rhs_lin` is already computed
once outside the Picard loop; the room's two terms depend only on `p̄_free` and `w^{n-1}`, both fixed
for the step. So the wrapper hands the loop **one constant vector** and only
`couple_factor · l(μw, μF)` varies per sweep. The loop's arithmetic is otherwise `VKPlate.step()`'s
verbatim.

### 5.2 The seam's fourth member is a loop hook — and it returns both fields

```
    solve(lu, rhs_extra) -> (w_next, F_next)
```

Three design rules, each of which is a failure this seam has already documented:

* **The loaded factorization is an argument, never assigned to `vk._lu`.** Mutating the model would
  make bare-vs-loaded invisible and would make §7.1's regression meaningless.
* **Return `F^{n+1}` rather than stashing it on the adapter**, and let `commit(w_next, F_next)` take
  both. Hidden ordered state is precisely what `_PlateSurface`'s docstring warns about; the loop
  makes it worse by putting arbitrary work between the read and the commit.
* **`u_prev` is still read once, before `commit`** — now with an entire fixed-point iteration in
  between, so the ordering comment matters more here than anywhere it has so far.

The hook owns the predictor, the Airy solve, the bracket, the convergence test and the
`n_iters`/`converged`/`last_residual` diagnostics. Duplicating that across the baffled and suspended
wrappers is the four-copies problem §5.2 of batch 5 exists to prevent.

### 5.3 `nonlinear=False` must reduce to batch 3/4, bit-for-bit

`VKPlate(nonlinear=False)` is already bit-identical to `Plate` — the repo asserts it for both
boundaries. Therefore **`RoomLoadedVKPlate(nonlinear=False)` must be bit-identical to
`RoomLoadedPlate`**, and likewise suspended. This is §7.1 and it is the reason the batch splits.

---

## 6. Traps — measured or read out of the source before a line of core code

### 6.1 `rho_v` vs `rho_s` — §2.1's factor of 1000, invisible to every ledger

The one substitution the adapter makes is the one that fails silently. Caught by §7.1 only.

### 6.2 The coupled residual must come from the **committed** state

Batch 4's third detector — the coupled residual against the room's own post-closure pressure jump —
must be recomputed from `(w^{n+1}, F^{n+1})` *after* the loop, not from the last sweep's cached
coupling term. Built the lazy way it reports the **Picard increment** instead, and then it passes or
fails for a reason that has nothing to do with the air load.

### 6.3 The spectral peak is not an observable here (§0.3)

Measured: at `w/e = 3` a peak tracker reads a mode's frequency as 0.53× its own linear value. Any
test that identifies a mode by an FFT peak is unreliable in the loud regime. Use modal projection
under the mass matrix, which is what `probe_cascade.py` does and what §7.6 will do.

### 6.4 Everything batches 3, 4 and 5 already pay for, inherited unchanged

Staircasing, the span-wise footprint check, `blocked_area` vs the live moving surface, the
`1 ≤ index ≤ N−2` face-index range, port disjointness. None re-litigated.

### 6.5 Three detectors, and the family rule says no single one is sufficient

Batch 3's blind spot was the conserved total; batch 4's was the money test; batch 5's was the money
test again for the opposite reason. §7.3 predicts batch 6 splits them along a **new** axis.

---

## 7. Oracles — what must pass

**§7.1 (the split point) — `nonlinear=False` is bit-identical to batch 3 and batch 4.** Both
boundaries, both tiers, with a **non-zero `f_ext`** so the seam's new force path has a byte-exact
counterpart (§3), and with **`sigma > 0` on at least one case per branch**: `a_bare`'s `(1 + σk)`
factor and `rhs`'s `sk * u_nm1` term are where a `rho_s`-vs-`rho_v` slip could hide *asymmetrically*
between `A` and the RHS, and a lossless run never exercises them. Plus batch 3/4's existing pinned
numbers — the `StringPlateBridge` margins `0.2061806714931906` / `0.2061840079056186`, `nnz_growth`,
`lu_nnz` — reproduced to the last digit. **This lands first, alone, as a commit whose entire claim is
"zero new physics, here is the proof."**

*The comparison `Plate` must be constructed with `rho=vk.rho_s`*, which is what
`tests/test_vk_free.py:45` already does. Miss it and §7.1 fails in a way that looks exactly like a
load bug.

**§7.2 — `T = 0` is the bare `VKPlate`.** A zero-area surface factors the plate's own matrix; the
loaded class reduces to the model exactly, nonlinear path included.

**§7.3 — the two-parameter money test, and the detector asymmetry. Measure all three; predict none
of them.** VK conservation holds only *at the Picard fixed point*, so `couple_tol` is an error source
alongside the load — a second axis no previous batch had. Sweep `couple_tol` = 1e-13 / 1e-6 / 1e-3
(the last deliberately under-converged, the negative control) and tabulate **three** quantities:

1. **`|radiated − injected|`** — expected blind, because it is arithmetic on whatever `w^{n+1}` came
   out of the solve (`q = T(w^{n+1} − w^{n−1})/2k`, `p̄ = p̄_free + Rq`, inject `q`), i.e. a property
   of the port relation alone. Batch 5's stated reason it was blind there, on a new axis.
2. **The scene total** — the obvious candidate for the sensitive one, *but not obviously so*: an
   under-converged Picard produces a `w^{n+1}` that is wrong yet **self-consistent**, and
   `VKPlate.energy()` is computed from `(u, u_prev, F, F_prev)`, which the roll keeps mutually
   consistent regardless of convergence. So the total may degrade only slowly, or barely at all.
3. **`last_residual` / `n_iters`** — the only quantity that sees the fixed-point error *directly*,
   and therefore the candidate for the sharp detector if (2) turns out blind too.

Enumerating all three matters because **the interesting outcome is the one where (2) is also nearly
blind** — that would be a *fourth* blind spot on a new axis and a stronger finding than the
two-way split, but it is only claimable if all three were measured rather than two predicted. The
self-certifying claim either way: **loaded drift falls with `couple_tol` at the same rate as
unloaded**, i.e. the air load adds no new error floor.

Per §0.5, `converged` and `n_iters` are asserted **per step**, never sampled at the end: the 2 mm
variant converged on its final step while sitting at the 50-sweep cap throughout.

**§7.4 — the coupled residual at two timesteps**, from the committed state (§6.2), for both tiers.
Batch 4's guard, which is the only one that catches a wrong factor of 2 on the two loaded faces.

**§7.5 — passivity and the ledger.** Lossless room + lossless plate: scene total flat. Lossy: monotone.
`energy()` overridden explicitly on both wrappers (never delegated) for the fourth batch running.

**§7.6 — the headline (§0.4), on §0.5's build cell.** `Lx = Ly = 0.10 m`, `e = 1 mm`, free edge,
broad strike (width 0.20 `Lx`), at `w/e ≈ 0.05` (the frozen control) and `w/e = 3` (drifting) — same
geometry, same room, same strike position, one flag apart. Modal-share drift measured under the mass
matrix (never by spectral peak — §6.3) against radiated fraction over the same windows. States the
`ka` band its energy occupies (centroid 1.52, p90 2.70) and reports radiated numbers as **ratios**,
not magnitudes (batch 2). The observation window is **part of the claim** and is fixed and named.

**§7.7 — directivity of the suspended cymbal**, at batch 4's snapped radii/angles, measured in two
time windows of one run. The claim is that the *pattern* differs between windows for the loud plate
and not for the quiet one. Report the in-plane null for both.

**§7.8 — the compact-safe alternative, held in reserve.** If §7.7's pattern change proves too small
to separate from the sweep's own resolution, the fallback needs no `ka` at all and uses batch 3's
headline as its mechanism: on the **supported** plate, even-index modes have *identically zero* net
volume displacement while radiating 5.6× the (1,1) mode, so modal drift moves energy between modes
the lumped tier calls silent and modes it calls loud. That is measurable as the **monopole read-out
and the true radiated power diverging during a single strike** — at any `ka`, on a compact source.
§0.4 already measured the half of it that needs no room: the monopole is linear in strike amplitude
while the shape content is not. Choose between §7.7 and §7.8 on measurement, not in advance.

---

## 8. Cost budget — owned, not discovered

* §7.1–§7.5 are cheap: small rooms, short runs, no sweeps. Suite-safe — but note that any
  *nonlinear* suite test pays §0.5's sweep multiplier, so keep the guards at low amplitude where the
  count is 3–5, and leave `w/e = 3` to the diagnose script.
* §7.6/§7.7 belong in `scripts/diagnose_airbox_vk.py`, not the suite — §4 budgets ~2–3 min for the
  build cell, so the whole script should land well inside `diagnose_airbox_dipole.py`'s ~7 min / ~1 GB.
* The coincidence window (§0.2, ~5.25 h / 3.6 GB) is **not run**. It is costed so the deferral is a
  decision.

---

## 9. Deliverables

1. **Commit A (plumbing, zero new physics):** `_VKPlateSurface`, the seam's loop hook,
   `RoomLoadedVKPlate` + `RoomSuspendedVKPlate`, and §7.1's bit-identical regression.
2. **Commit B (the guards):** §7.2–§7.5, including the deliberately under-converged negative control.
3. **Commit C (the physics):** `scripts/diagnose_airbox_vk.py`, §7.6/§7.7, and the post-build record
   (§10) with every claim that died on measurement.
4. **Commit D (the record):** `docs/memory/air-box-state.md` and HANDOFF §12H updated.

---

## 10. What the build changed — the post-build record

*(To be written after the build, per the family's convention. §0 is already the plan-stage half of
it: three claims died before this document asserted them.)*
