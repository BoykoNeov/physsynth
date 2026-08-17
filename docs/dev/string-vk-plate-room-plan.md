# The gong on a string, in the room — the three-way chain

`string -> StringVKPlateBridge -> RoomLoadedVKPlate | RoomSuspendedVKPlate -> AirBox`

The last thing the air-box family and the bridge batch both deferred to each other. The bridge
plan (§3) handed it over in one sentence — *"the pieces exist and compose in principle; it is a
third fixed point (string spring, Picard, room load) and belongs to whoever owns the room next"* —
and `HANDOFF.md` §12H's "room coupling" bullet stops at the **linear** distributed body.

Two things turned out to be true before this document asserted anything: **it composes with zero
core edits**, and **there is no third fixed point**. What is left is a physics question, and the
answer to it kills the obvious headline and replaces it with a better one.

---

## 0. What the probes killed before this document was written

Eight plan-stage probes (throwaways, deliberately not committed — every number they produced is
reproduced here). They killed **one headline of mine, one number of my own after I had written it
down**, and sharpened a line of the project's own documentation.

### 0.1 DEAD — "a string-driven gong's radiation pattern depends on how hard it was plucked"

The obvious batch. Air-box batch 6's headline is that *a loud plate's radiation is time-varying at
fixed geometry, and a quiet one's is not* — measured as `sigma_shape` moving **46.0%** at
`w/e = 3` against **1.4%** at `w/e = 0.05`, 33× the control. Replace its strike with a string,
keep the rig and the observable verbatim, and the claim writes itself: *how hard you pluck changes
the gong's radiation pattern*.

It does not. Batch 6's rig, batch 6's `sigma_shape`, batch 6's four 30 ms windows, a suspended
free plate, one pluck against another 60× smaller — and the plate reaching the **same peak
`w/e` = 3.02**:

| | window 1 | 2 | 3 | 4 | spread | modal-share drift |
|---|---|---|---|---|---|---|
| quiet, nonlinear | 1.0000 | 0.9996 | 0.9988 | 0.9979 | 1.0021 | 0.0043 |
| quiet, **linear control** | 1.0000 | 0.9996 | 0.9988 | 0.9979 | 1.0021 | 0.0043 |
| loud, nonlinear | 1.0000 | 0.9996 | 0.9988 | 0.9980 | **1.0020** | 0.0043 |
| loud, **linear control** | 1.0000 | 0.9996 | 0.9988 | 0.9979 | 1.0021 | 0.0043 |

Identical to four digits. What makes this unambiguous rather than merely *too small to see* is the
third row: the loud arm's spread is **below its own quiet control** (1.0020 against 1.0021). There
is no effect here with a sign, let alone a size. Batch 5's resolved-mode restriction changes
nothing (1.0014 against 1.0015), and the radiated fraction over the run is **0.007%** — in this
chain the room is a **read-out, not a variable**.

This is the strongest kind of dead claim this project produces: it died against a **shipped
batch's own numbers**, on that batch's own rig, with that batch's own code. Which forces the
question §0.2 answers.

### 0.2 What replaced it — `w/e` is not an amplitude when the drive is a point force

Batch 6 reaches `w/e = 3` by *strike* and gets a 46% swing. This chain reaches `w/e = 3` by
*string* and gets nothing. The displacement is the same, so the displacement is the wrong number.

| free plate, 30 ms, suspended | peak `w/e` | max plate energy | rigid `{1,x,y}` share |
|---|---|---|---|
| string-driven, 43.4 mm pluck | 2.997 | 6.80e-03 J | **95.52%** |
| struck, Gaussian `w0/e = 1.727` | 2.730 | 1.08e+00 J | 0.00% *(an identity — see below)* |

**158× less energy at matched peak displacement, 191× normalised by `(w/e)²`** (strain energy is
quadratic). 95.5% of the string-driven plate's motion is rigid-body translation and tilt, which
**stretches nothing** — and the von Kármán coupling is a functional of *stretching*, not of
displacement. The bridge is not a strike; it is a point force, and a point force on a free plate
feeds the one part of the motion that stores no strain energy.

Two disciplines this table has to carry, or it will be mis-cited:

* **The struck arm's 0.00% is an identity, not a measurement.** A displacement initial condition
  gives the nullspace no velocity, because there is no restoring force to accelerate it. It is
  structurally zero and proves nothing on its own. The contrast measured on *both* sides is
  string-driven **free** against string-driven **supported** (§0.3).
* **This number was 781× when first written down, and 781× was wrong.** It compared 2.997 against
  a struck arm that rings *up* to 5.213 — two different displacements. Air-box batch 6's own
  docstring warns that magnitudes do not survive a change of conditions; this is that lesson
  landing inside the batch that quotes it.

### 0.3 The two boundaries fail for different reasons — the sentence no earlier batch could write

* **free (the cymbal):** the string reaches the amplitude easily — `w/e = 3.00` — and 95.5% of it
  is rigid-body bounce on the bridge spring, which carries no strain.
* **supported (the gong):** there is no nullspace to hide in, so every bit of displacement is real
  flexure — and the string then reaches only **`w/e = 0.121`** at the same 43.4 mm pluck, **25×
  less**.

Both halves are measured, on the same rig, with the same pluck. The nonlinearity is out of reach
by *opposite* mechanisms on the two boundaries this batch ships.

### 0.4 DEAD — the bridge plan's "third fixed point"

The bridge plan named three (string spring, Picard, room load) and deferred the batch partly on
that. There is **one**. `F = K eta^n` depends only on time-`n` state, so it is sweep-invariant and
enters the RHS *outside* the Picard loop; the room's two terms are sweep-invariant and go into
`rhs_fixed`; `T^T R T` folds into `A` once at construction. Measured in one process, same pluck,
bare bridge against room-loaded bridge:

| pluck | bare sweeps | loaded sweeps | peak `w/e` bare | peak `w/e` loaded |
|---|---|---|---|---|
| loud | 4 | 4 | 3.0102 | 2.9973 |
| quiet | 3 | 3 | 0.0502 | 0.0500 |

**Phrase this as "the room adds no outer iteration," never as "the room does not affect
convergence."** Air-box batch 6 measured that *coarsening* the room breaks the plate's fixed point
(72 sweeps at 57.9 kHz, NaN at 33 kHz). Both facts are true: the room changes the operator the
loop contracts on, and it does not wrap a loop around it.

### 0.5 The coupling saturates in `K` — bridge stiffness is not a lever

Linear plate, 1 mm pluck, 5 ms: `K = 3e3` -> `w/e = 0.0678`; `1e4` -> `0.0691`; `3e4` -> `0.0693`.
A **10× stiffer bridge buys 2%**, and the guard refuses above that (`K = 1e5` -> margin 3.25). Over
that same 10× range the rigid share moves **less than one point** (95.92 / 94.99 / 95.52%). This
is the negative control that makes §1 a one-variable experiment: the thing that decides the
outcome is not the connection.

---

## 1. The claim — band overlap is the lever, not amplitude

Hold the plate at batch 6's validated 1 mm (first elastic mode **326.5 Hz**) and move the
**string's** fundamental across it by changing the string's **length only** — which holds the wave
impedance `sqrt(T rho)` fixed, unlike changing tension. Set the pluck amplitude to hold the
string's initial **energy** constant, so "the same pluck" means the same energy rather than the
same displacement on strings of different lengths.

**At a physically defensible pluck (amplitude 0.5–2.0% of string length):**

| `f1/f_elastic` | L (mm) | amp/L | rigid share | plate energy share | peak `w/e` | sweeps | departure |
|---|---|---|---|---|---|---|---|
| 0.28 | 599.9 | 0.5% | 95.52% | 1.46% | 0.220 | 3 | 1.87e-06 |
| 0.50 | 336.0 | 0.7% | 83.55% | 0.94% | 0.093 | 3 | 1.39e-05 |
| **1.00** | 167.8 | 1.0% | **4.54%** | **18.87%** | 0.151 | 4 | **1.03e-02** |
| 2.00 | 83.9 | 1.4% | 1.73% | 15.59% | 0.082 | 4 | 1.10e-02 |
| 4.00 | 41.9 | 2.0% | 1.55% | 10.50% | 0.032 | 4 | 2.83e-03 |

*departure* is the bridge batch's surviving observable — the plate's displacement history against
the **same chain with `nonlinear=False`**, which is a body that responds exactly proportionally.
Scene-total drift stays 7e-15 … 1e-13 and every step converged, throughout.

> **Whether a string can play a gong *nonlinearly* is decided by band overlap, not by how hard you
> pluck.** At batch 6's gong the plate's first flexural mode is 3.6× the string's fundamental: the
> string drives *below* it, 95.5% of the plate's motion is rigid-body bounce on the bridge, and the
> radiation-shape signature that moves 46% under a direct strike moves 0.20% — below its own quiet
> control. Bring the string's fundamental **onto** the plate's first flexural mode and the rigid
> fraction collapses to 4.5%, the plate's share of the energy rises **1.5% -> 18.9%**, and the
> departure from the plate's own linear self spans **5900× between the worst and best overlap** —
> all at peak displacements of 0.03–0.22 thicknesses, well inside where the model is faithful.

### 1.1 What is a range and what is a trend

`5900×` is **max over min across the sweep, not a monotone rise.** The departure column is
non-monotone: ratios 1.00 and 1.03e-02 and 2.00 at 1.10e-02 are the same number for this purpose,
and 4.00 falls back to 2.83e-03. The minimum is at ratio 0.28. Said as a trend it would be the
`781×` mistake again, one section later.

### 1.2 The optimum ratio moves with amplitude — and that is itself the signature

At the defensible pluck the plate's energy share peaks at ratio **1.00** (18.87%). At a hard pluck
188× larger in energy it peaks at ratio **2.00** (84.15%) instead, with ratio 1.00 at 9.02%. These
are not averaged or reconciled: an optimum that moves with drive amplitude is exactly what a
linear chain cannot do, so the amplitude-dependence of the optimum is a second, smaller claim
riding on the first. Neither arm's peak value is claimed as a magnitude; the *location moving* is.

### 1.3 Three controls, all measured

1. **Bridge stiffness is not the lever** — 10× in `K` moves the rigid share <1 point (§0.5).
2. **The rigid share is amplitude-invariant** — 95.52% and 83.55% read *identically* at plucks
   188× apart in energy, as does the plate's energy share (1.46%, 0.94%). This is the bridge
   batch's linear-body invariance arriving again, and it is what makes the sweep clean: the
   quantity being moved is a property of the coupling, not of the excitation.
3. **The ratio survives the change of conditions where the magnitude does not** — 7100× at the
   hard pluck, 5900× at the defensible one. The defensible arm is the one quoted.

### 1.4 The thin-plate arm — corroboration, never the claim

The same lever pulled the other way: hold the string at 91.3 Hz and thin the plate so its modes
descend onto it.

| `e` (mm) | `f_el/f1` | rigid | peak `w/e` | departure | sweeps |
|---|---|---|---|---|---|
| 1.000 | 3.58 | 95.52% | 3.00 | 3.46e-04 | 4 |
| 0.500 | 1.79 | 83.17% | 10.83 | 1.07e-01 | 7 |
| 0.250 | 0.89 | 30.16% | 54.41 | 7.63e-01 | 22 |

Monotone, consistent, and `K`-independent at the thin end (`K = 2e4` against `5e3`, 4× apart:
rigid 30.16% against 29.53%, departure 0.763 against 0.723). **But it arrives at `w/e` = 54**, far
outside von Kármán's moderate-rotation assumption. It is reported as corroboration and is never
the claim. Moving the *string* is the honest form of the same experiment, which is why §1 does.

---

## 2. What must change — nothing

`StringVKPlateBridge` constructs and runs against `RoomLoadedVKPlate` and `RoomSuspendedVKPlate`
on both boundaries with **no edit to `connection.py`, `airbox.py` or `plate.py`**. The wrappers'
`__getattr__` reaches every name `_stability_margin` assembles from (`theta`, `kappa`, `rho_s`,
`h`, `B` / `W` / `K`, `n_live`, `boundary`, `pickup_index_at`), `step(f_ext=...)` has the
signature the bridge calls, `energy()` is the mixin's override rather than the bare plate's, and
`converged` / `n_iters` / `last_residual` delegate to the loop the seam actually ran.

`_RoomLoadedVKPlateMixin.__getattr__`'s own docstring said no bridge composed with it *yet*, and
named the three things `connection.py` wanted that model #6 lacked. The bridge batch supplied
`rho_s` and `f_ext` and refused `pressure()`, which is exactly the set. **That comment is now
stale and is the one line of source this batch edits.**

### 2.1 The guard is bit-identical, and that is load-bearing

`_stability_margin` reassembles the plate's `G0` block from scratch, and delegation hands over
every ingredient happily — so the guard could be computed against physics that is not happening
and nothing would say so. It is safe for the same reason `test_string_bridge_plate_room_chain`
already pins for the linear plate: `G0 = M + (theta - 1/4) k² S` is a statement about mass and
theta-excess stiffness, while the air load is **dissipative** — it enters `A`, never `G0`.
Measured, all four combinations, bare against loaded, on the **suite's** 8 kHz rig at `K = 3000`:

| | margin | identical to bare | scene drift | sweeps |
|---|---|---|---|---|
| baffled, supported | 7.665222462503e-01 | yes | 2.2e-15 | 4 |
| baffled, free | 7.665222468590e-01 | yes | 2.5e-15 | 4 |
| suspended, supported | 7.665222462503e-01 | yes | 1.5e-15 | 4 |
| suspended, free | 7.665222468590e-01 | yes | 2.4e-15 | 4 |

The margin is linear in `K`, so the tests — which run at the helper's `K = 800` — pin
2.0440593233341828e-01 and 2.0440593249574418e-01 instead. **Only the identity is the assertion**;
the value is a property of the configuration, not of this batch.

Pinning the bit-identity means a future change that makes the load non-dissipative fails loudly
instead of silently mis-guarding — the reason the linear chain's test gives that assertion its
own paragraph.

---

## 3. Scope — and what is deliberately deferred

**In:** all four combinations (baffled/suspended × supported/free), the guard bit-identity, the
conserved scene total, the money test, passivity with loss on either part, the `K = 0` and
`nonlinear=False` regressions, the one-fixed-point measurement, and the band-overlap claim in a
diagnose script.

**Deferred, with reasons rather than silence:**

* **A `sigma_shape` assertion in the suite.** §0.1 is a null result at four-digit agreement. A
  test that passes because nothing happens is not a test; the numbers live in the diagnose script
  and in §0.1.
* **A pitch claim.** Inherited unchanged from the bridge plan §0.2 — it still needs a
  resolved-partial criterion nobody has built, and the room does not supply one.
* **A viewer batch.** Now has two new claims to choose from rather than one.
* **Directivity.** Refused by batch 6 on a costed *contradiction* (a 120 ms window needs a 41 m
  room), and the string does not change the arithmetic.

---

## 4. Traps

1. **The departure observable is computed from two separate runs.** It is the claim's whole
   quantitative content, and a `drive_index` that differs between them — both derive it from
   `pickup_index_at(0.3·Lx, 0.4·Ly)` — moves the number with **no ledger turning red**. This is
   air-box batch 6's `rho_v`/`rho_s` failure class arriving through a different door. Pass
   `drive_index` explicitly, and assert the twin's equals the bridge's.
2. **The linear twin must be built through the same construction path**, `nonlinear=False`, not
   hand-assembled — otherwise the control is not the control.
3. **`w/e` is not an amplitude** (§0.2). Any claim phrased in peak displacement on a
   point-driven plate is measuring the rigid nullspace.
4. **`E_conn` is a fourth channel**, and the family's standing rule is that no single ledger is
   sufficient. §5 says which detector is blind to what.
5. **The room is not stepped by the bridge.** `bridge.step()` then `room.step()`, once, exactly as
   batches 2–6.
6. **`f_ext` must reach the RHS before the solve**, and the bridge force must stay outside the
   Picard loop — inherited, and what makes §0.4 true.

---

## 5. Which detector is blind — the fourth insufficiency

Four batches, four different insufficiencies (batch 3: the conserved total is blind to a wrong
coupling constant; batch 4: the money test is blind to the two-faces factor; batch 5: the money
test is blind to a lagged-explicit load; batch 6: the money test is arithmetic on whatever
`w^{n+1}` came out, so an under-converged solve ports self-consistently). This batch adds the
bridge spring, and the deliberate-falsification pass must count which of the three detectors sees
which slip:

| slip | scene total | `radiated == injected` | guard bit-identity |
|---|---|---|---|
| wrong `K` in `E_conn` only (not in the force) | **CAUGHT**, 2.2e-15 -> 2.2e-01 | blind, 5.4e-15 | blind |
| wrong `beta_s` (the string's reaction impulse) | **CAUGHT**, 2.8e-15 -> 8.8e-02 | blind, 7.8e-15 | blind |
| `rho_v` for `rho_s` in `force_denominator` | blind | blind | **CAUGHT** by the `nonlinear=False` reduction |
| `drive_index` differing between the two departure runs | **blind** | **blind** | **blind** |

The first two rows are the spring's own arithmetic, and only the scene total sees them — the money
test is a property of the port relation alone, and the port never sees the string. The guard is
computed once at construction, so it cannot see a stepping slip at all; what it *does* catch is
the third row, through the `nonlinear=False` reduction that batch 6 built for exactly that.

**The fourth row is the batch's contribution to the family rule, and it is a new kind.** A
`drive_index` that differs between the two runs a departure figure is computed from moves that
figure **1.7×** while every detector sits at machine precision — 5.7e-15 scene drift and a money
gap at 2e-19 on *both* runs. There is nothing inconsistent to detect: each run is internally
consistent, and the error lives in the *pair*. So:

> the three detectors are jointly insufficient against a **comparison**, not only against a
> coefficient.

Which is why the plan requires `drive_index` to be passed explicitly wherever two chains are
compared (§4.1), and why the diagnose script pins it once in `main()` rather than letting each
`build()` derive it.

---

## 6. Cost

Batch 6's rig with a string attached: **10.6–10.9 ms/step**, so 120 ms is ~70 s per case and a
four-case diagnose script is ~5 minutes. The bridge plan's fear — that a third fixed point would
make this unaffordable — does not materialise, because there is no third fixed point (§0.4). The
string is nearly free: it is an explicit leapfrog at 285 nodes against a 53³ room.

---

## 7. Deliverables

* This document.
* `tests/helpers.py`: one builder for the chain, covering both tiers and both boundaries.
* `tests/test_airbox_vk.py`: the chain tests, mirroring
  `test_airbox_surface.py::test_string_bridge_plate_room_chain`.
* `scripts/diagnose_string_vk_room.py`: the band-overlap sweep and the dead `sigma_shape`
  comparison, alongside `scripts/diagnose_airbox_vk.py`.
* The stale comment in `_RoomLoadedVKPlateMixin.__getattr__` (§2).

---

## 8. What the build changed — the post-build record

*(filled in after the build)*
