# The gong on a string — the von Kármán plate as a bridged body (`StringVKPlateBridge`)

`HANDOFF.md` §12H left exactly one thing out of the air-box family and named it: a string
terminated on model #6. `connection.py` reads `plate.rho`, calls `plate.step(f_ext=...)` and
delegates `plate.pressure()`, and the nonlinear plate has none of the three. This batch supplies
the first, argues the second is a one-line addition in the *right* place, and refuses the third.

The physics question §12H asked with it — "what a *linear* 2-DOF stability guard means for an
amplitude-dependent stiffness" — is answered below, but not as asked: §12H's phrasing is stale
(§0.3), and the sharper question has a proof and a measurement that agree.

---

## 0. What the probes killed before this document was written

Four plan-stage probes (`probe.py` … `probe6.py`, throwaways, deliberately not committed — every
number they produced is reproduced here). They killed **three claims of mine and one line of the
project's own documentation** before this document asserted any of them.

### 0.1 DEAD — "the deviation from exact doubling is the headline"

The intended headline: pluck twice as hard; a linear body returns exactly twice the string motion
(scaling by 2 commutes with the ±, ×, ÷ of a leapfrog and an LU back-substitution), so the control
is **bit-exact**, and whatever the von Kármán body does instead is the claim.

The control half is true and better than hoped — `‖2·u_a − u_2a‖_∞ / ‖u_2a‖_∞` is **`0.00e+00`**,
not merely 1e-16, for 2000 steps, on both boundaries, for both `VKPlate(nonlinear=False)` and its
`Plate` twin. That is the strongest control this repo can produce and it is kept (§7.2).

The claim half is not an observable. Measured over a whole run it **saturates**: two waveforms that
have decorrelated in phase differ by ~2× their own amplitude no matter how mildly they decorrelated,
so the number rises, tops out and comes back down —

| pluck | 1e-5 | 1e-4 | 5e-4 | 1e-3 | 2e-3 | 4e-3 |
|---|---|---|---|---|---|---|
| deviation (2000 steps, supported) | 3.6e-2 | 2.24 | 5.53 | 3.23 | 2.82 | 1.14 |

— non-monotone, and 5.53 is not "553% more nonlinear" than 1.14. It is a *detector* (it separates
from zero cleanly) and it is not a *magnitude*. Batch 5's rule arriving again: the crossing is the
claim, the size is not. It survives as a detector on a **short** window, where it is second-order in
amplitude (2.426e-4 → 2.440e-2 for 10× the pluck, i.e. 100×), and that is what §7.2 asserts.

### 0.2 DEAD — "the string goes out of tune with pluck strength"

The musical version, and the one worth wanting: the string is perfectly linear, but it is terminated
on a body whose stiffness depends on amplitude, so the *string's* pitch should depend on how hard it
is plucked — a statement about the string that no linear body can make.

It failed on the estimator, twice, and the second failure is the instructive one.

* **A continuation tracker (follow each partial from the previous amplitude) is wrong**, and the
  linear control is what proved it: on a branch whose spectra are identical up to scale it reported
  shifts of **−5.7%, −9.5%** and then stuck. The narrow follow-window slides downhill into a
  neighbouring peak and never comes back. Anything that reads non-zero on the linear control is
  measuring itself.
* **Two estimators that *do* read exactly `0.000%` on the linear control** — a fixed-window partial
  peak (amplitude-independent by construction) and a normalised-autocorrelation pitch
  (scale-invariant by construction) — then disagree about the nonlinear branch, because there is no
  stable thing to track. The autocorrelation pitch **jumps** 66.93 Hz → 154.08 Hz between the
  smallest and the next pluck while its own periodicity strength falls 0.806 → 0.398; the
  fixed-window partials move `+10.8, +14.7, −16.7, −12.5` % on the low, weak-prominence peaks and
  monotonically `+0.09 → +1.25` % on one strong high one.

There *is* a real detuning in that last column. But "one partial out of twelve moves monotonically
while its neighbours wander" is a claim about a peak-picking rule, not about a string. **A pitch
claim needs a resolved-partial criterion this batch has not built**, and it is deferred rather than
weakened (§3).

### 0.3 DEAD — `HANDOFF.md` §12H's phrasing of the open question

§12H asks "what a *linear* 2-DOF stability guard means for an amplitude-dependent stiffness". The
2-DOF estimate is `StringBodyBridge`'s footgun, kept there as a diagnostic and documented as
necessary-but-not-sufficient. `StringPlateBridge` — the class this batch extends — replaced it with
the **exact** Sherman–Morrison margin two batches ago. The question is therefore narrower and
answerable: *is the exact linear margin still sufficient when the body stiffens with amplitude?*
§5 answers yes, with a reason, and §0.4 records what that answer does not cover.

### 0.4 DEMOTED — the body's energy share as a *magnitude* (killed on the shipping rig, §10.2)

The share of the pluck's energy that reaches the body looked, on the probe rig, like the headline:
for *any* linear body it is **exactly independent of how hard you pluck** (every energy scales as
amplitude², so the ratio does not scale at all — measured `0.717552853`, unchanged in all nine
printed digits across a **400× amplitude range**, on both boundaries), while the von Kármán body's
rose to **+21.1%** at `w/e = 11`, with fitted orders `1.99, 1.97, 1.84`.

The control half is true, rig-independent and kept (§7.4). **The magnitude half did not survive the
shipping rig** and is demoted to a qualitative claim — see §10.2. The short version: peak share is a
**bounded** quantity, so on a rig where the body already takes 82% it has no headroom left, goes
non-monotone, and even changes **sign**. Which sign it takes is a fact about where the drive point
sits relative to the impedance match, not about the nonlinearity.

Two further limits, measured, that apply however it is claimed:

* The `max`-over-run rule is the only run-length-stable one (`0.784622` at 4000, 8000 *and* 16000
  steps); the time-mean and the final value both still climb at 16000 and are not observables. But
  `max` is reached **early**, before the plate's boundary can matter — which is why the supported and
  free branches agree to nine digits at small amplitude. So it measures **the drive point's**
  amplitude-dependent impedance, not the whole plate's.
* The free branch is monotone only up to `w/e ≈ 3`.

### 0.5 WHAT SURVIVED — departure from the linear body, at second order, and a migrated failure

**The headline.** Run the *same* plate twice, `nonlinear=False` and `True`, and measure how far the
string's trajectory moves:

    dist(a) = max_t ‖u_nl(t) − u_lin(t)‖_∞ / max_t ‖u_lin(t)‖_∞

It is identically **zero** for a linear body — not 1e-16, but `0.0`, because doubling the pluck
doubles every quantity in a leapfrog and an LU back-substitution exactly — and it grows at **second
order in the pluck amplitude**, on both boundaries and on both rigs:

| pluck | 1e-5 | 2e-5 | 5e-5 | 1e-4 | 2e-4 | 5e-4 |
|---|---|---|---|---|---|---|
| distance (supported) | 5.19e-3 | 2.06e-2 | 1.22e-1 | 4.08e-1 | 7.39e-1 | 8.10e-1 |
| fitted order | — | **1.99** | **1.94** | 1.74 | 0.86 | 0.10 |

Second-order convergence away from a machine-precision zero is this repo's strongest evidence tier,
and it is available here with no closed form and no oracle. It **saturates near 0.81** once the two
trajectories decorrelate in phase, so only the small-amplitude orders are claimed — §0.1's lesson
applied to §0.1's own measure.

**The finding a later batch will want.** The exact linear stability margin stays sufficient (§5),
but the failure mode does not disappear — it **migrates**. At 90% of the margin, a configuration the
guard *passes*, a hard pluck fails by the plate's Picard iteration hitting its sweep cap, and goes
NaN. The guard is structurally blind to it: it is a statement about a quadratic form, and
non-convergence is a statement about a fixed point. Batch 6 met the same wall from the other side
(coarsening the room broke the plate's fixed point); this is the same wall reached by amplitude.

**The finding a later batch will want.** The exact linear stability margin stays sufficient (§5),
but the failure mode does not disappear — it **migrates**. At 90% of the margin, a configuration the
guard *passes*, the free branch at a 0.1 m pluck fails at step **182** by the plate's Picard
iteration hitting its 50-sweep cap, and goes NaN. The guard is structurally blind to it: it is a
statement about a quadratic form, and non-convergence is a statement about a fixed point. Batch 6
met the same wall from the other side (coarsening the room broke the plate's fixed point); this is
the same wall reached by amplitude instead of by grid.

---

## 1. Why — what a bridged nonlinear body says that no linear body can

`StringPlateBridge` already puts a string on a distributed plate, on both boundaries. Swapping in
model #6 changes one thing, and it is the thing the whole body-coupling family has been unable to
say: **the body's driving-point impedance is no longer a property of the body.** It is a property of
the body *and how hard the string is playing it*. Every body in this repo up to now — modal,
supported plate, free plate, air-loaded, room-loaded — presents the string a fixed termination; a
`RationalAirLoad` makes it frequency-dependent, and even that is fixed per frequency. Here it moves
with the note.

Batch 6 owns the outward-facing half of this (a loud plate radiates differently from a quiet one).
This batch owns the inward-facing half: the *string* on the other side of the spring can tell.

## 2. What must change — and nothing else

### 2.1 `VKPlate.step()` gains `f_ext`, and the coefficient is shared rather than copied

`Plate.step(f_ext=...)` already exists; `VKPlate.step()` takes no force. The addition is:

* `VKPlate.step(self, f_ext=None)` — the force is applied at time `n` and is **sweep-invariant**
  (it depends on `u_end^n` and `w_dp^n`, both fixed for the step), so it is added to `rhs_lin`
  **once, before** the Picard loop. This is strictly simpler than batch 6's room load, which is
  linear in `w^{n+1}` and had to fold into `A`; it is why this batch needs no `solve()` hook.
* The coefficient is `k² f_ext / denominator` with `denominator = rho_s·h²` (supported) and `rho_s`
  (free) — the same expression `_VKPlateSurface.__init__` already computes at `airbox.py:3455/3458`.
  It is exposed as a `VKPlate` attribute and `_VKPlateSurface` is changed to **read** it, so the
  two are the same expression by construction rather than by inspection. That is a behaviour-free
  edit to `airbox.py`, and batch 6's byte-exact regressions are what certify it as one.

### 2.2 `StringVKPlateBridge` is `StringPlateBridge` with the density fixed and no `pressure()`

Same spring, same explicit `F = K η^n`, same Newton's-third-law split, same exact margin. Three
differences, all forced:

* `_stability_margin` reads `p.rho`. `VKPlate` has no `rho` — but it **has `rho_v`, which exists
  and is 1000× wrong** for a 1 mm plate. This is batch 6's silent-failure trap arriving through a
  third door, and §7.1 is what catches it.
* `Plate` exposes the lumped-mass diagonal as `self.w`; `VKPlate` as `self.wdiag`. Anything lifted
  from `Plate` carries the wrong attribute name and fails loudly — the good case.
* **No `pressure()`.** Batch 6 deliberately gave model #6 no monopole read-out and measured why:
  for a gong it is 3e-7 of the truth, and for a suspended cymbal it moves the *wrong way*.
  Re-adding it on either class to satisfy a protocol would re-open a read-out a measured batch
  retired. The bridge's docstring points at `RoomLoadedVKPlate` / `RoomSuspendedVKPlate` instead.
  Checked: nothing calls `.pressure()` on a body generically — every call site constructs a
  specific bridge — so this is a documentation duty, not a composition break.

### 2.3 Nothing else

No change to the spring, the string, the energy decomposition, `simulate`, the air box's ports, or
the web backend.

## 3. Scope — and what is deliberately deferred

**In:** both boundaries (`supported` = gong, `free` = cymbal), lossless conservation, passivity with
loss on either part, the `K = 0` and `nonlinear=False` regressions, the exact margin, the energy
share claim with its order, rigid-mode immunity, and the migrated failure mode.

**Deferred, with reasons rather than silence:**

* **A pitch claim** (§0.2) — needs a resolved-partial criterion this batch has not built.
* **A viewer batch** — §12H names it separately, and it now has two new claims to choose from.
* **`StringVKPlateBridge` inside a room** (bridge → `RoomLoadedVKPlate`). The pieces exist and
  compose in principle; it is a *third* fixed point (string spring, Picard, room load) and belongs
  to whoever owns the room next.

## 4. The scheme

The string is an explicit leapfrog: its reaction is the post-step impulse `u_end -= β_s F`, exact
for a linear scheme. The plate is implicit, so `+F` enters the RHS before the solve. The three
per-step energy increments telescope exactly as in `StringPlateBridge`, because a time-`n` source
contributes `k F δ_{t·} w_dp` to the plate's energy *regardless of θ and regardless of the membrane
term* — the von Kármán coupling is µ-averaged and never touches the source's placement.
`E_conn = ½ K η^n η^{n-1}`, unchanged.

## 5. Stability — the argument, and what it does not cover

**The exact linear margin remains sufficient.** The conserved total is
`E_lin + H_mem + E_conn`, and `H_mem = ½(H(F^{n+1}) + H(F^n))` with each term `(1/2Y)‖∇²F‖² ≥ 0`.
Unlike the θ-weighted bending potential, which carries an indefinite cross-time term, the membrane
energy is a **sum of two squared norms with no cross-time coupling**: it cannot subtract. So if
`G0 − (k²/4) K aaᵀ` is positive definite the total stays coercive, and the Sherman–Morrison
condition already assembled by `StringPlateBridge` is the whole linear story.

Measured: at **99%** of the exact ceiling, driven to `w/e` = 7.1 (supported) and 35.8 (free), the
drift is 4.07e-13 and 1.36e-12 and Picard converges in ≤9 sweeps throughout.

**It is sufficient, not tight** — it may refuse configurations that would run — and it is **not the
whole safety story** (§0.4): conservation holds only *at* the fixed point, so the guard's blind spot
is non-convergence. The bridge therefore surfaces `n_iters` / `converged` / `last_residual` from the
plate, read **per step**: batch 6's warning that a plate can sit at the sweep cap all run and still
converge on its last step applies here verbatim.

## 6. Traps

1. **`rho_v` for `rho_s` in the guard** — 1000×, and every ledger stays green (§2.2). §7.1 catches it.
2. **`self.w` vs `self.wdiag`** — loud, but only if the free branch is actually exercised.
3. **The force must enter before the solve.** A post-solve correction is invalid: the `A`-solve
   couples all nodes. (Inherited from `StringPlateBridge`, unchanged.)
4. **`F^{n-1}` is captured before the step.** Batch 6's ordering rule; the Picard loop reads the
   cached `F_prev`, and a bridge that steps the plate twice per iteration would corrupt it.
5. **The string must run at `λ < 1`.** Unchanged, and unrelated to the nonlinearity.

## 7. Oracles — what must pass

**7.1 The regression, and it must be checked for the ability to fail.**
`VKPlate(nonlinear=False)` bridged to a string must be **bit-identical** — state, energy *and*
stability margin — to `Plate(rho=vk.rho_s, …)` bridged to the same string, on both boundaries.
`tests/helpers.py::vk_linear_twin` already builds the twin correctly and is reused rather than
re-derived. Then, as batch 6 did: **deliberately falsify it** — substitute `rho_v` for `rho_s` and
confirm it goes red. Commit a checkpoint first, and revert the edit itself, never with
`git checkout -- <file>`.

**7.2 The bit-exact 2× control, and its second-order failure.** `‖2·u_a − u_2a‖_∞ / ‖u_2a‖_∞` is
exactly `0.0` on the linear branch (measured, 2000 steps, both boundaries) and second-order in
amplitude on the nonlinear one, on a short window (§0.1).

**7.3 Energy.** Lossless total drift `< 1e-10` on both boundaries at `w/e` ≳ 5 (measured 4.1e-13
supported, 3.3e-13 free); monotone decrease with `sigma` on either part; `K = 0` bit-identical to
the uncoupled parts; the string's own energy demonstrably *not* conserved.

**7.4 The claim (§0.5).** Departure from the same plate's linear self, second order in the pluck
(asserted `1.6 < order < 2.2` on the two smallest ratios, both boundaries). Plus the share's
*control* half, which survives as its own statement: the linear body's share is amplitude-invariant
to ~machine precision and the von Kármán body's is not, with neither size nor sign claimed (§10.2).

**7.5 Rigid-mode immunity (free).** The Monge–Ampère bracket is built from second derivatives, so
`l(w,w) ≡ 0` on span{1, x, y}: the nonlinearity is exactly blind to the rigid modes and
`StringPlateBridge`'s no-drift argument survives untouched. Measured: from a pure tilt, `|F|` is
3.0e-21 after 500 steps, the plate holds the exact tilt to 6.5e-14 and its energy is −3.2e-19.
(`F(w^{-1})` is 1.76e-40 rather than exactly 0 — the consistent-start acceleration term leaves a
rounding-level non-rigid residue. Stated, not hidden.)

**7.6 The migrated failure mode.** A configuration at 90% of the margin, plucked hard enough,
must be shown to fail by non-convergence rather than by instability (measured: free, 0.1 m pluck,
first non-converged step 182, sweep cap 50, NaN) — and the guard must be shown to have passed it.

## 8. Cost

The direction this family's cost normally runs is reversed here: **no room, no 3-D CFL, no `h⁻⁴`.**
The plate wants a high `fs` for the nonlinearity and `string.k == plate.k` is asserted, so the
string ends up heavily oversampled at `λ < 1` — which is only more 1-D nodes, and cheap. Measured on
the probe rig (16×16 plate, 140-node string, 48 kHz): 8000 steps ≈ 6–11 s nonlinear, ≈ 1 s linear;
24000 steps ≈ 28 s. Picard costs 3–9 sweeps at the amplitudes claimed. No under-resolution pressure,
so none of batch 6's defensiveness about it is inherited.

## 9. Deliverables

1. `physsynth/core/plate.py` — `VKPlate.step(f_ext=None)` + the shared force denominator.
2. `physsynth/core/airbox.py` — `_VKPlateSurface` reads that denominator (behaviour-free).
3. `physsynth/core/connection.py` — `StringVKPlateBridge`.
4. `tests/test_vk_connection.py` — §7.1–§7.6, both boundaries.
5. `tests/helpers.py` — a `make_vk_plate_bridge` builder alongside the existing bridge builders.
6. This document's §10, the post-build record.

---

## 10. What the build changed — the post-build record

30 tests, 19 s, both boundaries. Everything in §2 shipped as planned; §0.4's headline did not
survive its own shipping rig, and two things the plan did not anticipate came out of the rig itself.

### 10.1 What survived exactly as planned

* **The `f_ext` placement.** Sweep-invariant, added once outside the Picard loop, no `solve()` hook
  needed. `VKPlate.step()`'s signature now matches `Plate.step()`'s.
* **The shared denominator.** `VKPlate.force_denominator` is read by both `VKPlate.step()` and
  `_VKPlateSurface`, and the sharing paid for itself immediately: falsifying it (§10.3) turns **46**
  tests red across *both* consumers, where a duplicated expression would have failed only one.
* **The guard.** Sufficient, for the stated reason, and measured at 95% of the ceiling under a
  strongly nonlinear run. The nonlinear bridge's margin is bit-identically the linear twin's.
* **Rigid-mode immunity.** Started in a pure tilt the free plate stays tilted: `F(w^0)` is exactly
  `0.0`, `|F|` stays under 1e-18 for 400 steps, and the energy is under 1e-15.
* **No `pressure()`.** Asserted as an absence, on both the bridge and `VKPlate`, so a later batch
  that adds one has to argue with a test rather than with a comment.

### 10.2 DEAD — §0.4's energy share as a magnitude, killed by the rig it had to ship on

The probe rig gave `+21.1%`, monotone, orders `1.99/1.97/1.84`. The shipping rig gives orders
`1.86, 1.72` and then a **sign change**: `+2.76e-4, +1.00e-3, +4.82e-3, +7.84e-3, +6.34e-3,
+2.42e-2, −7.26e-3`.

The reason is structural and was visible in the plan if it had been read as a bound: **peak share is
a bounded quantity**. On the probe rig the body took 72% and had room to move; on the shipping rig
it takes 82% and does not. A bounded observable near its ceiling cannot carry an order.

What is kept is the half that does not depend on headroom — the **control**: for any linear body the
share is amplitude-invariant to ~machine precision (asserted `rel=1e-12` across 100×), and the von
Kármán body's is not (asserted only that it *moves*). Neither size nor sign is claimed. This is the
family's own rule arriving for the fourth time: ratios survive, magnitudes do not.

Its replacement (§0.5) was chosen for the opposite property — distance from the linear body is
**unbounded below saturation**, so it has an order to measure — and it reproduces `1.99/1.94`
(supported) and `1.97/1.87` (free) on the shipping rig, matching the probe rig.

### 10.3 The regression was checked for the ability to fail — three ways, and one is subtle

Committed a checkpoint first, and reverted each edit *itself* rather than with `git checkout --`.

| falsification | result |
|---|---|
| guard reads `rho_v` instead of `rho_s` | **6 red** (the bit-identity regression + the margin test) |
| `force_denominator` reads `rho_v` | **46 red**, across the bridge *and* the air-box VK suites |
| a pure **reassociation** of the `f_ext` term — `k²·f/d` → `(k²/d)·f` | **2 red** |

The third is the one worth recording. It is mathematically an identity and changes only the rounding,
and it is caught — but only by 2 of the regression's 4 parametrisations (`[0.0-free]` and
`[2.0-supported]`, not the other two). Batch 6 measured the same thing (5 of 8 there). **A
bit-identity regression's sensitivity to reassociation is parametrisation-dependent**, which is the
argument for parametrising it over both the boundary *and* the loss rather than picking one case.

### 10.4 The rig's size is set by the fixed point, and it costs the audio band

Not anticipated by the plan, and it is §8's cost claim inverted. The plan said cost runs the right
way here — no room, no 3-D CFL — and that is true of *wall-clock*. But the membrane coupling's
difficulty scales like `k²/h⁴`, so **shrinking the plate makes the Picard fixed point harder, fast**:
40 cm converges in ≤ 13 sweeps out to `w = 9 e`, while 8 cm hits the 50-sweep cap by `w = 6 e` and
takes the energy drift with it (2.8e-2).

The consequence is a real limit, stated in `helpers.py` rather than hidden: **this plate's modes sit
below the audio band** (`f11 ≈ 3 Hz`). Audio-range modes at 0.1 mm need a ~7 cm plate (`f ∝ e/L²`) —
exactly the size that will not converge — and thickening the plate instead costs about `e⁵` in the
energy needed to reach `w ≈ e`, which a string does not have. **Audio-band, string-drivable and
Picard-convergent cannot all hold at this sample rate.** This is a rig for the mechanism, not an
impression of a gong, and a batch that wants the impression needs a different exciter (a mallet
delivers the energy a pluck cannot) rather than a bigger budget.

### 10.5 A trap the plan did not have: two things called `rho`

`make_vk_plate_bridge` first took a single `rho` passthrough. `VKPlate` calls its *volumetric*
density `rho` and the string helper family calls the *string's linear* density `rho`, so one
argument silently retuned the wrong object — caught immediately, because it sent the stability
margin to **204** and the constructor refused it. Split into `rho` (string) and `rho_plate`.

It is the batch's own `rho_v`/`rho_s` trap one level up, and it failed *loudly* only by luck: the
guard happened to reject the result. Had the mistuned string still been stable, it would have been
exactly the silent kind.

### 10.6 Numbers a later batch will want

* Departure order: 1.99 / 1.94 (supported), 1.97 / 1.87 (free); saturation ceiling 0.81.
* Picard sweeps vs amplitude on the shipping rig: 6 / 8 / 10 / 13 at `w/e` = 0.8 / 1.5 / 3.4 / 9.0.
* Non-convergence at 90% of the margin: first failure at step 182 (probe rig, `w/e` unbounded).
* Guard margins are **identical** for `supported` and `free` on this rig (0.2139) — the drive point
  is an interior node, so it carries the same lumped mass either way.
* Cost: 30 tests in 19 s; ~2 s per 2500-step nonlinear run at 225 live nodes.
