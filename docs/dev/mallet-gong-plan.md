# Mallet–gong collision — model #7g, the nested solve

> **Status: BUILT & GREEN (2026-09-06).** `MalletVkPlate` in `crates/physsynth-core/src/mallet.rs`,
> bound as `MalletVKPlate` in `crates/physsynth-py/src/mallet.rs`, re-exported by
> `physsynth/core/mallet.py`.
> Tests: `crates/physsynth-core/tests/mallet_gong.rs` (12 native bars),
> `tests/test_mallet_gong.py` (22).
> Helpers: `make_mallet_gong`, `gong_linear_twin` in `tests/helpers.py`.
>
> Closes the half of HANDOFF §14.1 that `docs/dev/mallet-plate-plan.md` §6 deferred: *"a batch
> wanting the gong impression still needs a mallet, not a budget."* The mallet now exists.
>
> **This document is mostly about the two things §6 got wrong**, because it got the *algorithm*
> right and then priced it as though the outer loop had to work as hard as the inner one. It does
> not: measured cost is **1.8–2.3×** a bare gong step, not the ten to a hundred predicted. And the
> outer iteration **does** have a closed-form derivative.

---

## 1. The problem, and why model #7p cannot be pointed at a `VKPlate`

`MalletPlate` collapses a whole implicit plate into one number. Its step is affine in the external
force, so the field's response to a unit drive-point force is one constant vector — the influence
column — and the scalar contact equation closes on one entry of it, `g_s = influence[node]`.

The von Kármán step is a *nonlinear* solve. There is no such column, and superposition — the entire
basis of the collapse — is gone. Writing `Psi(f)` for the strike node after a **full** nonlinear
plate step driven by `-f e_node`, the step to solve is

```
eta^{n+1} = Psi(f) - z_free - g_h f ,        f = phi'(eta^{n+1}, eta^n)
```

one scalar equation, every evaluation of which costs a complete Picard or Newton plate solve. That
is the "solve wrapped around a solve".

## 2. What ships: a chord whose frozen tangent is the LINEAR plate's

The iteration is a chord — Newton with a frozen derivative — and the derivative it freezes is the
linear plate's own drive-point admittance, built from the same `A^-1 e_node` back-substitution
`MalletPlate` uses, against `VkParams::lin`'s factorization. One outer iteration is:

```
u_eff   = Psi(f_j) + g_s f_j     the "effective free node": where a LINEAR plate carrying this
                                 tangent would have had to start to land on Psi(f_j)
f_{j+1} = solve_contact(u_eff - z_free, eta^n, g = g_s + g_h, ...)
```

`solve_contact` is **model #7's, unchanged**. That is the point of the manoeuvre: the proven contact
root-find — with its safeguarded Newton, its scanned bracket, its `brentq` fallback and its `0/0`
Taylor branch — is reused rather than re-derived inside a nonlinear solver. The only new arithmetic
in the whole model is `u_eff`.

Three properties fall out, and each is a test.

**It degenerates exactly to model #7p.** With `nonlinear = false`, `Psi(f) = u_free - g_s f`
identically, so `u_eff = u_free` **whatever `f_j` is**: the first residual is zero and the loop
exits at one iteration. `n_outer == 1` on every step is therefore a *structural* claim, and it is
the one a sign error in `f_ext = -f e_node` cannot survive — with the sign flipped, `u_eff` stays a
function of `f` and the count goes to two or more. The field agreement is `2.08e-14` relative and
**not bitwise**, for §2 of the linear plan's reason one level out: this model puts the force in the
right-hand side *before* the solve while `MalletPlate` adds `influence * f` *after* it, and a sparse
LU back-substitution is not a linear map over doubles. The test asserts the *absence* of bitwise
agreement, because agreement there would mean one of the two stopped doing what its header says.

**A miss is exact.** A contact solve returning `force == 0.0` short-circuits to the force-free
advance and returns it *unmodified*, so a mallet that never lands leaves a trajectory bit-identical
to a bare gong's. That is control flow, not floating point: driving the plate with a zero force
*vector* would add `+0.0` into every right-hand-side entry, which is the identity for every double
except `-0.0`.

**The seed is the linear model's answer.** The force-free advance that starts each step is also the
miss's answer, which is why it is a solve rather than a continuation from the previous step's force
— a seed taken from history could not give the miss a bit-identical trajectory. The first trial
force is then exactly what `MalletPlate` would have applied, so the outer loop begins already close
and only has to close the *nonlinear* part of the gap. This is why the iteration counts below are
smaller than a contraction from an `O(1)` start would give.

## 3. The cost, which is the batch's headline and a falsification

`docs/dev/mallet-plate-plan.md` §6 and HANDOFF §14.1 both priced this at "roughly ten to a hundred
plate solves per timestep". Measured against a **bare `VkPlate` step taken from the mallet's own
state at that moment** — what this plate would have cost with no mallet on it, at the amplitude the
mallet actually drove it to — over 2000 steps:

| mallet | strike | whole run | during contact | outer mean | outer max | in contact |
|---|---|---|---|---|---|---|
| 20 g | 3 m/s | **1.89×** | 2.40× | 1.74 | 2 | 1166/2000 |
| 50 g | 6 m/s | **1.95×** | 2.43× | 1.84 | 2 | 1201/2000 |
| 200 g | 6 m/s | **2.29×** | 2.36× | 1.86 | 2 | 1810/2000 |
| 50 g | 12 m/s | **1.79×** | 2.47× | 1.89 | 2 | 911/2000 |

A miss costs exactly one bare step, which is what pulls the whole-run figure below the in-contact
one; the 200 g row is highest overall because a heavy head stays in contact longest, not because its
contacts are dearer. `n_solves` — back-substitutions, counted from the driver's own arithmetic
rather than off any solver's report — is the number asserted, because it is the portable one; wall
clock is secondary and machine-dependent.

**Why the prediction missed by two orders of magnitude.** Superposition fails, so the influence
column is not the *answer*; but it fails **by a little**, because the von Kármán coupling reaches
the one-step response scaled by `k²`. At audio rates that perturbs the drive-point response by a
part in ten thousand while it is busy reshaping the *trajectory* over thousands of steps. So the
linear column is an excellent frozen tangent exactly where it is a useless exact solution. §6 read
"not exact" as "not useful", and the distance between those two readings is the whole cost story.

## 4. The instrument: the outer tangent is closed-form

§6 also said there is "no closed-form derivative for the outer Newton (a secant or a bracket
instead)". There is one, and it is this crate's own. Differentiating the fixed point
`w = A^-1(rhs_lin(f) + c l(w_bar, F_bar))` gives `dG/df = +influence` against `dG/dw = J`, `J` being
exactly the operator `plate::VkCoupledStep::jacobian_vector` applies — the one the Newton batch
(`docs/dev/vk-newton-plan.md` Part 1) asserted against a finite difference. So

```
dw/df = -J^-1 influence        d eta^{n+1}/df = -( [J^-1 influence]_node + g_h ) = -g_exact
```

`vk_drive_point_tangent` computes it in one GMRES against that product (3–5 Krylov products
measured). It does **not** ship in the step — a GMRES per outer iteration costs more than the chord
it would replace — but it is what turns the iteration count from a story into a measurement, since
the chord's contraction factor is bounded by `|1 - g_exact/g|`:

| strike | 5 g | 20 g | 50 g | 200 g |
|---|---|---|---|---|
| 1 m/s | 2.2e-6 | 7.5e-6 | 9.8e-6 | 1.1e-5 |
| 3 m/s | 3.2e-5 | 4.0e-4 | 8.8e-4 | 1.3e-3 |
| 6 m/s | 1.7e-4 | 1.8e-3 | 5.1e-3 | 1.0e-2 |

Two to five decades of contraction per iteration, which is why two iterations reach `1e-13`.

**The mallet enters the contraction only through `g`, and that is asserted bitwise.** The numerator
`|g - g_exact| = |[(I - J^-1) influence]_node|` is a property of the plate's deflection in which no
mallet quantity appears at all. Driven to one plate state and then asked by five mallets spanning a
**200× mass range**, it comes back as `7.2757e-10` every time — the same double. The whole mass
dependence is therefore `1/g`, and it *saturates*: as `M → ∞`, `g → g_s`, so the bound rises from
3.29e-3 to 5.39e-3 and stops. A heavier mallet does converge more slowly, by a bounded factor
(1.64× across that range, which is exactly `g(5 g)/g(1 kg)`), not by an amount that grows.

This sharpens a prediction made before the measurement. "Heavy is worse" was right; the first
version of the measurement appeared to show it growing 60× with mass, and that was mostly the
*amplitude* — a heavier head at the same speed delivers a much larger peak force. Holding the plate
state fixed is what separates them, and it is the difference between a mechanism and a correlation.

`response` is returned **separately** from `g_exact` for a reason worth keeping: recovering it by
subtracting `g_h` back off `g_exact` does not return the same double when the two differ by orders
of magnitude, which is exactly what a heavy mallet is. A bitwise claim about a plate-only quantity
must not be routed through a mallet-sized addition and back.

## 5. Energy: the discrete gradient telescopes through a nested solve

The plate is implicit and the contact force enters its right-hand side, so dotting the θ-scheme with
`w^{n+1} - w^{n-1}` gives the same discrete power `f · δ_t· u` a force does on a membrane. The von
Kármán coupling telescopes on its own and adds nothing to the contact's ledger, so

```
H^n = E_lin^n + H_mem^n + 1/2 M (delta_t- z_H)^2 + 1/2 (phi(eta^n) + phi(eta^{n-1}))
```

is constant at `sigma = 0, lam_h = 0` and monotone above either. Measured over 2000 steps, supported,
lossless, at the shipped `outer_tol = 1e-13`:

| strike | drift | peak `w/e` | membrane share of `H` | strike energy |
|---|---|---|---|---|
| 3 m/s | 5.3e-12 | 1.70 | 1.8% | 0.225 J |
| 6 m/s | 2.7e-12 | 2.84 | 3.3% | 0.90 J |
| 12 m/s | 6.7e-12 | 4.39 | 4.1% | 3.6 J |

Against the project's `1e-10` contract. Passivity: worst per-step rise 1.2e-15 with `sigma = 2` and
4.8e-15 with `lam_h = 5`.

**The membrane share is asserted, not the velocity.** A conservation bar on a nonlinear plate that
never left the linear regime is a re-test of the linear θ-scheme wearing this model's name — which
is also why `make_mallet_gong` defaults to a **50 g** head rather than the 20 g the membrane and
linear-plate models use: 20 g at 3 m/s leaves the membrane term at 0.6% of the total.

What is new is that the identity now holds only as far as the **outer** loop converges, so this bar
is a function of `outer_tol` — not of `newton_tol`, which is what model #7p self-certifies against.
The self-certification is the same shape (`test_vk_energy.py`'s): drift falls when the solver is
asked for more, over two decades of `outer_tol`.

## 6. The prediction about a tolerance floor, which was wrong

Going in, the expectation — reasoned from the structure, and stated before it was measured — was
that every `Psi(f)` carries the plate solve's `O(couple_tol)` error, so the outer residual would
plateau near `couple_tol · ||w|| / (g · force_scale)`. On the shipped gong that predicts 9.7e-14,
which would put the floor *above* a `1e-14` request.

**It does not happen.** At `outer_tol = 1e-16` the loop reaches that tolerance in a mean of 2.6
iterations, and does so at `couple_tol = 1e-9` exactly as readily as at `1e-15`: four orders of
magnitude of the plate's own tolerance move the outer iteration count in the *second decimal place*
(2.59 / 2.60 / 2.59 / 2.60).

The reason is that the inner error is a **bias, not noise**. The inner solve is cold-seeded from
`2 w^n - w^{n-1}` and is otherwise deterministic, so a given `f` gives the same `Psi(f)` every time;
the chord then converges to the perturbed map's own fixed point, to whatever precision it is asked
for. A floor would need `Psi` to be *discontinuous* in `f` — the shape `docs/dev/`'s tension-string
batch found when a root-find branched on a reduction's iteration count — and it is not. Sweeping the
trial force by parts in `1e12` across the worst step of a run leaves the inner sweep count pinned at
5 and moves `Psi` smoothly.

What *is* real is much smaller, and it is the reason a stagnation exit ships. About **one in-contact
step in nine hundred** stops contracting around `5e-14` and stays there. It is not attributable to
either inner tolerance: tightening `couple_tol` removes it, tightening `newton_tol` removes it, and
loosening `newton_tol` a thousandfold does not make it worse — every one of those perturbs the
trajectory, and the step moves with it. So it is a single failure to contract rather than a property
of the model. Without a stagnation exit that step spent all twenty of its outer iterations — twenty
full plate solves to discover that the nineteenth was no better than the second. With one, spelled
as `!(residual < prev)` so a `NaN` leaves rather than iterating (the same shape and the same reason
as `collision::solve_contact`'s own Newton exit), it spends four.

**Why `outer_tol = 1e-13` is the default.** Tighter is available and does buy fidelity — the drift
falls to ~6e-13 at `1e-14` and then stops, because at that point the limit is not the solver at all
but the energy read-out's own reduction rounding over 361 live nodes plus 441 stress-function nodes.
The default is one decade looser than that knee because it is the tightest setting at which **no
step stalls**, and a shipped configuration that takes a stagnation exit once per run makes every
"did it converge" assertion a probabilistic one. The knee is documented so a caller who wants the
last half-decade knows exactly what it costs (~20% more outer iterations and one stall in 8000).

## 7. The physics payoff, with a bit-exact control

Model #7p's finding was that **the felt exponent is the only source of dynamic timbre**: at
`alpha = 1` the head is a mass, the felt a linear spring, and the one-sided switching is
scale-invariant, so the whole system is homogeneous of degree one and a four-times-harder strike
gives a four-times-larger response and nothing else.

That is a statement about a **linear resonator**, and the gong is where it stops being true. The
measurement isolates the two causes by removing one of them:

* linear plate, `alpha = 1`: departure from an exactly scaled response = **0.000e+00**
* gong, `alpha = 1`: **2.12**

The control is an **identity, not a tolerance**, and deliberately so — the loud/quiet ratio is 4,
an exact power of two, so scaling a double by it is exact and a system homogeneous of degree one
reproduces the loud trajectory from the quiet one *to the bit*. Any other ratio would put a rounding
floor under the control and turn the identity into a bound. With the felt thus contributing exactly
nothing, the gong's 2.12 is the plate's and nothing else's.

*How* the tone changes is the crash: energy cascades up the spectrum as the strike hardens. A
power-weighted spectral centroid across a **16× dynamic range** (0.75 → 12 m/s):

* linear plate, `alpha = 2.3`: **+0.16%** — this is the felt, #7p's effect, and it is real
* gong, `alpha = 2.3`: **+74%**

A factor of **450** between the two sources of dynamic timbre, where the scaled-response test splits
the same pair by under two (1.17 against 2.12). Two detectors, two very different splits of one pair
of causes — the air-box family's standing rule that no single detector is sufficient, arriving on a
contact model. The centroid's near-blindness to the felt has a reason worth keeping: on a plate it
is dominated by the lowest partials, whose comb a shorter contact pulse barely reshapes.

## 8. Failure, and who is blamed for it

Two solvers are nested, so the outer one can report the inner one's failure **in its own words** —
and its own words are actively misleading. Mapping the model's boundary (a low sample rate and a
hard strike), the Picard loop stops converging and the very next thing that happens is that the
contact solve scans its bracket six times and finds no sign change. `ContactError::NoRoot`'s text
calls that *"impossible for the monotone convex-potential force"*. It is. The force is still
monotone and still convex; what is not is `Psi`, because the field the residual was built from is
the output of an iteration that did not finish. Handed to a caller unqualified, that message sends
them to look for a bug in the one place the fault is not.

So `VkContactError::Contact` carries `inner_converged`, and the message says which solver actually
gave up and what to change. It is a native bar with a fixture chosen to break the inner solve, and
the bar asserts the attribution rather than just the failure.

## 9. What does not carry over from model #7p

* **Rectangles only.** `VkParams::new` builds its linear half with `Domain::Rectangle` hardcoded, so
  §3's "the outline is free" is a property of the *linear* mallet. No circle, no guitar outline.
* **No `accel`, so no `pressure()`.** `VkPlate` carries `u`, `u_prev` and the two stress-function
  caches and nothing else, so §5's stale-`_accel` trap cannot arise here — and neither can the
  radiated read-out.
* **The free branch is still a read-out bar** — but it was invisible at first. A point strike feeds
  the free plate's `{1, x, y}` rigid nullspace, the cymbal recoils for ever after, and the bending
  form's cancellation error is quadratic in the rigid drift; the von Kármán term adds nothing to it,
  because the Monge–Ampère bracket annihilates the nullspace too. At the `outer_tol = 1e-12` the
  model was first built with, **the nested solve's own error was larger than the cancellation error**
  and the free branch measured no worse than the supported one (ratio 0.9 at 500 steps). Tightening
  a solver tolerance is what uncovered a defect in a *read-out*. At `1e-14` the law is plain —
  `drift / rigid²` = 7.3e-8 / 6.9e-8 / 7.0e-8 at 2000 / 4000 / 8000 steps — and the supported branch
  on the identical mallet is the control that makes it an attribution: 5.9e-13 against 3.3e-10, a
  factor of 550.

## 10. Numbers a later batch will want

* Rig: 0.4 m square, 1 mm steel (`E = 2e11`, `nu = 0.3`, `rho = 7800`), simply supported, `N = 20`,
  **361 live nodes**, `fs = 48 kHz` (free, *not* a CFL — the θ-scheme is unconditionally stable),
  `k = 2.0833e-5`, `h = 0.02`. Fundamental ≈ **30.1 Hz**: physically right for a thin supported
  plate this size, and low. A 0.2 m plate puts it near 120 Hz and runs at `w/e` up to 4.0.
* At 50 g on `K = 5e4`, `alpha = 2.3`: `g_s = 1.344278e-07`, `g_h = 8.680556e-09`,
  `g = 1.431084e-07`, `steps_per_contact = 150.8`, `force_scale = 14400 N`, strike node 138 at
  `(0.12, 0.16)`.
* `w/e` reached: 0.50 at 1 m/s, 1.70 at 3, 2.84 at 6, 4.39 at 12. The membrane term is 1.8% / 3.3%
  / 4.1% of the total at 3 / 6 / 12 m/s.
* **The inner Picard fixed point is not what limits this model at audio rates.** Every scene above
  runs with zero non-convergent inner solves, up to `w/e = 7.8`. The wall is reachable, but only by
  lowering the sample rate: at `fs = 8 kHz` and 30 m/s the inner solve fails and the run dies. The
  `k²` in the coupling is what buys the headroom, and it is the same `k²` that makes the chord cheap.
* **Newton buys nothing here** — measured at the shipped settings across three mallets, not at
  one. `couple_method="newton"` reaches the same peak deflection to eleven figures
  (`w/e` agrees to 6e-12 .. 2e-11) and costs **1.47x to 1.72x** the back-substitutions -- 44404
  against 25842 at 20 g / 3 m/s, 50724 against 33780 at 50 g / 6 m/s, 71774 against 48798 at
  200 g / 6 m/s, over 2000 steps each. Picard is the right inner solver at the amplitudes a
  mallet reaches; Newton's advantage is at the amplitudes a *pluck* reaches, which is what
  `vk-newton-plan.md` §14.2 measured. (The first version of this bullet was a single fixture from
  the first probe, before `outer_tol` moved to 1e-13 and before the stagnation exit existed — a
  margin measured at one fixture is a claim about one fixture.)
  **And this bullet is itself such a claim — dated 2026-09-06 by `mallet-vk-room-plan.md` §6.3.**
  Every number above is at **48 kHz**, where the `k²` in the coupling gives the mallet so much
  headroom that it never approaches the iteration wall, so the comparison is between two solvers
  doing easy work and Picard's cheaper sweep wins. Put the same mallet on the room's fixture at
  **8 kHz** and the ratio crosses one at `w/e ≈ 4.3` and settles at **0.71x**, with Picard's inner
  solves starting to fail at `w/e = 6.0` and the run dying at 7.8 while Newton reaches 11.6 with no
  failures at all. Sample rate is the axis this bullet did not vary. The parenthesis above was
  right and did not go far enough.
* `force_scale = M v0 / k`, and the outer residual is normalised by it rather than by `|f_j|` on
  purpose: the contact force passes through zero at the start and end of every contact, so a
  residual relative to the current force would demand the most absolute accuracy exactly where there
  is least force to be accurate about, and the first and last step of every contact would take the
  cap.
* Cost: 12 native bars in ~15 s (release), 22 Python tests in ~80 s.

## 11. Not attempted, and not blocked by anything here

* **A warm-started inner solve.** Seeding each trial's Picard loop from the previous outer iterate
  would cut inner sweeps, but it makes the inner exit point depend on the outer history, so `Psi`
  stops being a function of `f` and the outer loop chases a moving target. Cold-seeding is shipped
  for that reason and not for cost. The ceiling on what it could save is small anyway: at a mean of
  1.9 outer iterations, only the second trial could be warm.
* Two mallets on one gong (the influence columns superpose, so it is a 2×2 solve — but on a
  nonlinear plate the outer iteration becomes a 2-vector chord, which is a real design question,
  not a widening).
* Sub-grid strike interpolation (snap-to-node, the bow and mallet precedent).
* A viewer scene. Nothing here blocks one; `MalletVKPlate` exposes `state`, `displacement_at` and
  the full telemetry the viewer's other models use.
* ~~A gong **in a room** — `RoomLoadedVKPlate` and this model have never met. The airbox seam
  replaces the plate's `solve`, and this model calls `vk_step` directly, so they do not compose as
  written.~~ **Done 2026-09-06, `docs/dev/mallet-vk-room-plan.md`.** The diagnosis above was right
  about the mechanism and wrong about the remedy: the fix was not to make one call the other but to
  give `vk_plate_step` a **trial-solver closure**, so `vk_step` and the room's loaded step are two
  closures ending in the same `vk_step_with`. `MalletVKPlate` now takes either a `VKPlate` or a room
  wrapper. Two of this section's other bullets are dated by it — see §10's Newton note below.
