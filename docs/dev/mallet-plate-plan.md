# Mallet–plate collision — model #7p, the strike on an *implicit* resonator

> **Status: BUILT & GREEN (2026-09-06).** `MalletPlate` in `crates/physsynth-core/src/mallet.rs`,
> bound in `crates/physsynth-py/src/mallet.rs`, re-exported by `physsynth/core/mallet.py`.
> Tests: `crates/physsynth-core/tests/mallet.rs` (+9 native bars, 21 in the file),
> `tests/test_mallet_plate.py` (32), `tests/test_mallet_plate_signature.py` (5).
> Helper: `make_mallet_plate` in `tests/helpers.py`.
>
> Closes the gap HANDOFF §14.1 named and left open: *"there is no mallet-on-a-plate composition to
> reach for — `MalletMembrane` casts its collaborator to a `Membrane`, so that is a model batch
> someone has to build."* The hammer plan predicted the route four batches before anyone took it:
> *"Plate/bar follow-on reuses the same scalar collapse with `g_s` from the plate's driving-point
> admittance instead of the local nodal mass."* That prediction was right, and this document is
> mostly about the two things it did not say.
>
> **It is not the gong.** `MalletPlate` takes a linear `Plate` and refuses a `VKPlate`, with the
> reason in the `TypeError`. See §6.

---

## 1. What is actually new, and it is one vector

Model #7 struck a membrane, and the membrane made the coupling free. The membrane is **explicit**:

```
u^{n+1} = (2 u^n - (1 - sigma k) u^{n-1} + c^2 k^2 L u^n) / (1 + sigma k)
```

so a nodal force at the struck node reaches **only that node** by the next step — the Laplacian
coupling is already frozen into `u^n`. The driving-point admittance is therefore the bare local
nodal mass, `g_s = k^2 / (rho h^2 (1 + sigma k))`, a scalar with a closed form, and the force is
injected by writing one array entry.

The plate is **implicit**. Its step solves

```
A u^{n+1} = rhs ,      A = (1 + sigma k) W + theta k^2 kappa^2 K
```

(`W = I`, `K = B` on the supported branch), so a force at one node reaches *every* node next step,
and there is no local admittance to read off. What replaces it is the **influence column**

```
influence = (k^2 / force_den) A^-1 e_node ,        g_s = influence[node]
```

`force_den` being `rho h^2` supported and `rho` free — the same quantity `plate::step_rhs` divides
`f_ext` by, and now exposed as `plate::Params::force_denominator()` so there is exactly one spelling
of it. One back-substitution at construction, reused every step: `g_s` closes the scalar contact
equation, and the whole column spreads the force back into the field.

This is the bow's manoeuvre against the implicit stiff string (`a = A^-1 e_i`, one banded solve
baked into a vector), applied to the mallet's contact solve. **Everything else in the model is
model #7's and already proven**: the contact root-find, the discrete-gradient force, its removable
`0/0` Taylor branch, the `State`, the five refusals, the under-resolved-contact warning. That is why
the first test in `tests/test_mallet_plate.py` is not an energy bar.

## 2. The superposition identity, and why it is a whole-field claim

`Plate::step` is affine in `f_ext`, so stepping force-free and adding `influence * f` afterwards is
*algebraically identical* to putting `f` in the right-hand side before the solve:

```
A (u_free + d) = rhs_0 + A d = rhs_0 + (k^2 f / force_den) e_node
```

The plate's own `step_rhs` comment says a post-solve correction is invalid because `A` couples all
nodes. That is right about a **one-node** correction — which is exactly what the membrane does.
Correcting along the whole column is the same thing done properly.

The guard is asserted **at every node**, natively and from Python, on both boundary branches. A
column that is right at the strike node and wrong elsewhere would satisfy a drive-point check, pass
every energy bar (a wrong field is simply a different self-consistent trajectory), and sound wrong.
There is also a specific way to get it wrong here, §5.

**It is not bitwise, and cannot be.** A sparse LU back-substitution is not a linear map over
doubles: `solve(rhs_0) + c*solve(e)` and `solve(rhs_0 + c*e)` round differently. Measured, the two
fields agree to better than `1e-14` relative. The one case where superposition *is* exact is a
**miss**: `f == 0.0` makes every increment a signed zero and `x - (+-0.0) == x` for finite `x`, so a
mallet that never touches the plate leaves a trajectory bit-identical to the bare plate's — which is
what `test_missing_mallet_is_bit_identical_to_bare_plate` asserts with `array_equal`.

## 3. The energy identity carries over untouched

Dotting the theta-scheme with `u^{n+1} - u^{n-1}` gives the same discrete power a force does on the
membrane, `f . delta_t. u`. So the discrete-gradient telescoping is unchanged and

```
H^n = E_plate^n + 1/2 M (delta_t- z_H)^2 + 1/2 (phi(eta^n) + phi(eta^{n-1}))
```

is constant at `sigma = 0, lam_h = 0` and monotone decreasing above either. Measured on the
supported branch over 4000 steps, across four `(K, M, v0, alpha)` combinations: **3.0e-13 to
6.3e-13** relative, against the project's `1e-10` contract, with the plate taking 51–99% of the
strike energy. Drift-proportional-to-`newton_tol` self-certification holds, so the conservation is
the discrete gradient's and not the solver's luck.

Two consequences of the plate being implicit, both in the model's favour:

* **the strike imposes no CFL.** The theta-scheme is unconditionally stable for `theta >= 1/4`, so
  unlike the membrane there is no Courant number the collision has to stay under. What still has to
  be resolved is the **felt**, whose half-period is `pi sqrt(M/K)` — and the plate's sample rate
  comes out of *its* Courant number (`fs = kappa / (mu h^2)`), which knows nothing about the felt.
  `make_mallet_plate` therefore defaults to `mu = 1.0` rather than the plate suite's 2.0 (23 steps
  through the contact at `N = 24`), and `test_the_felt_is_resolved_on_the_shipped_defaults` asks the
  question of the defaults directly rather than trusting a comment.
* **the outline is free.** The influence column is built from whatever `A` the plate factored, so a
  circle or a guitar outline changes only which nodes are live. Both are tested.

## 4. The finding: a struck FREE plate's energy read-out is not the scheme's

Unplanned, and the batch's real result.

A point strike feeds the free plate's `{1, x, y}` rigid nullspace: the plate takes net momentum and
translates for ever after at constant velocity. That is physics (an unsupported cymbal recoils), it
is exact — projecting the step onto `1^T W` kills the stiffness term because `K 1 = 0`, leaving
`m^{n+1} = 2 m^n - m^{n-1}`, so the mass-weighted mean is a **linear function of the step index** to
`1e-9` — and it is asserted as such.

But the energy read-out cannot survive it. The potential form is `P(f,g) = kappa^2 (K f).g`, in
which the rigid part cancels *mathematically*; numerically it cancels only to `eps`, and because `P`
is a **quadratic** form the leftover absolute error goes like the **square** of the rigid
displacement. Measured on a **bare** free plate — no mallet anywhere — handed a uniform velocity:

| rigid velocity | mean displacement after 2000 steps | absolute energy error | error / drift² |
|---|---|---|---|
| 0.0 | 6.3e-05 | 6.4e-16 J | — |
| 0.1 | 1.74e-02 | 8.8e-14 J | 2.90e-10 |
| 0.3 | 5.21e-02 | 8.0e-13 J | 2.96e-10 |
| 1.0 | 1.74e-01 | 9.3e-12 J | 3.09e-10 |
| 3.0 | 5.21e-01 | 8.2e-11 J | 3.01e-10 |

Constant to three digits across a thirtyfold range of drift. With **no** net momentum the read-out
is exact to `6.4e-16 J`.

Three things follow, and they are the ones a later batch needs:

1. **It belongs to `Plate.energy()` on the free branch, not to the mallet.** The mallet is simply
   the first thing in this project that ever gave a free plate net momentum — every existing
   free-plate test starts from a displacement with zero mean (`plate_bump` subtracts the mean on the
   free branch *on purpose*, for a different reason: a net piston is the most efficient radiator the
   geometry has). `test_the_free_readout_error_is_quadratic_in_the_rigid_drift` runs with no mallet
   in the room, which is what makes it an attribution rather than an excuse.
2. **The free branch's bar is therefore a read-out bar, not an energy bar.** `1e-8` over 2000 steps
   (measured 9.8e-10), with the supported branch run on the identical mallet as a control and
   required to be orders better — a control that would go red if the excess were ever something
   other than the rigid mode.
3. **No amount of rig tuning fixes it, and the reason is worth writing down.** The relative drift is
   `plate_share x (cancellation error relative to the plate's own energy)`. Making the plate heavier
   shrinks the recoil, but it shrinks the plate's share of the strike in the same proportion — at
   `rho = 2.0 kg/m^2` the drift falls to 5.6e-10 and the plate takes **6%** of the strike, i.e. the
   mallet is bouncing off a wall. Conservation to `1e-10` over a long window and a free plate that
   actually takes the strike are not simultaneously available through this read-out.

## 5. Traps

* **Live versus full-grid indexing.** `plate::pickup_index_at` returns a **live** index (it counts
  live nodes and skips dead ones) while `Params::x` and `y` are **full-grid** arrays. A confusion
  between the two reports a plausible-looking strike point and builds a *wrong influence column*,
  which only the whole-field guard of §2 would catch. `live_coords` therefore walks the mask with
  the same traversal `pickup_index_at` uses, which makes the mixup unrepresentable rather than
  merely tested for — and the Python guard asserts the reported point against the plate's own answer
  for that live index.
* **`_accel` has to be corrected too.** The plate defines it as the actual second difference
  `(u^{n+1} - 2u^n + u^{n-1}) / k^2`, and only the first term moved. Leaving it stale makes
  `pressure()` — the radiated read-out — report the *unstruck* plate: a silent wrong answer in
  exactly the observable a listener cares about. Its guard needs its own normalisation, because a
  displacement error `e` arrives in the acceleration as `e / k^2` while the acceleration's own scale
  is `(omega k)^2` smaller than the displacement's; normalising by `max|accel|` would demand three
  orders more agreement than the displacements can offer. The claim asserted is the honest one:
  multiplied back by `k^2`, the two accelerations came from displacements agreeing to `1e-14`.
* **Two spellings of `k^2`, deliberately.** The influence scale uses `k * k` because it has to be
  the number `step_rhs` divides `f_ext` by, and that spells its `k^2` as a multiply; `g_h = k^2 / M`
  keeps `scalar_pow` because it is the *mallet's* own constant and must equal what
  `MalletMembrane`/`MalletWall` compute for the same `M` and `k`. §16.2 of the migration plan is why
  those are two different doubles.
* **The column is a snapshot of the factorization**, taken at construction. So is the plate's own
  `lu`, and `Plate::set_B` deliberately does not rebuild it, so the two go stale together and stay
  consistent with each other. Assigning a different operator to a plate a mallet is already holding
  gives an inconsistent plate exactly as it does without one.

## 6. What is deferred, and why it is a different algorithm

**The gong.** `MalletPlate` refuses a `VKPlate`, and the refusal message says why, because the
obvious reading is that someone forgot a branch. The von Kármán step is a **nonlinear** solve
(Picard or Newton), so it is not affine in `f_ext` and the influence column *does not exist*.
Superposition is the entire basis of the scalar collapse, so a mallet on a gong is not this model
with a different collaborator — it needs an **outer** contact solve wrapped around a full plate
solve per residual evaluation, at roughly ten to a hundred plate solves per timestep, with no
closed-form derivative for the outer Newton (a secant or a bracket instead). That is a batch.

It is also the batch HANDOFF §14.1 actually wants: *"a batch wanting the gong impression still needs
a mallet, not a budget."* Half of that is now on the shelf. The other half is the nested solve, and
`StringVKPlateBridge` is the precedent for what *doesn't* work — its spring force `F = K eta^n` is
**sweep-invariant** and enters the right-hand side outside the Picard loop, which is exactly the
property the mallet's discrete-gradient force does not have (it is implicit in `eta^{n+1}`, and that
implicitness is what makes it conserve).

**Not attempted, and not blocked by anything here:** more than one simultaneous mallet on one plate
(a gap in the *exciter* layer, as HANDOFF §11.3a says of the membrane — note the influence columns
of two mallets superpose, so the extension is a 2x2 solve rather than a redesign); sub-grid strike
interpolation (snap-to-node, the bow and mallet precedent); a viewer scene.

## 7. Numbers a later batch will want

* Supported, `N = 24`, `mu = 1.0`, `Lx = Ly = 1`: `fs = 11520 Hz`, 529 live nodes,
  `steps_per_contact = 22.9`, `g_s = 2.45e-04`, `g_h = 3.77e-07`.
* The free branch's `g_s` is **identical** to the supported branch's at an interior drive point
  (2.4535e-04 both), because `W_ii = h^2` in the interior and `force_den` differs by exactly that
  `h^2`. Same coincidence `docs/dev/string-vk-plate-bridge-plan.md` §10.6 records for the bridge
  guard's margin, and for the same reason.
* Supported conservation over 4000 steps: 3.0e-13 / 4.6e-13 / 6.3e-13 / 3.1e-13 across the four
  parametrisations. Plate share of the total: 0.73 / 0.99 / 0.51 / 0.95.
* Free strike, `N = 24`: drift 3.7e-11 / 2.0e-10 / 9.8e-10 / 3.8e-09 / 1.8e-08 at 500 / 1000 / 2000
  / 4000 / 8000 steps — a ratio near 4 per doubling, which is the `t^2` law of §4.
* Modal comb of an off-centre strike, relative to (1,1): 1.00, 0.195, 0.083, 0.037, 0.018, 0.028,
  0.004 for (1,1) (2,1) (1,2) (2,2) (3,1) (1,3) (3,3). It falls away much faster than a drumhead's
  because a plate's frequencies go like `m^2 + n^2` where a membrane's go like `sqrt(m^2 + n^2)`, and
  an impulse hands each mode a *velocity* — displacement is that over omega. Nearly unchanged by
  felt hardness (1.00, 0.216, 0.092 at `K = 2e5`): at this contact length every one of these modes
  is inside the pulse's band, so the comb is the modal transfer function's shape, not the pulse's.
* **The felt exponent is the only source of dynamic timbre**, and it is graded. Departure from an
  exactly scaled response between a 3.0 m/s and a 0.75 m/s strike: **7.6e-13** at `alpha = 1`, then
  0.19 / 0.62 / 0.71 at `alpha` = 1.5 / 2.3 / 3.0. At `alpha = 1` the whole system is linear — plate
  linear, mallet a mass, and the one-sided switching scale-invariant — so the response is exactly
  proportional to how hard you hit it. Everything a player would call *dynamics* lives in one
  exponent.
* **What moves the two bars, measured, because one sweep confounds them.** `mu` is the plate's own
  Courant number and it sets the conditioning of `A` by itself — `cond(A) ~ 1 + 64 theta mu^2`,
  independent of `N`. Holding `fs` fixed needs `N^2 ~ mu`, so a `mu` sweep at fixed `fs` moves
  conditioning **and** node count together. Separated:
  * the **superposition identity** does not care about either: 4.0e-16 → 1.8e-15 across a 256x
    conditioning range (256 to 4489 live nodes). It is not near its `1e-14` bar anywhere.
  * the **supported energy bar** does not care about conditioning — holding `N` and scaling `kappa`
    with `mu` to keep `fs`, the drift is 1.6e-13 to 2.3e-12 across 256x, non-monotone. It cares
    about the **node count**, because `energy()` is a reduction and a longer sum accumulates more
    rounding: 1.9e-13 at 256 nodes, 3.5e-13 at 529, 4.7e-12 at 2209, 1.7e-11 at 4489. Six times
    inside `1e-10` at the largest rig tried, and asserted at 2209 nodes so the bar is not fitted to
    one size.
  * the **free branch's bar is not a constant of the model**. It bounds the rigid-to-elastic
    displacement ratio, so anything that shrinks the elastic response at fixed strike energy makes
    it worse: 9.8e-10 on the shipped rig, 1.2e-08 at `mu = 4, N = 48`, 4.5e-08 at `kappa = 160`.
    The transferable claim is the **ratio between the branches on the same mallet**, which is what
    the test asserts alongside the number.
* **The recoverable alternative, not taken, and it is the human's call.** `K Pi u = 0` exactly for
  the rigid projector `Pi`, so evaluating the *potential* term on `u - Pi u` (the kinetic term keeps
  the full velocity — rigid kinetic energy is real energy) is mathematically identical and has no
  cancellation to lose. Machine precision on the free branch is therefore recoverable. It is not
  done here because it changes `Plate.energy()` for every caller, and `tests/test_vk_energy.py` and
  the airbox parity files hold **bit-identity** anchors on that number.
* Cost: 36 tests in ~9 s; ~1.3 s per 4000-step run at 529 live nodes. Native: 21 bars in 2.3 s,
  green in both the debug and the release profile (`mallet.rs` is the file where that distinction
  has bitten before — a literal exponent is folded into a multiply in release).
