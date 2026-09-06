# The mallet on a gong in a room — the nested solve meets the loaded operator

Plan document for the follow-on batch named in `docs/dev/air-box-vk-newton-plan.md` §5 and, from
the other side, in `docs/dev/mallet-gong-plan.md` §11: *"a gong **in a room** — `RoomLoadedVKPlate`
and this model have never met."* Chosen by the human on 2026-09-06. Written **after probing and
before any code**, which is this project's ritual.

The probe scripts are in `M:\claud_projects\temp\mallet-vk-room\`. They drive the shipped models
through the binding and add no core code, so every number below is reproducible today.

---

## 1. What is actually blocked, and it is not the operator

The previous batch removed the operator obstacle. `physsynth_core::plate::ThetaSolve` lets a coupled
von Kármán step be pointed at a factorization assembled outside the crate, and `_VKPlateSurface`
already drives the model's own kernel against the room-**loaded** `A`. What remains in the way is
**ownership of the step**, and it is a genuine collision rather than a missing cast:

* **The mallet's outer chord re-solves the plate several times from the same time-`n` state** and
  commits exactly one of those trials. `physsynth_core::mallet::vk_plate_step` reaches the plate's
  four buffers directly and writes back once, for precisely this reason — its docstring says
  calling `step()` per trial would roll `u_prev` on the first one.
* **The room wrapper's `step()` is a once-per-step transaction.** It calls `port.require_ready()`,
  reads `free_pressure()` *before* the room advances, assembles the loaded right-hand side, solves,
  then computes the volume velocity `q`, calls `port.inject(q)` **once**, and adds `k · p̄ · q` to
  `radiated_energy`.

Driving the wrapper's `step()` once per outer trial would inject into the room `n_outer` times and
book the radiated energy `n_outer` times. Nothing on the plate side would see it: the plate's own
energy is unchanged by how many times the port was injected, and the scene total that *would* catch
it (`inst.energy() + room.energy()`) is exactly the number the miscount corrupts on both sides at
once. That is the failure this batch has to be built so as not to have.

### 1.1 The property that makes the composition possible

**The room's two load terms are invariant across the mallet's outer loop.** In `Wrap::step` the
right-hand side is

```
rhs = surface.rhs(f_ext) − k² (Tᵀ p̄_free) / denom + load_scale · (load_matrix @ w^{n−1})
```

and both room terms are functions of time-`n` and time-`n−1` state alone: `p̄_free` is the
open-circuit pressure read before `room.step()`, and the carry term is the `w^{n−1}` half of the
centred velocity whose `w^{n+1}` half is already inside the factorization. Neither depends on
`w^{n+1}`, and therefore neither depends on the trial force.

So the room half of the right-hand side is assembled **once per step**, and the only thing that
varies across the chord is `k² (−f) / denom` at the strike node. This is the premise the whole
design rests on and it is asserted, not assumed (§4, Part 3).

## 2. What the probe measured, and the three things it settles

### 2.1 The air load barely moves the DRIVE POINT and moves the rest of the column enormously

`probe_column.py`, on the shipped room fixture (0.3 m plate, `N = 8`, `fs = 8 kHz`, 1 mm steel,
`nnz_growth = 2.85`), comparing `k² A⁻¹ e_node / denom` factored bare against factored loaded:

| tier | boundary | node | `g_s` bare | `g_s` loaded | rel. diff at the node | worst rel. diff in the column |
|---|---|---|---|---|---|---|
| baffled | supported | 15 | 1.29790e-06 | 1.29753e-06 | **2.86e-04** | **7.71e-01** |
| baffled | free | 29 | 1.29794e-06 | 1.29757e-06 | 2.85e-04 | **4.14e+00** |
| suspended | supported | 15 | 1.29790e-06 | 1.29753e-06 | 2.86e-04 | 7.71e-01 |
| suspended | free | 29 | 1.29794e-06 | 1.29757e-06 | 2.85e-04 | 4.14e+00 |

Four decimal places apart at the node the mallet strikes, and **414% apart** at the far end of the
same column. That is not a paradox: near the drive point the plate's own local stiffness dominates
the response and the air is a small correction, while far from it the plate's response is small in
absolute terms and the load's share of it is large.

It matters because the shipped chord uses **only** `influence[node]`. `vk_drive_point_tangent` — the
*instrument* that measures the exact tangent, not a step — takes the whole column as its GMRES
right-hand side, and there 414% is not a rounding difference. So the column is one object with two
consumers and only one of them is nearly indifferent to which factorization built it.

### 2.2 The wrong column is a factor of about five in outer iterations, not a rounding error

Given §3.3's cancellation, the affine chord's contraction factor is `|g_s − g_s_true| / g`, which
`probe_fixture.py` evaluates directly on the room fixture (`nonlinear=False`, so `g_s_true` is the
loaded column exactly):

| mallet mass | column | `g_s` | `g_h` | `|dF/df|` | iterations to `outer_tol = 1e-13` |
|---|---|---|---|---|---|
| 20 g | loaded | 1.29753334e-06 | 7.8125e-07 | **0** | **1** |
| 20 g | bare | 1.29790390e-06 | 7.8125e-07 | 1.78e-04 | ~5 |
| 50 g | loaded | 1.29753334e-06 | 3.1250e-07 | **0** | **1** |
| 50 g | bare | 1.29790390e-06 | 3.1250e-07 | 2.30e-04 | ~5 |
| 200 g | loaded | 1.29753334e-06 | 7.8125e-08 | **0** | **1** |
| 200 g | bare | 1.29790390e-06 | 7.8125e-08 | 2.69e-04 | ~5 |

So a 2.9e-04 error in one scalar costs about five times the outer iterations, and therefore about
five times the plate solves, because the chord has to grind a 1e-04 contraction down to 1e-13. The
`n_outer == 1` bar of §3.3 is not merely tidier than a tolerance — it is worth a factor of five, and
it is the same factor whatever the mallet weighs. **This is the batch's argument for the loaded
column**, and it is a cost argument, exactly as §3.3 predicted it would have to be.

### 2.3 The room's fixture hosts a mallet, and the room is what sets the rate

`probe_fixture2.py`, a mallet on the room's **bare** plate (`make_air_vk_plate`, 8 kHz, `N = 8`,
`alpha = 2.3`, `K = 5e4`):

| mallet | `v0` | steps/contact | peak `w/e` | peak force | in-contact steps | max `n_outer` |
|---|---|---|---|---|---|---|
| 50 g | 3 m/s | 25.1 | 0.95 | 20.8 N | 256 / 800 | 5 |
| 50 g | 6 m/s | 25.1 | 1.89 | 54.1 N | 196 / 800 | 3 |
| 50 g | 12 m/s | 25.1 | 3.10 | 142.0 N | 150 / 800 | 4 |
| 200 g | 12 m/s | 50.3 | 4.48 | 381.6 N | 228 / 1200 | 4 |

The contact is resolved (25 steps, against the model's warning threshold of 8) and the strike
reaches `w/e = 4.5` — past where the *previous* batch's initial-condition strike blew Picard up on
step zero. That is worth checking rather than explaining away, because "the peak was reached" and
"the peak was reached by converged steps" are different claims. Measured over 1,200 steps
(`probe_inner.py`):

| mallet | `v0` | peak `w/e` | steps with a non-converged inner solve | worst inner `last_residual` | outer stalls |
|---|---|---|---|---|---|
| 50 g | 3 m/s | 0.95 | **0** | 9.31e-14 | 0 |
| 50 g | 12 m/s | 3.10 | **0** | 9.39e-14 | 0 |
| 200 g | 12 m/s | 4.48 | **0** | 9.84e-14 | 0 |

Every inner solve reaches `couple_tol = 1e-13`, so the amplitude is genuinely attained. The
explanation is the obvious one and it is now earned: a mallet builds its amplitude over hundreds of
steps while a displacement IC arrives with all of it at once, and `mallet-gong-plan.md` §10 already
recorded that the inner fixed point is not what limits this model at audio rates.

**A trap found while measuring that, which will mislead the next reader too.** `plate.n_iters` after
a mallet step is **not** one solve's sweep count — `PyMalletVKPlate::step` commits
`out.inner_iters`, which `vk_plate_step` sums over the force-free advance *and every chord trial*.
It reads 71 and 134 in the two hard rows above against a `couple_max_iter` of 50, and that is four
trials of thirty-odd sweeps, not an overrun. The quantity to compare against the cap does not
survive the step; the ones that do are `inner_converged` and `last_residual`, which is why the table
above is built from those.

The rate is the room's, not the gong's. `AIRBOX_SURFACE_FS = 8 kHz` against the shipped gong's
48 kHz, because the 3-D CFL runs the wrong way and the room sets the timestep for everything
attached to it. The mallet's chord works measurably harder there — `n_outer` up to 5 against the
shipped gong's mean of 1.9.

### 2.4 The air genuinely loads this plate, and the lighter plate is not the better fixture

The linear surface fixture uses `AIRBOX_SURFACE_RHO = 0.5 kg/m²`, chosen "light enough that the air
genuinely loads it"; the von Kármán one is 1 mm steel at **7.8 kg/m²**, fifteen times heavier, and
whether the room is decorative on it had never been measured. `probe_load.py`, 300 steps against the
bare `VKPlate` with nothing else different:

| tier | `rho_s` | `w/e` | trajectory departure (max, /amplitude) | radiated / `E0` |
|---|---|---|---|---|
| baffled | 7.80 | 1 | 2.78e-01 | 3.54e-02 |
| baffled | 7.80 | 3 | 6.11e-01 | 5.46e-03 |
| suspended | 7.80 | 1 | 5.32e-01 | 3.64e-03 |
| suspended | 7.80 | 3 | 1.21e+00 | 4.38e-02 |
| baffled | 0.78 | 1 | 5.82e+00 | 3.11e-01 |
| baffled | 0.78 | 3 | **both sides diverge** | — |

The shipped fixture is the right one. The air moves the trajectory by 28% to 121% of amplitude and
carries between 0.4% and 5.5% of the energy — a real load, not a decoration. Lightening the plate
tenfold loads it far harder (582%, 31% of the energy) and puts `w/e = 3` **past the Picard wall on
both sides**, room and bare alike, at 30,000 solves and a `NaN`. So the obvious "make it lighter so
the room matters more" move trades a measurable coupling for a fixture that cannot reach the
amplitudes the batch is about.

## 3. The design

### 3.1 The step is split, not copied

`Wrap::step` is decomposed into three phases that already exist inside it, in the order it already
runs them:

* **`prepare(f_ext) -> RoomStepHalf`** — `require_ready`, `free_pressure`, the tier's jump, the
  seam's `u_prev`, `surface.rhs(f_ext)`, and the two room terms. Returns the fixed right-hand side
  together with `p̄_free` and `w^{n−1}`, which the third phase needs.
* **the solve** — one or many, against `_lu_loaded`.
* **`finish(half, w^{n+1}, F^{n+1})`** — `surface.commit`, the centred velocity, `q = T v`,
  `p̄ = p̄_free + R q` (or `+ 2Rq` suspended), `port.inject(q)` **once**, the radiated ledger, and
  the four read-outs.

`Wrap::step` then *is* `prepare` → one solve → `finish`, and the shipped wrapper's arithmetic is
unchanged by construction rather than by inspection. This is deliberately not a second transcription
of the room's bookkeeping: `air-box-vk-newton-plan.md` deleted one such copy, and the finding it
left behind is that a copy stays invisible until something outside the file asks a question of it.

**Both stay Rust-internal.** Neither gets a `#[pymethods]` entry: a public seam is a surface the
suite starts reaching into, and this family's specific scar is a test that replaced a seam's
*methods* — a door no attribute grep finds. The split is a refactor, not an interface.

### 3.2 The mallet reaches the loaded operator through a trial solver

`core::mallet::vk_plate_step` currently hardcodes `plate::vk_step(u, u_prev, f_cache, f_prev, f_ext,
vk)` in two places — the force-free advance and each chord trial. Both become calls to a supplied
`trial: &dyn Fn(Option<&[f64]>) -> Result<plate::VkStep, SparseLuError>`:

* the **bare** gong passes a closure that builds `VkCoupledStep::new(...)` — which *is* `vk_step`,
  so the shipped model's arithmetic is untouched;
* the **room** composition passes a closure that builds `VkCoupledStep::with_rhs(rhs_fixed + the
  force term, u_prev, f_prev, vk, &LoadedLu)` and calls `vk_step_with`.

`LoadedLu` is the adapter `airbox_wrap.rs` already has. Nothing about Newton, the sweep loop, the
five diagnostics or the linear early return is duplicated: they are all inside `vk_step_with`, which
both closures end in.

**The refactor must be proved a no-op before anything is built on it** — `tests/test_mallet_gong.py`
green and unchanged, and the byte-exact miss test in particular. The previous batch's precedent is
exact: it measured the transcription as faithful *first*, and that is what made deleting it a
refactor rather than a re-derivation.

### 3.3 Which column the chord freezes — and why this is a rate question, not a physics one

`air-box-vk-newton-plan.md` §5 filed this as *"a real question with its own energy bars."* It is
half right: the question is real, but **energy cannot answer it, because the answer does not depend
on it.** `collision::solve_contact` solves

```
η = η_free − g f(η),        g = g_s + g_h
```

and `vk_plate_step` hands it `η_free = w_node(f) + g_s f − z_free`. At the fixed point the next
force equals `f`, so

```
η = w_node(f) − z_free − g_h f
```

and **`g_s` has cancelled**. The converged root — hence the committed field, the committed force,
and every energy bar in the family — is the same whichever column the chord froze. What `g_s`
changes is how fast the chord gets there.

The bar that *does* discriminate is exact rather than statistical. With `nonlinear=False` the plate
is affine in the contact force, so `w_node(f) = w_free,node − g_s^true f` where `g_s^true` is the
drive-point admittance of the operator actually inverted. Then

```
η_free = w_free,node + (g_s − g_s^true) f − z_free
```

and only when `g_s = g_s^true` does the second contact solve receive **bit-identical arguments** to
the first, exit with `outer_residual == 0.0`, and give `n_outer == 1`. That test already exists for
the bare gong (`tests/test_mallet_gong.py`, `n_outer == 1` under `nonlinear=False`); in a room it
holds only for the **loaded** column.

**The loaded column is the design**, justified by that bar and by §2.2's factor of five, with the
bare-versus-loaded difference reported on `n_outer` and `n_solves` — which is the honest axis, since
it is the only one that moves.

### 3.3a The column's OTHER consumer, which is wrong in a room in two ways at once

`vk_drive_point_tangent` — the instrument that measures the exact outer tangent `g_exact` by one
GMRES solve, and the thing `mallet-gong-plan.md` used to falsify "there is no closed-form
derivative" — reads `&p.influence` as its right-hand side **and** builds `plate::VkCoupledStep::new`
internally, hardcoded to the bare context. In a room both halves are wrong: the right-hand side is a
column that §2.1 measures as 414% off at its far end, and the Jacobian it inverts is the bare
plate's operator rather than the loaded one.

That is the previous batch's §3 hazard in a new place — a tangent quietly belonging to a different
problem than the residual — and it is the instrument any room-scene cost claim would be cited from.
So it is **routed**, not documented around: the tangent takes the same `&dyn ThetaSolve` the step
does, and the loaded influence column when there is one. It is cheap, because the tangent depends on
`theta` and not on `rhs_lin` at all: `averages` reads only the state, and `jacobian_vector` reads
only the operator.

### 3.5 The composition is a wider constructor, not a new class

`MalletVKPlate` accepts the room wrapper in the `plate=` slot rather than a second class being
added. Three reasons, none of them aesthetic: `RoomLoadedVKPlate` already delegates every read
through `__getattr__`; `StringVKPlateBridge` already takes the wrapper in exactly that slot, so the
precedent for "a wrapper stands where a plate stands" is set and tested; and `mallet-gong-plan.md`
says the constructor's shape exists **so that swapping a soundboard for a gong is a one-word edit**,
which a parallel class re-litigates.

The cast becomes three arms. A `PyVKPlate` takes the bare path, unchanged. An object carrying
`_lu_loaded`, `_surface` and `port` takes the room path, with `.plate` as the `PyVKPlate` underneath.
Anything else keeps the error message that ships today — it is load-bearing, because it tells a
caller holding a linear `Plate` to use `MalletPlate` instead.

### 3.4 Three things that will be got wrong if they are not written down first

* **The loaded column is computed once, and `_lu_loaded`'s staleness is guarded rather than
  out-run.** The first instinct here was to recompute the column every step, on the grounds that
  `_lu_loaded` has a setter and three tests replace the factorization wholesale
  (`inst._lu_loaded = splu(a)`). §2.2 kills that: `g_s_loaded` is the same nine digits at every
  mallet mass, and it is a function of `_lu_loaded`, `k`, `denom` and `node` alone — none of which
  move during a run. A per-step back-substitution would buy nothing and would land inside
  `n_solves`, which is the very number §2.2's claim is made of. Compute it at construction; make the
  `_lu_loaded` setter recompute it, so a swapped factorization cannot leave a stale column behind.
* **`plate.energy()` is the wrong number.** `Wrap::energy` is an explicit override because the
  delegated total omits the coupling channel. The scene total is `gong + radiated_energy +
  mallet KE + contact PE + room.energy()`.
* **The miss must stay byte-exact.** `vk_plate_step` short-circuits a zero contact force to the
  force-free advance and returns it *unmodified* — a property of the control flow, not of floating
  point. The force-free trial must therefore go through the identical path the wrapper uses today
  (`prepare(None)`, no `+0.0` added anywhere), so a mallet that never lands leaves a trajectory
  bit-identical to `RoomLoadedVKPlate` stepped alone. For in-contact trials there is no reference to
  be exact against, so the operand order there is a consistency choice — but it is chosen
  deliberately: the force goes into the seam's `rhs` **before** the room terms, exactly where
  `surface.rhs(f_ext)` puts it, so that `f_ext = 0` and `f_ext = None` differ only by the `+0.0`
  the miss path avoids.

## 4. The parts

**Part 1 — the trial solver in the core.** `vk_plate_step` takes a trial closure; `vk_step` becomes
the bare gong's closure. `vk_drive_point_tangent` takes the operator and the column too (§3.3a).
Native bar: the shipped trajectory is reproduced bit for bit. Gate: `tests/test_mallet_gong.py`
unchanged and green (§3.2).

**Part 2 — the room step, split.** `prepare` / `finish` in `airbox_wrap.rs`; `Wrap::step` rewritten
as the two around one solve. Gate: `tests/test_airbox_vk.py` and the rest of the airbox family
unchanged and green, the byte-exact `nonlinear=False` regression against `RoomLoadedPlate` first
among them.

**Part 3 — the composition.** `MalletVKPlate`'s constructor widened to take the room wrapper
(§3.5); its step drives `prepare` once, the chord against the loaded operator, and `finish` once.
The loaded column, computed at construction and refreshed by the `_lu_loaded` setter. Bars:

* the miss is byte-exact against `RoomLoadedVKPlate` alone;
* `nonlinear=False` gives `n_outer == 1` and `outer_residual == 0.0` with the loaded column, and
  more than one with the bare column — the discriminating bar of §3.3;
* the room's two load terms are the same array on every outer trial (§1.1's premise, asserted);
* `port.inject` is called **exactly once** per step whatever `n_outer` is, and `radiated_energy`
  advances by one step's worth;
* **the room is driven by the ACCEPTED iterate, not by a trial** — and this is the one error the
  inject-once bar above cannot see. `q` is built from `(w^{n+1} − w^{n−1}) / 2k`; hand `finish` a
  trial field instead of the committed one and the injection count is still 1, the ledger still
  advances by a plausible amount, and the room simply receives the wrong volume velocity. The bar is
  a recomputation: on a run where `n_outer > 1`, `nodal_volume_velocity` must equal
  `T @ (u_committed − u_prev_captured) / 2k` rebuilt in the test from the plate's own buffers after
  the step;
* the scene total is conserved on a lossless rig with a rigid room.

**Part 4 — the measurement.** Does the air load move the mallet's wall, as §2.2 of the previous plan
found it does not move the plate's? What does the composed scene cost in back-substitutions against
the bare gong? Picard versus Newton on it.

**No default moves.** `couple_method` stays `"picard"`, and every shipped number stays the number
it is.

## 5. Deliberately not done here

* **No Rust assembly or factorization of the loaded matrix** — design C of the previous plan, still
  a real batch and still not this one.
* **No viewer scene.** Nothing here blocks one.
* **No mallet on a room-loaded *linear* plate.** `MalletPlate`'s influence column is exact for an
  affine step and would need the loaded one for the same reason; it is a smaller version of this
  batch and is not folded into it.
* **No two-mallet widening**, and no sub-grid strike interpolation — both already listed as
  not-blocked in `mallet-gong-plan.md` §11.
* **`residual_ratio` is still `NaN` after a mallet step**, in a room exactly as on the bare gong.
  `PyVKPlate::commit` writes it that way because a chord runs *many* inner solves and there is no
  one ratio to report; the seam's `record_iteration` writes the real one because it runs one. So
  `couple_outcome()` on a room-mallet step is subject to the same misreport
  `air-box-vk-newton-plan.md` §2.4 fixed for the seam — it will file a non-converged step as
  `expansive` whether or not a larger cap would fix it. It is pre-existing rather than introduced
  here, and closing it means first deciding *which* of a chord's solves the ratio describes.

---

## 6. What was built, and what it measured

Landed 2026-09-06 in three parts, each gated before the next as §4 says.

**Part 1** put a trial-solver closure into `physsynth_core::mallet::vk_plate_step`; `vk_plate_step`
is now that function closed over `plate::vk_step`, so the bare gong's arithmetic is the room's by
construction. `vk_drive_point_tangent` gained the same two overrides (§3.3a). Gate: the native
suite green (28 binaries, 0 failures) and `tests/test_mallet_gong.py` unchanged.

**Part 2** split `Wrap::step` into `prepare` / `loaded_rhs` / `finish`, all `pub(crate)`. The two
room load terms are carried as separate vectors rather than as their sum, so a client re-assembling
the right-hand side once per trial gets the doubles `Wrap::step` gets assembling it once — `base +
(−a + b)` and `(base − a) + b` are not the same double, and the miss path's byte-exactness is a
claim about that expression. Gate: 253 tests across `test_mallet_gong.py`, `test_airbox_vk.py`,
`test_airbox_surface.py` and `test_airbox_membrane.py`, unchanged and green.

**Part 3** widened `MalletVKPlate`'s constructor to the three arms of §3.5 and added
`step_in_room`: one `prepare`, a chord of solves against the loaded factorization, one commit, one
`finish`. 20 tests in `tests/test_mallet_room_gong.py`.

### 6.1 The air load does not move the mallet's wall either — and the room is very nearly free

600 steps, 50 g on `K = 5e4`, `alpha = 2.3`, the shipped baffled fixture against the identical bare
`VKPlate`:

| `v0` | scene | peak `w/e` | max `n_outer` | `n_solves` | non-converged inner | scene drift |
|---|---|---|---|---|---|---|
| 3 | room | 0.9459 | 3 | 11,426 | 0 | 1.27e-12 |
| 3 | bare | 0.9533 | 5 | 11,454 | 0 | 1.18e-12 |
| 12 | room | 3.0919 | 4 | 18,032 | 0 | 1.13e-12 |
| 12 | bare | 3.1028 | 4 | 18,014 | 0 | 1.06e-12 |
| 30 | room | 5.4964 | 5 | 33,050 | 0 | 9.81e-13 |
| 30 | bare | 5.5063 | 5 | 32,976 | 0 | 1.09e-12 |

`n_solves` agrees within 0.5% at every amplitude, and where Picard eventually dies it dies at the
**same strike velocity** on both sides with the same number of non-converged steps (14 at
`v0 = 50`, 8 at 80, 7 at 120). This is `air-box-vk-newton-plan.md` §2.2's finding again, from a
different direction: the previous batch found the air load does not move the *plate's* iteration
wall, and it does not move the *mallet's* either. Attaching a room to a gong costs the chord
nothing.

The peak deflection is consistently about 1% *lower* in the room, which is the air taking energy
and is the only place the load shows up in this table at all.

### 6.2 The mallet's wall is far above the initial-condition strike's — 7.8e against 4.5e

`w/e = 5.5` runs with zero non-converged inner solves on the same `N = 8`, 8 kHz grid where
§2.3's displacement strike blew Picard up on step zero at `4.5e`. Picard's first non-converged
step arrives at `v0 = 35` (`w/e = 6.0`, 18 steps of 400) and the run dies at `v0 = 50`. A mallet
builds its amplitude over hundreds of steps; a displacement IC arrives with all of it at once, and
the difference between the two is worth more than a factor of one and a half in amplitude.

### 6.3 Newton is CHEAPER here below the wall, which contradicts the gong plan — for a good reason

`mallet-gong-plan.md` §10 records that Newton "buys nothing here", costing 1.47x to 1.72x the
back-substitutions, measured across three mallets at 48 kHz. On the room's fixture the ratio
crosses one and keeps going:

| `v0` | peak `w/e` | Picard `n_solves` | its non-converged steps | Newton `n_solves` | Newton / Picard |
|---|---|---|---|---|---|
| 3 | 0.946 | 9,886 | 0 | 14,066 | 1.42 |
| 6 | 1.885 | 11,468 | 0 | 15,280 | 1.33 |
| 12 | 3.092 | 15,466 | 0 | 18,878 | 1.22 |
| 20 | 4.282 | 19,834 | 0 | 20,804 | **1.05** |
| 30 | 5.496 | 28,690 | 0 | 21,930 | **0.76** |
| 35 | 6.022 | 33,636 | **18** | 23,990 | 0.71 |
| 45 | 6.943 | 39,986 | **34** | 28,502 | 0.71 |
| 50 | 7.79 | **died** | 14 | 29,804 | — |

The crossover is at `w/e ≈ 4.3`, and past it Newton settles at about 0.71x while Picard's cost
climbs and its inner solves start failing. Newton has **zero** non-converged steps at every row,
and reaches `w/e = 11.6` at `v0 = 120` where Picard is long dead.

Both iterations land on the same root wherever both converge — the peak `w/e` agrees to four or
five figures on every row, which is the check that makes the cost columns comparable at all.

This does not overturn §10; it dates it. §10's measurement is at 48 kHz, where the `k²` in the
coupling buys so much headroom that the mallet never approaches the wall, so the comparison was
always between two iterations doing easy work — and there Picard's cheaper sweep wins. The room
sets the sample rate at 8 kHz, the mallet *does* reach the wall, and the ranking inverts. It is the
same lesson §2.3 of the previous plan learned about its own payoff figure: **a margin measured at
one fixture is a claim about one fixture**, and sample rate is one of the axes it moves along.

**The default does not move.** `couple_method` stays `"picard"` everywhere; what this section
records is where a caller should reach for the other one.
