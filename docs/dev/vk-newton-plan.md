# Newton for the von Kármán plate — the nonlinear plate's iteration wall

Plan document for `docs/dev/scientific-hurdles.md` §5, chosen by the human on 2026-09-06 over §4
(the θ-scheme's rate suppression). Written **after probing and before any code**, which is this
project's ritual, and the probing changed the batch: two of the three things §5 says are wrong.

The probe scripts are in `M:\claud_projects\temp\vk-newton\` (`probe_picard.py`,
`probe_driver.py`, `probe_contraction.py`, `probe_cap.py`, `probe_rho.py`). They drive the shipped
Rust model through the binding and add no core code, so every number below is reproducible today.

---

## 1. What the iteration actually is

`vk_step` (`crates/physsynth-core/src/plate.rs:1703`) solves the conservative coupled step by
fixed-point sweeps. Written as a map on the live-node vector `w`, with `P` the rim-scatter
`to_full`, `A` the θ-scheme system matrix (prefactored, `lin.lu`), `l(·,·)` the Monge–Ampère
bracket (bilinear and symmetric, `ops2d::VonKarmanBracket::eval`) and the clamped Airy solve
`F(W) = Ainv(−½ Y l(W, W))`:

```text
w_{j+1} = A^-1 ( rhs_lin + c * to_live( l( ½(P w_j + W_prev), ½(F(P w_j) + F_prev) ) ) )
```

with `c = k²/ρ_s` (times `h²` on a free edge) and `rhs_lin` fixed across the sweeps — the external
force is added once outside the loop because a bridge spring's `K η^n` depends only on time-`n`
state.

Two facts follow immediately and neither is in §5:

* **The map is a Richardson iteration on a linear system.** Its residual is
  `G(w) = w − A^-1(rhs_lin + c*coupling(w))`, and Picard is exactly Newton with the Jacobian
  approximated by the identity. So Picard's convergence is governed by the spectral radius `ρ` of
  `c·A^-1 K`, where `K` is the linearisation of the coupling — it contracts iff `ρ < 1`.
* **The Newton system is already preconditioned.** `J = I − c·A^-1 K` carries `A^-1` inside it, so
  at zero amplitude `J = I` **exactly**. §5 proposes "GMRES preconditioned by the linear plate's
  existing `splu`"; that preconditioner is already present by construction, and the batch needs no
  second factorization.

## 2. What the probe measured, and the three claims it overturns

### 2.1 Where Picard stands today (40 cm steel square, 1 mm, N=20, 48 kHz, 300 steps)

| w/e | max sweeps | mean sweeps | steps over cap | energy drift |
|---|---|---|---|---|
| 1 | 5 | 3.9 | 0 | 1.8e-13 |
| 3 | 7 | 4.9 | 0 | 1.0e-13 |
| 6 | 11 | 6.9 | 0 | 1.1e-13 |
| 9 | 20 | 9.9 | 0 | 2.1e-13 |
| 12 | 45 | 15.3 | 0 | 3.4e-13 |
| 16 | 50 (cap) | 27.3 | **29** | **2.2e-06** |
| 20 | 50 (cap) | — | 1 | **overflow** |

### 2.2 Correction 1 — the difficulty does **not** scale like `1/h⁴`

§5 says the contraction factor "scales like `k² · (amplitude/thickness)² / h⁴`", and
`tests/helpers.py:729-733` repeats it. The `k²` half is right; the `h` half is measured wrong.

Refining the grid **4.3×** at fixed plate size *and fixed absolute strike width* costs two sweeps
and then flattens:

| N | h | max sweeps | drift |
|---|---|---|---|
| 12 | 0.0333 | 12 | 3.1e-13 |
| 20 | 0.0200 | 20 | 2.0e-13 |
| 28 | 0.0143 | 20 | 2.0e-13 |
| 40 | 0.0100 | 22 | 3.4e-13 |
| 52 | 0.0077 | 22 | 1.5e-12 |

Whereas shrinking the plate at **fixed** `h = 0.02` (N scaled with the side) hits the cap:

| side | N | max sweeps | drift |
|---|---|---|---|
| 0.40 | 20 | 11 | 1.1e-13 |
| 0.32 | 16 | 14 | 8.3e-14 |
| 0.24 | 12 | 23 | 2.7e-13 |
| 0.16 | 8 | 50 (cap) | 2.7e-09 |

— and so does narrowing the *strike* alone, at fixed plate, fixed grid and fixed peak amplitude
(7 → 11 → 22 → 50 sweeps for a 12 / 8 / 5 / 3 cm Gaussian). The driver is the **strain**, i.e. the
curvature of the deflection, not the grid spacing. This is good news for the batch and bad news for
the doc: grid refinement is nearly free, and the two observations §5 files under "geometry"
(`k²/h⁴`) are really one observation about curvature. **The observations stand; the attributed
mechanism is wrong, and both documents are corrected in this batch.**

### 2.3 Correction 2 — a third of the wall is the 50-sweep cap, not divergence

`ρ` is a clean constant in every *converging* fixture (0.016 / 0.251 / 0.675 / 0.820), so those
runs converge at any amplitude given sweeps. Re-running the failures with a generous cap:

| fixture | cap 50 | cap 400+ | wall clock |
|---|---|---|---|
| 40 cm, 3 cm strike, w=6e | 3 steps over cap, drift 1.7e-05 | **143 sweeps, drift 4.1e-13** | 0.36 s → 0.32 s |
| 16 cm, 3.2 cm strike, w=6e | 2 over cap, drift 2.7e-09 | **76 sweeps, drift 6.0e-13** | unchanged |
| 40 cm, 8 cm strike, w=16e | 29 over cap, drift 2.2e-06 | **724 sweeps, drift 4.3e-13** | 0.67 s → 0.82 s |
| 12 cm, 2.4 cm strike, w=6e | overflow | overflow at cap 3000 | — |
| 40 cm, 8 cm strike, w=20e | overflow | overflow at cap 3000 | — |
| 40 cm, 8 cm, w=12e, 24 kHz | overflow | overflow at cap 3000 | — |

Three of six come back **green on the energy bar at essentially no wall-clock cost**, because the
expensive steps are rare — the mean sweep count barely moves (14.4 → 14.7, 27.3 → 34.3) while the
max goes to 143 and 724. Only three are genuine divergence.

**This is the batch's structural hazard, and it is handled by ordering rather than by a warning:**
measuring Newton against a cap-50 Picard would credit Newton with territory that a *constant*
recovers. Part 0 lands first and re-draws the baseline; Newton's claim is then only the `ρ > 1`
region, and it is honest by construction.

### 2.4 Correction 3 — in the divergent cases `ρ` is not constant, it climbs through 1

Reading the first eight residuals of one solve (a run capped at `m` sweeps truncates the first step
at exactly `m`, and every cap starts from the same seeded state — so this recovers a single solve's
history with no new code):

| fixture | ρ across sweeps 1→8 | verdict |
|---|---|---|
| 40 cm, 3 cm, w=6e | 0.691 → 0.821 | contracts, slow |
| 40 cm, 8 cm, w=16e | 0.160 → 0.675 | contracts, slow |
| 40 cm, 8 cm, w=20e | 0.243 → **1.011** | stalls at 7.8e-4, then diverges |
| 40 cm, 8 cm, w=12e, 24 kHz | 0.334 → **1.114** | stalls at 4.2e-3, then diverges |
| 12 cm, 2.4 cm, w=6e | ≈1.0 from sweep 1, NaN by 7 | no useful iterate at all |
| **7 cm**, 1.4 cm, w=6e | ≈1.0 from sweep 1, NaN by 5 | no useful iterate at all |
| **7 cm**, 1.4 cm, w=2e | 0.6–1.5, wanders, → 1.0 | no useful iterate at all |
| **7 cm**, 1.4 cm, w=e | 0.241 → **0.183** | **converges in 8 sweeps** |

Two things follow. First, the recoverable failures are the *stalling* ones — the residual reaches
1e-3 or 1e-4 and then the map turns expansive; a Newton step from an iterate that good is exactly
what quadratic convergence is for. Second, **the 7 cm audio-band plate is not categorically
broken** — it converges at `w = e` and fails at `w = 2e`. So the ceiling is an amplitude, and the
batch's job is to measure how far it moves, not to claim the plate.

## 3. The claim, and what it is deliberately not

**The claim.** Replacing the fixed-point sweep with Newton on the same residual moves the
amplitude/strain ceiling of the von Kármán step, and the batch ships a **convergence map** over
`(w/e, strike curvature, fs)` drawn twice — once for best-effort Picard, once for Newton — so the
gain is a measured area, not an adjective.

**Not claimed, on the evidence above:** "an audio-band, string-drivable gong". The 7 cm plate at
`w = 6e` produces no useful iterate at any cap, and a root found there would be a root of a step
that is itself under-resolved. `tests/helpers.py:736-740` says audio-band, string-drivable and
Picard-convergent "cannot all hold at this sample rate"; this batch may narrow that sentence, and
the plan does not promise to retire it. Whether it does is an outcome, reported either way.

**Not claimed either:** any change to a shipped number. Picard stays the default; every existing
test, frozen value and parity anchor runs the path it runs today.

## 4. The scheme

Newton on `G(w) = w − A^-1(rhs_lin + c·to_live(l(w̄, F̄(w)))) = 0`, with `w̄ = ½(Pw + W_prev)` and
`F̄ = ½(F(Pw) + F_prev)`.

**The Jacobian-vector product, in closed form.** Both `l` and `F` are exactly bilinear/quadratic,
so for a live direction `d` with `D = P d`:

```text
F'(W)[D] = Ainv( -Y * l(W, D) )          the 2 from d/dw l(W,W) absorbs the -½
J d      = d - c * A^-1 * to_live( ½ l(D, F̄) + ½ l(w̄, F'(W)[D]) )
```

No finite differences, no approximation. Cost per product: three bracket evaluations, one Airy
back-substitution and one `A` back-substitution — about 1.5× one Picard sweep, and both
factorizations are already held by the model.

**The linear solve.** GMRES, matrix-free, restarted; no preconditioner (see §1 — `J = I` at zero
amplitude). No new dependency: the core crate's Cargo allowlist is empty by policy and stays so.

**The globalisation.** The residual is cubic in `w`, so a full Newton step can overshoot. A
backtracking (Armijo) line search on the norm of `G`, the same shape as `collision.rs`'s barrier
solve, which is this codebase's precedent. Whether it ever fires is recorded, not assumed — the
reed's Brent fallback is the precedent for *checking* rather than believing.

**Non-convergence contract, stated rather than inherited.** Today a Picard step that runs out of
sweeps returns its last iterate with `converged = false`; Newton keeps exactly that contract, and
the energy bars assert on the **converged path only**. "Any root conserves exactly" is a statement
about a root — an under-relaxed iterate is not one, and the tests must not pretend otherwise.

## 5. Work breakdown — each part's gate green before the next

**Part 0 — the diagnosis, and the honest baseline.** *Lands first, on its own commit.* **DONE —
see §9.** Today `converged = false` conflates two different failures: "ρ < 1 and the cap ran out"
(fixable with a number) and "ρ > 1" (not fixable at any cap). Expose the distinction — ~~the model
already holds two consecutive residuals, so ρ is one division~~ **wrong: `last_residual` is
overwritten every sweep, so there is no second residual anywhere; it is one carried local and one
new field** — and re-measure §2.1 / §2.3's tables against best-effort Picard.
**Do not change `couple_max_iter`'s default:** it is a public constructor
argument reached from `tests/helpers.py`, the airbox VK surfaces and the viewer payloads
(`VK_COUPLE_MAX_ITER`, `VKROOM_SWEEP_CAP`), and moving it silently moves sweep counts in runs other
machinery compares. *Gate:* the three cap-limited fixtures report "cap" and the three divergent ones
report "expansive", and the Picard baseline map is drawn.

**Part 1 — the Jacobian, asserted before it is used.** The Jacobian-vector product as a core
function, checked against a finite-difference directional derivative of `G` on several fixtures and
both boundaries. *Gate:* relative agreement at the finite-difference floor (~1e-7 with a
well-chosen step), and the linearity identity `J(d1+d2) = J d1 + J d2` exact to rounding.

**Part 2 — Newton–Krylov behind a flag.** `couple_method` on `VkSpec` / `VkParams`, default Picard.
Matrix-free GMRES plus the line search. **DONE — see §11.** *Gate:* on every fixture where Picard
converges, Newton reaches the same root to `couple_tol`, and the energy drift bar is met on the
converged path. **Met**, and the gate had to be sharpened first: "the same root" is a claim about
`w`, not about both methods reporting `converged`, and Newton's stopping test had to be the same
*residual* Picard stops on rather than its own step norm — §11.1.

**Part 3 — the convergence map.** Both methods over `(w/e, curvature, fs)`, as a diagnostic script
under `M:\claud_projects\temp\vk-newton\`, with the resulting numbers written into this document.
*Gate:* the boundary moves, quantified; and where it does not move, that is reported in the same
table.

**Part 4 — the documents the probe falsified.** `scientific-hurdles.md` §5's `k²/h⁴` and the same
claim in `tests/helpers.py:729-733`. §5's "the measurement exists" line is *right* and stays —
`n_iters`, `converged` and `last_residual` are exposed with getters and setters at
`crates/physsynth-py/src/plate.rs:1205-1226`, verified.

**Part 5 — the payoff, whatever it is.** Re-run the three scenes §5 names as bounded by the
iteration — the gong on a string, the gong in a room, the mallet on the gong — under Newton, and
report what changed. Including "nothing", if nothing. **One of the three is not runnable as
scoped:** the gong in a room goes through `_VKPlateSurface.solve`, which runs its own Picard loop
against the *loaded* factorization and never consults `couple_method` (§11.8). That is wrapper-tier
work, and Part 5 should scope it in rather than discover it.

## 6. Bars

* **Conservation is unmoved.** Any root of the discrete-gradient equation conserves exactly, so
  Newton changes no energy bar — asserted, not assumed, on the converged path.
* **`nonlinear=False` stays bit-identical** to model #5. Newton never touches the linear path.
* **Picard stays the default**, so the whole existing suite exercises today's code unchanged.
* **The Jacobian is asserted independently** of whether Newton converges (Part 1's gate).
* **Native bars in `crates/physsynth-core/tests/`**, per the migration plan §6 — new physics is
  Rust-first, and there is no Python body to port from.

## 7. Traps, pre-flagged

1. **Flattering the baseline.** Handled structurally by Part 0's ordering, not by care.
2. **A line search that never fires** is untested code. Record whether it fires; if it never does on
   any fixture, say so, and keep it only if a fixture can be found that needs it.
3. **A root that is not a plate.** Where Picard produces no useful iterate at all (residual ≈ 1 from
   sweep 1), a converged Newton root is not automatically physical. Check such roots against the
   energy bar and against a refined-`k` reference before claiming the territory.
4. **GMRES restarts change the iterate.** The restart length is part of the trajectory, exactly as
   the reed's branch choice was; pin it and record it rather than tuning it per fixture.
5. **`ρ` from two residuals is a local estimate.** §2.4 shows it drifting within a single solve, so a
   single ratio is a sample, not a spectral radius. Report it as what it is.
6. **The free boundary's `h²`.** `couple_factor` carries an extra `h²` on a free edge; the Jacobian
   product must carry it too, and the cymbal fixture is the one that catches it.

## 8. Cost

Part 0 is small (a diagnosis field and a re-measurement). Parts 1–2 are the batch: the Jacobian
product is ~40 lines given the existing bracket and Airy solvers, GMRES with restarts and an Armijo
search ~150 lines, and the flag plumbing through `VkSpec` / `VkParams` / the binding ~60. Parts 3–5
are measurement and prose. No new dependency, no new factorization, and no change to the default
path.

---

## 9. Part 0's result — the verdict the model now returns, and the honest baseline

Landed 2026-09-06, on its own commit, ahead of any Newton code.

### 9.1 What the model reports

`VkPlate` (and `VKPlate` through the binding) carries one new number and one derived verdict
alongside the existing `n_iters` / `converged` / `last_residual`:

* **`residual_ratio`** — the last two relative increments' ratio, `NaN` when fewer than two sweeps
  ran. Stored.
* **`couple_outcome`** — `"converged"` / `"capped"` / `"expansive"` / `"unknown"`. **Derived on
  every read** from the four fields above rather than stored, so a caller who writes `converged` or
  `last_residual` by hand (both are public, and both have binding setters) cannot leave a stale
  verdict behind.

Two things about the classifier are deliberate:

1. **The `capped` test is positive** — the ratio must be *finite and below one*. An overflowed step
   carries a NaN residual, and `NaN >= 1.0` is `false` just as `NaN < 1.0` is, so a negative test
   would have filed a plate that blew up under the one outcome a bigger cap fixes. There is a native
   bar on exactly this.
2. **`unknown` exists** because one sweep forms no ratio. `probe_rho.py` runs at `cap = 1`; without
   this fourth value that run would have to be called something it is not.

### 9.2 The ratio is not §2.4's ρ, and reporting it as such would be wrong

§2.4 measures the ratio at **sweeps 1–8 of step 1**, from a fresh plate per cap. `residual_ratio` is
measured at the **exit sweep of whatever step last ran**. Since §2.4's own finding is that the factor
*climbs through 1 within a single solve*, these are different quantities by construction and the new
field does not reproduce §2.4's table. That is correct behaviour, not a transcription error — and the
exit ratio is the better estimator for the question a cap actually poses, which is whether more
sweeps would help **from here**. Trap 5 stands: it is a sample, and the name says so.

### 9.3 The baseline, drawn against best-effort Picard

`M:\claud_projects\temp\vk-newton\part0_baseline.py`, the same six fixtures as §2.3, 300 steps,
counting outcomes **per step** — the last step's verdict is not the run's, because a divergent run
passes through capped steps before it expands.

| fixture | cap 50: capped / expansive | best effort | max sweeps | drift | verdict |
|---|---|---|---|---|---|
| 40 cm, 3 cm strike, w=6e | 3 / 0 | cap 400 | 143 | 4.1e-13 | **cap** |
| 40 cm, 8 cm strike, w=16e | 29 / 0 | cap 3000 | 724 | 4.3e-13 | **cap** |
| 16 cm, 3.2 cm strike, w=6e | 2 / 0 | cap 400 | 76 | 6.0e-13 | **cap** |
| 12 cm, 2.4 cm strike, w=6e | 0 / 1 | dies at every cap | — | overflow | **wall** |
| 40 cm, 8 cm strike, w=20e | 0 / 2 | dies at every cap | — | overflow | **wall** |
| 40 cm, 8 cm, w=12e, 24 kHz | 0 / 1 | dies at every cap | — | overflow | **wall** |

**The gate is met**, and the two halves rest on different amounts of evidence, which is worth
saying rather than averaging: the cap-limited fixtures reported `capped` and **never** `expansive`
across all 300 steps, while the divergent ones reported `expansive` on the one or two steps they got
to run before the energy overflowed. §9.4 is where the second half is measured properly.
The three cap-limited fixtures reproduce §2.3's sweep counts
and drifts exactly (143 / 724 / 76 sweeps; 4.1e-13 / 4.3e-13 / 6.0e-13), and the wall-clock cost of
the generous cap is 0.44 → 0.49 s, 0.83 → 1.07 s and 0.07 → 0.07 s over 300 steps — the expensive
steps stay rare.

### 9.4 The wall is decided on step zero — measured, not inferred

§9.3's run *suggests* this (`DIED@0`, and `DIED@1` for the loud one at cap 50) but does not show it:
the step at which the energy overflows **moves with the cap**, so a death at step 0 is not by itself
a verdict about step 0. `part0_first_step.py` asks the model directly, one step from rest:

| fixture | cap 50 | cap 400 |
|---|---|---|
| 40 cm, 3 cm strike, w=6e | capped (50 sweeps, ratio 0.820) | **converged** (143 sweeps) |
| 40 cm, 8 cm strike, w=16e | capped (50, 0.751) | **converged** (76) |
| 16 cm, 3.2 cm strike, w=6e | capped (50, 0.690) | **converged** (76) |
| 12 cm, 2.4 cm strike, w=6e | **expansive** | **expansive** |
| 40 cm, 8 cm strike, w=20e | **expansive** (ratio 1.000) | **expansive** |
| 40 cm, 8 cm, w=12e, 24 kHz | **expansive** | **expansive** |

Every divergent fixture is already `expansive` on its **first** step, at both caps, and every
cap-limited one is `capped` or `converged` there. The wall is a property of the initial condition,
not something a run drifts into. Two consequences for the parts that follow:

* Part 3's convergence map can be drawn from **one step per point** rather than a 300-step run,
  which makes a fine grid over `(w/e, curvature, fs)` affordable. This is the claim that cost
  estimate rests on, which is why it is measured here rather than read off §9.3.
* A Newton root claimed in that territory is a root of the *first* step from a struck initial
  condition, and trap 3 applies to it in full.

Note also that the contraction factor is essentially cap-independent on the converging fixtures
(0.820 / 0.819, 0.751 / 0.750, 0.690 / 0.689) — consistent with §2.3's "ρ is a clean constant in
every converging fixture", now visible from the model itself.

### 9.5 Deliberately not done here

* **`couple_max_iter`'s default does not move** — it stays 50. It is a public constructor argument
  reached from `tests/helpers.py`, the airbox VK surfaces and the viewer payloads, and moving it
  would silently move sweep counts in runs other machinery compares. Every table above passes the
  cap explicitly.
* **The viewer still counts `n_not_converged` without splitting it** (`web/serialize.py`). It could
  now report cap-versus-wall; that is a viewer change, not Part 0, and no part of this plan needs it.
* §5's `k²/h⁴` and the same claim in `tests/helpers.py` are **Part 4**, untouched here.

**One known fork, recorded now rather than discovered in Part 2.** `VkPlate::step` and
`PyVKPlate::step` each copy the step's diagnostics into their own struct in a **hand-written block**
— the binding does not delegate to the core model, it holds its own fields and calls `core::vk_step`
directly. So `residual_ratio` is copied twice, and an edit to one block does not fail the other.
That is the shape of the §45.9 `THETA_DEFAULT` fork already in this project's history. Part 2 adds
`couple_method` to both structs, and this is where it will drift if it drifts.

---

## 10. Part 1's result — the Jacobian, asserted before Newton is allowed to use it

Landed 2026-09-06, on its own commit, ahead of any Newton code. No flag, no binding change, no
solver: this part adds a derivative and four native bars, and changes no shipped number.

### 10.1 The refactor, and the claim a finite difference cannot make

§4 gives the Jacobian-vector product in closed form, and the obvious way to add it is to write it
next to the Picard loop. That would have been the §9.5 fork shape again, and worse than a fork: if
the sweep and the residual drift apart, `J` becomes the exact derivative of a function the loop is
not iterating, and **every finite-difference check still passes**. A finite difference verifies a
derivative; it cannot verify that the derivative is of the right map.

So `vk_step`'s loop body was *moved*, not copied. `VkCoupledStep` holds one step's sweep-invariant
parts — `rhs_lin`, `w_prev_full`, `f_prev`, `couple_factor` — and offers three readings of one map:

| method | what it is |
|---|---|
| `sweep(w)` | the Picard map `A^-1(rhs_lin + c·to_live(l(w̄, F̄)))`, which is now literally the loop body |
| `residual(w)` | `w − sweep(w)`, a **wrapper over** `sweep` |
| `jacobian_vector(av, d)` | `G'(w)d` in closed form |

The direction of that wrapping is load-bearing and is the reverse of what reads more naturally: the
loop keeps using `sweep`'s own output, because recovering it as `w − G(w)` is exact only under
Sterbenz and would have changed the shipped trajectory.

**The refactor is bit-identical**, measured rather than argued: 6 fixtures (both boundaries × three
amplitudes) × 50 steps = 300 rows of state hash, sweep count, `last_residual` and `residual_ratio`,
`diff`-clean across the change. That is a separate claim from the derivative being right, and it
was established separately, before the derivative existed.

`couple_factor` is formed **once**, in the context, and read by both `sweep` and
`jacobian_vector`. This is how §7's trap 6 — the free edge's extra `h²` — is disposed of, and §10.3
shows why it had to be structural.

### 10.2 The gate, and what it measured

Two linearisation points per fixture (the loop's seed `2w^n − w^{n-1}`, and where Picard actually
lands), one deterministic xorshift direction, central differences at `ε = 10⁻⁵‖w‖`.

| claim | bar | measured |
|---|---|---|
| `‖FD − Jd‖ / ‖Jd‖` | `< 1e-7` (plan §5) | **9.5e-12 … 1.1e-10** |
| `J(d₁+d₂) − Jd₁ − Jd₂`, relative | `< 1e-13` | **8.9e-17 … 1.4e-16** |
| `‖G(w)‖/‖w‖` at the step's own answer | `≤ 10·couple_tol` | **3.4e-14 … 7.5e-14** (tol 1e-13) |
| `‖Jd − d‖/‖d‖`, gated fixtures | `> 1e-2` | **0.024 … 0.124** |

**The step is chosen, not guessed.** `G` is exactly cubic in `w` (the bracket is bilinear, `F` is
quadratic, `w̄` is affine), so a central difference has a pure `ε²` truncation term and the error
curve is a clean V. Sweeping `ε` from `1e-1` to `1e-14` on one fixture: `2.1e-4, 2.1e-6, 2.1e-8,
2.1e-10, **2.1e-11**, 2.0e-10, 1.3e-9, 1.4e-8, 1.2e-7, …` — four decades of `ε²` descent, a floor
at `1e-5`, then roundoff. Two decades either side of the chosen step still sit three orders under
the bar, so this is not a number tuned to pass.

The third row's bar was **loosened from `couple_tol` to `10·couple_tol` after measuring**. At the
tolerance itself the headroom was 1.3×, which is not enough for a quantity produced by two sparse
back-substitutions on a machine that is not this one — the same argument as CLAUDE.md's refusal to
tighten the `1e-10` energy bar. It costs no discriminating power: a context that did not match the
model's would be wrong by a factor, not by a percent.

### 10.3 What the bars actually catch — six mutants, run

A bar is worth what it detects, so `jacobian_vector` was deliberately broken six ways and the four
tests re-run against each.

| mutant | FD | margin | linearity | root |
|---|---|---|---|---|
| `F'` taken at `w̄` instead of the raw `W` | **caught** | — | — | — |
| the whole coupling term dropped (`J d = d`) | **caught** | **caught** | — | — |
| one of the two `½`s dropped | **caught** | — | — | — |
| the sign of `−Y` flipped | **caught** | **caught** | — | — |
| the residual scaled by 1.01 | **caught** | — | — | — |
| the free edge's `h²` dropped | caught\* | caught\* | caught\* | caught\* |

Four things this says that the passing numbers do not.

**The margin floor is not decoration, and its position is measured.** At zero amplitude `J = I`
exactly (§1), so a `jacobian_vector` that returns `d` unchanged is a *good approximation* at low
amplitude and passes a finite-difference check there. This is not hypothetical. The first mutant —
`F'` at `w̄` instead of `W`, a plausible transcription error and roughly a factor of two — measures
`8.5e-3` on the loud fixtures and **`6.3e-8` at margin `5.9e-3`**: green, just under the bar, on a
fixture nobody would have flagged as weak. So the floor sits between two measurements rather than
at a round number: the gated fixtures run `0.024`–`0.124`, and the bar demonstrably goes soft by
`0.0059`.

**`couple_factor` is where the honest sentence differs from the tempting one.** The asterisked row
was caught by all four bars — but as **the physics blowing up**, not as a derivative disagreement:
removing the `h²` multiplies the coupling by 2500, the map diverges, and the point overflows to
`NaN`. A finite difference is *structurally blind* to `couple_factor`, because `sweep` and
`jacobian_vector` read the same field. **That sharing is the guard for trap 6; the bar is not**, and
a Part 2 that believed otherwise would be relying on nothing.

**The linearity identity caught nothing on its own merits, and that is worth saying plainly.** Its
single hit is the `h²` row, and it saw that only because the map diverges and the point overflows —
not because anything stopped being linear. Every other mutant, *including the one that scales the
residual by 1.01*, leaves `J` exactly linear in `d`, which is what a linear operator assembled from
bilinear pieces does whichever piece you get wrong. So it is a structural check against a `J` that
is not an operator at all, and counting it as one-in-six would overstate it in both directions.

**The root check has the same single hit, and its blind spot is instructive.** It exists to pin the
claim no finite difference can make — that `VkCoupledStep` is the same map `VkPlate::step`
iterates. The 1.01 mutant genuinely changes that map and the check stayed quiet, because 1% of a
residual already at `7e-14` is under any bar this check could carry (tightening it back to
`couple_tol` would not have caught it either). What it does guard is a context that is *wrong*
rather than *slightly off* — an error of a factor, which is what a mismatched `rhs_lin`, `f_prev`
slot or `couple_factor` produces, and which is exactly what the `h²` row demonstrates.

### 10.4 A centred strike makes a free plate and a supported one the same arithmetic

Trap 6 says "the cymbal fixture is the one that catches it". The free fixture nearly did not exist
in any meaningful sense, and the tell was a coincidence: a free and a supported 40 cm plate,
identical but for the boundary, with **different node counts** (441 vs 361) and `couple_factor`s
differing by exactly `h²`, returned margins agreeing to **twelve digits**.

The explanation is the strike. A 3 cm or 8 cm Gaussian centred on a 40 cm plate is `exp(−44)` at
the rim — zero in doubles. The free plate's extra rows never move, and by construction
`c_free·A_free⁻¹` reduces to `c_sup·A_sup⁻¹` on the interior (the extra `h²` is precisely the one
the free mass matrix divides back out). So the two runs were the same interior problem, and the
free boundary was untested.

The fixture table now carries a free plate struck **off-centre** (`0.12 m` from the middle), which
puts 35–38% of the peak on a free edge; its margin separates from its supported twin as it should.
The rim reach was measured, not assumed. **Generalisable:** on this project a "free boundary
fixture" is only a free-boundary fixture if the excitation reaches the boundary, and no detector in
the suite says otherwise — the energy bar, the FD bar and the margin all pass identically either
way.

### 10.5 Deliberately not done here

* **No `couple_method` flag, no binding change, no GMRES.** That is Part 2. `vk_step`'s signature
  and `VkStep`'s fields are untouched, so §9.5's hand-written diagnostics fork in
  `PyVKPlate::step` is exactly as it was, and Part 2 inherits it unchanged.
* **The `f` argument of `vk_step` is dead on the nonlinear path** — `f_new_full` is unconditionally
  overwritten on sweep 1, since the spec forbids a zero cap. Noticed while moving the loop; not
  fixed, because changing that signature is a Part 2-sized ripple for no gain here.
* **Nothing was measured about Newton's convergence.** Part 1 asserts a derivative, and that is
  all it asserts. Whether a Newton step built on it moves the wall is Parts 2 and 3, and §2.3's
  ordering hazard — that a cap-50 Picard baseline would flatter it — was already discharged by
  Part 0.
* **The probe scripts stay in `M:\claud_projects\temp\vk-newton\`** (`redteam.py`,
  `redteam2.py`, and the two dump comparisons). They mutate `crates/physsynth-core/src/plate.rs`
  in place and restore it from memory at exit, so a killed run leaves a **mutated core on disk** —
  one of them changes the free plate's default path by a factor of 2500. Verify with `git diff`
  after running one; do not trust the script's own "restored" line.

---

## 11. Part 2's result — Newton behind the flag, and the wall it walks through

Landed 2026-09-06, on its own commit. `couple_method` defaults to `"picard"`, so no shipped number
moves; what is new is a second iteration, a linear solver asserted before the plate touches it, and
one number on which the two methods can honestly be compared.

### 11.1 The stopping test is a residual, and getting that wrong would have hollowed out the gate

§4 said "Newton on `G(w) = 0`" and left the convergence test implicit, and the obvious reading —
stop when the Newton step `Δ = −J⁻¹G` is small — is wrong here in a way that would not have shown
up on any fixture in Part 1.

Picard's `last_residual` *looks* like an increment and **is** a residual: the loop's
`diff = sweep(w_j) − w_j` is exactly `−G(w_j)`, so the shipped loop already stops on
`‖G‖ / ‖w‖`. Newton's step is a different quantity, and the two agree only while `J` is close to
`I` — which is 2.4%–12.4% on Part 1's fixtures and **unbounded** in the `ρ > 1` region this batch
exists for. Stopping Newton on its step norm there would mean stopping at an unknown residual
level, and §5 Part 2's gate ("Newton reaches the same root to `couple_tol`") would have been
uncheckable while reading as though it had been checked.

So Newton stops on `‖G(w_new)‖ / max(‖w_new‖, 1e-30) ≤ couple_tol` — the same quantity Picard
stops on. It is free (that residual is the next iteration's right-hand side anyway), it keeps
Part 0's `couple_outcome` classifier meaning what it meant, and it is what lets Part 3 draw one
picture instead of two.

One convention still differs and is written down rather than smoothed over: Picard measures the
residual at the *previous* iterate (`G(w_{j−1})`, normalised by `w_j`), because a sweep produces
both at once; Newton measures it at the iterate it returns. Newton's is the honest one and the
difference is one tolerance wide.

### 11.2 GMRES is a module, and it is asserted against matrices, not against the plate

`crates/physsynth-core/src/krylov.rs` — matrix-free restarted GMRES over a closure, modified
Gram–Schmidt with Givens rotations, no preconditioner (§1: `A⁻¹` is already inside `J`). No new
dependency; the core crate's Cargo allowlist is still empty.

It is a sibling module rather than a private function in `plate.rs` for the reason `root` is not
inside `string_nonlinear`: *"the plate converged"* is a weak bar for a linear solver, because a
Newton iteration that overshoots and is caught by the line search still ends up at the root.
`crates/physsynth-core/tests/krylov.rs` asserts it on systems whose answer is known in closed form
— seven bars, of which three are worth naming:

* **`b` is built as `A x_true`**, so the answer is known rather than compared against a second
  solver that could share a mistake. Non-symmetric on purpose: `J = I − c A⁻¹ K` is a product of
  two symmetric operators and is therefore not symmetric, so a solver asserted only on an SPD
  system would be asserted on a case the caller never presents.
* **Trap 4, in its exact form.** The restart length is allowed to move the *cost* and forbidden to
  move the *answer*: restarts 3, 8 and 30 on a 25-dimensional system agree to `1e-10` while the
  short one demonstrably pays for forgetting. That is what makes a pinned restart a pin rather
  than a tuning knob.
* **The recursion's residual is the true one.** Givens tracks `‖b − Ax‖` without forming it; if
  that bookkeeping were wrong, the solver would stop early or late and *every other test would
  still pass*, because they all check the answer at a tolerance the recursion itself chose. One
  test forms the residual explicitly and closes the loop.

The near-identity fixture measures what the unpreconditioned choice is worth: 8, 13 and 25 products
at dimension 40 for spectra spread 0.018, 0.092 and 0.367 around 1. The *shape* is the claim — the
cost tracks the distance from `I` and stays under the dimension.

### 11.3 The driver, and the one thing that keeps trap 6 shut

`vk_newton` takes a `&VkCoupledStep` and reads only `couple_tol` and `couple_max_iter` off the
parameters. It forms no `h`, no `k` and no `rho_s`. That is deliberate and it is the whole of trap
6's guard: §10.3 established that a finite difference is **structurally blind** to `couple_factor`,
because `sweep` and `jacobian_vector` share the field — so a Newton driver that recomputed the free
edge's extra `h²` for itself would have re-opened a hazard nothing in the suite can see. It does
not compute it; it never sees it.

Two smaller decisions, recorded because they are choices:

* **The seed is Picard's**, `2wⁿ − wⁿ⁻¹`, with no Picard warm-up hybrid. A hybrid is a tuning knob
  that would make Part 3's map a map of the knob.
* **`F` is the stress function of the iterate that is returned.** Picard hands back `F` of the
  sweep's *incoming* iterate — the Python original's convention, kept because changing it moves
  every shipped number. Newton's final residual evaluation has already computed `F(w)` at the
  accepted `w`, so that is what it returns. This makes Newton's energy bookkeeping consistent at
  the returned state rather than to within a tolerance. It is a difference **between the methods**,
  not evidence that one of them is better physics, and a drift comparison between the two should
  not be read as one.

### 11.4 `n_solves` — one field, because a field here costs two

§9.5's fork is still there: `VkPlate::step` and `PyVKPlate::step` each copy the step's diagnostics
in a hand-written block, and every field is somewhere the two can drift apart. So `VkStep` gains
exactly **one**: `n_solves`, the count of back-substitutions the step spent.

It is the cost axis, and `n_iters` cannot be. One Picard sweep is two solves (one Airy, one
theta-scheme); one Newton iteration is two, plus two per Krylov product and two per line-search
trial. Reporting `n_iters` for both would have shown Newton ahead by a factor of twenty on runs
where the two are level. A count rather than a wall clock because a count does not move with the
runner.

Everything else a Newton step is worth knowing — line-search halvings, Krylov products, inner
stalls — lives on `VkNewtonReport`, which native tests read directly and which nothing has to
mirror. That is how trap 2 is answered without paying the fork twice.

The binding's `n_solves` is read by `tests/test_binding_surface.py`, which is what exercises the
second copy of the block at all. **`airbox`'s seam does not write it:** `_VKPlateSurface.solve`
runs its own Picard loop against the loaded factorization and writes `n_iters`, `converged` and
`last_residual` by hand — not `residual_ratio` (already true after Part 0) and not this. After a
room step both are the last *bare* step's.

### 11.5 The gate, met — and the claim stated as a claim about `w`

`G` is cubic, so "both converged" leaves open that they converged to *different* roots. The gate is
therefore stated on the displacement, over Part 1's six fixtures (both boundaries, three
amplitudes, the off-centre free strike included):

| claim | bar | measured |
|---|---|---|
| `‖w_newton − w_picard‖ / ‖w_picard‖`, one step | `< 1e-8` | passes on all six |
| energies of the two states | `< 1e-12` relative | passes on all six |
| Newton's energy drift, 300 steps, both boundaries | `< 1e-10` (project bar) | passes, every step converged |
| `nonlinear=False`, 40 steps, either method | `array_equal` | exact, both boundaries |
| Picard's solve count | `== 2 · n_iters` | exact |
| Newton's solve count | `== 2 + 2·products + 2·trials` | exact |

The linear bar is structural as well as measured: `vk_step`'s `!nonlinear` early return is *before*
the method branch, so the two paths are the same line of code. It is asserted anyway, because
"before" is a property of the source that a later edit can quietly reverse.

### 11.6 The line search does not fire, and *where* it starts to is the interesting number

Trap 2 says a line search that never fires is untested code, so it was measured rather than
assumed. On all six gate fixtures and all three of §9.4's wall fixtures, the full Newton step is
accepted **every single time**: zero halvings.

The seed had to be pushed a long way before backtracking became necessary, and further than one
would guess:

| seed × | Newton iterations | halvings | Krylov products | inner stalls |
|---|---|---|---|---|
| 1 | 4 | 0 | 20 | 0 |
| 10 | 8 | 0 | 115 | 0 |
| 40 | 11 | 0 | 1039 | 5 |
| 200 | 21 | **6** | 2524 | 12 |
| 1000 | 32 | 14 | 4752 | 23 |
| 10⁴ | 48 | 33 | 8103 | 40 |

All of them converge. So the residual stays convex enough for an undamped step across two orders of
magnitude of nonsense, and the search first earns its place at 200×. Both halves are asserted: that
it does *not* fire at 40×, and that it does at 200× and still reaches the root. The test says
explicitly that a future failure of the first half is a finding to re-measure, not a number to
raise.

Note the third column. GMRES starts hitting its 200-product cap well before the line search fires,
and the inexact correction is then handed to the search — which is the designed behaviour, recorded
rather than tuned away.

### 11.7 Newton converges on all three walls — what that does and does not say

This is Part 3's territory and Part 3 will draw it properly, over a grid. But the three fixtures
§9.4 classified as `expansive`-on-step-zero were run under both methods, because it costs nothing
and because a Part 2 that shipped without looking would have been odd. **These are §9.4's fixtures
to the digit** — same geometry, same `N = 20`, same `rho = 7800` — which is the only thing that
makes the Picard column below a cross-reference rather than a new measurement. (An earlier draft of
this section ran the 12 cm plate at `N = 6` and got `capped` rather than `expansive`; the row was
not §9.4's, and the discretisation was the whole difference.)

| fixture | Picard, cap 400 | Newton, first step | solves | 300-step drift |
|---|---|---|---|---|
| 12 cm, 2.4 cm strike, 6e | `expansive`, NaN at 400 sweeps | **6 iterations** | 96 | 6.3e-13 |
| 40 cm, 8 cm strike, 20e | `expansive`, NaN | **4 iterations** | 54 | 1.4e-12 |
| 40 cm, 8 cm, 12e, 24 kHz | `expansive`, NaN | **4 iterations** | 60 | 9.1e-13 |

Every step of all three 300-step runs converged, and the worst drift is two orders under the
project's `1e-10` bar.

**What this says:** there is a root there, and Newton finds it while Picard's map is expansive.
That is a property of the *iteration*.

**What it does not say:** that the territory is now a plate. Trap 3 — "a root that is not a plate"
— is answered here only as far as this project's primary detector goes. Energy conservation is a
property of *any* root of the discrete-gradient equation, so a converged run conserving to 1e-12 is
exactly what a root must do and is not independent evidence that the root resolves the physics. The
other half of trap 3, a comparison against a refined-`k` reference, is Part 3's, and until it is
done the honest reading is "the solver got there", not "the plate is audible". §3's refusal to
claim an audio-band string-drivable gong stands untouched by this table.

### 11.8 Deliberately not done here

* **The airbox VK scenes cannot run under Newton, and no amount of flag plumbing fixes that.**
  `_VKPlateSurface.solve` (`crates/physsynth-py/src/airbox_wrap.rs`) runs its **own** Picard loop,
  in Python-object arithmetic, against the room-**loaded** factorization — which is the whole point
  of that seam, since the bare model's `lu` is the wrong operator once the air is attached.
  `couple_method` lives on `VkParams` and that loop never consults it. So Part 5's "gong in a room"
  is not runnable under Newton without wrapper-tier work, and Part 5 should scope that in rather
  than discover it. The gong on a string and the mallet on the gong go through `VKPlate.step` and
  are unaffected.
* **The convergence map.** Part 3. §11.7 is six points and a footnote, not a map, and the
  boundary's *position* is not asserted anywhere.
* **No default moves.** `couple_method` defaults to `"picard"` at both levels (the core's
  `VkSpec::default()` and PyO3's signature, asserted separately because they are separate facts),
  `couple_max_iter` is still 50, and every existing test, frozen value and viewer payload runs the
  path it ran yesterday.
* **`tests/helpers.py`'s `make_vk_room_bare_twin` now copies `couple_method`** along with the tol
  and the cap. Not a live bug — nothing sets the flag today — but that function exists precisely so
  a defaulting difference cannot masquerade as the physics, and a new constructor argument it did
  not copy would have been one.
* **The `f` argument of `vk_step` is still dead on the nonlinear path** (§10.5). Newton does not
  read it either.
