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
**DONE — see §13.** *Gate:* the boundary moves, quantified; and where it does not move, that is
reported in the same table. **Met** — the boundary moves in fourteen of fifteen cells and the
fifteenth is censored rather than flat. The gate turned out to be the smaller half of the result:
the map has **two** boundaries (a step-zero one and a 300-step one, and they differ in five cells),
the outcome is **not monotone** in amplitude, and trap 3's answer is **graded along curvature**
rather than yes or no.

**Part 4 — the documents the probe falsified.** `scientific-hurdles.md` §5's `k²/h⁴` and the same
claim in `tests/helpers.py:729-733`. §5's "the measurement exists" line is *right* and stays —
`n_iters`, `converged` and `last_residual` are exposed with getters and setters at
`crates/physsynth-py/src/plate.rs:1205-1226`, verified. **DONE — see §12**, and this scope was
**two sites out of six**: the claim had propagated to `HANDOFF.md`, this plan's predecessor and a
batch record, none of which the part named.

**Part 5 — the payoff, whatever it is.** Re-run the three scenes §5 names as bounded by the
iteration — the gong on a string, the gong in a room, the mallet on the gong — under Newton, and
report what changed. Including "nothing", if nothing. **DONE — see §14**, and **only one of the
three is a scene at all.** The gong in a room is wrapper-tier work, stated once in §11.8 and not
restated here. The mallet on the gong turned out never to have been built: `MalletMembrane` casts
its collaborator to a `Membrane` and there is no mallet-on-a-plate composition anywhere in the repo
(§14.1). Scoped by the human on 2026-09-06 to **measure the one and report the other two**; the
wrapper work is deferred to its own batch.

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

---

## 12. Part 4's result — six sites, two kinds of edit, and a second claim under the first

Landed 2026-09-06, docs plus one comment block, no behaviour change and no new measurement:
§2.2–§2.4 had already measured every leg, so this part is transcription.

### 12.1 The part's own scope was wrong by four sites

§5 named two places: `scientific-hurdles.md` §5 and `tests/helpers.py:729-733`. A repo-wide grep for
the mechanism (`h^4`, `h⁴`, `k^2/h`, `k²/h`) found **six** — the two named, plus `HANDOFF.md:759`,
`docs/dev/string-vk-plate-bridge-plan.md:355` and `docs/memory/air-box-state.md:678`, with
`scientific-hurdles.md` carrying it twice (the summary table's row 5 said "small `h`" as well as the
body's formula). The other twenty-odd `1/h⁴` hits in the repo are the biharmonic operator's own
stencil and the explicit scheme's CFL, and are correct; the discriminator is `k²` next to it.

This is the same shape as the migration's "a derived CI list was wrong by 43 of 65 files": **a claim
propagates by quotation, and the part that retires it inherits whatever the plan author happened to
remember.** Grep for the claim, not for the files.

### 12.2 Two kinds of site, and only one of them gets rewritten

A **live** claim is one a reader is told to consult before doing work — `HANDOFF.md`,
`scientific-hurdles.md` and a comment sitting on the constructor argument it justifies. Those are
rewritten in place, because a wrong mechanism there gets *acted on*.

A **batch record** — `string-vk-plate-bridge-plan.md` §10.4, `air-box-state.md` batch 6 — is a report
of what that batch concluded, and its measured numbers and its own conclusion are both still right
(§10.4's title, "the rig's size is set by the fixed point", survives the correction untouched).
Those get an annotation and a forward pointer, and keep their narrative. Erasing the trail would
cost more than a stale mechanism that says where it was corrected.

### 12.3 The mechanism was not the only falsified claim at those sites

Every live site carried a second one underneath: the categorical *"audio-range modes at 0.1 mm need
a ~7 cm plate, **which is exactly the size that will not converge**"*, echoed as "cannot all hold at
this sample rate". §2.4 measured a 7 cm plate converging in 8 sweeps at `w = e` and failing by
`w = 2e` — so at a fixed size the ceiling is an **amplitude**, and attributing the failure to the
size is the same category error as attributing it to `h`.

**But the trilemma itself survives, and checking that took one number.** `probe_rho.py` runs
`e = 1.0e-3`; `VK_BRIDGE_MAT` is `e = 1.0e-4`. The 7 cm plate that converges is a **1 mm** plate —
audio-band and Picard-convergent, but a decade too thick to be string-drivable, which is the
trilemma's third leg. So the honest edit is not "the trilemma is false": it is that the *reason*
given for it was wrong, that 7 cm **at 0.1 mm** has never been measured, and that the sentence is
therefore a standing refusal rather than a proven impossibility. Had the probe run at 0.1 mm the
edit would have been much stronger, and the difference is one constant in a scratch script.

Two general things follow. **A falsified mechanism is worth re-reading the sentences around it**,
because the wrong mechanism is usually load-bearing for a nearby conclusion. And **the thing that
decides how strong a correction may be is often a parameter the correcting measurement did not
restate** — §2.4's table names side, strike width and amplitude, and not the thickness that turns
out to settle it.

### 12.4 What was deliberately not written

* **Part 3's answer.** §11.7 is three fixtures; the boundary's *position* is still unasserted, so
  the edits downgrade "cannot converge" to "converges up to an amplitude" and leave the map as a
  named gap rather than filling it with six points.
* **A "fixed" status.** Row 5 of the summary table moves to "Open, narrowed", not to fixed: Picard
  is still the default, three fixtures were genuine divergence, the refined-`k` half of trap 3 is
  untouched, and one of the three bounded scenes cannot run under Newton at all (§11.8).
* **`VK_BRIDGE_SIDE = 0.4` does not move**, and the comment now says so explicitly. The measurement
  that picked it (40 cm converges to `w = 9e`, 8 cm caps out by `w = 6e`) is unaffected by the
  correction; only its explanation was. A constant whose rationale is corrected invites the next
  reader to re-litigate the constant, and the cheapest defence is one sentence.
* **`docs/memory/MEMORY.md`** needed no edit. Its only `1/h⁴` is the orthotropic free plate's
  roundoff scar (`h²` × `1/h⁴`), an unrelated and correct use.

---

## 13. Part 3's result — the map, its second boundary, and where a root stops being a plate

Landed 2026-09-06. Three scripts under `M:\claud_projects\temp\vk-newton\` (`part3_map.py`,
`part3_runmap.py`, `part3_refine.py` / `part3_refine2.py`) and two native bars in
`crates/physsynth-core/tests/plate.rs`. No default moves and no shipped number changes.

### 13.1 What was drawn, and the one thing a reader cannot infer from the numbers

A 40 cm steel square, 1 mm thick, `N = 20`, supported, struck from rest with a centred Gaussian —
the probe's plate, so §2.1–§2.4 continue here rather than restarting. Axes: `w/e` over sixteen
values from 1 to 40, strike width over 12 / 8 / 5 / 3 / 2 cm, `fs` over 24 / 48 / 96 kHz. Both
methods at every point: Picard at a **cap of 2000** (best effort, because §2.3's whole point is that
a cap-50 baseline would credit Newton with territory a constant recovers) and Newton at 50.
480 points, one step each, 77 seconds.

**The width sweep holds the peak amplitude fixed**, so the injected energy falls as the strike
narrows. That is §2.2's knob — "narrowing the strike alone, at fixed plate, fixed grid and fixed
peak amplitude" — and it is stated because a fixed-*energy* sweep is an equally defensible
experiment that would draw a different boundary, and no column below reveals which was run.

**One step per point**, per §9.4. **No bisection**, per §2.4: the 7 cm plate's `ρ` wandered between
0.6 and 1.5 at `w = 2e` while converging cleanly at `w = e`, which is exactly the shape that breaks
a bisection invariant — and §13.5 shows the invariant does in fact fail somewhere on this grid. The
boundary is therefore reported as **two columns**, largest converged and smallest failed, so that a
cell where they are not adjacent shows up as a finding instead of being averaged into a midpoint.

**Newton's four pins are reported, not tuned** (`crates/physsynth-core/src/plate.rs`: "Part 3
reports what they gave"): GMRES restart **30**, at most **200** Krylov products per iteration, a
constant forcing term of **1e-4**, and an Armijo constant of **1e-4** with at most 40 halvings. One
value per fixture would have turned this map into a map of the choices.

### 13.2 The boundary moves in every cell that is not censored

`w/e` at which each method last converged on step zero, and where it first did not:

| fs | strike | Picard: last ok | first bad | Newton: last ok | first bad | moved |
|---|---|---|---|---|---|---|
| 24 kHz | 12 cm | 16 | 18 | **40** | — | +24 |
| 24 kHz | 8 cm | 10 | 12 | 32 | 40 | +22 |
| 24 kHz | 5 cm | 6 | 8 | 18 | 20 | +12 |
| 24 kHz | 3 cm | 3 | 4 | 8 | 10 | +5 |
| 24 kHz | 2 cm | 2 | 3 | 4 | 6 | +2 |
| 48 kHz | 12 cm | 28 | 32 | **40** | — | +12 |
| 48 kHz | 8 cm | 18 | 20 | **40** | — | +22 |
| 48 kHz | 5 cm | 10 | 12 | 32 | 40 | +22 |
| 48 kHz | 3 cm | 6 | 8 | 18 | 20 | +12 |
| 48 kHz | 2 cm | 4 | 6 | 10 | 12 | +6 |
| 96 kHz | 12 cm | **40** | — | **40** | — | 0 (both censored) |
| 96 kHz | 8 cm | 32 | 40 | **40** | — | +8 |
| 96 kHz | 5 cm | 20 | 24 | **40** | — | +20 |
| 96 kHz | 3 cm | 12 | 14 | 32 | 40 | +20 |
| 96 kHz | 2 cm | 8 | 10 | 20 | 24 | +12 |

Bold entries are **censored at the top of the grid**: the method never failed up to `w = 40e`, so
the number is a floor and the true boundary is somewhere above it. Six of Newton's fifteen cells are
censored, which is why "moved" is a floor too; the one cell reading 0 is the one where *both* are
censored, and it is not a cell where Newton gained nothing. Everywhere the comparison is uncensored,
Newton's boundary is **two to four times** Picard's amplitude.

The axes behave as §2.2 predicted: at fixed `fs`, halving the strike width roughly halves the
boundary for both methods, and doubling `fs` roughly doubles it. Curvature and `k` are the two
knobs, and the map shows Newton *shifting* that surface rather than changing its shape.

### 13.3 Newton is not free, and this is the argument for Picard staying the default

Back-substitutions per step (`n_solves` — the cost axis the binding says is the comparable one;
`n_iters` is not, because one Picard sweep is two solves and one Newton iteration is two plus two
per Krylov product and two per line-search trial):

| fs | strike | at `w = 6e`: Picard / Newton | at `w = 16e`: Picard / Newton |
|---|---|---|---|
| 24 kHz | 12 cm | 20 / 26 (**0.77×**) | 3834 / 56 (**68.5×**) |
| 24 kHz | 8 cm | 44 / 32 (1.38×) | expansive / 86 |
| 24 kHz | 5 cm | 382 / 60 (6.37×) | expansive / 138 |
| 48 kHz | 12 cm | 12 / 22 (**0.55×**) | 30 / 28 (1.07×) |
| 48 kHz | 8 cm | 22 / 26 (**0.85×**) | 152 / 50 (3.04×) |
| 48 kHz | 5 cm | 44 / 40 (1.10×) | expansive / 66 |
| 48 kHz | 3 cm | 286 / 48 (5.96×) | expansive / 186 |
| 96 kHz | 12 cm | 10 / 14 (**0.71×**) | 16 / 24 (**0.67×**) |
| 96 kHz | 8 cm | 14 / 20 (**0.70×**) | 28 / 28 (1.00×) |
| 96 kHz | 5 cm | 22 / 26 (**0.85×**) | 82 / 42 (1.95×) |
| 96 kHz | 2 cm | 94 / 32 (2.94×) | expansive / 66 |

A ratio below 1 means **Picard is cheaper**, and that is most of the easy half of the map: a Newton
iteration carries its Krylov products, and where Picard converges in five sweeps there is nothing to
amortise them against. The 68× at the other end is the same solver on the same plate. So
`couple_method` is a real choice rather than a strictly better setting, and §11.8's "no default
moves" is now backed by a measurement instead of by caution.

### 13.4 The map has a second boundary, and §9.4's affordability claim does not transfer

§9.4 established that the wall is decided on step zero, and that is what made a 480-point grid
affordable. It is a claim about **Picard's divergence from a struck rest state**, and it is true.
It does not say that a Newton run which converges on step zero keeps converging, and it does not:

| fs | strike | step-0 edge | 300-step edge | lost | drift at the run edge | first bad step |
|---|---|---|---|---|---|---|
| 24 kHz | 12 cm | 40 | **32** | 8 | 2.83e-13 | 2 |
| 24 kHz | 8 cm | 32 | **20** | 12 | 3.54e-13 | 24 |
| 24 kHz | 5 cm | 18 | **16** | 2 | 1.04e-13 | 16 |
| 24 kHz | 3 cm | 8 | 8 | 0 | 2.29e-13 | — |
| 24 kHz | 2 cm | 4 | 4 | 0 | 1.65e-13 | — |
| 48 kHz | 12 cm | 40 | 40 | 0 | 4.17e-13 | — |
| 48 kHz | 8 cm | 40 | 40 | 0 | 2.32e-13 | — |
| 48 kHz | 5 cm | 32 | 32 | 0 | 1.42e-13 | — |
| 48 kHz | 3 cm | 18 | **16** | 2 | 1.31e-13 | 50 |
| 48 kHz | 2 cm | 10 | 10 | 0 | 1.61e-13 | — |
| 96 kHz | 12 cm | 40 | 40 | 0 | 4.06e-13 | — |
| 96 kHz | 8 cm | 40 | 40 | 0 | 5.61e-13 | — |
| 96 kHz | 5 cm | 40 | 40 | 0 | 1.63e-13 | — |
| 96 kHz | 3 cm | 32 | **28** | 4 | 1.59e-13 | 24 |
| 96 kHz | 2 cm | 20 | 20 | 0 | 1.19e-13 | — |

The 300-step edge is the largest `w/e` for which **every** step converged *and* the energy drift met
the project's `1e-10` bar. Ten cells hold; five lose between 2 and 12 in `w/e`, and the step at which
they lose it is 2, 16, 24, 24 and 50 — nowhere near step zero. The failures are loud (drift ~1e-1,
`n_iters` pinned at the cap), so nothing here is silent, but the honest boundary for a *sound* is
the second column and not the first.

**Generalisable, and it is the reason A2 was run at all:** "step zero has a root" and "three hundred
steps have roots" are different claims, and a one-step map answers only the first. §9.4's result
licenses the *cheap scan*; it does not license reading the scan as a statement about a run. When a
map is drawn from one step per point because a prior section made that affordable, check what that
section actually measured — here it was the other method's failure mode.

### 13.5 The outcome is not monotone in amplitude

Found in the free-edge slice and then walked out fully: a supported plate, 3 cm strike, struck
**0.12 m off-centre**, 48 kHz. One step from rest, Newton:

| `w/e` | Picard | Newton | Newton iters | Newton solves | Newton residual |
|---|---|---|---|---|---|
| 6 | converged | converged | 5 | 50 | 2.85e-16 |
| 8–14 | expansive | converged | 5–6 | 56–108 | ≤ 9.6e-14 |
| 16 | expansive | converged | 13 | 466 | 8.57e-16 |
| **18** | expansive | **expansive** | 50 | 17510 | 5.76e-02 |
| **20** | expansive | **converged** | 19 | 7266 | 6.75e-15 |
| 24–40 | expansive | capped / expansive | 50 | ~20000 | ≥ 2.2e-03 |

Newton fails at `w = 18e` and succeeds at `w = 20e`, to a residual of 7e-15. **A bisection on this
cell would have returned a boundary of 18 and been wrong, and it would have looked clean.** This is
the concrete instance of the hazard §2.4 predicted from `ρ`'s wandering, and it is the entire reason
the scan is a grid and the boundary is reported as two columns.

The same table carries the map's other quiet result: **the boundary is a cost ramp, not a cliff.**
Fifty solves at `w = 6e`, 466 at 16, 7266 at 20 — two orders of magnitude of Krylov work spent
before the verdict changes at all. A cost budget hits this ramp long before the convergence boundary
does, which matters more for a real instrument than the boundary's position does.

### 13.6 Trap 3, answered — and the answer is graded along curvature, not yes or no

Energy cannot settle this: *any* root of the discrete-gradient equation conserves exactly, so a run
at 1e-13 is what a root must do and is not evidence that the root resolves the physics. The other
half is a refined-`k` reference — `N = 20` fixed so only `k` moves, the same initial condition in
physical units, and all four rates compared at the same physical time.

Relative `L²` difference between successive rates, and their ratio (**4 is the second order the
scheme claims** — `plate.rs` derives `u^{-1} = u^0 - k v^0 + ½k²a^0`, "the consistent second-order
start"):

| fixture | 83 µs | 333 µs | 667 µs | 1.33 ms | 2 ms |
|---|---|---|---|---|---|
| **control** 12 cm, `w=6e` | 4.62 / 4.35 | 4.16 / 4.09 | 4.08 / 4.05 | 4.07 / 4.03 | 3.98 / 4.01 |
| **control** 8 cm, `w=6e` | 4.52 / 4.31 | 4.20 / 4.11 | 4.16 / 4.10 | 4.09 / 4.06 | 4.06 / 4.04 |
| 8 cm, `w=20e` (Newton-only) | 4.33 / 4.29 | 3.91 / 4.07 | 3.58 / 3.95 | 2.47 / 3.37 | 1.70 / 2.23 |
| 5 cm, `w=16e` (Newton-only) | 3.81 / 4.21 | 2.88 / 3.70 | 2.73 / 2.98 | 1.49 / 3.69 | 1.53 / 1.51 |
| 3 cm, `w=12e` (Newton-only) | 3.65 / 2.62 | 3.86 / 2.11 | 1.70 / 2.19 | 4.04 / 1.26 | 1.98 / 1.08 |

**The controls hold 4.0 at every checkpoint**, which is what makes the rest of the table readable:
the rig, the matched-time comparison and the fixed-`h` refinement are sound, and the degradation
below is not an artefact of any of them. (The spatial twin,
`tests/test_vk_stability.py::test_richardson_second_order`, warns that a narrow strike sits
pre-asymptotic at ratio ~3 — that warning is about refining `h` into a doubled wavenumber content,
and it does not bite a temporal refinement at fixed `h`.)

**And the answer is three different answers.** The 8 cm plate at `w = 20e` — territory Picard cannot
reach at all — converges at a clean second order out to 667 µs. That is trap 3 answered
affirmatively: there is a root, it is a plate, and Newton found it where Picard's map is expansive.
The 5 cm plate at `w = 16e` holds second order only at the first checkpoint. The 3 cm plate at
`w = 12e` never does, and its 48 kHz-to-96 kHz difference is already **37%** at 83 µs — that root
exists and both iterations agree on it to twelve digits, but 48 kHz is not resolving that plate, and
calling it audible would be exactly the error trap 3 names.

So: **the same curvature axis that sets Picard's wall also sets the resolution horizon, and Newton
moves only the first of them.** Newton buys real, physical territory at broad strikes and buys
*arithmetic* territory at narrow ones. A scene that wants a narrow strike still needs `fs`, and no
iteration substitutes for it. §3's refusal to claim an audio-band string-drivable gong is
untouched — and now it has a mechanism rather than a Picard failure behind it.

The independent half of the check: **Newton and Picard, both at 384 kHz, land on the same
trajectory** — 8.7e-15 to 7.9e-10 relative across all five fixtures and all six checkpoints, growing
with time exactly as two decorrelating trajectories should. The root is not an artefact of the
iteration that found it.

### 13.7 The free edge moves the same way, and is slightly harder

Struck 0.12 m off-centre, because §10.4 established that a centred Gaussian on a 40 cm plate is
`exp(−44)` at the rim and makes a free plate and a supported one the same interior problem. At
48 kHz: with an 8 cm strike both boundaries are identical to the supported case (Picard 20, Newton
censored at 40); with a 3 cm strike Picard is identical (6) and **Newton's edge is 16 free against
20 supported** — and the free plate shows no non-monotone recovery, failing from 18 upward without
the hole §13.5 found. The direction is the same and the free edge is marginally harder, which is
what trap 6's extra `h²` predicts and what Part 1's Jacobian already had to carry.

### 13.8 The line search still never fires — now over a population, and necessarily in Rust

`the_line_search_over_part_threes_population` in `crates/physsynth-core/tests/plate.rs` runs
fourteen points drawn from the map's Newton-only territory — three sample rates, five strike widths,
every one `expansive` under best-effort Picard — and records what the driver spent:

* **zero** Armijo halvings, across all fourteen;
* **zero** GMRES stalls (nothing reached `NEWTON_GMRES_MAX_PRODUCTS`);
* worst inner cost **38** Krylov products, worst **7** Newton iterations.

§11.6 answered trap 2 on the six *gate* fixtures and found the search first firing only from a seed
200× the physical one. This widens that from six comfortable points to fourteen hard ones and the
answer does not change: **the full Newton step is accepted everywhere the map says Newton works.**
The globalisation is insurance, and this is the population that says so.

**It had to be a native bar, and that is a structural fact rather than a preference.**
`n_line_search`, `gmres_products` and `gmres_stalls` live on `VkNewtonReport` and deliberately not
on `VkStep` — §11.4's "a field here costs two edits" — so no Python client can see them. A question
about a solver's internal accounting over a population can only be asked where the accounting lives.
A second bar, `the_free_edge_holds_the_same_way_off_centre`, asserts §13.7's direction with the same
counters.

### 13.9 Deliberately not done here

* **The boundary is censored above `w = 40e`** in six cells and the grid was not extended. At those
  amplitudes the strike is 4 cm of deflection on a 1 mm plate; the map's job was to locate the
  boundary in the musical range, and a floor is the honest report.
* **No second material and no second plate size.** The map is one 40 cm steel square. §2.2's
  side-shrinking leg is what says the curvature axis carries over, and re-deriving it here would
  have doubled the grid to confirm something already measured.
* **`n_line_search` is still not on `VkStep`.** §11.4's reasoning holds and §13.8 is the reason it
  can hold: the question it would answer is answerable in Rust.
* **Part 5's wrapper-tier work is untouched.** The gong in a room still runs `_VKPlateSurface`'s own
  Picard loop against the room-loaded factorization and still never consults `couple_method`
  (§11.8). Nothing in this part changes that, and Part 5 should still scope it in.
* **No default moves**, again: `couple_method` is `"picard"`, `couple_max_iter` is 50, and §13.3 is
  now the measured argument for the first of those rather than a precaution.

---

## 14. Part 5's result — the bound moved off the solver, and two of the three scenes were not scenes

Landed 2026-09-06. `M:\claud_projects\temp\vk-newton\part5_scene.py` and `part5_scene2.py`.
Measurement and prose; no code, no default moves.

### 14.1 Only one of the three compositions exists

§5 lists three scenes as "bounded by the iteration rather than by the physics", and
`scientific-hurdles.md` §5 repeats the list. One of them is blocked at the wrapper tier and §11.8
says why, once. **The other was never built.** `PyMalletMembrane`
(`crates/physsynth-py/src/mallet.rs`) casts its collaborator to a `PyMembrane` and refuses anything
else; there is no `MalletPlate`, no mallet-on-a-plate constructor and no test that composes the two.
The phrase came from `string-vk-plate-bridge-plan.md` §10.4, which *recommends* a mallet as the
exciter a future batch would need — a recommendation that became, by quotation, a list of existing
compositions.

That is Part 4's finding arriving twice in one batch: **a claim propagates by quotation, and a list
of scenes is worth a grep before it is worth a plan.** The check is one `#[pyclass]` signature.

The human's call, 2026-09-06: measure the one that runs, report the other two, and give the wrapper
work its own batch rather than smuggling it into a measurement part.

### 14.2 The ceiling — and the first grid read as a null result because it was censored

The scene is `make_vk_plate_bridge`: a 1 m string at 22.2 kHz terminated on a 40 cm, 0.1 mm steel
plate (`N = 16`), `K = 3000 N/m`, `boundary="supported"` the gong and `"free"` the cymbal. The
ceiling is the largest triangular pluck that runs 300 steps with **every** step converged and the
energy drift under the project's `1e-10` bar:

| boundary | Picard, shipped cap 50 | Picard, best effort (cap 2000) | Newton |
|---|---|---|---|
| supported (gong) | 30 mm | 70 mm | **≥ 300 mm** |
| free (cymbal) | 20 mm | 20 mm | **≥ 300 mm** |

Newton is at least **4.3× the gong's** best-effort Picard ceiling and at least **15× the cymbal's**,
and both Newton entries are censored — it never failed anywhere on the grid. Note the two Picard
columns: on the gong the shipped cap costs a factor of 2.3 and a *generous* cap recovers it, exactly
§2.3's point; on the cymbal both caps give 20 mm, so that half is genuine divergence and no constant
touches it.

**The first pass got this wrong, and the way it got it wrong is the lesson.** Its pluck grid stopped
at 50 mm, which censored three of six rows — and because best-effort Picard and Newton were *both*
censored on the gong, the table read as "the gong's entire gain is the sweep cap", a clean and
completely false null result. §13.2 had already invented a "censored" column for exactly this, and
the trap was walked into one section later anyway. **A censored grid is not a measurement, and a
censored grid that makes two rows equal is worse than one that makes them differ** — the first
invites a conclusion, the second invites another run.

### 14.3 There is no Newton tax on this scene

Worst back-substitutions in any step of a 300-step run:

| boundary | pluck | Picard cap 50 | Picard cap 2000 | Newton |
|---|---|---|---|---|
| supported | 1 mm | 20 | 20 | 20 |
| supported | 5 mm | 32 | 32 | 28 |
| supported | 15 mm | 50 | 50 | **38** |
| free | 15 mm | 52 | 52 | **40** |

Level at the bottom, cheaper at the top. §13.3 found Picard cheaper by up to 1.8× across the map's
easy half; that penalty does not appear here, because the plate's per-step work sits next to a
string's and a few extra Krylov products are lost in it. **A cost result from a bare-model map does
not transfer to a scene** — the model is one term in the budget there, not the whole of it.

And at the top of the extended grid Newton takes **four to six iterations**: 4 at a 20 mm pluck, 5
at 300 mm on the gong, 6 at 300 mm on the cymbal, where the plate is deflected to **154×** and
**546×** its own thickness respectively.

### 14.4 The new territory is resolved, and §13.6 would have predicted otherwise

Plate field at a matched physical time under joint refinement — `N_string` doubles so `fs` doubles
with it, while the plate's grid stays `N = 16` in all three runs so the fields compare directly.
Ratio of successive differences; 4 is the scheme's claimed second order:

| pluck | 1.35 ms | 2.70 ms | 5.40 ms | 13.50 ms |
|---|---|---|---|---|
| 1 mm | 4.05 | 4.03 | 3.85 | 3.96 |
| 15 mm | 3.83 | 3.70 | 3.50 | 1.15 |
| 50 mm | 4.34 | 3.84 | 5.73 | 1.06 |

(The 5.73 is a ratio taken between two nearly decorrelated differences and is noise, not
super-convergence; it is left in rather than smoothed. The same row's *differences* are non-monotone
 in time — 5.2e-03 at 2.70 ms against 2.36e-02 at 1.35 ms, an error that shrank as the run advanced,
which no convergence sequence does — and that is the same noise seen from the other side.) The small pluck holds second order for the
whole run. The two large ones hold it out to about 5 ms and then decorrelate — the same horizon
§13.6 found, and the same shape.

**§13.6 said the curvature axis that sets Picard's wall also sets the resolution horizon, and this
scene has the narrowest drive in the project — a point force at a single bridge node.** It resolves
anyway, and the reason is worth stating: §13.6's fixtures set the plate's *state* to a narrow
Gaussian, so the **deflection** was narrow. Here the deflection is whatever the plate's modes make
of a small per-step impulse, which is smooth. §2.2 said this in the first place — "the driver is the
strain, i.e. the curvature of the deflection" — and the restatement is that **the curvature axis is
about the deflection, not the drive.** That distinction matters because the drive is the thing a
reader can see in the code and the deflection is not.

A cross-check fell out unasked: the supported and free rows are **identical to four digits** at
1.35 ms and 2.70 ms and separate at 5.40 ms. That is §10.4's phenomenon appearing in a scene rather
than a fixture — until the response reaches the rim, a free plate and a supported one are the same
interior problem — and it dates this plate's rim arrival at roughly 5 ms.

### 14.5 What this scene is bounded by now

Not the iteration. At the top of the extended grid the gong reaches 154× its thickness and the
cymbal 546×, in four to six Newton iterations, conserving energy to the bar, with the trajectory
resolved for the first several milliseconds. **Von Kármán is a moderate-rotation theory**; a plate
deflected to five hundred times its thickness is far outside the range the model is derived for,
and it is out of that range long before the solver is in trouble. So §5's "bounded by the iteration
rather than by the physics" is **retired for this scene** — it is bounded by the physics now, which
is what the sentence was asking for.

Two things that does not say. The refinement check covers plucks to 50 mm, not to 300 — the
resolution claim stops where the measurement does. And this is one scene on one plate: the gong in a
room is still blocked (§11.8) and the mallet on the gong still does not exist (§14.1).

### 14.6 Deliberately not done here

* **No wrapper work**, by the human's call. `_VKPlateSurface.solve` still runs its own Picard loop
  against the room-loaded factorization; that is its own batch, and §11.8 is its statement.
* **No mallet-on-a-plate composition.** Building one is a model-composition batch with its own
  energy bars, not a line in a measurement part. It is now on the list as *missing* rather than as
  *bounded*, which is the correction §14.1 makes.
* **The ceiling is still censored**, at 300 mm. 30% of the string's length is not a pluck, and
  extending the grid further would measure the arithmetic rather than the instrument.
* **No default moves.** `couple_method` is `"picard"` everywhere, and every shipped number in the
  suite is the number it was.
