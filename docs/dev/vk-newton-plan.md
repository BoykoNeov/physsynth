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

**Part 0 — the diagnosis, and the honest baseline.** *Lands first, on its own commit.* Today
`converged = false` conflates two different failures: "`ρ < 1` and the cap ran out" (fixable with a
number) and "`ρ > 1`" (not fixable at any cap). Expose the distinction — the model already holds
two consecutive residuals, so `ρ` is one division — and re-measure §2.1 / §2.3's tables against
best-effort Picard. **Do not change `couple_max_iter`'s default:** it is a public constructor
argument reached from `tests/helpers.py`, the airbox VK surfaces and the viewer payloads
(`VK_COUPLE_MAX_ITER`, `VKROOM_SWEEP_CAP`), and moving it silently moves sweep counts in runs other
machinery compares. *Gate:* the three cap-limited fixtures report "cap" and the three divergent ones
report "expansive", and the Picard baseline map is drawn.

**Part 1 — the Jacobian, asserted before it is used.** The Jacobian-vector product as a core
function, checked against a finite-difference directional derivative of `G` on several fixtures and
both boundaries. *Gate:* relative agreement at the finite-difference floor (~1e-7 with a
well-chosen step), and the linearity identity `J(d1+d2) = J d1 + J d2` exact to rounding.

**Part 2 — Newton–Krylov behind a flag.** `couple_method` on `VkSpec` / `VkParams`, default Picard.
Matrix-free GMRES plus the line search. *Gate:* on every fixture where Picard converges, Newton
reaches the same root to `couple_tol`, and the energy drift bar is met on the converged path.

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
report what changed. Including "nothing", if nothing.

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
