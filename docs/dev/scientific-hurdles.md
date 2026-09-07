# Scientific hurdles — the register

> **What this is.** Every open *scientific* problem in the project in one place: what is known,
> the evidence (with the file or plan section that measured it), and a costed approach where one
> exists. The migration's *porting* findings live in `docs/dev/rust-migration-findings.md`; this
> file is about the physics and the numerics, and it exists because the open problems were spread
> over twenty memory notes, fourteen plan documents and a handful of docstrings, each recorded
> "not fixed" in its own place and none listing the others.
>
> **How to read the status column.** *Fixed* — closed with code and a test. *Accounted for* — the
> oracle predicts it exactly, no test asserts it is gone. *Refused, measured* — a configuration
> the model declines at construction, on evidence. *Open* — a real gap with a proposed approach.
> *Deferred* — a stage this project has not reached (real-time, plugin).
>
> Started 2026-09-02. Add a row when a batch records a defect; move it, never delete it.

## 0. Summary table

| # | Hurdle | Where it lives | Status |
|---|--------|----------------|--------|
| 1 | DG Jacobian `(v,v)` block cancelled two `O(1)` terms at musical strain | `string_geometric` | **Fixed 2026-09-02** (§1) |
| 2 | Room energy books were a tolerance rather than exact across the port | `airbox` (Rust) | **Fixed 2026-09-02** (§2) |
| 3 | NumPy's own transcendentals disagree with libm by an ulp on some CPUs — a read-out asserted exactly across languages fails on a runner and passes on another | `airbox.mode_frequency`, four `pow`s, one `tan`, `exp`, `cos`/`sin` | **Symptom cleared 2026-09-03** — the parity step is green on five consecutive runs; the **rule** stays live (§3) |
| 4 | The θ-scheme suppresses every discrete decay rate by `1/(1+θk²Q)`; "highs die faster" turns over past mode ~32 | `string_damped`, `string_stiff`, both plates | Accounted for; fix derived, not built — and the **payoff claim is FALSIFIED 2026-09-06**: the rate suppression and the scheme's *pitch* flattening are one factor and its square root, so a mode's decay is never audibly wrong before that same mode is most of a semitone flat, and the turnover never lands closer than **330 cents** from in tune over 90 realistic configurations. The observations all stand; "an audible payoff in the whole register" does not (§4) |
| 5 | The von Kármán Picard iteration stops contracting at large amplitude / high strain / high `fs` — the gong-on-a-string, the gong in a room and grid coarsening all die there | `plate.VKPlate`, `connection`, `airbox` | Open, **narrowed 2026-09-06** — the `1/h⁴` mechanism is falsified, a third of the wall was the sweep cap, and Newton is **built** behind `couple_method` (default still Picard), the boundary is **mapped** (2–4× in amplitude; a root is a resolved *plate* only where the **deflection** is smooth), and the **gong on a string is no longer iteration-bound** — two of this row's three scenes were misfiled (one was never built), and the **wrapper block is gone 2026-09-06** — the room seam drives the model's own kernel against the loaded factorization, so the gong in a room runs under Newton too, and its wall moves ~1.6-1.9x in amplitude depending on the grid (§5) |
| 6 | The geometrically exact string's Newton solve stops converging past `λ_long ≈ 4`, and `h`-refinement makes it worse | `string_geometric` | Warned at 1, unresolved regime; **§1 eliminated as the cause 2026-09-03** (edge identical in 9/9 cells) and the threshold split into a **convergence** edge at 4 and an **energy** edge at 5–10 (§6) |
| 7 | A point port's added mass is a grid quantity: refinement makes it *worse* | `airbox.RoomPort` | Refused, measured — `radius` has no default (§7) |
| 8 | At `λ = 1/√3` the room's corner mode is defective: broadband content grows linearly while the energy stays flat | `airbox` | Accounted for — a flat energy is not a stability certificate here (nor in §6's under-resolved band, found 2026-09-03) (§8) |
| 9 | The conserved total is blind to a wrong coupling constant; `radiated == injected` is blind to half of them | every coupled scene | Accounted for — three detectors, jointly (§9) |
| 10 | Aliasing around every nonlinearity | bow, reed, mallet, VK plate, both nonlinear strings | Mitigated by oversampling; no anti-aliased scheme (§10) |
| 11 | Twenty ARPACK oracles were not bit-reproducible run to run | `analysis/modal.py` | Fixed in §24.9 — kept here as the pattern (§11) |
| 12 | Raw physics is not a musician's interface (parameter mapping) | none yet | Deferred (§12) |
| 13 | Which models can ever run in real time | engine, room | Deferred, and now decoupled from the language (§13) |
| 14 | `piston_radiation_resistance`'s `ka < 1e-8` series threshold was about three decades too small: just above it the direct form cancelled catastrophically and was 544% wrong | `core/radiation.py` | **Fixed 2026-09-03** — three Taylor terms below `ka = 3e-2`; worst error 5.24 → 6.7e-13 (§14) |
| 15 | No model states where its own answer stops being in tune, and the boundary is much lower than anyone assumed | every resonator | **Measured 2026-09-06** — two families with opposite signs (implicit errors compound into a hard floor at ~8% of the grid; explicit errors cancel at the magic Courant number), the plate is space-limited at every `fs` the suite uses, and the membrane's ceiling cancellation is **diagonal-only** (§15) |

---

## 1. The DG Jacobian's `(v,v)` block — fixed

**What it was.** `GeometricString._dg_jacobian` (and `geo::dg_jacobian`) assembled the
longitudinal-longitudinal entry of the Newton Jacobian as
`chi/2 − 1/2 + (1+v̄_x)(1+v_x⁺)/(2Λ̄²Λ⁺)`: two `O(1)` terms cancelling to an `O(strain²)`
remainder. The memory note recorded 7e-11 relative at strain 1e-3 with `q⁺ = q⁻`; measured on
2026-09-02 against a 60-digit reference with `q⁺ ≠ q⁻` (the case the solver actually sees) it is
worse and grows like `1/strain²`:

| strain | old spelling | new spelling |
|---|---|---|
| 0.1 | 3.9e-11 | 8.5e-16 |
| 1e-2 | 2.4e-9 | 8.2e-16 |
| 1e-3 | 1.2e-6 | 7.3e-16 |
| 1e-4 | 5.8e-5 | 9.7e-16 |

Musical strings sit at 1e-4 … 1e-3, so the Jacobian the Newton solve steered by was wrong in its
fifth digit exactly where the model is used.

**Why it was harmless, and why it was fixed anyway.** The residual defines the root; the Jacobian
only steers the iteration, so no trajectory or energy number depended on it. But §29's finding is
that this model's cost *is* its Newton iteration (a factorization per iteration per step), a
worse Jacobian is more iterations, and the cross-check test in `test_geometric_rotating_wave.py`
had to block at 1e-8 with a note saying "if this is ever reused where accuracy matters, this is
the note".

**The fix.** With `Λ̄ − 1 = mean(Λ − 1)` the three terms collapse exactly to
`((1+v̄_x)(1+v_x⁺)/Λ⁺ − Λ̄)/(2Λ̄²)`, and with `d = Λ − (1 + v_x)` (the stable third output of
`_stretch_terms`, `d̄ = mean(d)`) the numerator is exactly `−(d⁺(1+v̄_x) + d̄Λ⁺)/Λ⁺`, so

```
J_vv = −a (d⁺ (1 + v̄_x) + d̄ Λ⁺) / (2 Λ⁺ Λ̄²)
```

At `q⁺ = q⁻` this is `−a r²/(2Λ³)`, half the continuum Hessian, which is what the cross-check
asserts — now at 1e-12 like the other three blocks. Both languages spell it identically and the
parity file's `array_equal` on the Jacobian still holds. The general form is the one
`_stretch_terms` already stated: **any quantity that is `O(strainⁿ)` must be assembled from
quantities that are `O(strainⁿ)`**, and a Newton Jacobian is not exempt because it "only steers".

## 2. The room's energy books — fixed

**What it was.** Plan §30 declined to make `AirBox.acoustic_energy`, the per-face wall flux and
the port injection bit-identical across the port, because `np.sum` is pairwise-blocked above
eight terms and the smallest room has exactly eight nodes. §31 found the blocking is one fixed
algorithm and transcribed it (`crate::reduce`) for the ports, whose sums reach the update; the
books were left as a parked tightening for five batches (§31.11, §33.11, §34.10).

**The fix.** `reduce::sum_by` — the same blocking read through a closure, so a computed term
array costs no allocation — now carries all six sums. Measured: `acoustic_energy`, `dissipated`,
`injected` and `energy` are **equal** on all five wall types over 2,000 steps, and the port book
is equal at one node and at 123. The parity assertions moved from `<= 1e-13` to `==`. What that
buys is not a number but a detector: a mis-transcribed booking now fails by a bit instead of
hiding inside a tolerance that was there for a different reason.

## 3. NumPy's transcendentals are not libm — symptom cleared, rule live

**What it is.** Plan §22.1: NumPy computes `sin`, `cos`, `tan`, `exp`, `arcsin` and non-shortcut
`pow` with its own CPU-dispatched SIMD routines, chosen at import from the machine's feature set.
CPython's `math.*` and Rust's `f64::*` call the platform libm. So a value that passes through one
of NumPy's routines and is asserted **exactly** against Rust is a claim about the runner's CPU.

**Evidence.** The parity step of the `rust` CI job has been red on `main` for the last four runs
(2026-08-31 to 2026-09-01, 2 failed of 2,517). On this session's Linux x86-64 box,
`tests/test_rust_parity_airbox.py::test_the_exact_discrete_mode_is_bit_identical_and_so_is_its_frequency`
failed on **unchanged code** at mode `(3,2,2)`: `np.arcsin` and `math.asin` differ by one ulp
(0.4456879574695274 vs 0.44568795746952744) while `sin` agrees — §22.6's "per-function and
per-CPU" exactly.

**The fix taken here.** `AirBox._mu_squared` and `mode_frequency` take the portable spelling
(`math.sin`, `math.asin`, `math.sqrt`), §22.3's manoeuvre a seventh time. The rule is now stated
in one line: **a read-out asserted exactly across languages must not pass through
`np.<transcendental>`; a field quantity may, because it is compared by the physics bars.** The
remaining exposed surface §22.6 enumerated (four `pow`s, one `tan`, one `exp`, three `cos`/`sin`)
is bounded by ulp assertions rather than exact ones and stays as it is.

**Confirmed green 2026-09-03.** The question above — whether the runner's two failures were this
test or a sibling in the same class — is answered by five consecutive green runs of the `rust`
job's parity step (`33739866073`, `33740337663`, `33740757804`, `33747940441`, `33749384028`), and
`tests/test_rust_parity_airbox.py` is in that step's file list. So it was this test, the portable
spellings fixed it, and there is no sibling to chase.

**The rule stands even though the symptom is gone**, and that is the reason this section is not
marked fixed: nothing prevents the next batch from asserting a `np.<transcendental>` read-out
exactly across languages, and the failure would again appear only on a runner with the wrong CPU,
with no local repro (§22.2). The bounded exposure §22.6 enumerated — four `pow`s, one `tan`, one
`exp`, three `cos`/`sin` — is still bounded by ulp assertions rather than exact ones, deliberately.

## 4. The θ-scheme's rate suppression — accounted for, fix derived

**What it is.** Every implicit θ-scheme string and plate in this project discretizes loss as
`−2σ δ_t· u` (and `+2σ₁ δ_t· δ_xx u`), while the stiffness sits under the θ average. In the modal
domain, with `Q = c²p² + κ²p⁴` the eigenvalue of the linear operator, the per-step energy decay is
`g_m = (1 + θk²Q − σ_eff k)/(1 + θk²Q + σ_eff k)`, so the discrete rate is

```
Γ_m ≈ 2 σ_eff(m) / (1 + θ k² Q_m)        (continuum: 2 σ_eff(m))
```

The denominator grows like `p⁴` with stiffness, so "highs die faster" — the whole point of
model #3 — turns over past mode ~32 at `N = 128, κ = 2` (`docs/memory/damped-string-state.md`),
and the plate family inherits it with a fourth-power denominator. The oracle
`analysis/damping.py` predicts `g_m` exactly, the tests assert the rise only over `[1..16]`, and
the state is recorded as "symptom-cured, not fixed. No test asserts the artifact is gone."

**Why it matters — RETRACTED 2026-09-06, and the retraction is the useful part.** This paragraph
used to read: "Frequency-dependent loss is the model's *audible* claim, and the scheme silently caps
it in the band where a piano's or a plate's partials are densest. It is also the one artifact that a
later calibration against recordings (HANDOFF §6.6, §12D) would fit *around* rather than through."

Both sentences are wrong, for one reason. **The rate suppression and the θ-scheme's frequency
suppression are the same factor.** The lossless amplification factor gives
`sin²(ωk/2) = k²Q / (4(1 + θk²Q))`, so the discrete frequency carries `1/√(1 + θk²Q)` while the
decay rate carries `1/(1 + θk²Q)` — and in the unit a listener uses,

```
pitch error (cents) = 600 · log₂(S)          S = the decay-rate suppression 1/(1 + θk²Q)
```

verified to 0.57 cents over 108 string configurations wherever `S > 0.99`
(`docs/dev/theta-loss-compensation-plan.md` §2). A mode whose T60 comes out 10% long — about one
decay-time JND — is **82 cents flat**, some fifteen pitch JNDs. There is no register in which the
loss artifact is audible first, so it is not "the model's audible claim" and a calibration would be
fitting around the *pitch* error, which the compensation does not touch. The turnover is worse still:
swept over 90 configurations with `(σ₀, σ₁)` derived from real T60 targets, the closest it ever came
to being in tune was mode 26 at **330 cents flat**, a minor third (§2.4 there). The one knob that
fixes both halves is a smaller `k`.

The *observations* in this section all stand — the turnover exists, the rates are suppressed, the
plate inherits it worse. What was falsified is the payoff framing, the same distinction §5's probe
drew when it kept that section's observations and threw out its mechanism.

**The fix, derived (not built).** Pre-compensate the loss operator by the θ-denominator:

```
−2σ δ_t· u        →   −2 (σ₀ I − σ₁ D₂)(I + θ k² 𝓛) δ_t· u
```

In the modal domain `σ_eff → σ_eff (1 + θk²Q)`, and `g_m` becomes `(1 − σ_eff k)/(1 + σ_eff k)`
for **every** mode — the continuum rate up to the bilinear warp `(1/k) ln((1+σk)/(1−σk)) =
2σ(1 + σ²k²/3 + …)`, which is `O(k²)` and mode-independent. Properties, each checkable by the
existing rig:

* **Passivity is kept.** Under simply-supported boundaries `𝓛 = −c²D₂ + κ²D₂²` and `D₂` are
  polynomials in one SPD matrix, so `(σ₀I − σ₁D₂)(I + θk²𝓛)` is SPD and the loss power
  `−⟨δ_t· u, M δ_t· u⟩ ≤ 0` is a sum of squares — the same SBP argument as today, with `M` in
  place of `σI`. On a free plate the operators still commute (both built from the same free
  stiffness), so the argument transfers; on the *guitar outline* it does not automatically and
  must be checked (the masked operators need not commute).
* **The energy form is unchanged** — loss never enters `E^n`, only its rate — so `energy()`,
  every drift bar and every reduction anchor (`σ₁ = 0`, `EA = 0`, `EA = T`) survive as long as
  the compensation is behind a flag that defaults off.
* **The cost is bandwidth.** `A` gains a `D₂³` term: the string's pentadiagonal system becomes
  heptadiagonal (the banded Cholesky takes a bandwidth argument, so this is a constructor change,
  not a solver change); the plate's `splu` does not care.
* **The oracle already exists.** `discrete_damped_mode_decay` returns `g_m` for the uncompensated
  scheme; the compensated one is the same function with `σ_eff (1 + θk²Q)` in place of `σ_eff`,
  and the test that pins the per-mode rate to 5e-4 pins the fix the same way. The test that
  *cannot* exist today — "the rate rises monotonically over the whole resolved band" — becomes
  writable.

**Where it should be built, if it is.** Under plan §6 (the frontier flipped at the end of Phase 2),
new physics is written in Rust first. This is a `Params` flag on `string_damped` and the two plates,
a bandwidth change in `banded.rs` (`kd` 2 → 3, and `apply_ainv` has three external consumers), and
one new oracle branch. `docs/dev/theta-loss-compensation-plan.md` §4 scopes it in five parts.

**It is not "the smallest physics change with an audible payoff in the whole register"** — that
sentence stood here until 2026-09-06 and is retracted above. What a build would still buy is stated
honestly in that plan's §3: the rate's error constant improves by about seven orders of magnitude,
`loss_coefficients_from_T60` starts delivering what it promises (today 2.75% long at its own second
target mode, 19% by mode 40), the coarse-timestep plate's tail stops running 7.4× long, and the
monotone-rate test this section says cannot exist becomes writable. **The human's call, made
2026-09-06: record the finding and do not build it**, and open the wider thread the probe exposed —
that the *pitch* error is what actually bounds every accuracy claim in this project, and no model has
its resolution horizon written down.

## 5. Von Kármán Picard non-convergence — the deep end's wall

**What it is.** `VKPlate.step` solves the conservative coupled step by fixed-point (Picard)
iteration: predictor `2w^n − w^{n−1}`, then sweeps of "Airy solve → bracket → linear solve"
until the relative increment is below `couple_tol` (default 1e-13) or `couple_max_iter` (50)
runs out. Picard contracts only while the nonlinear coupling is small against the linear
operator, and the contraction factor grows with `k²` and with the **strain** — the curvature of the
deflection, of which `(amplitude/thickness)²` is only one factor. It dies three ways, all measured:

* **Amplitude.** `w ≈ 10e` at 96 kHz blows up (76k non-converged steps, overflow — "NOT a
  cascade", `docs/memory/von-karman-plate-state.md`) and converges only at 384 kHz.
* **Geometry.** Shrinking the plate breaks it too; an audio-band string-drivable
  Picard-convergent gong "cannot all hold at this sample rate" (`string-vk-bridge-state.md`).
* **The room.** Coarsening the air grid to buy affordability breaks the plate's fixed point,
  because the room sets `fs` and the plate's `k` with it (72 sweeps at 57.9 kHz, NaN at 33 kHz —
  `air-box-state.md` batch 6).

**Corrected 2026-09-06 — this section said `k² · (amplitude/thickness)² / h⁴`, and the `h` half was
wrong.** `docs/dev/vk-newton-plan.md` §2.2 measured all three legs separately. Refining the grid
**4.3×** at fixed plate size *and fixed absolute strike width* costs two sweeps and then flattens
(12 → 22 max sweeps from `N = 12` to `N = 52`). Shrinking the plate at **fixed `h`** hits the cap
(11 → 50 sweeps from 40 cm to 16 cm). And narrowing the *strike* alone — fixed plate, fixed grid,
fixed peak amplitude — does the same (7 → 11 → 22 → 50 for a 12 / 8 / 5 / 3 cm Gaussian). Grid
refinement is therefore nearly free, and the two observations filed above under "geometry" are one
observation about **curvature**. The observations stand; the attributed mechanism did not.

**And two measurements narrow the wall itself.** A third of it is the 50-sweep **cap** rather than
divergence: of the six failing fixtures, three come back green on the energy bar at a generous cap
and at essentially no wall-clock cost, because the expensive steps are rare — 143, 724 and 76
sweeps, drifts 4.1e-13, 4.3e-13, 6.0e-13 (§2.3, re-drawn as a per-step verdict in §9.3). Only three
are genuine divergence, and in those `ρ` is not a constant: it climbs *through* 1 mid-solve, so the
recoverable failures are the **stalling** ones, which is exactly what a quadratically convergent
step is for. The wall is also decided on step **zero** — every divergent fixture is already
expansive on its first step from rest, at both caps (§9.4) — which is what makes a convergence map
affordable from one step per point rather than a 300-step run.

The bridge's exact linear guard is provably sufficient and the failure mode *migrates* past it to
non-convergence, "which a quadratic form can't see". §27.5 and §28.6 then found that the Picard
sweep count is the discriminator between the random-walk and the chaotic parity regimes.

**Why it is the deep end.** Gongs and cymbals are the payoff HANDOFF §2.2 chose the energy
framework *for*. This section used to say that "every composition that reaches them — the gong on a
string, the gong in the room, the mallet on the gong — is currently bounded by the iteration rather
than by the physics", and **two thirds of that list was wrong** (`vk-newton-plan.md` §14.1): the
gong in a room is blocked at the wrapper tier rather than by the iteration, and **the mallet on the
gong was never built at all** — `MalletMembrane` casts its collaborator to a `Membrane`, and the
phrase came from a plan that *recommended* a mallet for a future batch. The gong on a string was
real, was bounded by the iteration, and **is not any more** (§14.5): under Newton it runs to a pluck
4.3–15× past best-effort Picard's ceiling, deflecting the plate to 150–550× its thickness in four
to six iterations — far outside von Kármán's moderate-rotation range, so the binding constraint on
that scene is now the model rather than the solver.

**The approach, now built: Newton on the discrete-gradient system, as model #10 already does.**
`couple_method="newton"` ships behind a flag as of 2026-09-06 (`vk-newton-plan.md` Parts 1–2), with
Picard still the default; on the three fixtures §9.4 classifies as expansive-on-step-zero it
converges in 4–6 iterations with 300-step energy drifts of 6.3e-13 to 1.4e-12 (§11.7). That is a
property of the *iteration*, not yet a claim about the territory: the convergence map and the
refined-`k` reference are Part 3, and until they land the honest reading is "the solver got there",
not "the plate is audible". The VK
step is a nonlinear system `G(w^{n+1}) = 0` whose residual is exactly what the Picard loop
evaluates; Newton on it converges quadratically wherever Picard converges linearly and keeps
converging where Picard's factor exceeds one. Three facts make it cheaper than it looks:

1. **The Jacobian is available in closed form.** `∂/∂w` of the bracket term is
   `L(·, F̄) + L(w̄, ∂F/∂w ·)` with `∂F/∂w` the Airy solve applied to `L(w̄, ·)` — a product of
   sparse operators and one factorization the model already holds. Assembling it explicitly is
   dense-ish (the Airy inverse is dense); the right form is **Newton–Krylov**: a matrix-free
   Jacobian-vector product (two brackets and one Airy solve per product), GMRES preconditioned by
   the linear plate's existing `splu` (which is the exact Jacobian at zero amplitude).
2. **Energy conservation is a property of the root, not of the iteration.** Any root of the
   discrete-gradient equation conserves exactly (§29's corollary: "any root … conserves
   exactly"), so switching the iteration changes no bar, and the Picard loop can stay as the
   fallback and as the parity reference.
3. **The measurement exists.** `n_iters`, `converged` and `last_residual` are public — verified,
   with getters and setters at `crates/physsynth-py/src/plate.rs`. Since 2026-09-06 the model also
   reports *which* failure it had (`capped` versus `expansive`), because "did not converge"
   conflated a number with a wall (§9.1). The claim to make is a **convergence map** over
   `(w/e, strike curvature, fs)` — note `h` is not an axis, per the correction above — drawn twice,
   once for best-effort Picard and once for Newton.

**Status.** The flag, the closed-form Jacobian-vector product, the matrix-free GMRES and the Armijo
line search are built and asserted, and the map is drawn (Parts 0–3; no new dependency, the crate's
allowlist is still empty). **What the map says**, in `vk-newton-plan.md` §13: Newton's amplitude
boundary is two to four times Picard's wherever the comparison is not censored at the top of the
grid; Picard is *cheaper* at easy points (0.55–0.85×) and up to 68× more expensive at hard ones,
which is the measured argument for leaving it the default; the boundary is **not monotone** in
amplitude in at least one cell; a 300-step run holds a lower boundary than a single step does in
five cells of fifteen; and trap 3's answer is **graded** — at a broad strike a Newton root refines
at the scheme's claimed second order and is a plate, while at a 3 cm strike the 48 kHz solution is
37% away from the 96 kHz one at 83 µs, so the root is real arithmetic and not a resolved plate. **The
curvature axis that sets Picard's wall also sets the resolution horizon, and Newton moves only the
first**, which is why §3's refusal of an audio-band string-drivable gong stands — now with a
mechanism behind it rather than an iteration failure.

**The wrapper-tier work is done (2026-09-06, `docs/dev/air-box-vk-newton-plan.md`).**
`_VKPlateSurface.solve` no longer carries its own transcription of the Picard sweep: the core's
coupled step now reaches its theta-scheme operator through a trait, so the seam points that step at
the loaded factorization and gets `couple_method`, `residual_ratio` and `n_solves` with it. Measured
on the room scene, Picard blows up on step zero from `w = 4.5e` and Newton runs clean to `6e` at
N=20 (to `10e` on the suite's own N=8 fixture) with an energy drift of 1e-13. Two things came out of
that measurement and neither was predicted: the air load does **not** move the wall — room and bare
transition cell for cell, so this was never the room's fault — and the amplitude gain is **1.6-1.9x
rather than §5's 2-4x**, because the boundary moves with sample rate *and* with resolution. There is
no single payoff number for this hurdle; there is a fixture and a measurement.

**The mallet composes with it too (2026-09-06, `docs/dev/mallet-vk-room-plan.md`).** `MalletVKPlate`
now takes a room wrapper, reaching the loaded operator through a trial-solver closure in
`vk_plate_step`; the room half of the step is assembled once and the port is injected once however
long the outer chord runs. Three measurements came out of it and two of them move this row:

* **The air load does not move the mallet's wall either.** Room and bare die at the same strike
  velocity with the same number of non-converged steps, and `n_solves` agrees within 0.5% at every
  amplitude. That is the second independent confirmation of the same fact.
* **A mallet reaches much further than a displacement strike on the same grid** — `w/e = 7.8`
  against the initial-condition strike's 4.5, because a mallet builds its amplitude over hundreds
  of steps while an IC arrives with all of it at once. The wall is a property of how the amplitude
  is *delivered*, not only of how large it is.
* **Newton is cheaper below the wall at 8 kHz**, crossing one at `w/e ≈ 4.3` and settling at
  **0.71x** — which contradicts `mallet-gong-plan.md` §10's "Newton buys nothing here" and dates
  rather than overturns it: that measurement is at 48 kHz, where the mallet never reaches the wall.

What is left is the *scientific* half: Newton has its own wall (between `6e` and `9e` at N=20), and
it arrives there by becoming unaffordable rather than by diverging. This remains the largest
scientific unlock in the register and the one the human has to prioritise against §4.

## 6. `λ_long` — the geometric string's unresolved regime

**What it is.** The longitudinal wave speed is `√(EA/ρ)`, 10–30× the transverse one, so the
"familiar `lam = 0.5`" silently means `λ_long ≈ 11`. The θ-scheme is unconditionally stable, so
nothing refuses it; what happens instead is that Newton stops converging past `λ_long ≈ 4`
(drift 1e+3 … 1e+5), and `h`-refinement makes it *worse* because `λ_long ∝ 1/h`. The constructor
warns at `LAM_LONG_WARN = 1.0`; `λ_long = 2` conserves to 1e-12, so a hard bar would forbid
working configurations (`docs/memory/geometric-string-state.md`, the human's call).

**Status.** Warned, not understood — and as of 2026-09-03 the §1 candidate cause is **eliminated
by measurement**, so the mechanism is the fixed point's and §5's Newton–Krylov applies here too.

**The measurement (2026-09-03).** `scripts/sweep_geometric_lam_long.py` runs both Jacobian
spellings — the current one and the pre-§1 expression, carried verbatim in the script as
`OldJacobianString` — on identical fixtures in one process, over nine `(N, amplitude, IC)` cells.
Comparing against the old table instead would have attributed to the Jacobian whatever else drifted
in the rig over the intervening months.

**The edge did not move: 9 cells of 9, identical.** Not "close" — the same grid point.

**Why it could not have moved, which is the useful half.** The old Jacobian's relative error grows
like `1/strain²` (§1's table: 3.9e-11 at strain 0.1, 5.8e-5 at 1e-4), so it is worst where the
string is *quietest*. The Newton solve's difficulty grows with the nonlinearity, so it is worst
where the string is *loudest*. The amplitude sweep shows the two regions are disjoint: at strain
1.9e-4 — where the old spelling was ~5e-5 wrong — both spellings converge in 1–2 iterations with
zero stalls at every `λ_long` up to 10; at strain 3.8e-2 — where the stalls and the blow-up live —
the old spelling was already ~1e-11 right. In the narrow overlap (strain ~9e-3) the corrected
Jacobian does trim cap-hits, 14–16 against 18–20 per 300 steps, but the sign is not systematic:
across the nine cells the first-stall `λ_long` differs in two, once in each direction. **A fix to
an accuracy that is worst at small amplitude cannot rescue a convergence failure that only exists
at large amplitude.**

**What the measurement did find — there are two edges, and the old table conflated them.** Reading
the Newton iteration counter beside the drift (§1's own argument says drift alone cannot see a
Jacobian change) separates:

| | `λ_long` | what happens |
|---|---|---|
| **Convergence edge** | **4** (7 of 9 cells) | steps begin to exhaust `newton_maxiter` |
| **Energy edge** | **5–10**, case-dependent | drift breaks the 1e-10 gate, runs to 1e+5 |

The old table's `λ_long = 4` row was right about *convergence* and its `1e+3 … 1e+5` drift belonged
to a different, higher threshold. Between the two the solve stalls on up to a fifth of its steps
and the energy still conserves to ~1e-15 — better, in fact, than a well-resolved long run, because
there are fewer steps to accumulate round-off over. **This is the second place in the project where
a flat energy is not a stability certificate** (§8 is the first), and unlike §8 it is blind in the
safe-looking direction. `LAM_LONG_WARN` stays at 1.0 — the human's call, and its 4× margin is now
explicitly margin against the *convergence* edge.

Pinned by `tests/test_geometric_energy.py::test_a_flat_energy_is_not_a_convergence_certificate_in_the_under_resolved_band`,
which asserts both halves at `λ_long = 6` (stalls fire, gate passes) against a `λ_long = 2` control,
and names the sweep script as the thing to re-run if the edge ever moves. The expensive nine-cell
sweep stays in the script: the suite is bulk-bound and a run that blows up on purpose is not a cheap
test.

## 7. The point port does not converge — refused, measured

A `radius=None` port is a sphere of radius `≈ h/3.1`, so refining the grid halves its equivalent
radius and doubles the added mass it hangs on the body; a fixed-radius ball barely moves (0.493,
0.496 against 1.045, 1.038 over two halvings). The energy identity is exact either way; the
*magnitude* of a point port's load is a grid quantity. Settled by making `radius` mandatory and
the point port an explicit opt-in (`airbox.py`, `RoomPort`), and the spread port's shape factor
confirmed to 0.3% (the 6/5 uniformly-injecting ball). Nothing to do but not to forget it: any
future "coarsen the room" work reopens it.

## 8. The defective corner mode at the 3-D CFL ceiling — accounted for

At `λ = 1/√3` the corner mode's dispersion argument reaches exactly 1 (`ω_d k = π`), the
leapfrog's amplification matrix is defective, and broadband content grows **linearly** while the
energy stays flat — the one place in this repo where a flat energy is not a stability certificate
(`airbox.py` module docstring). The same ceiling is the only dispersionless direction (the grid
diagonal). Accounted for by construction: the room is run a hair below the ceiling, and the modal
test asserts the exact frequency law. Worth a line here because it is the counterexample to the
project's own first rule ("energy is the primary bug detector") and every new 3-D scheme has to
be checked against it.

## 9. Detector blind spots — three detectors, jointly

Four air-box batches found four blind spots (`docs/memory/air-box-state.md`):

* the **conserved total** telescopes against whatever pressure each side used, so it is flat while
  the coupling constant is wrong (drift 4.9e-15 with the wall-closure factor dropped, *smaller*
  than the correct run's);
* **`radiated == injected`** catches that and misses a `1×` in place of the dipole's `2` when it
  is consistent on both sides;
* the **coupled residual** at two timesteps against the room's own post-closure pressure jump
  catches both;
* and a `drive_index` differing between two runs fools all three (`string-vk-room-chain-state.md`).

The standing rule is that no single one is sufficient and a scene test carries all three. Kept
here because it is the project's most important negative result about its own method.

## 10. Aliasing around nonlinearities — mitigated, not solved

Every nonlinear element (bow friction, reed, mallet contact, the fret barrier, the two nonlinear
strings, the VK plate) folds high-frequency content down as garbage. The project's answer is
oversampling (HANDOFF §8, `plate.py:964`, `mallet.py:195`'s "raise fs" refusal, the bore's
oversampled rate for the reed). That is the correct accuracy-first answer and the expensive one;
an anti-aliased formulation (band-limited nonlinearities, or the energy-based schemes' own
implicit averaging used as a filter) is §12C territory and is not proposed here. It interacts
with §5: raising `fs` for aliasing is what breaks Picard, so the two hurdles push in opposite
directions on the same knob.

## 11. Oracle reproducibility — fixed, keep the pattern

Plan §7's "four non-reproducible oracles" were twenty: every `eigsh` call without a start vector
wobbled in its last digits, and the ones returning *eigenvectors* fed to `set_state` made two
exactly-degenerate rigid modes come back 1e-1 apart run to run (§24.9). Fixed by pinning `v0`.
The pattern to keep: **an oracle that is not bit-reproducible reads as a cross-language
discrepancy the first time someone tightens a comparison**, so every randomised or iterative
oracle takes a seed or a start vector, and every new one is checked by running it twice.

## 12. Parameter mapping — deferred

Raw physics (Young's modulus, tension, `σ₀/σ₁`, `EA`) is not a musician's interface. The
`loss_coefficients_from_T60` helper is the only mapping in the tree. HANDOFF §8 budgets "real
work" for it and §12D proposes learning it; nothing is scheduled. It becomes urgent at the viewer
sandbox (§12G) and at the plugin, not before.

## 13. Which models are real-time — deferred, decoupled

The room's cost runs as `h⁻⁴` and it sets the sample rate of everything coupled to it; it will
not be real-time in any language. The migration decoupled this from the language decision (plan
§8) but did not answer it. The measurements that will: §29's 15.5× and §34's 22× on the geometric
string's step, §28's 2.99× on the VK step, the room's ~1.1–1.5× above 4,000 nodes (§30.11). The
answer is a table of per-model per-step costs at their shipped fixtures, which the parity files
already print and nothing collects.

---

## 14. The baffled piston's series threshold was three decades too small — fixed

**What it is.** `physsynth/core/radiation.py`'s `piston_radiation_resistance` computes Rayleigh's
result

    R_a(ka) = (rho0 c0 / S) [1 - J1(2ka)/(ka)],    S = pi a^2,   k = omega/c0

and the bracket is a genuine `0/0` as `ka -> 0`, since `J1(2ka)/ka -> 1`. The function guards it
with a series branch:

```python
bracket = 0.5 * ka * ka if ka < 1e-8 else 1.0 - j1(2.0 * ka) / ka
```

The guard is in the right place and the threshold is in the wrong one. The bracket's true value is
`(ka)²/2`, so the subtraction `1 - J1(2ka)/ka` is removing two numbers that agree to about
`-log10((ka)²/2)` digits. At `ka = 1e-8` that is sixteen digits — the entire mantissa — and the
answer is noise.

**Measured (2026-09-03), with SciPy's own `j1` doing the work, against the exact series:**

| `ka` | relative error of the shipped direct branch |
|---|---|
| 1e-8 (just above the threshold) | **5.4** — i.e. 544% |
| 1e-7 | 2.3e-2 |
| 1e-6 | 3.1e-4 |
| 1e-5 | 8.3e-8 |
| 1e-4 | 6.1e-9 |

So the function returns a number with no correct digits for `ka` just above its own cutoff, and
does not reach `1e-6` accuracy until about `ka = 1e-5`.

**Why it has never shown.** Nothing calls it in the band. The two call sites the suite makes are at
`ka = 9.2e-5` (`test_radiation.py`'s Rayleigh-limit test, which asks for `rel = 1e-6` and gets
6e-9) and `ka = 1.83` (its Bessel-formula test at `rel = 1e-12`). The bore's bell sits around
`ka = 2e-2`. A caller *would* land in the band for a small radiator at a low frequency — a 1 mm
source below 1 Hz, say — which is not a musical configuration, and that is the whole reason this is
a register entry rather than a bug report.

**How it was found.** Phase 7 batch 2 wrote a Rust `J1` and a native bar asserted that the two
branches meet at the threshold. They do not, by a factor of six, and the disagreement was in the
*Python* all along: two implementations of `J1` differing in their last bits disagree by 300% after
the cancellation eats sixteen digits. The port reproduces the threshold rather than moving it,
because changing a shipped physics number inside a porting batch is not a port
(`crates/physsynth-analysis/src/radiation.rs` and the Python footer both say so), and
`tests/test_rust_parity_analysis.py::test_the_pistons_cancellation_band_is_reported_and_no_caller_is_in_it`
measures the band so a future caller arriving in it fails there.

**The fix, as applied 2026-09-03 (the human's call).** One expression on each side, in one commit,
as §1 and §2 got. Three terms of the bracket's own Taylor series in Horner form, below `ka = 3e-2`:

```python
ka2 = ka * ka
bracket = (ka2 * (0.5 - ka2 * (1.0 / 12.0 - ka2 / 144.0))
           if ka < PISTON_SERIES_CUTOFF_KA else 1.0 - j1(2.0 * ka) / ka)
```

**Measured against a 60-digit `mpmath` reference over `ka ∈ [1e-10, 10]`**, worst relative error of
the whole function:

| variant | worst |
|---|---|
| shipped (one term, `ka < 1e-8`) | **5.24** |
| one term, `ka < 2e-4` (its optimum) | 1.3e-8 |
| two terms, `ka < 5e-3` | 9.2e-12 |
| **three terms, `ka < 3e-2` (shipped)** | **6.7e-13** |

An improvement of **7.8e12×**, and the branches agree to 7e-13 across the seam, so the function has
no step in it.

**The correction worth keeping, and it is why this section is longer than the fix.** The threshold
above was *measured*; the estimate written into the first draft of this section was **wrong**. That
draft derived the one-term crossover algebraically as `ka ≈ 7.2e-4` giving 8.6e-8; the measured
optimum is `2e-4` giving 1.3e-8 — off by 3.6× in the threshold and 6.6× in the error. And the
shipped choice is not at a crossover at all: `3e-2` sits deliberately *past* the direct form's own
noisy region (~5e-12 around `ka = 3e-3 .. 1e-2`), which is what buys the last order of magnitude
over the two-term option. Plan §36.2's "measure the margin before you claim it" applies to fixes,
not only to ports.

**What the fix does and does not make exact across the two languages**, also measured rather than
assumed — the first draft of the code comment claimed the whole function was now bit-identical, and
it is not:

* **Below the cutoff: bit-identical, 0 of 3,000 sampled values differ.** The series is `+ - * /`
  only, so IEEE-754 pins it, and `tests/test_rust_parity_analysis.py` asserts equality there.
* **Above it: 1,444 of 3,000 differ, worst 9.8e-13.** The direct form runs through two different
  `J1` implementations (Cephes vs a Miller recurrence) and never can be exact. That 9.8e-13 is far
  larger than the ~1e-16 the two `J1`s differ by, and the factor is the threshold's whole
  justification: at `ka = 3e-2` the bracket is 4.5e-4, so the subtraction still amplifies a last bit
  about 2,200×. Lower the cutoff and more of the domain goes to a branch that magnifies
  disagreement; raise it and more goes to a truncated series.

**What moved for existing callers.** Nothing that any bar can see. The bore's bell (`ka = 2.5e-2`)
moves by 4.1e-14; `test_radiation.py`'s Rayleigh-limit fixture (`ka = 9.2e-5`) moves by 6.9e-9,
which is the error it was previously carrying, *toward* the truth and against a `rel = 1e-6` bar;
the Bessel-formula fixture (`ka = 1.83`) is on the direct branch and does not move at all.

**The two tests that asserted the defect are replaced, not deleted.**
`crates/physsynth-analysis/tests/oracles.rs` now asserts that the branches *meet* at the seam (the
property a future threshold edit would break, and which no physics bar in this project could see),
plus that the series really is the bracket's Taylor expansion term by term — checked against the
expansion written out flat rather than in Horner form, so a transposed coefficient shows and a
re-association does not. The parity file asserts the exact/tolerant split above.

## 15. The resolution horizon — measured 2026-09-06, and it is small

**What it is.** Every accuracy claim this project makes about a mode is bounded by whether that
mode's *frequency* is right, and until 2026-09-06 no model had that boundary written down. §4's
probe forced the question: the θ-scheme's decay error is locked to its pitch error, so the pitch
error is what actually binds. `docs/dev/resolution-horizon-plan.md` is the measurement,
`tests/test_resolution_horizon.py` the 32 tests that keep it honest, and
`tests/helpers.py::pitch_horizon` the primitive.

**The structural finding, and the first draft of it was wrong.** The horizon is **not**
`min(time floor, space floor)`. Whether the two errors add or cancel depends on their signs, and
the two scheme families here have opposite answers. The implicit θ-scheme's time factor
`1/√(1 + θk²Q)` and the spatial `sinc(u)` are **both flat**, so they compound and there is a hard
floor no sample rate passes. The explicit leapfrog's time factor is **sharp**, so it cancels the
spatial droop exactly at `λ = 1`. **"Refine the timestep" is correct advice for one family and
actively wrong for the other.**

**The numbers that should change what somebody does.**

* The **space floor has a closed form** and no `k`, `c`, `L` or `fs` in it: `sin(u)/u =
  2^(−cents/1200)`, `m*/N = 2u/π`. At 5 cents that is **8.38% of the grid** for a pure wave, and
  stiffness lowers it (6.6% at `κ=2`, 5.9% at `κ=8`, and it keeps falling under refinement).
* The canonical `λ = 1` damped string at `N = 256` resolves **eleven partials**. A 64× sample-rate
  increase takes that to nineteen and then stops dead.
* The **plate is space-limited at every sample rate the suite uses** — a 40× change of `fs` moves
  its horizon by at most one mode. At the `N = 16` fixture the fundamental is **9.6 cents flat**
  and the (2,2) mode is 83 cents flat.
* The **membrane's CFL-ceiling cancellation is diagonal-only.** At `λ = 1/√2` the diagonal modes
  are exact (127 of 127) and the axial modes are not (15 of 127). Reading only the diagonal would
  have produced "the membrane is in tune at the ceiling", a claim about one mode family.

**Status.** Measured and asserted for the string family, the rectangular membrane, the supported
plate and the beam. Four rows of the plan's inventory are explicitly **not** done and say why: the
staircased domains (circular membrane, guitar outline) fail a cents comparison because the reference
is a different *shape*; the free plate's reference is a table rather than a formula; the nonlinear
family has no linear modal oracle and needs a refinement horizon instead (one point already exists
in `vk-newton-plan.md` §13); the bore's question is the area function's resolution. The named
follow-on — converting the suites' hand-picked assertion bands into derived ones — was **done
2026-09-07** and mostly answered *no*: of thirteen hand-picked bands, **two** are bounded by pitch
and were derived (the stiff string's 10 became 48, the beam's 1 and 4 became 2 and 6); five compare
against the scheme's **own** discrete oracle so the horizon is irrelevant by construction; three sit
at `λ = 1` where the explicit family is **exact**; one is a staircased domain; and the band §6 named
as the example is a **decay-rate** band whose limiter is a turnover at `~m=32`, where deriving it
would have shrunk the range. Every literal turned out conservative, none over-claiming. The one
genuine candidate left is the 2-D plate's, blocked on splitting a mixed mode family
(`resolution-horizon-plan.md` §7).
