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
| 3 | NumPy's own transcendentals disagree with libm by an ulp on some CPUs — a read-out asserted exactly across languages fails on a runner and passes on another | `airbox.mode_frequency`, four `pow`s, one `tan`, `exp`, `cos`/`sin` | **Live** — the parity step has been red on `main` for four runs (§3) |
| 4 | The θ-scheme suppresses every discrete decay rate by `1/(1+θk²Q)`; "highs die faster" turns over past mode ~32 | `string_damped`, `string_stiff`, both plates | Accounted for; **fix derived, not built** (§4) |
| 5 | The von Kármán Picard iteration stops contracting at large amplitude / small `h` / high `fs` — the gong-on-a-string, the gong in a room and grid coarsening all die there | `plate.VKPlate`, `connection`, `airbox` | Open; **Newton proposed** (§5) |
| 6 | The geometrically exact string's Newton solve stops converging past `λ_long ≈ 4`, and `h`-refinement makes it worse | `string_geometric` | Warned at 1, unresolved regime (§6) |
| 7 | A point port's added mass is a grid quantity: refinement makes it *worse* | `airbox.RoomPort` | Refused, measured — `radius` has no default (§7) |
| 8 | At `λ = 1/√3` the room's corner mode is defective: broadband content grows linearly while the energy stays flat | `airbox` | Accounted for — the one place a flat energy is not a stability certificate (§8) |
| 9 | The conserved total is blind to a wrong coupling constant; `radiated == injected` is blind to half of them | every coupled scene | Accounted for — three detectors, jointly (§9) |
| 10 | Aliasing around every nonlinearity | bow, reed, mallet, VK plate, both nonlinear strings | Mitigated by oversampling; no anti-aliased scheme (§10) |
| 11 | Twenty ARPACK oracles were not bit-reproducible run to run | `analysis/modal.py` | Fixed in §24.9 — kept here as the pattern (§11) |
| 12 | Raw physics is not a musician's interface (parameter mapping) | none yet | Deferred (§12) |
| 13 | Which models can ever run in real time | engine, room | Deferred, and now decoupled from the language (§13) |
| 14 | `piston_radiation_resistance`'s `ka < 1e-8` series threshold is about three decades too small: just above it the direct form cancels catastrophically and is 544% wrong | `core/radiation.py` | **Live** — no caller is in the band; fix proposed, not applied (§14) |

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

## 3. NumPy's transcendentals are not libm — live on `main`

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

**Still open.** The CI job's own two failures are on a runner this session cannot read the log of
in full (the log blob is outside the proxy's allowlist); whether they are this test or a sibling
in the same class is to be confirmed on the next push. If they are a sibling, apply the same rule.

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

**Why it matters.** Frequency-dependent loss is the model's *audible* claim, and the scheme
silently caps it in the band where a piano's or a plate's partials are densest. It is also the
one artifact that a later calibration against recordings (HANDOFF §6.6, §12D) would fit *around*
rather than through.

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

**Where it should be built.** Under plan §6 (the frontier flipped at the end of Phase 2), new
physics is written in Rust first. This is a `Params` flag on `string_damped` and the two plates,
a bandwidth change in `banded.rs`, and one new oracle branch. It is the smallest physics change
with an audible payoff in the whole register and the human's call on whether to spend it.

## 5. Von Kármán Picard non-convergence — the deep end's wall

**What it is.** `VKPlate.step` solves the conservative coupled step by fixed-point (Picard)
iteration: predictor `2w^n − w^{n−1}`, then sweeps of "Airy solve → bracket → linear solve"
until the relative increment is below `couple_tol` (default 1e-13) or `couple_max_iter` (50)
runs out. Picard contracts only while the nonlinear coupling is small against the linear
operator, and the contraction factor scales like `k² · (amplitude/thickness)² / h⁴`, so it dies
three ways, all measured:

* **Amplitude.** `w ≈ 10e` at 96 kHz blows up (76k non-converged steps, overflow — "NOT a
  cascade", `docs/memory/von-karman-plate-state.md`) and converges only at 384 kHz.
* **Geometry.** Shrinking the plate breaks it too (`k²/h⁴`); an audio-band string-drivable
  Picard-convergent gong "cannot all hold at this sample rate" (`string-vk-bridge-state.md`).
* **The room.** Coarsening the air grid to buy affordability breaks the plate's fixed point,
  because the room sets `fs` and the plate's `k` with it (72 sweeps at 57.9 kHz, NaN at 33 kHz —
  `air-box-state.md` batch 6).

The bridge's exact linear guard is provably sufficient and the failure mode *migrates* past it to
non-convergence, "which a quadratic form can't see". §27.5 and §28.6 then found that the Picard
sweep count is the discriminator between the random-walk and the chaotic parity regimes.

**Why it is the deep end.** Gongs and cymbals are the payoff HANDOFF §2.2 chose the energy
framework *for*, and every composition that reaches them — the gong on a string, the gong in the
room, the mallet on the gong — is currently bounded by the iteration rather than by the physics.

**The approach: Newton on the discrete-gradient system, as model #10 already does.** The VK
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
3. **The measurement exists.** `n_iters`, `converged` and `last_residual` are public; the
   claim to make is a **convergence map** over `(w/e, fs, h)` — the regime boundary moves from
   Picard's to Newton's, and the plan document's job is to draw both.

**Costed.** A `couple_method="newton"` flag on `VkParams`, a matrix-free GMRES (~150 lines, no
dependency — the crate's allowlist is empty by policy), and a diagnostic script that draws the
two convergence maps. Rust-first under §6. This is the largest scientific unlock in the register
and the one the human has to prioritise against §4.

## 6. `λ_long` — the geometric string's unresolved regime

**What it is.** The longitudinal wave speed is `√(EA/ρ)`, 10–30× the transverse one, so the
"familiar `lam = 0.5`" silently means `λ_long ≈ 11`. The θ-scheme is unconditionally stable, so
nothing refuses it; what happens instead is that Newton stops converging past `λ_long ≈ 4`
(drift 1e+3 … 1e+5), and `h`-refinement makes it *worse* because `λ_long ∝ 1/h`. The constructor
warns at `LAM_LONG_WARN = 1.0`; `λ_long = 2` conserves to 1e-12, so a hard bar would forbid
working configurations (`docs/memory/geometric-string-state.md`, the human's call).

**Status.** Warned, not understood. The §1 fix removes one candidate cause (a Jacobian wrong in
its fifth digit steers Newton badly exactly when the longitudinal stiffness dominates); whether
the convergent window moves is a measurement to take — re-run the `λ_long` sweep the memory note
describes and record the new edge beside the old one. If the edge does not move, the mechanism
is the fixed point's, and §5's Newton–Krylov applies here too (the same discrete-gradient
structure). Either way the number belongs in `test_geometric_limits.py`, which today asserts the
warning and not the edge.

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

## 14. The baffled piston's series threshold is three decades too small — live, uncalled

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

**The fix, costed.** One line on each side. Move the threshold to where the series and the direct
form actually cross in accuracy. The series `(ka)²/2` is the first term of
`(ka)²/2 - (ka)⁴/12 + ...`, so its own relative error is `(ka)²/6`; the direct form's is about
`eps / ((ka)²/2) = 4.4e-16/(ka)²`. They meet at `(ka)⁴ = 2.7e-15`, i.e. `ka ≈ 7.2e-4`, where both
are around 8.6e-8. Taking `ka < 1e-3` for the series — or better, keeping two terms,
`(ka)²/2 - (ka)⁴/12`, and switching at `ka = 1e-2` — puts the worst error anywhere below 1e-11.

This is **not applied**, and applying it needs the human's call, because it changes a shipped number
for `ka` in `(1e-8, 1e-3)` — a band nothing currently reads, but a band the acceptance contract
nominally covers. It must land on both sides in one commit, as §1 and §2 did on 2026-09-02.
