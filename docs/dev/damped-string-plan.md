# Damped String — Phase 2 Plan (model #3, frequency-dependent loss)

> **Status: PLANNED → IMPLEMENTED (2026-06-21).** `core/string_damped.py`
> (`DampedStiffString`, implicit θ-scheme + a frequency-dependent loss term), oracle in
> `analysis/damping.py`, suite in `tests/test_damped_string.py`, viz in
> `viz/plots.py::plot_decay_rates` + `scripts/diagnose_damped_string.py`. Builds directly on
> [model #2](stiff-string-plan.md); mirrors its structure. Tolerances inherit Milestone 1's bar.
>
> **Key build decisions (recorded after implementation):**
> - Frequency-dependent loss term `+2σ₁ u_txx` discretized **centered in time** (`2σ₁ δ_t.(δ_xx u)`),
>   so it adds a single `−σ₁k·D2` block to the (still pentadiagonal, still SPD) update matrix `A`
>   and a `−σ₁k·D2 uⁿ⁻¹` term to the RHS — **no new factorization machinery**, `cholesky_banded`
>   carries over unchanged.
> - **The energy form `E^n` is identical to model #2's** — the loss terms never enter the stored
>   energy, only its *rate of change*. So `energy()` is reused verbatim and passivity (monotone
>   decrease) is automatic: both losses are dissipative by SBP.
> - `σ₁` is guarded exactly like `κ` (`if self.sigma1 != 0.0:`), so `DampedStiffString(σ₁=0)`
>   reduces **bit-for-bit** to `StiffString` — a free regression anchor.
> - The decay test is **per-mode and tight** (machine-precision-class, not model #2's 1% low-mode
>   bar): a single eigenmode's cross-time energy decays as `g_mⁿ` essentially ripple-free, measured
>   over an *interior* window (the lossless Taylor start is skipped — it is slightly inconsistent
>   under damping).

## Goal

The first model whose **decay is right**: high partials die faster than low ones, as every real
plucked/struck string does. Model #2 added stiffness but its frequency-*independent* loss decays a
mode at `2σ(1−θQk²)` — which, because `Q` is huge for high modes, makes the top partials *under-damp*
(retain too much energy), audibly **backwards**. Model #3 adds a deliberate frequency-*dependent*
loss term that sets per-mode decay growing with mode number, curing that artifact's audible symptom.

The deliverable is again **the resonator + the rig that measures its deviation from theory** — here
the "theory" is the closed-form per-mode decay rate. HANDOFF §5 row 3 names the validation: *measured
decay rates per partial*. That is exactly the money test.

## Physics

Add the standard two-parameter (Bilbao Ch. 7; after Chaigne–Askenfelt for piano strings) loss to the
stiff string:

```
u_tt = c²·u_xx − κ²·u_xxxx − 2σ₀·u_t + 2σ₁·u_txx
       \________ model #2 (conservative + κ) ________/  \__ losses __/
```

- `c² = T/ρ`, `κ² = E·I/ρ` — wave + bending, identical to model #2.
- `σ₀ ≥ 0` — **frequency-independent** loss (this is model #2's `σ`, renamed `sigma0` here). Damps
  every mode at the same base rate.
- `σ₁ ≥ 0` — **frequency-dependent** loss (the new term). `+2σ₁ u_txx`: in the modal domain
  `∂_xx → −β²` (`β = mπ/L`), so it is a velocity-proportional loss with a mode-dependent coefficient
  `σ₁β²`. High modes (large `β`) are damped harder — the physically-correct ordering.

**Per-mode loss (continuum):** for a mode `sin(βx)`, `q̈ = −ω²q − 2(σ₀ + σ₁β²)q̇`, so the **energy**
decays as

```
E_m(t) ~ exp(−2·σ_eff(m)·t),   σ_eff(m) = σ₀ + σ₁·β_m²,   β_m = m·π/L.
```

`σ₀` shifts the whole decay-rate curve up; `σ₁` tilts it so highs decay faster. Two numbers, the same
expressive surface a luthier hears: overall sustain + how fast the brightness fades.

### Energy / passivity (why the energy form is unchanged)

Multiplying the PDE by `u_t` and integrating by parts (fixed/supported ends kill the boundary flux):

```
dE/dt = −2σ₀‖u_t‖² − 2σ₁‖u_tx‖²  ≤ 0.
```

Both terms are ≤ 0 — passive. **Crucially neither loss term appears in `E` itself** (the stored
mechanical energy is the same kinetic + strain + bending as model #2); they appear only in `dE/dt`.
The discrete analog is identical: the same `E^n` from `StiffString.energy()`, now decreasing because
the discrete losses `−2σ₀‖δ_t.u‖² − 2σ₁‖δ_x⁺δ_t.u‖² ≤ 0` (SBP). So `energy()` is reused as-is.

## Scheme (one banded solve per step, matrix gains one term)

Time-average the conservative operator `𝓛 = c²δ_xx − κ²δ_xxxx` with the θ-weight (as model #2), and
discretize **both** loss terms with a centered time difference `δ_t.`:

```
δ_tt u = 𝓛·(θ·uⁿ⁺¹ + (1−2θ)·uⁿ + θ·uⁿ⁻¹) − 2σ₀·δ_t.u + 2σ₁·δ_t.(δ_xx u)
```

Rearranged (collect `uⁿ⁺¹`; `D2 = δ_xx` matrix; `k = 1/fs`):

```
A·uⁿ⁺¹ = 2uⁿ + (1−2θ)k²·𝓛uⁿ − uⁿ⁻¹ + θk²·𝓛uⁿ⁻¹ + σ₀k·uⁿ⁻¹ − σ₁k·D2·uⁿ⁻¹
A = (1+σ₀k)·I − θk²·𝓛 − σ₁k·D2
```

- `A` is **pentadiagonal, symmetric positive-definite for any σ₀,σ₁,κ ≥ 0, θ>0**: the D2 part of `A`
  is `−(θk²c² + σ₁k)·D2` and `−D2` is PD, the D4 part is `+θk²κ²·D4` (PD), plus `(1+σ₀k)·I`.
  Constant in time → factor once with `scipy.linalg.cholesky_banded`, back-substitute each step.
  **No CFL limit** (unconditional for θ ≥ ¼), so `λ = ck/h > 1` stays admissible.
- `σ₁ = 0` skips both new terms → bit-for-bit `StiffString`.

### Discrete per-mode decay oracle (single source of truth)

Insert `uⁿ = zⁿ·sin(mπx/L)` (an *exact* discrete eigenvector, since the biharmonic block is
`(δ_xx)²`). With `p² = (4/h²)sin²(mπ/2N)` (so `δ_xx → −p²`, `δ_xxxx → +p⁴`), `Q = c²p² + κ²p⁴`, and
the **discrete** per-mode loss `σ_eff = σ₀ + σ₁·p²`:

```
a·z² + b·z + cc = 0,
a  = 1 + θk²Q + σ_eff·k,   b = −2 + (1−2θ)k²Q,   cc = 1 + θk²Q − σ_eff·k.
```

For an **underdamped** mode the roots are a complex-conjugate pair with `|z|² = cc/a`, so the energy
(amplitude²) decays by a constant factor **per step**:

```
g_m = cc/a = (1 + θk²Q − σ_eff·k) / (1 + θk²Q + σ_eff·k)   ∈ (0,1],
E_m(t) = E_m(0)·g_m^(t/k),   Γ_m ≡ −(1/k)·ln g_m   (per-second energy rate).
```

As `k → 0` (and `θk²Q → 0`), `Γ_m → 2σ_eff(m) → 2(σ₀ + σ₁β²)` — the continuum rate. This `g_m` is the
oracle the simulated per-mode decay is checked against, tightly.

### The θ-artifact is *not removed* — it is accounted for, and swamped

`Γ_m ≈ 2(σ₀ + σ₁p²)/(1 + θk²Q)`. The model-#2 denominator `1/(1+θk²Q)` (which suppresses high-mode
decay — the under-damping artifact) is **still present**. Model #3 does not delete it; it adds a
numerator `σ₁p²` that grows with mode, so:

- **Audibly** the symptom is cured: in the musical band the deliberate `σ₁p²` growth dominates, so
  highs decay faster (correct).
- **Quantitatively** the oracle `g_m` includes the `θk²Q` terms exactly, so the per-mode decay test
  is tight regardless. *No test asserts the artifact is gone* (it isn't).
- The growth is **not monotone forever**: numerator ~p², denominator ~p⁴ with stiffness, so `Γ_m`
  turns over at very high modes. The "highs die faster" test asserts the ordering only over a
  mode range confirmed to be still rising (or "top tested mode ≫ fundamental"), never `diff>0` over
  the whole spectrum.

## T60 → (σ₀, σ₁) mapping (the deepened-physics payoff; continuum, loose round-trip)

Musicians specify *decay times*, not loss coefficients. Given two `(frequency, T60)` targets, invert
for `(σ₀, σ₁)`. Amplitude decays as `exp(−σ_eff t)` so a 60 dB drop (×10⁻³ amplitude) takes
`T60 = 3·ln10 / σ_eff = 6.9078/σ_eff`, i.e. `σ_eff(ω) = 6.9078/T60(ω)`. With `β²(ω)` from the
continuum dispersion `ω² = c²β² + κ²β⁴`:

```
[1  β₁²] [σ₀]   [6.9078/T60₁]
[1  β₂²]·[σ₁] = [6.9078/T60₂]
```

Solve the 2×2 system. **Pick targets with T60 decreasing in frequency** so `σ₀,σ₁ ≥ 0`. This is a
**continuum** inversion; the sim decays at the *discrete* rate (with the `1/(1+θk²Q)` θ-factor and
`p²` vs `β²`), so the round-trip lands a **few %** off — a documented physics demo, **not** a
machine-precision test. (Want it tight? Invert against `g_m` instead — deferred; the per-mode `g_m`
test already gives the rigorous decay validation.)

## Work breakdown (file by file)

**Build order (de-risk):** (1) per-mode oracle, (2) core scheme, (3) the per-mode decay test green,
then (4) T60 mapping, (5) viz/diagnose. The decay oracle is the required core; the mapping is polish.

1. **`analysis/damping.py`** (new, pure) — `spatial_eigenvalue_p2`, `modal_loss_rate_continuum`
   (`2σ_eff` from `β²`), `discrete_damped_mode_decay` (per-step `g_m`), `discrete_damped_mode_rate`
   (`Γ_m`), and `loss_coefficients_from_T60`. Single source of truth for the decay relation; no
   dependency on the core.
2. **`core/string_damped.py`** (new) — `DampedStiffString`, same interface as `StiffString`
   (`__init__`, `set_state`, `step`, `state`, `energy`, `displacement_at`). New inputs `sigma0`,
   `sigma1`. Standalone (one-file-per-model culture; subclassing would force a double-factor or a
   refactor of frozen `StiffString.__init__`). Store `_D2`; build/factor `A` with the σ₁ term.
3. **`tests/helpers.py`** — `make_damped_string(...)` and `measure_mode_decay_rate(...)` (single-mode
   init, fit the energy decay over an interior window).
4. **`tests/test_damped_string.py`** (new) — the validation suite (below).
5. **`viz/plots.py`** — `plot_decay_rates` (measured vs oracle `Γ_m` vs mode; σ₁=0 falling vs σ₁>0
   rising overlay — the cure made visible).
6. **`scripts/diagnose_damped_string.py`** (new) — energy passivity, per-mode decay-rate curve,
   T60 round-trip, displacement. Writes to `out/`.

## Tests — acceptance criteria (tolerances = Milestone 1's bar)

1. **Lossless reduction:** `σ₀=σ₁=0` → energy drift < 1e-10; partials on the stiff oracle. (Anchor.)
2. **σ₁=0 == StiffString:** with `σ₀>0`, the energy trace equals `StiffString`'s to machine precision
   (the σ₁ guard makes `A`/`step` identical). Free regression.
3. **Passivity (broadband pluck):** energy monotone non-increasing for `σ₀,σ₁>0`; **plus a σ₁-only
   case** (`σ₀=0, σ₁>0`) isolating the new term (bisection culture).
4. **Per-mode decay rate matches the oracle (the money test):** init a single mode, fit
   `(E[n₂]/E[n₁])^{1/(n₂−n₁)}` over an interior window, compare to `g_m` across a range of *under-
   damped* modes. Tight (≪ 1%).
5. **Highs die faster (the cure):** with `σ₁>0`, `Γ_m` rises with mode over a confirmed-rising range;
   and at `σ₁=0` (model-#2 regime) it *falls* with mode. Assert both — σ₁ flips the ordering.
6. **T60 round-trip:** invert two `(f,T60)` targets → `(σ₀,σ₁)`; simulate those two modes; measured
   T60 within a few % of target. Continuum demo, loose tol.
7. **Construction guards:** `σ₁<0` rejected; `σ₀<0` rejected; `λ>1` accepted (unconditional);
   non-physical params rejected; oracle/continuum agreement on refinement.
8. **Portability:** auto-covered — `test_stability.py` sweeps every `core/` submodule, so
   `string_damped.py` is checked by the headless/allowlist/no-sibling guards with no edits.

## Traps (pre-flagged)

- **Don't assert `Γ_m` is monotone across the whole spectrum** — it turns over at high modes
  (`p²` numerator vs `p⁴` denominator). Test on a confirmed-rising range. *Most likely tuning trap.*
- **Skip the start in the decay fit** — `set_state` uses the lossless Taylor `uⁿ⁻¹`; it is slightly
  inconsistent under damping, so the first few steps deviate from `g_mⁿ`. Fit an interior window.
- **Keep tested modes underdamped** (`b²−4ac < 0`) so `g_m`'s complex-conjugate-root form holds; a
  heavily overdamped mode is two real roots and not a single clean exponential.
- **Energy drift triangulation (inherited):** right frequency + wrong drift ⇒ energy-measure bug;
  wrong frequency ⇒ scheme/operator bug. Here additionally: right lossless drift but wrong *decay
  rate* ⇒ the loss discretization (`σ₁k·D2` sign/placement), not the conservative scheme.
- **T60 mapping is continuum** — never chase 1e-3 on its round-trip; the discrete θ-factor guarantees
  a few-% gap. The rigorous decay check is the per-mode `g_m` test.
- `A` is constant only while params are fixed; a setter mutating σ₀/σ₁/κ must re-factor.
```
