# Stiff String — Phase 2 Plan (model #2)

> Proposed plan for HANDOFF §5 model #2 + §11.2 (implicit scheme). Plan-only; not yet built.
> Mirrors `ideal-string-plan.md`. Tolerances inherit Milestone 1's bar (human's call, 2026-06-21).

## Goal

The first model that *sounds like a real instrument*: a stiff string whose bending stiffness
stretches the partials away from a perfect harmonic series (piano-like inharmonicity). The
deliverable is again **the resonator + the rig that measures its deviation from theory** — here the
"theory" is the stretched-partial law `fₙ = n·f₀·√(1 + B·n²)`. The entire Milestone-1 harness
(energy / passivity / modal / convergence / dispersion) carries over; the stiff string's own
dispersion relation makes the dispersion curve a *direct* stretched-partials check.

## Physics

Add the biharmonic (bending) term to the wave equation (HANDOFF §5, Bilbao Ch. 7):

```
u_tt = c²·u_xx − κ²·u_xxxx − 2σ·u_t      c = √(T/ρ),  κ² = E·I/ρ   (ρ = linear density, kg/m)
```

- `c² = T/ρ` — the wave term, identical to the ideal string.
- `κ² = E·I/ρ` — the **stiffness coefficient** (E Young's modulus, I area moment of inertia). New
  input. Larger κ → more inharmonicity. `κ = 0` must recover the ideal string *exactly* (a strong
  regression anchor — see Tests #7).
- `σ ≥ 0` — simple frequency-independent loss, carried over verbatim. (Frequency-*dependent* loss is
  model #3, a later phase — not in scope here.)

**Stretched-partial oracle (simply-supported ends):** the eigenfunctions are exactly `sin(nπx/L)`
with dispersion `ωₙ² = c²(nπ/L)² + κ²(nπ/L)⁴`, hence

```
fₙ = n·f₀·√(1 + B·n²),   f₀ = c/(2L),   B = π²·κ²/(c²·L²)   (inharmonicity coefficient)
```

`B` is the single audible parameter; for a piano B ~ 1e-4…1e-3. This closed form holds cleanly only
for **simply-supported** ends (see Boundaries).

> **Use `f₀ = c/(2L)`, not the true fundamental.** HANDOFF §5 loosely writes `n·f₁·√(1+Bn²)`. If
> someone substitutes the *actual* fundamental as `f₁`, the n=1 partial is double-stretched by an
> extra `√(1+B)` — ~0.9 cents at B=1e-3, nearly the whole 1-cent budget. The oracle must satisfy
> `f₁ = f₀·√(1+B)` (assert this explicitly), i.e. even the fundamental is stretched off `c/2L`.

## Scheme decision (HANDOFF §11.2): implicit θ-scheme on the whole operator; θ is the accuracy knob

Treating the biharmonic explicitly tightens the CFL brutally (the `κ²k²/h⁴` term forces a fine grid).
Per the human's 2026-06-21 call we go **straight to implicit**. Apply a θ-weighted time average to
the **whole** spatial operator `𝓛 = c²δ_xx − κ²δ_xxxx`:

```
δ_tt u = 𝓛·(θ·u^{n+1} + (1−2θ)·u^n + θ·u^{n-1}) − 2σ·δ_t·u
```

θ=0 is the (CFL-bound) explicit scheme; θ=½ is the centered μ-average. The family is
**unconditionally stable for any θ ≥ ¼** (derivation below), so the binding `κ²k²/h⁴` constraint is
removed for the whole admissible range — and **θ is then a pure accuracy knob**, not a stability one.

**Update (one banded solve per step).** Rearranged with `A = I − θk²𝓛 + σk·I`:

```
A·u^{n+1} = 2u^n + (1−2θ)k²·𝓛u^n − u^{n-1} + θk²·𝓛u^{n-1} + σk·u^{n-1}
```

`A` is **pentadiagonal** (biharmonic = 5-point stencil), symmetric positive-definite, and **constant
in time** → factor once at construction, back-substitute each step. Use `scipy.linalg.solveh_banded`
(SPD banded) or a prefactored `scipy.sparse` LU. *scipy is already on the core dependency allowlist*
(`tests/test_stability.py::test_core_dependency_allowlist` permits `{physsynth, numpy, scipy}`), so
the implicit solver needs no contract change.

**Why it's unconditionally stable (derivation result).** Inserting `u^n = z^n·sin(mπx/L)`, with the
spatial eigenvalue `p² = (4/h²)·sin²(mπ/2N)` (so `δ_xx → −p²`, `δ_xxxx → +p⁴`) and `Q = c²p² + κ²p⁴`:

```
s ≡ sin²(ωk/2) = Q·k² / (4 + 4θ·Q·k²)        →   s ≤ 1  ⇔  Qk²(1−4θ) ≤ 4
```

For θ ≥ ¼ the LHS is ≤ 0, so **no bound** on k, h, or κ. The discrete dispersion oracle is

```
f_m^discrete = arcsin(√s) / (π·k),   s = Q·k²/(4 + 4θ·Q·k²),   Q = c²p² + κ²p⁴
```

which → the continuum `fₙ = n·f₀·√(1+Bn²)` as `h,k → 0` for any θ. The leading **temporal**
frequency error is `ω² ≈ Q − θ·Q²k²` — **proportional to θ** — so smaller θ ⇒ less numerical
dispersion.

**θ choice (the real accuracy/robustness fork; accuracy-first ⇒ lean toward ¼):**

| θ | Dispersion error | Energy / positivity | Note |
|---|---|---|---|
| **¼** | minimal (½ that of θ=½) | energy is *exactly* the cross-time form (stabilizer term vanishes) | **marginally** stable: high modes → Nyquist (`s→1`), zero positivity margin |
| **slightly > ¼** | near-minimal | manifestly-positive energy with a small margin | **recommended default** |
| ½ | ~2× the error | most robust positive energy | the plain μ-average; fallback if a margin issue appears |

**Recommendation:** default to **θ a hair above ¼** (e.g. 0.25–0.30), keep θ a constructor parameter,
and have the implementation compare a couple of θ values against the dispersion oracle empirically.
This honors non-negotiable #1 (accuracy first) rather than picking ½ for convenience. (Contrast:
keeping the wave term explicit and only the bending implicit leaves `λ ≲ 1` in force — *not*
unconditional — so we average the whole operator regardless of θ.)

## Energy (form is θ-dependent)

`E^n` is again kinetic + potential. **The exact form depends on θ** (this is why θ and the energy
must be chosen together):

```
E^n = ρ·[ ½‖δ_t⁻u^n‖²_w  +  (potential from 𝓛)  +  (θ−¼)·k²·(stabilizer term) ]
```

- Kinetic term: reuse the ideal string's trapezoid-weighted `½‖δ_t⁻u^n‖²` verbatim.
- Potential: the wave part keeps the proven cross-time form `(c²/2)⟨δ_x⁺u^n, δ_x⁺u^{n-1}⟩`; the
  bending part adds the analogous biharmonic cross-time term `(κ²/2)⟨δ_xx u^n, δ_xx u^{n-1}⟩`.
- **Stabilizer:** at **θ=¼ this term vanishes** and the energy is *exactly* the cross-time form
  (clean, but no positivity margin); for **θ>¼** the `(θ−¼)k²` term is what makes the total
  manifestly ≥ 0 with no step-size condition. Tie the implemented energy to the chosen θ.
- Exact algebra pinned against Bilbao Ch. 7 during implementation and **verified by the drift test**,
  not trusted from memory.
- Lossless ⇒ drift < 1e-10 (M1 bar). Lossy ⇒ monotone decrease at `e^{-2σt}` (passivity), reusing
  the ideal-string loss bookkeeping.

> **Triangulation when drift fails:** check the *dispersion-oracle* test first. Right frequency but
> wrong drift ⇒ the **energy measure** is wrong (the `E^n` formula / a missing θ-stabilizer term),
> *not* the scheme. Wrong frequency ⇒ the **scheme/operator** is wrong (boundary rows, sign, coeff).
> This split saves hunting the bug in the wrong half.

## Boundaries — simply supported first (SBP-sensitive; check here first if E drifts)

A 4th-order operator needs **two** conditions per end:

- **Simply supported (pinned): `u = 0` and `u_xx = 0`.** Primary, validated config. Gives the clean
  `sin(nπx/L)` oracle above and keeps `sin` an *exact discrete eigenvector* of the whole scheme — so
  the modal-projection dispersion measurement and `modal.mode_shape` carry over **unchanged**.
  Discrete realization: clamp `u[0]=u[N]=0` and use the odd-reflection ghost `u[-1] = −u[1]`
  (from `u_xx=0`), which makes the biharmonic's boundary row `…+5·u[1]` (5, not the interior 6).
- **Clamped (`u=0, u_x=0`) and free (`u_xx=0, u_xxx=0`):** no simple closed-form oracle; defer.

Per HANDOFF §8, *most conservation drift lives in boundary handling* — the `5-vs-6` boundary row and
the ghost-node reflection are the prime suspects; the energy test is the detector.

## Work breakdown (file by file)

**Build order (de-risk):** bring up the **implicit machinery at κ=0 first** — the θ-scheme wave
solver + its energy, and prove drift < 1e-10 there — *then* add the biharmonic term and the boundary
rows. If energy drifts after step one, it's the implicit machinery / energy form; if it drifts only
after the bending term is added, it's the `5-vs-6` boundary or the biharmonic energy term. This
bisects the two riskiest pieces instead of debugging them entangled.


1. **`core/operators.py`** — add `delta_xxxx(u, h)` (interior 5-point `(u[l+2]−4u[l+1]+6u[l]−4u[l-1]
   +u[l-2])/h⁴`) and a `biharmonic_matrix(N, h, boundary)` builder (banded, with the simply-supported
   boundary rows). Keep pure NumPy; the *matrix* is fine here, the *solve* lives in the resonator.
2. **`core/string_stiff.py`** — `StiffString` resonator, same interface as `IdealString`
   (`__init__`, `set_state`, `step`, `state`, `energy`, `displacement_at`). New input `kappa`
   (or `E, I`); derive `B`. Construct + factor the pentadiagonal `A` once (scipy banded). `step()`
   builds the RHS and back-substitutes. CFL guard becomes a no-op note (unconditional) but still
   reject non-physical params. `κ=0` path must be bit-for-bit the ideal scheme’s behavior.
3. **`analysis/modal.py`** — add `inharmonicity_B(c, L, kappa)`, `stiff_harmonic_frequencies(c, L,
   kappa, n)` = `n·f₀·√(1+Bn²)`, and `discrete_stiff_mode_frequency(c, L, N, kappa, k, m)` (the
   closed form above, single source of truth for the oracle).
4. **`analysis/dispersion.py`** — generalize `dispersion_frequencies` (or add a stiff variant) to
   call the stiff oracle. `phase_velocity` is unchanged (`v_p = 2Lf/m`).
5. **`tests/helpers.py`** — a `make_stiff_string(...)` builder and a stiff-aware
   `measure_mode_frequencies` (the sin-projection trick is identical; just point the peak search at
   the stiff oracle).
6. **`tests/test_stiff_string.py`** (new) — the validation suite (below).
7. **`viz/plots.py`** — a `plot_stiff_partials` overlaying detected vs the *stretched* law and vs the
   plain harmonic `n·f₀` (so the stretch is visible). `plot_dispersion` already generalizes.
8. **`scripts/diagnose_stiff_string.py`** (new) — energy-vs-time, stretched-partials, dispersion,
   displacement animation, and an inharmonicity figure. Writes to `out/`.

## Tests — acceptance criteria (tolerances = Milestone 1's bar)

1. **Energy conserved (lossless):** `max|Eⁿ−E⁰|/E⁰ < 1e-10` over ≥ 2 s, across a sweep of κ and a
   sweep of λ — *including a coarse grid / large timestep that the explicit stiff scheme could not
   run* (demonstrates the unconditional-stability win).
2. **Passivity (σ>0):** energy monotone non-increasing; decay matches `e^{-2σt}` to a few %.
3. **Stretched partials (the money test):** detected partials match `n·f₀·√(1+Bn²)` within ~1 cent
   for the first ~10 partials; assert `f₁ = f₀·√(1+B)` (even the fundamental is stretched). **Plus the
   real bending anchor:** fit `B` from the measured partials across a **κ sweep** and assert it tracks
   `π²κ²/(c²L²)` quantitatively (B ∝ κ²). This validates the new term's sign *and* scale far better
   than a κ=0 reduction can, and confirms the partials are distinguishable from the harmonic series.
4. **Convergence:** halving h (at fixed λ<1, κ>0) shrinks the partial error at the scheme's rate
   (expect ~O(h²); the θ=½ time scheme is also 2nd order). Mean order ~2.
5. **Stability / no-blow-up:** no NaN across κ and λ sweeps; explicitly assert a config that
   *violates the explicit biharmonic bound* runs stably (unconditional). Non-physical params rejected.
6. **Dispersion:** measured `v_p(m)/c` vs the discrete stiff oracle across the mode range (sin
   projection); droops with mode as bending stiffens high partials.
7. **κ=0 self-consistency anchor (NOT equality with IdealString).** `StiffString` at κ=0 is the
   *implicit θ-scheme*, a **different scheme** from the explicit `IdealString` — they do **not** agree
   to machine precision (the implicit one is not even exact at λ=1). So assert instead:
   (a) κ=0 partials match `discrete_stiff_mode_frequency(κ=0)` to machine precision (implicit path is
   self-consistent); (b) κ=0 energy is *conserved* (drift < 1e-10) — not "energy trace == IdealString";
   (c) agreement with `IdealString` only in the refined / low-mode limit, at a *loose* tolerance.
   (The genuine new-term sign/scale check lives in #3's B-vs-κ² sweep, which is stronger.)
8. **Portability:** auto-covered — `test_stability.py`'s `_IMPORT_ALL_CORE` iterates every
   `core/` submodule, so `string_stiff.py` is swept by the headless/allowlist/no-sibling guards with
   no edits. Confirm scipy stays within the allowlist (it does).

## Open sub-decisions (surface; recommendation in **bold**)

- **θ (the accuracy/robustness knob):** **default a hair above ¼** (≈0.25–0.30) for accuracy-first,
  keep it a constructor param, compare ¼-ish vs ½ against the dispersion oracle empirically. ½ is the
  robust fallback. See the scheme table.
- **Stiffness input:** raw `κ` vs `(E, I)` vs musician-facing `B`. **Recommend `κ` as the core
  input** (one number, matches the math), with `inharmonicity_B` exposed for reporting; `(E,I)` and
  `B`-driven constructors can wrap it later (HANDOFF §8 parameter-mapping is a separate concern).
- **Boundary for the milestone:** **simply supported** (clean oracle, sin eigenvectors, max harness
  reuse). Clamped/free are follow-ups.
- **Tolerances:** **inherit M1** — drift < 1e-10, partials ~1 cent, dispersion `ORACLE_RTOL=1e-4`,
  `CONTINUUM`-analog for the κ=0 reduction. Revisit only if the implicit scheme over/under-performs.

## Traps (pre-flagged)

- Energy drift → **triangulate via the dispersion-oracle test** (right freq + wrong drift = energy
  measure bug; wrong freq = scheme/boundary bug). Then suspect the **biharmonic boundary rows**
  (5-vs-6, ghost reflection) and the **θ-dependent stabilizer term** in `E^n`. Never relax the
  tolerance (HANDOFF §8).
- Don't trust the implicit energy algebra from memory — pin it to Bilbao Ch. 7, tie it to the chosen
  θ (stabilizer vanishes at ¼), and let the drift test arbitrate.
- **κ=0 ≠ IdealString** — the implicit scheme differs from the explicit one even at λ=1; assert
  self-consistency against `discrete_stiff_mode_frequency(κ=0)`, not equality with `IdealString`.
- The factored matrix is constant only while params are fixed; if any setter mutates κ/σ/h, re-factor.
- Keep the *solve* out of energy()/operators (purity); the resonator owns the scipy factorization.
- Convergence at λ=1 is *not* exact for the stiff string (the κ-term disperses even at λ=1) — unlike
  the ideal string, so the convergence study has signal at any λ; still prefer a higher mode.
```
