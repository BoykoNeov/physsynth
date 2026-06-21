# Ideal String — Milestone 1 Plan

> Accepted plan for HANDOFF §10. Living document — update as implementation progresses.

## Goal

An ideal-string FDTD solver **plus a validation harness that proves it correct**. The deliverable is
the string *and the rig that measures its deviation from theory*, not just a string.

## Approach (decisions taken on documented defaults)

- **Language:** Python / NumPy (non-negotiable #3). Not blocked on the human.
- **Scheme:** explicit second-order scheme (HANDOFF §4.2). The handoff's own recommendation for
  milestone 1; implicit path deferred to the stiff-string stage (§11.2).
- **Energy:** discrete energy with the **cross-time potential term**
  `Eⁿ = ½‖δ_t⁻ uⁿ‖² + (c²/2)·⟨δ_x⁺ uⁿ, δ_x⁺ uⁿ⁻¹⟩` (the non-obvious move that makes conservation
  exact). Same-time `‖δ_x⁺ uⁿ‖²` would drift ~1e-3 — do not use it.
- **Boundaries:** fixed (Dirichlet) for the milestone; free (Neumann) supported and tested for energy.
- **Loss:** centered-difference `−2σu_t` term → passive (energy monotonically decreasing).

## Stages

1. **Scaffold (durable first).** Package layout, pyproject, gitignore, README, CI, dev-docs, git init.
2. **Core.** `operators`, `string_ideal` (`step`/`state`/`energy`), `exciter` (pluck + mode init), `engine`.
3. **Analysis.** `modal` (analytic `fₙ = n·c/(2L)`, discrete dispersion oracle), `spectrum` (FFT peaks).
4. **Tests.** energy / passivity / modal / convergence / stability.
5. **Diagnostics.** `viz/plots` + `scripts/diagnose_ideal_string.py`; run suite, report real numbers.

## Acceptance criteria (HANDOFF §10)

1. Lossless energy: `max|Eⁿ − E⁰| / E⁰ < 1e-10` over ≥ 2 s, **across a λ sweep in (0,1]** (the identity
   is algebraic, not λ=1-only).
2. Partials match `n·c/(2L)` within ~1 cent for the first ~10 partials at λ ≈ 1.
3. Convergence: halving `h` reduces partial error at the O(h²) rate — measured at **fixed λ < 1** (at
   λ = 1 the scheme is exact, so refinement shows nothing).
4. No NaN / blow-up across valid λ ∈ (0,1]; deliberate λ > 1 rejected at construction.
5. Diagnostics render energy-vs-time, detected-vs-analytic partials, displacement (and convergence).

## Traps (flagged by review)

- Energy drift → suspect the cross-time term first, then SBP/boundary. Never relax the tolerance.
- Convergence at λ=1 is exact → measure at λ<1, and prefer higher modes (larger, cleaner dispersion error).
- Headless core: matplotlib must never be importable from `physsynth/core`.

## Results — achieved (Milestone 1 complete)

All five acceptance criteria pass, and the HANDOFF §6.5 dispersion relation is now measured and
validated too (the one §6 test that was outstanding). Measured numbers (canonical string c=200 m/s,
f1=100 Hz):

| # | Criterion | Bar | Achieved |
|---|-----------|-----|----------|
| 1 | Lossless energy drift `max|Eⁿ−E⁰|/E⁰` (2 s, λ∈{1,.99,.9,.7,.5}) | < 1e-10 | **~3–7e-14** (λ=1: 6.1e-15); free BC 5.7e-14 |
| 1 | Passivity (σ>0): monotonic decrease | non-increasing | max positive step **0.0**; decay matches `e^{-2σt}` to <2% |
| 2 | Partials vs `n·c/2L`, first 10 at λ=1 | ~1 cent | **0.003 cents** worst |
| 3 | Convergence order at λ=0.9, mode 8 | ~2 | **[2.015, 2.006]** |
| 4 | No NaN over λ∈(0,1]; λ>1 rejected at construction | — | pass (7-λ sweep finite; `ValueError` on λ=1.05) |
| 5 | Diagnostics render | — | `out/ideal_string_diagnostics.png`, `_spectrum.png`, `_dispersion.png`, `.gif` |
| §6.5 | Dispersion: measured `v_p(m)/c` vs the discrete oracle (full mode sweep) | matches | λ=1 flat on continuum (**worst 8e-10**); λ=0.8 measured↔oracle **worst 1.3e-5**, v_p/c droops monotonically to 0.88 @ m=96 |

42 pytest cases pass; `ruff check` clean. Run `python scripts/diagnose_ideal_string.py` to reproduce.
(35 from the milestone proper + 2 portability-contract guards + 5 dispersion.)

### Dispersion — design notes (HANDOFF §6.5)

- **Oracle:** `analysis/dispersion.py` (`dispersion_frequencies`, `phase_velocity`), pure, wrapping
  the closed form in `modal.discrete_mode_frequency` so the relation has one source of truth.
- **Measurement:** each mode is excited *alone* as an exact discrete eigenvector `sin(mπx/L)`; the
  field stays `u(t)=q(t)·φ_m`, so the modal coordinate `q(t)=⟨u(t),φ_m⟩` is a pure tone measured by
  the existing FFT tooling. Projection (not a point pickup) — a point pickup can land on a node
  (mode N/2 vanishes at every even grid node), the projection never does, for any N.
- **What it adds over the existing modal/convergence tests:** those cover the λ=1 corner and the
  h→0 *rate*; dispersion checks `step()` reproduces the predicted frequency across the *whole mode
  range at λ<1* — where a coefficient/indexing bug hides while staying invisible at λ=1. Anchored to
  physics by also asserting the λ=1 sweep matches the *continuum* (not just the recurrence-derived
  oracle) and that the λ<1 phase velocity droops monotonically below c.
- **Tolerances provisional** (§11.5 still open): `ORACLE_RTOL=1e-4`, `CONTINUUM_RTOL=1e-7`, set a few
  × above the observed clean residuals.

**Next (Phase 2):** stiff string — add `-κ²·u_xxxx` (biharmonic), stretched partials
`fₙ = n·f₁·√(1+B·n²)`, tighter CFL. Scheme decision (HANDOFF §11.2): **go straight to implicit**
(θ/μ-scheme, unconditionally stable) — the human's call, made 2026-06-21. The harness
(energy/modal/convergence/**dispersion**) carries over; the stiff string has its *own* richer
dispersion relation `f_n = n·f₁·√(1+B·n²)`, so this curve becomes a direct stretched-partials check.
