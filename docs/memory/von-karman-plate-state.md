---
name: von-karman-plate-state
description: Phase-4 model
metadata: 
  node_type: memory
  type: project
  originSessionId: 4b4bd7cd-a9b8-46a4-bd14-4e7f78d0211d
---

Phase-4 **model #6 — von Kármán nonlinear plate**: **Parts 1–3 (+ Part-4 validation) BUILT & GREEN
(2026-07-01)**; Parts 5 (diagnostics) + 6 (free-edge) still to build. `VonKarmanBracket` (P1) +
`AiryStressSolver` (P2) in `operators2d.py`; **`VKPlate` coupled resonator (P3)** in `core/plate.py`.
`tests/test_vk_bracket.py` (15) + `test_vk_airy.py` (13) + `test_vk_{energy,modal,stability}.py` (27),
full suite **349** green, ruff clean.

**Part 3 DONE — coupled `VKPlate` (`core/plate.py`), conservative Picard-iterated scheme:** new class,
model #5's `Plate` **left untouched** — `nonlinear=False` is **bit-identical** to `Plate(supported)`
(regression, exact-arithmetic `u_prev` match). Materials surface **`(E, e, ν, ρ)`** (human decision #3,
2026-07-01; the memory's old "(kappa,…)" was self-contradictory) → derive `ρ_s=ρe`, `D=Ee³/(12(1-ν²))`,
`κ=√(D/ρ_s)`, `Y_mem=Ee`. **Attr names (2026-07-02 cleanup):** VKPlate grid is `self.X,self.Y`
(conforms to membrane/Plate/free-plate convention; was `Y_grid`), membrane coeff is `self.Y_mem`
(was `self.Y`), volumetric density is `self.rho_v` (was `self.rho`; areal stays `rho_s`). NB
**Plate.rho is areal** — VKPlate has no `.rho`. **Scheme (advisor-derived from first principles):**
`ρ_s δ_tt w = -D B(θ-avg w) + l(μ_{t·}w, μ_{t·}F)`, `F^m` solved from `w^m`, `μ_{t·}g=(g^{n+1}+g^{n-1})/2`.
Coupling `⟨l(μw,μF),w^{n+1}-w^{n-1}⟩` telescopes **exactly** to `-(H_mem^{n+1}-H_mem^{n-1})` via P1
triple-self-adjointness (`l(w^m,w^m)=-(2/Y)B_F F^m`; Wa-vs-h² harmless: F=0 on rim). **Went straight to
IMPLICIT Picard, NOT the plan's explicit `l(w^n,F^n)`** — the membrane potential is **quartic**, so any
frozen-coeff explicit coupling drifts at O(k²), never `<1e-10`; exact conservation *requires* the new
level → implicitness unavoidable (the plan's blessed "fallback", built directly). Per-step: predictor
`2w^n-w^{n-1}`, sweeps of one prefactored `B_F`-solve (`F^{n+1}`) + one `A`-solve (`+k²·coupling/ρ_s`,
`A`=model-#5 verbatim), converge `‖Δw‖/‖w‖≤couple_tol=1e-13` (≤11 sweeps at `w≈e`; `converged`/
`last_residual` exposed for the cascade). **Live↔full-grid seam** (advisor catch): `self.u` is
live/interior, bracket+Airy want full-grid rim-0 → `embed`→bracket/solve→restrict `[mask.ravel()]`.
**F-cache bookkeeping:** `self.F`/`self.F_prev` always track `(u, u_prev)`; coupling uses `F_prev`
(=F^{n-1}), energy uses F^{n+1},F^n. **Energy = half-step-averaged** `E_lin + ½(H_mem(F^{n+1})+H_mem(F^n))`
(advisor: raw integer-membrane sum is a 2-step odd/even invariant → spurious oscillation); the ½ factor
is *certified by* the high-membrane-fraction drift test (57 %→2.6e-13; a 1× factor would give
tens-of-%). **Gates (all green):** lossless drift **2.6e-13** at `w≈3e` (membrane **57 %** of H — real
bracket exercise, not linear re-test; 93 % at w=10e still drift 9e-13), **drift ∝ couple_tol**
(1e-4→3.5e-5 … 1e-12→9.3e-13 — the machine-precision self-cert absent a closed form), non-negativity,
passivity **exact** (worst rise 0.0), `w→0`→model-#5 fundamental (reldiff **0**), **pitch-glide
hardening** monotone (+74 % at w=5e), **Richardson O(h²)** ratio **4.40** (N=24/48/96 smooth-IC; N=16
pre-asymptotic ~3 since `l(w,w)` doubles wavenumbers). Probed in `M:/claud_projects/temp/vk_*.py`.
Next: Part 5 (`scripts/diagnose_vk_plate.py` + viz: energy trace w/ membrane broken out, pitch-glide
spectrogram, w/e sweep), then Part 6 free-edge cymbal (swap `B→K`, free F-BC; bracket/F-solve/energy
carry over).

**Part 2 DONE — F elliptic solve (`AiryStressSolver`):** F-BC **resolved with human = CLAMPED
`F=0, F,n=0`** (physically-correct SS-*movable* edge; DT DAFx-15 §4.2 Eq.11 `F,tt=F,nt=0`; the plan's
old `L²`/Navier `F=0,ΔF=0` default is a *different, nonstandard* edge — advisor). Built **energy-first**
(advisor: simpler + symmetric-by-construction vs a hand-assembled 13-pt stencil): `B_F = Lc_rᵀ Wa Lc_r`
squaring the **clamped Laplacian** `Lc = kron(iy,c2c_x)+kron(c2c_y,ix)`, `c2c = _clamped_d2_1d` = the
`_centered_d2_1d` end rows with off-diag **doubled 1→2** (ghost mirror `F_{-1}=F_1` from `F,n=0`).
**Trapezoidal `Wa` (reused from `free_plate_stiffness`) is load-bearing:** 1D Gram then reproduces the
textbook clamped biharmonic exactly (diag `7,6,…,6,7`, off `-4,1`); `Wa=I` gives wrong `9`. Symmetric
SPD, **empty** nullspace (clamping kills `{1,x,y}`), `splu`-prefactored. Repr: full-grid `Lc`, drop rim
**columns** keep all **rows**; `solve(source_full)`→restrict→**`Wa`-weighted Galerkin load (interior
`h²`)**→solve→embed rim=0. **Advisor's subtle bug:** manufactured RHS must be `Wa`-weighted
(`B_F F = Wa·∇⁴F`). **Discriminator (proves clamped≠`L²`):** manufactured `(1−cos)(1−cos)` (F=F,n=0
but ΔF≠0 on rim) → clamped recovers O(h²) (rates 2.01→2.00), SS `biharmonic_from_mask` *saturates* at
O(1) (ratio 185→2941). **Real `F→0` gate + P1↔P2 seam test** (advisor): `bracket(w,w)→solve` end-to-end,
`F ∝ ‖w‖²` so doubling `w` → **4×** F (bilinear bracket × linear solve; advisor slipped saying 16×,
verified 4×). Part-3 energy closure re-derived by advisor & closes (Wa-vs-h² mismatch harmless: F=0 on
rim). Part-3 membrane energy = plain `(1/2Ee)·Fᵀ B_F F` (`laplacian_norm_sq`).
Probed in `M:/claud_projects/temp/vk-Bf-probe/` (caught my `∇⁴F` sign: `g''''=−a⁴cos`).
Plan `docs/dev/von-karman-plate-plan.md` (commit 1938941, 2026-06-30). The deep end of
HANDOFF §5 row 6 — gongs/cymbals, pitch glide, the crash/shimmer cascade. **First model with genuine
nonlinear coupling and NO analytic modal oracle** → energy conservation becomes *the* correctness
test, not a cross-check (HANDOFF §4.5: von Neumann analysis fails for nonlinear; energy analysis is
the only robust route). Builds on [[plate-state]] (SS `B=L²`) and [[free-plate-state]] (free `K`/`W`,
the cell-centered twist `Dxy`).

**Human decisions taken (2026-06-30):**
- **SS-first de-risk, then free-edge follow-on** (Part 2). SS sounds like a nonlinear tom w/ pitch
  glide, NOT a cymbal — the cymbal/gong is the *free-edge* Part 2. Mirrors beam→free-plate culture.
- **Core params `(kappa, E, e, nu)`** → derive `D=Ee³/(12(1−ν²))`, membrane coeff `Ee`, `κ`. Thickness
  `e` is now *physically meaningful* — nonlinearity onset is at `w≈e` (unlike linear models, κ-only).
- **Human reviews the plan doc before any code is written** (build is HELD pending that review).

**Load-bearing approach (advisor-confirmed):**
- **Airy-stress-function form**: transverse `w` + stress function `F`. Two coupled fields:
  `ρ_s w_tt = −D∇⁴w + L(w,F) − 2ρ_sσw_t`, `∇⁴F = −(Ee/2)L(w,w)` (elliptic solve for F each step).
  Bracket `L(α,β)=α_xx β_yy+α_yy β_xx−2α_xy β_xy` (Monge–Ampère). `L(w,F)` is cubic in w → vanishes as
  w→0 (recovers model #5). `L(w,w)`=2×Gaussian curvature = the F source.
- **THE crux / operator money test (no 1D analog → this IS the de-risk):** the discrete bracket
  `l(·,·)` must satisfy **triple self-adjointness** `⟨l(a,b),c⟩=⟨l(a,c),b⟩=⟨l(c,b),a⟩` on random
  fields to machine precision. Unit-test standalone BEFORE any time loop. One `l()` impl, two call
  sites (`l(w,w)` F-source, `l(w,F)` coupling) — sharing mandatory or conservation breaks. The
  cell-centered twist `Dxy=kron(d1y,d1x)` already in operators2d.py is the natural mixed-deriv block.
- **Two factorizations, not one** (advisor trap): `A=(1+σk)M+θk²κ²B` for the w θ-step (model #5's
  matrix verbatim), a SEPARATE biharmonic `B_F` for the F-solve. Both `splu`-prefactored.
- **F's in-plane BC is independent of w's transverse SS** and ⏳ **TO PIN from Bilbao NSS Ch.13 at
  build** (Part 1 step 2). Default `F=0,ΔF=0` → `B_F=L²` (reuse model #5); but may be `F=0,F_n=0`
  (clamped biharmonic = new operator). Nonlinearity is active either way (F-solve is a *forced*
  biharmonic; source `l(w,w)≠0` builds tension regardless; BC shapes *which* in-plane physics).
- **Energy** `H=½ρ_s‖w_t‖² + (D/2)U_bend(w) + (1/2Ee)‖∇²F‖²` (kinetic + bending[reused] + NEW
  membrane). Conservation needs the discrete `l` symmetry above (the trilinear cancellation).

**Validation (energy-method-first, no nonlinear closed form):** headline = lossless energy drift
`<1e-10` **at LARGE amplitude `w≳e`** (membrane term ≥10–20% of H — a bracket bug HIDES at small
amplitude, just re-tests the linear scheme); energy **non-negativity** `H^n≥0` (stability does NOT
transfer from the linear θ-scheme — expect amplitude-dependent step bound, implicit conservative
variant is the fallback); small-amplitude→model-#5 modes; **pitch glide** (fundamental rises w/
amplitude, hardening — qualitative, no cents bar); Richardson O(h²); passivity. Oversample (cascade
aliasing).

**Part 1 DONE — how the bracket was pinned (empirically, not from the book):** advisor reframed the
money test as a *self-certifying gate* → build a candidate, let the test certify it. Decisive probes
(`M:/claud_projects/temp/vk-bracket-probe/`): straight collocated terms + **centered** twist = O(1)
asymmetric; straight + **cell-centered forward-forward** twist (product scattered to nodes by the
corner-average adjoint `Acell.T`) = **triple self-adjoint to 1e-15** — the twist asymmetry exactly
cancels the straight terms' (Bilbao's contribution, reproduced & confirmed). **Domain contract
(tested):** cancellation holds *only for rim-vanishing fields* (SS `w=F=0`); non-zero border → O(1)
asymmetric (expected, why SS is first). Money test uses rim-vanishing random fields — full-grid
random would falsely FAIL a correct bracket. **Consistency test added** (advisor: `l≡0` passes
symmetry) → O(h²) to analytic bracket (rate 2.00→1.98). Field:
`l(a,b)=(δxx a)(δyy b)+(δyy a)(δxx b)−2 Acellᵀ[(Dxy a)(Dxy b)]`, reusing `_forward_d1_1d` for `Dxy`,
new `_centered_d2_1d`/`_avg_d1_1d`. **Get Bilbao PDF (downloaded, unread) for Part-3 μ time-averaging
operators** — no cheap self-cert there (advisor).

**Build order (de-risk, each gate green first):** (1) bracket `l()` + triple-self-adjointness test
[DONE ✅] → (2) F elliptic solve `B_F` + manufactured-soln check [DONE ✅ clamped] → (3)
coupled `VKPlate` resonator (`nonlinear=False` bit-identical to model #5) [DONE ✅] → (4) validation
tests [DONE ✅, 27 tests] → (5) diagnostics (pitch-glide spectrogram) [TODO] → (6) Part 2 free-edge
cymbal (swap B→K + free F-BC; bracket/F-solve/energy all carry over) [TODO].

**Sources:** Bilbao 2008 "A Family of Conservative FD Schemes for the Dynamical von Kármán Plate
Equations" (Numer. Methods PDEs 24(1):193–216) + NSS 2009 Ch.13 = primary FD source (pin bracket +
F-BC here, don't reconstruct from memory). Ducceschi–Touzé DAFx-15 = continuous eqns (verbatim) +
the `w/e` regimes (linear/pitch-glide/cascade) + free-circular F-BCs; a *modal* route, independent
cross-check. See [[commit-push-at-batch-end]]; mind [[respect-ruff-line-length]].
