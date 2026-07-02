---
name: von-karman-plate-state
description: Phase-4 model
metadata: 
  node_type: memory
  type: project
  originSessionId: 4b4bd7cd-a9b8-46a4-bd14-4e7f78d0211d
---

Phase-4 **model #6 — von Kármán nonlinear plate**: **ALL 6 PARTS BUILT & GREEN** (P1–4 2026-07-01,
**P5 diagnostics + P6 free-edge cymbal 2026-07-02**). `VonKarmanBracket` (P1) + `AiryStressSolver`
(P2) in `operators2d.py`; **`VKPlate` coupled resonator (P3)** in `core/plate.py`.
`tests/test_vk_bracket.py` (15) + `test_vk_airy.py` (13) + `test_vk_{energy,modal,stability}.py` (27)
+ **`test_vk_free.py` (12)**, full suite **362** green, ruff clean.

**Part 6 DONE — free-edge cymbal/gong (`VKPlate(boundary="free")`), 2026-07-02.** The plan's
one-liner "only the two boundary operators change" was **WRONG**: the Part-1 bracket is triple
self-adjoint **only for rim-vanishing fields**, but a free plate has `w≠0` on the rim (every node a
free unknown) — so conservation is NOT inherited. **It dissolves anyway** because the Airy `F` is
**still clamped-zero** (`F=0,F,n=0`) regardless of the transverse edge: the free-case energy
telescoping needs only the weaker **swap identity `⟨l(x,F),g⟩=⟨l(x,g),F⟩`** with `F` (not `w`) in the
rim-vanishing slot — which the **existing `VonKarmanBracket` + existing clamped `AiryStressSolver`
satisfy VERBATIM** (advisor-confirmed; empirically pinned in `M:/claud_projects/temp/vk-free-bracket-
probe/`: probe2 swap-identity **1.6e-15**, probe3 real-operator telescoping **uniform-h²→5e-15 / Wa
→fail**, probe4 end-to-end free scheme **drift 1.67e-13**). **The one subtlety = MIXED WEIGHTING:**
mass is `W=Wa` (trapezoidal) but the coupling force pairs under **uniform h²** (`H_mem` is *secretly*
uniform-h² since `wa≡h²` wherever `F`≠0, `F` rim-vanishing → `H_mem=-¼⟨F,l(w,w)⟩_Wa=…_{h²}`).
Concretely: `step()` RHS gains an `h²` (`k²·h²·l/ρ_s`, the mass matrix carries `W`'s h²; /W done by
the A-solve, NOT here); `set_state` `w^{-1}` coupling carries `h²/ρ_s` **AND** the per-node `/W`
divide (the #5b `W⁻¹` pattern; bending accel_term likewise `½k²κ²(K@u0)/w`). **Design: a
`boundary="free"` branch on `VKPlate`** (mirrors `Plate`; NOT a separate class — advisor's pick),
bracket/Airy code paths unforked, all-nodes-live so the live↔full-grid seam is identity.
`nonlinear=False,boundary="free"` is **bit-identical to `Plate(boundary="free")`** (model #5b). Gates
(`tests/test_vk_free.py`): lossless **drift 1.67e-13** @ w≈3e (**membrane 57 %** of E — real free-rim
bracket exercise), small-amp→#5b reldiff **2.8e-13**, **drift∝couple_tol** self-cert, passivity,
non-negativity, Picard converges, **pitch-glide hardening +33 %** by w=3e (eigenmode-IC + FFT-peak;
the crude zero-crossing f0 is too noisy on the multi-modal free plate, and w≫e cascade smears "the"
fundamental → glide test stays ≤3e), **Richardson O(h²) ratio 5.66** (N=24/48/96, cos-IC, pickup
(0.1,0.1) a node on all; degrades at long times as nonlinear phase drift accumulates → 80 steps).
Diagnostics `scripts/diagnose_vk_free_plate.py`: energy breakdown (drift 9.5e-13, membrane 56 %) +
w/F snapshot, **curved-Chladni elastic modes** (eigsh K/W: 20.5/29.8/36.9/53.0×2-degenerate/92.7 Hz),
**tonal-vs-crash** spectrograms — soft w~e sings (+87 % hardened), **hard w~6e crashes** (spectral
centroid 238→1134 Hz, 22 % energy >1 kHz, a real energy **cascade**). **Oversampling honesty
(HANDOFF §8):** w~10e @ 96 kHz *blows up* (Picard non-contractive, 76 k non-converged + overflow —
NOT a cascade); it converges cleanly only at 384 kHz. Diagnostic uses w~6e (0 non-converged) and
**warns on any non-convergence** rather than draw a divergent panel (advisor-style honesty; never
mislabel a blow-up as physics). Also confirmed: net coupling force `Σh²·l(w,F)`=**0 to 4e-17** (the
swap identity with a constant test field → no rigid-body creep; the advisor's low-freq-creep caution
resolved). Struck GIF. Plan doc `von-karman-plate-plan.md` has a full **Part 6**
section. Free-case FD source is NOT Bilbao 2008 (SS-only) — NSS Ch.13 + gong/cymbal papers; empirical
self-cert stands in (not on disk).

**Part 5 DONE — `scripts/diagnose_vk_plate.py` + 3 `viz/plots.py` helpers** (`plot_energy_breakdown`,
`plot_pitch_glide`, `plot_spectrogram`; no dedicated tests — repo grep confirms **no test imports
`physsynth.viz.plots`**, viz is diagnostics-as-visuals verified by eye, consistent w/ the other
`diagnose_*.py`). Four visuals, all eyeballed correct: (a) **lossless energy breakdown** — flat total
riding over **anti-correlated** linear(kinetic+bending)↔membrane(Airy `F`) exchange, drift **8.4e-13**
@ `w≈3e` (membrane **51 %** of E; differs from the test's 57 % only by strike shape/N=24-vs-20 —
benign); (b) **`w/e` sweep** — zero-crossing fundamental (NOT `measure_partials_near`: its ±40 %
window on the *linear* f0 misses the +75 % hardened peak — advisor catch), grid held fixed so the
curve is pure physics, monotone **+75 % by w=5e**, `w→0` lands on SS law (213.4 vs 214.0 Hz); (c)
**`σ=3` ring-down spectrogram** (needs σ>0; lossless=flat pitch) — fundamental **glides down** ~370→214
Hz onto the linear-limit line, **0 non-converged steps**, worst Picard resid 1e-13; (d) bonus `w`+`F`
**stress-field snapshot** at peak membrane (the unique-to-VK visual — F is blue/compressive opposite
the red displacement dome); (e) struck GIF. Console prints drift, membrane frac, glide table, Picard
residuals — judgeable from console alone. Advisor blessed plan + completion.

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
Next: **Part 6 free-edge cymbal** (swap `B→K` = model #5b free stiffness, free F-BC; bracket/F-solve/
energy all carry over — only the two boundary operators change) — the iconic gong/cymbal, the deep
end. (Optional aside: expose VKPlate in the web viewer.)

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
tests [DONE ✅, 27 tests] → (5) diagnostics (energy breakdown + pitch-glide sweep + spectrogram +
stress-field) [DONE ✅ 2026-07-02] → (6) free-edge cymbal (swap B→K + I→W; bracket/F-solve/energy
**reused verbatim** thanks to the clamped-zero `F` + mixed weighting) [DONE ✅ 2026-07-02].
**Model #6 COMPLETE.** Next horizon: HANDOFF §12 (more methods/coupling) or expose VKPlate (both
boundaries) in the web viewer.

**Sources:** Bilbao 2008 "A Family of Conservative FD Schemes for the Dynamical von Kármán Plate
Equations" (Numer. Methods PDEs 24(1):193–216) + NSS 2009 Ch.13 = primary FD source (pin bracket +
F-BC here, don't reconstruct from memory). Ducceschi–Touzé DAFx-15 = continuous eqns (verbatim) +
the `w/e` regimes (linear/pitch-glide/cascade) + free-circular F-BCs; a *modal* route, independent
cross-check. See [[commit-push-at-batch-end]]; mind [[respect-ruff-line-length]].
