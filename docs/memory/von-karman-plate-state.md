---
name: von-karman-plate-state
description: Phase-4 model
metadata: 
  node_type: memory
  type: project
  originSessionId: 4b4bd7cd-a9b8-46a4-bd14-4e7f78d0211d
---

Phase-4 **model #6 — von Kármán nonlinear plate**: **Part 1 (discrete bracket + money test) BUILT &
GREEN (2026-07-01)**; Parts 2–5 still to build. `VonKarmanBracket` in `operators2d.py`,
`tests/test_vk_bracket.py` (15 tests), full suite **309** green, ruff clean.
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
[DONE ✅] → (2) F elliptic solve `B_F` + manufactured-soln check → (3)
coupled `core/plate.py` resonator (keep linear SS/free branches byte-identical) → (4) validation
tests → (5) diagnostics (pitch-glide spectrogram) → (6) Part 2 free-edge cymbal (swap B→K + free
F-BC; bracket/F-solve/energy all carry over).

**Sources:** Bilbao 2008 "A Family of Conservative FD Schemes for the Dynamical von Kármán Plate
Equations" (Numer. Methods PDEs 24(1):193–216) + NSS 2009 Ch.13 = primary FD source (pin bracket +
F-BC here, don't reconstruct from memory). Ducceschi–Touzé DAFx-15 = continuous eqns (verbatim) +
the `w/e` regimes (linear/pitch-glide/cascade) + free-circular F-BCs; a *modal* route, independent
cross-check. See [[commit-push-at-batch-end]]; mind [[respect-ruff-line-length]].
