---
name: free-plate-state
description: Phase-3 model
metadata: 
  node_type: memory
  type: project
  originSessionId: c7e3f34a-89a1-41b0-814a-0c24a661c871
  modified: 2026-08-17T14:23:18.894Z
---

Phase-3 **model #5b — 2D free-edge (FFFF) Kirchhoff plate + curved Chladni figures** built & all
green (**294 tests**, up from 263). `operators2d.free_plate_stiffness(Nx,Ny,h,nu)→(K,W,index_map)`,
`Plate(boundary="free", nu=…)` (W-weighted θ-scheme, splu), `analysis/modal.{free_plate_ffff_square_lambdas,
free_plate_freq_from_lambda}`, `viz/plots.plot_chladni`, `tests/test_free_plate_{modal,energy}.py`,
`scripts/diagnose_free_plate.py`. Plan + build results in `docs/dev/plate-free-edge-plan.md` Part 1.
This is the payoff of the free-free beam de-risk ([[beam-state]]) — the first **2D** free-boundary
flexural resonator, the iconic curved-Chladni plate.

**The advisor's collapse held: the only genuinely new code was the ν-coupling.** The A-vs-B fork
(masked-all-nodes vs tensor) was a red herring — both are rectangular Gram assemblies (no mask; the
Leissa anchor is FFFF-**square**). The real decision was the **edge stencil**, already answered by the
beam: interior-only `D2` + trapezoidal mass, **no normal curvature centered at the free edge**.

**Load-bearing construction (energy-first, build K FROM the strain energy):**
`P(f,g)=∫[f_xx g_xx + f_yy g_yy + ν(f_xx g_yy + f_yy g_xx) + 2(1−ν)f_xy g_xy]dA` (Kirchhoff form;
ν=1 collapses to (∇²)²). Discretized collocated-at-nodes with **one area weight `Wa=kron(m_y,m_x)`**
(the 1D trapezoidal masses → interior h² / **edge h²/2 / corner h²/4 automatically** — the 2D echo of
the beam's h/2 end cells; NO hand-weighting):
`K = C2xᵀWaC2x + C2yᵀWaC2y + ν(C2xᵀWaC2y + C2yᵀWaC2x) + 2(1−ν)·h²·DxyᵀDxy`, **W = Wa**.
- `C2x=kron(I_y, c2x_1d)`, c2x_1d = collocated 2nd-diff, **zero rows at the free edges** (no edge
  curvature). The two bending-diagonal blocks **== `kron(M_y,S_x)+kron(S_y,M_x)`** with
  `S,M=free_beam_stiffness` — i.e. bending IS the validated free *beam* operator per axis (symmetry,
  per-line {1,x} nullspace, O(h²) inherited, re-derived nowhere). White-box test pins this.
- **Twist = cell-centered `Dxy=kron(d1y,d1x)`** (forward diffs, on Nx·Ny cells),
  `(u[i+1,j+1]−u[i+1,j]−u[i,j+1]+u[i,j])/h²` — NOT the collocated centred u_xy, whose checkerboard
  `(−1)^{i+j}` mode would inject a spurious 4th near-zero mode. **Money test: count near-zero
  generalized eigenvalues = exactly 3.**
- **Kron ordering = C-order (j outer, i inner)**, matching `embed`/`index_map`; `kron(y_factor,
  x_factor)`. Verified vs an independent explicit-loop dense assembly (the advisor's belt-and-suspenders).

**Nullspace money test (replaces the SS `B=L²==Λ²`):** K symmetric PSD, nullspace **exactly {1,x,y}**
(bilinear ∩ additively-separable = linear — proven analytically). **`K(xy)≠0` scaling EXACTLY ∝
(1−ν)** (4.0/2.8/2.04 → all 4.0 after ÷(1−ν)) is the dropped-ν catch. ν∈(−1,½) (energy PD; default
0.3). K is PSD ⇒ generalized eigsh `Kφ=μWφ` needs a **negative** shift (`-1e-3·(13/a²)²`); A
`=(1+σk)W+θk²κ²K` SPD because W is. f=κ√μ/2π. Reuses beam's θ-scheme map `discrete_beam_eigenfrequency`.

**Validation (NO closed form → three anchors):** O(h²) self-convergence (Richardson order ≈ 2.1–2.3);
**Leissa FFFF-square anchor matched to 0.01 %** at N=80 (λ=13.467/19.598/24.269/34.803/34.803 vs
table 13.468/19.596/24.270/34.801/34.801, modes 4,5 degenerate → match by **sorted** eigenvalue);
energy drift **1e-13 even at mu=16**; **fundamental = the saddle/twist** (corners alternate sign,
center node — NOT a bulge). Chladni figures textbook: cross → X → ring → S-curves → stripes.
**Cited λ source = Narita (2022) EPI-IJE 5(1):26–36 Table 1 12×12, which IMPROVES on Leissa's classic
1969 monograph** (≈13.49 etc., slightly higher); pin digits, don't trust memory (in
`modal.free_plate_ffff_square_lambdas`). λ=ωa²√(ρ_s/D)=ωa²/κ ⇒ f=λκ/2πa², a=full square side.

**SS branch kept byte-identical** (only the old `boundary="free"`-raises test flipped). Plan test #8
(SS regression) done at the eigen level: generalized `Kφ=μWφ` fed SS ops (K=h²B, W=h²I) → model-#5
spectrum.

**Side fix, APPLIED in a separate commit (not open work):** the portability allowlist test
(`test_stability.py::test_core_dependency_allowlist`) was failing because Windows **pywin32**'s `.pth`
injects `pywin32_bootstrap/pywin32_system32` at interpreter startup into every subprocess. Fixed
to measure the **delta** of modules pulled *by importing the core* (snapshot sys.modules before
import), not the absolute set — still the shipped form. Correct semantics, harmless on Linux CI
(where the .pth never loads, so the test passed there anyway). The alternative floated at the time
(add pywin32 to the allowlist instead) was never taken up and needs no decision — the delta
measurement makes it moot. Allowlist policy itself: [[stiff-string-state]].

**Later correction (2026-08-17, [[free-plate-orthotropic-state]]):** "the fundamental is the
saddle/twist" is a fact about an **isotropic** free plate. Give the plate a grain and the twist races
a cross-grain bending mode, losing below `g_y/g_xy = 1.025`; real spruce sits only 12.6% above that
crossing. Same batch: this plate's `nu` is the isotropic special case of a four-constant grain, and
the free branch's `K` still comes out **bit-identical** through the generalized assembly.

**Both follow-ons DONE:** model #6 — **nonlinear (von Kármán) plate** (the gong/cymbal deep end,
[[von-karman-plate-state]]) — and wiring the SS + free plate into the web viewer (Phase C,
[[web-viewer-state]]). See [[commit-push-at-batch-end]]; mind [[respect-ruff-line-length]]
(I drifted ~28 E501s this batch — write ≤100 in the FIRST draft).
