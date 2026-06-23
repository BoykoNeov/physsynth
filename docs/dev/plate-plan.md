# Plate (Kirchhoff) — Phase 3 Plan (model #5, simply-supported rectangular plate)

> **Status: IMPLEMENTED (2026-06-23).** `core/plate.py` (implicit θ-scheme), biharmonic
> `operators2d.biharmonic_from_mask` (`B = L@L`), oracles `rectangular_plate_freqs` +
> `discrete_plate_eigenfrequency` in `analysis/modal.py`, suite in `tests/test_plate_{energy,modal,
> stability}.py` (231 tests total, all green), diagnostics in `scripts/diagnose_plate.py` (reuses the
> generic 2D field/animation plotters — no new viz code). Build followed the de-risk order below.
> The interactive **web viewer** plate panel + the **free-edge Chladni** plate remain *separate later*
> batches (CLAUDE.md #4: viz depends on a validated core).
>
> **Build results (vs. the plan):**
> - **Money test green to machine precision** — assembled `B = L²` eigenvalues match `Λ_{mn}²` at
>   2.4e-13. **Energy drift 2e-13** even at μ=16 (≫ the explicit bound ¼). **Low modes < 1 cent**
>   (0.15–0.62 c at N=96); end-to-end FFT 0.01 c. All advisor traps held (bending-only, no κ=0 anchor,
>   broad damping caveat, `splu` solver, `Q=κ²Λ²`).
> - **Diagnostic-script bug caught & fixed (not a core bug):** `Ly` is *snapped* to an integer number
>   of cells at construction; the oracle must use the snapped `p.Ly`, not the requested value, or a
>   ~0.26% length mismatch reads as a spurious ~10-cent error. Tests use a square plate (Ly=1.0 snaps
>   to itself) so were unaffected.
>
> **The plate is the composition of the two prior advances** — nothing fundamentally new to build:
> - **Membrane's masked Laplacian** (`operators2d.laplacian_from_mask` → symmetric `L`).
> - **Stiff-string's implicit θ-scheme + "biharmonic via squaring."** The 1D string built its
>   biharmonic as `D2@D2`; the plate builds the 2D biharmonic `∇⁴ = (∇²)²` as **`B = L @ L`**. Because
>   `L` is the *Dirichlet* Laplacian (zero ghost outside the rim), `L²` automatically enforces **both**
>   simply-supported conditions (`u = 0` *and* `∇²u = 0`) and keeps `sin·sin` an **exact discrete
>   eigenvector** — the exact 2D analog of the 1D `D2²` trick (advisor-confirmed).
>
> **Human decision taken (2026-06-23): "SS now, free as follow-on."**
> - **Primary = rectangular, simply-supported (Navier) plate — build & validate now as model #5.**
>   Clean closed-form oracle, `L²` exact eigenvectors, O(h²) convergence, machine-precise eigenvalues,
>   maximal harness reuse. Accuracy-first.
> - **Free-edge "Chladni" plate = a separate later batch, not the validation backbone.** The iconic
>   curved Chladni figures require *free* edges, which have **no closed-form oracle** (validate via
>   `eigsh` + Leissa's tabulated frequency parameters + energy). SS nodal lines are only a plain
>   rectangular grid. So the "visual showpiece" half of HANDOFF row 5 is *deferred* to that follow-on.
> - **Clamped (`u=0, u_x·n=0`) deferred** with free (no clean oracle; ν re-enters).

## Goal

The first **flexural** resonator: a plate whose restoring force is pure bending stiffness (no
tension). The deliverable is again **the resonator + the rig that measures its deviation from
theory** — here the theory is the simply-supported modal law (HANDOFF §5 row 5):

```
f_{mn} = (π/2)·√(D/ρ_s)·[(m/Lx)² + (n/Ly)²]
```

The entire 2D harness from the membrane (masked Laplacian, energy via the same matrix, `embed`,
`inner2d`, `eigsh` eigenvalue money test) and the implicit-scheme machinery from the stiff string
carry over. Because the SS rectangle has **machine-precise eigenvalues**, this is the cleanest 2D
model yet — we hold a **tight** cents bar, not the loosened convergence-rate bar the membrane circle
needed.

## Physics

Transverse displacement `u(x,y,t)` of a thin Kirchhoff plate — **bending only, no membrane tension**:

```
ρ_s·u_tt = −D·∇⁴u − 2σρ_s·u_t      ⇒      u_tt = −κ²·∇⁴u − 2σ·u_t
```

- `D = E·e³ / (12(1−ν²))` — flexural rigidity (`E` Young's modulus, `e` thickness, `ν` Poisson).
- `ρ_s` — areal density (kg/m²).
- **`κ² = D/ρ_s`** — the single stiffness coefficient (units m⁴/s², so `κ` is m²/s — the **same
  convention as the stiff string's `κ`**). One number, matches the math; `(E, e, ν, ρ_s)` and a
  musician-facing macro can wrap it later (HANDOFF §8 parameter mapping).
- `∇⁴ = ∂_xxxx + 2∂_xxyy + ∂_yyyy` — the **biharmonic**. This is the whole operator: **there is no
  `c²∇²` wave term.** (Trap 1: do *not* copy the stiff string's `c²δ_xx + κ²δ_xxxx` shape — the plate
  is `−κ²∇⁴` alone. Confirm against the oracle: `f_{mn}` is pure 4th-power in the wavenumbers, no
  `√(tension·γ + bending·γ²)` shape.)
- `σ ≥ 0` — frequency-independent loss (`−2σu_t`), same bookkeeping as #1/#4. Frequency-*dependent*
  plate loss is deferred (mirrors #3, not needed for #5).

**ν drops out for the SS rectangle.** The plate strain energy is
`(D/2)∫∫[(∇²u)² − 2(1−ν)(u_xx·u_yy − u_xy²)]`; the twisting (Gaussian-curvature) term integrates to a
boundary term that **vanishes for simply-supported** edges. So only `κ² = D/ρ_s` enters the SS model
— one parameter. ν re-enters only for free/clamped edges (boundary moment/shear conditions and the
corner term), and is deferred with them.

Continuous energy (kinetic + bending), `∫∫ over Ω`:

```
E(t) = (ρ_s/2)∫∫(u_t)²  +  (D/2)∫∫(∇²u)²
dE/dt = (boundary flux)  −  2σρ_s∫∫(u_t)²
```

Simply-supported edges kill the boundary flux (`u = 0` and `∇²u = 0` on ∂Ω), so lossless ⇒
`dE/dt = 0`, lossy ⇒ `≤ 0`.

### Modal oracle (simply-supported rectangle `[0,Lx]×[0,Ly]`)

Eigenfunctions are exactly `sin(mπx/Lx)·sin(nπy/Ly)` (the **same** `sin·sin` as the rectangular
membrane — reuse `modal.rectangular_mode_field`), with

```
ω_{mn} = κ·γ_{mn},   γ_{mn} = (mπ/Lx)² + (nπ/Ly)²   (the Laplacian eigenvalue magnitude)
f_{mn} = ω_{mn}/(2π) = (π/2)·√(D/ρ_s)·[(m/Lx)² + (n/Ly)²]
```

The biharmonic eigenvalue is `γ_{mn}²` (Laplacian eigenvalue **squared**). Square rectangles add the
`(m,n)↔(n,m)` degeneracy; match modes by sorted eigenvalue + count, never by a fragile per-peak label.

## Scheme — implicit θ-scheme on the biharmonic (explicit plate CFL is brutal)

Treating `∇⁴` explicitly forces `k ≲ h²/κ` (the `κ²k²/h⁴` term) — unusable at a musical grid. As with
the stiff string (HANDOFF §11.2), go **straight to implicit**: θ-weighted time average of the whole
spatial operator `𝓛 = −κ²·B`, `B = L² ⪰ 0`:

```
δ_tt u = 𝓛·(θ·u^{n+1} + (1−2θ)·u^n + θ·u^{n-1}) − 2σ·δ_t·u
```

**Update (one sparse solve per step):** with `A = (1+σk)·I − θk²·𝓛 = (1+σk)·I + θk²κ²·B`,

```
A·u^{n+1} = 2u^n + (1−2θ)k²·𝓛u^n − u^{n-1} + θk²·𝓛u^{n-1} + σk·u^{n-1}
```

`A` is **symmetric positive-definite and constant in time** → factor once at construction,
back-substitute each step.

> **Solver: `scipy.sparse.linalg.splu` / `factorized`, NOT `cholesky_banded`.** `B = L²` is a
> **13-point stencil** (bandwidth ~2·Nx in the flat ordering), so the 1D pentadiagonal banded
> Cholesky is impractical, and **scipy has no sparse Cholesky** (cholmod lives in scikit-sparse,
> off the dependency allowlist). Use a prefactored sparse LU (`splu(A.tocsc())`) — `A` is SPD so the
> factorization is stable. scipy.sparse(.linalg) is already on the core allowlist; no contract change.

**Unconditionally stable for θ ≥ ¼** (same derivation as #2, with `Q = κ²·Λ²`, `Λ` = Laplacian
eigenvalue magnitude so the biharmonic eigenvalue is `Λ²`):

```
s ≡ sin²(ωk/2) = Q·k² / (4 + 4θ·Q·k²)        →   s ≤ 1  ⇔  Q·k²·(1−4θ) ≤ 4
```

For θ ≥ ¼ the bound is unconditional. **θ default = 0.28** (inherit the stiff string: a hair above ¼,
accuracy-first with a small positivity margin; keep it a constructor param).

### Discrete eigenfrequency oracle (the θ-scheme `f_disc`)

```
f_disc(Λ) = arcsin(√s) / (π·k),   s = Q·k²/(4 + 4θ·Q·k²),   Q = κ²·Λ²
```

> **Trap: `Q = κ²·Λ²` uses the BIHARMONIC eigenvalue.** `Λ` is the *Laplacian* eigenvalue magnitude;
> the biharmonic eigenvalue is `Λ²`, so `Q = κ²·Λ²`. Easy to under-square (`κ²Λ`) or double-square
> (`κ²Λ⁴`). Pin it against the assembled-`B` eigenvalues in the test below.

## Energy (θ-dependent; bending-only potential)

Mirror the stiff string's θ-energy, but with a **single bending term** (Trap 1 again — no wave part):

```
E^n = ρ_s·[ ½‖δ_t⁻ u^n‖²  +  (θ/2)(P_nn + P_pp)  +  (½−θ)·P_np ]
P(f,g) = ⟨−𝓛 f, g⟩ = κ²·⟨B f, g⟩ = κ²·⟨L f, L g⟩  ≥ 0     (B = L², L symmetric ⇒ ⟨L²f,g⟩=⟨Lf,Lg⟩)
```

with the 2D inner product `⟨f,g⟩ = h²·Σ f g` over the live nodes. Evaluating `P` through the **same**
matrix `B` (equivalently `L`) used in the update makes `E^{n+1}=E^n` an exact algebraic identity
(machine-precision lossless; monotone decreasing lossy). At θ=¼ the stabilizer vanishes and `E^n` is
exactly the cross-time form; for θ>¼ the `(θ−¼)` weighting gives a manifest positivity margin. `set_state`
uses the Taylor start `u^{-1} = u^0 − k·v^0 + ½k²·𝓛u^0 = u^0 − k·v^0 − ½κ²k²·(B u^0)`.

### Damping caveat — BROADER than the stiff string (whole spectrum, not just the top)

The θ-scheme makes frequency-independent loss effectively frequency-*dependent*: mode `m` decays at
`2σ·(1 − θ·Q·k²)`, not `2σ`. For the stiff string only the *top* partials hit this (low/mid modes
ride the gentle `c²p²` term). **The plate has `Q = κ²·Λ²` — 4th-power across the ENTIRE spectrum** —
so the under-damping bites **mid-spectrum** too, not just the top. Passivity still holds
unconditionally (`ρ² ≤ 1`); it is the *rate*, not the sign, that is wrong above low modes. This is a
genuine property of the discretization, not a bug. Cure = frequency-dependent loss (a later model),
deferred exactly as the membrane deferred it. Document the σ-path broadband-decay fidelity as a known
(broader) limitation; the lossless physics (energy, modes) is unaffected.

## Boundaries — simply-supported (Navier) first

A 4th-order operator needs **two** conditions per edge.

- **Simply-supported: `u = 0` and `∇²u = 0`.** Primary, validated config. Realized **for free** by
  `B = L @ L` with the membrane's Dirichlet Laplacian (the zero ghost makes `w = Lu` satisfy `w = 0`
  on the rim, so `L(Lu)` enforces both conditions). `sin·sin` stays an exact discrete eigenvector of
  `B` with eigenvalue `Λ²` ⇒ machine-precise eigenvalues, O(h²) continuum, max harness reuse.
- **Free (`∇²u = 0` and the third-order edge/corner condition)** — the true Chladni plate; **no
  closed-form oracle** (ν re-enters; corner conditions; 4th-order SBP on free edges is delicate).
  Validate via `eigsh` eigenmodes + **Leissa's tabulated dimensionless frequency parameters** for a
  free plate + the energy drift test (operator symmetry). The iconic curved nodal figures live here.
- **Clamped (`u=0, ∂u/∂n=0`)** — no simple oracle; deferred with free.

> **Why SS first (accuracy-first):** validation is code against closed-form physics (CLAUDE.md). SS
> gives that cleanly *and* the eigenmode money test below. Free is the visual stretch, sequenced after.

## The biharmonic via `L²` — why it keeps energy exact and modes exact

`B = L @ L` where `(L, index_map) = laplacian_from_mask(rectangle_mask(Nx,Ny), h)` (the membrane's
already-tested operator). `L` is symmetric ⇒ `B = L²` is **symmetric positive-semidefinite** (its
eigenvalues are `Λ² ≥ 0`). Two consequences, exactly paralleling the membrane's "energy ⊥ geometry":

- **Energy conservation needs only symmetry** — `B` symmetric ⇒ `E^{n+1}=E^n` is exact (1e-10 drift).
- **The SS BC is baked in by `L²`** — no hand-coded 13-point boundary rows (the 2D analog of the 1D
  decision to build the biharmonic as `D2@D2` rather than hand-coding "5" rows).

### Operator-assembly money test (replaces the κ=0 anchor)

> **Trap 2: there is NO κ=0 regression anchor.** At κ=0 the plate is `u_tt = 0` (linear drift, not a
> model), so the stiff string's "κ=0 self-consistency" test #7 does **not** carry over. The de-risk
> anchor is instead:
> 1. We **inherit the membrane's already-tested `L`** for free (its symmetry/eigenvalue tests pass).
> 2. Assert the **assembled `B = L²` eigenvalues equal the *squared* known Laplacian eigenvalues
>    `Λ_{mn}²`** to machine precision, where `Λ_{mn} = (4/h²)[sin²(mπ/2Nx) + sin²(nπ/2Ny)]` (already in
>    `modal.rectangular_discrete_eigenvalues`). That proves `B` is assembled right.
> Energy drift + O(h²) convergence finish the correctness argument.

This eigenvalue test is the 2D plate analog of #3's per-mode money test. A complementary end-to-end
FFT test (pluck → pickup spectrum → detected peaks land on the low `f_disc`) validates the actual
time-stepping; single-mode runs use analytic `sin·sin` initial conditions so the field stays one tone.

## Work breakdown (build order de-risks)

**Order:** (1) `B = L²` assembled + eigenvalues == `Λ²` (money test) → (2) energy green at κ>0 →
(3) rectangle modal exact + O(h²), **tight cents** → (4) stability / unconditional / NaN guards →
(5) viz mode-shape + nodal-line heatmaps. Operator-correctness first (the membrane proved `L`; here we
prove `L²`), energy second, then modes. Web viewer + free-edge Chladni are separate later batches.

1. **`core/operators2d.py`** — add `biharmonic_from_mask(mask, h) → (B, index_map)` (build `L` via
   `laplacian_from_mask`, return `B = (L @ L).tocsr()` plus the same `index_map`). Pure NumPy/SciPy;
   document the SS-BC bake-in. Keep the *matrix* here; the *solve* lives in the resonator (purity).
2. **`core/plate.py`** (new) — `Plate` resonator, standard interface (`__init__`, `set_state`, `step`,
   `state`, `energy`, `displacement_at`, `pickup_index_at`). Rectangle SS via `rectangle_mask`. New
   input `kappa` (= √(D/ρ_s)). Build `B`, form SPD `A = (1+σk)I + θk²κ²B`, factor once with
   `scipy.sparse.linalg.splu`/`factorized`; `step()` builds the RHS and back-substitutes. No CFL
   limit (unconditional for θ≥¼) but reject non-physical params and θ∉(0,1]. `state` embeds live
   nodes back to the 2D grid (zeros at rim) via `embed`.
3. **`analysis/modal.py`** — add `rectangular_plate_freqs(kappa, Lx, Ly, max_m, max_n)` (sorted with
   `(m,n)`) and `discrete_plate_eigenfrequency(Lambda_lap, kappa, k, theta)` (the θ-scheme `f_disc`
   with `Q = κ²·Λ²` — single source of truth for the oracle). Reuse `rectangular_mode_field`
   (`sin·sin`) and `rectangular_discrete_eigenvalues` (Laplacian `Λ_{mn}`, square for `B`).
4. **`core/exciter.py`** — reuse `raised_cosine_2d` (built for the membrane); no new exciter.
5. **`tests/helpers.py`** — `make_plate(...)` builder, plate single-mode init (analytic `sin·sin`),
   reuse the 2D frequency/eigenvalue measurement helpers.
6. **`tests/test_plate_energy.py`**, **`test_plate_modal.py`**, **`test_plate_stability.py`** — suite.
7. **`viz/plots.py`** + **`scripts/diagnose_plate.py`** — mode-shape heatmaps (SS nodal lines = a
   rectangular grid; label them honestly, *not* as Chladni curves), detected-vs-oracle partials,
   energy trace, displacement-field animation frames. Writes to `out/`.

## Tests — acceptance criteria (TIGHT tolerances; do NOT loosen like the membrane circle)

1. **Energy conserved (lossless):** drift `< 1e-10` over ≥ 1 s, across a sweep of κ and of (N, fs) —
   **including a coarse grid / large k that the explicit plate scheme could not run** (demonstrates
   the unconditional-stability win). No λ to sweep (unconditional).
2. **Passivity (σ>0):** energy monotone non-increasing. Decay rate matches `2σ` **only for low
   modes**; assert that, and *document* the broad mid-spectrum under-damping caveat above (don't
   assert `2σ` for mid/high modes — it is genuinely wrong there).
3. **Operator-assembly money test:** assembled `B = L²` eigenvalues equal `Λ_{mn}²` (squared known
   Laplacian eigenvalues) to machine precision. Replaces the (nonexistent) κ=0 anchor.
4. **Rectangle modal — exact + convergent, TIGHT bar:** `f_disc` from `B`'s eigenvalues matches the
   `discrete_plate_eigenfrequency` oracle to machine precision; continuum `f_{mn}` error shrinks at
   **O(h²)** under refinement; detected partials within **~1 cent** (1D-style — *not* the loosened
   membrane-circle bar). Because eigenvalues are machine-precise, this bar is justified.
5. **End-to-end FFT (sanity):** pluck the plate, detected pickup peaks land on the low `f_disc`.
6. **Stability / unconditional:** NaN-free over κ and (N, fs) sweeps including a config that would
   violate the *explicit* plate CFL (runs stably here); non-physical params and θ∉(0,1] rejected.
7. **Portability:** auto-covered — `test_stability.py`'s core sweep iterates every `core/` submodule,
   so `plate.py` is checked against the headless/allowlist/no-sibling guards with no edits. Confirm
   scipy.sparse(.linalg) stays within the allowlist (it does).

## Open decisions (surface to the human; recommendation in **bold**)

- **Boundary for the milestone: simply-supported.** Clean Navier oracle, `L²` exact eigenvectors,
  tight tolerance, max harness reuse — accuracy-first. **Free-edge Chladni = a visual follow-on**
  (eigsh + Leissa tables + energy, no closed form); clamped deferred with it. *This is the one real
  fork — confirm before building.* Honest tradeoff: SS nodal lines are a plain rectangular grid, so
  the iconic curved Chladni figures wait for the free-edge follow-on.
- **Stiffness input: `κ = √(D/ρ_s)` as the core input** (one number, matches the math, mirrors the
  stiff string), with `(E, e, ν, ρ_s)` and a musician-facing macro wrapping it later.
- **θ: default 0.28** (inherit the stiff string), constructor param; compare ¼-ish vs ½ against the
  dispersion oracle empirically during implementation.
- **Tolerances: TIGHT** — drift < 1e-10, partials ~1 cent, eigenvalues machine-precise. Do **not**
  loosen to a convergence-rate bar (that was only for the *staircased* membrane circle; the SS
  rectangle has no staircase error).

## Traps (pre-flagged)

- **Bending-ONLY.** No `c²∇²` wave term in the operator *or* the energy. Copying the stiff string's
  `c²δ_xx + κ²δ_xxxx` operator (or its `c²⟨∇f,∇g⟩` energy term) is the top trap. The oracle's pure
  4th-power shape is the tell.
- **No κ=0 anchor.** κ=0 ⇒ `u_tt = 0` (degenerate). Use the `B = L²` eigenvalue == `Λ²` money test
  for operator correctness, not a κ=0 reduction.
- **Damping caveat is broader.** `Q = κ²Λ²` is 4th-power across the *whole* spectrum → mid-spectrum
  under-damps, not just the top partials. Don't assert `2σ` decay above low modes.
- **Solver: `splu`/`factorized`, not `cholesky_banded`.** 13-point stencil, bandwidth ~2Nx; scipy has
  no sparse Cholesky. Factor `A` (SPD) once via sparse LU.
- **`Q = κ²·Λ²` uses the biharmonic eigenvalue.** `Λ` is the Laplacian eigenvalue; the biharmonic is
  `Λ²`. Don't under-square or double-square — pin it to the assembled `B` eigenvalues.
- **ν drops out for SS rectangular** (twisting term integrates to a vanishing boundary term); one
  parameter `κ` suffices. ν re-enters only for free/clamped — deferred.
- **Don't loosen the cents bar.** Machine-precise eigenvalues justify a tight 1D-style tolerance; the
  membrane-circle loosening was a staircase artifact that does not exist here.
- **`B` (hence `A`, the LU factor) is constant only while geometry/params are fixed** — any future
  parameter setter must rebuild `B` and re-factor `A`.
- **Energy drift triangulation:** right frequency + wrong drift ⇒ the **energy measure** is wrong
  (a spurious wave term, or a missing θ-stabilizer); wrong frequency ⇒ the **operator** is wrong
  (`B` assembly, sign, or `Q` mis-squared). Never relax the tolerance (HANDOFF §8).
