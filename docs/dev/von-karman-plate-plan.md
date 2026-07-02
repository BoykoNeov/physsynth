# Nonlinear (von Kármán) plate — Phase 4 Plan (model #6)

> **Status: Parts 1–3 BUILT & GREEN (2026-07-01).** Part 1 = discrete bracket + money test
> (`VonKarmanBracket`, `tests/test_vk_bracket.py`, 15 tests). Part 2 = Airy stress-function elliptic
> solve `B_F` (`AiryStressSolver`, `tests/test_vk_airy.py`, 13 tests) — **clamped** `F = 0, F,n = 0`
> BC (human decision #2, 2026-07-01), energy-first `B_F = Lc_rᵀ Wa Lc_r`, `splu`-prefactored,
> manufactured-solution O(h²) + clamped-vs-Navier discriminator + the `VonKarmanBracket → solve` seam
> (F ∝ ‖w‖², the 4× doubling check). **Part 3 = the coupled `VKPlate` resonator** (`core/plate.py`,
> materials surface `(E, e, ν, ρ)` per human decision #3) — conservative θ-scheme + Picard-iterated
> `l(μ_{t·}w, μ_{t·}F)` coupling, `tests/test_vk_{energy,modal,stability}.py` (27 tests). Lossless
> drift **2.6e-13** at `w≈3e` (membrane 57 % of `H`), drift∝`couple_tol` self-cert, passivity exact,
> small-amp→#5 (reldiff 0), pitch-glide hardening (+74 % at `w=5e`), Richardson O(h²) (4.40),
> `nonlinear=False` **bit-identical** to model #5. All in `operators2d.py` / `plate.py`; full suite
> **349** green, ruff clean. See "Part 1/2/3 — done" below. Human decisions taken
> (2026-06-30): **(1) SS-first de-risk, then free-edge follow-on** ✅; **(2) core parameter surface
> `(kappa, E, e, nu)`** ✅ (derive `D`, membrane coeff `Ee`, `κ`); **(3) human reviews this doc before
> any code is written** ✅. Still to pin *from the source at implementation*: the discrete-bracket
> stencil and the SS `F` in-plane BC (Bilbao NSS Ch. 13 / Bilbao 2008). The deep end of HANDOFF §5
> (row 6) and Phase 4 (§9): the first
> model with **genuine nonlinear coupling** and **no analytic modal oracle**, so the energy method
> stops being a cross-check and becomes *the* correctness tool (HANDOFF §4.5 — von Neumann analysis
> does not apply to nonlinear systems; energy analysis does). Built **energy-first / conservative**
> per the Bilbao framework (CLAUDE.md #2). Sonic payoff: gongs, cymbals, the amplitude-dependent
> *pitch glide* and the high-amplitude *crash/shimmer* cascade.
>
> **Advisor-confirmed direction (load-bearing):**
> - **Airy-stress-function form** of the Föppl–von Kármán system (transverse `w` + stress function
>   `F`), **not** an in-plane-displacement `(u, v)` formulation. One extra field, one extra elliptic
>   solve per step, and the conservative structure is cleanest.
> - **SS-first as the de-risk milestone; free-edge as the follow-on that delivers the iconic sound.**
>   This mirrors the project's beam→free-plate de-risk culture. *Flagged to the human* — "model #6"
>   in their mind may be cymbals, which are *free* plates; the SS first deliverable will sound like a
>   nonlinear tom/drum with pitch glide, not a cymbal. See Open Decisions.
> - **The discrete von Kármán bracket is the operator money test.** Energy conservation rests on the
>   discrete trilinear form being *fully* symmetric ("triple self-adjointness"). Unit-test it
>   standalone **before any time loop** — it is the analog of `B = L² == Λ²` / `nullspace = {1,x,y}`,
>   and (unlike those) the crux has **no 1D analog**, so this *is* the minimal de-risk.
> - **Two factorizations, not one.** The `w` time-step inverts `A` (θ-scheme, model #5/#5b); the `F`
>   elliptic solve inverts a biharmonic `B_F`. Different matrices — prefactor each once with `splu`.
> - **The energy-drift test must run at LARGE amplitude** (`w ≳ h`) so the membrane term is ≳10–20 %
>   of the total — a bracket bug hides when membrane energy is negligible (it just re-tests the
>   linear scheme).
> - **Do not assume unconditional stability transfers.** The linear θ-scheme is unconditional; the
>   coupled scheme is not automatically. Check energy *non-negativity* (not just conservation), and
>   expect a possibly amplitude-dependent step bound to re-derive.

## Why this is genuinely a new model (not a template reuse)

Models #1–#5b were all *linear*: one constant operator (`B`, `K`), constant `A = (1+σk)·M + θk²κ²·op`,
factor once, march. Everything testable against closed-form modes. **None of that survives the
nonlinearity:**

- **Two coupled fields.** Transverse `w` *and* the Airy stress function `F`. Each step: solve an
  elliptic (biharmonic) problem for `F` from the current `w`, then advance `w` with the `F`-coupling.
- **No analytic modal oracle.** The spectrum is amplitude-dependent (the whole point). The tight
  cents bar is gone; correctness anchors on **energy conservation through the nonlinearity**, the
  **small-amplitude limit recovering model #5's linear modes**, **Richardson self-convergence**, and
  the **qualitative pitch-glide / cascade** signatures.
- **Energy conservation is no longer automatic from operator symmetry.** For the linear models a
  symmetric operator gave conservation for free. Here it requires the *discrete bracket* to satisfy a
  trilinear symmetry identity — a genuinely new, non-obvious algebraic property to construct and test.
- **Oversample around the nonlinearity** (HANDOFF §8 — aliasing). The quadratic/cubic coupling folds
  HF energy down; the cascade regime especially needs headroom. Render at high `fs`.

## Goal

A simply-supported rectangular **von Kármán** plate resonator + the rig that proves it conserves
energy *through the nonlinearity* and reproduces the linear plate in the small-amplitude limit. The
deliverable is, as always, **the resonator + the rig that measures its deviation from theory** —
here the "theory" is (a) the conservation law itself (machine precision), (b) model #5's linear
modes as `w → 0`, and (c) the O(h²) self-convergence order. Free-edge gong/cymbal is the follow-on
(Part 2) that turns this into the iconic instrument.

---

## Physics — the Föppl–von Kármán plate (Airy-stress-function form)

Transverse displacement `w(x, y, t)`; **Airy stress function** `F(x, y, t)` carrying the in-plane
membrane stress (`σ_xx = F_yy`, `σ_yy = F_xx`, `σ_xy = −F_xy`). The "simple" von Kármán system
(longitudinal inertia neglected — Bilbao 2008; verbatim from Ducceschi–Touzé DAFx-15):

```
rho_s w_tt = -D ∇⁴w + L(w, F) + p(x,t) - 2 rho_s sigma w_t         (transverse, dynamic)
∇⁴F        = -(E e / 2) L(w, w)                                     (in-plane, elliptic — solve for F)
```

with `rho_s = rho·e` the areal density (`e` = thickness), `D = E e³ / (12(1−ν²))` the flexural
rigidity, and the **von Kármán / Monge–Ampère bracket** (symmetric in its two arguments):

```
L(alpha, beta) = alpha_xx beta_yy + alpha_yy beta_xx - 2 alpha_xy beta_xy
               = Δalpha Δbeta - ∇∇alpha : ∇∇beta
```

- `L(w, F)` is the **restoring force from membrane stretching** — geometric stiffening. It vanishes
  as `w → 0` (`F` is quadratic in `w`, so `L(w,F)` is cubic in `w`): the model reduces to the linear
  Kirchhoff plate #5. **This limit is a headline test.**
- `L(w, w) = 2(w_xx w_yy − w_xy²)` is **twice the Gaussian curvature** — the source that builds `F`.
- **Amplitude scale = the thickness `e`.** `w ≪ e`: linear (model #5). `w ≈ e`: weakly nonlinear,
  amplitude-dependent frequency = **pitch glide / pitch-bend**. `w > e`: strongly nonlinear, energy
  **cascade** large→small scales = crash/shimmer (cymbal). (Ducceschi–Touzé §2.) This sets the test
  amplitudes: drive `w ≳ e` to exercise the bracket; quote `w/e`.

### The conserved energy (the spine)

Total energy = kinetic + **bending** potential + **membrane** potential:

```
H = (rho_s/2)‖w_t‖²  +  (D/2)·U_bend(w)  +  (1 / (2 E e))·‖∇²F‖²
            kinetic          bending                membrane (stored in-plane stretch)
```

- `U_bend(w)` is **exactly model #5/#5b's bending form** — `‖∇²w‖²` for SS (`B = L²`), the full
  ν-dependent Kirchhoff form for free edges (`K`). Reused verbatim.
- The **membrane term `(1/2Ee)‖∇²F‖²`** is new. With `F` solved from `w` each step, it is the energy
  stored in stretching. Driving `w ≳ e` makes this ≳10–20 % of `H` — *that* is the regime the
  energy-drift test must run in (advisor point 5).
- Dissipation: `dH/dt = −2σ rho_s ‖w_t‖² ≤ 0` (passive). Lossless (`σ=0`): `dH/dt = 0`.

**Why conservation is non-trivial.** `d/dt[bending+kinetic]` produces `⟨L(w,F), w_t⟩`;
`d/dt[membrane]` produces `−(1/Ee)⟨∇²F, ∇²F_t⟩`, and `∇⁴F_t = −Ee·L(w,w_t)` so this is
`+⟨L(w,w_t), F⟩` up to the inverse biharmonic. These two cancel **iff** the trilinear form
`T(a,b,c) = ⟨L(a,b), c⟩` is symmetric under swapping its arguments (`⟨L(w,F),w_t⟩ = ⟨L(w,w_t),F⟩`).
The continuous `T` is fully symmetric for the right BCs; **the discrete scheme conserves iff the
discrete `l(·,·)` reproduces that symmetry exactly.** This is the crux.

---

## The conservative scheme

### Discrete bracket `l(α,β)` — pin from Bilbao NSS Ch. 13; reuse existing twist operator

The discrete bracket must satisfy the **discrete triple self-adjointness** identity

```
⟨ l(a,b), c ⟩ = ⟨ l(a,c), b ⟩ = ⟨ l(c,b), a ⟩      for all grid fields a, b, c
```

to machine precision. Bilbao's construction (NSS §13) combines collocated straight second
differences with a **cell-centered** mixed difference, averaged back to nodes, so the trilinear form
is exactly symmetric. **The cell-centered twist `Dxy = kron(d1y, d1x)` already in
`operators2d.py`** (built for the free plate, deliberately *not* the collocated centred `u_xy` whose
checkerboard mode would break things) is the natural building block for the `−2 α_xy β_xy` term —
the same operator, a second use site. **Pin the exact stencil + averaging (`μ` operators) from NSS
Ch. 13 at implementation; do not reconstruct the collocation from memory** (advisor; same discipline
as the free-plate corner stencils). One implementation of `l(·,·)`, two call sites:
`l(w,w)` (F-source) and `l(w,F)` (coupling) — sharing the function is mandatory or the identity
breaks.

### Time stepping — F-solve then θ-scheme w-update, two factorizations

```
1. Solve the F elliptic problem from w^n:   ∇⁴F^n = -(E e / 2) l(w^n, w^n)
      → F^n = B_F⁻¹ rhs_F          (B_F prefactored once with splu)
2. Advance w with the bending θ-scheme + the membrane coupling l(w^n, F^n):
      A w^{n+1} = [θ-scheme bending+inertia RHS, model #5] + k² l(w^n, F^n)
      → w^{n+1} = A⁻¹ rhs_w        (A prefactored once with splu — the SAME A as model #5)
```

- `A = (1+σk)·M + θk²κ²·B_bend` is **model #5's matrix verbatim** (SS: `M=I` scalar `h²`,
  `B_bend = B = L²`). `B_F` is a **separate** biharmonic for the stress function — different BCs,
  different matrix, its own `splu`. "One factorization serves both" is the trap (advisor point 2).
- **DEVIATION (Part 3, as built — and why):** the plan above sketched an *explicit* `l(w^n, F^n)`
  coupling ("per-step solve stays linear, no Newton") with the implicit variant as a "fallback." The
  build went **straight to implicit Picard** — deliberately, not by oversight. Reason (advisor-derived
  & confirmed): the membrane potential is a **quartic** in `w`, so *any* frozen-coefficient explicit
  coupling conserves only to `O(k²)` truncation, never the project's `< 1e-10` machine bar. Exact
  discrete conservation *requires* the new level (`H_mem^{n+1}` contains `w^{n+1}`), hence some
  implicitness is unavoidable. The averaging that closes the identity is the **symmetric two-level**
  form `l(μ_{t·}w, μ_{t·}F)` with `μ_{t·}g = (g^{n+1}+g^{n-1})/2` and `F^m` solved from `w^m`; its
  trilinear form telescopes **exactly** to `-(H_mem^{n+1}-H_mem^{n-1})` via triple self-adjointness
  (derived from first principles, since the Bilbao FD PDF is not on disk — the energy-drift test is
  the self-cert). The step is solved by **fixed-point (Picard)** iteration on the *prefactored* `A`
  and `B_F` (predictor `2w^n - w^{n-1}`; ≤11 sweeps at `w≈e`), converging on `‖Δw‖/‖w‖ ≤ couple_tol`.
  The energy is reported half-step-averaged (`E_lin + ½(H_mem^{n+1}+H_mem^n)`) to kill a spurious
  odd/even oscillation. The cross-time `ψ`/SAV variant that would linearise the step is documented as
  a future optimisation only (it trades the prefactored `A` for a dense per-step operator).
- **Stability is not free.** Re-derive the energy-non-negativity bound for the coupled scheme; it may
  be amplitude-dependent. The fully-conservative implicit variant (Bilbao 2008) extends the linear
  unconditional bound to the nonlinear system at the cost of an implicit (iterative) step — keep that
  as the fallback if the explicit-coupling bound is too tight in the cascade regime. Start with the
  explicit-coupling form (simplest), escalate only if the energy test forces it.

### Boundary conditions (the advisor's blocking decision — pin & state, don't inherit)

- **`w` (transverse): simply supported** — `w = 0`, `Δw = 0` (Navier). Reuses model #5's `B = L²`
  exactly (Dirichlet Laplacian squared bakes both conditions in).
- **`F` (in-plane / stress function): a SEPARATE choice that `B = L²` would silently impose.** The
  self-consistent SS pairing `F = 0, ΔF = 0` (vanishing stress function and stress trace at the edge
  — an in-plane "movable" / stress-relieved edge) makes `B_F = L²` too, so the existing Dirichlet
  machinery is reused for both factorizations. **But this is a *modeling* choice, not a free
  inheritance** — Bilbao's canonical SS von Kármán case must be checked: if it uses `F = 0, F_n = 0`
  (clamped biharmonic), `B_F` is a genuinely different 13-point operator and must be built. **Action:
  pin Bilbao's SS `F`-BC from NSS Ch. 13 / Bilbao 2008 before assembling `B_F`; state it in code +
  this doc; surface to the human.** Note the nonlinearity is **active either way** — the F-solve is a
  *forced* biharmonic (source `l(w,w) ≠ 0`), so membrane tension builds regardless of the BC; the BC
  sets *which* in-plane physics and *how strong* the glide, not on/off.

---

## Operator money test — discrete bracket triple self-adjointness (replaces `B=L²==Λ²`)

Before any time loop, on **random** grid fields `a, b, c` (and across grid sizes):

1. **Symmetry of arguments:** `l(a,b) == l(b,a)` to machine precision (the bracket itself).
2. **Triple self-adjointness:** `⟨l(a,b),c⟩ == ⟨l(a,c),b⟩ == ⟨l(c,b),a⟩` to machine precision —
   **the single algebraic identity the whole model's energy conservation stands on.**
3. **Linear-data annihilation:** `l(a, linear) == 0` and `l(linear, b) == 0` (the bracket is built
   from second derivatives — any field linear in `x, y` has zero contribution), the "not everything
   survives" counter-check.
4. **Small-amplitude consistency:** `l(w,F)` with `F` from the linear elastic response → vanishes
   like `w³` as the amplitude → 0 (so the model collapses to #5). Spot-check the scaling.

If #2 fails, **stop** — the time loop cannot conserve energy. This is the gate.

## Validation oracles (energy-method-first; no closed form for the nonlinear spectrum)

| # | Test | Bar | What it catches |
|---|------|-----|-----------------|
| 1 | Bracket triple self-adjointness | machine precision (random fields) | the conservation crux — **first gate, pre-time-loop** |
| 2 | **Lossless energy drift at LARGE amplitude** (`w ≳ e`, membrane ≳10–20 % of `H`) | `< 1e-10` over ≥ 1 s | the headline — a bracket/averaging bug that hides at small amplitude |
| 3 | Energy **non-negativity** `H^n ≥ 0` + NaN-free | all `n`, across the step-bound | coupled-scheme stability (not inherited from the linear θ-scheme) |
| 4 | **Small-amplitude → linear model #5** | `w/e ≲ 1e-3`: partials match `f_mn=(π/2)κ[(m/Lx)²+(n/Ly)²]` to model-#5 tolerance | the nonlinear term correctly vanishes; ties back to #5 |
| 5 | **Pitch glide (hardening)** | fundamental rises monotonically with drive amplitude; quote Δcents vs `w/e` | the signature weakly-nonlinear behaviour is present & correct sign |
| 6 | Richardson self-convergence | low-frequency content / energy converges at **O(h²)** under `h→h/2` | scheme is the order it claims (free-plate had this) |
| 7 | Passivity (σ>0) | `H` monotone non-increasing | loss is dissipative, not generative |
| 8 | SS-regression (`l ≡ 0`) | bypassing the coupling reproduces model #5 bit-for-bit | the linear path is untouched |
| 9 | Params / portability | reject non-physical inputs; `core/` headless allowlist sweep | conventions |

- **Pitch glide** is the qualitative-but-characteristic nonlinear oracle. A struck plate's
  fundamental should *rise* with amplitude (geometric hardening) and relax toward the linear value as
  it decays — audible and measurable (track the instantaneous fundamental over a decaying strike).
- **Cascade (`w > e`)** is a *qualitative* showcase for Part 2 (free-edge cymbal), not a pass/fail
  test — energy conservation (#2) is what makes it trustworthy.

---

## Work breakdown (de-risk order — each gate green before the next)

1. **Discrete bracket `l(·,·)` + its test (`operators2d.py`, `tests/test_vk_bracket.py`).** Pin the
   stencil from NSS Ch. 13; reuse the cell-centered `Dxy`. **Gate: triple self-adjointness +
   linear-annihilation to machine precision.** No time-stepping yet. *This is the whole de-risk.*
   **✅ DONE (2026-07-01).** The stencil was **pinned empirically** (advisor: the money test is a
   self-certifying gate, so any construction that passes it *plus a consistency check* is correct)
   rather than transcribed from the book, then confirmed to match Bilbao's cell-centered-twist form.
   Decisive probe results (`M:/claud_projects/temp/vk-bracket-probe/`):
   - **Straight collocated terms + centered twist = O(1) asymmetric** (0.5–0.8); every naive
     combination fails. **Straight + *cell-centered* (forward-forward) twist, product scattered back
     to nodes by the corner-average adjoint = triple self-adjoint to 1e-15.** The twist's asymmetry
     *exactly cancels* the straight terms' — Bilbao's actual contribution, reproduced.
   - **Domain contract discovered & tested:** the cancellation is a discrete SBP identity with **no
     leftover boundary term only when the fields vanish on the rim** (the SS `w = F = 0` edge). On
     non-zero borders the form is O(1) asymmetric — *expected, not a bug*, and the reason SS is the
     natural first home. The money test therefore uses rim-vanishing random fields (a naive
     full-grid-random test would have *failed a correct bracket*).
   - **Consistency added (advisor's key catch — `l ≡ 0` passes symmetry):** `l(a,b)` converges to
     the analytic bracket at **O(h²)** (rate 2.00→1.98) on smooth manufactured data.
   Field form: `l(a,b) = (δxx a)(δyy b) + (δyy a)(δxx b) − 2·Aᵀ[(Dxy a)(Dxy b)]`, reusing
   `_forward_d1_1d` for `Dxy` and new `_centered_d2_1d` / `_avg_d1_1d` 1D pieces.
2. **F elliptic solve (`B_F`, prefactored `splu`).** Pin & implement the SS `F`-BC. **Gate: `∇⁴F =
   source` solves; `F → 0` as `w → 0`; a manufactured-solution check (`F` of known curvature).**
   **✅ DONE (2026-07-01).** BC **pinned & resolved with the human = clamped `F = 0, F,n = 0`** (the
   physically-correct SS-*movable* edge, DT DAFx-15 §4.2 Eq. 11 `F,tt = F,nt = 0`; the plan's earlier
   `L²`/Navier default is a *different, nonstandard* edge — see Open Decisions #2). `AiryStressSolver`
   in `operators2d.py`, `tests/test_vk_airy.py` (12 tests). Built **energy-first** (advisor-confirmed,
   simpler than a hand-assembled 13-point stencil): `B_F = Lc_rᵀ Wa Lc_r`, squaring the **clamped
   Laplacian** `Lc = kron(iy, c2c_x) + kron(c2c_y, ix)` where `c2c` = `_clamped_d2_1d` (the
   `_centered_d2_1d` end rows with the off-diagonal **doubled 1→2**: ghost mirror `F_{-1}=F_1` from
   `F,n=0`). `Wa` = the trapezoidal area weight (reused from `free_plate_stiffness`) — **load-bearing,
   not polish**: it makes the 1D Gram reproduce the textbook clamped biharmonic exactly (diag
   `7,6,…,6,7`, off `-4,1`); `Wa=I` gives a wrong `9`. Symmetric SPD by construction (Gram; clamping
   ⇒ **empty** nullspace, no `{1,x,y}`), `splu`-prefactored. Representation: full-grid `Lc`, drop rim
   **columns** (`F=0`) keep all **rows**; `solve(source_full)` restricts → `Wa`-weighted Galerkin
   load (interior weight `h²`) → solve → embed rim=0. **The subtle bug** (advisor): the manufactured
   RHS must be `Wa`-weighted (`B_F F = Wa·∇⁴F`, not `∇⁴F`) — else O(1) error against a fine operator.
   **The discriminator** (proves clamped, *not* `L²`): manufactured `F=(1−cos)(1−cos)` has `F=F,n=0`
   but `ΔF≠0` on the rim, so the clamped solve recovers it at O(h²) (rates 2.01→2.00) while the
   SS/Navier `biharmonic_from_mask` solve *saturates* at O(1) error (ratio grows 185→2941 with `N`).
   Part-3 membrane energy is the plain quadratic `(1/2Ee)·Fᵀ B_F F` (`Wa` folded in;
   `laplacian_norm_sq`). Empirically de-risked in `M:/claud_projects/temp/vk-Bf-probe/` first
   (caught: my analytic `∇⁴F` sign — `g''''=−a⁴cos`, both `xxxx`/`yyyy` terms negative).
3. **Coupled resonator (`core/plate.py`, `boundary="supported"` + `nonlinear=True` or a `VKPlate`).**
   F-solve → θ-scheme `w`-update with `l(w,F)`; energy with the membrane term. **Keep the linear SS
   and free branches byte-identical** (regression). **Gate: lossless energy drift `< 1e-10` at large
   amplitude; non-negativity; small-amplitude → #5.**
   **✅ DONE (2026-07-01).** New **`VKPlate`** class (model #5's `Plate` left untouched; `nonlinear=False`
   is **bit-identical** to `Plate(boundary="supported")`, the regression). Materials surface
   `(E, e, ν, ρ)` (human decision #3) → derive `ρ_s=ρe`, `D=Ee³/(12(1-ν²))`, `κ=√(D/ρ_s)`, `Y=Ee`.
   Conservative **Picard-iterated** implicit scheme (see the DEVIATION note above): predictor
   `2w^n-w^{n-1}`, each sweep one prefactored `B_F`-solve (`F^{n+1}`) + one `A`-solve (with
   `+k²l(μ_{t·}w,μ_{t·}F)/ρ_s`), `couple_tol=1e-13`; the **live↔full-grid seam** (`embed`→bracket/Airy
   →restrict) bridges the interior `w`-state and the full-grid bracket/`F`. `converged`/`last_residual`
   exposed so a cascade run sees silent non-convergence. Half-step-averaged energy. Empirical gates
   (all green): drift **2.6e-13** at `w≈3e` (**membrane 57 %** of `H` — a real bracket exercise, not a
   linear re-test), **drift ∝ couple_tol** (1e-4→3.5e-5, 1e-12→9.3e-13 — the machine-precision
   self-cert), non-negativity, passivity exact, `w→0`→model-#5 fundamental (reldiff 0).
4. **Validation (`tests/test_vk_{energy,modal,stability}.py`, `analysis/`).** Large-amplitude drift,
   small-amplitude modal recovery, pitch-glide measurement, Richardson, passivity, params.
   **✅ DONE (2026-07-01)** alongside Part 3 (27 tests): energy (drift, non-negativity, drift-vs-tol
   self-cert, passivity, regression, component accessors, Picard convergence), modal (small-amp→#5
   trajectory + fundamental, **pitch-glide hardening** monotone +74 % by `w=5e`), stability
   (**Richardson O(h²)** ratio 4.40 on N=24/48/96 smooth-IC, derived materials, `Ly`-snap, param
   validation). `analysis/` diagnostics deferred to Part 5.
5. **Diagnostics (`scripts/diagnose_vk_plate.py`, `viz/`).** Energy trace (with the membrane
   component broken out), spectrogram showing the pitch glide, displacement animation, `w/e` sweep.
   **✅ DONE (2026-07-02).** `scripts/diagnose_vk_plate.py` + three `viz/plots.py` helpers
   (`plot_energy_breakdown`, `plot_pitch_glide`, `plot_spectrogram`): (a) lossless energy breakdown —
   flat total over anti-correlated linear/membrane exchange, **drift 8.4e-13** at `w≈3e`
   (membrane 51 % of `E`); (b) `w/e` sweep — hardening fundamental **+75 % by `w=5e`**, sitting on
   the SS linear limit (213.4 vs 214.0 Hz) at `w→0`, grid held fixed; (c) `σ=3` ring-down
   spectrogram — the fundamental **glides down** from ~370 Hz onto the 214 Hz linear limit
   (0 non-converged steps); (d) bonus `w`+Airy-`F` stress-field snapshot at peak membrane energy;
   (e) struck-plate GIF. Console prints drift, membrane fraction, glide table, Picard residuals.
6. **(Part 6, follow-on) Free-edge cymbal** — swap `B_bend → K` (model #5b free stiffness, `I → W`
   lumped mass) and reuse the nonlinear machinery. The one-line "only the two boundary operators
   change" was **wrong** — see the dedicated Part-6 section below. **✅ CRUX DE-RISKED (2026-07-02,
   empirically in `M:/claud_projects/temp/vk-free-bracket-probe/`); build pending human review of
   the plan section below.**

---

## Part 6 — free-edge cymbal (the plan, de-risked before code)

**The trap the one-liner hid.** The discrete bracket's triple self-adjointness — the entire
conservation spine — holds **only for rim-vanishing fields** (the Part-1 domain contract). A free
plate has `w ≠ 0` on the rim (every node is a free unknown), so "the bracket carries over" is *not*
inherited. This is the one genuinely-new piece of Part 6, and the advisor confirmed it is the whole
risk. It was resolved **empirically first** (Part-1/Part-2 discipline), before any repo code.

**The resolution — the crux dissolves because the Airy `F` is still clamped-zero.** The scheme never
needs full triple self-adjointness on arbitrary fields. Working through the free-case energy
telescoping, the coupling work is `⟨l(μ_{t·}w, μ_{t·}F), δ_{t·}w⟩` and it must equal
`-(H_mem^{n+1} - H_mem^{n-1}) = -4c·⟨l(μ_{t·}w, δ_{t·}w), μ_{t·}F⟩`, so the **only** identity
required is the **swap of arguments 2↔3**

    ⟨ l(x, F), g ⟩  ==  ⟨ l(x, g), F ⟩        (x, g arbitrary free fields; F rim-vanishing)

with `x = μw`, `g = δw` **arbitrary (nonzero on the rim)** and `F = μF` **rim-vanishing** — because
the Airy stress function is clamped (`F = 0, F,n = 0`) *regardless* of the transverse edge. `w` never
occupies the rim-vanishing slot; only `F` does. Empirical results (`probe2.py`, machine precision =
`1.6e-15`):

- The **existing `VonKarmanBracket`** (centered `δ_xx`, uniform-`h²` inner product) already satisfies
  the swap identity to `1.6e-15` when `F` is rim-vanishing (`F = 0` on the rim). No new stencil.
- Under **Wa (trapezoidal)** weighting the identity *fails* (`0.42`); it holds only under **uniform
  `h²`**. So the coupling inner product is uniform `h²`, exactly as in the SS case.

**Why the mixed weighting is right (and forced, not a coincidence).** `wa` equals `h²` everywhere
except the rim/near-rim. The Airy `F` is clamped-zero there, so the membrane energy is *secretly*
uniform-`h²`:

    H_mem = (1/2Y) Fᵀ B_F F = -(1/4)⟨F, l(w,w)⟩_Wa = -(1/4)⟨F, l(w,w)⟩_{h²}

(the two inner products agree whenever one factor is rim-vanishing). The coupling work must then match
that same uniform-`h²` pairing under the swap identity — which is exactly the pairing that holds. So
`AiryStressSolver` stays **Wa-weighted internally** (its clamped-biharmonic fidelity depends on `Wa`,
Part 2) while the coupling force pairs under **uniform `h²`**; the mismatch is confined to the rim,
where `F = 0` kills it. This mixed weighting is load-bearing.

**End-to-end certification (`probe4_endtoend.py`, the seam `probe2/3` do not cover).** `probe3.py`
isolates *coupling ↔ membrane* telescoping with the real operators (uniform `h²` → exact `5e-15`; Wa
→ `0.14`–`0.58`). `probe4_endtoend.py` then runs the **full** free scheme (free `K/W` bending + reused
bracket + reused clamped Airy + Picard loop) and measures `E_lin + H_mem` drift at `w ≈ 3e`: **drift
`1.67e-13`** (`H_mem` = 57 % of `E` at `t = 0`, worst Picard residual `9.9e-14`, converged). The
conservation mechanism is certified before repo code.

**What actually changes (and what does not).**

| Piece | Free-edge Part 6 | vs SS Part 3 |
|---|---|---|
| Discrete bracket `l(·,·)` | **reused verbatim** (`VonKarmanBracket`) | identical |
| Airy `F`-solve `B_F` | **reused verbatim** (clamped `AiryStressSolver`; free edge is in-plane traction-free `N_nn=N_nt=0` ⇒ `F=0,F,n=0`) | identical |
| Membrane energy `H_mem` | **reused** (`(1/2Y)FᵀB_F F`, Wa inside `B_F`) | identical |
| Coupling inner product | **uniform `h²`** | identical |
| Bending operator | `K` = `free_plate_stiffness` (ν re-enters) | was `B = L²` |
| Mass / inertia | lumped `W` (edge-½/corner-¼) | was scalar `h²`·`I` |
| Live↔full-grid seam | identity (all nodes live) | interior-only restrict |

**The two coefficients to check by eye (highest bug risk — the `I → W` asymmetry).** The free coupling
force differs from SS in *two* independent ways; the trap is applying only one. Mirror model #5b's
**bending** acceleration (which carries the `/W` divide), not SS's coupling (which carries neither):

- `step()` RHS: `rhs += k²·(h²/ρ_s)·l(μ_{t·}w, μ_{t·}F)` — the extra `h²` (because `A` now carries
  `W`'s `h²`); **no** manual `/W` (the `A`-solve applies `W⁻¹`).
- `set_state` `w^{-1}`: `u_prev += ½k²·(h²/ρ_s)·l(w⁰, F⁰)/w` — the `h²/ρ_s` **and** the per-node `/w`
  (= `W⁻¹`), exactly like #5b's `accel_term = ½k²κ²(K@u0)/w`.

Get the `h²`-here-but-not-there / `/W`-there-but-not-here asymmetry wrong and the drift mimics a
bracket bug while the bracket is fine.

**Class design.** Add a **`boundary="free"` branch to `VKPlate`** (mirroring how `Plate` carries both
SS and free), *not* a separate `FreeVKPlate` — keeps the `nonlinear=False → #5b` regression alongside
the existing `nonlinear=False → #5`, and does **not** fork the bracket/Airy code paths (they are
boundary-agnostic). `nonlinear=False, boundary="free"` must be **bit-identical** to
`Plate(boundary="free")`.

**Build order & gates (drift-first, per Part-3 culture).**
1. `VKPlate(boundary="free")` construction: `K/W` from `free_plate_stiffness`, all-nodes-live mask,
   the two pinned coefficients. **First gate = large-amplitude lossless drift `< 1e-10`** (the seam
   `probe4` validates; this is the headline, run it before the rest of the suite).
2. `nonlinear=False, boundary="free"` **bit-identical** to `Plate(boundary="free")` (regression).
3. `w → 0` recovers model #5b's free modes; energy non-negativity; passivity (σ>0); drift∝`couple_tol`
   self-cert; Richardson O(h²).
4. Diagnostics: the gong/cymbal ring-down — pitch glide + the crash/shimmer cascade, curved-Chladni
   nodal figures of the *stiffened* plate. (Watch for low-freq creep: `Σh²·l` over the free rim is
   nonzero — energy-neutral per `probe4`, but note it in the ring-down.)

**References for this part:** the free case is **not** in Bilbao 2008 (SS-only); it lives in Bilbao
*NSS* Ch. 13 and the gong/cymbal papers (Bilbao 2005 "A family of conservative FD schemes for
gongs/cymbals"; Ducceschi–Bilbao). Not on disk — the empirical self-certification above stands in,
consistent with how Parts 1–3 were pinned. If a structural issue surfaces during the build, WebSearch
those before grinding stencil variations.

---

## Open decisions — surface to the human, do not silently guess (HANDOFF §11; CLAUDE.md)

1. **Scope/ordering: SS-first de-risk, then free-edge.** ✅ **RESOLVED (human, 2026-06-30).** SS
   rectangle first isolates the nonlinear bracket + conservative coupling with the cleanest boundary
   handling (`B = L²` reused for both factorizations) and a clean small-amplitude → model-#5 check.
   It will sound like a **nonlinear drum/tom with pitch glide**, **not** a cymbal — the cymbal/gong
   is the *free-edge* follow-on (Part 2).
2. **`F` in-plane boundary condition.** ✅ **RESOLVED (human, 2026-07-01): CLAMPED `F = 0, F,n = 0`.**
   Pinned from Ducceschi–Touzé DAFx-15 §4.2 Eq. (11): the true SS-movable in-plane condition is
   `F,tt = F,nt = 0`; the clamped pairing `F = 0, F,n = 0` satisfies it **exactly** (F=0→F,tt=0;
   F,n=0→F,nt=0) while leaving `F,nn`=σ_tt free (a movable edge *does* carry tangential membrane
   stress). The plan's earlier default `F = 0, ΔF = 0` (→ `B_F = L²`) is **wrong for this plate**:
   `ΔF=0` forces `F,nn=0` (a constraint (11) does not ask for) *and* leaves `F,nt`≠0 (which (11)
   requires to vanish) — a different, nonstandard edge (advisor). Energy conservation holds for either
   symmetric `B_F` with `F=0` on the rim (Part-1 bracket result), so the BC is a *fidelity* choice,
   and on fidelity clamped wins. `B_F` = a genuinely new symmetric **clamped biharmonic** (not `L²`);
   its membrane-energy form `H_mem = (1/2Ee)⟨F, B_F F⟩` must use the *same* operator. (DT is the modal
   ref; Bilbao FD PDF not on disk — stale memory — but DT + the derivation settle it without it.)
3. **Parameter surface: `(kappa, E, e, nu)`.** ✅ **RESOLVED (human, 2026-06-30).** The nonlinearity
   onset is at `w ≈ e` (thickness), so `e` becomes a *physically meaningful* new input — unlike the
   linear models where only `κ` mattered. Core surface `(kappa, E, e, nu)` → derive `D`, the membrane
   coefficient `Ee`, and `κ`. Default a thin metal-plate set. (Macro/`musician-knob` mapping still
   deferred, HANDOFF §8.)
4. **Excitation.** Strike (raised-cosine / point impulse) as the natural gong/cymbal exciter; drive
   amplitude reported as `w_max / e`. (Bow/continuous deferred.) — sensible default, not contentious.
5. **Test tolerances.** Energy drift `< 1e-10` (project standard). Small-amplitude modal bar inherits
   model #5's. Pitch-glide is qualitative (monotone, correct sign) — no cents bar invented.

## Traps (pre-flagged)

- **The bracket symmetry is everything.** If `⟨l(a,b),c⟩` is not fully symmetric to machine
  precision, energy *will* drift and no amount of step-size tuning fixes it. Test it standalone first.
- **One bracket, two call sites.** `l(w,w)` and `l(w,F)` **must** be the same function. Two
  near-copies that differ in a `μ`-average silently break conservation.
- **Test at large amplitude.** `drift < 1e-10` at `w/e = 1e-3` only re-tests the *linear* scheme. The
  bracket bug lives where the membrane energy is non-negligible. Drive `w ≳ e`.
- **Two factorizations.** `A` (w-step) and `B_F` (F-solve) are different matrices with different BCs.
  Reusing `A` for the F-solve (or vice versa) is wrong even when both happen to be `L²`-shaped.
- **`F`-BC is not inherited from `w`-BC.** `w` SS does not fix `F`'s in-plane condition. Pin it.
- **Stability does not transfer.** The coupled scheme's step bound may be amplitude-dependent; assert
  `H^n ≥ 0` explicitly and watch the cascade regime. Implicit (iterative) conservative variant is the
  documented fallback.
- **Aliasing in the cascade.** `w > e` folds HF energy down (HANDOFF §8). Oversample; render high `fs`.
- **No analytic nonlinear spectrum.** Don't invent a cents bar for the nonlinear modes. Anchor on
  conservation (machine), small-amplitude → #5 (cents, *that* limit only), Richardson order, glide
  sign.
- **Energy-drift triangulation:** drift ⇒ bracket symmetry / energy-measure / averaging wrong;
  small-amplitude frequency off ⇒ bending operator / `A` wrong; glide wrong sign ⇒ coupling sign /
  `F`-source sign wrong. Never relax the tolerance (HANDOFF §8).

## References

- **Bilbao, S. (2008)**, "A Family of Conservative Finite Difference Schemes for the Dynamical von
  Kármán Plate Equations," *Numer. Methods PDEs* **24**(1):193–216 — the **simple** von Kármán
  system (transverse `w` + Airy stress function `F`), SS rectangle, the conservative FD scheme and
  the discrete bracket. **Primary FD source — pin the bracket + `F`-BC here.**
- **Bilbao, *Numerical Sound Synthesis* (2009), Ch. 13** — nonlinear plate / von Kármán, the discrete
  bracket and energy bookkeeping in the book's notation (`δ`, `μ` operators). Pin the stencil here.
- **Ducceschi, M. & Touzé, C. (2015)**, "Simulations of Nonlinear Plate Dynamics: An Accurate and
  Efficient Modal Algorithm," *DAFx-15* — the continuous FvK equations (used verbatim above), the
  `w/e` amplitude regimes (linear / pitch-glide / cascade), and the free-circular `F`-BCs. A *modal*
  (not FD) route; useful as an independent cross-check oracle later.
- **Bilbao, Ducceschi, Webb / Torin & Bilbao** — full-system and multi-plate extensions (Part 2+).
- `docs/dev/plate-plan.md` (model #5, SS `B = L²`) and `docs/dev/plate-free-edge-plan.md` (model #5b,
  free `K`/`W`, the cell-centered twist `Dxy`) — the linear machinery this composes with.
