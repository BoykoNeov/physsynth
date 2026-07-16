# Geometrically-Exact String — Plan (model #10, two polarizations + longitudinal)

> **Status: PLANNED — human decisions SETTLED (see below). No code yet.**
> Follows [model #9](tension-modulated-string-plan.md) (`TensionModulatedString`), whose closing
> caveats this model exists to discharge. Builds on [model #3](damped-string-plan.md)
> (`DampedStiffString`) for the linear operator, θ-scheme, and energy form, and on
> [model #6](von-karman-plate-plan.md) / [model #7](hammer-collision-plan.md) for the
> quartic-potential and discrete-gradient culture.

## Goal — discharge model #9's two refusals

Model #9 ends by explicitly refusing two claims, twice each (module docstring and plan §Traps):

> **It does NOT buy piano phantom partials.** Those are longitudinal-mode products and require a
> geometrically-exact (transverse + longitudinal) string — a later batch. Do not claim them.

> This is the **planar** modal-exchange instability. It is *not* the out-of-plane whirling
> instability of real strings, which needs two transverse polarizations — this model has one.

This is that later batch. Both refusals have the *same* root cause: model #9 has **one field**.
Kirchhoff–Carrier collapses the tension to a **spatial scalar** `T(t) = T₀ + (EA/2L)·I`, which is
precisely what makes it (a) blind to longitudinal dynamics and (b) planar. Model #10 keeps the
tension a **local field** and carries all three displacement components:

```
r(x,t) = ( x + v(x,t),  u(x,t),  w(x,t) )        v = longitudinal, u/w = the two polarizations
```

**The thesis, in one sentence:** model #9's exact elliptic Duffing oracle is exact *for model #9*
**because** #9's tension is a spatial scalar — and it **degrades to a limit oracle** the moment the
tension is promoted to a field. That degradation is not a regression; it *is* the model. Do not blur
it (see *Oracle taxonomy*).

The deliverable is, as always, **the resonator + the rig that measures its deviation from theory**.

## Physics — the geometrically-exact potential

Let `x` be the coordinate in the **rest (tensioned) configuration**, `q ≡ (u_x, w_x, v_x)` the local
strain, and

```
Λ(q) ≡ |∂r/∂x| = √( (1+v_x)² + u_x² + w_x² )        the local STRETCH RATIO  (Λ = 1 at rest)
```

The potential energy density is (derived below; `T₀` = rest tension, `EA` = effective axial stiffness):

```
V(q) = (T₀/2)(u_x² + w_x²) + (EA/2)v_x²  +  (EA − T₀)·[ (u_x²+w_x²)/2 + 1 + v_x − Λ ]
       \_________ LINEAR: model #3's θ-scheme _________/  \____ NONLINEAR EXCESS: DG, θ=½ ____/
```

### Derivation (verified, not assumed)

The rest configuration is *already* stretched: a rest element `dx` has natural length `dx/(1+ε₀)`
with `ε₀ = T₀/EA_n` the pre-strain and `EA_n` the true axial stiffness. Strain relative to natural is
`ε = Λ(1+ε₀) − 1`, and the elastic energy per unit **rest** length is `(EA_n/2)ε²/(1+ε₀)`:

```
V_true(q) = ( EA_n / (2(1+ε₀)) ) · ( Λ·(1+ε₀) − 1 )²        ≥ 0 always
```

Expanding with `Λ² = 1 + 2v_x + |q|²` and matching term-by-term against the boxed `V` gives an
**exact** identification:

```
EA_n = EA − T₀ ,      1 + ε₀ = EA/(EA − T₀)      ⟹  ε₀ = T₀/(EA − T₀) = T₀/EA_n  ✓ self-consistent
V(q) = V_true(q) − V_true(0) − T₀·v_x
```

The `T₀·v_x` discrepancy is a **null Lagrangian**: `∂ₓ(T₀) = 0` contributes no interior force, and
`h·Σ T₀·(v_x)ⱼ = T₀·(v_N − v₀) = 0` at fixed ends. So the boxed `V` **is** the geometrically-exact
potential, and the parameterisation is:

| quantity | value | note |
|---|---|---|
| transverse wave speed | `c = √(T₀/ρ)` | both polarizations — **isotropic** |
| longitudinal wave speed | `c_long = √(EA/ρ)` | |
| nonlinearity coefficient | `EA − T₀` | **`EA = T₀` ⟹ exactly linear** |
| governing ratio | `EA/T₀ = (c_long/c)²` | **model #9's ratio, verbatim** |

`EA/T₀ = (c_long/c)²` is *literally* the number model #9's `string_coefficients_from_material`
already reports (steel: `c_long ≈ 5000` vs `c ≈ 200–400` ⟹ `EA/T₀ ≈ 150–600`). The helper carries
over — **but mind the identification `EA_#9 ↔ (EA − T₀)_#10`** (see *Traps*).

### What the nonlinear excess *is* — the two claims, as two terms

Expanding the excess for small slopes (`r² ≡ u_x² + w_x²`):

```
V_nl = (EA − T₀)·[ r²·v_x/2  +  r⁴/8  + … ]
                   \_______/    \____/
                   PHANTOM      model #9's KC quartic (local, with EA → EA−T₀)
                   PARTIALS
```

- **`r²·v_x/2` — quadratic in transverse × linear in longitudinal.** Two transverse partials at
  `f_i`, `f_j` drive the longitudinal field at `f_i ± f_j`. **This term is phantom partials**, and it
  is *structurally absent* from model #9, which has no `v`.
- **`r⁴/8` — model #9's quartic**, recovered *locally* (`u_x⁴`) rather than spatially averaged
  (`I²/L`). The KC↔GE distinction in one term.

## Scheme

### The discrete gradient — closed form, and NO 0/0 branch

`V` is **not** quadratic (unlike model #9), so a genuine discrete gradient is required:
`⟨∇̄V, q⁺−q⁻⟩ = V(q⁺) − V(q⁻)` exactly. It has a closed form, via two exact facts:

1. `g ≡ Λ²` is **quadratic** in `q`, so the midpoint rule is exact: `⟨∇g(q̄), q⁺−q⁻⟩ = g⁺ − g⁻`.
2. The `√` difference quotient **rationalizes**:
   `(√g⁺ − √g⁻)/(g⁺ − g⁻) = 1/(Λ⁺ + Λ⁻)` — **exactly, with no limit and no Taylor branch**.

Chaining them collapses the whole thing to one substitution:

> **The exact discrete gradient is the CONTINUUM gradient at the midpoint strain `q̄`, with the
> single replacement `Λ(q̄) → Λ̄ ≡ (Λ⁺ + Λ⁻)/2`.  `mean(Λ)`, not `Λ(mean)`.**

```
∇̄V_nl = (EA − T₀) · ( χ·ū_x ,  χ·w̄_x ,  χ·(1+v̄_x) − v̄_x ) ,      χ ≡ 1 − 1/Λ̄  =  (Λ̄−1)/Λ̄
```

`χ` is the mean strain-ratio: `χ = 0` at rest, `χ > 0` stretched (hardening), `χ < 0` slack.
Verified numerically to **7e-16** across strain scales `1e-3 … 0.5`; the naive `Λ(mean)` midpoint
fails the identity at **1.4e-2 … 1.6e0** (see *Traps* — its error **shrinks with amplitude**).

**Contrast the family.** Model #7/#8 need `[DG]`'s `0/0` Taylor branch (power-law potential, genuinely
`0/0` in the quiet region). Model #9 needs no DG at all (quadratic `V` ⟹ midpoint collapse). Model #10
sits between: a real DG, but **exactly regular** — the denominator is `Λ⁺+Λ⁻ ≈ 2` for any physical
configuration (`Λ` is a *stretch ratio*; `Λ → 0` means an element crushed to zero length). **Do not
import `collision.py`'s `[DG]`** — there is no `0/0` here to protect against.

### The split — why `EA = T₀` stays bit-identical to model #3

Model #9's trick, verbatim. The nonlinear excess needs θ=½ (`μ_t`) to telescope, but model #3
averages its operator at `θ ≈ 0.28`. Split, and average only the **excess** at θ=½:

```
ρ·δ_tt u = 𝓛_u·(θu⁺ + (1−2θ)uⁿ + θu⁻)  −  2ρσ₀δ_t.u + 2ρσ₁δ_t.(δ_xx u)   ← model #3, VERBATIM
           + δ_x−[ ∇̄V_nl ]_u                                              ← NEW, θ=½ (DG)
ρ·δ_tt w = 𝓛_w·(θw⁺ + (1−2θ)wⁿ + θw⁻)  −  2ρσ₀δ_t.w + 2ρσ₁δ_t.(δ_xx w)   ← same, with κ_w
           + δ_x−[ ∇̄V_nl ]_w
ρ·δ_tt v = EA·δ_xx(θv⁺ + (1−2θ)vⁿ + θv⁻)  −  2ρσ₀ˡδ_t.v + 2ρσ₁ˡδ_t.(δ_xx v)
           + δ_x−[ ∇̄V_nl ]_v                                              ← no κ: no bending
```

`𝓛_u = c²δ_xx − κ_u²δ_xxxx`, `𝓛_w = c²δ_xx − κ_w²δ_xxxx` — **the detuning lives here and nowhere
else** (decision #1). `c` is shared (the tension is isotropic *by construction*) and `∇̄V_nl` sees the
polarizations only through `r² = u_x²+w_x²`, so `κ_u ≠ κ_w` splits **only the linear operator**. That
is precisely why the knob cannot corrupt Tier A/1: at `κ_u == κ_w` the two rows are the same
expression, so the rotational certificate is a statement about code that is *literally shared*.

The pieces telescope independently, so the split is free.
**`EA = T₀ ⟹ ∇̄V_nl ≡ 0` exactly** (guarded `if self.EA != self.T:`) ⟹ the `u`-polarization is
**bit-for-bit `DampedStiffString`**. This is model #10's regression anchor — the direct analog of
model #9's `EA = 0`, and the reason to keep the split. Requires assembling model #3's RHS
**expression-for-expression** (float addition is not associative — model #9's lesson).

### Telescoping (derived)

With `q^m = δ_x+ y^m` (`y = (u,w,v)` stacked) and `δ_x−` the SBP adjoint of `δ_x+`:

```
ρ⟨δ_tt y, δ_t.y⟩ = δ_t+[ (ρ/2)‖δ_t− yⁿ‖² ]                                       ← standard identity
⟨δ_x−∇̄V, δ_t.y⟩ = −⟨∇̄V, δ_t.q⟩ = −(1/2k)(V(q⁺) − V(q⁻)) = −δ_t+[ ½(V(qⁿ)+V(qⁿ⁻¹)) ]
```

⟹ `E^n = (ρ/2)‖δ_t− yⁿ‖² + [model #3's θ-cross-time linear potential] + h·Σ ½(V_nl(qⁿ)+V_nl(qⁿ⁻¹))`
is conserved exactly. The nonlinear term is the **two-time ½-average** — model #6/#9's odd/even
lesson, here *derived* rather than certified after the fact.

### The solve

Vector Newton on `3(N−1)` unknowns `y⁺ = [u;w;v]`. `∇̄V_nl` is **local per cell**, so the Jacobian is
`A₀ − (k²/ρ)·δ_x−·D_cell·δ_x+` with `D_cell` block-diagonal (3×3 per cell) ⟹ **sparse**.
`A₀ = diag(A_u, A_w, A_v)` is model #3's matrix twice plus the longitudinal one.

- **`splu`, not `cholesky_banded`** — the DG Jacobian is **not symmetric** (a discrete gradient is not
  the gradient of anything). Beam/plate precedent.
- **Damped Newton + Armijo** (model #8's precedent), `newton_tol` exposed; **drift ∝ newton_tol** is
  the self-certification, absent a closed form for general motion.
- **Do not gate uniqueness** (bow / model #9 precedent): *any* root conserves energy exactly, since
  telescoping needs only the DG identity. Non-uniqueness is branch selection, not correctness.
- **Warn, don't render** on non-convergence (model #6's honesty rule).

## Oracle taxonomy — what is exact, what is a limit

The single most important section. Model #10 has **fewer exact closed forms than model #9**, and
saying so plainly is the point.

### Tier A — exact, bit-testable (the money tests)

1. **Rotational invariance.** `V` depends on `u_x, w_x` only through `r² = u_x²+w_x²` ⟹ the model is
   **exactly isotropic** in the `(u,w)` plane. Rotating the IC rotates the whole trajectory. The 90°
   `u↔w` swap should be **bit-exact**; an arbitrary angle to ~1e-15. The cheapest and strongest
   certificate that the two-polarization coupling is right — energy cannot see any of it.
   (The sympathetic-strings `max|w_b| = 0.0` culture.)
2. **Planar-subspace invariance.** `w → −w` is a reflection symmetry ⟹ a perfectly planar IC stays
   **bit-exactly planar forever** (`max|w| == 0.0`). This is *why* an isotropic free string does not
   spontaneously whirl — see *Whirling, honestly*.
3. **The polarization discriminator — THE HEADLINE.** Planar single mode: `r² ∝ cos²(Ωt)` pumps the
   longitudinal field at `2Ω` ⟹ phantoms. Circular single mode (`u = φcosΩt, w = φsinΩt`):
   `r²(x)` is **time-independent** ⟹ longitudinal forcing is **static** ⟹ no dynamic longitudinal
   motion. **Same string, same amplitude, opposite longitudinal spectrum, purely from polarization.**
   Exact for the full model; survives untouched. *State it as integrated longitudinal energy orders of
   magnitude apart — NOT "zero"* (see *Traps*).
4. **`EA = T₀` ⟹ bit-identical to model #3** (per polarization). The regression anchor.
5. **Small-amplitude linear limit.** Three decoupled linear waves: transverse `n·c/(2L)` (both
   polarizations), longitudinal `n·c_long/(2L)`. Models #1/#3's oracle, reused.

### Tier B — exact, but needs a numerical solve (BUILD IT — human decision, last in Batch 2)

6. **The rotating-wave relative equilibrium.** `u = φ(x)cosΩt, w = φ(x)sinΩt, v = ψ(x)` solves the
   full PDE exactly, where `(φ, ψ, Ω)` satisfy a nonlinear BVP (`−ρΩ²φ = (T(x)φ')'`, `∂V/∂v_x = const`).
   From a **converged BVP IC** the helix rotates rigidly and `v` is *exactly* static.

   This is the **exact frequency oracle for the circular sector** — the model's only escape from Tier
   C's "measure the residual, don't promise cents". It also upgrades two things elsewhere:
   Tier A/3's discriminator sharpens from *orders-of-magnitude apart* to **bit-zero longitudinal
   motion**, and Tier C/8's `Ω ≠ √(ω₀²+εR²)` breakage stops being a hand-wave — the BVP's converged
   `φ` **is** the non-sine mode shape that explains it, so the residual gets a mechanism, not just a
   scaling.

   **Human decision: build it** (the steer names this case almost exactly). **Build it last**, behind
   every Tier A gate, so it can be abandoned without blocking anything.

### Tier C — KC-limit closed forms (measure the residual; do NOT promise cents)

7. **Planar single-mode Duffing** (`analysis/duffing.py`, reused with `EA → EA−T₀`). A **limit**
   oracle for model #10: it breaks via **phantom leakage** into the longitudinal field — the very
   physics being added. Lead with the **amplitude shift** (dispersion-cancelling), as model #9 taught.
8. **Circular single-mode `Ω = √(ω₀² + εR²)`.** Also a **limit** oracle: it breaks via **mode-shape
   deformation**. A rigid helix is stretched *non-uniformly* (most near the nodes, where `φ'` is
   largest) ⟹ `T(x)` is non-uniform ⟹ `φ` is **not a sine** ⟹ `Ω ≠ √(ω₀²+εR²)`. This was gotten
   backwards once at plan time and caught; it is exactly model #9's collapse **failing**, on purpose.
9. **Cross-model: model #9 IS the KC limit of model #10.** Quasi-static longitudinal (`c_long ≫ c`)
   + small slopes ⟹ Kirchhoff–Carrier. So model #9 becomes an *oracle* for model #10's transverse
   limit — a genuine cross-model gate.

**For 7/8/9: measure the residual's amplitude scaling empirically. Do not pin the error order from a
Taylor argument** — KC's `ε` is a *uniform* tension average while the GE circular mode feels a
`cos²`-weighted tension, so even the O(R²) coefficient may not match. The empirical fit is what
settles exact-vs-limit.

### Phantom partials — the discriminating oracle

The observable: **combination tones in the longitudinal field** (or the longitudinal end-force
`EA·v_x` at the bridge — what actually radiates in a real piano).

Excite transverse modes `m=1, 2`; the longitudinal spectrum must show peaks at `f₁+f₂`, `f₂−f₁`,
`2f₁`, `2f₂` — all **quadratic** combinations. **With stiffness `κ > 0` this is discriminating:**
inharmonicity stretches the partials (`f_n = n·f₁·√(1+Bn²)`), so `f₁+f₂` sits **~9·B·f₁ below `f₃`** —
a phantom lands where **no transverse partial exists**. That is the signature (Conklin 1999), and it
is the claim model #9 structurally cannot make.

Phantom *frequencies* are set by the well-resolved **transverse** partials, so they are safe even
though the longitudinal field is under-resolved (below). Longitudinal *mode* frequencies are not.

> **BUILT (batch 2), with three corrections.** (1) **`κ=8`, not the default 2** — at `κ=2` the
> 1.29 Hz hardening shift *exceeds* the 0.89 Hz gap and the phantom **crosses** `f₃`; the test would
> be wrong, not merely blurry. (2) **`f₃` from the *discrete* ladder** (`stiff_dispersion_
> frequencies`), never `n·f₁√(1+Bn²)` — and *earned* by an `amp→0` run landing on it to `1.5e-4`,
> which is what licenses using it for an `f₃` that is never excited. (3) The sharper,
> **oracle-free** form of the same signature: `|f₂ − 2f₁|` (see *What batch 2's oracles cost*).
> Also true and load-bearing: the free longitudinal modes at `n·2236 Hz` sit ~4.5× above the ≤500 Hz
> phantom band, so that band is **purely forced response** — which is both the piano physics and why
> band-limiting the peak search is a physical cut rather than a fudge.

### Whirling, honestly

**An isotropic free string does not spontaneously whirl** (Tier A/2). Stronger: `δw = q_u(t)` is an
**exact** solution of the out-of-plane variational equation (it is the rotation generator — check:
`q̈_u = −[ω₀² + εq_u²]q_u`), so one Floquet multiplier is `+1`; the Wronskian is conserved, so the
other is too ⟹ **both multipliers `+1` ⟹ marginal, not exponential**. The second solution grows
*secularly* (linear in `t`) ∝ injected angular momentum. So:

- There is **no universal precession rate**. A leading-order `Ω_prec = εA²/(8ω₀)` was derived at plan
  time by naive averaging and is **WRONG** — the averaging drops the `2ω` pump, which sits exactly on
  the 2:1 parametric resonance and contributes the entire missing `εA²/(8ω₀²)`. Rotational symmetry
  forces `ω_w = ω_u` **exactly, at any amplitude**. (That degeneracy is itself a good non-obvious test.)
- **Exponential whirling** (Gough 1984; threshold `εA² ~ Δω₀²`) requires **breaking the degeneracy**.
  Settled (decision #1): **`κ_u ≠ κ_w`** supplies `Δω₀`, so Batch 2's whirling section is a genuine
  **threshold instability** — planar motion is *stable* below `εA² ~ Δω₀²` and whirls above it, with
  the threshold moving as `Δω₀` is dialed. That sweep is the oracle; it is unavailable to a degenerate
  string, which is why the knob was built.

**The two regimes are complementary tests, not rivals** — and note they *contradict* each other's
symmetry claims by design, which is the point:

| | `κ_u == κ_w` (default) | `κ_u ≠ κ_w` (the knob) |
|---|---|---|
| planar IC | stays planar **bit-exactly** (Tier A/2) | stays planar **exactly** too — reflection symmetry survives detuning |
| planar + perturbation | perturbation is **marginal** (both Floquet multipliers `+1`) | **grows exponentially** above threshold |
| rotational invariance | **bit-exact** (Tier A/1) | broken *in the operator* — but the nonlinearity stays isotropic |

Tier A/2 holds in **both** columns (`w → −w` is a symmetry of `κ_u ≠ κ_w` too), which is what makes
the whirl test honest: whirling must be seeded by a perturbation, never by the planar IC drifting.

## Human decisions — SETTLED

The human's steer, verbatim: *"i prefer to do extra machinery, which later we may decide we dont need,
but it s good to have it developed and available as an option."* Every decision below resolves that
way — **build the knob, and make the simpler behaviour its default case**. This is the same instinct
as [[unphysical-params-are-a-feature]] read from the other side: there, don't *forbid* the exotic
setting; here, don't *omit* the machinery that would reach it.

1. **Detuning knob for whirling — BUILD `κ_u, κ_w`.** Per-polarization bending stiffness; physically a
   **non-circular cross-section**. `κ_w` defaults to `κ_u`, so the degenerate string stays the default
   and Tier A/1 survives **as the `κ_u == κ_w` case**. Linear-additive ⟹ the geometric nonlinearity
   remains exactly isotropic; the knob only splits the *linear* operator. Buys Gough's `εA² ~ Δω₀²`
   threshold in Batch 2. (Rejected: transverse tension anisotropy `T_u, T_w` — that would break
   geometric exactness and kill Tier A/1 outright.)
2. **Staging — full 3-field core now**, validation staged over two batches. The second polarization is
   nearly free in code and buys the bit-exact rotational certificate immediately.
3. **Longitudinal loss — separate `σ₀_long`, `σ₁_long`.** Both default to `None` ⟹ **inherit the
   transverse values**, so the constructor makes no silent physics claim. Real strings damp
   longitudinal motion far less than transverse; that is a setting to **opt into**, documented on the
   parameter, not baked into the default. (`σ₁_long` is the same machinery as `σ₀_long` for free —
   built for the same reason.)
4. **θ — SPLIT** (model #9's trick, verbatim): model #3's operator at `θ ≈ 0.28`, the nonlinear excess
   at `θ = ½`. Keeps the `EA = T₀` bit-identity; `E ≥ 0` becomes an **empirical gate** (test 5) rather
   than a structural guarantee. θ=½-throughout would invert that trade — rejected, the anchor is worth
   more than a free `E ≥ 0`. **`θ = ½` on the excess is not a knob**: it is what makes the DG telescope.
5. **`EA < T₀` floor — BUILD BOTH.** Reject at construction **plus** an explicit `allow_softening=True`
   escape hatch. The floor is real (`EA_n < 0` ⟹ potential unbounded below ⟹ blow-up, not
   hyperreality), so the *default* protects; the hatch means the human, not the constructor, draws the
   hyperreal line. Consistent with [[unphysical-params-are-a-feature]]: the guard is at the physics, and
   it is *offered*, not imposed.

### What the steer changes beyond the four questions

**Tier B is promoted from "nice-to-have" to a Batch-2 deliverable.** The rotating-wave BVP solver is a
near-verbatim instance of the steer: real machinery, plausibly unnecessary (Tier A/3 already certifies
the circular sector), but the only route to an **exact** circular-mode frequency rather than a limit —
and it converts the "circular IC is not bit-zero in `v`" trap into an actual bit-zero test. Build it
**last**, behind every Tier A gate, so it cannot balloon the batch by blocking it.

## Work breakdown — THREE batches, one per session

Human's staging call: *"split it into relatively similar size batches and stop after the first one."*
Promoting Tier B (decision #5) made the old two-batch split lopsided; this is the rebalance. Each
batch is independently green, committed, and pushed — **no batch leaves the suite red**.

**Batch 1 — the resonator + its structural gates** — ✅ **DONE & GREEN (47 new tests)**
1. `core/string_geometric.py` (new) — `GeometricString`. Models #3/#9 left untouched.
2. `tests/helpers.py` — `make_geometric_string(...)`, single-mode / circular / planar IC helpers.
3. `tests/test_geometric_energy.py` (36) — the `EA=T₀` anchor, drift, the newton-tol self-cert, the
   energy floor, passivity, linear limit, guards.
4. `tests/test_geometric_polarization.py` (11) — tests 7–8 (Tier A/1, A/2: the bit-exact certificates).

*Rationale for the cut:* batch 1 is the only batch that can fail **structurally**. Everything after it
is oracles against a resonator already certified conservative and isotropic — so if batch 1 is green,
batches 2–3 cannot be blocked by the scheme, only by their own physics.

**What batch 1 actually cost, and what it bought.** The scheme was right the first time — the DG
telescopes to 1.5e-16, its Jacobian matches finite differences to 1e-10, `EA=T₀` is bit-identical, and
`max|w| == 0.0` exactly. **Every failure was in the *tests*, not the model**, and each was a real
lesson now written into Traps:
- `λ_long ≈ 11` (from the innocuous-looking `lam=0.5`) made the model diverge. The fix is the helper
  default; the finding is that **nothing in the model enforces this** (open decision, above).
- The **energy floor is 0**, not the negative bound the plan asserted and the code shipped.
- **`drift ∝ newton_tol` is false** at a resolved `λ_long`: Newton converges quadratically, so the
  residual leaps `1e-4 → 1e-11 → 1e-18` and every tol in between exits at the *same root* (drift is a
  step function of the iteration count). The five-decade proportionality that appeared to confirm it
  was measured in the `λ_long ≈ 11` **stalling** regime — an artifact of the broken parameterisation
  certifying itself. The test now asserts the control (loosen the solve → drift moves 9 decades),
  which is the half that is genuinely load-bearing.
- Two rotation bars were written as `1e-13 * scale` — i.e. **below machine epsilon** on a 4 mm string.
  Bit-exact symmetry (90° swap, planar) is exact; arbitrary-angle rotation is round-off (~2e-13
  *relative*) and must be gated relatively.

**Batch 2 — the payoff claims** (discharges model #9's two refusals)
5. `tests/test_geometric_phantom.py` — the phantom oracle (test 10) + Tier A/3 (test 11). ✅ **DONE
   & GREEN (6 tests)** — see *What batch 2's oracles cost* below; three of the plan's own statements
   about these two tests were wrong and are corrected there.
6. `tests/test_geometric_whirl.py` — the `κ_u ≠ κ_w` threshold sweep (test 12) + degenerate control.
   ⬅ **the one batch-2 item still open.**
7. Tier C cross-checks vs `analysis/duffing.py` and model #9 (test 13). ✅ **DONE & GREEN**
   (`tests/test_geometric_limits.py`, 4 tests).
8. **Convergence order — batch 1 does NOT have it, and it is a standard gate.** CLAUDE.md lists order
   beside conservation / passivity / modal frequency, and the family carries one Richardson number per
   model (#5b: 5.66, #6: 4.40/5.66). Batch 1 has conservation ✓, passivity ✓, modal ✓ (the
   three-linear-waves test), **order ✗** — deferred here on purpose, because the Duffing oracle it
   should be measured against lands in this batch. Written down so a green batch-1 suite is not
   mistaken for a complete one.
9. **Re-derive the `EA < T₀` softening rationale — the stated reason is probably FALSE.**
   `test_softening_EA_is_rejected_by_default_and_permitted_on_request` justifies the guard as
   "`EA < T₀` ⟹ `a < 0` ⟹ a potential **unbounded below**". But the corrected floor identity
   `V = (EA/2)(Λ−Λ₀)² − T₀²/(2EA) − T₀v_x` is exact **for either sign of `a`**, and the same Jensen
   argument gives `E ≥ 0` when `a < 0` too — and `tension = EA·Λ − a = EA·Λ + |a| > 0`, so a softening
   string cannot even go slack. The guard and its test still pass (they only check raise-vs-permit),
   so nothing is broken — but the *reason* is inconsistent with the floor batch 1 corrected. Decide
   what softening actually breaks (dynamic softening instability? nothing?) and either re-justify the
   guard or drop it. **Do not reopen this inside batch 1.**
10. **`λ_long` boundary nit:** the `EA = T₀` bit-identity anchor sits at `λ_long == 1.0` *exactly*
    (margin `+0.0`), i.e. flush against `LAM_LONG_WARN`. It does not warn (`1.0 > 1.0` is False) and
    there is no `filterwarnings = error`, so it is harmless today — but a float wobble would fire a
    spurious warning on the single most important regression test the day CI makes warnings errors.
    The principled fix is probably to skip the warning when `_a == 0.0`: the model is then exactly
    model #3 ×3, and model #3 does not warn about `λ` either. ✅ **DONE** — exempted on `_a == 0`,
    exactly as proposed; items 8 and 9 likewise (9 landed as "the guard is right, the *reason* was
    false": the line is **materials** — `Λ₀ = a/EA < 0` is a natural length below zero — **not**
    stability, and a softening string provably conserves, stays `E ≥ 0`, and cannot go slack).

**What batch 2's oracles cost, and what they bought.** Batch 1's lesson repeated exactly: *the
scheme was right; the plan's **tests** were wrong.* Three statements above are false, and the
measurements are in `test_geometric_phantom.py`'s docstrings:

- **Tier A/3's stated metric does not work.** "State it as *integrated longitudinal energy*, orders
  of magnitude apart" — measured, planar/circular is **1.00×**. `v = 0` is not the longitudinal
  equilibrium, so *both* runs radiate a broadband longitudinal transient (the free modes at
  `n·c_long/2L`), and it dominates the energy integral **equally**. The plan's own physics says
  where to look instead: `r²` pumps `v` **at `2Ω`**, so measure the **bridge-force spectral
  magnitude at `2f₁`**, band-limited below the first free longitudinal mode. That separates
  **113,000×**.
- **The circular residual is ellipticity (`Ω`), NOT mode shape (`φ`).** The plan blames the
  non-sine relative equilibrium: "the helix relaxes to the true (non-sine) shape and radiates a
  transient". Measured, that is the *small* half. Drive the same **sine** helix at the KC circular
  `Ω = √(ω₀²+εA²)` instead of the linear `Ω` and the residual pump falls **300×** (367× → 113,000×).
  At the wrong `Ω` the circle is an *ellipse*, `r²` is no longer static, and it pumps `2Ω` like a
  planar mode in miniature. **This retargets Tier B: the BVP's job is mostly `Ω`, not `φ`.**
- **`κ` must be raised for the Conklin signature — and NOT for wall-clock.** At the `κ=2` default the
  gap `f₃−(f₁+f₂) ≈ 9B·f₁` is **0.89 Hz**, but `f₁`,`f₂` are measured from the phantom run, so they
  are *hardened*, and hardening drives the phantom **up** by a measured **1.29 Hz** — *more than the
  gap*. The phantom would **cross** `f₃` and the test would confidently report a phantom landing
  **on** a partial. No run length fixes a physical confound. `κ=8` ⟹ gap 11.4 Hz, hardening 11 % of
  it. This exaggerates the **contrast**, not the effect: the mechanism (`r²` pumping `v`) is
  `κ`-independent; `κ` only sets where the partials sit.

**Two oracles batch 2 found that the plan did not have:**
- **The confound-free defect.** The plan states the signature as `f₁+f₂` vs `f₃`, which needs an
  oracle for the unexcited `f₃` *and* inherits the hardening confound. But for a **harmonic** string
  `f₂−f₁ = f₁` and `2f₁ = f₂` **exactly**, so both displacements equal `|f₂ − 2f₁|` — the
  inharmonicity defect, **measured in one run, no oracle, no confound** (hardening moves phantoms and
  partials *together*; measured it *widens* the defect, 4.416 → 4.574 Hz, working against the claim).
- **The circular static-stretch ratio = 2, exactly.** A planar `r²` time-averages to `A²φ'²/2`; a
  circular one *is* `A²φ'²`. So the **DC** bridge force must be exactly **2×** planar — measured
  **1.987×**. This is what makes the Tier A/3 null non-vacuous: the circular string is stretched
  *twice as hard* and radiates 113,000× less. The nonlinearity is not off; it is on and **silent**.

**Batch 3 — the exact circular oracle + the rig**
8. `analysis/rotating_wave.py` (Tier B, decision #5) — the relative-equilibrium BVP solver +
   `tests/test_geometric_rotating_wave.py` (test 16). Abandonable without loss: nothing depends on it.
9. `viz/plots.py` + `scripts/diagnose_geometric_string.py` — longitudinal spectrum with phantom
   markers, polarization orbit (`u` vs `w`), energy breakdown (transverse ↔ longitudinal ↔ nonlinear).

## Tests — acceptance criteria

1. **`EA = T₀` == model #3, bit-for-bit** (per polarization). Free anchor; the split earns it.
2. **Lossless drift < 1e-10**, with the **nonlinear energy fraction reported and asserted** — a
   nonlinearity bug *hides* at small amplitude (model #6/#9's lesson).
3. **Drift < 1e-10 from a PLUCKED (broadband) IC** — the general-case test. A single-mode IC is
   secretly a scalar Duffing test and never exercises the local tension field.
4. **Drift ∝ `newton_tol`** — the self-certification.
5. **`E^n ≥ 0`** — explicitly gated, **not** assumed (see *Traps*: non-convexity is real).
6. **Passivity** with σ>0: monotone non-increasing.
7. **Rotational invariance** (Tier A/1): 90° swap bit-exact; arbitrary angle ~1e-15.
8. **Planar-subspace invariance** (Tier A/2): `max|w| == 0.0` exactly.
9. **Linear-limit modal** (Tier A/5): transverse `n·c/(2L)`, longitudinal `n·c_long/(2L)`.
10. **Phantom partials**: longitudinal peaks at `f_i ± f_j`, **not** at transverse partials (κ>0).
11. **Polarization discriminator** (Tier A/3): circular vs planar longitudinal energy, orders apart.
12. **Whirling threshold** (decision #1): with `κ_u ≠ κ_w`, a perturbed planar mode is **stable below**
    and **whirls above** `εA² ~ Δω₀²`; the threshold **moves with `Δω₀`**. Plus the negative control:
    `κ_u == κ_w` ⟹ marginal, no exponential growth. Both columns keep test 8 (planar stays planar).
13. **Tier C residuals**: amplitude-shift vs Duffing (dispersion-cancelling) + Richardson O(h²), both
    short-time; **report the residual's amplitude scaling** rather than asserting a cents bar.
14. **Guards**: `EA < T₀` rejects, **and `allow_softening=True` lets it through** (decision #5 — test
    both sides); non-convergence warns; `apply_Ainv` raises.
15. **Portability**: auto-covered by `test_stability.py`'s `core/` sweep.
16. **Tier B (last)**: from a converged BVP IC the helix rotates rigidly — `v` static to ~1e-15 and `Ω`
    matching the BVP eigenvalue; the **exact** circular-frequency oracle, and the mechanism behind
    Tier C/8's breakage.

## Traps (pre-flagged)

- **`mean(Λ)`, not `Λ(mean)`.** The single highest-risk line in the model. The naive midpoint's DG
  error **shrinks with amplitude** (measured: 1.4e-2 → 1.3e-5 → 1.8e-8 as strain goes 0.1 → 0.01 →
  0.001), so it will look *nearly right* in every qualitative test — right glide, right spectrum — and
  fail **only** the 1e-10 energy gate. Model #9's θ=½ trap, one level deeper.
- **`λ_long = (c_long/c)·λ ≈ 12–25 ≫ 1` — THE TRAP OF THIS MODEL. Measured in batch 1, and it is
  much worse than "not accurate".** Implicit θ≥¼ means this is *stable* (no CFL, no error raised,
  nothing to violate), so the model **silently produces garbage**. Measured, over `{pluck, mode-3} ×
  {4e-3, 1e-2}`:

  | `λ_long` | drift | verdict |
  |---|---|---|
  | ≤ 2 | ~1e-12 … 1e-13 | conserves, every case |
  | 4 | 1e-13 … 2e+3 | case-dependent |
  | 8–10 | 1e+3 … 1e+5 | Newton stops converging; blow-up |

  Every knob that looked like it mattered (IC shape, amplitude, `EA`, `N`, `fs`) moves *only this
  number*. Note **spatial refinement makes it worse** at fixed `k` (`λ_long ∝ 1/h`) — a defect that
  grows under refinement is never physics. *Stable is not accurate*, and here inaccurate is not
  graceful. `tests/helpers.py` therefore **defaults `make_geometric_string` to `lam_long=0.5`** with
  `lam=` an explicit opt-in — the reverse of models #1–#9, on purpose. **SETTLED (human, batch 1):
  the constructor WARNS above `λ_long > 1`** (`LAM_LONG_WARN`) — warn, not reject, because the scheme
  really is unconditionally stable and `λ_long=2` demonstrably conserves to 1e-12, so a hard bar would
  forbid configurations that work; and the unresolved regime is worth being able to *study*, just not
  to trust. The bar sits at 1 rather than the measured-safe 2 to mirror the project's "tune toward
  λ=1" idiom and keep 4× margin on a sharp cliff. It fires on `lam=0.5` — i.e. on the parameters a
  reader of models #1–#9 reaches for first. That is its purpose.
- **The energy floor is `0`, NOT `−L·T₀²/(2EA)`.** This went wrong in *both* directions before it went
  right: the plan first said `E ≥ 0`, "corrected" that to a negative floor, and the negative floor is
  what shipped into the code. It is wrong. Measure against the natural length `Λ₀ = a/EA`:

  ```
  V(q) = (EA/2)(Λ − Λ₀)²  −  T₀²/(2EA)  −  T₀·v_x          (exact, any sign of a)
  ```

  The `−T₀·v_x` null Lagrangian telescopes to `T₀(v_N − v₀) = 0` at fixed ends and drops from the
  *sum*, while the pre-stress density `T₀²/(2EA)` survives on **every** cell — which is exactly what
  tempts the conclusion `E ≥ −L·T₀²/(2EA)`, the energy of a string relaxed everywhere. **That state
  is inadmissible.** Relaxing every element needs `v_x = −T₀/EA` *throughout*, i.e.
  `v(L) − v(0) = −L·T₀/EA ≠ 0` — the string would have to get shorter, and both ends are clamped.
  Impose the constraint instead: `Λ = √((1+v_x)² + r²) ≥ 1 + v_x` (the root only *adds* transverse
  length), so `mean(Λ) ≥ 1 + mean(v_x) = 1`; Jensen then gives
  `mean((Λ−Λ₀)²) ≥ (mean(Λ) − Λ₀)² ≥ (1−Λ₀)²`, whose weight `(EA/2)·L·(1−Λ₀)²` **is** the pre-stress
  term. They cancel exactly:

  ```
  ⟹   E ≥ 0,   equality iff Λ ≡ 1 everywhere and the string is at rest.
  ```

  `−L·T₀²/(2EA)` is the floor for a **free** string, where the null Lagrangian no longer telescopes —
  keep it in mind only if a free end is ever added. Caveat unchanged: the split gives the linear part
  a θ-cross-time form and the excess a ½-average, which recombine into `h·Σ V` only **at θ=½**, so the
  bar is structural at θ=½ and empirical at θ≈0.28. Keep `energy_floor` a property anyway — it is
  where the derivation lives.
- **Slack is physics, and it is NOT the failure mode.** A batch-1 dead end worth not repeating:
  negative `tension` (`Λ < 1 − T₀/EA`, i.e. ~6e-4 of compression) *looks* like the smoking gun and is
  not. Measured, the scheme conserves energy to **1e-12 straight through** local slackness
  (`T_min = −98 N`, drift 4.2e-15). Hard plucks and high modes genuinely go slack. **Judge failure on
  drift, never on `min Λ`** — conflating the two cost an afternoon.
- **`EA_#9 ↔ (EA − T₀)_#10`.** Cross-model KC checks will show a ~0.2–0.7 % offset (`T₀/EA ~ 1/150 …
  1/600`). **That is the identification, not a discrepancy — do not chase it.**
- **The circular IC is not bit-zero in `v`.** From a **sine**-circular IC the helix relaxes to the true
  (non-sine) relative equilibrium and radiates a transient. State the discriminator as *integrated
  longitudinal energy, orders of magnitude apart* — **not "zero"**. Bit-zero holds only from a
  converged BVP IC — which decision-#5's Tier B build now supplies, but **only after** Tier A/3 has
  landed in its honest orders-apart form. Do not wait for Tier B to state the discriminator.
  > **MEASURED IN BATCH 2 — half of this is wrong.** (a) *Integrated longitudinal energy* is the
  > wrong instrument: it reads **1.00×**, because the `v=0` transient dominates it equally in both
  > runs. Use the **bridge-force magnitude at `2f₁`**, band-limited below the first free
  > longitudinal mode ⟹ **113,000×**. (b) The residual is **ellipticity**, not the non-sine shape:
  > tuning `Ω` to the KC circular relative equilibrium `√(ω₀²+εA²)` collapses it **300×** on an
  > unchanged **sine** helix. "Orders apart, not zero" survives — the *reason* does not.
- **Don't promise cents against Tier C.** Both closed forms are limits for this model, breaking by
  *different* mechanisms (phantom leakage / mode-shape deformation). Measure the scaling.
- **`EA < T₀` is softening** (`EA_n < 0`) and the potential is unbounded below ⟹ blow-up, not
  hyperreality. Unlike model #9's `EA`, this parameter has a **hard floor** — settled (decision #5):
  reject by default, `allow_softening=True` overrides. [[unphysical-params-are-a-feature]] is
  *satisfied*, not contradicted: the effective-coefficient surface stands (`(T₀, ρ, κ_u, κ_w, EA)` are
  mutually unconstrained **above** the floor), and the floor is offered rather than imposed. **If the
  hatch is used, the `E ≥ 0` gate and the drift gate are both void** — say so in the warning; a
  softening string has no lower energy bound to conserve *toward*.
- **Bending is added linearly** (Euler–Bernoulli, per polarization), *not* geometrically-exact. Honest
  framing: a "stiff geometrically-exact string", not a Timoshenko/Cosserat rod. Say so; don't imply
  the bending is exact too.
- **Energy triangulation (inherited):** right frequency + wrong drift ⟹ energy-measure bug; wrong
  frequency ⟹ operator bug; right drift but wrong *hardening* ⟹ the `(EA−T₀)` factor, not the scheme.
- **`apply_Ainv` does not carry over** (model #9's flag, unchanged): `A` is state-dependent, so the
  bow's rank-1 driving-point admittance precompute is invalid. Coupling an exciter here needs a joint
  solve — out of scope; flag, don't half-build.

## Sources

- **Bilbao, NSS (2009) Ch. 8** — geometrically-exact string FD schemes. **Primary source; get the PDF
  and read it** (the `vk` lesson: pin the scheme, don't reconstruct from memory).
- **Chabassier & Joly (2010)**, "Energy preserving schemes for nonlinear Hamiltonian systems of wave
  equations: Application to the vibrating piano string", *CMAME* — the DG treatment of the
  non-quadratic, non-separable potential. The closest prior art to this plan's scheme.
- **Conklin (1999)**, "Generation of partials due to nonlinear mixing in a stringed instrument",
  *JASA* — phantom partials; the oracle's provenance.
- **Bank & Sujbert (2005)** — longitudinal modes and phantoms in piano strings.
- **Gough (1984)**, "The nonlinear free vibration of a damped elastic string", *JASA* — whirling,
  degeneracy-breaking, the `εA² ~ Δω₀²` threshold.
- Builds on [[tension-string-state]] (model #9), [[damped-string-state]] (models #1–3),
  [[von-karman-plate-state]] (quartic potential / conservative implicit),
  [[mallet-collision-state]] + [[barrier-collision-state]] (`[DG]` culture — *not* imported here).
  See [[commit-push-at-batch-end]]; mind [[respect-ruff-line-length]].
