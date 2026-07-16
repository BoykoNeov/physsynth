# Tension-Modulated String — Plan (model #9, Kirchhoff–Carrier)

> **Status: PLANNED → IMPLEMENTED & GREEN (2026-07-16).** `core/string_nonlinear.py`
> (`TensionModulatedString` + `string_coefficients_from_material`), oracle in `analysis/duffing.py`,
> suite in `tests/test_tension_string.py` (39 tests), viz in `viz/plots.py`
> (`plot_duffing_frequency_curve`, `plot_mode_purity`, generalized `plot_energy_breakdown`) +
> `scripts/diagnose_tension_string.py`. Human decisions settled 2026-07-16 (below) before any code,
> mirroring the [von Kármán plate](von-karman-plate-plan.md) culture. Builds directly on
> [model #3](damped-string-plan.md) (`DampedStiffString`), whose scheme, energy form, and oracle
> machinery are reused wholesale.
>
> **Headline results:** `EA=0` bit-identical to model #3 (exact `0.0`); lossless drift **3.5e-13** at
> **82 %** nonlinear energy fraction and **10× T₀** tension; drift **1.4e-13** from a broadband
> pluck; measured ω(A) lands on the **exact elliptic Duffing oracle** to 1.3e-3 across a **+80 %**
> glide; waveform on the exact `cn` to 3.7e-4 of A; Richardson order **2.25**; passivity exact
> (worst rise `0.0`).
>
> **Two claims in this plan were WRONG and were corrected at build** — both flagged inline below:
> the "guaranteed bracket" (§scalar collapse) and "purity holds forever" (§oracle 1 / §oracle 4).
> The second turned out to be a **real physical discovery**, not just an error.

## Goal

The **string family's missing nonlinearity**. Every other family got one — the plate has von Kármán
(model #6) — but models #1–3 are strictly linear: pluck a string twice as hard and you get exactly
twice the amplitude at exactly the same pitch. Real strings don't do that. Displace a string and you
*stretch* it, which raises its tension, which raises its pitch. Hit it hard and the note starts sharp
and glides down as it decays.

This is the string's direct analog of the von Kármán plate: a **quartic** potential, geometric
(not material) nonlinearity, energy-method-first. It buys pitch glide on hard plucks, the
frequency-doubled tension harmonics, and the physically-real tension modulation that tanpura and
electric guitar behaviour depend on.

**It does NOT buy piano phantom partials.** Those are longitudinal-mode products and require a
geometrically-exact (transverse + longitudinal) string — a later batch. Do not claim them.

The deliverable is, as always, **the resonator + the rig that measures its deviation from theory**.
Unlike model #6 — which had *no* analytic modal oracle and had to lean entirely on energy — this
model retains an **exact closed-form nonlinear oracle** (below). That is the headline.

## Physics — Kirchhoff–Carrier

Tension becomes a **scalar functional of the whole string state** (spatially averaged — hence
"Kirchhoff–Carrier", the standard first step short of the geometrically-exact string):

```
ρ·u_tt = T(t)·u_xx − ρκ²·u_xxxx − 2ρσ₀·u_t + 2ρσ₁·u_txx
         \__ NEW: T depends on u __/  \____ model #3, verbatim ____/

T(t) = T₀ + (EA/2L)·I,        I ≡ ∫₀^L u_x² dx     (the "stretch")
```

- `T₀` — the rest tension (model #3's `T`). `I ≥ 0` always ⇒ tension only ever **rises**: a string
  cannot be un-stretched by transverse motion. Hardening, never softening.
- `EA` — axial stiffness (N). `EA = 0` ⇒ `T ≡ T₀` ⇒ **exactly model #3**.
- The nonlinearity is **cubic in `u`** (`I` is quadratic, times `u_xx`) ⇒ vanishes as `u → 0`,
  recovering the linear string. Same structure as `l(w,F)` in model #6.

### Energy

```
H = (ρ/2)‖u_t‖² + (T₀/2)·I + (EA/8L)·I²  +  (ρκ²/2)‖u_xx‖²
                             \_ NEW, QUARTIC _/
```

The nonlinear potential `V_nl = (EA/8L)·I²` is **quadratic in `I`** — the fact that makes this model
much simpler than model #6 (see *The quadratic simplification*). It is non-negative, so `H ≥ 0`
follows from the linear θ-scheme's PSD energy — but **discrete non-negativity does not transfer
automatically** (model #6's lesson), so it gets an explicit gate rather than an assertion.

Continuum passivity is model #3's, unchanged: `dH/dt = −2ρσ₀‖u_t‖² − 2ρσ₁‖u_tx‖² ≤ 0`. Neither loss
term enters `H`; the nonlinearity adds a *stored* term, not a dissipative one.

## Scheme

### The quadratic simplification (why there is no `[DG]` here)

The force is `T·u_xx` with `T = 2·V'(I)`, `V = (T₀/2)I + (EA/8L)I²`. The energy-conserving choice is
the **discrete gradient** in `I`, `T_eff = 2·(V(I⁺)−V(I⁻))/(I⁺−I⁻)`. For a **quadratic** `V` that
collapses *exactly* to `2·V'(Ī)` at the mean stretch — no limit, no `0/0`, no Taylor branch:

```
T_eff = T₀ + (EA/2L)·Ī,     Ī = (I⁺ + I⁻)/2,     I^m = h·Σⱼ (δ_x⁺u^m)ⱼ²
```

**Do not import `collision.py`'s `[DG]`.** Its 0/0 machinery exists for the mallet's *power-law*
`φ(η) = K η^α/(α+1)`, where the discrete gradient genuinely is 0/0 in the quiet region. Here the
potential is quadratic and the discrete gradient is a closed-form midpoint. This is the classic
Bilbao NSS Ch. 8 conservative Kirchhoff–Carrier scheme.

### The split — why `EA=0` stays bit-identical to model #3

**The trap:** exact conservation of the nonlinear potential requires the tension term to be averaged
at **θ = ½** (`μ_t·u = (u⁺+u⁻)/2`). But model #3 averages its whole operator `𝓛 = c²δ_xx − κ²δ_xxxx`
at `θ ≈ 0.28`. Naively moving the *whole* tension term to μ_t would give, at `EA=0`, a θ=½ wave term
against model #3's θ=0.28 — **not bit-identical**, and the family's regression anchor is lost.

**The fix:** split `T_eff = T₀ + ΔT` and average only the **nonlinear excess** at θ=½:

```
δ_tt u = 𝓛·(θ·u⁺ + (1−2θ)·uⁿ + θ·u⁻)          ← model #3, verbatim, θ≈0.28
         + (ΔT/ρ)·D2·(u⁺ + u⁻)/2               ← NEW, θ=½ (μ_t) — required for telescoping
         − 2σ₀·δ_t.u + 2σ₁·δ_t.(δ_xx u)        ← model #3, verbatim

ΔT = (EA/4L)·(I⁺ + I⁻)     ( = (EA/2L)·Ī )
```

The two pieces telescope **independently**, so this is free. `EA = 0 ⇒ ΔT = 0 ⇒` model #3 bit-for-bit
(guarded `if self.EA != 0.0:`, exactly as `κ` and `σ₁` are).

### Telescoping (derived, not assumed)

By SBP with fixed/supported ends, `h·⟨D2·f, f⟩ = −I(f)`, and `D2` is symmetric so the cross terms
cancel:

```
h·⟨D2(u⁺+u⁻), u⁺−u⁻⟩ = h·⟨D2u⁺,u⁺⟩ − h·⟨D2u⁻,u⁻⟩ = −(I⁺ − I⁻)
```

The nonlinear power is then `h·⟨(ΔT/ρ)·D2·(u⁺+u⁻)/2, δ_t.u⟩ ·ρ`, and with `ΔT = (EA/4L)(I⁺+I⁻)`:

```
power_nl = (ΔT/4k)·(−(I⁺−I⁻)) = −(EA/16Lk)·((I⁺)² − (I⁻)²) = −δ_t.[ (EA/8L)·I² ] = −δ_t.[V_nl]
```

— exactly the nonlinear potential's rate. **Matched by the stored energy** (verify by `δ_t+`):

```
E_nl^n = ½·( V_nl(Iⁿ) + V_nl(Iⁿ⁻¹) ) = (EA/16L)·( (Iⁿ)² + (Iⁿ⁻¹)² )
```

This is model #6's **½-averaged, two-time** energy form — the odd/even lesson (a raw single-level
`V_nl(Iⁿ)` is a 2-step invariant and oscillates spuriously). Here it is *derived* rather than
certified after the fact: `δ_t+ E_nl^n = (EA/16Lk)((I^{n+1})² − (I^{n-1})²)` cancels `power_nl`
exactly. So `energy()` = model #3's `energy()` **+ this one scalar term**.

### The scalar collapse (one unknown, bracket found by doubling)

> **CORRECTED AT BUILD (2026-07-16).** The "guaranteed bracket `[0, dT_hi]`" claimed below was
> **wrong**, and the self-certifying gate caught it on the first run. `I⁺` does **not** tend to 0 as
> `ΔT → ∞`: the update `(A₀ − βD2)u⁺ = rhs₀ + βD2u⁻` tends to **`u⁺ → −u⁻`**, so `I⁺ → I⁻`. `I⁺` is
> therefore **non-monotone** in `ΔT` (it dips, then climbs back to `I⁻`), and the naive bracket fails
> on ~2 % of steps — exactly those where `I⁺(0) < I⁻` (the string winding back up).
>
> **The corrected guarantees are cleaner than the wrong one:** `resid(0) ≤ 0` always (stretches are
> non-negative), and `resid → +∞` because `I⁺ → I⁻` is **bounded** while `ΔT` grows linearly. So a
> bracket *always* exists and **doubling provably terminates**. Seed with
> `dT_hi = (EA/4L)(max(I⁺(0), I⁻) + I⁻)` (0/400 steps invalid, vs 7/400 for the naive seed) and let
> doubling handle the rest **silently** — it is the normal mechanism, not an anomaly. Warn only on
> hitting `MAX_BRACKET_EXPANSIONS`, a true failure.
>
> **Uniqueness is deliberately not gated** (the `BowedString` precedent): *any* root conserves energy
> exactly, because the telescoping needs only a self-consistent `ΔT = (EA/4L)(I⁺+I⁻)`. Non-uniqueness
> would be branch selection, not correctness. Empirically `resid` is strictly increasing (one sign
> change over a dense sweep) at any usable timestep.

`ΔT` is **one scalar**, so given a guess the step is one banded solve. With `β = k²ΔT/(2ρ) ≥ 0`:

```
(A₀ − β·D2)·u⁺ = rhs₀ + β·D2·u⁻ ,        A₀, rhs₀ = model #3's matrix and RHS, verbatim
```

`A₀` is SPD and `−D2` is SPD with `β ≥ 0` ⇒ **`A` stays SPD, pentadiagonal** ⇒ `cholesky_banded`
carries over. `A` now depends on `ΔT` so the **prefactorization is lost** — one banded refactor per
residual evaluation. In 1D that is O(n) and offline; accepted (see *Open decisions*).

The step reduces to a **scalar root-find** — the bow's shape, not model #6's vector Picard:

```
resid(ΔT) = ΔT − (EA/4L)·( I⁺(ΔT) + I⁻ )
```

The rest of this subsection is the **original (WRONG) reasoning, superseded by the box above** —
kept because the caveat it ends on is exactly what saved it:

> - `resid(0) = −(EA/4L)(I⁺(0) + I⁻) ≤ 0`.  ← *this part is right*
> - At `ΔT_hi = (EA/4L)(I⁺(0) + I⁻)`: `resid ≥ 0` **provided** `I⁺` is non-increasing in `ΔT`.
>   ← ***the proviso is FALSE***
>
> ⇒ a ~~guaranteed bracket `[0, ΔT_hi]`~~ for `brentq`, with no continuation and no branch-picking —
> cleaner than the bow. **Caveat, to pin empirically at build:** `I⁺(ΔT)` monotone-decreasing is
> physically obvious (more tension ⇒ less stretch) but not proven here —
> `du⁺/dβ = (A₀−βD2)⁻¹D2(u⁺+u⁻)` is not sign-definite in general, because the RHS moves with `β` too.
> Treat monotonicity as a **self-certifying gate** (the model-#6 bracket precedent): assert the
> bracket is valid every step and fall back to expansion + Newton if it ever isn't. If it never trips
> across the test matrix, that is the certificate.

**The lesson:** the "physically obvious" intuition was simply wrong, and the *only* reason it cost
minutes instead of a silent wrong model is that the plan refused to trust it and demanded a gate.
Build the gate for every claim you cannot prove — it trips on run 1 or it certifies you.

## Oracles — the payoff

### 1. Mode purity (headline structural certificate, bit-exact, fs-independent)

`sin(mπx/L)` is an **exact discrete eigenvector** of both `D2` and the SS biharmonic `B = D2²` (the
model #2/#3/#5 money-test culture). And `I` depends on the state **only through `q²`**. So
`A(β)·s = (λ₀ + βp²)·s` for **any** tension, and a single-mode state maps to a single-mode state —
the nonlinearity cannot generate another mode:

```
u = q·s ⇒ D2·u = −p²q·s ,  I = q²·p²·L/2   ⇒  ΔT is a scalar function of q² alone
```

**Test it over a SHORT run (~20 steps), at any amplitude.** The property is *structural and
per-step*, so it does not need — and must not be conflated with — the sub-threshold dynamics of §4.
Assert `‖u − (⟨u,s⟩/⟨s,s⟩)·s‖ / ‖u₀‖ ~ 1e-14`. Normalize by the **initial amplitude**, never the
instantaneous `‖u‖`: at a zero crossing `‖u‖ → 0` and an instantaneous ratio reports a spurious 1.0
(this cost a diagnosis at build). This is the bit-exact structural test (the sympathetic-strings
`max|w_b| = 0.0` culture): it certifies the tension really is a spatial scalar and that `D2`/`B`
share the eigenvector. Energy cannot see any of it.

### 4. The single-mode instability (discovered at build — physics, and a signature test)

**Purity does NOT persist forever, and the plan originally claimed it did.** In exact arithmetic the
reduction is eternal; in floating point, roundoff seeds the other modes, the tension pumps at `2ω_m`,
and those perturbations sit in **parametric (Mathieu) resonance tongues**. Above `ΔT/T₀ ≈ 3` they
grow **exponentially** and the mode disintegrates into its neighbours (for `m=3`: into `m=4` and
`m=8`).

**It is physics, not a bug** — three independent pieces of evidence:
- **Energy is conserved to ~1e-13 throughout** the breakup. A numerical instability does not
  redistribute energy conservatively; it blows up.
- **Refinement-invariant**: onset time 0.029→0.022 s and the *same* unstable modes (4, 8) across
  `fs` 20k→80k and `N` 100→400. A scheme artifact moves with the grid. (Onset drifts slightly
  because the seed is roundoff and exponential growth makes onset depend only *logarithmically* on
  seed size.)
- **Sharp threshold**: pure at `ΔT/T₀ = 2.6` (2.7e-14) → 1.8e-12 at 3.2 → 6.6e-5 at 4.6 → 1.6e-2
  at 5.4. `m=1` is far more robust (1.2e-5 even at `ΔT/T₀ = 28.8`) — the fundamental has no lower
  resonance partner to feed.

This separates two properties cleanly, and the suite must respect it: **energy conservation is
structural** (holds at any amplitude, through the breakup); **mode purity is dynamical** (long-run
only below threshold). It is also a *feature* — modal energy exchange a linear string can never do.

**Honesty:** this is the **planar** modal-exchange instability. It is **not** the out-of-plane
whirling instability of real strings, which needs two transverse polarizations — this model has one.
Same caveat class as phantom partials.

### 2. The Duffing reduction + exact elliptic frequency

Because the single-mode ansatz is exact, the discrete PDE collapses **exactly** to a 1-DOF Duffing
oscillator (with the **discrete** `p² = (4/h²)sin²(mπ/2N)`, matching `analysis/damping.py`):

```
q̈ + ω₀²q + ε·q³ = 0 ,      ω₀² = c²p² + κ²p⁴ ,      ε = (EA/4ρ)·p⁴
```

The hardening Duffing has an **exact closed-form frequency** in complete elliptic integrals
(`q(t) = A·cn(Ωt, m)`, `scipy.special.ellipk` takes the parameter `m`):

```
ω(A) = π·√(ω₀² + εA²) / ( 2·K(m) ) ,        m = εA² / ( 2·(ω₀² + εA²) )
```

Sanity: `ε→0 ⇒ m→0`, `K(0)=π/2`, `ω→ω₀` ✅. Small-amplitude expansion `ω ≈ ω₀(1 + 3εA²/8ω₀²)` ✅.

**But do NOT promise cents-level agreement at one fs.** The measured discrete frequency also carries
the linear θ-scheme's *temporal* dispersion error, which will dominate the nonlinear shift being
measured. Two honest ways to use the oracle (both, ideally):

- **Convergence** to `ω(A)` under refinement (Richardson, model #6's N=24/48/96 pattern).
- **Amplitude shift** `ω(A) − ω(A→0)` vs the Duffing shift — the common linear dispersion error
  **cancels**, isolating the nonlinear physics. This is the tighter of the two.

### 3. 1-DOF reference solver (nice-to-have)

The same reduction means a scalar Duffing stepper with matched coefficients should track the full
`N`-node string to ~1e-14 on a single-mode IC — the `MalletWall` standalone-oracle pattern. Cheap,
and it isolates "scheme wrong" from "spatial operator wrong".

## Human decisions taken (2026-07-16)

1. **`EA` enters as a plain core parameter `EA: float = 0.0`, plus a *non-enforcing* consistency
   helper.** Rejected the model-#6-style materials surface `(E, radius, ρ_v)`. Rationale:
   - The string family **already** exposes *effective coefficients*, not materials — model #3 takes
     `κ = √(EI/ρ)` directly, and `(T, ρ, κ)` are already mutually unconstrained. `EA` standalone adds
     **no new class of inconsistency**; that decision was made at model #2.
   - `EA = 0.0` keeps the model-#3 bit-identity **for free**, with no `nonlinear=False` flag. The
     split-θ scheme was designed specifically to earn that anchor — don't lose it in the API.
   - **Model #6's decision does not transfer.** A plate genuinely *is* a homogeneous sheet of
     thickness `e`; a real musical string is often **wound** (steel core + bronze overwind), for
     which a single `(E, radius, ρ_v)` does not exist. Wound strings are characterized in the
     literature by exactly the effective `(T, ρ, κ, EA)` this API exposes — so the materials
     surface's consistency guarantee would be a *fiction* precisely where strings get interesting.
   - **Unrealistic strings are a feature, not a bug** (human, 2026-07-16). Pairing steel's `κ` with
     rubber's `EA` is not a defect to be designed out — it is HANDOFF **§12.J hyperreal
     instruments** ("physics beyond real materials"), reachable *only* because the surface is
     effective coefficients. The helper **offers** realism; it must never **impose** it.
   - The helper `string_coefficients_from_material(E, radius, rho_v, T, L)` → consistent
     `(rho, kappa, EA)` is the **modeling-oracle** pattern (cf. radiation's `R_a` helpers). It also
     reports the governing ratio (below) — the number that predicts audible glide.
2. **Tolerances: accepted as proposed** (human, 2026-07-16) — see *Tests* below. The only bar left
   open is amplitude-shift-vs-Duffing, to be set empirically after the first measurement.
3. **Solver: the honest iterative scheme** (human, 2026-07-16). Refactor-per-residual accepted
   (~5–15 banded refactors/step, O(n), offline). **Rejected energy quadratisation (SAV /
   Ducceschi–Bilbao)** for now: it restores a single prefactored solve + Sherman–Morrison, but
   conserves a *modified* numerical energy whose auxiliary variable can drift from its defining
   relation — the 1e-10 drift gate would stop certifying the **physical** energy, which is against
   the accuracy-first culture. **Note it as the real-time-port option** (HANDOFF §12.E).

### The governing ratio (worked out at plan time — drives the helper's report)

For a plain (unwound) string the nonlinearity's governing ratio is **radius-independent**:

```
EA/T₀ = EπR² / (ρ_v πR² c²) = E/(ρ_v c²) = (c_long/c)²      ( c_long = √(E/ρ_v) )
```

The radius **cancels exactly**. Hardening is set by the ratio of **longitudinal to transverse wave
speed** — steel: `c_long ≈ 5000 m/s` vs `c ≈ 200–400 m/s` ⇒ `EA/T₀ ≈ 150–600`. Two consequences:
the materials surface is *over-parameterized* for this physics (three numbers for what one governs),
and `EA/T₀` is the right thing for the helper and the diagnostics to report.

## Work breakdown (build order = de-risk, each gate green first)

1. **`analysis/duffing.py`** (new, pure) — `duffing_frequency(A, omega0, eps)` (elliptic), the
   small-amplitude expansion, and `kc_mode_coefficients(...)` → `(ω₀², ε)` from `(c, κ, EA, ρ, p)`.
   Single source of truth for the nonlinear oracle; no core dependency. **Unit-test standalone
   first** (ε→0 limit, expansion agreement at small `A`) — the model-#6 build-order culture.
2. **`core/string_nonlinear.py`** (new) — `TensionModulatedString`, model #3's interface verbatim
   plus `EA`. Standalone file (the string family is one-file-per-model). Model #3 **left untouched**.
3. **`tests/helpers.py`** — `make_tension_modulated_string(...)`, single-mode IC helper.
4. **`tests/test_tension_string.py`** (new) — the suite below.
5. **`viz/plots.py` + `scripts/diagnose_tension_string.py`** — energy breakdown (linear ↔ nonlinear
   exchange), `ω(A)` sweep vs the Duffing curve, pitch-glide spectrogram (needs σ>0).

## Tests — acceptance criteria

1. **`EA=0` == model #3, bit-for-bit.** Free regression anchor (the split earns this). Measured: an
   exact `0.0` state difference over 400 steps, at three loss settings.
2. **Lossless drift < 1e-10 at LARGE amplitude**, nonlinear fraction of `H` *reported and asserted*.
   The headline energy gate. Measured 3.5e-13–4.4e-13 at fractions to **82 %**, tension to **10×T₀**.
3. **Drift < 1e-10 from a PLUCKED (triangular) IC** — **the general-case energy test, and the one
   the plan originally missed** (advisor catch). Every single-mode energy test is secretly a *scalar
   Duffing* test: the cross-mode coupling through `I = ∫u_x²` — the whole point of the model — is
   never exercised from a clean start. A corner gives broadband content mixing from step 1.
4. **Drift ∝ `tension_tol`** — the self-cert, absent a closed form for the general motion.
5. **Energy non-negativity `H^n ≥ 0`** — explicitly gated, not assumed (model #6's lesson).
6. **Passivity** with σ>0: monotone non-increasing.
7. **Mode purity ~1e-14 over a SHORT run at any amplitude** (structural, oracle §1). Normalize by
   `‖u₀‖`, not instantaneous `‖u‖`.
8. **The instability (oracle §4), as two separate dynamical tests:** long-run **sub-threshold**
   (`ΔT/T₀ ≲ 2.6`) stays pure to ~1e-13; long-run **super-threshold** (`ΔT/T₀ ≳ 5`) breaks up
   (off-mode ≫ 1e-3) **while energy still conserves** to 1e-10. The second is discriminating: a
   linear string holds its mode forever.
9. **Amplitude shift vs Duffing** (oracle §2, the dispersion-cancelling form — lead with this, not an
   absolute single-`fs` comparison) + **Richardson O(h²)**, both **sub-threshold and short-time**
   (breakup makes an O(h²) rate meaningless — model #6's lesson).
10. **Hardening direction & monotonicity:** `ω(A)` strictly rises with `A`; `A→0` lands on model #3's
    linear fundamental.
11. **Guards:** `EA < 0` rejected; `apply_Ainv` raises (time-varying `A`); non-convergence **warns**;
    the material helper reproduces `EA/T = (c_long/c)²` and is radius-independent.
12. **Portability:** auto-covered — `test_stability.py` sweeps every `core/` submodule. Verified
    directly: `string_nonlinear.py` pulls in **zero** forbidden libraries.

> **Do not run pytest suites concurrently in this repo.** `test_core_is_headless` false-failed twice
> on 2026-07-16 while three suites + the matplotlib diagnostics ran at once (it passed alone, and a
> direct check showed no forbidden imports). It asserts `subprocess.run(...).returncode == 0`, so a
> subprocess that dies from contention (or Windows `.pyc` locking) is reported as *"core imported
> forbidden libraries: "* with an empty list. Hardening it — distinguish the crash case from the
> violation case and surface `stderr` — is **flagged for the human**, not done here: portability-test
> edits have needed review twice before.

## Traps (pre-flagged)

- **Track the `h` in `I = h·Σ(δ_x⁺u)²`** — not `Σ`. Same class of bug as model #8's "force *density*
  `k²/ρ` not `k²/ρh`". A missing `h` will look like a wrong `EA` and pass every qualitative test.
- **The nonlinear term averages at θ=½ regardless of the linear θ.** Using θ≈0.28 there breaks the
  telescoping and drifts at O(k²) — it will *look* nearly right and fail only the 1e-10 gate.
- **`energy()`'s nonlinear term is the two-time ½-average**, not `V_nl(Iⁿ)`. The single-level form is
  a 2-step odd/even invariant → spurious oscillation (model #6, exactly).
- **Never gate the nonlinearity at small amplitude.** `w ≪ e`-equivalent: the test passes by testing
  nothing. Report the nonlinear energy fraction in every energy test.
- **A single-mode energy test is a *scalar Duffing* test in disguise** — it never exercises the
  cross-mode coupling through `I`. Always carry a **plucked/broadband** IC in the drift suite.
- **Never normalize mode purity by the instantaneous `‖u‖`.** At a zero crossing `‖u‖ → 0` and the
  ratio reports a spurious `1.0` that looks exactly like a catastrophic bug. Normalize by `‖u₀‖`.
- **Don't mistake the parametric breakup for a bug** (it conserves energy and is refinement-
  invariant) — **or** for whirling (that needs two polarizations; this model is planar).
- **Oversample** around the nonlinearity (HANDOFF §8) and **warn on non-convergence** rather than
  render a blow-up as physics (model #6's honesty rule: never mislabel divergence as a cascade).
- **Don't claim phantom partials** — KC gives pitch glide + frequency-doubled tension harmonics.
  True phantom partials need longitudinal coupling (geometrically-exact string, a later batch).
- **`apply_Ainv` does not carry over.** `A` is now time-varying, so the bow's rank-1 driving-point
  admittance precompute (`a = A⁻¹eᵢ`) is invalid against this string. Coupling a bow/collision to a
  tension-modulated string is **out of scope for this batch** — flag, don't half-build.
- **Energy triangulation (inherited):** right frequency + wrong drift ⇒ energy-measure bug; wrong
  frequency ⇒ scheme/operator bug; right lossless drift but wrong *hardening* ⇒ the `ΔT` factor
  (the `4L` vs `2L`, or the missing `h`), not the conservative scheme.

## Sources

- **Bilbao, NSS (2009) Ch. 8** — the conservative Kirchhoff–Carrier FD scheme; primary source, pin
  the scheme here rather than reconstruct from memory (the `vk` lesson: get the PDF, read it).
- **Carrier (1945)**, "On the non-linear vibration problem of the elastic string" — the original.
- **Nayfeh & Mook**, *Nonlinear Oscillations* — the Duffing elliptic-integral frequency.
- Builds on [[damped-string-state]] (models #1–3) and [[von-karman-plate-state]] (the quartic-
  potential / conservative-implicit lessons). See [[commit-push-at-batch-end]];
  mind [[respect-ruff-line-length]].
</content>
