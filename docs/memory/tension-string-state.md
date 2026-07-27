---
name: tension-string-state
description: Model
metadata: 
  node_type: memory
  type: project
  originSessionId: 16bcf34c-2240-4459-a026-52fa69e62fbc
  modified: 2026-07-27T12:47:02.028Z
---

**Model #9 — tension-modulated string (Kirchhoff–Carrier)**, `core/string_nonlinear.py`
(`TensionModulatedString`), oracle `analysis/duffing.py`, 39 tests in `tests/test_tension_string.py`,
suite **762** green, ruff clean. Plan `docs/dev/tension-modulated-string-plan.md`. The **string
family's missing nonlinearity** (models #1–3 were strictly linear) and the 1D analog of
[[von-karman-plate-state]] — same quartic potential, same conservative-implicit lesson.

PDE `ρu_tt = T(t)u_xx − ρκ²u_xxxx − losses`, `T = T₀ + (EA/2L)·I`, `I = ∫u_x²` ("stretch"). `I≥0` ⇒
**hardening only** (a string can't be un-stretched).

**Load-bearing design (each one a correction to my first instinct):**
- **NO `[DG]` from `collision.py`.** `V(I)=(T₀/2)I+(EA/8L)I²` is **quadratic in I**, so the discrete
  gradient collapses *exactly* to the midpoint `T_eff = T₀+(EA/2L)·Ī`. The mallet's 0/0 Taylor branch
  exists only for its **power-law** φ. (Advisor caught; I was about to import the collision code.)
- **The θ-SPLIT is what earns `EA=0` bit-identity.** Conservation needs the tension at **θ=½** (μ_t),
  but model #3 averages its operator at θ≈0.28. Moving the *whole* tension term to μ_t breaks the
  anchor. Fix: split `T_eff=T₀+ΔT`, average only the **nonlinear excess** at θ=½ — the two telescope
  **independently**. `EA=0` → model #3 **exact `0.0`**.
- **Energy `E_nl = (EA/16L)((Iⁿ)²+(Iⁿ⁻¹)²)`** = VK's ½-averaged two-time form, but here **derived**
  not certified-after: SBP gives `h⟨D2(u⁺+u⁻),u⁺−u⁻⟩ = −(I⁺−I⁻)` ⇒ power `= −δ_t·[(EA/8L)I²]`, which
  `δ_t+` of that term cancels identically.
- **Scalar collapse** (bow-shaped, not VK's vector Picard): `(A₀−βD2)u⁺ = rhs₀+βD2u⁻`, `β=k²ΔT/2ρ`;
  A stays SPD pentadiagonal (`−D2` SPD, β≥0) but **prefactorization is lost** → refactor per residual
  (human chose this over SAV/quadratisation, which conserves a *modified* energy — HANDOFF §12.E note).
- **`apply_Ainv` RAISES**: A is time-varying, so the bow's constant admittance `A⁻¹eᵢ` is invalid →
  coupling an exciter here needs a joint solve. Deliberately out of scope.

**MY "GUARANTEED BRACKET" WAS WRONG** (the self-cert gate caught it on run 1): `I⁺` does **NOT** → 0
as ΔT→∞ — `u⁺ → −u⁻`, so **`I⁺ → I⁻`**, hence `I⁺` is **non-monotone** and the naive bracket fails
~2 % of steps (those with `I⁺(0)<I⁻`, i.e. winding back up). **Correct guarantees are cleaner:**
`resid(0)≤0` always, and `resid→+∞` *because* `I⁺→I⁻` is **bounded** ⇒ bracket always exists,
**doubling provably terminates**. Seed `max(I⁺(0),I⁻)` (0/400 invalid vs 7/400). Doubling is NORMAL —
don't warn on it. **Uniqueness NOT gated** ([[bow-state]] precedent): *any* root conserves energy
exactly (telescoping only needs self-consistent ΔT); resid is empirically strictly increasing.

**THE ORACLE — what VK never had.** `sin(mπx/L)` is an exact eigenvector of D2 *and* SS `B=D2²`, and
`I` depends on state **only via q²** ⇒ `A(β)s=(λ₀+βp²)s` ∀β ⇒ the scheme collapses **exactly** onto a
1-DOF Duffing `q̈+ω₀²q+εq³=0`, `ω₀²=c²p²+κ²p⁴`, `ε=(EA/4ρ)p⁴` (discrete p²; L cancels via
`I=q²p²L/2`, exact on the grid). Closed forms: `ω(A)=π√(ω₀²+εA²)/(2K(m))`, `m=εA²/(2(ω₀²+εA²))`, and
the **waveform** `q(t)=A·cn(Ωt,m)`. Certified vs independent ODE integration to **1.8e-15**.
Hardening ⇒ `m∈[0,½)` — `K(m)` **never** near its pole (rounds to 0.5 in float at extreme A; assert
`≤0.5`). **Lead with the AMPLITUDE SHIFT** `ω(A)−ω(A→0)` (`duffing_frequency_shift`): a single-fs
absolute ω carries the linear θ-dispersion error which would swamp the shift (abs err 0.0153 Hz →
0.0056 Hz when cancelled). Never `measure_partials_near` (window anchored on the *linear* f0 — the
VK trap, same again).

**DISCOVERY — single-mode motion is PARAMETRICALLY UNSTABLE (physics, not a bug).** Purity is exact
*in exact arithmetic* and per-step ∀amplitude, but roundoff seeds neighbours, tension pumps at
`2ω_m`, and above **ΔT/T₀ ≈ 3** they grow **exponentially** (Mathieu tongues) → mode disintegrates
(m=3 → **m=4 and m=8**; m=4 ends up *bigger* than the driven m=3). **Three proofs it's real:**

> **CORRECTED 2026-07-27 by the viewer's batch 16 (see [[web-viewer-state]]), which mapped this
> properly.** (a) **The partner is NOT fixed — it moves with ΔT/T₀.** For m=3: **m=7** just above
> the edge (ΔT/T₀ 2–3), **m=4 and 11** deeper in, and **m=4 and 8** at ≈9. The "m=4 and m=8" above
> was measured at ΔT/T₀≈11 and is *one tongue among several*, not the answer. (b) **The threshold
> is ≈2 for m=3, not 3** (bracket (1.75, 2.25]) — and it is strongly **mode-dependent**: m=1 (4, 6],
> m=2 (3, 4], m=4 and m=5 both (1, 2]. (c) **"Refinement-invariant" holds only DEEP above.** It was
> measured here at ΔT/T₀≈11, in the merged region. Near the edge the tongue's fine position moves
> with N (2.00/2.00/2.25 at N=100/150/200, plus an N=200-only tongue at 1.75 into partner m=10),
> because a tongue has edges on *both* sides in detuning and which discrete partner is available
> depends on the grid. (d) Just above the edge the instability **SATURATES at a bounded level**
> (0.130/0.155/0.190 at ΔT/T₀ 2.25/2.50/2.75, flat across 40/60/90 periods) — a *partial* exchange
> that degrades the mode without destroying it, not a slow full breakup.
(1) **energy conserved to 1e-13 throughout** the breakup (a blow-up grows energy; a parametric
instability only *redistributes*); (2) **refinement-invariant** — onset 0.029→0.022 s and the *same*
unstable modes across fs 20k→80k, N 100→400 (onset drifts only logarithmically because the seed is
roundoff); (3) **sharp threshold** (2.6→pure 2.7e-14; 5.4→1.6e-2). m=1 far more robust (1.2e-5 even
at ΔT/T₀=28.8 — no lower resonance partner). ⇒ **Energy conservation is STRUCTURAL, mode purity is
DYNAMICAL.** Test purity **short-run (~20 steps) at any amplitude**; two *separate* long-run tests for
sub-/super-threshold. It's the **PLANAR** modal-exchange instability — **NOT** out-of-plane whirling
(needs 2 polarizations), same honesty class as **no phantom partials** (need longitudinal coupling).

**Gates:** `EA=0` bit-identical (exact `0.0`, 3 loss configs); lossless drift **3.5e-13** @ **82 %**
nonlinear fraction / **10×T₀**; **plucked broadband drift 1.4e-13** ← *the advisor's gap catch: every
single-mode energy test is secretly a **scalar Duffing** test and never exercises the cross-mode
coupling through `I`* — always carry a broadband IC; drift ∝ `tension_tol` (1e-4→2.8e-6 … 1e-8→1.6e-12,
then floors); passivity worst rise **0.0**; H≥0; amplitude-shift ≤2.9e-3; Richardson **2.25** (orders
2.97→2.40→2.25, approaching 2 from *above* — spatial/temporal errors partly cancel; must be
sub-threshold + short-time); waveform vs exact `cn` **3.7e-4** of A; glide **+80 %**.

**API decision (human, 2026-07-16): plain `EA: float = 0.0` + non-enforcing helper**
`string_coefficients_from_material(E, radius, rho_v, T)`. Rejected VK's materials surface because
(a) the family already exposes *effective coefficients* (κ=√(EI/ρ) since model #2) so EA adds no new
inconsistency, (b) `EA=0` keeps the anchor free, (c) **real strings are WOUND** — no single
(E,radius,ρ_v) exists, and the literature characterizes them by exactly these effective coefficients,
so the "consistency guarantee" would be fiction, and (d) **human: unrealistic strings are a FEATURE**
(HANDOFF §12.J hyperreal instruments) — the helper *offers* realism, never imposes it.
**Governing ratio `EA/T₀ = E/(ρ_v c²) = (c_long/c)²` — the RADIUS CANCELS EXACTLY** (steel ≈ 150–600);
the materials surface is *over-parameterized* for this physics. `EA_DEFAULT=1e5` (EA/T=500).

**`test_core_is_headless` flakes under CONCURRENT pytest runs (Windows) — not a real violation.**
Hit it 2026-07-16 by running 3 pytest suites + the matplotlib diagnose script at once: failed 2/2
concurrent, passed 2/2 alone, and a direct subprocess check showed **zero** forbidden imports (both
for its 4 modules and for the all-core sweep that loads `string_nonlinear`). Cause: the test asserts
`subprocess.run(...).returncode == 0`, which **conflates "forbidden import found" with "subprocess
died for any reason"** (contention / Windows `.pyc` locking), and then prints an *empty* hit list —
a maximally misleading message. **Don't parallelize full-suite runs in this repo.** Hardening it
(separate the crash case from the violation case, surface stderr) is a **1-line-ish fix flagged for
the human** — portability-test edits have needed review twice before ([[stiff-string-state]],
[[free-plate-state]]), so it was NOT touched in this batch.

**If CI ever flakes on the instability test, look here first:** `test_single_mode_breaks_up_above_threshold...` is
**roundoff-seeded** — it relies on floating-point noise seeding the instability, then exponential
growth. Margin is ~300× (3e-1 vs the 1e-3 bar) and onset depends only *logarithmically* on seed size,
so it's robust — but it is the one test that could behave differently on another BLAS/arch. The fix
would be a tiny **explicit off-mode seed**, NOT a physics change.
**Batch 16 measured the fix and it is free:** the growth RATE is seed-INDEPENDENT (70.1/s at seeds
1e-11, 1e-9 and 1e-6 alike, r²=1.000, vs 68.4/s roundoff-only) — only the *onset time* shifts, by
`log(seed)/rate`. The viewer already seeds explicitly (unit-norm combination of modes 1..24 with
fixed `default_rng(12345)`, scaled to `1e-6·A`), so **nothing shipped depends on roundoff**; this
core test is the only remaining roundoff-seeded claim, and batch 16 pins the self-seeding property
separately (seed = 0 still breaks up, still conserves, still lands in the neighbours).

**Traps:** `I = h·Σ(δ_x⁺u)²` — **track the h** (the model-#8 force-*density* trap; looks like a
mis-scaled EA and passes every qualitative test). **Never normalize purity by instantaneous ‖u‖** —
at a zero crossing ‖u‖→0 and it reports a spurious **1.0** that looks like catastrophe (cost a
diagnosis). Probes in `M:/claud_projects/temp/tension-string-probe/`.
**Sources:** Bilbao NSS Ch.8 (conservative KC scheme — pin from the PDF, not memory), Carrier 1945,
Nayfeh & Mook (Duffing elliptic frequency). **Next:** geometrically-exact string (2 polarizations →
whirling + true phantom partials), or coupling an exciter (needs the joint solve).
See [[commit-push-at-batch-end]]; mind [[respect-ruff-line-length]] (I broke it — 33 E501s).
