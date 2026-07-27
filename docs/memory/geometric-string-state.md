---
name: geometric-string-state
description: Model #10 (geometrically-exact string, u/w/v) — ALL 3 BATCHES DONE, MODEL COMPLETE; batch 3 = Tier B rotating-wave BVP + rig; the cos(Ωk) DG factor, the set_state trap, ψ corrects batch 2's prediction, KC error = (4/3)·shape deformation
metadata: 
  node_type: memory
  type: project
  originSessionId: 60f5cf84-f34a-40e7-afae-77cb6645bdd2
---

Model #10 = **geometrically-exact string** (`core/string_geometric.py`, `GeometricString`): transverse
`u`,`w` + longitudinal `v`. Exists to discharge the two claims [[tension-string-state]] (#9) refuses:
**phantom partials** and **whirling**. Both refusals share one root cause — KC has *one field* and
collapses tension to a spatial scalar. Here tension is a **field**.

**BATCH 1 of 3 DONE & GREEN (2026-07-16), 47 new tests** (`test_geometric_energy.py` 36 +
`test_geometric_polarization.py` 11). Batch 2 = phantom + whirl oracles; batch 3 = Tier B rotating-wave
BVP + viz/diagnose. Staging + all 5 decisions are in `docs/dev/geometrically-exact-string-plan.md`.

**BATCH 2 COMPLETE (2026-07-17) — whirl test #12 landed ⟹ family 70 tests, suite 832.**

# **BATCH 3 COMPLETE (2026-07-17) ⟹ MODEL #10 DONE.** 99 family tests, suite 861.

`analysis/rotating_wave.py` + `tests/test_geometric_rotating_wave.py` (29) +
`scripts/diagnose_geometric_string.py` (4 figs) + 3 `viz` helpers. **The scheme was right a THIRD
time — every difficulty was in the oracle. That is now a law of this model.**

**The helix `u=φcosΩt, w=φsinΩt, v=ψ` is an EXACT solution OF THE SCHEME.** `r²=φ'²(cos²+sin²)` is
time-independent ⟹ `Λ⁺=Λ⁻` ⟹ **the DG's `mean(Λ)` averaging is INERT** and the whole nonlinearity
freezes. BVP in `[φ; ψ; s]`, `s≡Ω_d²=(4/k²)sin²(Ωk/2)`:

```
F_φ = ρ(1−θk²s)·L_u φ + (1−k²s/2)·Gm[a·χ·Gp φ] + ρ s φ = 0
F_ψ = ρ·L_v ψ + Gm[a(Λ−1−Gp ψ)/Λ]                       = 0    ← NO time factors (ψ static ∀θ)
F_norm = ⟨φ,sin_m⟩(2/L)h − R                             = 0    ← modal projection = KC's amplitude
```

- **`cos(Ωk)=1−k²s/2` on the DG row is THE crux** — a naive port drops it. The DG pairs `q^{n+1}`
  against `q^{n−1}`, spanning **2k not k**, so `q̄` picks up `cos(Ωk)` on the *transverse* strains.
  `s` (not Ω) is the unknown because **both factors are linear in it**.
- **Jacobian is NOT symmetric — I derived it as symmetric and ADVISOR CAUGHT IT.** The reduced system
  *looks* variational (cell blocks = Hessian of `V_nl` on the **planar** slice `(p,0,z)`, since
  `r²=p²` either way), but `cos(Ωk)` hits the transverse row only ⟹
  **`∂F_φ/∂ψ = cos(Ωk)·∂F_ψ/∂φ`**. Verified: bit-zero vs that relation, `5.4e-6` (`=1−cos(Ωk)`) vs
  plain symmetry, exactly symmetric at `time_discrete=False`. `splu`; **FD-check the `d/ds` column**.
- **`H_zz = −a p²/Λ³` EXACTLY** via `(1+z)²−Λ²=−p²` (no cancellation; the literal form cancels O(1)).
- **`R→0` is a FREE gate and the strongest one**: collapses to `s=Q/(1+θk²Q)` ≡
  `discrete_stiff_mode_frequency` (written 8 models earlier, knows nothing of helices) — matches
  **0.0–3e-15** across modes 1/2/5 × κ=0/2 in 2 iters. Analysis must **not import core** ⟹ operators
  rebuilt locally (~10 lines). `planar_hessian_cells == 2×core._dg_jacobian(q,q)` (2 = `d(q̄)/d(q⁺)`).
- `time_discrete=False` = semi-discrete (both factors→1); the gap to the discrete one is the
  θ-scheme's `O(k²)` temporal dispersion — keep it OUT of the Tier C/8 comparison.

**THE `set_state` TRAP — the whole 1e-15 claim lives in one function.** Its `y^{-1}` is a 2nd-order
Taylor start: *consistent*, not *exact* ⟹ `O(k³)` history error → straight into the longitudinal
field. Exact history **2e-26** vs `set_state` **1e-16** — **10 orders**, and 1e-16 still *looks* like
machine precision. ⟹ `helpers.seed_rotating_wave` assigns the history directly
(`u⁰=φ, w⁰=0, v⁰=ψ; u⁻¹=φcos(Ωk), w⁻¹=−φsin(Ωk), v⁻¹=ψ`) so no test can reach for the wrong one.

**Don't assert `v` static — `ψ` is a NONZERO static stretch** (the plan's own criterion 16 was wrong
here; `v==0` asserts the physics away). Assert longitudinal **KINETIC** energy
(`helpers.longitudinal_kinetic_energy`): **1.3e-26** over a full revolution vs **6.6e-03** planar =
**23 orders**, where batch 2 could only say five.

**BATCH 2 WAS WRONG ABOUT WHAT BATCH 3 SOLVES FOR — 3 unknowns, it only ever varied 2.** Every
batch-2 circular IC was `set_state(shape, 0.0, w_dot=Ω·shape)` ⟹ **`v⁰=0`** ⟹ **ψ pinned at 0 and
invisible** — and it is the biggest lever. Isolated (`long_kin/E`, others at BVP values):
nothing wrong **1.3e-26** | **ψ=0 → 3.7e-03 (23 orders)** | φ=sine → 3.3e-11 (15) | Ω=KC → 1.6e-15
(11). A helix at `v=0` is *released into the stretch it should already be holding* and rings — a
**relaxation transient, not a pump**, which is exactly why batch 2's 2f₁-band metric couldn't see it.
Batch 2's bound wasn't wrong; it was **blind to a third axis**. **Batch 1's own scratch note (bottom
of this file) already knew `v=0` is not the equilibrium start — batch 2 just never applied it to the
circular runs.** Generalizable: *when a prediction ranks two candidates, check nothing was held
fixed at a wrong constant.*

**OPEN, + one hypothesis I refuted:** the φ-vs-Ω ordering **INVERTS between metrics** (2f₁-pump: Ω
costs 69× more than φ; total longitudinal motion: φ costs 2e4× more than Ω) though both
perturbations are the same relative size (~1.2e-5). My explanation — *shape=broadband,
ellipticity=2f₁ pump* — was **measured and REFUTED** (both put ~51–57 % of longitudinal energy at
2f₁). Real, reproducible, **unexplained**; recorded in the plan, **NOT asserted**. (Unverified
sketch: the BVP family is a 1-param manifold in amplitude, so a wrong Ω still sits near *some* exact
solution while a sine φ is off the manifold entirely.)

**Tier C/8 got a MECHANISM (the plan only hoped for a scaling):** KC's `Ω=√(ω₀²+εR²)` assumes φ is a
sine; the converged φ **IS** the non-sine shape. **`(Ω_BVP−Ω_KC)/Ω_KC → (4/3)·shape_residual` as
R→0, UNIVERSAL** (1.31–1.33 across EA/T=50/100/400 × modes 1/2) ⟹ a single geometric fact about
spinning a helix, not a parameter accident. Sign is physical (`Ω_BVP>Ω_KC`): tension peaks at the
**nodes** where φ' is largest ⟹ Rayleigh quotient beats uniform tension — **the plan once had this
backwards**. **A limit that EXPIRES**: ratio 1.264@4mm, 1.085@8mm, **0.529@16mm** ⟹ assert a band at
small R, never `4/3`. Use `time_discrete=False` + the **discrete** `p2` so only the shape is left.
Tension spread is only **0.5 % of the rise** — that smallness IS the taxonomy: why KC is *good*
(1e-6) and still *wrong*. **Model #10 exists for the 0.5 %.**

**CORE defect found, recorded NOT fixed:** `_dg_jacobian`'s `(v,v)` block assembles `a(χ−1)+…` ⟹
**7e-11 rel at strain 1e-3** (7e-15 at 0.1) — *worse the more realistic the string*, the very
pathology `_stretch_terms` cures. **Harmless where it lives**: a Newton Jacobian only steers the
iteration; the **residual** defines the root (drift untouched). Test bars that block at 1e-8 and says
why. **If that Jacobian is ever reused where accuracy matters, this is the note.**

**RIG GOTCHAS — every panel lied once, and all four looked plausible:** (1) whirl showed **1.1×**
with hand-picked κ_w — it's a **Mathieu tongue**, params are NOT free; use batch 2's recipe (drive
the **soft** plane, ΔT/T₀=1.5, κ_w at the peak `Δ=εA²/4`) ⟹ 63×. (2) **63× STILL drew a flat line** —
the suite only *measures* the rate (0.06 s) but a **picture must SATURATE**: at 0.06 s `w` is 63× a
1e-3 seed = ~9 % of `u` = planar on equal axes; 0.22 s ⟹ `max|w|/max|u|=0.705`, drift **9.7e-13**
through a **61,678×** blow-up (memory's "conserves THROUGH the blow-up", visually). (3) tension panel
rendered **dead flat**, contradicting its own title, because the axis included `T₀` (rise +63 N,
spread 0.3 N) ⟹ plot **`T(x)−⟨T⟩`**, the thing KC discards, and KC becomes exactly the zero line.
(4) phantom spectrum showed **the opposite of its claim** at 0–2 kHz (`f₁+f₂=308` is **11.4 Hz** from
partial 3 at 319 ⟹ sub-pixel, marker families overlap) ⟹ zoom to the combination band.

**WHIRL (`test_geometric_whirl.py`, 8 tests, ~2.9 min) — the ONE section of the plan that was RIGHT.**
Batch 1 & the phantom batch each found the plan's *tests* wrong; *Whirling, honestly* measured true in
every claim. **Why: it was argued from SYMMETRY** (`w→−w`, rotation generator, Wronskian) **not from an
averaged formula — symmetry has no small parameter to be wrong about.** Where that same section *did*
reach for averaging it produced its one wrong number. **Generalizable: prefer the symmetry argument.**

- **The threshold is a MATHIEU TONGUE — the plan never wrote the equation down, and it is the whole
  test.** Reduce to one mode pair (`φ`=sine is exact for BOTH `d_xx` and `d_xxxx` under SS ⟹ the two
  polarizations SHARE it even when detuned); linearise out of plane about `q_u=A cos Ωt` ⟹
  `δq̈_w + [ω_w² + εA²/2 + (εA²/2)cos 2Ωt]δq_w = 0` — a **pump at `2Ω` on the principal resonance**.
  With `Δ=ω_w²−ω_u²`: **unstable ⟺ `0 < Δ < εA²/2`**, peak at `Δ=εA²/4`. `ε` = model #9's OWN
  `kc_mode_coefficients` under `EA→a` (the quartic is isotropic ⟹ planar reduction untouched).
- **`Δ/(εA²)` is THE coordinate** — everything is dimensionless in it. Measured map (t=0.06s):
  `1.00 → 14.7 → 76.3 → 37.4 → 8.4 → 1.63×` at `0 / .07 / .25 / .41 / .5 / .8`. **Peak at .25 exactly**;
  upper edge is **SOFT** (8.4× still at .5 — leading-order `ε`/`Ω`) ⟹ assert dead-by-.8, report the edge.
- **`Δ=0` sits exactly ON the tongue edge** ⟹ leading-order Mathieu reproduces the exact `+1,+1`
  Floquet result it knows nothing about — cheapest evidence the reduction is right.
- **THE FINDING — the plan's discarded `Ω_prec=εA²/(8ω₀)` is the WHIRL GROWTH RATE** (under `ω₀→Ω`).
  The rejection was right (a degenerate string has no precession rate); the diagnosis was right to the
  word (*"the averaging drops the 2ω pump"*) — but **the dropped pump's STRENGTH is that very
  quantity, and the pump IS the instability**. Misattributed, never spurious. Full rate
  `(Ω/2)√(q_M²−σ²)` holds across the tongue to **5–11 %, systematically LOW** (leading-order + the
  seed's non-growing component) ⟹ Tier C, reported. Hardening is 46 % (`ω₀`:117.6/s vs `Ω`:80.7/s).
- **Only the SOFT plane whirls (Gough's real signature; the plan doesn't state it).** `Δ>0` ⟹ the
  driven polarization must be the LOWER one. Same string, same amp, same seed: **76.3× vs 1.00×**.
  Sharpest claim in the model; no energy/spectrum/amplitude measurement can make it.
- **Threshold moves as `√Δ`** (sharper than the plan's "moves with Δω₀"): `A_c=√(2Δ/ε)`. 3 runs —
  `A₁`@`Δ₁` whirls (25.9×), **same `A₁`@`2Δ₁` STABLE** (1.88×), `√2·A₁`@`2Δ₁` whirls again (215×).
  **Run 3 is what makes it a scaling law** — run 2 alone is consistent with "more detuning = stabler".
- **Degenerate control NEEDS the VELOCITY seed** (advisor): `δw=δA·φ` at rest **IS the rotation
  generator** ⟹ degenerate string just runs planar in a ROTATED plane, pins at 1.00× — that's Tier A/1
  restated, **not marginality**. `δẇ` injects angular momentum ⟹ secular solution. Discriminator =
  envelope **shape** over 4 quarters, no fit: secular `1:2:3:4`, exponential = constant ratio.
  Measured **`1 : 1.91 : 3.08 : 4.01`** (1.3 % off linear) vs `1 : 3.80 : 12.3 : 37.6`.
  **⚠ SCOPE CORRECTION (viewer batch 3, MEASURED).** The rule is often remembered as the blanket
  "a velocity seed, NEVER a displacement one — a displaced `w` pins growth at 1.00× *even inside the
  tongue*" (that wording is in `diagnose_geometric_string.py`'s figure-3 comment). **The 'even inside
  the tongue' half is FALSE.** Growth at t=0.06 s, `frac = δ/(εA²)` = 0 / .07 / .25 / .5 / .8:
  **disp** `1.00 / 14.69 / 60.17 / 6.08 / 1.17×` · **vel** `6.88 / 28.52 / 63.00 / 0.85 / 0.78×`.
  A displaced seed grows fine inside the tongue (60× at the peak); the pinning happens **only at
  frac=0**, the degenerate string — which is exactly what the careful sentence above says. The
  **disp row IS the tongue map** recorded below (it reproduces `1.00` and `14.7` exactly) ⟹ that map
  was measured with a *displacement* seed. Rule of thumb: **disp reads the TONGUE** (frac=0 → 1.00×
  = "a degenerate string cannot whirl"), **vel reads MARGINALITY** (frac=0 → 6.88×, secular).
- **SEED MAGNITUDE CONVENTION — the script and the suite differ by ~1000×, and it explains 0.22 s.**
  `diagnose_geometric_string.py` fig-3: `w_dot = 1e-3·A·φ` (m/s), **no `ω_u`** ⟹ initial out-of-plane
  *displacement* ≈ `1e-3·A/Ω`, i.e. `w/u₀ ≈ 1.1e-6`. `test_geometric_whirl._whirl_run` `seed="vel"`:
  `dw' = s·A·ω_u·φ` ⟹ `w/u₀ ≈ 1e-3`, matching `seed="disp"`. So **"63× ⟹ ~9 % of u" is TRUE for a
  1e-3 RELATIVE seed** (the suite's) and the script's own figure reads `8e-5` at 0.06 s — the two
  coexist, and the script needs **0.22 s only because its seed starts 1000× lower**, not because the
  rate is slow. From a 1e-3 relative seed, saturation (`w/u→0.7`) takes only ~**0.1 s**.
- **THE GATE (advisor):** parametric instability = energy **REDISTRIBUTION** ⟹ lossless model
  **conserves THROUGH the blow-up** (drift ~1e-12 while `max|w|` grows 76×). *That* separates a whirl
  from a diverging solve — the other thing that makes `|w|` grow orders of magnitude. **Energy is NOT
  sufficient**: model #9's IN-plane exchange conserves too — and it is the **SAME `2ω` pump aimed at
  neighbouring MODES instead of the other POLARIZATION**. So assert the driven field stays
  single-mode. **`κ_u=0` ⟹ `εA²/ω_u² == ΔT/T₀` EXACTLY** ⟹ `ΔT/T₀=1.5` is a *measured* half-margin
  to #9's ≈3, not a hoped-for one (off-mode 0.09 %, and 0.2 % at the 2.16 the `√Δ` test reaches).
- **Honesty gate:** unseeded planar at the tongue centre (35 mm, 17 Hz detune) ⟹ `max|w| == 0.0`
  bit-exact. Without it every growth ratio is partly measuring a leak.
- **Rig:** `N=16` (mode 1 carries the claim; `p²` 0.3 % off continuum), `lam_long=0.9`, `t=0.06s`,
  seed `1e-3·A`, 15 runs in 4 module fixtures. Cost is the Newton solve (**2.0 ms/step vs 0.081 ms for
  `energy()`** ⟹ the drift gate is FREE, take it every step). **Refinement-invariant**: tongue centre
  `κ_w = 39.05/39.01/39.00` at `N=16/24/32` (rate is ~6 % `N`-dependent — `ε` is a continuum
  coefficient — so assert WHERE the tongue is, not how fast it grows there).
- **The rate has a closed form and it is `_mathieu_rate`** (`(Ω/2)√(q_M²−σ²)`, `q_M=εA²/(4Ω²)`,
  `σ=(δ−εA²/4)/Ω²`) — `Ω` is the **PLANAR Duffing** `√(ω₀²+¾εA²)`, *not* the rotating wave's circular
  `√(ω₀²+εA²)`: the driven motion is a plane oscillation. (Viewer batch 3 measured the ¾ matters —
  it sets the envelope window/anim stride; circular would be 8 % off, linear f₁ 46 %.) Predicted
  **80.69/s** at the tongue peak, measured **74.16** ⟹ ratio 0.92, inside the 5–11 %-low band.
- **Exaggeration (`κ_w≈39` ⟹ 17 Hz detune, `A`=35 mm) is a microscope, same as `κ=8`:** the tongue is
  dimensionless, so scaling `Δ` and `εA²` **together** preserves the physics exactly and only
  compresses wall-clock (rate `~εA²/8Ω`). A realistic sub-Hz detuning whirls over *seconds*.
- **Known-tightest line, flagged for the human (advisor):** the rate test's `assert meas < pred` (the
  residual is *systematically* low) has only **4.9 % headroom** at `frac=0.07` (0.951) vs its own
  `abs=0.2` primary bar — a 0-margin sign claim under a 20 % tolerance. Left as-is deliberately: the
  measurement is **deterministic** (the seed is macroscopic `1e-3·A`, so cross-BLAS round-off ~1e-16
  amplifies only to ~1e-14 over the run's e^4.3 — nine decades below the headroom; the NumPy-2.4
  Windows cliff in [[barrier-collision-state]] was *performance*, not numerics). **If CI ever flakes,
  this is the line** — fix = drop the strict inequality, keep `approx(1.0, abs=0.2)`; the
  "systematically low" story lives in the docstring either way.
- **The `εA²/(8ω₀)` resurrection is an INTERPRETATION, not a measurement** (advisor — matters because
  it is now in plan+README+commit): the **fact** is the rate matching `(Ω/2)√(q_M²−σ²)` to 5–11 %.
  The identification with the plan's discarded precession rate is the *story* — defensible (the same
  `2ω` Fourier component of `r²` both sources the averaged precession and drives the parametric
  pump, so one quantity legitimately appears in both analyses). A pushback on the narrative is **not**
  a problem with the test.
- **Test bug I hit (family's recurring one):** asserted `max|s.u| > 0.5·amp` on the **final step** —
  a mode-1 string passes through `u≈0` twice a period, so the last step is an arbitrary phase (0.42 A).
  **Track across the run, never sample at the last step** (same trap `mode_off_fraction` + the phantom
  test's `w_max` note warn about).

**BATCH 2 — the PHANTOM ORACLE + Tier A/3 DONE & GREEN (2026-07-16).** Landed: `test_geometric_phantom.py` (6),
`test_geometric_limits.py` (4 — Richardson **self**-convergence 3.4<ratio<5.2, Duffing amplitude
shift, model #9 cross-gate), plus all 3 inherited gaps discharged (order ✓; softening re-justified =
**materials not stability**, `Λ₀=a/EA<0` is a natural length below zero, and a softening string
provably conserves + `E≥0` + can't go slack; `λ_long` warn exempt at `_a==0`).
**Batch 1's lesson repeated exactly: the scheme was right, the PLAN's tests were wrong.** Three
plan statements were false:
1. **Tier A/3's stated metric (`integrated longitudinal energy, orders apart`) reads 1.00×** — `v=0`
   isn't the longitudinal equilibrium, so BOTH runs radiate a broadband transient (free modes at
   `n·c_long/2L`) that dominates the integral equally. Use the **bridge-force spectral magnitude at
   `2f₁`**, band-limited below the 1st free longitudinal mode ⟹ **113,000×**.
2. **The circular residual is ELLIPTICITY (`Ω`), not the non-sine mode shape (`φ`)** — tuning the
   **sine** helix to the KC circular `Ω=√(ω₀²+εA²)` (`kc_mode_coefficients`, `EA→a`) collapses the
   pump **300×** (367×→113,000×). ⟹ ~~batch 3's BVP job is mostly `Ω`, not `φ`.~~
   **⚠ CORRECTED BY BATCH 3 — see the batch-3 section above.** The `Ω`-beats-`φ` finding holds *on
   this metric* (2f₁ pump), but the conclusion missed a **third unknown**: every circular IC here has
   `v⁰=0`, so **`ψ` (the static stretch) was pinned at zero and never varied** — and it dominates
   both (23 orders vs φ's 15 and Ω's 11). The φ-vs-Ω ordering also **inverts** on the
   total-longitudinal-motion metric. Batch 3's BVP solves for all three.
3. **`κ=8` not the default 2 — and NOT for wall-clock:** `f₁,f₂` are measured from the phantom run
   so they're *hardened*, and the +1.29 Hz shift **exceeds** κ=2's 0.89 Hz gap ⟹ the phantom
   **crosses** `f₃` and the test would report a phantom landing ON a partial. No run length fixes a
   physical confound. Exaggerates **contrast** not effect (mechanism is κ-independent).
   **Sharpened while wiring the viewer (batch 4) — the κ=2 trap is WORSE than this, and `N` is a
   SECOND control:** the defect `f₂−2f₁` is **not pure stiffness**, because the θ-scheme's dispersion
   drags `f₂` *flat* and contributes a NEGATIVE defect. Ladder defect at κ=0/2/8 — N=16
   `−0.965/−0.677/+3.571`, N=24 `−0.430/−0.137/+4.168`, N=32 `−0.242/+0.052/+4.377`. At κ=0 it is
   *pure numerical dispersion*, O(h²) (0.965/0.242 = **4.00** exactly). So at κ=2/N=32 dispersion
   nearly **cancels** the stiffness (+0.05 net), and at κ=8/**N=8** — plenty of stiffness — the grid
   eats it (+0.38, unshowable). ⟹ any defect gate must be **one-sided** (`>=`, never `abs`): a coarse
   grid displaces phantoms to the *wrong side* by artifact, which `abs()` would happily score.

**Two oracles the plan didn't have:** (a) **the confound-free defect `f₂−2f₁`** — for a harmonic
string `f₂−f₁==f₁` and `2f₁==f₂` EXACTLY, so both displacements *are* `|f₂−2f₁|`, measured in ONE
run, no oracle, no confound (hardening *widens* it 4.416→4.574, working against the claim);
(b) **circular static stretch == 2× planar EXACTLY** (planar `r²` time-avgs to `A²φ'²/2`, circular
*is* `A²φ'²`) — measured **1.987×** ⟹ the null is non-vacuous: the circular string is stretched
**twice as hard** and radiates 113,000× less. Nonlinearity ON and **silent**.

**NEVER say "same energy" about Tier A/3** (advisor): at equal amplitude circular runs BOTH
polarizations at full amplitude ⟹ **2× the planar energy** (measured 1.99×). The tempting summary
"same amplitude, same energy, opposite longitudinal spectrum" is false in the middle clause — batch
1 had it in a TEST NAME (now renamed + asserts the 2×). True claim is stronger: **twice as
energetic AND twice as stretched, radiating ~1e5× less**. Also: the φ finding is a **BOUND not an
observation** — φ is unchanged across both circular runs (both sines), so the tuned run's 9.4e-3
still contains all of φ's contribution ⟹ `φ ≲ 9.4e-3 ≪ 2.9` ⟹ shape error **≲0.3 %** ⟹
~~batch 3's BVP + a **sine** φ should already reach most of Tier B's bit-zero.~~
**⚠ THE LAST CLAUSE IS FALSE (batch 3).** A sine φ + the right ψ + the KC Ω lands at `long_kin/E`
**3.3e-11**, not bit-zero — the BVP's φ is worth the last **15 orders** on that metric. The bound
itself is sound; what it bounds is φ's share of *this metric* (the 2f₁ pump), and "most of the way
to bit-zero" does not follow from it.

**Phantom rig (reusable):** readout = **bridge force `EA·v_x(0)=EA·v[1]/h`** (what radiates in a
piano; quasi-static ⟹ carries `r²` directly). Modes 1+2, `N=32, κ=8, amp=1.5e-3, lam_long=0.9,
T=0.1s` (~2.5 min). **`lam_long=0.9` ≈ bit-identical phantom freqs to 0.5 at 3.6× less wall clock**
(they ride the transverse partials, `λ=0.04`); stay <1 — never run the headline in the warned
regime. **T=0.1 not 0.05**: at 0.05 the weakest peak (`2f1`) is mislocated 0.52 Hz by neighbours'
leakage skirts (margin 170×→8×). `detect_peaks` (blind) not `measure_partials_near`; combos from
**MEASURED** `f1,f2`; `f₃` from the **discrete** `stiff_dispersion_frequencies`, *earned* by an
`amp→0` run landing on it to 1.5e-4. **`EA=T₀` control: `v` is identically `0.0`** (bit-exact) ⟹
proves the readout can't leak the transverse field. Margins: 4 strongest in-band peaks == the 4
combos to **0.039 Hz**, strongest non-combo **5.4×** weaker, defect 4.574 Hz = **118×** err,
headline gap 11.41 Hz = **295×**.

**The scheme was right the first time; every failure was in the TESTS.** DG telescopes 1.5e-16,
Jacobian vs FD 1e-10, `EA=T₀` bit-identical to #3, `max|w| == 0.0` exact. Lessons:

1. **`λ_long = c_long·k/h` IS THE TRAP — and nothing enforces it.** `c_long/c = √(EA/T₀) ≈ 22`, so the
   familiar `lam=0.5` silently means `λ_long ≈ 11`. Implicit θ≥¼ ⇒ unconditionally stable ⇒ **no CFL,
   no error, silent garbage**. Measured: `λ_long ≤ 2` → drift ~1e-12 on every hard case; `λ_long ≥ 4`
   → Newton stops converging, drift **1e+3…1e+5**. Every knob that seemed to matter (IC shape, amp,
   EA, N, fs) moves *only this number*; **h-refinement makes it WORSE** (λ_long ∝ 1/h) — a defect that
   grows under refinement is never physics. Fix = `make_geometric_string` **defaults `lam_long=0.5`**,
   `lam=` is the explicit opt-in (reverse of #1–#9, on purpose). **SETTLED (human): constructor
   WARNS at `λ_long > 1`** (`LAM_LONG_WARN`) — warn NOT reject (scheme genuinely unconditionally
   stable; `λ_long=2` conserves 1e-12 ⇒ a hard bar would forbid working configs; unresolved regime
   worth studying, not trusting). Bar=1 not 2 → mirrors "tune toward λ=1", 4× margin. **It fires on
   `lam=0.5` — the params a reader of #1–#9 reaches for first. That is its purpose.**
2. **Energy floor is `0`, NOT `−L·T₀²/(2EA)`** — I got this wrong in *both* directions. `−T₀v_x` is a
   null Lagrangian (telescopes to 0 at fixed ends) but the pre-stress `T₀²/(2EA)` survives per cell,
   which tempts the negative bound = a string **relaxed everywhere**. That state is **inadmissible**:
   it needs `v_x=−T₀/EA` throughout ⇒ `v(L)−v(0)≠0`, both ends clamped. With the constraint,
   `Λ ≥ 1+v_x` ⇒ `mean(Λ) ≥ 1` ⇒ Jensen ⇒ pre-stress cancels exactly ⇒ **E ≥ 0**, equality at rest.
   `−L·T₀²/(2EA)` is the **free**-string floor (matters only if a free end is added).
3. **`drift ∝ newton_tol` is FALSE here** (unlike #9's couple_tol). At resolved `λ_long` Newton is
   quadratic: residual leaps `1e-4 → 1e-11 → 1e-18`, so *every* tol in between exits at the **same
   root** — drift is a **step function of the iteration count**. The "five-decade proportionality" I
   measured was in the `λ_long≈11` **stalling** regime: **the broken parameterisation certifying
   itself**. Test now asserts the *control* (loosen solve → drift moves 9 decades).
4. **Slack is physics, NOT the failure mode.** Negative `tension` (`Λ < 1−T₀/EA`, ~6e-4 compression)
   looks like the smoking gun and isn't — scheme conserves to **1e-12 straight through** slackness
   (T_min=−98 N, drift 4.2e-15). **Judge failure on DRIFT, never on `min Λ`.** Cost an afternoon.
5. **Rotation bars must be RELATIVE.** `1e-13 * scale` on a 4 mm string = a bar *below machine
   epsilon*. Bit-exact holds for the 90° swap + planar (`== 0.0`); arbitrary angle is round-off
   (~2e-13 relative) because it does different arithmetic.

**Physics/design worth keeping:** `Λ=√((1+v_x)²+u_x²+w_x²)`; `a = EA−T₀` ⇒ `EA=T₀` exactly linear
(the anchor). Excess `= a[r²v_x/2 (PHANTOM) + r⁴/8 (#9's KC quartic, LOCAL not averaged)]`.
**DG has a closed form and NO 0/0 branch** (unlike #7/#8's `[DG]` — do not import it): `g=Λ²` is
quadratic ⇒ midpoint exact, and `(√g⁺−√g⁻)/(g⁺−g⁻) = 1/(Λ⁺+Λ⁻)` rationalizes ⇒ **the continuum
gradient at `q̄` with the single swap `Λ(q̄) → Λ̄ = mean(Λ)`** — **mean(Λ), NOT Λ(mean)** (the naive
one's error *shrinks* with amplitude ⇒ passes every qualitative test, fails only energy).
`splu` not `cholesky_banded` (a DG is not the gradient of anything ⇒ **non-symmetric** Jacobian).
`A3` is **CONSTANT** (unlike #9) — nonlinearity is an RHS force ⇒ the banded factors are the `EA=T`
fast path *and* the Newton seed. `_stretch_terms` rationalizes `Λ−1` and `Λ−(1+v_x)` against
catastrophic cancellation (measured 4e-16 → 7e-8 as strain 0.5 → 1e-3 — **worse the more realistic the
string**; mpmath@50 digits confirmed). `set_state`'s `v0` is a **displacement** (name clash: #1–3/#9's
`v0` is velocity) ⇒ velocities are keyword-only.

**Batch 2 inherits three flagged gaps (advisor, not blocking batch 1):**
- **NO convergence-order test yet** — batch 1 has conservation ✓ passivity ✓ modal ✓ **order ✗**
  (deferred: the Duffing oracle it compares against lands in batch 2). A green batch-1 suite is NOT
  a complete one; the family carries one Richardson number per model.
- **The `EA<T₀` softening rationale is probably FALSE** and is now inconsistent with the corrected
  floor: the identity holds for *either sign of `a`*, Jensen still gives `E ≥ 0` at `a<0`, and
  `tension = EA·Λ + |a| > 0` ⇒ a softening string can't even go slack. Guard/test still pass (they
  only check raise-vs-permit) — but re-justify or drop it. Don't reopen inside batch 1.
- **`EA=T₀` anchor sits at `λ_long == 1.0` EXACTLY** (margin +0.0, flush against `LAM_LONG_WARN`).
  Harmless (no `filterwarnings=error`), but a float wobble fires a spurious warning on the most
  important regression test if CI ever errors on warnings. Principled fix: skip the warn when
  `_a == 0` (the model is then literally #3 ×3, and #3 doesn't warn about λ).

**Verified margins (not inferred):** `peak_nl` 1.45e-2 (14.5× its 1e-3 bar), `peak_long` 1.93e-2
(19.3×) — the nonlinearity really is engaged at `lam_long=0.5` despite physical time shrinking ~22×.
Drift 2.3e-11 vs the 1e-10 gate = only ~4× margin.

**Scratch (not shipped, batch-2 material):** a longitudinal-**equilibrium** start
(`v_x = (a/EA)(⟨r²⟩−r²)/2`) is the physically-correct plucked IC — `v=0` is not, since a held string
has already settled. It rescues a pluck at `λ_long≈10` (drift 1.1e5 → 1.15e-13) but does **not** fix
mode-3, so it is not a substitute for resolving `λ_long`. Advisor: keep it out of batch 1 (scope).
