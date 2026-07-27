---
name: bow-state
description: "Bowed string — first continuous nonlinear EXCITER (core/bow.py), closes the exciter→ leg; energy-balance not conservation; Helmholtz + Schelleng validated (421 tests)"
metadata: 
  node_type: memory
  type: project
  originSessionId: 6ad89864-f2ae-42f8-a37b-71302a3a528f
---

Bowed string built & green **2026-07-02** (human picked "bowed string" over sympathetic/hammer/wind
after the whole method-ladder #1–6 + body-coupling Steps 1–5 were done). First continuous
**nonlinear exciter** — closes the `exciter →` leg of `exciter → resonator (± coupling) → body`.
Advisor endorsed the design up front ("build it, don't re-plan"). Full suite **523 passed** (+59
bow tests). ruff-clean. Related: [[von-karman-plate-state]] (the nonlinear-solve lesson), [[damped-string-state]] (the resonator it drives), [[body-bridge-state]] (sibling rank-1 driving-point coupling).

**What it is.** `BowedString` (composition, has-a `DampedStiffString`) drives model #3 with a friction
force at a point. Smooth single-hump friction `Φ(v)=force·√(2a)·v·exp(−a v²+½)` (odd; peak `|Φ|=force`
at `v=±1/√(2a)`; max slope `force√(2a)e^{½}` at 0; the negative-slope past the peak is what makes
stick-slip). Force on string `f_B=−Φ(v_rel)`. Use `kappa=0` (flexible fixed-end string) to isolate
bow physics; materials surface is the string's `(L,T,rho,fs,N,sigma0,sigma1)`.

**THE load-bearing move (VK Part-3 lesson, verbatim):** `v_rel` is the **centered** velocity
`I·δ_t·u − v_B`, `δ_t·u=(uⁿ⁺¹−uⁿ⁻¹)/2k` → friction is **implicit**. Explicit friction (Φ at `uⁿ`
velocity) drifts O(k²) and looks non-passive. Same converged `(v_rel,f_B)` feeds both `Φ` and the
power term.

**Rank-1 single-solve scheme (my divergence from advisor's f_ext-hook; bit-identical, cleaner).**
The string update is linear in `f_B` except at the bow node → `uⁿ⁺¹=uⁿ⁺¹_free+(k²/ρh)f_B·a`,
`a=A⁻¹eᵢ` the **one-step driving-point admittance** (precomputed once via new
`DampedStiffString.apply_Ainv(rhs_int)` — the *only* edit to the string). Reading the bow node gives
affine `v_rel=v_free+g·f_B`, `g=k·aᵢ/(2ρh)`. So `f_B=−Φ(v_rel)` is one **scalar** root problem
`v_rel=v_free−gΦ(v_rel)`. step(): capture `uⁿ⁻¹ᵢ`, call `string.step()` force-free (commits
`uⁿ⁺¹_free`, rolls history), read `v_free`, scalar solve, then `string.u += (k²/ρh)f_B·a_full`
(post-step rank-1 correction — same style as the bridges poking `string.u[-1]`). ONE banded solve/step.
NO `f_ext` hook needed (advisor assumed one; rank-1 wins for the implicit force).

**Energy = BALANCE, not conservation** (bow is *active* — that's sustain). `energy()` just returns
`string.energy()` (bow stores none; friction memoryless — no E_conn term unlike the spring bridge).
Discrete identity `Eⁿ⁺¹−Eⁿ = k·f_B·(δ_t·u)ᵢ − loss` telescopes exactly (δ_t· KE telescoping +
`J=(1/h)Iᵀ` spread/read duality; holds ∀θ). **Trick that makes the balance exact regardless of Newton
residual:** apply `f_B` exactly, then report power from the **TRUE** post-correction velocity
`v_true=v_free+g·f_B` (not the Newton iterate) → `bow_power=f_B(v_true+v_bow)`, `bow_work+=k·bow_power`.
Newton convergence is a *separate* guarantee the friction law holds. **Lossless money test:
`E−E₀==bow_work` to 6e-15** (diagnose lossless residual 1.14e-13). Lossy: `E−E₀≤bow_work` (loss only
removes energy — passivity survives coupling).

**Stability ≠ conservation; DON'T over-gate (advisor).** Friction bounded (`Φmax=force`), so no
blowup, and **both stability AND balance hold for ANY root** → do NOT gate on uniqueness. `r(v)`
runs −∞→+∞ so a root always exists. Solver = continuation-Newton (seed prev `v_rel` → physical branch)
+ **bracketed fallback** when Newton stalls (scan `|v−v_free|≤g·force` for sign-changes, `brentq` each,
pick root **nearest the seed** = correct branch pick: at a slip the vanished stick root leaves the slip
root as nearest). `helmholtz_number = g·max|Φ'| = g·force·√(2a)·e^{½}` is a **DIAGNOSTIC** (below 1
single-valued "surface sound", above 1 multivalued Helmholtz = real bowing), NEVER asserted; test:
H#<1 ⇒ fallbacks==0, H#>1 ⇒ fallbacks>0.

**σ1 is REQUIRED for clean Helmholtz (empirical crux).** Without frequency-dependent loss the sharp
corner excites undamped high partials → raucous ~18 slips/period. With `sigma1>0` (rounds the corner):
**exactly 1 slip/period**, stick fraction = 1−β to 3 decimals, slip velocity ≈ −v_bow(1−β)/β (Helmholtz
two-slope), pitch = c/2L. helpers default `sigma0=0.5, sigma1=0.05, force=1.0, sharpness=60, beta=0.13,
v_bow=0.1, lam=0.9`.

**Schelleng is real physics the scheme reproduces (test-param crux).** Clean fundamental only when
force ∈ [F_min,F_max] for that β; window **narrows as β grows** (F_max↓) and **scales with v_bow**.
Fixed force + swept β → subharmonics/wrong pitch (β=0.25 force=1.0 gave 56 Hz). Fixes: slip/one-slip
tests use **force=0.4** (clean up to β=0.25); bow-speed tests scale **force=4·v_bow** (holds the window
→ pitch-independent + amplitude∝v_bow, ratio ~2 for 2×). Low-v_bow (0.05) = crushed/dead note
(F_min edge, amp~1e-17) — avoid in tests. Slip detector = **absolute** `|v_rel|≥0.5·v_bow` (relative
0.25·max underestimates the rounded-corner slip *duration*).

Files: `physsynth/core/bow.py` (`BowedString`, `friction_smooth`, `friction_smooth_deriv`),
`DampedStiffString.apply_Ainv` (only string edit), `tests/test_bow_{energy,modal,stability}.py` (59),
`tests/helpers.py::make_bowed_string` + `BOW_*_DEFAULT`, `scripts/diagnose_bow.py` (4 figs: energy
balance, Helmholtz corner+stick-slip+sawtooth, harmonic spectrum, **Schelleng wedge** 23/48 clean).
Next breadth options still open: sympathetic/coupled strings (reuse body node), hammer/mallet contact,
wind/bore+reed.
