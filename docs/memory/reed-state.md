---
name: reed-state
description: "Wind leg BATCH 3 = dynamic single reed (core/reed.py ReedBore) — the wind nonlinear self-oscillating EXCITER (clarinet), acoustic dual of the bow; built & green; closes the wind leg (bore batches 1-2-3 all done). Bow-shaped implicit scalar coupling at node-0 half-cell, exact energy BALANCE 8e-15, odd-harmonic tone, gamma~1/3 threshold"
metadata: 
  node_type: memory
  type: project
  originSessionId: cfb84861-ad0a-4b29-8e67-0a322e9b1bb0
---

**Wind BATCH 3 DONE & GREEN (2026-07-10): `core/reed.py` `ReedBore` — the clarinet.** Closes the
wind leg (bore [[bore-state]] batches 1→2→3 all done) and closes the `exciter→resonator` loop for
winds, the acoustic dual of the bow [[bow-state]]. Human chose **dynamic mass-spring-damper reed**
(not quasi-static) for batch 3. `tests/test_reed_{energy,signature,stability}.py` (33 tests).

**Physical model (Bilbao §9; Chatziioannou & van Walstijn; Dalmont/Kergomard).** Inward-striking
reed = damped SHO whose tip opens a channel `H=H0+y`; a **positive** pressure drop `dp=p_m−p0`
(mouth minus mouthpiece) **closes** it: `mu(y''+g y'+wr² y)=−dp`. Statically shuts (`y=−H0`) at the
**closing pressure `p_closing=mu·wr²·H0`** → the control parameter is **`gamma=p_mouth/p_closing`**.
Flow into the bore at node 0 = **Bernoulli jet + reed sweep**: `U = U_B − Sr·y'`,
`U_B = w·H⁺·sign(dp)·sqrt(2|dp|/rho)`, `H⁺=max(H0+y,0)` (beating clamp).

**THE reed-sweep flow `−Sr·y'` is NOT optional (advisor):** it makes the reed-force work and the
bore sweep-flow work the *same* reactive term with opposite sign → they cancel exactly, leaving a
clean sign-definite budget. Drop it → the term floats free, balance lost. **`Sr` must be the ONE
area shared by the pressure force AND the sweep** (energy identity *requires* it — force-area ==
sweep-area, else the discrete cancellation breaks). The identity even **fixes the sign** to `−Sr·y'`.

**Coupling = bow-shaped implicit scalar solve, made two-field.** Reed = centered leapfrog (`y''` &
`g y'` centered; spring `wr² y` **explicit at n** = string's cross-time potential; `wr·k<2` guard,
trivially met — bore is ~800× oversampled). Nonlinearity at **CENTERED** `dp_bar=p_m−(p0^{n+1}+
p0^n)/2` and **CENTERED** `y'=(y^{n+1}−y^{n-1})/2k` — the *same* two quantities in reed force AND
bore injection → reactive coupling telescopes to machine precision (VK/bow centering lesson; off-
center → O(k) leak). Reed response is **affine in dp_bar**, so eliminating `y^{n+1}` collapses to
**ONE scalar** `D·dp_bar = C_const + p_pref0·U_B(dp_bar)` (D≥2 = reactive stiffening from reed
inertia+sweep). **Advisor lever: freeze opening `H⁺` explicit at n** (it only scales the passive
Bernoulli conductance, never the reactive coupling) → residual has a single `sqrt` cusp, no clamp
kink, strictly **monotone** (unique root). Solve = continuation Newton + guaranteed bracket fallback
(`sqrt` → infinite slope at dp=0 → fallback fires MORE than the bow's, ~760/34300 steps).

**Node-0 half-cell C₀ injection gotcha (advisor, the paste-gotcha again):** flow enters node 0's
continuity, gain `k/C₀` with the **HALF-cell** `C₀=(h/2)S₀/(ρc²)` (trapezoidal wall weight the bore
already carries) — wrong C rescales the flow → balance drifts. Reed rides the bore's `"closed"` left
end (live half-cell DOF); injection applied **INSIDE `Bore.step` via a new `source` hook** BETWEEN
the pressure and momentum sub-steps (so `U^{n+3/2}` sees the corrected `p0` — same ordering the
radiating bell obeys, [[bore-state]]). `source=None` (default) is **bit-identical** to the un-driven
bore (batch-1/2 regression); the hook is the ONLY edit to `bore.py`.

**Money test = energy BALANCE, not conservation (reed is ACTIVE + STORES energy, unlike bow).**
Conserved book = `E = E_bore + E_reed` (reed = SHO, cross-time potential ½Mr·wr²·y^n·y^{n-1}). Exact
discrete identity `E^n − E^0 == mouth_work − jet_loss − reed_damp_work` (+ bore loss/radiation),
every channel sign-definite: `jet_loss=Σk·dp_bar·U_B≥0` (Bernoulli), `reed_damp_work=Σk·Mr·g·y'²≥0`,
`mouth_work=Σk·p_m·U` (active breath). **NOT the bow's residual-independent true-force trick (advisor
caught the wrong claim):** the bow is memoryless (force applied==force booked, one number), but the
reed's two-field coupling sets the bore node from `p_bar` while booking flow separately →
**per-step balance error = `k·p_bar·R/p_pref0`, LINEAR in the scalar residual R** (`u_eff−u_total =
R/p_pref0`). So the balance REQUIRES *and* verifies a converged solve each step; it's 8e-15 only
because `newton_tol=1e-10` keeps R tiny (verified: tol 1e-4→balance 3e-9, tol 1e-2→2e-6). Keep
newton_tol tight. **Gate: lossless-bore drift 8.4e-15 (@ newton_tol 1e-10).**
GOTCHA in diagnostics: baseline must be the REST energy (E=0), NOT the post-first-step energy — the
first step carries the initial work; subtracting e[1] drops it from only the LHS (a constant offset,
not a drift — chased this down, model was always exact).

**Balance is necessary but NOT sufficient — a wrong-sign/dead reed rings down and still passes on
silence (advisor).** Independent oracle = the **clarinet signature** (`test_reed_signature`): (1)
self-sustains above a blowing threshold, decays below; (2) pitch **locked near bore `c/4L`** set by
the AIR COLUMN not the reed (inward-striking → sweeping f_reed barely moves pitch: 162-167 Hz across
f_reed 2000-3000 at iso-gamma); (3) **odd harmonics dominate** (fundamental ≫100× the 2nd harmonic);
(4) reed **beats shut** (H⁺→0 once/period) when blown hard, stays open below threshold. Threshold
lands at **gamma≈0.35 ≈ 1/3** (Dalmont/Kergomard small-osc rule) — reported/bracketed as diagnostic,
NOT hard-asserted (exact value shifts with bore losses). Mouthpiece pressure = clean **square wave**.

**Defaults (clarinet-plausible):** f_reed=2500 Hz, q_reed=4 (heavy lip damping), mu=0.03 kg/m²,
Sr=1.5e-4 m², width=1.5e-2 m, H0=4e-4 m → p_closing≈2961 Pa; default p_mouth=1500 (gamma≈0.5).
**Below-threshold at high f_reed excites a stable grid-Nyquist component** (γ drops as p_closing∝
f_reed²; NOT a blow-up — balance stays 1e-14) → tests/signature use the speaking regime; pickups
interior (node 1) or `pressure()` (bell far-field, composes with `AirRadiation`).

**API:** `ReedBore(bore=, p_mouth=, f_reed=, q_reed=, mu=, Sr=, width=, H0=)`; helpers
`bernoulli_flow`, `make_reed_bore` ([[bore-state]] `make_radiating_bore` sibling);
`mouthpiece_pressure()` (natural pickup), `reed_opening()`, `.gamma`, `.p_closing`,
`.mouth_work/.jet_loss/.reed_damp_work`. `scripts/diagnose_reed.py` (4 figs: energy channels,
square-wave+beating, odd spectrum, threshold sweep). Contact FORCE (beating) deferred to HANDOFF §12
collisions — batch-3 uses the flow-only `max(0,H)` clamp (reed passes through the wall in
displacement, harmless, balance still exact). Full suite green.

Per [[commit-push-at-batch-end]]: commit + push at batch end.
