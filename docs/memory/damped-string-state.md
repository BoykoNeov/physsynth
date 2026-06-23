---
name: damped-string-state
description: "Physical Synthesis Phase-2 model #3 (damped stiff string, frequency-dependent loss) — built & passing; the +2σ₁u_txx term, energy unchanged, per-mode g_m oracle, rate turnover, θ-artifact symptom-cured"
metadata: 
  node_type: memory
  type: project
  originSessionId: 214aa2bf-71ca-4f49-b6ea-52f618afc6af
---

As of **2026-06-21**, Phase 2 **model #3 (damped stiff string, frequency-dependent loss) is complete
and green** — the model whose *decay is right* (high partials die faster, as real strings do).
Follows [[stiff-string-state]]; plan is `docs/dev/damped-string-plan.md` (marked IMPLEMENTED).

**PDE:** `u_tt = c²u_xx − κ²u_xxxx − 2σ₀u_t + 2σ₁u_txx` — model #2 plus the **frequency-dependent**
loss term `+2σ₁u_txx`. `σ₀` = the old frequency-independent loss (model #2's `sigma`, renamed
`sigma0`); `σ₁ ≥ 0` = new. In the modal domain `δ_xx→−p²`, so a mode's **energy** decays as
`exp(−2σ_eff t)`, `σ_eff(m) = σ₀ + σ₁·p²` (continuum `σ₀ + σ₁β²`, `β=mπ/L`) — highs die faster.

**What was built:** `core/string_damped.py` (`DampedStiffString`, **standalone** mirror of
`StiffString`, not a subclass — one-file-per-model culture, avoids a double-factor), oracle module
`analysis/damping.py` (`discrete_damped_mode_decay`→`g_m`, `discrete_damped_mode_rate`→`Γ_m`,
`modal_loss_rate_continuum`, `discrete_damped_mode_is_underdamped`, `loss_coefficients_from_T60`),
helpers `make_damped_string` + `measure_mode_decay_factor` (log-linear fit over an interior window),
suite `tests/test_damped_string.py`, viz `viz/plots.py::plot_decay_rates` +
`scripts/diagnose_damped_string.py`. **Full repo 118 pytest pass, ruff clean.**

**Load-bearing build decisions:**
- **Scheme = model #2 + one matrix term.** Centered loss `2σ₁δ_t.(δ_xx u)` → the update matrix
  gains a single block: `A = (1+σ₀k)I − θk²𝓛 − σ₁k·D2` (still pentadiagonal, **SPD for all
  σ₀,σ₁,κ≥0** since `−D2` is PD), one extra RHS term `−σ₁k·D2·uⁿ⁻¹`. `cholesky_banded` unchanged.
- **Energy form is model #2's, UNCHANGED.** The loss terms never enter the stored `E^n`, only its
  rate of change → `energy()` is **reused verbatim**; passivity (monotone decrease) is automatic
  (both discrete losses dissipative by SBP: `−2σ₀‖δ_t.u‖² − 2σ₁‖δ_x⁺δ_t.u‖² ≤ 0`).
- **σ₁ guarded like κ** (`if self.sigma1 != 0.0:` skips both the A and RHS terms) → `σ₁=0` is
  **bit-for-bit `StiffString`** (energy + pickup trace identical to machine zero — a free regression
  anchor, `test_sigma1_zero_reduces_to_stiff_string_bit_for_bit`).
- **Money test = per-mode `g_m` oracle.** A single eigenmode's cross-time energy decays ~ripple-free
  as `g_m^n`, `g_m = (1+θk²Q−σ_eff k)/(1+θk²Q+σ_eff k)`, `Q=c²p²+κ²p⁴`. Measured (log-linear fit,
  **interior window** — skip the lossless Taylor start, inconsistent under damping) matches the
  oracle to ~2e-5 rate-relative (worst at the lowest mode, least dynamic range) → **far tighter than
  model #2's 1% low-mode bar**; here *every* mode is checkable. Test bar 5e-4.

**Achieved numbers:** lossless drift ~7.7e-12 (σ₀=σ₁=0 ≡ model #2 pluck); passivity max energy step
≤ −2.7e-6·E0 (strict decrease, incl. σ₁-only); per-mode decay rate-relative err 2e-5 (m=1) → 4e-8
(high m); T60→(σ₀,σ₁) continuum round-trip +0.00% @100Hz, +1.30% @2.3kHz (the documented few-% gap).

**Validation chain is three independent anchors** (advisor-confirmed; do NOT add a σ₁ scale-sweep,
it's redundant): (1) `test_discrete_rate_tends_to_continuum` pins the oracle to the *standard
physics* `2(σ₀+σ₁β²)` (independent of our code); (2) the per-mode test pins the implementation to
that oracle; (3) the T60 round-trip is a sim→continuum check. A σ₁ sign/scale error survives none.

**KNOWN CAVEAT — "highs die faster" is NOT monotone forever.** `Γ_m ≈ 2(σ₀+σ₁p²)/(1+θk²Q)`:
numerator ~p² but the θ-denominator ~p⁴ with stiffness, so the rate **turns over past ~m=32**
(at N=128, κ=2: rises to m≈32 then falls; θQk² hits ~8 by m=120). The "highs die faster" test
asserts the rise **only over a confirmed in-band range [1..16]**, never `diff>0` across the spectrum.
The σ₁=0 contrast (same σ₀) *falls* over [1..16] → σ₁ flips the ordering (the model #2 artifact's
audible symptom). **The θ-scheme `1/(1+θk²Q)` rate-suppression is NOT removed by model #3** — the
oracle accounts for it exactly and the deliberate σ₁p² growth swamps it in-band; framing is
"symptom-cured / accounted-for," NOT "fixed." No test asserts the artifact is gone.

Next: Phase 3 (2D membrane/plate, models #4–5) — the string family (#1–3) is done. Tolerances still
inherit M1's bar (human's §11.5 call). Plot/diagnose follow the model #2 pattern (no viz unit tests).
