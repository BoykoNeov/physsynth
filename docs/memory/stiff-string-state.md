---
name: stiff-string-state
description: "Physical Synthesis Phase-2 stiff string (model #2) — implicit theta-scheme built & passing; build decisions, achieved numbers, and the damping caveat"
metadata: 
  node_type: memory
  type: project
  originSessionId: adeb84a3-126a-4535-9e89-0420a2167ff4
---

As of **2026-06-21**, Phase 2 **model #2 (stiff string) is complete and green** — the first model
that sounds like a real instrument (piano-like inharmonicity). Follows [[milestone-1-state]]; the
plan is `docs/dev/stiff-string-plan.md` (now marked IMPLEMENTED).

**What was built:** `core/string_stiff.py` (`StiffString`, implicit θ-scheme, one banded SPD solve
per step via `scipy.linalg.cholesky_banded`), interior matrix builders in `core/operators.py`
(`second_difference_matrix`, `biharmonic_matrix`, `delta_xxxx`), oracles in `analysis/modal.py`
(`inharmonicity_B`, `stiff_harmonic_frequencies`, `discrete_stiff_mode_frequency` — θ-dependent!) +
`analysis/dispersion.py` (`stiff_dispersion_frequencies`), suite `tests/test_stiff_string.py`,
viz `viz/plots.py::plot_stiff_partials` + `scripts/diagnose_stiff_string.py`. **91 pytest pass,
ruff clean.**

**PDE:** `u_tt = c²u_xx − κ²u_xxxx − 2σu_t`. Stretched partials `fₙ = n·f₀·√(1+B·n²)`, `f₀=c/(2L)`,
`B = π²κ²/(c²L²)` (κ=2 → B≈9.87e-4, piano-ish). **Even the fundamental is stretched** off f₀ (≈+0.85
cents at B=1e-3) — assert `f₁=f₀√(1+B)`, not `f₁=f₀`.

**Load-bearing build decisions (some deviate from the plan-as-written):**
- **Biharmonic = `D2 @ D2`** (squared Dirichlet 2nd-difference), NOT hand-coded "5" boundary rows.
  This makes the energy SBP identity `⟨δ_xxxx f,g⟩=⟨δ_xx f,δ_xx g⟩` exact to machine precision, gives
  the correct `5/h⁴` boundary row for free, and keeps `sin(mπx/L)` an exact discrete eigenvector
  (eigenvalue `p⁴`) so the modal/dispersion harness carries over unchanged.
- **Energy `P(f,g)=⟨−𝓛f,g⟩` is computed through the SAME matrix 𝓛 used in the update** → conservation
  is an exact algebraic identity (drift ~1e-12, vs the 1e-10 bar). Energy form is θ-dependent:
  `Eⁿ=ρ[½‖δ_t⁻u‖² + (θ/2)(P_nn+P_pp) + (½−θ)P_np]`, reduces to IdealString's `½P(uⁿ,uⁿ⁻¹)` at θ=0.
- **θ default 0.28** (a hair above ¼, accuracy-first + small positivity margin). Unconditionally
  stable for θ≥¼ — `lam=ck/h>1` is ALLOWED (constructor does NOT reject it, unlike IdealString);
  tested at λ=2,4. κ=0 is the implicit wave scheme, NOT equal to the explicit IdealString (not even
  exact at λ=1) — assert self-consistency vs `discrete_stiff_mode_frequency(κ=0,θ)`, not equality.

**Achieved numbers:** energy drift ~1e-12 across κ/λ/θ sweeps (incl. λ>1); partials match the
discrete oracle to ~0.006 cents; B fit from simulated partials tracks π²κ²/(c²L²) to −4.4%(κ=1)→
−0.4%(κ=5), bias one-directional LOW (numerical dispersion flattens, bending sharpens); convergence
order ~1.99 (O(h²)); dispersion v_p/c RISES above c with mode (bending stiffens highs — opposite of
the ideal string's droop).

**KNOWN LIMITATION — damping caveat (documented in the plan):** the θ-scheme decays mode m at
`2σ(1−θQk²)`, not `2σ`. Low modes are fine (Qk²≪1) but high modes under-damp HARD (θQk² reaches
O(10) with stiffness → a broadband pluck retained ~100× the analytic energy, ~57% off e^{−2σt}).
Passivity (monotone decrease) still holds unconditionally; it's the *rate* that's wrong at HF, and
it's **backwards** from real strings (highs should die faster). Not a bug — a property of
frequency-independent loss θ-averaged at finite k. **Cured by model #3 (frequency-dependent loss).**
The decay-rate test therefore uses a LOW mode; lossless physics is unaffected.

**Portability-contract change (RESOLVED 2026-06-21 — human chose hardcoded allowlist):**
`string_stiff.py` is the first core module to import scipy, which transitively pulls
`charset_normalizer`/`cython_runtime`/a hash-suffixed mypyc runtime — these broke
`test_core_dependency_allowlist`'s old hardcoded `{numpy,scipy}` set (a latent bug: the core never
imported scipy before). I first rewrote it as a baseline-closure derive; **human reviewed and
preferred a hardcoded allowlist** for visibility. Final form: `_CORE_DEP_ALLOWLIST =
{numpy, scipy, charset_normalizer, cython_runtime, physsynth}`, the hash-suffixed `…__mypyc` runtime
matched by suffix (not name), underscore-private plumbing excluded by leading-`_` rule. Verified
empirically: bare interpreter pulls zero non-underscore third-party modules; the numpy/scipy stack
alone pulls exactly that set. If a new platform's scipy drags in an unlisted name, add it to
`_CORE_DEP_ALLOWLIST` — a deliberate reviewed edit (that visibility is the point). `docs/dev/
portability-contract.md` updated to match.

Next: **model #3 (frequency-dependent damping)** — the natural follow-up that fixes the damping
caveat; then Phase 3 (2D membrane/plate). Tolerances still inherit M1's bar (human's §11.5 call).
