---
name: milestone-1-state
description: "Physical Synthesis project state — Milestone 1 (ideal string + harness) is built and passing; what's decided and what's still open"
metadata: 
  node_type: memory
  type: project
  originSessionId: 54851fe8-37fb-409e-922c-6b9a1dc7f027
---

As of 2026-06-20, the project was initialized from HANDOFF.md and **Milestone 1 (§10) is
complete**: ideal-string FDTD solver + validation harness. Two commits on `main` (not pushed):
`71b90c1` scaffold (Phase 0), `98ee807` the milestone.

**Update 2026-06-21 — M1 polish: §6.5 dispersion added** (the one §6 validation test that was
missing). New: `physsynth/analysis/dispersion.py` (pure: `dispersion_frequencies`, `phase_velocity`),
`viz.plot_dispersion`, `tests/test_dispersion.py` (5 cases), `helpers.measure_mode_frequencies`,
and an `out/ideal_string_dispersion.png` figure (phase velocity v_p/c vs mode — λ=1 flat on the
continuum, λ<1 droops). Measurement uses the **modal-projection** trick: excite one exact eigenmode
`sin(mπx/L)`, measure q(t)=⟨u,φ_m⟩ (a pure tone) — robust to nodes that defeat a point pickup (mode
N/2 is zero at every even node). Numbers: λ=1 measured↔continuum worst 8e-10; λ=0.8 measured↔oracle
worst 1.3e-5, v_p/c monotonically 1.00→0.88 @ m=96. **42 pytest cases pass, `ruff check` clean.**

Achieved numbers (M1): lossless energy drift ~6.1e-15 at λ=1 (~3–7e-14 across a λ sweep; tol 1e-10);
partials within 0.003 cents of n·c/2L; convergence orders [2.015, 2.006] at λ=0.9 mode 8; passivity
monotonic with decay matching e^{-2σt}.

**Decisions taken on documented defaults** (HANDOFF §11 — were "ask the human", proceeded
autonomously and disclosed): Python (settled by non-negotiable #3); **explicit** scheme for M1
(handoff's own rec); diagnostics as script not notebook (better headless).

**Decided 2026-06-21 (closes §11.2):** the Phase-2 stiff string goes **straight to implicit**
(θ/μ-scheme, unconditionally stable) — human's call, skipping an explicit stiff-string pass. The
energy/modal/convergence/**dispersion** harness carries over; the stiff string's own dispersion
`fₙ = n·f₁·√(1+B·n²)` makes the new dispersion curve a direct stretched-partials check.

**Still genuinely the human's to set:** test tolerances (§11.5 — currently 1e-10 drift, ~1 cent;
dispersion adds provisional ORACLE_RTOL=1e-4, CONTINUUM_RTOL=1e-7); which models are polyphonic
(§11.3); first interactive-viz target (§11.4).

**Language strategy (resolved 2026-06-20):** the human floated switching to Julia, then Rust/C++.
Decided to **stay Python** for the research phase; the systems-language port is deferred to the
plugin stage. Why: discipline drift is language-independent (fixed by an *enforced boundary*, not a
new language), and the eventual port is the cheap, test-gated part by design (tiny hot kernel,
harness as the contract) — so Python maximizes physics-iteration speed + a cheap validation harness.
Real-time DAW-plugin tooling is immature in both Julia and Rust (C++/JUCE is the mature path), so the
native port likely happens either way. Added `docs/dev/portability-contract.md` + three `core/`
enforcement tests in `tests/test_stability.py`: headless blocklist, dependency **allowlist**
(numpy/scipy only, auto-discovers new submodules), and a no-sibling-imports direction check.

**CI** (`.github/workflows/ci.yml`, Python 3.11/3.12 on Linux) **first executed 2026-06-21** on the
initial push to `origin/main` (it had been written-but-never-run until the remote existed). Check
`gh run list` for status. `.claude/settings.json` permissions were NOT added (auto-mode blocked
wildcard self-modification); user can run `/fewer-permission-prompts`.

**Committed + pushed 2026-06-21 (batch end):** both logical changes are committed and pushed to
`main` per the new standing rule [[commit-push-at-batch-end]] — `9b3d299` portability contract
(`test_stability.py` +2 guards, `docs/dev/portability-contract.md`), `5a6497b` dispersion polish.
Tree clean. **Remote now exists:** `origin` = git@github.com:BoykoNeov/physsynth.git (PUBLIC,
created 2026-06-21 via `gh repo create`); local `main` tracks `origin/main`. Four commits on `main`.

Next: Phase 2 stiff string (implicit) — add `-κ²·u_xxxx` biharmonic, stretched partials
`fₙ = n·f₁·√(1+B·n²)`, tighter CFL; the energy/modal/convergence/dispersion harness carries over.
See `docs/dev/ideal-string-plan.md` for the full results table + dispersion design notes.
