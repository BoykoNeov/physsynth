---
name: retirement-phase-e-state
description: "Python retirement phase E DONE 2026-09-28 — 36 scripts + physsynth/viz deleted (9,590 lines), matplotlib gone; lam_long sweep -> first Cargo example; convergence edge is implementation-sensitive"
metadata:
  node_type: memory
  type: project
  originSessionId: 83e6ecf8-ce3d-4838-a2cf-608814834af3
  modified: 2026-09-28T12:02:49.858Z
---

Phase E of `docs/dev/python-retirement-plan.md` (§22), 2026-09-28. The user picked E ("2") from
D/E/bars; the human's calls: delete all 33 diagnostics, slim Rust sweep, delete freezers citing commit.

**Done:** 33 `diagnose_*.py`, `freeze_analysis.py`, `freeze_horizon.py`, `sweep_geometric_lam_long.py`,
`physsynth/viz/` deleted; `viz` extra + matplotlib out of `pyproject`. Collected suite unchanged
(2,099 ids, all pass); cargo 801 pass. Freezer provenance = commit `17efb1e`.

**Triage bar that worked:** not "is there a native bar" (that is the bars work item) but "does it print
a number recorded NOWHERE — no test, no doc?" CI never ran the scripts. AST-extract every `print`,
bin into validation-report / investigation-record / nowhere; grep distinctive numbers in `docs/`.
Zero "nowhere". A script's printed verdict is not a gate: `diagnose_membrane` printed passivity
False over a 6e-19 rise the test tolerates.

**Not deletable in E:** `shard_tests.py`, `shard_costs*`, `nicepytest.py` (CI + `test_shard_partition`
use them) go at F; `verify_web_headless.py` goes with D. Plan §6 was wrong on timing.

**Sweep finding:** it was already unimportable (subclassing a pyo3 class). Successor
`crates/physsynth-core/examples/geometric_lam_long.rs`. Checked against the original run at its own
commit in a `git worktree` (PHYSSYNTH_RS unset → Python): energy edge 9/9 identical, convergence edge
differs in 4/9 by one grid point both ways — the binding reproduces the example exactly, so it is the
Python→Rust move (a stall is a 1e-15 tolerance miss = last-bit event). The pinned test
`test_a_flat_energy_is_not_a_convergence_certificate...` now has half a grid step of margin
(2 stalls at lam_long 6, none at 5). Not fixed — recorded in scientific-hurdles §6.

**Gotchas:** running a diagnose script writes into the repo's `out/` (cwd-relative) — run scratch from
W:\temp\claude. Bash heredocs mangle em dashes; use Edit. Never `cd` in Bash (hook blocks it).

**Next:** D (viewer), remaining uncarried physics bars (+ ops2d parity file).

Related: [[retirement-phase-a-state]], [[python-retirement-state]], [[rust-phase5-outline-state]].
