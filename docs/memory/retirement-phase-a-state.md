---
name: retirement-phase-a-state
description: "Python retirement phase A DONE 2026-09-28 — PHYSSYNTH_RS read by nothing; banded.py + its binding deleted, exciter re-exports, operators delegating wrappers; rust-harness CI job gone; default suite == old flagged suite (1,920 -> 1,940 reconciled)"
metadata:
  node_type: memory
  type: project
  modified: 2026-09-28T11:34:39.827Z
  originSessionId: 26999e72-85dd-4982-a12f-e7b0a699a567
---

Phase A of `docs/dev/python-retirement-plan.md` (§21), 2026-09-28. User said "continue with getting
rid of python"; I recommended phase A (smallest self-contained step, plan's forced order B → A) and
proceeded without asking; the advisor agreed.

**Done:** `exciter.py` = 3 re-exports (binding signatures identical, keywords included);
`operators.py` = delegating wrappers (`_csr` rebuild from triplets + `_asarray` coercion — the old
swap block's two seams as the whole body, same shape as `operators2d`); `banded.py` deleted whole
(only its parity file imported it) and `crates/physsynth-py/src/banded.rs` (113 lines of Rust) with
it; `test_rust_parity_{banded,operators}.py` deleted (116 cases), 28 cases harvested into
`test_binding_surface.py` (biharmonic re-aimed at SciPy's own `D2 @ D2`, inner at `h*np.dot`);
`rust-harness` CI job + `--exclude-parity` + its 2 guard tests deleted.

**Acceptance method worth reusing:** before editing, run the FLAGGED suite minus parity on a fresh
wheel (1,920); after, the unflagged suite (1,940); reconcile the difference **per file against a
`git worktree` of HEAD** (not `git stash`, not arithmetic): +28 harvest, −6 shard-partition, −2
`test_xdist_groups` rows parametrized over test files. `pyproject` addopts has `-q`, so
`--collect-only -q` prints NOTHING — grep `::` lines from plain `--collect-only` instead.

**Guards:** `test_the_rust_swap_matches_the_environment` → `test_no_module_chooses_between_two_
implementations`; `ported_expected`, `_USE_RUST` tuple, `if expected_rust:` block DELETED (all went
`x is x`); flag + `Py`/`_py` alias checks WIDENED to every `pkgutil` module.

**`test_rust_parity_ops2d.py` is NOT phase A** — its Python side went at unit 5; it compares against
SciPy. Stays until F. §2's "275 parity tests" counted it.

**Gotcha:** Bash-tool heredocs containing long markdown/Python with backticks+quotes failed with
"unexpected EOF while looking for matching `''" — write to a file with Write, then append/splice.
Several repo files are CRLF (`ci.yml`, `lib.rs`): exact-match Python replace fails; normalise or use Edit.

**Next:** D (viewer, longest pole), E (scripts), remaining uncarried physics bars (+ ops2d file).

Related: [[python-retirement-state]], [[rust-retirement-batch7-state]], [[parity-files-run-unflagged]].
