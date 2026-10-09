---
name: retirement-phase-f-state
description: "Phase F (take Python apart, retirement plan §47+) — step 1 frozen analysis record native (§47); step 2 (2026-10-09, §48) binding-file outside referees carried + physsynth/analysis deleted; next ops2d vs SciPy"
metadata:
  node_type: memory
  type: project
  originSessionId: af072a4b-7d15-4a58-93c7-e6a5a9448efa
  modified: 2026-10-09T01:26:44.528Z
---

Phase F of `docs/dev/python-retirement-plan.md` = deleting what is left of Python, in §46.7's order,
one step per "go" from the human.

- **Step 1 (§47, 2026-10-08):** the frozen analysis record → `crates/physsynth-analysis/tests/` JSON. See [[analysis-freeze-state]].
- **Step 2 (§48, 2026-10-09):** `tests/test_binding_surface.py` was NOT deleted — only its tests
  whose referee is a leaving library (NumPy `np.sum`/`np.dot`, SciPy `D2 @ D2`) were recorded into
  `crates/physsynth-core/tests/reference/{numpy_reductions,scipy_biharmonic}.json` and carried
  (`tests/reductions.rs`, `ops.rs`, `connection*.rs`). Binding properties stay until the binding
  goes (the human agreed to that scope). `physsynth/analysis/` deleted (the human: "they should
  go"); `horizon.py`'s long prose moved into `horizon.rs` first. pytest 259 → 232.

**Lessons worth keeping:**
- Record a library's answers with inputs from an LCG both languages rebuild exactly
  (`2u-1`, u64 wrapping) — store outputs + first/last canaries, not the inputs.
- `reduce.rs`' own structure tests were blind to a wrong accumulator combine order and to block
  size 128→64; only the Windows-only viewer freeze saw them natively. A transcription's own tests
  cannot certify it is NumPy's — a recorded NumPy answer can.
- Reversing a sparse product's contraction is an EQUIVALENT mutant on `B = D2@D2` (every entry's
  terms are palindromes) — it moves the 2-D operators instead, seen only by the viewer freeze.
  The ops2d step's record must catch it (§48.5 plant F).
- When a guard's population goes to zero (sibling-layer import check), delete it.
- `test_ci_workflow.py`'s two named positive controls ARE `test_binding_surface.py`: whichever step
  deletes that file must re-aim or delete them in the same commit.

**Next (§48.6):** `tests/test_rust_parity_ops2d.py` (159) against SciPy — record SciPy first; then
stability/ci guards; then the rest with `crates/physsynth-py/`. Related: [[retirement-phase-c-carrying-state]], [[rust-tests-parallel-nice]].
