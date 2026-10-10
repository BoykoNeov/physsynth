---
name: retirement-phase-f-state
description: "Phase F (take Python apart, retirement plan §47+) — steps 1-3 (§47-§49) record the leaving libraries' answers natively; step 4 (2026-10-10, §50) the package guards gone, CI checks native in ci_workflow.rs; next and LAST: binding + shims + Python CI jobs"
metadata:
  node_type: memory
  type: project
  originSessionId: af072a4b-7d15-4a58-93c7-e6a5a9448efa
  modified: 2026-10-10T07:43:42.310Z
---

Phase F of `docs/dev/python-retirement-plan.md` = deleting what is left of Python, in §46.7's order,
one step per "go" from the human.

- **Step 1 (§47, 2026-10-08):** the frozen analysis record → `crates/physsynth-analysis/tests/` JSON. See [[analysis-freeze-state]].
- **Step 2 (§48, 2026-10-09):** `tests/test_binding_surface.py` was NOT deleted — only its tests
  whose referee is a leaving library (NumPy `np.sum`/`np.dot`, SciPy `D2 @ D2`) were recorded into
  `crates/physsynth-core/tests/reference/{numpy_reductions,scipy_biharmonic}.json` and carried
  (`tests/reductions.rs`, `ops.rs`, `connection*.rs`). Binding properties stay until the binding
  goes (that was the scope the human approved when step 1 was described to them). `physsynth/analysis/` deleted (the human: "they should
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
- A scripted prose move is not a diff: re-read the deleted file sentence by sentence (the review
  found two dropped passages).
- When a guard's population goes to zero (sibling-layer import check), delete it.
- `test_ci_workflow.py`'s two named positive controls ARE `test_binding_surface.py`: whichever step
  deletes that file must re-aim or delete them in the same commit.

- **Step 3 (§49, 2026-10-10):** `tests/test_rust_parity_ops2d.py` deleted; SciPy recorded into
  `crates/physsynth-core/tests/reference/scipy_ops2d.json` (Airy B_F right product, corner-average
  scatter, SuperLU on the 16 small grids, FNV fingerprints of `L @ L` + free-plate K). Human's
  calls: SuperLU small grids only (2.7 MB otherwise); big-grid backward error release-only
  (`ops2d_airy_large.rs`); F's other targets guarded by fingerprints. pytest 232 → 73.

**Lessons from step 3:**
- A plant's reach must be PROBED, not reasoned: fingerprint every builder instance before/after
  (a per-word FNV was too weak — hash per BYTE). §48 said F moves "the 2-D operators"; it also moved
  `L @ L` on staircased rims and the guitar free plate's K, which the Python file never tested.
- A canary taken after masking can be constant (a rim node is always 0) — keep the raw draw.
- A quoted measured minimum may be from a WIDER survey than the test's sweep (1.9e7 vs 4.686e9):
  re-run the old test to read its own number before quoting it.
- SciPy's left-associated Gram differs only because its intermediate is descending; natively both
  bracketings contract ascending → the parentheses plant is an equivalent mutant.
- Bash-tool heredocs break on apostrophes/backslashes here: write edit scripts with Write, run them.

- **Step 4 (§50, 2026-10-10):** `test_stability.py` + `test_ci_workflow.py` deleted. The shim
  identity guard MOVED into `test_binding_surface.py` (shims and binding die together; deleting it a
  step early leaves the claim unasserted). Three workflow checks → `crates/physsynth-core/tests/
  ci_workflow.rs`; the named-file check WIDENED to `cargo -p X --test Y` targets, control = the
  viewer freeze pair (the human's call). No guard on `std::fs`/`println!` in core src (declined,
  known gap, in `portability-contract.md`). pytest 73 → 67.

**Lessons from step 4:**
- Measure a limit's margin before carrying it: the "well under 120" line limit had a legitimate
  line at exactly 120. Raised to 160.
- A carried scan needs a positive control even if the original had none (the run-block scanner).
- Write each native check as a function over text and plant into text inside the test: the
  plants become permanent self-tests.
- A `--test $n` operand is a query too — skip `$`/`*` in the operand, not only the token.

**Next (§50.4), the last step:** `test_binding_surface.py`, `physsynth/`, `conftest.py`,
`pyproject.toml`, `scripts/nicepytest.py`, `crates/physsynth-py/`, and the `validate`/`lint` jobs +
the `rust` job's Python steps. Related: [[retirement-phase-c-carrying-state]],
[[rust-tests-parallel-nice]].
