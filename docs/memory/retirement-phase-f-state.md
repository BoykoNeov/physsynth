---
name: retirement-phase-f-state
description: "Phase F DONE 2026-10-17 (§51): Python at zero; §52 restored core's lost derivations, §53 (2026-10-18) the analysis crate's; nothing queued after"
metadata:
  node_type: memory
  type: project
  originSessionId: af072a4b-7d15-4a58-93c7-e6a5a9448efa
  modified: 2026-10-17T22:40:50.887Z
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
- (RESOLVED at §50: `test_ci_workflow.py` is gone. Its successor `ci_workflow.rs` has two controls
  that are literal workflow lines — `-p physsynth-viewer --test frozen` and, inside a `run: |`
  block, `cargo test -p physsynth-analysis --tests`. The workflow edit at the last step must keep
  them or re-aim the controls in the same commit.)

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
- "Only X holds this" must be PLANTED workspace-wide: the ideal string's message text was
  held by two viewer tests the grep never looked at (§50.5).

- **Step 5 (§51, 2026-10-17): DONE — Python is at zero.** Deleted the binding crate, `physsynth/`,
  `tests/test_binding_surface.py` (67 binding-property tests, none holding physics), `conftest.py`,
  `pyproject.toml`, `nicepytest.py`, CI `validate`/`lint` and the `rust` job's Python steps.
  Native 1,824 passed before and after. Binding crate had NO `#[test]`s (checked). Prose with no
  other copy carried first: `airbox.py`'s whole physics docstring (airbox.rs said "the reference
  docstrings there are the physics"), `LAM_LONG_WARN` etc. Plants: plate θ default and `VkSpec`
  `auto` default already held natively; beam θ held by NOTHING → the human: beam now IS
  `string_stiff::THETA_DEFAULT`. Damped `NegativeKappa` message: left unguarded (the human).

**Lessons from step 5:**
- Read every shim AGAINST the Rust file it points to: a Rust header saying "the Python docstring
  is the physics" means the deletion loses the derivation. TEN headers (first counted six by a
  phrase grep — read the line after EVERY "Port of" header) point at docstrings cut in Sept 2026.
- `sed s/X/Y/` on a plan file hits every earlier section with the same boilerplate line
  ("When the human says so." is in §46–§50): address a line by number.

- **§52 (2026-10-17): the ten derivations are back**, from `74af04b^`/`6c23a45^`/`f6a3a3c^`/
  `58654e5^`. Scope = the human's **option 3**: every module docstring + only class/method
  docstrings carrying a DERIVATION (proofs, worked equations, measured arguments) onto the native
  item of the same name; parameter remarks only where Rust said nothing. +1,690 lines, comments.

**Lessons from §52:**
- A header saying "the docstring is the physics" can mean CLASS or FUNCTION docstrings, not the
  module's: inventory every docstring (`ast` walk) and score each against Rust comments by word
  5-grams before scoping. A low score often means reworded, not lost — read before carrying.
- reST → rustdoc: an indented block is a DOCTEST — fence every literal as ```` ```text ````.
  Rewrap never starting a line with `- `/`#`/`>`/`1. `. Word-diff each block vs its source.
- `cargo doc` already has 6 warnings (other files), so `-D warnings` fails on HEAD: compare the
  warning list before/after instead.
- Git Bash rewrites any argv starting with `//` (e.g. `//!`) — `MSYS_NO_PATHCONV=1` plus `W:/`
  paths. Some src files are CRLF in the worktree; tools must normalise. `python -I` ignores
  `PYTHONIOENCODING` — use `-X utf8`.
- Scope grew mid-batch; the human wanted it surfaced as a choice, not one sentence.
- Faithful ≠ true today: carried text quotes PYTHON-measured figures (label them as such, re-measure
  before citing as Rust) and names retired pytests / Python kwargs (respell to native). Harvest
  `test_\w+`, `*.py|*.md`, kwargs from the added lines and check each (§52 follow-up, `0ad5018`).
- Under `MSYS_NO_PATHCONV=1`, `cmd //c` is NOT converted and opens an interactive prompt that
  hangs forever — unset it before the below-normal `cmd //v:on //c` wrapper.

- **§53 (2026-10-18): the analysis crate's derivations are back** (`b1d37c8^`, six modules,
  comments only), plus — the human's call, asked up front — every present-tense mention of the
  deleted parity files / binding in the analysis crate made past tense; `dispersion.rs`' Python
  import-hazard header shrunk to a history note.

**Lessons from §53:**
- `ast.get_docstring` misses ATTRIBUTE docstrings (bare string after an assignment: constants,
  NamedTuple fields) and `#` comment blocks — both held derivations. Walk every string `Expr` and
  every comment run.
- A carried docstring can be WRONG, not just stale: "roots from above" (false for root 2) and a
  self-contradicting "droops"/"rises". Check each behavioural claim against a native test; carry
  the corrected text and say what changed.
- Analysis crate may not intra-doc-link into core (`tests/deps.rs`): code spans only.
- A rewrap script must skip lines that were already over the limit before the edit — it flattened
  an aligned comment table in `spectrum.rs`' FFT.
- "N Python test files import this" counts are stale the day the Python goes.

**Next:** nothing queued — the retirement plan has no named work left; the direction is the
human's (HANDOFF §12). Related: [[retirement-phase-c-carrying-state]], [[rust-tests-parallel-nice]].
