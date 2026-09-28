---
name: ci-flagged-suite-collapse
description: "The CI rust job's 21 per-batch steps became one 3-shard flagged run (49m -> 9m); the parity family is INSIDE the shards so it must be excluded, and excluded AFTER the split"
metadata: 
  node_type: memory
  type: project
  originSessionId: 2d381787-81fe-408c-aa3e-260eac2cbf68
  modified: 2026-09-03T09:41:38.873Z
---

Done 2026-09-03 as plan §35.7, first step of the spectrum batch. See
[[parity-files-run-unflagged]] for why the exclusion exists at all and
[[test-suite-performance]] for the sharding it reuses.

**The union of 21 subset lists was never the suite.** Each per-batch step named by hand the files
that batch's port could reach — a claim when written. "The flagged suite is green" was therefore an
*assumption* for as long as those steps existed. It was measured before anything was deleted:
**2,001 passed**, whole suite minus parity, `PHYSSYNTH_RS=1`. Deleting a check on the strength of a
claim the check never made is the failure checks exist to prevent.

**§35.7's instruction, taken literally, goes red.** It says run "the same three shards `validate`
uses" with the flag set — but `shard_tests.py` covers `tests/` **exactly once**, so
`test_rust_parity_*.py` are *inside* those shards, and a flagged parity file compares Rust against
Rust (assertions vacuous, negative controls red). A plan section can be wrong about its mechanism
while right about its intention.

**The exclusion must happen AFTER the split.** `scripts/shard_tests.py --exclude-parity` selects
the shard out of the full partition and only then filters. Filtering first hands LPT a different
file set → a *different partition* → flagged and unflagged runs both green, both complete, and a
file sitting in shard 2 of one and shard 3 of the other. Harmless until they disagree about what
exists. `tests/test_shard_partition.py` asserts both halves, plus a canary that the family is found
at all (an exclusion that excludes nothing fails green).

**Cost and result:** 3 matrix jobs each build the wheel, so the Rust compile is paid 3x in
parallel. The `CI` run went **49m46s → 9m21s**.

**Two guard notes.** `test_ci_workflow.py`'s named-file floor dropped 50 → 20 — a real weakening;
the parity list stays **literal** in the YAML so that canary keeps something to count. And the
guard had to learn a token containing `*` is a *query*, not a file name (the both-flags step derives
its list with `grep -l`). Writing that step reproduced §19.7's escaping bug **twice**; both were
caught by the guard, not by review. Build backslashes as `chr(92)`, or write no continuations.
