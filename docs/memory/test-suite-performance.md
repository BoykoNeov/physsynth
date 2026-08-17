---
name: test-suite-performance
description: "The validation suite's cost, why it is BULK-bound and not bottleneck-bound, the sharded gate and its two guards, and why CI wall-clock cannot be compared across runs"
metadata:
  node_type: memory
  type: project
  originSessionId: aae47a22-23c6-4ed1-8713-4f9ae587e626
  modified: 2026-08-17T16:38:19.541Z
---

**1808 tests** as of 2026-08-17 (1721 before this batch), **5027.9 core-seconds** of measured work.
The gate is now **three concurrent jobs**, ~5-8 min wall depending on the runners, down from 15-21.

## The suite is BULK-bound — the "two pinned modules" story was wrong

The story for a week was that `test_geometric_whirl.py` and `test_geometric_phantom.py` were pinned
to one worker each and *were* the critical path. The first `--durations=0` run (the top-50 table had
never priced the tail: its 50th entry was 14.31 s with **1671 tests below it**) settles it:

- 5027.9 core-seconds measured against a 1266.51 s wall on **4 cores** = 5066 available.
  **99% of the budget is genuinely busy.** The longest chain was under half the wall.
- The confirming number had been on the record for a week, read backwards: the old fast lane
  deselects every one of those chains — 69 of 1721 tests — and still cost ~85% of the full run.

**Lesson worth generalising: "X is the longest chain" is not evidence that X sets the wall.** Ask
what fraction of the core-second budget is idle before optimising a chain. Measure the *tail*, not
the top-N — a top-50 table cannot see 97% of the tests.

## The gate is sharded, and the shard split is COMPUTED

Three jobs, one third of the **files** each, LPT-balanced on measured per-file cost:
**1723.5 / 1682.1 / 1682.2** core-seconds. Machinery, all new 2026-08-17:

- `scripts/shard_tests.py` — partitions `glob("tests/test_*.py")`. **Never a hand-written list**: a
  file in no shard never runs and every job is green. Shard count is ONE Python constant (`SHARDS`)
  that the workflow reads via `--matrix` / `--count`, because writing it in both places fails
  silently too (matrix `[1,2]` against a split computed `--of 3` runs two thirds and passes).
- `scripts/shard_costs.json` — a scheduling **hint**, never a claim. Stale means unbalanced, never
  untested. Refresh with `gh run view <id> --log | python scripts/shard_costs_from_durations.py`
  (it sums setup+call+teardown per file, and handles a multi-shard log in one pass).
- **Two guards, deliberately different.** `tests/test_shard_partition.py` asserts coverage from
  inside — but cannot catch a split that dropped *its own* file. The `checks` CI job asks pytest to
  collect each shard and the whole suite and does the subtraction from outside. That subtraction
  replaces the old `1721 − 1652 = 69` fast-lane ritual as the "nothing was silently dropped" check.

**Three shards and not four**, because `test_web_backend.py` alone is **1723.5 s = 34% of the
suite** and a file cannot split across machines. It is the floor; a fourth runner would idle against
it. Splitting that file is the next lever if the wall ever needs to halve again — and it is exactly
the file the repo deliberately refuses to fragment (it holds the wrapper contract), so that is a
decision, not a chore.

## Per-FIXTURE xdist groups — the coarse pin only became expensive once sharded

Pinning a whole module was free at a 15-min wall and fatal at a 4-min one. Measured in the first
sharded run: shard 2 finished in **375.27 s** and whirl's pinned chain inside it was **~372 s** — the
file *was* the wall, to within 1%. Shard 3: 490.92 s against phantom's ~426 s.

So the pin is now **one group per fixture**: whirl → `tongue` / `threshold` / `marginal`, phantom →
`ladder` / `circular`, and the two tests that need no fixture float free. **Nothing is rebuilt** —
the fixtures were already measured to share no work, which is why the coarse pin had nothing to buy.
`slow` stays at module level (that claim really is about the file).

After: whirl's longest chain 336 s against a 485 s wall (69%, was 99%). The variance-immune version
of that statement — **the same run, no cross-run comparison** — is that the whole file on that
machine would have been 721 s, i.e. more than the 485 s the shard actually took.

**The new failure mode is silent**, hence `tests/test_xdist_groups.py`: a test requesting `tongue`
under the neighbouring group rebuilds 200 s of string on another worker and **passes**. It reads all
70 files with `ast` (no imports, 0.3 s) and asserts each module-scoped fixture's consumers agree on
one group — plus a second test that these two files are still split per fixture and not re-merged.

## NEVER compare CI wall-clock across runs — and now, across SHARDS

GitHub runner class varies by **~1.6x**, and each shard is its own machine. Measured this batch:
the *unchanged* shard 1 (web-backend only, same 372 tests, untouched code) read **268.36 s** in one
run and **476.36 s** in the next. Two shards in the *same* run are no more comparable than two runs.

**What is legitimate: a within-shard, within-run comparison** — a chain's length against its own
shard's wall. That is how both claims above were made.

Local timings are worse still (~3x inflation, and the ranking reorders): the box runs the human's
own heavy Python. Run via `python scripts/nicepytest.py …` (BelowNormal / `os.nice(10)`; workers
inherit it) — the human asked for this explicitly again on 2026-08-17.

## Still true

- `--dist loadgroup`, **not** `loadscope`. `-n auto` in CI (4 cores); pick the count deliberately
  locally.
- `conftest.py` pins BLAS to one thread **only when `PYTEST_XDIST_WORKER` is set** — serial keeps
  16-thread OpenBLAS, ~14% faster; under `-n N` it inverts. Must happen before numpy imports.
- **NEVER pass `-q`** — `addopts` already sets it and `-qq` suppresses the `N passed` summary. This
  was live in CI itself until 2026-08-10 and is part of why 20 red runs went unread.
- The `slow` marker means a **CLAIM** ("this validation re-runs the thing it studies"), never a
  stopwatch reading. It has **no CI consumer any more** — the per-push 3.11 fast lane was deleted
  (at ~13 min it would have capped the sharded gate's win at 15%); 3.11 is covered by the nightly,
  which runs both interpreters full. `-m "not slow"` survives as a local edit/run-loop lane.
- **Windows trap:** `GetCurrentProcess()` needs an explicit `restype = wintypes.HANDLE`, or
  `SetPriorityClass` fails silently.
- Do the **collection arithmetic on every batch** (see [[free-plate-orthotropic-state]]): a green
  exit code cannot tell "all passed" from "half never ran". The shard sum is now that check.
- **A green local suite is NOT the gate.** 2026-08-17 produced the first case of CI and local
  disagreeing on a *result* (a bare `<= 0.0` monotonicity bar on a lossy energy array is a coin toss
  at step 0, because every test plucks from rest — see [[ci-runner-variance]] and the fix in
  `test_plate_orthotropic.py`). Check `gh run list --branch main` after every push, **and the
  previous commit's run too**.
