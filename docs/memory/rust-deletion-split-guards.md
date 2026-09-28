---
name: rust-deletion-split-guards
description: "A deletion split across two commits empties the guard that would have covered it — and the airbox's parity family could not be split at all"
metadata: 
  node_type: memory
  type: project
  originSessionId: 29392b97-3e94-434a-b224-fe2c3350d010
  modified: 2026-09-03T19:56:40.147Z
---

Unit 6 (`airbox`, the 3-D room) was deleted in **halves** on the human's call — the room and three
ports first (plan §47, all four parity files, 2,990 lines), the seven wrappers, two mixins and three
seams second (§48). `airbox.py` went **4,112 → 245 lines**, the migration's largest deletion, and
the second half retired **no test file** (the parity family had already gone with the ports).
Four things came out of doing it that way.

**Why:** splitting changes what the existing guards can say, and two of them stopped saying anything
without failing.

**How to apply:**

- **`deleted_bodies` asserts three things at once** — the module reads no `PHYSSYNTH_RS`, defines no
  `<Name>Py` alias, and each named object *is* the Rust one. A half-deleted module satisfies only
  the third, so the entry cannot be added — and leaving it out means nothing asserts the room is the
  Rust room for the whole interval between the two commits, with every physics test still green
  (there is only one implementation left for them to run). The cure is a second table,
  `half_deleted_bodies`, asserting the identity third **and** that the module still reads the flag,
  so it cannot be forgotten when the halves rejoin. See [[rust-deletion-phase-state]].
- **The parity family does not partition along the tiers it tests.** All four
  `test_rust_parity_airbox*.py` files name a *port* alias, because the wrapper file deliberately
  builds a Rust wrapper over a **Python** port. So either half's deletion retires all four and the
  other half retires none — check which aliases each test file names before assuming the tests
  divide the way the classes do.
- **The guard surface is seven tables, not five.** The two beyond the usual list are
  `ported_expected` (the `<name>_py` *function* table in `tests/test_stability.py`) and the
  extension-import scan in `tests/test_ci_workflow.py`. The second failed at 7 ≥ 10: its population
  had been widened off the draining parity family onto "every test file mentioning the extension"
  and its docstring called that "a population that grows" — but the parity files mention it too, so
  the wider set is **mixed** and one deletion took four out. Widening a draining population is not
  the cure; **naming a permanent positive control** and asserting it is found is (a count was never
  the subject). Related: [[parity-files-run-unflagged]].

- **The half-table is DELETED at zero, not emptied.** An empty dict iterates zero times and asserts
  nothing while still reading as a live guard — an empty `parametrize` collecting as a skip through
  a second mechanism.

Reconciling the suite delta is worth the worktree: −291 tests decomposed exactly as 287 parity tests
plus four cases from `test_xdist_groups.py`, which parametrizes one per file in `tests/`. And after
the second half the suite reports the *same* counts 33% faster (105 s → 70 s unflagged), because
nothing was retired — only the Python that ran between the kernels.
