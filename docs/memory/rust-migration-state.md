---
name: rust-migration-state
description: "Python is being retired entirely for Rust (2026-08-26) — scope is total incl. the test suite; the destination is NOT the order, and the 1e-15 cross-language promise is retracted"
metadata: 
  node_type: memory
  type: project
  originSessionId: d50bab7a-4456-48ae-8486-af5a6f9535f3
  modified: 2026-08-26T11:54:54.643Z
---

**2026-08-26, the human's call: Python goes — all of it.** Core, `analysis/`, the viewer backend
**and the 26k-line test suite**. Target **Rust**. New physics is written natively as soon as the
harness is proven (end of Phase 2), not derived in Python first. Plan:
`docs/dev/rust-migration-plan.md`. This **supersedes non-negotiable #3** in `CLAUDE.md` (struck
through in place, not deleted) and two clauses of `docs/dev/portability-contract.md`.

**The rule that makes a total scope safe: the DESTINATION IS NOT THE ORDER.** Per model — write the
Rust model → bind it → run the **existing unmodified Python tests** against it → port those tests →
re-point the viewer → delete the Python model and its tests *together*. No model ever exists without
a validated oracle. The failure mode this prevents is porting the tests **first** because they are
the biggest chunk: a Rust suite written against an unchecked Rust model asserts whatever that model
does — green and meaningless.

**Two founding promises retracted, not discovered later.** "Agreement to ~1e-15" is arithmetically
impossible across languages (sparse LU pivot orderings differ; even solve-free models miss
bit-identity because NumPy sums pairwise and dots through BLAS) — the acceptance bar is the
**physics harness**, which is what §6.1's 50× headroom always was, see
[[handoff-decisions-closed]]. And the three portability-contract enforcement tests change meaning by
construction once a compiled extension is in `sys.modules`; the hardcoded-allowlist policy from
[[stiff-string-state]] migrates to a **Cargo-side** check.

**The risk map is FOUR solver groups, not "seven models that solve a matrix"** (that first count was
wrong twice over). 10 files never solve anything · **4 use banded Cholesky — no pivoting, fixed
elimination order, so that group is nearly FREE** · 1 dense LU · only **6 are sparse LU**. Since
scipy's `splu` **is SuperLU**, linking SuperLU itself would collapse group D's risk — but that is a
**hypothesis Phase 4 tests** (`permc_spec`, `diag_pivot_thresh`, equilibration defaults all
unchecked), which is why the 254-line `beam` goes first, as it did the first time
([[beam-state]] de-risked [[plate-state]]). Files can appear in **two** groups.

**Two traps found by review, both after the plan read as finished.** `bow` is **not** a
transcription-only model — it runs a safeguarded Newton iteration and **delegates a banded solve to
the string it bows** ([[bow-state]]). And the **viewer is the binding's second consumer**:
`web/serialize.py` imports `physsynth` 21 times, so deleting a Python model without re-pointing it
turns 403 web tests red for a reason unrelated to the port — the binding is a **live dependency for
the whole migration**, not test scaffolding.

**Sizes that drive the schedule:** core is 13,413 lines but **43 % is docstrings** (~7,710 real) ·
tests are **3.4× the core's code volume and ARE the migration** · ~84 % of call sites bind
trivially, **~255 reach inside** (208 in tests, 47 in the viewer). See
[[test-suite-performance]] — cargo's thread parallelism drops xdist's **process isolation**, so
memory pressure, not test count, replaces sharding.
