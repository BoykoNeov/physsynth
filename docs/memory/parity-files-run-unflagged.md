---
name: parity-files-run-unflagged
description: "HISTORY — no flag since phase A (2026-09-28), no wheel or pytest since §51 (2026-10-17); decoder for old plan sections only"
metadata:
  node_type: memory
  type: project
  originSessionId: d551764f-a918-4308-9ed6-1f5b27277f07
  modified: 2026-09-28T11:34:29.465Z
---

**Superseded 2026-09-28 by retirement phase A** ([[retirement-phase-a-state]]): nothing reads
`PHYSSYNTH_RS` any more, the `rust-harness` CI job and `shard_tests.py --exclude-parity` are gone,
and the only file still named `test_rust_parity_*` (`ops2d`) compares against SciPy, not a Python
twin. Setting the variable now does nothing at all. Plain `pytest` is the one Python run.

**Fully superseded 2026-10-17 (retirement plan §51):** the wheel, its crate and pytest are deleted, so
the "reinstall the wheel before believing any number" rule that kept this note alive is gone too.
Kept only as a decoder for old plan sections ([[rust-phase5-wrappers-state]]).

Historical rule, for reading old plan sections: parity files built both sides themselves and patched
module globals, so running them with the flag set compared Rust against Rust and produced a spurious
failure that looked exactly like a real parity break. See also [[test-suite-performance]].
