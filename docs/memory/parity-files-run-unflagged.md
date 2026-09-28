---
name: parity-files-run-unflagged
description: "SUPERSEDED 2026-09-28 — PHYSSYNTH_RS is read by nothing since retirement phase A; historically parity files had to run WITHOUT the flag"
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

What stays true and is the reason to keep this note: **reinstall the wheel before believing any
number** — `pip install ./crates/physsynth-py` — because nothing in the suite can tell a stale wheel
from a fresh one ([[rust-phase5-wrappers-state]]).

Historical rule, for reading old plan sections: parity files built both sides themselves and patched
module globals, so running them with the flag set compared Rust against Rust and produced a spurious
failure that looked exactly like a real parity break. See also [[test-suite-performance]].
