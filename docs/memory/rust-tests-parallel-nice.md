---
name: rust-tests-parallel-nice
description: "Run Rust tests via scripts/cargo-test-nice.ps1 — parallel binaries, BelowNormal, one canonical build; QUICK lane by default (skip list), -Full before commit; goal = Rust faster than Python"
metadata:
  node_type: memory
  type: feedback
  originSessionId: a63c609d-849a-4c29-bb86-f193c9074edd
  modified: 2026-10-05T20:20:20.895Z
---

When running the Rust tests, run them **in parallel, at below-normal priority, without rebuilding
tests whose code did not change**, and the human's stated priority (2026-10-05) is that **the Rust
run is faster than the Python one** — "optimize the tests, fewer runs". Tool:
`scripts/cargo-test-nice.ps1` (PowerShell; launch from the PowerShell tool, always the same shell).

**Why:** plain `cargo test` runs ~90 test binaries one after another while one slow file idles the
cores; at normal priority it starves the human's own work (see [[test-suite-performance]]).

**How to apply:**
- Edit loop: `powershell -NoProfile -File scripts\cargo-test-nice.ps1` = QUICK lane, skips what
  `scripts\quick-skip.txt` lists (71 tests + 44 frozen scenes over 2 CPU-s each). **Before a
  commit: `-Full`.** CI never reads the skip list.
- Measured 2026-10-05, quiet box: quick 24.5 s, full 94 s, Python suite 45 s. Rebuild after a
  physsynth-core edit ~33-37 s on top (unavoidable: every test binary links the library).
- Narrow with `-Filter '<regex over package/target>'` (picks binaries to RUN); never `cargo test
  -p …` — a different package set re-resolves serde_json's `float_roundtrip` = separate build.
- `-TestArgs` under `-File`: pass several as ONE space-separated string (`'--nocapture --ignored'`);
  `a,b` arrives as the single string "a,b".
- **Never set `RUSTC_BOOTSTRAP` in an environment cargo builds in** — it is a build input and
  recompiled 8 crates (a 37 s rebuild, 2026-10-05). For per-test times (`-Z unstable-options
  --report-time`), set it only AFTER `cargo --no-run`, for the test exes alone.
- Choose skip entries by **CPU time** of each test run alone (`TotalProcessorTime`), not wall:
  another project often loads this box to 100% and wall times reorder (a 1.4 s scene read 21 s).
  The measuring scripts are in `W:\temp\claude\` (`cputime.ps1`, `cputime2.ps1`, `make-skip.ps1`,
  `frozen-timing\`); PowerShell's ConvertFrom-Json is case-insensitive and chokes on the frozen
  corpora (keys `E`/`e`) — list cases with the Rust `listcases` helper instead.
- No cargo-nextest: per-test processes would rebuild the `OnceLock` fixtures (whirl, phantom).
- The 4-thread cap per binary is deliberate: uncapped, frozen took 132 s vs 117 s (the minute-long
  scenes lost their cores to oversubscription; 16 logical cores).
