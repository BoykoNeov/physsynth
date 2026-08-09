---
name: test-suite-performance
description: The validation suite is ~39 min serial; parallelised with pytest-xdist loadgroup — why loadscope was the wrong mode and why BLAS pinning is xdist-only
metadata: 
  node_type: memory
  type: project
  originSessionId: 71fc832f-6e8c-4733-b0be-4a941d69a247
  modified: 2026-08-09T20:09:40.664Z
---

The validation harness is **1355 tests across 63 files** as of 2026-08-09 (air-box batch 3). It was
**1287 across 62 files, ~39 min serial** when the parallelisation was measured and wired in earlier
the same day — **every timing in this note is a 1287-test figure** and none has been re-measured
since. Treat them as the shape of the problem, not as current wall-clock.

**The config** — `pytest -n auto --dist loadgroup` (CI) / `-n 8 --dist loadgroup` (local):

- `--dist loadgroup`, **not** `loadscope`. loadscope pins a whole module to one worker, and
  `tests/test_web_backend.py` is **336 tests / 16m18s on its own** — so loadscope's floor *is* that
  file, and it measured 21m39s vs 16m40s for loadgroup. loadgroup scatters everything except tests
  carrying `pytest.mark.xdist_group`.
- Only **21 tests** are grouped, and only because they share expensive *module-scoped fixtures*
  that scattering would rebuild per worker: `test_geometric_whirl.py` (whole module, ~110+55+45 s of
  setup), `test_geometric_phantom.py` (whole module, ~81+47+29 s), and the 7 `test_phantom_*` tests
  in `test_web_backend.py` (one shared ~23 s run). The other 329 web-backend tests scatter freely —
  that is where the win comes from.
- `conftest.py` pins BLAS to one thread **only when `PYTEST_XDIST_WORKER` is set**. Serial runs keep
  the full 16-thread OpenBLAS, which measured **~14% FASTER** than pinned; under `-n N` it inverts
  (N×16 threads thrash). Must happen before numpy imports — verified `"numpy" in sys.modules` is
  False at root-conftest import in both controller and workers.

**Do not quote a speedup ratio from the LOCAL numbers.** Both the serial and parallel local runs
were measured while the human's own heavy Python work was running (see
[[identify-processes-before-killing]]) and the parallel run was at BelowNormal priority. Per-test
call times ballooned ~3× under 8 workers, so the binding constraint was machine *throughput*, not
the critical path — meaning further group splitting would not have helped.

**The trustworthy number is CI's**, which runs on an idle 4-core runner: the last serial commit took
**39m50s**, the parallelising commit **24m31s** (includes install + lint). That is the quiet-machine
figure the local timings could not give — **both measured at 1287 tests**.

**Run it via `python scripts/nicepytest.py <pytest args…>`** — a priority wrapper (BelowNormal on
Windows, `os.nice(10)` on POSIX) that forwards everything to `pytest.main`. Workers inherit the
priority. Without it a parallel run saturates every core and the desktop goes sluggish.

**NEVER pass `-q` on the command line — `pyproject.toml` already sets `addopts = "-q"`, so your `-q`
makes it `-qq`, and double-quiet SUPPRESSES the `N passed in …` summary line entirely.** Diagnosed
2026-08-09 after a full green run finished with `EXIT=0`, every progress line through `[100%]`, and
no summary at all. It was first blamed on block-buffered stdout under redirection; that was wrong,
and `python -u` does not fix it. Confirmed both ways: `pytest tests/test_airbox_modal.py -q` prints
dots only, the same command without `-q` prints `23 passed in 19.10s`. **Just run
`python scripts/nicepytest.py -n 8 --dist loadgroup` with no verbosity flag.**

If a run does come back summary-less, it is not evidence of a failure. **The exit code is the
verdict**, and the progress marks are the count: total the `.` characters, subtract the 6 in the two
`bringing up nodes...` lines, and check for `F`/`E`. On the batch-3 run that gave exactly 1355 dots
and zero of either.

**Unresolved:** two early parallel runs aborted at ~3 min / ~65% with `EXIT=1`, no summary and no
junitxml. Four later full runs (including one with the identical `--junitxml` flag) completed all
green, and the abort never reproduced. No failure text was ever seen — and the `-qq` finding above
explains only the missing *summary*, not the missing failure text or the abort itself, so this stays
open.

**Windows trap:** `ctypes.windll.kernel32.GetCurrentProcess()` without an explicit
`restype = wintypes.HANDLE` returns the pseudo-handle `-1` as a 32-bit int, which zero-extends to
`0x00000000FFFFFFFF` on x64 — an invalid handle, so `SetPriorityClass` fails *silently*. Set
restype/argtypes or the priority drop does nothing. xdist workers inherit the priority class.
