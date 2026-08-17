---
name: test-suite-performance
description: "The validation suite's cost, the CLEAN CI profile that inverts the local one, the slow-marker/fast-lane split, and why CI wall-clock cannot be compared across runs"
metadata: 
  node_type: memory
  type: project
  originSessionId: aae47a22-23c6-4ed1-8713-4f9ae587e626
  modified: 2026-08-17T10:44:56.403Z
---

**1638 tests** as of 2026-08-17 (was 1476 when the profile below was taken), ~23 min full on CI.
Reworked 2026-08-10 (see [[ci-runner-variance]] for the measurement trap that dominates all of this).

## The clean profile INVERTS the local one — the geometric family is the suite

`--durations=50` on an idle 4-core CI runner (added to CI 2026-08-10; before that no
quiet-machine profile existed at all). The top is **not** `test_web_backend.py`:

```
281s  test_geometric_energy::test_small_amplitude_recovers_three_linear_waves
252s  test_geometric_whirl   [module fixture setup]
217s  test_geometric_phantom [module fixture setup]
180s  test_geometric_whirl::test_the_tongue_does_not_move_with_the_grid
165s  test_geometric_limits::test_amplitude_shift_tracks_the_duffing_limit
 83s  test_web_backend::test_fret_brightness…   <- the top of the file I had profiled
```

**The shape matters more than the ranking.** `test_geometric_whirl.py` is `xdist_group`-pinned, so
its ~460 s of fixture setup + ~260 s of calls run on ONE worker — **~720 s of single chain against
a ~1400 s wall**. `test_geometric_phantom.py` is the same at ~425 s. **Two pinned modules are the
critical path**, and no worker count divides them.

**Verified irreducible before reaching for the marker:** the three whirl fixtures build 11 runs with
**measured zero overlap** (distinct `kappa_w`/amplitude/seed). Sharing is already maximal; the cost
is the physics. Check this first — overlapping fixtures would be free to fix, marking is not.

## The `slow` marker means a CLAIM, not a stopwatch reading

Was 20 marks, **all** in the string files (stiff/damped/dispersion/convergence), **zero** on the
geometric family, and no lane ever used it. Now 65 marks: whirl / phantom / limits at **module**
level (the claim is about the file — "a convergence study re-runs the thing it studies"), and
`test_geometric_energy.py` at **test** level for just 2 of its 26, because the other 24 are the
cheap energy/passivity assertions a fast lane most wants to keep.

Deliberately NOT marked: `test_web_backend.py` (fragmenting the one file holding the wrapper
contract), `vk_free`/`vk_modal`/`plate_stability` (~3 % of CPU for three more places a mark drifts).

**Never define the marker as "took more than N seconds"** — that is a measurement, and measurements
in this repo have now been wrong three times over for being taken on one machine.

CI = two jobs: full suite on 3.12 every push (the gate) + **fast lane on 3.11** with
`-m "not slow"`, which restores the per-push second-interpreter coverage the single-version matrix
gave up. Nightly + `workflow_dispatch` run both interpreters full. Push/PR no longer runs the whole
suite twice.

## Measured effect (controlled)

| | full suite | note |
|---|---|---|
| before | 1501.9 s | 3 failing |
| + `_sim` memo & fret split | **1366.1 s** | **−9.0 %** |
| + slow marker | 1376.9 s | marker cannot affect the full lane; +0.8 % is noise |
| fast lane, same commit | **725.0 s** | **1.90×**, 1411 of 1476 tests |

Only 9 % because **the memo does not touch the bottleneck** — it cuts `test_web_backend.py`, and the
geometric family is the critical path.

## NEVER compare CI wall-clock across runs

GitHub runner class varies by **~1.6×**. The *identical, untouched* whirl fixture measured
**252.05 / 252.09 / 158.50 / 252.76 s** across four runs. A 15:56 run and a 22:46 run of nearly the
same code differed only in which runner they landed on. **Always normalise against an unchanged
reference test in the same run before attributing a wall-clock change to your own edit** — the 9 %
above is trustworthy *only* because its two runs read 252.05 and 252.09 on that reference.

Local timings are worse still: the box runs the human's own heavy Python, inflating ~3× and
reordering the ranking outright.

## Still true from before

- `--dist loadgroup`, **not** `loadscope` (loadscope's floor is `test_web_backend.py`; 21m39s vs
  16m40s when measured at 1287 tests).
- `conftest.py` pins BLAS to one thread **only when `PYTEST_XDIST_WORKER` is set** — serial keeps
  16-thread OpenBLAS, ~14 % faster; under `-n N` it inverts. Must happen before numpy imports.
- Run via `python scripts/nicepytest.py …` (BelowNormal / `os.nice(10)`); workers inherit it.
- **NEVER pass `-q`** — `addopts` already sets it and `-qq` suppresses the `N passed` summary.
  **This was live in CI itself** until 2026-08-10 and is part of why 20 red runs went unread.
- **Windows trap:** `GetCurrentProcess()` needs an explicit `restype = wintypes.HANDLE`, or
  `SetPriorityClass` fails silently.
- **Unresolved:** two early parallel runs aborted at ~65 % with `EXIT=1`, no summary, never
  reproduced across many later full runs.

## 2026-08-10 — air-box batch 5

**1544** tests (was 1489): +44 `tests/test_airbox_membrane.py`, +11 in `test_airbox_surface.py`'s
new footprint section. Standalone the two new blocks cost about **40 s** together.

**The full-suite wall clock that day read 44 min and is NOT a usable reading** — two runs of
`scripts/diagnose_airbox_membrane.py` (~150 s each of single-core 3-D FDTD) and the seam-pin script
ran *concurrently* with it on the same machine. Recorded so a future session does not read 23 → 44
min as a regression. The rule this repeats: normalise against an unchanged reference measured in
the SAME run, or do not compare at all ([[ci-runner-variance]]).

## 2026-08-17 — airbox batch 6 + `StringVKPlateBridge`

**1638** tests (was 1544 after air-box batch 5): +~64 from [[air-box-state]] batch 6, +30 from
`tests/test_vk_connection.py` ([[string-vk-bridge-state]]), which costs ~19–23 s standalone.

The local full run read **2017 s (33:37)**, and — same trap as the 44-min reading above — it is
**NOT comparable**: ruff and a targeted 23 s re-run of `test_vk_connection.py` ran concurrently on
the same box, on top of the human's own Python. Green (exit 0), which is all it was for.
