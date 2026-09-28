---
name: retirement-phase-d-state
description: "Phase D (viewer to Rust) DONE 2026-09-28 — all 22 keys native; D7: Python viewer deleted, outputs frozen (exact on Windows only, structure on Linux), browser check is a Cargo example"
metadata:
  node_type: memory
  type: project
  originSessionId: 34ac3fbb-bad2-4c84-9cfb-b3de45c0935d
  modified: 2026-09-28T19:23:13.958Z
---

Phase D of `docs/dev/python-retirement-plan.md` (§23), started and finished 2026-09-28 ("rebuild the
viewer in rust"; D7 go-ahead "do all of D7"). The served viewer is `crates/physsynth-viewer`
(`cargo run --release -p physsynth-viewer -- serve`); `web/` holds only `static/`.

**D7 (§23.19):**
- `tests/frozen.rs` + `tests/frozen/*.json` (3.3 MB): the Python's own payloads for 588 requests / 26
  corpora, a digest (long numeric lists and b64 buffers → length + FNV-1a; classed ones keep hash +
  64 samples + 64 block max/min). `compare.py`'s tolerance classes are ported as hand-written path
  matchers (no regex crate). Coverage derived from `MODELS` + `DOMAIN_OPTS` in app.js (airbox sends
  its select as `walls`). The Rust verdicts matched `compare.py` count for count on all 26 corpora.
- **Exactness is platform-bound, measured:** windows-latest 588/588 bit-exact, ubuntu-latest 158
  differ (last bits via C-runtime sin/exp/ln; parametric tension grows it to ~1e-6). So: EXACT on
  `x86_64-pc-windows-msvc` (new CI job `frozen-windows`), structure-only (keys/types/lengths/ints/
  strings) elsewhere. `orbit.u`/`orbit.w` are f32 buffers not named `…b64` — found by Linux.
- `examples/verify_headless.rs`: std-only WebSocket (a dev-dep would dodge `tests/deps.rs`),
  `--profile DIR --out DIR [filters]`; Chrome stub PID ≠ browser PID, so it closes via CDP
  `Browser.close` and waits for the port. 40/40 pass. DevTools ignores `Connection: close` — read
  by Content-Length.
- `tests/front_end.rs`: mark vocabulary app.js ↔ harness; harness covers every model.
- Free plate defaults: plate base N 48 / mu 2 (both rectangles; a `plate:free` key would break the
  supported↔free slider-keeping rule). Drift 1.49e-11 free, 1.60e-11 supported.
- Test accounting: `docs/dev/viewer-test-accounting.md`, 338/338 carried (2 added). pytest
  2,099 → 1,669 (429 + one `test_xdist_groups` parametrization).

**Traps found in D7:**
- A CI job inserted "after the last line of a step" landed INSIDE the job: the rest of the job's
  steps became the new job's. Append jobs at the end and yaml-load to list each job's steps.
- The running viewer server locks `target/release/physsynth-viewer.exe`; `cargo test --release`
  relinks it after any src edit — stop the server (by PID) first, or the test run prints nothing.
- Python `print` on Windows writes CRLF: `diff` two verdict files with `tr -d '\r'`.

**Left:** the binding's viewer-only names (`_stretch`, `_bridge_displacement`, `_support`, `_b`,
`_open_left`, `_open_right`, geometric state setters) are dead; they go with the binding.
`W:\temp\claude\viewer-port\ref\` (177 MB of Python payloads) is the freeze's source — keep until
nothing needs a re-freeze.

**Earlier batches (D1–D6), still useful:** `serde_json` makes NaN `null` silently → every float via
`py::num`; needs `float_roundtrip`; Python `round` is ties-even; `np.percentile` linear =
`(n-1)q`; the eigensolver's convergence is the OUT-OF-BASIS residual (ARPACK's); a band-normalized
spectrum amplifies FFT rounding by record/band peak; the juari `N` pre-read quirk is PORTED and
pinned. Details in plan §23.1–§23.18.

Related: [[python-retirement-state]], [[retirement-phase-e-state]], [[web-viewer-state]],
[[numpy-libm-cpu-dispatch]].
