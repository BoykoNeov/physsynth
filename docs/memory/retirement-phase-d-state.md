---
name: retirement-phase-d-state
description: "Phase D (viewer to Rust) IN PROGRESS — D1-D6 DONE 2026-09-28: all 22 keys native in crates/physsynth-viewer; D7 (freeze, headless port, switch, delete web/*.py) waits on the human"
metadata:
  node_type: memory
  type: project
  originSessionId: 34ac3fbb-bad2-4c84-9cfb-b3de45c0935d
  modified: 2026-09-28T14:26:38.935Z
---

Phase D of `docs/dev/python-retirement-plan.md` (§23), started 2026-09-28 when the user said "rebuild
the viewer in rust". Scope = plan §5: backend to Rust, `web/static/` JS untouched.

**Done:** D1 (§23.1–23.7) crate + std::net server + ideal/stiff/damped; D2 (§23.8) tension (duffing +
parametric) + bow; D3 (§23.9) sympathetic, geometric, reed (memo dropped), radbody, airload — all
139 D3 requests + 11 browser requests matched, 149 of the 150 to the bit; D4 (§23.10) body, jawari,
juari, fret + `physsynth_analysis::spectrum::rfft` (Bluestein, any length) — 73 + 8 browser, 0 failing. `physsynth-core/src/engine.rs` = native `simulate`, `Resonator::step` is FALLIBLE
(step failure → payload kind `internal`). No key is refused as `unported` any more. Python server
stays the live viewer until D7.

**How each batch is checked:** `W:\temp\claude\viewer-port\`:
- `dump.py <corpus>` records Python payloads (reinstall the wheel first);
- `compare.py <corpus>` runs the Rust `payload` mode and diffs parsed trees, exact unless a named
  class;
- `diffsummary.py <corpus>` groups differences by field;
- `corpus_browser.py` reads `requests.jsonl`, written by the Rust server when
  `PHYSSYNTH_VIEWER_REQUEST_LOG` is set during the headless run.

D1: 56/60 exact. D2: 41/60 exact + 19 in class, browser 7/7. Every TRAJECTORY bit-identical; only
post-run BLAS dot/norm quantities differ (off-mode fractions: use ABSOLUTE class 1e-15, not relative).

**Traps found:**
- `serde_json` makes NaN `null` silently, so every float goes via `py::num`.
- Needs the `float_roundtrip` feature.
- Python `round()` is ties-even.
- `ParamError` subclasses `ValueError`, so `try: int(_fnum(..)) except ValueError` REWORDS `_fnum`'s
  refusal (the `mode_number` message).
- The numpy RNG seed is frozen as 25 constants (`PARAM_SEED_COEF`).
- Monkeypatched constants become function arguments (`measure_mode1(.., periods)`,
  `build_payload_parametric_with(.., work_max)`).
- `cargo metadata` lists `cfg(any())` edges; the deps walker skips them.
- tiny_http pulls `log`.
- Write edit scripts to FILES: python heredocs with quotes break in this Bash tool.

**Headless check:** `VIEWER_BASE=http://127.0.0.1:8765 python <copy of verify_web_headless.py> string_`.
Start the Rust server via PowerShell Start-Process -PassThru and kill it by that PID. The script's
Chrome LEAKS: Chrome relaunches itself, so the PID it holds is a dead launcher. Afterwards, check
port 9333 and close via CDP `Browser.close` (never by name).

**Found in D3:** `uniform_filter1d` = running sum, divide on output (the only one of 3 orders that
matched SciPy); `np.geomspace` overwrites BOTH ends; the step-failure error kind follows the
binding's exception type (RuntimeError -> internal, ValueError -> construction).

**Found in D4:** a fitted decay rate needs an ABSOLUTE class (at sigma0 = 0 it is ~6e-12 of noise,
0.5 % apart relatively); the reference's NaN (body K=0 omega2) is emitted as `null` on purpose
(Python server would 500); the juari's `N` pre-read quirk (N=40.0 → main run's thread on the N=100
snap) is PORTED and PINNED by a test, fix only as a decision after the switch. The running viewer
server locks `target/release/physsynth-viewer.exe` — stop it (by PID) before `cargo test`.

**D5 so far (§23.11-23.12):** `physsynth_core::eigs::eigsh_shift_invert` (shift-invert, block of
3, full reorth, SplitMix start) + `eig::symmetric_eigen` (vectors). Membrane: 30 + 2 browser, 0
failing; markers identical to SciPy after the payload's 4-decimal rounding. A grid disk's ODD
angular pairs are exact repeats, EVEN ones split. D6 vkroom's per-mode shares are basis-dependent
on its square plate: decide (group sums vs mirror basis) BEFORE diffing.

**Found in D5 plate (§23.13):** the solver's convergence test must be the OUT-OF-BASIS residual
(ARPACK's), not `||Op y - theta y||` — on a free plate the shifted matrix is nearly singular and the
solve error floored the full residual, growing the basis to the whole space (minutes). Guitar
parities/overlaps matched SciPy exactly (sign-invariant by construction). Free plate at browser
defaults drifts 1.32e-10 > 1e-10 in the PYTHON too — model behaviour, flagged to the human.
`release_only` excludes by BASENAME, so viewer tests/plate.rs rides core's `plate` entry.

**Found in D5 bore (§23.14):** a band-normalized spectrum amplifies FFT rounding by (record peak /
band peak) — 1e6x at an anechoic bell, so its class is 1e-9 not 1e-12; `dispersion.order` at
lambda = 1 is a ratio of two rounding residues (meaningless in the reference too). Python's `.3e`
is `1.000e+03`, Rust's `1.000e3` — use a local `sci3`.

**Found in D6 airbox (§23.17):** 37/37 bit-exact. `np.percentile` linear = `(n-1) q` virtual index,
NOT the general `n q + (alpha + q(1-alpha-beta)) - 1` (last-bit different). `py::sci` / `py::fmt_g`
are the Python `.2e` / `:g` formats. vkroom: only `modal_drift(_twin)` is basis-dependent (per-mode
shares on a square free plate's repeated pairs) — MEASURED moot: rotating all 71 repeated pairs
changes no shipped digit (5 configs x 4 angles), so ported as defined. Dense generalized solve =
`eig::generalized_eigen_diag` (D-orthonormal). vkroom's ledger last bit = §16's deliberate
read-out spelling; `compare.py` MODEL_CLASSES keep that loosening vkroom-only. Final regression:
17 corpora, 561 requests, 0 failing. OPEN for the human: D7 go-ahead (deleting the Python viewer
= deleting the reference); the free plate's default render fails its own 1e-10 bar (1.32e-10,
identical in Python).

**Next (§23.7):**
- D6: airbox, vkroom (dense eigh vectors);
- D7: FREEZE the reference outputs as a native fixture first, then port the headless check, switch
  servers, delete `web/*.py` + `test_web_backend.py`. Only the reed has a cache (`_REED_SWEEP_MEMO`):
  dropped in D3, since Rust is fast enough.

Related: [[python-retirement-state]], [[retirement-phase-e-state]], [[web-viewer-state]].
