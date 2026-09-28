---
name: retirement-phase-d-state
description: "Phase D (viewer to Rust) IN PROGRESS — D1+D2 done 2026-09-28: crates/physsynth-viewer, 5 of 22 keys (ideal/stiff/damped/tension/bow); every trajectory bit-identical to the Python; next D3"
metadata:
  node_type: memory
  type: project
  originSessionId: 34ac3fbb-bad2-4c84-9cfb-b3de45c0935d
  modified: 2026-09-28T14:26:38.935Z
---

Phase D of `docs/dev/python-retirement-plan.md` (§23), started 2026-09-28 when the user said "rebuild
the viewer in rust". Scope = plan §5: backend to Rust, `web/static/` JS untouched.

**Done:** D1 (§23.1–23.7) crate + std::net server + ideal/stiff/damped; D2 (§23.8) tension (duffing +
parametric) + bow. `physsynth-core/src/engine.rs` = native `simulate`, `Resonator::step` is FALLIBLE
(step failure → payload kind `internal`). Other 17 keys refused with kind `unported`. Python server
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

**Next (§23.7):**
- D3: geometric, sympathetic, reed (loss-channel sum order matters), radbody, airload;
- D4: body, jawari, juari, fret (arbitrary-length rfft);
- D5: membrane, mallet, plate, vk, bore, platebody (eigsh);
- D6: airbox, vkroom (dense eigh vectors);
- D7: FREEZE the reference outputs as a native fixture first, then port the headless check, switch
  servers, delete `web/*.py` + `test_web_backend.py`. Only the reed has a cache (`_REED_SWEEP_MEMO`):
  measure first, and drop it if Rust is fast enough.

Related: [[python-retirement-state]], [[retirement-phase-e-state]], [[web-viewer-state]].
