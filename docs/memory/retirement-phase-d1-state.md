---
name: retirement-phase-d1-state
description: "Phase D batch 1 DONE 2026-09-28 — crates/physsynth-viewer (std::net server + payload builder), ideal/stiff/damped ported; 56/60 payloads bit-identical to Python; order of the other 19 derived by missing numerics"
metadata:
  node_type: memory
  type: project
  originSessionId: 34ac3fbb-bad2-4c84-9cfb-b3de45c0935d
  modified: 2026-09-28T14:06:16.531Z
---

Phase D of `docs/dev/python-retirement-plan.md` (§23), batch 1, 2026-09-28. The user said "rebuild
the viewer in rust"; the scope is the plan's §5 (backend to Rust, `web/static/` JS untouched).

**Done:** `crates/physsynth-viewer` (lib `simulate_to_payload` + bin `serve` / `payload`),
`physsynth-core/src/engine.rs` (native `simulate` + `Resonator` trait). Ported: ideal, stiff, damped.
Other 19 keys refused with error kind `unported`. Python server remains the live viewer until D7.
Workspace release suite 848 pass.

**How each batch is checked:** `W:\temp\claude\viewer-port\` — `dump.py <corpus>` records Python
payloads (reinstall the wheel first), `compare.py <corpus>` runs the Rust binary's `payload` mode and
compares parsed trees: exact unless a named tolerance class. Batch 1: 56 exact, 4 within class
(polyfit 1.9e-14 rel; audio near-zero cancellation noise — measure audio vs buffer PEAK, not ulps).
Audio resampler (transcribed resample_poly) came out bit-identical in 59/60 despite libm risk.
Headless check: `VIEWER_BASE=http://127.0.0.1:8765 python scripts/verify_web_headless.py string_`
(its Chrome is PID-owned; start the Rust server via PowerShell Start-Process -PassThru, kill by PID).

**Traps found:** `serde_json` makes NaN `null` silently → every float via `py::num` (marker +
top-level refusal). `float_roundtrip` feature needed. Python `round()` is ties-even; `round(x,n)` ==
Rust `format!("{:.n}")` round-trip (verified). Python `min(a,b,c)` NaN-first semantics. `cargo
metadata` lists `cfg(any())` edges (serde_json's serde pin) — deps walker skips that literal cfg.
tiny_http pulls `log` (measured) → std::net server.

**Next (§23.7, derived from missing numerics):** D2 tension+bow (none) · D3 geometric, sympathetic,
reed, radbody, airload (none) · D4 body, jawari, juari, fret (arbitrary-length rfft) · D5 membrane,
mallet, plate, vk, bore, platebody (sparse shift-invert eigsh) · D6 airbox, vkroom (dense eigh vecs) ·
D7 port headless check (WebSocket dev-dep), switch, delete web/*.py + test_web_backend.py. Four
monkeypatched constants need a decision when their model ports (§23.7).

Related: [[python-retirement-state]], [[retirement-phase-e-state]], [[web-viewer-state]].
