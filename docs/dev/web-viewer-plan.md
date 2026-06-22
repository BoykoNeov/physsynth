# Web viewer plan — local backend + live params (Phase 3.5)

> First departure from the physics ladder. Goal: an **interactive web viewer** for the existing
> models. Decided with the human: **architecture B** — a *local Python backend + web frontend*.
> Move a control → backend recomputes **offline** (seconds, not real-time) → streams frames + audio +
> energy back → the browser animates the vibrating object, plays the sound, and shows the live
> validation numbers. This honours the non-negotiables: **accuracy-first**, **real-time deferred to
> Phase 5** (no audio-callback / RT-safety work, no WASM), **headless core untouched**.

> **Status:** Phase A (string family) **+ Phase B (2D membrane) both built, tested, and
> browser-verified.** 37 web tests in `tests/test_web_backend.py` pin the payload contract (197 total,
> green); a headless-Chrome render (`scripts/verify_web_headless.py`, CDP over `websocket-client`)
> confirms the end-to-end pipeline for both — sliders build, `/simulate` round-trips, the base64 field
> decodes to the *correct* shape (string pluck / membrane mode, not byte-order garbage), and every
> panel shows the real backend numbers. Browser-verified energy drift survives in 2D: circle
> `5.07e-15`, rectangle `3.93e-15`, both **PASS** (conservation ⊥ geometry, as in the core); the
> string is unchanged at `5.51e-14`. Mode spectrum: FFT peaks land on the discrete eigenmode lines
> (fundamental `0.002 cents`), geometry tier circle `-13.94 c` vs rectangle `-0.05 c` — the O(h)
> staircase shown, not scored. **The web viewer (Phase 3.5) is complete;** next physics-ladder item is
> model #5, the Kirchhoff plate.

## Decisions (resolved with the human)

- **Architecture: B** — local Python backend + web frontend, offline recompute on param change.
- **Phase A model scope:** all three string models (ideal + stiff + damped) — shared render, per-model
  ctor/oracle dispatch.
- **Render trigger:** explicit Render button **+** debounced auto-render (~400 ms after slider release);
  Render disabled + "computing…" while a run is in flight.
- **Animation:** slow-mo physics window (~10 frames/period) with a playback-speed control, **decoupled**
  from audio length; full pickup plays as audio at 1× (advisor catch #2).

## Why B (not A or WASM)

- **WASM / live in-browser recompute = the Phase-5 port.** Foreclosed now by "offline rendering is
  fine; real-time is a later port." Not built here.
- **A (precomputed static bundles)** ships without a server but only switches among *canned* presets.
- **B (chosen)** gives real parameter exploration with **one** physics implementation (the Python
  core — no separate export path to drift), at the cost of being a *local dev tool*, not a hostable
  artifact. That trade is right for a fidelity-first project: the browser is a thin client; the
  single source of truth stays the validated Python core.

## Non-negotiable: the headless-core contract

- **No edits to `physsynth/core/`.** The server and serializer live in a new top-level `web/`
  wrapper (sibling of `scripts/`), depending on the core, never the reverse (HANDOFF §3.1 layering;
  "wrappers depend on the core").
- The engine *already* records decimated state snapshots + a pickup signal into a pure `SimResult`
  (`core/engine.py::simulate`, `snapshot_stride` / `pickup_index`) — that is the recorder. The
  wrapper only **serializes** that data and speaks HTTP. No new capture logic in core.
- Auto-checked: `tests/test_stability.py`'s core dependency-allowlist guard must keep passing with
  zero edits (it scans only `core/`).

## Architecture

```
browser (vanilla JS + Canvas2D, single page)
   │  POST /simulate  {model, params...}            GET / → static files
   ▼
web/server.py        stdlib http.server — routes, no framework, no new deps
   │  calls
   ▼
web/serialize.py     PURE (numpy only, no socket): params → resonator → simulate()
                     → decimate field (display grid ≪ sim grid) + decimate energy
                     → pack frames (base64 float32) + audio (base64 float32) + energy + drift
   │  imports
   ▼
physsynth.core (UNCHANGED)  +  physsynth.analysis (modal/spectrum, for the validation overlay)
```

- **Stack = the floor** (advisor): vanilla JS + Canvas2D, one `index.html` + `app.js` + `style.css`,
  no framework, no build step. WebGL/Three.js (3D membrane surface) is explicitly deferred.
- **Deps = stdlib only**: `http.server` (**`ThreadingHTTPServer`**, not the single-threaded
  `HTTPServer` — a multi-second compute must not block static-asset serving, catch #3), `json`,
  `base64`, `struct`. No Flask. Run with `python web/server.py` → open `http://localhost:8000`.
- **Resample** the pickup to a **fixed 48 kHz** in `serialize.py` (catch #1): `fs_sim` rides N and c
  (sliders) and can exceed the browser's `AudioBuffer` cap (~96 kHz) — resample (preserve pitch &
  duration), don't reinterpret. Use `scipy.signal.resample_poly` (SciPy already a core dep).
- **Clamp** N / durations / derived fs_sim with a clean `error` payload before constructing — a local
  tool will OOM on N=10000 or audio_duration=60 (catch #1 minor).

## The pure seam (so the backend is testable without a socket)

`web/serialize.py::simulate_to_payload(params: dict) -> dict` does *all* the work; `server.py` is a
thin HTTP shell over it. Tests call the pure function directly — no port binding.

**Request params (Phase A, 1D string).** Note the param set is **model-dependent** (see dispatch
table below); the frontend shows/hides κ, θ, σ₁ by model.
```
model: "ideal" | "stiff" | "damped"
L, T, rho                       # physical
N, lambda                       # discretization (sim fs derived: fs_sim = c·N/(L·λ); λ guard in core ctor)
sigma            (ideal/stiff)  # loss; for damped split into sigma0 + sigma1
kappa, theta     (stiff/damped) # stiffness + time-average weight
pluck_position, amplitude       # pluck pos as fraction of L, 0..1
pickup_position                 # fraction of L, 0..1
audio_duration                  # seconds of SOUND rendered (pickup at full fs_sim → resampled to 48 k)
animation_window                # seconds of PHYSICS shown in the animation (decoupled from audio — catch #2)
playback_speed                  # animation slow-mo factor (e.g. 0.02 = 50× slow); audio always 1×
```
- `fs_sim = c·N/(L·λ)`; **snapshot_stride resolves the fundamental**, not the wall clock: target
  ~10 frames per fundamental period ⇒ `stride = max(1, round(fs_sim / (f1 · frames_per_period)))`,
  capturing only the first `animation_window` seconds (catch #2 — a wall-clock 60 fps stride would be
  *longer than one oscillation* and alias the wiggle into slow beating).

**Response payload:**
```
model, lambda, fs_sim
audio:   {b64: float32, fs: 48000, peak}    # pickup RESAMPLED to fixed 48 k (catch #1), normalized; report peak
grid:    {x: [...]}                          # 1D node coords
frames:  base64 float32  + {n_frames, width} # decimated displacement fields (animation_window only)
frame_times: [...]                           # physics time of each frame (for the slow-mo clock)
energy:  {sigma_is_zero: bool,
          lossless: {time, value, drift}     # σ=0 → drift vs 1e-10 bar  (catch #4)
          lossy:    {time, value, monotone: bool, decay_rate, decay_oracle}}  # σ>0 → passivity, not "drift"
meta:    {c, f1, num_steps, partials?: {analytic:[...], detected:[...], cents:[...]}}
error?:  {kind, message}                     # construction guard / clamp violation → clean payload, not a 500
```

**Per-model dispatch + oracle (verified against the ctors — catch #5):**

| model | class | extra params | boundary | partials oracle |
|---|---|---|---|---|
| ideal | `IdealString` | `sigma` | `fixed` | `modal.harmonic_frequencies` |
| stiff | `StiffString` | `kappa, sigma, theta` | `supported` | `modal.stiff_harmonic_frequencies` |
| damped | `DampedStiffString` | `kappa, sigma0, sigma1, theta` | `supported` | `stiff_harmonic_frequencies` + per-mode `σ_eff=σ0+σ1·p²` |

## Build order — string first, membrane second (tightest-constraint-first)

**Phase A — 1D string MVP (proves the whole pipeline).** A wiggling polyline is trivial to render;
it exercises record → serialize → fetch → animate → sync WebAudio → transport end-to-end. Cover all
three 1D models (`ideal`/`stiff`/`damped`) — they share the line render, differ only by ctor + a
couple of params (κ/B for stiff, σ₁ for damped).
- `web/serialize.py` + `web/server.py` + `web/static/{index.html,app.js,style.css}`.
- Controls: model dropdown, sliders (T, ρ, N, λ, σ, pluck pos, pickup pos, duration), **Render**
  button (debounced; recompute is seconds, so explicit/debounced, not per-frame).
- Canvas: auto-scaled displacement polyline. Transport: play/pause/scrub/loop, audio via
  `AudioContext.createBuffer(1, len, fs)` (browser resamples; sim fs ≈ 20 kHz is fine).
- Live panels: **energy-vs-time trace + drift readout** (the project signature) and detected-vs-
  analytic partials (reuse `analysis.spectrum` / `analysis.modal`).

**Phase B — 2D membrane (the showpiece). DONE.** Heatmap render + the data-size handling. Split off
as `_build_payload_membrane` (the string path stays bit-for-bit unchanged); shared sub-blocks
(`_energy_block`, `_resample_normalize`, `_b64f32`) reused.
- `state` is a full 2D field (`membrane.py` embeds dead nodes as 0); frames = 2D, base64 float32 +
  `{nx, ny, dims:2}`. **Decimated spatially** to a ≤`DISPLAY_MAX`(=64) *display* grid (`stride =
  ceil(max(ny,nx)/64)`, same stride on field **and** mask so they stay aligned) and **temporally**
  at the fundamental-resolving stride (catch #2) — full sample rate stays only in the audio.
  `field_amp` + the colour scale come from the *decimated* frames we actually ship (not the full
  field), and the range is fixed symmetric `[−amp, +amp]` so decay stays visible.
- Render: Canvas2D `ImageData` heatmap (diverging cool/warm colormap, centred at 0) blitted to the
  main canvas at the physical aspect ratio (snapped `Ly` for a rectangle, read back off the ctor);
  the decimated mask blanks the exterior so the staircased rim reads. Circle + rectangle domains.
- Pickup as an (x,y) fraction → `membrane.pickup_index_at`; an (x,y) strike via `raised_cosine_2d`.
- **Modal panel = a mode *spectrum*, not per-partial cents bars** (advisor review 3): the FFT
  magnitude with vertical markers at the **discrete** eigenfreqs (`eigsh(-L)` →
  `discrete_membrane_eigenfrequency` — where the time-stepper actually rings; peaks landing on them =
  self-consistency) and fainter markers at the continuum oracle (Bessel/rect — the geometry tier,
  *shown not scored*). Two headline numbers: fundamental detected-vs-discrete cents (robust — the
  (0,1) mode is always excited) and the discrete-vs-continuum "geometry tier (O(h) staircase)" gap.
  Per-mode cents bars are a trap here: a struck drum rings high/odd modes weakly, so
  `measure_partials_near` would lock onto noise for unexcited targets and report confident garbage;
  and the Bessel gap is ~9 c *by design*. **Energy stays the hard pass/fail signature.**
- **Cost must be bounded by the actual problem size, not N** (advisor review 4 caught this — the
  first cut clamped N only and a thin rectangle still hung for ~45 s). Profiled: cost is pure FDTD
  (`eigsh` is ~0.06 s, negligible — *not* the bottleneck), with **two** independent drivers the
  sliders push:
  - per-step cost ∝ `n_live`, with a **razor-sharp ~3.2× cache cliff at `n_live ≈ 10_000`** (87 µs →
    281 µs/step as the working set crosses L2). A thin rectangle reaches it fast: `n_live ~
    N²·(Ly/Lx)`, so `Lx=0.3, Ly=2.0, N=100` is ~66k nodes. ⇒ `MEMBRANE_NLIVE_MAX=9_900` keeps
    `n_live` strictly below the cliff (admits a square/disk at N=100).
  - step count ∝ `fs = c/(λ·h) ∝ 1/min_dimension` — a small drum / fine grid inflates steps (the
    audio is resampled to 48 kHz regardless, so a high sim rate buys *no* fidelity). ⇒
    `MEMBRANE_WORK_MAX = 7e8` node-steps caps `n_live × total_steps` (audio + animation runs).
  - plus `MEMBRANE_N_MAX=100` (coarse gate), `MEMBRANE_AUDIO_MAX=2.0`, λ ≤ `1/√2`, and `eigsh
    k=min(12, n_live−1)`. Under all guards the worst *passing* render ≈ 5.6 s; the default (N=80
    disk) ≈ 2.7 s; the work budget scales with duration so a small drum still renders at short audio.
    The string's `N_MAX=2000` would OOM/hang in 2D.

## Tests — `tests/test_web_backend.py` (web wrapper, not core; keep core count stable)

Drive the pure `simulate_to_payload` with short durations:
1. **Energy honoured through the wrapper:** lossless ideal string → `payload.energy.drift < 1e-10`
   (the serializer must not corrupt the signature number).
2. **Frame bookkeeping:** `n_frames ≈ duration·display_fps` (rounding); 1D frame `width == len(x)`.
3. **Audio:** length `== num_steps+1`; all-finite; `peak ≤ 1` after normalization.
4. **Guard propagation:** `lambda > 1` (string) / `> 1/√2` (membrane) → raises at construction,
   surfaced as a clean error payload, not a 500/NaN.
5. **Passivity smoke:** `damped`, σ>0 → decimated energy monotone non-increasing.
6. **(Phase B) membrane payload** (17 tests): lossless drift `< 1e-10` through the 2D wrapper;
   `{nx,ny}` match the decoded buffer + frame_times and stay `≤ 64`; spatial decimation strictly
   shrinks the field at N=80; decoded peak `== field_amp` and exterior (mask==0) nodes clamp to 0 in
   every frame (the 2D analogue of the string boundary test — catches byte-order garbage *and* a
   field/mask stride mismatch); rectangle extent uses the snapped `Ly`; spectrum fundamental
   self-consistent `< 5 c`; σ>0 passivity at 2σ; the full guard battery (CFL `1/√2`, N ceiling, bad
   domain/geometry/pluck) → clean error payloads.

**Headless browser smoke (`scripts/verify_web_headless.py`, not in the pytest suite — needs Chrome +
a live server):** a minimal CDP driver (`websocket-client`, no Selenium/Puppeteer) navigates the
`?model=…&domain=…` deep-link, waits for the *real* "ok" status (not a virtual-time guess — the
`requestAnimationFrame` loop never goes idle), samples the canvas pixels to prove the field painted
(not background-only), and screenshots each case to `out/viewer_*.png`.

## Traps (pre-flagged; ✎ = added in advisor review 2)

- **Don't leak the server/serializer into `core/`.** Wrapper-only; re-run the allowlist guard.
- **Don't recompute per slider tick.** Recompute is offline-seconds; debounce or require Render —
  otherwise the UI feels broken (this is B's known cost vs A). ✎ Disable Render + show "computing…"
  while a run is in flight so clicks don't queue.
- **Audio rate ≠ display rate.** Decimate the *field* for animation; keep the pickup at full `fs_sim`
  for the audio path — conflating them either makes audio aliased or animation huge.
- **JSON size.** Pack frames + audio as **base64 float32**, not nested JSON number arrays (10× bloat).
- **Normalize audio**, but report `peak` so the energy/loudness story stays physical, not hidden.
- **Membrane data size** is the real scaling risk — enforce the display-grid decimation in Phase B,
  not as an afterthought.
- ✎ **catch #1 — don't reinterpret `fs_sim` as the AudioBuffer rate.** It can exceed the browser cap.
  Resample the pickup to a fixed 48 kHz; that's what `audio.fs` means.
- ✎ **catch #2 — animation stride must resolve the fundamental, not 60 fps.** A wall-clock stride is
  longer than one oscillation → the wiggle aliases into beating. Show a short `animation_window` of
  physics in slow-mo (≥10 frames/period) while the full pickup plays as audio; expose `playback_speed`.
- ✎ **catch #3 — `ThreadingHTTPServer`, not `HTTPServer`** (single-threaded blocks static assets
  mid-compute).
- ✎ **catch #4 — "drift" is a correctness figure only for σ=0.** For σ>0, `energy_drift` is large by
  design (energy decays); show the monotone-decrease / decay-rate check instead, or it reads as a bug.
- ✎ **catch #5 — per-model params + oracle differ** (table above). Model-dependent slider panel; the
  partials overlay must dispatch the right oracle (and pass the resonator's own `theta` to the
  discrete oracle where used).
```
