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

## Phase D — consolidation: the models the viewer never surfaced

Phases A–C wired 6 of ~15 built models (3 strings, membrane, both plates, both von Kármán
boundaries). Everything since — tension #9, geometric #10, bow, reed, bore, mallet #7, barrier #8,
jawari, sympathetics, body/bridge, radiation — has no viewer at all. Against the project's
"interactive, beautifully visualized" vision that built-vs-shown gap is the biggest one open, so
Phase D is *consolidation*, not new physics. Group by **what the frontend needs**, not by model
number; only the current batch is firm (there are no hard dependencies between batches, so order is
pure value/effort and may flex).

Two capabilities cross-cut the batches — build each **once**:
- **Energy-BALANCE panel** (`E − E₀ == work_in − losses`) — a *third* verdict type beside the σ=0
  drift check and the σ>0 monotone check. Wanted by **bow and reed**. (Mallet is *conservation* —
  it rides the existing drift panel.)
- **Multi-field / orbit viz** — wanted by **geometric #10** (u/w/v, the whirl) and **sympathetics**
  (N lines).

### Batch 1 (DONE) — tension-modulated string #9

The de-risking pick: re-validates the seam on the cheapest model (reuses `drawString`, the 1-D line
path, and the existing convergence gate). Load-bearing decisions, all measured rather than assumed:

- **The headline is the amplitude *shift* `ω(A) − ω(A→0)`, never an absolute frequency.** A measured
  `ω(A)` carries the θ-scheme's linear temporal dispersion error, and `ω(A→0)` carries the same one,
  so their difference cancels it and isolates the nonlinear physics. This is the model's own oracle
  (`test_amplitude_shift_matches_duffing`); `duffing_frequency` alone matches only "loosely".
- **The measurement pair is LOSSLESS and short, and is NOT the audio run.** The oracle predicts the
  frequency at a *fixed* amplitude; a lossy tension string is a downward-gliding chirp, so
  zero-crossing the decaying pickup reports an amplitude-*averaged* frequency that undershoots — the
  panel would then diverge from the oracle as σ rose, reading as a bug that isn't one. The audio
  stays lossy on purpose (the glide is the signature). Two runs, two purposes. Pinned by
  `test_tension_shift_is_immune_to_loss` (bit-identical shift at σ=0 and σ=5).
- **IC = the mode-1 sine**, because the Kirchhoff–Carrier→Duffing reduction is a *single-mode*
  ansatz; a triangular pluck is broadly multi-mode and the shift would be a lying number (model #6's
  `mode11` lesson). Hence no `pluck_position` slider for this model.
- **Amplitude is the star control, re-ranged.** The shift scales as `A²`, and the string path's
  inherited `amplitude=1e-3` renders a 0.8-cent near-null — the flagship panel would look broken.
  Default 0.02 → ~270 cents.
- **Bound `dT/T0`, NOT amplitude** (`dT/T0 = EA·A²·p1²/(4T)`, exact and free). Amplitude is a proxy:
  EA and T move `dT/T0` just as hard, and `EA=2e5` at `A=0.06` would break up with an
  amplitude-only cap none the wiser (the membrane's "bound the actual problem size" lesson).
- **The "~3" breakup threshold is NOT mode-invariant** — it was measured at mode 3. Measured here for
  **mode 1** over a run 1000 fundamental periods long: `A=0.06` (`dT/T0=4.44`) stays PURE
  (off-mode 3.2e-13), `A=0.07` (`dT/T0=6.05`) BREAKS UP — so mode 1's threshold is in (4.44, 6.05].
  Also PURE at the cap with `kappa=0` (stiffness detunes the Mathieu resonance, so κ=0 *looked* like
  the dangerous corner; measured, it moves the threshold nowhere).
- **A guard that cannot trip in the reachable range is a *guarantee*, not dead code.** The cap makes
  breakup unreachable, so the purity gate is insurance — and is unit-tested by driving
  `_measure_tension_mode1` directly past the guard, where breakup is real and fast (N=64, A=0.20,
  20 periods → off-mode 6.9e-2). Do **not** widen the range just to make a gate fire.
- **This model needs its own cost budget.** Every step is a tension root-find: ~176 µs at N=128,
  about *2× a 2D membrane step* for a 1-D string. The string path's `N_MAX=2000` / 10 s would be ~4M
  root-finds. `TENSION_N_MAX=256`, `TENSION_AUDIO_MAX=3`, `TENSION_WORK_MAX=60_000` total steps.
- **Measure length buys nothing past ~12 periods**: the shift lands 1.47e-3 / 1.17e-3 / 1.03e-3 from
  the oracle at 6 / 12 / 24 periods, so the residual is the scheme's genuine `O(h²)+O(k²)` gap to the
  continuum Duffing, not crossing noise. Crossings *are* interpolated (`_interp_zero_cross_frequency`):
  the shared `_zero_cross_fundamental` quantizes to a whole sample and its mean spacing telescopes to
  `(zc[-1]−zc[0])/(M−1)`, so that error lands straight on the headline.
- `nonlinear_fraction` is read **at the IC** — the peak (max displacement, zero velocity ⇒ all of E
  potential ⇒ stretch maximal). The stretch oscillates twice a period, so the *final* state reports
  wherever the run stopped (0.115 vs 8e-6 for a 0.5 s vs 0.4 s render — same physics).
- Frontend seams generalized *additively*: the convergence branch takes optional `detail`/`note`
  wording (von Kármán's Picard text unchanged), and `drawDiagnostics` checks `kind === "tension"`
  **before** the `dims` gate (a 1-D model that still wants a spectrum panel). `amplitude` had to be
  added to `MODEL_RANGES._default` at 1e-3: it is shown only for tension, but `gatherParams` sends
  *every* slider, so without the reset the linear string models would re-render at 0.02.

### Later batches (rough map — not firm)

- **Excited strings** — bow (builds the balance panel), barrier #8, jawari (barrier profile drawn
  under the string).
- **Wind** — bore + reed (new field type: pressure along an `S(x)` profile; reed reuses the balance
  panel).
- **Mallet #7** — cheap: conservation energy + the membrane heatmap with a strike marker.
- **Geometric #10** — the multi-field/orbit viz; the richest panel in the project.
- **The parametric-instability demo** deserves its own batch with real viz (energy cascading into the
  neighbour modes, the Mathieu tongue) — *not* a bolt-on to justify batch 1's purity gate.

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
