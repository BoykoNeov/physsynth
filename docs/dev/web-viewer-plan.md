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
  it rides the existing drift panel.) **BUILT in batch 2** (`_balance_verdict` + `drawBalance`,
  behind `energy.kind === "balance"`); the reed gets it by passing `balance_work=` to
  `_energy_block`.
- **Multi-field / orbit viz** — wanted by **geometric #10** (u/w/v, the whirl) and **sympathetics**
  (N lines). **BUILT in batch 3** (`drawGeometric` = `drawOrbit` + `drawFields`, dispatched on
  `payload.model`; frames carry `fields: ["u","w","v"]` with `dims` still 1, so the string path's
  `(n_frames, width)` contract is untouched). Sympathetics reuses `drawFields`' stacked-strip layout;
  the accumulating-trail mechanism in `drawOrbit` is the part that is geometric-specific.

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

### Batch 2 (DONE) — the bowed string, and the energy-BALANCE panel

The infrastructure pick (chosen by the human over the mallet quick-win and the geometric-#10
showcase): it builds the balance panel, which the reed batch then gets for free, and it stays on
the 1-D `drawString` path batch 1 validated. 23 web tests added (86 → 109 web). Browser-verified:
`bow work 1.46e-2 J = stored + loss 1.42e-2 J`, `Helmholtz: 0.97 slips/period ✓ · pitch 97.9 Hz =
f₁ 100 Hz −37 c`. Load-bearing decisions, all measured:

- **The balance is a third verdict type because both older ones are *actively wrong* for a driven
  model — not merely weaker.** At σ=0 the bow pumps energy in, so `energy_drift` is enormous by
  design and the lossless branch reports a catastrophic FAIL; at σ>0 the energy *rises* from rest
  to the Helmholtz limit cycle, so the passivity/monotone branch FAILs too. Either would paint a
  red badge on a perfectly correct run, so the balance **replaces** both rather than joining them
  (`kind: "balance"`, dispatched before both, von-Kármán-override style). Pinned by
  `test_bow_balance_replaces_both_older_verdicts_because_both_would_lie`, which reconstructs the
  numbers the old branches *would* have reported.
- **The balance is itself σ-gated, and the lossy branch is deliberately NOT a residual.** At σ=0:
  `max|(E−E₀) − work| / (|E|+|work|)` vs 1e-11 → the money number (**2.2e-14**), the same
  normalization as `test_bow_energy.test_lossless_energy_balance`, computed on the **full**
  per-step arrays (a max over the decimated arrays samples fewer steps and understates it). At
  σ>0 dissipation is never measured — it is *inferred* as `work − ΔE` — so a "balance residual"
  there is **identically zero by construction**: a tautology, a green tick that cannot fail. The
  honest lossy content is the core's own criterion 2: inferred dissipation ≥ 0 and monotone
  non-decreasing.
- **No convergence gate, unlike von Kármán.** The balance is exact for *any* Newton residual (the
  friction force is applied exactly and the power read from the true post-correction velocity) —
  that is the model's whole trick. Copying VK's Picard gate would gate on something that cannot
  spoil the number.
- **Loss defaults ON — the opposite of batch 1, and load-bearing.** σ₀>0 settles the note to a
  steady Helmholtz limit cycle instead of growing without bound; σ₁>0 damps the high partials so
  the corner stays clean (one slip/period) rather than raucous (~18). Each model's default hides
  something, so each gets a nudge: tension's points at the glide, the bow's points at the exact
  closure (and warns the σ=0 tone grows without bound — the price of seeing it).
- **The slip-fraction oracle is claimed only inside the Helmholtz window.** `slip_fraction == β`
  and one slip/period describe *clean* bowing; outside Schelleng's window (which has a floor and a
  ceiling, both narrowing off the bridge) the note legitimately stops being Helmholtz. That is real
  physics, not a solver failure, so `slips_per_period` is always *reported* and the β-match is
  *scored* only when the motion is one-slip — else the panel labels it and scores nothing (the free
  cymbal's `null` "crash cascade" precedent). A zero onset count is ambiguous (never sticking and
  never slipping both give zero stick→slip transitions), so `regime` uses the slip fraction to say
  which. **Energy is structural, Helmholtz purity is dynamical**: the balance still passes on a
  raucous note — same split batch 1 found for the tension string.
- **A rate needs a long window.** `slips_per_period` is an integer onset count over the measured
  tail, so a short tail quantizes it coarsely: over 6 periods it can only land on multiples of 1/6,
  and clean one-slip motion read **0.83 at N=64** (and 1.00 at every other N) purely from where the
  window edge fell — straddling the 0.85 gate and mislabelling a clean note raucous. The tail is
  the settled last 40% of the run (the core's own `_bow_to_steady` choice, ~40 periods → ~0.025
  quantization). Caught by the tests, not by reading the code.
- **Its own cost budget**, like tension: every step is a friction root-find. `BOW_N_MAX=256`,
  `BOW_AUDIO_MAX=3`, `BOW_WORK_MAX=60_000` steps.
- **The animation needed no new viz** — Helmholtz motion *is* the string's shape, so the 1-D line
  path draws the travelling corner for free. It does need the *settled* window, though: from rest
  the first frames are a near-flat string, so the animation runs silently up to the tail before
  capturing.
- **Schelleng's window has no closed form in the core** (the tests just pick known-good points), so
  the hint reports `helmholtz_number` and the tests' empirical rule (`force ≈ 4·v_bow`; `force ≤
  0.4` clean to β=0.25) rather than inventing an Fmin/Fmax.
- **`_default` re-range leak, round two:** `gatherParams` sends every slider, so every param a model
  re-ranges must reset in `MODEL_RANGES._default`. The bow wants σ₁=0.05 — 25× the damped string's
  default *and* outside its own slider max of 0.01 — so without the reset a bow→damped switch
  renders a wildly over-damped string on a stale range. `sigma0`/`sigma1`/`pickup_position` joined
  `amplitude` (which also fixes the same leak tension's `sigma0=0` already had).

### Batch 3 (DONE) — the geometrically-exact string #10, and the orbit viz

The second cross-cutting capability, and the viewer's **first viz-only model**. 24 web tests added
(109 → 133 web). Load-bearing decisions, all measured:

- **There is no audio, and the reason is physics rather than budget.** `c_long = sqrt(EA/rho)` is
  ~22× the transverse `c` you hear (`EA/T ~ 500` on a real string), so resolving the longitudinal
  wave — which is what `lam_long <= 1` *means* — forces `fs ~ 22×` a normal string's. Measured:
  ~2 ms/step at small amplitude, ~4 ms on the whirl, and the cost is **N-independent** (it is
  per-Newton-iteration Python/`splu` overhead, not grid work), so one second of listenable audio at
  N=32 is ~318k steps ≈ **10 minutes**. There is no cheat: `lam_long > 1` is exactly the
  silent-garbage regime the model exists to warn about. The human chose **viz-only now, with the
  phantom-partials spectrum deferred to its own increment** (see below) over a stub player.
- **The orbit hero is the ROTATING WAVE, not the whirl** (advisor's first framing was the whirl and
  it was wrong; corrected on measurement). At the affordable 0.06 s the whirl grows ~60×, but from a
  `1e-3` *relative* seed that only reaches `max|w|/max|u|` ≈ **0.076** — which on equal axes still
  reads as a line (confirmed in the rendered screenshot). Saturation, an orbit you can *see* open,
  needs ~0.1 s ≈ 7.5k steps — past the work budget. The rotating wave, by contrast, is an **exact
  solution of the scheme**: round from frame 1, no growth needed — roundness **1.2e-12**,
  `long_kin/E` **1.5e-29**, in ~5 s. Pair it with the planar bit-exact line.
  - **Seed magnitude is a convention trap worth writing down.** `diagnose_geometric_string.py`'s
    figure-3 uses `w_dot = 1e-3·A·φ` (m/s) with **no `ω_u` factor**, so its initial out-of-plane
    *displacement* is ~`1e-3·A/Ω` — a **thousandfold** below a displacement seed of the same nominal
    `1e-3·A`. That, and not the growth rate, is why that figure needs 0.22 s (~60 s of compute) to
    saturate. The viewer follows the *test suite's* convention (`_whirl_run`'s `seed="vel"` →
    `dw' = s·A·ω_u·φ`), which makes the two seed kinds start at a comparable displacement, so
    switching kind changes the physics rather than the size of the perturbation. Measured
    `max|w|/max|u|` at 0.06 s: disp **7.6e-2**, vel **5.2e-2** — the same order, as intended.
- **The whirl is the growth-and-gate story, on a log-y ENVELOPE.** A straight line on log axes *is*
  the Mathieu instability. It must be the envelope (a sliding ~1-period max), never the raw
  `max|w|`: every node crosses zero twice a period, so the instantaneous spatial max is
  non-monotone (4.6e-8 → 1.8e-8 → 2.9e-7 → 1.5e-7 → 8.5e-7 on the default run) and the line is lost
  in the spikes. Enveloped, it is monotone **399/399** points. The family's recurring "never read an
  oscillating field at one phase" trap, in a new dress.
- **No new energy verdict type, unlike the bow.** Nothing drives this string — it is seeded and left
  alone — so the ordinary lossless drift check *is* the claim. What makes it a claim is what it
  survives: a parametric instability is energy **redistribution**, so the drift holds at **7.7e-13
  through a 63× blow-up**, which is what separates a whirl from the other thing that makes `|w|`
  grow orders of magnitude (a diverging solve). The growth ratio is therefore printed *next to* the
  drift, not in a panel of its own.
- **The tongue coordinate is the control.** `frac = δ/(εA²)` (δ = ω_w² − ω_u², the detuning κ_w
  buys) is dimensionless and **refinement-invariant**, so κ_w is recomputed from `p2` at the actual
  N (39.05 / 39.01 / 39.00 at N = 16 / 24 / 32) rather than pinned. Amplitude is bounded via
  **ΔT/T₀**, not A — at κ_u = 0, `εA²/ω_u² == ΔT/T₀` exactly, so the cap is model #9's own
  planar-breakup coordinate and the margin to its ≈3 is measured, not hoped for.
- **The rate has a closed-form oracle and it is Tier C.** `(Ω/2)√(q_M²−σ²)` with the **planar**
  Duffing `Ω = √(ω₀²+¾εA²)` — *not* the rotating wave's circular `√(ω₀²+εA²)`; the driven motion is
  a plane oscillation. Predicted **80.69** s⁻¹ vs measured **74.16** (ratio 0.92): the match runs
  5–11% and *systematically* low (leading-order ε, plus the seed's non-growing component), so it is
  reported and never scored. Measured off the **last two** quarter-envelopes — the seed is not the
  growing Floquet mode, so quarter one is contaminated by its decaying partner.
- **The "velocity seed, NEVER a displacement one" rule is only half right — measured.** Growth at
  t = 0.06 s: `frac` 0 / 0.07 / 0.25 / 0.5 / 0.8 → **disp** 1.00 / 14.69 / 60.17 / 6.08 / 1.17×;
  **vel** 6.88 / 28.52 / 63.00 / 0.85 / 0.78×. Inside the tongue a *displaced* seed grows perfectly
  well (60× at the peak); the pinning at 1.00× happens **only at frac = 0**, on the degenerate
  string, where `δw = δA·φ` at rest is exactly the rotation generator and the run is just planar
  motion in a rotated plane. So the viewer ships **both**, displacement as the default (it reads the
  tongue cleanly: 1.00× at frac=0 *is* "a degenerate string cannot whirl"), and the velocity seed as
  a toggle for the marginality story — the degenerate string then grows **secularly** (6.88×, linear
  in t), which the log-y envelope discriminates for free: secular *bends*, exponential is straight.
  The disp row also reproduces the earlier map exactly at frac = 0 and 0.07 (1.00, 14.7), which is
  what identifies that map's seed.
- **Both bit-exact zeros are kept as gates**: planar `max|w| == 0.0` (the `w → −w` reflection
  symmetry, not a small number — the orbit model #9 *can* draw), and the unseeded whirl at the
  tongue's centre, also **0.0** — without which every growth ratio would partly measure a leak.
- `lam_long` is a **hard cap in the viewer** though the core only warns (the core is right to warn:
  the scheme is genuinely unconditionally stable there, and `λ_long = 2` still conserves to 1e-12).
  The viewer must never render a *headline* in an unresolved regime. Its own budget:
  `GEOM_N_MAX = 32`, `GEOM_WORK_MAX = 6_000` steps (~25 s worst case, by far the slowest model).
- Frontend seams: `data-hide-domain` is a **new, additive** attribute — the inverse of `data-domain`,
  for sliders SHARED with a domain-less model (κ and amplitude, which the whirl derives). Plain
  `data-domain` would hide them under *every* model that has no secondary select. `u`/`w` share one
  animation-wide scale (their ratio is the claim); `v` gets its own (a different quantity, orders
  smaller) and each strip prints its own. Scales are computed once over the whole run — a per-frame
  autoscale would renormalize the whirl's growth away.

### Batch 4 (DONE) — phantom partials, the geometric string's bridge force

The debt batch 3 took on when model #10 shipped viz-only, and the discharge of model #9's **first
refusal**: #9's tension is a spatial *scalar*, so it has no longitudinal field and nowhere to put a
combination tone. #10's tension is a **field**, and its nonlinear excess carries `a·r²v_x/2` —
quadratic in the transverse fields, linear in the longitudinal one — so two transverse partials at
`f₁`, `f₂` drive `v` at `f_i ± f_j`. Conklin's phantom partials, read off the bridge force
`EA·v_x(0)`: what actually radiates in a piano. 13 web tests added (133 → 146). Browser-verified
13/13. Load-bearing decisions, all measured:

- **It reproduces `tests/test_geometric_phantom.py`'s rig EXACTLY** — same `lam_long`, 0.10 s window,
  two-mode IC, `v=0` start, blind band-limited detector — so the viewer *inherits the suite's
  validation* rather than asserting something new. It reports the suite's own numbers back: max peak
  error **0.039 Hz**, dominance **5.4×**, defect **4.57 Hz**. That match is the point, and it is also
  how the amplitude bug below got caught (at the wrong amplitude it read 0.035 / 4.7×).
- **`v` is NOT pre-solved to its quasi-static equilibrium**, though `v=0` is not the longitudinal
  equilibrium and the startup transient it radiates is the **largest** feature in the *full* bridge
  spectrum. Tempting, and wrong twice: the tests start from `v=0`, so a hand-rolled static solve puts
  numbers on screen no test backs (straight against "validation is code"); and it buys nothing — the
  transient sits at `c_long/2L ≈ 2236 Hz` while the phantoms live below 500 Hz, so the panel's band
  never shows it. That 4.5× separation is *why* the rig band-limits, and why the phantom band is
  purely forced response (which is also the piano physics).
- **The headline is the defect `f₂ − 2f₁`, in ONE run, with no oracle.** For a harmonic string the low
  phantoms coincide with partials *exactly* (`f₂−f₁ = f₁`, `2f₁ = f₂`), so the difference tone's
  distance to `f₁` and `2f₁`'s distance to `f₂` are **both** exactly `|f₂ − 2f₁|` — the same physical
  number approached from opposite sides (measured 4.56 / 4.55 against a defect of 4.57). The plan's
  other form (`f₁+f₂` landing ~9B·f₁ below `f₃`) is deliberately **not** computed: mode 3 is not
  excited, so it needs the discrete ladder as an oracle, and that oracle must be *earned* by a second
  amp→0 run — whose step count is amplitude-independent, so it costs the same ~45 s. 90 s for a
  strictly weaker statement. No hardening confound either: hardening moves phantoms and partials
  *together*, and measured it slightly **widens** the defect (works against the claim).
- **THE DEFECT IS NOT PURE STIFFNESS — the θ-scheme's dispersion drags `f₂` flat and contributes a
  NEGATIVE defect.** Measured off the linear discrete ladder at `lam_long=0.9`:

  | N | κ=0 | κ=2 | κ=8 |
  |---|-----|-----|-----|
  | 16 | −0.965 | −0.677 | +3.571 |
  | 24 | −0.430 | −0.137 | +4.168 |
  | 32 | −0.242 | +0.052 | +4.377 |

  At κ=0 it is *pure numerical dispersion*, converging as O(h²) (0.965/0.242 = **4.00** across a 2×
  refinement, exactly). Two consequences. **N is a second control, not just κ**: at κ=8 — plenty of
  stiffness — N=8 still cannot show it (+0.38). And the **κ=2 trap is worse than recorded**: at N=32
  dispersion very nearly *cancels* the true inharmonicity (+0.05 Hz net), and by N=16 it overwhelms
  it and the defect goes negative. Hence the gate is **one-sided (`>=`, never `abs`)**: a coarse grid
  displaces phantoms to the *wrong side* of the partials by artifact, which an `abs()` gate would
  happily score. Requiring a positive defect means "stretched by real stiffness, by enough to see".
- **Verdict ORDER is a trap, not a nicety.** `detect_peaks` returns `[]` on a zero signal rather than
  raising, so the `EA = T₀` control (`a = 0` ⇒ the fields decouple ⇒ the bridge force is **bit-exactly
  zero**) must be checked *before* the defect gate — otherwise a defect above the gate paints "the 4
  strongest peaks ARE the 4 combinations" over a spectrum with **no peaks in it at all**. Found by
  probing the detector, not by reading the code. That control is also nearly free: `EA = T₀` makes
  `c_long = c`, so `fs` collapses 159 kHz → 1.8 kHz and the run is 178 steps.
- **Two strips, because one axis cannot carry both halves of the claim.** The wide band (0 → 4.8 f₁,
  the tightest that still contains `2f₂`) carries "peaks at the four combinations". But there the
  4.6 Hz defect is ~4 px, so the half that says "and NOT on a partial" renders as its own opposite —
  the diagnose figure's 2 kHz trap, one level down. The zoom strip frames the `f₁ / (f₂−f₁)` pair,
  where 4.6 Hz over a 56.6 Hz axis is ~31 px and the two lines are plainly separate.
- **The display grid is separated from the measurement grid.** At the rig's 2× zero-pad the bins are
  4.85 Hz and the zoom band got **12 points** — not a picture. The drawn traces pad 32× (~190 points);
  the detector keeps the rig's default. Padding densifies the bin grid without adding resolution:
  exactly what a *plot* needs and nothing a *measurement* may lean on. Verified by the measured
  numbers being **bit-identical** before and after the change.
- **Peak POSITION is the claim, not resolution** — and the panel says so unasked. At a 0.1 s window
  the Hann main lobe is `4/T ≈ 40 Hz`, so `f₁` and `f₂−f₁` sit inside **one lobe** and the spectrum
  cannot resolve them. It does not have to: `f₁` is *absent* from `v`, so the phantom has no
  neighbour to be resolved from, and parabolic refinement locates it to ~0.04 Hz.
- **The audio debt discharges in its honest form.** This is the one geometric regime with sound, and
  only because its window is *already* 0.1 s of the radiating channel — the clip is free, not
  affordable. It is labelled for what it is: a 0.1 s blip, dominated by the longitudinal startup
  transient rather than by the phantoms (which are 4.5× lower, where the panel looks).
- **Its own budget, ~2.75× the other regimes'** (`GEOM_PHANTOM_WORK_MAX = 16_500`): ~45 s at the
  default N=32. Measured, not extrapolated — the bare Newton loop is 35.6 s and the panel telemetry
  carries it the rest. `GEOM_WORK_MAX` would reject the regime's own defaults. The window is fixed
  physics (the `animation_window` slider is ignored *and* hidden), so the budget really caps N ×
  lam_long. **The headless verifier's wait had to grow 40 s → 90 s**; at 40 s this case would have
  timed out mid-"computing…" and reported a false FAIL.
- **`MODEL_RANGES` merging was SHALLOW — a real bug the rendered picture caught.** A regime's
  `{val: 0.0015}` *replaced* the whole `{min,max,step,fixed}` object instead of overriding one field,
  so the slider kept `index.html`'s stale `step="0.001"` and snapped amplitude **0.0015 → 0.002**,
  quietly rendering the wrong physics (and leaving N's cap at 512 against `GEOM_N_MAX=32`). Now
  merged **per param** (`mergeSpecs`). The `_default`-leak trap, one level down — and the reason the
  panel now reports the core suite's exact numbers. Regime ranges are keyed `"model:domain"` and
  applied on domain change **only for models that declare them** (`hasRegimeRanges`), so a membrane
  circle→rectangle switch still does not reset the user's sliders.

### Batch 5 (DONE) — mallet → drum, model #7 (the first CONTACT model)

The cheap pick, and it paid off: it reuses the membrane heatmap wholesale, so the whole batch is
wire-up plus one shared-code catch. 15 mallet web tests (→ 161 web), full suite green, verifier
15/15, both PNGs eyeballed, the CDP switch-check ALL PASS. See `mallet-collision-state` memory for
model #7 itself. Load-bearing, all *measured* not assumed:

- **CONSERVATION rides the ordinary drift panel — no new verdict type (the closed-system contrast
  with the bow).** The mallet+felt+membrane is a *closed* system whose `energy()` includes the mallet
  KE, so `E₀ = ½M·v₀²` is a clean nonzero denominator and the σ=0 drift check works (1.5e-12 through a
  strike). The bow needed the balance panel only because it is driven from rest with `E₀ = 0`; this
  model is not driven, so its money test is plain conservation.
- **The blocking catch = `decay_oracle=False` (advisor, verified empirically before building on it).**
  With restitution ≈ 1 the mallet flies off force-free after the felt separates, so the total energy
  sits on a near-constant `½M·v₀²` floor with only a ~0.05 % membrane wiggle. `_fit_decay` over that
  near-flat trace reports `measured_2σ ≈ 0.0004` against `oracle_2σ = 4.0` — a lying "broken match"
  over physics that is actually fine. An additive flag drops the 2σ line, so the lossy verdict is
  pure **passivity** (monotone non-increasing). There is no closed form for mallet+hysteresis+
  membrane decay, so passivity *is* the honest lossy verdict. Ran the σ>0 probe first (0.0004 vs
  4.0), then built — the same "verify the number before wiring the panel" discipline as batch 1.
- **Contact duration is NOT the ~2 ms rigid-wall oracle — the coupled head yields.** On the
  compliant membrane the local reactive dimple relaxes, so contact lasts **~40-60 ms (~20×)** and
  scales with **√M** (M=0.02 → separates at 40 ms, restitution 0.9997, final head 0.05 %; M=0.05 →
  59 ms; peak head ~65 % of the strike mid-contact, a transient dimple). Consequence: the contact
  diagnostics must run over the *full* audio (which captures separation), not a short window — a
  rigid-wall "contact time" under-reports the coupled duration by the compliance ratio.
- **One instrumented audio loop, not `simulate()`.** `simulate()` gives energy + pickup but not the
  mallet internals (velocity, contact force, in-contact), and the contact IS the headline — so the
  audio run is a hand-rolled step loop capturing both, yielding the conservation number and the
  contact diagnostics in one pass. A `SimResult` is constructed from the captured energy/time for
  `_energy_block`. The animation run stays separate (fresh mallet from rest, fundamental stride).
- **The headline is the CONTACT, not the tone (advisor).** A point mass is an inefficient membrane
  exciter (restitution ≈ 1, the head keeps ~0.05 %), so the second panel is a *contact episode* — the
  mallet velocity crossing zero into a visible bounce, the felt-force pulse, and a separation line —
  NOT a mode spectrum, because a soft felt low-passes the strike and per-mode locking would lock onto
  noise. Retention is read at the **peak** (the core signature-test lesson). Reported, never tuned to
  ring louder: "a point mass barely rings a drum" is printed in the readout.
- **Strike marker = the snapped node** (the payload reports the ctor-snapped fractions, not the raw
  slider) — a filled red dot, distinct from the hollow yellow pickup cross. **`pluck_width` is hidden
  for the mallet** (a point contact has no width) via a `data-show` exclusion.
- **Its own budget, `MALLET_WORK_MAX = 3.5e8` = half the membrane's 7e8**, because a coupled step is
  1.5-3× a bare membrane step (a ~constant ~20 µs root-find, dominant at low N). `MALLET_N_MAX = 80`;
  both cost guards (n_live, node-steps) are reachable. Default N=60 renders in ~1.4 s, worst ~5.6 s.
- **The switch-driving CDP check (batch-4 lesson applied — and the trap resurfaced in the harness).**
  The deep-link verifier structurally cannot fire the model-switch handler, so a CDP-driven switch
  (autorender off) confirmed the transitions: felt sliders in + `pluck_width` hidden + sigma reset +
  domain populated on switch-into; mallet-only sliders out + no stale-range garbage on switch-away.
  Meta-trap: the check script's own f-string + plain-string `}}` concatenation silently produced a JS
  syntax error that no-op'd the switch, so it read the initial `ideal` state and (with a naive
  `.hidden` check) reported false results — the exact "a fresh-load verifier proves nothing about
  transitions" lesson, one level up in the *test* harness. Fixes: build the whole IIFE in one
  f-string; check effective visibility with `closest('[hidden]')` (headless has no `offsetParent`
  layout, and a slider div's own `.hidden` is false while its parent fieldset is hidden). The
  switch-check lives at `M:\claud_projects\temp\mallet_switch_check.py`.

### Batch 6 (DONE) — sympathetic / coupled strings (the first MULTI-STRING model)

The viewer's first multi-string model, the second customer of the stacked-strip `drawFields`, and
almost pure wire-up (the multi-field viz was built in batch 3). **J is fixed at 2** — the validated
oracles are two-string, and a free string-count slider would break them. 17 web tests (→ 178 web),
full suite green, verifier 17/17, both PNGs eyeballed, the CDP switch-check ALL PASS. See
`sympathetic-strings-state` memory for the core model. Load-bearing (advisor pre-build catches):

- **The claim lives in the SECOND panel, not the energy panel.** Energy conservation + passivity are
  *automatic* from the linear-leapfrog structure and pass even a flipped coupling sign → table-stakes
  green (σ=0 drift ~4e-14). So it rides the ORDINARY drift panel (a closed undriven system: no
  balance, no decay oracle). The money is the shared bridge displacement `w_b(t)` in the second panel.
- **`normal` regime runs BOTH ICs.** The antisymmetric start `u_B = -u_A` keeps `w_b == 0.0`
  *bit-exact* (IEEE float negation is exact) and `E_body == 0` forever; the symmetric contrast swings
  the bridge and loads the body to ~61 %. A flat-zero line alone reads as "broken", so both traces
  are shipped and plotted — the zero is only meaningful against the contrast (the whirl-needs-planar
  lesson). **The detune slider is gated OUT of `normal`** (any per-string difference degrades the
  bit-exact zero to ~1e-13); it belongs to `transfer` only.
- **`transfer` regime = the money visual.** Pluck string A over the full slosh; the per-string
  energy-fraction panel shows the classic coupled-oscillator exchange (unison: neighbour drains 88 %
  of the total; detune 4 semis: 9 %, a >3× ratio — the frequency-selective coupling). The full-slosh
  animation resolves the fundamental in ~1480 frames (`f1`·duration at ~10 frames/period lands just
  under MAX_FRAMES), so string A ringing down as B rings up is animated *and* the panel carries it.
- **A hand-rolled instrumented loop is mandatory.** `SympatheticStrings.state` is string 0 only, and
  `simulate()` gives neither the J stacked fields, nor `w_b(t)`, nor per-string energy — all three are
  what the panels are made of. Capture all, then *construct* a `SimResult` for `_energy_block` (the
  geometric/mallet pattern). Frames are `(n_frames, J, N+1)`, `fields: ["string A","string B"]`,
  `dims` still 1 — the string path is untouched and `drawFields` generalizes from 3 fixed strips to J.
- **Audio is a real string pickup, NOT body `pressure()`** (advisor): body pressure ≈ 0 on the
  antisym mode — that near-silence is real physics (the aftersound doesn't radiate) but would render
  as broken. The string still rings, so its pickup carries every regime. Sympathetics is *not*
  viz-only (fs ~22 kHz, unlike the geometric string's 22×), so audio is cheap and real.
- **`drawFields` generalized without disturbing geometric.** Field count / per-field amps / display
  names / a shared colour palette are module state set at load; geometric = 3 strips (u,w share
  `uwAmp`, v its own) with its three original labels + colours (bit-identical picture, re-verified in
  the PNG), sympathetic = J strips all sharing one amp (one string ringing up while the other rings
  down is the picture — a per-strip autoscale would flatten it). No orbit (unlike geometric): `w_b`
  goes in the panel, so the strings get the full canvas width.
- **The `normal` regime runs TWICE**, so its work budget is on `2·n_steps`.
- **Latent bug fixed in passing:** the domain-select container's `data-show` never listed
  `geometric`, so the geometric *regime* dropdown was reachable only by `?domain=` deep-link, never
  interactively. Added `geometric sympathetic` — the CDP switch-check confirms both regime selects now
  work interactively (the batch-4 lesson: a deep-link verifier proves nothing about the switch).
- The switch-check (`M:\claud_projects\temp\symp_switch_check.py`) also confirms the regime gate
  RECOMPUTES both ways (detune hidden in normal → shown in transfer → hidden again, no latch), K
  resets 8000↔1500 per regime, and switch-away leaves no out-of-range sliders.

### Batch 7 (DONE) — Weinreich two-stage decay (the first body-loss slider)

The deferred third sympathetic regime, and the FIRST body-loss slider in the viewer. Strike ONE of
two near-unison strings over a LOSSY bridge: the symmetric normal mode loads the lossy body and dies
fast (the "prompt"); the antisymmetric mode barely loads it and lingers (the "aftersound"). The
string-energy envelope shows the fast-then-slow knee of a real piano unison. All-wrapper (a
concurrent full-suite run stayed green; `physsynth/core` untouched). 11 web tests added (→ 189 web);
verifier + CDP switch-check both fully green. Load-bearing, all *measured* not assumed:

- **`sigma_body` gates the VERDICT, not just the physics.** At σ_body = 0 the closed system
  conserves → the ordinary drift check (`sigma_zero=True`); at σ_body > 0 it flips to **passivity
  with `decay_oracle=False`** — the total energy is a two-rate decay to a *nonzero aftersound floor*,
  so a single fitted 2σ against a flat oracle would be a lying "broken match" (the mallet's
  `decay_oracle=False` case, second customer). There is no closed form for the coupled modal decay
  over an off-harmonic body, so passivity IS the honest verdict; the two slopes are the claim. The
  guard is σ-blind (built from the lossless `omega`-only leapfrog operator), so a lossy body only
  adds headroom — it cannot destabilise a config the σ = 0 guard passed (advisor, verified).
- **The anchor sub-claim ties back to batch 6's normal-mode oracle.** Strike-one = ½ symmetric +
  ½ antisymmetric; at detune = 0 the antisymmetric mode is bit-exactly bridge-decoupled (w_b ≡ 0),
  so the body's damping *never activates on it* and its tail is **lossless in the discrete scheme,
  exactly** → aftersound slope ≈ 0, rising clearly with detune (measured 0.02 → 0.23 s⁻¹ at
  detune 0 → 0.3 semis). The floor value (~50 %) is *report-only* (the ~50/50 split is approximate).
- **Two runs, and strike-both is NOT droppable.** The contrast is plucking BOTH strings (the pure
  symmetric mode): it loads the body fully and decays away single-slope (~5 %), no aftersound. This
  proves the strike-one plateau is the un-decaying antisymmetric mode, *not a noise floor* — remove
  that excitation and the energy DOES decay away. Budget: 2 × n_steps, same `SYMP_WORK_MAX`
  (default 2 s × 2 ≈ 89 k < 130 k; a high T/N trips it cleanly).
- **The ripple trap (advisor).** String energy `E_A + E_B` EXCLUDES `E_conn`/`E_body`, so it
  oscillates (string↔spring slosh + beating) even in a lossless tail — a raw read/polyfit measures
  ripple, not decay. Fix: a **sliding-MEAN envelope** over one fundamental period (`uniform_filter1d`;
  the whirl's sliding-envelope precedent, mean not max — max sits on the slosh peaks, the mean is the
  average stored string energy). The envelope's first point dips slightly below 1 (the centered mean
  over the prompt's fast decay) — cosmetic, tested for `[0.85, 1.05]`.
- **The rate fit handles a SUM of exponentials.** `log(E_str)` curves through the knee (fast mode +
  slow/flat mode), so a naive early fit reads the curvature. The antisymmetric plateau is a real
  floor → the **prompt** rate is fit on `log(E_env − floor)` over the steep part, the **aftersound**
  rate on `log(E_env)` over the late window. None of the rates is a validated oracle (report-only,
  like the mallet's contact time); the sharp claim is the *aftersound-vs-detune* trend above.
- **Params, all measured, not the diagnose rig's verbatim.** K = 6000 (its own MODEL_RANGES key),
  σ_body default 20 (a visible prompt + a long aftersound in ~2 s; the diagnose's 10 was too weak —
  the prompt read 0.6 s⁻¹, no visible knee), max 80 (heavier → an invisible cliff). detune re-ranged
  to a **fine 0..0.4 semis, step 0.01 (~1 cent)** — a piano unison is mistuned by a few cents, not
  transfer's 0..12 semitones. `_default` provides the semitone fallback range, so weinreich→transfer
  restores it; **`sigma_body` reset in `_default`** (the recurring leak). detune default 0 leads with
  the exact invariant; the hint dials it up for the realistic finite aftersound.
- **Wiring gotchas cleared:** the `{"domain": "weinreich"}` bad-param test was updated to `nonesuch`
  (weinreich is now valid); `drawEnergy`'s `decay_oracle=False` readout branched (mallet's "½M·v₀²
  floor" wording is wrong here → "two-rate decay to a nonzero aftersound floor"); `symp-hint` +
  `drawSympathetic` (log-y, `drawWhirl` precedent) + `drawDiagnostics` title all gained a weinreich
  branch. The CDP switch-check (`M:\claud_projects\temp\wein_switch_check.py`) drove the sliders
  (the deep-link verifier structurally can't): sigma_body appears only in weinreich (recomputes, no
  latch), detune range 0.4↔12 per regime, and the RENDER checks confirm the verdict TOGGLE —
  σ_body = 0 flips "passive"→"conserved" + "nothing decays" panel; detune 0.3 → "cents mistuned".

### Batch 8 (DONE) — the jawari / buzzing bridge (the first BARRIER model in the viewer)

The sitar & tanpura bridge, and the viewer's first contact model with a *distributed, persistent*
contact (the mallet, batch 5, was a single point that separates). It is **not new core physics**: a
`BarrierString` (model #8) whose barrier is a parabola hugging the termination. The string wraps
onto the curve each downswing and its departure point travels along it, re-injecting high partials —
the shimmer. All-wrapper; `physsynth/core` untouched. 11 web tests added (→ 200 web). Load-bearing,
every number *measured before the wiring*, in two probes:

- **The claim is the shimmer, NOT the energy** (batch 6's lesson in a new model). Energy
  conservation through contact passes for a *flat rail* too — model #8 already gates it — so it is
  table-stakes green, not what makes this a jawari. The headline is the **late-window spectral
  centroid elevation over a clean string** (measured **3.44×**, gate 2.5×) and the second panel is
  the **travelling wrap edge** (std **4.89** over nodes 0–14).
- **The energy panel is the damped string's, UNCHANGED — `decay_oracle` stays TRUE.** This is the
  batch's one genuine reversal of a prior template. The bridge is a *lossless elastic* barrier: it
  redistributes energy into the highs but **dissipates none**, so every mode still decays at exactly
  `2σ₀` and the flat-loss oracle survives the wrap (**measured 2σ = 1.009 against oracle 1.000, log-
  fit rms 0.000**). Neither the mallet's nor weinreich's `decay_oracle=False` applies: the mallet is
  a closed system on a ½Mv₀² floor, weinreich is two rates to a nonzero floor — this is a plain
  resonator decaying toward rest, and dropping the oracle would throw away a *strictly stronger*
  verdict for a weaker one. **Generalizable: `decay_oracle=False` is for models with no single-
  exponential form, not for "anything with contact in it" — check the dissipation, not the vibe.**
- **σ₀ gates the verdict** (the weinreich pattern, second customer): σ₀ = 0 → conservation drift
  *through the sustained curved wrap* (**1.0e-12**, with contact duty 42 % of the run, so it is a
  drift through real many-node contact, not a contact-free run proving nothing); σ₀ > 0 → passivity
  + the 2σ₀ oracle.
- **`sigma1` and `hysteresis` are FIXED at 0 and deliberately not exposed.** Both would break the
  oracle above for nothing: σ₁ makes the highs decay faster than 2σ₀ (**measured 2.77 against a
  1.00 oracle** — a 2.7× "mismatch" on correct physics), and hysteresis is contact *damping*, which
  fights the very re-injection the model is about. The 3.44× headline is a σ₁ = 0 result.
- **Loss defaults ON (σ₀ = 0.5) — the bow's pattern, and load-bearing here.** "SUSTAINED brightness"
  is meaningless on a lossless string, where every mode sustains by definition. The signal exists
  only because σ₀ would darken a clean string while the bridge keeps re-injecting highs.
- **The IC is a mode-1 sine and there is NO pluck-position slider** (advisor's pre-build catch, the
  one thing neither probe would have caught). The headline is a *contrast*, so the clean baseline
  must be spectrally pure: a mode-1 start puts the clean centroid exactly on f₁ (measured 100.0 Hz
  against f₁ = 100) and the bridge's highs stand out. A raised-cosine pluck gives the clean string
  its own highs and shrinks the contrast — possibly under the gate. The #6 `mode11` / #9 single-mode
  lesson, third customer.
- **`downswing/depth` is the control, not `depth`.** What decides whether the string wraps is the
  near-termination downswing (`≈ amplitude·π·width_frac`) against the curve's drop; the dimensionless
  ratio is the refinement- and units-invariant coordinate (the tension `dT/T₀` / geometric `frac`
  pattern). Measured, it is a **ONE-SIDED FLOOR, not a window** — ratio 15.1/7.5/3.8/1.9/0.9/0.6 →
  elevation 3.48/3.75/3.44/3.15/2.33/1.63×, with the wrap contracting toward the crest (max node
  14 → 11 → 8) as it falls. More amplitude / less depth only ever helps. **Amplitude moves it exactly
  as hard as depth** (a depth-only guard would let an under-plucked string through none the wiser),
  which a test pins.
- **Below the floor → LABEL, never FAIL** (the bow's Schelleng-window rule). A too-deep bridge that
  the string only grazes is a legitimate stiff *point* contact — just not a jawari. The panel reports
  the elevation, the ratio, and what to change; it never paints a red verdict on real physics.
- **Two param collisions closed, and they were silent.** `K` is the sympathetic bridge *spring*
  (~8000 N/m; this bridge is 2e6 N/mᵅ, **250× stiffer**) and `alpha` is the mallet's felt exponent
  (2.3 vs 1.5 here). `gatherParams` sends every slider including hidden ones, so a user who had
  merely *visited* those models would have rendered a different bridge with nothing on screen to say
  so. Fix: a distinct `bridge_stiffness` param, and the contact exponent fixed server-side at the
  validated 1.5 rather than earning a slider (and a third chance to leak). Pinned by a test that
  ships `alpha=2.3, K=8000` and asserts the render is *unchanged*, while `bridge_stiffness` still
  bites.
- **THE INDEXING TRAP, and a test caught it, not the eye.** `BarrierString._b` and `contact_mask()`
  are both over the **SUPPORT** (the ~15 nodes the finite barrier covers), *not* over the grid. The
  first cut shipped both straight through to a frontend that indexes by grid node: the barrier
  arrived as a 15-long array drawn across the first 15 nodes, and the wrap marker landed **one node
  short** of the contact it marks. Near a termination that looks entirely plausible — it would have
  survived eyeballing the PNG. Both are now scattered onto the grid; the wrap *statistics* stay
  support-relative because that is the frame the model's own tests report them in.
  **Generalizable: when a core object exposes a masked/compacted array, the frame is part of the
  contract — assert the length against the grid, not against itself.**
- **THE OFF-GRID SNAP — the tests pinned a config the UI could not express, and only the CDP check
  saw it.** `MODEL_RANGES.jawari` set `audio_duration` to 0.24 s but re-ranged no `step`, so the
  slider kept index.html's `step="0.1"` (right for every other model's multi-second runs) and a
  range input **silently snaps an off-grid value**: the browser shipped **0.2 s**. Nothing looked
  wrong — the render succeeded, the verdict still read SHIMMER — but the elevation came out
  **2.75× where the tests assert 3.44×**: a hair over the 2.5× gate instead of comfortably clear
  of it, and the "viewer reruns the suite's exact rig" oracle was quietly false. It now renders
  3.44× in the browser, matching the suite.
  Fixed by re-ranging `step`/`fixed` with the val, and resetting both in `_default` (the leak rule
  applies to `step` exactly as to `min`). **Generalizable — and now enforced for EVERY model, not
  just this one: a `val` must lie on its own `step` grid from `min`, or the shipped number is not
  the tested number.** The switch-check grew a sweep asserting exactly that across all 12 models
  (all clean). This is the third distinct member of the `MODEL_RANGES` family of bugs, after the
  shallow-merge stale-step (batch 4) and the `_default` bound leak (batches 2 and 8) — all three
  share one shape: **a slider spec that is partially overridden keeps stale fields from the layer
  below, and the render is wrong without being broken.**
- **Only two runs are paid for.** The flat-rail control that separates "buzzes" from "travels" is
  already validated in `tests/test_jawari.py` (wrap std 4.89 curve vs 2.35 flat), so the panel
  *cites* it rather than paying for a third run — and the shipped sweep (nodes 0–14 of 15) is
  self-evidently travelling. The work budget counts **both** runs (`2·n_steps`), because a per-run
  cap would silently licence twice the wall clock.
- **The viewer reruns the suite's exact rig, so the suite's numbers are a free end-to-end oracle**
  (batch 4's win, second customer): same mode-1 8 mm IC, same σ₀ = 0.5, same curved parabola, same
  clean contrast at clearance = 1 m. The web test asserts elevation ≈ 3.44 and wrap std ≈ 4.89 — if
  the wrapper ever perturbs the rig (a different IC, a stale param, a shifted window) these drift.
- **Two views in the animation, because one cannot carry both halves** (the phantom batch's two
  spectra, one level over). The bridge spans ~15 % of the string and its curve drops ~1 mm against
  an 8 mm swing, so in the full view it is a few pixels near the termination — you see *that* the
  string is held off the rest line, not that it is lying along a curve. The zoom pane rescales y to
  the bridge depth and is the money picture. The spectrum panel puts both late-window traces on **one
  shared scale** (per-trace normalization would render two similar curves and delete the result) and
  on a **log axis** (the re-injected partials are 1e-2–1e-3 of the fundamental — linear would flatten
  them onto the axis exactly where the claim lives).
- Own budget: `JAWARI_N_MAX=128`, `JAWARI_AUDIO_MAX=1.5`, `JAWARI_WORK_MAX=150_000` steps across
  both runs. Cost is ~143 µs/step (jawari) + ~117 µs/step (clean) at fs = 50 kHz; the dense contact
  solve is over the **support** (≤51 nodes), not N, which is why N = 100 sits safely under the NumPy
  2.4 Windows BLAS cliff. Default render **4.9 s**; worst *passing* render **34.2 s** — *measured,
  not extrapolated* (N = 128, width_frac 0.4, 1.0 s), comfortably inside the verifier's 90 s window.

### Later batches (rough map — not firm)

- **Excited strings** — the jawari landed in batch 8 above; the bow in batch 2. What remains of the
  barrier family is **fret buzz / a flat rail or point fret** (model #8's *intermittent* regime, the
  physical opposite of the jawari's persistent travelling wrap) and the tanpura **cotton thread
  (juari)** = one more barrier node at a chosen position.
- **Wind** — bore + reed (new field type: pressure along an `S(x)` profile). The reed now reuses
  batch 2's balance panel; its telemetry differs (mouth / jet / reed-damping channels are each
  sign-definite and separately measured, so unlike the bow it can close the balance *with* loss on
  — the lossy branch may be a genuine residual there rather than an inferred one).
- **Sympathetics Weinreich two-stage decay** — DONE, batch 7 above.
- **The parametric-instability demo** deserves its own batch with real viz (energy cascading into the
  neighbour modes — model #9's IN-plane exchange, which is the SAME `2ω` pump batch 3's whirl aims at
  the other polarization) — *not* a bolt-on to justify batch 1's purity gate.

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
