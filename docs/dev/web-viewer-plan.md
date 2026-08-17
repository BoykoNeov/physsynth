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
- **A SCREENSHOT MUST SAY WHICH FRAME IT IS.** The verifier grabs whatever frame the animation
  happens to be on, and for the jawari it landed on an *upswing* — string clear of the bridge. That
  confirms the profile and layout but not the picture the zoom pane exists for. Two attempts to
  freeze a contact frame then failed **silently**: assigning `animPlaying`/`currentFrame` from
  `Runtime.evaluate` does not stick (it reads back `true`), so the rAF loop kept advancing and the
  capture was of an unrelated frame that still looked plausible. The fix is to drive the **scrub
  control** (`input` only — `change` clears `scrubbing` and hands the frame back to the clock) and,
  crucially, to **read `currentFrame`/`wrap`/`u_mid` back at the moment of capture and print them**,
  so a screenshot of the wrong frame can never be mistaken for evidence about the right one. With
  that, a contact frame confirms the string lying *along* the curve and the red departure dot on the
  node `wrapFrames` names. Independently, a field probe pinned the alignment: `wrap >= 0` on exactly
  the frames where `u` is negative near the bridge (4–8), `-1` on exactly the positive ones (0–2,
  9–11), and contact duty 44 %. **Generalizable: an animated viz needs its frame index in the
  evidence, or eyeballing proves nothing about the frame you meant.**
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

### Batch 9 (DONE) — the acoustic bore + radiating bell (the first WIND model, new field type)

The human chose the wind leg over body/bridge, the barrier family's leftovers, and the parametric
demo. Advisor then split it: **batch 9 = the linear bore, batch 10 = the reed**, on batch 1's own
de-risking precedent — the reed *contains* a `Bore` and both draw as pressure along the same tube,
so the new **pressure / S(x)** viz gets built and validated on the model with no root-find per step,
and the reed inherits a proven viz plus a whole batch for its balance panel. Every prior batch was
one model; this stays one. All-wrapper — `physsynth/core` untouched.

Pre-build probe numbers (`temp/bore-viewer-probe/`), all **measured before any wiring**:

- **THE STRUCTURAL FACT THIS BATCH TURNS ON: the bell's loss is BOOKED, so the verdict stays
  CONSERVATION — the first lossy model in the viewer that does not give up the drift panel.**
  `Bore.energy()` is `acoustic_energy() + radiated_energy`, and the bell's `k·R·U_out²` is
  accumulated, so σ = 0 with a *radiating* end still conserves: measured drift **3.3e-16 – 7.4e-15**
  across N = 64/128/200 while the bell sheds **9.8 %** of `E₀` (physical `R/Z₀ ≈ 3e-4`) and **100 %**
  at the matched `R = Z₀`. Energy leaves the tube and the book still closes. Contrast the bow, whose
  σ-loss is *inferred* and which therefore cannot score its lossy branch at all.
- **…and the corollary that constrains the sliders: bore viscous σ is NOT booked.** There is no
  viscous accumulator in `bore.py` — `energy()` conserves only at σ = 0. A σ slider would silently
  re-introduce an inferred channel and demote the conservation verdict back to a bare monotone
  check, buying a loss the bell already provides *physically*. So **σ is fixed at 0 and not exposed;
  `R_bell` is the loss control.** (Generalizable, and the reed batch inherits it: a measured-channel
  verdict is only as strong as the *least*-booked channel you let the user switch on.)
- **The bell reflection oracle is the money panel, and it is nearly free.** `r = (R−Z₀)/(R+Z₀)`: a
  centred Gaussian splits into two halves and one bounce sheds `½(1−r²)`. Measured against oracle
  across `R/Z₀ ∈ [0.03, 30]` → worst absolute error **1.4e-16**, with the exact **anechoic null at
  `R = Z₀`** (shed = 0.500000 — the entire right-going half absorbed, `r = 0`). It costs only ~N
  steps, so unlike the jawari's second run it is genuinely cheap to ship live rather than cite.
- **`R/Z₀` is the control, not `R`.** The dimensionless ratio is the units- and geometry-invariant
  coordinate (the tension `dT/T₀` / geometric `frac` / jawari `downswing/depth` pattern, fourth
  customer): `Z₀ = ρ₀c₀/S` moves with the bore radius, so a raw-`R` slider would mean something
  different at every geometry. Physical clarinet ≈ 3e-4; anechoic = 1.
- **The exactness claim belongs to the EIGENVALUE oracle, and the measured spectrum must be
  INTERPOLATED — a crude bin peak-pick invents a bogus N-dependence.** The first probe cut reported
  1.69 cents at N = 100/200 and 0.00 at N = 64/128, which is not physics but which bin the
  fundamental landed on. Through the suite's parabolic-refined `measure_partials_near` every
  measured partial is **≤ 0.007 cents**; the eigenvalue oracle is **0.0000 cents at λ = 1 for every
  N**, owes nothing to any FFT window, and is what structurally certifies the half-cell wall.
  (Batch 1's interpolated-zero-crossing lesson, second customer.)
- **The dispersion demo is an EIGENVALUE computation, not a render — and that is what keeps λ out
  of the step budget.** At λ = 0.9 the eigenvalue oracle departs the continuum by 0.6715 / 0.2744 /
  0.1674 / 0.0685 cents at N = 64 / 100 / 128 / 200 — a ratio of **4.01 across a 2× refinement,
  i.e. exactly O(h²)** — collapsing to 0.0000 at λ = 1. But a λ *slider on the render* is a trap
  (advisor): `fs = c₀N/(λL)`, so steps scale as **1/λ**, and at N_MAX = 256 / 1.5 s the cap is hit
  at **λ = 0.878** — λ = 0.9 already sits at 293k of the 300k budget, a 2 % margin, *before* the
  reflection run. Worse, the payoff isn't there: 0.07–0.67 cents is inaudible and invisible on a
  spectrum, and making it *watchable* needs λ ≈ 0.5–0.7, exactly where the cost explodes — and
  where CLAUDE.md's "tune toward λ = 1" says not to live. `bore_low_eigenfrequencies` is one
  `eigsh` call (milliseconds), so a **cents-vs-λ curve across a dozen λ values costs nothing, shows
  the O(h²) departure crisply, and needs no time-stepping at all.** The audio render stays pinned
  at λ = 1. **Generalizable: when a claim is about the *operator*, compute it from the operator —
  don't buy it with wall clock by rendering audio nobody can hear the difference in.**
- **The energy panel must show the SPLIT, or "conserved" reads as a bug (advisor).** The σ = 0
  drift branch fires and paints "conserved, drift 1e-15 ✓" — beside an interior pickup that
  audibly *decays*, because acoustic energy is leaving to `radiated_energy`. A flat green line next
  to a dying tone looks self-contradictory and hides the very physics the batch is about. So the
  payload emits **`acoustic_energy(t)` and `radiated_energy(t)` separately** and the panel plots
  both: acoustic falling, radiated rising, **sum flat**. That is "watch the sound leave the tube,"
  and it is what makes `R_bell` legible (9.8 % shed at `R/Z₀ = 3e-4` vs 100 % at the anechoic
  `R = Z₀`). `_energy_block` emits only the total today, so this is a genuine addition, not a reuse.
- **`R_bell` defaults ON (radiating).** A σ = 0 closed-open tube with no bell rings forever at
  constant amplitude — a sustained buzz that never decays. Loss-default-ON, the bow's and jawari's
  pattern, third customer; here it is also the only loss channel the batch permits at all.
- **Two ICs, and both runs counted.** The tone/signature run uses the suite's near-wall Gaussian
  bump (the odd-harmonic claim is *boundary*-determined — only odd resonances exist for closed-open
  — so it is robust to bump position, which is worth saying rather than assuming); the reflection
  oracle needs a **centred** Gaussian. That is a second run, and it counts against `WORK_MAX` —
  the jawari's "a per-run cap silently licences twice the wall clock" rule holds even though at
  ~N steps the reflection run is negligible.
- **The headline is the reflection oracle and the acoustic→radiated split.** Odd harmonics are
  table-stakes (the boundary condition guarantees them); dispersion is now a free operator-side
  panel. The two novel, *visual* claims are the 1.4e-16 reflection match with its anechoic null and
  the energy handing itself from the tube to the field — a linear tube is the least eventful model
  in the viewer, so it needs its claim chosen, not merely listed.
- **The odd/even ratio is set by the FFT WINDOW, not by N or by physics — so the gate belongs at the
  SHORTEST allowed duration.** Measured **2.29e5 at 0.5 s** but **3.6e4 at 0.25 s**, and flat in N
  to three digits at each. Gating on the 0.5 s number would fail a legitimately-correct short
  render. (The bow's "a rate needs a long window" lesson in spectral clothing.)
- **Cost is the cheapest of any recent batch, but N buys the sample rate, not just the grid.** At
  λ = 1, `fs = c₀N/L`, so N sets the step count *and* the grid: ~9–11 µs/step is flat in N
  (vectorized), yet CPU per second of audio runs **0.43 s (N=64) → 1.48 s (N=200) → 3.15 s (N=400)**.
  Budget on the product, not on N: `BORE_N_MAX = 256`, `BORE_AUDIO_MAX = 1.5`, `BORE_WORK_MAX =
  300_000` steps → worst passing render ≈ **2.7 s**, against the jawari's 34.2 s.

#### The viz design (SETTLED — task 2, measured 2026-07-20)

- **`drawBore` is its own path, and the forcing reason is correctness, not precedent.** `drawString`
  pins **both** endpoints to the rest line. The bore's closed end is a pressure **antinode** (p free
  and large); only the open end is a node (p = 0). That boundary asymmetry **is** the odd-harmonic
  claim, so `drawString` would not merely look wrong — it would render the batch's own physics
  backwards. The polyline/margin/scale arithmetic it shares is ~8 lines; duplicating that is cheaper
  and far clearer than parameterizing `drawString` with five flags. Dispatch extends the existing
  ternary chain at `app.js:934` with `isBore`.
- **Ends render by switching on `meta.ends`, never hardcoded.** This viz is the gate batch 10
  reuses, and the reed *is* a bore with a new end type at the mouth. `ends = ["closed",
  "radiating"]` → a `"reed"` case is an addition, not a rewrite. Closed = a hatched solid block
  (you cannot pass); open = the tube simply stops, with a dashed vertical marking the p = 0 node;
  radiating = a flared mouth plus an outward glow whose intensity tracks the **booked** radiated
  power — the field-side dual of the energy panel's radiated split, so energy leaving is *seen*
  leaving. At the matched `R = Z₀` the pulse reaches the mouth and simply vanishes: the anechoic
  null, visible.
- **THE TRAP — transit time ≪ render length; pace the animation on the TRANSIT, not on f₁.** One
  transit is `L/c₀ = 1.458 ms`, but `f₁ = c₀/4L = 171.5 Hz` (5.83 ms = 4 transits), and a 1.5 s
  audio render is ~1000 transits. Pacing on f₁ the way every string model does gives a **measured
  2.98–3.05 frames per transit at every N** — the bounce-and-flip picture aliases into noise, and
  `playback_speed` cannot rescue it because the frames are already decimated in *sim* time. The fix
  needs no new mechanism: the existing `anim_stride = round((fs / f_ref) / fpp)` seam already takes
  a model-appropriate reference (the bow substitutes `f_hard_est`, the tension string `f_osc`). Use
  **`f_ref = c₀/L`** — one transit — so "frames per period" reads as "frames per transit": measured
  **11.6–12.8 frames/transit, flat in N** (64→256). `BORE_ANIM_WIN` default **0.03 s** = 20.6
  transits ≈ 250 frames, under `MAX_FRAMES` at every N.
  **Generalizable: pace the animation on the timescale of the claim the picture makes, not on the
  fundamental. They coincide for a string and differ 4× for a bore.**
- **Cost hole found while measuring the pacing: `ANIM_WIN_MAX = 2.0` is a *shared* guard that the
  bore's budget does not cover.** At N = 256 a 2 s window is **351,232 animation steps** — over
  `BORE_WORK_MAX` on its own, and the `MAX_FRAMES` re-stride does **not** save it (it caps the
  frames emitted, not the steps simulated). Follow the jawari precedent: count **`n_anim + n_audio`**
  against `BORE_WORK_MAX`, rather than inventing a second constant.
  **Generalizable: a frame-count ceiling is not a cost ceiling — re-striding hides the sim cost it
  does not remove.**
- **Two runs, two emissions — do not architect around one shared radiated array.** Frames come from
  `anim_res`, the energy split from `_energy_block(audio_res)`; radiation is booked in both, but
  they are different sims at different strides. Emit per-frame cumulative radiated energy **in the
  frames block** (for the mouth glow) and keep the audio-run split in the energy block.
- **Envelope and polyline have different sources, each sized to its job.** The animated polyline is
  the *short* transit-paced window (the live bounce); the envelope is a running `max|p(x)|` over the
  *full-length audio* run, shipped as a **static** overlay — that is what makes the formed
  node/antinode structure legible, which no single instantaneous frame can show. Honest caveat to
  print: at the anechoic `R = Z₀` there **is** no standing wave, and the envelope correctly degrades
  to the trace of the pulse's single pass.
- **Scope split so task 3 does not balloon.** MVP = tube outline + pressure polyline + per-end
  rendering + envelope, and **keep the mouth glow** (it is the field-side half of the batch
  headline). The `divColor` tube-fill (reusing the membrane's diverging colormap) is the genuinely
  cuttable piece — build it only if the polyline alone fails to read as a tube.

The `MODEL_RANGES` discipline applies in full — val on its own `step` grid from `min`, reset
**every** field in `_default`, extend the switch-check sweep to the new model.

#### What task 3 (the wiring) actually found

Every pre-build number reproduced through the payload on the first cut — drift 1.2e-14 (default) /
3.3e-16 (anechoic), reflection error 1.4e-16, the exact anechoic 0.500000, O(h²) ratio 4.005,
0.0000 cents at λ = 1, 11.6–12.2 frames/transit flat in N, worst passing render 3.0 s at
N = 256 / 1.5 s. Nothing in the settled design needed re-opening. What the *wiring* added:

- **`_energy_block` grew a `split` parameter, and the verdict rides on the TOTAL.** The failure
  mode is specific and quiet: feeding `acoustic_energy` into the lossless branch fails the drift
  check (acoustic sheds ~10–100 %), which reads as a bug and invites "fixing" it by flipping to a
  passivity verdict — silently demoting the exact CONSERVATION claim the batch exists to make. The
  split follows the `balance_work` precedent (same-`idx` decimation, additive, no branch rewrite);
  the full 216-test suite re-ran clean afterwards, which is the check that matters for a shared-code
  touch.
- **The headline test needs BOTH halves, or it passes on broken wiring.** `drift < 1e-10` on the
  total is necessary but not sufficient — it also passes with the bell wired to shed *nothing*. So
  the test asserts the split actually moves: acoustic → < 5 % of E₀, radiated → > 95 %, radiated
  monotone, sum flat. **Generalizable: when a verdict is computed from a sum, a test on the sum
  alone cannot see a dead summand.**
- **The analytic curve must span the SLIDER's range, not the interesting part of the curve.** The
  first cut shipped `logspace(-2, 1.5)` — correct physics, and the default configuration's own
  measured point (a physical clarinet at `R/Z₀ = 3e-4`) landed *off the left edge of its own panel*.
  Only the rendered PNG showed it; every test passed. Now `logspace(-4, 1.5)`, with `1.0` inserted
  exactly so the peak is the true 0.500000 rather than a straddling 0.499973.
- **A conserving total plots exactly ON the frame's top edge.** `vmax = max(v)` puts the flat green
  line on the box border, where it reads as part of the chrome and not as the result. 12 % headroom,
  applied *only* when a split is present so every other model's panel stays pixel-identical.
- **The mouth glow tracks the RATE, not the cumulative total.** The payload ships cumulative
  radiated energy (monotone — the honest quantity for the energy panel), but a glow driven by it
  only ever brightens: a steady ramp that says nothing about *when* sound leaves. Differenced in the
  frontend, it pulses as each wavefront reaches the mouth. **Generalizable: the honest quantity for
  a ledger is rarely the legible one for a field-side animation — difference it at the draw call,
  not in the payload.**
- **Asymmetric margins, because the bell draws OUTWARD.** The flare and its glow live past the mouth,
  so a symmetric margin clipped the one element that shows energy leaving. `drawBore` sizes each
  margin from `meta.ends` — which is also the seam batch 10's reed needs.
- **Two `const bHint` in one scope.** `drawBore`'s hint collided with the bow's; caught by
  `node --check`, which is worth running on `app.js` before every browser check.
- **`_default` grew `L` and `animation_window`.** The bore is the first model to re-range either
  (L → 0.5 m; the animation window down to a 0.1 s max, because the shared 0.3 s is a cost hole its
  budget cannot cover). Without the resets a bore → string switch strands a 0.5 m string on a window
  slider that can no longer reach 0.06 s — the `MODEL_RANGES` leak family, fourth member.
- **A second regime was added beyond the plan: `open` (the ideal pressure-release end).** It costs a
  boundary tuple, gives the lossless contrast, and — the real reason — exercises the `meta.ends`
  switch *now*, so batch 10's reed inherits a dispatch that has been proven with more than one case.
- **The headless verifier grew a name filter** (`python scripts/verify_web_headless.py bore`), so a
  single-model batch can re-check its own two cases without paying for the geometric string's ~2
  minutes.

Honesty gates that fire and are LABELLED, never failed: at `R/Z₀ ≳ 0.05` the bell absorbs the pulse
before it can return, so no standing wave forms and the odd-harmonic / partial claims stop applying
(`spectrum.applies = false`) — the envelope correctly degrades to the trace of a single pass. That
is a correct render with nothing to measure, which is not the same as a wrong one.

### Batch 10 (DONE) — the dynamic single reed: the clarinet speaks

The second half of the wind split. Batch 9 built the pressure/`S(x)` field type on the model with no
root-find per step; batch 10 puts a **self-oscillating exciter** on the mouth end of that same tube.
`ReedBore` (`core/reed.py`, wind batch 3) *contains* a `Bore`, so the tube, the envelope, the
per-end rendering and the `meta.ends` dispatch all come across unchanged — the reed is a new **mouth
end type**, an addition to the switch batch 9 deliberately built with two cases rather than one.
All-wrapper again; `physsynth/core` stays untouched.

The reed is the acoustic dual of the bow (batch 2): both are continuous nonlinear exciters solved by
a scalar root-find per step, both start from **rest** and climb to a limit cycle, both therefore need
the **energy-BALANCE** panel rather than drift or passivity. But the reed's balance is a *different
and stronger* claim than the bow's, and that difference is the batch's structural point.

Pre-build probe numbers (`temp/reed-viewer-probe/`), all **measured before any wiring**:

- **THE STRUCTURAL FACT: unlike the bow, the reed's balance is a GENUINE residual, not a tautology.**
  Batch 2's lossy branch could not be scored, because the bow's dissipation is never measured — it is
  *inferred* as `work − ΔE`, so a "residual" there is identically zero by construction. The reed
  measures **every channel independently**: `mouth_work = Σk·p_m·U` (active breath in),
  `jet_loss = Σk·dp̄·U_B ≥ 0` (Bernoulli), `reed_damp_work = Σk·Mr·g·y′² ≥ 0` (lip damping), plus the
  bell's **booked** radiation. So `ΔE − (mouth − jet − damp)` is a number that can *fail*, and it
  doesn't: measured **7.6e-15 relative** on a lossless open bore and **2.9e-15** with a radiating
  bell, with every channel non-trivially populated (mouth 1.58e-2, jet 9.01e-3, damp 6.63e-4,
  radiated 5.41e-3 J over 0.25 s). This is the strongest energy claim any driven model in the viewer
  can make, and the panel should say so rather than reusing the bow's wording.
- **…and it inherits batch 9's "a sum cannot see a dead summand".** A residual of 1e-15 is also what a
  *dead* reed reports — that is precisely `reed-state`'s "balance is necessary but not sufficient; a
  wrong-sign reed rings down and passes on silence". So the headline test asserts the **channels are
  populated** (each `> 0`, mouth ≫ 0) alongside the residual, exactly as batch 9's split test had to
  assert the split *moves*. Second customer for that rule, one batch later.
- **NO VK-style convergence gate — verified, not assumed.** The worry was the tension/VK precedent:
  a non-converged solve masking as a good number. The reed *cannot* mask, because the per-step balance
  error is `k·p̄·R/p_pref0` — **linear in the Newton residual R** (the core's identity, not something
  the sweep below isolates) — so a bad solve shows up directly *as* a bad balance. Confirmed by
  loosening `newton_tol`: 1e-10 → **1.8e-15**, 1e-8 → 1.7e-14, 1e-6 → 1.3e-11, 1e-4 → 3.7e-10,
  1e-2 → **7.8e-9** — a monotone six-decade degradation. (`newton_tol` bounds R but does not equal it,
  so the sweep shows the *direction and magnitude* of the coupling, not its slope; the linearity claim
  rests on the identity.) So
  `newton_tol` is pinned tight (1e-10) and **not exposed**, and the balance is left to self-diagnose;
  no verdict override, no convergence block. *Generalizable: a gate is needed when a bad solve can
  look like a good result — check the direction of the coupling before copying one in.*
  (The bracket fallback fires 2.2 % of steps on the lossless bore, 0.26 % with the bell — the `sqrt`
  cusp at `dp = 0`, expected and already covered by the core's stability tests.)
- **THE HEADLINE IS THE SIGNATURE, not the balance** — the balance is table-stakes-plus, the physics
  claim is *that the clarinet speaks*. Three sub-claims, all measured:
  - **The blowing threshold brackets `γ ≈ 1/3`** (Dalmont/Kergomard's small-oscillation rule), swept
    on the settled-tail **AC** rms normalized by `p_closing` (see the next bullet) — **but only at a
    window ≥ 0.8 s, and that is a real trap, not a tolerance.** Near the onset the reed *critically
    slows down*: the 0.04 s settling time was measured at γ = 0.5, far above threshold, and a
    marginally sub-threshold reed's transient takes far longer to die. At a 0.4 s window γ = 0.355
    reads **0.01123** — an apparently intermediate, half-speaking point; at 1.6 s the same γ reads
    **0.00441**, i.e. sitting on the silent floor (γ = 0.338 → 0.00427). The window was measuring a
    transient that had not finished decaying. Converged at 0.8–1.6 s the sweep is
    γ = 0.304 → 0.0040, 0.338 → 0.0043, **0.355 → 0.0044** (silent), **0.372 → 0.062** (speaking),
    0.388 → 0.186, 0.405 → 0.259, 0.507 → 0.476. So the transition is *sharper* than the first
    measurement suggested — a **14× jump across a 5 % step in γ** — and the bracket **tightens to
    (0.355, 0.372]**, a touch above 1/3, exactly as `reed-state` records (this bore's losses raise it).
    *Generalizable: a settling time measured deep in the oscillating regime does not bound the
    settling time near the bifurcation — critical slowing down is the whole character of an onset,
    and a fixed window read across one will manufacture an intermediate point that is not there.*
    (The bow's and batch 9's "a rate needs a long window" lesson, now aimed at the onset itself.)
    `γ = p_mouth/p_closing` is
    the star control, and it is the dimensionless-coordinate rule's fifth customer (after `dT/T₀`,
    `frac`, `downswing/depth`, `R/Z₀`) — `p_mouth` alone means nothing without the reed that resists it.
  - **The pitch is set by the AIR COLUMN, not the reed — but state it as LEVERAGE, not as "it barely
    moves".** At iso-γ = 0.5, sweeping `f_reed` 2000 → 3000 Hz (**+50 %**) moves the played note
    162.43 → 167.39 Hz (**+3.1 %**, −94 → −42 cents against `c/4L` = 171.50). The suite's test asserts
    only `< 6 %`; the honest panel reports the *ratio* (a 50 % reed sweep buys 3 % of pitch) and the
    trend, which is itself physics: the reed's compliance acts as an **end correction**, so a stiffer
    reed lands *closer* to `c/4L` — monotone across all three points. The bell contributes only ~8
    cents over `R/Z₀` 0.005 → 0.097, so the flatness is the reed's, not the bell's. Measured with
    parabolic-refined `measure_partials_near` (batch 9's interpolate-the-spectrum rule, third
    customer) — a raw bin pick at this window is worth 34 cents and would invent the trend.
- **BOTH SWEEPS ARE PINNED AT A FIXED N, DECOUPLED FROM THE RENDER SLIDER — a budget fix that turns
  out to be a correctness fix (advisor's catch).** The naive design recomputes each sweep at the
  user's N, which (a) blows the budget — at N = 256 the threshold sweep alone is ~490k steps, larger
  than the entire proposed render cap, for a total ~2× over — and (b) quietly makes the *headline
  number a function of the display grid*. Measured, it need not be either: the sweep is
  **N-invariant to the fourth significant digit** (γ = 0.338 → 0.00431/0.00429/0.00428/0.00427/0.00426
  and γ = 0.372 → 0.06802/…/0.06792 across N = 64/96/128/200/256), and the pitch leverage is
  **3.05 % at N = 64, 128 and 200 to three digits**. Both are properties of *(bore geometry, f_reed,
  γ)*, not of the grid. So they are computed once at **`REED_SWEEP_N = 64`** and **memoized on the
  geometry key** (nothing the γ slider touches changes them), which bounds their cost as a constant
  and makes the threshold bracket stable while the user drags N. This is batch 9's dispersion rule —
  *when a claim is not about the render, don't buy it with the render's wall clock* — with the
  N-invariance measured rather than assumed.
  - **Odd harmonics dominate:** `peak(f₁)/peak(2f₁)` = **1.43e4**, `peak(3f₁)/peak(2f₁)` = **27**.
    Table-stakes for a closed-open tube (boundary-determined, batch 9's rule) but it is the clarinet's
    audible hallmark and belongs on the spectrum strip.
- **Sub-threshold is NOT silent, and the panel must normalize or it lies.** The raw mouthpiece rms
  below threshold is **9–15 Pa** — almost all of it the **DC** mouthpiece pressure, not oscillation
  (measured DC 8.2 Pa vs AC 13.0 Pa at γ = 0.338). A raw-rms gate would report a "note" where there
  is only a static pressure. So every amplitude figure is **mean-removed AC rms normalized by
  `p_closing`**, which collapses the whole sweep onto one dimensionless axis and makes the 16× jump
  legible. Below threshold → **LABEL ("did not speak"), never FAIL** — the bow's Schelleng rule and
  the jawari's grazing rule, third customer: a reed blown too gently is correct physics.
- **"Beats shut once per period" needs debouncing — measured 1.94 raw.** The opening trace at γ = 0.5
  is a textbook slam, but with a **rebound**: per 800-sample period the episodes are a 35-sample
  precursor, a 66-sample re-opening, then the main **291-sample** closure (**37 % duty**). Counting
  raw zero-crossings therefore reports ~2 closures per period for what is plainly one beat. The panel
  merges episodes separated by less than ~10 % of a period → **1.00 per period**, and reports the
  **duty** (which needs no debouncing) as the primary number. *Generalizable: a per-period event count
  over a signal that chatters at the threshold measures the chatter, not the event.* (The bow's
  "a rate needs a long window" trap, one level down — here the window is fine and the *event
  definition* is what quantizes.)
- **The mouthpiece pressure is a clean SQUARE WAVE** (measured: flat at −1490 Pa, snap to +1460 Pa,
  snap back), which is both the iconic clarinet signature and the natural audio pickup. The bell's
  far-field `pressure()` is what one would actually *hear*; which of the two is the audio and which
  is a trace is a task-2 (viz design) decision, not a human question.
- **Settling is FAST, which makes the bow's tail-capture cheap here.** The tone reaches 90 % of its
  final rms at **t ≈ 0.04 s** and is flat to 0.8 % thereafter. So the animation window is captured
  out of the audio run via batch 2's `snapshot_from` — **ONE run, never two.** Re-running a second
  resonator silently doubled the cost of a root-find-per-step model in batch 2 and the work budget
  could not see it; that lesson applies verbatim to a model that is also a root-find per step.
- **Animation pacing inherits batch 9's rule unchanged, and the probe confirms it transfers.** Transit
  `L/c₀` = 1.458 ms vs the tone period 5.831 ms = exactly 4 transits. Pacing on `f_ref = f₁` gives
  **2.98–3.05 frames/transit** (aliased); pacing on **`f_ref = c₀/L`** gives **11.6–12.8
  frames/transit, flat in N** — identical to batch 9 to two digits, because the reed changes the
  *excitation*, not the tube's timescale.
- **Bore viscous σ stays fixed at 0 and unexposed** — batch 9's least-booked-channel corollary,
  inherited exactly as it was written to be. `R_bell` is booked, so a *radiating* reed still closes
  the balance (measured above), and the bell stays the loss control and stays ON by default.
- **Cost is cheap and the reed barely charges for itself: ~11 µs/step, FLAT in N** (10.7 / 11.1 /
  11.3 / 11.8 at N = 64 / 128 / 200 / 256) against batch 9's bare bore at 9–11 µs — the root-find is
  ~15 % on top, not the bow's or jawari's multiple. But `fs = c₀N/L` still buys the sample rate, so
  budget the **product**: 1.0 s of audio costs 0.5 s (N=64) → 1.5 s (N=200) → 2.1 s (N=256) of wall
  clock. Two budgets, because there are two kinds of run and one cap over both would hide the larger:
  - **Render (tone + animation), N-dependent:** `REED_N_MAX = 256`, `REED_AUDIO_MAX = 1.0`,
    `REED_WORK_MAX = 300_000` steps counting **`n_anim + n_audio`** (the jawari/batch-9 rule) → worst
    passing render ≈ **2.5 s**. The audio cap is *shorter* than batch 9's 1.5 s because the reed needs
    no ring-down — it settles in 0.04 s and the tail is what we want anyway.
  - **Sweeps, fixed-N and memoized:** at `REED_SWEEP_N = 64`, 7 threshold points × 0.8 s + 3 pitch
    points × 0.6 s ≈ **324k steps ≈ 3.5 s**, paid once per geometry and free on every subsequent
    render. Budgeted as its own constant, *outside* `REED_WORK_MAX`, so neither cap can launder the
    other. Worst *cold* first render ≈ **6 s** total, against the jawari's 34.2 s — comfortably inside
    the verifier's 90 s window.

Build surface (to be enumerated against at review time): `_build_payload_reed` behind the
`_build_payload` dispatch; a `"reed"` mouth value in `meta.ends` + its `drawBore` branch; the batch-2
balance panel with a **reed sub-branch** in `drawEnergy` (the bow's "inferred dissipation" wording is
wrong here — this one is measured); a new **signature panel** (threshold sweep + pitch leverage +
opening/beating trace); `gamma` as the star control with `MODEL_RANGES` + **`_default` resets** (the
leak family's fifth member — `p_mouth`, `f_reed`, `q_reed`, and whatever else the reed re-ranges,
each on its own **step grid**, batch 8's off-grid-snap rule); a hand-rolled instrumented loop
(`simulate()` gives neither the balance channels nor the opening trace nor the frames — all three
*are* the panels), constructing a `SimResult` for `_energy_block`.

#### The viz design (SETTLED — task 2, measured 2026-07-20)

Probe `temp/reed-viewer-probe/probe_viz.py`, at the default clarinet (γ = 0.51, N = 128, radiating).

- **THE MONEY ANIMATION IS A TRAVELLING PRESSURE STEP — and it is the acoustic dual of batch 2's
  Helmholtz corner.** Sampled across the tube the limit-cycle field is not a wobbling standing wave
  but a **near-square kink that propagates mouth → bell and reflects**: three consecutive frames run
  `−1385 … −1320, +165` / `−1423 … −122, +121` / `−1380 … −49, +19`, i.e. the transition point
  marching down the tube. That is precisely what makes the mouthpiece signal a square wave, and it is
  the same picture the bow gave on a string one domain over — the corner *is* the sound. It also
  vindicates batch 9's pacing rule for a second model: the step crosses in one transit, so the
  transit-paced stride (12 frames/transit) resolves it and an f₁-paced stride (3) would alias the
  batch's best picture into noise.
- **The standing-wave envelope goes nearly FLAT, and that is a RESULT, not a failure — but it must be
  labelled or it reads as a broken overlay.** Batch 9's static `max|p(x)|` overlay drew a clean
  single-mode bump. Here it measures **1569 / 1543 / 1552 / … / 1550 / 1435 / 932 / 56 Pa** — flat to
  ~1 % across 85 % of the tube, then collapsing at the open end. That is exactly right: a square wave
  is *many* harmonics superposed, each with its own node pattern, so the max-over-time envelope fills
  in every node except the one the boundary pins. So the overlay is kept and captioned as the
  contrast with batch 9 (**one mode → a bump; a reed-driven square wave → a flat top**), and the
  number that carries the physics is the **mouth/bell envelope ratio = 27.9** — the closed end is a
  pressure *antinode* and the open end a node, which is the odd-harmonic claim rendered in one figure.
  *Generalizable: an overlay that was informative for a single mode can go featureless under a rich
  spectrum without anything being wrong — say which it is, rather than deleting the panel or
  "fixing" it.*
- **The reed needs its OWN pane and its own y-scale — to scale it is invisible.** `H0` = 0.400 mm
  against a 16 mm bore diameter = **2.5 %**, and the tail swings 0 → 442 µm. Drawn in the tube's
  units the entire headline gesture is sub-pixel. So the mouth end renders as a hinged flap in a
  dedicated pane with its own scale, exactly as the jawari's bridge needed a zoom pane against an
  8 mm string swing (that lesson's second customer). `meta.ends` gains **`"reed"`**, an addition to
  the switch batch 9 built with two cases precisely so this would not be a rewrite.
- **Audio is the MOUTHPIECE, and the caveat ships with it.** Measured side by side over the settled
  tail: mouthpiece **crest 1.116** (1.0 is a perfect square wave), `f₁/2f₁` = **460**, peak 1569 Pa;
  bell far-field `pressure()` crest **3.684**, `f₁/2f₁` = **25.5**, peak **4.77 Pa**. The mouthpiece
  is the square wave and carries the odd-harmonic signature ~18× more strongly, so it is the audio
  and the trace. But it is emphatically **not what a listener hears** — radiation differentiates, so
  the far field is spikier, ~300× quieter, and much less odd-dominated. The panel says so rather than
  letting "the iconic clarinet tone" quietly mean the pressure inside the mouthpiece. (Batch 6's
  "audio = string pickup, not body pressure" decision, arrived at from the opposite direction: there
  the radiating channel was misleadingly *silent*, here it is misleadingly *unlike the claim*.)
- **The beating trace is the reed pane's second job**, drawn under the flap: opening vs time with the
  closure episodes shaded, the **debounced** count (1.00 per period, merging episodes separated by
  < 10 % of a period) and the **duty (37 %)** printed. The duty is the primary number because it
  needs no event definition at all.
- Panel inventory: **tube + travelling step + flat-top envelope + reed pane** (animation);
  **balance** (batch 2's, with the reed sub-branch — measured channels, not inferred);
  **signature** (threshold sweep with the bracket marked, pitch leverage, odd-harmonic spectrum).
  The cuttable piece is the pitch-leverage sub-panel — it is the one claim that reads fine as two
  numbers in the readout if the panel budget runs out.


#### What the WIRING added (found by building — task 3)

- **The reed is NOT σ-gated, and routing it through the bow's branch would have reported a
  catastrophic IMBALANCE on a perfectly balanced model.** The bow's σ=0 branch scores
  `max|ΔE − work|`, valid *only* because a lossless string has `ΔE == work`. The reed's jet and
  lip-damping channels are on **even at bore-σ=0**, so `ΔE − mouth_work == −(jet+damp)` ≈ 60 % of
  scale. `_balance_verdict` gained an additive `measured_loss` mode that **replaces both** σ
  branches; `_energy_block` threads it through (the `split` / `decay_oracle` precedent). The full
  suite re-ran clean after, which is the check that matters for a shared-code touch.
  *Generalizable: before reusing a verdict, check the identity it assumes, not just the shape of
  the data it wants.*
- **The threshold turned out to be BELL-DEPENDENT, which the plan did not anticipate — and it
  vindicated the memo key.** Measured at N = 64: `R/Z₀` = 3e-4 and 2.5e-3 bracket **(0.30, 0.338]**;
  1e-2 gives (0.338, 0.355]; 2.4e-2 gives **(0.355, 0.372]**; and by 6.3e-2 the note **never speaks
  at all**. So the plan's bracket (measured at 2.4e-2) and the shipped default's differ *because
  they are different bells*, not because either is wrong — a lossier bell needs a harder blow. This
  is exactly why `R_bell` had to be in the sweep memo key, and it is now reported in the panel.
- **γ = 1/3 went ON the sweep grid exactly** (batch 9's anechoic-null lesson, second customer) and
  landed on a **knife edge**: 0.0196 against a 0.02 gate at the default bell. That is *correct* —
  1/3 IS the threshold — but it means the tests assert the **curve's shape** (three orders of
  magnitude across the onset, monotone in γ, bracket straddling 1/3) and **never that point's
  label**. A verdict that flips on a hair is a fine thing to *show* and a terrible thing to gate on.
- **TWO bugs that only the RENDERED page could catch, both invisible to 233 green backend tests:**
  - `drawReedSignature` looked up a `#partials-verdict` badge **that does not exist** — that panel
    has only a readout, and every sibling folds its verdict into the text. Setting `.textContent`
    on `null` throws, the render handler swallows it, and the page reports a generic
    **"network error"** — so the symptom does not even name the cause. Fixed by drawing the verdict
    into the canvas, as the siblings do.
  - On the **ideal open end** the far-field caveat computed `1.6e+33× quieter (crest null)`: nothing
    radiates there, so the comparison divides by ~0. Now **withdrawn and labelled** (batch 9's
    `applies = false` pattern) rather than faked, and pinned by a test.
  *Generalizable: a payload can be numerically perfect and still render a lying number or throw on a
  DOM id — the browser check is not a formality, and "network error" can mean "a null in your draw
  path".*
- **The reed pane had to become a real pane, not an end decoration.** First cut drew only the flap
  plus a caption; at H0 = 2.5 % of the bore diameter that reads as a stray tick. The pane now draws
  the **opening over the animated window with every closure episode shaded** and the duty printed —
  which is what makes "beats shut once a period" something you *see*. The reed's tube is also drawn
  **higher and slimmer** (`midY = 0.40 H`, `wall = 0.26 H` vs the bore's 0.5/0.32) so the pane clears
  the lower wall.
- **A sweep level can be EXACTLY zero** (the lossless open end below threshold is bit-silent), and
  `log10` of a clamped 1e-9 falls far below the axis floor, drawing the point *outside* the panel.
  The floor is now the axis bound itself, so a silent point sits **on** the bottom — which is what
  it means.
- **ΔE ramps with a bell and plateaus on an open end**, and the readout says which and why: radiated
  energy accumulates *inside* the book while the note sounds, whereas on an ideal open end the limit
  cycle puts breath-in and jet-loss-out on the same slope. The bow's "the tone grows without bound"
  line is true of neither. This contrast is visible in the two shipped renders and is the clearest
  single picture of what "booked radiation" means.
- **Cost as built:** ~5 s cold at N = 128 / 0.5 s (render ~1.4 s + the fixed-N sweeps ~3.5 s), ~1 s
  once the sweep memo is warm. Worst passing render stays inside the verifier window.
- **Process:** `node --check web/static/app.js` before every browser check earned its keep again —
  it caught a syntax error where shell-collapsed `\n` escapes landed inside a double-quoted
  `join()`. The CDP switch-check (`M:\claud_projects\temp\reed_switch_check.py`) passes 20/20
  including the batch-4 latch and batch-8 off-grid rules; the render capture
  (`M:\claud_projects\temp\reed_capture.py`) reads the frame index and every readout back **through
  the DOM**, because `payload`/`currentFrame` are module-scoped `let`s and `let`/`const` never
  become `window` properties.

### Batch 11 (DONE) — the fret / flat rail: model #8 on its own terms

Batch 8 shipped the **jawari**, which is a *configuration* of model #8 (a curved barrier hugging the
termination). The general case — the **flat rail / fret**, string–fret buzz, prepared-piano rattle —
has never been shown, and it is the jawari's physical **opposite**: not one departure point gliding
persistently along a curve, but **slap-and-release**, an intermittent contact that springs off. That
contrast is the batch's structural point; without it the batch collapses into "the jawari with a
flatter barrier". All-wrapper again; `physsynth/core` stays untouched.

The user's ask named the **vector-Newton contact**, so the second job is to make that machinery
*visible during the buzz* rather than as a static demo. Geometry follows from that: a **point** fret
collapses to `m = 1` and *is* the mallet's scalar solve (the model's own scalar-collapse test) — it
under-exercises the very thing being shown. A **flat rail** gives genuine distributed contact, stays
intermittent, and is the configuration `tests/test_collision_signature.py` already validates
(`barrier ≈ -2e-3`, `lam = 0.4`) — so the viewer runs the validated geometry, not a lookalike (the
jawari's own rule).

Pre-build probe numbers (`M:\claud_projects\temp\fret-viewer-probe\`), all **measured before any
wiring**, at `L=1, T=200, rho=0.005` (`f₁ = 100 Hz`), mode-1 pluck `A = 5 mm`, `K = 2e6`, `α = 1.5`:

- **THE HEADLINE IS INTERMITTENCY, and it is measured, not asserted.** At the default (flat rail,
  `clearance = 2 mm`, `σ₀ = 0.5`, N = 100) the string makes **1.24 contact episodes per period** at a
  **15.4 % duty** — it slaps and springs off, never pinned. Contrast the jawari, whose claim is a
  *travelling wrap edge* on a sustained contact. The money animation is therefore a **contact raster**
  (x-vs-t spark map of where and when the string touches), which next to the shipped jawari reads
  instantly as the opposite regime. Episode counting inherits the reed's **debounce** rule (merge
  episodes closer than 10 % of a period): raw onsets and debounced episodes agree here (31 vs 31 at
  `rail_frac = 1.0`), but the rule is cheap and the chatter regime is one slider away.
- **THE ACTIVE SET IS GENUINELY A VECTOR, and that is the number to put on screen.** Of `m = 99`
  support nodes, **up to 69 are simultaneously in contact** (mean 42 while touching). This is what
  makes the solve a vector Newton rather than the mallet's scalar — so `|𝒞|_active` gets its own
  readout and its own trace, alongside **Newton iters/step: max 2, mean 1.16**. That cheapness is
  itself the `λ_min(J) ≥ 1` proof showing up as a measurement (unique root, global convergence, no
  branch-picking) — the plan's central theoretical claim, visible as a number for the first time.
- **THE 2σ₀ DECAY ORACLE DOES *NOT* TRANSFER FROM THE JAWARI — and the reason is real physics, not a
  bug.** The jawari's energy panel keeps `decay_oracle` TRUE because its bridge is a *lossless elastic*
  barrier, so every mode still decays at exactly `2σ₀` (measured 1.009 there). The flat rail is
  *equally* lossless — the σ=0 drift is **7.2e-13** through genuine contact, which is proof no energy
  is dissipated — yet its decay reads **1.07–1.09 × 2σ₀**, i.e. 6–9 % fast. The mechanism, measured:
  the loss identity is `dE/dt = −2σ₀·(2·KE)`, so "rate == 2σ₀" is really an **equipartition**
  assumption (`⟨KE⟩ = E/2`), which holds for harmonic motion and **not** for a string being slapped by
  a stiff one-sided spring. Measured `⟨2KE/E⟩` against the measured rate ratio, with the **centered**
  velocity `(u^{n+1}−u^{n−1})/2k` the identity actually uses:

  | clearance | rate/(2σ₀) | ⟨2KE/E⟩ centered | err |
  |---|---|---|---|
  | out of reach | 1.0000 | 0.9999 | 0.01 % |
  | 4.0 mm | 1.0069 | 1.0086 | 0.17 % |
  | 2.0 mm | 1.0725 | 1.0724 | 0.01 % |
  | 1.0 mm | 1.0872 | 1.0875 | 0.03 % |

  So the identity is intact and the excess is the non-equipartition. **This also explains the jawari's
  1.009 retroactively: its gentle wrap stays near-equipartitioned, so the naive oracle passed there by
  luck rather than by transfer.** *Generalizable: an oracle that holds in one configuration of a model
  may be passing on an accident of that configuration — re-measure it per configuration instead of
  inheriting it.*
- **…but the identity is a CONSISTENCY CHECK, not a gate (advisor's catch, load-bearing).** Both sides
  come from one run of one identity, so they *must* agree if it holds — what that certifies is
  "barrier lossless + accounting intact", which the σ=0 drift (7.2e-13) and the σ>0 passivity
  (`max ΔE = −5.2e-18`) already certify at machine precision. And the agreement is **not uniform**:
  with the *backward* velocity the α=1 case disagrees by **3.0 %**, collapsing to 0.41 % under
  centering — the residual being the known α=1 discrete-gradient `(½,0,½)` weighting vs the string's
  θ-centering. Calibrating a tight tolerance on the pretty 0.01 % number would ship a test that fails
  at other α/λ/N for non-bug reasons. **So: gate on drift (σ=0) + passivity (σ>0); report the rate as
  a diagnostic TRIPLE** — `rate`, `2σ₀`, and `2σ₀·⟨2KE/E⟩` side by side, which *shows* the physics
  without pretending to be a precision oracle.
- **BRIGHTNESS IS NON-MONOTONE IN CLEARANCE — the panel must not claim "closer is brighter".**
  Centroid elevation over an out-of-reach control: **2.50× at 4 mm, 2.71× at 3 mm, 3.33× at 2 mm,
  2.59× at 1 mm, 2.83× at 0.5 mm** — it *peaks at an intermediate clearance* and falls as the string
  starts to pin against the rail. `test_closer_barrier_is_brighter` compares only 4 mm vs 1 mm and
  passes by a hair (2.50 vs 2.59), so a monotone panel label would be **disproved by its own slider**.
  The panel reports the measured elevation and names the peak; the monotone claim is scoped to the
  diagnostic pair the test uses. *Generalizable: a signature validated on a two-point comparison is
  not a monotone law — the slider samples the range the test never did.*
- **PITCH RISES STEEPLY BUT IS DIAGNOSTIC-ONLY — do NOT build a shortened-length oracle.** Measured
  `+69 / +225 / +598 / +690 / +1271` cents at clearance `4 / 3 / 2 / 1 / 0.5` mm. Large and clearly
  real, but `f = c/(2·L_eff)` would **overclaim**: `test_contact_is_intermittent` is explicit that the
  string is *not* pinned to a shorter length, and the plan only ever *mentions* pitch — it is not a
  validated claim anywhere in the suite. Reported as a measured signature, gating nothing (the
  Schelleng-window / grazing-ratio precedent, now fourth customer).
- **The static-equilibrium magnitude oracle is CITED, not led.** `S u* = (K/ρ)b` held to 3.4e-15 is
  the model's exact money test and has never been in the viewer — but it is α=1, non-musical, and its
  "result" is a flat line that does not move. It belongs as the verdict panel's **magnitude
  credential** (optionally a small static analytic-`u*`-vs-sim overlay), never as a co-headline
  alongside the buzz. Lead audible, cite the oracle.
- **Cost: this is the most expensive model in the viewer per second of audio, and the dense solve is
  NOT the reason.** Measured µs/step (full rail): **189 / 201 / 257 / 261 / 320** at N = 64 / 80 / 96 /
  100 / 128, i.e. ~2× the jawari's 143 µs. Halving the rail span halves `m` but buys only ~20 %
  (N=128: 320 → 201 µs), so the `|𝒞|×|𝒞|` solve is not dominant — the string step and the rank-`m`
  correction are. Since `fs = cN/(Lλ)` also buys the sample rate, budget the **product**: 1 s of audio
  costs **6 s (N=64) → 13 s (N=100) → 20 s (N=128)** of wall clock. Hence `FRET_N_MAX = 100`,
  `FRET_AUDIO_MAX = 0.6`, `FRET_WORK_MAX` counting `n_anim + n_audio` (the jawari/batch-9 rule), sized
  so the worst passing render stays inside the verifier's 90 s window. ONE run, not two — the
  out-of-reach brightness control is a *second* run and must be budgeted explicitly if kept (the
  jawari paid for exactly this and said so).
- **Slider names must not collide — the leak family, sixth member.** `K` is the sympathetic bridge
  spring, `alpha` is the mallet's felt exponent, `bridge_stiffness`/`depth`/`width_frac` are the
  jawari's. The rail needs its own names (`clearance`, `rail_frac`, `rail_stiffness`) **plus
  `_default` resets on every one**, or a visit to another model silently re-renders a different fret.
  Each on its own step grid (batch 8's off-grid-snap rule).

Build surface (to be enumerated against at review time): `_build_payload_fret` behind the
`_build_payload` dispatch; a **contact-raster** panel (the new frontend primitive — an x-vs-t image,
which no shipped panel draws); an `|𝒞|_active` + Newton-iters trace; the energy panel = the jawari's
σ-gated drift/passivity branch with the **decay oracle replaced by the diagnostic triple**; a
brightness/pitch signature panel with the non-monotonicity labelled; `clearance` as the star control.
Task 2 (settle the viz design, measured) comes before any wiring, as in batches 8–10.

#### The viz design (SETTLED — task 2, measured 2026-07-20)

Probe `temp/fret-viewer-probe/probe_viz.py`, at the default (flat rail, `clearance = 2 mm`,
`rail_frac = 1.0`, `σ₀ = 0.5`, N = 100, mode-1 pluck `A = 5 mm`), 0.4 s = 40 periods per run. The
physics was settled above; this measures only how it **renders and reads**.

- **THE GATED CLAIM IS INTERMITTENCY — say it here so task 3 knows what the verifier checks.** The
  raster + duty + episodes-per-period is the claim; the decay reading is a **diagnostic triple**, the
  brightness is scoped to the test's two-point pair, and the pitch gates nothing. (Batch 6's
  "claim = the bridge trace, NOT energy" rule, sixth customer.)
- **THE RASTER NEEDS ≳10 TIME-COLUMNS PER PERIOD, AND ITS INK IS *NOT* THE DUTY.** OR-reducing the
  full-rate contact mask into columns: at **400 columns** (10/period over 40 periods) the debounced
  episode count is exactly right (49 = truth) and stays right at 800/1600; at 200 and 100 columns the
  debounce window rounds to **less than one column**, so counting degenerates to raw-onset and
  *fragments* — 83 and 60 apparent episodes against a truth of 49. Coarse columns do not blur the
  slaps together, they **split them**, which is the more dangerous failure because a too-coarse raster
  looks *busier*, not emptier. Meanwhile the raster's **apparent (column-collapsed) duty overstates
  the true duty at every finite resolution** — **28.8 % / 21.4 % / 18.4 %** at 400/800/1600 columns
  against a true **15.5 %**, converging only at full rate — because a column is lit if *any* step in
  it touched, so the image is inherently a **dilation** in time. (The drawn image's own **ink fill**
  is a different and much smaller quantity — **13.1 % / 9.5 % / 7.8 %** at those resolutions, against
  **6.1 %** for the undilated full-rate mask — since most lit columns light only a few of the 99
  rows.) **So the duty and the
  episode count are computed at full rate and printed; they are never read off the picture.**
  *Generalizable: an OR-reduced raster is an honest map of WHERE and WHEN, and a biased estimator of
  HOW MUCH — compute the scalar from the signal, not from the pixels.*
- **BINNING x IS FREE FOR THE PICTURE AND FATAL FOR THE NUMBER.** Dropping the support from 99 to 33
  x-bins leaves the full-rate ink fill **identical (6.11 %)** — the raster reads the same — but the
  maximum simultaneous count collapses from **69 to 23**. Since `|𝒞|_active` is the batch's second headline
  (the thing that makes this a *vector* Newton), the trace is computed on the full support and only
  the *image* may be binned. Same lesson as the duty, one axis over.
- **BINARY CONTACT IS ENOUGH; FORCE MAGNITUDE IS AN OPTIONAL LINEAR GREY.** Contact force spans
  median **10.8 N** → p99 47.6 → max **77.2 N**, a dynamic range of only **7.1×**, so no log mapping
  is needed; and the faintest half of the contact pixels carry **14.2 %** of the total impulse, i.e. a
  binary mask does not badly over-weight grazing touches. Grayscale is a refinement, not a
  correctness fix — it ships if the primitive is cheap, and the panel is honest either way.
- **f₁ PACING ALIASES THE SLAP AWAY — TRANSIT PACING FIXES IT, AND CHEAPLY.** Sampling the
  contact-any signal at **3 frames/period** (f₁-paced) recovers only **19 of 49 episodes = 38.8 %**:
  the animation would show a string that mostly *isn't* touching, which is the exact opposite of the
  claim. Transit-paced (`transit = L/c = 5.00 ms`, half a period) at **8 frames/period already
  recovers 100 %**, as do 24 and 48. Notably the sampled *duty* is fine at every stride
  (15.7 / 15.2 / 15.7 / 15.3 % vs 15.5 %) — **duty is stride-robust, episode structure is not**, so
  the stride must be chosen for the events, not for the average. Batch 9's transit rule, **third
  customer**, and the cheap end of it suffices: 8 frames/period, not 24.
- **THE FRET IS THE FIRST CONTACT MODEL THAT NEEDS *NO* ZOOM PANE — the inversion is the batch's viz
  structural point.** The jawari (b8) and the reed (b10) both needed a dedicated zoom pane because
  their headline gesture was sub-pixel against the string/tube swing. Here the clearance is **2.0 mm
  against a 5.33 mm peak swing over the rail = 37.5 %**, i.e. **≈75 px** in a 200 px half-height pane.
  The string, the rail, and the gap all live at the same scale in one honest to-scale view, and the
  raster — not a zoom — is the money panel. *Generalizable: "contact models need a zoom pane" was a
  fact about two configurations' amplitude ratios, not about contact.*
- **BUT `rail_frac` CAN SILENTLY KILL THE CLAIM, AND IT IS A SLIDER.** Peak swing *over the rail*
  falls as the rail shortens toward the nut — 5.33 / 5.00 / 2.94 / 1.24 mm at `rail_frac` =
  1.0 / 0.5 / 0.2 / 0.08 — so at the default 2 mm clearance the rail goes **out of reach**: measured
  duty **15.3 % / 18.2 % / 8.8 % / 0.0 % / 0.0 %** at `rail_frac` = 1.0 / 0.5 / 0.2 / **0.12** /
  0.08, with zero episodes and `|𝒞|max = 0` below ≈0.15. A user dragging `rail_frac` alone reaches a
  blank raster and an empty claim with nothing wrong. So `rail_frac` gets a **floor at 0.2**, and the
  panel names the interaction (a short fret needs a proportionally smaller clearance). *Same family as
  the brightness non-monotonicity: a slider whose far end disproves — or here erases — the panel's own
  headline. Sweep every slider to its ends before shipping its range.*
- **AUDIO IS A NEAR-TERMINATION PICKUP AT 0.05 L, and the control is exactly clean.** Centroid
  elevation over the out-of-reach control, by pickup: **6.03× at 0.02 L, 4.68× at 0.05 L, 3.46× at
  0.10, 2.56× at 0.25, 2.78× at 0.50** — the buzz signature is strongest nearest the termination and
  decays monotonically inward, while the level rises the other way (0.46 / 1.06 / 1.65 / 3.54 /
  5.33 mm peak). **0.05 L is the knee**: 78 % of the best available elevation at 2.3× the level of the
  0.02 L node. The control's centroid reads **exactly 100.0 Hz = f₁ at every pickup** (a mode-1 pluck
  with no rail is a pure sinusoid), so the elevation ratio is *entirely* harmonic content the rail
  added — an unusually clean baseline, worth printing. Inherits the jawari's near-termination pickup
  and batch 6's "audio = string pickup" decision.
- **THE OUT-OF-REACH CONTROL VERY NEARLY DOUBLES THE RENDER — budget it or drop it.** Measured
  side by side, the control costs **504.7 µs/step against the default's 530.5**, i.e. **0.95×** — no
  contact solve to skip, because the string step and the rank-`m` correction (not the `|𝒞|×|𝒞|`
  solve) already dominate, exactly as the span-halving measurement showed. So "one run, not two" is
  now a *number*: keeping a live control means budgeting ≈**1.95×** the render. Decision: the
  brightness panel ships the elevation against a **short, separately budgeted** control run (a fraction
  of the audio length — the elevation is a spectral centroid and does not need the full render), and
  `FRET_WORK_MAX` counts `n_anim + n_audio + n_control`. *Absolute µs/step drifts run to run on this
  machine (331 vs 530 µs for the same config in one session), so budget on the committed figures and
  on measured ratios, never on a single fresh timing.*
- Panel inventory: **contact raster** (money, the new primitive) + **to-scale string-on-rail
  animation** at 8 frames/period, no zoom pane; **`|𝒞|_active` + Newton-iters trace** (full support);
  **energy** = the jawari's σ-gated drift/passivity branch with the **diagnostic triple**;
  **signature** (brightness elevation vs the control with the peak named, pitch as diagnostic).
  The cuttable piece is the Newton-iters half of the trace — `max 2 / mean 1.16` reads fine as two
  numbers in the readout if the panel budget runs out.

#### The backend, built (task 3a — `_build_payload_fret`, 19 web tests, all green)

The rig reproduces through the payload: duty **15.45 %**, **1.225** episodes/period, `|𝒞|max` **69**
of 99 support nodes, Newton **max 2 / mean 1.159**, contact force max **77.24 N**, σ=0 drift
**7.14e-13**, elevation **4.682×** at the 0.05 L pickup — every one matching the probe's committed
figure. Two things measured during the build changed the settled design, both flagged here because
the numbers, not a preference, decided them:

- **The equipartition triple is confirmed and its *absolute* ratio is duration-dependent — which is
  itself the argument for not gating it.** Agreement between the measured rate and
  `2σ₀·⟨2KE/E⟩` is **0.056 %** at the default, and holds under 1 % across clearances 4/2/1 mm. But
  the raw `rate/(2σ₀)` reads 1.0927 over a 0.25 s window and 1.0872 over 0.4 s — it is a *fitted
  average over a changing contact regime*, so no single committed value exists to gate against.
  Reported as a triple, exactly as the plan settled, and now for a second measured reason.
- **`pinned` is a GUARANTEE, not a live label, and the first cut of its bar was mis-calibrated.**
  Duty is bounded away from pinning across the *whole* slider range, for a structural reason: a
  lossless one-sided spring always pushes back, so the string can never come to **rest** on the rail.
  Softening the rail *raises* the duty monotonically (0.188 / 0.300 / 0.434 / 0.456 / 0.466 at
  `rail_stiffness` 2e6…2e1) but it asymptotes just under **0.5** — and 0.5 is the *free-sinusoid*
  limit, a string crossing the rail line untouched, i.e. the **no-rail** limit, not pinning.
  Stiffening drives it the other way (the rail repels rather than admits). A first cut put the bar
  at 0.5 and would have shipped a "pinned" label sitting exactly on the soft-rail limit — firing on
  the one regime where the rail does *least*. Bar moved to 0.9, where not-releasing actually lives;
  nothing reaches it, and that unreachability is the result: **the intermittency is structural, not
  a tuning accident.** *Generalizable: before gating on a bound, check which physical limit the
  bound sits on — a threshold placed on the wrong asymptote fires on the opposite regime.*
- **The control stays SHORT, as the plan settled — a deviation I made and then measured away.** A
  first cut doubled the control to full length believing the elevation needed it (4.42 vs 4.682).
  The control's centroid is in fact **window-invariant** — 99.996 / 99.997 / 100.002 / 100.026 Hz
  over 0.4 / 0.2 / 0.1 / 0.05 s, elevation 4.682 / 4.682 / 4.681 / 4.680 — because a rail-free mode-1
  pluck is a pure sinusoid. The gap came from the *comparison* truncating the **fret** pickup to the
  control's length (`min(len_a, len_b)`), and the fret's centroid is emphatically not window-invariant
  (468.2 over 0.4 s, 442.2 over 0.2 s). The bug was coupling the two windows, not the control's
  length. Reverting saved ~2.7 s per render and left 20 000 steps of headroom under `FRET_WORK_MAX`.
  *Generalizable: when a ratio moves after a window change, find out WHICH side moved before paying
  to lengthen the other.*

Also settled while building: the pitch figure ships as **`crossing_rate`**, not "pitch" — it
reproduces the probe exactly (+1600 cents at the 0.05 L pickup, +601 at the probe's own 0.1 L) but a
buzz adds zero crossings *within* a period, so the quantity mixes the fundamental with the rail's
contribution. Naming it pitch would launder that into a note. It gates nothing, as the plan required.
The work guard is **reachable** inside the shipped ranges (lowering λ raises `fs = cN/(Lλ)` and trips
it), so it is a live guard rather than dead code; the raster's x-binning branch is **not** reachable
through the API (`m ≤ 99 < 128`) and is inert future-proofing, claimed as nothing more.

#### The frontend, built (task 3b — `drawFretViz` + the raster primitive; 20 fret web tests, +1 this task, all green)

The eighth leak-family member and the first panel that is a **picture of a field's history** rather
than of a field. `drawFret*` renders left ≈42 % to-scale string-on-rail (one `sy` for the string
AND the rail — no zoom, the batch's structural point), right ≈58 % contact raster + `|𝒞|_active`
trace with a playhead at `frame_times[idx]`; the gated claim is **painted on-canvas** (the `#string`
canvas has no `<p>` readout, only the overlay), the energy panel gets the diagnostic triple, and the
signature panel names the brightness peak and refuses to call the crossing rate pitch. Every figure
matches the probe's committed oracle exactly through the live payload (duty 15.45 %, 1.225 ep/period,
|𝒞|max 69/99, Newton 2/1.159, force 77.24 N, elevation 4.682×, control centroid 100.0 Hz, +1600
cents). Three things the build surfaced, flagged because each is a rule with a fourth+ customer:

- **The `kind` key is not a missing PANEL, it is the WRONG one.** `drawDiagnostics` dispatches on
  `meta.spectrum.kind` with a chain of *equality* branches, so a block without `kind` falls through
  every one to `drawPartials`, which reads per-partial arrays the fret never ships. The contact block
  already carried `"kind": "fret"`; the signature block did not, and adding it was the one change
  that reopened the backend. *Generalizable: when a dispatcher is a chain of equality tests, a
  missing discriminant is a silent mis-route, not a no-op — pin the discriminant in a test.*
- **A hint that guards a render must be driven by EVERY param it reads.** `onControlChange` refreshes
  hints from an allowlist, and the fret's out-of-reach warning depends on `clearance` and
  `rail_frac` — neither of which was in it. The hint went stale: it read "the string reaches the
  rail" at swing/clearance 2.50 *while the rail had been dragged out of reach*, i.e. the one warning
  meant to fire **before** a ~15 s render was confidently saying the opposite. Caught only by a
  live-DOM slider sweep — a deep link renders once and cannot see it. (The jawari's `depth` hint had
  the same shape and got it right; the fret's second control is what exposed the gap.)
- **The transition verifier's own traps cost more than the code's.** The deep-link harness
  structurally cannot see a transition, so a separate CDP driver sweeps sliders and switches models
  live (`temp/fret-viewer-probe/switch_check.py`). It found the real hint bug — but only after three
  self-inflicted false results: a mis-escaped f-string brace that silently never switched the model
  (every param then read as a "leak" against the fret's own page), a whole-canvas ink count that read
  the rail slab + claim text as raster ink (12623 on an empty raster), and — the load-bearing one —
  `renderBtn.disabled` makes a Render click during the ~15 s page-load render a **silent no-op**, so
  the pre-existing "ok" satisfied the wait and every screenshot predated the render under test.
  *Generalizable: an automated UI check that keys on a status string must stamp a sentinel first and
  confirm the button was actually actionable — a stale "ok" is indistinguishable from a fresh one.*

### Batch 12 (DONE) — the body speaks: string → modal body/bridge + radiation read-out

The whole **coupling/radiation leg** of `exciter → resonator → body/radiation` has no viewer at all —
the biggest built-vs-shown gap left (Phases A–C surfaced only bare resonators; everything since batch
1 is one exciter/resonator variation after another, but the *third stage* — the body that colours the
tone and the air that carries it — has never been shown). This batch opens it with the canonical
case: a fixed/free string terminated on a **modal body** (`StringBodyBridge` + `ModalBody`,
`core/connection.py` + `core/body.py`) — the guitar-like bridge — read out to the far field by
`AirRadiation` (`core/radiation.py`). All-wrapper; `physsynth/core` stays untouched.

**Scope, deliberately narrow (advisor-endorsed, matches the core's own build order).** This batch is
the modal body + the radiation **read-out** only. Two things are explicitly *deferred* to follow-ons,
because each is a **different energy verdict** and bundling them blurs the story:

- **`StringPlateBridge`** (plate #5/#5b as a *distributed* body — the cymbal/soundboard you can watch
  ring on the heatmap) is the visually-richer sibling but carries the harder explicit-string ↔
  implicit-plate coupling. The human picked the modal body to lead (de-risking, the canonical
  terminus); the plate-as-body is a clean later batch that reuses the heatmap wholesale.
- **`RadiatedBody`** (the radiation *load* with back-reaction: `E_body + ∫P_rad = const`, the bore's
  booked-loss channel, batch 9's pattern) is a genuinely different panel. This batch's radiation is
  the **read-out** `AirRadiation` — a pure output transform, **no back-reaction, no `energy()`**.

**Be honest about what "radiated sound" buys here (advisor).** For a *single* source `AirRadiation`'s
output is just `body.pressure() × ρ₀/(4πr)`, integer-sample delayed — the **same spectrum**, scaled
and delayed. So the genuinely new content of this batch is the **coupling**, not a new signal: the
viewer's **first three-way energy decomposition** (`E_string + E_body + E_conn`), the **first time
`E_string` alone is not conserved** (it sloshes into the body — assert on the total, the docstring's
rule), and the body **colouring** the string's spectrum. Radiation adds far-field framing: the 1/r
law and a **distance knob** `r`. Frame it exactly that way; do not sell the radiated pressure as a
new sound.

**The headline (MEASURE it in task 2, don't assert):** the radiated spectrum = the string's partials,
**ω²-tilted** (a monopole radiates ∝ volume *acceleration*), **boosted near the body's modal
frequencies — but NOT clean formants** (the body modes `[110, 196, 261, 440]` Hz are *off-harmonic*
against the string's `100·n` Hz, so they lift neighbouring partials without forming a tidy resonance
peak — `diagnose_body_bridge.py` already flags "no pure formants", the honest read). The supporting,
hard pass/fail claim is **σ=0 conservation *through the coupling*** — the total drifts ~1e-13 while
`E_string` visibly does not conserve. Same σ-gating as weinreich (batch 7): σ_body=0 → the ordinary
**drift** panel on the total; σ_body>0 → **passivity** (monotone total, `decay_oracle` TBD in task 2
— there is no closed form for coupled modal decay over an off-harmonic body, so it is likely
`False`/passivity-only, but *measure* the single-exponential fit before deciding, the jawari lesson).

**The money viz = the energy EXCHANGE, reusing `drawFields` (sympathetics' 2nd customer, built in
batch 3).** A two-strip (or stacked-bar) time trace of `E_string` vs `E_body` (+ the small `E_conn`),
showing the pluck's energy sloshing from string into body and back — the classic coupled-oscillator
exchange, exactly the panel `transfer` used for two strings, here for string-vs-body. The string line
is the only moving *shape* (the modal body has no spatial extent to animate — that's the cost of
leading with the modal body over the plate). Second panel = the **radiated-pressure spectrum** with
the string-partial ladder + faint markers at the body modal freqs (shown, the ω²-tilt + body-boost
read off it). Audio = the far-field pressure (real, cheap — `fs ≈ 22 kHz` like sympathetics, not
geometric's 22×).

**Controls / defaults (SETTLE by measurement in task 2 — these are the starting guesses from
`helpers.make_bridge`):** body modal set `[110, 196, 261, 440]` Hz, modal mass ~0.02 kg (≈ the
string's `ρL = 0.005` kg, so the body genuinely loads the string), **K = 8000 N/m** the star control
(memory: the body carries up to ~62 % of the energy at a good K; the terminus fundamental glides
**free `c/4L` ≈ 50 Hz → clamped `c/2L` ≈ 100 Hz** as K stiffens — a second thing to show), N=100,
λ=0.9 (`< 1` hard-required — the string Nyquist is marginal at λ=1 and the spring pushes it unstable),
distance `r` (the 1/r knob), σ_body (the loss gate). **Each default hides something** (too-weak K =
invisible exchange; too-strong K = near-rigid *and* near the guard ceiling) — task 2 measures the K
that makes the exchange visible *and* stays safely below the exact guard.

**Pre-flagged traps (all our own recurring lessons):**
- **`MODEL_RANGES`/`_default` leak — K is yet ANOTHER meaning.** Bridge spring here (~8000) vs
  sympathetic bridge (8000, same — but different model key), jawari barrier (2e6), mallet felt
  exponent lives on `alpha`. Plus new params: modal freqs/σ_body/masses and radiation distance `r`.
  **Every param this model re-ranges must reset in `_default`** or a switch renders stale physics with
  nothing on screen to say so (the recurring leak, batches 2/4/7/8). Consider a distinct
  `bridge_stiffness` name as jawari did, to avoid the collision outright.
- **Guard: use the EXACT stability guard, never the cheap 2-DOF bound** (`connection.py` already does
  — `k²·λ_max(A) < 4`, assembled matrix-free; the isolated `K(β_s+β_b)<4` estimate is a *footgun*,
  the system can NaN at half that bound). Surface a guard violation as a **clean error payload**, not
  a 500/NaN. λ<1 hard-required.
- **The radiated spectrum == the body-pressure spectrum for one source** — do not build a second
  "radiated" analysis that pretends otherwise. The 1/r + delay only rescale/shift it. The distance
  knob changes *level and latency*, never the spectrum shape; say so in the readout.
- **σ-gated verdict + `decay_oracle` decided by MEASUREMENT** (not by "it has a body so copy the
  mallet"): probe the single-exponential fit of the lossy total first (the jawari reversal — a
  lossless *elastic* bridge dissipates nothing; here σ_body IS a real loss, so passivity, but confirm
  whether a clean 2σ oracle survives the off-harmonic modal spread before printing one).
- **Instrumented loop, not `simulate()`** (geometric/mallet/sympathetic pattern): `simulate()` gives
  energy+pickup but not the per-part `E_string`/`E_body`/`E_conn` split that IS the money panel.
  Capture all three in one hand-rolled step loop; construct a `SimResult` for `_energy_block`.

**Task breakdown (as batches 8–11):** task 1 = this plan (docs). task 2 = settle the viz design,
**measured** in `M:\claud_projects\temp\body-viewer-probe\` before any wiring (the K that makes the
exchange visible + stays under the guard; the actual energy split; *which* body modes boost the
spectrum; the ω²-tilt; the 1/r law; the σ_body decay-oracle decision). task 3a = `_build_payload_body`
backend + web tests. task 3b = frontend (energy-exchange + radiated-spectrum panels, distance/K
sliders) + verifier case + CDP switch-check + PNG.

#### The viz design (SETTLED — task 2, measured 2026-07-20)

All numbers from `M:\claud_projects\temp\body-viewer-probe\` at the `helpers.make_bridge` rig
(`L=1, T=200, ρ=0.005` → string `c=200`, `f₁=100` Hz; `N=100, λ=0.9, fs=22222` Hz; body modes
`[110, 196, 261, 440]` Hz, modal mass `0.02` kg). **Nothing was wired first.**

- **THE GUARD CEILING IS LOW — `K_c ≈ 21 480 N/m` — and that shapes the K slider.** At `K=0` the
  string's own Nyquist mode already puts `k²·λ_max(A) = 3.24` (of the 4.0 limit), so the coupling has
  little headroom: the exact guard trips at `K ≈ 21.5k` (`K=20k` OK at 3.91; `K=50k` raises at 5.99).
  ⟹ **K slider `[0, 19000]`, default `8000`** (leaves margin); a too-high K surfaces as a **clean
  error payload** (the exact guard, `connection.py`, already raises — the wrapper catches it). The
  cheap `cfl_2dof` diagnostic reads only 0.65 at the default and 1.75 at the ceiling — it is the
  **footgun** the memory warns of (it never reaches its own "< 4" bound before the real guard fires).
- **THE MONEY PANEL = the energy EXCHANGE, and at `K=8000` it is a BIG visible slosh (measured).**
  Lossless pluck: `E_body` swings **0.1 % → 56.9 %** of the total (mean 26.9 %), `E_string`
  **95.7 % → 42.6 %** in counter-phase, the body first peaking at **t ≈ 22 ms**. That is the classic
  coupled-oscillator exchange, plotted with `drawFields` (sympathetics' `transfer` pattern) over
  ~0.4 s. **`E_conn` must NOT be a stacked fraction bar** — being the cross-time spring term it swings
  `[−1.1 %, +17.8 %]` and **goes negative**; plot `E_string` and `E_body` as the two exchanging
  channels and the **flat total** as the conservation line (E_conn folded into the total, or a thin
  separate trace). Total drift `~3e-14` through the whole slosh = the σ=0 verdict, *while `E_string`
  alone visibly is not conserved* (that contrast is the batch's point).
- **The terminus fundamental GLIDES `58 → 92 Hz` as K stiffens** (`K=100`→`58.3`; `8000`→`90.8`;
  `19000`→`91.7`) — free `c/4L=50` toward clamped `c/2L=100`. It **asymptotes below 100** because the
  guard caps K *and* the body is finite-mass (not a rigid wall) — an **honest** secondary story, shown
  as a K-dependent readout, never claimed to hit 100. (Measured at the free end = the mode-1 antinode,
  band 30–130 Hz; a pickup near the nut or a wide band locks onto a higher fixed/free partial at
  50·(2n−1) Hz — the batch's own measurement gotcha.)
- **The radiated spectrum — TWO honest claims, and a first-cut "ω²-tilt" that the advisor rightly
  killed.** My first probe read `radiated/string-pickup` and called the rising ratio an ω² tilt; it
  is not (`tilt/f²` spanned 60×, non-monotonic even on the *strongest* partials). The reason:
  `p = Σ aᵢq̈ᵢ` gives ω² between the radiated pressure and the **body** motion, but the string pickup
  reaches the output through the **resonant coupling transfer** (peaked at the body modes), so
  `rad/pick = f² × body-transfer` — never a clean f². And there is **no `100·n` ladder to tilt**: a
  fixed/free string rings at `50·(2n−1)`, and at K=8000 (f₁≈91) the coupled spectrum is *hybridised*,
  fitting neither `50·(2n−1)` nor `100·n` (measured). What IS measured and honest:
  - **The ω² monopole law, shown where it is real:** radiated-pressure spectrum ÷ **body
    bridge-displacement** spectrum `= gain·ω²` to **ratio 1.00 at every peak** (90.8→298 Hz all read
    1.00). Report this as a clean diagnostic number ("monopole ω² law measured/theory = 1.00 ✓"),
    **not** as a per-string-partial tilt.
  - **Boosted near the body modes, NOT clean formants:** every radiated peak neighbouring a body mode
    (110/196/261/440) is lifted, and the mechanism is visible — the pickup partials come in
    **doublets straddling each body mode** (90.8 & 115.9 around 110; 185.3 & 206.5 around 196) = the
    coupled system's **avoided crossing / mode splitting**, the honest reason there is no tidy formant.
  - **Panel** = the radiated-pressure FFT + **faint markers at the body modal freqs** (the boost +
    doublet split *shown*, not scored). Peak-pick the FFT; do **not** overlay an imposed `100·n`
    ladder (the batch-8 off-grid lesson — markers between the real partials read backwards).
- **The 1/r law is exact** (`gain·r` constant to all digits) — the distance knob `r` changes **level
  and latency only, never the spectrum shape**; the readout says so. `r` slider e.g. `[0.5, 4] m`.
- **`decay_oracle = False` (passivity-only) for σ_body>0 — DECIDED BY MEASUREMENT, not by template.**
  The lossy total is **monotone** (passive) but **NOT a single exponential**: a log-linear fit leaves
  rms `0.038` at σ_body=5 and `0.14` at σ_body=20 (grows with loss) — the off-harmonic coupled decay
  is genuinely multi-rate, so a single fitted `2σ` vs a flat oracle would lie (the jawari check, run
  first, gave the opposite answer *there* because that barrier was lossless-elastic; here σ_body is a
  real loss ⟹ passivity). **σ-gated verdict like weinreich:** σ_body=0 → the ordinary **drift** panel
  on the total; σ_body>0 → **passivity** (monotone), no 2σ oracle line.
- **Instrumented loop, not `simulate()`** (geometric/mallet/sympathetic pattern): `simulate()` does
  not expose the per-part `E_string`/`E_body`/`E_conn` split that IS the money panel. One hand-rolled
  step loop captures all three + the string pickup + the radiated pressure; a `SimResult` is
  constructed for `_energy_block`. IC = the **raised-cosine pluck** (not a mode-1 sine) — the spectrum
  headline needs the whole partial ladder to show the tilt + boost.
- **Params to add + reset in `_default`** (the recurring leak): `bridge_stiffness` (distinct name to
  dodge the K collision with sympathetic/jawari), `sigma_body`, `distance`. Body modal set/mass fixed
  server-side (not sliders — a modal-freq editor is its own later feature). `amplitude`/`pluck_position`
  reuse the string path's.

#### The frontend, built (task 3b — `drawBodyEnergy` + `drawBodySpectrum`; the shared-slider fix; all green)

The coupling/radiation leg finally has a viewer, and the batch's whole point rides on **one panel
carrying two things at once**. The `#string` canvas reuses `drawString` unchanged (the string is the
only moving shape — the modal body is lumped). The **money panel is the Energy card**, not a bare
conservation line: `drawBodyEnergy` plots the `E_string ⇄ E_body` slosh (blue drains, orange fills,
counter-phase over ~0.4 s) with `E_conn` on its own **signed** axis and the flat green `total` as a
**100 % reference** — while the σ-gated verdict rides the badge + readout on the **absolute** total.
That is the batch: *the total conserves through the bridge (drift 8.5e-14 ✓) while `E_string` alone
sloshes 100 % → 22 % into the body.* `drawBodySpectrum` takes the second panel — the far-field
`|Q″(f)|` with faint dashed markers at the body modes, the terminus glide (free `c/4L` → clamped
`c/2L`), the ω² sanity number, and the 1/r "level + latency only" readout. Verified live: fresh-load
`?model=body` renders "ok" (drift 8.5e-14, terminus 90.8 Hz); the in-place `jawari → body → jawari`
switch re-ranges `bridge_stiffness` guard-safely (2e6 → 8k → 2e6) and body renders ok after it. Three
things the build surfaced, each a rule with a fourth+ customer:

- **A slider two models need with different ranges must be ONE element re-ranged, never duplicated.**
  The backend reads `bridge_stiffness` for BOTH the jawari contact bridge (~2e6 N/mᵅ) and the body's
  linear spring (~8k N/m), and `sigma_body` for both the sympathetic weinreich loss and the body. A
  second `data-param="bridge_stiffness"` element would split the `sliders` map (keyed by param) — the
  visible slider and the one `gatherParams`/`applyModelRanges` touch would diverge, so dragging it
  would do nothing. The fix is the leak-family discipline: single elements, re-ranged in
  `MODEL_RANGES.body`, **reset one level up in `_default`** (the jawari range restored on switch-away,
  or a body → jawari switch renders a 100× soft bridge; the reverse — jawari's 2e6 into the body's
  ~21.5k guard — is caught by the body override). *Generalizable: a shared param is a shared element.*
- **`data-hide-domain` gates a slider a domain-model shares with a domain-LESS one — no visibility-pass
  edit needed.** `sigma_body` must show in sympathetic's *weinreich regime only* AND for the
  domain-less body. `data-domain="weinreich"` alone force-hides it for every non-domain model (the
  pass overwrites unconditionally); the clean answer was `data-show="sympathetic body"` +
  `data-hide-domain="normal transfer"` — the same inverse-gate κ and `amplitude` already use, which
  hides only in the named regimes and otherwise defers to `data-show`. It moved out of the sympathetic
  fieldset (a hidden parent fieldset hides its children regardless), into a shared "Bridge stiffness /
  body & radiation" group. *Generalizable: reach for the existing inverse-gate before touching a
  shared visibility pass — the advisor flagged a `updateVisibility` rewrite as the one edit that could
  silently break every other slider, and `data-hide-domain` made it unnecessary.*
- **The fresh-load verifier structurally cannot see this model's #1 trap.** A deep-link
  `?model=body` initialises `bridge_stiffness = 8k` by construction and *never* ships a stale K — so a
  passing PNG proves nothing about the leak. The hazard is the IN-PLACE `jawari → body` switch
  (`gatherParams` ships the hidden 2e6). A separate CDP driver drives the real `modelSel` change
  handler and asserts the re-range in both directions plus a post-switch "ok" render. *Generalizable:
  a per-navigation harness tests initialisation, not transition; a leak lives in the transition.*

### Batch 13 (DONE) — the body you WATCH ring: string → distributed plate #5/#5b + radiation

Batch 12 opened the coupling/radiation leg with the **lumped modal body** (no spatial extent — the
string was the only moving shape). This batch swaps that body for a **distributed grid plate** —
`StringPlateBridge` + `Plate`, `core/connection.py` + `core/plate.py` — so the third stage finally
has a *picture*: the soundboard / cymbal **lighting up and ringing on the heatmap** as the pluck's
energy transfers into it. The advisor-endorsed frame: this is a **wiring + viz + measurement** batch,
not a physics one. The hard part — the explicit-string ↔ *implicit*-plate coupling and the **exact
Sherman–Morrison stability guard** (`k²/4·K·[(G0_str⁻¹)_end + (G0_plate⁻¹)_dp] < 1`, two small solves
at construction) — is already built and validated in core (`test_free_plate_connection.py`, 23 tests
+ the supported-bridge battery, `free-plate-bridge-state`). Drive it and measure; do not re-derive.
All-wrapper; `physsynth/core` stays untouched.

**Scope — BOTH boundaries, via the `boundary` domain selector (settled, not open).** `StringPlateBridge`
covers `boundary="supported"` (model #5, a rectangular **soundboard** — the guitar/piano body) and
`boundary="free"` (model #5b, the suspended **cymbal/gong**) in one class. The frontend's ranging is
already **domain-conditional** (`MODEL_RANGES` keyed `"model:domain"`, merged in `applyModelRanges`;
the standalone `plate`/`vk` models already use `boundary` as their domain), so this is **one model
key `platebody`** with a supported/free `boundary` toggle — mirroring the standalone plate exactly —
NOT two keys. That matters because **the guard ceiling differs per boundary** (`free-plate-bridge`
memory: free `K_c ≈ 13.97k`; supported TBD; batch-12 modal was `≈ 21.5k`), so `bridge_stiffness`
must re-range in **`MODEL_RANGES["platebody:supported"]` vs `["platebody:free"]`**, not just per model.

**What is genuinely new vs batch 12 (be honest about it).** Two things, no more:
1. **The distributed body is a second animated field.** Batch 12 animated only the 1D string; here
   the viewer shows the 1D string canvas AND the **2D plate heatmap ringing simultaneously** — the
   plate model's heatmap + mask renderer (`dims:2`, `nx/ny`, decimated `mask`) reused wholesale, fed
   by `bridge.plate.state.copy()` snapshots. This *is* the batch's visual point.
2. **The plate's spatial modes colour the spectrum** — rectangular (supported) or curved-Chladni
   (free) — boosting radiated partials near the plate modes, shown not scored (the batch-12 rule).

Everything else carries over unchanged: the three-way energy split `E_string + E_plate + E_conn`
(the money exchange panel `drawBodyEnergy`), the radiated-pressure spectrum (`drawBodySpectrum` — the
1/r "level + latency only" read-out, the ω² monopole sanity number), and the far-field audio
(`AirRadiation.radiate(bridge)`, reading `bridge.pressure()` off the plate). The **hard pass/fail
claim stays σ=0 conservation *through the coupling*** (core drifts ~1e-13/1e-14; the wrapper test
proves the payload preserves that signature number). **Do NOT sell the radiated pressure as a new
sound** — for one source the far field is the same spectrum, scaled 1/r and integer-delayed.

**The terminus glide is the OPPOSITE story per boundary (advisor — do not copy batch-12's text).**
- **Free plate OVERSHOOTS `c/2L`** — the diagnostic measured `60 → 118 Hz` as K stiffens, *above*
  the rigid-clamp `c/2L = 100`, because below its first elastic mode the floating plate loads the
  string end as a **reactive mass-spring** (its `{1,x,y}` rigid-body modes act mass-like), not a
  rigid anchor (`free-plate-bridge-state`).
- **Supported plate lands NEAR `c/2L ≈ 98`** — the pinned soundboard is a near-rigid termination
  (Step-4 physics). Each read-out must state its own correct story; a shared "toward clamped c/2L"
  line would be wrong for the free edge.

**Controls / defaults (STARTING GUESSES — SETTLE by measurement in task 2).** Reuse the string rig
(`L=1, T=200, ρ=0.005` → `c=200`, `f₁=100` Hz; `N_string=100`, `λ=0.9` **hard-required < 1**). Plate:
`N_plate` (tests use 16; the heatmap wants enough cells to read the Chladni pattern while the work
budget `n_live × steps` stays bounded — measure the N that reads AND renders in budget), `κ` the plate
stiffness, `ρ_plate` areal density, `bridge_stiffness` K the **star control** (re-ranged per boundary
under its measured guard ceiling, with margin; over-stiff → **clean construction-error payload**, the
exact guard already raises), `sigma_plate` the loss gate (σ=0 → conservation drift on the total; σ>0
→ passivity), `distance` r (the 1/r knob), `pluck_position`/`amplitude` reused. Body geometry
(`Lx=Ly`, `nu` for the free edge) fixed server-side or lightly exposed — a full plate-geometry editor
is its own later feature. **Each default hides something** (too-weak K = invisible slosh; too-strong
= near-rigid *and* near the guard ceiling) — task 2 measures the K that makes the exchange visible
*and* stays safely under the guard, **per boundary** (the free cymbal likely gives the biggest, most
legible slosh — memory: free carries up to ~89 % of the energy at a good K; supported TBD).

**Pre-flagged traps (our recurring lessons + this batch's new wrinkle):**
- **The `bridge_stiffness` leak, now with a per-BOUNDARY twist.** K here (~a few kN/m, linear spring)
  collides with the jawari contact bridge (2e6 N/mᵅ, its index.html home) AND batch-12's body
  (~8k N/m) — and now re-ranges *differently for supported vs free*. Reset in `_default` (jawari
  range restored on switch-away) AND override in **both** `platebody:supported` / `platebody:free`;
  a boundary switch that failed to re-range would render a stale-guard K with nothing on screen to
  say so (the leak family, batches 2/4/7/8/12).
- **Use the EXACT Sherman–Morrison guard, never a cheap per-part 2-DOF bound** — `connection.py`
  already does (two solves at construction). Surface a violation as a clean error payload, not a
  500/NaN. `λ < 1` hard-required.
- **One instrumented loop** (batch-12 pattern): scalar `E_string`/`E_plate`/`E_conn` + `qaccel` +
  far-field pressure + `u_end` + `w_dp` every step; `bridge.plate.state.copy()` frames only within a
  short **plate-animation window at a fundamental-resolving stride** — chosen *independently* of the
  0.4 s energy-slosh trace (the plate rings at its own modal freqs; the compelling shot is the plate
  lighting up as energy arrives ~20 ms in). No two-run split; `simulate()` cannot give the per-part
  split. Bound the work budget on plate `n_live × steps`.
- **σ_plate decay-oracle by MEASUREMENT** (not templated): probe the single-exponential fit of the
  lossy total first; likely passivity-only (off-modal coupled decay is multi-rate, batch-12's read),
  but confirm before printing a 2σ line.
- **The radiated spectrum == the plate-pressure spectrum for one source** — the distance knob changes
  level + latency only, never the shape; say so. Peak-pick the FFT, faint markers at the plate modes,
  do NOT overlay an imposed harmonic ladder (the batch-8 off-grid lesson).

**Task breakdown (as batches 8–12):** task 1 = this plan (docs). task 2 = settle the viz design,
**measured** in `M:\claud_projects\temp\platebody-viewer-probe\` before any wiring (the K that makes
the slosh visible + stays under guard, **per boundary**; the energy split; the `N_plate` that reads
and renders in budget; which plate modes boost the spectrum; the terminus glide sign per boundary;
the σ_plate decay-oracle decision; the 1/r + ω² sanity). task 3a = `_build_payload_platebody` backend
+ web tests. task 3b = frontend (the dual string+heatmap field view + the batch-12 energy/spectrum
panels reused, per-boundary K sliders) + verifier case + CDP boundary/model-switch check + PNG.

#### The viz design (SETTLED — task 2, measured 2026-07-21)

All numbers from `M:\claud_projects\temp\platebody-viewer-probe\` at the string rig (`L=1, T=200,
ρ=0.005` → `c=200`, `f₁_string=100` Hz; `N_string=100, λ=0.9, fs=22222` Hz) with a square plate
`κ=20, ρ_plate=0.005, N_plate=16` (17×17 grid), `ν=0.3` (free). **Nothing was wired first.** The
probe reproduces the validated core diagnostics (`diagnose_string_free_plate.py`).

- **THE GUARD CEILING IS THE SAME FOR BOTH BOUNDARIES — `K_c ≈ 13 968 N/m` at N_plate=16 — so
  `bridge_stiffness` needs NO per-boundary re-ranging (the advisor's differ-per-boundary worry was
  empirically false).** The exact Sherman–Morrison margin is *dominated by the STRING end-node term*
  `(G0_str⁻¹)_end`; the plate driving-point admittance `(G0_plate⁻¹)_dp` is near-identical for
  supported vs free at an interior node (`W_dp = h²` either way) — matching `free-plate-bridge-state`'s
  "margin identical to SS (0.2148)". Margin is **exactly linear in K** (K_c predicted = measured to
  all digits). **K_c SHRINKS as N_plate grows** (margin@K=3000: 0.159 at N=8 → 0.215 at N=16 → 0.292
  at N=24 ⟹ K_c ≈ 10.3k at N=24). ⟹ **`bridge_stiffness` slider `[0, 12000]`, default `3000`** (the
  core's own `K_PLATE_BRIDGE_DEFAULT`) — one range for both boundaries; a high-N × high-K corner
  trips the **exact guard → clean construction-error payload** (the designed behaviour, worth showing).
  `n_plate` is its own slider `[8, 24]`, default `16` (n_live 49→529 supported / 81→625 free; the
  17×17 grid reads the low Chladni pattern and renders far under the `7e8` node-step budget — worst
  case 27.8M).
- **THE MONEY PANEL = the energy EXCHANGE, and it is a BIG visible slosh on BOTH bodies (biggest on
  the free cymbal).** Lossless pluck, K=3000, 0.4 s: **supported** `E_plate` peaks **76.9 %** of the
  total (`E_string` min 22.7 %); **free** peaks **82.9 %** (`E_string` min 14.9 %), rising to ~90 %
  near the guard ceiling — the free plate is the more dramatic exchange (`free-plate-bridge` memory:
  up to ~89 %). Body first peaks at **~9.7 ms**. Total drift **~6e-14** through the whole slosh = the
  σ=0 verdict, *while `E_string` alone visibly is not conserved*. Reuse batch-12's `drawBodyEnergy`
  unchanged (`E_string`⇄`E_plate` counter-phase, `E_conn` on its own **signed** axis, flat total as
  the conservation reference). The one **new** panel content is the **plate heatmap** (below).
- **THE HEATMAP is the batch's new content — the plate you WATCH ring** (batch 12 had only the string
  moving). Reuse the plate model's `dims:2` heatmap + decimated `mask` (17×17), fed by
  `bridge.plate.state.copy()`. Its animation window/stride is chosen **independently** of the 0.4 s
  slosh trace: a short, fundamental-resolving window on the plate's own motion so you watch it light
  up as energy arrives (~10 ms in). String 1D canvas (`drawString`) + plate 2D heatmap shown
  together = the dual field view.
- **THE TERMINUS GLIDE IS THE OPPOSITE STORY PER BOUNDARY — and it must be measured from a NEAR-NUT
  pickup (0.23 L), NOT the free end.** As K stiffens the string end gets *pinned*, so `u_end`'s own
  motion collapses onto the plate's slow driving-point bounce (~55–67 Hz) and **misleads** — batch-12's
  `u_end` readout worked only because a lumped modal body never pins the end that hard. Measured at
  0.23 L over 1.5 s (matching the diagnostic):
  - **Supported: `55 → 98 Hz`** — rises from the free quarter-wave `c/4L=50` and lands **near** the
    rigid-clamp `c/2L=100`; the pinned soundboard is a near-rigid termination (Step-4 physics).
  - **Free: `60 → 118 Hz`** — rises and **OVERSHOOTS** `c/2L=100`, landing ~118, because below its
    first elastic mode the floating plate loads the end as a **reactive mass-spring** (its `{1,x,y}`
    rigid-body modes act mass-like), not a rigid anchor (`free-plate-bridge-state`, matched exactly).
  Each boundary's read-out states its own correct story; a shared "toward clamped c/2L" line (the
  batch-12 text) would be wrong for the free edge.
- **THE RADIATED SPECTRUM — boosted near the plate modes, NOT clean formants; markers shown not
  scored.** Faint dashed markers at the low plate modes (supported `[62, 155, 248, 306]` Hz; free
  `[43, 62, 76, 109]` Hz), where the string partials lift. Peak-pick the FFT; do **NOT** overlay an
  imposed harmonic ladder (the batch-8 off-grid lesson). The **ω² monopole sanity = `|Q″(f)| /
  |Q_vol(f)| / (2πf)² = 0.997`** (median over the strong peaks, spread `[0.98, 1.05]`) — the
  denominator is the plate **volume displacement** `Q_vol` (`h²·Σuᵢ` supported / `Σ Wᵢᵢ uᵢ` free),
  **NOT** the driving-point `w_dp` (that gives 0.4/0.14, not 1.0 — the real correction over batch-12's
  `w_b` denominator, since `Q″ = Q_vol″` exactly for a distributed body). A near-tautological
  bookkeeping check on the pressure read / weights / byte order, NOT a radiation oracle.
- **The 1/r law is exact** (`gain·r` constant at 1 m and 2 m to all digits) — the distance knob
  changes **level and latency only, never the spectrum shape**; the readout says so. `distance`
  slider `[0.5, 4] m`, default 1 (reuse batch-12's).
- **`sigma_plate` decay_oracle = False (passivity-only) — DECIDED BY MEASUREMENT.** The lossy total is
  **monotone** (passive) but **NOT a single exponential**: the log-linear fit rms grows with loss
  (supported 0.0035 → 0.143; free 0.0066 → 0.189 at σ=2→20) — the off-modal coupled decay is
  genuinely multi-rate (batch-12's read). **σ-gated verdict like weinreich/batch 12:** σ=0 → the
  ordinary **drift** panel on the total; σ>0 → **passivity** (monotone), no 2σ oracle line.
  `sigma_plate` slider `[0, 80]`, default 0 (σ=0 conservation headline first).
- **Instrumented loop, not `simulate()`** (batch-12 pattern): one hand-rolled step loop captures the
  per-part `E_string`/`E_plate`/`E_conn` split + `Q″` + far-field pressure + the near-nut pickup +
  `Q_vol` + `plate.state.copy()` frames (within the short heatmap window at stride). IC = the
  raised-cosine / triangular pluck (the whole partial ladder, for the spectrum boost).
- **Params to add + reset in `_default`** (the recurring leak): `bridge_stiffness` (re-ranged
  `[0, 12000]` from the jawari's home 2e6), `n_plate` (new — `[8, 24]`), `sigma_plate` (new),
  `distance` (reuse body's). Plate geometry (`Lx=Ly`, `ν`) fixed server-side (a plate-geometry editor
  is its own later feature). `amplitude`/`pluck_position` reuse the string path's.

#### The backend + frontend, built (tasks 3a/3b — the dual field view; all green)

The coupling/radiation leg now has its *distributed*-body viewer. `_build_payload_platebody` drives
the validated `StringPlateBridge` (both boundaries) through one instrumented loop, and the frontend
(`drawPlateBody`) shows the **dual field view**: the string rides a thin strip along the top while
the plate — soundboard (#5) or free cymbal (#5b) — **rings on the heatmap** below, reusing the plate
model's `dims:2` renderer wholesale. The Energy card reuses batch-12's `drawBodyEnergy` verbatim (the
plate rides the body's exchange keys; the legend just relabels `E_body → E_plate`), and
`drawBodySpectrum` gains a per-boundary terminus readout. Verified live: both fresh-load PNGs
eyeballed (the cymbal rings a curved-Chladni blob, the soundboard a rectangular one; drift 2.2e-13 /
1.8e-13; plate fills to 83 % / 77 %); the CDP switch-check **12/12** (`platebody_switch_check.py` —
jawari→platebody re-ranges `bridge_stiffness` to the guard-safe `[0,12000]/3000`, the free↔supported
boundary switch both render ok, `sigma_body` stays hidden, and platebody→jawari restores the 2e6
leak-reset). Two things the build surfaced, each a rule:

- **The terminus f1 CANNOT be read off the user's run — it needs a dedicated near-nut-pluck probe.**
  The measured design read the terminus from the near-nut pickup of the main run, and the fresh-load
  PNG at the **default 2 s** exposed the bug the 0.5 s test could not: free read **99 Hz** (NOT an
  overshoot), flatly contradicting the "OVERSHOOTS" text. The cause is physical — the user's 0.3 L
  pluck excites the avoided-crossing doublet so that over long integration the argmax **flips**
  between its ~99 and ~117 Hz partners (measured: free 0.6–1.5 s → 117, but 2.0 s → 99). A **near-nut
  pluck (0.137 L)** — the core diagnostic's recipe — reads f1 **robustly at every duration** (free
  117, supported 97). So the terminus is now a **separate short 0.6 s near-nut-pluck sub-run**
  (`_platebody_terminus_f1`), decoupled from `pluck_position`/`audio_duration`; a `duration-robust`
  test pins that free still overshoots at 2 s. *Generalizable: a headline number that varies with a
  user control or the run length is not measured — probe it under fixed, controlled conditions. The
  PNG at the DEFAULT params, not just the fast test, is what caught it.*
- **The ω² sanity denominator for a DISTRIBUTED body is the volume displacement, not the driving
  point.** Batch 12's `w_b` (a scalar modal displacement) does not transfer: for the plate `Q″` is
  the *area-weighted* volume acceleration, so `|Q″|/|w_dp|` gives ~0.4, not ~1. The right, still
  near-tautological denominator is `Q_vol = h²Σuᵢ` (supported) / `Σ Wᵢᵢ uᵢ` (free), for which
  `Q″ = Q_vol″` exactly → 0.997. *Generalizable: when a batch-12 read-out moves from a lumped to a
  distributed body, re-derive what each spectral ratio actually relates — the lumped identity may
  quietly break.*

### Batch 14 (DONE) — the tanpura cotton thread (juari): a POSITION-SELECTIVE point contact

**Built & browser-verified.** All-wrapper (`physsynth/core` untouched); 12 web tests + 1 core
scalar-collapse test at the juari's numbers. Shipped defaults, measured through the payload: thread
at x = 0.10 L (node 10) buzzes **2.84×** the clean string; the **sweet spot is node 11 (x = 0.11 L)
at 2.99×**, which *rivals the whole curved jawari bridge* (**3.45×**, flat) — a well-placed point
thread is as bright as the distributed wrap, but position-tunable; clean baseline sits on **f₁ =
100 Hz**. The headless verifier renders "ok" with the tuning curve, the flat clean/jawari reference
lines, the band-spectrum mechanism pane, and the energy card's `decay_oracle` verdict (measured 2σ =
1.002 vs oracle 1.000). Cost: the **sweep is the driver** (~11 thread positions at a canonical
0.24 s), ~37 s at the default — so a hidden `sweep_duration` override keeps the tests fast while the
UI holds the canonical window. One measured decision beyond the plan below: **the buzz is a SETTLED
quantity** — a 0.10 s run peaks at the wrong node (the spectrum is still cascading upward), so the
sweep runs at a fixed 0.24 s DECOUPLED from the audio slider (pinned by a test), keeping the map
stable when the user changes the sound length.


The last unshown member of the barrier family, and the jawari's (batch 8) complement rather than a
preset of it. The tanpura *juari* (the *jvari*-tuning thread) is a single thread of cotton laid on
the bridge under the string; sliding it a millimetre is how a player tunes the instrument's buzz.
Physically it is **one more barrier node** (the jawari-state follow-on): a `BarrierString` (model #8)
whose barrier is `-inf` everywhere except **one interior grid node** — the thread — grazing the rest
line. **Zero core code; `physsynth/core` untouched**, like every batch since Phase B.

**The whole batch was gated on a separability PROBE before any wiring** (advisor: "the probe is the
whole decision"). The redundancy trap is real — a thread node *inside* the span the curved jawari
bridge already wraps is a no-op, so the honest framing is the thread **alone**, not thread-on-curve.
The probe (three passes, `M:\claud_projects\temp\juari_probe\`) confirmed a distinct, honest model.
Every number below is measured, before the wiring:

- **The claim is POSITION-SELECTIVITY — the tuning curve — NOT "more buzz."** A point contact couples
  to the mode shapes: it clips partials with an **antinode** at the thread and barely perturbs those
  with a **node** there. So sliding the thread reshapes *which* partials buzz — the tanpura tuning
  gesture, and the clean separator from the jawari, whose distributed curved wrap gives a
  **position-INDEPENDENT** broadband shimmer. Measured, with the jawari's own string params (loss ON
  σ₀=0.5, mode-1 IC, N=100, λ=0.4, K=2e6, α=1.5, amp=8 mm): the clean string sits on its fundamental
  (late-window centroid **100 Hz = f₁**), the jawari is a flat **3.39× clean** at every position, and
  the juari **sweeps 1.0×→3.44×→~2.0× as the thread moves nut→sweet-spot→mid-string.** The curve IS
  the headline; the jawari's single number cannot express it.
- **The sweet spot is x≈0.09 (node 8–9), ~3.44×** — which slightly *exceeds* the jawari's 3.39×: a
  well-placed thread is as bright as the whole curved bridge, but tunable. The near-nut region is a
  smooth **rise** from the termination (node 1 = 1.02×, negligible — displacement vanishes at the
  fixed end) to that peak, then a gentle decline (mid-string ~1.6–2.0×, where the even partials have
  a node at x=0.5 and go unclipped). **20 of 25 near-nut nodes exceed 2.0×.**
- **The tuning curve survives the window/duration wobble** (jawari-state's late/early caveat; batch
  13's 117→99 flip). Measured at three settings — late-50%/0.35 s, late-30%/0.35 s, last-0.20 s — the
  **peak node agrees (9, 8, 9)** and the shape holds; only the 2nd digit moves (node 9:
  3.44/3.41/3.38×). The *feature* is robust, so the numbers are safe to quote.
- **Grid quantization is load-bearing and stated honestly.** `thread_position` snaps to the nearest
  grid node (model #8's documented sub-grid limitation); the tuning curve's resolution **is** the
  node spacing, `h = L/N = 0.01 L`. The physically-real regime — near the nut, where a real tanpura
  thread lives — is precisely where that resolution gives the **fewest distinct positions** (~10
  across the rise nut→peak). The panel plots **discrete points at node resolution, not a smooth
  line** — the quantization is honest content, not a rendering flaw — and the count is foregrounded.
- **The mechanism panel is the band spectrum at the selected position, calibrated to the data.**
  Probe 2: the thread **suppresses the fundamental** where it has an antinode there (node 50: f₁
  −18.8 dB) and pumps that energy into the **low-order** partials with an antinode at the thread
  (node 50: odd n=3/5 at +95/+103 dB vs even n=2/6 at +61/+57 dB). The selectivity is **clean at low
  order and washes out at high n** as the nonlinear clipping spreads energy broadly (node 50: n=9
  antinode +68 actually *below* n=10 node +79). Write it that way — fundamental-suppression + low-
  order antinode selection, masked at high n — never an idealized odd/even alternation.
- **The tuning curve is the MAP; `thread_position` is where you stand on it.** The sweep panel is a
  precomputed reference (clean 1.0× and jawari 3.39× drawn as flat guide lines; the sweet spot
  marked); the `thread_position` slider picks the operating point that drives the audio, the band
  spectrum, and the string-over-thread animation, with a marker on the curve showing the current
  position. That is the tanpura-tuning UX in one screen.
- **Below-signal positions LABEL, never FAIL** (the bow Schelleng / jawari floor rule). A thread at
  the nut, or on a partial's node, is legitimate physics that simply does not buzz much; the panel
  reports the elevation and where to move the thread, never a red verdict on real physics.
- **Jawari templates carried UNCHANGED — they are table stakes, not the batch's content.**
  `decay_oracle` stays **TRUE**: the thread is a *lossless elastic* contact (dissipates none), so
  every mode still decays at exactly 2σ₀ and the flat-loss oracle survives the point wrap. Loss
  defaults ON (σ₀=0.5 — "which partials buzz" is meaningless on a lossless string). Mode-1 IC, **no
  `pluck_position` slider** (the clean baseline must be spectrally pure for the contrast — the #6
  `mode11` / #9 single-mode lesson). σ₁ and hysteresis fixed at 0.
- **Correctness confirmed on the ACTUAL juari config, not by analogy** (advisor: the probe measured
  *signal*, never *correctness*): σ=0 lossless **energy drift 1.08e-12** through 587 real contact
  steps (613 N peak force); σ>0 **passivity** (max inter-step ΔE = −2.78e-17, monotone); and the
  config-specific money oracle — **m=1 scalar-collapse**, the single contacting node's vector solve
  == the imported scalar `solve_contact` to **1.93e-15** (barrier-collision-state's oracle, re-run at
  K=2e6/α=1.5 rather than cited from #8's K=8e5). The suite already carries
  `test_single_node_collapses_to_scalar_solver` on this exact single-node shape; the new tests pin it
  at the juari's numbers.
- **Param hygiene: `thread_position` is a NEW name** (fraction of L; no other model sends it, so no
  leak — the jawari's `bridge_stiffness`/`K`/`alpha` collision lesson). Reuses `bridge_stiffness`
  (same 2e6 bridge), `amplitude`, `lambda`, `sigma0`, `N`, `pickup_position` from the jawari ranges;
  α fixed server-side at 1.5.
- **THE SUPPORT→GRID SCATTER TRAP, harder with one node.** `BarrierString._b`/`contact_mask()` are
  over the **support** (here length 1), not the grid — the jawari's off-by-one is *easier* to ship
  and eyeball past with a single node than with fifteen. The thread marker and any contact readout
  are scattered onto the grid (`grid[support] = value`) exactly as the jawari does, and a test asserts
  the marker sits on the driven node.
- **Own cost budget — the SWEEP is the cost driver, not the audio.** The audio/band/animation is one
  full run + one clean run (cheap, single-node scalar solve). The tuning curve needs ~24 sweep points,
  each a short run; the sweep dominates. `JUARI_N_MAX`, `JUARI_AUDIO_MAX`, a fixed `JUARI_SWEEP_POINTS`
  at a reduced `JUARI_SWEEP_DUR` (the centroid stabilises fast — window study above), and a combined
  `JUARI_WORK_MAX` over sweep+audio+clean. Exact numbers pinned against measured per-step cost during
  implementation (jawari was ~143 µs/step for a 15-node vector solve; a 1-node scalar solve is
  cheaper). The clean run is reused for both the tuning-curve baseline and the band-spectrum contrast.

### Batch 15 (DONE) — the air pushes back (`RadiatedBody`): the BOOKED radiation channel

**Built & browser-verified.** All-wrapper (`physsynth/core` untouched); 17 web tests (307 in the web
file, 1100 in the suite), ruff clean, headless verifier `radbody` PASS, CDP switch-check **17/17**,
both panel crops eyeballed. Shipped defaults, measured through the payload: at R = 133 the air takes
**98.4 %** of the pluck within 2 s while the booked total drifts **1.60e-14**; the body itself peaks
at **0.10 %** (a conduit, not a reservoir); the drain is fastest at **R ≈ 5.16 (20.0 ms)** in a
basin **2.29–7.74**, against **252 ms** at the default — the panel's whole point. Default render
**2.93 s** (44 444 audio steps + 53 453 sweep steps). Two bugs only the *rendered page* could catch,
both recorded below.

The last member of the body/radiation family, and the one thing batches 12–13 structurally *could
not* show. There the air was a pure **read-out**: `AirRadiation` scales the body's volume
acceleration to a far-field pressure and takes nothing — by construction it has no `energy()`. Here
the air is a **load**: `RadiatedBody` (a rank-1 passive dashpot, [[radiation-state]] batch 2) puts a
resistance `R` on the body's net volume velocity, removes `P_rad = R·U²` watts, and books the total
in `radiated_energy`. Because `RadiatedBody` is a `ModalBody` **drop-in**, it slots into batch 12's
`StringBodyBridge` as the body with **zero core edits and zero bridge edits** — the rig is batch 12's
with one substitution, so every difference on screen is the back-reaction and nothing else.

Model key **`radbody`**, its own entry (not a domain of `body`) — batch 13's `platebody` precedent:
it keeps batch 12's 15 tests and payload untouched, and earns its own ranges and cost budget. Scope
is the **lumped** modal body only; `RadiatedBody` wraps a `ModalBody`, and grid/plate radiation
load is explicitly deferred in the core. All-wrapper as always.

**Everything below is measured** (`temp/radbody-probe/`, probes 1–6), and three of the numbers were
gates the advisor set before any panel code:

- **THE BOOKED DRIFT CLEARS THE 1e-10 BAR BY FOUR ORDERS — the flagged coupling-order hazard is
  empirically FALSE.** [[radiation-state]] records the standalone identity at 7.86e-14 but the full
  chain only as "< 1e-9", and `StringBodyBridge` precomputes `beta_b` from the *bare* body and
  commits its spring force *before* `RadiatedBody.step` applies the rank-1 correction — so an
  `O(k·R·U)` mismatch growing with `R` and `K` was plausible, and would have landed exactly where
  the demo lives. Measured over `R ∈ {0 … 20 000}` × `K ∈ {0, 8 000, 19 000}`: drift **3.9e-15 …
  6.8e-14**, no trend in either. The `< 1e-9` was a test tolerance, not the number. So the verdict
  needs no redesign: **σ_body = 0 → the ordinary lossless drift check, on the BOOKED total.**
- **The verdict rides the booked total, and `sigma_zero` stays `(sigma_body == 0)` REGARDLESS of R**
  — the bore's precedent (batch 9), second customer. `R > 0` is not a loss, it is a *channel*:
  `bridge.energy()` already includes `body.energy() = E_body + ∫P_rad`, so a radiating run is still
  a *conserving* run. Default drift **2.15e-14** while 98.4 % of the pluck has left as sound.
- **The money panel is the four-channel BOOKED SPLIT**, and the four channels sum to the flat
  reference to **1.6e-14** — `E_string + E_body + E_conn + ∫P_rad ≡ 1`. Batch 12's `drawBodyEnergy`
  gains ONE additive line (`e_rad_frac`); it does **not** route through `_energy_block(split=…)`
  (the bore's mechanism renders on a different path, and the body family's panel is the exchange
  panel — batch 13 already reused it verbatim). Batch 12's rules carry: `E_conn` **goes negative**
  (measured −0.06 %) so it is never clamped and never stacked, and the flat total is a **reference**,
  not the verdict.
- **`R = 0` is BIT-IDENTICAL to batch 12** — `np.array_equal` True on both the energy and the
  pressure traces over 4 000 steps, not merely close. The anchor test builds both payloads from ONE
  explicit shared param dict and compares arrays (the `model` key differs by design).
- **The second panel is the `t₅₀`-vs-`R` sweep: radiation has an OPTIMUM, and more air is worse.**
  `t₅₀` = the time for the radiated channel to reach 50 % of the pluck's energy. Measured minimum at
  **R ≈ 4** (19.6 ms) inside a broad basin `R ≈ 2–10`, rising to 251 ms at R = 133 and 1.49 s at
  R = 800 — the air can *choke* the body it is draining (`U = U_free/(1+RG)`, so `P = R·U² → 0` as
  `R → ∞`).
- **…and that optimum is PHYSICS, not the timestep — the advisor's impostor, checked and cleared.**
  The scheme's own turnover sits at `R = 1/G`, `G = (k/2)Σaᵢ²/(mᵢ(1+σᵢk)) ∝ k`, so `1/G ∝ fs`:
  111 → 222 → 444 across N = 50/100/200. The measured curve does **not** move — `t₅₀` at R = 3 is
  **24.48 / 24.52 / 24.57 ms** over that 4× refinement (0.4 %), and every window's peak lands on the
  same R at all three N. The physical scale it tracks instead is `σ_rad/ω₁ = a²R/(2mω₁) = R/27.6`,
  which is the dimensionless control the sweep's axis and the readout both use (#9's `ΔT/T₀`, #10's
  `frac`). The optimum is also **K-invariant** (R = 4 at K = 2 000 / 8 000 / 15 000 / 19 000 — the
  bridge sets how fast energy *reaches* the body, not which R best empties it), and it is
  **amplitude-invariant bit-exactly** (linear model: `max|frac(A) − frac(2A)| == 0.0`) — a free
  oracle no nonlinear model in the viewer can offer.
- **`t₅₀` is the observable BECAUSE fraction-radiated is window-dependent — batch 14's exact trap.**
  The peak of "fraction radiated at time T" walks **R = 3 → 10 → 30** as T goes 0.05 → 0.15 → 0.4 s,
  and at σ_body = 0 the fraction saturates toward 1 for any R > 0 given enough time, so *any* fixed
  window reports a different sweet spot. `t₅₀` is a rate: window-free (it only needs the run to get
  there) and grid-converged. Points that do not reach 50 % inside the cap are **labelled censored,
  never plotted as a number** (the jawari/bow "label, never fail" rule).
- **The sweep is a CONTROLLED REFERENCE CURVE, decoupled from the render** (batch 13's terminus
  probe, batch 1's measurement pair): fixed `N = 100` and **`sigma_body = 0` always**, with the
  user's `K`/`L`/`T`/`ρ`/pluck. A fixed N is legitimate *because* the curve is N-independent — but
  it cannot be an arbitrary one, because the coupled guard ceiling *shrinks with fs*
  (K_c = 8 563 / 10 716 / 21 479 / 34 394 N/m at N = 40 / 50 / 100 / 160), so a sweep at a **lower**
  fixed N than the render could trip a guard the render passed. N = 100 is above the whole
  `bridge_stiffness` range (max 19 000 < K_c(100) = 21 479), so it is always constructible. σ_body = 0
  is required too: on a lossy body most of the energy goes to σ, `t₅₀` censors nearly everywhere, and
  the sweep both lies and costs 7× more.
- **The sweep is CHEAP because it exits early** — each point stops the instant it crosses 50 %.
  18 log-spaced points over `R ∈ [0.3, 300]`, cap 0.8 s each: **47 845 steps / 1.4 s** at the default
  (K = 8 000), 46 151 at K = 19 000, and 120 573 / 3.4 s at the soft-coupling corner K = 1 000 (2
  points censored). The worst case is **K = 0**, where nothing ever reaches the body (radiated energy
  after 0.3 s is *exactly* 0.0) and all 18 points would run the full cap — so **K = 0 skips the sweep
  entirely** with a labelled "no coupling — nothing radiates", rather than burning 320 000 steps to
  draw a curve of `NaN`.
- **Cost is step-count only — the per-step cost is N-INDEPENDENT** (28.8 µs bare / 56.8 µs
  instrumented at both N = 100 and N = 160; the Python-level body+bridge overhead dominates, not the
  grid). Default render (N = 100, 2 s) = 44 444 steps ≈ 2.3 s, plus ~1.4 s of sweep. Budget mirrors
  batch 12: `RADBODY_N_MAX = 160`, `RADBODY_AUDIO_MAX = 3`, `RADBODY_WORK_MAX = 200 000`, plus a
  sweep-specific step budget that censors the remainder rather than hanging.
- **Default `R = 133` is the free-space monopole resistance at the first body mode** — `R_a(ω) =
  ρ₀ω²/(4πc₀)` = 133.4 at 110 Hz — which makes the read-out and the load *exactly* consistent at one
  frequency, and the panel prints which: `f_match = √(4πc₀R/ρ₀)/2π` = **109.8 Hz** at R = 133. It
  also renders the best picture: the split ramps 0 → 0.13 → 0.42 → 0.66 → 0.92 → 0.98 over
  0.05/0.2/0.4/1/2 s, a legible drain across the whole panel.
- **HONESTY: `R` is constant in frequency and the true monopole `R_a ∝ ω²`** (133 / 424 / 751 / 2135
  at the rig's 110 / 196 / 261 / 440 Hz body modes) — one constant *cannot* be right at all four. The
  readout states it and names the one frequency where load and read-out agree; frequency-dependent
  `R` is a separate, harder core batch. No second "radiated" analysis pretends otherwise (batch 12's
  rule).
- **The load KILLS batch 12's slosh, and that is the finding, not a defect.** Peak `E_body` falls
  monotonically 0.773 → 0.491 → 0.195 → 0.072 → 0.0010 at R = 0 / 1 / 4 / 10 / 133: loaded by the
  air, the body stops being a *reservoir* and becomes a *conduit*. Slosh visibility and drain
  legibility are anti-correlated (both governed by σ_rad), so no single R shows both — the batch
  leads with the drain, and **the slider makes the contrast live**: pull `R` to 0 and batch 12's
  77 % slosh returns, bit-identically.
- **Param hygiene:** the new slider is **`radiation_R`** (distinctive; no other model sends it),
  linear `[0, 300]` step 1, default 133 — reachable endpoints for the anchor (0), the basin (2–10)
  and the physical value (133), with the log-axis sweep panel carrying the decades the slider
  cannot. **Reset in `MODEL_RANGES._default`** — the leak family (batches 2/3/7/8/12). Reuses
  `bridge_stiffness` / `sigma_body` / `distance` / `amplitude` / `pluck_position` / `lambda` / `N`
  from the body ranges, re-ranged per that model's existing entries.

**Found during the build (not in the plan above):**

- **A SLIDER THAT DOES NOT EXIST LOOKS EXACTLY LIKE ONE THAT WORKS, because the backend default
  fills in.** The `radiation_R` markup landed *nested inside* the `distance` slider's `<div>` (an
  insertion that reused the opening tag and left the original `</div>` behind), so `buildSliders`
  never created it — and the fresh-load PNG still rendered a flawless R = 133 panel, because
  `gatherParams` simply shipped no `radiation_R` and the server applied `RADBODY_R_DEFAULT`. The
  deep-link verifier's "does it draw?" check cannot see this class of bug at all; only enumerating
  the built `.slider` params over CDP did. *Generalizable: when a param has a server-side default,
  a missing control is invisible in the picture — assert the control EXISTS, not just that the
  render is right.*
- **A leftover headless Chrome answers `/json` on the debug port and hands you ITS page** — the
  stale-server trap one level over. Three runs of the switch-check "found" the missing slider in a
  browser that had loaded the pre-fix DOM, which is also how the *fix* appeared not to work. The
  driver now proves the attached DOM matches disk before asserting anything, and uses a fresh
  `mkdtemp` profile per run. (`--remote-allow-origins=*` is also required, as the main verifier
  already does.)
- **Two panel collisions the numbers could not show:** a centred x-axis title lands exactly on the
  `100` decade tick and a rotated y-axis title lands on the `300` tick, so the units moved to the
  empty corners (`Pa·s/m³` after the last decade, `t₅₀ ms` inside the plot's top-left). And `∫P_rad`
  was first drawn teal, a colour that reads as the green *total* at a glance — the one pair that
  must never be confused, since the whole claim is one filling up to the other. It is now amber.
  `body_frac_peak` needed a sub-1 % format too: the conduit finding rounds to a bare "0 %".
- **`_RadBodyRun` is deliberately NOT a subclass of `_BodyRun`.** Inheriting batch 12's telemetry
  meant sampling `wb` / `qaccel` / `u_end` — three extra calls on every step of a model whose entire
  budget is step count — to fill arrays that feed the far-field spectrum panel this batch replaces.
- **The hidden `sweep_points`/`sweep_cap` overrides needed the shipped path pinned SEPARATELY.**
  They keep the suite from paying for the full 18-point sweep seventeen times, but because every
  radbody test sets them, the configuration the UI actually ships — both keys *absent* — was
  exercised only by the verifier. A constant can be right while the default path reading it is not,
  so the settings test now renders with the keys omitted and asserts the grid length and cap that
  come back.
- **`RADBODY_SWEEP_WORK_MAX` is a GUARANTEE, not dead code — and is unit-tested past itself.** The
  worst reachable sweep is ~120 k steps against the 200 k cap (K = 0, the catastrophe, is *skipped*
  rather than budgeted), so `truncated` cannot fire in the shipped range. Batch 1's rule applies:
  do not widen a range to make a gate fire — drive it past the guard, where the behaviour is real.
  With the budget lowered the tail must come back as labelled nulls with `truncated` set, keeping
  its place on the axis; silent truncation reading as "covered everything" is the failure the
  no-silent-caps rule names.

### Batch 16 (DONE) — the parametric instability: model #9's second refusal, discharged

The last batch named in this plan's own backlog, and the discharge of a refusal the tension batch
wrote into the code. `_build_payload_tension` raises above `TENSION_DT_MAX = 4.45` with *"the
breakup is real, energy-conserving physics; it wants a panel of its own."* This is that panel. It is
the **in-plane** modal exchange — the same `2ω` parametric pump batch 3's whirl aims at the other
polarization — and it is what batch 1's purity gate was protecting the Duffing oracle *from*.

**A REGIME of `tension`, not a new model key.** The pattern the viewer already follows: a different
resonator class earns its own key (`radbody`/`platebody`, `RadiatedBody` vs `ModalBody`); the same
class with a different IC is a regime (geometric's planar/whirl/rotating/phantom). This is
`TensionModulatedString` with a different mode IC and a larger amplitude — the same-class case, and
the exact analogue of batch 4, which discharged model #9's *first* refusal as a regime of geometric.
`tension` has no secondary select today, so this adds one (`DOMAIN_OPTS.tension`, the batch-6
`data-show` fix on the domain container, `tension:parametric` keys for `hasRegimeRanges`). Core is
untouched; `_build_payload_tension` stays bit-for-bit; **`TENSION_DT_MAX` is NOT relaxed** — the
refusal stays exactly where it is, and now it points somewhere.

**Everything below is MEASURED before any panel code** (`temp/parametric-probe/`, probes 1–6,
`FINDINGS.md` + raw JSON). Three of the six probes overturned a prior assumption, two of them the
advisor's.

- **`m` MUST be a control, and m=1 is the one mode that does not do this.** The tension path is
  hardwired to `_mode1_shape`, and the m=1 threshold is `ΔT/T₀ ∈ (4, 6]` (batch 1 measured (4.44,
  6.05] independently) because it has no lower resonance partner. Measured thresholds: m=1 (4, 6];
  m=2 (3, 4]; **m=3 (1.75, 2.25]** — the documented case, the core test's case, the default; m=4 and
  m=5 both (1, 2]. `_mode1_shape` generalizes to `_mode_shape(res, m)`; the animation stride rides
  **mode m**, not f₁; and **run length is specified in PERIODS OF MODE m, never steps** — at a fixed
  step count, raising N raises fs and shrinks the physics-time window until the breakup slides off
  the end of the panel (`TENSION_MEASURE_PERIODS`' pattern).
- **THE PARTNER IS NOT FIXED — it moves with `ΔT/T₀`, which corrects the recorded result.** For m=3
  the energy goes to **m=7** just above the edge (`ΔT/T₀` 2–3), to **4 and 11** deeper in, and to
  **4 and 8** at `ΔT/T₀ ≈ 9`. Memory's "m=3 → m=4 and m=8" was measured at ≈11 and is *one tongue
  among several*, not the answer. The cascade panel must therefore **never label a fixed pair** —
  it reads the partners off the run. Off-mode energy sits **1e5–1e7 above grid scale** throughout,
  which is the discriminator against a numerical instability (the core test's `>100×` bar, with
  four orders of headroom).
- **The claim runs are BOTH deeply nonlinear — "below" is not a linear control, and that is the
  stronger claim.** At the m=3 pair straddling the edge, `nonlinear_fraction` at the IC (which is
  the peak: max displacement, zero velocity) reads **0.42 at `ΔT/T₀` = 1.5** against **0.49 at 2.0**
  — tension 2.5× vs 3.0× rest, i.e. *equally* nonlinear at that pair. The shipped default pairs 1.5
  against 3.0, where it is 42 % vs 59 % — still both deep in the nonlinear regime, no longer equal,
  so the panel claims only the part that holds at every pair: the tongue is the entire difference,
  and the stable run is no linear control. (The panel first said "equally", measured here and
  rendered there — see the found-during-the-build note below.)
- **NO new verdict type** (batch 3's precedent). Nothing drives this string, so the ordinary σ=0
  drift check IS the claim; what makes it one is what it survives. **Drift stays 1e-15…1e-12 across
  all ~150 probe runs, including complete disintegration** (off-mode 0.5) — a numerical instability
  grows the energy, a parametric one only redistributes it. Printed *next to* the disintegration,
  the whirl's pattern. σ is **fixed at 0 and deliberately unexposed**: loss decays the amplitude
  back down through the threshold, so the instability self-extinguishes, and there is no
  single-exponential form for multi-rate redistribution to fit a decay oracle to (jawari's
  σ₁/hysteresis precedent — say why in the docstring so nobody "fixes" it into a slider).
- **Gate the claim on `n_not_converged == 0`.** A tension root-find failure at high `ΔT/T₀` looks
  exactly like a breakup, and the drift number is meaningless if the step never converged. The
  tension `convergence` block already ships; reuse it and let it override the badge (VK's gate).

**The seed — explicit, fixed, printed, and defined MODALLY.**

- **The growth rate is seed-independent; only the onset time shifts** (by `log(seed)/rate`).
  Measured at m=3, `ΔT/T₀`=2: envelope-fitted rate **70.1/s at seeds 1e-11, 1e-9 and 1e-6 alike,
  r² = 1.000**, against 68.4/s for roundoff-only — 2.5 %. So an explicit seed costs nothing.
- It is taken anyway, because it makes the below/above contrast **honest**: with roundoff seeding,
  "below is flat" is partly "below was never seeded." Same seed both sides, every sweep point
  starting from the same floor with identical dynamic range to grow through. The self-seeding claim
  is pinned in a **test** (seed = 0 still breaks up, still conserves, still lands in the neighbours)
  rather than in the shipped render — which also closes [[tension-string-state]]'s flagged BLAS
  risk, since nothing shipped depends on roundoff any more.
- **Convention: a unit-norm combination of modes 1..24 (mode `m` removed) with fixed
  `default_rng(12345)` coefficients, scaled to `1e-6·A`.** Defined by MODAL content on purpose —
  probe 2's `standard_normal(N+1)` draw spread a fixed norm over twice as many modes at N=200 and
  confounded the two-N comparison. Broadband so the tongue picks its own winner rather than us
  naming a partner in advance. `1e-3` is too large: no dynamic range below saturation, the fit
  returns garbage. Batch 3's displacement-vs-velocity convention trap applies — state it in the
  payload.

**The sweep panel — the saturated off-mode LEVEL, after two rejected observables.**

- **Rejected #1: the fitted growth rate.** Clean only *at* the edge (r² = 1.000 at `ΔT/T₀` = 2.0),
  degrading to r² ≈ 0.4 above 3.0 and returning non-monotone rates (m=4: 2.3/s at 3 vs 44.8/s at 4).
  The cause is physical, not estimator choice — above the edge several tongues overlap and the
  content saturates within ~5 periods, so there is no single exponential to fit. (Fitting the
  ENVELOPE rather than the raw signal is still mandatory — raw off-mode content is a growing
  *oscillation*, batch 3's whirl lesson — it is just not sufficient.)
- **Rejected #2: `t_break`.** The natural radbody-`t50` analogue, and it looked right — monotone,
  refinement-convergent (46.6/39.5/38.4 ms at N=100/150/200 for `ΔT/T₀`=4), cheaper the deeper it
  goes. It dies on the near-edge points, which **never break at any cap**: at `ΔT/T₀` = 2.25 the
  off-mode content sits on **0.130 flat across 40, 60 and 90 periods**; 2.50 → 0.155, 2.75 → 0.190.
  They are not slow, they are **bounded** — a *partial* exchange in which the mode is degraded, not
  destroyed. A fixed break level is an arbitrary line through a continuum of plateaus.
- **Shipped: the saturated level.** Window-free for the reason that matters — it *plateaus*, and
  saturation is **verified per point** (chunk-to-chunk in periods of mode m; a chunk that beats the
  previous by < 10 % is saturated) so a point that has not settled by the cap is **labelled, never
  reported as if it had**. Measured curve (m=3): flat at the seed floor for `ΔT/T₀` ≤ 1.75, then
  0.061 / 0.130 / 0.155 / 0.190 / 0.207 / 0.238 / 0.261 / 0.295 / **0.332** rising monotonically to
  `ΔT/T₀` = 6. On a log axis the growth ratio jumps **1.1× → 4e5×** across the edge: that jump *is*
  the sharp-threshold claim.
- **Three bins, not two** — `stable` (level < 100× the seed floor), `unsaturated` (grew, had not
  settled at the cap), `saturated`. Two bins conflate genuinely stable points with slow ones and
  read as a much wider stable region than exists. No silent caps: everything keeps its place on the
  axis with its label.
- **What is N-invariant, and what is NOT — stated, not glossed.** Above `ΔT/T₀` = 2.25 both grids
  agree within ~10 % (0.130/0.112, 0.190/0.178, 0.238/0.251, 0.332/0.373 at N=100/200) and the curve
  keeps its shape. The **edge location does not hold still**: 2.00 (N=100), 2.00 (N=150), 2.25
  (N=200), and N=200 carries an isolated narrow tongue at **1.75 that grows into partner m=10**
  where every other case goes to m=7, plus a null at 2.00. This is real Mathieu structure — a tongue
  has edges on *both* sides in detuning, so a partner enters and leaves resonance as the amplitude
  rises — whose fine position depends on which discrete partner frequency is available.
  [[tension-string-state]]'s "refinement-invariant" was measured at `ΔT/T₀` ≈ 11, deep in the merged
  region, where refinement-robustness does hold. **Claim the convergent quantity; describe the edge
  as the tongue structure it is.** A naive two-N invariance claim would have been wrong here.
- **Controlled reference curve** (radbody's precedent): fixed N, σ = 0, the user's m — the threshold
  is *not* mode-invariant, so a sweep at the wrong m is a different curve. The fixed N and the seed
  are printed on the panel, and the y-axis reads "saturated off-mode level **from a 1e-6 seed**",
  never a bare level (batch 14's canonical-duration lesson: the seed sets the intercept, the physics
  sets the slope).

**Panels.** (1) *Headline* — off-mode envelope vs time on log-y, below and above, with the unstable
run's **energy drift on the same axes** at ~1e-13: straight line up, flat line at machine precision.
`physsynth/viz/plots.py:plot_mode_purity` is the template; `drawWhirl`/`drawSympathetic` are already
log-y, so this is their third customer, not new infrastructure. (2) *The cascade* — modal content at
the end of the unstable run, partners read off the run with a grid-scale reference line. (3) *The
tongue sweep*, above. The **animation needs no new viz**: a clean m-lobe sine visibly losing its
shape is the 1-D line path already there, and the frames come out of the above-threshold run.
**Audio** is cheap from that same run — shipped, not headlined, and it must not set the claim runs'
duration (batch 14).

**Cost — the one item still unmeasured, deliberately.** A 12-point sweep with the saturation exit is
**9.9 s at N=100 / 26.4 s at N=200**, and the claim runs are ~1.0 s each at 60 periods, so the sweep
is the driver and gets its own budget with the *stable* points priced (they alone run the full cap).
Measured per-step cost here is ~350 µs at N=100 — **2× the 176 µs the tension batch recorded at
N=128**, which is exactly why this gets re-measured on the real payload build rather than
extrapolated (batch 4's rule: extrapolating a loop understates a payload build). Trim levers, in
order: fewer sweep points, a shorter saturation chunk, a lower cap.

**Traps carried in from prior batches.** `_default` leak family (2/3/7/8/12) — every param this
re-ranges resets, `m` included, and this regime's amplitude range must exceed the Duffing regime's.
Batch 15's **control-existence check**: `m` and the sweep knobs will have server-side defaults,
which is precisely the bug class where a missing slider still renders a flawless panel — enumerate
the built `.slider` params over CDP. Score the unstable run's drift over the **FULL** array, not the
decimated one (batch 2). And the deep-link verifier structurally cannot fire the regime switch
(batch 4) — a CDP switch-check drives it.

**Found during the build (not in the plan above):**

- **A SHARED slider is only shared if it is BUILT in both places — and `dt_over_t0` lived inside the
  whirl's own fieldset**, which is `data-show="geometric"`. Sharing the name would therefore have
  produced, under `tension`, a control that never gets created at all while `gatherParams` shipped
  nothing and the server quietly applied its default: batch 15's bug class exactly, reproduced by a
  different route. The slider moved to the shared *Nonlinearity* box, which both models show. The
  placement is also the honest one — it is one physical coordinate (the driven mode's tension
  excess), and the whirl's 2.2 ceiling exists *because* of the instability this regime crosses.
- **THE LEFTOVER-CHROME TRAP RECURRED, AND BATCH 15'S RECORDED FIX DOES NOT CATCH IT.** That fix
  was "prove the attached DOM matches disk" — but the marker it used is a *version* marker
  (`does the DOM contain radiation_R / mode_number?`), and a leftover browser **from an earlier run
  of the same script** has identical markup and sails straight through. `subprocess.terminate()`
  kills the launcher, not the browser tree, so every run left a live Chrome holding the debug port;
  runs 2 and 3 failed to bind it, attached to run 1's page, and read back the state run 1 had left
  (`domain=parametric`) as two FAILs that were not real. Two changes: the check kills its own
  leftovers first, matched on its `param-chrome-` profile prefix (never the user's browser —
  21 processes identified before anything was killed, per the standing rule), and the freshness
  marker is now a **SESSION** marker: a pristine page has the markup defaults in its selects
  (`model = ideal`, `domain = ""`), where a leftover is mid-experiment. *Generalizable: to detect a
  stale attachment, assert something unique to THIS session, not to this version of the code.*
- **"Has it rendered yet?" is NOT that session marker** — the first attempt, and it failed
  immediately on a genuinely fresh page. Autorender is on by default, so a pristine page has
  already rendered `ideal` by the time the websocket attaches.
- **ROUNDING A LOG-AXIS SERIES IN THE PAYLOAD DELETES THE CLAIM.** `_finite_list(drift, 12)` rounds
  to twelve *decimal places*, so a drift of 1e-13 ships as a bare `0.0` — and the flat
  machine-precision line that is half of this batch's argument silently disappears from the panel.
  The off-mode traces have the same problem at their seed floor. Both now ship unrounded (they are
  ~425 points each). *Round and format for display; never in the payload of a series that will be
  drawn on a log axis.* Pinned by a test asserting the smallest positive drift value survives.
- **A partner that rounds to "0.0 %" is not a partner.** The first cut named the top four off-modes,
  which beside a real m=7 at 20 % printed "m=10 0.0 %" — modes a whisker above the grid scale, which
  pass a naive `> grid_scale` filter and then round to nothing on screen. The bar is the **core
  signature test's own**: 100× the grid scale, plus `PARAM_PARTNER_FLOOR` on top, set at 1e-3 by
  the *readout's* resolution (one decimal of a percentage) rather than by a physics scale — what is
  not worth printing is not worth naming, and that bar belongs in the payload, not the drawing code.
  At the shipped default the two bars are not close: the real partners sit at 2.0e-1 and 1.4e-1 and
  the phantom at 6.6e-8 against a 100×-grid-scale bar of 4.7e-6, three orders of headroom. Below
  the tongue the list comes back EMPTY, and `over_grid` is `null` rather than `0.0`, which would
  read as "at grid scale" instead of "there is no winner to price".
- **THE BACKEND DOES NOT HOT-RELOAD, so a "regenerated" screenshot can be a PRE-FIX render.** The
  phantom above was fixed, the verifier re-run, the PNG re-read — and the phantom was **still
  there**, which sent me chasing a second bug (a floor an order too low) that did not exist. The
  server had been running since before the fix landed; `serialize.py` is imported once. The false
  trail cost more than the original bug. *Sibling of the leftover-Chrome trap, and the same shape:
  the browser was fresh, the tab was fresh, the **module** was stale — a re-render only re-runs the
  code the server already has.* Restarting it (the launcher's `terminate()` again leaves the Python
  process holding the port, so the port must be reclaimed by PID) made the phantom vanish on the
  spot. Confirmed after the fact by rebuilding the payload in-process with the old floor restored:
  identical output, i.e. the 1e-4 floor had *always* been sufficient. **Re-render after a backend
  edit ⟹ restart the server first, or the picture is testing the previous commit.**
- **"Both runs are equally nonlinear" was refuted by the two numbers printed beside it.** The line
  was written from the probe pair (0.42 at ΔT/T₀ = 1.5 vs 0.49 at 2.0 — genuinely equal), but the
  shipped default pairs 1.5 against **3.0**, where it renders as "equally nonlinear: 59 % vs 42 %".
  The claim only ever needed the weaker form, which is also the one that carries the batch: the
  stable run is **no linear control** — both runs are deep in the nonlinear regime and the tongue
  is the entire difference. *A sentence measured at one operating point ships at another; re-read
  the rendered claim against its own numbers, not against the probe it came from.*
- **A hint that reads a control the regime HIDES prints a stale number.** `tension-hint` computes
  ΔT/T₀ from the `amplitude` slider, which the parametric regime hides because A is derived — so it
  had to be gated to the Duffing regime, and the parametric regime got its own hint that speaks in
  the coordinate it actually exposes.
- **Cost, measured on the real payload build rather than extrapolated:** the shipped default render
  is **5.45 s** (two 40-period claim runs plus the 8-point sweep), against the membrane's 5.6 s
  worst passing render. Per-step cost is **200–290 µs** across N = 100…200 and both regimes —
  the tension batch's recorded "~176 µs at N=128" was optimistic, and the sweep's *stable* points
  are the expensive ones because they alone run the full cap.

### Batch 17 (DONE) — the air as an IMPEDANCE: `Z_a(ω)` stores as well as radiates

**Built & browser-verified.** All-wrapper (`physsynth/core` untouched, core test count unchanged,
the core-dep allowlist guard green); **21 web tests (357 in the web file, 1474 in the suite —
+21 and all of them in `test_web_backend.py`, so the delta *is* the batch)**, ruff clean, headless
verifier `airload` PASS
alongside `body`/`platebody`/`radbody`, CDP switch-check PASS. Shipped defaults, measured through
the payload: the air takes **97.4 %** of the pluck within 2 s and **holds up to 18.1 %** of it in
the radiation mass on the way, while the body still sloshes to **60.5 %** and the five channels sum
to the conserved total with a ledger residual of **2.30e-16** (drift 3.56e-14). The sweep reports
`α` spanning **105.85×** over 110–1760 Hz against a constant-`R` impostor pinned at **8.26 s⁻¹**
(**69.31×** under-damped at the top), and a fundamental **10.13 % flat** (~175 cents). Default
render **3.84 s** (44 444 audio steps + a 44 710-step sweep).

**Phase D's model list closed in 2026-07 and this document forbade inferring a batch 17 from it —
"if a new viewer batch is wanted it needs a new model or a new capability first."** That condition
is now met: the human took the fork's **core `R(ω)`** arm on 2026-07-27 and
`RationalAirLoad` / `ReactiveRadiatedBody` shipped ([[radiation-state]] batch 3), which is the
*first* new resonator-side core since Phase D closed. This batch surfaces it. (The other new core
family, the 3-D air box, is deliberately **not** in scope: it needs a 3-D field type the frontend
does not have, and it is its own batch.)

The batch that discharges a caveat batch 15 wrote against itself. There the air was **one number**:
`RadiatedBody` puts a constant resistance `R` on the body's volume velocity, and the readout had to
confess *"`R` is constant in frequency and the true monopole `R_a ∝ ω²` (133 / 424 / 751 / 2135 at
the rig's four body modes) — one constant cannot be right at all four."* Here the air is an
**impedance**: a resistance in parallel with the radiation mass `M_a`, which is the *exact* acoustic
load of a pulsating sphere and a first-order rational function of `jω`, so it needs no filter fit.
The audible consequence is the one a constant `R` cannot produce — high partials radiate better and
die first — and the *inaudible* one is that the air now **stores**, which gives batch 15's booked
ledger a **fifth channel**.

Model key **`airload`**, its own entry (batch 13/15's precedent: a different load class earns a
key). All-wrapper as always; `physsynth/core` is untouched.

**Everything below is measured** (`M:\claud_projects\temp\reactive-probe\`, probes 1–5), and the
first bullet is a headline this batch had already drafted **backwards** — the advisor's catch,
recorded because the wrong version was one edit from shipping:

- **`RadiatedBody`'s `R` and `RationalAirLoad`'s `R` are NOT the same physical quantity, and that —
  not legibility — is why the default moves 133 → 13146.** Batch 15 set `R = 133` from
  `monopole_radiation_resistance`, the **compact-source** (`ka → 0`) law `ρ₀ω²/(4πc₀)` evaluated at
  110 Hz. `RationalAirLoad`'s `R` is the **saturated plane-wave** (`ka → ∞`) value `ρ₀c₀/S`. Reading
  batch 15's number as a saturation implies a **0.497 m** sphere, whose exact `Re Z_a` over the band
  is 66.6 / 101.3 / 113.0 / 125.3 — flat, and **−50 % … −94 %** against batch 15's quoted law. That
  reading would have made batch 15's honesty note look like a confession to a sin it never
  committed. It is wrong. The sphere batch 15's number actually describes is the **5 cm** one, whose
  exact `Re Z_a` is 132.1 / 410.4 / 710.7 / 1836.8 — **−1.0 % / −3.1 % / −5.4 % / −14.0 %** against
  the compact law it quoted. **Batch 15's note was correct**, its numbers were good to 14 % over the
  band, and what it could not do was *use* them. The shipped default here is that same 5 cm sphere,
  which is why all four body modes sit **below** the corner, where `Re Z_a` climbs. *Generalizable:
  two batches can name a slider `R`, cite the same medium, and mean different limits of the same
  formula; check which limit a number came from before claiming continuity between them.*
- **The exact-algebra identity is real but near-tautological — it ships as a sanity check, not as
  evidence.** The sphere-consistent corner `f_c = √(R c₀ / (ρ₀ π))` equals batch 15's `f_match`
  identically (measured rel. difference **0.00e+00** at R = 10 / 133 / 1000, 1.7e-16 at 300). It has
  to: `f_match` is where the compact law meets the constant, and the corner is where the compact
  limit meets the saturation — both are `ka = 1`, by construction.
- **THE ANCHOR IS A SLIDER SETTING, AND IT REACHES BATCH 15's SHIPPED PICTURE, NOT JUST ITS
  MECHANISM.** `air_corner = 0` → `M_a = ∞` → `R_eff = R` exactly and the auxiliary state stays
  exactly zero, so with `radiation_weight = 1` and `radiation_R = 133` the whole chain is
  **bit-identical** to batch 15: `np.array_equal` True on the energy, the far-field pressure *and*
  the radiated channel over 8000 steps. Two things this rests on, asserted rather than assumed:
  `ModalBody(radiation=1.0)` broadcasts to exactly `phi.copy()` (checked), and the construction path
  (boundary, `BODY_BODY_FREQS`/`BODY_BODY_MASS`, `fs` from λ/N) is batch 15's verbatim. All three
  values are reachable on the shipped sliders, so the anchor is something the *user* can dial.
- **THE MONEY TEST IS THE LEDGER RESIDUAL, NOT THE DRIFT — the air-box lesson, third customer.**
  The conserved total clears the 1e-10 bar everywhere measured (**7.7e-15 … 1.1e-13** over corner ×
  R × K × weight), but it is **structurally blind to a wrong panel**: `load.energy()` returns stored
  *plus* radiated while `body.energy()` is bare, so a batch-15 copy-paste can double-count or drop a
  channel and leave the drift green. The guard is the per-step
  `max|E_total − (E_string + E_body + E_conn + E_stored + ∫P_rad)| / E₀`, measured **2.30e-16**,
  against controls that are loud: sign-flip `E_stored` → **3.63e-01**, drop it → **1.81e-01**, with
  drift unmoved at 4.7e-14 in both.
- **The money panel is therefore batch 15's booked split with a FIFTH line — and the fifth one is
  the batch.** `E_string + E_body + E_conn + E_stored + ∫P_rad ≡ 1`. `E_stored` peaks at **18.1 %**
  of the pluck at the shipped default: the air is genuinely a reservoir, not a rounding error.
  Batch 12's rules carry: `E_conn` goes negative (measured −0.108 %) so it is never clamped and
  never stacked, and the flat total is a **reference**, not the verdict.
- **…but it is a rounding error in batch 15's OWN rig, which is why batch 15 could not have drawn
  it.** At `radiation_weight = 1` the stored channel peaks at **0.16 %** (corner 30 Hz) to
  **0.69 %** (corner 110 Hz) — invisible. It only becomes a line at a physical radiation weight.
- **THE RADIATION WEIGHT, NOT `R`, IS THE KNOB THAT DECIDES WHETHER ANYTHING IS VISIBLE.** Batch 15
  inherited `a_i = phi_i = 1` from `ModalBody`'s default (`radiation` falls back to `phi`), which
  puts the body at `α/ω₁ = 4.8` — **past critical damping**, and mode-coupled through the shared
  volume velocity, so exciting one mode returns a mixture (a 110 Hz mode measures 152 Hz) and
  `loaded_mode` diverges. The core's own validated regime is `radiation = 0.02` at `mass = 0.02`.
  This batch exposes the weight as a slider and defaults it to **0.05**.
- **AND AT A PHYSICAL WEIGHT, BATCH 15's ANTI-CORRELATION DOES NOT BIND.** Batch 15 measured that
  "slosh visibility and drain legibility are anti-correlated, so no single R shows both" and shipped
  the drain. With the reactive load at weight 0.05 the render shows **both**: 97.4 % radiated by 2 s
  *and* a 60.5 % body slosh. The claim was true of `R` at weight 1; it is not a property of the
  radiation channel. Measured across the weight slider — peak `E_body` 0.716 / **0.605** / 0.139 /
  0.0085 and radiated-at-2 s 0.824 / **0.974** / 0.974 / 0.762 at weight 0.02 / 0.05 / 0.1 / 0.2 —
  0.05 is the only setting that is near the top of *both*, which is the impedance match, and it is
  also the **loudest** (peak far-field 0.56 / **0.96** / 0.33 Pa at weight 0.02 / 0.05 / 0.2).
- **The second panel is the CONTROLLED SINGLE-MODE SWEEP: `α(f)` and the pitch drop, measured
  against the closed form, with the constant-`R` impostor beside them.** One mode at a time, body
  alone, no string — because that is the only rig in which `loaded_mode` is defined and in which
  "per-mode" means anything (see the mode-coupling bullet above). Batch 15's precedent exactly: a
  reference curve with pinned parameters, decoupled from the render. At the shipped default `α`
  climbs **5.41 → 572.06 s⁻¹** over 110 → 1760 Hz (**106×**), against a flat impostor pinned at
  **8.26** — under-damping the top of the sweep **69×**.
- **The pitch drop is the half a constant `R` cannot even attempt**, and it is large: **−10.14 %**
  at the fundamental (≈185 cents) against an oracle of **−10.11 %**, falling to −3.50 % at 1760 Hz
  because `Im Z_a / ω` decays above the corner. A constant `R` has `Im Z ≡ 0`, so its shift is
  **exactly zero** — the one number in the panel that needs no measurement.
- **THE PITCH SHIFT MUST BE READ AGAINST THE SCHEME'S OWN UNLOADED FREQUENCY, NOT THE NOMINAL `f₀` —
  otherwise the leapfrog's warping is credited to the air, and at the render's own rate it gets the
  SIGN wrong.** At fs = 22 222 Hz (N = 100, λ = 0.9) a 1760 Hz mode measures **+0.55 %** against
  nominal while the oracle says −0.53 %; against a measured unloaded run at the same fs it is
  −0.65 %. So every sweep point pays for a second, `R = 0` run, and the panel plots the difference.
- **The sweep is pinned at 48 kHz, and that is a decision the render's rate forces.** The oracle
  error at 1760 Hz is 2.61 % at 22 kHz, 1.15 % at 48 kHz, 0.86 % at 96 kHz and 0.79 % at 192 kHz —
  i.e. it converges to a floor that is *not* the scheme. Pinning at 48 kHz puts the remaining
  disagreement in the formula, where the panel can name it.
- **…and the remaining disagreement is EXACTLY the weak-loading formula's own stated order, with
  the coefficient measured.** `loaded_mode` documents a residual "second order in `α/ω`, ~1 % at
  `α/ω ~ 1 %`". Over the shipped sweep `α/ω` runs 0.0078 → 0.0540 and the error runs 0.10 % →
  5.49 %; the ratio **err/(α/ω)² is flat across a 7× range** — 16.4 → 18.8 on the 25-cycle probe,
  **15.9 → 21.5 as shipped** at 12 cycles (the shipped figure; the probe's tighter digits are not
  what renders). So the panel does not say "the oracle disagrees at the top" — it says the residual
  is second order with coefficient ≈ 16–21, and the measurement is primary while the oracle is an
  overlay with a printed validity.
- **The sweep's cost is set by its per-point run length, and that is NOT what limits accuracy** —
  40 cycles capped at 24 000 steps costs 149 060 steps / 3.57 s and 12 cycles capped at 6 000 costs
  **44 710 steps / 0.97 s**, with the *same* median oracle error (1.51 %) and max (5.53 % vs
  5.47 %). Shipped: 12 points log-spaced over 110 → 1760 Hz, 12 cycles, 6 000-step cap, two runs per
  point (loaded + unloaded). That is batch 15's sweep budget almost exactly (47 845 steps / 1.4 s).
- **The coupled `K` guard is untouched by the load** — measured ceilings bare vs reactive are
  identical to the digit at N = 40 / 50 / 100 / 160: **8563.1 / 10715.7 / 21478.5 / 34393.8**, which
  are batch 15's recorded numbers. `StringBodyBridge` takes `beta_b` from the *bare* body, so
  `K_MAX = 19000` and the fixed-N reasoning carry over verbatim.
- **Param hygiene.** Three new sliders. **`radiation_R`** is *re-ranged* from radbody's `[0, 300]`
  to `[0, 20000]` — its `index.html` home is radbody's range, so it must reset in
  `MODEL_RANGES._default` or an `airload → radbody` switch renders a 44× load: the leak family
  (batches 2/3/7/8/12/15), and this is its worst instance yet because both models *accept* the
  param. **`air_corner`** (Hz, `[0, 3000]`, default 1091.8) is distinctive; `0` is the batch-15
  anchor and is self-documenting on the slider. **`radiation_weight`** (`[0.005, 1]`, default 0.05)
  is distinctive. Reuses `bridge_stiffness` / `sigma_body` / `distance` / `amplitude` /
  `pluck_position` / `lambda` / `N` / `audio_duration` from radbody's ranges.
- **HONESTY, and it is a different one from batch 15's.** (a) The audio is `AirRadiation`'s
  **compact** read-out — batch 15's path verbatim, which is what makes the anchor bit-identical —
  while the load's own far field is the finite sphere's, low-passed by a further `1/(1 + jka)`; the
  two differ by up to 14 % in `Re Z` over this band and only the sphere's balances the booked
  channel exactly (`test_far_field_power_balances_the_booked_radiated_power` in
  `tests/test_radiation.py` pins that, so this batch cites it rather than re-deriving it).
  (b) `radiation_weight` scales the
  read-out, so **the audio level moves with a slider that is not a volume control** — and not even
  monotonically (peak 0.56 / 0.96 / 0.33 Pa at weight 0.02 / 0.05 / 0.2, peaking at the impedance
  match). The readout prints the peak so "quieter" does not read as physics. (c) `R` and
  `air_corner` are exposed as **independent effective coefficients**, so most of the plane is not
  any sphere; the readout prints the equivalent radius implied by each (`a_R = √(ρ₀c₀/4πR)`,
  `a_τ = c₀/2πf_c`) and whether they agree, rather than forbidding the combination.

**Found during the build (not in the plan above):**

- **`R = 0` — the one setting that means "no air at all" — RAISED, because the corner's own formula
  collapses at it.** `M_a = R / (2π f_c)` is the inversion of the exposed coordinate, and at `R = 0`
  it returns a *zero* mass, which `RationalAirLoad` rejects outright (`M_a > 0`). So the anchor the
  plan leaned on hardest was the single value the parameterisation could not express. The fix is one
  clause — decoupling is a property of `R`, and the corner has nothing left to say about it — but
  the lesson generalises: *when you expose a derived coordinate, check the endpoints of the
  coordinate you derived it FROM, not just of the one on the slider.*
- **BATCH 15's INVISIBLE-CONTROL BUG WAS STILL LIVE, and it is a DIFFERENT mechanism from the one
  batch 15 recorded — the correction matters more than the finding.** `pluck_position`'s markup
  lists `body platebody radbody` in its `data-show`, but the Excitation *fieldset* around it listed
  only `body`, and the fieldset gate wins. Batch 15's case was "the control is never built, so
  `gatherParams` ships nothing and the *server* default fills in". **This one is the opposite:**
  `buildSliders` iterates every `.slider` with **no visibility filter** and `updateVisibility` only
  sets `el.hidden`, so the control *is* built and `gatherParams` — which iterates the built map —
  ships it on every request. The value it ships is `MODEL_RANGES.<model>.pluck_position.val`,
  re-asserted by `applyModelRanges` on every switch. So the user simply **lost the knob**: on two
  shipped models the pluck point could not be moved at all. Verified, rather than assumed, that the
  worse reading is *not* live — `body`, `platebody`, `radbody` and `airload` all carry
  `pluck_position: { val: 0.3 }`, so a value dragged on another model is overwritten on entry
  rather than leaking in. It was silent only because the frontend's 0.3 and the backend's 0.3
  happen to agree; had they differed, the picture would have changed with nothing on screen to say
  so. **Two gates, two lists, and only the narrower one is visible in a diff.** Fixed for all four
  models.
- **The switch-check's readiness condition was itself the bug it was written to catch.** Waiting on
  `document.querySelectorAll('.slider').length > 5` is satisfied *before any JavaScript runs* —
  the divs ship in `index.html` and are empty until `buildSliders` fills them — so the first run
  "found" all 64 sliders visible and none of them containing an `<input>`, i.e. three false
  failures and one false pass. The real signal is the presence of a *built* `input#s-…`.
  *A readiness check that can be satisfied by the static markup is not a readiness check.*
- **Two panel collisions the numbers could not show, radbody's lesson repeating one axis over:**
  the x-unit `Hz` placed after the last octave label lands *on* the `1760` tick (the last tick is
  centred on the axis end), and a right-hand axis title lands on the corner marker's label. The unit
  moved to the bottom-left gutter — the one spot this panel leaves empty — and the Δf unit rides its
  legend entry instead of being written twice.
- **`E_air` renders as a BAND, not a curve, and that is the physics.** The stored energy oscillates
  at **2ω** — energy moves in and out of the radiation mass twice per cycle — so at the exchange
  panel's time resolution it fills a hairy band near the axis rather than tracing a line. Left
  un-smoothed and *said* in the readout: an envelope would have been prettier and would have hidden
  the one visual fact that distinguishes a reservoir from a drain.
- **The readout quoted a peak its own picture did not reach.** `stored_frac_peak` is run-wide while
  the exchange panel plots only `AIRLOAD_EXCHANGE_WINDOW` seconds, so a window-peak scalar was added
  and the sentence about it is now *conditional* — at the shipped default the two coincide, and
  printing "the run-wide peak comes later" beside a picture already showing it would be a caption
  contradicting its panel.
- **The label for `radiation_R` had to stay NEUTRAL**, and that is this batch's headline in
  miniature: the shared `LABELS` map is consulted by both models, and any qualifier ("saturated",
  "compact") would be correct for one and wrong for the other. The per-model hint carries the
  meaning; the label carries only the name.
- **`resid_coef` ships wider than the 25-cycle probe measured** — 16.4–18.8 there, **15.9–21.5** as
  shipped at 12 cycles. Same claim (flat across a 7× range in `α/ω`, so the residual is second
  order), looser digits. *A coefficient measured at one run length ships at another; re-read the
  rendered number.* (Corrected at the source bullet above too, not only here — the air-box rule:
  put the correction where the wrong number is.)
- **Only 4 of the 12 sweep points sit inside `α/ω ≤ 0.02`** at the shipped weight, and the panel
  prints that fraction rather than hiding it. The measurement is primary everywhere; the *oracle* is
  what has a validity band, and saying which points are inside it is the honest version of an
  overlay that visibly diverges at the top.
- **THIS BATCH SHIPPED INTO A TREE WHOSE CI WAS ALREADY RED, AND NOBODY NOTICED FOR 18 RUNS.**
  Discovered *after* the close-out, on 2026-08-10: the last green run is `31325446976`
  (air-box batch 4's `SurfacePort` feature commit) and every push since — through both air-box
  close-outs, the HANDOFF §11 decisions and all five batch-17 commits — has been `failure`. The
  seven failing tests are **all pre-existing air-box** (six
  `test_airbox_dipole.py::test_surface_port_is_bit_identical_across_the_shared_spreading_refactor`
  parametrisations — since renamed `…_is_unchanged_across_…`, see the fix below — and
  `test_airbox_surface.py::…_even_mode_is_silent_…[mode1]`); batch 17's own
  357 web tests are green on Linux, and the one thing the failure list *does* say about this batch
  is that its `…_bit_identically` anchor is the safe kind (bit-identity between two runs in the same
  process, not against literals captured on one machine). **Do not blame `-qq` for the blindness.**
  CI passes `-q` on top of `addopts = "-q"`, which suppresses the `N passed` summary — but the
  `FAILED …` lines were printed in every one of those runs. The suite was not silent; it was
  *unread*. **A 24-minute suite whose verdict nobody opens is worse than no suite: it converts
  "tests exist and pass" into an assumption while charging full price for it.**
- **BOTH failures were the same mistake: a number captured on ONE machine, asserted as if it were
  physics.** Fixed 2026-08-10. The dipole goldens were compared with `==` against literals captured
  on Windows, where Linux differs in the last ULP (`6708.019018618780` vs `…779`) — so the test was
  a *platform* assertion wearing a refactor assertion's clothes, and it passed only on the machine
  that wrote it. And `_peak_monopole` measured `|Σq|/net_area`, a *dimensional* quantity, against an
  **absolute** `1e-13`: it reads `7e-14` here and `1.6e-13` there, and the bar was simply sitting on
  the roundoff. The cures are opposite in shape and the same in kind — the golden test drops to
  `rel=1e-12`/`1e-11` while keeping `node_count`, `nnz_T` and the index-weighted digest as the exact
  *permutation* detectors (**and its docstring now says what the tolerance no longer catches**: a
  pure summation reordering of the stepping loop); `_peak_monopole` becomes the dimensionless
  cancellation ratio `|Σq|/Σ|q|`, which caps at 1, separates the zeros (`<1e-11`) from the radiating
  controls (`~1`) by **eight** orders instead of six, and cannot be flattered by a run that barely
  moves — measured, the `(1,1)` control now reads exactly `1.000000`, which is a *better* statement
  of "this mode has a monopole" than the old `3.16` was. **A tolerance that is not scale-free is a
  golden value with extra steps.**
- **And the first cure was still wrong for one of the six, for the third instance of the same
  mistake.** `rel=1e-11` on `radiated` held, but three `p` cases stayed red at 8.0e-11 / 3.3e-10 /
  1.9e-10 relative. `room.p.sum()` is a **cancelling** sum — `Σp ~ 1e2` out of `Σ|p| ~ 1e6`, a
  cancellation factor of **650 to 362 000** across the six cases — so its roundoff is set by the
  *terms* and reading it as a fraction of the *result* inflates it by exactly that factor. Against
  `Σ|p|` the same three deviations are **3.8e-14, 9.1e-16, 1.1e-13**: plain last-ULP, like
  everything else. The tell is that **`nearest` is the worst case under the wrong denominator and
  the best under the right one** — its cancellation factor is 3.6e5, two orders above the others.
  *Choosing "relative" over "absolute" is not the decision; choosing the scale the error is
  generated at is. A cancelling sum has no meaningful relative tolerance against itself.*
- **CI green on 2026-08-10 after 20 red runs — and the first quiet-machine profile of this suite
  says the web backend is NOT its bottleneck.** Adding `--durations=50` to CI cost nothing and
  inverted the ranking a contended local run had given: the top is `test_geometric_energy`
  (281 s), then the *module fixture setup* of `test_geometric_whirl` (252 s) and
  `test_geometric_phantom` (217 s) — batch 17's most expensive test is 83 s, sixth. And the shape
  outranks the ranking: whirl and phantom are `xdist_group`-**pinned**, so ~720 s and ~425 s each
  run on a **single worker** against a ~1400 s wall. Two modules are the critical path. Checked
  before treating it as reclaimable: their fixtures build 11 runs with **measured zero overlap**,
  so the sharing is already maximal and the cost is the physics.
- **The suite got 9.0 % faster and the honest part of that sentence is how hard it was to say.**
  Memoising identical simulations at the `_sim` seam (50 calls → 10 runs, reproducibly) plus
  splitting the 4-simulation fret test took the full suite 1501.9 s → 1366.1 s; the `-m "not slow"`
  lane runs 1411 of 1476 in 725.0 s (**1.90×**). But **GitHub's runner class varies ~1.6×**: the
  *identical untouched* whirl fixture read **252.05 / 252.09 / 158.50 / 252.76 s** across four
  consecutive runs, and a 15:56 total sat between two 22:46s purely on hardware. The 9 % is
  quotable **only** because its two runs read 252.05 and 252.09 on that reference. *A wall-clock is
  not a measurement until it is normalised against something in the same run that did not change —
  which is the airbox lesson above, one more time, in a costume that looks like a stopwatch.*

### Batch 18 (DONE) — the room: the first field with a SHAPE in three dimensions

**Built & browser-verified.** All-wrapper (`physsynth/core` untouched, core test count unchanged,
the core-dep allowlist guard green — `airbox` is a new *core* import for the web tier but adds no
third-party dependency, and the guard measures the `sys.modules` delta); **13 web tests** (372 in
the web file, **1489 in the suite** — +13 and all of them in `test_web_backend.py`, so the delta
*is* the batch), ruff clean repo-wide.

**The two headless cases are ADDED but were NOT run by the script**, and the build record says so
rather than implying otherwise. `scripts/verify_web_headless.py` launches its own Chrome on the
DevTools port and could not reach that endpoint while an already-running Chrome held the browser —
this document's own leftover-Chrome trap (b16), one costume further. What *was* verified, in a live
browser over CDP, is the same ground the two cases cover: both deep-links render "ok", the slice
panes paint, `domain=absorbing` reaches the backend as `walls` **and the ζ slider is built AND
reachable** (`offsetParent !== null`, value 3 — the b15/b17 invisible-control check, passed), and
the absorbing field max drops to 5.41e-2 Pa against rigid's 1.14e-1. *The script's own pass/fail
path for these two names remains unexercised; running it needs a machine with no competing Chrome.* Shipped defaults, measured through the payload:
the cone reads **46 cells, measured 46, MATCH**, against a Euclidean arrival of **52.8 steps** and a
physical-Manhattan one of **88.5** — the claim beats both, and it is an integer. Cross-ledger
residual **6.89e-18**, scene drift **2.49e-14**, air holding at most **2.76e-2 %** of the scene.
Default render **8.6 s** (13 202 steps × 35 588 nodes), payload **2.8–3.1 MB**, 155 slice frames.

**Six things the build corrected, every one of them a measurement rather than a review comment.**
Recorded because the wrong version of each was working code at the time:

- **The animation clock was the STRING's, and the room does not run on it.** `anim_stride` came
  from `fs / (f1 · FRAMES_PER_PERIOD)` like every other model — but the thing being animated is a
  *wavefront*, which crosses the shipped room in ~3.5 ms and advances `λ_air` **cells** per step.
  On the string's clock that rendered **18 frames** and showed the crossing in about four of them.
  On the room's (one frame per cell of travel) it renders **155**. *Generalizable: catch #2 says
  the stride must resolve the fundamental — but "the fundamental" means the oscillator you are
  drawing, and in a coupled scene that is not always the one the model is named after.*
- **The slice decode was TRANSPOSED, and it looked like physics.** The backend ravels
  `room.p[::s, ::s, at]` with shape `(nu, nv)` in C order, so **`nv` is the fastest axis**; the
  frontend read `u = q % nu`. The result was not a crash or a garbled buffer — it was smooth
  **horizontal banding**, which is a perfectly plausible thing for a room to do. What caught it was
  arithmetic, not the picture: at 110–440 Hz the wavelength is ~3 m against a 1.2 m room, so the
  field *must* be smooth and a striped one is impossible. **No payload assertion could have seen
  this** — the bytes were correct — which is exactly why the headless case earns its cost.
- **`open` walls are LOSSLESS, and calling them lossy would have hidden the bug the bar exists to
  catch.** `Z = 0` is pressure-release: it reflects with inversion and dissipates **exactly 0.0**
  (measured). The first draft keyed "lossless" off `walls == "rigid"`, which quietly exempted a
  conservative scene from the 1e-10 conservation bar. It now keys off the token, and *both* rigid
  and open sit on the bar (drift 6.1e-15 and 7.9e-15).
- **A NaN builds a perfect payload and dies at the transport.** `zeta` was `float("nan")` on
  non-absorbing walls; `web/server.py` dumps with `allow_nan=False`. `simulate_to_payload` returned
  a valid dict, every in-process assertion passed, and the browser got a 500. *An in-process
  serializer test structurally cannot see a serialization bug* — so the suite now calls
  `json.dumps(..., allow_nan=False)` the way the server does.
- **Seven whole-room energy sums per step cost ~3 s of the render.** The run now samples at **two
  rates**: the mic and the port every step (the cone is an exact integer that a decimated trace
  would destroy), the five channels every `e_stride`. The residual and drift are therefore maxima
  over ~595 samples, not over every step, and `e_stride` ships in the payload so the bound is
  readable rather than implied.
- **The coupled cone must be timed from the first nonzero INJECTION, not from step 0.** The string
  is plucked in *displacement*, so the body — and hence the port — starts at rest and first injects
  at step 1. Timing from step 0 would have put the integer off by one, and it would have looked
  like a rounding problem in the claim rather than a bookkeeping one in the origin.

**And one design fact that only appeared once the two grids had to share a timestep:** `air_cfl`
near 1 is refused by the **bridge**, not by the room — at `cfl = 0.99` the lower `fs` trips
`StringBodyBridge`'s exact guard (`k²λ_max(A) = 4.15 ≥ 4`) before the air's CFL is anywhere near
violated. The binding constraint at the top of that slider is the *coupling*, and the error message
that surfaces is the core's own.

*What is below is the plan as written before the build, kept because four of its claims died in
probing and that record is the point.*

### Batch 18 (PLAN, as written before the build) — the room

**The legality check first, because this document forbids inferring a batch from itself.** The
standing rule is "if a new viewer batch is wanted it needs a new model or a new **capability**
first," and batch 17 named this very batch's blocker: the 3-D air box "needs a 3-D field type the
frontend does not have, and it is its own batch." That capability is what this batch builds. The
core side has been ready since 2026-08-10 — `AirBox` and its four batches ([[air-box-state]]) are
the one remaining built-but-unshown core family, and the *only* one.

**Everything below is measured** (`M:\claud_projects\temp\airbox-viewer-probe\`, probes 1–3),
before a line of the batch was written. Four plan claims died in those probes; they are recorded
here with the numbers that killed them, because in this project a probe that only confirms is a
probe that was aimed at the wrong thing.

#### The gate: what a 3-D room costs, which is not what 1-D and 2-D intuition says

- **The CFL runs the WRONG WAY in 3-D, and the cost is `h⁻⁴`, not `h⁻³`.** `λ = c₀k/h ≤ 1/√3`
  means `fs ≥ c₀√3/h`, so a **coarser** grid forces a **higher** sample rate: refining `h` buys
  `h⁻³` nodes *and* `h⁻¹` steps. Measured on a fixed room and duration, halving `h` twice:
  work `3.49e5 → 4.94e6 → 7.59e7`, i.e. **×14.2 then ×15.4** against the predicted 16. Every 1-D
  and 2-D model in this viewer has a `WORK_MAX` derived from an `h⁻²`/`h⁻³` instinct; this one
  cannot borrow it. Measured throughput is **~5e7 node-steps/s** at ≥1e5 nodes (1e8 at 1e4 — the
  large grids are memory-bound), and `AIRBOX_WORK_MAX` comes from that, the membrane/plate/VK way.
- **The shipped viewer room is `(1.2, 0.9, 0.8)` m at `h = 0.03`** → `N = (40, 30, 27)`,
  35 588 nodes, `fs = 22 003 Hz`, **~7.4 s wall per simulated second** bare. The coupled port chain
  measures **12–16 s/s**, which is the same order as the shipped `vk` and `platebody` renders.
- **`λ = 0.9/√3` is the default, NOT the ceiling.** `λ = 1/√3` makes grid-diagonal modes exact —
  tempting, since it would turn `mode_frequency` into a machine-precision oracle worth printing —
  but it is also where the corner mode goes **defective**, so a flat energy there is not a
  stability certificate ([[air-box-state]]). The claim below turns out not to need it (see the
  λ-independence measurement), so the ceiling stays a refusal rather than a setting.

#### DIED IN PROBING: the two shapes this batch was going to have

- **DIED — "the echo and the null live in the same run."** They cannot. The directivity rig needs
  the first reflection to arrive *after* the measurement window; the echo claim needs the
  reflection to **be** the point. Opposite requirements, so opposite rigs — which is what actually
  forces this batch's shape, and it was an assumption, not a measurement, until probe 2.
- **DIED — the `object` regime (a coupled `RoomSuspendedPlate`) as the carrier of the null.** It is
  the most expensive option in the batch (12–16 s/s) buying the **weakest** claim. The 85× null is
  a **prescribed-velocity** figure for a documented reason (`RoomSuspendedPlate.radiated_energy`:
  "a radiation figure needs a prescribed-velocity rig"), and a *free* plate's fundamental is the
  **saddle/twist** ([[free-plate-state]]) — a quadrupole, whose arc is not the dipole's. Coupling
  the plate and printing 85× beside it would have been a number measured on one rig and captioned
  with another's. Cut.
- **DIED — directivity as an audio-bearing regime.** The rig is a **5 m cube** (1 030 301 nodes) —
  it has to be, to window the burst before the first image. A 200-step burst costs a measured
  **4.1 s per arm** (~12 s for all three, *cheaper than the juari sweep's 37 s*), but 0.3 s of
  audio in that same room is `3.7e9` node-steps ≈ **74 s**. So directivity can never carry audio.
  It ships as a **decoupled fixed-cost panel** with its own rig parameters printed beside it —
  which is exactly the juari precedent ([[juari-state]]: the sweep is a canonical duration
  *decoupled* from audio), arrived at here from the opposite direction.
- **DIED — my own first echo probe, which never tested its claim.** The mic was axis-aligned, so
  the direct path and the Manhattan path were the *same* 0.600 m and the measurement could not
  distinguish them. The claim was only actually tested after the mic moved off-axis. *A probe whose
  geometry makes two candidate answers equal has measured neither.*

#### The shape that survives: one audio run + one decoupled panel

Model key **`airbox`** (batch 13/15/17's precedent: a different coupling class earns a key).
All-wrapper as always — `physsynth/core` is untouched, and the core-dep allowlist guard is re-run
because `airbox` has never been imported by the web tier before.

**The audio-bearing run is `string → bridge → RoomLoadedBody → AirBox`, not a bare room.** A bare
read-out has exactly **one** ledger and therefore nothing to be blind to, which makes for a weak
honesty gate on a batch whose real risk is the slicer. `RoomPort` is a rank-1 scalar solve that
costs almost nothing over the bare room, `RoomLoadedBody` delegates every `ModalBody` accessor so
batch 12's machinery is reused unchanged, and it buys a genuine **cross-ledger** test.

#### The claim: the lattice light cone — an exact integer, and λ-independent

The headline is an **index**, never a magnitude — batch 2's own finding is that the room
contaminated the port's measured size by more than the effect, and the viewer room is small.

On a 7-point Yee stencil the numerical domain of dependence after `n` steps is
`|di| + |dj| + |dk| ≤ n` **cells**, so the first nonzero at a mic sits on the **Manhattan distance
in cells** — earlier than the physical wavefront, and independent of `λ`. Measured, off-axis mic at
cell offset `(21, 10, 10)`, where the three candidate answers are cleanly separated:

| | Euclidean | physical Manhattan | lattice cone (L1 cells) | **measured first nonzero** |
|---|---|---|---|---|
| `λ = 0.9/√3` | 48.72 steps | 78.90 steps | 41 | **41** |
| `λ = 1/√3`   | 43.85 steps | 71.01 steps | 41 | **41** |

plus an axis-aligned control (cone 21, measured **21**, both λ) — **4/4 exact**. The integer does
not move when `λ` does, which is why the default can sit safely below the ceiling *and* why no λ
caveat belongs in the panel text. The detector is exact-zero-vs-nonzero on a field with zero
initial data, so it is not a threshold choice: at `1e-3` of peak the same mic reads 43 and at 2 %
it reads 46, and **those** numbers do move with λ. The panel prints the cone index as the claim and
the amplitude-threshold arrivals beside it as what they are — the physical wavefront, arriving
later than its own precursor.

#### The money test is the CROSS-LEDGER residual, not the room's drift

Third and fourth customers for this lesson already exist ([[air-box-state]] batches 3 and 4;
[[web-viewer-state]] b17). `AirBox.energy()` = `acoustic + dissipated − injected` reads flat at
~1e-21 in every probe run above **with a wrong coupling constant**, because each side's ledger
telescopes against whatever pressure *it* used. The gate is therefore
`|body.radiated_energy − room.injected|` against the room's own post-closure pressure, **at two
timesteps**, with the scene total shipped as a sanity line and explicitly *not* as the evidence.

#### The new capability: `dims: 3` is a SLICE SET, and the doc says so plainly

Three named orthogonal planes per frame (`xy`, `xz`, `yz`) through a slice point, each decoded by
the existing `drawHeatmap` path. **Not volumetric, and the payload field is named so nobody reads
it as volumetric.** A 32³ volume × 300 frames is ~10M floats — dead on this document's own
JSON-size trap. Measured: 3 slices × 300 frames at the 64-cap decimation is **5.26 MB base64**,
comparable to a shipped membrane payload. The slice point is a **server-side** slider (a re-render);
a client-side movable slice would require shipping the volume, which is the thing that does not fit.

**The colour map is a measurement, not a taste.** A wavefront's **leading edge** is the faintest
part of the frame and the thing the viewer must see. Measured mid-flight, as a fraction of frame
max: median **2.2e-3**, faintest live decile **3.0e-7** early, rising to 2.7e-2 later, while frame
max itself swings ~2× within the run on top of a **385×** first-frame-to-peak spread. So
global-linear renders the early frames black and per-frame-linear hides the decay: **a compressed
signed map is forced**, and *which* map ships is printed in the payload rather than left implicit
(the batch-3 scar: five plan claims died because the prototype measured where the effect was
hidden).

#### Pre-flagged traps

- **The fieldset beats the slider.** `data-show` on the enclosing `<fieldset>` wins over the
  slider's own, and `buildSliders` has **no visibility filter**, so a control can be *built and
  shipped* yet unreachable — live twice already (b15, b17, by opposite mechanisms). Add `airbox`
  to every enclosing fieldset, not just the new sliders.
- **The backend does not hot-reload.** Restart it between backend edits when verifying in Chrome,
  and use a **session** marker for stale-tab detection — a version marker cannot catch leftover
  Chrome (b16).
- **`h` is snapped, and the snap is the resolution.** `N_d = round(L_d/h)`, so the room the user
  gets is `L_actual`; report it, never silently resample it away (the juari precedent).
- **Do not print a `t50`.** It is a magnitude, and at viewer room size the room contaminates it.
- **Never quote probe 2's `image` times for the 2.0/3.0 m rows** — the monkeypatch did not take
  (`free_room(..., room_m=FREE_ROOM)` binds the default at def time), so all three arcs ran the 5 m
  cube. The 49 / 72.4 / **88.7×** series is therefore a clean pure **arc-radius** series (r = 0.70 /
  1.05 / 1.20 m) and nothing else; the null is a far-field measure and the radius is load-bearing.
- Ruff ≤100 chars in the first draft; `addopts` already carries `-q`, so **never pass `-q`**.

#### Touch list

`web/serialize.py` (constants + `_build_payload_airbox` + the dispatch at ~857) ·
`web/static/index.html` (option, sliders, **and every enclosing fieldset**) · `web/static/app.js`
(`MODEL_RANGES` + `_default`, the viz dispatch at ~2050, the energy branch at ~2392/2538,
diagnostics at ~2908, the new slice-set draw fn) · `tests/test_web_backend.py` (cross-ledger
residual through the wrapper, frame bookkeeping, slice decimation, `λ > 1/√3` → a clean error
payload, not a 500/NaN) · `scripts/verify_web_headless.py` (add the `airbox` case).

### Batch 19 (DONE) — the gong in the room: radiation that CHANGES mid-strike

**Built & browser-verified.** All-wrapper (`physsynth/core` untouched, core test count unchanged, the
core-dep allowlist guard green — `airbox`'s VK classes are a new *import* for the web tier but add
no third-party dependency, and the guard measures the `sys.modules` delta); **12 web tests**, ruff
clean repo-wide. **The two headless cases are ADDED but were NOT run by the script**, and this record
says so rather than implying otherwise -- b18's precedent, and its exact situation. What was verified
is a live browser over CDP, not `scripts/verify_web_headless.py`'s own pass/fail path. That gap
matters more than usual here, because the case added is precisely the one that would have caught the
hidden-tab failure below, so its unexercised state is worth a line rather than silence.

Both tiers render in a live browser: the hung cymbal shows two-sided lobes and the
baffled gong a half-space arc off its wall, which is the physics of the two mountings and not a
styling difference. Shipped defaults, measured through the payload: `fs = 57,904 Hz` (set by the
ROOM), a 0.353 m cube of 32,768 nodes, 6,948 steps, 72 max Picard sweeps at a mean of 11.6, **all
steps converged**, scene drift **2.6e-13**, cross-ledger residual **9.3e-15**, render ~23 s against
the shipped `vk` model's 15.1 s default and 52.7 s ceiling.

**Six things the build corrected, every one a measurement rather than a review comment.** Recorded
because the wrong version of each was working code at the time:

- **THE MAGNITUDE IS NOT GRID-CONVERGED, AND THE PLAN DID NOT KNOW IT.** The plan quoted 29.3× and
  65.0× as if they were properties of the plate. Refining the air cell by 12 % (11.40 → 10.00 mm)
  takes the struck arm's resolved spread **7.04 % → 5.00 % → 2.37 % → 1.99 %** while the linear twin
  holds at 0.05–0.12 %; even a **0.1 %** change in `h` moves it 7.04 % → 2.78 %. So the **separation
  is the claim** — never below ~17× at any rig measured, and 102× at the shipped one — and the
  multiplier is **refused**, in the panel text as well as here. This is batch 5's and batch 6's own
  doctrine arriving for a **third** time and with a stronger reason than either: not merely that
  unresolved modes contaminate the figure, but that it is **not converged even inside the resolved
  band**. *Generalizable: "quote the resolved band" is necessary and not sufficient — a
  resolution-restricted observable can still be a function of the resolution.*

  **...and that series had a confound, which was removed rather than argued away.** Refining `h`
  moves the resolved SET with it -- `n_res` read 17 / 19 / 19 / 21 across those four rows -- so each
  row measured a different set of modes, and the fall could have been the observable *changing*
  rather than failing to converge. The fix is exact, not statistical: the modal basis does **not**
  depend on the air grid (`plate.K`, `kappa`, `wdiag` are functions of the plate alone), asserted
  rather than assumed -- `vecs` and `freqs` come out **bit-identical** at every `h`, and only the
  boolean mask moves. Pinning the 17-mode mask from `h = 11.40 mm` and reusing it verbatim at every
  finer grid, the spread falls **6.99 % -> 4.99 % -> 2.37 % -> 2.02 %**, i.e. the moving-band series
  to within a tenth of a point, while that same fixed band holds the linear control at **0.08 % and
  0.05 %**. The resolved-set change contributed almost nothing. The non-convergence is real, and the
  correction this batch makes to a shipped batch's headline therefore rests on a controlled
  measurement rather than a suggestive one.
- **…and part of that is the STATISTIC, not the physics: the spread is max/min of four near-equal
  numbers.** The probe rig and the shipped payload differ by 0.03 % in sample rate. Their four
  per-window efficiencies agree to ~1 % each — 0.6339/0.6523/0.6676/0.6662 against
  0.6324/0.6532/0.6769/0.6694 — and yet the quoted multiplier differs by **33 %** (1.053 against
  1.070), because a ratio of near-equal numbers amplifies whatever moves either end. The payload
  therefore ships the **per-window values** as the evidence and the spread as a derived, captioned
  number. *Generalizable, and cheap to miss: an effect quoted as max/min is reported with a
  sensitivity nobody chose.*
- **A free plate's fundamental is the FOURTH eigenvalue, and "the first frequency above epsilon"
  silently picks a rigid-body mode.** The nullspace is exactly `{1, x, y}` (model #5b) but `eigh`
  returns those three at ~1e-5 Hz rather than at 0, so the filter selected one, `f1` printed as
  **0.0**, the animation stride came out ~3× longer than the entire window, and the plate pane
  rendered **one frame**. The index 3 is a *fact about the model*, not a threshold, and that is why
  it is now written as one.
- **The claim's read-out is free to decimate 5×, and that was worth checking rather than assuming.**
  Sampling the modal projection every step against every 5th moves the headline from 6.991 % to
  7.045 % — inside the sensitivity above — so ~400 samples per window bound the window mean
  honestly, and `proj_stride` ships so the bound is readable rather than implied (b18's `e_stride`
  precedent).
- **A HIDDEN CHROME TAB MAKES A CORRECT FIELD READ AS A BLANK CANVAS.** The field canvas measured
  **0.0 % painted** with no console error, while calling the same draw function directly painted
  **123,904** pixels: `requestAnimationFrame` does not fire in a background tab (measured **0 ticks
  in 1.2 s**). This is b18's browser-verification hazard in a new costume and the *inverse* of it —
  there, a leftover Chrome held the DevTools port so the script could not run at all; here the
  browser runs fine and **throttles the very thing under test**, which fails in the direction of
  looking like a bug rather than looking like a pass. Verify a canvas by driving its draw directly,
  or foreground the tab first.
- **The room was one commit away from shipping as DEAD PAYLOAD.** The field pane took the plate
  (`dims: 2`) and *nothing* drew `room_frames` — a 0.4 MB block, built and serialized and never
  looked at. That is precisely the built-but-unshown gap Phase D exists to close, about to be
  created *inside the batch closing it*. It ships as a dual pane instead (b13's precedent), with the
  two halves on **different clocks** — the plate on its first flexural mode, the room on acoustic
  transit — mapped by time and clamped, because the plate window outlasts the room window.

**What was refused, each with the number that decided it.** The **supported gong**: it runs at this
rig (72 sweeps, converged) but resolves only 6 of 225 modes, on which the separation collapses to
3.8× baffled and runs **backwards** suspended (0.3×) — so `domain` carries the mounting and the
boundary is pinned to `free`. The **compact monopole** read-out (measured 3e-7…3e-6 of the truth) is
shipped beside the claim as a caption, never as it, and the absence of a far-field pressure is
asserted by a test. No `t50`, no directivity panel, and **no string drive** — the three-way-chain
batch already measured this exact headline flat under a string (0.27 % against a 0.28 % control),
because `w/e` is not an amplitude under a point force.

**Two things this batch does NOT know, stated rather than left to be discovered.** The plate pane's
animation window is 4 ms of a 120 ms run -- the *stride* was chosen deliberately (b18's trap) but the
*window* is 3 % of the render, so the pane shows the **strike** and not the shape drift the batch is
named for; the drift is carried quantitatively by the claim panel instead. And `plate_N` is dialable
down to 8, where the plate is known to **converge** (probe 1D) but where the resolved band and the
separation were never measured -- the range is legal and the numbers are probably fine, and that
"probably" is exactly the distinction being flagged.

**And one plan claim that died in the build.** "The two amplitude sliders are one knob" — the strike
curvature `w/e ÷ width²`, which both cliff brackets put at 83–85 from opposite directions — is wrong
off the diagonal: at a constant ratio of 75.3, `w/e` = 1 and 2 run (23 and 40 sweeps) while 5 and 8
are dead. Amplitude binds and a broader strike does not buy it back.

#### The three things batch 19 recorded rather than resolved, resolved (2026-08-17)

The batch closed by naming three items it had not done. All three are now measured. **Two of them
end as measurements rather than as code**, which is the outcome and not a shortfall — one of them
because the measurement says the change would have shown the eye nothing.

**1. The headless cases run now — and the gap was TWO batches wide, not one.** `out/` contained no
`viewer_airbox*.png` **at all**, so batch 18's pair had never been run either; that is a file
listing rather than the inference b19 recorded. All five (`airbox`, `airbox_absorbing`, `vkroom`,
`vkroom_baffled`, and the new `vkroom_coarse`) now pass. The dual pane paints into the **single**
`#string` canvas, so the shipped probe was correctly targeted after all — worth establishing before
trusting its verdict, because a probe pointed at one half of a two-half pane would have passed for
the wrong reason. Both mountings differ in *pixels* rather than in styling (warm 62,676 vs 73,411,
background 44,006 vs 34,092 at the full duration).

Running them found what never running them had hidden:

- **THE DEEP LINK SILENTLY DROPPED EVERY PARAMETER EXCEPT `model` AND `domain`.** `applyUrlParams`
  read exactly those two, so batch 19's `?model=vkroom&audio_duration=0.02` — written to keep the
  cases cheap — was ignored, and both cases ran the full 0.12 s default: **~23 s each, not the ~7 s
  their own comment claimed**, and the short run they were written to exercise was never exercised.
  Nothing failed, which is exactly why it survived a batch: a deep link that ignores what it is
  handed reads precisely like one that honoured it. Fixed as `applyUrlSliders()`, which must run
  **after** `applyModelRanges()` (that function rewrites `val` per model and would overwrite the
  link), takes values in the **backend's** units so `data-scale` is undone, and records every
  unknown name and every clamp or snap in `window.__urlParamNotes`. **That list is now part of the
  pass/fail verdict** — for the cases that actually pass a parameter, which is the three `vkroom`
  ones; the two `airbox` cases pass none, so their empty list is silence rather than evidence, and
  the guard is not claimed to cover them. Proof it took: 961 audio samples against 5,761, and
  `plate_N = 8` reaching the payload from the URL.
- **The left-hand rig panel does NOT go stale on it**, which was the live hazard in making the fix —
  `setSlider` deliberately does not fire `onControlChange`, so a deep-linked value could have
  rendered 1,158 steps while the panel describing the rig still said 6,948. It does not, because
  `updateLambdaHint()` already runs after the new call in the boot sequence. Checked in pixels, not
  reasoned about: the panel reads **1,158 steps** and `audio length 0.02 s`.
- **The script attaches to an already-listening Chrome instead of insisting on launching one.** This
  inverts b18's scar — there a leftover Chrome held the port and the script *could not run at all* —
  into the supported way to drive it. It also gives a way through when spawning Chrome from the
  calling process does not work: measured here, three `Popen` variants (plain, `DETACHED_PROCESS`,
  `DETACHED|NEW_PROCESS_GROUP`) all had the launcher **exit 0 with the port never opening**, while
  the identical command line from a shell came up in 3.6 s. **The cause is not established and is
  not claimed to be a property of the machine** — the `airload` screenshot dated 2026-08-10 is proof
  the launch path worked before. Use a **fresh** profile after editing `web/static/*`, or the
  attached browser may serve the cached old file and hand back a PASS that proves nothing.

**2. The plate pane's animation window STAYS where it is, because the drift is not in the picture.**
The batch flagged that the pane shows 4 ms of a 120 ms run — the strike, not the shape drift it is
named for. Measured before changing anything: the per-window **mean-square velocity map**, which is
the spatial pattern `sigma_shape` is a functional of, compared between window 1 and window 4 —

| | correlation w1↔w4 | total-variation distance |
|---|---|---|
| struck | **0.9583** | 0.1269 |
| linear twin | 1.0000 | 0.0035 |

The load-bearing pair is the **correlation**, which is well conditioned and says the whole thing:
the struck plate's pattern is measurably not the same pattern, the linear one's is the same to four
decimals, and the struck one is still **96 % unchanged**. *The ratio of the two TV distances is ~36×
and is deliberately not quoted as the finding — it divides by a number sitting on the numerical
floor of a nearly-identical pair, which is the same conditioning objection that keeps the spread
separation below from being called a correction.*
Widening the animation to the full run would have shown the eye nothing, while costing ~490 kB of
frames and **freezing the room half** — the slices are on the wavefront's clock (87 frames) and
would sit on their last frame for 98 % of the plate's animation. The 46 % modal-share drift is
energy moving *between modes with near-identical mean-square footprints*, which is why the number is
large and the map is not. **The claim panel is the drift's right home, and that is now a measurement
rather than a default.** *A window-4-minus-window-1 difference map would render it — measured and
declined, because it is a new panel rather than a tidy.*

**3. The claim is ALIVE at the coarsest legal plate — and the multiplier wanders on one more axis.**
`plate_N` was legal down to 8 and never measured there. At the shipped 0.12 s:

| `plate_N` | resolved | struck spread | twin | spread separation | **drift separation** |
|---|---|---|---|---|---|
| 8 | 19/81 | 4.968 % | 0.062 % | 80.6× | **63.3×** |
| 10 | 19/121 | 2.553 % | 0.144 % | 17.8× | **34.8×** |
| 12 | 19/169 | 1.995 % | 0.177 % | 11.2× | **94.1×** |
| 14 | 17/225 | 3.314 % | 0.105 % | 31.5× | **43.1×** |
| 16 | 17/289 | 7.045 % | 0.069 % | 102.0× | **65.5×** |

The `plate_N = 16` row reproduces the shipped 7.045 % and 102× exactly, which is what makes the
other four rows worth reading. **The drift separation clears the suite's 20× bar at every legal
setting**, so the range is safe as it stands and the "probably fine" the batch flagged is now
measured. The **spread** separation is non-monotone over **11–102×**: that *widens the range the
record quotes* rather than refuting its "never below ~17×" floor, because a ratio of two max/min
spreads of four near-equal numbers is worse conditioned than either of them — the record's own
1.053-vs-1.070 on a 0.03 % rig change. One more axis on which the separation is the claim and the
multiplier is refused. Pinned by `test_vkroom_the_claim_survives_the_COARSEST_legal_plate` and by a
`vkroom_coarse` browser case.

*Two observations left deliberately unfixed, both wider than this batch.* The second panel's text
element is never cleared between models, so a canvas-only panel (`vkroom`, `airbox`, `whirl` — none
of which write it) shows the previously selected model's text; that is viewer-wide behaviour, not a
batch 19 item. And `drawVKRoomClaim` draws its own text into the canvas by design, which is why the
harness reads `(none)` for it rather than a missing readout.

*What is below is the plan as written before the build, kept because two of its claims died in
the build and that record is the point.*

### Batch 19 (PLAN, as written before the build) — the gong in the room: radiation that CHANGES mid-strike

**The legality check first**, because this document forbids inferring a batch from itself. The
standing rule is "a new viewer batch needs a new model or a new capability first." Air-box **batch
6** (`docs/dev/air-box-vk-plate-plan.md`, HANDOFF §12H) supplies both halves of it in its own
closing line — *"**Still out:** a viewer batch, which now has both the new model and the new claim
its own rule requires."* The models are `RoomLoadedVKPlate` / `RoomSuspendedVKPlate` (the nonlinear
plate #6 radiating into the 3-D room, baffled and hung); the claim is quoted below. Batch 18 built
the 3-D field type, so no new capability is needed — this batch is a **composition** of b18's slice
set with the shipped `vk` plate view.

Model key **`vkroom`** (b13/b15/b17/b18's precedent: a different coupling class earns a key).
All-wrapper as always; `physsynth/core` is untouched.

**The claim, inherited and re-measured rather than re-derived:**

> **A loud plate's radiation is time-varying at fixed geometry, and a quiet one's is not.**

Every other radiator in this repo is linear in its excitation, so radiated fraction and pattern are
amplitude-**invariant** by construction. The von Kármán coupling is quadratic, so the *shape* of the
motion evolves during a single strike — and a surface radiates by the shape of its motion, not by
its net volume displacement (batch 3). No `R(ω)` in `radiation.py` can state this: a
scalar-per-frequency load has one pattern per frequency.

#### The gate, and it is the whole reason this batch was probed before it was planned

**Everything below is measured** (`M:\claud_projects\temp\vkroom-probe\`, probes 1–6), before a line
of the batch was written. The gate was expected to kill or reshape it, and it reshaped it.

- **The air grid cannot be coarsened along the CFL line, and the edge is *sharper* than batch 6
  recorded.** Batch 6 measured 72 Picard sweeps at 57.9 kHz and NaN at 33.0 kHz. Between those, at
  a fixed Courant fraction of 0.9: `h = 11.40 mm` (57.9 kHz) converges at 72 sweeps; **`h = 14 mm`
  (47.2 kHz) already fails** (sweep cap 120, `converged` False, no NaN — the quiet failure); and
  `h ≥ 17 mm` (38.8 kHz) is an **immediate NaN at step 0**. So the shipped `airbox` room's
  22.0 kHz cannot host this plate at any size, and `AIRBOX_H_*` cannot be reused.
- **DIED — "the Courant fraction is the affordability lever."** It is not, but *not* for the reason
  on record, and the distinction is this batch's first generalizable finding. Batch 6's warning is
  that **coarsening the room breaks the plate's fixed point**; that is a statement about coarsening
  **along the CFL line**, where `h` and `fs` are tied. Hold `fs` fixed and lower the Courant
  fraction instead — `h = c₀√3/(λ·fs)` grows with no rate change — and the fixed point is
  **untouched**: measured **identical** 72/5 sweeps and `converged` True at fractions 0.90, 0.60 and
  0.45, while the node count falls as `(L·λ)³` (148 877 → 42 875 → 17 576). The plate does not care
  about `h`; it cares about `k`. **What it breaks instead is the claim.** Batch 5's ≥5-air-cells-
  per-structural-wave floor is a statement about `h`, so the resolved mode count collapses
  **17 → 8 → 6 of 289**, and with it the resolved-band spread → **1.000×** at every fraction below
  0.9. The refusal to coarsen therefore **stands, for a different reason than the one it inherited**
  — and the two reasons are not interchangeable, because one is a stability limit and the other is
  a validity limit that leaves every ledger green.
- **What survives is shrinking the room at the fixed fine grid**, which batch 6 never needed to try:
  `L = 0.35 m` against its 0.60 m keeps the resolved band (17/289) *and* the resolved-band
  separation — **29.3×** against the reference rig's **30.6×** — at roughly a third of the cost.
  Below 0.35 m nothing more is bought: `L = 0.25 m` is **indistinguishable** from 0.35 m (418.8
  against 414.5 steps/s), because by then the **plate** is the floor, not the room.
- **The bar was never the air-box batch's 8.6 s.** Measured through `simulate_to_payload` itself:
  shipped `vk` defaults **15.1 s**, `vk` at its own reachable loud/long ceiling (free, `w/e = 3`,
  1.0 s) **52.7 s**, `airbox` defaults 7.1 s, `platebody` 7.6 s. So ~20–30 s is an ordinary render
  here, and this batch fits between the `vk` default and the `vk` ceiling.
- **THE COST NUMBERS ARE QUOTED AS RANGES, BECAUSE THIS MACHINE'S OWN NOISE FLOOR IS 1.88×.**
  Probe 5's duration sweep had to be **thrown away and re-run**: it timed 0.060 s of audio at 37.4 s
  and 0.300 s at 42.2 s, and the *same* configuration at 26.6 s in one section and 47.9 s in
  another. Re-measured as a **rate** with an unchanged reference interleaved between every case
  ([[ci-runner-variance]]'s rule, arriving locally for the first time), the reference alone spans
  **220.6 → 414.5 steps/s = 1.88×**. Only two parameter effects clear that floor: the room at 0.60 m
  is ~2.8× slower than at 0.35 m, and the quiet arm is ~2.6× faster than the loud one (mean sweeps
  4.0 against 13.0). Tier (baffled vs suspended) and audio duration do **not** clear it, which is
  itself the useful half — cost is linear in duration, as it should be. *Generalizable: this repo's
  CI-variance discipline is not a CI fact. The local machine is worse than the runners.*

#### The shipping rig, and what it costs

Free-edge von Kármán plate (the honest cymbal — free edge, hung in air), `L = 0.10 m`, `N = 16`,
steel sheet `e = 1 mm`; room a **0.35 m cube** at `h = 11.40 mm`, lossy walls `ζ = 4`;
**`fs = 57.9 kHz`, and the ROOM sets it** — b18's rule again, one model further. `couple_max_iter`
must be **120**, not the viewer's shipped `VK_COUPLE_MAX_ITER = 50`, which the loud arm caps out.
Default audio **0.12 s** — batch 6's own observation window, and *the observation window is part of
the claim*, so it is named rather than defaulted.

Measured at that rig, 0.12 s, both tiers, quiet (`w/e = 0.05`) against loud (`w/e = 3`):

| tier | w/e | sweeps | converged | modal drift | spread (all) | spread (resolved) |
|---|---|---|---|---|---|---|
| baffled | 0.05 | 5 | True | 0.0291 | 1.013× | 1.002× |
| baffled | 3.00 | 72 | True | 0.5892 | 1.428× | 1.068× |
| suspended | 0.05 | 5 | True | 0.0090 | 1.004× | 1.001× |
| suspended | 3.00 | 72 | True | 0.3358 | 1.224× | 1.053× |

→ **resolved band: baffled quiet 0.2 % against loud 6.8 % = 29.3×; suspended 0.1 % against 5.3 % =
65.0×.** Batch 5's doctrine carries: the **separation** is the claim and it survives the resolution
restriction; the all-modes multiplier does not travel (it reads 38.4 / 42.8 / 91.1 / 14.0 / 191.9 %
across the five probed rigs — junk from unresolved modes — while the resolved band reads a steady
6.8–7.1 %). Only the separation is claimed. Render cost ~17–27 s loud, ~8–13 s quiet.

**The resolved band earns its privilege twice, and the second leg is free.** Rig-invariance is
above; **duration-invariance** is the other. Over a 5× range of observation window (0.06 / 0.12 /
0.20 / 0.30 s) the resolved-band spread reads **1.044 / 1.053 / 1.050 / 1.053** — flat — while the
all-modes spread **falls monotonically** 1.395 → 1.224 → 1.155 → 1.103. An observable that depends
on how long you watch is not a property of the plate; the restricted one is.

#### The shipped boundary is the FREE cymbal, and the supported gong is REFUSED with numbers

Every claim above is measured on `boundary="free"`. The supported gong **runs** at this rig (72
sweeps, `converged` True, 13.9–14.3 mean — probe 7), so the refusal is not a stability one. It is
that **the claim is not measurable on it here**: the supported plate resolves only **6 of 225**
modes against the free plate's 17 of 289, and on that band the separation collapses to **3.8×**
baffled and, suspended, runs **backwards** — quiet 1.71 % against loud 0.44 %, a ratio of **0.3×**.
A six-mode band is noise, and shipping it under this batch's headline would be captioning one rig's
number with another's. So `domain` selects the **tier** (`baffled` / `suspended`) — which is the
axis the comparison claim needs — and the boundary is fixed to `free`. The gong is reachable in the
core and is a later batch's business, at a rig sized for it.

#### The two amplitude sliders sit on a CLIFF, and it is not the one knob it looks like

The defaults are on an edge, not in an interior, and both edges are **immediate**: at width 0.20,
`w/e = 3.0` runs at 72 sweeps, **3.2 runs at 109**, and **3.4 is dead** (`converged` False at step
0, NaN one step later). At `w/e = 3.0`, width **0.19 runs at 109** and **0.18 is dead**. Same
signature, same failure — so `VK_WOVERE_MAX = 6.0` is unusable here and strike width cannot be a
free slider.

**DIED — "the two sliders are one knob."** The identical signatures, plus the fact that the von
Kármán coupling is a functional of *stretching*, suggested a single guard on the strike's curvature
`w/e ÷ width²`: both brackets put the cliff at 83–85, from opposite directions. Tested **off the
diagonal** — same ratio, very different amplitudes — it fails outright. At a constant ratio of
~75.3, `w/e` = 1.0 and 2.0 run (23 and 40 sweeps) while 5.0 and 8.0 are dead; at ~85.5, identically
so. **Amplitude is the binding variable and a broader strike does not buy it back**, which sharpens
batch 6's "a narrow strike is what fails to converge": narrow hurts, but broad does not rescue.
Shipped: `w_over_e` capped at **3.0** with the cliff at 3.4 recorded beside it, and strike width
**fixed at 0.20**, not exposed.

#### What the payload does when it does not converge — the two failure modes are different

Probe 1 and probe 8 measured both, and they want different handling:

- **`converged` False with finite output** (`w/e = 3.4`, width 0.18, `h = 14 mm`): the run produced
  numbers, and they are wrong in a way no ledger shows. This **renders**, with the per-step
  convergence record carried as a first-class honesty line — `converged` over all steps, max and
  mean sweeps — and a plain warning that the result is not trustworthy.
- **NaN** (`w/e ≥ 3.6`, width ≤ 0.17, `h ≥ 17 mm`): caught in the wrapper and returned as a **clean
  error payload**. It must never reach `json.dumps(..., allow_nan=False)` — b18's transport scar,
  where a valid-looking payload died as a browser 500.

The slider caps above put both outside reach; the handling exists because a cap is a claim about
where the cliff is, and the cliff was measured on one rig.

#### The control is a SLIDER, and it is the batch's own anchor

`VKPlate(nonlinear=False)` is bit-identical to `Plate`, so this class with the flag off is
bit-identical to `RoomLoadedPlate` — "frozen versus drifting is one class and one flag, not two
rigs." Shipped as a user-dialable toggle (b17's lesson: the anchor being reachable on the sliders is
what makes it evidence rather than a test fixture), and the render computes the **linear twin of the
user's own settings** alongside the main arm so the claim is visible in one picture. The twin is
cheap — sweeps collapse to 1 — measured **~13 s**, for a default total of ~30–35 s.

**Measured, and it is the cleanest number in the batch:** at `w/e = 3` with the flag **off**, the
spread is **1.013×** baffled and **1.004×** suspended — *identically the quiet arm's values to three
decimals*. So the effect is the **nonlinearity, not the amplitude**, and the control says so without
needing a second amplitude. This is also the only catch for batch 6's silent-failure trap (`rho_v`
for `rho_s`: the air load 1000× too weak at `e = 1 mm`, with every ledger still green).

#### Both ledgers are green everywhere, INCLUDING where the claim is dead — so neither is the gate

Scene drift **3.2e-13 … 7.3e-13** and the cross-ledger money test `|radiated − injected|`
**2.2e-16 … 1.7e-14**, at every rig probed — *including the coarsened ones whose resolved-band
claim had collapsed to 1.000×*. That is the air-box family's standing rule ("no single detector is
sufficient") arriving with a **fifth** distinct blind spot, and batch 6's own third reason underneath
it: the money test is arithmetic on whatever `w^{n+1}` came out of the solve, so an under-converged
one is ported self-consistently. **The honesty line for this batch is therefore the per-step
convergence record — `converged` (all steps) and the max/mean sweep count — carried in the payload
as a first-class number beside the two ledgers, never as a test-only assertion.**

#### Refusals, inherited and asserted

- **No compact/monopole far-field read-out.** Batch 6 measured it at **3e-7** of the truth and
  moving the **wrong way** for the suspended cymbal (rising while the true efficiency falls). The
  bridge batch refused `pressure()` and asserted the absence with a test; same here.
- **No `t50`, no decay time** — a magnitude, and at this room size the room contaminates it (b18).
- **No directivity panel.** Batch 6 refused it on a costed contradiction (a 120 ms window needs a
  41 m room), and b18 had already refused it as audio-bearing.
- **No string drive.** The three-way-chain batch measured this exact headline **flat** under a
  string — 0.27 % against a 0.28 % control — because `w/e` is not an amplitude under a point force.
  The plate must be **struck**. Say so in the panel text rather than leaving the absence unexplained.
- **Identify modes by projection under the mass matrix, never by an FFT peak** (batch 6: at
  `w/e = 3` a peak tracker reads a mode at 0.53× its own linear value, because the field has gone
  broadband and "the" frequency has stopped existing).

#### Pre-flagged traps

- **The animation clock is neither the string's nor the wavefront's.** b18's slip was inheriting the
  string's stride for a wavefront. Here the subject is **shape drift over ~0.12 s** — a slow panel,
  not a per-period wiggle — while the room slices still want the wavefront's clock. Two clocks, both
  chosen deliberately and printed.
- **The audio is a 120 ms burst, not a note.** State it. This is the shortest audio-bearing render
  in the viewer, and the reason is the gate above, not an oversight.
- **The fieldset beats the slider** — add `vkroom` to every enclosing `<fieldset>`'s `data-show`,
  not just the new sliders (live twice already: b15, b17, by opposite mechanisms).
- **`h` is snapped and the snap is the resolution** — report the actual `L`, never resample silently.
- **A NaN builds a perfect payload and dies at the transport** — test through
  `json.dumps(..., allow_nan=False)`, the way the server dumps (b18's scar).
- **The slice decode: `nv` is the fastest axis.** Reuse b18's path unchanged; do not re-derive it. A
  transposed decode renders as plausible banding, and no payload assertion can see it.
- **The backend does not hot-reload**; restart between edits, and use a session marker for stale-tab
  detection.
- Neither existing work cap covers this pairing (`VK_WORK_MAX` is `n_live × steps × sweeps`,
  `AIRBOX_WORK_MAX` is `nodes × steps`, and at this rig **both pass comfortably** while the render
  still takes ~25 s). A **two-term** budget is needed, and the plate term is the one that binds:
  measured ~0.16 ms per Picard sweep against ~2e-8 s per node-step. **Price the plate term at
  `couple_max_iter`, not at the measured mean**, the way `VK_WORK_MAX` already does — the mean is 13
  and the cap is 120, so a configuration that is legal on the mean can be 9× past the 52.7 s
  ceiling this batch cites as its bar, and the mean itself carries a 1.88× noise floor.

#### Touch list

`web/serialize.py` (constants + `_build_payload_vkroom` + the dispatch at ~857) ·
`web/static/index.html` (option, sliders, **and every enclosing fieldset**) · `web/static/app.js`
(`MODEL_RANGES` + `_default`, the viz dispatch, the energy/diagnostics branches, the dual
plate-field + room-slice view) · `tests/test_web_backend.py` · `scripts/verify_web_headless.py`.

### Later batches (rough map — not firm)

- **Body / radiation** — the modal body + radiation read-out is **batch 12**; `StringPlateBridge`
  (plate #5/#5b as a *distributed* body — the cymbal you watch ring) is **batch 13**; the radiation
  **load** / back-reaction (`RadiatedBody`, the booked `∫P_rad` channel) is **batch 15** (above),
  which closes the family. Beyond it the core itself would have to grow: frequency-dependent
  `R(ω)`, or a distributed/3-D air box (HANDOFF §12H). *Both since grew — and the first of them is
  **batch 17** (above): `RationalAirLoad`/`ReactiveRadiatedBody` as `airload`. The air box is still
  unsurfaced and is the next candidate, not a resumption of this map.*
- **Wind** — the reed is **batch 10** (above); the wind leg closes with it.
- **Excited strings** — the jawari landed in batch 8 above; the bow in batch 2; **fret buzz / the flat
  rail is batch 11** (above); the tanpura **cotton thread (juari)** is **batch 14** (above). The
  barrier family is then fully surfaced.
- **Sympathetics Weinreich two-stage decay** — DONE, batch 7 above.
- **The parametric-instability demo** — **batch 16** (above), planned and probed. It landed as a
  *regime of tension* rather than the standalone demo sketched here, and with real viz (energy
  cascading into the neighbour modes — model #9's IN-plane exchange, the SAME `2ω` pump batch 3's
  whirl aims at the other polarization); *not* a bolt-on to justify batch 1's purity gate.

**This map is now EMPTY — every bullet above resolves to a shipped batch, so Phase D's model list
is CLOSED** (verified 2026-07-27). There is no next batch to look up: every model the core had
built is surfaced in the viewer, and the built-vs-shown gap Phase D existed to close is gone. What
remains is a fork for the human, not a lookup — the plan-endorsed candidates are the two *core*
growths named in the Body/radiation bullet above (frequency-dependent `R(ω)`, or a distributed/3-D
air box, HANDOFF §12H — *both since taken: `R(ω)` 2026-07-27, the air box in four batches*),
HANDOFF §11's two never-closed decisions (#3 polyphony — a gate on engine design, cheaper to settle
before a port; #5 tolerance tuning — *both CLOSED 2026-08-10, see §11.3a/§6.1*), Phase 5 (the
real-time C++/Rust port,
HANDOFF §9's literal next phase), or a §12 high-leverage thread (port-Hamiltonian coupling,
differentiable models, modal synthesis from measured IRs/FEM). **Don't infer a batch 17 from this
document — if a new viewer batch is wanted it needs a new model or a new capability first.**

*Amended 2026-08-10: that condition was MET, not bypassed. The `R(ω)` fork arm delivered a new core
model class, and **batch 17 (above) surfaces it** — so the rule stands and this is what satisfying
it looks like. The same rule still governs what comes after: the 3-D air box is the one remaining
built-but-unshown core family, and it needs a **new frontend capability** (a 3-D field type) before
it is a batch, which is exactly the bar this sentence sets.*

*Amended 2026-08-17: the gap **reopened**, which is the useful thing this line now records. Air-box
batches 5 and 6 added core models after b18 shipped, and batch 6 closed itself with "still out: a
viewer batch" — so "the built-but-unshown gap closes" is a statement about a moment, not a
steady state, and the rule that governs it earns its keep by being re-applied rather than retired.
**Batch 19 (above) closes it a third time.** The fork's arms are again Phase 5 (the real-time port)
and a §12 thread — with the standing caveat that any new core model reopens this one.*

*Amended 2026-08-10 (second time, same day): **batch 18 (above) is that capability**, and it clears
the bar the sentence directly above sets rather than dodging it — the 3-D field type is built as
part of the batch, and it ships as a **slice set**, not as volumetric rendering, for a measured
payload reason. With it the built-but-unshown gap closes a second time and the fork's arms reduce
to **Phase 5 (the real-time port)** and **a §12 thread**.*

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
