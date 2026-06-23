---
name: web-viewer-state
description: "Phase-3.5 interactive web viewer (architecture B, local backend + browser) COMPLETE — string Phase A + membrane Phase B both built, tested (197), browser-verified; payload-contract + membrane decisions from advisor catches; next = model #5 plate"
metadata: 
  node_type: memory
  type: project
  originSessionId: 45e7cf82-678c-4fd9-8f6c-ec5838432243
---

Interactive web viewer started after model #4 (the human picked it over model #5 plate; see
[[membrane-state]]). **Architecture B** (chosen with the human): a *local Python backend + browser
frontend* — move a control → backend recomputes **offline** (seconds) → streams displacement field +
audio + energy → browser animates/plays/diagnoses. NOT WASM/live-in-browser (that is the deferred
Phase-5 real-time port) and NOT precomputed-static (architecture A). The validated Python core stays
the single source of truth.

**Phase A = string family (ideal/stiff/damped), built, passing & browser-verified.** 20 web tests
added (total now 180). A headless-Chrome render of the live page closed the advisor's one acceptance
gap (frontend never run in a browser): sliders build, `/simulate` round-trips, the base64→Float32
field decodes to the *correct* pluck shape (not byte-order garbage), energy panel shows `drift
5.51e-14 → PASS`, partials `worst 0.002 cents`. A durable test (`test_frames_decode_to_field_values_
and_boundary`) now pins the decoded *values* against `field_amp` + fixed-boundary endpoints — a
length-only check can't catch endianness garbage (right size, wrong values).
Files (all *wrappers*, core untouched — `physsynth/core` never imports these):
- `web/serialize.py` — pure `simulate_to_payload(params) -> dict`, the socket-free **testable seam**
  (the server is a thin shell over it). Per-model ctor + partials-oracle dispatch.
- `web/server.py` — `ThreadingHTTPServer` (stdlib, no Flask): `POST /simulate`, `GET /` static.
- `web/static/{index.html,app.js,style.css}` — vanilla JS + Canvas2D, no framework/build.
- `tests/test_web_backend.py` — drives the pure function (drift survives wrapper <1e-10; frame/audio
  bookkeeping; loss-gated energy; clean error payloads).

**Payload-contract decisions (from advisor review 2 — these are the load-bearing "why"s):**
- **catch #1** audio resampled to a *fixed 48 kHz* (`fs_sim = c·N/(L·λ)` rides N and c and can blow
  past the browser `AudioBuffer` cap ~96 kHz). `audio.fs` always = 48000.
- **catch #2** animation is *decoupled from audio length*: a short `animation_window` captured at a
  stride that resolves the **fundamental** (~12 frames/period), played in slow-mo with a
  `playback_speed` control. A wall-clock 60 fps stride is longer than one oscillation → aliases the
  wiggle into beating. The sound plays full-length at 1× separately.
- **catch #4** energy report **gated by σ**: σ=0 → drift vs the 1e-10 bar; σ>0 → passivity (monotone
  + measured 2σ vs base oracle), because `energy_drift` is huge-by-design for a decaying run.
- **catch #5** the three string models do NOT share a uniform ctor: ideal=`sigma`,boundary=fixed;
  stiff=`kappa,sigma,theta`,supported; damped=`kappa,sigma0,sigma1,theta`,supported. Oracle differs
  too (harmonic vs stiff_harmonic + per-mode decay). Frontend slider panel is model-dependent.

**Phase B = 2D membrane heatmap, built, passing & browser-verified (advisor review 3).** Split as
`_build_payload_membrane` so the string path stays bit-for-bit (shared `_energy_block` /
`_resample_normalize` / `_b64f32`). 17 web tests added (197 total). Browser-verified energy drift
survives in 2D: circle `5.07e-15`, rect `3.93e-15`, both PASS (conservation ⊥ geometry — same as the
core; see [[membrane-state]]). Load-bearing Phase-B decisions:
- **Modal panel = a mode *spectrum*, NOT per-partial cents bars.** FFT magnitude + vertical markers at
  the **discrete** eigenfreqs (`eigsh(-L)`→`discrete_membrane_eigenfrequency` — where the stepper
  actually rings; peaks landing on them = self-consistency) + fainter continuum (Bessel/rect) markers
  (the geometry tier, *shown not scored*). Two headlines: fundamental detected-vs-discrete cents
  (robust — (0,1) always excited; `~0.002 c`) and discrete-vs-continuum "geometry tier (O(h)
  staircase)" gap (circle `-13.94 c` at N=80 vs rect `-0.05 c` — O(h²) coincides). Per-mode bars are a
  trap: a struck drum rings odd/high modes weakly → `measure_partials_near` locks onto noise; and the
  Bessel gap is ~9 c by design. **Energy stays the hard pass/fail.**
- **2D data-size trap handled:** frames `{nx,ny,dims:2}` base64 f32, spatially decimated to a
  ≤64 *display* grid (`stride=ceil(max(ny,nx)/64)`, **same stride on field AND mask** so they align),
  temporally at the fundamental stride; `field_amp` + colour scale from the *decimated* frames shipped
  (not the full field), fixed symmetric `[−amp,+amp]`. Mask shipped once (base64 uint8) to blank the
  exterior. Heatmap = Canvas2D ImageData (cool/warm diverging) blitted at physical aspect (snapped
  `Ly` read off the ctor). (x,y)-fraction strike (`raised_cosine_2d`) + pickup (`pickup_index_at`).
- **Cost guards — bound the *actual problem size*, not N (advisor review 4; first cut clamped N only
  and a thin rect still hung ~45s).** Profiled: cost is pure FDTD (`eigsh`≈0.06s, NOT the
  bottleneck), two drivers: (1) per-step ∝ n_live with a **razor-sharp ~3.2× cache cliff at
  n_live≈10_000** (87→281 µs/step crossing L2) — a thin rect's `n_live~N²·(Ly/Lx)` hits it (Lx.3
  Ly2 N100 = 66k nodes) → `MEMBRANE_NLIVE_MAX=9_900` keeps n_live below the cliff; (2) steps ∝
  `fs=c/(λh) ∝ 1/min_dim` (small drum inflates steps; audio resamples to 48k regardless) →
  `MEMBRANE_WORK_MAX=7e8` caps n_live×total_steps (audio+anim). Plus N≤100, audio≤2s, λ≤1/√2,
  `eigsh k=min(12,n_live-1)`. Worst *passing* render ≈5.6s, default (N=80)≈2.7s, work-budget scales
  with duration. The string's `N_MAX=2000` would OOM in 2D. **The earlier "eigsh LU cliff at N≈128"
  story was wrong — it's an FDTD L2-cache cliff at n_live≈10k.**
- Frontend gained: `model=membrane` + `domain` select, per-model slider ranges (`applyModelRanges`),
  `data-domain` geometry gating, `?model=&domain=` deep-link (also drives the headless verifier),
  `drawHeatmap`/`drawSpectrum` dispatched on `frames.dims`.

**Why:** records the architecture fork the human resolved and the non-obvious contract choices, so a
later session doesn't re-litigate B-vs-A-vs-WASM, re-derive the audio-rate / animation-decoupling
traps, or re-discover the membrane spectrum-panel / N-cliff decisions. Full plan:
`docs/dev/web-viewer-plan.md`. Headless smoke harness: `scripts/verify_web_headless.py` (CDP over
`websocket-client`, needs Chrome + live server; not in pytest).

**How to apply:** the web viewer (Phase 3.5) is **complete**. Next physics-ladder item is **model #5,
the Kirchhoff plate** (biharmonic, Chladni patterns) — the resonator+harness first (CLAUDE.md #4: viz
depends on a validated core), *then* it slots into this same viewer as another heatmap model behind
the existing `_build_payload_membrane`-style dispatch. θ-artifact + portability-test loose ends from
[[stiff-string-state]] still untouched (don't gate the plate).
