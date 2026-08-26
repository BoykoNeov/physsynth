---
name: web-viewer-state
description: "Phase-3.5 interactive web viewer (architecture B, local backend + browser) — Phases A (strings) + B (membrane) + C (plates #5/#5b + von Kármán #6) + D batch 1 (tension #9) + batch 2 (bow + the energy-BALANCE panel, the 3rd verdict type) + batch 3 (geometric #10 + the ORBIT viz, the first VIZ-ONLY model) + batch 4 (phantom partials = #10's bridge force) + batch 5 (mallet #7 = the first CONTACT model; rides the drift panel via the decay_oracle=False catch; the CONTACT is the headline not the tone) + batch 6 (sympathetic/coupled strings = the first MULTI-STRING model + drawFields' 2nd customer; the CLAIM is the bridge trace w_b NOT energy; normal runs BOTH ICs = bit-exact w_b=0 vs symmetric; transfer = 88% slosh; audio = string pickup NOT body pressure) + batch 7 (Weinreich two-stage decay = the FIRST body-loss slider; sigma_body GATES the verdict: drift at 0 / passivity+decay_oracle=False at >0; anchor = detune-0 antisym tail bit-exactly LOSSLESS; strike-one plateau vs strike-both decay-away contrast; sliding-MEAN envelope kills the slosh ripple) + batch 8 (jawari buzzing bridge = the FIRST BARRIER model, a CONFIGURATION of #8 with zero core edits; claim = the SHIMMER not energy; decay_oracle stays TRUE against the advisor's template — a lossless elastic barrier dissipates nothing; the OFF-GRID SNAP made the tested config unreachable from the UI) + batch 9 (acoustic bore + radiating bell = the FIRST WIND model and a new field type, PRESSURE along a tube; the first lossy model that KEEPS the conservation verdict because the bell's loss is BOOKED — so the verdict rides on the TOTAL and _energy_block grew a `split`; the headline test must assert the split MOVES or it passes on a dead bell; the analytic curve must span the SLIDER range not the interesting part; the glow tracks the RATE not the cumulative; pace the animation on the TRANSIT not f₁) + batch 10 (the dynamic REED = the wind leg's exciter, closing it; the balance is a GENUINE residual not the bow's tautology — measured channels, so dropping one moves it 1.6e-14→0.95; NOT σ-gated, since jet+lip loss are on in every regime; no convergence gate because the balance is LINEAR in the Newton residual; sweeps pinned at fixed N off the render slider AND memoized on everything that moves p_closing; the threshold is BELL-dependent; two bugs only the RENDERED page could catch) + batch 11 (the FRET / flat rail = model #8's OTHER config, the jawari's OPPOSITE — slap-and-release not a travelling wrap; the money panel is a CONTACT RASTER, the first x-vs-t picture-of-history; the claim is INTERMITTENCY, measured 1.225 ep/period at 15.45% duty, printed from the payload NEVER read off the raster which OVER-states duty 28.8% vs 15.5%; |𝒞|max 69/99 = the VECTOR Newton, binning x is fatal to it; decay_oracle=False + a diagnostic TRIPLE because the flat rail BREAKS equipartition 6–9% where the jawari's gentle wrap didn't — re-measure oracles per config; brightness NON-monotone in clearance so no monotone label survives its own slider; crossing_rate NEVER called pitch; the first contact model needing NO zoom pane; the `kind` on the signature block was the one backend reopen; the energy TRAP that a "does it draw?" check passes; the render-button-disabled no-op that made the transition verifier lie) + batch 12 (the BODY speaks = the FIRST coupled-resonator + far-field model, the whole exciter→resonator→BODY/radiation third stage; the money panel is the ENERGY card itself carrying the E_string⇄E_body slosh AND the σ-verdict at once (bore-split precedent); E_conn GOES NEGATIVE so never clamp it + the flat total_frac is a 100% REFERENCE not the conservation verdict; bridge_stiffness/sigma_body are SHARED-BY-NAME single elements re-ranged per model + reset in _default, the collision the plan flagged; data-hide-domain gates sigma_body for weinreich-AND-body without the risky updateVisibility rewrite the advisor flagged; the fresh-load verifier can't see the jawari→body leak so a CDP switch-check drives it; radiated spectrum == body spectrum for one source so no fake 2nd analysis, ω²~1 is SANITY not oracle, body modes BOOST not formant) + batch 13 (distributed PLATE body, the DUAL FIELD view) + batch 14 (tanpura cotton thread, the TUNING CURVE) + batch 15 (the radiation LOAD `RadiatedBody` = the air pushes BACK and the energy it takes is BOOKED, so a run that hands 98.4% of the pluck to the air still CONSERVES — sigma_zero stays (sigma_body==0) regardless of R; the 2nd panel is a t50-vs-R map with a real OPTIMUM, and the discriminator that it is physics not the timestep is that the scheme's own 1/G turnover moves 3.2x with N while the measured curve does not move at all; t50 NOT "fraction radiated by T" whose peak walks with the window; R=0 is bit-identical to batch 12; a param with a server-side DEFAULT hides its own missing slider) + batch 16 (the PARAMETRIC instability = model #9's SECOND refusal and the LAST batch in the plan's own backlog, so Phase D's model list CLOSES; a REGIME of tension, not a model key; the driven mode m HAD to become a control because m=1 is the ONE mode that cannot do this; the PARTNER MOVES with dT/T0, so the cascade is read off the run and never named in advance; TWO sweep observables measured and REJECTED before the shipped one — the fitted growth rate (no single exponential exists: overlapping tongues, saturates in ~5 periods) and t_break (the near-edge points are BOUNDED not slow, so a fixed level is an arbitrary line through a continuum of plateaus) — SHIPPED the saturated LEVEL, window-free because it plateaus; the EDGE is NOT N-invariant and the panel says so (real Mathieu tongue structure, edges on both sides in detuning) while the above-edge curve is; 'below threshold' is NOT a linear control, which is the stronger claim; drift stays 1e-15..1e-12 THROUGH complete disintegration so no new verdict type, gated on the Picard counter because a failed root-find looks exactly like a breakup; the leftover-Chrome trap RECURRED and batch 15's recorded fix structurally cannot catch it — that fix is a VERSION marker and a leftover of the SAME version sails through, so assert a SESSION marker; rounding a log-axis series in the PAYLOAD deletes the claim; and THE BACKEND DOES NOT HOT-RELOAD, so a 'regenerated' screenshot can be a PRE-FIX render) — both cross-cutting capabilities built, so the rest is wire-up; payload-contract / membrane / plate-VK / tension / bow / geometric / mallet decisions from advisor catches; Phase D = CONSOLIDATION of the built-but-unshown models, grouped by frontend need; **PHASE D IS NOW CLOSED — the plan's 'Later batches' backlog is EMPTY (verified 2026-07-27), every bullet resolves to a shipped batch, so there is NO batch 17 to look up and none may be inferred from the document; what remains is a human fork (core R(ω)/3-D air box, HANDOFF §11 #3 polyphony + #5 tolerances, Phase 5 real-time port, or a §12 high-leverage thread), baseline 1129 tests green** + batch 17 `airload` (2026-08-10, THE RULE WAS SATISFIED NOT BYPASSED — the R(ω) fork arm shipped a new core model class, so a batch 17 now exists; the air as an IMPEDANCE, a FIFTH booked channel that STORES; **the money test is the LEDGER RESIDUAL not the drift** because load.energy() is stored PLUS radiated while body.energy() is bare — 2.3e-16 vs 3.6e-01 sign-flipped, drift green either way; the anchor is a DIALABLE slider setting reaching batch 15 bit-for-bit; the RADIATION WEIGHT not R decides visibility and at a physical one batch 15's slosh-vs-drain anti-correlation does NOT bind; the pitch shift must be read against the SCHEME's unloaded frequency or the sign FLIPS; the oracle's error IS its own stated 2nd order, coefficient ~16-21; R=0 RAISED because the corner's inversion gives a zero mass; and batch 15's INVISIBLE-SLIDER bug was still live — fieldset data-show beats slider data-show) + **batch 18 `airbox` (2026-08-10) — the 3-D ROOM, the LAST built-but-unshown core family, and the viewer's first `dims:3`; details in [[airbox-viewer-state]]**"
metadata: 
  node_type: memory
  type: project
  originSessionId: 45e7cf82-678c-4fd9-8f6c-ec5838432243
  modified: 2026-07-20T15:48:32.208Z
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

**Phase C = both Kirchhoff plates (#5 supported, #5b free) + the von Kármán nonlinear plate (#6
supported gong + free cymbal) wired as heatmap models, built, passing & browser-verified 7/7.** 30
web tests added (39→69 web; 392 total). Split as `_build_payload_plate` / `_build_payload_vk` behind
the `_build_payload` dispatch; the tested string/membrane paths stayed bit-for-bit (shared helpers
`_decimate_field_mask` / `_pooled_spectrum` / `_modal_spectrum_block` extracted, `_membrane_spectrum_
block` now a thin wrapper). Load-bearing decisions (advisor pre-build catches):
- **Secondary "domain" select is repurposed as the plate/VK *boundary*** (`supported`/`free`),
  populated per-model in JS (`DOMAIN_OPTS`, `populateDomain`); membrane keeps circle/rectangle. Plates
  are always a *rectangle* heatmap (`grid.domain="rectangle"`); the payload carries `boundary`.
- **Controls differ per model:** plate = the **plate Courant μ** (`fs=κ/(μh²)`, `h=Lx/N`; implicit →
  no CFL, large μ coarse-but-stable). Cost explodes at **LOW μ** (high fs → step blow-up) — the
  *opposite* of the membrane's λ cap, so the work-budget guard message says "**RAISE mu**". VK = **fs
  directly** (κ is derived, so no Courant knob; oversample the nonlinearity). VK material surface is
  `(E, e, ν, ρ_volumetric)` — ρ here is **volumetric** kg/m³ (VKPlate derives ρ_s=ρe), unlike the
  areal ρ everywhere else.
- **Shared eigen helper `_plate_discrete_eigenfreqs` duck-types on `.W`**, NOT `.w`/`.wdiag` (those
  names *diverge* Plate-vs-VKPlate — a landmine). Supported: `eigsh(-L)` → the **Laplacian** Λ →
  `discrete_plate_eigenfrequency` (which squares internally, Q=κ²Λ²; must NOT feed `B`'s Λ² or you
  double-square). Free: generalized `eigsh(K, M=W, sigma=<neg>)` → **drop exactly the 3 rigid modes**
  → `discrete_beam_eigenfrequency` (shift strictly negative; K only PSD). Continuum tier: SS =
  Navier `rectangular_plate_freqs` (tight ~1 cent, not the membrane's O(h) staircase); free = Leissa
  FFFF-square anchor **only when (near-)square** else empty (no closed form to mislabel off-square).
- **VK energy verdict is Picard-convergence-gated (the subtle catch):** the discrete energy identity
  telescopes *only at the fixed point*, so a non-converged step's drift is iteration noise, not
  physics. `_energy_block(convergence=…)` folds in `{all_converged, n_not_converged, worst_residual,
  max_iters, couple_tol}`; frontend `drawEnergy` reads it and overrides the badge to "NOT CONVERGED"
  **before both** the lossless and lossy branches. `nonlinear=False` sends no convergence block
  (linear regression path).
- **VK spectrum reads the hardening, not a cents error (`kind="vk"`).** Marker lines are the
  **linear (w→0)** modes; the real peaks ride **ABOVE** them by the amplitude hardening — the
  *opposite* of a linear model where peaks-on-lines = good. `drawVkSpectrum` shows a yellow detected-
  f₀ line + `shift_pct`. **The hardened f₀ is honest ONLY for supported + a (1,1)-eigenmode IC** —
  `_vk_strike` excites the supported gong with `mode11` (not a broad Gaussian), because zero-crossing
  spacing *overcounts* a multi-mode strike (a broad strike gave a bogus +451% even on the *linear*
  plate; mode11 gives 0/+6/+36/+75 % at w/e=0.01/1/3/5, matching `diagnose_vk_plate`'s sweep). The
  free cymbal is a positioned crash (mode wash) → f₀/shift reported as `null` ("crash cascade"), never
  a lying number. Browser-verified: supported gong f₁ 53.4 → hardened 63.9 Hz, drift 3.5e-13, Picard
  ≤5 sweeps; free cymbal converges ≤8 sweeps, drift 9.7e-13.
- **Frontend param-conflict handling:** ONE slider per backend param, **re-ranged per model on
  switch** via a generalized `applyModelRanges` (`MODEL_RANGES` full specs min/max/step/fixed/val).
  Two params carry different meaning per model and reset on switch: **κ** (string ~1 vs plate ~20)
  and **ρ** (areal 0.005 vs VK volumetric 7800). New `data-scale` attr lets E show in **GPa** / e in
  **mm** (slider raw × scale in `gatherParams`). Plate/VK default to a compact square (1.0 / 0.3 m) so
  the out-of-box gong is a sensible pitch. `theta` is wired through `_build_plate` too (was a shown-
  but-ignored slider). Browser verify (`scripts/verify_web_headless.py`) gained plate_supported/
  plate_free/vk_supported/vk_free cases; **run the verifier alone — a concurrent full-suite pytest
  starves the local server and a case can probe mid-"computing…" (false FAIL).**

**Why:** records the architecture fork the human resolved and the non-obvious contract choices, so a
later session doesn't re-litigate B-vs-A-vs-WASM, re-derive the audio-rate / animation-decoupling
traps, or re-discover the membrane spectrum-panel / N-cliff / plate-μ / VK-convergence-gate / VK-
hardening-vs-cents / mode11-zero-cross decisions. Full plan: `docs/dev/web-viewer-plan.md`. Headless
smoke harness: `scripts/verify_web_headless.py` (CDP over `websocket-client`, needs Chrome + live
server, `VIEWER_BASE` env to override the port; not in pytest).

**Phase D = CONSOLIDATION (the strategic fork the human picked over more physics).** Phases A–C
covered every model built *at the time*; the project then built ~9 more with no viewer at all
(tension #9, geometric #10, bow, reed, bore, mallet #7, barrier #8, jawari, sympathetics,
body/bridge, radiation) — the biggest built-vs-shown gap against the "interactive, beautifully
visualized" vision. Group batches by **what the frontend needs**, not model number; no hard
dependencies between them, so only the current batch is firm. Two capabilities cross-cut — build
each ONCE: **energy-BALANCE panel** (a *third* verdict type beside σ=0 drift and σ>0 monotone) for
**bow + reed** (mallet is *conservation* → rides the drift panel); **multi-field/orbit viz** for
**geometric #10 + sympathetics**.

**Phase D batch 1 = tension string #9 (DONE, 16 web tests → 85 web / 877 suite).** Full rationale in
`docs/dev/web-viewer-plan.md` §"Phase D". The load-bearing, all *measured* not assumed:
- **Headline = the amplitude SHIFT `ω(A) − ω(A→0)`, never an absolute frequency** (the θ-scheme's
  dispersion error cancels in the difference; `duffing_frequency` alone matches only "loosely").
- **The measurement pair is LOSSLESS + short and is NOT the audio run** (advisor's blocking catch):
  the oracle predicts ω at a *fixed* A, but a lossy tension string is a downward-gliding chirp, so
  zero-crossing the decaying pickup reports an amplitude-*averaged* freq that undershoots — the panel
  would diverge from the oracle as σ rose, reading as a bug that isn't one. Audio stays lossy (the
  glide IS the signature). Pinned by a **shift-is-bit-identical-at-σ=0-vs-σ=5** test.
- **IC = mode-1 sine** (KC→Duffing is a single-mode ansatz; a pluck is multi-mode ⇒ lying number —
  #6's `mode11` lesson) ⇒ no `pluck_position` slider. **Amplitude is the star control**: shift ∝ A²,
  so the string path's inherited `1e-3` renders a 0.8-cent near-null (panel looks broken); 0.02 → 270 c.
- **Bound `dT/T0 = EA·A²·p1²/(4T)` (exact, free), NOT amplitude** — A is a proxy; EA/T move it just
  as hard (`EA=2e5, A=0.06` breaks up with an amplitude-only cap none the wiser). The membrane's
  "bound the actual problem size" lesson.
- **The core's "~3" breakup threshold is NOT mode-invariant** (measured at mode 3). For **mode 1**
  over 1000 periods: `A=0.06`(dT/T0=4.44) PURE 3.2e-13; `A=0.07`(6.05) BREAKS ⇒ threshold ∈
  (4.44, 6.05]. κ=0 *looked* like the dangerous corner (stiffness detunes Mathieu) — measured, moves
  it nowhere.
- **A guard that can't trip in the reachable range is a GUARANTEE, not dead code** (advisor). Don't
  widen a range to make a gate fire; unit-test it *past* the guard where breakup is real+fast (N=64,
  A=0.20, 20 periods → off-mode 6.9e-2). Purity normalizes by the **fixed ‖u₀‖**, never instantaneous
  ‖u‖ (a mode passes through u≈0 twice a period → spurious 1.0).
- **This model needs its OWN cost budget**: every step is a tension root-find, ~176 µs at N=128 =
  **~2× a 2D membrane step for a 1D string**; the string path's `N_MAX=2000`/10 s = ~4M root-finds.
  `TENSION_N_MAX=256`, `TENSION_AUDIO_MAX=3`, `TENSION_WORK_MAX=60_000` steps.
- Measure length buys nothing past ~12 periods (6/12/24 → 1.47/1.17/1.03e-3 off the oracle ⇒ the
  residual is the scheme's real O(h²)+O(k²) gap, not crossing noise). But crossings must be
  **interpolated** (`_interp_zero_cross_frequency`): shared `_zero_cross_fundamental` quantizes to a
  sample and its mean telescopes to `(zc[-1]−zc[0])/(M−1)`, landing straight on the headline.
- `nonlinear_fraction` read **at the IC = the peak** (max displacement, zero velocity ⇒ all E
  potential); off the *final* state it reports wherever the run stopped (0.115 vs 8e-6 for 0.5 vs
  0.4 s — same physics). **Gotchas:** spectrum block must nest under `meta.spectrum` (top-level
  renders nothing); `drawDiagnostics` must test `kind==="tension"` **before** the `dims` gate (1-D
  model, spectrum panel); `amplitude` needed adding to `MODEL_RANGES._default` at 1e-3 because
  `gatherParams` sends *every* slider incl. hidden ones. Convergence branch generalized *additively*
  (`detail`/`note`; VK's Picard wording untouched) — tension's telemetry is `n_not_converged` /
  `tension_tol`, it has no `worst_residual`/`max_iters` and would throw on `undefined`.

**Phase D batch 2 = bowed string + the energy-BALANCE panel (DONE, 23 web tests → 109 web / 901
suite).** The human picked bow (infra-first) over mallet (quick win) and geometric #10 (showcase),
because each cross-cutting capability is built once and bow's makes the reed cheap. The
load-bearing:
- **The balance is a third verdict type because both older ones are ACTIVELY WRONG for a driven
  model, not merely weaker** — σ=0: the bow *pumps*, so drift is enormous by design → lossless
  branch FAILs; σ>0: energy *rises* from rest to the limit cycle → passivity branch FAILs. So the
  balance **replaces** both (`kind:"balance"`, dispatched before both). Test reconstructs the
  numbers the old branches *would* have reported. **The bow starts from REST ⇒ E₀=0 ⇒ a ratio drift
  isn't even defined** (`energy_drift` falls back to max|E|).
- **The lossy branch is deliberately NOT a residual** (the subtle one): dissipation is never
  measured, it's *inferred* as `work − ΔE`, so a lossy "balance residual" is **identically zero by
  construction** — a tautology/green tick that can't fail. Lossy verdict = core's criterion 2
  (inferred dissipation ≥0 + monotone). σ=0 verdict = residual vs 1e-11 → **2.2e-14**, same
  normalization as the core test (per-sample scale, max over steps) on the **FULL** arrays (a
  decimated max understates it).
- **NO convergence gate (unlike VK)** — the balance is exact for *any* Newton residual (force
  applied exactly, power from the true post-correction velocity). That's the model's whole trick.
- **Loss defaults ON — opposite of tension, load-bearing** (σ₀ settles the limit cycle, σ₁ keeps the
  corner clean vs ~18 slips). Each default hides something ⇒ each model gets its own nudge.
- **Slip oracle `slip_fraction == β` claimed ONLY in the Helmholtz window** (advisor's blocking
  catch): outside Schelleng's window (floor AND ceiling, both narrowing off the bridge) the note
  legitimately isn't Helmholtz — real physics ⇒ **label, never FAIL**. `regime` disambiguates the
  **zero-onset ambiguity** (never-sticks vs never-slips both give 0 stick→slip transitions) via
  slip_fraction. **Energy=STRUCTURAL, Helmholtz purity=DYNAMICAL** — balance passes on a raucous
  note (same split as tension #9).
- **A RATE NEEDS A LONG WINDOW** (my bug, caught by tests): `slips_per_period` = integer onsets /
  periods, so a 6-period tail quantizes to multiples of 1/6 → clean motion read **0.83 at N=64**
  (1.00 at every other N) purely from window-edge placement, straddling the 0.85 gate and
  mislabelling a clean note raucous. Tail = settled last **40 %** of the run (core's
  `_bow_to_steady` choice).
- **Animation needed NO new viz** — Helmholtz motion *is* the string's shape (1-D line path draws
  the corner free) — but must capture the **settled** window (from rest the first frames are flat)
  ⇒ **ONE run, frames captured out of the audio run** via a new `snapshot_from` arg. Every other
  model re-runs a 2nd resonator because its window is the *attack* at t=0; the bow's is the *tail*,
  which the audio run already passes through. My first cut re-ran and silently **doubled the cost**
  of a root-find-per-step model — and the work budget couldn't see it (1.13 s vs ~2×).
- Own cost budget (`BOW_N_MAX=256`, `BOW_AUDIO_MAX=3`, `BOW_WORK_MAX=60_000`) — friction root-find
  per step. Schelleng has **no closed form in the core** ⇒ hint reports `helmholtz_number` + the
  tests' empirical rule (`force≈4·v_bow`), never a fabricated Fmin/Fmax. `helmholtz_number` stays
  diagnostic-only.
- **`_default` re-range leak, round two:** bow wants σ₁=0.05 = 25× damped's default AND outside its
  own max 0.01 ⇒ without a `_default` reset a bow→damped switch renders a wildly over-damped string
  on a stale range. Added `sigma0`/`sigma1`/`pickup_position` (also fixes tension's pre-existing
  `sigma0=0` leak). **Every param a model re-ranges must reset in `_default`** — `gatherParams`
  sends every slider.

**Phase D batch 3 = geometrically-exact string #10 + the ORBIT viz (DONE, 24 web tests → 133 web /
925 suite).** Builds the 2nd cross-cutting capability ⟹ every remaining batch is now wire-up, not
infra. Browser-verified 3/3. See [[geometric-string-state]] for model #10 itself. Load-bearing:
- **THE VIEWER'S FIRST VIZ-ONLY MODEL, and it's physics not budget** (human chose "viz-only now,
  phantom-partials audio later" over a stub player). `c_long=√(EA/ρ)` ≈ **22× the c you hear**, so
  resolving it (= `lam_long≤1`, the model's whole point) forces `fs ≈ 22×` a normal string's ⟹ 1 s of
  audio ≈ **10 min**. Step cost ~2 ms (4 ms on the whirl) and **N-INDEPENDENT** (per-Newton-iteration
  Python/`splu` overhead, not grid) ⟹ only `fs` × duration is the cost. No cheat: `lam_long>1` IS the
  silent-garbage regime. `audio: null` + an `audio_note`; JS guards `data.audio` everywhere.
- **The orbit hero is the ROTATING WAVE, not the whirl — advisor framed it as the whirl and was
  WRONG; I measured.** At 0.06 s the whirl grows ~60× but only reaches `max|w|/max|u|` ≈ **0.076**,
  which on equal axes still reads as a line (confirmed in the rendered PNG). The rotating wave is an
  **exact solution of the scheme** ⟹ round from frame 1, no growth needed: roundness **1.2e-12**,
  `long_kin/E` **1.5e-29**, ~5 s. Advisor conceded.
- **SEED MAGNITUDE IS A CONVENTION TRAP — and it made me "correct" a memory that was RIGHT.**
  `diagnose_geometric_string.py`'s fig-3 seeds `w_dot = 1e-3·A·φ` **with no `ω_u` factor** ⟹ initial
  out-of-plane *displacement* ~`1e-3·A/Ω` = a **1000×** smaller start than a displacement seed of the
  same nominal `1e-3·A`. That — not the rate — is why THAT figure needs 0.22 s to saturate. Measuring
  with the script's seed I got `w/u = 8e-5` and wrongly "corrected" [[geometric-string-state]]'s
  "63× = ~9 % of u" as a units error. **It was right**: ~9 % is what a `1e-3` *relative* seed gives.
  The viewer follows the **test suite's** convention (`_whirl_run` `seed="vel"` → `dw' = s·A·ω_u·φ`)
  ⟹ both seed kinds start at comparable displacement (disp **7.6e-2**, vel **5.2e-2** at 0.06 s) ⟹
  switching kind changes the PHYSICS, not the size of the perturbation. **Generalizable: before
  calling a recorded number wrong, check you are running the same convention it was measured in.**
- **Whirl = log-y ENVELOPE + the drift gate, not an opening orbit.** Envelope = sliding ~1-period max
  (raw `max|w|` is a growing OSCILLATION — non-monotone `4.6e-8→1.8e-8→2.9e-7→1.5e-7→8.5e-7`; the
  family's "never read at one phase" trap again). Enveloped ⟹ monotone **399/399** = a clean line.
- **NO new verdict type (unlike the bow).** Nothing drives this string ⟹ the ordinary σ=0 drift check
  IS the claim; what makes it one is what it *survives* — drift **8.4e-13 THROUGH a 60× blow-up**
  (redistribution vs a diverging solve) ⟹ growth is printed *next to* the drift.
- **`frac = δ/(εA²)` is the control** (dimensionless, refinement-invariant ⟹ recompute `κ_w` from
  `p2` at the actual N). Amplitude bounded by **ΔT/T₀** not A (at κ_u=0, `εA²/ω_u² == ΔT/T₀` exactly
  ⟹ it's #9's own breakup coordinate). Rate oracle = `_mathieu_rate`, **Tier C, reported never
  scored** (5–11 % systematically low), off the **last two** quarter-envelopes.
- **Ships BOTH seeds, disp as default — the received "never displacement" rule is only half right;
  measured.** Full map + correction now in [[geometric-string-state]]. disp reads the tongue cleanly
  (frac=0 → 1.00× = the claim); vel reads marginality (6.88×, secular — and the log-y envelope
  discriminates for free: secular BENDS, exponential is straight).
- Both **bit-exact zeros** kept as gates: planar `max|w|==0.0`, and the unseeded whirl at the tongue
  centre `== 0.0` (without it every growth ratio partly measures a leak).
- **`data-hide-domain` is a NEW additive attr** — the inverse of `data-domain`, for sliders shared
  with a domain-less model (κ, amplitude — the whirl derives both). Plain `data-domain` hides an
  element under *every* model with no secondary select, which would have hidden amplitude on tension.
- **Viz scales computed ONCE over the run, never per frame** (a per-frame autoscale renormalizes the
  whirl's growth away). `u`/`w` SHARE a scale (their ratio is the claim); `v` gets its own (different
  quantity, orders smaller); each strip prints its own. Frames are `(n_frames,3,N+1)`, `fields:
  ["u","w","v"]`, `dims` still 1 ⟹ string path untouched.
- **Verifier gotcha, round two:** plate_supported/plate_free FAILed on "computing…" — **false FAIL,
  environmental**: the plate renders fine (43.7 s, 46 frames) but the machine was loaded with the
  *human's own* pytest + a ballistics sweep (see [[identify-processes-before-killing]] — I checked
  cmdlines and killed nothing). The 40 s window is the binding constraint, not a regression.

**Phase D batch 4 = phantom partials, model #10's bridge force (DONE & PUSHED, 13 web tests → 146
web / 938 suite, all green). Browser-verified 13/13.** The debt batch 3 took on going viz-only, and the discharge of model
#9's FIRST refusal (a scalar tension has nowhere to put a combination tone; a tension FIELD carries
`a·r²v_x/2` ⟹ two partials drive `v` at `f_i ± f_j` = Conklin phantoms, read off `EA·v_x(0)`, what
radiates in a piano). A 4th geometric regime. Load-bearing:
- **It reproduces `tests/test_geometric_phantom.py`'s rig EXACTLY** (same `lam_long` / 0.10 s window /
  two-mode IC / `v=0` / blind band-limited detector) ⟹ **inherits the suite's validation** instead of
  asserting something new, and reports the suite's own numbers back (err **0.039 Hz**, dominance
  **5.4×**, defect **4.57 Hz**). *That match is a TEST*: at the wrong amplitude it read 0.035/4.7×,
  which is how the merge bug below surfaced. **Generalizable: make the viewer run the rig's exact
  config and the rig's own numbers become a free end-to-end oracle.**
- **Don't pre-solve `v` to quasi-static equilibrium** (advisor's blocking catch — I floated it). `v=0`
  isn't the longitudinal equilibrium so it radiates a transient that IS the largest feature in the
  *full* bridge spectrum — but the tests start from `v=0`, so a hand-rolled static solve shows numbers
  **no test backs**. And it buys nothing: transient at 2236 Hz, phantoms below 500 ⟹ 4.5× above the
  band, which is *why* the rig band-limits (and why the phantom band is purely forced response).
- **Headline = the defect `f₂−2f₁`, ONE run, no oracle** (harmonic ⟹ `f₂−f₁=f₁` and `2f₁=f₂` exactly ⟹
  both displacements are the SAME number from opposite sides: 4.56/4.55 vs 4.57). The `f₃`-gap form
  needs an amp→0 run to earn the ladder; **step count is amplitude-independent** ⟹ +45 s for a
  strictly weaker claim. No hardening confound (it moves phantoms+partials together, and *widens*).
- **THE DEFECT IS NOT PURE STIFFNESS — θ-dispersion drags `f₂` flat = a NEGATIVE contribution.**
  Ladder defect at κ=0/2/8: N=16 `−0.965/−0.677/+3.571`; N=24 `−0.430/−0.137/+4.168`; N=32
  `−0.242/+0.052/+4.377`. κ=0 is *pure dispersion*, O(h²) (0.965/0.242 = **4.00** exactly). ⟹ **N is a
  SECOND control, not just κ** (κ=8 at N=8 = +0.38, unshowable), and **the κ=2 trap is worse than
  recorded**: at N=32 dispersion nearly *cancels* the stiffness (+0.05 net). ⟹ the gate is
  **one-sided (`>=`, never `abs`)** — a coarse grid displaces phantoms to the WRONG side by artifact,
  which `abs()` would score.
- **Verdict ORDER is the trap:** `detect_peaks` returns `[]` on a zero signal *rather than raising*,
  so the `EA=T₀` control (`a=0` ⟹ fields decouple ⟹ bridge **bit-exactly 0**) must fire BEFORE the
  defect gate — else a passing defect paints the claim green over an **empty spectrum**. Found by
  probing the detector, not reading code. Control is ~free: `EA=T₀` ⟹ `c_long=c` ⟹ fs 159k→1.8k.
- **TWO strips — one axis cannot carry both halves.** Wide (0→4.8f₁, tightest containing `2f₂`) =
  "peaks at the 4 combos"; but 4.6 Hz there is **~4 px** ⟹ "NOT on a partial" renders backwards (the
  diagnose figure's 2 kHz trap, one level down). Zoom on `f₁/(f₂−f₁)` ⟹ ~31 px, plainly two lines.
- **Display grid ⊥ measurement grid:** at the rig's 2× pad the zoom band held **12 points**. Traces
  pad 32× (~190 pts); the detector keeps the rig's default. Padding is for the eye, never the number —
  proved by the measured values being **bit-identical** across the change.
- **Position ≠ resolution, and the panel says so unasked:** the 0.1 s Hann lobe is `4/T ≈ 40 Hz` ⟹ f₁
  and f₂−f₁ are inside ONE lobe. Fine: `f₁` is ABSENT from `v` ⟹ no neighbour to blur it ⟹ parabolic
  refinement locates to 0.04 Hz. (A DSP-literate reader would otherwise catch the panel out.)
- **The verifier is DEEP-LINK-ONLY ⟹ it structurally cannot see the interactive switch** (advisor's
  catch). It navigates `?model=&domain=`, so it never fires the `domainSel` change handler nor the
  *switch-away reset*. Driving the switch by CDP (dispatch `change`, read slider values, autorender
  OFF so nothing kicks a 45 s render) found a **second real bug**: `data-hide-domain` early-returned
  on `if (el.hidden) return` ("model gating hid it, leave it") which **LATCHED** — it only ever
  worked because every element using the attr also had `data-show`, whose pass rewrites `hidden`
  unconditionally first. `animation_window` has none ⟹ once you visited phantom its slider stayed
  hidden through *every* regime until reload. Fix: recompute from scratch, never read `el.hidden`
  (`hidden = modelHides || domainHides || regimeHides`). **Generalizable: visibility/range state must
  be RECOMPUTED per pass, never accumulated — and a fresh-load verifier proves nothing about
  transitions.**
- **`MODEL_RANGES` merge was SHALLOW — a real bug THE RENDERED PICTURE CAUGHT.** A regime's
  `{val: 0.0015}` replaced the whole `{min,max,step,fixed}` ⟹ slider kept index.html's stale
  `step="0.001"` ⟹ **snapped 0.0015 → 0.002 and rendered the wrong physics** (and left N's cap at 512
  vs `GEOM_N_MAX=32`). Now `mergeSpecs` merges **per param**. The `_default`-leak trap one level down.
  Regime ranges keyed `"model:domain"`, applied on domain change only for models that declare them
  (`hasRegimeRanges`) ⟹ membrane circle→rect still doesn't reset sliders.
- Own budget `GEOM_PHANTOM_WORK_MAX=16_500` ≈ **45 s** at N=32 — *measured, not extrapolated* (bare
  Newton 35.6 s; telemetry carries the rest ⟹ **extrapolating a loop understates a payload build**).
  Window is fixed physics (`animation_window` ignored AND hidden). **Verifier wait 40 s → 90 s** or
  this case false-FAILs mid-"computing…".
- Audio debt discharges honestly: the ONE geometric regime with sound, only because its window is
  already 0.1 s of the radiating channel (free, not affordable). Labelled a blip dominated by the
  longitudinal transient, not the phantoms.

**Phase D batch 5 = mallet → drum, model #7, the FIRST contact model in the viewer (DONE & PUSHED,
15 mallet web tests → 161 web / suite green; verifier 15/15; both PNGs eyeballed; CDP switch-check
ALL PASS).** Reuses the membrane heatmap wholesale (`_build_membrane`, decimation, mask). See
[[mallet-collision-state]] for model #7 itself. Load-bearing, all *measured*:
- **CONSERVATION rides the ordinary drift panel — NO new verdict (the closed-system contrast with the
  bow).** The mallet+felt+membrane is a *closed* system whose `energy()` includes the mallet KE, so
  E₀ = ½M·v₀² is a clean nonzero denominator ⟹ the σ=0 drift check works (1.5e-12 THROUGH a strike).
  The bow needed the balance panel only because it is driven from REST (E₀ = 0); this is not driven.
- **THE catch = `decay_oracle=False` (advisor's blocking catch, VERIFIED before building).** With
  restitution ≈ 1 the mallet flies off force-free, so total E sits on a near-constant ½M·v₀² floor
  with a ~0.05 % membrane wiggle ⟹ `_fit_decay` reports `measured_2σ ≈ 0.0004` vs `oracle_2σ = 4.0`
  = a lying "broken match" over perfect passivity. Additive flag drops the 2σ line ⟹ lossy verdict =
  pure passivity (monotone). There is NO closed form for mallet+hysteresis+membrane decay, so
  passivity IS the honest verdict. **Ran the σ>0 probe first, confirmed 0.0004-vs-4.0, THEN built.**
- **THE contact duration is NOT ~2 ms — that is the RIGID-WALL oracle.** On the coupled *yielding*
  membrane the head relaxes and contact lasts **~40-60 ms (~20×)** and scales with **√M** (measured:
  M=0.02 sep@40 ms restitution 0.9997 finalHead 0.05 %; M=0.05 sep@59 ms; peakHead ~65 % mid-contact
  transient dimple). ⟹ diagnostics must run over the FULL audio (which captures separation), not a
  short window. **Generalizable: a "contact time" from a rigid-wall closed form under-reports the
  coupled duration by the compliance ratio — measure the coupled run.**
- **ONE instrumented audio loop, not simulate().** `simulate()` gives energy+pickup but NOT the
  mallet internals (velocity, contact force, in_contact) — and the CONTACT is the headline — so the
  audio run is a hand-rolled step loop capturing both ⟹ conservation number + contact diagnostics in
  one pass. A `SimResult` is *constructed* from the captured energy/time for `_energy_block`. Anim
  run stays separate (fresh mallet from rest, fundamental stride → heatmap).
- **The headline is the CONTACT, not the tone (advisor).** A point mass is an inefficient membrane
  exciter (restitution ≈ 1, head keeps ~0.05 %) — the 2nd panel is a **contact episode** (mallet
  velocity crossing zero into a bounce + the felt-force pulse + a separation line), NOT a mode
  spectrum (a soft felt low-passes the strike ⟹ per-mode locking would lock onto noise). Retention
  read at the **PEAK** (the signature-test lesson). Reported, **never tuned to ring louder** — "a
  point mass barely rings a drum" is printed in the readout.
- **Strike marker = the SNAPPED node** (payload reports the ctor-snapped fractions, not the raw
  slider) — a filled red dot, distinct from the hollow yellow pickup cross. **`pluck_width` hidden
  for the mallet** (point contact, no width) via a `data-show` exclusion (`"membrane plate vk"`).
- **Own budget `MALLET_WORK_MAX = 3.5e8` = HALF the membrane's 7e8** because a coupled step is
  1.5-3× a bare membrane step (a ~constant ~20 µs root-find, dominant at low N). `MALLET_N_MAX=80`.
  Both cost guards reachable (nlive + node-steps). Default N=60 ≈ 1.4 s render, worst ≈ 5.6 s.
- **Audio is the real 3-mode ring** (FFT peak on f1, faint, normalized to 0.9, `peak` reported) — the
  advisor's "sanity-check it's the ring not the dimple relaxation" passed.
- **The switch-driving CDP check (batch-4 lesson, applied — and the trap RESURFACED in the harness
  itself).** The deep-link verifier structurally cannot fire the model-switch handler, so a
  CDP-driven switch (autorender OFF) confirmed the transitions: felt sliders IN + `pluck_width`
  hidden + sigma reset + domain populated on switch-into; mallet-only sliders OUT + no stale-range
  garbage on switch-away. **Meta-trap:** the check script's OWN f-string+plain-string `}}`
  concatenation silently produced a JS syntax error that no-op'd the switch, so it read the initial
  `ideal` state and (with a naive `.hidden` check) reported false greens/reds — the *exact*
  "a fresh-load verifier proves nothing about transitions" lesson, one level up. Fixes: build the
  whole IIFE in one f-string; check effective visibility with `closest('[hidden]')` (headless has no
  `offsetParent` layout, and a slider div's own `.hidden` is false while its parent fieldset is
  hidden). The switch-check lives at `M:\claud_projects\temp\mallet_switch_check.py`.

**Phase D batch 6 = sympathetic / coupled strings, the FIRST MULTI-STRING model (DONE & PUSHED, 17
web tests → 178 web / suite green; verifier 17/17; both PNGs eyeballed; CDP switch-check ALL PASS).**
`drawFields`' 2nd customer (built in batch 3). See [[sympathetic-strings-state]] for the core model.
**J FIXED at 2** (the validated oracles are two-string; a count slider would break them). Load-bearing:
- **The claim is the 2nd panel, NOT energy (advisor's #1 catch).** Conservation + passivity are
  AUTOMATIC from the linear-leapfrog structure and pass even a flipped coupling sign ⟹ table-stakes
  green (σ=0 drift ~4e-14). Closed & undriven ⟹ rides the ORDINARY drift panel — **no balance panel,
  no decay oracle**. The money is the shared bridge displacement `w_b(t)`.
- **`normal` regime runs BOTH ICs (advisor).** Antisym `u_B=−u_A` ⟹ `w_b == 0.0` **bit-exact** (IEEE
  float negation is exact) + `E_body==0` forever; symmetric contrast swings the bridge, loads the body
  ~61%. A flat-zero alone reads "broken" ⟹ ship & plot BOTH (the whirl-needs-planar lesson). **Detune
  slider GATED OUT of `normal`** — any per-string difference degrades the bit-exact 0 to ~1e-13.
- **`transfer` = the money visual:** pluck A over the full slosh ⟹ per-string energy-fraction panel =
  the classic coupled-oscillator exchange (unison 88%, detune-4-semi 9%, >3× ratio = frequency-
  selective). Full-slosh animation resolves f1 in ~1480 frames (≈10 fr/period, just under MAX_FRAMES)
  ⟹ A rings down / B rings up is animated AND panelled.
- **Hand-rolled instrumented loop MANDATORY (advisor):** `.state` is string 0 only, `simulate()` gives
  neither the J stacked fields nor `w_b(t)` nor per-string E — all three ARE the panels. Capture all,
  construct a `SimResult` for `_energy_block` (geometric/mallet pattern). Frames `(n_frames,J,N+1)`,
  `fields:["string A","string B"]`, `dims` still 1 ⟹ string path untouched.
- **Audio = string pickup, NOT body `pressure()` (advisor):** body pressure ≈ 0 on the antisym mode
  (real physics: the aftersound doesn't radiate) but would render as broken. The string still rings.
  NOT viz-only (fs~22kHz, unlike geometric's 22×) ⟹ audio cheap & real.
- **`drawFields` generalized WITHOUT disturbing geometric** (field count/amps/labels/palette = module
  state at load; geometric = 3 strips u/w-share-uwAmp + v-own with its 3 original labels+colours =
  bit-identical picture re-verified in PNG; sympathetic = J strips sharing ONE amp — per-strip
  autoscale would flatten the ring-up/ring-down). **No orbit** (w_b is in the panel) ⟹ full width.
- **LATENT BUG FIXED:** the domain-select container's `data-show` never listed `geometric`, so the
  geometric REGIME dropdown was reachable only by `?domain=` deep-link, never interactively. Added
  `geometric sympathetic`; the CDP switch-check confirms both regime selects now work (batch-4 lesson:
  a deep-link verifier proves nothing about the switch). Switch-check
  (`M:\claud_projects\temp\symp_switch_check.py`) also confirms the regime gate RECOMPUTES both ways
  (detune hidden→shown→hidden, no latch), K resets 8000↔1500 per regime, no stale out-of-range.
- **Weinreich two-stage decay DEFERRED** (loss regime) → **DONE in batch 7 below**.

**Phase D batch 7 = Weinreich two-stage decay, the FIRST body-loss slider (DONE & PUSHED; 11 web
tests → 189 web / suite green; verifier symp_weinreich PASS; CDP switch-check ALL PASS; PNG
eyeballed).** The deferred 3rd sympathetic regime. Strike ONE of two near-unison strings over a LOSSY
bridge: symmetric mode loads the lossy body → dies fast (prompt); antisymmetric barely loads it →
lingers (aftersound). All-wrapper (core untouched). Load-bearing, all *measured*:
- **`sigma_body` GATES the verdict, not just the physics.** σ_body=0 → closed system conserves → the
  ordinary DRIFT check; σ_body>0 → **passivity + `decay_oracle=False`** (mallet's 2nd customer): total
  E is a two-rate decay to a NONZERO aftersound floor, so a single fitted 2σ vs a flat oracle would
  LIE. No closed form for coupled modal decay over an off-harmonic body ⟹ passivity IS the honest
  verdict. The core guard is σ-BLIND (lossless `omega`-only operator) ⟹ a lossy body only adds
  headroom, can't destabilize a σ=0-passing config (advisor, verified).
- **Anchor sub-claim = batch-6's oracle resurfacing.** strike-one = ½sym+½antisym; at detune=0 the
  antisym mode is bit-exactly bridge-decoupled (w_b≡0) ⟹ body damping NEVER activates on it ⟹ its
  tail is LOSSLESS in the discrete scheme, exactly ⟹ aftersound slope≈0, rising with detune (measured
  0.02→0.23 s⁻¹ at detune 0→0.3 semis). Floor value (~50%) is REPORT-ONLY (split is approximate).
- **Two runs; strike-both is NOT droppable** — it's the contrast that PROVES the plateau. strike-both
  (pure symmetric mode) loads the body fully → decays away single-slope (~5%). So the strike-one
  plateau (~50%) is the un-decaying antisym mode, NOT a noise floor. Budget 2×n_steps, same
  SYMP_WORK_MAX (default 2s×2≈89k<130k).
- **Ripple trap (advisor):** E_str=E_A+E_B EXCLUDES E_conn/E_body ⟹ oscillates (slosh+beat) even
  lossless ⟹ raw fit measures ripple. Fix = **sliding-MEAN envelope** over 1 fundamental period
  (`uniform_filter1d`; whirl's precedent but MEAN not max — max rides the slosh peaks). First point
  dips <1 (centered mean over the prompt) — cosmetic, tested [0.85,1.05].
- **Rate fit handles a SUM of exponentials:** log(E_str) CURVES through the knee ⟹ prompt rate fit on
  `log(E_env−floor)` (steep part above the plateau), aftersound on `log(E_env)` (late window). Rates
  are REPORT-ONLY (no oracle, like the mallet's contact time); the sharp claim is aftersound-vs-detune.
- **Params measured, NOT the diagnose rig verbatim:** K=6000 (own MODEL_RANGES key), **σ_body default
  20 not the diagnose's 10** (10 gave a 0.6 s⁻¹ prompt = no visible knee), max 80 (heavier→invisible
  cliff). detune re-ranged to a **FINE 0..0.4 semis step 0.01 (~1 cent)** — a piano unison is mistuned
  a few cents, not transfer's 0..12 semis. `_default` gives the semitone fallback (weinreich→transfer
  restores it); **sigma_body reset in `_default`** (the recurring leak). detune default 0 leads with
  the exact invariant.
- **Wiring:** `{"domain":"weinreich"}` bad-param test → `nonesuch` (weinreich now valid); drawEnergy's
  `decay_oracle=False` readout BRANCHED (mallet's "½Mv₀² floor" wording wrong here); symp-hint +
  drawSympathetic (log-y, drawWhirl precedent) + drawDiagnostics title all gained a weinreich branch.
  CDP switch-check (`M:\claud_projects\temp\wein_switch_check.py`) DRIVES THE SLIDERS (deep-link
  verifier can't): sigma_body appears only in weinreich (recomputes, no latch), detune range 0.4↔12,
  and the RENDER checks confirm the verdict TOGGLE — σ_body=0 flips "passive"→"conserved" + "nothing
  decays"; detune=0.3 → "cents mistuned".

**Phase D batch 8 = the jawari / buzzing bridge, the FIRST BARRIER model in the viewer (DONE; 11 web
tests → 200 web / suite green; verifier 19/19; CDP switch-check ALL PASS; PNG eyeballed).** A
*configuration* of model #8 (`BarrierString` with a parabolic bridge), zero core edits. See
[[jawari-state]] and [[barrier-collision-state]]. The mallet (batch 5) was a *point* contact that
separates; this is **distributed and persistent** — the string wraps onto the curve each downswing
and its departure point travels. Load-bearing, all *measured before wiring*:
- **The claim is the SHIMMER, not the energy** (batch 6's lesson in a new model). Energy passes for
  a flat rail too ⟹ table-stakes green. Headline = late-window spectral-centroid **elevation over a
  clean string** (3.44×, gate 2.5×); 2nd panel = the **travelling wrap edge** (std 4.89, nodes 0–14).
- **`decay_oracle` stays TRUE — the batch's one genuine REVERSAL of a prior template, and I was
  right against the advisor.** The advisor's pre-build guidance was to copy the mallet's
  `decay_oracle=False`. I probed first: the bridge is a **lossless elastic** barrier — it moves
  energy into the highs but **dissipates none** — so every mode still decays at exactly 2σ₀ and the
  flat-loss oracle survives the wrap (**measured 2σ = 1.009 vs oracle 1.000, log-fit rms 0.000**).
  I surfaced the conflict in a reconcile call rather than switching silently; the advisor conceded.
  **Generalizable: `decay_oracle=False` is for models with no single-exponential form, NOT for
  "anything with contact in it" — check the dissipation, not the vibe.**
- **σ₀ GATES the verdict** (weinreich's pattern, 2nd customer): σ₀=0 → drift **1.0e-12 through a
  sustained curved wrap** (contact duty 42 % ⟹ a real many-node contact, not a contact-free run
  proving nothing); σ₀>0 → passivity + the 2σ₀ oracle. **σ₁ and hysteresis FIXED at 0, deliberately
  unexposed** — σ₁ makes the highs decay faster than 2σ₀ (measured 2.77 vs a 1.00 oracle = a 2.7×
  "mismatch" on correct physics) and hysteresis is contact *damping*, fighting the re-injection the
  model is about. **Loss defaults ON (σ₀=0.5)**, the bow's pattern: "SUSTAINED brightness" is
  meaningless on a lossless string where every mode sustains by definition.
- **IC = mode-1 sine, NO pluck slider** (advisor's pre-build catch — the one thing no probe would
  have caught). The headline is a *contrast*, so the clean baseline must be spectrally pure: mode-1
  puts the clean centroid exactly on f₁ (100.0 vs 100 Hz). A raised-cosine pluck gives the clean
  string its own highs and shrinks the contrast, possibly under the gate. #6's `mode11` lesson, 3rd
  customer.
- **`downswing/depth` is the control, not `depth`** — measured a **ONE-SIDED FLOOR, not a window**
  (ratio 15.1/7.5/3.8/1.9/0.9/0.6 → 3.48/3.75/3.44/3.15/2.33/1.63×, wrap contracting max node
  14→11→8). **Amplitude moves it exactly as hard as depth** (a depth-only guard lets an under-plucked
  string through). Below the floor → **LABEL (GRAZING), never FAIL** — the bow's Schelleng rule.
- **Two silent param collisions closed:** `K` is the sympathetic bridge *spring* (~8000 N/m; this is
  2e6 N/mᵅ, **250× stiffer**) and `alpha` the mallet's felt exponent (2.3 vs 1.5). `gatherParams`
  sends every slider incl. hidden ⟹ merely *visiting* those models would have rendered a different
  bridge with nothing on screen to say so. Fix: a distinct `bridge_stiffness`; alpha fixed
  server-side rather than earning a third leak. Pinned by a test shipping `alpha=2.3, K=8000`.
- **THE INDEXING TRAP — a test caught it, not the eye.** `_b` and `contact_mask()` are over the
  **SUPPORT** (~15 nodes), not the grid: the barrier shipped compacted across the first 15 nodes and
  the wrap marker landed **one node off** the contact it marks — near a termination that survives
  eyeballing the PNG. Both now scattered onto the grid; wrap *statistics* stay support-relative (the
  model's own test frame). **Generalizable: when a core object exposes a masked/compacted array, the
  FRAME IS PART OF THE CONTRACT — assert length against the grid, not against itself.**
- **THE OFF-GRID SNAP — the tests pinned a config the UI could NOT express; only the CDP check saw
  it.** jawari set `audio_duration` val 0.24 but re-ranged no `step`, so the slider kept index.html's
  `step="0.1"` and a range input **silently snaps off-grid values** → the browser shipped **0.2 s**
  and rendered **2.75× where the tests assert 3.44×** — still SHIMMER, still green, but a hair over
  the gate and the "viewer reruns the suite's exact rig" oracle quietly false. Fix: re-range
  `step`/`fixed` with the val + reset both in `_default`. **Now enforced for EVERY model: the
  switch-check sweeps all 12 asserting each visible slider's value lies on its own step grid from
  min.** Third member of the `MODEL_RANGES` family after the shallow-merge stale-step (batch 4) and
  the `_default` bound leak (batches 2/8) — one shape: **a partially-overridden slider spec keeps
  stale fields from the layer below, and the render is WRONG WITHOUT BEING BROKEN.**
- **Only two runs are paid for**: the flat-rail control ("buzzes" vs "travels") is already validated
  in `tests/test_jawari.py` (std 4.89 curve vs 2.35 flat) ⟹ the panel *cites* it. Budget counts
  **both** runs (`2·n_steps`); a per-run cap would licence twice the wall clock. `JAWARI_N_MAX=128`,
  `AUDIO_MAX=1.5`, `WORK_MAX=150_000`. Default ~4.9 s, worst *passing* **34.2 s measured**.
- **A SCREENSHOT MUST SAY WHICH FRAME IT IS.** The verifier caught the jawari mid-*upswing* — string
  clear of the bridge — so the zoom pane's whole purpose went unverified. Two freeze attempts then
  failed SILENTLY: **assigning `animPlaying`/`currentFrame` from `Runtime.evaluate` does not stick**
  (reads back `true`), so the rAF loop kept advancing and each capture showed an unrelated but
  plausible frame. Fix: drive the **scrub control** (`input` ONLY — `change` clears `scrubbing` and
  returns the frame to the clock) and **read `currentFrame`/`wrap`/`u_mid` back at capture time and
  print them**. Then a contact frame confirmed the string lying *along* the curve with the red
  departure dot on the node `wrapFrames` names. A field probe pinned alignment independently:
  `wrap>=0` on exactly the frames with `u<0` near the bridge, `-1` on the positive ones, duty 44 %.
  **Generalizable: an animated viz needs its frame index IN the evidence, or eyeballing proves
  nothing about the frame you meant.** Captures: `M:\claud_projects\temp\jawari_contact_frame.py`,
  `jawari_frame_align.py`.
- **Port note:** something of the human's took **8000** mid-session (bind → WinError 10013, and `/`
  answered 404). Did NOT kill it ([[identify-processes-before-killing]]) — the server takes
  `--port` and both harnesses honour `VIEWER_BASE`, so 8010 works.
- **TWO views in the animation** (the phantom batch's two-spectra lesson, one level over): the bridge
  spans ~15 % of the string and drops ~1 mm against an 8 mm swing ⟹ in the full view it is a few
  pixels. The zoom pane rescales y to the depth and is the money picture. Spectra share **ONE scale**
  (per-trace normalization would render two similar curves and delete the result) on a **log axis**
  (the re-injected partials are 1e-2–1e-3 of f₁). Switch-check: `M:\claud_projects\temp\jawari_switch_check.py`.

**How to apply:** any *new* resonator slots in as another `_build_payload_*` behind the dispatch + a
panel; a *driven* one passes `balance_work=` to `_energy_block` and gets the balance panel free; a
*closed/struck* one rides the drift panel with `decay_oracle=False` (mallet + weinreich are the two
templates; a loss slider can GATE the verdict — `sigma_zero=(loss==0)` → drift-at-0 / passivity-at->0);
a
*multi-field* one reuses `drawFields`' stacked strips (geometric u/w/v + sympathetic string A/B are
the two customers; the field count/amps/labels are module state set at load); a *regime*
that needs its own slider defaults adds a `"model:domain"` key to `MODEL_RANGES`. The θ-artifact
loose end from [[stiff-string-state]] is still untouched (the portability-test one is RESOLVED). See
[[mallet-collision-state]] for model #7; [[geometric-string-state]] for #10; [[tension-string-state]]
for #9; [[commit-push-at-batch-end]]; mind [[respect-ruff-line-length]].

**Phase D batch 9 = the acoustic bore + radiating bell, the FIRST WIND model and a new field type
(DONE, 2026-07-20; 16 web tests → 216 web / suite green; verifier `bore_radiating` + `bore_open`
PASS; both PNGs eyeballed).** All-wrapper, `physsynth/core` untouched. Plan + probe findings below;
what the WIRING added, all found by building:

- **The verdict rides on the TOTAL, and that is the quiet failure mode.** Feeding `acoustic_energy`
  into `_energy_block(sigma_zero=True)` FAILS the drift check (acoustic sheds 10–100 %), which reads
  as a bug and invites "fixing" it by flipping to passivity — silently demoting the CONSERVATION
  claim the batch exists to make. `_energy_block` grew an additive `split` param (the `balance_work`
  precedent: same-`idx` decimation, no branch rewrite); the full suite re-ran clean after, which is
  the check that matters for a shared-code touch.
- **The headline TEST needs both halves or it passes on broken wiring.** `drift < 1e-10` on the sum
  also passes with the bell shedding *nothing*, so the test asserts the split MOVES (acoustic → <5 %
  of E₀, radiated → >95 %, radiated monotone, sum flat). *Generalizable: a test on a sum cannot see
  a dead summand.*
- **The analytic curve must span the SLIDER's range, not the interesting part of the curve.** First
  cut shipped `logspace(-2, 1.5)` — correct physics, and the DEFAULT config's own measured point
  (a physical clarinet at R/Z₀=3e-4) landed **off the left edge of its own panel**. Every test
  passed; only the PNG showed it. Now `-4..1.5`, with `1.0` inserted EXACTLY (else the peak draws
  0.499973 and the anechoic null — the claim — is missed by the plot).
- **A conserving total plots exactly ON the frame's top edge** (`vmax = max(v)`) and reads as chrome,
  not as the result. 12 % headroom, applied ONLY when a split exists so every other model's panel
  stays pixel-identical.
- **The mouth glow tracks the RATE, not the cumulative total.** The payload ships cumulative
  radiated energy (monotone — right for the ledger), but a glow driven by it only ever brightens: a
  ramp that says nothing about *when* sound leaves. Differenced at the draw call it pulses as each
  wavefront hits the mouth. *Generalizable: the honest ledger quantity is rarely the legible
  animation quantity — difference it in the frontend, not the payload.*
- **Margins are ASYMMETRIC per end** — the flare and glow draw OUTWARD past the mouth, so a
  symmetric margin clipped the one element showing energy leave. Sized from `meta.ends`, which is
  also batch 10's seam.
- **A second regime beyond the plan: `open` (ideal pressure-release).** Costs a boundary tuple,
  gives the lossless contrast, and — the real reason — exercises the `meta.ends` switch NOW, so the
  reed inherits a dispatch proven with more than one case.
- **`_default` grew `L` and `animation_window`** — the bore is the first model to re-range either
  (L→0.5 m; anim window max 0.3→0.1 because the shared cap is a cost hole). The `MODEL_RANGES` leak
  family, 4th member.
- **Two `const bHint` in one scope** (drawBore's hint vs the bow's) — caught by `node --check
  web/static/app.js`, worth running before every browser check.
- Honesty gate that fires and is LABELLED never failed: at R/Z₀ ≳ 0.05 no standing wave forms, so
  `spectrum.applies = false` and the odd-harmonic/partial claims are withdrawn (nothing to measure ≠
  wrong). The envelope correctly degrades to the pulse's single pass.
- `scripts/verify_web_headless.py` gained a **name filter** (`… bore`) so a one-model batch skips the
  geometric string's ~2 min.
- Every pre-build number reproduced first try: drift 1.2e-14 / 3.3e-16, reflection err 1.4e-16, exact
  0.500000 anechoic, O(h²) ratio 4.005, 0.0000 cents at λ=1, 11.6–12.2 frames/transit, worst render
  3.0 s at N=256/1.5 s.

**Batch 9 plan & probe findings (2026-07-20).** Human chose the **wind leg** for the batch
batch over body/bridge, the barrier leftovers, and the parametric demo; advisor split it into
**batch 9 = linear bore + radiating bell, batch 10 = the reed** (batch 1's de-risking precedent — the
reed *contains* a `Bore` and both draw as pressure along the same tube, so the new **pressure/S(x)
field type** gets validated on the model with no per-step root-find). Full plan + every measured
number: `docs/dev/web-viewer-plan.md` § "Batch 9"; probe scripts in
`M:\claud_projects\temp\bore-viewer-probe\`. The load-bearing findings:

- **Radiation is BOOKED (`energy() = acoustic + radiated`), so σ=0 with a radiating bell still
  CONSERVES** (drift 3.3e-16..7.4e-15 while the bell sheds 9.8 %..100 % of E₀) — **the first lossy
  model in the viewer that keeps the drift verdict.** Corollary that constrains the sliders: **bore
  viscous σ is NOT booked** (no accumulator in `bore.py`) → σ fixed at 0 and unexposed; `R_bell`,
  as the dimensionless **`R/Z₀`**, is the loss control. Generalizable, and batch 10 inherits it: *a
  measured-channel verdict is only as strong as the least-booked channel you let the user enable.*
- **Panel must plot acoustic vs radiated SEPARATELY** — a flat "conserved ✓" beside an audibly
  decaying tone reads as a bug. `_energy_block` emits only the total today ⟹ a real addition.
- **Dispersion is an `eigsh` computation, NOT a render.** A λ slider costs steps as **1/λ** (fs =
  c₀N/λL), tripping the 300k cap at **λ=0.878**, to show 0.07–0.67 cents nobody can hear. Compute
  cents-vs-λ from `bore_low_eigenfrequencies` (ms); pin the audio at λ=1.
- **Interpolate the spectrum.** A crude bin peak-pick invented a bogus 1.69-cent N-dependence;
  `measure_partials_near` (parabolic) gives ≤0.007 cents. The exactness claim is the **eigenvalue**
  oracle: **0.0000 cents at λ=1, every N**. O(h²) confirmed: 4.01 ratio over a 2× refinement.
- **Reflection oracle = the headline** (with the acoustic→radiated split): `r=(R−Z₀)/(R+Z₀)` matched
  to **1.4e-16** across R/Z₀∈[0.03,30], exact **anechoic null at R=Z₀**. Costs ~N steps (a 2nd run,
  centred-Gaussian IC — still counted in `WORK_MAX`, the jawari rule). Odd harmonics are table-stakes
  (boundary-determined ⟹ robust to bump position); the **odd/even ratio is set by the FFT WINDOW**
  (2.29e5 @0.5 s vs 3.6e4 @0.25 s, flat in N) ⟹ gate at the SHORTEST allowed duration.
- Budget: ~9–11 µs/step **flat in N**, but fs = c₀N/L ⟹ budget the *product*. `BORE_N_MAX=256`,
  `BORE_AUDIO_MAX=1.5`, `BORE_WORK_MAX=300_000` → worst render ≈**2.7 s** (jawari's was 34.2 s).

**VIZ DESIGN SETTLED (task 2, 2026-07-20)** — plan §"The viz design"; probe `probe_anim.py`:

- **`drawBore` = its own path; the reason is CORRECTNESS not precedent.** `drawString` pins BOTH
  ends to the rest line, but the bore's closed end is a pressure **antinode** (only the open end is
  a node) — and that asymmetry IS the odd-harmonic claim. Generalizing would render the batch's own
  physics backwards.
- **Ends switch on `meta.ends`, never hardcoded** — this viz is the gate batch 10 reuses, and the
  reed is a bore with a new mouth end type ⟹ `"reed"` becomes an addition, not a rewrite. Radiating
  end = flared mouth + glow tracking the **booked** radiated power (field-side dual of the energy
  split); at `R=Z₀` the pulse vanishes — the anechoic null made visible.
- **THE TRAP: pace the animation on the TRANSIT, not f₁.** Transit `L/c₀`=1.458 ms vs f₁ period
  5.83 ms (=4 transits); a 1.5 s render is ~1000 transits. String-style f₁ pacing measured
  **2.98–3.05 frames/transit** ⟹ aliased; `playback_speed` can't fix it (already decimated in *sim*
  time). No new mechanism needed — `anim_stride = round((fs/f_ref)/fpp)` already takes a model
  reference (bow: `f_hard_est`, tension: `f_osc`). Use **`f_ref = c₀/L`** ⟹ **11.6–12.8
  frames/transit, flat in N**. `BORE_ANIM_WIN`=0.03 s ≈ 20.6 transits ≈ 250 frames.
  *Generalizable: pace on the timescale of the CLAIM, not the fundamental — they coincide for a
  string, differ 4× for a bore.*
- **Cost hole: `ANIM_WIN_MAX=2.0` is shared and the bore's budget doesn't cover it.** N=256 ⟹
  **351,232 anim steps**, over `BORE_WORK_MAX` alone; the `MAX_FRAMES` re-stride does NOT save it
  (caps frames emitted, not steps simulated). Count **`n_anim + n_audio`** against `WORK_MAX`
  (jawari precedent). *Generalizable: a frame ceiling is not a cost ceiling.*
- **Two runs, two emissions** — frames from `anim_res`, split from `_energy_block(audio_res)`;
  radiation booked in both but different sims/strides ⟹ don't share one array. Envelope = running
  `max|p(x)|` over the **audio** run, STATIC overlay (no single frame shows node structure);
  polyline = the short transit-paced window. Caveat to print: at `R=Z₀` there is no standing wave
  and the envelope correctly degrades to the single pass.
- MVP = tube + polyline + per-end rendering + envelope + **mouth glow** (headline half). The
  `divColor` tube-fill is the cuttable piece.

**Phase D batch 10 = the dynamic single reed, the WIND LEG'S EXCITER (DONE & PUSHED, 2026-07-20;
17 web tests → 233 web / suite green, ruff clean).** All-wrapper, `physsynth/core` untouched;
`ReedBore` *contains* a `Bore`, so batch 9's whole field type (tube, envelope, transit pacing,
`meta.ends` dispatch) came across and the reed added only a mouth end. The acoustic dual of the bow
[[bow-state]] — both driven from rest to a limit cycle by a scalar root-find per step. Load-bearing,
all *measured before wiring* (`M:\claud_projects\temp\reed-viewer-probe\`):

- **THE STRUCTURAL CLAIM: the reed's balance is a GENUINE RESIDUAL, not the bow's tautology.** The
  bow *infers* dissipation as `work − ΔE`, so its lossy residual is identically zero **by
  construction** — a green tick that cannot fail. The reed measures `jet_loss` and `reed_damp_work`
  independently and books the bell's radiation into `energy()`, so `ΔE − (mouth − jet − damp)` can
  actually fail: **1.6e-14** measured, while the same run with the measured loss **dropped** reads
  **0.95**. *That contrast IS the test* — batch 9's "a sum cannot see a dead summand" aimed at the
  residual itself. `_balance_verdict` gained an additive `measured_loss` mode.
- **THE TRAP, and it is in the SHARED code: the reed must NOT be σ-gated.** The bow's σ=0 branch
  scores `max|ΔE − work|`, valid *only* because a lossless string has `ΔE == work`. The reed's jet
  and lip-damping channels are on **even at bore-σ=0**, so `ΔE − mouth_work == −(jet+damp)` ≈ **60 %
  of scale** — routing it through the bow's branch reports a catastrophic **IMBALANCE on a perfectly
  balanced model**. The measured branch REPLACES both σ branches. *Generalizable: before reusing a
  verdict, check the identity it assumes, not just the shape of the data.*
- **NO VK-style convergence gate — verified, not copied.** The balance error is `k·p̄·R/p_pref0`,
  **linear in the Newton residual**, so a bad solve shows up directly *as* a bad balance instead of
  masking as a good one (tol 1e-10→1.8e-15, 1e-6→1.3e-11, 1e-2→7.8e-9, monotone over six decades).
  `newton_tol` pinned at 1e-10 and unexposed. *Generalizable: a gate is for when a bad solve can LOOK
  like a good result — check the direction of the coupling before copying a precedent in.*
- **BOTH SWEEPS PINNED AT A FIXED N off the render slider — a budget fix that is also a CORRECTNESS
  fix (advisor).** Naively recomputing at the user's N blew the cap ~2× (the threshold sweep alone is
  ~490k steps at N=256) *and* made the headline a function of the display grid. Measured
  **N-invariant to the 4th significant digit** (γ=0.338 → 0.00431…0.00426 across N=64…256; pitch
  leverage **3.05 % at every N**) ⟹ `REED_SWEEP_N=64`, memoized. Batch 9's compute-it-off-the-operator
  rule, with the invariance *measured* rather than assumed.
- **THE MEMO KEY IS THE CACHE TRAP (advisor's catch, and it paid off immediately).** The threshold
  lives in γ but `p_closing = mu·wr²·H0`, so a key missing `f_reed` — which IS an exposed slider —
  serves **stale numbers on interaction while passing on the defaults**. Key carries L, f_reed,
  q_reed, H0, radius, radiating, R_bell. Vindicated by the next finding.
- **THE THRESHOLD IS BELL-DEPENDENT AND MONOTONE — a lossier bell needs a harder blow.** Measured at
  N=64: R/Z₀ = 3e-4 and 2.5e-3 → bracket **(0.30, 0.338]**; 1e-2 → (0.338, 0.355]; 2.4e-2 →
  **(0.355, 0.372]**; and by 6.3e-2 the note **never speaks at all**. So the plan's bracket and the
  shipped one differ *because they are different bells*, not because either is wrong — and that is
  exactly why the memo key must carry `R_bell`.
- **γ = 1/3 is on the sweep grid EXACTLY** (batch 9's anechoic-null lesson, 2nd customer) — and it
  lands on a **knife edge**: 0.0196 against a 0.02 gate at the default bell. That is *correct* (1/3
  IS the threshold) but means the tests assert **the curve's shape, never that point's label**.
- **THE THRESHOLD NEEDS A ≥0.8 s WINDOW — critical slowing down (advisor's blocking catch).** The
  0.04 s settling time was measured at γ=0.5, *far* above threshold; near the onset a 0.4 s window
  read γ=0.355 as **0.01123** (half-speaking) where converged it is **0.00441** (silent). The
  transition is *sharper* than first measured. *Generalizable: a settling time measured deep in the
  oscillating regime does NOT bound the settling time near the bifurcation.*
- **The signature is the HEADLINE, the balance is table-stakes-plus** — reed-state's "necessary but
  NOT sufficient": a wrong-sign reed rings down with populated channels and still closes the book.
  So "the reed SPEAKS" is an **assertion**, not a reported number. Below threshold → **LABEL**, never
  FAIL (the bow's Schelleng / jawari's grazing rule, 3rd customer).
- **Pitch stated as LEVERAGE, not the suite's binary "<6 %":** +50 % on `f_reed` buys **+3.05 %** of
  pitch (−95.4 → −43.4 cents vs c/4L), monotone — the reed's compliance is an **end correction**, so
  a stiffer reed lands *closer* to c/4L. Interpolated (`measure_partials_near`); a raw bin pick at
  this window is worth 34 cents and would invent the trend.
- **The money animation is a travelling PRESSURE STEP — the acoustic dual of the bow's Helmholtz
  corner**, and batch 9's transit pacing carries it unchanged (11.6–12.8 frames/transit; f₁ pacing
  gives ~3.0 and aliases the batch's best picture). The window is the **TAIL** out of ONE run
  (batch 2's `snapshot_from` lesson — a 2nd resonator silently doubles a root-find-per-step model).
- **Batch 9's envelope goes nearly FLAT here and that is a RESULT, not a broken overlay:** a square
  wave superposes many harmonics, so max-over-time fills in every node but the boundary's (1569…1550
  …56 Pa). Captioned as the contrast with batch 9; the **mouth/bell ratio** carries the
  antinode/node claim. *Generalizable: an overlay informative for ONE mode can go featureless under
  a rich spectrum without anything being wrong — say which it is, don't "fix" it.*
- **The reed pane needs its OWN scale:** H0 = 0.4 mm against a 16 mm bore = **2.5 %**, sub-pixel to
  scale (the jawari zoom-pane lesson, 2nd customer). **Beating reported as DUTY (~44 %) primarily** —
  it needs no event definition; the raw crossing count reads **1.94/period** for one slam because the
  reed *chatters* at closure (a 35-sample precursor, a 66-sample re-opening, then the main 291), so
  episodes separated by <10 % of a period are merged → **1.00**.
- **Audio = the MOUTHPIECE and the caveat ships with it:** crest **1.116** (1.0 = a perfect square)
  and `f₁/2f₁` = 460, against the bell far field's crest 3.68, ratio 25.5 and **~270× quieter**.
  Radiation *differentiates*, so the mouthpiece square wave is **not what a listener hears** — the
  panel says so (batch 6's "audio = pickup not body pressure", from the opposite direction: there the
  radiating channel was misleadingly *silent*, here misleadingly *unlike the claim*).
- **Cost:** ~11 µs/step **flat in N** (the root-find is only ~15 % over batch 9's bare bore), but
  `fs = c₀N/L` ⟹ budget the product. `REED_N_MAX=256`, `AUDIO_MAX=1.0`, `WORK_MAX=300_000` bounds the
  **render only**; the fixed-N sweeps are a **separate** ~3.5 s constant so neither cap can launder
  the other. Worst *cold* render ≈ 6 s, ~1 s once memoized. `_default` grew `gamma`/`f_reed`/
  `q_reed`/`bell_ratio_exp` — the `MODEL_RANGES` leak family's **5th member** (the reed RAISES the
  bell default to −2.6, so without the reset a reed→bore switch renders a different tube silently).
- **TWO bugs that ONLY the rendered page could catch, both invisible to 233 green backend tests —
  the batch's most generalizable lesson.** (1) `drawReedSignature` looked up a `#partials-verdict`
  badge **that does not exist** (that panel has only a readout; every sibling folds its verdict into
  the text). Setting `.textContent` on `null` throws, the render handler swallows it, and the page
  reports a generic **"network error"** — *the symptom does not name the cause*. (2) On the ideal
  open end the far-field caveat computed **`1.6e+33× quieter (crest null)`**: nothing radiates there,
  so the comparison divides by ~0. Now withdrawn + labelled (batch 9's `applies=false`) and pinned by
  a test. **Generalizable: a payload can be numerically perfect and still throw on a DOM id or render
  a lying number — the browser check is not a formality, and "network error" can mean "a null in your
  draw path". An element id assumed by analogy with another panel is a real hazard.**
- **I under-delivered against my OWN committed viz design and the render caught that too:** task 2
  promised a reed *pane* (opening vs time, closures shaded, duty printed); the first cut shipped only
  a flap + a caption, which at H0 = 2.5 % of the bore diameter reads as a stray tick. Built properly,
  plus the reed's tube drawn higher/slimmer (`midY=0.40H`, `wall=0.26H` vs the bore's 0.5/0.32) so
  the pane clears the lower wall. **Re-read the settled design doc before declaring the build done.**
- **A sweep level can be EXACTLY zero** (the lossless open end below threshold is bit-silent) and
  `log10` of a clamped 1e-9 falls below the axis floor, drawing the point *outside* the panel. The
  floor is now the axis bound itself — a silent point sits ON the bottom, which is what it means.
- **`payload` / `currentFrame` are module-scoped `let`s — `let`/`const` NEVER become `window`
  properties**, so a CDP capture script must read state through the **DOM** (`#scrub.value`,
  `#energy-readout`, …), which is also what a viewer can actually see. Cost me a full false
  "NO FRAMES" round where the render had in fact succeeded (status said 240 frames).
- **Restart the server after editing `serialize.py`** — Python does not hot-reload, and a re-capture
  against the stale process reproduced an already-fixed bug and nearly sent me chasing it twice.
- **Process notes:** run `node --check web/static/app.js` before every browser check (batch 9's rule)
  — it caught a real syntax error where shell-collapsed `\n` escapes landed inside a double-quoted
  `join()`. And **I broke my own recorded rule** by starting the verifier concurrently with the full
  pytest suite (the Phase-C note above) — that run produced nothing and had to be redone alone; the
  machine also carried **5 of the human's own Python processes**, so identifying by command line
  before killing my stale duplicate genuinely mattered ([[identify-processes-before-killing]]).

See [[bore-state]] and [[reed-state]] for the core physics.

**Phase D batch 11 = the fret / flat rail, model #8 on its OWN terms (backend 3a pushed earlier;
frontend 3b + verification DONE this session; 20 fret web tests (19 from 3a + 1 dispatch-kind pin
this session), 181 web tests total, suite green, ruff clean;
verifier fret PASS; CDP switch-check ALL PASS; PNGs eyeballed).** A *second* configuration of
`BarrierString` (a STRAIGHT rail, not the jawari's curve), zero core edits — the jawari's physical
**opposite**: slap-and-release intermittent contact, not a persistent travelling wrap. See
[[barrier-collision-state]] and [[jawari-state]]. Runs the geometry `tests/test_collision_signature.py`
validates (`barrier≈−2e-3`, `lam=0.4`), so the viewer exercises the validated config, not a lookalike.
All-wrapper. Load-bearing, all *measured before wiring* (`M:\claud_projects\temp\fret-viewer-probe\`):

- **THE HEADLINE IS INTERMITTENCY and it is MEASURED, not asserted.** At the default the string makes
  **1.225 episodes/period at 15.45 % duty** — it slaps and springs off, never pinned. The money panel
  is a **contact RASTER** — an x-vs-t spark map, the viewer's first panel that is a picture of a
  field's HISTORY, not of a field. Next to the shipped jawari it reads instantly as the opposite
  regime (batch 6's "claim = the 2nd panel, NOT energy", 7th customer — energy passes for any rail).
- **THE ACTIVE SET IS A VECTOR, and that number goes on screen.** Of 99 support nodes, **up to 69**
  are in contact at once (mean ~39 while touching) at **Newton max 2 / mean 1.159**. That cheapness
  IS the `λ_min(J) ≥ 1` proof showing up as a measurement (unique root, global convergence). So
  `|𝒞|_active` gets its own trace on the FULL support; **binning x is free for the picture and fatal
  for the number** (69 → 23 at 33 bins), so only the *image* may be binned.
- **NEVER read a scalar off the raster — it is an OR-reduction ⟹ a DILATION in time.** Column-collapsed
  duty reads **28.8 % against a true 15.5 %** (and coarse columns SPLIT slaps, not blur them: 83/60
  apparent episodes at 200/100 cols vs a truth of 49 — a too-coarse raster looks *busier*). Duty,
  episodes, `|𝒞|max` are computed at FULL RATE in the payload and PRINTED; the image is a map of
  WHERE/WHEN, a biased estimator of HOW MUCH. Raster ships ≥10 cols/period for this reason.
- **THE 2σ₀ DECAY ORACLE DOES NOT TRANSFER FROM THE JAWARI — and the reason is real physics.** The flat
  rail is equally lossless (σ=0 drift **7.14e-13** through genuine contact) yet its decay reads
  **1.07–1.09× 2σ₀** (6–9 % fast). The loss identity is `dE/dt = −2σ₀·(2·KE)`, so "rate == 2σ₀" is an
  **equipartition** assumption `⟨KE⟩=E/2` — true of harmonic motion, NOT of a string slapped by a
  stiff one-sided spring. So the energy panel ships a **diagnostic TRIPLE** (`rate`, naive `2σ₀`,
  corrected `2σ₀·⟨2KE/E⟩`, agree ~0.06 %), **reported never gated** — both sides come from one run of
  one identity, the gate is the drift + passivity (already machine-precision). **This retroactively
  explains the jawari's 1.009: its gentle wrap stays near-equipartitioned, so the oracle passed there
  by ACCIDENT of that configuration, not by transfer.** *Generalizable: re-measure an oracle per
  configuration; a pass in one config may be an accident of it.* [[jawari-state]]'s decay_oracle=TRUE
  is still right for the jawari — this is decay_oracle=**False** + the triple, the third contact model
  (mallet/weinreich/fret) to drop the single-exponential oracle for a measured reason.
- **BRIGHTNESS IS NON-MONOTONE IN CLEARANCE — the panel must not say "closer is brighter".** Elevation
  over an out-of-reach control **peaks at ~2 mm** (2.50/2.71/3.33/2.59/2.83× at 4/3/2/1/0.5 mm) and
  falls as the string starts to pin. `test_closer_barrier_is_brighter` compares only 4 mm vs 1 mm and
  passes by a hair ⟹ a monotone label would be **disproved by the model's own star slider**. The
  signature panel draws the hump and marks the user's clearance against the peak. *Generalizable: a
  signature validated on a two-point comparison is not a monotone law.*
- **PITCH RISES STEEPLY BUT SHIPS AS `crossing_rate`, NOT pitch.** +1600 cents at the 0.05 L pickup,
  large and real, but a buzz adds crossings *within* a period so it mixes f₁ with the rail's
  contribution; `f=c/(2L_eff)` would overclaim (the string is NOT pinned to a shorter length). Gates
  nothing (Schelleng/grazing precedent, 4th customer). The control's centroid reads **exactly f₁**
  (a rail-free mode-1 pluck is a pure sinusoid) ⟹ the whole elevation is harmonic content the rail
  added — an unusually clean baseline, printed. Static-equilibrium oracle `S u* = (K/ρ)b` (residual
  3.4e-15) is **CITED as the contact-force magnitude credential, never led** (it is α=1, static, a
  flat line).
- **`rail_frac` CAN SILENTLY ERASE THE CLAIM, and it is a slider.** Peak swing over the rail is
  `A·sin(π·min(rail_frac,½))` and falls as the rail shortens toward the nut ⟹ at the default 2 mm
  clearance the rail goes **out of reach below rail_frac ≈ 0.15** (duty 0). Hence a **hard floor at
  0.2 (enforced server-side too)** and a **live pre-render hint** driven by `clearance` AND
  `rail_frac` — the frontend bug this session: `onControlChange` refreshed hints from an allowlist
  missing both, so the warning went STALE ("the string reaches the rail" while it was out of reach),
  i.e. the guard meant to fire *before* a ~15 s render said the opposite. `out_of_reach` and `pinned`
  are **labels not failures** — out_of_reach is live, pinned is a measured GUARANTEE that never fires
  (a lossless one-sided spring always pushes back ⟹ the string can't rest on the rail; duty asymptotes
  under 0.5 = the free-sinusoid limit, so the "pinned" bar sits at 0.9 where not-releasing lives).
- **THE FIRST CONTACT MODEL THAT NEEDS *NO* ZOOM PANE — the inversion is the batch's viz point.** The
  jawari (b8) and reed (b10) both needed a zoom because their gesture was sub-pixel; here the 2 mm
  clearance is **37.5 % of the 5.33 mm peak swing ≈ 75 px**, so the string, rail and gap all live at
  one to-scale `sy` (reuse `drawString`'s, do NOT autoscale like `drawJawariPane`). Layout is the
  jawari's INVERSE: left ≈42 % animation, right ≈58 % raster (the money panel gets the room). The
  gated claim is **painted on-canvas** (`#string` has no `<p>` readout, only the overlay). *Batch 2's
  transit pacing, 3rd customer: 8 frames/period recovers 100 % of episodes; f₁ pacing recovers 38.8 %
  (aliases the slap away). Duty is stride-robust, episode structure is not.*
- **Cost: the MOST expensive model per second of audio, and the dense solve is NOT why.** ~13 s wall
  clock per second of audio at N=100; halving the rail buys only ~20 % (the string step + rank-`m`
  correction dominate, not the `|𝒞|×|𝒞|` solve). Two runs: the fret + the out-of-reach control (0.95×
  the fret's cost, no contact solve to skip). `FRET_N_MAX=100`, `FRET_AUDIO_MAX=0.6`, `FRET_WORK_MAX`
  counts `n_steps + n_control`. Control stays SHORT (`FRET_CONTROL_MAX`) — its centroid is
  window-invariant (a rail-free pluck is a sinusoid); a first cut doubled it believing the elevation
  needed it, but the gap was the *comparison* truncating the fret pickup to the control's length. The
  work guard is REACHABLE (lower λ raises `fs=cN/(Lλ)`); the raster's x-binning branch is inert
  (m ≤ 99 < 128).
- **`kind` on the signature block was the ONE backend reopen.** `drawDiagnostics` dispatches on
  `meta.spectrum.kind` with a chain of *equality* branches ⟹ a block without it falls through to
  `drawPartials`, which reads per-partial arrays the fret never ships (the panel still draws, wrong).
  The contact block had `"kind":"fret"`; the signature block did not. *Generalizable: a missing
  discriminant in an equality-chain dispatcher is a silent MIS-ROUTE, not a no-op — pin it.*
- **The energy TRAP that passes a careless self-check** (advisor's flagged catch): `drawEnergy`'s lossy
  branch keys on `hasOracle = measured_2sigma !== undefined`; the fret ships `decay_oracle=False` ⟹
  `hasOracle` false ⟹ it would fall into the mallet/weinreich `else` and print *"felt loss removes
  energy from a ½M·v₀² floor"* (nonsense for a string on a rail) while `decay_triple` went UNSHOWN —
  panel renders, badge says "passive", a "does it draw? does it pass?" check sails past a lying
  readout. Fret branch keys on `isFret` (the MODEL), placed before the `isWein` check. σ₀=0 needs no
  handling — `e.sigma_is_zero` routes it to the lossless drift branch above.
- **The transition verifier's own traps cost more than the code's.** The deep-link harness can't see a
  transition, so `temp/fret-viewer-probe/switch_check.py` sweeps sliders + switches models live. It
  found the real stale-hint bug — after three self-inflicted false results: a mis-escaped f-string
  brace that silently never switched the model (every param then a "leak" against the fret's own page,
  batch-5 meta-trap resurfacing); a whole-canvas ink count that read the rail slab + claim text as
  raster ink (12623 on an empty raster); and — load-bearing — **`renderBtn.disabled` makes a Render
  click during the ~15 s page-load render a SILENT no-op**, so the pre-existing "ok" satisfied the
  wait and every screenshot predated the render under test. *Generalizable: a UI check keying on a
  status string must stamp a sentinel AND confirm the button was actionable — a stale "ok" is
  indistinguishable from a fresh one.* Standing rules re-confirmed: `node --check` first; restart the
  server after `serialize.py` (no hot-reload — two stale servers from the 3a session were still
  running the pre-edit code); run the verifier ALONE (never beside pytest); eyeball the PNG against
  the settled viz design before declaring done — the on-canvas claim text overlapped (wrapText returns
  a baseline, advance by +lh not +gap) and the rail fill flooded the pane on the first cut.

**Phase D batch 12 = the body speaks: string → modal body/bridge + radiation read-out (backend 3a
pushed earlier; frontend 3b + verification DONE this session; 15 body web tests from 3a, 266 suite
green, ruff clean; verifier body PASS; CDP switch-check + visibility-check ALL PASS; PNG eyeballed).**
The viewer's FIRST coupled-resonator + far-field model — the whole `exciter → resonator →
body/radiation` third stage, never shown before. A fixed/free `IdealString` terminated on a lumped
`ModalBody` through a linear bridge SPRING (`StringBodyBridge`), read out to the far field by
`AirRadiation`. All-wrapper, core untouched. See [[body-bridge-state]] and [[radiation-state]].
Load-bearing (frontend, on top of 3a's measured design):
- **THE MONEY PANEL IS THE ENERGY CARD ITSELF, carrying two things at once (the bore-split precedent,
  2nd customer).** `drawBodyEnergy` plots the `E_string ⇄ E_body` slosh (blue drains 100 %→22 %,
  orange body fills to 77 %, counter-phase over ~0.4 s) on the `#energy` canvas, WHILE the σ-gated
  verdict rides the badge+readout on the **absolute** total. That contrast — *total conserves
  (drift 8.5e-14 ✓) while E_string alone does NOT* — is the batch, and it lives in one panel. The
  `#string` canvas reuses `drawString` unchanged (the string is the only moving shape; the modal body
  is lumped, no spatial extent). `drawBodySpectrum` takes the 2nd panel = far-field `|Q″(f)|` +
  faint body-mode markers + the terminus glide/ω²-sanity/1-over-r readouts.
- **`E_conn` GOES NEGATIVE — never clamp it (advisor's #1 trap).** The transfer panel's
  `Math.max(0,Math.min(1,·))` would erase the whole "E_conn is its own channel" point (the
  cross-time spring term dips to −1 %). It gets its own SIGNED axis (y spans slightly negative to
  ~1.05), never stacked. And the flat green `total_frac` line is the **100 % REFERENCE the three
  channels sum to by construction — NOT the conservation verdict** (at σ>0 the absolute total decays
  while that line stays pinned at 1; the badge carries the verdict). *Generalizable: a fraction-of-
  total that is 1.0 by construction is a reference, not a result — don't label it "conserved".*
- **A SLIDER TWO MODELS NEED WITH DIFFERENT RANGES MUST BE ONE ELEMENT, re-ranged — never
  duplicated.** The backend (3a) reads `bridge_stiffness` for BOTH the jawari contact bridge (~2e6
  N/mᵅ) and the body's linear spring (~8k N/m), and `sigma_body` for both weinreich and the body. A
  second `data-param` element would split the `sliders{}` map (keyed by param) ⟹ the visible slider
  and the one `gatherParams`/`applyModelRanges` touch diverge ⟹ dragging does nothing. Fix = the
  leak-family discipline: single elements, re-ranged in `MODEL_RANGES.body` (`bridge_stiffness`
  [0,19k] val 8k, unit N/m not N/mᵅ; λ capped 0.99; audio ≤3 s), **reset one level up in `_default`**
  (jawari's [2e5,8e6] val 2e6 restored on switch-away). The exact dense guard trips ~21.5k on this rig
  (the string alone sits at 3.24 of the 4.0 limit at K=0) ⟹ [0,19k] with margin, over-stiff K = a
  clean construction ERROR. *Generalizable: a shared backend param name is a shared DOM element.*
- **`data-hide-domain` gates a slider a domain-model shares with a domain-LESS one — the advisor's
  flagged `updateVisibility` rewrite was AVOIDABLE.** `sigma_body` must show in sympathetic's
  *weinreich regime only* AND for the domainless body. `data-domain="weinreich"` force-hides it for
  every non-domain model (the pass overwrites unconditionally); the clean answer reused the existing
  inverse-gate (κ/amplitude's, batch 3): `data-show="sympathetic body"` +
  `data-hide-domain="normal transfer"`, moved out of the sympathetic fieldset into a shared "Bridge
  stiffness / body & radiation" group (a hidden parent fieldset hides its children regardless).
  The advisor flagged the visibility-pass rewrite as *the one edit that could silently break every
  slider*; `data-hide-domain` made it unnecessary. *Generalizable: reach for the existing inverse-gate
  before touching a shared visibility pass.*
- **THE FRESH-LOAD VERIFIER STRUCTURALLY CANNOT SEE THIS MODEL'S #1 TRAP (advisor's #4).** A deep-link
  `?model=body` initialises `bridge_stiffness=8k` by construction and NEVER ships a stale K ⟹ a passing
  PNG proves nothing about the leak. The hazard is the IN-PLACE `jawari → body` switch (`gatherParams`
  ships the hidden 2e6 → the body's ~21.5k guard → error). A separate CDP driver
  (`M:\claud_projects\temp\body_switch_check.py`) drives the real `modelSel` change handler and
  asserts the re-range BOTH ways (2e6 → 8k → 2e6) + a post-switch "ok" render; a visibility check
  (`temp/sigma_vis.py`) confirms `sigma_body`/`bridge_stiffness`/`distance` show/hide correctly across
  jawari / all-3-symp-regimes / body (10/10 PASS). *Generalizable: a per-navigation harness tests
  initialisation, not transition; a leak lives in the transition.*
- **Honest framing kept from 3a:** the radiated spectrum for ONE source == the body-pressure spectrum
  (1/r + delay only rescale/shift) ⟹ NO second "radiated" analysis pretending otherwise; the distance
  knob changes level+latency only, the readout says so (gain·r const). ω²-consistency ~1.00 is a
  near-tautology SANITY number, NOT a radiation oracle. Body modes BOOST neighbouring partials but form
  NO clean formants (off-harmonic avoided-crossing doublets) — markers shown, never an imposed 100·n
  ladder (batch-8 off-grid lesson). Terminus f₁ glides free c/4L→clamped c/2L, asymptotes BELOW 100
  (guard caps K, body is finite-mass) — honest secondary story, never claimed to hit 100. pickup marker
  SUPPRESSED for body (audio is far-field, backend probes x=L — a movable marker would mislead).
- **`decay_oracle=False` decided by MEASUREMENT (3a):** σ_body=0 → drift on the total; σ_body>0 →
  passivity, no 2σ oracle (off-harmonic coupled decay is genuinely multi-rate — the OPPOSITE of the
  jawari's lossless-elastic TRUE, because here σ_body is a REAL loss). drawEnergy's body branch keys on
  `payload.model==="body"` and early-returns (the fret's `else`-branch-lies lesson, so the mallet's
  "½Mv₀² floor" wording never leaks in). Standing rules re-confirmed: `node --check` first; TWO stale
  servers from the 3a session were STILL running pre-edit code and answered "unknown model 'body'"
  (both `web/server.py`, identified before killing per [[identify-processes-before-killing]] /
  [[port-reclaim-modus-operandi]]); ruff ≤100 in the first draft (three verifier-comment lines caught).

---

**Batch 13 (2026-07-21) — the plate BODY: the distributed body you WATCH ring (#5/#5b).** Batch 12's
lumped `ModalBody` swapped for a grid `Plate` via the already-validated `StringPlateBridge`, so the
coupling/radiation leg finally has a PICTURE: the soundboard (supported #5) / cymbal (free #5b) rings
on the heatmap. Model key `platebody`, ONE model with a **supported/free boundary DOMAIN** (ranging
is domain-conditional — `MODEL_RANGES["model:domain"]`, as the standalone plate already does). The
genuinely-new frontend work = the **dual field view** (`drawPlateBody`): the string on a thin strip
along the top, the plate heatmap (reusing the `dims:2` renderer wholesale) below. `drawBodyEnergy`
reused verbatim (plate rides the body's exchange keys; legend relabels `E_body→E_plate`). All-wrapper;
`physsynth/core` untouched. Load-bearing, all MEASURED (`temp/platebody-viewer-probe`):

- **THE TERMINUS f1 CANNOT BE READ OFF THE USER RUN — it needs a dedicated near-nut-pluck probe. THE
  #1 TRAP, and only the DEFAULT-params PNG caught it.** The measured design read the terminus from the
  near-nut pickup of the main run; the 0.5 s test passed (free 116.7) but the fresh-load PNG at the
  DEFAULT 2 s showed free = **99 Hz** — NOT an overshoot, flatly contradicting the "OVERSHOOTS" text.
  Physical cause: the user's 0.3 L pluck excites the avoided-crossing doublet so the argmax FLIPS
  between its ~99/~117 Hz partners over long integration (0.6–1.5 s → 117, 2.0 s → 99). A **near-nut
  pluck (0.137 L)** — the core diagnostic's recipe — reads f1 robustly at EVERY duration (free 117,
  supported 97). Fix: `_platebody_terminus_f1` is a **separate short 0.6 s near-nut-pluck sub-run**,
  decoupled from `pluck_position`/`audio_duration`; a `duration-robust` test pins free still overshoots
  at 2 s. *Generalizable: a headline number that varies with a user control or run length is not
  measured — probe it under fixed controlled conditions, and eyeball the PNG at the DEFAULT params,
  not just the fast test.* See [[respect-ruff-line-length]] (the fast test hid it).
- **The OPPOSITE terminus story per boundary (matched to the core diagnostic):** supported lands NEAR
  c/2L (~97, the pinned soundboard is a near-rigid termination); FREE OVERSHOOTS it (~117, the floating
  plate loads the end as a reactive mass-spring, its {1,x,y} rigid modes act mass-like). Batch-12's
  single "toward clamped, asymptotes below 100" text would be WRONG for the free edge — each boundary's
  readout branches on `sp.boundary`.
- **The ω² sanity denominator for a DISTRIBUTED body is the VOLUME displacement, not the driving
  point.** Batch-12's `w_b` (scalar modal displacement) does NOT transfer: `Q″` is the area-weighted
  volume acceleration, so `|Q″|/|w_dp|` gives ~0.4, not ~1. The right near-tautological denominator is
  `Q_vol = h²Σuᵢ` (supported) / `Σ Wᵢᵢ uᵢ` (free), for which `Q″ = Q_vol″` exactly → 0.997. Reused
  `_body_omega2_consistency` by feeding it `Q_vol`, not `w_dp`. *Generalizable: when a batch-12
  read-out moves lumped→distributed, re-derive what each spectral ratio actually relates.*
- **The guard ceiling K_c ≈ 13,968 N/m is the SAME for both boundaries — the advisor's "differs per
  boundary" worry was empirically FALSE.** The exact Sherman-Morrison margin is dominated by the STRING
  end-node term; the interior driving point has `W_dp=h²` either way. So `bridge_stiffness` is ONE
  range `[0,12000]` (default 3000, the core's own default) for both — no per-boundary re-range needed.
  K_c SHRINKS with `n_plate` (margin@3k 0.16→0.29 for N=8→24), so a high-N × high-K corner trips the
  exact guard → clean construction-error payload (the designed behaviour; tested both boundaries).
- **Energy slosh big on both, biggest on the free cymbal** (E_plate peaks 77 % supported / 83 % free at
  K=3000, up to ~90 % near K_c); drift ~6e-14 = the σ=0 verdict while E_string is NOT conserved.
  `sigma_plate` gates the verdict (0 → drift, >0 → passivity, decay_oracle=False — off-modal multi-rate,
  measured, the batch-12 read). New params `n_plate`/`sigma_plate` reset in `_default` (leak family).
- **Verified:** full backend suite green (+12 platebody web tests: the OPPOSITE-per-boundary straddle,
  duration-robust terminus, K=0 bit-identity anchor, 2D-heatmap-masked-and-ringing, guard battery);
  both fresh-load PNGs eyeballed; CDP **switch-check 12/12** (`temp/platebody_switch_check.py` —
  jawari→platebody re-ranges K to the guard-safe [0,12000]/3000, free↔supported both render ok,
  sigma_body stays hidden, platebody→jawari restores the 2e6 leak-reset, body→platebody re-ranges).
  Stale-server reprise: a prior-session `web/server.py` (old code) answered "unknown model 'platebody'"
  — identified before killing per [[identify-processes-before-killing]] / [[port-reclaim-modus-operandi]].
  This closes the distributed-body sibling of [[body-bridge-state]]; the remaining follow-on is
  `RadiatedBody` (the radiation LOAD / back-reaction, [[radiation-state]]).

---

**Batch 15 (2026-07-27) — the air pushes BACK: `RadiatedBody`, the BOOKED radiation channel.** The
last member of the body/radiation family and the one thing batches 12–13 structurally COULD NOT
show: there `AirRadiation` is a pure read-out that takes nothing (it deliberately has no
`energy()`), here the air is a LOAD that removes `P_rad = R·U²` and books it. `RadiatedBody` is a
`ModalBody` drop-in ⟹ batch 12's rig with ONE substitution, zero core and zero bridge edits, so
every difference on screen is the back-reaction. Model key `radbody` (its own key, batch 13's
`platebody` precedent — batch 12's 15 tests untouched). See [[radiation-state]] and
[[body-bridge-state]]. Load-bearing, all MEASURED (`temp/radbody-probe`, probes 1–6) — three of them
advisor gates that had to clear BEFORE any panel code:

- **THE FLAGGED COUPLING-ORDER HAZARD IS EMPIRICALLY FALSE — and it was the blocking gate.**
  [[radiation-state]] records the standalone identity at 7.86e-14 but the full chain only as
  "< 1e-9", while the viewer's badge fires at 1e-10; and `StringBodyBridge` precomputes `beta_b` from
  the BARE body and commits its spring force BEFORE the rank-1 correction lands, so an `O(k·R·U)`
  mismatch growing with R and K was plausible — exactly where the demo lives. Measured over
  R∈{0…20000} × K∈{0,8000,19000}: drift **3.9e-15…6.8e-14, no trend**. The "< 1e-9" was a test
  TOLERANCE, not the number. *Generalizable: a recorded tolerance is not a measurement — re-measure
  before redesigning a verdict around it.*
- **`sigma_zero` stays `(sigma_body == 0)` REGARDLESS of R** (the bore's booked-bell precedent, 2nd
  customer): `R > 0` is a CHANNEL, not a loss — `bridge.energy()` already holds
  `body.energy() = E_body + ∫P_rad` — so radiating and conserving are the same run (drift 1.60e-14
  while 98.4 % leaves as sound). Written into the docstring so nobody "fixes" it into passivity.
- **THE OPTIMUM HAS A NUMERICAL IMPOSTOR, and the discriminator is N.** t50-vs-R has a real minimum
  (**20.0 ms at R≈5.16**, basin **2.29–7.74**, vs 252 ms at the default R=133 and 1.49 s at R=800):
  past the match the air CHOKES the body it drains (`U = U_free/(1+RG)` ⟹ `P = R·U² → 0`). But the
  scheme's own turnover sits at `R = 1/G`, `G ∝ k` ⟹ `1/G ∝ fs`, which moves **111→222→444** across
  N=50/100/200. **The measured curve does not move**: t50 at R=3 is **24.48/24.52/24.57 ms** over a
  4× refinement. It tracks the PHYSICAL `σ_rad/ω₁ = a²R/(2mω₁)` (= R/27.6 here) instead. Also
  K-invariant (R=4 at K=2000…19000) and amplitude-invariant BIT-EXACTLY (linear model). Pinned as a
  test, not asserted. *Generalizable: when a scheme has its own scale near the physical one, refine
  the grid — the artifact moves with fs, the physics does not.*
- **t50 is the observable BECAUSE "fraction radiated by time T" is window-dependent** — its peak
  walks **R = 3 → 10 → 30** as T goes 0.05→0.4 s, and at σ=0 the fraction saturates toward 1 for any
  R>0. Batch 14's trap exactly. A rate is window-free; points that never reach 50 % inside the cap
  are LABELLED censored, never plotted.
- **The sweep is a CONTROLLED reference curve** (fixed N=100, σ_body=0, the user's K/string/pluck).
  A fixed N is legitimate only because the curve is N-independent — but it cannot be an ARBITRARY
  one: the coupled guard ceiling SHRINKS with fs (K_c = 8563/10716/21479/34394 at N=40/50/100/160),
  so a sweep BELOW the render's N could trip a guard the render passed. N=100 clears the whole
  `bridge_stiffness` range (19000 < 21479). σ_body=0 is required too — on a lossy body t50 censors
  nearly everywhere and the sweep both lies and costs 7× more.
- **Cheap because it EXITS EARLY** (~48k steps / 1.4 s at the default; 120k at the soft-coupling
  corner K=1000). Worst case is **K=0**, where nothing ever reaches the body (radiated energy after
  0.3 s is *exactly* 0.0) ⟹ K=0 **SKIPS** the sweep with a label rather than burning 320k steps
  drawing NaN. Per-step cost is **N-INDEPENDENT** (28.8 µs bare / 56.8 µs instrumented at N=100 AND
  160 — Python-level bridge overhead, not the grid) ⟹ the budget is steps alone.
- **Default R = 133 = the free-space monopole `R_a` at the first body mode**, which is the ONE
  frequency where the load and the batch-1 read-out are consistent — the panel prints it
  (`f_match = √(4πc₀R/ρ₀)/2π` = 109.8 Hz). HONESTY: R is constant in frequency while true `R_a ∝ ω²`
  spans 133/424/751/2135 over the rig's modes, so one R cannot fit all four; said with numbers, not
  a disclaimer.
- **The load KILLS batch 12's slosh, and that IS the finding** (peak E_body 0.773→0.491→0.195→0.072
  →0.0010 at R=0/1/4/10/133): loaded by the air the body stops being a RESERVOIR and becomes a
  CONDUIT. Slosh visibility and drain legibility are anti-correlated (both ride σ_rad), so no single
  R shows both — lead with the drain and let the SLIDER restore the slosh (bit-identically, at R=0).
- **TWO BUGS ONLY THE RENDERED PAGE COULD CATCH.** (1) **A param with a server-side DEFAULT hides
  its own missing control**: `radiation_R`'s markup landed NESTED inside the `distance` slider's div
  (an insertion that reused the opening tag and orphaned its `</div>`), so `buildSliders` never made
  it — and the fresh-load PNG rendered a *flawless* R=133 panel, because `gatherParams` shipped no
  `radiation_R` and the server applied `RADBODY_R_DEFAULT`. Only enumerating the built `.slider`
  params over CDP saw it. *Assert the CONTROL exists, not just that the render is right.*
  (2) **A leftover headless Chrome answers `/json` and hands you ITS page** — the stale-server trap
  one level over: three runs "confirmed" the missing slider in a browser holding the PRE-FIX DOM,
  which is also why the fix appeared not to work. The driver now proves the attached DOM matches
  disk before asserting anything, uses a fresh `mkdtemp` profile per run, and needs
  `--remote-allow-origins=*`. Related to [[identify-processes-before-killing]] — 30 leftover Chrome
  children, all mine, identified by their `radbody-chrome` / `chrome-verify-*` profile dirs first.
- **Panel craft:** `∫P_rad` was first drawn TEAL, which reads as the green TOTAL at a glance — the
  one pair that must never be confused, since the entire claim is one filling up to the other (now
  amber). A centred x-axis title lands exactly on the `100` decade tick and a rotated y-axis title on
  the `300` tick ⟹ units moved to the empty corners. `body_frac_peak` needs a sub-1 % format or the
  conduit finding prints a bare "0 %". `drawBodyEnergy`'s radbody branch RETURNS early (the fret's
  else-branch-lies lesson): the body wording calls a conserving radiating run a slosh.
- **`_RadBodyRun` deliberately does NOT subclass `_BodyRun`** — inheriting batch 12's telemetry meant
  three extra calls (`wb`/`qaccel`/`u_end`) on every step of a step-count-bound model, to fill arrays
  feeding the far-field spectrum panel this batch REPLACES with the sweep.
- **Verified:** 17 web tests (307 web / **1100 suite** green), ruff clean, verifier `radbody` PASS,
  CDP switch-check **17/17** (jawari↔radbody re-ranges `bridge_stiffness` both ways, `radiation_R`
  recomputes with no latch, and the R=0 / K=0 render branches say the right thing), both panel crops
  eyeballed. Default render **2.93 s**. **This closes the body/radiation family** — going further
  needs core work: frequency-dependent `R(ω)` or a distributed/3-D air box (HANDOFF §12H).

---

**Batch 16 (2026-07-27) — the parametric instability: model #9's SECOND refusal, discharged.** The
last batch named in the plan's own backlog, so this **closes Phase D's model list**: every core
resonator class except `FreeBeam` now has a viewer surface. `_build_payload_tension` refuses above
`TENSION_DT_MAX` with *"the breakup is real, energy-conserving physics; it wants a panel of its
own"* — this is that panel, and **the refusal was NOT relaxed**. It is the **IN-PLANE** modal
exchange: the same `2ω` pump batch 3's whirl aims at the other polarization, and exactly what batch
1's purity gate was protecting the Duffing oracle from. See [[tension-string-state]]. Load-bearing,
all MEASURED first (`temp/parametric-probe`, probes 1–6) — **three probes overturned an assumption,
two of them the advisor's**:

- **A REGIME of `tension`, not a model key** (advisor's structural catch, and the pattern holds):
  different resonator CLASS → own key (`radbody`/`platebody`); same class, different IC → REGIME
  (geometric's four). Batch 4's exact analogue — it discharged #9's *first* refusal as a regime of
  geometric. `tension` gained its first secondary select; core untouched, `_build_payload_tension`
  bit-for-bit (pinned by a test asserting absent-`domain` == `"duffing"` payload equality).
- **`m` HAD to become a control, and that decides whether the demo works at all.** The tension path
  is hardwired to `_mode1_shape` — and **m=1 is the ONE mode that does not do this** (no lower
  resonance partner). Measured thresholds in ΔT/T₀: m=1 (4, 6] (batch 1 independently got (4.44,
  6.05]); m=2 (3, 4]; **m=3 (1.75, 2.25]** = the default; m=4, m=5 both (1, 2]. Run length is in
  **PERIODS OF MODE m, never steps** — at fixed steps, raising N raises fs and the breakup slides
  off the panel.
- **THE PARTNER MOVES WITH ΔT/T₀ — a correction to [[tension-string-state]].** For m=3: **m=7** just
  above the edge (2–3), **4 and 11** deeper, **4 and 8** at ≈9. The recorded "m=3 → m=4 and m=8" was
  measured at ≈11 and is ONE TONGUE AMONG SEVERAL. The cascade panel reads partners off the run and
  names none in advance.
- **TWO SWEEP OBSERVABLES MEASURED AND REJECTED before the shipped one.** (1) the **fitted growth
  rate** (the advisor's prescription): clean only AT the edge (r²=1.000 at ΔT/T₀=2) and r²≈0.4 above
  3 with non-monotone rates — *physical*, not estimator choice: several tongues overlap and it
  saturates in ~5 periods, so no single exponential exists. (2) **`t_break`** (radbody's `t50`
  analogue — monotone, refinement-convergent 46.6/39.5/38.4 ms at N=100/150/200): the near-edge
  points **NEVER break at ANY cap** — ΔT/T₀=2.25 sits on **0.130 flat across 40/60/90 periods**,
  2.50→0.155, 2.75→0.190. They are **BOUNDED, not slow** (a partial exchange that degrades the mode
  without destroying it), so a fixed level is an arbitrary line through a continuum of plateaus.
  **SHIPPED: the SATURATED LEVEL** — window-free *because* it plateaus, with saturation verified per
  point chunk-to-chunk; unsettled points labelled, never reported as settled. Growth ratio jumps
  **1.1× → 4e5×** across the edge: that jump IS the sharp-threshold claim, on a log axis.
- **What is N-invariant and what is NOT, stated rather than glossed.** Above ΔT/T₀=2.25 both grids
  agree within ~10 % and the curve keeps its shape; the **EDGE does not hold still** — 2.00/2.00/
  2.25 at N=100/150/200, plus an **N=200-only narrow tongue at 1.75 that grows into partner m=10**
  where every other case goes to m=7. Real Mathieu structure (a tongue has edges on BOTH sides in
  detuning, so a partner enters *and leaves* resonance as amplitude rises). The recorded
  "refinement-invariant" was measured at ΔT/T₀≈11, deep in the MERGED region where it does hold.
  **Claim the convergent quantity; describe the edge as tongue structure.** The advisor had made
  two-N invariance the sweep's whole defence and conceded: *"that's a stronger panel than the one I
  described — it says something true that a naive version would have gotten wrong."*
- **"BELOW THRESHOLD" IS NOT A LINEAR CONTROL, and that is the stronger claim.** `nonlinear_fraction`
  at the IC (= the peak) is **0.42 at ΔT/T₀=1.5 vs 0.49 at 2.0** — tension 2.5× vs 3.0× rest;
  **the tongue is the entire difference**, and the panel says so with those numbers. But the panel
  first said "both runs are **equally** nonlinear" — true of the *probe* pair, and refuted by its
  own printed numbers at the *shipped* default (1.5 vs 3.0 → **42 % vs 59 %**). Ships as the weaker,
  always-true form. *A sentence measured at one operating point renders at another — re-read a claim
  against the numbers beside it, not against the probe it came from.*
- **NO new verdict type** (batch 3's precedent): nothing drives this string, so the ordinary σ=0
  drift check IS the claim — what makes it one is that drift stays **1e-15…1e-12 through complete
  disintegration** (off-mode 0.5) across ~150 probe runs. Gated on `n_not_converged == 0` (VK's
  Picard gate): a failed root-find looks EXACTLY like a breakup. σ is **FORCED to 0**, not defaulted
  — loss decays the amplitude back through the threshold and the instability self-extinguishes.
- **The seed: explicit, fixed, printed, MODAL.** Rate is seed-independent (**70.1/s at 1e-11, 1e-9,
  1e-6 alike, r²=1.000**; roundoff-only 68.4/s), so it costs nothing — and it makes the contrast
  honest, since "below is flat" would otherwise be partly "below was never seeded". Defined by modal
  content (modes 1..24, fixed `default_rng(12345)`), NOT `standard_normal(N+1)`, which spreads a
  fixed norm over twice the modes at 2N and confounds the two-N comparison. Self-seeding is pinned
  in a TEST (seed=0 still breaks up, still conserves) — which retires
  [[tension-string-state]]'s flagged BLAS risk, since nothing shipped depends on roundoff now.
- **A SHARED slider is only shared if it is BUILT in both places.** `dt_over_t0` is shared with the
  whirl by name (same physical coordinate; the whirl's 2.2 ceiling exists *because* of this
  instability) — but it lived inside the whirl's `data-show="geometric"` fieldset, so under tension
  it would never have been created while the server default filled in. **Batch 15's bug class by a
  different route.** Moved to the shared Nonlinearity box; both regimes name it in full in
  `MODEL_RANGES` and `_default` restores the whirl's 2.2 (a leaked 3.0 would trip the whirl's guard).
- **THE LEFTOVER-CHROME TRAP RECURRED — and batch 15's recorded fix DOES NOT CATCH IT.** That fix
  proves "the DOM matches disk", i.e. a **VERSION** marker; a leftover *from an earlier run of the
  same script* has identical markup and sails through. `terminate()` kills the launcher, not the
  browser tree, so runs 2 and 3 attached to run 1's page and read its final state back as two FAILs
  that were not real. Fix: kill own leftovers by `param-chrome-` profile prefix (21 processes
  IDENTIFIED before killing, per [[identify-processes-before-killing]]) + a **SESSION** marker —
  a pristine page has the markup defaults (`model=ideal`, `domain=""`). **"Has it rendered yet?" is
  NOT that marker**: autorender means a fresh page has already rendered by attach time.
  *Generalizable: to detect a stale attachment, assert something unique to THIS SESSION, not to this
  version of the code.*
- **ROUNDING A LOG-AXIS SERIES IN THE PAYLOAD DELETES THE CLAIM.** `_finite_list(drift, 12)` rounds
  to twelve DECIMAL PLACES, so a 1e-13 drift ships as a bare `0.0` and the flat machine-precision
  line — half the argument — silently vanishes. Both traces now ship unrounded (~425 pts each).
  *Round/format for display; never in the payload of a log-axis series.* Pinned by a test.
- **A partner that rounds to "0.0 %" is not a partner:** the naive `> grid_scale` filter printed
  "m=10 0.0 %" beside a real m=7 at 20 %. The bar is the **core signature test's own 100×** grid
  scale + `PARAM_PARTNER_FLOOR`, set at **1e-3 by the READOUT's resolution** (one decimal of a
  percent), not by a physics scale — *what is not worth printing is not worth naming, and that bar
  belongs in the payload, not the drawing code*. Below the tongue the list is EMPTY and `over_grid`
  is `null` (not 0.0, which reads as "at grid scale" rather than "no winner to price"). And
  `tension-hint` had to be gated to the Duffing regime — it reads `amplitude`, which this regime
  HIDES.
- **THE BACKEND DOES NOT HOT-RELOAD — a "regenerated" screenshot can be a PRE-FIX render.** Fixed
  the phantom, re-ran the verifier, re-read the PNG: **phantom still there** ⟹ chased a second bug
  that did not exist. The server had been up since before the fix; `serialize.py` imports ONCE. The
  false trail cost more than the bug. **Sibling of the leftover-Chrome trap, same shape**: fresh
  browser, fresh tab, STALE MODULE — a re-render only re-runs the code the server already holds.
  (And `terminate()` on the launcher leaves the Python process holding the port, exactly as it
  leaves Chrome alive — reclaim by PID.) Proved after the fact by rebuilding the payload IN-PROCESS
  with the old floor: identical, i.e. the first fix had always worked. *Backend edit → restart the
  server, or the picture is testing the previous commit.*
- **Panels:** headline = off-mode envelope + **the same run's energy drift on ONE log axis**
  (`plots.py:plot_mode_purity`'s viewer counterpart; `drawWhirl`/`drawSympathetic`'s third log-y
  customer). Wide canvas = the string visibly losing its m-lobe shape (dashed overlay is the
  **current** mode-m projection, not frame 0 — the amplitude breathes with the hardening) + the
  tongue sweep. Cascade rides the readout. Envelope is sliding-**MAX** (batch 3's whirl), not
  weinreich's mean.
- **Verified:** 29 web tests (336 web / **1129 suite** green), ruff clean, the FULL headless
  verifier **30/30** (not just the new case — `index.html` moved `dt_over_t0` out of the
  whirl's fieldset, and only a fresh deep-link to `geom_whirl` exercises that), CDP
  switch-check **29/29** (incl. the control-existence census over the
  62 built sliders, the whirl↔parametric re-range both ways, and all three straddle branches), PNG
  eyeballed. Default render **5.45 s**; per-step **200–290 µs** (the recorded "176 µs at N=128" was
  optimistic).


### Batch 17 (2026-08-10) — `airload`: the air as an IMPEDANCE, and the rule that unlocked it

**The "no batch 17" rule was SATISFIED, not bypassed.** It said a new viewer batch needs a new
*model* or *capability* first; the `R(ω)` fork arm delivered one (`RationalAirLoad` /
`ReactiveRadiatedBody`, [[radiation-state]] batch 3), so this batch surfaces it. The rule still
governs what comes next: the 3-D air box ([[air-box-state]], four core batches shipped) is the one
remaining built-but-unshown family and it needs a **3-D field type** the frontend does not have.

- **THE HEADLINE I HAD BACKWARDS, and the advisor caught it one edit from shipping.** I was about to
  write that batch 15 "confessed to a sin it never committed" — reasoning from its `R = 133` as a
  *saturated* resistance, which implies a 0.497 m sphere. Wrong: batch 15 set `R` from the
  **compact-source** law at 110 Hz, its sphere is the **5 cm** one, and its numbers were good to
  14 % over the band. The real finding is better and explains my own default: **the two batches'
  `R` are different physical quantities**, which is why the default moves 133 → 13146. *Check which
  LIMIT a number came from before claiming continuity between two batches that share a name.*
- **THE MONEY TEST IS THE LEDGER RESIDUAL, NOT THE DRIFT** — the air-box lesson's third customer,
  and the advisor asked for it. `load.energy()` returns stored PLUS radiated while `body.energy()`
  is bare, so a channel can be double-counted or dropped and the conserved total stays green:
  measured **2.30e-16**, against **3.6e-01** (sign-flipped stored) and **1.8e-01** (dropped), with
  drift unmoved at 4.7e-14 in both. *A conserved total cannot see a wrong PANEL.*
- **The anchor is a SLIDER SETTING that reaches batch 15 as shipped**, not merely as a mechanism:
  `air_corner = 0` + `radiation_R = 133` + `radiation_weight = 1` → `array_equal` on energy, audio
  and the radiated channel. Stronger than a within-batch anchor, and the user can dial it.
- **The RADIATION WEIGHT, not `R`, decides whether anything is visible.** Batch 15 inherited
  `a_i = phi_i = 1` from `ModalBody`'s default, putting the body at `α/ω₁ = 4.8` — past critical
  damping AND mode-coupled through the shared volume velocity, so a "per-mode" measurement returns
  a mixture (a 110 Hz mode reads 152 Hz) and `loaded_mode` diverges. The per-mode panel therefore
  had to be a **single-mode** rig, body alone, no string.
- **…and at a physical weight batch 15's own anti-correlation does NOT bind**: it measured that
  slosh and drain cannot both show, but weight 0.05 gives 97.4 % radiated AND 60.5 % slosh AND an
  18.1 % stored channel (in batch 15's rig that channel peaks at 0.7 % — invisible, which is why
  batch 15 could not have drawn it).
- **The pitch shift MUST be read against the SCHEME's unloaded frequency.** Against nominal `f0`
  the leapfrog's warping at the render's own rate outweighs the air and **flips the sign**
  (+0.55 % vs −0.65 % at 1760 Hz). So every sweep point pays for a second `R = 0` run, and the
  sweep is pinned at 48 kHz (oracle error converges 2.61 → 0.79 % as fs → ∞, so the floor is the
  FORMULA, not the scheme).
- **The oracle's disagreement IS its own stated order, with the coefficient measured**:
  `err/(α/ω)²` is flat at ≈16–21 across a 7× range. The panel says that instead of claiming
  agreement it does not have — the measurement is primary, the oracle is an overlay with a printed
  validity, and only 4/12 points sit inside `α/ω ≤ 0.02`.
- **`R = 0` RAISED** — the corner's own inversion `M_a = R/(2πf_c)` returns a *zero* mass there,
  which the core rejects, so the one value meaning "no air at all" was unreachable. *When you
  expose a DERIVED coordinate, check the endpoints of the one you derived it from.*
- **BATCH 15's INVISIBLE-CONTROL BUG WAS STILL LIVE — by a DIFFERENT mechanism than batch 15
  recorded, and the advisor caught me writing batch 15's version of it.** `pluck_position` lists
  `body platebody radbody` in its own `data-show` but the Excitation *fieldset* listed only `body`,
  and the fieldset gate wins. Batch 15's case: control never built → `gatherParams` ships nothing →
  the SERVER default fills in. **This one is the opposite:** `buildSliders` has **no visibility
  filter** and `updateVisibility` only sets `el.hidden`, so the control IS built and `gatherParams`
  (which iterates the built map) ships it every time — the user just **loses the knob**. Verified
  the worse reading is NOT live: all four body models carry `pluck_position: {val: 0.3}` and
  `applyModelRanges` re-asserts `val` on entry, so nothing leaks in from another model. It was
  silent only because the frontend's 0.3 and the backend's 0.3 agree. **Two gates, two lists, only
  the narrower one shows in a diff — and "hidden" here means shipped-but-unreachable, not absent.**
- **The switch-check's own readiness condition was the same class of bug**: waiting on
  `.slider` element count is satisfied *before any JS runs* (the divs ship in `index.html` empty),
  so the first run reported 64 visible sliders and no `<input>`s — three false failures, one false
  pass. Wait for a **built** `input#s-…`.
- **`E_air` renders as a BAND, not a curve, and that is the physics** — the stored energy oscillates
  at **2ω**. Left un-smoothed and said in the readout; an envelope would have hidden the one visual
  fact that distinguishes a reservoir from a drain.
- **Verified:** 21 web tests, ruff clean, headless verifier `airload` PASS beside
  body/platebody/radbody, CDP switch-check PASS (control census + the 44× `radiation_R` re-range
  resetting on switch-away), PNG eyeballed. Default render **3.84 s** (44 444 audio steps +
  a 44 710-step sweep — batch 15's sweep budget almost exactly).

**PHASE D IS CLOSED — the plan's "Later batches (rough map)" section is now EMPTY (verified
2026-07-27).** Every bullet in it resolves to a shipped batch (body/radiation → b12/b13/b15, wind →
b9/b10, excited strings → b2/b8/b11/b14, weinreich → b7, parametric demo → b16). So there was **no
batch 17 to look up** at that date, and none may be inferred from the document: a new viewer batch
needs a new *model* or a new *capability* to exist first. **That condition was later MET** — the
`R(ω)` arm shipped a new core model class and **batch 17 (`airload`, above) surfaces it, 2026-08-10**
— so the rule stands and that is what satisfying it looks like. The built-vs-shown gap that Phase D existed
to close is gone — every model the core has is surfaced. The candidate forks (recorded in
`web-viewer-plan.md` at the end of that section, so the doc and this memory agree): the two **core**
growths the plan itself names — frequency-dependent `R(ω)` (**TAKEN 2026-07-27: radiation batch 3,
built & green — see [[radiation-state]]**) and a distributed/3-D air box (HANDOFF §12H, still open,
the remaining half of that fork); HANDOFF **§11's two never-closed decisions** (#3 *which models are polyphonic* — a gate on
engine design, cheaper to settle before a port than after; #5 *test-tolerance tuning*, still on M1's
provisional bar per `ideal-string-plan.md`); **Phase 5** (the real-time C++/Rust port, HANDOFF §9's
literal next phase and the thing non-negotiable #1 deferred); or a §12 high-leverage thread
(port-Hamiltonian coupling, differentiable models, modal synthesis from measured IRs/FEM).
**This is a human fork, not a lookup** — surface the menu, don't pick (see the global rule on
ambiguous "continue"). Baseline at close: **1129 tests green**, 336 of them web.

**The model list is not closed once — it REOPENS.** It was declared closed in 2026-07, reopened by `R(ω)` (b17), by the 3-D air box (b18, [[airbox-viewer-state]]), and again by air-box batches 5–6, which b19 surfaces ([[vkroom-viewer-state]], 2026-08-17). So "built-but-unshown is closed" is a statement about a moment, not a steady state: every new core model reopens it, and the rule that a batch needs a new model or a new capability earns its keep by being re-applied rather than retired.

**Batch 20 (2026-08-26) closes it a FOURTH time** — model #5g's guitar outline, surfaced as the plate's third *domain* rather than a new model id ([[guitar-plate-viewer-state]]). Two corrections it forced on this plan: the amendment claiming #5g needed a new frontend capability (masked field rendering) was **false** — that shipped with the membrane; and the slider re-range gate was keyed on the *model*, so the first `plate:` regime would silently have started resetting sliders on a supported↔free switch the plan explicitly forbids.
