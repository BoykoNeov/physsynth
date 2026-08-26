---
name: vkroom-viewer-state
description: "Viewer batch 19 `vkroom` — the gong/cymbal in the 3-D room; the claim's SEPARATION survives but its MAGNITUDE is not grid-converged, a hidden Chrome tab makes a correct field read blank, and the deep link silently dropped every parameter but model/domain"
metadata: 
  node_type: memory
  type: project
  originSessionId: e2ff4425-287c-4369-8e95-2369b0c59a80
  modified: 2026-08-17T20:24:54.713Z
---

Viewer batch 19 (2026-08-17) surfaces air-box batch 6's `RoomLoadedVKPlate` / `RoomSuspendedVKPlate`
as model key **`vkroom`** — all-wrapper, core untouched, 12 web tests. See [[web-viewer-state]],
[[airbox-viewer-state]], [[air-box-state]], [[von-karman-plate-state]].

**The rig, and it is pinned by stability not taste.** Free-edge plate 0.10 m / N=16 in a **0.35 m
cube** at `h = 11.40 mm`, `fs = 57.9 kHz`, 0.12 s, `couple_max_iter = 120` (the core's 50 caps out;
72 measured). Along the CFL line `h = 14 mm` stops converging and `h >= 17 mm` is an immediate NaN,
so the shipped `airbox` room's 22 kHz **cannot host this plate at any size**.

**The finding that generalizes furthest: batch 6's "coarsening the room breaks the plate's fixed
point" is about coarsening ALONG THE CFL LINE.** Hold `fs` fixed and lower the Courant fraction
instead and the fixed point is untouched — identical 72/5 sweeps at 0.90/0.60/0.45 — while nodes
fall as `(L·λ)³`. What it breaks instead is the **claim**: the resolved mode count collapses
17 → 8 → 6 of 289 and the resolved-band spread goes to 1.000×. A **stability** limit and a
**validity** limit are not interchangeable; the second leaves every ledger green.

**THE MAGNITUDE IS NOT GRID-CONVERGED — quote the separation, never the multiplier.** Refining the
air cell 12 % takes the struck arm's resolved spread 7.04 % → 1.99 % while the linear twin holds at
0.05–0.12 %; a **0.1 %** change in `h` alone moves it 7.04 % → 2.78 %. The separation is robust
(17–102× everywhere measured). Two reasons underneath it, and the second is a *statistics* bug not a
physics one: the spread is **max/min of four near-equal numbers**, so two rigs whose per-window
values agree to ~1 % differ by 33 % in the quoted ratio. **The obvious confound was checked and does
not explain it**: refining `h` moves the resolved SET (17/19/19/21), but the modal basis is
grid-INDEPENDENT (`vecs` and `freqs` bit-identical at every `h`, only the mask moves), so pinning the
17-mode mask and reusing it gives 6.99/4.99/2.37/2.02 % - the same series - against a control held at
0.08/0.05 %. *"Quote the resolved band" is necessary and
not sufficient — a resolution-restricted observable can still be a function of the resolution.*

**The control is the `nonlinear` flag at the SAME amplitude**, and it lands on the quiet arm's value
to three decimals — so the effect is the nonlinearity, not the amplitude. It is also the only catch
for batch 6's `rho_v`/`rho_s` trap.

**Scars worth carrying.**
- **A hidden Chrome tab makes a correct field read as a blank canvas**: `requestAnimationFrame` does
  not fire in a background tab (0 ticks in 1.2 s), so the canvas measured 0.0 % painted with no
  console error while the same draw called directly painted 123,904 pixels. Drive the draw directly
  or foreground the tab. This is [[airbox-viewer-state]]'s verification hazard inverted — there a
  leftover Chrome blocked the script; here the browser runs and throttles the thing under test.
- **A free plate's fundamental is the FOURTH eigenvalue.** `{1, x, y}` come out of `eigh` at ~1e-5
  Hz, not 0, so "first frequency above epsilon" picks a rigid mode — the stride landed 3× past the
  window and the pane rendered ONE frame. Use the nullspace dimension, which is a fact (model #5b).
- **The room nearly shipped as dead payload** — built, serialized, drawn by nothing, i.e. the
  built-but-unshown gap being created inside the batch that closes it.
- **The defaults sit on a CLIFF**: at width 0.20, `w/e` 3.0 runs at 72 sweeps, 3.2 at 109, 3.4 is
  dead. DIED — "the two sliders are one knob": the curvature ratio `w/e ÷ width²` fits both
  brackets at 83–85 and fails off the diagonal (at ratio 75.3, `w/e` 1 and 2 run, 5 and 8 die).
  **Amplitude binds; a broader strike does not buy it back.**
- **The supported gong is REFUSED with numbers, not for stability** — it runs, but resolves 6 of 225
  modes, where the separation collapses to 3.8× baffled and runs *backwards* suspended (0.3×).
- **The local machine's wall-clock noise floor is 1.88×** on an unchanged interleaved reference,
  which forced a whole duration sweep to be thrown away and re-measured as a rate.
  [[ci-runner-variance]]'s rule is not a CI fact.

**The three loose ends, closed 2026-08-17 — and TWO of them close as measurements, not code.**

- **THE DEEP LINK DROPPED EVERY PARAMETER EXCEPT `model` AND `domain`, for the whole of Phase D.**
  Found only by actually running the never-run cases: `?model=vkroom&audio_duration=0.02` was
  ignored, so both cases paid the 0.12 s default (~23 s, not the ~7 s their comment claimed) and
  never exercised the short run. **Nothing failed** — a deep link that ignores what it is handed
  reads exactly like one that honoured it. Fixed as `applyUrlSliders()`, which MUST run *after*
  `applyModelRanges()` (that rewrites `val` per model), takes **backend units** (undoes
  `data-scale`), and records unknown names + clamps/snaps in `window.__urlParamNotes` — now part of
  the harness verdict, so it cannot recur silently. The near-miss in fixing it: `setSlider` does
  not fire `onControlChange`, so the rig panel could have said 6,948 steps while the run was 1,158;
  it does not, because `updateLambdaHint()` already runs after the new call.
- **The never-run gap was TWO batches wide.** No `viewer_airbox*.png` existed at all, so
  [[airbox-viewer-state]]'s pair had never run either — a file listing, not b19's inference. All
  five cases pass now. The dual pane paints into the **single** `#string` canvas, so the probe was
  correctly targeted — worth establishing *before* trusting a PASS from it.
- **`Popen`-launched headless Chrome exits 0 with the port never opening** (plain, `DETACHED_PROCESS`
  and `DETACHED|NEW_PROCESS_GROUP` all measured), while the same command line from a shell comes up
  in 3.6 s. Cause NOT established, and not a machine fact (the 2026-08-10 `airload` screenshot
  proves the path worked). The fix is **attach to an already-listening Chrome** — which inverts
  b18's leftover-Chrome scar into the supported path. Use a FRESH profile after editing
  `web/static/*` or you get a cached-file PASS that proves nothing.
- **THE SHAPE DRIFT IS NOT IN THE PICTURE, so the animation window stays at 4 ms.** The per-window
  mean-square velocity map — the spatial pattern `sigma_shape` is a functional of — correlates
  **0.958** between window 1 and window 4 (TV 0.127) against **1.0000 / 0.0035** for the linear
  twin. Real, ~36× the control, and **96 % unchanged to the eye**: the 46 % modal-share drift is
  energy moving between modes with near-identical mean-square footprints. Widening the window would
  have cost ~490 kB and **frozen the room half** (87 frames on the wavefront's clock, stuck on its
  last frame for 98 % of the plate's animation). *Measure whether an effect is visible before
  building the thing that would show it.*
- **The claim is ALIVE at the coarsest legal `plate_N = 8`** (63.3× drift separation), and clears
  the suite's 20× bar at **every** legal setting (63/35/94/43/66 at 8/10/12/14/16). The **spread**
  separation is non-monotone over **11–102×**, which *widens the quoted range* rather than refuting
  the "never below ~17×" floor — a ratio of two max/min spreads is worse conditioned than either,
  so do not correct a shipped headline with a number weaker than the one being corrected. `N = 16`
  reproduces the shipped 7.045 % / 102× exactly, which is what makes the other rows readable.
