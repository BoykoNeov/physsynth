---
name: airbox-viewer-state
description: "Web viewer batch 18 `airbox` (2026-08-10) — the 3-D room surfaced; `dims:3` is a SLICE SET not a volume; the claim is the LATTICE LIGHT CONE (first nonzero at the Manhattan distance in CELLS — an exact integer, lambda-INDEPENDENT, beating both physical arrivals); money test = cross-ledger residual; cost is h^-4 because the 3-D CFL runs the WRONG WAY; the ROOM sets fs so the string's lambda is DERIVED; six build-time corrections incl. a TRANSPOSED slice decode that rendered as plausible banding and a NaN that only died at the JSON boundary"
metadata: 
  node_type: memory
  type: project
  originSessionId: 93fec4ea-5c8b-49a1-9019-d53f2d063a6d
  modified: 2026-08-10T14:31:44.693Z
---

Batch 18 of the web viewer ([[web-viewer-state]]), shipped 2026-08-10. Surfaces
[[air-box-state]]'s `AirBox` — the last built-but-unshown core family — as model key `airbox`,
running `string → bridge → RoomLoadedBody → AirBox`. All-wrapper; `physsynth/core` untouched.
13 web tests. Default render 8.6 s, payload 2.8–3.1 MB, 155 slice frames.

**The new capability is `dims: 3`, and it is a SLICE SET, not a volume.** Three named orthogonal
planes per frame (`xy`, `xz`, `yz`), each an ordinary decimated heatmap. A 32³ volume × 300 frames
is ~10M floats — dead on the payload trap; three slices at the 64-cap measure 5.26 MB base64. The
slice point is a **server-side** re-render, because a client-side movable slice is exactly the thing
that requires shipping the volume.

**The claim is the lattice light cone — an INDEX, never a magnitude** (batch 2's rule: the room
contaminates a magnitude by more than the effect). On a 7-point Yee stencil the numerical domain of
dependence after n steps is `|di|+|dj|+|dk| ≤ n` **cells**, so a mic's first nonzero lands exactly
on the **Manhattan distance in cells**. Measured 41/41 off-axis and 21/21 on-axis in the core probe,
46/46 through the coupled wrapper, and 11/24/35/46/58 as the mic slides — always matching. It is
**λ-independent** (cone 46 at cfl 0.4/0.6/0.9, fs 22k–49.5k), which is what lets the default sit
below the 1/√3 ceiling with **no caveat on the claim**. It beats both physical arrivals: Euclidean
52.8 steps, walking the Manhattan path at c₀ 88.5. The amplitude-threshold arrivals beside it DO
move with λ — that contrast is the panel.

**The money test is the CROSS-LEDGER residual (6.89e-18), not the drift (2.49e-14).** Third and
fourth customers for this already existed ([[air-box-state]] batches 3/4, [[web-viewer-state]] b17).
`RoomLoadedBody.energy` folds the coupling in so it **cancels out of the scene total identically** —
the total is structurally blind to a wrong coupling constant. Gate on `|radiated − injected|`.

**Cost is `h⁻⁴`, and the room sets `fs` for the whole scene.** The 3-D CFL runs the wrong way:
`λ = c₀k/h ≤ 1/√3` means a **coarser** grid forces a **higher** sample rate, so refining buys `h⁻³`
nodes *and* `h⁻¹` steps (measured ×14.2 then ×15.4 per halving vs the predicted 16). Throughput
~5e7 node-steps/s; the coupled chain ~14 s of wall-clock per simulated second. Because
`RoomLoadedBody` refuses a sample-rate mismatch, **the string's λ becomes a DERIVED read-out**, the
reverse of every other model here — and N is bounded by the room (~110 at h = 0.03, ~66 at 0.05).
At the top of the `air_cfl` slider the binding constraint is the **bridge** guard, not the air's CFL.

## Traps this batch paid for

- **A TRANSPOSED slice decode renders as plausible physics, not as an error.** The backend ravels
  shape `(nu, nv)` in C order so **`nv` is fastest**; reading u-fastest produced smooth horizontal
  **banding**. The bytes were correct, so no payload assertion could see it. What caught it was
  arithmetic — at 110–440 Hz the wavelength is ~3 m against a 1.2 m room, so the field *must* be
  smooth. This is why the headless/browser case earns its cost.
- **A NaN builds a perfect payload and dies at the transport.** `zeta` was `float("nan")` on
  non-absorbing walls and `web/server.py` dumps with `allow_nan=False`: valid dict in-process,
  500 in the browser. *An in-process serializer test structurally cannot see a serialization bug* —
  assert `json.dumps(..., allow_nan=False)`.
- **The animation clock is the ROOM's, not the string's.** Striding on the string's `f1` gave 18
  frames and showed the 3.5 ms crossing in ~4 of them; one frame per cell of travel gives 155.
  Catch #2's "resolve the fundamental" means *the oscillator you are drawing*.
- **`open` walls (Z = 0) are LOSSLESS** — pressure-release, reflects with inversion, dissipates
  exactly 0.0. Classing them lossy exempts a conservative scene from the 1e-10 bar, i.e. hides the
  bug the bar exists to catch. Key the verdict off the token, not the printed label.
- **Time the cone from the first nonzero INJECTION, not step 0** — the string is plucked in
  displacement, so the body/port starts at rest and first injects at step 1.
- Seven whole-room energy sums per step cost ~3 s: sample the mic/port every step (the cone is an
  exact integer a decimated trace destroys), the five channels every `e_stride`, and **ship
  `e_stride`** so the bound is readable rather than implied.
- **Backgrounded servers kept dying**; `Start-Process ... -PassThru` detached survives longer but
  still gets reaped between tool calls — start the server and use it in ONE shell invocation. See
  [[port-reclaim-modus-operandi]] and [[identify-processes-before-killing]].
- **`scripts/verify_web_headless.py` could NOT run here**: it launches its own Chrome on the
  DevTools port and cannot reach that endpoint while another Chrome is already running (the b16
  leftover-Chrome trap again). Its `urlopen(BASE, timeout=2)` reachability probe is also marginal
  against a cold server on this machine — `timeout=5` succeeds where 2 fails. The two airbox cases
  are ADDED but UNEXERCISED by the script; equivalent ground was covered over CDP instead, and both
  the plan and the commit say so rather than implying a check that did not run.

## Died in probing, before a line was written

Four plan claims: that the echo and the directivity null could share a run (opposite rigs — one
needs the reflection after its window, the other needs it to BE the point); that a coupled
`RoomSuspendedPlate` could carry the 85× null (that is a *prescribed-velocity* figure, and a free
plate's fundamental is the saddle/twist quadrupole — see [[free-plate-state]]); that directivity
could be audio-bearing (4.1 s for a 200-step burst, but 74 s for 0.3 s of audio in the same 5 m
cube, so it can only ship decoupled, the [[juari-state]] precedent); and my own first echo probe,
whose **axis-aligned mic made the two candidate answers equal** so it never tested its claim.
