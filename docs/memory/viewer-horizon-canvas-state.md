---
name: viewer-horizon-canvas-state
description: "Viewer batch — the horizon drawn ON the diagnostic panel; a mark's POSITION comes from the payload and the panel's EXTENT from the display array, so off-panel draws NOTHING — and at the shipped default both 2-D panels are off-panel"
metadata: 
  node_type: memory
  type: project
  originSessionId: f56fe2ca-94ec-420c-a21c-1c4be27c87ab
  modified: 2026-09-07T17:43:27.190Z
---

2026-09-07. The resolution horizon now shades the second diagnostic panel as well as filling the
strip under the transport ([[viewer-horizon-readout-state]] shipped the strip). Plan §13 in
`M:\claud_projects\physical synthesis\docs\dev\resolution-horizon-plan.md`. No payload field and no
physics changed — three helpers in `web/static/app.js` plus call sites in `drawPartials`,
`drawSpectrum` and `drawVkSpectrum`.

**The rule, and it is the batch:** *where* the mark goes comes from `payload.horizon`; *how far the
panel reaches* comes from the display array; the two are never mixed. §12.5's trap was a horizon
**read off** a 12-partial display list (a fact about the list, not the scheme). This is the same
trap backwards — a mark clamped to the right edge of a panel the horizon sits beyond is that lie
*drawn* instead of computed. So **off-panel draws nothing** and puts the number in the readout.
Absence plus a sentence is the correct output, not a degraded one.

**The consequence nobody predicted: at the default 5¢ bound BOTH 2-D panels are off-panel.** The
mode-spectrum panel's range is set by the modes it draws (~1.6× the top one) — 637 Hz for the
membrane against an 866 Hz horizon, 392 Hz for the plate against 408 Hz. Nothing is wrong with
either number; the panel is scaled for a different job. So **the drawn mark is a 1-D feature at the
shipped default**, and the 2-D panels carry the claim as a sentence. Not fixed by rescaling the
panel — that changes what the panel is for.

Four more things worth keeping:

- **Each panel in its own units.** §12.3 warns the hertz reading does not license a mode count. The
  mirror image is what this batch had to get right: the 1-D partials axis **is** an index, so a
  frequency drawn on it is the same overstatement pointing the other way. Frequency axes get `hz`,
  the index axis gets the partial count, neither carries both.
- **The cents selector must redraw the PANEL, not just the strip.** Otherwise the strip's number
  moves while the shading stays put — the read-out contradicting itself on screen, which is worse
  than never having drawn it. `drawDiagnostics` is safe to re-call (it only reads and paints).
- **Gate on `horizon.kind === "prefix"`, never a model list.** The union arm already knows which
  scenes have a number, including ones the frontend never enumerated — the von Kármán panel draws
  with the nonlinearity off and refuses with it on, through the same call.
- **Four scenes have a measured horizon and NO axis that can express it** (bow, jawari, juari,
  fret — stick-slip, shimmer, tuning curve, raster). Drawing nothing there is correct, so a `null`
  mark is a pass; `window.__horizonMark` is reset at the top of `drawDiagnostics` or the previous
  render's decision reads as this one's.

**What is asserted is the decision, not the pixels.** Each panel records `{drawn, reason, at,
units}` on `window.__horizonMark` — the device `applyUrlSliders` already uses for
`__urlParamNotes`, and for the reason that one exists ([[web-viewer-state]]: batch 19's deep link
dropped every parameter for a whole batch because nothing failed when it did). The harness's
`_horizon_mark_ok` cross-checks it against the strip's wording; five reasons (`refused`,
`whole-grid`, `off-panel`, `zero-modes`, `drawn`). That vocabulary lives in **two files**, so one
pytest test derives both sides — the `markHorizon("…")` literals out of `app.js`, the reason set out
of the harness — and asserts they are the same five words.
