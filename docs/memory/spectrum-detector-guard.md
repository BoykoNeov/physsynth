---
name: spectrum-detector-guard
description: measure_partials_near could return a NEGATIVE frequency — the refiner was fed a WINDOW-EDGE argmax; the guard is local-max, NOT sign-of-curvature, and 15 test files leaned on this untested primitive
metadata:
  type: project
---

`physsynth/analysis/spectrum.py` — fixed 2026-08-26, found while sweeping the guitar viewer's
geometry grid (see [[guitar-plate-viewer-state]]; the sweep's `log2` warning had **two** producers,
the mislabelled continuum oracle and this).

**The bug.** `measure_partials_near` takes its argmax *inside a search window*. A window whose
magnitudes fall monotonically puts that argmax on the window's own **edge** — a bin with no peak
under it. `_parabolic_refine` assumed a peak, fitted a parabola through three near-collinear
log-magnitudes, got curvature `-7.9e-3`, and "refined" by **-22.1 bins** from bin 4 → **-502 Hz**.
That fed `modal.cents` → NaN → `json.dumps` refuses → a 500 in the viewer. But the frequency was
already wrong before anything serialised it.

**The trap, and it is the whole lesson.** Both real witnesses had **negative** curvature — a
sign-of-`denom` guard catches *neither*. `a=3, b=2, c=0` is concave, decreasing, and still moves the
estimate 1.5 bins. The property that makes parabolic interpolation meaningful, and the one that
bounds `|delta| <= 1/2`, is that the bin is a genuine **local maximum**. Guard on that; a non-peak
bin gets its bin centre.

**The negative number is the loud version, not the bug.** The same edge argmax on a *rising* slope
extrapolates **+7 bins** onto the peak just *outside* the window: 137 Hz probe → 191.5 Hz. Positive,
finite, plausible, and the wrong partial. A `> 0` sanity check would never see it. One real viewer
case moved 440.58 → 226.76 Hz for exactly this reason.

**Why it survived.** **Fifteen** test files depend on this function and **not one tested it** — they
all assert `abs(found - oracle) < tol`, which is a claim about the *model*. A detector that is wrong
where no model looks satisfies every one of them. `tests/test_spectrum_detector.py` now tests the
primitive directly.

**How to verify a change to a shared measurement primitive.** "The suite still passes" is satisfied
by a small shift. Record every call's return as `float.hex()`, keyed by nodeid + call ordinal, run
the dependent files **before and after**, and diff for exact equality — a pytest plugin that
monkeypatches the function does it in ~13 min. Same discipline as the one-grid bit-identity scar in
[[guitar-plate-viewer-state]].

**Do not** fix this class by clamping `audio_duration`'s lower bound: short renders are legitimate
(the viewer's own animation windows are milliseconds), and clamping hides one instance while leaving
the primitive broken for every other short window.
