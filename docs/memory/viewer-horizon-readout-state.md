---
name: viewer-horizon-readout-state
description: "The viewer's trust read-out ships — and the surprise is how much of the viewer has to REFUSE; the discriminator for a string is the FREE end, and hertz needs a different computation from index"
metadata: 
  node_type: memory
  type: project
  originSessionId: 730f094f-dd92-4f51-ab1a-54f12b18852a
  modified: 2026-09-07T16:40:17.164Z
---

The user-facing surface of [[resolution-horizon-state]] and [[horizon-promotion-state]], built
2026-09-07: every payload from `M:\claud_projects\physical synthesis\web\serialize.py` carries a
`horizon` block and the page draws it as a strip under the transport (badge + sentence + a 1/5/25
cent selector). Record: `M:\claud_projects\physical synthesis\docs\dev\resolution-horizon-plan.md`
§12; 19 tests in `tests\test_web_backend.py`; `scripts\verify_web_headless.py` now fails a case
whose strip is hidden. No physics was written — every number comes from
`physsynth.analysis.horizon` + `modal`.

**The open decision (§11.8) closed as: the WORST member of the chain, and any member with no horizon
refuses for the WHOLE scene.** A modal body has no spatial discretisation error so it never limits;
only grid resonators can. Quoting the string's number for a string-plus-nonlinear-plate scene "with
a caveat" is how a read-out becomes a lie.

**Most of the viewer refuses, and that is the batch's real content.** Eleven model keys measure
(three strings, bow, jawari, juari, fret, membrane, mallet, plate, linear VK); eleven refuse. The
1-D discriminator was a surprise worth keeping: the θ-strings report `boundary == "supported"` and
the ideal string `"fixed"` — **both are the sine series the modal oracles are derived for**, so a
naive `== "fixed"` check refuses six models that are fine (it did, on the first run). The refusal
that matters is the **free end**: every bridge-coupled scene builds
`IdealString(boundary=("fixed","free"))`, and that is a *physics* refusal, not a missing-oracle one
— the bridge shifts the continuum partials, so a cents comparison would report the coupling as a
discretisation error. Two errors in one number, same reason a staircased outline is refused.

**Hertz and index are two different computations, and only one of them may use `pitch_horizon`.**
The frequency ceiling is a **first failure over the exhaustive sorted spectrum** — "nothing below
361 Hz is out by more than 5 cents" stays true whether or not the error curve is monotone, which is
exactly what plan §8.6 says a prefix over the sorted 2-D union does not. The block index is
`pitch_horizon` along the **corner families** (all three asked, minimum taken — no branch on the
regime, so it is right for the plate's diagonal corner and the membrane's axial one alike). The two
differ by a factor of four on a membrane at 25 cents, which is why both ship.

Four things measured rather than assumed:

- **A horizon read off a display array is a fact about the array.** `N_PARTIALS=12` and
  `N_PLATE_MODES=6` were already in every builder; the mode set is built from the dispersion
  relation over the whole grid instead (`1..N-1`, or the full `(m,n)`), and costs **1.1 ms** for a
  1999-mode string and **16.7 ms** for 98×98 against renders of seconds.
- **`horizon == 0` is live on a shipped fixture**: the viewer's own linear von Kármán default is 7.4
  cents flat at `(1,1)` and has no horizon at 1 or 5 cents. `hz` is `None` there — `0.0` reads as a
  measurement and NaN is refused by `server.py`'s `allow_nan=False`.
- **A tie between corner families is the COMMON case here**, and it is coarser than the degeneracy
  §10.6 warns about: three families whose prefixes land on the same small integer. Reported as a tie
  (`family_tied`, plus every family's own index). A **non-square** rectangle breaks it for a real
  reason — `Ly=0.8` puts `(1,n)` two modes short of `(m,1)`, §9.5's index-versus-hertz point
  arriving through geometry instead of a grain.
- **The model population is derived from the `<select>` in `index.html`**, and the two tables
  (builder-computed, `HORIZON_ABSENT`) must partition it — same move as the migration's ledger #67.

Also: the JS bug worth not repeating — the tied family names were joined by **splitting the joined
string on `", "`**, and the names contain their own commas, so "axial (m, 1)" printed as "axial (m
and 1)". Build a list from the rows, never by re-splitting a rendered string.

Related: [[resolution-horizon-state]], [[horizon-promotion-state]], [[web-viewer-state]],
[[horizon-membrane-block-state]], [[plate-mode-family-split-state]], [[viewer-stays-python]].
