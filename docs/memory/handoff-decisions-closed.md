---
name: handoff-decisions-closed
description: "HANDOFF §11's five \"ask the human\" decisions are ALL CLOSED (2026-08-10) — three had been de-facto answered by practice for months; polyphony = per-instance (fields) vs per-voice (strings); tolerances STAND because the 1e-10 bar is the native port's acceptance contract, not slack"
metadata: 
  node_type: memory
  type: project
  originSessionId: 8f723436-aed6-417e-9342-6bbda0f8d562
  modified: 2026-08-10T07:24:04.358Z
---

**HANDOFF §11 is closed — all five — as of 2026-08-10.** The human took this arm of the post-Phase-D
fork right after [[air-box-state]] batch 4 shipped. Doc-only batch: `HANDOFF.md` (§3.2, §6.1 new,
§8, §11 rewritten) and `CLAUDE.md`. **Zero code, zero test edits** — that was the decision, not a
shortcut.

**Three of the five had been answered by practice and nobody had written it down.** #1 Python
(milestone 1), #2 explicit *and* implicit (the θ-scheme arrived with the stiff string exactly as
§11's own recommendation proposed), #4 the web viewer (16 batches). ⇒ **When a doc says "open
decision", check whether history already answered it before asking the human.** Two of the three
were traceable in one grep each. The real scope was "close §11 out", not "decide two items".

**#3 POLYPHONY — it is TWO questions, and only one needs an engine.** The structural half is physics
and was answerable immediately:
- **Field models are polyphonic PER INSTANCE.** Membrane #4, plate #5/#5b/#6, `AirBox`: excitations
  superpose on one state array and **the DOF count does not grow with the note count**. Striking a
  drum twice is one drum with two strikes, not two voices. Already wired: `AirBox` takes N disjoint
  ports, `SurfacePort` couples a whole node set. **Not** wired: two simultaneous mallets on one
  membrane — `MalletMembrane` is a single mallet. That is an **exciter-layer gap**, not a model limit.
- **1-D strings are polyphonic PER VOICE.** One instance per sounding note, cost linear in voice
  count. Already wired: `SympatheticStrings` = N `IdealString`s sharing one bridge point.
- ⇒ the answer is **"all of them, by two different mechanisms"**, and the engine must express both.
  Note `engine.simulate()` still drives **exactly one** resonator while the scene half already exists
  ad hoc (N ports, N strings).

**The CPU-budget half stays deferred to Phase 5** — §8 already said so. The tempting move, a measured
cost model, was **argued down and should stay down**: offline-NumPy cost/second does not transfer to
a C++/Rust real-time budget; at best the relative ranking survives, and this repo's own lesson
([[air-box-state]] batch 2) is that **a ratio survives a change of conditions where a magnitude does
not**.

**#5 TOLERANCES — the survey INVERTED the question, and one file read decided it.**
`docs/dev/portability-contract.md` §"When we *do* port" already said: the ported kernel is correct
*iff* it reproduces these numbers — "lossless energy drift < 1e-10 (**in practice ~1e-15**)".
⇒ **the ~50× gap between bound and observed is the port's headroom for a different compiler/BLAS, not
slack.** Tightening the acceptance bar would make the future port fail for reasons that are not
physics. **Do not tighten tier 1.** Now written in `HANDOFF.md` §6.1 as three tiers:
1. **acceptance `1e-10`** — the lossless drift bar, identical for every resonator (`DRIFT_TOL` in
   ~15 modules, several literally commenting "the same bar as every other resonator");
2. **machine-exact `1e-12`…`1e-15`** — for quantities that are *exact* (an exact discrete
   eigenvector, `radiated == injected`, a bit-identity). Tightening here is fine — that's the point;
3. **physical, per test, cents or %** — no global number possible; measured first, justified in the
   assertion message, and **re-measured per configuration, never inherited**.

**The practice was already coherent — it was the RATIONALE that was missing.** ~700 numeric bounds,
no central module, and centralizing was explicitly rejected: the per-test bound carries its own
measured justification (`CONSERVE_TOL = 1e-10  # observed ~1e-12`) and that is a **strength**. The
gap was that the reason lived in a file about *portability*, so nothing in §6 or `CLAUDE.md`
explained why the bars look loose.

**Left deliberately untouched, flagged not fixed:** `CLAUDE.md`'s "Start here" still names the
ideal-string milestone as the first deliverable — stale in an always-loaded file, but outside this
batch's scope.

Related: [[milestone-1-state]] (whose "still the human's to set" line this retires),
[[test-suite-performance]], [[air-box-state]].
