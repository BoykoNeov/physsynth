---
name: ci-runner-variance
description: "GitHub runner class varies ~1.6x, so a CI wall-clock is only comparable against an unchanged reference test measured in the SAME run"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: aae47a22-23c6-4ed1-8713-4f9ae587e626
  modified: 2026-08-17T16:38:44.237Z
---

**A CI wall-clock number is not a measurement until it is normalised.** GitHub hands out runner
classes that differ by **~1.6×** for the same job.

Measured 2026-08-10 on `physsynth`: the *identical, untouched* module fixture
`test_geometric_whirl.py::…tongue` read **252.05 s, 252.09 s, 158.50 s, 252.76 s** across four
consecutive CI runs. The suite total went 25:01 → 15:56 → 22:46 → 22:56 over that stretch, and the
15:56 was **purely the fast runner** — nothing in the code was faster.

**Why:** I nearly reported a 6.6× speedup from a local run and then a regression from a CI pair, and
both would have been runner/box noise. This is the same failure this repo keeps producing in a
different costume — a number that is true on one machine asserted as if it were a property of the
code (see [[test-suite-performance]], and the airbox goldens/`_peak_monopole`/`p.sum()` trio).

**How to apply:** before attributing any CI wall-clock change to an edit, pull an **unchanged**
expensive test's duration from `--durations` in *both* runs and check they agree. If they don't, the
comparison is void — wait for a matched pair rather than reporting the delta. A cross-run comparison
needs the reference check; a local comparison on this box is worthless either way (~3× inflation,
and it reorders the ranking outright).

**Correction, 2026-08-17: "two jobs of the same push" is NOT the strong form this file used to call
it.** Every job gets its own runner, so two jobs of one push are exactly as incomparable as two
runs. Measured when the gate was sharded ([[test-suite-performance]]): shard 1 is a fixed file set
on untouched code and read **268.36 s** in one push and **476.36 s** in the next; inside a single
push, one shard's machine ran ~2× the speed of another's. The only variance-immune comparison is
**within one job** — a chain's length against its own job's wall, both timed by the same clock on
the same machine. Both of that batch's claims were made that way on purpose.

Corollary: never put a stopwatch reading into a test marker, a docstring, or an acceptance
threshold. Encode the *claim* ("this is a refinement study") or normalise against a scale the
run itself provides.
