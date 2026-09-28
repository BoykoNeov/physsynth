---
name: rust-phase5-wrappers-state
description: "Phase 5 batch 8 (2026-08-31) — airbox's wrapper tier ported; the tier BELOW decides what porting this tier can mean, so it is the first module with no core half; §31.11's Group D prediction was wrong; connection is unblocked"
metadata: 
  node_type: memory
  type: project
  originSessionId: 384148a1-cc2e-40d8-bd91-96e12070ee8a
  modified: 2026-08-31T16:33:44.127Z
---

Phase 5, batch 8 (2026-08-31) — `airbox.py`'s **wrapper tier**: `RoomLoadedBody`, both plate
wrappers, both von Kármán wrappers and the two seams (`_PlateSurface`, `_VKPlateSurface`). Plan
§32. Only the **membrane** pair (~410 lines) is still Python in that file. Ten `test_airbox_*`
files untouched for the third batch running. Everything bit-identical, 95 parity tests.

**The finding — a tier below can decide what porting the tier above is allowed to mean.** §31
stored a port's `T`, `R`, `load_matrix` as plain Python slots because eight tests *replace* them
(and two replace the port's methods). Three more tests replace the **factorization**
(`inst._lu_loaded = splu(a)`) and two call the seam's `rhs()`/`a_bare()` directly. So the port, the
seam and the solver are objects this tier **holds and calls** — a wrapper that cached any of them
would keep loading a plate a test had switched off, and every one of those tests would pass having
compared a loaded plate with itself (§23.6's emptied comparison, **seventh** door). Hence the first
ported module with **no core half at all**: sparse products, assembly, factorization and `np.dot`
all go through Python; Rust owns the control flow, guards, ledgers and elementwise arithmetic
(exact — elementwise ops admit no reassociation).

**Ask, before scoping a batch on top of a ported one: what did the tier below promise its clients
it would let them replace?** No name grep finds it. Six searches now, none finding the others:
private names · re-derivation · duck-typed types · written public attributes · replaced methods ·
**replaced collaborators**.

Five things worth carrying:

* **Bit-identical including the von Kármán trajectory**, which §27.5 says is impossible for a
  nonlinear plate — possible here because the two sides are not two discretizations but *one*
  Picard loop through *one* factorization. Exactness tests the transcription, not the dynamics.
* **§31.11's "Group D batch" prediction was wrong**: the wrapper does not *own* a factorization, it
  calls one (`splu` is a module global read at construction). **A solver group is a property of
  ownership, not of the file a factorization appears in.** §24.4's shared-factorization manoeuvre,
  fifth use and first as a *negative control* — it moved nothing.
* **No speed win, and that is the price of the above**: 1.06–1.13x linear, *neutral* on von Kármán
  (machine drift 716–1234 µs on the Python arm alone exceeds the effect — interleave arms, take a
  min, say so). Sharpens §29's 15.5x: an inner iteration pays **only if its body ports with it** —
  a ported *caller* over unported callees multiplies boundary crossings by the sweep count (the
  first draft folded the Picard averages in Rust and read 0.91x).
* **A `#[pyclass]` getter is the opposite default from `__getattr__`** — a data descriptor beats
  both the instance dict and delegation, permanently. On a drop-in wrapper every getter silently
  steals a name from `connection.py`'s bridges; `cargo test` and every physics bar are blind to it.
* **§25.8a faked a library bug**: a stale wheel (built, not installed) made a correct
  `Option<Option<_>>` fix look like PyO3 contradicting §31.7's documented arm order. The wrong
  diagnosis a stale wheel produces need not look like staleness.

**`connection.py` is now unblocked** — all three collaborators §31.11 named are Rust and it needs
no membrane wrapper. Next: the membrane pair, then `connection`, then `analysis/`. See
[[rust-phase5-ports-state]], [[rust-phase5-airbox-state]], [[rust-migration-state]].
