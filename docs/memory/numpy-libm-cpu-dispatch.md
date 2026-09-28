---
name: numpy-libm-cpu-dispatch
description: "NumPy computes transcendentals with its own CPU-dispatched routines, not the platform libm — so a Rust-vs-Python bit-identity assertion is a claim about which machine ran CI"
metadata: 
  node_type: memory
  type: project
  originSessionId: 2001450e-e951-4604-9fec-32002dfecae3
  modified: 2026-08-27T15:31:12.506Z
---

Found 2026-08-27 by two CI runs of effectively identical code reading **one** failure and then
**eighteen** — same runner image, same NumPy 2.5.2, same SciPy. The only variable was the CPU.

**The mechanism.** NumPy does not call the platform C library for `pow`, `tan`, `exp`, `cos` etc.
It carries its own vectorised routines, selected at import from the processor's feature set. Rust
calls libm. So any "bit-identical" parity assertion whose value passes through one of those is two
*implementations* agreeing, and whether they do belongs to the machine.

**How it was diagnosed without a local repro** (there is none — see below): the pass/fail split
correlated *perfectly* with NumPy's power-ufunc shortcut ladder (`-1, 0, 0.5, 1, 2`). Every
exponent NumPy spells as `sqrt`/`x*x`/`x`/`1`/`1/x` agreed; every exponent it hands to its own
`pow` failed by 1–2 ulp. The three non-power failures were one `np.tan`, whose transcendental-free
sibling method passed. When a parity test goes red with no code change, **look for a correlation
with the ladder before looking at the port.**

**No local repro is possible on Windows, and that is structural.** `NPY_DISABLE_CPU_FEATURES`
changes nothing here because Windows NumPy has no dispatched `pow` loop to strip — both languages
reach UCRT and always agree. This whole class is invisible to local development. CI now prints
`/proc/cpuinfo` model name and `numpy.show_runtime()` before the parity step.

**Confirmed from the passing side 2026-08-27** (run `33090052206`, the first green run): on an
**AMD EPYC 9V74** (baseline `X86_V2`, found `X86_V3`) all eight off-ladder power comparisons read
**0 of 20000 differ, 0.0 ulp** — the exact cases that failed two runs earlier. Same commit, same
NumPy, same runner image, different processor. So the divergent CPU is the **minority** case, and
the ulp bars pass at *zero* on the common machine: a run printing non-zero counts means the job
landed on the other kind of machine, not that something regressed.

**The standing rule** (the human's call, 2026-08-27): exactness where it is *provable*, an ulp
bound where it is not. Before writing an exact-equality assertion, ask which library computes each
value on each side.

- Exact is legitimate only at ladder exponents, where both sides do the same multiply. `alpha = 1.0`
  is the *only* value putting all three contact-primitive exponents (`a-1, a, a+1`) on the ladder.
- Off the ladder: four ulp, with the measured count printed.
- Where a portable spelling is free, prefer it to either — `impedance_discrete` moved from `np.tan`
  to `math.tan` (scalar argument, no vectorisation lost) and keeps its exact assertion. That is
  [[rust-phase3-strings-state]]'s `portable.py` manoeuvre a fourth time, first one aimed at a
  transcendental rather than a summation order.

**Three traps this exposed.** (1) [[rust-phase3-collision-state]]'s blind-fixture finding is set by
a *physical parameter*: moving the soft-rail test to `alpha = 1.0` for exactness **lost** the blind
spot, because tangent stiffness vanishes at grazing contact for `a > 1` and is flat `K` at `a = 1`
(`cond(J)` 1.0625 vs 1.0032). "Engages the solver" and "arithmetic is provable" can be in direct
conflict. (2) A tolerance must be read against the *expression*: the discrete gradient's derivative
divides by `da²` after a same-order cancellation, so a one-ulp nudge moves it **15% of its scale**
near the branch cutoff — no elementwise bound is meaningful; keep `|da|` away from `tol` instead.
(3) The negative result is the useful half — `np.cos` was already documented as being in this class
and **passed** on the divergent machine. The exposure is per-function and per-CPU, not blanket.

Full detail in `docs/dev/rust-migration-plan.md` §22. Related: [[ci-runner-variance]],
[[rust-phase3-bow-state]] (§20.2 predicted the Windows/Linux asymmetry), [[rust-phase2-mallet-state]].
