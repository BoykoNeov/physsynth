---
name: rust-phase3-bow-state
description: "Rust migration Phase 3 batch 5 — the bowed string; a HAND HOIST is the third two-spelling hazard, a recurring nonlinearity that CONTRACTS, the normaliser that moved the number 40x, and Phase 3 is NOT finished"
metadata: 
  node_type: memory
  type: project
  originSessionId: 29813cb5-c485-467e-8134-3dbdc38dd3c7
  modified: 2026-08-27T12:50:38.250Z
---

Phase 3 batch 5 (2026-08-27), plan §20: `physsynth/core/bow.py` — the bowed string, the project's
first continuous nonlinear **exciter**. Almost a pure shell: the banded solve came in §15, the
string in §18, the transcribed Brent in §13.3, the scan-and-bracket idiom in §16. Bit-identical in
**every** observable including `energy()` under a shared solver, at every one of **20,000** steps on
all three fixtures — Newton eval count and fallback branch included. 7.8–8.4x at N=100, 3.3–3.6x at N=400.

**A HAND HOIST is the third way one expression becomes two computations.** After NumPy's power-ufunc
ladder ([[rust-phase3-collision-state]], §16.2) and LLVM's constant fold ([[rust-phase2-mallet-state]],
§17.2), this one is *in the source and looks like a duplicate that wants tidying*: `bow.py`'s scalar
residual multiplies by `g` last, its array residual factors `g * force * sqrt(2a)` out so NumPy can
apply one scalar to the whole 512-point scan. Different doubles in **4,158 of 20,000** samples at
the flagship fixture (568-5,372 across the three; the fraction is set by how big `g*force*sqrt(2a)`
is next to `v - v_free`). The scan's only job is sign detection, so merging them changes **which
brackets exist** — at a slip event, a different branch — while every physics bar stays green. Pinned
in both suites. Negative result worth as much, but it is **a claim about a RUNNER**: `np.exp` (array)
and `math.exp` (scalar) agreed **20,000/20,000 on Windows**, where all three of NumPy, CPython and
Rust reach UCRT — on Linux NumPy uses its own SIMD loop and only the other two reach glibc. Bounded:
the scan decides only whether a bracket *exists* and `brentq` re-evaluates through the scalar path.

**A measurement taken at a parameter the model never uses is §16.4's blind fixture, seen from the
measuring side.** The first number written for the hoist was 306/20,000, from a scratch script that
had hardcoded `g = 1e-3` when the canonical rig's is 0.318 — an order-of-magnitude understatement,
caught only by reading the script back against the model.

**§19.2's branch rule needs a SIZE.** [[rust-phase3-tension-state]] found a reduction flipping a
root-find's iteration count on 1,400/5,000 steps. Here the same class of question comes out the
other way: a ~1e-14 **relative** perturbation asked about a 1e-13 **absolute** Newton tolerance on a
velocity of order 0.1 is two orders of magnitude from the threshold, so the fallback branch never
differs and the eval count differs **at most once in 20,000 steps**. Ask how far the fed quantity
sits from the branch's threshold, not only whether it reaches a branch.

**A recurring nonlinearity can CONTRACT — the fifth agreement regime and the first that does not
grow.** §17.5 said to ask whether the nonlinearity *recurs*; the bow's does, once per period
forever, so the barrier's exponential separation was the prediction. Wrong: the gap is **flat at
~1e-14 of the running peak out to 20,000 steps**, never above 6.7e-14. Helmholtz motion is a
**stable limit cycle**, so a perturbation is squeezed back onto it. The missing word: does the
recurrence drive the system **onto** an attractor or **off** one.

**The normaliser is part of the claim, and it moved the number 40x.** The same runs read
**2.9e-12** normalised by the *instantaneous* field maximum and **6.6e-14** by the *running peak* —
Helmholtz motion beats, so the instantaneous maximum passes through near-nodes and a fixed numerator
over a dipping denominator looks like divergence. §14.2's "normalise by amplitude, never pointwise"
([[rust-phase2-radiation-state]]) needs **monotone** added to it. The first bar written for this
batch failed on exactly this.

**§19.7's literal `\n` was REINTRODUCED by the batch that cites it** — same file, within an hour of
quoting the finding — because the failure is introduced by *tooling* passing a string through one
round of escaping too few, and tooling does not read section headers. Now asserted in
`tests/test_ci_workflow.py`, which scans the **raw text** (after YAML parsing the distinction is
gone), checks every `tests/...py` token names a real file, and asserts the token **count** so the
scan cannot silently stop matching.

**PHASE 3 IS NOT FINISHED.** §19.11 called `bow` the phase's last model; that was inherited from
§16, where `BarrierString` was correctly described as waiting on its host `DampedStiffString` — a
host that landed in §18, after which nobody revisited the sentence. Checked: `BarrierString` now
needs only ported machinery. It is the phase's true last model. General shape: a statement justified
by a dependency expires when the dependency lands, and nobody is notified.

Smaller: `collision::linspace` is now `pub` and shared with `bow` — the one NumPy spelling in the
crate that is shared, because the two run the *same algorithm* (not merely the same one-liner) and
the grid decides which brackets exist. The borrow is one phase (like the mallet, unlike the reed) —
two of the three coupled models avoid §13.2 *by their physics*. `u +=` had to stay an in-place
write. The `force = 0` anchor is a cross-class bit-identity claim and must not be short-circuited.

See [[rust-migration-state]], [[rust-phase3-strings-state]], [[bow-state]].
