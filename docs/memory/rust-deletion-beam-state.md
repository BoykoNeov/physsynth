---
name: rust-deletion-beam-state
description: "Deletion 8 (2026-09-03), the free beam: a draining `parametrize` must be DELETED at zero (empty = a silent SKIP), a model's own factored matrix is useless as an eigensolver, a threshold written for one mode measures the fixture when a pluck is fed to it, and a default the helpers always override is dead code"
metadata: 
  node_type: memory
  type: project
  originSessionId: 1cd91ea8-aafa-4747-a740-44a90b00f069
  modified: 2026-09-03T18:19:23.939Z
---

Plan `docs/dev/rust-migration-plan.md` **§45**, findings **#52–#56**. Unit 8 — the free-free beam —
is deleted: `physsynth/core/beam.py` 288 → 35 lines. Nine of eleven units, 18,630 lines,
twenty-two of twenty-three model bodies. See [[rust-deletion-phase-state]], [[beam-state]].

This unit was blocked on **work, not a question**: `FreeBeam` had a Rust core half and no native
test file named it. So the batch is two commits in §44's order — **bars first, while the Python is
still alive**, then the deletion.

## An empty `parametrize` COLLECTS AS A SKIP

`tests/test_rust_parity.py` — the migration's first comparison file — is **deleted, not emptied**.
It was down to one table with one row (`FreeBeam`, the last surviving Python twin). Emptying the
list leaves a parametrize over nothing: pytest reports green, the suite total drops by one, and
nothing says the claim has gone. That is finding #40 one step on — #40 is a *floor* on a draining
population going unsatisfiable; this is the **population itself reaching zero**, where the mechanism
converts silently to a pass. **Write the exit condition down when you create a draining list**: this
file's own docstring said "when it is empty this file goes" a batch before it was needed, which is
the only reason it was noticed.

## A native bar has to BUILD what the Python bar imported

685 lines of Rust test for a 288-line model, because `physsynth-core` has no SciPy and (by the
crate-split argument) no path to `physsynth-analysis`:

- **Derive the oracle, do not import it.** An oracle that imports the implementation is not an
  oracle. The roots of `cos(βL)cosh(βL)=1` are bisected in the file — spelled **`cos(x) − sech(x)`,
  not `cos·cosh − 1`**, whose residual scales like `cosh` so one tolerance means a different
  accuracy at every root. Bracket on `[jπ, (j+1)π]`: the cosine is monotone there, so exactly one
  sign change per bracket and bisection cannot land on a neighbour.
- **The matrix a model already factored is factored for its TIMESTEP, not for its spectrum.**
  Inverse-iterating on the beam's own `A = W + θk²κ²K` is not slow, it is impossible: the
  coefficient is **4.4e-11**, so consecutive modes sit `1 − 1.5e-7` apart (~1e8 sweeps) and reading
  the eigenvalue back subtracts 1 from 1.000000022. Factor a separately shifted `B = K + εW`
  instead (`ε = 1e-3·μ₁`, the shift the Python's `eigsh` had already picked) → ratio 0.13.
- Three details that would each have produced a green test measuring something else: project the
  known nullspace out **every sweep** (its shifted eigenvalue `1/ε` is the *largest* in the problem,
  so roundoff regrows it fastest); **fix** the sweep count rather than stopping on a tolerance
  (#33 — an iteration count is not comparable); read the eigenvalue back as a **Rayleigh quotient**;
  and seed with something not `W`-orthogonal to the mode you asked for (a symmetric seed skips every
  antisymmetric mode and the cents bar then fails with a message about physics).
- **Prefer an algebraic identity of the update to a measurement of its output.** With a
  machine-precision eigenvector, `u^{n+1} + u^{n-1} = 2cos(ωk)u^n` is exact — asserted at every node
  to 1e-13, where the Python's FFT bar could only reach 5 cents.

## Two fixtures that had to be SEARCHED for

- **`7*(0.7/7) == 0.7`.** The `np.linspace`-overwrites-its-endpoint bar was written at `L = 0.7` by
  analogy and asserted nothing. `L = 0.9` works. The test now asserts the negative half too, so a
  fixture that stops exercising the overwrite fails instead of passing vacuously.
- **A plucked lossy beam does not decay at `exp(−2σt)`.** Retention runs 52.9% / 38.9% / 28.6% /
  8.6% at 4k/8k/16k/60k steps — an asymptote, because a pluck is broadband and this scheme's high
  modes underdamp (`2σ(1 − θQk²)`, `Q = κ²μ`, fourth-power in the mode). A "the loss did work"
  threshold is therefore a statement about the fixture's spectrum. Assert the **gap** instead:
  retained > 10× the single-mode rate. That is a claim about the model, and it makes the passivity
  bar and the underdamping-caveat bar say the same thing from two directions.

## Guard edits (five, plus one retirement)

`deleted_bodies` gains `beam`; `beam` leaves the swapped-class derive, `expected_classes` **and the
`_USE_RUST` reader tuple** (that third one is easy to miss — it failed the full run); two entries
leave `REMAINING_PARITY_FAMILY` and two leave the `rust` job's list. The
`beam.free_beam_stiffness is operators.free_beam_stiffness` capture is **retired** — no capture is
left — and the claim moved to the native bar rather than going.

## Three things caught in REVIEW, after green and pushed (like §44.9)

- **The shim re-declared `THETA_DEFAULT` instead of re-exporting it.** That looked like the
  "keep measured constants" rule; it is not. `string_stiff.py`'s header claims to be the family's
  one source and names `beam` and `plate` among its importers — **both were wrong**, and had been.
  Fix: `beam.py` re-exports it, the header is corrected, and both halves are now *asserted* rather
  than claimed (#49 again — a header is a claim and nothing checks it).
- **The sharper half the duplication hid: `tests/helpers.py` passes `theta` explicitly on EVERY
  construction**, so the binding's own default is exercised by no physics test at all and could
  drift to any value with the suite green. **A default the helpers always override is dead code as
  far as the suite is concerned** (#56). The guard is in `tests/test_binding_surface.py`: build each
  θ-scheme class twice, once with `theta` omitted and once with the module constant, assert they
  agree.
- **Reconcile a suite-count delta, do not reason it out.** The flagged total moved by 2 and the
  first draft of the plan wrote "predates it" from reasoning. Measured in a worktree at the previous
  commit: §44.8's middle shard figure was a **transcription slip** (761, written 760), and the real
  cause is `tests/test_xdist_groups.py` parametrizing one test **per file in `tests/`** — so
  deleting two test files removes two cases from a *non-parity* file. A fourth glob-tracking
  population, and the only benign one.

**What survives:** `tests/test_beam_{energy,modal,stability}.py`, 32 tests, run unchanged against
Rust on the *default* path afterwards — which is also the proof the binding hands back real
`scipy.sparse` matrices, since they pass `beam.K`/`beam.W` straight to ARPACK. Run that flagged as
the batch's pre-flight before writing anything.
