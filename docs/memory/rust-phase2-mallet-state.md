---
name: rust-phase2-mallet-state
description: "Phase 2's LAST batch (mallet, 2026-08-27) — Phase 2 is COMPLETE; a constant exponent is not a scalar path (LLVM folds powf, so §16.2's own test asserted nothing in release and turned CI red); the agreement window question needs \"does the nonlinearity RECUR\"; a swap guard silently covered nothing for a batch"
metadata: 
  node_type: memory
  type: project
  originSessionId: 5173546e-28a5-42ef-9229-a8b3b8527977
  modified: 2026-08-27T08:00:11.583Z
---

The mallet (model #7) ported on 2026-08-27, closing **Phase 2** — the prediction in
[[rust-phase2-state]] that Phase 2 would finish *after* Phase 3 started came due. Both hard parts
were already Rust (the drumhead from batch 1, the contact solve from [[rust-phase3-collision-state]]),
so what moved was the shell: the force-free flight, two force-injection sites, two admittances.
Plan §17.

**The headline is about the compiler, and it arrived as a red CI run.** The last two CI runs were
red when this batch started. Batch 1's was clippy/fmt (already fixed). Batch 2's was live and was
its *own* test: `collision.rs` asserted the array and scalar power paths "disagree somewhere", and
CI said they came out identical. The cause is **not** the runner's libm — it is **LLVM constant-
folding `powf` at a known exponent into exactly the rungs of NumPy's ladder** (`powf(x,0.5)` →
`sqrt`, `powf(x,2.0)` → `x*x`). Handed the literal `1.5`, the optimiser folded `PowPath::Scalar`
into `PowPath::Array` and the test compared a thing with itself. **It passed in debug and failed in
release, which is what CI builds.** Measured in release on this machine: 0 differing samples with
the exponent folded, 91 with it behind `black_box`.

So: **a distinction between two spellings of the same arithmetic is only observable while the
compiler cannot see which one you meant.** The port was never wrong — the binding takes `alpha`
from Python at runtime and can never be folded — the *test* was. Two rules from it:
- **Run a native test that pins an arithmetic spelling in BOTH profiles.** The whole native suite
  was run in debug and release for the first time here; only `collision.rs` differed.
- **Replace "they differ" with "it equals an opaque `pow`".** How *often* two spellings differ is a
  property of the C library — a claim about a runner, which is [[rust-phase2-radiation-state]]'s
  §14.2 rule arriving in a **test** instead of in a port. The count is now reported, not required.

**The same finding pointed the other way is the shell's one real trap.** Every mallet constant is a
squaring (`k**2/(rho h**2 (1+sigma k))`, `k**2/M`, `0.5 M v**2`) and those are **Python floats**, so
`**` is libm `pow`, not `x*x` — 225 of 400,000 samples differ. The admittances multiply the contact
force every timestep, and getting it wrong **conserves energy perfectly**, so no bar in the repo
could catch it. `mallet::scalar_pow` is therefore `#[inline(never)]`, and per the above that
attribute is the *only* thing making it survive optimisation.

**Everything is bit-identical, further than any batch before it.** Trajectory (position,
penetration, force, contact flag, fallback count, and the drumhead's *entire field*) identical over
2,000 steps across 7 fixtures and over **50,000** on the default one. `MalletWall` identical in
every observable including energy over **200,000** steps. The one exception is
`MalletMembrane.energy()`, 2.2e-16 at 381 of 20,000 steps — and `MalletWall` being exact is what
**attributes** that to `Membrane.energy()`'s read-out reduction rather than assuming it.

**The window question gets one more word.** [[rust-phase3-collision-state]] said the agreement
window is set by the dynamics. Refined: ask not "is it nonlinear" but **"does the nonlinearity
recur?"** The barrier re-contacts every period, so differences feed back and it separates
exponentially; the mallet's contact is a **transient** — the felt engages once and the drumhead is
linear thereafter — so there is no mechanism to amplify anything and the window never closes. The
four nonlinear models still to port (`string_nonlinear`, `string_geometric`, von Kármán, `bow`) are
all *sustained*, so expect barrier behaviour, not this.

**A swap guard can silently cover nothing.** `collision` was never in `test_stability.py`'s
`ported_expected` table — ten swapped functions unguarded for a whole batch — because three of its
public names carry a **leading underscore** while their `_py` aliases do not, so the derive could
not resolve them and the module fell out. Same shape as §16.8's parity job that ran no files:
third door, same room. Fixed by teaching the lookup the underscored spelling, keeping the set
derived rather than listed.

Smaller: the borrow is **one phase** (unlike the reed's two — the mallet corrects *after* the
membrane steps rather than injecting inside it, so no `step_native` hook); the under-resolved
warning stays in Rust with **`stacklevel=1`**, because a Rust `__new__` pushes no Python frame and
1 there means what 2 means from `__init__` (the mirror of §16.8's decision, and a parity test pins
that both blame the same caller line). Speed: **97x** on `MalletWall` (a pure scalar loop touching
no compiled NumPy at all — [[rust-phase2-body-state]]'s crossover at its limit), 11.2x/3.2x/1.8x on
the coupled model as the grid grows.
