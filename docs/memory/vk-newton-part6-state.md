---
name: vk-newton-part6-state
description: "VK Newton Part 6 — the default moves to `auto` and it is a FALLBACK not a flip; the SEED is the line that could have made it worse than either method, the sweep cap becomes a WASTE budget, and only tests that assert failure POSITIVELY noticed"
metadata: 
  node_type: memory
  type: project
  originSessionId: 69ee136e-5127-48be-87e7-2c7df33bbab6
  modified: 2026-09-07T12:17:19.296Z
---

Part 6 of the von Kármán Newton batch (`docs/dev/vk-newton-plan.md` §15), landed 2026-09-07.
Follows [[vk-newton-part5-state]]. The human's call, from a menu describing Newton as "built but
not switched on".

**The obvious move — flip the default to Newton — is the one the measurement forbids.** §13.3 has
Picard cheaper in six of eleven mapped cells at musical amplitude (0.55–0.85×) and Newton up to 68×
cheaper at the wall. So neither dominates, and what shipped is a third spelling,
`couple_method="auto"`, now the default: run the sweeps, return their result untouched when they
converge, re-solve with Newton when they do not. `CoupleMethod::Auto`, `vk_step_with` split into
`vk_picard_step` + `vk_newton_step`, one new `VkStep` field, the keyword and getter on both
bindings.

**The seed is the single line that could have made `auto` worse than either method alone.** An
expansive Picard exit parks an enormous or non-finite iterate in `w`. A Newton seeded from *that*
returns a NaN and reports `converged: false` through the same five read-outs a genuine Newton
failure uses — strictly worse than either method, and silent. The rescue therefore re-seeds from
`2 w^n − w^{n−1}`, and `vk_newton_step` takes `(ctx, u, u_prev)` with **no parameter** through
which a failed iterate could be passed, which is the cheapest way to make that structural rather
than a promise.

**The bar for it must be `assert_eq!`, not a tolerance.** Two Newton solves from *different* seeds
still land on the same root to ~1e-13, so any tolerance loose enough to be safe against that would
also have passed on the bug. Bit equality is the only statement that pins **which seed was used**.
Same shape as the bit-identity bar one level up: `auto` versus `picard` compared on `u`, the stress
cache and all five diagnostics over twenty steps × six fixtures — the diagnostics deliberately in,
because a step agreeing in `w` while reporting a different sweep count still moves a number the
suite reads.

**A count, not a method name — and the aggregating caller is the reason.** The obvious field was
`method_used: CoupleMethod`, and it is wrong one level up: `mallet::VkContactStep` sums `n_iters`
and `n_solves` over *every* plate solve in its outer chord, so a step there contains several coupled
solves and a method **label has no referent** across them while "how many needed rescuing" still
does. `n_fallbacks` also happens to be the only read-out separating a cheap step from a rescued one
— after a fallback the other five describe Newton.

**The sweep cap silently becomes a WASTE budget.** A rescued step pays `2 · couple_max_iter` on the
abandoned sweeps before Newton starts: 1.4–3.0× solo Newton at the default cap of 50, and **15.8×**
at a cap of 400. A small cap used to be a cost *ceiling*; under `auto` it is not one any more.
Raising it to help Picard now also raises the price of every rescue. `"picard"` stays reachable
precisely so that ceiling stays available.

**`auto` rescues failure, not slowness.** Where the sweeps converge expensively (one mapped fixture:
60 solves against Newton's 36) `auto` still pays the sweeps. Fixing that needs a rule that
*predicts* the outcome, and §13.5 is the standing reason not to — the boundary is not monotone in
amplitude, so a cheap prediction returns a clean **wrong** answer. So the claim is bounded: `auto`
is not "the best of both methods", it is **the sweeps plus a floor under them**.

**Moving a default breaks exactly the tests whose subject is the old default's FAILURE — and only
the ones that assert it positively notice.** Seven test functions had to be pinned back to
`"picard"` (three native, four Python), and every one failed *loudly*, because each says
`assert !converged`, `assert_eq!(outcome, Expansive)` or `expect("this rig is chosen to break the
inner solve")`. A test that merely steps a scene and checks an energy would have started exercising
a different solver in **silence**. See [[vk-newton-part4-state]] for the companion sweep discipline.

**So the silent half had to be MEASURED, not reasoned about.** The room builders pass no
`couple_method` and therefore inherit the new default. Building each inheriting fixture the way its
own tests build it and reading `n_fallbacks`: **zero, everywhere** — both room wrapper tiers ×
both boundaries at `w = 2e`, the bare air-loaded plate, the gong on a string at `3e-4` *and* at the
hundred-times-harder `3e-2`, and the three-way chain. The mechanism is the cap: every shipped
fixture converges inside the default 50, and it is exactly the tests that deliberately **starve**
it (cap 2, 8, 12) that changed. **A fixture that fires the rescue is one that was built to fail.**

**A `-k` filter is NOT a file list, and a commit message that rounds it to one is a false
verification claim.** `pytest tests/ -k "vk or plate or mallet or airbox"` matches test *function
names* across the whole directory: wider than those files (it reaches any test carrying one of
those words, wherever it lives) and narrower (it misses tests in the plate files whose names do
not). Write out the command, not a summary of it — §15.9 exists because the first commit did the
opposite.

**How to apply:** the headline is that Part 0's canonical wall — the sixty-thickness strike whose
whole point was that no cap at any size converges it — converges under the default on all twenty
steps. What is *not* done: no cost-aware rule, `couple_max_iter` not re-tuned for its new second
job, nothing measured over a run rather than one step from rest (§13.4's standing warning), and no
viewer surface for `n_fallbacks`. See [[von-karman-plate-state]], [[vk-newton-part0-state]],
[[airbox-vk-newton-state]], [[mallet-vk-room-state]].
