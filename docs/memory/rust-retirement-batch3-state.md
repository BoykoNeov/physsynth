---
name: rust-retirement-batch3-state
description: "Python retirement phase C batch 3 — the distributed port tier; the room's cut is INSIDE the port; a shape refusal has no analogue; the first PARTIAL file retirement"
metadata: 
  node_type: memory
  type: project
  originSessionId: beef380b-cd7d-46eb-8c58-ddd3cf14139a
  modified: 2026-09-08T11:02:44.427Z
---

Phase C batch 3 (2026-09-08), the retirement's third re-homing: `SurfacePort` and
`InteriorSurfacePort` moved into `crates/physsynth-core/src/airbox_port.rs`, with 28 native bars in
`crates/physsynth-core/tests/airbox_surface.rs`. **The port tier is now complete** — the hole
[[rust-retirement-batch2-state]] measured at sixteen classes is down to **eleven across two files**
(six grid wrappers + three surface adapters in `airbox_wrap.rs`, three bridges in `connection.rs`).
Plan section is `docs/dev/python-retirement-plan.md` §14.

Five things that generalise past this batch.

**The room's partial CUT was inside the port, not adjacent to it.** `InteriorSurfacePort` calls
`room._register_cut(...)`, and the native `AirBox` had only a full-plane `cut_plane` — no extents,
no refusals, no record of who owns which faces. So one of the two named classes could not be ported
without first giving the room `cut_records`, `register_cut`, `add_cut` and `cut_faces()`. Same shape
as batch 2 finding the ports underneath the wrappers: **before scoping a port batch, ask what the
class CALLS on its collaborator, not only what it is.**

**A refusal about the SHAPE of a Python argument has no analogue; one about a VALUE does.** Four
here — `coords must be (n, 2)`, `origin must be a pair`, `unknown spreading 'cubic'`, `extent must
be ((lo,hi),(lo,hi))` — all become types (`&[[f64;2]]`, `Option<(f64,f64)>`, an enum,
`Option<[[i64;2];2]>`). Not a coverage hole; coverage that moved to the compiler. But the
retirement table needs a **third verdict** beside "replaced" and "moved": **no analogue**, written
down, or a later reader finds a Python refusal with no twin and concludes the port dropped it.

**A test file can retire in PART, and the rule is the referent, not the name.** First time in the
migration. `tests/test_airbox_surface.py` kept its 19 wrapper-driven tests and lost 15 port-tier
ones. The rule: *a test retires when its referent is the ported class — it constructs one directly,
or its every assertion reads that class's attributes; a test that DRIVES an unported caller stays.*
Sharper than "which class does it mention": `test_ledgers_agree` is named after the port's
resistance and is a wrapper test, `test_refuses_a_footprint_outside_the_face` is built through the
wrapper's helper and is a port test. **The tell is `inst.step()`.**

**A partial retirement does not reduce a file's cost in proportion to its test count.** Deleting 15
of 34 functions moved the runtime by 1.04x — inside the noise, measured within one machine. The
expensive tests were the wrapper's 200-step trajectories and all of them stayed. So
`scripts/shard_costs.json` was left alone rather than scaled. Test count and test cost are
uncorrelated *within* a file.

**The one-time cross-implementation check found ZERO differences, and had to be proven sensitive.**
392 keys over 11 fixtures — `T`/`R`/load-matrix `indptr`+`indices`+`data`, node and flat *ordering*,
the interior tier's low/high split, and a 200-step hand-driven trajectory over every pressure node
and velocity face. Unlike [[rust-retirement-batch2-state]] there was no fused-multiply-add to chase:
this tier has no `np.dot`. A green comparison proves nothing until a deliberate error reddens it, so
**two mutations were run**: folding the triple product's diagonal right reddened 9 of 11 fixtures
and left green exactly the two using `spreading="nearest"` (there every stored `T` entry in a row is
the same number, so both associations are the same double); reversing the divergence's axis order
reddened all 11, and on three of them `free_pressure` at step 0 still agreed while only the 200-step
field differed — a static-only check would have passed all three. The first mutation is now a
permanent bar, because the association is the one decision with no referent left after phase F.

**Two things to carry forward.** The exposure of a partial retirement is worth naming *specifically*
rather than generally: here it is that `parse_spreading`'s **omitted-argument arm** (the one whose
arms were backwards in the first draft) is now unreached by the suite — a grep, not a guess. And the
wrapper batch must measure, while Python still exists, **how the load matrix reaches the six `splu`
calls** (is `a_loaded` an add of two CSRs or one assembly, and in which order) — the values and
stored order are already verified, the *assembly* is not, and its reference disappears with the
wrapper.

**Second commit: `tests/test_airbox_cut.py` went whole** (14 functions, 42 cases -> 11 native bars),
because the cut primitive the port needed made the room-tier file portable. Plan §15. Two rules from
it. **Randomness does NOT have to be reproduced.** The reference seeds and drives from NumPy's
PCG64; neither claim is about the numbers, both are about a field with no structure, so a splitmix64
hash of the index is correct rather than sloppy — *a bar asserting a tolerance on a broadband field
is a claim about the class of fields, not one field*, and the tell is that nothing in the retired
file compared two runs. This is the one place the migration's exactness discipline does not apply.
**And a helper whose referent has gone is residue**: `make_cut_room` and `sub_room_mode` left
`tests/helpers.py` with the file (`airbox_noise` stayed — two other files use it). The modal
oracle's half-cell was mutation-tested rather than trusted: removing the `0.5` takes the field error
from below 1e-12 to **2.49** against an amplitude of 1. Two more from the same file, both about
what a carried-over bar *fails* to assert: **a negative about a float is not an assertion** (the
reference's `(lo/h) % 1 != 0.0` passes for any wrong value and needs no tolerance because it claims
nothing — the positive form does, and immediately: the first fixture returns `5.500000000000002`),
and **a `continue` is silent where `pytest.skip` reports**, so a loop carried over from a
parametrized test must count what ran and assert the count.

Related: [[python-retirement-state]], [[rust-retirement-batch2-state]], [[air-box-state]],
[[rust-phase5-ports-state]], [[commit-push-at-batch-end]].
