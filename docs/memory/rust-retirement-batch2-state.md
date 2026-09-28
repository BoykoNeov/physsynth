---
name: rust-retirement-batch2-state
description: "Python-retirement phase C batch 2 — the lumped room port; the hole is 16 classes / 3 files (derived, not hand-listed) so the port order is FORCED, and BLAS ddot FUSES where no native bar can see it"
metadata: 
  node_type: memory
  type: project
  originSessionId: aff59ea5-c2fc-4817-9deb-61fb767f381e
  modified: 2026-09-08T08:28:44.447Z
---

Batch 2 of the phase-C re-homing (2026-09-08): `RoomPort` and `RoomLoadedBody` moved from the
binding into `physsynth-core` (`airbox_port.rs` + a new `airbox_wrap.rs`), 26 native bars replacing
`tests/test_airbox_port.py` (25 functions, deleted). Plan `docs/dev/python-retirement-plan.md` §13.
Follows [[python-retirement-batch1-state]].

**The hole was hand-counted and undercounted.** [[python-retirement-state]]'s §11 said 14 classes
in 2 binding files. Derived — every `#[pyclass(name=)]` in `physsynth-py/src` (43) minus every
`pub struct`/`pub enum` in `physsynth-core/src` (88), then clear case-spelling false positives by
hand (`VKPlate` vs core's `VkPlate`) — it is **16 across 3**. The missed file, `airbox_port.rs`, is
the tier the other two stand on, so **the porting order is forced, not chosen: ports → wrappers →
bridges.** Each tier's slots take the tier below as arguments. Same understatement one level down:
`core::AirBox` was a *shell* whose `step` never called the port kernels sitting beside it, because
only the binding reached them — and the binding does not hold a `core::AirBox` at all (it holds
`core::Params` and re-assembles the step), which is why fixing that could not disturb a shipping
path.

**The room-ownership decision, which the other 14 classes inherit.** `test_two_heads_share_one_room`
rules out a wrapper owning its room. Chosen: **the port is a value the caller owns and the room is
passed at each call** (`port.inject(&mut room, q)`, `inst.step(&mut room, force)`). Rejected:
`Rc<RefCell<AirBox>>` (interior mutability in the core's public API) and room-owns-ports (an
unchecked index handle). The two invariants the reference kept on the room each cost one field:
`claims` (the disjointness refusal's data, `room._ports` reduced) and `epoch` (bumped by
`set_state`, replacing writing `_queued_at = -1` into ports the room no longer holds).

**BLAS `ddot` FUSES its multiply-add, and no native bar can see it.** The one `np.dot` here that
reaches the timestep (`u_free`, the modal velocity read). `s += x*y` diverges from the reference at
the *second* step by one ulp; 200 steps later 661/693 room pressure nodes differ at 1.9e-12 with the
body's own state still bit-identical — the room amplifying one ulp. `x.mul_add(y, s)` is
bit-identical at 2 and 5 modes. **Restoring the plain loop leaves all 26 bars green**, so nothing
but the retiring reference could ever choose between them: the cross-implementation check has to
happen in the batch that ports the class. See [[rust-phase2-radiation-state]] for the older half of
this (a BLAS reduction that feeds back into state).

**Mutate the trap the bar is for.** The new cross-ledger bar was written to catch a wrong `R_room`
and passed with the `1/(1+beta)` factor deleted — both its ports sat at interior nodes where `beta`
is 0 and the factor is exactly 1. Moved to a corner it goes red. Under the same mutation all six
conservation/passivity bars stay green, which turns the retired file's "the total is structurally
blind to this" from a quoted claim into a measurement.

**Forward constraint for the bridge batch:** `RoomLoadedBody` overrides `energy()` and delegates the
rest. The bridge's trait must send `energy()` to the **wrapper** and modal reads to the inner
`ModalBody`; handing it `inst.body` compiles, conserves nothing, and only the chain test moved into
`tests/test_connection.py` asserts otherwise.
