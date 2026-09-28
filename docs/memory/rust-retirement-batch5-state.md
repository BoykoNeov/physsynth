---
name: rust-retirement-batch5-state
description: "Python retirement phase C batch 5 — the von Kármán room seam; wrapper tier FINISHED; iteration counts bit-identical to SuperLU; the mallet's room MODE is a hidden piece of the hole"
metadata:
  node_type: memory
  type: project
  originSessionId: 9ce14d33-b586-4851-98f4-1afe9183f1b1
  modified: 2026-09-23T13:34:24.741Z
---

Phase C batch 5 (2026-09-23), commit 2 of the human's "linear first, gong second" split. The gong's
room wrappers re-homed as `VkSeam`, a third seam of `RoomGrid<S>` in
`crates/physsynth-core/src/airbox_wrap.rs`; 17 native bars in
`crates/physsynth-core/tests/airbox_vk.rs`; `tests/test_airbox_vk.py` 26 -> 9 (the 9 left drive
`StringVKPlateBridge`, unported). Plan: `docs/dev/python-retirement-plan.md` §17. **Wrapper tier
finished.**

**Design:** the iterating solve fits the trait as ONE provided method, `GridSeam::advance(lu, rhs)`
(default = solve + commit, so the linear seams are untouched). `VkPlate::record(VkStep)` is now the
single commit path for the bare step and the seam — a hand-written commit list is how the binding's
seam once lost two read-outs. Deliberate departure: on `nonlinear=false` the seam leaves `F` alone
(bare plate's behaviour) where the binding rolled `F_prev <- F`.

**One-time check:** 8 scenes × 120 steps, debug and release: `w`, `F`, whole room field, and
**every per-step iteration count** (Picard 6–19 sweeps, Newton with GMRES) bit-identical to the
binding's SciPy-`splu` path. The worry that many solves per step + a branch on a norm would flip a
count did not happen. Only `radiated_energy` differs (≤2e-16, the `ddot` read-out, no feedback);
`room.injected` exact. Method: dump inputs from Python as raw doubles, replay in a scratch crate
under `W:\temp\claude` with a path dependency on physsynth-core. Windows gotcha hit: `x.f.bin` and
`x.F.bin` are the SAME file.

**Mutations:** wrong operator (`VkCoupledStep::new` for `with_rhs`) → 6 bars fail, and
`radiated == injected` still PASSES (a gong not in the room books consistently); `rho_v` for
`rho_s` → 7; hand commit → 4.

**The finding:** `MalletVKPlate` handed a room wrapper switches to `step_in_room`
(`crates/physsynth-py/src/mallet.rs`) — a composition that exists only in the binding and is a MODE
of an existing class, so no class-name audit lists it. A grep for every room-wrapper recognition
site found only this one. Hole = 3 bridges (`StringBodyBridge`, `StringPlateBridge`,
`StringVKPlateBridge`) + the mallet room mode. **The mallet room mode was ported next, on its own — DONE, see [[rust-retirement-batch6-state]]; then the bridges.** Not ported this batch: the human approved a
5-step plan without it. Native analogue for its swapped-factorization pointer check: a generation
counter bumped by `RoomGrid::refactor`.

Related: [[rust-retirement-batch4-state]], [[python-retirement-state]], [[mallet-vk-room-state]],
[[numpy-libm-cpu-dispatch]], [[air-box-state]].
