---
name: rust-retirement-batch6-state
description: "Python retirement phase C batch 6 — the mallet's room MODE ported as its own type MalletVkRoom; test_mallet_room_gong.py retired WHOLE; the tangent bar was vacuous (difference from a twin that differs twice); hole = 3 bridges"
metadata:
  node_type: memory
  type: project
  originSessionId: d815c53e-f0a0-4db1-b084-4c489489d4c2
  modified: 2026-09-23T14:35:05.411Z
---

Phase C batch 6 (2026-09-23), the human's call to do the mallet's room mode on its own before the
bridges. Plan: `docs/dev/python-retirement-plan.md` §18. Finding #78 in
`docs/dev/rust-migration-findings.md`.

**Design:** a *type*, not a mode — `MalletVkRoom` in `crates/physsynth-core/src/mallet.rs` holds
`pub grid: RoomGrid<VkSeam>` and steps with `step(&mut AirBox)` (the grid never owns the room). The
binding's pointer check for a swapped factorization became `RoomGrid::generation()`, bumped only by
`refactor`. Error = `RoomGongError { Room(WrapError), Gong(VkContactError) }` (no ParkedErr needed).
Binding's room arm (`step_in_room` etc.) LEFT in place, untested, dies with the crate.

**One-time check:** 6 scenes (both mounts, linear, miss, free+Newton, mid-run swap) — every state
quantity and every count bit-identical to the binding, debug and release; only `radiated_energy`
differs (≤3.2e-16 of its size). Scratch under `W:\temp\claude\mallet_room` (dump_binding.py needs
`PYTHONPATH` = repo root).

**Two finds:** (1) the native bare `MalletVkPlate::step` hand-wrote 3 of 6 plate read-outs (§17.1
drift again) → both mallets now go through `plate_step_of` + `VkPlate::record`. (2) The retired
tangent bar asserted room ≠ bare; the retargeted column alone makes them differ, so it was green with
the operator unrouted — replaced by a finite-difference oracle (routed ≤9.5e-11, unrouted 9.1e-6,
bar 1e-8). **Mutate each half of a two-part override separately.**

Gotcha: a suspended port queues TWO `pending_ports` entries per inject (-q/+q), so count injections
against the bare wrapper's own single step, not a literal 1.

**Next:** the three bridges (`StringBodyBridge`, `StringPlateBridge`, `StringVKPlateBridge`) — the
whole remaining hole.

Related: [[rust-retirement-batch5-state]], [[mallet-vk-room-state]], [[python-retirement-state]].
