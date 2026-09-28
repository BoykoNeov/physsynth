---
name: rust-retirement-batch7-state
description: "Python retirement phase C batch 7 — the three bridges, in two commits; body bridge = StringBodyBridge<B: BridgeBody>, plate pair = ONE StringPlateBridge<P: BridgePlate>; the hole is CLOSED; finding #79 (a defensive reset the native type makes unobservable)"
metadata:
  node_type: memory
  type: project
  originSessionId: 52482a8e-b3fb-4e68-8657-aaef0db007d9
  modified: 2026-09-23T15:26:39.292Z
---

Phase C batch 7 (2026-09-23). User said "work on the three bridges"; split into two commits along the
only exact anchor (the advisor's call, reported to the user). Plan `docs/dev/python-retirement-plan.md`
§19 (body) and §20 (plates). **With §20 nothing is implemented only in the binding** — re-derived
§13.1's name join: the 10 unmatched binding classes are all recorded re-homings (§16.7 + the VK bridge).

**Design:** `BridgeBody` / `BridgePlate` traits with an associated `Room` type (`()` free air, `AirBox`
mounted), so `br.step(&mut ())` or `br.step(&mut room)`; the bridge never steps the room.
`BridgePlate::linear()` returns `plate::Params` — for `VkPlate` it is `p.lin`, whose `rho` is already
rho_s, so the binding's two plate classes are one type. `pressure()` is an impl only on the two
linear plates (the VK bridge's absence is a type fact). Setters: `set_beta_s`, `set_stiffness_unguarded`.
Body bridge has NO lambda<1 refusal (sympathetic does). Guard: tridiagonal string block solved directly,
plate block via `SparseLu`.

**One-time checks** (scratch `W:\temp\claude\bridges`, dumps need PYTHONPATH=repo root; replay crate
`replay/` with bins `replay` and `plate`): body 6 scenes, plate 10 scenes — all state + sweep counts
bit-identical; margins exact (one at 4.3e-16); only np.dot/ddot read-outs differ.

**Finding #79:** the binding's post-step zeroing of the force vector is unobservable natively (drive
index immutable) — its mutation failed 0 bars; removed. **Grep for writes over ANY variable name**:
the first grep (only `bridge.`) missed `over.K = ...`.

**Gotchas:** Python `open(...,"w")` on Windows writes CRLF — use `newline=""`; a stray `&` relaunched a
source-mutating script in the background; a mutant that does not COMPILE reads as "0 fail" — make the
sweep report "did not run".

**CI cost (open, the human's call):** the Rust job's native step went 6m56s -> 18m46s with
`connection_plate.rs` (debug profile; two long conservation bars dominate); the Rust job is now the
whole CI run's critical path (~20 min vs ~3 per Python shard). Options noted in §20.4, none taken.

**"Hole closed" was derived three ways** (§20.5): class join, 34 computing methods read, and a pass
over calls on duck-typed collaborators (the first pass is blind to pure orchestration like `step`).

**Next:** phase C's hole work is done; remaining = viewer (phase D), scripts (E), uncarried physics bars.

Related: [[rust-retirement-batch6-state]], [[body-bridge-state]], [[python-retirement-state]].
