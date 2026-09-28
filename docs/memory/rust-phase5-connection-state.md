---
name: rust-phase5-connection-state
description: "Rust migration Phase 5 batch 10 — connection.py's four bridges, which FINISHES physsynth/core/; a ported caller that computes nothing is SLOWER than the Python it replaced"
metadata: 
  node_type: memory
  type: project
  originSessionId: e6a1e7e6-5a91-44b8-a61d-48825c1f42e4
  modified: 2026-08-31T19:55:54.486Z
---

Phase 5 batch 10 (2026-08-31), plan §34: `physsynth/core/connection.py` — `StringBodyBridge`,
`StringPlateBridge`, `StringVKPlateBridge`, `SympatheticStrings` — is Rust.
**`physsynth/core/` is finished.** Only `analysis/` is left in the whole migration.

**The headline, and it gives the migration's oldest rule a sign: a ported caller that computes
nothing is SLOWER than the Python it replaced.** §11.6 said the win is per-call overhead; §32.4
said an inner iteration wins only if its body ports with it, and measured neutral. Here it goes
negative, because **Rust pays that overhead too when it is the one making the call** — a `getattr`
+ `get_item` + `extract` from Rust is not cheaper than CPython's `LOAD_ATTR`/`BINARY_SUBSCR`, which
have inline caches and a specialising interpreter. What Rust wins is the *work between* the calls.
One file contains both cases, flag on (Rust collaborators), best of five:

- `StringBodyBridge` — five attribute touches, two float multiplies — **0.96-0.97x** (a loss)
- `StringPlateBridge` — a sparse solve dominates — **1.00x**
- `SympatheticStrings` — per-string force array, loop over J, reduce — **1.51-1.83x**

Ask before scoping a batch: *how much work sits between the collaborator calls?* If none, the port
buys fidelity and costs a few per cent.

**Everything is bit-identical**, the von Kármán bridge's trajectory included (§32.5's mechanism: one
Picard loop through one factorization, so exactness is a sharp test of the transcription).

Six things worth carrying:

1. **All four classes are ONE batch and it was not a choice** ([[rust-phase3-banded-state]]'s
   §15.2). `test_sympathetic.py` asserts a one-string `SympatheticStrings` is `array_equal` to a
   `StringBodyBridge`; `test_airbox_vk.py` asserts
   `StringVKPlateBridge.stability_margin == StringPlateBridge.stability_margin` exactly. The parity
   file re-asserts both in all **four** language combinations — the cell no existing test reaches.
2. **Second module with no core half, reached from a different direction than
   [[rust-phase5-wrappers-state]]'s.** There the tier below had promised replaceable matrices; here
   the class is **polymorphic over its collaborators' types** — eight kinds of body/plate, no
   `isinstance` anywhere. A downcast would pass the whole airbox and radiation family, so the parity
   file builds a hand-written stand-in that is a `ModalBody` in neither language.
3. **The plan's claim that `connection.py` reaches no private name was WRONG** — `string._bc_right`
   ×4, `string._second_diff` ×2 — and the cheerful general form is that **a dependency written down
   at the start survives being forgotten in the middle**: Phase 0's `lib.rs` exposed both names
   deliberately three phases early, so the port worked on the first build.
4. **Both reductions and both dot products are invisible to every shipped fixture.** `np.sum` is
   transcribed (`reduce::sum`), `np.dot` is not (`ddot` fuses) — and every body here has 4-5 modes
   and every sympathetic rig 2-3 strings, below §30.2's eight-term cutoff (0/5,000 disagreements at
   M = 4, 5, 7 against **2,022/5,000** at M = 8), while `phi = 1.0` blinds the suite to the fused
   multiply-add. So the parity file **searches** for fixtures at M = 12 and J = 8.
5. **A cached keyword dict is the wrapper tier's caching hazard inside the batch that cites it.**
   Reusing the `plate.step(f_ext=)` dict is worth ~3% of the step; caching the *array* in it would
   ignore a caller who replaced `_f_ext`. Cache the dict, re-set the entry every step.
6. `string.u[-1] -= beta_s * F` must be an **in-place item write** on the live array object — the
   line `lib.rs` names as the reason the string's buffers are Python-owned. Reading into a `Vec`
   and rebinding passes every physics bar and loses every snapshot.

Scars: **§19.7's YAML line continuation happened a sixth time from a sixth tool** (a shell heredoc
collapsed a doubled backslash before Python saw it) and never reached CI — the general remedy is to
build the backslash as `chr(92)` so no layer can unescape it. See also
[[rust-phase5-membrane-state]], [[rust-migration-state]].
