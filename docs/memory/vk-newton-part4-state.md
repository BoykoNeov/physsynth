---
name: vk-newton-part4-state
description: "VK Newton Part 4 — a falsified claim had SIX homes, not the two the plan named; live docs get rewritten and batch records get annotated; and the wrong mechanism was load-bearing for a second claim"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 8d72a366-f87e-40e3-b252-a42a1f72c712
  modified: 2026-09-06T09:26:49.974Z
---

Part 4 of the von Kármán Newton batch (`docs/dev/vk-newton-plan.md` §12), landed 2026-09-06.
Retiring the falsified `k² · (amplitude/thickness)² / h⁴` claim: the `k²` half is right, the `h`
half is measured wrong — grid refinement is nearly free and the driver is the **strain**, the
curvature of the deflection. Docs plus one comment block, no behaviour change.

**Grep for the CLAIM, not for the files the plan named.** The part's own scope named two sites and
there were **six**: `HANDOFF.md`, the predecessor plan (`string-vk-plate-bridge-plan.md` §10.4) and
a batch record (`docs/memory/air-box-state.md`) had all quoted it, and `scientific-hurdles.md`
carried it twice (its summary row said "small `h`" as well as the body's formula). A claim
propagates by quotation, and the part that retires it inherits whatever the plan author happened to
remember. Same shape as the migration's "a derived CI list was wrong by 43 of 65 files".

**Two kinds of site, two kinds of edit.** A **live** claim — one a reader is told to consult before
doing work, or a comment sitting on the constructor argument it justifies — is **rewritten in
place**, because a wrong mechanism there gets acted on. A **batch record** is a report of what that
batch concluded; its numbers and its own conclusion are still right, so it gets an **annotation and
a forward pointer** and keeps its narrative. Erasing the trail costs more than a stale mechanism
that says where it was corrected.

**A falsified mechanism is usually load-bearing for a nearby claim — re-read the sentences around
it.** Every live site also carried the categorical *"audio-range modes at 0.1 mm need a ~7 cm plate,
which is exactly the size that will not converge"*. §2.4 measured a 7 cm plate converging at `w = e`
and failing by `w = 2e`, so at a fixed size the ceiling is an **amplitude**. Correcting only the
mechanism would have fixed the smaller error and shipped the bigger one.

**And what decides how strong a correction may be is often a parameter the correcting measurement
did not restate.** The probe ran `e = 1.0e-3`; the rig it corrects is `e = 1.0e-4`. So the 7 cm
plate that converges is a **1 mm** plate — audio-band and Picard-convergent, but a decade too thick
to be string-drivable, which is the trilemma's third leg. The trilemma **survives**; only its reason
was wrong, and 7 cm at 0.1 mm has never been measured. One constant in a scratch script separated
"the claim is false" from "the claim is unproven".

**How to apply:** when a constant's *rationale* is corrected but the constant stands, say so
explicitly in the comment (`VK_BRIDGE_SIDE = 0.4` now does) — otherwise the next reader re-litigates
the constant. Don't write a later part's answer into an earlier one: "cannot converge" was
downgraded to "converges up to an amplitude" with the boundary left as a named gap for
[[vk-newton-part3-state]], not filled with the three fixtures Part 2 had. And a register row moves
to "Open, narrowed", not "fixed", while the default is unchanged and the territory unmapped.
