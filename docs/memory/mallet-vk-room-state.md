---
name: mallet-vk-room-state
description: "Model #7g in a room (2026-09-06) — the collision was OWNERSHIP OF THE STEP, the frozen column is a cost question not an energy one, and Newton's ranking inverts with sample rate"
metadata: 
  node_type: memory
  type: project
  originSessionId: 248c85a6-17be-48cd-8b59-95a38c1a6e1e
  modified: 2026-09-06T19:11:46.665Z
---

The mallet on a gong **in a room** — the follow-on batch `air-box-vk-newton-plan.md` §5 named as
"half of what the human asked for on 2026-09-06", closed the same day. Plan and results:
`M:\claud_projects\physical synthesis\docs\dev\mallet-vk-room-plan.md`.

**What was blocked was never the operator.** The previous batch's `ThetaSolve` already let a coupled
von Kármán step be pointed at a factorization from outside the crate. The obstacle was that **two
things both wanted to own the step**: the mallet's chord re-solves the plate several times from one
time-`n` state, while the room wrapper's `step()` is a once-per-step transaction (read the port,
inject once, book the radiated energy once). Driving the wrapper per trial injects `n_outer` times —
and the scene total that would normally catch it is the very number a double injection corrupts on
**both sides at once**. Ask of any composition: does one side's loop re-enter the other's ledger?

**The remedy was not to make one call the other.** It was a **trial-solver closure** in
`vk_plate_step`, so the bare gong and the room scene are two closures ending in the same
`vk_step_with`; plus splitting `Wrap::step` into `prepare` / `loaded_rhs` / `finish` rather than
transcribing the room's bookkeeping a second time. See [[rust-phase5-airbox-state]] for why a second
transcription is the thing this family most regrets.

**The inherited plan said the frozen column was "a real question with its own energy bars". Energy
cannot answer it.** `solve_contact` solves `η = η_free − g f` with `g = g_s + g_h`, and the chord
feeds it `η_free = w_node(f) + g_s f − z_free`; at the fixed point the two `g_s` terms **cancel**.
The committed force, the committed field and every energy bar are identical whichever column was
frozen. What moves is the **rate** — a 2.9e-04 error in one scalar costs about **five times** the
outer iterations. The discriminating bar is therefore exact, not statistical: `nonlinear=False`
exits at `n_outer == 1` only for the loaded column. **When a quantity cancels at the fixed point,
stop looking for an energy bar and go find an exact one.**

Three measurements, two of which move an existing claim:

* **The air load does not move the mallet's wall either.** Room and bare die at the same strike
  velocity with the same non-converged counts, `n_solves` within 0.5% at every amplitude. Second
  independent confirmation of [[airbox-vk-newton-state]]'s finding. A room is nearly free.
* **A mallet reaches `w/e = 7.8` where a displacement strike blew Picard up at 4.5** on the same
  grid. **The wall is a property of how the amplitude is DELIVERED**, not only how large it is — a
  mallet builds over hundreds of steps, an initial condition arrives with all of it at once.
* **Newton is CHEAPER below the wall at 8 kHz** (crosses 1.0 at `w/e ≈ 4.3`, settles at 0.71x, zero
  failures to `w/e = 11.6`), contradicting `mallet-gong-plan.md` §10's "Newton buys nothing here
  (1.47–1.72x)". It **dates** rather than overturns it: §10 is at 48 kHz where the `k²` headroom
  means the mallet never reaches the wall. §10's own caveat — a margin at one fixture is a claim
  about one fixture — was right and stopped one axis short. **Sample rate flips solver rankings.**

Two smaller traps worth keeping:

* **`plate.n_iters` after a mallet step is the SUM over the chord's trials**, not one solve's
  sweeps, so comparing it to `couple_max_iter` is a category error (134 against a cap of 50 is four
  trials). `inner_converged` and `last_residual` are what survive the step.
* **A column with two consumers can be nearly right for one and badly wrong for the other.** The
  loaded and bare columns differ by 2.9e-04 at the drive point and **414%** at the far end — near
  the drive point the plate's own stiffness dominates and the air is a correction. The chord reads
  only `influence[node]`; `vk_drive_point_tangent` reads the whole vector as a GMRES right-hand
  side. It was wrong twice over in a room (column *and* tangent operator) and is now routed.

**The batch's own scar: an override added in one part and consumed in another is not finished when
it compiles.** Part 1 added the loaded-operator override for `vk_drive_point_tangent`; Part 3
retargeted the influence column and forgot to point the binding at the override. Result: a **loaded
right-hand side inverted against the BARE Jacobian** — neither problem's derivative, reported as
`g_exact`. A `pub` function with no caller draws **no dead-code warning** (the `pub(crate)` one the
same batch left behind was deleted the moment the compiler mentioned it). **Half-routed is worse
than unrouted, because unrouted is at least self-consistent.** Grep for callers of any override you
add across a part boundary.

Related: [[mallet-gong-state]], [[mallet-plate-state]], [[airbox-vk-newton-state]],
[[vk-newton-batch-state]], [[air-box-state]].
