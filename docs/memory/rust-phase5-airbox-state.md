---
name: rust-phase5-airbox-state
description: "Phase 5 batch 6 — the AirBox room ported (first half of a file); np.sum's eight-element cutoff makes exactness decidable; the seam is 15 names wide"
metadata: 
  node_type: memory
  type: project
  originSessionId: d39181d4-a506-4a4c-888c-18e440745778
  modified: 2026-08-31T12:48:04.899Z
---

Phase 5, batch 6 (2026-08-31), plan §30: **`AirBox` is Rust**. This is the **first half of a
file** rather than a file — `airbox.py` is 3,976 lines and only the 3-D room ported; the ports
(`RoomPort`, `SurfacePort`, `InteriorSurfacePort`) and the six `RoomLoaded*` / `RoomSuspended*`
wrappers stay Python and work unchanged. So it is the first batch whose success condition is
"does the **seam** hold", not only "are the numbers right".

**The finding — count the terms.** `np.sum`'s pairwise blocking has a constant in it: **below
eight elements it is a plain left-to-right loop**. Measured 0 disagreements at n = 4, 6, 7 against
803/2,000 at n = 8, 1,398 at 56, 1,944 at 4,641. So whether bit-identity is available across a
ported `np.sum` costs **no measurement** — that is [[rust-phase3-barrier-state]]'s "how long is
the sum?" applied to a blocking rule rather than to a cancellation, and it does not depend on the
values. Consequences here: the smallest room the class builds has exactly **eight** pressure
nodes, so `acoustic_energy` is *never* structurally exact; a 1x1-cell wall face has four, so that
ledger **is** bit-identical over 2,000 steps; a one-node port books a sum of length one, exact.

**The bar.** Field bit-identical (`0.0` over 2,000 steps on five wall types, with cuts, with a
driven source, with a hand-built port injection, and from an exact discrete mode); energy books a
tolerance (~1e-16 relative). Affordable for a reason no earlier refusal had: `dissipated` and
`injected` are pure bookkeeping, so §14.2's "does the reduction reach the next timestep?" is
answered **no** for the first time on a reduction fed back into a running accumulator. The books
**wander in and out** of agreement at a last bit (one lossy wall differs on 59/2,000 steps and
finishes on the same double) — a one-shot equality check will mislead.

**The seam is fifteen names wide and the fifteenth is public.** Fourteen private (`_w`, `_W`,
`_Wx/_Wy/_Wz`, `_beta`, `_open`, `_has_walls`, `_pending`, `_pending_ports`, `_ports`,
`_cut_mask`, `_cut_index`, `_cuts`, `_register_cut`, `_plane_axis`, `_divergence`), **six of which
a client writes**, plus `source_index` — public, **assigned** by `test_airbox_freefield.py`, and
invisible to the private-name grep that found the other fourteen. Four searches now, none finding
the others: private names, re-derivation ([[rust-phase5-plate-state]]), duck-typed collaborator
types ([[rust-phase5-geometric-state]]), and written public attributes. **Grep for assignment, not
only for reference.**

**A binding that copies its buffers has made an algorithmic choice.** The first draft `to_vec()`d
`p`/`ux`/`uy`/`uz` every step; every answer stayed bit-identical and at one room size (41x33x25)
it ran **3.2x slower than NumPy** (0.31x). Borrowing the NumPy arrays across the whole step fixes
it: 13.1x at 27 nodes, 4.2x at 560, ~1.1-1.5x above 4,000. This is §29.2's "assert on the work"
arriving one batch after it was written, in the *binding* rather than in an algorithm.

**Two identity bugs no physics bar could see.** `Bound::clone` is a **reference** clone, so
`_cut_mask` and `_cut_index` built from one `PyList::new` were one object — `cut_faces` reported
375 instead of 136, a wrong *count*. And `source_index` was not settable. Both caught by the
**existing, unmodified** suite on the first flagged run.

**Two smaller ones.** Booking the wall flux as a per-step subtotal is `D + (f0 + f1)` where the
reference books per face, `(D + f0) + f1`; fixing that free association moved the gap 2.2e-15 →
1.4e-16, and the draft's comment had claimed an order it did not have. And `mode_shape`'s cosines
moved to `math.cos` — [[numpy-libm-cpu-dispatch]]'s portable spelling a **sixth** time and the
first aimed at an *initial condition* rather than a read-out, taken purely on the strength of the
written-down finding (0 differences in 2,239 values locally, so nothing here could justify it).

**What is left:** the second half of `airbox.py` (the ports and the six wrappers, ~3,000 lines and
all six `splu` factorizations — `AirBox` itself has none), then `connection`, then `analysis/`.
`connection` is still blocked: its bridges must accept `RoomLoadedBody`, `_PlateSurface` and
`RoomSuspendedPlate`, which are exactly the half not ported.

**CORRECTED 2026-08-31 by [[rust-phase5-ports-state]]:** the eight-element cutoff is
right and still the cheap question to ask, but the conclusion drawn from it — that
bit-identity is *unavailable* at eight terms or more — is wrong. NumPy's pairwise blocking
is one fixed algorithm and transcribes **exactly** (`crates/physsynth-core/src/reduce.rs`).
So the question is not "is exactness available?" but "do I need to transcribe the
blocking?", and §14.2's rule decides that: transcribe when the reduction reaches the next
timestep. The room's two energy books do not, so they were left a tolerance — a parked
tightening rather than a refusal. **Taken 2026-09-02** (merge of `claude/project-structure-
planning-t7aaro`): the books and `acoustic_energy` go through `reduce::sum_by`, a closure-reading
form that allocates no term array, and the parity assertions moved from `<= 1e-13` to `==`. The
parked list is empty.
