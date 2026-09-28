---
name: rust-phase5-ports-state
description: "Phase 5 batch 7 — the airbox PORT tier; NumPy's pairwise np.sum IS transcribable, so §30.2's refusal was avoidable and every port size is bit-identical"
metadata: 
  node_type: memory
  type: project
  originSessionId: 57d42e27-48fa-454f-bed9-a0b2310950c3
  modified: 2026-08-31T15:08:50.833Z
---

Phase 5 batch 7 (2026-08-31): `airbox.py`'s **port tier** — `RoomPort`, `SurfacePort`,
`InteriorSurfacePort` and the `_free_pressure_nodes` helper they share. The six `RoomLoaded*` /
`RoomSuspended*` wrappers above them are still Python and are the next batch. Plan `docs/dev/rust-migration-plan.md` §31.

**The finding, and it corrects the previous batch.** [[rust-phase5-airbox-state]] measured that
`np.sum` is a plain left-to-right loop below **eight** elements and declined to transcribe the
blocking above it, calling that "a claim about a library internal, and after §22.1 a claim about the
CPU as well". The measurement stands; the conclusion is **wrong**. NumPy's pairwise sum is one fixed
algorithm (eight accumulators seeded from `a[0..8]`, ragged tail into the *combined* result, split
at `n/2` rounded down to a multiple of 8) and transcribing it reproduces `np.sum` **exactly** — 0
disagreements over sixteen lengths from 1 to 40,000, over 3-D whole-array sums, and over a strided
reduction. The distinction from [[numpy-libm-cpu-dispatch]] is the point: a transcendental's last bit
is fixed by an **instruction selection** chosen per CPU; a summation's is fixed by an **order**,
which the unroll-by-eight exists to preserve under vectorisation. New module
`crates/physsynth-core/src/reduce.rs`.

It mattered here and not there because §14.2's question ("does the reduction reach the next
timestep?") is answered **yes** three times in this tier — `w = W/W.sum()`, `R_room`, and
`free_pressure` are all on the update path. So **every port size is bit-identical**, where the
prediction going in was "point ports exact, ball ports a tolerance."

**Why:** the risky half is now "NumPy's blocking is the same on every machine", and it is asserted in
exactly ONE named test (`test_numpy_pairwise_blocking_is_an_algorithm_not_a_kernel`, with
`_pairwise_sum` exposed from the crate for no other purpose) so a CI runner that disagrees produces
one diagnosis instead of §22.1's eighteen simultaneous red assertions.

**How to apply:**
- Reach for `reduce::sum` when a ported `np.sum` reaches the next timestep; keep the plain loop when
  it does not. Two tightenings were **parked on purpose** (plan §31.11) and **both are now taken**:
  the `powf(2.0)` literals in batch 9, and `airbox.rs`'s two energy books plus `acoustic_energy` on
  2026-09-02, through `reduce::sum_by` — a closure-reading form of the same blocking, so a caller
  whose terms are *computed* pays no allocation. The parked list is empty.
- **Take the callee first**, and the reason is §13.2 not dependency order: a wrapper calls
  `port.free_pressure()`, solves, then `port.inject(q)`, so a Rust *caller* over a Python callee is
  a `&mut self` pymethod handing control out mid-step, which PyO3 refuses.
- **Ask what a client DOES to the object, not only what it reads.** A test replaces the port's
  *methods* on the instance (`port.free_pressure = lambda: ...`), which no attribute grep finds and
  which a `#[pyclass]` refuses without `dict`. Getter/setter pairs are data descriptors and beat the
  instance dict; pymethods are non-data descriptors and lose to it — which is exactly what is wanted.
- **A diagonal inside a product is not neutral.** `T.T @ diags(R) @ T` left-associates, so SciPy
  forms `(T_ki R_k) T_kj` — different from `T_ki (R_k T_kj)` in 2,028 of 6,845 entries. The blind
  fixture is `spreading="nearest"`, and it is blind *provably*: its `T` rows are all-equal, and
  `(x d) x == x (d x)` identically when the outer factors are the same number.
- **Two SciPy routines in one expression disagree about explicit zeros**: `coo.tocsr()` keeps a
  stored `0.0`, `csr_matmat` prunes one. Hence `Csr::from_rows_keeping_zeros` alongside
  `from_rows`. No suite fixture contains an explicit zero, so nothing measured it.
- **§24.7's arm order bit again despite being written down**: with `Option<Option<_>>`, `Some(None)`
  means *omitted* and a bare `None` is the caller's literal `None`.
- The speed curve **does not converge**: `free_pressure` is 14.7x at 27 room nodes and still 3.7x at
  41,615, because the work is `O(patch)` in an `O(room)` array. Corollary — a binding that copied
  its buffers would be **asymptotically** wrong here, not merely slower: 678 us of copying against a
  12.9 us read at 139,995 nodes (see [[rust-phase5-airbox-state]] §30.5 for the constant-factor
  version).
