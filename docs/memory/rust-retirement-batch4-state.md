---
name: rust-retirement-batch4-state
description: Python retirement phase C batch 4 — the LINEAR wrapper tier; six classes become one generic; the name-join audit EXPIRES; an operand-order pin on a power of two is vacuous; np.allclose keeps rtol
metadata: 
  node_type: memory
  type: project
  originSessionId: b280e298-dfff-4ebb-9d57-fb6f01a497bc
  modified: 2026-09-08T14:29:09.423Z
---

Phase C batch 4 (2026-09-08), commit 1 of a **split the human chose** ("linear first, gong second"):
the ordinary plate and the drumhead — four of the six `RoomLoaded*`/`RoomSuspended*` wrappers and
two of the three seams — re-homed into `crates/physsynth-core/src/airbox_wrap.rs`, with 45 native
bars in `crates/physsynth-core/tests/airbox_grid.rs`. `tests/test_airbox_membrane.py` deleted whole
(20 functions), `test_airbox_surface.py` 19 -> 1 and `test_airbox_dipole.py` 28 -> 1 (each keeping
only the bridge chain, which is not ported). Plan: `docs/dev/python-retirement-plan.md` §16.
**The hole is now SIX**: the von Kármán seam + its two wrappers, and the three bridges.

**Six classes become ONE generic, and that is a design decision, not tidying.** The reference has
six wrappers only because Python has no generics — they are one body of arithmetic with two enum
arms (which face, therefore which port). Natively: a `GridSeam` trait (`PlateSeam`, `MembraneSeam`),
a `GridPort` enum, and `RoomGrid<S>` with `baffled`/`suspended` constructors. It drops no
distinction the reference made and removes five chances to drift. **The consequence is that
[[rust-retirement-batch2-state]]'s derived audit EXPIRES.** §13.1's "every `#[pyclass]` minus every
core `pub struct`" is a **name join**; it was only ever correct while every port was 1:1 and
name-preserving, and this batch broke that on purpose. Re-run today it still lists all twelve
remaining names, six of them already re-homed under new names. General form: **an audit built on a
name join measures the port's fidelity to a naming convention, not to a contract, and it expires the
first time a port is allowed to be a better shape than its original.** §16.7 carries the mapping
table; read the two together.

**An operand-order pin on a POWER OF TWO is vacuous.** The binding carried a careful comment pinning
`(2 R) q` over `2 (R q)`; mutating it changes **no digit anywhere** — 49 plate nodes, 1560 room
nodes, 200 ledger values, four scenes. Doubling a double is exact and binary rounding is
scale-invariant, so `round(2 r q) = 2 round(r q)`. The other two mutations were seen instantly
(re-associating the room's two load terms reddens everything; reversing the interior port's pressure
jump reddens exactly the two suspended scenes), so the check discriminates. The hazard is the same
note written about a constant that is *not* a power of two — `(0.5 k) / rho` vs `0.5 (k / rho)`
pins a real bit. Findings ledger #75.

**`np.allclose(a, b, atol=X)` KEEPS `rtol=1e-5`.** A retired bar read `atol=1e-18` and looks
absolute; what ran was `|a-b| <= 1e-18 + 1e-5|b|`, seven orders looser. Carried as the `atol` alone
the native bar failed — correctly, since the triple product leaves a node and its 180° image 2e-15
apart *relatively*, which the reference never asserted. Cure is not to relax to the reference's bar
but to write the inequality out and then choose (native keeps the relative form at 1e-12). Same
question for `pytest.approx` (rel 1e-6) and `assert_allclose` (rel 1e-7). Findings ledger #76.

**The one-time check: 65 of 72 quantities bit-identical, and the disposition of the 7 is MEASURED.**
Four scenes × 200 coupled steps, matrices as raw `indptr`/`indices`/`data`, plus the room's whole
pressure field. The only divergence is the coupling ledger `k * np.dot(pbar, q)` — BLAS `ddot`,
irreproducible at one of the four operand lengths by anything tried (serial, fused, `np.sum(a*b)`,
2/4/8-way blocking). But it **does not feed back**: every state array is exact while it differs at
1e-16. So it takes the crate's **read-out** spelling (`plate::dot`'s plain `s += x*y`), because
matching the room's own booking of the same identity matters more than matching a `ddot` that is
about to stop existing. Contrast [[rust-retirement-batch2-state]], where the one `dot` that *does*
reach the timestep had to be `mul_add`. Initial conditions were built from **integers**
(`1e-3((i mod 7)-3)/8`, not a Gaussian bump) so both sides start from identical doubles —
[[numpy-libm-cpu-dispatch]] applied in advance rather than diagnosed afterwards.

**`eliminate_zeros()` never fires, and that is provable rather than lucky.** The reference does
`(a_bare + load*scale).tocsc()` then `eliminate_zeros()`; measured at every reachable fixture —
including the zero-**area** surface the call's own comment says it exists for — `nnz` is unchanged.
`spread` drops zero weights so `T` has no explicit zero, `R` is strictly positive, and `T^T R T` is
a Gram matrix over positive weights. And `Csr::add` goes through `Csr::from_rows`, which drops exact
zeros — the same thing SciPy's `csr_binop_csr` does. So `nnz_growth` and `lu_nnz` are the *same*
quantities on both sides and match to the digit. Worth knowing: the load's sparsity **subsumes** the
plate's under `bilinear` but not under `nearest`, so the assembly genuinely needs a union merge and
a `bilinear`-only bar would never have found that.

**Two more "no analogue" verdicts, of different kinds** (batch 3's third verdict, still earning its
keep). A test that works by **replacing a live object's methods** has none once the object is a
value — the binding keeps that seam alive on purpose by giving every port class a `dict`. And a test
asserting an attribute is **absent** becomes a type: `not hasattr(inst, "pressure")` on the membrane
wrapper would not compile natively, because `pressure()` is inherent on `RoomGrid<PlateSeam>` alone.

**Goldens are two thirds carried, one third spent.** Construction digests (`node_count`, `nnz(T)`,
index-weighted sums) carry over unchanged and are a free cross-implementation check; the 200-step
run-end values are **not** re-frozen — the refactor they guarded shipped long ago and re-recording
from one machine promotes incidental digits to a cross-machine claim ([[rust-deletion-connection-state]]).
`RoomGrid::refactor` replaces five retired tests' hand re-derivation of `A_loaded`, so a deliberately
wrong coupling now goes through the *one* assembly spelling.

**DONE in batch 5 ([[rust-retirement-batch5-state]]). Was: Next (commit 2): the von Kármán seam + `RoomLoadedVKPlate`/`RoomSuspendedVKPlate`**, retiring
`tests/test_airbox_vk.py` (26 functions, 65.2 s) and whatever of `test_mallet_room_gong.py` has no
unported caller. Four things known in advance: the VK seam's `solve` **iterates** and its `commit`
takes `(w, F)`, so `GridSeam` must grow an arm — decide before writing bars; the
`nonlinear=False` ↔ `RoomLoadedPlate` `array_equal` anchor becomes a **native** bar and §16.2's
single transcription is what makes it cheap; `LoadedLu`/`ThetaSolve` has no native counterpart
because the loaded factorization *is* a `SparseLu`; and `test_mallet_room_gong.py` drives the
mallet client (`VkRoom` in `crates/physsynth-py/src/mallet.rs`), so its disposition is a question
about the mallet tier.

Related: [[python-retirement-state]], [[rust-retirement-batch3-state]],
[[rust-retirement-batch2-state]], [[rust-phase5-wrappers-state]], [[rust-phase5-membrane-state]],
[[air-box-state]], [[commit-push-at-batch-end]].
