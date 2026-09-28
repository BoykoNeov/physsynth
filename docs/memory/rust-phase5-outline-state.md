---
name: rust-phase5-outline-state
description: Phase 5 batch 1 (guitar outline) — a DISCRETE output is a different porting problem; margin measured at 1.9e7 ulps (real guitars) vs 1 ulp (degenerate lens); portable spelling taken for the mask and refused for the 2e6-point quadrature
metadata: 
  node_type: memory
  type: project
  originSessionId: 6a3261d2-39f4-49a5-8cc1-fb3d5ee02f69
  modified: 2026-08-28T08:43:31.390Z
---

**Phase 5, batch 1, 2026-08-28.** Ported the *geometry* half of what was left of
`operators2d`: `guitar_half_width`, `guitar_scale`, `guitar_mask`, `guitar_area`, `live_cells`,
`cells_per_node`, `prune_to_area_carrying`. The **matrices** they serve
(`biharmonic_from_mask`, `orthotropic_biharmonic`, six private 1-D differences and
`free_plate_stiffness*`) plus `VonKarmanBracket`/`AiryStressSolver` are the next batch, and
`portable.canonical` goes with them (§18.4 pre-registered it — `B = L @ L` comes back from SciPy's
SMMP kernel with **descending** column indices, and `plate.py` multiplies by it every step).

**The finding — the batch boundary is drawn by what the output IS.** Every rule the migration had
before this one assumes the ported quantity is a *number* and the verdict is a tolerance:
"does the reduction reach the next timestep", "does anything branch on it", "does the nonlinearity
amplify or contract it". `guitar_mask` is `|x| < half(y)` at every node and its answer is **which
nodes exist** — a last bit is a *different plate*, and energy, nullspace, spectrum and area deficit
all pass on a plate with one node too few. So port the discrete-output functions as a group and
settle their arithmetic **before** porting anything that consumes them. See
[[guitar-plate-viewer-state]] and [[guitar-plate-state]] for the model side of the same statement.

**Measure the margin before choosing a spelling — it is one script and it settles the design.**
Over 130 shipped configurations (2 geometries × 5 outline parameter sets × 13 grids), smallest
`| |x| − half |` in ulps of `half`:

- real guitars: **1.9e7 … 1.9e10 ulps**, median 8.6e11 → exactness is **structural**, not luck
- **degenerate lens** (`waist = 0, asym = 0`): **1 ulp** at N = 32 (four nodes, at `t = 1/6` where
  `sin(π/6)` is ½ and the grid puts a node exactly there) and **0** at `t = ½`. Reachable — the
  viewer's waist sweep is `linspace(0.0, MAX, n)`.

**The portable spelling was taken for the mask and REFUSED for the quadrature** — §22.3's manoeuvre
a fifth time and the first where "where it is free" bites. `math.sin` (CPython) and `f64::sin`
(Rust) are the same libm call; `np.sin` on an array is a third implementation chosen by CPU. Cost
of the scalar spelling: 0.21→4.4 ms at `guitar_scale`'s 20,001 points (nothing), **45.9→467.4 ms**
at `guitar_area`'s 2,000,000 (half a second per plate). So `operators2d.py` carries `_profile`
(scalar, portable) *and* `_profile_vec` (NumPy, fast), and the rule that makes the split safe is
§19.2 again: **the consumer that branches gets the portable spelling, the consumer that averages
gets the fast one.** Both call sites are asserted to the bit so a tidy-up fails loudly
([[rust-phase3-bow-state]]'s hand-hoist hazard).

`guitar_area` is the one number **not** bit-identical, by decision: `np.sum` is pairwise over 2e6
terms, a Rust loop is not, and reproducing NumPy's blocking is the bargain §18.2 refused for
SciPy's SMMP. Measured 1.2e-13 relative. The native side gets a real oracle instead — with
`asym = 0` the profile integrates to `(2/π)(1 + w/15)` in closed form.

**A third pin, and it is a different kind.** Every parity assertion compares Rust against the
Python *as it is now*, so a transcription slip in `_profile` would be reproduced faithfully by the
port and agreed on by every one of them — and no physics bar can see a mask. So
`tests/test_guitar_plate.py` (the **default** suite, not the parity file) holds the **pre-change
NumPy expression written out verbatim** and asserts the shipped masks and their prunes node for node
against it — **with the degenerate lens held out**, because that pin compares two different `sin`
implementations and at the lens "the mask did not move" is a claim about which libm rounded; the
lens gets a 4-ulp bar on the half-width instead. So: **a pin must assert the weakest statement that
still catches what it is for** (a transcription slip moves the profile by orders of magnitude), or
it becomes a machine claim wearing a correctness claim's error message. General rule: **when a port
changes the reference, pin the reference's old behaviour somewhere the port cannot reach.** The same hole opens at the next batch,
where `portable.canonical` changes `plate.py`'s operator — but there the quantity is continuous, so
the pin is a tolerance on the spectrum rather than an equality on a mask.

**Two smaller scars.** `np.ascontiguousarray` promotes a **0-d** array to shape `(1,)`, so the
shared reader written in Phase 2 returned the wrong shape the instant a function was vectorised
over an arbitrary rank — the shape is now read from `asarray`. And **Phase 4 had never been through
the lint gate**: `cargo fmt --all --check` was red in 17 places and `clippy -D warnings` in 4, both
of which CI runs *before* any test, because §24.11's success condition listed the pytest
invocations and not the linters. Fourth time in six batches that CI was red for something visible
locally.

**And the whole-suite run found a leftover from Phase 4.** §24.7 fixed `boundary=None` in all six
bindings; `tests/test_rust_parity_bore.py` still held a Phase-2 test asserting the OPPOSITE, and it
stayed green for a whole batch because **the installed extension was older than the source** — the
fix was in `crates/` and never in `site-packages`. Two rules: **a batch that changes a shared
behaviour must re-run the tests that recorded the old one** (grep for the argument), and
**`pip install ./crates/physsynth-py` before believing any parity number**, because nothing in the
suite can tell a stale wheel from a fresh one.

Everything on this machine came out unmoved: **0** mask flips, **0** ulps on `scale`, **0**
relative change in `guitar_area` against the old spelling — the change buys portability, not a
different number. See [[rust-phase4-beam-state]] for what Phase 5 inherited and
[[numpy-libm-cpu-dispatch]] for why the spelling question exists at all.
