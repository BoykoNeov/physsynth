---
name: rust-phase2-state
description: "Rust migration Phase 2 batch 1 (exciter, membrane, operators2d builder half) — the first timestepping loop in Rust, the first module ported in HALVES, and the phase list is a risk map not a schedule"
metadata: 
  node_type: memory
  type: project
  originSessionId: 0c26d2cb-d5d4-4ef7-8d03-5cb3247880a7
  modified: 2026-08-26T15:35:42.457Z
---

Phase 2 batch 1 of the Rust migration landed 2026-08-26 (plan §11): `exciter` and `membrane` in
full, plus the **builder half** of `operators2d`. Continues [[rust-phase1-state]] and
[[rust-phase0-state]]; the overall shape is in [[rust-migration-state]].

**The phase list is a risk map, not a schedule — two ways.**

1. **A module can port in halves.** §4's groups are keyed to *files*, and a file's group is the
   group of its hardest function. `operators2d` is Group D (Phase 5) because of `VonKarmanBracket`
   and `AiryStressSolver`, which factor with SuperLU — but the five functions the membrane needs
   never solve anything, they assemble. So half the module shipped at Phase 2 and the solver half
   waits for the plate family. Expect the same at `plate`, and most consequentially at `airbox`
   (3,925 lines that are six factorizations surrounded by arithmetic that is not).
2. **`mallet` imports `collision`**, which is Group C — Phase 3. So **Phase 2 finishes after Phase
   3 starts.** Measured intra-`core` deps: `exciter`/`bore`/`body`/`engine` need nothing;
   `membrane` needs the `operators2d` builders; `reed` needs `bore`; `radiation` needs `body`.

**The headline finding: a sparse matvec inside a timestepping loop is STILL bit-identical**, over
2,000 steps with feedback, both domains, lossless and lossy. Phase 1 found matrix *builds* match;
this is the sharper version, because any 1-ulp slip would compound. The Rust CSR `matvec`
accumulates in ascending column index, which is what SciPy's kernel does. So the real line is
**"is the summation order knowable"**, not "is there a reduction" — and it survives a loop.
`energy()` still cannot match (`np.dot`/BLAS) and is held to the 1e-13 Group A target. Consequence
for Phase 3: **the first thing that breaks bit-identity will be a solver, not a step.**

**`physsynth-py` is now a SciPy client.** Phase 1's trick — return CSR triplets, rebuild in a
Python shim — works for a function return and NOT for `membrane.L`, which is a `csr_matrix` *on the
instance*. There is no call to wrap, so the Rust constructor imports `scipy.sparse` itself.
`physsynth-core`'s dependency list is still empty and still enforced.

**Two traps that every detector would have missed:**
- **`L` must be built ONCE.** `airbox._MembraneSurface.rhs` does `m.L @ m.u` every timestep; a
  getter that rebuilt the `csr_matrix` per access would assemble a sparse matrix inside the inner
  loop of the heaviest tests — every physics bar green, and the flagged run mysteriously *slower*
  than Python. Asserted by **identity** (`rs.L is rs.L`); a value comparison cannot see it.
- **A mask is a geometry decision, not a number.** `disk_mask` is a strict `<`, so a node one ulp
  from the rim changes the number of unknowns — and a membrane with one node fewer conserves energy
  just as beautifully. Compare masks **elementwise**, never through anything downstream. Same
  lesson as [[guitar-plate-state]]'s "the mask is not the outline", from the other direction.

**The clients WRITE the state, they do not just read it.** `mallet` does `mem.u[i] = ...` (in-place
single-element write through the property) and `airbox` does `m.u_prev = m.u` (rebind). Both halves
of Phase 0's buffer contract are load-bearing in 2-D. No membrane test makes either call — which is
why the gate names the clients.

**The `Resonator` trait is deferred to the `engine` batch, and polyphony was never the blocker** —
HANDOFF §11.3a settled it in 2026-08-10 (field models per instance, strings per voice); only the
voice-count budget is deferred and no signature depends on it. The reason to wait is §10.2's:
`engine` is the trait's only consumer, and `string_ideal` (1-D grid) and `membrane` (live-node
vector over a mask, plus `state`/`to_live`/`index_map` with no 1-D counterpart) share almost
nothing, so a trait fitted to that pair would be chosen by a scheduling coincidence.

**Smaller scars.** `cos` is the first transcendental — NumPy does not use platform libm for it, so
the raised cosines rest on two implementations agreeing (bit-identical here, asserted and swept).
`fmt.rs` exists because Rust's `{}` prints `1` for `1.0` and `0.00001` for `1e-5` where Python's
`repr` prints `1.0` and `1e-05`; the "error messages verbatim" convention was free until now.
`triangular_pluck` does not preserve a float32 dtype (Rust always returns f64) — latent, nothing
under `physsynth/` uses float32.

**The swap guard now DERIVES the set of ported names** from the `_py` aliases the module defines
and checks it against a written-down expectation, because `operators2d` is ported in part and
`__all__` no longer describes it.

**The speed prediction was measured and it is WRONG in the direction that matters for planning.**
Plan §7 said the test suite would get dramatically faster as models move. The first ported
timestepping loop makes the whole suite **3% SLOWER** (2,612 vs 2,534 core-seconds; same sign in
all three shards, and the same sign Phase 1 saw). The controlled `step()` benchmark says why:
**Rust removes the per-step interpreter overhead, not the arithmetic.** 8.7x at 69 unknowns
(Python needs ~5 us to step a 69-node membrane — the unknowns are not the cost, the dispatch is),
decaying to ~1.1x by 5,000 unknowns where SciPy's compiled matvec dominates. Crossover at N~80.
So: the suite's expensive tests are all on the wrong side of it, and the payoff is a **latency**
result (real-time, small grids, 48k steps/s) not a **throughput** one. Whether the gate ever
speeds up is now a Phase 4 question — the SuperLU factorizations — not a Phase 2 one.

Whole suite under the flag: **2,283 passed, 0 failed**, identical count on the default path.
