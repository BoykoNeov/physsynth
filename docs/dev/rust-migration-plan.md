# Rust Migration Plan — retiring Python from `physsynth`

> **Decision (2026-08-26, the human's):** Python goes. All of it — core, analysis, viewer backend
> **and the test suite**. Target language **Rust**. New physics is written directly in Rust as soon
> as the harness is proven, not derived in Python first.
>
> This reopens non-negotiable **#3** in `CLAUDE.md` ("prototype in Python … not C++/JUCE yet") and
> the `portability-contract.md` clause that keeps Python as the permanent reference oracle. Both are
> superseded by this document. HANDOFF §9's Phase 5 is absorbed into it: the "port" is no longer a
> terminal phase, it is the spine of the next stretch of work.

---

## 1. The one rule that makes "delete the tests too" safe

The instruction is total, and read naively it destroys the thing that makes this project what it is:
1,808 tests are the *only* reason anyone believes the physics. Delete them alongside the code they
check and every ported model lands with nothing trustworthy to be checked against.

**The destination is not the order.** Python is retired *model by model*, and within each model the
oracle is the last thing to go, never the first:

1. Write the Rust model.
2. Expose it through a Python binding satisfying the same interface.
3. Run **the existing, unmodified Python tests** against the Rust model. Green means the Rust model
   reproduces the physics contract — asserted by the same code that asserted it for Python.
4. *Then* port that model's tests to Rust, and run both.
5. Re-point the **viewer** (`web/serialize.py`) at the Rust model — see §1.1, this clause is not
   optional — then, and only when the Rust tests are green **and** cover what the Python ones
   covered, delete the Python model and the Python tests together.

At no point does a model exist without a validated oracle behind it, and at the end there is no
Python left. The binding layer (`crates/physsynth-py`) is **temporary by construction** — it is
deleted when the last model finishes step 5.

### 1.1 The viewer is the binding's second consumer, and it breaks at Phase 0 without this

`web/serialize.py` imports `physsynth` **21 times** and builds core objects directly. Step 5 as
first drafted said nothing about it — so the moment `string_ideal.py` is deleted at the end of
Phase 0, the viewer breaks and `tests/test_web_backend.py` (403 tests) goes red for a reason that
has nothing to do with the port. Phase 8 is seven phases away and does not help.

So the binding is **not** scaffolding that only the tests use. It is the viewer's live dependency
for the entire migration, and its surface must satisfy the viewer's access pattern as well as the
suite's — measured in §3.1.

**The failure mode this rule exists to prevent:** porting the tests *first*, because they are the
biggest chunk and it feels like progress. A Rust test suite written against a Rust model that was
never checked against Python asserts whatever the Rust model does. It would be green, and it would
mean nothing.

### 1.2 Step 5 does not fire per model — a ported model waits for its *clients*

Discovered building Phase 0, 2026-08-26, and it generalises past the string.

§1.1 named the viewer as the thing that breaks when a Python model is deleted. It is not the only
one. `physsynth/core/connection.py` imports `IdealString` directly and reaches into
`_bc_right` and `_second_diff` — **private** attributes — and `connection` is a Phase 5 model.
`body.py`, `string_stiff.py` and `exciter.py` name it too (in docstrings, so those are free).

So the ritual's step 5 is structurally impossible for `string_ideal` at the end of Phase 0, and the
same will be true for `operators` at the end of Phase 1 and for every model something else builds
on. **Phase 0 ends with both implementations alive, and that is the correct end state, not a
shortfall.** A model's Python side comes out when its last *client* has moved, which for
`string_ideal` means after `connection.py` (Phase 5) and the viewer (Phase 8).

Two consequences worth stating rather than rediscovering:

- **The binding must expose private names.** `_bc_left`, `_bc_right` and `_second_diff` are part of
  the surface, because for the whole migration a Python module is a client of them. §3.1's table
  measured the *test* surface; this is the same measurement run over `core/` itself.
- **The switch is a better lever than a deletion.** Because `connection.py`, `body.py` and
  `web/serialize.py` all import the one name `IdealString`, flipping `PHYSSYNTH_RS` swings every
  dependent module onto Rust at once — so the port gets exercised through its real clients long
  before anything is deleted. Deletion stops being the milestone; a green run under the flag is.

---

## 2. Two things the founding documents promise that cannot be kept

### 2.1 "Agreement to ~1e-15" is not achievable and must stop being the bar

`portability-contract.md` says the ported kernel is correct iff it reproduces the Python numbers,
and to "check the new kernel against it to ~1e-15." For six of the models that is arithmetically
impossible: a sparse LU factorization's fill-reducing ordering and pivot choices differ between
libraries, so the per-step solution differs at ~1e-14 and the trajectories separate over 10^5 steps.
Even the models with *no* solve will not match bit-for-bit, because NumPy's `sum` uses pairwise
summation and its dot products go through BLAS — neither of which a hand-written Rust loop
reproduces by accident.

**The acceptance bar is the physics harness, which is what it was always for:** lossless energy
drift < 1e-10, partials within ~1 cent of the analytic oracle, convergence order ≈ 2. §6.1's
deliberate 50× headroom between the 1e-10 bar and the ~1e-15 typically observed is *exactly* this
situation, and is why that bar must not be tightened. Cross-implementation agreement is a
**diagnostic**, not the contract — see §4 for what to hold each solver group to.

**Corrected 2026-08-26, from Phase 0's measurements — the claim above is too pessimistic by half,
and the half it is wrong about is worth having.** The paragraph lumps two different things together
and they behave completely differently:

- **A step made only of elementwise arithmetic is bit-identical.** Measured on the ideal string
  over 4,000 steps, across all four boundary spellings and with and without loss:
  `np.array_equal(py.u, rs.u)` is `True`, not "close". With no reduction and no library call in
  the update, IEEE-754 fixes the answer exactly — *provided the operation order matches*.
  `(a - b) + c` and `a - (b - c)` are different functions in floating point, so that provision is
  real work, but it is work, not luck. The Rust kernels are written out longhand in NumPy's
  evaluation order deliberately, and `tests/test_rust_parity.py` asserts the result stays
  bit-identical — which turns an accidental reassociation into a hard failure rather than a drift.
- **Reductions cannot match, and that is where the ~1e-15 lives.** `energy()` calls `np.dot`, which
  goes through BLAS, which accumulates in an order no portable loop reproduces. Measured worst
  disagreement over the same runs: **7e-16 relative** — two orders inside the Group A target.

**The line is "does the step contain a reduction", not "is it Group A".** Getting this wrong the
other way would be just as expensive as the original over-pessimism, so it is worth being precise
before Phase 2 writes eight parity files: `body`'s modal displacement is a sum over mode
coefficients, and `mallet`'s contact force is a weighted sum over the membrane — both are `np.dot`
*inside the timestep*, so both belong in the ~1e-15 bucket even though neither solves anything.
Group A's other members look exact, but that is a per-model reading, not a group property. Check
the update for a dot product before asserting exactness on a new model; assert the tolerance if
there is one.

So the retraction stands where it was aimed — at the *solving* groups, and at any claim that a
whole trajectory reproduces to 1e-15 — and should not be read as forbidding a bit-exact comparison
where one is available. Where the arithmetic permits exactness, assert exactness: it is a far
sharper detector than a tolerance, and it costs nothing to keep.

### 2.2 The three enforcement tests change meaning, deliberately

`tests/test_stability.py` guards the portability contract with `test_core_is_headless`,
`test_core_dependency_allowlist` and `test_core_does_not_import_sibling_layers`. A compiled
extension module changes what all three *mean* — `physsynth_rs` is a third-party binary as far as
`sys.modules` is concerned, and the allowlist is hardcoded **for visibility** (the human's call, so
that a new name is a reviewed edit, not a silent one).

The replacement, which must be built in Phase 0 and not retrofitted:

- The Python-side allowlist gains `physsynth_rs` as a **deliberate, single, reviewed entry**.
- The rule itself migrates to the Rust side as a **Cargo dependency allowlist** — `physsynth-core`
  may depend on the numeric stack and nothing else; no I/O, no logging, no plugin framework. Checked
  in CI against `cargo metadata`, same spirit, same visibility.
- The one-way dependency arrow becomes a crate-graph property: `physsynth-core` depends on nothing
  of ours; `physsynth-py`, the CLI and the viewer depend on it and never the reverse.

---

## 3. The measured shape of the problem

Everything below is counted, not estimated.

| Component | Lines | Note |
|---|---:|---|
| `physsynth/core` | 13,413 | **43 % is docstrings** (5,703 lines) → **~7,710 lines of code** |
| `physsynth/analysis` | 1,930 | the analytic oracles — test-side reference math |
| `tests/` | 26,000 | **1,808 tests**, 5,027.9 core-seconds |
| `web/serialize.py` | 9,915 | viewer backend — glue, no physics |
| `web/static/app.js` + html/css | 6,314 | browser side; unaffected by the language change |

The core is the *smallest* piece. **The test suite is the migration**, at roughly 3.4× the core's
code volume, and any schedule that treats it as a tail-end chore is wrong by a factor of three.

### 3.1 How deep the tests reach into the objects

This number decides whether the binding is thin or leaky, and therefore whether step 3 above is
cheap:

| Surface | `tests/` | `web/` | Binding cost |
|---|---:|---:|---|
| `step` · `energy` · `set_state` · `state` · `displacement_at` | ~1,080 | 156 | trivial — the `Resonator` protocol |
| `.u`, `.u_prev` (raw state arrays) | 128 | 32 | zero-copy NumPy views via `rust-numpy` |
| `.K`, `.W`, `.B` (assembled operators) | 66 | 15 | return CSR triplets; a shim rebuilds `scipy.sparse` |
| `._lu` (the factorization itself) | 14 | 0 | **no clean binding** — port these tests to Rust early |

So: "the caller runs unchanged" is true for ~84 % of call sites, and the remaining **~255** (208 in
the suite, 47 in the viewer) need a binding designed for them. That is a day of design, not a
rewrite — but it is not free, and it is the reason Phase 0 exists.

---

## 4. The risk map: four solver groups, not one

An earlier summary of "seven models that solve a matrix" was too coarse and hid the actual risk
gradient. Measured per file:

**A file can appear in two groups** — `string_geometric` uses banded Cholesky *and* a sparse LU — so
the group sizes below do not sum to 22. That is real, not a miscount.

> **Correction, 2026-08-26, from building Phase 2.** The map above is keyed to **files**, and a
> file's group is the group of its *hardest function*. That is the right way to price risk and the
> wrong way to schedule work, because **the unit of porting is a function group, not a file.**
>
> Measured on `operators2d`: it is in Group D because of `VonKarmanBracket` and `AiryStressSolver`,
> which factor with SuperLU — but `membrane` is a Phase 2 model and the five functions it reaches
> for (`grid_coords`, `rectangle_mask`, `disk_mask`, `laplacian_from_mask`, `embed`) never solve
> anything. So the module ported **in half** at Phase 2 and the solver half waits for the plate
> family. Expect the same at `plate` (geometry vs the θ-scheme solve) and, most consequentially, at
> `airbox`, whose 3,925 lines are six factorizations surrounded by a great deal of arithmetic that
> is not. §11.2.1.

**Group A — no linear solve at all (10 files, ~2,700 lines).**
`string_ideal` · `membrane` · `bore` · `reed` · `mallet` · `body` · `radiation` · `exciter` ·
`engine` · `operators`.
Pure vectorized array arithmetic. The port is transcription. *Agreement target: state arrays to
~1e-13 relative over a short run; the physics bars thereafter.*

**`bow` is not in Group A** — a first pass put it there and that was wrong. It owns no matrix, but
it runs a **safeguarded Newton iteration** each step, seeded from the previous step's relative
velocity (continuation through the multivalued Helmholtz regime), and it **delegates a banded solve
to the string it bows**. So it ports after Group B, not with Group A, and its risk is the iteration
count matching — not the arithmetic.

**Group B — banded Cholesky (4 files).**
`string_stiff` · `string_damped` · `string_nonlinear` · `string_geometric`, via
`cholesky_banded` / `cho_solve_banded` — LAPACK `pbtrf`/`pbtrs`.
**This group is nearly free.** Banded Cholesky has *no pivoting* and a fixed elimination order, so
linking LAPACK from Rust reproduces it essentially exactly; hand-writing it is ~50 lines and still
deterministic. *Agreement target: same as Group A.*

**Group C — dense LU (1 file).** `collision`, via `lu_factor`/`lu_solve` (LAPACK `getrf`/`getrs`).
Partial pivoting, but deterministic given the same LAPACK. Note the existing scar: `scipy.lu_solve`
was chosen over `np.linalg.solve` because of a NumPy 2.4 BLAS cliff — the Rust side must not
silently reintroduce the slow path.

**Group D — sparse LU / SuperLU (6 files, ~7,600 lines).**
`beam` · `plate` · `operators2d` · `connection` · `string_geometric` · `airbox`.
`scipy.sparse.linalg.splu` **is SuperLU**. This is the only genuinely hard group: fill-reducing
ordering (COLAMD) and partial pivoting are library-specific, so a different solver gives a different
— equally valid — answer, and the two trajectories separate. It also contains the largest file in
the project (`airbox`, 3,925 lines, six factorizations).

### 4.1 The manoeuvre that collapses Group D's risk

**During migration, link SuperLU itself.** It is a C library, and SciPy wraps the same one. If it
can be called from Rust reproducing SciPy's own defaults, the Group D models can be held to the
*same* tight agreement as Groups A–C, and any divergence is a bug in the port rather than a property
of the solver — the comparison stays sharp exactly where it is hardest to keep sharp.

**This is a hypothesis Phase 4 tests, not a risk already collapsed.** Matching SciPy means matching
its column-ordering choice (`permc_spec`), its `diag_pivot_thresh`, and its equilibration defaults —
none of which have been checked yet. That is precisely why `beam` (254 lines, one factorization) is
Phase 4: the smallest possible surface on which to find out. **If the manoeuvre fails there, Group D
falls back to tolerance-level agreement validated by the physics bars alone** — which is survivable,
but it is a different and looser comparison, and it should be discovered on 254 lines rather than on
the 3,925-line room.

**After migration, swapping solvers becomes a normal optimization.** Once the Rust test suite is the
authority — physics bars, not Python — `beam`, `plate` and `airbox` can move to a pure-Rust solver
([`faer`](https://docs.rs/faer/latest/faer/), which has sparse LU) or to
[`klu-rs`](https://github.com/pascalkuthe/klu-rs), whose design point (*sparsity pattern fixed,
values change*) is precisely this project's usage: factor once at construction, back-substitute
every step. Several of these matrices are **SPD**, so a sparse Cholesky is both faster and
pivot-free — but changing the algorithm changes the numbers, so it happens **after** the comparison
is retired, never during.

---

## 5. Order of work

The repo already established the principle: *the beam de-risked the plate.* Same move throughout —
the smallest member of each new risk class goes first, purely to prove the machinery.

**Phase 0 — the harness, proven on one trivial model.** *(Landed 2026-08-26 — see §9.)*
`crates/physsynth-core` + `crates/physsynth-py` (PyO3/maturin), CI compiling on push, the
allowlist tests redefined per §2.2, and **`string_ideal` (206 lines)** ported. Success is not
"a string works" — it is *the existing ideal-string tests passing unmodified against Rust*.
Nothing else starts until that is green.

> **Correction, 2026-08-26.** This phase originally named `tests/test_string_ideal*.py` as the
> thing that has to go green. **There is no such file and there never was.** The ideal string's
> tests are spread by *criterion*, not by model — `test_energy.py` (conservation and passivity),
> `test_modal.py` (frequencies vs the analytic oracle), `test_convergence.py` (order of accuracy),
> `test_dispersion.py`, and `test_stability.py` (CFL and construction guards) — **38 tests across
> five files.** That layout is deliberate and predates the migration, so every later phase should
> expect the same shape: a model's tests are not in a file named after it. Look them up rather than
> guessing the filename; a success condition naming a path that does not exist is one `pytest` away
> from being vacuously green.

**Phase 1 — `operators` (165 lines).** *(Landed 2026-08-26 — see §10.)* Shared infrastructure
everything downstream needs.

> **The success condition is not the file named after the module.** `tests/test_operators.py` is
> four tests, and it touches `inner`, `norm2`, `delta_x_forward` and `delta_xx` — none of
> `delta_xxxx`, `second_difference_matrix`, `biharmonic_matrix` or `free_beam_stiffness`, which are
> the four hardest things in the phase. This is Phase 0's §5 correction repeating itself one phase
> on, and it generalises: **a module's tests are not in the file named after it, and neither are
> its clients'.** What actually exercises the ported code is `test_stiff_string.py`,
> `test_damped_string.py`, `test_tension_string.py`, `test_geometric_energy.py`,
> `test_beam_modal.py`, `test_beam_stability.py`, `test_bow_stability.py`,
> `test_free_plate_modal.py` and `test_free_plate_orthotropic.py` — because `operators` is not a
> model, it is what five models are *built out of*.

**Phase 2 — Group A, the remaining explicit models.** Eight files. Bulk transcription, low risk, and
the phase where the Rust idiom for this project settles (state layout, the `Resonator` trait, how
`energy()` is expressed). Get the idiom right here; every later model inherits it.
*(Batch 1 landed 2026-08-26 — `exciter`, `membrane` and the builder half of `operators2d`; see
§11. Batch 2 — `body` — §12. Batch 3 — `bore` and `reed`, the wind leg — §13. Batch 4 —
`radiation`, minus its one Bessel helper — §14. What is left in the phase is `engine`, deferred
for the design reason in §14.10, and `mallet`, blocked on Phase 3.)*

> **Correction, 2026-08-26.** The eight files are not independent, and one of them is blocked on a
> later phase. Measured intra-`core` dependencies: `exciter`, `bore`, `body` and `engine` need
> nothing; `membrane` needs the `operators2d` builders; `reed` needs `bore`; `radiation` needs
> `body`; and **`mallet` needs `collision`**, which is Group C — Phase 3. So the honest statement of
> the order below is that **Phase 2 finishes after Phase 3 starts.** §1.2 found that a ported model
> waits on its *clients*; this is the same thing seen from the other side, and neither relation is
> encoded in the phase numbering. §11.2.2.
>
> `engine` is also deferred within the phase, deliberately and for a design reason rather than a
> scheduling one — see §11.3.

**Phase 3 — Group B + C, then `bow`.** `string_stiff` first (259 lines, the smallest banded solve),
then `string_damped`, `string_nonlinear`; then `collision`. Cheap, and they prove the LAPACK link.
`bow` comes last in the phase because it borrows the string's banded solve and cannot be checked
before the string is trustworthy.

> **Correction, 2026-08-27, from building the phase.** The order above cannot be run, and the
> obstacle is in `tests/`, not in the models. `string_stiff`, `string_damped`, `string_nonlinear`
> and `string_geometric` are chained by three **bit-identity reduction anchors** — `sigma1 = 0`,
> `EA = 0`, `EA = T` — each asserting `array_equal` between two *different* model classes. Porting
> any one of them turns an intra-Python comparison into a cross-language one, and the banded solve
> cannot carry that (OpenBLAS's `DTBSV` is a blocked kernel; no scalar transcription reproduces it
> — §15.3). So the four are indivisible under a change of solver, and one of them is Group D.
>
> The phase therefore starts by porting the **solver** rather than a model: `physsynth/core/banded.py`,
> the Phase 1 manoeuvre one level down. All four models change arithmetic together, every anchor
> stays valid, and no sparse LU is touched — so `beam` keeps its §4.1 de-risking job. §15.
>
> Also: "they prove the LAPACK link" is not what happened. **Nothing is linked.** The allowlist is
> still empty, and Group B's banded Cholesky is ~150 lines of transcription that needs no library.
> §4's "this group is nearly free" was right; its assumption that freedom would come from linking
> LAPACK was not.
>
> **Second correction, 2026-08-27, from batch 3.** With the solver common, §15.9 expected the
> anchors to stop being the obstacle. They did not: they bind on **evaluation order**, twice over,
> and only one of the two orders was the solver's. The energy anchors compare `np.dot` reductions
> and the trajectory anchors compare a sparse matvec whose accumulation order SciPy leaves
> *descending*. Both were resolved the way the solver was — by moving all four models to one
> spelling at once, this time on the Python side (`physsynth/core/portable.py`) — after which
> `string_stiff` and `string_damped` ported normally. §18.
>
> **Third correction, 2026-08-27, from batch 5.** "then `bow`" is right about the bow and wrong
> about the phase: `bow` ported last of the models §16 could see, but `collision.BarrierString`
> was described there as waiting on its host `DampedStiffString`, that host landed in §18, and the
> sentence was never revisited. So **`BarrierString` is the phase's last model**, and Phase 3 is not
> finished when the bow is. §20.11 — the general shape being that a statement justified by a
> dependency expires when the dependency lands, and nobody is notified (§18.3).
>
> **Phase 3 is complete, 2026-08-27.** `BarrierString` landed in §23 and with it Group B and Group
> C are done. The phase ran: the *solver* (§15), the contact leg (§16), the two theta-scheme
> strings (§18), the tension-modulated string (§19), the bow (§20), the barrier (§23) — six batches
> against the four the order above imagined, and the two extra ones are both explained by the same
> thing: what a model waits on is not visible from the model. `string_geometric` is the only string
> left and it is **Phase 5**, not a Phase 3 leftover — it needs a sparse LU as well as the banded
> Cholesky, so it needs Group D.

**Phase 4 — `beam` (254 lines, one `splu`).** The Group D de-risker, chosen for exactly the reason
it was chosen the first time. Proves the SuperLU link and the §4.1 manoeuvre on the smallest
possible surface.

**Phase 5 — the rest of Group D except the room.** `operators2d` · `plate` · `connection` ·
`string_geometric`.

**Phase 6 — `airbox` (3,925 lines, six factorizations).** Last and largest, on machinery that six
phases have already proven.

**Phase 7 — `analysis/` (1,930 lines).** The analytic oracles: Bessel roots, closed-form
eigenfrequencies, the spectrum detector. Pure math, no state. Deliberately late — while Python still
holds these, the Rust models are being checked against an oracle that hasn't moved.

**Phase 8 — the viewer backend (9,915 lines).** Becomes a Rust HTTP server. The browser side
(`app.js`, 5,719 lines) is untouched by any of this — it speaks JSON and does not care.

**Throughout — the tests.** Each model's tests port in its own phase, at step 4 of §1's ritual.
Never batched, never ahead of the model.

---

## 6. When the frontier flips

**At the end of Phase 2.** Once the explicit-model idiom has settled and the harness has carried ten
models, all new physics is written directly in Rust — including the two outstanding plate shapes
(the clamped rim, and the nonlinear gong equations on a guitar outline).

This is the decision that determines whether Python actually shrinks. If new models keep being
derived in Python and then transcribed, the pile never gets smaller — it just gets a second copy,
and the migration becomes permanent duplication with no end state. The flip is what makes this a
migration rather than a mirror.

**The cost of flipping, stated honestly:** deriving new physics in Rust is slower than in Python, and
this project's method is exploratory — write the energy identity, discretize, watch the drift, be
wrong, adjust. Losing the fast loop is a real tax on *new* research, paid to make the existing
research permanent. The two plate shapes are the first models to pay it.

---

## 7. Second-order consequences worth pricing now

**CI gains a compile step**, on a gate that already runs 3 concurrent shards / ~5,027 core-seconds.
Rust compile times for a numeric crate of this size are minutes, and the cache is the difference
between tolerable and not.

**But the suite should get dramatically faster** as models move. The gate is bulk-bound at ~99 %
core utilisation, and the bulk is FDTD timestepping — the thing Rust is good at. This is a genuine
upside, not a consolation: the 15–21-minute gate is a Python cost.

> **Measured at Phase 2 and it is wrong — see §11.6, sharpened by §12.7.** The first ported timestepping loop makes the
> whole suite **3 % slower**, not faster, and the controlled measurement says why: Rust removes the
> per-step *interpreter overhead* (8.7× on a 69-unknown membrane) and not the arithmetic (~1.1× by
> 5,000 unknowns, where SciPy's compiled matvec dominates). The suite's expensive tests are all on
> the wrong side of that crossover. The claim that survives is narrower and more useful: **small
> grids stepped very often get dramatically faster**, which is the real-time case (§8, HANDOFF §9
> Phase 8) and not the gate. Whether the gate speeds up is now a Phase 4 question — the SuperLU
> factorizations — not a Phase 2 one.

**Cargo test parallelism is threads, not processes**, unlike pytest-xdist's shards. The sharding
machinery — computed from a glob, guarded inside and out — does not carry over and should not be
recreated speculatively. But note what else the shards were quietly providing: **process
isolation**. Cargo runs the whole suite in one address space, and the airbox runs are memory-heavy,
so expect **memory pressure, not test count, to be the constraint that replaces sharding** — and
expect to discover it at Phase 6, where the room lands.

**The four non-reproducible oracles.** `beam_low_eigenfrequencies` and its siblings call ARPACK
(`eigsh`) with no fixed start vector, so the reference frequency wobbles in its last digits run to
run (measured: 71.19597710094551 → 71.19597710083288). Harmless against a 5-cent bar, but it means
the *oracle* is not bit-reproducible — which will read as a Rust-vs-Python discrepancy the first
time someone tightens a comparison. Fix it (pass `v0`) before Phase 4, not during.

---

## 8. Open, and deliberately not decided here

- **Which models are ever meant to run in real time.** The 3-D room's cost scales as `h^-4` and it
  dictates the sample rate of everything coupled to it; it will not be real-time in any language.
  Rust makes the *plugin* possible but does not make every model playable, and "which subset ships"
  remains an unanswered scoping question — now decoupled from the language decision, which is the
  main thing this document changes about it.
- ~~**Polyphony** (HANDOFF §11.3a) is still open and is an *engine-shape* question. It should be
  settled before Phase 2 fixes the `Resonator` trait, or the trait is rebuilt later.~~
  **Checked 2026-08-26 and it is not open:** HANDOFF §11.3a settled it — field models per
  instance, strings per voice — with only the voice-count *budget* deferred, and that is a
  real-time-stage concern. So the trait is not blocked on the human. It is deferred to the batch
  that ports `engine`, because that is its only consumer and §10.2's reasoning applies: do not fix
  an interface before the requirement that ought to choose it exists. §11.3.
- **The plugin framework** (`nih-plug`/CLAP vs a bespoke host) — still a genuine plugin-stage
  decision, and nothing before Phase 8 depends on it.

---

## 9. Phase 0, as built (2026-08-26)

What exists, what was chosen and why, and the two things that were only discoverable by building it.

### 9.1 The shape on disk

```
Cargo.toml                              workspace; Cargo.lock IS committed
crates/physsynth-core/                  the DSP core. Zero dependencies.
  src/ops.rs                            delta_x_forward, inner — the two the string calls
  src/string_ideal.rs                   Params · kernels · a native owning IdealString
  tests/string_ideal.rs                 native physics bars (13 tests)
  tests/deps.rs                         the Cargo dependency allowlist (§2.2's Rust half)
crates/physsynth-py/                    the PyO3 binding, exposed as `physsynth_rs`
tests/test_rust_parity.py               Rust vs Python (75 tests) — the diagnostic, not the bar
physsynth/core/string_ideal.py          + a swap block at the bottom, gated on PHYSSYNTH_RS
```

Each model splits three ways, and Phase 2 should keep the split rather than reinvent it:
**parameters** (a validated immutable struct — all derivation and every rejection), **kernels**
(free functions over `&[f64]`, holding no state), and **a native owning struct** (parameters plus
`Vec` state, for Rust callers and `cargo test`). The binding does **not** wrap the owning struct —
it calls the kernels directly, because its buffers have to be something else entirely. §9.3.

### 9.2 Toolchain, and the one thing that had to be proven before any physics was written

`pyo3` 0.29 · `rust-numpy` 0.29 · `maturin` 1.15, built **`abi3-py311`**.

The `abi3` choice is not cosmetic. One wheel then serves every interpreter from 3.11 up, so the
development machine (3.14) and CI (3.12, plus 3.11 on the nightly) exercise **the same binary** —
which deletes an entire axis of "it only fails on the runner" before Phase 6 can land on it.
Verified by import and round-trip on 3.13 and 3.14 locally, and on 3.12 in CI.

That verification came *first*, before a line of the string was written, as a throwaway extension
doing nothing but handing a zero-copy array back to Python. §3.1 prices 160 of the 255 hard call
sites on zero-copy views working under `abi3`; if they had not, the binding design would have been
a different one and finding out at Phase 5 would have been expensive.

**Two Python distributions, deliberately.** The root `pyproject.toml` stays on hatchling and stays
pure Python; the binding ships as its own maturin-built distribution. Converting the root would
have grown a Rust compile onto all three existing gate shards and — worse — made the pure-Python
side no longer installable alone. That side *is* the baseline every ported model is measured
against, so it has to stay standalone for as long as the comparison matters.

### 9.3 The finding: who owns the buffers, and why the obvious answer is a use-after-free

This is the load-bearing design decision of the whole binding layer, and it is invisible to every
physics test in the repo.

Python's `step()` **rebinds** `self.u = u_next`; it does not write into the existing array. Two
properties follow, and real callers depend on both:

1. A reference taken *before* a step stays valid afterwards and keeps showing that step's values.
2. A write *through* `.u` reaches the string — which is exactly how a bridge applies its force:
   `self.string.u[-1] -= self.beta_s * F`, at four sites in `connection.py`.

A Rust struct holding `Vec<f64>` state cannot honour both, and both wrong answers are silent:
handing out a **copy** loses the write into a temporary, and handing out a **zero-copy view** over
a `Vec` that a later step reallocates is worse — it is a use-after-free. Measured while probing
this: a view held across a `Vec` reassignment *still returned the old contents*, which is precisely
what a correct snapshot looks like, right up until the allocator reuses the page.

So the binding's state buffers are **NumPy arrays owned by Python**; the type holds `Py<PyArray1>`
handles and `step()` allocates a fresh array and rebinds. Lifetime is then refcounted by the
interpreter, which is the only thing that actually knows who is still holding what. This costs
nothing: it is the same allocation pattern the Python original already had.

**None of the 38 ideal-string tests can see any of this** — they all reach the state through
`state`, which copies. `tests/test_rust_parity.py` therefore asserts the buffer-lifetime properties
directly, against *both* implementations, so the claim is about behaviour rather than about Rust.
Every later phase inherits the same hazard the moment its model holds state.

### 9.4 What was measured

| | |
|---|---|
| The 38 ideal-string tests, unmodified, under `PHYSSYNTH_RS=1` | **38 passed** |
| **The WHOLE suite under `PHYSSYNTH_RS=1`** | **1,954 passed, 0 failed** |
| Native `cargo test` (physics bars + allowlist + unit) | **19 passed** |
| `tests/test_rust_parity.py` | **75 passed** |
| State agreement, 4,000 steps × 4 boundary spellings × loss on/off | **bit-identical** |
| Energy agreement, worst relative | **7e-16** (Group A target: 1e-13) |
| The 38 ideal-string tests, wall clock, same machine | **23.7 s → 4.2 s** |
| The whole suite, wall clock, same machine, back to back | **421.8 s → 382.4 s** (−9.3 %) |

**The second row is the phase's real result, and it was not the expected one.** The plan budgeted
for a *list of failures* here — the files whose access patterns the binding did not yet satisfy,
to be recorded as the surface spec and left for Phases 3–8. There were none. Every consumer is
green: `connection.py`'s reach into the string's private `_bc_right` and `_second_diff`, the
sympathetic-string bridge, the collision models, the string↔plate and string↔room chains, and all
403 web-backend tests. §3.1's "~255 call sites need a designed binding" is answered for this model:
they needed a designed binding, they got one, and nothing else needed changing.

That claim is only worth as much as the swap actually reaching those consumers, so it was checked
rather than assumed — under the flag, `helpers.make_string` returns a `physsynth_rs.IdealString`,
and `connection.IdealString`, `body`'s and `web.serialize.IdealString` are all *the same object* as
`physsynth_rs.IdealString`. `tests/test_stability.py::test_the_rust_swap_matches_the_environment`
now asserts this on both paths, so a run that claims to be testing Rust cannot quietly be testing
Python — which is the §1 failure mode one level up, and the only way this phase could go green
while meaning nothing.

**The two wall-clock rows should be read for direction, not for size.** Both are one machine, both
are one model out of twenty-two ported, and neither was repeated — the whole-suite delta in
particular is 9 % against an uncontrolled back-to-back pair, which is not separable from run-to-run
noise on its own evidence and should not be quoted as if it were. What they *do* establish is a
**floor**: the suite did not get slower, the default path is unchanged (1,954 green both ways), and
the one slice that is genuinely all-Rust moved 5.6×. §7's prediction that the bulk is a Python cost
is still a prediction; the number that will test it is Phase 2's, when ten models have moved.

The 38th test is worth naming: `test_core_dependency_allowlist` **failed first**, on
`physsynth_rs`. That is the §2.2 tripwire working — a new compiled dependency of the core could not
appear without a human editing the allowlist, which is the entire point of it being hardcoded. The
entry was then added deliberately, with its reasoning, in the same commit.

### 9.5 What Phase 1 inherits

- **The idiom above is settled but not yet stressed.** It has carried one model with no linear
  solve and no shared operator infrastructure. Phase 1 (`operators`) is where the sparse matrix
  builders arrive and the first real dependency decision gets made — and `ALLOWED` in
  `crates/physsynth-core/tests/deps.rs` is deliberately **empty**, so that decision is an edit
  someone has to write a reason next to.
- **`ops.rs` is deliberately incomplete.** Two functions, because the string calls two. The other
  four pointwise differences and all three matrix builders are Phase 1's, and leaving them visibly
  absent is better than transcribing them now and having them look finished.
- **The `PHYSSYNTH_RS` switch is the lever, not the deletion.** Per §1.2, nothing gets deleted for
  a long time. What the switch buys is that flipping it swings `connection.py`, `body.py` and the
  viewer onto the Rust string too — so the ported model is exercised through its real clients well
  before its Python twin comes out.

---

## 10. Phase 1, as built (2026-08-26)

`physsynth/core/operators.py` in full: the four pointwise differences, `inner`/`norm2`, and the
three sparse builders. What follows is what was chosen, what was measured, and the three things
that were only discoverable by building it.

### 10.1 The shape on disk

```
crates/physsynth-core/src/sparse.rs     a hand-written CSR type. Zero dependencies. §10.2
crates/physsynth-core/src/ops.rs        the module, complete — Phase 0's stub filled in
crates/physsynth-core/tests/ops.rs      native operator bars (14 tests)
crates/physsynth-py/src/lib.rs          + nine free functions; the matrices come back as TRIPLETS
physsynth/core/operators.py             + a swap block at the bottom, gated on PHYSSYNTH_RS
tests/test_rust_parity_operators.py     Rust vs Python (70 tests)
```

The §9.1 three-way split (parameters · kernels · owning struct) does not apply here, because
`operators` has no state and therefore no owning struct. What it has instead is the seam the rest
of the migration will keep hitting: **a return type that Python cares about and the core must not
know exists.** `physsynth-core` returns a `Csr`; the binding turns it into
`(data, indices, indptr, shape)`; the shim at the bottom of `operators.py` rebuilds a
`scipy.sparse.csr_matrix`. The core never learns what SciPy is, and the five modules that
`from .operators import ...` never learn what Rust is.

### 10.2 The dependency decision that was deferred rather than taken

§9.5 said Phase 1 was where "the first real dependency decision gets made" and left `ALLOWED`
empty so that it would have to be written down. **The decision was to keep it empty**, and the
reasoning is worth recording because it comes up again at Phase 3 and Phase 4:

Phase 1 only ever *constructs* matrices — transpose, multiply by another sparse matrix, scale.
It never solves with one. The constraint that should actually pick a sparse library is a **solver**
constraint (banded Cholesky at Phase 3, the SuperLU hypothesis at §4.1/Phase 4), and none of it has
been measured yet. Taking `faer` or `nalgebra-sparse` now, to get a `matmul`, would fix the
interchange type before the requirement that ought to choose it exists. So: ~200 lines of CSR with
no dependencies, and when a solver does land, this type becomes the thing that converts *into*
whatever that solver wants. That is a smaller commitment than the reverse, and it is reversible.

### 10.3 The findings

**A sparse matrix build is a reduction, and it still comes out bit-identical.** §2.1's correction
draws the line at "does the step contain a reduction". Each entry of `D2 @ D2` is a sum of up to
three products, so by that rule the matrices belong in the ~1e-15 bucket. They do not: `data`,
`indices`, `indptr` and `nnz` all match SciPy exactly, at eight grid sizes and on a
non-power-of-two grid. The line is therefore sharper than "reduction or not" — it is **whether the
summation order is knowable**. BLAS's is not; a three-term sum over an explicit sparsity pattern
is. The Rust `matmul` accumulates in ascending order of the contracted index, which is what SciPy's
SMMP kernel does. (Measured separately, and worth knowing: for *these* structures ascending and
descending accumulation give the same answer anyway, so the exactness is not balanced on that one
choice.)

**SciPy's own output is not canonical, and copying that would have been a mistake.** Measured:
`biharmonic_matrix` comes back from `(d2 @ d2).tocsr()` with `has_sorted_indices == False` and its
columns in *descending* order — SMMP's output list is a stack — while `free_beam_stiffness`, whose
product goes through a transpose, comes back sorted. Reproducing that split would mean
reimplementing a SciPy internal and pinning the port to a detail a point release is free to change:
a red gate on an upgrade, for a non-bug. The Rust side is canonical in both cases and the parity
test canonicalises the SciPy side before comparing. What is *not* relaxed is the arithmetic — the
values are still required to be equal bit-for-bit, which is the part that can actually be wrong.

**`h ** 4` is not `h*h*h*h`.** Python's `**` calls libm's `pow`, which returns the
correctly-rounded fourth power; three chained multiplications round three times and land elsewhere.
Measured over `h = 1/N` for `N = 2..3999`, the two disagree in **1400** of 3998 cases (and
`(h*h)*(h*h)` in 1934). So `delta_xxxx` says `powf(4.0)`, and it is the one kernel in the module
whose exactness rests on two libms agreeing rather than on IEEE-754 alone. The parity test sweeps
`N` for that reason: a platform whose `pow` differs shows up there as a 1-ulp mismatch rather than
as a physics failure three phases later.

**And one scar, from the native bars rather than the port.** The free beam's stiffness `K`
annihilates the rigid-body space `{1, x}` — that *is* the free-free boundary condition, and the
first draft of the test asserted `K @ 1 == 0` exactly. It is not. `D2 @ 1` cancels exactly in
IEEE-754, but `K` is an *assembled* matrix: applying it sums a row of already-rounded entries
rather than re-deriving the cancellation. 8.2e-12 against an operator of scale 1/h³ = 8000, i.e.
~1e-15 relative. The claim the builder actually supports is annihilation to rounding, stated
relative to what the same operator does to a field that genuinely bends.

### 10.4 The success condition, and the trap it walked into again

Phase 0's §5 correction records that the phase originally named a test file that did not exist.
Phase 1's version of the same mistake is subtler and was live until it was checked:
`tests/test_operators.py` **does** exist, and it is the wrong file. Four tests, touching `inner`,
`norm2`, `delta_x_forward` and `delta_xx` — **none** of `delta_xxxx` or the three matrix builders,
which are the four hardest things in the phase. A gate naming it would have been green and would
have asserted nothing about the difficult half.

Generalised, for every later phase: **a module's tests are not in the file named after it, and
neither are its clients'.** `operators` is not a model; it is what five models are built out of.
The files that actually exercise the ported code are `test_stiff_string.py`,
`test_damped_string.py`, `test_tension_string.py`, `test_geometric_energy.py`,
`test_beam_modal.py`, `test_beam_stability.py`, `test_bow_stability.py`, `test_free_plate_modal.py`
and `test_free_plate_orthotropic.py`, and they are in CI as a named claim rather than a partition.

**The guard against the swap being a no-op is wider here too.**
`test_the_rust_swap_matches_the_environment` gained an operator arm, and it could not be a copy of
the string's: operators are functions, so there is no class identity to compare — the question is
whether the public name still *is* the `_py` one. It also asserts something the string's version
had no need to, because the string has one importer and the operators have five:
`string_stiff.biharmonic_matrix is operators.biharmonic_matrix`, and the same for `beam`. Those
five modules capture their operators at import time, so a swap that landed after them would leave
all five on Python while `operators` reported Rust — green, and testing the wrong thing for five
models at once.

### 10.5 What was measured

| | |
|---|---|
| Native `cargo test --workspace` | **42 passed** (19 at the end of Phase 0) |
| `tests/test_rust_parity_operators.py` | **70 passed** |
| Matrix agreement — `data`, `indices`, `indptr`, `nnz`; 8 sizes × 2 grid spacings | **bit-identical** |
| Pointwise-difference agreement — 4 operators × 5 sizes × 2 fields | **bit-identical** |
| `inner` / `norm2` agreement, worst relative | inside the 1e-13 Group A target |
| **The WHOLE suite under `PHYSSYNTH_RS=1`** | **2,026 passed, 0 failed** (7 min 13 s) |
| The whole suite on the default Python path, same tree | **2,026 passed, 0 failed** |
| Whole-suite wall clock, same machine, Rust vs default | 432.6 s vs 397.9 s |

**One tripwire did not fire, and the reason should be on the record.** Phase 0's
`test_core_dependency_allowlist` went red the first time it met `physsynth_rs` — that was the §2.2
rule working, refusing a new compiled dependency of the core without a human editing the allowlist.
Phase 1 also makes `physsynth/core/operators.py` import `physsynth_rs` on the flagged path, and the
test stayed green, because the entry Phase 0 reviewed already covers it. That is correct, not a
gap. But it means **every later phase that imports the same extension gets the same free pass**,
and the tripwire should not be read as still armed against it: what it guards is a *new name*, and
there will not be another one until the binding is deleted. The rule that is still live for the
Rust side is `crates/physsynth-core/tests/deps.rs`, whose allowlist Phase 1 deliberately left empty
(§10.2).

**On the two wall-clock numbers: they are not a regression, and they are not evidence of
anything.** The flagged run took 432.6 s against the default path's 397.9 s, on one machine, one
run each, not back to back and not controlled. §9.4 already warned that its own 9 % figure was
inseparable from run-to-run noise on its own evidence; this is the same caveat with the sign
flipped, and it should be read the same way. The plausible reading is that the operators are not
where the bulk is — nothing in this phase touched a timestepping inner loop, and five models still
run their Python schemes around Rust-built matrices that are assembled *once* at construction.
§7's prediction that the suite gets dramatically faster is still a prediction, and the number that
tests it is Phase 2's, where the explicit models' step functions move.

### 10.6 What Phase 2 inherits

- **`Csr` is the interchange type, and it is deliberately minimal.** `from_rows`, `transpose`,
  `matmul`, `scaled`, `matvec`, `get`. Phase 2's Group A models need no more; the first solver will
  need conversion *out* of it, which is the direction that stays cheap.
- **The `_py` alias convention now covers a module of free functions, not just a class.** Nine
  aliases, one per public name, and the swap guard iterates `__all__` — so a tenth function added
  without an alias fails the guard rather than silently escaping the comparison.
- **The trait question from §8 is now due.** Phase 2 is where the `Resonator` trait gets fixed, and
  the plan says polyphony (HANDOFF §11.3a) should be settled first or the trait is rebuilt later.
  Nothing in Phase 0 or Phase 1 forced the question; the eight explicit models will.

## 11. Phase 2, batch 1, as built (2026-08-26)

`physsynth/core/exciter.py` and `physsynth/core/membrane.py` in full, plus the *builder half* of
`physsynth/core/operators2d.py`. Two things make this batch different in kind from Phases 0 and 1
rather than merely larger:

- it is the first to move a **timestepping inner loop** into Rust — §10.5 said the number that
  tests §7's speed prediction would be this one, and §11.6 is that number;
- it is the first to port a module **in halves**, which turned out not to be a special case but a
  correction to how the whole plan is ordered (§11.2.1).

### 11.1 The shape on disk

```
crates/physsynth-core/src/ops2d.rs         2-D grid, masks, the masked 5-point Laplacian, embed
crates/physsynth-core/src/membrane.rs      model #4, both domains, energy()
crates/physsynth-core/src/exciter.rs       the three excitation shapes
crates/physsynth-core/src/fmt.rs           repr() of a float, for ERROR TEXT only. §11.4
crates/physsynth-core/tests/ops2d.rs       native geometry bars (10 tests)
crates/physsynth-core/tests/membrane.rs    native membrane bars (12 tests)
crates/physsynth-core/tests/exciter.rs     native excitation bars (8 tests)
crates/physsynth-py/src/shape.rs           rank-agnostic array reads; Python's shape repr
crates/physsynth-py/src/ops2d.rs           the seven builders
crates/physsynth-py/src/membrane.rs        the class. The first place the binding calls SciPy
crates/physsynth-py/src/exciter.rs         the three shapes
physsynth/core/{membrane,operators2d,exciter}.py   + a swap block each, gated on PHYSSYNTH_RS
tests/test_rust_parity_ops2d.py            Rust vs Python, geometry + excitations (174 tests)
tests/test_rust_parity_membrane.py         Rust vs Python, the model (81 tests)
```

### 11.2 Two ordering findings — the phase list is a risk map, not a schedule

#### 11.2.1 The unit of porting is a function group, not a file

§4's risk map is keyed to files, and **a file's group is the group of its hardest function**. That
is the right way to price risk and the wrong way to schedule work.

`operators2d` is the demonstration. It sits in Group D — Phase 5 — because it contains
`VonKarmanBracket` and `AiryStressSolver`, which factor with SuperLU. But `membrane` is a Phase 2
model, and the five functions it reaches for (`grid_coords`, `rectangle_mask`, `disk_mask`,
`laplacian_from_mask`, `embed`) never solve anything; they assemble. Waiting for Phase 5 would have
held a Phase 2 model behind a solver it does not call. So the module ported in half, and the half
that is deliberately absent is named in `ops2d.rs`'s header rather than left to be discovered:
`guitar_*`, `live_cells`, `cells_per_node`, `prune_to_area_carrying`, `biharmonic_from_mask`,
`orthotropic_biharmonic`, `free_plate_stiffness*`, `VonKarmanBracket`, `AiryStressSolver`.

Expect the same split at `plate` (geometry versus the θ-scheme solve) and — most consequentially —
at `airbox`, whose 3,925 lines are six factorizations surrounded by a great deal of arithmetic that
is not. Reading §4 as a schedule would put the largest file in the project at the very end because
of six functions inside it.

**The rule this batch adds:** the phase of a *file* is the phase of its hardest function; the phase
of a *function group* is its own. Port the group.

#### 11.2.2 `mallet` needs `collision`, so Phase 2 finishes after Phase 3 starts

The eight Group A files were listed as a phase, which implies they are independent. Measured, they
are not. Intra-`core` dependencies: `exciter`, `bore`, `body` and `engine` need nothing; `membrane`
needs the `operators2d` builders; `reed` needs `bore`; `radiation` needs `body`; and **`mallet`
imports `collision`**, which is Group C — Phase 3.

So the honest statement is that Phase 2 cannot finish before Phase 3 begins. §1.2 found that a
ported model waits on its *clients*; this is the same relation seen from the supplier side, and
neither is encoded in the phase numbering. The remaining batch order that follows from it:
`bore` + `body` (nothing blocks them), then `reed` + `radiation`, then `engine` (§11.3), and
`mallet` after `collision` lands in Phase 3.

### 11.3 The `Resonator` trait, deferred within the phase — and polyphony was never the blocker

§10.6 recorded the trait question as due at Phase 2, and §8 said it was blocked on polyphony being
settled. **Checked, and it is not blocked:** HANDOFF §11.3a settled polyphony on 2026-08-10 — field
models per instance, strings per voice — with only the voice-count *budget* deferred, and that is a
real-time-stage concern that no trait signature depends on.

The trait is deferred anyway, deliberately, and by §10.2's reasoning rather than for want of an
answer. `engine` is the trait's only consumer. Fixing an interface now would mean deriving it from
the two models that happen to be ported, and those two share almost nothing: `string_ideal` holds a
1-D field on a uniform grid; `membrane` holds a live-node vector over a mask, with a second
geometry (`state`, `to_live`, `index_map`) that has no 1-D counterpart at all. A trait fitted to
that pair would be an interface chosen by a coincidence of scheduling. It arrives with `engine`,
the batch that actually has a requirement for it.

What the batch *does* fix, and what every later model should copy, is a set of conventions rather
than a type:

- the §9.1 three-way split — a plain-data `Params` that validates, free-function kernels, an owning
  struct that holds state — now shown to survive a model with geometry attached;
- `energy()` as an inherent method returning `f64`, evaluated through the same operator the update
  uses;
- the buffer contract (§9.3): rebound state in Python-owned arrays, everything immutable built once
  and handed back by reference (§11.4);
- errors as an enum whose `Display` reproduces the Python message verbatim.

### 11.4 The findings

**A sparse matvec inside a timestepping loop is still bit-identical — over 2,000 steps with
feedback.** This is the sharpest form of §10.3's result and it was the batch's open question.
`u^{n+1}` depends on `L @ u^n`, which is a reduction per row; the output is fed straight back in,
so a single 1-ulp disagreement anywhere would compound and be visible long before step 2,000. It
does not happen: the state matches `np.array_equal` at every checkpoint of a 2,000-step run, for
both domains, lossless and lossy, from a displacement start and from a velocity start. The Rust
`matvec` accumulates in ascending column index, which is what SciPy's CSR kernel does.

The line §10.3 drew — *not* "does it contain a reduction" but "**is the summation order knowable**"
— therefore holds through a loop and not merely through a build. What still cannot match is
`energy()`, whose inner products go through `np.dot` and BLAS; it is held to the Group A target.
The practical consequence for Phase 3: the first thing that will break bit-identity is a **solver**,
not a step.

**`physsynth-py` is now a SciPy client, and that is new.** Phase 1's seam handed matrices back as
`(data, indices, indptr, shape)` triplets and rebuilt them in a Python shim, so the binding never
imported SciPy. That works for a function return and not for `membrane.L`, which is a
`csr_matrix` **on the instance** — there is no call to wrap. So the constructor imports
`scipy.sparse` and builds the object itself. The portability rule is unweakened: it is about
`physsynth-core`, whose dependency list is still empty and still enforced by
`crates/physsynth-core/tests/deps.rs`. But the binding's Python-side dependencies are now real, and
the day the binding is deleted is the day this import goes with it.

**`L` had to be built once, and getting that wrong would have passed every test.**
`airbox._MembraneSurface.rhs` evaluates `m.L @ m.u` once per timestep. A `L` getter that rebuilt a
`csr_matrix` per access would assemble a sparse matrix inside the inner loop of some of the heaviest
tests in the suite — every physics bar green, and the flagged run mysteriously *slower* than the
Python one. `X`, `Y`, `mask`, `index_map` and `L` are built in the constructor and handed back by
`clone_ref`; the parity test asserts it by **identity** (`rs.L is rs.L`), because a
rebuild-per-access is invisible to any comparison of values.

**The clients do not just read the state — they write it, and one of them rebinds it.**
`mallet` does `mem.u[i] = u_free - g_s * f`: an in-place write to a single element, through the
property, expecting the model to see it. `airbox._MembraneSurface.commit` does `m.u_prev = m.u` and
then assigns a fresh array into `m.u`. Both halves of §9.3's contract are therefore load-bearing in
2-D and not hypothetical, which is why the membrane's clients are in the gate as a named claim
(§11.5) — no membrane test makes either call.

**`cos` is the first transcendental in the port.** NumPy does not call the platform libm for
`np.cos` on a float64 array; it has vectorised implementations with their own ~1 ulp budget. So the
raised cosines are the first kernels whose exactness rests on two implementations agreeing rather
than on IEEE-754 alone — the same shape as `delta_xxxx`'s reliance on two `pow`s (§10.3). Measured
on this machine they agree bit-for-bit, and that is asserted and swept over widths and grid sizes,
so a platform where it stops being true fails there, with an obvious cause, rather than as a
mysterious partial three phases later.

**A rejection message can need a float formatter, and this is the first time the "verbatim
messages" convention was not free.** Most ported messages interpolate with an explicit precision,
where Rust and Python already agree. `exciter.triangular_pluck` interpolates bare — and there
Python's `repr(1.0)` is `1.0` where Rust's `{}` is `1`, and `repr(1e-5)` is `1e-05` where Rust's is
`0.00001`. Rust's `Debug` is much closer (same shortest-round-trip algorithm, keeps the `.0`), so
`fmt.rs` starts there and fixes what remains: the exponent's sign and two-digit padding, and
`NaN`/`nan`. It is for error text only; nothing numeric goes through it, and no state crosses the
boundary as text.

**A mask is a geometry decision, not a number — and every detector this project owns is blind to
it.** `disk_mask` is a strict `x² + y² < r²`, so a node one ulp from the rim changes whether it is
an unknown at all. A membrane with one node fewer conserves energy just as beautifully as the right
one, decays just as monotonically, and lands close enough to the Bessel oracle to pass a
convergence-rate bar. So the masks are compared **elementwise**, against both parities of `N` (an
even `N` puts a node exactly at the origin; an odd one does not, which changes the staircase), and
never through anything downstream of them. This is the same lesson the guitar plate learned about
its outline, arriving from the other direction.

**One divergence is real, latent, and worth having written down.** `triangular_pluck` builds its
result with `np.empty_like(x)`, so on a float32 grid the Python version returns float32 while the
Rust one always returns float64. Checked: nothing under `physsynth/` uses float32 — every grid comes
from `np.linspace` — so no caller can see it today. It is recorded at the swap block because the
dtype is genuinely not preserved and a future caller could.

### 11.5 The success condition, and what it covers that no membrane test can

§10.4 generalised that a module's tests are not in the file named after it. The membrane confirms
it twice over: there is no `tests/test_membrane.py`, the bars live in four files split by
*criterion* (energy, modal, dispersion, stability), and the calls that decide whether the binding is
right are in neither set. The CI step is therefore a named claim in three parts:

- **the membrane's own four files**, which establish the physics;
- **its clients** — `mallet` (three files) and `airbox` (membrane, surface) — which are where `.u`,
  `.u_prev`, `.n` and `.L` get read, written and rebound once per timestep;
- **the plate family**, because `plate.py` imports `rectangle_mask`, `disk_mask`,
  `laplacian_from_mask` and `embed` from `operators2d`. Flipping the flag puts every plate —
  supported, free, orthotropic, guitar-shaped — and the von Kármán bracket on **Rust-built geometry
  while all of them are still Python models**. Same lever as Phase 1's, one dimension up.

`exciter` gets no list of its own, on purpose: nearly every model's tests call `triangular_pluck` or
a raised cosine, so a list would be the suite. The whole-suite run under the flag is its bar, and it
is in §11.6.

**The swap guard now derives what it checks.** Phase 1's version iterated `operators.__all__`, which
works only for a module ported in full. `operators2d` is not, so the guard instead reads the set of
`_py` aliases the module actually defines and compares it against a written-down expectation. A
function ported without an alias fails the comparison; an alias added without a swap fails it too;
and widening the ported set is a reviewed edit rather than a silent one — the same reasoning as the
hardcoded dependency allowlist. The guard also gained the membrane's version of Phase 1's
import-order hazard: `mallet` does `from .membrane import Membrane` at import time, so
`mallet.Membrane is membrane.Membrane` is asserted, and likewise
`membrane.laplacian_from_mask is operators2d.laplacian_from_mask`.

**One configuration the swap creates that is worth naming rather than tripping over.**
`membrane.py` imports its operators from `operators2d`, whose swap block has already run. So on the
flagged path the *Python* membrane — `MembranePy`, the name every parity check reaches for — is
stepping a *Rust*-built Laplacian. That is the lever working as designed (§1.2), and it is a third
useful configuration rather than a gap; the bit-parity claim itself is measured on the default path,
where both sides are unambiguous, which is why the parity CI step runs **without** the flag.

### 11.6 What was measured

| | |
|---|---|
| Native `cargo test --workspace` | **76 passed** (42 at the end of Phase 1) |
| `tests/test_rust_parity_ops2d.py` | **174 passed** |
| `tests/test_rust_parity_membrane.py` | **81 passed** |
| Grid, masks, `index_map` — 9 sizes, both parities of `N`, both domains | **elementwise identical** |
| Laplacian — `data`, `indices`, `indptr`, `nnz` | **bit-identical** |
| Excitation shapes, including the two that call `cos` | **bit-identical** |
| Membrane state over 2,000 steps — 2 domains × lossless/lossy × displacement/velocity start | **bit-identical** |
| `inner2d` / `norm2_2d` / `energy()`, worst relative | inside the 1e-13 Group A target |
| **The WHOLE suite under `PHYSSYNTH_RS=1`** | **2,283 passed, 0 failed** |
| The whole suite on the default Python path, same tree | **2,283 passed, 0 failed** |
| Whole-suite core-seconds, same machine, same partition, Rust vs default | 2,612.3 s vs 2,534.3 s |

**The speed prediction, finally testable — and the answer is not the one §7 expects.**
§10.5 said the number that tests §7's "the suite gets dramatically faster" would be Phase 2's,
because Phase 2 is where a timestepping inner loop first moves. It moved, and at whole-suite scale
the flagged run is **3.1 % slower**, not faster (per shard: +5.7 %, +1.1 %, +2.8 % — same sign three
times, which is more than the two single runs of §10.5 could say). One machine, one run each, so the
magnitude is still not worth much; the sign is now consistent across five measurements over two
phases and should be believed.

> **Weakened by batch 2 — see §12.7.** A matched pair taken after `body` landed reads +0.14 %
> with the per-shard signs *disagreeing*, and between the two pairs the same machine ran the
> same suite ~20 % faster. The drift is larger than the effect, so "same sign three times" was
> less evidence than it looked. The controlled per-step number below is the one that stands.

The controlled measurement explains it. Timing `step()` alone, best of seven runs, same process,
release build:

| `N` | live nodes | Python µs/step | Rust µs/step | speedup |
|---:|---:|---:|---:|---:|
| 10 | 69 | 5.17 | 0.59 | **8.7×** |
| 20 | 305 | 6.47 | 1.62 | 4.0× |
| 30 | 697 | 11.55 | 3.69 | 3.1× |
| 40 | 1,245 | 10.60 | 6.58 | 1.6× |
| 60 | 2,809 | 20.39 | 11.99 | 1.7× |
| 80 | 5,013 | 23.60 | 22.29 | 1.1× |
| 120 | 11,277 | 59.07 | 55.48 | 1.1× |
| 160 | 20,069 | 106.43 | 90.26 | 1.2× |

**What Rust removes is the per-step interpreter overhead, not the arithmetic.** (§12.7 sharpens
this: the decay to ~1x is not about the state getting large, it is about SciPy's matvec being
*already compiled*. A model with no compiled kernel under it keeps its advantage at any size.)
Python needs ~5 µs
to take one step of a 69-unknown membrane — the unknowns are not the cost; the dispatch is. Rust
needs 0.59 µs. By 5,000 unknowns the sparse matvec dominates, SciPy's matvec is already compiled C,
and the two converge to within ~10 %. The crossover is around `N = 80`.

That single fact reconciles everything else on this page:

- **The suite does not get faster because the suite's expensive tests are large-grid.** Convergence
  studies, modal oracles and the airbox run at the resolutions where the win has already decayed to
  nothing, and they spend most of their time in SciPy eigensolvers that this port has not touched.
  §7's prediction should be restated: *the suite* will not get dramatically faster by porting
  explicit models. It may still get faster at Phase 4, where the SuperLU factorizations live.
- **The win lands exactly where the project actually needs it.** Real-time playability (HANDOFF §9
  Phase 8) means small grids stepped 48,000 times a second, which is the left-hand end of that
  table — 4× to 9×, and the whole reason the language decision was taken. The migration's payoff is
  a *latency* result, not a *throughput* one, and measuring it on the test suite was always going to
  under-read it.
- **It also retires a worry.** A `L` getter that rebuilt its `csr_matrix` per access (§11.4) would
  have shown up as the flagged run being *dramatically* slower, not 3 % slower. The small
  consistent penalty is the boundary crossings themselves, which is what it should be.

### 11.7 What batch 2 inherits

- **Batch order, from §11.2.2:** `bore` + `body` next (nothing blocks either), then `reed` (needs
  `bore`) and `radiation` (needs `body`), then `engine` — and `mallet` only after `collision` lands
  in Phase 3.
- **The `Resonator` trait arrives with `engine`, not before** (§11.3). Until then, models are
  concrete types that share conventions rather than a signature.
- **`Csr` is still minimal and still dependency-free**, and now carries a `matvec` that is exercised
  2,000 times per parity case. The first conversion *out* of it is still Phase 3's to design.
- **The swap guard's expectation table is the thing to edit when a port lands.** Adding a `_py`
  alias without adding its name to `ported_expected` in `tests/test_stability.py` fails the gate,
  which is the intended cost of widening the ported surface.
- **`bore` and `reed` will be the first models with no matrix at all**, so they are also the first
  chance to check whether the §9.1 three-way split survives a model whose state is two staggered
  fields of different lengths.

## 12. Phase 2, batch 2, as built (2026-08-26)

`physsynth/core/body.py` in full — the modal body, 203 lines, the smallest resonator in the
project. It is also the one with the **longest client list of anything ported so far**, and that
asymmetry is the whole content of this batch: the transcription took no findings with it, and the
interface did.

`bore` was the other candidate and was deliberately **not** started here. See §12.5.

### 12.1 The shape on disk

```
crates/physsynth-core/src/body.rs       Params, five kernels, the native owning struct
crates/physsynth-core/tests/body.rs     native body bars (14 tests)
crates/physsynth-py/src/body.rs         the class — three settable state arrays, no SciPy
physsynth/core/body.py                  + a swap block at the bottom, gated on PHYSSYNTH_RS
tests/test_rust_parity_body.py          Rust vs Python (51 tests)
```

Nothing else changed shape. There is no matrix, no geometry, no mask and no immutable half to hand
back by reference: every array is length `M` (the mode count, typically 3–20) and every kernel is
elementwise. Measured against the membrane, this is the cheapest port in the migration so far.

### 12.2 The finding: a leading underscore is not a statement about the interface

The original keeps the modal acceleration in `_accel`, taken from the *actual* second difference of
the last step so the pressure read-out carries every force — including a bridge force applied from
outside. **Three modules assign to it**, once per timestep:

```text
radiation.RadiatedBody.step      b.q = b.q - (R*u) * corr  ;  b._accel = (b.q - 2 b.q_prev + q_nm1)/k^2
radiation.<rational air load>    b.q = b.q - p * corr      ;  b._accel = (...)
airbox.RoomLoadedBody.step       b.q = b.q - pbar * corr   ;  b._accel = (...)
```

All three follow the same idiom — snapshot `q^{n-1}`, step, apply a rank-1 correction to `q^{n+1}`,
then rewrite `_accel` from the *corrected* second difference — and all three **rebind** `q` rather
than writing into it. So under §9.3's rule the body has **three** Python-owned state buffers where
the string and the membrane had two, and the third one is spelled private.

Phase 0 recorded that `connection.py` *reads* the string's private names, and treated that as a
scheduling problem (a client that pins a model's internals delays its deletion). This is the same
discovery one step worse and with a different consequence: **a name's leading underscore says
nothing about whether the port may change it.** The rule for the rest of the migration is to derive
the buffer list from what clients actually touch, not from what the module's author marked public —
and the cheapest way to get that list is `grep` for assignment, not for reference.

A binding that got this wrong would not crash. A `_accel` setter that copied instead of adopting, or
no setter at all, leaves each wrapper's correction on the floor: the body still conserves energy,
still decays monotonically, still lands on every modal frequency, and radiates a sound that is
missing its coupling term. The parity file asserts the whole idiom, against **both**
implementations, because that is the only place it is visible.

### 12.3 Two smaller things worth keeping

**The step's `force` is only observable through the acceleration.** A port that reconstructed
`q'' = -omega^2 q - 2 sigma q'` instead of taking the second difference passes every free-response
test in the project and silently drops the bridge force. So the parity sweep drives the body as well
as releasing it, and the native bars assert that a forced step's acceleration is *not* the
free-response one.

**The CFL rejection names the `argmax`, not the first offender.** With two modes over `omega k = 2`
the original reports the larger CFL number. That is message-only — but the messages are matched on
elsewhere in the suite, and reproducing `np.argmax` rather than "the first `i` that fails" is the
kind of detail that is free to get right now and expensive to discover later.

### 12.4 The success condition

`tests/test_body.py` is nine tests. The gate for this batch is **268**, because `connection` and
`radiation` both do `from .body import ModalBody` at module scope and the flag therefore swings the
string-on-a-body bridge, the sympathetic-string bank, the plate and free-plate bridges, the von
Kármán bridge, and all three radiation tiers. `airbox` imports the name only under `TYPE_CHECKING`
and so captures nothing at runtime — deliberate, and worth not undoing.

The swap guard gained the two-importer version of Phase 1's import-order hazard: it asserts
`connection.ModalBody is body.ModalBody` and the same for `radiation`, so a swap that landed after
either import would fail loudly rather than leave the whole body/radiation leg on Python while the
run reported Rust.

### 12.5 Why `bore` is not in this batch

`bore.step(source=...)` takes a Python callable that mutates the pressure field **in place, mid-step**
— between the pressure and momentum sub-steps — and `reed.ReedBore` is the caller that exists for it.
That is not a transcription question. It is an interface decision about how the *exciter seam*
crosses the language boundary, and the two answers (call back into Python per step through PyO3, or
port `reed` in the same batch so the hook never crosses at all) commit the project for `bow`
(Phase 3) and every continuous exciter after it.

§11.3's reasoning applies unchanged: do not fix an interface before the requirement that ought to
choose it exists. Here the requirement is `reed`, and §11.2.2 already says `reed` follows `bore`. So
they go together, as one batch, next.

### 12.6 What was measured

| | |
|---|---|
| Native `cargo test --workspace` | **90 passed** (76 at the end of batch 1) |
| `tests/test_rust_parity_body.py` | **51 passed** |
| Every derived parameter — 5 configurations | **identical** |
| State over 4,000 free steps and 2,000 driven steps, `q` / `q_prev` / `_accel` | **bit-identical** |
| The rank-1 correction idiom, 500 rounds, both implementations | **bit-identical** |
| `energy` / `pressure` / `bridge_displacement` / `bridge_velocity`, worst relative | inside the 1e-13 Group A target |
| The body's clients under `PHYSSYNTH_RS=1` (10 files) | **268 passed, 0 failed** |
| **The WHOLE suite under `PHYSSYNTH_RS=1`** | **2,335 passed, 0 failed** |
| The whole suite on the default Python path, same tree | **2,335 passed, 0 failed** |
| Whole-suite core-seconds, same machine, same partition, Rust vs default | 2,082.9 s vs 2,079.9 s |

### 12.7 The speed result, sharpened: it is not about size, it is about what NumPy was already calling

§11.6 measured the membrane's step at 8.7x on a small grid decaying to ~1.1x by 5,000 unknowns, and
read that as "Rust removes the per-step interpreter overhead, not the arithmetic". The body sharpens
it, because the body has **no compiled kernel on the Python side at all** — no sparse matvec, no
BLAS call, just eight elementwise NumPy operations on short arrays. Timing `step()` alone, best of
seven, release build:

| modes | Python us/step | Rust us/step | speedup |
|---:|---:|---:|---:|
| 1 | 7.05 | 0.43 | **16.4x** |
| 3 | 6.73 | 0.43 | 15.5x |
| 8 | 7.52 | 0.45 | 16.8x |
| 20 | 7.36 | 0.46 | 15.9x |
| 64 | 7.52 | 0.54 | 13.9x |
| 256 | 8.49 | 0.89 | 9.5x |
| 1,024 | 12.43 | 2.17 | 5.7x |
| 4,096 | 26.16 | 7.32 | 3.6x |

A real modal body has between one and a few dozen modes, so the operating point is the flat top of
that table: **~15x, and Python spends ~7 microseconds per step whether the bank has one mode or
sixty-four.** That is dispatch, not arithmetic — eight NumPy calls at roughly a microsecond each.

The refinement to §11.6: the membrane's speedup decayed to ~1x not because its grid got large but
because at large `N` its hot operation was `scipy.sparse`'s **compiled** matvec, and Rust cannot beat
compiled C at the same algorithm. The body never reaches such an operation, so it keeps a 3.6x edge
even at an absurd 4,096 modes. **The question is not how big the state is, it is whether the Python
side's hot path was already in C.** Applied forward: the models that will gain most are the ones
whose steps are many short NumPy expressions — the exciters, the lumped models, the boundary
corrections — and the ones that will gain least are the ones already dominated by a SciPy solve.

**And a caution about the whole-suite numbers, stronger than §11.6's.** Batch 1's matched pair read
+3.1% (flagged slower) with the same sign in all three shards. Batch 2's matched pair reads **+0.14%**,
and the per-shard signs *disagree* (-0.6%, -1.8%, +2.9%). Between the two pairs the same machine ran
the same suite about 20% faster overall (2,612 s -> 2,083 s flagged). So the machine's throughput
drifts by far more than the effect being measured, cross-pair comparison is worthless, and batch 1's
"same sign three times" should be read as weaker evidence than it looked at the time. **The only
speed claim this migration can support is the controlled per-step one**; the suite-level number is
useful for saying "nothing blew up", and for nothing else.

### 12.8 What the `bore` + `reed` batch inherits — the pre-work, done before any Rust was written

§12.2's rule ("derive the buffer list from what clients assign to, not from what the author marked
public") was applied to `bore` and `reed` before designing their binding. Four results, and three of
them change the design:

**The Python-callable seam is load-bearing, not transitional.** `reed.ReedBore._inject` is not the
only caller that passes `source=`: `tests/test_reed_stability.py` passes its own `lambda p: None` to
assert the hook is inert when unused. So the binding must accept an arbitrary Python callable
whatever happens to `reed` — porting `reed` in the same batch removes the *hot* crossing, not the
capability. The design that follows: `physsynth-core`'s `Bore` takes a **Rust closure**
(`Option<&mut dyn FnMut(&mut [f64])>`), and the binding wraps a Python callable into one. A PyO3
type inside `physsynth-core` would break exactly what `crates/physsynth-core/tests/deps.rs` exists
to guard, and it is the one mistake here that is expensive to undo.

**`Lop`, `Cmat` and `dof` are public attributes, and they are the membrane's `L` problem again.**
`_build_pressure_operator` is private and called once from `__init__`, but what it *returns* is
assigned to three public names, and `tests/helpers.py`, `tests/test_bore_energy.py`,
`tests/test_bore_modal.py` and `web/serialize.py` all reach for them — with fancy indexing
(`bore.Lop[dof][:, dof]`) and as arguments to a generalized `eigsh`. So they must be real
`scipy.sparse` objects on the instance, built once, which means the binding builds them itself
(§11.4's "`physsynth-py` is a SciPy client"). Phase 1's triplet-shim does not apply: there is no
call to wrap.

**The step's internal ordering is load-bearing and no energy test can see it.** The hook fires
*after* the open-end pin, *before* the radiating-end drain, *before* the momentum sub-step — so that
`U^{n+3/2}` sees the corrected node pressure. Get it wrong and the reed still oscillates and the
books still roughly balance; the project's own reed work already established that balance is not a
sufficient detector there and the **signature** oracle is. `tests/test_reed_signature.py` therefore
belongs in this batch's named CI step, not just `test_reed_energy.py`.

**And one finding that is not about Rust at all.** `reed.py` computed its node-0 half-cell
compliance from the bore's *public* geometry — deliberately, to avoid reaching into private arrays —
and carried a comment claiming the result equals the bore's own `_p_pref[0]`. It does not. The bore
spells the compliance `rho0 * c0**2` (one libm `pow`); the reed spells it `rho0 * c0 * c0` (two
multiplies). **Measured, they disagree by one ulp in 3,531 of 3,552 tube/grid combinations**, worst
4.1e-16 relative. This is §10.3's `h ** 4` finding in Python-vs-Python form, it predates the
migration by a long way, and the physical consequence is nil — it scales an injection that is itself
a correction. What matters is that a port which "tidied" the two spellings into agreement would be
changing a number the acceptance runs were taken with. The comment is corrected in place rather than
the code.

### 12.9 A limit of the gate, stated so nobody assumes otherwise

CI's Rust coverage is **exactly the four named lists** — the ideal string's tests, the operators'
clients, the membrane's clients plus the 2-D builders, and the body's clients — plus the four parity
files, which run *without* the flag. The three sharded `validate` jobs that cover the whole suite run
on the **default Python path**; `PHYSSYNTH_RS=1` is never set there.

That is the right shape while the named lists are the claim: a partition would hide which models the
flag is actually asserted over, and §10.4's lesson is that the interesting tests are never in the
file named after the module. But it means the **whole-suite flagged run is a manual measurement**,
taken by hand at the end of each batch and recorded in §11.6 and §12.6 — not a gate. If it should
become one, that is a deliberate decision to price (a fourth shard axis, roughly doubling the gate),
not an omission to fix quietly.

---

## 13. Phase 2, batch 3, as built (2026-08-26)

`physsynth/core/bore.py` and `physsynth/core/reed.py` — the whole wind leg, 890 lines, ported as
**one batch** because §12.5 held them together deliberately: the question this batch existed to
answer was not how to transcribe either model but **how the exciter seam crosses the language
boundary**, and that question needs both sides of the seam present to answer.

### 13.1 The shape on disk

```
crates/physsynth-core/src/bore.rs      Params, the staggered p/U kernels, the rank-1 drain
crates/physsynth-core/src/reed.rs      Params, State, the scalar solve, inject/commit
crates/physsynth-core/src/root.rs      Brent, transcribed from SciPy — see §13.3
crates/physsynth-core/tests/bore.rs    native bore bars (24 tests)
crates/physsynth-core/tests/reed.rs    native reed bars (19 tests)
crates/physsynth-core/tests/root.rs    native Brent bars (8 tests)
crates/physsynth-py/src/bore.rs        the class; three state arrays, Lop/Cmat/dof via SciPy
crates/physsynth-py/src/reed.rs        the class; holds the caller's PyBore, injects natively
physsynth/core/{bore,reed}.py          + a swap block at the bottom, gated on PHYSSYNTH_RS
tests/test_rust_parity_bore.py         Rust vs Python (148 tests)
tests/test_rust_parity_reed.py         Rust vs Python (104 tests)
```

### 13.2 The answer to the seam question, and the finding that came with it

**The hook stays a general Python callable, and the reed stops using it.** Both halves matter:

- `PyBore.step(source=...)` accepts any Python callable and hands it a **live, writable view** of
  the `p_next` the step is about to commit. §12.8 established this is not the reed's private
  channel — `tests/test_reed_stability.py` passes its own `lambda p: None` — so the capability is
  interface, not scaffolding. `p_next` is a Python-owned `PyArray1` rather than a `Vec` with a
  temporary view over it, which is §9.3 for the third time.
- `PyReedBore` **requires a `PyBore`** and injects through `PyBore::step_native`, a Rust closure.
  So the clarinet's hot loop crosses once per `step()` rather than twice, and the scalar solve, the
  Bernoulli jet and the Brent fallback never touch the interpreter. Handed the pure-Python `BorePy`
  it raises `TypeError` rather than falling back, because a silent fallback would be a Rust reed
  reporting Rust while blowing a Python tube.

`physsynth_core::bore::Source` is `&mut dyn FnMut(&mut [f64])`. A PyO3 type inside `physsynth-core`
would break exactly what `crates/physsynth-core/tests/deps.rs` guards, and §12.8 already named that
as the one mistake here expensive to undo.

**The finding: a `&mut self` `#[pymethods]` function cannot hand control back to Python and still be
read.** PyO3 holds a `PyRefMut` on the object for the whole body of such a method. `Bore.step` calls
out mid-step, and the reed's hook reads `self.bore.p[0]` — an ordinary read the original allows.
The obvious binding refuses it with `RuntimeError: Already mutably borrowed`.

So `step` takes the **object** (`slf: &Bound<'_, Self>`) and borrows it in two short phases with the
callback in between, holding nothing. That restores the original exactly: while the hook runs, `p`
is still the uncommitted `p^n`, `U` is still `U^{n+1/2}`, and `n` has not advanced.

Three things about this are worth carrying forward rather than rediscovering:

- **It is invisible to `cargo test`**, which never crosses the boundary. The native `Bore` takes the
  same hook and is perfectly happy. Only a Python-side test can see it, which is why
  `test_the_hook_can_read_the_bore_mid_step` exists and asserts *what* the hook sees, not merely
  that the read did not raise — a binding that committed early would pass the weaker version.
- **Every model that calls back into Python mid-step inherits the shape**, so `bow` (Phase 3) and
  every continuous exciter after it should start from this pattern rather than meet the error.
- **The error path is part of the contract too.** When a Python callable raises, `step` propagates
  before committing and the bore is left un-stepped. `step_native` therefore takes a *fallible*
  hook, so the native path refuses the same way rather than committing a step the Python path
  would have abandoned.

### 13.3 The measurement that decided the reed's design: the fallback fires

The reed's scalar solve is a continuation-seeded Newton with a bracketed `scipy.optimize.brentq`
fallback for the `sqrt` cusp at `dp = 0`. Before any Rust was written, the obvious question was
whether that fallback is reachable in the configurations the suite actually builds. Measured over
4,000 steps each:

| configuration | fallbacks |
|---|---:|
| `p_mouth = 1200`, closed-open, lossless | 0 |
| `p_mouth = 1500` (the flagship) | 5 |
| `p_mouth = 1800` | 13 |
| radiating bell, `gamma ~ 0.5` | 4 |
| below threshold (`gamma ~ 0.1`) | 0 |
| **`N = 40` (coarse grid)** | **219** |

So it fires, routinely, and `physsynth-core`'s dependency list is empty by design — there is no
SciPy to call. The choice was between transcribing SciPy's ~90-line `Zeros/brentq.c` and dropping
the reed out of the bit-identical bucket. **The transcription was checked before it was relied on:**
implemented in Python first and run against the real `brentq` on the reed's own residuals over
**248 real calls**, the two returned bit-identical roots every time — not close, equal. That is what
makes `tests/test_rust_parity_reed.py` able to assert `array_equal` rather than a tolerance.

Two consequences that generalise past this model:

- **A branch choice is part of the trajectory, not a diagnostic.** If Rust stalls Newton on a
  different step than Python, the two separate *structurally* rather than by rounding, and no energy
  bar sees it. So `fallbacks` is compared **step for step** over 2,000 steps, not at the end — a
  sampled comparison would find the trajectories still identical long after the branches diverged,
  because the two roots agree to ~1e-13 and it takes a while to show.
- **The stall test is `!(|r_new| < |r|)`, not `|r_new| >= |r|`.** The original spells it
  `if not (abs(r_new) < abs(r))`, which is *true* for a NaN residual. The inverted spelling is false
  there and would iterate on a NaN forever. `clippy::neg_cmp_op_on_partial_ord` asks for the wrong
  one; the `allow` carries the reason, and `tests/reed.rs::the_stall_test_is_nan_true` pins it.

### 13.4 The other pre-measurement: `**2` is not `x * x`, but the one-ulp finding is not about `pow`

§12.8 recorded that `bore` and `reed` compute the same physical compliance with different spellings
and disagree by one ulp in 3,531 of 3,552 tube/grid combinations. The natural reading is that the
bore's `rho0 * c0**2` goes through libm's `pow` and the reed's `rho0 * c0 * c0` does not. **That
reading is wrong, and the correct one is more useful.** Measured:

- `c0 ** 2 == c0 * c0` is `True` at the ambient 343 m/s — 343² = 117649 is exact in doubles.
- The divergence is **associativity**: `rho0 * (c0*c0)` versus `(rho0*c0) * c0`.
- But `x ** 2 != x * x` in **79 of 200,007** random positive doubles, so the `pow` spelling still
  has to be reproduced in general — `powf(2.0)`, exactly as Phase 1 spelled `h ** 4` (§10.3).

Both spellings are preserved on both sides, and both parity files assert it, so a future "tidy-up"
that made the two agree fails loudly rather than quietly changing a number the acceptance runs were
taken with.

### 13.5 Three smaller things worth keeping

**A bell at both ends books each end's energy separately.** `_radiate_node` accumulates
`radiated_energy` itself, once per node, so a two-ended bell computes `(E + e_l) + e_r` and never
`E + (e_l + e_r)`. That is a claim about the order of two additions rather than about physics; the
two agree to 1e-12 and differ in the last bit. Asserted natively and in parity, with the test
*failing* if the chosen configuration cannot distinguish the orders — a vacuous version of this
test would be worse than none.

**`R_bell > 0` with neither end radiating is a legal bore**, and the original's early exit keys on
the resistance rather than on the ends — so the `U_out` read-out pair still rotates while nothing
radiates. A port that keyed the exit on the ends passes every other test in both files.

**`boundary=` unpacks any 2-sequence, and a tuple-only parse is invisible until a client uses a
list.** `(boundary, boundary) if isinstance(boundary, str) else boundary` accepts a list, a NumPy
array of two strings, anything of length two — and a **list is exactly what a JSON round-trip
produces**, which matters because `web/serialize.py` is a live client of this binding for the whole
migration (§1.1). The first draft cast to `PyTuple`, compiled, and passed every test in the repo,
because every call site today writes a literal tuple. Found by *trying* it rather than by reading
the code, and it applies to `string_ideal`'s `boundary` too.

One divergence there **cannot** be closed and is recorded instead: `Bore(boundary=None)` raises
`TypeError` in Python and yields the default clarinet in Rust, because PyO3 maps both an omitted
keyword and an explicitly-passed `None` to the same Rust `None` and no signature distinguishes
them. `None` is not a legal boundary for either — the difference is refuse versus default — and
nothing in the repo passes it, but the property belongs to *every* object-typed defaulted parameter
in this binding layer, `string_ideal`'s included. `test_rust_parity_bore.py` asserts the current
behaviour of both sides so a change becomes a failing test rather than a surprise.

### 13.6 The success condition

`tests/test_bore_*.py` and `tests/test_reed_*.py` are **97 tests**, and unlike the body's this list
is not widened by clients: nothing in `core/` imports `Bore` or `ReedBore` except `reed` itself.
The wide client is the **viewer** — `web/serialize.py` builds both for its clarinet pages, and it
is the only caller that drives `Lop`/`Cmat` through a *shift-invert* `eigsh` at `k = n_modes + 1`
rather than the `k = 1` the tests use. Its 408 tests were run under the flag by hand and pass; they
are covered by the whole-suite run rather than by the named step, which is a deliberate choice
(§12.9) and not an omission.

`tests/test_reed_signature.py` is in the CI list and that is not padding. The step's internal
ordering — open-end pin, then the hook, then the radiating drain, then momentum — is load-bearing
and **no energy test can see it**: get it wrong and the reed still oscillates and the books still
roughly balance. The project's own reed work established that balance is not a sufficient detector
there and the signature is.

`bernoulli_flow` is swapped as well as `ReedBore`, and that is not tidiness:
`tests/test_reed_stability.py` imports the function **by name** and asserts its oddness and
passivity directly. Without the swap that file — which is *in* the flagged step — would go on
asserting the Python function while the run reported Rust. Found by grepping the clients for direct
imports rather than assuming a module's tests only reach it through its class; the same habit that
found `_accel`.

The swap guard gained `bore` and `reed`, plus the two-importer hazard in its sharper form: `reed.py`
does `from .bore import Bore` at module scope, so a swap landing after it would leave the clarinet
blowing a Python air column while the run reported Rust — and every reed test would still pass,
because the reed's own physics is unchanged by which bore it drives. The guard asserts
`reed.Bore is bore.Bore` **and** actually constructs a `ReedBore` on a swapped bore, which the Rust
reed refuses if the two implementations ever came apart.

### 13.7 What was measured

| | |
|---|---|
| Native `cargo test --workspace` | **141 passed** (90 at the end of batch 2) |
| `tests/test_rust_parity_bore.py` | **148 passed** |
| `tests/test_rust_parity_reed.py` | **104 passed** |
| Every derived scalar and array — 8 bore / 10 reed configurations | **identical** |
| `Lop` / `Cmat` `nnz`, `indptr`, `indices`, `data` (canonicalised) | **identical** |
| Bore state over 2,000 steps, undriven and driven through the hook | **bit-identical** |
| Reed state over 2,000 steps, all 10 configurations, **every step** | **bit-identical** |
| `fallbacks`, step for step, coarse grid (>100 fallbacks in 3,000 steps) | **identical** |
| `energy()` (inherits the bore's `np.dot`), worst relative | inside the 1e-13 Group A target |
| The bore's and reed's own tests under `PHYSSYNTH_RS=1` | **97 passed, 0 failed** |
| `tests/test_web_backend.py` under `PHYSSYNTH_RS=1` (the viewer) | **408 passed** in 772 s |
| **The WHOLE suite under `PHYSSYNTH_RS=1`** | *(pending)* |
| The whole suite on the default Python path, same tree | *(pending)* |

One caveat on the manual whole-suite measurement, since it is not a gate (§12.9): it is taken with
`-p no:randomly`, which **fixes import order** — and import order is exactly what the
`reed.Bore is bore.Bore` hazard needs randomised to probe. The CI named step does not disable
`randomly`, so the gate still covers it; the manual run should not be read as equivalent.

### 13.8 What the next batch inherits

- **Batch order, from §11.2.2:** `radiation` (needs `body`, which landed in batch 2) and then
  `engine` — and `mallet` only after `collision` lands in Phase 3.
- **§11.7's prediction that these would be "the first models with no matrix at all" was wrong, and
  the correction is the useful part.** The bore's *step* has no matrix; the bore *object* carries
  two, `Lop` and `Cmat`, built once by the binding as real `scipy.sparse` objects because four
  files slice them and feed them to a generalized `eigsh`. So the question a new model owes is not
  "does the step multiply by a matrix" but **"does any client reach for one"** — and the second
  question is answered by grepping the clients, like every other surface question in this
  migration. A batch that read "no matrix" and skipped the build-once check would put a
  `csr_matrix` assembly inside `airbox`'s inner loop, which is the hazard `membrane`'s header
  already documents.
- **The re-entrant `step` shape is now the house pattern** for any model that calls back into
  Python mid-step, and `bow` is the next one that will.
- **`root::brentq` exists and is exercised.** `collision` (Phase 3) and `bow` both do scalar solves;
  the question for each of them is not whether a root-finder is available but whether the *original*
  called SciPy's, because that is what decides bit-identity. Measure before assuming.
- **The first thing to break bit-identity is still ahead.** Batch 1 predicted it would be a solver
  rather than a step, and this batch is the first to contain one — a scalar solve, transcribed, and
  it held. A *banded* or *sparse* solve (Phases 3-6) is a different proposition, because there the
  library's pivoting is the thing that cannot be reproduced.

  > **Wrong, and batch 4 is the correction (§14.2).** Bit-identity broke one batch later, in
  > `radiation` — a Group A model with no matrix in it — because `np.dot` fuses its multiply-add
  > and `RadiatedBody.step` **feeds that reduction back into state** instead of reading it out.
  > The right question is not "does this model solve something" but "does a reduction reach the
  > next timestep", and it should be asked of every remaining model rather than of the solver
  > groups alone.

---

## 14. Phase 2, batch 4, as built (2026-08-26)

`physsynth/core/radiation.py` — the air node in its three tiers (read-out, constant-`R` load,
rational impedance), 807 lines, ported **minus one function**. It is the batch where the
migration's bit-identity claim runs out, and where it runs out is not where §13.8 predicted.

### 14.1 The shape on disk

```
crates/physsynth-core/src/radiation.rs   AirParams/AirRadiation, rank_one, LoadParams,
                                         RationalAirLoad, both loaded bodies, py_round, c_div
crates/physsynth-core/src/fmt.rs         + py_exp, Python's `{:.3e}` spelling
crates/physsynth-core/tests/radiation.rs native bars (24 tests)
crates/physsynth-py/src/radiation.rs     the four classes; the two loaded ones hold a PyModalBody
crates/physsynth-py/src/body.rs          + four pub(crate) accessors the wrappers reach for
physsynth/core/radiation.py              + a swap block at the bottom, gated on PHYSSYNTH_RS
tests/test_rust_parity_radiation.py      Rust vs Python (73 tests)
```

### 14.2 The finding: bit-identity ends at a BLAS reduction, not at a solver

Batch 1 predicted the first divergence would be "a solver, not a step". §13.8 sharpened that to
pivoting, in Phases 3-6. **Both were wrong, and the correction is more useful than the prediction
was.** The first thing to break bit-identity is one line in a Group A model that owns no matrix:

```python
u_free = float(np.dot(b.a, b.q - q_nm1)) / (2.0 * self.k)
```

`body.pressure()` has had the *identical* reduction since batch 2 and was held to Group A without
anyone thinking twice, because it is a **read-out**: its last bit reaches an assertion and stops
there. Here the same arithmetic decides `q^{n+1}`. That is the whole difference, and it is not
visible in a file listing or a line count — the question a new model owes is not "does it solve
something" but **"does a reduction feed back into state?"**

The reason it cannot be matched is worth stating precisely, because "use the same algorithm" is
the obvious wrong answer. `np.dot` on contiguous doubles is OpenBLAS, and OpenBLAS **fuses the
multiply-add**. Measured here (numpy 2.4.6, scipy-openblas 0.3.31, `DYNAMIC_ARCH`, SkylakeX
kernel), over 1000 random pairs per length:

| terms | agrees with `fma`-per-term | agrees with multiply-then-add |
|---|---|---|
| 1 | 1000/1000 | 1000/1000 |
| 2 – 15 | **1000/1000** | 302 – 752 /1000 |
| 16 and up | 140 – 346 /1000 | 242 – 325 /1000 |

So below sixteen terms it is a single-accumulator sequential `fma` loop, and at sixteen it
vectorises. Reproducing that in Rust would mean `f64::mul_add` — which on the default x86-64
target is a libm call rather than an instruction — *and* accepting that the threshold and the
vector layout are chosen by `DYNAMIC_ARCH` at run time. A bit-identity assertion built on it
would pass on this machine and fail on a CI runner that dispatches a Haswell kernel. **It would
be a claim about a CPU, not about a port.** Rust sums plainly, left to right.

Two reductions in the same file are unaffected, and keeping the three apart is what makes the
residual attributable:

* `_G` uses `np.sum`, whose pairwise summation is **plain left-to-right for seven terms or
  fewer** (measured: 0/2000 mismatches at lengths 1-7, roughly half at 8+, where numpy switches
  to eight partial accumulators). No body in this repo has eight modes, so `_G` and `_corr` come
  out bit-identical and the state difference is `u_free`'s alone.
* `AirRadiation.process` has no reduction at all — a scalar multiply and a delay line — so tier 1
  is bit-identical over 20,000 random samples and the parity test asserts it exactly.

### 14.3 The part that should change how the next batch is read: the suite cannot see this

A fused multiply-add differs from a rounded one **only when the product rounds**. And:

* `tests/helpers.py` builds every body with `phi=1.0` and no `radiation` weight, so `a_i = 1` and
  every product `a_i * d_i` is exact;
* the parity suite's own five-mode case uses `phi = [1, -0.5, 0.25, -0.125, 0.0625]` — all powers
  of two, so the products are exact again, one step more subtly.

Measured consequence: under `PHYSSYNTH_RS=1` the loaded body's state is **bit-identical over
20,000 steps** in every configuration the test suite hands these four types. The named CI step for
this batch will therefore be green with a divergence of exactly zero — and that number is a
property of the *tests*, not a measurement of the port. Only a body built with `radiation=0.02`
(the weight the core's own single-mode rig uses, and not a power of two) makes the reduction
differ at all, and `tests/test_rust_parity_radiation.py` builds one on purpose.

**The scope of that claim was checked rather than assumed, and the check is reusable.** Every
`ModalBody(` construction site in the repo was read: the largest body anywhere is **five modes**
(`[137, 213, 330, 471, 620]`, the sympathetic-string rig), no caller overrides the `freqs=`
default of any `tests/helpers.py` builder, and every multi-mode body handed to a `RadiatedBody` or
a `ReactiveRadiatedBody` uses `phi=1.0` with no `radiation` weight. Three thresholds therefore all
sit above anything this repo builds: `np.sum`'s pairwise split (8 terms), OpenBLAS's vectorised
`ddot` (16), and — the one that would bite hardest — the point where **summation order alone**
diverges even with exact products, which is that same 16. Above sixteen modes the powers-of-two
argument stops saving anything, so **the guard to write into a future batch is `M >= 16`, not
`M >= 8`.**

Two refinements worth keeping with it:

* `_G` is safer than "seven terms or fewer" suggests. Where `m` and `sigma` are uniform across the
  bank — which is most fixtures — every term of the sum is *identical*, so pairwise order cannot
  change the answer at any `M`. The seven-term bound is what holds when the per-mode vectors vary.
* **`airbox`'s own fixture already has the shape that breaks this**, and it is the only one that
  does: `tests/helpers.py::make_room_loaded_body` defaults `radiation` to a geometric series
  `AIRBOX_PORT_RADIATION * 0.65 ** arange(M)` on a four-mode body. `RoomLoadedBody` runs the
  identical rank-1 correction with its own `np.dot`, so when Phase 6 ports it the divergence
  measured here will appear there **through the existing tests**, with no new fixture needed.

**Generalisation worth carrying into Phases 3-6:** a suite whose fixtures use 1.0 and powers of
two for its weights is systematically blind to fused-multiply-add divergence. Every later batch
that compares a reduction should include at least one fixture whose coefficients are *not*
exactly representable, or its bit-identity result means less than it looks like it means.

### 14.4 Group A is a SHORT-run target, and this is the batch that needed the qualifier

§4 words the Group A target as "state arrays to ~1e-13 relative over a **short run**; the physics
bars thereafter". Four batches never had to test that qualifier because they were exact. Measured
here on the weighted nine-mode body, worst state difference as a fraction of the run's amplitude:

| steps | constant-`R` load | rational load |
|---|---|---|
| 100 | 0 | 1.1e-19 |
| 1,000 | 1.8e-15 | 7.6e-16 |
| **2,000** | **2.4e-14** | **1.4e-14** |
| 5,000 | 2.4e-14 | 1.3e-13 |
| 10,000 | 2.4e-14 | 1.9e-13 |
| 20,000 | 2.4e-14 | **3.4e-13** |

So a fed-back reduction's error **grows with run length** where a read-out's saturates, and
1e-13 is met at 2,000 steps and exceeded by 3.4x at 20,000. The parity test asserts both lengths
at their own bars rather than picking one number and calling it the answer. Neither *physics* bar
moves at either length: each implementation conserves its own energy identity to the suite's
1e-10, which is exactly the hand-off §4 describes.

**A metric trap sits next to this and is worth naming, because it made the first measurement read
1e-7.** The loaded body decays by four orders of magnitude over the run, so an element-wise
relative difference divides a frozen absolute error (~2e-17) by a vanishing signal. Against the
run's amplitude the same data reads 2.4e-14. **Normalise a decaying trajectory by its amplitude,
never pointwise** — and the same applies to the read-outs, where peak-normalised energy and
pressure differences are 2.0e-15 and 5.4e-15 against pointwise figures three orders of magnitude
larger.

### 14.5 `piston_radiation_resistance` did not port, and that is the §11.2.1 manoeuvre again

It is the one name in the file that needs a Bessel `J1`, and `scipy.special.j1` is Cephes.
Reproducing it means transcribing some forty-five seventeen-digit rational-approximation
coefficients from a source this batch does not have open, or inventing a series/asymptotic split
and owning its accuracy analysis. Neither is a load-batch decision, and **Phase 7 already exists
for exactly this class of work** ("Bessel roots, closed-form eigenfrequencies, the spectrum
detector"). It is stateless and coupled to nothing else in the module, so the swap block rebinds
five names and leaves that one pointing at the Python function — `operators2d`'s half-a-module
port, done for a special function instead of for a solver.

The generalisable form: **the unit of porting is a function group (§11.2.1), and a special
function is its own group.** A file's Bessel call does not drag the file into Phase 7; it drags
that one function there.

### 14.6 Three things that are not arithmetic and would each pass every physics bar

1. **`int(round(x))` is round-half-to-EVEN.** Rust's `f64::round` is half-away-from-zero, so a
   delay of 2.5 samples becomes 3 instead of 2. A one-sample error in the delay line is invisible
   to energy, passivity, modal frequency and convergence order alike — the four detectors this
   project owns. `radiation::py_round` transcribes CPython's `float.__round__` (C `round`, then
   `2.0 * round(x / 2.0)` if the argument sat exactly halfway) and the parity test carries four
   discriminating cases.
2. **`np.isclose(a, b, rtol, atol=0.0)` is asymmetric** — the tolerance scales on the *second*
   argument. `ReactiveRadiatedBody` uses it to compare the load's timestep against the body's, so
   a symmetric transcription would accept and reject different pairs at the boundary.
3. **Complex division is CPython's Smith's algorithm, not `a conj(b) / |b|^2`.** The two disagree
   in the last ulp. Transcribed as `_Py_c_quot` spells it, and measured 20000/20000 agreement with
   `complex.__truediv__` over the `(R, omega, tau)` ranges this model reaches — which is what lets
   `impedance`, `impedance_discrete` and `loaded_mode` be asserted *exactly* across a 500-point
   sweep rather than to a tolerance.

### 14.7 Two smaller things worth keeping

**A pyclass has no instance `__dict__`, so an attribute WRITE that Python accepted silently now
raises.** That is the `_accel` finding (§12.2) in a new costume, and it was checked the same way —
by grepping the clients rather than reading intent off the code. Result this time: `airbox.py`,
`web/serialize.py`, `tests/` and `scripts/` read `_buf`, `u_l`, `_G`, `_corr`, `pressure_load`,
`sphere_radius`, `tau` and `R_eff` off these four types, and **write none of them**. So the
private names are on the binding's surface (as always), but `_buf` and `_idx` could stay plain
Rust `Vec`/`usize` rather than Python-owned arrays — the first time in this migration that a
buffer did *not* have to cross by reference, and the check is what established it rather than the
underscore.

**The refusal message this batch transcribes contains a bug, and it was transcribed anyway.**
`loaded_mode`'s `for/else` reports `abs(w_next - w) / w_next` after the loop body has assigned
`w = w_next`, so the "last relative step" it quotes is *always exactly zero*. Nothing matches on
the text (`web/serialize.py`, the one caller, catches the `ValueError` and censors the point), so
correcting it would have been free — and wrong. A port that improves messages stops being a port,
and the two sides stop being comparable. It is recorded here and in both implementations'
comments as a fix owed to `radiation.py`, to be applied to both sides at once.

### 14.8 The success condition

Same shape as every batch before it, with one difference stated up front: the named CI step
proves the *physics* survives the swap, and it is `test_rust_parity_radiation.py` — not the step —
that measures the port, for the reason §14.3 gives.

* `tests/test_radiation.py` (72 tests) unmodified against Rust.
* Its clients, because a `RadiatedBody` is a drop-in body: `test_bore_radiation.py`,
  `test_airbox_freefield.py`, `test_airbox_scene.py`, `test_connection.py` and
  `test_web_backend.py` — the last because `web/serialize.py` hands a `RadiatedBody` to
  `StringBodyBridge`, which puts a Rust wrapper *inside* a Python bridge and exercises the
  `__getattr__` delegation surface that no radiation test reaches.

### 14.9 What was measured

| | |
|---|---|
| Native `cargo test --workspace` | **166 passed** (141 at the end of batch 3) |
| `tests/test_rust_parity_radiation.py` | **73 passed** |
| Every construction constant, 5 air / 5 load / 3 sphere configurations | **identical** |
| `latency_samples` and `retardation_residual`, halfway cases included | **identical** |
| `AirRadiation.process`, 20,000 random samples x 5 configurations | **bit-identical** |
| `_G` and `_corr`, all four body cases (M = 1, 4, 5, 9) | **identical** |
| `impedance` / `impedance_discrete`, 500-point sweep x 5 loads | **identical** |
| `loaded_mode`, 200-point sweep | **identical** |
| Loaded state, 20,000 steps, every body the suite builds | **bit-identical** |
| Loaded state, 20,000 steps, weights not powers of two | 3.4e-13 of amplitude |
| Loaded state, 2,000 steps, same body | 1.4e-14 — inside Group A |
| Peak-normalised `energy()` / `pressure()`, 2,000 steps | 2.0e-15 / 5.4e-15 |
| `R = 0` vs a bare body, and `M_a = inf` vs the constant-`R` load, **within** each side | **bit-identical** |
| The air node's clients under `PHYSSYNTH_RS=1` (785 tests) | **785 passed** in 765 s |
| Largest `ModalBody` anywhere in the repo (every call site read) | **5 modes** — below all three thresholds |
| **The WHOLE suite under `PHYSSYNTH_RS=1`** | *(pending)* |
| The whole suite on the default Python path, same tree | *(pending)* |

### 14.10 What the next batch inherits

- **Batch order:** `engine` is the last Group A file that is not blocked, and `mallet` still waits
  on `collision` (§11.2.2). Phase 3 can start in parallel.
- **`engine` is deliberately not in this batch, and the reason is not scheduling.** `simulate` is
  the *loop*, not the step: it calls `resonator.step()`, `.energy()`, `.state` and
  `.displacement_at()` on an arbitrary Python object once per iteration. Porting it moves the loop
  into Rust and leaves four boundary crossings per iteration where §11.6's win — per-step
  overhead, not arithmetic — does not apply, so it would make things *slower* until its callees
  are Rust too. Its natural time is after Phase 3, and `SimResult` (a dataclass with a
  `list[tuple[int, ndarray]]` field and a computed property) is fiddly surface with no physics in
  it. Same shape of judgement as §12.5's on `bore`.
- **The bit-identity question is now settled, and its answer is a rule rather than a phase.** Ask
  of each remaining model: *does a reduction feed back into state?* If yes, the target is Group A
  over a short run and the physics bars thereafter, whatever group the file is in. If no, expect
  bit-identity and assert it. Group B's banded Cholesky, Group C's dense LU and Group D's SuperLU
  are still their own question — but they are no longer the *first* question.
- **A fixture whose coefficients are all powers of two proves less than it appears to.** §14.3.
  This applies immediately to Phase 3: `collision` and `bow` both run iterations whose *count* is
  the thing being compared, and an iteration count is exactly the kind of discrete quantity a
  last-ulp difference flips.
- **`fmt::py_exp` exists** alongside `py_float`, for a `{:.3e}`-style interpolation. Phase 7's
  analytic oracles are the next place a message is likely to want one.

---

## 15. Phase 3, batch 1, as built (2026-08-27)

`physsynth/core/banded.py` — a new module holding one thing, the banded Cholesky that four models
share. No model was ported. That is not a shortfall in the batch; it is the batch's finding, and
§15.2 is why.

### 15.1 The shape on disk

```
crates/physsynth-core/src/banded.rs      DPBTF2 and DPBTRS, transcribed; BandedError
crates/physsynth-core/tests/banded.rs    native bars (8 tests)
crates/physsynth-py/src/banded.rs        the two functions + the NotPositiveDefinite exception
physsynth/core/banded.py                 NEW — reference is SciPy's; swap block gated on the flag
physsynth/core/string_stiff.py           import line only
physsynth/core/string_damped.py          import line only
physsynth/core/string_nonlinear.py       import line only
physsynth/core/string_geometric.py       import line only
tests/test_stability.py                  the swap guard, + the four models' captured bindings
tests/test_rust_parity_banded.py         Rust vs LAPACK (49 tests)
```

### 15.2 The finding: a bit-identity anchor is a porting constraint, and it binds models together

§5 says Phase 3 starts with `string_stiff` — "259 lines, the smallest banded solve". It does not,
and the reason is not in the file. It is in `tests/`:

```
test_damped_string.py::test_sigma1_zero_reduces_to_stiff_string_bit_for_bit
    StiffString                    == DampedStiffString(sigma1=0)   array_equal, 1500 steps
test_tension_string.py::test_EA_zero_is_model3_bit_identical
    TensionModulatedString(EA=0)   == DampedStiffString             array_equal, 400 steps
test_geometric_energy.py::test_EA_equals_T_is_bit_identical_to_damped_string
    GeometricString(EA=T)          == DampedStiffString             array_equal, 300 steps
```

These are *reduction anchors*: each says a richer model collapses onto a simpler one when its extra
physics is switched off, and each asserts it with `array_equal` rather than a tolerance, on purpose
— the linear path is the same expressions in the same order, and float addition is not associative,
so bit-identity is a **provable** claim about the code and a tolerance would be a weaker one. They
are among the most valuable tests in the repo.

They are also a constraint on the migration that nothing in §§1–14 anticipated. **Port one of those
four models and an intra-Python anchor becomes a cross-language one**, which §15.3 says the banded
solve cannot carry. So `{string_stiff, string_damped, string_nonlinear, string_geometric}` is
**indivisible** under a change of solver — and `string_geometric` is Group B *and* Group D, i.e.
Phase 5. Taken literally, "port the smallest banded model first" drags a sparse-LU model into
Phase 3 and forfeits §4.1's plan to test the SuperLU hypothesis on `beam` first.

The way out is the manoeuvre Phase 1 already made and §11.2.1 already generalised, applied one level
down: **the unit of porting is a function group, and here the function group is the solver, not the
model.** `operators` was ported before any model that uses it, and flipping it swung five models at
once. `banded` is the same shape: not a model, but what four models are built out of. Every anchor
stays valid because all four models call the same code, whatever that code is.

**The generalisable question, and it is cheap to ask:** before porting a model, grep the suite for
`array_equal` against a *different* class. A same-class comparison (a bowed string against a bare
one) survives any port, because both sides move together. A cross-class one is a chain, and the
chain — not the file — is the unit of work. Four of this repo's `array_equal` anchors are
cross-class; the rest are not, which is why fourteen batches went by without meeting one.

### 15.3 What is transcribable here, and what is not

SciPy dispatches `cholesky_banded` to `dpbtrf` and `cho_solve_banded` to `dpbtrs`. At `kd = 2` the
blocked path in `dpbtrf` is never taken (`NB > KD`), so the factor is the unblocked `DPBTF2`, and
`dpbtrs` is two `DTBSV` calls. Measured on 120 of this family's own matrices (2026-08-27):

| transcription of DPBTF2 | agrees with OpenBLAS |
|---|---|
| reciprocal-once **and** fused multiply-add | **120/120** |
| reciprocal-once, plain multiply-add | 82/120 |
| divide-per-element, either way | 19/120 |

So the **factor is reproducible**, and two details decide it. `DSCAL` forms `1/ajj` **once** and
multiplies, which is worth 19/120 → 82/120 and is the reference algorithm's own behaviour — that is
transcribed. `DSYR` fuses its multiply-add, which is worth 82/120 → 120/120 and is a property of the
kernel `DYNAMIC_ARCH` picks at run time — that is **not** transcribed, for exactly the reason §14.2
gives. A bit-identity assertion resting on it would pass here and fail on a runner that dispatched a
different kernel: a claim about a CPU, not about a port.

The **solve is not reproducible at all**, and that was established rather than assumed. Taking one
15-element system, seeding each element from LAPACK's own exact predecessors, and asking which of
{forward, reverse} × {plain, fused} × {divide, reciprocal} could produce it:

* element 2 admitted only the dividing forms;
* element 5 admitted only the fused forms;
* element 9 admitted only the forward loop order;
* element 14 excluded the forward order;
* **element 7 admitted nothing at all.**

The intersection is empty, and one element is outside the space entirely. No scalar recurrence
produces OpenBLAS's `DTBSV`; it is blocked and vectorised. The Rust side therefore transcribes the
reference `DTBSV` plainly and stops chasing.

Worth naming, because it is the *shape* of this finding rather than its detail: §14.2 said the first
thing to break bit-identity would be a reduction that feeds back into state, and predicted solvers
would be a later and separate question. Both halves were right, and the two collapsed into one batch
— a banded back-substitution **is** a chain of reductions, and its output is `u^{n+1}`.

### 15.4 The first swap that changes the numbers, and where Group A actually lands

Every batch through §14 could open with "bit-identical, except here". This one cannot, and the
consequence is a correction to how §4's Group A target should be read.

Measured on the stiff string (`N = 128`, `kappa = 2.7`, `sigma = 3`), worst state difference so far
as a fraction of the run's amplitude:

| steps | 100 | 500 | 1,000 | 2,000 | 5,000 | 20,000 |
|---|---|---|---|---|---|---|
| lossless | 8.7e-14 | 2.7e-13 | 3.4e-13 | 8.0e-13 | 1.1e-12 | 2.5e-12 |
| `sigma = 3` | 1.1e-13 | 2.9e-13 | 4.1e-13 | 9.7e-13 | 2.0e-12 | 3.2e-12 |

§14.4 established that Group A's "~1e-13 over a short run" is a **2,000-step** claim for a fed-back
reduction. For a fed-back **solve** it is a **hundred-step** claim — an order of magnitude shorter
at the same tolerance. The growth is roughly square-root, not linear and not saturating, which is
what a random-walk accumulation of last-bit differences through a well-conditioned solve looks
like.

**And the physics does not move.** Lossless energy drift, same string, 4,000 steps: LAPACK 1.14e-12,
Rust 1.16e-12, against the project's 1e-10 bar. The transcription is not worse than OpenBLAS, it is
*different* from it — which is exactly the hand-off §4 describes and the first time this migration
has had to lean on it.

**The property that replaces bit-identity as this batch's sharp claim** is the one §15.2 is about:
the four models still agree with **each other** to the bit. That is asserted three ways in
`tests/test_rust_parity_banded.py` and again, through the suite's own anchors, under the flag.

### 15.5 The shim's own validation cost more than the port saved

`cholesky_banded(..., check_finite=True)` is the default, and the first version of the Python shim
honoured it the obvious way — `np.isfinite(ab).all()` before handing the array over. Measured:

| | one solve, `n = 31` | one factor, `n = 127` | 4,000 steps, `N = 64` |
|---|---|---|---|
| SciPy | 7.7 µs | 11.6 µs | 0.072 s |
| Rust, check in the shim | — | — | 0.077 s (**0.96x**) |
| Rust, check inside the binding | 2.6 µs | 2.8 µs | 0.049 s (**1.47x**) |

The primitive is 2.2–2.9x faster on the solve and ~4x on the factor. A single extra pass over a
`(3, n)` array in Python **erased all of it and then some**, because the win being spent is
per-call overhead (§11.6) and an `np.isfinite` is another call of exactly that kind. Moving the
check into the pass the binding already makes over the input restored it, and it belongs there
anyway: it has to happen *before* the factorization, or a NaN diagonal comes back as
`NotPositiveDefinite` — the right refusal for the wrong reason and the wrong exception type.

**Generalisation for every remaining swap block:** the shim is on the hot path. Coercion and
validation written there are paid per call, in the interpreter, against a saving that is measured in
microseconds. Push them across the boundary, or measure what they cost.

Model-level speedups: 1.47x at `N = 64`, 1.35x at 256, 1.17x at 1024 — falling with size exactly as
§12.7 predicts, since what is being displaced is a compiled SciPy call and what remains is NumPy
arithmetic that is still Python-driven.

### 15.6 Four smaller things worth keeping

1. **`LinAlgError` is not a `ValueError` subclass.** Reporting a non-SPD band as a `ValueError`
   would silently change what a caller can catch. The binding raises its own
   `physsynth_rs.NotPositiveDefinite` and the shim re-raises it as the `LinAlgError` the original
   promises, with LAPACK's message text unchanged. Nothing in the repo catches it today — checked,
   not assumed — but "no client catches it yet" is a fact about the clients, not a licence.
2. **`b / sqrt(a) / sqrt(a)` is not `b / a`.** A diagonal band solved through its Cholesky factor
   divides twice, and at `a = 2` the two spellings differ in the last bit (0.49999999999999994
   against 0.5). The native test asserts the form the algorithm *computes*, so that a later "this is
   just a division" simplification fails rather than quietly moving every model.
3. **`kappa = 0` hands over a pentadiagonal band whose second superdiagonal is all zeros.** It is a
   numerically empty band, not a structurally absent one, so a `kd = 2` loop with wrong bounds does
   arithmetic that changes nothing and passes. The native bar is that the `kd = 2` and `kd = 1`
   paths agree **to the bit** on that input.
4. **The captured-binding hazard is at its widest here.** Four modules do
   `from .banded import ...` at module scope. Everywhere else in this migration a mis-ordered swap
   would leave one model on Python; here it would leave *three of four* on Rust and break the
   anchors, which fail with a message about physics. `tests/test_stability.py` now asserts all four
   captured bindings are the ones `banded` currently exposes.

### 15.7 The success condition

* `tests/test_stiff_string.py`, `test_damped_string.py`, `test_tension_string.py` and the six
  `test_geometric_*.py` files unmodified against Rust — the four models themselves, including all
  three anchors.
* Their clients, because `DampedStiffString` is the most-reused resonator in the project:
  `test_bow_*.py`, `test_collision_*.py`, `test_jawari.py`, `test_connection.py`,
  `test_sympathetic.py`.
* `test_beam_modal.py` and `test_beam_stability.py` as the **control**: `beam` is built on the same
  operators but calls no banded solver, so a failure there is about Phase 1, not about this batch.

### 15.8 What was measured

| | |
|---|---|
| Native `cargo test --workspace` | **174 passed** (166 at the end of batch 4) |
| Cargo dependency allowlist | still **EMPTY** — no LAPACK link, no sparse crate |
| `tests/test_rust_parity_banded.py` | **49 passed**, 1 skipped off-flag |
| DPBTF2 transcription vs OpenBLAS, 120 configs | 120/120 with fma, **82/120 without** (shipped) |
| DTBSV: scalar recipes consistent with OpenBLAS | **none** — one element admits no candidate |
| Factor, worst relative difference from LAPACK | **6.2e-16** |
| Solve, worst relative difference from LAPACK, 6 sizes × 3 stiffnesses | inside 1e-13 |
| State, 100 / 2,000 / 20,000 steps, as a fraction of amplitude | 1.1e-13 / 9.7e-13 / 3.2e-12 |
| Lossless energy drift, 4,000 steps | LAPACK 1.14e-12, Rust 1.16e-12 (bar 1e-10) |
| The four models' reduction anchors on the Rust solver | **bit-identical** |
| One banded solve, `n` = 31 / 127 / 511 / 2047 | **2.9x / 2.2x / 1.5x / 1.1x** faster |
| One factorization, `n` = 31 / 127 / 511 | **3.9x / 4.2x / 4.2x** faster |
| A whole `StiffString` step, `N` = 64 / 256 / 1024 | **1.47x / 1.35x / 1.17x** faster |
| The same, with `np.isfinite` in the shim | **0.96x** — slower than SciPy |
| The four models and their clients under `PHYSSYNTH_RS=1` | *(pending)* |

### 15.9 What the next batch inherits

- **`collision` is next, and it is now unblocked in a way it was not before.** Group C is one file
  and one dense LU, and it wraps a `DampedStiffString` rather than being one — so it does not join
  the chain §15.2 found. Its own reduction anchor (`test_collision_energy.py:135`) compares a
  barriered string to a bare one, both `DampedStiffString`, so it survives any port of either.
- **`bow` after it**, for the reason §4 already gives, with one addition from this batch: `bow`
  calls `string.apply_Ainv`, which is now a Rust solve, so its precomputed driving-point
  admittance already differs from the numbers its acceptance run produced. §14.10's warning about
  iteration counts is therefore live rather than prospective — and worth stating precisely,
  because the suite is weaker here than it looks: `test_collision_energy.py` and `test_jawari.py`
  assert `newton_iters < newton_maxiter`, i.e. that the contact solve *converged*, **not** that it
  took the same number of iterations as before. A last-bit difference that costs one extra Newton
  step is invisible to every test in the repo. That is survivable — the count is not a physical
  quantity — but a batch that ports `collision` itself and wants to compare trajectories will need
  to compare the counts explicitly, the way §13.3 compared the reed's branch choices.
- **The four models themselves still have to port**, and when they do the anchors are no longer the
  obstacle: with the solver already common, a Rust `StiffString` and a Python `DampedStiffString`
  do the same arithmetic everywhere it matters (the sparse matvec was proved bit-identical in
  batch 1, the elementwise arithmetic in Phase 0). Whether that survives contact is the next
  batch's measurement, not this one's claim.
- **Ask the `array_equal` question before every remaining model.** §15.2. It costs one grep and it
  is the difference between a batch and a phase.
- **Never write validation into a swap block without pricing it.** §15.5.
- **§4.1's SuperLU hypothesis is still untested.** This batch deliberately links nothing: the
  allowlist is still empty, and `beam` keeps its de-risking job at Phase 4.

---

## 16. Phase 3, batch 2, as built (2026-08-27)

`physsynth/core/collision.py` — the contact primitives and both contact solves, plus `dense`, the
project's only dense LU. `BarrierString` does not port: it wraps a `DampedStiffString`, which has
not, so porting it would mean building a model on a model that has not moved. The port swings two
models under the flag all the same, because `mallet.py` re-exports these primitives and calls the
scalar solve — which is §11.2.2's ordering finding coming due. **Phase 2 is now unblocked**, not
finished: `mallet` itself is the next batch.

### 16.1 The shape on disk

```
crates/physsynth-core/src/dense.rs        dgetrf/dgetrs, transcribed unblocked; DenseError, Lu
crates/physsynth-core/src/collision.rs    the primitives, both solves, PowPath
crates/physsynth-core/tests/dense.rs      native bars (6 tests)
crates/physsynth-core/tests/collision.rs  native bars (9 tests)
crates/physsynth-py/src/collision.rs      the primitives, both solves, the LU, rank-dispatched
physsynth/core/collision.py               swap block gated on the flag; `import os`
tests/test_rust_parity_collision.py       Rust vs NumPy/SciPy (76 tests)
.github/workflows/ci.yml                  the batch's flagged step, + a fix (§16.8)
```

### 16.2 The finding: a NumPy array and a NumPy scalar do not compute the same power

This is the thing that had to be discovered before anything would match, and it is not about Rust.

`contact_potential`, `contact_force_elastic` and `contact_stiffness` are written once in
`collision.py` and called two ways: with a float, by the mallet's scalar solve, and with an array,
by the barrier's vector solve. The module's docstrings treat the two as the same function used
twice. **They are not the same computation.**

NumPy's float64 `power` **ufunc loop** carries a fast-path ladder for the exponents `-1, 0, 0.5, 1,
2` — it spells `x ** 0.5` as `sqrt` and `x ** 2` as `x * x`. Its **scalar** path takes no such
shortcut and calls the C library's `pow`. Measured over 200,000 realistic penetrations:

| exponent | array path vs `pow` |
|---|---|
| `0.5` | **94** disagree |
| `2.0` | **53** disagree |
| `1.0`, `1.5`, `2.5`, `3.0`, `-0.5` | 0 |

Always by one ulp, always with the shortcut being the more accurate of the two (`sqrt` is
correctly rounded; `pow` is not).

The exponents this module uses are `α+1`, `α` and `α−1`. So the split lands **exactly on the two
configurations the project uses most**: `α = 1`, the closed-form-oracle case, where `α+1 = 2`; and
`α = 1.5`, the barrier default, where `α−1 = 0.5`. There is no realistic configuration where the
distinction is academic.

Two consequences.

**First, the Rust side carries both spellings.** `PowPath::Array` reproduces the ladder;
`PowPath::Scalar` calls `powf`. The binding picks between them by the *rank of the argument*, which
is the same thing as picking by which Python code path it is standing in for. That dispatch is not
a convenience — it is the port's arithmetic contract.

**Second, an existing docstring in `collision.py` is wrong, and it predates this port.**
`_force_total_vec` says it is "numerically identical to calling `contact_force_total` per
component". Measured over 200,000 pairs at `α = 1`, the two disagree in **174** of them, by up to
`1.5e-12` relative; the derivative pair disagrees in 86 of 100,000, by up to `5.6e-12`. Repairing
that would move both models' trajectories — it is a physics change, not a cleanup — so the port
reproduces the inconsistency and this note records it. Whoever retires the Python side inherits the
decision.

**The generalisation worth carrying forward:** when a Python function is vectorized, "the same
function" can mean two different computations, and which one a caller gets is decided by the rank
of what it passes. Every remaining port should ask, of every primitive with more than one call
shape: *does NumPy special-case anything this reaches?*

### 16.3 What is bit-identical, and the cause-separator that made the answer checkable

The batch was designed around a question it could answer before it could answer anything else: the
vector solve's `G @ F(η)` is a dense BLAS matvec **feeding back into the next Newton iterate**,
which is §14.2's construction exactly, so bit-identity was known to be off the table for the
barrier. That makes any divergence ambiguous — BLAS, or a transcription bug?

The separator is the **single contact node**. With one finite barrier node (`-inf` elsewhere) the
admittance block `G` is `1 × 1`, the matvec is one multiply, and the LU is a scalar divide with a
degenerate pivot. Nothing in that case *can* round differently, so it must be bit-identical, and
everything else in the solve — the primitives, the Newton path, the Armijo logic, the force
injection — is shared with the general case.

Measured, with the string's own banded solve deliberately left on SciPy so §15's change does not
confound the reading:

| | 2,000 steps |
|---|---|
| point fret, `m = 1` | **bit-identical** |
| two frets, `m = 2` | **bit-identical** |
| flat rail, `m = 79`, default `K` | **bit-identical** |
| the whole mallet trajectory, 5 fixtures | **bit-identical**, fallback counts included |
| `MalletWall`, every observable | **bit-identical** |

The scalar half of this batch is therefore in the same bucket as Phases 0–2: it contains no
reduction anywhere, and it matches. That includes the `brentq` bracket fallback, which is not
hypothetical — measured over the mallet's own fixtures it fires once per 3,000 steps in the
flagship configuration and eight times at `α = 1`, so §13.3's transcription of `brentq` is load
bearing here for the second time, and `np.linspace`'s exact spelling (form the step once, then
**overwrite the last element with `stop`**) had to be transcribed with it.

One caveat that belongs next to the mallet's result rather than buried: under the *real*
`PHYSSYNTH_RS=1` flag the mallet's `energy()` does differ, at `2e-15`. That is not the contact
scheme — it is `Membrane.energy()`, a reduction that is a read-out and never reaches the state.
`MalletWall`, which owns no field, is bit-identical in every observable including its energy, which
is what makes the attribution rather than assumes it.

### 16.4 The blind spot: a soft contact hides the solver, and the default fixture is in it

The flat rail with 79 nodes in contact came out bit-identical, which is *not* what a batch that
just introduced a new dense LU should have expected. The explanation is the finding:

**The divergence tracks how far the Newton Jacobian `J = I + G·diag(F')` is from the identity — not
the number of contact nodes, and not `α`.**

(cond(J) here is taken at the run's deepest penetration with `η⁺ = η⁻`, which is a shape
measurement rather than the exact matrix a given Newton step factors — the claim is an order
of magnitude, not a digit.)

| fixture | max cond(J) | active nodes | 2,000 steps |
|---|---|---|---|
| `α = 1.5`, `K = 1e6` (the default) | 1.004 | 59 | **bit-identical** |
| `α = 2.3`, `K = 1e6` | 1.000 | 59 | **bit-identical** |
| `α = 1`, `K = 1e4` | 1.001 | 59 | **bit-identical** |
| `α = 1`, `K = 1e6` | 1.063 | 39 | 1.5e-13 |
| `α = 1.5`, `K = 1e8` | 1.144 | 39 | 1.4e-13 |
| `α = 1.5`, `K = 1e10` | 7.082 | 26 | 8.7e-14 |

When `J` is within a percent of the identity the LU is effectively solving `I·δ = -r`, so its
disagreement with LAPACK — which is real, 5,088 of 6,241 factor entries differ at `m = 79` — never
reaches the answer. A bit-identical reading there is not evidence that the solvers match. **It is
evidence that the solver was barely used.**

This is §14.3's blindness finding one level up. There, the suite could not see a class of
divergence because every fixture weight was `1.0`. Here, the fixture the suite uses most sits in a
regime where the solver under test is nearly a no-op. Both have the same shape and the same
remedy: *the parity test has to bring a fixture chosen to exercise the thing being ported, because
the physics fixtures were chosen to exercise the physics.* `tests/test_rust_parity_collision.py`
carries a stiff case for exactly this reason, and one test asserts **both halves**: the soft case
identical, *and* the two dense solves disagreeing on the soft run's own Newton Jacobian — so the
bit-identical reading above is pinned as "the solver was not exercised" rather than left ambiguous
with "the solvers agree".

**How that test had to be rewritten is itself worth keeping.** Its first version asserted that the
*stiff fixture's trajectory* separated by a non-zero amount, which reads naturally and is wrong as
a test: it passed with the flag unset and failed with it set, because `PHYSSYNTH_RS=1` moves the
string's banded solve, which changes the admittance block, which changes whether the LU's last bits
survive into the field. **An assertion that something differs is a measurement, and a measurement
of a chaotic system is not a contract.** The rewritten version puts a real Jacobian in front of both
solvers directly, which is deterministic and says the same thing. Every parity test in this
migration should be green with the flag both set and unset. That had never been *checked* before;
it was checked here for all nine earlier files as well, and all nine pass (banded reports 50 with
the flag against 49 and a skip without, which is that file's own documented behaviour). So this is
a convention now made explicit rather than a bug found — but it was found by a bug, in the one
file where the ambient flag actually changed the answer.

### 16.5 Group A's window is shorter here, and it closes for a dynamical reason

§14.4 established that Group A is a run-length claim; §15.4 shortened it to ~100 steps for a
fed-back *solve*. This batch shortens the framing again, and changes its kind.

Measured — the step at which each fixture first exceeds `1e-13` of amplitude:

| fixture | first step over 1e-13 |
|---|---|
| point fret, two frets, lossy rail | none within 6,000 |
| flat rail, default `K` | none within 6,000 (8.2e-14 at 6,000) |
| flat rail, `α = 1` | **1,175** |
| flat rail, stiff `K = 1e8` | **1,584** |

And past that the growth is not linear. On the stiff lossless fixture: `1.2e-13` at 5,000 steps,
`3.4e-12` at 10,000, **`1.1e-7` at 20,000**. Every earlier batch's divergence grew roughly like the
run length. This one grows like an exponent, because **a string buzzing against a one-sided barrier
is chaotic** — the two trajectories do not drift apart, they separate.

So the honest statement of this batch's agreement has three parts rather than one: bit-identical
where no reduction is exercised; Group A out to about a thousand steps where one is; and past that
**the physics bars, which do not move at all** — lossless drift `1.12e-12` on SciPy against
`1.14e-12` on Rust, against a bar of `1e-10`.

The generalisation: **for a nonlinear model, the agreement window is a property of the model's
dynamics, not of the port.** A chaotic model has a short one no matter how good the transcription
is, and asking for a longer one is asking the wrong question. That is a new thing to check before
every remaining model — `string_nonlinear`, `string_geometric`, the von Kármán plate and the bow
are all nonlinear, and the question to ask of each is not "how well does it agree" but "how long
before it cannot".

### 16.6 The branch hazard that did not fire, measured rather than assumed

The sharpest risk identified before writing any code was **not** the LU. It was the Armijo test:

```python
if 0.5 * float(r_try @ r_try) < (1.0 - 1e-4 * t) * f0:
```

Both sides are reductions, and they sit inside a **branch condition**. A last-bit difference there
does not perturb the answer by an ulp — it flips the acceptance, halves the step, and changes the
iterate by `O(1)`. That is §13.3's "a branch choice is part of the trajectory" with a reduction
behind it.

It never fired. `newton_iters` came out **identical at every step of every fixture, out to 20,000
steps** — 40,000 solves — and the mallet's `fallbacks` counter likewise. §15.9 pointed out that
nothing in the repo compares iteration counts, only that they stayed under the cap; this batch's
parity test compares them, so the hazard is now watched rather than merely survived.

Two details of the original were reproduced deliberately on the way, either of which a "sensible"
rewrite would have changed:

* **The line search has no failure exit.** If all 40 backtracks are rejected the loop ends with
  `t = 2⁻⁴⁰` and the step is taken anyway, unguarded. That is the reference behaviour.
* **`np.max(np.abs(r))` propagates NaN, and `f64::max` discards it.** A fold written the obvious
  Rust way would report a converged solve on a diverged state — silently, because the one thing
  that comparison feeds is the convergence test that decides whether to warn.

### 16.7 The dense LU: what was chased, and what deliberately was not

§15.3 spent a batch establishing that OpenBLAS's `DTBSV` admits no scalar recipe. The same question
could have been asked of `dgetrf` and was not, because the answer no longer decides anything: the
matvec upstream of it is already irreproducible, so a perfectly reproduced factorization buys
nothing. `dense.rs` is written the plain way — right-looking, summed left to right — and the
agreement is a tolerance from the start.

One thing it *does* keep from LAPACK, and holds to equality in the parity test: **the pivot
sequence**. A pivot is a discrete decision, not a rounding. Choosing a different row is a different
elimination, and it would separate the trajectories by far more than the arithmetic does. Measured:
pivots match LAPACK at every size tried (`m` = 1, 2, 5, 20, 79), while the factor entries differ in
5,088 of 6,241 at `m = 79` and the solve agrees to `9.6e-14`.

The `IDAMAX` detail that makes that reproducible is worth naming: the search is **strictly
greater**, so two equal candidates pivot to the **first**. A `>=` there would pass every accuracy
test and pick different rows on a tie.

### 16.8 Three smaller things worth keeping

* **`G` is borrowed, not copied.** `BarrierString` holds the `m × m` admittance block and hands it
  over once per timestep. Reading it through `PyReadonlyArray2` borrows the NumPy buffer;
  `as_f64_field`, which the rest of the binding uses, would have copied 6,241 doubles per step for
  the default fixture. The shim's `np.ascontiguousarray` is a type check rather than a second pass —
  it returns *the same object* for an already-contiguous float64 array — which is the §15.5 lesson
  applied rather than rediscovered.
* **The warning is raised from Python, not from Rust.** `solve_contact_vector` warns rather than
  failing when it hits the iteration cap, and it does so with `stacklevel=2`. That number means
  "my caller", and it cannot mean the same thing issued from inside an extension module. So the
  core returns `(residual, converged)` and the shim does the warning — the same split `radiation`
  used for its refusals.
* **The parity job has been failing to assert anything since batch 1, and the cause was one
  character.** `ci.yml`'s Rust-vs-Python step listed its files across continued lines, and the last
  continuation was a literal `\n` rather than a backslash and a newline. The shell reads that as
  the argument `n`, so pytest was handed a path that does not exist and the step errored out before
  running `test_rust_parity_banded.py` at all. Fixed here. The lesson is the same one §12.9 drew
  about the gate's limits: a step that fails loudly is fine, but this one's failure mode is
  indistinguishable at a glance from the file simply not being listed yet — and a parity test that
  never runs is exactly the empty-assertion shape the `import physsynth_rs` line above it exists to
  prevent, arriving through a different door.

  There is a second, harmless instance of the same corruption elsewhere in the file: several steps
  have their continuations collapsed into one long space-separated line. That *works* — the shell
  sees the same argument list — so it is left alone rather than churned.

  **And it was not the only broken gate.** `cargo clippy --workspace --all-targets -- -D warnings`
  and `cargo fmt --all --check` were both failing on `banded.rs` as well as on this batch's new
  files — the clippy failures on two spellings that batch chose *deliberately* (`!(ajj > 0.0)`,
  which catches a NaN diagonal that `<= 0.0` would not, and `-1.0 * x`, which is DSYR's own
  ordering). Both now carry a scoped `#[allow]` with the reason next to it, which is the right
  outcome: the lint is correct in general and wrong here, and saying so in the file is better than
  either silencing it globally or rewriting arithmetic to please it. Three gates red from one
  batch, all three found only because this batch happened to run them — worth a habit, not just a
  fix.

### 16.9 The success condition

* `tests/test_collision_energy.py`, `test_collision_modal.py`, `test_collision_signature.py` and
  `test_jawari.py` unmodified against Rust — the barrier model and its buzzing-bridge
  configuration.
* `tests/test_mallet_energy.py`, `test_mallet_signature.py` and `test_mallet_wall.py` — the scalar
  solve through its real client, which is the only place the `brentq` fallback is exercised.
* `tests/test_rust_parity_collision.py` — 76 tests, the cause-separator and the blind-spot pin,
  green **both with and without** the flag (§16.4).
* `tests/test_damped_string.py` as the **control**: the barrier's host, ported in §15 and untouched
  here, so a failure there is about the banded solver rather than about this batch.

### 16.10 What was measured

| | |
|---|---|
| Native `cargo test --workspace` | **189 passed** (174 at the end of batch 1) |
| Cargo dependency allowlist | still **EMPTY** — no BLAS, no LAPACK, no linear-algebra crate |
| `tests/test_rust_parity_collision.py` | **76 passed**, flag set and unset |
| NumPy array vs scalar power, exponent 0.5 / 2.0, 200k samples | **94** / **53** disagree |
| `_force_total_vec` vs per-component scalar, `α = 1`, 200k pairs | **174** disagree, to 1.5e-12 |
| Primitives, array and scalar paths, 5 exponents | **bit-identical** |
| Scalar `solve_contact`, 18k configurations | **bit-identical** |
| Mallet trajectory, 5 fixtures, 2,000 steps | **bit-identical**, fallback counts too |
| Barrier trajectory, `m` = 1 and 2, 6,000 steps | **bit-identical** |
| Barrier trajectory, `m = 79` default `K`, 2,000 / 6,000 steps | **bit-identical** / 8.2e-14 |
| Group A window, `α = 1` / stiff `K = 1e8` | first over 1e-13 at step **1,175** / **1,584** |
| Stiff fixture at 5,000 / 10,000 / 20,000 steps | 1.2e-13 / 3.4e-12 / **1.1e-7** |
| `newton_iters`, every fixture, to 20,000 steps | **identical at every step** |
| Dense LU pivot sequence vs LAPACK, `m` = 1…79 | **identical** |
| Dense LU factor entries differing, `m = 79` | 5,088 / 6,241 |
| Dense solve vs LAPACK, `m = 79` | 9.6e-14 relative |
| Lossless energy drift, stiff fixture, 5,000 steps | SciPy 1.12e-12, Rust 1.14e-12 (bar 1e-10) |
| A whole `BarrierString` step, `m = 79` / `m = 1` | **2.97x / 3.39x** faster |
| A whole mallet step (scalar solve), 4,000 steps | **1.42x** faster |
| The two contact models + membrane + string family under `PHYSSYNTH_RS=1` | **405 passed** |
| All ten parity files, flag unset | **910 passed**, 1 skipped |
| All ten parity files, flag **set** | **all pass** — first time this was checked |

### 16.11 What the next batch inherits

- **`mallet` is next, and it finishes Phase 2.** §11.2.2 said Phase 2 would end after Phase 3
  began, and this is that. Its host `Membrane` is already Rust and its contact solve is now Rust,
  so what remains is the model shell: the flight integrator, the two force-injection sites, and
  `MalletWall`. Both were measured bit-identical through the Python shell this batch, so a
  divergence after porting the shell is the shell's.
- **Ask the vectorization question before every remaining primitive.** §16.2. A NumPy function
  called with an array and with a scalar is two computations whenever the exponents `-1, 0, 0.5,
  1, 2` are in reach — and `**` is not the only ufunc with a loop-level fast path.
- **Ask how long the model stays comparable, not how well it agrees.** §16.5. Four of the models
  still to port are nonlinear, and for a chaotic one the agreement window is set by the physics.
- **Bring a fixture that exercises the solver.** §16.4. The physics fixtures were chosen to
  exercise the physics, and for a near-identity Jacobian that means the solver is not under test at
  all. This one cost nothing to catch and would have been invisible for the rest of the migration.
- **§4.1's SuperLU hypothesis is still untested**, and Group D is now the only untouched solver
  class. `beam` keeps its de-risking job at Phase 4.

---

## 17. Phase 2, last batch, as built (2026-08-27) — **Phase 2 is finished**

`physsynth/core/mallet.py` — `MalletMembrane` and `MalletWall`. §11.2.2 predicted that Phase 2
would end *after* Phase 3 began, because `mallet` needs `collision` and `collision` is Group C.
This is that batch, and it is the shortest in the migration: both of the model's hard parts were
already ported, so what moved is the shell — a two-line force-free flight integrator, two
force-injection sites, and the two admittances that scale the contact force.

### 17.1 The shape on disk

```
crates/physsynth-core/src/mallet.rs       Params/WallParams/State, the free functions, scalar_pow
crates/physsynth-core/tests/mallet.rs     native bars (12 tests)
crates/physsynth-py/src/mallet.rs         PyMalletMembrane, PyMalletWall
crates/physsynth-py/src/membrane.rs       four methods widened to pub(crate) + 3 node accessors
physsynth/core/mallet.py                  swap block gated on the flag; `import os`
tests/test_rust_parity_mallet.py          Rust vs NumPy/SciPy (45 tests)
tests/test_stability.py                   the swap guard, + the `collision` gap it had (§17.6)
crates/physsynth-core/tests/collision.rs  the red gate from batch 2, fixed (§17.2)
.github/workflows/ci.yml                  the batch's flagged step, + the contact-leg step split
```

### 17.2 The finding: a constant exponent is not a scalar path, and it turned CI red

**This batch started with two red CI runs, and the live one was §16.2's own test.**

`crates/physsynth-core/tests/collision.rs` asserted that the two power spellings — NumPy's ufunc
fast-path ladder (`x**0.5` as `sqrt`, `x**2` as `x*x`) and the scalar path's `pow` — *disagree
somewhere* over 200,000 samples. It passed locally and failed on CI with "the two paths came out
identical". The obvious explanation is the runner's libm, and it is wrong.

**The cause is LLVM.** At a compile-time-known exponent, `powf` is constant-folded into exactly
the rungs of the ladder it is supposed to be distinguished from: `powf(x, 0.5)` becomes `sqrt(x)`
and `powf(x, 2.0)` becomes `x * x`. The test passed the literal `1.5`, the optimiser propagated it
into `alpha - 1.0`, folded `PowPath::Scalar` into `PowPath::Array`, and made the two arms the same
code. CI builds `--release`; the local run that had been trusted was `cargo test` in debug, where
no folding happens. Measured on this machine in a release build:

| | differing samples, 200,000 |
|---|---|
| array vs scalar, exponent folded (literal `1.5`) | **0** |
| array vs scalar, exponent behind `black_box` | **91** |
| `x.powf(2.0)` literal vs `x * x` | **0** |
| an `#[inline(never)]` `pow` vs `x * x` | **105** |

**Nothing was wrong with the port.** The binding takes `alpha` from Python at runtime, so it can
never be folded, which is why every Python parity test agreed throughout. What was wrong was the
test, and the general form is worth carrying: **a distinction between two spellings of the same
arithmetic is only observable while the compiler cannot see which one you meant.** Two corollaries
that are not obvious from the statement:

* **The debug/release split is not a detail of this bug, it is the mechanism.** A native test that
  pins a floating-point *spelling* asserts something in debug that it does not assert in release.
  Every existing test of that kind should be run both ways; this batch ran the whole native suite
  in both profiles for the first time, and only `collision.rs` differed. `reed.rs`'s neighbouring
  one-ulp test survives because its finding is **associativity** (`rho0 * c0**2` against
  `rho0 * c0 * c0`), which no exponent folding can erase — §13's distinction earning its keep.
* **The audit the rule implies was run, and it is narrow.** Only exponent **2.0** is folded:
  measured in release on this machine, a literal `powf(x, 2.0)` differs from an opaque one in 90 of
  200,000 samples, while literal `powf(x, 3.0)` and `powf(x, 4.0)` differ in **0** — they reach the
  real `pow`. So `ops.rs`'s `h.powf(4.0)` (§10's `h ** 4` finding) is safe as written, which is
  independently confirmed by the operator parity tests still passing. Three production sites do use
  a literal `2.0` and are therefore compiled as multiplies today: `bore.rs`'s `rho0 * c0.powf(2.0)`,
  which is **provably** harmless because `c0` is exactly `343.0` and `343.0 ** 2` is exact; and
  `reed.rs`'s `(wr * k).powf(2.0)` and `reed_velocity.powf(2.0)`, whose arguments are arbitrary
  doubles and so *could* disagree at about one value in 2,000. Measured over 120 bore/reed
  configurations × 300 steps, neither ever did — the reed's `_cy_n` was never spelling-sensitive
  and `reed_damp_work` never differed. Latent, then, not live; recorded here so whoever next
  touches `reed.rs` knows the comment above those lines describes an intent the optimiser discards.
* **The replacement assertion had to change kind, not tighten.** "The two paths differ" is a
  measurement of the C library; "the scalar path equals an opaque `pow`" is a property of this
  code. Only the second is portable, and how *often* the two differ is now reported rather than
  required. That is §14.2's rule — matching a reduction would be a claim about a runner — arriving
  in a **test** rather than in a port, which is a door nobody had watched.

### 17.3 The trap in the shell, which is the same finding pointed the other way

Every constant the mallet owns is a squaring:

```
g_s = k**2 / (rho h**2 (1 + sigma k))     g_h = k**2 / M     KE = 0.5 M ((z - z')/k)**2
```

Those are **Python floats**, so `**` is `float.__pow__`, which is the C library's `pow` and not
`x * x`. Measured 2026-08-27 over 400,000 samples from the range these quantities occupy, the two
spellings disagree in **225**. `g_s` and `g_h` multiply the contact force at every timestep, so
`k * k` would put a last-bit error on the state of every step of every run — **while conserving
energy perfectly**, which is why no bar in this repo could have caught it.

So `mallet::scalar_pow` is `#[inline(never)]`, and per §17.2 that attribute is the only thing
making the distinction survive optimisation: here the exponent genuinely *is* a literal in the
source, which is precisely the condition under which LLVM rewrites it. The native test pins the
structural claim (`scalar_pow(x, 2.0) == x.powf(black_box(2.0))`) rather than a witness value, and
the parity test pins the Python one (`rs._g == k ** 2 / M`) with the inequality asserted only where
the platform makes it observable.

### 17.4 What is bit-identical: everything, and further than anything before it

§16.11 wrote down in advance that a divergence after this batch would be the shell's. It is not.

| | |
|---|---|
| `MalletMembrane` trajectory, 7 fixtures, 2,000 steps | **bit-identical**, drumhead field included |
| the same, default fixture, **50,000 steps** | **bit-identical** — position, penetration and the field; no first differing step |
| `MalletWall`, every observable, **200,000 steps** | **bit-identical**, energy included |
| `fallbacks`, compared step by step | identical; fires **27** times at `alpha = 1` |
| `MalletMembrane.energy()`, 20,000 steps | differs at **381** steps, max `2.2e-16` (`2.5e-15` rel) |

The energy exception is `Membrane.energy()` and nothing else. Two `np.dot` reductions against
left-to-right sums — §14.2's construction — but a **read-out**, not a feedback path: it never
reaches the next timestep, which is exactly why the trajectory stays identical while the number
does not. `MalletWall` owns no field and is exact in every observable, and that contrast is what
**attributes** the gap rather than assuming it. Isolated, the membrane's own energy differs by
`5.7e-15` relative on the same grid.

### 17.5 The question §16.11 said to ask, answered — and the answer is "never"

§16.5 established that for a nonlinear model the agreement window is set by the **dynamics**: the
barrier string is chaotic, so its two trajectories separate exponentially (`1e-13` at ~1,200 steps,
`1.1e-7` at 20,000), and the right question before porting a nonlinear model is "how long before it
cannot be compared". Asked of the mallet, the answer is that the window **does not close**, out to
every run length tried.

The reason is worth stating because it sharpens §16.5 rather than contradicting it. The barrier is
a *sustained* nonlinearity: the string re-contacts the rail every period, so any difference is fed
back through the contact indefinitely. The mallet's contact is a **transient** — the felt engages
once, separates, and thereafter the mallet flies free and the drumhead is a linear FDTD. There is
no mechanism to amplify a difference, and with no difference introduced in the first place there is
nothing to amplify. So the refined question for the four nonlinear models still to port is not
"is it nonlinear" but **"does the nonlinearity recur?"** — `string_nonlinear`, `string_geometric`,
the von Kármán plate and the bow are all sustained, and should be expected to behave like the
barrier rather than like this.

### 17.6 Three smaller things worth keeping

* **The borrow is one phase, and that is a property of the model.** §13.2 cost the reed a
  `step_native` hook and a Rust closure because it injects *inside* the bore's leapfrog. The mallet
  lets the membrane advance force-free and *then* corrects one node, so `step()` takes a single
  `borrow_mut()` and never re-enters the interpreter. Reading and writing that one node goes
  through `readonly()`/`readwrite()`, which borrow the NumPy buffer; `as_f64_field` would have
  copied the whole live field twice per timestep to move one double (§15.5's lesson applied rather
  than rediscovered).
* **The warning stays in Rust, and its `stacklevel` changes number.** The original warns from
  `__init__` with `stacklevel=2`, meaning "my caller". A Rust `__new__` pushes **no Python frame**,
  so the caller is already at level 1 — the same frame, reached by a different count. This is the
  mirror image of §16.8's split: `collision` moved its warning *out* of Rust because a shim frame
  existed to host it; this one stays in because no such frame does. A parity test asserts both
  implementations blame the same line of the same file, which is the only way that arithmetic is
  observable at all.
* **The swap guard had a hole one module wide, and it was shaped like §16.8's.** `collision` was
  never added to `test_stability.py`'s `ported_expected` table, so its ten swapped functions were
  unguarded for a whole batch. The cause is mechanical: three of its public names carry a **leading
  underscore** while their `_py` aliases do not (`_force_total_vec` vs `force_total_vec_py`), so the
  table's derive could not resolve them and the module was simply left out. A guard that silently
  covers nothing is the same failure as a parity job that silently runs nothing — third door, same
  room. Fixed by teaching the lookup the underscored spelling, which keeps the set **derived** from
  the aliases rather than listed.

### 17.7 The success condition

* `tests/test_mallet_energy.py`, `test_mallet_signature.py`, `test_mallet_wall.py` unmodified
  against Rust — the conservation money test, the strike signature, and the closed-form oracle.
* `tests/test_membrane_energy.py` as the **control**: the mallet's host, ported in Phase 2 batch 1
  and untouched here, so a failure there is about the drumhead rather than about this batch.
* `tests/test_stability.py` under the flag, which is where the two new class-identity assertions
  and the repaired `collision` guard live.
* `tests/test_rust_parity_mallet.py` — 45 tests, green **both with and without** the flag (§16.4's
  convention, checked here as it now is everywhere).

### 17.8 What was measured

| | |
|---|---|
| Native `cargo test --workspace` | **201 passed** (189 at the end of Phase 3 batch 2) |
| The same, run in **both** debug and release for the first time | both green; §17.2 is why it matters |
| Cargo dependency allowlist | still **EMPTY** |
| `tests/test_rust_parity_mallet.py` | **45 passed**, flag set and unset |
| All eleven parity files | **955 passed**, 1 skipped, flag set and unset |
| `MalletMembrane` trajectory, 7 fixtures, 2,000 steps | **bit-identical** |
| `MalletMembrane`, default fixture, 50,000 steps | **bit-identical** |
| `MalletWall`, every observable, 200,000 steps | **bit-identical** |
| `MalletMembrane.energy()`, 20,000 steps | max `2.2e-16` (`2.5e-15` of amplitude) |
| `Membrane.energy()` alone, same grid | `5.7e-15` relative — the attribution |
| `float ** 2` vs `x * x`, 400,000 samples | **225** disagree |
| array vs scalar power in release, exponent folded / opaque | **0** / **91** of 200,000 |
| A whole `MalletMembrane` step, `n_live` = 225 / 1,521 / 6,241 | **11.2x / 3.2x / 1.8x** faster |
| A whole `MalletWall` step | **97x** faster |

The `MalletWall` number is the modal body's finding (§12.7) at its limit: the rig calls no compiled
NumPy kernel at all, so what is being removed is *entirely* per-step interpreter overhead. The
coupled model's 11.2x to 1.8x slide across grid sizes is §11.6's crossover, unchanged — as the
membrane's compiled sparse matvec grows, the overhead being saved stops dominating.

### 17.9 What the next batch inherits

- **Phase 2 is closed and Phase 3 has one batch behind it.** Every Group A/B model is ported. What
  remains is Group C and D: the four theta-scheme strings (whose *solver* is already Rust), the
  plate family, `BarrierString`, `bow`, `connection`, `airbox`, `engine`, and the two Group D
  files. `beam` keeps its SuperLU de-risking job at Phase 4, and **§4.1's SuperLU hypothesis is
  still untested** — Group D is the only untouched solver class.
- **Run native tests in both profiles when they pin an arithmetic spelling.** §17.2. It is one
  extra command and it is the difference between a test and a debug-only test.
- **Ask whether the nonlinearity recurs, not whether it exists.** §17.5. A transient nonlinearity
  has no agreement window to close; a sustained one has a short one no matter how good the
  transcription is.
- **Check that a swap guard actually resolves the names it claims to.** §17.6. `collision`'s entry
  was absent for a batch because three public names start with an underscore, and nothing failed.

---

## 18. Phase 3, batch 3, as built (2026-08-27)

`physsynth/core/string_stiff.py` and `physsynth/core/string_damped.py` — models #2 and #3, the
first two **models** out of the four-string chain §15.2 found. The port itself is a transcription;
what makes this batch worth reading is that **neither of its two obstacles was in Rust**, and both
were removed by changing the *Python* side.

### 18.1 The shape on disk

```
physsynth/core/portable.py                  the two spellings, and the reason (new module)
physsynth/core/string_stiff.py              portable.dot + portable.canonical; swap block; import os
physsynth/core/string_damped.py             the same, plus its own swap block
physsynth/core/string_nonlinear.py          the same two edits; `_stretch` deliberately NOT changed
physsynth/core/string_geometric.py          the same two edits, via `_wave_operator`
crates/physsynth-core/src/pyfloat.rs        `scalar_pow`, moved out of `mallet` (new module)
crates/physsynth-core/src/sparse.rs         `Csr::sub`
crates/physsynth-core/src/string_stiff.rs   Params/kernels/StiffString
crates/physsynth-core/src/string_damped.rs  Params/kernels/DampedStiffString/apply_ainv
crates/physsynth-core/tests/string_stiff.rs native bars for both models (14 tests)
crates/physsynth-py/src/string_stiff.rs     PyStiffString + the helpers both classes share
crates/physsynth-py/src/string_damped.rs    PyDampedStiffString
tests/test_rust_parity_strings.py           Rust vs NumPy/SciPy (148 tests)
tests/test_stability.py                     the class half of the swap guard, now DERIVED (§18.7)
.github/workflows/ci.yml                    the batch's flagged step, the parity file, both profiles
```

### 18.2 The finding: both obstacles were an evaluation *order*, and the fix is on the Python side

§15.2 said the four theta-scheme strings are bound into one unit by three `array_equal` anchors
across model *classes* — `sigma1 = 0`, `EA = 0`, `EA = T` — and that the solver therefore had to
port before any model. With the solver common, §15.9 predicted the anchors would no longer be the
obstacle. **They still were, twice, and neither time for the reason that had been anticipated.**

| | Python spells it | Rust can spell | measured disagreement |
|---|---|---|---|
| the energy reduction | `np.dot` → BLAS `ddot` | a left-to-right `for` | **16,797 / 20,000** vectors at n = 99 |
| the operator's column order | SciPy SMMP → **descending** | canonical `Csr` → ascending | **2,000 / 2,000** matvecs, every grid size |

The first is §14.2's rule arriving where nobody had looked for it: all three anchors assert
`a.energy() == b.energy()`, so a Rust model's energy has to equal a *Python* model's exactly, and a
BLAS reduction cannot be reproduced portably.

The second is worse and was on no list at all. `biharmonic_matrix` is `D2 @ D2`, and SciPy's SMMP
kernel emits each row as a **stack** — `has_sorted_indices == False`, columns descending. A CSR
matvec accumulates a row in *stored* order, so `L @ u` is a different sum in the two spellings.
That is not a read-out: it builds the right-hand side of every timestep, so it is a different
trajectory from step one.

**Both were fixed by moving the Python side to the spelling both languages can express**, in a new
module (`portable.py`) scoped to exactly the four models an anchor binds. That is the Phase 1
manoeuvre for the third time and one level down again: `operators` was not a model and swung five;
`banded` was not a model and swung four; this is neither a model nor a solver but an *order of
evaluation*, and it swings the same four.

Three things about the fix are worth stating precisely.

* **It changes the reference implementation's numbers, unconditionally.** `banded`'s swap only
  changed them under the flag. Flag-gating a *Python* model's arithmetic would be worse — it makes
  the default path depend on an environment variable — so the edit is unconditional and in the
  open. Measured: 156 string tests green on the default path, every physics bar unmoved.
* **The reduction was transcribable and the banded solve was not**, which is the contrast that
  makes §15.3 legible. `np.cumsum` is the one NumPy spelling that is sequential by construction —
  measured equal to a naive Python loop in 3,300/3,300 samples at lengths 1 … 4,097 — so the Python
  side stays compiled and costs ~2.5 µs against `np.dot`'s 0.6 µs. `np.sum`, `arr.sum()` and
  `np.add.reduce` are all **pairwise** above a blocksize of 128 and would have been a third answer;
  so would `math.fsum`, which is correctly-rounded and therefore a fourth.
* **The read-out / update-path split is a decision, not a sweep.** `string_nonlinear._stretch` is
  also an `np.dot`, and it is *inside* the tension solve's residual — changing it would move model
  #9's trajectory rather than a reported number. It was left on `np.dot`, and the anchors do not
  reach it (`EA = 0` returns before it). The question to ask at each `np.dot` is not "is this a
  reduction" but **"does this reduction reach the next timestep?"**

  > **Reversed one batch later, by its own rule (§19.2).** Porting model #9 made "it is compared to
  > nothing" false, so both stretch reductions moved to `portable.dot` after all. The bullet's
  > *question* survives and the answer it gave did not — which is §18.3 arriving where §18.2 was
  > standing. And the follow-up question the reversal adds: **does anything downstream of the
  > reduction branch on it?** Here `brentq` does, so the difference was never going to stay at one
  > bit.

### 18.3 Phase 1's canonical-`Csr` decision: the justification expired, the conclusion got stronger

§10 wrote down that the Rust `Csr` is canonical while SciPy's is not, and gave two reasons:
reproducing SciPy's order would mean reimplementing a SciPy internal, and *"nothing downstream reads
`.data` or `.indices`, the matrices are only ever used as operators."*

**The second reason is now false** — a stiff string multiplies by `L` in its inner loop, and that
matvec is exactly a read of `.data` in `.indices` order. But the first reason is *more* forceful
than it was, not less: under the alternative the Rust side would carry SciPy 1.16's stack order as
a constant, and a point release that reordered SMMP's output would silently move every string
trajectory in the project. Sorting the Python side makes the operator independent of which kernel
assembled it, which is what the migration is for. It also avoids giving `Csr` — the type every
future model's matrices flow through — a "which SciPy path built me" mode in order to serve one
model family.

The generalisable form: **a decision justified by "nothing downstream depends on this" has to be
re-examined the moment something downstream is ported, and the question is whether the original
conclusion survives the loss of its stated reason.** Here it did, on a different argument.

### 18.4 What this means for the plate family, recorded rather than fixed

`plate.py` evaluates `self.B @ self.u_prev` every timestep with `B` built from `biharmonic_matrix`,
and `beam.py` does the same with `K`. **Phase 5 hits this wall**, and it now has the answer in
advance: the fix is `portable.canonical` at the operator's assignment, applied to the whole plate
family at once for the same anchor-shaped reason. It is deliberately not done here — those models
have shipped parity measurements built on their current behaviour (§14.2, §17.4), and changing a
number nobody is comparing yet buys nothing.

### 18.5 What is bit-identical, and the qualifier the claim cannot be stated without

| | |
|---|---|
| parameters, `x`, `_L` (`data`, `indices`, `indptr`, `nnz`), 70 fixtures | **bit-identical** |
| `set_state`, all `v0` spellings | **bit-identical**, `u^{-1}` included |
| `StiffString` trajectory + `energy()`, 6 fixtures × 2,000 steps | **bit-identical** |
| `DampedStiffString` trajectory + `energy()`, 6 fixtures × 2,000 steps | **bit-identical** |
| the same, 20,000 steps | **bit-identical** |
| `apply_Ainv`, random and unit right-hand sides | **bit-identical** |
| a **Rust** stiff string vs a **Python** damped string at `sigma1 = 0`, 600 steps | **bit-identical**, energy included |
| all of the above **without** a shared banded solver | 1.7e-14 of amplitude at 100 steps, 1.6e-13 at 20,000 |

**Every exact row above requires both sides to use the same banded solver**, which is what
`PHYSSYNTH_RS=1` arranges and what the parity file's `shared_solver()` arranges otherwise. Without
it SciPy calls OpenBLAS's blocked `DTBSV` and Rust runs the reference `DTBSV` transcribed, and the
two differ in the last bit from the first step. That gap is §15.3's; this batch neither introduced
it nor can remove it — but the parity file *separates* the two causes by holding the solver fixed,
which is what makes "bit-identical" a claim about the port rather than about OpenBLAS. A batch that
could not make that separation would have had to report a tolerance and guess at the cause.

### 18.6 A third agreement regime, and it is the boring one

§16.5 asked how long a model stays comparable and answered "the dynamics decide". §17.5 sharpened
it to "does the nonlinearity *recur*". This batch supplies the third case, and it is the one that
needed no qualifier: **a linear model has no mechanism to amplify a difference at all.**

| model | mechanism | 1e-13 reached at |
|---|---|---|
| `BarrierString` (§16.5) | chaotic re-contact, sustained | ~1,200 steps |
| `MalletMembrane` (§17.5) | contact is a transient | never (exact at 50,000) |
| `DampedStiffString` (here) | linear, no contact at all | never with a shared solver; **1.6e-13 at 20,000** without |

The last row is the interesting one, because it is a divergence that *is* introduced (by the
solver) and still does not run away: it grows like a random walk — 1.7e-14 → 4.8e-14 → 8.6e-14 →
1.6e-13 across 100/500/2,000/20,000 steps, i.e. roughly as the square root of the step count. So
Group A's target is not a "short run" bar here the way §14.4 and §15.4 had to make it. **What sets
the window is not the size of the perturbation but whether the model amplifies it.**

### 18.7 Three smaller things worth keeping

* **The two cores are deliberately near-duplicates, and the duplication buys a detector.** Model #3
  is model #2 plus one term, and one superset `Params` would have been ~150 lines less code. It
  would also have made `test_damped_string.py`'s `sigma1 = 0` anchor **vacuous under the flag** —
  the anchor compares two independent transcriptions, and two names for one implementation compare
  equal for free. §17.6's lesson (a guard that silently covers nothing) says to pay the 150 lines.
* **The class half of the swap guard is now derived, like the function half.** §17.6 found
  `collision` missing from the *function* table and fixed that derive; the *class* half was a block
  of hand-written `assert mallet.MalletWall is physsynth_rs.MalletWall` lines with exactly the same
  hole, one forgotten paste wide. It now reads the swapped classes off the `<Name>Py` aliases the
  modules actually define and checks that set against a written-down expectation — so adding a
  model is a reviewed edit and forgetting one is a failure.
* **`A` is never assembled.** Only its three diagonals are read, and `csr.diagonal(d)` picks by
  column and is independent of stored order — so §18.2's sort provably cannot move the Cholesky
  factor. Measured before the port was written, over 288 parameter combinations: zero of the three
  bands changed. That check is the one assumption in the batch that would have invalidated it
  silently, because a moved `ab` moves the factor and every number downstream of it.

### 18.8 The success condition

* `tests/test_stiff_string.py` and `tests/test_damped_string.py` unmodified against Rust —
  including the `sigma1 = 0` anchor, now between two Rust classes.
* `tests/test_tension_string.py` and `tests/test_geometric_energy.py` unmodified against Rust —
  the anchors' **other end**, still Python, still `array_equal` to a Rust string. These are the
  files that would have failed had `portable.py` not existed.
* The clients: `test_bow_*`, `test_collision_energy.py`, `test_jawari.py`, `test_connection.py`,
  `test_sympathetic.py`, `test_free_plate_connection.py` — every model that vibrates a
  `DampedStiffString`, all still Python, now driving a Rust one.
* `tests/test_beam_modal.py` as the **control**: `beam` builds its own operators and never touches
  this family's, so a failure there is about `operators` rather than about this batch.
* And the eight files already in §15's flagged step that this batch's own step does not repeat —
  `test_geometric_{phantom,polarization,rotating_wave,whirl}.py`, `test_bow_modal.py`,
  `test_collision_{modal,signature}.py`, `test_beam_stability.py`. They are not redundant with the
  default-path run: their flagged configuration *changed* under this batch (Rust banded solver plus
  a **sorted** `_L`, where before it was Rust banded plus a descending one), and they are the
  tolerance-tight dynamical bars — the Mathieu tongue, the phantom-partial ladder, the Helmholtz
  slip pattern. Run flagged in the new configuration: 89 passed. **A batch that changes numbers has
  to re-run every step whose configuration moved, not only the step it added.**
* `tests/test_rust_parity_strings.py` — 148 tests, green **both with and without** the flag.

### 18.9 What was measured

| | |
|---|---|
| Native `cargo test --workspace` | **215 passed** (201 at the end of Phase 2) |
| The same in **both** profiles, now enforced in CI | both green |
| Cargo dependency allowlist | still **EMPTY** |
| `tests/test_rust_parity_strings.py` | **148 passed**, flag set and unset |
| All twelve parity files | **1,104 passed** flagged; 1,103 + 1 skipped unflagged |
| The batch's CI step, flagged | **297 passed** |
| Batch 1's step's other eight files, flagged in the **new** configuration | **89 passed** |
| The four string files on the **default** path, after the `portable.py` edits | **156 passed** |
| `np.dot` vs a left-to-right sum, n = 99 | **16,797 / 20,000** differ |
| `np.cumsum(a*b)[-1]` vs a naive loop, n = 1 … 4,097 | **0 / 3,300** differ |
| `L @ u`, descending vs ascending indices | **2,000 / 2,000** differ |
| `A.diagonal(0/1/2)` before vs after the sort, 288 configs | **0** differ |
| Trajectory + energy, shared solver, 12 fixtures × 2,000 steps | **bit-identical** |
| Worst lossless drift over 1 s at 44.1 kHz, Python / Rust | 9.4e-12 / 9.6e-12 (bar: 1e-10) |
| The same, 20,000 steps | **bit-identical** |
| Trajectory without a shared solver, 100 / 500 / 2,000 / 20,000 steps | 1.7e-14 / 4.8e-14 / 8.6e-14 / 1.6e-13 of amplitude |
| A whole `DampedStiffString` step, `N` = 16 / 64 / 256 / 1,024 / 4,096 | **19.8x / 10.1x / 4.1x / 1.8x / 1.2x** faster |
| `energy()` alone, `N` = 64 / 1,024 | **15.6x / 2.7x** faster |

The step numbers are §11.6's crossover again, unchanged in shape: what is being removed is per-step
interpreter overhead, so the win is large where the grid is small and fades as SciPy's compiled
sparse matvec and banded solve come to dominate. `energy()` slides the same way for the same
reason — at `N` = 64 it is four Python-level calls around three tiny matvecs; at `N` = 1,024 it is
three real matvecs.

### 18.10 What the next batch inherits

- **`string_nonlinear` is the natural next model**, and it is the last one in the chain that does
  not need Group D. Its `_L` and its energy already use the portable spellings, so what remains is
  the tension solve: a `brentq` on a residual that refactors the band at every candidate `dT`,
  which is `reed`'s transcribed Brent (§13.3) meeting `banded`'s factor. **Its `_stretch` is on the
  update path and stays on `np.dot`** (§18.2) — a port must reproduce *that* reduction rather than
  route around it, which is the first time this migration has had to match a BLAS call head-on.
  Measure whether it can before planning the batch.
- **`bow` is still the phase's last model** and is now unblocked in the way §15.9 described: it
  calls `string.apply_Ainv`, which is a Rust solve on a Rust string. Its safeguarded Newton
  iteration count is compared by nothing in the repo — that has to be added, the way §13.3 compared
  the reed's branch choices.
- **Ask whether a reduction reaches the next timestep, not whether it is a reduction.** §18.2.
- **Ask whether an operator is multiplied in a loop before trusting `Csr`'s canonical form.**
  §18.3, and it is live at Phase 5 rather than hypothetical (§18.4).
- **Re-examine a decision when its stated reason expires**, rather than either keeping it on
  inertia or reversing it on the loss of the reason alone. §18.3.
- **§4.1's SuperLU hypothesis is still untested.** Group D remains the only untouched solver class,
  and `beam` keeps its de-risking job at Phase 4.

---

## 19. Phase 3, batch 4, as built (2026-08-27)

`physsynth/core/string_nonlinear.py` — model #9, the tension-modulated (Kirchhoff–Carrier) string.
The third model out of the four-string chain §15.2 found, and the **first model in the project
whose update matrix moves every step**: `A = A0 - beta D2` depends on the tension, so the banded
factorization is inside a scalar root-find rather than at construction, and `banded` (§15) meets
`root` (§13.3) for the first time.

Two things make this batch worth reading. The obstacle §18.10 predicted turned out to have a
different answer than the one it predicted — and the answer was written down one section later, in
§18.3, before anyone knew it applied here. And the batch's one real porting error was **invisible
to the trajectory** and caught by a telemetry attribute, which changes what a parity file for this
class of model has to compare.

### 19.1 The shape on disk

```
physsynth/core/string_nonlinear.py            portable.dot in both stretches; module note; swap block
physsynth/core/portable.py                    the read-out/update-path split, retired (§19.2)
crates/physsynth-core/src/string_nonlinear.rs Params/kernels/solve_tension/TensionModulatedString
crates/physsynth-core/src/lib.rs              the module, and the batch's line in the header
crates/physsynth-core/tests/string_nonlinear.rs native bars (13 tests)
crates/physsynth-py/src/string_nonlinear.rs   PyTensionModulatedString — telemetry, the refusal, the warning
crates/physsynth-py/src/lib.rs                registration
tests/test_rust_parity_tension.py             Rust vs NumPy/SciPy (90 tests)
tests/test_stability.py                       the model added to the derived swap guard
.github/workflows/ci.yml                      the batch's flagged step, the parity file — and a FIX (§19.7)
```

### 19.2 The finding: §18.10's prediction was right about the obstacle and wrong about the fix

§18.10 said the port would have to **match `np.dot` head-on** — "the first time this migration has
had to match a BLAS call" — and told the next batch to measure whether it can before planning.

Measured first, as instructed. **It cannot**, and by a wide margin:

| | measured 2026-08-27 |
|---|---|
| `np.dot(du, du)` vs a left-to-right sum, random vectors n = 16 … 1,024 | 53 % … 93 % differ |
| the same, on the **real** `_stretch` vectors from a 400-step run | **612 / 800** differ |
| worst relative disagreement on those | 6.7e-16 |

So the choice was the one §18.2 had already framed: leave `_stretch` on `np.dot` and accept a
tolerance, or move it to `portable.dot` and change the reference implementation's numbers. §18.2
chose the first, on the stated grounds that the stretch "is compared to nothing".

**Porting this model is what made that grounds false**, which is exactly the rule §18.3 wrote down
one section later — *a decision justified by "nothing downstream depends on this" has to be
re-examined the moment something downstream is ported*. There it survived the loss of its reason on
a different argument. Here it does not: the reduction moved, unconditionally and on the default
path, and the module docstring says so where the old comment said the opposite.

**What settles it is not the ulp, it is what the ulp reaches.** For models #2 and #3 an
implementation difference of one bit stayed one bit. Here the stretch is inside `_solve_tension`'s
residual, so a last-bit disagreement changes `brentq`'s **iterate sequence** — an integer:

| | `np.dot` | `portable.dot` |
|---|---|---|
| residual evaluations over 5,000 steps, flagship fixture | 34,010 | 34,046 |
| steps whose evaluation **count** differs | — | **1,400 / 5,000** |
| `bracket_expansions`, `n_not_converged` | 0, 0 | 0, 0 |

That is the argument against the alternative, and it is stronger than "the numbers would be
looser". Under `np.dot` the parity bar would have had to be ~2e-12 of amplitude — **set by
`tension_tol = 1e-13`, not by the reduction** — which is four orders looser than the last bit it
would supposedly be guarding. A reassociation bug in the Rust stretch would have sat invisibly
underneath it. A test that cannot fail for the reason it exists is not a test.

**The generalisable form, which extends §18.2's own question by one clause.** Ask not only *does
this reduction reach the next timestep?* but **does anything downstream of it branch on the
answer?** A reduction feeding a linear update contributes a last bit. A reduction feeding a
root-find, a Newton safeguard, an Armijo backtrack or a bracket test contributes a *control-flow
decision*, and those do not average out.

### 19.3 What is bit-identical

| | |
|---|---|
| parameters, `x`, `_L` / `_D2` (`data`, `indices`, `indptr`, `nnz`), `_ab0`, `_ab_D2` | **bit-identical** |
| `set_state` including `u^{-1}` — which carries `dT_0` — all `v0` spellings | **bit-identical** |
| trajectory + `energy()` + `nonlinear_energy()`, 8 fixtures × 600 steps | **bit-identical** |
| **all four telemetry attributes**, step for step, over the same runs | **bit-identical** |
| the same, 20,000 steps | **bit-identical** |
| a **Rust** model #9 at `EA = 0` vs a **Python** model #3, 4 loss settings × 600 steps | **bit-identical**, energy included |
| all of the above **without** a shared banded solver | Group A, §19.5 |

Every exact row requires both sides to run the same banded Cholesky — `PHYSSYNTH_RS=1` arranges it,
and the parity file's `shared_solver()` arranges it otherwise. That qualifier is §15.3's and this
batch neither introduced nor can remove it. But it **binds harder here than it did in §18**: there
the solver ran once per step, so a solver gap was a per-step perturbation; here it runs ~7 times
per step *inside a residual*, so it moves the root `brentq` converges on rather than only the
solve. Measured: the step at which two otherwise-identical models first separate is **1,882 with
SciPy's LAPACK and 210 with the Rust transcription**, on the same fixture. The separation step is a
property of the solver, not of the port.

### 19.4 The batch's one real porting error, and why the trajectory could not see it

`_stretch_int` is `((dot + u_0**2) + u_last**2) / h`. The port grouped the two end terms first —
`dot + (u_0**2 + u_last**2)` — which is a different sum in floating point.

**The state was bit-identical through it, for 2,000 steps.** What caught it was `delta_tension`,
differing in its last bit on roughly half the steps. The reason the trajectory is blind is
quantitative rather than lucky: `beta = k^2 dT / (2 rho)` is ~4e-9 in every realistic fixture, so a
one-ulp change in `dT` perturbs `beta` by ~1e-25 and the band entries — which are O(1) — round to
exactly the same doubles.

Two consequences, and the second is the one that generalises.

* **The parity file compares the telemetry first, and exactly.** `delta_tension`, `converged`,
  `bracket_expansions` and `n_not_converged` are public attributes; two of them are integers, so
  they are compared for *equality*, not for closeness. A file that compared only `u` would have
  passed with the kernel wrong.
* **A public read-out can be a strictly sharper detector than the state, and the sharp one is the
  quantity nearest the branch.** §14.3 found the suite systematically blind to a class of
  divergence because every fixture weight was 1.0; this is the same shape one level up — the
  observable that would have shown the bug was not the one a trajectory test looks at. Before
  writing a parity file, ask **which quantity in this model is closest to a control-flow decision**
  and compare that one exactly.

The measurement that pins it, for the record: two Python models differing *only* in the stretch
reduction take a different number of `brentq` evaluations within the first 100 steps, and their
states stay `array_equal` for **1,882 steps** (LAPACK) or **210** (Rust banded). The parity file
asserts both halves rather than describing them.

### 19.5 A fourth agreement regime, and it is the first one set by the model's own dynamics

§16.5 asked how long a model stays comparable and answered "the dynamics decide". §17.5 sharpened
it to "does the nonlinearity *recur*". §18.6 added the linear case, where nothing amplifies at all.
Model #9 supplies a case none of those three cover: **the same model, the same code, gives two
completely different answers depending only on amplitude**, because it has a dynamical threshold
inside it.

Model #9's single-mode motion is parametrically unstable above `dT/T0 ~ 3` — real physics, not an
artifact (the tension pumps at twice the mode frequency and roundoff-seeded neighbours sit in
Mathieu tongues). Below that threshold the motion is regular; above it the mode disintegrates into
its neighbours while conserving energy. Perturbing with the banded-solver gap, on the suite's own
`EA = 1e5` fixture, mode 3:

| step | sub-threshold (peak 2.11 × T0) | above threshold (peak 11.05 × T0) |
|---|---|---|
| 100 | 2.0e-14 | 1.4e-13 |
| 500 | 1.4e-13 | **1.3e-8** |
| 1,000 | 3.0e-13 | **2.6e-3** |
| 2,000 | 1.4e-12 | **3.7e-1** |
| 20,000 | 9.7e-11 | 7.3e-1 (saturated) |

Both runs hold energy to better than 1e-10 throughout, which is the whole point: the divergence is
the model being chaotic, not either implementation being wrong.

**So a parity fixture's amplitude is part of its claim**, in the same way §16.4 found a parity
fixture's stiffness was. The question to ask before porting a model is no longer "is it nonlinear"
(§16.5) or "does the nonlinearity recur" (§17.5) but **"does this model have an operating point
above which it amplifies, and where is my fixture relative to it?"** For #9 the honest statement is
that it is comparable to the bit under a shared solver at *any* amplitude, and comparable at all
without one only below threshold.

### 19.6 The blind fixture, again — and this time it is blind to the thing the batch changed

§16.4's rule fires for the third time, and it is worth recording because the *reason* is new each
time. The barrier's default contact made the Newton Jacobian 1.004, so the new dense LU was a
no-op. The mallet's contact was a transient. Here:

**At small amplitude the tension excess is so small that `brentq` converges the same way whatever
the stretch reduction is.** Measured: two Python models differing only in that reduction are
`array_equal` at 400 steps at amplitude 1e-6, and their `dT` never exceeds 1e-8. A gentle fixture
would therefore have been green **with the port's reduction wrong** — the very thing this batch
existed to get right.

So the parity file's trajectory fixtures are all at an amplitude where the root-find does real
work, and the two halves of that claim are assertions rather than comments:
`test_the_gentle_fixture_is_not_a_test` shows the blindness, and
`test_the_root_find_sees_the_reduction_long_before_the_state_does` shows the chosen fixture is not
blind. Both carry a message saying that a failure means the *measurement* needs re-taking, not the
port.

### 19.7 A CI step had been failing since the previous batch, and the fix is in this one

While adding this batch's step, the `Rust vs Python parity` job was found to contain

```
tests/test_rust_parity_mallet.py \n            tests/test_rust_parity_strings.py
```

— a **literal backslash-n** rather than a line continuation, committed with §18's edit. In `bash`,
`\n` outside quotes is an escaped `n`, so `pytest` was handed a path called `n`, could not find it,
and exited 4. The step has been red since, and the twelve-parity-file number §18.9 reports was a
local run.

It is fixed here, along with the file this batch adds. The generalisable bit is small but real:
**a shell continuation is not checked by anything** — YAML parses the block as an opaque string, no
linter reads it, and the failure mode is a step that fails *loudly for the wrong reason*, which is
easy to misread as flaky. Two other steps in the same file carry the opposite mangling (their
continuations collapsed into one very long line); those are correct and were left alone.

### 19.8 Four smaller things worth keeping

* **The `dict[float, ...]` cache is performance only, and it was nearly dropped on a wrong
  estimate.** The same `dT` refactors to the same bits, so the memo changes nothing observable, and
  the first draft of the port left it out with a note saying it "saves the two evaluations `brentq`
  makes at the bracket ends" and that recomputing costs "the same number". **Measured, that was
  wrong by about a factor of two**: a step takes **4.4** banded solves with the memo and roughly
  twice that without, because `brentq`'s first two evaluations land exactly on the bracket ends the
  caller has already solved for *and* its last lands on the root the caller is about to solve for
  again. So the memo is reproduced — as a short `Vec` scanned linearly, which is what a per-call
  dict of a handful of entries actually is, keyed on `to_bits` because that is the comparison a
  Python dict makes. The lesson is not about caches: **an "it costs nothing" claim in a comment is
  a measurement, and this one was cheap to take and had not been.**
* **`apply_Ainv` raises, and the refusal is part of the interface.** Every other string in the
  family implements it; this one cannot, because `A` moves with the tension and a *constant*
  driving-point admittance does not exist. `bow`, `collision::BarrierString` and `connection`'s
  bridges all call it on whatever string they are handed, so the Rust class has to produce a clean
  `NotImplementedError` with the same text — not a panic, and not a wrong number. This is the
  §13.2 shape again: a surface property `cargo test` cannot see, because it is about what happens
  at the boundary.
* **The `RuntimeWarning` goes through Python's own `warnings` machinery.** A failed bracket warns
  and quotes `self.n` *before* the increment. Emitting it from Rust as a print, or not at all,
  would be invisible to `pytest.warns` and to `-W error` — and the message is the honesty gate that
  says a run must not be rendered as physics. **It is not exercised by any test**, and that is
  stated rather than hidden: the branch is reachable only when 40 bracket doublings all fail, which
  the termination argument says cannot happen for a finite state. It is in the same class as the
  doubling loop itself — written, reviewed, and unreached.
* **The bracket-doubling loop is a dormant branch, and it is driven natively rather than assumed.**
  Swept across ten fixtures spanning amplitude, grid, damping and Courant number, `bracket_expansions`
  was **0** every time: the `max(I^{n+1}(0), I^{n-1})` seed already brackets. §16.6's Armijo hazard
  in a new place — a safety net nothing exercises is a safety net nothing has checked — so
  `crates/physsynth-core/tests/string_nonlinear.rs` calls `solve_tension` directly with a state
  built to enter it.

### 19.9 The success condition

* `tests/test_tension_string.py` unmodified against Rust, including the `EA = 0` anchor (now
  between a Rust model #9 and a Rust model #3) and the parametric-breakup signature test.
* `tests/test_web_backend.py` against Rust — `web/serialize.py` builds a `TensionModulatedString`
  for its `tension` model, so the viewer is this port's only other client (§1.1).
* `tests/test_damped_string.py` against Rust — the anchor's other end, and the control that says a
  failure is about this batch rather than about §18's.
* `tests/test_stability.py`'s swap guard, with the model added to the **derived** class half.
* `tests/test_rust_parity_tension.py` — 90 tests, green with and without the flag.
* The four string files on the **default** path after the `portable.dot` edits, which is §18.2's
  precedent: an unconditional change to the reference implementation has to show every physics bar
  unmoved, measured rather than argued.
* **Not** re-run: the geometric string's files. `string_geometric` does not import
  `string_nonlinear` and the shared evaluation orders were already in place, so its flagged
  configuration did not move. §18.8's rule is to re-run every step whose configuration *changed*,
  which is not the same as re-running everything nearby.
* One file **did** move without being listed anywhere, and it is worth naming because the way it
  moved is the one that hides. `tests/test_rust_parity_banded.py`'s
  `test_the_family_still_reduces_to_itself_exactly_on_the_rust_solver` builds all four strings
  through their **swapped** names, so under the flag its `EA = 0` anchor became Rust-model-#9 versus
  Rust-model-#3 where it had been Rust-versus-Python. That is only a real comparison because the two
  Rust cores are separate transcriptions — `string_nonlinear` writes out model #3's right-hand side
  again rather than calling `string_damped::step_rhs`, deliberately and for §18.7's reason. Had it
  reused it, the anchor would have compared one implementation with itself and **still passed**,
  which is what a vacuous test looks like from outside. It is covered by the parity step in both
  modes (1,194 flagged / 1,193 + 1 skipped unflagged) rather than by a step of its own.

  What the two cores *do* share is `update_matrix_bands`, so the `EA = 0` anchor is not an
  independent check of `A0`'s assembly — the `sigma1 = 0` anchor is, because model #2 has its own.
  The native test says so in place of implying otherwise.

### 19.10 What was measured

| | |
|---|---|
| Native `cargo test --workspace` | **228 passed** (215 at the end of §18) |
| The same in `--release` | **228 passed** |
| Cargo dependency allowlist | still **EMPTY** |
| `tests/test_rust_parity_tension.py` | **90 passed**, flag set and unset |
| All thirteen parity files | **1,194 passed** flagged; 1,193 + 1 skipped unflagged |
| `tests/test_tension_string.py` against Rust | **39 passed** |
| The batch's CI step, flagged (adds the viewer's 403) | **475 passed** |
| The same files plus the swap guard, on the **default** path | **494 passed** |
| `np.dot` vs a left-to-right sum, real `_stretch` vectors | **612 / 800** differ, worst 6.7e-16 |
| brentq evaluation count, the two spellings, 5,000 steps | differs on **1,400** steps |
| `bracket_expansions` over ten fixtures | **0** — the branch is dormant |
| Banded solves per step, with the memo / without | **4.4** / roughly twice that |
| Trajectory + energy + telemetry, shared solver, 8 fixtures × 600 steps | **bit-identical** |
| The same, 20,000 steps | **bit-identical** |
| First separation without a shared solver, LAPACK / Rust banded | step **1,882** / **210** |
| Divergence sub-threshold, 100 / 1,000 / 20,000 steps | 2.0e-14 / 3.0e-13 / 9.7e-11 of amplitude |
| Divergence above threshold, 100 / 1,000 / 20,000 steps | 1.4e-13 / **2.6e-3** / 7.3e-1 |
| `np.float64(x) ** 2` vs `math.pow(x, 2.0)` | **0 / 200,000** differ — `scalar_pow` is the right spelling |
| `np.float64(x) ** 2` vs `x * x` | 92 / 200,000 differ |
| A whole step, `N` = 16 / 64 / 256 / 1,024 | **32.1x / 13.8x / 5.6x / 3.4x** faster |

The step numbers are §11.6's crossover once more, and this model sits **higher on it than any
string before it**: 32x at `N` = 16 against model #3's 19.8x, and 3.4x at `N` = 1,024 against its
1.2x. The reason is the same reason the model is expensive — what Rust removes is per-call
interpreter overhead, and a root-find pays that overhead **4.4 times per step** where a linear
theta-scheme string pays it once. The models that gain most from the port are the ones whose Python
loop calls a compiled kernel *many times with small arguments*, which is a property of the
algorithm rather than of the grid.

### 19.11 What the next batch inherits

- **`bow` is the phase's last model**, and it is now fully unblocked: it calls
  `string.apply_Ainv`, which is a Rust solve on a Rust string. Its safeguarded Newton iteration
  count is compared by nothing in the repo — §19.2 is the reason that has to be added rather than
  assumed, because a Newton *iteration count* is exactly the kind of downstream branch a last-bit
  difference reaches. §13.3 compared the reed's branch choices; do the same here.
- **`string_geometric` is the fourth and last of the chain**, and it is the one that needs Group D:
  it uses banded Cholesky *and* a sparse LU. It is a Phase 5 model for that reason and not a
  candidate yet.
- **Ask what a reduction *feeds*, not only whether it reaches the next timestep.** §19.2. A
  reduction under a root-find, a Newton safeguard or a bracket test contributes a control-flow
  decision rather than a last bit.
- **Ask which quantity is nearest a branch, and compare that one exactly.** §19.4. The trajectory
  can be blind to an error the telemetry shows immediately.
- **Ask where the fixture sits relative to the model's own amplification threshold.** §19.5. It is
  the third form of §16.4's fixture question and the first one that is about *amplitude*.
- **§4.1's SuperLU hypothesis is still untested.** Group D remains the only untouched solver class,
  and `beam` keeps its de-risking job at Phase 4.

---

## 20. Phase 3, batch 5, as built (2026-08-27)

`physsynth/core/bow.py` — the bowed string, the project's first **continuous nonlinear exciter** and
the model §19.11 called the phase's last. Almost everything hard about it was ported before this
batch: the banded solve in §15, the string it draws across in §18, the transcribed Brent in §13.3,
and the scan-and-bracket idiom in §16. What is left is a shell, one line of arithmetic, and the
question §19.11 asked and could not answer from where it stood.

Three things make the batch worth reading. The one line of arithmetic is a **third kind** of
two-spelling hazard, after §16.2's ufunc ladder and §17.2's compiler fold. §19.11's question — does a
last bit reach the Newton iteration count and the fallback branch — was **measured rather than
assumed**, and the answer is the opposite of the tension string's. And §16.5's agreement-window
question gets a fifth answer that is the first one where the gap **does not grow at all**, for a
reason that is about the physics rather than about the port.

### 20.1 The shape on disk

```
physsynth/core/bow.py                     module note; the two _py aliases; swap block
crates/physsynth-core/src/bow.rs          Params/State/friction/residual+scan_residual/solve/apply/BowedString
crates/physsynth-core/src/collision.rs    `linspace` made `pub` — the two models share one scan grid
crates/physsynth-core/src/lib.rs          the module, and the batch's line in the header
crates/physsynth-core/tests/bow.rs        native bars (11 tests), including the spelling pin
crates/physsynth-py/src/bow.rs            PyBowedString — the refusal, the getters, `step_reporting`
crates/physsynth-py/src/string_damped.rs  four crate-internal accessors the bow needs
crates/physsynth-py/src/lib.rs            registration
tests/test_rust_parity_bow.py             Rust vs NumPy/SciPy (34 tests)
tests/test_stability.py                   the model and its two functions added to the swap guard
tests/test_ci_workflow.py                 NEW — §19.7's bug, asserted rather than remembered (§20.7)
.github/workflows/ci.yml                  the batch's flagged step, the parity file — and §19.7 again
```

### 20.2 The finding: a *hand hoist* is the third kind of two-spelling hazard

`physsynth/core/bow.py` evaluates the friction residual in two places. They look like the same
expression and are not:

```python
# _residual, which Newton and brentq both call
v_rel - v_free + self._g * self._friction(v_rel)
#   ... where _friction is  force * sqrt(2a) * v * exp(-a v^2 + 1/2)
#   so the assembled shape is:   (v - v_free) + g * (((force*sqrt(2a)) * v) * exp(...))

# _bracketed_root, which scans for sign changes
vs - v_free + g * (force * math.sqrt(2.0 * a)) * vs * np.exp(-a * vs * vs + 0.5)
#   so the assembled shape is:   (v - v_free) + (((g * (force*sqrt(2a))) * v) * exp(...))
```

The second hoists `g * (force * sqrt(2a))` into a single Python float so NumPy can apply one scalar
to the whole 512-point scan array, instead of multiplying by `g` after the array product. Floating-
point multiplication is not associative, so these are **different doubles**. Measured 2026-08-27 at
the canonical rig's real `g` (0.318), over 20,000 samples per fixture: they disagree in **4,158** of
them at the flagship `force = 4, a = 120`, in 5,372 at the default `force = 1`, and in 568 at the
weak `force = 0.02`. The spread is the point — the fraction is set by how large `g * force * sqrt(2a)`
is next to `v - v_free`, so it is a property of where the bow is being played and not a fixed number.
(An earlier draft of this section quoted 306, from a scratch script that had hardcoded `g = 1e-3`
rather than reading it off the model. Recorded rather than quietly corrected: a measurement taken at
a parameter the model never uses is the same error as a fixture that does not exercise the solver
(§16.4), reached from the measuring side.)

This is the third distinct way this migration has found one expression to be two computations, and
the three have nothing in common except the consequence:

| | what makes the two spellings | who introduced it |
|---|---|---|
| §16.2 | array vs scalar dispatch — NumPy's power *ufunc loop* shortcuts `-1, 0, 0.5, 1, 2` | NumPy |
| §17.2 | a literal exponent — LLVM folds `powf(x, 2.0)` into `x * x` | the compiler |
| §20.2 | a **hoist**, written by hand for the array path | the author of `bow.py` |

The first two are a library's or a compiler's business, and the port's job is to know they exist.
The third is *in the source*, visible, and reads like a duplicate that wants tidying. Which is
exactly why it needs a pin: nothing about the physics changes if the two are merged, and no bar in
this repo moves. What changes is **which brackets exist**. The scan's only purpose is
`rs[:-1] * rs[1:] < 0.0`, so a value that moves by a last bit across zero is one `brentq` call that
does not happen — and at a slip event, where the stick root has just vanished, the surviving root is
the branch the string takes. So the port carries `residual` and `scan_residual` as two functions with
a comment saying they must not be merged, and both suites pin it:
`crates/physsynth-core/tests/bow.rs::the_two_residual_spellings_are_not_the_same_double` and the
Python file's counterpart. Unlike §17.2's pin this one needs no `#[inline(never)]` — LLVM may not
reassociate floating-point multiplication without fast-math, which Rust does not enable — but it is
run in both profiles anyway, because that is what §17.2 established and it costs nothing.

**And one hazard that was checked and is not real — on this machine.** The scan calls `np.exp` on an
array while `_residual` calls `math.exp` on a scalar, which is precisely §16.2's shape. Measured over
20,000 samples across this model's argument range (`-a v^2 + 0.5`, so everything at or below 0.5),
the two agreed **20,000 times out of 20,000**. NumPy's power ufunc has a shortcut ladder; its `exp`
does not.

That is a claim about a **runner**, in §14.2's exact sense, and it is worth saying which one. On
Windows all three implementations — NumPy's array loop, CPython's `math.exp`, Rust's `f64::exp` —
reach UCRT's `exp`, which is why they agree transitively. On the Linux CI runner NumPy uses its own
SIMD loop while CPython and Rust both reach glibc, so the array and scalar paths are genuinely two
implementations there and may differ in a last bit.

**The exposure if they do is bounded and does not reach the trajectory**, which is why this is an
accuracy note rather than a risk. The scan's values are used for exactly one thing —
`rs[:-1] * rs[1:] < 0.0` — so a last bit changes an answer only for a sample within an ulp of zero,
and even then the bracket on either side of it still contains the same root. `brentq` is then handed
`_residual`, which is the same libm call on both sides. So a divergence costs at most one redundant
or one skipped bracket around a root that is found anyway.

### 20.3 §19.11's question, answered — and the answer is the opposite of §19.2's

§19.11 asked for the bow's safeguarded-Newton iteration count and its fallback branch to be
*compared* rather than assumed, because §19.2 had just found a last bit in a reduction changing the
tension string's `brentq` iteration count on **1,400 of 5,000 steps**. The comparison needed
something to compare: neither implementation reported the count. So the Rust binding grew
`step_reporting`, which is not part of the Python model's interface and exists only for the parity
file, and the Python side is instrumented in the test by patching `_residual`, counting calls, and
muting the bracket for the duration.

What is counted is **residual evaluations in the Newton phase, seed included** — not accepted steps.
Two reasons, both about being comparable at all: it is what the Python side can count without being
rewritten, and it separates a *rejected* Newton attempt from a converged one, which an accepted-step
count folds together.

Measured 2026-08-27, over 4,000 and 20,000 steps, on three fixtures:

* under a **shared solver**, the eval counts are identical on every step and the fallback fires on
  the same steps — **0 differences in 20,000 steps on all three fixtures**, with the string's field
  bit-identical at every one of those steps;
* with **independent solvers** — SciPy's blocked `DTBSV` against the transcription — the fallback
  branch still never differs, and the eval count differs **at most once in 20,000 steps**.

The reason it is not the tension string's answer is worth stating, because it is the rule rather
than the number. There, the perturbed quantity was inside a `brentq` bracket test whose scale was the
quantity itself, so a last bit was a coin flip near the tolerance. Here the perturbation is ~1e-14
**relative** while the Newton tolerance is 1e-13 **absolute** in velocity units of order 0.1 — so the
test `abs(r) <= newton_tol` is being asked about a number two orders of magnitude away from the
threshold on almost every step. So §19.2's rule gains its own qualifier: *ask what a reduction feeds*
is right, and the follow-up is **how far the fed quantity sits from the branch's threshold**. A
control-flow difference needs the perturbation and the threshold to be the same size.

### 20.4 What is bit-identical

Under a shared banded solver — which is what `PHYSSYNTH_RS=1` arranges, and what
`test_rust_parity_bow.py`'s `shared_solver()` arranges without it — **everything**, on all three
fixtures and at every one of 20,000 steps (the parity file asserts 4,000, which is what fits a test's
budget; the 20,000 is the measurement behind it): the string's whole field, `v_rel`, `bow_force`, `bow_power`, `bow_work`,
`fallbacks`, `n`, `energy()`, the Newton eval count per step, and the fallback flag per step.

`energy()` being on that list is not the usual case and is worth attributing rather than enjoying:
`DampedStiffString.energy()` is a reduction, and §14.2 is the section that says a reduction is where
bit-identity ends. It survives here because §18.2 already moved this family's reduction off `np.dot`
and onto `portable.dot`, a left-to-right sum both languages can express. The bow inherits that; it
did not earn it.

`_a_full` — the driving-point admittance — is the one construction-time quantity that is *not*
bit-identical without a shared solver, because it is a banded solve. It agrees to 1e-16 relative,
well inside Group A, and it is also the reason this model escapes §15.4's window: **the admittance is
solved once**, at construction, not once per step.

### 20.5 A fifth agreement regime, and the first one that does not grow

§16.5 asked how long a model stays comparable rather than how well it agrees. The four answers so
far: the barrier separates exponentially because it is chaotic (§16.5); the mallet never separates
because its nonlinearity is a transient (§17.5); the linear strings drift like a random walk because
nothing amplifies (§18.6); the tension string does either depending on which side of its parametric
threshold the fixture sits (§19.5).

The bow is a **recurring** nonlinearity — a slip every period, forever — so §17.5's rule predicts the
barrier's regime. It is wrong. Measured 2026-08-27 with independent solvers, over 20,000 steps, on
all three fixtures: the field gap sits at ~1e-14 of the run's peak amplitude from step 500 onward and
never exceeds **6.7e-14**. Flat. Not a trend, and in the flagship fixture the last reading is *smaller*
than the peak.

The reason is that a bowed string is driven onto a **stable limit cycle**. Helmholtz motion is an
attractor: the amplitude is set by the balance of bow work against loss, so a perturbation transverse
to the cycle is squeezed back onto it rather than amplified. The barrier's re-contact is chaotic —
positive Lyapunov exponent, neighbouring trajectories separate; the bow's is contracting.

So §17.5's question — *does the nonlinearity recur?* — is necessary and not sufficient, and the
missing word is the one that decides the sign:

> Ask whether the recurring nonlinearity drives the system **onto** an attractor or **off** one.
> Recurrence says the difference keeps being fed; only the sign of the transverse exponent says
> whether it is amplified or absorbed.

This is the first regime in the migration where a longer run is *better* evidence than a short one,
and it is why the parity file asserts a **ceiling over 8,000 steps** rather than a short-run Group A
target. A ceiling is a meaningless assertion against an exponential and a real one against a flat
line.

### 20.6 The normaliser is part of the claim, and it moved the number by two orders of magnitude

§14.2 established that a decaying trajectory must be normalised by amplitude rather than pointwise —
a pointwise comparison of a decaying signal reads 1e-7 and means nothing. The bow needs one more
word, and the first bar written for this batch failed because of it.

Normalising by the **instantaneous** field maximum, the same runs read a worst case of **2.9e-12**,
with visible spikes at particular steps. Normalising by the **running peak** amplitude, they read
**6.6e-14**, flat. The difference is entirely the denominator: Helmholtz motion beats, so the
instantaneous maximum passes through near-nodes where it is an order of magnitude below the run's
scale, and a fixed numerator divided by a dipping denominator looks like a divergence event. The
spikes are in the normaliser, not the trajectory.

The general form, which applies to every remaining oscillatory model:

> Normalise by a **monotone** scale — the running peak — not by the instantaneous one. An
> instantaneous normaliser reports the *signal's* zeros as the *comparison's* excursions, and a bar
> set from it is a bar set from a beat pattern.

This one cost only a failing test, because the trajectory was flat and the artefact was 40x. Had the
model been genuinely diverging, the same artefact would have been indistinguishable from the thing
being measured.

### 20.7 §19.7's bug, reintroduced by the batch that cites it — and now guarded

§19.7 found a CI step that had been red since the previous batch because a shell line continuation in
a `run:` block had been written as the two characters backslash-`n` instead of a backslash and a
newline. The section's closing line was that "a shell line continuation is checked by nothing".

Adding this batch's step to the same file **reproduced it**, in the same file, within an hour of
citing the finding. Not by typing it: the step was added by a script, and the string passed through
one round of escaping too few. That is the whole mechanism, and it is why remembering the finding was
not enough to avoid it — the failure is introduced by *tooling*, and tooling does not read section
headers.

So `tests/test_ci_workflow.py` now asserts it, and the design of that file is the part worth keeping:

* it scans the **raw text**, not the parsed YAML. After parsing, a literal backslash-`n` and a real
  newline are both just characters in a string and the distinction is gone. The raw file is where the
  two are still different things — which also means no YAML parser is imported and the test adds no
  dependency;
* it asserts the *general* form as well as the specific one: every `tests/...py` token anywhere in
  the workflow must be a file that exists. That catches a renamed test and a deleted one as well as a
  mangled continuation;
* it asserts the **count** of tokens it found (117 today, bar set at 50). A scan that quietly stopped
  matching would pass forever while checking nothing — §16.8's shape, reached through a fourth door.

### 20.8 Four smaller things worth keeping

* **`collision::linspace` is now `pub`, and it is the one NumPy spelling in this crate that is
  shared.** The crate open-codes `np.linspace` in eight places — every model's `grid()`,
  `membrane::linspace_from_zero`, `ops2d::grid_coords` — and that stays as it is, because those are
  one-off coordinate axes that are read once. This one is different in kind: `collision::solve_contact`
  and `bow::bracketed_root` run the *same algorithm* over different residuals, and the grid decides
  which brackets exist. Two copies of it would be two things to keep in step at the exact point where
  drifting changes an answer. The distinction is "same algorithm" versus "same one-liner", and it is
  the reason for the exception rather than a general move toward sharing.
* **The borrow is one phase, and that is a property of the model rather than a precaution.** §13.2
  found that a `&mut self` pymethod cannot hand control back to Python and still be read — the reed
  pays for that with `step_native` and a Rust closure because it injects *inside* the bore's leapfrog.
  The bow, like the mallet, lets the string advance force-free and *then* corrects it, so `step()`
  takes one `borrow_mut()` and never re-enters the interpreter. Two of the three coupled models
  avoid §13.2 by their physics; only the reed had to be engineered around it.
* **`u +=` had to stay an in-place write.** The original's
  `self.string.u += self._force_pref * f_B * self._a_full` is an in-place `__iadd__` followed by an
  assignment of *the same object* back through the property, so a caller holding `.u` from before the
  step sees the correction — the property `connection.py` depends on for its bridge force. The
  binding writes through the live buffer for that reason; rebinding to a fresh array would have been
  green in every bow test and wrong for the viewer.
* **The `force = 0` anchor is a cross-class bit-identity claim and must not be short-circuited.**
  `test_zero_force_is_bit_identical_to_bare_string` compares a bowed string against a bare
  `DampedStiffString` — §15.2's shape, one model further out. It holds because `u += (pref * 0) * a`
  is an addition of a signed zero, which is the identity on every entry. A `if f_B == 0: return`
  fast path on one side only would empty the anchor while making it faster, which is the same failure
  shape as a guard that covers nothing (§17.6).

### 20.9 The success condition

* `tests/test_bow_energy.py`, `tests/test_bow_modal.py` and `tests/test_bow_stability.py`
  unmodified against Rust — the model's own four criteria, including the stick-slip signature and
  the slip-fraction-equals-`beta` oracle that a last-bit perturbation could plausibly move.
* `tests/test_web_backend.py` against Rust — `web/serialize.py` builds a `BowedString` for its `bow`
  model, so the viewer is this port's only other client (§1.1).
* `tests/test_damped_string.py` against Rust — the string underneath, and the control that says a
  failure is about this batch rather than about §18's.
* `tests/test_stability.py`'s swap guard, with the model added to the **derived** class half, both
  module functions added to the function half, and a new captured-binding assertion
  (`bow.DampedStiffString is string_damped.DampedStiffString`). That last one is belt-and-braces
  here and deliberately kept: the Rust `BowedString` raises `TypeError` on a Python string, so a
  mis-ordered import fails loudly at construction anyway — the assertion exists to say *which* of the
  two broke.
* `tests/test_rust_parity_bow.py` — 34 tests, green with and without the flag.
* `tests/test_ci_workflow.py` — 2 tests, and they are about the gate rather than the physics (§20.7).
* **Not** re-run: the barrier's and the mallet's files. Neither model's configuration changed —
  `collision` and `mallet` do not import `bow` and nothing they call moved. §18.8's rule is to re-run
  every step whose configuration *changed*, which is not the same as re-running everything nearby.

### 20.10 What was measured

| | |
|---|---|
| Native `cargo test --workspace` | **239 passed** (228 at the end of §19) |
| The same in `--release` | **239 passed** |
| Cargo dependency allowlist | still **EMPTY** |
| `tests/test_rust_parity_bow.py` | **34 passed**, flag set and unset |
| All fourteen parity files | **1,228 passed** flagged; 1,227 + 1 skipped unflagged |
| The bow's own three files against Rust | **59 passed** |
| `tests/test_web_backend.py` against Rust | **408 passed** |
| The batch's CI step, flagged (all five files) | **495 passed** |
| Bit-identity, shared solver, 3 fixtures × 20,000 steps | **exact** in every observable, `energy()` included |
| Newton eval count, shared solver | identical on **20,000 of 20,000** steps, all three fixtures |
| Newton eval count, independent solvers | differs on **at most 1** step in 20,000 |
| Fallback branch, independent solvers | identical on **20,000 of 20,000** steps |
| Field gap, independent solvers, 20,000 steps | **6.6e-14** worst of running peak amplitude — flat |
| The same, normalised instantaneously | 2.9e-12 worst — an artefact of the denominator (§20.6) |
| `residual` vs `scan_residual`, at the model's real `g` | differ in **4,158 of 20,000** (flagship); 568-5,372 across fixtures |
| `np.exp` (array) vs `math.exp` (scalar), **on Windows** | agree in **20,000 of 20,000** — a claim about a runner (§20.2) |
| Speed, `N = 100`, three fixtures | **7.8x – 8.4x** (26.7–28.7 µs/step → 3.4–3.6 µs) |
| Speed, `N = 400`, three fixtures | **3.3x – 3.6x** (34.0–34.9 µs/step → 9.7–10.2 µs) |

The speed numbers are §11.6's rule with the clearest illustration the migration has produced. The
bow's per-step cost is a scalar root-find written in Python — an interpreted loop of attribute
lookups and `math.exp` calls, with no compiled kernel to hide behind — so at `N = 100` the win is the
full per-step-overhead factor. At `N = 400` the string's own compiled banded back-substitution starts
to dominate and the factor falls by more than half, without either implementation getting slower.
Same crossover, same cause, and here it is visible within one model rather than across two.

### 20.11 What the next batch inherits

- **`BarrierString` is the phase's true last model, not `bow`.** §19.11 said the bow was, and that
  was inherited from §16, where `BarrierString` was correctly described as waiting on its host
  `DampedStiffString`. §18 ported that host and the sentence was never revisited. Checked rather than
  assumed: `collision.BarrierString` now needs only ported machinery — `apply_Ainv` on a Rust string,
  `solve_contact_vector`, and the dense LU — so **Phase 3 is not finished**, and finishing it is a
  batch rather than an audit. This is §18.3's rule in its plainest form: a statement justified by a
  dependency expires when the dependency lands, and nobody is notified.
- **`string_geometric` remains a Phase 5 model.** It uses banded Cholesky *and* a sparse LU, so it
  needs Group D. Unchanged by this batch.
- **A hoist is a spelling.** §20.2. Before porting any function that evaluates the same expression
  on an array and on a scalar, diff the two by *association*, not only by which library call they
  reach. The array path will have had a scalar factored out of it, because that is what makes the
  array path worth writing.
- **Ask how far the perturbed quantity sits from the branch's threshold**, not only whether it
  reaches a branch. §20.3. §19.2's rule finds the branches; this one says which of them can actually
  flip.
- **Ask whether a recurring nonlinearity drives the system onto an attractor or off one.** §20.5.
  Recurrence decides that the difference keeps being fed; the sign of the transverse exponent decides
  whether it grows.
- **Normalise a comparison by a monotone scale.** §20.6. An instantaneous normaliser turns the
  signal's own zeros into the comparison's excursions.
- **§4.1's SuperLU hypothesis is still untested.** Group D remains the only untouched solver class,
  and `beam` keeps its de-risking job at Phase 4.

## 21. The CI repair between batches 5 and 6 (2026-08-27)

Not a batch. `main` was red for **seven consecutive runs**, and the reason is one defect class
written four times. §16.2 found it, §17.2 named it, and §19 and §20 then reintroduced it — so it is
recorded here as a rule rather than as another scar.

### 21.1 The rule the four instances share

**Opacity to the compiler makes an arithmetic difference *observable*; it does not make one
*exist*.** §17.2 correctly concluded that a test pinning a spelling asserts nothing unless the
exponent is hidden from LLVM, and both later batches applied that faithfully. What neither noticed
is that the conclusion is only half of the condition. Whether `pow(x, 2.0)` and `x * x` differ *at
a given x* is the C library's business, and whether `np.dot` and a left-to-right sum differ *at a
given weight* is OpenBLAS picking a kernel by CPU. A test may assert **self-consistency** — that
the port evaluates the spelling the model specifies — on any machine. It may **not** assert that
the machine separates the two spellings, at a hardcoded witness or at all.

The operational form: **search for the witness, report the count, and let the strict branch fire
only where the search succeeds.** `collision` said so in a comment and `test_rust_parity_mallet`
implemented it as a guard; the two tests below did neither.

### 21.2 The two that were red, and what distinguishes them

- `crates/physsynth-core/tests/string_nonlinear.rs` asserted that a witness found on Windows/UCRT
  separates `pow` from a multiply. It does not on the runner, and **whether anything does there is
  unmeasured** — the test is now written so it does not have to be known. A correctly rounded `pow`
  returns the exact product rounded once, which is what `x * x` already is, so a library that
  rounds it correctly has no witnesses at all; UCRT does not, and 78 of 199,999 sampled values
  separate here. Where none separate, the two spellings are the same function and the port is right
  either way. **`test_rust_parity_mallet` had already reached this** and guards its own strict
  assert with `if k ** 2 != k * k`; this is that pattern with the witness searched for rather than
  given, and generalising it is what §21.1 is.

  A caution recorded because the first draft of this section got it wrong: that mallet test passing
  on Linux is **not** evidence that witnesses exist there — its guard may simply have skipped — and
  `test_rust_parity_collision`'s inequality is about the dense LU, not about `pow`, so it is not
  evidence either. **A guarded assert reports nothing about the machine that skipped it**, which is
  the price of the pattern and the reason the count is printed as well.
- `tests/test_rust_parity_radiation.py` asserted `diverged` — that a weight of 0.02 makes NumPy's
  fused multiply-add visible against Rust's plain sum. **Here the phenomenon itself is
  machine-specific**, because a fused product differs from a rounded one only when the product
  rounds, and that is a kernel choice. The runner's kernel does not separate this weight. The two
  Group A bars the test exists to justify hold either way and stay asserted; the divergence is now
  printed.

### 21.3 What the seven red runs actually cost

The visible failure was never the expensive part. Because the native bars run **before** the twelve
`PHYSSYNTH_RS=1` steps and the parity step, a red assert in `cargo test` skips all thirteen. Batches
4 and 5 died there, so **the banded solver, collision, the mallet, the two theta-scheme strings, the
tension string and the bow have never had their parity steps run green on the runner** — five
batches. The one run that got furthest (the mallet's) is the only evidence any of it works on Linux,
and it is what §21.2's first bullet is argued from.

**Order a gate's steps so the cheap machine-specific claims cannot mask the expensive portable
ones**, or accept that one red assert hides five batches of coverage.

### 21.4 The one still-unproven claim of this class

`tests/test_rust_parity_tension.py`'s `evals_a != evals_b` (§19.2's finding, that the two stretch
spellings take a different number of `brentq` iterations) is the same kind of claim and **has never
run on Linux**. It is left asserted deliberately: unlike radiation's fused product, it rests on
summation *order across a whole vector*, which is far more robust than one product's rounding, and
softening a guard on speculation costs a real test. If the runner goes red there, this is why.
`test_rust_parity_bow`'s `np.any(hoisted != direct)` is **not** at risk — on Linux `np.exp` uses
NumPy's SIMD loop while `math.exp` reaches glibc, so those two separate *more* there, not less.

### 21.5 What the first green-through run actually found

The repair above got the gate through all twelve `PHYSSYNTH_RS` steps — **the first time the banded
solver, collision, the mallet, the two theta-scheme strings, the tension string and the bow have
been exercised against Python on the runner.** All twelve passed. So did 1,226 of 1,227 parity
tests. Five batches of porting are now verified on a second platform, which is the result; the
seven red runs were never about the asserts they printed.

**§21.4's prediction was wrong**, and worth keeping as a calibration note.
`test_rust_parity_tension`'s `evals_a != evals_b` — the claim flagged as the one still-unproven
machine assertion — **passed**. The reasoning for leaving it (summation order across a whole vector
is far more robust than one product's rounding) held. The single failure came from somewhere the
section did not look at all.

### 21.6 A bar can be decided by the machine without ever mentioning one

`test_rust_parity_strings::test_trajectory_meets_group_a_against_scipys_own_solver[coarse]` read
1.01e-13 against a 1e-13 bar. Nothing asserts a platform fact here — it is an ordinary agreement
tolerance — but the *quantity* it measures is one BLAS's blocked `DTBSV` against the reference
transcription, so its value is a property of the runner exactly as §21.1's cases are. Windows read
7.6e-14 on the same fixture: 76% of the bar. **A threshold sitting that close to its measurement is
decided by whichever machine runs it**, and that is the same defect as §21.1 wearing a tolerance
instead of an inequality.

The fix changed **no tolerance**. The run was shortened from 500 steps to 100, which is the window
§15.4 defines for a fed-back solve — Group A is a run-length claim (§14.4), and the 500-step version
was testing four times past the claim it cites and buying its margin from the overshoot. At 100
steps the worst fixture reads 3.0e-14, a third of the bar.

**The general form, and the one to carry into Phase 4:** when a bar is close to its measurement,
ask what the measurement is a property of *before* adjusting the bar. Where the answer is "the
machine", the run length is usually the thing that drifted from the claim, not the tolerance.

## 22. NumPy does not use the platform libm — so an exact claim can be a claim about a CPU

The run after §21's repair went red again, with **eighteen** failures where the previous run of
effectively the same code had one. Nothing in the environment moved: same runner image
(`ubuntu24/20260823.283`), same NumPy 2.5.2, same SciPy 1.18.1, same commit but for a step count.
The only variable left is the hardware, and the failures name it precisely.

### 22.1 The finding

**NumPy does not call the platform C library for the transcendentals.** It carries its own
vectorised routines, selected at import from the CPU's feature set. Rust's `f64::powf`/`tan` call
libm. So every "bit-identical" claim whose quantity passes through one of those is two
*implementations* agreeing, and whether they do is a property of the processor GitHub happened to
hand the job.

The correlation is exact and is what makes this a diagnosis rather than a guess. Fifteen of the
eighteen were `collision`'s power primitives, and the pass/fail split follows §16.2's shortcut
ladder without a single exception:

| primitive | exponent | ladder exponents (passed) | off-ladder (failed) |
|---|---|---|---|
| `contact_potential` | `alpha + 1` | 1.0 | 1.5, 2.0, 2.3, 3.0 |
| `contact_force_elastic` | `alpha` | 1.0, 2.0 | 1.5, 2.3, 3.0 |
| `contact_stiffness` | `alpha - 1` | 1.0, 1.5, 2.0, 3.0 | 2.3 |

Wherever NumPy spells the power as `sqrt`/`x*x`/`x`/`1`/`1/x`, both sides perform the same
multiply and agree. Wherever it hands the exponent to its own `pow`, they differ by one or two
ulp. The **scalar** path passed everywhere, because CPython's `float.__pow__` and Rust both reach
glibc — §20.2's Windows/Linux note, now confirmed from the other side.

The remaining three were `radiation`'s `impedance_discrete`, which is `np.tan` of a scalar. Its
sibling `impedance` — the same rational function at `s = j omega`, no transcendental — passed.
That pair isolates the cause on its own.

**This is the fourth mechanism by which one expression is two computations**, after NumPy's ufunc
ladder (§16.2), LLVM's constant fold (§17.2) and a hand-written hoist (§20.2). It is the first
that is not visible anywhere in the source of either language: nothing about `x ** 1.5` says which
library will evaluate it, and the answer is not even stable across two machines of the same
operating system, compiler and NumPy build.

### 22.2 A local repro was not available, and the reason is worth recording

The obvious discriminator — `NPY_DISABLE_CPU_FEATURES`, stripping the top SIMD tier to force a
different loop — **does nothing here**. On this Windows machine (NumPy 2.4.6, baseline `X86_V2`,
found `X86_V3`) the divergence is zero at every setting, because on Windows NumPy has no
dispatched `pow` loop to strip: both languages reach UCRT and always agree. §20.2 predicted
exactly this asymmetry and it is why the whole class is invisible to local development. The
diagnosis therefore had to be made from CI logs, and the failure/ladder correlation carried it.

The workflow now prints `/proc/cpuinfo`'s model name and `numpy.show_runtime()` before the parity
step, so the next occurrence is a comparison rather than an investigation.

The first reading came back on the run that made the tree green again (2026-08-27, run
`33090052206`), and it is worth keeping because it confirms the mechanism from the *passing* side.
The machine was an **AMD EPYC 9V74**, NumPy baseline `X86_V2`, found `X86_V3`, and every one of the
eight off-ladder power comparisons — the exact cases that failed two runs earlier — read

```
contact_potential at exponent 2.5: 0 of 20000 differ, worst 0.0 ulp on this math library
... 3.0, 3.3, 4.0, contact_force_elastic 1.5, 2.3, 3.0, contact_stiffness 1.3: all 0 of 20000
```

Same commit, same NumPy, same runner image, a different processor: **twenty thousand agreements
where the other machine had one or two ulp of disagreement.** So §22.1's diagnosis is no longer an
inference from a correlation, it is measured on both sides of the split. Two consequences. The
divergent CPU is the *minority* case, which is why this class went unnoticed for twenty-one
sections and why it will keep arriving as a surprise. And the ulp bars added in §22.3 are, on this
machine, passing at **zero** — they are not exercised here, so a future run whose printed counts
are non-zero is the signal that the job landed on the other kind of machine, not that anything
regressed.

### 22.3 The rule, and where exactness survives

The human's call: **keep exactness where it is provable, bound it where it is not.** Concretely —

* an exact claim is legitimate when both sides perform *the same arithmetic*, which for the power
  primitives means an exponent on the ladder. `PowPath::Array` in `collision.rs` spells `2.0` as
  `x * x`, so this is readable from the source rather than assumed;
* off the ladder the assertion becomes an ulp bound with the measured separation printed. Four ulp
  is the bar. A transcription error is an O(1) relative difference, so nothing real is lost;
* where a portable spelling exists at no cost, prefer it to either. `impedance_discrete` moved
  from `np.tan` to `math.tan` — `omega` is a scalar, there was no vectorisation to give up, and
  CPython and Rust reach the same libm on both platforms. Its assertion stays exact. **This is
  `portable.py`'s manoeuvre (§18.2) a fourth time**, and the first applied to a transcendental
  rather than to a summation order.

That last change carries a second one that is easy to miss and is worth stating: `2j * np.tan(...)`
produced an `np.complex128`, so the division that follows was **NumPy's** complex division;
`2j * math.tan(...)` produces a Python `complex`, so it is now **CPython's Smith algorithm** --
which is the one the Rust side transcribed, and which that test's own comment says the port was
built against. So the spelling moved the read-out onto the intended division as well as onto the
intended `tan`. Confirmed by type and by value: the method now returns
`0.9868 + 38.4603j` at the fixture's failing frequency, which is the value **Rust** produced on the
divergent runner.

Also widened in the same pass: the off-ladder allowance started at 4 ulp, which is 1.1x-2.2x the
one machine-dependent measurement that exists (4.0e-16 relative is 1.8-3.6 ulp depending on the
binade). That is §21.6's defect rebuilt. It is now **64**, and nothing is lost by it -- there is no
detection power between 4 and 64, because a transcription error is an O(1) relative difference and
therefore ~1e16 ulp. The printed count, not the bar, is what tracks drift; the parity step gained
`-rP` so those reports survive a green run instead of being captured and discarded.

`alpha = 1.0` is the only exponent that puts all three primitives' exponents — `alpha - 1`,
`alpha`, `alpha + 1` — on the ladder simultaneously, so it is where the contact leg's exact
trajectory claims now live. It is still a one-sided contact: contact-set detection, the Newton
solve, the dense LU and the discrete gradient's 0/0 branch are all exercised.

### 22.4 The blind spot is a property of the exponent, not only of the stiffness

Moving §16.4's soft-rail test to `alpha = 1.0` for exactness **failed locally on Windows**, where
it had been bit-identical at 1.5 for two thousand steps. That is not a regression; it is §16.4
arriving from the other side. Measured on the fixture's own Jacobian:

| `alpha` | `cond(J)` | `max abs(J - I)` |
|---|---|---|
| 1.0 | 1.0625 | 5.76e-02 |
| 1.5 | 1.0032 | 2.96e-03 |
| 2.3 | 1.0001 | 6.62e-05 |

The tangent stiffness is `K a eta^(a-1)`, which **vanishes at grazing contact for `a > 1` and is
the flat constant `K` at `a = 1`**. So a *higher* exponent hides the solver *better*, and the
linear law is the one case where this rail is not soft in the sense §16.4 cares about — the LU
reaches the answer and the run separates within 2,000 steps. The soft-rail test therefore keeps
1.5 and moves to Group A over 500 steps.

The general form: **§16.4 said the fixture the suite uses most can be the one that does not
exercise the thing being ported. This adds that the exercise level is set by a physical parameter,
so "pick a fixture that engages the solver" and "pick a fixture whose arithmetic is provable" can
be in direct conflict** — here they were, and the conflict is resolved per test rather than
globally.

### 22.5 A tolerance has to be read against the expression, not against the port

`deriv_total_vec` cannot be given a useful elementwise bound at an off-ladder exponent, and the
reason is nothing to do with the port. The discrete gradient divides by `da = eta_next - eta_prev`
and its derivative divides by `da^2` after a cancellation of the same order, so a last-bit
difference in a `pow` is amplified by roughly `1/da^2` — unbounded as `da` approaches the `tol`
that selects the Taylor branch. Injecting a one-ulp nudge into the powers at the rate CI observed
(2.4% of entries) moves the derivative by **15% of its own scale** on the existing fixture.

Calibration for that injection model: it predicts 1.3e-7 for `force_total_vec` at `alpha = 1.5`,
where the runner measured 1.8e-7. Close enough to trust the derivative figure too.

The floor on `|da|` buys agreement at exactly the `1/da^2` rate:

| `abs(da)` floor | worst force | worst derivative |
|---|---|---|
| 1e-5 | 1.2e-11 | 3.2e-06 |
| 1e-4 | 8.8e-13 | 6.3e-08 |
| 1e-3 | 6.1e-14 | 6.3e-10 |
| 1e-2 | 1.2e-14 | 1.3e-11 |

So the off-ladder comparison runs on a fixture with `abs(da) >= 1e-3` and the near-`tol` regime —
the one that actually tests the branch condition — is covered exactly at `alpha = 1.0`. **The
question to ask of a failing agreement bar is not "how far apart are they" but "how far apart
could the same formula put itself".**

### 22.6 What did not need changing, and why that is the useful half

`np.cos` was already known to be in this class — `exciter.rs`'s header says so, and
`test_the_raised_cosine_survives_a_sweep_of_the_transcendental` asserts exact equality with the
comment "a platform where they diverge by an ulp fails here with an obvious cause". That platform
arrived, and the raised cosine **passed on it**. So this is not "NumPy differs from libm for
transcendentals"; it is per-function and per-CPU, and `cos` is not currently exposed while `pow`
and `tan` are. The exposed surface across ported modules is small enough to list: `collision`'s
four powers with a caller-supplied exponent, `radiation`'s one `tan` (now fixed), `bow`'s `np.exp`
(bounded by §20.2 — the array scan only decides whether a bracket exists), `exciter`'s two
`np.cos`, and `operators2d`'s `sin`/`cos` in the guitar outline, where a last-bit difference is
not an ulp but a **live/dead node**. The last two are asserted exactly and currently agree.

**The rule for Phase 4 and beyond:** before writing an exact-equality assertion, ask which library
computes each value on each side. Where the answer is "NumPy's own, chosen by CPU", the assertion
is a claim about a runner and belongs in an ulp bound — or, better, the Python side should be
spelled so both languages reach the same library.

---

## 23. Phase 3, batch 6, as built (2026-08-27) — **Phase 3 is finished**

`physsynth/core/collision.BarrierString` — a damped stiff string vibrating against a one-sided
distributed barrier, model #8. String–fret buzz, the sitar and tanpura *jawari*, the tanpura's
cotton thread, prepared-piano rattle.

It is here, after `bow`, for a reason that is itself the migration's most-repeated lesson. §16
described it as waiting on its host `DampedStiffString`; §18 ported that host; nobody revisited the
sentence, and §19.11 and §20 both went on calling the bow the phase's last model. §20.11 caught it
by *checking* rather than inheriting. So this batch closes a gap that had been open for three
batches and was invisible in every one of them.

Almost nothing here is new machinery. The contact primitives, the vector solve and the dense LU
landed in §16; the string in §18; the banded solve in §15. What is new is the **shell**:
construction (broadcast the profile, pick the support, solve `m` admittance columns, form
`k**2/rho`), the two penetration gathers, the rank-`m` force injection, and the barrier's
two-time-averaged potential energy — plus one line of arithmetic and one dense matvec that nothing
had ever compared across the two languages.

Four things make the batch worth reading. The matvec turns out to be exact at two contact nodes for
a **reason** rather than by luck, and the reason is a general fact about short reductions (§23.2).
The one line of arithmetic is a spelling the *sister model does differently*, so "follow the bow"
would have been wrong (§23.3). §17.2's constant-fold arrived for a **third** time, inside the test
written to catch it, and made that test empty in `--release` only (§23.4). And the batch had to
**rewrite** an existing parity section rather than extend it, because porting a class silently
emptied it (§23.6).

### 23.1 The shape on disk

```
physsynth/core/collision.py                        module note; the BarrierStringPy alias; swap block
crates/physsynth-core/src/collision.rs             BarrierParams/BarrierState/penetration_of/apply/
                                                   barrier_energy/BarrierString; header rewritten
crates/physsynth-core/tests/collision_barrier.rs   NEW — native bars (12 tests)
crates/physsynth-py/src/collision.rs               PyBarrierString — the refusal, the settable
                                                   underscored getters, the warning
crates/physsynth-py/src/string_damped.rs           two crate-internal read accessors; `set_state`
                                                   made crate-visible; header corrected
crates/physsynth-py/src/lib.rs                     registration
tests/test_rust_parity_collision.py                sections 4 rewritten and 5 added (117 tests)
tests/test_stability.py                            `collision` added to the class-guard DERIVE tuple,
                                                   and `BarrierString` to its expectation
.github/workflows/ci.yml                           the batch's flagged step; batch 2's stale sentence
```

### 23.2 The finding: a two-term reduction's error is correlated with its own smallness

The shell injects the contact force through

```text
u[1:-1] += force_pref * (cols_mat @ f)
```

which is a dense BLAS matvec on the **update path**. §16 never compared it across the languages,
because the shell was Python on both sides of every comparison it made. Porting the shell makes it a
cross-language reduction, and §14.2 says that is where bit-identity ends.

Measured 2026-08-27 against the left-to-right row sum the port writes, over the parity file's
fixtures and **6,000** steps each (the 2,000-step figures the tests themselves run are in the last
column):

| fixture | rows compared | matvec differs | survives `* force_pref` | reaches `u` | reaches `u` by step 2,000 |
|---|---|---|---|---|---|
| point fret, `m = 1`, `a = 1.0` | 474,000 | **0** | 0 | 0 | 0 |
| point fret, `m = 1`, `a = 1.5` | 474,000 | **0** | 0 | 0 | 0 |
| two frets, `m = 2`, `a = 1.0` | 474,000 | 2,232 | 2,152 | **0** | 0 |
| two frets, `m = 2`, `a = 1.5` | 474,000 | 3,719 | 3,625 | **0** | 0 |
| flat rail, `m = 79`, `a = 1.5` | 474,000 | 45,822 | 44,653 | **30** | 7 |

`m = 1` is not interesting — the sum is one product, and §16 already named that the cause-separator.
`m = 2` is: the matvec **does** differ, thousands of times, and the trajectory is nevertheless
bit-identical over 6,000 steps.

The first explanation drafted for that was wrong, and the way it was wrong is the useful part. It
said the correction is a small fraction of the field it is added to (at most `5.7e-3` of `u`), so
one of its ulps sits below `u`'s last bit and rounds away. That is a *magnitude* argument, and the
79-node row refutes it: there the ratio is **smaller** (`2.1e-3`) and differences reach the state
anyway.

The real mechanism is about the **length of the sum**. Two doubles summed in either order give the
same result unless the addition **cancels** — and where it cancels, the sum is small. So a two-term
reduction cannot disagree without also being tiny. Measured on the same run, restricted to the rows
where the two spellings actually differ:

| fixture (2,000 steps) | rows differing | median `|correction| / |u|` there | max |
|---|---|---|---|
| two frets, `m = 2`, `a = 1.5` | 1,291 | **2.5e-18** | **9.3e-13** |
| two frets, `m = 2`, `a = 1.0` | 764 | — | **7.0e-12** |
| flat rail, `m = 79`, `a = 1.5` | 14,746 | 1.2e-4 | 2.1e-3 |

At two nodes the correction is at worst `7e-12` of `u` *at exactly the rows where it is wrong* (and
`9.3e-13` at the shipped exponent), so one of its ulps is worth about `1e-11` of one of `u`'s and
cannot survive the addition — which is why the parity test's bar on that ratio is `1e-9`, three
orders above the worst reading and eleven below where it would start to matter. At 79 terms
the correlation is gone — a long sum reorders without the result being small — the correction is an
ordinary size where it differs, and roughly one difference in `1/1.2e-4` crosses a rounding
boundary. That predicts about 36 hits in 44,653; 30 were observed.

So: **`m = 1` is exact because the sum has no additions; `m = 2` is exact because a two-term sum's
error is correlated with its own smallness; `m = 79` is not exact, and the model does not say it
is.** That is §16's cause-separator generalised one term along, with a reason attached rather than
an observation recorded — and the reason is what makes it safe to keep an exact assertion there.

**Which half of that is proved and which is measured, because the distinction decides how to read a
red bar.** *Proved:* two doubles sum identically in either order unless the addition cancels, so a
two-term reduction cannot disagree without its result being small. That does not depend on the
fixture. *Measured:* that `force_pref` times such a cancelled sum stays below `u`'s last bit — the
`7e-12` above — which is a property of this model's coupling strength at these fixtures, and the
parity test asserts it separately (`< 1e-9`) rather than folding it into the exact claim. A fixture
with a much stiffer contact or a much deeper rail could move it; the round-trip test in §23.9 walks
straight into that regime on purpose, doubles the coupling, and is a Group A claim for exactly this
reason. So if the exact `m = 2` assertion ever goes red, the first question is which of the two
halves moved — and if it is the measured half, the fix is to downgrade that assertion with the count
printed (§22.3's rule), not to "repair" a port that is faithful.
Both halves are asserted: one test pins that the matvec differs and that the correction is tiny
where it does, and a second pins the 79-node case as the control that shows where the argument
stops applying. An exact test with no control next to it would read as "the two matvecs agree",
which is false.

One deliberate restraint in how those two are written: the **ratio** is the bar and the **count of
differences that reach `u`** is printed, not asserted. That count is a handful of rounding-boundary
crossings out of tens of thousands of chances, and which kernel OpenBLAS picks for a `dgemv` is a
property of the CPU (§14.2) — so a bar on it would be a small integer partly decided by the runner,
which is §21.6's failure. Both tests also **skip** rather than fail if a machine's BLAS turns out to
sum these matvecs left to right, because on such a machine there is simply nothing to demonstrate.

The general rule for the phases ahead: **before asserting bit-identity across a ported reduction,
ask how many terms it has.** Two is qualitatively different from many, and the difference is
provable rather than empirical.

### 23.3 `portable.py` was considered and rejected, on evidence

§18.2's manoeuvre — move the *Python* side to a spelling both languages can express — was the
obvious response to §23.2's matvec, and `portable.py` already holds a `dot` for exactly this class
of problem. It was rejected, and the reasoning is recorded because the same question will arrive at
Phase 5 with the plate family:

* the exactness it would buy at `m = 1` and `m = 2` is **already there**, for the structural reason
  above;
* the exactness it would buy at `m = 79` is **not available at any price** — the *solve's* own
  `G @ F` matvec already spends it (§16), and the shell's contribution is not even the binding
  constraint. Measured: the shell alone contributes at most `1.9e-14` of peak at 500 steps against
  a `1e-13` bar, and first crosses that bar between steps 1,597 and 3,076 by fixture, where the
  solve's own window is 1,175–1,584;
* and it would change a **shipped model's reference numbers** unconditionally — the barrier's, and
  with it the viewer's fret, jawari and juari output — which `portable.py` has done twice before
  and should only do when something is bought.

The rule that generalises: `portable.py` is for a reduction whose order is *the only thing*
standing between two implementations. Where a coarser divergence is already present downstream,
moving the Python side buys nothing and costs a change to numbers people have looked at.

### 23.4 The one line of arithmetic, and why "follow the bow" would have got it wrong

```python
self._force_pref = string.k ** 2 / string.rho      # collision.py
self._force_pref = self.k * self.k / (string.rho * string.h)   # bow.py
```

Both are the rank-`m`/rank-1 force-injection prefactor of a model that holds a `DampedStiffString`
and corrects it once per step. They are written differently, and `**` is `float.__pow__`, i.e. the C
library's `pow` — a *different double* from the multiply for a small fraction of arguments. So the
Rust barrier uses `scalar_pow` (§17.2's `#[inline(never)]` wrapper) where the Rust bow correctly
uses `k * k`, and a port that had copied its sister model would have put a last-bit error on the
state of every step of every run.

Nothing in either suite could have caught it before this batch: the two models are never compared to
each other, and an energy bar is blind to a one-ulp prefactor. Both suites now pin it.

### 23.5 The witness could not be hardcoded, and that is §22.1 arriving before it bit

The obvious pin for §23.4 is "at this sample rate the two spellings differ; assert the model took
the `pow`". §22.1 says that is a bar decided by the runner: *which* arguments `pow` rounds
differently from a multiply is a property of the C library the machine links, so a witness measured
here would be a value that a different CI box is free to disagree about — §21.6's failure exactly.

Both suites therefore **search** for a witness at runtime, over a deterministic sweep, and say so
out loud if the machine has none. That makes the test bite wherever the distinction exists and stay
quiet where it does not, without either outcome being a red bar about the wrong thing.

Two smaller scars from getting that search working, both worth keeping:

* **The predicate has to be the whole expression.** The first witness a `k ** 2 != k * k` sweep
  returned had its difference **absorbed by the `/ rho`** that follows it, so the negative control
  compared a value against itself and failed. What is being pinned is `force_pref`, so the sweep
  has to test `force_pref`.
* **§17.2, for the third time, inside the search itself.** Written the obvious way — `k.powf(2.0)`
  — the Rust sweep passed in `--debug` and found **nothing** in `--release`: LLVM folds a literal
  `2.0` exponent into `x * x`, so the search's own predicate became `x * x != x * x` and the test
  reported a green tick having asserted nothing. Routing it through `scalar_pow` fixes it, and that
  is also the function under test. §17.2 found this in a test, §16.8 found the same *shape* in an
  empty CI job, and this is its third door: **a test written to catch a compiler rewrite is itself
  subject to that rewrite.** Native tests that pin an arithmetic spelling must be run in both
  profiles, and this one now is.

### 23.6 Porting a class silently emptied an existing parity section

§16 measured the vector solve *through* the Python `BarrierString`, by pinning
`collision.solve_contact_vector` to each implementation in turn. That is the right rig — and it
stops working the moment the class itself swaps, because with `PHYSSYNTH_RS=1` the Rust model never
looks the name up. Every test in that section would have compared Rust against Rust and passed.

Verified rather than assumed: with the flag set, a pin that raises on call is simply never called.
Section 4 now builds `BarrierStringPy` explicitly, exactly as `test_rust_parity_mallet.py` learned
to at §17, and section 5 is the new comparison of the shell.

This is a **class** of hazard rather than an incident, and it is the same one as §17.6's empty guard
and §16.8's empty CI job seen from a third angle: a comparison whose two sides are selected by a
*module-level name* stops being a comparison when that name is rebound. The rule for the phases
ahead: **when a model class ports, grep the parity suite for anything that pins one of its
collaborators by name and check that the pin still reaches it.**

### 23.7 The class guard was one name short, and had been for six batches

`test_stability.py`'s class half derives the swapped set from the `<Name>Py` aliases the modules
define — §17.6's fix. But the derive runs over a **hand-written tuple of modules**, and `collision`
was not in it. So from §16 onwards a `BarrierStringPy` alias could have appeared with nothing
noticing, in the very half of the guard §17.6's own text says it was applying the lesson to.

Checked the way §17.6 says to: `collision` was added to the tuple *first*, the guard was run and
observed to **fail** (`Extra items in the left set: ('physsynth.core.collision', 'BarrierString')`),
and only then was the expectation updated. A guard that is not seen to fail is a guard that has not
been shown to be live.

The residue: **a derive is only as wide as the list it derives over**, and that list is the one part
of it still written by hand.

### 23.8 The warning had nowhere left to be raised from

`solve_contact_vector` warns when the Newton cap is hit without convergence, and §16's swap block
deliberately kept that warning **on the Python side**, because `stacklevel=2` names
`BarrierString.step` and cannot mean the same thing from inside an extension module.

Once the model itself is Rust, that frame does not exist. Nothing in the repo asserts the
attribution — `tests/test_collision_energy.py` and `tests/test_jawari.py` both check
`newton_iters < newton_maxiter` instead — so the warning would have been lost silently rather than
loudly. The decision, taken rather than discovered: the Rust `step()` raises it with `stacklevel=1`,
naming the Python code that called `step()`, which is the nearest true statement about who to blame.
The message is byte-for-byte the original's, and a new parity test drives both implementations into
a stall and compares the text.

One implementation detail is load-bearing and is commented where it lives: the warning is raised
**after** the string's `borrow_mut` is dropped. A `UserWarning` can be promoted to an exception by
the caller's filters, and unwinding through a live PyO3 borrow is a panic rather than a Python
error.

### 23.9 The underscored half of the interface, and the stronger reading of §12.2

§12.2's rule is that a leading underscore is not a statement about the interface. The barrier forces
the stronger version: four of its underscored attributes are not merely *read* by clients but
**assigned**.

| attribute | written by | why |
|---|---|---|
| `_G` | `test_collision_modal.py`, `test_jawari.py` | doubling the coupling is the negative control that gives the magnitude gate teeth |
| `_force_pref` | the same two files | the other half of the same double |
| `_b` | `test_jawari.py` | flattening a curved bridge to a rail at its own crest — the contrast that says the wrap edge *travels* |
| `penetration` | `test_collision_modal.py`, `test_jawari.py` | seating the model at a static equilibrium by hand |

A binding with getters only would have looked complete and left three physics tests unable to run —
and one of them (`_b`) was missed on the first pass and found by running the suite, not by reading
it. `_b`'s setter deliberately does **not** rebuild `G` or the admittance columns, because the
original does not either and is right not to: the support is chosen by which heights are *finite*,
and a rewrite that keeps them finite leaves the string's admittance untouched.

### 23.10 What is bit-identical

Under a shared banded solver (`shared_solver()`, or `PHYSSYNTH_RS=1`, which arranges it anyway):

* **construction** — `G`, the admittance columns, `_b`, `_support`, `_int_idx` and `force_pref` —
  exact on every fixture. This is the cause-separator for everything below: a trajectory that
  separated on a fixture whose `G` already differed would be reporting the string, not the shell;
* **`m = 1`** (point fret) — the field, `energy()` and the Newton iteration count, exact over 2,000
  steps;
* **`m = 2`** (two frets) — the field and `energy()`, exact over 2,000 steps, for §23.2's reason;
* **`m = 79`** (flat rail, all four variants) — Group A, `<= 1e-13` of amplitude over 500 steps,
  with the iteration count compared step for step and identical.

Without a shared solver the two strings differ in the last bit from step one for §15.3's reason,
which this batch did not introduce.

### 23.11 The success condition

* `tests/test_collision_energy.py`, `tests/test_collision_modal.py`,
  `tests/test_collision_signature.py` and `tests/test_jawari.py` unmodified against Rust — the
  model's own bars, including the static-equilibrium magnitude oracle and its doubled-coupling
  negative control, both of which run *through* the settable underscored attributes of §23.9.
* `tests/test_web_backend.py` against Rust — `web/serialize.py` builds a `BarrierString` for
  **three** of its models (jawari, juari, fret), so the viewer is this port's largest client (§1.1).
* `tests/test_damped_string.py` against Rust — the string underneath, and the control that says a
  failure is about this batch rather than about §18's.
* `tests/test_stability.py`'s swap guard, with `collision` added to the derive tuple (§23.7) and
  `BarrierString` to its expectation.
* `tests/test_rust_parity_collision.py` — 117 tests, green with and without the flag, section 4
  rewritten (§23.6) and section 5 new.
* `crates/physsynth-core/tests/collision_barrier.rs` — 12 native bars, in **both** profiles (§23.5).

### 23.12 What was measured

| | |
|---|---|
| Native `cargo test --workspace` | **251 passed** (239 at the end of §20) |
| The same in `--release` | **251 passed** |
| Cargo dependency allowlist | still **EMPTY** |
| `tests/test_rust_parity_collision.py` | **117 passed**, flag set and unset |
| The barrier's four own files against Rust | **31 passed** |
| `tests/test_web_backend.py` against Rust | **408 passed** |
| All fourteen parity files | **1,269 passed** flagged; 1,268 + 1 skipped unflagged |
| Injection matvec, `m = 1`, 6,000 steps | **0 of 474,000 rows** differ, both exponents |
| Injection matvec, `m = 2`, 6,000 steps | 2,232–3,719 rows differ; **0** reach `u` |
| Injection matvec, `m = 79`, 6,000 steps | 45,822 rows differ; **30** reach `u` (7 by step 2,000) |
| `\|correction\| / \|u\|` where the matvec differs, `m = 2` | max **9.3e-13** (`a = 1.5`), **7.0e-12** (`a = 1.0`) |
| The same at `m = 79` | median **1.2e-4**, max 2.1e-3 |
| Shell's own contribution at 500 steps, `m = 79` | `<= 1.9e-14` of peak (bar: 1e-13) |
| Shell's own first crossing of 1e-13 | steps **1,597–3,076** by fixture (solve's own: 1,175–1,584) |
| `k ** 2` vs `k * k` over sample rates in range | differ in **86 of 200,000** |
| Bit-identity, shared solver, `m = 1` and `m = 2` | **exact** in field, `energy()` and Newton count |
| Group A, shared solver, six fixtures × 500 steps | `<= 1e-13` of amplitude; iteration counts identical |
| Speed, point fret (`m = 1`, `N = 80`) | **47.6x** (149.7 µs/step → 3.1 µs) |
| Speed, flat rail (`m = 79`, `N = 80`) | **5.3x** (274.4 µs/step → 52.0 µs) |
| Speed, flat rail (`m = 199`, `N = 200`) | **2.6x** (1,315 µs/step → 514 µs) |
| The two dense LUs, head to head, `m = 79` / `m = 199` | LAPACK 42.9 / 580.0 µs, port 46.7 / 617.9 µs |
| The barrier's four own files, flag on vs off | **14.1 s** vs **55.0 s** |

The `m = 1` figure is the largest speedup the migration has measured, and the `m = 199` figure the
smallest for a model whose Python side has no compiled inner kernel — both for the same reason, and
it is §11.6's rather than anything new. At one contact node the step is a Newton loop over
**one-element NumPy arrays**: there is no arithmetic to speak of, so the whole cost is per-call
overhead and essentially all of it goes away. As `m` grows the dense factorization comes to dominate,
and the last row above is the check that this is the ordinary story rather than a weak transcription:
the port's reference `dgetrf` is within **9%** of LAPACK's blocked one at both sizes, so neither side
wins the arithmetic and the ratio falls toward 1 as the arithmetic's share rises.

One consequence for the real-time port, which is what §11.6 says these numbers are actually about:
the *cheap* configurations gain most. A fret, a jawari bridge and a tanpura thread are all small-`m`
contacts — 1 to ~15 nodes — so they sit at the top of that table, not the bottom.

### 23.13 What Phase 4 inherits

* **Phase 3 is finished.** Group B and Group C are ported: `banded`, the four theta-scheme strings'
  solver, `string_stiff`, `string_damped`, `string_nonlinear`, `collision` in full including
  `BarrierString`, and `bow`. `string_geometric` is the one string that remains, and it is a
  **Phase 5** model rather than a Phase 3 leftover — it needs a sparse LU as well as the banded
  Cholesky, so it needs Group D.
* **`beam` is next, and its job is unchanged.** §4.1's SuperLU hypothesis is still untested; Group D
  is the only solver class the migration has not touched; `beam` is 254 lines with one `splu` and
  was chosen as the de-risker for exactly that. Nothing in Phases 2 or 3 moved it.
* **Ask how many terms a reduction has before asserting bit-identity across it.** §23.2. Two is
  provably safe; many is not; one is trivial. This is the sharpest tool the migration has for
  deciding in advance which claims are available, and it costs no measurement.
* **When a class ports, re-read the parity tests that pin its collaborators by name.** §23.6. The
  pin does not fail — it stops reaching anything, and the tests go on passing.
* **A derive is only as wide as the list it derives over.** §23.7.
* **Check inherited sentences about what is blocked on what.** §20.11 caught one that had been
  false for three batches; §23 and the CI comment corrected two more written by earlier batches.
  The general form (§18.3) is that a statement justified by a dependency expires when the
  dependency lands, and nobody is notified.
