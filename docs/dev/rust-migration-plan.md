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

> **Folded into Phase 5, 2026-08-31.** The room went as Phase 5's batches 6–9 (§30–§33) and the
> bridges as batch 10 (§34), so there was never a Phase 6 as such; `physsynth/core/` finished at
> §34. What is left of this list — `analysis/`, the viewer backend, the test-suite port and the
> deletions — is re-planned in **§35**, which supersedes the two entries below on order and
> scope.

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

---

## 24. Phase 4, as built (2026-08-28) — **`beam`, and §4.1's hypothesis answered**

`physsynth/core/beam.FreeBeam` — the free-free Euler–Bernoulli beam, model #5b-pre, and the
smallest member of Group D at 254 lines with one `splu`.

This batch was sent to answer a question rather than to add a model. §4.1 proposed that Group D's
risk could be collapsed by **linking SuperLU itself**, so that the six sparse-LU models could be
held to the same bit-identity as Groups A–C, and it named `beam` as the smallest surface on which
to find out. Six batches carried the hypothesis forward untested. It is now tested, and it fails —
for a reason that is not on §4.1's list, and after one of that list's three items turned out to be
real in a way the batch's own first two fixtures could not see.

The human's call (2026-08-28), with the measurements below in front of them: **Group D runs on
tolerance-level agreement, quantified**, which is the fallback §4.1 had already named as
survivable. No C dependency is added, and `physsynth-core`'s `[dependencies]` list stays empty for
the fifth phase running.

### 24.1 The shape on disk

```
crates/physsynth-core/src/sparse_lu.rs      NEW — Gilbert–Peierls left-looking sparse LU, natural
                                            order, diagonal-preferring pivot; the Group D solver
crates/physsynth-core/src/beam.rs           NEW — Params/initial_previous/step_rhs/step_into/
                                            energy/FreeBeam
crates/physsynth-core/tests/sparse_lu.rs    NEW — native solver bars (11 tests)
crates/physsynth-core/src/lib.rs            two registrations
crates/physsynth-py/src/beam.rs             NEW — PyFreeBeam; `K` and `W` as built-once csr_matrix
crates/physsynth-py/src/sparse_lu.rs        NEW — PySparseLu, so the PYTHON beam can be driven
                                            through the Rust factorization
crates/physsynth-py/src/{lib,string_stiff,string_damped,string_nonlinear,bore}.rs
                                            the `Option<Option<_>>` boundary fix (§24.7)
physsynth/core/beam.py                      module note; the FreeBeamPy alias; swap block
tests/helpers.py                            `arpack_v0`, and the five oracle eigsh sites pinned
tests/test_*.py (9 files)                   the other fifteen eigsh sites pinned
tests/test_stability.py                     the two ARPACK guards; `beam` added to BOTH swap-guard
                                            tuples and `FreeBeam` to the class expectation
tests/test_rust_parity_beam.py              NEW — the batch's comparison (53 tests)
tests/test_rust_parity.py                   the boundary=None guard, over all six classes
tests/test_ci_workflow.py                   the swallowed-continuation guard (§24.8)
.github/workflows/ci.yml                    the batch's flagged step; three pre-existing joined
                                            `run:` lines repaired
```

### 24.2 The finding: §4.1 named three obstacles, two are non-issues, one is real, and a fourth decides it

§4.1 said matching SciPy means matching "its column-ordering choice (`permc_spec`), its
`diag_pivot_thresh`, and its equilibration defaults — none of which have been checked yet."
Measured on the beam's own `A = (1 + σk)W + θk²κ²K`:

**The column ordering is a closed form in `n`.** COLAMD on this pentadiagonal family returns the
identity except that the two pairs at `n-5, n-4` and `n-3, n-2` are exchanged — verified over
seventeen grid sizes from `N = 4` to `N = 200`, without exception. It could have been hardcoded and
asserted. It was not reproduced, because once the hypothesis fails there is nothing left for it to
buy: COLAMD exists to reduce fill, and in the natural order this matrix has none.

**Equilibration never runs.** `Equil=True` and `Equil=False` produce bit-identical factors, in both
directions. The reason is structural rather than a coincidence of defaults: SciPy's `splu` calls
`_superlu.gstrf`, the *factorization*, and equilibration lives in the `gssvx` **driver**, which
SciPy does not call. So there is no hidden row/column scaling to reproduce, for `beam` or for any
other Group D model.

**The pivot threshold is real, and it is real only above a grid size the first two fixtures did not
reach.** At `N = 8` and `N = 32` — the two sizes the first probe used — `perm_r == perm_c` and
`diag_pivot_thresh=0.0` reproduces the default exactly, which reads as "pivoting is moot here".
It is not. The stiffness term grows like `h⁻³` against the mass's `h`, so the matrix stops being
diagonally largest as the grid refines: SuperLU takes the diagonal up to `N = 48` and starts
swapping rows at `N = 64`, and the fill follows — `U` holds 773 entries at `N = 200` where the band
is 600. **This is §16.4's blind fixture, arriving in the measurement rather than in the model**, and
it was caught only because the parity test parametrised over `N` and the second fixture went past
the transition.

**What actually decides it is a fourth thing, and it is invisible in SciPy's Python.** SuperLU is
**supernodal**. `relax` and `panel_size` visibly change the factors; and — the measurement that
settles the question — handed SuperLU's *own* `L` and `U`, a longhand column-oriented triangular
solve still disagrees with `lu.solve` in **about 20 % of entries at ~4e-16**, at every size tried.
Reproducing that means reproducing supernode and panel blocking, which depends on how SciPy
**built** its copy: the `relax`/`panel_size` defaults compiled in, whether an external BLAS was
linked, the vendored patch level. That is **§22.1's shape one layer down** — a claim about a build
rather than about a library — and it is why linking was declined rather than attempted.

Four spelling variants were tried against SuperLU's factors before the supernodal cause was
identified (right-looking vs left-looking; divide vs multiply-by-reciprocal). The reciprocal
spelling is clearly closer, as §15.3 found for the banded factor, and none of the four is exact:
the residual disagreement moves between 0 and 6 entries as `N` and `relax` change, without
converging. **A recipe that is exact at two sizes out of four is not a recipe**; §15.3's "no scalar
recipe" for `DTBSV`, one solver class down.

### 24.3 The solver, and the one place it parts company with the reference on purpose

`crates/physsynth-core/src/sparse_lu.rs` is a left-looking Gilbert–Peierls sparse LU: each column's
nonzero pattern is the set of nodes reachable from `A(:,k)` in the graph of `L`, found by an
iterative depth-first search, and the factorization costs the nonzeros of the factors rather than
`n²`. The DFS is iterative rather than recursive so that Phase 6's room cannot overflow the stack.
Natural column order; no fill-reducing ordering in front of it, for the reason above.

The pivot rule is where it deliberately differs. SciPy's default `diag_pivot_thresh` is `1.0`
(strict partial pivoting); this module's `DIAG_PIVOT_THRESH` is `0.1`, i.e. take the diagonal
whenever it is at least a tenth of the largest candidate. That is legal here for a reason SuperLU
has no way to know: **every Group D matrix in this project is symmetric positive definite** — all
of them are `(mass) + (positive coefficient)·(PSD stiffness)` — and on an SPD matrix elimination
with no pivoting at all is unconditionally stable, which is exactly what makes Cholesky a valid
algorithm. The margin was measured rather than assumed: the beam's diagonal stays the chosen pivot
for any threshold below **~0.50** at every size from `N = 8` to `N = 200`, so `0.1` has a factor of
five and the margin stops shrinking.

`dense.rs` warns that a pivot choice is a different *elimination* rather than a different last bit,
and that warning is honoured here by making the divergence a stated and priced decision rather than
an accident — see §24.5. The payoff is that the Rust factorization has **zero fill at every size**
against the reference's growing `U`, and that a coarse-grid and a fine-grid beam are the same
computation on this side.

### 24.4 The manoeuvre that separates the port from the solver, and what it found

The whole comparison would otherwise be confounded: any transcription error would be buried under a
solver gap two orders of magnitude larger, which is §19.4's finding — a real bug that left the
trajectory bit-identical for 2,000 steps — waiting to happen. So `physsynth-py` exposes the Rust
factorization as a `SparseLu` class wearing `splu`'s interface, and `tests/test_rust_parity_beam.py`
patches `beam.splu` for the length of a block. That is `test_rust_parity_strings.py`'s
`shared_solver()` a second time; unlike there, it is **never a no-op**, because the Rust beam
factors internally under the flag as much as without it.

With the solver held constant the two beams are **bit-identical** — `u`, `u_prev` and the whole
history — over 2,000 steps at four fixtures spanning `θ ∈ {0.25, 0.28, 0.5, 1.0}`, `σ ∈ {0, 4}` and
`N ∈ {8, 32, 48}`. `energy()` is the single exception at ~3e-16 relative, which is the `np.dot`
reduction and nothing else. So the port is exact and **the entire divergence below is the solver**.

### 24.5 A sixth agreement regime, and the first one set by a boundary condition

`K`'s nullspace is exactly the rigid-body space `{1, x}` — that is what free-free *means*, and #5b
was built to have it. A per-step solver difference therefore has a component the scheme never
restores: along `{1, x}` the beam is a **free particle**, so a velocity error integrates once into a
displacement error and then again every step after. Splitting the difference in the `W` inner
product, at `N = 32`, `σ = 0`, normalised by the running peak (§20.6):

| steps | total | rigid | elastic | ΔE/E |
| --- | --- | --- | --- | --- |
| 1 | 2.2e-16 | 3.7e-17 | 1.9e-16 | 4.8e-15 |
| 100 | 8.2e-14 | 8.8e-14 | 3.4e-14 | 1.7e-13 |
| 1,000 | 6.8e-12 | 6.9e-12 | 6.0e-14 | 3.0e-13 |
| 5,000 | 2.0e-10 | 2.0e-10 | 7.9e-13 | 8.4e-14 |
| 20,000 | 3.3e-09 | 3.3e-09 | 1.4e-12 | 5.7e-13 |

The rigid part is `t²` and swamps everything past about a hundred steps. The elastic part is not:
it grows by 40 while the step count grows by 200, which is §18.6's random walk — what a *linear*
scheme does with a perturbation it can restore. **Damping attenuates the growth without removing
it**: 3.3e-9 at `σ = 0` against 2.1e-10 at `σ = 100`, because a damped free particle's velocity
error decays but its accumulated displacement does not come back.

The five earlier regimes were set by nonlinearity (§16.5), by whether it recurs (§17.5), by
linearity (§18.6), by amplitude (§19.5) and by an attractor (§20.5). This one is set by a **boundary
condition**, and the rule it generates is inherited by every free-edge model in Phases 5 and 6:
**a parity bar on a model with a rigid-body nullspace reads the rigid/elastic split or the energy,
never `max|du|/amp`** — which will read as a failure that is not one.

Two things bound the damage. The energy — which is what the acceptance contract is written on —
stays inside **7.2e-12 relative** across every fixture measured, four orders inside CLAUDE.md's
1e-10 bar, and does not inherit the `t²`. And the price of §24.3's deliberate pivot disagreement is
visible but small: the rigid divergence at 5,000 steps grows about 20× between `N = 48` (below the
reference's pivot transition) and `N = 96` (above it), from 2.4e-10 to 9.3e-9, while the energies
agree to ~1e-12 at both. `test_the_divergence_grows_where_the_reference_starts_pivoting` asserts it,
so the cost cannot quietly stop being paid or quietly grow.

### 24.6 `portable.py` was re-taken for `beam` and declined again, on both counts

`portable.py`'s scope note names `beam` as out of scope "because no anchor binds it to anything",
and §19.2's rule is that such a decision expires the moment the model ports. Re-taken:

* **The matvec order** (§18.2's hazard — `K @ u` runs twice per step and SciPy's sparse product can
  leave descending column indices) buys nothing, because under the flag the Python beam gets its
  `K` from the *same* Rust `free_beam_stiffness` this module builds. Asserted rather than assumed:
  the parity file compares `K.indices` and `W.indices`, not only the data.
* **The energy reduction** buys nothing for §23.3's reason exactly — nothing exact is *available*
  downstream of a coarser divergence, and the trajectories differ at 1e-9 from the solver.

So this is the second time `portable.py` has been declined on evidence, and the two refusals have
different grounds: §23.3's was "exactness is already structural", this one is "exactness is not
available". Both are worth keeping, because they are the two halves of the question.

### 24.7 An omitted keyword and an explicit `None` were the same argument, in every binding

Found while writing the beam's boundary rejection test, and it was true of **all six** classes that
take a boundary: PyO3 maps a Python `None` and a *missing* argument onto the same Rust
`Option::None`, so a binding written the obvious way treats `boundary=None` as "not supplied" and
silently builds the default, where the Python original raises. No parity file had ever passed
`None` to a constructor, because `None` is not a plausible boundary and nobody thinks to try it —
§23.6's shape through a fourth door: a comparison that was never made rather than one that stopped
being made.

The fix is `Option<Option<_>>`, and **the arm order is the surprising half**: PyO3 wraps the
*default expression*, so `Some(None)` means "argument omitted" and a bare `None` is the caller's
literal. The first attempt had them the other way round and inverted the behaviour rather than
fixing it — it passed the beam's own test while breaking the default path — so both halves are now
pinned by `tests/test_rust_parity.py`. Four of the six now match Python exactly; `IdealString` and
`Bore` unpack their boundary (`left, right = boundary`), so Python's refusal is an incidental
`TypeError` from the unpacking rather than a designed message, and the Rust side raises its own
`ValueError` naming what it wanted. Both **refuse**, which is the property that mattered;
reproducing an accident was judged not worth a special case, and is recorded here rather than
silently left.

### 24.8 §19.7's escaping bug with the opposite sign, pre-existing on three steps

§19.7 found a `run:` continuation written as the two characters backslash-`n`; §20.7 saw it
reintroduced by the batch citing it and turned it into an assertion. The beam's step produced the
**other** variant: the same tooling round-trip that can leave a visible backslash-`n` can instead
consume *both* the backslash and the newline, leaving two shell lines **joined into one**. Three
earlier steps in `ci.yml` already had it — at 308, 550 and 852 characters — and none of them ever
failed, because `pytest a.py b.py` runs the same whether the names were on one line or four.

It is harmless only while what follows a continuation is another *argument*. A swallowed
continuation between two **commands** makes the second an argument of the first, and the step
silently stops doing half its job — invisible to the backslash-`n` test (there is no backslash left
to find), to a YAML parser (the block scalar is valid) and to the eye (the line runs off the
screen). `tests/test_ci_workflow.py` now caps a `run:` line at 120 characters, which separates the
two shapes without asserting anything about content, and the three joined lines are repaired.

### 24.9 The ARPACK chore §7 asked for, and it was wider and sharper than §7 said

§7 recorded that `beam_low_eigenfrequencies` "and its siblings" call `eigsh` with no fixed start
vector, called it four oracles, judged it harmless against a 5-cent bar, and said to fix it
**before** Phase 4. Both halves of that sentence needed correcting.

**It is twenty sites, not four** — five in `tests/helpers.py` and fifteen in nine test files. The
count was written early and nobody revisited it, which is §18.3's inherited-sentence failure again.

**The frequencies were the harmless half.** Measured on the beam at `N = 48`, four runs in one
process: the elastic eigenvalues wobble by ~1e-12 relative and their eigenvectors by ~5e-11, which
is indeed nothing against a 5-cent bar. But three sites return **eigenvectors** and feed them to
`set_state` — and an eigenvector is an *initial condition*, so an unpinned one is a different
trajectory, not a last digit. Worse, the two rigid-body modes are exactly degenerate at `μ ≈ 0`, so
ARPACK returns an arbitrary basis of the `{1, x}` nullspace and they came back **~1e-1 apart** run
to run. Against a beam whose two implementations differ at 1e-9, an oracle that moves by 1e-1 would
have made the parity numbers unreadable.

`helpers.arpack_v0` supplies a fixed start vector — uniform doubles from a seeded PCG64 stream,
which is deterministic across platforms and NumPy versions, generic (so it is orthogonal to no
eigenvector, unlike the obvious `arange`, which is antisymmetric about the centre of a symmetric
operator and would starve every symmetric mode), and free of any transcendental, whose
CPU-dispatched NumPy loop would put the *start vector* back in §22.1's class.
`tests/test_stability.py` asserts both that two oracle calls now agree to the bit and that no
`eigsh` in `tests/` lacks a `v0` — the second by walking the AST, because a guard that lists its
call sites is one paste away from covering nothing (§17.6, §23.7).

Deliberately not done: the six `eigsh` calls in `web/serialize.py`. They render pictures rather than
feed a comparison, and the viewer is Phase 8. The nondeterminism is real there — a free plate's
Chladni figure can pick a different basis of its nullspace between runs — and it is recorded here so
that Phase 8 does not have to rediscover it.

### 24.10 What is bit-identical

* **Structure, always:** every scalar parameter, `x`, `w`, and both `K` and `W` down to `indices`,
  `indptr`, `data` and `nnz`, at `N ∈ {4, 8, 32, 64}`. So is `u^{-1}` out of `set_state`, which
  involves no solve.
* **The trajectory, under a shared solver:** `u` and `u_prev` over 2,000 steps at four fixtures.
  This is the porting claim and it is the only exact claim about the *model* available.
* **`energy()`, never:** ~3e-16 relative under a shared solver (the `np.dot` reduction, §14.2), and
  up to 7.2e-12 with each side on its own factorization.
* **The trajectory, each side on its own solver:** not at any run length. See §24.5 for what
  replaces it.

### 24.11 The success condition

* `cargo test --workspace` green, debug **and** release — 11 new native solver bars, and the
  `sparse_lu` file asserts the no-fill property of the *shipped* beam operator rather than of a
  lookalike, because that property is what the natural-order decision rests on.
* `PHYSSYNTH_RS=1 pytest tests/test_beam_energy.py tests/test_beam_modal.py
  tests/test_beam_stability.py tests/test_stability.py` green — the **existing, unmodified** Python
  beam tests against the Rust model, including the closed-form `cos(βL)·cosh(βL) = 1` oracle run
  through a generalized `eigsh` on the Rust-built `K` and `W`.
* `pytest tests/test_rust_parity_beam.py` green with the flag and without it — 53 tests.
* The whole Python suite green on the default path, unchanged by the ARPACK pin.

### 24.12 What was measured

| quantity | value |
| --- | --- |
| COLAMD order vs the closed form | identical at 17 sizes, `N = 4 … 200` |
| `Equil=True` vs `Equil=False` factors | bit-identical, both directions |
| SuperLU pivots (`perm_r ≠ perm_c`) | never at `N ≤ 48`; always at `N ≥ 64` |
| reference `nnz(U)` at `N = 200` | 773, against a band of 600 |
| Rust `nnz(U)` at `N = 200` | 600 — no fill at any size |
| longhand solve vs `lu.solve`, SuperLU's own factors | ~20 % of entries differ, worst ~4.3e-16 |
| diagonal-pivot margin on the beam | threshold `0.50` at `N = 200`, vs the `0.1` used |
| Rust vs Python beam, shared solver | `array_equal` at 2,000 steps, 4 fixtures |
| … its `energy()` | ≤ 1e-14 relative |
| Rust vs Python beam, own solvers, `N = 32` | rigid 3.3e-9, elastic 1.4e-12 at 20,000 steps |
| … at `σ = 100` | rigid 2.1e-10 at 20,000 steps |
| … energy, worst over four fixtures | 7.2e-12 relative |
| rigid divergence at 5,000 steps, `N = 48` → `N = 96` | 2.4e-10 → 9.3e-9 |
| ARPACK eigenvector reproducibility, unpinned | elastic ~5e-11; rigid pair ~1e-1 |
| … pinned | `array_equal`, values and vectors |

### 24.13 What Phase 5 inherits

* **Group D's bar is tolerance-level and the shape of the tolerance is known.** Do not write an
  exact assertion across a `splu`; do write one with the solver held constant, which is now a
  two-line patch (`SparseLu` wears `splu`'s interface).
* **`sparse_lu.rs` is built and is the phase's solver.** `operators2d`, `plate`, `connection` and
  `string_geometric` need no new solver work. What they *will* need is `portable.canonical`, which
  §18.4 wrote down in advance — `plate.py` multiplies by a `biharmonic_matrix`-derived operator
  every step, and that decision was deferred to this phase with the measurement already in hand.
* **Ask whether the model has a rigid-body nullspace before choosing a parity bar.** §24.5. The free
  plate and the free beam have one; the supported plate does not; the room does not. It is the
  difference between a `t²` divergence and a random walk.
* **Ask how many terms a reduction has before asserting bit-identity across it.** §23.2, unchanged
  and still the sharpest advance tool the migration has.
* **A fixture that is coarse is a fixture that may be blind.** §24.2's pivot transition sits between
  `N = 48` and `N = 64`, and the two sizes first probed were both below it. Parametrise over the
  grid before concluding anything about a solver, and put one fixture past every threshold the
  model has.
* **Check inherited sentences.** §24.9's "four oracles" was twenty, and §24.2's third obstacle had
  been called moot on two fixtures. Both were written by earlier batches in good faith.

---

## 25. Phase 5, batch 1, as built (2026-08-28) — the guitar outline, and a batch boundary drawn by what the output *is*

Phase 5 is the rest of Group D except the room: `operators2d`'s remaining half, `plate`,
`connection` and `string_geometric`. §11.2.1 already said that half ports as a group rather than as
a file. This batch says something narrower and, it turns out, more useful: **that group splits
again, and the line is not size or difficulty but whether the function's output is a number or a
decision.**

Seven functions ported here — `guitar_half_width`, `guitar_scale`, `guitar_mask`, `guitar_area`,
`live_cells`, `cells_per_node`, `prune_to_area_carrying` — and they are exactly the functions whose
answer is a **set of nodes**. The matrices they serve (`biharmonic_from_mask`,
`orthotropic_biharmonic`, six private 1-D differences and `free_plate_stiffness*`) stay behind for
the next batch, with `VonKarmanBracket` and `AiryStressSolver`, together with `portable.canonical`, which §18.4 pre-registered for them.

### 25.1 The shape on disk

```
crates/physsynth-core/src/ops2d.rs          the outline: guitar_half_width / guitar_scale /
                                            guitar_mask / guitar_area, live_cells /
                                            cells_per_node / prune_to_area_carrying; header rewritten
crates/physsynth-core/tests/ops2d.rs        10 new native bars, including the closed-form area
                                            oracle and the left-association pin
crates/physsynth-py/src/ops2d.rs            seven bindings, three refusals reproduced verbatim
crates/physsynth-py/src/shape.rs            `to_shaped_f64`; and `as_f64_field` now reads the shape
                                            BEFORE `ascontiguousarray` (§25.7)
crates/physsynth-py/src/lib.rs              seven registrations
(crates/physsynth-core/src/sparse_lu.rs     four `needless_range_loop`s and seventeen formatting
                                            diffs, fixed in Phase 4's own commit — §25.8)
physsynth/core/operators2d.py               `_profile` / `_profile_vec`; `guitar_half_width` and
                                            `guitar_scale` moved to the scalar libm; the seven
                                            `_py` aliases and the swap block
tests/test_rust_parity_ops2d.py             the batch's comparison (174 tests -> 406)
tests/test_guitar_plate.py                  the pre-change spelling, written out once, and
                                            the shipped masks pinned against it (90 tests)
tests/test_stability.py                     seven names added to the guard's operators2d set, and
                                            `plate`'s captured bindings asserted for the first time
```

### 25.2 The finding: a discrete output is a different porting problem from a continuous one

Everything the migration has learned about last bits so far has been about *how far a difference
travels* — does the reduction reach the next timestep (§14.2), does anything branch on it (§19.2),
does the nonlinearity amplify it or contract it (§20.5). All of those questions presuppose that the
quantity is a number, and that the answer is a tolerance.

`guitar_mask` is not a number. It is `|x| < half(y)` evaluated at every node, and its output is
which nodes exist. A last bit does not make the plate slightly wrong; it makes it a **different
plate**, with one node more or fewer, and §22.6 already recorded that this is the sharpest instance
of NumPy's CPU-dispatched transcendentals in the whole repo. Every detector this project owns —
energy drift, the rigid-body nullspace, the spectrum, the area deficit — passes on a plate with one
node too few. That was measured on the disk two model-batches ago and written down as "the mask is
not the outline"; this is the same statement arriving on the porting side.

So the batch boundary was drawn there, and the reason is worth stating as a rule: **port the
discrete-output functions together, and settle their arithmetic before porting anything that
consumes them.** Had the geometry ridden along with `free_plate_stiffness`'s 500 lines of Kronecker
products, the one decision that actually needed making would have been one line in a large diff.

### 25.3 The measurement that decided it, and the outline it does not cover

The question is how much room the comparison has. Measured before a line of Rust was written, over
130 shipped configurations (two plate geometries × five outline parameter sets × thirteen grid
sizes), as `min | |x| − half |` in **ulps of `half`**:

| outline | smallest margin |
|---|---|
| `waist = 0.42, asym = 0.30` (shipped default) | 1.9e7 ulps |
| `waist = 0.97, asym = 0.30` | 9.9e7 ulps |
| `waist = 0.88, asym = 0.0` | 1.9e10 ulps |
| `waist = 0.60, asym = −0.30` | 5.5e9 ulps |
| median over all 130 cases | 8.6e11 ulps |
| **`waist = 0.0, asym = 0.0`** (the degenerate lens) | **1 ulp**, and 0 |

For every real guitar the exactness of the mask is **structural**: a hundred million ulps of slack
is not a coincidence that could go the other way on another CPU, and the exact assertion has a proof
behind it rather than a hope. That is §23.2's move — ask a cheap question about the *shape* of the
arithmetic before writing an exact assertion — applied to a comparison instead of to a sum.

The lens is the exception and it is exact in the other direction. With `waist = asym = 0` the
profile is `0.5·Lx·sin(πt)`, and the grid puts nodes at rational `t`:

* at `t = 1/2`, `sin` returns exactly `1.0`, so `half` is `0.5·Lx` and the bounding-box node is
  `0.5·Lx` — the same double. The strict `<` makes it dead, reproducibly, on any libm.
* at `t = 1/6`, `sin(π/6)` is `1/2` in real arithmetic and the `N = 32` grid puts a node exactly at
  `0.25·Lx`. In doubles it comes out one ulp low and the node is dead — **decided by the last bit of
  one `sin` call**, four times over (the ±x, t = 1/6, 5/6 quadruple).

This is reachable, not hypothetical: the viewer's waist sweep is `np.linspace(0.0, WAIST_MAX, n)`
and its first point is `0.0`.

### 25.4 §22.3's manoeuvre a fifth time — and the first one that costs something

§22.3 established the human's rule: exactness where it is provable, an ulp bound where it is not,
and *where a portable spelling is free, prefer it*. `portable.py` took that route three times for a
summation order and `impedance_discrete` took it a fourth time for `np.tan`. This is the fifth, and
the first where the parenthetical bites: the portable spelling is **not** free.

`math.sin` in CPython and `f64::sin` in Rust are the same call — the platform C library, UCRT on
Windows and glibc on Linux. `np.sin` on a float64 array is a third implementation chosen at import
from the CPU's features (§22.1). So the portable spelling is the scalar one, and taking it means
`guitar_half_width` evaluates its profile **one point at a time in a Python loop**. Measured:

| call | NumPy | scalar loop |
|---|---|---|
| `guitar_half_width`, 5,000 points (a mask row set) | 0.05 ms | 1.1 ms |
| `guitar_scale`, 20,001 points | 0.21 ms | 4.4 ms |
| `guitar_area`, **2,000,000** points | 45.9 ms | 467.4 ms |

The first two are nothing. The third is half a second on every guitar plate anyone builds, and the
viewer's waist sweep builds sixteen to thirty-two of them behind an endpoint that currently answers
in 1.4–2.7 s. So the spelling was taken for the mask path and refused for the quadrature, and
`operators2d.py` now carries both: `_profile` (scalar, portable) and `_profile_vec` (NumPy, fast).

**What makes that split safe is not a magnitude argument, it is §19.2's question.** `guitar_mask`
*branches* on the profile — the value decides whether a node exists — so it needs the spelling both
languages can express. `guitar_area` *averages* two million of them into a reported denominator, so
a last bit is invisible by construction. The rule generalises past this module: **give the portable
spelling to the consumer that branches, and the fast one to the consumer that averages.**

### 25.5 The deliberate duplicate, and why it is pinned twice rather than commented once

`_profile` and `_profile_vec` are the same formula written twice, four lines apart, and they differ
in the last bit — 221,580 of the profile values over the 130-case sweep. That is §20.2's hazard
exactly: a hand-written duplicate that *reads like something that wants tidying*, where tidying it
changes which computation runs. The bow's version was a hoist and this one is a spelling, but the
shape is the same and so is the remedy — a pin in the suite rather than a comment in the source:

* `guitar_half_width` is asserted **equal, to the bit,** to a list comprehension over `_profile`. A
  tidy-up back onto `np.sin` fails there, on any machine where the two disagree — which is exactly
  the set of machines where the tidy-up would matter.
* `guitar_area` is asserted **equal, to the bit,** to a quadrature spelled through `_profile_vec`. A
  tidy-up the other way — routing the quadrature through the public profile for tidiness — fails
  there, and would otherwise have cost half a second per plate silently.

Neither assertion is a claim about a CPU. Each says only "this call site uses this spelling", which
is a property of the code.

**And one more pin, which is a different kind and belongs in the default suite rather than the
parity file.** Both of the above compare the code against *itself*: they say which spelling each
call site uses, and every other test in the batch compares Rust against the **new** Python. That
leaves a hole exactly the width of a transcription slip — a mis-parenthesised `4.0 * pi * (t - 0.5)`
in `_profile` would be reproduced faithfully by the port and agreed on by every parity assertion,
and no physics bar can see a mask. So `tests/test_guitar_plate.py` now carries the **pre-change
NumPy expression written out verbatim**, and asserts the shipped masks — and their prunes — node for
node against it. That is the assertion that protects the geometry the plate family was validated on,
and it lives in the default suite because it is a claim about the *model*, not about the port.

**With one outline held out, and holding it out is the same reasoning as §25.3's.** That pin
compares two genuinely different `sin` implementations, so at the degenerate lens — where four nodes
sit one ulp from the rim — "the mask did not move" is a statement about which libm rounded, not
about this code. It is true on Windows and unverifiable on a Linux runner (§22.2). What the pin is
*for* is a transcription slip, and a slip moves the profile by orders of magnitude, so the four real
outlines catch it with 1.9e7 ulps to spare. The lens gets the strongest claim it can support
instead: the two spellings agree on the half-width **to within four ulps**, which a slip fails and a
rounding difference does not. The general form — **a pin must assert the weakest statement that
still catches what it is for**, or it becomes a machine claim wearing a correctness claim's error
message.

### 25.6 The one number not asserted exact, and the precedent it follows

`guitar_area` sums two million midpoints. `np.sum` is **pairwise** above a blocksize of 128; a Rust
`for` loop is not. Reproducing NumPy's blocking is possible — the algorithm is deterministic — and
it is refused for the reason §18.2 refused to reproduce SciPy's SMMP index order: it would pin the
port to a library internal that a point release is free to change, and here it would buy nothing,
because nothing branches on the answer and it reaches no timestep. Measured gap: **1.2e-13
relative**, worst over the sweep, against a `rel=1e-9` bar that the plate's own test already uses.

The native side gets a real oracle instead of a stored number. With `asym = 0` the profile
integrates in closed form —

    ∫₀¹ sin(πt)·[1 − w·cos(4π(t − ½))] dt = (2/π)·(1 + w/15)

— so the outline's area is `2·scale·L` times that, and the quadrature is checked against it at three
waists to 1e-12. That is the one bar here that would catch a wrong midpoint rule rather than a
transcription slip.

### 25.7 A rank-zero array is a shape, and `ascontiguousarray` disagrees

`guitar_half_width` returns the shape it was handed, and the original hands `np.asarray(t)` straight
through — so a bare float in gives a **0-d array** out. The binding returned `(1,)`.

The cause is one line of the shared 2-D reader, which has been there since Phase 2:
`np.ascontiguousarray` **promotes a 0-d array to shape `(1,)`**, and the shape was being read from
its result. Every previous caller demanded rank 2 and rejected anything else, so the promotion was
invisible for three phases; the first function vectorised over an arbitrary shape found it
immediately. The fix reads the shape from `asarray` and the values from the contiguous copy.

`as_f64_field` is shared by every 2-D binding — `disk_mask`, `embed`, `laplacian_from_mask` and the
membrane's `u0`, which is the one that *branches* on rank — so the change is not local. It is
verified rather than argued: the parity files for all of them, plus 524 flagged tests across the
membrane and the whole plate family, re-ran green after it.

Same family as §24.7's `boundary=None`: **an interface detail that no existing caller exercises is
not an interface detail that no caller will ever exercise**, and the way it surfaces is a new
function with a wider signature, not a new test on an old one.

### 25.8 Phase 4 had never been through the lint gate

Found at the start of this batch, on Phase 4's *committed-pending* tree: `cargo fmt --all --check`
was red in seventeen places and `cargo clippy --workspace --all-targets -- -D warnings` in four,
both of which CI runs before it runs a single test. Phase 4's own success condition (§24.11) lists
`cargo test` in both profiles and four pytest invocations, and does not mention either linter — so
the batch was checked against its own list and its list was one item short.

The residue is small but it is the fourth time in six batches that CI was red for something the
batch could have seen locally (§19.7, §20.7, §21, now this): **a success condition is only as good
as its coverage of the gate**, and the gate's first two steps are the ones a batch never thinks
about because they never fail while you are writing physics.

### 25.8a Phase 4 changed a behaviour and left the test that recorded the old one

The whole-suite run at the end of this batch came back **1 failed, 3464 passed**, and the failure
was not in anything this batch touched. `tests/test_rust_parity_bore.py` held a test called
*"an explicit None boundary is the one divergence and it cannot be closed"*, asserting that
`Bore(boundary=None)` raises in Python and quietly builds the default clarinet in Rust — the
behaviour §24.7 **fixed**, in all six bindings, one batch earlier.

Two things kept it green until now. The first is that it is the only place in the suite that passes
`None` to a *Bore*, so §24.7's new guard in `test_rust_parity.py` and this one are the only two
opinions on the subject and nothing compared them. The second is more mundane and is the real
lesson: **the installed extension was older than the source.** `PHYSSYNTH_RS` and the parity files
read whatever wheel is installed, and Phase 4's fix was in `crates/` without ever being in
`site-packages`, so the batch's own suite run — and this batch's, until the wheel was rebuilt at the
start of it — was measuring the code from before the fix.

So the rule has two halves, and the second is the one that generalises past this repo:

* **A batch that changes a shared behaviour has to re-run the tests that recorded the old one**, and
  a `grep` for the argument is how you find them. §24.13's "check inherited sentences" arriving as a
  red test rather than as a stale paragraph.
* **`pip install ./crates/physsynth-py` before believing any parity number.** Nothing in the suite
  can tell a stale wheel from a fresh one; the failure mode is a green run measuring last week's
  code, which is the same shape as §16.8's empty CI job and §23.6's emptied parity section, reached
  through a fourth door.

The test is now the opposite assertion with the history in its docstring, and it pins both halves —
`None` refused, omission still defaulting — because §24.7's `Option<Option<_>>` arm order is
inverted from the obvious guess and is exactly the sort of thing a later refactor flips back.

### 25.9 The guard, widened in both halves

§23.7's residue — *a derive is only as wide as the list it derives over* — applied twice here.

* The function half's `operators2d` entry gained the seven new names. Checked the way §23.7 says to:
  one name was removed, the guard was **observed to fail** with the right message, and only then
  restored.
* A new entry: `plate` does `from .operators2d import guitar_mask, …` at module scope, so it is a
  **client** of ported names while being unported itself. That is the captured-binding hazard the
  guard already covers for `membrane`, `mallet`, `reed` and four strings, and it is worse here than
  for any of them — a Python `guitar_mask` under a run that reports Rust is a plate with a possibly
  different node set, which every physics bar in the suite would pass.

### 25.10 What is bit-identical

| | |
|---|---|
| `guitar_half_width`, 20,001 points × 5 outlines | **bit-identical** |
| … and the returned *shape*, ranks 0 through 3 | identical |
| `guitar_scale`, 5 outlines × 3 widths | **bit-identical** |
| `guitar_mask`, 2 geometries × 5 outlines × 8 grids | **identical, node for node** |
| `live_cells`, `cells_per_node` (dtype included) | **identical** |
| `prune_to_area_carrying`, mask *and* dropped count | **identical** |
| the three refusals, message for message | identical |
| `guitar_area` | 1.2e-13 relative — **by decision**, §25.6 |

The exact rows are a claim about the port. They are *not* a claim about a CPU, and that is the
batch's point: for the four real outlines the comparison has ≥1.9e7 ulps of slack, so the mask
cannot move even if `sin` does.

### 25.11 The success condition

* `cargo test --workspace` green, debug **and** release — 10 new native bars, including one that
  searches for a witness distinguishing `(a·b)·c` from `a·(b·c)` and **fails if it finds none**,
  which is §17.2's constant fold and §23.5's empty witness search guarded in advance.
* `cargo fmt --all --check` and `cargo clippy --workspace --all-targets -- -D warnings` green —
  which is §25.8, and they are now part of a batch's list rather than of CI's.
* `pytest tests/test_rust_parity_ops2d.py` green with the flag and without it — 406 tests.
* `PHYSSYNTH_RS=1 pytest` green on the whole plate family — the **existing, unmodified** Python
  tests for the supported plate, the free plate, the orthotropic plate, the guitar plate, the von
  Kármán bracket and the membrane, all running on a Rust-built outline.
* The whole Python suite green on the default path — 3,565 passed, 1 skipped, after §25.8a — and the shipped
  numbers **unmoved**: 0 mask
  node flips, 0 ulps on `scale`, 0 relative change in `guitar_area` against the previous spelling.
  That first number is a *test* — `tests/test_guitar_plate.py` holds the old expression and 100
  parametrisations of the comparison — not a scratch script that ran once.

### 25.12 What was measured

| quantity | value |
| --- | --- |
| smallest comparison margin, four real outlines | 1.9e7 … 1.9e10 ulps of `half` |
| … median over 130 configurations | 8.6e11 ulps |
| … degenerate lens, `N = 32` | **1 ulp** (four nodes), and 0 at `t = ½` |
| `np.sin`/`np.cos` vs `math.sin`/`math.cos`, profile values | differ in 221,580 of the sweep |
| … resulting mask flips on this machine | **0** |
| … resulting change in `guitar_scale` | **0 ulps** |
| Rust vs Python: profile, scale, mask, cells, prune | **bit-identical / identical** |
| Rust vs Python: `guitar_area` | 1.2e-13 relative |
| quadrature vs the closed form, three waists | < 1e-12 relative |
| scalar vs vectorised profile, 2,000,000 points | 467.4 ms vs 45.9 ms |
| scalar vs vectorised profile, 20,001 points | 4.4 ms vs 0.21 ms |
| spikes the shipped outline produces, `N = 20…48` | 2–4, all removed by one prune |

### 25.13 What the next batch inherits

* **The matrices are what is left of `operators2d`**, and `portable.canonical` goes on with them —
  §18.4 wrote the decision down in advance and §24.13 repeated it. `biharmonic_from_mask` is
  `L @ L` through SciPy's SMMP kernel, so it comes back with **descending** column indices, and
  `plate.py` multiplies by that operator every timestep.
* **Ask whether the output is discrete before asking how exact it is.** §25.2. The remaining
  functions in this module are all matrices, so the question is settled for them — but `connection`
  and the plate's own construction contain index choices, and an index is a decision too.
* **A margin measurement costs one script and settles a spelling.** §25.3. It is cheaper than the
  argument about whether an exact assertion is safe, and it produces a *test* rather than a note.
* **A batch's success condition must cover the gate's first two steps.** §25.8.
* **Rebuild the wheel before believing a parity number, and re-run what recorded the behaviour you
  changed.** §25.8a. Phase 4's `boundary=None` fix sat in `crates/` and not in `site-packages`, so
  the test contradicting it stayed green for a whole batch.
* **When a port changes the reference, pin the reference's *old* behaviour somewhere the port
  cannot reach.** §25.5's third pin. Every parity assertion compares Rust against the Python as it
  is now; only a test holding the previous expression can tell a faithful port of a slip from a
  faithful port. The next batch changes `plate.py`'s operator assignment (`portable.canonical`), so
  the same hole opens there — and there the quantity is continuous, so the pin is a tolerance on
  the spectrum rather than an equality on a mask.
* **Check inherited sentences** — still true, and now with a fourth instance: Phase 4's §24.11 was
  complete about the tests and silent about the linters.

## 26. Phase 5, batch 2, as built (2026-08-28) — the plate's matrices, and an order that was never the algebra

§25 split Phase 5's first group on what a function's output *is*. This batch takes the other half of
that split — the matrices — and splits it again, on a different question: **what kind of claim is
available at the end.** `biharmonic_from_mask`, `dirichlet_interior_d2_1d`, `orthotropic_biharmonic`
and `free_plate_stiffness*` are assemblies, so an *exact* claim is available for all four once one
question is settled. `VonKarmanBracket` and `AiryStressSolver` are not: the second factors with
SuperLU, and §24.2 already established that SuperLU parity is a measured tolerance and cannot be
anything else. Bundling them would have made the exact half hostage to the negotiation over the
inexact one, so the Airy solver and the four private 1-D differences that serve it go to batch 3.

The batch's own finding is smaller than §25's and more useful than it looks: **the thing standing
between the two implementations was never the arithmetic. It was where each number is stored.**

### 26.1 The shape on disk

```
crates/physsynth-core/src/sparse.rs         `Csr::add`, `Csr::kron`, `Csr::identity`; four new
                                            unit bars
crates/physsynth-core/src/ops2d.rs          biharmonic_from_mask / dirichlet_interior_d2_1d /
                                            orthotropic_biharmonic / free_plate_stiffness{,_from_mask};
                                            header rewritten
crates/physsynth-core/tests/ops2d.rs        8 new native bars, including the nullspace money test
                                            and two spelling pins that fail if they find no witness
crates/physsynth-py/src/ops2d.rs            five bindings, five refusals reproduced verbatim
crates/physsynth-py/src/lib.rs              five registrations
physsynth/core/operators2d.py               `portable.canonical` at the two squared-operator
                                            assignments; five `_py` aliases and the swap block
physsynth/core/plate.py                     `Plate` and `VKPlate` route their isotropic `B` through
                                            `biharmonic_from_mask` instead of spelling `L @ L`
physsynth/core/portable.py                  the scope paragraph, whose prediction came true
tests/test_rust_parity_ops2d.py             the batch's comparison (406 tests -> 520)
tests/test_plate_modal.py                   the pre-change spelling, held where the port cannot
                                            reach it, plus the free plate's "unmoved" assertion
tests/test_stability.py                     five names in the guard's operators2d set, four more in
                                            `plate`'s captured bindings
```

### 26.2 The finding: SciPy's product disagreed on order, not on value — and the order is on the update path

Every previous batch that met a SciPy sparse product asked whether the *values* could be matched.
Here they could, immediately and everywhere. An accumulation that runs the contracted index `k` over
row `i` of the left operand in ascending order — which is what SciPy's SMMP kernel does and what a
plain Rust loop does — reproduces `L @ L` **bit for bit**: 0 differing entries out of 2,629 at
`N = 16`, and 0 at every one of seven grids and two staircased outlines. The same holds for the
three products inside `orthotropic_biharmonic` and the four inside `free_plate_stiffness`.

What did not match is the **stored column order**, and the shape of that disagreement is worth
stating precisely, because §18.2 has a sentence about it that turns out to be too specific. §18.2
found `biharmonic_matrix` coming back **descending** and wrote the rule that way. In two dimensions
it is not descending. Over the shipped grids, of 610 rows with more than one entry:

| row order out of SciPy's `L @ L` | rows |
| --- | --- |
| ascending | 0 |
| descending | 10 |
| **neither** | **600** |

The order is the order the kernel happened to touch the columns in. It is not a property of the
algebra, it is not reversible by a rule, and it is not something a second implementation can
reproduce without transcribing a SciPy internal — which is the bargain §10 refused and §18.3
re-refused.

And it reaches the trajectory. A CSR matvec accumulates a row in *stored* order, and `Plate.step`
forms `B @ u` twice per timestep. So this is §18.2's situation exactly: not a read-out discrepancy
but a different sum on the update path from step one.

The fix is §18.2's fix, applied where §18.4 said in advance it would be needed:
`portable.canonical` at the two assignments that build a squared operator. **The general rule this
sharpens:** when a port meets a library kernel, ask *separately* whether the values agree and
whether the order does. They are different questions with different answers and different remedies —
the values are arithmetic and can be matched; the order is a library internal and must be
normalised on the side that has one.

### 26.3 The free plate needed nothing, and that is a measurement rather than a hope

`free_plate_stiffness`'s `K` is a Gram product, `C2xᵀ (Wa C2x)` and friends, and SciPy returns those
**already sorted** — measured canonical in *every* row of every rectangle (5 grids), every disk and
every guitar (3 resolutions each) tried. So `canonical` is a no-op there and was not applied.

The consequence is the batch's whole risk profile. Only the **supported** plate's numbers move:

| | |
|---|---|
| supported plate, `max\|Δu\|/amp` over 2,000 steps, `N = 12` | 1.2e-13 |
| … `N = 16` | 6.0e-14 |
| … orthotropic supported, `N = 12` | 7.7e-14 |
| lossless energy drift, before the sort | 2.2e-14 |
| … after | 2.3e-14 |
| free / orthotropic-free / guitar plate | **bit-identical, unmoved** |

The drift bar is 1e-10 and does not notice. This matters more than the size of the number, because
§24.5 established that a free-edge model *integrates* a per-step difference twice along its
`{1, x, y}` nullspace — growth like t². Had `K` needed the sort, this batch would have been
re-tolerancing the free plate, the orthotropic free plate and the guitar plate as a side effect of
porting a matrix. It did not, and the reason is checkable rather than lucky, so
`tests/test_plate_modal.py` asserts it: if a SciPy release ever starts returning a Gram product
unsorted, the free plate joins the supported one in moving, and that test is where it surfaces.

### 26.4 Two model classes were bound together by an anchor, again

`plate.py` did not call `biharmonic_from_mask` at all before this batch. Both `Plate` and `VKPlate`
spelled `self.B = (self.L @ self.L).tocsr()` **inline**, in two places, and `biharmonic_from_mask`
was a helper used only by tests.

That is §15.2's shape and it had to be resolved the same way. `tests/test_vk_energy.py` and
`tests/test_vk_free.py` assert that a `VKPlate` with `nonlinear=False` is `array_equal` to a
`Plate` — a bit-identity anchor between two *different* model classes. Canonicalising one operator
and not the other would have broken it, with a message about physics. So both call sites moved to
the shared builder in one edit, which also removes a §25.5-style deliberate duplicate rather than
pinning it: two expressions that agree today, one of which was about to stop agreeing.

### 26.5 The association that looks dangerous is provably harmless, and the reason costs no measurement

The two Gram products in this module associate **differently**, and it is visible in the source:

```python
free_plate_stiffness_from_mask:  cross  = C2x.T @ (Wa @ C2y)     # explicit parens -> RIGHT
AiryStressSolver:                self.Bf = (Lc_r.T @ Wa @ Lc_r)  # no parens -> Python gives LEFT
```

Same mathematical form `BᵀWB`, opposite association. A port that wrote one helper and reused it, or
that followed the sibling (§23.4's hazard), would silently pick one — and for a general three-factor
product that is a different matrix. Measured: **about a third of random value triples distinguish
`(x·w)·z` from `x·(w·z)`** (69,943 of 200,000).

None of *this* operator's do — 0 of 16, and 0 differing entries over five rectangles, three guitars
and three disks. The reason is structural and needed no measurement to see. Every curvature entry is
`1/h²` times an exact power of two (`1`, `-2`), and every area weight is `h²` times one of
`{1, ½, ¾, ¼}`. Scaling by a power of two is exact, so both associations reduce to `fl(fl(a·w)·a)`
and `fl(a·fl(w·a))` with the *same* `a` at both ends — and those are equal because IEEE
multiplication commutes. The outer two factors sharing a mantissa is what does it, and a staircased
rim's `¾` weight does not disturb it.

This is §23.2's move applied to a product instead of a sum: **ask a cheap question about the shape
of the arithmetic before assuming a reassociation is visible.** The port keeps the faithful
right-association regardless, because the property belongs to the values and not to the code — and
`crates/physsynth-core/tests/ops2d.rs` asserts both halves, the agreement on the plate's value set
*and* a searched witness off it, so the first half cannot go green while saying nothing.

### 26.6 The two spelling pins, and why both are searches rather than constants

Two arithmetic spellings in this operator are load-bearing and neither is visible in the algebra.

The **twist coefficient** is `(1/h) * (1/h)` and not `1/(h*h)`, because the cell-centred twist is a
product of two forward first differences. Those differ in the last digit whenever `h` is not exactly
representable, which showed up on exactly one grid of the seven-grid survey this operator was first
validated over — so the pin sweeps seven grids and **fails if it finds no witness**, and the helper
functions carry `#[inline(never)]` so §17.2's constant fold cannot merge them.

The **association** pin, above, is a search for the same reason but a sharper one, and it caught its
own first draft. The witness was written as three hand-picked constants; they landed in the agreeing
two-thirds, the `assert_ne!` fired, and the test was rewritten to search a deterministic sequence
until it finds one. That is §23.5 with the roles reversed: there, a search found nothing and the
test went green; here, a *constant* found nothing and the test went red. Both failures are the same
failure — **a spelling test must be able to tell "no difference exists" from "I did not look in the
right place"** — and only the searching form can.

### 26.7 The guard caught the exact hazard it was written for, on the same day

§17.6 recorded that `collision` fell out of the swap guard's table for a whole batch because three
of its public names carry a leading underscore while their `_py` aliases do not, and the derive
could not resolve them. §23.7 recorded that the *class* half of the guard had been one name short
for six batches.

`dirichlet_interior_d2_1d` is this module's first private-but-swapped name, and the first draft
aliased it `_dirichlet_interior_d2_1d_py` — the natural spelling, and one the derive skips because
it filters names starting with `_`. The guard failed immediately and by name. Cost: one rename, and
a comment in `operators2d.py` saying which convention applies and why, so the next private swap does
not rediscover it. This is the first time in the migration that a guard has caught a guard-shaped
hole *inside the batch that made it*, which is the only outcome that makes the machinery worth its
weight.

The rename then broke the parity file, which had been written and run against the *old* alias — and
that is §25.8a arriving inside the batch that cites it, for the third batch running. The rule keeps
being restated because it keeps being the same shape: **a change to a shared name is not finished
until the tests that named it have been re-run**, and the only reliable way to know is to re-run
them rather than to reason about who imports what. It cost one `sed` here because the batch's own
success condition names the file; a batch whose condition named only the *models* would have shipped
it.

### 26.8 What the port did not buy: speed

Honest and worth recording, because §11.6's rule predicts it. Construction-time only, `N = 32`:

| | Python | Rust |
| --- | --- | --- |
| `biharmonic_from_mask` | 0.63 ms | 0.44 ms |
| `free_plate_stiffness` | 2.53 ms | **3.49 ms** |

The stiffness is *slower* in Rust, and that is §11.6 exactly: the win is per-call overhead, not
arithmetic, and `free_plate_stiffness` is five sparse products through SciPy's compiled SMMP kernel
with only a handful of Python-level calls around them. There is nothing to win, and the Rust side
pays for building each row as its own `Vec`. Neither number is on any hot path: `free_plate_stiffness` is called **exactly once per
`Plate.__init__`**, and `biharmonic_from_mask` likewise. A future reader should not read the table
as a regression — no optimisation was made and none is warranted until `plate.py` itself ports and
the *step* becomes the thing being measured.

### 26.9 What is bit-identical

| | |
|---|---|
| `biharmonic_from_mask`, 5 rectangles | **bit-identical** (`data`, `indices`, `indptr`, `nnz`) |
| … 3 guitar and 3 disk outlines | **bit-identical** |
| `dirichlet_interior_d2_1d`, 5 sizes × 4 grids | **bit-identical** |
| `orthotropic_biharmonic`, 5 grids × 4 grain triples | **bit-identical** |
| `free_plate_stiffness`, 5 grids × 4 values of `nu` | **bit-identical** (`K` and `W`) |
| … 3 grids × 4 constant splits | **bit-identical** |
| `free_plate_stiffness_from_mask`, 6 outlines × 2 `nu` | **bit-identical** |
| every `index_map`, dtype included | identical |
| the five refusals, message for message | identical |

There is no tolerance-level row in this batch. That is what the split in §26 bought, and it is why
the Airy solver — which cannot have one — is in the next batch instead.

### 26.10 The success condition

* `cargo test --workspace` green, **debug and release** — 8 new native bars in `ops2d`, 4 new unit
  bars in `sparse`, including two spelling pins that fail if they find no witness.
* `cargo fmt --all --check` and `cargo clippy --workspace --all-targets -- -D warnings` green, and
  `ruff check .` green — §25.8, in the batch's list rather than CI's.
* `pip install ./crates/physsynth-py` **before** any parity number is believed — §25.8a.
* `pytest tests/test_rust_parity_ops2d.py` green with the flag and without it — 520 tests.
* `PHYSSYNTH_RS=1 pytest` green on the whole plate family — the existing, unmodified Python tests
  for the supported plate, the free plate, the orthotropic plate, the guitar plate, the von Kármán
  bracket, the Airy solver and the membrane, all running on Rust-built matrices.
* The whole Python suite green on the default path, and the shipped numbers moved **only** where
  this section says: supported plate 1.2e-13 of amplitude with drift unmoved, free / orthotropic-
  free / guitar plates bit-identical. That is a *test* — `tests/test_plate_modal.py` holds the
  pre-port expression and asserts both halves — not a scratch script that ran once.

### 26.11 What the next batch inherits

* **What is left of `operators2d` is the nonlinear plate**: `_collocated_d2_1d`, `_forward_d1_1d`,
  `_centered_d2_1d`, `_avg_d1_1d`, `VonKarmanBracket` and `AiryStressSolver`. The infrastructure it
  needs — `add`, `kron`, `identity`, and the sparse LU from Phase 4 — is already in place, so that
  batch is cheaper than this one by everything except the solver.
* **Its claim will be a tolerance, and that is settled in advance** (§24.2). `AiryStressSolver`
  factors with SuperLU, which is supernodal; matching it is a claim about how SciPy was built. Do
  not spend the batch re-litigating that.
* **The order question there is not `_clamped_d2_1d`, it is the slice.** That matrix is built
  through `lil` and two scalar assignments, and `lil.tocsr()` sorts, so it is almost certainly
  canonical already. What is unmeasured is `Lc.tocsc()[:, self._cols]` — a CSC *column slice* — and
  then `Lc_rᵀ @ Wa @ Lc_r`, which Python **left**-associates where `free_plate_stiffness` writes the
  same form right-associated. §26.5's mantissa argument probably extends (the ghost-mirror end rows
  are `2/h²` alongside `1/h²` and `-2/h²`, still all powers of two times one mantissa) — but
  "probably" is what §26.5 replaced with a one-line argument, so make the argument rather than
  inheriting the conclusion.
* **Ask the value question and the order question separately.** §26.2. They have different answers
  and different remedies, and conflating them is how a batch ends up believing the arithmetic is
  unmatchable when only the storage was.
* **Ask whether the outer factors share a mantissa** before assuming a reassociation is visible.
  §26.5. It costs no measurement and it settled a hazard the previous batch would have spent a
  script on.
* **A spelling pin must search, not assert a constant.** §26.6, and §23.5 from the other side.
* **Check inherited sentences** — still true. §18.2's "descending" was accurate about the 1-D
  biharmonic and wrong as a general rule; in two dimensions the order is *neither*, in 600 of 610
  rows. A rule written from one measurement should be re-measured before it is relied on in a new
  place.

## 27. Phase 5, batch 3, as built (2026-08-28) — the nonlinear plate, and a reassociation that moved a sum rather than a product

This finishes `operators2d` and with it the fourth of the module's four batches. What is ported is
the *nonlinear* plate: the five private 1-D differences, `VonKarmanBracket` and `AiryStressSolver`.

§26 split the previous batch on **what kind of claim is available at the end** and sent the two
inexact pieces here on purpose, so that the exact half would not be hostage to the negotiation over
the inexact one. That was the right split and it paid twice: the bracket turned out to be *exactly*
reproducible — every matvec in it is a canonical row gather — while the Airy solve is a SuperLU
solve and can only ever be a measured tolerance (§24.2). Bundling them would have hidden the first
result under the second.

The batch's own finding is a correction to a rule §26 wrote one batch ago, and it is worth stating
precisely because §26's version is not wrong so much as **too narrow**.

### 27.1 The shape on disk

```
crates/physsynth-core/src/sparse.rs         `Csr::select_columns`
crates/physsynth-core/src/ops2d.rs          collocated_d2_1d / forward_d1_1d / centered_d2_1d /
                                            clamped_d2_1d / avg_d1_1d, `VonKarmanBracket`,
                                            `AiryStressSolver`; header rewritten
crates/physsynth-core/tests/ops2d.rs        11 new native bars, including the bracket's triple
                                            self-adjointness and its negative control
crates/physsynth-py/src/ops2d.rs            five bindings, two pyclasses, the refusals reproduced
crates/physsynth-py/src/lib.rs              five registrations, two classes
physsynth/core/operators2d.py               ONE pair of parentheses in `AiryStressSolver`; five
                                            `_py` aliases, two `*Py` aliases, the swap block
tests/test_rust_parity_ops2d.py             the batch's comparison (520 tests -> 694)
tests/test_stability.py                     `operators2d` joins the CLASS derive's tuple; five
                                            names and two classes added to the expectations
.github/workflows/ci.yml                    one new flagged step
```

### 27.2 The finding: §26.5's mantissa rule is about products, and the association moved the sum

`AiryStressSolver` assembles `B_F = Lc_rᵀ Wa Lc_r`, and the reference wrote it with no parentheses,
which Python left-associates. §26.5 had already established the question to ask of a form like that
— *do the outer factors share a mantissa?* — and had used it to retire the same hazard in
`free_plate_stiffness` without a measurement. §26.11 predicted the argument "probably extends" here
and told the next batch to make it rather than inherit it.

It does extend, exactly as predicted, and it settles nothing.

Every entry of `Lc` is `{1, 2, 4}` times the single reciprocal `1.0/(h*h)` (the `2` is the clamped
ghost mirror, the `4` is the two axes' diagonals meeting under `+`); every entry of `Wa` is
`{1, 1/2, 1/4}` times the single product `h*h`. So in every term `A_ki · W_kk · B_kj` the two outer
factors carry the *same* mantissa, `fl(fl(αω)·α)` and `fl(α·fl(ωα))` differ only by commuting a
multiplication, and IEEE multiplication commutes exactly. Measured term by term at the entries that
disagree: **not one term differs**.

And the two matrices differ anyway. The association does not move the products, it moves the
**sum** — because the two bracketings route the contraction through differently-*ordered*
intermediates. SciPy hands `Lc_rᵀ @ Wa` back with every row **descending**, and a sparse product
contracts the shared index in the stored order of its left operand's rows, so:

| bracketing | contraction order |
| --- | --- |
| `(Lc_rᵀ @ Wa) @ Lc_r` — what Python does with no parentheses | **descending** |
| `Lc_rᵀ @ (Wa @ Lc_r)` | **ascending** |

Over the 22 grids the test suite actually builds an `AiryStressSolver` on — enumerated from an
instrumented run, not sampled — those are different matrices on **2**: 2 entries of 501 at
`(8, 8, h = 0.0375)` and **46 of 1,889** at `(16, 12, h = 0.06)`, both at ~1e-16 of the entry.
So the hazard is live on shipped configurations, and §16.4's blind-fixture story does *not* apply
here: the suite's own grids contain witnesses.

**The remedy is a pair of parentheses**, and that is the part worth carrying forward. This project
has now needed three different fixes for an ordering problem:

1. **reproduce the values** — an ascending-`k` accumulation matches SciPy's SMMP kernel (§26.2);
2. **sort the storage** — `portable.canonical`, where the kernel's output order is the difference
   and no rule reproduces it (§18.2, §26.2);
3. **re-associate** — where the order enters through an *intermediate* the algebra never asked for,
   and one of the two legal bracketings contracts over an operand that is already canonical.

The third is the cheapest of the three by a wide margin: it changes no module, adds no call, and
costs nothing at run time. The question that finds it is "which operand's stored order does this
contraction run over, and did I choose it or did SciPy?"

One corollary about what the Rust side could not have done: **the descending intermediate is not
expressible in this crate at all.** `Csr::from_rows` sorts, so a Rust `Csr` is canonical by
construction and both associations of the Gram give the same matrix. There was therefore no way to
"just match the reference" from the Rust side — which is §10's decision to keep the Rust `Csr`
canonical paying off for the third time, and the reason the edit had to be Python's.

### 27.3 The bracket is exact, including the one place the two languages do different loops

`VonKarmanBracket.__call__` is the nonlinear plate's update path: `l(w, w)` sources the Airy solve
and `l(w, F)` is the coupling force, both once per Picard sweep of every timestep. It comes out
**bit-identical**, at every grid and every field tried, and three of its four matvecs are trivially
so — `Sxx`, `Syy` and `Dxy` are used as `M @ v` and all three come out of `kron` ascending.

The fourth is not trivial and is worth a paragraph because the answer is a *lemma* rather than a
measurement. `Acell` is used **transposed**, and in SciPy `csr.T` is a **CSC**, whose matvec
scatters columns where a CSR gathers rows. Those are genuinely different loops. They are also the
same order: a CSC matvec accumulates each output entry over increasing *column* index, and a
sorted-CSR row gather accumulates over increasing column index. Identical for any canonically
stored matrix, and measured 0 differing entries in 21,780 at four grids.

That is §23.2's move — ask a cheap question about the shape of the arithmetic before measuring it —
applied to a *loop structure* rather than to a sum's length. The premise is canonical storage,
which is exactly what this batch's other half turns on, so the parity file pins it rather than
leaving it as a remark.

The one part of the bracket that is *not* exact is `trilinear`, and for a reason already on the
books: it contracts with `inner2d`, which is `np.dot`, which is BLAS (§14.2). A read-out, held to
the Group A target.

### 27.4 The Airy solve, and why one constant is the wrong shape for its bar

`AiryStressSolver.solve` factors with SuperLU on the Python side and with `sparse_lu` on the Rust
side, and §24.2 settled in advance that this can only be a measured tolerance. What the measurement
says is more useful than a number:

| grid | max &#124;Δ&#124; / amplitude |
| --- | --- |
| 4 × 4 | 3.1e-16 |
| 12 × 10 | 8.5e-15 |
| 16 × 12 | 3.6e-14 |
| 24 × 19 | 1.3e-13 |
| 48 × 48 | 2.3e-12 |
| 80 × 64 | 9.5e-12 |
| 96 × 96 | 5.4e-11 |
| 160 × 128 (the largest the suite builds) | **5.2e-10** |

That is not a port degrading; it is `N⁴`, which is a biharmonic's condition number. Both solves are
**backward stable** — the residual `‖B_F f − rhs‖ / (‖B_F‖ ‖f‖)` sits at machine precision on both
sides at every grid — so the forward difference between them is conditioning times epsilon and says
nothing about either implementation. The parity file asserts the backward errors precisely so that
the growing forward tolerance cannot be misread. A single small constant would have failed at
24 × 19 and been meaningless at 4 × 4.

The parity file asserts this as a **scaling law** rather than a constant — `1e-17 · (n_x n_y)²`,
fitted to the worst measured ratio with ~8x slack, which is uniform across four orders of grid size
— and it parametrises over the large grids as well as the small ones. A bar tested only up to
24 × 19 would have been slack by two orders exactly where the claim bites, which is §16.4's shape
arriving in a *tolerance* instead of in a fixture.

It is worth saying why the flagged CI step is green at 160 × 128 rather than noting that it is.
Nothing in the von Kármán or airbox suites compares a *displacement* across implementations: the
airbox bars are `DRIFT_TOL` and `LEDGER_TOL`, both `1e-12` **relative on an energy**, and every
`array_equal` in `test_airbox_vk.py` compares two configurations of the *same* implementation (the
zero-load reduction, the `nonlinear=False` reduction) rather than two languages. A 5.2e-10
displacement difference cannot reach any of them. That is the same conclusion as §27.5's from the
other end, and it is the reason the step passes for a reason rather than by luck.

**And §24.4's manoeuvre separates the two questions completely.** Driving the *Python* Airy solver
through the *Rust* factorization makes the two solves **bit-identical, at every grid tried**. So the
assembly is exactly right and the entire residue above is the solver — which is the strongest
statement available here, and the only thing that would have caught a reassociated assembly, since
the solver gap is two orders larger than one would be (§19.4's finding waiting to happen).

### 27.5 A seventh agreement regime, and the first where the trajectory becomes unrelated while every physics bar stays green

§16.5 asked how long two implementations of a nonlinear model stay comparable, and the answers have
accumulated: the model class (§16.5), whether the nonlinearity recurs (§17.5), whether the
recurrence drives the system onto an attractor or off one (§20.5), the amplitude (§19.5), the
absence of amplification in a linear model (§18.6), and a boundary condition's nullspace (§24.5).
The von Kármán plate adds one more, and it is the sharpest so far because **the two answers differ
by twelve orders of magnitude in the same model at the same amplitude**.

Measured at `N = 12`, `Lx = Ly = 0.4`, `fs = 48 kHz`, `σ = 0`, with only the Airy solve differing
(the bracket being bit-identical):

* one regime **random-walks and stays there** — 5.6e-15 of the running peak at step 100, 1.1e-13 at
  1,000, 5.7e-13 at 4,000 — and it does *not* leave that regime when driven harder: at ten times the
  plate thickness it is still 1.2e-13 at 1,000. Below `w/e ≈ 0.1` the two are **bit-identical for
  4,000 steps**, because the coupling correction is a small enough fraction of the field that its
  last bits fall off the end of the addition — §23.2's mechanism, in a model rather than in a matvec.
* the other is **chaotic**, and the gap e-folds every **~57 steps**: 4.2e-14 at step 100, 4.1e-13 at
  200, 2.4e-9 at 700, 3.4e-7 at 1,000, and **0.59 of the peak by step 2,000** — completely
  decorrelated.
* **the energy does not move through any of it.** At step 2,000, with the displacements unrelated,
  the two energies agree to 5.1e-14 relative and the lossless drift is 2.6e-14 on both sides against
  the 1e-10 bar.

**What separates the two is not the fixture, and naming the fixture would have been the wrong
lesson.** The first attempt at this said "a smooth mode against a broadband one", and a broadband
start *normalised to the same peak* falsifies that immediately — it random-walks, 3.4e-13 at 1,500.
The next attempt said "amplitude", and the amplitude sweep falsifies that too: from `w/e = 0.5` to
`w/e = 10` the window is flat at ~1e-13. The observable that separates every run actually done is
the **Picard sweep count**: two to six sweeps a step is the random walk, eleven or more is the
exponential. That is the model's own report of how hard the nonlinearity is working, it is already
a public attribute (`n_iters`), and a future batch can read it without reconstructing this fixture
set — which is the only form of this finding worth carrying.

So the parity bar for a von Kármán trajectory reads the **energy**, never `max|du|/amp`. That is
§24.5's conclusion reached by an entirely different mechanism — there a rigid-body nullspace
integrated a per-step gap twice, here a positive Lyapunov exponent multiplies it — and the general
form is worth writing down plainly: **a conserved quantity is not a trajectory comparison.** Two
runs can agree on every invariant the project measures and share no digits of state.

A second mechanism was watched for and is a *consequence* rather than a cause. `VKPlate`'s Picard
loop branches on `‖Δw‖/‖w‖ ≤ couple_tol`, which is §19.2's hazard exactly: a solve that reaches a
branch. The sweep counts differ on 90 of 2,000 steps — but the **first** difference is at step
1,553, by which time the deviation is already 2.1e-3. On the smooth fixtures the counts never differ
at all, over 4,000 steps at every amplitude tried. So §20.3's refinement applies: the residual falls
geometrically and crosses `couple_tol` with orders of room, so a 1e-14 perturbation cannot flip the
comparison until the trajectories have separated for other reasons.

### 27.6 What did not have to change, and one thing that did

`plate.py` needed **no edit**. `VKPlate` holds a bracket and a solver and calls three methods on
them; the two Rust classes present the same attributes, so the swap is a rebinding and nothing else.
That is the first Group D port with no client change at all.

The one Python edit outside the parentheses is the swap block, and it carries §23.6's warning
explicitly: with the flag set, a test that pins `operators2d.VonKarmanBracket` to measure the thing
that name builds is comparing Rust against Rust. The parity file reaches for `VonKarmanBracketPy`
and `AiryStressSolverPy` instead, and `tests/test_stability.py` grew `operators2d` in the **class**
derive's tuple — which it had never been in, so a `*Py` alias here would have been invisible to the
guard. That is §23.7's finding coming due exactly where §23.7 said it would: *a derive is only as
wide as the list it derives over.*

`_collocated_d2_1d` is worth one line of its own: it has **no production call site left**. §26 moved
`free_plate_stiffness` onto a mask-built curvature, so the only thing that still calls this function
is `tests/test_free_plate_modal.py`, which uses it as an independent oracle for that operator. It is
ported anyway — an oracle nobody compares is an oracle nobody is checking — and the parity file
asserts the one property that distinguishes it from its near-twin `centered_d2_1d`: the two end rows
are **empty**, not zero-valued, a difference visible only in `nnz`.

### 27.7 §19.7's line continuation, a third time — and the pin worked

A literal `\n` in a YAML `run:` block was introduced in §19, cited-and-reintroduced by §20 (which is
what proved the failure comes from *tooling* rather than from typing), and asserted in
`tests/test_ci_workflow.py` as a result. It happened again in this batch, from a third tool.

This time it never reached CI: `test_no_run_block_line_is_a_swallowed_continuation` failed on the
first local run, named the line, and the fix took one minute. That is what the pin was for, and it
is the cheapest confirmation available that a scar written down as a *test* outperforms the same
scar written down as a *paragraph* — §20.7 argued that in the abstract and this is the receipt.

### 27.8 The measured comparison

| | |
|---|---|
| `collocated_d2_1d`, 9 sizes × 5 grids | **bit-identical** (`data`, `indices`, `indptr`, `nnz`) |
| `forward_d1_1d`, `centered_d2_1d`, `clamped_d2_1d`, `avg_d1_1d`, same sweep | **bit-identical** |
| `Sxx`, `Syy`, `Dxy`, `Acell`, 4 grids | **bit-identical** |
| `VonKarmanBracket.__call__`, 4 grids × 3 fields | **bit-identical** |
| `Acell.T @ v` — CSC scatter vs CSR gather | **0 of 21,780** entries differ |
| `AiryStressSolver.Bf`, 16 grids, CSC arrays included | **bit-identical** (after the parentheses) |
| ... with the left-associated spelling instead | 2 grids differ, up to 46 of 1,889 entries |
| `AiryStressSolver.solve`, same grids | 3.1e-16 to 1.3e-13 of amplitude; both backward stable |
| ... with the Rust factorization on both sides | **bit-identical** |
| `trilinear`, `laplacian_norm_sq` | ~1e-16 relative — `np.dot`, §14.2 |
| `mask`, `index_map`, dtype included | identical |
| the refusals, message for message | identical |

### 27.9 The success condition

* `cargo test --workspace` green, **debug and release** — 11 new native bars.
* `cargo fmt --all --check`, `cargo clippy --workspace --all-targets -- -D warnings` and
  `ruff check .` green — §25.8, in the batch's list rather than CI's.
* `pip install ./crates/physsynth-py` **before** any parity number is believed — §25.8a.
* `pytest tests/test_rust_parity_ops2d.py` green with the flag and without it — 694 tests.
* `PHYSSYNTH_RS=1 pytest` green on the whole von Kármán family — bracket, Airy solve, energy,
  modal, free, stability, connection and the room-loaded plate — plus the rest of the plate family,
  all on Rust-built operators.
* The whole Python suite green on the default path, and the shipped numbers moved **only** where
  §27.2 says: two of the 22 Airy grids, at ~1e-16 of an entry, construction-time.

### 27.10 What the next batch inherits

* **`operators2d` is finished.** What is left of Phase 5 is `plate`, `connection` and
  `string_geometric`; then `airbox` and `analysis/`.
* **The plate is now the only Python thing between two Rust halves.** `VKPlate` holds a Rust
  bracket, a Rust Airy solver and Rust-built matrices, and does its own timestepping in NumPy. That
  makes it the cheapest remaining model to port and the one whose port will change the least.
* **Ask which operand's stored order a contraction runs over.** §27.2. It is a third remedy for an
  ordering problem, it costs nothing, and the question is answerable by reading the expression.
* **Read the energy, not the displacement, on anything chaotic.** §27.5. And measure the window at
  *two* fixtures — a smooth one and a rough one — because the same model at the same amplitude gave
  answers twelve orders apart.
* **Hold the solver constant before believing an assembly claim.** §24.4's manoeuvre, third use,
  and the only reason this batch can say the assembly is exact rather than "within the solver gap".
* **The Rust factorization is slower than SuperLU, it is construction-time, and it is invisible.**
  2.5 s against 0.16 s at 160 × 128, 0.43 s against 0.06 s at 96 × 96 — §11.6's shape again, nothing
  to win around five compiled kernel calls. It was worth checking whether that is felt, because
  unlike §26.8's 1 ms it is seconds; it is not. The same 24 files, on this machine, back to back:
  **350.9 s on the default path against 291.6 s with the flag set**, 614 tests either way. The
  flagged path is *faster*, because the per-step overhead the rest of the tree wins back is larger
  than the factorization it pays for. A paired same-machine run is the only variance-immune way to
  ask that question, which is why it was asked that way.

## §28 Phase 5, batch 4, as built (2026-08-28) — the plate, and an anchor that reached a client

`plate.py` is ported: `Plate` (models #5, #5b, #5o, #5of, #5g) and `VKPlate` (model #6), together,
with the material helper and the component count. What is left of the core is `connection` and
`string_geometric`, then `airbox` and `analysis/`.

The batch's finding is about **where a batch boundary has to stop**, and it answers it the opposite
way from §25. There the question was how far *back* to go — settle the discrete-output functions
before porting what consumes them. Here it is how far *forward*: a bit-identity anchor does not
only bind two models (§15.2) or two classes in a file, it can bind a model to a **client that is not
being ported at all**, and the client has to move with it.

### 28.1 Why both classes, and why that is not a choice

`tests/test_vk_energy.py` and `tests/test_vk_free.py` each assert that a `VKPlate` with the coupling
switched off is `array_equal` to a `Plate` at every one of 150 steps, and that their energies compare
`==`. The port moves the solve — `scipy.sparse.linalg.splu` there, the crate's `SparseLu` here, which
§24.2 settled cannot agree to the bit — so swapping one class and not the other breaks that anchor
for a reason having nothing to do with either model. §15.2's finding, reaching two classes in one
file rather than four models across four files. A class cannot be half-swapped either, so the free
branch, all three outlines and both grains came with it.

**In this implementation the anchor is structural rather than transcribed**, and that is worth more
than the parity number it produces. `VkParams` *owns* a `Params` — the nonlinear plate literally
holds the linear plate — and both classes step through the same `step_rhs`. The Python original keeps
them in step by writing the theta-scheme out twice and saying so in a docstring; here `nonlinear =
False` reduces to `Plate` because there is only one expression. The 150-step assertion is repeated
natively in `crates/physsynth-core/tests/plate.rs`, where it cannot fail for a transcription reason.

### 28.2 The finding: the anchor reached `airbox.py`, which is not being ported

Fourteen tests went red under the flag on the first whole-family run, and twelve of them were one
thing. `airbox.py`'s `_PlateSurface` and `_VKPlateSurface` **reassemble** the plate's system matrix
and factor it themselves, deliberately — the docstring says "reassembled here rather than reached
into, because `plate.py` stays untouched — which is exactly why the coupled cross-check at two
timesteps exists to pin it." Four of the family's reduction tests then turn that into a bit-identity
claim: with the load switched off, `a_loaded` **is** the plate's own `A`, so a loaded plate must
reproduce a bare one byte for byte. That is what pins `airbox`'s transcription of the right-hand
side, and it holds only while both sides factor with the same solver.

So the anchor is between a **model and its client**, the client is three phases from its own port,
and the claim it protects is about the client's Python code rather than about the plate at all. The
remedy is one name: `airbox.py` gets the same module-level swap every ported module has, rebinding
`splu` to a Rust-backed shim under the flag. `splu` is looked up as a global at call time, so the
rebinding is the whole of the change and nothing else in the file knows. Three test files that
re-derive the same matrix now import `splu` **from `airbox`** rather than from SciPy, which is more
correct than it was: a test that factors the matrix a module would factor should use the module's
factorizer, not a second one that happened to coincide.

The general form, and the question to ask before scoping a batch: **does anything outside this
module re-derive what this module computes, and does a test compare the two exactly?** If so the
client is inside the batch whether or not it is being ported. Grepping the clients for *private
names* — §0's prediction, and the thing this migration has checked five times — was not enough here,
because `airbox` reaches nothing private on the linear plate. It re-derives instead, which is a
dependency no name search finds.

One corollary that is a relief rather than a scar. `connection.py` was §0's named Phase-5 client of a
model's private names, and for the string that came true. For the **plate** it did not: `connection`
reads `n_live`, `u`, `u_prev`, `converged`, `n_iters`, `last_residual` and calls `step(f_ext=...)`,
`energy()` and `pressure()`, and touches nothing private at all. Two tests read `plate._lu` to check
that a bridge's coupling force reached the acceleration, which is §12.2 for a third time and is
answered the way §12.2 is always answered — the attribute is exposed, because a leading underscore is
not a statement about the interface.

### 28.3 A read-out whose relative error is meaningless, and it is physics

`Plate.pressure()` is the monopole read-out, `sum_i area_i u_i''`. The first parity bar written for
it compared the two implementations relatively and failed on **every free-edge fixture, by 1e-4 to
1e-1** — six orders worse than the supported branch and far too large for a summation order.

It is not the port. A free plate's stiffness annihilates the constant vector, so `1ᵀ W δ_tt u =
-κ² 1ᵀ K (θ-average) = 0` **term for term**: an unforced free plate has no monopole at all, and what
floating point returns is the cancellation residue. Measured over 200 steps, `|p| / Σ|terms|` is

| | unforced | forced |
|---|---|---|
| supported | 6.3e-4 … 5.7e-1 | 6.4e-1 |
| free | 2.1e-16 … 1.0e-13 | 3.4e-1 |

so on the free branch the quantity being compared *is* the rounding, and a relative comparison of
two roundings is a comparison of nothing. An external force breaks the nullspace and the read-out
becomes a real number again, which is where it can be compared — and that is what the parity test
does. It is also, retroactively, why this family has dipole classes at all.

Two general forms, and the second is the one to carry:

* **Normalise a reduction by the sum of its absolute terms, not by its own value.** §20.6 said the
  normaliser is part of the claim and asked for a *monotone* one; this adds that for a **sum** the
  right denominator is what went into it. With that normaliser every branch, guitar included, comes
  in under 1e-15 and the bar is a genuine last bit.
* **Ask whether the quantity being compared is identically zero before writing a relative bar.** No
  detector in this project can tell "the port is wrong" from "the physics cancels", and the second
  reads exactly like the first.

### 28.4 §23.6's fourth door: a test whose subject the implementation cannot express

`test_plate_modal.py::test_the_canonical_sort_left_the_shipped_plate_where_it_was` is §26's
regression pin: it steps one plate on the canonical biharmonic and another on SciPy's kernel-order
one and asserts the trajectory did not move. Under the flag it failed on `p_old.B = ...`, because a
Rust `Plate` does not offer a settable operator.

The fix is not a setter, and that is the point. `Csr::from_rows` **sorts** — a descending row is not
expressible in the crate at all, which is precisely why §27.2's remedy had to be applied on the
Python side. With the flag set both plates would therefore carry the *same* operator and the
comparison would assert nothing while staying green. So the test is pinned to `PlatePy` and says why.

That is §23.6's emptied-comparison shape reached through a fourth door. The first three were a
rebound module-level name (§23.6), an empty guard table (§17.6) and an empty CI job; this one is
**an implementation in which the difference being measured does not exist**. A fifth arrived inside
this batch's own parity file: `VKPlatePy.__init__` looks up the module-global `AiryStressSolver`,
which the swap has already rebound, so the Python model was holding a *Rust* Airy solver and the
shared-factorization test compared Rust against Rust — green, and asserting nothing. It now builds
`VonKarmanBracketPy` and `AiryStressSolverPy` explicitly. **A class that constructs its collaborators
by module-global name is swapped further than it looks.**

### 28.5 Three parity bars, because there are three ways this plate diverges

The exactness claim is §24.4's manoeuvre for the third time and it is unqualified: drive the Python
model through the Rust factorization and the two are **bit-identical over 400 steps at all eight
fixtures**, with and without an external nodal force, acceleration cache included. There are four
spellings of the right-hand side here (supported/free × forced/unforced) and this is the only test
that can see a reassociation in any of them — the solver gap is two orders larger.

What is left is the solver, and it is read differently on each branch. `du` normalised by the running
peak, over 20,000 steps:

| step | supported | free (rigid) | free (elastic) | free (energy) |
|---|---|---|---|---|
| 1 | 9.5e-16 | 2.0e-17 | 6.9e-16 | 1.4e-15 |
| 100 | 3.0e-13 | 3.4e-14 | 1.6e-13 | 1.0e-13 |
| 2,000 | 1.5e-12 | 3.0e-11 | 2.1e-12 | 5.1e-14 |
| 20,000 | 1.7e-11 | 2.5e-09 | 1.1e-11 | 5.6e-13 |

* **supported, linear** — §18.6's random walk, three orders over four of run length.
* **free, linear** — §24.5, in two dimensions and with a **three**-dimensional nullspace `{1, x, y}`.
  The rigid part grows like `t²` (a factor 7.2e4 over a 200-fold increase in steps, i.e. `t^1.9`)
  while the elastic part random-walks (68-fold, `t^0.8`), and the two **cross over around step 500**:
  at step 100 the rigid part is the *smaller* of the two and by step 20,000 it is 229 times the
  larger. A bar written at a short run length would therefore measure the elastic part and a bar
  written at a long one would measure the rigid part, without the fixture changing. Read the split,
  or read the energy, which moves by three orders less than either.
* **von Kármán** — §27.5's conclusion, below.

The guitar behaves as the rectangle does on every line of that table, which is the useful negative
result: the outline changes the mask and changes nothing about the divergence.

### 28.6 §27.5's threshold, reproduced one level up — and what amplitude actually does

§27 found that the *Airy solver* had two agreement regimes twelve orders apart, that the fixture and
the amplitude were both falsified as discriminators, and that the observable that separated every run
was the **Picard sweep count**: two to six is a random walk, eleven or more is exponential. The whole
model reproduces that threshold exactly, at three amplitudes of one fixture:

| `w/e` | mean sweeps | `du/peak` at 2,000 | energy agreement | drift |
|---|---|---|---|---|
| 0.3 | 3.11 | 3.4e-12 | 5.2e-13 | 5.8e-15 |
| 3 | 5.02 | 8.0e-13 | 2.7e-13 | 9.9e-14 |
| 12 | 13.50 | **2.3e-03** | 3.1e-15 | 8.6e-13 |

This **refines** §27.5 rather than contradicting it. Amplitude is still not the discriminator — 0.3
and 3 are an order apart in amplitude and land in the same regime with the same window — but
amplitude is one of the things that *moves the sweep count*, and the sweep count is the
discriminator. The right statement is that the regime is set by how hard the nonlinearity is working,
and amplitude selects it only *through* that. So the question to ask of a von Kármán fixture is not
"how hard is it driven" but "what does `n_iters` say", which costs nothing and is already public.

And the energy is flat across all of it: **3.1e-15 relative agreement at the amplitude whose
trajectory is completely decorrelated**, with the shipped drift bar at 8.6e-13 against 1e-10. *A
conserved quantity is not a trajectory comparison* — §27.5's plain form, now asserted of a model.

### 28.7 The reductions, and `portable.py` declined a fourth time

`energy()` and `_P` are `np.dot` (§14.2's BLAS reduction), `pressure()` is `np.sum` on the supported
branch (NumPy's pairwise blocking) and `np.dot` on the free one, and `area` is `W.diagonal().sum()`.
None is reproduced. `ops2d::guitar_area` and `collision::barrier_energy` already declined NumPy's
blocking with the reason written down — transcribing it is a claim about a library internal that a
point release may change, and §22.1 later added that it is also a claim about the runner's CPU — and
this port follows them.

`portable.py` is declined a fourth time, on §24.6's grounds rather than §23.3's: exactness is not
*available* downstream of the solve, so buying it in a read-out would buy nothing. With the solver
held constant the state is bit-identical and the residue is the reduction alone, measured at **under
1e-14 for the energy** and **under 1e-15 of the terms for the pressure**, at every branch.

Three construction-time numbers are not bit-identical and they are the only three: `area`,
`outline_area` (on a guitar, where it is `guitar_area`'s two-million-point quadrature) and their
quotient `area_deficit`. Nothing branches on them and none reaches a timestep. One consequence is
worth recording because it looks like a regression and is not: a rectangle's trapezoidal weights
really do sum to `Lx*Ly`, and **in NumPy's pairwise order they land on it exactly**, so the Python
deficit is a literal `0.0` where this one is 1.0e-15 away from it. The shipped bar is `abs=1e-14`
and both clear it by an order.

### 28.8 Speed

Paired, back to back on one machine:

| | `n_live` | construct | step |
|---|---|---|---|
| supported, N = 24 | 529 | 2.24 ms → 3.04 ms (**0.74x**) | 50.6 µs → 38.3 µs (1.32x) |
| guitar, N = 40 | 907 | 69.9 ms → 30.0 ms (2.33x) | 100.5 µs → 86.2 µs (1.17x) |
| `VKPlate`, N = 20 | 361 | — | 818 µs → 274 µs (**2.99x**) |

§11.6 exactly, in both directions. A supported plate's construction is five compiled SciPy kernel
calls with nothing around them to win, so Rust *loses* there; a guitar's is dominated by the outline
quadrature and the pruning loop, which are Python, so Rust wins 2.3x. The step gains a third on the
linear plate — one matvec and one solve, both already compiled — and **triples** on the nonlinear
one, because a von Kármán step is a Picard loop making three to fifteen small calls per sweep and the
thing being paid for is per-call overhead. That is the largest per-step win the migration has
measured for a field model, and it is the one that matters for the eventual real-time port.

The whole suite, paired and back to back on one machine, is **1,939 s on the default path against
998 s with the flag set** — 4,036 tests either way, 1.94x. That is the first whole-suite figure this
migration has been willing to quote, and only because it is a *within-session* pair: the batch-3
number of 654 s is from another session and is not comparable to either (the parity file itself is
6 s, so it explains none of that gap). What the pair does say is that the flagged tree is now
roughly twice the speed of the Python one on real work, and §27.10's much narrower 350.9 s / 291.6 s
was measured before the plate moved.

### 28.9 The measured comparison

| | |
|---|---|
| every derived scalar, 8 fixtures | **bit-identical** (26 of them) |
| `X`, `Y`, `mask`, `index_map`, dtype and shape included | **bit-identical** |
| `B`, `L` (supported); `K`, `W`, `w` (free) | **bit-identical**, all 8 fixtures |
| `area`, `outline_area`, `area_deficit` | ≤1e-13 relative — three reductions, by decision |
| the second-order start, three `v0` spellings | **bit-identical** |
| the trajectory, on a shared factorization, 400 steps | **bit-identical**, forced and unforced |
| ... the acceleration cache with it | **bit-identical** |
| the trajectory, own factorizations | §28.5's three bars |
| `energy()`, shared solver, 200 steps | < 1e-14 relative — `np.dot` |
| `pressure()`, shared solver, 100 steps | < 1e-15 of `Σ|terms|` |
| an unforced free plate's monopole | cancels on both sides, 1e-16…1e-13 |
| `pickup_index_at`, ties included | identical — a **discrete** output |
| `to_live`, `state`, `displacement_at` | **bit-identical** |
| `grain_ratios_from_material`, 4 materials × 7 fields | **bit-identical** |
| `VKPlate` construction, both branches | **bit-identical**, bracket and Airy included |
| `_to_full`, `_to_live`, `_linear_rhs` | **bit-identical** |
| the von Kármán step, both factorizations shared, 120 steps | **bit-identical**, Picard count included |
| `VKPlate(nonlinear=False)` vs `Plate`, 150 steps | **bit-identical** *within* each implementation |
| the refusals — 26 for `Plate`, 12 for `VKPlate`, 4 materials | identical, message for message |
| a branch-only attribute (`B` on a free plate) | absent on both |

### 28.10 The success condition

* `cargo test --workspace` green, **debug and release** — 19 new native bars.
* `cargo fmt --all --check`, `cargo clippy --workspace --all-targets -- -D warnings` and
  `ruff check .` green — §25.8, in the batch's list rather than CI's.
* `pip install ./crates/physsynth-py` **before** any parity number is believed — §25.8a.
* `pytest tests/test_rust_parity_plate.py` green with the flag and without it — **183** tests.
* The whole-suite count reconciles exactly: 3,852 before, **4,036** after. 183 of the 184 new tests are the parity file; the 184th is `test_xdist_groups.py`, which is parametrised over the *glob* of test modules, so adding a file adds a test there. A count that did not reconcile would falsify §28.10's claim that nothing on the default path moved, which is why it is written down rather than eyeballed.
* `PHYSSYNTH_RS=1 pytest` green on the **whole suite**, not a chosen subset. That is §25.8a's
  discipline and it is load-bearing here in a way it was not for the last four batches: `plate` has
  more clients than any other model in the tree, and `tests/helpers.py` imports both classes, so
  every helper that builds a plate now builds a Rust one. The flagged CI step's file list is
  likewise **derived** rather than chosen -- `grep -l` over the plate constructors, minus what the
  batch-3 step already runs -- which is what added `test_web_backend.py`, `test_airbox_membrane.py`
  and `test_airbox_port.py` to it after the first draft had missed all three.
* The whole Python suite green on the default path — **4,035 passed, 1 skipped** (that skip is
  `test_rust_parity_banded.py`'s, which exists only on the Rust path and is not this batch's) — and
  **no shipped number moves at all**: the only Python edits are the swap blocks, `airbox`'s `splu`
  name, and a test helper.
* The whole Python suite green **under the flag**: **4,036 passed, none skipped, none failed**.

### 28.12 A red CI run, and a scar that *does* have a local repro

The batch went in green on everything above and came back red on two jobs, for one line.
`tests/test_rust_parity_plate.py` opened with a bare `import physsynth_rs`. The other **fifteen**
parity files reach the extension through `pytest.importorskip`, and the reason is that the default
gate does not build it: the sharded validation harness installs only the Python package, and the
shard-reconciliation step beside it runs a plain `pytest --collect-only`. A module-scope import is
therefore a **collection error** rather than a skip, so

* the shard the file lands in fails outright — 659 passed, 1 error;
* and the reconciliation step's `pytest --collect-only | grep -oE '[0-9]+ tests collected'` finds
  nothing at all, its `test -n "$total"` fails, and a second job goes red for the same line.

Two things make this shape worth writing down rather than just fixing.

**It is invisible on a development machine and it is ruff-shaped.** The extension is always
installed here, so every local run — including the two whole-suite runs §28.10 records — passes. And
the import does not look wrong: `ruff check --fix` *sorted* it into the third-party block, where it
sits between `numpy` and `pytest` looking exactly like the other two.

**Unlike §22.1's class, it has a local repro, and it is one command.** `pip uninstall physsynth-rs`
and the failure reproduces exactly, in seconds — collection errors, the reconciliation loop finds no
count. That is worth stating next to §22.2, which had to conclude the opposite about NumPy's CPU
dispatch: *check whether the local repro exists before concluding it does not.* With the extension
removed the suite collects 1,995 tests instead of 4,036 and the three shards sum to exactly that,
which is the whole reconciliation claim.

The fix is `importorskip`, and the pin is `test_ci_workflow.py::
test_every_rust_parity_file_guards_its_extension_import`, which reads all sixteen parity files and
refuses a bare import in any of them. That is §20.7's trade a third time — a scar written down as a
test outperforms the same scar written down as a paragraph — and it is the *fifth* time in eight
batches that CI was red for something a local check could have caught (§19.7, §20.7, §25.8, §27.7,
and now this).

### 28.11 What the next batch inherits

* **Left in the core: `connection` and `string_geometric`, then `airbox` and `analysis/`.**
  `connection` is the cheapest of them — it touches no private names on either of its two resonator
  families and its arithmetic is a rank-1 update it already spells explicitly.
* **Grep for re-derivation, not only for private names.** §28.2. A client that recomputes what a
  model computes is bound to it by any test that compares the two exactly, and no name search finds
  that dependency. `airbox.py` is now half-swapped by one name; its own port inherits the rest.
* **Ask whether a compared quantity is identically zero.** §28.3, and normalise a reduction by the
  sum of its absolute terms.
* **Read `n_iters`, not the amplitude.** §28.6. It is public, it costs nothing, and it is what
  actually predicts whether a nonlinear trajectory can be compared at all.
* **A class that builds its collaborators by module-global name is swapped further than it looks.**
  §28.4. The parity file must construct the `*Py` collaborators explicitly or it compares Rust to
  Rust.
* **A parity file must `importorskip` the extension.** §28.12, now pinned. And the wider habit it
  belongs to: before concluding a failure has no local repro, try removing the thing CI does not
  have.
* **The nonlinear step is where the speed is.** 2.99x, against 1.17–1.32x for a linear one. Every
  remaining model with an inner iteration — the geometrically exact string, the room-loaded plates —
  is in that regime.
* **A module-scope `from .plate import Plate` is a capture, and `connection.py` has one.**
  `tests/test_stability.py` now asserts `connection.Plate is plate.Plate`, which is §0's named
  failure mode made checkable. `airbox.py` needs no such entry — its import of the two classes sits
  under `if TYPE_CHECKING:`, so it captures nothing at runtime; what it captures is `splu`, which
  §28.2 rebinds and the bare-vs-loaded reduction tests already pin exactly.

---

## §29 Phase 5, batch 5, as built (2026-08-31) — the geometrically exact string, and an ordering that was never a rounding

`string_geometric.py` is ported: `GeometricString` (model #10), the **last of the four theta-scheme
strings** and the last model in `physsynth/core/` outside `connection`, `airbox` and `analysis/`.
Flipping it closes the bit-identity chain `portable.py` was written to protect — `sigma1 = 0`, `EA =
0` and `EA = T` now all compare one Rust class against another.

The batch's finding is about **a kind of divergence the migration had not met**. Every finding since
§14 has been about *which digits* two implementations produce: a reduction's order (§14.2, §18.2,
§27.2), a compiler's fold (§17.2), a library's CPU dispatch (§22.1), a solver's blocking (§24.2).
This one changes no digit at all. The two implementations of the sparse LU agree to a tolerance
either way; what the port got wrong on its first draft was **how much work the elimination does**,
by a factor of thirteen — and no bar in this project could have seen it, because the answers stay
right and the model merely gets slow.

### 29.1 Why this model, and not `connection` — §28.11's own estimate, corrected

§28.11 named `connection` as the next batch and "the cheapest of them", on the grounds that it
"touches no private names on either of its two resonator families". That is true and it is the wrong
instrument, which is the same complaint §28.2 had just made about a name grep. `connection` is
**polymorphic over its collaborator's type**, and that is a third door:

* `StringBodyBridge` is handed a `ModalBody`, a `RadiatedBody`, a `ReactiveRadiatedBody` — all Rust
  — **and** `airbox.RoomLoadedBody`, which is Python and three phases from its own port
  (`tests/test_airbox_port.py`, `tests/test_radiation.py`, `web/serialize.py` three times over).
* `StringPlateBridge` and `StringVKPlateBridge` are handed `Plate`/`VKPlate` **and** airbox's
  `_PlateSurface` / `RoomSuspendedPlate` wrappers, which reach the plate through `__getattr__`.
* And `tests/test_airbox_surface.py`, `test_airbox_dipole.py` and `test_airbox_vk.py` each assert
  `bridge.stability_margin == bridge_bare.stability_margin` **exactly** across that boundary —
  §28.2's anchor-to-an-unported-client, one level further out.

So a Rust bridge would have to either refuse the duck-typed wrappers (three test files red, and the
`bow.rs` precedent says refusing is the *right* answer to a mixed pair) or call back into Python for
its collaborator every step, which is the reentrancy shape §13.2 documents. It also has two problems
of its own: `_max_leapfrog_eigenvalue` is a **dense nonsymmetric eigensolver** (`np.linalg.eigvals`,
LAPACK `dgeev`) with the project's dependency list empty, and `_stability_margin` needs the plate's
sparse operators *read back out* of a Rust model to reassemble `G0`. And the payoff is ~zero: the
bridge is a handful of scalar operations per step wrapped around collaborators that are already Rust.

**The decision is to port `connection` after `airbox`, not before it.** One note for whoever does:
the eigenproblem is easier than it looks. `A = M^-1 S` with `M` diagonal positive and `S` symmetric
(the string's strain energy, the body's modal stiffness, and the spring's rank-1 block), so its
spectrum is real and equals that of the symmetric `M^-1/2 S M^-1/2`. `dgeev` becomes a symmetric
eigensolver plus §25.3's margin measurement on the raise/no-raise decision — a few hundred lines
rather than a research problem.

The general form, and it is the third instrument this migration has needed: **ask what a client is
polymorphic over, not only what names it reads.** §0 predicted `connection` would be blocked by
*private* names; it is blocked by *types*, and the two searches look nothing alike.

### 29.2 The finding: the first Group D model that factors on the hot path

Every earlier Group D model — `beam`, `plate`, `operators2d`'s Airy solve, `airbox` — factors
**once** at construction and back-substitutes per step. This one builds a fresh Newton Jacobian and
factors it *inside* the iteration. So §24's decision to leave `sparse_lu` in the natural column
order stops being free, and §24 wrote its own escape clause: *"every Group D matrix in this project
is a banded FDTD operator whose natural order already has none [no fill] to speak of ... if a later
model makes fill the constraint, an ordering goes in front of this, not inside it."*

This is that model, and not by a little. Its `3(N-1)` unknowns are stacked **by field** — all of `u`,
then all of `w`, then all of `v` — while the discrete-gradient force couples the three fields *at the
same cell*. In that order every coupling sits `N-1` columns off the diagonal and a left-looking
elimination fills the entire envelope between:

| | `nnz(L) + nnz(U)` at N = 32 / 64 / 128 | factor at N = 128 |
|---|---|---|
| SciPy (SuperLU + COLAMD) | 676 / 1,380 / 2,788 | 156 us |
| this crate, natural order | **2,311 / 8,743 / 33,895** | **2,068 us** |
| this crate, reordered by node | 629 / 1,301 / 2,645 | 58 us |

The reordering is `(u_i, w_i, v_i)` taken together — `string_geometric::interleave_perm` — and it is
a **closed form in `N`**, so no ordering heuristic was needed or wanted. That is §24's finding about
the beam's permutation being a closed form arriving from the other side of the ledger: there it meant
the reference's choice could be *predicted*, here it means ours does not have to be *searched for*.
It also beats COLAMD by about 5 % at every size, which the test asserts loosely (`<= 1.5x`) and on
purpose — SuperLU's ordering heuristic is a SciPy internal that a point release may change, and
§18.3 and §26.2 both say not to pin one.

**The reordering costs nothing, and that is a property of this model rather than a general one.**
Every operator on the update path is block diagonal by field (`A3`, `Gp3`, `Gm3`) or diagonal per
cell (the DG Jacobian), so each output entry is a reduction over one block's entries and the global
index order never enters a single sum. The permutation therefore lives *inside*
`SparseLu::factor_permuted`, and the residual, the state and the energy all stay in Python's
`[u; w; v]` order. Had one reduction crossed a block, the reordering would have been a trade rather
than a gift.

### 29.3 The corollaries

**a. The project's first Group D matrix that is not SPD, and a written justification that stopped
covering it.** `sparse_lu`'s `DIAG_PIVOT_THRESH` prefers the diagonal, and the reason recorded in
§24 is that "every one of them is symmetric positive definite ... for an SPD matrix elimination
without any pivoting is unconditionally stable". A discrete gradient is not the gradient of anything,
so this Jacobian is genuinely unsymmetric — which is exactly why the model uses a sparse LU and not
the banded Cholesky the rest of the family uses — and that sentence no longer applies.

What replaces it is measured, and the measurement had to be chosen carefully. Row-sum diagonal
dominance is a *poor proxy*: it is set by the **time resolution** and by nothing else — 8.06 at
`lam_long = 0.5`, 2.51 at 1.0, 1.10 at 4.0 and **0.285** at 8.0, where the matrix is not diagonally
dominant at all — while amplitude (10x) moves it only from 8.06 to 7.34 and the grid does not move it
at all. The observable that matters is `is_natural`, because the threshold compares the diagonal
against the largest candidate in its own *column*: over **854 Jacobians** spanning grid, amplitude,
mode and `lam_long` from 0.5 to 8, **no pivot fires at any of them**. That is what the native test
asserts.

**b. §19.2's branch rule, answered more gently than model #9 answered it — because a `max` is not a
sum.** The tension string's iteration count differed on 1,400 of 5,000 steps because a BLAS reduction
fed a `brentq` bracket. Here the convergence test is `max|r| <= newton_tol * max|Y_seed|`, and a
maximum is order-independent, so the reduction feeding the branch is reproducible by construction.
What still varies is *which side of the bar* one Newton step lands on, and the rate is set by
something new: **where the mean iteration count sits between two integers.** Measured over 20,000
steps —

| fixture | mean iters | flips |
|---|---|---|
| mode 1, amp 1e-3 | 1.00 | **0** |
| mode 4, amp 1e-3 | 1.97 | 36 |
| mode 1, amp 1e-2 | 1.70 | 293 |
| `lam_long = 2` | 1.50 | 475 |

— zero when the count is pinned at an integer, hundreds when it sits mid-way. And a flip costs about
two orders of *trajectory* agreement (2.5e-10 against 1.4e-12 at 20,000 steps) and **nothing at all**
on the energy (2.0e-12 either way), because *any* root of the discrete-gradient equation conserves
exactly — which is precisely why the model declines to gate uniqueness. §27.5's "a conserved quantity
is not a trajectory comparison", reached from the other direction: there the trajectory decorrelated
and the energy held; here the trajectory holds and the *branch* differs.

**c. The Armijo line search is dormant, and the measurement is the test.** Swept over grid, amplitude
and mode across 1,600 steps, the backtracking loop fires **zero** times: the discrete gradient is
smooth (no kink, unlike the barrier's `[eta]+`), so the seed is already inside the basin and a full
Newton step always decreases the residual norm. That is §16.6's hazard again — a safety net nothing
in the suite exercises is a safety net nothing has ever checked — so it is driven directly in
`crates/physsynth-core/tests/string_geometric.rs`. It is also what disposes of the one branch here
that *is* on a sum: `0.5 r.r` is `np.dot` on the Python side and not reproducible, and a branch that
never fires cannot flip.

**d. `portable.py` was not needed at all, and that is a first.** Every matrix on this model's update
path — `D2`, the three `L`s, the three `A`s, `A3`, `Gp`, `Gm`, `Gp3`, `Gm3` and the DG Jacobian —
arrives from SciPy already canonically ordered, measured at four grid sizes and asserted in the
parity file rather than assumed. So this is the only one of the four theta-scheme strings whose port
required **no Python-side edit** beyond the swap block. §18.2's biharmonic problem was specific to
`D2 @ D2`, and the module already applies `canonical` to that one itself.

**e. A fixture can be wrong in the *physics* rather than in the coverage.** §16.4 says a fixture may
fail to exercise what is being ported. This batch found the other failure: the first draft of the
native tests fixed `fs = 48 kHz`, which at `EA/T = 500` puts `lam_long` at **4.6** and **9.2** — past
the model's own documented cliff, where the Newton solve stops converging and the drift explodes by
fourteen orders. Three native bars went red and **all three were correct**; the port was not
involved. `c_long/c = sqrt(EA/T) ~ 22`, so the familiar transverse `lam = 0.5` silently means
`lam_long ~ 11`, and `LAM_LONG_WARN` exists to say so. The rule: **a fixture for a model with two
wave speeds must be built at the fast one**, and a red physics bar on a new fixture is a question
about the fixture before it is a question about the port.

**f. §19.7's YAML line continuation, a fourth time and from a fourth tool — and it never reached
CI.** Editing the workflow through a heredoc collapsed every backslash-newline pair in one `run:`
block into a single 376-character line, and a second edit left a literal backslash-n in another. Both
were caught locally and immediately by `tests/test_ci_workflow.py`, which §20.7 wrote *because* the
paragraph version of this scar kept failing to prevent it. Four occurrences, four different tools,
zero red CI runs since the test exists. That is the trade §20.7 proposed, now with a sample size.
(The same tooling then ate the heredoc that was writing *this section*, which is the joke telling
itself: the fix is to write a long document with a file-writing tool and not through a shell.)

**g. A shape is part of the interface.** `_chol_u` came back as the flat `3n` buffer the core stores,
with every value correct, where `scipy.linalg.cholesky_banded` returns `(3, n)`. §25.7's finding
(`ascontiguousarray` promoting a 0-d array) in a different disguise, and `np.array_equal` was the
only thing in the suite that would have noticed.

**h. §24.7's inverted `boundary` arms, and they are still counter-intuitive.** PyO3 wraps the
*default* expression, so `Some(None)` is "argument omitted" and a bare `None` is the caller's literal
`None`. Written the obvious way round, the constructor rejected every call that omitted `boundary` —
which at least fails loudly, unlike §24.7's original, which silently accepted `boundary=None`.

**i. A rejection's type is part of it.** `displacement_at` out of range must raise `IndexError` —
what `float(self.u[index])` raises — and not `ValueError`. Caught in review rather than by a test,
and now pinned by one.

**j. The derived CI list returned nothing new, and the step was added anyway.** §28.10's convention
is to derive the flagged file list by grepping the clients and subtracting what an earlier step
covers. Here that returns the empty set: Phase 3's batch 1 already flagged all six geometric test
files when it ported the banded **solver** these four models share. What changed is the *claim*, not
the files — there the six ran a Python model on a Rust banded solve, here they run a Rust model whose
Newton solve factors a sparse LU per iteration — so a red run means a different thing in the two
steps. Dropping the step because "the files are already covered" would be §28.12's empty-CI-job shape
through a fourth door.

### 29.4 The measured comparison

| | |
|---|---|
| every derived scalar, 10 fixtures | **bit-identical** (34 of them) |
| `x` | **bit-identical** |
| `D2`, `L_u`, `L_w`, `L_v`, `A_u`, `A_w`, `A_v`, `A3`, `Gp`, `Gm`, `Gp3`, `Gm3` | **bit-identical** — values, `nnz` **and stored order** |
| the three banded factors, shared solver | **bit-identical**, shape included |
| the second-order start, all six arrays | **bit-identical** |
| `_stretch_ratio`, `_stretch_terms`, `_dg_force`, `_dg_jacobian` | **bit-identical**, physical strains and the inverted-element arm alike |
| `_nl_density` | < 1e-14 relative — `np.sum`'s pairwise blocking, declined |
| the trajectory, both solvers shared, 2,000 steps | **bit-identical** — state, energy, `newton_iters`, `total_newton_iters` |
| `EA = T` vs `DampedStiffString`, 300 steps | **bit-identical** within each implementation, and across them |
| the trajectory, own solvers, 2,000 steps | 2.6e-13 ... 7.0e-13 of the running peak |
| ... at 20,000 steps | 1.4e-12, or 2.5e-10 where the iteration count flips |
| `energy()`, own solvers | <= 1.4e-13 relative |
| the lossless drift, both implementations | < 1e-10, the shipped bar, on every fixture |
| the refusals — 19 of them, plus softening | identical, message for message |
| `apply_Ainv`, `displacement_at` out of range | same exception **type** and text |
| the two `RuntimeWarning`s | raised by both, matched by `pytest.warns` |

### 29.5 Speed

| | Python | Rust | |
|---|---|---|---|
| `step`, N = 48, nonlinear | 1,902 us | 122 us | **15.5x** |

That is by a wide margin the largest per-step win the migration has measured — against 2.99x for the
von Karman plate, which held the record, and 1.17-1.32x for linear field models. §11.6's regime taken
to its limit: a Newton step is a dozen small NumPy and SciPy calls per iteration (three matvecs, nine
`sparse.diags`, a `bmat`, an `splu`, an Armijo trial) and every one of them is per-call overhead with
almost no arithmetic inside it. **The models with an inner iteration are where the real-time port
lives**, and this is the sharpest evidence yet.

### 29.6 The success condition

* `cargo test --workspace` green, **debug and release** — 14 new native bars.
* `cargo fmt --all --check`, `cargo clippy --workspace --all-targets -- -D warnings` and
  `ruff check .` green.
* `pip install ./crates/physsynth-py` **before** any parity number is believed — §25.8a.
* `pytest tests/test_rust_parity_geometric.py` green with the flag and without it — **151** tests.
* The whole-suite count reconciles exactly: **4,037** before (§28.10's 4,036 plus the one test the
  commit after it added), **4,189** after — 151 for the parity file and 1 for
  `test_xdist_groups.py`, which is parametrised over the *glob* of test modules. A count that did
  not reconcile would falsify the claim that nothing on the default path moved, which is why it is
  written down rather than eyeballed (§25.8a).
* `PHYSSYNTH_RS=1 pytest` green on the **whole suite**.
* The whole suite green on the default path, with **no shipped number moving**: the only Python edits
  are the swap block, `tests/test_stability.py`'s guard table, and the new parity file.

### 29.7 What the next batch inherits

* **Left in the core: `airbox`, then `connection`, then `analysis/`** — in that order, and the
  reordering is §29.1's finding rather than a preference. `connection` is polymorphic over its
  collaborators' *types* and three airbox tests pin an exact equality across that boundary, so it
  cannot move before its collaborators do.
* **Ask what a client is polymorphic over.** §29.1. The migration has now needed three different
  searches to find a blocking dependency: private names (§0), re-derivation (§28.2) and duck-typed
  collaborators (§29.1). None of the three finds the others.
* **A cost regression is invisible to every bar this project owns.** §29.2. The answers stayed right.
  If a port introduces an algorithmic choice — an ordering, a factorization strategy, a caching
  decision — the assertion has to be about the *work*, because nothing else will notice.
* **Ask whether the branch is on a sum or on a max.** §29.3b. §19.2 said to ask whether anything
  downstream branches on a reduction; the sharper question is what *kind* of reduction, because a
  maximum is order-independent and reproducible by construction where a sum is neither.
* **Read the mean iteration count, not the amplitude** — §28.11's rule, refined: what predicts a
  branch flip is not how large the count is but how far it sits from an integer.
* **Build a two-speed model's fixture at the fast speed.** §29.3e.
* **`airbox` is already half-swapped** (§28.2 gave it `splu`) and it is the largest file in the
  project: 3,976 lines and six factorizations, of which §11.2.1 predicts most of the arithmetic is
  not a solve at all. Expect it to port in halves.

## §30 Phase 5, batch 6, as built (2026-08-31) — the room, and a cutoff that makes exactness decidable

`airbox.py` is 3,976 lines, the largest file in the project, and §29.7 predicted it would "port in
halves." It does. This batch is the **first half**: `AirBox` itself, the 3-D room on a Yee grid.
The three tiers above it — `RoomPort`, `SurfacePort`, `InteriorSurfacePort` and the six
`RoomLoaded*` / `RoomSuspended*` wrappers — stay Python and keep working unchanged.

That makes it the first batch whose success condition is not only "are the numbers right" but
**"does the seam hold"**, and the seam turned out to be wider than any private-name grep this
migration has run.

### 30.1 The shape on disk

* `crates/physsynth-core/src/airbox.rs` — `Params` (validation, the trapezoid weights, the wall
  closure, the CFL gate), the kernels (`divergence`, `momentum`, `apply_cut`, `pressure_step`, the
  two injections, `apply_walls`, the two books, `acoustic_energy`, the modal oracle) and a native
  `AirBox` shell with ten `cargo test`s.
* `crates/physsynth-py/src/airbox.rs` — the binding. Larger than the core, and the reason is §30.3.
* `physsynth/core/airbox.py` — `AirBoxPy = AirBox` plus `AirBox = _rs.AirBox` under the flag, and
  **one edit to the reference**: `mode_shape` builds its cosines with `math.cos` (§30.6).
* `tests/test_rust_parity_airbox.py` — 35 tests.
* `tests/test_stability.py` — `airbox` added to *both* guard tables (§30.7).
* `.github/workflows/ci.yml` — a new "The room, unmodified, against Rust" step, and the parity file.

### 30.2 The finding: NumPy's pairwise reduction has a written-down cutoff, so exactness across a ported sum is decidable by counting terms

Every batch since §14 has asked some version of "will this reduction agree?", and the answers have
been probabilistic or empirical. §14.2 measured a BLAS `ddot` and concluded matching it would be a
claim about a runner. §23.2 asked "how long is the sum?" and answered *at two terms it is provable*
— because two doubles sum the same in either order unless they cancel. Both are statements about
the *values* being summed.

`AirBox` has four reductions and all four are `np.sum` rather than `np.dot`: the volume compliance
sum and three kinetic sums in `acoustic_energy`, the per-face wall flux in `step`, and a port
injection's `pbar`. And `np.sum`'s pairwise blocking has a **constant in it**:

```text
n < 8      ->  plain left-to-right loop
n <= 128   ->  eight accumulators, unrolled by eight, combined pairwise
n  > 128   ->  split at n/2 rounded down to a multiple of eight, recurse
```

Measured (2026-08-31, random positive vectors, this machine), against a plain left-to-right
accumulation:

| n | 4 | 6 | 7 | 8 | 16 | 56 | 560 | 4,641 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| disagreements | **0** | **0** | **0** | 803/2000 | 1,097/2000 | 1,398/2000 | 1,802/2000 | 1,944/2000 |

Below eight it is not "usually equal", it is **the same computation**. So for this class of
reduction the question "is bit-identity available?" costs no measurement at all — count the terms.
That is §23.2's move applied to a *blocking rule* rather than to a cancellation, and it is sharper,
because it does not depend on the values.

The corollary is what decides this batch's bar, and it is a hard no:

* **The volume sum is never below the cutoff.** The smallest room `AirBox` will build is one cell
  per axis, which is `2 x 2 x 2 = 8` pressure nodes. The model's own minimum sits exactly one node
  past the boundary, so `acoustic_energy` cannot be structurally exact for *any* room.
* **A wall face can be.** A face carries `(N+1)(N+1) >= 4` nodes, so a 1x1-cell face is below it —
  and measured over 2,000 steps at `N = (1,1,1)` the two `dissipated` ledgers are **bit-identical**,
  0 differing steps. At the 56-node face of the default fixture they are not.
* **A one-node port books exactly**, because its `pbar` is a sum of length one. Asserted.

So the bar is: the **field** is bit-identical and the **energy books** are a tolerance. That is
affordable here for a reason no earlier refusal had. §14.2's rule is "ask whether the reduction
reaches the next timestep"; `dissipated` and `injected` are pure bookkeeping — nothing in the update
path reads them, they exist only so `energy()` can close the ledger. The answer is **no**, for the
first time in this migration on a reduction that *is* fed back into a running accumulator and still
does not matter.

One number worth keeping honest, because a one-shot equality check will mislead. The two
`dissipated` books do not diverge once and stay apart; they **wander in and out of agreement at a
last bit**. Over 2,000 steps at `N = (9,7,6)`: one lossy wall differs on **59** steps and finishes on
exactly the same double; two lossy walls differ on 2,000 of 2,000 and finish 1.7e-16 apart; six
differ on 1,997. The first draft of this section's measurement script ran one configuration for 200
steps, read `identical=True`, and would have recorded a coincidence as a structural claim.

### 30.2a The association inside the accumulator, which was nearly a false comment

The first draft of `step` computed a per-step subtotal of the wall flux and added it once. The
reference books **per face**: with two lossy walls it forms `(D + f0) + f1` where the draft formed
`D + (f0 + f1)`. Different association on a running accumulator. The draft carried a comment
claiming the two were "the same sum in the same order", which was true of `injected` and false of
`dissipated` — §18.3's shape, a decision justified by a sentence that is not quite true.

Fixing it costs nothing (`apply_walls` takes `&mut f64` and books each face onto it) and it is
*measurable*: the final relative gap on the two-lossy-wall fixture fell from **2.2e-15 to 1.4e-16**,
an order of magnitude, and on the all-lossy fixture from 6.6e-15 to 1.4e-16. A tolerance-level
quantity is still worth getting right when the right version is free.

### 30.3 The seam: fifteen names, six of them written from outside — and the fifteenth is public

The ports and wrappers reach into the room through names the reference spells with a leading
underscore. Enumerated before a line of Rust was written:

| name | read by | **written** by |
| --- | --- | --- |
| `_w` | `_free_pressure_nodes` | — |
| `_W` | every port, `test_airbox_modal`, `test_airbox_port` | — |
| `_beta`, `_open`, `_has_walls` | `_free_pressure_nodes`, two test files | — |
| `_pending` | three test files | the same |
| `_pending_ports` | every port | every port, and `test_airbox_dipole` |
| `_ports` | the disjointness check | every port |
| `_cut_mask`, `_cut_index`, `_cuts` | `cut_faces`, `step` | `test_airbox_dipole::_uncut` |
| `_register_cut`, `_plane_axis` | `InteriorSurfacePort` | — |
| `_divergence` | `test_airbox_port`, `test_airbox_surface` | — |

That is §12.2 ("a leading underscore is not a statement about the interface") at its widest: three
phases after `body` found three modules assigning to a `_accel`, six containers here are Python
objects a client *writes*, so none of them can be mirrored in Rust. The binding holds them as plain
`Py<PyAny>` slots and reads them back every step.

**And the enumeration was still one name short.** `tests/test_airbox_freefield.py` relocates the
source with `box.source_index = box.node_index(centre)` — a **public** attribute, invisible to a
private-name grep, and it turned two tests red. The migration has now needed four different searches
to find a blocking dependency: private names (§0), re-derivation (§28.2), duck-typed collaborator
types (§29.1), and now **public attributes that are written rather than read**. None of the four
finds the others, and the general form is: *grep for assignment, not only for reference.*

The one seam claim worth stating separately is `_free_pressure_nodes`. It is a module-level Python
helper the ports share, it is unchanged, and it now reads a **Rust** room's arrays. Its docstring
already claimed that a local divergence read reproduces `_divergence()`-then-closure exactly, at
wall, edge and corner nodes alike — a within-language claim while the room was Python on both sides.
Porting the room turned it into a cross-language one. It holds, and it is asserted; note the
direction, which is §28.2 seen from the other end: there an unported client re-derived what the port
computed and the *anchor* broke, here the unported client re-derives it and the *port* is the
reference.

### 30.4 Two identity bugs, and what caught them

Neither was arithmetic and neither would have been caught by an energy bar.

**`_cut_mask` and `_cut_index` were the same list.** The constructor built one `PyList` of three
`None`s and used `Bound::clone` for the second field — which is a *reference* clone. So
`_register_cut`, which writes both, overwrote each with the other, and `cut_faces` (which counts
non-zeros in the mask) read an index tuple instead: **375 faces instead of 136**. It was caught by
`test_airbox_cut.py::test_cuts_are_additive`, an existing test, on the first flagged run — and it is
now pinned directly in the parity file, because the failure is a property of the *binding* that
nothing in the physics can see.

**`source_index` was not settable.** §30.3. Caught by `test_airbox_freefield.py`.

Both arrived from running the **existing, unmodified** suite under the flag, which is the whole
argument for plan §1's design.

### 30.5 §29.2's cost regression, in the batch that cites it — and this time it was in the binding

§29.7 wrote it down: *"a cost regression is invisible to every bar this project owns; if a port
introduces an algorithmic choice, the assertion has to be about the work."* One batch later, here it
is — and it was not an algorithmic choice, it was four `to_vec()` calls.

The first draft copied `p`, `ux`, `uy` and `uz` out of their NumPy buffers into fresh `Vec`s on
every step. Every answer stayed bit-identical. The speed did not:

| nodes | 27 | 120 | 560 | 4,641 | 33,825 | 274,625 |
| --- | --- | --- | --- | --- | --- | --- |
| copying draft | 15.1x | 9.8x | 5.7x | 1.49x | **0.31x** | 1.15x |
| borrowing | 13.1x | 10.7x | 4.2x | 1.45x | **1.09x** | 1.53x |

`0.31x` is **3.2x slower than NumPy**, reproducibly, at one room size (41 x 33 x 25 nodes) — four
array-sized memcpys per step crossing whatever allocator threshold that size sits on. The fix is to
hold read borrows of the four buffers across the whole step and copy nothing; it needs the
`&mut self` work (reading `_pending`, `_pending_ports`, `_cut_index`) hoisted above the borrow,
which is a small restructuring and no arithmetic change at all.

The lesson is narrower and more useful than §29.2's: **a binding's buffer discipline is an
algorithmic choice.** §9.3 established that state must live in Python-owned arrays; what this batch
adds is that *reading* them must borrow, not copy, and that the cost of getting it wrong is not a
constant factor but a cliff at one size.

The rest of the curve is §11.6 exactly: 13x on a tiny room where per-call overhead is everything,
converging to ~1.1-1.5x once NumPy's compiled loops dominate. A room is not where the real-time port
lives; §29.5's models-with-an-inner-iteration still are.

### 30.6 §22.3's portable spelling a sixth time, and the first aimed at an initial condition

`mode_shape` is a tensor cosine, `set_mode` seeds a run from it, and §22.1 established that NumPy
computes transcendentals with its own CPU-dispatched routines rather than the platform libm — so
`np.cos` makes a value a claim about which machine ran the job.

Everywhere that has bitten before, it cost a last bit in a **read-out**. Here it would seed an
initial **condition**: two implementations starting from different fields are not two roundings of
one run, they are two runs. So `airbox.py` now builds its three cosine vectors with `math.cos`,
where CPython and Rust meet at the same libm.

It is free — about sixty calls per room, against §25.4's two-million-point quadrature where the same
manoeuvre was **refused**. And it is invisible here: measured 0 differences in 2,239 values on this
Windows machine, where NumPy, CPython and Rust all reach UCRT. §22.2 said that whole class has no
local repro; this is the first time the migration has taken the portable spelling *purely on the
strength of the written-down finding*, with no local measurement able to justify it.

### 30.7 The two discrete outputs, and the guard that was one module short again

`N = int(round(L / h))` and `node_index` are **decisions**, not numbers (§25.2). Python's `round` is
half-to-**even**; Rust's `f64::round` is half-away-from-zero. At a tie the naive port builds a room
one cell larger on an axis, which conserves energy perfectly and reports a plausible spectrum — it
is simply a different room. Both go through `round_ties_even`, the scar `membrane` and `radiation`
already carry, and the parity test **searches** for ties rather than asserting a constant (§26.6):
of 35 exact ties in the search range, 17 have an even floor and are witnesses (half-to-even keeps
the value, half-away goes up), and all 17 are caught.

`_LAMBDA_MAX` is spelled `1.0 / 3.0_f64.sqrt()` rather than as a literal, so a fixture built exactly
at the CFL ceiling — which is deliberately allowed — constructs on both sides.

`self.c0 ** 2` is CPython's `float.__pow__`, i.e. libm's `pow`, and **not** `c0 * c0`: they disagree
in **99 of 200,000** sound speeds in the range this class accepts, and the quantity multiplies the
divergence at every timestep. It goes through `pyfloat::scalar_pow`, whose `#[inline(never)]` is what
keeps LLVM from folding it back into a multiply in `--release` (§17.2). The parity test searches for
a witness `c0` and drives 500 steps at it.

And the mechanical one: `airbox` had read `PHYSSYNTH_RS` since §28.2 gave it the `splu` swap, and was
**not** in `test_stability.py`'s `_USE_RUST` tuple — a whole batch in which its reading of the flag
could have diverged with nothing noticing. §17.6/§23.7/§26.7's finding for the fourth time: *the
derive is only as wide as the list it derives over.*

### 30.8 §19.7's line continuation, a fifth time from a fifth tool

Writing the new CI step through a Python heredoc collapsed every backslash-newline pair into one
500-character line. Same failure, fifth different tool, and `tests/test_ci_workflow.py` — the test
§20.7 wrote when the scar was reintroduced by the batch citing it — caught it in under a second.
**Five occurrences, zero red CI runs since the test exists.** The fix is to build the backslash from
`chr(92)` so no layer can eat it.

### 30.9 What is bit-identical

Everything on the update path, at every fixture measured:

* every construction product — `N`, `L_actual`, `lam`, `walls`, `_w`, `_W`, `_Wx`/`_Wy`/`_Wz`,
  `_beta`, `_open`, `_has_walls`, `source_index`;
* `p`, `ux`, `uy`, `uz`, `ux_prev`, `uy_prev`, `uz_prev` over 2,000 steps at **max |dp| = 0.0**, on
  all five wall configurations (rigid, one lossy face, two lossy faces, all lossy, an open face),
  with a cut room, with a driven source, with a hand-built port injection, and started from an exact
  discrete mode;
* `mode_shape`, `mode_frequency`, `continuum_mode_frequency`, `node_index`, `snapped`,
  `pressure_at`, `_divergence`, `cut_faces`, `_cut_mask`, `_cut_index`;
* `injected` for a scalar source **and** for a one-node port — both are sums of length one;
* `dissipated` for a room whose lossy face is below the eight-node cutoff.

Not bit-identical, by decision: `acoustic_energy` and `energy` (~1e-16 relative), `dissipated` for a
face at or above the cutoff (~1.7e-16 relative), and a spread port's booked work.

### 30.10 The measured comparison

| claim | fixture | result |
| --- | --- | --- |
| field, 2,000 steps | five wall configurations | max abs difference `0.0` |
| field, 2,000 steps | cut room, two cuts, one with an extent | `0.0` |
| field, 1,000 steps | driven soft source | `0.0`, `injected` **equal** |
| field, 500 steps | one-node port on `_pending_ports` | `0.0`, `injected` **equal** |
| field, 400 steps | from `set_mode`, four modes | `0.0` |
| energy | rigid | 9.8e-16 relative |
| energy | two lossy faces | 1.4e-16 relative |
| energy | all lossy | 1.4e-16 relative |
| energy drift, 1,000 steps | both implementations, all five walls | < 1e-10 (the standing bar) |
| `_free_pressure_nodes` | wall, edge, corner and interior nodes | `array_equal`, both directions |

### 30.11 Speed

See §30.5's table. 13.1x at 27 nodes, 4.2x at 560, ~1.1-1.5x from 4,641 nodes up.

### 30.12 The success condition

* `cargo test --workspace` — 25 test binaries green, including ten new native `airbox` tests and the
  dependency allowlist (still empty).
* `cargo fmt --all --check` and `cargo clippy --workspace --all-targets -- -D warnings` — clean.
* `ruff check .` — clean.
* `PHYSSYNTH_RS=1 pytest` over the ten airbox files plus `test_stability.py` — **450 passed**.
* `PHYSSYNTH_RS=1 pytest tests/test_web_backend.py` — green (the viewer builds `AirBox` for its
  `airbox` and `vkroom` models).
* `pytest tests/test_rust_parity_airbox.py` — **35 passed**.
* Default path unchanged: the same ten airbox files, `test_stability.py`, `test_shard_partition.py`
  and `test_web_backend.py` all green without the flag.

### 30.13 What the next batch inherits

* **What is left of `airbox.py`:** the ports (`RoomPort`, `SurfacePort`, `InteriorSurfacePort`) and
  the six `RoomLoaded*` / `RoomSuspended*` wrappers — roughly 3,000 of the file's 3,976 lines, and
  the half that owns all six `splu` factorizations. `AirBox` itself has none.
* **`connection` is still blocked, and §29.1's reason is only *half* discharged.** Its three bridges
  are polymorphic over their collaborators' types, and the Python collaborators they must still
  accept — `RoomLoadedBody`, `_PlateSurface`, `RoomSuspendedPlate` — are exactly the half of
  `airbox.py` this batch did not port. The order stays `airbox`'s second half, then `connection`,
  then `analysis/`.
* **Count the terms.** §30.2. For an `np.sum` the availability of bit-identity is decidable without
  measuring anything: below eight elements it is the same computation, at eight or above it is not.
  Ask it of every remaining reduction before writing a bar.
* **Grep for assignment, not only for reference.** §30.3. Four searches now, none of which finds the
  others: private names, re-derivation, duck-typed types, and public attributes that are *written*.
* **A binding that copies its buffers has made an algorithmic choice.** §30.5. Borrow the NumPy
  arrays for the whole step; the cost of not doing so is a cliff, not a constant.
* **`Bound::clone` is a reference clone.** §30.4. Two struct fields built from one `PyList::new` are
  one object, and the symptom is a wrong *count*, not a wrong number.
* **Two things this batch found and deliberately did not fix, both in already-shipped code.**
  * `crates/physsynth-core/src/lib.rs`'s module header still narrates the crate as far as Phase 3's
    fifth batch. It omits Phase 4 and all six Phase 5 batches, and it now declares `pub mod airbox;`
    with no prose behind it. That is §17.6/§23.7/§26.7/§30.7's shape — a hand-maintained list that
    has fallen behind — with one difference that makes it worse: there is no derive and no test
    watching it, so nothing will ever fire. Rewriting it is a batch's worth of prose; knowing it is
    stale is one line, which is this one.
  * `bore.rs:262` spells the compliance denominator `c0.powf(2.0)` with a **literal** exponent, and
    `reed.rs:229` and `:505` do the same. §17.2 established that LLVM folds exactly that into
    `c0 * c0` in `--release`, where CPython's `float.__pow__` calls libm's `pow` — so those three are
    a last bit on the state of every step at any `c0` where the two spellings differ. §30.7 measured
    that at **99 in 200,000** plausible sound speeds; it is invisible at the ambient 343 m/s because
    `343^2 = 117649` is exact. The fix is `pyfloat::scalar_pow`, which did not exist when `bore` was
    written. It is a re-tolerancing of two shipped models, so it belongs to a batch of its own.

## §31 Phase 5, batch 7, as built (2026-08-31) — the ports, and a refusal that was avoidable

`airbox.py`'s second half, first tier. §30 ported the room; this batch ports the three **ports** on
it — `RoomPort`, `SurfacePort`, `InteriorSurfacePort` — plus the module-level
`_free_pressure_nodes` they share. The six `RoomLoaded*` / `RoomSuspended*` wrappers above them are
still Python and are the next batch.

Not one line of the ten `test_airbox_*.py` files was touched. That is plan §1's design paying out
again, and it is why the CI step's file list is unchanged.

### 31.1 The shape on disk

* `crates/physsynth-core/src/reduce.rs` — **new**. NumPy's pairwise summation, transcribed, with six
  native tests. §31.2 is why it exists.
* `crates/physsynth-core/src/airbox_port.rs` — the kernels: the ball node set, the bilinear/nearest
  spreading operator, the plane-node gather, `T`, the per-node resistance, the triple product, the
  span-wise footprint count, and `free_pressure_nodes`. Nine `cargo test`s.
* `crates/physsynth-core/src/sparse.rs` — one new constructor, `Csr::from_rows_keeping_zeros`
  (§31.4).
* `crates/physsynth-py/src/airbox_port.rs` — the three `#[pyclass(dict)]`es and two module
  functions. Larger than the core, as `airbox`'s binding was, and for the same reason one tier up.
* `physsynth/core/airbox.py` — `RoomPortPy` / `SurfacePortPy` / `InteriorSurfacePortPy` /
  `free_pressure_nodes_py`, and four more names under the flag. **No edit to the reference.**
* `tests/test_rust_parity_airbox_port.py` — 58 tests.
* `tests/test_stability.py` — the three classes into the class table, `airbox` into the function
  table. The guard fired by name on the first flagged run.
* `.github/workflows/ci.yml` — the room's step renamed and its comment extended; the new parity
  file added to the parity step.

### 31.2 The finding: `np.sum`'s blocking is an algorithm, not a kernel — so §30.2's refusal was avoidable

§30.2 is one batch old and it is half wrong. Its measurement stands and is the sharpest thing in the
migration's toolkit: **below eight elements `np.sum` is a plain left-to-right loop**, so exactness
across a short reduction is free and decidable by counting terms, whatever the values are. Its
*conclusion* — that above eight, matching it "would be a claim about a library internal, and after
§22.1 a claim about the CPU as well" — is not true, and this batch needed it not to be.

NumPy's pairwise sum is thirty lines and fully determined:

```text
n < 8      ->  plain left-to-right loop from 0.0
n <= 128   ->  eight accumulators SEEDED from a[0..8], unrolled by eight, combined
               ((r0+r1) + (r2+r3)) + ((r4+r5) + (r6+r7)), then the ragged tail left to right
n  > 128   ->  split at n/2 rounded DOWN to a multiple of eight, recurse, add the halves
```

Transcribed, it reproduces `np.sum` **exactly**: 0 disagreements in 2,000 random vectors at each of
n = 1, 4, 7, 8, 9, 15, 16, 20, 30, 56, 128, 129, 200, 560 and 4,641; 0 in 200 at n = 40,000; 0 in 200
for whole-array sums of 3-D arrays at five shapes up to 41x33x25; and 0 in 200 for a **strided**
reduction, which is a different code path.

**Why this is not §22.1, said plainly, because the two look alike.** §22.1's hazard is that NumPy
computes `pow`, `sin` and `exp` with its own routines, *dispatched at import from the CPU's feature
set* — two machines, two instruction selections, two last bits, and nothing in the source of either
language shows it. A summation has no comparable freedom to exercise. The order is fixed by the
blocking above, and the unroll by eight exists precisely so a vector unit can be used **without
changing that order**. A transcription is a claim about an algorithm; §22.1's would have been a
claim about an instruction selection. It is also not §14.2's bargain: BLAS `ddot` *fuses* its
multiply-add and OpenBLAS picks its kernel by CPU, so there is no scalar recipe at all. Here there
is one.

**Why it mattered here and did not matter there.** §14.2's question — does the reduction reach the
next timestep? — was answered *no* for the room, whose two energy books are pure bookkeeping. It is
answered **yes** three times over in this tier: `w = W / W.sum()` is the share of the volume
velocity each node receives, `R_room` is what the coupled solve divides by, and `free_pressure` is
the pressure the body is pushed by. A last bit in any of them is a different trajectory. So the
choice was transcribe or give up exactness on the update path, and the prediction going in — a
point port exact, a ball port a tolerance — turned out to be avoidable. **Every port size is
bit-identical.**

The risk is stated rather than buried. "NumPy's blocking is the same on every machine" is the
riskiest sentence in the batch, and `test_numpy_pairwise_blocking_is_an_algorithm_not_a_kernel` is
the single place it is asserted, with `_pairwise_sum` exposed from the crate for no other purpose.
If it is ever false on a runner, that named test says so — instead of §22.1's morning, where
eighteen exact assertions went red at once with no diagnosis.

Two mis-transcriptions are worth recording because each is invisible at the lengths the other is
wrong at, and one of them is a **non**-difference. The ragged tail folds into the *combined* result
and not back into the accumulators — a real distinction, asserted. Starting the eight accumulators
at zero instead of seeding them from `a[0..8]` is **not** a distinction: `0.0 + x` is exactly `x`,
so the zeroed variant's first block reproduces the seeding step for step. That was written first as
a test that could never fail, and it went red having searched for a difference that does not exist —
§23.5's empty search, once more inside the test written to catch that class.

### 31.3 The association: a diagonal is not neutral, and the blind fixture is provably blind

`load_matrix = (T.T @ diags(R) @ T).tocsr()` is a sparse contraction, so after §26 and §27 it is
three questions. Measured over the fixtures the suite's own builders make — §27.2's method,
enumerate rather than sample:

* **Stored order**: not an issue here, which is unusual for this migration. `T` comes from
  `coo_matrix(...).tocsr()` and the product ends in a CSC-to-CSR conversion; both canonicalize, and
  every row of every fixture came back ascending. `portable.canonical` was not needed at all, and
  `Csr::from_rows`'s own sorting is already right. That is the first time in Phase 5 that the answer
  to §18.2's question was simply *nothing to do*.
* **Values**: an ascending-`k` accumulation reproduces SciPy's kernel bit for bit — 0 differing
  entries of 6,845 over five fixtures, §26.2 holding again.
* **Association**: live, and the reason the first two are easy is not that this one is. `diags(R)`
  sits *between* the two factors and Python left-associates, so SciPy forms `(T_ki R_k) T_kj` and
  not `T_ki (R_k T_kj)`. Those are different doubles in **2,028 of 6,845** entries — 30%.

So §26.5's question ("do the outer factors share a mantissa, in which case the association is
invisible?") has to be asked of a *diagonal sitting inside a product*, not only of a bracketing. And
the fixture that cannot see it is worth naming, because it is one of the six goldens
`tests/test_airbox_dipole.py` pins: with `spreading="nearest"` each surface node lands on exactly
one air node with weight 1, so **every stored entry in a row of `T` is the same uniform node area**
— and `(x d) x` is `x (d x)` **identically**, for every `x` and `d`, because the two outer factors
are not merely commensurate, they are the same number (0 differences in 200,000 random pairs,
against 69,943 when the factors differ). §16.4's blind fixture for a fourth time, and the first with
a proof instead of a measurement.

### 31.4 Two SciPy routines in one expression disagree about whether a stored zero survives

`coo_matrix(...).tocsr()` **keeps** an explicit `0.0`. `csr_matmat` **prunes** one — it writes an
entry only `if (sums[head] != 0)`. Both are called by the port's construction, four lines apart.

The reference is explicit that this matters: `_spread` drops entries whose *geometric* weight is
zero and keeps entries whose node **area** is zero, "so a zero-area surface still names the nodes it
covers and the `T = 0` reduction to the bare resonator stays exercisable". So on a surface with
zero-area nodes the reference's `T` carries stored zeros and its load matrix does not — 182 and 91
stored entries where a uniform treatment gives 182 and 208.

`Csr::from_rows` drops zeros, which is right for every matrix this crate had assembled until now (a
structural zero and a dropped one are the same operator, and the sparser storage is free). The port
needs both behaviours, so `Csr::from_rows_keeping_zeros` was added and `T` uses it while the load
matrix does not. **No fixture the suite builds contains an explicit zero**, so nothing measured this
and every physics bar passes either way — §16.4 again, this time in the library rather than in the
model. The parity file constructs a half-zero-area surface on purpose.

### 31.5 Why the ports go before the wrappers, which is not the dependency argument

The obvious reason is that a wrapper holds a port. The real one is §13.2. A wrapper's `step` calls
`port.free_pressure()`, solves, then calls `port.inject(q)` — it hands control **out** twice per
step. A Rust wrapper over a Python port is therefore a `&mut self` pymethod that must release and
re-take its own state mid-step, which is exactly what `bore` and `reed` paid for and exactly what
PyO3 refuses. A Python wrapper over a Rust port is the ordinary direction. **Take the callee first**
is the general form, and it is worth carrying into the wrapper batch, whose own callers
(`connection`'s three bridges) sit above it in the same relation.

The second reason is what made this batch cheap: the port tier owns **no factorization**. All six
`splu` calls in `airbox.py` are in the wrapper tier, so nothing here is in §4's sparse-LU risk group
and every claim above can be exact. The next batch will not have that.

A port also reads its room through **Python attribute access and Python method calls**, never
through the Rust room's `Params` — `room.node_index(at)`, `room._plane_axis(plane)`,
`room._register_cut(...)`, `room._ports.append(self)`, `room._pending_ports.append(...)`. Three
reasons, in increasing order of what they cost to get wrong: a port must accept `AirBoxPy` *and*
`_rs.AirBox` (§29.1's duck typing, one tier down, and asserted); the refusals are the room's to
write, so calling `node_index` gets "outside the room" exactly right for free in whichever
implementation the caller has; and `_register_cut` **mutates** the room, so re-implementing it here
would be a second writer of `_cut_mask` — §30.4's bug through another door.

### 31.6 The seam from the other side: a test replaces the port's *methods*

§30.3's rule was *grep for assignment, not only for reference*, and applied to this tier it finds
four attributes a test writes on a port — `T`, `load_matrix`, `R` and `areas`, all replaced wholesale
with arbitrary SciPy or NumPy objects across five test files to switch a coupling off or halve it —
plus `_queued_at`, which `AirBox.set_state` resets on every registered port. All five are `Py<PyAny>`
slots with a setter, and `net_area` reads `areas` **live** so that zeroing it works.

And one door further on, which no attribute search finds:
`test_a_sign_flip_is_invisible_to_every_energy_quantity` **replaces the port's methods on the
instance** — `port.free_pressure = lambda: tuple(reversed(free()))` — and the wrapper then calls the
lambda. A `#[pyclass]` has no `__dict__` and refuses that outright, so all three classes carry
`dict`. The precedence works because CPython's lookup rules do not care which language defined the
class: a `#[getter]`/`#[setter]` pair is a *data* descriptor and beats the instance dict, while a
`#[pymethod]` is a *non-data* descriptor and loses to it. So `port.T = x` still runs the setter and
an assigned lambda still shadows the method, which is exactly what is wanted.

The general form is the sixth entry on §30.3's list, and it is not "grep harder": **ask what a
client does to the object, not only what it reads off it** — and *replacing a method* is a thing
Python clients do that a port to a compiled language does not support by default.

Attributes nothing writes — `nodes`, `w`, `R_room`, `index`, `_flat` — are exposed get-only, so an
assignment raises `AttributeError` rather than leaving a cached Rust index vector disagreeing with
the array the room is handed. A deliberate narrowing of the reference, in the loud direction.

### 31.7 §24.7 again, in the arm the plan had already written down

`spreading` has a default and a value the reference rejects by name, so an omitted argument and an
explicit `spreading=None` are different calls — and PyO3 collapses them. `beam` hit this in §24.7,
the plan recorded that the fix's arm order is inverted from the obvious guess, and this batch got it
wrong anyway: PyO3 wraps the **default expression**, so with `Option<Option<_>>` it is `Some(None)`
that means "omitted" and a bare `None` that is the caller's literal. Written the obvious way round,
eleven tests in `test_airbox_surface.py` went red — because their fixture forwards a `spreading=None`
default. Caught immediately and cheaply; recorded because a written-down finding did not prevent it,
which is a fact about how the finding was written rather than about PyO3.

### 31.8 What is bit-identical

Everything, at every fixture measured. There is no tolerance in this batch.

* **Construction**: `index`, `nodes`, `_flat`, `w`, `R_room`, `volume`, `node_count`, `radius`;
  `coords`, `areas`, `origin`, `n_surface`, `in_plane_axes`, `_where`, `footprint_empty`,
  `net_area`, `R`, `nodes_lo`, `nodes_hi`, `_in_plane`, `face_count`, `blocked_area`; and `T` and
  `load_matrix` **as stored** — `indptr`, `indices` and `data`, so structure, order and values.
* **The field**: `max |dp| = 0.0` over 2,000 steps of read-solve-inject on four wall
  configurations at a point port and at a 123-node ball; over 1,000 steps through a wall-mounted
  patch with both spreadings; and over 1,000 steps through an interior patch's `-q`/`+q` pair
  across its own cut. The loop is **closed** — the injection is `amp - pbar_free / (4 R)`, the
  scalar Thevenin solve a wrapper does — which is what makes these comparisons about the *port*.
  The first draft injected a prescribed function of the step index, and would have produced an
  identical field even if `free_pressure` had returned garbage: it would have been §30's comparison
  of the room, made again. A closed loop also has to be scale-free — a fixed termination impedance
  divided by a 123-element pressure vector made the feedback gain grow with the node count and
  overflowed, which is why the divisor is the port's own `R`.
* **The read-out, step for step**: `free_pressure()` compared at every one of 400 steps, not only
  at the end — it is the quantity the coupled solve consumes.
* **The books**: a port's `injected` at **one node** — a sum of length one, exact on any spelling.
  At 123 nodes it is **not**, and that is the batch's one tolerance and belongs to the *room*:
  `AirBox.step` books `np.sum(w * 0.5 * (p_next + p_old))` over the port's nodes, and `airbox.rs`
  was written one batch before `reduce` existed and still books it with a plain loop. Measured
  1.6e-16 relative under an open-loop drive and **exactly 0.0** under the closed loop — §30.2's
  "they wander in and out of agreement at a last bit", so the bar is a bound and not a difference.
  Taking §31.11's parked tightening would make it exact at every size.
* **The cut**: `cut_faces`, `_cut_mask` and `_cut_index` after an interior port registers.
* **The shared helper**: `_free_pressure_nodes` Python against Rust at wall, edge, corner and
  interior nodes on four wall types — and, separately, the Rust helper against the Rust room's own
  `_divergence()`-then-closure, because porting the helper turned that docstring claim from
  cross-language into within-Rust and it had to be re-asserted rather than inherited (§23.6, a
  sixth door: the Python spelling is kept alive under `free_pressure_nodes_py` for exactly this).

### 31.9 Speed, and a curve that does not converge

The port tier is the first thing in this migration whose work is **not proportional to the array it
reads**, and the speed curve says so. Per step, read-solve-inject with the room's own step included,
and then `free_pressure()` alone:

| room nodes | port nodes | whole step, Python | Rust | x | `free_pressure` Py | Rust | x |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 27 | 1 | 77.0 us | 8.5 | **9.0** | 41.5 us | 2.8 | **14.7** |
| 560 | 1 | 90.4 | 15.9 | 5.7 | 36.4 | 3.0 | 12.1 |
| 560 | 123 | 103.1 | 24.3 | 4.3 | 44.4 | 7.0 | 6.3 |
| 2,856 | 123 | 134.3 | 42.9 | 3.1 | 41.1 | 8.0 | 5.2 |
| 15,225 | 257 | 293.7 | 233.2 | 1.3 | 56.1 | 14.7 | 3.8 |
| 41,615 | 257 | 635.3 | 492.4 | 1.3 | 43.3 | 11.9 | **3.7** |

The *step* column is §11.6 exactly — 9x where per-call overhead is everything, converging to ~1.3x
once NumPy's compiled loops own the room's update. The `free_pressure` column does **not** converge:
it flattens at ~3.7x and stays there, because neither implementation's cost grows with the room. The
rule §11.6 gave was "the win is per-step overhead, and it disappears once the compiled kernel
dominates"; the missing clause is *once the compiled kernel dominates **the same work***. An
`O(patch)` read into an `O(room)` array never reaches that point.

Which is also the sharpest available form of §30.5's lesson. The room's copying draft cost a
constant factor because its copy was proportional to its work. A copying **port** binding would pay
a cost proportional to the *room* for work proportional to the *patch*:

| room nodes | `free_pressure` (borrowing) | what four `to_vec()`s would cost | ratio |
| --- | --- | --- | --- |
| 27 | 2.78 us | 1.15 us | 0.4 |
| 2,856 | 6.92 | 3.28 | 0.5 |
| 15,225 | 11.94 | 12.49 | 1.0 |
| 41,615 | 17.76 | 26.80 | 1.5 |
| 139,995 | 12.85 | **678.18** | **52.8** |

So the penalty for getting the buffer discipline wrong is not §30.5's cliff at one size, it is
**asymptotic** — and every answer would have stayed bit-identical throughout. §29.2's rule ("if a
port introduces an algorithmic choice, the assertion has to be about the work") with the strongest
instance the migration has produced. The last row carries §30.5's cliff on top of the asymptotics:
26.8 us to 678 us for a 3.4x larger copy.

### 31.10 The success condition

* `cargo test --workspace` — 25 test binaries green, including six new `reduce` tests, nine new
  `airbox_port` tests and the dependency allowlist (still empty).
* `cargo fmt --all --check` and `cargo clippy --workspace --all-targets -- -D warnings` — clean.
  Both were red first and both were run before the batch was called done, which is §25.8's scar
  holding for the third batch running.
* `ruff check .` — clean.
* `PHYSSYNTH_RS=1 pytest` over the ten airbox files plus `test_stability.py` and
  `test_web_backend.py` — **893 passed**.
* `pytest tests/test_rust_parity_airbox_port.py` — **58 passed**.
* Default path unchanged: the same files green without the flag.
* `pip install ./crates/physsynth-py` before believing any of it. §25.8a bit again — the first run
  of the new parity file failed on a missing `_pairwise_sum` that had been compiled but not
  installed, and nothing in the suite can tell a stale wheel from a fresh one.

### 31.11 What the next batch inherits

* **What is left of `airbox.py`:** the six `RoomLoaded*` / `RoomSuspended*` wrappers and the four
  private surface adapters (`_PlateSurface`, `_MembraneSurface`, `_VKPlateSurface`, plus the two
  mixins) — about 1,750 lines of code, and **all six `splu` factorizations in the file**. That makes
  it a Group D batch under §24's measured-tolerance rule, unlike this one.
* **`connection` is now blocked on exactly one thing.** §30.13 said its three bridges must still
  accept `RoomLoadedBody`, `_PlateSurface` and `RoomSuspendedPlate`, which are precisely the
  wrapper tier. One correction to §29.1 worth having: `connection.py` contains **no** `isinstance`,
  `hasattr`, `getattr` or `type(` at all — it is pure duck typing. So "polymorphic over its
  collaborators' types" means it calls methods, not that it discriminates, and no module-global name
  swap is needed to satisfy it. The blocker is only that the objects must exist and answer.
* **Take the callee first.** §31.5. Not dependency order — §13.2. A Rust caller that hands control
  back to Python mid-step cannot hold `&mut self` across the gap.
* **`reduce::sum` exists now.** §31.2. Use it wherever a ported `np.sum` reaches the next timestep,
  and keep the plain loop where it does not. Two tightenings are deliberately **left undone**, both
  in shipped code, both listed here rather than done silently:
  * `crates/physsynth-core/src/airbox.rs` still books `dissipated` and `injected` with a plain loop
    and still computes `acoustic_energy` with one, so §30.9's "~1e-16 relative" tolerances stand.
    They could now all be exact. It is a re-tolerancing of a shipped model's parity file, which
    §30.13's own precedent says belongs to a batch of its own.
  * §30.13's other parked item is unchanged: `bore.rs:262`, `reed.rs:229` and `:505` spell the
    compliance denominator `c0.powf(2.0)` with a **literal** exponent, which LLVM folds to `c0 * c0`
    in `--release` where CPython calls libm's `pow` (99 in 200,000 plausible sound speeds differ).
    The fix is `pyfloat::scalar_pow`. Still a batch of its own — and now there are two of them,
    which together would make one.
* **The wrappers construct their ports by module-global name**, so §28.4's trap is already
  loaded: today that is why the flagged suite exercises Rust ports through Python wrappers and it
  is correct, but the moment the wrappers port, a `RoomLoadedPlatePy` will silently hold a **Rust**
  port unless the parity file builds `RoomPortPy` explicitly. That is verbatim §28.4's "the parity
  file's Python `VKPlate` was holding a Rust Airy solver", queued one batch in advance.
* **Ask what a client *does* to the object.** §31.6. Reading and writing attributes are two
  searches; **replacing a method** is a third, it is what a Python test does when it wants to invert
  a sign convention without touching the model, and a compiled class refuses it unless it was built
  with `dict`.
* **`crates/physsynth-core/src/lib.rs`'s module header is still stale**, as §30.13 recorded, and now
  by two more modules (`airbox_port`, `reduce`). Nothing watches it. Unchanged, and still one line
  to know.

## §32 Phase 5, batch 8, as built (2026-08-31) — the room's resonators, and a tier that cannot own its arithmetic

`airbox.py`'s third and last slice. §30 ported the room, §31 the three ports, and this batch ports
what sits on top of them: `RoomLoadedBody`, `RoomLoadedPlate`, `RoomSuspendedPlate`,
`RoomLoadedVKPlate`, `RoomSuspendedVKPlate` and the two seams (`_PlateSurface`, `_VKPlateSurface`)
the four plate wrappers drive. The membrane wrappers stay Python — nothing in `connection.py` needs
them and no exact anchor binds them to these classes.

Not one line of the ten `test_airbox_*.py` files was touched, for the third batch running, and the
flagged CI step's file list has not changed once across the three tiers.

### 32.1 The shape on disk

* `crates/physsynth-py/src/airbox_wrap.rs` — **new**, and the first ported module with **no core
  half at all**. §32.2 is why.
* `physsynth/core/airbox.py` — seven `*Py` aliases and seven names under the flag. **No edit to the
  reference.**
* `tests/test_rust_parity_airbox_wrap.py` — 107 tests.
* `tests/test_stability.py` — the seven classes into the class table.
* `.github/workflows/ci.yml` — the new parity file, and the flagged step renamed.

### 32.2 The finding: a tier below can decide what "porting" this tier is allowed to mean

Every ported module until now holds its own state and does its own arithmetic. This one does
neither, and the reason is not a judgement call — it is an interface decision the tier *below* made
deliberately one batch earlier, and it propagates upward.

§31 stored a port's `T`, `R` and `load_matrix` as plain `Py<PyAny>` slots rather than as crate
types, because **eight tests replace them wholesale** to switch a coupling off, halve it or flip
its sign, and two more replace the port's *methods* on the instance. That decision was correct
there and it is binding here: a wrapper that cached any of those would read a matrix a test had
already replaced, and every one of those tests would pass having compared a loaded plate with
itself — §23.6's emptied comparison through a seventh door. It is not only the ports. Three tests
assign `inst._lu_loaded = splu(a)`, replacing the factorization, and two call `inst._surface.rhs()`
and `.a_bare()` directly. The factorization, the seam and the port are all objects this tier
**holds and calls**, never things it is.

So the line falls in a place no earlier batch had to draw:

* **Through Python:** every sparse product (`B @ u`, `T.T @ pbar`, `load_matrix @ u`), the assembly
  of `a_loaded`, the factorization and its `solve`, `np.dot`, and every call on a port, a seam or a
  model.
* **In Rust:** the control flow, the guards, the ledgers and the elementwise arithmetic between
  those calls — exact by construction, because elementwise `+ - * /` on doubles admits no
  reassociation, which is what lets the five-term theta-scheme right-hand side be a Rust fold over
  slices rather than five NumPy temporaries.

The general form, and the question to ask before scoping a batch that sits on top of a ported one:
**what did the tier below promise its clients it would let them replace?** A port that stores its
coupling as substitutable Python objects has decided that its callers compute through Python. No
name grep finds this — the six searches this migration has needed are now private names,
re-derivation, duck-typed types, written public attributes, replaced methods (§31.6) and now
**replaced collaborators**, and none of them finds the others.

Two consequences are worth separating, because one is good and one is a cost.

The good one: **everything in this batch is bit-identical, including the von Karman wrapper's
trajectory**, which §27.5 says should be impossible for a nonlinear plate over any useful run. It
is possible here because the two implementations are not two discretizations: they are one Picard
loop, called through one factorization, over one set of SciPy kernels. What would separate them is
a transcription difference and nothing else — so the parity file's exactness is a *sharp* test of
the port rather than a statement about the dynamics.

The cost: there is no arithmetic left for Rust to win. §32.4.

### 32.3 §31.11's Group D prediction is wrong, and the correction is a rule

§31.11 wrote that this would be "a Group D batch under §24's measured-tolerance rule, unlike this
one", on the grounds that **all six of the file's `splu` factorizations are in this tier**. That is
true and it does not follow. The wrapper does not *own* a factorization, it *calls* one — `splu` is
read as a module global at construction, exactly as the reference reads it, so both languages'
wrappers factor the same matrix with the same routine and there is no solver difference to
tolerance at all. `test_the_shared_factorization_changes_nothing` puts both sides on the crate's
sparse LU instead of SuperLU (§24.4's manoeuvre, fifth use) and the trajectory does not move,
because it never could: this is the first use of that manoeuvre as a **negative control**.

So: **a solver group is a property of ownership, not of the file a factorization appears in.** Ask
who *implements* the solve, not what is downstream of one. Counting `splu` call sites, which is
what §31.11 did, counts the wrong thing.

### 32.4 The speed, which is the price of §32.2 and is stated rather than buried

Measured on the flagged composition, five interleaved rounds, minimum per arm:

| model | live nodes | Python | Rust | ratio |
| --- | --- | --- | --- | --- |
| linear plate, baffled | 49 | 132.8 us | 117.8 us | 1.13x |
| linear plate, baffled | 225 | 171.3 us | 151.8 us | 1.13x |
| linear plate, baffled | 961 | 613.0 us | 578.5 us | 1.06x |
| von Karman plate, baffled | 49 | 716.5 us | 725.0 us | 0.99x |
| von Karman plate, baffled | 225 | 1479.2 us | 1001.6 us | 1.48x |

The last two rows are not a measurement of anything: over nine interleaved rounds the *Python* arm
alone spans 716-1234 us at 49 nodes and 1479-1998 us at 225, so the machine's own drift is larger
than the effect and the median-to-median ratios (0.93x and 1.01x) point the other way from the
min-to-min ones. The honest report is **the linear wrapper is a small consistent win and the von
Karman wrapper is indistinguishable from neutral**, which is §11.6's rule with the calls still
present: what a port buys is per-call overhead, and this tier's calls did not go away, they changed
language.

That is the expected outcome of §32.2 rather than a failed batch — the alternative was ten tests
that assert nothing. But it sharpens §29's headline, which needs a clause it did not have. §29
measured **15.5x** on a model with an inner iteration and concluded that models with inner
iterations are where the real-time port lives. True, *and only if the iteration's body is ported
with it*: a ported **caller** over unported callees multiplies the boundary crossings by the sweep
count instead of eliminating them. The first draft of the Picard hook made that mistake in the
small — it folded the two `mu`-averages in Rust, which cost a copy in and a copy out per sweep, and
that alone read 0.91x; keeping them as NumPy objects (three calls, no copies) recovered it. Two
notes from the same tuning, both cheap and both worth having:

* `vec1` originally went through `np.ascontiguousarray` unconditionally, which is an **import and
  two calls per extraction** and seven extractions per step. A downcast fast path costs nothing and
  is what the reference effectively gets for free, because it reaches NumPy through operators that
  are already C where a port reaches it through the interpreter.
* A timing pair measured once, apart, is not a comparison on this machine. Interleave the arms and
  take a minimum, and say so.

### 32.5 The anchor decided the batch's shape, and the grep that found it is one line

§15.2's rule again, and the fifth time it has set a batch boundary. `test_airbox_vk.py` asserts
that `RoomLoadedVKPlate(nonlinear=False)` reproduces `RoomLoadedPlate` **`array_equal`** on both
stored levels, on the coupling ledger and on the energy, for 50 steps, across both tiers and both
boundaries. So the four plate wrappers are one unit and cannot split — which reversed the plan the
batch started with (body and the linear plates now, von Karman with the membrane later). The
membrane wrappers, by contrast, are anchored only against a *bare* `Membrane`, so they are
separable and are deferred.

The structural answer, which is §28.4's one tier down: both seams share one `assemble_a_bare`, and
the von Karman seam's linear half is the model's own `_linear_rhs`, so the reduction is exact by
construction rather than by two transcriptions kept in step by docstring.

### 32.6 A `#[pyclass]` getter is the opposite default from `__getattr__`, and the source says so

These wrappers are **drop-ins for the models they hold**: `connection.py`'s three bridges reassemble
a plate's coupling block out of `n_live`, `u`, `u_prev`, `k`, `boundary`, `Lx`, `Ly`,
`pickup_index_at` and a dozen more, and the reference's own comment reads *"NOTHING here may shadow
a name that bridge reads"*. In Python that warning is nearly free, because `__getattr__` fires only
on a miss. In PyO3 it is the opposite: a getter is a data descriptor on the type and **beats both
the instance dict and `__getattr__`, permanently**. Every getter added to one of these classes is
therefore a name taken away from the model, silently, and no physics bar and no `cargo test` can
see it.

So the exposed set is exactly the reference's instance attributes plus its four overrides. Two
tests guard it and the second is the one that matters: `test_every_delegated_name_reaches_the_model`
walks `connection.py`'s list **by name**, which documents the contract and only catches names
somebody thought to write down, while `test_the_wrappers_own_surface_matches_the_reference`
**derives** it — `dir(instance)` on both implementations, asserted equal — so it fires on the *next*
getter added to any of these classes whether or not the name it shadows was ever listed. That is
§17.6 and §23.7's lesson a third time: a derive catches the paste a list forgets. (Compared on
*instances*, not classes: the reference sets these in `__init__` where PyO3 makes them type
descriptors, so the class-level sets differ by construction and only the instance-level sets are the
contract.) A companion, `test_no_model_name_is_unreachable_through_the_wrapper`, walks `dir(plate)`
from the other end.

Two details of the interface that are easy to leave implicit and are not: the bridges call
`body.step(force=F)` and `plate.step(f_ext=...)` **by keyword**, so the parameter *names* are part
of the contract and are asserted; and the von Karman wrappers must have **no `pressure()`**, because
model #6 has none and `StringVKPlateBridge` composes with them precisely because the bridge batch
refused to invent one.

### 32.7 Collaborators are looked up as module globals, deliberately

`RoomLoadedPlate.__init__` calls the *module-global* `_PlateSurface`, `SurfacePort` and `splu`, so
the Rust class imports `physsynth.core.airbox` and reads the same three names at call time. A Rust
class reaching back into the Python module it replaces looks odd and is the faithful transcription;
it is also what makes three things possible at once: the monkeypatch contract keeps working, the
parity file can put both languages on one factorization (§32.3), and §28.4's trap becomes visible
instead of silent — a Rust wrapper built in the default gate holds **Python** collaborators, which
is a useful comparison and a misleading one to make by accident, so
`test_a_rust_wrapper_built_without_the_patch_holds_python_collaborators` asserts which is which.

### 32.8 Two scars, both already written down, both caught by the thing that was written

**§25.8a, and this time it faked a library bug.** After tightening `set_state`'s `v0` to keep an
omitted argument and an explicit `None` apart (§24.7), the parity file failed with the two
collapsed — and the failure looked exactly like PyO3 behaving differently for a positional argument
than for the keyword-only one §31.7 documented. It was a **stale wheel**: `cargo build` had run and
`pip install` had not. Probing the arms directly (a temporary marker in each branch) showed them
behaving precisely as §31.7 says. The lesson is one this plan already carries and is worth
restating with its new failure mode: *nothing in the suite can tell a stale wheel from a fresh one*,
and the wrong diagnosis it produces need not look like staleness at all — it can look like a
well-documented library invariant being false.

**§19.7's line continuation, a sixth time, from a sixth tool — and the mechanism is now identified.**
A literal backslash-n went into the YAML again, and §20.7's test caught it in seconds. The cause is
not typing: the editing path collapses one backslash level even inside a quoted heredoc, so a
correctly written continuation arrives as a two-character escape. Writing the newline as an
explicit character code sidesteps it entirely. Six occurrences, and still zero red CI runs since
that scar became a test.

### 32.9 The success condition

* `cargo test --workspace` — 25 test binaries green, including the dependency allowlist (still
  empty). This batch adds **no** native tests, which is itself a consequence of §32.2: there is no
  arithmetic in it that can be exercised without a Python interpreter.
* `cargo fmt --all --check` and `cargo clippy --workspace --all-targets -- -D warnings` — clean.
* `ruff check .` — clean.
* `PHYSSYNTH_RS=1 pytest` over the ten airbox files plus `test_stability.py` and
  `test_web_backend.py` — **858 passed**, which is the whole of that list (858 collected). §31.10's
  893 for "the same" list included batch 6's `test_rust_parity_airbox.py` as well; the difference is
  bookkeeping, not a lost test, and is recorded here so the next batch does not re-derive it.
* `pytest tests/test_rust_parity_airbox_wrap.py` — **107 passed**.
* Default path unchanged: the same files green without the flag.
* `pip install ./crates/physsynth-py` before believing any of it (§32.8).

### 32.10 What the next batch inherits

* **`airbox.py` is one tier from finished.** What is left is `_MembraneSurface`, the two membrane
  wrappers and their shared mixin — about 410 lines, the same shape as the plate pair, and the
  transcription is now a known quantity. `_MembraneSurface` differs from `_PlateSurface` in two
  places only (the mass is the membrane's own and there is no acceleration cache to refresh), so
  the seam should reuse `assemble_a_bare`'s pattern rather than gain a third copy.
* **`connection.py` is unblocked, and that was the point of the ordering.** §31.11 named
  `RoomLoadedBody`, `_PlateSurface` and `RoomSuspendedPlate` as the three collaborators its bridges
  are handed; all three are Rust now, and `test_airbox_vk.py`'s bridge tests pass under the flag
  with a Rust wrapper standing in for a plate. The remaining blocker named in §31.11 — that the
  wrapper objects must exist and answer — is discharged. `connection` needs no membrane wrapper.
* **Do not add a getter to these classes without checking the delegated list.** §32.6. The failure
  is silent in both `cargo test` and every physics bar.
* **The two parked tightenings are still parked**, unchanged since §31.11 and now three batches
  old: the room's own energy books could be exact with `reduce::sum`, and `bore.rs:262`,
  `reed.rs:229` and `:505` still spell `c0.powf(2.0)` with a literal exponent. Together they are one
  small batch, and the second of them is a `--release`-only divergence in shipped code.
* **`crates/physsynth-core/src/lib.rs`'s module header is still stale**, as §30.13 and §31.11 both
  recorded. Nothing watches it. Unchanged, still one line to know, and now three batches behind.
## §33 Phase 5, batch 9, as built (2026-08-31) — the membrane pair, and a getter that takes a write away

`airbox.py` finished. §30 ported the room, §31 the three ports, §32 the plate and body wrappers,
and this batch ports what was left: `_MembraneSurface`, `RoomLoadedMembrane`,
`RoomSuspendedMembrane` and the mixin they share — plus the two module-level helpers nobody had
noticed were still Python, `_face_axes` and `impedance_from_zeta`. **Every class the module exposes
and every function any client outside it reaches now has a Rust implementation behind the flag.**

That sentence is deliberately narrower than "the file is Rust", and §33.8 is why the difference is
worth spelling out. What is *not* swapped, after checking rather than assuming: the module's
constants (`FACES`, `PLANES`, `RHO0_AIR`, `C0_AIR`, `_AXES`, `_LAMBDA_MAX`, `_LAMBDA_TOL`,
`_SPREADINGS`), which are values rather than implementations and which a swap would buy nothing
for; and `_require_same_rate`, which **has a Rust twin** (`airbox_wrap::require_same_rate`, used by
every Rust wrapper) but no client outside the module at all — grepped, not assumed. It stays live
for exactly the reason `AirBoxPy` does: its only callers are the reference's own classes, which
stay live until §1 deletes them. It is not a fourth orphan and the next batch should not treat it
as one.

Not one line of the ten `test_airbox_*.py` files was touched, for the fourth batch running, and the
flagged CI step's file list has not changed once across the four tiers.

The batch adds **no stepping arithmetic**, and that is worth stating first because it decides
everything else. `RoomLoadedMembrane.step` and `RoomLoadedPlate.step` are the same eleven lines
over a different seam, so §32's `Wrap` is reused unchanged and the two membrane classes are two
more arms of the same enum. What is genuinely new is a seam whose `a_bare` is `(1 + sigma k) I`,
a `commit` that is a two-level roll with no acceleration cache, and one right-hand-side term with
no oracle.

### 33.1 The shape on disk

* `crates/physsynth-py/src/airbox_wrap.rs` — `PyMembraneSurface`, `PyRoomLoadedMembrane`,
  `PyRoomSuspendedMembrane`, and `Wrap`/`plate_wrapper!` generalised into `grid_wrapper!`.
* `crates/physsynth-py/src/airbox_port.rs` — the two module helpers bound.
* `crates/physsynth-core/src/lib.rs` — the module header, three batches stale since §30.13, brought
  current through Phase 5. Nothing watches it; it is still one line to know.
* `crates/physsynth-core/src/bore.rs`, `src/reed.rs` — the three parked `powf(2.0)` literals
  (§33.6). **This is a change to shipped numbers**, on the Rust arm only, and it is a fix.
* `physsynth/core/airbox.py` — five `*Py` aliases and five names under the flag. **No edit to the
  reference.**
* `tests/test_rust_parity_airbox_memb.py` — **new**, 83 tests.
* `tests/test_rust_parity_bore.py`, `tests/test_rust_parity_reed.py` — three pins for §33.6.
* `tests/test_stability.py` — three classes into the class table, two functions into the ported
  table.
* `.github/workflows/ci.yml` — the new parity file, and the airbox comment block.

No `physsynth-core` model code and no new native tests, for §32.2's reason: there is no arithmetic
in this tier that can be exercised without a Python interpreter.

### 33.2 The finding: a `#[getter]` with no `#[setter]` silently makes a plain attribute read-only

§32.6 found that a `#[pyclass]` getter is a data descriptor and therefore **shadows** `__getattr__`
permanently, where Python's `__getattr__` only fires on a miss — so every getter added to a
drop-in wrapper silently takes a name away from the model it stands in for. This batch found the
other half of the same fact, and it points the opposite way.

A `#[getter]` with no `#[setter]` is *still* a data descriptor. Its `__set__` exists and raises. So
porting a class does not only decide which names can be **read** through it; it silently decides
which can be **written**, and the default is *none* — where the reference, being Python, allowed
every one of them. The reference's `RoomLoadedMembrane.n` is a plain integer attribute a caller may
advance; the ported one, written the obvious way, is read-only, and neither `cargo test` nor any
physics bar can see the difference.

There is exactly one client of it in the whole tree, and it is not the shape any earlier search
would have found:

```python
# tests/test_airbox_membrane.py:387 -- the lagged-velocity negative control
port.require_ready()
pbar_free = port.free_pressure()
q = port.T @ ((m.u - m.u_prev) / m.k)      # BACKWARD, not centered: the whole change
rhs = inst._surface.rhs(None) - m.k * m.k * (port.T.T @ pbar) / inst._denominator
inst._surface.commit(lu.solve(rhs))
inst.radiated_energy += m.k * float(np.dot(pbar, q))
inst.n += 1
```

That test does not replace a collaborator (§31, §32) and does not reach a private name (§0, §3.1).
It **bypasses `step` entirely** and hand-rolls a different scheme out of the wrapper's own parts, so
what it needs is not to read the object but to *drive* it. §30.3's rule — grep for assignment, not
only for reference — is the right search and it was aimed one object too far away: §30.3 ran it
against the room's *collaborators*, and what it has to be run against is **the class being ported
itself**.

The failure mode is the good one, which is luck rather than design: an unwritable `n` raises, so a
correct port goes red instead of quietly green. It is worth noticing that it is luck. Had the
reference's attribute been one a client *reads back* after writing — say a ledger a test seeds
before a run — a swallowed write would have produced the ordinary silent shape instead.

The remedy is one setter, and it went into the shared macro rather than into the membrane arms, so
the four plate wrappers gained it too. That is deliberate: the reference lets a caller write `n` on
all six classes, and a Rust class that refuses is a fidelity gap whether or not a test happens to
exist. **The general form is that the port of a Python class starts from "every attribute is
writable" and has to justify each refusal, not the other way round.**

### 33.3 Three spellings of one name, and the failure is not where you look for it

The wrapper's model attribute is named in three places, and they must agree:

1. the `#[getter]` the class exposes (`inst.membrane`),
2. the label `_require_same_rate` puts in its refusal (`"membrane fs = ... but room fs = ..."`),
   which `test_airbox_membrane.py` matches on, and
3. the name `__getattr__` refuses to delegate, so a lookup cannot recurse.

Reusing §32's macro without parameterising it gets all three wrong at once, and the interesting one
is the third. With the getter still named `plate`, `inst.membrane` is a **miss**, so it falls
through to `__getattr__`; the guard compares against `"plate"` and does not match; the lookup is
delegated to the model — and a `Membrane` has no `.membrane`. The wrapper loses access to its own
resonator, and it does so by way of a delegation that is working exactly as designed.

So `Wrap` carries a `model_name: &'static str` and the macro takes the getter's identifier, and the
two are documented as halves of one decision. `test_the_wrapper_answers_for_its_own_model` is the
one assertion; it also checks `not hasattr(inst, "plate")` on both languages, because a membrane
wrapper that answered to `.plate` would be a name the reference does not have.

### 33.4 The one term with no oracle, and the parametrization that reaches it

§32.10 called `_MembraneSurface` a two-place difference from `_PlateSurface`. The seam's own
docstring names three, and the third is the one that shapes the parity file:

> `rhs`'s `f_ext` term has no counterpart in the model. `Plate.step` has its own `f_ext` path, so
> batch 3's copy of it could be checked against the original; `Membrane.step()` takes no force at
> all. This term is therefore *new* arithmetic.

For the plate, `rhs(None)` reduces to `Plate.step`'s own line and the model is the oracle. For the
membrane there is nothing to reduce to — and `f_ext=None` is the default that every airbox test
and every natural parity fixture passes. A parity file written the obvious way compares the shared
half twice and never touches the half with no oracle, and passes: §23.6's emptied comparison
through an **eighth** door, reached this time not by a swap or a replaced collaborator but by a
**default argument**.

Every trajectory test in the new file is therefore parametrized over `forced`, and the seam's own
right-hand-side test asserts, on the forced arm, that `rhs(f_ext)` is not `rhs(None)` — so the
parametrization cannot decay into two spellings of one comparison.

### 33.5 The two associations, pinned by a search — and the search failed twice first

The seam has two scalar folds the reference writes left to right and a tidy-up would rewrite:
`denominator = (rho * h) * h` and `c2k2 = ((c * c) * k) * k`. §26.6 says a spelling pin must
*search* for a witness rather than assert hand-picked constants, and that rule held: witnesses for
both exist within a few hundred neighbouring values and the pin is a real one.

The same rule then failed twice in this batch's other half, in two ways worth separating, because
between them they say what a witness search actually has to do.

**A predicate on the sub-expression is not a predicate on the expression.** The first pin for the
reed's `(wr k)**2` searched for a value where `t**2 != t*t` — found one immediately — and then
asserted the trajectory, which was bit-identical, because the enclosing `2.0 - t^2` **absorbs** one
ulp of `t^2` whenever `t^2` is small against 2. That is §23.5 verbatim ("a predicate tested on the
sub-expression rather than the whole expression found a witness whose difference the following
division absorbed"), arriving inside the test written to catch §17.2, one batch after §23.5 was
written down. The predicate has to be the whole coefficient `(2.0 - (wr k)^2) / den`, and the
fixture that satisfies it is a very stiff reed near `wr k = sqrt(2)`, where the subtraction cancels
and the last bit is the answer. §16.4's rule, in the measurement rather than in the model.

**An `np.nextafter` walk is the wrong search space, and it fails by reporting "none".** `pow` and a
multiply disagree in roughly 5 of every 10,000 values drawn from a decade-wide band, and they do it
in **clusters**. A walk of 200,000 consecutive doubles spans about 1e6 times less range than that
sampling, so it explores one neighbourhood and either finds several witnesses or none at all —
measured, **0 in 200,000 consecutive doubles from 1.41421356** against **47 per 100,000 samples
drawn from [1, 2)**. Both searches now step by a *relative* 1e-9, which covers 2e-4 of the value in
the same budget, and both assert that a witness was found.

**And a third: the comparison has to be made while the accumulator is empty.** The pin for
`reed_velocity ** 2` first ran 4,000 steps at the shipped fixture, verified that three of them
landed on a witness — and passed against a deliberately reverted binary anyway, because by step
4,000 `reed_damp_work` is large against one increment and the addition swallows the last bit
outright. That is §23.2's mechanism in a ledger instead of a field. The fixture is now searched
over `p_mouth` for one whose **first** step lands on a witness, and the ledger is read after that
one step, where it *is* the increment.

Each of the three was verified the same way: revert the Rust spelling, rebuild, watch the pin go
red, restore, rebuild, watch it go green.

### 33.6 The parked `powf(2.0)`, which was a shipped divergence and had been for six batches

§31.11 parked three `c0.powf(2.0)`-style literals in `bore.rs:262`, `reed.rs:229` and `reed.rs:505`
as "a `--release`-only divergence in shipped code". That description was right and understated.

CPython's `float.__pow__` is the C library's `pow` for every exponent, `2.0` included; LLVM folds
`powf(x, 2.0)` into `x * x` whenever the exponent is a visible literal, and it does so **only in
release** (§17.2). `pip install` builds release. So for six batches the shipped extension computed
`c0 * c0` where the reference computed `pow(c0, 2)`, and nothing saw it — because the ambient
`c0 = 343.0` is one of the values where the two agree (343² = 117649 is exact in doubles) and no
fixture in the suite happened to sit anywhere else.

Measured before the fix, at the first `c0` above 343 where they disagree: the two bores separate at
**9.4e-15 of amplitude over 200 steps**. Measured after: bit-identical, at that `c0` and at the two
next witnesses. The reed's stiffness coefficient diverges at **step 1** at its witness fixture, and
its damping ledger at the first step. All three now go through `pyfloat::scalar_pow`, whose
`#[inline(never)]` is the whole point of the function.

`crates/physsynth-core/src/pyfloat.rs` has said since §17 that this is the shape to watch for. What
this batch adds is that the *shipped* code had it, that a native spelling test in both profiles does
not cover a site with no native test, and that the cheap general guard is a **Python** pin at a
searched fixture: it runs against the installed extension, which is a release build, so unlike a
`cargo test` it needs no second profile to mean anything.

### 33.7 The speed, and the room's own step diluting it

§32 measured the plate wrapper tier at 1.06–1.13x and called it the price of §32.2. The membrane
tier is the same shape, and the measurement is reported two ways because the obvious one is
misleading.

End to end — one `inst.step()` plus one `room.step()`, which is the contract's order — the Rust arm
is **0.98x–1.35x**. But the room in *both* arms is the Python `AirBox` (the parity fixtures do not
set the flag), and it costs **175 µs** per step at the 8,410-node fixture, which is most of the
measurement. Subtracting it:

| tier | N | Python | Rust | ratio |
|---|---|---|---|---|
| baffled | 12 | 197 µs | 144 µs | 1.36x |
| baffled | 24 | 326 µs | 274 µs | 1.19x |
| baffled | 40 | 628 µs | 642 µs | 0.98x |
| suspended | 12 | 309 µs | 196 µs | 1.58x |
| suspended | 24 | 387 µs | 241 µs | 1.61x |
| suspended | 40 | 710 µs | 601 µs | 1.18x |

§11.6 exactly: the win is per-call overhead, it decays as SciPy's compiled work grows, and it
crosses to a small loss when the sparse solve dominates. The suspended tier wins more than the
baffled one for the reason the shape predicts rather than a surprising one — it reads two pressure
planes and forms a jump per step, so it has more interpreter calls to remove and the same solve.

**Two figures a reader should not take from this table.** It is not a claim about the real-time
port, which is §29's territory (an inner iteration whose body ports with it, 15.5x). And it is not
a reason to regret §32.2: the alternative to computing through SciPy was ten tests passing having
compared a loaded head with itself.

### 33.8 The file's last two functions, which were not a tier

`airbox.py` still owned `_face_axes` and `impedance_from_zeta`. Both had Rust twins with native
tests **since §31** and were simply never bound and never swapped, so the module was one tier *and
two functions* from finished while every note said "one tier". Neither is hard — an integer table
and a left-folded product of three doubles — and both are now bound, swapped and pinned (values
across all six faces and six zetas, defaults and explicit air, plus the refusal text word for
word, which is the only thing in `_face_axes` that can differ and which `test_airbox_surface.py`
matches on).

The small lesson is about bookkeeping rather than arithmetic: **"what is left in this file" was
tracked by tier, and two functions that belong to no tier fell out of the count.** The swap guard
could not catch it either — its `ported_expected` table is a written-down expectation, so a
function that is neither aliased nor swapped is simply absent from both sides of the comparison and
nothing fires. A derive over `dir(module)` catches a *wrong* alias; only reading the module catches
a *missing* one.

### 33.9 Everything is bit-identical, and here that is a sharp test

Every comparison in the new parity file is exact: the seam's surface, matrix, right-hand side
(forced and unforced) and commit; construction including both fill reports; sixty coupled steps of
state and every ledger on both tiers, both head shapes, lossless and lossy, forced and unforced;
the reduction to a bare `Membrane` under a zeroed load; and all four substitutions the reference's
own tests make.

This is §32.2's reason, unchanged — the sparse products, the assembly, the factorization and
`np.dot` are the same calls on the same objects in both languages, and elementwise `+ - * /`
admits no reassociation — and it is worth restating that exactness *here* is a claim about the
transcription rather than about the dynamics. The two implementations are not two discretizations.

### 33.10 The success condition

* `cargo test --workspace` — 25 test binaries green, including the dependency allowlist (still
  empty).
* `cargo fmt --all --check` and `cargo clippy --workspace --all-targets -- -D warnings` — clean.
* `ruff check .` — clean.
* `PHYSSYNTH_RS=1 pytest` over the ten airbox files plus `test_stability.py` and
  `test_web_backend.py` — **858 passed**, the same number as §32.9 and the whole of that list. The
  count does not move because the new parity file is not in it and no test was added to the
  reference's own suite.
* `pytest tests/test_rust_parity_airbox_memb.py` — **83 passed**.
* Every parity file together — **2,478 passed, 1 skipped**, which includes four tests
  added to `test_rust_parity_airbox_wrap.py` for the setter this batch widened onto the
  four plate wrappers. Nothing in the batch's own run could see that widening otherwise:
  a setter adds no name to `dir()`, so the derived surface guard passes identically
  either way.
* Default path unchanged: the same files green without the flag.
* The swap guard's class table went up by **three** and its function table by **two** — checked, per
  §23.7, rather than assumed.
* `pip install ./crates/physsynth-py` before believing any of it (§32.8), and `--no-cache-dir
  --force-reinstall` when a rebuild has to be certain: an ordinary `pip install` can serve a cached
  wheel, which is §25.8a with a second way in.

### 33.11 What the next batch inherits

* **`airbox.py` is finished in the sense §33's opening defines, and `connection.py` is the next
  file.** §32.10 discharged its last
  blocker and this batch changes nothing about it: `connection` needs no membrane wrapper, touches
  no private name on any collaborator, and its three bridges are now handed Rust objects on every
  path the suite exercises. It is pure duck typing (§31.11), so what it needs from a port is that
  the objects answer — and §33.2 adds one clause to that: **that they answer to a write, too.**
  Grep `tests/test_*connection*.py` and the bridge tests for assignment against the bridge itself,
  not only against its collaborators.
* **After `connection`, `analysis/`**, and that is the whole of the core.
* **A `#[pyclass]` getter decides both directions.** §32.6 for reads, §33.2 for writes. Neither is
  visible to `cargo test` or to any physics bar, and the read half is silent while the write half is
  loud — so the write half is the one that has been getting caught.
* **The parked room-energy tightening is still parked**, and is now four batches old: the room's own
  `dissipated`/`injected` books could be exact with `reduce::sum` (§31.11's half of it survives;
  the `powf` half is discharged by §33.6). It changes numbers in a ledger, so it wants its own
  measurement and its own batch.

## §34 Phase 5, batch 10, as built (2026-08-31) — the bridges, and a port that is slower on purpose

`connection.py` — `StringBodyBridge`, `StringPlateBridge`, `StringVKPlateBridge` and
`SympatheticStrings`, all four in one batch. **This is the last file of `physsynth/core/`.** What
remains of the whole migration after it is `analysis/`.

All four move together and that was not a judgement call. Two exact anchors bind them (§15.2):
`tests/test_sympathetic.py::test_single_string_bit_identical_to_string_body_bridge` asserts a
one-string `SympatheticStrings` is `array_equal` to a `StringBodyBridge`, and
`tests/test_airbox_vk.py::test_the_nonlinear_false_chain_is_the_linear_bridge_bit_identical`
asserts `StringVKPlateBridge.stability_margin == StringPlateBridge.stability_margin` to the last
digit. Port either half of either pair alone and the anchor breaks for a reason having nothing to
do with the physics.

**Everything is bit-identical**, the von Kármán bridge's trajectory included, over every fixture
tried. That is §32.5's shape again — the two arms are not two discretizations, they are one Picard
loop through one factorization — and it makes the exactness a sharp test of the transcription
rather than a claim about the dynamics.

Not one line of the eleven existing test files that drive a bridge was touched — the four
`*connection*.py`, `test_sympathetic.py`, `test_radiation.py`, four `test_airbox_*.py` and
`test_web_backend.py`.

### 34.1 The shape on disk

* `crates/physsynth-py/src/connection.rs` — **new**, 1,415 lines. Four `#[pyclass]`es and one
  shared `PlateBridge` the two plate classes delegate to.
* `crates/physsynth-py/src/lib.rs` — the module and four `add_class` lines.
* `physsynth/core/connection.py` — `import os`, four `*Py` aliases and four names under the flag.
  **No edit to the reference implementation.**
* `tests/test_rust_parity_connection.py` — **new**, 36 tests.
* `tests/test_stability.py` — `connection` into the class-swap derive's tuple and its
  `_USE_RUST` tuple, four entries into the expected-class table.
* `.github/workflows/ci.yml` — a new flagged step, and the parity file.

No `physsynth-core` code and no new native tests. This is the **second module with no core half**,
for §32.2's reason arrived at by a different road — §34.3.

### 34.2 The finding: a ported caller that computes nothing is *slower* than the Python it replaced

Every batch since §11.6 has said the win is per-call overhead rather than arithmetic. The clause
that was missing, and that this batch supplies with a sign rather than a magnitude, is that **Rust
pays that overhead too when it is the one making the call.** A `getattr` + `get_item` + `extract`
issued from Rust is not cheaper than CPython's `LOAD_ATTR` / `BINARY_SUBSCR`, which have inline
caches and a specialising interpreter behind them; what Rust wins is the *work between* the calls,
and a pure delegator has none.

`connection.py` contains both cases in one file, which is what makes the measurement a finding
rather than a disappointment. Measured on this machine, best of five runs of 2,000–5,000 steps
each, with the flag **on** so every collaborator is already Rust:

| class | Python µs/step | Rust µs/step | ratio |
| --- | --- | --- | --- |
| `StringBodyBridge` (M=4, N=100) | 1.61 | 1.66 | **0.97x** |
| `StringBodyBridge` (M=32, N=100) | 1.65 | 1.72 | **0.96x** |
| `StringPlateBridge` (N=16) | 12.54 | 12.56 | 1.00x |
| `StringPlateBridge` (N=40) | 166.94 | 166.15 | 1.00x |
| `SympatheticStrings` (J=2) | 4.77 | 2.61 | **1.83x** |
| `SympatheticStrings` (J=8) | 10.23 | 6.77 | **1.51x** |

Those numbers are within-run, best-of-five, and directionally confirmed by an intervention (the
keyword-dict change below moved `StringBodyBridge` from 0.93x to 0.97x, in the predicted direction),
which is what makes a 0.05 µs effect on a 1.7 µs step reportable at all. The first, less careful
sweep read `StringPlateBridge (N=40)` at **1.62x** where the careful one reads 1.00x — same binary,
same fixture — so the machine's own drift is comfortably larger than the effect being reported, and
only the three properties above separate them. Do not read 0.96x as precise; read it as "a few per
cent behind, reproducibly".

The discriminator is not the class, the model or the grid. It is **how much work sits between the
collaborator calls.** `StringBodyBridge.step` is five attribute touches and two float multiplies —
the pure delegator, and it comes out a consistent few per cent *behind*. `SympatheticStrings.step`
allocates a NumPy array of per-string forces every step in Python and a `Vec` in Rust, loops over
`J`, and reduces; that is real per-step work, and it is 1.5–1.8x. The plate bridges sit at exactly
1.00x because a sparse solve dominates and neither implementation touches it.

This sharpens §32.4 rather than contradicting it. §32.4 measured the wrapper tier at
"indistinguishable from neutral" and concluded that an inner iteration is where the real-time win
lives *only if the iteration's body ports with it*. Here the neutral point is crossed and the
answer goes negative, so the rule gets its general form: **porting a caller buys nothing when the
callees are separate Python objects, because the boundary crossings do not go away — they are
merely initiated from the other side.** Removing them would mean the bridge holding its string as
a native `physsynth_core::IdealString` rather than as a `Py<PyAny>`, and §34.3 is why it cannot.

One mitigation was worth taking and is the only optimisation in the file: both `body.step(force=)`
and `plate.step(f_ext=)` are keyword calls, and building a fresh `PyDict` for the keyword every
step is measurable at this altitude — it was ~0.05 µs of a 1.78 µs step, and moving to a
per-instance dict whose one entry is re-set each step took `StringBodyBridge` from 0.93x to 0.97x.
The dict caches the *dict*, never the array: `_f_ext` is writable (§34.6) and a cached array object
would ignore a caller who replaced it, which is §32.2's hazard inside the batch that cites it.
`tests/test_rust_parity_connection.py::test_a_replaced_force_vector_reaches_the_plate` asserts the
object identity, and the first draft of that test asserted the *contents* and failed for a third
reason — the bridge writes the force into the vector before the call and zeroes it after, so a copy
taken during the call is never all-zero.

### 34.3 Why there is no core half: the class is polymorphic over its collaborators' types

§32's tier could not own its arithmetic because the tier *below* had promised its clients they
could replace a port's matrices. This one cannot own its arithmetic for a reason that was true
before the migration started: `connection.py` is **pure duck typing**. §31.11 established that it
contains no `isinstance`, `hasattr`, `getattr` or `type(` at all, and the slots are wide:

* `body=` takes a `ModalBody` of either language, a `RadiatedBody`, a `ReactiveRadiatedBody` and a
  `RoomLoadedBody`.
* `plate=` takes a `Plate`, a `VKPlate`, `RoomLoadedPlate`, `RoomSuspendedPlate`,
  `RoomLoadedVKPlate`, `RoomSuspendedVKPlate` — and, through `airbox`'s seams, objects that are
  none of those.

So every collaborator attribute is read by name and every collaborator method called by name. A
downcast to a concrete `#[pyclass]` would turn a class that handles eight kinds of body into one
that handles two — and would pass the entire airbox and radiation family, because those hand the
bridge real models. `test_the_bridge_never_looks_at_its_collaborator_s_type` therefore builds a
hand-written delegating stand-in that is not a `ModalBody` in either language and asserts the
bridge drives it, reads its `phi`/`m`/`omega`/`M`/`q_prev`, and reproduces the Python arm's
trajectory bit for bit.

The line drawn is §32's verbatim:

* **Through Python:** `sparse.diags`/`sparse.identity`, the sparse products, `spsolve`,
  `splu(...).solve`, `np.linalg.eigvals`, both `np.dot`s, the string's `_second_diff`, and every
  `step`/`energy`/`pressure` on a collaborator.
* **In Rust:** the four validation chains, the `beta` precomputes, `_apply_A`'s elementwise
  arithmetic, the stretch/force/energy ledgers and `step`'s sequencing.

§32.3's correction applies unchanged and is worth restating because this file *looks* like Group D:
it calls `splu` and `spsolve` twice at construction and owns neither, so a solver group is a
property of **ownership, not of the file a factorization appears in**. Both are read as
`connection`'s own module globals at call time — the faithful transcription (§32.7), and what keeps
§24.4's shared-factorization manoeuvre available here. `test_the_guard_reads_its_solver_from_the_module_at_call_time` counts the calls through a monkeypatched name.

### 34.4 §33.11's claim that this file reaches no private name is wrong

The plan has said three times, most recently in §33.11, that `connection.py` "touches no private
name on any collaborator". It reaches two: `string._bc_right` in **four** constructors, and
`string._second_diff` in **two** `_apply_A`s. §31.11's actual finding — that the file contains no
`isinstance`/`hasattr`/`getattr`/`type(` — is correct and is about a different question; the
private-name claim was drift, restated until it read as measured.

It cost nothing, and the reason is the most cheerful thing in this batch: **Phase 0 predicted
exactly these two names three phases early.** `crates/physsynth-py/src/lib.rs` has carried a
section headed *"Private names are part of the surface"* since the first commit of the binding,
exposing `_bc_left`, `_bc_right` and `_second_diff` on the ground that "connection is a Phase 5
model — so for the whole migration a Python module is a client of this binding's *private* names."
The general form is worth keeping because it is the opposite of the usual scar: **a dependency
written down at the start survives being forgotten in the middle.** Nothing re-derived it, three
sections asserted its negation, and the port still worked on the first build.

### 34.5 Which reductions transcribe, which do not — and the fixture that can see neither

Two reductions and two `np.dot`s, and §14.2's question ("does it reach the next timestep?") gets
three different answers:

* `body.step(force=float(np.sum(forces)))` in `SympatheticStrings.step` — **yes**, directly.
* `np.dot(phi, q_prev)` in `_stretch(prev=True)` — **yes**: it is on the hot path and feeds
  `energy()`. `np.dot(phi, q)` in `_apply_A` is construction-time but feeds the guard that decides
  whether the object exists at all.
* `beta_b = k^2 * np.sum(phi*phi/m)` — **no**, and this is worth stating precisely because the
  obvious reading is wrong. `beta_b`'s only consumer is `cfl_2dof`, which the reference's own
  comment calls a cheap diagnostic and *not the real guard*; `step` uses `beta_s`, and `_apply_A`
  reaches `K`, `phi`, `m`, `rho` and `h` and never `beta_b`. So this is the migration's **second**
  reduction where §14.2 is answered "no" (§30's room ledgers were the first) — and the first where
  exactness was taken anyway, because here it costs one `reduce::sum` where there it would have
  cost a refusal.

The sums are transcribed with `reduce::sum` (§31.2); the dot products are not (§14.2 — `ddot`
fuses its multiply-add and admits no scalar recipe). Both choices are right and **neither is
exercised by a single fixture the suite ships**, which is §16.4 arriving in the *reductions* rather
than in a solver:

* §30.2's cutoff makes the sum question decidable by counting terms, and every body in this repo
  has **four or five** modes while every sympathetic rig has **two or three** strings. Measured
  here: `np.sum` and a left-to-right loop agree in 5,000/5,000 draws at M = 4, 5 and 7, and differ
  in **2,022/5,000** at M = 8. So `reduce::sum`'s blocking is dead code in every existing test.
* §14.3's finding governs the dot products: every body fixture in `tests/helpers.py` has
  `phi = 1.0`, and a fused multiply-add differs from a rounded one only when the **product** rounds.

The parity file therefore *searches* for its fixtures rather than picking them (§26.6). At M = 12 it
finds a witness where the Rust `beta_b` reproduces `np.sum` exactly (1.1926596366525052e-05) and a
left-to-right loop would have given 1.1926596366525054e-05; at J = 8 the shared bridge force differs
from a left-to-right sum on **296/600** steps while the trajectory stays bit-identical; and with
non-power-of-two `phi`, BLAS and a left-to-right multiply-add differ on **247/800** steps, which is
what makes "the dot products were not transcribed" an assertion rather than a comment.

### 34.6 §33.2's write question, aimed at the class being ported

§33.2's rule is that the port of a Python class starts from *every attribute is writable* and
justifies each refusal. Run against these four classes, the reference's own `self.X = ...` list is
nine names each (`string`/`strings`, `body`/`plate`, `K`, `k`, `beta_s`, `beta_b`, `cfl_2dof`,
`drive_index`, `_f_ext`, `_offsets`, `J`, `spectral_radius`, `stability_margin`, `n`), and every one
of them has a `#[setter]`. No refusal was needed and none was taken. The classes carry `dict`, so a
name the reference never used can be set on an instance exactly as on a Python class.

No test in the tree writes to a bridge — the grep §33.11 asked for was run and came back empty —
so unlike §33 this is a precaution rather than a repair. It is cheap and the alternative is a
failure mode that is silent in one direction: an attribute a client writes and then reads back
through a getter would swallow the write.

### 34.7 §19.7's line continuation, a **sixth** time, from a sixth tool

Adding the flagged CI step reintroduced it again: a `run:` block whose continuations arrived as
neither a backslash nor a newline, joining eleven `pytest` arguments into one 464-character line.
The tool this time was a shell heredoc that collapsed a doubled backslash before Python saw it.
§20.7's test caught it in under a second, for the third time in a row that the scar has been a test
rather than a paragraph, and for the sixth occurrence with **zero red CI runs** since it exists.
The fix — build the backslash as `chr(92)` so no layer can unescape it — is worth writing down as
the general remedy rather than "be careful".

### 34.8 The measured comparison

Default path (Python collaborators on both arms), Python bridge against Rust bridge:

| rig | steps | max\|du\| | max\|dE\| |
| --- | --- | --- | --- |
| `StringBodyBridge` | 2,000 | 0.0 | 0.0 |
| `SympatheticStrings` (J=3) | 2,000 | 0.0 | 0.0 |
| `StringPlateBridge` supported / free | 600 | 0.0 | 0.0 |
| `StringVKPlateBridge` supported / free, nonlinear on and off | 400 | 0.0 | 0.0 |

Plus, exactly: `beta_s`, `beta_b`, `cfl_2dof`, `spectral_radius` (the dense leapfrog eigenvalue),
`drive_index`, `stability_margin`, `n_iters`, `converged`, `last_residual`, `pressure()`, and the
`_offsets` array. Both cross-class anchors hold in all four language combinations
(`py/py`, `rs/rs`, `rs/py`, `py/rs`), which is the cell no existing test can reach.

The three `test_airbox_*` exact-margin anchors — a bridge over a room-loaded plate against a bridge
over a bare one — pass under the flag with a Rust bridge on both sides, which is what pins the
guard's bracketing across a boundary the parity file cannot construct.

### 34.9 The success condition

1. `cargo fmt --check`, `cargo clippy --workspace --all-targets -- -D warnings`,
   `cargo test --workspace`, `ruff check .` — all green. (§25.8: the linters are part of the bar,
   not an afterthought.)
2. `pip install ./crates/physsynth-py` **before** believing any parity number (§25.8a).
3. `pytest tests/test_rust_parity_connection.py` — 36 tests, default path.
4. `PHYSSYNTH_RS=1 pytest` over the coupled leg — the four bridge files, `test_radiation.py`,
   `test_airbox_surface.py`, `test_airbox_dipole.py`, `test_airbox_vk.py`, `test_web_backend.py`
   and `test_stability.py` — unmodified, green.
5. The swap guard's class table gains exactly four entries and `connection` joins both of the
   derive's tuples (§17.6/§23.7/§26.7: verify the count moved, do not assume).

### 34.10 What the next batch inherits

* **`physsynth/core/` is finished.** Every model, operator, solver, room, port, wrapper and bridge
  has a Rust implementation behind the flag. What is left of the migration is **`analysis/`**, and
  it is a different kind of module: it consumes trajectories rather than producing them, so §14.2's
  question ("does the reduction reach the next timestep?") is answered *no* everywhere and the
  interesting question becomes which of its read-outs a test compares exactly.
* **The speed rule now has a sign.** §11.6 said the win is per-call overhead; §32.4 said an inner
  iteration only wins if its body ports with it; §34.2 says a ported caller with no arithmetic of
  its own is **slower**. Before scoping a batch, ask how much work sits between the collaborator
  calls — if the answer is "none", the port buys fidelity and costs a few per cent, and that is a
  decision to make deliberately rather than discover.
* **The deletions are what is left after `analysis/`.** §1.2 says a Python model dies when its
  clients do, not when its own phase ends. `connection.py` was the last *client* blocking anything,
  so the remaining question is `web/serialize.py` and the `*Py` reference oracles — a batch about
  scope, not about arithmetic.
* **The parked room-energy tightening is still parked**, and is now five batches old (§31.11,
  §33.11).

---

## §35 Where the migration stands, and the plan for what is left (2026-09-02)

### 35.1 The state, in one paragraph

Every module under `physsynth/core/` has a Rust implementation behind the one flag: twenty-three
files, four solver groups, both halves of `airbox.py`, all four bridges. Nothing has been deleted,
by design (§1.2): the Python side is still the oracle every parity file compares against, the
viewer's live dependency, and the thing `PHYSSYNTH_RS=1 pytest` proves the Rust side reproduces.
What remains is the *other* three-quarters of the original brief — `analysis/` (1,930 lines), the
viewer backend (9,915 lines), the test suite (26k lines) and the deletions — and none of it is a
model. This section is the plan for that, and it corrects §5's Phase 7/8 ordering with what
twenty-seven findings have taught about where a batch boundary goes.

### 35.2 What changed on 2026-09-02, and why it is here rather than in a batch section

Three things landed together, none of them a port:

* **The two parked items were taken.** The room's energy books go through `reduce::sum_by`
  (a closure-reading form of `reduce::sum`, so no term array is allocated) and the parity
  assertions on `acoustic_energy`, `dissipated`, `injected` and the port book moved from
  `<= 1e-13` / `< 1e-15` to `==` — equal on all five wall types over 2,000 steps and at every port
  size. And the geometric string's DG Jacobian `(v,v)` block, recorded "not fixed" in §29's
  memory note, is assembled cancellation-free on both sides: measured against a 60-digit
  reference the old spelling was wrong by 5.8e-5 relative at strain 1e-4 (growing like
  `1/strain²`), the new one by 1e-15, and the rotating-wave cross-check's bar moved from 1e-8 to
  1e-12. Both are in `docs/dev/scientific-hurdles.md` §1–§2 with the numbers.
* **The parity step had been red on `main` for four runs**, and both failures were *negative
  controls* — tests asserting that a difference exists — not ports. `test_rust_parity_connection`
  required BLAS `ddot` to disagree with a scalar loop and GitHub's AMD EPYC kernel agreed;
  `test_rust_parity_airbox_memb` took a scalar witness and asserted its last bit survived into a
  vector it is 3.5e-3 of, which on this SciPy it never did. The first now reports rather than
  requires (the precedent is §14's `part_company` test, which made the same correction a phase
  earlier); the second searches at the whole expression (927 of 5,000 neighbours survive, the
  first at step 4). A third failure of the same class surfaced on the Linux dev box only:
  `np.arcsin` and `math.asin` one ulp apart at the room's mode `(3,2,2)`, so `mode_frequency`
  and `_mu_squared` took §22.3's portable spelling. The general form, worth a line in the
  findings ledger as its **twenty-eighth** entry: **a negative control whose predicate is a
  per-CPU kernel is a claim about the runner**, and the searching form must search the whole
  expression it will assert on.
* **`CLAUDE.md` is lean again.** The findings narrative under non-negotiable #3 had reached 70 KB
  in a file whose first line says "lean, always-loaded"; it moved verbatim to
  `docs/dev/rust-migration-findings.md` with an index and the eight questions to ask before an
  exact assertion. Nothing was cut.

### 35.3 `analysis/` — the next batch, and why it is not one batch

`analysis/` is six files: `modal.py` (742 lines, the closed-form oracles), `rotating_wave.py`
(602, the geometric string's BVP oracle), `duffing.py` (190), `damping.py` (185), `spectrum.py`
(133, the partial detector) and `dispersion.py` (77). §34.10 already said what makes it a
different kind of module — it *consumes* trajectories, so §14.2's question is answered "no"
everywhere — and the useful question is which of its outputs a test compares **exactly**. Measured
by reading the suite rather than guessing (the discipline §5's two corrections insisted on):

* **`spectrum.py` is the one file whose output is a decision** (§25.2): `measure_partials_near`
  returns *which* peak, and the spectrum-detector guard (`docs/memory/spectrum-detector-guard.md`)
  records that fifteen test files leaned on it without testing it. It goes first, alone, with the
  guard's two witnesses as its parity fixture, and its FFT is the batch's whole risk — an FFT is a
  library kernel in NumPy (pocketfft) and there is no dependency to take in Rust, so a hand-rolled
  radix-2 is Group-A-over-a-short-window arithmetic and the *peak index* is the exact claim, never
  the bin values.
* **`modal.py`, `damping.py`, `dispersion.py`, `duffing.py` are numbers, not decisions**, and
  every consumer compares them by a cents or a relative bar — so the Rust twins are tolerance
  ports and the only exact anchors are the ones the *tests* own: `cents` on identical inputs, and
  the discrete eigenfrequencies that feed `set_state` (§24.9's ARPACK pins). Two library facts set
  the risk: Bessel zeros (`scipy.special.jn_zeros`) and `brentq` have no crate behind them, and
  `radiation`'s one unported Bessel helper (§14) is the same debt — so the batch that writes a
  Bessel routine pays it twice and should be one batch. The complete elliptic integrals in
  `duffing.py` are the same shape (AGM, ~30 lines, exact to an ulp).
* **`rotating_wave.py` is a nonlinear BVP solved by Newton on a sparse Jacobian** — the only
  Group D member of the package, and §29's fill lesson applies unchanged. It goes last, after the
  §1 Jacobian fix has had its `λ_long` sweep re-run (hurdles §6), because its `_jacobian` is
  asserted equal to `2 · _dg_jacobian(q, q)` and that identity has just changed spelling.

Order: `spectrum` → (`modal` + `damping` + `dispersion` + `duffing` + the radiation Bessel) →
`rotating_wave`. Three batches, and the first is the one with a discrete output.

### 35.4 The test suite — port the *bars*, not the files

§1's ritual says each model's tests port at step 4 and never ahead of the model. Twenty-three
models are now at step 3, so the suite is the migration's largest remaining chunk and the one
whose order matters most. Three rules, from the findings:

1. **Port by criterion, not by file.** The suite is laid out by claim — energy, modal,
   convergence, dispersion, stability, signature — and a model's tests are not in a file named
   after it (§5's Phase 0 correction). The native `crates/physsynth-core/tests/` already mirrors
   that shape; each Rust test file should keep it.
2. **The parity files do not port.** Every `test_rust_parity_*.py` compares Python against Rust;
   when the Python side is deleted they have nothing to compare and they are deleted *with* it,
   in the same commit (§1.2's "together"). Until then they are the acceptance gate for every
   change to a ported module, which is what they were for today.
3. **A test that pins a NumPy-specific spelling does not port — it is retired with a note.** The
   `portable.py` pins, the ufunc-ladder witness searches, the `np.sum` cutoff measurement: each
   asserts something about NumPy, and a Rust suite has no NumPy to assert it about. Their
   *findings* are in the ledger; the tests themselves die with the Python.

The success condition for the suite is not "the Rust tests pass" — §1 says why that means
nothing on its own — it is that for every Python test retired, a native test asserts the same
physics bar at the same fixture, and the retirement commit names both.

### 35.5 The viewer backend — a scope question, not an arithmetic one

`web/serialize.py` imports `physsynth` 21 times and reaches inside 47 call sites (§3.1). Two
routes, and the choice is the human's: (a) a Rust HTTP server (§5's Phase 8 as written, ~10k
lines to move); (b) keep the thin Python serializer and have it import only the binding, which
is what `PHYSSYNTH_RS=1` already makes it do. Route (b) contradicts "Python goes, all of it"
and is named only because it is the cheapest way to make every *model* deletion safe now; route
(a) is the plan. Either way the browser side is untouched.

### 35.6 The deletions — one model, one commit, in dependency order

The order a Python model can die in is the reverse of the order its clients ported, and it is
now computable rather than argued: `string_ideal` waits on nothing (its last client,
`connection`, is Rust), the theta-scheme strings wait on the bow and barrier (both Rust), the
plates wait on `airbox`'s wrappers (Rust) and the bridges (Rust), the room waits on the viewer.
So every model but the ones the viewer builds can go **today**, and the viewer decides the rest.
Each deletion is its own commit carrying the model, its parity file, and the Python-only tests
§35.4 retires, so a bisect lands on one model.

### 35.7 CI — collapse the twenty-one steps into one flagged run

The `rust` job runs twenty-one per-batch "unmodified, against Rust" steps, each a hand-listed
subset of the suite that was a *claim* when the batch landed. With the core finished the claim is
the whole suite: one `PHYSSYNTH_RS=1` run of the same three shards the `validate` job uses, plus
the parity files. That is a workflow edit that cannot be verified from this environment (the
runner's log blobs are outside the proxy), so it is written here and not done: do it as the first
step of the `spectrum` batch, and keep `tests/test_ci_workflow.py`'s two guards — they are the
reason §19.7's continuation bug has reached CI zero times in six occurrences.

### 35.8 What the next batch inherits

* The parked list is **empty** for the first time since §31.
* The findings ledger has an index and eight questions; add the twenty-eighth entry (§35.2) when
  the next batch confirms the CI runner is green on the corrected controls.
* `docs/dev/scientific-hurdles.md` is the register of open *physics*; the two costed proposals in
  it (θ-loss compensation, Newton–Krylov for the von Kármán step) are Rust-first under §6 and
  wait on the human's priority call.
* `spectrum.py` is next, alone, and its exact claim is a peak index.

---

## §36 Phase 7, batch 1, as built (2026-09-03) — the detector, and a decision whose margin is zero

`analysis/spectrum.py` is in Rust, and with it the first crate outside `physsynth-core`. The batch
also did §35.7's CI collapse, which is written up here because the two are one commit's worth of
argument apart: the collapse is what makes "the suite is green under the flag" a *result*, and the
port is the first thing that had to be measured against it.

### 36.1 The shape on disk

* `crates/physsynth-analysis/` — a **third crate**, mirroring the Python package split one for one.
  `src/spectrum.rs` (the module), `src/lib.rs` (the flag argument), `tests/spectrum.rs` (11 native
  bars), `tests/deps.rs` (its own allowlist, empty).
* `crates/physsynth-py/src/spectrum.rs` — four free functions, no state, no buffer question.
* `physsynth/analysis/spectrum.py` — four `_py` aliases and the swap, gated on
  **`PHYSSYNTH_RS_ANALYSIS`**.
* `tests/test_rust_parity_spectrum.py` — 30 tests, and the first parity file whose two halves
  assert different *kinds* of thing.
* `.github/workflows/ci.yml` — 21 per-batch steps deleted, one three-shard flagged job added, one
  both-flags step added, the new parity file appended to the literal list.

Why a separate crate rather than a module in the core: `physsynth-core/tests/deps.rs` walks the
dependency graph rooted at *its own* package, by name. A new crate in the workspace therefore
inherits the portability contract's convention and none of its enforcement — it could take a
dependency and nothing in the repo would say a word. The gap opened the moment the workspace
stopped being two crates, and the fix is one file per package, which is also what makes "adding a
dependency is a reviewed edit in the package that takes it" true rather than aspirational.

### 36.2 The measurement that came before the port, and what it licenses

§35.3 said the exact claim for this module is a peak index. That is a hypothesis until someone
counts, so the counting came first: a pytest plugin rebound the three public functions and recorded,
for every call the dependent suite makes, how close each decision came to flipping. **384 real
`measure_partials_near` calls and 92,261 candidate peaks**, over the 18 test files that reach the
detector.

| decision | decided by | margin |
|---|---|---|
| which bin wins a search window | magnitudes | **>= 1.4e12 ulps** (3.0e-4 relative) |
| is that bin a genuine local max | magnitudes | **>= 1.6e10 ulps** |
| ordering of candidates by strength | magnitudes | >= 7.6e7 ulps, **zero** exact ties in 92,261 |
| does a candidate clear the separation | frequencies | **exactly zero** |
| is a candidate above `f_min` | frequencies | 0.2 Hz |
| the FFT length `nfft` | integers | exact by construction |

Two things fell out that no amount of reading would have produced. The guard the module exists for
is **live**: it fires on 14 of 384 real calls, and 17 take a window-edge argmax — it is not a
defensive branch carried across for completeness, it is load-bearing in the shipped suite. And
`detect_peaks`'s sort has **no ties to break**, which is what makes a stable Rust sort equivalent
to NumPy's unstable `argsort` reversed — a *measured precondition over these fixtures*, not an
identity, and the parity file says so.

### 36.3 The finding: a decision's margin and its axis are separate questions, and here they run opposite

Every previous batch asked one question — does the Rust arithmetic reproduce the Python arithmetic,
and if not, over what window does the gap stay under a bar. This module splits, and not along any
seam the earlier findings predict. It is not a reduction versus a step (#6), nor a solver's group
(#16), nor values versus stored order (#18). It is **two axes of the same computation**:

* **The frequency axis** — `freqs`, `df`, the window bounds, `min_separation_hz`, and every
  comparison among them — is `+ - * /` and nothing else. IEEE-754 specifies those exactly, so a
  transcription reproduces them bit for bit on any machine, with no claim about a CPU.
* **The magnitude axis** cannot be matched at all. Three library kernels sit on it: NumPy's own
  CPU-dispatched `cos` inside the Hann window (#14), `np.fft.rfft` (pocketfft, mixed-radix, whose
  bit pattern is not a target a radix-2 transcription could hit), and `np.abs` on a complex array
  (`hypot`). Measured, the magnitudes differ from NumPy's in 4,074 of 4,097 bins, at 3.2e-16 of
  the peak.

The port is safe because **the two orderings are opposite**: the only decision with zero margin is
on the exact axis, and every decision on the inexact axis clears by ten orders of magnitude more
than a rounding can move. That is a general question worth asking of any module whose output is a
decision — not "is this exact?" but "which axis is each decision on, and are the tight ones on the
exact one?" A module where they lined up the other way would not be portable at this fidelity at
all, and no amount of care in the transcription would fix it.

### 36.4 The second flag, and the argument §35.3 forgot to re-take

`PHYSSYNTH_RS=1` makes every model Rust and runs the existing Python suite against it. What makes
that gate worth anything is that the *instrument does not move*: a Rust string is measured by the
same Python detector, against the same analytic oracle, that the Python string was.

Put `analysis/` behind that same flag and the property is gone — the model and the ruler become
Rust together, and a shared misreading cancels instead of showing up. **Phase 7 was scheduled late
in §5 for exactly this reason** ("while Python still holds these, the Rust models are being checked
against an oracle that hasn't moved"), and §35.3 re-planned the order without re-taking the
argument. So this module reads `PHYSSYNTH_RS_ANALYSIS`, and the three combinations mean three
different things:

* `PHYSSYNTH_RS=1` — Rust models, Python instrument. The three-shard harness job, unchanged in
  meaning by this batch, which is the point.
* both — Rust models, Rust instrument. One CI step, over a **derived** file list.
* `PHYSSYNTH_RS_ANALYSIS=1` alone — Python models, Rust instrument. The sharpest test of this port
  by itself, and what was run locally: 677 passed.

The general form, and it is the rule this batch adds to §1's ritual: **a flag's meaning is a
property of what it does *not* swap.** Before widening one, ask what the existing runs were relying
on staying still. Nothing in `physsynth/core` imports `physsynth/analysis` — checked with a grep,
not assumed — so the two are genuinely independent.

### 36.5 The knife edge that does not always clear

`detect_peaks` suppresses a weaker peak closer than `min_separation_hz` to a stronger one, default
four raw bins. Candidates live on the bin grid. So the comparison is `|i*val - c*val| >= 4.0*val`
with `i - c == 4` — two spellings of the same real number, and a margin of **exactly zero**.

The first draft of the parity test asserted that the gap clears. It does not: at **100 kHz with
`nfft = 16` it comes out short and the candidate is rejected**. So this is not a theoretical hazard
that happens to be benign; it is a live comparison whose answer changes with the sample rate. The
claim a port can make is therefore not about the outcome but about *agreement* — both sides reach
the same verdict, which follows from the frequency axis being bit-identical.

And that only holds because the chain was transcribed rather than tidied. `val` is
`1.0 / (n * d)` with `d = 1.0 / fs` — three roundings — and `fs / n` is one rounding and a
**different number**. At the rates this project actually uses (8 k, 44.1 k, 48 k, 22.05 k, 96 k)
the two agree at every power-of-two size, so a test built from those rates would "prove" the tidy
form fine and leave the port accidentally correct. Searched over random rates they differ for about
**one pair in eight**. The parity file searches, per the standing scar about a hand-picked spelling
witness.

### 36.6 A transcendental refused inside a discrete decision, and the measurement that made it free

The original computes the FFT length as `int(2 ** np.ceil(np.log2(max(n, 2))))` — a `log2` sitting
directly inside a discrete decision, which is #17's shape at its sharpest, since a last bit next to
an integer is a *different spectrum*, not a different digit. The Rust side uses
`n.max(2).next_power_of_two()`: integer arithmetic, which cannot round.

That is a substitution, so it is measured rather than argued. The two spellings agree for **every**
length from 1 to 2^20, and at each of `2^k - 1`, `2^k`, `2^k + 1` up to 2^31. The float path is safe
on the range anyone can reach and the integer path is safe on every range, so the integer one runs
and the equivalence is pinned in the parity file. Where a transcendental *can* be removed from a
decision rather than portably spelled, removing it is strictly better than §22.3's manoeuvre — it
retires the claim instead of relocating it.

### 36.7 The CI collapse, and the trap §35.7 did not see

Twenty-one hand-written per-batch steps became one flagged run of the whole suite, sharded over the
same three runners as `validate`. Two things are worth carrying forward.

**The claim was unmeasured.** The union of the 21 subset lists was never the suite, so "the flagged
suite is green" was an assumption for as long as those steps existed. It was measured before
anything was deleted: 2,001 passed. Deleting a check on the strength of a claim the check never
made would have been the failure the checks exist to prevent.

**§35.7's instruction, taken literally, goes red.** It says to run "the same three shards the
`validate` job uses" with the flag set. But `shard_tests.py` covers `tests/` *exactly once*, so
`test_rust_parity_*.py` are **inside** those shards — and a parity file run with the flag compares
Rust against Rust, most assertions vacuous and the negative controls (which assert a difference
*exists*) red. A section can be wrong about its own mechanism while being right about its
intention, and the intention is what survived.

The exclusion lives in `shard_tests.py --exclude-parity` rather than a `grep -v` in the YAML, so it
is assertable — and it filters **after** the split, not before. Filtering first would hand LPT a
different file set and silently produce a different partition: the flagged and unflagged runs would
both be green, both complete, and a file could sit in shard 2 of one and shard 3 of the other,
harmless until the day they disagree about what exists.

Result: the `CI` run went from **49m46s to 9m21s**.

Two smaller scars. `test_ci_workflow.py`'s "every named test file exists" floor dropped from 50 to
20, which is a real weakening and is written down as one — the parity list stays literal precisely
so that canary keeps something to count. And that guard had to learn that a token containing `*` is
a *query*, not a file name, because the both-flags step derives its list with `grep -l` rather than
typing one, one commit after twenty-one typed lists were deleted.

### 36.8 Speed: three loop orders, and the instructive one is the middle

| loop order | 2^10 | 2^18 |
|---|---|---|
| blocks outside, `sin_cos` inside | 0.60x | 0.21x |
| k outside, blocks inside | 1.28x | 0.11x |
| tabulate per stage, blocks outside | **1.25x** | **0.51x** |

The middle row is the finding. Hoisting the twiddle out of the block loop cuts the transcendental
count from `(n/2)*log2(n)` pairs to `n-1` — and made short transforms twice as fast and long ones
**five times slower**, because what it actually hoisted was the memory access pattern: the inner
loop then strides by `len` and leaves cache. Tabulating per stage buys both. The general form is
older than this project but it bit here in a batch where every other number was about rounding: **a
loop-order change is a cache change first and an arithmetic change second**, and the two can point
opposite ways at different sizes, so one size is not a measurement.

Where it lands is stated plainly: faster than NumPy on short records, about half its speed on long
ones. pocketfft is mixed-radix and vectorised; this is a textbook radix-2. 2x is the honest price of
`tests/deps.rs` staying empty, and it is paid by code that runs once per test rather than inside a
timestep.

### 36.9 What is claimed, and what is only reported

Exact, asserted: `nfft`; the whole `freqs` array (`array_equal`); which bin won each search window
(recovered from the returned frequency, since the correction is bounded by half a bin); the bins
`detect_peaks` chose; the NaN pattern; and the guard's early return on both recorded witnesses,
where exactness is fair because the guard returns *before* touching a logarithm.

Tolerance, asserted: magnitudes within 1e-12 of the peak (observed 3.2e-16); refined frequencies
within 1e-6 of a bin (observed 0.0 on every fixture tried — a structural consequence of `i`
dominating `delta`, not luck, but not something to assert).

**Reported and never required:** the observed agreement itself. Requiring the magnitudes to differ,
or to match, would be §35.2's mistake one batch after it was written down — a predicate that is
really a claim about which CPU ran CI.

### 36.10 The success condition

* `cargo test --workspace` — 11 new native bars, plus the new crate's own allowlist. Green.
* `tests/test_rust_parity_spectrum.py` — 30 tests. Green.
* The detector's 18 dependent test files with `PHYSSYNTH_RS_ANALYSIS=1` (Python models, Rust
  instrument): **677 passed**. With both flags: **677 passed**.
* The whole suite minus parity with `PHYSSYNTH_RS=1`: **2,001 passed** — the number the CI collapse
  rests on.

### 36.11 What the next batch inherits

* **`modal` + `damping` + `dispersion` + `duffing` + the radiation Bessel helper**, as §35.3
  ordered, and the Bessel debt is why they are one batch. Their outputs are numbers, not decisions,
  so they are tolerance ports; the only exact anchors are the ones the *tests* own.
* `rotating_wave` last, after §1's Jacobian fix has had its `λ_long` sweep re-run.
* The findings ledger gains **#28** (§35.2's negative-control correction, now confirmed green on a
  runner) and **#29** (this batch). The parked list is still empty.
* `PHYSSYNTH_RS_ANALYSIS` now exists and its meaning is fixed: it swaps the *instrument*. Every
  further `analysis/` module goes behind it, and no `core/` module ever does.
