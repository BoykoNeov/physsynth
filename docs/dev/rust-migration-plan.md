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
