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

**Phase 3 — Group B + C, then `bow`.** `string_stiff` first (259 lines, the smallest banded solve),
then `string_damped`, `string_nonlinear`; then `collision`. Cheap, and they prove the LAPACK link.
`bow` comes last in the phase because it borrows the string's banded solve and cannot be checked
before the string is trustworthy.

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
- **Polyphony** (HANDOFF §11.3a) is still open and is an *engine-shape* question. It should be
  settled before Phase 2 fixes the `Resonator` trait, or the trait is rebuilt later.
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
