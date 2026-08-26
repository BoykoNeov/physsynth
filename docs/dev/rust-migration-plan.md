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

**Phase 0 — the harness, proven on one trivial model.**
`crates/physsynth-core` + `crates/physsynth-py` (PyO3/maturin), CI compiling on push, the
allowlist tests redefined per §2.2, and **`string_ideal` (206 lines)** ported. Success is not
"a string works" — it is *the existing `tests/test_string_ideal*.py` passing unmodified against
Rust*. Nothing else starts until that is green.

**Phase 1 — `operators` (165 lines).** Shared infrastructure everything downstream needs.

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
