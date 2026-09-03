# CLAUDE.md — Physical Synthesis Simulator

Lean, always-loaded context. The full spec, math, and first-milestone definition live in
`HANDOFF.md` — read it before starting real work. Long-term expansion directions (more methods,
deeper physics, port-Hamiltonian coupling, differentiable/ML surrogates, GPU/WASM, haptics, the
build-your-own-instrument sandbox) are mapped in `HANDOFF.md` §12 — a horizon, not current scope.

## What this project is

A physical-modeling sound-synthesis tool (standalone + future DAW plugin). Many synthesis methods,
starting with one done deeply, expanding in breadth and depth. Interactive, beautifully visualized.

## Non-negotiable decisions (do not re-open)

1. **Accuracy first.** Fidelity over polish and over real-time. Offline rendering is fine now;
   real-time is a later port.
2. **Energy-based / passive numerical methods** (Bilbao framework) are the foundation. Chosen for
   provable stability, measurable fidelity, and a path to nonlinear models (gongs/cymbals).
3. ~~**Prototype in Python (NumPy/SciPy).** Julia acceptable if the human prefers. Not C++/JUCE
   yet.~~ **SUPERSEDED 2026-08-26 (the human's call):** Python is being retired entirely — core,
   analysis *and* the test suite — in favour of **Rust**, gradually and model by model.
   **One exception, 2026-09-03 (the human's call):** the **viewer backend stays Python** and talks
   to Rust through the binding — `web/serialize.py` is a serializer, not a model, and there is no
   Rust HTTP server to write (plan §35.5 closes it, and strikes §5's Phase 8). What remains of the
   viewer's port is an *import audit*: every place `serialize.py` reaches past a public constructor
   is the gate on deleting the Python model behind it.
   See `docs/dev/rust-migration-plan.md`; it also supersedes the portability contract's
   "Python stays the reference oracle" clause and absorbs HANDOFF §9's Phase 5.
   **State (2026-09-03): translation is over and DELETION has started.** Every model, operator,
   solver, room, port, wrapper and bridge has a Rust implementation (`crates/physsynth-core` +
   `crates/physsynth-py`), and all six of `analysis/`'s modules — `spectrum`, `modal`, `damping`,
   `dispersion`, `duffing`, `rotating_wave`, plus the Bessel and elliptic functions they stand on —
   are in a **third crate**, `crates/physsynth-analysis`.

   **Fifteen of twenty-three Python model bodies are GONE** — 16,146 lines, six of eleven deletion
   units, and not one physics bar retired. Plan **§39** is the audit that made the order computable
   (eleven units, from the reference-alias graph) and **§40–§43** are the deletions. Read §39's
   tables before scoping the next one. Four things about the new state:

   - **The wheel is now REQUIRED**, everywhere. A deleted model's module is an unconditional
     `from physsynth_rs import X`, so `pip install ./crates/physsynth-py` is a precondition for
     `pytest` to *collect*, not just to pass. The `validate` and `checks` CI jobs install it (the
     human's call, §39.6 route 1); `validate` is therefore no longer a pure-Python baseline, and
     nothing is.
   - **A deleted module is not empty.** Three things survive every time — types with no runtime
     implementation (`Literal`, `Callable`, `Protocol`, `NamedTuple`), measured constants *with their
     docstrings*, and re-exports of names defined elsewhere but reached through this module. And the
     binding **depends on Python** in four places (`grep -rn 'import("physsynth' crates/physsynth-py/src/`):
     `GeometricState` and `GrainSpec` are constructed *by Rust*, and `airbox` / `connection` have
     their namespaces read at call time. Run that grep before deleting anything (§41.2).
   - **Every deletion edits the guards**, or it is not green: `deleted_bodies` in
     `tests/test_stability.py`, `REMAINING_PARITY_FAMILY` in `tests/test_shard_partition.py`,
     `STILL_HAVE_A_PYTHON_TWIN` in `tests/test_rust_parity.py` (one row left), and the `rust` job's
     file list. `tests/test_binding_surface.py` is the permanent home for residue that can never be
     a native bar (buffer lifetime, PyO3 argument mapping).
   - **A deletion reaches modules it does not contain.** §43 is the case: `airbox.py` *re-derives*
     the plate's system matrix and factors it, and four of its reduction anchors are `array_equal`,
     so moving the plate's solver to Rust on the default path broke 15 tests in three modules
     outside the unit. That dependency has no import and no shared name — the tell is an
     `array_equal` in a test on a module you are not deleting, and the fix was one rebinding of
     `splu`. A deletion also **changes which tests share a process**, so it surfaces order- and
     process-dependent defects that are not yours (ledger #44, #46).

   `cargo test --workspace` runs the native bars and the Cargo dependency allowlists; `PHYSSYNTH_RS=1
   pytest` runs the **existing, unmodified** Python tests against the Rust code for the models whose
   Python side is still alive (reinstall the wheel before believing any parity number — nothing can
   tell a stale wheel from a fresh one).

   **There are TWO flags and they must not be merged.** `PHYSSYNTH_RS` swaps the *models*;
   `PHYSSYNTH_RS_ANALYSIS` swaps the *instrument that measures them*. The acceptance run sets only
   the first, so a Rust model is still checked by a Python detector against an unmoved oracle —
   widen one flag over both and a shared misreading would cancel (plan §36.4). Every further
   `analysis/` module goes behind the second; no `core/` module ever does. The one case that reads
   like an exception is not: `piston_radiation_resistance` is *implemented* in the analysis crate
   (the core crate's dependency list must stay empty, so it cannot reach a Bessel function) while
   its Python name swaps on `PHYSSYNTH_RS`, because it lives in a `core/` module. The crate a
   function is implemented in and the flag its name is swapped by are separate questions (§37.7).
   **Answered 2026-09-03 (the human's call):** `analysis/`'s Python bodies (units 10 and 11) *are*
   to be deleted, but **the numbers get frozen first** — what the Python detector measures across a
   spread of fixtures is recorded as written-down constants, so the Rust instrument stays checked
   against numbers a second implementation produced. `PHYSSYNTH_RS_ANALYSIS` becomes a no-op at
   that point and the two-flag separation goes with it; what replaces it is weaker in one specific
   way (the check can be repeated but not re-derived) and the file holding the constants must say
   so. Until that batch lands, the rule above stands as written.
   **Before scoping any batch, read `docs/dev/rust-migration-findings.md`** — the thirty-two
   findings about when two implementations agree to the bit and when they cannot (fed-back
   reductions, libm vs NumPy's own transcendentals, LLVM's constant fold, `np.sum`'s pairwise
   cutoff, descriptors that take a write away, anchors that bind two classes into one batch).
   They lived here verbatim until 2026-09-02 and moved so this file could be what its first line
   says it is; the questions to ask before writing an exact assertion are listed at its top.
   The open *scientific* problems — as opposed to porting ones — are in
   `docs/dev/scientific-hurdles.md`.

4. **Headless DSP core.** No I/O, no graphics inside `core/`. Viz and wrappers depend on the core,
   never the reverse. Keeps the physics portable to C++/Rust later.
5. **Unifying abstraction:** `exciter -> resonator (+- nonlinear coupling) -> body/radiation`.
   Adding a method = new resonator/exciter behind a stable interface.

## Working rules

- **Validation is code, not listening.** Every resonator exposes `energy()`. Correctness is asserted
  against closed-form physics: energy conservation (lossless run drifts < 1e-10), passivity (lossy
  run decreases monotonically), modal frequencies vs analytic oracle, convergence order. These tests
  exist and pass before any new model is added.
- **Do not tighten the `1e-10` acceptance bar.** The gap to the ~`1e-15` typically observed is
  deliberate headroom: this harness is the acceptance contract for the eventual native port, which
  must reproduce these numbers under a different compiler and BLAS. Tiers and rationale in §6.1.
- **Energy report is mandatory** on every resonator — it is the primary bug detector.
- When `E^n` drifts in a lossless run, suspect the boundary handling (summation-by-parts) first.
- Assert `lambda = c*k/h <= 1` for explicit schemes; reject construction otherwise. Tune toward
  `lambda = 1` (exact, zero dispersion).
- Oversample around any nonlinearity (aliasing).
- Round/format numbers before display.

## Start here

The first deliverable is the **ideal-string solver + validation harness** (HANDOFF.md §10). The
deliverable is the string *plus the rig that measures its deviation from theory* — not just a string.
Acceptance criteria are in §10.

## Open decisions — all closed (2026-08-10)

HANDOFF.md §11's five decisions are settled and kept there as a decision record: Python; explicit
*and* implicit (θ-scheme from the stiff string on); polyphony is per-instance for field models and
per-voice for strings (§11.3a, budget half deferred to Phase 5); the web viewer was the first viz
target; tolerances stand as they are (§6.1). New open questions go to the human as before — but do
not re-open these five.
