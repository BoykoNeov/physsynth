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
   **State (2026-09-05): translation is over and DELETION is FINISHED.** Every model, operator,
   solver, room, port, wrapper and bridge has a Rust implementation (`crates/physsynth-core` +
   `crates/physsynth-py`), and all six of `analysis/`'s modules — `spectrum`, `modal`, `damping`,
   `dispersion`, `duffing`, `rotating_wave`, plus the Bessel and elliptic functions they stand on —
   are in a **third crate**, `crates/physsynth-analysis`.

   **All twenty-three Python model bodies are GONE** — 23,396 lines, **eleven of eleven**
   deletion units, and not one physics bar retired. Plan **§39** is the audit that made the order
   computable (eleven units, from the reference-alias graph) and **§40–§49** are the deletions. Read
   §39's tables before touching one again — but **§39.3's "zero native bars" for `airbox` was
   measured wrong** and §46 corrects it: the module carried nineteen `#[test]`s in `mod tests`
   blocks *inside* `src/`, which `cargo test --workspace` runs exactly like the files under
   `tests/`, and the audit had run `ls crates/physsynth-core/tests/`. §46 closed the remaining gap
   with eighteen more, so unit 6's blocker was cleared and **§47–§48 deleted it, in halves (the
   human's call)**: the room and three ports first with the whole four-file parity family, then the
   seven wrappers, two mixins and three seams. `airbox.py` is **4,112 -> 245 lines**, the largest
   deletion of the migration, and the second half retired **no test file at all**.

   **§49 deleted the last one**, `physsynth/core/connection.py`, **973 -> 74 lines**. §39.3 had
   filed it under a blocker it called permanent, and that conflated two questions: it is true that
   `connection` can never have a *native* bar (it lives only in the binding crate, so a
   `physsynth-core` test cannot reach it), and that is **not** a reason its body cannot be deleted
   — the airbox wrapper tier was under the identical blocker and §48 deleted it, with the physics
   asserted by Python tests running against Rust, which is what §39.5 says a body-deletion actually
   retires. **`PHYSSYNTH_RS` can no longer change which model a run exercises.** It still swaps
   `operators`, `exciter` and `banded`, but none of those is a resonator: the flag now chooses
   between two spellings of an operator or between two solvers, and that is the migration's real
   finish line.

   Seven things about the state this leaves:

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
   - **A module deleted in HALVES needs its own table.** `deleted_bodies` asserts three things at
     once — no `_USE_RUST`, no `<Name>Py` aliases, and each name *is* the Rust object — and a
     half-deleted module satisfies only the third, so it cannot be added and the identity claim
     would go unasserted for the whole interval between the two commits with every test still
     green. `half_deleted_bodies` in `tests/test_stability.py` is that table; it merges back into
     `deleted_bodies` at §48 (ledger #60).
   - **A guard that reaches zero is deleted, or re-derived over a WIDER population — never left
     empty.** Three tables hit zero and took three different exits. `tests/test_rust_parity.py`'s
     `STILL_HAVE_A_PYTHON_TWIN` and `half_deleted_bodies` were **deleted**, because an empty
     `parametrize` collects as a *skip* and an empty dict iterates zero times — green, and
     asserting nothing (ledger #52, #60). `expected_classes` was **widened**: the tuple its derive
     read went from one module to all twenty-one, and the claim became "no module in
     `physsynth.core` chooses between two implementations of a class any more", which is worth
     more at zero than it was at four (ledger #67). What is left to edit is `deleted_bodies` and
     `ported_expected` (the `<name>_py` FUNCTION table — the one §46.7 missed) and the `_USE_RUST`
     reader tuple, all in `tests/test_stability.py`; `REMAINING_PARITY_FAMILY` in
     `tests/test_shard_partition.py` (now **stable at three** rather than draining — none of the
     three has a Python body to delete); the extension-import scan in `tests/test_ci_workflow.py`;
     and the `rust` job's file list.
   - **`tests/test_binding_surface.py` is where a parity file's survivors go**, and a parity file
     is **harvested, not dropped**: 26 of `connection`'s 36 tests outlived their twin, and three
     got *sharper* by being re-aimed at what the transcription was actually copying — "Rust agrees
     with Python about `np.sum`" became "Rust agrees with **NumPy**", and a trajectory comparison
     for the reduction that must *not* be transcribed became a **count of the binding's `np.dot`
     calls** (ledger #64, #65). Only a comparison with no referent outside the twin really dies.
   - **Two `ruff --fix` hazards live in every shim**, both silent until something outside the file
     asks for a name. A re-export is an unused import, so **`__all__` is load-bearing** — without
     it `--fix` deletes the module's whole public surface (#66). And residue that is *not* a Rust
     object falls straight through `deleted_bodies`' identity loop: `connection.py` keeps `sparse`,
     `spsolve` and `splu` solely because `connection.rs` reads them off its namespace, and they
     need their own guard for **presence and reachability separately** (#63, §49.2).
   - **A deletion reaches modules it does not contain.** §43 is the case: `airbox.py` *re-derives*
     the plate's system matrix and factors it, and four of its reduction anchors are `array_equal`,
     so moving the plate's solver to Rust on the default path broke 15 tests in three modules
     outside the unit. That dependency has no import and no shared name — the tell is an
     `array_equal` in a test on a module you are not deleting, and the fix was one rebinding of
     `splu`. A deletion also **changes which tests share a process**, so it surfaces order- and
     process-dependent defects that are not yours (ledger #44, #46).

   `cargo test --workspace` runs the native bars and the Cargo dependency allowlists. `PHYSSYNTH_RS=1
   pytest` no longer changes any **model** — it swaps `operators`, `exciter` and `banded` only — so
   the default run and the flagged run now differ by three parity files and 275 tests, and *both*
   exercise Rust physics end to end (reinstall the wheel before believing any number — nothing can
   tell a stale wheel from a fresh one).

   **There is ONE flag now, and the second one's job was handed to a file.** `PHYSSYNTH_RS` swaps
   the *models*. `PHYSSYNTH_RS_ANALYSIS` swapped the *instrument that measures them* and is read by
   nothing since units 10 and 11 were deleted (plan §44) — `physsynth/analysis/` has one
   implementation. The separation it protected is worth understanding rather than forgetting,
   because it is what makes the acceptance run mean anything: with only the model flag set, a Rust
   model was read by a **Python ruler**, so a misreading shared by a model and its detector could
   not cancel (§36.4). That check was made and passed; what the deletion removed is the ability to
   *re-derive* it, and the human's condition for allowing it was that the numbers be frozen first.
   **`tests/analysis_frozen_values.py` is where they live** — 62 fixtures, 3,708 floats, recorded
   to the last digit from the Python implementation before it was deleted, asserted on every run by
   `tests/test_analysis_frozen.py`, with the case list derived from each module's `__all__` so a new
   oracle cannot be added unfrozen. It catches a transcription error, a wrong branch and a
   regression; it cannot catch an error the Python made too — that is what
   `crates/physsynth-analysis/tests/` is for, and §37.11 is the precedent (a native bar found a
   544% defect the Python always had, which no parity test could).
   The crate-versus-flag distinction that used to sit here is still true and still worth knowing:
   `piston_radiation_resistance` is *implemented* in the analysis crate (the core crate's dependency
   list must stay empty, so it cannot reach a Bessel function) while its Python name lives in a
   `core/` module. **The crate a function is implemented in and the module its name lives in are
   separate questions** (§37.7).
   **Before scoping any batch, read `docs/dev/rust-migration-findings.md`** — the **sixty-seven**
   findings about when two implementations agree to the bit, when they cannot, and what a
   *deletion* breaks that a port does not (fed-back reductions, libm vs NumPy's own
   transcendentals, LLVM's constant fold, `np.sum`'s pairwise cutoff, descriptors that take a
   write away, anchors that bind two classes into one batch, clients that re-derive a model's
   arithmetic, and reflective tests that are really claims about how a model stores its
   attributes).
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
