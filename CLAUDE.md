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
   analysis, viewer backend *and* the test suite — in favour of **Rust**, gradually and model by
   model. See `docs/dev/rust-migration-plan.md`; it also supersedes the portability contract's
   "Python stays the reference oracle" clause and absorbs HANDOFF §9's Phase 5.
   **Phases 0, 1 and the first two batches of 2 are built** (plan §9-§12):
   `crates/physsynth-core` + `crates/physsynth-py`, with `string_ideal`, all of `operators`,
   `membrane`, `exciter`, `body` and the *builder half* of `operators2d` ported. `cargo test --workspace`
   runs the native bars and the Cargo dependency allowlist; `pip install ./crates/physsynth-py`
   then `PHYSSYNTH_RS=1 pytest` runs the **existing, unmodified** Python tests against the Rust
   code. The flag is one switch for the whole tree: with it set, five still-Python string/beam
   models run on Rust-built operators; every plate — supported, free, orthotropic,
   guitar-shaped — plus the von Karman bracket runs on Rust-built geometry; and the whole
   body/radiation leg (bridges, sympathetic strings, all three radiation tiers) runs on a Rust
   modal body. Both implementations stay alive for now — deleting a Python model waits on its
   clients, not on its own phase (§1.2).
   Four facts worth knowing before planning work: a file's risk group is the group of its
   hardest function, so a module can port in halves (§11.2.1); `mallet` needs `collision`, so
   **Phase 2 finishes after Phase 3 starts** (§11.2.2); and the speed win is **per-step
   overhead, not arithmetic** — 8.7x on a small grid, ~1.1x once SciPy's compiled matvec
   dominates, so the *test suite* does not get faster and the *real-time* case does (§11.6)
   — and the crossover is about whether NumPy's hot path was **already compiled**, not about
   size: the modal body, which calls no compiled kernel at all, is **~15x** at every realistic
   mode count (§12.7). A fourth fact, from `body`: a leading **underscore is not a statement
   about the interface** — three modules assign to the body's `_accel` every timestep (§12.2).
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
