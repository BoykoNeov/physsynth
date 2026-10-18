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
3. **Rust, and no Python.** The project was prototyped in Python (NumPy/SciPy); on 2026-08-26 the
   human chose to retire Python entirely — models, analysis, viewer *and* test suite — in favour of
   Rust, and on 2026-10-17 it reached zero (`docs/dev/python-retirement-plan.md` §51). Do not
   bring Python back, not even for a script or a plot: where a diagnostic used to draw a figure,
   Rust emits the data instead.

   **Where things live.**
   - `crates/physsynth-core` — every model, operator, solver, room, port and bridge. Its
     dependency list stays **empty** (enforced by `tests/deps.rs`; the portability contract,
     `docs/dev/portability-contract.md`).
   - `crates/physsynth-analysis` — the oracles (spectrum, modal, damping, dispersion, Duffing,
     rotating wave, Bessel and elliptic functions) and the resolution horizon. Its dependency list
     is empty too (its own `tests/deps.rs`); it shares core's root finder and sparse LU as
     `#[path]` includes of core's files, not as a dependency, so it cannot intra-doc-link into core.
   - `crates/physsynth-viewer` — the served viewer: `cargo run --release -p physsynth-viewer --
     serve`, serving the JavaScript front-end in `web/static/`. Every payload float goes through
     `py::num`, because `serde_json` turns NaN into `null` silently.

   **How validation runs.** `cargo test --workspace --release` (CI's `rust` job, which also runs
   `fmt`, `clippy -D warnings` and `cargo doc` with warnings denied); the unoptimised half
   (`rust-debug`: core and analysis, because a debug build fires `debug_assert!` and does not let
   LLVM fold a literal exponent, so a test pinning an arithmetic *spelling* only means something
   there); and the exact viewer freeze (`frozen-windows`). Locally:
   `scripts/cargo-test-nice.ps1` (`-Full` before a commit). `cargo test` also runs the
   `mod tests` blocks inside `src/` — count those when asking what covers a module.

   **Frozen references** — answers from libraries and code that no longer exist here, recorded
   before they went. They catch a transcription error, a wrong branch and a regression; they
   cannot catch an error the original made too, which is what the physics bars are for.
   - `crates/physsynth-core/tests/reference/` — NumPy, SciPy and LAPACK answers.
   - `crates/physsynth-analysis/tests/reference/analysis_frozen.json` — the deleted Python
     analysis, 74 fixtures, its inputs frozen as exact doubles too.
   - `crates/physsynth-viewer/tests/frozen.rs` — the Python viewer's payloads for 588 requests.
     **Exact only on `x86_64-pc-windows-msvc`** (the C library's `sin`/`exp`/`ln` differ in last
     bits on Linux); elsewhere it checks structure only.
   - Reading any frozen float needs `serde_json`'s `float_roundtrip` feature.

   The browser check is `cargo run --release -p physsynth-viewer --example verify_headless` with
   the server running; it closes its Chrome through the DevTools port, never by name.

   **Writing a bar — what ~24 carrying batches taught** (§ = the retirement plan):
   - **Run every parameter off its default.** Grep a model's fixtures for each parameter's range.
     A `θ` dropped or hard-wired passed the whole workspace six times (§33–§46); so did the air's
     density and sound speed (§44, §45) and seven wind-instrument parameters (§42).
   - **A bar about a constant must not import the constant** (§29).
   - **A control must assert that it differs from its subject** (§25). A bar asserting a difference
     from a twin that differs in two ways certifies neither (finding #78), and a twin comparison
     going red says two numbers differ, not which is wrong (§32).
   - **Prove a bar is live by planting the defect it is for.** Re-plant any "nothing else catches
     it" claim workspace-wide before writing it down (§31, §50.5); a "this pins X" comment is a
     claim to plant too (§42). Before calling a plant nobody catches a gap, probe it — a plant
     that changes no bit anywhere is an equivalent mutant, not a gap (§40, §41). Plants can hang
     a test, so give each one a time limit (§40).
   - **A start-up needs its own bar.** Several first-step defects were invisible to every
     conservation bar (§30, §32, §35).
   - **A spelling cannot be pinned at a power of two** — a power-of-two length or constant makes
     the two spellings identical (§43).
   - A defect seen **only** by the Windows-exact viewer freeze is a gap: offer the human a bar of
     its own (the answer has always been yes).

   **Read before scoping work:** `docs/dev/rust-migration-findings.md` — eighty findings about
   when two computations agree to the bit; the questions to ask before writing an *exact*
   assertion follow its index. The open *scientific* problems are in
   `docs/dev/scientific-hurdles.md`. `docs/dev/rust-migration-plan.md` and the retirement plan are
   the history of the port, section by section.

4. **Headless DSP core.** No I/O, no graphics inside the core (`crates/physsynth-core`). The
   viewer, the analysis crate and any future wrapper depend on the core, never the reverse. Keeps
   the physics portable to every host it may need (plugin, browser, embedded). The empty
   dependency list is enforced; "no `std::fs`/`println!` in core" is not (a known gap, recorded in
   `docs/dev/portability-contract.md`).
5. **Unifying abstraction:** `exciter -> resonator (+- nonlinear coupling) -> body/radiation`.
   Adding a method = new resonator/exciter behind a stable interface.

## Working rules

- **Validation is code, not listening.** Every resonator exposes `energy()`. Correctness is asserted
  against closed-form physics: energy conservation (lossless run drifts < 1e-10), passivity (lossy
  run decreases monotonically), modal frequencies vs analytic oracle, convergence order. These tests
  exist and pass before any new model is added.
- **Do not tighten the `1e-10` acceptance bar.** The gap to the ~`1e-15` typically observed is
  deliberate headroom: this harness is the acceptance contract every implementation must meet
  under a different compiler, platform and BLAS (the viewer freeze already differs in last bits
  between Windows and Linux). Tiers and rationale in `HANDOFF.md` §6.1.
- **Energy report is mandatory** on every resonator — it is the primary bug detector.
- When `E^n` drifts in a lossless run, suspect the boundary handling (summation-by-parts) first.
- Assert `lambda = c*k/h <= 1` for explicit schemes; reject construction otherwise. Tune toward
  `lambda = 1` (exact, zero dispersion).
- Oversample around any nonlinearity (aliasing).
- Round/format numbers before display.

## Start here

The first milestone — the ideal string *plus the rig that measures its deviation from theory*
(`HANDOFF.md` §10) — was done long ago, and that pattern is how every model since was added: a
model ships with its validation harness, not after it. Nothing is queued. Direction is the
human's, drawn from `HANDOFF.md` §12 and `docs/dev/scientific-hurdles.md`.

## Open decisions — all closed (2026-08-10)

HANDOFF.md §11's five decisions are settled and kept there as a decision record. The first
(Python) was superseded by the human's own call on 2026-08-26 — see item 3. The other four stand:
explicit *and* implicit (θ-scheme from the stiff string on); polyphony is per-instance for field
models and per-voice for strings (§11.3a, budget half deferred to Phase 5); the web viewer was the
first viz target; tolerances stand as they are (§6.1). New open questions go to the human as
before — but do not re-open these.
