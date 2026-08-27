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
   **Phases 0, 1, four batches of 2 and two of 3 are built** (plan §9-§16):
   `crates/physsynth-core` + `crates/physsynth-py`, with `string_ideal`, all of `operators`,
   `membrane`, `exciter`, `body`, `bore`, `reed`, the *builder half* of `operators2d`, all of
   `radiation` **except** its one Bessel helper, the **banded Cholesky** the four theta-scheme
   strings share, and all of `collision` **except** `BarrierString` itself — the contact
   primitives, both contact solves and the project's one **dense LU** — ported. `cargo test --workspace` runs the
   native bars and the Cargo dependency allowlist; `pip install ./crates/physsynth-py` then
   `PHYSSYNTH_RS=1 pytest` runs the **existing, unmodified** Python tests against the Rust code. The
   flag is one switch for the whole tree: with it set, five still-Python string/beam models run on
   Rust-built operators; every plate — supported, free, orthotropic, guitar-shaped — plus the von
   Karman bracket runs on Rust-built geometry; and the whole body/radiation leg (bridges,
   sympathetic strings, all three radiation tiers) runs on a Rust modal body; and the whole wind leg
   — air column, radiating bell and the reed that blows it — is Rust end to end; and the air node
   itself — far-field read-out, radiation load, rational impedance — is Rust too; and the four
   theta-scheme strings — stiff, damped, tension-modulated, geometrically exact — plus everything
   that vibrates one (the bow, the fret barrier, the bridges, the sympathetic strings) now
   back-substitute in Rust while the models themselves are still Python; and the whole contact leg
   — the mallet on its drumhead, the string buzzing on its fret, the sitar bridge — solves its
   contact in Rust, again with the two model shells still Python. Both
   implementations stay alive for now — deleting a Python model waits on its clients, not on its own
   phase (§1.2). Seven facts worth knowing before planning work: a file's risk group is the group of
   its hardest function, so a module can port in halves (§11.2.1); `mallet` needs `collision`, so
   **Phase 2 finishes after Phase 3 starts** (§11.2.2); and the speed win is **per-step overhead,
   not arithmetic** — 8.7x on a small grid, ~1.1x once SciPy's compiled matvec dominates, so the
   *test suite* does not get faster and the *real-time* case does (§11.6) — and the crossover is
   about whether NumPy's hot path was **already compiled**, not about size: the modal body, which
   calls no compiled kernel at all, is **~15x** at every realistic mode count (§12.7). A fourth
   fact, from `body`: a leading **underscore is not a statement about the interface** — three
   modules assign to the body's `_accel` every timestep (§12.2). And a fifth, from `bore`: a **`&mut
   self` pymethod cannot hand control back to Python and still be read** — the reed's hook does an
   ordinary `self.bore.p[0]` and PyO3 refuses it, so a model that calls out mid-step takes the
   *object* and borrows it in two phases (§13.2). That failure is invisible to `cargo test`, which
   never crosses the boundary. And a sixth, from `radiation`, which retires an assumption the first
   five batches were built on: **bit-identity is not available everywhere, and what ends it is a
   BLAS reduction that feeds back into state — not a solver** (§14.2). `np.dot` fuses its
   multiply-add and OpenBLAS picks its kernel by CPU, so matching it would be a claim about a
   runner. The bar there is Group A over a *short* run and the physics bars thereafter, and the
   question to ask of every remaining model is "does a reduction reach the next timestep?" One
   corollary bites immediately: a fused multiply-add differs from a rounded one only when the
   **product** rounds, and every body fixture in `tests/helpers.py` uses weights of 1.0, so the
   suite is systematically blind to this class of divergence — a comparison of reductions needs at
   least one fixture whose coefficients are not powers of two (§14.3). A seventh, from the
   banded solver, which is about `tests/` rather than about arithmetic: **a bit-identity anchor
   between two different model classes is a porting constraint that binds them into one unit**
   (§15.2). Three `array_equal` reduction anchors chain the four theta-scheme strings together, one
   of which is Group D — so the phase had to port the *solver* rather than any model, the Phase 1
   manoeuvre one level down, and the question to ask before porting any model is "does the suite
   compare it, to the bit, against a **different** class?" Two corollaries. The Group A target is
   **run-length-dependent and shorter than it looks**: a fed-back reduction held 1e-13 out to 2,000
   steps (§14.4), a fed-back *solve* holds it only to ~100 (§15.4) — while the energy bar does not
   move at all. And **validation written into a swap block is on the hot path**: one
   `np.isfinite(...).all()` in Python turned a 1.47x win into a 0.96x loss, because what is being
   spent is per-call overhead and that check is another call of exactly that kind (§15.5).
   And an eighth, from `collision`, which is about NumPy rather than about either language: **a
   vectorized function called with an array and with a scalar is two different computations.**
   NumPy's float64 power *ufunc loop* shortcuts the exponents -1, 0, 0.5, 1 and 2 (`x**0.5` is
   `sqrt`, `x**2` is `x*x`) and its *scalar* path does not — so a port has to carry both spellings
   and pick by argument rank, and one existing docstring claiming the two paths are identical is
   measurably wrong (§16.2). A ninth, from the same batch, is about what the tests can see: **the
   fixture the suite uses most can be the one that does not exercise the thing being ported.**
   The barrier's default contact makes the Newton Jacobian 1.004 — near enough to the identity
   that the new dense LU is a no-op and 79 nodes in contact still come out bit-identical — so a
   parity test must bring a fixture chosen to exercise the *solver*, not the physics (§16.4). And
   a tenth, which retires a question rather than answers one: **for a nonlinear model the
   agreement window is set by the dynamics, not by the port.** A string buzzing on a barrier is
   chaotic, so the two trajectories separate exponentially — 1e-13 at a thousand steps, 1e-7 at
   twenty thousand — and the right question before porting the four remaining nonlinear models is
   "how long before it cannot be compared", not "how well does it agree" (§16.5).
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
