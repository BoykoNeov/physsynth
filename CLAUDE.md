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
   **PHASE 2 IS COMPLETE**; phases 0, 1, all five batches of 2 and five of 3 are built (plan
   §9-§20): `crates/physsynth-core` + `crates/physsynth-py`, with `string_ideal`, all of
   `operators`, `membrane`, `exciter`, `body`, `bore`, `reed`, **`mallet`**, **`string_stiff`**,
   **`string_damped`**, **`string_nonlinear`** and **`bow`**, the *builder half* of
   `operators2d`, all of `radiation` **except** its one Bessel helper, the **banded Cholesky** the
   four theta-scheme strings share, and all of `collision` **except** `BarrierString` itself — the
   contact primitives, both contact solves and the project's one **dense LU** — ported. `cargo test --workspace` runs the
   native bars and the Cargo dependency allowlist; `pip install ./crates/physsynth-py` then
   `PHYSSYNTH_RS=1 pytest` runs the **existing, unmodified** Python tests against the Rust code. The
   flag is one switch for the whole tree: with it set, five still-Python string/beam models run on
   Rust-built operators; every plate — supported, free, orthotropic, guitar-shaped — plus the von
   Karman bracket runs on Rust-built geometry; and the whole body/radiation leg (bridges,
   sympathetic strings, all three radiation tiers) runs on a Rust modal body; and the whole wind leg
   — air column, radiating bell and the reed that blows it — is Rust end to end; and the air node
   itself — far-field read-out, radiation load, rational impedance — is Rust too; and the four
   theta-scheme strings — stiff, damped, tension-modulated, geometrically exact — plus everything
   that vibrates one (the bow, the fret barrier, the bridges, the sympathetic strings)
   back-substitute in Rust, and three of the four are now Rust models outright — only the
   geometrically exact one, which needs Group D, is still Python; the whole contact leg
   — the mallet on its drumhead, the string buzzing on its fret, the sitar bridge — solves its
   contact in Rust; and the **mallet itself is now Rust end to end**, model shell included, both
   the drumhead version and the standalone mass-vs-wall rig that holds its closed-form oracle; and
   the **bow is Rust too** — the project's first continuous nonlinear exciter, friction curve,
   scalar root-find, bracketed fallback and all. Both
   implementations stay alive for now — deleting a Python model waits on its clients, not on its own
   phase (§1.2). Facts worth knowing before planning work: a file's risk group is the group of
   its hardest function, so a module can port in halves (§11.2.1); `mallet` needed `collision`, so
   **Phase 2 finished after Phase 3 started** — that prediction came due in §17; and the speed win is **per-step overhead,
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
   "how long before it cannot be compared", not "how well does it agree" (§16.5). An **eleventh**,
   from `mallet`, which is about the compiler rather than about either language and which arrived
   as a **red CI run**: **a distinction between two spellings of the same arithmetic is only
   observable while the compiler cannot see which one you meant.** LLVM constant-folds `powf` at a
   known exponent into exactly the rungs of NumPy's ladder — `powf(x, 0.5)` becomes `sqrt`,
   `powf(x, 2.0)` becomes `x*x` — so §16.2's own test, handed a literal, folded the two paths into
   one and asserted nothing; it passed in **debug** and failed in **release**, which is what CI
   builds (§17.2). The port was never wrong (the binding takes its exponent from Python at
   runtime), the *test* was — so a native test that pins an arithmetic spelling must be run in both
   profiles, and the fix is an `#[inline(never)]` wherever the exponent really is a literal, as it
   is in the mallet's `k**2` admittances. Two corollaries. **§16.5's window question needs one more
   word:** ask not "is it nonlinear" but "does the nonlinearity **recur**" — the mallet's contact is
   a transient, so its trajectory is bit-identical at 50,000 steps where the barrier's chaotic
   re-contact separates at ~1,200 (§17.5). And **a swap guard can silently cover nothing**:
   `collision` was missing from the guard table for a whole batch because three of its public names
   start with an underscore and the derive could not resolve them (§17.6).
   And a **thirteenth**, from `string_nonlinear`, which is §18.3's own rule coming due one batch
   after it was written: **a reduction's summation order matters more when something downstream
   *branches* on it than when something merely adds to it.** §18.2 deliberately left the tension
   string's stretch on `np.dot` because "it is compared to nothing"; porting the model made that
   false, and the reduction turned out not to be absorbable as a last bit at all — it sits inside a
   `brentq` residual, so the two spellings take a **different number of iterations on 1,400 of
   5,000 steps** (§19.2). So the question to ask at each reduction gains a clause: not only "does
   this reach the next timestep" but "does anything downstream branch on it". Three corollaries.
   **A public read-out can be a strictly sharper detector than the state**: the batch's one real
   porting error — a mis-associated `((dot + u_0^2) + u_last^2)` — left the trajectory
   **bit-identical for 2,000 steps** and was caught by `delta_tension`, because `beta` is ~4e-9 and
   a last bit of the tension never reaches an O(1) band entry (§19.4). **The agreement window can
   be set by an operating point inside the model**: the same code diverges 9.7e-11 at 20,000 steps
   below the parametric threshold and **2.6e-3 by step 1,000** above it, so a parity fixture's
   *amplitude* is part of its claim (§19.5) — the fourth answer to §16.5's question and the first
   that is about neither the model class nor the nonlinearity's persistence. And **a shell line
   continuation is checked by nothing**: a literal `\n` in a YAML `run:` block,
   committed with the previous batch, had been failing the parity CI step ever since — loudly,
   and for the wrong reason (§19.7).

   A **fourteenth**, from `bow`, which finishes the list of ways one expression can be two
   computations: after NumPy's ufunc ladder (§16.2) and LLVM's constant fold (§17.2), the third is
   a **hoist somebody wrote by hand**. `bow.py` evaluates its friction residual twice — the scalar
   path multiplies by `g` last, the array path factors `g * force * sqrt(2a)` out so NumPy can
   apply one scalar to the whole 512-point scan — and those are different doubles in **4,158 of
   20,000** samples at the flagship fixture (568-5,372 across the three the suite builds; the
   fraction is set by how large `g*force*sqrt(2a)` is next to `v - v_free`, so it is a property of
   where the bow is played). Unlike the first two this one is *visible in the source and reads like
   a duplicate that wants tidying*, which is exactly why it needs a pin in both suites: merging them
   changes **which brackets exist**, and at a slip event that is a different branch, while no physics
   bar moves (§20.2). Its negative result is worth as much but is **a claim about a runner**: `np.exp`
   on an array and `math.exp` on a scalar agreed 20,000/20,000 *on Windows*, where all three of
   NumPy, CPython and Rust reach UCRT — on Linux NumPy uses its own SIMD loop and only the other two
   reach glibc. The exposure is bounded and cannot reach the trajectory (the scan decides only
   whether a bracket *exists*, and `brentq` re-evaluates through the scalar path). A methodological
   scar with it: the first draft of that 306 came from a scratch script that had hardcoded
   `g = 1e-3` when the model's is 0.318 — **a measurement taken at a parameter the model never uses
   is §16.4's blind fixture seen from the measuring side.**
   Three corollaries. **§19.2's branch rule needs a size**: the tension string's reduction flipped
   an iteration count on 1,400 of 5,000 steps because the perturbation and the bracket test were the
   same size, whereas here a ~1e-14 *relative* perturbation is asked about a 1e-13 *absolute*
   tolerance on a quantity of order 0.1, so the branch never flips and the count differs at most
   once in 20,000 steps (§20.3) — ask how far the fed quantity sits from the threshold, not only
   whether it reaches one. **A recurring nonlinearity can CONTRACT**: §17.5 said to ask whether the
   nonlinearity recurs, and the bow's does — once per period, forever — yet the two implementations
   do not separate at all (flat at ~1e-14 out to 20,000 steps), because Helmholtz motion is a
   *stable limit cycle* and a perturbation is squeezed back onto it. The missing word is whether the
   recurrence drives the system **onto** an attractor or **off** one (§20.5). And **the normaliser
   is part of the claim**: the same runs read 2.9e-12 normalised by the *instantaneous* amplitude
   and 6.6e-14 normalised by the *running peak*, because Helmholtz motion beats and the
   instantaneous maximum passes through near-nodes — §14.2's "normalise by amplitude" needs
   "monotone" added to it (§20.6). Two more, neither about arithmetic: §19.7's literal `\\n` in a
   YAML `run:` block was **reintroduced by the batch that cites it**, because the failure is
   introduced by *tooling* rather than by typing, so it is now asserted in `tests/test_ci_workflow.py`
   rather than remembered (§20.7); and **`bow` is not the phase's last model** — `BarrierString` was
   described in §16 as waiting on its host `DampedStiffString`, that host landed in §18, and nobody
   revisited the sentence, so Phase 3 is not finished (§20.11).

   A **fifteenth**, which is not a batch but a **red CI run on unchanged code**, and which retires
   the assumption every exact claim in the suite was written on: **NumPy does not call the platform
   C library for the transcendentals — it carries its own vectorised routines, chosen at import
   from the CPU's feature set — so a bit-identity claim whose value passes through one of them is a
   claim about the machine GitHub happened to hand the job** (§22.1). Two runs of effectively
   identical code, same runner image and same NumPy, read one failure and then eighteen. The
   diagnosis is not a guess: fifteen of them were `collision`'s powers, and the pass/fail split
   follows §16.2's shortcut ladder *without a single exception* — every exponent NumPy spells as
   `sqrt`/`x*x`/`x`/`1`/`1/x` agreed, every exponent it hands to its own `pow` did not; the other
   three were one `np.tan`, whose transcendental-free sibling passed. This is the **fourth** way one
   expression is two computations (after the ufunc ladder §16.2, LLVM's fold §17.2 and a hand hoist
   §20.2) and the first that is **invisible in the source of either language**. The human's call is
   *exactness where it is provable, an ulp bound where it is not*: exact only at exponents on the
   ladder (`alpha = 1.0` is the sole value putting all three of a contact primitive's exponents
   there at once), four ulp with a printed count elsewhere, and where a portable spelling is free,
   prefer it — `impedance_discrete` moved from `np.tan` to `math.tan` and keeps its exact
   assertion, which is `portable.py`'s manoeuvre a **fourth** time and the first aimed at a
   transcendental rather than a summation order (§22.3). Four corollaries. **A local repro is not
   available and that is structural**: `NPY_DISABLE_CPU_FEATURES` does nothing on Windows because
   there is no dispatched `pow` loop to strip — both languages reach UCRT — so this whole class is
   invisible to local development and the workflow now prints the CPU model and
   `numpy.show_runtime()` (§22.2). **§16.4's blind spot is set by a physical parameter**: moving the
   soft-rail test to `alpha = 1.0` for exactness *lost* the blind spot, because the tangent
   stiffness vanishes at grazing contact for `a > 1` and is the flat constant `K` at `a = 1` —
   `cond(J)` is 1.0625 at 1.0 against 1.0032 at 1.5 — so "pick a fixture that engages the solver"
   and "pick a fixture whose arithmetic is provable" can be in **direct conflict**, resolved per
   test (§22.4). **A tolerance must be read against the expression, not the port**: the discrete
   gradient's derivative divides by `da^2` after a cancellation of the same order, so a one-ulp
   nudge moves it by **15% of its own scale** near the branch cutoff — no elementwise bound is
   meaningful there, and the fix is a fixture that keeps `|da|` away from `tol`, which buys
   agreement at exactly the `1/da^2` rate (§22.5). And **the negative result is the useful half**:
   `np.cos` was already documented as being in this class and **passed on the divergent machine**,
   so the exposure is per-function and per-CPU, not a blanket property — the whole exposed surface
   across ported modules is four powers, one `tan`, one `exp` and three `cos`/`sin`, of which the
   guitar outline's is the one where a last bit is not an ulp but a **live/dead node** (§22.6).

   And a **twelfth**, from the two theta-scheme strings, which is really the same finding twice and
   is about **SciPy rather than about Rust**: what blocked the first *model* out of the four-string
   chain was not the solver §15 had already dealt with but an **order of evaluation**, in two
   places, and both were fixed by changing the **Python** side (§18.2). `np.dot` is BLAS `ddot` and
   disagrees with a left-to-right sum in ~84% of this family's vectors, which matters because all
   three chain anchors assert `a.energy() == b.energy()` across model classes; and
   `biharmonic_matrix` comes back from SciPy's sparse-product kernel with **descending** column
   indices, so `L @ u` — which builds every timestep's right-hand side — is a different sum from
   the sorted spelling in **2,000 of 2,000** vectors. A new module, `physsynth/core/portable.py`,
   holds both spellings and is applied to all four theta-scheme strings at once; it is the Phase 1
   manoeuvre a third time, one level lower again (not a model, not a solver, an *order*). Three
   corollaries. **A decision justified by "nothing downstream depends on this" expires the moment
   something downstream ports** — §10 kept the Rust `Csr` canonical partly because nothing read
   `.indices`, which is now false, and the decision survived only because its *other* reason got
   stronger (§18.3); the plate family hits the same wall at Phase 5 and the answer is written down
   in advance (§18.4). **Ask whether a reduction reaches the next timestep**, not whether it is a
   reduction — `string_nonlinear`'s stretch integral is inside the tension solve and was
   deliberately left on `np.dot` (§18.2). And §16.5's agreement-window question gets its third
   answer: **a linear model does not amplify a difference at all**, so the introduced solver gap
   grows like a random walk (1.7e-14 at 100 steps, 1.6e-13 at 20,000) rather than exponentially,
   and what sets the window is amplification, not perturbation size (§18.6).
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
