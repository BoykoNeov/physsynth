# Rust migration — the findings ledger

> **What this is.** The fifty-nine findings the migration has made about when two
> implementations of the same arithmetic agree to the bit and when they cannot — and about the
> ways a port can be silently wrong while every physics bar stays green. Until 2026-09-02 this
> text lived verbatim inside `CLAUDE.md`'s non-negotiable #3, where it had grown to 70 KB in a
> file whose first line calls it "lean, always-loaded context". It moved here unchanged; each
> finding names the plan section (`docs/dev/rust-migration-plan.md` §N) that measured it, and
> the plan section is the authority where the two disagree.
>
> **How to use it.** Read the index before scoping any batch, and open the finding whose hook
> matches the shape of the work. The questions to ask before writing an *exact* assertion are
> collected at the end.

## Index — one line per finding

| # | Hook | Where |
|---|------|-------|
| 1 | A file's risk group is the group of its hardest function, so a module can port in halves | §11.2.1 |
| 2 | `mallet` needed `collision`, so Phase 2 finished after Phase 3 started — what a model waits on is not visible from the model | §11.2.2, §17 |
| 3 | The speed win is per-step **overhead**, not arithmetic: 8.7x on a small grid, ~1.1x once a compiled kernel dominates; the crossover is whether NumPy's hot path was already compiled | §11.6, §12.7 |
| 4 | A leading underscore is not a statement about the interface — three modules write the body's `_accel` every step | §12.2 |
| 5 | A `&mut self` pymethod cannot hand control back to Python and still be read; a model that calls out mid-step borrows in two phases | §13.2 |
| 6 | Bit-identity ends at a **BLAS reduction that feeds back into state**, not at a solver; the question is "does a reduction reach the next timestep?" | §14.2, §14.3 |
| 7 | A bit-identity anchor between two model classes is a porting constraint that binds them into one unit — port the *solver* instead | §15.2, §15.4, §15.5 |
| 8 | A vectorised function called with an array and with a scalar is two computations (NumPy's power-ufunc ladder) | §16.2, §16.4 |
| 9 | For a nonlinear model the agreement window is set by the dynamics: ask "how long before it cannot be compared" | §16.5 |
| 10 | A spelling distinction is observable only while the compiler cannot see which one you meant — LLVM folds `powf` at a literal exponent | §17.2, §17.5, §17.6 |
| 11 | What blocked the first theta-string was an **order of evaluation** in SciPy, fixed on the Python side (`portable.py`) | §18.2, §18.3, §18.6 |
| 12 | A reduction's summation order matters more when something downstream **branches** on it (a `brentq` iteration count) | §19.2, §19.4, §19.5, §19.7 |
| 13 | The third way one expression is two computations: a hoist somebody wrote by hand (`bow`'s two residual paths) | §20.2, §20.3, §20.5, §20.6, §20.7, §20.11 |
| 14 | NumPy does not call the platform libm — a bit-identity claim through a transcendental is a claim about the runner's CPU | §22.1, §22.3, §22.5, §22.6 |
| 15 | Whether bit-identity survives a ported reduction is decided by how many terms it has; at two terms it is provable | §23.2, §23.3, §23.5, §23.6, §23.7 |
| 16 | Matching SuperLU is a claim about how SciPy was *built* (supernodal); Group D gets tolerance-level agreement, quantified | §24.2, §24.4, §24.5, §24.7, §24.9 |
| 17 | A function whose output is a **decision** (a mask) is a different porting problem from one whose output is a number | §25.2, §25.3, §25.4, §25.7, §25.8 |
| 18 | When a port meets a library kernel, whether the *values* agree and whether the *stored order* agrees are separate questions | §26.2, §26.3, §26.5, §26.6, §26.7 |
| 19 | An association can move the **sum** rather than the products — re-associate so the contraction runs over a canonical operand | §27.2, §27.3, §27.4, §27.5 |
| 20 | An anchor can bind a model to a client that is not being ported (`airbox` re-derives the plate's system matrix) | §28.2, §28.3, §28.5, §28.6 |
| 21 | A port can change no digit and still be wrong: a Newton Jacobian factored in the natural order fills 13x — assert the **work** | §29.2, §29.3 |
| 22 | `np.sum` has a written-down cutoff (eight), so exactness across a ported sum is decidable by counting terms | §30.2, §30.3, §30.5 |
| 23 | NumPy's pairwise blocking is one fixed algorithm, not a kernel — it transcribes exactly (`reduce::sum`), and the refusal above it was avoidable | §31.2, §31.3, §31.4, §31.6, §31.9 |
| 24 | A tier below can decide what porting the tier above is allowed to *mean* (replaceable collaborators) | §32.2, §32.3, §32.4, §32.6 |
| 25 | A `#[getter]` with no `#[setter]` makes a plain attribute read-only — start from "every attribute is writable" | §33.2, §33.3, §33.4, §33.6, §33.7 |
| 26 | A ported caller that computes nothing is **slower** than the Python it replaced — ask how much work sits between collaborator calls | §34.2, §34.3, §34.4, §34.5 |
| 27 | Six searches for a blocking dependency (private names, re-derivation, duck-typed types, written attributes, replaced methods, replaced collaborators) and none finds the others | §30.3, §31.6, §32.2, §33.2 |
| 28 | A negative control whose predicate is a per-CPU kernel is a claim about the runner — report it, or search the whole expression for a witness that survives | §35.2 |
| 29 | A decision's **margin** and the **axis** it is decided on are separate questions; a decision-valued module ports safely when the tight decisions sit on the exact axis | §36.2, §36.3, §36.5, §36.6, §36.8 |
| 30 | The **denominator** of an agreement bar is a modelling choice: for anything oscillatory the honest one is the amplitude, not the sample — a relative bar on a function that crosses zero is a claim about where the fixture landed | §37.3, §37.4 |
| 31 | When two crates that must not depend on each other both need one transcription, ask which direction is **real** — a helper no `step()` calls is not a core dependency however core a file it sits in, and checking that is a grep | §37.2 |
| 32 | A **derived** file list is only as honest as the reachability it models: one hop through `tests/helpers.py` made a CI grep wrong by 43 of 65 files, and a floor cannot catch a grep asking a narrower question than its name claims | §37.8 |
| 33 | A differing solver **iteration count** is a defect only when the iterate is fed forward; when only the root survives, two implementations may take visibly different paths to the same answer. And the general form: **a margin measured at one fixture is a claim about one fixture** — the fix is a grid, not a better probe | §38.1, §38.2 |
| 34 | What may cross a deliberate crate boundary is decided by **what the shared text is**: a numerical method with no physics in it may be included; the discretisation under test may not, or the oracle would agree with a divergent core by construction | §38.4 |
| 35 | A swap that **delegates** keeps its Python `__module__`, so "is this object Rust?" is the wrong detector for an import audit — ask whether the name's **definition site moves** when the flag moves, which is a two-process fingerprint diff and not a reading of the source | §39.1 |
| 36 | An anchor's blast radius is a property of **one test**, not of a file: a single six-class spelling assertion welded ten modules into one deletion unit, and reading that one test dissolved the blob into nine. Corollary: sort a mixed file by reading each body — a `for cls in (...)` loop is a property test wearing a comparison's clothes | §39.2, §39.5 |
| 37 | An enumeration built from a **naming convention** is a claim about the convention, not about the code: `core/` marks its reference implementations `<Name>Py` and `analysis/` marks its own `<name>_py`, so a graph built from the first covered six modules and 2,458 lines in no unit at all. Catch it by checking the counts against the files on disk | §40.4 |
| 38 | #25's read-only getter comes back at **deletion** time and is a different blocker: a test reaches for the reference implementation *because* it permits a write the binding forbids (`plate.B = ...`), so a test that injects state blocks a deletion rather than a port, and no porting bar could have seen it | §40.5 |
| 39 | **The binding depends on Python**, and that is what pins each module's floor: four places in `physsynth-py` call `py.import("physsynth.core...")` and pull an object back out — `GeometricState`, `GrainSpec`, and two whole module namespaces. A fifth search for a blocking dependency, running Rust -> Python, which no audit of Python's own imports can find. One grep: `import("physsynth` | §41.2 |
| 40 | A canary that counts a **draining** population must name it instead of flooring it: a floor has to be hand-lowered per deletion and becomes unsatisfiable at zero, where "the exclusion excludes nothing" stops being a bug. And the property the canary was really testing — does the predicate fire? — usually has a subject that does not drain | §41.5 |
| 41 | Three CI guards were **counting a draining population** and each needed a different cure: write down the set when the population is the subject, name a **positive control** when the count only ever proved the scan fires, or **widen** to a population that grows. Ask of any threshold what it does at the end of the migration — a floor whose answer is "unsatisfiable" is measuring the wrong thing already. Widening one found a latent bug whose fix was a *scope* rule, not a rewrite | §41.5, §42.2, §42.3 |
| 42 | A captured-binding check is retired only when **both** sides of the capture are Rust; while either is Python it still catches a mis-ordered swap, and retiring it early leaves a vacuous pass. And a parity file can survive a deletion losing only **half** of itself — sort per *test*: untouched (the comparand is not Python), retired (both sides were the implementations), or kept with the scaffolding removed | §42.4, §42.5 |
| 43 | #38's read-only getter was cheaper to **remove than to route around**, and the honest cost was a documented invariant break: one constructor that does not sort, plus `debug_assert`s on the three kernels that need sorted rows. The setter alone would not have been enough — `Csr` sorts, so the injected operator would have come back as the shipped one and the comparison would have passed while comparing a plate against a copy of itself. **A data structure that cannot represent the difference being measured is §23.6's emptied comparison through a fifth door** | §43.1, §43.2 |
| 44 | A deletion reaches the clients that **re-derive** the deleted module's arithmetic and compare the result exactly — a dependency with no import, no attribute access and no shared name. Fifteen tests in three modules outside the unit failed at ~1e-16 because the plate's solver moved to Rust on the default path and its room-loaded wrappers still factored with SuperLU. The tell is an `array_equal` in a test on a module you are not deleting; the remedy may be a **solver** rather than a name | §43.4 |
| 45 | A reflective test over `dir(model)` is a claim about how the model **stores** its attributes, which is exactly what a port changes: `dir()` of a `#[pyclass]` lists branch-only getters that *raise* (a supported plate has no `K`), and a getter may build a **fresh object per access** so `x.attr is x.attr` is already false. Ask the instance (`hasattr`), and compare values only where the model itself hands back a stable one | §43.5 |
| 46 | **A deletion changes which tests share a process**, so it surfaces order- and process-dependent defects that have nothing to do with it — here a fixture seeded `hash(str(x))`, which Python randomises per process, under a **pointwise** relative bar that divides a last bit by a near-cancellation (worst pointwise 1.1e-13; worst scaled 6.0e-16). The first instinct "my change broke this" is wrong about half the time, and the discriminator is cheap: run the file alone, then run the suite on the stashed tree | §43.6 |
| 47 | **A freeze is a real substitute for a second implementation, with one named limit**: the check can be *repeated* but never *re-derived*. Structure and integers compared exactly (a search returning a different number of roots is not a small error), floats on one tolerance with the observed gaps recorded per case as a record rather than a bar. Derive the case list from `__all__` so a new function cannot be added unfrozen, and assert that every case carries a real measurement rather than a reason one could not be taken | §44.1, §44.2 |
| 48 | Measure the two sides of a recording **through the public names, in a subprocess with the flag set** — never by guessing the binding's name from the Python one. The names are not a mechanical transform, and several public names are wrappers that *adapt arguments*, so a gap taken against the raw binding measures something nobody runs. Guessing reported "no twin" for 15 of 62 cases that have one | §44.2 |
| 49 | **A module header is a claim, and nothing checks it.** `crates/physsynth-analysis/tests/rotating_wave.rs` listed the Jacobian identities among what it asserted, for a phase, while they existed only in a Python test — found because the deletion forced a comparison between the retiring test's name and the header's sentence. A deletion audits the documentation as well as the code. And when the blocked test reaches *private internals*, the cure is a `mod tests` **inside** the module, not a widened API | §44.4 |
| 50 | **Scan a finite-difference step, never choose it.** Moving a Jacobian bar failed at 1.5e-6; the gap runs 3.0e-9 · 4.9e-8 · 1.5e-6 · 1.5e-5 as the step shrinks from 1e-5 to 1e-8, so it *grows* as the step shrinks — cancellation in the difference, not truncation in the derivative. A check that takes the smallest step it can is measuring its own subtraction, and a step carried across from another language's arithmetic is carrying an accident | §44.5 |
| 51 | **A recording mechanism applies one rule to everything it records**, and a hand-written comparison did not: each *exactly / to a tolerance / not at all* was an argument somebody had, and a generic flattener cannot know which integers were arguments. Freezing `RotatingWave` pinned its Newton **iteration count** — the one quantity #33 says is not comparable — silently and green, one commit after deleting the test whose whole job was to say so. Sweep every integer and boolean in a record: **structure** (length, label, count — exact, and often the sharpest thing there) or **decision** (needs a recorded margin, or an exclusion) | §44.9 |
| 52 | **An empty `parametrize` collects as a SKIP**, so a draining table has to be *deleted* at zero, not emptied. #40 is a floor on a draining population becoming unsatisfiable; this is the population itself reaching zero, where the mechanism converts silently to green. Write the exit condition down when the list is created — `test_rust_parity.py`'s own docstring had prescribed "when it is empty this file goes" a batch before it was needed | §45.5 |
| 53 | **The matrix a model already factored is factored for its timestep, not for its spectrum.** Inverse-iterating on the beam's own `A = W + theta k^2 kappa^2 K` is not slow but impossible — the coefficient is 4.4e-11, so consecutive modes sit one part in 1e7 apart (~1e8 sweeps) and reading the eigenvalue back subtracts 1 from 1.000000022. Factor a separately shifted `K + eps W` instead. Corollaries: project a known nullspace out **every** sweep (its shifted eigenvalue is the largest in the problem, so roundoff regrows it fastest), **fix** the sweep count rather than stopping on a tolerance (#33), read the eigenvalue back as a **Rayleigh quotient**, and pick a seed that is not orthogonal to the mode you are asking for | §45.2 |
| 54 | **A threshold written for a single mode is a statement about the fixture's spectrum when a pluck is fed to it.** A lossy beam retains 29% of a broadband pluck where one low mode retains 2e-3, because this scheme's high modes underdamp — so "the loss did work" is fixture-tuning waiting to happen. Assert the **gap** to the single-mode rate instead: it is a claim about the model, and it makes the passivity bar and the caveat bar say the same thing from two directions | §45.4 |
| 55 | **A native bar is not a translation of the Python one, and where it differs it is usually sharper.** Derive an oracle rather than importing it (an oracle that imports the implementation is not an oracle) and spell its residual so one tolerance means the same accuracy at every root — `cos - sech`, not `cos*cosh - 1`. And prefer an **algebraic identity of the update** to a measurement of its output: `u^{n+1} + u^{n-1} = 2 cos(omega k) u^n` holds pointwise at 1e-13 where the FFT bar it replaces could only reach five cents | §45.1, §45.3 |
| 56 | **A default the test helpers always override is dead code as far as the suite is concerned.** Every θ-scheme model's `theta` default lives twice — in the Rust constructor's signature and in a Python module constant — and `tests/helpers.py` passes `theta` explicitly on every construction, so the binding's default could drift to any value with the whole suite green. Related: a **shim that re-declares** a constant instead of re-exporting it silently forks it, and the neighbouring module's header claiming to own the number is #49 again (it was already wrong about two of its five named importers) | §45.9 |
| 57 | **A capability inventory has to ask what the test *runner* runs, not what a directory holds.** §39.3 filed `airbox` under "zero native bars" on the strength of `ls crates/physsynth-core/tests/`; the module carries nineteen `#[test]`s in `mod tests` blocks inside `src/`, which `cargo test --workspace` runs identically. Six core modules use the inline form and the split is "needs private access" versus "is a model", not "is tested" versus "is not". #49 (a header is a claim) one level up, at an audit table — and the cure is the same, check the sentence against the thing it describes | §46.1 |
| 58 | **One algebraic identity can be exact in doubles at one fixture and not at another, and which you pick decides whether `==` is available.** The grid-diagonal mode at `lambda = 1/sqrt(3)` is exact by cancellation of an arcsine against a sine; on a cube it lands at 1.1e-16 to 3.3e-16 (a `pow`, a square root, an arcsine, a sine) while the **corner** mode of a non-cube is exactly `0.0`, because every axis reaches `sin(pi/2)` — the one argument the sine returns without rounding. Assert each at what it is: writing the tolerance twice throws away a real exactness, writing the equality twice is #14 waiting for a different CPU | §46.2 |
| 59 | **A growth law is a claim about differences, not about the sequence.** Windowed peaks normalized by the seed asserted against `[1, 2, 3, 4]` passed at 60–80% of its threshold, because secular growth is `a + b i` and window 1 already contains some — so the integers state `a`, a property of the initial condition. Assert the differences are equal (plus a separate "and it did grow", or a flat field passes with every difference zero). #54's shape in a second place: a threshold carried across from another bar measures the fixture it was carried onto | §46.3 |

## The questions to ask before writing an exact assertion

1. Does the reduction reach the next timestep? (#6) Does anything downstream branch on it, and how far does the fed quantity sit from the threshold? (#12, #13)
2. How long is the sum? Below eight terms `np.sum` is a plain loop; above, use `reduce::sum`. (#15, #22, #23)
3. Does the value pass through a transcendental? Then it is a claim about a CPU unless spelled through `math.*` / libm on both sides. (#14)
4. Is the output a number or a decision? (#17)
5. Does the suite compare this class, to the bit, against a *different* class — or a client that re-derives it? (#7, #20)
6. Does the nonlinearity recur, and does it drive the system onto an attractor or off one? Read `n_iters`. (#9, #19)
7. What did the tier below promise its clients it would let them replace, and what does a client *write*? (#24, #25, #27)
8. How much work sits between the collaborator calls? (#3, #26)
9. Which *axis* is each decision made on — one built from `+ - * /`, or one running through a library kernel — and how much margin does it have? Measure the margins over the suite's real calls before claiming any of them is exact. (#29)
10. Does the assertion require a *difference*? That is as machine-dependent as requiring equality. (#28)
11. What is the bar *divided by*? If the quantity oscillates through zero, or is a difference of two close numbers, a relative bar reports the fixture rather than the port. (#30)
12. Was the margin measured at **one fixture or over a grid**? A probe that holds `N`, the mode and the amplitude fixed answers for that cell only — and for a solver, ask separately whether the thing that moved is *returned* or merely *traversed*. (#33)

---

## The ledger, verbatim (as it stood in `CLAUDE.md` on 2026-09-02)

~~**Prototype in Python (NumPy/SciPy).** Julia acceptable if the human prefers. Not C++/JUCE
yet.~~ **SUPERSEDED 2026-08-26 (the human's call):** Python is being retired entirely — core,
analysis, viewer backend *and* the test suite — in favour of **Rust**, gradually and model by
model. See `docs/dev/rust-migration-plan.md`; it also supersedes the portability contract's
"Python stays the reference oracle" clause and absorbs HANDOFF §9's Phase 5.
**PHASES 2, 3 AND 4 ARE ALL COMPLETE** and **PHASE 5 IS UNDER WAY**; phases 0, 1, all five
batches of 2, all six of 3, phase 4 and the first ten batches of 5 are built (plan §9-§34) —
**`physsynth/core/` IS FINISHED; only `analysis/` is left**: `crates/physsynth-core` + `crates/physsynth-py`, with `string_ideal`, all
of `operators`, `membrane`, `exciter`, `body`, `bore`, `reed`, **`mallet`**, **`string_stiff`**,
**`string_damped`**, **`string_nonlinear`**, **`bow`**, **all of `collision`** — the contact
primitives, both contact solves, the project's one **dense LU** and `BarrierString` itself —
and **`beam`**, and now **all of `operators2d`**: the builder half, the guitar-outline geometry,
all five of its matrices (the biharmonic, the interior second difference, the orthotropic
bending operator and both free-plate stiffnesses) and, as of batch 3, the **nonlinear plate** —
the five private 1-D differences, the **von Kármán bracket** and the **clamped Airy solve** —
plus all of `radiation` **except** its
one Bessel helper, the **banded Cholesky** the four theta-scheme strings share and the
**sparse LU** the whole of Group D shares — and, as of batch 4, **all of `plate`**: both
classes at once, the linear plate on every one of its four branches (supported, free,
circle, guitar) with both grains, and the **von Karman plate** with them — and, as of batch 5,
**`string_geometric`** (model #10, the geometrically exact string), which is the **last of the
four theta-scheme strings**, so the whole string family is Rust and the three bit-identity
anchors that chain it now compare one Rust class against another — and, as of batch 6, the
**`AirBox`** class itself, the 3-D room, which is the **first half of a file** rather than a
file: `airbox.py` is 3,976 lines and only the room ported, the ports and the six `RoomLoaded*` /
`RoomSuspended*` wrappers above it staying Python and working unchanged — ported; and as of
batch 7 the **port tier** on top of it — `RoomPort`, `SurfacePort`, `InteriorSurfacePort` and
the `_free_pressure_nodes` helper they share — is Rust too, so a room and everything that opens
a terminal into it are ported; and as of batch 8 the **wrapper tier** on top of *that* —
`RoomLoadedBody`, both plate wrappers, both von Kármán wrappers and the two seams
(`_PlateSurface`, `_VKPlateSurface`) they drive; and as of batch 9 the **membrane pair**, its
seam and its mixin — plus the two module-level helpers (`_face_axes`, `impedance_from_zeta`)
that belonged to no tier and had been missed by every "one tier left" note — so **`airbox.py`
is finished**: every class it exposes and every function any outside client reaches is Rust.
(Two things are deliberately not swapped, checked rather than assumed: the module's constants,
which are values and not implementations, and `_require_same_rate`, which has a Rust twin but no
caller outside the file — it stays live for the same reason `AirBoxPy` does.) And as of **batch
10 `connection.py` — all four bridges — is Rust, which finishes `physsynth/core/`**; the only
thing left in the whole migration is **`analysis/`**. The order that got here was a finding
rather than a preference: §28.11 called `connection` the cheapest
remaining file because it touches no private names, and §29.1 found that the instrument was
wrong. `connection` is **polymorphic over its collaborators' types** — its three bridges are
handed Rust bodies and plates *and* `airbox`'s duck-typed wrappers, with three `test_airbox_*`
files pinning `bridge.stability_margin == bare.stability_margin` **exactly** across that
boundary — so it could not move before those wrappers did. **As of §32 it can**: all three
collaborators §31.11 named (`RoomLoadedBody`, `_PlateSurface`, `RoomSuspendedPlate`) are Rust,
the bridge tests pass under the flag with a Rust wrapper standing in for a plate, and
`connection` needs no membrane wrapper. §31.11 also corrects §29.1 on one point worth keeping:
`connection.py` contains no `isinstance`, `hasattr`, `getattr` or `type(` at all, so it is *pure
duck typing*, and what blocked it was only that the wrapper objects must exist and answer —
to which §33.2 adds one clause: **and answer to a write**. (§33.11's *other* claim, that the
file reaches no private name, is **wrong**: it reaches `string._bc_right` four times and
`string._second_diff` twice, and it cost nothing only because Phase 0 exposed both deliberately
three phases early — §34.4.)
`cargo test --workspace` runs the native bars and the Cargo dependency allowlist;
`pip install ./crates/physsynth-py` then
`PHYSSYNTH_RS=1 pytest` runs the **existing, unmodified** Python tests against the Rust code. The
flag is one switch for the whole tree: with it set, the one still-Python string model runs on
Rust-built operators; every plate — supported, free, orthotropic, guitar-shaped — runs on
Rust-built geometry **and on Rust-built stiffness matrices**, and the **nonlinear** one sources
its Airy stress function from a Rust factorization and forms its coupling force with a Rust
bracket; and as of batch 4 **every plate in the tree is a Rust model outright** — the shipped
plates, the gong, the cymbal, the guitar soundboard, the bodies the bridges drive and the
surfaces the room loads — with `airbox.py`, which is *not* ported, given the one swapped name
(`splu`) its own bit-identity claims turn out to need; and the
whole body/radiation leg (bridges,
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
scalar root-find, bracketed fallback and all; and so is the **barrier string**, so the fret, the
sitar jawari and the tanpura thread are Rust end to end and the viewer draws three of its models
from a Rust one; and the **free-free beam is Rust**, carrying with it the **sparse LU** the rest
of Group D will use. Both
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
revisited the sentence, so Phase 3 was not finished (§20.11). It is now: §23 ported the barrier
and closed the phase.

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

And a **sixteenth**, from `BarrierString` — the phase's true last model, and the one that closes
it: **whether bit-identity is available across a ported reduction is decided by how many terms
the reduction has, and at two terms it is provable rather than lucky.** The barrier's shell
injects its contact force through a dense BLAS matvec on the *update* path, which nothing had
compared across the languages while the shell was Python on both sides of every comparison. It
differs from a left-to-right row sum in **0 of 158,000** rows at one contact node (the sum is one
product), **2,232-3,719** at two, and **45,822** at seventy-nine — and yet the trajectory is
bit-identical at two nodes and not at seventy-nine. The first explanation written for that was a
magnitude argument ("the correction is a small fraction of the field, so its last bits fall off
the end") and it is **wrong**: at seventy-nine nodes the ratio is *smaller* and the difference
reaches the state anyway. The real mechanism is that two doubles sum the same in either order
**unless they cancel**, and a cancelled sum is tiny — so restricted to the rows where the matvec
actually differs, the correction is at most **9.3e-13** of the field at two nodes against a
median **1.2e-4** at seventy-nine, and one ulp of the former cannot survive the addition while
about one in 1/1.2e-4 of the latter does (predicting ~36 of 44,653; 30 observed). So the question
to ask before writing an exact assertion across a ported reduction is **"how long is it?"**, and
it costs no measurement (§23.2). Five corollaries. **`portable.py` was declined for the first
time, on evidence**: it would buy nothing where exactness is already structural and nothing is
*available* where a coarser divergence sits downstream, while changing a shipped model's numbers
and the viewer's output — the rule being that it is for a reduction whose order is the *only*
thing between two implementations (§23.3). **The sister model spells the same quantity
differently**, so "follow the model this one most resembles" would have been wrong: `collision.py`
writes `k ** 2` and `bow.py` writes `k * k` at the structurally identical spot, and those are
different doubles in 86 of 200,000 sample rates (§23.4). **§17.2's constant fold arrived inside
the test written to catch it** — the witness search passed in debug and found *nothing* in
release, so the test went green having asserted nothing; and in the same search, a predicate
tested on the sub-expression rather than the whole expression found a witness whose difference the
following division absorbed (§23.5). **Porting a class silently emptied an existing parity
section**: §16 measured the contact solve by pinning a module-level name, and a Rust model never
looks that name up — so with the flag set those tests compared Rust against Rust and passed, which
is the empty-guard and empty-CI-job shape reached through a third door (§23.6). And **a derive is
only as wide as the list it derives over** — the class swap guard had been one module short for
six batches, in the very half §17.6 claimed to be fixing (§23.7).

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

A **seventeenth**, from `beam` — Phase 4, which was sent to answer a question rather than to add
a model, and which **answers it in the negative**: §4.1's plan to link SuperLU and hold the whole
sparse-LU group to the bit-identity everything else gets **does not work**, and the reason is not
on §4.1's own list of three. Two of that list are non-issues — the fill-reducing column order is
a *closed form in n* for this family (verified at seventeen grid sizes), and equilibration never
runs at all because SciPy calls the factorization routine and not the driver that would scale the
matrix. The third, the pivot threshold, is **real but only above a grid size the batch's first
two fixtures did not reach**: the reference takes the diagonal up to N = 48 and starts swapping
rows at N = 64, which is §16.4's blind fixture arriving in the *measurement* rather than in the
model, and the general form is **parametrise over the grid before concluding anything about a
solver**. What actually decides it is a fourth thing invisible in SciPy's Python: SuperLU is
**supernodal**, and handed its *own* factors a longhand triangular solve still disagrees with
`lu.solve` in ~20% of entries at ~4e-16 — so matching it is a claim about how SciPy was *built*,
which is §22.1 one layer down. The human's call (2026-08-28) is **tolerance-level agreement,
quantified**, and the dependency list stays empty (§24.2). Five corollaries. **The port itself is
exact and the tooling to prove it is a two-line patch**: driving the Python beam through the Rust
factorization makes the two models bit-identical over 2,000 steps at four fixtures, which is the
strings' `shared_solver()` manoeuvre a second time and is the only thing that would catch a
reassociation, because the solver gap is two orders larger (§24.4). **A sixth agreement regime,
and the first set by a boundary condition**: a free-free beam's stiffness has the rigid-body
nullspace `{1, x}`, so along it the beam is a *free particle* and a per-step solver difference is
integrated twice — the rigid part grows like t² (3.7e-17 at one step, 3.3e-9 at 20,000) while the
elastic part random-walks to 1.4e-12 and the energy stays inside 7.2e-12, so a parity bar on any
free-edge model reads the **rigid/elastic split or the energy**, never `max|du|/amp` (§24.5).
**`portable.py` was declined a second time and on the opposite grounds** — §23.3's was "exactness
is already structural", this one is "exactness is not *available* downstream of a coarser
divergence" (§24.6). **An omitted keyword and an explicit `None` were the same argument in every
binding** — PyO3 collapses them, so `boundary=None` silently built the default where Python
raises; no parity file had ever passed `None` to a constructor because nobody thinks to try it,
and the fix's arm order is inverted from the obvious guess (§24.7). And **§7's "four
non-reproducible oracles" were twenty**, of which the ones that mattered return *eigenvectors*
fed to `set_state`: an unpinned ARPACK start vector made the two exactly-degenerate rigid modes
come back **1e-1 apart** run to run, which is an initial condition, not a last digit (§24.9).

And an **eighteenth**, from the guitar outline — Phase 5's first batch, which is about *where a
batch boundary goes* and answers it with a distinction none of the previous seventeen needed:
**a function whose output is a decision is a different porting problem from one whose output is
a number.** Every rule the migration has built — does the reduction reach the next timestep
(§14.2), does anything branch on it (§19.2), does the nonlinearity amplify or contract it
(§20.5) — assumes the answer is a quantity and the verdict is a tolerance. `guitar_mask` is
`|x| < half(y)` at every node and its answer is *which nodes exist*, so a last bit is not a
rounding but a **different plate** — and every detector this repo owns (energy, nullspace,
spectrum, area deficit) passes on a plate with one node too few. So the seven discrete-output
functions ported as a group, ahead of the seven matrices they serve, and the rule is **settle
the arithmetic of the discrete-output functions before porting anything that consumes them**
(§25.2). Five corollaries. **A margin measurement costs one script and settles a spelling**:
over 130 shipped configurations every *real* guitar clears its own rim by ≥1.9e7 ulps of the
half-width, so the mask's exactness is **structural** — §23.2's "ask a cheap question about the
shape of the arithmetic" applied to a comparison instead of a sum — while the **degenerate lens**
(`waist = 0`) does not: four nodes sit **one ulp** out at N = 32 and two sit *exactly* on the rim,
because `sin(π/6)` is ½ and the grid puts a node there, and the viewer's waist sweep starts at
0.0 (§25.3). **§22.3's portable-spelling manoeuvre a fifth time, and the first where the
parenthetical bites**: `math.sin` and Rust's `f64::sin` are the same libm call and `np.sin` is a
third implementation, but taking the portable spelling means a Python loop — 0.2 ms→4.4 ms at
`guitar_scale`'s 20,001 points is nothing, **45.9 ms→467.4 ms** at `guitar_area`'s two million is
half a second on every plate built. So the module now carries **both** spellings, and what makes
that safe is §19.2 again: **give the portable spelling to the consumer that branches and the fast
one to the consumer that averages** (§25.4). **A deliberate duplicate needs a pin at each end**,
not a comment in the middle — §20.2's hazard, and both call sites are asserted to the bit, which
is a claim about the code rather than about a CPU (§25.5). **A rank-zero array is a shape**:
`np.ascontiguousarray` promotes 0-d to `(1,)`, so a reader written three phases ago returned the
wrong shape the moment a function was vectorised over an arbitrary one — §24.7's lesson again,
surfacing through a *wider signature* rather than a new test (§25.7). And **a batch's success
condition must cover the gate's first two steps**: Phase 4 shipped with `cargo fmt --check` red
in seventeen places and `clippy -D warnings` red in four, because §24.11 listed the tests and not
the linters — the fourth time in six batches that CI was red for something visible locally
(§25.8) — and its sibling, found by the whole-suite run at the end of this batch: **a batch that
changes a shared behaviour must re-run the tests that recorded the old one**, because §24.7's
`boundary=None` fix left a Phase-2 test asserting the opposite, and that test stayed green for a
whole batch for the mundane reason that **the installed extension was older than the source** —
so `pip install ./crates/physsynth-py` before believing any parity number, since nothing in the
suite can tell a stale wheel from a fresh one (§25.8a).

And a **nineteenth**, from the plate's matrices — Phase 5's second batch — which is about a
question every previous batch asked as one and which is really two: **when a port meets a library
kernel, whether the *values* agree and whether the *stored order* agrees are separate questions,
with different answers and different remedies.** Here the values agreed everywhere, immediately —
an ascending-`k` accumulation reproduces SciPy's sparse product bit for bit, 0 differing entries
out of 2,629 — and only the order did not. And §18.2's own rule turned out to be too specific:
it found the 1-D biharmonic coming back **descending** and wrote the rule that way, but in two
dimensions the order is **neither ascending nor descending in 600 of 610 rows**. It is whatever
the kernel touched first, so no rule reproduces it and the fix is `portable.canonical` on the
Python side — §18.4's prediction coming true unchanged (§26.2). Five corollaries. **The free
plate needed nothing, and that is what made the batch safe**: `free_plate_stiffness`'s `K` is a
Gram product and SciPy returns those already sorted, measured canonical in *every* row of every
rectangle, disk and guitar — so only the supported plate moved (1.2e-13 of amplitude over 2,000
steps, drift unmoved at 2e-14 against the 1e-10 bar) and the free, orthotropic-free and guitar
plates are bit-identical. Had `K` needed it, §24.5's t² integration along the `{1,x,y}` nullspace
would have made porting a matrix into a re-tolerancing of three models (§26.3). **§15.2's anchor
rule reached the plate family**: `plate.py` spelled `L @ L` inline in *two* classes, and
`VKPlate(nonlinear=False)` must be `array_equal` to `Plate`, so both call sites had to move to
the shared builder in one edit (§26.4). **Ask whether the outer factors share a mantissa before
assuming a reassociation is visible**: the module writes `BᵀWB` right-associated in one place and
left-associated in the other, ~35% of value triples distinguish those, and *none* of this
operator's do — every curvature entry is `1/h²` times an exact power of two, so both associations
commute into the same product. §23.2's cheap-question move applied to a product instead of a sum
(§26.5). **A spelling pin must search, not assert a constant** — the association witness was
first written as three hand-picked numbers, they landed in the agreeing two-thirds, and the test
went *red*; that is §23.5's empty search seen from the other side, and only the searching form
can tell "no difference exists" from "I did not look in the right place" (§26.6). And **the swap
guard caught its own hazard inside the batch that made it**: `dirichlet_interior_d2_1d` is the
module's first private-but-swapped name, the natural alias `_dirichlet_interior_d2_1d_py` is
invisible to a derive that filters leading underscores, and §17.6's finding fired by name on the
first run instead of hiding for a batch (§26.7). One non-finding worth keeping: the Rust
`free_plate_stiffness` is **slower** than SciPy's (3.49 ms vs 2.53 ms at N = 32), which is §11.6
exactly — five compiled SMMP calls with nothing around them to win, and construction-time only
(§26.8).

And a **twentieth**, from the nonlinear plate — Phase 5's third batch, which finishes
`operators2d` — which corrects a rule §26 wrote one batch earlier and is really about *where an
ordering difference enters an expression*: **§26.5's mantissa test is a statement about the
products, and an association can move the sum instead.** The Airy solver assembles `BᵀWB` with
no parentheses, which Python left-associates. Ask §26.5's question and the answer is the
predicted one — every entry of the operator is `{1,2,4}` times one reciprocal and every entry of
the weight is `{1,½,¼}` times one product, so the two outer factors of every term share a
mantissa and **not one term differs** — and the two matrices differ anyway, because the two
bracketings route the contraction through differently-*ordered* intermediates: SciPy hands
`Lc_rᵀ @ Wa` back **descending**, and a sparse product contracts the shared index in the stored
order of its left operand's rows. Over the **22 grids the suite actually builds** (enumerated
from an instrumented run, not sampled) those are different matrices on **2** — 46 entries of
1,889 at one of them — so unlike §16.4 the suite's own fixtures contain the witness. The remedy
is the cheapest this project has found for an ordering problem and the **third kind**: after
"reproduce the values" (§26.2) and "sort the storage" (`portable.py`, §18.2), **re-associate** —
pick the bracketing whose contraction runs over an operand that is already canonical. It changes
no module, adds no call and costs nothing at run time, and the question that finds it is "which
operand's stored order does this contraction run over, and did I choose it or did SciPy?"
(§27.2). Six corollaries. **The Rust side could not have fixed this**: `Csr::from_rows` sorts, so
a descending row is not expressible in the crate at all — §10's decision paying off a third time,
and the reason the edit had to be Python's. **A CSC scatter and a canonical CSR gather are the
same sum**, which is a lemma rather than a measurement (both accumulate each output entry over
increasing column index) and is why the bracket is bit-identical even though `Acell.T` is a CSC
in SciPy and a CSR in Rust — §23.2's cheap-question move applied to a *loop structure* (§27.3).
**One constant is the wrong shape for a Group D bar**: the Airy solve's gap is 3.1e-16 at 4×4 and
**5.2e-10** at the 160×128 the airbox tests build, which is `N⁴` — a biharmonic's condition
number — and since both sides are *backward stable* (asserted, so the growing tolerance cannot be
misread as the port decaying) the forward difference says nothing about either implementation
(§27.4). **§24.4's shared-factorization manoeuvre, third use, is what makes the assembly claim
sayable at all**: put the Python solver on the Rust factorization and the two solves go
bit-identical, so the entire residue is the solver (§27.4). **A seventh agreement regime, and the
first where the trajectory becomes unrelated while every physics bar stays green**: the same
plate at the same amplitude gives answers twelve orders apart — one regime random-walks to
1.1e-13 at 1,000 steps (and is *bit-identical* for 4,000 below `w/e ≈ 0.1`, §23.2's mechanism in
a model), the other is chaotic and e-folds every **~57 steps**, reaching **0.59 of the peak by
step 2,000** — with the energy at 5.1e-14 and the drift at 2.6e-14 on both sides throughout. So
the bar reads the **energy**, never `max|du|/amp`: §24.5's conclusion by a different mechanism,
and the plain form is that **a conserved quantity is not a trajectory comparison**. What
separates the two regimes is **not the fixture and not the amplitude** — both of those were
tried and falsified — it is the **Picard sweep count**, which is the model's own public report
of how hard the nonlinearity is working: two to six is the random walk, eleven or more is the
exponential (§27.5). §19.2's Picard branch was watched and is a *consequence*, not a cause —
first flip at step 1,553, when the deviation is already 2.1e-3. And **§19.7's YAML line
continuation happened a third time, from a third tool, and never reached CI**, because §20.7 had
turned that scar from a paragraph into a test (§27.7).

And a **twenty-first**, from the plate itself — Phase 5's fourth batch, which finishes the
model — which is about **where a batch has to stop** and answers it the opposite way from §25:
**a bit-identity anchor does not only bind two models, it can bind a model to a client that is
not being ported at all.** `airbox.py` deliberately *reassembles* the plate's system matrix
rather than reaching into it, and four of its reduction tests turn that into an exact claim —
switch the room's load off and a loaded plate must reproduce a bare one byte for byte, which is
what pins `airbox`'s own transcription of the theta-scheme. Move the plate's solver to Rust and
that claim breaks for a reason having nothing to do with either side. Twelve of the batch's
fourteen red tests were this. The remedy is one name — `airbox` gets the module-level `splu`
swap every ported module has — and the general question to ask before scoping a batch is **"does
anything outside this module re-derive what it computes, and does a test compare the two
exactly?"** Grepping clients for *private names*, which this migration has done five times and
which §0 predicted for `connection.py`, does not find it: `airbox` reaches nothing private on the
linear plate, it recomputes instead. (`connection.py` itself, §0's named suspect, turned out to
touch no plate privates at all.) Six corollaries. **The anchor made the batch, not a judgement
call**: `VKPlate(nonlinear=False)` must be `array_equal` to `Plate` over 150 steps, so §15.2 put
both classes in one batch — and here the anchor became **structural**, because `VkParams` owns a
`Params` and both classes step through one `step_rhs`, where the original keeps two spellings in
step by docstring. **A read-out's relative error can be meaningless for a physical reason**: a
free plate's stiffness annihilates the constant vector, so an unforced free plate has **no
monopole at all** and `pressure()` returns pure cancellation residue (2e-16 to 1e-13 of the sum
of its terms, against 0.16-0.57 supported) — the first parity bar failed by 1e-1 and the port was
right; drive the plate and the quantity exists again (0.34). Two rules follow: **normalise a sum
by the sum of its absolute terms**, which is §20.6's normaliser question answered for a
reduction, and **ask whether the compared quantity is identically zero before writing a relative
bar**. **§23.6's emptied comparison has a fourth and a fifth door**: a test whose subject is a
*stored column order* cannot run against an implementation where `Csr::from_rows` sorts — the
difference being measured is inexpressible, so it would pass having compared a plate with itself
— and a class that constructs its collaborators by **module-global name** is swapped further than
it looks, so the parity file's Python `VKPlate` was holding a *Rust* Airy solver until it was made
to build `AiryStressSolverPy` explicitly. **§24.5's free-particle error crosses over inside the
run**: on the free plate's three-dimensional nullspace the rigid part grows like t^1.9 and the
elastic part like t^0.8, and they **cross at about step 500** — at step 100 the rigid part is the
smaller and by 20,000 it is 229x the larger, so the same bar measures two different things
depending only on run length. And **§27.5's Picard threshold is reproduced one level up and
refined**: the whole model random-walks at 3.1 and 5.0 mean sweeps (3.4e-12 and 8.0e-13 at 2,000
steps) and decorrelates at 13.5 (**2.3e-3**), with the energy at **3.1e-15** and the drift at
8.6e-13 throughout — so amplitude is still not the discriminator, but it is one of the things
that *moves the sweep count*, which is. Read `n_iters`. One measurement worth keeping: the
nonlinear step is **2.99x** faster in Rust against 1.17-1.32x for a linear one, because a Picard
loop is per-call overhead and nothing else — the largest per-step win the migration has measured
for a field model, and the one the real-time port is about.

And a **twenty-second**, from the geometrically exact string — Phase 5's fifth batch, which
finishes the string family — which is about a **kind of divergence the migration had not met**:
every finding since §14 has been about *which digits* two implementations produce, and this one
changes no digit at all. It is the first Group D model that factors **inside** its solve loop (a
fresh Newton Jacobian per iteration per step, where `beam`, `plate` and the Airy solve all factor
once at construction), so §24's decision to leave the Rust sparse LU in the natural column order
stops being free — and §24 had written its own escape clause, *if a later model makes fill the
constraint, an ordering goes in front of this*. It does, by thirteen times: the unknowns are
stacked **by field** while the nonlinearity couples the three fields at the same **cell**, so
every coupling sits `N-1` columns off the diagonal and the elimination fills the whole envelope
between. At N = 128 the natural order stores **33,895** nonzeros against SuperLU's 2,788 and
costs 2,068 µs against 156; reordered by node — `(u_i, w_i, v_i)` together, a **closed form in
N**, so nothing had to be searched for — it stores 2,645 and costs 58. **No bar in this project
could have caught that**, because the answers stay right and the model merely gets slow: if a
port introduces an algorithmic choice, the assertion has to be about the *work*. The reordering
is free here only because every update-path operator is block diagonal by field, so no reduction
crosses a block and the permutation can live inside the factorization. Six corollaries. **The
first Group D matrix that is not SPD**, which retires `DIAG_PIVOT_THRESH`'s written
justification; what replaces it is measured, and the *proxy matters* — row dominance is set by
`lam_long` alone (8.06 at 0.5, **0.285** at 8.0, where the matrix is not dominant at all) while
the observable that decides it, `is_natural`, holds over **854 Jacobians** with no pivot
anywhere. **§19.2's branch question needs one more word: is it a sum or a max?** The convergence
test here is `max|r| <= tol`, order-independent by construction, where model #9's was a sum
feeding a `brentq` bracket — so what varies is only which side of the bar one step lands on, and
the flip rate is set by **how far the mean iteration count sits from an integer** (0 flips in
20,000 at 1.00, 475 at 1.50), while the *energy* does not move at all, because any root of the
discrete-gradient equation conserves exactly. **`portable.py` was not needed at all** — a first
for this family; every matrix already arrives canonical. **A fixture can be wrong in the
physics rather than in the coverage**: the first native fixtures fixed `fs = 48 kHz`, landing at
`lam_long = 4.6` and 9.2, past the model's own documented cliff — three bars went red and all
three were *correct*, so build a two-speed model's fixture at the **fast** speed. **§19.7's YAML
continuation happened a fourth time from a fourth tool and never reached CI**, because §20.7's
test catches it locally in seconds — four occurrences, zero red runs since it exists. And the
speed: the nonlinear step is **15.5x** faster in Rust, which retires §28's 2.99x as the record by
a wide margin — a Newton step is a dozen tiny NumPy/SciPy calls per iteration and every one of
them is per-call overhead. **Models with an inner iteration are where the real-time port lives.**

And a **twenty-third**, from the room — Phase 5's sixth batch, the first **half of a file** and
the first whose question is "does the seam hold" rather than "are the numbers right": **NumPy's
pairwise reduction has a written-down cutoff, so whether bit-identity is available across a
ported `np.sum` is decidable by counting the terms and costs no measurement at all.** Below
**eight** elements `np.sum` *is* a left-to-right loop (measured 0 disagreements at n = 4, 6 and 7
against 803/2,000 at n = 8, 1,398 at 56, 1,944 at 4,641) — §23.2's "how long is the sum?" applied
to a blocking rule instead of a cancellation, and sharper, because it does not depend on the
values. For this model the answer is a hard no in one direction and a yes in two: the smallest
room the class can build has exactly **eight** pressure nodes, so `acoustic_energy` is never
structurally exact; a 1x1-cell wall face has four, so *that* ledger is bit-identical over 2,000
steps; and a one-node port books a sum of length one, so it is exact too. The bar is therefore
the **field** bit-identical (0.0 over 2,000 steps on five wall types, with cuts, with a source,
from an exact mode) and the **energy books** a tolerance (~1e-16) — affordable for a reason no
earlier refusal had: `dissipated` and `injected` are pure bookkeeping, so §14.2's question
("does the reduction reach the next timestep?") is answered *no* for the first time on a
reduction that is fed back into a running accumulator. Six corollaries. **The seam is fifteen
names wide and the fifteenth is public**: fourteen private ones the ports and wrappers reach
through, six of which they *write*, plus `source_index`, which a test **assigns** — so the
migration has now needed four different searches to find a blocking dependency (private names,
re-derivation, duck-typed types, and written public attributes) and none finds the others; the
general form is **grep for assignment, not only for reference**. **A binding that copies its
buffers has made an algorithmic choice**: the first draft `to_vec()`d the four state arrays every
step, every answer stayed bit-identical, and at one room size (41x33x25) it ran **3.2x slower
than NumPy** — §29.2's cost regression arriving one batch after it was written down, in the
binding rather than in an algorithm; borrowing instead gives 13x at 27 nodes falling to ~1.1-1.5x
above 4,000. **`Bound::clone` is a reference clone**, so `_cut_mask` and `_cut_index` built from
one `PyList::new` were one object and `cut_faces` reported 375 instead of 136 — a wrong *count*,
not a wrong number, caught by an existing test on the first flagged run. **A comment can claim an
order it does not have**: booking the wall flux as a per-step subtotal is `D + (f0 + f1)` where
the reference books per face, `(D + f0) + f1`, and fixing that free association moved the gap
from 2.2e-15 to **1.4e-16**. **§22.3's portable spelling, a sixth time and the first aimed at an
initial condition** — `mode_shape`'s cosines moved to `math.cos` because `set_mode` seeds a *run*
rather than a read-out, and it is the first time the manoeuvre was taken purely on the strength
of §22.1 with no local measurement able to justify it (0 differences in 2,239 values here).
And **§19.7's line continuation happened a fifth time from a fifth tool and never reached CI**.

And a **twenty-fourth**, from the ports — Phase 5's seventh batch, the *second* half of
`airbox.py`'s first tier — which corrects the batch immediately before it and is the first time
this migration has taken back a refusal: **NumPy's pairwise `np.sum` is one fixed algorithm, not
a CPU-dispatched kernel, so it transcribes exactly and §30.2's "bit-identity is unavailable at
eight terms or more" was avoidable.** §30.2's *measurement* stands and is still the cheapest
question in the toolkit — below eight elements `np.sum` **is** a left-to-right loop, whatever the
values. What it got wrong was the conclusion: the blocking above eight (eight accumulators
**seeded** from the first eight, the ragged tail folded into the *combined* result, a split at
`n/2` rounded **down** to a multiple of eight) reproduces `np.sum` with **0 disagreements** over
sixteen lengths from 1 to 40,000, over whole-array sums of 3-D arrays, and over a **strided**
reduction, which is a different code path. The distinction from §22.1 is the whole finding and
the two look alike: a transcendental's last bit is fixed by an **instruction selection** chosen
per CPU at import, where a summation's is fixed by an **order** — and NumPy unrolls by eight
precisely so that a vector unit cannot change it. It is also not §14.2's bargain, because BLAS
`ddot` *fuses* its multiply-add and has no scalar recipe at all. It mattered here and did not
matter in §30 because §14.2's question is answered **yes** three times in this tier — the port's
weights, its resistance and its free-pressure read are all on the update path — so the choice was
transcribe or lose exactness, and **every port size is bit-identical** where the prediction going
in was "a point port exact, a ball port a tolerance". The risky half is stated rather than
buried: "NumPy's blocking is the same on every machine" is asserted in exactly **one** named
test, so a runner that disagrees produces one diagnosis instead of §22.1's eighteen. Six
corollaries. **Take the callee first, and the reason is §13.2 rather than dependency order**: a
wrapper calls `free_pressure()`, solves, then `inject(q)`, so a Rust *caller* over a Python
callee would be a `&mut self` pymethod handing control out twice mid-step — which is also why
this tier could be exact at all, since every one of the file's six `splu` calls is in the wrapper
half. **Ask what a client DOES to the object, not only what it reads**: §30.3's "grep for
assignment" finds four attributes a test replaces wholesale, and then one door further on a test
**replaces the port's methods on the instance** (`port.free_pressure = lambda: ...`) to invert a
sign convention — which no attribute search finds and which a `#[pyclass]` refuses outright
without `dict`; it works because CPython's descriptor rules do not care which language defined
the class, so a getter/setter pair beats the instance dict and a pymethod loses to it. **A
diagonal inside a product is not neutral**: `T.T @ diags(R) @ T` left-associates, so SciPy forms
`(T_ki R_k) T_kj` and that differs from the right-folded spelling in **2,028 of 6,845** entries —
and the fixture that cannot see it, one of six pinned goldens, is blind *provably* rather than by
measurement, because `spreading="nearest"` makes every stored entry in a row of `T` the same
number and `(x d) x` is `x (d x)` identically. **Two SciPy routines in one expression disagree
about explicit zeros** — `coo.tocsr()` keeps a stored `0.0` and `csr_matmat` prunes one — so the
crate needed a second constructor, and no fixture the suite builds contains an explicit zero, so
nothing measured it. **§24.7's inverted arm order bit again despite being written down**: with
`Option<Option<_>>`, `Some(None)` is *omitted* and a bare `None` is the caller's literal, and
getting it backwards silently accepted `spreading=None`. And **the speed curve does not
converge**: §11.6 said the win is per-call overhead and disappears once the compiled kernel
dominates, but the missing clause is *dominates the **same work*** — the port's read is
`O(patch)` in an `O(room)` array, so it is 14.7x at 27 room nodes and still **3.7x** at 41,615,
and a binding that copied its buffers would be **asymptotically** wrong rather than merely slower
(678 µs of copying against a 12.9 µs read at 139,995 nodes, every answer bit-identical
throughout). Two tightenings are parked on purpose: the room's own energy books could now be
exact, and `bore`/`reed` still spell `c0.powf(2.0)` with a literal exponent. (The second was
discharged in batch 9 and was a **shipped** divergence all along; the first is still parked.)

And a **twenty-fifth**, from the wrapper tier — Phase 5's eighth batch, which finishes
`airbox.py` except for its membrane pair — which is about a kind of constraint the migration had
not met: **a tier below can decide what porting the tier above it is even allowed to mean.**
§31 stored a port's `T`, `R` and `load_matrix` as plain Python slots rather than as crate types,
because eight tests replace them wholesale to switch a coupling off, halve it or flip its sign,
and two more replace the port's *methods* on the instance. That was right there and it is
binding here: three more tests replace the **factorization** (`inst._lu_loaded = splu(a)`) and
two call the seam's `rhs()` and `a_bare()` directly, so the port, the seam and the solver are
all objects this tier *holds and calls*, never things it is. A wrapper that cached any of them
would keep loading a plate the test had switched off and every one of those tests would pass
having compared a loaded plate with itself — §23.6's emptied comparison through a **seventh**
door. So this is the first ported module with **no core half at all**: every sparse product, the
assembly, the factorization and `np.dot` go through Python, and what is Rust is the control
flow, the guards, the ledgers and the elementwise arithmetic between those calls — exact by
construction, because elementwise `+ - * /` admits no reassociation. The question to ask before
scoping a batch on top of a ported one is therefore **"what did the tier below promise its
clients it would let them replace?"**, and no name grep finds it: the searches this migration
has needed are now six (private names, re-derivation, duck-typed types, written public
attributes, replaced methods, replaced collaborators) and none finds the others. Five
corollaries. **Everything is bit-identical, the von Kármán trajectory included**, which §27.5
says is impossible for a nonlinear plate over any useful run — and is possible here precisely
because the two implementations are not two discretizations but one Picard loop called through
one factorization, so exactness is a *sharp* test of the transcription rather than a claim about
the dynamics. **§31.11's Group D prediction is wrong and the correction is a rule**: the wrapper
does not *own* a factorization, it calls one (`splu` is a module global read at construction), so
a solver group is a property of **ownership, not of the file a factorization appears in** —
§24.4's shared-factorization manoeuvre gets its fifth use and its first as a *negative control*,
moving nothing because nothing was there to move. **The price is the speed, and it is stated
rather than buried**: 1.06-1.13x on the linear plate and *indistinguishable from neutral* on the
von Kármán one (the machine's own drift, 716-1234 µs on the Python arm alone, exceeds the
effect). That sharpens §29's headline with the clause it lacked — an inner iteration is where
the real-time win lives **only if the iteration's body ports with it**, since a ported *caller*
over unported callees multiplies boundary crossings by the sweep count; the first draft folded
the Picard averages in Rust, paid a copy in and out per sweep, and read 0.91x. **A `#[pyclass]`
getter is the opposite default from `__getattr__`** — it is a data descriptor and beats both the
instance dict and delegation, permanently — so on a class whose whole job is to be a drop-in for
the model it holds, every getter silently takes a name away from `connection.py`'s bridges, and
nothing in `cargo test` or any physics bar can see it. And **§25.8a faked a library bug**: a
stale wheel (built, not installed) made a correct `Option<Option<_>>` fix look exactly like
PyO3 contradicting §31.7's documented arm order — the wrong diagnosis a stale wheel produces
need not look like staleness at all.

And a **twenty-sixth**, from the membrane pair — Phase 5's ninth batch, which **finishes
`airbox.py`** — which is §32.6's own finding pointing the other way and is about what porting a
class silently decides: **a `#[getter]` with no `#[setter]` is still a data descriptor, so
porting a class decides not only which names can be *read* through it but which can be
*written*, and the default is none — where the reference, being Python, allowed every one.**
§32.6 found that a getter *shadows* `__getattr__` permanently and so takes a name away from the
model a wrapper stands in for; this is the same descriptor rule aimed at assignment. The
reference's `RoomLoadedMembrane.n` is a plain integer a caller may advance, and the ported one
written the obvious way refuses. There is exactly one client in the tree and it is a shape no
earlier search would have found: `test_airbox_membrane.py`'s lagged-velocity negative control
does not replace a collaborator (§31, §32) and reaches no private name (§0) — it **bypasses
`step` entirely**, hand-rolling a different scheme out of the wrapper's own parts, so what it
needs is to *drive* the object rather than read it. §30.3's rule (grep for assignment, not only
for reference) is the right search aimed one object too far away: it was run against the room's
*collaborators*, and it has to be run against **the class being ported itself**. The failure
mode here is the good one — an unwritable attribute raises, so a correct port goes red rather
than quietly green — and that is luck, not design: an attribute a client writes and then reads
back would have swallowed the write silently. So the port of a Python class starts from
**every attribute is writable** and justifies each refusal, not the other way round.
Six corollaries. **Three spellings of one name, and the failure is not where you look**: the
wrapper's model attribute names the getter, the label `_require_same_rate` puts in its message,
and the name `__getattr__` refuses to delegate — and reusing the plate tier's macro unchanged
makes `inst.membrane` a *miss*, which falls through to a delegation the membrane itself cannot
answer, so the wrapper loses its own model by way of a mechanism working exactly as designed.
**A default argument is an eighth door onto §23.6's empty comparison**: the membrane seam's
`f_ext` term is the one piece of arithmetic with nothing in the model to be bit-identical *to*
(`Membrane.step()` takes no force at all), and `f_ext=None` is what every airbox test and every
natural fixture passes — so a parity file written the obvious way compares the shared half twice
and passes. **A witness search has to be wide, whole-expression, and read before the accumulator
fills**, which the batch learned by getting all three wrong on the parked `powf` work: a
predicate on `t**2` rather than on `(2 - t^2)/den` finds a witness the subtraction absorbs
(§23.5 arriving *inside* the test written to catch §17.2); an `np.nextafter` walk spans about a
millionth of the range that measured the ~5e-4 disagreement rate and reports "none" from a
neighbourhood that has none (**0 in 200,000 consecutive doubles from 1.41421356** against **47
per 100,000 samples from [1, 2)**); and a ledger compared after 4,000 steps passes against a
deliberately reverted binary because the addition swallows one ulp of one increment (§23.2 in an
accumulator instead of a field). **The parked `powf(2.0)` was a shipped divergence and had been
for six batches**: `pip install` builds release, LLVM folds a literal exponent of 2.0 into a
multiply only in release, and the ambient `c0 = 343.0` is one of the values where `pow` and a
multiply agree — so nothing could see it. Measured one ulp above 343: **9.4e-15 of amplitude
over 200 steps**, and bit-identical after. The cheap general guard is a **Python** pin at a
searched fixture, because it runs against the installed extension and therefore needs no second
build profile. **The speed is §11.6 again and the room dilutes it**: 0.98-1.35x end to end,
1.0-1.6x with the Python room's own 175 µs step subtracted, decaying to a small loss as the
sparse solve grows — the price of §32.2, which bought ten tests that assert something. And
**"what is left in this file" was tracked by tier, so two functions belonging to no tier fell
out of the count** — `_face_axes` and `impedance_from_zeta` had Rust twins with native tests
since §31 and were simply never bound; the swap guard cannot catch that, because a function that
is neither aliased nor swapped is absent from both sides of its comparison.

And a **twenty-seventh**, from the bridges — Phase 5's tenth batch, which **finishes
`physsynth/core/`** — which gives the migration's oldest rule a sign: **a ported caller that
computes nothing is SLOWER than the Python it replaced.** §11.6 said the win is per-call
overhead; §32.4 said an inner iteration wins only if its body ports with it and measured
"indistinguishable from neutral". Here the neutral point is crossed, and the reason is that
**Rust pays that overhead too when it is the one making the call** — a `getattr` + `get_item` +
`extract` from Rust is not cheaper than CPython's `LOAD_ATTR`/`BINARY_SUBSCR`, which have inline
caches. What Rust wins is the *work between* the calls, and one file contains both cases:
`StringBodyBridge.step` is five attribute touches and two float multiplies and comes out
**0.96-0.97x**, while `SympatheticStrings.step` allocates a per-string force array, loops over
`J` and reduces, and comes out **1.51-1.83x**; the two plate bridges sit at exactly 1.00x
because a sparse solve dominates and neither side touches it. So before scoping a batch, ask
how much work sits between the collaborator calls — if the answer is "none", the port buys
fidelity and costs a few per cent, and that is a decision to take deliberately rather than
discover (§34.2). Six corollaries. **All four bridges are one batch and it was not a choice**:
§15.2's anchors bind a one-string `SympatheticStrings` to `StringBodyBridge` by `array_equal`
and a `nonlinear=False` `StringVKPlateBridge` to `StringPlateBridge` by an exact stability
margin — and the parity file re-asserts both in all **four** language combinations, which is the
cell no existing test can reach. **This is the second module with no core half, reached by a
different road from §32.2's**: there the tier below had promised its clients replaceable
matrices; here the class is **polymorphic over its collaborators' types** and always was — eight
kinds of body and plate, no `isinstance` anywhere — so a downcast would pass the entire airbox
and radiation family and fail only on an object nobody in the tree builds, which is why the
parity file builds one. **§33.11's claim that this file reaches no private name is wrong** — it
reaches `string._bc_right` four times and `string._second_diff` twice — and the cheerful half is
the general form: **a dependency written down at the start survives being forgotten in the
middle**, because Phase 0 exposed both names deliberately three phases early and the port worked
on the first build (§34.4). **Both reductions and both dot products are invisible to every
shipped fixture**: §30.2's cutoff makes the sum question decidable by counting, and every body
here has four or five modes and every sympathetic rig two or three strings (measured: 0/5,000
disagreements at M = 4, 5, 7 against **2,022/5,000** at M = 8), while §14.3's `phi = 1.0` blinds
the suite to the fused multiply-add — so the parity file **searches** for its fixtures at M = 12
and J = 8 rather than picking them. **A cached keyword dict is §32.2's hazard inside the batch
that cites it**: reusing the `plate.step(f_ext=)` dict is worth ~3% of the step, but caching the
*array* in it would ignore a caller who replaced `_f_ext` — the dict is cached, the entry is
re-set every step. And **§19.7's line continuation happened a sixth time from a sixth tool and
never reached CI**; the general remedy, rather than "be careful", is to build the backslash as
`chr(92)` so no layer can unescape it.


### §35 — the two parked defects, the red parity step, and the plan for what is left (2026-09-02)

**A negative control whose predicate is a per-CPU kernel is a claim about the runner** (§35.2). The
parity step had been red on `main` for four runs and neither failure was a port:
`test_rust_parity_connection` *required* BLAS `ddot` to disagree with a scalar loop and GitHub's AMD
EPYC kernel agreed; `test_rust_parity_airbox_memb` took a scalar witness and asserted its last bit
survived into a vector it is 3.5e-3 of, which on that SciPy it never did. The first now **reports**
rather than requires (the precedent is §14's `part_company`, which made the same correction a phase
earlier); the second **searches at the whole expression** it will assert on — 927 of 5,000
neighbours survive, the first at step 4 — rather than at a hand-picked witness. A third of the same
class surfaced on Linux only: `np.arcsin` and `math.asin` one ulp apart at the room's mode (3,2,2),
so `mode_frequency` and `_mu_squared` took §22.3's portable spelling. The general form is #14 with
the sign flipped: a test that asserts a difference *exists* is exactly as machine-dependent as one
that asserts equality, and the remedy is the same one — report it, or search for a witness that
survives.

### §36 — the partial detector, and a decision whose margin is zero (2026-09-03)

**A decision's margin and the axis it is decided on are separate questions, and a port is safe when
they run opposite** (§36.3). `spectrum.py` splits along a seam none of #6, #16 or #18 predicts: not
reduction versus step, not solver group, not values versus stored order, but two *axes of one
computation*. The **frequency axis** (`freqs`, `df`, window bounds, `min_separation_hz`, and every
comparison among them) is `+ - * /` alone, which IEEE-754 pins, so it transcribes bit-exactly with
no claim about a CPU. The **magnitude axis** cannot be matched at all — NumPy's own dispatched `cos`
in the Hann window, pocketfft, and `hypot` — and differs in 4,074 of 4,097 bins at 3.2e-16 of the
peak. The port is licensed because the tight decisions and the exact axis coincide, and that was
**measured before writing anything**, over the dependent suite's 384 real `measure_partials_near`
calls and 92,261 candidate peaks: winner-vs-runner-up **>= 1.4e12 ulps**, the local-max guard
**>= 1.6e10** (and it **fires** on 14 of 384 real calls, so it is load-bearing, not defensive),
candidate ordering >= 7.6e7 ulps with **zero** exact ties — against a separation test whose margin
is **exactly zero**.

Six corollaries. **A flag's meaning is a property of what it does *not* swap** (§36.4): `analysis/`
reads `PHYSSYNTH_RS_ANALYSIS`, because under `PHYSSYNTH_RS` alone the acceptance run measures Rust
models with an unmoved Python instrument, and widening the flag would let a shared misreading
cancel — §5 scheduled Phase 7 late for that reason and §35.3 re-planned the order without re-taking
the argument. **A zero-margin comparison does not always clear**: `|i*val - c*val| >= 4.0*val` at
`i-c == 4` *rejects* at 100 kHz with `nfft = 16`, so the claim is agreement and never outcome — and
it survives only because `1.0/(n*(1.0/fs))` was transcribed rather than tidied to `fs/n`, which is a
different number for about **one random sample rate in eight** while agreeing at every rate this
project uses (a hand-picked witness would have blessed the tidy form). **Where a transcendental can
be removed from a decision it beats spelling it portably**: `2**ceil(log2(n))` became integer
arithmetic, checked equal for every length 1..2^20 and every 2^k boundary to 2^31, which retires the
§22.3 claim instead of relocating it. **A loop-order change is a cache change first**: hoisting the
FFT twiddle out of the block loop cut transcendental calls from `(n/2)log2(n)` to `n-1` and made
long transforms **five times slower**, so one size is not a measurement (§36.8). **A new crate
inherits the portability contract's convention and none of its enforcement**, since `deps.rs` is
rooted at its own package by name — hence `physsynth-analysis` carries its own allowlist. And
**§35.7's own instruction went red**: the shards cover `tests/` exactly once, so the parity family
is *inside* them, and a flagged shard makes those files compare Rust against Rust — the exclusion
must also happen **after** the split, or LPT silently produces a different partition that is green
either way. §19.7's escaping bug recurred **twice more** while writing the step, caught by
`tests/test_ci_workflow.py` both times; the block now carries no continuation at all.


### §37 — the closed-form oracles, and a defect the port found rather than made (2026-09-03)

**The denominator of an agreement bar is a modelling choice** (§37.3), and this batch produced three
instances of it in one afternoon. The Rust `J_n` agrees with SciPy to **6.7e-16 absolute** and
**2.1e-12 relative** — the same measurement, with the relative worst case sitting where `J_3` is
`-9.8e-05`, near one of its own zeros, so it is the *denominator* that collapsed rather than the
error that grew. `duffing_displacement` agrees to 1.4e-17 on an amplitude of 0.07 (484 of 501 samples
bit-identical) and reads **2.7e-4 relative** at the sample nearest a zero crossing.
`duffing_frequency_shift` is `ω(A) − ω₀`: 3.5e-15 of the shift, **1.6e-16 of the frequency**. And the
first attempt at the disk-scan margin said 1e-35 because it divided by the scan's *global* maximum,
when the determinant near `λ → 0` is legitimately 1e-187 and perfectly signed — the right
denominator is the **local cancellation** `|m₀₀m₁₁| + |m₀₁m₁₀|`, which puts the true margin at
4.6e-6 against a 1e-15 perturbation, i.e. **~5e9×**. This is #6's amplitude-not-pointwise scar from
Phase 2 batch 4 arriving in a module with no trajectory in it, which is what makes it general.

**Which direction of a shared dependency is real is a grep, not a design question** (§37.2). This
batch looked like a cycle: `modal` needs core's Brent transcription, and `piston_radiation_resistance`
in `core/radiation.py` needs a Bessel `J1`. But the piston helper is **never called inside a model's
`step()`** — two tests, two scripts, three docstrings — so it is an oracle in a core file and the
second edge does not exist. Only Brent was left, and it is reached by a `#[path]` **source include**
rather than a Cargo edge: one copy of the code, no dependency for `deps.rs` to catch, and the
argument written in `lib.rs` because the coupling is invisible in `Cargo.toml`. Taking
`physsynth-core` as a real dependency was refused — it inverts the argument the crate split exists
to make — and copying the file was refused because it duplicates a numerical method whose whole
justification is reproducing SciPy exactly.

Four corollaries. **A native bar that asserts two independent routes to one number agree finds what
a parity test cannot**, because a parity test compares two implementations of the same mistake: a
bar asserting `piston_radiation_resistance`'s two branches meet at their threshold failed, and the
defect was in the *Python* — its `ka < 1e-8` series guard is three decades too small, so just above
it the shipped function is **544% wrong** (hurdles §14, reproduced deliberately, fix costed, no
caller in the band). **Almost everything in this batch is bit-identical and none of it is asserted**
(§37.5) — including every `sin`, `arcsin`, `arccos` and `log2` — because #14 makes that a claim
about the runner; equality is required only where IEEE-754 requires it, plus one physics equality
(a lossless string's decay factor is exactly `1.0`). **A search's *count* is a claim a tolerance
cannot make**: the disk scan asserts the number of roots as well as their values, because a missing
root is the dangerous direction and all three of the function's self-checks catch a spurious one
instead. **#14's early-binding hazard was not live here**: `dispersion.py`'s
`from .modal import ...` copies the *post-swap* names, because a footer runs inside its own module
body — pinned by a subprocess probe rather than argued, along with #17's 0-d-returns-a-scalar
convention and the fact that the analysis flag moves no model.

**And a derived list is only as honest as the reachability it models** (§37.8, ledger #32). The CI
step that runs "the instrument's clients" was widened from the detector's three names to files
importing `physsynth.analysis` — 27 of them — and that was **wrong by 43 files**. `tests/helpers.py`
imports the instrument and measures on its callers' behalf, so sixty-five test files reach it and
twenty-seven say so. Among the silent ones was `test_radiation.py`, which carries the only check
holding the ported Bessel `J₁` against `scipy.special.j1` — i.e. the batch's own safety argument was
outside its own coverage, and a floor would never have said so, because the grep was not broken, it
was asking a narrower question than its name claimed. Run separately it is green (93 passed, and the
composed resistance is bit-identical to the Cephes-built expectation), which is the difference
between a verified argument and an asserted one.
