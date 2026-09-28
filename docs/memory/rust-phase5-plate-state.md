---
name: rust-phase5-plate-state
description: "Rust migration Phase 5 batch 4 — the plate, both classes at once; a bit-identity anchor bound the model to an UNPORTED client (airbox), and a free plate's monopole is identically zero"
metadata: 
  node_type: memory
  type: project
  originSessionId: 6b704d3a-a52d-42a1-87db-17429ad85d5b
  modified: 2026-08-28T20:03:53.570Z
---

Phase 5, batch 4 (2026-08-28), plan §28. Ports the whole of `plate.py`: `Plate` (models #5, #5b,
#5o, #5of, #5g — supported/free × rectangle/circle/guitar, both grains) and `VKPlate` (model #6),
plus `grain_ratios_from_material`. **Every plate in the tree is now a Rust model outright.** What is
left of the core is `connection` and `string_geometric`, then `airbox` and `analysis/`.

**The finding — an anchor can bind a model to a client that is NOT being ported.** `airbox.py`
deliberately *reassembles* the plate's system matrix instead of reaching into it, and four of its
reduction tests turn that into an exact claim: switch the room's load off and a loaded plate must
reproduce a bare one byte for byte. That pins `airbox`'s own transcription of the theta-scheme, and
it holds only while both sides factor with the same solver. **12 of the batch's 14 red tests were
this.** The remedy is one name: `airbox` gets the module-level `splu` swap every ported module has.
The scoping question is now **"does anything outside this module RE-DERIVE what it computes, and
does a test compare the two exactly?"** — grepping clients for *private names*
([[rust-phase0-state]]'s §0 prediction, checked five times) does not find it, because re-derivation
is a dependency no name search sees. `connection.py`, §0's named suspect, touches **no** plate
privates at all.

**Both classes had to swap together** — `VKPlate(nonlinear=False)` is `array_equal` to `Plate` over
150 steps, so [[rust-phase3-banded-state]]'s §15.2 made the batch. Here the anchor became
**structural**: `VkParams` OWNS a `Params` and both step through one `step_rhs`, where the Python
original keeps two spellings in step by docstring.

**A relative parity bar can be meaningless for a PHYSICAL reason.** A free plate's stiffness
annihilates the constant vector, so an unforced free plate has **no monopole at all**: `pressure()`
returns pure cancellation residue (2e-16…1e-13 of the sum of its terms, against 0.16–0.57 on a
supported plate). The first bar failed by 1e-1 and the port was right. Drive the plate and the
quantity exists again (0.34). Two rules: **normalise a sum by the sum of its ABSOLUTE terms**
([[rust-phase3-bow-state]]'s normaliser question, answered for a reduction), and **ask whether the
compared quantity is identically zero before writing a relative bar**.

**[[rust-phase3-barrier-state]]'s emptied comparison has a fourth and fifth door.** (4) A test whose
subject is a *stored column order* cannot run against an implementation where `Csr::from_rows`
sorts — the difference is inexpressible, so it passes having compared a plate with itself; pinned to
`PlatePy`. (5) A class that builds its collaborators by **module-global name** is swapped further
than it looks — the Python `VKPlate` was holding a *Rust* Airy solver until the parity file built
`AiryStressSolverPy` explicitly.

**[[rust-phase4-beam-state]]'s free-particle error CROSSES OVER inside the run.** On the free plate's
three-dimensional `{1,x,y}` nullspace the rigid part grows like t^1.9 and the elastic part like
t^0.8; they **cross at about step 500** (rigid is the *smaller* at step 100 and 229× the larger at
20,000). The same bar measures two different things depending only on run length. Read the split or
the energy.

**[[rust-phase5-vk-state]]'s Picard threshold, reproduced one level up and refined.** The whole model
random-walks at 3.1 and 5.0 mean sweeps (3.4e-12 and 8.0e-13 at 2,000 steps) and decorrelates at
13.5 (**2.3e-3**), with the energy at **3.1e-15** and the drift at 8.6e-13 throughout. Amplitude is
still not the discriminator — but it is one of the things that *moves the sweep count*, which is.
Read `n_iters`.

**Exactness:** §24.4's shared-factorization manoeuvre, third use — drive the Python model through
the Rust LU and the two are **bit-identical over 400 steps at all eight fixtures**, forced and
unforced, acceleration cache included. Four spellings of the right-hand side; nothing else can see a
reassociation in them.

**Speed — the nonlinear step is where it is.** `VKPlate` step **2.99×**, linear step 1.17–1.32×,
supported *construction* 0.74× (Rust loses: five compiled SciPy calls with nothing around them),
guitar construction 2.33× (its outline quadrature and prune loop are Python). A Picard loop is
per-call overhead and nothing else — the largest per-step win the migration has measured for a
field model. And the **whole suite, paired back to back on one machine: 1,939 s default vs 998 s
flagged** (4,036 tests either way, **1.94×**) — the first whole-suite figure worth quoting, and only
because it is a *within-session* pair (see [[ci-runner-variance]]; the batch-3 654 s is from another
session and is not comparable).

Three construction-time numbers are not bit-identical and are the only three: `area`,
`outline_area` and `area_deficit` — NumPy pairwise blocking, declined as `guitar_area` declined it.
A rectangle's weights sum to `Lx*Ly` *exactly* in NumPy's order, so its Python deficit is a literal
`0.0` and this one is 1e-15 away; the shipped bar is `abs=1e-14`.

**Scar: a bare `import physsynth_rs` in a parity file turns TWO CI jobs red.** The default gate does
not build the extension (the sharded harness installs only the Python package, and the
shard-reconciliation step runs a plain `pytest --collect-only`), so a module-scope import is a
**collection error**, not a skip — it fails its shard AND makes the reconciliation `grep` find no
count. Fifteen parity files use `pytest.importorskip`; the sixteenth did not, and **ruff's isort
tidied the bare import into the third-party block** where it looked normal. Invisible locally
because the extension is always installed — **but unlike [[numpy-libm-cpu-dispatch]] this one HAS a
one-command local repro: `pip uninstall physsynth-rs`.** *Check whether the local repro exists
before concluding it does not.* Now pinned by
`test_ci_workflow.py::test_every_rust_parity_file_guards_its_extension_import`.
