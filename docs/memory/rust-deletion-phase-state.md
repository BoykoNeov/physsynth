---
name: rust-deletion-phase-state
description: "The deletion phase, FINISHED 2026-09-05: route 1 puts the wheel in `validate`, all 11 units gone (23,396 lines, zero physics bars retired), the binding depends on Python, and a deletion reaches modules it does not contain"
metadata: 
  node_type: memory
  type: project
  originSessionId: 1384bf6b-dd8f-4158-a97a-7ca431d94d28
  modified: 2026-09-03T15:52:43.432Z
---

The Rust port's *translation* was finished on 2026-09-03; this is the phase that follows —
**deleting the Python**. Plan `docs/dev/rust-migration-plan.md` **§39–§49**, findings **#35–#68**.

## The decisions that unblocked it (all the human's, 2026-09-03)

1. Route **1** of §39.6: **the `validate` CI job installs the wheel** and stops being the pure-Python
   baseline. It could not stay one — a deleted model's module is an unconditional
   `from physsynth_rs import X`, so there is no pure-Python path to run. **Route 1's blast radius is
   every job that imports the package**: the `checks` job runs `pytest --collect-only` four times and
   had to grow a Rust compile too. Grep the workflow for `pip install`, never for the comments.
2. **Unit 5's `Plate.B`** (§40.5): *add the setter* rather than retire the test.
3. **Units 10/11 (`analysis/`)**: *delete, but freeze the numbers first* — **DONE**, see
   [[analysis-freeze-state]]. `PHYSSYNTH_RS_ANALYSIS` is now read by nothing, so **there is one
   flag**, and `CLAUDE.md`'s two-flag rule is history-with-an-explanation rather than instruction.

## Eleven deletion units, and FOUR kinds of blocker

The unit is a connected component of the reference-alias graph — a module's Python body cannot go
alone if another file names its `<Name>Py` (or, in `analysis/`, its `<name>_py`).

**Done — ALL ELEVEN (23 modules):** 7 `string_ideal` · 4 `mallet`+`membrane` · 3 `bore`+`reed` ·
2 `body`+`radiation` · 1 `bow`+`collision`+the four θ-scheme strings · 5 `operators2d`+`plate` ·
**10 + 11, the whole of `analysis/`** (see [[analysis-freeze-state]]) · **8 `beam`** (see
[[rust-deletion-beam-state]]).

**Left — NONE. All eleven are done (2026-09-05).** Unit **6** `airbox` went in halves, §47–§48
(its "zero native bars" was a mis-measurement — see [[rust-airbox-native-bars]]); unit **9**
`connection` went at §49, see [[rust-deletion-connection-state]]. §39.3 had called `connection`'s
blocker *permanent* and that conflated two questions: it can never have a **native** bar (it lives
only in the binding crate), and that is not a reason its **body** cannot be deleted.

## TWO dependencies that no name grep finds

- **The binding depends on PYTHON.** `grep -rn 'import("physsynth' crates/physsynth-py/src/` —
  four hits, running **Rust → Python**. `string_geometric.py` must keep `GeometricState`,
  `plate.py` must keep `GrainSpec` (the Rust helper *constructs* it), and `airbox.py` /
  `connection.py` have their whole namespaces read at call time.
- **A client that RE-DERIVES the model's arithmetic** (§43.4, finding #44). `airbox.py` reassembles
  the plate's system matrix and factors it itself, and four reduction anchors are `array_equal`, so
  unit 5 broke **15 tests in three modules outside the unit** at ~1e-16. No import, no attribute
  access, no shared name. The tell is an `array_equal` in a test on a module you are *not* deleting,
  and the remedy was a **solver** (one `splu` rebinding), not a name.

## Three things survive every deletion, and one thing does not follow

Kept: **types with no runtime implementation**, **measured constants with their docstrings**
(`PISTON_SERIES_CUTOFF_KA` has no Python reader and stays), and **re-exports of names defined
elsewhere but reached through this module**. "No Rust twin" is **not** a reason to delete
(`string_coefficients_from_material` has none and stays); "no caller at all" is.

**Four module groups now have no core half**: `airbox.py`'s wrapper tier, `operators2d.py`, and
all six of `analysis/`. The binding returns matrices as CSR *triplets* (a headless core cannot know
SciPy) and the analysis wrappers coerce arguments, so the delegating Python wrappers stay — lifted
**verbatim** out of the old `if _USE_RUST:` block, never retyped.

## What a body-deletion actually retires — and it is not a physics bar

Zero physics bars across all ten deletion batches. Sort a parity file per *test*: **untouched** (the comparand
is SciPy or the binding), **retired** (both sides were the implementations), **kept with the
scaffolding removed**. `test_rust_parity_ops2d.py` kept 159 of 699 that way;
`test_rust_parity_plate.py` died entirely, four claims moving to `tests/test_binding_surface.py`
and one becoming a native bar.

## The guards need editing on EVERY deletion

`deleted_bodies` **plus the swapped-class derive plus the `_USE_RUST` reader tuple** (three
separate lists, all in `tests/test_stability.py` — the third is the one that gets missed) ·
`REMAINING_PARITY_FAMILY` in `tests/test_shard_partition.py` · the `rust` job's file list. Retire a
**captured-binding** check only when *both* sides are Rust — or when the capturer no longer exists.

**Three of these reached zero and took three different exits.** `tests/test_rust_parity.py`'s table
and `half_deleted_bodies` were **deleted** (an empty `parametrize` collects as a SKIP; an empty dict
iterates zero times — #52, #60); the swapped-class derive stopped reading a tuple at all and
now reads the **package** (`pkgutil`, 23 modules), so its empty expectation became a live claim —
widening a tuple by hand only moves the hole (#67, [[rust-deletion-connection-state]]).
And a shim's residue that is **not a Rust object** falls straight through `deleted_bodies` — it
needs its own presence-and-reachability guard (#63).

## Two traps that will recur

- **A reflective test over `dir(model)` is a claim about how the model STORES its attributes**, and
  that is what a port changes: a `#[pyclass]`'s `dir()` lists branch-only getters that *raise*, and
  a getter may build a **fresh object per access** so `x.attr is x.attr` is already false (#45).
- **A deletion changes which tests share a process.** It surfaces order- and process-dependent
  defects that are not yours — a fixture seeded `hash(str(x))` under a pointwise relative bar cost
  half an hour here. Discriminator: run the file alone, then run the suite on the stashed tree (#46).

See [[rust-migration-state.md]], [[viewer-stays-python]], [[parity-files-run-unflagged]],
[[test-suite-performance]], [[plate-state]], [[von-karman-plate-state]].
