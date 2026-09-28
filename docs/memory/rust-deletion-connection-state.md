---
name: rust-deletion-connection-state
description: "Deletion 10 (2026-09-05) — the LAST Python model body: the flag can no longer swap a model, a parity file is HARVESTED not dropped (26 of 36 tests, three of them sharper without their twin), and two silent `ruff --fix` hazards live in every shim"
metadata: 
  node_type: memory
  type: project
  originSessionId: b34db39e-1e50-4ae0-8910-b85e217d4fec
  modified: 2026-09-05T20:20:33.399Z
---

`physsynth/core/connection.py` **973 -> 74 lines**, the eleventh of eleven deletion units. Plan
`docs/dev/rust-migration-plan.md` **§49**, findings **#63–#68**. There is no Python model body left
in the project — **23,396 lines gone across ten batches, not one physics bar retired.**

## The finish line is that the flag stopped mattering

(**Superseded 2026-09-28** by [[retirement-phase-a-state]]: the flag and those three swaps are gone.)
`PHYSSYNTH_RS` still existed and still swapped `operators`, `exciter` and `banded` — but **none of
those is a resonator**. It now chooses between two spellings of an operator or between two solvers,
and cannot change which *model* a run exercises. Default and flagged runs differ by three parity
files and 275 tests, and both are Rust physics end to end.

## §39.3's "permanent blocker" conflated two questions

It said `connection` can never be deleted because a `physsynth-core` test cannot reach a class that
lives only in the binding crate. The first half is true (the bridges are polymorphic over their
collaborators' *Python* types, so no downcast and no native bar, ever). It is **not a reason the
body cannot go** — the airbox wrapper tier was under the identical blocker and §48 deleted it. What
a body-deletion retires is the *diagnostic*, not a physics claim.

## A parity file is HARVESTED, not dropped — and killing the twin can make a test STRONGER

744 lines, 36 collected tests, and **26 of them moved** to `tests/test_binding_surface.py`. Only the
four bit-identical trajectories and the two cross-language anchors died — and both anchors were
already asserted *within* Rust by `test_sympathetic.py` and `test_airbox_vk.py` on the default path.

Three got **sharper** by being re-aimed at what the transcription was actually copying:

| was | is now |
|---|---|
| `py.beta_b == rs.beta_b` at a searched 12-mode witness | `rs.beta_b == rs.k**2 * float(np.sum(terms))`, and `!=` the left-to-right loop |
| coupler's body vs a Python twin's at J=8 | two **bare** bodies driven alongside it with the two spellings; one must track `q`, the other must not |
| trajectories agree after 800 steps | `numpy.dot` monkeypatched and the calls **counted** |

The old `beta_b` test would have passed had the Python been wrong about NumPy's pairwise blocking.
The `np.dot` one is the clearest: §14.2's decision was *not to transcribe a reduction*, and the
binding looks `dot` up on the numpy module at call time — so **count the calls**, and the counts are
the scheme: `step` -> **0** (it takes the body's own `bridge_displacement`), `energy` -> **1**,
`__init__` -> **exactly `N + M`** (the exact stability guard builds the coupled leapfrog operator
column by column; 52 at N=48, M=4).

Refusal messages were **frozen** (11 of them) on `analysis_frozen_values.py`'s precedent — but
**a freeze is not the same move as a harvest, and the first draft got it wrong.** The parity test
asserted a *substring of the prose* and then compared the two languages' full strings **on one
machine**; writing the message down as a literal silently promotes every digit to a cross-machine
claim. Two of the eleven interpolate `np.linalg.eigvals` (LAPACK `dgeev`, 52x52 dense) and
`spsolve` + `splu(...).solve(...)`, whose ordering is a claim about how SciPy was *built*. Those two
carry a `{n}` placeholder: prose frozen exactly, number required only to be there and finite. **Ask
which characters the old test actually compared, and on how many machines** (#68). There is no local
repro on Windows for this class of failure — a green here says nothing about the CI runner.

**Ask of every test in a dying parity file: what is its referent outside the twin?** A comparison
with one survives; usually only the trajectories really die.

## Two silent `ruff --fix` hazards in EVERY shim

1. **`__all__` is load-bearing.** A shim's re-exports are unused imports (F401), so without
   `__all__` marking them used, `ruff check --fix` deletes the module's whole public surface.
   `airbox.py` had one; `connection.py` never did and had to grow one.
2. **Residue that is not a Rust object falls through `deleted_bodies`.** That loop asserts
   `module.X is physsynth_rs.X`. `connection.py` keeps `sparse`, `spsolve` and `splu` solely because
   `connection.rs` does `py.import("physsynth.core.connection")` and reads them **by name at call
   time** — they are SciPy objects, so the loop iterates straight past, and `noqa: F401` is all that
   protects them. Deletion surfaces as an `AttributeError` from inside the extension, three layers
   from the cause. The airbox seams were under the same `noqa` and *were* covered, by the accident of
   being Rust classes. Cure: **presence and reachability are two claims** — assert identity against
   `scipy`, *and* monkeypatch both solver names and require the guard to have called each once.

## A guard at zero: delete, or re-derive over a WIDER population

`expected_classes` reached zero rows. Emptying gives `set() == set()` plus two loops over an empty
dict — #52's mechanism again. But the **derive** was worth more at zero than at four, so it was
widened rather than removed, and the claim became *no module in `physsynth.core` chooses between two
implementations of a class any more*.

**Widening the tuple BY HAND does not retire the hole — it restates it at a larger number.** The
first draft listed twenty-one modules; a twenty-fourth added tomorrow sits outside it exactly as
`collision` did. The population is now read off the package with
`pkgutil.iter_modules(physsynth.core.__path__)` — 23 modules, `engine` and `portable` included,
neither ever in the tuple — with four **named positive controls** rather than a floor proving the
scan fired (#61's cure, not the thing #61 warns about). So the question at zero is not "delete or
empty" but **"is there a wider population this was always a special case of, and can I compute
it?"**

## Measured

2,399 -> **2,388** unflagged (1 skipped); 2,089 -> **2,114** flagged; `test_binding_surface.py`
30 -> **56**. Reconciles exactly, measured in a worktree at the parent commit: −36 (the parity
file) −1 (`test_xdist_groups` parametrizes per file) +26 (the harvest).

See [[rust-deletion-phase-state]], [[rust-deletion-split-guards]], [[rust-phase5-connection-state]],
[[rust-migration-state]], [[viewer-stays-python]].
