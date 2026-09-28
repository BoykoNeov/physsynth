---
name: rust-phase5-membrane-state
description: "Rust migration Phase 5 batch 9 — the airbox membrane pair; a getter with no setter silently makes an attribute READ-ONLY; a witness search must be wide, whole-expression and read before the accumulator fills; the parked powf(2.0) was a SHIPPED divergence"
metadata: 
  node_type: memory
  type: project
  originSessionId: 4caced01-3624-411c-9427-f19d6df3040e
  modified: 2026-08-31T19:02:20.735Z
---

Phase 5 batch 9 (2026-08-31) ported `airbox.py`'s **membrane tier** — `_MembraneSurface`,
`RoomLoadedMembrane`, `RoomSuspendedMembrane`, their mixin — plus the two module-level helpers
(`_face_axes`, `impedance_from_zeta`) that belonged to no tier. **`airbox.py` is finished** in the
precise sense: every class it exposes and every function any outside client reaches. Deliberately
NOT swapped, checked not assumed: the module constants (values, not implementations) and
`_require_same_rate`, which has a Rust twin but **no caller outside the file** — it stays live for
the same reason `AirBoxPy` does. Plan §33. Next: `connection.py`, then `analysis/`. See
[[rust-phase5-wrappers-state]].

**The batch added no stepping arithmetic.** `RoomLoadedMembrane.step` is `RoomLoadedPlate.step`
over a different seam, so §32's `Wrap` was reused unchanged and the macro was generalised
(`plate_wrapper!` → `grid_wrapper!`, taking the model's attribute name).

## The headline: a getter with no setter takes a WRITE away

§32.6 found a `#[pyclass]` getter *shadows* `__getattr__` permanently (reads). This is the same
descriptor rule aimed at assignment: **a `#[getter]` with no `#[setter]` is still a data
descriptor, so its `__set__` raises** — porting a class silently makes every attribute read-only
where the Python reference allowed every write. One client in the tree
(`tests/test_airbox_membrane.py:387`, `inst.n += 1`), and it is a shape no earlier search finds:
it replaces no collaborator and reaches no private name, it **bypasses `step` entirely** and
hand-rolls a lagged-velocity scheme out of the wrapper's own parts. §30.3's "grep for assignment"
was the right search **aimed one object too far away** — run it against *the class being ported*,
not only its collaborators. The failure is loud (raises) by luck, not design; an attribute written
then read back would swallow silently. Rule: **start from "every attribute is writable" and
justify each refusal.** The setter went into the SHARED macro, so the four plate wrappers gained
it too — and nothing in the batch's own run could see that, because **a setter adds no name to
`dir()`** and the derived surface guard passes identically either way. It needed its own test.

## Three spellings of one name

The model attribute names (1) the getter, (2) `_require_same_rate`'s message label, (3) the name
`__getattr__` refuses to delegate. Reuse the plate macro unchanged and `inst.membrane` becomes a
*miss* → delegated to the membrane → which has no `.membrane`. **The wrapper loses its own model
via a delegation working exactly as designed.**

## A default argument is an 8th door onto the empty comparison (§23.6)

`_MembraneSurface.rhs`'s `f_ext` term has **nothing in the model to be bit-identical to**
(`Membrane.step()` takes no force). `f_ext=None` is what every natural fixture passes, so an
obvious parity file compares the shared half twice and passes. Every trajectory test is
parametrized over `forced`.

## Witness searches: three ways the parked `powf` work got it wrong first

1. **Predicate on the sub-expression, not the expression.** `t**2 != t*t` finds a witness the
   enclosing `2.0 - t^2` absorbs. §23.5 arriving *inside* the test written to catch §17.2. The
   right fixture is a stiff reed near `wr k = sqrt(2)`, where the subtraction cancels.
2. **`np.nextafter` walks are the wrong search space.** They span ~1e-6 of the range that measured
   the ~5e-4 disagreement rate, and witnesses cluster — **0 in 200,000 consecutive doubles from
   1.41421356** vs **47 per 100,000 samples from [1, 2)**. Step by a *relative* 1e-9 instead.
3. **Compare while the accumulator is empty.** A ledger read after 4,000 steps passed against a
   deliberately reverted binary — the addition swallows one ulp of one increment (§23.2 in an
   accumulator). Search `p_mouth` for a fixture whose **first** step lands on a witness.

## The parked `powf(2.0)` was SHIPPED, for six batches

`pip install` builds release; LLVM folds a literal `powf(x, 2.0)` into `x * x` **only** in release
(§17.2); CPython's `**` is libm `pow` always. Ambient `c0 = 343.0` is a value where the two agree
(343² exact), so nothing saw it. Measured one ulp above 343: **9.4e-15 of amplitude over 200
steps**; bit-identical after routing through `pyfloat::scalar_pow`. Three sites: `bore.rs:262`,
`reed.rs:229`, `reed.rs:505`. **The cheap general guard is a *Python* pin at a searched fixture** —
it runs against the installed (release) extension, so it needs no second build profile, unlike a
native spelling test.

## Speed, and the file's bookkeeping

0.98–1.35x end to end; **1.0–1.6x** with the Python room's own **175 µs** step subtracted, decaying
to a small loss as the sparse solve grows (§11.6). The suspended tier wins more (two pressure
planes read per step = more interpreter calls to remove, same solve). And: **"what is left in this
file" was tracked by tier, so two functions belonging to no tier fell out of the count** — the swap
guard cannot catch a *missing* alias, only a wrong one.

Verification: 25 cargo binaries; `PHYSSYNTH_RS=1 pytest` over the airbox family = **858 passed**
(unchanged); the new `tests/test_rust_parity_airbox_memb.py` = **83**; all parity files together =
**2,478 passed, 1 skipped**. Everything bit-identical. `pip install --no-cache-dir
--force-reinstall` when a rebuild must be certain — an ordinary `pip install` can serve a cached
wheel ([[rust-phase5-wrappers-state]]'s stale-wheel scar with a second way in).
