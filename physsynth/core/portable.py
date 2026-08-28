"""The two spellings the theta-scheme string family needs so that Python and Rust agree.

Four models share a pair of numerical habits that NumPy and SciPy express one way and a portable
implementation cannot reproduce: :mod:`~physsynth.core.string_stiff`,
:mod:`~physsynth.core.string_damped`, :mod:`~physsynth.core.string_nonlinear` and
:mod:`~physsynth.core.string_geometric`. Both habits are about **evaluation order**, and neither is
about the algebra -- every value below is the same number to within a last bit, and no physics bar
in this repo moves by a measurable amount.

**Why this module exists at all** (``docs/dev/rust-migration-plan.md`` section 18). Those four
models are chained by *bit-identity* reduction anchors: ``sigma1 = 0`` makes the damped string
``array_equal`` to the stiff one, ``EA = 0`` makes the tension-modulated string ``array_equal`` to
the damped one, ``EA = T`` makes the geometrically exact string ``array_equal`` to it as well. The
anchors compare one model class against *another*, so they hold only while every model in the chain
does the same arithmetic in the same order. Port one model to Rust and each anchor it touches turns
into a cross-language claim -- which it can only survive if the *order* is expressible on both
sides. That is what these two functions are for.

The same move has now been made three times, one level down each time: :mod:`.operators` was not a
model and flipping it swung five; :mod:`.banded` was not a model and flipping it swung four; this is
neither a model nor a solver but an *order of evaluation*, and it swings the same four.

**1. The reduction.** ``np.dot`` on two float64 vectors is BLAS ``ddot``, which blocks and may fuse
its multiply-add; the kernel OpenBLAS selects depends on the CPU. Measured 2026-08-27 on this
machine, it disagrees with a plain left-to-right accumulation in ~84 % of random vectors of the
lengths this family uses (16797/20000 at n = 99). Section 14.2 established the general rule --
matching a BLAS reduction would be a claim about a *runner*, not about this code -- so the fix is
to stop asking for one. :func:`dot` is a left-to-right sum, which is what a Rust ``for`` loop does
(LLVM may not reassociate floating-point addition without ``fast-math``, which Rust does not
enable).

Note what this is *not*: ``math.fsum`` would be a third answer again (correctly-rounded, so neither
BLAS's nor a loop's), and ``np.sum`` / ``arr.sum()`` / ``np.add.reduce`` are **pairwise** above a
blocksize of 128, so they are BLAS's problem wearing a different hat. ``np.cumsum`` is the one
NumPy spelling that is sequential by construction, and it stays compiled -- measured equal to a
naive Python loop in 3300/3300 samples at lengths 1 .. 4097, at ~4x ``np.dot``'s cost per call
(2.5 us vs 0.6 us at n = 127), which is nothing next to the ``for`` loop it replaces.

**2. The matrix.** SciPy's sparse product kernel (SMMP) returns its rows as a *stack*, so
``biharmonic_matrix`` -- built as ``D2 @ D2`` -- comes back with column indices in **descending**
order, and ``has_sorted_indices == False``. Subtracting it from ``c**2 * D2`` propagates that order
into ``L``. A CSR matvec accumulates each row in stored-index order, so ``L @ u`` with descending
indices is a *different sum* from ``L @ u`` with ascending ones: measured 2026-08-27, they differ in
**2000 of 2000** random vectors at every grid size tried. And ``L @ u`` is on the update path -- it
builds the right-hand side of every timestep -- so this is not a read-out discrepancy, it is a
different trajectory from step one.

The Rust ``Csr`` is canonical (ascending) by construction, and section 10 wrote down why it should
stay that way: reproducing SciPy's stack order would pin the port to a SciPy internal that a point
release is free to change. That argument only got stronger once something started multiplying by
the matrix in an inner loop -- under the alternative, a SciPy upgrade would silently move every
string trajectory in the project. So :func:`canonical` sorts the *Python* side instead, which is
the spelling both languages can express and the one SciPy itself calls canonical.

**This changes the reference implementation's numbers, unconditionally** -- unlike
:mod:`.banded`, whose swap only changes them under the flag. That is a deliberate choice: gating a
Python model's arithmetic on an environment variable would be worse than a last-bit change made
once, in the open. Measured 2026-08-27 on this machine, every string physics bar is unmoved: the
worst lossless drift over one second at 44.1 kHz is 9.4e-12 for the Python string and 9.6e-12 for
the Rust one, against the 1e-10 acceptance bar. All three anchors stay exact.

**A read-out / update-path split was the original rule, and it did not survive the next batch.**
As first written this module covered only *reported* numbers: the energy reductions an anchor
compares across model classes. :func:`dot` was deliberately kept out of
``string_nonlinear._stretch``, which is on the **update** path, on the stated grounds that "this
one is compared to nothing". Porting model #9 to Rust made that grounds false, and the port is now
what compares it -- so both stretch reductions moved here too, unconditionally, in the same shape
of edit and for the same reason (plan section 19.2).

Two things are worth carrying forward from that. The first is the general rule: **a decision
justified by "nothing downstream depends on this" has to be re-taken the moment something
downstream ports**, and the answer may come out the other way. The second is why it could not be
absorbed as a last bit: the stretch feeds a ``brentq`` residual, so a disagreement in the reduction
changes the root-find's **iteration count** rather than its answer's last digit -- measured at 1,400
of 5,000 steps. The question at each ``np.dot`` is therefore still "does this reduction reach the
next timestep?", but the follow-up is "and does anything *downstream* of it branch on the answer?".

**Scope was four models, and Phase 5's second batch made it six functions as well.** The
prediction in the paragraph this replaces came true unchanged: ``plate.py`` multiplies by a
squared-Laplacian operator every step, so :func:`canonical` is now applied inside
:func:`~physsynth.core.operators2d.biharmonic_from_mask` and
:func:`~physsynth.core.operators2d.orthotropic_biharmonic`, at the assignment that builds the
operator. Three things about that are worth carrying:

* **The values never differed.** Measured over seven grids and two staircased outlines, an
  ascending-``k`` accumulation reproduces SciPy's sparse product bit for bit -- 0 differing entries
  out of 2,629. What differs is only *where each entry is stored*, and that reaches the trajectory
  solely because a CSR matvec sums a row in stored order.

* **The free plate needed nothing.** ``free_plate_stiffness``'s ``K`` is a ``AᵀWB`` Gram product and
  SciPy returns those already sorted -- measured canonical in every row of every rectangle, disk and
  guitar tried. So the sort moved the *supported* plate by ~1e-13 of its amplitude over 2,000 steps
  (drift unmoved at 2e-14 against the 1e-10 bar) and left the free, orthotropic-free and guitar
  plates bit-identical. ``tests/test_plate_modal.py`` holds the pre-change spelling and asserts
  both halves.

* **Two model classes were bound by it.** ``VKPlate`` at ``nonlinear=False`` must be
  ``array_equal`` to ``Plate``, so both had to take the sort in the same edit -- section 15.2's
  rule, arriving in the plate family.

:mod:`.beam`, :mod:`.membrane` and :mod:`.string_ideal` still keep ``np.dot`` and SciPy's index
order, because no anchor binds them to anything and each has a shipped parity measurement built on
their current behaviour.

Headless: NumPy. (:func:`canonical` acts on a SciPy matrix but does not import one.)
No I/O, no plotting.
"""

from __future__ import annotations

from typing import TypeVar

import numpy as np
from numpy.typing import NDArray

__all__ = ["canonical", "dot"]

_M = TypeVar("_M")


def dot(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    """``a . b`` accumulated strictly left to right -- the portable spelling of ``np.dot``.

    Equal to ``sum(a[i] * b[i] for i in range(n))`` evaluated in order, to the bit, which is what
    the Rust port computes. See the module docstring for why ``np.dot``, ``np.sum`` and
    ``math.fsum`` are each a *different* number.

    Both arguments must be 1-D and the same length; an empty pair sums to ``0.0``, as an empty sum
    does. Returns a Python float, matching the ``float(np.dot(...))`` this replaces.
    """
    prod = np.asarray(a, dtype=float) * np.asarray(b, dtype=float)
    if prod.size == 0:
        return 0.0
    return float(np.cumsum(prod)[-1])


def canonical(m: _M) -> _M:
    """Sort ``m``'s column indices in place and return it -- the portable CSR index order.

    A CSR matvec accumulates each row in *stored* order, so an operator built through SciPy's SMMP
    kernel (descending columns) and the same operator in canonical order are different functions in
    floating point. Sorting makes the matrix independent of which SciPy kernel assembled it, which
    is the property the Rust side needs and the one a SciPy upgrade should not be able to take away.

    Returns the same object it was handed (``sort_indices`` is in place); the return is so the call
    reads as an expression at the assignment that builds the operator.
    """
    m.sort_indices()  # type: ignore[attr-defined]
    return m
