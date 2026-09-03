"""Free-free Euler–Bernoulli beam — implicit theta-scheme FDTD (model #5b-pre).

**The implementation is Rust**: ``crates/physsynth-core/src/beam.rs``, bound in
``crates/physsynth-py`` and re-exported here (``docs/dev/rust-migration-plan.md`` §39, unit 8).
That module's header is now the only copy of the physics — the energy-first free-edge operator
``K = h D2ᵀD2`` with its exactly-``{1, x}`` nullspace, the trapezoidal mass whose ``h/2`` end cells
*are* the free-end closure, the theta-time-average that removes the brutal ``kappa² k²/h⁴`` CFL,
and the broad damping caveat that comes with it.

This is the 1D rehearsal of the free-edge Chladni plate and the one free-boundary flexural model in
the project with a genuine closed-form spectrum, ``cos(βL)·cosh(βL) = 1``. The bars that check it
against that spectrum are ``crates/physsynth-core/tests/beam.rs`` — which derives the oracle by
bisection rather than importing it, and solves ``K φ = mu W φ`` with a shifted inverse iteration
built on the crate's own sparse LU — and ``tests/test_beam_modal.py``, which is unchanged and now
measures the Rust beam through SciPy.

What stays here is ``Boundary`` (a `Literal`) and the ``THETA_DEFAULT`` **re-export**. That
constant is defined once, in :mod:`physsynth.core.string_stiff`, and reached through this module by
callers who have a beam rather than a string; the pre-deletion body declared its own copy of the
number, which nothing compared against the original. The binding's own default is pinned against it
in ``tests/test_binding_surface.py``.

Headless: no I/O, no graphics.
"""

from __future__ import annotations

from typing import Literal

from physsynth_rs import FreeBeam

from .string_stiff import THETA_DEFAULT

Boundary = Literal["free"]

__all__ = ["Boundary", "FreeBeam", "THETA_DEFAULT"]
