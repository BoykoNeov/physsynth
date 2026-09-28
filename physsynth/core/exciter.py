"""Excitations for the string: initial-condition shapes (pluck) and velocity strikes.

These return arrays sampled on the resonator's grid ``x``; feed them to
:meth:`IdealString.set_state`.

**The implementation is Rust**: ``crates/physsynth-core/src/exciter.rs``, bound in
``crates/physsynth-py`` and re-exported here (``docs/dev/python-retirement-plan.md`` §21, phase A).
The three shapes are:

* ``triangular_pluck(x, L, position, amplitude=1.0)`` — a tent peaked at ``position``
  (``0 < position < L``), zero at both ends. Excites the full harmonic series, so it is the right
  shape for the modal/partial-detection test.
* ``raised_cosine(x, L, center, width, amplitude=1.0)`` — a C^1 hump, zero outside
  ``[center - width, center + width]``, with both end nodes clamped to zero. Band-limited, which
  makes it the cleaner excitation for a grid-convergence study.
* ``raised_cosine_2d(X, Y, center, width, amplitude=1.0)`` — the radial 2-D analogue, zero outside
  radius ``width``.

A bad ``position`` or a non-positive ``width`` raises ``ValueError``, as the Python body did. The
bars are ``crates/physsynth-core/tests/exciter.rs``.

One difference from the deleted body, recorded when the swap landed and still true: the Python
``triangular_pluck`` built its result with ``np.empty_like(x)`` and so returned float32 on a float32
grid, while the Rust one always returns float64. Every grid in this project comes from
``np.linspace``, so nothing sees it.

Headless: no I/O, no graphics.
"""

from __future__ import annotations

from physsynth_rs import raised_cosine, raised_cosine_2d, triangular_pluck

__all__ = ["triangular_pluck", "raised_cosine", "raised_cosine_2d"]
