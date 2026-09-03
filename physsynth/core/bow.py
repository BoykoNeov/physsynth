"""Bowed string — the first **continuous nonlinear exciter**, and the one that closes that leg.

**The implementation is Rust**: ``crates/physsynth-core/src/bow.rs``, bound in
``crates/physsynth-py`` and re-exported here (``docs/dev/rust-migration-plan.md`` §39, unit 1).
That module's header is now the only copy of the physics — the smooth friction curve, the
safeguarded Newton iteration seeded from the previous step's relative velocity (continuation through
the multivalued Helmholtz regime), and the two facts the model is easiest to get wrong on: its money
test is an energy **balance** rather than conservation, and ``σ₁ > 0`` is *required* for a clean
one-slip Helmholtz motion rather than merely nice to have.

``friction_smooth`` and ``friction_smooth_deriv`` come from Rust rather than being re-derived here:
``tests/test_bow_stability.py`` imports them by name and asserts the curve's oddness, peak and
derivative directly.

Headless: no I/O, no graphics.
"""

from __future__ import annotations

from physsynth_rs import BowedString, friction_smooth, friction_smooth_deriv

from .string_damped import DampedStiffString

__all__ = ["BowedString", "DampedStiffString", "friction_smooth", "friction_smooth_deriv"]
