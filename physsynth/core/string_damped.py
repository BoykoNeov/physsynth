"""Damped stiff string — frequency-dependent loss on top of the θ-scheme (model #3).

**The implementation is Rust**: ``crates/physsynth-core/src/string_damped.rs``, bound in
``crates/physsynth-py`` and re-exported here (``docs/dev/rust-migration-plan.md`` §39, unit 1).
That module's header is now the only copy of the physics — the ``+2σ₁u_txx`` term that makes the
loss frequency-dependent, the per-mode decay oracle that is this model's money test, and the fact
that ``σ₁ = 0`` reduces it bit-for-bit to model #2.

Only the class and its ``Boundary`` `Literal` survive. The banded-solver re-exports that used to sit
here (``cholesky_banded``, ``cho_solve_banded``) are gone with the body that captured them: they
existed so ``tests/test_stability.py`` could prove the solver swap landed before this module was
imported, and a Rust model does not capture a Python solver.

Headless: no I/O, no graphics.
"""

from __future__ import annotations

from typing import Literal

from physsynth_rs import DampedStiffString

from .string_stiff import THETA_DEFAULT

Boundary = Literal["supported"]

__all__ = ["Boundary", "DampedStiffString", "THETA_DEFAULT"]
