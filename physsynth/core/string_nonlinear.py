"""Tension-modulated string — Kirchhoff–Carrier, the string family's nonlinearity (model #9).

**The implementation is Rust**: ``crates/physsynth-core/src/string_nonlinear.rs``, bound in
``crates/physsynth-py`` and re-exported here (``docs/dev/rust-migration-plan.md`` §39, unit 1).
That module's header is now the only copy of the physics — the tension that rises with the string's
own stretch, the *exact* Duffing oracle it is validated against, and the finding that single-mode
motion is parametrically unstable above ``ΔT/T₀ ≈ 3`` (real physics, not a scheme artefact). Energy
is a **structural** bar here and spectral purity a **dynamical** one; they fail differently.

The material helper ``string_coefficients_from_material`` lived here until retirement plan §35:
§34 ported it to ``crates/physsynth-core/src/string_nonlinear.rs`` and kept this copy for callers
it named in ``tests/test_geometric_*``, but none of them imported it, so the copy is gone.

Headless: no I/O, no graphics.
"""

from __future__ import annotations

from typing import Literal

from physsynth_rs import TensionModulatedString

from .string_stiff import THETA_DEFAULT

Boundary = Literal["supported"]

TENSION_TOL_DEFAULT = 1e-13
"""Relative tolerance on the tension root-find -- the binding's default, recorded here."""

MAX_BRACKET_EXPANSIONS = 40
"""How far the tension bracket may be widened before the solve gives up."""


__all__ = [
    "Boundary",
    "MAX_BRACKET_EXPANSIONS",
    "TENSION_TOL_DEFAULT",
    "THETA_DEFAULT",
    "TensionModulatedString",
]
