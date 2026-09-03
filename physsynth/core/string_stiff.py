"""Stiff string — implicit θ-scheme with a biharmonic bending term (model #2).

**The implementation is Rust**: ``crates/physsynth-core/src/string_stiff.rs``, bound in
``crates/physsynth-py`` and re-exported here (``docs/dev/rust-migration-plan.md`` §39, unit 1).
That module's header is now the only copy of the physics — the θ-scheme, the biharmonic assembled
as ``D2²`` rather than a fourth-difference stencil, and the reason the scheme is implicit at all.
This is a *different, still-validated* scheme from the explicit ideal string's, not a refinement of
it.

What stays here is ``Boundary`` (a `Literal`), ``THETA_DEFAULT`` — which the other three
θ-scheme strings **and the beam** import *from this module*; the plate is the one exception and
says why in its own header (it carries the reasoning for the number while the Rust carries the
value) — and the ``biharmonic_matrix`` re-export, which ``tests/test_stability.py``'s
captured-binding guard reaches as ``string_stiff.biharmonic_matrix`` to prove the operator swap
landed before this module was imported.

Headless: no I/O, no graphics.
"""

from __future__ import annotations

from typing import Literal

from physsynth_rs import StiffString

from .operators import biharmonic_matrix

Boundary = Literal["supported"]

# The θ-scheme's default weight. Defined here and imported by `string_damped`, `string_nonlinear`,
# `string_geometric` and `beam` — so this module is the family's one source for it. `plate` is not
# in that list and never was: it declares its own copy, deliberately and with its reasoning.
THETA_DEFAULT = 0.28

__all__ = ["Boundary", "StiffString", "THETA_DEFAULT", "biharmonic_matrix"]
