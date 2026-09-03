"""Mallet–membrane and mallet–wall collision — the first **contact** models (model #7).

**The implementation is Rust**: ``crates/physsynth-core/src/mallet.rs``, bound in
``crates/physsynth-py`` and re-exported here (``docs/dev/rust-migration-plan.md`` §39, unit 4).
That module's header is now the only copy of the physics — the sign convention for the penetration
``η = u_S − z_H``, the strict-conservation Hamiltonian that makes this model's money test
conservation rather than the bow's energy *balance*, and the discrete-gradient contact force whose
``0/0`` Taylor branch is the model's single biggest trap. ``docs/dev/hammer-collision-plan.md``
still holds the derivation.

The contact primitives below are **not** this module's: they live in ``core.collision``, promoted
there when the distributed-barrier model became their second consumer, and are re-exported so
importers of ``mallet.contact_*`` and ``mallet.solve_contact`` resolve unchanged. ``collision`` is
unit 1 and still has a Python body; when it is deleted these names follow it there, and nothing
here changes.

Headless: no I/O, no graphics.
"""

from __future__ import annotations

from physsynth_rs import MalletMembrane, MalletWall

from .collision import (
    contact_force_dg,
    contact_force_elastic,
    contact_force_total,
    contact_potential,
    contact_stiffness,
    solve_contact,
)
from .membrane import Membrane

__all__ = [
    "MalletMembrane",
    "MalletWall",
    "Membrane",
    "contact_potential",
    "contact_force_elastic",
    "contact_stiffness",
    "contact_force_dg",
    "contact_force_total",
    "solve_contact",
]
