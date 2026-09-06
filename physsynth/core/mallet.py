"""Mallet–membrane, mallet–plate and mallet–wall collision — the **contact** models (model #7).

**The implementation is Rust**: ``crates/physsynth-core/src/mallet.rs``, bound in
``crates/physsynth-py`` and re-exported here (``docs/dev/rust-migration-plan.md`` §39, unit 4).
That module's header is now the only copy of the physics — the sign convention for the penetration
``η = u_S − z_H``, the strict-conservation Hamiltonian that makes this model's money test
conservation rather than the bow's energy *balance*, and the discrete-gradient contact force whose
``0/0`` Taylor branch is the model's single biggest trap. ``docs/dev/hammer-collision-plan.md``
still holds the derivation.

The contact primitives below are **not** this module's: they live in ``core.collision``, promoted
there when the distributed-barrier model became their second consumer, and are re-exported so
importers of ``mallet.contact_*`` and ``mallet.solve_contact`` resolve unchanged. That module's
Python body is gone too (unit 1), so the re-exports below reach Rust objects through a shim rather
than through an implementation.

``MalletPlate`` (2026-09-06) is the third model here and the first with **no Python original** —
it was written in Rust rather than transcribed, so that module header is not a second copy of a
derivation, it is the only one. It strikes a linear :class:`~physsynth.core.plate.Plate`
(simply-supported soundboard or free-edge cymbal, on any of the three outlines) and refuses a
``VKPlate``: the plate's implicit step is affine in an external force, which is what lets one
precomputed drive-point column stand in for the local nodal mass the *explicit* membrane hands over
for free — and the von Kármán step is not affine, so the gong needs a nested solve and a batch of
its own.

Headless: no I/O, no graphics.
"""

from __future__ import annotations

from physsynth_rs import MalletMembrane, MalletPlate, MalletWall

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
    "MalletPlate",
    "MalletWall",
    "Membrane",
    "contact_potential",
    "contact_force_elastic",
    "contact_stiffness",
    "contact_force_dg",
    "contact_force_total",
    "solve_contact",
]
