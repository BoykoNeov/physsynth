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
for free — and the von Kármán step is not affine, so the gong needs a nested solve.

``MalletVKPlate`` (2026-09-06) **is** that nested solve — model #7g, the gong, and the fourth model
here. An outer iteration on the contact force wraps a full von Kármán plate solve per evaluation,
and its frozen tangent is the *linear* plate's drive-point admittance, which makes it degenerate
exactly to ``MalletPlate`` when the coupling is switched off. Two things the batch measured are
worth carrying at the import site, because both contradict what this module used to say:
the nested solve costs **1.8–2.3×** a bare gong step and not the ten to a hundred that was
predicted, and the outer iteration **does** have a closed-form derivative — it is the plate's own
Jacobian-vector product, and ``_drive_point_tangent`` returns it.
Rectangles only, and no ``pressure()``: ``VKPlate`` carries no acceleration field.
See ``docs/dev/mallet-gong-plan.md``.

Headless: no I/O, no graphics.
"""

from __future__ import annotations

from physsynth_rs import MalletMembrane, MalletPlate, MalletVKPlate, MalletWall

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
    "MalletVKPlate",
    "MalletWall",
    "Membrane",
    "contact_potential",
    "contact_force_elastic",
    "contact_stiffness",
    "contact_force_dg",
    "contact_force_total",
    "solve_contact",
]
