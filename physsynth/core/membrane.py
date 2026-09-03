"""2-D transverse membrane — explicit FDTD on a masked Cartesian grid (model #4).

**The implementation is Rust**: ``crates/physsynth-core/src/membrane.rs``, bound in
``crates/physsynth-py`` and re-exported here under the name every caller already imports
(``docs/dev/rust-migration-plan.md`` §39, unit 4). That module's header is now the only copy of the
physics — the staircased Dirichlet boundary, the 2-D CFL ceiling ``λ ≤ 1/√2`` (and the fact that
unlike the 1-D string there is *no* dispersionless λ), and the reason the energy is orthogonal to
the geometry: the mask enters the operator, not the inner product.

What still lives here is the module's **type** and its **re-exports**. ``Domain`` is a `Literal`
that has no runtime existence to port, and the five mask/grid helpers are re-exported from
``operators2d`` because callers reach them as ``membrane.disk_mask`` and have since before the
promotion. Neither is a shim to be cleaned up later; they are the module's surface.

Headless: no I/O, no graphics.
"""

from __future__ import annotations

from typing import Literal

from physsynth_rs import Membrane

from .operators2d import (
    disk_mask,
    embed,
    grid_coords,
    laplacian_from_mask,
    rectangle_mask,
)

Domain = Literal["rectangle", "circle"]

__all__ = [
    "Domain",
    "Membrane",
    "disk_mask",
    "embed",
    "grid_coords",
    "laplacian_from_mask",
    "rectangle_mask",
]
