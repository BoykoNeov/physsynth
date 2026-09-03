"""Two-dimensional finite-difference operators on a masked Cartesian grid.

The 2D analogue of :mod:`physsynth.core.operators`, for the membrane and both plates. A domain — a
rectangle, a disk or a guitar outline — is a boolean **live-node mask** over a Cartesian grid: a
node is *live* (an unknown) iff it lies strictly inside the domain; everything else is held at
``u = 0`` (Dirichlet rim). :func:`laplacian_from_mask` builds the standard 5-point Laplacian
restricted to the live nodes.

**The load-bearing property is symmetry.** A neighbour that is not live contributes its zero ghost
and simply drops from the row, so the restricted operator is a *principal submatrix* of the
symmetric full-grid Laplacian — hence still symmetric. Energy conservation needs only that
symmetry, not boundary fidelity, which is exactly why a *staircased* circular rim still conserves
the discrete energy to machine precision while only its modal (Bessel) accuracy degrades. See
``docs/dev/membrane-plan.md``.

# The Python body is gone — 2026-09-03, plan §43

Every operator here is now assembled in `crates/physsynth-core/src/ops2d.rs`; the module's whole
remaining job is to hand the result back in the shape SciPy's callers expect. That is why this is
the **second module with no core half** (`airbox.py`'s wrapper tier was the first, §31): the
binding returns each matrix as CSR *triplets* — `(data, indices, indptr, shape)` — because a
`scipy.sparse.csr_matrix` cannot be constructed from Rust without importing SciPy into the
extension, so `_csr2d` below is real Python work rather than a re-export.

Two things about the surface, both deliberate:

* **The five 1-D difference primitives stay**, underscore and all, though only two of them
  (`_collocated_d2_1d`, `_clamped_d2_1d`) have a caller left. They are one set — the differences
  the free plate's stiffness is built from — and the two with callers prove the set is test-facing.
  Dropping the other three would leave an arbitrary hole in a family, which is a different thing
  from `collision`'s three private helpers, deleted with unit 1 because nothing anywhere named
  them and they had no Rust twin either.
* **`guitar_mask` and `prune_to_area_carrying` are read by the viewer** (`web/serialize.py`), so
  the outline half of this module is on the seam the viewer's import audit covers (§37.0).

Headless: NumPy + SciPy (sparse). No I/O, no plotting.
"""

from __future__ import annotations

import physsynth_rs as _rs
from physsynth_rs import AiryStressSolver, VonKarmanBracket
from scipy import sparse

__all__ = [
    "rectangle_mask",
    "disk_mask",
    "guitar_mask",
    "guitar_half_width",
    "guitar_scale",
    "guitar_area",
    "live_cells",
    "cells_per_node",
    "prune_to_area_carrying",
    "grid_coords",
    "laplacian_from_mask",
    "biharmonic_from_mask",
    "orthotropic_biharmonic",
    "free_plate_stiffness",
    "free_plate_stiffness_from_mask",
    "VonKarmanBracket",
    "AiryStressSolver",
    "embed",
    "inner2d",
    "norm2_2d",
]

# Each function below is `physsynth_core::ops2d::<the same name>` with a SciPy matrix put back
# around it. The reference documentation — the stencils, the mask convention, the four orthotropic
# constants, why the biharmonic is `L @ L` and the free-plate stiffness is a Gram product, and
# which of these come back in canonical column order — is the doc comment on the Rust item, and
# `crates/physsynth-analysis/tests/` plus `tests/test_rust_parity_ops2d.py` are where each is
# asserted. Deliberately not paraphrased here: a second prose copy of a stencil is a second thing
# that can be wrong.


def _csr2d(triplets: tuple) -> sparse.csr_matrix:
    """Rebuild a ``csr_matrix`` from the binding's ``(data, indices, indptr, shape)``."""
    data, indices, indptr, shape = triplets
    return sparse.csr_matrix((data, indices, indptr), shape=shape)


def grid_coords(N, half_extent):
    return _rs.grid_coords(N, half_extent)


def rectangle_mask(Nx, Ny):
    return _rs.rectangle_mask(Nx, Ny)


def disk_mask(X, Y, radius):
    return _rs.disk_mask(X, Y, radius)


def guitar_half_width(t, waist=0.42, asym=0.30):
    return _rs.guitar_half_width(t, waist, asym)


def guitar_scale(width, waist, asym):
    return _rs.guitar_scale(width, waist, asym)


def guitar_mask(X, Y, length, width, waist=0.42, asym=0.30):
    return _rs.guitar_mask(X, Y, length, width, waist, asym)


def guitar_area(length, width, waist=0.42, asym=0.30):
    return _rs.guitar_area(length, width, waist, asym)


def live_cells(mask):
    return _rs.live_cells(mask)


def cells_per_node(mask):
    return _rs.cells_per_node(mask)


def prune_to_area_carrying(mask):
    return _rs.prune_to_area_carrying(mask)


def laplacian_from_mask(mask, h):
    triplets, index_map = _rs.laplacian_from_mask_csr(mask, h)
    return _csr2d(triplets), index_map


def biharmonic_from_mask(mask, h):
    triplets, index_map = _rs.biharmonic_from_mask_csr(mask, h)
    return _csr2d(triplets), index_map


def _dirichlet_interior_d2_1d(n_int, h):
    return _csr2d(_rs.dirichlet_interior_d2_1d_csr(n_int, h))


def orthotropic_biharmonic(
    Nx, Ny, h, grain_x=1.0, grain_cross=1.0, grain_y=1.0
):
    triplets, index_map = _rs.orthotropic_biharmonic_csr(
        Nx, Ny, h, grain_x, grain_cross, grain_y
    )
    return _csr2d(triplets), index_map


def free_plate_stiffness(
    Nx, Ny, h, nu, *, grain_x=1.0, grain_y=1.0,
    grain_coupling=None, grain_torsion=None,
):
    k, w, index_map = _rs.free_plate_stiffness_csr(
        Nx, Ny, h, nu, grain_x, grain_y, grain_coupling, grain_torsion
    )
    return _csr2d(k), _csr2d(w), index_map


def free_plate_stiffness_from_mask(
    mask, h, nu, *, grain_x=1.0, grain_y=1.0,
    grain_coupling=None, grain_torsion=None,
):
    k, w, index_map = _rs.free_plate_stiffness_from_mask_csr(
        mask, h, nu, grain_x, grain_y, grain_coupling, grain_torsion
    )
    return _csr2d(k), _csr2d(w), index_map


def embed(values, index_map):
    return _rs.embed(values, index_map)


def inner2d(f, g, h):
    return _rs.inner2d(f, g, h)


def norm2_2d(f, h):
    return _rs.norm2_2d(f, h)


def _collocated_d2_1d(N, h):
    return _csr2d(_rs.collocated_d2_1d_csr(N, h))


def _forward_d1_1d(N, h):
    return _csr2d(_rs.forward_d1_1d_csr(N, h))


def _centered_d2_1d(N, h):
    return _csr2d(_rs.centered_d2_1d_csr(N, h))


def _clamped_d2_1d(N, h):
    return _csr2d(_rs.clamped_d2_1d_csr(N, h))


def _avg_d1_1d(N):
    return _csr2d(_rs.avg_d1_1d_csr(N))

