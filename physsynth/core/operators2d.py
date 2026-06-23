"""Two-dimensional finite-difference operators on a masked Cartesian grid.

The 2D analogue of :mod:`physsynth.core.operators`, for the membrane (HANDOFF §5 model #4). A
domain — a rectangle or a disk — is represented as a boolean **live-node mask** over a Cartesian
grid: a node is *live* (an unknown) iff it lies strictly inside the domain; everything else is held
at ``u = 0`` (Dirichlet rim). :func:`laplacian_from_mask` builds the standard 5-point Laplacian
restricted to the live nodes.

**The load-bearing property is symmetry.** A neighbour that is not live contributes its zero ghost
and simply drops from the row, so the restricted operator is a *principal submatrix* of the
symmetric full-grid Laplacian — hence still symmetric. Energy conservation needs only that symmetry,
not boundary fidelity, which is exactly why a *staircased* circular rim still conserves the discrete
energy to machine precision while only its modal (Bessel) accuracy degrades. See
``docs/dev/membrane-plan.md``.

Headless: NumPy + SciPy (sparse). No I/O, no plotting.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import sparse

__all__ = [
    "rectangle_mask",
    "disk_mask",
    "grid_coords",
    "laplacian_from_mask",
    "biharmonic_from_mask",
    "embed",
    "inner2d",
    "norm2_2d",
]


def grid_coords(
    N: int, half_extent: float
) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    """Square grid of ``N+1`` nodes per axis over ``[-half_extent, half_extent]^2``.

    Returns ``(X, Y, h)`` where ``X, Y`` have shape ``(N+1, N+1)`` (indexed ``[j, i]`` with
    ``x = X[j, i]``, ``y = Y[j, i]``) and ``h = 2*half_extent/N`` is the (square-cell) spacing.
    """
    coords = np.linspace(-half_extent, half_extent, N + 1)
    X, Y = np.meshgrid(coords, coords)  # X varies along axis 1 (i), Y along axis 0 (j)
    h = 2.0 * half_extent / N
    return X, Y, h


def rectangle_mask(Nx: int, Ny: int) -> NDArray[np.bool_]:
    """Live-node mask for a rectangle: every interior node of an ``(Ny+1) x (Nx+1)`` grid.

    The bounding-box edge nodes are the clamped Dirichlet rim (``u = 0``); the interior
    ``(Nx-1) x (Ny-1)`` nodes are the unknowns. The Laplacian built from this mask is exactly the
    tensor-product 5-point operator, whose ``sin(mπx/Lx) sin(nπy/Ly)`` eigenvectors are analytic —
    the clean O(h²) reference that de-risks the harness before the staircase error enters.
    """
    mask = np.zeros((Ny + 1, Nx + 1), dtype=bool)
    mask[1:-1, 1:-1] = True
    return mask


def disk_mask(X: NDArray[np.float64], Y: NDArray[np.float64], radius: float) -> NDArray[np.bool_]:
    """Live-node mask for a disk of ``radius`` centred at the origin on grid ``(X, Y)``.

    A node is live iff ``x² + y² < radius²`` (strict — a node on the rim is treated as boundary).
    The round rim is *staircased* onto the Cartesian grid; this is what taxes the Bessel match to
    ~O(h) while leaving energy conservation exact (the operator stays symmetric).
    """
    return (X * X + Y * Y) < (radius * radius)


def laplacian_from_mask(
    mask: NDArray[np.bool_], h: float
) -> tuple[sparse.csr_matrix, NDArray[np.int64]]:
    """Symmetric 5-point Laplacian on the live nodes of ``mask``.

    Returns ``(L, index_map)``:

    - ``L`` is the ``(nlive x nlive)`` sparse operator with ``-4/h²`` on the diagonal and ``+1/h²``
      for each in-domain neighbour (up/down/left/right). A neighbour that is not live drops (its
      ``u = 0`` ghost), so ``L`` is a principal submatrix of the symmetric full-grid Laplacian —
      **symmetric and negative-definite**. Its eigenvalues are ``-Λ`` with ``Λ > 0``.
    - ``index_map`` has the shape of ``mask``: ``index_map[j, i]`` is the flat unknown index of live
      node ``(j, i)`` and ``-1`` at dead nodes. The flat ordering is C-order over the live positions
      (matching :func:`numpy.nonzero`), so the live coordinates are ``np.nonzero(mask)``.
    """
    mask = np.asarray(mask, dtype=bool)
    ny, nx = mask.shape
    nlive = int(mask.sum())
    index_map = np.full(mask.shape, -1, dtype=np.int64)
    index_map[mask] = np.arange(nlive)

    jj, ii = np.nonzero(mask)  # live coords in C-order -> flat index p = 0 .. nlive-1
    p = np.arange(nlive)
    inv_h2 = 1.0 / (h * h)

    rows = [p]
    cols = [p]
    data = [np.full(nlive, -4.0 * inv_h2)]

    for dj, di in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nj, ni = jj + dj, ii + di
        valid = (nj >= 0) & (nj < ny) & (ni >= 0) & (ni < nx)
        neigh_live = np.zeros(nlive, dtype=bool)
        neigh_live[valid] = mask[nj[valid], ni[valid]]
        rows.append(p[neigh_live])
        cols.append(index_map[nj[neigh_live], ni[neigh_live]])
        data.append(np.full(int(neigh_live.sum()), inv_h2))

    L = sparse.coo_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(nlive, nlive),
    ).tocsr()
    return L, index_map


def biharmonic_from_mask(
    mask: NDArray[np.bool_], h: float
) -> tuple[sparse.csr_matrix, NDArray[np.int64]]:
    """Symmetric 2D biharmonic ``∇⁴ = (∇²)²`` on the live nodes, built as ``B = L @ L``.

    The plate's flexural operator (HANDOFF §5 model #5). ``L`` is the *Dirichlet* (zero-ghost)
    5-point Laplacian from :func:`laplacian_from_mask`, so ``w = L u`` already satisfies ``w = 0`` on
    the rim; applying ``L`` again therefore enforces **both** simply-supported (Navier) conditions —
    ``u = 0`` *and* ``∇²u = 0`` — automatically, with no hand-coded 13-point boundary rows. This is
    the 2D analogue of the 1D ``(δ_xx)²`` biharmonic (see
    :func:`physsynth.core.operators.biharmonic_matrix`).

    Returns ``(B, index_map)`` sharing ``L``'s ``index_map``. Because ``L`` is symmetric, ``B = L²``
    is **symmetric positive-(semi)definite** — its eigenvalues are ``Λ²`` where ``-Λ`` are ``L``'s
    (so ``Λ > 0`` for the Dirichlet interior, hence ``B`` is positive-definite). The ``sin·sin``
    rectangular modes stay *exact* discrete eigenvectors (eigenvalue ``Λ_{mn}²``). Energy
    conservation for the plate needs only this symmetry — exactly as the membrane's conservation
    needed only ``L``'s. See ``docs/dev/plate-plan.md``.
    """
    L, index_map = laplacian_from_mask(mask, h)
    B = (L @ L).tocsr()
    return B, index_map


def embed(
    values: NDArray[np.float64], index_map: NDArray[np.int64]
) -> NDArray[np.float64]:
    """Scatter a flat live-node vector back onto the full 2D grid (zeros at dead nodes).

    The inverse of selecting ``field[mask]``; used to turn the solver's state vector into a 2D
    field for heatmaps/animation.
    """
    field = np.zeros(index_map.shape, dtype=float)
    live = index_map >= 0
    field[live] = values[index_map[live]]
    return field


def inner2d(f: NDArray[np.float64], g: NDArray[np.float64], h: float) -> float:
    """Discrete 2D inner product ``<f, g> = h² Σ f g`` over the live nodes."""
    return float(h * h * np.dot(f, g))


def norm2_2d(f: NDArray[np.float64], h: float) -> float:
    """Squared discrete 2D norm ``||f||² = <f, f> = h² Σ f²`` (>= 0)."""
    return float(h * h * np.dot(f, f))
