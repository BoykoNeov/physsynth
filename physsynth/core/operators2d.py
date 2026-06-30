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
    "free_plate_stiffness",
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
    5-point Laplacian from :func:`laplacian_from_mask`, so ``w = L u`` already satisfies ``w = 0``
    on the rim; applying ``L`` again therefore enforces **both** simply-supported (Navier)
    conditions — ``u = 0`` *and* ``∇²u = 0`` — automatically, with no hand-coded 13-point boundary
    rows. This is
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


def _collocated_d2_1d(N: int, h: float) -> sparse.csr_matrix:
    """``(N+1)×(N+1)`` collocated second difference: interior rows ``[1,-2,1]/h²``, **zero ends**.

    Row ``l`` (``l = 1 .. N-1``) is the curvature ``(u[l+1]-2u[l]+u[l-1])/h²``; rows ``0`` and ``N``
    are identically zero (no normal curvature *centered* at a free edge — the 1D beam's choice,
    where curvature was evaluated at interior nodes only). Annihilates linear data exactly.
    """
    inv_h2 = 1.0 / (h * h)
    li = np.arange(1, N)  # interior rows only
    rows = np.repeat(li, 3)
    cols = np.empty(3 * (N - 1), dtype=np.int64)
    cols[0::3], cols[1::3], cols[2::3] = li - 1, li, li + 1
    data = np.tile(np.array([inv_h2, -2.0 * inv_h2, inv_h2]), N - 1)
    return sparse.coo_matrix((data, (rows, cols)), shape=(N + 1, N + 1)).tocsr()


def _forward_d1_1d(N: int, h: float) -> sparse.csr_matrix:
    """``N×(N+1)`` forward first difference: row ``i`` is ``(u[i+1]-u[i])/h`` on cell ``i``.

    Lives on the ``N`` cell midpoints (the dual grid). Annihilates constants; its tensor product
    :func:`free_plate_stiffness` uses for the **cell-centered** twist ``u_xy`` — chosen over the
    collocated centred mixed difference, whose checkerboard ``(-1)^{i+j}`` nullspace would inject
    spurious near-zero modes into the low plate spectrum.
    """
    i = np.arange(N)
    rows = np.repeat(i, 2)
    cols = np.empty(2 * N, dtype=np.int64)
    cols[0::2], cols[1::2] = i, i + 1
    data = np.tile(np.array([-1.0 / h, 1.0 / h]), N)
    return sparse.coo_matrix((data, (rows, cols)), shape=(N, N + 1)).tocsr()


def free_plate_stiffness(
    Nx: int, Ny: int, h: float, nu: float
) -> tuple[sparse.csr_matrix, sparse.csr_matrix, NDArray[np.int64]]:
    """Energy-first free-edge Kirchhoff-plate bending operator on the full ``(Nx+1)×(Ny+1)`` grid.

    Returns ``(K, W, index_map)`` — the building block for the **free** flexural resonator (the
    curved-Chladni plate, ``docs/dev/plate-free-edge-plan.md`` Part 1). The 2D generalisation of
    :func:`physsynth.core.operators.free_beam_stiffness`: **every node is a free unknown** (no
    Dirichlet rim, so the simply-supported ``B = L²`` trick of :func:`biharmonic_from_mask` does not
    apply), and the operator is assembled **from the strain energy** so symmetry, the natural free
    boundary conditions (zero bending moment, zero Kirchhoff shear, corner force) and the rigid-body
    nullspace all fall out *by construction* — never ghost-point elimination on a 13-point stencil.

    **The bilinear form.** The Kirchhoff bending energy is ``U = (rho_s κ²/2)·fᵀ K f`` with ``K``
    representing (``ν`` = Poisson's ratio)

        P(f, g) = ∫∫ [ f_xx g_xx + f_yy g_yy + ν(f_xx g_yy + f_yy g_xx) + 2(1-ν) f_xy g_xy ] dA

    (the standard form; ``ν = 1`` collapses to ``∫(∇²f)(∇²g)``, the simply-supported part, and the
    Gaussian-curvature ``(1-ν)`` term is what re-enters for free edges — it is the part that makes
    the saddle ``xy`` carry energy, the tell for a correct ``ν``).

    **Construction (separable Gram form, reusing the validated 1D beam pieces).** Let ``C2x`` /
    ``C2y`` be the collocated second differences (:func:`_collocated_d2_1d`, zero at the respective
    free edges) applied along x / y, ``Dxy`` the **cell-centered** mixed difference
    ``(u[i+1,j+1]-u[i+1,j]-u[i,j+1]+u[i,j])/h²`` (a tensor product of forward first differences,
    :func:`_forward_d1_1d`), and ``Wa`` the **diagonal area weight** ``= kron(m_y, m_x)`` of the 1D
    trapezoidal masses (``h`` interior, ``h/2`` edge). Then

        K = C2xᵀ Wa C2x + C2yᵀ Wa C2y + ν(C2xᵀ Wa C2y + C2yᵀ Wa C2x) + 2(1-ν)·h²·Dxyᵀ Dxy,
        W = Wa   (interior h², edge h²/2, corner h²/4 — the lumped mass / area quadrature).

    ``Wa`` supplies the edge-½ / corner-¼ weighting *automatically* (the 2D echo of the beam's
    ``h/2`` end cells), and the two bending-diagonal blocks equal ``kron(M_y,S_x) + kron(S_y,M_x)``
    with ``S = free_beam_stiffness`` — i.e. the free plate's bending is the **validated free beam
    operator** per direction, so symmetry, the per-line ``{1, x}`` nullspace and O(h²) are
    inherited, not re-earned.

    **Nullspace (the operator money test).** ``K`` is symmetric positive-semidefinite with nullspace
    **exactly** the rigid-body space ``{1, x, y}`` (3-dimensional): the bending-diagonal blocks kill
    everything linear-per-line (so ``{1, x, y, xy}``), the twist block kills the additively
    separable fields ``{a(x)+b(y)}``, and the intersection (bilinear ∩ separable) is ``{1, x, y}``.
    ``K @ (x·y) ≠ 0`` — supplied solely by the ``2(1-ν)`` twist term and scaling with ``(1-ν)`` — so
    a dropped-``ν`` bug that spuriously kills the saddle is caught immediately. ``K`` being PSD, a
    generalized eigensolve ``K φ = μ W φ`` needs a small **negative** shift; the time-step matrix
    ``A = (1+σk)W + θk²κ²K`` is still SPD because ``W`` is.

    ``index_map`` is the trivial full-grid map (all nodes live): ``index_map[j, i] = j*(Nx+1) + i``,
    matching the C-order (``j`` outer, ``i`` inner) flattening used by :func:`embed` and the
    Kronecker products here. ``nu`` must lie in ``(-1, 1/2)`` (energy positive-definite, physical).
    """
    if Nx < 2 or Ny < 2:
        raise ValueError("Nx, Ny must be >= 2 (need at least one interior node per axis).")
    if not (-1.0 < nu < 0.5):
        raise ValueError(f"nu (Poisson's ratio) must be in (-1, 1/2), got {nu}.")

    ix = sparse.identity(Nx + 1, format="csr")
    iy = sparse.identity(Ny + 1, format="csr")
    c2x_1d = _collocated_d2_1d(Nx, h)
    c2y_1d = _collocated_d2_1d(Ny, h)

    # Second differences applied along one axis (kron(y_factor, x_factor) matches C-order order).
    C2x = sparse.kron(iy, c2x_1d, format="csr")  # u_xx at every node (0 at the x free edges)
    C2y = sparse.kron(c2y_1d, ix, format="csr")  # u_yy at every node (0 at the y free edges)

    # Cell-centered twist u_xy = (forward-x) ⊗ (forward-y) on the Nx·Ny cells.
    Dxy = sparse.kron(_forward_d1_1d(Ny, h), _forward_d1_1d(Nx, h), format="csr")

    # Diagonal area weight Wa = kron(m_y, m_x): interior h², edge h²/2, corner h²/4.
    mx = np.full(Nx + 1, h)
    mx[0] = mx[-1] = 0.5 * h
    my = np.full(Ny + 1, h)
    my[0] = my[-1] = 0.5 * h
    wa = np.kron(my, mx)  # C-order: node (j, i) -> my[j] * mx[i]
    Wa = sparse.diags(wa, format="csr")

    cross = C2x.T @ (Wa @ C2y)
    K = (
        C2x.T @ (Wa @ C2x)
        + C2y.T @ (Wa @ C2y)
        + nu * (cross + cross.T)
        + 2.0 * (1.0 - nu) * (h * h) * (Dxy.T @ Dxy)
    ).tocsr()
    W = Wa

    index_map = np.arange((Nx + 1) * (Ny + 1), dtype=np.int64).reshape(Ny + 1, Nx + 1)
    return K, W, index_map


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
