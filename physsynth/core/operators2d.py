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

import os

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.sparse.linalg import SuperLU, splu

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


def guitar_half_width(
    t: NDArray[np.float64] | float, waist: float = 0.42, asym: float = 0.30
) -> NDArray[np.float64]:
    """Un-normalised half-width profile of a guitar outline at ``t = y/L in [0, 1]``.

        W(t) = sin(pi t) * [1 - waist*cos(4 pi (t - 1/2))] * [1 + asym*(t - 1/2)]

    ``sin(pi t)`` closes the outline at both ends; the ``cos(4 pi ...)`` term places maxima at the
    two bouts and the minimum at the **waist** (``waist`` is its depth, ``0`` = a plain lens);
    ``asym`` widens the lower bout relative to the upper. Because the shape is defined as
    ``|x| < W(y)`` it is simply connected and vertically convex **by construction** — there is no
    outline for which the region can be a ring or two disjoint lobes, which matters because the
    connectivity of the mask is otherwise a live failure mode (see :func:`prune_to_area_carrying`).

    Returned un-normalised: :func:`guitar_mask` rescales so the widest point equals the requested
    width. Exposed because the area of the true outline — the denominator of the plate's area
    deficit — is a quadrature of this profile, and it must be the *same* profile.
    """
    t = np.asarray(t, dtype=float)
    return (
        np.sin(np.pi * t)
        * (1.0 - waist * np.cos(4.0 * np.pi * (t - 0.5)))
        * (1.0 + asym * (t - 0.5))
    )


def guitar_scale(width: float, waist: float, asym: float) -> float:
    """Factor taking :func:`guitar_half_width` to a half-width whose maximum is ``width/2``."""
    peak = float(guitar_half_width(np.linspace(0.0, 1.0, 20001), waist, asym).max())
    return 0.5 * float(width) / peak


def guitar_mask(
    X: NDArray[np.float64],
    Y: NDArray[np.float64],
    length: float,
    width: float,
    waist: float = 0.42,
    asym: float = 0.30,
) -> NDArray[np.bool_]:
    """Live-node mask for a guitar-shaped outline (HANDOFF §12B's non-rectangular plate).

    ``X`` is measured from the centre line, ``Y`` from the neck end, so the region is
    ``|x| < scale * W(y/length)`` with ``W`` from :func:`guitar_half_width` and ``scale`` chosen so
    the widest point spans ``width``. The two end rows (``t = 0``, ``t = 1``) are excluded.

    Like :func:`disk_mask` this is *staircased* onto the Cartesian grid, and the plate pays a
    steeper price for it than the membrane did: see ``docs/dev/guitar-plate-plan.md`` §5.2. Two
    facts worth carrying to any other outline. The error is **O(h), not O(h²)** — the same
    first-order tax the membrane's Bessel match paid, now in a 4th-order operator. And it is
    *largely* a **domain-size** error: the frequency error tracks the mask's area deficit
    mode-independently, so a staircased plate behaves like a well-modelled slightly *smaller*
    plate. That second reading was
    measured on a disk first, where it is worth 6–15x, and it survives here at only 3.7–7.4x and
    without the monotonicity — a uniformly convex boundary cannot distinguish "area deficit" from
    "distance from the rim", and a concave waist beside convex bouts can.

    **The result of this function is not yet a usable mask** — a curved outline produces nodes that
    carry no area at all. Pass it through :func:`prune_to_area_carrying`.
    """
    if length <= 0.0 or width <= 0.0:
        raise ValueError("length and width must be positive.")
    if not (0.0 <= waist < 1.0):
        raise ValueError(f"waist must lie in [0, 1); got {waist}.")
    if abs(asym) >= 2.0:
        raise ValueError(f"|asym| must be < 2 (the profile would go negative); got {asym}.")
    t = np.asarray(Y, dtype=float) / float(length)
    half = guitar_scale(width, waist, asym) * guitar_half_width(t, waist, asym)
    return (t > 0.0) & (t < 1.0) & (np.abs(X) < half)


def guitar_area(length: float, width: float, waist: float = 0.42, asym: float = 0.30) -> float:
    """Area of the *true* guitar outline (fine midpoint quadrature of ``2 W(y)``).

    The denominator of the area deficit that :class:`physsynth.core.plate.Plate` reports. Kept
    beside :func:`guitar_mask` so the two can never drift onto different profiles.
    """
    m = 2_000_000
    t = (np.arange(m) + 0.5) / m
    prof = guitar_scale(width, waist, asym) * guitar_half_width(t, waist, asym)
    return float(2.0 * np.sum(prof) * (float(length) / m))


def live_cells(mask: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """Cells of the dual grid whose **four** corner nodes are all live.

    The quadrature cells of the free-plate energy: the twist ``u_xy`` is evaluated on them, and a
    node's area weight counts them (:func:`free_plate_stiffness_from_mask`).
    """
    m = np.asarray(mask, dtype=bool)
    return m[:-1, :-1] & m[1:, :-1] & m[:-1, 1:] & m[1:, 1:]


def cells_per_node(mask: NDArray[np.bool_]) -> NDArray[np.int64]:
    """Number of live cells (0…4) touching each node. ``4`` interior, ``2`` edge, ``1`` corner."""
    cell = live_cells(mask)
    out = np.zeros(np.shape(mask), dtype=np.int64)
    out[:-1, :-1] += cell
    out[:-1, 1:] += cell
    out[1:, :-1] += cell
    out[1:, 1:] += cell
    return out


def prune_to_area_carrying(mask: NDArray[np.bool_]) -> tuple[NDArray[np.bool_], int]:
    """Drop live nodes that touch **no** live cell, to a fixed point. Returns ``(mask, n_dropped)``.

    **The mask is not the outline.** The outline is a predicate on coordinates; the mask is the set
    of nodes that carry *area*. A curved rim staircases into one-node spikes whose trapezoidal area
    weight is exactly ``0``, which makes the free plate's mass matrix ``W`` **singular** — and
    ``A = (1+sigma k) W + theta k² kappa² K`` then cannot be factored at all. Two such nodes are
    enough; the default guitar outline produces 2–4 of them at every grid tried.

    Dropping a node can orphan its neighbour, so this iterates to a fixed point (one sweep sufficed
    everywhere measured, but the loop is the correct statement, not an optimisation).

    **This is a silent geometry change and callers must price it.** The rule is purely
    topological — it says nothing about *where* the node was — so on a coarse grid with a deep
    waist it can fire in the middle of the plate rather than at a tip, and energy, nullspace and
    spectrum all look
    healthy afterwards. :class:`physsynth.core.plate.Plate` therefore asserts that every dropped
    node lay within one ``h`` of the outline boundary — and exposes the measured depth as
    ``prune_depth_max``, because a bar checked only when it fails is never observed on a passing
    grid. Measured on the shipped default guitar: **0.70–0.75 ``h``** across N = 20…80, so the bar
    is tight rather than decorative.
    """
    m = np.array(mask, dtype=bool, copy=True)
    before = int(m.sum())
    while True:
        keep = m & (cells_per_node(m) > 0)
        if np.array_equal(keep, m):
            return m, before - int(m.sum())
        m = keep


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


def _dirichlet_interior_d2_1d(n_int: int, h: float) -> sparse.csr_matrix:
    """``n_int × n_int`` second difference ``[1,-2,1]/h²`` on the *interior* nodes of a segment.

    The 1D Dirichlet operator whose eigenvectors are exactly ``sin(mπx/L)`` sampled at the interior
    nodes, with eigenvalue ``-(4/h²)sin²(mπh/2L)``. Distinct from :func:`_centered_d2_1d`, which is
    the ``(N+1)×(N+1)`` *full-grid* operator: here the two rim nodes are not unknowns at all, so the
    end rows simply have one neighbour. This is the 1D factor of the tensor-product rectangle —
    :func:`laplacian_from_mask` on a :func:`rectangle_mask` is exactly
    ``kron(I_y, D_x) + kron(D_y, I_x)`` built from it.
    """
    inv_h2 = 1.0 / (h * h)
    main = np.full(n_int, -2.0 * inv_h2)
    off = np.full(max(n_int - 1, 0), inv_h2)
    return sparse.diags([off, main, off], [-1, 0, 1], shape=(n_int, n_int), format="csr")


def orthotropic_biharmonic(
    Nx: int,
    Ny: int,
    h: float,
    grain_x: float = 1.0,
    grain_cross: float = 1.0,
    grain_y: float = 1.0,
) -> tuple[sparse.csr_matrix, NDArray[np.int64]]:
    """Orthotropic (grain-direction) bending operator on a simply-supported rectangle.

    The plate of model #5 made of a material with a **grain**: stiffer along one axis than across
    it. Where :func:`biharmonic_from_mask` builds the isotropic ``∇⁴ = (∇²)²``, this builds

        B = g_x (δ_xx)² + 2 g_h (δ_xx δ_yy) + g_y (δ_yy)²

    the discrete form of ``D_x w_xxxx + 2H w_xxyy + D_y w_yyyy`` divided by a reference rigidity
    ``D_ref``, so the three arguments are **dimensionless ratios** ``g_x = D_x/D_ref`` etc. and the
    caller's ``kappa² = D_ref/rho_s`` multiplies the whole thing (see :class:`.plate.Plate`).

    **The factor of 2 belongs on the cross term here, in the operator, not inside** ``H``. With
    ``H = D_1 + 2 D_xy`` (``D_1 = ν_yx D_x`` the coupling rigidity, ``D_xy = G_xy t³/12`` the
    torsional one), the isotropic material gives ``H = D`` exactly. The two rival packagings that
    the orthotropic literature invites — ``H = D_1`` and ``H = D_1 + D_xy`` — come out at 0.30 and
    0.65 of the correct term for ν = 0.3, and **both produce a perfectly stable, exactly
    energy-conserving, wrong plate**. See ``docs/dev/orthotropic-plate-plan.md`` §1.

    Setting ``g_x = g_h = g_y`` recovers ``g · L @ L``, but **whether it does so bit-for-bit is
    grid-dependent**: the three products are summed in a different order than ``L`` is squared, so a
    grid where ``1/h²`` is exactly representable comes out identical and one where it is not comes
    out 1.7–2.4e-16 apart (measured over seven grids; see ``docs/dev/orthotropic-plate-plan.md``
    §2). Because neither behaviour can be relied on, :class:`.plate.Plate` keeps the isotropic
    default on the untouched ``L @ L`` path rather than routing everything through here — otherwise
    every shipped plate number would move in its last digit on *some* grids and not others.

    Returns ``(B, index_map)`` with the same live-node ordering as
    ``laplacian_from_mask(rectangle_mask(Nx, Ny), h)``.

    **Definiteness is a condition here, not a freebie.** The eigenvalue on mode ``(m,n)`` is
    ``g_x λ_x² + 2 g_h λ_x λ_y + g_y λ_y²`` with ``λ > 0``, which is positive for every mode iff
    ``g_h > -sqrt(g_x g_y)``. The isotropic ``L²`` satisfies this with room to spare; a caller
    dialling ``grain_cross`` does not. :class:`.plate.Plate` rejects the violation at construction —
    this function does not, so it can be used to *build* the indefinite case for a test.
    """
    if Nx < 2 or Ny < 2:
        raise ValueError(f"Nx and Ny must both be >= 2, got ({Nx}, {Ny}).")
    mask = rectangle_mask(Nx, Ny)
    index_map = np.full(mask.shape, -1, dtype=np.int64)
    index_map[mask] = np.arange(int(mask.sum()))

    # Live nodes are walked in C order (y outer, x inner) -- so x is the *inner* tensor factor.
    dxx = sparse.kron(
        sparse.identity(Ny - 1), _dirichlet_interior_d2_1d(Nx - 1, h), format="csr"
    )
    dyy = sparse.kron(
        _dirichlet_interior_d2_1d(Ny - 1, h), sparse.identity(Nx - 1), format="csr"
    )
    B = (
        float(grain_x) * (dxx @ dxx)
        + (2.0 * float(grain_cross)) * (dxx @ dyy)
        + float(grain_y) * (dyy @ dyy)
    ).tocsr()
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


def _centered_d2_1d(N: int, h: float) -> sparse.csr_matrix:
    """``(N+1)×(N+1)`` standard collocated second difference ``[1,-2,1]/h²`` at **every** node.

    Unlike :func:`_collocated_d2_1d` (which zeroes the two end rows for the free beam's natural
    edge), the end rows here keep the one-sided Dirichlet-ghost curvature ``(u[1]-2u[0])/h²`` — the
    ordinary tridiagonal second difference. This is the ``δ_xx`` used by the von Kármán bracket's
    *straight* terms (:class:`VonKarmanBracket`); for a field that vanishes on the rim (the
    simply-supported case) the end-row values are multiplied by a zero test field and drop out of
    the trilinear form, so only the interior curvature matters.
    """
    inv_h2 = 1.0 / (h * h)
    main = np.full(N + 1, -2.0 * inv_h2)
    off = np.full(N, inv_h2)
    return sparse.diags([off, main, off], [-1, 0, 1], format="csr")


def _clamped_d2_1d(N: int, h: float) -> sparse.csr_matrix:
    """``(N+1)×(N+1)`` second difference with the **clamped** ghost mirror at both ends.

    Interior rows are the ordinary ``[1, -2, 1]/h²`` (as :func:`_centered_d2_1d`); the two end rows
    **double** their single off-diagonal (``1 → 2``). The clamped edge condition ``F,n = 0`` gives
    the ghost mirror ``F_{-1} = F_1``, so the boundary-node curvature is
    ``(F_1 - 2F_0 + F_{-1})/h² = (2F_1 - 2F_0)/h²`` — row 0 becomes ``[-2, 2, 0, …]/h²`` (and the
    high edge symmetrically). The matrix itself is **not symmetric** (the end rows are one-sided),
    but the Gram form ``Lc.T @ Wa @ Lc`` (:class:`AiryStressSolver`, with the trapezoidal area
    weight ``Wa``) is, and reproduces the standard clamped-plate biharmonic **exactly**:
    near-boundary diagonal ``7``, interior ``6``, off-diagonals ``-4`` and ``1``. This is the
    contrast with the Dirichlet/Navier ``[1, -2, 1]`` end rows of :func:`_centered_d2_1d`, which
    square to the simply-supported ``B = L²`` operator (``F = 0, ΔF = 0``) instead of the
    clamped ``F = F,n = 0``. Trapezoidal ``Wa`` is load-bearing: with ``Wa = I`` the near-boundary
    diagonal comes out ``9`` (a different, wrong operator).
    """
    inv_h2 = 1.0 / (h * h)
    main = np.full(N + 1, -2.0 * inv_h2)
    off = np.full(N, inv_h2)
    M = sparse.diags([off, main, off], [-1, 0, 1], format="lil")
    M[0, 1] = 2.0 * inv_h2  # ghost mirror F_{-1} = F_1 at the low edge
    M[N, N - 1] = 2.0 * inv_h2  # ... and the high edge
    return M.tocsr()


def _avg_d1_1d(N: int) -> sparse.csr_matrix:
    """``N×(N+1)`` node→cell averaging ``(u[i]+u[i+1])/2`` on cell ``i`` — the transpose-partner of
    :func:`_forward_d1_1d`.

    Its tensor product ``kron(avg_y, avg_x)`` maps a node field to the 0.25-weighted average of the
    four corner nodes of each cell; the *adjoint* of that (``.T``) scatters a cell-centered quantity
    back onto nodes. :class:`VonKarmanBracket` uses it to bring the cell-centered twist product back
    to nodes — the step that makes the trilinear form exactly triple self-adjoint.
    """
    i = np.arange(N)
    rows = np.repeat(i, 2)
    cols = np.empty(2 * N, dtype=np.int64)
    cols[0::2], cols[1::2] = i, i + 1
    data = np.tile(np.array([0.5, 0.5]), N)
    return sparse.coo_matrix((data, (rows, cols)), shape=(N, N + 1)).tocsr()


def free_plate_stiffness(
    Nx: int,
    Ny: int,
    h: float,
    nu: float,
    *,
    grain_x: float = 1.0,
    grain_y: float = 1.0,
    grain_coupling: float | None = None,
    grain_torsion: float | None = None,
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

    **Grain (orthotropy) — model #5of, all four constants.** Wood, not metal, and *four* numbers
    rather than the supported branch's three. The energy above generalises to

        P(f, g) = ∫∫ [ D_x f_xx g_xx + D_y f_yy g_yy + D_1 (f_xx g_yy + f_yy g_xx)
                       + 4 D_xy f_xy g_xy ] dA

    with the dimensionless ratios ``grain_x = D_x/D_ref``, ``grain_y = D_y/D_ref``,
    ``grain_coupling = D_1/D_ref`` and ``grain_torsion = D_xy/D_ref``. **The free branch needs the
    coupling and torsional rigidities separately** — unlike the supported branch, which sees only
    their combination ``H = D_1 + 2 D_xy`` (:func:`orthotropic_biharmonic`). The reason is the
    boundary: merging them takes two integrations by parts whose boundary terms vanish only on a
    pinned rim. On a free rim they do not, and the three free-edge conditions see the two rigidities
    differently — the corner force ``4 D_xy w_xy`` is *pure* torsion. Two materials with the same
    ``H`` and different splits are the **same** supported plate (bit-identically) and **different**
    free plates: measured 6.5x apart in the fundamental. See
    ``docs/dev/orthotropic-free-plate-plan.md``.

    ``grain_coupling`` / ``grain_torsion`` default to the ``nu``-derived isotropic split
    ``(nu, (1-nu)/2)``, which reproduces the isotropic assembly **bit-for-bit on every grid** — the
    four coefficients multiply the same four matrix products in the same order, at ``1.0``, ``1.0``,
    ``nu`` and an exactly-representable halving. (Contrast :func:`orthotropic_biharmonic`, whose
    isotropic collapse is only *grid-dependently* bit-exact, which is why the supported branch keeps
    a separate default path and this one does not.)

    **Semi-definiteness is a condition here, and this function does not enforce it** (so a test can
    build the indefinite case — the same policy as :func:`orthotropic_biharmonic`).
    :class:`.plate.Plate` rejects at construction on

        grain_x > 0,  grain_y > 0,  grain_coupling² < grain_x·grain_y,  grain_torsion > 0

    which is *provably sufficient* on any grid (the energy is a sum of pointwise forms in
    ``(w_xx, w_yy)`` plus a non-negative twist term) and measurably **conservative**: the discrete
    operator survives 4–20% past it on coarse grids, with the margin shrinking under refinement, so
    the pointwise bound is the sharp *continuum* condition. Note ``grain_torsion = 0`` is worse than
    merely inadmissible — it puts the saddle ``xy`` **into** the nullspace (a 4th zero mode), which
    is the cleanest statement of what the torsional rigidity is for.
    """
    if Nx < 2 or Ny < 2:
        raise ValueError("Nx, Ny must be >= 2 (need at least one interior node per axis).")
    # A full bounding box IS the rectangle: every node is a free unknown, every cell is live, and
    # the general routine's "live adjacent cells / 4" area rule evaluates to exactly the trapezoidal
    # kron(m_y, m_x) it replaced. Collapsing the two paths was gated on re-measuring the
    # bit-identity across the #5o survey rather than at one grid -- see
    # free_plate_stiffness_from_mask, and test_guitar_plate's 84-case assertion.
    mask = np.ones((Ny + 1, Nx + 1), dtype=bool)
    return free_plate_stiffness_from_mask(
        mask,
        h,
        nu,
        grain_x=grain_x,
        grain_y=grain_y,
        grain_coupling=grain_coupling,
        grain_torsion=grain_torsion,
    )


def free_plate_stiffness_from_mask(
    mask: NDArray[np.bool_],
    h: float,
    nu: float,
    *,
    grain_x: float = 1.0,
    grain_y: float = 1.0,
    grain_coupling: float | None = None,
    grain_torsion: float | None = None,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix, NDArray[np.int64]]:
    """:func:`free_plate_stiffness` on an **arbitrary outline** — model #5g.

    Same bilinear form, same four constants, same natural free-edge conditions; what changes is the
    set of nodes it is assembled on. Every plate the core built before this one was a rectangle, and
    the rectangle is now just the mask that happens to be all-ones.

    **The generalisation, term by term.** Each rectangle rule turns out to be a special case of a
    rule stated in terms of what is *live*, not in terms of where the bounding box ends:

    - **Curvature.** ``[1,-2,1]/h²`` centred at a node **iff both neighbours along that axis are
      live**, else a zero row. On a rectangle that reproduces :func:`_collocated_d2_1d` exactly,
      including its zeroed end rows — "no normal curvature centred at a free edge" was never a
      statement about index ``0``, it was a statement about a missing neighbour.
    - **Twist.** One row per **live cell** (:func:`live_cells`), the cell-centred
      ``(u[j+1,i+1]-u[j+1,i]-u[j,i+1]+u[j,i])/h²``.
    - **Area weight.** ``h² · (live cells touching the node)/4``. This is not a new convention: it
      *is* the trapezoidal weight, since an interior node touches 4 cells, an edge node 2 and a
      corner 1. Three cases collapse into one expression.

    ``W`` is that area weight; ``K`` is symmetric positive-semidefinite with nullspace **exactly**
    the rigid-body space ``{1, x, y}`` — measured on a staircased guitar outline at every grid tried
    (3 zero modes, rigid-body residual 6.7e-13, saddle ``xy`` not null). Note that this is a
    *necessary* check, not a sufficient one: it looks identical for an assembly with the wrong
    boundary treatment, which is why ``docs/dev/guitar-plate-plan.md`` §5 goes to a derived free-
    **circular**-plate oracle instead.

    ``index_map`` has the shape of ``mask``: the flat unknown index at a live node, ``-1`` elsewhere
    (matching :func:`laplacian_from_mask`'s convention). For an all-ones mask it is the trivial
    full-grid map, so the rectangle callers see no change.

    **The mask must already carry area** — pass it through :func:`prune_to_area_carrying` first, or
    a one-node spike at a tip gives ``W`` a zero on the diagonal and the plate's time-step matrix
    becomes singular.

    **On the bit-identity with the Kronecker assembly this replaced.** It holds *exactly* — a
    measured ``0.0``, not a small residual — across the #5o 7-grid survey x 4 values of ``nu`` x
    three grain splits, 84 cases. One thing had to be right for that, and it was wrong at first: the
    twist coefficient is a **product of two forward differences**, ``(1/h)*(1/h)``, and *not*
    ``1/(h*h)``. Those differ in the last digit whenever ``h`` is not exactly representable, which
    showed up on exactly one grid of the survey (``h = 0.05``) for every ``nu`` and every split, and
    on none of the others. Checking one grid would have reported success.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2-D, got shape {mask.shape}.")
    # `nu` is validated only where it is actually USED, i.e. where it supplies a missing half of the
    # split. An orthotropic plate's implied nu_yx = D_1/D_x legitimately exceeds 1/2 (its bound is
    # nu_xy nu_yx < 1, i.e. grain_coupling² < grain_x grain_y, which Plate enforces instead), so
    # applying the isotropic 3-D admissibility range to a superseded argument would reject a valid
    # material -- and did, for grain_coupling = 0.70 at grain_y = 0.5.
    if grain_coupling is None or grain_torsion is None:
        if not (-1.0 < nu < 0.5):
            raise ValueError(f"nu (Poisson's ratio) must be in (-1, 1/2), got {nu}.")
    g_1 = nu if grain_coupling is None else float(grain_coupling)
    g_xy = 0.5 * (1.0 - nu) if grain_torsion is None else float(grain_torsion)

    ny, nx = mask.shape
    nlive = int(mask.sum())
    if nlive < 1:
        raise ValueError("the mask has no live nodes.")
    index_map = np.full(mask.shape, -1, dtype=np.int64)
    index_map[mask] = np.arange(nlive)
    jj, ii = np.nonzero(mask)  # live coords in C-order -> flat index p
    inv_h2 = 1.0 / (h * h)

    def _curvature(dj: int, di: int) -> sparse.csr_matrix:
        """``[1,-2,1]/h²`` centred at each node with both ``(dj, di)`` neighbours live."""
        jm, im, jp, ip = jj - dj, ii - di, jj + dj, ii + di
        inside = (jm >= 0) & (im >= 0) & (jp < ny) & (ip < nx)
        both = np.zeros(nlive, dtype=bool)
        both[inside] = mask[jm[inside], im[inside]] & mask[jp[inside], ip[inside]]
        p = np.nonzero(both)[0]
        rows = np.repeat(p, 3)
        cols = np.stack(
            [index_map[jm[both], im[both]], index_map[jj[both], ii[both]],
             index_map[jp[both], ip[both]]],
            axis=1,
        ).ravel()
        data = np.tile(np.array([inv_h2, -2.0 * inv_h2, inv_h2]), p.size)
        return sparse.coo_matrix((data, (rows, cols)), shape=(nlive, nlive)).tocsr()

    C2x = _curvature(0, 1)
    C2y = _curvature(1, 0)

    # Cell-centred twist on the live cells. The coefficient is (1/h)*(1/h) and NOT 1/(h*h): this
    # operator is a PRODUCT of two forward first differences, and the two spellings differ in the
    # last digit for an h that is not exactly representable. See the docstring.
    cell = live_cells(mask)
    cj, ci = np.nonzero(cell)
    ncell = cj.size
    d1 = 1.0 / h
    twist = d1 * d1
    Dxy = sparse.coo_matrix(
        (
            np.tile(np.array([twist, -twist, -twist, twist]), ncell),
            (
                np.repeat(np.arange(ncell), 4),
                np.stack(
                    [index_map[cj, ci], index_map[cj, ci + 1],
                     index_map[cj + 1, ci], index_map[cj + 1, ci + 1]],
                    axis=1,
                ).ravel(),
            ),
        ),
        shape=(ncell, nlive),
    ).tocsr()

    # Area weight: h^2 * (live cells touching the node)/4 -- exactly h^2, h^2/2, h^2/4 on a
    # rectangle, and bit-identical to the kron(m_y, m_x) it replaced (0.25 and 0.5 scale exactly).
    Wa = sparse.diags((h * h) * (cells_per_node(mask)[mask] * 0.25), format="csr")

    # One code path for isotropic and orthotropic: at the nu-derived split the four coefficients are
    # 1.0, 1.0, nu and 4*((1-nu)/2) == 2*(1-nu) exactly, so this is byte-identical to the isotropic
    # assembly it replaced -- verified on all 7 grids of the #5o survey. See the docstring.
    cross = C2x.T @ (Wa @ C2y)
    K = (
        float(grain_x) * (C2x.T @ (Wa @ C2x))
        + float(grain_y) * (C2y.T @ (Wa @ C2y))
        + g_1 * (cross + cross.T)
        + (4.0 * g_xy) * (h * h) * (Dxy.T @ Dxy)
    ).tocsr()
    return K, Wa, index_map


class VonKarmanBracket:
    """Discrete von Kármán / Monge–Ampère bracket ``l(a, b)`` on the full ``(Nx+1)×(Ny+1)`` grid.

    The bracket (HANDOFF §5 model #6; ``docs/dev/von-karman-plate-plan.md``)

        L(a, b) = a_xx b_yy + a_yy b_xx − 2 a_xy b_xy

    is the *nonlinear* coupling of the Föppl–von Kármán plate: ``l(w, w)`` sources the Airy stress
    function and ``l(w, F)`` is the membrane restoring force. ``L(a, b) = L(b, a)`` and, with the
    right boundary conditions, its trilinear form ``T(a, b, c) = ⟨L(a, b), c⟩`` is **fully symmetric
    under any permutation of the three arguments** ("triple self-adjointness"). Energy conservation
    of the whole nonlinear scheme rests on the *discrete* bracket reproducing that symmetry to
    machine precision — the operator money test, with no 1D analogue.

    **Construction (empirically pinned, then cross-checked against Bilbao's cell-centered form).**
    The naive collocated bracket is *not* self-adjoint: moving a straight second difference across
    the inner product drags the other factor along and leaves an O(1) remainder. The remainder is
    cancelled **exactly** only if the twist term ``−2 a_xy b_xy`` is discretised on **cell centres**
    (forward-forward mixed differences, the ``Dxy`` already used by :func:`free_plate_stiffness`)
    and its product averaged back to nodes by the adjoint of the corner average. With

        l(a, b) = (δ_xx a)(δ_yy b) + (δ_yy a)(δ_xx b)  −  2 · Aᵀ[ (D_xy a)(D_xy b) ]

    — ``δ_xx, δ_yy`` the collocated second differences (:func:`_centered_d2_1d`), ``D_xy`` the
    cell-centered twist (``kron`` of :func:`_forward_d1_1d`), ``A`` the node→cell corner average
    (``kron`` of :func:`_avg_d1_1d`) — the trilinear form is triple self-adjoint **to machine
    precision for fields that vanish on the rim** (the simply-supported ``w = 0, F = 0`` edge), and
    ``l(a, b) → L(a, b)`` at O(h²). Both properties are verified in ``tests/test_vk_bracket.py``.

    **Domain requirement (not a bug — a contract).** The exact cancellation is a discrete
    summation-by-parts identity with *no leftover boundary term* only when the fields are zero on
    the bounding-box rim. On fields with a non-zero border the trilinear form is asymmetric at O(1);
    that
    is expected and is why the simply-supported case (Dirichlet ``w = F = 0``) is the natural first
    home for the nonlinear plate. Callers pass **full-grid** vectors of length ``(Nx+1)(Ny+1)`` with
    the rim held at zero.

    Parameters
    ----------
    Nx, Ny : int
        Number of segments along x / y (grid is ``(Nx+1)×(Ny+1)`` nodes). Both must be ``>= 2``.
    h : float
        Grid spacing (square cells). Must be positive.
    """

    def __init__(self, Nx: int, Ny: int, h: float) -> None:
        if Nx < 2 or Ny < 2:
            raise ValueError("Nx, Ny must be >= 2 (need at least one interior node per axis).")
        if h <= 0:
            raise ValueError("h (grid spacing) must be positive.")
        self.Nx = int(Nx)
        self.Ny = int(Ny)
        self.h = float(h)
        self.n_nodes = (Nx + 1) * (Ny + 1)

        ix = sparse.identity(Nx + 1, format="csr")
        iy = sparse.identity(Ny + 1, format="csr")
        # Straight collocated second differences applied along one axis (C-order: y outer, x inner).
        self.Sxx = sparse.kron(iy, _centered_d2_1d(Nx, h), format="csr")
        self.Syy = sparse.kron(_centered_d2_1d(Ny, h), ix, format="csr")
        # Cell-centered twist D_xy = (forward-y) ⊗ (forward-x) on the Nx·Ny cells, and the node→cell
        # corner average A = (avg-y) ⊗ (avg-x). A.T scatters the cell twist product back to nodes.
        self.Dxy = sparse.kron(_forward_d1_1d(Ny, h), _forward_d1_1d(Nx, h), format="csr")
        self.Acell = sparse.kron(_avg_d1_1d(Ny), _avg_d1_1d(Nx), format="csr")

    def __call__(
        self, a: NDArray[np.float64], b: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """The nodal field ``l(a, b)`` (full-grid vector, length :attr:`n_nodes`).

        ``a, b`` are full-grid vectors (rim held at zero for the simply-supported plate). Symmetric
        in its two arguments by construction (the two straight terms swap, the twist product is
        symmetric).
        """
        a = np.asarray(a, dtype=float).ravel()
        b = np.asarray(b, dtype=float).ravel()
        straight = (self.Sxx @ a) * (self.Syy @ b) + (self.Syy @ a) * (self.Sxx @ b)
        twist = self.Acell.T @ ((self.Dxy @ a) * (self.Dxy @ b))
        return straight - 2.0 * twist

    def trilinear(
        self,
        a: NDArray[np.float64],
        b: NDArray[np.float64],
        c: NDArray[np.float64],
    ) -> float:
        """The trilinear form ``T(a, b, c) = ⟨l(a, b), c⟩ = h² Σ l(a, b) c`` (full-grid vectors).

        Triple self-adjoint — ``T(a, b, c) = T(a, c, b) = T(c, b, a)`` to machine precision —
        **iff** the fields vanish on the rim. The quantity the energy-conservation crux stands on.
        """
        return inner2d(self(a, b), np.asarray(c, dtype=float).ravel(), self.h)


class AiryStressSolver:
    """Elliptic solve for the von Kármán **Airy stress function** ``F`` (model #6, Part 2).

    Solves the in-plane (membrane) equation of the Föppl–von Kármán plate,

        ∇⁴F = source        (source = ``−(E e / 2) l(w, w)`` in the coupled scheme),

    on a rectangular grid with the **clamped** in-plane boundary condition ``F = 0, F,n = 0``.
    This is the physically-correct simply-supported *movable*-edge condition (Ducceschi–Touzé
    DAFx-15 §4.2, Eq. 11: the true in-plane SS condition is ``F,tt = F,nt = 0``, which
    ``F = F,n = 0`` satisfies exactly while leaving ``F,nn = σ_tt`` free — a movable edge carries
    tangential membrane stress). It is **not** the ``B = L²`` (Navier, ``F = 0, ΔF = 0``) operator
    of :func:`biharmonic_from_mask`, which forces ``σ_tt = 0`` and leaves ``σ_nt ≠ 0`` — a
    different, nonstandard edge. See ``docs/dev/von-karman-plate-plan.md`` (open decision #2).

    **Energy-first construction (symmetric SPD by squaring a clamped Laplacian).** The membrane
    energy is ``(1 / (2 E e))·‖∇²F‖²``; for clamped edges ``‖∇²F‖² = ∫(F_xx² + 2F_xy² + F_yy²)``
    (the two integrands differ by a boundary term that vanishes when ``F = F,n = 0``), so no mixed
    ``F_xy`` term is needed — a single Laplacian, squared. With ``Lc`` the full-grid Laplacian built
    from the **clamped** 1D second difference (:func:`_clamped_d2_1d`, the ghost-mirror end rows),
    and ``Wa = kron(m_y, m_x)`` the trapezoidal area weight (as in :func:`free_plate_stiffness`:
    ``h²`` interior / ``h²/2`` edge / ``h²/4`` corner),

        B_F = Lc_rᵀ Wa Lc_r,

    where ``Lc_r`` drops the rim **columns** (``F = 0`` on the rim) but keeps all **rows**.
    Curvature is sampled at every node, including the ghost-mirror rim rows, which is what makes the
    near-boundary biharmonic diagonal ``7`` rather than the Navier ``6``. ``B_F`` is symmetric
    positive-definite by construction (a Gram product; clamping removes all rigid-body modes, so —
    unlike the free plate's ``{1, x, y}`` nullspace — the nullspace here is **empty**), and is
    factored once with :func:`scipy.sparse.linalg.splu`.

    **Galerkin load (the subtle bit).** Because ``Wa`` lives *inside* ``B_F``, the consistent
    right-hand side is the **``Wa``-weighted** load ``Wa · source`` (the interior area weights are
    all ``h²``), not the bare ``source``. Forgetting the weight yields O(1) error against a fine
    operator. :meth:`solve` applies it; the matching Part-3 membrane energy is then the plain
    quadratic ``(1 / (2 E e))·Fᵀ B_F F`` (:meth:`laplacian_norm_sq`).

    **Interoperation.** ``F`` and the ``source`` are passed as **full-grid** vectors of length
    ``(Nx+1)(Ny+1)`` with the rim held at zero — the same representation the
    :class:`VonKarmanBracket` uses, so the pipeline is
    ``l(w, w)`` (full grid) → :meth:`solve` → ``F`` (full grid, rim 0) → ``l(w, F)``. The bracket's
    uniform-``h²`` inner product and this solver's trapezoidal ``Wa`` agree on the interior (they
    differ only at rim nodes, where the paired field vanishes), so Part-1 triple self-adjointness
    transfers to the ``Wa``-weighted membrane energy unchanged.

    Parameters
    ----------
    Nx, Ny : int
        Number of segments along x / y (grid is ``(Nx+1)×(Ny+1)`` nodes). Both must be ``>= 2``.
    h : float
        Grid spacing (square cells). Must be positive.
    """

    def __init__(self, Nx: int, Ny: int, h: float) -> None:
        if Nx < 2 or Ny < 2:
            raise ValueError("Nx, Ny must be >= 2 (need at least one interior node per axis).")
        if h <= 0:
            raise ValueError("h (grid spacing) must be positive.")
        self.Nx = int(Nx)
        self.Ny = int(Ny)
        self.h = float(h)
        self.n_nodes = (Nx + 1) * (Ny + 1)

        self.mask = rectangle_mask(Nx, Ny)  # interior unknowns (F = 0 on the bounding-box rim)
        self._cols = self.mask.ravel()
        self.n_interior = int(self.mask.sum())
        self.index_map = np.full(self.mask.shape, -1, dtype=np.int64)
        self.index_map[self.mask] = np.arange(self.n_interior)

        ix = sparse.identity(Nx + 1, format="csr")
        iy = sparse.identity(Ny + 1, format="csr")
        # Full-grid clamped Laplacian Lc = ∂xx + ∂yy (C-order: y outer, x inner).
        Lc = sparse.kron(iy, _clamped_d2_1d(Nx, h)) + sparse.kron(_clamped_d2_1d(Ny, h), ix)

        # Trapezoidal area weight Wa = kron(m_y, m_x): interior h², edge h²/2, corner h²/4.
        mx = np.full(Nx + 1, h)
        mx[0] = mx[-1] = 0.5 * h
        my = np.full(Ny + 1, h)
        my[0] = my[-1] = 0.5 * h
        wa = np.kron(my, mx)
        Wa = sparse.diags(wa, format="csr")

        Lc_r = Lc.tocsc()[:, self._cols]  # drop rim columns (F = 0); keep all rows
        self.Bf: sparse.csc_matrix = (Lc_r.T @ Wa @ Lc_r).tocsc()
        # Interior Galerkin load weights (all h² for a rectangle — the rim half-weights are dropped
        # with the rim columns), used to form Wa · source in solve().
        self._load_weight = wa[self._cols]
        self._lu: SuperLU = splu(self.Bf)

    def solve(self, source: NDArray[np.float64]) -> NDArray[np.float64]:
        """Solve ``∇⁴F = source`` for ``F`` (both full-grid, rim held at zero).

        ``source`` is the full-grid right-hand-side field (length :attr:`n_nodes`) — e.g.
        ``−(E e / 2) l(w, w)`` from :class:`VonKarmanBracket`; its rim values are ignored. Returns
        ``F`` as a full-grid vector with the rim at zero (ready for the ``l(w, F)`` coupling call).
        The interior load is ``Wa``-weighted internally (see the class docstring), so the caller
        passes the physical source directly and never has to remember the quadrature weight.
        """
        source = np.asarray(source, dtype=float).ravel()
        rhs = self._load_weight * source[self._cols]
        f_interior = self._lu.solve(rhs)
        return embed(f_interior, self.index_map).ravel()

    def laplacian_norm_sq(self, F: NDArray[np.float64]) -> float:
        """Discrete ``‖∇²F‖² = Fᵀ B_F F`` (full-grid ``F``; area weights are folded into ``B_F``).

        The Part-3 membrane energy is ``(1 / (2 E e))`` times this — a plain quadratic form, no
        extra weighting. ``>= 0`` (``B_F`` is positive-definite).
        """
        fi = np.asarray(F, dtype=float).ravel()[self._cols]
        return float(fi @ (self.Bf @ fi))


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


# --- the Rust swap (docs/dev/rust-migration-plan.md, Phase 2) -----------------------------------
#
# **Only part of this module is ported, deliberately.** The plan's Group D puts `operators2d` at
# Phase 5 because of `VonKarmanBracket` and `AiryStressSolver`, which factor with SuperLU. That is
# right about the module and wrong about the unit of work: the membrane is a Phase 2 model, and the
# seven functions below -- the grid, the two masks, the Laplacian, `embed` and the two inner
# products -- assemble rather than solve. So they port with the membrane and the rest waits for the
# plate family. `crates/physsynth-core/src/ops2d.rs` names what is deliberately absent.
#
# The matrix comes back from the binding as CSR triplets and is rebuilt here, exactly as in
# `operators.py`: the core never learns what SciPy is, and the modules that
# `from .operators2d import ...` never learn what Rust is.
#
# Off by default. The Python implementations are still the reference oracle.
grid_coords_py = grid_coords
rectangle_mask_py = rectangle_mask
disk_mask_py = disk_mask
laplacian_from_mask_py = laplacian_from_mask
embed_py = embed
inner2d_py = inner2d
norm2_2d_py = norm2_2d

_USE_RUST = os.environ.get("PHYSSYNTH_RS", "").strip() not in ("", "0", "false", "False")

if _USE_RUST:  # pragma: no cover - exercised by the dedicated CI job, not the default gate
    import physsynth_rs as _rs

    def _csr2d(triplets: tuple) -> sparse.csr_matrix:
        """Rebuild a ``csr_matrix`` from the binding's ``(data, indices, indptr, shape)``."""
        data, indices, indptr, shape = triplets
        return sparse.csr_matrix((data, indices, indptr), shape=shape)

    def grid_coords(N, half_extent):  # type: ignore[misc]  # noqa: F811
        return _rs.grid_coords(N, half_extent)

    def rectangle_mask(Nx, Ny):  # type: ignore[misc]  # noqa: F811
        return _rs.rectangle_mask(Nx, Ny)

    def disk_mask(X, Y, radius):  # type: ignore[misc]  # noqa: F811
        return _rs.disk_mask(X, Y, radius)

    def laplacian_from_mask(mask, h):  # type: ignore[misc]  # noqa: F811
        triplets, index_map = _rs.laplacian_from_mask_csr(mask, h)
        return _csr2d(triplets), index_map

    def embed(values, index_map):  # type: ignore[misc]  # noqa: F811
        return _rs.embed(values, index_map)

    def inner2d(f, g, h):  # type: ignore[misc]  # noqa: F811
        return _rs.inner2d(f, g, h)

    def norm2_2d(f, h):  # type: ignore[misc]  # noqa: F811
        return _rs.norm2_2d(f, h)
