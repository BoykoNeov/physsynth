"""Finite-difference operators (HANDOFF Appendix A) and the discrete inner product.

All spatial operators act on a 1-D grid array ``u`` of shape ``(N + 1,)`` sampling
``u(l*h)`` for ``l = 0 .. N``. The discrete inner product ``<f, g> = h * sum_l f[l] g[l]``
(and its norm) is the bookkeeping device behind every energy proof in this project, so it lives
here next to the operators it pairs with.

Headless: NumPy only. No I/O, no plotting.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import sparse

__all__ = [
    "delta_x_forward",
    "delta_x_backward",
    "delta_xx",
    "delta_xxxx",
    "inner",
    "norm2",
    "second_difference_matrix",
    "biharmonic_matrix",
    "free_beam_stiffness",
]


def delta_x_forward(u: NDArray[np.float64], h: float) -> NDArray[np.float64]:
    """Forward spatial difference ``delta_x+ u[l] = (u[l+1] - u[l]) / h``.

    Returns the ``N`` inter-node strains (length ``len(u) - 1``); element ``l`` lives on the
    segment between nodes ``l`` and ``l+1``.
    """
    return (u[1:] - u[:-1]) / h


def delta_x_backward(u: NDArray[np.float64], h: float) -> NDArray[np.float64]:
    """Backward spatial difference ``delta_x- u[l] = (u[l] - u[l-1]) / h`` (length ``len(u) - 1``).

    Numerically equal array to :func:`delta_x_forward`; provided for notational symmetry.
    """
    return (u[1:] - u[:-1]) / h


def delta_xx(u: NDArray[np.float64], h: float) -> NDArray[np.float64]:
    """Second spatial difference at interior nodes.

    ``delta_xx u[l] = (u[l+1] - 2 u[l] + u[l-1]) / h^2`` for ``l = 1 .. N-1``.
    Returns the ``N - 1`` interior values (length ``len(u) - 2``); boundary handling is the
    caller's responsibility.
    """
    return (u[2:] - 2.0 * u[1:-1] + u[:-2]) / (h * h)


def delta_xxxx(u: NDArray[np.float64], h: float) -> NDArray[np.float64]:
    """Fourth spatial difference at interior nodes ``l = 2 .. N-2`` (the stiffness operator).

    ``delta_xxxx u[l] = (u[l+2] - 4u[l+1] + 6u[l] - 4u[l-1] + u[l-2]) / h^4``. Returns the
    ``N - 3`` values where the 5-point stencil fits without a ghost node (length ``len(u) - 4``);
    boundary rows are the caller's responsibility. Provided as the pure-operator reference that
    :func:`biharmonic_matrix` (built as ``delta_xx`` squared) must reproduce in the interior.
    """
    return (u[4:] - 4.0 * u[3:-1] + 6.0 * u[2:-2] - 4.0 * u[1:-3] + u[:-4]) / (h ** 4)


def second_difference_matrix(N: int, h: float) -> sparse.csr_matrix:
    """``(N-1) x (N-1)`` Dirichlet second-difference operator ``delta_xx`` on interior nodes.

    Acts on the interior unknowns ``u[1 .. N-1]`` with the boundary condition ``u[0] = u[N] = 0``
    baked into the stencil (the ghost values are simply zero). Tridiagonal: ``-2/h^2`` on the
    diagonal, ``1/h^2`` on the off-diagonals. Symmetric negative-definite; its eigenvalues are
    ``-(4/h^2) sin^2(m pi / 2N)`` with eigenvectors ``sin(m pi l / N)``.
    """
    n = N - 1
    main = np.full(n, -2.0 / (h * h))
    off = np.full(n - 1, 1.0 / (h * h))
    return sparse.diags([off, main, off], offsets=[-1, 0, 1], format="csr")


def biharmonic_matrix(N: int, h: float) -> sparse.csr_matrix:
    """``(N-1) x (N-1)`` simply-supported biharmonic operator ``delta_xxxx`` on interior nodes.

    Built as ``D2 @ D2`` with ``D2 = second_difference_matrix(N, h)``. Doing so (rather than
    hand-coding the 5-point rows) bakes in *both* simply-supported conditions ``u = 0`` and
    ``u_xx = 0``: the second condition is exactly ``delta_xx u = 0`` at the ends, which the squared
    operator enforces via the zero ghost of the inner ``delta_xx``. Two payoffs:

    - the boundary-adjacent diagonal comes out ``5/h^4`` (not the interior ``6/h^4``) automatically;
    - the discrete biharmonic *is* ``(delta_xx)^2`` exactly, so the energy identity
      ``<delta_xxxx f, g> = <delta_xx f, delta_xx g>`` holds to machine precision (since ``D2`` is
      symmetric: ``(D2 D2 f) . g = (D2 f) . (D2 g)``). This is what keeps the lossless energy
      conserved and makes ``sin(m pi l / N)`` an exact eigenvector (eigenvalue ``+p^4``).

    Symmetric positive-definite (pentadiagonal).
    """
    d2 = second_difference_matrix(N, h)
    return (d2 @ d2).tocsr()


def free_beam_stiffness(N: int, h: float) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    """Energy-first free-free Euler–Bernoulli bending operator on the ``N+1`` nodes.

    Returns ``(K, W)`` — the building block for the **free** flexural resonator (the 1D rehearsal
    of the free-edge Chladni plate, ``docs/dev/plate-free-edge-plan.md`` Part 0). Unlike the
    simply-supported :func:`biharmonic_matrix` (clamped/Dirichlet ends), **both ends are free
    unknowns**, so the operator acts on the *full* node set:

    - ``K`` is the symmetric **positive-semidefinite** stiffness representing the bending energy
      ``∫(u_xx)² dx ≈ uᵀ K u``. It is assembled *from the energy* as a Gram product
      ``K = D2ᵀ Wc D2`` with ``D2`` the ``(N-1)×(N+1)`` **interior** second-difference operator
      (curvature at nodes ``1 .. N-1``) and ``Wc = h·I`` the curvature-quadrature weight, i.e.
      ``K = h · D2ᵀ D2``. Because ``D2`` annihilates linear data, ``K``'s nullspace is **exactly the
      rigid-body space ``{1, x}``** (translation + tilt) — the free-free natural BCs ``u_xx = 0``
      and ``u_xxx = 0`` are enforced by construction, with no hand-coded boundary rows.
    - ``W`` is the diagonal **trapezoidal** mass (lumped quadrature): ``h`` at interior nodes,
      ``h/2`` at the two end nodes. It sits on the LHS of ``W u_tt = -kappa² K u``.

    **Why no special end stiffness rows are needed (the de-risk payoff):** the mass-normalised
    operator ``W⁻¹K`` comes out as Bilbao's energy-conserving free-free bar exactly — end row
    ``(1/h⁴)[2,-4,2]``, next row ``(1/h⁴)[-2,5,-4,1]``, interior ``(1/h⁴)[1,-4,6,-4,1]``. The factor
    of two at the very end is supplied by the ``h/2`` mass cell, **not** by a hand-written stiffness
    stencil. The free-edge closure falls out of the mass lumping; in 2D the edge-½/corner-¼ ``W`` is
    expected to supply the corner closure the same way (see the plan). Second-order accurate; the
    low eigenvalues converge at O(h²) (there is no exact discrete eigenvector here — free edges have
    no ``sin·sin`` analogue).

    ``K`` is only PSD (the ``{1, x}`` nullspace), so a generalized eigensolve ``K φ = μ W φ`` needs
    a small **negative** shift; the time-step matrix ``A = (1+σk)W + θk²κ²K`` is still SPD because
    ``W`` is. Pairs with :func:`physsynth.analysis.modal.free_free_beam_freqs` (the closed-form
    oracle ``f_n = kappa (β_n L)²/(2π L²)``).
    """
    if N < 2:
        raise ValueError("N must be >= 2 (need at least one interior curvature).")
    n_curv = N - 1
    inv_h2 = 1.0 / (h * h)

    # D2: interior second difference. Row r (r = 0 .. N-2) is the curvature at node l = r+1, with
    # entries [1, -2, 1]/h² at columns l-1, l, l+1.
    nodes = np.arange(1, N)  # interior node indices l = 1 .. N-1
    rows = np.repeat(np.arange(n_curv), 3)
    cols = np.empty(3 * n_curv, dtype=np.int64)
    cols[0::3] = nodes - 1
    cols[1::3] = nodes
    cols[2::3] = nodes + 1
    data = np.tile(np.array([inv_h2, -2.0 * inv_h2, inv_h2]), n_curv)
    D2 = sparse.coo_matrix((data, (rows, cols)), shape=(n_curv, N + 1)).tocsr()

    K = (h * (D2.T @ D2)).tocsr()  # = D2ᵀ (h I) D2, curvature-quadrature weight Wc = h

    w = np.full(N + 1, h)
    w[0] = w[-1] = 0.5 * h  # trapezoidal: half cells at the two free ends
    W = sparse.diags(w, format="csr")
    return K, W


def inner(f: NDArray[np.float64], g: NDArray[np.float64], h: float) -> float:
    """Discrete inner product ``<f, g> = h * sum_l f[l] g[l]``."""
    return float(h * np.dot(f, g))


def norm2(f: NDArray[np.float64], h: float) -> float:
    """Squared discrete norm ``||f||^2 = <f, f> = h * sum_l f[l]^2`` (>= 0)."""
    return float(h * np.dot(f, f))
