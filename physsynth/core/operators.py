"""Finite-difference operators (HANDOFF Appendix A) and the discrete inner product.

All spatial operators act on a 1-D grid array ``u`` of shape ``(N + 1,)`` sampling
``u(l*h)`` for ``l = 0 .. N``. The discrete inner product ``<f, g> = h * sum_l f[l] g[l]``
(and its norm) is the bookkeeping device behind every energy proof in this project, so it lives
here next to the operators it pairs with.

**The implementation is Rust**: ``crates/physsynth-core/src/ops.rs``, whose doc comments are now the
only copy of the physics — why ``biharmonic_matrix`` is built as ``D2 @ D2`` (both simply-supported
conditions baked in, a ``5/h^4`` boundary-adjacent diagonal, the energy identity exact), and why
``free_beam_stiffness`` is the Gram product ``h D2ᵀD2`` with a trapezoidal mass whose ``h/2`` end
cells *are* the free-end closure. The bars are ``crates/physsynth-core/tests/ops.rs``. The Python
body was deleted with phase A of ``docs/dev/python-retirement-plan.md`` (§21).

Unlike ``exciter``, this module is **not** a set of re-exports, and the two reasons are the two
seams the swap block had carried since Phase 1, now the whole body:

1. **The SciPy rebuild.** ``physsynth-core`` must not know what SciPy is, so the binding hands the
   three matrix builders back as CSR triplets ``(data, indices, indptr, shape)`` and :func:`_csr`
   turns them into the ``csr_matrix`` callers expect — the same arrangement as ``operators2d``.
2. **The input coercion.** The binding wants a contiguous float64 array and says so; NumPy would
   happily difference a strided view or an int array. :func:`_asarray` re-widens the door to the
   original's, copying only when it has to.

``second_difference_matrix`` and ``biharmonic_matrix`` with ``N < 2`` raise ``ValueError`` with a
message saying what is wrong; the deleted Python fell through to NumPy's "negative dimensions are
not allowed". Same exception type, better text.

Headless: NumPy + SciPy (sparse). No I/O, no plotting.
"""

from __future__ import annotations

import numpy as np
import physsynth_rs as _rs
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


def _asarray(a: object) -> NDArray[np.float64]:
    """Whatever the caller passed, as the contiguous float64 array the binding requires."""
    return np.ascontiguousarray(np.asarray(a, dtype=np.float64))


def _csr(triplets: tuple) -> sparse.csr_matrix:
    """Rebuild a ``csr_matrix`` from the binding's ``(data, indices, indptr, shape)``."""
    data, indices, indptr, shape = triplets
    return sparse.csr_matrix((data, indices, indptr), shape=shape)


def delta_x_forward(u: NDArray[np.float64], h: float) -> NDArray[np.float64]:
    """Forward difference ``(u[l+1] - u[l]) / h``: the ``N`` inter-node strains."""
    return _rs.delta_x_forward(_asarray(u), h)


def delta_x_backward(u: NDArray[np.float64], h: float) -> NDArray[np.float64]:
    """Backward difference ``(u[l] - u[l-1]) / h``; the same array as :func:`delta_x_forward`."""
    return _rs.delta_x_backward(_asarray(u), h)


def delta_xx(u: NDArray[np.float64], h: float) -> NDArray[np.float64]:
    """Second difference at the ``N - 1`` interior nodes; boundaries are the caller's."""
    return _rs.delta_xx(_asarray(u), h)


def delta_xxxx(u: NDArray[np.float64], h: float) -> NDArray[np.float64]:
    """Fourth difference at nodes ``2 .. N-2``, where the 5-point stencil fits without a ghost."""
    return _rs.delta_xxxx(_asarray(u), h)


def inner(f: NDArray[np.float64], g: NDArray[np.float64], h: float) -> float:
    """Discrete inner product ``<f, g> = h * sum_l f[l] g[l]``."""
    return _rs.inner(_asarray(f), _asarray(g), h)


def norm2(f: NDArray[np.float64], h: float) -> float:
    """Squared discrete norm ``||f||^2 = <f, f>`` (>= 0)."""
    return _rs.norm2(_asarray(f), h)


def second_difference_matrix(N: int, h: float) -> sparse.csr_matrix:
    """``(N-1) x (N-1)`` Dirichlet second-difference operator on the interior nodes."""
    return _csr(_rs.second_difference_matrix_csr(N, h))


def biharmonic_matrix(N: int, h: float) -> sparse.csr_matrix:
    """``(N-1) x (N-1)`` simply-supported biharmonic, ``D2 @ D2``; symmetric positive-definite."""
    return _csr(_rs.biharmonic_matrix_csr(N, h))


def free_beam_stiffness(N: int, h: float) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    """``(K, W)`` for the free-free beam on all ``N + 1`` nodes: ``K = h D2ᵀD2``, lumped ``W``."""
    k, w = _rs.free_beam_stiffness_csr(N, h)
    return _csr(k), _csr(w)
