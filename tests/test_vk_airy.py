"""Airy stress function elliptic solve ``B_F`` — von Kármán plate, Part 2 (model #6).

The nonlinear plate couples transverse ``w`` to an Airy stress function ``F`` via ``∇⁴F = source``.
Part 2 builds and validates that elliptic solve in isolation (no time loop yet), with the
**clamped** in-plane boundary condition ``F = 0, F,n = 0`` — the physically-correct
simply-supported *movable*-edge condition (Ducceschi–Touzé DAFx-15 §4.2, Eq. 11), **not** the
Navier ``B = L²`` (``F = 0, ΔF = 0``). See ``docs/dev/von-karman-plate-plan.md`` (decision #2).

The gate is a **manufactured-solution O(h²) convergence** check plus the **discriminator** that
proves the operator really is clamped and not simply-supported: the manufactured field
``F = (1-cos)(1-cos)`` has ``F = F,n = 0`` on every edge (so it *is* a clamped solution) but
``ΔF ≠ 0`` there (so it is *not* a Navier solution). Solving the same right-hand side through the
simply-supported ``biharmonic_from_mask`` operator therefore fails to converge, while the clamped
solver recovers it at O(h²).
"""

import numpy as np
import pytest
from scipy import sparse
from scipy.sparse.linalg import splu

from physsynth.core.operators2d import (
    AiryStressSolver,
    _clamped_d2_1d,
    biharmonic_from_mask,
    rectangle_mask,
)

Lx, Ly = 1.0, 0.8


def _manufactured(Nx, Ny):
    """``F = (1-cos(2πx/Lx))(1-cos(2πy/Ly))`` (rim-clamped, ``ΔF ≠ 0`` on the rim) and ``∇⁴F``.

    ``g = 1-cos(ax)`` gives ``g'''' = -a⁴cos(ax)`` (note the sign), so
    ``∇⁴F = -a⁴cos(ax)q + 2a²b²cos(ax)cos(by) - g·b⁴cos(by)``.
    """
    a, b = 2 * np.pi / Lx, 2 * np.pi / Ly
    xs = np.linspace(0.0, Lx, Nx + 1)
    ys = np.linspace(0.0, Ly, Ny + 1)
    X, Y = np.meshgrid(xs, ys)
    g = 1.0 - np.cos(a * X)
    q = 1.0 - np.cos(b * Y)
    F = g * q
    lap4 = (
        -(a**4) * np.cos(a * X) * q
        + 2.0 * a**2 * b**2 * np.cos(a * X) * np.cos(b * Y)
        - g * b**4 * np.cos(b * Y)
    )
    return F, lap4


# -- THE gate: manufactured-solution O(h^2) convergence --------------------------------------------
def test_manufactured_solution_second_order():
    errs, hs = [], []
    for Nx in (40, 80, 160):
        h = Lx / Nx
        Ny = int(round(Ly / h))
        F_exact, lap4 = _manufactured(Nx, Ny)
        solver = AiryStressSolver(Nx, Ny, h)
        F = solver.solve(lap4.ravel())
        errs.append(np.max(np.abs(F - F_exact.ravel())))
        hs.append(h)
    errs, hs = np.array(errs), np.array(hs)
    rates = np.log(errs[:-1] / errs[1:]) / np.log(hs[:-1] / hs[1:])
    assert np.all(rates > 1.9)  # O(h^2)
    assert errs[-1] < 5e-3


# -- The discriminator: clamped is NOT the simply-supported B = L^2 --------------------------------
def test_clamped_is_not_navier_l2():
    """Solving the same manufactured RHS with the SS (Navier) ``B = L²`` operator does not converge
    — the one check that proves ``AiryStressSolver`` builds the *clamped* biharmonic, not ``L²``.
    """
    Nx = 80
    h = Lx / Nx
    Ny = int(round(Ly / h))
    F_exact, lap4 = _manufactured(Nx, Ny)

    clamped = AiryStressSolver(Nx, Ny, h)
    err_clamped = np.max(np.abs(clamped.solve(lap4.ravel()) - F_exact.ravel()))

    mask = rectangle_mask(Nx, Ny)
    B, _ = biharmonic_from_mask(mask, h)  # SS operator, strong-form solve of the same source
    f_ss = splu(B.tocsc()).solve(lap4[mask])
    err_ss = np.max(np.abs(f_ss - F_exact[mask]))

    assert err_ss > 50.0 * err_clamped  # SS error saturates ~O(1); clamped -> 0 at O(h^2)


# -- Structural properties -------------------------------------------------------------------------
def test_operator_symmetric():
    solver = AiryStressSolver(14, 11, 0.07)
    B = solver.Bf.toarray()
    assert np.max(np.abs(B - B.T)) / np.max(np.abs(B)) < 1e-12


def test_operator_positive_definite():
    """``B_F`` is SPD — clamping removes *all* rigid-body modes, so (unlike the free plate's
    ``{1, x, y}``) the nullspace is empty. A smoke test: it cannot distinguish clamped from Navier
    (both are SPD); the manufactured discriminator above does that.
    """
    solver = AiryStressSolver(16, 12, 0.06)
    smallest = np.linalg.eigvalsh(solver.Bf.toarray())[0]
    assert smallest > 0.0


def test_zero_source_gives_zero_field():
    """``F → 0`` as source → 0 (a stated Part-2 gate; also the ``w → 0`` limit of the coupling)."""
    solver = AiryStressSolver(12, 10, 0.05)
    F = solver.solve(np.zeros(solver.n_nodes))
    assert np.max(np.abs(F)) == 0.0


def test_solved_field_is_rim_vanishing():
    """The returned ``F`` is full-grid with the clamped rim held exactly at zero."""
    _, lap4 = _manufactured(20, 16)
    solver = AiryStressSolver(20, 16, Lx / 20)
    F = solver.solve(lap4.ravel()).reshape(solver.mask.shape)
    assert np.all(F[0, :] == 0.0) and np.all(F[-1, :] == 0.0)
    assert np.all(F[:, 0] == 0.0) and np.all(F[:, -1] == 0.0)


def test_laplacian_norm_sq_matches_quadratic_form():
    """The membrane-energy quadratic ``Fᵀ B_F F`` (Part 3 uses ``(1/2Ee)`` of it), a plain dot."""
    rng = np.random.default_rng(7)
    solver = AiryStressSolver(13, 9, 0.08)
    F = rng.standard_normal(solver.n_nodes)
    F[~solver.mask.ravel()] = 0.0
    fi = F[solver.mask.ravel()]
    assert solver.laplacian_norm_sq(F) == pytest.approx(float(fi @ (solver.Bf @ fi)))
    assert solver.laplacian_norm_sq(F) > 0.0


# -- The 1D ghost-mirror crux ----------------------------------------------------------------------
def test_clamped_1d_gram_is_standard_clamped_biharmonic():
    """``D2c.T @ Wa @ D2c`` (trapezoidal ``Wa``) reproduces the textbook clamped-beam biharmonic:
    near-boundary diagonal ``7``, interior ``6``, off-diagonals ``-4`` and ``1`` — the whole reason
    the end-row off-diagonal is doubled. With ``Wa = I`` the ``7`` would come out ``9`` (wrong).
    """
    N, h = 8, 0.1
    d2c = _clamped_d2_1d(N, h)
    m = np.full(N + 1, h)
    m[0] = m[-1] = 0.5 * h
    Wa = sparse.diags(m)
    cols = np.zeros(N + 1, dtype=bool)
    cols[1:-1] = True
    d2c_r = d2c.tocsc()[:, cols]
    B = (d2c_r.T @ Wa @ d2c_r).toarray() * h**3
    diag = np.diag(B)
    assert diag[0] == pytest.approx(7.0)
    assert diag[-1] == pytest.approx(7.0)
    assert np.allclose(diag[1:-1], 6.0)
    assert np.allclose(np.diag(B, 1), -4.0)
    assert np.allclose(np.diag(B, 2), 1.0)


# -- Constructor validation ------------------------------------------------------------------------
@pytest.mark.parametrize("Nx,Ny,h", [(1, 5, 0.1), (5, 1, 0.1), (5, 5, 0.0), (5, 5, -0.1)])
def test_constructor_rejects_bad_params(Nx, Ny, h):
    with pytest.raises(ValueError):
        AiryStressSolver(Nx, Ny, h)
