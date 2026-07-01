"""Discrete von Kármán bracket ``l(a, b)`` — the operator money test (model #6, Part 1).

The nonlinear plate has **no analytic modal oracle**; energy conservation becomes *the* correctness
tool, and the whole conservation law rests on one algebraic property of the discrete bracket:
**triple self-adjointness** of its trilinear form ``T(a, b, c) = ⟨l(a, b), c⟩`` — the analogue
of the linear models' ``B = L² == Λ²`` / ``nullspace = {1, x, y}`` gates, and — unlike those — its
crux has **no 1D analogue**, so this standalone test (no time loop) *is* the model's de-risk.

Two properties are necessary and, together, sufficient for a correct bracket
(``docs/dev/von-karman-plate-plan.md``; advisor):

1. **Triple self-adjointness** ``T(a,b,c) = T(a,c,b) = T(c,b,a)`` to machine precision — the
   conservation crux. Holds **iff the fields vanish on the rim** (the simply-supported ``w = F = 0``
   edge); on non-zero borders the form is asymmetric at O(1) (a documented domain contract, not a
   bug). Empirically pinned in ``M:/claud_projects/temp/vk-bracket-probe`` and matching Bilbao's
   cell-centered-twist construction.
2. **Consistency** ``l(a,b) → a_xx b_yy + a_yy b_xx − 2 a_xy b_xy`` at O(h²) on smooth data — the
   counter to "``l ≡ 0`` also passes symmetry." Symmetry alone is *not* enough.

Plus the structural counter-checks: argument symmetry ``l(a,b) = l(b,a)`` and affine annihilation
``l(a, {1,x,y}) = 0`` (the bracket is built from second differences).
"""

import numpy as np
import pytest

from physsynth.core.operators2d import VonKarmanBracket, inner2d

SEED = 20260701


def _rim_vanishing(Nx, Ny, rng):
    """Full-grid random field (length ``(Nx+1)(Ny+1)``) that is zero on the bounding-box rim."""
    f = rng.standard_normal((Ny + 1, Nx + 1))
    f[0, :] = f[-1, :] = f[:, 0] = f[:, -1] = 0.0
    return f.ravel()


def _full_random(Nx, Ny, rng):
    """Full-grid random field with a *non-zero* border (the contract-violating case)."""
    return rng.standard_normal((Ny + 1) * (Nx + 1))


# -- 1. Argument symmetry: l(a, b) == l(b, a) ------------------------------------------------------
def test_bracket_symmetric_in_its_two_arguments():
    rng = np.random.default_rng(SEED)
    br = VonKarmanBracket(11, 9, 0.1)
    a = _full_random(11, 9, rng)
    b = _full_random(11, 9, rng)
    lab = br(a, b)
    lba = br(b, a)
    scale = np.max(np.abs(lab)) + 1e-30
    assert np.max(np.abs(lab - lba)) / scale < 1e-13


# -- 2. THE gate: triple self-adjointness on rim-vanishing fields ----------------------------------
@pytest.mark.parametrize("Nx,Ny", [(9, 7), (12, 12), (17, 11), (24, 20)])
def test_triple_self_adjointness(Nx, Ny):
    """``T(a,b,c) == T(a,c,b) == T(c,b,a)`` to machine precision — the conservation crux."""
    rng = np.random.default_rng(SEED + Nx * 100 + Ny)
    br = VonKarmanBracket(Nx, Ny, 0.07)
    a = _rim_vanishing(Nx, Ny, rng)
    b = _rim_vanishing(Nx, Ny, rng)
    c = _rim_vanishing(Nx, Ny, rng)
    t1 = br.trilinear(a, b, c)
    t2 = br.trilinear(a, c, b)  # swap b, c
    t3 = br.trilinear(c, b, a)  # swap a, c
    scale = max(abs(t1), abs(t2), abs(t3), 1e-30)
    assert abs(t1 - t2) / scale < 1e-11
    assert abs(t1 - t3) / scale < 1e-11


def test_trilinear_matches_manual_inner_product():
    """``trilinear`` really is ``⟨l(a,b), c⟩`` (guards against a silent helper drift)."""
    rng = np.random.default_rng(SEED + 1)
    br = VonKarmanBracket(10, 8, 0.05)
    a, b, c = (_rim_vanishing(10, 8, rng) for _ in range(3))
    assert br.trilinear(a, b, c) == pytest.approx(inner2d(br(a, b), c, br.h), rel=1e-12)


# -- 3. Domain contract: symmetry REQUIRES rim-vanishing fields ------------------------------------
def test_symmetry_requires_rim_vanishing_fields():
    """With a non-zero border the trilinear form is asymmetric at O(1) — documented, not a bug.

    This pins *why* the simply-supported (Dirichlet ``w = F = 0``) case is the natural first home
    for the nonlinear plate: only there do the discrete summation-by-parts boundary terms vanish.
    """
    rng = np.random.default_rng(SEED + 2)
    br = VonKarmanBracket(12, 10, 0.1)
    a, b, c = (_full_random(12, 10, rng) for _ in range(3))
    t1 = br.trilinear(a, b, c)
    t2 = br.trilinear(a, c, b)
    scale = max(abs(t1), abs(t2), 1e-30)
    assert abs(t1 - t2) / scale > 1e-3  # macroscopically asymmetric, not machine-precision


# -- 4. Affine annihilation: l(a, {1, x, y}) == 0 --------------------------------------------------
@pytest.mark.parametrize("kind", ["const", "x", "y"])
def test_affine_annihilation(kind):
    """The bracket is built from second differences, so any field affine in (x, y) drops out."""
    Nx, Ny, h = 13, 11, 0.08
    rng = np.random.default_rng(SEED + 3)
    br = VonKarmanBracket(Nx, Ny, h)
    xs = np.arange(Nx + 1) * h
    ys = np.arange(Ny + 1) * h
    X, Y = np.meshgrid(xs, ys)
    affine = {"const": np.ones_like(X), "x": X, "y": Y}[kind].ravel()
    a = _rim_vanishing(Nx, Ny, rng)
    assert np.max(np.abs(br(a, affine))) < 1e-9
    assert np.max(np.abs(br(affine, a))) < 1e-9  # symmetric, so both orders vanish


# -- 5. Consistency: l(a, b) -> analytic bracket at O(h^2) -----------------------------------------
def _manufactured(Nx, Ny, Lx, Ly):
    """``a = sin(px)sin(qy)``, ``b = sin(rx)sin(sy)`` (rim-vanishing) and their analytic bracket.

    ``L(a,b) = (p²s² + q²r²) a b − 2 p q r s cos(px)cos(qy)cos(rx)cos(sy)``.
    """
    p, q = np.pi / Lx, 2 * np.pi / Ly
    r, s = 3 * np.pi / Lx, np.pi / Ly
    xs = np.linspace(0.0, Lx, Nx + 1)
    ys = np.linspace(0.0, Ly, Ny + 1)
    X, Y = np.meshgrid(xs, ys)
    a = np.sin(p * X) * np.sin(q * Y)
    b = np.sin(r * X) * np.sin(s * Y)
    exact = (p**2 * s**2 + q**2 * r**2) * a * b - 2 * p * q * r * s * (
        np.cos(p * X) * np.cos(q * Y) * np.cos(r * X) * np.cos(s * Y)
    )
    return a.ravel(), b.ravel(), exact


def test_consistency_second_order_convergence():
    Lx, Ly = 1.0, 0.8
    errs, hs = [], []
    for Nx in (40, 80, 160, 320):
        h = Lx / Nx
        Ny = int(round(Ly / h))
        br = VonKarmanBracket(Nx, Ny, h)
        a, b, exact = _manufactured(Nx, Ny, Lx, Ly)
        got = br(a, b).reshape(Ny + 1, Nx + 1)
        interior = np.zeros_like(got, bool)
        interior[2:-2, 2:-2] = True  # skip the 1-node rim (one-sided end stencils)
        errs.append(np.max(np.abs(got[interior] - exact.reshape(Ny + 1, Nx + 1)[interior])))
        hs.append(h)
    errs, hs = np.array(errs), np.array(hs)
    rates = np.log(errs[:-1] / errs[1:]) / np.log(hs[:-1] / hs[1:])
    assert np.all(rates > 1.9)  # O(h^2)


# -- Constructor validation ------------------------------------------------------------------------
@pytest.mark.parametrize("Nx,Ny,h", [(1, 5, 0.1), (5, 1, 0.1), (5, 5, 0.0), (5, 5, -0.1)])
def test_constructor_rejects_bad_params(Nx, Ny, h):
    with pytest.raises(ValueError):
        VonKarmanBracket(Nx, Ny, h)
