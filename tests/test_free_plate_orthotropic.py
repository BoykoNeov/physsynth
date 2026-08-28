"""Free-edge orthotropic plate (model #5of): the free plate gets a grain.

Four bending constants, not three.

``docs/dev/orthotropic-free-plate-plan.md``. Model #5o gave the *simply-supported* plate a grain and
needed three bending numbers, because a pinned rim sees the coupling and torsional rigidities only
through their combination ``H = D_1 + 2 D_xy``. A **free** rim sees them apart — its corner force is
pure torsion — so this branch takes four, and the two boundaries stop being the same material
question. That is not a claim inherited from the literature:
``test_supported_is_blind_to_the_split`` and ``test_free_plate_is_not`` measure both halves of it in
one file, and the supported half comes out **bit-identical** while the free fundamental moves 6.5x.

There is no closed-form spectrum here (there is none for the isotropic free plate either) and no
freely-citable orthotropic FFFF table, so the four constants are validated by **four independent
probes**, three of them exact, one detector per constant:

======================  ==========================================================================
``grain_torsion``       the centred saddle's Rayleigh quotient — provably blind to the other three,
                        numerator ``4 g_xy ab`` in closed form, and a one-sided bound on the
                        fundamental (``modal.free_plate_twist_bound``)
``grain_x``/``_y``      an **exact** reduction to the shipped 1-D free beam on ``y``- (``x``-)
                        independent fields — but only at zero coupling, and the way it breaks with
                        coupling is anticlastic curvature, i.e. physics
``grain_coupling``      the ``(x², y²)`` bilinear probe, whose discrete value is exactly
                        ``4 g_1 h²(Nx-1)(Ny-1)`` (``modal.free_plate_coupling_form``) — the *only*
                        one of the four that responds to the coupling rigidity at all
======================  ==========================================================================

Plus the usual tier-1 ledger (conservation/passivity), which is nearly blind here and says so: any
symmetric ``K`` conserves exactly, so a wrong coefficient is perfectly stable and perfectly
conservative.
"""

import numpy as np
import pytest
from helpers import (
    KAPPA_PLATE_DEFAULT,
    RHO_AREAL_DEFAULT,
    SPRUCE,
    arpack_v0,
    free_plate_low_eigenfrequencies,
    make_free_plate,
    make_orthotropic_free_plate,
    spruce_free_grain,
)
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh

from physsynth.analysis import modal
from physsynth.core.engine import simulate
from physsynth.core.exciter import raised_cosine_2d
from physsynth.core.operators import free_beam_stiffness
from physsynth.core.operators2d import free_plate_stiffness
from physsynth.core.plate import Plate, grain_ratios_from_material

KAPPA = KAPPA_PLATE_DEFAULT
DRIFT_TOL = 1e-10  # tier 1: the same acceptance bar as every other resonator

# The 7-grid survey of #5o §2, where the *supported* branch's two assemblies agree bit-for-bit on
# four grids and differ by ~2e-16 on three. Reused here precisely because it contains both kinds:
# asserting the free branch's collapse only on a friendly grid would pass while proving nothing.
GRIDS = [
    (12, 12, 1.0 / 12), (24, 24, 1.0 / 24), (20, 14, 0.05), (17, 17, 1.0 / 17),
    (13, 9, 0.62 / 13), (16, 16, 0.7 / 16), (11, 7, 0.31 / 11),
]


def _centred(Nx, Ny, h):
    """Grids measured from the plate's CENTROID — what makes ``xy`` orthogonal to ``{1, x, y}``."""
    xs = (np.arange(Nx + 1) - 0.5 * Nx) * h
    ys = (np.arange(Ny + 1) - 0.5 * Ny) * h
    return np.meshgrid(xs, ys)  # (Ny+1, Nx+1), C-order matches K


def _pencil(K, W):
    """Dense generalized eigen-decomposition of ``K φ = mu W φ`` (small grids only)."""
    return eigh(K.toarray(), np.diag(W.diagonal()))


def _lam(mu, side):
    """The dimensionless frequency parameter ``lambda = omega a²/kappa = a² sqrt(mu)``."""
    return side * side * np.sqrt(np.clip(mu, 0.0, None))


# =====================================================================================
# Assembly: one code path, and it must not have moved the shipped isotropic plate
# =====================================================================================


def test_isotropic_split_is_bit_identical_on_every_grid():
    """The default free plate routes through the **same** four-constant assembly, and the
    coefficients ``1.0, 1.0, nu, 4*((1-nu)/2)`` reproduce it **byte for byte**.

    This is the design decision #5o could not make for the supported branch, where the isotropic
    collapse is only *grid-dependently* bit-exact and the default therefore keeps a separate path.
    Here it is exact everywhere, so there is one path — asserted on the same 7 grids, including the
    three where the supported assemblies differ.
    """
    for Nx, Ny, h in GRIDS:
        for nu in (0.3, 0.0, 0.49, -0.5):
            K_plain, W_plain, _ = free_plate_stiffness(Nx, Ny, h, nu)
            K_split, W_split, _ = free_plate_stiffness(
                Nx, Ny, h, nu, grain_x=1.0, grain_y=1.0,
                grain_coupling=nu, grain_torsion=0.5 * (1.0 - nu),
            )
            a, b = K_plain.tocsr(), K_split.tocsr()
            a.sort_indices()
            b.sort_indices()
            assert np.array_equal(a.indptr, b.indptr) and np.array_equal(a.indices, b.indices), (
                f"sparsity changed at {Nx}x{Ny}, nu={nu}"
            )
            assert np.array_equal(a.data, b.data), (
                f"the nu-derived split is not bit-identical at {Nx}x{Ny} h={h!r} nu={nu}: "
                f"max |diff| = {abs(a.data - b.data).max():.3e}"
            )
            assert np.array_equal(W_plain.diagonal(), W_split.diagonal())


def test_resonator_default_free_plate_is_bit_identical_to_the_helper():
    """And the same through :class:`Plate`, whose free branch now always passes four constants."""
    p = make_free_plate(N=16, nu=0.3)
    Ny = round(p.Ly / p.h)
    K, W, _ = free_plate_stiffness(p.N, Ny, p.h, 0.3)
    assert np.array_equal(K.tocsr().data, p.K.tocsr().data), "default free plate moved"
    assert np.array_equal(W.diagonal(), p.W.diagonal())
    assert (p.grain_coupling, p.grain_torsion) == (0.3, 0.35), "the nu-derived split"


def test_resonator_builds_the_operator_its_own_parameters_imply():
    """The **seam**: a grained :class:`Plate` must build exactly what its four constants say.

    Every other test here probes ``free_plate_stiffness`` directly, so a mis-wired keyword in the
    constructor's plumbing (``grain_y`` where ``grain_coupling`` belongs) would survive all of
    them. Deliberately **non-square** with all four constants distinct — every other ``Plate``-level
    check here is square, where an x/y transposition is invisible.
    """
    g = dict(grain_x=2.3, grain_y=0.41, grain_coupling=-0.37, grain_torsion=0.62)
    p = Plate(
        Lx=0.62, Ly=0.42, kappa=KAPPA, rho=RHO_AREAL_DEFAULT, fs=40000.0, N=13,
        boundary="free", **g,
    )
    K, W, _ = free_plate_stiffness(p.N, round(p.Ly / p.h), p.h, p.nu, **g)
    assert np.array_equal(K.tocsr().data, p.K.tocsr().data), "resonator K != its own parameters"
    assert np.array_equal(W.diagonal(), p.W.diagonal())
    assert p.grain_cross == -0.37 + 2.0 * 0.62
    assert p.nu == -0.37 / 2.3, "the implied nu_yx = g_1/g_x"


def _dense_ortho_free_K(Nx, Ny, h, g_x, g_y, g_1, g_xy):
    """Explicit per-node assembly of the 4-constant form — no Kronecker products anywhere.

    The independent reference for the ordering. #5b's version of this took one constant; extending
    it rather than trusting it is the plan's §7 note about the isotropic bit-identity exercising
    exactly one (x/y-symmetric) point of a 4-dimensional space.
    """
    nn = (Nx + 1) * (Ny + 1)

    def idx(i, j):
        return j * (Nx + 1) + i

    inv_h2 = 1.0 / (h * h)
    Dxx = np.zeros((nn, nn))
    Dyy = np.zeros((nn, nn))
    for j in range(Ny + 1):
        for i in range(1, Nx):
            r = idx(i, j)
            Dxx[r, idx(i - 1, j)] += inv_h2
            Dxx[r, idx(i, j)] += -2.0 * inv_h2
            Dxx[r, idx(i + 1, j)] += inv_h2
    for j in range(1, Ny):
        for i in range(Nx + 1):
            r = idx(i, j)
            Dyy[r, idx(i, j - 1)] += inv_h2
            Dyy[r, idx(i, j)] += -2.0 * inv_h2
            Dyy[r, idx(i, j + 1)] += inv_h2

    wa = np.zeros(nn)
    for j in range(Ny + 1):
        wy = h if 0 < j < Ny else 0.5 * h
        for i in range(Nx + 1):
            wx = h if 0 < i < Nx else 0.5 * h
            wa[idx(i, j)] = wx * wy
    Wa = np.diag(wa)

    cross = Dxx.T @ Wa @ Dyy
    K = g_x * (Dxx.T @ Wa @ Dxx) + g_y * (Dyy.T @ Wa @ Dyy) + g_1 * (cross + cross.T)

    Dxy = np.zeros((Nx * Ny, nn))
    for j in range(Ny):
        for i in range(Nx):
            c = j * Nx + i
            Dxy[c, idx(i, j)] += inv_h2
            Dxy[c, idx(i + 1, j)] += -inv_h2
            Dxy[c, idx(i, j + 1)] += -inv_h2
            Dxy[c, idx(i + 1, j + 1)] += inv_h2
    K += 4.0 * g_xy * (h * h) * (Dxy.T @ Dxy)
    return K


def test_matches_direct_assembly_with_four_distinct_constants():
    """Non-square grids AND ``g_x != g_y != g_1 != g_xy``, so an axis swap cannot hide."""
    for Nx, Ny in [(5, 3), (4, 5), (6, 4)]:
        h = 0.1
        g = (2.3, 0.41, -0.37, 0.62)
        K, _, index_map = free_plate_stiffness(
            Nx, Ny, h, 0.3, grain_x=g[0], grain_y=g[1], grain_coupling=g[2], grain_torsion=g[3]
        )
        ref = _dense_ortho_free_K(Nx, Ny, h, *g)
        scale = abs(ref).max()
        diff = np.abs(K.toarray() - ref).max()
        assert diff < 1e-12 * scale, f"kron != direct at {Nx}x{Ny}: {diff:.2e} (scale {scale:.2e})"
        assert np.array_equal(index_map.ravel(), np.arange((Nx + 1) * (Ny + 1)))


def test_operator_stays_symmetric_with_a_grain():
    """Symmetry makes the energy an exact algebraic identity — it must survive orthotropy."""
    K, _, _ = free_plate_stiffness(12, 10, 0.1, 0.3, **spruce_free_grain())
    asym = abs(K - K.T).max()
    assert asym < 1e-12 * abs(K).max(), f"K not symmetric with a grain: {asym:.3e}"


# =====================================================================================
# The nullspace: three rigid-body modes, and a fourth if the torsion is switched off
# =====================================================================================


def test_rigid_body_nullspace_survives_the_grain():
    """``K{1, x, y} = 0`` to machine precision for a *grained* plate, with ``K(xy) != 0``.

    The nullspace argument is grain-independent by construction (the bending-diagonal blocks kill
    everything linear-per-line whatever ``g_x``, ``g_y``; the twist block kills separable fields),
    so this is the cheapest check that four coefficients did not break the operator's structure.
    """
    Nx, Ny, h = 14, 12, 0.1
    K, _, _ = free_plate_stiffness(Nx, Ny, h, 0.3, **spruce_free_grain())
    X, Y = np.meshgrid(np.arange(Nx + 1) * h, np.arange(Ny + 1) * h)
    k_fro = np.sqrt((K.toarray() ** 2).sum())
    rel = {
        name: np.linalg.norm(K @ v) / (k_fro * np.linalg.norm(v))
        for name, v in {
            "1": np.ones_like(X).ravel(), "x": X.ravel(), "y": Y.ravel(),
            "xy": (X * Y).ravel(),
        }.items()
    }
    for name in ("1", "x", "y"):
        assert rel[name] < 1e-12, f"K@{name} not in the nullspace with a grain: {rel[name]:.2e}"
    assert rel["xy"] > 1e6 * max(rel["1"], rel["x"], rel["y"]), "nullspace contrast too small"


def test_zero_torsion_puts_the_saddle_into_the_nullspace():
    """``grain_torsion = 0`` is **degenerate, not stiff** — a 4th zero mode appears.

    The operator helper permits it (so this test can build it); :class:`Plate` rejects it. This is
    the cleanest statement of what the torsional rigidity is *for*: it is the only term that gives
    the saddle ``xy`` any energy at all.
    """
    Nx, Ny, h = 10, 8, 0.05
    K, W, _ = free_plate_stiffness(
        Nx, Ny, h, 0.3, grain_x=1.0, grain_y=0.5, grain_coupling=0.1, grain_torsion=0.0
    )
    vals = np.sort(_pencil(K, W)[0])
    scale = abs(vals).max()
    n_zero = int(np.sum(np.abs(vals) < 1e-9 * scale))
    assert n_zero == 4, f"expected the 3 rigid modes + the saddle, got {n_zero}: {vals[:6]}"
    with pytest.raises(ValueError, match="joins the rigid-body nullspace"):
        make_orthotropic_free_plate(N=8, grain_torsion=0.0)


# =====================================================================================
# Detector 1 (grain_torsion): the saddle's Rayleigh quotient
# =====================================================================================


def _twist_quotient(Nx, Ny, h, g_x, g_y, g_1, g_xy):
    K, W, _ = free_plate_stiffness(
        Nx, Ny, h, 0.3, grain_x=g_x, grain_y=g_y, grain_coupling=g_1, grain_torsion=g_xy
    )
    X, Y = _centred(Nx, Ny, h)
    xy = (X * Y).ravel()
    return float(xy @ (K @ xy)), float(xy @ (W.diagonal() * xy))


def test_twist_quotient_is_blind_to_the_other_three_constants():
    """``R(xy)`` must depend on ``grain_torsion`` **alone** — that is what makes it a probe.

    The collocated second differences annihilate the saddle (it is linear along every grid line), so
    the ``g_x``, ``g_y`` and coupling terms cannot contribute in exact arithmetic. In doubles they
    cancel rather than vanish, which is why the bar is 1e-11 and not 1e-16 — measured 1.6e-13.
    """
    Nx, Ny, h = 16, 12, 0.05
    quotients = [
        _twist_quotient(Nx, Ny, h, g_x, g_y, g_1, 0.37)[0]
        / _twist_quotient(Nx, Ny, h, g_x, g_y, g_1, 0.37)[1]
        for g_x, g_y, g_1 in [
            (1.0, 1.0, 0.3), (1.0, 0.073, 0.02), (3.1, 0.5, -0.4), (1.0, 1.0, 0.0),
        ]
    ]
    spread = (max(quotients) - min(quotients)) / abs(quotients[0])
    assert spread < 1e-11, f"R(xy) is not blind to g_x/g_y/g_1: relative spread {spread:.2e}"


def test_twist_numerator_is_the_closed_form_and_exact_without_cancellation():
    """``(xy)ᵀK(xy) = 4 g_xy ab`` — on any grid, with no discretization error.

    ``Dxy(xy) = 1`` on every cell, so the twist term sums ``4 g_xy h²`` over exactly ``Nx·Ny``
    cells. "Exact" means *in exact arithmetic*: each cell contributes a product of ``h²`` with
    ``1/h⁴``, so the sum carries ordinary roundoff either way — measured 2.5e-14 relative with the
    other three terms switched off and ~1e-13 with them on (where they must also cancel). The bar
    is set at the measured magnitude rather than at 1e-16, which this could never reach.
    """
    for Nx, Ny, h in [(16, 12, 0.05), (13, 9, 0.62 / 13), (24, 24, 1.0 / 24)]:
        a, b = Nx * h, Ny * h
        g_xy = 0.37
        exact = 4.0 * g_xy * a * b
        clean, _ = _twist_quotient(Nx, Ny, h, 0.0, 0.0, 0.0, g_xy)
        dirty, _ = _twist_quotient(Nx, Ny, h, 1.0, 1.0, 0.3, g_xy)
        assert abs(clean - exact) < 1e-13 * exact, f"twist numerator {clean} != {exact}"
        assert abs(dirty - exact) < 1e-11 * exact, f"with bending on: {dirty} vs {exact}"
        assert abs(clean - exact) < abs(dirty - exact), (
            "with the bending terms off there is nothing to cancel, so the clean numerator must be "
            f"the closer of the two: {clean!r} vs {dirty!r}"
        )


def test_twist_quotient_converges_to_the_continuum_bound_at_h2():
    """``R(xy) -> 576 D_xy/(rho_s a²b²)``, second order — and the error lives in the **mass**.

    The numerator is exact (previous test), so every bit of the ``O(h²)`` is the trapezoidal
    ``(xy)ᵀW(xy) -> a³b³/144``. Measured orders 1.95, 1.99, 2.00.
    """
    g_xy = 0.37
    errs = []
    for N in (8, 16, 32, 64):
        h = 1.0 / N
        num, den = _twist_quotient(N, N, h, 1.0, 1.0, 0.3, g_xy)
        errs.append(abs(num / den - 576.0 * g_xy))
    orders = np.log2(np.array(errs[:-1]) / np.array(errs[1:])) - np.log2(576.0 * g_xy) * 0.0
    assert np.all(np.array(errs[1:]) < np.array(errs[:-1])), f"not converging: {errs}"
    assert np.min(orders) > 1.8, f"twist-quotient convergence orders {orders} (want ~2)"


def test_fundamental_is_below_the_twist_bound_and_the_bound_is_informative():
    """Rayleigh: the first *elastic* frequency sits below ``24 sqrt(D_xy/rho_s)/(2 pi ab)``.

    Two arms, because a bound is only worth asserting if it is tight enough to catch anything: for
    the isotropic square the margin is ~5% (Narita's tabulated 13.468 against the bound's 14.199),
    and for spruce — whose fundamental *is* essentially the twist mode — it is tighter still. The
    weakness is stated rather than hidden: this is **one-sided**, so a uniformly too-soft operator
    passes it. The two-sided guard is the free-beam reduction below.
    """
    for tag, grain, lam_hint in [
        ("isotropic", dict(grain_coupling=0.3, grain_torsion=0.35), 13.5),
        ("spruce", spruce_free_grain(), 6.0),
    ]:
        p = make_orthotropic_free_plate(N=32, mu=0.5, **grain)
        bound = modal.free_plate_twist_bound(p.kappa, p.Lx, p.Ly, p.grain_torsion)
        f1 = free_plate_low_eigenfrequencies(p, 1, lam1_hint=lam_hint)[0]
        margin = (bound - f1) / bound
        assert f1 < bound, f"{tag}: fundamental {f1:.3f} Hz above the Rayleigh bound {bound:.3f}"
        assert margin < 0.15, f"{tag}: bound is loose ({margin:.1%}) — is the operator too soft?"


# =====================================================================================
# Detector 2 (grain_x / grain_y): the exact reduction to the shipped 1-D free beam
# =====================================================================================


def _beam_pencil(N, h):
    S, M = free_beam_stiffness(N, h)
    vals, vecs = eigh(S.toarray(), np.diag(M.diagonal()))
    return vals, vecs


@pytest.mark.parametrize("axis", ["x", "y"])
def test_transverse_independent_spectrum_is_the_free_beam_exactly(axis):
    """At **zero coupling**, a field constant along one axis is an exact eigenvector, and its
    eigenvalue is the shipped free beam's scaled by that axis's stiffness ratio.

    With ``w = 1_y ⊗ v``: ``C2y w = 0`` and ``Dxy w = 0`` identically, so
    ``K w = g_x kron(m_y, S v)`` and ``W w = kron(m_y, M v)`` — the trapezoidal ``y`` weights factor
    out of *both* sides and the generalized problem collapses to the 1-D ``S v = mu M v``. This is
    the batch's only **two-sided**
    external anchor, and it borrows the free beam's own closed-form ``cosh·cos`` oracle (model
    #5b-pre) rather than inventing one.
    """
    Nx, Ny, h = 20, 14, 0.05
    g_along, g_across = (2.3, 0.6) if axis == "x" else (0.6, 2.3)
    K, W, _ = free_plate_stiffness(
        Nx, Ny, h, 0.3, grain_x=g_along, grain_y=g_across, grain_coupling=0.0, grain_torsion=0.25
    )
    n_line = Nx if axis == "x" else Ny
    n_other = Ny if axis == "x" else Nx
    vals_b, vecs_b = _beam_pencil(n_line, h)
    for j in (2, 3, 4):  # skip the beam's own 2 rigid-body modes
        v = vecs_b[:, j]
        w = np.kron(np.ones(n_other + 1), v) if axis == "x" else np.kron(v, np.ones(n_other + 1))
        mu_pred = (g_along if axis == "x" else g_across) * vals_b[j]
        target = mu_pred * (W.diagonal() * w)
        resid = np.linalg.norm(K @ w - target) / np.linalg.norm(target)
        assert resid < 1e-10, f"{axis}-independent beam mode {j - 1}: residual {resid:.2e}"


def test_coupling_breaks_the_beam_reduction_without_changing_its_energy():
    """With a coupling rigidity the beam-like field stops being an eigenvector while its Rayleigh
    quotient stays **exactly** the beam's. That is anticlastic curvature, and it is physics.

    ``cross`` annihilates the field but ``cross.T`` does not — it survives on the two free ``y``
    edges, where the natural condition ``M_y = -(D_1 w_xx + D_y w_yy) = 0`` is *not* satisfied by a
    cylindrically-bent strip. So the mode acquires a different **shape** at the same energy, and a
    plate is not a wide beam. Anyone tempted to "fix" the residual below has removed the physics;
    the plan's Q7 records that the resulting eigenvalue shift is downward for a strip and upward for
    a square, which is why the shift itself is not asserted as an oracle.
    """
    Nx, Ny, h = 20, 14, 0.05
    vals_b, vecs_b = _beam_pencil(Nx, h)
    v = vecs_b[:, 2]
    w = np.kron(np.ones(Ny + 1), v)
    residuals = []
    for g_1 in (0.0, 0.1, 0.3):
        K, W, _ = free_plate_stiffness(
            Nx, Ny, h, 0.3, grain_x=2.3, grain_y=0.6, grain_coupling=g_1, grain_torsion=0.25
        )
        mu_pred = 2.3 * vals_b[2]
        Ww = W.diagonal() * w
        residuals.append(np.linalg.norm(K @ w - mu_pred * Ww) / np.linalg.norm(mu_pred * Ww))
        rayleigh = float(w @ (K @ w)) / float(w @ Ww)
        assert abs(rayleigh / mu_pred - 1.0) < 1e-12, (
            f"the coupling term changed the beam-like field's ENERGY at g_1={g_1}: "
            f"{rayleigh / mu_pred:.12f} (it must only change the shape)"
        )
    assert residuals[0] < 1e-10 < residuals[1] < residuals[2], (
        f"the eigenvector residual must be zero at g_1=0 and grow with coupling: {residuals}"
    )


# =====================================================================================
# Detector 3 (grain_coupling): the (x^2, y^2) probe -- the only one that sees D_1
# =====================================================================================


def test_coupling_probe_hits_its_exact_discrete_value():
    """``P(x², y²) = 4 g_1 h² (Nx-1)(Ny-1)`` exactly, and ``-> 4 D_1 ab`` at first order.

    Every other term of the form dies on this pair, and the collocated second differences return
    ``2`` at each interior node and ``0`` on the free edges — where the *natural* boundary condition
    puts them — so the discrete form is the continuum one short of exactly one boundary strip. Built
    with the other three constants at zero: there is then nothing to cancel, and the closed form is
    hit at machine precision rather than at 1e-12.
    """
    g_1 = 0.153
    cont_errs = []
    for Nx, Ny, h in [(8, 8, 1.0 / 8), (32, 32, 1.0 / 32), (13, 9, 0.62 / 13), (20, 14, 0.05)]:
        K, _, _ = free_plate_stiffness(
            Nx, Ny, h, 0.3, grain_x=0.0, grain_y=0.0, grain_coupling=g_1, grain_torsion=0.0
        )
        X, Y = _centred(Nx, Ny, h)
        val = float((X**2).ravel() @ (K @ (Y**2).ravel()))
        exact = modal.free_plate_coupling_form(g_1, h, Nx, Ny)
        assert abs(val - exact) < 1e-13 * abs(exact), f"{Nx}x{Ny}: {val!r} != {exact!r}"
        cont = 4.0 * g_1 * (Nx * h) * (Ny * h)
        cont_errs.append(abs(exact - cont) / cont)
    # ... and the continuum statement is the O(h) one: the missing strip is a fraction h/a of the
    # area, which is itself an exact expression.
    for (Nx, Ny), err in zip([(8, 8), (32, 32)], cont_errs[:2], strict=False):
        predicted = 1.0 - (1.0 - 1.0 / Nx) * (1.0 - 1.0 / Ny)
        assert abs(err - predicted) < 1e-12, f"continuum gap {err} != predicted {predicted}"


def test_only_the_coupling_probe_sees_the_coupling_rigidity():
    """The blind-spot claim, made a test: change ``g_1`` alone and the *other two* probes do not
    move, while this one moves proportionally. It is why the least glamorous detector is
    load-bearing (plan §7)."""
    Nx, Ny, h = 16, 12, 0.05
    twist = [_twist_quotient(Nx, Ny, h, 1.0, 0.5, g_1, 0.3)[0] for g_1 in (0.0, 0.2)]
    assert abs(twist[1] - twist[0]) < 1e-11 * abs(twist[0]), "the twist probe saw g_1"
    vals_b, vecs_b = _beam_pencil(Nx, h)
    w = np.kron(np.ones(Ny + 1), vecs_b[:, 2])
    energies = []
    for g_1 in (0.0, 0.2):
        K, W, _ = free_plate_stiffness(
            Nx, Ny, h, 0.3, grain_x=1.0, grain_y=0.5, grain_coupling=g_1, grain_torsion=0.3
        )
        energies.append(float(w @ (K @ w)) / float(w @ (W.diagonal() * w)))
    assert abs(energies[1] - energies[0]) < 1e-12 * energies[0], "the beam probe saw g_1"
    probes = [modal.free_plate_coupling_form(g_1, h, Nx, Ny) for g_1 in (0.1, 0.2)]
    assert abs(probes[1] / probes[0] - 2.0) < 1e-12, "the coupling probe is not linear in g_1"


# =====================================================================================
# The construction-time guard -- a DIFFERENT set from the supported branch's, not a tighter one
# =====================================================================================


def test_free_grain_admissibility_is_rejected_at_construction():
    for g_1 in (0.71, -0.71, 1.5):  # |g_1| >= sqrt(1.0 * 0.5) = 0.7071...
        with pytest.raises(ValueError, match="grain_coupling"):
            make_orthotropic_free_plate(N=8, grain_y=0.5, grain_coupling=g_1, grain_torsion=0.3)
    for g_t in (0.0, -0.1):
        with pytest.raises(ValueError, match="grain_torsion"):
            make_orthotropic_free_plate(N=8, grain_y=0.5, grain_coupling=0.1, grain_torsion=g_t)
    # ... and just inside it builds.
    p = make_orthotropic_free_plate(N=8, grain_y=0.5, grain_coupling=0.70, grain_torsion=0.3)
    assert p.grain_coupling == 0.70


def test_the_two_boundaries_admissible_sets_differ():
    """A split the *supported* branch is happy with can be inadmissible on the free branch.

    ``g_1 = 0.9``, ``g_xy = 0.05``, ``g_y = 0.5`` gives ``H = 1.0``, which the supported guard
    (``H > -sqrt(g_x g_y)``) accepts without comment — while the free branch rejects it, because
    ``|g_1| > sqrt(g_x g_y)``. The guards are not nested in either direction: the supported one is
    sharp for its operator, the free one is a *sufficient* pointwise condition, and they are
    conditions on different objects.
    """
    common = dict(Lx=1.0, Ly=1.0, kappa=KAPPA, rho=RHO_AREAL_DEFAULT, fs=40000.0, N=8)
    Plate(**common, boundary="supported", grain_y=0.5, grain_coupling=0.9, grain_torsion=0.05)
    with pytest.raises(ValueError, match="grain_coupling"):
        Plate(**common, boundary="free", grain_y=0.5, grain_coupling=0.9, grain_torsion=0.05)


def test_the_guard_is_conservative_and_the_claim_is_one_sided():
    """Inside the guard: semi-definite with exactly 3 zero modes, on several grids. **Outside** it,
    on a coarse grid, the discrete operator is *still* semi-definite — the pointwise bound is the
    sharp **continuum** condition, and the discrete margin (4–20%, shrinking with refinement) is
    measured here rather than claimed away. This is why the rejection above is one-sided by design.
    """
    for Nx, Ny, g_y, g_t in [(10, 8, 0.5, 0.2), (12, 12, 0.5, 0.02), (10, 8, 1.0, 0.35)]:
        h = 0.05
        edge = np.sqrt(g_y)
        K, W, _ = free_plate_stiffness(
            Nx, Ny, h, 0.3, grain_x=1.0, grain_y=g_y, grain_coupling=0.98 * edge, grain_torsion=g_t
        )
        vals = np.sort(_pencil(K, W)[0])
        scale = abs(vals).max()
        assert vals[0] > -1e-12 * scale, f"indefinite INSIDE the guard: {vals[0] / scale:.2e}"
        assert int(np.sum(np.abs(vals) < 1e-9 * scale)) == 3, f"not 3 zero modes: {vals[:5]}"
        # The measured conservatism: 2% past the bound is still fine on a grid this coarse. This arm
        # is a claim about THESE resolutions, not about the operator -- Q3c bisected the discrete
        # threshold at 1.198 on 7x5 but only 1.042 on 24x24, so refining the grids below would
        # legitimately break it. The grids are hardcoded for that reason.
        K2, W2, _ = free_plate_stiffness(
            Nx, Ny, h, 0.3, grain_x=1.0, grain_y=g_y, grain_coupling=1.02 * edge, grain_torsion=g_t
        )
        v2 = np.sort(_pencil(K2, W2)[0])
        assert v2[0] > -1e-12 * abs(v2).max(), (
            "the discrete threshold moved BELOW the pointwise bound — the guard would no longer be "
            f"sufficient: min mu/scale = {v2[0] / abs(v2).max():.2e}"
        )


def test_the_split_api_refuses_every_ambiguous_call():
    common = dict(Lx=1.0, Ly=1.0, kappa=KAPPA, rho=RHO_AREAL_DEFAULT, fs=40000.0, N=8,
                  boundary="free")
    with pytest.raises(ValueError, match="together"):
        Plate(**common, grain_coupling=0.3)  # half a split
    with pytest.raises(ValueError, match="together"):
        Plate(**common, grain_torsion=0.35)
    with pytest.raises(ValueError, match="either nu or"):
        Plate(**common, nu=0.3, grain_coupling=0.3, grain_torsion=0.35)  # nu would be ignored
    with pytest.raises(ValueError, match="contradicts the split"):
        Plate(**common, grain_cross=0.9, grain_coupling=0.3, grain_torsion=0.35)
    with pytest.raises(ValueError, match="separately"):
        Plate(**common, grain_y=0.5)  # a grain with no split
    # A consistent grain_cross alongside the split is accepted (1.0 == 0.3 + 2*0.35).
    p = Plate(**common, grain_cross=1.0, grain_coupling=0.3, grain_torsion=0.35)
    assert p.nu == 0.3, "the implied nu_yx = g_1/g_x must be exposed as .nu"


# =====================================================================================
# The physics: what a free edge can measure that a pinned one cannot
# =====================================================================================


def test_supported_is_blind_to_the_split():
    """Four different splits of the same ``H``: the supported operator is **bit-identical**.

    Not "agrees to 1e-16" — byte-equal, on a grid #5o chose *because* it distinguishes assemblies.
    This is the exact null control for the next test, and it is what makes the free plate's
    sensitivity a statement about the boundary rather than about arithmetic.
    """
    base = None
    for g_1 in (-0.1, 0.0, 0.05, 0.1):
        p = Plate(
            Lx=0.62, Ly=0.42, kappa=KAPPA, rho=RHO_AREAL_DEFAULT, fs=40000.0, N=13,
            boundary="supported", grain_y=0.073,
            grain_coupling=g_1, grain_torsion=0.5 * (0.153 - g_1),
        )
        data = p.B.tocsr()
        data.sort_indices()
        if base is None:
            base = data
        else:
            assert np.array_equal(base.data, data.data), f"supported B moved at g_1={g_1}"


def test_free_plate_is_not():
    """... while the free plate's fundamental spans **6.5x** across the same admissible splits.

    Fixed ``g_x = 1``, ``g_y = 0.073``, ``H = 0.153`` (spruce's own cross term) — only the *split*
    varies. The headline of the batch: whether a free plate is stiff is decided by how its cross
    term divides between coupling and torsion, which no supported plate can see. The sweep is
    deliberately reported as a range and not a single ratio, because it is non-monotone (it peaks at
    zero coupling) and its low end is a plate whose torsion has been driven almost to zero.
    """
    g_h = 0.153
    lams = []
    for g_1 in (-0.1, 0.0, 0.05, 0.1, 0.15):
        p = make_orthotropic_free_plate(
            N=24, mu=0.5, grain_y=0.073, grain_coupling=g_1, grain_torsion=0.5 * (g_h - g_1)
        )
        vals = np.sort(
            eigsh(p.K, k=6, M=p.W, sigma=-1e-4, which="LM", return_eigenvectors=False,
                  v0=arpack_v0(p.K))
        )
        lams.append(_lam(vals[3], p.Lx))
    assert max(lams) / min(lams) > 4.0, (
        f"the free plate barely noticed the split ({max(lams) / min(lams):.2f}x) — the batch's "
        f"premise is that it notices a lot: {[round(x, 3) for x in lams]}"
    )


def test_the_grain_reorders_the_free_plates_modes():
    """The free plate's fundamental **changes identity** with the grain — which never happens on the
    supported branch (#5o measured the ordering to be grain-invariant across the whole range between
    solid wood and isotropic material).

    The low spectrum is a race between a twist mode governed by ``D_xy`` alone and a cross-grain
    bending mode governed by ``D_y``. Below ``g_y/g_xy ~ 1.03`` the bender wins, above it the twist
    does. Real spruce sits at 1.154 — 12% above the crossing, i.e. on the twist side but close
    enough that the ordering is genuinely a material property and not a fact about free plates.
    """
    N, h, g_xy = 24, 1.0 / 24, 0.063
    kinds = {}
    for g_y in (0.04, 0.10):
        K, W, _ = free_plate_stiffness(
            N, N, h, 0.3, grain_x=1.0, grain_y=g_y, grain_coupling=0.0, grain_torsion=g_xy
        )
        vals, vecs = _pencil(K, W)
        X, Y = _centred(N, N, h)
        corr = abs(np.corrcoef((X * Y).ravel(), vecs[:, 3])[0, 1])
        kinds[g_y] = "twist" if corr > 0.9 else "bend"
    assert kinds[0.04] == "bend", f"below the crossing the cross-grain bender must lead: {kinds}"
    assert kinds[0.10] == "twist", f"above it the twist must lead: {kinds}"
    spec = grain_ratios_from_material(**SPRUCE)
    assert spec.grain_y / spec.grain_torsion > 1.0, "spruce should sit on the twist-first side"


def test_the_grain_is_worth_more_here_than_the_supported_branch_suggested():
    """#5o's surviving result was that a grain *detunes selectively*, 1.3%–29%. On a free edge the
    same material moves the fundamental by a factor, not a percentage: spruce's twist mode sits at
    roughly ``sqrt(g_xy)`` of the isotropic plate's, because the fundamental is governed by the
    torsional rigidity alone and spruce's is ~6% of its along-grain stiffness."""
    iso = make_orthotropic_free_plate(N=32, mu=0.5)
    spruce = make_orthotropic_free_plate(N=32, mu=0.5, **spruce_free_grain())
    f_iso = free_plate_low_eigenfrequencies(iso, 1, lam1_hint=13.5)[0]
    f_spr = free_plate_low_eigenfrequencies(spruce, 1, lam1_hint=6.0)[0]
    ratio = f_spr / f_iso
    predicted = np.sqrt(spruce.grain_torsion / iso.grain_torsion)
    assert 0.3 < ratio < 0.5, f"spruce/isotropic fundamental ratio {ratio:.3f}"
    assert abs(ratio / predicted - 1.0) < 0.10, (
        f"the drop should track sqrt(g_xy) to ~10% (the two plates' Rayleigh margins differ): "
        f"{ratio:.4f} vs {predicted:.4f}"
    )


# =====================================================================================
# Tier 1: the ledger. Nearly blind here -- and that is the point of saying so.
# =====================================================================================


def _pluck(p, amplitude=1e-3):
    a = p.Lx
    return raised_cosine_2d(p.X, p.Y, (0.4 * a, 0.55 * a), 0.25 * a, amplitude)


@pytest.mark.parametrize("mu", [0.5, 4.0])
def test_energy_conserved_with_a_grain(mu):
    """Unconditional conservation survives four constants. Note this test cannot *validate* them:
    any symmetric ``K`` conserves exactly, so a mis-set coefficient passes here and is caught only
    by the three probes above. It is here to catch the time-stepper, not the physics."""
    p = make_orthotropic_free_plate(N=24, mu=mu, **spruce_free_grain())
    p.set_state(_pluck(p))
    res = simulate(p, num_steps=int(0.5 * p.fs))
    assert res.energy_drift < DRIFT_TOL, f"drift {res.energy_drift:.2e} at mu={mu}"


def test_passivity_with_a_grain():
    p = make_orthotropic_free_plate(N=24, mu=1.0, sigma=8.0, **spruce_free_grain())
    p.set_state(_pluck(p))
    res = simulate(p, num_steps=int(0.3 * p.fs))
    d = np.diff(res.energy)
    assert np.all(d <= 1e-12 * res.energy[0]), f"energy increased: max step {d.max():.3e}"
    assert res.energy[-1] < res.energy[0]


def test_self_convergence_order_h2_with_a_grain():
    """Richardson on the low eigenvalues — no external data needed, and it is the only *two-sided*
    check on the spectrum as a whole (the twist bound is one-sided)."""
    mus = []
    for N in (20, 40, 80):
        p = make_orthotropic_free_plate(N=N, mu=0.5, **spruce_free_grain())
        f = free_plate_low_eigenfrequencies(p, 3, lam1_hint=6.0)
        mus.append((2.0 * np.pi * f / p.kappa) ** 2)
    mus = np.array(mus)
    d1, d2 = mus[0] - mus[1], mus[1] - mus[2]
    orders = np.log2(np.abs(d1) / np.abs(d2))
    assert np.all(np.abs(d2) < np.abs(d1)), f"not converging: {mus}"
    assert np.min(orders) > 1.6, f"convergence orders {orders} (want ~2)"


# =====================================================================================
# The material chain, and the shipped nonlinear plate that shares this operator
# =====================================================================================


def test_material_chain_returns_a_consistent_split():
    """``grain_coupling + 2*grain_torsion == grain_cross``, and isotropic material splits at
    ``(nu, (1-nu)/2)`` — the isotropic free plate's own coefficients, which is what makes the whole
    thing one code path."""
    spec = grain_ratios_from_material(**SPRUCE)
    assert abs(spec.grain_coupling + 2.0 * spec.grain_torsion - spec.grain_cross) < 1e-15
    # H = D_1 + 2 D_xy, so the torsional SHARE carries the factor of two: 2*0.0630/0.1529 = 82%.
    torsional_share = 2.0 * spec.grain_torsion / spec.grain_cross
    assert torsional_share > 0.8, (
        f"82% of spruce's cross term is torsional ({torsional_share:.1%} here) — the lopsidedness "
        "is why a free edge, whose fundamental is pure torsion, is the boundary that measures it"
    )
    E, nu, t, rho = 1.1e10, 0.3, 3.0e-3, 420.0
    iso = grain_ratios_from_material(
        E_x=E, E_y=E, nu_xy=nu, G_xy=E / (2.0 * (1.0 + nu)), thickness=t, rho=rho
    )
    assert abs(iso.grain_coupling - nu) < 1e-15, f"isotropic D_1/D != nu: {iso.grain_coupling}"
    assert abs(iso.grain_torsion - 0.5 * (1.0 - nu)) < 1e-15, "D_xy/D != (1-nu)/2"


def test_von_karman_plate_is_untouched():
    """The nonlinear plate shares this operator and stays **isotropic** (orthotropic von Kármán
    needs a four-constant in-plane compliance tensor and has no closed-form oracle — refused, and
    this batch is its prerequisite). Its bending operator must be byte-identical to the default."""
    from physsynth.core.plate import VKPlate

    vk = VKPlate(
        Lx=1.0, Ly=1.0, E=2.0e11, e=1.0e-3, nu=0.3, rho=RHO_AREAL_DEFAULT, fs=200000.0, N=12,
        boundary="free",
    )
    K, W, _ = free_plate_stiffness(12, 12, vk.h, 0.3)
    a, b = K.tocsr(), vk.K.tocsr()
    a.sort_indices()
    b.sort_indices()
    assert np.array_equal(a.data, b.data), "VKPlate's bending operator moved"
    assert np.array_equal(W.diagonal(), vk.W.diagonal())


def test_an_implied_poisson_ratio_above_one_half_is_admissible():
    """A grained free plate whose implied ``nu_yx = D_1/D_x`` exceeds 1/2 must **build**.

    The isotropic ``(-1, 1/2)`` range is 3-D thermodynamic admissibility for one Poisson ratio; an
    orthotropic sheet is bounded instead by ``nu_xy nu_yx < 1``, i.e. exactly this branch's guard
    ``grain_coupling² < grain_x grain_y``. Applying the isotropic range to the *implied* ratio
    rejected a valid material (``g_1 = 0.70`` at ``g_y = 0.5``) until the operator was taught to
    validate ``nu`` only where it actually supplies a missing half of the split.
    """
    p = make_orthotropic_free_plate(N=8, grain_y=0.5, grain_coupling=0.70, grain_torsion=0.3)
    assert p.nu == 0.70 > 0.5, "the implied nu_yx is exposed unclamped"
    K, W, _ = free_plate_stiffness(
        8, 8, p.h, p.nu, grain_x=1.0, grain_y=0.5, grain_coupling=0.70, grain_torsion=0.3
    )
    assert np.array_equal(K.tocsr().data, p.K.tocsr().data)
    # ... but a plain isotropic call with that nu is still refused, since there it IS used.
    with pytest.raises(ValueError, match="Poisson"):
        free_plate_stiffness(8, 8, 0.1, 0.70)
