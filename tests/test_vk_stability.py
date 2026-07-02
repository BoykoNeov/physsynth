"""Von Kármán nonlinear plate — convergence order, derived materials, param validation (Part 3).

Richardson **self**-convergence replaces the missing analytic spectrum as the order check: with no
closed form, refine ``h → h/2 → h/4`` and confirm the (nonlinear) solution differences shrink by 4×,
i.e. O(h²) — the same discipline used for the free plate. The scheme composes an O(h²) bracket
(Part 1) and an O(h²) clamped Airy solve (Part 2) with the O(k²) θ-scheme, so second order in space
is the claim. Plus the material-derivation and constructor-validation smoke tests.
"""

import numpy as np
import pytest

from physsynth.core.plate import VKPlate

MAT = dict(E=2.0e11, e=1.0e-3, nu=0.3, rho=7800.0)


# -- #6 Richardson self-convergence: O(h²) ---------------------------------------------------------
def test_richardson_second_order():
    """Center displacement after a fixed time self-converges at O(h²) under ``h → h/2``.

    A **smooth** low-mode initial condition (not a narrow strike) keeps the run in the asymptotic
    regime: the nonlinear source ``l(w, w)`` doubles the wavenumber content, so a coarse grid /
    under-resolved strike sits pre-asymptotic (ratio ~3), while grids 24/48/96 on a smooth IC give
    the clean 4×.
    """
    Lx = Ly = 0.4
    fs = 96000.0

    def center_after(N, nsteps=80):
        vk = VKPlate(Lx=Lx, Ly=Ly, fs=fs, N=N, **MAT)
        x, y = vk.X, vk.Y
        ic = np.sin(np.pi * x / Lx) * np.sin(np.pi * y / vk.Ly) + 0.4 * np.sin(
            2 * np.pi * x / Lx
        ) * np.sin(np.pi * y / vk.Ly)
        vk.set_state(MAT["e"] * ic)
        pk = vk.pickup_index_at(0.5 * Lx, 0.5 * vk.Ly)
        for _ in range(nsteps):
            vk.step()
        return vk.displacement_at(pk)

    w = {N: center_after(N) for N in (24, 48, 96)}
    e1 = abs(w[24] - w[48])
    e2 = abs(w[48] - w[96])
    ratio = e1 / e2
    assert ratio > 3.4  # → 4 for O(h²) (measured ≈ 4.4)


# -- derived material quantities (the (E, e, ν, ρ) surface) ----------------------------------------
def test_derived_material_quantities():
    vk = VKPlate(Lx=0.4, Ly=0.4, fs=48000.0, N=12, **MAT)
    E, e, nu, rho = MAT["E"], MAT["e"], MAT["nu"], MAT["rho"]
    assert vk.rho_s == pytest.approx(rho * e)
    assert vk.D == pytest.approx(E * e**3 / (12.0 * (1.0 - nu**2)))
    assert vk.kappa == pytest.approx(np.sqrt(vk.D / vk.rho_s))
    assert vk.Y_mem == pytest.approx(E * e)  # membrane coefficient E e


def test_ly_snapped_to_square_cells():
    """``Ly`` is snapped to an integer number of ``h = Lx/N`` cells (square cells)."""
    vk = VKPlate(Lx=1.0, Ly=0.83, fs=48000.0, N=10, **MAT)
    assert vk.h == pytest.approx(0.1)
    assert vk.Ly == pytest.approx(vk.Ny * vk.h)
    assert abs(vk.Ly - 0.83) <= vk.h  # within one cell of the request


# -- constructor validation ------------------------------------------------------------------------
@pytest.mark.parametrize(
    "over",
    [
        dict(Lx=0.0),
        dict(Ly=-1.0),
        dict(fs=0.0),
        dict(E=0.0),
        dict(e=-1e-3),
        dict(rho=0.0),
        dict(N=1),
        dict(sigma=-1.0),
        dict(theta=0.0),
        dict(theta=1.5),
        dict(nu=0.5),
        dict(nu=-1.0),
        dict(couple_tol=0.0),
        dict(couple_max_iter=0),
    ],
)
def test_constructor_rejects_bad_params(over):
    kw = dict(Lx=0.4, Ly=0.4, fs=48000.0, N=12, **MAT)
    kw.update(over)
    with pytest.raises(ValueError):
        VKPlate(**kw)
