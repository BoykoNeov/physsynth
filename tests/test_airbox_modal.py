"""The 3-D air box: room modes — the money oracle, in two tiers.

A rigid rectangular room is the rare case where the scheme's own eigenmodes are known in **closed
form on the grid**, not merely in the continuum. The tensor cosine
``cos(l pi i/Nx) cos(m pi j/Ny) cos(n pi k/Nz)`` is an *exact* eigenvector of the discrete Neumann
Laplacian **including at the h/2 wall nodes** — a grid-aligned rectangle staircases nothing — with
eigenvalue ``-mu^2``, ``mu^2 = (4/h^2) sum_d sin^2(l_d pi / (2 N_d))``, and the leapfrog then makes
it oscillate at exactly ``omega_d = (2/k) arcsin(c0 k mu / 2)``.

That puts this a **tier above** the membrane's Bessel test, which could only ever be a
convergence-rate check because a staircased circle is not the circle. Here the assertion is machine
precision, and it splits in two:

- **(a) spatial** — the eigen*value*: one application of the scheme, over **every** ``l`` from 0 to
  ``N`` on each axis, including DC and Nyquist;
- **(b) temporal** — the eigen*frequency*: the field must equal ``cos(omega_d n k) * mode`` at
  *every* step, compared against the predicted amplitude, not merely "still proportional to the
  mode shape". A shape-only test passes even when ``omega_d`` is wrong, and ``omega_d`` is the whole
  point.

Below that sits the ordinary tier: the discrete frequency converges at **order 2** to the textbook
rigid-room formula ``f_lmn = (c0/2) sqrt((l/Lx)^2 + (m/Ly)^2 + (n/Lz)^2)``. The gap between the two
*is* the scheme's numerical dispersion — which in 3-D never vanishes at any ``lambda``, unlike the
1-D string's ``lambda = 1``.
"""

import numpy as np
import pytest
from helpers import airbox_noise, make_airbox

EXACT_TOL = 1e-12  # the discrete mode is exact, not approximate: hold it to the energy bar


def _window_peaks(box, windows=4, steps=2000):
    """Peak ``|p|`` within each of several consecutive time windows, relative to the start.

    The right shape for a boundedness claim: a *running* max only ever rises, so it cannot tell a
    bounded oscillation from a slowly growing one. Windowed peaks can — flat means bounded, and a
    ``1, 2, 3, 4`` progression means linear growth.
    """
    start = float(np.max(np.abs(box.p)))
    peaks = []
    for _ in range(windows):
        peak = 0.0
        for _ in range(steps):
            box.step()
            peak = max(peak, float(np.max(np.abs(box.p))))
        peaks.append(peak / start)
    return peaks


def _mode_error(box, lmn, amp):
    """Max deviation of the field from ``amp * mode_shape``, relative to the mode's own scale."""
    mode = box.mode_shape(*lmn)
    return float(np.max(np.abs(box.p - amp * mode)) / np.max(np.abs(mode)))


# -- (a) Spatial: the eigenvalue, at every index, in one application --------------------------
#    One step from the exact initialisation is exactly one application of the scheme's Laplacian:
#    p^1 = cos(omega_d k) p^0 holds iff mu^2 is the true eigenvalue of the discrete operator. This
#    is the cheap, exhaustive form of the assertion -- the whole sweep costs one step per mode.
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_every_axial_mode_is_an_exact_eigenvector(axis):
    box = make_airbox()
    worst = 0.0
    for q in range(box.N[axis] + 1):
        lmn = [0, 0, 0]
        lmn[axis] = q
        f_d = box.set_mode(*lmn)
        assert _mode_error(box, lmn, 1.0) == 0.0, "set_mode did not start on the mode"
        box.step()
        worst = max(worst, _mode_error(box, lmn, np.cos(2.0 * np.pi * f_d * box.k)))
    assert worst < EXACT_TOL, f"axis {axis}: worst eigenvector residual {worst:.2e}"


def test_every_oblique_mode_is_an_exact_eigenvector():
    """The full 3-D index grid, corners included: ``(0,0,0)`` is the DC nullspace direction (a
    uniform pressure, which a rigid room genuinely preserves) and ``(Nx,Ny,Nz)`` is the corner
    Nyquist mode, whose arcsin argument is ``lambda sqrt(3)`` — the value the CFL exists to cap."""
    box = make_airbox()
    Nx, Ny, Nz = box.N
    worst = 0.0
    for lmn in [(l, m, n) for l in (0, 1, Nx) for m in (0, 2, Ny) for n in (0, 3, Nz)]:
        f_d = box.set_mode(*lmn)
        box.step()
        worst = max(worst, _mode_error(box, lmn, np.cos(2.0 * np.pi * f_d * box.k)))
    assert worst < EXACT_TOL


def test_dc_mode_is_stationary():
    """``(0,0,0)`` has ``mu = 0``, so a uniform pressure in a sealed rigid room never moves. It is
    the discrete statement that the scheme conserves mass."""
    box = make_airbox()
    assert box.mode_frequency(0, 0, 0) == 0.0
    box.set_mode(0, 0, 0)
    for _ in range(200):
        box.step()
    assert _mode_error(box, (0, 0, 0), 1.0) < EXACT_TOL


# -- (b) Temporal: the frequency, tracked step by step ----------------------------------------
@pytest.mark.parametrize("lmn", [(1, 0, 0), (0, 1, 0), (0, 0, 1), (2, 1, 1), (3, 2, 1)])
def test_mode_tracks_its_predicted_amplitude(lmn):
    box = make_airbox()
    f_d = box.set_mode(*lmn)
    assert f_d > 0.0
    worst = 0.0
    for t in range(1, 501):
        box.step()
        worst = max(worst, _mode_error(box, lmn, np.cos(2.0 * np.pi * f_d * t * box.k)))
    assert worst < EXACT_TOL, f"mode {lmn} drifted from cos(omega_d t) by {worst:.2e}"


def test_two_modes_superpose_without_talking():
    """A linear room: start in a sum of two modes and each keeps its own frequency. Projecting the
    field back onto each shape must reproduce two independent cosines — a coupling bug (a weight
    applied on the wrong axis, say) shows up as one mode bleeding into the other long before it
    shows up in the energy."""
    box = make_airbox()
    a, b = (1, 0, 0), (0, 1, 2)
    fa, fb = box.mode_frequency(*a), box.mode_frequency(*b)
    ma, mb = box.mode_shape(*a), box.mode_shape(*b)
    s = box.k / (2.0 * box.rho0 * box.h)
    p0 = ma + mb
    box.set_state(p0, tuple(s * np.diff(p0, axis=d) for d in range(3)))
    # The tensor cosines are orthogonal under the node weights W (a discrete Neumann basis).
    W = box._W
    na, nb = float(np.sum(W * ma * ma)), float(np.sum(W * mb * mb))
    worst = 0.0
    for t in range(1, 301):
        box.step()
        ca = float(np.sum(W * box.p * ma)) / na
        cb = float(np.sum(W * box.p * mb)) / nb
        worst = max(
            worst,
            abs(ca - np.cos(2.0 * np.pi * fa * t * box.k)),
            abs(cb - np.cos(2.0 * np.pi * fb * t * box.k)),
        )
    assert worst < EXACT_TOL


# -- The continuum tier: second-order convergence to the textbook room formula -----------------
@pytest.mark.parametrize("lmn", [(1, 0, 0), (1, 1, 0), (2, 1, 1)])
def test_converges_to_the_continuum_room_modes(lmn):
    """``f_lmn = (c0/2) sqrt(sum (l_d/L_d)^2)`` recovered at order 2 in ``h``. This is a pure
    closed-form comparison (no stepping): tier (b) above has already proved the scheme *runs* at
    ``mode_frequency``, so what is left to show is that ``mode_frequency`` is the room's."""
    L = (0.96, 0.72, 0.6)
    errs = []
    for N in (8, 16, 32, 64):
        box = make_airbox(L=L, h=L[0] / N)
        f_d = box.mode_frequency(*lmn)
        f_c = box.continuum_mode_frequency(*lmn)
        errs.append(abs(f_d - f_c) / f_c)
    rates = [round(np.log2(errs[i] / errs[i + 1]), 3) for i in range(len(errs) - 1)]
    assert min(rates) > 1.9, f"rates {rates} (errs {errs})"
    assert max(rates) < 2.1, f"rates {rates} (errs {errs})"


def test_discrete_frequency_is_below_the_continuum_one():
    """Yee dispersion is *negative* on this stencil — the grid always runs a mode slightly flat.
    Worth pinning, because a sign slip in the arcsin would still converge at order 2."""
    box = make_airbox()
    for lmn in [(1, 0, 0), (2, 1, 1), (3, 2, 1)]:
        assert box.mode_frequency(*lmn) < box.continuum_mode_frequency(*lmn)


def test_axial_modes_are_dispersive_at_every_courant_number():
    """No ``lambda`` makes an **axis-aligned** mode exact — the membrane batch's 2-D lesson, one
    dimension worse. Note the error does *not* simply worsen toward the ceiling (it improves
    slightly, as the spatial and temporal errors partly cancel), so "smaller lambda is more
    accurate" is not the rule it is in 1-D. It never reaches zero, and that is the claim."""
    errs = {
        cfl: abs(
            make_airbox(cfl=cfl).mode_frequency(2, 0, 0)
            / make_airbox(cfl=cfl).continuum_mode_frequency(2, 0, 0)
            - 1.0
        )
        for cfl in (0.1, 0.3, 0.5, 0.7, 0.9, 1.0)
    }
    assert min(errs.values()) > 1e-3, f"an axial mode came out dispersionless: {errs}"


# -- The CFL ceiling: the reward and the price, at the same lambda -------------------------------
@pytest.mark.parametrize("q", [1, 2, 3, 8])
def test_diagonal_modes_are_exact_at_the_ceiling(q):
    """**The reward.** At ``lambda = 1/sqrt(3)`` exactly, a mode along the grid diagonal
    (``l/Nx = m/Ny = n/Nz``, here a cube with ``l = m = n``) runs at the *exact continuum*
    frequency: the arcsin and the sine cancel identically. It is the only exactness available
    anywhere in 3-D, and it is why anyone would want to sit on the ceiling at all."""
    box = make_airbox(L=(0.8, 0.8, 0.8), cfl=1.0)
    err = abs(box.mode_frequency(q, q, q) / box.continuum_mode_frequency(q, q, q) - 1.0)
    assert err < 1e-14, f"diagonal mode ({q},{q},{q}) off by {err:.2e}"
    axial = abs(box.mode_frequency(q, 0, 0) / box.continuum_mode_frequency(q, 0, 0) - 1.0)
    assert axial > 1e-3, "the axial mode should NOT be exact — only the diagonal is"


def test_the_diagonal_is_the_grid_diagonal_not_l_equals_m():
    """On a non-cube the exact direction is ``l/Nx = m/Ny = n/Nz``, so the room ``(0.9, 0.7, 0.6)``
    at ``h = 0.1`` has it at the **corner** mode ``(9, 7, 6)`` — and ``(1, 1, 1)`` is not it."""
    box = make_airbox(cfl=1.0)
    assert box.mode_frequency(*box.N) == pytest.approx(box.continuum_mode_frequency(*box.N))
    assert abs(box.mode_frequency(1, 1, 1) / box.continuum_mode_frequency(1, 1, 1) - 1.0) > 1e-4


def test_ceiling_is_marginally_stable_and_energy_does_not_notice():
    """**The price.** At ``lambda = 1/sqrt(3)`` the corner mode is defective, so broadband content
    grows **linearly** — and the energy identity stays flat right through it, because the discrete
    energy is only positive *semi*-definite there. A flat energy is not a stability certificate at
    the ceiling; this is the one place in the repo where that holds, so it is pinned rather than
    hidden. Strictly below the ceiling the growth is gone."""
    box = airbox_noise(make_airbox(cfl=1.0))
    e0 = box.energy()
    peaks = _window_peaks(box)
    # Linear (secular) growth, not exponential: the peak over window i is proportional to i.
    ratios = [p / peaks[0] for p in peaks]
    assert np.allclose(ratios, [1.0, 2.0, 3.0, 4.0], rtol=0.08), f"not linear: {ratios}"
    assert abs(box.energy() - e0) / abs(e0) < 1e-9, "the energy identity should survive it"


def test_bounded_just_below_the_ceiling():
    """The safe operating point: at 0.999 of the ceiling the defect is gone and the same broadband
    run is flat across every window — while *at* the ceiling it is 4x higher by the last one."""
    peaks = _window_peaks(airbox_noise(make_airbox(cfl=0.999)))
    assert peaks[-1] / peaks[0] < 1.4, f"windowed peaks trend upward: {peaks}"
