"""Orthotropic plate (model #5 with a **grain**): the plate stops being the same in every direction.

Every plate shipped before this module is made of a material equally stiff along every axis. Real
soundboards are wood, and wood is roughly ten times stiffer along the grain than across it. The
supported plate now takes three bending-stiffness ratios instead of one, replacing ``B = L²`` with

    B = g_x (δ_xx)² + 2 g_h (δ_xx δ_yy) + g_y (δ_yy)²

The reason this model earns a batch rather than the more exciting neighbours (a curved bell, a
mass-spring network) is that it satisfies the project's acceptance contract on both counts at once:
switching the grain off reproduces a shipped model, **and** the grained model has a *closed-form*
oracle rather than a convergence rate. ``sin(mπx/Lx) sin(nπy/Ly)`` is already an exact discrete
eigenvector of ``δ_xx`` and ``δ_yy`` **separately**, so it survives orthotropy exactly and carries
its analytic frequency with it. See ``docs/dev/orthotropic-plate-plan.md``.

What this module asserts, and in what order of trust:

- the assembled operator's eigenvalue on the analytic sine is the closed form, to machine precision
  (the money test — there is no ``kappa = 0`` anchor for a plate, so operator correctness has to be
  proved directly, exactly as model #5 did);
- the isotropic default provably stays on the **untouched** ``L @ L`` line, pinned on a grid where
  the two assemblies are known to differ in the last bit;
- energy is conserved and passivity holds — and, importantly, that this proves *less* here than it
  looks like it proves (:func:`test_the_energy_ledger_cannot_see_a_wrongly_wired_grain`);
- the guard on the cross term, which is what replaces the isotropic operator's free definiteness.

Then the findings: what the grain actually does, and two things it does **not** do.
"""

import numpy as np
import pytest
from helpers import (
    KAPPA_PLATE_DEFAULT,
    RHO_AREAL_DEFAULT,
    SPRUCE,
    convergence_orders,
    make_orthotropic_plate,
    make_plate,
    orthotropic_mode_freqs,
)

from physsynth.analysis import modal, spectrum
from physsynth.core.engine import simulate
from physsynth.core.exciter import raised_cosine_2d
from physsynth.core.operators2d import laplacian_from_mask, orthotropic_biharmonic, rectangle_mask
from physsynth.core.plate import Plate, grain_ratios_from_material

KAPPA = KAPPA_PLATE_DEFAULT
THETA = 0.28
DRIFT_TOL = 1e-10  # the project's acceptance bar, unchanged -- see CLAUDE.md

# A strongly anisotropic test material, deliberately NOT a real one: the core API takes ratios so
# that a test can dial combinations no sheet of wood has. The physical spruce case is SPRUCE.
G_STRONG = dict(grain_x=1.0, grain_cross=0.153, grain_y=0.0727)  # ~ the spruce ratios
G_WILD = dict(grain_x=11.0, grain_cross=2.5, grain_y=0.9)  # nothing is made of this


# Passivity is asserted against a roundoff bar relative to the initial energy, the same shape as
# `test_bore_radiation.py` and `test_airbox_port.py` -- NOT against a bare `<= 0.0`.
#
# This is not defensive slack, it is required, and the reason is the *first* step. Every test here
# plucks from rest, so at step 0 the velocity is exactly zero and the step dissipates essentially
# nothing: measured -2.70e-18 against -1.50e-10 for the very next step and ~1.09e-08 typical. The
# sign of a decrement four billion times smaller than its neighbours is decided by summation order
# alone, so a bare `<= 0.0` is a coin toss on any given BLAS -- it read negative locally and
# positive on CI, which is exactly how it failed there while passing here.
#
# The bar cannot hide a real failure: 1e-12 * e[0] is ~3.3e-17 here, still five orders of magnitude
# BELOW a typical genuine decrement, so any physically meaningful energy gain fails it loudly.
PASSIVITY_ROUNDOFF = 1e-12


def _assert_monotone(e, why="energy increased somewhere in a lossy run"):
    d = np.diff(e)
    bar = PASSIVITY_ROUNDOFF * e[0]
    assert np.all(d <= bar), f"{why}: max step {d.max():.3e} exceeds the roundoff bar {bar:.3e}"
    assert e[-1] < e[0], f"{why}: energy did not fall at all ({e[-1]:.6e} vs {e[0]:.6e})"


def _sine_field(plate, m, n):
    """``sin(mπx/Lx) sin(nπy/Ly)`` sampled on the live nodes, in the plate's own ordering."""
    jj, ii = np.nonzero(plate.mask)
    return np.sin(m * np.pi * ii * plate.h / plate.Lx) * np.sin(
        n * np.pi * jj * plate.h / plate.Ly
    )


def _axis_eigs(plate, m, n):
    lam_x = float(modal.dirichlet_axis_eigenvalue(m, plate.Lx, plate.h))
    lam_y = float(modal.dirichlet_axis_eigenvalue(n, plate.Ly, plate.h))
    return lam_x, lam_y


def _pluck(p, amplitude=1e-3):
    field = raised_cosine_2d(p.X, p.Y, (0.4 * p.Lx, 0.55 * p.Ly), 0.25 * min(p.Lx, p.Ly), amplitude)
    field[~p.mask] = 0.0
    return field


# =====================================================================================
# Tier 1 -- the contract
# =====================================================================================


@pytest.mark.parametrize("grain", [G_STRONG, G_WILD])
def test_operator_eigenvalue_is_the_closed_form(grain):
    """The money test. ``B v = (g_x λ_x² + 2 g_h λ_x λ_y + g_y λ_y²) v`` on the analytic sine.

    This is the orthotropic replacement for model #5's ``B``-eigenvalue-equals-``Λ²`` test, and it
    plays the same role: a plate has no ``kappa = 0`` reduction to anchor against (``kappa = 0``
    gives ``u_tt = 0``, degenerate), so the operator has to be proved directly against closed form.
    It is a *residual*, not an eigenvalue solve -- if the sine were only approximately an
    eigenvector the residual would expose it, whereas ``eigsh`` would happily return a nearby
    eigenvalue of some other vector.
    """
    p = make_orthotropic_plate(N=24, Lx=0.62, Ly=0.43, **grain)
    worst = 0.0
    for m, n in [(1, 1), (2, 1), (1, 2), (3, 2), (2, 3), (5, 4)]:
        v = _sine_field(p, m, n)
        lam_x, lam_y = _axis_eigs(p, m, n)
        q = (
            grain["grain_x"] * lam_x**2
            + 2.0 * grain["grain_cross"] * lam_x * lam_y
            + grain["grain_y"] * lam_y**2
        )
        worst = max(worst, np.linalg.norm(p.B @ v - q * v) / (q * np.linalg.norm(v)))
    assert worst < 1e-11, f"sine is not an exact eigenvector of the grained B: residual {worst:.2e}"


def test_the_isotropic_default_stays_on_the_untouched_squaring_path():
    """The default plate must be **byte-identical** to ``L @ L``, not merely close to it.

    The two assemblies agree only to rounding, and *how* closely is grid-dependent: on ``Lx = 1``
    grids they happen to come out bit-for-bit equal, while on others they differ in the last bit
    (measured 1.7e-16 to 2.4e-16 relative; see the plan §2 Q1). Because it is grid-dependent it
    cannot be relied on either way, which is exactly why :class:`Plate` keeps the isotropic default
    on the original line rather than routing everything through the general assembly — otherwise
    every shipped plate number would move in its last digit on *some* grids and not others.

    So the assertion is made on a grid **known to distinguish the two paths**. A grid where they
    coincide would pass this test while proving nothing.
    """
    Nx, Ny, Lx = 16, 16, 0.7
    h = Lx / Nx
    L, _ = laplacian_from_mask(rectangle_mask(Nx, Ny), h)
    squared = (L @ L).toarray()
    general, _ = orthotropic_biharmonic(Nx, Ny, h, 1.0, 1.0, 1.0)
    general = general.toarray()

    gap = np.abs(general - squared).max() / np.abs(squared).max()
    assert 0.0 < gap < 1e-15, (
        f"this grid was chosen because the two assemblies differ there; they now differ by "
        f"{gap:.2e}. If this is 0 the test has lost its teeth -- pick another grid."
    )

    p = Plate(Lx=Lx, Ly=Lx, kappa=KAPPA, rho=RHO_AREAL_DEFAULT, fs=40000.0, N=Nx, theta=THETA)
    assert p.grain_is_isotropic
    assert np.array_equal(p.B.toarray(), squared), "the default plate left the L @ L path"
    assert not np.array_equal(p.B.toarray(), general), (
        "the default plate is being built by the general orthotropic assembly"
    )


def test_uniform_grain_collapses_to_an_isotropic_plate():
    """``g = (r, r, r)`` is an isotropic plate of stiffness ``kappa sqrt(r)`` — through the *new*
    code path.

    The previous test pins that the default avoids the new assembly; this one pins that the new
    assembly is nonetheless right, by scaling all three ratios together. The twin has to be built
    at a **matched sample rate**, not merely a matched ``mu``: :func:`make_plate` solves ``fs`` from
    ``kappa``, so a twin with a larger ``kappa`` would silently get a different timestep and the
    theta-scheme frequencies would differ by ~3% — a discrepancy that looks like a modelling bug
    and is entirely a rig error.
    """
    r = 2.7
    a = make_orthotropic_plate(N=24, grain_x=r, grain_cross=r, grain_y=r)
    b = Plate(
        Lx=1.0,
        Ly=1.0,
        kappa=KAPPA * np.sqrt(r),
        rho=RHO_AREAL_DEFAULT,
        fs=a.fs,
        N=24,
        theta=a.theta,
    )
    assert a.k == b.k and a.h == b.h  # the rig error above, made impossible to make silently

    modes = [(1, 1), (2, 1), (1, 2), (3, 2)]
    lam = modal.rectangular_discrete_eigenvalues(b.h, 24, 24, modes)
    f_iso = modal.discrete_plate_eigenfrequency(lam, b.kappa, b.k, b.theta)
    f_ortho = orthotropic_mode_freqs(a, modes)
    rel = np.max(np.abs(f_ortho - f_iso) / f_iso)
    assert rel < 1e-14, f"uniform grain is not an isotropic plate: {rel:.2e}"

    # ... and it steps identically too, not just spectrally. Not bit-exact and not expected to be:
    # the two operators differ by the reassociation of §2 Q1, so the states separate by the last
    # bit per step and accumulate. 200 steps buys ~1.6e-12 relative, which is the rounding floor,
    # not a modelling gap -- the assertion is set an order above it rather than tuned onto it.
    field = _pluck(a)
    a.set_state(a.to_live(field))
    b.set_state(b.to_live(field))
    for _ in range(200):
        a.step()
        b.step()
    scale = np.abs(b.u).max()
    assert np.max(np.abs(a.u - b.u)) / scale < 1e-11, "uniform grain diverges from its twin"


def test_continuum_oracle_and_its_isotropic_reduction():
    """The continuum law, and the fact that it *is* the shipped isotropic law at ``g = (1,1,1)``."""
    Lx, Ly = 0.62, 0.43
    modes = [(1, 1), (2, 1), (1, 2), (2, 2), (3, 1)]
    f_iso = modal.rectangular_plate_freqs(KAPPA, Lx, Ly, modes)
    f_via_ortho = modal.orthotropic_plate_freqs(KAPPA, Lx, Ly, modes, 1.0, 1.0, 1.0)
    assert np.max(np.abs(f_via_ortho - f_iso) / f_iso) < 1e-14, (
        "the orthotropic continuum law does not reduce to the isotropic one -- suspect the "
        "factor of 2 on the cross term (a² + 2ab + b² = (a+b)²)"
    )
    # and the grained law is genuinely different, so the reduction above is not vacuous
    f_grained = modal.orthotropic_plate_freqs(KAPPA, Lx, Ly, modes, **G_STRONG)
    assert np.all(f_grained <= f_iso), "every ratio is <= 1, so no mode may rise"
    assert f_grained[0] < 0.5 * f_iso[0], "the fundamental must drop by more than an octave"
    # But NOT every mode drops much, and that asymmetry is the model working rather than a defect:
    # (3,1) has its curvature almost entirely along the *stiff* axis, so it keeps most of its
    # frequency (~0.84x) while (1,1) loses more than half. A grained plate is not a uniformly
    # detuned one -- which is why the mode ordering can move at all (see the headline below).
    # Compared on a SQUARE plate, so that only the material differs. On the 0.62x0.43 rectangle
    # above the two modes are not comparable at all -- (1,3) puts its three half-waves across the
    # *shorter* side, and the extra wavenumber cancels most of the softer stiffness (1.6x, not 3x).
    # A shape asymmetry impersonating a material one is the easy mistake here.
    stiff_axis = modal.orthotropic_plate_freqs(KAPPA, Lx, Lx, [(3, 1)], **G_STRONG)[0]
    soft_axis = modal.orthotropic_plate_freqs(KAPPA, Lx, Lx, [(1, 3)], **G_STRONG)[0]
    assert stiff_axis > 2.5 * soft_axis, (
        f"curvature along the grain ({stiff_axis:.1f} Hz) should cost far more than the same "
        f"curvature across it ({soft_axis:.1f} Hz)"
    )


def test_discrete_converges_to_the_continuum_at_second_order():
    """O(h²), the same order and the same source as the isotropic plate.

    The spatial operator is *exact* on these modes, so the whole error is the per-axis
    ``(4/h²)sin²(mπh/2L) → (mπ/L)²`` discrepancy, plus the theta-scheme's temporal error which at
    fixed ``mu`` scales as ``h⁴`` and so does not set the order.
    """
    Lx, mu = 1.0, 0.5
    modes = [(1, 1), (2, 1)]
    f_cont = modal.orthotropic_plate_freqs(KAPPA, Lx, Lx, modes, **G_STRONG)
    hs, errs = [], []
    for N in (16, 32, 64, 128):
        h = Lx / N
        k = mu * h * h / KAPPA
        lam_x = modal.dirichlet_axis_eigenvalue(np.array([m for m, _ in modes]), Lx, h)
        lam_y = modal.dirichlet_axis_eigenvalue(np.array([n for _, n in modes]), Lx, h)
        f_disc = modal.discrete_orthotropic_plate_eigenfrequency(
            lam_x, lam_y, KAPPA, k, THETA, **G_STRONG
        )
        hs.append(h)
        errs.append(np.max(np.abs(f_disc - f_cont)))
    orders = convergence_orders(np.array(errs), np.array(hs))
    assert np.all(np.array(errs[1:]) < np.array(errs[:-1])), f"errors not decreasing: {errs}"
    assert orders[-1] > 1.8, f"order {orders[-1]:.2f} < 1.8"


def test_the_time_stepper_actually_rings_at_the_grained_frequency():
    """End-to-end: FFT the output of a grained plate and find the analytic mode where predicted.

    The operator tests above prove the matrix; this proves the matrix is the one being stepped.
    """
    p = make_orthotropic_plate(N=48, mu=1.0, **G_STRONG)
    f_11 = float(orthotropic_mode_freqs(p, [(1, 1)])[0])
    p.set_state(_pluck(p))
    pickup = p.pickup_index_at(0.3 * p.Lx, 0.28 * p.Ly)
    res = simulate(p, num_steps=int(0.5 * p.fs), pickup_index=pickup)
    found = spectrum.measure_partials_near(res.output, res.fs, np.array([f_11]), search_hz=20.0)[0]
    err = abs(modal.cents(found, f_11))
    assert err < 5.0, (
        f"grained fundamental off by {err:.2f} cents (found {found:.2f}, want {f_11:.2f})"
    )


@pytest.mark.parametrize("mu", [0.5, 2.0, 8.0])
def test_energy_is_conserved_with_grain(mu):
    """Lossless drift below the acceptance bar, at timesteps an explicit scheme could not run.

    Orthotropy does not touch the reason conservation holds: ``B`` is still symmetric (the cross
    term ``δ_xx δ_yy`` is a product of two commuting symmetric factors and comes out exactly
    symmetric), and the energy is evaluated through the *same* matrix the update uses.
    """
    p = make_orthotropic_plate(N=32, mu=mu, **G_STRONG)
    p.set_state(_pluck(p))
    res = simulate(p, num_steps=int(0.5 * p.fs))
    assert res.energy_drift < DRIFT_TOL, f"drift {res.energy_drift:.2e} at mu={mu}"


def test_lossy_grained_plate_is_passive_and_a_low_mode_decays_at_2sigma():
    """Monotone decrease, and a *single low mode* decays at the physical rate.

    The rate half has to be run on one mode, not on a broadband pluck: the plate's shipped damping
    caveat means the top of the spectrum is effectively undamped at any usable timestep, so a
    broadband pluck decays at whatever mixture the excitation happened to contain (measured 0.67
    over 0.3 s against the 0.09 the physical rate would give) and says nothing about correctness.
    """
    p = make_orthotropic_plate(N=24, mu=2.0, sigma=4.0, **G_STRONG)
    p.set_state(_sine_field(p, 1, 1) * 1e-3)
    e = np.asarray(simulate(p, num_steps=int(0.3 * p.fs)).energy)
    _assert_monotone(e)
    rate = -np.log(e[-1] / e[0]) / 0.3
    assert abs(rate / (2.0 * p.sigma) - 1.0) < 0.02, (
        f"fundamental decays at {rate:.3f}/s, want ~{2 * p.sigma:.3f}/s"
    )


# =====================================================================================
# The guard -- what replaces the isotropic operator's free definiteness
# =====================================================================================


def test_the_cross_term_guard_is_sharp_and_rejected_at_construction():
    """``g_h > -sqrt(g_x g_y)`` or the operator is indefinite and the scheme is not stable.

    The isotropic ``L²`` is positive-definite for nothing — it is a square. With three coefficients
    it is a genuine condition, and the theta-scheme's "unconditionally stable for theta >= 1/4"
    quietly assumes it. Measured sharp: at 1.02x the boundary the assembled operator has a negative
    eigenvalue, at 0.98x it does not.
    """
    gx, gy = 11.0, 0.9
    floor = -np.sqrt(gx * gy)
    with pytest.raises(ValueError, match="indefinite"):
        make_orthotropic_plate(N=8, grain_x=gx, grain_cross=floor * 1.02, grain_y=gy)
    with pytest.raises(ValueError, match="indefinite"):
        make_orthotropic_plate(N=8, grain_x=gx, grain_cross=floor, grain_y=gy)
    ok = make_orthotropic_plate(N=8, grain_x=gx, grain_cross=floor * 0.98, grain_y=gy)
    lo = float(np.linalg.eigvalsh(ok.B.toarray()).min())
    assert lo > 0.0, f"just inside the guard the operator is still indefinite ({lo:.3e})"

    # and just outside it really is indefinite -- the guard is not merely conservative
    bad, _ = orthotropic_biharmonic(8, 8, 1 / 8, gx, floor * 1.02, gy)
    assert float(np.linalg.eigvalsh(bad.toarray()).min()) < 0.0, (
        "the guard rejects a valid operator"
    )


def test_degenerate_grain_arguments_are_rejected():
    for kwargs in (dict(grain_x=0.0), dict(grain_x=-1.0), dict(grain_y=0.0), dict(grain_y=-2.0)):
        with pytest.raises(ValueError, match="must be positive"):
            make_orthotropic_plate(N=8, **kwargs)


def test_grain_on_the_free_boundary_needs_the_split_and_says_so():
    """The free plate assembles from the strain energy and needs the coupling and torsional
    rigidities *separately*, not just their combination ``H``. As of model #5of it accepts them —
    but a grain **without** them is still refused rather than silently completed from Poisson's
    ratio, which would be a wrong default rather than a permitted unphysical choice.
    """
    common = dict(Lx=1.0, Ly=1.0, kappa=KAPPA, rho=RHO_AREAL_DEFAULT, fs=40000.0, N=8,
                  boundary="free")
    with pytest.raises(ValueError, match="separately"):
        Plate(**common, grain_y=0.5)
    # ... and with the split it builds (model #5of; the free branch's own suite covers the physics).
    p = Plate(**common, grain_y=0.5, grain_coupling=0.1, grain_torsion=0.3)
    assert p.grain_cross == 0.1 + 2.0 * 0.3, "grain_cross must be derived from the split"


# =====================================================================================
# The material chain -- where the factor of 2 lives
# =====================================================================================


def test_an_isotropic_material_returns_exactly_no_grain():
    """The convention test. ``H = D_1 + 2 D_xy`` must land on ``D`` **exactly** at isotropy."""
    E, nu, t, rho = 1.1e10, 0.3, 3.0e-3, 420.0
    spec = grain_ratios_from_material(
        E_x=E, E_y=E, nu_xy=nu, G_xy=E / (2.0 * (1.0 + nu)), thickness=t, rho=rho
    )
    gx, gh, gy = spec.grain_x, spec.grain_cross, spec.grain_y
    assert (gx, gh, gy) == (1.0, 1.0, 1.0), (
        f"isotropic material did not give an isotropic plate: {(gx, gh, gy)} -- the cross term is "
        f"mis-packaged"
    )
    D = E * t**3 / (12.0 * (1.0 - nu * nu))
    assert abs(spec.kappa / np.sqrt(D / (rho * t)) - 1.0) < 1e-14

    # The density the caller must hand to Plate is AREAL, and it is returned by name so that the
    # volume density cannot be passed through by accident. That slip would leave every frequency
    # correct (kappa carries them) and every energy wrong by a factor of the thickness -- the exact
    # class of error this module's own detector tests show nothing catches. Here, 333x.
    assert spec.rho_s == rho * t
    assert abs(rho / spec.rho_s - 1.0 / t) < 1e-9
    assert spec.rho_s != rho, "if these ever coincide the trap is invisible and so is this test"


def test_the_two_rival_cross_term_packagings_are_measurably_wrong():
    """Name the near-misses and price them, so a future transcription slip has a known signature.

    The orthotropic literature invites two other readings of the cross term: ``H = D_1`` and
    ``H = D_1 + D_xy``. At nu = 0.3 they come out at 0.30x and 0.65x of the correct value. Both are
    positive, both keep the operator definite, and both therefore produce a perfectly stable,
    exactly energy-conserving, **wrong** plate. This test is the record of what they cost.
    """
    E, nu, t = 1.1e10, 0.3, 3.0e-3
    D = E * t**3 / (12.0 * (1.0 - nu * nu))
    G = E / (2.0 * (1.0 + nu))
    D_1, D_xy = nu * D, G * t**3 / 12.0
    assert abs((D_1 + 2.0 * D_xy) / D - 1.0) < 1e-14  # correct
    assert abs(D_1 / D - 0.30) < 5e-3  # H = D_1 alone
    assert abs((D_1 + D_xy) / D - 0.65) < 5e-3  # H = D_1 + D_xy


def test_spruce_is_not_a_stretched_isotropic_plate():
    """The number the headline rests on: ``H / sqrt(D_x D_y) ≈ 0.57``, not 1.

    A plate that were merely "isotropic, stretched along one axis" would sit at exactly 1 — its
    cross term is forced by the two direct stiffnesses. Real wood does not, so the cross term is an
    independent axis and not a consequence of the stiffness ratio. Everything in the findings
    section below is downstream of this one number.
    """
    _spec = grain_ratios_from_material(**SPRUCE)
    gx, gh, gy = _spec.grain_x, _spec.grain_cross, _spec.grain_y
    assert 13.0 < gx / gy < 15.0, f"spruce stiffness ratio {gx / gy:.2f} outside the expected band"
    ratio = gh / np.sqrt(gx * gy)
    assert 0.5 < ratio < 0.65, f"H/sqrt(Dx*Dy) = {ratio:.3f}"
    assert ratio < 0.9, "if this ever reaches 1, wood IS a stretched isotropic plate and the "
    "headline below is vacuous"


# =====================================================================================
# Findings
# =====================================================================================


def test_the_energy_ledger_cannot_see_a_wrongly_wired_grain():
    """**The detector finding.** Four constants with clashing published conventions is exactly the
    class of slip no energy ledger can catch, because a wrong-but-consistent coefficient is carried
    self-consistently through every step — the same shape as the volume-vs-areal density slip that
    stayed green for a whole batch elsewhere in this project.

    So falsify the wiring on purpose and record, per detector, what happens. Three deliberate
    mutations (grain swapped end for end, the factor of 2 dropped, the cross term taken as ``D_1``
    alone) plus one control (the same operator, reassociated).

    Result, and it is the point of the test: **the energy ledger is green for all four** — every
    one of them is a symmetric definite operator and conserves to machine precision. Only the modal
    oracle separates them, and it separates them by a lot (tens of percent in frequency).
    """
    _spec = grain_ratios_from_material(**SPRUCE)
    gx, gh, gy = _spec.grain_x, _spec.grain_cross, _spec.grain_y
    truth = dict(grain_x=gx, grain_cross=gh, grain_y=gy)
    mutations = {
        "swapped": dict(grain_x=gy, grain_cross=gh, grain_y=gx),
        "factor_2_dropped": dict(grain_x=gx, grain_cross=gh / 2.0, grain_y=gy),
        "D_1_only": dict(grain_x=gx, grain_cross=gh * 0.30, grain_y=gy),
    }
    modes = [(1, 1), (2, 1), (1, 2), (2, 2)]

    p_true = make_orthotropic_plate(N=24, mu=2.0, **truth)
    f_true = orthotropic_mode_freqs(p_true, modes)
    p_true.set_state(_pluck(p_true))
    drift_true = simulate(p_true, num_steps=int(0.2 * p_true.fs)).energy_drift
    assert drift_true < DRIFT_TOL

    for name, g in mutations.items():
        p = make_orthotropic_plate(N=24, mu=2.0, **g)
        p.set_state(_pluck(p))
        drift = simulate(p, num_steps=int(0.2 * p.fs)).energy_drift
        # DETECTOR 1 -- the energy ledger. Blind, every time.
        assert drift < DRIFT_TOL, (
            f"'{name}' happened to break conservation; the finding is that it does NOT, so this "
            f"failure means the mutation is not the intended one"
        )
        # DETECTOR 2 -- the modal oracle. This is the one that works.
        f_bad = orthotropic_mode_freqs(p, modes)
        worst = np.max(np.abs(f_bad - f_true) / f_true)
        assert worst > 0.05, f"'{name}' is within 5% of the truth -- not a sharp enough mutation"

    # The control: the same three products summed in a different order must stay green on BOTH.
    p_ctl = make_orthotropic_plate(N=24, mu=2.0, **truth)
    assert np.max(np.abs(orthotropic_mode_freqs(p_ctl, modes) - f_true) / f_true) < 1e-14


def test_a_square_plates_diagonal_modes_are_blind_to_the_grain_running_the_wrong_way():
    """**And the oracle has a blind spot of its own — a specific, avoidable one.**

    Swapping the grain end for end (running it across the plate instead of along it) is the most
    likely wiring mistake of the three, and on a **square** plate every diagonal mode ``(m, m)`` is
    invariant under it. The reason is trivial once seen — swapping ``g_x`` and ``g_y`` while
    ``λ_x = λ_y`` leaves ``g_x λ_x² + g_y λ_y²`` untouched — but the consequence is not: a
    validation that checks a square plate's fundamental, or its first few *diagonal* modes,
    **passes a plate whose grain runs 90 degrees wrong**.

    Invariant in exact arithmetic; in floating point the two groupings round differently and the
    modes agree to 0 or 2.2e-16 depending on the mode, so the assertion is a tight relative bound
    rather than ``array_equal``. (It was written as ``array_equal`` first, and (2,2) — alone among
    the four — failed it at the last bit.)

    Two ways out, both asserted here: use an off-diagonal mode (the (2,1)/(1,2) pair swaps
    outright, 126% apart), or use a non-square plate (the fundamental alone then moves 62%). The
    suite's own falsification test above uses off-diagonal modes for exactly this reason, and this
    test is the record of why that was not an arbitrary choice.
    """
    _spec = grain_ratios_from_material(**SPRUCE)
    gx, gh, gy = _spec.grain_x, _spec.grain_cross, _spec.grain_y
    square = dict(N=32, mu=2.0, Lx=1.0, Ly=1.0)
    right = make_orthotropic_plate(**square, grain_x=gx, grain_cross=gh, grain_y=gy)
    wrong = make_orthotropic_plate(**square, grain_x=gy, grain_cross=gh, grain_y=gx)

    diagonal = [(1, 1), (2, 2), (3, 3), (4, 4)]
    f_r, f_w = orthotropic_mode_freqs(right, diagonal), orthotropic_mode_freqs(wrong, diagonal)
    gap = float(np.max(np.abs(f_w / f_r - 1.0)))
    assert gap < 1e-14, (
        f"diagonal modes are supposed to be blind to the swap; they now differ by {gap:.2e} "
        f"({f_r} vs {f_w}). If this grows the blind spot has closed and the warning in this "
        f"docstring is stale -- but check for an accidental asymmetry in the rig first."
    )

    # Way out 1: an off-diagonal mode sees it immediately.
    off = orthotropic_mode_freqs(right, [(2, 1)])[0] / orthotropic_mode_freqs(wrong, [(2, 1)])[0]
    assert abs(off - 1.0) > 0.5, f"the off-diagonal mode should catch the swap; ratio {off:.3f}"

    # Way out 2: on a non-square plate even the fundamental catches it.
    rect = dict(N=32, mu=2.0, Lx=0.62, Ly=0.43)
    r2 = make_orthotropic_plate(**rect, grain_x=gx, grain_cross=gh, grain_y=gy)
    w2 = make_orthotropic_plate(**rect, grain_x=gy, grain_cross=gh, grain_y=gx)
    fund = orthotropic_mode_freqs(w2, [(1, 1)])[0] / orthotropic_mode_freqs(r2, [(1, 1)])[0]
    assert abs(fund - 1.0) > 0.3, f"a rectangle's fundamental should catch the swap; {fund:.3f}"


def test_grain_makes_the_theta_damping_anisotropic_and_the_ledger_stays_green():
    """A second thing the ledger cannot see, for a different reason.

    The plate's shipped damping caveat is that the theta time-average turns frequency-independent
    loss into frequency-*dependent* loss: mode decay goes as ``2σ(1 - θ Q k²)``. Under isotropy
    ``Q`` is a function of the Laplacian eigenvalue alone, so the two halves of a degenerate pair
    such as ``(2,1)`` and ``(1,2)`` on a square plate decay identically. With grain they no longer
    do — ``Q`` depends on how the mode's curvature splits along and across the grain, and the pair
    splits in *decay rate* as well as frequency.

    Passivity is untouched (the operator is SPD), so ``energy()`` is monotone and machine-precise
    either way. The rate is a property the ledger has no access to.

    **Measured, not derived.** An earlier version of this test computed both decay rates from the
    same formula it was claiming and so proved nothing. Here each mode is actually started alone
    and its energy decay is fitted, with the isotropic plate run as the control in the same rig.
    Result at ``mu = 4``, modes (4,1) against (1,4), nominal rate ``2 sigma = 8``/s:

        isotropic:  5.857 and 5.857   -- identical to six figures, as degeneracy requires
        grained:    6.024 and 7.751   -- 29% apart

    Both under-damped relative to 8/s, which is the shipped caveat; the finding is the *split*.
    """
    sigma, mu, secs = 4.0, 4.0, 0.3

    def measured_rate(plate, m, n):
        plate.set_state(_sine_field(plate, m, n) * 1e-3)
        e = np.asarray(simulate(plate, num_steps=int(secs * plate.fs)).energy)
        _assert_monotone(e, why="the ledger must stay monotone -- passivity is untouched")
        return -np.log(e[-1] / e[0]) / secs

    # Control: on a square isotropic plate (4,1) and (1,4) are exactly degenerate, in frequency
    # AND in decay, because Q depends only on Lambda = lam_x + lam_y.
    iso_a = measured_rate(make_plate(N=24, mu=mu, sigma=sigma), 4, 1)
    iso_b = measured_rate(make_plate(N=24, mu=mu, sigma=sigma), 1, 4)
    assert abs(iso_a - iso_b) / iso_a < 1e-6, (
        f"the isotropic control is not degenerate ({iso_a:.6f} vs {iso_b:.6f}); the grained split "
        f"below would then not be attributable to the grain"
    )

    a = measured_rate(make_orthotropic_plate(N=24, mu=mu, sigma=sigma, **G_STRONG), 4, 1)
    b = measured_rate(make_orthotropic_plate(N=24, mu=mu, sigma=sigma, **G_STRONG), 1, 4)
    split = abs(a - b) / max(a, b)
    assert split > 0.15, (
        f"the degenerate pair still decays at nearly the same rate ({a:.4f} vs {b:.4f}, "
        f"{split * 100:.1f}% apart); the anisotropic-damping claim would then be vacuous"
    )
    # Both are under-damped against the nominal 2*sigma -- the caveat, not a new defect.
    assert a < 2.0 * sigma and b < 2.0 * sigma


def test_the_cross_term_detunes_selectively_without_reordering_anything():
    """**The headline — and it is not the one this batch set out to prove.**

    The plan's claim was that the cross term reorders the modes, so that no amount of
    stiffness-ratio intuition reaches the answer. **It died on measurement.** What replaced it is
    sharper and splits in two.

    Hold the along/across stiffness ratio fixed at the spruce value 13.75 and sweep the cross term
    across a 5x range, from 0.2x to 1.0x of ``sqrt(g_x g_y)`` — the value a "stretched isotropic"
    plate would be pinned at. That range is not arbitrary: real spruce sits at **0.567x** of it and
    an isotropic material sits at **exactly 1.0x**, so the sweep spans everything between solid wood
    and no grain at all.

    **Half one — the ordering does not move.** All sixteen low modes stay in the same order across
    the whole sweep, at *both* grid resolutions. So on the question of which mode is which, the
    naive stiffness-ratio intuition is simply right, and the extra freedom buys nothing.

    **Half two — the frequencies move, and very unevenly.** Over the same sweep individual modes
    shift by between **1.3% and 29%** — a 22x spread in leverage — with a clear mechanism. The
    cross term enters as ``2 g_h λ_x λ_y`` against direct terms ``g_x λ_x²`` and ``g_y λ_y²``. A
    mode bending mainly *along* the grain has a huge ``g_x λ_x²`` that swamps it: (3,1) moves 2.3%.
    A mode bending mainly *across* the grain has only ``g_y λ_y²``, and ``g_y`` is 13.75x smaller,
    so the cross term is comparable to it: (2,4) moves 29.0%. **The cross term matters exactly where
    the direct stiffness is weakest**, which is the opposite of where a "stretched isotropic"
    picture would put it.

    **And the ordering result is a statement about the range, not about the term being inert:**
    push the cross term to 2x the stretched-isotropic value — past any solid wood, into engineered
    laminate territory — and (2,4) and (3,1) do swap. So the term *can* reorder; it just does not
    anywhere a plank lives.

    Both halves are checked at two grid resolutions, because a reshuffle between near-degenerate
    modes is exactly what a discretisation error can manufacture. If the grids disagreed, the
    reordering result would be numerical and would not be reportable either way.
    """
    modes = [(m, n) for m in range(1, 5) for n in range(1, 5)]
    gx, gy = 1.0, 1.0 / 13.75
    stretched = np.sqrt(gx * gy)  # what a stretched-isotropic plate would be forced to

    def freqs(N, factor):
        p = make_orthotropic_plate(
            N=N, mu=1.0, grain_x=gx, grain_cross=factor * stretched, grain_y=gy
        )
        assert p.grain_x / p.grain_y == pytest.approx(13.75, rel=1e-9)  # ratio held FIXED
        return orthotropic_mode_freqs(p, modes)

    # -- half one: no reordering between solid wood (0.567) and isotropic (1.0), and either side --
    physical = [0.2, 0.567, 1.0]
    orderings = {N: [tuple(np.argsort(freqs(N, f))) for f in physical] for N in (24, 48)}
    assert len(set(orderings[24])) == 1, (
        f"the ordering moved inside the physically spanned range: {orderings[24]}"
    )
    assert orderings[24] == orderings[48], "the two grids disagree about the ordering"

    # -- half two: but the frequencies move, by wildly different amounts per mode --
    lo, hi = freqs(48, 0.2), freqs(48, 1.0)
    shift = hi / lo - 1.0
    by_mode = dict(zip(modes, shift, strict=True))
    assert by_mode[(3, 1)] < 0.05, (
        f"a mode bending along the grain should barely notice the cross term; moved "
        f"{by_mode[(3, 1)] * 100:.1f}%"
    )
    assert by_mode[(2, 4)] > 0.25, (
        f"a mode bending across the grain should notice it strongly; moved "
        f"{by_mode[(2, 4)] * 100:.1f}%"
    )
    assert shift.max() / shift.min() > 15.0, (
        f"the cross term's leverage is nearly uniform across modes "
        f"({shift.max() / shift.min():.1f}x spread); the selectivity half of the finding dies too"
    )

    # -- and the ordering claim is about the RANGE: past isotropy, it does reorder --
    beyond = tuple(np.argsort(freqs(48, 2.0)))
    assert beyond != orderings[48][0], (
        "even at 2x the stretched-isotropic cross term nothing reorders, so the term cannot "
        "reorder at all and the 'range' framing above overstates it"
    )


def test_the_grain_is_in_the_partial_series_and_not_in_the_level():
    """**The negative result, and it qualifies what grain buys a coupled instrument.**

    An instrument does not listen to the plate's field; it is coupled at a point — a bridge — and a
    point sees a weighted sum over modes. So: with raw pitch removed (compare against an isotropic
    plate **matched on the fundamental**), where does the grain actually show up?

    **Not in the level.** Across five very different pluck/pickup geometries the RMS at a single
    node lands at 1.116, 0.809, 0.873, 0.916, 0.834 times the matched isotropic plate's. That
    straddles 1 — it swings both above and below — so the spread is the *geometry* talking, not the
    grain. Any single one of those numbers quoted alone would read as a measurement of the wood and
    would be a measurement of where the pluck happened to land. (An earlier version of this test
    quoted exactly one of them.)

    **All of it is in the partial series.** The mode ratios above the fundamental move by 37%,
    which no choice of pluck or pickup position affects, because it is a property of the operator.

    The practical consequence, and the reason this is worth a test rather than a comment: if you
    couple this plate into a chain as an instrument body and judge it by how loud the terminus
    rings, **you cannot tell wood from metal**. The grain is audible as tuning, not as output.
    """
    modes = [(m, n) for m in range(1, 4) for n in range(1, 4)]
    _spec = grain_ratios_from_material(**SPRUCE)
    gx, gh, gy = _spec.grain_x, _spec.grain_cross, _spec.grain_y

    grained = make_orthotropic_plate(N=32, mu=1.0, grain_x=gx, grain_cross=gh, grain_y=gy)
    f_grained = np.sort(orthotropic_mode_freqs(grained, modes))
    iso_probe = make_plate(N=32, mu=1.0)
    f_iso_probe = np.sort(
        modal.discrete_plate_eigenfrequency(
            modal.rectangular_discrete_eigenvalues(iso_probe.h, 32, 32, modes),
            iso_probe.kappa,
            iso_probe.k,
            iso_probe.theta,
        )
    )
    scale = f_grained[0] / f_iso_probe[0]

    # (1) the partial series, pitch removed -- a property of the operator, geometry-independent
    shape = np.max(
        np.abs(f_grained / f_grained[0] - f_iso_probe / f_iso_probe[0])
        / (f_iso_probe / f_iso_probe[0])
    )
    assert shape > 0.2, f"the grained partial series barely reshaped ({shape:.3f}); no contrast"

    # (2) the level at one node, over five unrelated pluck/pickup geometries
    def ring(p, pluck_xy, pick_xy):
        field = raised_cosine_2d(
            p.X, p.Y, (pluck_xy[0] * p.Lx, pluck_xy[1] * p.Ly), 0.25 * min(p.Lx, p.Ly), 1e-3
        )
        field[~p.mask] = 0.0
        p.set_state(field)
        idx = p.pickup_index_at(pick_xy[0] * p.Lx, pick_xy[1] * p.Ly)
        out = simulate(p, num_steps=int(0.15 * p.fs), pickup_index=idx).output
        return float(np.sqrt(np.mean(np.square(out))))

    geometries = [
        ((0.40, 0.55), (0.31, 0.27)),
        ((0.50, 0.50), (0.50, 0.50)),
        ((0.25, 0.70), (0.60, 0.35)),
        ((0.60, 0.30), (0.20, 0.80)),
        ((0.45, 0.45), (0.70, 0.70)),
    ]
    ratios = []
    for pluck_xy, pick_xy in geometries:
        a = make_orthotropic_plate(N=32, mu=1.0, grain_x=gx, grain_cross=gh, grain_y=gy)
        # Matched SAMPLE RATE, not matched mu -- `make_plate` solves fs from kappa, so a twin
        # built through it would run at 12.0 kHz against the grained plate's 20.5 kHz and the two
        # RMS figures would not be comparable. (This is the same rig error as in
        # `test_uniform_grain_collapses_to_an_isotropic_plate`, made a second time in a second
        # test; it turned out not to move these numbers, but that was luck, not design.)
        b = Plate(
            Lx=a.Lx, Ly=a.Ly, kappa=KAPPA * scale, rho=RHO_AREAL_DEFAULT,
            fs=a.fs, N=32, theta=a.theta,
        )
        assert a.fs == b.fs and a.k == b.k
        ratios.append(ring(a, pluck_xy, pick_xy) / ring(b, pluck_xy, pick_xy))
    ratios = np.array(ratios)

    # The twin really is pitch-matched at this sample rate (0.0024%), so nothing below is raw pitch.
    f_twin = np.sort(
        modal.discrete_plate_eigenfrequency(
            modal.rectangular_discrete_eigenvalues(b.h, 32, 32, modes), b.kappa, b.k, b.theta
        )
    )
    assert abs(f_twin[0] / f_grained[0] - 1.0) < 1e-3, "the twin is not pitch-matched"


    assert ratios.min() < 1.0 < ratios.max(), (
        f"the level ratios {np.round(ratios, 3)} all sit on one side of 1, so the level DOES carry "
        f"a consistent grain signature and this finding is wrong as stated"
    )
    # The 0.3 bar is a JUDGEMENT about what still counts as geometry noise, not a derived
    # threshold -- measured values span 0.811-1.117, so it sits 0.19 clear. Do not read it as
    # physics; the load-bearing assertion is the straddle above, which cannot be tuned.
    assert np.max(np.abs(ratios - 1.0)) < 0.3, (
        f"the level moved by more than 30% ({np.round(ratios, 3)}) -- too large to call geometry "
        f"noise; the claim that the grain is absent from the level needs re-examining"
    )
