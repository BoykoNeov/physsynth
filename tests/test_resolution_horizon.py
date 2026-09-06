"""Where each scheme stops being in tune — the resolution horizon, and its two mechanisms.

Every accuracy claim this project makes about a mode is bounded by whether that mode's *frequency*
is right, and until 2026-09-06 no model had that boundary written down. The batch that produced
this file is ``docs/dev/resolution-horizon-plan.md``; it came out of
``docs/dev/theta-loss-compensation-plan.md``, whose probe found that the theta-scheme's decay-rate
suppression and its pitch flattening are one factor and its square root.

**Nothing here is a frozen integer.** A horizon is a property of ``(model, N, k, params)``, so a
number recorded for one fixture is a claim about that fixture — the lesson this project keeps
paying for. What is asserted instead is *family behaviour*: which way the horizon moves when a
knob turns, that one family has a floor no sample rate passes, that the other family's errors
cancel at a magic Courant number, and one genuine closed form the measurement can be checked
against from outside itself.

**The two mechanisms, because a reader must not assume the first is universal.**

* **Implicit theta-scheme** (stiff and damped strings, both plates, the beam). The time error
  gives ``omega/omega_c = 1/sqrt(1 + theta k^2 Q) < 1`` and the spatial operator gives
  ``sinc(u) < 1``. Both flatten, so they **compound**: there is a hard space floor and refining
  the timestep cannot pass it.
* **Explicit leapfrog** (ideal string, membrane). ``sin(omega k / 2) = lambda sin(m pi h / 2L)``,
  whose time factor is **sharp** and cancels the spatial droop exactly at ``lambda = 1``
  (and along the diagonal at ``lambda = 1/sqrt(2)`` in 2-D). Here refining the timestep moves the
  horizon the *wrong* way.
"""

from __future__ import annotations

import numpy as np
import pytest
from helpers import (
    L_DEFAULT,
    make_damped_string,
    make_membrane,
    make_plate,
    pitch_error_cents,
    pitch_horizon,
    spatial_operator_horizon,
    wave_speed,
)
from scipy.optimize import brentq

from physsynth.analysis import modal

CENTS = 5.0  # a sustained tone's pitch JND is around this; the bound is an argument, not a law


# =====================================================================================
# The primitive itself, including the case the conservative reading exists for
# =====================================================================================


def test_an_exact_scheme_has_no_horizon_to_state():
    f = np.array([100.0, 200.0, 300.0, 400.0])
    horizon, monotone = pitch_horizon(f, f, CENTS)
    assert horizon == 4, "every mode is exact, so the prefix is the whole set"
    assert monotone


def test_the_prefix_reading_is_conservative_and_says_so_when_the_curve_is_not_monotone():
    # Modes 1-2 in tune, mode 3 far out, mode 4 back in tune. "The last mode inside the bound" is
    # 4; the leading prefix is 2. They differ exactly here, which is why the flag is returned.
    f_cont = np.array([100.0, 200.0, 300.0, 400.0])
    f_disc = f_cont * np.array([1.0, 1.0, 0.9, 1.0])
    horizon, monotone = pitch_horizon(f_disc, f_cont, CENTS)
    assert horizon == 2, "the prefix must stop at the first mode outside the bound"
    assert not monotone, "a non-monotone error curve must be reported, not hidden behind one int"
    last_inside = int(np.nonzero(np.abs(pitch_error_cents(f_disc, f_cont)) <= CENTS)[0][-1]) + 1
    assert last_inside == 4 and last_inside != horizon


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_a_non_positive_bound_is_refused(bad):
    with pytest.raises(ValueError):
        pitch_horizon([100.0], [100.0], bad)


def test_mismatched_families_are_refused_rather_than_broadcast():
    with pytest.raises(ValueError):
        pitch_horizon([100.0, 200.0], [100.0], CENTS)


# =====================================================================================
# The one closed form: the space floor is checkable from OUTSIDE the measurement
# =====================================================================================


def _sinc_horizon_fraction(cents: float) -> float:
    """``m*/N`` for a pure wave, from the second difference's own eigenvalue error.

    The discrete eigenvalue is ``(2/h) sin(m pi h / 2L)`` against the continuum ``m pi / L``, so
    the frequency ratio is ``sinc(u)`` with ``u = m pi / 2N`` and the horizon solves
    ``sin(u)/u = 2^(-cents/1200)``. Independent of ``c``, ``L`` and ``N`` — which is the claim.
    """
    target = 2.0 ** (-cents / 1200.0)
    u = brentq(lambda z: np.sin(z) / z - target, 1e-12, np.pi / 2.0)
    return 2.0 * u / np.pi


@pytest.mark.parametrize("cents", [5.0, 25.0, 100.0])
@pytest.mark.parametrize("n", [64, 128, 256, 512, 1024])
def test_the_wave_space_floor_matches_its_closed_form(n, cents):
    """The measured floor against ``sin(u)/u = 2^(-cents/1200)``.

    This is the only bar in the file that checks the primitive against something outside itself,
    and it is analytic rather than recorded. The tolerance is one mode: the horizon is an integer
    count and the closed form is not, so they can differ by the quantisation and no more.
    """
    horizon, monotone = spatial_operator_horizon(n, kappa=0.0, cents=cents)
    assert monotone, "a pure wave's sinc droop is monotone in the mode index"
    predicted = _sinc_horizon_fraction(cents) * n
    assert abs(horizon - predicted) <= 1.0, (
        f"N={n}, {cents:g} cents: measured floor {horizon} against the closed form "
        f"{predicted:.2f} — more than the integer quantisation apart"
    )


def test_the_wave_space_floor_is_a_fraction_of_the_GRID_and_not_a_frequency():
    """Doubling ``N`` doubles the floor. Nothing about ``c``, ``L`` or ``fs`` enters it.

    The bar is **derived rather than picked**: the horizon is an integer count, so on a grid of
    ``N`` the fraction can only be read to ``1/N``, and the coarsest grid in the sweep sets how
    much spread a genuinely constant fraction is still allowed to show. Anything tighter would be
    a bar on the quantisation rather than on the claim, and would flip red the first time a
    rounding boundary moved by one mode.
    """
    grids = (128, 256, 512, 1024)
    fractions = [spatial_operator_horizon(n, kappa=0.0)[0] / n for n in grids]
    quantisation = 1.0 / min(grids)
    assert max(fractions) - min(fractions) <= quantisation, (
        f"the floor is not a constant fraction of the grid: {fractions} spread "
        f"{max(fractions) - min(fractions):.5f} against the coarsest grid's own quantisation "
        f"{quantisation:.5f}"
    )


def test_stiffness_lowers_the_floor_because_the_biharmonic_errs_at_fourth_power():
    """``Q = c^2 p^2 + kappa^2 p^4``, and the ``p^4`` term carries twice the sinc droop.

    So a stiffer string resolves a *smaller* share of its own grid, and the share keeps shrinking
    as the grid refines — at fixed ``L`` a finer ``h`` admits higher ``p^2``, where the quartic
    term dominates. Both directions are asserted; neither number is.
    """
    at_n = {k: spatial_operator_horizon(512, kappa=k)[0] / 512 for k in (0.0, 2.0, 8.0)}
    assert at_n[0.0] > at_n[2.0] > at_n[8.0], f"stiffness did not lower the floor: {at_n}"
    coarse = spatial_operator_horizon(128, kappa=8.0)[0] / 128
    fine = spatial_operator_horizon(1024, kappa=8.0)[0] / 1024
    assert fine < coarse, (
        f"a stiff string's resolved share should shrink under refinement: {coarse} -> {fine}"
    )


# =====================================================================================
# Family 1 — the implicit theta-scheme: the two errors COMPOUND, so the floor is real
# =====================================================================================


def _string_horizon(n: int, lam: float, kappa: float, cents: float = CENTS) -> tuple[int, bool]:
    c, s = wave_speed(), make_damped_string(N=n, lam=lam, kappa=kappa)
    modes = np.arange(1, n)
    f_disc = np.array(
        [modal.discrete_stiff_mode_frequency(c, L_DEFAULT, n, kappa, s.k, int(m), s.theta)
         for m in modes]
    )
    f_cont = np.asarray(modal.stiff_harmonic_frequencies(c, L_DEFAULT, kappa, int(modes[-1])))
    return pitch_horizon(f_disc, f_cont, cents)


@pytest.mark.parametrize("kappa", [0.0, 2.0, 8.0])
def test_the_theta_string_horizon_rises_with_the_sample_rate_and_then_STOPS(kappa):
    """The batch's headline for this family: refining ``k`` buys modes until it buys nothing.

    Asserted as three claims, none of them a number: the horizon never *falls* as ``k`` falls, it
    never passes the space floor, and by the smallest ``k`` in the sweep it has arrived there. The
    third is what makes the first two more than a tautology — a floor nothing reaches is not
    demonstrably a floor.
    """
    n = 256
    lams = [2.0, 1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125]
    horizons = [_string_horizon(n, lam, kappa)[0] for lam in lams]
    floor = spatial_operator_horizon(n, kappa=kappa)[0]

    assert all(b >= a for a, b in zip(horizons, horizons[1:], strict=False)), (
        f"kappa={kappa}: a finer timestep lost modes: {list(zip(lams, horizons, strict=True))}"
    )
    assert max(horizons) <= floor, (
        f"kappa={kappa}: horizon {max(horizons)} passed the space floor {floor} — the time and "
        "space errors are supposed to compound, so this would mean they cancelled"
    )
    assert horizons[-1] >= floor - 1, (
        f"kappa={kappa}: a 64x refinement stalled at {horizons[-1]}, short of the floor {floor}"
    )


def test_the_canonical_theta_string_resolves_under_a_tenth_of_its_grid_and_stops_early():
    """The floor is not a theoretical limit reached at absurd rates — it arrives early.

    At the canonical ``lambda = 1`` the string is time-limited, and it is already within a couple
    of modes of its own space floor by ``lambda = 1/8``. What that costs is the point: the
    canonical fixture resolves a small share of its grid, and eight times the sample rate roughly
    doubles it and then stops.
    """
    n, kappa = 256, 2.0
    at_canonical = _string_horizon(n, 1.0, kappa)[0]
    at_eighth = _string_horizon(n, 0.125, kappa)[0]
    floor = spatial_operator_horizon(n, kappa=kappa)[0]
    assert at_canonical < at_eighth, "8x the sample rate should still be buying modes"
    assert at_eighth >= floor - 1, "and by then it should have essentially stopped"
    assert at_canonical / n < 0.10, (
        f"the canonical lambda = 1 string resolves {at_canonical}/{n} of its grid to {CENTS:g} "
        "cents; if this ever exceeds a tenth the fixture or the oracle changed"
    )


def test_the_plate_is_space_limited_at_every_sample_rate_the_suite_uses():
    """A 40x change of sample rate moves the plate's horizon by at most one mode.

    The plate's frequency is proportional to the Laplacian eigenvalue rather than its square root,
    so it carries **twice** the sinc droop of a string — and the theta average adds to it rather
    than cancelling. The consequence is that the plate's horizon is essentially the space floor
    already, which is why ``mu`` is not the knob that fixes it.
    """
    n = 32
    horizons = []
    for mu in (2.0, 0.05):
        p = make_plate(N=n, mu=mu, sigma=0.0)
        modes = np.arange(1, n)
        lam_disc = np.array(
            [2.0 * modal.dirichlet_axis_eigenvalue(int(m), p.Lx, p.h) for m in modes]
        )
        lam_cont = 2.0 * (modes * np.pi / p.Lx) ** 2
        f_disc = np.array(
            [modal.discrete_plate_eigenfrequency(float(x), p.kappa, p.k, p.theta)
             for x in lam_disc]
        )
        f_cont = p.kappa * lam_cont / (2.0 * np.pi)
        horizon, monotone = pitch_horizon(f_disc, f_cont, CENTS)
        assert monotone
        horizons.append(horizon)
    assert abs(horizons[0] - horizons[1]) <= 1, (
        f"a 40x sample-rate change moved the plate's horizon from {horizons[0]} to {horizons[1]}; "
        "the plate is supposed to be space-limited"
    )


# =====================================================================================
# Family 2 — the explicit leapfrog: the two errors CANCEL, and only at one Courant number
# =====================================================================================


def _ideal_horizon(n: int, lam: float) -> tuple[int, bool]:
    c = wave_speed()
    modes = np.arange(1, n)
    f_disc = np.array(
        [modal.discrete_mode_frequency(c, L_DEFAULT, n, lam, int(m)) for m in modes]
    )
    f_cont = modes * c / (2.0 * L_DEFAULT)
    return pitch_horizon(f_disc, f_cont, CENTS)


def test_the_explicit_string_beats_its_own_space_floor_at_lambda_one():
    """The claim that the two families are genuinely different, in one assertion.

    An implicit scheme can never pass the space floor. The explicit one does — by an order of
    magnitude — because its time error is sharp where the space error is flat, and at
    ``lambda = 1`` they cancel identically rather than merely partially.
    """
    n = 256
    horizon, _ = _ideal_horizon(n, 1.0)
    floor = spatial_operator_horizon(n, kappa=0.0)[0]
    assert horizon > 8 * floor, (
        f"lambda = 1 resolved {horizon} of {n - 1} modes against a space floor of {floor}; the "
        "cancellation that makes lambda = 1 exact did not happen"
    )
    assert horizon >= n - 2, "lambda = 1 is the dispersionless case — essentially every mode"


def test_refining_the_explicit_timestep_makes_the_string_WORSE():
    """The direction that surprises, and the reason it is worth a test rather than a comment.

    Below ``lambda = 1`` the cancellation is partial, and it gets less complete as ``lambda``
    falls. A user who lowers the Courant number for safety is buying nothing and paying pitch
    accuracy for it — the opposite of the implicit family, where the same move buys modes.
    """
    n = 256
    lams = [1.0, 0.99, 0.95, 0.9, 0.75, 0.5]
    horizons = [_ideal_horizon(n, lam)[0] for lam in lams]
    assert all(b <= a for a, b in zip(horizons, horizons[1:], strict=False)), (
        f"the horizon did not fall monotonically as lambda fell: "
        f"{list(zip(lams, horizons, strict=True))}"
    )
    assert horizons[-1] < horizons[0] / 5, (
        f"halving the Courant number should cost most of the band: {horizons[0]} -> {horizons[-1]}"
    )


def _membrane_family_horizon(n: int, lam: float, second) -> tuple[int, bool]:
    mem = make_membrane(domain="rectangle", N=n, lam=lam)
    modes = [(int(m), int(second(int(m)))) for m in range(1, n)]
    lam_disc = np.asarray(modal.rectangular_discrete_eigenvalues(mem.h, n, n, modes))
    f_disc = np.asarray(modal.discrete_membrane_eigenfrequency(lam_disc, mem.c, mem.k))
    f_cont = np.asarray(modal.rectangular_membrane_freqs(mem.c, 1.0, 1.0, modes))
    return pitch_horizon(f_disc, f_cont, CENTS)


@pytest.mark.parametrize("n", [64, 128])
def test_the_membranes_cancellation_is_DIAGONAL_ONLY(n):
    """The check that stopped a wrong headline being written.

    At the 2-D Courant ceiling the diagonal modes are exact — and reading only that family would
    say "the membrane is in tune at ``lambda = 1/sqrt(2)``". The axial modes are not: they sit
    near the space floor, a factor of eight or nine below. A margin measured on one mode family is
    a claim about that family, which is this project's oldest recurring scar in a new place.
    """
    lam_ceiling = 1.0 / np.sqrt(2.0)
    diagonal, _ = _membrane_family_horizon(n, lam_ceiling, lambda m: m)
    axial, _ = _membrane_family_horizon(n, lam_ceiling, lambda m: 1)
    assert diagonal >= 0.9 * (n - 1), (
        f"the diagonal family should be essentially exact at the ceiling, got {diagonal}/{n - 1}"
    )
    assert axial < 0.25 * (n - 1), (
        f"the axial family should NOT be, got {axial}/{n - 1} — if this passes the cancellation is "
        "not diagonal-only and the doc's section 4 is wrong"
    )
    assert diagonal > 5 * axial, f"diagonal {diagonal} against axial {axial}"


def test_both_membrane_families_converge_to_the_same_space_floor():
    """Away from the magic number the cancellation is gone and only the grid is left.

    Which is the sentence that ties the two families together: the space floor is where *every*
    scheme ends up, and what distinguishes the families is only whether a Courant number exists
    that beats it.
    """
    n = 128
    tiny = 0.125
    diagonal, _ = _membrane_family_horizon(n, tiny, lambda m: m)
    axial, _ = _membrane_family_horizon(n, tiny, lambda m: 1)
    floor = spatial_operator_horizon(n, kappa=0.0)[0]
    assert abs(diagonal - axial) <= 1, (
        f"far below the ceiling the two families should agree: {diagonal} against {axial}"
    )
    assert abs(diagonal - floor) <= 2, (
        f"and both should sit on the 1-D space floor {floor}, got {diagonal}"
    )
