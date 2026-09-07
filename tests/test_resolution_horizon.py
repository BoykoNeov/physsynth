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
    KAPPA_PLATE_DEFAULT,
    L_DEFAULT,
    PLATE_THETA_DEFAULT,
    make_damped_string,
    make_membrane,
    make_plate,
    mode_block,
    mode_family,
    pitch_error_cents,
    pitch_horizon,
    sinc_horizon_fraction,
    spatial_operator_horizon,
    wave_speed,
)

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
    predicted = sinc_horizon_fraction(cents) * n
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


def _membrane_family_horizon(n: int, lam: float, kind: str) -> tuple[int, bool]:
    mem = make_membrane(domain="rectangle", N=n, lam=lam)
    modes = mode_family(kind, n - 1)
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
    diagonal, _ = _membrane_family_horizon(n, lam_ceiling, "diagonal")
    axial, _ = _membrane_family_horizon(n, lam_ceiling, "axial")
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
    diagonal, _ = _membrane_family_horizon(n, tiny, "diagonal")
    axial, _ = _membrane_family_horizon(n, tiny, "axial")
    floor = spatial_operator_horizon(n, kappa=0.0)[0]
    assert abs(diagonal - axial) <= 1, (
        f"far below the ceiling the two families should agree: {diagonal} against {axial}"
    )
    assert abs(diagonal - floor) <= 2, (
        f"and both should sit on the 1-D space floor {floor}, got {diagonal}"
    )


# =====================================================================================
# The 2-D spectrum split by mode family — the plate, and the case section 7.4 could not derive
# =====================================================================================
#
# `pitch_horizon` reads a leading prefix, so it needs a sequence along which the error is
# monotone. A 2-D spectrum sorted by frequency is not one. The membrane showed that at its Courant
# ceiling, where the two families sit a factor of nine apart; the plate is the same question with
# the opposite answer, and that answer is why `test_plate_modal.py`'s band could finally be
# derived.


def _plate_family_frequencies(n: int, mu: float, modes, kappa: float = KAPPA_PLATE_DEFAULT):
    """``(f_discrete, f_continuum)`` for a square plate, analytically — no time-stepping.

    ``h = Lx/N`` and ``k = mu h^2 / kappa`` is ``make_plate``'s sample rate inverted, the same
    arithmetic ``test_plate_modal.py``'s convergence test does. Building an actual ``Plate`` at
    ``N = 512`` would assemble a quarter-million-unknown biharmonic to answer a question about
    its eigenvalues, which are closed-form.
    """
    h = L_DEFAULT / n
    k = mu * h * h / kappa
    lam = np.asarray(modal.rectangular_discrete_eigenvalues(h, n, n, modes))
    f_disc = np.asarray(modal.discrete_plate_eigenfrequency(lam, kappa, k, PLATE_THETA_DEFAULT))
    f_cont = np.asarray(modal.rectangular_plate_freqs(kappa, L_DEFAULT, L_DEFAULT, modes))
    return f_disc, f_cont


@pytest.mark.parametrize("n", [64, 256])
def test_the_plate_diagonal_family_IS_the_strings_spatial_droop_SQUARED(n):
    """The mechanism, as an exact identity with no timestep in it.

    A plate's frequency is proportional to the Laplacian eigenvalue rather than its square root,
    so where a string's spatial droop is ``sinc(u)`` the plate's is ``sinc(u)^2``. Along the
    **diagonal** that is exact rather than asymptotic: both axes carry the same ``u``, so
    ``Lambda_disc / Lambda_cont`` is ``sinc(u)^2`` to the last bit.

    The **axial** family only approaches it — ``(m, 1)`` carries an undrooped ``p_1^2`` in the
    numerator — and the gap closes like ``1/N^2``, which is why the two families end up sharing a
    floor anyway.
    """
    h = L_DEFAULT / n
    m = np.arange(1, n)
    u = m * np.pi / (2.0 * n)
    sinc_sq = (np.sin(u) / u) ** 2

    diagonal = np.asarray(
        modal.rectangular_discrete_eigenvalues(h, n, n, mode_family("diagonal", n - 1))
    ) / (2.0 * (m * np.pi / L_DEFAULT) ** 2)
    assert np.max(np.abs(diagonal - sinc_sq)) < 1e-14, (
        "the diagonal family's eigenvalue ratio is sinc(u)^2 exactly, not approximately"
    )

    axial = np.asarray(
        modal.rectangular_discrete_eigenvalues(h, n, n, mode_family("axial", n - 1))
    ) / ((m * m + 1) * (np.pi / L_DEFAULT) ** 2)
    gap = float(np.max(np.abs(axial - sinc_sq)))
    assert 0.0 < gap < 1e-3, f"the axial family should be near sinc^2, not equal to it: {gap:.2e}"


def test_the_plates_space_floor_is_the_strings_at_HALF_the_cents_bound():
    """The closed form, and it is an identity rather than a fit.

    ``sinc(u)^2 = 2^(-c/1200)`` is ``sinc(u) = 2^(-(c/2)/1200)``, so **a plate resolves the same
    share of its grid as a string given half the cents budget** — 5.925% against 8.378% at five
    cents, a factor of ``sqrt(2)`` in the small-``u`` limit. That is section 3.2's "twice the
    droop" written as a number, and it is checkable from outside any measurement.
    """
    for cents in (0.5, 1.0, 5.0, 25.0, 100.0):
        plate = sinc_horizon_fraction(cents, power=2)
        string_half = sinc_horizon_fraction(cents / 2.0, power=1)
        assert abs(plate - string_half) < 1e-12, (
            f"{cents:g} cents: plate fraction {plate:.15f} against the string's at half the "
            f"bound {string_half:.15f} — these are the same equation"
        )
        assert plate < sinc_horizon_fraction(cents, power=1), "the plate must resolve less"
    assert abs(sinc_horizon_fraction(5.0, power=2) - 0.059250) < 1e-5


@pytest.mark.parametrize("cents", [1.0, 5.0, 25.0])
@pytest.mark.parametrize("n", [64, 128, 256, 512])
@pytest.mark.parametrize("kind", ["diagonal", "axial"])
def test_the_plate_space_floor_matches_its_closed_form(kind, n, cents):
    """The plate's analogue of the string's section 3.1 bar: measured against ``sinc(u)^2``.

    ``mu`` is taken to nothing so this is the *floor* — what remains when the timestep is refined
    away — which for the implicit family is a floor no sample rate passes. The tolerance is one
    mode, the integer quantisation, and the second assertion fixes the **direction**: a timestep
    can only cost modes, never buy them, so the measurement may sit below the closed form and
    never above it.
    """
    modes = mode_family(kind, n - 1)
    horizon, monotone = pitch_horizon(*_plate_family_frequencies(n, 1e-5, modes), cents)
    assert monotone, f"the {kind} family's droop is monotone in the mode index"
    predicted = sinc_horizon_fraction(cents, power=2) * n
    assert abs(horizon - predicted) <= 1.0, (
        f"{kind}, N={n}, {cents:g} cents: measured floor {horizon} against the closed form "
        f"{predicted:.2f} — more than the integer quantisation apart"
    )
    assert horizon <= predicted, (
        f"{kind}, N={n}: measured {horizon} ABOVE the space floor {predicted:.2f}, which a "
        "timestep cannot buy"
    )


@pytest.mark.parametrize("cents", [1.0, 5.0, 25.0])
@pytest.mark.parametrize("n", [64, 128, 256, 512])
def test_the_plates_two_families_agree_in_INDEX_unlike_the_membranes(n, cents):
    """Section 7.4 predicted a factor of nine here by analogy with the membrane. There is none.

    The membrane's gap is a *cancellation*: the explicit scheme's sharp time error wipes out the
    spatial droop, and only along the diagonal at ``lambda = 1/sqrt(2)``. The implicit plate has
    no magic Courant number to cancel at, so both families sit on the same ``sinc^2`` floor and
    have the **same horizon in their own mode index** — at every grid and every bound.

    They are not the same modes: at index ``m`` the diagonal sits at roughly twice the frequency
    of the axial one, which is what the pitch test below is about.
    """
    horizons = {
        kind: pitch_horizon(
            *_plate_family_frequencies(n, 1e-5, mode_family(kind, n - 1)), cents
        )[0]
        for kind in ("diagonal", "axial")
    }
    assert horizons["diagonal"] == horizons["axial"], (
        f"N={n}, {cents:g} cents: the families should be identical at the floor, got {horizons}"
    )


def test_a_finite_timestep_breaks_the_family_tie_toward_the_AXIAL_one():
    """And the tie-break has a direction, which is why the agreement above is a ``k -> 0`` claim.

    At index ``m`` the axial mode sits at about half the diagonal's frequency, so it takes about
    half the time droop. Refine ``k`` away and the two coincide; raise it and the axial family
    reaches further. Anyone quoting "the plate's horizon" from one family at a working sample
    rate is therefore quoting the wrong one by a few modes.
    """
    n, cents = 512, 25.0
    coarse = {
        kind: pitch_horizon(
            *_plate_family_frequencies(n, 2.0, mode_family(kind, n - 1)), cents
        )[0]
        for kind in ("diagonal", "axial")
    }
    assert coarse["axial"] > coarse["diagonal"], (
        f"the time droop should cost the diagonal family more, got {coarse}"
    )
    floor = pitch_horizon(
        *_plate_family_frequencies(n, 1e-5, mode_family("diagonal", n - 1)), cents
    )[0]
    assert coarse["axial"] < floor, f"neither family may pass the space floor {floor}: {coarse}"


@pytest.mark.parametrize("m", [8, 16, 32])
def test_the_plates_families_differ_by_a_factor_of_two_in_PITCH(m):
    """Where the two families *do* differ: at the same frequency, an axial mode is twice as flat.

    The droop weight is ``(m^4 + n^4) / (m^2 + n^2)``, and at a fixed continuum frequency
    ``rho^2 = m^2 + n^2`` that is maximal on the axis (``rho^4``) and minimal on the diagonal
    (``rho^4 / 2``). So "the plate is in tune to 5 cents up to 2 kHz" is a claim about a
    *direction*: on the axis the same 5 cents is reached at ``1/sqrt(2)`` of that frequency.

    The ratio approaches two from below because the axial mode carries an undrooped ``n = 1``.
    Measured at ``k -> 0``; a finite timestep adds a family-independent droop at equal pitch and
    pulls the ratio toward one, which the second half asserts.
    """
    n = 512
    axial_m = int(round(m * np.sqrt(2.0)))  # about the same continuum frequency as (m, m)

    def pitch_normalised_ratio(mu):
        f_disc, f_cont = _plate_family_frequencies(n, mu, [(m, m), (axial_m, 1)])
        err = np.abs(pitch_error_cents(f_disc, f_cont))
        return float(err[1] / err[0] * (f_cont[0] / f_cont[1]))

    floor_ratio = pitch_normalised_ratio(1e-5)
    assert abs(floor_ratio - 2.0) < 0.05, (
        f"(m,m)=({m},{m}) against ({axial_m},1): the axial mode should be twice as flat at equal "
        f"pitch, got {floor_ratio:.4f}"
    )
    assert pitch_normalised_ratio(2.0) < floor_ratio, (
        "a finite timestep must pull the ratio toward one — it droops both families equally at "
        "equal pitch"
    )


@pytest.mark.parametrize("n", [96, 256])
def test_the_sorted_2d_spectrum_is_not_monotone_so_its_prefix_is_not_a_horizon(n):
    """The reason a 2-D band must be split at all, with its witness named.

    Sorted by frequency, the plate's spectrum interleaves the families and the error does not
    rise with pitch: ``(3,1)`` is *lower* in frequency than ``(2,3)`` and *worse* in error,
    because the weight ``(m^4 + n^4)/(m^2 + n^2)`` orders differently from ``m^2 + n^2``.
    ``pitch_horizon`` still returns an integer over such a list, and its ``monotone`` flag is the
    only thing saying that integer is meaningless — which is what the flag exists for.
    """
    modes = mode_block(8)
    f_disc, f_cont = _plate_family_frequencies(n, 0.5, modes)
    _, monotone = pitch_horizon(f_disc, f_cont, CENTS)
    assert not monotone, (
        "a frequency-sorted 2-D spectrum is not monotone in pitch error; if this passes, the "
        "block reading in test_plate_modal.py is resting on the wrong reason"
    )
    err = np.abs(pitch_error_cents(f_disc, f_cont))
    fall = int(np.nonzero(np.diff(err) < 0)[0][0])
    assert modes[fall] == (3, 1) and modes[fall + 1] == (2, 3), (
        f"the first fall should be the named witness, got {modes[fall]} -> {modes[fall + 1]}"
    )
    assert err[fall] > err[fall + 1]


@pytest.mark.parametrize("n", [96, 256])
def test_a_2d_blocks_worst_mode_is_its_DIAGONAL_corner(n):
    """What makes a block readable despite the test above: its horizon is a family's horizon.

    A "the first few modes are in tune" claim asserts over an index **block**, not a family. The
    weight ``(m^4 + n^4)/(m^2 + n^2)`` has an interior minimum in ``n``, so its maximum over a
    block sits at a corner — and the diagonal corner ``(M, M)`` beats the axial ``(M, 1)`` for
    every ``M >= 2``. So a block's horizon *is* its diagonal family's, and that has a prefix.
    """
    for m_max in range(2, 9):
        modes = mode_block(m_max)
        err = np.abs(pitch_error_cents(*_plate_family_frequencies(n, 0.5, modes)))
        worst = modes[int(np.argmax(err))]
        assert worst == (m_max, m_max), (
            f"the {m_max}x{m_max} block's worst mode should be its diagonal corner, got {worst}"
        )
