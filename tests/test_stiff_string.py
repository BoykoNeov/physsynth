"""Validation suite for the stiff string (model #2, implicit theta-scheme).

Mirrors the ideal-string harness (energy / passivity / modal / convergence / dispersion) and adds
the stiff string's defining check: the partials are *stretched* off the harmonic series by bending
stiffness, ``f_n = n f0 sqrt(1 + B n^2)`` with ``B = pi^2 kappa^2 / (c^2 L^2)``. See
``docs/dev/stiff-string-plan.md`` and HANDOFF section 5 / section 6.

Test decomposition (the honest split, per the dispersion-test precedent):

* **Implementation** -- ``step()`` reproduces the scheme's own closed-form *discrete* oracle to
  ~1e-3 cent (tight, regardless of grid).
* **Physics anchor** -- the discrete oracle tends to the continuum stretched law as the grid
  refines (a *pure* computation, no simulation); the stretch is real and matches theory.
* **Money test** -- ``B`` fitted from *simulated* partials across a kappa sweep tracks
  ``pi^2 kappa^2 / (c^2 L^2)`` quantitatively (sign + scale), the strongest evidence the new term
  is right.

Tolerances inherit Milestone 1's bar (drift < 1e-10; partials ~1 cent; dispersion oracle ~1e-4),
set a few times looser than the observed clean residuals so a real regression trips them.
"""

import numpy as np
import pytest
from helpers import (
    KAPPA_DEFAULT,
    L_DEFAULT,
    convergence_orders,
    make_stiff_string,
    measure_stiff_mode_frequencies,
    wave_speed,
)

from physsynth.analysis import dispersion, modal, spectrum
from physsynth.core.engine import simulate
from physsynth.core.exciter import triangular_pluck
from physsynth.core.operators import biharmonic_matrix, delta_xxxx, second_difference_matrix
from physsynth.core.string_stiff import THETA_DEFAULT, StiffString

DRIFT_TOL = 1e-10  # acceptance criterion 1 (inherited)
ONE_CENT = 1.0


def _pluck_run(s, secs=2.0, pluck_frac=0.137, pickup_frac=0.241):
    s.set_state(triangular_pluck(s.x, s.L, pluck_frac * s.L, amplitude=1e-3))
    steps = int(secs * s.fs)
    return simulate(s, num_steps=steps, pickup_index=int(round(pickup_frac * s.N)))


# =====================================================================================
# Energy: conservation (lossless) and passivity (lossy)  -- HANDOFF section 6.1-6.2
# =====================================================================================


def test_energy_conserved_quick():
    # One representative lossless case kept out of the slow sweep so a gross energy-form regression
    # trips in the fast suite.
    s = make_stiff_string(N=100, lam=1.0, kappa=KAPPA_DEFAULT)
    res = _pluck_run(s, secs=1.0)
    assert res.energy_drift < DRIFT_TOL, f"drift {res.energy_drift:.2e}"
    assert np.all(res.energy > 0.0)


@pytest.mark.slow
@pytest.mark.parametrize("kappa", [0.0, 0.5, 2.0, 5.0])
@pytest.mark.parametrize("lam", [1.0, 0.5, 2.0, 4.0])  # lam > 1 is admissible (unconditional)
def test_energy_conserved_across_kappa_and_lambda(kappa, lam):
    # The conservation identity is algebraic, so it must hold for every kappa and every lambda --
    # *including* coarse-grid / large-timestep regimes (lam = 2, 4) that the explicit stiff scheme
    # could not run. That is the unconditional-stability win, asserted directly.
    s = make_stiff_string(N=100, lam=lam, kappa=kappa)
    res = _pluck_run(s, secs=2.0)
    assert res.energy_drift < DRIFT_TOL, f"drift {res.energy_drift:.2e} (kappa={kappa}, lam={lam})"
    assert np.all(np.isfinite(res.energy))


@pytest.mark.slow
@pytest.mark.parametrize("theta", [0.25, THETA_DEFAULT, 0.5])
def test_energy_conserved_across_theta(theta):
    # Conservation is exact for *every* theta > 0 (positivity margin is a separate property): a
    # clean bisection -- drift here is an energy-form bug, isolated from the biharmonic boundary.
    s = make_stiff_string(N=120, lam=1.0, kappa=KAPPA_DEFAULT, theta=theta)
    res = _pluck_run(s, secs=2.0)
    assert res.energy_drift < DRIFT_TOL, f"drift {res.energy_drift:.2e} at theta={theta}"
    # theta = 1/4 is the zero-positivity-margin case (the stabilizer term vanishes); lock in that
    # the energy stays strictly positive there for a real pluck, not just conserved.
    assert np.all(res.energy > 0.0), f"energy went non-positive at theta={theta}"


@pytest.mark.slow
def test_passivity_monotonic_decrease():
    # With loss, energy never increases (passivity is unconditional: rho^2 <= 1 for sigma >= 0),
    # for a broadband pluck that excites high modes too.
    s = make_stiff_string(N=100, lam=1.0, kappa=KAPPA_DEFAULT, sigma=5.0)
    res = _pluck_run(s, secs=2.0)
    e0 = res.energy[0]
    steps_up = np.diff(res.energy)
    assert np.all(steps_up <= 1e-12 * e0), f"max positive step {steps_up.max():.2e}"


@pytest.mark.slow
def test_decay_rate_matches_2sigma_low_mode():
    # Frequency-independent loss decays a mode as E ~ E0 exp(-2 sigma t). The discrete theta-scheme
    # decays mode m at 2 sigma (1 - theta Q k^2); for a LOW mode Q k^2 << 1 so the rate is 2 sigma
    # to a fraction of a percent even with stiffness. (High modes under-damp HARD -- theta Q k^2
    # reaches O(10) with stiffness, so a broadband pluck is far off 2 sigma; a documented scheme
    # property cured by model #3, see the plan's "Damping caveat".) Checked vs the continuous rate.
    sigma, secs = 4.0, 1.0
    s = make_stiff_string(N=100, lam=1.0, kappa=KAPPA_DEFAULT, sigma=sigma)
    s.set_state(modal.mode_shape(s.x, s.L, 1) * 1e-3)
    res = simulate(s, num_steps=int(secs * s.fs))
    measured = res.energy[-1] / res.energy[0]
    expected = np.exp(-2.0 * sigma * secs)
    rel = abs(np.log(measured) - np.log(expected)) / abs(np.log(expected))
    assert rel < 0.01, f"decay rate off by {rel:.3%} (got {measured:.3e}, want {expected:.3e})"


# =====================================================================================
# Stretched partials -- the stiff string's defining behavior  -- HANDOFF section 6.3
# =====================================================================================


@pytest.mark.slow
def test_partials_match_discrete_oracle():
    # Implementation test: every detected partial of a plucked stiff string lands on the scheme's
    # own closed-form discrete oracle (which depends on theta). Tight -- this is what proves step()
    # reproduces the intended dispersion.
    c, L = wave_speed(), L_DEFAULT
    s = make_stiff_string(N=128, lam=1.0, kappa=KAPPA_DEFAULT)
    res = _pluck_run(s, secs=2.0)
    n_partials = 8
    oracle = np.array(
        [modal.discrete_stiff_mode_frequency(c, L, s.N, s.kappa, s.k, m, s.theta)
         for m in range(1, n_partials + 1)]
    )
    detected = spectrum.measure_partials_near(res.output, res.fs, oracle)
    assert not np.any(np.isnan(detected)), "a partial was not detected"
    err = np.abs(modal.cents(detected, oracle))
    assert np.max(err) < 0.05, f"worst partial-vs-oracle error {np.max(err):.4f} cents"


def test_discrete_oracle_converges_to_continuum_stretched_law():
    # Physics anchor (no simulation): on a fine grid the discrete oracle sits on the continuum
    # stretched law f_n = n f0 sqrt(1 + B n^2) to well under a cent for the first 10 partials, and
    # the fundamental is itself stretched (f1 = f0 sqrt(1+B), NOT f0).
    c, L, kappa = wave_speed(), L_DEFAULT, KAPPA_DEFAULT
    N, fs = 4000, 8.0e5
    k = 1.0 / fs
    oracle = np.array(
        [modal.discrete_stiff_mode_frequency(c, L, N, kappa, k, m, THETA_DEFAULT)
         for m in range(1, 11)]
    )
    continuum = modal.stiff_harmonic_frequencies(c, L, kappa, 10)
    assert np.max(np.abs(modal.cents(oracle, continuum))) < ONE_CENT

    f0 = c / (2.0 * L)
    B = modal.inharmonicity_B(c, L, kappa)
    # The continuum oracle stretches even the fundamental off f0 by sqrt(1+B).
    assert continuum[0] == pytest.approx(f0 * np.sqrt(1.0 + B), rel=1e-12)
    assert continuum[0] > f0  # genuinely sharp of the ideal fundamental
    assert oracle[0] == pytest.approx(continuum[0], rel=1e-4)


@pytest.mark.slow
def test_B_tracks_kappa_squared():
    # The money test: fit B from SIMULATED partials across a kappa sweep and check it tracks
    # B = pi^2 kappa^2 / (c^2 L^2) in sign AND scale. Numerical dispersion biases the fit slightly
    # LOW (it pulls high partials flat, opposite to bending's sharpening), so on this fine grid the
    # fit lands a few percent under B_true -- one-directional and well understood. The
    # proportionality B ~ kappa^2 and absolute scale rule out a sign/scale error in the new term.
    c, L = wave_speed(), L_DEFAULT
    f0 = c / (2.0 * L)
    kappas = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    n_partials = 10
    n = np.arange(1, n_partials + 1)

    B_fit, B_true = [], []
    for kappa in kappas:
        s = make_stiff_string(N=512, lam=1.0, kappa=kappa)
        res = _pluck_run(s, secs=1.0)
        oracle = np.array(
            [modal.discrete_stiff_mode_frequency(c, L, s.N, kappa, s.k, m, s.theta)
             for m in range(1, n_partials + 1)]
        )
        det = spectrum.measure_partials_near(res.output, res.fs, oracle)
        assert not np.any(np.isnan(det)), f"a partial undetected at kappa={kappa}"
        # (f_n / (n f0))^2 - 1 = B n^2  ->  least-squares slope through the origin.
        y = (det / (n * f0)) ** 2 - 1.0
        B_fit.append(float(np.sum(n**2 * y) / np.sum(n**4)))
        B_true.append(modal.inharmonicity_B(c, L, kappa))
    B_fit, B_true = np.array(B_fit), np.array(B_true)

    # (a) Sign + absolute scale: fitted B within a few % of theory, biased low (never sharp of it).
    ratio = B_fit / B_true
    assert np.all(ratio > 0.92) and np.all(ratio < 1.02), f"B_fit/B_true = {ratio}"
    # (b) Monotone: more stiffness -> more inharmonicity.
    assert np.all(np.diff(B_fit) > 0.0), f"B_fit not increasing with kappa: {B_fit}"
    # (c) Proportionality B ~ kappa^2 (where bending dominates the small flat bias, kappa >= 2):
    #     B_fit / kappa^2 is constant, and doubling kappa quadruples B.
    coeff = B_fit[1:] / kappas[1:] ** 2  # kappa = 2..5
    assert coeff.max() / coeff.min() < 1.03, f"B/kappa^2 not constant: {coeff}"
    assert B_fit[3] / B_fit[1] == pytest.approx(4.0, rel=0.03)  # kappa 2 -> 4


# =====================================================================================
# Convergence (criterion 3) and dispersion (criterion 5)
# =====================================================================================


@pytest.mark.slow
def test_second_order_convergence():
    # Unlike the ideal string at lam = 1 (exact), the stiff string disperses at every lambda, so the
    # partial error vs the continuum stretched law has signal to refine. Halving h shrinks it ~4x
    # (O(h^2); the theta time scheme is also 2nd order).
    c, L, kappa = wave_speed(), L_DEFAULT, KAPPA_DEFAULT
    lam, mode, grids = 0.9, 4, (64, 128, 256)
    f_cont = modal.stiff_harmonic_frequencies(c, L, kappa, mode)[-1]

    errors, step_sizes = [], []
    for N in grids:
        f_det = measure_stiff_mode_frequencies(
            np.array([mode]), N=N, lam=lam, kappa=kappa, secs=0.6
        )[0]
        errors.append(abs(f_det - f_cont))
        step_sizes.append(L / N)
    errors = np.array(errors)
    orders = convergence_orders(errors, np.array(step_sizes))

    assert np.all(np.diff(errors) < 0.0), f"errors did not shrink: {errors}"
    assert np.all(orders > 1.7), f"a refinement step fell below 2nd order: {orders}"
    assert 1.85 < orders.mean() < 2.15, f"mean order {orders.mean():.3f} not ~2"


@pytest.mark.slow
def test_dispersion_matches_stiff_oracle_and_stiffens_high_partials():
    # Measured FDTD frequency of each swept mode lands on the discrete stiff oracle, and the phase
    # velocity RISES above c with mode number -- bending stiffens high partials, the opposite of the
    # ideal string's numerical droop. (lam = 0.8 keeps the curve cleanly monotone; at lam closer to
    # 1 the implicit scheme's own top-mode flattening competes with the bending rise.)
    c, L, kappa = wave_speed(), L_DEFAULT, KAPPA_DEFAULT
    N, lam = 256, 0.8
    modes = np.array([2, 4, 8, 16, 32, 48, 64])
    measured = measure_stiff_mode_frequencies(modes, N=N, lam=lam, kappa=kappa, secs=0.4)
    s = make_stiff_string(N=N, lam=lam, kappa=kappa)
    oracle = dispersion.stiff_dispersion_frequencies(c, L, N, kappa, s.k, s.theta, modes)

    assert not np.any(np.isnan(measured)), "a mode frequency was not detected"
    rel = np.abs(measured - oracle) / oracle
    worst = modes[np.argmax(rel)]
    assert np.max(rel) < 1e-4, f"worst measured-vs-oracle {np.max(rel):.2e} at mode {worst}"

    vp_over_c = dispersion.phase_velocity(measured, L, modes) / c
    assert np.all(vp_over_c > 1.0), f"phase velocity not above c (no stiffening): {vp_over_c}"
    assert np.all(np.diff(vp_over_c) > 0.0), f"phase velocity not rising with mode: {vp_over_c}"


# =====================================================================================
# Stability / construction guards (criterion 4) and the kappa=0 self-consistency anchor
# =====================================================================================


@pytest.mark.slow
@pytest.mark.parametrize("lam", [4.0, 2.0, 1.0, 0.5])
def test_no_nan_including_explicit_forbidden_lambda(lam):
    # No CFL limit: configurations that violate the explicit biharmonic bound (lam >= 1, large
    # kappa) must run stably to completion. The headline unconditional-stability assertion.
    s = make_stiff_string(N=100, lam=lam, kappa=5.0)
    res = _pluck_run(s, secs=0.5)
    assert np.all(np.isfinite(res.output))
    assert np.all(np.isfinite(res.energy))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"rho": -1.0},
        {"T": 0.0},
        {"L": -2.0},
        {"kappa": -0.1},
        {"sigma": -0.1},
        {"N": 1},
        {"theta": 0.0},
        {"theta": 1.5},
        {"boundary": "clamped"},
    ],
)
def test_invalid_parameters_rejected(kwargs):
    base = {"L": 1.0, "T": 200.0, "rho": 0.005, "fs": 20000.0, "N": 100}
    base.update(kwargs)
    with pytest.raises(ValueError):
        StiffString(**base)


def test_lambda_above_one_accepted():
    # Unlike IdealString, the implicit scheme must NOT reject lam > 1.
    s = make_stiff_string(N=100, lam=2.5, kappa=KAPPA_DEFAULT)
    assert s.lam == pytest.approx(2.5, rel=1e-9)


def test_kappa_zero_is_self_consistent_not_ideal_string():
    # Self-consistency anchor (NOT equality with IdealString): at kappa = 0 the StiffString is the
    # implicit theta-scheme, a *different* scheme from the explicit IdealString (not exact even at
    # lam = 1). Assert (a) its partials match its OWN oracle to machine precision, and (c) it only
    # agrees with IdealString loosely, in the low-mode limit. (Energy conservation -- (b) -- is
    # covered by the energy tests.)
    c, L, N = wave_speed(), L_DEFAULT, 128
    s = make_stiff_string(N=N, lam=1.0, kappa=0.0)

    # (a) kappa = 0 reduces to the implicit wave oracle s = lam^2 sin^2 / (1 + 4 theta lam^2 sin^2),
    #     which is strictly BELOW the explicit ideal oracle for theta > 0 (the implicit scheme is
    #     not exact at lam = 1). Check the closed form, vectorised vs scalar.
    for m in (1, 5, 20):
        f_stiff0 = modal.discrete_stiff_mode_frequency(c, L, N, 0.0, s.k, m, s.theta)
        f_ideal = modal.discrete_mode_frequency(c, L, N, 1.0, m)  # explicit, exact at lam=1
        assert f_stiff0 < f_ideal  # implicit theta-scheme lies below
        # closed form: s = lam^2 sin^2 / (1 + 4 theta lam^2 sin^2), lam = 1 here
        sin2 = np.sin(m * np.pi / (2 * N)) ** 2
        s_val = sin2 / (1.0 + 4.0 * s.theta * sin2)
        assert f_stiff0 == pytest.approx(np.arcsin(np.sqrt(s_val)) / (np.pi * s.k), rel=1e-12)

    # (c) agreement with IdealString only in the refined / low-mode limit, at a loose tolerance.
    for m in (1, 2, 4):
        f_stiff0 = modal.discrete_stiff_mode_frequency(c, L, N, 0.0, s.k, m, s.theta)
        f_ideal = modal.discrete_mode_frequency(c, L, N, 1.0, m)
        assert abs(f_stiff0 - f_ideal) / f_ideal < 1e-2  # close, but NOT machine precision


# =====================================================================================
# Fast unit tests: operators and oracles (no simulation)
# =====================================================================================


def test_biharmonic_matrix_structure():
    # Built as (delta_xx)^2: interior rows are the 5-point [1,-4,6,-4,1] stencil; the boundary-
    # adjacent diagonal is 5/h^4 (NOT the interior 6), the simply-supported ghost reflection. And
    # sin(m pi l / N) is an exact eigenvector with eigenvalue p^4.
    N, h = 12, 0.1
    D4 = biharmonic_matrix(N, h).toarray()
    assert np.allclose(D4[0, :5] * h**4, [5.0, -4.0, 1.0, 0.0, 0.0])  # boundary row: 5, not 6
    assert np.allclose(D4[2, :6] * h**4, [1.0, -4.0, 6.0, -4.0, 1.0, 0.0])  # interior row
    assert np.allclose(D4, D4.T)  # symmetric (SBP)
    ell = np.arange(1, N)
    for m in (1, 3, 5):
        phi = np.sin(m * np.pi * ell / N)
        p4 = ((4.0 / h**2) * np.sin(m * np.pi / (2 * N)) ** 2) ** 2
        assert np.allclose(D4 @ phi, p4 * phi, rtol=1e-9)


def test_biharmonic_matrix_matches_pure_operator_in_interior():
    # The matrix's interior rows must reproduce the pure delta_xxxx stencil away from the boundary.
    N, h = 16, 0.05
    D4 = biharmonic_matrix(N, h).toarray()
    rng = np.random.default_rng(0)
    u = rng.standard_normal(N + 1)
    u[0] = u[-1] = 0.0
    full = D4 @ u[1:-1]  # interior nodes 1..N-1
    stencil = delta_xxxx(u, h)  # nodes 2..N-2
    assert np.allclose(full[1:-1], stencil, rtol=1e-10)


def test_second_difference_matrix_eigenvalues():
    N, h = 20, 0.05
    D2 = second_difference_matrix(N, h).toarray()
    ell = np.arange(1, N)
    for m in (1, 4, 9):
        phi = np.sin(m * np.pi * ell / N)
        p2 = (4.0 / h**2) * np.sin(m * np.pi / (2 * N)) ** 2
        assert np.allclose(D2 @ phi, -p2 * phi, rtol=1e-10)


def test_inharmonicity_B_and_stretched_law():
    c, L = wave_speed(), L_DEFAULT
    assert modal.inharmonicity_B(c, L, 0.0) == 0.0
    # B scales as kappa^2.
    assert modal.inharmonicity_B(c, L, 2.0) == pytest.approx(4.0 * modal.inharmonicity_B(c, L, 1.0))
    # kappa = 0 recovers the exact harmonic series.
    harm = modal.stiff_harmonic_frequencies(c, L, 0.0, 6)
    assert np.allclose(harm, modal.harmonic_frequencies(c, L, 6))
    # kappa > 0: every partial sharp of n f0, increasingly so (the stretch grows with n).
    f0 = c / (2.0 * L)
    stiff = modal.stiff_harmonic_frequencies(c, L, KAPPA_DEFAULT, 6)
    n = np.arange(1, 7)
    stretch = stiff / (n * f0)  # sqrt(1 + B n^2), strictly increasing and > 1
    assert np.all(stretch > 1.0)
    assert np.all(np.diff(stretch) > 0.0)


def test_discrete_stiff_oracle_tends_to_continuum_on_refinement():
    # As N, fs grow the discrete oracle approaches the continuum stretched value (a fixed mode).
    c, L, kappa, m = wave_speed(), L_DEFAULT, KAPPA_DEFAULT, 3
    cont = modal.stiff_harmonic_frequencies(c, L, kappa, m)[-1]
    errs = []
    for N in (256, 512, 1024):
        fs = c * N / L  # lam = 1
        f = modal.discrete_stiff_mode_frequency(c, L, N, kappa, 1.0 / fs, m, THETA_DEFAULT)
        errs.append(abs(f - cont))
    assert errs[0] > errs[1] > errs[2]  # monotone convergence


def test_stiff_dispersion_frequencies_match_scalar_oracle():
    c, L, N, kappa, theta = wave_speed(), L_DEFAULT, 128, KAPPA_DEFAULT, THETA_DEFAULT
    k = L / (c * N)  # lam = 1
    modes = np.array([1, 5, 10, 25, 50])
    vec = dispersion.stiff_dispersion_frequencies(c, L, N, kappa, k, theta, modes)
    scalar = np.array(
        [modal.discrete_stiff_mode_frequency(c, L, N, kappa, k, int(m), theta) for m in modes]
    )
    assert np.allclose(vec, scalar, rtol=1e-15)
    # Stiff phase velocity rises above c (contrast the ideal string, where it droops below).
    vp = dispersion.phase_velocity(vec, L, modes) / c
    assert np.all(vp > 1.0)
