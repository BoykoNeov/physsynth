"""Free-free beam (model #5b-pre): the energy-first free-edge operator vs the closed-form oracle.

This is the de-risk for the free-edge Chladni plate
(``docs/dev/plate-free-edge-plan.md`` Part 0). The
free-free Euler–Bernoulli beam is the one free-boundary flexural model with a *genuine* closed-form
spectrum (``cos(βL)·cosh(βL) = 1``), so it pins the free-end stencil + operator symmetry
**before** the
2D corners and Poisson term enter. Unlike the simply-supported plate there is no exact ``sin``
eigenvector — the bar is the **O(h²) tier** (operator eigenvalues converging at order 2),
not machine
precision. The machine-precision anchor here is the rigid-body nullspace ``{1, x}``.
"""

import numpy as np
from helpers import (
    KAPPA_BEAM_DEFAULT,
    beam_low_eigenfrequencies,
    convergence_orders,
    make_beam,
)

from physsynth.analysis import modal, spectrum
from physsynth.core.engine import simulate
from physsynth.core.exciter import raised_cosine

KAPPA = KAPPA_BEAM_DEFAULT


# -- Operator symmetry: the energy-first K = h D2ᵀD2 is symmetric by construction
# (=> conservation). --
def test_operator_symmetric():
    beam = make_beam(N=100, mu=1.0)
    asym = abs(beam.K - beam.K.T).max()
    assert asym < 1e-12, f"K not symmetric: ||K-K^T||_max = {asym:.3e}"


# -- Money test: the rigid-body nullspace {1, x} (machine precision) + x² counter-check
# (nonzero). --
def test_rigid_body_nullspace():
    """``K{1,x}=0`` to machine precision; ``K x² ≠ 0``. Tolerances are RELATIVE: the
    absolute residual
    of ``K@x`` scales as ``eps·‖K‖ ~ eps/h³`` (grows as h shrinks), so an absolute bar would
    be flaky
    and N-dependent. The discriminating signal is the *contrast* — ``K{1,x}`` is ~12 orders below
    ``K x²``."""
    beam = make_beam(N=100, mu=1.0)
    K = beam.K
    k_fro = np.sqrt((K.toarray() ** 2).sum())  # ‖K‖_F, the scale residuals are measured against
    x = beam.x

    rel = {
        name: np.linalg.norm(K @ v) / (k_fro * np.linalg.norm(v))
        for name, v in (("1", np.ones_like(x)), ("x", x), ("x2", x * x))
    }
    assert rel["1"] < 1e-12, f"K@1 not in nullspace: rel residual {rel['1']:.2e}"
    assert rel["x"] < 1e-12, f"K@x not in nullspace: rel residual {rel['x']:.2e}"
    # x² has real curvature energy: a wrong operator that killed it would be the tell of a dropped
    # stiffness term (the 1D rehearsal of the 2D dropped-(1-ν) catch). Demand many orders of
    # contrast.
    assert rel["x2"] > 1e-9, f"K@x² spuriously ~0: rel {rel['x2']:.2e}"
    assert rel["x2"] > 1e6 * max(rel["1"], rel["x"]), "nullspace contrast too small"


# -- Closed-form oracle: low spatial eigenfrequencies match cos(βL)cosh(βL)=1
# (tight at fine N). --
def test_modal_frequencies_match_closed_form():
    beam = make_beam(N=200, mu=0.5)  # fine grid + fine timestep -> the tight (O(h²)) regime
    measured, rigid = beam_low_eigenfrequencies(beam, 5, return_rigid=True)
    oracle = modal.free_free_beam_freqs(beam.kappa, beam.L, 5)

    # The 2 discarded eigenvalues are genuinely ~0 (a free cross-check that the nullspace is wired
    # right). Scale: mu_1 ~ (β_1/L)^4 ~ 500 for L=1, so 1e-6*mu1 is a generous "is zero" bar.
    mu1_est = (4.730041 / beam.L) ** 4
    assert np.max(np.abs(rigid)) < 1e-6 * mu1_est, f"rigid-body modes not ~0: {rigid}"

    err_cents = np.abs(modal.cents(measured, oracle))
    assert err_cents[0] < 0.5, f"fundamental off by {err_cents[0]:.3f} cents (want < 0.5)"
    assert np.max(err_cents[:4]) < 2.0, f"low-4 modes off by {np.max(err_cents[:4]):.3f} cents"


# -- The operator eigenvalues converge at O(h²) (the real rigor; no external table needed). --
def test_convergence_order_h2():
    Ns = [50, 100, 200, 400]  # errors stay well above the ~1e-8 noise floor at the finest grid
    oracle = modal.free_free_beam_freqs(KAPPA, 1.0, 3)
    hs, err_fund, err_low3 = [], [], []
    for N in Ns:
        beam = make_beam(N=N, mu=1.0)
        f = beam_low_eigenfrequencies(beam, 3)
        hs.append(beam.h)
        err_fund.append(abs(f[0] - oracle[0]))
        err_low3.append(np.max(np.abs(f - oracle)))
    hs = np.array(hs)
    assert np.all(np.diff(err_fund) < 0), f"fundamental error not decreasing: {err_fund}"
    order_fund = convergence_orders(np.array(err_fund), hs)
    assert order_fund[-1] > 1.9, f"fundamental convergence order {order_fund[-1]:.2f} < 1.9"
    order_low3 = convergence_orders(np.array(err_low3), hs)
    assert order_low3[-1] > 1.9, f"low-mode convergence order {order_low3[-1]:.2f} < 1.9"


# -- End-to-end: the time-stepper actually rings at the discrete fundamental (FFT sanity). --
def test_fft_rings_at_fundamental():
    beam = make_beam(N=120, mu=1.0)
    # Spatial fundamental -> implied 4th-power eigenvalue mu -> theta-scheme time-discrete
    # frequency.
    f_spatial = beam_low_eigenfrequencies(beam, 2)
    mu = (2.0 * np.pi * f_spatial / beam.kappa) ** 2
    f_disc = modal.discrete_beam_eigenfrequency(mu, beam.kappa, beam.k, beam.theta)

    # Off-centre bump (breaks symmetry -> excites the fundamental); free ends are bending antinodes,
    # so a pickup near an end has strong modal content.
    beam.set_state(raised_cosine(beam.x, beam.L, 0.3 * beam.L, 0.15 * beam.L, amplitude=1e-3))
    pickup = int(0.12 * beam.N)
    res = simulate(beam, num_steps=int(0.5 * beam.fs), pickup_index=pickup)
    found = spectrum.measure_partials_near(
        res.output, res.fs, np.array([f_disc[0]]), search_hz=15.0
    )[0]
    cents = abs(modal.cents(found, f_disc[0]))
    assert cents < 5.0, (
        f"FFT fundamental off by {cents:.2f} cents "
        f"(found {found:.2f}, want {f_disc[0]:.2f})"
    )


# -- The standalone operator helper matches what the resonator builds (single source of truth). --
def test_resonator_uses_operator_helper():
    from physsynth.core.operators import free_beam_stiffness

    beam = make_beam(N=40, mu=1.0)
    K, W = free_beam_stiffness(beam.N, beam.h)
    assert abs(K - beam.K).max() < 1e-12, "resonator K != free_beam_stiffness K"
    assert abs(W - beam.W).max() < 1e-12, "resonator W != free_beam_stiffness W"
