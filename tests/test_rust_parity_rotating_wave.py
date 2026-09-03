"""The rotating-wave BVP, Python against Rust -- and the first port whose *answer* is a root.

Every parity file before this one compares arithmetic: the two sides evaluate the same expressions
and the question is over what window the last bits stay together. This module compares a
**converged root**, which is a different question with a much better answer and one sharp caveat.

**The better answer.** Newton's step is a means; the residual defines the root. Measured before the
port was written, by perturbing this module's own Newton step in Python: a *relative 1e-10* nudge --
six orders beyond anything two LU implementations can differ by -- moves ``Omega`` by one ulp. So
the two implementations do not need to agree on their arithmetic to agree on their answer, and
across 126 fixtures they agree on ``Omega`` to at worst **8.4e-15**, far inside the tolerance
``docs/dev/rust-migration-plan.md`` §24 grants Group D.

**The caveat, which cost a wrong sentence in the Rust module's header before this file caught it.**
That perturbation probe held the fixture fixed at ``N = 32``, mode 1, ``kappa = 0``, and reported
the *iteration count* unmoved. True -- and a claim about that one fixture. Compared over the grid
below, the count differs in **17 of the 108 fixtures that converge**, by as much as 13. It is
harmless (the widest gap, 28 against 41, still agrees on ``Omega`` to 9.0e-16) but it is real, so
this file asserts the root and deliberately does **not** assert the count. The general form is worth
more than the instance: *a margin measured at one fixture is a claim about one fixture.*

**Where the BVP does not converge, nothing is compared but the flag.** Eighteen of the 126 fail,
they are the same eighteen on both sides, and past that point the two answers diverge freely --
``dphi`` reaches 0.5. That is correct rather than alarming: ``converged is False`` means "the helix
is not a relative equilibrium; do not treat this as physics", and it is the same lesson
``docs/dev/scientific-hurdles.md`` §6 landed on the same day, that a margin measured in the
converged regime says nothing about the failing one.

**Two last-bit divergences are reported and not chased, and the reason is a catalogued trap.** Both
come from §16.2's finding that a NumPy power is not a multiplication: ``planar_hessian_cells``
spells ``1/lam**3`` where the Rust spells ``1/(l*l*l)`` (NumPy's ``**3`` calls ``pow``; measured, it
disagrees with ``lam*lam*lam`` on 2 of 5 ordinary values), and ``kc_circular_frequency`` squares a
Python *scalar*, which goes through libm's ``pow`` rather than a multiply. Matching either would
mean writing ``powf(3.0)`` / ``powf(2.0)`` in Rust -- which is precisely the spelling §17.2 records
turning CI red, because LLVM folds a constant exponent back into a multiply in **release** and the
assertion then tests nothing in the profile that ships. One ulp on a tolerance-ported limit oracle
does not buy that. So both are bounded here at a few ulps and their cause is named.
"""

from __future__ import annotations

import numpy as np
import pytest

import physsynth.analysis.rotating_wave as RW

rs = pytest.importorskip("physsynth_rs")

L, T, RHO, EA, THETA = 1.0, 200.0, 0.005, 1.0e5, 0.28
"""The suite's own string: c = 200 m/s, EA/T = 500, and the theta the stiff string defaults to."""


def _fs(n: int, lam_long: float = 0.5) -> float:
    """``fs`` from the *longitudinal* Courant number -- the knob this model insists on."""
    return float(np.sqrt(EA / RHO)) * n / (L * lam_long)


def _both(*, n=32, amp=5e-3, mode=1, kappa=0.0, time_discrete=True, steps=8, maxiter=50):
    """Solve the same BVP twice: the Python original, then the Rust binding directly.

    Deliberately *not* routed through the swap. This file builds both sides itself, so it asserts
    what it says it asserts whether or not ``PHYSSYNTH_RS_ANALYSIS`` is set in the environment.
    """
    fs = _fs(n)
    py = RW.solve_rotating_wave_py(
        L=L, T=T, rho=RHO, EA=EA, fs=fs, N=n, theta=THETA, amplitude=amp, mode=mode,
        kappa=kappa, time_discrete=time_discrete, continuation_steps=steps, maxiter=maxiter,
    )
    phi, psi, stretch, tension, scalars = rs.rotating_wave_solve(
        L, T, RHO, EA, fs, n, THETA, amp, mode, kappa, time_discrete, steps,
        RW.NEWTON_TOL_DEFAULT, maxiter,
    )
    return py, (phi, psi, stretch, tension, scalars)


def _rel(a: float, b: float) -> float:
    return abs(a - b) / abs(b) if b else abs(a - b)


def _rel_field(a, b) -> float:
    """Field disagreement normalised by the field's own **amplitude**, never pointwise.

    A profile passes through zero at both clamped ends and near them, where a pointwise ratio
    reports a spurious 1.0 that looks exactly like a catastrophic bug -- the normalisation lesson
    the airbox batch paid for.
    """
    a, b = np.asarray(a), np.asarray(b)
    scale = float(np.max(np.abs(b)))
    return float(np.max(np.abs(a - b))) / scale if scale else float(np.max(np.abs(a - b)))


# -- the converged root -------------------------------------------------------------------


def test_the_canonical_helix_agrees_on_every_field():
    """The test suite's own fixture, field by field. This is the one the oracle is used at."""
    py, (phi, psi, stretch, tension, scalars) = _both()
    omega, frequency, s, amplitude, mode, shape_residual, iterations, converged, td, _, _ = scalars

    assert converged and py.converged
    assert _rel(omega, py.Omega) < 1e-13, f"Omega {omega} vs {py.Omega}"
    assert _rel(s, py.s) < 1e-13, f"s {s} vs {py.s}"
    assert _rel(frequency, py.frequency) < 1e-13
    assert _rel_field(phi, py.phi) < 1e-13
    assert _rel_field(psi, py.psi) < 1e-12, "psi is four orders smaller than phi -- a looser bar"
    assert _rel_field(stretch, py.stretch_ratio) < 1e-14
    assert _rel_field(tension, py.tension) < 1e-13
    assert _rel(shape_residual, py.shape_residual) < 1e-8, (
        "shape_residual is a ratio of two nearly-cancelling differences, so it carries the "
        "loosest bar of anything here -- see the module docstring on what is claimed"
    )
    # The metadata is passed through untouched and must be identical, not close.
    assert (amplitude, int(mode), bool(td)) == (py.amplitude, py.mode, py.time_discrete)
    # Both ends are clamped, and exactly so on both sides.
    assert phi[0] == phi[-1] == psi[0] == psi[-1] == 0.0
    assert py.phi[0] == py.phi[-1] == py.psi[0] == py.psi[-1] == 0.0


@pytest.mark.parametrize("time_discrete", [True, False])
@pytest.mark.parametrize("mode,kappa,amp", [(1, 0.0, 5e-3), (2, 2.0, 3e-3), (3, 8.0, 1e-3)])
def test_the_root_agrees_across_modes_stiffness_and_both_time_schemes(
    time_discrete, mode, kappa, amp
):
    """The same claim off the canonical fixture, including the semi-discrete branch.

    ``time_discrete=False`` drops both scheme factors, so it exercises a different pair of code
    paths in the residual and the Jacobian -- a transcription that dropped ``cos(Omega k)`` would
    pass the semi-discrete half and fail the discrete one.
    """
    py, (phi, psi, _, _, scalars) = _both(
        mode=mode, kappa=kappa, amp=amp, time_discrete=time_discrete
    )
    assert scalars[7] and py.converged
    assert _rel(scalars[0], py.Omega) < 1e-13
    assert _rel_field(phi, py.phi) < 1e-12
    assert _rel_field(psi, py.psi) < 1e-11


@pytest.mark.slow
def test_convergence_agrees_everywhere_and_the_root_agrees_wherever_it_converges():
    """The grid the module header's numbers come from, at a size a test can afford.

    Two separate claims, and the split is the point:

    * ``converged`` must agree **everywhere**, because it is the one flag a caller branches on and
      the only thing standing between a non-equilibrium helix and a test that treats it as physics.
    * ``Omega`` is compared only **where both converged**. Past that the two Newton paths are the
      answer rather than a means to it, and they diverge by design.
    """
    grid = [
        (n, mode, kappa, amp)
        for n in (16, 32, 48)
        for mode in (1, 3)
        for kappa in (0.0, 2.0)
        for amp in (1e-3, 5e-3)
    ]
    worst_omega = 0.0
    iteration_mismatches = 0
    converged_cells = 0
    for n, mode, kappa, amp in grid:
        py, (_, _, _, _, scalars) = _both(n=n, mode=mode, kappa=kappa, amp=amp)
        assert bool(scalars[7]) is bool(py.converged), (
            f"the convergence FLAG must agree at N={n} mode={mode} kappa={kappa} amp={amp}: "
            f"rust={scalars[7]} python={py.converged}"
        )
        if not py.converged:
            continue
        converged_cells += 1
        worst_omega = max(worst_omega, _rel(scalars[0], py.Omega))
        iteration_mismatches += int(scalars[6] != py.iterations)

    assert converged_cells >= len(grid) // 2, "the grid must mostly converge or it tests nothing"
    assert worst_omega < 1e-13, f"worst Omega disagreement over the grid: {worst_omega:.3e}"
    # REPORTED, not required. See the module docstring: the count differs in ~16% of converged
    # fixtures and it does not matter, because the roots still agree. Asserting it equal here
    # would be asserting a coincidence of the fixture grid.
    assert 0 <= iteration_mismatches <= converged_cells


def test_the_iteration_count_is_not_compared_and_this_is_the_witness():
    """A fixture where the two continuations genuinely take different paths to the same root.

    Kept as an executable statement of what this file declines to assert. If it ever starts
    agreeing, that is interesting rather than wrong -- but the header's claim would need re-taking,
    which is why the assertion is on the *roots* and the counts are only reported.
    """
    py, (phi, _, _, _, scalars) = _both(n=48, mode=1, kappa=2.0, amp=3e-3)
    assert py.converged and scalars[7]
    assert scalars[6] != py.iterations, (
        f"expected the Newton paths to differ here (rust={scalars[6]}, python={py.iterations}); "
        "if they now agree, re-measure the grid before tightening anything"
    )
    assert _rel(scalars[0], py.Omega) < 1e-14, "and yet the root is the same to a few ulps"
    assert _rel_field(phi, py.phi) < 1e-13


# -- the pure functions, and the two last-bit divergences ----------------------------------


@pytest.mark.parametrize("strain", [1e-4, 1e-3, 1e-2, 0.1])
def test_planar_hessian_agrees_to_a_few_ulps_and_the_gap_is_a_numpy_power(strain):
    """``H_pp, H_pz, H_zz`` -- pure arithmetic, and *still* not bit-identical.

    The only transcendental in sight is ``sqrt``, which IEEE-754 pins, so the natural expectation is
    equality. It fails on about a quarter of the entries, and the cause is §16.2's: the Python
    divides by ``lam**3``, whose NumPy power ufunc calls ``pow`` rather than multiplying, and it
    disagrees with ``lam*lam*lam`` on 2 of 5 ordinary values. See the module docstring for why the
    Rust is not respelled to match.
    """
    rng = np.random.default_rng(hash(str(strain)) % (2**32))
    p = strain * rng.standard_normal(64)
    z = 0.1 * strain * rng.standard_normal(64)
    a = EA - T
    want = RW.planar_hessian_cells_py(p, z, a)
    got = rs.rotating_wave_planar_hessian_cells(p, z, a)
    for name, x, y in zip(("H_pp", "H_pz", "H_zz"), want, got, strict=True):
        nz = np.abs(x) > 0
        worst = float(np.max(np.abs((x[nz] - y[nz]) / x[nz]))) if nz.any() else 0.0
        assert worst < 1e-14, f"{name} at strain {strain}: worst relative gap {worst:.3e}"

    # `H_zz = -a p^2 / Lambda^3` is EXACTLY zero at rest on both sides -- the identity, not the
    # value. The literal spelling this rearrangement replaced cannot produce it.
    zero = np.zeros(4)
    assert np.all(RW.planar_hessian_cells_py(zero, zero, a)[2] == 0.0)
    assert np.all(np.asarray(rs.rotating_wave_planar_hessian_cells(zero, zero, a)[2]) == 0.0)


def test_kc_circular_frequency_agrees_to_a_ulp_over_the_musical_range():
    """The KC limit oracle. One value in 200 differs, by 1.4e-16 -- a Python scalar's ``**2``."""
    rng = np.random.default_rng(7)
    worst = 0.0
    for _ in range(200):
        omega0_sq = 10.0 ** rng.uniform(2, 7)
        eps = 10.0 ** rng.uniform(4, 9)
        amplitude = 10.0 ** rng.uniform(-5, -2)
        want = RW.kc_circular_frequency_py(omega0_sq=omega0_sq, eps=eps, amplitude=amplitude)
        got = rs.rotating_wave_kc_circular_frequency(omega0_sq, eps, amplitude)
        worst = max(worst, _rel(got, want))
    assert worst < 1e-15, f"worst relative gap {worst:.3e}"
    # `eps = 0` removes the shift entirely and both sides return sqrt(omega0^2) exactly.
    assert RW.kc_circular_frequency_py(
        omega0_sq=4.0e5, eps=0.0, amplitude=5e-3
    ) == rs.rotating_wave_kc_circular_frequency(4.0e5, 0.0, 5e-3)


def test_the_history_helper_agrees_but_is_not_asserted_exact():
    """The seed the 1e-15 rigid-rotation claim rests on.

    Observed bit-identical on this machine, and asserted only to a tolerance on purpose: it runs
    through ``cos`` and ``sin``, and ``docs/dev/scientific-hurdles.md`` §3 is the standing rule that
    a value crossing ``np.<transcendental>`` must not be asserted exactly across languages -- that
    is a claim about the runner's CPU, and it has already turned this project's CI red once.
    """
    py, _ = _both()
    fs = _fs(32)
    want = RW.rotating_wave_history_py(py, fs=fs)
    got = rs.rotating_wave_history(py.phi, py.psi, py.Omega, fs)
    names = ("u0", "w0", "v0", "u_prev", "w_prev", "v_prev")
    for name, x, y in zip(names, want, got, strict=True):
        assert _rel_field(y, x) < 1e-14, f"{name} disagrees"
    # `w` starts at exactly zero and `v` does not move: both are structural, not numerical.
    assert np.all(np.asarray(got[1]) == 0.0)
    assert np.array_equal(np.asarray(got[2]), np.asarray(got[5]))


# -- the refusals -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "override,fragment",
    [
        ({"n": 1}, "N must be >= 2"),
        ({"mode": 0}, "mode must be in"),
        ({"mode": 99}, "mode must be in"),
        ({"kappa": -1.0}, "kappa must be >= 0"),
        ({"steps": 0}, "continuation_steps must be >= 1"),
        ({"maxiter": 0}, "maxiter must be >= 1"),
    ],
)
def test_both_sides_refuse_the_same_parameters_with_the_same_message(override, fragment):
    """The messages are matched on by ``test_geometric_rotating_wave.py``, so they are interface."""
    kwargs = dict(n=32, amp=5e-3, mode=1, kappa=0.0, steps=8, maxiter=50)
    kwargs.update(override)
    with pytest.raises(ValueError, match=fragment):
        RW.solve_rotating_wave_py(
            L=L, T=T, rho=RHO, EA=EA, fs=_fs(32), N=kwargs["n"], theta=THETA,
            amplitude=kwargs["amp"], mode=kwargs["mode"], kappa=kwargs["kappa"],
            continuation_steps=kwargs["steps"], maxiter=kwargs["maxiter"],
        )
    with pytest.raises(ValueError, match=fragment):
        rs.rotating_wave_solve(
            L, T, RHO, EA, _fs(32), kwargs["n"], THETA, kwargs["amp"], kwargs["mode"],
            kwargs["kappa"], True, kwargs["steps"], RW.NEWTON_TOL_DEFAULT, kwargs["maxiter"],
        )


def test_a_softening_string_is_refused_by_both():
    """``EA < T`` is outside the oracle's scope -- the continuation seed is the hardening branch."""
    with pytest.raises(ValueError, match="softening string"):
        RW.solve_rotating_wave_py(
            L=L, T=T, rho=RHO, EA=100.0, fs=_fs(32), N=32, theta=THETA, amplitude=1e-3
        )
    with pytest.raises(ValueError, match="softening string"):
        rs.rotating_wave_solve(
            L, T, RHO, 100.0, _fs(32), 32, THETA, 1e-3, 1, 0.0, True, 8,
            RW.NEWTON_TOL_DEFAULT, 50,
        )
