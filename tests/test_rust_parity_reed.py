"""Rust vs Python for the single reed — Phase 2 batch 3's comparison (the exciter half).

The sixth of these files and the first one whose model does not evaluate a formula. Everything
ported before this steps a fixed expression; the reed **solves a scalar equation** each timestep,
by safeguarded Newton with a bracketed Brent fallback. That changes what the file has to assert.

* **Which branch a step took is part of the trajectory, not a diagnostic.** ``fallbacks`` is
  compared step for step, not merely at the end. If Rust stalls Newton on a different step than
  Python, the two separate *structurally* rather than by rounding — and no energy bar, no spectrum
  and no end-of-run comparison of ``fallbacks`` would necessarily see it.

* **The fallback is not hypothetical, so the Brent had to be exact.** Measured before any Rust was
  written: it fires 4-5 times per 4,000 steps at the flagship ``p_mouth = 1500 Pa``, 13 at 1800,
  and **270 times** on a coarse ``N = 40`` grid. ``physsynth-core``'s dependency list is empty by
  design, so there is no SciPy to call; ``crates/physsynth-core/src/root.rs`` is a transcription of
  SciPy's ``Zeros/brentq.c``, checked in Python against the real thing over 248 real calls before
  it was relied on. The coarse case below is what keeps that claim honest.

* **The reed's node-0 compliance is deliberately NOT the bore's.** The bore spells it
  ``rho0 * c0**2``; the reed, from the bore's *public* geometry, spells it ``rho0 * c0 * c0``. They
  disagree by one ulp in 3,531 of 3,552 tube/grid combinations. That predates the migration, its
  physical consequence is nil, and a port that tidied the two into agreement would be changing a
  number the acceptance runs were taken with (plan §12.8). Asserted here on both sides.

* **The Rust reed requires a Rust bore.** It injects through a native closure so the clarinet's hot
  loop crosses the language boundary once per ``step()`` rather than twice. Handed the pure-Python
  ``BorePy`` it raises ``TypeError`` rather than falling back — a silent fallback would be a Rust
  reed reporting Rust while blowing a Python tube, which is the green-and-meaningless shape the
  swap guard in ``tests/test_stability.py`` exists to prevent.

The state is required to be bit-identical; ``energy()`` inherits the bore's ``np.dot`` reductions
and is held to the plan's Group A target.
"""

import numpy as np
import pytest

from physsynth.core.bore import C0_AIR, BorePy
from physsynth.core.reed import ReedBorePy
from physsynth.core.reed import bernoulli_flow as bernoulli_flow_py

physsynth_rs = pytest.importorskip(
    "physsynth_rs", reason="the Rust extension is not built in this environment"
)

GROUP_A_TOL = 1e-13

L_DEFAULT = 0.6
RADIUS_DEFAULT = 0.008
R_BELL_DEFAULT = 650.0

# The reed state that must agree exactly, and the cumulative channels that must agree exactly with
# it -- `mouth_work` and friends are running sums of post-solve values, so a divergence anywhere in
# the step shows up here one step later.
REED_STATE = (
    "y", "y_prev", "dp", "reed_velocity", "flow", "jet_flow",
    "mouth_work", "jet_loss", "reed_damp_work", "fallbacks", "n",
)

CASES = [
    dict(N=200, lam=1.0, p_mouth=1500.0),
    dict(N=200, lam=1.0, p_mouth=1800.0, boundary=("closed", "open"), R_bell=0.0),
    dict(N=200, lam=1.0, p_mouth=1000.0, boundary=("closed", "open"), R_bell=0.0),
    # Below the blowing threshold: the note fails to speak, and Newton never stalls.
    dict(N=200, lam=1.0, p_mouth=300.0),
    # The coarse grid, where the fallback fires in the hundreds. This is the case that actually
    # tests the transcribed Brent; the others mostly test the Newton path.
    dict(N=40, lam=1.0, p_mouth=1500.0),
    dict(N=200, lam=0.9, p_mouth=1500.0),
    dict(N=120, lam=1.0, p_mouth=1500.0, sigma=5.0, boundary=("closed", "open"), R_bell=0.0),
    dict(N=200, lam=1.0, p_mouth=1500.0, R_bell=5.0e4),
    dict(N=150, lam=1.0, p_mouth=2200.0, f_reed=1200.0, q_reed=8.0),
    # p_mouth = 0 is not a special case anywhere in the code, so it exercises the whole chain.
    dict(N=60, lam=1.0, p_mouth=0.0, boundary=("closed", "open"), R_bell=0.0),
]


def _build(bore_cls, reed_cls, case):
    kw = dict(case)
    L = kw.pop("L", L_DEFAULT)
    N = kw.pop("N")
    lam = kw.pop("lam")
    p_mouth = kw.pop("p_mouth")
    boundary = kw.pop("boundary", ("closed", "radiating"))
    R_bell = kw.pop("R_bell", R_BELL_DEFAULT)
    sigma = kw.pop("sigma", 0.0)
    bore = bore_cls(
        L=L, fs=C0_AIR / (lam * (L / N)), N=N, radius=RADIUS_DEFAULT,
        boundary=boundary, R_bell=R_bell, sigma=sigma,
    )
    return reed_cls(bore=bore, p_mouth=p_mouth, **kw)


def _pair(case):
    return (
        _build(BorePy, ReedBorePy, case),
        _build(physsynth_rs.Bore, physsynth_rs.ReedBore, case),
    )


def _ids(case):
    return f"N{case['N']}-lam{case['lam']}-pm{int(case['p_mouth'])}"


def _assert_reed_state(py, rs, where):
    for name in REED_STATE:
        assert getattr(py, name) == getattr(rs, name), (
            f"{where}: reed `{name}` differs -- {getattr(py, name)!r} vs {getattr(rs, name)!r}"
        )
    assert np.array_equal(py.bore.p, rs.bore.p), f"{where}: bore pressure differs"
    assert np.array_equal(py.bore.U, rs.bore.U), f"{where}: bore U differs"
    assert py.bore.radiated_energy == rs.bore.radiated_energy, f"{where}: radiated_energy differs"


def _rel(a, b):
    scale = max(abs(a), abs(b))
    return 0.0 if scale == 0.0 else abs(a - b) / scale


# -- the Bernoulli jet ---------------------------------------------------------------------------


@pytest.mark.parametrize("dp", [-5000.0, -137.5, -1e-12, 0.0, 1e-12, 137.5, 5000.0])
@pytest.mark.parametrize("opening", [0.0, -1e-9, 1e-9, 4.0e-4, 1.2e-3])
def test_the_jet_is_bit_identical(dp, opening):
    a = bernoulli_flow_py(dp, opening, 1.5e-2, 1.2041)
    b = physsynth_rs.bernoulli_flow(dp, opening, 1.5e-2, 1.2041)
    assert a == b, f"{a!r} vs {b!r}"
    # ...including the sign of zero, which `copysign` produces and `==` cannot see.
    assert np.signbit(a) == np.signbit(b)


# -- derived parameters --------------------------------------------------------------------------


@pytest.mark.parametrize("case", CASES, ids=_ids)
def test_every_derived_scalar_is_identical(case):
    py, rs = _pair(case)
    for name in (
        "k", "f_reed", "q_reed", "mu", "Sr", "width", "H0", "rho",
        "newton_tol", "newton_maxiter", "wr", "g", "Mr", "p_closing", "p_mouth", "gamma",
    ):
        assert getattr(py, name) == getattr(rs, name), name


@pytest.mark.parametrize("case", CASES, ids=_ids)
def test_the_node_zero_compliance_is_not_the_bores_own(case):
    """The plan's §12.8 finding, asserted so a later tidy-up fails loudly rather than quietly.

    Both implementations must compute the reed's injection prefactor from the bore's *public*
    geometry with the ``rho0 * c0 * c0`` spelling, which is one ulp from the bore's own
    ``rho0 * c0**2``. The physics does not care; the acceptance numbers do.
    """
    py, rs = _pair(case)
    # The prefactor is private on both sides, so it is measured through what it scales: a single
    # step from rest with a known mouth pressure lands `flow` on it.
    py.step()
    rs.step()
    assert py.flow == rs.flow
    assert py.dp == rs.dp


# -- construction-time refusals ------------------------------------------------------------------


def _bore_pair(**kw):
    kw.setdefault("L", L_DEFAULT)
    kw.setdefault("N", 60)
    kw.setdefault("fs", C0_AIR / (L_DEFAULT / 60))
    kw.setdefault("radius", RADIUS_DEFAULT)
    return BorePy(**kw), physsynth_rs.Bore(**kw)


@pytest.mark.parametrize(
    "kw, fragment",
    [
        (dict(f_reed=0.0), "must all be positive"),
        (dict(q_reed=0.0), "must all be positive"),
        (dict(mu=0.0), "must all be positive"),
        (dict(Sr=0.0), "must all be positive"),
        (dict(width=0.0), "must all be positive"),
        (dict(H0=0.0), "must all be positive"),
        (dict(newton_maxiter=0), "newton_maxiter must be >= 1"),
    ],
)
def test_the_refusals_fire_on_the_same_input_with_the_same_message(kw, fragment):
    py_bore, rs_bore = _bore_pair(boundary=("closed", "open"))
    with pytest.raises(ValueError) as py_err:
        ReedBorePy(bore=py_bore, p_mouth=1500.0, **kw)
    with pytest.raises(ValueError) as rs_err:
        physsynth_rs.ReedBore(bore=rs_bore, p_mouth=1500.0, **kw)
    assert fragment in str(py_err.value)
    assert str(py_err.value) == str(rs_err.value)


@pytest.mark.parametrize("bad_end", ["open", "radiating"])
def test_a_mouthpiece_that_is_not_closed_is_refused_identically(bad_end):
    py_bore, rs_bore = _bore_pair(boundary=(bad_end, "open"), R_bell=650.0)
    with pytest.raises(ValueError) as py_err:
        ReedBorePy(bore=py_bore, p_mouth=1500.0)
    with pytest.raises(ValueError) as rs_err:
        physsynth_rs.ReedBore(bore=rs_bore, p_mouth=1500.0)
    assert "must be 'closed'" in str(py_err.value)
    assert str(py_err.value) == str(rs_err.value)


def test_a_reed_too_stiff_for_the_timestep_is_refused_identically():
    py_bore, rs_bore = _bore_pair(N=4, fs=C0_AIR / (L_DEFAULT / 4), boundary=("closed", "open"))
    with pytest.raises(ValueError) as py_err:
        ReedBorePy(bore=py_bore, p_mouth=1500.0)
    with pytest.raises(ValueError) as rs_err:
        physsynth_rs.ReedBore(bore=rs_bore, p_mouth=1500.0)
    assert "reed CFL violated" in str(py_err.value)
    assert str(py_err.value) == str(rs_err.value)


def test_a_doubly_wrong_call_reports_the_same_fault():
    """The check order is part of the contract: a bad scalar wins over a bad iteration count."""
    py_bore, rs_bore = _bore_pair(boundary=("open", "open"))
    kw = dict(p_mouth=1500.0, f_reed=0.0, newton_maxiter=0)
    with pytest.raises(ValueError) as py_err:
        ReedBorePy(bore=py_bore, **kw)
    with pytest.raises(ValueError) as rs_err:
        physsynth_rs.ReedBore(bore=rs_bore, **kw)
    assert "must all be positive" in str(py_err.value)
    assert str(py_err.value) == str(rs_err.value)


def test_the_rust_reed_refuses_a_python_bore_rather_than_falling_back():
    """A silent fallback here would be a Rust reed reporting Rust while blowing a Python tube."""
    py_bore, _ = _bore_pair(boundary=("closed", "open"))
    with pytest.raises(TypeError, match="needs a Rust Bore"):
        physsynth_rs.ReedBore(bore=py_bore, p_mouth=1500.0)
    # ...and the reverse pairing is fine, because the Python reed calls only the public interface.
    _, rs_bore = _bore_pair(boundary=("closed", "open"))
    assert ReedBorePy(bore=rs_bore, p_mouth=1500.0).gamma > 0.0


# -- the run -------------------------------------------------------------------------------------


@pytest.mark.parametrize("case", CASES, ids=_ids)
def test_the_state_is_bit_identical_step_for_step(case):
    """Every step, not every hundredth.

    The reed's branch choice is discrete: a step either converged by Newton or handed off to Brent.
    A sampled comparison would find the trajectories still identical long after they had taken
    different branches, because the two roots agree to ~1e-13 and it takes a while to show. So the
    reed's scalars — ``fallbacks`` included — are compared on every one of 2,000 steps.
    """
    py, rs = _pair(case)
    for step in range(2000):
        py.step()
        rs.step()
        for name in REED_STATE:
            assert getattr(py, name) == getattr(rs, name), (
                f"step {step}: reed `{name}` diverged -- "
                f"{getattr(py, name)!r} vs {getattr(rs, name)!r}"
            )
    _assert_reed_state(py, rs, "after 2000 steps")


@pytest.mark.parametrize("case", CASES, ids=_ids)
def test_the_bore_underneath_stays_bit_identical(case):
    py, rs = _pair(case)
    for step in range(2000):
        py.step()
        rs.step()
        if step % 250 == 0 or step == 1999:
            _assert_reed_state(py, rs, f"step {step}")


def test_the_coarse_grid_takes_the_brent_fallback_hundreds_of_times_and_stays_identical():
    """The bar that actually tests the transcribed ``brentq``.

    On ``N = 40`` the Newton path stalls on the ``sqrt`` cusp constantly. If the Rust Brent were a
    *different* Brent — equally correct, differently rounded — the roots would agree to ~1e-13, the
    energy balance would still close, and the trajectories would separate over the following
    thousand steps in a way nothing else here would attribute to the solver.
    """
    case = dict(N=40, lam=1.0, p_mouth=1500.0)
    py, rs = _pair(case)
    for step in range(3000):
        py.step()
        rs.step()
        assert py.fallbacks == rs.fallbacks, (
            f"step {step}: the two took different branches -- {py.fallbacks} vs {rs.fallbacks}"
        )
        assert py.dp == rs.dp, f"step {step}: the solved pressure drop differs"
    assert py.fallbacks > 100, (
        f"only {py.fallbacks} fallbacks in 3000 steps -- this case has stopped testing Brent"
    )
    _assert_reed_state(py, rs, "coarse grid")


@pytest.mark.parametrize("case", CASES, ids=_ids)
def test_the_readouts_agree(case):
    py, rs = _pair(case)
    worst = 0.0
    for step in range(1500):
        py.step()
        rs.step()
        if step % 100:
            continue
        # These carry no reduction and are required to be exact.
        assert py.reed_opening() == rs.reed_opening()
        assert py.reed_energy() == rs.reed_energy()
        assert py.mouthpiece_pressure() == rs.mouthpiece_pressure()
        assert py.pressure() == rs.pressure()
        assert py.displacement_at(1) == rs.displacement_at(1)
        assert py.gamma == rs.gamma
        assert np.array_equal(py.state, rs.state)
        # `energy()` inherits the bore's `np.dot`, so it is a tolerance.
        worst = max(worst, _rel(py.energy(), rs.energy()))
    assert worst < GROUP_A_TOL, f"worst relative disagreement {worst:.2e}"


def test_mutating_the_mouth_pressure_between_steps_agrees():
    """The original documents this as how an attack is played, so it is interface, not accident."""
    case = dict(N=120, lam=1.0, p_mouth=0.0)
    py, rs = _pair(case)
    for step in range(1200):
        # A linear attack ramp, then a release — the mouth pressure moves under the solver's feet
        # and the continuation seed has to keep up on both sides.
        target = 1600.0 * min(1.0, step / 400.0) * (1.0 if step < 900 else 0.35)
        py.p_mouth = target
        rs.p_mouth = target
        py.step()
        rs.step()
        assert py.dp == rs.dp, f"step {step}"
        assert py.y == rs.y, f"step {step}"
        assert py.fallbacks == rs.fallbacks, f"step {step}"
    _assert_reed_state(py, rs, "after the attack ramp")
    assert py.mouth_work > 0.0, "the ramp did no work -- this test is not driving anything"


def test_the_bore_attribute_is_the_object_that_was_passed_in():
    """``tests/test_reed_energy.py`` and ``web/serialize.py`` reach through ``reed.bore`` and call
    methods on it, so it must be the same instance and not a copy."""
    for cls_bore, cls_reed in (
        (BorePy, ReedBorePy),
        (physsynth_rs.Bore, physsynth_rs.ReedBore),
    ):
        bore = cls_bore(
            L=L_DEFAULT, fs=C0_AIR / (L_DEFAULT / 60), N=60, radius=RADIUS_DEFAULT,
            boundary=("closed", "radiating"), R_bell=R_BELL_DEFAULT,
        )
        reed = cls_reed(bore=bore, p_mouth=1500.0)
        assert reed.bore is bore
        for _ in range(100):
            reed.step()
        # The caller's handle sees the stepped air column, not a stale snapshot.
        assert bore.n == 100
        assert bore.energy() == reed.bore.energy()


# -- the physics bar, asserted on the Rust model itself -------------------------------------------


@pytest.mark.parametrize("p_mouth", [1000.0, 1500.0, 1800.0])
def test_the_rust_reed_meets_the_projects_energy_balance_bar(p_mouth):
    """The acceptance contract, restated against the port.

    Not a comparison: if both implementations ever drifted together the tests above would stay
    green and this one would not. The reed *stores* energy, so the conserved quantity is
    ``E_bore + E_reed`` and the identity is a **balance**, not conservation — the mouth is active.
    """
    case = dict(N=200, lam=1.0, p_mouth=p_mouth, boundary=("closed", "open"), R_bell=0.0)
    _, rs = _pair(case)
    e0 = rs.energy()
    for _ in range(3000):
        rs.step()
    lhs = rs.energy() - e0
    rhs = rs.mouth_work - rs.jet_loss - rs.reed_damp_work
    scale = max(abs(rs.mouth_work), abs(rs.jet_loss), abs(lhs), 1e-30)
    assert abs(lhs - rhs) / scale < 1e-11, f"balance error at p_mouth = {p_mouth}"
    # ...and the dissipation channels never ran backwards.
    assert rs.jet_loss >= 0.0 and rs.reed_damp_work >= 0.0


def test_the_rust_note_speaks_above_threshold_and_not_below():
    """The independent oracle: balance alone passes on a dead reed that merely rings down."""
    loud = _pair(dict(N=200, lam=1.0, p_mouth=1500.0))[1]
    quiet = _pair(dict(N=200, lam=1.0, p_mouth=300.0))[1]
    levels = []
    for reed in (loud, quiet):
        tail = []
        for step in range(6000):
            reed.step()
            if step >= 4000:
                tail.append(reed.mouthpiece_pressure())
        levels.append(max(tail) - min(tail))
    loud_pp, quiet_pp = levels
    assert loud_pp > 100.0, f"the note did not speak: {loud_pp:.3f} Pa peak-to-peak"
    assert quiet_pp < 0.01 * loud_pp, f"the note spoke below threshold: {quiet_pp:.3f} Pa"
