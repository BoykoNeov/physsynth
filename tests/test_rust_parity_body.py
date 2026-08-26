"""Rust vs Python for the modal body — Phase 2 batch 2's comparison.

The fourth of these files and it inherits the others' bars unchanged: the acceptance contract is
the physics harness (``docs/dev/rust-migration-plan.md`` §2.1), and this is the *diagnostic* that
catches what the physics bars are too loose to see. The state is required to be bit-identical; the
five read-outs go through ``np.dot``/BLAS and are held to the plan's Group A target.

What is new here, and what each new thing is worth:

* **There is a third state buffer, and its name starts with an underscore.** ``_accel`` is the
  modal acceleration of the step actually taken. Three modules — ``radiation.RadiatedBody``, the
  rational air load, and ``airbox.RoomLoadedBody`` — correct ``q`` after ``step()`` returns and
  then **assign** to ``_accel``, once per timestep. So it is part of the interface whatever its
  spelling says, and the whole rank-1 idiom is asserted here against both implementations. No
  ``tests/test_body.py`` case reaches it.

* **The step takes a force, and the acceleration is the only place it is visible.** A port that
  reconstructed ``q'' = -omega^2 q - 2 sigma q'`` instead of taking the second difference would
  conserve energy, decay monotonically, hit every modal frequency — and radiate the wrong sound
  the moment a bridge is attached. So the forced step is swept, not just the free one.

* **``set_state`` seeds ``_accel`` with the free response, not with zero.** ``pressure()`` is
  readable before the first ``step()``, and a zero there would be a silent wrong answer rather
  than an obvious one.

* **The CFL message names the *argmax*, not the first offender.** With two modes over the
  ceiling the original reports the larger one. That is a message-only difference, and the messages
  are matched on elsewhere in the suite.
"""

import numpy as np
import pytest

from physsynth.core.body import ModalBodyPy

physsynth_rs = pytest.importorskip(
    "physsynth_rs", reason="the Rust extension is not built in this environment"
)

# The plan's Group A agreement target. Only the read-outs are measured against it; state is exact.
GROUP_A_TOL = 1e-13

FS = 48000.0

CASES = [
    dict(freqs=np.array([220.0]), fs=FS),
    dict(freqs=np.array([220.0, 337.0, 512.5]), fs=FS, sigmas=1.5),
    dict(
        freqs=np.array([98.0, 196.5, 311.0, 440.0, 623.25]),
        fs=FS,
        sigmas=np.array([0.5, 1.0, 2.0, 4.0, 8.0]),
        masses=np.array([0.02, 0.03, 0.05, 0.07, 0.11]),
        phi=np.array([1.0, -0.5, 0.25, -0.125, 0.0625]),
    ),
    dict(freqs=[220.0], fs=FS, masses=0.05, radiation=1e-3),
    dict(freqs=np.array([110.0, 220.0]), fs=96000.0, sigmas=0.0, masses=2.0, phi=0.75),
]


def _pair(case):
    return ModalBodyPy(**case), physsynth_rs.ModalBody(**case)


def _q0(body, scale=1e-3):
    """A deterministic, sign-varying initial displacement — one value per mode."""
    return scale * np.array([(-0.6) ** i * (1.0 + 0.3 * i) for i in range(body.M)])


def _ids(case):
    return f"{len(np.atleast_1d(case['freqs']))}modes-fs{int(case['fs'])}"


# -- construction ---------------------------------------------------------------------------------


@pytest.mark.parametrize("case", CASES, ids=_ids)
def test_every_derived_parameter_agrees_exactly(case):
    py, rs = _pair(case)
    assert py.fs == rs.fs
    assert py.k == rs.k
    assert py.M == rs.M
    for name in ("freqs", "sigma", "m", "phi", "a", "omega", "omega_k"):
        assert np.array_equal(getattr(py, name), getattr(rs, name)), name
        assert getattr(rs, name).dtype == np.float64, name


@pytest.mark.parametrize("case", CASES, ids=_ids)
def test_a_fresh_body_is_at_rest_in_all_three_buffers(case):
    py, rs = _pair(case)
    for name in ("q", "q_prev", "_accel"):
        assert np.array_equal(getattr(py, name), getattr(rs, name)), name
        assert np.array_equal(getattr(rs, name), np.zeros(rs.M)), name
    assert py.n == rs.n == 0


def test_radiation_defaults_to_phi_and_is_not_the_same_array():
    kw = dict(freqs=np.array([220.0, 330.0]), fs=FS, phi=np.array([0.7, -0.3]))
    py, rs = _pair(kw)
    assert np.array_equal(py.a, py.phi)
    assert np.array_equal(rs.a, rs.phi)
    assert np.array_equal(py.a, rs.a)


def test_the_construction_refusals_match_verbatim():
    bad = [
        (dict(freqs=np.array([100.0, -5.0]), fs=FS), "all modal frequencies must be positive."),
        (dict(freqs=np.array([220.0]), fs=0.0), "fs must be positive."),
        (dict(freqs=np.array([220.0]), fs=FS, sigmas=-1e-9), "sigmas (loss) must all be >= 0."),
        (dict(freqs=np.array([220.0]), fs=FS, masses=0.0), "masses must all be positive."),
    ]
    for kw, message in bad:
        with pytest.raises(ValueError) as py_err:
            ModalBodyPy(**kw)
        with pytest.raises(ValueError) as rs_err:
            physsynth_rs.ModalBody(**kw)
        assert str(py_err.value) == message
        assert str(rs_err.value) == message


def test_the_cfl_refusal_names_the_same_mode_and_reads_the_same():
    # Two modes over the ceiling. The original reports `np.argmax(omega_k)` -- the *larger* CFL
    # number -- not the first offender, and a port that reported the first would still be green on
    # every physics bar in the project.
    kw = dict(freqs=np.array([18000.0, 22000.0]), fs=FS)
    with pytest.raises(ValueError) as py_err:
        ModalBodyPy(**kw)
    with pytest.raises(ValueError) as rs_err:
        physsynth_rs.ModalBody(**kw)
    assert str(py_err.value) == str(rs_err.value)
    assert "for mode 1" in str(rs_err.value)


# -- the state ------------------------------------------------------------------------------------


@pytest.mark.parametrize("case", CASES, ids=_ids)
@pytest.mark.parametrize("v0", [0.0, 0.25])
def test_set_state_is_bit_identical_including_the_acceleration_seed(case, v0):
    py, rs = _pair(case)
    q0 = _q0(py)
    py.set_state(q0, v0)
    rs.set_state(q0, v0)
    assert np.array_equal(py.q, rs.q), "q"
    assert np.array_equal(py.q_prev, rs.q_prev), "the second-order start disagreed"
    assert np.array_equal(py._accel, rs._accel), "the acceleration seed disagreed"
    # And the seed is the free response, not zero -- a wrong answer that reads as a quiet model.
    assert not np.allclose(rs._accel, 0.0)


def test_a_scalar_initial_displacement_broadcasts_the_same_way():
    py, rs = _pair(CASES[2])
    py.set_state(1e-4)
    rs.set_state(1e-4)
    assert np.array_equal(py.q, rs.q)
    assert np.array_equal(py.q_prev, rs.q_prev)
    assert np.array_equal(py._accel, rs._accel)


@pytest.mark.parametrize("case", CASES, ids=_ids)
def test_the_free_state_stays_bit_identical_over_a_long_run(case):
    py, rs = _pair(case)
    q0 = _q0(py)
    py.set_state(q0)
    rs.set_state(q0)
    for i in range(4000):
        py.step()
        rs.step()
        if i % 250 == 0 or i == 3999:
            assert np.array_equal(py.q, rs.q), f"state diverged at step {i}"
            assert np.array_equal(py.q_prev, rs.q_prev), f"history diverged at step {i}"
            assert np.array_equal(py._accel, rs._accel), f"acceleration diverged at step {i}"
    assert py.n == rs.n == 4000


@pytest.mark.parametrize("case", CASES, ids=_ids)
def test_a_driven_run_is_bit_identical_too(case):
    # The force is the only thing `_accel` carries that the restoring term does not, so a port that
    # reconstructed the acceleration would pass the free test above and fail here.
    py, rs = _pair(case)
    py.set_state(0.0)
    rs.set_state(0.0)
    for i in range(2000):
        force = 0.7 * np.sin(0.013 * i) + 0.2
        py.step(force)
        rs.step(force)
        if i % 200 == 0 or i == 1999:
            assert np.array_equal(py.q, rs.q), f"state diverged at step {i}"
            assert np.array_equal(py._accel, rs._accel), f"acceleration diverged at step {i}"
    assert py.energy() > 0.0, "a driven body must be moving -- otherwise this proves nothing"


@pytest.mark.parametrize("case", CASES, ids=_ids)
def test_the_read_outs_agree_to_the_group_a_target(case):
    py, rs = _pair(case)
    q0 = _q0(py)
    py.set_state(q0, 0.1)
    rs.set_state(q0, 0.1)
    worst = 0.0
    for i in range(1500):
        force = 0.4 * np.cos(0.021 * i)
        py.step(force)
        rs.step(force)
        if i % 100:
            continue
        for a, b in (
            (py.energy(), rs.energy()),
            (py.pressure(), rs.pressure()),
            (py.bridge_displacement(), rs.bridge_displacement()),
            (py.bridge_velocity(), rs.bridge_velocity()),
        ):
            scale = max(abs(a), abs(b), 1e-300)
            worst = max(worst, abs(a - b) / scale)
    assert worst <= GROUP_A_TOL, f"worst relative read-out disagreement {worst:e}"


def test_state_and_displacement_at_agree_including_a_negative_index():
    py, rs = _pair(CASES[2])
    q0 = _q0(py)
    py.set_state(q0)
    rs.set_state(q0)
    for _ in range(37):
        py.step(0.1)
        rs.step(0.1)
    assert np.array_equal(py.state, rs.state)
    assert py.state is not py.q, "`state` must be a copy"
    assert rs.state is not rs.q, "`state` must be a copy"
    for idx in (0, 2, py.M - 1, -1, -py.M):
        assert py.displacement_at(idx) == rs.displacement_at(idx), idx


# -- the buffer contract (plan §9.3), asserted against both implementations ------------------------


def _make(kind, **kw):
    return ModalBodyPy(**kw) if kind == "py" else physsynth_rs.ModalBody(**kw)


@pytest.mark.parametrize("kind", ["py", "rs"])
def test_a_reference_held_across_a_step_is_a_snapshot(kind):
    b = _make(kind, **CASES[1])
    b.set_state(_q0(b))
    held = b.q
    before = held.copy()
    b.step()
    assert np.array_equal(held, before), (
        "the array handed out before a step changed underneath the caller -- `step` must rebind, "
        "not overwrite"
    )
    assert b.q is not held


@pytest.mark.parametrize("kind", ["py", "rs"])
def test_the_previous_array_is_the_object_q_used_to_be(kind):
    b = _make(kind, **CASES[1])
    b.set_state(_q0(b))
    was = b.q
    b.step()
    assert b.q_prev is was, "the history roll must hand the same object over, not a copy"


@pytest.mark.parametrize("kind", ["py", "rs"])
def test_a_write_through_q_reaches_the_model(kind):
    b = _make(kind, **CASES[1])
    b.set_state(_q0(b))
    b.q[0] += 1.0
    assert b.state[0] == b.q[0]
    assert np.isfinite(b.energy())


@pytest.mark.parametrize("kind", ["py", "rs"])
def test_assigning_q_and_accel_directly_is_supported(kind):
    # Verbatim the idiom `RadiatedBody`, the rational air load and `RoomLoadedBody` all use.
    b = _make(kind, **CASES[1])
    b.set_state(_q0(b))
    fresh = np.ascontiguousarray(b.q * 0.5)
    b.q = fresh
    assert b.q is fresh
    accel = np.ascontiguousarray(np.ones(b.M))
    b._accel = accel
    assert b._accel is accel
    assert b.pressure() == pytest.approx(float(np.dot(b.a, accel)))


@pytest.mark.parametrize("kind", ["py", "rs"])
def test_the_rank_one_correction_idiom_round_trips(kind):
    # The exact shape of `RadiatedBody.step`: snapshot q^{n-1}, step, correct q^{n+1} in place,
    # then rewrite `_accel` from the corrected second difference and read `pressure()` back.
    b = _make(kind, **CASES[2])
    b.set_state(_q0(b))
    k = b.k
    q_nm1 = b.q_prev.copy()
    b.step(0.0)
    corr = 1e-9 * np.arange(1.0, b.M + 1.0)
    b.q = b.q - corr
    b._accel = (b.q - 2.0 * b.q_prev + q_nm1) / (k * k)
    assert b.pressure() == pytest.approx(float(np.dot(b.a, b._accel)), rel=1e-13)
    assert np.isfinite(b.energy())


def test_the_correction_gives_the_same_answer_in_both_implementations():
    py = _make("py", **CASES[2])
    rs = _make("rs", **CASES[2])
    q0 = _q0(py)
    py.set_state(q0)
    rs.set_state(q0)
    for _ in range(500):
        for b in (py, rs):
            q_nm1 = b.q_prev.copy()
            b.step(0.3)
            corr = 1e-9 * np.arange(1.0, b.M + 1.0)
            b.q = b.q - corr
            b._accel = (b.q - 2.0 * b.q_prev + q_nm1) / (b.k * b.k)
    assert np.array_equal(py.q, rs.q), "the corrected state diverged"
    assert np.array_equal(py._accel, rs._accel), "the corrected acceleration diverged"
    assert abs(py.pressure() - rs.pressure()) <= GROUP_A_TOL * abs(py.pressure())
