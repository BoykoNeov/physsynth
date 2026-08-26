"""Rust vs Python for the air node — Phase 2 batch 4's comparison, and the first one that cannot
ask for bit-identity everywhere.

The four earlier files in this series all end with "the state is required to be bit-identical".
This one does not, and the reason is the batch's headline. `RadiatedBody.step` computes

    u_free = float(np.dot(b.a, b.q - q_nm1)) / (2.0 * self.k)

and then *feeds that number back into* ``q^{n+1}``. `body.pressure()` has had the identical
reduction since batch 2 and was held to the plan's Group A target (1e-13) — but it is a read-out,
so its last bit reaches an assertion and nothing else. Here it reaches the next timestep.

`np.dot` on contiguous doubles is OpenBLAS, and OpenBLAS **fuses the multiply-add**. Measured on
this machine (numpy 2.4.6, scipy-openblas 0.3.31, `DYNAMIC_ARCH`, SkylakeX): below sixteen terms
it is a single-accumulator sequential `fma` loop, agreeing with an `fma`-per-term model 1000/1000
and with plain multiply-then-add 0/1000 on random data; at sixteen and above it vectorises. Which
kernel runs is a property of the CPU, not of any code in this repo, so reproducing it would mean
pinning a runner rather than writing a port. Rust sums plainly, left to right, and this file
measures the gap instead of pretending it away.

**The gap is smaller than it looks, and the reason no existing test could have found it is the
interesting part.** A fused multiply-add differs from a rounded one only when the *product*
rounds. `tests/helpers.py` builds every body with ``phi=1.0`` and no explicit ``radiation``, so
the weights ``a_i`` are exactly 1 and every product ``a_i * d_i`` is exact; the parity case with
``phi = [1, -0.5, 0.25, -0.125, 0.0625]`` is the same story one step subtler, since those are all
powers of two. Under either, the two implementations are **bit-identical over 20,000 steps** — so
the whole suite, run under `PHYSSYNTH_RS`, sees a divergence of exactly zero and always would
have. It takes a weight like ``radiation=0.02`` to make the reduction differ at all. Both cases
are asserted below, on purpose: the exact one because it is what the suite actually runs, and the
inexact one because it is the honest measurement.

**Read the difference against the run's amplitude, never pointwise.** The loaded body decays by
four orders of magnitude over 20,000 steps, so an element-wise relative difference divides a
frozen absolute error (~2e-17) by a vanishing signal and reports 1e-7 — an artifact of the metric,
not of the port. Against the amplitude, the same run measures 2.4e-14 for the constant-`R` load
and 3.4e-13 for the reactive one.

**And Group A is a short-run bar, which is what the plan already says.** Section 4 words it as
"state arrays to ~1e-13 relative over a *short* run; the physics bars thereafter", and the
qualifier earns its keep here: the weighted case measures 1.4e-14 of the amplitude at 2,000 steps
and 3.4e-13 at 20,000, so a fed-back reduction's error *grows with run length* rather than
saturating the way a read-out's does. Both lengths are asserted below, at their own bars. Neither
physics bar moves at either length — each implementation conserves its own energy identity to the
suite's 1e-10 — which is the division of labour section 4 describes.

What comes out bit-identical, and is asserted that way rather than to a tolerance:

* every construction constant, on all three parameterised types — including `latency_samples`,
  which is `int(round(...))` and therefore **round-half-to-even**, where Rust's `f64::round` is
  half-away-from-zero. A one-sample delay error passes every energy, passivity and modal bar in
  the suite, so it gets its own discriminating case here;
* `AirRadiation.process` over a long random signal, because that path has no reduction in it at
  all — a scalar multiply and a delay line;
* `_G` and `_corr`, the rank-1 precomputes: `np.sum`'s pairwise summation is plain left-to-right
  for seven terms or fewer, and no body in this repo has eight modes;
* `impedance`, `impedance_discrete` and `loaded_mode` across a sweep — complex division included,
  which is CPython's Smith's algorithm rather than the textbook formula;
* the two reductions the original documents as exact (`R = 0` is a bare body, `M_a = inf` is the
  constant-`R` load), asserted **within** each implementation, so they keep their meaning after
  the Python side is deleted.

`piston_radiation_resistance` is absent from this file because it is absent from the port: it
needs a Bessel `J1` and belongs with Phase 7's analytic oracles. `physsynth/core/radiation.py`
keeps the Python one under the flag as well as without it.
"""

import numpy as np
import pytest

from physsynth.core.body import ModalBodyPy
from physsynth.core.radiation import (
    C0_AIR,
    RHO0_AIR,
    AirRadiationPy,
    RadiatedBodyPy,
    RationalAirLoadPy,
    ReactiveRadiatedBodyPy,
    monopole_radiation_resistance_py,
)

physsynth_rs = pytest.importorskip(
    "physsynth_rs", reason="the Rust extension is not built in this environment"
)

# The plan's Group A agreement target, read against the run's amplitude (see the module docstring).
# Section 4 states it as "state arrays to ~1e-13 relative over a SHORT run; the physics bars
# thereafter", and that qualifier turns out to be load-bearing here -- see SHORT/LONG below.
GROUP_A_TOL = 1e-13

# A short run, where Group A is the bar, and a long one, where it is not. Measured on the weighted
# nine-mode case: the state difference is 1.4e-14 of the amplitude at 2,000 steps and 3.4e-13 at
# 20,000, so a fed-back reduction's error GROWS with run length rather than saturating. The
# physics bars do not move at either length -- each implementation conserves its own energy
# identity to the suite's 1e-10 -- which is exactly the division of labour section 4 describes.
LONG_RUN_TOL = 1e-12

FS = 48000.0
R_DEFAULT = 2000.0
SHORT = 2000
STEPS = 20000

# Four bodies. The first three have radiation weights that are exactly representable powers of two
# -- 1.0 by default, and the explicit `phi` of the third -- which is what makes the fused
# multiply-add in `np.dot` indistinguishable from a rounded one. The fourth does not, and it is
# the only case in this file where the two implementations part company at all.
EXACT_CASES = [
    dict(freqs=np.array([220.0]), fs=FS),
    dict(freqs=np.array([110.0, 196.0, 261.0, 440.0]), fs=FS),
    dict(
        freqs=np.array([98.0, 196.5, 311.0, 440.0, 623.25]),
        fs=FS,
        sigmas=np.array([0.5, 1.0, 2.0, 4.0, 8.0]),
        masses=np.array([0.02, 0.03, 0.05, 0.07, 0.11]),
        phi=np.array([1.0, -0.5, 0.25, -0.125, 0.0625]),
    ),
]
# `radiation=0.02` is the weight the core's own validated single-mode rig uses, and 0.02 is not a
# power of two.
INEXACT_CASE = dict(freqs=np.linspace(110.0, 1760.0, 9), fs=FS, masses=0.02, radiation=0.02)
ALL_CASES = EXACT_CASES + [INEXACT_CASE]


def _ids(case):
    return f"M{len(np.atleast_1d(case['freqs']))}" + (
        "-weighted" if "radiation" in case else ""
    )


def _bodies(case):
    return ModalBodyPy(**case), physsynth_rs.ModalBody(**case)


def _q0(case):
    m = len(np.atleast_1d(case["freqs"]))
    return 1e-3 * np.cos(np.arange(m))


def _amplitude_error(py_state, rs_state, scale):
    """Worst absolute difference, as a fraction of the run's amplitude ``scale``.

    Deliberately NOT an element-wise relative difference: see the module docstring.
    """
    return float(np.max(np.abs(np.asarray(py_state) - np.asarray(rs_state)))) / scale


# =================================================================================================
# Tier 1 — the read-out. No reduction anywhere in it, so everything here is exact.
# =================================================================================================

AIR_CASES = [
    dict(fs=FS, distance=1.0),
    dict(fs=FS, distance=3.43),
    dict(fs=FS, distance=5.0, retarded=False),
    dict(fs=FS, distance=0.017),
    dict(fs=96000.0, distance=2.5, rho0=1000.0, c0=1481.0),  # water, for a different retardation
]


@pytest.mark.parametrize("case", AIR_CASES, ids=lambda c: f"r{c['distance']}")
def test_every_read_out_constant_agrees_exactly(case):
    py, rs = AirRadiationPy(**case), physsynth_rs.AirRadiation(**case)
    for name in (
        "fs",
        "k",
        "distance",
        "rho0",
        "c0",
        "retarded",
        "gain",
        "retardation_seconds",
        "latency_samples",
        "retardation_residual",
        "n",
    ):
        assert getattr(py, name) == getattr(rs, name), name
    assert np.array_equal(py._buf, rs._buf)
    assert py._idx == rs._idx


@pytest.mark.parametrize(
    ("distance", "expected"),
    [
        # fs = 2 Hz, c0 = 1 m/s, so the delay in samples is 2 * distance.
        (0.25, 0),  # 0.5 samples -> 0 (half to EVEN), not 1
        (0.75, 2),  # 1.5 samples -> 2
        (1.25, 2),  # 2.5 samples -> 2, not 3
        (1.75, 4),  # 3.5 samples -> 4
    ],
)
def test_the_delay_length_rounds_halves_to_even_on_both_sides(distance, expected):
    # THE trap of this batch. Python's `round` is half-to-even; C's (and Rust's) is
    # half-away-from-zero, and a one-sample delay error passes every physics bar in the suite.
    kw = dict(fs=2.0, distance=distance, c0=1.0)
    assert AirRadiationPy(**kw).latency_samples == expected
    assert physsynth_rs.AirRadiation(**kw).latency_samples == expected


@pytest.mark.parametrize("case", AIR_CASES, ids=lambda c: f"r{c['distance']}")
def test_the_read_out_is_bit_identical_over_a_long_random_signal(case):
    py, rs = AirRadiationPy(**case), physsynth_rs.AirRadiation(**case)
    signal = np.random.default_rng(0).standard_normal(20000) * 1e-2
    out_py = np.array([py.process(v) for v in signal])
    out_rs = np.array([rs.process(v) for v in signal])
    assert np.array_equal(out_py, out_rs)
    assert py.n == rs.n
    py.reset()
    rs.reset()
    assert np.array_equal(py._buf, rs._buf)
    assert py.n == rs.n == 0


def test_radiate_duck_types_on_anything_with_a_pressure():
    # `radiate` is the one place the Rust side does NOT demand a ported type: its callers hand it
    # a Bore, a StringBodyBridge and a Plate as well as a body.
    class Source:
        def __init__(self):
            self.calls = 0

        def pressure(self):
            self.calls += 1
            return 0.25 * self.calls

    py, rs = AirRadiationPy(fs=FS, retarded=False), physsynth_rs.AirRadiation(
        fs=FS, retarded=False
    )
    a, b = Source(), Source()
    for _ in range(10):
        assert py.radiate(a) == rs.radiate(b)
    assert a.calls == b.calls == 10


@pytest.mark.parametrize(
    "kwargs",
    [
        {"fs": 0.0},
        {"fs": -48000.0},
        {"fs": FS, "distance": 0.0},
        {"fs": FS, "distance": -1.0},
        {"fs": FS, "rho0": 0.0},
        {"fs": FS, "c0": 0.0},
    ],
)
def test_the_read_outs_refusals_read_the_same(kwargs):
    with pytest.raises(ValueError) as py_err:
        AirRadiationPy(**kwargs)
    with pytest.raises(ValueError) as rs_err:
        physsynth_rs.AirRadiation(**kwargs)
    assert str(py_err.value) == str(rs_err.value)


def test_the_monopole_helper_agrees_exactly():
    for f in np.geomspace(1.0, 20000.0, 200):
        w = 2.0 * np.pi * f
        assert monopole_radiation_resistance_py(w) == (
            physsynth_rs.monopole_radiation_resistance(w)
        )
    # ...and the medium is a keyword on both.
    assert monopole_radiation_resistance_py(1000.0, rho0=1000.0, c0=1481.0) == (
        physsynth_rs.monopole_radiation_resistance(1000.0, rho0=1000.0, c0=1481.0)
    )
    assert physsynth_rs.RHO0_AIR == RHO0_AIR
    assert physsynth_rs.C0_AIR == C0_AIR


# =================================================================================================
# The rank-1 precomputes — `np.sum`, not `np.dot`, and therefore still exact
# =================================================================================================


@pytest.mark.parametrize("case", ALL_CASES, ids=_ids)
def test_the_rank_one_precomputes_agree_exactly(case):
    # `np.sum` is pairwise from eight terms on; below that it is plain left-to-right, and nothing
    # in this repo has an eight-mode body. Keeping this exact is what lets the state difference
    # measured further down be attributed to `u_free` alone.
    bp, br = _bodies(case)
    lp = RadiatedBodyPy(body=bp, R=R_DEFAULT)
    lr = physsynth_rs.RadiatedBody(body=br, R=R_DEFAULT)
    assert lp._G == lr._G
    assert np.array_equal(lp._corr, lr._corr)


# =================================================================================================
# Tier 2 — the constant-R load
# =================================================================================================


@pytest.mark.parametrize("case", EXACT_CASES, ids=_ids)
def test_the_loaded_state_is_bit_identical_when_the_weights_are_exact(case):
    # This is the configuration the WHOLE SUITE runs under -- `tests/helpers.py` never passes a
    # `radiation` weight, so `a_i = phi_i = 1` and every product inside `np.dot` is exact. The
    # fused multiply-add and the rounded one cannot be told apart, and the port is bit-for-bit.
    bp, br = _bodies(case)
    lp = RadiatedBodyPy(body=bp, R=R_DEFAULT)
    lr = physsynth_rs.RadiatedBody(body=br, R=R_DEFAULT)
    lp.set_state(_q0(case))
    lr.set_state(_q0(case))
    for _ in range(STEPS):
        lp.step()
        lr.step()
        assert np.array_equal(bp.q, br.q)
    assert lp.radiated_energy == lr.radiated_energy
    assert lp.volume_velocity == lr.volume_velocity
    assert lp.n == lr.n == STEPS


def test_a_non_power_of_two_weight_is_where_the_two_part_company():
    """The measurement, and the qualifier on Group A that it forces.

    The reduction differs in the last bit -- measured on this case, `np.dot` disagrees with a plain
    left-to-right sum on 2817/5000 steps, worst 2.1e-13 relative -- so the trajectories are
    genuinely different rather than merely differently rounded. Group A holds over a short run
    (1.4e-14 of the amplitude at 2,000 steps) and is exceeded over a long one (3.4e-13 at 20,000),
    which is why section 4 says "over a short run; the physics bars thereafter" rather than naming
    a single number. Both bars are asserted here so a regression in either is legible.
    """
    case = INEXACT_CASE
    bp, br = _bodies(case)
    lp = RadiatedBodyPy(body=bp, R=R_DEFAULT)
    lr = physsynth_rs.RadiatedBody(body=br, R=R_DEFAULT)
    lp.set_state(_q0(case))
    lr.set_state(_q0(case))
    scale = float(np.max(np.abs(_q0(case))))
    worst = short = 0.0
    diverged = False
    for n in range(STEPS):
        lp.step()
        lr.step()
        diverged |= not np.array_equal(bp.q, br.q)
        worst = max(worst, _amplitude_error(bp.q, br.q, scale))
        if n + 1 == SHORT:
            short = worst
    assert diverged, "a weight of 0.02 should make the fused product visible"
    assert short <= GROUP_A_TOL, f"short run diverged by {short:.3e} of the amplitude"
    assert worst <= LONG_RUN_TOL, f"long run diverged by {worst:.3e} of the amplitude"


@pytest.mark.parametrize("case", ALL_CASES, ids=_ids)
def test_the_loaded_read_outs_agree_to_the_group_a_target(case):
    """Energy and pressure over a short run, measured against the run's own peak.

    Against the *peak* on purpose. Both read-outs decay with the body -- four orders of magnitude
    over the full run -- so a pointwise relative comparison divides a frozen absolute difference by
    a vanishing signal and reports 1e-9 for a port that is agreeing to 5e-15. Measured worst here:
    2.0e-15 (energy) and 5.4e-15 (pressure) on the weighted nine-mode case, zero on the others.
    """
    bp, br = _bodies(case)
    lp = RadiatedBodyPy(body=bp, R=R_DEFAULT)
    lr = physsynth_rs.RadiatedBody(body=br, R=R_DEFAULT)
    lp.set_state(_q0(case))
    lr.set_state(_q0(case))
    peak_e, peak_p = abs(lp.energy()), 0.0
    worst_e = worst_p = 0.0
    for _ in range(SHORT):
        lp.step()
        lr.step()
        peak_e = max(peak_e, abs(lp.energy()))
        peak_p = max(peak_p, abs(lp.pressure()), 1e-300)
        worst_e = max(worst_e, abs(lp.energy() - lr.energy()))
        worst_p = max(worst_p, abs(lp.pressure() - lr.pressure()))
    assert worst_e / peak_e <= GROUP_A_TOL, f"energy {worst_e / peak_e:.3e}"
    assert worst_p / peak_p <= GROUP_A_TOL, f"pressure {worst_p / peak_p:.3e}"
    assert lp.radiated_energy == pytest.approx(lr.radiated_energy, rel=GROUP_A_TOL)


@pytest.mark.parametrize("kind", ["py", "rs"])
def test_zero_resistance_is_bit_identical_to_a_bare_body(kind):
    # An exact reduction the original documents, compared WITHIN one implementation -- so it keeps
    # its meaning after the Python side is deleted. It is also the sharpest check that the `_accel`
    # rewrite reproduces what `ModalBody.step` computed, since `pressure()` reads it.
    make_body = ModalBodyPy if kind == "py" else physsynth_rs.ModalBody
    make_loaded = RadiatedBodyPy if kind == "py" else physsynth_rs.RadiatedBody
    kw = dict(freqs=np.array([110.0, 196.0, 261.0, 440.0]), fs=FS, sigmas=1.5)
    inner = make_body(**kw)
    loaded = make_loaded(body=inner, R=0.0)
    bare = make_body(**kw)
    loaded.set_state(_q0(kw))
    bare.set_state(_q0(kw))
    for _ in range(2000):
        loaded.step()
        bare.step()
        assert np.array_equal(inner.q, bare.q)
        assert loaded.pressure() == bare.pressure()
    assert loaded.radiated_energy == 0.0


@pytest.mark.parametrize("kind", ["py", "rs"])
def test_the_loaded_body_is_a_drop_in_for_the_bare_one(kind):
    # `web/serialize.py` hands a RadiatedBody to StringBodyBridge as the body, and `connection.py`
    # then reads these names off it. A pyclass has no instance `__dict__`, so this is the check
    # that `__getattr__` delegation carries the whole surface across.
    make_body = ModalBodyPy if kind == "py" else physsynth_rs.ModalBody
    make_loaded = RadiatedBodyPy if kind == "py" else physsynth_rs.RadiatedBody
    inner = make_body(freqs=np.array([110.0, 196.0]), fs=FS)
    loaded = make_loaded(body=inner, R=R_DEFAULT)
    assert loaded.M == inner.M
    assert np.array_equal(loaded.phi, inner.phi)
    assert np.array_equal(loaded.m, inner.m)
    assert np.array_equal(loaded.omega, inner.omega)
    assert np.array_equal(loaded.q_prev, inner.q_prev)
    assert loaded.bridge_displacement() == inner.bridge_displacement()
    assert loaded.displacement_at(0) == inner.displacement_at(0)
    assert loaded.body is inner
    # ...and a name neither class has is still an AttributeError, not something stranger.
    with pytest.raises(AttributeError):
        _ = loaded.no_such_attribute


def test_the_loads_refusal_reads_the_same():
    body_py, body_rs = _bodies(EXACT_CASES[0])
    with pytest.raises(ValueError) as py_err:
        RadiatedBodyPy(body=body_py, R=-1.0)
    with pytest.raises(ValueError) as rs_err:
        physsynth_rs.RadiatedBody(body=body_rs, R=-1.0)
    assert str(py_err.value) == str(rs_err.value)


def test_the_rust_wrappers_refuse_a_python_body():
    # The reed's rule (plan section 13.2's neighbour): a Rust wrapper driving a Python body would
    # be a green run reporting Rust while stepping through the interpreter, so it is refused
    # rather than silently supported.
    body_py = ModalBodyPy(freqs=np.array([220.0]), fs=FS)
    with pytest.raises(TypeError):
        physsynth_rs.RadiatedBody(body=body_py, R=R_DEFAULT)
    with pytest.raises(TypeError):
        physsynth_rs.ReactiveRadiatedBody(
            body=body_py, load=physsynth_rs.RationalAirLoad(fs=FS, R=R_DEFAULT)
        )
    with pytest.raises(TypeError):
        physsynth_rs.ReactiveRadiatedBody(
            body=physsynth_rs.ModalBody(freqs=np.array([220.0]), fs=FS),
            load=RationalAirLoadPy(fs=FS, R=R_DEFAULT),
        )


# =================================================================================================
# Tier 3 — the rational impedance
# =================================================================================================

LOAD_CASES = [
    dict(fs=FS, R=2000.0),  # M_a = inf: the constant-R load
    dict(fs=FS, R=1500.0, M_a=0.03),
    dict(fs=FS, R=2000.0, M_a=0.2),
    dict(fs=FS, R=0.0, M_a=0.2),  # decoupled
    dict(fs=FS, R=1500.0, M_a=0.05, rho0=1000.0, c0=1481.0),
]


@pytest.mark.parametrize("case", LOAD_CASES, ids=lambda c: f"R{c['R']}-Ma{c.get('M_a', 'inf')}")
def test_every_load_constant_agrees_exactly(case):
    py, rs = RationalAirLoadPy(**case), physsynth_rs.RationalAirLoad(**case)
    for name in (
        "fs",
        "k",
        "R",
        "M_a",
        "rho0",
        "c0",
        "R_eff",
        "tau",
        "sphere_radius",
        "sphere_area",
        "u_l",
        "radiated_energy",
        "volume_velocity",
        "pressure_load",
        "n",
    ):
        assert getattr(py, name) == getattr(rs, name), name


@pytest.mark.parametrize("radius", [0.01, 0.05, 0.2])
def test_the_sphere_constructor_agrees_exactly(radius):
    py = RationalAirLoadPy.from_sphere(fs=FS, radius=radius)
    rs = physsynth_rs.RationalAirLoad.from_sphere(fs=FS, radius=radius)
    for name in ("R", "M_a", "R_eff", "tau", "sphere_radius", "sphere_area"):
        assert getattr(py, name) == getattr(rs, name), name


@pytest.mark.parametrize("case", LOAD_CASES, ids=lambda c: f"R{c['R']}-Ma{c.get('M_a', 'inf')}")
def test_both_impedance_read_outs_agree_exactly_across_a_sweep(case):
    # Complex division is CPython's Smith's algorithm, not the textbook conjugate formula; the two
    # disagree in the last ulp, so this sweep is what says the transcription is the right one.
    py, rs = RationalAirLoadPy(**case), physsynth_rs.RationalAirLoad(**case)
    for f in np.geomspace(1.0, 20000.0, 500):
        w = 2.0 * np.pi * f
        assert py.impedance(w) == rs.impedance(w), f
        assert py.impedance_discrete(w) == rs.impedance_discrete(w), f


def test_loaded_mode_agrees_exactly_across_a_sweep():
    py = RationalAirLoadPy.from_sphere(fs=FS, radius=0.05)
    rs = physsynth_rs.RationalAirLoad.from_sphere(fs=FS, radius=0.05)
    for f in np.geomspace(110.0, 1760.0, 200):
        w0 = 2.0 * np.pi * f
        assert py.loaded_mode(w0, weight=0.02, mass=0.02) == rs.loaded_mode(
            w0, weight=0.02, mass=0.02
        )


def test_loaded_modes_refusals_read_the_same():
    py = RationalAirLoadPy.from_sphere(fs=FS, radius=0.05)
    rs = physsynth_rs.RationalAirLoad.from_sphere(fs=FS, radius=0.05)
    for kwargs in (
        dict(weight=0.02, mass=0.0),
        dict(weight=0.02, mass=-1.0),
    ):
        with pytest.raises(ValueError) as py_err:
            py.loaded_mode(2.0 * np.pi * 110.0, **kwargs)
        with pytest.raises(ValueError) as rs_err:
            rs.loaded_mode(2.0 * np.pi * 110.0, **kwargs)
        assert str(py_err.value) == str(rs_err.value)
    with pytest.raises(ValueError) as py_err:
        py.loaded_mode(0.0, weight=0.02, mass=0.02)
    with pytest.raises(ValueError) as rs_err:
        rs.loaded_mode(0.0, weight=0.02, mass=0.02)
    assert str(py_err.value) == str(rs_err.value)
    # The non-convergence message, including its "last relative step 0.000e+00" -- which is the
    # ORIGINAL's arithmetic (it reads `w_next - w` after assigning `w = w_next`), transcribed
    # rather than corrected. Fixing it is a change to both sides, not to the port.
    with pytest.raises(ValueError) as py_err:
        py.loaded_mode(2.0 * np.pi * 110.0, weight=0.5, mass=0.02, iterations=1)
    with pytest.raises(ValueError) as rs_err:
        rs.loaded_mode(2.0 * np.pi * 110.0, weight=0.5, mass=0.02, iterations=1)
    assert str(py_err.value) == str(rs_err.value)
    assert "last relative step 0.000e+00" in str(py_err.value)


def test_the_far_field_read_out_agrees_and_refuses_the_same_way():
    py = RationalAirLoadPy.from_sphere(fs=FS, radius=0.05)
    rs = physsynth_rs.RationalAirLoad.from_sphere(fs=FS, radius=0.05)
    for u in (1e-4, -3e-3, 0.0):
        assert py.step(u) == rs.step(u)
        assert py.far_field_pressure(2.0) == rs.far_field_pressure(2.0)
        assert py.far_field_pressure(2.0, p_load=0.5) == rs.far_field_pressure(2.0, p_load=0.5)
    for bad in (RationalAirLoadPy(fs=FS, R=2000.0, M_a=0.2),):
        with pytest.raises(ValueError) as py_err:
            bad.far_field_pressure(1.0)
        with pytest.raises(ValueError) as rs_err:
            physsynth_rs.RationalAirLoad(fs=FS, R=2000.0, M_a=0.2).far_field_pressure(1.0)
        assert str(py_err.value) == str(rs_err.value)
    with pytest.raises(ValueError) as py_err:
        py.far_field_pressure(-1.0)
    with pytest.raises(ValueError) as rs_err:
        rs.far_field_pressure(-1.0)
    assert str(py_err.value) == str(rs_err.value)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"fs": 0.0, "R": 1.0},
        {"fs": FS, "R": -1.0},
        {"fs": FS, "R": 1.0, "M_a": 0.0},
        {"fs": FS, "R": 1.0, "M_a": -1.0},
        {"fs": FS, "R": 1.0, "M_a": float("nan")},
        {"fs": FS, "R": 1.0, "rho0": 0.0},
        {"fs": FS, "R": 1.0, "c0": 0.0},
    ],
)
def test_the_loads_construction_refusals_read_the_same(kwargs):
    with pytest.raises(ValueError) as py_err:
        RationalAirLoadPy(**kwargs)
    with pytest.raises(ValueError) as rs_err:
        physsynth_rs.RationalAirLoad(**kwargs)
    assert str(py_err.value) == str(rs_err.value)


def test_the_sphere_constructor_refuses_the_same_way():
    with pytest.raises(ValueError) as py_err:
        RationalAirLoadPy.from_sphere(fs=FS, radius=0.0)
    with pytest.raises(ValueError) as rs_err:
        physsynth_rs.RationalAirLoad.from_sphere(fs=FS, radius=0.0)
    assert str(py_err.value) == str(rs_err.value)


@pytest.mark.parametrize("case", EXACT_CASES, ids=_ids)
def test_the_reactive_state_is_bit_identical_when_the_weights_are_exact(case):
    bp, br = _bodies(case)
    lp = ReactiveRadiatedBodyPy(body=bp, load=RationalAirLoadPy(fs=FS, R=1500.0, M_a=0.03))
    lr = physsynth_rs.ReactiveRadiatedBody(
        body=br, load=physsynth_rs.RationalAirLoad(fs=FS, R=1500.0, M_a=0.03)
    )
    lp.set_state(_q0(case))
    lr.set_state(_q0(case))
    for _ in range(STEPS):
        lp.step()
        lr.step()
        assert np.array_equal(bp.q, br.q)
    assert lp.load.u_l == lr.load.u_l
    assert lp.radiated_energy == lr.radiated_energy
    assert lp.volume_velocity == lr.volume_velocity


def test_the_reactive_state_stays_inside_group_a_with_a_weighted_body():
    case = INEXACT_CASE
    bp, br = _bodies(case)
    lp = ReactiveRadiatedBodyPy(body=bp, load=RationalAirLoadPy(fs=FS, R=1500.0, M_a=0.03))
    lr = physsynth_rs.ReactiveRadiatedBody(
        body=br, load=physsynth_rs.RationalAirLoad(fs=FS, R=1500.0, M_a=0.03)
    )
    lp.set_state(_q0(case))
    lr.set_state(_q0(case))
    scale = float(np.max(np.abs(_q0(case))))
    worst = short = 0.0
    for n in range(STEPS):
        lp.step()
        lr.step()
        worst = max(worst, _amplitude_error(bp.q, br.q, scale))
        if n + 1 == SHORT:
            short = worst
    assert short <= GROUP_A_TOL, f"short run diverged by {short:.3e} of the amplitude"
    assert worst <= LONG_RUN_TOL, f"long run diverged by {worst:.3e} of the amplitude"


@pytest.mark.parametrize("kind", ["py", "rs"])
def test_an_infinite_radiation_mass_is_bit_identical_to_the_constant_r_load(kind):
    # The second exact reduction, again compared within one implementation. It survives only if
    # both loaded bodies share ONE copy of the rank-1 precomputes and `solve` keeps its operation
    # order -- forming `u*` first, then multiplying by `R_eff`.
    make_body = ModalBodyPy if kind == "py" else physsynth_rs.ModalBody
    make_flat = RadiatedBodyPy if kind == "py" else physsynth_rs.RadiatedBody
    make_load = RationalAirLoadPy if kind == "py" else physsynth_rs.RationalAirLoad
    make_reactive = (
        ReactiveRadiatedBodyPy if kind == "py" else physsynth_rs.ReactiveRadiatedBody
    )
    kw = dict(freqs=np.array([110.0, 196.0, 261.0, 440.0]), fs=FS)
    inner2, inner3 = make_body(**kw), make_body(**kw)
    batch2 = make_flat(body=inner2, R=R_DEFAULT)
    batch3 = make_reactive(body=inner3, load=make_load(fs=FS, R=R_DEFAULT, M_a=np.inf))
    batch2.set_state(_q0(kw))
    batch3.set_state(_q0(kw))
    for _ in range(2000):
        batch2.step()
        batch3.step()
        assert np.array_equal(inner2.q, inner3.q)
        assert batch2.radiated_energy == batch3.radiated_energy
    assert batch3.load.u_l == 0.0
    assert batch3.load.stored_energy() == 0.0


def test_the_timestep_mismatch_refusal_reads_the_same():
    body_py, body_rs = _bodies(EXACT_CASES[0])
    with pytest.raises(ValueError) as py_err:
        ReactiveRadiatedBodyPy(body=body_py, load=RationalAirLoadPy(fs=44100.0, R=1000.0))
    with pytest.raises(ValueError) as rs_err:
        physsynth_rs.ReactiveRadiatedBody(
            body=body_rs, load=physsynth_rs.RationalAirLoad(fs=44100.0, R=1000.0)
        )
    assert str(py_err.value) == str(rs_err.value)
    assert "load fs (44100.0) must match the body's (48000.0)" in str(rs_err.value)


@pytest.mark.parametrize("kind", ["py", "rs"])
def test_the_reactive_body_shares_the_load_object_it_was_given(kind):
    # `web/serialize.py` reads `loaded.load` and then asks it for `stored_energy()` and
    # `impedance()`; a copy here would report an air that had never been stepped.
    make_body = ModalBodyPy if kind == "py" else physsynth_rs.ModalBody
    make_load = RationalAirLoadPy if kind == "py" else physsynth_rs.RationalAirLoad
    make_reactive = (
        ReactiveRadiatedBodyPy if kind == "py" else physsynth_rs.ReactiveRadiatedBody
    )
    load = make_load(fs=FS, R=1500.0, M_a=0.03)
    inner = make_body(freqs=np.array([110.0, 196.0]), fs=FS)
    loaded = make_reactive(body=inner, load=load)
    assert loaded.load is load
    assert loaded.body is inner
    loaded.set_state(np.array([1e-3, -5e-4]))
    for _ in range(100):
        loaded.step()
    assert load.n == 100
    assert loaded.radiated_energy == load.radiated_energy
    assert loaded.volume_velocity == load.volume_velocity
    assert loaded.energy() == inner.energy() + load.energy()
