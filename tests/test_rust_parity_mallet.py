"""Rust vs NumPy/SciPy for the mallet — the batch that finishes Phase 2, and the shortest one.

``docs/dev/rust-migration-plan.md`` section 17. Both of this model's hard parts were ported
before it: the drumhead in Phase 2 batch 1 (bit-identical over 2,000 fed-back steps) and the
contact root-find in Phase 3 batch 2 (bit-identical over 18,000 configurations, and through the
mallet's own fixtures with the bracket fallback firing). So what this file is testing is the
**shell** — the force-free flight integrator, the two force-injection sites, and the admittances
that scale the force — and section 16.11 wrote down in advance that any divergence here would be
the shell's. That is the prediction; the assertions below are what makes it falsifiable.

Three things worth reading before the assertions.

1. **Both sides are pinned to a named implementation, never to the swapped module name.** With
   ``PHYSSYNTH_RS`` set, ``mallet.MalletMembrane`` and ``membrane.Membrane`` are *already* the
   Rust ones, so a comparison that only pinned the Rust side would be comparing Rust against Rust
   and passing for the wrong reason. Every test here names ``MalletMembranePy`` on ``MembranePy``
   for one side and ``physsynth_rs`` for the other, so the file measures the same thing with the
   flag set and unset. (Under the flag ``MembranePy`` builds its Laplacian with the Rust
   ``operators2d``; that builder was proved bit-identical down to ``indices`` and ``nnz`` in
   Phase 1, so the two experiments coincide rather than merely resembling each other.)

2. **``MalletWall`` is the attribution, not a second smoke test.** The coupled model's
   ``energy()`` calls ``Membrane.energy()``, which is a **reduction** — ``np.dot`` against a
   left-to-right sum — and section 14.2 established that a reduction is where bit-identity ends.
   The wall rig owns no field and performs no reduction, so it must be bit-identical in *every*
   observable including its energy. Asserting both is what turns "the coupled energy differs by
   2e-16" from an observation into an attribution: the difference is the drumhead's read-out and
   nothing in this batch.

3. **The fallback counter is compared step by step, not at the end.** Section 16.6 made that a
   convention: a bracketed fallback taken at a different step is a *branch* difference, and the
   penetrations might still absorb it into a number that looks fine. Measured 2026-08-27, the
   ``alpha = 1`` fixture takes it 27 times in 2,000 steps, so this is exercised rather than
   hypothetical.
"""

import warnings

import numpy as np
import pytest
from helpers import RADIUS_DEFAULT, RHO_AREAL_DEFAULT, T_DEFAULT

from physsynth.core import mallet as M
from physsynth.core import membrane as ME

physsynth_rs = pytest.importorskip(
    "physsynth_rs", reason="the Rust extension is not built in this environment"
)

SHORT_RUN = 2000        # long enough for a bit-identity claim to mean something
GROUP_A_TOL = 1e-13     # the plan's short-run agreement target
DRIFT_TOL = 1e-10       # CLAUDE.md's energy bar, which neither implementation may cross

MASS = 0.02             # kg    -- tests/helpers.MALLET_MASS_DEFAULT
STIFFNESS = 5.0e4       # N/m^a -- tests/helpers.MALLET_K_DEFAULT
ALPHA = 2.3             # piano-ish felt
VELOCITY = 3.0          # m/s


def _membrane_kw(*, N, lam, sigma, domain):
    """``helpers.make_membrane``'s parameters, spelled out so the class can be chosen per side."""
    c = float(np.sqrt(T_DEFAULT / RHO_AREAL_DEFAULT))
    h = 1.0 / N if domain == "rectangle" else 2.0 * RADIUS_DEFAULT / N
    kw = dict(domain=domain, T=T_DEFAULT, rho=RHO_AREAL_DEFAULT, fs=c / (lam * h), N=N,
              sigma=sigma)
    if domain == "rectangle":
        kw.update(Lx=1.0, Ly=1.0)
    else:
        kw.update(radius=RADIUS_DEFAULT)
    return kw


def _mallet_pair(*, N=40, lam=0.5, sigma=0.0, domain="rectangle", alpha=ALPHA, hysteresis=0.0,
                 strike=(0.5, 0.5), gap=0.0):
    """One Python mallet on a Python drumhead and one Rust mallet on a Rust drumhead.

    Two independent membranes rather than one shared object: the mallet writes into its head's
    field every step, so a shared drumhead would make the second run start from the first one's
    state. Both are constructed from the identical keyword set.
    """
    mkw = _membrane_kw(N=N, lam=lam, sigma=sigma, domain=domain)
    kw = dict(mass=MASS, stiffness=STIFFNESS, alpha=alpha, hysteresis=hysteresis,
              strike_x=strike[0], strike_y=strike[1], strike_velocity=VELOCITY, gap=gap)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        py = M.MalletMembranePy(membrane=ME.MembranePy(**mkw), **kw)
        rs = physsynth_rs.MalletMembrane(membrane=physsynth_rs.Membrane(**mkw), **kw)
    return py, rs


def _wall_pair(*, alpha=1.0, hysteresis=0.0, fs=96000.0, wall_position=0.0, gap=0.0):
    kw = dict(mass=MASS, stiffness=STIFFNESS, fs=fs, alpha=alpha, hysteresis=hysteresis,
              wall_position=wall_position, strike_velocity=2.0, gap=gap)
    return M.MalletWallPy(**kw), physsynth_rs.MalletWall(**kw)


FIXTURES = [
    ("default", {}),
    ("linear felt", {"alpha": 1.0}),
    ("hysteretic felt", {"hysteresis": 1.0e3}),
    ("lossy head", {"sigma": 2.0}),
    ("off-centre strike", {"strike": (0.37, 0.62)}),
    ("with a gap", {"gap": 5.0e-4}),
    ("disk", {"domain": "circle", "N": 32, "strike": (0.0, 0.0)}),
]
FIXTURE_IDS = [name for name, _ in FIXTURES]
FIXTURE_KW = [kw for _, kw in FIXTURES]


# -- 1. the shell's constants ---------------------------------------------------------------------

@pytest.mark.parametrize("kw", FIXTURE_KW, ids=FIXTURE_IDS)
def test_the_derived_constants_are_bit_identical(kw):
    """The admittances, the snapped strike node and its coordinates.

    ``_g_s`` and ``_g_h`` multiply the contact force at *every* timestep, so a last-bit error in
    either is a last-bit error in the state of every step of every run -- while conserving energy
    perfectly, which is why no physics bar could catch it. They are built from ``k**2`` and
    ``h**2``, which on Python floats is the C library's ``pow`` and **not** ``x * x``; measured
    2026-08-27 over 400,000 samples the two spellings disagree in 225. This assertion is what
    holds the port to the right one of them.
    """
    py, rs = _mallet_pair(**kw)
    assert rs.node == py.node
    assert rs.x_strike == py.x_strike
    assert rs.y_strike == py.y_strike
    assert rs._g_s == py._g_s
    assert rs._g_h == py._g_h
    assert rs._g == py._g
    assert rs.contact_frequency == py.contact_frequency
    assert rs.k == py.k
    assert rs.penetration == py.penetration
    assert rs.in_contact == py.in_contact


def test_the_squaring_is_pow_and_not_a_multiply():
    """The same claim as above, isolated from any model so a failure names its own cause.

    ``k**2 / M`` with ``k = 1/fs``: the wall rig's only admittance, and the shortest expression in
    the project that can distinguish the two spellings.

    **The equality is the contract; the inequality is a platform fact.** Whether ``x ** 2`` and
    ``x * x`` are different doubles depends on how correctly the C library rounds ``pow``, which
    varies by platform. A red CI run on 2026-08-27 taught that one file over -- see plan section
    17.2, where the deeper cause turned out to be LLVM folding a literal exponent -- so the
    inequality below is asserted only where this machine makes it observable. The Rust side carries
    the ``pow`` spelling either way, which is a superset and therefore portable.
    """
    fs = 1.0 / 0.005419571191933292  # a k where Windows' pow(k, 2) and k * k are different doubles
    k = 1.0 / fs
    _, rs = _wall_pair(fs=fs)
    assert rs._g == k ** 2 / MASS, "the Rust admittance is not Python's ** 2"
    if k ** 2 != k * k:
        assert rs._g != k * k / MASS, "the Rust admittance folded ** 2 into a multiply"


# -- 2. the standalone rig: bit-identical in EVERY observable, energy included --------------------

@pytest.mark.parametrize("kw", [
    {},
    {"alpha": 2.3},
    {"alpha": 1.5, "hysteresis": 5.0e3},
    {"wall_position": 1.0e-3},
    {"gap": 2.0e-4},
], ids=["linear", "hard felt", "hysteretic", "raised wall", "with a gap"])
def test_the_wall_rig_is_bit_identical(kw):
    """No field, no reduction -- so this one has nowhere for a difference to come from, and its
    energy is held to equality rather than to a tolerance. That is what attributes the coupled
    model's 2e-16 energy gap to ``Membrane.energy()`` instead of to the contact scheme."""
    py, rs = _wall_pair(**kw)
    for i in range(SHORT_RUN):
        py.step()
        rs.step()
        assert rs.z_H == py.z_H, f"step {i}"
        assert rs.z_H_prev == py.z_H_prev, f"step {i}"
        assert rs.penetration == py.penetration, f"step {i}"
        assert rs.contact_force == py.contact_force, f"step {i}"
        assert rs.in_contact == py.in_contact, f"step {i}"
        assert rs.fallbacks == py.fallbacks, f"step {i}"
        assert rs.velocity() == py.velocity(), f"step {i}"
        assert rs.energy() == py.energy(), f"step {i}"
    assert py.penetration is not None and max(py.n, rs.n) == SHORT_RUN


def test_the_wall_rig_actually_makes_contact():
    """A guard against the bit-identity above being a comparison of two mallets flying past each
    other in free space, where every quantity is trivially equal."""
    py, rs = _wall_pair()
    deepest = 0.0
    for _ in range(SHORT_RUN):
        py.step()
        rs.step()
        deepest = max(deepest, py.penetration)
    assert deepest > 0.0, "the mass never touched the wall"
    assert py.velocity() > 0.0, "the mass never rebounded"


# -- 3. the coupled model: the trajectory is bit-identical, the energy is not ----------------------

@pytest.mark.parametrize("kw", FIXTURE_KW, ids=FIXTURE_IDS)
def test_the_coupled_trajectory_is_bit_identical(kw):
    """Every state variable the model owns, plus the drumhead's whole field, step for step.

    The field matters more than the scalars: the mallet writes one node of it per step and the
    membrane's explicit stencil spreads that write over the grid, so a last-bit error in the
    injection shows up here long before it shows up in the penetration.
    """
    py, rs = _mallet_pair(**kw)
    for i in range(SHORT_RUN):
        py.step()
        rs.step()
        assert rs.z_H == py.z_H, f"step {i}"
        assert rs.z_H_prev == py.z_H_prev, f"step {i}"
        assert rs.penetration == py.penetration, f"step {i}"
        assert rs.contact_force == py.contact_force, f"step {i}"
        assert rs.in_contact == py.in_contact, f"step {i}"
        assert rs.fallbacks == py.fallbacks, f"step {i}"   # section 16.6: a branch, not a rounding
        assert rs.mallet_velocity() == py.mallet_velocity(), f"step {i}"
    np.testing.assert_array_equal(np.asarray(rs.membrane.u), np.asarray(py.membrane.u))
    np.testing.assert_array_equal(np.asarray(rs.membrane.u_prev), np.asarray(py.membrane.u_prev))
    np.testing.assert_array_equal(np.asarray(rs.state), np.asarray(py.state))


def test_the_bracket_fallback_fires_and_fires_at_the_same_steps():
    """Section 16.3 measured the ``brentq`` fallback firing eight times per 3,000 steps at
    ``alpha = 1``; through the ported shell it fires 27 times in 2,000. If it ever stops firing,
    the test above becomes a weaker claim than it reads as -- so the count is asserted to be
    non-zero as well as equal."""
    py, rs = _mallet_pair(alpha=1.0)
    for i in range(SHORT_RUN):
        py.step()
        rs.step()
        assert rs.fallbacks == py.fallbacks, f"step {i}"
    assert py.fallbacks > 0, "the scanned-bracket fallback never fired; this fixture proves less"


@pytest.mark.parametrize("kw", FIXTURE_KW, ids=FIXTURE_IDS)
def test_the_coupled_energy_agrees_to_group_a_and_only_because_of_the_head(kw):
    """The one observable this batch does not claim to the bit, and the reason is upstream.

    ``MalletMembrane.energy()`` is ``Membrane.energy() + KE + PE``. The last two terms are pinned
    to equality by the trajectory test above; the first is two ``np.dot`` reductions, which is
    section 14.2's construction. So the residual here IS the drumhead's read-out, and it is a
    read-out rather than a feedback path -- it never reaches the next timestep, which is why the
    trajectory stays identical while this number does not.
    """
    py, rs = _mallet_pair(**kw)
    e0 = abs(py.energy())
    worst = 0.0
    for _ in range(SHORT_RUN):
        py.step()
        rs.step()
        worst = max(worst, abs(rs.energy() - py.energy()))
    assert worst / e0 <= GROUP_A_TOL, f"energy gap {worst / e0:.2e} of amplitude"
    # And the terms that are not the head's reduction are exactly equal, which localises it.
    assert rs.mallet_velocity() == py.mallet_velocity()
    assert rs.penetration == py.penetration


@pytest.mark.parametrize("kw", FIXTURE_KW, ids=FIXTURE_IDS)
def test_neither_implementation_crosses_the_projects_energy_bar(kw):
    """The physics bar, which is the claim that survives past any agreement window: a lossless
    run conserves and a lossy one is monotone. Asserted on both sides so a future batch that
    breaks the tie can see which one moved."""
    py, rs = _mallet_pair(**kw)
    lossless = kw.get("sigma", 0.0) == 0.0 and kw.get("hysteresis", 0.0) == 0.0
    for model in (py, rs):
        e0 = model.energy()
        prev = e0
        worst = 0.0
        for _ in range(SHORT_RUN):
            model.step()
            now = model.energy()
            worst = max(worst, abs(now - e0))
            if not lossless:
                assert now - prev <= 1e-12 * abs(e0), "a lossy run gained energy"
            prev = now
        if lossless:
            assert worst / abs(e0) < DRIFT_TOL, f"drift {worst / abs(e0):.2e} (bar {DRIFT_TOL})"


# -- 4. the seams the swap depends on --------------------------------------------------------------

def test_the_rust_mallet_refuses_a_python_drumhead():
    """A silent fallback here would be a Rust mallet reporting Rust while striking a Python
    membrane -- the green-and-meaningless run the whole swap guard exists to prevent. The reed
    established the rule (plan section 12.8); this is the second model to need it."""
    mkw = _membrane_kw(N=16, lam=0.5, sigma=0.0, domain="rectangle")
    with pytest.raises(TypeError, match="needs a Rust Membrane"):
        physsynth_rs.MalletMembrane(
            membrane=ME.MembranePy(**mkw), mass=MASS, stiffness=STIFFNESS,
            strike_x=0.5, strike_y=0.5, strike_velocity=VELOCITY,
        )


def test_membrane_is_the_object_that_was_passed_in():
    """``tests/test_mallet_energy.py`` reads the head's field through ``mal.membrane`` and
    ``web/serialize.py`` calls ``energy()`` on it every step of the audio run, so this handle has
    to be identity and not a copy -- and the field it exposes has to be the one the mallet just
    wrote into."""
    mkw = _membrane_kw(N=16, lam=0.5, sigma=0.0, domain="rectangle")
    mem = physsynth_rs.Membrane(**mkw)
    mal = physsynth_rs.MalletMembrane(
        membrane=mem, mass=MASS, stiffness=STIFFNESS, strike_x=0.5, strike_y=0.5,
        strike_velocity=VELOCITY,
    )
    assert mal.membrane is mem
    for _ in range(50):
        mal.step()
    assert np.any(np.asarray(mem.u) != 0.0), "the strike never reached the head"
    assert mal.energy() == pytest.approx(mem.energy() + 0.5 * MASS * mal.mallet_velocity() ** 2,
                                         abs=1.0)


@pytest.mark.parametrize("bad,message", [
    ({"mass": 0.0}, "mallet mass must be > 0."),
    ({"stiffness": -1.0}, "felt stiffness K must be > 0."),
    ({"alpha": 0.5}, "felt exponent alpha must be >= 1."),
    ({"hysteresis": -1.0}, "hysteresis lambda_h must be >= 0."),
    ({"gap": -1e-6}, "initial gap must be >= 0."),
])
def test_the_refusals_are_the_same_text_in_the_same_order(bad, message):
    mkw = _membrane_kw(N=16, lam=0.5, sigma=0.0, domain="rectangle")
    kw = dict(mass=MASS, stiffness=STIFFNESS, alpha=ALPHA, hysteresis=0.0,
              strike_x=0.5, strike_y=0.5, strike_velocity=VELOCITY, gap=0.0)
    kw.update(bad)
    for cls, mem in ((M.MalletMembranePy, ME.MembranePy(**mkw)),
                     (physsynth_rs.MalletMembrane, physsynth_rs.Membrane(**mkw))):
        with pytest.raises(ValueError, match=message.replace(".", r"\.")):
            cls(membrane=mem, **kw)
    wall_kw = dict(mass=MASS, stiffness=STIFFNESS, fs=96000.0, alpha=1.0, hysteresis=0.0,
                   strike_velocity=2.0, gap=0.0)
    wall_kw.update(bad)
    for cls in (M.MalletWallPy, physsynth_rs.MalletWall):
        with pytest.raises(ValueError, match=message.replace(".", r"\.")):
            cls(**wall_kw)


def test_a_scalar_fault_is_reported_before_a_wrong_drumhead():
    """The original checks its five scalars before it touches the membrane. The Rust class has to
    cast the membrane before it can derive anything, so the check order is restored explicitly --
    a call that is wrong in both ways must still report the mass."""
    mkw = _membrane_kw(N=16, lam=0.5, sigma=0.0, domain="rectangle")
    with pytest.raises(ValueError, match=r"mallet mass must be > 0\."):
        physsynth_rs.MalletMembrane(
            membrane=ME.MembranePy(**mkw), mass=0.0, stiffness=STIFFNESS,
            strike_x=0.5, strike_y=0.5, strike_velocity=VELOCITY,
        )


def test_the_under_resolved_warning_says_the_same_thing_from_the_same_frame():
    """Both classes warn with the same text, and both blame *this* line rather than their own
    constructor. The original passes ``stacklevel=2`` from a Python ``__init__``; a Rust
    ``__new__`` pushes no Python frame, so it passes 1 to mean the same thing. That arithmetic is
    invisible to every other test, and wrong in either direction it would point the user's warning
    at library code."""
    # A stiff felt on a coarse grid: pi*sqrt(M/K)/k well under 8 steps.
    mkw = _membrane_kw(N=8, lam=0.5, sigma=0.0, domain="rectangle")
    kw = dict(mass=1e-4, stiffness=1.0e8, alpha=ALPHA, strike_x=0.5, strike_y=0.5,
              strike_velocity=VELOCITY)
    seen = []
    for cls, mem in ((M.MalletMembranePy, ME.MembranePy(**mkw)),
                     (physsynth_rs.MalletMembrane, physsynth_rs.Membrane(**mkw))):
        with pytest.warns(UserWarning, match="stiff contact under-resolved") as rec:
            cls(membrane=mem, **kw)
        assert len(rec) == 1
        seen.append((str(rec[0].message), rec[0].filename, rec[0].lineno))
    assert seen[0][0] == seen[1][0], "the warning text differs between implementations"
    assert seen[0][1] == seen[1][1] == __file__, "the warning did not blame the caller"
    assert seen[0][2] == seen[1][2], "the two warnings blame different lines of the caller"
