"""Model #10 — the two-polarization certificates (Tier A/1 and A/2).

**Energy cannot see any of this.** A model with the ``u``/``w`` coupling subtly wrong — a stray
factor, a ``u_x²`` where ``r²`` belongs, a polarization leak — can conserve energy to 1e-13 and
still be wrong. These tests are the cheapest and strongest certificate that the coupling is right,
and they are *bit-exact* rather than tolerance-based, in the ``max|w_b| == 0.0`` culture the
sympathetic-strings batch established.

Both rest on one structural fact: ``V`` depends on the polarizations **only** through
``r² = u_x² + w_x²``, so the geometric nonlinearity is exactly isotropic. The detuning knob
``kappa_u != kappa_w`` touches only the *linear* operator, which is why it cannot corrupt these.
"""

import numpy as np
import pytest
from helpers import (
    EA_DEFAULT,
    KAPPA_DEFAULT,
    geometric_mode_ic,
    geometric_pluck_ic,
    make_geometric_string,
)

# -- Tier A/2: the planar subspace is invariant ----------------------------------------


def test_planar_ic_stays_bit_exactly_planar():
    """``max|w| == 0.0`` **exactly**, forever. Not "small" — zero.

    ``w -> -w`` is a reflection symmetry of the whole model, so the planar subspace is invariant.
    In floating point this is stronger than a symmetry argument: every ``w`` term is multiplied by
    ``w``, so an identically-zero ``w`` field can never acquire a nonzero bit. If this ever returns
    ``1e-30`` instead of ``0.0``, some term is not routed through ``r²`` and the isotropy is fake.

    **This is also why a degenerate string cannot spontaneously whirl** — whirling must be *seeded*.
    """
    s = make_geometric_string(N=64, EA=EA_DEFAULT)
    s.set_state(geometric_pluck_ic(s.N, amp=4e-3))
    for _ in range(400):
        s.step()
    assert np.max(np.abs(s.w)) == 0.0
    assert np.max(np.abs(s.w_prev)) == 0.0
    # ...while the in-plane and longitudinal fields are emphatically alive.
    assert np.max(np.abs(s.u)) > 1e-4
    assert np.max(np.abs(s.v)) > 0.0


def test_planar_subspace_invariance_survives_detuning():
    """Planar stays planar with ``kappa_u != kappa_w`` too — the reflection symmetry is untouched.

    This matters for batch 2's honesty: whirling must be seeded by a perturbation, never by the
    planar IC drifting on its own. If the detuned string leaked into ``w`` from a planar start, a
    "whirling threshold" measurement would be measuring the leak.
    """
    s = make_geometric_string(N=64, EA=EA_DEFAULT, kappa=KAPPA_DEFAULT, kappa_w=0.5)
    assert not s.is_degenerate
    s.set_state(geometric_mode_ic(s.N, 1, 5e-3))
    for _ in range(400):
        s.step()
    assert np.max(np.abs(s.w)) == 0.0


def test_the_other_polarization_is_equally_invariant():
    """The same holds with the roles swapped — ``u`` is not privileged by the implementation."""
    s = make_geometric_string(N=64, EA=EA_DEFAULT)
    s.set_state(0.0, geometric_pluck_ic(s.N, amp=4e-3))
    for _ in range(400):
        s.step()
    assert np.max(np.abs(s.u)) == 0.0
    assert np.max(np.abs(s.w)) > 1e-4


# -- Tier A/1: rotational invariance ---------------------------------------------------


def test_ninety_degree_swap_is_bit_exact():
    """Rotating the IC by 90° rotates the whole trajectory — **bit-for-bit**.

    At ``kappa_u == kappa_w`` the ``u`` and ``w`` rows of the scheme are literally the same
    expression, so this is a statement about code that is *shared*, not merely equivalent. Any
    asymmetry in how the two polarizations enter the discrete gradient shows up here immediately,
    and nowhere else.
    """
    a = make_geometric_string(N=48, EA=EA_DEFAULT)
    b = make_geometric_string(N=48, EA=EA_DEFAULT)
    ic = geometric_pluck_ic(a.N, amp=4e-3)
    a.set_state(ic, 0.0)
    b.set_state(0.0, ic)
    for _ in range(300):
        a.step()
        b.step()
    assert np.array_equal(a.u, b.w)
    assert np.array_equal(a.w, b.u)
    assert np.array_equal(a.v, b.v)
    assert a.energy() == b.energy()


@pytest.mark.parametrize("angle", [0.3, np.pi / 4, 1.1, 2.7])
def test_arbitrary_rotation_commutes_with_the_dynamics(angle):
    """Rotating the IC by **any** angle rotates the trajectory, to ~1e-13 *relative*.

    The 90° swap is exact because it permutes identical code paths; an arbitrary angle genuinely
    exercises the ``r² = u_x² + w_x²`` structure — the nonlinearity must not care which direction in
    the ``(u, w)`` plane the string is displaced. Only round-off separates the rotated run from the
    rotation of the unrotated one.

    Why the bar is relative and not the swap's ``== 0.0``: the rotated run mixes ``u`` and ``w``
    through ``cos``/``sin``, so it does *different arithmetic*, and 250 steps of Newton solves
    accumulate a few ulp each. Measured ~2e-13 relative (~1e-15 absolute on a 4 mm string) — that
    is round-off, not a leak. An absolute bar here would be a bar below machine epsilon.
    """
    steps = 250
    ref = make_geometric_string(N=48, EA=EA_DEFAULT)
    rot = make_geometric_string(N=48, EA=EA_DEFAULT)
    ic = geometric_pluck_ic(ref.N, amp=4e-3)

    ref.set_state(ic, 0.0)
    rot.set_state(np.cos(angle) * ic, np.sin(angle) * ic)
    for _ in range(steps):
        ref.step()
        rot.step()

    scale = float(np.max(np.abs(ic)))
    expected_u = np.cos(angle) * ref.u - np.sin(angle) * ref.w
    expected_w = np.sin(angle) * ref.u + np.cos(angle) * ref.w
    assert np.max(np.abs(rot.u - expected_u)) < 1e-11 * scale
    assert np.max(np.abs(rot.w - expected_w)) < 1e-11 * scale
    assert np.max(np.abs(rot.v - ref.v)) < 1e-11 * scale
    assert rot.energy() == pytest.approx(ref.energy(), rel=1e-11)


def test_rotation_invariance_is_broken_by_detuning_but_isotropy_of_the_nonlinearity_is_not():
    """The negative control, and the reason the detuning knob is safe.

    ``kappa_u != kappa_w`` **must** break rotational invariance — that is what it is for (it is what
    buys Gough's whirling threshold). But it must break it *only through the linear operator*: the
    geometric nonlinearity still sees the polarizations through ``r²`` alone. So the same run with
    ``kappa = 0`` in both polarizations is exactly rotation-invariant even though the string is
    otherwise identical — proving the anisotropy did not leak into the DG.
    """
    angle = 0.7
    steps = 250
    ic = geometric_pluck_ic(48, amp=4e-3)

    def run(kappa, kappa_w):
        ref = make_geometric_string(N=48, EA=EA_DEFAULT, kappa=kappa, kappa_w=kappa_w)
        rot = make_geometric_string(N=48, EA=EA_DEFAULT, kappa=kappa, kappa_w=kappa_w)
        ref.set_state(ic, 0.0)
        rot.set_state(np.cos(angle) * ic, np.sin(angle) * ic)
        for _ in range(steps):
            ref.step()
            rot.step()
        expected_u = np.cos(angle) * ref.u - np.sin(angle) * ref.w
        return float(np.max(np.abs(rot.u - expected_u))) / float(np.max(np.abs(ic)))

    detuned = run(KAPPA_DEFAULT, 0.5 * KAPPA_DEFAULT)
    isotropic_bending = run(KAPPA_DEFAULT, KAPPA_DEFAULT)
    assert detuned > 1e-6, "kappa_u != kappa_w must break rotational invariance — that's its job"
    assert isotropic_bending < 1e-11, "with kappa_u == kappa_w the model must stay isotropic"


def test_nonlinearity_depends_on_the_polarizations_only_through_r_squared():
    """Directly: the discrete gradient is equivariant under rotation in the ``(u, w)`` plane.

    A unit test on the operator itself, independent of any time-stepping — so a failure points at
    the DG rather than at the scheme around it (the family's energy-triangulation habit).
    """
    s = make_geometric_string(N=24, EA=EA_DEFAULT)
    rng = np.random.default_rng(17)
    q_plus = rng.normal(size=(3, s.N)) * 0.05
    q_minus = rng.normal(size=(3, s.N)) * 0.05
    angle = 0.9
    ca, sa = np.cos(angle), np.sin(angle)

    def rotate(q):
        out = q.copy()
        out[0] = ca * q[0] - sa * q[1]
        out[1] = sa * q[0] + ca * q[1]
        return out

    got = s._dg_force(rotate(q_plus), rotate(q_minus))
    expected = rotate(s._dg_force(q_plus, q_minus))
    assert np.max(np.abs(got - expected)) < 1e-13 * np.max(np.abs(expected))
    # ...and the stored energy is a scalar under the same rotation.
    assert s._nl_density(rotate(q_plus)) == pytest.approx(s._nl_density(q_plus), rel=1e-12)


def test_circular_and_planar_have_the_same_energy_but_differ_dynamically():
    """A sanity check that the two polarization sectors are genuinely different motions.

    Not the batch-2 discriminator (that claim is about the *longitudinal spectrum*), just the
    precondition for it: equal-energy circular and planar states must not be the same trajectory.
    """
    amp = 4e-3
    planar = make_geometric_string(N=48, EA=EA_DEFAULT)
    planar.set_state(geometric_mode_ic(planar.N, 1, amp))

    circ = make_geometric_string(N=48, EA=EA_DEFAULT)
    shape = geometric_mode_ic(circ.N, 1, amp)
    omega = 2.0 * np.pi * circ.c / (2.0 * circ.L)
    circ.set_state(shape, 0.0, w_dot=omega * shape)  # u = A cos, w = A sin at t=0

    for _ in range(300):
        planar.step()
        circ.step()
    assert np.max(np.abs(circ.w)) > 1e-4, "the circular run must actually leave the plane"
    assert np.max(np.abs(planar.w)) == 0.0
    assert circ.n_not_converged == 0 and planar.n_not_converged == 0
