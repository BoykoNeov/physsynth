"""Physical-signature validation for the mallet -> plate strike (model #7p).

Conservation proves the scheme is *consistent*; these prove it is the right physics. A struck plate
speaks in many modes at once; a harder felt makes briefer contact and a brighter spectrum; the
mallet **bounces** and flies clear; and the strike position decides which modes speak — a strike on
a mode's node line cannot excite it.

The mode-null test is the sharp one, and it is sharper here than on the drumhead: the supported
plate's ``B = L @ L`` keeps ``sin(m pi x/Lx) sin(n pi y/Ly)`` an **exact** discrete eigenvector, so
"the (2,1) mode is not excited by a centre strike" is an identity of the scheme rather than an
approximation — provided the strike really is at the centre. On an even ``N`` the grid has a node
exactly at ``Lx/2``, and the test asserts the snapped point before it claims the null, because
otherwise the measurement is about where the strike landed rather than about symmetry.

Everything here runs on the **supported** branch. The free plate translates away under a point
strike (its ``{1, x, y}`` rigid nullspace), which is real physics but makes a spectrum a moving
target; ``test_mallet_plate.py`` owns that behaviour.
"""

import numpy as np
from helpers import make_mallet_plate, plate_mode_shape


def _spectral_centroid(sig, fs):
    """Amplitude-weighted mean frequency (Hz) of ``sig`` — a brightness proxy."""
    sig = sig - np.mean(sig)
    mag = np.abs(np.fft.rfft(sig * np.hanning(len(sig))))
    freqs = np.fft.rfftfreq(len(sig), d=1.0 / fs)
    return float(np.sum(freqs * mag) / (np.sum(mag) + 1e-30))


def _mode_energies(mal, modes, steps):
    """Peak |projection| of the plate field onto each analytic mode, over ``steps`` steps."""
    plate = mal.plate
    basis = np.asarray([plate_mode_shape(plate, m, n) for m, n in modes])
    basis = basis / np.linalg.norm(basis, axis=1, keepdims=True)
    peak = np.zeros(len(modes))
    for _ in range(steps):
        mal.step()
        peak = np.maximum(peak, np.abs(basis @ plate.u))
    return peak


def _pickup(mal, steps, index=None):
    """A displacement trace at a pickup node, plus the contact-force trace."""
    index = mal.plate.pickup_index_at(0.62 * mal.plate.Lx, 0.71 * mal.plate.Ly) if index is None \
        else index
    sig = np.empty(steps)
    force = np.empty(steps)
    for i in range(steps):
        mal.step()
        sig[i] = mal.displacement_at(index)
        force[i] = mal.contact_force
    return sig, force


def test_strike_excites_many_modes():
    # A single strike speaks in many modes rather than ringing one. Measured by projection onto the
    # analytic eigenmodes, which for this plate are exact.
    #
    # The threshold is 1% of the fundamental, not the drumhead suite's 5%, and the difference is
    # physics rather than slack: a plate's frequencies go like `m^2 + n^2` where a membrane's go
    # like `sqrt(m^2 + n^2)`, so the same impulse leaves a *displacement* comb that falls away far
    # faster -- an impulse gives each mode a velocity, and displacement is that over omega.
    # Measured here: 1.00, 0.195, 0.083, 0.037, 0.018, 0.028, 0.004.
    modes = [(1, 1), (2, 1), (1, 2), (2, 2), (3, 1), (1, 3), (3, 3)]
    mal = make_mallet_plate(N=24, strike_x=0.3, strike_y=0.4)
    peak = _mode_energies(mal, modes, 3000)
    strong = int(np.sum(peak > 0.01 * peak.max()))
    assert strong >= 6, f"only {strong} of {len(modes)} modes spoke: {peak / peak.max()}"


def test_a_centre_strike_nulls_the_antisymmetric_modes():
    # The mode comb. `sin(2 pi x/Lx)` vanishes at `x = Lx/2`, so a force applied exactly there
    # cannot drive the (2,1) or (1,2) modes -- and on this plate that is an identity, not an
    # approximation, because the sine product is an exact discrete eigenvector.
    #
    # ASSERT THE SNAP FIRST. With an even N the grid has a node at exactly Lx/2; if it did not,
    # a "null" would be a statement about where the strike landed.
    mal = make_mallet_plate(N=24, strike_x=0.5, strike_y=0.5)
    assert mal.x_strike == 0.5 and mal.y_strike == 0.5, (
        f"the centre strike snapped to ({mal.x_strike}, {mal.y_strike}); the null below would be "
        "measuring the snap"
    )
    modes = [(1, 1), (2, 1), (1, 2), (3, 3)]
    centred = _mode_energies(mal, modes, 2500)
    offset = _mode_energies(make_mallet_plate(N=24, strike_x=0.3, strike_y=0.4), modes, 2500)

    # (2,1) and (1,2) are dead at the centre, relative to the (1,1) that the same strike drives.
    for i, mode in [(1, (2, 1)), (2, (1, 2))]:
        assert centred[i] < 1e-10 * centred[0], (
            f"a centre strike drove {mode} at {centred[i] / centred[0]:.2e} of (1,1)"
        )
        assert offset[i] > 1e-3 * offset[0], (
            f"the off-centre control failed to drive {mode}; the null above proves nothing"
        )
    # (3,3) is symmetric about the centre and survives it -- so the null is about symmetry and not
    # about the centre being a quiet place to hit.
    assert centred[3] > 1e-3 * centred[0], "a symmetric mode should survive a centre strike"


def test_a_harder_felt_makes_a_shorter_brighter_strike():
    # Hardness is the felt's stiffness. A stiffer felt has a higher contact frequency
    # `sqrt(K/M)`, so the pulse is shorter -- and a shorter pulse has more high-frequency content
    # to hand the plate.
    durations, centroids = [], []
    for K in (2.0e4, 2.0e5):
        mal = make_mallet_plate(N=24, K=K)
        fs = mal.plate.fs
        sig, force = _pickup(mal, 3000)
        durations.append(int(np.sum(force != 0.0)))
        centroids.append(_spectral_centroid(sig, fs))
        assert mal.steps_per_contact >= 8.0, f"K={K:.0e} under-resolves its own contact"
    assert durations[1] < durations[0], (
        f"the harder felt was in contact longer ({durations[1]} vs {durations[0]} steps)"
    )
    assert centroids[1] > centroids[0], (
        f"the harder felt was not brighter ({centroids[1]:.1f} vs {centroids[0]:.1f} Hz)"
    )


def test_the_mallet_bounces_and_flies_clear():
    # A single contact then separation: the felt is one-sided, so once the mallet's velocity has
    # reversed and it has cleared the surface it never comes back within the window.
    mal = make_mallet_plate(N=24)
    in_contact = np.empty(4000, dtype=bool)
    for i in range(4000):
        mal.step()
        in_contact[i] = mal.in_contact
    # Count contiguous runs of contact.
    edges = np.diff(in_contact.astype(int))
    starts = int(np.sum(edges == 1)) + int(in_contact[0])
    assert starts == 1, f"the mallet made contact {starts} times, expected one clean bounce"
    assert not in_contact[-1], "the mallet never left the plate"
    assert mal.mallet_velocity() > 0.0, (
        f"the mallet did not rebound: velocity {mal.mallet_velocity():.3e} m/s"
    )


def test_the_felt_exponent_is_the_only_source_of_dynamic_timbre():
    """A softer strike is an exactly scaled copy at ``alpha = 1``, and stops being one above it.

    Everything in this model is linear except the felt: the plate is linear, the mallet is a mass,
    and the contact is one-sided but *scale-invariant* in its switching (the sign of ``eta`` does
    not depend on amplitude). So at ``alpha = 1`` the whole system is linear and the response must
    be **exactly proportional** to the strike velocity -- which it is, to 7.6e-13 across a fourfold
    change. Above ``alpha = 1`` the contact stiffens with penetration, a harder blow makes a
    briefer pulse, and the *shape* of the response changes with how hard you hit it.

    That is the piano's dynamic timbre, and this test locates it precisely: not in the plate, not in
    the collision scheme, but in one exponent. The departure is graded, which is the strongest form
    the claim comes in -- measured 1.9e-1 / 6.2e-1 / 7.1e-1 at ``alpha`` = 1.5 / 2.3 / 3.0.
    """
    def trace(alpha, v0, steps=2000):
        mal = make_mallet_plate(N=20, strike_velocity=v0, alpha=alpha)
        index = mal.plate.pickup_index_at(0.62 * mal.plate.Lx, 0.71 * mal.plate.Ly)
        out = np.empty(steps)
        for i in range(steps):
            mal.step()
            out[i] = mal.displacement_at(index)
        return out

    departures = []
    for alpha in (1.0, 1.5, 2.3, 3.0):
        loud, quiet = trace(alpha, 3.0), trace(alpha, 0.75)
        departures.append(
            float(np.max(np.abs(loud - 4.0 * quiet)) / np.max(np.abs(loud)))
        )

    assert departures[0] < 1e-10, (
        f"a linear felt did not give an exactly scaled response ({departures[0]:.2e}); then "
        "something other than the felt is amplitude-dependent, and this model has nothing else"
    )
    assert departures[1] > 0.05, "alpha = 1.5 should already break the scaling"
    assert departures == sorted(departures), (
        f"the departure from scaling is not monotone in the felt exponent: {departures}"
    )
