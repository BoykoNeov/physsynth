"""The 3-D air box: the cross-tier oracles — the room *contains* the lumped air, and the bore.

This is the batch's headline. :mod:`physsynth.core.radiation` and :mod:`physsynth.core.bore` are
independent implementations of the same physics in lower-dimensional disguises; if the box is right,
it must reproduce both in the limits where they apply.

- **Free field = the lumped monopole (§7.8).** Put the walls far away, drive a point source, and
  inside the window before the first reflection arrives the room must reproduce
  ``p = rho0 Q''(t - r/c0) / (4 pi r)`` — batch 1's :class:`AirRadiation` law, with batch 1's own
  gain constant. Because that closed form *already contains* ``1/(4 pi r)``, a least-squares fit of
  a single gain per radius makes ``gain == 1`` **the inverse-distance law itself** — a far better
  estimator than a log-log slope fit on dispersive pulse peaks.
- **Wall reflection (§7.9).** A pulse onto a locally-reacting impedance face reflects with
  ``R = (zeta - 1)/(zeta + 1)``.
- **The bore (§7.10).** A box one cell thick in ``y`` and ``z`` *is* a 1-D duct, and must track the
  repo's :class:`Bore` step for step.
- **The full chain (§7.11).** ``string -> bridge -> ReactiveRadiatedBody -> AirBox`` runs with no
  edits to ``body.py``, ``connection.py`` or ``radiation.py``.

Two traps govern the free-field measurement, and both were found by measuring:

1. **The reflection-free window is per-probe and hard-edged.** With pressure *on* the walls, a rigid
   wall is a pressure **antinode** — the first reflection arrives at full amplitude, with no
   forgiving roll-off. At radius ``r`` in a box of side ``L`` with the source centred the clean
   window is ``t in [r/c0, (L-r)/c0]``, which **collapses to zero as r -> L/2**. Running every probe
   to one global stop time truncates the far probes' pulses before they arrive and turns the
   measured slope into ``-2.5``.
2. **The arrival time is fitted, not asserted.** Batch 1's delay line is integer-sample and
   dispersionless; the FDTD arrival is dispersive with an ``O(h)`` effective source origin, so the
   two cannot agree on timing to better than a sample and it would be dishonest to demand it. So
   **assert the gain, report the lag** — and let the lag search run *negative*, or it pins at its
   boundary and inflates the residual from 9e-3 to 0.59.

   The measured lag is worth recording because it is **not** what the plan predicted. A prototype
   that injected the source *before* the velocity sub-step measured a constant ``-1.5`` samples;
   with the shipped ordering (pressure, walls, then velocity — :class:`Bore`'s, adopted so that
   ``energy()`` is a pure state function) the source enters half a step later and the offset is
   **gone**: measured ``|lag| < 0.4`` samples at every radius, drifting by ~0.5 of a sample across
   the radius range as dispersion accumulates. The offset was an artefact of *where in the step* the
   source was applied, not a property of the read-out.
"""

import numpy as np
import pytest
from helpers import gaussian_pulse, make_airbox, make_bridge

from physsynth.core.airbox import C0_AIR, RHO0_AIR, AirBox, impedance_from_zeta
from physsynth.core.bore import Bore
from physsynth.core.connection import StringBodyBridge
from physsynth.core.exciter import triangular_pluck
from physsynth.core.radiation import AirRadiation, RationalAirLoad, ReactiveRadiatedBody


def _free_field_fit(N, *, L=1.0, f0=1400.0, cfl=0.9):
    """Drive a centred point source in a big rigid room; fit (gain, lag) per radius.

    Returns ``(radii, gains, lags, residuals, box)``. The reference is batch 1's own monopole gain
    ``AirRadiation(distance=r).gain``, so a fitted gain of 1 says the distributed tier reproduces
    the lumped tier — not merely "something proportional to 1/r".
    """
    box = make_airbox(L=(L, L, L), h=L / N, cfl=cfl)
    q, qdot, pulse_seconds = gaussian_pulse(box.fs, f0)
    centre = tuple(0.5 * v for v in box.L_actual)
    box.source_index = box.node_index(centre)
    x0, y0, z0 = box.snapped(centre)

    # Trap 1: size the radii so every probe gets a full pulse inside its own window.
    r_max = 0.5 * (L - C0_AIR * pulse_seconds)
    r_min = max(6.0 * box.h, 0.3 * r_max)
    cells = list(range(int(np.ceil(r_min / box.h)), int(np.floor(r_max / box.h)) + 1))
    cells = cells[:: max(1, len(cells) // 6)]
    probes = [(x0 + c * box.h, y0, z0) for c in cells]
    # The SNAPPED radius is what the closed form gets evaluated at (here exact by construction,
    # but taken through snapped() so the discipline is the code's, not the test author's).
    radii = np.array([box.snapped(p)[0] - x0 for p in probes])

    steps = int(np.floor((L - radii.min()) / C0_AIR / box.k))
    rec = np.zeros((steps, len(probes)))
    box.set_state(np.zeros(box.p.shape))
    for n in range(steps):
        box.inject(q(n * box.k))
        box.step()
        for j, point in enumerate(probes):
            rec[n, j] = box.pressure_at(point)

    t = np.arange(steps) * box.k
    gains, lags, resids = [], [], []
    for j, r in enumerate(radii):
        window = t < (L - r) / C0_AIR  # trap 1: per-probe, hard-edged
        tw, sw = t[window], rec[window, j]
        gain_far = AirRadiation(fs=box.fs, distance=float(r)).gain  # batch 1's own constant
        best = None
        for d in np.linspace(-3.0, 3.0, 121) * box.k:  # trap 2: the lag search must go negative
            ref = gain_far * qdot(tw - r / C0_AIR - d)
            g = float(np.dot(sw, ref) / np.dot(ref, ref))
            err = float(np.max(np.abs(sw - g * ref)) / np.max(np.abs(g * ref)))
            if best is None or err < best[2]:
                best = (g, d / box.k, err)
        gains.append(best[0])
        lags.append(best[1])
        resids.append(best[2])
    return radii, np.array(gains), np.array(lags), np.array(resids), box


# -- §7.8 the headline: the distributed air contains the lumped air ----------------------------
def test_free_field_reproduces_the_monopole_law():
    """``gain == 1`` at every radius **is** the ``1/r`` law, because the reference already carries
    ``rho0 / (4 pi r)``. The lag is *reported*, never asserted (trap 2)."""
    radii, gains, lags, resids, box = _free_field_fit(48)
    report = " ".join(f"r={r * 100:.0f}cm g={g:.4f} lag={l:+.2f}sa" for r, g, l in
                      zip(radii, gains, lags, strict=True))
    assert np.max(np.abs(gains - 1.0)) < 3e-2, report
    assert np.max(resids) < 0.25, report
    # Reported, not asserted as a physical claim -- but pinned as sub-sample, because a *whole*
    # sample of offset would mean the source or the read-out sits at the wrong instant.
    assert np.max(np.abs(lags)) < 1.0, f"the arrival is off by a whole sample or more: {lags}"
    assert lags.max() - lags.min() < 1.0, f"lag should barely move with r: {lags}"


def test_free_field_gain_improves_with_refinement():
    """Falls with refinement — and deliberately **not** stated as a convergence order. The radii
    themselves move with ``N`` (trap 1 ties the usable range to ``h`` and to the pulse), so the
    implied rate is not a property of the scheme; a fixed-radius study gives a different number."""
    coarse = np.max(np.abs(_free_field_fit(32)[1] - 1.0))
    fine = np.max(np.abs(_free_field_fit(48)[1] - 1.0))
    assert fine < coarse, f"coarse {coarse:.3e} -> fine {fine:.3e}"


def test_the_lumped_tier_agrees_with_the_same_closed_form():
    """The other half of the cross-tier claim: batch 1's :class:`AirRadiation` — an integer-sample
    delay line, a completely different construction — emits that same closed form. The box and the
    lumped port therefore agree *with each other*, via a law neither of them defines."""
    fs, r = 40000.0, 0.4
    air = AirRadiation(fs=fs, distance=r)
    _, qdot, _ = gaussian_pulse(fs, 1400.0)
    out = np.array([air.process(qdot(n / fs)) for n in range(400)])
    t = np.arange(400) / fs
    expect = air.gain * qdot(t - air.latency_samples / fs)
    # Skip the wavefront-in-transit prologue: batch 1 emits documented silence for the first
    # latency_samples calls, while the closed form has the pulse's leading tail already arriving.
    live = slice(air.latency_samples, None)
    assert np.max(np.abs(out[live] - expect[live])) < 1e-14 * np.max(np.abs(expect))
    assert np.all(out[: air.latency_samples] == 0.0)


# -- §7.9 the wall reflection coefficient --------------------------------------------------------
@pytest.mark.parametrize("zeta", [0.5, 1.0, 2.0, 5.0])
def test_wall_reflection_coefficient(zeta):
    """``|R| = |(zeta - 1)/(zeta + 1)|`` at normal incidence. A **convergence** tier, not machine
    precision — and note ``zeta = 1`` absorbs everything *here* only because the incidence is
    normal: a locally-reacting matched wall is not an anechoic chamber (which is why the free-field
    oracle above leans on windowing, never on absorption)."""
    h, Nx = 0.1, 400
    duct = make_airbox(L=(Nx * h, h, h), h=h, walls={"x1": impedance_from_zeta(zeta)})
    x = np.arange(Nx + 1) * h
    x0, width = 8.0, 0.6
    p0 = np.exp(-((x - x0) ** 2) / (2.0 * width**2))
    # A purely right-going plane wave: u = p / (rho0 c0) at the faces.
    face = 0.5 * (p0[:-1] + p0[1:]) / (RHO0_AIR * C0_AIR)
    duct.set_state(
        np.broadcast_to(p0[:, None, None], duct.p.shape).copy(),
        (np.broadcast_to(face[:, None, None], duct.ux.shape).copy(), 0.0 * duct.uy, 0.0 * duct.uz),
    )
    probe, arrival = int(round(20.0 / h)), (Nx * h - x0) / (C0_AIR * duct.k)
    incident = reflected = 0.0
    for n in range(int(arrival * 3.2)):
        duct.step()
        v = abs(duct.p[probe, 0, 0])
        if n < 0.9 * arrival:
            incident = max(incident, v)
        else:
            reflected = max(reflected, v)
    predicted = abs((zeta - 1.0) / (zeta + 1.0))
    assert abs(reflected / incident - predicted) < 0.02, (
        f"zeta={zeta}: measured |R|={reflected / incident:.4f}, closed form {predicted:.4f}"
    )


# -- §7.10 the bore, one dimension down ----------------------------------------------------------
def test_quasi_1d_box_tracks_the_repo_bore():
    """A one-cell-thick box against :class:`Bore`, same ``p^0``, 4000 steps.

    This is an ``allclose`` **cross-model** check, not a family reduction: ``Bore`` carries the area
    ``S`` through both updates and the box carries none, so the float operation order differs and
    bit-identity is *not* promised (it is asserted absent below, so nobody tightens it later).

    Match ``lambda`` on both sides. The 3-D CFL caps ``lambda <= 0.577``, so a quasi-1-D box **can
    never** reach the bore's ``lambda = 1`` dispersionless sweet spot; comparing a box at 0.51
    against a bore at 1.0 compares two different amounts of dispersion and disagrees for an
    uninteresting reason.

    The closed-open clarinet's **odd-harmonic series** ``f_n = (2n-1) c0/(4L)`` reaches the box
    *through* this agreement, inherited from ``test_bore_modal``. It is deliberately **not**
    re-asserted here by FFT peak-picking off a short box run: at 4000 steps the bin width is 12 Hz,
    and a Gaussian IC puts spectral nulls on some of the very partials being counted. A trace that
    matches the bore's to 1e-14 for 4000 steps is strictly stronger evidence, and far cheaper, than
    a spectrum that resolves five peaks badly.
    """
    L, N, fs = 0.5, 36, 48000.0
    h = L / N
    box = AirBox(L=(L, h, h), fs=fs, h=h, walls={"x1": "open"})
    bore = Bore(L=L, fs=fs, N=N, radius=0.008, boundary=("closed", "open"), sigma=0.0)
    assert box.lam == pytest.approx(C0_AIR / fs / h)  # the same lambda, by construction

    x = np.arange(N + 1) * h
    p0 = np.exp(-((x - 0.15) ** 2) / (2.0 * 0.02**2))
    p0[-1] = 0.0  # Bore.set_state pins its open end; match it so step 0 already agrees
    box.set_state(np.broadcast_to(p0[:, None, None], box.p.shape).copy())
    bore.set_state(p0.copy())

    probe = N // 3
    traces = np.empty((4000, 2))
    for n in range(4000):
        box.step()
        bore.step()
        traces[n] = (box.p[probe, 0, 0], bore.pressure_at(probe))
    err = np.max(np.abs(traces[:, 0] - traces[:, 1])) / np.max(np.abs(traces[:, 1]))
    assert err < 1e-12, f"box vs bore {err:.3e}"
    assert not np.array_equal(traces[:, 0], traces[:, 1]), "bit-identity is not promised here"
    # The h/2 transverse half-cells must raise no dynamics of their own, even against the open face.
    assert np.max(np.abs(box.p - box.p[:, :1, :1])) == 0.0


# -- §7.11 the full chain, with zero edits elsewhere ---------------------------------------------
def test_string_bridge_body_air_load_drives_the_room():
    """``string -> bridge -> ReactiveRadiatedBody -> AirBox``, driven by the body's public
    ``volume_velocity``.

    Volume velocity is the primitive because it is what the continuity equation's source term *is*.
    The lumped tier's ``_VolumeAccelerationSource`` protocol hands out ``Q''`` instead, so the chain
    reads ``volume_velocity`` — the exact quantity, already public — and needs **no edits** to
    ``body.py``, ``connection.py`` or ``radiation.py``. Integrating ``Q''`` here to fake the
    protocol would introduce a DC drift mode with nothing to restore it, and it would be silent.
    """
    bridge = make_bridge(sigma_string=0.0, sigma_body=0.0, K=8000.0)
    fs = 1.0 / bridge.body.k
    loaded = ReactiveRadiatedBody(
        body=bridge.body, load=RationalAirLoad(fs=fs, R=1500.0, M_a=0.05)
    )
    chain = StringBodyBridge(string=bridge.string, body=loaded, K=8000.0)
    s = chain.string
    s.set_state(triangular_pluck(s.x, s.L, 0.3 * s.L, amplitude=1e-3))

    # The room has to run at the string's rate for the drive to be sample-aligned, so the CFL fixes
    # h -- a real 3-D constraint, not a tuning choice: h >= sqrt(3) c0 / fs.
    h = np.sqrt(3.0) * C0_AIR / fs
    room = AirBox(L=(8 * h, 6 * h, 5 * h), fs=fs, h=h, walls=RHO0_AIR * C0_AIR)
    room.set_state(np.zeros(room.p.shape))

    heard = 0.0
    for _ in range(400):
        chain.step()
        room.inject(loaded.volume_velocity)
        room.step()
        heard = max(heard, abs(room.pressure_at((6 * h, 3 * h, 2 * h))))
    assert heard > 0.0, "the room heard nothing"
    assert room.injected_energy() != 0.0
    # The room's own three-channel identity still closes while an external chain drives it.
    assert abs(room.energy()) / abs(room.injected_energy()) < 1e-9
