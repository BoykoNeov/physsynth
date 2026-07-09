"""Diagnostics for the "air" node (batch 1): free-space monopole radiation of the full instrument
chain string -> bridge -> body -> air (:class:`physsynth.core.radiation.AirRadiation`).

This is the read-out tier: the body's net **volume acceleration** ``Q'' = body.pressure()`` is
mapped to the far-field pressure a listener hears at distance ``r``,

    p_far(t, r) = rho0 / (4 pi r) * Q''(t - r / c0) ,

with no back-reaction (the radiation *load* — air damping the body, radiated-power energy channel —
is the next batch). The figures:

  1. **Source vs radiated pressure.** The body's volume acceleration and the radiated pressure at
     1 m: the same waveform scaled by ``rho0/(4 pi r)`` and shifted by the travel time ``r/c0``
     (visible as the initial silence while the wavefront is in transit).
  2. **Inverse-distance law.** The radiated pressure at r = 1, 2, 4 m collapses onto one curve when
     rescaled by ``r`` — the ``p ∝ 1/r`` monopole fall-off, measured.
  3. **Radiated spectrum = an acceleration-tilted body spectrum.** The far-field magnitude spectrum
     is the body's displacement spectrum times ``omega^2`` (pressure ∝ volume *acceleration*), so
     the high partials are lifted vs the string pickup. The body's modal frequencies are marked.

Run:  python scripts/diagnose_radiation.py
Figures land in ./out/.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from physsynth.core.body import ModalBody  # noqa: E402
from physsynth.core.connection import StringBodyBridge  # noqa: E402
from physsynth.core.exciter import triangular_pluck  # noqa: E402
from physsynth.core.radiation import AirRadiation  # noqa: E402
from physsynth.core.string_ideal import IdealString  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out")
os.makedirs(OUT, exist_ok=True)

# Canonical string (c = 200 m/s, f1 = 100 Hz) below lambda = 1 for coupling headroom, terminated on
# a small modal body (guitar-top-ish modes) through a bridge spring. A light frequency-independent
# loss on the body so the radiated note decays like a real one.
L, T, RHO = 1.0, 200.0, 0.005
N, LAM = 100, 0.9
c = np.sqrt(T / RHO)
fs = c * N / (L * LAM)

BODY_FREQS = np.array([110.0, 196.0, 261.0, 440.0])  # Hz
BODY_MASS = 0.02
K_BRIDGE = 8000.0


def build_chain(sigma_body: float = 6.0) -> StringBodyBridge:
    string = IdealString(L=L, T=T, rho=RHO, fs=fs, N=N, boundary=("fixed", "free"))
    body = ModalBody(freqs=BODY_FREQS, fs=fs, sigmas=sigma_body, masses=BODY_MASS)
    return StringBodyBridge(string=string, body=body, K=K_BRIDGE)


def run(bridge: StringBodyBridge, rad: AirRadiation, steps: int):
    """Pluck the string, step the chain, capture body volume acceleration and radiated pressure."""
    bridge.string.set_state(triangular_pluck(bridge.string.x, L, 0.3 * L, amplitude=1e-3))
    qdd = np.empty(steps)   # body volume acceleration Q''
    p = np.empty(steps)     # radiated far-field pressure
    pickup = np.empty(steps)
    for i in range(steps):
        bridge.step()
        qdd[i] = bridge.pressure()
        p[i] = rad.radiate(bridge)
        pickup[i] = bridge.displacement_at(N // 3)
    return qdd, p, pickup


def fig_source_vs_radiated():
    bridge = build_chain()
    rad = AirRadiation(fs=fs, distance=1.0)
    qdd, p, _ = run(bridge, rad, steps=int(0.05 * fs))
    t_ms = np.arange(qdd.size) / fs * 1e3

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    ax0.plot(t_ms, qdd, lw=0.6, color="tab:gray")
    ax0.set_ylabel("body volume accel.  Q''  (m³/s²)")
    ax0.set_title("String → bridge → body → air : source vs radiated pressure (r = 1 m)")
    ax1.plot(t_ms, p, lw=0.6, color="tab:blue")
    ax1.axvline(rad.retardation_seconds * 1e3, color="tab:red", ls="--", lw=1,
                label=f"wavefront arrival  r/c0 = {rad.retardation_seconds * 1e3:.2f} ms")
    ax1.set_ylabel("radiated pressure  p_far  (Pa)")
    ax1.set_xlabel("time (ms)")
    ax1.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    path = os.path.join(OUT, "radiation_source_vs_far_field.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def fig_inverse_distance():
    fig, ax = plt.subplots(figsize=(9, 4.5))
    steps = int(0.05 * fs)
    for r, color in [(1.0, "tab:blue"), (2.0, "tab:orange"), (4.0, "tab:green")]:
        bridge = build_chain()
        # retarded=False so the three distances align in time (isolate the 1/r amplitude law).
        rad = AirRadiation(fs=fs, distance=r, retarded=False)
        _, p, _ = run(bridge, rad, steps=steps)
        t_ms = np.arange(p.size) / fs * 1e3
        ax.plot(t_ms, r * p, lw=0.6, color=color, label=f"r·p  at r = {r:g} m")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("distance-scaled pressure  r · p_far  (Pa·m)")
    ax.set_title("Inverse-distance law: r·p collapses to one curve (p ∝ 1/r)")
    ax.set_xlim(0, 8)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(OUT, "radiation_inverse_distance.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def fig_radiated_spectrum():
    bridge = build_chain()
    rad = AirRadiation(fs=fs, distance=1.0, retarded=False)
    _, p, pickup = run(bridge, rad, steps=int(0.5 * fs))

    def spectrum(sig):
        sig = sig - np.mean(sig)
        win = np.hanning(sig.size)
        mag = np.abs(np.fft.rfft(sig * win))
        freq = np.fft.rfftfreq(sig.size, 1.0 / fs)
        return freq, mag / np.max(mag)

    f_pick, m_pick = spectrum(pickup)
    f_rad, m_rad = spectrum(p)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.semilogy(f_pick, m_pick, lw=0.7, color="tab:gray", label="string pickup (displacement)")
    ax.semilogy(f_rad, m_rad, lw=0.7, color="tab:blue", label="radiated pressure (∝ ω² accel.)")
    for fb in BODY_FREQS:
        ax.axvline(fb, color="tab:red", ls=":", lw=0.8)
    ax.set_xlim(0, 2000)
    ax.set_ylim(1e-4, 1.5)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("normalised magnitude")
    ax.set_title("Radiated spectrum is the body spectrum tilted up by ω² (body modes dotted)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(OUT, "radiation_spectrum.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


if __name__ == "__main__":
    print("radiation diagnostics -> ./out/")
    for f in (fig_source_vs_radiated(), fig_inverse_distance(), fig_radiated_spectrum()):
        print("  wrote", os.path.relpath(f))
