"""Diagnostics for the "air" node (batch 2): the radiation **load** / back-reaction
(:class:`physsynth.core.radiation.RadiatedBody`).

Batch 1 read the body's monopole out for free; here the air pushes back. The air presents a
resistance ``R`` (acoustic, Pa·s/m³) to the body's net volume velocity ``U = sum_i a_i q_i'``, so it
removes power ``P_rad = R U^2`` and hands it to the far field. That lost energy is tracked as an
explicit channel, ``radiated_energy = integral P_rad dt``, giving the passivity identity

    E_body^n + integral_0^n P_rad dt = const      (lossless body, any R >= 0).

The figures:

  1. **Energy channel.** A lossless body rings down under the load: ``E_body`` falls, the radiated
     channel rises by exactly the same amount, and their sum is flat to machine precision — the
     radiation analogue of the sigma-damping energy check. The residual ``|total - E_0|`` is drawn
     on a second axis to show it sitting at the roundoff floor.
  2. **Louder radiation = faster ring-down.** The body's modal energy for a sweep of ``R``: a bigger
     radiation resistance drains the body faster (and, past critical, over-damps it). ``R = 0`` is
     the flat, lossless reference.
  3. **The closed-form resistance oracle.** ``R_a(f)`` for the free-space monopole (``∝ f^2``, the
     batch-1-consistent choice) and the baffled circular piston (Rayleigh ``(ka)^2/2`` rise, then
     the ``J1`` roll-off), with the ``2x`` half-space/free-space offset marked. This is the modeling
     surface a constant-``R`` load samples at one frequency.

Run:  python scripts/diagnose_radiation_load.py
Figures land in ./out/.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from physsynth.core.body import ModalBody  # noqa: E402
from physsynth.core.radiation import (  # noqa: E402
    RadiatedBody,
    monopole_radiation_resistance,
    piston_radiation_resistance,
)

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out")
os.makedirs(OUT, exist_ok=True)

# A small guitar-top-ish modal body at audio rate. Lossless modes (sigmas = 0) so the ONLY energy
# sink is the radiation channel and the identity E_body + radiated_energy = const is exact.
FS = 48000.0
BODY_FREQS = np.array([110.0, 196.0, 261.0, 440.0])  # Hz
BODY_MASS = 0.02
Q0 = np.array([1e-3, -8e-4, 6e-4, 4e-4])  # a mixed-mode initial displacement


def make_loaded(R: float, sigmas: float = 0.0) -> RadiatedBody:
    body = ModalBody(freqs=BODY_FREQS, fs=FS, sigmas=sigmas, masses=BODY_MASS)
    loaded = RadiatedBody(body=body, R=R)
    loaded.set_state(Q0)
    return loaded


def fig_energy_channel():
    loaded = make_loaded(R=2000.0)
    steps = 6000
    e0 = loaded.energy()
    body_e = np.empty(steps)
    rad_e = np.empty(steps)
    total = np.empty(steps)
    for i in range(steps):
        loaded.step()
        body_e[i] = loaded.body.energy()
        rad_e[i] = loaded.radiated_energy
        total[i] = loaded.energy()
    t_ms = np.arange(steps) / FS * 1e3

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(t_ms, body_e / e0, lw=1.0, color="tab:blue", label="body energy  E_body")
    ax.plot(t_ms, rad_e / e0, lw=1.0, color="tab:orange", label="radiated channel  ∫P_rad dt")
    ax.plot(t_ms, total / e0, lw=1.2, color="tab:green", label="total (flat)")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("energy / E₀")
    ax.set_title("Radiation load: body energy bleeds into the far field, the sum is conserved")
    ax.legend(loc="center right", fontsize=8)

    ax2 = ax.twinx()
    ax2.semilogy(t_ms, np.abs(total - e0) / e0 + 1e-18, lw=0.6, color="tab:red", alpha=0.5)
    ax2.set_ylabel("|total − E₀| / E₀  (red)", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax2.set_ylim(1e-16, 1e-9)

    fig.tight_layout()
    path = os.path.join(OUT, "radiation_load_energy_channel.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def fig_ringdown_vs_R():
    steps = 6000
    t_ms = np.arange(steps) / FS * 1e3
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for R, color in [
        (0.0, "tab:gray"),
        (500.0, "tab:blue"),
        (2000.0, "tab:orange"),
        (8000.0, "tab:green"),
        (4.0e4, "tab:red"),
    ]:
        loaded = make_loaded(R=R)
        e0 = loaded.body.energy()
        e = np.empty(steps)
        for i in range(steps):
            loaded.step()
            e[i] = loaded.body.energy()
        ax.semilogy(t_ms, e / e0, lw=0.9, color=color, label=f"R = {R:g}")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("body energy / E₀")
    ax.set_title("Louder radiation drains the body faster (R = 0 is the lossless reference)")
    ax.set_ylim(1e-4, 2.0)
    ax.legend(fontsize=8, title="acoustic R (Pa·s/m³)")
    fig.tight_layout()
    path = os.path.join(OUT, "radiation_load_ringdown_vs_R.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def fig_resistance_oracle():
    f = np.linspace(20.0, 8000.0, 2000)
    omega = 2.0 * np.pi * f
    r_mono = np.array([monopole_radiation_resistance(w) for w in omega])
    # Two piston radii: a small source (stays near the Rayleigh limit) and a larger one (rolls off).
    r_pist_small = np.array([piston_radiation_resistance(w, 0.02) for w in omega])
    r_pist_big = np.array([piston_radiation_resistance(w, 0.10) for w in omega])
    # Rayleigh (ka -> 0) half-space asymptote = 2x the free-space monopole.
    r_rayleigh = 2.0 * r_mono

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.loglog(f, r_mono, lw=1.2, color="tab:blue",
              label="free-space monopole  ρ₀ω²/(4πc)  (∝ f²)")
    ax.loglog(f, r_rayleigh, lw=0.9, ls="--", color="tab:gray",
              label="half-space Rayleigh limit  (= 2× monopole)")
    ax.loglog(f, r_pist_small, lw=1.1, color="tab:orange", label="baffled piston  a = 20 mm")
    ax.loglog(f, r_pist_big, lw=1.1, color="tab:red", label="baffled piston  a = 100 mm")
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("acoustic radiation resistance  R_a  (Pa·s/m³)")
    ax.set_title("Closed-form load oracle: monopole ∝ f², piston rolls off past ka ≈ 1")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, which="both", alpha=0.2)
    fig.tight_layout()
    path = os.path.join(OUT, "radiation_load_resistance_oracle.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


if __name__ == "__main__":
    print("radiation-load diagnostics -> ./out/")
    for f in (fig_energy_channel(), fig_ringdown_vs_R(), fig_resistance_oracle()):
        print("  wrote", os.path.relpath(f))
