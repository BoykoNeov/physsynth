"""Diagnostics for the mallet -> membrane collision — the first contact/collision model (model #7).

A soft mallet (a lumped mass) strikes a square drumhead through a nonlinear felt. Four figures land
in ./out/:

  1. **The contact pulse.** Felt force and the penetration eta over the brief contact, with the
     mallet and head trajectories: the mallet dips in, the head yields, the felt force rises and
     falls in a single half-sine-ish pulse, then the mallet flies clear (it bounces).
  2. **Energy partition (conservation).** With a lossless head and an elastic felt the total energy
     H = E_membrane + mallet KE + felt PE is flat to machine precision while the three parts trade
     off — kinetic -> potential -> membrane during contact. The residual is drawn.
  3. **Hardness -> brightness.** Interior-pickup spectrum for a soft vs a stiff felt: the stiffer
     felt makes briefer contact and a brighter (higher-centroid) spectrum.
  4. **The hysteresis loop.** Felt force vs penetration for an elastic (lambda_h = 0, single-valued)
     and a Hunt-Crossley (lambda_h > 0) felt: the lossy felt traces a loop whose enclosed area is
     the energy the strike leaves behind — loading force > unloading force.

Run:  python scripts/diagnose_mallet.py
Figures land in ./out/.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from physsynth.analysis import spectrum  # noqa: E402
from physsynth.core.mallet import MalletMembrane  # noqa: E402
from physsynth.core.membrane import Membrane  # noqa: E402

T, RHO, N = 200.0, 0.005, 40  # c = 200 m/s square drumhead
OUT = "out"


def build(*, K=5.0e4, mass=0.02, alpha=2.3, hysteresis=0.0, v0=3.0, sigma=0.0,
          strike=(0.3, 0.4), lam=0.5):
    c = float(np.sqrt(T / RHO))
    h = 1.0 / N
    fs = c / (lam * h)
    mem = Membrane(domain="rectangle", T=T, rho=RHO, fs=fs, N=N, Lx=1.0, Ly=1.0, sigma=sigma)
    return MalletMembrane(
        membrane=mem, mass=mass, stiffness=K, alpha=alpha, hysteresis=hysteresis,
        strike_x=strike[0], strike_y=strike[1], strike_velocity=v0,
    )


def contact_pulse(steps=800):
    """Force, penetration and the mallet/head trajectories through the strike."""
    mal = build()
    k = mal.k
    t = np.arange(steps) * k
    f = np.empty(steps)
    eta = np.empty(steps)
    z_h = np.empty(steps)
    u_s = np.empty(steps)
    for i in range(steps):
        mal.step()
        f[i] = mal.contact_force
        eta[i] = mal.penetration
        z_h[i] = mal.z_H
        u_s[i] = mal.membrane.u[mal.node]
    return t * 1e3, f, eta, z_h, u_s


def energy_partition(steps=4000):
    """Membrane / mallet-KE / felt-PE partition of the (conserved) total energy."""
    from physsynth.core.mallet import contact_potential
    mal = build()
    k = mal.k
    t = np.arange(steps + 1) * k
    H = np.empty(steps + 1)
    e_mem = np.empty(steps + 1)
    ke = np.empty(steps + 1)
    pe = np.empty(steps + 1)

    def sample(i):
        mem = mal.membrane
        eta_n = mem.u[mal.node] - mal.z_H
        eta_nm1 = mem.u_prev[mal.node] - mal.z_H_prev
        e_mem[i] = mem.energy()
        ke[i] = 0.5 * mal.M * ((mal.z_H - mal.z_H_prev) / k) ** 2
        pe[i] = 0.5 * (float(contact_potential(eta_n, mal.K, mal.alpha))
                       + float(contact_potential(eta_nm1, mal.K, mal.alpha)))
        H[i] = mal.energy()

    sample(0)
    for i in range(1, steps + 1):
        mal.step()
        sample(i)
    return t * 1e3, H, e_mem, ke, pe


def hardness_spectrum(secs=0.25):
    """Interior-pickup spectrum for a soft vs a stiff felt."""
    out = {}
    for label, K in (("soft (K=1e4)", 1.0e4), ("hard (K=2e5)", 2.0e5)):
        mal = build(K=K)
        n = int(secs / mal.k)
        idx = mal.membrane.pickup_index_at(0.6, 0.55)
        sig = np.empty(n)
        for i in range(n):
            mal.step()
            sig[i] = mal.displacement_at(idx)
        freqs, mag, _ = spectrum.magnitude_spectrum(sig, 1.0 / mal.k)
        out[label] = (freqs, mag / mag.max())
    return out


def hysteresis_loop(steps=600):
    """Felt force vs penetration for an elastic vs a Hunt-Crossley (lossy) felt."""
    out = {}
    for label, lam_h in (("elastic ($\\lambda_h=0$)", 0.0), ("hysteretic ($\\lambda_h>0$)", 6.0e3)):
        mal = build(K=3.0e4, alpha=2.0, hysteresis=lam_h, v0=3.0)
        eta = np.empty(steps)
        f = np.empty(steps)
        for i in range(steps):
            mal.step()
            eta[i] = mal.penetration
            f[i] = mal.contact_force
        m = eta > 0.0
        out[label] = (eta[m] * 1e3, f[m])
    return out


def main() -> None:
    os.makedirs(OUT, exist_ok=True)

    tp, f, eta, z_h, u_s = contact_pulse()
    te, H, e_mem, ke, pe = energy_partition()
    drift = float(np.max(np.abs(H - H[0])) / abs(H[0]))
    spectra = hardness_spectrum()
    loops = hysteresis_loop()

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # 1. contact pulse + trajectories
    ax = axes[0, 0]
    ax.plot(tp, z_h * 1e3, color="C3", label="mallet $z_H$ (mm)")
    ax.plot(tp, u_s * 1e3, color="C0", label="head $u_S$ (mm)")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("displacement (mm)")
    # Zoom to a few multiples of the contact peak so the brief pulse is visible.
    end = min(int(np.argmax(eta) * 3) + 1, len(tp) - 1)
    ax.set_xlim(tp[0], tp[end])
    ax2 = ax.twinx()
    ax2.plot(tp, f, color="C1", lw=1.0, label="felt force (N)")
    ax2.set_ylabel("force (N)", color="C1")
    ax2.tick_params(axis="y", labelcolor="C1")
    ax.set_title("the contact pulse: mallet dips in, head yields, mallet bounces")
    ax.legend(fontsize=8, loc="lower right")

    # 2. energy partition
    ax = axes[0, 1]
    ax.plot(te, e_mem, color="C0", label="membrane")
    ax.plot(te, ke, color="C3", label="mallet KE")
    ax.plot(te, pe, color="C1", label="felt PE")
    ax.plot(te, H, "k--", lw=1.0, label="total $H$")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("energy (J)")
    ax.set_title(f"energy partition (conserved): drift {drift:.1e} rel.")
    ax.legend(fontsize=8)

    # 3. hardness -> brightness
    ax = axes[1, 0]
    for label, (freqs, mag) in spectra.items():
        sel = freqs <= 3000.0
        ax.semilogy(freqs[sel], mag[sel], lw=1.0, label=label)
    ax.set_ylim(1e-6, 2.0)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("|spectrum| (norm.)")
    ax.set_title("harder felt -> brighter spectrum")
    ax.legend(fontsize=8)

    # 4. hysteresis loop
    ax = axes[1, 1]
    for label, (eta_mm, f_l) in loops.items():
        ax.plot(eta_mm, f_l, lw=1.0, label=label)
    ax.set_xlabel("penetration $\\eta$ (mm)")
    ax.set_ylabel("felt force (N)")
    ax.set_title("hysteresis loop: lossy felt loses the enclosed area")
    ax.legend(fontsize=8)

    fig.suptitle("Mallet–membrane collision (model #7) diagnostics", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    path = os.path.join(OUT, "mallet_diagnostics.png")
    fig.savefig(path, dpi=110)
    print(f"wrote {path}")
    print(f"energy drift (conserved run): {drift:.2e} relative")


if __name__ == "__main__":
    main()
