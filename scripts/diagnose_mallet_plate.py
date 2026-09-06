"""Diagnostics for the mallet -> plate collision (model #7p) — the strike on an *implicit* body.

The mallet of model #7, unchanged, striking a Kirchhoff plate instead of a drumhead. What changes is
the one number the scalar contact solve needs from the resonator: the membrane is explicit, so its
driving-point admittance is the bare local nodal mass; the plate is implicit, so a nodal force
reaches every node next step and the admittance is one entry of a precomputed **influence column**
`(k^2 / force_den) A^-1 e_node`. Four figures land in ./out/:

  1. **The influence column.** The plate's response, one step after a unit force at the strike node
     — a compact peaked kernel, with its own centre value marked: that entry IS `g_s`. This is the
     figure that shows why a one-node correction (which is what the membrane gets away with) would
     be wrong here.
  2. **Energy partition (conservation).** Lossless supported plate, elastic felt: the total
     H = E_plate + mallet KE + felt PE is flat to machine precision while the three parts trade off.
  3. **The free plate recoils, and its energy read-out pays for it.** A struck *free* plate takes
     net momentum into its {1, x, y} rigid nullspace and translates for ever; the potential form
     annihilates that rigid part mathematically and only to eps numerically, so the energy read-out
     degrades **quadratically in the drift**. Both are drawn, against the supported control. The
     scheme conserves; the read-out is what decays.
  4. **The felt exponent is the whole of dynamic timbre.** A loud and a quiet strike, the quiet one
     scaled up by the velocity ratio. At alpha = 1 the two curves lie on top of each other to
     machine precision (the entire system is linear); above it they part company, and how far is set
     by alpha alone.

Run:  python scripts/diagnose_mallet_plate.py
Figures land in ./out/.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from physsynth.core.mallet import MalletPlate, contact_potential  # noqa: E402
from physsynth.core.plate import Plate  # noqa: E402

KAPPA, RHO, N, MU = 20.0, 0.005, 24, 1.0  # fs = kappa / (mu h^2) = 11520 Hz -> 23 steps of felt
OUT = "out"


def build(*, K=5.0e4, mass=0.02, alpha=2.3, hysteresis=0.0, v0=3.0, sigma=0.0,
          strike=(0.3, 0.4), boundary="supported"):
    h = 1.0 / N
    fs = KAPPA / (MU * h * h)
    plate = Plate(
        Lx=1.0, Ly=1.0, kappa=KAPPA, rho=RHO, fs=fs, N=N, sigma=sigma, boundary=boundary, nu=0.3,
    )
    return MalletPlate(
        plate=plate, mass=mass, stiffness=K, alpha=alpha, hysteresis=hysteresis,
        strike_x=strike[0], strike_y=strike[1], strike_velocity=v0,
    )


def influence_field():
    """The influence column as a 2-D field, plus `g_s` and the strike point."""
    mal = build()
    plate = mal.plate
    full = np.zeros(plate.mask.shape)
    full[plate.mask] = mal._influence
    return full, mal._g_s, (mal.x_strike, mal.y_strike), plate.Lx, plate.Ly


def energy_partition(steps=3000, boundary="supported"):
    """Plate / mallet-KE / felt-PE partition of the total energy."""
    mal = build(boundary=boundary)
    k = mal.k
    t = np.arange(steps + 1) * k
    H = np.empty(steps + 1)
    e_pl = np.empty(steps + 1)
    ke = np.empty(steps + 1)
    pe = np.empty(steps + 1)

    def sample(i):
        plate = mal.plate
        eta_n = plate.u[mal.node] - mal.z_H
        eta_nm1 = plate.u_prev[mal.node] - mal.z_H_prev
        e_pl[i] = plate.energy()
        ke[i] = 0.5 * mal.M * ((mal.z_H - mal.z_H_prev) / k) ** 2
        pe[i] = 0.5 * (float(contact_potential(eta_n, mal.K, mal.alpha))
                       + float(contact_potential(eta_nm1, mal.K, mal.alpha)))
        H[i] = mal.energy()

    sample(0)
    for i in range(1, steps + 1):
        mal.step()
        sample(i)
    return t * 1e3, H, e_pl, ke, pe


def recoil_and_readout(steps=6000):
    """The free plate's rigid drift, and the energy error that tracks its square."""
    out = {}
    for boundary in ("supported", "free"):
        mal = build(boundary=boundary)
        plate = mal.plate
        # The mass-weighted mean is the rigid coordinate; supported plates have no `w`.
        weight = (plate.w / np.sum(plate.w)) if boundary == "free" \
            else np.full(plate.n_live, 1.0 / plate.n_live)
        e0 = mal.energy()
        t = np.arange(steps) * mal.k
        drift = np.empty(steps)
        err = np.empty(steps)
        for i in range(steps):
            mal.step()
            drift[i] = float(np.dot(weight, plate.u))
            err[i] = abs(mal.energy() - e0) / abs(e0)
        out[boundary] = (t * 1e3, drift, err)
    return out


def dynamic_timbre(steps=1200):
    """Loud vs (scaled) quiet strike at several felt exponents."""
    out = {}
    for alpha in (1.0, 2.3):
        traces = []
        for v0 in (3.0, 0.75):
            mal = build(alpha=alpha, v0=v0)
            index = mal.plate.pickup_index_at(0.62, 0.71)
            sig = np.empty(steps)
            for i in range(steps):
                mal.step()
                sig[i] = mal.displacement_at(index)
            traces.append(sig)
            k = mal.k
        loud, quiet = traces[0], traces[1] * 4.0
        rel = float(np.max(np.abs(loud - quiet)) / np.max(np.abs(loud)))
        out[alpha] = (np.arange(steps) * k * 1e3, loud * 1e3, quiet * 1e3, rel)
    return out


def main() -> None:
    os.makedirs(OUT, exist_ok=True)

    column, g_s, strike, Lx, Ly = influence_field()
    te, H, e_pl, ke, pe = energy_partition()
    drift_rel = float(np.max(np.abs(H - H[0])) / abs(H[0]))
    recoil = recoil_and_readout()
    timbre = dynamic_timbre()

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # 1. the influence column
    ax = axes[0, 0]
    # NOT `column.T`. `Plate.X` varies along axis 1 and `Y` along axis 0, which is already
    # `imshow`'s own (row -> y, col -> x) convention with `origin="lower"`; transposing here draws a
    # plausible field with the peak reflected about the diagonal. The strike ring is in this figure
    # partly so that mistake cannot be made silently -- it has to land on the peak.
    im = ax.imshow(column, origin="lower", extent=(0, Lx, 0, Ly), cmap="magma")
    ax.plot(*strike, "o", mfc="none", mec="w", ms=10, mew=1.5)
    fig.colorbar(im, ax=ax, label="metres per newton, one step later")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"the influence column $A^{{-1}}e$: $g_s$ = {g_s:.3e} m/N at the ring")

    # 2. energy partition
    ax = axes[0, 1]
    ax.plot(te, e_pl, color="C0", label="plate")
    ax.plot(te, ke, color="C3", label="mallet KE")
    ax.plot(te, pe, color="C1", label="felt PE")
    ax.plot(te, H, "k--", lw=1.0, label="total $H$")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("energy (J)")
    ax.set_title(f"energy partition (conserved): drift {drift_rel:.1e} rel.")
    ax.legend(fontsize=8)

    # 3. the recoil, and the read-out that pays for it
    ax = axes[1, 0]
    t_f, drift_f, err_f = recoil["free"]
    t_s, _, err_s = recoil["supported"]
    ax.plot(t_f, np.abs(drift_f) * 1e3, color="C2", label="free: rigid drift (mm)")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("rigid drift (mm)", color="C2")
    ax.tick_params(axis="y", labelcolor="C2")
    ax2 = ax.twinx()
    ax2.semilogy(t_f, np.maximum(err_f, 1e-16), color="C3", lw=1.0, label="free: energy error")
    ax2.semilogy(t_s, np.maximum(err_s, 1e-16), color="C0", lw=1.0, label="supported: control")
    ax2.set_ylabel("relative energy error")
    ax2.legend(fontsize=8, loc="lower right")
    ax.set_title("a free plate recoils; its READ-OUT decays as the drift squared")

    # 4. dynamic timbre
    ax = axes[1, 1]
    for i, (alpha, (t, loud, quiet, rel)) in enumerate(sorted(timbre.items())):
        ax.plot(t, loud, color=f"C{2 * i}", lw=1.0,
                label=rf"$\alpha$={alpha}: loud (departure {rel:.1e})")
        ax.plot(t, quiet, color=f"C{2 * i + 1}", lw=0.8, ls="--",
                label=rf"$\alpha$={alpha}: 4$\times$quiet")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("pickup displacement (mm)")
    ax.set_title(r"a linear felt ($\alpha=1$) scales exactly; above it, timbre follows dynamics")
    ax.legend(fontsize=7)

    fig.suptitle("Mallet–plate collision (model #7p) diagnostics", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    path = os.path.join(OUT, "mallet_plate_diagnostics.png")
    fig.savefig(path, dpi=110)
    print(f"wrote {path}")
    print(f"driving-point admittance g_s = {g_s:.4e} m/N")
    print(f"energy drift (supported, conserved run): {drift_rel:.2e} relative")
    for alpha, (_, _, _, rel) in sorted(timbre.items()):
        print(f"alpha = {alpha}: departure from an exactly scaled response = {rel:.2e}")


if __name__ == "__main__":
    main()
