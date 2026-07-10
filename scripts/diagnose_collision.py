"""Diagnostics for the string <-> distributed-barrier collision — the first *distributed* contact
model (model #8): a string buzzing against a one-sided nonlinear barrier (fret / tanpura jawari).
Four figures land in ./out/:

  1. **String over the barrier.** Snapshots of the string shape at several instants, with the
     barrier profile drawn: the antinode swings down, slaps the rail, and springs off — the string
     conforms to the barrier where it is in contact (the jawari "grazing" contact).
  2. **Energy partition (conservation).** With a lossless string and an elastic barrier the total
     E = E_string + barrier PE is flat to machine precision while the two parts trade off — string
     energy dips into contact PE during each slap and returns. The residual is drawn.
  3. **The buzz spectrum.** Interior-pickup spectrum with the barrier out of reach (a clean plucked
     tone) vs in reach (the buzz): the barrier pumps energy into the high partials, raising the
     spectral centroid.
  4. **Contact space-time map.** Where and when the string touches the barrier over the run — the
     characteristic intermittent, travelling contact pattern of a buzzing string.

Run:  python scripts/diagnose_collision.py
Figures land in ./out/.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from physsynth.analysis import spectrum  # noqa: E402
from physsynth.core.collision import BarrierString, contact_potential  # noqa: E402
from physsynth.core.string_damped import DampedStiffString  # noqa: E402

L, T, RHO, N = 1.0, 100.0, 0.005, 120  # c = sqrt(T/rho) ~ 141 m/s, f1 = c/2L ~ 70.7 Hz
OUT = "out"


def build(*, K=1.0e6, alpha=1.5, barrier=-2.0e-3, hysteresis=0.0, sigma0=0.0, sigma1=0.0, lam=0.4):
    c = float(np.sqrt(T / RHO))
    h = L / N
    fs = c / (lam * h)
    string = DampedStiffString(
        L=L, T=T, rho=RHO, fs=fs, N=N, kappa=0.0, sigma0=sigma0, sigma1=sigma1
    )
    return BarrierString(
        string=string, barrier=barrier, stiffness=K, alpha=alpha, hysteresis=hysteresis
    )


def _pluck(bar, amplitude=5.0e-3):
    x = bar.string.x
    bar.set_state(amplitude * np.sin(np.pi * x / L))


def string_snapshots(steps=1200, n_snap=6):
    """String shapes at evenly spaced instants (mm), plus the barrier profile and grid x."""
    bar = build(sigma0=0.2)
    _pluck(bar)
    x = bar.string.x
    take = np.linspace(steps // 6, steps - 1, n_snap).astype(int)
    snaps = []
    for i in range(steps):
        bar.step()
        if i in take:
            snaps.append((i * bar.k * 1e3, bar.string.u.copy()))
    barrier_line = np.full_like(x, np.nan)
    barrier_line[bar._support] = bar._b
    return x, barrier_line, snaps


def energy_partition(steps=6000):
    """String / barrier-PE partition of the (conserved) total energy — lossless, elastic."""
    bar = build(sigma0=0.0)
    _pluck(bar)
    k = bar.k
    t = np.arange(steps + 1) * k
    E = np.empty(steps + 1)
    e_str = np.empty(steps + 1)
    pe = np.empty(steps + 1)

    def sample(i):
        s = bar.string
        eta_n = bar._b - s.u[bar._support]
        eta_nm1 = bar._b - s.u_prev[bar._support]
        e_str[i] = s.energy()
        pe[i] = 0.5 * s.h * float(
            np.sum(contact_potential(eta_n, bar.K, bar.alpha))
            + np.sum(contact_potential(eta_nm1, bar.K, bar.alpha))
        )
        E[i] = bar.energy()

    sample(0)
    for i in range(1, steps + 1):
        bar.step()
        sample(i)
    return t * 1e3, E, e_str, pe


def buzz_spectrum(secs=0.4):
    """Interior-pickup spectrum with the barrier out of reach vs in reach."""
    out = {}
    for label, barrier in (("clean (no barrier)", -100.0), ("buzz (barrier)", -2.0e-3)):
        bar = build(sigma0=0.5, barrier=barrier)
        _pluck(bar)
        n = int(secs / bar.k)
        node = max(1, N // 10)
        sig = np.empty(n)
        for i in range(n):
            bar.step()
            sig[i] = bar.string.displacement_at(node)
        freqs, mag, _ = spectrum.magnitude_spectrum(sig, 1.0 / bar.k)
        out[label] = (freqs, mag / mag.max())
    return out


def contact_map(steps=2500, stride=2):
    """Boolean in-contact map over (support node x, time) — the buzz pattern."""
    bar = build(sigma0=0.15)
    _pluck(bar)
    rows = []
    times = []
    for i in range(steps):
        bar.step()
        if i % stride == 0:
            rows.append(bar.penetration > 0.0)
            times.append(i * bar.k * 1e3)
    xnodes = bar.string.x[bar._support]
    return xnodes, np.array(times), np.array(rows)


def main() -> None:
    os.makedirs(OUT, exist_ok=True)

    x, barrier_line, snaps = string_snapshots()
    te, E, e_str, pe = energy_partition()
    drift = float(np.max(np.abs(E - E[0])) / abs(E[0]))
    spectra = buzz_spectrum()
    xnodes, times, cmap = contact_map()

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # 1. string over the barrier
    ax = axes[0, 0]
    for tms, u in snaps:
        ax.plot(x, u * 1e3, lw=1.0, alpha=0.8, label=f"t={tms:.0f} ms")
    ax.plot(x, barrier_line * 1e3, "k--", lw=1.5, label="barrier")
    ax.set_xlabel("position x (m)")
    ax.set_ylabel("displacement (mm)")
    ax.set_title("string slaps the barrier and springs off")
    ax.legend(fontsize=7, ncol=2)

    # 2. energy partition
    ax = axes[0, 1]
    ax.plot(te, e_str, color="C0", label="string")
    ax.plot(te, pe, color="C1", label="barrier PE")
    ax.plot(te, E, "k--", lw=1.0, label="total E")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("energy (J)")
    ax.set_title(f"energy partition (conserved): drift {drift:.1e} rel.")
    ax.legend(fontsize=8)

    # 3. buzz spectrum
    ax = axes[1, 0]
    for label, (freqs, mag) in spectra.items():
        sel = freqs <= 4000.0
        ax.semilogy(freqs[sel], mag[sel], lw=1.0, label=label)
    ax.set_ylim(1e-6, 2.0)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("|spectrum| (norm.)")
    ax.set_title("barrier -> buzz (high partials pumped up)")
    ax.legend(fontsize=8)

    # 4. contact space-time map
    ax = axes[1, 1]
    ax.imshow(
        cmap, aspect="auto", origin="lower", cmap="magma",
        extent=(float(xnodes[0]), float(xnodes[-1]), float(times[0]), float(times[-1])),
        interpolation="nearest",
    )
    ax.set_xlabel("position x (m)")
    ax.set_ylabel("time (ms)")
    ax.set_title("contact map: where/when the string touches (bright = contact)")

    fig.suptitle("String–barrier collision (model #8) diagnostics", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    path = os.path.join(OUT, "collision_diagnostics.png")
    fig.savefig(path, dpi=110)
    print(f"wrote {path}")
    print(f"energy drift (conserved run): {drift:.2e} relative")


if __name__ == "__main__":
    main()
