"""Diagnostics for the bowed string — the first continuous nonlinear exciter.

The bow is an *active* element, so the diagnostics tell a different story from the conservative
resonators/bridges. Four figures land in ./out/:

  1. **Energy balance, not conservation.** Stored string energy E(t), the accumulated bow work
     W(t) = sum k*f_B*(I d_t. u), and their difference (the loss). Lossless: E == W to machine
     precision — the bow's power accounting is exact. Lossy: E rides below W by the dissipation and
     settles into a steady Helmholtz limit cycle (bow input == loss).
  2. **Helmholtz motion.** The travelling corner (string snapshots over one period), the two-state
     stick-slip velocity at the bow point (sticks at v_bow, one slip per period), and the sawtooth
     pickup waveform — the textbook signatures.
  3. **The bowed spectrum.** A full harmonic series at f_n = n c/2L (the sawtooth), with the
     fundamental and its harmonics marked. Pitch is set by the string, not the bow.
  4. **Schelleng diagram.** Sweep bow force x bow position; classify each note as crushed (too much
     force -> no fundamental), clean Helmholtz (one slip/period at f1), or raucous/subharmonic
     (too much force for that beta). Reproduces the classic min/max-bow-force playability wedge.

Run:  python scripts/diagnose_bow.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from physsynth.core.bow import BowedString  # noqa: E402
from physsynth.core.string_damped import DampedStiffString  # noqa: E402

# Canonical flexible (kappa=0) string: c = 200 m/s, f1 = c/2L = 100 Hz. lam < 1 for coupling
# headroom; a little frequency-dependent loss (sigma1) rounds the Helmholtz corner into clean tone.
L, T, RHO, C = 1.0, 200.0, 0.005, 200.0
N, LAM = 200, 0.9
FS = C * N / (L * LAM)
F1 = C / (2.0 * L)
OUT = "out"


def build(*, force, v_bow, beta=0.13, sigma0=0.5, sigma1=0.05, sharpness=60.0):
    s = DampedStiffString(L=L, T=T, rho=RHO, fs=FS, N=N, kappa=0.0, sigma0=sigma0, sigma1=sigma1)
    return BowedString(
        string=s, bow_position=beta * L, v_bow=v_bow, force=force, sharpness=sharpness
    )


def run(bow, secs, pickup_frac=0.33):
    steps = int(secs * FS)
    pk = int(pickup_frac * bow.string.N)
    e = np.empty(steps + 1)
    w = np.empty(steps + 1)
    disp = np.empty(steps + 1)
    vbow = np.empty(steps + 1)
    e[0], w[0], disp[0], vbow[0] = bow.energy(), bow.bow_work, bow.displacement_at(pk), 0.0
    snaps = []
    snap_at = set(int(steps * f) for f in np.linspace(0.80, 0.82, 9))  # ~1 period near the end
    for i in range(1, steps + 1):
        bow.step()
        e[i], w[i] = bow.energy(), bow.bow_work
        disp[i], vbow[i] = bow.displacement_at(pk), bow.bow_velocity()
        if i in snap_at:
            snaps.append((i, bow.state.copy()))
    return dict(e=e, w=w, disp=disp, vbow=vbow, snaps=snaps, steps=steps)


# -- 1. Energy balance ----------------------------------------------------------------------
def fig_energy_balance():
    lossless = build(force=1.0, v_bow=0.1, sigma0=0.0, sigma1=0.0)
    lossy = build(force=1.0, v_bow=0.1)
    r0 = run(lossless, 0.4)
    r1 = run(lossy, 1.2)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 4.2))
    t0 = np.arange(r0["steps"] + 1) / FS * 1e3
    ax0.plot(t0, r0["e"] * 1e6, label="stored E", lw=1.6)
    ax0.plot(t0, r0["w"] * 1e6, "--", label="bow work W", lw=1.2)
    resid = np.max(np.abs(r0["e"] - r0["e"][0] - r0["w"]) / (np.abs(r0["w"]) + 1e-30))
    ax0.set(title=f"Lossless: E == bow work (residual {resid:.1e})",
            xlabel="time (ms)", ylabel="energy (uJ)")
    ax0.legend(frameon=False)

    t1 = np.arange(r1["steps"] + 1) / FS * 1e3
    ax1.plot(t1, r1["w"] * 1e6, "--", label="bow work W (in)", lw=1.2)
    ax1.plot(t1, (r1["w"] - (r1["e"] - r1["e"][0])) * 1e6, ":", label="dissipated", lw=1.2)
    ax1.plot(t1, r1["e"] * 1e6, label="stored E", lw=1.6)
    ax1.set(title="Lossy: E settles to a Helmholtz limit cycle (in == loss)",
            xlabel="time (ms)", ylabel="energy (uJ)")
    ax1.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(f"{OUT}/bow_1_energy_balance.png", dpi=130)
    plt.close(fig)
    print(f"  [1] energy balance: lossless residual {resid:.2e}")


# -- 2. Helmholtz motion --------------------------------------------------------------------
def fig_helmholtz():
    bow = build(force=0.4, v_bow=0.1, beta=0.13)
    r = run(bow, 2.0)
    steps = r["steps"]
    period = int(FS / F1)
    tail = slice(steps - 3 * period, steps)
    t_ms = np.arange(steps - 3 * period, steps) / FS * 1e3

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
    for _step, u in r["snaps"]:
        axs[0].plot(bow.string.x, u * 1e3, lw=1.0, alpha=0.7)
    axs[0].axvline(bow.x_bow, color="k", ls=":", lw=1, label=f"bow (beta={bow.beta:.2f})")
    axs[0].set(title="Travelling Helmholtz corner", xlabel="x (m)", ylabel="u (mm)")
    axs[0].legend(frameon=False)

    axs[1].plot(t_ms, r["vbow"][tail], lw=1.0)
    axs[1].axhline(bow.v_bow, color="g", ls="--", lw=1, label="v_bow (stick)")
    axs[1].set(title="Bow-point velocity: stick-slip", xlabel="time (ms)", ylabel="v (m/s)")
    axs[1].legend(frameon=False)

    axs[2].plot(t_ms, r["disp"][tail] * 1e3, lw=1.0)
    axs[2].set(title="Pickup displacement (sawtooth)", xlabel="time (ms)", ylabel="u (mm)")
    fig.tight_layout()
    fig.savefig(f"{OUT}/bow_2_helmholtz.png", dpi=130)
    plt.close(fig)
    print("  [2] Helmholtz waveform saved")


# -- 3. Bowed spectrum ----------------------------------------------------------------------
def fig_spectrum():
    bow = build(force=0.4, v_bow=0.1)
    r = run(bow, 2.5)
    steps = r["steps"]
    tail = r["disp"][int(0.5 * steps):]
    tail = (tail - tail.mean()) * np.hanning(len(tail))
    spec = 20.0 * np.log10(np.abs(np.fft.rfft(tail)) + 1e-12)
    freqs = np.fft.rfftfreq(len(tail), 1.0 / FS)

    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.plot(freqs, spec - spec.max(), lw=0.8)
    for n in range(1, 16):
        ax.axvline(n * F1, color="r", ls=":", lw=0.6, alpha=0.5)
    ax.set(title="Bowed tone: full harmonic series at n*c/2L (sawtooth)",
           xlabel="frequency (Hz)", ylabel="magnitude (dB)", xlim=(0, 16 * F1), ylim=(-80, 2))
    fig.tight_layout()
    fig.savefig(f"{OUT}/bow_3_spectrum.png", dpi=130)
    plt.close(fig)
    print("  [3] spectrum saved")


# -- 4. Schelleng diagram -------------------------------------------------------------------
def _classify(bow, secs=1.6):
    """Classify a note: 0 crushed (silent), 1 clean Helmholtz (1 slip/period), 2 raucous."""
    steps = int(secs * FS)
    pk = bow.string.N // 3
    sig = np.empty(steps)
    vrel = np.empty(steps)
    for n in range(steps):
        bow.step()
        sig[n] = bow.displacement_at(pk)
        vrel[n] = bow.v_rel
    tail_v = vrel[int(0.6 * steps):]
    amp = np.max(np.abs(sig[int(0.7 * steps):] - np.mean(sig[int(0.7 * steps):])))
    if amp < 1e-6:
        return 0  # crushed / no oscillation
    slipping = np.abs(tail_v) >= 0.5 * bow.v_bow
    onsets = int(np.sum((~slipping[:-1]) & (slipping[1:])))
    spp = onsets / (len(tail_v) * F1 / FS)
    return 1 if 0.85 < spp < 1.2 else 2


def fig_schelleng():
    betas = np.array([0.06, 0.09, 0.13, 0.18, 0.25, 0.33])
    forces = np.array([0.1, 0.2, 0.4, 0.7, 1.0, 1.5, 2.2, 3.2])
    grid = np.zeros((len(forces), len(betas)))
    for j, beta in enumerate(betas):
        for i, force in enumerate(forces):
            grid[i, j] = _classify(build(force=force, v_bow=0.1, beta=beta))
    fig, ax = plt.subplots(figsize=(8, 5.5))
    im = ax.pcolormesh(betas, forces, grid, cmap="RdYlGn", shading="nearest", vmin=0, vmax=2)
    ax.set_yscale("log")
    ax.set(title="Schelleng diagram (v_bow=0.1): playability wedge",
           xlabel="bow position  beta = x_bow / L", ylabel="bow force (N)")
    cb = fig.colorbar(im, ax=ax, ticks=[0, 1, 2])
    cb.ax.set_yticklabels(["crushed", "Helmholtz", "raucous"])
    fig.tight_layout()
    fig.savefig(f"{OUT}/bow_4_schelleng.png", dpi=130)
    plt.close(fig)
    clean = int(np.sum(grid == 1))
    print(f"  [4] Schelleng diagram saved ({clean}/{grid.size} clean-Helmholtz cells)")


def main():
    os.makedirs(OUT, exist_ok=True)
    print("Bowed-string diagnostics -> ./out/")
    fig_energy_balance()
    fig_helmholtz()
    fig_spectrum()
    fig_schelleng()
    print("Done.")


if __name__ == "__main__":
    main()
