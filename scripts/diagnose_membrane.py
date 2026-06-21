"""Circular-membrane (model #4) diagnostic harness — the drumhead showpiece.

Runs the explicit 5-point FDTD membrane and renders diagnostics-as-visuals (HANDOFF §7): the
energy-conservation drift (identical for the staircased disk and a rectangle — the decoupling),
the low Bessel mode shapes as Chladni-style heatmaps, detected-vs-Bessel partials in cents (the
~O(h) staircase tier, stated honestly), passive energy decay under loss, and a struck-drum
displacement animation. Prints the real validation numbers so the model can be judged from the
console alone.

Run:  python scripts/diagnose_membrane.py
Figures land in ./out/.
"""

from __future__ import annotations

import os
import sys

# Make `physsynth` importable when run directly (no editable install): repo root on sys.path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.sparse.linalg import eigsh  # noqa: E402

from physsynth.analysis import modal, spectrum  # noqa: E402
from physsynth.core.engine import simulate  # noqa: E402
from physsynth.core.exciter import raised_cosine_2d  # noqa: E402
from physsynth.core.membrane import Membrane  # noqa: E402
from physsynth.core.operators2d import embed  # noqa: E402
from physsynth.viz import plots  # noqa: E402

# Canonical membrane: T = 200 N/m, rho = 0.005 kg/m^2 -> c = 200 m/s; disk radius a = 0.5 m
# -> fundamental f_01 = c * j_{0,1} / (2*pi*a) ~ 153.1 Hz.
T, RHO, A = 200.0, 0.005, 0.5
C = float(np.sqrt(T / RHO))
OUT = "out"


def build(domain, N, lam, *, sigma=0.0, **geom):
    h = (2.0 * A / N) if domain == "circle" else (geom["Lx"] / N)
    fs = C / (lam * h)
    return Membrane(domain=domain, T=T, rho=RHO, fs=fs, N=N, sigma=sigma, **geom)


def pluck(m, center, width, amp=1e-3):
    f = raised_cosine_2d(m.X, m.Y, center, width, amplitude=amp)
    f[~m.mask] = 0.0
    return f


def main() -> None:
    os.makedirs(OUT, exist_ok=True)

    # --- 1. Energy conservation: disk vs rectangle (the decoupling, as a printed number) ---
    print("Energy drift (lossless, ~1 s):")
    for domain, geom in [("rectangle", dict(Lx=1.0, Ly=1.0)), ("circle", dict(radius=A))]:
        m = build(domain, 48, 0.6, **geom)
        c0 = (0.4, 0.55) if domain == "rectangle" else (0.2 * A, -0.15 * A)
        w = 0.25 if domain == "rectangle" else 0.55 * A
        m.set_state(pluck(m, c0, w))
        res = simulate(m, num_steps=int(1.0 * m.fs))
        print(f"  {domain:9s}  drift = {res.energy_drift:.2e}   (nlive={m.n_live})")

    # --- 2. Low Bessel mode shapes (the Chladni showpiece) + detected-vs-Bessel partials ---
    m = build("circle", 128, 0.6, radius=A)
    K = 6
    Lam, V = eigsh(-m.L, k=K, sigma=0.0, which="LM")
    order = np.argsort(Lam)
    Lam, V = Lam[order], V[:, order]
    f_disc = modal.discrete_membrane_eigenfrequency(Lam, C, m.k)

    # Expand the Bessel oracle by degeneracy (m>=1 modes are cos/sin pairs) so it lines up with the
    # numerical spectrum, which contains BOTH copies of each degenerate mode. Without this the
    # sorted lists mis-pair past the first degenerate mode (the classic 2D degeneracy trap).
    exp_f, exp_lab = [], []
    for mm, nn, f, deg in modal.circular_membrane_freqs(C, A, K):
        exp_f.extend([f] * deg)
        exp_lab.extend([f"{mm},{nn}"] * deg)
    f_oracle = np.array(exp_f[:K])
    labels = exp_lab[:K]

    fig, axes = plt.subplots(2, 3, figsize=(11, 7))
    for j, ax in enumerate(axes.flat):
        field = embed(V[:, j], m.index_map)
        plots.plot_membrane_field(
            ax, m.X, m.Y, field, mask=m.mask, title=f"mode {labels[j]}  {f_disc[j]:.1f} Hz"
        )
    fig.suptitle("Circular-membrane low modes (numerical eigenvectors)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "membrane_modes.png"), dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    plots.plot_membrane_partials(ax, f_disc, f_oracle, labels=labels)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "membrane_partials.png"), dpi=110)
    plt.close(fig)

    print("\nDetected vs Bessel (disk, N=128, the ~O(h) staircase tier):")
    for lab, fd, fo in zip(labels, f_disc, f_oracle, strict=True):
        ec = 1200 * np.log2(fd / fo)
        print(f"  mode {lab:>4s}  disc={fd:8.3f}  bessel={fo:8.3f}  err={ec:+6.2f} c")

    # --- 3. Passivity under loss ---
    md = build("circle", 64, 0.6, sigma=8.0, radius=A)
    md.set_state(pluck(md, (0.2 * A, 0.0), 0.5 * A))
    resd = simulate(md, num_steps=int(0.8 * md.fs))
    fig, ax = plt.subplots(figsize=(7, 4))
    plots.plot_energy(ax, resd.time, resd.energy, drift=None)
    ax.set_yscale("log")
    ax.set_title(f"Passive decay (sigma=8): monotone = {np.all(np.diff(resd.energy) <= 0)}")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "membrane_passivity.png"), dpi=110)
    plt.close(fig)
    monotone = bool(np.all(np.diff(resd.energy) <= 0))
    print(f"\nPassivity (sigma=8): energy monotone non-increasing = {monotone}")

    # --- 4. Struck-drum animation ---
    ma = build("circle", 64, 0.6, radius=A)
    ma.set_state(pluck(ma, (0.25 * A, -0.1 * A), 0.45 * A))
    n_anim = int(0.06 * ma.fs)
    resa = simulate(ma, num_steps=n_anim, snapshot_stride=max(1, n_anim // 60))
    snaps = [(s, embed(v[ma.mask] if v.shape == ma.mask.shape else v, ma.index_map))
             for s, v in resa.snapshots]
    ok = plots.save_membrane_animation(
        os.path.join(OUT, "membrane_drum.gif"), ma.X, ma.Y, snaps, ma.fs, mask=ma.mask
    )
    print(f"\nAnimation written: {ok}  ->  out/membrane_drum.gif")

    # --- 5. FFT sanity: the time-stepper rings at the discrete fundamental ---
    mf = build("circle", 64, 0.6, radius=A)
    mf.set_state(pluck(mf, (0.25 * A, -0.1 * A), 0.5 * A))
    pk = mf.pickup_index_at(0.3 * A, 0.2 * A)
    resf = simulate(mf, num_steps=int(0.5 * mf.fs), pickup_index=pk)
    f_fund = modal.discrete_membrane_eigenfrequency(
        eigsh(-mf.L, k=1, sigma=0.0, which="LM", return_eigenvectors=False), C, mf.k
    )
    found = spectrum.measure_partials_near(resf.output, resf.fs, f_fund, search_hz=20.0)[0]
    print(f"FFT fundamental: found {found:.2f} Hz vs discrete {f_fund[0]:.2f} Hz "
          f"({1200*np.log2(found/f_fund[0]):+.2f} cents)")


if __name__ == "__main__":
    main()
