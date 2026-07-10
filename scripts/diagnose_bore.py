"""Acoustic-bore diagnostic harness (the wind leg, batch 1).

Runs the clarinet (closed-open cylinder) and renders the diagnostics-as-visuals: the lossless
energy trace, the pressure pickup spectrum with the odd-harmonic oracle overlaid, the **odd-vs-even
harmonic bars** (the clarinet signature — even harmonics at the noise floor), the first few
pressure standing-wave shapes (antinode at the reed wall, node at the open bell), and a grid
convergence plot. Prints the real validation numbers so the batch can be judged from the console.

Run:  python scripts/diagnose_bore.py
Figures land in ./out/.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.sparse.linalg import eigsh  # noqa: E402

from physsynth.analysis import modal, spectrum  # noqa: E402
from physsynth.core.bore import C0_AIR, Bore  # noqa: E402
from physsynth.core.engine import simulate  # noqa: E402
from physsynth.viz import plots  # noqa: E402

L, RADIUS = 0.5, 0.008  # closed-open f1 = c0/(4L) = 171.5 Hz on ambient air
OUT = "out"


def build(N: int, lam: float, boundary=("closed", "open"), sigma: float = 0.0) -> Bore:
    fs = C0_AIR / (lam * (L / N))
    return Bore(L=L, fs=fs, N=N, radius=RADIUS, boundary=boundary, sigma=sigma)


def bump(bore: Bore, center_frac: float, width_frac: float = 0.05, amp: float = 1e-3):
    c, w = center_frac * bore.L, width_frac * bore.L
    return amp * np.exp(-((bore.x - c) ** 2) / (2.0 * w * w))


def pressure_modes(bore: Bore, n_modes: int):
    """The lowest ``n_modes`` pressure standing-wave shapes (free-node eigenvectors of L vs C)."""
    dof = bore.dof
    _, vecs = eigsh(bore.Lop[dof][:, dof], k=n_modes, M=bore.Cmat[dof][:, dof], sigma=0.0)
    shapes = []
    for j in range(n_modes):
        phi = np.zeros(bore.N + 1)
        phi[dof] = vecs[:, j]
        phi = phi / np.max(np.abs(phi))  # normalise for display
        shapes.append(phi)
    return shapes


def run_convergence(grids=(50, 100, 200, 400), lam: float = 0.7, n: int = 3):
    """Discrete-oracle error vs the continuum odd harmonics, refining h at fixed lambda (O(h²))."""
    oracle = modal.bore_resonance_frequencies(C0_AIR, L, n, "closed-open")
    hs, errs = [], []
    for N in grids:
        bore = build(N, lam)
        f = np.sort(_low_freqs(bore, n))
        hs.append(bore.h)
        errs.append(float(np.max(np.abs(f - oracle))))
    hs, errs = np.array(hs), np.array(errs)
    orders = np.log(errs[:-1] / errs[1:]) / np.log(hs[:-1] / hs[1:])
    return hs, errs, orders


def _low_freqs(bore: Bore, n_modes: int):
    dof = bore.dof
    w2 = eigsh(
        bore.Lop[dof][:, dof], k=n_modes, M=bore.Cmat[dof][:, dof], sigma=0.0,
        which="LM", return_eigenvectors=False,
    )
    return np.asarray(modal.discrete_bore_eigenfrequency(np.sort(w2), bore.k))


def odd_even_bars(bore: Bore, res, f1: float, n: int = 6):
    """Spectral magnitude at the first ``n`` odd and even harmonics of ``f1``."""
    freqs, mag, _ = spectrum.magnitude_spectrum(res.output, res.fs)
    df = freqs[1] - freqs[0]

    def peak(f):
        i = int(round(f / df))
        return float(mag[max(1, i - 2) : i + 3].max())

    odd_f = np.array([(2 * m - 1) * f1 for m in range(1, n + 1)])
    even_f = np.array([2 * m * f1 for m in range(1, n + 1)])
    return odd_f, np.array([peak(f) for f in odd_f]), even_f, np.array([peak(f) for f in even_f])


def main() -> None:
    os.makedirs(OUT, exist_ok=True)

    # --- main lossless clarinet run: energy + pickup spectrum --------------------------------
    bore = build(N=256, lam=1.0)
    bore.set_state(bump(bore, center_frac=0.1))  # bump near the closed (reed) wall
    res = simulate(bore, num_steps=int(0.6 * bore.fs), pickup_index=1)

    f1 = modal.bore_resonance_frequencies(C0_AIR, L, 1, "closed-open")[0]
    oracle = modal.bore_resonance_frequencies(C0_AIR, L, 6, "closed-open")
    detected = spectrum.measure_partials_near(res.output, res.fs, oracle)
    worst_cents = float(np.nanmax(np.abs(modal.cents(detected, oracle))))
    freqs, mag, _ = spectrum.magnitude_spectrum(res.output, res.fs)

    odd_f, odd_m, even_f, even_m = odd_even_bars(bore, res, f1)
    odd_even_ratio = float(np.min(odd_m) / np.max(even_m))

    # --- standing-wave pressure shapes --------------------------------------------------------
    shapes = pressure_modes(build(N=200, lam=1.0), 4)

    # --- convergence sweep --------------------------------------------------------------------
    hs, errs, orders = run_convergence()

    # --- open-open cross-check (full harmonic series) -----------------------------------------
    bore_oo = build(N=256, lam=1.0, boundary=("open", "open"))
    bore_oo.set_state(bump(bore_oo, center_frac=0.23))
    res_oo = simulate(bore_oo, num_steps=int(0.6 * bore_oo.fs), pickup_index=bore_oo.N // 3)
    oracle_oo = modal.bore_resonance_frequencies(C0_AIR, L, 6, "open-open")
    detected_oo = spectrum.measure_partials_near(res_oo.output, res_oo.fs, oracle_oo)
    worst_oo = float(np.nanmax(np.abs(modal.cents(detected_oo, oracle_oo))))

    # --- figures ------------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    plots.plot_energy(axes[0, 0], res.time, res.energy, drift=res.energy_drift)

    ax_bar = axes[0, 1]
    floor = max(np.max(even_m), 1e-30)
    ax_bar.bar(np.arange(1, 7) - 0.18, odd_m / floor, width=0.36, label="odd (2n-1)f₁", color="C0")
    ax_bar.bar(np.arange(1, 7) + 0.18, even_m / floor, width=0.36, label="even 2n·f₁", color="C3")
    ax_bar.set_yscale("log")
    ax_bar.set_xlabel("harmonic index n")
    ax_bar.set_ylabel("|spectrum| (÷ noise floor)")
    ax_bar.set_title(f"clarinet signature: odd-only  (min-odd/max-even ≈ {odd_even_ratio:,.0f}×)")
    ax_bar.legend()

    ax_modes = axes[1, 0]
    for j, phi in enumerate(shapes):
        ax_modes.plot(np.linspace(0, L, len(phi)), phi + 0.0, label=f"mode {j + 1}")
    ax_modes.axvline(0.0, color="k", lw=2, alpha=0.5)
    ax_modes.text(0.0, 1.05, "closed\n(antinode)", ha="left", va="bottom", fontsize=8)
    ax_modes.text(L, 1.05, "open\n(node)", ha="right", va="bottom", fontsize=8)
    ax_modes.set_xlabel("x (m)")
    ax_modes.set_ylabel("pressure (normalised)")
    ax_modes.set_title("standing-wave pressure shapes")
    ax_modes.legend(fontsize=8)

    plots.plot_convergence(axes[1, 1], hs, errs)
    fig.suptitle("Acoustic bore (clarinet) — wind-leg batch-1 validation", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    panel_path = os.path.join(OUT, "bore_diagnostics.png")
    fig.savefig(panel_path, dpi=130)
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    plots.plot_spectrum(ax2, freqs, mag, detected, f_max=oracle[-1] * 1.3)
    ax2.set_title("closed-open pickup spectrum (odd harmonics only)")
    fig2.tight_layout()
    spec_path = os.path.join(OUT, "bore_spectrum.png")
    fig2.savefig(spec_path, dpi=130)
    plt.close(fig2)

    # --- console report -----------------------------------------------------------------------
    print("=" * 70)
    print("Acoustic bore (clarinet, closed-open) - wind-leg batch-1 report")
    print("=" * 70)
    print(f"  c0 = {C0_AIR:.1f} m/s,  L = {L} m,  f1 = {f1:.2f} Hz,  fs = {res.fs:.0f} Hz")
    print(f"  [1] energy drift  max|E^n-E^0|/E^0 = {res.energy_drift:.3e}   (tol 1e-10)")
    print(f"  [2] worst partial vs odd oracle    = {worst_cents:.4f} cents  (tol ~1 cent)")
    print(f"  [*] clarinet signature min-odd/max-even = {odd_even_ratio:,.0f}x  (even = noise)")
    print(f"  [3] convergence orders (lam=0.7)   = {np.array2string(orders, precision=3)}   (~2)")
    print(f"  [3] errors (Hz)                    = {np.array2string(errs, precision=4)}")
    print(f"  [x] open-open full-series check    = {worst_oo:.4f} cents worst (even present)")
    print(f"      detected odd partials (Hz): {np.array2string(detected, precision=1)}")
    print(f"      oracle   odd partials (Hz): {np.array2string(oracle, precision=1)}")
    print("  figures:")
    print(f"      {panel_path}")
    print(f"      {spec_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
