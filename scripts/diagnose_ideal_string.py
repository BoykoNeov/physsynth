"""Milestone-1 diagnostic harness (HANDOFF §10, acceptance criterion 5).

Runs the ideal string and renders the diagnostics-as-visuals: energy-vs-time, detected-vs-analytic
partials, the pickup spectrum, the string displacement (static montage + GIF), and a grid
convergence plot. Also prints the real validation numbers (energy drift, worst partial error in
cents, convergence orders) so the milestone can be judged from the console alone.

Run:  python scripts/diagnose_ideal_string.py
Figures land in ./out/.
"""

from __future__ import annotations

import os
import sys

# Make `physsynth` importable when this script is run directly (python scripts/diagnose_...py)
# without an editable install: put the repository root on sys.path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from physsynth.analysis import modal, spectrum  # noqa: E402
from physsynth.core.engine import simulate  # noqa: E402
from physsynth.core.exciter import triangular_pluck  # noqa: E402
from physsynth.core.string_ideal import IdealString  # noqa: E402
from physsynth.viz import plots  # noqa: E402

# Canonical string: c = sqrt(T/rho) = 200 m/s, f1 = c/(2L) = 100 Hz.
L, T, RHO = 1.0, 200.0, 0.005
C = float(np.sqrt(T / RHO))
OUT = "out"


def build(N: int, lam: float, sigma: float = 0.0) -> IdealString:
    fs = C * N / (L * lam)
    return IdealString(L=L, T=T, rho=RHO, fs=fs, N=N, boundary="fixed", sigma=sigma)


def run_convergence(lam: float = 0.9, mode: int = 8, grids=(64, 128, 256)):
    f_cont = mode * C / (2 * L)
    errors, step_sizes = [], []
    for N in grids:
        s = build(N, lam)
        s.set_state(modal.mode_shape(s.x, L, mode) * 1e-3)
        res = simulate(s, num_steps=int(1.5 * s.fs), pickup_index=int(round(0.413 * N)))
        f_det = spectrum.measure_partials_near(res.output, res.fs, np.array([f_cont]))[0]
        errors.append(abs(f_det - f_cont))
        step_sizes.append(L / N)
    errors, step_sizes = np.array(errors), np.array(step_sizes)
    orders = np.log(errors[:-1] / errors[1:]) / np.log(step_sizes[:-1] / step_sizes[1:])
    return step_sizes, errors, orders


def main() -> None:
    os.makedirs(OUT, exist_ok=True)

    # --- main lossless run: energy + pickup signal -------------------------------------------
    s = build(N=100, lam=1.0)
    s.set_state(triangular_pluck(s.x, L, 0.137 * L, amplitude=1e-3))
    pickup = int(round(0.241 * s.N))
    res = simulate(s, num_steps=int(2.0 * s.fs), pickup_index=pickup)

    n_partials = 10
    analytic = modal.harmonic_frequencies(C, L, n_partials)
    detected = spectrum.measure_partials_near(res.output, res.fs, analytic)
    worst_cents = float(np.nanmax(np.abs(modal.cents(detected, analytic))))
    freqs, mag, _ = spectrum.magnitude_spectrum(res.output, res.fs)

    # --- short run for snapshots (a couple of fundamental periods) ----------------------------
    s_snap = build(N=100, lam=1.0)
    s_snap.set_state(triangular_pluck(s_snap.x, L, 0.137 * L, amplitude=1e-3))
    snap_res = simulate(s_snap, num_steps=550, snapshot_stride=11)
    snapshots = snap_res.snapshots

    # --- convergence sweep --------------------------------------------------------------------
    hs, errors, orders = run_convergence()

    # --- figures ------------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    plots.plot_energy(axes[0, 0], res.time, res.energy, drift=res.energy_drift)
    plots.plot_partials(axes[0, 1], analytic, detected)
    montage = snapshots[:: max(len(snapshots) // 8, 1)]
    plots.plot_displacement_snapshots(axes[1, 0], s_snap.x, montage, snap_res.fs)
    plots.plot_convergence(axes[1, 1], hs, errors)
    fig.suptitle("Ideal string — Milestone 1 validation diagnostics", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    panel_path = os.path.join(OUT, "ideal_string_diagnostics.png")
    fig.savefig(panel_path, dpi=130)
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    plots.plot_spectrum(ax2, freqs, mag, detected, f_max=analytic[-1] * 1.3)
    fig2.tight_layout()
    spec_path = os.path.join(OUT, "ideal_string_spectrum.png")
    fig2.savefig(spec_path, dpi=130)
    plt.close(fig2)

    gif_path = os.path.join(OUT, "ideal_string.gif")
    gif_ok = plots.save_displacement_animation(gif_path, s_snap.x, snapshots, snap_res.fs)

    # --- console report (the real numbers) ----------------------------------------------------
    print("=" * 70)
    print("Ideal string - Milestone 1 validation report")
    print("=" * 70)
    print(f"  c = {C:.1f} m/s,  f1 = {analytic[0]:.3f} Hz,  fs = {res.fs:.0f} Hz,  N = {s.N}")
    print(f"  [1] energy drift  max|E^n-E^0|/E^0 = {res.energy_drift:.3e}   (tol 1e-10)")
    print(f"  [2] worst partial error           = {worst_cents:.4f} cents   (tol ~1 cent)")
    print(f"  [3] convergence orders (lam=0.9)   = {np.array2string(orders, precision=3)}   (~2)")
    print(f"  [3] errors (Hz)                    = {np.array2string(errors, precision=4)}")
    print("  figures:")
    print(f"      {panel_path}")
    print(f"      {spec_path}")
    print(f"      {gif_path}" if gif_ok else "      (GIF skipped: no pillow writer)")
    print("=" * 70)


if __name__ == "__main__":
    main()
