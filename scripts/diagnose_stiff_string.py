"""Stiff-string (model #2) diagnostic harness — the inharmonicity money shot.

Runs the implicit theta-scheme stiff string and renders diagnostics-as-visuals (HANDOFF §7): energy
conservation (unconditionally stable — shown also at a Courant number lambda > 1 the explicit scheme
could not run), the *stretched* partials walking off the harmonic series, the phase-velocity
dispersion curve rising above c (bending stiffens high partials — the opposite of the ideal string's
droop), and the displacement. Prints the real validation numbers so the model can be judged from the
console alone.

Run:  python scripts/diagnose_stiff_string.py
Figures land in ./out/.
"""

from __future__ import annotations

import os
import sys

# Make `physsynth` importable when run directly (no editable install): repo root on sys.path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from physsynth.analysis import dispersion, modal, spectrum  # noqa: E402
from physsynth.core.engine import simulate  # noqa: E402
from physsynth.core.exciter import triangular_pluck  # noqa: E402
from physsynth.core.string_stiff import THETA_DEFAULT, StiffString  # noqa: E402
from physsynth.viz import plots  # noqa: E402

# Canonical string: c = 200 m/s, f0 = c/(2L) = 100 Hz. kappa = 2 -> B ~ 9.87e-4 (piano-ish).
L, T, RHO, KAPPA = 1.0, 200.0, 0.005, 2.0
C = float(np.sqrt(T / RHO))
THETA = THETA_DEFAULT
OUT = "out"


def build(N: int, lam: float, kappa: float = KAPPA, sigma: float = 0.0) -> StiffString:
    fs = C * N / (L * lam)
    return StiffString(L=L, T=T, rho=RHO, fs=fs, N=N, kappa=kappa, sigma=sigma, theta=THETA)


def measure_mode(N: int, lam: float, m: int, kappa: float = KAPPA, secs: float = 0.4) -> float:
    """Pure-tone frequency of a single mode via modal projection (node-free for every mode)."""
    s = build(N, lam, kappa)
    phi = modal.mode_shape(s.x, L, m)
    s.set_state(phi * 1e-3)
    res = simulate(s, num_steps=int(secs * s.fs), snapshot_stride=1)
    states = np.array([st for _, st in res.snapshots])
    q = states @ phi
    f_oracle = modal.discrete_stiff_mode_frequency(C, L, N, kappa, s.k, m, THETA)
    return float(spectrum.measure_partials_near(q, res.fs, np.array([f_oracle]))[0])


def run_dispersion(N: int = 256, lam: float = 0.8, modes=None):
    """Phase-velocity curve v_p(m)/c, measured vs the stiff oracle, for one Courant number."""
    if modes is None:
        modes = np.array([2, 4, 8, 16, 32, 48, 64, 96])
    measured = np.array([measure_mode(N, lam, int(m)) for m in modes])
    s = build(N, lam)
    oracle = dispersion.stiff_dispersion_frequencies(C, L, N, KAPPA, s.k, THETA, modes)
    vp_meas = dispersion.phase_velocity(measured, L, modes) / C
    vp_oracle = dispersion.phase_velocity(oracle, L, modes) / C
    return modes, [(lam, vp_meas, vp_oracle)]


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    f0 = C / (2.0 * L)
    B = modal.inharmonicity_B(C, L, KAPPA)
    n_partials = 10
    n = np.arange(1, n_partials + 1)

    # --- main lossless run: energy + pickup spectrum + stretched partials --------------------
    s = build(N=256, lam=1.0)
    s.set_state(triangular_pluck(s.x, L, 0.137 * L, amplitude=1e-3))
    res = simulate(s, num_steps=int(2.0 * s.fs), pickup_index=int(round(0.241 * s.N)))

    oracle = np.array(
        [modal.discrete_stiff_mode_frequency(C, L, s.N, KAPPA, s.k, m, THETA) for m in n]
    )
    continuum = modal.stiff_harmonic_frequencies(C, L, KAPPA, n_partials)
    detected = spectrum.measure_partials_near(res.output, res.fs, oracle)
    worst_oracle = float(np.nanmax(np.abs(modal.cents(detected, oracle))))
    worst_cont = float(np.nanmax(np.abs(modal.cents(detected, continuum))))
    # Fit B from the detected partials: (f_n/(n f0))^2 - 1 = B n^2.
    y = (detected / (n * f0)) ** 2 - 1.0
    B_fit = float(np.sum(n**2 * y) / np.sum(n**4))
    freqs, mag, _ = spectrum.magnitude_spectrum(res.output, res.fs)

    # --- unconditional-stability demo: same string at lambda = 2 (explicit scheme would blow up) -
    s_big = build(N=256, lam=2.0)
    s_big.set_state(triangular_pluck(s_big.x, L, 0.137 * L, amplitude=1e-3))
    res_big = simulate(s_big, num_steps=int(1.0 * s_big.fs))

    # --- displacement snapshots (a couple of fundamental periods) -----------------------------
    s_snap = build(N=200, lam=1.0)
    s_snap.set_state(triangular_pluck(s_snap.x, L, 0.137 * L, amplitude=1e-3))
    snap_res = simulate(s_snap, num_steps=600, snapshot_stride=12)

    # --- dispersion sweep (phase velocity rises above c) --------------------------------------
    disp_modes, disp_cases = run_dispersion()

    # --- figures ------------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    plots.plot_energy(axes[0, 0], res.time, res.energy, drift=res.energy_drift)
    plots.plot_stiff_partials(axes[0, 1], detected, continuum, f0, B)
    plots.plot_displacement_snapshots(
        axes[1, 0], s_snap.x, snap_res.snapshots[:: max(len(snap_res.snapshots) // 8, 1)],
        snap_res.fs,
    )
    plots.plot_dispersion(axes[1, 1], disp_modes, disp_cases)
    fig.suptitle("Stiff string (model #2) — implicit theta-scheme diagnostics", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    panel_path = os.path.join(OUT, "stiff_string_diagnostics.png")
    fig.savefig(panel_path, dpi=130)
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    plots.plot_spectrum(ax2, freqs, mag, detected, f_max=continuum[-1] * 1.15)
    fig2.tight_layout()
    spec_path = os.path.join(OUT, "stiff_string_spectrum.png")
    fig2.savefig(spec_path, dpi=130)
    plt.close(fig2)

    gif_path = os.path.join(OUT, "stiff_string.gif")
    gif_ok = plots.save_displacement_animation(gif_path, s_snap.x, snap_res.snapshots, snap_res.fs)

    # --- console report (the real numbers) ----------------------------------------------------
    print("=" * 74)
    print("Stiff string (model #2) -- validation report")
    print("=" * 74)
    print(f"  c={C:.1f} m/s  f0={f0:.3f} Hz  kappa={KAPPA:g}  B={B:.4e}  theta={THETA:g}")
    print(f"  fs = {res.fs:.0f} Hz,  N = {s.N},  lambda = {s.lam:.3f}")
    print(f"  [1] energy drift  max|E^n-E^0|/E^0     = {res.energy_drift:.3e}   (tol 1e-10)")
    print(f"  [1] energy drift at lambda = {s_big.lam:.1f} (>1!)   = {res_big.energy_drift:.3e}   "
          "(unconditional: explicit scheme could not run this)")
    print(f"  [3] partials vs discrete oracle (impl) = {worst_oracle:.4f} cents  (tol ~0.05)")
    print(f"  [3] partials vs continuum law (physics)= {worst_cont:.4f} cents  "
          "(numerical-dispersion gap; shrinks as O(h^2))")
    print(f"  [3] inharmonicity B: fit = {B_fit:.4e}  vs theory {B:.4e}  ({100*(B_fit-B)/B:+.2f}%)")
    f1_stretch = modal.cents(continuum[0], f0)
    print(f"      fundamental is itself stretched: f1 = {continuum[0]:.4f} Hz "
          f"({f1_stretch:+.3f} cents off f0)")
    top_m = int(disp_modes[-1])
    vp_meas, vp_oracle = disp_cases[0][1], disp_cases[0][2]
    print(f"  [5] dispersion v_p/c @ m={top_m} (lam={disp_cases[0][0]:g}) = {vp_meas[-1]:.4f} "
          f"(oracle {vp_oracle[-1]:.4f}) -- RISES above c (bending), unlike ideal-string droop")
    print("  figures:")
    print(f"      {panel_path}")
    print(f"      {spec_path}")
    print(f"      {gif_path}" if gif_ok else "      (GIF skipped: no pillow writer)")
    print("=" * 74)


if __name__ == "__main__":
    main()
