"""Damped-stiff-string (model #3) diagnostic harness — the frequency-dependent-loss money shot.

Runs the implicit theta-scheme damped stiff string and renders diagnostics-as-visuals (HANDOFF §7):
passive energy decay, the per-mode decay-rate curve (high partials dying faster — the cure for model
#2's backwards under-damping, with the sigma1=0 droop drawn alongside), a T60 -> (sigma0, sigma1)
round-trip, and the decaying displacement. Prints the real validation numbers so the model can be
judged from the console alone.

Run:  python scripts/diagnose_damped_string.py
Figures land in ./out/.
"""

from __future__ import annotations

import os
import sys

# Make `physsynth` importable when run directly (no editable install): repo root on sys.path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from physsynth.analysis import damping, modal  # noqa: E402
from physsynth.core.engine import simulate  # noqa: E402
from physsynth.core.exciter import triangular_pluck  # noqa: E402
from physsynth.core.string_damped import DampedStiffString  # noqa: E402
from physsynth.core.string_stiff import THETA_DEFAULT  # noqa: E402
from physsynth.viz import plots  # noqa: E402

# Canonical string: c = 200 m/s, f0 = c/(2L) = 100 Hz. kappa = 2 -> B ~ 9.87e-4 (piano-ish).
L, T, RHO, KAPPA = 1.0, 200.0, 0.005, 2.0
C = float(np.sqrt(T / RHO))
THETA = THETA_DEFAULT
OUT = "out"


def build(N, lam, *, sigma0=0.0, sigma1=0.0, kappa=KAPPA):
    fs = C * N / (L * lam)
    return DampedStiffString(
        L=L, T=T, rho=RHO, fs=fs, N=N, kappa=kappa, sigma0=sigma0, sigma1=sigma1, theta=THETA
    )


def measure_decay_factor(N, lam, m, *, sigma0, sigma1, steps, window=(0.1, 0.7)):
    """Per-step energy decay factor g_m of a single mode, via a log-linear fit (interior window)."""
    s = build(N, lam, sigma0=sigma0, sigma1=sigma1)
    s.set_state(modal.mode_shape(s.x, L, m) * 1e-3)
    res = simulate(s, num_steps=steps)
    e = res.energy
    i0, i1 = int(window[0] * steps), int(window[1] * steps)
    idx = np.arange(i0, i1 + 1)
    slope = np.polyfit(idx, np.log(e[idx]), 1)[0]
    return float(np.exp(slope))


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    N, sigma0, sigma1 = 128, 2.0, 1e-4
    k = 1.0 / build(N, 1.0).fs

    # --- passive energy decay of a broadband pluck (monotone, both losses) --------------------
    s = build(N, 1.0, sigma0=sigma0, sigma1=sigma1)
    s.set_state(triangular_pluck(s.x, L, 0.137 * L, amplitude=1e-3))
    res = simulate(s, num_steps=int(1.0 * s.fs), pickup_index=int(round(0.241 * N)))
    max_up = float(np.max(np.diff(res.energy)) / res.energy[0])

    # --- per-mode decay-rate curve: measured vs oracle (sigma1>0), and the sigma1=0 droop ------
    modes = np.arange(1, 17)
    measured_g = np.array(
        [measure_decay_factor(N, 1.0, int(m), sigma0=sigma0, sigma1=sigma1, steps=15000)
         for m in modes]
    )
    measured_rate = -np.log(measured_g) / k

    def oracle_rate(s0, s1):
        return np.array(
            [damping.discrete_damped_mode_rate(C, L, N, KAPPA, k, THETA, s0, s1, int(m))
             for m in modes]
        )

    oracle_with = oracle_rate(sigma0, sigma1)
    oracle_without = oracle_rate(sigma0, 0.0)  # same sigma0: the curves fork purely on sigma1
    worst_rate_rel = float(np.max(np.abs(measured_rate - oracle_with) / oracle_with))

    # --- T60 -> (sigma0, sigma1) round-trip (the deepened-physics payoff) ----------------------
    Nt = 256
    kt = 1.0 / build(Nt, 1.0).fs
    m1, m2 = 1, 20
    f1 = modal.discrete_stiff_mode_frequency(C, L, Nt, KAPPA, kt, m1, THETA)
    f2 = modal.discrete_stiff_mode_frequency(C, L, Nt, KAPPA, kt, m2, THETA)
    T60_1, T60_2 = 4.0, 0.8
    sig0_fit, sig1_fit = damping.loss_coefficients_from_T60(C, L, KAPPA, f1, T60_1, f2, T60_2)
    t60_meas = []
    for m in (m1, m2):
        g = measure_decay_factor(Nt, 1.0, m, sigma0=sig0_fit, sigma1=sig1_fit, steps=20000)
        t60_meas.append(2.0 * np.log(1e-3) / np.log(g) * kt)

    # --- decaying displacement snapshots ------------------------------------------------------
    s_snap = build(160, 1.0, sigma0=8.0, sigma1=4e-4)
    s_snap.set_state(triangular_pluck(s_snap.x, L, 0.137 * L, amplitude=1e-3))
    snap_res = simulate(s_snap, num_steps=900, snapshot_stride=18)

    # --- figures ------------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    plots.plot_energy(axes[0, 0], res.time, res.energy)
    axes[0, 0].set_title(
        f"Passive energy decay (broadband pluck; max step {max_up:+.1e} E0 <= 0)"
    )
    plots.plot_decay_rates(axes[0, 1], modes, measured_rate, oracle_with, oracle_without)
    plots.plot_displacement_snapshots(
        axes[1, 0], s_snap.x,
        snap_res.snapshots[:: max(len(snap_res.snapshots) // 8, 1)], snap_res.fs,
    )
    # T60 round-trip as a small text panel.
    axes[1, 1].axis("off")
    lines = [
        "T60 -> (sigma0, sigma1) round-trip",
        "",
        f"targets:  f1={f1:7.1f} Hz  T60={T60_1:.2f} s",
        f"          f2={f2:7.1f} Hz  T60={T60_2:.2f} s",
        "",
        f"inverted: sigma0={sig0_fit:.4f}   sigma1={sig1_fit:.3e}",
        "",
        f"measured: T60(f1)={t60_meas[0]:.3f} s  ({100*(t60_meas[0]-T60_1)/T60_1:+.2f}%)",
        f"          T60(f2)={t60_meas[1]:.3f} s  ({100*(t60_meas[1]-T60_2)/T60_2:+.2f}%)",
        "",
        "(continuum mapping; few-% gap is the",
        " discrete theta-factor, see the plan)",
    ]
    axes[1, 1].text(0.02, 0.98, "\n".join(lines), va="top", ha="left",
                    family="monospace", fontsize=10, transform=axes[1, 1].transAxes)
    fig.suptitle("Damped stiff string (model #3) — frequency-dependent loss", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    panel_path = os.path.join(OUT, "damped_string_diagnostics.png")
    fig.savefig(panel_path, dpi=130)
    plt.close(fig)

    gif_path = os.path.join(OUT, "damped_string.gif")
    gif_ok = plots.save_displacement_animation(gif_path, s_snap.x, snap_res.snapshots, snap_res.fs)

    # --- console report (the real numbers) ----------------------------------------------------
    print("=" * 74)
    print("Damped stiff string (model #3) -- validation report")
    print("=" * 74)
    print(f"  c={C:.1f} m/s  f0={C/(2*L):.1f} Hz  kappa={KAPPA:g}  theta={THETA:g}")
    print(f"  fs = {s.fs:.0f} Hz,  N = {N},  lambda = {s.lam:.3f}")
    print(f"  sigma0 = {sigma0:g}  sigma1 = {sigma1:g}")
    print(f"  [passivity] max energy step (broadband) = {max_up:+.2e} E0   (<= 0: monotone)")
    print(f"  [decay]     per-mode rate, measured vs discrete oracle worst = {worst_rate_rel:.2e} "
          "(rate-relative)")
    print(f"  [cure]      rate(m=1)={oracle_with[0]:.3f}/s  ->  rate(m=16)={oracle_with[-1]:.3f}/s "
          "(RISES: highs die faster)")
    print(f"              sigma1=0 instead: rate(m=1)={oracle_without[0]:.3f}/s -> "
          f"rate(m=16)={oracle_without[-1]:.3f}/s (FALLS: model #2's backwards artifact)")
    print(f"  [T60 map]   targets T60({f1:.0f}Hz)={T60_1}s, T60({f2:.0f}Hz)={T60_2}s -> "
          f"sigma0={sig0_fit:.3f}, sigma1={sig1_fit:.3e}")
    e1 = 100 * (t60_meas[0] - T60_1) / T60_1
    e2 = 100 * (t60_meas[1] - T60_2) / T60_2
    print(f"              measured T60: {t60_meas[0]:.3f}s ({e1:+.2f}%), "
          f"{t60_meas[1]:.3f}s ({e2:+.2f}%)  (continuum demo, few-% gap)")
    print("  figures:")
    print(f"      {panel_path}")
    print(f"      {gif_path}" if gif_ok else "      (GIF skipped: no pillow writer)")
    print("=" * 74)


if __name__ == "__main__":
    main()
