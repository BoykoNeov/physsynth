"""Acoustic-bore diagnostic harness — the radiating bell (wind leg, batch 2).

Batch 1's open end reflected perfectly and radiated nothing. Here the bell presents a radiation
resistance ``R_bell`` (acoustic, Pa·s/m³): a rank-1, unconditionally-passive dashpot on the
terminating half-cell node — the :class:`~physsynth.core.radiation.RadiatedBody` pattern moved onto
a boundary. This is what makes the tube passively lossy and how sound leaves. The figures:

  1. **Passivity energy channel.** A lossless clarinet with a radiating bell rings down: the air-
     column energy ``E_bore`` falls, the radiated channel rises by exactly as much, and the total
     ``E_bore + radiated_energy`` is flat to machine precision — the acoustic dual of the ``sigma``
     energy check and of the radiation load.
  2. **Reflection oracle (the teeth).** The fraction of a pulse's energy shed by one reflection vs
     ``1 - r^2 = 4 R Z0 / (R + Z0)^2`` (``r = (R - Z0)/(R + Z0)``): the measurement lands on the
     closed-form curve across four decades of ``R/Z0``, peaking at the matched (anechoic) load
     ``R = Z0``. An oracle *independent* of the self-consistent energy identity.
  3. **R interpolates the two ideal ends.** The interior pickup spectrum for an open-ish bell
     (``R << Z0``: the odd-harmonic clarinet, lightly damped) and a rigid bell (``R -> inf``:
     closed-closed, the full series with even harmonics back).
  4. **The far field is clean.** The overdamped end node carries a cosmetic Nyquist ripple that
     cancels in ``U_out``; the far-field read-out ``pressure() = dU_out/dt`` is smooth. Both traces
     are drawn to show the artifact is invisible in the validated quantity.

Run:  python scripts/diagnose_bore_bell.py
Figures land in ./out/.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from physsynth.analysis import modal, spectrum  # noqa: E402
from physsynth.core.bore import C0_AIR, Bore  # noqa: E402
from physsynth.core.engine import simulate  # noqa: E402
from physsynth.core.radiation import piston_radiation_resistance  # noqa: E402

L, RADIUS = 0.5, 0.008  # closed-open f1 = c0/(4L) = 171.5 Hz on ambient air
OUT = "out"


def build(N, lam, boundary=("closed", "radiating"), R_bell=650.0, sigma=0.0):
    fs = C0_AIR / (lam * (L / N))
    return Bore(L=L, fs=fs, N=N, radius=RADIUS, boundary=boundary, R_bell=R_bell, sigma=sigma)


def bump(bore, center_frac, width_frac=0.05, amp=1e-3):
    c, w = center_frac * bore.L, width_frac * bore.L
    return amp * np.exp(-((bore.x - c) ** 2) / (2.0 * w * w))


def energy_channel(N=256, R_bell=5e4, secs=0.25):
    """Air-column, radiated, and total energy traces for a ringing radiating clarinet."""
    bore = build(N, lam=0.9, R_bell=R_bell)
    bore.set_state(bump(bore, center_frac=0.1))
    field, radiated, total, t = [], [], [], []
    for step in range(int(secs * bore.fs)):
        bore.step()
        field.append(bore.acoustic_energy())
        radiated.append(bore.radiated_energy)
        total.append(bore.energy())
        t.append(step * bore.k)
    return (np.array(t), np.array(field), np.array(radiated), np.array(total))


def reflection_sweep(N=400, fracs=(0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0)):
    """Measured single-reflection energy-shed fraction vs the 1 - r^2 oracle, across R/Z0."""
    Z0 = build(N, 1.0).Z0
    measured, oracle = [], []
    for frac in fracs:
        R = frac * Z0
        bore = build(N, lam=1.0, R_bell=R)
        c, w = 0.5 * bore.L, 0.03 * bore.L
        bore.set_state(1e-3 * np.exp(-((bore.x - c) ** 2) / (2.0 * w * w)))
        e0 = bore.energy()
        simulate(bore, num_steps=N, pickup_index=1)  # one clean right-bounce, no second yet
        r = (R - Z0) / (R + Z0)
        measured.append(bore.radiated_energy / e0)
        oracle.append(0.5 * (1.0 - r * r))
    return np.array(fracs), np.array(measured), np.array(oracle)


def interior_spectrum(R_bell, center_frac, N=256, secs=0.5):
    bore = build(N, lam=1.0, R_bell=R_bell)
    bore.set_state(bump(bore, center_frac=center_frac))
    res = simulate(bore, num_steps=int(secs * bore.fs), pickup_index=1)  # INTERIOR pickup
    freqs, mag, _ = spectrum.magnitude_spectrum(res.output, res.fs)
    return freqs, mag


def far_field_traces(R_bell=650.0, N=256, secs=0.06):
    bore = build(N, lam=1.0, R_bell=R_bell)
    bore.set_state(bump(bore, center_frac=0.1))
    p_far, p_raw, t = [], [], []
    for step in range(int(secs * bore.fs)):
        bore.step()
        p_far.append(bore.pressure())
        p_raw.append(bore.p[-1])
        t.append(step * bore.k)
    return np.array(t), np.array(p_far), np.array(p_raw)


def main() -> None:
    os.makedirs(OUT, exist_ok=True)

    f1_co = modal.bore_resonance_frequencies(C0_AIR, L, 1, "closed-open")[0]
    Z0 = build(256, 1.0).Z0
    R_phys = piston_radiation_resistance(2.0 * np.pi * f1_co, RADIUS)

    t, field, radiated, total = energy_channel()
    total_drift = float(np.max(np.abs(total - total[0])) / total[0])

    fracs, measured, oracle = reflection_sweep()
    refl_err = float(np.max(np.abs(measured - oracle) / oracle))

    freqs_open, mag_open = interior_spectrum(R_bell=R_phys, center_frac=0.1)
    freqs_rigid, mag_rigid = interior_spectrum(R_bell=1e10, center_frac=0.12)

    tf, p_far, p_raw = far_field_traces()

    def nyq_frac(x):
        m = np.abs(np.fft.rfft(x - x.mean()))
        return float((m[int(0.95 * len(m)) :] ** 2).sum() / (m**2).sum())

    nf_far, nf_raw = nyq_frac(p_far), nyq_frac(p_raw)

    # --- figure -------------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0, 0]
    ax.plot(t, field, label="air column $E_{bore}$", color="C0")
    ax.plot(t, radiated, label="radiated (far field)", color="C3")
    ax.plot(t, total, label="total (conserved)", color="k", lw=1.2)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("energy (J)")
    ax.set_title(f"passivity: $E_{{bore}}$ + radiated = const  (drift {total_drift:.1e})")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    order = np.argsort(fracs)
    ax.plot(fracs[order], oracle[order], "-", color="C7", label=r"oracle $\frac{1}{2}(1-r^2)$")
    ax.plot(fracs[order], measured[order], "o", color="C0", label="measured shed / $E_0$")
    ax.axvline(1.0, color="C3", ls=":", label="matched $R=Z_0$ (anechoic)")
    ax.set_xscale("log")
    ax.set_xlabel(r"$R / Z_0$")
    ax.set_ylabel("energy shed by one reflection / $E_0$")
    ax.set_title(f"reflection oracle (worst {refl_err:.1e} rel.)")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    fmax = 12.0 * f1_co
    for freqs, mag, lab, col in (
        (freqs_open, mag_open, f"open-ish bell  $R/Z_0$={R_phys / Z0:.1e}", "C0"),
        (freqs_rigid, mag_rigid, r"rigid bell  $R\to\infty$", "C3"),
    ):
        sel = freqs <= fmax
        ax.semilogy(freqs[sel], mag[sel] / mag[sel].max(), color=col, lw=1.0, label=lab)
    for n in range(1, 7):
        ax.axvline((2 * n - 1) * f1_co, color="C0", ls=":", alpha=0.25)
        ax.axvline(2 * n * f1_co, color="C3", ls=":", alpha=0.25)
    ax.set_ylim(1e-8, 2.0)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("|spectrum| (norm.)")
    ax.set_title("R interpolates: odd-only clarinet ↔ closed-closed full series")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.plot(tf * 1e3, p_raw / np.max(np.abs(p_raw)), color="C7", lw=0.8,
            label=f"raw end node (Nyquist {nf_raw:.1e})")
    ax.plot(tf * 1e3, p_far / np.max(np.abs(p_far)), color="C0", lw=1.0,
            label=f"far field $dU_{{out}}/dt$ (Nyquist {nf_far:.1e})")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("normalised")
    ax.set_title("far-field read-out is clean (end-node ripple cancels in $U_{out}$)")
    ax.legend(fontsize=8)

    fig.suptitle("Acoustic bore — radiating bell (wind-leg batch-2 validation)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    panel_path = os.path.join(OUT, "bore_bell_diagnostics.png")
    fig.savefig(panel_path, dpi=130)
    plt.close(fig)

    # --- console report -----------------------------------------------------------------------
    print("=" * 70)
    print("Acoustic bore — radiating bell (wind-leg batch-2 report)")
    print("=" * 70)
    print(f"  c0 = {C0_AIR:.1f} m/s,  L = {L} m,  f1(closed-open) = {f1_co:.2f} Hz")
    print(f"  Z0 = {Z0:.3e} Pa·s/m^3   physical bell R = {R_phys:.1f}  (R/Z0 = {R_phys / Z0:.2e})")
    print(f"  [1] passivity  max|total - E0|/E0     = {total_drift:.3e}   (tol 1e-10)")
    print(f"      air-column energy shed by bell    = {(1 - field[-1] / field[0]) * 100:.1f}%")
    print(f"  [2] reflection oracle worst rel. err  = {refl_err:.3e}   (1 - r^2 across R/Z0)")
    print("      R/Z0, measured shed/E0, oracle:")
    for fr, me, orc in zip(fracs, measured, oracle, strict=False):
        print(f"        {fr:6.2f}   {me:.4f}   {orc:.4f}")
    print(f"  [3] far-field Nyquist frac: read-out {nf_far:.2e}  vs raw end node {nf_raw:.2e}")
    print(f"  figure: {panel_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
