"""Diagnostics for the single-reed mouthpiece — the wind leg's nonlinear self-oscillating exciter.

The reed is an *active* element that turns a steady breath into a sustained clarinet tone (the
acoustic dual of the bow). Four figures land in ./out/:

  1. **Energy balance, not conservation.** For a lossless bore the total stored energy
     E = E_bore + E_reed is accounted for exactly by the breath minus the (sign-definite) jet and
     reed-damping losses: E - E0 == mouth_work - jet_loss - reed_damp_work to machine precision.
     The channels are drawn (breath in, jet + reed-damp out, stored energy) with the residual.
  2. **The note speaks.** Mouthpiece pressure (goes square-ish) and the reed opening H^+ over a few
     periods, showing the once-per-period **beating** (the reed slamming shut) under normal blowing.
  3. **The clarinet spectrum.** Interior pickup spectrum: **odd** harmonics at (2n-1) c/4L dominate,
     the even harmonics deeply suppressed — the clarinet's hallmark. Pitch is set by the bore.
  4. **Oscillation threshold.** Steady tone amplitude vs the blowing ratio gamma = p_m / p_closing:
     a sharp speak-onset near gamma ~ 1/3 (Dalmont/Kergomard), here a touch higher for this bore's
     losses. Below threshold the note fails to speak; above it the amplitude climbs.

Run:  python scripts/diagnose_reed.py
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
from physsynth.core.reed import ReedBore  # noqa: E402

L, RADIUS, N = 0.5, 0.008, 200  # closed-open f1 = c0/4L = 171.5 Hz
OUT = "out"
F1 = modal.bore_resonance_frequencies(C0_AIR, L, 1, "closed-open")[0]


def build(*, p_mouth, boundary=("closed", "radiating"), R_bell=650.0, sigma=0.0, f_reed=2500.0):
    fs = C0_AIR / (1.0 * (L / N))  # lambda = 1 (dispersionless)
    bore = Bore(L=L, fs=fs, N=N, radius=RADIUS, boundary=boundary, R_bell=R_bell, sigma=sigma)
    return ReedBore(bore=bore, p_mouth=p_mouth, f_reed=f_reed)


def energy_channels(p_mouth=1500.0, secs=0.25):
    """Lossless-bore energy balance: stored energy vs the breath/jet/reed-damp channels."""
    reed = build(p_mouth=p_mouth, boundary=("closed", "open"), sigma=0.0)
    n = int(secs * reed.k**-1)
    t = np.empty(n + 1)
    e, mouth, jet, damp = (np.empty(n + 1) for _ in range(4))
    # Index 0 is the rest state (E = 0, all channels 0) — the correct balance baseline; subtracting
    # the post-first-step energy instead would drop the first step's work from only one side.
    t[0], e[0], mouth[0], jet[0], damp[0] = 0.0, reed.energy(), 0.0, 0.0, 0.0
    for i in range(1, n + 1):
        reed.step()
        t[i] = i * reed.k
        e[i] = reed.energy()
        mouth[i], jet[i], damp[i] = reed.mouth_work, reed.jet_loss, reed.reed_damp_work
    residual = (e - e[0]) - (mouth - jet - damp)
    return t, e, mouth, jet, damp, residual


def waveform(p_mouth=1500.0, secs=0.4):
    """Mouthpiece pressure and reed opening over the last few periods of a settled tone."""
    reed = build(p_mouth=p_mouth)
    n = int(secs * reed.k**-1)
    mp = np.empty(n)
    opening = np.empty(n)
    for i in range(n):
        reed.step()
        mp[i] = reed.mouthpiece_pressure()
        opening[i] = reed.reed_opening()
    t = np.arange(n) * reed.k
    win = t >= (secs - 5.0 / F1)  # last ~5 periods
    return t[win] * 1e3, mp[win], opening[win], reed.H0


def interior_spectrum(p_mouth=1500.0, secs=0.8):
    reed = build(p_mouth=p_mouth)
    n = int(secs * reed.k**-1)
    sig = np.empty(n)
    for i in range(n):
        reed.step()
        sig[i] = reed.displacement_at(1)  # INTERIOR pickup (node 1)
    freqs, mag, _ = spectrum.magnitude_spectrum(sig[n // 2 :], reed.k**-1)
    return freqs, mag


def threshold_sweep(secs=0.5):
    """Steady tone amplitude vs blowing ratio gamma (scan p_mouth), bracketing the speak-onset."""
    ref = build(p_mouth=1.0)
    p_closing = ref.p_closing
    gammas = np.linspace(0.08, 0.9, 22)
    rms = np.empty_like(gammas)
    for j, g in enumerate(gammas):
        reed = build(p_mouth=g * p_closing)
        n = int(secs * reed.k**-1)
        mp = np.empty(n)
        for i in range(n):
            reed.step()
            mp[i] = reed.mouthpiece_pressure()
        rms[j] = np.sqrt(np.mean(mp[n // 2 :] ** 2))
    return gammas, rms, p_closing


def main() -> None:
    os.makedirs(OUT, exist_ok=True)

    t, e, mouth, jet, damp, residual = energy_channels()
    bal = float(np.max(np.abs(residual)) / (np.max(np.abs(mouth)) + 1e-30))

    tw, mp, opening, H0 = waveform()
    freqs, mag = interior_spectrum()
    gammas, rms, p_closing = threshold_sweep()

    # --- figure -------------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0, 0]
    ax.plot(t * 1e3, e - e[0], color="C0", label="stored $E-E_0$")
    ax.plot(t * 1e3, mouth, color="C2", label="breath in (mouth work)")
    ax.plot(t * 1e3, jet, color="C3", label="jet loss")
    ax.plot(t * 1e3, damp, color="C1", label="reed damping")
    ax.plot(t * 1e3, mouth - jet - damp, "k--", lw=1.0, label="in $-$ losses")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("energy (J)")
    ax.set_title(f"energy balance (lossless bore): residual {bal:.1e} rel.")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(tw, mp, color="C0", label="mouthpiece $p_0$ (Pa)")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("pressure (Pa)", color="C0")
    ax.tick_params(axis="y", labelcolor="C0")
    ax2 = ax.twinx()
    ax2.plot(tw, opening * 1e3, color="C3", lw=1.0, label="opening $H^+$ (mm)")
    ax2.axhline(H0 * 1e3, color="C7", ls=":", lw=0.8)
    ax2.axhline(0.0, color="k", ls=":", lw=0.8)
    ax2.set_ylabel("reed opening (mm)", color="C3")
    ax2.tick_params(axis="y", labelcolor="C3")
    ax.set_title("the note speaks: square-ish $p_0$, reed beating shut ($H^+\\to 0$)")

    ax = axes[1, 0]
    fmax = 12.0 * F1
    sel = freqs <= fmax
    ax.semilogy(freqs[sel], mag[sel] / mag[sel].max(), color="C0", lw=1.0)
    for nn in range(1, 7):
        ax.axvline((2 * nn - 1) * F1, color="C0", ls=":", alpha=0.35)
        ax.axvline(2 * nn * F1, color="C3", ls=":", alpha=0.35)
    ax.set_ylim(1e-7, 2.0)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("|spectrum| (norm.)")
    ax.set_title("clarinet spectrum: odd harmonics (blue) dominate, even (red) suppressed")

    ax = axes[1, 1]
    ax.plot(gammas, rms, "o-", color="C0", ms=4)
    ax.axvline(1.0 / 3.0, color="C3", ls=":", label=r"$\gamma=1/3$ (small-osc. threshold)")
    ax.set_xlabel(r"blowing ratio $\gamma = p_m / p_{closing}$")
    ax.set_ylabel("steady mouthpiece rms (Pa)")
    ax.set_title(f"oscillation threshold  ($p_{{closing}}={p_closing:.0f}$ Pa)")
    ax.legend(fontsize=8)

    fig.suptitle("Single-reed mouthpiece — clarinet (wind-leg batch-3 validation)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path = os.path.join(OUT, "reed_diagnostics.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)

    # --- console report -----------------------------------------------------------------------
    onset = gammas[np.argmax(rms > 0.05 * rms.max())]
    print("=" * 70)
    print("Single-reed mouthpiece — clarinet (wind-leg batch-3 report)")
    print("=" * 70)
    print(f"  c0 = {C0_AIR:.1f} m/s,  L = {L} m,  f1(closed-open) = {F1:.2f} Hz")
    print(f"  p_closing = {p_closing:.0f} Pa   (gamma = p_mouth / p_closing)")
    print(f"  [1] lossless energy balance residual   = {bal:.3e}   (tol 1e-11)")
    print(f"      breath = {mouth[-1]:.3e} J,  jet = {jet[-1]:.3e} J,  reed-damp = {damp[-1]:.3e} J")
    print(f"  [2] mouthpiece pressure amplitude       = {np.max(np.abs(mp)):.0f} Pa (beats shut)")
    print(f"  [4] speak-onset near gamma              = {onset:.2f}  (small-osc. rule ~ 1/3)")
    print(f"  figure: {path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
