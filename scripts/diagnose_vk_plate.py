"""Von Kármán nonlinear-plate (model #6) diagnostic harness.

Runs the coupled ``VKPlate`` and renders the diagnostics-as-visuals (HANDOFF §7) that are *specific
to the nonlinearity* — the linear plate's oracle-vs-detected bars are covered by
``diagnose_plate.py``; here the energy method and the amplitude-dependent pitch are the physics.
Because the von Kármán plate has **no analytic modal oracle**, energy conservation *is* the
correctness test, so the headline visual is the energy trace itself.

Four panels + a stress-field snapshot + a struck animation:

- **Energy breakdown (lossless).** Total, linear (kinetic + bending) and membrane (the stretched
  Airy field ``F``) energy over a large-amplitude strike (``w ≈ 3 e``). The linear and membrane
  stores anti-correlate — energy sloshes between bending and membrane stretch — while the **total**
  stays flat to machine precision. That flat total riding over two out-of-phase components is the
  visual proof of conservation-with-coupling.
- **Pitch-glide sweep.** The fundamental vs drive amplitude ``w/e``, holding the grid fixed so the
  curve is pure physics: it rises monotonically (geometric hardening) off the ``w → 0`` linear
  limit (model #5's frequency).
- **Pitch-glide spectrogram.** A struck plate under light loss: as it rings down, the hardened
  fundamental glides back *down* toward its linear limit — a descending track (needs ``sigma > 0``;
  a lossless run would hold pitch flat).
- **Stress field.** The displacement ``w`` and its induced Airy stress function ``F`` at peak
  membrane energy, side by side — ``F`` is the in-plane state that makes this model nonlinear and
  has no analogue in the linear plate.

Prints the real validation numbers (drift, peak membrane fraction, glide table, Picard convergence)
so the model can be judged from the console alone.

Run:  python scripts/diagnose_vk_plate.py
Figures land in ./out/.
"""

from __future__ import annotations

import os
import sys

# Make `physsynth` importable when run directly (no editable install): repo root on sys.path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from physsynth.core.engine import simulate  # noqa: E402
from physsynth.core.operators2d import embed  # noqa: E402
from physsynth.core.plate import VKPlate  # noqa: E402
from physsynth.viz import plots  # noqa: E402

# A thin steel-like plate (E, e, nu, rho) -> derives rho_s, D, kappa, Y = E e. Same material as the
# validation tests. Thickness e is the amplitude scale of the nonlinearity (onset at w ~ e).
MAT = dict(E=2.0e11, e=1.0e-3, nu=0.3, rho=7800.0)
OUT = "out"


def ss_fundamental(vk, Lx):
    """Analytic simply-supported linear fundamental ``f = (pi/2) kappa [(1/Lx)^2 + (1/Ly)^2]``.

    The ``w -> 0`` limit of the nonlinear plate (model #5's frequency) — the asymptote the pitch
    glide rises off / descends toward. Uses the snapped ``vk.Ly``.
    """
    return (np.pi / 2.0) * vk.kappa * ((1.0 / Lx) ** 2 + (1.0 / vk.Ly) ** 2)


def zero_cross_fundamental(sig, fs):
    """Fundamental from the mean spacing of zero-crossings (robust to the hardened pitch shift).

    The pitch shifts +tens of % with amplitude, so an FFT window placed on the *linear* frequency
    would miss it; zero-crossing spacing tracks the moving fundamental directly (the method proven
    on this exact sweep in ``test_pitch_glide_hardening``).
    """
    sig = np.asarray(sig, dtype=float)
    sig = sig - sig.mean()
    zc = np.where(np.diff(np.signbit(sig)))[0]
    if len(zc) < 3:
        return float("nan")
    return fs / (2.0 * np.mean(np.diff(zc)))


def centered_strike(vk, Lx, amp, width_frac=0.28):
    """A broad centered raised-Gaussian strike (full field, rim zeroed), peak ``amp``.

    Broad + centered favours the (1,1) fundamental, so the glide reads as one clean descending tone
    rather than a busy multi-mode wash (the crash/cascade is Part 6's job, not Part 5's).
    """
    dx = vk.X - 0.5 * Lx
    dy = vk.Y - 0.5 * vk.Ly
    w = width_frac * Lx
    s = amp * np.exp(-((dx * dx + dy * dy) / (w * w)))
    s[~vk.mask] = 0.0
    return s


def mode11(vk, Lx):
    """The (1,1) simply-supported eigenmode ``sin(pi x/Lx) sin(pi y/Ly)`` (peak 1.0)."""
    return np.sin(np.pi * vk.X / Lx) * np.sin(np.pi * vk.Y / vk.Ly)


def energy_breakdown():
    """Lossless large-amplitude run: total / linear / membrane energy + a peak-membrane snapshot."""
    Lx = Ly = 0.4
    fs = 48000.0
    n_steps = int(0.2 * fs)
    vk = VKPlate(Lx=Lx, Ly=Ly, fs=fs, N=24, **MAT)
    vk.set_state(3.0 * MAT["e"] * centered_strike(vk, Lx, 1.0))  # peak w ~ 3 e -> strong nonlinear

    time = np.empty(n_steps + 1)
    total = np.empty(n_steps + 1)
    linear = np.empty(n_steps + 1)
    membrane = np.empty(n_steps + 1)
    iters_max = 0
    all_converged = True
    peak_frac = 0.0
    peak_w_field = vk.state
    peak_F_field = vk.stress_field
    for i in range(n_steps + 1):
        if i > 0:
            vk.step()
            iters_max = max(iters_max, vk.n_iters)
            all_converged = all_converged and vk.converged
        e_lin = vk.linear_energy()
        e_mem = vk.membrane_energy()
        e_tot = e_lin + e_mem
        time[i] = i * vk.k
        total[i], linear[i], membrane[i] = e_tot, e_lin, e_mem
        frac = e_mem / e_tot if e_tot > 0 else 0.0
        if frac > peak_frac:
            peak_frac = frac
            peak_w_field, peak_F_field = vk.state, vk.stress_field

    drift = float(np.max(np.abs(total - total[0])) / abs(total[0]))
    print("Energy conservation (lossless, w ~ 3 e, N=24, ~0.2 s):")
    print(f"  drift max|E-E0|/E0 = {drift:.2e}   peak membrane fraction = {100 * peak_frac:.0f}%")
    print(f"  Picard: max sweeps/step = {iters_max}   all steps converged = {all_converged}")

    fig, ax = plt.subplots(figsize=(8, 4))
    plots.plot_energy_breakdown(ax, time, total, linear, membrane, drift=drift)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "vk_energy_breakdown.png"), dpi=110)
    plt.close(fig)

    # Stress-field snapshot at peak membrane energy: w and its induced Airy F side by side.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    plots.plot_membrane_field(axes[0], vk.X, vk.Y, peak_w_field, mask=vk.mask,
                              title=f"displacement $w$  (membrane {100 * peak_frac:.0f}% of E)")
    plots.plot_membrane_field(axes[1], vk.X, vk.Y, peak_F_field, mask=vk.mask,
                              title="Airy stress function $F$  (the in-plane store)")
    fig.suptitle("Von Kármán coupling: transverse $w$ stretches the in-plane stress field $F$")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "vk_stress_field.png"), dpi=110)
    plt.close(fig)


def pitch_glide_sweep():
    """Fundamental vs drive amplitude at a fixed grid — the hardening curve."""
    Lx = Ly = 0.15  # small plate -> high fundamental -> many periods in a short run
    fs = 48000.0
    N = 18
    amps = [0.01, 0.5, 1.0, 2.0, 3.0, 5.0]
    f0s = []
    print("\nPitch-glide sweep (fixed grid N=18, fs=48 kHz; f0 via zero-crossings):")
    f_lin = None
    for amp in amps:
        vk = VKPlate(Lx=Lx, Ly=Ly, fs=fs, N=N, **MAT)
        if f_lin is None:
            f_lin = ss_fundamental(vk, Lx)
        vk.set_state(amp * MAT["e"] * mode11(vk, Lx))
        pk = vk.pickup_index_at(0.47 * Lx, 0.53 * vk.Ly)
        sig = np.empty(5000)
        worst_res = 0.0
        for i in range(5000):
            vk.step()
            worst_res = max(worst_res, vk.last_residual)
            sig[i] = vk.displacement_at(pk)
        f0 = zero_cross_fundamental(sig, fs)
        f0s.append(f0)
        print(f"  w = {amp:4.2f} e   f0 = {f0:7.2f} Hz   ({100 * (f0 / f0s[0] - 1):+5.0f}% vs "
              f"w=0)   max Picard resid = {worst_res:.1e}")
    print(f"  linear (w->0) SS law f0 = {f_lin:.2f} Hz")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    plots.plot_pitch_glide(ax, amps, f0s, f_lin)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "vk_pitch_glide.png"), dpi=110)
    plt.close(fig)


def pitch_glide_spectrogram():
    """Struck plate under light loss: the fundamental glides down over the ring-down."""
    Lx = Ly = 0.15
    fs = 48000.0
    N = 18
    dur = 1.0
    n_steps = int(dur * fs)
    vk = VKPlate(Lx=Lx, Ly=Ly, fs=fs, N=N, sigma=3.0, **MAT)  # light loss -> ~1 s ring-down
    f_lin = ss_fundamental(vk, Lx)
    vk.set_state(4.0 * MAT["e"] * centered_strike(vk, Lx, 1.0))  # strike hard -> big initial rise
    pk = vk.pickup_index_at(0.47 * Lx, 0.53 * vk.Ly)
    sig = np.empty(n_steps)
    worst_res = 0.0
    n_not_conv = 0
    for i in range(n_steps):
        vk.step()
        if not vk.converged:
            n_not_conv += 1
        worst_res = max(worst_res, vk.last_residual)
        sig[i] = vk.displacement_at(pk)
    print(f"\nPitch-glide spectrogram (w~4e strike, sigma=3, {dur:.1f} s ring-down):")
    print(f"  linear-limit fundamental = {f_lin:.1f} Hz   non-converged steps = {n_not_conv}"
          f"   max Picard resid = {worst_res:.1e}")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    plots.plot_spectrogram(ax, sig, fs, f_max=3.0 * f_lin, f_linear=f_lin, nperseg=2048)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "vk_spectrogram.png"), dpi=110)
    plt.close(fig)

    # A struck-plate displacement animation (reuses the membrane animator).
    va = VKPlate(Lx=Lx, Ly=Ly, fs=fs, N=N, **MAT)
    va.set_state(3.0 * MAT["e"] * centered_strike(va, Lx, 1.0))
    n_anim = int(0.03 * fs)
    resa = simulate(va, num_steps=n_anim, snapshot_stride=max(1, n_anim // 60))
    snaps = [(s, embed(v[va.mask] if v.shape == va.mask.shape else v, va.index_map))
             for s, v in resa.snapshots]
    ok = plots.save_membrane_animation(
        os.path.join(OUT, "vk_strike.gif"), va.X, va.Y, snaps, fs, mask=va.mask
    )
    print(f"  animation written: {ok}  ->  out/vk_strike.gif")


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    energy_breakdown()
    pitch_glide_sweep()
    pitch_glide_spectrogram()


if __name__ == "__main__":
    main()
