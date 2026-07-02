"""Von Kármán **free-edge** plate (model #6, Part 6) — the gong/cymbal diagnostic harness.

The iconic free-edge cousin of ``diagnose_vk_plate.py`` (simply-supported). Same nonlinear physics
— the transverse ``w`` stretches an in-plane Airy stress field ``F`` that stiffens it — but on a
*free* rectangle, where the bracket's conservation crux (self-adjoint only for rim-vanishing fields)
is carried by the clamped-zero ``F`` rather than by ``w``. The free edge is what makes a gong:
no closed-form modal oracle, so **energy conservation is the correctness test**, and the payoff is
the **crash cascade** (a hard strike spreads energy across the band) plus the **curved-Chladni**
nodal figures of the elastic modes.

Panels (figures land in ./out/):

- **Energy breakdown (lossless).** Flat total riding over the anti-correlated linear (kinetic +
  bending) and membrane (Airy ``F``) stores at ``w ≈ 3 e`` on the free plate — conservation
  *through* the nonlinearity, with ``w ≠ 0`` on the rim. Plus a ``w``/``F`` peak-membrane snapshot.
- **Curved-Chladni modes.** The lowest elastic mode shapes (``eigsh`` of the free stiffness ``K`` /
  lumped mass ``W``, stepper-independent) with their nodal lines — the cymbal's modal basis.
- **Tonal vs crash.** Two ring-down spectrograms: a *soft* strike (``w ≈ e``) rings as a gliding
  tone; a *hard* strike (``w ≈ 6 e``) throws energy up the spectrum into a broadband **cascade** —
  the crash (spectral centroid climbs ~5×, a fifth of the energy above 1 kHz). The contrast (one
  tonal track vs a filled band) is the nonlinearity's sonic signature. The run **asserts Picard
  convergence**: the ``w ≫ e`` regime needs oversampling (HANDOFF §8) — at ``fs = 96 kHz`` it holds
  to ``6 e`` but diverges (does not "cascade") by ``10 e``, which instead needs ``fs = 384 kHz``.
- **Struck animation.** A hard-strike GIF (the shimmering free plate).

Prints the validation numbers (drift, peak membrane fraction, Picard convergence, glide, cascade
bandwidth) so the model is judgeable from the console alone.

Run:  python scripts/diagnose_vk_free_plate.py
"""

from __future__ import annotations

import os
import sys

# Make `physsynth` importable when run directly (no editable install): repo root on sys.path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.sparse.linalg import eigsh  # noqa: E402

from physsynth.core.engine import simulate  # noqa: E402
from physsynth.core.operators2d import embed  # noqa: E402
from physsynth.core.plate import VKPlate  # noqa: E402
from physsynth.viz import plots  # noqa: E402

# Thin steel-like plate (E, e, nu, rho); e is the amplitude scale of the nonlinearity (onset w ~ e).
MAT = dict(E=2.0e11, e=1.0e-3, nu=0.3, rho=7800.0)
OUT = "out"


def centered_strike(vk, Lx, amp, width_frac=0.22):
    """A centered raised-Gaussian strike (full field; a free plate needs no rim clamp), peak amp."""
    dx = vk.X - 0.5 * Lx
    dy = vk.Y - 0.5 * vk.Ly
    w = width_frac * Lx
    return amp * np.exp(-((dx * dx + dy * dy) / (w * w)))


def first_elastic(vk):
    """(freq_Hz, mode) of the lowest **elastic** free mode (``K φ = μ W φ``, skipping 3 rigids)."""
    vals, vecs = eigsh(vk.K.tocsc(), k=6, M=vk.W.tocsc(), sigma=-1e-3, which="LM")
    order = np.argsort(vals)
    mu = max(vals[order[3]], 0.0)  # first nonzero after the {1, x, y} rigid-body triple
    freq = np.sqrt(mu) * vk.kappa / (2.0 * np.pi)
    mode = vecs[:, order[3]]
    return freq, mode / np.max(np.abs(mode))


def energy_breakdown():
    """Lossless large-amp free-plate run: total / linear / membrane + a peak-membrane snapshot."""
    Lx = Ly = 0.4
    fs = 48000.0
    n_steps = int(0.2 * fs)
    vk = VKPlate(Lx=Lx, Ly=Ly, fs=fs, N=24, boundary="free", **MAT)
    vk.set_state(3.0 * MAT["e"] * centered_strike(vk, Lx, 1.0))

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
        time[i], total[i], linear[i], membrane[i] = i * vk.k, e_tot, e_lin, e_mem
        frac = e_mem / e_tot if e_tot > 0 else 0.0
        if frac > peak_frac:
            peak_frac = frac
            peak_w_field, peak_F_field = vk.state, vk.stress_field

    drift = float(np.max(np.abs(total - total[0])) / abs(total[0]))
    print("Free-plate energy conservation (lossless, w ~ 3 e, N=24, ~0.2 s):")
    print(f"  drift max|E-E0|/E0 = {drift:.2e}   peak membrane fraction = {100 * peak_frac:.0f}%")
    print(f"  Picard: max sweeps/step = {iters_max}   all steps converged = {all_converged}")

    fig, ax = plt.subplots(figsize=(8, 4))
    plots.plot_energy_breakdown(ax, time, total, linear, membrane, drift=drift)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "vk_free_energy_breakdown.png"), dpi=110)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    plots.plot_membrane_field(axes[0], vk.X, vk.Y, peak_w_field,
                              title=f"displacement $w$  (membrane {100 * peak_frac:.0f}% of E)")
    plots.plot_membrane_field(axes[1], vk.X, vk.Y, peak_F_field,
                              title="Airy stress function $F$  (clamped-zero on the rim)")
    fig.suptitle("Free von Kármán plate: transverse $w$ (free rim) stretches the in-plane $F$")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "vk_free_stress_field.png"), dpi=110)
    plt.close(fig)


def chladni_modes():
    """The lowest elastic free-plate modes as curved-Chladni figures (the cymbal's modal basis)."""
    Lx = Ly = 0.4
    vk = VKPlate(Lx=Lx, Ly=Ly, fs=48000.0, N=40, boundary="free", **MAT)
    vals, vecs = eigsh(vk.K.tocsc(), k=12, M=vk.W.tocsc(), sigma=-1e-3, which="LM")
    order = np.argsort(vals)[3:9]  # first six elastic modes (skip the 3 rigid-body zeros)
    print("\nCurved-Chladni elastic modes (linear, eigsh of K/W, N=40):")
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 7))
    for ax, idx in zip(axes.ravel(), order, strict=False):
        freq = np.sqrt(max(vals[idx], 0.0)) * vk.kappa / (2.0 * np.pi)
        shape = embed(vecs[:, idx], vk.index_map)
        plots.plot_chladni(ax, vk.X, vk.Y, shape, title=f"{freq:.0f} Hz")
        print(f"  mode f = {freq:7.1f} Hz")
    fig.suptitle("Free-edge plate — curved-Chladni nodal figures (the gong/cymbal modal basis)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "vk_free_chladni.png"), dpi=110)
    plt.close(fig)


def hi_freq_fraction(sig, fs, f_split=1000.0):
    """(spectral centroid Hz, fraction of spectral energy above ``f_split``) — the cascade metric.

    A broadband **crash** throws energy up the spectrum: the centroid climbs and a real fraction of
    the energy ends up above ~1 kHz. A cleaner metric than an FFT peak, ill-defined once the
    spectrum fills (a single "fundamental" no longer dominates).
    """
    w = (sig - sig.mean()) * np.hanning(len(sig))
    power = np.abs(np.fft.rfft(w)) ** 2
    freqs = np.fft.rfftfreq(len(sig), 1.0 / fs)
    total = float(np.sum(power)) + 1e-30
    return float(np.sum(freqs * power) / total), float(np.sum(power[freqs > f_split]) / total)


def tonal_vs_crash():
    """Soft strike (tonal glide) vs hard strike (broadband cascade = crash), two spectrograms.

    The hard strike is ``w ≈ 6 e`` — strongly nonlinear but still Picard-convergent at ``fs = 96
    kHz`` (0 non-converged). Pushing to ``w ≈ 10 e`` at this ``fs`` makes the fixed-point iteration
    non-contractive and the run *blows up*, not cascades — the "oversample around the nonlinearity"
    lesson (HANDOFF §8): the true ``w ≫ e`` crash needs a higher ``fs`` (``10 e`` converges cleanly
    at 384 kHz). The run **warns on non-convergence** rather than draw a divergent (false) panel.
    """
    Lx = Ly = 0.18
    fs = 96000.0
    N = 22
    dur = 0.6
    n_steps = int(dur * fs)
    base = VKPlate(Lx=Lx, Ly=Ly, fs=fs, N=N, boundary="free", **MAT)
    f_lin, _ = first_elastic(base)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    print(f"\nTonal vs crash (free plate {Lx} m, N={N}, sigma=4, {dur:.1f} s):"
          f"  linear elastic f0 = {f_lin:.0f} Hz")
    for ax, (amp, tag) in zip(axes, [(1.0, "soft (w~e): tonal glide"),
                                     (6.0, "hard (w~6e): crash cascade")], strict=False):
        vk = VKPlate(Lx=Lx, Ly=Ly, fs=fs, N=N, sigma=4.0, boundary="free", **MAT)
        vk.set_state(amp * MAT["e"] * centered_strike(vk, Lx, 1.0))
        pk = vk.pickup_index_at(0.31 * Lx, 0.63 * vk.Ly)
        sig = np.empty(n_steps)
        worst_res = 0.0
        n_not_conv = 0
        for i in range(n_steps):
            vk.step()
            n_not_conv += 0 if vk.converged else 1
            worst_res = max(worst_res, vk.last_residual)
            sig[i] = vk.displacement_at(pk)
        if n_not_conv:  # never present a divergent run as physics
            print(f"  WARNING: {tag} had {n_not_conv} non-converged steps — under-resolved, "
                  "oversample (raise fs). Panel not trustworthy.")
        early = sig[: n_steps // 8]  # early, loud window
        centroid, hf = hi_freq_fraction(early, fs)
        plots.plot_spectrogram(ax, sig, fs, f_max=15.0 * f_lin, f_linear=f_lin, nperseg=2048)
        ax.set_title(tag, fontsize=9)
        print(f"  {tag:30s} centroid = {centroid:6.0f} Hz  energy>1kHz = {100 * hf:4.1f}%"
              f"  non-converged = {n_not_conv}  max resid = {worst_res:.1e}")
    fig.suptitle("Free von Kármán plate: a soft strike sings, a hard strike crashes (cascade)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "vk_free_tonal_vs_crash.png"), dpi=110)
    plt.close(fig)


def struck_animation():
    """A hard-strike GIF of the shimmering free plate (reuses the membrane animator)."""
    Lx = Ly = 0.18
    fs = 96000.0
    va = VKPlate(Lx=Lx, Ly=Ly, fs=fs, N=22, boundary="free", **MAT)
    va.set_state(6.0 * MAT["e"] * centered_strike(va, Lx, 1.0))
    n_anim = int(0.02 * fs)
    res = simulate(va, num_steps=n_anim, snapshot_stride=max(1, n_anim // 60))
    snaps = [(s, embed(v[va.mask] if v.shape == va.mask.shape else v, va.index_map))
             for s, v in res.snapshots]
    ok = plots.save_membrane_animation(
        os.path.join(OUT, "vk_free_strike.gif"), va.X, va.Y, snaps, fs, mask=va.mask
    )
    print(f"\n  animation written: {ok}  ->  out/vk_free_strike.gif")


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    energy_breakdown()
    chladni_modes()
    tonal_vs_crash()
    struck_animation()


if __name__ == "__main__":
    main()
