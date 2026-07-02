"""Diagnostic plots — the Phase-1 "visuals are the diagnostics" layer (HANDOFF §7).

Pure plotting helpers that take NumPy arrays and draw onto Matplotlib axes; they import the core's
*data*, never its modules, so the headless-core invariant is untouched. The Agg backend is forced so
everything renders without a display (CI / headless).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless: no display required

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from numpy.typing import NDArray  # noqa: E402

__all__ = [
    "plot_energy",
    "plot_partials",
    "plot_stiff_partials",
    "plot_decay_rates",
    "plot_displacement_snapshots",
    "plot_convergence",
    "plot_spectrum",
    "plot_dispersion",
    "save_displacement_animation",
    "plot_membrane_field",
    "plot_membrane_partials",
    "plot_chladni",
    "plot_energy_breakdown",
    "plot_pitch_glide",
    "plot_spectrogram",
    "save_membrane_animation",
]


def plot_energy(ax, time: NDArray, energy: NDArray, drift: float | None = None) -> None:
    """Energy vs time, annotated with the relative drift (the conservation figure of merit)."""
    e0 = energy[0]
    ax.plot(time, energy / e0, lw=1.0)
    ax.set_xlabel("time (s)")
    ax.set_ylabel(r"$E^n / E^0$")
    title = "Discrete energy (lossless)"
    if drift is not None:
        title += f"  —  drift $\\max|E^n-E^0|/E^0$ = {drift:.2e}"
    ax.set_title(title)
    ax.grid(True, alpha=0.3)


def plot_partials(ax, analytic: NDArray, detected: NDArray) -> None:
    """Detected partials vs the analytic harmonic series, with per-partial error in cents."""
    n = np.arange(1, len(analytic) + 1)
    err_cents = 1200.0 * np.log2(detected / analytic)
    ax.axhline(0.0, color="k", lw=0.8)
    ax.stem(n, err_cents, basefmt=" ")
    ax.set_xlabel("partial number $n$")
    ax.set_ylabel("error (cents)")
    worst = np.nanmax(np.abs(err_cents))
    ax.set_title(f"Detected vs analytic $f_n = n\\,c/2L$  (worst |err| = {worst:.3f} cents)")
    ax.grid(True, alpha=0.3)


def plot_stiff_partials(
    ax, detected: NDArray, stretched: NDArray, f0: float, B: float
) -> None:
    """Stiff-string partials: the *stretch* off the harmonic series, detected vs theory.

    Plotted as deviation in cents from the plain harmonic grid ``n f0`` (so a perfectly harmonic
    string would sit on zero). Bending pushes every partial sharp by ``sqrt(1 + B n^2)``, a curve
    that grows with ``n`` — the audible inharmonicity. The detected partials (markers) must land on
    the continuum stretched law (solid line); their agreement is the validation, their rise off zero
    is the physics.
    """
    n = np.arange(1, len(detected) + 1)
    harmonic = n * f0
    detected_cents = 1200.0 * np.log2(detected / harmonic)
    theory_cents = 1200.0 * np.log2(stretched / harmonic)
    ax.axhline(0.0, color="k", lw=0.8, label=r"harmonic series $n f_0$")
    ax.plot(n, theory_cents, "-", color="C3", lw=1.4,
            label=r"theory $n f_0\sqrt{1+Bn^2}$")
    ax.plot(n, detected_cents, "o", color="C0", ms=6, label="detected (FDTD)")
    ax.set_xlabel("partial number $n$")
    ax.set_ylabel("stretch off $n f_0$ (cents)")
    worst = float(np.nanmax(np.abs(detected_cents - theory_cents)))
    ax.set_title(
        f"Stretched partials (inharmonicity $B$ = {B:.2e};  detected−theory < {worst:.3f} cents)"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")


def plot_decay_rates(
    ax,
    modes: NDArray,
    measured: NDArray,
    oracle_with: NDArray,
    oracle_without: NDArray,
) -> None:
    """Per-mode energy decay rate ``Gamma_m`` vs mode -- the frequency-dependent-loss cure, drawn.

    Three curves over the in-band mode range:

    - ``oracle_with`` (solid, rising) -- the closed-form discrete rate *with* ``sigma1 > 0``: high
      partials decay faster, as real strings do;
    - markers (``measured``) -- the rate measured from single-mode FDTD runs, landing on that line;
    - ``oracle_without`` (dashed, falling) -- the same string with ``sigma1 = 0`` (model #2's
      frequency-*independent* loss): the rate *droops* with mode (high partials under-damp), the
      audibly backwards artifact ``sigma1`` cures.

    The contrast -- markers on the rising solid line, the dashed line sloping the other way -- is
    the whole point: ``sigma1`` flips the high-vs-low decay ordering.
    """
    modes = np.asarray(modes)
    ax.plot(modes, oracle_with, "-", color="C0", lw=1.4,
            label=r"oracle, $\sigma_1>0$ (highs die faster)")
    ax.plot(modes, measured, "o", color="C0", ms=6, label="measured (FDTD)")
    ax.plot(modes, oracle_without, "--", color="C3", lw=1.4,
            label=r"oracle, $\sigma_1=0$ (model #2: highs under-damp)")
    ax.set_xlabel("mode number $m$")
    ax.set_ylabel(r"energy decay rate $\Gamma_m$ (1/s)")
    worst = float(np.nanmax(np.abs(measured - oracle_with) / oracle_with))
    ax.set_title(
        rf"Per-mode decay: $E_m(t)\sim e^{{-\Gamma_m t}}$  (measured vs oracle < {worst:.1e})"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")


def plot_displacement_snapshots(ax, x: NDArray, snapshots, fs: float) -> None:
    """Small-multiples of the string shape u(x) at a sequence of times (a static 'animation')."""
    cmap = plt.get_cmap("viridis")
    m = len(snapshots)
    for j, (step, state) in enumerate(snapshots):
        t_ms = 1e3 * step / fs
        ax.plot(x, state, color=cmap(j / max(m - 1, 1)), lw=1.0, label=f"{t_ms:.1f} ms")
    ax.set_xlabel("position x (m)")
    ax.set_ylabel("displacement u (m)")
    ax.set_title("String displacement over time")
    ax.grid(True, alpha=0.3)
    if m <= 12:
        ax.legend(fontsize=7, ncol=2, loc="upper right")


def plot_convergence(ax, step_sizes: NDArray, errors: NDArray) -> None:
    """Log-log frequency error vs grid spacing h, with an O(h^2) reference line."""
    step_sizes = np.asarray(step_sizes, dtype=float)
    errors = np.asarray(errors, dtype=float)
    ax.loglog(step_sizes, errors, "o-", label="measured partial error")
    ref = errors[0] * (step_sizes / step_sizes[0]) ** 2
    ax.loglog(step_sizes, ref, "k--", alpha=0.6, label=r"$O(h^2)$ reference")
    ax.set_xlabel("grid spacing h (m)")
    ax.set_ylabel("frequency error (Hz)")
    ax.set_title("Grid convergence at fixed $\\lambda < 1$")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()


def plot_spectrum(ax, freqs: NDArray, mag: NDArray, detected: NDArray, f_max: float) -> None:
    """Magnitude spectrum (dB) with detected partials marked."""
    sel = freqs <= f_max
    mag_db = 20.0 * np.log10(mag / np.max(mag) + 1e-12)
    ax.plot(freqs[sel], mag_db[sel], lw=0.8)
    for f in detected:
        if f <= f_max:
            ax.axvline(f, color="r", lw=0.6, alpha=0.5)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("magnitude (dB)")
    ax.set_ylim(-120, 5)
    ax.set_title("Pickup spectrum with detected partials")
    ax.grid(True, alpha=0.3)


def plot_dispersion(ax, modes: NDArray, cases) -> None:
    """Phase-velocity dispersion curve ``v_p/c`` vs mode number for one or more lambda cases.

    ``cases`` is a sequence of ``(lam, measured_vp_over_c, oracle_vp_over_c)``. Markers are the
    measured FDTD phase velocities; the solid line is the closed-form discrete-dispersion oracle.
    The dashed line at ``1.0`` is the continuum (dispersionless) reference: ``lambda = 1`` sits on
    it, while ``lambda < 1`` droops below as the mode number rises (high partials travel too slow).
    """
    modes = np.asarray(modes)
    cmap = plt.get_cmap("viridis")
    n = len(cases)
    ax.axhline(1.0, color="k", ls="--", lw=0.9, alpha=0.7, label=r"continuum ($v_p = c$)")
    for j, (lam, vp_measured, vp_oracle) in enumerate(cases):
        color = cmap(0.15 + 0.7 * j / max(n - 1, 1))
        ax.plot(modes, vp_oracle, "-", color=color, lw=1.2, alpha=0.9)
        ax.plot(
            modes, vp_measured, "o", color=color, ms=5,
            label=rf"$\lambda = {lam:g}$ (measured)",
        )
    # One neutral proxy so the reader knows the solid lines are the closed-form oracle (the markers
    # sit on them — that coincidence is the result).
    ax.plot([], [], "-", color="0.4", lw=1.2, label="closed-form oracle")
    ax.set_xlabel("mode number $m$")
    ax.set_ylabel(r"phase velocity $v_p / c$")
    ax.set_title(r"Numerical dispersion: $v_p(m) = 2L f_m / m$")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")


def plot_membrane_field(
    ax, X: NDArray, Y: NDArray, field: NDArray, mask: NDArray | None = None, title: str = ""
) -> None:
    """Heatmap of a 2D membrane field (a mode shape or a displacement snapshot).

    Dead (rim) nodes are masked to white so the domain — a rectangle or a staircased disk — reads as
    its actual shape. The colour scale is symmetric about zero (diverging) so nodal lines sit at the
    neutral colour: this *is* the Chladni-style visual the project is after (HANDOFF §7).
    """
    f = np.array(field, dtype=float)
    if mask is not None:
        f = np.where(mask, f, np.nan)
    vmax = float(np.nanmax(np.abs(f))) or 1.0
    im = ax.pcolormesh(X, Y, f, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    if title:
        ax.set_title(title)
    return im


def plot_membrane_partials(ax, detected: NDArray, oracle: NDArray, labels=None) -> None:
    """Detected membrane partials vs the analytic oracle (Bessel for a disk), error in cents.

    The 2D analogue of :func:`plot_partials`. ``labels`` (optional) are mode tags like ``"0,1"`` for
    the x-axis; the staircased-disk error is the looser, ~O(h) convergence-tier deviation (not the
    ~1-cent string bar), so the worst-case cents is annotated to keep that honest.
    """
    detected = np.asarray(detected, dtype=float)
    oracle = np.asarray(oracle, dtype=float)
    idx = np.arange(len(detected))
    err_cents = 1200.0 * np.log2(detected / oracle)
    ax.axhline(0.0, color="k", lw=0.8)
    ax.stem(idx, err_cents, basefmt=" ")
    ax.set_xlabel("mode")
    ax.set_ylabel("error (cents)")
    if labels is not None:
        ax.set_xticks(idx)
        ax.set_xticklabels(labels, fontsize=7, rotation=45)
    worst = float(np.nanmax(np.abs(err_cents)))
    ax.set_title(f"Detected vs analytic membrane partials (worst |err| = {worst:.2f} cents)")
    ax.grid(True, alpha=0.3)


def plot_chladni(
    ax, X: NDArray, Y: NDArray, field: NDArray, title: str = ""
) -> None:
    """A Chladni figure: the mode-shape heatmap with its **nodal lines** (zero contour) overlaid.

    The free-edge plate's iconic deliverable (``docs/dev/plate-free-edge-plan.md``). The
    displacement heatmap is diverging about zero (so the nodal set sits at the neutral colour, as in
    :func:`plot_membrane_field`); the solid black contour is the ``u = 0`` set — exactly the
    *curved* nodal lines that sand on a vibrating free plate collects into (the lines the SS plate,
    with its plain rectangular nodal grid, cannot show). Rendered straight from an ``eigsh``
    eigenvector, so it is independent of the time-stepper and doubles as a qualitative oracle (the
    low FFFF-square patterns — diagonal saddle cross, ring, grid — are textbook).
    """
    f = np.array(field, dtype=float)
    vmax = float(np.nanmax(np.abs(f))) or 1.0
    ax.pcolormesh(X, Y, f, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
    # The nodal lines: the zero level set of the mode shape (the sand pattern).
    ax.contour(X, Y, f, levels=[0.0], colors="k", linewidths=1.3)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=9)


def plot_energy_breakdown(
    ax,
    time: NDArray,
    total: NDArray,
    linear: NDArray,
    membrane: NDArray,
    drift: float | None = None,
) -> None:
    """Von Kármán energy conservation *with exchange* — total, linear (kinetic+bending), membrane.

    The nonlinear-plate signature (model #6): unlike a linear resonator, energy sloshes between the
    bending/kinetic store and the **membrane** store (the stretched Airy field ``F``) — they
    anti-correlate — while the **total** stays flat to machine precision. That flat total riding
    over two oscillating, out-of-phase components is the visual proof of conservation-with-coupling
    (there is no analytic modal oracle to check against; energy *is* the correctness test). All
    three are normalised to ``E^0`` so the total sits at 1.
    """
    e0 = total[0]
    ax.plot(time, total / e0, color="k", lw=1.4, label="total")
    ax.plot(time, linear / e0, color="C0", lw=1.0, label="linear (kinetic + bending)")
    ax.plot(time, membrane / e0, color="C3", lw=1.0, label="membrane (Airy $F$)")
    ax.set_xlabel("time (s)")
    ax.set_ylabel(r"$E / E^0$")
    title = "VK energy: flat total over anti-correlated linear/membrane exchange"
    if drift is not None:
        title += f"  —  drift = {drift:.2e}"
    ax.set_title(title, fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="center right")


def plot_pitch_glide(ax, amp_over_e: NDArray, f0: NDArray, f_linear: float) -> None:
    """Fundamental vs drive amplitude ``w/e`` — the hardening pitch glide (model #6).

    The weakly-nonlinear signature: geometric (membrane) stretching stiffens the plate, so the
    fundamental *rises* monotonically with amplitude. The dashed line is the small-amplitude
    (``w → 0``) linear limit — the frequency of the validated model #5 plate — which the curve must
    approach as the drive vanishes. There is no closed-form value for the shifted frequency; the
    *sign and monotonicity* are the physics.
    """
    amp_over_e = np.asarray(amp_over_e, dtype=float)
    f0 = np.asarray(f0, dtype=float)
    ax.axhline(f_linear, color="C3", ls="--", lw=1.1, label=r"linear limit ($w\to 0$, model #5)")
    ax.plot(amp_over_e, f0, "o-", color="C0", ms=6, lw=1.2, label="detected fundamental")
    ax.set_xlabel(r"drive amplitude $w_{\max}/e$ (thicknesses)")
    ax.set_ylabel("fundamental (Hz)")
    rise = 100.0 * (f0[-1] / f0[0] - 1.0)
    ax.set_title(
        f"Hardening pitch glide — fundamental rises {rise:+.0f}% by $w={amp_over_e[-1]:g}e$",
        fontsize=9,
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")


def plot_spectrogram(
    ax,
    signal: NDArray,
    fs: float,
    f_max: float,
    f_linear: float | None = None,
    nperseg: int = 2048,
) -> None:
    """Time–frequency spectrogram of a decaying strike — the pitch glide, *seen*.

    Under light loss the amplitude rings down, so the hardened fundamental **glides back down**
    toward its linear limit as the plate quietens: a descending track in the spectrogram. (A
    lossless run holds constant amplitude → flat pitch → no glide, so this needs ``sigma > 0``.) The
    dashed line marks the small-amplitude linear fundamental — the asymptote the glide descends
    toward. Uses a Hann STFT via :func:`scipy.signal.spectrogram`.
    """
    from scipy.signal import spectrogram

    sig = np.asarray(signal, dtype=float)
    sig = sig - sig.mean()
    nperseg = min(nperseg, len(sig))
    f, t, sxx = spectrogram(sig, fs=fs, window="hann", nperseg=nperseg,
                            noverlap=nperseg * 3 // 4)
    sel = f <= f_max
    sxx_db = 10.0 * np.log10(sxx[sel] / (sxx[sel].max() + 1e-30) + 1e-12)
    ax.pcolormesh(t, f[sel], sxx_db, cmap="magma", vmin=-80, vmax=0, shading="auto")
    if f_linear is not None:
        ax.axhline(f_linear, color="c", ls="--", lw=1.1, alpha=0.8,
                   label=r"linear limit ($w\to 0$)")
        ax.legend(fontsize=8, loc="upper right")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("frequency (Hz)")
    ax.set_title("Pitch glide over the ring-down (fundamental descends toward the linear limit)",
                 fontsize=9)


def save_membrane_animation(path, X: NDArray, Y: NDArray, snapshots, fs: float, mask=None) -> bool:
    """Write a GIF of the membrane vibrating (displacement heatmap over time).

    ``snapshots`` is the engine's ``[(step, field2d), ...]``. Returns False if no GIF writer is
    available. The seed of the later interactive web viewer — same precomputed-frames model.
    """
    from matplotlib import animation

    if not animation.writers.is_available("pillow"):
        return False

    vmax = max(float(np.nanmax(np.abs(s))) for _, s in snapshots) or 1.0
    fig, ax = plt.subplots(figsize=(5, 4.5))

    def draw(frame):
        ax.clear()
        step, field = snapshots[frame]
        f = np.where(mask, field, np.nan) if mask is not None else field
        ax.pcolormesh(X, Y, f, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
        ax.set_aspect("equal")
        ax.set_title(f"t = {1e3 * step / fs:6.2f} ms")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")

    anim = animation.FuncAnimation(fig, draw, frames=len(snapshots), interval=40)
    anim.save(path, writer="pillow", fps=25)
    plt.close(fig)
    return True


def save_displacement_animation(path, x: NDArray, snapshots, fs: float) -> bool:
    """Write a GIF of the string vibrating. Returns False if no GIF writer is available."""
    from matplotlib import animation

    if not animation.writers.is_available("pillow"):
        return False

    fig, ax = plt.subplots(figsize=(7, 3))
    ymax = max(np.max(np.abs(s)) for _, s in snapshots) * 1.1
    (line,) = ax.plot(x, snapshots[0][1], lw=1.5)
    ax.set_ylim(-ymax, ymax)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("u (m)")
    ax.grid(True, alpha=0.3)

    def update(frame):
        step, state = snapshots[frame]
        line.set_ydata(state)
        ax.set_title(f"t = {1e3 * step / fs:6.2f} ms")
        return (line,)

    anim = animation.FuncAnimation(fig, update, frames=len(snapshots), interval=40, blit=False)
    anim.save(path, writer="pillow", fps=25)
    plt.close(fig)
    return True
