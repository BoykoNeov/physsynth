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
