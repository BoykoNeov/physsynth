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
    "plot_displacement_snapshots",
    "plot_convergence",
    "plot_spectrum",
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
