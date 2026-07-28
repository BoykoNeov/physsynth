"""Diagnostics for the 3-D air box: :class:`physsynth.core.airbox.AirBox` (HANDOFF §12.H).

The distributed tier of the air node. :mod:`physsynth.core.radiation` models the air as a lumped
port — one terminal, one impedance, one distance — which is enough for "how loud, how damped" and
structurally cannot represent anything that depends on *where you are*. These figures are the three
things that changes.

  1. **The room's own resonances.** A rigid box has closed-form modes twice over: the textbook
     ``f_lmn = (c0/2) sqrt((l/Lx)^2 + (m/Ly)^2 + (n/Lz)^2)``, and — rarer, and the stronger oracle —
     the **exact discrete** ``omega_d = (2/k) arcsin(c0 k mu / 2)`` that the scheme itself will
     produce. The measured spectrum of a probe is drawn against both, so the visible gap between the
     two curves *is* the numerical dispersion, not an error. The right-hand panel resolves the
     dispersion by direction, which is where 3-D differs from 1-D: at ``lambda = 1/sqrt(3)`` a mode
     along the grid **diagonal** is exact to machine precision while an axial mode is not — and no
     other ``lambda`` is exact anywhere.
  2. **The three-channel energy ledger.** Stored, absorbed by the walls, injected by the source, and
     their flat total. The wall channel is the passivity proof: the box can shed 98% of its energy
     to an absorbing wall and the total still sits at the roundoff floor. Note that the *stored*
     curve is what falls — asserting conservation on it would assert the physics away.
  3. **The point of the batch: the distributed air contains the lumped air.** Drive a point source
     in a big room, and inside the window before the first reflection arrives the field reproduces
     batch 1's monopole law ``p = rho0 Q'' / (4 pi r)`` at every radius. The window is drawn,
     because it is the whole measurement: a rigid wall is a pressure **antinode**, so the reflection
     arrives at full amplitude with no roll-off, and reading past it turns ``1/r`` into ``1/r^2.5``.

Run:  python scripts/diagnose_airbox.py
Figures land in ./out/.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from physsynth.core.airbox import C0_AIR, RHO0_AIR, AirBox  # noqa: E402
from physsynth.core.radiation import AirRadiation  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out")
os.makedirs(OUT, exist_ok=True)

ROOM = (2.4, 1.8, 1.5)  # m — a small, ordinary room
H_ROOM = 0.05           # m
CFL = 0.9               # fraction of the 3-D ceiling 1/sqrt(3)


def build(L=ROOM, h=H_ROOM, cfl=CFL, **kw) -> AirBox:
    """An :class:`AirBox` at a chosen fraction of the 3-D CFL ceiling (``fs`` solved from ``h``)."""
    return AirBox(L=L, fs=C0_AIR * np.sqrt(3.0) / (cfl * h), h=h, **kw)


# -- 1. room modes -------------------------------------------------------------------------------

def _freq_error(box: AirBox, *lmn: int) -> float:
    """|discrete / continuum - 1| for one mode — the scheme's dispersion, made a number."""
    return abs(box.mode_frequency(*lmn) / box.continuum_mode_frequency(*lmn) - 1.0)


def fig_room_modes():
    box = build()
    rng = np.random.default_rng(7)
    box.set_state(rng.standard_normal(box.p.shape))
    steps = 24000
    probe = (0.37, 0.29, 0.23)  # off every symmetry plane, so nothing is accidentally silent
    trace = np.empty(steps)
    for n in range(steps):
        box.step()
        trace[n] = box.pressure_at(probe)

    spec = np.abs(np.fft.rfft(trace * np.hanning(steps)))
    freq = np.fft.rfftfreq(steps, box.k)
    f_hi = 160.0
    band = (freq > 20.0) & (freq < f_hi)  # DC is a genuine (stationary) mode; it just plots badly

    modes = [(l, m, n) for l in range(4) for m in range(3) for n in range(3)]
    modes = [q for q in modes if any(q) and box.continuum_mode_frequency(*q) < f_hi]
    f_disc = np.array([box.mode_frequency(*q) for q in modes])
    f_cont = np.array([box.continuum_mode_frequency(*q) for q in modes])

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.5, 4.6))
    ax.semilogy(freq[band], spec[band] / spec[band].max(), lw=1.0, color="0.25",
                label="measured probe spectrum")
    ax.plot(f_disc, np.full(f_disc.size, 1.6), "|", ms=13, mew=1.6, color="tab:blue",
            label="exact discrete $f_{lmn}$")
    ax.plot(f_cont, np.full(f_cont.size, 2.6), "|", ms=13, mew=1.4, color="tab:red",
            label="continuum $f_{lmn}$ (the gap is dispersion)")
    for fd in f_disc:
        ax.axvline(fd, color="tab:blue", lw=0.7, alpha=0.25)
    for q, fd in zip(modes, f_disc, strict=True):
        if fd < 105.0:
            ax.annotate(f"{q[0]}{q[1]}{q[2]}", (fd, 3.6), fontsize=7, ha="center",
                        color="0.35")
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("normalized magnitude")
    ax.set_title(f"room modes of a {ROOM[0]}x{ROOM[1]}x{ROOM[2]} m box (h = {H_ROOM * 100:.0f} cm)")
    ax.set_ylim(1e-4, 6.0)
    ax.set_xlim(20.0, f_hi)
    ax.legend(fontsize=8, loc="lower right")

    # Dispersion by direction, at and below the ceiling. The diagonal is exact only AT the ceiling.
    cube = 1.6
    for cfl, style in ((CFL, "-"), (1.0, "--")):
        b = build(L=(cube, cube, cube), h=0.05, cfl=cfl)
        qs = np.arange(1, 17)
        axial = [_freq_error(b, q, 0, 0) for q in qs]
        diag = [_freq_error(b, q, q, q) for q in qs]
        lam = f"$\\lambda$ = {b.lam:.3f}"
        ax2.semilogy(qs, np.maximum(axial, 1e-17), style, color="tab:red", label=f"axial, {lam}")
        ax2.semilogy(qs, np.maximum(diag, 1e-17), style, color="tab:blue",
                     label=f"diagonal, {lam}")
    ax2.set_xlabel("mode index $q$")
    ax2.set_ylabel("|relative frequency error|")
    ax2.set_title("3-D has no dispersionless $\\lambda$ — only an exact *direction*")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3, which="both")

    fig.tight_layout()
    path = os.path.join(OUT, "airbox_room_modes.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)

    peaks = [
        freq[i] for i in range(2, band.sum() - 2)
        if spec[i] > spec[i - 1] and spec[i] > spec[i + 1] and spec[i] > 0.02 * spec[band].max()
    ]
    matched = [min(abs(p - f_disc)) for p in peaks[:8]]
    return path, float(np.max(matched)) if matched else float("nan"), freq[1] - freq[0]


# -- 2. the three-channel energy ledger ------------------------------------------------------------

def fig_energy_channels():
    # One weakly-absorbing wall (zeta = 12, a hard plastered surface rather than an anechoic wedge),
    # so the room rings for a visible while instead of draining inside the first few milliseconds.
    box = build(h=0.1, walls={"x1": 12.0 * RHO0_AIR * C0_AIR})
    box.set_state(np.zeros(box.p.shape))
    steps = 3000
    f0, amp = 220.0, 5e-3
    sig, t0 = 1.0 / (2.0 * np.pi * f0), 4.0 / (2.0 * np.pi * f0)

    stored = np.empty(steps)
    lost = np.empty(steps)
    given = np.empty(steps)
    for n in range(steps):
        t = n * box.k
        box.inject(amp * np.exp(-((t - t0) ** 2) / (2.0 * sig * sig)))
        box.step()
        stored[n] = box.acoustic_energy()
        lost[n] = box.dissipated_energy()
        given[n] = box.injected_energy()

    t = np.arange(steps) * box.k * 1e3
    total = stored + lost - given
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(9.5, 6.4), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1]})
    ax.plot(t, given, color="tab:green", label="injected by the source")
    ax.plot(t, stored, color="tab:blue", label="stored in the air")
    ax.plot(t, lost, color="tab:red", label="absorbed by the wall")
    ax.set_ylabel("energy (J)")
    ax.set_title("three channels: the walls only take, the source only gives, the total is flat")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax2.plot(t, total / max(given[-1], 1e-300), color="0.3", lw=0.9)
    ax2.set_xlabel("time (ms)")
    ax2.set_ylabel("residual / injected")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(OUT, "airbox_energy_channels.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path, float(np.max(np.abs(total)) / given[-1]), float(lost[-1] / given[-1])


# -- 3. the headline: free field = the lumped monopole ---------------------------------------------

def fig_free_field():
    L, N, f0 = 1.0, 64, 1400.0
    box = build(L=(L, L, L), h=L / N)
    sig, t0, amp = 1.0 / (2.0 * np.pi * f0), 4.0 / (2.0 * np.pi * f0), 1e-3
    pulse_seconds = 8.0 * sig

    def q(t):
        return amp * np.exp(-((t - t0) ** 2) / (2.0 * sig * sig))

    def qdot(t):
        return -amp * (t - t0) / (sig * sig) * np.exp(-((t - t0) ** 2) / (2.0 * sig * sig))

    centre = tuple(0.5 * v for v in box.L_actual)
    box.source_index = box.node_index(centre)
    x0, y0, z0 = box.snapped(centre)
    r_max = 0.5 * (L - C0_AIR * pulse_seconds)
    cells = np.linspace(max(6, int(0.3 * r_max / box.h)), int(r_max / box.h), 6).astype(int)
    cells = np.unique(cells)
    radii = cells * box.h

    steps = int(np.floor((L - radii.min()) / C0_AIR / box.k))
    rec = np.zeros((steps, cells.size))
    box.set_state(np.zeros(box.p.shape))
    for n in range(steps):
        box.inject(q(n * box.k))
        box.step()
        for j, c in enumerate(cells):
            rec[n, j] = box.pressure_at((x0 + c * box.h, y0, z0))

    t = np.arange(steps) * box.k
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.5, 4.8))
    gains = []
    for j, r in enumerate(radii):
        window = t < (L - r) / C0_AIR
        gain_far = AirRadiation(fs=box.fs, distance=float(r)).gain
        ref = gain_far * qdot(t - r / C0_AIR)
        g = float(np.dot(rec[window, j], ref[window]) / np.dot(ref[window], ref[window]))
        gains.append(g)
        colour = plt.cm.viridis(j / max(len(radii) - 1, 1))
        ax.plot(t * 1e3, rec[:, j], color=colour, lw=1.0, label=f"r = {r * 100:.0f} cm")
        ax.plot(t * 1e3, ref, color=colour, lw=2.4, alpha=0.3)
        ax.axvline((L - r) / C0_AIR * 1e3, color=colour, ls=":", lw=0.8)
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("pressure (Pa)")
    ax.set_title("thin: the room · thick pale: batch 1's monopole law · dotted: reflection arrives")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)

    # The claim itself. Because the reference already carries rho0/(4 pi r), a fitted gain of 1 IS
    # the inverse-distance law -- so this panel plots the gain, not a slope. The naive peak-ratio
    # estimator is drawn beside it to show why: a dispersive pulse's peak is not its amplitude, so
    # peak-ratio drifts several percent while the fit stays flat.
    peak = np.array([np.max(np.abs(rec[t < (L - r) / C0_AIR, j])) for j, r in enumerate(radii)])
    naive = peak * radii / (peak[0] * radii[0])
    ax2.axhline(1.0, color="0.4", lw=1.0, ls="--")
    ax2.plot(radii * 100, gains, "o-", color="tab:blue", label="least-squares gain vs batch 1")
    ax2.plot(radii * 100, naive, "s--", color="tab:red", alpha=0.7,
             label="naive peak-ratio estimator")
    ax2.set_xlabel("radius (cm)")
    ax2.set_ylabel("measured / monopole law")
    worst = np.max(np.abs(np.array(gains) - 1.0))
    ax2.set_title(f"gain == 1 IS the $1/r$ law:  max|gain - 1| = {worst:.2e}")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    path = os.path.join(OUT, "airbox_free_field.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path, radii, np.array(gains)


def main() -> None:
    box = build()
    print(f"room {ROOM} m at h = {H_ROOM * 100:.0f} cm -> N = {box.N}, "
          f"{np.prod([n + 1 for n in box.N])} nodes, fs = {box.fs:.0f} Hz, lambda = {box.lam:.4f}")
    actual = tuple(round(v, 4) for v in box.L_actual)
    print(f"  grid snap: requested {box.L} m -> actual {actual} m")

    p1, worst, binw = fig_room_modes()
    print(f"room modes              -> {p1}")
    print(f"  worst measured peak vs exact discrete f_lmn : {worst:.2f} Hz "
          f"(FFT bin width {binw:.2f} Hz)")

    p2, resid, absorbed = fig_energy_channels()
    print(f"energy channels         -> {p2}")
    print(f"  |stored + absorbed - injected| / injected   : {resid:.3e}")
    print(f"  fraction of the injected energy the walls took: {absorbed * 100:.1f}%")

    p3, radii, gains = fig_free_field()
    print(f"free field vs batch 1   -> {p3}")
    for r, g in zip(radii, gains, strict=True):
        print(f"  r = {r * 100:5.1f} cm -> fitted gain {g:.5f}  (1.0 IS the 1/r law)")
    print(f"  max|gain - 1| = {np.max(np.abs(gains - 1.0)):.3e}")


if __name__ == "__main__":
    main()
