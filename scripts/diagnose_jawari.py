"""Diagnostics for the **jawari / buzzing bridge** — the sitar & tanpura curved-bridge timbre.

The jawari is not a new model: it is model #8 (:class:`BarrierString`, string against a one-sided
distributed nonlinear barrier) driven by a *curved* barrier hugging the termination. The string
wraps onto the parabolic bridge each downswing and its departure point travels along the curve —
the shimmering, sustained, high-partial "life" of these instruments. Four figures land in ./out/:

  1. **The wrap (zoom on the bridge).** String snapshots near the termination with the parabolic
     bridge drawn: the string conforms to the curve over a span that grows and shrinks, its
     departure point sliding along the bridge — not a point slap on a flat fret.
  2. **Sustained brightness (shimmer).** Early- vs late-window pickup spectra for the jawari vs a
     clean string. The clean tone darkens (highs decay); the jawari's curved contact keeps
     re-injecting highs, so its late spectrum stays rich — the perceptual signature.
  3. **Travelling contact (precession).** The wrap-edge (furthest-in-contact node) over time, curve
     vs a flat rail at the same minimum clearance: on the curve it sweeps the whole bridge; on the
     rail it stays a pinned cluster.
  4. **Energy partition (conservation).** Lossless string + elastic bridge: total E is flat to
     machine precision while string energy and bridge PE trade off through the wrap.

Run:  python scripts/diagnose_jawari.py
Figures land in ./out/.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from physsynth.analysis import spectrum  # noqa: E402
from physsynth.core.collision import BarrierString, contact_potential  # noqa: E402
from physsynth.core.string_damped import DampedStiffString  # noqa: E402

L, T, RHO, N = 1.0, 200.0, 0.005, 100  # c = sqrt(T/rho) = 200 m/s, f1 = c/2L = 100 Hz
WIDTH_FRAC, DEPTH = 0.15, 1.0e-3       # bridge span (of L) and crest-to-far-edge drop (m)
AMP = 8.0e-3                            # first-mode pluck amplitude (m)
OUT = "out"


def jawari_profile(x, *, width_frac=WIDTH_FRAC, depth=DEPTH, clearance=0.0):
    """Parabolic bridge hugging x=0: ``b = -clearance - depth·(x/d)²`` on ``0<x≤d=width_frac·L``."""
    d = width_frac * L
    b = np.full_like(x, -np.inf)
    on = (x > 0.0) & (x <= d)
    b[on] = -clearance - depth * (x[on] / d) ** 2
    return b


def build(*, K=2.0e6, alpha=1.5, clearance=0.0, sigma0=0.0, sigma1=0.0, lam=0.4, flat=False):
    c = float(np.sqrt(T / RHO))
    fs = c * N / (L * lam)
    string = DampedStiffString(
        L=L, T=T, rho=RHO, fs=fs, N=N, kappa=0.0, sigma0=sigma0, sigma1=sigma1
    )
    b = jawari_profile(string.x, clearance=clearance)
    if flat:  # flatten the same support to a rail at the crest (matched minimum clearance)
        crest = float(np.max(b[np.isfinite(b)]))
        b = np.where(np.isfinite(b), crest, b)
    return BarrierString(string=string, barrier=b, stiffness=K, alpha=alpha)


def _pluck(bar, amplitude=AMP):
    bar.set_state(amplitude * np.sin(np.pi * bar.string.x / L))


def wrap_snapshots(steps=1400, n_snap=6):
    """String shapes near the termination at several instants, plus the bridge profile and grid."""
    bar = build(sigma0=0.2)
    _pluck(bar)
    x = bar.string.x
    take = np.linspace(steps // 5, steps - 1, n_snap).astype(int)
    snaps = []
    for i in range(steps):
        bar.step()
        if i in take:
            snaps.append((i * bar.k * 1e3, bar.string.u.copy()))
    bridge = np.full_like(x, np.nan)
    bridge[bar._support] = bar._b
    span = WIDTH_FRAC * L
    return x, bridge, snaps, span


def early_late_spectra(secs=0.35):
    """Early- and late-window pickup spectra for the jawari and a clean string."""
    out = {}
    for label, clearance in (("jawari", 0.0), ("clean", 1.0)):
        bar = build(sigma0=0.5, clearance=clearance)
        _pluck(bar)
        n = int(secs / bar.k)
        node = N // 2
        sig = np.empty(2 * n)
        for i in range(2 * n):
            bar.step()
            sig[i] = bar.string.displacement_at(node)
        fe, me, _ = spectrum.magnitude_spectrum(sig[:n], 1.0 / bar.k)
        fl, ml, _ = spectrum.magnitude_spectrum(sig[n:], 1.0 / bar.k)
        norm = me.max()
        out[label] = (fe, me / norm, fl, ml / norm)
    return out


def wrap_edge_traces(steps=9000):
    """Wrap-edge (furthest-in-contact node index) over time, for the curve and a flat rail."""
    traces = {}
    for label, flat in (("curved jawari bridge", False), ("flat rail (same clearance)", True)):
        bar = build(sigma0=0.5, flat=flat)
        _pluck(bar)
        we = np.full(steps, np.nan)
        for i in range(steps):
            bar.step()
            m = bar.contact_mask()
            if np.any(m):
                we[i] = float(np.max(np.where(m)[0]))
        t = np.arange(steps) * bar.k * 1e3
        traces[label] = (t, we, float(np.nanstd(we)))
    return traces


def energy_partition(steps=6000):
    """String / bridge-PE partition of the (conserved) total energy — lossless, elastic."""
    bar = build(sigma0=0.0)
    _pluck(bar)
    t = np.arange(steps + 1) * bar.k
    E = np.empty(steps + 1)
    e_str = np.empty(steps + 1)
    pe = np.empty(steps + 1)

    def sample(i):
        s = bar.string
        eta_n = bar._b - s.u[bar._support]
        eta_nm1 = bar._b - s.u_prev[bar._support]
        e_str[i] = s.energy()
        pe[i] = 0.5 * s.h * float(
            np.sum(contact_potential(eta_n, bar.K, bar.alpha))
            + np.sum(contact_potential(eta_nm1, bar.K, bar.alpha))
        )
        E[i] = bar.energy()

    sample(0)
    for i in range(1, steps + 1):
        bar.step()
        sample(i)
    return t * 1e3, E, e_str, pe


def main() -> None:
    os.makedirs(OUT, exist_ok=True)

    x, bridge, snaps, span = wrap_snapshots()
    spectra = early_late_spectra()
    traces = wrap_edge_traces()
    te, E, e_str, pe = energy_partition()
    drift = float(np.max(np.abs(E - E[0])) / abs(E[0]))

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # 1. the wrap, zoomed on the bridge
    ax = axes[0, 0]
    for tms, u in snaps:
        ax.plot(x, u * 1e3, lw=1.1, alpha=0.85, label=f"t={tms:.0f} ms")
    ax.plot(x, bridge * 1e3, "k--", lw=1.6, label="jawari bridge")
    ax.set_xlim(0.0, 1.6 * span)
    ax.set_ylim(-1.4 * DEPTH * 1e3, 1.4 * DEPTH * 1e3)
    ax.set_xlabel("position x (m)")
    ax.set_ylabel("displacement (mm)")
    ax.set_title("the wrap: string conforms to the curved bridge")
    ax.legend(fontsize=7, ncol=2)

    # 2. sustained brightness: early vs late spectra, jawari vs clean
    ax = axes[0, 1]
    colors = {"jawari": "C3", "clean": "C0"}
    for label, (fe, me, fl, ml) in spectra.items():
        sel = fe <= 4000.0
        ax.semilogy(fe[sel], me[sel], color=colors[label], lw=1.0, alpha=0.45,
                    label=f"{label} (early)")
        ax.semilogy(fl[sel], ml[sel], color=colors[label], lw=1.3,
                    label=f"{label} (late)")
    ax.set_ylim(1e-6, 2.0)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("|spectrum| (norm.)")
    ax.set_title("sustained shimmer: jawari keeps highs late; clean darkens")
    ax.legend(fontsize=7)

    # 3. travelling contact: wrap-edge over time, curve vs flat
    ax = axes[1, 0]
    for label, (t, we, std) in traces.items():
        ax.plot(t, we, ".", ms=1.5, alpha=0.5, label=f"{label} (std {std:.2f})")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("wrap-edge (support node index)")
    ax.set_title("travelling departure point: curve sweeps, rail stays pinned")
    ax.legend(fontsize=7, markerscale=4)

    # 4. energy partition
    ax = axes[1, 1]
    ax.plot(te, e_str, color="C0", label="string")
    ax.plot(te, pe, color="C1", label="bridge PE")
    ax.plot(te, E, "k--", lw=1.0, label="total E")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("energy (J)")
    ax.set_title(f"energy partition (conserved): drift {drift:.1e} rel.")
    ax.legend(fontsize=8)

    fig.suptitle("Jawari / buzzing bridge diagnostics (curved barrier on model #8)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    path = os.path.join(OUT, "jawari_diagnostics.png")
    fig.savefig(path, dpi=110)
    print(f"wrote {path}")
    print(f"energy drift (conserved run): {drift:.2e} relative")
    for label, (_, _, std) in traces.items():
        print(f"wrap-edge std, {label}: {std:.2f} nodes")


if __name__ == "__main__":
    main()
