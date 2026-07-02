"""Diagnostics for Step 5 of the body/radiation node: a string terminated on a **free-edge (FFFF)**
Kirchhoff plate (model #5b — a suspended cymbal/gong) through the energy-conserving bridge
(IdealString + Plate(boundary="free") + StringPlateBridge).

The Step-5 counterpart of ``diagnose_string_plate.py``. The body is now the free-edge plate: it has
a ``{1, x, y}`` rigid-body nullspace, so its driving point is coupled through the spring while the
un-driven zero-modes stay quiescent (no rigid drift). Three pictures:

  1. **Energy exchange, flat total.** E_string, E_plate and E_conn slosh between each other while
     the total rides flat to machine precision — the *coupled* free system is conservative just like
     the supported one; the free plate's diagonal mass ``W`` anchors the guard.
  2. **The cymbal colours the sound.** String pickup spectrum vs the radiated pressure
     ``p = sum_i W_ii u_i''`` (free-plate volume acceleration). The plate re-radiates the string's
     partials through its own free-edge (curved-Chladni) bending modes, marked as dotted lines — an
     acceleration tilt plus boosts where a string harmonic lands near a plate mode.
  3. **The bridge sets the terminus character — and the floating plate overshoots the clamp.**
     Sweep the bridge stiffness K; the string's fundamental climbs from the free-terminus
     quarter-wave (~c/4L=50 Hz) but *overshoots* the rigid-clamp half-wave (c/2L=100 Hz), landing
     near ~118 Hz. Unlike the pinned SS plate (Step 4, which lands cleanly near c/2L), a *floating*
     free plate below its first elastic mode (~569 Hz here) loads the string end as a reactive
     mass-spring — its {1,x,y} rigid-body modes act mass-like — not as a rigid anchor, so the
     terminus resonance is pushed above c/2L. (Argmax over 35–130 Hz tracks this fundamental; the
     pluck's stronger upper harmonics sit higher and are not what this panel measures.)

Run:  python scripts/diagnose_string_free_plate.py
Figures land in ./out/.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.sparse.linalg import eigsh  # noqa: E402

from physsynth.analysis import modal, spectrum  # noqa: E402
from physsynth.core.connection import StringPlateBridge  # noqa: E402
from physsynth.core.exciter import triangular_pluck  # noqa: E402
from physsynth.core.plate import Plate  # noqa: E402
from physsynth.core.string_ideal import IdealString  # noqa: E402

# Canonical string (c = 200 m/s, f1 = 100 Hz) below lambda = 1 for coupling headroom; a coarse free
# square plate whose per-node interior mass rho_s h^2 is comparable to the string's end half-cell,
# so it genuinely reacts. fs is shared (string-set); the plate's mu follows from N_PLATE. nu = 0.3
# re-enters for free edges (it drops out for simply-supported).
L, T, RHO, C = 1.0, 200.0, 0.005, 200.0
N, LAM = 100, 0.9
FS = C * N / (L * LAM)
KAPPA_PLATE, RHO_PLATE, N_PLATE, NU = 20.0, 0.005, 16, 0.3
OUT = "out"


def build(K: float, sigma_plate: float = 0.0) -> StringPlateBridge:
    s = IdealString(L=L, T=T, rho=RHO, fs=FS, N=N, boundary=("fixed", "free"))
    p = Plate(
        Lx=1.0, Ly=1.0, kappa=KAPPA_PLATE, rho=RHO_PLATE, fs=FS, N=N_PLATE,
        sigma=sigma_plate, boundary="free", nu=NU,
    )
    bridge = StringPlateBridge(string=s, plate=p, K=K)
    s.set_state(triangular_pluck(s.x, L, 0.137 * L, amplitude=1e-3))
    return bridge


def plate_mode_freqs(plate: Plate, n_modes: int = 8) -> np.ndarray:
    """Lowest *discrete* free-plate modal frequencies (Hz), honest markers on a coarse grid.

    The free plate is a generalized eigenproblem ``K phi = mu W phi`` (W the lumped-area mass); the
    three lowest eigenvalues are the ``{1, x, y}`` rigid-body zero-modes and are dropped.
    """
    vals = eigsh(
        plate.K, k=n_modes + 3, M=plate.W, sigma=-1e-6, which="LM", return_eigenvectors=False
    )
    elastic = np.sort(vals)[3:]  # drop the 3 rigid-body modes {1, x, y}
    return np.asarray(
        modal.discrete_plate_eigenfrequency(elastic, plate.kappa, plate.k, plate.theta)
    )


def run(bridge: StringPlateBridge, secs: float, pickup_frac: float = 0.23):
    """Step the coupled system, capturing per-component energy, a string pickup, and pressure."""
    s, p = bridge.string, bridge.plate
    steps = int(secs * FS)
    idx = int(round(pickup_frac * N))
    e_str = np.empty(steps + 1)
    e_pla = np.empty(steps + 1)
    e_con = np.empty(steps + 1)
    pickup = np.empty(steps + 1)
    press = np.empty(steps + 1)

    def sample(i):
        e_str[i] = s.energy()
        e_pla[i] = p.energy()
        e_con[i] = bridge.energy() - e_str[i] - e_pla[i]
        pickup[i] = s.displacement_at(idx)
        press[i] = bridge.pressure()

    sample(0)
    for i in range(1, steps + 1):
        bridge.step()
        sample(i)
    total = e_str + e_pla + e_con
    time = np.arange(steps + 1) / FS
    return time, total, e_str, e_pla, e_con, pickup, press


def main() -> None:
    os.makedirs(OUT, exist_ok=True)

    # --- 1. energy exchange, flat total ------------------------------------------------------
    bridge = build(K=6000.0)
    time, total, e_str, e_pla, e_con, pickup, press = run(bridge, secs=1.0)
    drift = np.max(np.abs(total - total[0])) / total[0]
    plate_frac = (e_pla / total).max()
    print("=== String -> FREE plate bridge validation (Step 5) =====================")
    print(f"  total-energy drift (lossless) : {drift:.2e}   (< 1e-10 required)")
    print(f"  peak plate-energy fraction    : {plate_frac:.1%}   (energy really transfers)")
    print(f"  exact stability margin        : {bridge.stability_margin:.4f}  (< 1)")
    print(f"  driving-point live index      : {bridge.drive_index}")

    e0 = total[0]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(time, total / e0, color="k", lw=1.4, label="total")
    ax.plot(time, e_str / e0, color="C0", lw=1.0, label="string")
    ax.plot(time, e_pla / e0, color="C3", lw=1.0, label="free plate")
    ax.plot(time, e_con / e0, color="C2", lw=0.8, alpha=0.8, label="connection")
    ax.set_xlabel("time (s)")
    ax.set_ylabel(r"$E / E^0$")
    ax.set_title(
        f"Free-plate bridge energy: flat total over string<->plate exchange — drift = {drift:.2e}",
        fontsize=9,
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="center right")
    fig.tight_layout()
    fig.savefig(f"{OUT}/string_free_plate_energy.png", dpi=130)
    plt.close(fig)

    # --- 2. the cymbal colours the sound: string pickup vs radiated pressure ------------------
    # Light plate damping so the pressure spectrum shows finite-width resonances (not line spectra).
    col = build(K=6000.0, sigma_plate=8.0)
    fmodes = plate_mode_freqs(col.plate, n_modes=8)
    _, _, _, _, _, pickup_c, press_c = run(col, secs=1.5)
    f_pick, m_pick, _ = spectrum.magnitude_spectrum(pickup_c, FS)
    f_pres, m_pres, _ = spectrum.magnitude_spectrum(press_c, FS)
    fmax = 1500.0
    fig, ax = plt.subplots(figsize=(9, 4))
    sel_p = f_pick <= fmax
    sel_r = f_pres <= fmax
    ax.plot(f_pick[sel_p], m_pick[sel_p] / m_pick[sel_p].max(), color="C0", lw=0.9,
            label="string pickup")
    ax.plot(f_pres[sel_r], m_pres[sel_r] / m_pres[sel_r].max(), color="C3", lw=0.9,
            label=r"radiated pressure $p=\sum W_{ii}\ddot u_i$")
    for fb in fmodes[fmodes <= fmax]:
        ax.axvline(fb, color="C3", ls=":", lw=0.7, alpha=0.6)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("normalised magnitude")
    ax.set_title("Radiated pressure: string partials, acceleration-tilted + boosted near free "
                 "plate modes (dotted)", fontsize=9)
    ax.set_xlim(0, fmax)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/string_free_plate_spectrum.png", dpi=130)
    plt.close(fig)

    # --- 3. the bridge sets the terminus: fundamental free (c/4L) -> clamped (c/2L) vs K -------
    Ks = np.array([200.0, 1000.0, 3000.0, 6000.0, 10000.0])
    f1s = []
    for K in Ks:
        br = build(K=K)
        _, _, _, _, _, pick, _ = run(br, secs=1.5)
        f_axis, mag, _ = spectrum.magnitude_spectrum(pick, FS)
        band = (f_axis > 35) & (f_axis < 130)
        f1s.append(f_axis[band][np.argmax(mag[band])])
    f1s = np.array(f1s)
    print(f"  fundamental vs K (Hz)         : "
          f"{', '.join(f'{k:.0f}->{f:.2f}' for k, f in zip(Ks, f1s, strict=True))}")
    print("    (rises from free quarter-wave c/4L = 50 Hz and OVERSHOOTS rigid-clamp c/2L = 100 Hz")
    print("     -- the floating free plate loads the end as a reactive mass-spring, not a rigid "
          "anchor)")
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.semilogx(Ks, f1s, "o-", color="C4")
    ax.axhline(C / (4 * L), color="C7", ls=":", lw=0.8, label="free  c/4L = 50 Hz")
    ax.axhline(C / (2 * L), color="C7", ls="--", lw=0.8, label="rigid clamp  c/2L = 100 Hz")
    ax.set_xlabel("bridge stiffness K (N/m)")
    ax.set_ylabel("measured fundamental (Hz)")
    ax.legend(fontsize=8)
    ax.set_title("Bridge stiffness raises the terminus above free (c/4L); floating plate "
                 "overshoots rigid clamp (c/2L)", fontsize=9)
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(f"{OUT}/string_free_plate_loading.png", dpi=130)
    plt.close(fig)

    print("=========================================================================")
    print(f"figures -> {OUT}/string_free_plate_energy.png, _spectrum.png, _loading.png")


if __name__ == "__main__":
    main()
