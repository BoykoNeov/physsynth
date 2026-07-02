"""Diagnostics for Step 4 of the body/radiation node: a string terminated on a **grid** Kirchhoff
plate (model #5) through the energy-conserving bridge (IdealString + Plate + StringPlateBridge).

The plate is the first *distributed* body — the string couples to it at a single driving-point
node, so the diagnostics parallel the modal-body script but the body is now a 2D field:

  1. **Energy exchange, flat total.** E_string, E_plate and E_conn sloshing between each other while
     the total rides flat to machine precision — the visual proof the *coupled* system is
     conservative (the string alone pumps energy into the plate and back). Same picture as the modal
     bridge, one distributed body up.
  2. **The plate colours the sound.** String pickup spectrum vs the radiated pressure
     ``p = h^2 sum u_ij''`` (plate volume acceleration). The plate re-radiates the string's partials
     through its own bending modes: an acceleration (``p ∝ omega^2``) tilt, plus boosting where a
     string harmonic lands near a plate mode. The plate's discrete modal frequencies are marked.
  3. **The bridge sets the terminus character.** Sweep the bridge stiffness K; the string's
     fundamental climbs from the free-terminus quarter-wave (~c/4L) toward the clamped half-wave
     (~c/2L) as the spring stiffens the right end.

Run:  python scripts/diagnose_string_plate.py
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

# Canonical string (c = 200 m/s, f1 = 100 Hz) below lambda = 1 for coupling headroom; a coarse
# simply-supported plate whose per-node mass rho_s h^2 is comparable to the string's end half-cell,
# so it genuinely reacts. fs is shared (string-set); the plate's mu follows from N_plate.
L, T, RHO, C = 1.0, 200.0, 0.005, 200.0
N, LAM = 100, 0.9
FS = C * N / (L * LAM)
KAPPA_PLATE, RHO_PLATE, N_PLATE = 20.0, 0.005, 16
OUT = "out"


def build(K: float, sigma_plate: float = 0.0) -> StringPlateBridge:
    s = IdealString(L=L, T=T, rho=RHO, fs=FS, N=N, boundary=("fixed", "free"))
    p = Plate(Lx=1.0, Ly=1.0, kappa=KAPPA_PLATE, rho=RHO_PLATE, fs=FS, N=N_PLATE, sigma=sigma_plate)
    bridge = StringPlateBridge(string=s, plate=p, K=K)
    s.set_state(triangular_pluck(s.x, L, 0.137 * L, amplitude=1e-3))
    return bridge


def plate_mode_freqs(plate: Plate, n_modes: int = 8) -> np.ndarray:
    """The plate's lowest *discrete* modal frequencies (Hz) — honest markers on a coarse grid."""
    lam_vals = eigsh(-plate.L, k=n_modes, sigma=0.0, which="LM", return_eigenvectors=False)
    lam_vals = np.sort(lam_vals)
    return np.asarray(
        modal.discrete_plate_eigenfrequency(lam_vals, plate.kappa, plate.k, plate.theta)
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
    print("=== String -> plate bridge validation (Step 4) ==========================")
    print(f"  total-energy drift (lossless) : {drift:.2e}   (< 1e-10 required)")
    print(f"  peak plate-energy fraction    : {plate_frac:.1%}   (energy really transfers)")
    print(f"  exact stability margin        : {bridge.stability_margin:.4f}  (< 1)")
    print(f"  driving-point live index      : {bridge.drive_index}")

    e0 = total[0]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(time, total / e0, color="k", lw=1.4, label="total")
    ax.plot(time, e_str / e0, color="C0", lw=1.0, label="string")
    ax.plot(time, e_pla / e0, color="C3", lw=1.0, label="plate")
    ax.plot(time, e_con / e0, color="C2", lw=0.8, alpha=0.8, label="connection")
    ax.set_xlabel("time (s)")
    ax.set_ylabel(r"$E / E^0$")
    ax.set_title(
        f"Plate bridge energy: flat total over string<->plate exchange  —  drift = {drift:.2e}",
        fontsize=9,
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="center right")
    fig.tight_layout()
    fig.savefig(f"{OUT}/string_plate_energy.png", dpi=130)
    plt.close(fig)

    # --- 2. the plate colours the sound: string pickup vs radiated pressure -------------------
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
            label=r"radiated pressure $p=h^2\sum \ddot u_{ij}$")
    for fb in fmodes[fmodes <= fmax]:
        ax.axvline(fb, color="C3", ls=":", lw=0.7, alpha=0.6)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("normalised magnitude")
    ax.set_title("Radiated pressure: string partials, acceleration-tilted + boosted near plate "
                 "modes (dotted)", fontsize=9)
    ax.set_xlim(0, fmax)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/string_plate_spectrum.png", dpi=130)
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
    print("    (free quarter-wave c/4L = 50 Hz  ->  clamped half-wave c/2L = 100 Hz)")
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.semilogx(Ks, f1s, "o-", color="C4")
    ax.axhline(C / (4 * L), color="C7", ls=":", lw=0.8, label="free  c/4L = 50 Hz")
    ax.axhline(C / (2 * L), color="C7", ls="--", lw=0.8, label="clamped  c/2L = 100 Hz")
    ax.set_xlabel("bridge stiffness K (N/m)")
    ax.set_ylabel("measured fundamental (Hz)")
    ax.legend(fontsize=8)
    ax.set_title("Bridge stiffness moves the terminus from free (c/4L) toward clamped (c/2L)",
                 fontsize=9)
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(f"{OUT}/string_plate_loading.png", dpi=130)
    plt.close(fig)

    print("=========================================================================")
    print(f"figures -> {OUT}/string_plate_energy.png, _spectrum.png, _loading.png")


if __name__ == "__main__":
    main()
