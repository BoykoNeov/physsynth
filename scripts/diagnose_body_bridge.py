"""Diagnostics for the body/radiation node: a string terminated on a modal body through the
energy-conserving bridge connection (models: IdealString + ModalBody + StringBodyBridge).

Renders the diagnostics-as-visuals and prints the real validation numbers:

  1. **Energy exchange, flat total.** E_string, E_body and E_conn sloshing between each other while
     the total rides flat to machine precision — the visual proof that the *coupled* system is
     conservative (the string alone is not; it pumps energy into the body and back). The
     body/radiation analogue of the von Kármán energy-breakdown plot.
  2. **The body colours the sound.** Pickup spectrum of the bare string vs the radiated pressure
     ``p = sum_i a_i q_i''`` of the coupled body. The body re-radiates the string's partials through
     its own response: a high-frequency (acceleration, ``p ∝ omega^2``) emphasis, plus boosting
     where a string harmonic lands near a body mode. This is what a soundboard *does*.
  3. **The bridge sets the terminus character.** Sweep the bridge stiffness K; the string's
     fundamental climbs from the free-terminus quarter-wave (~c/4L) toward the clamped half-wave
     (~c/2L) as the spring stiffens the right end.

Run:  python scripts/diagnose_body_bridge.py
Figures land in ./out/.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from physsynth.analysis import spectrum  # noqa: E402
from physsynth.core.body import ModalBody  # noqa: E402
from physsynth.core.connection import StringBodyBridge  # noqa: E402
from physsynth.core.exciter import triangular_pluck  # noqa: E402
from physsynth.core.string_ideal import IdealString  # noqa: E402

# Canonical string (c = 200 m/s, f1 = 100 Hz) below lambda = 1 for coupling headroom; a modal body
# whose masses (~0.02 kg) are comparable to the string's rho*L = 0.005 kg so it genuinely reacts.
L, T, RHO, C = 1.0, 200.0, 0.005, 200.0
N, LAM = 100, 0.9
FS = C * N / (L * LAM)
BODY_FREQS = np.array([137.0, 213.0, 330.0, 471.0, 620.0])  # off-harmonic body resonances
BODY_MASS = 0.02
OUT = "out"


def build(K: float, sigma_body: float = 0.0) -> StringBodyBridge:
    s = IdealString(L=L, T=T, rho=RHO, fs=FS, N=N, boundary=("fixed", "free"))
    b = ModalBody(freqs=BODY_FREQS, fs=FS, masses=BODY_MASS, sigmas=sigma_body, phi=1.0)
    bridge = StringBodyBridge(string=s, body=b, K=K)
    s.set_state(triangular_pluck(s.x, L, 0.137 * L, amplitude=1e-3))
    return bridge


def run(bridge: StringBodyBridge, secs: float, pickup_frac: float = 0.23):
    """Step the coupled system, capturing per-component energy, a string pickup, and pressure."""
    s, b = bridge.string, bridge.body
    steps = int(secs * FS)
    idx = int(round(pickup_frac * N))
    e_str = np.empty(steps + 1)
    e_bod = np.empty(steps + 1)
    e_con = np.empty(steps + 1)
    pickup = np.empty(steps + 1)
    press = np.empty(steps + 1)

    def sample(i):
        e_str[i] = s.energy()
        e_bod[i] = b.energy()
        e_con[i] = bridge.energy() - e_str[i] - e_bod[i]
        pickup[i] = s.displacement_at(idx)
        press[i] = bridge.pressure()

    sample(0)
    for i in range(1, steps + 1):
        bridge.step()
        sample(i)
    total = e_str + e_bod + e_con
    time = np.arange(steps + 1) / FS
    return time, total, e_str, e_bod, e_con, pickup, press


def main() -> None:
    os.makedirs(OUT, exist_ok=True)

    # --- 1. energy exchange, flat total ------------------------------------------------------
    bridge = build(K=8000.0)
    time, total, e_str, e_bod, e_con, pickup, press = run(bridge, secs=1.0)
    drift = np.max(np.abs(total - total[0])) / total[0]
    body_frac = (e_bod / total).max()
    print("=== Body/bridge validation ==============================================")
    print(f"  total-energy drift (lossless) : {drift:.2e}   (< 1e-10 required)")
    print(f"  peak body-energy fraction     : {body_frac:.1%}   (energy really transfers)")
    print(f"  exact stability margin        : k^2 lambda_max = "
          f"{bridge.k ** 2 * bridge.spectral_radius:.4f}  (< 4)")

    e0 = total[0]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(time, total / e0, color="k", lw=1.4, label="total")
    ax.plot(time, e_str / e0, color="C0", lw=1.0, label="string")
    ax.plot(time, e_bod / e0, color="C3", lw=1.0, label="body")
    ax.plot(time, e_con / e0, color="C2", lw=0.8, alpha=0.8, label="connection")
    ax.set_xlabel("time (s)")
    ax.set_ylabel(r"$E / E^0$")
    ax.set_title(
        f"Bridge energy: flat total over string<->body exchange  —  drift = {drift:.2e}",
        fontsize=9,
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="center right")
    fig.tight_layout()
    fig.savefig(f"{OUT}/body_bridge_energy.png", dpi=130)
    plt.close(fig)

    # --- 2. the body colours the sound: string pickup vs radiated pressure --------------------
    # Light body damping so the pressure spectrum shows finite-width resonances (not line spectra).
    col = build(K=8000.0, sigma_body=6.0)
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
            label=r"radiated pressure $p=\sum a_i \ddot q_i$")
    for fb in BODY_FREQS:
        ax.axvline(fb, color="C3", ls=":", lw=0.7, alpha=0.6)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("normalised magnitude")
    ax.set_title("Radiated pressure: string partials, acceleration-tilted + boosted near body "
                 "modes (dotted)", fontsize=9)
    ax.set_xlim(0, fmax)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/body_bridge_spectrum.png", dpi=130)
    plt.close(fig)

    # --- 3. the bridge sets the terminus: fundamental free (c/4L) -> clamped (c/2L) vs K -------
    # The band spans the free quarter-wave (c/4L = 50 Hz) to just past the clamped half-wave
    # (c/2L = 100 Hz); the plucked fundamental is the lowest strong peak in it.
    Ks = np.array([200.0, 1000.0, 4000.0, 8000.0, 15000.0])
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
    fig.savefig(f"{OUT}/body_bridge_loading.png", dpi=130)
    plt.close(fig)

    print("=========================================================================")
    print(f"figures -> {OUT}/body_bridge_energy.png, _spectrum.png, _loading.png")


if __name__ == "__main__":
    main()
