"""Diagnostics for the **frequency-dependent** radiation load (batch 3):
:class:`physsynth.core.radiation.RationalAirLoad` / :class:`ReactiveRadiatedBody`.

Batch 2 loaded the body with one constant resistance ``R``; real air is frequency-dependent
(``R_a ∝ omega^2`` at low ``ka``), so a constant over-damps the low modes and under-damps the high
ones. The fix needs no filter approximation: the exact acoustic impedance of a pulsating sphere is
already first-order rational,

    Z_a(j omega) = R j omega tau / (1 + j omega tau) ,   R = rho0 c0 / S ,  M_a = rho0/(4 pi a) ,

i.e. a resistance in **parallel** with the radiation mass. The volume velocity splits: the resistor
branch is radiated (gone), the inertance branch is stored (returned) — which is why the frequency
dependence is passive by construction, and why the energy identity gains a *stored* term.

The figures:

  1. **The impedance itself.** The measured sweep (drive the load standalone at a prescribed volume
     velocity, read ``p / U``) against the closed form, magnitude and phase. Two closed forms are
     drawn: the **pre-warped** one the trapezoid actually realises (``s = (2j/k) tan(omega k/2)``),
     which the measurement matches to machine precision, and the continuous ``Z_a(j omega)`` — the
     gap between them is the honest discretisation error, ``O((omega k)^2)``. Both closed-form
     limits are marked: the ``rho0 omega^2/(4 pi c0)`` monopole rise (batch 2's own helper — this
     impedance *contains* it) and the ``rho0 c0/S`` plane-wave saturation.
  2. **The three-way energy channel.** Body energy, stored radiation-mass energy, radiated energy,
     and their flat total. The stored curve both rises and falls — batch 2's air could only take;
     this one gives back — and the residual sits at the roundoff floor.
  3. **The point of the batch: spectral shaping.** Per-mode decay rates of a weakly loaded body,
     measured against the closed form ``alpha = a^2 Re Z_a / (2 m_eff)`` — where the *reactance*
     supplies the added mass ``m_add = a^2 Im Z_a / omega`` that also drops the pitch — with a
     constant-``R`` load matched at the fundamental as the negative control. The control is flat, so
     it cannot make high partials die first no matter how it is tuned, and having no reactance it
     cannot flatten the pitch at all.

Run:  python scripts/diagnose_radiation_impedance.py
Figures land in ./out/.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from physsynth.core.body import ModalBody  # noqa: E402
from physsynth.core.radiation import (  # noqa: E402
    C0_AIR,
    RHO0_AIR,
    RationalAirLoad,
    ReactiveRadiatedBody,
    monopole_radiation_resistance,
)

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out")
os.makedirs(OUT, exist_ok=True)

FS = 48000.0
SPHERE_A = 0.05                                      # m — a 5 cm pulsating sphere
BODY_FREQS = np.array([110.0, 196.0, 261.0, 440.0])  # Hz
BODY_MASS = 0.02
RAD_WEIGHT = 0.02                                    # weak coupling: alpha_i << omega_i


def measure_impedance(load: RationalAirLoad, omega: float, n_window: int, warmup: int) -> complex:
    """Drive the load standalone at prescribed volume velocity (``G = 0``) and read ``p / U``."""
    k, n = load.k, 0
    for _ in range(warmup):
        load.step(np.cos(omega * n * k))
        n += 1
    u = np.empty(n_window)
    p = np.empty(n_window)
    for i in range(n_window):
        u[i] = np.cos(omega * n * k)
        p[i], _ = load.step(u[i])
        n += 1
    phase = np.exp(-1j * omega * k * np.arange(n_window))
    return complex(np.sum(p * phase) / np.sum(u * phase))


def fig_impedance():
    n_window = 4096
    bins = np.unique(np.round(np.geomspace(4, 2000, 40)).astype(int))
    freqs = FS * bins / n_window
    omega = 2.0 * np.pi * freqs

    measured = np.empty(freqs.size, dtype=complex)
    for i, w in enumerate(omega):
        load = RationalAirLoad.from_sphere(fs=FS, radius=SPHERE_A)
        measured[i] = measure_impedance(load, w, n_window, warmup=8000)

    ref = RationalAirLoad.from_sphere(fs=FS, radius=SPHERE_A)
    z_warp = np.array([ref.impedance_discrete(w) for w in omega])
    z_cont = np.array([ref.impedance(w) for w in omega])
    r_mono = np.array([monopole_radiation_resistance(w) for w in omega])
    plane_wave = RHO0_AIR * C0_AIR / (4.0 * np.pi * SPHERE_A * SPHERE_A)

    fig, (ax, axp) = plt.subplots(2, 1, figsize=(9, 7.0), sharex=True)
    ax.loglog(freqs, np.abs(z_cont), lw=1.2, color="tab:blue", label="closed form |Z_a(jω)|")
    ax.loglog(freqs, np.abs(z_warp), lw=1.0, ls="--", color="tab:green",
              label="pre-warped (what the scheme realises)")
    ax.loglog(freqs, np.abs(measured), ls="none", marker="o", ms=3.5, mfc="none",
              color="tab:red", label="measured  p / U")
    ax.loglog(freqs, r_mono, lw=0.8, ls=":", color="tab:gray",
              label="ρ₀ω²/(4πc₀)  — batch 2's helper (ka→0)")
    ax.axhline(plane_wave, lw=0.8, ls=":", color="tab:purple",
               label="ρ₀c₀/S — plane-wave saturation")
    ax.axvline(C0_AIR / (2.0 * np.pi * SPHERE_A), lw=0.8, color="k", alpha=0.3)
    ax.text(C0_AIR / (2.0 * np.pi * SPHERE_A) * 1.05, plane_wave * 0.05, "ka = 1", fontsize=8,
            alpha=0.6)
    ax.set_ylabel("|Z_a|  (Pa·s/m³)")
    ax.set_title(f"Rational radiation impedance — pulsating sphere a = {SPHERE_A * 100:.0f} cm")
    ax.legend(fontsize=8, loc="lower right")

    axp.semilogx(freqs, np.degrees(np.angle(z_cont)), lw=1.2, color="tab:blue")
    axp.semilogx(freqs, np.degrees(np.angle(z_warp)), lw=1.0, ls="--", color="tab:green")
    axp.semilogx(freqs, np.degrees(np.angle(measured)), ls="none", marker="o", ms=3.5,
                 mfc="none", color="tab:red")
    axp.set_xlabel("frequency (Hz)")
    axp.set_ylabel("∠Z_a  (deg)")
    axp.set_ylim(-5, 95)

    err_warp = np.max(np.abs(measured - z_warp) / np.abs(z_warp))
    err_cont = np.max(np.abs(z_warp - z_cont) / np.abs(z_cont))
    axp.text(
        0.02, 0.06,
        f"measured vs pre-warped: max rel err {err_warp:.1e}   (machine precision)\n"
        f"pre-warped vs continuous: max rel err {err_cont:.1e}   (the O((ωk)²) warp — real)",
        transform=axp.transAxes, fontsize=8, va="bottom",
    )

    fig.tight_layout()
    path = os.path.join(OUT, "radiation_impedance_sweep.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path, err_warp, err_cont


def fig_energy_channels():
    load = RationalAirLoad.from_sphere(fs=FS, radius=SPHERE_A)
    loaded = ReactiveRadiatedBody(
        body=ModalBody(freqs=BODY_FREQS, fs=FS, sigmas=0.0, masses=BODY_MASS), load=load
    )
    loaded.set_state(np.array([1e-3, -8e-4, 6e-4, 4e-4]))
    steps = 6000
    e0 = loaded.energy()
    body_e = np.empty(steps)
    stored = np.empty(steps)
    rad = np.empty(steps)
    total = np.empty(steps)
    for i in range(steps):
        loaded.step()
        body_e[i] = loaded.body.energy()
        stored[i] = loaded.load.stored_energy()
        rad[i] = loaded.radiated_energy
        total[i] = loaded.energy()
    t_ms = np.arange(steps) / FS * 1e3

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(t_ms, body_e / e0, lw=1.0, color="tab:blue", label="body  E_body")
    ax.plot(t_ms, stored / e0, lw=1.0, color="tab:purple",
            label="stored in the radiation mass  ½M_a U_L²  (rises AND falls)")
    ax.plot(t_ms, rad / e0, lw=1.0, color="tab:orange", label="radiated  ∫R U_R² dt")
    ax.plot(t_ms, total / e0, lw=1.2, color="tab:green", label="total (flat)")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("energy / E₀")
    ax.set_title("Frequency-dependent load: the air now stores as well as dissipates")
    ax.legend(loc="center right", fontsize=8)

    ax2 = ax.twinx()
    ax2.semilogy(t_ms, np.abs(total - e0) / e0 + 1e-18, lw=0.6, color="tab:red", alpha=0.5)
    ax2.set_ylabel("|total − E₀| / E₀  (red)", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax2.set_ylim(1e-16, 1e-9)

    fig.tight_layout()
    path = os.path.join(OUT, "radiation_impedance_energy_channels.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path, float(np.max(np.abs(total - e0) / e0))


def _measure_single_mode(f0: float, load: RationalAirLoad, seconds: float = 0.6):
    """Excite one mode of a loaded body; return its measured frequency and decay rate."""
    b = ModalBody(freqs=np.array([f0]), fs=FS, sigmas=0.0, masses=BODY_MASS, radiation=RAD_WEIGHT)
    loaded = ReactiveRadiatedBody(body=b, load=load)
    loaded.set_state(np.array([1e-3]))
    steps = int(seconds * FS)
    q = np.empty(steps)
    for n in range(steps):
        loaded.step()
        q[n] = b.q[0]
    sign = np.signbit(q)
    crossings = np.flatnonzero(sign[:-1] != sign[1:])
    f_meas = 0.5 * FS * (crossings.size - 1) / (crossings[-1] - crossings[0])
    env = np.abs(q)
    peaks = np.flatnonzero((env[1:-1] > env[:-2]) & (env[1:-1] >= env[2:])) + 1
    t = np.arange(steps) / FS
    return f_meas, -float(np.polyfit(t[peaks], np.log(env[peaks]), 1)[0])


def fig_spectral_shaping():
    sphere = RationalAirLoad.from_sphere(fs=FS, radius=SPHERE_A)
    r_const = sphere.impedance(2.0 * np.pi * BODY_FREQS[0]).real     # matched at the fundamental
    flat = RationalAirLoad(fs=FS, R=r_const, M_a=np.inf)             # = batch 2, bit for bit

    measured = [_measure_single_mode(f, sphere) for f in BODY_FREQS]
    alpha_reactive = np.array([m[1] for m in measured])
    f_reactive = np.array([m[0] for m in measured])
    alpha_constant = np.array([_measure_single_mode(f, flat)[1] for f in BODY_FREQS])
    predicted = np.array([
        sphere.loaded_mode(2.0 * np.pi * f, weight=RAD_WEIGHT, mass=BODY_MASS)[1]
        for f in BODY_FREQS
    ])

    fig, ax = plt.subplots(figsize=(9, 4.8))
    f_dense = np.geomspace(60.0, 1200.0, 300)
    ax.plot(
        f_dense,
        [sphere.loaded_mode(2.0 * np.pi * f, weight=RAD_WEIGHT, mass=BODY_MASS)[1]
         for f in f_dense],
        lw=1.0, color="tab:blue", label="closed form  a²·Re Z_a/(2 m_eff)",
    )
    ax.plot(f_reactive, alpha_reactive, ls="none", marker="o", ms=7, mfc="none", color="tab:red",
            label="measured, rational load")
    ax.plot(BODY_FREQS, alpha_constant, ls="none", marker="s", ms=7, mfc="none",
            color="tab:gray", label="measured, constant R (matched at the fundamental)")
    ax.axhline(alpha_constant[0], lw=0.8, ls=":", color="tab:gray")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("mode frequency (Hz)")
    ax.set_ylabel("decay rate α (s⁻¹)")
    ax.set_title("Why the reactance matters: high partials radiate better, so they die first")
    ax.legend(fontsize=8, loc="upper left")
    worst = float(np.max(np.abs(alpha_reactive / predicted - 1.0)))
    ax.text(
        0.98, 0.05,
        f"rational load: max |α_meas/α_pred − 1| = {worst:.1%}\n"
        f"constant R: flat — top mode under-damped {predicted[-1] / alpha_constant[-1]:.1f}×\n"
        f"and the air's added mass flattens the top mode by "
        f"{(1.0 - f_reactive[-1] / BODY_FREQS[-1]) * 100:.2f}% in pitch",
        transform=ax.transAxes, fontsize=8, ha="right", va="bottom",
    )
    fig.tight_layout()
    path = os.path.join(OUT, "radiation_impedance_spectral_shaping.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path, f_reactive, alpha_reactive, alpha_constant, predicted


def main() -> None:
    p1, err_warp, err_cont = fig_impedance()
    print(f"impedance sweep         -> {p1}")
    print(f"  measured vs pre-warped closed form : max rel err {err_warp:.3e}")
    print(f"  pre-warped vs continuous Z_a(jw)   : max rel err {err_cont:.3e}  (the bilinear warp)")
    p2, drift = fig_energy_channels()
    print(f"energy channels         -> {p2}")
    print(f"  peak |total - E0| / E0             : {drift:.3e}")
    p3, f_meas, a_react, a_const, pred = fig_spectral_shaping()
    print(f"spectral shaping        -> {p3}")
    for f, fm, am, ac, ap in zip(BODY_FREQS, f_meas, a_react, a_const, pred, strict=True):
        print(
            f"  f = {f:6.1f} Hz -> {fm:7.2f} Hz loaded   alpha {am:8.3f} (predicted {ap:8.3f})"
            f"   constant-R {ac:8.3f}  s^-1"
        )


if __name__ == "__main__":
    main()
