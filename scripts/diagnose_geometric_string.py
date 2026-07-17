"""Geometrically-exact-string (model #10) diagnostics — the two refusals, discharged on screen.

Renders diagnostics-as-visuals (HANDOFF §7) for the string family's *exact* geometry, and prints
the real validation numbers so the model can be judged from the console alone. Model #9 ends by
explicitly refusing two claims — **phantom partials** and **whirling** — because it has one field.
This model has three, and these are the pictures of that difference:

1. **Energy breakdown** — a flat total riding over anti-correlated transverse ↔ longitudinal
   exchange. *That exchange is the phantom mechanism*: the transverse field pumps the longitudinal
   one. The nonlinear excess is on a twin axis and is **signed**, unlike model #9's.
2. **Phantom partials** — the longitudinal spectrum with lines at ``f_i ± f_j`` and *not* at the
   transverse partials. Model #9's first refusal, discharged. (Conklin 1999.)
3. **Polarization orbits** — planar (a bit-exact line), the rotating wave (a closed circle), and
   whirling (a spiral opening out). Model #9's second refusal, discharged: the orbit *opening up*
   is the instability.
4. **The rotating wave (Tier B)** — the exact relative equilibrium: its deformed mode shape, its
   non-uniform tension field, and the Kirchhoff–Carrier frequency error that the deformation
   explains.

Run:  python scripts/diagnose_geometric_string.py
Figures land in ./out/.
"""

from __future__ import annotations

import os
import sys

# Make `physsynth` importable when run directly (no editable install): repo root on sys.path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from physsynth.analysis.damping import spatial_eigenvalue_p2  # noqa: E402
from physsynth.analysis.dispersion import stiff_dispersion_frequencies  # noqa: E402
from physsynth.analysis.duffing import kc_mode_coefficients  # noqa: E402
from physsynth.analysis.rotating_wave import (  # noqa: E402
    kc_circular_frequency,
    rotating_wave_history,
    solve_rotating_wave,
)
from physsynth.analysis.spectrum import detect_peaks, magnitude_spectrum  # noqa: E402
from physsynth.core.string_geometric import GeometricString  # noqa: E402
from physsynth.viz import plots  # noqa: E402

# Canonical string: c = 200 m/s, f1 = c/(2L) = 100 Hz; EA/T = 500 -> c_long/c = 22.4, a real
# steel string's ratio. The SAME numbers models #9 and #10's tests use, so the figures and the
# suite are talking about one string.
L, T, RHO, EA = 1.0, 200.0, 0.005, 1.0e5
C = float(np.sqrt(T / RHO))
C_LONG = float(np.sqrt(EA / RHO))
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out")


def _string(*, N=32, lam_long=0.9, kappa=0.0, kappa_w=None, **kw):
    """Build the canonical string with ``fs`` set from **lam_long** — the trap of this model.

    ``lam_long = c_long k / h`` is what governs, and nothing enforces it: the theta-scheme is
    unconditionally stable, so at the familiar transverse ``lam = 0.5`` (which here means
    ``lam_long = 11``) the model returns quiet nonsense with no warning from the CFL and no
    exception. Every run in this script is sized by the **fast** field.
    """
    fs = C_LONG * N / (L * lam_long)
    return GeometricString(L=L, T=T, rho=RHO, fs=fs, N=N, EA=EA, kappa=kappa, kappa_w=kappa_w,
                           **kw)


def _split(s):
    """``(total, transverse, longitudinal, nonlinear)`` energies (J), all via the public API.

    ``transverse = energy - nonlinear - longitudinal``: the class exposes the total, the nonlinear
    excess and the longitudinal channel, and the transverse store is what is left.
    """
    total = s.energy()
    nl = s.nonlinear_energy()
    lon = s.longitudinal_energy()
    return total, total - nl - lon, lon, nl


# -- 1. energy breakdown -----------------------------------------------------------------


def figure_energy():
    """Flat total over transverse ↔ longitudinal exchange, from a **plucked** (broadband) IC."""
    s = _string(N=32, lam_long=0.9)
    x = np.linspace(0.0, L, s.N + 1)
    at = 0.2
    s.set_state(4e-3 * np.where(x <= at, x / at, (1.0 - x) / (1.0 - at)))

    n_steps = int(0.05 * s.fs)
    t = np.arange(n_steps) / s.fs
    tot, tra, lon, nl = (np.empty(n_steps) for _ in range(4))
    for i in range(n_steps):
        s.step()
        tot[i], tra[i], lon[i], nl[i] = _split(s)
    drift = float(np.max(np.abs(tot - tot[0])) / tot[0])

    fig, ax = plt.subplots(figsize=(9, 4.2))
    plots.plot_geometric_energy_breakdown(ax, t, tot, tra, lon, nl, drift=drift)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "geometric_energy.png"), dpi=130)
    plt.close(fig)

    print("1. ENERGY (plucked, broadband — the general case, not a scalar Duffing test)")
    print(f"   lam_long              = {s.lam_long:.3f}   (lam = {s.lam:.4f})")
    print(f"   drift |E-E0|/E0       = {drift:.3e}   (gate: < 1e-10)")
    print(f"   |nonlinear| / E       = {np.max(np.abs(nl)) / tot[0]:.3e}  (0 would mean the "
          f"nonlinearity never engaged)")
    print(f"   longitudinal / E      = {np.max(lon) / tot[0]:.3e}")
    print(f"   min energy            = {np.min(tot):.6e} J   (floor is 0, and E >= 0 holds)")
    print(f"   min Lambda            = {s.stretch_ratio.min():.9f}  (< 1 would be a slack cell — "
          f"physics, not a bug)")
    print(f"   Newton stalls         = {s.n_not_converged}\n")


# -- 2. phantom partials -----------------------------------------------------------------


def figure_phantoms():
    """The longitudinal spectrum: lines at ``f_i ± f_j``, *not* at the transverse partials.

    Two modes, not one: a single mode gives only ``2 f1``, and the **sum and difference** tones are
    what cannot be mistaken for a harmonic of anything. ``kappa = 8`` on purpose — the partials are
    then inharmonic enough that ``f1 + f2`` is visibly *not* ``f3``, so a phantom cannot be
    confused with a partial that was always going to be there.
    """
    s = _string(N=32, lam_long=0.9, kappa=8.0)
    sin1 = np.sin(np.pi * s.x / L)
    sin2 = np.sin(2 * np.pi * s.x / L)
    s.set_state(1.5e-3 * (sin1 + sin2))

    n_steps = int(0.10 * s.fs)
    d1, d2 = float(np.dot(sin1, sin1)), float(np.dot(sin2, sin2))
    bridge = np.empty(n_steps)
    q1, q2 = np.empty(n_steps), np.empty(n_steps)
    for i in range(n_steps):
        s.step()
        bridge[i] = s.EA * s.v[1] / s.h  # the piano's radiating channel: EA v_x(0)
        q1[i] = np.dot(s.u, sin1) / d1
        q2[i] = np.dot(s.u, sin2) / d2

    # Measured partials, never predicted: the phantom rides on whatever the partials actually are,
    # and predicting them would fold the theta-scheme's dispersion AND the hardening shift into the
    # marker positions -- measuring the formula instead of the string.
    f1 = float(detect_peaks(q1, s.fs, 1, f_min=10.0)[0])
    f2 = float(detect_peaks(q2, s.fs, 1, f_min=10.0)[0])
    combos = np.array([f2 - f1, 2 * f1, f1 + f2, 2 * f2])
    ladder = stiff_dispersion_frequencies(s.c, L, s.N, s.kappa_u, s.k, s.theta, np.arange(1, 6))
    freqs, mag, _ = magnitude_spectrum(bridge, s.fs)
    f_long1 = C_LONG / (2 * L)

    # Zoom to the combination band, NOT out to the first longitudinal mode (~2 kHz). The claim is
    # that f1+f2 = 308 Hz is *11.4 Hz away* from the 3rd partial at 319 Hz -- on a 2 kHz axis that
    # gap is sub-pixel, the red and green markers sit on top of each other, and the figure appears
    # to show the opposite of what it says. The discrimination is the picture; frame it.
    fig, ax = plt.subplots(figsize=(9, 4.2))
    plots.plot_phantom_spectrum(ax, freqs, mag, ladder, combos, f_max=4.8 * f1)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "geometric_phantoms.png"), dpi=130)
    plt.close(fig)

    peaks = detect_peaks(bridge, s.fs, 40, f_min=10.0)
    peaks = peaks[peaks < 0.9 * f_long1]
    print("2. PHANTOM PARTIALS — model #9's first refusal, discharged")
    print(f"   transverse partials   f1 = {f1:.3f} Hz   f2 = {f2:.3f} Hz")
    print(f"   predicted phantoms    f2-f1 = {combos[0]:.3f}   2f1 = {combos[1]:.3f}   "
          f"f1+f2 = {combos[2]:.3f}   2f2 = {combos[3]:.3f}")
    print(f"   detected (blind, < {0.9 * f_long1:.0f} Hz): "
          f"{', '.join(f'{p:.2f}' for p in peaks[:6])}")
    print(f"   3rd transverse partial would be at {ladder[2]:.2f} Hz — f1+f2 = {combos[2]:.2f} Hz "
          f"is {ladder[2] - combos[2]:.2f} Hz away, so it is NOT a partial")
    print(f"   Newton stalls         = {s.n_not_converged}\n")


# -- 3. polarization orbits --------------------------------------------------------------


def figure_orbits():
    """Line, circle, spiral — the three transverse orbits, one per panel.

    The plot model #9 structurally cannot draw: it has one polarization, so its only orbit is a
    point on a line.
    """
    probe = None
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))

    # (a) PLANAR: a bit-exact straight line. max|w| == 0.0 forever (Tier A/2).
    s = _string(N=32, lam_long=0.9)
    s.set_state(4e-3 * np.sin(np.pi * s.x / L))
    probe = s.N // 4
    n_steps = int(0.02 * s.fs)
    u_h, w_h = np.empty(n_steps), np.empty(n_steps)
    for i in range(n_steps):
        s.step()
        u_h[i], w_h[i] = s.u[probe], s.w[probe]
    plots.plot_polarization_orbit(axes[0], u_h, w_h,
                                  title=f"Planar: $\\max|w| = {np.abs(w_h).max():.1f}$ exactly")
    planar_w = float(np.abs(w_h).max())

    # (b) The ROTATING WAVE: a closed circle, from the converged BVP (Tier B).
    s = _string(N=32, lam_long=0.9)
    wave = solve_rotating_wave(L=L, T=T, rho=RHO, EA=EA, fs=s.fs, N=s.N, theta=s.theta,
                               amplitude=4e-3, mode=1)
    u0, w0, v0, up, wp, vp = rotating_wave_history(wave, fs=s.fs)
    s.u, s.w, s.v = u0, w0, v0
    s.u_prev, s.w_prev, s.v_prev = up, wp, vp
    n_steps = int(1.5 * s.fs / wave.frequency)
    u_h, w_h = np.empty(n_steps), np.empty(n_steps)
    lk = 0.0
    for i in range(n_steps):
        s.step()
        u_h[i], w_h[i] = s.u[probe], s.w[probe]
        dt_v = (s.v[1:-1] - s.v_prev[1:-1]) / s.k
        lk = max(lk, 0.5 * s.rho * s.h * float(np.dot(dt_v, dt_v)))
    r = np.hypot(u_h, w_h)
    roundness = float((r.max() - r.min()) / r.mean())
    plots.plot_polarization_orbit(axes[1], u_h, w_h,
                                  title=f"Rotating wave (Tier B): radius constant to "
                                        f"{roundness:.1e}")

    # (c) WHIRLING: kappa_u != kappa_w, seeded out of plane -> the orbit opens out (batch 2).
    #
    # The parameters are NOT free, and picking them by eye gives 1.1x and a picture of nothing.
    # Whirling is a Mathieu tongue: it needs 0 < delta < eps A^2/2, where delta = ww^2 - wu^2 is
    # the detuning the kappa_w knob buys. Batch 2's recipe, verbatim -- drive the SOFT plane
    # (kappa_u = 0) at dT/T0 = 1.5, and place kappa_w at the tongue's predicted peak
    # delta = eps A^2/4. Off the tongue there is no exponential growth at all, at any amplitude.
    n_w = 16
    p2 = spatial_eigenvalue_p2(n_w, L / n_w, 1)
    w0sq, eps_kc = kc_mode_coefficients(c=C, kappa=0.0, EA=EA - T, rho=RHO, p2=p2, L=L)
    amp = float(np.sqrt(1.5 * w0sq / eps_kc))  # dT/T0 = 1.5: fast growth, below #9's planar 3.0
    kappa_w = float(np.sqrt(0.25 * eps_kc * amp**2 / p2**2))  # the tongue's peak
    s = _string(N=n_w, lam_long=0.9, kappa=0.0, kappa_w=kappa_w)
    shape = np.sin(np.pi * s.x / L)
    # A VELOCITY seed, never a displacement one: a displaced w is the rotation generator (an exact
    # solution of the out-of-plane variational equation), so it pins the growth at 1.00x and the
    # panel would show a line even inside the tongue. Batch 2 learned this the hard way.
    s.set_state(amp * shape, w_dot=1e-3 * amp * shape)
    probe_w = s.N // 4
    # 0.22 s, not batch 2's 0.06: the suite only needs to *measure* the exponential rate, but a
    # picture has to SATURATE. At 0.06 s the growth is already 63x and the panel is still a flat
    # line -- w is 63x a 1e-3 seed, i.e. still ~9 % of u, which on equal axes reads as planar.
    # By 0.22 s the orbit has opened to max|w|/max|u| ~ 0.7 and the whirl is what you see.
    e0_whirl = s.energy()
    n_steps = int(0.22 * s.fs)
    u_h, w_h = np.empty(n_steps), np.empty(n_steps)
    for i in range(n_steps):
        s.step()
        u_h[i], w_h[i] = s.u[probe_w], s.w[probe_w]
    growth = float(np.abs(w_h[-len(w_h) // 8:]).max() / max(np.abs(w_h[: len(w_h) // 8]).max(),
                                                            1e-300))
    whirl_drift = abs(s.energy() - e0_whirl) / e0_whirl
    whirl_ratio = float(np.abs(w_h).max() / np.abs(u_h).max())
    plots.plot_polarization_orbit(axes[2], u_h, w_h,
                                  title=f"Whirling ($\\kappa_u \\neq \\kappa_w$): the orbit opens "
                                        f"out\n$|w|$ grows {growth:.0f}$\\times$, energy drift "
                                        f"{whirl_drift:.0e}")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "geometric_orbits.png"), dpi=130)
    plt.close(fig)

    print("3. POLARIZATION ORBITS — model #9's second refusal, discharged")
    print(f"   planar        max|w|  = {planar_w:.1f}       (bit-exact zero: the reflection "
          f"symmetry, not a small number)")
    print(f"   rotating wave roundness = {roundness:.3e}   (radius constant -> a true circle)")
    print(f"   rotating wave long_kin  = {lk:.3e} J   (the longitudinal field does not move)")
    print(f"   whirling      |w| growth = {growth:.0f}x   -> max|w|/max|u| = {whirl_ratio:.3f} "
          f"(the orbit really opens)")
    print(f"                 kappa_w = {kappa_w:.1f} at the tongue's peak; a DEGENERATE string "
          f"gives 1.0x at any amplitude")
    print(f"   whirl energy drift    = {whirl_drift:.3e}   (it conserves straight THROUGH a "
          f"{growth:.0f}x blow-up: redistribution, not instability of the scheme)\n")


# -- 4. the rotating wave (Tier B) -------------------------------------------------------


def figure_rotating_wave():
    """The exact relative equilibrium: deformed shape, non-uniform tension, and the KC error.

    The payoff panel. Everything Tier C could only *measure* gets a **mechanism** here: the helix's
    mode shape is not a sine, its tension is not uniform, and the two facts are the same fact.
    """
    s = _string(N=32, lam_long=0.9)
    p2 = spatial_eigenvalue_p2(s.N, s.h, 1)
    omega0_sq, eps = kc_mode_coefficients(c=C, kappa=0.0, EA=EA - T, rho=RHO, p2=p2, L=L)
    amps = np.array([5e-4, 1e-3, 2e-3, 4e-3, 8e-3, 1.6e-2])

    waves = [solve_rotating_wave(L=L, T=T, rho=RHO, EA=EA, fs=s.fs, N=s.N, theta=s.theta,
                                 amplitude=a, mode=1, time_discrete=False) for a in amps]
    om_bvp = np.array([w.Omega for w in waves])
    om_kc = np.array([kc_circular_frequency(omega0_sq=omega0_sq, eps=eps, amplitude=a)
                      for a in amps])
    shape_res = np.array([w.shape_residual for w in waves])

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
    big = waves[-1]

    # (a) the deformed mode shape, against the sine it is NOT
    sine = big.amplitude * np.sin(np.pi * s.x / L)
    ax = axes[0]
    ax.plot(s.x, big.phi * 1e3, color="C0", lw=1.6, label=r"BVP $\phi(x)$")
    ax.plot(s.x, sine * 1e3, color="C3", lw=1.1, ls="--", label=r"$R\sin(\pi x/L)$")
    twin = ax.twinx()
    twin.plot(s.x, (big.phi - sine) * 1e6, color="C2", lw=1.0)
    twin.set_ylabel(r"$\phi - R\sin$ ($\mu$m)", color="C2")
    twin.tick_params(axis="y", labelcolor="C2")
    ax.set_xlabel("$x$ (m)")
    ax.set_ylabel(r"$\phi$ (mm)")
    ax.set_title(f"The helix is a DEFORMED sine\n(residual {big.shape_residual:.1e}, "
                 f"right axis)", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")

    # (b) the tension FIELD -- what a Kirchhoff-Carrier scalar cannot represent.
    #
    # Plotted as the deviation from the mean, NOT as T(x) itself. The first version of this panel
    # drew T(x) on an axis that included T0, and it rendered as a dead flat line that flatly
    # contradicted its own title: the rise is +63 N and the spread across x is 0.3 N, so on any
    # axis wide enough to show T0 the physics is one pixel. The deviation from the mean IS the
    # quantity KC throws away, so it is the honest thing to draw -- and the KC reference is then
    # exactly the zero line.
    ax = axes[1]
    x_cell = 0.5 * (s.x[:-1] + s.x[1:])
    mean_t = float(np.mean(big.tension))
    ax.plot(x_cell, big.tension - mean_t, color="C0", lw=1.6,
            label=r"GE: $T(x) - \langle T \rangle$")
    ax.axhline(0.0, color="C3", ls="--", lw=1.2, label="KC: one scalar (this line IS KC)")
    spread = (big.tension.max() - big.tension.min()) / (mean_t - T)
    ax.set_xlabel("$x$ (m)")
    ax.set_ylabel(r"$T(x) - \langle T \rangle$ (N)")
    ax.set_title(f"Tension is a FIELD — peaked at the ends\n"
                 f"$T_0$={T:.0f}, $\\langle T \\rangle$={mean_t:.1f} N (rise +{mean_t - T:.1f}); "
                 f"spread {100 * spread:.1f}% of the rise", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="lower center")

    # (c) the KC frequency error, and the shape residual that explains it
    ax = axes[2]
    rel = (om_bvp - om_kc) / om_kc
    ax.loglog(amps * 1e3, rel, "o-", color="C0", ms=5, lw=1.3,
              label=r"$(\Omega_{BVP} - \Omega_{KC})/\Omega_{KC}$")
    ax.loglog(amps * 1e3, shape_res, "s--", color="C2", ms=5, lw=1.1,
              label=r"shape residual")
    ax.loglog(amps * 1e3, rel[0] * (amps / amps[0]) ** 2, ":", color="0.5", lw=1.0,
              label=r"$O(R^2)$")
    ax.set_xlabel("amplitude $R$ (mm)")
    ax.set_ylabel("relative")
    ax.set_title("KC's error IS the shape deformation\n"
                 r"(both $O(R^2)$; ratio $\to 4/3$ as $R \to 0$)", fontsize=9)
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=8, loc="upper left")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "geometric_rotating_wave.png"), dpi=130)
    plt.close(fig)

    print("4. THE ROTATING WAVE (Tier B) — the model's only exact frequency oracle")
    print(f"   {'R (mm)':>8} {'f_BVP (Hz)':>12} {'f_KC (Hz)':>12} {'rel err':>11} "
          f"{'shape resid':>12} {'ratio':>7}")
    for a, wv, ok in zip(amps, waves, om_kc, strict=True):
        r = (wv.Omega - ok) / ok
        print(f"   {a * 1e3:8.2f} {wv.Omega / (2 * np.pi):12.5f} {ok / (2 * np.pi):12.5f} "
              f"{r:11.3e} {wv.shape_residual:12.3e} {r / wv.shape_residual:7.3f}")
    print("   ratio -> 4/3 as R -> 0, and it does NOT move with EA/T or with the mode (1.31-1.33")
    print("   across EA/T = 50/100/400 and modes 1/2): the deformation is a single geometric fact")
    print("   about spinning a helix, not a parameter-dependent accident.")
    print("   It is a SMALL-AMPLITUDE relation and the table above shows it expiring: by R = 8 mm")
    print("   the ratio is 1.08 and by 16 mm it is 0.53, as higher-order terms take over. Read the")
    print("   4/3 as a limit, not a law: Tier C measures residuals, it does not promise them.\n")


def main():
    os.makedirs(OUT, exist_ok=True)
    print("=" * 78)
    print("GEOMETRICALLY-EXACT STRING (model #10) — diagnostics")
    print(f"  L={L} m  T={T} N  rho={RHO} kg/m  EA={EA:.0f} N")
    print(f"  c={C:.1f} m/s (f1={C / (2 * L):.1f} Hz)   c_long={C_LONG:.1f} m/s   "
          f"EA/T={EA / T:.0f} -> c_long/c={C_LONG / C:.1f}")
    print("=" * 78 + "\n")
    figure_energy()
    figure_phantoms()
    figure_orbits()
    figure_rotating_wave()
    print(f"Figures written to {OUT}")


if __name__ == "__main__":
    main()
