"""Tension-modulated-string (model #9) diagnostics — the Kirchhoff–Carrier money shots.

Renders diagnostics-as-visuals (HANDOFF §7) for the string family's nonlinearity, and prints the
real validation numbers so the model can be judged from the console alone:

1. **Energy breakdown** — a flat total riding over anti-correlated linear ↔ stretch exchange. Same
   shape as model #6's, from the same quartic potential.
2. **The closed-form hardening curve** — measured ω(A) landing *on* the exact Duffing elliptic
   oracle. This is what model #6 could never draw: it had no analytic nonlinear oracle.
3. **The Duffing waveform** — q(t) against the exact ``cn``, showing the visibly non-sinusoidal
   shape a linear string cannot produce.
4. **The parametric instability** — single-mode motion breaking up above the pump threshold while
   the energy stays flat at machine precision (physics, not a blow-up).
5. **Pitch glide** — a ringing spectrogram (needs σ>0): the note starts sharp and glides down onto
   the linear limit as it decays. The audible payoff.

Run:  python scripts/diagnose_tension_string.py
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
from physsynth.analysis.duffing import (  # noqa: E402
    duffing_displacement,
    duffing_frequency,
    kc_mode_coefficients,
)
from physsynth.core.exciter import triangular_pluck  # noqa: E402
from physsynth.core.string_nonlinear import (  # noqa: E402
    TensionModulatedString,
    string_coefficients_from_material,
)
from physsynth.viz import plots  # noqa: E402

# Canonical string: c = 200 m/s, f0 = c/(2L) = 100 Hz. kappa = 2 -> B ~ 9.87e-4 (piano-ish).
L, T, RHO, KAPPA = 1.0, 200.0, 0.005, 2.0
EA = 1.0e5  # EA/T = 500 -- a real steel string's ratio (c_long/c)^2
C = float(np.sqrt(T / RHO))
OUT = "out"


def build(N=100, lam=1.0, *, EA=EA, sigma0=0.0, sigma1=0.0):
    return TensionModulatedString(
        L=L, T=T, rho=RHO, fs=C * N / (L * lam), N=N, kappa=KAPPA, EA=EA,
        sigma0=sigma0, sigma1=sigma1,
    )


def mode(s, m):
    return np.sin(m * np.pi * np.arange(s.N + 1) / s.N)


def off_fraction(u, shape, scale):
    proj = np.dot(u, shape) / np.dot(shape, shape) * shape
    return float(np.linalg.norm(u - proj) / scale)


def measure_frequency(s, shape, n_crossings=10):
    """Nonlinear frequency (Hz) from descending zero crossings of the modal projection."""
    denom = float(np.dot(shape, shape))
    prev, times = float(np.dot(s.state, shape)) / denom, []
    for n in range(1, 400_000):
        s.step()
        cur = float(np.dot(s.state, shape)) / denom
        if prev > 0.0 >= cur:
            times.append((n - 1 + prev / (prev - cur)) * s.k)
            if len(times) >= n_crossings:
                break
        prev = cur
    return 1.0 / float(np.mean(np.diff(times)))


def panel_energy(ax):
    """A broadband pluck: the general case, where the cross-mode coupling through I is live."""
    s = build()
    s.set_state(triangular_pluck(s.x, L, 0.3 * L, amplitude=0.04))
    e0 = s.energy()
    t, tot, lin, nl = [], [], [], []
    tmax, drift = 0.0, 0.0
    for n in range(1500):
        s.step()
        e = s.energy()
        t.append(n * s.k)
        tot.append(e)
        nl.append(s.nonlinear_energy())
        lin.append(e - nl[-1])
        tmax = max(tmax, s.tension / s.T)
        drift = max(drift, abs(e - e0) / e0)
    tot, nl = np.array(tot), np.array(nl)
    plots.plot_energy_breakdown(
        ax, np.array(t), tot, np.array(lin), nl, drift=drift,
        linear_label="linear (kinetic + strain + bending)",
        nonlinear_label=r"stretch store $(EA/8L)\,I^2$",
        title="KC energy: flat total over anti-correlated linear/stretch exchange (plucked)",
    )
    print(f"  [1] plucked broadband: drift = {drift:.3e}, nonlinear fraction max = "
          f"{np.max(nl / tot):.1%}, tension max = {tmax:.2f} x T0, "
          f"non-converged steps = {s.n_not_converged}")
    return drift


def panel_frequency_curve(ax):
    """Measured omega(A) vs the EXACT elliptic oracle -- model #9's unique visual."""
    N, m = 100, 1
    p2 = spatial_eigenvalue_p2(N, L / N, m)
    w0sq, eps = kc_mode_coefficients(c=C, kappa=KAPPA, EA=EA, rho=RHO, p2=p2, L=L)
    amps = np.array([1e-5, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05])
    meas, orc = [], []
    for a in amps:
        s = build(N=N)
        shape = mode(s, m)
        s.set_state(a * shape)
        meas.append(measure_frequency(s, shape))
        orc.append(duffing_frequency(a, w0sq, eps) / (2 * np.pi))
    meas, orc = np.array(meas), np.array(orc)
    f_linear = float(np.sqrt(w0sq) / (2 * np.pi))
    plots.plot_duffing_frequency_curve(ax, amps, meas, orc, f_linear)
    worst = float(np.max(np.abs(meas - orc) / orc))
    print(f"  [2] hardening curve: {f_linear:.2f} Hz (linear) -> {meas[-1]:.2f} Hz at A=0.05 "
          f"({100 * (meas[-1] / f_linear - 1):+.0f}%); worst |meas-oracle|/oracle = {worst:.2e}")
    return worst


def panel_waveform(ax):
    """q(t) vs the exact cn solution -- the non-sinusoidal Duffing shape."""
    N, m, amp = 200, 1, 0.04
    beta2 = (m * np.pi / L) ** 2  # continuum: comparing to the continuum closed form
    w0sq, eps = kc_mode_coefficients(c=C, kappa=KAPPA, EA=EA, rho=RHO, p2=beta2, L=L)
    s = build(N=N, lam=0.25)
    shape = mode(s, m)
    s.set_state(amp * shape)
    denom = float(np.dot(shape, shape))
    n_steps = int(round(2.2 * (2 * np.pi / duffing_frequency(amp, w0sq, eps)) * s.fs))
    t = np.arange(n_steps + 1) * s.k
    q = [amp]
    for _ in range(n_steps):
        s.step()
        q.append(float(np.dot(s.state, shape)) / denom)
    q = np.array(q)
    q_exact = duffing_displacement(t, amp, w0sq, eps)
    cn_label = r"exact $A\,\mathrm{cn}(\Omega t, m)$"
    ax.plot(t * 1e3, q_exact * 1e3, color="C3", lw=2.0, label=cn_label)
    ax.plot(t * 1e3, q * 1e3, color="C0", lw=1.0, ls="--", label="FDTD")
    ax.plot(t * 1e3, amp * np.cos(np.sqrt(w0sq) * t) * 1e3, color="0.6", lw=1.0, ls=":",
            label=r"linear $A\cos(\omega_0 t)$ (model #3)")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("modal amplitude $q$ (mm)")
    err = float(np.max(np.abs(q - q_exact)) / amp)
    ax.set_title(
        f"The Duffing waveform: FDTD on the exact $\\mathrm{{cn}}$ (max err {err:.1e} of $A$) — "
        "note the drift off the linear cosine",
        fontsize=9,
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")
    print(f"  [3] waveform vs exact cn: max|q - cn|/A = {err:.3e} (N={N}, fs={s.fs:.0f})")
    return err


def panel_instability(ax):
    """The discovery: single-mode motion is parametrically unstable -- and conserves energy."""
    steps, m = 1500, 3
    series = {}
    for tag, amp in (("below", 0.01), ("above", 0.03)):
        s = build()
        shape = mode(s, m)
        s.set_state(amp * shape)
        scale = float(np.linalg.norm(amp * shape))
        e0 = s.energy()
        off, drift, tmax = [], [], 0.0
        for _ in range(steps):
            s.step()
            off.append(off_fraction(s.state, shape, scale))
            drift.append(abs(s.energy() - e0) / e0)
            tmax = max(tmax, s.tension / s.T)
        series[tag] = (np.array(off), np.array(drift), tmax)
        u = s.state
        amps = {mm: abs(float(np.dot(u, mode(s, mm)) / np.dot(mode(s, mm), mode(s, mm))))
                for mm in (3, 4, 8)}
        print(f"  [4] {tag:5s} threshold (T max = {tmax:5.2f} x T0): off-mode -> "
              f"{off[-1]:.3e}, drift -> {max(drift):.3e}, "
              f"modes {{3: {amps[3]:.2e}, 4: {amps[4]:.2e}, 8: {amps[8]:.2e}}}")
    t = np.arange(steps) * (1.0 / build().fs)
    plots.plot_mode_purity(ax, t, series["below"][0], series["above"][0], series["above"][1])
    return series["above"][0][-1], max(series["above"][1])


def panel_glide(ax):
    """The audible payoff: struck hard it starts sharp and glides down as it decays (needs sigma>0).

    Resolving a 100->180 Hz glide is a time-frequency budget, not an afterthought: the STFT window
    must be long enough to separate those (~5 Hz => >= 0.2 s) *and* the ring-down long enough to
    give many windows across it (~2 s). A short window at a high sample rate -- the reflex choice --
    smears the whole glide into one flat band and shows nothing.
    """
    secs, nperseg = 2.0, 4096
    s = build(N=100, lam=1.0, sigma0=1.5, sigma1=1e-3)  # fs = 20 kHz; decays ~e^-1.5t
    s.set_state(0.05 * mode(s, 1))
    p2 = spatial_eigenvalue_p2(s.N, s.h, 1)
    w0sq, eps = kc_mode_coefficients(c=C, kappa=KAPPA, EA=EA, rho=RHO, p2=p2, L=L)
    f_start = duffing_frequency(0.05, w0sq, eps) / (2 * np.pi)
    f_lin = float(np.sqrt(w0sq) / (2 * np.pi))
    pickup = int(round(0.1 * s.N))
    sig = []
    for _ in range(int(secs * s.fs)):
        s.step()
        sig.append(s.displacement_at(pickup))
    plots.plot_spectrogram(ax, np.array(sig), s.fs, f_max=250.0, f_linear=f_lin, nperseg=nperseg)
    ax.set_title(
        f"Pitch glide: struck hard it starts sharp (~{f_start:.0f} Hz) and falls onto the linear "
        f"limit ({f_lin:.0f} Hz)",
        fontsize=9,
    )
    print(f"  [5] glide: {f_start:.1f} Hz -> {f_lin:.1f} Hz over {secs} s "
          f"(fs={s.fs:.0f}, window={nperseg / s.fs * 1e3:.0f} ms -> {s.fs / nperseg:.1f} Hz res), "
          f"non-converged = {s.n_not_converged}")


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    print("Tension-modulated string (model #9, Kirchhoff-Carrier) — diagnostics")
    print(f"  string: L={L} m, T={T} N, rho={RHO} kg/m -> c={C:.0f} m/s, f1={C / (2 * L):.0f} Hz")
    print(f"  EA={EA:.1e} N -> EA/T = {EA / T:.0f} = (c_long/c)^2")
    co = string_coefficients_from_material(E=2.0e11, radius=2e-4, rho_v=7850.0, T=T)
    print(f"  (for reference, a real 0.2 mm steel string: EA/T = {co.EA_over_T:.0f}, "
          f"c_long = {co.c_long:.0f} m/s, kappa = {co.kappa:.3f})")
    print()

    fig, axes = plt.subplots(3, 2, figsize=(16, 13))
    panel_energy(axes[0, 0])
    panel_frequency_curve(axes[0, 1])
    panel_waveform(axes[1, 0])
    panel_instability(axes[1, 1])
    panel_glide(axes[2, 0])
    axes[2, 1].axis("off")
    axes[2, 1].text(
        0.02, 0.98,
        "Model #9 — Kirchhoff-Carrier tension modulation\n\n"
        "The string family's nonlinearity: displacement stretches\n"
        "the string, raising tension, raising pitch. Hardening only\n"
        "(I >= 0 => T >= T0) -- a string cannot be un-stretched.\n\n"
        "Unlike model #6 (von Karman), this model keeps a CLOSED-FORM\n"
        "nonlinear oracle: the single-mode reduction is an exact\n"
        "Duffing oscillator (elliptic frequency + cn waveform).\n\n"
        "EA = 0 is model #3 bit-for-bit (the theta-split earns it).\n\n"
        "NOT claimed: piano phantom partials (need longitudinal\n"
        "coupling), and the instability here is the PLANAR modal\n"
        "exchange -- not out-of-plane whirling (needs 2 polarizations).",
        va="top", ha="left", fontsize=9, family="monospace",
    )
    fig.tight_layout()
    path = os.path.join(OUT, "tension_string_diagnostics.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print()
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
