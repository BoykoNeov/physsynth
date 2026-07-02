"""Von Kármán nonlinear plate — small-amplitude limit & pitch glide (model #6, Part 3).

Two qualitative-but-decisive oracles that replace the (nonexistent) closed-form nonlinear spectrum:

- **Small-amplitude → model #5.** As ``w/e → 0`` the coupling ``l(w, F)`` vanishes like ``w³`` and
  the plate must collapse to the linear simply-supported Kirchhoff plate — same fundamental as
  :class:`Plate`, same trajectory. This ties the nonlinear model back to the validated linear one.
- **Pitch glide (hardening).** The signature weakly-nonlinear behaviour: the fundamental *rises*
  monotonically with drive amplitude (geometric stiffening from the membrane stretch). We assert the
  sign, not a cents bar — the frequency law is amplitude-dependent and has no analytic value.
"""

import numpy as np

from physsynth.core.plate import Plate, VKPlate

MAT = dict(E=2.0e11, e=1.0e-3, nu=0.3, rho=7800.0)


def _fundamental(sig, fs):
    """Crude fundamental from the mean spacing of zero-crossings (enough for sign/agreement)."""
    sig = sig - sig.mean()
    zc = np.where(np.diff(np.signbit(sig)))[0]
    return fs / (2.0 * np.mean(np.diff(zc)))


# -- #4 small amplitude collapses onto the linear model #5 ----------------------------------------
def test_small_amplitude_tracks_model5():
    """At ``w/e ~ 1e-6`` the VK trajectory matches the linear plate to the coupling's ``w³`` floor.

    The nonlinearity is negligible, so the coupled scheme must reproduce the linear (1,1) mode.
    """
    Lx = Ly = 0.4
    fs = 48000.0
    vk = VKPlate(Lx=Lx, Ly=Ly, fs=fs, N=20, **MAT)
    p5 = Plate(Lx=Lx, Ly=Ly, kappa=vk.kappa, rho=vk.rho_s, fs=fs, N=20, boundary="supported")
    mode = np.sin(np.pi * vk.X / Lx) * np.sin(np.pi * vk.Y / vk.Ly)  # (1,1) SS eigenmode
    ic = 1e-6 * MAT["e"] * mode
    vk.set_state(ic)
    p5.set_state(ic)
    pk = vk.pickup_index_at(0.5 * Lx, 0.5 * vk.Ly)
    worst = 0.0
    scale = 0.0
    for _ in range(2000):
        vk.step()
        p5.step()
        a, b = vk.displacement_at(pk), p5.displacement_at(pk)
        worst = max(worst, abs(a - b))
        scale = max(scale, abs(b))
    assert worst / scale < 1e-4  # nonlinearity is negligible → reproduces the linear mode


def test_small_amplitude_fundamental_near_analytic():
    """The small-amplitude fundamental matches the SS law ``f = (π/2)κ[(1/Lx)²+(1/Ly)²]``."""
    Lx, Ly = 0.4, 0.32
    fs = 48000.0
    vk = VKPlate(Lx=Lx, Ly=Ly, fs=fs, N=24, **MAT)
    mode = np.sin(np.pi * vk.X / Lx) * np.sin(np.pi * vk.Y / vk.Ly)
    vk.set_state(1e-4 * MAT["e"] * mode)
    pk = vk.pickup_index_at(0.47 * Lx, 0.53 * vk.Ly)
    sig = np.empty(4000)
    for i in range(4000):
        vk.step()
        sig[i] = vk.displacement_at(pk)
    f_meas = _fundamental(sig, fs)
    f_an = (np.pi / 2.0) * vk.kappa * ((1.0 / Lx) ** 2 + (1.0 / vk.Ly) ** 2)
    assert abs(f_meas - f_an) / f_an < 0.02  # grid dispersion only (as model #5)


# -- #5 pitch glide: fundamental rises with amplitude (hardening, correct sign) --------------------
def test_pitch_glide_hardening():
    Lx = Ly = 0.15  # small plate → high fundamental → many periods in a short run
    fs = 48000.0
    freqs = []
    for amp in (0.01, 1.5, 3.0, 5.0):
        vk = VKPlate(Lx=Lx, Ly=Ly, fs=fs, N=18, **MAT)
        mode = np.sin(np.pi * vk.X / Lx) * np.sin(np.pi * vk.Y / vk.Ly)
        vk.set_state(amp * MAT["e"] * mode)
        pk = vk.pickup_index_at(0.47 * Lx, 0.53 * vk.Ly)
        sig = np.empty(5000)
        for i in range(5000):
            vk.step()
            sig[i] = vk.displacement_at(pk)
        freqs.append(_fundamental(sig, fs))
    assert all(lo < hi for lo, hi in zip(freqs, freqs[1:], strict=False))  # monotone rising
    assert freqs[-1] > 1.2 * freqs[0]  # a substantial glide by w ≈ 5 e
