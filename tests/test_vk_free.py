"""Von Kármán nonlinear plate — **free-edge cymbal** (model #6, Part 6).

The iconic gong/cymbal: a *free* von Kármán plate. The Part-1 bracket is triple self-adjoint only
for rim-vanishing fields, yet a free plate has ``w ≠ 0`` on the rim — so "swap the boundary
operators" does **not** carry conservation over for free. It carries anyway because the Airy stress
function ``F`` is *still* clamped-zero: the coupling needs only the weaker swap identity
``⟨l(x, F), g⟩ = ⟨l(x, g), F⟩`` with ``F`` (not ``w``) in the rim-vanishing slot, which the existing
bracket satisfies under the **uniform-h²** pairing. The bending swaps to model #5b's free stiffness
``K`` / lumped mass ``W`` (ν re-enters); the mass is ``W`` but the coupling force pairs under
uniform ``h²`` (the mixed weighting — see ``docs/dev/von-karman-plate-plan.md`` Part 6). With no
analytic modal oracle, **energy conservation at large amplitude is the correctness test**, backed by
the ``nonlinear=False → model #5b`` regression and the ``w → 0 → model #5b`` collapse.
"""

import numpy as np
import pytest
from scipy.sparse.linalg import eigsh

from physsynth.core.plate import Plate, VKPlate

MAT = dict(E=2.0e11, e=1.0e-3, nu=0.3, rho=7800.0)
Lx = Ly = 0.4
FS = 48000.0


def _strike(vk, width=0.08):
    """Centered raised-Gaussian strike (full field; a free plate needs no rim clamp), peak 1.0."""
    dx = vk.X - 0.5 * Lx
    dy = vk.Y - 0.5 * vk.Ly
    return np.exp(-((dx * dx + dy * dy) / (width * width)))


def _make(**over):
    kw = dict(Lx=Lx, Ly=Ly, fs=FS, N=20, boundary="free", **MAT)
    kw.update(over)
    return VKPlate(**kw)


# -- #8 regression: coupling off is the *free* linear plate (model #5b), bit-for-bit --------------
def test_nonlinear_false_free_is_model5b_bit_identical():
    """``nonlinear=False, boundary="free"`` reproduces model #5b (``Plate(boundary="free")``)
    **exactly** — the free linear path is provably untouched by the nonlinear machinery."""
    vk = _make(nonlinear=False)
    p5b = Plate(
        Lx=Lx, Ly=Ly, kappa=vk.kappa, rho=vk.rho_s, fs=FS, N=20, boundary="free", nu=MAT["nu"]
    )
    ic = 1e-4 * _strike(vk)
    vk.set_state(ic)
    p5b.set_state(ic)
    for _ in range(150):
        vk.step()
        p5b.step()
        assert np.array_equal(vk.state, p5b.state)  # bit-identical, not merely close
    assert vk.energy() == p5b.energy()


# -- #2 the headline: lossless drift at LARGE amplitude (membrane a real fraction of H) -----------
def test_lossless_energy_drift_large_amplitude_free():
    vk = _make()
    vk.set_state(3.0 * MAT["e"] * _strike(vk))  # peak w ≈ 3 e → strongly nonlinear
    e0 = vk.energy()
    mem_frac0 = vk.membrane_energy() / e0
    emin = emax = e0
    for _ in range(600):
        vk.step()
        ei = vk.energy()
        emin, emax = min(emin, ei), max(emax, ei)
    drift = (emax - emin) / abs(e0)
    assert mem_frac0 > 0.1  # bracket genuinely exercised on the free rim, not a linear retest
    assert drift < 1e-10  # the project standard, through the nonlinearity, on a FREE plate


# -- #3 non-negativity + NaN-free across the run -------------------------------------------------
def test_energy_nonnegative_and_finite_free():
    vk = _make()
    vk.set_state(3.0 * MAT["e"] * _strike(vk))
    for _ in range(600):
        vk.step()
        ei = vk.energy()
        assert np.isfinite(ei) and ei >= 0.0
        assert np.all(np.isfinite(vk.u))


# -- the self-certifying gate: drift falls with the Picard tolerance ------------------------------
def test_drift_falls_with_couple_tolerance_free():
    """Exact conservation holds only *at* the Picard fixed point, so the residual drift is bounded
    by ``couple_tol`` — drift ↓ as ``couple_tol`` ↓. The machine-precision certificate that the
    *scheme* (not luck) conserves on the free plate, standing in for the absent closed form."""
    ic_amp = 3.0 * MAT["e"]
    drifts = []
    for tol in (1e-4, 1e-8, 1e-12):
        vk = _make(couple_tol=tol)
        vk.set_state(ic_amp * _strike(vk))
        e0 = vk.energy()
        emin = emax = e0
        for _ in range(400):
            vk.step()
            ei = vk.energy()
            emin, emax = min(emin, ei), max(emax, ei)
        drifts.append((emax - emin) / abs(e0))
    assert drifts[0] > drifts[1] > drifts[2]  # monotone with tolerance
    assert drifts[2] < 1e-10  # tightest tol reaches the project standard


# -- #7 passivity: σ > 0 → energy monotone non-increasing ----------------------------------------
def test_passivity_monotone_decay_free():
    vk = _make(sigma=8.0)
    vk.set_state(3.0 * MAT["e"] * _strike(vk))
    e0 = prev = vk.energy()
    for _ in range(800):
        vk.step()
        cur = vk.energy()
        assert cur <= prev + 1e-9 * e0  # never rises (loss is dissipative, not generative)
        prev = cur
    assert prev < e0  # and it genuinely decays


# -- #4 small amplitude collapses onto the free linear model #5b ---------------------------------
def test_small_amplitude_tracks_model5b():
    """At ``w/e ~ 1e-6`` the free-VK trajectory matches ``Plate(boundary="free")`` to the coupling's
    ``w³`` floor — the nonlinearity is negligible and the model reduces to the linear free plate."""
    vk = _make()
    p5b = Plate(
        Lx=Lx, Ly=Ly, kappa=vk.kappa, rho=vk.rho_s, fs=FS, N=20, boundary="free", nu=MAT["nu"]
    )
    ic = 1e-6 * MAT["e"] * _strike(vk)
    vk.set_state(ic)
    p5b.set_state(ic)
    pk = vk.pickup_index_at(0.47 * Lx, 0.53 * vk.Ly)
    worst = scale = 0.0
    for _ in range(2000):
        vk.step()
        p5b.step()
        a, b = vk.displacement_at(pk), p5b.displacement_at(pk)
        worst = max(worst, abs(a - b))
        scale = max(scale, abs(b))
    assert worst / scale < 1e-4  # negligible nonlinearity → reproduces the linear free mode


# -- energy bookkeeping accessors ----------------------------------------------------------------
def test_membrane_energy_positive_and_zero_when_linear():
    vk = _make()
    vk.set_state(2.0 * MAT["e"] * _strike(vk))
    for _ in range(20):
        vk.step()
    assert vk.energy() == pytest.approx(vk.linear_energy() + vk.membrane_energy())
    assert vk.membrane_energy() > 0.0  # a real membrane store at this amplitude
    lin = _make(nonlinear=False)
    lin.set_state(2.0 * MAT["e"] * _strike(lin))
    for _ in range(20):
        lin.step()
    assert lin.membrane_energy() == 0.0
    assert lin.energy() == lin.linear_energy()


def test_picard_converges_in_gate_regime_free():
    """In the gate regime (``w ≈ e``) the Picard iteration reaches ``couple_tol`` every step; the
    ``converged`` flag lets a cascade run (``w ≫ e``) detect silent non-convergence."""
    vk = _make()
    vk.set_state(1.0 * MAT["e"] * _strike(vk))
    for _ in range(300):
        vk.step()
        assert vk.converged
        assert vk.last_residual <= vk.couple_tol
        assert vk.n_iters < vk.couple_max_iter  # nowhere near the safety cap


# -- #5 pitch glide (hardening) on the free plate ------------------------------------------------
def _fft_peak(sig, fs, fmin=100.0):
    """Dominant spectral-peak frequency of ``sig`` above ``fmin`` (Hann-windowed, DC-removed)."""
    w = (sig - sig.mean()) * np.hanning(len(sig))
    spec = np.abs(np.fft.rfft(w))
    freqs = np.fft.rfftfreq(len(sig), 1.0 / fs)
    spec[freqs < fmin] = 0.0
    return freqs[int(np.argmax(spec))]


def test_pitch_glide_hardening_free():
    """The fundamental *rises* with drive amplitude (geometric stiffening). Driven by the lowest
    elastic free mode (clean single-mode response) and tracked by the FFT peak, in the weakly
    nonlinear regime ``w ≤ 3 e`` (beyond that the cascade smears "the" fundamental)."""
    Lxs = Lys = 0.15  # small plate → high fundamental → many periods per run
    fs = 96000.0
    base = VKPlate(Lx=Lxs, Ly=Lys, fs=fs, N=18, boundary="free", **MAT)
    vals, vecs = eigsh(base.K.tocsc(), k=6, M=base.W.tocsc(), sigma=-1e-3, which="LM")
    mode = vecs[:, np.argsort(vals)[3]]  # first elastic mode (skip the 3 rigid-body zeros)
    mode = mode / np.max(np.abs(mode))
    freqs = []
    for amp in (0.01, 1.0, 2.0, 3.0):
        vk = VKPlate(Lx=Lxs, Ly=Lys, fs=fs, N=18, boundary="free", **MAT)
        vk.set_state(amp * MAT["e"] * mode)
        pk = vk.pickup_index_at(0.30 * Lxs, 0.62 * vk.Ly)
        sig = np.empty(8000)
        for i in range(8000):
            vk.step()
            sig[i] = vk.displacement_at(pk)
        freqs.append(_fft_peak(sig, fs))
    assert all(lo < hi for lo, hi in zip(freqs, freqs[1:], strict=False))  # monotone rising
    assert freqs[-1] > 1.15 * freqs[0]  # a substantial glide by w ≈ 3 e


# -- #6 Richardson self-convergence: O(h²) -------------------------------------------------------
def test_richardson_second_order_free():
    """A fixed off-center pickup after a fixed time self-converges at O(h²) under ``h → h/2``.

    Smooth (cosine) IC keeps the run asymptotic; the pickup ``(0.1, 0.1)`` is a grid node on all of
    N = 24/48/96. Ratio → ≥ 4 (measured ≈ 5.7), as the free plate's O(h²) is (like model #5b)."""
    Lxs = Lys = 0.4
    fs = 96000.0

    def pickup_after(N, nsteps=80):
        vk = VKPlate(Lx=Lxs, Ly=Lys, fs=fs, N=N, boundary="free", **MAT)
        x, y = vk.X, vk.Y
        ic = np.cos(np.pi * x / Lxs) * np.cos(np.pi * y / vk.Ly) + 0.4 * np.cos(
            2 * np.pi * x / Lxs
        ) * np.cos(np.pi * y / vk.Ly)
        vk.set_state(MAT["e"] * ic)
        pk = vk.pickup_index_at(0.1, 0.1)
        for _ in range(nsteps):
            vk.step()
        return vk.displacement_at(pk)

    w = {N: pickup_after(N) for N in (24, 48, 96)}
    ratio = abs(w[24] - w[48]) / abs(w[48] - w[96])
    assert ratio > 3.4  # → ≥ 4 for O(h²)


# -- construction ---------------------------------------------------------------------------------
def test_free_construction_all_nodes_live():
    """The free plate makes *every* node a free unknown (no Dirichlet rim), carrying ``K``/``W``."""
    vk = _make(N=8)
    assert vk.n_live == vk.mask.size == (vk.N + 1) * (vk.Ny + 1)
    assert bool(np.all(vk.mask))  # no dead rim
    assert hasattr(vk, "K") and hasattr(vk, "W")
    assert vk.wdiag.shape == (vk.n_live,)


def test_rejects_bad_boundary():
    kw = dict(Lx=0.4, Ly=0.4, fs=48000.0, N=12, boundary="clamped", **MAT)
    with pytest.raises(ValueError):
        VKPlate(**kw)
