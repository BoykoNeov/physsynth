"""Von Kármán nonlinear plate — energy conservation & passivity (model #6, Part 3).

The nonlinear plate has **no analytic modal oracle**, so energy conservation *is* the correctness
test (HANDOFF §4.5; ``docs/dev/von-karman-plate-plan.md`` Part 3). The headline is the lossless
drift at **large amplitude** ``w ≳ e``, where the membrane term is a non-negligible fraction of the
total energy — a bracket / time-averaging bug hides at small amplitude (it just re-tests the linear
θ-scheme). The conservative scheme (``ρ_s δ_tt w = -D B w_θ + l(μ_{t·}w, μ_{t·}F)`` with ``F^m``
solved from ``w^m``) conserves ``E_lin + ½(H_mem(F^{n+1})+H_mem(F^n))`` exactly at the Picard
fixed point; the **drift-vs-tolerance** test certifies that (drift falls with
``couple_tol``) — the machine-precision replacement for a closed form.
"""

import numpy as np
import pytest

from physsynth.core.plate import Plate, VKPlate

# A thin steel-like plate (E, e, ν, ρ) → derives ρ_s, D, κ, Y = E e.
MAT = dict(E=2.0e11, e=1.0e-3, nu=0.3, rho=7800.0)
Lx = Ly = 0.4
FS = 48000.0


def _strike(vk, width=0.08):
    """Centered raised-Gaussian strike shape (full field, rim zeroed), peak 1.0."""
    dx = vk.X - 0.5 * Lx
    dy = vk.Y_grid - 0.5 * vk.Ly
    s = np.exp(-((dx * dx + dy * dy) / (width * width)))
    s[~vk.mask] = 0.0
    return s


def _make(**over):
    kw = dict(Lx=Lx, Ly=Ly, fs=FS, N=20, **MAT)
    kw.update(over)
    return VKPlate(**kw)


# -- #8 regression: coupling off is model #5, bit-for-bit ------------------------------------------
def test_nonlinear_false_is_model5_bit_identical():
    """``nonlinear=False`` reproduces the simply-supported model #5 plate **exactly** (the linear
    path is provably untouched — the whole point of a separate class)."""
    vk = _make(nonlinear=False)
    p5 = Plate(Lx=Lx, Ly=Ly, kappa=vk.kappa, rho=vk.rho_s, fs=FS, N=20, boundary="supported")
    ic = 1e-4 * _strike(vk)
    vk.set_state(ic)
    p5.set_state(ic)
    for _ in range(150):
        vk.step()
        p5.step()
        assert np.array_equal(vk.state, p5.state)  # bit-identical, not merely close
    assert vk.energy() == p5.energy()


# -- #2 the headline: lossless drift at LARGE amplitude (membrane a real fraction of H) ------------
def test_lossless_energy_drift_large_amplitude():
    vk = _make()
    vk.set_state(3.0 * MAT["e"] * _strike(vk))  # peak w ≈ 3 e → strongly nonlinear
    e0 = vk.energy()
    emin = emax = e0
    mem_frac_peak = 0.0
    peak_w = 0.0
    for _ in range(600):
        vk.step()
        ei = vk.energy()
        emin, emax = min(emin, ei), max(emax, ei)
        wmax = float(np.max(np.abs(vk.state)))
        if wmax > peak_w:
            peak_w, mem_frac_peak = wmax, vk.membrane_energy() / ei
    drift = (emax - emin) / abs(e0)
    assert mem_frac_peak > 0.1  # the bracket is genuinely exercised, not a linear re-test
    assert drift < 1e-10  # the project standard, through the nonlinearity


# -- #3 non-negativity + NaN-free across the run --------------------------------------------------
def test_energy_nonnegative_and_finite_large_amplitude():
    vk = _make()
    vk.set_state(3.0 * MAT["e"] * _strike(vk))
    for _ in range(600):
        vk.step()
        ei = vk.energy()
        assert np.isfinite(ei) and ei >= 0.0
        assert np.all(np.isfinite(vk.u))


# -- the self-certifying gate: drift falls with the Picard tolerance -------------------------------
def test_drift_falls_with_couple_tolerance():
    """Exact conservation holds only *at* the fixed point, so the residual drift is bounded by the
    Picard tolerance — drift ↓ monotonically as ``couple_tol`` ↓. This is the machine-precision
    certificate that the *scheme* conserves (a scheme bug would floor the drift regardless of tol),
    standing in for the absent closed-form oracle."""
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


# -- #7 passivity: σ > 0 → energy monotone non-increasing -----------------------------------------
def test_passivity_monotone_decay():
    vk = _make(sigma=8.0)
    vk.set_state(3.0 * MAT["e"] * _strike(vk))
    e0 = prev = vk.energy()
    for _ in range(800):
        vk.step()
        cur = vk.energy()
        assert cur <= prev + 1e-9 * e0  # never rises (loss is dissipative, not generative)
        prev = cur
    assert prev < e0  # and it genuinely decays


# -- energy bookkeeping accessors ------------------------------------------------------------------
def test_energy_is_linear_plus_membrane():
    vk = _make()
    vk.set_state(2.0 * MAT["e"] * _strike(vk))
    for _ in range(20):
        vk.step()
    assert vk.energy() == pytest.approx(vk.linear_energy() + vk.membrane_energy())
    assert vk.membrane_energy() > 0.0  # a real membrane store at this amplitude


def test_picard_converges_in_gate_regime():
    """In the gate regime (``w ≈ e``) the Picard iteration reaches ``couple_tol`` every step — the
    ``converged`` flag is the visible signal that lets a cascade run (``w ≫ e``, a later payoff)
    detect silent non-convergence instead of emitting non-conservative output that looks normal."""
    vk = _make()
    vk.set_state(1.0 * MAT["e"] * _strike(vk))
    for _ in range(300):
        vk.step()
        assert vk.converged
        assert vk.last_residual <= vk.couple_tol
        assert vk.n_iters < vk.couple_max_iter  # nowhere near the safety cap


def test_membrane_energy_zero_when_linear():
    vk = _make(nonlinear=False)
    vk.set_state(2.0 * MAT["e"] * _strike(vk))
    for _ in range(20):
        vk.step()
    assert vk.membrane_energy() == 0.0
    assert vk.energy() == vk.linear_energy()
