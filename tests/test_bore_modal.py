"""Acoustic bore (wind leg): resonances vs the closed-form air-column oracle.

The model-specific correctness signature (HANDOFF §5-§6, the clarinet analogue of the string's
harmonic series and the plate's Chladni patterns):

- a **closed-open** cylinder resonates at the **odd** harmonics ``f_n = (2n-1) c0 / 4L`` and the
  **even** harmonics are *absent* — the defining clarinet timbre and the money test here;
- an **open-open** cylinder resonates at the **full** harmonic series ``f_n = n c0 / 2L``;
- the discrete eigenvalue oracle matches the continuum exactly at ``lambda = 1`` (dispersionless)
  and converges to it at O(h²) for ``lambda < 1`` (numerical dispersion), and it tracks the measured
  FDTD spectrum to a small fraction of a cent.
"""

import numpy as np
from helpers import bore_low_eigenfrequencies, convergence_orders, make_bore

from physsynth.analysis import modal, spectrum
from physsynth.core.engine import simulate


def _bump(bore, center_frac, width_frac=0.06, amplitude=1e-3):
    c = center_frac * bore.L
    w = width_frac * bore.L
    return amplitude * np.exp(-((bore.x - c) ** 2) / (2.0 * w * w))


# -- Operator symmetry: L = Gᵀ M⁻¹ G is symmetric by construction (=> energy conservation). ------
def test_pressure_operator_symmetric():
    bore = make_bore(N=120, lam=1.0)
    asym = abs(bore.Lop - bore.Lop.T).max()
    assert asym < 1e-12, f"L not symmetric: ||L-Lᵀ||_max = {asym:.3e}"


# -- Money test: closed-open is ODD-harmonic; the even harmonics are at the noise floor. ---------
def test_closed_open_is_odd_harmonic_only():
    bore = make_bore(N=256, lam=1.0, boundary=("closed", "open"))
    bore.set_state(_bump(bore, center_frac=0.12, width_frac=0.06))  # bump near the closed wall
    res = simulate(bore, num_steps=int(0.5 * bore.fs), pickup_index=1)
    f1 = modal.bore_resonance_frequencies(bore.c0, bore.L, 1, "closed-open")[0]

    freqs, mag, _ = spectrum.magnitude_spectrum(res.output, res.fs)
    df = freqs[1] - freqs[0]

    def peak_near(f):
        i = int(round(f / df))
        return mag[max(1, i - 2) : i + 3].max()

    odd = [peak_near((2 * n - 1) * f1) for n in range(1, 6)]   # f1, 3f1, 5f1, ...
    even = [peak_near(2 * n * f1) for n in range(1, 6)]        # 2f1, 4f1, ...
    # Every odd harmonic dwarfs every even one: a real spectral gap of many orders of magnitude,
    # not a threshold fudge. (Empirically ~2e5x; assert a conservative 1e3.)
    assert min(odd) > 1e3 * max(even), (
        f"even harmonics not suppressed: min(odd)={min(odd):.2e}, max(even)={max(even):.2e}"
    )


# -- Open-open is the FULL harmonic series (even harmonics now present). -------------------------
def test_open_open_full_series_present():
    bore = make_bore(N=256, lam=1.0, boundary=("open", "open"))
    bore.set_state(_bump(bore, center_frac=0.23, width_frac=0.05))  # asymmetric -> excites all n
    res = simulate(bore, num_steps=int(0.5 * bore.fs), pickup_index=bore.N // 3)
    oracle = modal.bore_resonance_frequencies(bore.c0, bore.L, 4, "open-open")
    found = spectrum.measure_partials_near(res.output, res.fs, oracle)
    # The 2nd harmonic (an "even" one, absent for closed-open) must be present and on pitch.
    err = np.abs(modal.cents(found, oracle))
    assert np.all(err < 2.0), f"open-open partials off by {err} cents (want < 2)"


# -- Dispersionless at lambda = 1: the discrete oracle equals the continuum odd harmonics. -------
def test_discrete_equals_continuum_at_lambda_one():
    bore = make_bore(N=200, lam=1.0, boundary=("closed", "open"))
    f_disc = bore_low_eigenfrequencies(bore, 6)
    f_cont = modal.bore_resonance_frequencies(bore.c0, bore.L, 6, "closed-open")
    err = np.abs(modal.cents(f_disc, f_cont))
    assert np.max(err) < 0.05, f"lambda=1 not dispersionless: up to {np.max(err):.3f} cents"


# -- The eigenvalue oracle tracks the measured FDTD spectrum to a small fraction of a cent. ------
def test_oracle_tracks_measured_spectrum():
    bore = make_bore(N=150, lam=0.7, boundary=("closed", "open"))
    f_disc = bore_low_eigenfrequencies(bore, 5)
    bore.set_state(_bump(bore, center_frac=0.1, width_frac=0.06))
    res = simulate(bore, num_steps=int(0.6 * bore.fs), pickup_index=1)
    found = spectrum.measure_partials_near(res.output, res.fs, f_disc)
    err = np.abs(modal.cents(found, f_disc))
    assert np.max(err) < 0.5, f"measured spectrum off the oracle by up to {np.max(err):.3f} cents"


# -- The discrete oracle converges to the continuum at O(h²) (numerical dispersion, lambda < 1). --
def test_convergence_order_h2():
    oracle = modal.bore_resonance_frequencies(343.0, 0.5, 3, "closed-open")
    Ns = [50, 100, 200, 400]
    hs, errs = [], []
    for N in Ns:
        bore = make_bore(N=N, lam=0.7, boundary=("closed", "open"), L=0.5, c0=343.0)
        f = bore_low_eigenfrequencies(bore, 3)
        hs.append(bore.h)
        errs.append(np.max(np.abs(f - oracle)))
    assert np.all(np.diff(errs) < 0), f"dispersion error not decreasing: {errs}"
    order = convergence_orders(np.array(errs), np.array(hs))
    assert order[-1] > 1.9, f"convergence order {order[-1]:.2f} < 1.9"
