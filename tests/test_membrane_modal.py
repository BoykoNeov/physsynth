"""Membrane (model #4): modal frequencies vs the analytic oracles.

Two geometries, two acceptance bars (per the human's 2026-06-21 decision):

- **Rectangle** (harness unit-test): the assembled masked Laplacian's eigenvalues must equal the
  closed-form ``Λ_{mn}`` to machine precision, and the continuum ``f_{mn}`` error must fall at
  **O(h²)**. This de-risks the 2D operator + eigen-machinery before the staircase enters.
- **Circle** (the drumhead, model #4): the Bessel match is judged by **convergence rate** — the
  error shrinks at the ~O(h) staircase rate under refinement, with a *loose* absolute bound at the
  finest grid. NOT the 1D ~1-cent single-grid bar (staircasing caps accuracy at O(h)).
"""

import numpy as np
from helpers import (
    RADIUS_DEFAULT,
    convergence_orders,
    make_membrane,
    membrane_low_eigenfrequencies,
    wave_speed,
)
from scipy.sparse.linalg import eigsh

from physsynth.analysis import modal
from physsynth.core.engine import simulate
from physsynth.core.exciter import raised_cosine_2d

C = wave_speed()  # 200 m/s with the canonical membrane params


# -- Rectangle: assembled operator reproduces the closed-form discrete spectrum exactly. --
def test_rectangle_eigenvalues_match_closed_form():
    N = 24
    m = make_membrane(domain="rectangle", N=N, lam=0.6, Lx=1.0, Ly=1.0)
    Ny = round(m.Ly / m.h)
    modes = [(1, 1), (2, 1), (1, 2), (2, 2), (3, 1), (1, 3)]
    oracle = np.sort(modal.rectangular_discrete_eigenvalues(m.h, N, Ny, modes))
    numeric = np.sort(eigsh(-m.L, k=len(modes), sigma=0.0, which="LM", return_eigenvectors=False))
    rel = np.max(np.abs(numeric - oracle) / oracle)
    assert rel < 1e-10, f"discrete eigenvalue mismatch {rel:.2e} (operator is mis-assembled)"


# -- Rectangle: continuum frequency error converges at O(h^2). --
def test_rectangle_continuum_convergence_order():
    Ns = [16, 32, 64, 128]
    modes = [(1, 1), (2, 1)]
    f_cont = modal.rectangular_membrane_freqs(C, 1.0, 1.0, modes)
    hs, errs = [], []
    for N in Ns:
        m = make_membrane(domain="rectangle", N=N, lam=0.6, Lx=1.0, Ly=1.0)
        Ny = round(m.Ly / m.h)
        Lam = modal.rectangular_discrete_eigenvalues(m.h, N, Ny, modes)
        f_disc = modal.discrete_membrane_eigenfrequency(Lam, C, m.k)
        hs.append(m.h)
        errs.append(np.max(np.abs(f_disc - f_cont)))
    orders = convergence_orders(np.array(errs), np.array(hs))
    # Second-order scheme: each h-halving quarters the error -> p ~ 2.
    assert np.all(np.array(errs[1:]) < np.array(errs[:-1])), f"errors not decreasing: {errs}"
    assert orders[-1] > 1.8, f"continuum convergence order {orders[-1]:.2f} < 1.8 (expected ~2)"


# -- Circle: Bessel error converges at the staircase rate (~O(h)) + a loose absolute bound. --
def test_circle_bessel_convergence_rate():
    a = RADIUS_DEFAULT
    f01 = modal.circular_membrane_freqs(C, a, 1)[0][2]  # non-degenerate fundamental (0,1)
    Ns = [32, 64, 128]
    hs, errs, cents_fine = [], [], None
    for N in Ns:
        m = make_membrane(domain="circle", N=N, lam=0.6, radius=a)
        f_fund = membrane_low_eigenfrequencies(m, 1)[0]
        hs.append(m.h)
        errs.append(abs(f_fund - f01))
        cents_fine = abs(1200.0 * np.log2(f_fund / f01))
    errs = np.array(errs)
    orders = convergence_orders(errs, np.array(hs))
    # Monotone convergence at the staircase rate: clearly first-order-ish, neither stalled (p>0.5)
    # nor the clean O(h^2) of an exactly-fitted boundary (p<1.5).
    assert np.all(errs[1:] < errs[:-1]), f"Bessel error not decreasing: {errs}"
    assert np.all((orders > 0.5) & (orders < 1.5)), f"staircase orders out of band: {orders}"
    # Loose absolute bound at the finest grid (NOT the 1D ~1-cent bar; documents the looser tier).
    assert cents_fine < 12.0, f"fundamental off by {cents_fine:.2f} cents at N={Ns[-1]} (bound 12)"


# -- Circle: the sorted low spectrum tracks the sorted Bessel series (degeneracy included). --
def test_circle_low_spectrum_tracks_bessel():
    a = RADIUS_DEFAULT
    m = make_membrane(domain="circle", N=128, lam=0.6, radius=a)
    # Expand the (m,n) Bessel oracle into a degeneracy-weighted, sorted frequency list.
    oracle = []
    for _mm, _nn, f, deg in modal.circular_membrane_freqs(C, a, 8):
        oracle.extend([f] * deg)
    oracle = np.sort(oracle)[:8]
    measured = membrane_low_eigenfrequencies(m, 8)
    # Every measured mode within the loose staircase band of its Bessel partner (in cents).
    worst = np.max(np.abs(1200.0 * np.log2(measured / oracle)))
    assert worst < 20.0, f"worst low-mode error {worst:.2f} cents (bound 20); measured={measured}"


# -- End-to-end: the time-stepper actually rings at the discrete fundamental (FFT sanity). --
def test_circle_fft_peak_at_fundamental():
    a = RADIUS_DEFAULT
    m = make_membrane(domain="circle", N=64, lam=0.6, radius=a)
    f_disc = membrane_low_eigenfrequencies(m, 1)[0]
    # Off-centre bump (breaks symmetry so m>=1 modes ring too) and an off-axis pickup.
    field = raised_cosine_2d(m.X, m.Y, (0.25 * a, -0.1 * a), 0.5 * a, amplitude=1e-3)
    field[~m.mask] = 0.0
    m.set_state(field)
    pickup = m.pickup_index_at(0.3 * a, 0.2 * a)
    res = simulate(m, num_steps=int(0.5 * m.fs), pickup_index=pickup)
    detected = modal.discrete_membrane_eigenfrequency(  # anchor the search at the discrete fund.
        eigsh(-m.L, k=1, sigma=0.0, which="LM", return_eigenvectors=False), C, m.k
    )
    from physsynth.analysis import spectrum

    found = spectrum.measure_partials_near(res.output, res.fs, detected, search_hz=20.0)[0]
    cents = abs(1200.0 * np.log2(found / f_disc))
    assert cents < 5.0, (
        f"FFT fundamental off by {cents:.2f} cents (found {found:.2f}, want {f_disc:.2f})"
    )
