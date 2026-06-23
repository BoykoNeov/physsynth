"""Plate (model #5): modal frequencies vs the analytic simply-supported oracle.

Unlike the staircased membrane circle, the simply-supported rectangle has **machine-precise
eigenvalues** (``sin·sin`` is an exact discrete eigenvector of ``B = L²`` with eigenvalue ``Λ²``),
so we hold a **tight** bar — operator eigenvalues to machine precision, continuum convergence at
O(h²), and low modes within ~1 cent. There is no κ=0 reduction to lean on (κ=0 ⇒ u_tt=0,
degenerate);
operator correctness is proved instead by the ``B``-eigenvalue == ``Λ²`` money test.
"""

import numpy as np
from helpers import KAPPA_PLATE_DEFAULT, convergence_orders, make_plate, plate_low_eigenfrequencies
from scipy.sparse.linalg import eigsh

from physsynth.analysis import modal, spectrum
from physsynth.core.engine import simulate
from physsynth.core.exciter import raised_cosine_2d
from physsynth.core.operators2d import biharmonic_from_mask

KAPPA = KAPPA_PLATE_DEFAULT
THETA = 0.28


# -- Money test: the assembled biharmonic B = L² reproduces the squared Laplacian spectrum. --
def test_biharmonic_eigenvalues_are_squared_laplacian():
    """Replaces the (nonexistent) κ=0 anchor: B's eigenvalues must equal Λ² to machine
    precision."""
    N = 24
    p = make_plate(N=N, mu=1.0)
    Ny = round(p.Ly / p.h)
    modes = [(1, 1), (2, 1), (1, 2), (2, 2), (3, 1), (1, 3)]
    lap = np.sort(modal.rectangular_discrete_eigenvalues(p.h, N, Ny, modes))
    bih_oracle = lap ** 2
    bih_numeric = np.sort(
        eigsh(p.B, k=len(modes), sigma=0.0, which="LM", return_eigenvectors=False)
    )
    rel = np.max(np.abs(bih_numeric - bih_oracle) / bih_oracle)
    assert rel < 1e-10, f"biharmonic eigenvalue mismatch {rel:.2e} (B is mis-assembled)"

    # The standalone operators2d.biharmonic_from_mask must reproduce Plate's inline B = L @ L.
    B_helper, _ = biharmonic_from_mask(p.mask, p.h)
    d = (B_helper - p.B).tocoo()
    assert d.nnz == 0 or np.abs(d.data).max() < 1e-12, "biharmonic_from_mask != Plate's assembled B"


# -- Continuum frequency error converges at O(h²) (k ∝ h² at fixed mu, so temporal error ∝ h⁴).
# --
def test_rectangle_continuum_convergence_order():
    Ns = [16, 32, 64, 128]
    mu, Lx = 1.0, 1.0
    modes = [(1, 1), (2, 1)]
    f_cont = modal.rectangular_plate_freqs(KAPPA, Lx, Lx, modes)
    hs, errs = [], []
    for N in Ns:
        h = Lx / N  # square plate -> Ny = N; analytic (no time-stepping needed for this test)
        k = mu * h * h / KAPPA  # make_plate's fs = kappa/(mu h²) inverted
        Lam = modal.rectangular_discrete_eigenvalues(h, N, N, modes)
        f_disc = modal.discrete_plate_eigenfrequency(Lam, KAPPA, k, THETA)
        hs.append(h)
        errs.append(np.max(np.abs(f_disc - f_cont)))
    orders = convergence_orders(np.array(errs), np.array(hs))
    assert np.all(np.array(errs[1:]) < np.array(errs[:-1])), f"errors not decreasing: {errs}"
    assert orders[-1] > 1.8, f"continuum convergence order {orders[-1]:.2f} < 1.8 (expected ~2)"


# -- Tight bar: low modes within ~1 cent at a fine grid + fine timestep (NOT a loosened bound). --
def test_low_modes_within_one_cent():
    N, mu, Lx = 96, 0.5, 1.0
    h = Lx / N
    k = mu * h * h / KAPPA
    modes = [(1, 1), (2, 1), (1, 2), (2, 2)]
    Lam = modal.rectangular_discrete_eigenvalues(h, N, N, modes)
    f_disc = modal.discrete_plate_eigenfrequency(Lam, KAPPA, k, THETA)
    f_cont = modal.rectangular_plate_freqs(KAPPA, Lx, Lx, modes)
    err_cents = np.abs(modal.cents(f_disc, f_cont))
    assert np.max(err_cents) < 1.0, f"max error {np.max(err_cents):.3f} cents > 1 (tight bar)"


# -- eigsh on the actual assembled L, mapped through the scheme oracle, tracks the analytic
# series. --
def test_low_spectrum_via_eigsh_matches_oracle():
    N = 64
    p = make_plate(N=N, mu=0.5)
    Ny = round(p.Ly / p.h)
    # Sorted analytic discrete frequencies for the low (m,n) modes (degeneracy included by sorting).
    modes = [(m, n) for m in range(1, 5) for n in range(1, 5)]
    Lam = modal.rectangular_discrete_eigenvalues(p.h, N, Ny, modes)
    f_oracle = np.sort(modal.discrete_plate_eigenfrequency(Lam, p.kappa, p.k, p.theta))[:6]
    measured = plate_low_eigenfrequencies(p, 6)
    worst = np.max(np.abs(modal.cents(measured, f_oracle)))
    assert worst < 0.5, f"worst low-mode error {worst:.3f} cents (tight); measured={measured}"


# -- End-to-end: the time-stepper actually rings at the discrete fundamental (FFT sanity). --
def test_fft_peak_at_fundamental():
    N = 48
    p = make_plate(N=N, mu=1.0)
    f_disc = plate_low_eigenfrequencies(p, 1)[0]
    field = raised_cosine_2d(p.X, p.Y, (0.35 * p.Lx, 0.42 * p.Ly), 0.4 * p.Lx, amplitude=1e-3)
    field[~p.mask] = 0.0
    p.set_state(field)
    pickup = p.pickup_index_at(0.3 * p.Lx, 0.28 * p.Ly)
    res = simulate(p, num_steps=int(0.5 * p.fs), pickup_index=pickup)
    found = spectrum.measure_partials_near(
        res.output, res.fs, np.array([f_disc]), search_hz=20.0
    )[0]
    cents = abs(modal.cents(found, f_disc))
    assert cents < 5.0, (
        f"FFT fundamental off by {cents:.2f} cents (found {found:.2f}, want {f_disc:.2f})"
    )
