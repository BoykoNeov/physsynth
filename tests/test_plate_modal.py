"""Plate (model #5): modal frequencies vs the analytic simply-supported oracle.

Unlike the staircased membrane circle, the simply-supported rectangle has **machine-precise
eigenvalues** (``sin·sin`` is an exact discrete eigenvector of ``B = L²`` with eigenvalue ``Λ²``),
so we hold a **tight** bar — operator eigenvalues to machine precision, continuum convergence at
O(h²), and low modes within ~1 cent. There is no κ=0 reduction to lean on (κ=0 ⇒ u_tt=0,
degenerate);
operator correctness is proved instead by the ``B``-eigenvalue == ``Λ²`` money test.
"""

import numpy as np
from helpers import (
    KAPPA_PLATE_DEFAULT,
    arpack_v0,
    convergence_orders,
    make_plate,
    plate_kwargs,
    plate_low_eigenfrequencies,
)
from scipy.sparse.linalg import eigsh

from physsynth.analysis import modal, spectrum
from physsynth.core.engine import simulate
from physsynth.core.exciter import raised_cosine_2d
from physsynth.core.operators2d import biharmonic_from_mask, laplacian_from_mask
from physsynth.core.plate import Plate

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
        eigsh(p.B, k=len(modes), sigma=0.0, which="LM", return_eigenvectors=False,
              v0=arpack_v0(p.B))
    )
    rel = np.max(np.abs(bih_numeric - bih_oracle) / bih_oracle)
    assert rel < 1e-10, f"biharmonic eigenvalue mismatch {rel:.2e} (B is mis-assembled)"

    # `Plate` builds its B through operators2d.biharmonic_from_mask (it used to spell `L @ L`
    # inline); a fresh build must give the same matrix, structure included.
    B_helper, _ = biharmonic_from_mask(p.mask, p.h)
    assert np.array_equal(B_helper.indptr, p.B.indptr)
    assert np.array_equal(B_helper.indices, p.B.indices)
    assert np.array_equal(B_helper.data, p.B.data), "biharmonic_from_mask != Plate's assembled B"


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


# -- The pre-port spelling, held here so a faithful port of a slip is distinguishable. ------------
#
# The Rust port of `biharmonic_from_mask` (plan section 26) changed the reference: SciPy's sparse
# product returns each row in whatever order its kernel touched the columns, and a CSR matvec sums a
# row in STORED order, so `B @ u` -- which `Plate.step` forms twice per timestep -- was a different
# sum on the two sides. `portable.canonical` sorts the Python side to the order both languages can
# express. That is a change to the shipped numbers, and every parity assertion in the suite compares
# Rust against the Python AS IT IS NOW: only a test holding the PREVIOUS expression can tell a
# faithful port of a slip from a faithful port. Plan section 25.5, third pin -- and here the
# quantity is continuous, so the pin is a tolerance on the trajectory rather than an equality.


def _biharmonic_before_2026_08_28(mask, h):
    """``Plate``'s biharmonic exactly as it was assembled before the canonical sort."""
    L, _ = laplacian_from_mask(mask, h)
    return (L @ L).tocsr()


def test_the_canonical_sort_changed_an_order_and_not_a_value():
    """The two operators hold the *same numbers*; only where they are stored moved."""
    for N in (8, 12, 16, 24):
        p = make_plate(N=N, mu=1.0)
        before = _biharmonic_before_2026_08_28(p.mask, p.h)
        assert not before.has_sorted_indices, (
            f"N={N}: SciPy now returns this product sorted, so this test compares an operator "
            "against itself and the pin has quietly stopped pinning anything"
        )
        after = before.copy()
        after.sort_indices()
        assert np.array_equal(after.indptr, p.B.indptr)
        assert np.array_equal(after.indices, p.B.indices)
        assert np.array_equal(after.data, p.B.data), (
            f"N={N}: sorting the old operator does not reproduce the shipped one -- the port "
            "changed a VALUE, which is a different (and much worse) claim than changing an order"
        )


def test_the_canonical_sort_left_the_shipped_plate_where_it_was():
    """... and the trajectory it produces is unmoved at 1e-11 of its amplitude, drift unmoved.

    **Runs on whichever plate the flag selects, and that took a change to the Rust side.** This
    test's subject is a *stored column order*: it steps one plate on the canonical operator and
    another on the kernel-order one and asserts the trajectory did not move. For a year of this
    port that made it inexpressible in Rust twice over -- `Csr` sorted every row it was handed, so
    both plates would have carried the *same* operator and the comparison would have asserted
    nothing while staying green (section 23.6's emptied comparison, reached through a fourth
    door), and `Plate.B` was a getter with no setter, so there was nowhere to put the other order
    anyway (section 40.5).

    Both are now answered, on the human's call of 2026-09-03: `Csr::from_arrays_preserving_order`
    is the one constructor in the crate that does not sort, and `Plate.B` has a setter that uses
    it. Neither is a convenience -- they exist for this test, they say so in their own doc
    comments, and the three merge kernels `debug_assert` the invariant they relax so a
    non-canonical operator cannot leak into an assembly.

    The `has_sorted_indices` assertion below is what keeps all of that honest: it fails if the
    matrix being injected has become the shipped one by another route.
    """
    rng = np.random.default_rng(20260828)
    for N in (12, 16):
        p_new = Plate(**plate_kwargs(N=N, mu=1.0))
        p_old = Plate(**plate_kwargs(N=N, mu=1.0))
        p_old.B = _biharmonic_before_2026_08_28(p_old.mask, p_old.h)
        assert not p_old.B.has_sorted_indices, (
            f"N={N}: the injected operator is canonically sorted, so both plates now carry the "
            "shipped B and this test compares one plate against a copy of itself"
        )
        u0 = 1e-4 * rng.standard_normal(p_new.n_live)
        v0 = np.zeros(p_new.n_live)
        drifts = []
        states = []
        for plate in (p_new, p_old):
            plate.set_state(u0, v0)
            e0 = plate.energy()
            worst = 0.0
            for _ in range(2000):
                plate.step()
                worst = max(worst, abs(plate.energy() / e0 - 1.0))
            drifts.append(worst)
            states.append(plate.state.copy())
        amp = np.abs(states[0]).max()
        moved = np.abs(states[0] - states[1]).max() / amp
        assert moved < 1e-11, f"N={N}: the shipped plate moved by {moved:.2e} of its amplitude"
        for drift, which in zip(drifts, ("canonical", "pre-port"), strict=True):
            assert drift < 1e-10, f"N={N}: the {which} operator drifts {drift:.2e}"


def test_the_free_plate_was_not_touched_at_all():
    """The free branch's ``K`` comes back from SciPy already sorted, so the sort is a no-op there.

    Which is why the free plate, the orthotropic free plate and the guitar plate keep their shipped
    numbers to the bit across this batch, and only the supported plate's last digits moved. Asserted
    rather than assumed: if a SciPy release started returning a Gram product unsorted, the free
    plate would join the supported one in moving, and this is where that would surface.

    Asserted on the **builder**, not on ``Plate.K``. The claim belongs to ``operators2d`` -- it is
    about what SciPy's kernel returns -- and routing it through the model would let a future change
    in how ``Plate`` stores its stiffness quietly empty the test, which is the shape of the parity
    section that stopped comparing anything in plan section 23.6.
    """
    from physsynth.core.operators2d import (
        free_plate_stiffness,
        free_plate_stiffness_from_mask,
        grid_coords,
        guitar_mask,
        prune_to_area_carrying,
    )

    built = []
    for N in (8, 12, 16):
        built.append(free_plate_stiffness(N, N, 1.0 / N, 0.3)[0])
        X, Y, h = grid_coords(N, 0.5)
        mask, _ = prune_to_area_carrying(guitar_mask(X, Y + 0.5, 1.0, 0.62, 0.42, 0.30))
        if mask.sum():
            built.append(free_plate_stiffness_from_mask(mask, h, 0.3)[0])
    for K in built:
        K = K.tocsr()
        for r in range(K.shape[0]):
            idx = K.indices[K.indptr[r] : K.indptr[r + 1]]
            assert np.all(np.diff(idx) > 0), (
                "SciPy no longer returns this Gram product in canonical order -- the free plate "
                "now needs `portable.canonical` too, and its shipped numbers are about to move"
            )
