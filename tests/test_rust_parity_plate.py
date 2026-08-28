"""Rust parity for ``physsynth.core.plate`` — models #5, #5b, #5o, #5of, #5g and #6 (plan §28).

Both classes are measured here because the suite binds them into one. ``test_vk_energy.py`` and
``test_vk_free.py`` each assert that ``VKPlate(nonlinear=False)`` is ``array_equal`` to ``Plate``
over 150 steps *and* that their energies compare ``==``, and this port moves the solve — SuperLU
there, the crate's own sparse LU here, which §24.2 settled cannot agree to the bit. Swapping one
class and not the other would break that anchor for a reason having nothing to do with either
model. That is §15.2's finding reaching two classes in one file.

**Three kinds of claim live here, and keeping them apart is the point of the file.**

* Everything built at construction — the grids, the masks, the index maps, ``B``, ``K``, ``W``,
  ``L`` — is **bit-identical**, every branch and every grid. So is the consistent second-order
  start on the linear branches.
* The **trajectory** is not, and it is not supposed to be: the solve differs. What keeps that from
  swallowing a transcription error in the right-hand side is §24.4's manoeuvre for the third time —
  drive the *Python* model through the *Rust* factorization and the two go bit-identical, at every
  branch and with an external force applied. There are four spellings of that right-hand side and
  the shared-solver test is the only thing that can see a reassociation in any of them.
* The **agreement window** is read differently on each of the three branches. See
  ``test_the_three_parity_bars_are_three_different_measurements``.

``energy()`` and ``pressure()`` are ``np.dot`` and ``np.sum``: §14.2's BLAS reduction and NumPy's
pairwise blocking, neither reproduced, both read-outs that reach no timestep. They are measured as
tolerances, and the tolerance is asserted to be a *last bit* rather than merely small.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import sparse

from physsynth.core import operators2d
from physsynth.core.plate import (
    THETA_DEFAULT,
    PlatePy,
    VKPlatePy,
)
from physsynth.core.plate import (
    grain_ratios_from_material_py as grain_ratios_py,
)

# NOT a bare `import physsynth_rs`. The default gate -- the sharded validation harness and the
# shard-reconciliation step beside it -- does **not** build the extension, so a module-scope import
# is a collection ERROR there rather than a skip, and it takes the whole shard down with it. That
# is invisible on a development machine, where the extension is always installed; it is pinned in
# `tests/test_ci_workflow.py` instead.
physsynth_rs = pytest.importorskip(
    "physsynth_rs", reason="the Rust extension is not built in this environment"
)

FS = 20_000.0
MAT = dict(E=2.0e11, e=1.0e-3, nu=0.3, rho=7860.0)

# The four branches, each with a grid the shipped tests actually build on. A guitar is here because
# it is the only outline that prunes, and a circle because it is the only one whose mask comes from
# a comparison against a radius (§25.2 -- a discrete output).
BRANCHES = [
    ("supported", "rectangle", 12, {}),
    ("supported", "rectangle", 20, {}),
    ("supported", "rectangle", 16, dict(grain_x=1.0, grain_cross=0.5, grain_y=0.3)),
    ("free", "rectangle", 12, {}),
    ("free", "rectangle", 16, dict(grain_coupling=0.0269, grain_torsion=0.063, grain_y=0.0727)),
    ("free", "circle", 16, {}),
    ("free", "guitar", 20, {}),
    ("free", "guitar", 28, {}),
]

SCALARS = (
    "Lx Ly kappa rho fs N sigma theta nu waist asym grain_x grain_cross grain_y "
    "grain_coupling grain_torsion grain_is_isotropic k h mu n_live n_pruned prune_depth_max"
).split()

# The three construction-time numbers that are NOT bit-identical, and the only three. `area` is
# `W.diagonal().sum()`, NumPy pairwise blocking; `outline_area` on a guitar is `guitar_area`, the
# two-million-point midpoint quadrature plan section 25 already declined to transcribe; and the
# deficit is their quotient. Nothing branches on any of them and none reaches a timestep.
REDUCED_SCALARS = ["area", "outline_area", "area_deficit"]


def kwargs(boundary, domain, n, grain, **over):
    kw = dict(
        Lx=0.4, Ly=0.4, kappa=1.0, rho=2.0, fs=FS, N=n, boundary=boundary, domain=domain, **grain
    )
    kw.update(over)
    return kw


def pair(boundary, domain, n, grain, **over):
    """The same plate built twice — Python, then Rust."""
    kw = kwargs(boundary, domain, n, grain, **over)
    return PlatePy(**kw), physsynth_rs.Plate(**kw)


def vk_pair(boundary, n, **over):
    kw = dict(Lx=0.4, Ly=0.4, fs=48_000.0, N=n, boundary=boundary, **MAT)
    kw.update(over)
    return VKPlatePy(**kw), physsynth_rs.VKPlate(**kw)


def strike(m, amp, width=0.08):
    """A centred raised Gaussian over the live nodes, peak ``amp`` — the suite's own shape."""
    dx = m.X - 0.5 * m.Lx
    dy = m.Y - 0.5 * m.Ly
    s = amp * np.exp(-((dx * dx + dy * dy) / (width * width)))
    s[~m.mask] = 0.0
    return s


def same_matrix(a, b):
    a, b = sparse.csr_matrix(a), sparse.csr_matrix(b)
    return (
        a.shape == b.shape
        and a.nnz == b.nnz
        and np.array_equal(a.indptr, b.indptr)
        and np.array_equal(a.indices, b.indices)
        and np.array_equal(a.data, b.data)
    )


class RustLu:
    """``splu(...)``'s one method, from Rust — §24.4's shared-factorization hook, third use."""

    def __init__(self, m):
        m = sparse.csr_matrix(m, copy=True)
        m.sort_indices()
        self.lu = physsynth_rs.SparseLu(
            m.data, m.indices.astype(np.int32), m.indptr.astype(np.int32), m.shape[0]
        )

    def solve(self, b):
        return np.asarray(self.lu.solve(np.ascontiguousarray(b, dtype=float)))


def a_bare(p):
    """The system matrix, reassembled exactly as ``airbox._PlateSurface.a_bare`` does."""
    sk = p.sigma * p.k
    coeff = p.theta * p.k * p.k * p.kappa * p.kappa
    if p.boundary == "supported":
        return (1.0 + sk) * sparse.identity(p.n_live, format="csc") + coeff * p.B
    return (1.0 + sk) * p.W + coeff * p.K


def share_solver(py):
    """Put the Python plate on the Rust factorization, so the only residue is everything else."""
    py._lu = RustLu(a_bare(py))
    return py


# -- construction: bit-identical, every branch ---------------------------------------------------


@pytest.mark.parametrize("boundary,domain,n,grain", BRANCHES)
def test_every_derived_scalar_is_bit_identical(boundary, domain, n, grain):
    py, rs = pair(boundary, domain, n, grain)
    for name in SCALARS:
        a, b = getattr(py, name), getattr(rs, name)
        assert a == b, f"{name}: {a!r} != {b!r}"


@pytest.mark.parametrize("boundary,domain,n,grain", BRANCHES)
def test_the_quadrature_area_is_a_reduction_and_is_measured_as_one(boundary, domain, n, grain):
    """``area`` is the one construction-time number that is not bit-identical.

    It is ``W.diagonal().sum()`` -- NumPy pairwise blocking, which no portable loop reproduces and
    which this port declines to transcribe, exactly as ``ops2d.guitar_area`` declined it. Nothing
    branches on it and it reaches no timestep: it is divided into the outline area to report how
    converged a staircase is.

    A rectangle is where the difference is most visible and least consequential. The trapezoidal
    weights really do sum to ``Lx*Ly`` there, and in NumPy order they land on it **exactly**, so
    the Python deficit is a literal ``0.0`` while this one is a last bit away from it. The shipped
    bar on that is ``abs=1e-14`` (``test_guitar_plate.py``) and both sides clear it by an order.
    """
    py, rs = pair(boundary, domain, n, grain)
    for name in REDUCED_SCALARS:
        assert getattr(rs, name) == pytest.approx(getattr(py, name), rel=1e-13), name
    if domain == "rectangle":
        assert py.area_deficit == 0.0, "the rectangle quadrature is exactly right in NumPy order"
        assert abs(rs.area_deficit) < 1e-14


@pytest.mark.parametrize("boundary,domain,n,grain", BRANCHES)
def test_the_grid_the_mask_and_the_index_map_are_bit_identical(boundary, domain, n, grain):
    """A last bit here is not a rounding but a **different plate** — §25.2.

    ``Ly`` is snapped through ``int(round(Ly/h))``, which is half-to-**even** in Python and
    half-away-from-zero in Rust; the node coordinates come from a ``linspace`` whose last entry is
    *overwritten* rather than computed; and on a curved rim one ulp of a coordinate is a live node
    or a dead one. None of that is visible to an energy bar.
    """
    py, rs = pair(boundary, domain, n, grain)
    assert np.array_equal(py.X, rs.X) and py.X.dtype == rs.X.dtype
    assert np.array_equal(py.Y, rs.Y)
    assert np.array_equal(py.mask, rs.mask)
    assert np.array_equal(py.index_map, rs.index_map)
    assert py.mask.shape == rs.mask.shape


@pytest.mark.parametrize("boundary,domain,n,grain", BRANCHES)
def test_the_operators_are_bit_identical(boundary, domain, n, grain):
    py, rs = pair(boundary, domain, n, grain)
    if boundary == "supported":
        assert same_matrix(py.B, rs.B), "the biharmonic moved"
        assert same_matrix(py.L, rs.L), "the Laplacian moved"
    else:
        assert same_matrix(py.K, rs.K), "the stiffness moved"
        assert same_matrix(py.W, rs.W), "the lumped mass moved"
        assert np.array_equal(py.w, rs.w)


@pytest.mark.parametrize("boundary,domain,n,grain", BRANCHES)
def test_the_second_order_start_is_bit_identical(boundary, domain, n, grain):
    """``u^{-1}`` and the seeded acceleration involve no solve, so they are exact."""
    py, rs = pair(boundary, domain, n, grain)
    ic = strike(py, 1e-4)
    for v0 in (0.0, 3.7, strike(py, 2e-3)):
        py.set_state(ic, v0)
        rs.set_state(ic, v0)
        assert np.array_equal(py.u, rs.u)
        assert np.array_equal(py.u_prev, rs.u_prev)
        assert np.array_equal(py._accel, rs._accel)
        assert py.n == rs.n == 0


@pytest.mark.parametrize("boundary,domain,n,grain", BRANCHES)
def test_a_live_vector_start_is_accepted_the_same_way(boundary, domain, n, grain):
    py, rs = pair(boundary, domain, n, grain)
    rng = np.random.default_rng(4)
    u0 = 1e-4 * rng.standard_normal(py.n_live)
    v0 = 1e-2 * rng.standard_normal(py.n_live)
    py.set_state(u0, v0)
    rs.set_state(u0, v0)
    assert np.array_equal(py.u_prev, rs.u_prev)


# -- §24.4: hold the solver constant, and the whole step is exact --------------------------------


@pytest.mark.parametrize("boundary,domain,n,grain", BRANCHES)
def test_on_a_shared_factorization_the_trajectory_is_bit_identical(boundary, domain, n, grain):
    """The claim the port is actually making, and the only test that can see a reassociation.

    With the solve held constant, everything else in a step — two matvecs, five scalar
    coefficients, the acceleration cache — has to agree to the bit or this fails. §24.4's
    manoeuvre, third use.
    """
    py, rs = pair(boundary, domain, n, grain)
    share_solver(py)
    ic = strike(py, 1e-4)
    py.set_state(ic)
    rs.set_state(ic)
    for step in range(400):
        py.step()
        rs.step()
        assert np.array_equal(py.u, rs.u), f"state diverged at step {step}"
    assert np.array_equal(py._accel, rs._accel), "the acceleration cache diverged"


@pytest.mark.parametrize("boundary,domain,n,grain", BRANCHES)
def test_the_external_force_path_is_exact_too(boundary, domain, n, grain):
    """The second and fourth spellings of the right-hand side.

    ``f_ext`` divides by ``rho h^2`` on the supported branch and by ``rho`` on the free one, where
    ``W`` lives inside ``A`` and is divided out by the solve. Getting that backwards is a factor of
    ``h^2`` that every energy bar would pass, because the scheme stays exactly conservative either
    way — the plate would simply be driven by the wrong force.
    """
    py, rs = pair(boundary, domain, n, grain)
    share_solver(py)
    rng = np.random.default_rng(17)
    ic = strike(py, 1e-4)
    py.set_state(ic)
    rs.set_state(ic)
    for step in range(60):
        f = rng.standard_normal(py.n_live)
        py.step(f_ext=f)
        rs.step(f_ext=f)
        assert np.array_equal(py.u, rs.u), f"forced state diverged at step {step}"
    assert py.energy() == pytest.approx(rs.energy(), rel=1e-14)


def test_a_zero_force_is_the_same_as_no_force_in_both_implementations():
    """Adding ``k^2 * 0 / den`` must not perturb anything — a cheap check that the term is added
    rather than the whole right-hand side re-derived."""
    for boundary in ("supported", "free"):
        py, rs = pair(boundary, "rectangle", 12, {})
        share_solver(py)
        ic = strike(py, 1e-4)
        py.set_state(ic)
        rs.set_state(ic)
        zero = np.zeros(py.n_live)
        for _ in range(30):
            py.step()
            rs.step(f_ext=zero)
        assert np.array_equal(py.u, rs.u)


# -- the reductions ------------------------------------------------------------------------------


@pytest.mark.parametrize("boundary,domain,n,grain", BRANCHES)
def test_the_energy_differs_by_a_last_bit_and_that_is_the_reduction(boundary, domain, n, grain):
    """``energy()`` is three ``np.dot``s and a fourth for the kinetic term.

    §14.2 retired bit-identity for a BLAS reduction, and this is the same thing on a read-out. With
    the solve held constant the *state* is identical, so whatever is left is the summation order
    alone — and it has to be a last bit, not merely small.
    """
    py, rs = pair(boundary, domain, n, grain)
    share_solver(py)
    ic = strike(py, 1e-4)
    py.set_state(ic)
    rs.set_state(ic)
    worst = 0.0
    for _ in range(200):
        py.step()
        rs.step()
        assert np.array_equal(py.u, rs.u)
        a, b = py.energy(), rs.energy()
        worst = max(worst, abs(a - b) / abs(a))
    assert worst < 1e-14, f"the energy reduction differs by {worst:.3e}, which is not a last bit"


@pytest.mark.parametrize("boundary,domain,n,grain", BRANCHES)
def test_the_pressure_readout_differs_by_a_last_bit(boundary, domain, n, grain):
    """``np.sum`` is pairwise above eight elements on the supported branch and ``np.dot`` on the
    free one. Neither is reproduced -- see ``ops2d.guitar_area``, which declined the same bargain
    -- and nothing feeds this back: the radiation tiers read it, and the room-loaded ones couple
    through the system matrix instead.

    **A free plate is driven here and a supported one is not, and that is the finding.** An
    unforced free plate monopole is *identically zero*: its stiffness annihilates the constant
    vector, so the weighted sum of accelerations vanishes term for term and what floating point
    returns is the cancellation residue -- measured at 1e-16 to 1e-13 of the sum of the absolute
    terms, against 0.16 to 0.30 for a supported plate. A *relative* comparison of that residue is
    a comparison of two roundings and means nothing. An external force breaks the nullspace and
    the read-out becomes a real number again (0.53 of the terms), which is where it can be
    compared. It is also why this family has dipole classes at all.
    """
    py, rs = pair(boundary, domain, n, grain)
    share_solver(py)
    rng = np.random.default_rng(23)
    ic = strike(py, 1e-4)
    py.set_state(ic)
    rs.set_state(ic)
    forced = boundary == "free"
    worst = 0.0
    for _ in range(100):
        f = rng.standard_normal(py.n_live) if forced else None
        py.step(f_ext=f)
        rs.step(f_ext=f)
        assert np.array_equal(py.u, rs.u)
        weights = py.w if boundary == "free" else np.full(py.n_live, py.h * py.h)
        scale = float(np.abs(weights * py._accel).sum())
        worst = max(worst, abs(py.pressure() - rs.pressure()) / scale)
    assert worst < 1e-15, f"pressure differs by {worst:.3e} of the terms that went into it"


@pytest.mark.parametrize("boundary,domain,n,grain", BRANCHES)
def test_an_unforced_free_plate_has_no_monopole_at_all(boundary, domain, n, grain):
    """The claim the test above leans on, asserted rather than assumed.

    A free plate radiates nothing as a simple source until something pushes on it; a supported one
    does, because its rim carries the reaction. Both implementations agree about which case they
    are in, which is the only cross-implementation claim available on a quantity that is zero.
    """
    py, rs = pair(boundary, domain, n, grain)
    share_solver(py)
    ic = strike(py, 1e-4)
    py.set_state(ic)
    rs.set_state(ic)
    weights = py.w if boundary == "free" else np.full(py.n_live, py.h * py.h)
    worst_py = worst_rs = 0.0
    for _ in range(50):
        py.step()
        rs.step()
        scale = np.abs(weights * py._accel).sum()
        worst_py = max(worst_py, abs(py.pressure()) / scale)
        worst_rs = max(worst_rs, abs(rs.pressure()) / scale)
    if boundary == "free":
        assert worst_py < 1e-11 and worst_rs < 1e-11, (
            f"the free plate monopole should cancel: {worst_py:.2e} / {worst_rs:.2e}"
        )
    else:
        assert worst_py > 1e-3 and worst_rs > 1e-3, (
            f"a supported plate monopole should not cancel: {worst_py:.2e} / {worst_rs:.2e}"
        )


# -- the three parity bars -----------------------------------------------------------------------


def rigid_elastic_split(p, field):
    """Project a free plate's live field onto ``span{1, x, y}`` and its complement.

    §24.5's measurement, in two dimensions. The free plate's stiffness annihilates that span, so
    along it the plate is a **free particle** and a per-step solver difference is integrated twice
    — it grows like ``t^2`` while the elastic part only random-walks. A bar that read
    ``max|du|/amp`` on a free-edge model would therefore be measuring the rigid part and nothing
    else.
    """
    live = p.index_map >= 0
    basis = np.column_stack(
        [np.ones(p.n_live), p.X[live], p.Y[live]]
    )
    q, _ = np.linalg.qr(basis)
    rigid = q @ (q.T @ field)
    return float(np.abs(rigid).max()), float(np.abs(field - rigid).max())


def test_the_three_parity_bars_are_three_different_measurements():
    """The batch's own finding, asserted rather than only written down.

    A supported plate random-walks, a free plate's rigid part integrates the same per-step gap
    twice, and a von Karman plate must be read on its energy. Three bars for one file.
    """
    # 1. supported and linear: a random walk, still tiny at 4,000 steps.
    py, rs = pair("supported", "rectangle", 12, {})
    ic = strike(py, 1e-4)
    py.set_state(ic)
    rs.set_state(ic)
    for _ in range(4000):
        py.step()
        rs.step()
    amp = np.abs(py.u).max()
    walk = np.abs(py.u - rs.u).max() / amp
    assert walk < 1e-10, f"the supported plate's random walk reached {walk:.3e}"

    # 2. free and linear: the rigid part is a free particle and dominates the elastic part.
    py, rs = pair("free", "rectangle", 12, {})
    ic = strike(py, 1e-4)
    py.set_state(ic)
    rs.set_state(ic)
    for _ in range(4000):
        py.step()
        rs.step()
    rigid, elastic = rigid_elastic_split(py, py.u - rs.u)
    assert rigid > 10.0 * elastic, (
        f"the rigid part ({rigid:.3e}) should dominate the elastic one ({elastic:.3e}) -- §24.5"
    )
    # ... and the *energy* is unmoved through all of it, which is why it is the bar.
    assert py.energy() == pytest.approx(rs.energy(), rel=1e-9)


@pytest.mark.parametrize("boundary,domain,n,grain", BRANCHES)
def test_the_energy_drift_bar_is_met_by_both(boundary, domain, n, grain):
    """The project's acceptance contract, on both implementations of every branch."""
    py, rs = pair(boundary, domain, n, grain)
    for model in (py, rs):
        model.set_state(strike(model, 1e-4))
        e0 = model.energy()
        worst = 0.0
        for _ in range(600):
            model.step()
            worst = max(worst, abs(model.energy() / e0 - 1.0))
        assert worst < 1e-10, f"drift {worst:.3e}"


# -- the read-outs that are not floats -----------------------------------------------------------


@pytest.mark.parametrize("boundary,domain,n,grain", BRANCHES)
def test_the_discrete_readouts_agree_exactly(boundary, domain, n, grain):
    """``pickup_index_at`` returns an **index**, and ``np.argmin`` breaks ties at the first
    minimum. The squares are spelled ``x*x`` on both sides because the original squares a NumPy
    *array*, where ``** 2`` is the ufunc ladder's multiply and not a ``pow`` call (§16.2)."""
    py, rs = pair(boundary, domain, n, grain)
    rng = np.random.default_rng(9)
    for _ in range(40):
        x, y = rng.uniform(-0.05, 0.45, size=2)
        assert py.pickup_index_at(x, y) == rs.pickup_index_at(x, y)
    # A tie: exactly between two nodes along x.
    assert py.pickup_index_at(0.5 * py.h, 0.0) == rs.pickup_index_at(0.5 * rs.h, 0.0)


@pytest.mark.parametrize("boundary,domain,n,grain", BRANCHES)
def test_to_live_and_state_round_trip_identically(boundary, domain, n, grain):
    py, rs = pair(boundary, domain, n, grain)
    field = strike(py, 3e-4)
    assert np.array_equal(py.to_live(field), rs.to_live(field))
    py.set_state(field)
    rs.set_state(field)
    assert np.array_equal(py.state, rs.state)
    assert py.state.shape == rs.state.shape
    assert py.displacement_at(0) == rs.displacement_at(0)
    assert py.displacement_at(-1) == rs.displacement_at(-1)


def test_a_wrongly_shaped_field_is_refused_with_the_same_message():
    py, rs = pair("supported", "rectangle", 12, {})
    bad = np.zeros((3, 4))
    with pytest.raises(ValueError) as a:
        py.to_live(bad)
    with pytest.raises(ValueError) as b:
        rs.to_live(bad)
    assert str(a.value) == str(b.value)


# -- the refusals, message for message ------------------------------------------------------------

REFUSALS = [
    dict(Lx=0.0),
    dict(fs=-1.0),
    dict(kappa=0.0),
    dict(rho=-2.0),
    dict(N=1),
    dict(sigma=-1e-9),
    dict(theta=0.0),
    dict(theta=1.5),
    dict(boundary="clamped"),
    dict(boundary=7),
    dict(domain="triangle"),
    dict(domain="circle", boundary="supported"),
    dict(domain="guitar", boundary="supported"),
    dict(nu=0.7),
    dict(nu=-1.0),
    dict(grain_x=0.0),
    dict(grain_y=-1.0),
    dict(grain_cross=-1.01),
    dict(grain_coupling=0.3),
    dict(grain_torsion=0.4),
    dict(boundary="free", grain_coupling=1.5, grain_torsion=0.5),
    dict(boundary="free", grain_coupling=0.3, grain_torsion=0.0),
    dict(boundary="free", grain_y=0.3),
    dict(boundary="free", nu=0.2, grain_coupling=0.3, grain_torsion=0.2),
    dict(boundary="free", grain_coupling=0.0269, grain_torsion=0.063, grain_cross=0.9),
    dict(domain="guitar", boundary="free", N=4, waist=0.95),
]


@pytest.mark.parametrize("over", REFUSALS, ids=lambda d: ",".join(f"{k}={v}" for k, v in d.items()))
def test_a_refused_plate_is_refused_with_the_original_message(over):
    kw = dict(Lx=0.4, Ly=0.4, kappa=1.0, rho=2.0, fs=FS, N=12)
    kw.update(over)
    with pytest.raises(ValueError) as a:
        PlatePy(**kw)
    with pytest.raises(ValueError) as b:
        physsynth_rs.Plate(**kw)
    assert str(a.value) == str(b.value)


def test_an_explicit_none_boundary_is_refused_rather_than_defaulted():
    """§24.7, which arrived as a silent default in every binding: PyO3 collapses an omitted
    argument and an explicit ``None`` unless the signature is written to keep them apart."""
    kw = dict(Lx=0.4, Ly=0.4, kappa=1.0, rho=2.0, fs=FS, N=12)
    for extra in (dict(boundary=None), dict(domain=None)):
        with pytest.raises(ValueError) as a:
            PlatePy(**kw, **extra)
        with pytest.raises(ValueError) as b:
            physsynth_rs.Plate(**kw, **extra)
        assert str(a.value) == str(b.value)
    # ... while OMITTING them builds the documented default.
    assert physsynth_rs.Plate(**kw).boundary == "supported"
    assert physsynth_rs.Plate(**kw).domain == "rectangle"


# -- the material helper ---------------------------------------------------------------------------

MATERIALS = [
    dict(E_x=11.0e9, E_y=0.8e9, nu_xy=0.37, G_xy=0.7e9, thickness=3.0e-3, rho=420.0),
    dict(E_x=1.1e10, E_y=1.1e10, nu_xy=0.3, G_xy=1.1e10 / 2.6, thickness=3.0e-3, rho=420.0),
    dict(E_x=2.0e11, E_y=2.0e11, nu_xy=0.28, G_xy=7.8e10, thickness=1.0e-3, rho=7860.0),
    dict(E_x=9.0e9, E_y=0.5e9, nu_xy=0.4, G_xy=0.6e9, thickness=2.5e-3, rho=390.0),
]


@pytest.mark.parametrize("mat", MATERIALS)
def test_the_material_helper_is_bit_identical(mat):
    """``thickness ** 3`` is a Python ``float`` power, i.e. the C library's ``pow`` — not a chain
    of multiplies, and not something LLVM may fold into one (§17.2). Every one of the seven fields
    is compared, because ``grain_cross`` is taken from ``H / D_x`` rather than recombined and that
    is what lands *exactly* 1.0 for isotropic material."""
    a = grain_ratios_py(**mat)
    b = physsynth_rs.grain_ratios_from_material(**mat)
    assert type(a) is type(b)
    for name in a._fields:
        assert getattr(a, name) == getattr(b, name), name


def test_the_material_helper_refuses_the_same_things():
    # The last one has to make `nu_xy * nu_yx >= 1`, and `nu_yx = nu_xy E_y / E_x` -- so it is
    # the RATIO that has to be extreme, not the modulus. A first draft set `E_y = 40` beside an
    # `E_x` of 11 GPa and refused nothing at all.
    for over in (
        dict(E_x=0.0),
        dict(thickness=-1.0),
        dict(rho=0.0),
        dict(E_x=1.0, E_y=40.0, nu_xy=0.9),
    ):
        mat = dict(MATERIALS[0])
        mat.update(over)
        with pytest.raises(ValueError) as a:
            grain_ratios_py(**mat)
        with pytest.raises(ValueError) as b:
            physsynth_rs.grain_ratios_from_material(**mat)
        assert str(a.value) == str(b.value)


# -- the nonlinear plate ---------------------------------------------------------------------------

VK_BRANCHES = [("supported", 12), ("supported", 16), ("free", 12), ("free", 16)]


@pytest.mark.parametrize("boundary,n", VK_BRANCHES)
def test_the_nonlinear_plates_construction_is_bit_identical(boundary, n):
    py, rs = vk_pair(boundary, n)
    for name in "E e nu rho_v rho_s D kappa Y_mem fs N Ny Lx Ly sigma theta k h mu".split():
        assert getattr(py, name) == getattr(rs, name), name
    for name in "nonlinear couple_tol couple_max_iter n_live n_nodes force_denominator".split():
        assert getattr(py, name) == getattr(rs, name), name
    assert np.array_equal(py.mask, rs.mask)
    assert np.array_equal(py.index_map, rs.index_map)
    if boundary == "supported":
        assert same_matrix(py.B, rs.B)
    else:
        assert same_matrix(py.K, rs.K)
        assert np.array_equal(py.wdiag, rs.wdiag)
    # The bracket and the Airy solver come back as the objects `airbox.py` calls.
    assert same_matrix(py.bracket.Sxx, rs.bracket.Sxx)
    assert same_matrix(py.airy.Bf, rs.airy.Bf)


@pytest.mark.parametrize("boundary,n", VK_BRANCHES)
def test_a_linear_nonlinear_plate_is_the_linear_plate_bit_for_bit(boundary, n):
    """§15.2's anchor, asserted **within each implementation**.

    This is what forces the two classes to swap together: the port moves the solve, so a Rust
    ``VKPlate`` against a Python ``Plate`` would differ, and the shipped ``test_vk_energy.py``
    assertion is an ``array_equal``. Here it is checked on both sides so the constraint is visible
    in the parity file rather than only in its consequence.
    """
    for Model, Linear in ((VKPlatePy, PlatePy), (physsynth_rs.VKPlate, physsynth_rs.Plate)):
        vk = Model(Lx=0.4, Ly=0.4, fs=48_000.0, N=n, boundary=boundary, nonlinear=False, **MAT)
        p5 = Linear(
            Lx=0.4, Ly=0.4, kappa=vk.kappa, rho=vk.rho_s, fs=48_000.0, N=n, boundary=boundary
        )
        ic = strike(vk, 1e-4)
        vk.set_state(ic)
        p5.set_state(ic)
        for step in range(150):
            vk.step()
            p5.step()
            assert np.array_equal(vk.state, p5.state), f"{Model.__name__}: step {step}"
        assert vk.energy() == p5.energy()


@pytest.mark.parametrize("boundary,n", VK_BRANCHES)
def test_the_nonlinear_seam_airbox_steps_through_is_exact(boundary, n):
    """``_to_full``, ``_to_live``, ``_airy_F`` and ``_linear_rhs`` — private by name and public by
    use. ``airbox._VKPlateSurface`` calls all four per Picard sweep, so they are part of the
    interface whether or not the underscore says so (§12.2).

    Three of them are exact; ``_airy_F`` is a Group D solve and is a tolerance, which is why it is
    also checked on a *shared* factorization below.
    """
    py, rs = vk_pair(boundary, n)
    ic = strike(py, 3e-4)
    py.set_state(ic)
    rs.set_state(ic)
    # `_linear_rhs` reads `u` and `u_prev`, and a *nonlinear* start already differs between the two
    # (its coupling term goes through the Airy solve, which is Group D). Align the state first, or
    # this measures the start rather than the right-hand side.
    py.u, py.u_prev = np.array(rs.u), np.array(rs.u_prev)
    rng = np.random.default_rng(2)
    live = 1e-4 * rng.standard_normal(py.n_live)
    assert np.array_equal(py._to_full(live), rs._to_full(live))
    full = rng.standard_normal(py.n_nodes)
    assert np.array_equal(py._to_live(full), rs._to_live(full))
    assert np.array_equal(py._linear_rhs(), rs._linear_rhs())
    w = py._to_full(live)
    a, b = py._airy_F(w), rs._airy_F(w)
    scale = max(np.abs(a).max(), 1e-300)
    assert np.abs(a - b).max() / scale < 1e-9


@pytest.mark.parametrize("boundary,n", VK_BRANCHES)
def test_on_a_shared_factorization_the_nonlinear_plate_is_bit_identical(boundary, n):
    """§24.4 again, and here it has to hold **two** factorizations constant — the theta-scheme's
    and the Airy solve's — because a von Karman step calls both, once and once per sweep."""
    py, rs = vk_pair(boundary, n)
    # With the flag set, `VKPlatePy.__init__` looks up the module-global `AiryStressSolver`, which
    # the swap has already rebound to the Rust class -- so `py.airy` would BE Rust and this test
    # would compare Rust against Rust while staying green. Section 23.6, reached through the
    # constructor of an unswapped class rather than through a test pinning a name. The fix is to
    # give the Python model the Python operators explicitly.
    py.bracket = operators2d.VonKarmanBracketPy(py.N, py.Ny, py.h)
    py.airy = operators2d.AiryStressSolverPy(py.N, py.Ny, py.h)
    py._lu = RustLu(a_bare_vk(py))
    py.airy._lu = RustLu(py.airy.Bf)
    ic = strike(py, 3.0 * MAT["e"])
    py.set_state(ic)
    rs.set_state(ic)
    assert np.array_equal(py.u_prev, rs.u_prev), "the coupled start diverged"
    assert np.array_equal(py.F, rs.F)
    assert np.array_equal(py.F_prev, rs.F_prev)
    for step in range(120):
        py.step()
        rs.step()
        assert np.array_equal(py.u, rs.u), f"state diverged at step {step}"
        assert py.n_iters == rs.n_iters, f"the Picard count diverged at step {step}"
    assert py.converged == rs.converged


def a_bare_vk(p):
    sk = p.sigma * p.k
    coeff = p.theta * p.k * p.k * p.kappa * p.kappa
    if p.boundary == "supported":
        return (1.0 + sk) * sparse.identity(p.n_live, format="csc") + coeff * p.B
    return (1.0 + sk) * p.W + coeff * p.K


@pytest.mark.parametrize("boundary,n", VK_BRANCHES)
def test_the_nonlinear_bar_is_the_energy_and_not_the_displacement(boundary, n):
    """§27.5's conclusion, applied to a whole model rather than to an operator.

    The same plate at the same amplitude either random-walks or decorrelates completely, and which
    one it does is set by how hard the Picard loop is working — not by the fixture and not by the
    amplitude, both of which were tried and falsified. The energy does not move through either, so
    **the energy is the bar**: *a conserved quantity is not a trajectory comparison.*
    """
    py, rs = vk_pair(boundary, n)
    ic = strike(py, 3.0 * MAT["e"])
    py.set_state(ic)
    rs.set_state(ic)
    e0 = py.energy()
    worst_energy = 0.0
    worst_drift = 0.0
    for _ in range(400):
        py.step()
        rs.step()
        a, b = py.energy(), rs.energy()
        worst_energy = max(worst_energy, abs(a - b) / abs(a))
        worst_drift = max(worst_drift, abs(a / e0 - 1.0))
    assert worst_energy < 1e-9, f"the two energies parted by {worst_energy:.3e}"
    assert worst_drift < 1e-10, f"the shipped drift bar moved: {worst_drift:.3e}"


VK_REFUSALS = [
    dict(Lx=0.0),
    dict(E=0.0),
    dict(e=0.0),
    dict(rho=-1.0),
    dict(N=1),
    dict(sigma=-1.0),
    dict(theta=0.0),
    dict(nu=0.5),
    dict(nu=-1.0),
    dict(boundary="clamped"),
    dict(couple_tol=0.0),
    dict(couple_max_iter=0),
]


@pytest.mark.parametrize(
    "over", VK_REFUSALS, ids=lambda d: ",".join(f"{k}={v}" for k, v in d.items())
)
def test_a_refused_nonlinear_plate_matches_the_original_message(over):
    kw = dict(Lx=0.4, Ly=0.4, fs=48_000.0, N=12, **MAT)
    kw.update(over)
    with pytest.raises(ValueError) as a:
        VKPlatePy(**kw)
    with pytest.raises(ValueError) as b:
        physsynth_rs.VKPlate(**kw)
    assert str(a.value) == str(b.value)


def test_the_defaults_are_the_same_defaults():
    """``THETA_DEFAULT`` is a module constant on the Python side and a Rust one here; a plate
    built with neither ``theta`` nor ``sigma`` must land on the same numbers."""
    kw = dict(Lx=0.4, Ly=0.4, kappa=1.0, rho=2.0, fs=FS, N=12)
    assert physsynth_rs.Plate(**kw).theta == THETA_DEFAULT == PlatePy(**kw).theta
    assert physsynth_rs.Plate(**kw).sigma == 0.0
    assert physsynth_rs.Plate(**kw).waist == PlatePy(**kw).waist
    assert physsynth_rs.Plate(**kw).asym == PlatePy(**kw).asym
    vk = dict(Lx=0.4, Ly=0.4, fs=48_000.0, N=12, **MAT)
    assert physsynth_rs.VKPlate(**vk).couple_tol == VKPlatePy(**vk).couple_tol
    assert physsynth_rs.VKPlate(**vk).couple_max_iter == VKPlatePy(**vk).couple_max_iter
    assert physsynth_rs.VKPlate(**vk).nonlinear is True


# -- the branch-only attributes -----------------------------------------------------------------


def test_a_branch_only_attribute_is_absent_on_the_other_branch():
    """A free plate has no ``B`` and a supported one has no ``K``/``W``/``w``, in the original
    because they were never assigned. A binding that offered all of them would be a wider
    interface than the model, and code that branches on ``hasattr`` would take the wrong arm."""
    for Model in (PlatePy, physsynth_rs.Plate):
        kw = dict(Lx=0.4, Ly=0.4, kappa=1.0, rho=2.0, fs=FS, N=12)
        sup = Model(**kw, boundary="supported")
        free = Model(**kw, boundary="free")
        assert hasattr(sup, "B") and hasattr(sup, "L")
        assert not hasattr(sup, "K") and not hasattr(sup, "W") and not hasattr(sup, "w")
        assert hasattr(free, "K") and hasattr(free, "W") and hasattr(free, "w")
        assert not hasattr(free, "B") and not hasattr(free, "L")


def test_the_state_buffers_are_settable_because_airbox_writes_them():
    """§12.2, for a third and fourth model. ``airbox._PlateSurface.commit`` assigns ``_accel``,
    ``u``, ``u_prev`` and ``n``; ``_VKPlateSurface`` adds ``F``, ``F_prev`` and the three Picard
    diagnostics. None of that is optional — it is how the room puts its load inside the solve."""
    for Model in (PlatePy, physsynth_rs.Plate):
        p = Model(Lx=0.4, Ly=0.4, kappa=1.0, rho=2.0, fs=FS, N=12)
        fresh = np.arange(p.n_live, dtype=float)
        p._accel = fresh.copy()
        p.u_prev = p.u
        p.u = fresh.copy()
        p.n = 7
        assert np.array_equal(p.u, fresh)
        assert np.array_equal(p._accel, fresh)
        assert p.n == 7
    for Model in (VKPlatePy, physsynth_rs.VKPlate):
        v = Model(Lx=0.4, Ly=0.4, fs=48_000.0, N=12, **MAT)
        fresh = np.arange(v.n_nodes, dtype=float)
        v.F_prev = v.F
        v.F = fresh.copy()
        v.n_iters = 4
        v.converged = False
        v.last_residual = 1e-5
        assert np.array_equal(v.F, fresh)
        assert v.n_iters == 4 and v.converged is False and v.last_residual == 1e-5


def test_the_snapped_side_length_rounds_half_to_even():
    """``Ny = max(int(round(Ly / h)), 1)``. Python's ``round`` is half-to-**even** and Rust's
    ``f64::round`` is half-away-from-zero, and the two disagree on exactly the geometries where
    ``Ly / h`` lands on a half — which is a **different plate**, not a rounding (§25.2)."""
    ties = 0
    for n in (8, 10, 12, 16, 20):
        for mult in (2.5, 3.5, 4.5, 5.5, 6.5):
            ly = (0.4 / n) * mult
            a = PlatePy(Lx=0.4, Ly=ly, kappa=1.0, rho=2.0, fs=FS, N=n)
            b = physsynth_rs.Plate(Lx=0.4, Ly=ly, kappa=1.0, rho=2.0, fs=FS, N=n)
            if math.isclose(ly / (0.4 / n) % 1.0, 0.5, abs_tol=1e-9):
                ties += 1
            assert a.Ly == b.Ly, f"N={n} mult={mult}: {a.Ly} != {b.Ly}"
            assert a.mask.shape == b.mask.shape
    assert ties >= 8, f"only {ties} of these geometries actually land on a tie"
