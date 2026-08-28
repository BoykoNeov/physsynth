"""Rust vs Python for the 2-D builders and the excitation shapes — Phase 2's comparison.

The third of these files (after ``test_rust_parity.py`` and ``test_rust_parity_operators.py``) and
it inherits their bars unchanged: the acceptance contract is the physics harness
(``docs/dev/rust-migration-plan.md`` §2.1), and this is the *diagnostic* that catches what the
physics bars are too loose to see.

What is new here, and what each new thing is worth:

* **A mask is a geometry decision, not a number.** ``disk_mask`` is a strict ``x² + y² < r²``, so a
  node one ulp from the rim changes whether it is an unknown at all — and a membrane with one node
  fewer conserves energy just as beautifully as the right one. Every detector this project owns
  stays green on a wrong mask, which is exactly why the mask is compared elementwise here and not
  through anything downstream of it.

* **``cos`` is the first transcendental in the port.** NumPy does not call the platform libm for
  ``np.cos`` on a float64 array — it has its own vectorised implementation with a ~1 ulp error
  budget — so the raised cosines are the first kernels whose exactness rests on *two*
  implementations agreeing rather than on IEEE-754 alone, the way ``delta_xxxx``'s rests on two
  ``pow``s (plan §10.3). Measured on this machine they agree bit-for-bit; that is asserted rather
  than assumed, and swept over widths and grid sizes so a platform where it stops being true fails
  *here*, with an obvious cause, rather than as a mysterious partial three phases later.

* **The Laplacian is a reduction that is still exact.** Its entries are single stored constants
  (``-4/h²`` and ``1/h²``) with no summation at all, so it is the easy end of §10.3's finding: the
  bit-identity covers ``data``, ``indices``, ``indptr`` and ``nnz``.

* **``inner2d``/``norm2_2d`` may differ at ~1e-15**, because they call ``np.dot`` and BLAS
  accumulates in an order no portable loop reproduces. Held to the plan's Group A target.

Phase 5's first batch adds the guitar outline, and it sharpens the first of those points into a
measurement. ``guitar_mask`` is a strict ``|x| < half(y)`` where ``half`` is built from ``sin`` and
``cos``, so §22.1's finding — NumPy computes transcendentals with its own CPU-dispatched routines
rather than the platform libm — lands directly on a live/dead decision. How much room that leaves
was measured over 130 shipped configurations before the port was written:

* every **real** guitar outline clears the comparison by at least **1.9e7 ulps** of ``half``, so a
  last bit cannot move a node and the mask's exactness is structural rather than lucky;
* the **degenerate lens** (``waist = 0, asym = 0``) does not: four nodes sit **1 ulp** from the
  rim at ``N = 32`` and two sit *exactly* on it, because ``sin(pi/6)`` is ``1/2`` in real
  arithmetic and the grid puts a node there. That case is reachable — the viewer's waist sweep
  starts at ``0.0``.

So ``physsynth/core/operators2d.py`` gives the mask path the *scalar* libm (``math.sin``, which is
where Rust's ``f64::sin`` also lands) and keeps NumPy's vectorised spelling only for the two-million
point area quadrature, which averages rather than branches. Both halves of that split are pinned
below.

Phase 5's **second** batch adds the matrices those masks are for — ``biharmonic_from_mask``,
``orthotropic_biharmonic`` and ``free_plate_stiffness*`` — and every one of them is compared
**exactly**, on ``data``, ``indices``, ``indptr`` and ``nnz``. That took one edit to the reference
and it is worth saying which:

* **The values were already identical; only the stored order was not.** SciPy's sparse product
  hands back each row in whatever order its kernel touched the columns — measured here as neither
  ascending nor descending, and not a property of the algebra. A CSR matvec sums a row in *stored*
  order and :class:`.plate.Plate` forms ``B @ u`` twice per timestep, so that order reaches the
  trajectory. :func:`physsynth.core.portable.canonical` sorts the Python side, which is §18.2's
  manoeuvre for the string family arriving where §18.4 said in advance that it would.

* **``free_plate_stiffness`` needed nothing.** Its ``K`` is a ``AᵀWB`` Gram product, and SciPy
  returns those already sorted — measured canonical in every row of every rectangle, disk and
  guitar tried. So the free plate, the orthotropic free plate and the guitar plate are *unmoved*
  by this batch, and only the supported plate's last bits move.

* **The one association that looks dangerous is provably harmless.** ``operators2d.py`` writes the
  free plate's Gram products right-associated, ``C2xᵀ @ (Wa @ C2y)``, while ``AiryStressSolver``
  writes the same mathematical form with no parentheses at all, which Python left-associates. About
  a third of value triples distinguish those two — but *none* of this operator's do, because every
  curvature entry is the same mantissa ``1/h²`` times an exact power of two and IEEE multiplication
  commutes exactly. That is asserted natively rather than here; see
  ``crates/physsynth-core/tests/ops2d.rs``.
"""

import numpy as np
import pytest
from scipy import sparse

from physsynth.core import exciter as ex
from physsynth.core import operators2d as o2

physsynth_rs = pytest.importorskip(
    "physsynth_rs", reason="the Rust extension is not built in this environment"
)

# The plan's Group A agreement target. Only the two inner products are measured against it.
GROUP_A_TOL = 1e-13

# Grid resolutions. Both even and odd are present: an even N puts a node exactly at the origin and
# an odd one does not, which changes the disk's mask structure and the rim's staircase.
SIZES = [2, 3, 4, 5, 8, 15, 16, 33, 40]


def _canonical(m):
    """A copy of ``m`` in canonical CSR: sorted indices, no explicit zeros."""
    out = sparse.csr_matrix(m, copy=True)
    out.sort_indices()
    out.eliminate_zeros()
    return out


def _rebuild(triplets):
    data, indices, indptr, shape = triplets
    return sparse.csr_matrix((data, indices, indptr), shape=shape)


# -- the grid and the two masks ------------------------------------------------------------------


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("half_extent", [1.0, 0.15, 0.3333333333333333])
def test_grid_coords_is_bit_identical(n, half_extent):
    X, Y, h = o2.grid_coords_py(n, half_extent)
    Xr, Yr, hr = physsynth_rs.grid_coords(n, half_extent)

    assert Xr.shape == X.shape == (n + 1, n + 1)
    assert Xr.dtype == np.float64
    assert np.array_equal(X, Xr)
    assert np.array_equal(Y, Yr)
    assert h == hr


@pytest.mark.parametrize("n", SIZES)
def test_the_grid_endpoints_are_exact_on_both_sides(n):
    # `np.linspace` overwrites the last entry with the endpoint rather than computing it, so
    # `X[0, -1]` is `+half` exactly. Reproducing the overwrite is the reason this passes; computing
    # `i * h - half` for the last node would be off by an ulp, and that ulp decides rim membership.
    X, Y, _ = physsynth_rs.grid_coords(n, 0.15)
    assert X[0, 0] == -0.15
    assert X[0, -1] == 0.15
    assert Y[0, 0] == -0.15
    assert Y[-1, 0] == 0.15


@pytest.mark.parametrize("nx", [1, 2, 3, 8, 17])
@pytest.mark.parametrize("ny", [1, 2, 5, 16])
def test_rectangle_mask_is_identical(nx, ny):
    m = o2.rectangle_mask_py(nx, ny)
    r = physsynth_rs.rectangle_mask(nx, ny)
    assert r.shape == m.shape == (ny + 1, nx + 1)
    assert r.dtype == np.bool_
    assert np.array_equal(m, r)


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("radius_frac", [1.0, 0.999, 0.7, 0.4])
def test_disk_mask_is_identical_including_on_the_rim(n, radius_frac):
    # `radius_frac == 1.0` puts the rim exactly on the bounding-box nodes, which is the case where
    # a `<` and a `<=` differ and where a `hypot` spelling would differ from `x*x + y*y`.
    half = 0.15
    X, Y, _ = o2.grid_coords_py(n, half)
    m = o2.disk_mask_py(X, Y, half * radius_frac)
    r = physsynth_rs.disk_mask(X, Y, half * radius_frac)
    assert np.array_equal(m, r), f"{m.sum()} live vs {r.sum()}"


def test_a_node_exactly_on_the_rim_is_dead_on_both_sides():
    X, Y, _ = o2.grid_coords_py(8, 1.0)
    for mask in (o2.disk_mask_py(X, Y, 1.0), physsynth_rs.disk_mask(X, Y, 1.0)):
        assert not mask[4, 0]
        assert not mask[4, -1]
        assert not mask[0, 4]
        assert mask[4, 1]


# -- the Laplacian --------------------------------------------------------------------------------


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("domain", ["disk", "rectangle"])
def test_laplacian_from_mask_is_bit_identical(n, domain):
    if domain == "disk":
        X, Y, h = o2.grid_coords_py(n, 0.15)
        mask = o2.disk_mask_py(X, Y, 0.15)
    else:
        h = 0.4 / n
        mask = o2.rectangle_mask_py(n, max(n // 2, 1))

    L, index_map = o2.laplacian_from_mask_py(mask, h)
    triplets, index_map_r = physsynth_rs.laplacian_from_mask_csr(mask, h)
    Lr = _rebuild(triplets)
    Lc = _canonical(L)

    assert Lr.shape == Lc.shape
    assert Lr.nnz == Lc.nnz
    assert np.array_equal(Lc.data, Lr.data)
    assert np.array_equal(Lc.indices, Lr.indices)
    assert np.array_equal(Lc.indptr, Lr.indptr)
    assert np.array_equal(index_map, index_map_r)
    assert index_map_r.dtype == np.int64


@pytest.mark.parametrize("n", SIZES)
def test_the_laplacian_divisor_survives_a_sweep_of_h(n):
    # `-4.0 * (1/(h*h))` and `-4.0 / (h*h)` are different doubles for most `h`, and the difference
    # would reach every timestep. Sweeping `n` is what turns that into a caught bug rather than a
    # trajectory that drifts apart slowly.
    h = 1.0 / n
    mask = o2.rectangle_mask_py(4, 4)
    L, _ = o2.laplacian_from_mask_py(mask, h)
    Lr = _rebuild(physsynth_rs.laplacian_from_mask_csr(mask, h)[0])
    assert np.array_equal(_canonical(L).data, Lr.data)


def test_the_binding_hands_back_triplets_not_a_matrix():
    # The core must not learn what SciPy is; the shim in `operators2d.py` is where a matrix appears.
    mask = o2.rectangle_mask_py(6, 4)
    triplets, index_map = physsynth_rs.laplacian_from_mask_csr(mask, 0.1)
    assert isinstance(triplets, tuple) and len(triplets) == 4
    data, indices, indptr, shape = triplets
    assert data.dtype == np.float64
    assert indices.dtype == np.int32
    assert indptr.dtype == np.int32
    assert shape == (int(mask.sum()), int(mask.sum()))
    assert index_map.ndim == 2


# -- embed and the inner products -----------------------------------------------------------------


@pytest.mark.parametrize("n", [4, 9, 16])
def test_embed_is_bit_identical(n):
    X, Y, h = o2.grid_coords_py(n, 0.2)
    mask = o2.disk_mask_py(X, Y, 0.2)
    _, index_map = o2.laplacian_from_mask_py(mask, h)
    values = np.linspace(-1.0, 1.0, int(mask.sum())) ** 3

    a = o2.embed_py(values, index_map)
    b = physsynth_rs.embed(values, index_map)
    assert b.shape == a.shape == mask.shape
    assert np.array_equal(a, b)
    assert np.all(b[~mask] == 0.0)


@pytest.mark.parametrize("size", [1, 4, 17, 129, 1025])
def test_the_two_dimensional_inner_products_agree_to_the_group_a_target(size):
    rng = np.random.default_rng(20260826 + size)
    f = rng.standard_normal(size)
    g = rng.standard_normal(size)
    h = 0.0137

    for got, want in (
        (physsynth_rs.inner2d(f, g, h), o2.inner2d_py(f, g, h)),
        (physsynth_rs.norm2_2d(f, h), o2.norm2_2d_py(f, h)),
    ):
        assert abs(got - want) <= GROUP_A_TOL * max(abs(want), 1e-300)


def test_inner2d_is_exactly_norm2_when_the_operands_coincide():
    rng = np.random.default_rng(7)
    f = rng.standard_normal(64)
    assert physsynth_rs.inner2d(f, f, 0.5) == physsynth_rs.norm2_2d(f, 0.5)


# -- the excitations -------------------------------------------------------------------------------


@pytest.mark.parametrize("n_nodes", [5, 21, 101, 1025])
@pytest.mark.parametrize("frac", [0.1, 0.5, 0.7777, 0.99])
def test_the_pluck_is_bit_identical(n_nodes, frac):
    L = 0.65
    x = np.linspace(0.0, L, n_nodes)
    a = ex.triangular_pluck_py(x, L, frac * L, 1.5)
    b = physsynth_rs.triangular_pluck(x, L, frac * L, 1.5)
    assert np.array_equal(a, b)


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.5, 2.0])
def test_both_sides_reject_a_pluck_outside_the_string(bad):
    x = np.linspace(0.0, 1.0, 11)
    with pytest.raises(ValueError) as py_err:
        ex.triangular_pluck_py(x, 1.0, bad)
    with pytest.raises(ValueError) as rs_err:
        physsynth_rs.triangular_pluck(x, 1.0, bad)
    assert str(py_err.value) == str(rs_err.value)


@pytest.mark.parametrize("n_nodes", [11, 65, 257, 1001])
@pytest.mark.parametrize("width", [0.05, 0.1234, 0.5])
def test_the_raised_cosine_survives_a_sweep_of_the_transcendental(n_nodes, width):
    # The sweep is the point: `np.cos` on a float64 array is NumPy's own vectorised routine, not
    # the platform libm, so this is a claim about two implementations agreeing. A platform where
    # they diverge by an ulp fails here with an obvious cause.
    x = np.linspace(0.0, 1.0, n_nodes)
    a = ex.raised_cosine_py(x, 1.0, 0.4, width, 2.0)
    b = physsynth_rs.raised_cosine(x, 1.0, 0.4, width, 2.0)
    assert np.array_equal(a, b), f"worst |diff| = {np.max(np.abs(a - b)):e}"


def test_the_raised_cosine_clamps_the_ends_on_both_sides():
    x = np.linspace(0.0, 1.0, 41)
    for u in (ex.raised_cosine_py(x, 1.0, 0.0, 0.3), physsynth_rs.raised_cosine(x, 1.0, 0.0, 0.3)):
        assert u[0] == 0.0
        assert u[-1] == 0.0
        assert u[1] > 0.0


@pytest.mark.parametrize("n", [4, 15, 40])
@pytest.mark.parametrize("width", [0.03, 0.0777, 0.2])
def test_the_two_dimensional_bump_is_bit_identical_and_keeps_its_shape(n, width):
    X, Y, _ = o2.grid_coords_py(n, 0.15)
    a = ex.raised_cosine_2d_py(X, Y, (0.011, -0.007), width, 1e-3)
    b = physsynth_rs.raised_cosine_2d(X, Y, (0.011, -0.007), width, 1e-3)
    assert b.shape == a.shape == X.shape
    assert np.array_equal(a, b), f"worst |diff| = {np.max(np.abs(a - b)):e}"


@pytest.mark.parametrize("width", [0.0, -1.0])
def test_both_sides_reject_a_non_positive_width(width):
    x = np.linspace(0.0, 1.0, 11)
    X, Y, _ = o2.grid_coords_py(4, 1.0)
    for py_call, rs_call in (
        (
            lambda: ex.raised_cosine_py(x, 1.0, 0.5, width),
            lambda: physsynth_rs.raised_cosine(x, 1.0, 0.5, width),
        ),
        (
            lambda: ex.raised_cosine_2d_py(X, Y, (0.0, 0.0), width),
            lambda: physsynth_rs.raised_cosine_2d(X, Y, (0.0, 0.0), width),
        ),
    ):
        with pytest.raises(ValueError) as py_err:
            py_call()
        with pytest.raises(ValueError) as rs_err:
            rs_call()
        assert str(py_err.value) == str(rs_err.value)


# -- the guitar outline (Phase 5, batch 1) -------------------------------------------------------

# The outline parameters that ship: `plate.py`'s defaults, the two fixtures in
# `tests/test_guitar_plate.py`, `tests/test_web_backend.py`'s long narrow plate, the first point of
# the viewer's waist sweep (`waist = 0.0`) and a negative `asym`. The degenerate lens is in the
# list on purpose -- it is the only one whose mask a last bit could move.
OUTLINES = [(0.42, 0.30), (0.97, 0.30), (0.88, 0.0), (0.0, 0.0), (0.60, -0.30)]
# The same list minus the degenerate lens, for the one claim the lens does not satisfy: that the
# outline clears the `|x| < half` comparison by far more than a last bit. It is separated rather
# than special-cased inside the test because "which outlines are provably safe" is the batch's
# result, and a list is a clearer statement of it than a branch.
REAL_OUTLINES = [o for o in OUTLINES if o != (0.0, 0.0)]
GUITARS = [(0.37, 0.48), (0.15, 0.70)]
GUITAR_NS = [8, 16, 24, 32, 33, 48, 64, 96]


def _guitar_grid(Lx, Ly_asked, N):
    """The node grid a guitar plate builds on, exactly as ``plate.Plate`` builds it."""
    h = Lx / N
    Ny = max(int(round(Ly_asked / h)), 1)
    Ly = Ny * h
    X, Y = np.meshgrid(np.linspace(0.0, Lx, N + 1), np.linspace(0.0, Ly, Ny + 1))
    return X - 0.5 * Lx, Y, Ly


@pytest.mark.parametrize("waist,asym", OUTLINES)
def test_the_half_width_profile_is_bit_identical(waist, asym):
    t = np.linspace(0.0, 1.0, 20001)
    a = o2.guitar_half_width_py(t, waist, asym)
    b = physsynth_rs.guitar_half_width(t, waist, asym)
    assert b.shape == a.shape
    assert np.array_equal(a, b), f"worst |diff| = {np.max(np.abs(a - b)):e}"


@pytest.mark.parametrize("waist,asym", OUTLINES)
def test_the_profile_keeps_the_shape_it_was_handed(waist, asym):
    # `guitar_area` hands it a flat row of two million midpoints and `guitar_mask` hands it a
    # meshgrid; `plate._depth_inside_outline` hands it whatever the caller selected. Returning the
    # input's shape is part of the interface, and a binding that always returned 2-D would work
    # until the first 1-D caller.
    for shape in [(), (7,), (3, 5), (2, 3, 4)]:
        t = np.linspace(0.05, 0.95, int(np.prod(shape)) or 1).reshape(shape)
        a = o2.guitar_half_width_py(t, waist, asym)
        b = physsynth_rs.guitar_half_width(t, waist, asym)
        assert b.shape == a.shape == shape
        assert np.array_equal(a, b)


@pytest.mark.parametrize("waist,asym", OUTLINES)
@pytest.mark.parametrize("width", [0.37, 0.15, 0.2718281828459045])
def test_the_normalising_scale_is_bit_identical(width, waist, asym):
    # One ulp here is one ulp on every node of the outline at once, because `scale` multiplies
    # every half-width the mask compares against -- so this is the sharpest single number in the
    # batch, not a convenience.
    assert o2.guitar_scale_py(width, waist, asym) == physsynth_rs.guitar_scale(width, waist, asym)


@pytest.mark.parametrize("N", GUITAR_NS)
@pytest.mark.parametrize("waist,asym", OUTLINES)
@pytest.mark.parametrize("Lx,Ly", GUITARS)
def test_the_guitar_mask_is_identical_node_for_node(Lx, Ly, waist, asym, N):
    X, Y, Ly_snapped = _guitar_grid(Lx, Ly, N)
    m = o2.guitar_mask_py(X, Y, Ly_snapped, Lx, waist, asym)
    r = physsynth_rs.guitar_mask(X, Y, Ly_snapped, Lx, waist, asym)
    assert r.shape == m.shape
    assert r.dtype == np.bool_
    assert np.array_equal(m, r), (
        f"{int(np.count_nonzero(m != r))} node(s) differ -- the two implementations are not "
        "building the same plate"
    )


def _margin_in_ulps(Lx, Ly, N, waist, asym):
    """Every interior node's distance from the ``|x| < half`` comparison, in ulps of ``half``.

    The division by ``np.spacing`` is only meaningful while ``half`` stays a normal double —
    ``np.spacing(0.0)`` is 5e-324 and would turn every margin into an ``inf`` that passes the bar
    below while asserting nothing. Measured over this sweep the smallest interior ``|half|`` is
    8.7e-6, thirteen orders inside the normal range, and that is asserted here rather than assumed
    so the test cannot go vacuous if a new outline is added.
    """
    X, Y, Ly_snapped = _guitar_grid(Lx, Ly, N)
    t = Y / Ly_snapped
    half = o2.guitar_scale_py(Lx, waist, asym) * o2.guitar_half_width_py(t, waist, asym)
    interior = (t > 0.0) & (t < 1.0)
    inside = np.abs(half)[interior]
    assert inside.min() > 1e-300, (
        f"`half` reaches {inside.min():e} on this outline -- `np.spacing` is no longer a "
        "meaningful denominator and the margin bar would pass on infinities"
    )
    gap = np.abs(np.abs(X) - half)[interior]
    return gap / np.spacing(inside)


@pytest.mark.parametrize("N", GUITAR_NS)
@pytest.mark.parametrize("waist,asym", REAL_OUTLINES)
@pytest.mark.parametrize("Lx,Ly", GUITARS)
def test_the_comparison_margin_is_wide_for_every_real_outline(Lx, Ly, N, waist, asym):
    """How much room a last bit of ``sin`` has before it moves a node — measured, not assumed.

    This is the proof behind the exact mask assertion above, and it is the batch's central
    measurement. It is a *test* rather than a note because the margin is a property of the outline
    and the grid, so a new shipped fixture could lose it with nothing else noticing.
    """
    smallest = float(_margin_in_ulps(Lx, Ly, N, waist, asym).min())
    # The bar is `1e6` and the observed minimum over the whole sweep is 1.9e7 ulps -- thirteen
    # orders above one. The gap between the two is deliberate: the claim being defended is "a last
    # bit cannot move this node", not "the margin is exactly what it was in August 2026", so a new
    # fixture that is merely tighter should pass and only one that is genuinely marginal should
    # fail.
    assert smallest > 1e6, (
        f"Lx={Lx} waist={waist} asym={asym} N={N}: the closest node clears the outline by only "
        f"{smallest:.3e} ulps of `half` (the sweep's measured minimum was 1.9e7), so a last bit "
        "of `sin` could move it. The exact assertion on this mask is no longer safe -- see the "
        "plan's section 25."
    )


def test_the_degenerate_lens_is_the_exception_and_sits_on_the_rim():
    """The outline the margin proof does **not** cover, pinned as a node rather than a statistic.

    A plain lens is ``0.5*Lx*sin(pi t)``. At ``t = 1/6`` that is ``0.25*Lx`` in real arithmetic and
    the ``N = 32`` grid puts a node exactly there, so which side of a strict ``<`` it lands on is
    decided by the last bit of one ``sin`` call. At ``t = 1/2`` it is worse: ``sin`` returns
    exactly ``1.0`` and the node is exactly *on* the rim.

    Reachable, not hypothetical — the viewer's waist sweep starts at ``waist = 0.0``.
    """
    X, Y, Ly = _guitar_grid(0.37, 0.48, 32)
    t = Y / Ly
    half = o2.guitar_scale_py(0.37, 0.0, 0.0) * o2.guitar_half_width_py(t, 0.0, 0.0)

    j, i = 7, 8
    assert t[j, i] == pytest.approx(1.0 / 6.0, rel=1e-15)
    assert abs(X[j, i]) == 0.0925
    assert abs(abs(X[j, i]) - half[j, i]) <= 4.0 * np.spacing(half[j, i]), (
        "the lens's rim node is no longer within a few ulps of the outline"
    )

    # The widest point: `scale` is `0.5*Lx` exactly (the sampled peak is `sin(pi/2) == 1.0`), and
    # the bounding-box node is `0.5*Lx` from the centre line, so the two are the same double.
    row = int(np.argmin(np.abs(t[:, 0] - 0.5)))
    assert t[row, 0] == 0.5
    assert half[row, 0] == 0.185 == abs(X[row, 0])

    assert float(_margin_in_ulps(0.37, 0.48, 32, 0.0, 0.0).min()) == 0.0


@pytest.mark.parametrize("N", GUITAR_NS)
@pytest.mark.parametrize("waist,asym", OUTLINES)
def test_the_cell_counts_and_the_prune_are_identical(N, waist, asym):
    X, Y, Ly = _guitar_grid(0.37, 0.48, N)
    raw = o2.guitar_mask_py(X, Y, Ly, 0.37, waist, asym)
    assert np.array_equal(o2.live_cells_py(raw), physsynth_rs.live_cells(raw))
    counts = o2.cells_per_node_py(raw)
    counts_r = physsynth_rs.cells_per_node(raw)
    assert counts_r.dtype == counts.dtype == np.int64
    assert np.array_equal(counts, counts_r)

    pruned, dropped = o2.prune_to_area_carrying_py(raw)
    pruned_r, dropped_r = physsynth_rs.prune_to_area_carrying(raw)
    assert dropped == dropped_r
    assert np.array_equal(pruned, pruned_r)


@pytest.mark.parametrize("waist,asym", OUTLINES)
@pytest.mark.parametrize("Lx,Ly", GUITARS)
def test_the_outline_area_agrees_to_tolerance_and_is_not_asked_to_agree_exactly(
    Lx, Ly, waist, asym
):
    """The one number in this group that is **not** bit-identical, by decision.

    Two million midpoints are summed by ``np.sum`` on the Python side — pairwise above a blocksize
    of 128 — and left to right in Rust. Matching would mean transcribing NumPy's blocking, which is
    the bargain §18.2 refused for SciPy's sparse-product kernel: a claim about a library internal
    that a point release may change. It can afford to be inexact because nothing branches on it and
    it reaches no timestep — it is a denominator in a reported area deficit (§19.2's question,
    answered "no"). Measured 2026-08-28: worst 1.2e-13 relative over this sweep.
    """
    a = o2.guitar_area_py(Ly, Lx, waist, asym)
    b = physsynth_rs.guitar_area(Ly, Lx, waist, asym)
    assert b == pytest.approx(a, rel=1e-12)


@pytest.mark.parametrize("waist,asym", OUTLINES)
def test_the_two_profile_spellings_stay_two(waist, asym):
    """The pin on the deliberate duplicate (plan §20.2).

    ``guitar_half_width`` must be ``_profile`` — the scalar libm, the spelling Rust shares — and
    ``guitar_area`` must be ``_profile_vec``, NumPy's vectorised one. The two agree to a last bit
    and differ in it (221,580 of the profile values over a 130-case sweep, measured 2026-08-28),
    which is exactly what makes them look like a duplicate that wants tidying. Collapsing them
    would either move the plate's geometry or cost half a second on every plate built, so both
    call sites are asserted here rather than left to a comment.

    Neither assertion is a claim about a CPU. Each says only "this call site uses this spelling",
    which is a property of the code — so on a machine where NumPy and libm happen to agree
    everywhere, a tidy-up still fails here, and it fails for the right reason.
    """
    t = np.linspace(0.0, 1.0, 4001)
    scalar = np.array([o2._profile(float(tv), waist, asym) for tv in t])
    assert np.array_equal(o2.guitar_half_width_py(t, waist, asym), scalar), (
        "`guitar_half_width` is no longer the scalar-libm spelling -- the mask path has been "
        "tidied back onto NumPy's transcendentals and is a claim about the CPU again"
    )

    vector = o2._profile_vec(t, waist, asym)
    assert vector == pytest.approx(scalar, rel=1e-15, abs=1e-300)

    m = 2_000_000
    tt = (np.arange(m) + 0.5) / m
    expected = float(
        2.0
        * np.sum(o2.guitar_scale_py(0.37, waist, asym) * o2._profile_vec(tt, waist, asym))
        * (0.48 / m)
    )
    assert o2.guitar_area_py(0.48, 0.37, waist, asym) == expected, (
        "`guitar_area` is no longer using the vectorised spelling -- two million scalar libm "
        "calls cost half a second on every guitar plate"
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"length": 0.0},
        {"length": -1.0},
        {"width": 0.0},
        {"waist": 1.0},
        {"waist": -0.1},
        {"asym": 2.0},
        {"asym": -3.5},
    ],
)
def test_both_sides_reject_the_same_outlines_with_the_same_words(kwargs):
    X, Y, Ly = _guitar_grid(0.37, 0.48, 8)
    args = {"length": Ly, "width": 0.37, "waist": 0.42, "asym": 0.30} | kwargs
    with pytest.raises(ValueError) as py_err:
        o2.guitar_mask_py(X, Y, args["length"], args["width"], args["waist"], args["asym"])
    with pytest.raises(ValueError) as rs_err:
        physsynth_rs.guitar_mask(X, Y, args["length"], args["width"], args["waist"], args["asym"])
    assert str(py_err.value) == str(rs_err.value)


# -- the matrices (Phase 5 batch 2) ---------------------------------------------------------------

# Grids the plate family actually ships on, plus one where `1/h**2` is exactly representable and one
# where it is not -- the distinction that decides whether the orthotropic assembly collapses onto
# `L @ L` bit-for-bit, and the one a single-grid check would have missed.
MATRIX_GRIDS = [(6, 5, 0.037), (8, 8, 0.05), (12, 9, 0.0125), (16, 16, 0.03125), (20, 13, 0.1 / 7)]

# Grain triples: isotropic, a generic orthotropy, spruce's own cross term, and one that violates
# `g_h > -sqrt(g_x g_y)` -- the operator builds the indefinite case on purpose, so the port must
# too.
GRAINS = [(1.0, 1.0, 1.0), (1.3, 0.9, 1.1), (1.0, 0.153, 0.073), (1.0, -0.1, 0.5)]


def _assert_identical(py_m, rs_m, what):
    """Bit-for-bit, including the stored order -- the claim this batch is making."""
    a = py_m.tocsr()
    b = rs_m.tocsr()
    assert a.shape == b.shape, f"{what}: shape {a.shape} vs {b.shape}"
    assert a.nnz == b.nnz, f"{what}: nnz {a.nnz} vs {b.nnz}"
    assert np.array_equal(a.indptr, b.indptr), f"{what}: indptr"
    assert np.array_equal(a.indices, b.indices), (
        f"{what}: stored column order differs -- a CSR matvec sums a row in stored order, so this "
        "is a different operator on the update path, not a cosmetic difference"
    )
    assert np.array_equal(a.data, b.data), (
        f"{what}: {int((a.data != b.data).sum())} of {a.nnz} entries differ"
    )


def _pruned_guitar(n):
    X, Y, Ly = _guitar_grid(0.37, 0.48, n)
    return o2.prune_to_area_carrying_py(o2.guitar_mask_py(X, Y, Ly, 0.37, 0.42, 0.30))[0], 0.37 / n


def _pruned_disk(n):
    X, Y, h = o2.grid_coords_py(n, 0.5)
    return o2.prune_to_area_carrying_py(o2.disk_mask_py(X, Y, 0.4))[0], h


@pytest.mark.parametrize("nx,ny,h", MATRIX_GRIDS)
def test_the_biharmonic_is_bit_identical_on_a_rectangle(nx, ny, h):
    mask = o2.rectangle_mask_py(nx, ny)
    B, index_map = o2.biharmonic_from_mask_py(mask, h)
    triplets, im_rs = physsynth_rs.biharmonic_from_mask_csr(mask, h)
    _assert_identical(B, _rebuild(triplets), f"B({nx},{ny},{h})")
    assert np.array_equal(index_map, im_rs)
    assert im_rs.dtype == np.int64


@pytest.mark.parametrize("nx,ny,h", MATRIX_GRIDS)
def test_the_rust_product_is_canonical_before_anything_sorts_it(nx, ny, h):
    """The exact-index claim above is about the *port*, not about two calls to ``sort_indices``.

    Both sides of that comparison are sorted -- the Python reference by ``portable.canonical`` and
    the Rust side by ``Csr::from_rows``. So on its own it would still pass if the binding emitted
    rows in an arbitrary order and the shim tidied them up, which is a weaker claim than the one
    being made. This asserts the property at the source: what comes *out of the binding* is already
    canonical, so `physsynth/core/operators2d.py`'s shim is rebuilding a matrix rather than fixing
    one, and a future Rust-side reordering fails here with an obvious cause.
    """
    mask = o2.rectangle_mask_py(nx, ny)
    for raw in (
        _rebuild(physsynth_rs.biharmonic_from_mask_csr(mask, h)[0]),
        _rebuild(physsynth_rs.orthotropic_biharmonic_csr(nx, ny, h, 1.3, 0.9, 1.1)[0]),
        _rebuild(physsynth_rs.free_plate_stiffness_csr(nx, ny, h, 0.3, 1.0, 1.0, None, None)[0]),
    ):
        # `has_sorted_indices` is a lazily-set FLAG, and `csr_matrix((data, indices, indptr))`
        # trusts it rather than checking -- so the flag alone would assert nothing. Read the
        # indices.
        for r in range(raw.shape[0]):
            idx = raw.indices[raw.indptr[r] : raw.indptr[r + 1]]
            assert np.all(np.diff(idx) > 0), (
                f"row {r} of the raw binding output is not ascending -- the Rust side is relying "
                "on the Python shim to sort it, which makes the exact-index comparison a test of "
                "`sort_indices` rather than of the port"
            )


@pytest.mark.parametrize("n", [16, 24, 32])
@pytest.mark.parametrize("shape", ["guitar", "disk"])
def test_the_biharmonic_is_bit_identical_on_a_staircased_outline(n, shape):
    # A staircased rim is where `L`'s rows stop being uniform, so `L @ L`'s per-entry reduction
    # stops being the same length everywhere -- which is the thing plan section 23.2 says decides
    # whether an exact claim is available at all.
    mask, h = _pruned_guitar(n) if shape == "guitar" else _pruned_disk(n)
    B, _ = o2.biharmonic_from_mask_py(mask, h)
    _assert_identical(B, _rebuild(physsynth_rs.biharmonic_from_mask_csr(mask, h)[0]), shape)


@pytest.mark.parametrize("n_int", [1, 2, 5, 17, 40])
@pytest.mark.parametrize("h", [0.037, 0.05, 0.1 / 7, 0.5])
def test_the_interior_second_difference_is_bit_identical(n_int, h):
    # `n_int == 1` is the degenerate end -- `sparse.diags` is handed an empty off-diagonal, and a
    # plate at `N = 2` is a legal (if useless) plate. `n_int == 0` is left out deliberately: the
    # only caller is `orthotropic_biharmonic`, which refuses `Nx < 2` first, and what the reference
    # does at zero is `scipy.sparse.diags`'s own bounds error rather than anything this module
    # says. Pinning a library internal is the bargain plan section 18.2 refused.
    _assert_identical(
        o2.dirichlet_interior_d2_1d_py(n_int, h),
        _rebuild(physsynth_rs.dirichlet_interior_d2_1d_csr(n_int, h)),
        f"D2({n_int},{h})",
    )


@pytest.mark.parametrize("nx,ny,h", MATRIX_GRIDS)
@pytest.mark.parametrize("grain", GRAINS)
def test_the_orthotropic_biharmonic_is_bit_identical(nx, ny, h, grain):
    B, index_map = o2.orthotropic_biharmonic_py(nx, ny, h, *grain)
    triplets, im_rs = physsynth_rs.orthotropic_biharmonic_csr(nx, ny, h, *grain)
    _assert_identical(B, _rebuild(triplets), f"Bo({nx},{ny},{h},{grain})")
    assert np.array_equal(index_map, im_rs)


@pytest.mark.parametrize("nx,ny,h", MATRIX_GRIDS)
@pytest.mark.parametrize("nu", [0.0, 0.3, -0.5, 0.45])
def test_the_free_plate_stiffness_is_bit_identical_on_a_rectangle(nx, ny, h, nu):
    K, W, index_map = o2.free_plate_stiffness_py(nx, ny, h, nu)
    k, w, im_rs = physsynth_rs.free_plate_stiffness_csr(nx, ny, h, nu, 1.0, 1.0, None, None)
    _assert_identical(K, _rebuild(k), f"K({nx},{ny},{h},{nu})")
    _assert_identical(W, _rebuild(w), f"W({nx},{ny},{h},{nu})")
    assert np.array_equal(index_map, im_rs)


@pytest.mark.parametrize("nx,ny,h", MATRIX_GRIDS[:3])
@pytest.mark.parametrize("g_1", [-0.1, 0.0, 0.05, 0.1])
def test_the_free_plate_stiffness_is_bit_identical_across_the_four_constant_splits(nx, ny, h, g_1):
    # The free branch needs the coupling and torsional rigidities SEPARATELY -- four splits of the
    # same `H` that the supported branch cannot tell apart and that move the free plate's
    # fundamental by 6.5x. If the port collapsed them back to three constants, only this sees it.
    kw = dict(grain_x=1.0, grain_y=0.073, grain_coupling=g_1, grain_torsion=0.5 * (0.153 - g_1))
    K, W, _ = o2.free_plate_stiffness_py(nx, ny, h, 0.3, **kw)
    k, w, _ = physsynth_rs.free_plate_stiffness_csr(
        nx, ny, h, 0.3, kw["grain_x"], kw["grain_y"], kw["grain_coupling"], kw["grain_torsion"]
    )
    _assert_identical(K, _rebuild(k), f"K split g_1={g_1}")
    _assert_identical(W, _rebuild(w), f"W split g_1={g_1}")


@pytest.mark.parametrize("n", [16, 24, 32])
@pytest.mark.parametrize("shape", ["guitar", "disk"])
@pytest.mark.parametrize("nu", [0.0, 0.3])
def test_the_free_plate_stiffness_is_bit_identical_on_a_staircased_outline(n, shape, nu):
    mask, h = _pruned_guitar(n) if shape == "guitar" else _pruned_disk(n)
    K, W, index_map = o2.free_plate_stiffness_from_mask_py(mask, h, nu)
    k, w, im_rs = physsynth_rs.free_plate_stiffness_from_mask_csr(mask, h, nu, 1.0, 1.0, None, None)
    _assert_identical(K, _rebuild(k), f"K {shape} N={n} nu={nu}")
    _assert_identical(W, _rebuild(w), f"W {shape} N={n} nu={nu}")
    assert np.array_equal(index_map, im_rs)


@pytest.mark.parametrize("nx,ny,h", MATRIX_GRIDS[:3])
def test_a_full_bounding_box_is_the_rectangle_on_both_sides(nx, ny, h):
    # The rectangle is just the mask that happens to be all-ones, and that is asserted rather than
    # asserted-in-prose: one code path, and it has to be the same one code path in both languages.
    full = np.ones((ny + 1, nx + 1), dtype=bool)
    K_py, W_py, _ = o2.free_plate_stiffness_py(nx, ny, h, 0.3)
    M_py, _, _ = o2.free_plate_stiffness_from_mask_py(full, h, 0.3)
    _assert_identical(K_py, M_py, "python box vs rectangle")
    k, w, _ = physsynth_rs.free_plate_stiffness_from_mask_csr(full, h, 0.3, 1.0, 1.0, None, None)
    _assert_identical(K_py, _rebuild(k), "rust box vs python rectangle")
    _assert_identical(W_py, _rebuild(w), "rust box weight")


@pytest.mark.parametrize(
    "args",
    [
        (1, 5, 0.01, 1.0, 1.0, 1.0),
        (5, 1, 0.01, 1.0, 1.0, 1.0),
        (0, 0, 0.01, 1.0, 1.0, 1.0),
    ],
)
def test_both_sides_reject_the_same_orthotropic_grids_with_the_same_words(args):
    with pytest.raises(ValueError) as py_err:
        o2.orthotropic_biharmonic_py(*args)
    with pytest.raises(ValueError) as rs_err:
        physsynth_rs.orthotropic_biharmonic_csr(*args)
    assert str(py_err.value) == str(rs_err.value)


@pytest.mark.parametrize("nx,ny", [(1, 5), (5, 1), (0, 3)])
def test_both_sides_reject_the_same_free_plate_grids_with_the_same_words(nx, ny):
    with pytest.raises(ValueError) as py_err:
        o2.free_plate_stiffness_py(nx, ny, 0.01, 0.3)
    with pytest.raises(ValueError) as rs_err:
        physsynth_rs.free_plate_stiffness_csr(nx, ny, 0.01, 0.3, 1.0, 1.0, None, None)
    assert str(py_err.value) == str(rs_err.value)


@pytest.mark.parametrize("nu", [0.5, 1.0, -1.0, -2.5])
def test_both_sides_reject_the_same_poisson_ratios_with_the_same_words(nu):
    # ... and only where `nu` is actually USED. An orthotropic plate's implied nu_yx legitimately
    # exceeds 1/2, so supplying both halves of the split must make the same out-of-range `nu`
    # harmless on both sides -- a port that validated unconditionally would reject a valid material.
    with pytest.raises(ValueError) as py_err:
        o2.free_plate_stiffness_py(6, 5, 0.01, nu)
    with pytest.raises(ValueError) as rs_err:
        physsynth_rs.free_plate_stiffness_csr(6, 5, 0.01, nu, 1.0, 1.0, None, None)
    assert str(py_err.value) == str(rs_err.value)

    K, _, _ = o2.free_plate_stiffness_py(6, 5, 0.01, nu, grain_coupling=0.2, grain_torsion=0.4)
    k, _, _ = physsynth_rs.free_plate_stiffness_csr(6, 5, 0.01, nu, 1.0, 1.0, 0.2, 0.4)
    _assert_identical(K, _rebuild(k), f"superseded nu={nu}")


def test_both_sides_reject_an_empty_mask_with_the_same_words():
    empty = np.zeros((5, 5), dtype=bool)
    with pytest.raises(ValueError) as py_err:
        o2.free_plate_stiffness_from_mask_py(empty, 0.01, 0.3)
    with pytest.raises(ValueError) as rs_err:
        physsynth_rs.free_plate_stiffness_from_mask_csr(empty, 0.01, 0.3, 1.0, 1.0, None, None)
    assert str(py_err.value) == str(rs_err.value)


class _RustLu:
    """``splu(...)``'s one method, from Rust — the shared-factorization hook of plan §24.4."""

    def __init__(self, m):
        m = sparse.csr_matrix(m, copy=True)
        m.sort_indices()
        self.lu = physsynth_rs.SparseLu(
            m.data, m.indices.astype(np.int32), m.indptr.astype(np.int32), m.shape[0]
        )

    def solve(self, b):
        return np.asarray(self.lu.solve(np.ascontiguousarray(b, dtype=float)))


# -- Phase 5, batch 3: the nonlinear plate ---------------------------------------------------------
#
# The five private 1-D differences, the bracket and the clamped Airy solve. Three different kinds
# of claim live here and keeping them apart is the point of the section:
#
# * the five differences and the four bracket operators are **matrices** -- exact, every grid;
# * the bracket's `__call__` is an **update path** -- exact, because every matvec in it is a
#   canonical row gather and the elementwise algebra is spelled the same way on both sides;
# * `AiryStressSolver.solve` is a **SuperLU solve** -- a measured tolerance and nothing better,
#   settled in advance by plan section 24.2. What keeps that from swallowing an assembly error is
#   section 24.4's manoeuvre, repeated here: drive the *Python* solver through the *Rust*
#   factorization and the two go bit-identical, so the whole residue is the solver.

# The grids `AiryStressSolver` is actually built on, read off an instrumented run of the whole von
# Karman half of the suite, plus three that are not. Two of them -- (8, 8, 0.0375) and
# (16, 12, 0.06) -- are the ones where the Gram's two associations part company, which is why the
# list is enumerated rather than sampled.
AIRY_GRIDS = [
    (6, 6, 0.05),
    (8, 8, 0.0375),
    (8, 8, 0.05),
    (10, 8, 0.1),
    (12, 10, 0.05),
    (12, 12, 1.0 / 30.0),
    (13, 9, 0.08),
    (14, 11, 0.07),
    (16, 12, 0.06),
    (16, 16, 0.025),
    (18, 14, 0.05555555555555555),
    (20, 16, 0.05),
    (24, 19, 1.0 / 60.0),
    (5, 3, 1.0 / 3.0),
    (9, 4, 0.125),
    (4, 4, 0.25),
]

ONE_D_SIZES = [2, 3, 4, 5, 8, 9, 16, 17, 32]
ONE_D_STEPS = [0.1, 0.0625, 0.0375, 1.0 / 3.0, 0.006]


@pytest.mark.parametrize("n", ONE_D_SIZES)
@pytest.mark.parametrize("h", ONE_D_STEPS)
def test_the_five_one_d_differences_are_bit_identical(n, h):
    # Each is a fixed pattern of exact scalings of one reciprocal, so exactness here is structural
    # -- but only while both sides spell the reciprocal the same way. `-2.0 / (h*h)` is a different
    # rounding from `-2.0 * (1.0/(h*h))`, and the sweep over `h` is what would catch it.
    _assert_identical(
        o2.collocated_d2_1d_py(n, h),
        _rebuild(physsynth_rs.collocated_d2_1d_csr(n, h)),
        f"collocated n={n} h={h}",
    )
    _assert_identical(
        o2.forward_d1_1d_py(n, h),
        _rebuild(physsynth_rs.forward_d1_1d_csr(n, h)),
        f"forward n={n} h={h}",
    )
    _assert_identical(
        o2.centered_d2_1d_py(n, h),
        _rebuild(physsynth_rs.centered_d2_1d_csr(n, h)),
        f"centered n={n} h={h}",
    )
    _assert_identical(
        o2.clamped_d2_1d_py(n, h),
        _rebuild(physsynth_rs.clamped_d2_1d_csr(n, h)),
        f"clamped n={n} h={h}",
    )
    _assert_identical(
        o2.avg_d1_1d_py(n),
        _rebuild(physsynth_rs.avg_d1_1d_csr(n)),
        f"avg n={n}",
    )


@pytest.mark.parametrize("n", ONE_D_SIZES)
@pytest.mark.parametrize("h", ONE_D_STEPS)
def test_the_collocated_and_centered_differences_are_not_the_same_matrix(n, h):
    # The pair the module is most likely to confuse: same interior stencil, and the ONLY difference
    # is whether the two end rows exist. A port that used one for the other would be caught by
    # nothing else here -- both are symmetric, both annihilate constants on the interior, and the
    # bracket's identity would survive because rim-vanishing fields never weigh those rows.
    coll = o2.collocated_d2_1d_py(n, h)
    cent = o2.centered_d2_1d_py(n, h)
    assert coll.nnz + 4 == cent.nnz
    rs_coll = _rebuild(physsynth_rs.collocated_d2_1d_csr(n, h))
    rs_cent = _rebuild(physsynth_rs.centered_d2_1d_csr(n, h))
    assert rs_coll.nnz + 4 == rs_cent.nnz
    assert rs_coll.indptr[0] == rs_coll.indptr[1], "row 0 must be empty, not zero-valued"


BRACKET_GRIDS = [(4, 4, 0.25), (8, 6, 0.1), (12, 12, 1.0 / 3.0), (16, 12, 0.06)]


@pytest.mark.parametrize("nx,ny,h", BRACKET_GRIDS)
def test_the_bracket_operators_are_bit_identical(nx, ny, h):
    py = o2.VonKarmanBracketPy(nx, ny, h)
    rs = physsynth_rs.VonKarmanBracket(nx, ny, h)
    for name in ("Sxx", "Syy", "Dxy", "Acell"):
        _assert_identical(getattr(py, name), getattr(rs, name), f"{name} {nx}x{ny} h={h}")
    assert (rs.Nx, rs.Ny, rs.h, rs.n_nodes) == (py.Nx, py.Ny, py.h, py.n_nodes)


@pytest.mark.parametrize("nx,ny,h", BRACKET_GRIDS)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_bracket_evaluation_is_bit_identical(nx, ny, h, seed):
    # The nonlinear plate's UPDATE path: `l(w, w)` sources the Airy solve and `l(w, F)` is the
    # coupling force, both every Picard sweep of every timestep. Exactness here is not decoration,
    # it is what keeps the two trajectories comparable at all.
    py = o2.VonKarmanBracketPy(nx, ny, h)
    rs = physsynth_rs.VonKarmanBracket(nx, ny, h)
    rng = np.random.default_rng(seed)
    a = rng.standard_normal(py.n_nodes)
    b = rng.standard_normal(py.n_nodes)
    assert np.array_equal(np.asarray(rs(a, b)), py(a, b))
    # ... and symmetric in its two arguments on both sides, to the bit.
    assert np.array_equal(np.asarray(rs(b, a)), np.asarray(rs(a, b)))


@pytest.mark.parametrize("nx,ny,h", [(6, 6, 0.05), (12, 12, 1.0 / 3.0)])
def test_the_transposed_corner_average_is_the_same_sum_either_way(nx, ny, h):
    """``Acell.T @ v`` is a CSC *scatter* in SciPy and a CSR *gather* in Rust — and equal anyway.

    This is a lemma rather than a measurement, and it is the one place in the bracket where the two
    languages do genuinely different loops. A CSC matvec accumulates each output entry over
    **increasing column index**; a sorted-CSR row gather accumulates over increasing column index.
    Same order, same doubles, same result — for any canonically-stored matrix. The assertion is
    here because the *premise* (canonical storage) is the thing this whole batch turns on, so a
    future operator that arrived unsorted would break the lemma silently.
    """
    py = o2.VonKarmanBracketPy(nx, ny, h)
    scatter = py.Acell.T
    assert sparse.isspmatrix_csc(scatter), "SciPy's csr.T is a CSC; the lemma is about that"
    gather = sparse.csr_matrix(py.Acell.T)
    rng = np.random.default_rng(4)
    for _ in range(8):
        v = rng.standard_normal(py.Acell.shape[0])
        assert np.array_equal(scatter @ v, gather @ v)


@pytest.mark.parametrize("nx,ny,h", [(6, 6, 0.05), (12, 12, 1.0 / 3.0), (16, 12, 0.06)])
def test_the_trilinear_form_agrees_to_the_group_a_target(nx, ny, h):
    # The one quantity in the bracket that is NOT exact, and the reason is `inner2d`: NumPy
    # contracts it with `np.dot`, which is BLAS, and section 14.2 settled that matching a BLAS
    # reduction would be a claim about a runner rather than about this code. A read-out, so a
    # tolerance is all it needs.
    py = o2.VonKarmanBracketPy(nx, ny, h)
    rs = physsynth_rs.VonKarmanBracket(nx, ny, h)
    rng = np.random.default_rng(5)
    for _ in range(4):
        a, b, c = (rng.standard_normal(py.n_nodes) for _ in range(3))
        t_py = py.trilinear(a, b, c)
        t_rs = rs.trilinear(a, b, c)
        assert abs(t_rs - t_py) <= GROUP_A_TOL * max(abs(t_py), 1e-30)


@pytest.mark.parametrize("nx,ny,h", AIRY_GRIDS)
def test_the_airy_operator_is_bit_identical(nx, ny, h):
    """``B_F`` matches entry for entry — which took a pair of parentheses in the reference.

    See :class:`physsynth.core.operators2d.AiryStressSolver`. The *values* were never in question
    (both outer factors of every term share a mantissa, so the association cannot move a product);
    what the association moves is the **sum**, because SciPy hands ``Lc_rᵀ @ Wa`` back descending
    and the left bracketing then contracts the shared index in that order.
    """
    py = o2.AiryStressSolverPy(nx, ny, h)
    rs = physsynth_rs.AiryStressSolver(nx, ny, h)
    _assert_identical(py.Bf, rs.Bf, f"Bf {nx}x{ny} h={h}")
    # ... and in the CSC the original actually stores, arrays and all: `tocsr()` sorts, so the
    # comparison above would survive a stored-order difference that a re-factorization would not.
    assert sparse.isspmatrix_csc(rs.Bf), "the original stores a csc and splu wants one"
    assert np.array_equal(py.Bf.indptr, rs.Bf.indptr)
    assert np.array_equal(py.Bf.indices, rs.Bf.indices)
    assert np.array_equal(py.Bf.data, rs.Bf.data)
    assert np.array_equal(np.asarray(rs.mask), py.mask)
    assert np.array_equal(np.asarray(rs.index_map), py.index_map)
    assert np.asarray(rs.index_map).dtype == py.index_map.dtype
    assert (rs.Nx, rs.Ny, rs.h, rs.n_nodes, rs.n_interior) == (
        py.Nx, py.Ny, py.h, py.n_nodes, py.n_interior,
    )


def test_the_gram_association_is_a_different_sum_and_this_finds_the_witness():
    """The batch's finding, pinned so that removing the parentheses turns this red.

    Searching rather than asserting a constant (section 26.6): the two associations agree on most
    grids, so a hand-picked one is a coin flip and a test that landed on an agreeing grid would go
    green having asserted nothing. This walks the grids the suite actually builds and requires at
    least one witness — and then checks that the *shipped* spelling is the ascending one.
    """
    witnesses = []
    unsorted = 0
    for nx, ny, h in AIRY_GRIDS:
        ix = sparse.identity(nx + 1, format="csr")
        iy = sparse.identity(ny + 1, format="csr")
        Lc = sparse.kron(iy, o2.clamped_d2_1d_py(nx, h)) + sparse.kron(
            o2.clamped_d2_1d_py(ny, h), ix
        )
        mx = np.full(nx + 1, h)
        mx[0] = mx[-1] = 0.5 * h
        my = np.full(ny + 1, h)
        my[0] = my[-1] = 0.5 * h
        Wa = sparse.diags(np.kron(my, mx), format="csr")
        cols = o2.rectangle_mask_py(nx, ny).ravel()
        Lc_r = Lc.tocsc()[:, cols]
        # The intermediate SciPy hands back for the left bracketing: descending, in every row.
        mid = (Lc_r.T @ Wa).tocsr()
        if not mid.has_sorted_indices:
            unsorted += 1
        left = _canonical((Lc_r.T @ Wa @ Lc_r).tocsr())
        right = _canonical((Lc_r.T @ (Wa @ Lc_r)).tocsr())
        differing = int((left.data != right.data).sum())
        if differing:
            witnesses.append((nx, ny, h, differing, left.nnz))
        # The shipped operator is the right-associated one, whichever way this grid falls.
        _assert_identical(o2.AiryStressSolverPy(nx, ny, h).Bf, right, f"shipped {nx}x{ny}")
    assert unsorted, (
        "SciPy handed back a sorted `Lc_r.T @ Wa` on every grid -- the kernel behaviour the "
        "parentheses were chosen against is gone, and the reasoning needs re-measuring"
    )
    assert witnesses, (
        "no grid distinguishes the two associations -- this test asserts nothing, and the "
        "parentheses in AiryStressSolver would be free to disappear"
    )


# The grids the suite builds that are too large to want in every parametrisation, but where the
# claim below actually bites: the Airy gap is largest here and a bar tested only on small grids
# would be slack by two orders exactly where it matters (section 16.4's shape, in a tolerance).
AIRY_LARGE_GRIDS = [
    (40, 32, 0.025),
    (48, 48, 1.0 / 120.0),
    (80, 64, 0.0125),
    (96, 96, 1.0 / 240.0),
    (160, 128, 0.00625),
]


def airy_solve_tol(nx, ny):
    """Group D's bar for the Airy solve, which is a SCALING LAW and not a constant.

    SuperLU is supernodal, so matching ``lu.solve`` would be a claim about how SciPy was built
    (plan section 24.2). What that section could not say, and this one can, is how big the residue
    is: the gap runs 3.1e-16 at 4x4 to 1.3e-13 at 24x19 and **5.2e-10 at the 160x128 the airbox
    tests build**, which is ``N^4`` -- the condition number of a biharmonic. Both solves are
    backward stable to machine precision (asserted separately), so the forward difference between
    them is conditioning times epsilon and says nothing about either implementation.

    So the bar is proportional to the number of unknowns squared. The constant is fitted to the
    *worst* measured ratio and left about 8x slack, which is uniform across four orders of grid
    size -- a single constant would be meaningless at one end and wrong at the other.
    """
    return 1e-17 * float(nx * ny) ** 2


@pytest.mark.parametrize("nx,ny,h", AIRY_GRIDS + AIRY_LARGE_GRIDS)
def test_the_airy_solve_is_the_measured_superlu_tolerance(nx, ny, h):
    py = o2.AiryStressSolverPy(nx, ny, h)
    rs = physsynth_rs.AiryStressSolver(nx, ny, h)
    rng = np.random.default_rng(11)
    for _ in range(3):
        src = rng.standard_normal(py.n_nodes)
        src[~py.mask.ravel()] = 0.0
        f_py = py.solve(src)
        f_rs = np.asarray(rs.solve(src))
        scale = np.max(np.abs(f_py))
        assert np.max(np.abs(f_rs - f_py)) <= airy_solve_tol(nx, ny) * scale
        # The rim comes back exactly zero on both sides -- that is a structural claim, not a
        # tolerance, and it is what makes `F` a legal bracket argument.
        assert np.array_equal(f_rs[~py.mask.ravel()], np.zeros(int((~py.mask).sum())))


@pytest.mark.parametrize("nx,ny,h", AIRY_GRIDS[:8])
def test_the_python_solver_on_the_rust_factorization_is_bit_identical(nx, ny, h):
    """Section 24.4's manoeuvre: hold the solver constant and let only the assembly vary.

    Without this the SuperLU gap (~1e-14 of amplitude) is two orders larger than a reassociated
    assembly would be, so a genuine porting error would sit invisibly underneath it — which is
    section 19.4's finding, a real bug the trajectory could not see, waiting to happen. With the
    same factorization on both sides the two solves agree **to the bit**, which is the strongest
    statement available about the part of this class that is not the solver.
    """
    py = o2.AiryStressSolverPy(nx, ny, h)
    rs = physsynth_rs.AiryStressSolver(nx, ny, h)
    shared = _RustLu(py.Bf)
    saved, py._lu = py._lu, shared
    try:
        rng = np.random.default_rng(11)
        for _ in range(3):
            src = rng.standard_normal(py.n_nodes)
            src[~py.mask.ravel()] = 0.0
            assert np.array_equal(np.asarray(rs.solve(src)), py.solve(src))
    finally:
        py._lu = saved


@pytest.mark.parametrize("nx,ny,h", AIRY_GRIDS[:8])
def test_the_membrane_energy_read_out_agrees_to_the_group_a_target(nx, ny, h):
    # `laplacian_norm_sq` is `fi @ (Bf @ fi)`: an exact matvec followed by a BLAS dot. A read-out
    # -- `VKPlate.membrane_energy` is reported, never branched on -- so section 14.2's tolerance is
    # the right bar and nothing downstream cares.
    py = o2.AiryStressSolverPy(nx, ny, h)
    rs = physsynth_rs.AiryStressSolver(nx, ny, h)
    rng = np.random.default_rng(7)
    F = rng.standard_normal(py.n_nodes)
    F[~py.mask.ravel()] = 0.0
    a, b = py.laplacian_norm_sq(F), rs.laplacian_norm_sq(F)
    assert abs(b - a) <= GROUP_A_TOL * abs(a)
    assert a > 0.0 and b > 0.0


@pytest.mark.parametrize("nx,ny,h", [(1, 5, 0.1), (5, 1, 0.1), (5, 5, 0.0), (5, 5, -0.1)])
def test_both_sides_reject_the_same_von_karman_grids_with_the_same_words(nx, ny, h):
    for py_cls, rs_cls in (
        (o2.VonKarmanBracketPy, physsynth_rs.VonKarmanBracket),
        (o2.AiryStressSolverPy, physsynth_rs.AiryStressSolver),
    ):
        with pytest.raises(ValueError) as py_err:
            py_cls(nx, ny, h)
        with pytest.raises(ValueError) as rs_err:
            rs_cls(nx, ny, h)
        assert str(py_err.value) == str(rs_err.value)


@pytest.mark.parametrize("nx,ny,h", AIRY_GRIDS[:10])
def test_both_airy_solves_are_backward_stable_so_the_gap_is_conditioning(nx, ny, h):
    """The forward gap grows with the grid; the *backward* error does not, on either side.

    That is the whole explanation for why ``AIRY_SOLVE_TOL`` cannot be one small constant. A
    backward-stable solve returns the exact answer to a slightly perturbed problem, so
    ``||B_F f - rhs|| / (||B_F|| ||f||)`` sits at machine precision whatever the grid, while the
    *forward* difference between two such solves is that times the condition number — and a
    biharmonic's condition number grows like ``N^4``. Without this the growing tolerance above
    would look like the port getting worse on finer grids, which is the opposite of the truth.
    """
    py = o2.AiryStressSolverPy(nx, ny, h)
    rs = physsynth_rs.AiryStressSolver(nx, ny, h)
    live = py.mask.ravel()
    wa = np.zeros(py.n_nodes)
    mx = np.full(nx + 1, h)
    mx[0] = mx[-1] = 0.5 * h
    my = np.full(ny + 1, h)
    my[0] = my[-1] = 0.5 * h
    wa[:] = np.kron(my, mx)
    rng = np.random.default_rng(11)
    src = rng.standard_normal(py.n_nodes)
    src[~live] = 0.0
    rhs = wa[live] * src[live]
    norm_b = np.max(np.abs(py.Bf.toarray()).sum(axis=1))
    scale = norm_b * max(np.max(np.abs(py.solve(src)[live])), 1e-300)
    for solver in (py, rs):
        f = np.asarray(solver.solve(src))[live]
        residual = np.max(np.abs(py.Bf @ f - rhs))
        assert residual <= 1e-13 * scale, (
            f"{type(solver).__name__} is not backward stable at {nx}x{ny}: "
            f"{residual / scale:.3e}"
        )
