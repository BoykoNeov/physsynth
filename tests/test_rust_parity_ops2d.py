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
