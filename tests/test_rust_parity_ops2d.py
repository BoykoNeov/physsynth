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
