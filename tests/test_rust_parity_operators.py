"""Rust vs Python for the difference operators — the migration's comparison, not its acceptance.

The sibling of ``test_rust_parity.py``, one phase on. The acceptance bar is still the physics
harness (``docs/dev/rust-migration-plan.md`` §2.1); this file is the diagnostic that catches what
the bars are too loose to see. For an operator that matters especially: a transcription slip in a
boundary-adjacent row of ``biharmonic_matrix`` changes the stiff string's inharmonicity in the
fourth decimal and passes every modal test in the repo.

Three bars, and which one applies is decided by **whether the computation contains a reduction**:

* **The pointwise differences must agree bit-for-bit.** No reduction, no library call — IEEE-754
  fixes the answer exactly once the *operation order* matches, and the Rust kernels are written out
  longhand in NumPy's order for that reason. ``np.array_equal``, not ``allclose``.
* **The assembled matrices must agree bit-for-bit too**, in `data`, `indices`, `indptr` and `nnz`.
  Each entry of ``D2 @ D2`` is a sum of up to three products, so it *is* a reduction — but a short
  one, over a deterministic order that both sides can spell the same way. Checked here at eight
  grid sizes.
* **``inner`` and ``norm2`` may differ at ~1e-15 relative.** They call ``np.dot``, which goes
  through BLAS, which accumulates in an order no portable loop reproduces. Held to the plan's
  Group A target of 1e-13.

**One structural difference is deliberate and is normalised away below.** SciPy's
``biharmonic_matrix`` comes back with ``has_sorted_indices == False`` and its columns in descending
order — an artifact of SciPy's SMMP kernel, whose output list is a stack — while
``free_beam_stiffness``, which reaches its product through a transpose, comes back sorted. The Rust
side is canonical in both cases. Reproducing SciPy's split would mean reimplementing a SciPy
internal and pinning the port to a detail a point release is free to change; both spellings
describe the same matrix, and nothing in this repo reads ``.data`` or ``.indices`` off these
objects. So both sides are canonicalised before comparison, and the *values* are then required to
be exactly equal — which is the part that can actually be wrong.
"""

import numpy as np
import pytest
from scipy import sparse

from physsynth.core import operators as ops

physsynth_rs = pytest.importorskip(
    "physsynth_rs", reason="the Rust extension is not built in this environment"
)

# The plan's Group A agreement target. Only the two reductions are measured against it; everything
# else on this page is exact.
GROUP_A_TOL = 1e-13

# Grid sizes. 2 is the smallest the builders accept (one interior unknown, so `D2` is 1x1 and the
# matmul has a single term); 3 is the first size where `free_beam_stiffness` has a structural zero
# in the corner; the rest are ordinary. Both a power of two and an odd size are present because
# h = 1/N is exact for one and not the other, and `1/(h*h)` rounds differently as a result.
SIZES = [2, 3, 4, 5, 8, 16, 33, 64]


def _fields(n_nodes, seed=20260826):
    """A structured field and a random one, on ``n_nodes`` nodes.

    Both, deliberately. The structured field is what the physics uses and is smooth enough that a
    sign slip could cancel; the random one has no structure for a bug to hide in.
    """
    x = np.linspace(0.0, 1.0, n_nodes)
    smooth = np.sin(3.0 * np.pi * x) + 0.4 * x * x
    rng = np.random.default_rng(seed)
    return [smooth, rng.standard_normal(n_nodes)]


def _canonical(m):
    """A SciPy CSR in canonical form: duplicates summed, indices sorted, structure unchanged."""
    m = m.tocsr().copy()
    m.sum_duplicates()
    m.sort_indices()
    return m


def _assert_same_matrix(rs, py, what):
    """Structure and values, both exactly. No tolerance anywhere in here."""
    rs, py = _canonical(rs), _canonical(py)
    assert rs.shape == py.shape, f"{what}: shape {rs.shape} vs {py.shape}"
    assert rs.nnz == py.nnz, f"{what}: nnz {rs.nnz} vs {py.nnz}"
    assert np.array_equal(rs.indptr, py.indptr), f"{what}: indptr differs"
    assert np.array_equal(rs.indices, py.indices), f"{what}: indices differ"
    assert np.array_equal(rs.data, py.data), (
        f"{what}: values are not bit-identical; worst |delta| = "
        f"{np.abs(rs.data - py.data).max():.3e}"
    )
    # Index width is part of the interface, not an internal: SciPy picks int32 for every matrix
    # this project builds, and handing back int64 would silently change `.indices.dtype` under a
    # caller that never asked for it.
    assert rs.indices.dtype == py.indices.dtype, f"{what}: index dtype differs"


# -- the pointwise differences: exact ------------------------------------------------------------


@pytest.mark.parametrize("n_nodes", [5, 6, 9, 17, 65])
@pytest.mark.parametrize(
    "name", ["delta_x_forward", "delta_x_backward", "delta_xx", "delta_xxxx"]
)
def test_pointwise_differences_are_bit_identical(name, n_nodes):
    h = 1.0 / (n_nodes - 1)
    rs = getattr(physsynth_rs, name)
    py = getattr(ops, f"{name}_py")
    for u in _fields(n_nodes):
        a, b = rs(u, h), py(u, h)
        assert a.shape == b.shape, f"{name}: shape {a.shape} vs {b.shape}"
        assert np.array_equal(a, b), (
            f"{name} on {n_nodes} nodes is not bit-identical; worst |delta| = "
            f"{np.abs(a - b).max():.3e} — check the evaluation order in ops.rs against NumPy's"
        )


@pytest.mark.parametrize("n_nodes", [8, 9, 32, 33, 128, 129, 1024])
def test_the_fourth_difference_divisor_survives_a_sweep_of_h(n_nodes):
    # `delta_xxxx` divides by `h**4`, which Python computes with libm's `pow` (correctly rounded)
    # and a chain of multiplications does not: measured over h = 1/N for N = 2..3999, `h**4` and
    # `h*h*h*h` disagree in 1400 cases. Rust says `powf(4.0)` for exactly that reason, which makes
    # this the one kernel in the module whose exactness rests on two libms agreeing rather than on
    # IEEE-754 alone. Hence a sweep rather than a single size: if a platform's `pow` differs, this
    # is where it shows, and it shows as a 1-ulp mismatch rather than as a physics failure.
    h = 1.0 / (n_nodes - 1)
    for u in _fields(n_nodes):
        assert np.array_equal(physsynth_rs.delta_xxxx(u, h), ops.delta_xxxx_py(u, h))


def test_the_two_first_differences_are_the_same_function_on_both_sides():
    # `delta_x_backward` exists for notational symmetry in the energy proofs, not as a different
    # computation. If a port ever made them differ, every proof that swaps one for the other would
    # quietly stop being about the scheme that runs.
    u = _fields(12)[0]
    h = 0.1
    assert np.array_equal(
        physsynth_rs.delta_x_forward(u, h), physsynth_rs.delta_x_backward(u, h)
    )
    assert np.array_equal(ops.delta_x_forward_py(u, h), ops.delta_x_backward_py(u, h))


def test_a_too_short_field_yields_an_empty_array_on_both_sides():
    # NumPy's slicing returns an empty array rather than raising, and the Rust kernels document a
    # precondition and panic. A panic at the interpreter boundary surfaces as PanicException, so
    # the binding guards the length instead — this is the test that the guard matches NumPy rather
    # than merely avoiding the panic.
    for name, need in [
        ("delta_x_forward", 2),
        ("delta_x_backward", 2),
        ("delta_xx", 3),
        ("delta_xxxx", 5),
    ]:
        for n_nodes in range(need):
            u = np.zeros(n_nodes)
            rs = getattr(physsynth_rs, name)(u, 0.5)
            py = getattr(ops, f"{name}_py")(u, 0.5)
            assert rs.shape == py.shape == (0,), f"{name} on {n_nodes} nodes: {rs.shape}"


# -- the reductions: 1e-13 -----------------------------------------------------------------------


@pytest.mark.parametrize("n_nodes", [4, 17, 129, 1025])
def test_the_inner_product_agrees_to_the_group_a_target(n_nodes):
    h = 1.0 / (n_nodes - 1)
    f, g = _fields(n_nodes)
    for a, b in ((f, g), (g, f), (f, f)):
        rs = physsynth_rs.inner(a, b, h)
        py = ops.inner_py(a, b, h)
        assert abs(rs - py) <= GROUP_A_TOL * max(abs(rs), abs(py), 1e-300)
    rs = physsynth_rs.norm2(f, h)
    py = ops.norm2_py(f, h)
    assert abs(rs - py) <= GROUP_A_TOL * max(abs(rs), abs(py))
    assert rs >= 0.0


def test_inner_is_exactly_norm2_when_the_operands_coincide():
    # Not a tautology across the boundary: `norm2` is a separate binding entry point, and the two
    # would drift apart if it ever grew its own summation.
    f = _fields(65)[0]
    assert physsynth_rs.inner(f, f, 0.01) == physsynth_rs.norm2(f, 0.01)


# -- the assembled matrices: exact ---------------------------------------------------------------


@pytest.mark.parametrize("n", SIZES)
def test_second_difference_matrix_is_bit_identical(n):
    h = 1.0 / n
    _assert_same_matrix(
        _rebuild(physsynth_rs.second_difference_matrix_csr(n, h)),
        ops.second_difference_matrix_py(n, h),
        f"second_difference_matrix(N={n})",
    )


@pytest.mark.parametrize("n", SIZES)
def test_biharmonic_matrix_is_bit_identical(n):
    # The one that pays for this whole file. B is `D2 @ D2`, so its entries are genuine three-term
    # sums and its boundary-adjacent diagonal (5/h^4, not 6/h^4) is produced by the product rather
    # than written down — precisely the kind of thing a transcription gets subtly wrong.
    h = 1.0 / n
    _assert_same_matrix(
        _rebuild(physsynth_rs.biharmonic_matrix_csr(n, h)),
        ops.biharmonic_matrix_py(n, h),
        f"biharmonic_matrix(N={n})",
    )


@pytest.mark.parametrize("n", SIZES)
def test_free_beam_stiffness_is_bit_identical(n):
    h = 1.0 / n
    k_rs, w_rs = physsynth_rs.free_beam_stiffness_csr(n, h)
    k_py, w_py = ops.free_beam_stiffness_py(n, h)
    _assert_same_matrix(_rebuild(k_rs), k_py, f"free_beam_stiffness K (N={n})")
    _assert_same_matrix(_rebuild(w_rs), w_py, f"free_beam_stiffness W (N={n})")


@pytest.mark.parametrize("n", SIZES)
def test_the_matrices_agree_on_a_non_unit_grid_spacing(n):
    # Every size above uses h = 1/N, where the reciprocal is a clean power of two half the time.
    # A physical string is 0.65 m long; `1/(h*h)` rounds differently there, and an operator that
    # only agrees on the tidy grid is an operator that agrees by accident.
    h = 0.65 / n
    _assert_same_matrix(
        _rebuild(physsynth_rs.second_difference_matrix_csr(n, h)),
        ops.second_difference_matrix_py(n, h),
        f"D2 on a 0.65 m grid (N={n})",
    )
    _assert_same_matrix(
        _rebuild(physsynth_rs.biharmonic_matrix_csr(n, h)),
        ops.biharmonic_matrix_py(n, h),
        f"B on a 0.65 m grid (N={n})",
    )
    k_rs, w_rs = physsynth_rs.free_beam_stiffness_csr(n, h)
    k_py, w_py = ops.free_beam_stiffness_py(n, h)
    _assert_same_matrix(_rebuild(k_rs), k_py, f"K on a 0.65 m grid (N={n})")
    _assert_same_matrix(_rebuild(w_rs), w_py, f"W on a 0.65 m grid (N={n})")


@pytest.mark.parametrize("n", [-1, 0, 1])
def test_both_sides_reject_a_grid_too_coarse_to_have_an_interior(n):
    # Same exception type, deliberately different text: the Python builders fall through to NumPy's
    # "negative dimensions are not allowed", which is a leak rather than a message. What callers
    # can depend on is `ValueError`, and that is what is asserted.
    for name in ("second_difference_matrix", "biharmonic_matrix", "free_beam_stiffness"):
        with pytest.raises(ValueError):
            getattr(physsynth_rs, f"{name}_csr")(n, 0.5)
        with pytest.raises(ValueError):
            getattr(ops, f"{name}_py")(n, 0.5)


def test_the_binding_hands_back_triplets_not_a_matrix():
    # The seam this phase is built on (plan §3.1): `physsynth-core` must not know what SciPy is, so
    # the binding returns `(data, indices, indptr, shape)` and the shim in `operators.py` rebuilds.
    # If that ever changes to returning an object, the shim becomes a silent no-op and this file's
    # canonicalisation would start comparing a matrix with itself.
    out = physsynth_rs.second_difference_matrix_csr(8, 0.125)
    assert isinstance(out, tuple) and len(out) == 4
    data, indices, indptr, shape = out
    assert data.dtype == np.float64
    assert indices.dtype == np.int32 and indptr.dtype == np.int32
    assert shape == (7, 7)
    assert indptr[-1] == len(data) == len(indices)


def _rebuild(triplets):
    """The shim, spelled out here so this file does not depend on the swap being installed."""
    data, indices, indptr, shape = triplets
    return sparse.csr_matrix((data, indices, indptr), shape=shape)
