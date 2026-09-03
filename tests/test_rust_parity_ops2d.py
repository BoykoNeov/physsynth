"""The 2-D builders, held against SciPy and against themselves — what unit 5's deletion left.

This file was "Rust vs Python for the 2-D builders": 53 tests, nearly all of them one call to
``operators2d``'s Python body against one call to the binding. Unit 5's deletion (plan §43) removed
the Python body, so most of those tests lost the second of their two sides and are retired. What is
left is the minority whose comparand was never the Python implementation:

* **SciPy is still here**, and it is the sharpest comparand this file has. ``AiryStressSolver``'s
  operator is a Gram product whose two bracketings are genuinely different sums, and the shipped
  spelling can still be checked entry-for-entry against SciPy's own sparse product. So can the
  claim that its solve differs from SuperLU's by conditioning and not by an assembly error.
* **The binding's output is still a claim** — that what comes out of the extension is already in
  canonical column order, so `operators2d.py`'s shim is rebuilding a matrix rather than fixing one.
* **The outline's margin is still a measurement about the shipped model.** ``guitar_mask`` is a
  strict ``|x| < half(y)`` where ``half`` is built from ``sin``, so a last bit of one transcendental
  decides whether a node is an unknown at all. That used to be a statement about two
  implementations agreeing; with one implementation it is a statement about **portability** — how
  much room the shipped outlines leave before a different machine's ``sin`` moves a node. Same
  numbers, and the reason to keep measuring them has not changed.

**What was retired, and what carries those claims now.** Every exactness comparison on a builder —
the masks, the grid, ``embed``, the Laplacian, the biharmonic, the orthotropic biharmonic, the two
free-plate stiffnesses, the five 1-D differences, the bracket's operators and its evaluation, the
Airy operator, the paired rejection messages — was a diagnostic with two implementations in it.
The *correctness* of every one of them is asserted natively in
``crates/physsynth-core/tests/ops2d.rs``, which is where the mask, the index map, the five-point
stencil, the squared-Laplacian eigenvalues, the orthotropic mode eigenvalue, the free plate's
rigid-body nullspace, the five stencils and the Gram association's value-invariance are all
checked against what they are supposed to *be* rather than against a second spelling of themselves.
That is the trade the deletion makes everywhere: a comparison goes, a property stays.

Two retirements are worth naming because their claim moved rather than ended:

* ``test_the_airy_operator_is_bit_identical`` compared ``Bf`` against the Python assembly. The
  strictly stronger version of it survives below in
  ``test_the_gram_association_is_a_different_sum_and_this_finds_the_witness``, which builds the
  right-associated product **out of SciPy** on every shipped grid and asserts the Rust operator
  equals it — and does so having first proved the two associations differ somewhere, so the
  comparison cannot go vacuous.
* ``test_the_python_solver_on_the_rust_factorization_is_bit_identical`` was §24.4's manoeuvre:
  hold the factorization constant so only the two assemblies vary. There is one assembly now, so
  the manoeuvre has nothing to separate.
"""

import numpy as np
import pytest
from scipy import sparse
from scipy.sparse.linalg import splu

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


def _assert_identical(a_m, b_m, what):
    """Bit-for-bit, including the stored order."""
    a = a_m.tocsr()
    b = b_m.tocsr()
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


# -- the grid ------------------------------------------------------------------------------------


@pytest.mark.parametrize("n", SIZES)
def test_the_grid_endpoints_are_exact(n):
    # `np.linspace` overwrites the last entry with the endpoint rather than computing it, so
    # `X[0, -1]` is `+half` exactly. Reproducing the overwrite is the reason this passes; computing
    # `i * h - half` for the last node would be off by an ulp, and that ulp decides rim membership.
    # (Named `..._on_both_sides` until unit 5's deletion, when the other side stopped existing.)
    X, Y, _ = physsynth_rs.grid_coords(n, 0.15)
    assert X[0, 0] == -0.15
    assert X[0, -1] == 0.15
    assert Y[0, 0] == -0.15
    assert Y[-1, 0] == 0.15


def test_inner2d_is_exactly_norm2_when_the_operands_coincide():
    rng = np.random.default_rng(7)
    f = rng.standard_normal(64)
    assert physsynth_rs.inner2d(f, f, 0.5) == physsynth_rs.norm2_2d(f, 0.5)


# -- the guitar outline: how much room a last bit of `sin` has ------------------------------------
#
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
    half = o2.guitar_scale(Lx, waist, asym) * o2.guitar_half_width(t, waist, asym)
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

    Written in Phase 5 as the proof behind an exact mask comparison between two implementations.
    That comparison is gone with unit 5, and the measurement is **not**, because what it really
    says is a portability claim about the shipped model: ``half`` is built from ``sin``, ``sin``
    is the platform's, and a node that sits a few ulps from the rim is a node whose live/dead
    status is decided by which machine assembled the mask. A mask with one node fewer conserves
    energy just as beautifully as the right one, so nothing downstream would notice.
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
    half = o2.guitar_scale(0.37, 0.0, 0.0) * o2.guitar_half_width(t, 0.0, 0.0)

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


# -- the matrices: what comes out of the binding, before the shim touches it ----------------------
#
# Grids the plate family actually ships on, plus one where `1/h**2` is exactly representable and one
# where it is not -- the distinction that decides whether the orthotropic assembly collapses onto
# `L @ L` bit-for-bit, and the one a single-grid check would have missed.
MATRIX_GRIDS = [(6, 5, 0.037), (8, 8, 0.05), (12, 9, 0.0125), (16, 16, 0.03125), (20, 13, 0.1 / 7)]


@pytest.mark.parametrize("nx,ny,h", MATRIX_GRIDS)
def test_the_rust_product_is_canonical_before_anything_sorts_it(nx, ny, h):
    """What comes *out of the binding* is already in canonical column order.

    This began as the guard on an exact index comparison — both sides of that comparison were
    sorted, the Python reference by ``portable.canonical`` and the Rust side by ``Csr::from_rows``,
    so on its own it would have passed even if the binding emitted arbitrary rows and the shim
    tidied them up. The comparison is gone and the claim is not: a CSR matvec sums a row in stored
    order and ``Plate`` forms ``B @ u`` twice per timestep, so a Rust-side reordering would change
    the shipped trajectory. It fails here, with an obvious cause, rather than as a moved last digit
    somewhere downstream.
    """
    mask = o2.rectangle_mask(nx, ny)
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
                "on the Python shim to sort it, which would make every stored-order claim in this "
                "project a test of `sort_indices`"
            )


ONE_D_SIZES = [2, 3, 4, 5, 8, 9, 16, 17, 32]
ONE_D_STEPS = [0.1, 0.0625, 0.0375, 1.0 / 3.0, 0.006]


@pytest.mark.parametrize("n", ONE_D_SIZES)
@pytest.mark.parametrize("h", ONE_D_STEPS)
def test_the_collocated_and_centered_differences_are_not_the_same_matrix(n, h):
    # The pair the module is most likely to confuse: same interior stencil, and the ONLY difference
    # is whether the two end rows exist. A port that used one for the other would be caught by
    # nothing else here -- both are symmetric, both annihilate constants on the interior, and the
    # bracket's identity would survive because rim-vanishing fields never weigh those rows.
    rs_coll = _rebuild(physsynth_rs.collocated_d2_1d_csr(n, h))
    rs_cent = _rebuild(physsynth_rs.centered_d2_1d_csr(n, h))
    assert rs_coll.nnz + 4 == rs_cent.nnz
    assert rs_coll.indptr[0] == rs_coll.indptr[1], "row 0 must be empty, not zero-valued"


# -- the von Karman half: the Gram association, and the SuperLU gap -------------------------------
#
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
    bracket = o2.VonKarmanBracket(nx, ny, h)
    acell = sparse.csr_matrix(bracket.Acell)
    scatter = acell.T
    assert sparse.isspmatrix_csc(scatter), "SciPy's csr.T is a CSC; the lemma is about that"
    gather = sparse.csr_matrix(acell.T)
    rng = np.random.default_rng(4)
    for _ in range(8):
        v = rng.standard_normal(acell.shape[0])
        assert np.array_equal(scatter @ v, gather @ v)


def test_the_gram_association_is_a_different_sum_and_this_finds_the_witness():
    """The Airy operator, built out of SciPy and compared entry for entry against the crate.

    Two things at once, and neither works without the other. First, that the two bracketings of
    ``Lc_rᵀ Wa Lc_r`` are genuinely different sums — searched for rather than asserted at a
    hand-picked grid (§26.6), because they agree on most grids and a test that landed on an
    agreeing one would go green having asserted nothing. Second, that the **shipped** operator is
    the right-associated one, checked against SciPy's own sparse product on every grid the suite
    builds.

    That second half is what makes this the strongest surviving statement about the Airy assembly,
    and stronger than the exact comparison it replaces: the old one compared the crate against a
    Python transcription of the same recipe, and this compares it against SciPy's kernels doing
    the arithmetic themselves.
    """
    witnesses = []
    unsorted = 0
    for nx, ny, h in AIRY_GRIDS:
        ix = sparse.identity(nx + 1, format="csr")
        iy = sparse.identity(ny + 1, format="csr")
        Lc = sparse.kron(iy, o2._clamped_d2_1d(nx, h)) + sparse.kron(
            o2._clamped_d2_1d(ny, h), ix
        )
        mx = np.full(nx + 1, h)
        mx[0] = mx[-1] = 0.5 * h
        my = np.full(ny + 1, h)
        my[0] = my[-1] = 0.5 * h
        Wa = sparse.diags(np.kron(my, mx), format="csr")
        cols = o2.rectangle_mask(nx, ny).ravel()
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
        _assert_identical(o2.AiryStressSolver(nx, ny, h).Bf, right, f"shipped {nx}x{ny}")
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


def _area_weights(nx, ny, h):
    """``Wa = kron(m_y, m_x)`` — interior ``h²``, edge ``h²/2``, corner ``h²/4``."""
    mx = np.full(nx + 1, h)
    mx[0] = mx[-1] = 0.5 * h
    my = np.full(ny + 1, h)
    my[0] = my[-1] = 0.5 * h
    return np.kron(my, mx)


def _superlu_airy_solve(nx, ny, h, rs, source):
    """``AiryStressSolver.solve``, driven through SuperLU instead of the crate's own LU.

    The Python class did exactly this — ``splu(self.Bf)`` factored once, the ``Wa``-weighted
    interior load, the rim embedded back at zero. It is transcribed here rather than imported
    because the class it belonged to is gone, and the *solver* is the whole subject: the operator
    comes from the crate, so the only difference between this and ``rs.solve`` is which
    factorization ran.
    """
    live = np.asarray(rs.mask).ravel()
    rhs = _area_weights(nx, ny, h)[live] * np.asarray(source, dtype=float).ravel()[live]
    out = np.zeros(rs.n_nodes)
    out[live] = splu(sparse.csc_matrix(rs.Bf)).solve(rhs)
    return out


@pytest.mark.parametrize("nx,ny,h", AIRY_GRIDS + AIRY_LARGE_GRIDS)
def test_the_airy_solve_is_the_measured_superlu_tolerance(nx, ny, h):
    """The crate's sparse LU against SuperLU, on the same operator — Group D's residue, measured.

    Still a two-sided comparison after unit 5, and that is the point: the two sides are two
    *solvers*, not two implementations of this project's code, so deleting the Python body took
    nothing away from it. Both are handed ``rs.Bf``.
    """
    rs = physsynth_rs.AiryStressSolver(nx, ny, h)
    live = np.asarray(rs.mask).ravel()
    rng = np.random.default_rng(11)
    for _ in range(3):
        src = rng.standard_normal(rs.n_nodes)
        src[~live] = 0.0
        f_lu = _superlu_airy_solve(nx, ny, h, rs, src)
        f_rs = np.asarray(rs.solve(src))
        scale = np.max(np.abs(f_lu))
        assert np.max(np.abs(f_rs - f_lu)) <= airy_solve_tol(nx, ny) * scale
        # The rim comes back exactly zero on both sides -- that is a structural claim, not a
        # tolerance, and it is what makes `F` a legal bracket argument.
        assert np.array_equal(f_rs[~live], np.zeros(int((~live).sum())))


@pytest.mark.parametrize("nx,ny,h", AIRY_GRIDS[:10])
def test_both_airy_solves_are_backward_stable_so_the_gap_is_conditioning(nx, ny, h):
    """The forward gap grows with the grid; the *backward* error does not, on either side.

    That is the whole explanation for why ``airy_solve_tol`` cannot be one small constant. A
    backward-stable solve returns the exact answer to a slightly perturbed problem, so
    ``||B_F f - rhs|| / (||B_F|| ||f||)`` sits at machine precision whatever the grid, while the
    *forward* difference between two such solves is that times the condition number — and a
    biharmonic's condition number grows like ``N^4``. Without this the growing tolerance above
    would look like the port getting worse on finer grids, which is the opposite of the truth.
    """
    rs = physsynth_rs.AiryStressSolver(nx, ny, h)
    live = np.asarray(rs.mask).ravel()
    bf = sparse.csr_matrix(rs.Bf)
    rng = np.random.default_rng(11)
    src = rng.standard_normal(rs.n_nodes)
    src[~live] = 0.0
    rhs = _area_weights(nx, ny, h)[live] * src[live]
    norm_b = np.max(np.abs(bf.toarray()).sum(axis=1))
    scale = norm_b * max(np.max(np.abs(np.asarray(rs.solve(src))[live])), 1e-300)
    for name, f_full in (
        ("SuperLU", _superlu_airy_solve(nx, ny, h, rs, src)),
        ("the crate", np.asarray(rs.solve(src))),
    ):
        residual = np.max(np.abs(bf @ f_full[live] - rhs))
        assert residual <= 1e-13 * scale, (
            f"{name} is not backward stable at {nx}x{ny}: {residual / scale:.3e}"
        )
