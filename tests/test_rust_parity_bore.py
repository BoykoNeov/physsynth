"""Rust vs Python for the acoustic bore — Phase 2 batch 3's comparison (the air column half).

The fifth of these files, and it inherits the others' bars unchanged: the acceptance contract is
the physics harness (``docs/dev/rust-migration-plan.md`` §2.1), and this is the *diagnostic* that
catches what the physics bars are too loose to see. The state is required to be bit-identical; the
read-outs that go through ``np.dot``/BLAS are held to the plan's Group A target.

What is new here, and what each new thing is worth:

* **The step takes a caller's hook, and it is a general Python callable.** ``Bore.step(source=...)``
  hands the freshly-updated pressure field to an outside party who mutates it in place, between the
  pressure and momentum sub-steps. The plan deferred this whole batch until ``reed`` existed
  because the answer commits the project for ``bow`` and every continuous exciter after it
  (§12.5). It is exercised three ways below: not passed at all, passed an inert ``lambda p: None``
  (which ``tests/test_reed_stability.py`` also does, so the capability is not the reed's private
  channel), and passed a real corrector.

* **The hook's *position* in the step is load-bearing and no energy test can see it.** Open-end pin,
  then the hook, then the radiating drain, then momentum — so ``U^{n+3/2}`` sees the corrected node
  pressure. A port that ran the hook one line later still oscillates and still balances its books.
  Here it is pinned by requiring bit-identity of a run driven through the hook, which fails
  immediately if the ordering moves.

* **A hook may read the bore while the bore is mid-step**, and that is not free in Rust. A
  ``&mut self`` method holds a mutable borrow of the object for its whole body, so the obvious
  binding refuses ``self.bore.p[0]`` from inside the callback with ``RuntimeError: Already mutably
  borrowed`` — a read the original allows. ``test_the_hook_can_read_the_bore_mid_step`` is that
  case, and it is the one this file would have caught first.

* **A bell at both ends books each end's energy separately.** ``_radiate_node`` accumulates
  ``radiated_energy`` itself, once per end, so a two-ended bell computes ``(E + e_l) + e_r`` and
  never ``E + (e_l + e_r)``. That is a claim about the order of two additions, not about physics,
  and only bit-parity can see it.

* **``Lop``, ``Cmat`` and ``dof`` are real SciPy objects on the instance.** ``tests/helpers.py``,
  ``test_bore_energy.py``, ``test_bore_modal.py`` and ``web/serialize.py`` slice them with fancy
  indexing and hand ``Cmat`` to a generalized ``eigsh``, so the binding builds them once rather
  than handing back triplets. Their **values** are asserted bit-for-bit; their index *ordering* is
  canonicalised first, for the reason ``crates/physsynth-core/src/sparse.rs`` gives.

* **Three private names are on the surface.** ``_bc_left`` is read by ``reed.py``,
  ``_open_left``/``_open_right`` by ``test_bore_radiation.py`` and ``web/serialize.py``. Derived by
  grepping the clients, not by reading the author's intent off the underscore — the body's
  ``_accel`` lesson (§12.2) applied to a new model.
"""

import numpy as np
import pytest

from physsynth.core.bore import C0_AIR, BorePy

physsynth_rs = pytest.importorskip(
    "physsynth_rs", reason="the Rust extension is not built in this environment"
)

# The plan's Group A agreement target. Only the reductions are measured against it; state is exact.
GROUP_A_TOL = 1e-13

L_DEFAULT = 0.6
RADIUS_DEFAULT = 0.008


def _fs(L, N, lam):
    """The sample rate that puts the Courant number exactly at ``lam`` — ``helpers.make_bore``."""
    return C0_AIR / (lam * (L / N))


CASES = [
    dict(N=60, lam=1.0, boundary=("closed", "open")),
    dict(N=60, lam=1.0, boundary=("closed", "open"), sigma=25.0),
    dict(N=60, lam=0.87, boundary=("open", "open")),
    dict(N=45, lam=1.0, boundary="closed"),
    dict(N=60, lam=1.0, boundary=("closed", "radiating"), R_bell=650.0),
    dict(N=60, lam=0.93, boundary=("radiating", "radiating"), R_bell=650.0),
    dict(N=40, lam=1.0, boundary=("closed", "radiating"), R_bell=5.0e4, sigma=3.0),
    dict(N=17, lam=0.71, boundary=("radiating", "closed"), R_bell=650.0, L=0.31,
         radius=0.011, rho0=1.19, c0=340.5),
]


def _build(cls, case):
    kw = dict(case)
    L = kw.pop("L", L_DEFAULT)
    N = kw.pop("N")
    lam = kw.pop("lam")
    kw.setdefault("radius", RADIUS_DEFAULT)
    return cls(L=L, fs=_fs(L, N, lam), N=N, **kw)


def _pair(case):
    return _build(BorePy, case), _build(physsynth_rs.Bore, case)


def _ids(case):
    bc = case["boundary"]
    tag = bc if isinstance(bc, str) else "-".join(bc)
    return f"N{case['N']}-lam{case['lam']}-{tag}"


def _bump(bore, center_frac=0.35):
    """A narrow pressure bump — the initial condition the bore's own tests use."""
    c = center_frac * bore.L
    w = 0.04 * bore.L
    return np.exp(-(((bore.x - c) / w) ** 2))


def _canonical(matrix):
    """SciPy's product kernel does not promise sorted indices; the Rust CSR is always canonical.

    ``crates/physsynth-core/src/sparse.rs`` explains why reproducing SciPy's *ordering* is a bad
    trade. The values are the claim, so both sides are put in canonical form before comparing.
    """
    m = matrix.tocsr().copy()
    m.sort_indices()
    m.sum_duplicates()
    m.eliminate_zeros()
    return m


def _assert_state_identical(py, rs, where):
    assert np.array_equal(py.p, rs.p), f"{where}: pressure field differs"
    assert np.array_equal(py.U, rs.U), f"{where}: U differs"
    assert np.array_equal(py.U_prev, rs.U_prev), f"{where}: U_prev differs"
    assert py.radiated_energy == rs.radiated_energy, f"{where}: radiated_energy differs"
    assert py.n == rs.n, f"{where}: step count differs"


def _rel(a, b):
    scale = max(abs(a), abs(b))
    return 0.0 if scale == 0.0 else abs(a - b) / scale


# -- derived parameters --------------------------------------------------------------------------


@pytest.mark.parametrize("case", CASES, ids=_ids)
def test_every_derived_scalar_is_identical(case):
    py, rs = _pair(case)
    for name in (
        "L", "fs", "N", "radius", "sigma", "R_bell", "rho0", "c0", "h", "k", "lam", "Z0",
        "_bc_left", "_bc_right", "_open_left", "_open_right", "_rad_left", "_rad_right",
    ):
        assert getattr(py, name) == getattr(rs, name), name


@pytest.mark.parametrize("case", CASES, ids=_ids)
def test_every_derived_array_is_identical(case):
    py, rs = _pair(case)
    for name in ("x", "x_u", "S_node", "S_seg", "dof"):
        got, want = getattr(rs, name), getattr(py, name)
        assert np.array_equal(got, want), name
        assert got.dtype == want.dtype, f"{name} dtype: {got.dtype} vs {want.dtype}"


@pytest.mark.parametrize("case", CASES, ids=_ids)
def test_the_boundary_argument_is_echoed_unchanged(case):
    py, rs = _pair(case)
    assert py.boundary == rs.boundary
    # A bare string stays a string and a tuple stays a tuple — `web/serialize.py` reads this back.
    assert type(py.boundary) is type(rs.boundary)


@pytest.mark.parametrize("case", CASES, ids=_ids)
def test_the_pressure_operator_agrees_bit_for_bit(case):
    py, rs = _pair(case)
    for name in ("Lop", "Cmat"):
        a = _canonical(getattr(py, name))
        b = _canonical(getattr(rs, name))
        assert a.shape == b.shape, name
        assert a.nnz == b.nnz, f"{name}: nnz {a.nnz} vs {b.nnz}"
        assert np.array_equal(a.indptr, b.indptr), f"{name}: indptr"
        assert np.array_equal(a.indices, b.indices), f"{name}: indices"
        assert np.array_equal(a.data, b.data), f"{name}: data"


@pytest.mark.parametrize("case", CASES, ids=_ids)
def test_the_operator_survives_the_clients_slicing(case):
    """``bore.Lop[dof][:, dof]`` and ``Cmat`` as ``eigsh``'s ``M`` — the actual access pattern."""
    py, rs = _pair(case)
    for bore in (py, rs):
        dof = bore.dof
        assert bore.Lop[dof][:, dof].shape == (len(dof), len(dof))
        assert bore.Cmat[dof][:, dof].shape == (len(dof), len(dof))
    assert np.array_equal(
        py.Lop[py.dof][:, py.dof].toarray(), rs.Lop[rs.dof][:, rs.dof].toarray()
    )
    # `test_bore_modal.py` asserts symmetry via `abs(Lop - Lop.T).max()`.
    assert abs(py.Lop - py.Lop.T).max() == abs(rs.Lop - rs.Lop.T).max()


# -- construction-time refusals ------------------------------------------------------------------


@pytest.mark.parametrize(
    "kw, fragment",
    [
        (dict(L=0.0, fs=48000.0, N=60), "must all be positive"),
        (dict(L=0.6, fs=0.0, N=60), "must all be positive"),
        (dict(L=0.6, fs=48000.0, N=60, radius=0.0), "must all be positive"),
        (dict(L=0.6, fs=48000.0, N=1), "N must be >= 2"),
        (dict(L=0.6, fs=48000.0, N=60, sigma=-1.0), "sigma (loss) must be >= 0"),
        (dict(L=0.6, fs=48000.0, N=60, R_bell=-1.0), "R_bell (radiation resistance) must be >= 0"),
        (dict(L=0.6, fs=48000.0, N=60, boundary="wrong"), "each boundary end must be one of"),
        (dict(L=0.6, fs=48000.0, N=60, boundary=("closed", "radiating")), "needs R_bell > 0"),
        (dict(L=0.6, fs=1000.0, N=60), "CFL violated"),
    ],
)
def test_the_refusals_fire_on_the_same_input_with_the_same_message(kw, fragment):
    with pytest.raises(ValueError) as py_err:
        BorePy(**kw)
    with pytest.raises(ValueError) as rs_err:
        physsynth_rs.Bore(**kw)
    assert fragment in str(py_err.value)
    assert str(py_err.value) == str(rs_err.value)


def test_a_doubly_wrong_call_reports_the_same_fault():
    """The check *order* is part of the contract: two faults, one message, and it must match."""
    kw = dict(L=0.0, fs=48000.0, N=1, sigma=-1.0, R_bell=-1.0, boundary="wrong")
    with pytest.raises(ValueError) as py_err:
        BorePy(**kw)
    with pytest.raises(ValueError) as rs_err:
        physsynth_rs.Bore(**kw)
    assert "must all be positive" in str(py_err.value)
    assert str(py_err.value) == str(rs_err.value)


# -- state ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("case", CASES, ids=_ids)
def test_set_state_is_bit_identical(case):
    py, rs = _pair(case)
    p0 = _bump(py)
    py.set_state(p0.copy())
    rs.set_state(p0.copy())
    _assert_state_identical(py, rs, "set_state")
    # ...and with a seeded half-node velocity, which takes the other branch of the broadcast.
    u0 = 1e-6 * np.cos(np.arange(py.N))
    py.set_state(p0.copy(), u0.copy())
    rs.set_state(p0.copy(), u0.copy())
    _assert_state_identical(py, rs, "set_state with u0")
    # A scalar `u0` fills, which is the default's path.
    py.set_state(p0.copy(), 2.5e-7)
    rs.set_state(p0.copy(), 2.5e-7)
    _assert_state_identical(py, rs, "set_state with scalar u0")


@pytest.mark.parametrize("case", CASES, ids=_ids)
def test_the_state_is_bit_identical_over_a_long_run(case):
    py, rs = _pair(case)
    p0 = _bump(py)
    py.set_state(p0.copy())
    rs.set_state(p0.copy())
    for step in range(2000):
        py.step()
        rs.step()
        if step % 250 == 0 or step == 1999:
            _assert_state_identical(py, rs, f"step {step}")


@pytest.mark.parametrize("case", CASES, ids=_ids)
def test_set_state_resets_the_far_field_channel(case):
    py, rs = _pair(case)
    p0 = _bump(py)
    py.set_state(p0.copy())
    rs.set_state(p0.copy())
    for _ in range(300):
        py.step()
        rs.step()
    py.set_state(p0.copy())
    rs.set_state(p0.copy())
    assert py.radiated_energy == rs.radiated_energy == 0.0
    assert py.pressure() == rs.pressure() == 0.0
    _assert_state_identical(py, rs, "after reset")


@pytest.mark.parametrize("case", CASES, ids=_ids)
def test_a_step_rebinds_rather_than_overwrites(case):
    """A reference held across a step stays a valid snapshot, and lands under ``U_prev``.

    The §9.3 buffer contract, asserted against both implementations because the binding is the only
    place it could go wrong.
    """
    py, rs = _pair(case)
    py.set_state(_bump(py))
    rs.set_state(_bump(rs))
    for bore in (py, rs):
        held_p = bore.p
        held_u = bore.U
        snapshot = held_p.copy()
        bore.step()
        assert np.array_equal(held_p, snapshot), "the held pressure array was overwritten"
        assert bore.U_prev is held_u, "`U_prev` is not the object `U` was"


@pytest.mark.parametrize("case", CASES, ids=_ids)
def test_writing_through_the_state_reaches_the_model(case):
    """The other half of §9.3: a client writes *into* ``.p`` and the bore must see it."""
    py, rs = _pair(case)
    py.set_state(_bump(py))
    rs.set_state(_bump(rs))
    for bore in (py, rs):
        bore.p[bore.N // 2] += 1.0
    _assert_state_identical(py, rs, "after an in-place write")
    for _ in range(50):
        py.step()
        rs.step()
    _assert_state_identical(py, rs, "50 steps after an in-place write")


# -- the source hook -----------------------------------------------------------------------------


@pytest.mark.parametrize("case", CASES, ids=_ids)
def test_an_inert_hook_is_bit_for_bit_the_undriven_bore(case):
    """``tests/test_reed_stability.py`` asserts this on the Python side with its own lambda."""
    py, rs = _pair(case)
    py.set_state(_bump(py))
    rs.set_state(_bump(rs))
    plain_py, plain_rs = _pair(case)
    plain_py.set_state(_bump(plain_py))
    plain_rs.set_state(_bump(plain_rs))
    for _ in range(200):
        py.step(source=lambda p: None)
        rs.step(source=lambda p: None)
        plain_py.step()
        plain_rs.step()
    _assert_state_identical(py, rs, "inert hook")
    assert np.array_equal(py.p, plain_py.p)
    assert np.array_equal(rs.p, plain_rs.p)


@pytest.mark.parametrize("case", CASES, ids=_ids)
def test_a_driving_hook_is_bit_identical(case):
    """A real corrector, driven for 1,000 steps and fed back through the whole scheme.

    This is where the hook's *position* in the step gets pinned. Move it after the radiating drain
    or after the momentum sub-step and the physics still balances — but the numbers move at once.
    """

    def source(p_next):
        p_next[0] += 1e-2 * np.sin(0.01 * p_next[1] + 0.5)

    py, rs = _pair(case)
    py.set_state(_bump(py))
    rs.set_state(_bump(rs))
    for step in range(1000):
        py.step(source=source)
        rs.step(source=source)
        if step % 200 == 0 or step == 999:
            _assert_state_identical(py, rs, f"driven step {step}")


@pytest.mark.parametrize("case", CASES, ids=_ids)
def test_the_hook_can_read_the_bore_mid_step(case):
    """The hook reads the bore *while the bore is inside* ``step``. ``reed._inject`` does exactly
    this — ``p_old = float(self.bore.p[0])`` — and a binding that holds a mutable borrow of itself
    across the callback refuses it with ``RuntimeError: Already mutably borrowed``.

    What the hook must see is the **uncommitted** state: ``p`` is still ``p^n`` and ``U`` is still
    ``U^{n+1/2}``. Both are asserted, because a binding could plausibly commit early and still pass
    a test that only checked the read did not raise.
    """
    py, rs = _pair(case)
    py.set_state(_bump(py))
    rs.set_state(_bump(rs))
    for bore in (py, rs):
        seen = {}

        def source(p_next, bore=bore, seen=seen):
            seen["p"] = bore.p.copy()
            seen["U"] = bore.U.copy()
            seen["n"] = bore.n
            seen["p_next0"] = float(p_next[0])

        before_p = bore.p.copy()
        before_u = bore.U.copy()
        before_n = bore.n
        bore.step(source=source)
        assert np.array_equal(seen["p"], before_p), "the hook saw a committed pressure field"
        assert np.array_equal(seen["U"], before_u), "the hook saw a committed velocity"
        assert seen["n"] == before_n, "the hook saw an incremented step count"
        assert bore.n == before_n + 1


@pytest.mark.parametrize("case", CASES, ids=_ids)
def test_a_hook_that_raises_does_not_swallow_the_error(case):
    py, rs = _pair(case)
    py.set_state(_bump(py))
    rs.set_state(_bump(rs))

    def boom(p_next):
        raise KeyError("from inside the hook")

    for bore in (py, rs):
        with pytest.raises(KeyError, match="from inside the hook"):
            bore.step(source=boom)


# -- read-outs -----------------------------------------------------------------------------------


@pytest.mark.parametrize("case", CASES, ids=_ids)
def test_the_reductions_agree_to_the_group_a_target(case):
    """``acoustic_energy`` and ``energy`` are ``np.dot``/BLAS on the Python side, so they are held
    to a tolerance rather than to bit-identity (plan §2.1). Everything else here is exact."""
    py, rs = _pair(case)
    py.set_state(_bump(py))
    rs.set_state(_bump(rs))
    worst = 0.0
    for step in range(1500):
        py.step()
        rs.step()
        if step % 100:
            continue
        worst = max(
            worst,
            _rel(py.energy(), rs.energy()),
            _rel(py.acoustic_energy(), rs.acoustic_energy()),
        )
        # These three carry no reduction and are required to be exact.
        assert py.pressure() == rs.pressure()
        assert py.displacement_at(1) == rs.displacement_at(1)
        assert py.pressure_at(-1) == rs.pressure_at(-1)
        assert np.array_equal(py.state, rs.state)
    assert worst < GROUP_A_TOL, f"worst relative disagreement {worst:.2e}"


@pytest.mark.parametrize("case", CASES, ids=_ids)
def test_state_is_a_copy_on_both_sides(case):
    py, rs = _pair(case)
    py.set_state(_bump(py))
    rs.set_state(_bump(rs))
    for bore in (py, rs):
        taken = bore.state
        taken[0] = 1e9
        assert bore.p[0] != 1e9, "`state` handed back the live array"


def test_a_two_ended_bell_books_each_end_separately():
    """``_radiate_node`` accumulates ``radiated_energy`` itself, once per end, so a bell at both
    ends computes ``(E + e_l) + e_r``. A port that summed the ends first would agree to 1e-16 and
    disagree in the last bit — which is exactly what this file is for.
    """
    case = dict(N=31, lam=0.8, boundary=("radiating", "radiating"), R_bell=650.0)
    py, rs = _pair(case)
    py.set_state(_bump(py, 0.4))
    rs.set_state(_bump(rs, 0.4))

    b = 0.5 / py.R_bell
    summed_first = 0.0
    for _ in range(2000):
        p_left, p_right = float(py.p[0]), float(py.p[-1])
        py.step()
        rs.step()
        u_l = b * (float(py.p[0]) + p_left)
        u_r = b * (float(py.p[-1]) + p_right)
        summed_first += py.k * py.R_bell * u_l * u_l + py.k * py.R_bell * u_r * u_r

    assert py.radiated_energy == rs.radiated_energy
    # The physics agrees to any tolerance anyone would ask for...
    assert abs(py.radiated_energy - summed_first) / py.radiated_energy < 1e-12
    # ...and the last bits do not, which is why the per-node booking had to be reproduced.
    assert py.radiated_energy != summed_first, (
        "the two accumulation orders did not diverge here -- this case cannot see the claim"
    )


def test_a_bore_with_a_resistance_but_no_radiating_end_still_rotates_the_readout():
    """``R_bell > 0`` with both ends non-radiating is a legal bore, and the original still runs its
    ``_apply_radiating_ends`` early-exit *after* the ``R_bell <= 0`` test — so the ``U_out`` pair
    rotates while nothing radiates. A port that keyed the early exit on the *ends* instead of on
    ``R_bell`` passes every other test in this file."""
    case = dict(N=40, lam=1.0, boundary=("closed", "open"), R_bell=650.0)
    py, rs = _pair(case)
    py.set_state(_bump(py))
    rs.set_state(_bump(rs))
    for _ in range(100):
        py.step()
        rs.step()
    assert py.radiated_energy == rs.radiated_energy == 0.0
    assert py.pressure() == rs.pressure() == 0.0
    _assert_state_identical(py, rs, "resistance without a radiating end")


# -- the physics bars, asserted on the Rust model itself ------------------------------------------


@pytest.mark.parametrize("case", CASES, ids=_ids)
def test_the_rust_bore_meets_the_projects_energy_bar(case):
    """The acceptance contract, restated against the port: lossless conserves, viscous decreases.

    Not a comparison — the point of the plan's §2.1 is that agreement is the diagnostic and the
    physics harness is the contract. If the two implementations ever *both* drift, the tests above
    stay green and this one does not.
    """
    _, rs = _pair(case)
    rs.set_state(_bump(rs))
    e0 = rs.energy()
    previous = e0
    for _ in range(2000):
        rs.step()
        now = rs.energy()
        if case.get("sigma", 0.0) > 0.0:
            assert now <= previous * (1.0 + 1e-12), "a viscous tube gained energy"
        else:
            assert abs(now - e0) / e0 < 1e-10, "lossless drift exceeds the bar"
        previous = now
