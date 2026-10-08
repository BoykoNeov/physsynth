"""Properties of the **binding**, which no physics bar and no native Rust bar can see.

This file is the residue of ``test_rust_parity.py`` after the migration's first deletion
(``docs/dev/rust-migration-plan.md`` §39.5), and it **outlived** that file — the original was
deleted with unit 8, when the last Python twin it compared against went. It exists because two
kinds of test were mixed in there and only one of them dies when a Python reference implementation
is deleted:

* a **comparison** ("Python and Rust agree") has nothing left to compare and goes;
* a **property** ("a reference to ``.u`` held across a step is a snapshot") is a statement about
  the object handed to Python, and is as true and as necessary afterwards as before.

The properties below are permanent. They cannot move to ``crates/physsynth-core/tests/`` — that
crate has no Python objects, so it has no buffer whose lifetime it could check and no PyO3
signature whose argument mapping it could probe. §39.3 says the same of ``connection.py`` and the
airbox wrapper tier for the same underlying reason: what only exists at the binding can only be
checked from Python.

Deliberately **not** named ``test_rust_parity_*``: that prefix meant "must run unflagged, because
it builds both sides itself", and nothing here ever built a Python side. The prefix stopped meaning
anything when phase A removed the flag (retirement plan §21), and so did the shard script's
exclusion that read it.
"""

import importlib

import numpy as np
import pytest
import scipy.sparse.linalg
from scipy import sparse

import physsynth.core
from physsynth.core import connection, plate, string_stiff
from physsynth.core.exciter import triangular_pluck

physsynth_rs = pytest.importorskip(
    "physsynth_rs", reason="the Rust extension is not built in this environment"
)
IdealString = physsynth_rs.IdealString

L_DEFAULT = 1.0
T_DEFAULT = 200.0
RHO_DEFAULT = 0.005  # -> c = 200 m/s


def _params(*, N=100, lam=1.0, boundary="fixed", sigma=0.0):
    c = np.sqrt(T_DEFAULT / RHO_DEFAULT)
    return {
        "L": L_DEFAULT,
        "T": T_DEFAULT,
        "rho": RHO_DEFAULT,
        "fs": c * N / (L_DEFAULT * lam),
        "N": N,
        "boundary": boundary,
        "sigma": sigma,
    }


# -- what `state` hands out ----------------------------------------------------------------------


def test_state_is_a_copy_not_a_view():
    s = IdealString(**_params(N=8))
    s.set_state(triangular_pluck(s.x, s.L, 0.3 * s.L, amplitude=1e-3))
    snapshot = s.state
    snapshot[3] = 12345.0
    assert s.u[3] != 12345.0, "`state` handed out a live view"


def test_set_state_accepts_a_plain_list():
    # Not a numpy array: a caller holding parsed JSON has none (the Python viewer was that caller
    # until retirement plan §23.19 deleted it).
    s = IdealString(**_params(N=4))
    s.set_state([0.0, 1.0, 2.0, 1.0, 0.0])
    assert np.array_equal(s.u, np.array([0.0, 1.0, 2.0, 1.0, 0.0]))


# -- buffer lifetime -----------------------------------------------------------------------------


def test_a_reference_to_u_held_across_a_step_is_a_snapshot():
    # `step()` rebinds `self.u`; it does not write into it. So a reference taken before the step
    # keeps showing that step's values. The binding reproduces this by owning NumPy arrays rather
    # than Rust `Vec`s — a zero-copy view over a reallocated `Vec` would *look* right here
    # (measured: it still reads the old contents) while being a use-after-free.
    s = IdealString(**_params(N=16))
    s.set_state(triangular_pluck(s.x, s.L, 0.3 * s.L, amplitude=1e-3))
    held = s.u
    before = np.array(held, copy=True)

    s.step()

    assert np.array_equal(held, before), "the held reference changed under a step"
    assert not np.array_equal(s.u, before), "the string did not actually advance"
    assert s.u_prev is held, "u_prev after a step must be the very object u was"


def test_an_in_place_write_through_u_reaches_the_string():
    # This is exactly what a bridge connection does — `self.string.u[-1] -= beta_s * F` — and it
    # is the reason `.u` cannot be handed out as a copy.
    s = IdealString(**_params(N=16, boundary=("fixed", "free")))
    s.set_state(triangular_pluck(s.x, s.L, 0.3 * s.L, amplitude=1e-3))
    before = s.energy()

    s.u[-1] -= 1e-3

    assert s.energy() != before, "a write through `.u` did not reach the string"
    assert s.state[-1] == s.u[-1]
    s.step()  # and the modified state is what gets stepped, without complaint


def test_the_state_arrays_can_be_replaced_wholesale():
    # The retired `tests/test_collision_modal.py` did this to a string it owned
    # (`bar.string.u = uf.copy()`), so assignment has to work, not just mutation.
    s = IdealString(**_params(N=8))
    fresh = np.linspace(0.0, 1.0, 9)
    s.u = fresh.copy()
    s.u_prev = fresh.copy()
    s.n = 0
    assert np.array_equal(s.u, fresh)
    s.step()
    assert s.n == 1


# -- an omitted keyword and an explicit `None` are different arguments (plan §24.7) ---------------
#
# Found while writing the beam's parity file, and it was true of every binding in the crate. PyO3
# maps a Python `None` and a missing argument onto the same Rust `Option::None`, so a binding
# written the obvious way treats `boundary=None` as "not supplied" and quietly builds the DEFAULT
# boundary, while the Python original rejected it. Nothing caught it: no parity file passed `None`
# to a constructor, because `None` is not a plausible boundary and nobody thinks to try it.
#
# The fix is `Option<Option<_>>`, and the arm order is the surprising half -- PyO3 wraps the
# *default expression*, so `Some(None)` means "argument omitted" and a bare `None` is the caller's
# literal. That is exactly the kind of thing that gets silently inverted in a later refactor, so
# both halves are pinned: the default still applies when the argument is absent, and an explicit
# `None` is still refused.
#
# Both halves are binding properties and live here, for all seven classes, permanently. What
# stayed behind in `test_rust_parity.py` was only the claim that the *Python* twin refuses it the
# same way, and it drained one row per deletion until unit 8 took the last of them (`FreeBeam`) and
# the file with it. This is now the only place the `Option<Option<_>>` arms are pinned.

BOUNDARY_CASES = [
    ("StiffString", dict(L=1.0, T=200.0, rho=0.005, fs=48000.0, N=16), "supported"),
    ("DampedStiffString", dict(L=1.0, T=200.0, rho=0.005, fs=48000.0, N=16), "supported"),
    ("TensionModulatedString", dict(L=1.0, T=200.0, rho=0.005, fs=48000.0, N=16), "supported"),
    ("IdealString", dict(L=1.0, T=200.0, rho=0.005, fs=48000.0, N=16), "fixed"),
    ("Bore", dict(L=0.5, fs=48000.0, N=64), ("closed", "open")),
    ("FreeBeam", dict(L=1.0, rho=0.005, fs=48000.0, N=16, kappa=20.0), "free"),
    ("Plate", dict(Lx=0.4, Ly=0.4, kappa=1.0, rho=2.0, fs=20000.0, N=12), "supported"),
]


@pytest.mark.parametrize("name,kwargs,default", BOUNDARY_CASES)
def test_omitting_the_boundary_keeps_the_default(name, kwargs, default):
    assert getattr(physsynth_rs, name)(**kwargs).boundary == default


@pytest.mark.parametrize("name,kwargs,default", BOUNDARY_CASES)
def test_an_explicit_none_boundary_is_refused(name, kwargs, default):
    with pytest.raises(ValueError):
        getattr(physsynth_rs, name)(**kwargs, boundary=None)


def test_the_plates_domain_is_the_second_argument_of_this_shape():
    """``Plate`` takes *two* string-or-omitted arguments, and both need the same treatment.

    Arrived with ``test_rust_parity_plate.py``'s deletion (unit 5, plan §43). The table above
    covers ``boundary`` for six classes; ``domain`` exists only here, and a binding that got
    ``Option<Option<_>>`` right for one of a class's two such arguments and wrong for the other
    would pass every row of that table.
    """
    kw = dict(Lx=0.4, Ly=0.4, kappa=1.0, rho=2.0, fs=20000.0, N=12)
    assert physsynth_rs.Plate(**kw).domain == "rectangle"
    with pytest.raises(ValueError):
        physsynth_rs.Plate(**kw, domain=None)


# -- the plate's interface is narrower than its class, on purpose ---------------------------------


def test_a_branch_only_attribute_is_absent_on_the_other_branch():
    """A free plate has no ``B`` and a supported one has no ``K``/``W``/``w``.

    In the original because they were never assigned; here because the getter raises. A binding
    that offered all of them would be a wider interface than the model, and code that branches on
    ``hasattr`` — ``airbox.py``'s surface tiers do — would take the wrong arm. Moved out of
    ``test_rust_parity_plate.py`` when unit 5 was deleted: the Python half of the comparison is
    gone, the claim about what the extension exposes is not.
    """
    kw = dict(Lx=0.4, Ly=0.4, kappa=1.0, rho=2.0, fs=20000.0, N=12)
    sup = physsynth_rs.Plate(**kw, boundary="supported")
    free = physsynth_rs.Plate(**kw, boundary="free")
    assert hasattr(sup, "B") and hasattr(sup, "L")
    assert not hasattr(sup, "K") and not hasattr(sup, "W") and not hasattr(sup, "w")
    assert hasattr(free, "K") and hasattr(free, "W") and hasattr(free, "w")
    assert not hasattr(free, "B") and not hasattr(free, "L")


def test_the_plate_state_buffers_are_settable_because_airbox_writes_them():
    """§12.2 for the two plates. ``airbox._PlateSurface.commit`` assigns ``_accel``, ``u``,
    ``u_prev`` and ``n``; ``_VKPlateSurface`` adds ``F`` and ``F_prev``. It once wrote three
    iteration diagnostics through this surface too; since
    ``docs/dev/air-box-vk-newton-plan.md`` it drives the model's own kernel and all **five** come
    back in one ``VkStep``, so the setters below are asserted for the state alone. None of that is
    optional — it is how the room puts its load inside the solve, and a ``#[getter]`` with no
    ``#[setter]`` takes the write away *silently* (§33.2)."""
    p = physsynth_rs.Plate(Lx=0.4, Ly=0.4, kappa=1.0, rho=2.0, fs=20000.0, N=12)
    fresh = np.arange(p.n_live, dtype=float)
    p._accel = fresh.copy()
    p.u_prev = p.u
    p.u = fresh.copy()
    p.n = 7
    assert np.array_equal(p.u, fresh)
    assert np.array_equal(p._accel, fresh)
    assert p.n == 7

    mat = dict(E=2.0e11, e=1.0e-3, nu=0.3, rho=7860.0)
    v = physsynth_rs.VKPlate(Lx=0.4, Ly=0.4, fs=48_000.0, N=12, **mat)
    fresh = np.arange(v.n_nodes, dtype=float)
    v.F_prev = v.F
    v.F = fresh.copy()
    v.n_iters = 4
    v.converged = False
    v.last_residual = 1e-5
    v.residual_ratio = 0.5
    assert np.array_equal(v.F, fresh)
    assert v.n_iters == 4 and v.converged is False and v.last_residual == 1e-5
    assert v.residual_ratio == 0.5

    # `couple_outcome` is DERIVED from the four above on every read, which is why it has no setter:
    # a caller who writes `converged` by hand must not be able to leave a stale verdict behind.
    assert v.couple_outcome == "capped"
    v.residual_ratio = float("nan")
    assert v.couple_outcome == "expansive"
    v.converged = True
    assert v.couple_outcome == "converged"


# -- `couple_method`, the coupled step's iteration (plan §5 Part 2) -------------------------------
#
# The flag and the cost counter are the two things Part 2 adds to the binding, and both are here
# for the reason the file header gives: a native bar cannot see a PyO3 signature or a keyword
# argument's default. What the *physics* of Newton is worth is asserted natively, in
# `crates/physsynth-core/tests/plate.rs`.
#
# `n_solves` in particular is a field on two structs — `VkPlate` in the core and `PyVKPlate` here —
# each filled by its own hand-written block (plan §9.5's fork, which Part 2 inherits and feeds one
# more field). Reading it through the binding is what makes the second block exercised at all.


def _vk_struck(method="picard", *, amp=6.0, cap=400, N=12):
    """A 40 cm steel square, struck `amp` thicknesses tall with a 3 cm Gaussian."""
    v = physsynth_rs.VKPlate(
        Lx=0.4, Ly=0.4, E=2.0e11, e=1.0e-3, nu=0.3, rho=7860.0, fs=48_000.0, N=N,
        couple_max_iter=cap, couple_method=method,
    )
    x, y = v.X[v.mask], v.Y[v.mask]
    r2 = (x - 0.2) ** 2 + (y - 0.2) ** 2
    u0 = amp * v.e * np.exp(-r2 / 0.03 ** 2)
    v.set_state(u0, np.zeros_like(u0))
    return v


def test_the_coupling_method_defaults_to_auto_and_echoes_the_spelling_back():
    """Plan §15: the default is ``auto`` — the sweeps first, Newton only where they fail.

    Stated here against the *keyword argument*, which is the half a native bar cannot reach — the
    core's `VkSpec::default()` says one thing and PyO3's signature says another, and it is the
    signature that decides what `physsynth/core/plate.py`'s re-export hands the suite. §6's bar
    used to be that this said ``picard``; what that was protecting is now protected by
    ``auto_is_bit_identical_to_picard_wherever_the_sweeps_converge`` in
    ``crates/physsynth-core/tests/plate.rs``, which is a claim about the numbers rather than about
    a spelling.
    """
    assert physsynth_rs.VKPlate(
        Lx=0.4, Ly=0.4, E=2.0e11, e=1e-3, nu=0.3, rho=7860.0, fs=48_000.0, N=8
    ).couple_method == "auto"
    assert _vk_struck("picard").couple_method == "picard"
    assert _vk_struck("newton").couple_method == "newton"
    assert _vk_struck("auto").couple_method == "auto"

    # An unparseable spelling is refused and quoted back, as `boundary` is — never defaulted.
    with pytest.raises(ValueError, match="couple_method must be 'picard', 'newton' or 'auto'"):
        physsynth_rs.VKPlate(
            Lx=0.4, Ly=0.4, E=2.0e11, e=1e-3, nu=0.3, rho=7860.0, fs=48_000.0, N=8,
            couple_method="gmres",
        )
    with pytest.raises(ValueError, match="couple_method must be 'picard', 'newton' or 'auto'"):
        physsynth_rs.VKPlate(
            Lx=0.4, Ly=0.4, E=2.0e11, e=1e-3, nu=0.3, rho=7860.0, fs=48_000.0, N=8,
            couple_method=None,
        )


def test_auto_spends_nothing_on_newton_until_the_sweeps_fail():
    """``n_fallbacks``, the read-out that separates a cheap step from a rescued one.

    The physics of the fallback is asserted natively; what is here is the half a native bar cannot
    see — that the binding's own hand-written diagnostics block (plan §9.5's fork, third site)
    carries the new field through, on both branches.

    The hard fixture is starved of sweeps rather than driven past the wall, because a cap is
    deterministic and an amplitude is a property of the grid: the wall moves with ``N``, and this
    file's plate is ``N = 12``.
    """
    easy_p, easy_a = _vk_struck("picard", amp=2.0), _vk_struck("auto", amp=2.0)
    easy_p.step()
    easy_a.step()
    assert easy_p.converged, "the easy fixture must converge, or it is testing the other branch"
    assert easy_a.n_fallbacks == 0, "nothing failed, so nothing may be spent on Newton"
    assert np.array_equal(easy_p.u, easy_a.u), "`auto` moved a converging step"
    assert easy_a.n_solves == easy_p.n_solves

    hard_p = _vk_struck("picard", amp=6.0, cap=8)
    hard_n = _vk_struck("newton", amp=6.0, cap=8)
    hard_a = _vk_struck("auto", amp=6.0, cap=8)
    for v in (hard_p, hard_n, hard_a):
        v.step()
    assert not hard_p.converged, "eight sweeps must not be enough, or the rescue never fires"
    assert hard_n.converged, "Newton must reach a root under the same budget"
    assert hard_a.converged and hard_a.n_fallbacks == 1
    assert np.array_equal(hard_a.u, hard_n.u), "the rescue did not re-seed from `2u - u_prev`"
    assert hard_a.n_solves == hard_p.n_solves + hard_n.n_solves


def test_both_coupling_methods_reach_the_same_root_through_the_binding():
    """The gate, end to end: two `VKPlate`s differing only in the iteration, one step each.

    The native bar asserts this against `VkPlate`; this asserts that the *binding's* own step —
    which does not delegate to `VkPlate` at all, but calls `core::vk_step` and fills its own fields
    by hand — dispatches on the flag and fills them from the same result.
    """
    pic, new = _vk_struck("picard"), _vk_struck("newton")
    assert np.array_equal(pic.u, new.u), "the two plates were struck differently"

    pic.step()
    new.step()
    assert pic.converged and new.converged
    gap = np.linalg.norm(new.u - pic.u) / np.linalg.norm(pic.u)
    assert gap < 1e-8, f"the two roots differ by {gap:.3e}"
    assert abs(new.energy() / pic.energy() - 1.0) < 1e-12
    assert pic.couple_outcome == new.couple_outcome == "converged"


def test_the_solve_count_is_the_cost_axis_and_the_binding_reports_it():
    """`n_solves`, the one field Part 2 adds to the step's diagnostics.

    Picard's is exactly two back-substitutions per sweep — one Airy, one theta-scheme — so it can
    be checked against `n_iters` outright. Newton's cannot, which is the whole point: its
    iterations cost a Krylov subspace each, and comparing the two on `n_iters` would show Newton
    winning by twenty when the two are level.
    """
    pic = _vk_struck("picard")
    pic.step()
    assert pic.n_solves == 2 * pic.n_iters

    new = _vk_struck("newton")
    new.step()
    assert new.n_iters < pic.n_iters, "Newton should need far fewer iterations"
    assert new.n_solves > 2 * new.n_iters, "a Newton iteration costs more than a sweep"

    # A plate at rest has spent nothing, and the linear path spends exactly one solve.
    assert _vk_struck("newton").n_solves == 0
    lin = physsynth_rs.VKPlate(
        Lx=0.4, Ly=0.4, E=2.0e11, e=1e-3, nu=0.3, rho=7860.0, fs=48_000.0, N=8, nonlinear=False,
        couple_method="newton",
    )
    lin.step()
    assert lin.n_solves == 1 and lin.n_iters == 1 and lin.converged


# -- `Plate.B` is the one operator a caller may replace (plan §40.5, §43) -------------------------


def _plate(**over):
    kw = dict(Lx=0.4, Ly=0.4, kappa=1.0, rho=2.0, fs=20000.0, N=8)
    kw.update(over)
    return physsynth_rs.Plate(**kw)


def test_the_biharmonic_setter_keeps_the_order_it_is_handed():
    """The whole reason the setter exists: a row stored out of order stays out of order.

    ``tests/test_plate_modal.py`` is the test this was built for and asserts the physics; this
    asserts the *mechanism*, because the failure mode is silent. ``Csr::from_rows`` sorts every
    row it is given, so a setter written the obvious way would accept the pre-2026-08-28 operator,
    canonicalise it, and hand back a plate carrying the shipped ``B`` — and the comparison it
    exists for would compare a plate against a copy of itself and pass.
    """
    p = _plate()
    b = p.B.tocsr()
    scrambled = b.copy()
    for r in range(scrambled.shape[0]):
        lo, hi = scrambled.indptr[r], scrambled.indptr[r + 1]
        scrambled.indices[lo:hi] = scrambled.indices[lo:hi][::-1]
        scrambled.data[lo:hi] = scrambled.data[lo:hi][::-1]
    assert not scrambled.has_sorted_indices, "the fixture is already canonical; it proves nothing"

    p.B = scrambled
    assert p.B is scrambled, "the getter must hand back what was assigned, as the original did"
    # ... and the operator the step applies is the reversed one, not a sorted copy of it: the same
    # numbers summed in the opposite order move the trajectory in its last bits.
    q = _plate()
    ic = np.arange(p.n_live, dtype=float) * 1e-6
    for m in (p, q):
        m.set_state(ic.copy())
    for _ in range(200):
        p.step()
        q.step()
    moved = np.abs(p.state - q.state).max() / np.abs(q.state).max()
    assert 0.0 < moved < 1e-9, (
        f"the injected order moved the trajectory by {moved:.3e} of its amplitude -- zero means "
        "the setter sorted (and the pin in test_plate_modal.py is vacuous); large means it "
        "changed a value"
    )


def test_the_biharmonic_setter_refuses_what_is_not_that_operator():
    """A malformed matrix is a raise, never a panic, and the free branch has no ``B`` to set."""
    p = _plate()
    with pytest.raises(ValueError):
        p.B = sparse.identity(p.n_live + 1, format="csr")
    with pytest.raises(ValueError):
        p.B = np.eye(p.n_live)  # dense: no `indptr`, so there is no stored order to preserve
    free = _plate(boundary="free")
    with pytest.raises(AttributeError):
        free.B = sparse.identity(free.n_live, format="csr")


# -- the theta-scheme default, which nothing compared until unit 8 ------------------------------
#
# Every θ-scheme model takes `theta` with a default, and the default lives in TWO places: the Rust
# constructor's signature and a Python module constant that callers read. Nothing held them
# together. `tests/helpers.py` passed `theta` explicitly on every construction (it was deleted at
# retirement plan §46), so the binding's own default was never exercised by the physics suite at
# all -- it could have drifted to any value and the whole suite would have stayed green.
#
# The Python side is now one constant for the strings and the beam (`string_stiff.THETA_DEFAULT`,
# which `beam.py` re-exports rather than re-declaring as its pre-deletion body did) and a second,
# deliberate one for the plate, which carries its own reasoning in its header. Both are asserted
# here against the value the extension actually applies when the argument is omitted. This is a
# binding property in the same sense as the `Option<Option<_>>` arms above: it is about what PyO3
# fills in, so no native bar and no physics bar can see it.

THETA_DEFAULT_CASES = [
    ("StiffString", dict(L=1.0, T=200.0, rho=0.005, fs=48000.0, N=16), "string_stiff"),
    ("DampedStiffString", dict(L=1.0, T=200.0, rho=0.005, fs=48000.0, N=16), "string_damped"),
    (
        "TensionModulatedString",
        dict(L=1.0, T=200.0, rho=0.005, fs=48000.0, N=16),
        "string_nonlinear",
    ),
    ("FreeBeam", dict(L=1.0, rho=0.005, fs=48000.0, N=16, kappa=20.0), "beam"),
    ("Plate", dict(Lx=0.4, Ly=0.4, kappa=1.0, rho=2.0, fs=20000.0, N=12), "plate"),
]


@pytest.mark.parametrize("name,kwargs,module_name", THETA_DEFAULT_CASES)
def test_the_binding_default_theta_is_the_constant_its_module_publishes(name, kwargs, module_name):
    published = importlib.import_module(f"physsynth.core.{module_name}").THETA_DEFAULT
    omitted = getattr(physsynth_rs, name)(**kwargs)
    supplied = getattr(physsynth_rs, name)(**kwargs, theta=published)
    assert omitted.theta == published, (
        f"{name} built without `theta` uses {omitted.theta!r}, but "
        f"physsynth.core.{module_name}.THETA_DEFAULT is {published!r} -- the extension's default "
        "and the constant its callers read have drifted, and nothing else in the suite would "
        "notice because every helper passes `theta` explicitly"
    )
    assert supplied.theta == omitted.theta


def test_the_strings_and_the_beam_share_one_theta_and_the_plate_deliberately_does_not():
    """`string_stiff` is the family's one source; `plate` is the documented exception.

    Asserted rather than left to the docstrings, because §44.4's finding is that a header is a
    claim and nothing checks it -- and this particular claim was already wrong for two modules
    when it was read (`beam` and `plate` both declared their own copy while `string_stiff.py` said
    they imported one).

    The claim is about the **definition site**, so it is read off the source with `ast` rather than
    by comparing float identity. `0.28` written in two modules happens to be two distinct objects
    in CPython today, but that is an implementation detail of how code objects hold constants and
    not something this test should depend on.
    """
    import ast
    import pathlib

    core = pathlib.Path(physsynth.core.__file__).parent

    def declares_its_own(name):
        tree = ast.parse((core / f"{name}.py").read_text(encoding="utf-8"))
        return any(
            isinstance(node, ast.Assign)
            and any(getattr(t, "id", None) == "THETA_DEFAULT" for t in node.targets)
            for node in tree.body
        )

    assert declares_its_own("string_stiff"), (
        "`string_stiff` no longer defines THETA_DEFAULT -- it is the family's one source and its "
        "header says so"
    )
    for name in ("string_damped", "string_nonlinear", "string_geometric", "beam"):
        module = importlib.import_module(f"physsynth.core.{name}")
        assert not declares_its_own(name), (
            f"`{name}` declares its own THETA_DEFAULT instead of importing `string_stiff`'s. That "
            "is the drift this test exists to forbid, and it is what the beam's shim did until "
            "plan §45.9"
        )
        assert module.THETA_DEFAULT is string_stiff.THETA_DEFAULT

    assert declares_its_own("plate"), (
        "`plate` now imports the string family's constant. That may be an improvement, but its "
        "header claims to own the number and `string_stiff`'s names it as the one exception -- "
        "change all three together"
    )
    assert plate.THETA_DEFAULT == string_stiff.THETA_DEFAULT, (
        f"the plate's independent theta default has drifted from the string family's: "
        f"{plate.THETA_DEFAULT!r} against {string_stiff.THETA_DEFAULT!r}"
    )


# == the four bridges, after unit 10 (plan §49) ==================================================
#
# `tests/test_rust_parity_connection.py` was deleted with `connection.py`'s Python bodies, and it
# held two kinds of test mixed together — the distinction this file's own docstring is built on.
# The **comparisons** (four bit-identical trajectories, two cross-language anchors) went with the
# twin they compared against; the *within-language* half of both anchors survived in
# `tests/test_sympathetic.py` and `tests/test_airbox_vk.py`, and has since gone native with the
# bridges themselves (`crates/physsynth-core/tests/connection.rs`, `connection_plate.rs`;
# retirement plan sections 12 and 20), where one of the two became a single code path.
#
# What moved here is the other kind, and three of them got *stronger* on the way rather than
# weaker: a claim of the form "Rust agrees with Python about this reduction" becomes "Rust agrees
# with **NumPy** about this reduction", which is the thing the transcription was actually copying.

_L_STRING = 1.0
_T_STRING = 200.0
_RHO_STRING = 0.005
_N_STRING = 48
_LAM_STRING = 0.9
_BODY_FREQS = np.array([137.0, 213.0, 330.0, 471.0])
_N_PLATE = 8


def _conn_fs(T=_T_STRING):
    return np.sqrt(T / _RHO_STRING) * _N_STRING / (_L_STRING * _LAM_STRING)


def _conn_string(*, T=_T_STRING, N=_N_STRING, fs=None, pluck=1e-3, boundary=("fixed", "free")):
    s = physsynth_rs.IdealString(
        L=_L_STRING, T=T, rho=_RHO_STRING, fs=fs or _conn_fs(), N=N, boundary=boundary
    )
    if pluck:
        s.set_state(triangular_pluck(s.x, s.L, 0.3 * s.L, amplitude=pluck))
    return s


def _conn_body(*, fs=None, phi=1.0, masses=0.02, freqs=_BODY_FREQS):
    return physsynth_rs.ModalBody(
        freqs=freqs, fs=fs or _conn_fs(), sigmas=0.0, masses=masses, phi=phi
    )


def _conn_plate(boundary="supported", *, fs=None):
    return physsynth_rs.Plate(
        Lx=1.0, Ly=1.0, kappa=20.0, rho=2.0, fs=fs or _conn_fs(), N=_N_PLATE, boundary=boundary
    )


def _conn_vk_plate(boundary="supported", *, nonlinear=True, fs=None):
    return physsynth_rs.VKPlate(
        Lx=0.4, Ly=0.4, E=2.0e11, e=1.0e-4, nu=0.3, rho=7800.0,
        fs=fs or _conn_fs(), N=_N_PLATE, boundary=boundary, nonlinear=nonlinear,
    )


def _modal_bridge(*, phi=1.0, masses=0.02, freqs=_BODY_FREQS, K=8000.0):
    return physsynth_rs.StringBodyBridge(
        string=_conn_string(), body=_conn_body(phi=phi, masses=masses, freqs=freqs), K=K
    )


def _sympathetic(J=3, *, K=None, seed=20260831):
    r = np.random.default_rng(seed)
    fs = _conn_fs()
    strings = [_conn_string(T=_T_STRING * (1.0 + 0.03 * j), fs=fs, pluck=0.0) for j in range(J)]
    for j, s in enumerate(strings):
        s.set_state(triangular_pluck(s.x, s.L, 0.3 * s.L, amplitude=1e-3 * (1.0 + 0.1 * j)))
    if K is None:
        K = list(r.uniform(300.0, 900.0, J))
    return physsynth_rs.SympatheticStrings(strings=strings, body=_conn_body(fs=fs), Ks=K)


def _plate_bridge(boundary="supported", *, K=3000.0):
    return physsynth_rs.StringPlateBridge(
        string=_conn_string(), plate=_conn_plate(boundary), K=K
    )


def _vk_bridge(boundary="supported", *, nonlinear=True, K=3000.0):
    return physsynth_rs.StringVKPlateBridge(
        string=_conn_string(pluck=3e-3),
        plate=_conn_vk_plate(boundary, nonlinear=nonlinear),
        K=K,
    )


def _left_to_right(values):
    total = 0.0
    for v in values:
        total += float(v)
    return total


# -- the three SciPy names the binding reads off `connection`'s namespace ------------------------


def test_the_scipy_names_the_binding_reads_are_still_scipys():
    """``connection.py`` is a shim with no Python caller for ``sparse``, ``spsolve`` or ``splu``.

    They are there because ``crates/physsynth-py/src/connection.rs`` does
    ``py.import("physsynth.core.connection")`` and reads all three **by name at call time** — the
    faithful transcription of a reference that called its own module globals, and what makes
    §24.4's shared-factorization manoeuvre available on this file.

    Nothing else can say this. ``test_stability.py``'s ``deleted_bodies`` loop asserts
    ``module.X is physsynth_rs.X`` for every name a deletion leaves behind, and these three are
    not Rust objects, so they fall straight through it. They also carry ``noqa: F401``, which is
    the only thing standing between them and ``ruff check --fix`` — and their disappearance would
    surface as an ``AttributeError`` from inside the extension, three layers from the cause.
    """
    assert connection.sparse is scipy.sparse
    assert connection.splu is scipy.sparse.linalg.splu
    assert connection.spsolve is scipy.sparse.linalg.spsolve


def test_the_guard_reads_its_solver_from_the_module_at_call_time(monkeypatch):
    """And that the binding actually *goes* through them, which the identity check cannot say.

    Replace the names and both the factorization and the sparse solve must change with them. This
    is the half of the pair that would catch a future port capturing the binding at import.
    """
    calls = {"splu": 0, "spsolve": 0}
    real_splu, real_spsolve = connection.splu, connection.spsolve

    def counting_splu(a, *args, **kw):
        calls["splu"] += 1
        return real_splu(a, *args, **kw)

    def counting_spsolve(a, b, *args, **kw):
        calls["spsolve"] += 1
        return real_spsolve(a, b, *args, **kw)

    monkeypatch.setattr(connection, "splu", counting_splu)
    monkeypatch.setattr(connection, "spsolve", counting_spsolve)
    bridge = _plate_bridge()
    assert calls == {"splu": 1, "spsolve": 1}, (
        "the Rust guard did not go through `connection`'s own module globals -- a captured "
        "binding, and §24.4's manoeuvre is unavailable on this file"
    )
    assert bridge.stability_margin > 0.0


# -- the write through the string's buffer -------------------------------------------------------


def test_the_bridge_reaction_is_written_through_the_live_buffer():
    """``string.u[-1] -= beta_s * F`` must mutate the array, not rebind it.

    ``lib.rs``'s module docstring names that one line as the reason the string's buffers are
    Python-owned. A port that read ``u`` into a vector, subtracted, and assigned the result back
    through a setter would produce exactly the same numbers *for itself* while a caller holding a
    reference to ``u`` saw the un-reacted field — invisible to every physics bar, and fatal to the
    viewer, which holds exactly such references.
    """
    bridge = _modal_bridge()
    for _ in range(200):  # let the wave reach the terminus so F is nonzero
        bridge.step()
    f = bridge.connection_force()
    assert abs(f) > 0.0, "no bridge force to test against"

    bridge.string.step()  # step the string by hand, then hold the array it produced
    held = bridge.string.u
    before = float(held[-1])
    bridge.string.u[-1] -= bridge.beta_s * f
    assert float(held[-1]) == before - bridge.beta_s * f, (
        "writing u[-1] did not reach the array object a caller holds"
    )


def test_the_steps_reaction_reaches_a_held_reference():
    """The same property through ``step`` itself, which is where it actually matters.

    The string rebinds ``u`` inside its own step, so an array held across one is the PREVIOUS
    field — and it must be the object ``u_prev`` now names, reaction included. The comparison
    against a Python twin that used to stand here is gone; what is asserted instead is the
    identity that made the comparison meaningful, and it is a sharper statement than the equality
    was.
    """
    bridge = _modal_bridge()
    for _ in range(200):
        bridge.step()
    held = bridge.string.u
    before = np.array(held, copy=True)
    bridge.step()
    assert np.array_equal(np.asarray(bridge.string.u_prev), np.asarray(held))
    assert not np.array_equal(np.asarray(bridge.string.u), np.asarray(held))
    assert np.array_equal(np.asarray(held), before), (
        "the held array was written to by the NEXT step -- it is not a snapshot"
    )


# -- which reductions were transcribed, and which were not ---------------------------------------


def test_the_modal_inverse_mass_sum_is_numpys_pairwise_blocking():
    """``beta_b = k^2 sum_i phi_i^2 / m_i`` is ``reduce::sum``, and no shipped fixture can tell.

    §30.2's cutoff makes this decidable by counting: below **eight** terms ``np.sum`` IS a
    left-to-right loop whatever the values, and every body in this repo has four or five modes. So
    the transcription is unexercised by the whole suite, and the fixture is *searched* rather than
    picked (§26.6) — a hand-picked one lands in the agreeing majority and the test goes green
    having compared two spellings that happen to coincide.

    Unit 10 made this **stronger** rather than weaker. It used to read ``py.beta_b == rs.beta_b``,
    which asserts the two implementations agree; it now reads ``rs.beta_b == np.sum(...)``, which
    asserts the thing the transcription was copying. The Python reference could have been wrong
    about NumPy's blocking and this test would have passed.
    """
    rng = np.random.default_rng(20260831)
    for m_modes, expect_witness in ((5, False), (12, True)):
        witness = None
        agreed = 0
        for _ in range(4000):
            phi = rng.uniform(0.4, 1.8, m_modes)
            masses = rng.uniform(0.005, 0.05, m_modes)
            terms = phi * phi / masses
            if _left_to_right(terms) == float(np.sum(terms)):
                agreed += 1
            elif witness is None:
                witness = (phi.copy(), masses.copy())
        if not expect_witness:
            assert agreed == 4000, (
                f"M={m_modes} is below NumPy's pairwise cutoff, so np.sum must BE the "
                f"left-to-right loop -- it differed in {4000 - agreed} draws"
            )
            continue

        assert witness is not None, (
            f"no witness at M={m_modes} -- this test would then assert nothing (§23.5)"
        )
        phi, masses = witness
        freqs = np.linspace(110.0, 900.0, m_modes)
        rs = _modal_bridge(phi=phi, masses=masses, freqs=freqs, K=2000.0)
        terms = phi * phi / masses
        assert rs.beta_b == rs.k * rs.k * float(np.sum(terms)), (
            "the Rust sum is not NumPy's pairwise blocking"
        )
        assert rs.beta_b != rs.k * rs.k * _left_to_right(terms), (
            "the witness search found a fixture the two spellings cannot be told apart on"
        )
        print(f"beta_b at the witness: {rs.beta_b!r}")


def test_the_shared_bridge_force_sum_is_numpys_pairwise_blocking():
    """``body.step(force=sum_j F_j)`` reaches the next timestep, so this sum must be exact.

    Same blind spot as the test above, from the other end: every sympathetic rig in the repo has
    two or three strings, and the cutoff is eight. Reconstructed without a Python twin by driving
    two **bare** bodies alongside the coupler with the two spellings of the same sum — the one
    that tracks the coupler's body says which spelling is inside it, and the one that does not is
    the negative control that keeps the first from being vacuous.
    """
    j_strings = 8
    symp = _sympathetic(j_strings)
    fs = _conn_fs()
    pairwise = _conn_body(fs=fs)
    naive = _conn_body(fs=fs)
    differing = 0
    for _ in range(600):
        forces = np.asarray(symp.connection_forces())
        total_np = float(np.sum(forces))
        total_ltr = _left_to_right(forces)
        differing += total_np != total_ltr
        symp.step()
        pairwise.step(force=total_np)
        naive.step(force=total_ltr)
    assert differing > 0, (
        f"J={j_strings} never put np.sum and a left-to-right loop apart -- this test would then "
        "pass on a port that transcribed the sum wrongly"
    )
    print(f"sum(forces) differs from a left-to-right loop on {differing}/600 steps at J=8")
    assert np.array_equal(np.asarray(symp.body.q), np.asarray(pairwise.q)), (
        "the coupler's body did not follow `np.sum` of the per-string forces"
    )
    assert not np.array_equal(np.asarray(symp.body.q), np.asarray(naive.q)), (
        "the negative control tracked too -- the two spellings never reached the body apart, so "
        "the assertion above is vacuous on this fixture"
    )


def test_the_two_dot_products_are_not_transcribed(monkeypatch):
    """``np.dot(phi, q)`` stays a BLAS call -- ``ddot`` fuses its multiply-add (§14.2).

    The old test asserted this by *comparison*: Python and Rust agreed to the bit on a fixture
    whose weights were deliberately not powers of two, and separately reported whether this
    machine's OpenBLAS ``ddot`` kernel could be told apart from a left-to-right multiply-add at
    all (it could on Windows, it could not on GitHub's EPYC -- which is why the separation was
    reported and never required).

    With no twin the claim is made directly, and it is sharper than the equality was: the binding
    looks ``dot`` up on the ``numpy`` module at call time, so replacing the name COUNTS the calls.
    Both sites are pinned, and they are pinned at different rates --

    * ``energy`` -> exactly one, the previous step's ``phi . q_prev``;
    * ``__init__`` -> exactly ``N + M``, because the exact stability guard builds the coupled
      leapfrog operator column by column and every column's ``eta`` needs the same product.

    A transcription of either into Rust drops its count to zero, which is the failure §14.2 says
    must not happen. Adding one shows up as a count that is no longer the operator's dimension.
    """
    rng = np.random.default_rng(4711)
    bridge = _modal_bridge(phi=rng.uniform(0.4, 1.8, len(_BODY_FREQS)))
    n_modes = len(_BODY_FREQS)
    calls = []
    real_dot = np.dot

    def counting_dot(a, b, *args, **kw):
        calls.append((np.asarray(a).shape, np.asarray(b).shape))
        return real_dot(a, b, *args, **kw)

    monkeypatch.setattr(np, "dot", counting_dot)

    bridge.step()
    assert calls == [], (
        f"`step` reached `np.dot` {len(calls)} times -- the reference's step takes the body's "
        "own `bridge_displacement`, so a dot product here is a different scheme"
    )

    bridge.energy()
    assert calls == [((n_modes,), (n_modes,))], (
        f"`energy` made {len(calls)} `np.dot` calls, not the one the reference makes -- either "
        "the product was transcribed into Rust (it must not be: `ddot` fuses its multiply-add "
        "and admits no scalar recipe, §14.2) or another was added"
    )

    calls.clear()
    rebuilt = _modal_bridge(phi=rng.uniform(0.4, 1.8, n_modes))
    n_dof = rebuilt.string.N + rebuilt.body.M
    assert len(calls) == n_dof, (
        f"building the bridge made {len(calls)} `np.dot` calls, not the {n_dof} the exact "
        "stability guard needs (one per column of the coupled leapfrog operator)"
    )
    assert all(shapes == ((n_modes,), (n_modes,)) for shapes in calls)


# -- the duck typing -----------------------------------------------------------------------------


class _NotAModalBody:
    """A stand-in that is not a ``ModalBody`` and does not inherit from one.

    The reference contains no ``isinstance``, ``hasattr``, ``getattr`` or ``type(`` at all
    (§31.11): the ``body=`` slot takes four kinds of object and the ``plate=`` slot eight. A
    ``#[pyclass]`` that downcast its collaborators would still pass the whole airbox and radiation
    family, because those hand it real models.
    """

    def __init__(self, inner):
        self._inner = inner
        self.steps = 0

    @property
    def M(self):
        return self._inner.M

    @property
    def phi(self):
        return self._inner.phi

    @property
    def m(self):
        return self._inner.m

    @property
    def omega(self):
        return self._inner.omega

    @property
    def k(self):
        return self._inner.k

    @property
    def q(self):
        return self._inner.q

    @property
    def q_prev(self):
        return self._inner.q_prev

    def bridge_displacement(self):
        return self._inner.bridge_displacement()

    def step(self, force=0.0):
        self.steps += 1
        self._inner.step(force=force)

    def energy(self):
        return self._inner.energy()

    def pressure(self):
        return self._inner.pressure()


def test_the_bridge_never_looks_at_its_collaborators_type():
    rs = physsynth_rs.StringBodyBridge(
        string=_conn_string(), body=_NotAModalBody(_conn_body()), K=8000.0
    )
    for _ in range(300):
        rs.step()
    assert rs.body.steps == 300, "the port did not drive the stand-in through its own `step`"
    assert rs.energy() != 0.0 and np.isfinite(rs.energy())
    assert np.isfinite(rs.pressure())


# -- §33.2's write question, aimed at the class being ported --------------------------------------


ASSIGNED_BY_THE_REFERENCE = {
    "StringBodyBridge": ("string", "body", "K", "k", "beta_s", "beta_b", "cfl_2dof",
                         "spectral_radius", "n"),
    "StringPlateBridge": ("string", "plate", "K", "k", "drive_index", "beta_s", "_f_ext",
                          "stability_margin", "n"),
    "StringVKPlateBridge": ("string", "plate", "K", "k", "drive_index", "beta_s", "_f_ext",
                            "stability_margin", "n"),
    "SympatheticStrings": ("strings", "body", "K", "k", "J", "beta_s", "_offsets",
                           "spectral_radius", "n"),
}


@pytest.mark.parametrize("name", sorted(ASSIGNED_BY_THE_REFERENCE))
def test_every_attribute_the_reference_assigned_stays_writable(name):
    """A ``#[getter]`` with no ``#[setter]`` is a data descriptor whose ``__set__`` raises (§33.2).

    So porting a class decides not only which names can be read through it but which can be
    *written*, and the default is none — where the reference, being Python, allowed every one. The
    list is the reference's own ``self.X = ...`` set, read off the source before it was deleted.
    """
    built = {
        "StringBodyBridge": _modal_bridge,
        "StringPlateBridge": _plate_bridge,
        "StringVKPlateBridge": _vk_bridge,
        "SympatheticStrings": _sympathetic,
    }[name]()
    for attr in ASSIGNED_BY_THE_REFERENCE[name]:
        value = getattr(built, attr)
        setattr(built, attr, value)  # must not raise
        assert getattr(built, attr) is value or np.all(getattr(built, attr) == value)
    built.a_name_the_reference_never_used = 17  # the instance dict, as on a Python class
    assert built.a_name_the_reference_never_used == 17


def test_a_replaced_force_vector_reaches_the_plate():
    """``_f_ext`` is writable, and writing it must change what the plate is handed.

    The hazard is §32.2's, one tier up and inside this port: the keyword dict ``plate.step`` is
    called with is reused across steps for speed, so a cached *array* in it would ignore this
    assignment and the test would pass having driven the original vector.
    """
    rs = _plate_bridge()
    for _ in range(150):
        rs.step()
    seen = []

    class _Recorder:
        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, item):
            return getattr(self._inner, item)

        def step(self, f_ext=None):
            # The OBJECT, not a copy: what is being asserted is which array the plate was handed,
            # and its contents are the force the bridge has just written into it.
            seen.append(f_ext)
            self._inner.step(f_ext=f_ext)

    original = rs._f_ext
    rs.plate = _Recorder(rs.plate)
    replacement = np.zeros(rs.plate.n_live)
    rs._f_ext = replacement
    rs.step()
    assert seen and seen[-1] is replacement, (
        "the plate was handed a different array than `_f_ext` -- the keyword dict cached one"
    )
    assert seen[-1][rs.drive_index] == 0.0, "the bridge did not zero the drive node after the step"
    assert not np.any(np.asarray(original)), "the replaced vector was written to anyway"


# -- the refusals and the signatures --------------------------------------------------------------


def test_the_bridge_constructors_are_keyword_only():
    for cls, args in (
        (physsynth_rs.StringBodyBridge, (_conn_string(), _conn_body(), 1.0)),
        (physsynth_rs.StringPlateBridge, (_conn_string(), _conn_plate(), 1.0)),
        (physsynth_rs.StringVKPlateBridge, (_conn_string(), _conn_vk_plate(), 1.0)),
        (physsynth_rs.SympatheticStrings, ([_conn_string()], _conn_body(), [1.0])),
    ):
        with pytest.raises(TypeError):
            cls(*args)


# The reference's refusal messages, FROZEN — recorded verbatim from the Python bodies immediately
# before they were deleted, the way `tests/analysis_frozen_values.py` holds the analysis oracles'
# numbers (plan §38). The old tests raised the same failure through both implementations and
# compared the two strings; with one implementation left there is nothing to compare against
# except what the reference actually said, so that is written down. The numbers inside them are
# part of the message and are kept deliberately: a refusal that stops reporting *which* bound was
# exceeded is a worse refusal, and nothing else in the suite would notice.
#
# TWO of the eleven interpolate a number this suite may NOT freeze, and `{n}` marks them. The old
# test compared the digits across languages *on one machine* and matched only a substring of the
# prose; writing the digits down here would turn that into a claim about the CPU -- finding #14
# through a new door. `k^2 * lambda_max(A)` comes out of LAPACK's `dgeev` on a 52x52 dense matrix,
# and the margin out of `spsolve` PLUS `splu(...).solve(...)`, whose fill-reducing ordering this
# project has already measured to be a claim about how SciPy was built (Phase 4, Phase 5 b5). So
# the prose around them is frozen exactly and the number is required only to be *there* and to be a
# finite positive float -- which is the part that was worth asserting. A refusal that stops naming
# its bound stops being useful; a refusal that names it to eleven digits of somebody else's LAPACK
# is not a bound, it is a fingerprint of the runner.
#
# The nine literals are safe by inspection: `k=9.375e-05 vs 8.523e-05` is one IEEE division printed
# to four significant figures, `1000000` and `[0, 49)` are integers, and the rest is prose.
FROZEN_REFUSALS = [
    (
        "StringBodyBridge",
        lambda: physsynth_rs.StringBodyBridge(
            string=_conn_string(), body=_conn_body(fs=_conn_fs() * 1.1), K=1.0
        ),
        "string and body must share a timestep (got k=9.375e-05 vs 8.523e-05); build them at "
        "the same fs.",
    ),
    (
        "StringBodyBridge",
        lambda: physsynth_rs.StringBodyBridge(
            string=_conn_string(boundary=("fixed", "fixed"), pluck=0.0),
            body=_conn_body(),
            K=1.0,
        ),
        "the string's right end must be 'free' to attach a body bridge (build it with "
        "boundary=('fixed', 'free')).",
    ),
    (
        "StringBodyBridge",
        lambda: physsynth_rs.StringBodyBridge(
            string=_conn_string(), body=_conn_body(), K=-1.0
        ),
        "bridge stiffness K must be >= 0.",
    ),
    (
        "StringBodyBridge",
        lambda: physsynth_rs.StringBodyBridge(
            string=_conn_string(), body=_conn_body(), K=1e7
        ),
        "connection unstable: k^2 * lambda_max(A) = {n} >= 4. Reduce K, raise fs, or "
        "increase the body/string end mass.",
    ),
    (
        "StringPlateBridge",
        lambda: physsynth_rs.StringPlateBridge(
            string=_conn_string(), plate=_conn_plate(fs=_conn_fs() * 1.1), K=1.0
        ),
        "string and plate must share a timestep (got k=9.375e-05 vs 8.523e-05); build them at "
        "the same fs.",
    ),
    (
        "StringPlateBridge",
        lambda: physsynth_rs.StringPlateBridge(
            string=_conn_string(), plate=_conn_plate(), K=1.0, drive_index=10**6
        ),
        "drive_index 1000000 out of range [0, 49).",
    ),
    (
        "StringPlateBridge",
        lambda: physsynth_rs.StringPlateBridge(
            string=_conn_string(), plate=_conn_plate(), K=1e9
        ),
        "connection unstable: stability margin = {n} >= 1. Reduce K, raise fs, or "
        "increase the string/plate node mass.",
    ),
    (
        "SympatheticStrings",
        lambda: physsynth_rs.SympatheticStrings(strings=[], body=_conn_body(), Ks=[]),
        "need at least one string.",
    ),
    (
        "SympatheticStrings",
        lambda: physsynth_rs.SympatheticStrings(
            strings=[_conn_string(), _conn_string()], body=_conn_body(), Ks=[1.0]
        ),
        "Ks must have one stiffness per string (got (1,) for 2 strings).",
    ),
    (
        "SympatheticStrings",
        lambda: physsynth_rs.SympatheticStrings(
            strings=[_conn_string()], body=_conn_body(), Ks=[-1.0]
        ),
        "every bridge stiffness K must be >= 0.",
    ),
    (
        "SympatheticStrings",
        lambda: physsynth_rs.SympatheticStrings(
            strings=[_conn_string()], body=_conn_body(fs=_conn_fs() * 1.1), Ks=[1.0]
        ),
        "string 0 and the body must share a timestep (got k=9.375e-05 vs 8.523e-05); build them "
        "at the same fs.",
    ),
]


@pytest.mark.parametrize(
    "cls_name, build, message",
    FROZEN_REFUSALS,
    ids=[f"{n}-{m.split()[0]}-{i}" for i, (n, _b, m) in enumerate(FROZEN_REFUSALS)],
)
def test_the_bridge_refusals_reproduce_the_reference_verbatim(cls_name, build, message):
    with pytest.raises(ValueError) as err:
        build()
    text = str(err.value)
    if "{n}" not in message:
        assert text == message, (
            f"{cls_name}'s refusal text drifted from the reference's:\n  now: {text}\n  was: "
            f"{message}"
        )
        return

    # The prose is frozen; the bound is only required to still be reported, and to be a number.
    prefix, suffix = message.split("{n}")
    assert text.startswith(prefix) and text.endswith(suffix), (
        f"{cls_name}'s refusal PROSE drifted from the reference's:\n  now: {text}\n  was: "
        f"{message}"
    )
    reported = text[len(prefix) : len(text) - len(suffix)]
    value = float(reported)  # raises if the bound stopped being reported as a number at all
    assert np.isfinite(value) and value > 0.0, (
        f"{cls_name} reported {reported!r} as the bound it refused on"
    )


def test_an_explicit_none_drive_index_is_the_omitted_one():
    """§24.7 and §31.7: with ``Option<Option<_>>``, ``Some(None)`` is the omitted keyword and a
    bare ``None`` is the caller's literal — and getting the arms backwards is silent, because here
    the reference means the same thing by both."""
    omitted = physsynth_rs.StringPlateBridge(
        string=_conn_string(), plate=_conn_plate(), K=3000.0
    )
    explicit = physsynth_rs.StringPlateBridge(
        string=_conn_string(), plate=_conn_plate(), K=3000.0, drive_index=None
    )
    assert omitted.drive_index == explicit.drive_index
    assert omitted.stability_margin == explicit.stability_margin


# -- the 1-D operators: what `test_rust_parity_operators.py` left (plan §21) ----------------------
#
# That file compared each operator with its Python transcription, and phase A deleted the
# transcription. Twelve of its fourteen tests were comparisons with nothing left on the far side;
# the correctness they guarded is asserted natively in `crates/physsynth-core/tests/ops.rs`
# against what each operator is supposed to BE (exact on polynomials, exact discrete eigenpairs,
# `B = D2 D2`, the free beam's rigid-body nullspace). The ones below had a referent that is not
# being deleted -- NumPy's slicing, NumPy's `dot`, SciPy's sparse product, or the binding itself --
# and two of them are sharper for being re-aimed at it: "Rust agrees with a Python transcription of
# `D2 @ D2`" becomes "Rust agrees with SciPy's own `D2 @ D2`".

OPS_SIZES = [2, 3, 4, 5, 8, 16, 33, 64]


def _ops_fields(n_nodes, seed=20260826):
    """A smooth field and a random one: a sign slip can cancel in the first, not the second."""
    x = np.linspace(0.0, 1.0, n_nodes)
    rng = np.random.default_rng(seed)
    return [np.sin(3.0 * np.pi * x) + 0.4 * x * x, rng.standard_normal(n_nodes)]


def _ops_rebuild(triplets):
    data, indices, indptr, shape = triplets
    return sparse.csr_matrix((data, indices, indptr), shape=shape)


def test_the_two_first_differences_are_the_same_function():
    # `delta_x_backward` exists for notational symmetry in the energy proofs. If the two ever
    # differed, every proof that swaps one for the other would stop being about the scheme.
    u = _ops_fields(12)[0]
    forward, backward = physsynth_rs.delta_x_forward, physsynth_rs.delta_x_backward
    assert np.array_equal(forward(u, 0.1), backward(u, 0.1))


def test_a_too_short_field_yields_numpys_empty_slice_rather_than_a_panic():
    # The Rust kernels document a precondition and would panic; a panic at the interpreter boundary
    # is a PanicException, so the binding guards the length. What it must return is what NumPy's
    # slicing returns, which is the original's behaviour and is written out here, not imported.
    for name, need in [
        ("delta_x_forward", 2),
        ("delta_x_backward", 2),
        ("delta_xx", 3),
        ("delta_xxxx", 5),
    ]:
        for n_nodes in range(need):
            u = np.zeros(n_nodes)
            assert getattr(physsynth_rs, name)(u, 0.5).shape == u[need - 1 :].shape == (0,)


@pytest.mark.parametrize("n_nodes", [4, 17, 129, 1025])
def test_the_inner_product_agrees_with_numpys_dot_to_the_group_a_target(n_nodes):
    # `h * np.dot` goes through BLAS, which accumulates in an order no portable loop reproduces,
    # so this is a tolerance (the plan's Group A 1e-13), not a bit. The comparand was always NumPy;
    # the deleted transcription was one line wrapped around it.
    h = 1.0 / (n_nodes - 1)
    f, g = _ops_fields(n_nodes)
    for a, b in ((f, g), (g, f), (f, f)):
        rs, ref = physsynth_rs.inner(a, b, h), float(h * np.dot(a, b))
        assert abs(rs - ref) <= 1e-13 * max(abs(rs), abs(ref), 1e-300)
    rs, ref = physsynth_rs.norm2(f, h), float(h * np.dot(f, f))
    assert abs(rs - ref) <= 1e-13 * max(abs(rs), abs(ref))
    assert rs >= 0.0


def test_inner_is_exactly_norm2_when_the_operands_coincide():
    # Not a tautology across the boundary: `norm2` is a separate binding entry point, and the two
    # would drift apart if it ever grew its own summation.
    f = _ops_fields(65)[0]
    assert physsynth_rs.inner(f, f, 0.01) == physsynth_rs.norm2(f, 0.01)


@pytest.mark.parametrize("n", OPS_SIZES)
@pytest.mark.parametrize("length", [1.0, 0.65])
def test_the_biharmonic_is_scipys_own_product_of_the_second_difference(n, length):
    """The one that paid for the whole parity file, re-aimed at the library it was copying.

    ``B = D2 @ D2``, so its entries are genuine three-term sums and its boundary-adjacent diagonal
    (``5/h^4``, not ``6/h^4``) is produced by the product rather than written down. The deleted
    Python body was that product in SciPy; here SciPy computes it from the binding's own ``D2`` and
    the binding's ``B`` must match it value for value. Both a tidy ``h = 1/N`` and a 0.65 m string,
    where ``1/(h*h)`` rounds differently. Stored order is normalised away: SciPy's SMMP kernel
    returns unsorted columns, the crate returns canonical ones, and both are the same matrix.
    """
    h = length / n
    d2 = _ops_rebuild(physsynth_rs.second_difference_matrix_csr(n, h))
    rs = _ops_rebuild(physsynth_rs.biharmonic_matrix_csr(n, h))
    sp = (d2 @ d2).tocsr()
    for m in (rs, sp):
        m.sum_duplicates()
        m.sort_indices()
    assert rs.shape == sp.shape and rs.nnz == sp.nnz
    assert np.array_equal(rs.indptr, sp.indptr) and np.array_equal(rs.indices, sp.indices)
    assert np.array_equal(rs.data, sp.data), (
        f"B(N={n}, L={length}) is not SciPy's D2 @ D2; worst |delta| = "
        f"{np.abs(rs.data - sp.data).max():.3e}"
    )


@pytest.mark.parametrize("n", [-1, 0, 1])
def test_a_grid_too_coarse_to_have_an_interior_is_refused(n):
    for name in ("second_difference_matrix", "biharmonic_matrix", "free_beam_stiffness"):
        with pytest.raises(ValueError):
            getattr(physsynth_rs, f"{name}_csr")(n, 0.5)


def test_the_binding_hands_back_triplets_not_a_matrix():
    # `physsynth-core` must not know what SciPy is, so the binding returns `(data, indices, indptr,
    # shape)` and `operators.py` rebuilds. If that ever became an object, the shim would be a no-op.
    out = physsynth_rs.second_difference_matrix_csr(8, 0.125)
    assert isinstance(out, tuple) and len(out) == 4
    data, indices, indptr, shape = out
    assert data.dtype == np.float64
    assert indices.dtype == np.int32 and indptr.dtype == np.int32
    assert shape == (7, 7)
    assert indptr[-1] == len(data) == len(indices)


def test_the_operator_shim_takes_what_numpy_would():
    # The reason `operators` wraps rather than re-exports: the binding refuses a strided view or an
    # integer array, NumPy's slicing did not, and callers were written against NumPy.
    from physsynth.core import operators

    u = np.arange(21) ** 2  # int64, and exact on a quadratic
    for field in (u, u.astype(float)[::2], np.asfortranarray(u.astype(float))):
        out = operators.delta_xx(field, 1.0)
        expected = np.asarray(field, dtype=float)
        assert np.array_equal(out, expected[2:] - 2.0 * expected[1:-1] + expected[:-2])


# -- the horizon shim: what `test_resolution_horizon.py` left (retirement plan §46) ---------------
#
# That file's 48 tests went native (`crates/physsynth-analysis/tests/horizon.rs` and
# `crates/physsynth-core/tests/horizon_models.rs`) except these three, which are about
# `physsynth/analysis/horizon.py` itself: the Rust takes flat slices, so the shape logic -- ravel on
# the way in, reshape on the way out -- and the `ValueError` a caller sees exist only in the shim.
# `test_analysis_frozen.py` still drives that shim with 1-D families, so without the mesh test the
# reshape is written and never run until the shim goes with the rest of `physsynth/`.


def test_the_horizon_error_keeps_the_shape_it_was_given_including_a_mesh():
    from physsynth.analysis.horizon import pitch_error_cents

    ratio = np.array([[1.0, 1.001], [0.999, 1.0]])
    got = pitch_error_cents(100.0 * ratio, np.full((2, 2), 100.0))
    assert got.shape == (2, 2), f"a 2x2 mesh came back as {got.shape}"
    flat = pitch_error_cents(100.0 * ratio.ravel(), np.full(4, 100.0))
    assert np.array_equal(got.ravel(), flat), "the mesh answer is not the flat answer, reshaped"

    scalar = pitch_error_cents(np.float64(110.0), np.float64(100.0))
    assert scalar.shape == (), f"a pair of scalars came back as {scalar.shape}"


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_the_horizon_shim_refuses_a_non_positive_bound(bad):
    from physsynth.analysis.horizon import pitch_horizon

    with pytest.raises(ValueError):
        pitch_horizon([100.0], [100.0], bad)


def test_the_horizon_shim_refuses_mismatched_families_rather_than_broadcasting():
    from physsynth.analysis.horizon import pitch_horizon

    with pytest.raises(ValueError):
        pitch_horizon([100.0, 200.0], [100.0], 5.0)
