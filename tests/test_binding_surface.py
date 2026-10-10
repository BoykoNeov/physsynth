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
import pkgutil

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
# Two of those three -- the inverse modal mass sum and the shared bridge force, both `np.sum` --
# went native at retirement plan §48, against NumPy's own sums recorded first
# (`crates/physsynth-core/tests/reductions.rs`, and one bar each in `connection_body.rs` and
# `connection.rs`). The third, the `np.dot` count, is about the binding and stays.

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


# -- the three SciPy names the binding reads off `connection`'s namespace ------------------------


def test_the_scipy_names_the_binding_reads_are_still_scipys():
    """``connection.py`` is a shim with no Python caller for ``sparse``, ``spsolve`` or ``splu``.

    They are there because ``crates/physsynth-py/src/connection.rs`` does
    ``py.import("physsynth.core.connection")`` and reads all three **by name at call time** — the
    faithful transcription of a reference that called its own module globals, and what makes
    §24.4's shared-factorization manoeuvre available on this file.

    Nothing else can say this. The ``deleted_bodies`` loop at the end of this file asserts
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
# before they were deleted (and held against the CORE's own messages too since retirement plan
# §48, in `crates/physsynth-core/tests/connection*.rs`: ten verbatim, and the `Ks` count, which
# the reference printed as a NumPy shape, in its native wording). These pin the BINDING's copies.
# They were held the way the analysis oracles' numbers were (plan §38; that record is
# `crates/physsynth-analysis/tests/reference/analysis_frozen.json` since retirement plan §47).
# The old tests raised the same failure through both implementations and compared the two
# strings; with one implementation left there is nothing to compare against except what the
# reference actually said, so that is written down. The numbers inside them are
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
# being deleted -- NumPy's slicing, NumPy's `dot`, SciPy's sparse product, or the binding itself.
# The two whose referent was NumPy's `dot` and SciPy's `D2 @ D2` went native at retirement plan
# §48, with the library's answers recorded first (`crates/physsynth-core/tests/reductions.rs` and
# `ops.rs`); what is left here is about the binding's own entry points.

def _ops_fields(n_nodes, seed=20260826):
    """A smooth field and a random one: a sign slip can cancel in the first, not the second."""
    x = np.linspace(0.0, 1.0, n_nodes)
    rng = np.random.default_rng(seed)
    return [np.sin(3.0 * np.pi * x) + 0.4 * x * x, rng.standard_normal(n_nodes)]


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


def test_inner_is_exactly_norm2_when_the_operands_coincide():
    # Not a tautology across the boundary: `norm2` is a separate binding entry point, and the two
    # would drift apart if it ever grew its own summation.
    f = _ops_fields(65)[0]
    assert physsynth_rs.inner(f, f, 0.01) == physsynth_rs.norm2(f, 0.01)


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

# -- the shims are the Rust objects ---------------------------------------------------------------
#
# Moved here from `tests/test_stability.py` at retirement plan §50, when that file was deleted. The
# shims in `physsynth/core/` and the binding go TOGETHER, at phase F's last step, and the binding
# reads four of them back by name (`airbox`, `connection`, `plate`, `string_geometric`): deleting
# this guard one step early would leave the identity claim unasserted for the whole interval with
# every test still green, which is the gap `half_deleted_bodies` once existed to close. It dies
# with the shims, in this file, which is where a property that lives only as long as the binding
# belongs.


def test_no_module_chooses_between_two_implementations():
    # The guard that stood here for the whole migration was `test_the_rust_swap_matches_the_
    # environment`: `PHYSSYNTH_RS=1 pytest` claimed to run the suite against Rust, nothing in the
    # tests mentioned Rust, so a mistyped variable or a swap landing after its clients' imports
    # would have been green while testing Python. Phase A (python-retirement-plan §21) deleted the
    # last three modules that read the flag -- `operators` and `exciter` lost their Python bodies,
    # `banded` went whole -- so there is no second implementation for the environment to select,
    # and the guard's three flag-shaped halves went with it rather than being left empty:
    #
    #   * the `_USE_RUST` reader tuple (operators, exciter, banded) had nothing left to read;
    #   * `ported_expected`, the `<name>_py` FUNCTION table, had no aliases left to derive over;
    #   * the `if expected_rust:` captured-binding block compared names that are now one object
    #     with nothing to mis-order -- `string_stiff.biharmonic_matrix is operators.biharmonic_
    #     matrix`, `reed.Bore is bore.Bore` and the rest had become `x is x` (§42.4's rule).
    #
    # What survives is the claim those halves were approximating, WIDENED to the package: no
    # module in `physsynth.core` reads the flag or keeps a reference alias, and each public name
    # that is a Rust object is the Rust object.
    from physsynth.core import (
        airbox,
        beam,
        body,
        bore,
        bow,
        collision,
        connection,
        exciter,
        mallet,
        membrane,
        operators,
        operators2d,
        plate,
        radiation,
        reed,
        string_damped,
        string_geometric,
        string_ideal,
        string_nonlinear,
        string_stiff,
    )

    # The names each module must resolve to the Rust object, listed BY HAND: a deletion is a
    # reviewed edit, and a name that quietly stopped being a re-export must fail here.
    deleted_bodies = {
        string_ideal: {"IdealString"},
        membrane: {"Membrane"},
        # `MalletPlate` was never written in Python at all; a model born in Rust must not be able
        # to slip in without the same claim.
        mallet: {"MalletMembrane", "MalletPlate", "MalletWall"},
        bore: {"Bore"},
        reed: {"ReedBore", "bernoulli_flow"},
        body: {"ModalBody"},
        radiation: {
            "AirRadiation",
            "RadiatedBody",
            "RationalAirLoad",
            "ReactiveRadiatedBody",
            "monopole_radiation_resistance",
            "piston_radiation_resistance",
        },
        # The module keeps three SciPy names that this loop cannot speak about, because they are
        # not Rust objects; their guard is above, in
        # `test_the_scipy_names_the_binding_reads_are_still_scipys`.
        connection: {
            "StringBodyBridge",
            "StringPlateBridge",
            "StringVKPlateBridge",
            "SympatheticStrings",
        },
        # The seams are included because `airbox_wrap.rs` reads them off this module's namespace
        # by name, and a wrong one there is a silently different seam.
        airbox: {
            "AirBox",
            "InteriorSurfacePort",
            "RoomLoadedBody",
            "RoomLoadedMembrane",
            "RoomLoadedPlate",
            "RoomLoadedVKPlate",
            "RoomPort",
            "RoomSuspendedMembrane",
            "RoomSuspendedPlate",
            "RoomSuspendedVKPlate",
            "SurfacePort",
            "_MembraneSurface",
            "_PlateSurface",
            "_VKPlateSurface",
            "impedance_from_zeta",
        },
        string_stiff: {"StiffString"},
        string_damped: {"DampedStiffString"},
        string_nonlinear: {"TensionModulatedString"},
        string_geometric: {"GeometricString"},
        bow: {"BowedString", "friction_smooth", "friction_smooth_deriv"},
        collision: {
            "BarrierString",
            "contact_potential",
            "contact_force_elastic",
            "contact_stiffness",
            "contact_force_dg",
            "contact_force_total",
            "solve_contact",
        },
        # `GrainSpec` stays Python because the Rust helper CONSTRUCTS it, reaching back through
        # `py.import("physsynth.core.plate")`.
        plate: {"Plate", "VKPlate", "grain_ratios_from_material"},
        beam: {"FreeBeam"},
        operators2d: {"VonKarmanBracket", "AiryStressSolver"},
        # Phase A. The binding's signatures are the deleted body's, keywords and defaults included,
        # so these are bare re-exports.
        exciter: {"triangular_pluck", "raised_cosine", "raised_cosine_2d"},
    }
    for module, names in deleted_bodies.items():
        for name in sorted(names):
            assert getattr(module, name) is getattr(physsynth_rs, name), (
                f"{module.__name__}.{name} must be the Rust object"
            )

    # The DELEGATING modules, where the inverse is the claim. The binding hands every matrix back
    # as CSR triplets -- a Rust crate cannot construct a `scipy.sparse.csr_matrix` -- so each
    # builder stays a Python function that puts a matrix back around the result, and `operators`
    # also coerces its inputs to what NumPy would have accepted. Asserting `is physsynth_rs.<name>`
    # would be asserting that the shim had been bypassed and a caller was being handed a 4-tuple.
    delegating = {
        operators2d: {
            "grid_coords", "rectangle_mask", "disk_mask", "guitar_half_width", "guitar_scale",
            "guitar_mask", "guitar_area", "live_cells", "cells_per_node", "prune_to_area_carrying",
            "laplacian_from_mask", "biharmonic_from_mask", "_dirichlet_interior_d2_1d",
            "orthotropic_biharmonic", "free_plate_stiffness", "free_plate_stiffness_from_mask",
            "_collocated_d2_1d", "_forward_d1_1d", "_centered_d2_1d", "_clamped_d2_1d",
            "_avg_d1_1d", "embed", "inner2d", "norm2_2d",
        },
        operators: set(operators.__all__),
    }
    for module, names in delegating.items():
        for name in sorted(names):
            fn = getattr(module, name)
            assert fn is not getattr(physsynth_rs, name.lstrip("_"), None), (
                f"`{module.__name__}.{name}` is the Rust function itself -- the delegating "
                "wrapper has been bypassed"
            )
            assert getattr(fn, "__module__", None) == module.__name__, (
                f"`{module.__name__}.{name}` is no longer defined in this module ({fn!r}) -- the "
                "wrapper is the module's whole remaining body and cannot be re-exported away"
            )
    # `collision`'s underscored names, and the one that must NOT be a bare re-export.
    for name in ("_contact_force_total_deriv", "_force_total_vec", "_deriv_total_vec"):
        assert getattr(collision, name) is getattr(physsynth_rs, name[1:]), (
            f"`collision.{name}` must be the Rust function -- the underscored spelling is the one "
            "the model and the mallet reach for"
        )
    assert collision.solve_contact_vector is not physsynth_rs.solve_contact_vector, (
        "`collision.solve_contact_vector` must stay a Python wrapper: `stacklevel=2` on its "
        "non-convergence warning cannot mean the same thing from inside an extension module"
    )

    # The package-wide half, DERIVED rather than listed: `pkgutil` reads the directory, so a module
    # added tomorrow is in the population without anyone remembering it (ledger #67). Before phase
    # A the flag check covered a three-module tuple and the alias check covered the modules in
    # `deleted_bodies`; both now cover everything, which is what makes them worth more at zero.
    all_core_modules = [
        importlib.import_module(f"physsynth.core.{info.name}")
        for info in pkgutil.iter_modules(physsynth.core.__path__)
        if not info.ispkg
    ]
    # Named POSITIVE CONTROLS rather than a floor (finding #61): an `iter_modules` over the wrong
    # path yields nothing, and every assertion below would then pass over an empty list.
    for named in (connection, plate, string_ideal, operators):
        assert named in all_core_modules, (
            f"`{named.__name__}` is not in the scanned set, so `pkgutil.iter_modules` is not "
            "reaching `physsynth/core/` and this guard is checking nothing"
        )
    for module in all_core_modules:
        assert not hasattr(module, "_USE_RUST"), (
            f"{module.__name__} reads PHYSSYNTH_RS again -- the flag chose between two "
            "implementations, and phase A left one"
        )
        leftovers = [a for a in dir(module) if a.endswith("Py") or a.endswith("_py")]
        assert not leftovers, (
            f"{module.__name__} defines the reference alias(es) {sorted(leftovers)} -- either a "
            "Python reference implementation came back, or an alias outlived its body"
        )
