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

Deliberately **not** named ``test_rust_parity_*``: that prefix means "must run unflagged, because
it builds both sides itself" (``scripts/shard_tests.py``'s ``PARITY_PREFIX``). Nothing here builds
a Python side, so it should run in every shard of every job, flagged or not.
"""

import importlib

import numpy as np
import pytest
from scipy import sparse

import physsynth.core
from physsynth.core import plate, string_stiff
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
    # Not a numpy array: `web/serialize.py` deserialises JSON, and JSON has no ndarray.
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
    # `tests/test_collision_modal.py` does this to a string it owns (`bar.string.u = uf.copy()`),
    # so assignment has to work, not just mutation.
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
    ``u_prev`` and ``n``; ``_VKPlateSurface`` adds ``F``, ``F_prev`` and the three Picard
    diagnostics. None of that is optional — it is how the room puts its load inside the solve, and
    a ``#[getter]`` with no ``#[setter]`` takes the write away *silently* (§33.2)."""
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
    assert np.array_equal(v.F, fresh)
    assert v.n_iters == 4 and v.converged is False and v.last_residual == 1e-5


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
# together. `tests/helpers.py` passes `theta` explicitly on every construction, so the binding's own
# default is never exercised by the physics suite at all -- it could drift to any value and the
# whole suite would stay green.
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
