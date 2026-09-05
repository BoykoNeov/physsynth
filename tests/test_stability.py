"""Stability and construction guards (criterion 4) + the headless-core invariant.

- No NaN / blow-up across a sweep of valid lambda in (0, 1].
- A deliberate lambda > 1 is rejected at construction time, not silently run into an overflow.
- Non-physical parameters are rejected.
- physsynth.core imports no plotting/audio library (CLAUDE.md non-negotiable #4).
"""

import os
import subprocess
import sys

import numpy as np
import pytest
from helpers import make_string, wave_speed

from physsynth.core.engine import simulate
from physsynth.core.exciter import triangular_pluck
from physsynth.core.string_ideal import IdealString


@pytest.mark.parametrize("lam", [0.999, 0.95, 0.9, 0.75, 0.5, 0.3, 0.1])
def test_no_nan_across_valid_lambda(lam):
    string = make_string(N=100, lam=lam)
    string.set_state(triangular_pluck(string.x, string.L, 0.3 * string.L, amplitude=1e-3))
    res = simulate(string, num_steps=int(0.5 * string.fs), pickup_index=50)
    assert np.all(np.isfinite(res.output))
    assert np.all(np.isfinite(res.energy))


def test_lambda_above_one_rejected_at_construction():
    c, L, N = wave_speed(), 1.0, 100
    # lambda = c*N / (fs*L): a LOWER fs (coarser time step) raises lambda, so divide by 1.05.
    fs_unstable = c * N / (L * 1.05)  # forces lambda = 1.05 > 1
    with pytest.raises(ValueError, match="CFL"):
        IdealString(L=L, T=200.0, rho=0.005, fs=fs_unstable, N=N)


def test_lambda_exactly_one_is_accepted():
    # The CFL guard must not reject the exact (and most accurate) lambda = 1 case on round-off.
    string = make_string(N=100, lam=1.0)
    assert string.lam == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"rho": -1.0},
        {"T": 0.0},
        {"L": -2.0},
        {"sigma": -0.1},
        {"N": 1},
        {"boundary": "clamped"},
    ],
)
def test_invalid_parameters_rejected(kwargs):
    base = {"L": 1.0, "T": 200.0, "rho": 0.005, "fs": 20000.0, "N": 100}
    base.update(kwargs)
    with pytest.raises(ValueError):
        IdealString(**base)


def test_core_is_headless():
    # Import the whole core in a fresh interpreter and assert no plotting/audio library was pulled
    # in (transitively included). The core must stay portable to C++/Rust later.
    code = (
        "import sys;"
        "import physsynth.core.operators, physsynth.core.string_ideal,"
        "       physsynth.core.exciter, physsynth.core.engine;"
        "forbidden={'matplotlib','sounddevice','pyaudio','pygame','PyQt5','PySide6'};"
        "hit=sorted(m for m in sys.modules if m.split('.')[0] in forbidden);"
        "print(','.join(hit));"
        "sys.exit(1 if hit else 0)"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, f"core imported forbidden libraries: {result.stdout.strip()}"


# --- portability contract (docs/dev/portability-contract.md) ----------------------------------

# Boilerplate run in a *fresh* interpreter: import every submodule of physsynth.core so these
# guards auto-cover new core modules (e.g. a future string_stiff.py) with no edits here.
_IMPORT_ALL_CORE = (
    "import sys, importlib, pkgutil;"
    "import physsynth.core as _core;"
    "[importlib.import_module(m.name) "
    " for m in pkgutil.iter_modules(_core.__path__, _core.__name__ + '.')];"
)


def _run_core_probe(body: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", _IMPORT_ALL_CORE + body], capture_output=True, text=True
    )


# Hardcoded allowlist of top-level third-party packages the core is permitted to pull in: the
# declared numeric stack (numpy + scipy) plus the compiled-extension runtime baggage that stack
# unavoidably drags along. Verified empirically -- importing the numpy/scipy stack alone pulls
# exactly {numpy, scipy, charset_normalizer, cython_runtime, <hash>__mypyc}. Anything outside this
# set (torch, requests, PIL, sounddevice, ...) is a real portability leak and must fail the test.
# The mypyc runtime is named with a per-build hash prefix (e.g. "81d243...__mypyc"), so it is
# matched structurally by its "__mypyc" suffix, not by name.
#
# `physsynth_rs` (docs/dev/rust-migration-plan.md §2.2) is the ONE deliberate, reviewed addition
# this list has ever taken. It is the compiled Rust core, and `sys.modules` cannot tell it apart
# from any other third-party binary -- which is the whole reason it has to be named here rather
# than pattern-matched. Two things follow, and both are the point:
#
#   - It only appears when `PHYSSYNTH_RS` is set, so on the default path this entry is inert. The
#     list did not get looser for the runs that produce the acceptance numbers.
#   - The rule it used to carry alone -- "the core depends on the numeric stack and nothing else"
#     -- now lives on the Rust side too, as `crates/physsynth-core/tests/deps.rs`, which checks
#     the same thing against `cargo metadata`. This test can no longer see inside the extension
#     module, so something has to, and that is it. Same spirit, same visibility, other language.
_CORE_DEP_ALLOWLIST = {
    "numpy",
    "scipy",
    "charset_normalizer",
    "cython_runtime",
    "physsynth",
    "physsynth_rs",
}


def test_core_dependency_allowlist():
    # Stronger than the blocklist above: the core may use ONLY the allowlisted numeric stack and
    # its compiled-runtime baggage -- no third-party dependency of its own. Import every core
    # submodule (auto-discovers new ones, e.g. string_stiff) and assert nothing outside the
    # allowlist appears. We measure the DELTA -- modules pulled *by importing the core*, not the
    # absolute set -- by snapshotting sys.modules first: this excludes interpreter-startup baggage
    # injected via a .pth (e.g. Windows pywin32's pywin32_bootstrap/pywin32_system32), which is
    # present for *any* subprocess and is not something the core pulls. Underscore-private modules
    # (_csparsetools, editable-install finders, ...) are internal plumbing, excluded by the
    # leading-underscore rule; the hash-suffixed mypyc runtime is excluded by its "__mypyc" suffix.
    allowed = sorted(_CORE_DEP_ALLOWLIST)
    probe = (
        "import sys, importlib, pkgutil;"
        "stdlib=set(sys.stdlib_module_names)|set(sys.builtin_module_names);"
        "before=set(sys.modules);"
        "import physsynth.core as _core;"
        "[importlib.import_module(m.name) "
        " for m in pkgutil.iter_modules(_core.__path__, _core.__name__ + '.')];"
        "allowed=set(" + repr(allowed) + ");"
        "tp={n.split('.')[0] for n in set(sys.modules) - before"
        "    if n.split('.')[0] not in stdlib and not n.startswith('_')"
        "    and not n.endswith('__mypyc')};"
        "leaked=sorted(tp - allowed);"
        "print(','.join(leaked));"
        "sys.exit(1 if leaked else 0)"
    )
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    assert result.returncode == 0, (
        f"core pulled third-party module(s) outside the allowlist {allowed}: "
        f"{result.stdout.strip()}"
    )


def test_the_rust_swap_matches_the_environment():
    # The guard without which the whole Rust CI job can be green and mean nothing.
    #
    # `PHYSSYNTH_RS=1 pytest ...` claims to run the existing tests against the Rust model. But
    # nothing in those tests mentions Rust -- the substitution happens at the bottom of
    # `physsynth/core/string_ideal.py` -- so if the variable were mistyped, or the swap block were
    # refactored away, or an import-ordering change defeated it, the suite would run against
    # PYTHON and pass. Green, and asserting nothing about the port. That is verbatim the failure
    # mode `docs/dev/rust-migration-plan.md` §1 exists to prevent, one level up.
    #
    # This could never live in the parity family: those files import both implementations by name
    # and are immune to the swap by design. It belongs here, with the other three portability
    # guards, and it runs on BOTH paths -- the default one included, where it asserts the Rust
    # model is *not* silently in play.
    from physsynth.core import (
        airbox,
        banded,
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

    expected_rust = os.environ.get("PHYSSYNTH_RS", "").strip() not in ("", "0", "false", "False")

    # The DELETED half of the guard, added with unit 7 (plan 39). A module whose Python reference
    # implementation has been deleted has no second implementation to choose between, so it has no
    # `_USE_RUST`, no `<Name>Py` alias, and its public name is the Rust class on BOTH paths. Each
    # of those three is asserted rather than assumed, and the module is named here by hand for the
    # same reason the tuples below are: a deletion has to be a reviewed edit. Without this, a
    # module that quietly lost its swap block would simply fall out of the derive and nothing would
    # notice -- which is section 17.6's finding for a third time, now on the way *out*.
    physsynth_rs = pytest.importorskip("physsynth_rs")

    # The names each deleted module must now resolve to Rust, listed BY HAND. Classes and
    # functions together: both conventions are covered here because a deletion removes both
    # (`<Name>Py` for a class, `<name>_py` for a function), and the two halves further down the
    # file each only know about one of them.
    deleted_bodies = {
        string_ideal: {"IdealString"},
        membrane: {"Membrane"},
        mallet: {"MalletMembrane", "MalletWall"},
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
        # The LAST Python model body in the project, deleted by unit 10 (plan section 49). Four
        # classes and no function aliases -- and the module keeps three SciPy names in its
        # namespace that this loop cannot speak about, because they are not Rust objects. They get
        # their own guard in `tests/test_binding_surface.py`; see section 49.2.
        connection: {
            "StringBodyBridge",
            "StringPlateBridge",
            "StringVKPlateBridge",
            "SympatheticStrings",
        },
        # Unit 6 landed in two commits (plan sections 47 and 48) and passed through
        # `half_deleted_bodies` on the way. Fourteen names: the room, the three ports, the seven
        # wrappers and the three seams -- the seams included, because `airbox_wrap.rs` reads them
        # off this module's namespace by name and a wrong one there is a silently different seam.
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
        # `collision`'s three private spellings are checked separately below: the alias namespace
        # was flat and this module's is not, so `_force_total_vec` could only ever be aliased
        # `force_total_vec_py`, and that mismatch is what once made the whole module fall out of
        # the derived table.
        collision: {
            "BarrierString",
            "contact_potential",
            "contact_force_elastic",
            "contact_stiffness",
            "contact_force_dg",
            "contact_force_total",
            "solve_contact",
        },
        # `plate` is unit 5. Both classes and the material helper are plain re-exports; what the
        # module keeps is `GrainSpec` -- which the Rust helper CONSTRUCTS, reaching back through
        # `py.import("physsynth.core.plate")` -- plus two type aliases and `THETA_DEFAULT`.
        plate: {"Plate", "VKPlate", "grain_ratios_from_material"},
        # `beam` is unit 8, and it is the plainest deletion in this table: `FreeBeam` is a bare
        # re-export and the module keeps nothing but `Boundary` and `THETA_DEFAULT`.
        beam: {"FreeBeam"},
        # `operators2d` is unit 5's other half and only these two names can be checked this way:
        # everything else in it is a *delegating wrapper* rather than a re-export, and is asserted
        # below in the shape that fits.
        operators2d: {"VonKarmanBracket", "AiryStressSolver"},
    }

    # `operators2d` is the second module in the project with no core half (`airbox.py`'s wrapper
    # tier was the first). The binding returns every matrix as CSR *triplets* -- a Rust crate
    # cannot construct a `scipy.sparse.csr_matrix` -- so each builder here stays a Python function
    # that puts a matrix back around the result. Asserting `is physsynth_rs.<name>` on one of them
    # would therefore be asserting that the shim had been bypassed and a caller was being handed a
    # four-tuple. The claim that fits is the inverse, and it is the same shape as
    # `collision.solve_contact_vector`'s above.
    for name in sorted(
        {
            "grid_coords", "rectangle_mask", "disk_mask", "guitar_half_width", "guitar_scale",
            "guitar_mask", "guitar_area", "live_cells", "cells_per_node", "prune_to_area_carrying",
            "laplacian_from_mask", "biharmonic_from_mask", "_dirichlet_interior_d2_1d",
            "orthotropic_biharmonic", "free_plate_stiffness", "free_plate_stiffness_from_mask",
            "_collocated_d2_1d", "_forward_d1_1d", "_centered_d2_1d", "_clamped_d2_1d",
            "_avg_d1_1d", "embed", "inner2d", "norm2_2d",
        }
    ):
        fn = getattr(operators2d, name)
        assert fn is not getattr(physsynth_rs, name.lstrip("_"), None), (
            f"`operators2d.{name}` is the Rust function itself -- the shim that rebuilds a "
            "`csr_matrix` from the binding's triplets has been bypassed"
        )
        assert getattr(fn, "__module__", None) == "physsynth.core.operators2d", (
            f"`operators2d.{name}` is no longer defined in this module ({fn!r}) -- the delegating "
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

    # `half_deleted_bodies` stood here for one commit, holding `airbox` between the two halves of
    # its deletion (plan sections 47 and 48). Its five names merged into `deleted_bodies` above
    # when the wrapper tier went, and the table is DELETED rather than left empty: an empty dict
    # iterates zero times and asserts nothing while reading as a live guard, which is finding #52
    # (an empty `parametrize` collects as a skip) through a second door. If a future deletion has
    # to land in halves again, the shape to bring back is in section 47.3.

    for module, names in deleted_bodies.items():
        assert not hasattr(module, "_USE_RUST"), (
            f"{module.__name__} has a deleted Python body but still reads PHYSSYNTH_RS -- the "
            "flag chooses between two implementations and there is one"
        )
        leftovers = [a for a in dir(module) if a.endswith("Py") or a.endswith("_py")]
        assert not leftovers, (
            f"{module.__name__} has a deleted Python body but still defines the reference "
            f"alias(es) {sorted(leftovers)} -- there is nothing left for them to name"
        )
        for name in sorted(names):
            assert getattr(module, name) is getattr(physsynth_rs, name), (
                f"{module.__name__}.{name} must be the Rust object on both paths, flag or no flag"
            )

    for module in (
        operators,
        # `airbox` was here until unit 6's deletion. It read PHYSSYNTH_RS from Phase 5's fourth
        # batch (for the `splu` swap) and was not added to this tuple until the fifth ported
        # `AirBox` itself -- a whole batch in which its reading of the flag could have diverged
        # with nothing noticing. It reads no flag at all now: `deleted_bodies` asserts that.
        exciter,
        banded,
        # `connection` was here until unit 10's deletion, and it was the last MODEL in the tuple:
        # what is left reads the flag to choose between two implementations of an operator or a
        # solver, not of a resonator. `deleted_bodies` above asserts it reads no flag at all now.
    ):
        assert module._USE_RUST is expected_rust, (
            f"{module.__name__}'s reading of PHYSSYNTH_RS disagrees with this test's -- one of "
            "the two changed without the other"
        )

    # The class half of the guard, DERIVED rather than listed -- section 17.6's finding applied to
    # the half it was not applied to. `collision` fell out of the function table below for a whole
    # batch because its names were spelled unexpectedly, and nothing failed; a block of
    # hand-written `assert mallet.MalletWall is physsynth_rs.MalletWall` lines has exactly the same
    # hole, one forgotten paste wide. So the set of swapped CLASSES is read off the `<Name>Py`
    # aliases the modules actually define, and then checked against a written-down expectation --
    # which is what keeps adding a port a reviewed edit rather than a silent one.
    # ALL of them, and the widening is unit 10's doing. This derive read a hand-written tuple
    # for eight phases, and three separate findings in this file say the same thing about it: it
    # is only as wide as the tuple, so a module absent from the tuple could grow a `<Name>Py`
    # alias with nothing noticing (`collision` did, for a whole batch). With the last Python model
    # body deleted the expectation is EMPTY, and an empty expectation checked against a
    # zero-length tuple asserts nothing at all -- finding #52 through a fourth door. So the tuple
    # becomes every core module this test already imports, and the claim becomes the migration's
    # completion canary: no module in `physsynth.core` chooses between two implementations of a
    # CLASS any more. That is a live statement about twenty-two modules, and it is the one that
    # would fail if a Python reference implementation were ever reintroduced.
    all_core_modules = (
        airbox,
        banded,
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
    swapped_classes = {}
    for module in all_core_modules:
        for alias in dir(module):
            if not alias.endswith("Py") or alias.endswith("_py"):
                continue
            name = alias[:-2]
            reference = getattr(module, alias)
            if isinstance(reference, type) and isinstance(getattr(module, name, None), type):
                swapped_classes[(module.__name__, name)] = (module, name, reference)

    assert swapped_classes == {}, (
        f"the swapped classes are {sorted(swapped_classes)}, but every Python model body has "
        "been deleted (plan section 49) -- either a reference implementation came back, or a "
        "`<Name>Py` alias outlived the body it named"
    )

    # The two arms that stood here -- "with the flag set the public name must be Rust", "without
    # it, Python" -- iterated over `swapped_classes` and are gone with it. They were the whole
    # point of the derive for eight phases and they are now unreachable by construction: with no
    # Python class left, `deleted_bodies` above makes the STRONGER claim (the public name is the
    # Rust object on BOTH paths, flag or no flag) for every one of them. Keeping the loops would
    # have been two `for` statements over an empty dict, which is the shape this batch removed
    # three times.

    # The operators (plan Phase 1) need the same guard for the same reason, and it has to be
    # spelled differently: they are functions, so there is no class identity to compare. The
    # question that matters is whether the PUBLIC name still refers to the `_py` implementation.
    #
    # This one is wider than the string's. `string_stiff`, `string_damped`, `string_nonlinear`,
    # `string_geometric` and `beam` all do `from .operators import ...` at import time, so the
    # binding they captured is fixed the moment their module was first imported. If the swap ever
    # landed after them -- a lazy import, a reordered `physsynth.core.__init__` -- those five would
    # hold the Python functions while `operators` reported Rust, and the run would be green while
    # testing the wrong thing for five models at once.
    #
    # The set of swapped names is DERIVED from the `_py` aliases the module actually defines, not
    # listed here -- so a function ported without an alias escapes nothing and a function aliased
    # without being swapped fails loudly. It is then checked against a written-down expectation,
    # which is what makes adding a port a reviewed edit rather than a silent one (the same
    # reasoning as the hardcoded dependency allowlist below).
    ported_expected = {
        # `airbox` was here until unit 6's first deletion, with three function aliases. They
        # left in the two directions unit 5 established: `impedance_from_zeta` is public and is a
        # re-export now, so its claim moved to `half_deleted_bodies` above; `face_axes` and
        # `free_pressure_nodes` were private helpers whose only readers through this module were
        # the parity tests deleted alongside them, so they are not re-exported at all and there is
        # no claim left to move.
        # `operators` is ported in full (plan Phase 1).
        operators: set(operators.__all__),
        exciter: set(exciter.__all__),
        # `banded` is ported in full (plan Phase 3), and it is the odd one out in this table: its
        # `_py` aliases are SciPy's functions, not a Python transcription, because what this
        # module ports is a choice of solver rather than a piece of arithmetic somebody wrote
        # here. It is also the only swap so far that deliberately changes the numbers -- see the
        # module's own header for why that is safe and what it buys.
        banded: set(banded.__all__),
        # `reed` was here until unit 3's deletion. `bernoulli_flow` still has to resolve to Rust
        # -- `tests/test_reed_stability.py` imports it by name and asserts its oddness and
        # passivity directly -- but there is no longer a Python function to compare it against, so
        # the claim moved to `deleted_bodies` above. `radiation` was NEVER in this table even
        # though it had two `_py` aliases, which is the "a derive is only as wide as the tuple it
        # derives over" hole a FIFTH time: its swap could have come undone between section 37.11's
        # piston fix and unit 2's deletion with nothing noticing. Both are covered above now.
        # `collision` is ported in full, `BarrierString` included since Phase 3's last batch --
        # the class half of that is checked by identity above. This entry was missing until the
        # mallet's batch, and the reason is worth writing down rather than quietly fixing: three
        # of the module's public
        # names carry a LEADING UNDERSCORE while their aliases do not
        # (`_force_total_vec` <-> `force_total_vec_py`), so the derive below could not find them
        # and the whole module fell out of the table. A guard that silently covers nothing is the
        # same shape as the empty parity job section 16.8 found, reached through a third door --
        # so the lookup now tries the underscored spelling too, which keeps the set DERIVED rather
        # than listed.
        # `plate` and `operators2d` were here until unit 5's deletion, and they left in two
        # different directions -- which is the batch's own finding. `plate`'s three names are
        # re-exports and moved to `deleted_bodies` with everything else. `operators2d`'s could
        # not: its functions are still Python, because the binding hands back CSR triplets and
        # something has to rebuild the matrix, so the claim about them is the *inverted* one
        # written above the loop over `deleted_bodies`.
        # `bow` and `collision` were here until unit 1's deletion. Their functions still have to
        # resolve to Rust -- `tests/test_bow_stability.py` imports the friction curve by name and
        # asserts its oddness, peak and derivative directly -- but with no Python function left to
        # compare against, the claim moved to `deleted_bodies` above, together with `collision`'s
        # three underscored spellings and the one wrapper that must stay Python.
    }

    def _public(module, name):
        """The public name an alias `<name>_py` refers to -- underscored or not.

        `collision` exports `_force_total_vec` but aliases it `force_total_vec_py`, because the
        alias namespace is flat and the module's is not.
        """
        if hasattr(module, name):
            return getattr(module, name)
        return getattr(module, f"_{name}")

    for module, expected_names in ported_expected.items():
        aliased = {n[:-3] for n in dir(module) if n.endswith("_py") and not n.startswith("_")}
        assert aliased == expected_names, (
            f"{module.__name__}'s `_py` aliases are {sorted(aliased)}, but this guard expects "
            f"{sorted(expected_names)} -- a port landed (or left) without the guard being updated"
        )
        for name in sorted(expected_names):
            public = _public(module, name)
            reference = getattr(module, f"{name}_py")
            if expected_rust:
                assert public is not reference, (
                    f"PHYSSYNTH_RS is set but `{module.__name__}.{name}` is still the Python "
                    "function: this run is NOT exercising the Rust code, whatever it reports"
                )
            else:
                assert public is reference, (
                    f"PHYSSYNTH_RS is unset but `{module.__name__}.{name}` is not the Python "
                    "function -- the default path must stay the one the numbers came from"
                )

    if expected_rust:
        from physsynth.core import string_stiff

        # The banded solver's captured-binding check stood here and was the widest one in this
        # test: four models did `from .banded import cho_solve_banded, cholesky_banded` at module
        # scope, and they are chained by `array_equal` reduction anchors -- sigma1 = 0, EA = 0,
        # EA = T -- that hold only while all four do the SAME arithmetic, so a swap reaching three
        # of them would have made those anchors compare LAPACK against a transcription and fail
        # with a message about physics.
        #
        # Unit 1's deletion RETIRES it, and the reason is worth stating rather than leaving as an
        # absence. A Rust model does not capture a Python solver; all four now factor inside the
        # crate, so the hazard the check existed for cannot arise. The anchors it protected are
        # still asserted, in `test_geometric_limits.py` and its siblings, and they now compare four
        # Rust models against each other -- which is the same claim with both sides moved.
        #
        # `string_stiff.biharmonic_matrix` is NOT retired with it: that module keeps the re-export
        # precisely so this check can still be made, because `operators` still has a Python body
        # and a mis-ordered swap there is still possible.
        assert string_stiff.biharmonic_matrix is operators.biharmonic_matrix, (
            "`string_stiff` captured a different `biharmonic_matrix` than `operators` now "
            "exposes -- the swap landed after that module was imported, so the model is running "
            "on the Python operator while this run claims otherwise"
        )
        # The beam's capture stood here and unit 8's deletion retires it, in the first of §42.4's
        # two directions: `beam.py` did `from .operators import free_beam_stiffness` at module
        # scope, so a swap landing after its import would have built the free-edge operator out of
        # the Python code while the run reported Rust. There is no capture left to check -- the
        # module imports one class from the extension and nothing from `operators` at all. The
        # claim itself has not gone anywhere: `crates/physsynth-core/tests/beam.rs` asserts the
        # resonator's `K` against `ops::free_beam_stiffness` to the bit, on the side where both
        # halves now live.
        # The membrane's version of the same hazard stood here: `mallet` does
        # `from .membrane import Membrane` at import time, so a swap landing after it would have
        # left the mallet striking a Python drumhead while the run reported Rust. Unit 4's
        # deletion retires it for the same reason as the banded block -- there is one `Membrane`
        # now and no swap to mis-order. The import itself stays in `mallet.py`, because callers
        # reach it as `mallet.Membrane`.
        assert membrane.laplacian_from_mask is operators2d.laplacian_from_mask, (
            "`membrane` captured a different `laplacian_from_mask` than `operators2d` now exposes"
        )
        # TWO plate captures stood here and unit 5's deletion retires both, for the two different
        # reasons §42.4 separates.
        #
        # The first was `plate` against `operators2d`: `plate.py` did `from .operators2d import
        # guitar_mask, biharmonic_from_mask, ...` at module scope, so it was a CLIENT of nine
        # swapped names and a swap landing after its import would have built the plate's outline
        # -- or, worse, the operator `Plate.step` multiplies by twice a timestep -- out of the
        # Python code while the run reported Rust. It is retired because `plate.py` has no body
        # left to capture anything with: the module imports two classes from the extension and
        # nothing from `operators2d` at all. Not "both sides are Rust" but "there is no capture".
        #
        # The second was `connection` against `plate` -- `connection.py` still does
        # `from .plate import Plate, VKPlate` at module scope, and that import is still live. But
        # `plate.Plate` is now an unconditional `from physsynth_rs import Plate`: there is no swap
        # to land late, so the assertion has become `x is x`. That is §42.4's rule exactly, and a
        # captured-binding check kept past it is a guard that passes vacuously.
        #
        # `airbox.py` needed no entry in either and that is worth keeping on the record: its
        # `from physsynth.core.plate import Plate, VKPlate` sits under `if TYPE_CHECKING:`, so it
        # captures nothing at runtime. What airbox DOES capture is `splu` -- and that is no longer
        # a swap either, because unit 5 made the crate's LU unconditional there (plan §43.4): the
        # plate factors with the crate on every path now, so the wrapper tier has to as well or
        # the four bare-versus-loaded reduction anchors compare two solvers.

        # The body's version. It had TWO importers -- `connection` and `radiation` both did
        # `from .body import ModalBody` at module scope, so a swap landing after either would
        # leave the whole body/radiation leg on Python while this run reported Rust. (`airbox`
        # imports the name only under TYPE_CHECKING and so captures nothing at runtime --
        # deliberately, and worth not un-doing.)
        #
        # `connection` left the tuple with unit 10: its shim imports no collaborator at all, so
        # there is no capture to check rather than a vacuous one. That is the same retirement the
        # `plate` entry above got, reached the other way -- there the captured name stopped being
        # a swap, here the capture stopped existing.
        for client in (radiation,):
            assert client.ModalBody is body.ModalBody, (
                f"`{client.__name__}` captured a different `ModalBody` than `body` now "
                "exposes -- the swap landed after that module was imported"
            )

        # The string's version, and the one this batch added. `bow.py` does `from .string_damped
        # import DampedStiffString` at module scope, so a swap that landed after it would leave the
        # bow drawing across a PYTHON string while this run reported Rust. Unlike the four
        # `cho_solve_banded` captures above, this one is caught for free at construction -- the
        # Rust `BowedString` raises `TypeError` on a Python string rather than falling back -- so
        # the assertion is here to say WHICH of the two broke, not to be the only thing that
        # notices.
        assert bow.DampedStiffString is string_damped.DampedStiffString, (
            "`bow` captured a different `DampedStiffString` than `string_damped` now exposes -- "
            "the swap landed after that module was imported"
        )

        # The bore's version. `reed.py` does `from .bore import Bore` at module scope, so a swap
        # that landed after it would leave the clarinet blowing a PYTHON air column while this run
        # reported Rust -- and every reed test would still pass, because the reed's own physics is
        # unchanged by which bore it drives. Same hazard as `mallet.Membrane`, one model on.
        assert reed.Bore is bore.Bore, (
            "`reed` captured a different `Bore` than `bore` now exposes -- the swap landed "
            "after that module was imported"
        )
        # And the sharper version of the same question, which only this pair can ask: the Rust
        # reed REFUSES a Python bore rather than falling back to a per-step interpreter crossing,
        # so if the two swaps ever landed out of order the clarinet would not quietly run slow --
        # it would raise. Asserting it here means the failure is one line of test output rather
        # than a TypeError from inside somebody's model.
        rs_bore = bore.Bore(
            L=0.6, fs=343.0 / (0.6 / 60), N=60, boundary=("closed", "open")
        )
        reed.ReedBore(bore=rs_bore, p_mouth=1500.0)


def test_core_does_not_import_sibling_layers():
    # The dependency arrow points one way: analysis/viz/io depend on core, never the reverse.
    body = (
        "bad={'physsynth.viz','physsynth.analysis','physsynth.io'};"
        "hit=sorted(m for m in sys.modules if any(m==b or m.startswith(b+'.') for b in bad));"
        "print(','.join(hit));"
        "sys.exit(1 if hit else 0)"
    )
    result = _run_core_probe(body)
    assert result.returncode == 0, (
        f"core imported a sibling layer (must not): {result.stdout.strip()}"
    )


# -- the ARPACK start vector is pinned everywhere (rust-migration-plan.md Sec 7) ----------------
#
# `eigsh` without `v0` draws a RANDOM start vector, so the oracle it computes is not reproducible
# run to run. Measured on the free-free beam: elastic eigenvalues wobble ~1e-12 relative, their
# eigenvectors ~5e-11, and the two rigid-body modes come back as an arbitrary basis of the {1, x}
# nullspace (~1e-1 apart). An eigenvector fed to `set_state` is an INITIAL CONDITION, so that is a
# different trajectory, not a last-digit difference -- and it would read as a port bug. The first
# test asserts the property; the second asserts it cannot be lost by adding a call site, which is
# the shape of guard Sec 17.6 and Sec 23.7 record going quietly empty.


def test_arpack_oracles_are_bit_reproducible():
    from helpers import beam_low_eigenfrequencies, make_beam, make_free_plate
    from helpers import free_plate_low_eigenfrequencies as fp

    beam = make_beam(N=32)
    plate = make_free_plate(N=12)
    for name, first, second in (
        ("beam", beam_low_eigenfrequencies(beam, 4), beam_low_eigenfrequencies(beam, 4)),
        ("free plate", fp(plate, 3), fp(plate, 3)),
    ):
        assert np.array_equal(first, second), (
            f"{name} oracle is not bit-reproducible: {first} vs {second} -- an eigsh call lost "
            "its pinned v0"
        )


def test_every_eigsh_call_in_the_tests_pins_v0():
    import ast
    import pathlib

    here = pathlib.Path(__file__).parent
    unpinned = []
    for path in sorted(here.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", None)
            if name != "eigsh":
                continue
            if not any(kw.arg == "v0" for kw in node.keywords):
                unpinned.append(f"{path.name}:{node.lineno}")
    assert not unpinned, (
        "eigsh called without a pinned v0 (use helpers.arpack_v0), so the oracle is not "
        f"reproducible run to run: {', '.join(unpinned)}"
    )
