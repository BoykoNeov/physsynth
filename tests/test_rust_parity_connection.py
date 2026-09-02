"""Rust parity for ``physsynth.core.connection`` — the four bridges. Plan §34.

``StringBodyBridge``, ``StringPlateBridge``, ``StringVKPlateBridge`` and ``SympatheticStrings``,
ported as one unit because two exact anchors bind them (§15.2) — a one-string
``SympatheticStrings`` is ``array_equal`` to a ``StringBodyBridge``, and a ``nonlinear=False``
``StringVKPlateBridge`` must reproduce ``StringPlateBridge``'s stability margin to the last digit.
Both anchors are re-asserted here across the language boundary as well as within it.

**Everything is bit-identical**, the von Kármán bridge's trajectory included, and the reason is
§32's: this is the second module with no core half. It computes through Python for every sparse
assembly, both linear solves, ``np.linalg.eigvals`` and both ``np.dot``s, and what is Rust is the
control flow, the guards, the ledgers and the elementwise arithmetic between those calls — which
admits no reassociation. So exactness here is a sharp test of the *transcription* rather than a
claim about the dynamics.

Four things this file exists to check that no physics bar can see.

* **The write through the string's buffer.** ``step`` does ``self.string.u[-1] -= beta_s * F``, and
  ``lib.rs``'s module docstring names that one line as the reason the string's buffers are
  Python-owned. A port that read ``u`` into a vector and rebound it would pass every energy test
  and lose every snapshot; :func:`test_the_reaction_is_written_through_the_live_buffer` holds a
  reference across the step and asserts it moved.
* **Which reductions were transcribed and which were not.** ``np.sum`` is (§31.2), ``np.dot`` is
  not (§14.2) — and *both* choices need a witness, because the fixtures the suite ships cannot
  distinguish either. Every shipped body has four or five modes and every shipped sympathetic rig
  two or three strings, and below **eight** terms ``np.sum`` *is* a left-to-right loop (§30.2), so
  the transcription is dead code in every existing test. The two tests below search for a fixture
  that reaches past the cutoff.
* **The duck typing.** The reference contains no ``isinstance``, ``hasattr``, ``getattr`` or
  ``type(`` at all (§31.11): the ``body=`` slot takes four kinds of object and the ``plate=`` slot
  eight. A ``#[pyclass]`` that downcast its collaborators would still pass the whole airbox and
  radiation family — those hand it real models — so
  :func:`test_the_bridge_never_looks_at_its_collaborator_s_type` hands it an object that is not one.
* **§33.2's write question, aimed at the class being ported rather than at its collaborators.**
  A ``#[getter]`` with no ``#[setter]`` silently makes an attribute read-only, and the reference,
  being Python, allowed every one.
"""

from __future__ import annotations

import numpy as np
import pytest
from helpers import wave_speed

from physsynth.core import connection as conn
from physsynth.core.body import ModalBody
from physsynth.core.connection import (
    StringBodyBridgePy,
    StringPlateBridgePy,
    StringVKPlateBridgePy,
    SympatheticStringsPy,
)
from physsynth.core.exciter import triangular_pluck
from physsynth.core.plate import Plate, VKPlate
from physsynth.core.string_ideal import IdealString

# NOT a bare `import physsynth_rs`: the default gate does not build the extension, so a module-scope
# import is a collection error there rather than a skip.
physsynth_rs = pytest.importorskip(
    "physsynth_rs", reason="the Rust extension is not built in this environment"
)

L_STRING = 1.0
T_STRING = 200.0
RHO_STRING = 0.005
N_STRING = 48
LAM = 0.9
BODY_FREQS = np.array([137.0, 213.0, 330.0, 471.0])
N_PLATE = 8
BOUNDARIES = ("supported", "free")


def _fs(N=N_STRING, T=T_STRING):
    return wave_speed(T, RHO_STRING) * N / (L_STRING * LAM)


def _string(*, T=T_STRING, N=N_STRING, fs=None, pluck=1e-3):
    s = IdealString(
        L=L_STRING, T=T, rho=RHO_STRING, fs=fs or _fs(), N=N, boundary=("fixed", "free")
    )
    if pluck:
        s.set_state(triangular_pluck(s.x, s.L, 0.3 * s.L, amplitude=pluck))
    return s


def _body(*, fs=None, phi=1.0, masses=0.02, freqs=BODY_FREQS):
    return ModalBody(freqs=freqs, fs=fs or _fs(), sigmas=0.0, masses=masses, phi=phi)


def _plate(boundary="supported", *, fs=None):
    return Plate(
        Lx=1.0, Ly=1.0, kappa=20.0, rho=2.0, fs=fs or _fs(), N=N_PLATE, boundary=boundary
    )


def _vk_plate(boundary="supported", *, nonlinear=True, fs=None):
    return VKPlate(
        Lx=0.4,
        Ly=0.4,
        E=2.0e11,
        e=1.0e-4,
        nu=0.3,
        rho=7800.0,
        fs=fs or _fs(),
        N=N_PLATE,
        boundary=boundary,
        nonlinear=nonlinear,
    )


# -- builders. Each returns a fresh, identically-seeded rig for the class it is handed. -------


def _modal(cls, *, phi=1.0, masses=0.02, freqs=BODY_FREQS, K=8000.0):
    return cls(string=_string(), body=_body(phi=phi, masses=masses, freqs=freqs), K=K)


def _sympathetic(cls, J=3, *, K=None, seed=20260831):
    r = np.random.default_rng(seed)
    fs = _fs()
    strings = [_string(T=T_STRING * (1.0 + 0.03 * j), fs=fs, pluck=0.0) for j in range(J)]
    for j, s in enumerate(strings):
        amp = 1e-3 * (1.0 + 0.1 * j)
        s.set_state(triangular_pluck(s.x, s.L, 0.3 * s.L, amplitude=amp))
    if K is None:
        K = list(r.uniform(300.0, 900.0, J))
    return cls(strings=strings, body=_body(fs=fs), Ks=K)


def _plate_bridge(cls, boundary="supported", *, K=3000.0):
    return cls(string=_string(), plate=_plate(boundary), K=K)


def _vk_bridge(cls, boundary="supported", *, nonlinear=True, K=3000.0):
    return cls(string=_string(pluck=3e-3), plate=_vk_plate(boundary, nonlinear=nonlinear), K=K)


def _run_pair(build_py, build_rs, steps):
    """Step both arms together; return the worst state and energy divergence."""
    a, b = build_py(), build_rs()
    worst_u = 0.0
    worst_e = 0.0
    for _ in range(steps):
        a.step()
        b.step()
        worst_u = max(worst_u, float(np.max(np.abs(np.asarray(a.state) - np.asarray(b.state)))))
        worst_e = max(worst_e, abs(a.energy() - b.energy()))
    return a, b, worst_u, worst_e


# == the trajectories ==========================================================================


def test_the_modal_bridge_is_bit_identical():
    a, b, du, de = _run_pair(
        lambda: _modal(StringBodyBridgePy), lambda: _modal(physsynth_rs.StringBodyBridge), 2000
    )
    assert a.beta_s == b.beta_s and a.beta_b == b.beta_b and a.cfl_2dof == b.cfl_2dof
    assert a.spectral_radius == b.spectral_radius, "the dense leapfrog spectrum moved"
    assert du == 0.0 and de == 0.0, f"max|du| = {du:.3e}, max|dE| = {de:.3e}"
    assert np.array_equal(a.body.q, b.body.q), "the body's modal state diverged"


def test_the_shared_bridge_point_is_bit_identical():
    a, b, du, de = _run_pair(
        lambda: _sympathetic(SympatheticStringsPy),
        lambda: _sympathetic(physsynth_rs.SympatheticStrings),
        2000,
    )
    assert np.array_equal(np.asarray(a.beta_s), np.asarray(b.beta_s))
    assert np.array_equal(np.asarray(a._offsets), np.asarray(b._offsets))
    assert a.spectral_radius == b.spectral_radius
    assert du == 0.0 and de == 0.0, f"max|du| = {du:.3e}, max|dE| = {de:.3e}"
    for j in range(a.J):
        assert np.array_equal(a.strings[j].u, b.strings[j].u), f"string {j} diverged"


@pytest.mark.parametrize("boundary", BOUNDARIES)
def test_the_plate_bridge_is_bit_identical(boundary):
    a, b, du, de = _run_pair(
        lambda: _plate_bridge(StringPlateBridgePy, boundary),
        lambda: _plate_bridge(physsynth_rs.StringPlateBridge, boundary),
        400,
    )
    assert a.drive_index == b.drive_index
    assert a.stability_margin == b.stability_margin, (
        f"the Sherman-Morrison guard moved: {a.stability_margin!r} vs {b.stability_margin!r}"
    )
    assert du == 0.0 and de == 0.0, f"max|du| = {du:.3e}, max|dE| = {de:.3e}"
    assert np.array_equal(a.plate.u, b.plate.u), "the plate field diverged"
    assert a.pressure() == b.pressure()


@pytest.mark.parametrize("nonlinear", (True, False))
@pytest.mark.parametrize("boundary", BOUNDARIES)
def test_the_von_karman_bridge_is_bit_identical(boundary, nonlinear):
    """Exact, which §27.5 says is impossible for a nonlinear plate over any useful run.

    It is possible here for §32.5's reason: the two arms are not two discretizations, they are one
    Picard loop called through one factorization. That makes the exactness a sharp test of the
    transcription rather than a claim about the dynamics — and it means the sweep count is part of
    the comparison, so it is asserted too.
    """
    a, b, du, de = _run_pair(
        lambda: _vk_bridge(StringVKPlateBridgePy, boundary, nonlinear=nonlinear),
        lambda: _vk_bridge(physsynth_rs.StringVKPlateBridge, boundary, nonlinear=nonlinear),
        300,
    )
    assert a.stability_margin == b.stability_margin
    assert du == 0.0 and de == 0.0, f"max|du| = {du:.3e}, max|dE| = {de:.3e}"
    assert a.n_iters == b.n_iters and a.converged == b.converged
    assert a.last_residual == b.last_residual


# == the two anchors that made this one batch ==================================================


def test_one_string_reproduces_the_modal_bridge_in_both_languages():
    """§15.2's first anchor, re-asserted across the language boundary as well as within it.

    ``tests/test_sympathetic.py`` pins the same equality between two Python classes. Porting one
    of the pair alone would break it for a reason having nothing to do with the physics — so the
    interesting cell of the table is the mixed one, which no existing test can reach.
    """
    K = 8000.0
    arms = {
        ("py", "py"): (SympatheticStringsPy, StringBodyBridgePy),
        ("rs", "rs"): (physsynth_rs.SympatheticStrings, physsynth_rs.StringBodyBridge),
        ("rs", "py"): (physsynth_rs.SympatheticStrings, StringBodyBridgePy),
        ("py", "rs"): (SympatheticStringsPy, physsynth_rs.StringBodyBridge),
    }
    for label, (symp_cls, bridge_cls) in arms.items():
        symp = _sympathetic(symp_cls, J=1, K=[K])
        bridge = _modal(bridge_cls, K=K)
        for _ in range(400):
            symp.step()
            bridge.step()
        assert np.array_equal(symp.strings[0].u, bridge.string.u), f"{label}: string differs"
        assert np.array_equal(symp.body.q, bridge.body.q), f"{label}: body differs"
        assert symp.energy() == bridge.energy(), f"{label}: energy differs"


@pytest.mark.parametrize("boundary", BOUNDARIES)
def test_the_linear_von_karman_guard_matches_the_plate_guard_in_both_languages(boundary):
    """§15.2's second anchor: ``VKPlate(nonlinear=False)`` must give ``Plate``'s exact margin.

    The two classes reassemble ``G0`` out of the plate's own attributes, differing in one name
    (``rho`` against ``rho_s``, the reference's named 1000x trap). ``tests/test_airbox_vk.py``
    pins this equality; it is the reason the two classes had to move together, and it is asserted
    here in all four language combinations for the same reason as the anchor above.
    """
    fs = _fs()
    e = 1.0e-4
    rho_areal = 7800.0 * e

    def vk(cls):
        return cls(string=_string(), plate=_vk_plate(boundary, nonlinear=False, fs=fs), K=3000.0)

    def linear(cls):
        p = Plate(
            Lx=0.4,
            Ly=0.4,
            kappa=_vk_plate(boundary, nonlinear=False, fs=fs).kappa,
            rho=rho_areal,
            fs=fs,
            N=N_PLATE,
            boundary=boundary,
        )
        return cls(string=_string(), plate=p, K=3000.0)

    margins = {
        "py/py": (vk(StringVKPlateBridgePy).stability_margin, linear(StringPlateBridgePy)),
        "rs/rs": (
            vk(physsynth_rs.StringVKPlateBridge).stability_margin,
            linear(physsynth_rs.StringPlateBridge),
        ),
        "rs/py": (
            vk(physsynth_rs.StringVKPlateBridge).stability_margin,
            linear(StringPlateBridgePy),
        ),
    }
    for label, (vk_margin, lin) in margins.items():
        assert vk_margin == lin.stability_margin, (
            f"{label}: {vk_margin!r} vs {lin.stability_margin!r} -- the areal-density "
            "substitution or the guard's bracketing moved"
        )


# == the write through the string's buffer =====================================================


def test_the_reaction_is_written_through_the_live_buffer():
    """``string.u[-1] -= beta_s * F`` must mutate the array, not rebind it.

    The failure this catches is invisible to every physics bar: a port that read ``u`` into a
    vector, subtracted, and assigned the result back through a setter would produce exactly the
    same numbers *for itself* while a caller holding a reference to ``u`` — which is what the
    reference's semantics promise, and what ``lib.rs`` says the Python-owned buffers exist for —
    would see the un-reacted field. It is also impossible on a Python string, which has no setter.
    """
    for cls in (StringBodyBridgePy, physsynth_rs.StringBodyBridge):
        bridge = _modal(cls)
        for _ in range(200):  # let the wave reach the terminus so F is nonzero
            bridge.step()
        f = bridge.connection_force()
        assert abs(f) > 0.0, "no bridge force to test against"

        bridge.string.step()  # step the string by hand, then hold the array it produced
        held = bridge.string.u
        before = float(held[-1])
        bridge.string.u[-1] -= bridge.beta_s * f
        assert float(held[-1]) == before - bridge.beta_s * f, (
            f"{cls.__name__}: writing u[-1] did not reach the array object a caller holds"
        )


def test_the_step_s_reaction_reaches_a_held_reference():
    """The same property through ``step`` itself, which is where it actually matters."""
    bridge = _modal(physsynth_rs.StringBodyBridge)
    twin = _modal(StringBodyBridgePy)
    for _ in range(200):
        bridge.step()
        twin.step()
    held_rs = bridge.string.u
    held_py = twin.string.u
    bridge.step()
    twin.step()
    # Both implementations rebind `u` inside the string's own step, so the held arrays are the
    # PREVIOUS field -- and both must agree about that, including the reaction subtracted from the
    # step before.
    assert np.array_equal(np.asarray(held_rs), np.asarray(held_py))
    assert np.array_equal(np.asarray(bridge.string.u_prev), np.asarray(held_rs))


# == which reductions were transcribed, and which were not =====================================


def _left_to_right(values):
    total = 0.0
    for v in values:
        total += float(v)
    return total


def test_the_modal_inverse_mass_sum_is_transcribed_above_eight_modes():
    """``beta_b = k^2 sum_i phi_i^2 / m_i`` is ``reduce::sum``, and no shipped fixture can tell.

    §30.2's cutoff makes this decidable by counting: below **eight** terms ``np.sum`` is a
    left-to-right loop whatever the values, and every body in this repo has four or five modes. So
    the transcription is unexercised by the whole suite, and the fixture here is *searched* rather
    than picked (§26.6) — a hand-picked one lands in the agreeing majority and the test goes green
    having compared two spellings that happen to coincide.
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
        py = _modal(StringBodyBridgePy, phi=phi, masses=masses, freqs=freqs, K=2000.0)
        rs = _modal(physsynth_rs.StringBodyBridge, phi=phi, masses=masses, freqs=freqs, K=2000.0)
        assert py.beta_b == rs.beta_b, "the Rust sum is not NumPy's pairwise blocking"
        naive = py.k * py.k * _left_to_right(phi * phi / masses)
        assert naive != py.beta_b, (
            "the witness search found a fixture the reference itself cannot distinguish"
        )
        print(f"beta_b at the witness: {py.beta_b!r}; a left-to-right loop gives {naive!r}")


def test_the_shared_bridge_force_sum_is_transcribed_above_eight_strings():
    """``body.step(force=sum_j F_j)`` reaches the next timestep, so this sum must be exact.

    §14.2's question answered *yes*, and §31.2 says exactness is available. Same blind spot as the
    test above, from the other end: every sympathetic rig in the repo has two or three strings.
    """
    j_strings = 8
    py = _sympathetic(SympatheticStringsPy, j_strings)
    rs = _sympathetic(physsynth_rs.SympatheticStrings, j_strings)
    differing = 0
    for _ in range(600):
        forces = np.asarray(py.connection_forces())
        differing += _left_to_right(forces) != float(np.sum(forces))
        py.step()
        rs.step()
    assert differing > 0, (
        f"J={j_strings} never put np.sum and a left-to-right loop apart -- this test would then "
        "pass on a port that transcribed the sum wrongly"
    )
    print(f"sum(forces) differs from a left-to-right loop on {differing}/600 steps at J=8")
    for j in range(j_strings):
        assert np.array_equal(py.strings[j].u, rs.strings[j].u), f"string {j} diverged"
    assert np.array_equal(py.body.q, rs.body.q)


def test_the_two_dot_products_are_not_transcribed():
    """``np.dot(phi, q)`` stays a BLAS call — ``ddot`` fuses its multiply-add (§14.2).

    The fixture matters and is §14.3's finding applied here: every body in ``tests/helpers.py`` has
    ``phi = 1.0``, and a fused multiply-add differs from a rounded one only when the *product*
    rounds — so with unit weights the suite is systematically blind to this whole class. The
    weights below are deliberately not powers of two.

    **Whether BLAS separates them is reported, never required** — corrected 2026-09-02, the
    precedent being ``test_rust_parity_radiation.py``'s ``part_company`` test, which went through
    the same correction a phase earlier. The first draft asserted ``differing > 0`` and was red on
    ``main`` for four CI runs: on GitHub's AMD EPYC 7763 OpenBLAS's ``ddot`` kernel agreed with a
    left-to-right multiply-add on all 800 steps, while it disagreed on the Windows box this was
    written on. That is §14.2's own finding — OpenBLAS picks its kernel by CPU — turned against
    the test written to illustrate it: a negative control whose predicate is a per-CPU kernel is a
    claim about the runner, so it prints its verdict and asserts only what the port promises,
    which is that ``py`` and ``rs`` agree to the bit *whatever* the BLAS did.
    """
    rng = np.random.default_rng(4711)
    phi = rng.uniform(0.4, 1.8, len(BODY_FREQS))
    py = _modal(StringBodyBridgePy, phi=phi)
    rs = _modal(physsynth_rs.StringBodyBridge, phi=phi)
    differing = 0
    for _ in range(800):
        q_prev = np.asarray(py.body.q_prev)
        if _left_to_right(phi * q_prev) != float(np.dot(phi, q_prev)):
            differing += 1
        py.step()
        rs.step()
    verdict = f"differ on {differing}/800 steps" if differing else "agree on this BLAS"
    print(f"np.dot vs a left-to-right multiply-add: {verdict}")
    assert np.array_equal(py.string.u, rs.string.u)
    assert py.energy() == rs.energy()


# == the duck typing ===========================================================================


class _NotAModalBody:
    """A body-shaped object that is not a ``ModalBody`` in either language.

    Every airbox and radiation test hands the bridge a real model, so a port that downcast its
    collaborator would pass all of them. This one delegates by hand, which is exactly what the
    reference's pure duck typing (§31.11) promises will work.
    """

    def __init__(self, inner):
        self._inner = inner
        self.steps = 0

    @property
    def k(self):
        return self._inner.k

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


def test_the_bridge_never_looks_at_its_collaborator_s_type():
    py = StringBodyBridgePy(string=_string(), body=_NotAModalBody(_body()), K=8000.0)
    rs = physsynth_rs.StringBodyBridge(
        string=_string(), body=_NotAModalBody(_body()), K=8000.0
    )
    assert py.beta_b == rs.beta_b and py.spectral_radius == rs.spectral_radius
    for _ in range(300):
        py.step()
        rs.step()
    assert rs.body.steps == 300, "the port did not drive the stand-in through its own `step`"
    assert np.array_equal(py.string.u, rs.string.u)
    assert py.energy() == rs.energy()
    assert py.pressure() == rs.pressure()


# == the solver is a module global, not a captured binding =====================================


def test_the_guard_reads_its_solver_from_the_module_at_call_time(monkeypatch):
    """``_stability_margin`` calls ``connection.splu`` and ``connection.spsolve`` by name.

    That is the faithful transcription (§32.7) and it is what makes §24.4's shared-factorization
    manoeuvre available here: replace the name and both languages' guards change together, which
    is how a future Group D question about this file would be asked.
    """
    calls = {"splu": 0, "spsolve": 0}
    real_splu, real_spsolve = conn.splu, conn.spsolve

    def counting_splu(a, *args, **kw):
        calls["splu"] += 1
        return real_splu(a, *args, **kw)

    def counting_spsolve(a, b, *args, **kw):
        calls["spsolve"] += 1
        return real_spsolve(a, b, *args, **kw)

    monkeypatch.setattr(conn, "splu", counting_splu)
    monkeypatch.setattr(conn, "spsolve", counting_spsolve)
    rs = _plate_bridge(physsynth_rs.StringPlateBridge)
    assert calls == {"splu": 1, "spsolve": 1}, (
        "the Rust guard did not go through `connection`'s own module globals -- a captured "
        "binding, and §24.4's manoeuvre is unavailable on this file"
    )
    monkeypatch.undo()
    assert rs.stability_margin == _plate_bridge(StringPlateBridgePy).stability_margin


# == §33.2's write question, aimed at the class being ported ===================================


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
def test_every_attribute_the_reference_assigns_stays_writable(name):
    """A ``#[getter]`` with no ``#[setter]`` is a data descriptor whose ``__set__`` raises (§33.2).

    So porting a class decides not only which names can be read through it but which can be
    *written*, and the default is none — where the reference, being Python, allowed every one. The
    list is the reference's own ``self.X = ...`` set, read off the source rather than guessed.
    """
    built = {
        "StringBodyBridge": lambda: _modal(physsynth_rs.StringBodyBridge),
        "StringPlateBridge": lambda: _plate_bridge(physsynth_rs.StringPlateBridge),
        "StringVKPlateBridge": lambda: _vk_bridge(physsynth_rs.StringVKPlateBridge),
        "SympatheticStrings": lambda: _sympathetic(physsynth_rs.SympatheticStrings),
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
    rs = _plate_bridge(physsynth_rs.StringPlateBridge)
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


# == the refusals and the signatures ===========================================================


def test_the_constructors_are_keyword_only():
    for cls, args in (
        (physsynth_rs.StringBodyBridge, (_string(), _body(), 1.0)),
        (physsynth_rs.StringPlateBridge, (_string(), _plate(), 1.0)),
        (physsynth_rs.StringVKPlateBridge, (_string(), _vk_plate(), 1.0)),
        (physsynth_rs.SympatheticStrings, ([_string()], _body(), [1.0])),
    ):
        with pytest.raises(TypeError):
            cls(*args)


@pytest.mark.parametrize(
    "build, message",
    [
        (
            lambda cls: cls(string=_string(), body=_body(fs=_fs() * 1.1), K=1.0),
            "string and body must share a timestep",
        ),
        (
            lambda cls: cls(
                string=IdealString(
                    L=L_STRING, T=T_STRING, rho=RHO_STRING, fs=_fs(), N=N_STRING,
                    boundary=("fixed", "fixed"),
                ),
                body=_body(),
                K=1.0,
            ),
            "right end must be 'free'",
        ),
        (
            lambda cls: cls(string=_string(), body=_body(), K=-1.0),
            "bridge stiffness K must be >= 0",
        ),
        (
            lambda cls: cls(string=_string(), body=_body(), K=1e7),
            "connection unstable",
        ),
    ],
)
def test_the_modal_refusals_reproduce_verbatim(build, message):
    with pytest.raises(ValueError) as py_err:
        build(StringBodyBridgePy)
    with pytest.raises(ValueError) as rs_err:
        build(physsynth_rs.StringBodyBridge)
    assert message in str(py_err.value)
    assert str(rs_err.value) == str(py_err.value), (
        f"message text diverged:\n  py: {py_err.value}\n  rs: {rs_err.value}"
    )


@pytest.mark.parametrize(
    "build, message",
    [
        (
            lambda cls: cls(string=_string(), plate=_plate(fs=_fs() * 1.1), K=1.0),
            "string and plate must share a timestep",
        ),
        (
            lambda cls: cls(string=_string(N=N_STRING), plate=_plate(), K=1.0, drive_index=10**6),
            "out of range",
        ),
        (
            lambda cls: cls(string=_string(), plate=_plate(), K=1e9),
            "connection unstable",
        ),
    ],
)
def test_the_plate_refusals_reproduce_verbatim(build, message):
    with pytest.raises(ValueError) as py_err:
        build(StringPlateBridgePy)
    with pytest.raises(ValueError) as rs_err:
        build(physsynth_rs.StringPlateBridge)
    assert message in str(py_err.value)
    assert str(rs_err.value) == str(py_err.value), (
        f"message text diverged:\n  py: {py_err.value}\n  rs: {rs_err.value}"
    )


@pytest.mark.parametrize(
    "build, message",
    [
        (lambda cls: cls(strings=[], body=_body(), Ks=[]), "at least one string"),
        (
            lambda cls: cls(strings=[_string(), _string()], body=_body(), Ks=[1.0]),
            "one stiffness per string",
        ),
        (
            lambda cls: cls(strings=[_string()], body=_body(), Ks=[-1.0]),
            "every bridge stiffness K must be >= 0",
        ),
        (
            lambda cls: cls(strings=[_string()], body=_body(fs=_fs() * 1.1), Ks=[1.0]),
            "must share a timestep",
        ),
    ],
)
def test_the_sympathetic_refusals_reproduce_verbatim(build, message):
    with pytest.raises(ValueError) as py_err:
        build(SympatheticStringsPy)
    with pytest.raises(ValueError) as rs_err:
        build(physsynth_rs.SympatheticStrings)
    assert message in str(py_err.value)
    assert str(rs_err.value) == str(py_err.value), (
        f"message text diverged:\n  py: {py_err.value}\n  rs: {rs_err.value}"
    )


def test_an_explicit_none_drive_index_is_the_omitted_one():
    """§24.7 and §31.7: with ``Option<Option<_>>``, ``Some(None)`` is the omitted keyword and a
    bare ``None`` is the caller's literal — and getting the arms backwards is silent, because here
    the reference means the same thing by both."""
    for cls in (StringPlateBridgePy, physsynth_rs.StringPlateBridge):
        omitted = cls(string=_string(), plate=_plate(), K=3000.0)
        explicit = cls(string=_string(), plate=_plate(), K=3000.0, drive_index=None)
        assert omitted.drive_index == explicit.drive_index
        assert omitted.stability_margin == explicit.stability_margin
