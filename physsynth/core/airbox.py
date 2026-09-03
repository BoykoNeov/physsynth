"""The 3-D air box — a room the sound actually crosses (HANDOFF §12.H).

The **distributed** tier of the ``exciter -> resonator -> body/radiation`` abstraction's air node.
:mod:`physsynth.core.radiation` models the air as a **lumped port**: one volume-velocity terminal,
one impedance, one listening distance. That is enough for "how loud, how damped, how far", and
structurally cannot represent anything that depends on *where you are* — room modes, the finite
travel time across a space, the comb filtering of a direct sound against its reflections, more than
one listener, or any source that is not compact. This module is that missing dimension (all three
of them), and its headline claim is a **cross-tier** one:

    the distributed air *contains* the lumped air as its free-field limit.

Put the walls far enough away, drive an :class:`AirBox` and an
:class:`~physsynth.core.radiation.AirRadiation` from the same source, and inside the window before
the first reflection arrives the box reproduces the monopole law
``p = rho0 Q''(t - r/c0) / (4 pi r)`` — measured, not asserted. Then bring the walls in and get
everything the lumped tier cannot.

**The physics — the bore's Yee cell, one dimension up.** Same two first-order conservation laws as
:mod:`physsynth.core.bore`, now with a vector velocity and no area profile::

    rho0 du/dt = -grad p           (momentum / Euler)
    dp/dt      = -rho0 c0^2 div u  (continuity / mass)

**State layout — node-centered pressure, face-centered velocity.** ``p`` lives at the
``(Nx+1)(Ny+1)(Nz+1)`` grid nodes *including the walls*; ``u_x`` on the ``Nx(Ny+1)(Nz+1)`` faces
between neighbouring nodes in ``x`` (and likewise ``u_y``, ``u_z``), half a timestep offset in time.
This is the bore's layout tensored up, chosen over the cell-centered convention common in room
acoustics for one reason: **the bore's boundary machinery transfers unchanged.** A rigid wall is the
``h/2`` half-cell trapezoidal node weight and needs no ghost stencil; an impedance wall is the
bore's radiating-end 1x1 collapse. Both are already proven in this repo. Uniform spacing ``h`` in
all three directions is **required**, not a convenience: an anisotropic grid breaks the isotropic
CFL and the tensor-cosine exactness the modal oracle rests on.

**Leapfrog, with the adjoint pairing that makes the bore conserve**::

    u_x^{n+1/2} = u_x^{n-1/2} - (k / (rho0 h)) (p_{i+1} - p_i)^n      [and y, z]
    p^{n+1}     = p^n - k rho0 c0^2 div(u^{n+1/2})

``lambda = c0 k / h <= 1/sqrt(3)`` (the 3-D CFL); asserted at construction, rejected otherwise.
Unlike the 1-D string there is **no dispersionless lambda** — the fact the membrane batch recorded
for 2-D, one dimension worse. Do not tune toward the ceiling as if it were the string's
``lambda = 1``, because at ``lambda = 1/sqrt(3)`` exactly the reward and the price arrive together:

* **The reward.** A mode whose wavevector lies along the grid **diagonal** — meaning
  ``l/Nx = m/Ny = n/Nz``, which on a cube is just ``l = m = n`` — comes out at the *exact continuum*
  frequency (measured to 1e-16). Nothing else is exact at any ``lambda``; axis-aligned propagation
  stays dispersive by ~0.4% per index everywhere.
* **The price.** The corner mode's dispersion argument ``lambda sqrt(3)`` reaches exactly 1, i.e.
  ``omega_d k = pi``, and there the leapfrog's amplification matrix is **defective**. Broadband
  content then grows *linearly* in time (measured: peak amplitude x15 by 1000 steps, x88 by 6000,
  with successive ratios tracking ``1 + 1/n`` — secular, not exponential). The energy identity
  survives this untouched, because the discrete energy is only positive **semi**-definite at the
  ceiling: the potential and kinetic terms grow together with cancelling signs, so ``E`` stays flat
  while the field does not. **A flat energy is therefore not a stability certificate at
  ``lambda = lambda_max``** — the one place in this repo where that is true.

So ``lambda = 1/sqrt(3)`` is *marginally* stable, and construction allows it deliberately (it is
the textbook CFL, and the diagonal exactness is a genuine reason to want it). For any run with
broadband content, stay strictly below: ``0.9/sqrt(3)`` is the default the test helpers use.

**Energy — the cross-time product, in 3-D.** With the tensor-trapezoid node weight
``W_ijk = wx_i wy_j wz_k`` (``w = h`` interior, ``h/2`` at a wall, so an edge node carries
``h (h/2) (h/2)`` and a corner ``h^3/8``)::

    E^n = 1/2 sum_nodes (W / (rho0 c0^2)) (p^n)^2
        + 1/2 rho0 sum_faces W_face u^{n+1/2} u^{n-1/2}

The velocity term is the **cross-time product** of the staggered variable, never the same-time
square — the string/bore lesson, and the reason a lossless run conserves to machine precision
instead of oscillating. It telescopes to exactly zero change per step because the divergence uses
the transpose of the gradient the momentum uses, and the weights cancel in the pairing.

**One wall closure, three boundary types.** At a wall node the outward normal velocity is
``u_n = pbar / Z`` with ``pbar = (p^{n+1} + p^n)/2`` — **centered**, i.e. implicit, the standing
VK/bow/radiation-load lesson. Substituted into that node's divergence it collapses to a scalar::

    p^{n+1} = (p_rigid - beta p^n) / (1 + beta),    beta = k rho0 c0^2 / (2 Z w_wall)

where ``p_rigid`` is the force-free rigid-wall update and ``w_wall = h/2``. A node on an **edge or
corner touches two or three walls and simply sums their beta** (admittances add) — still 1x1, no
coupled solve anywhere. The two ideal ends fall out of the same line, and are this batch's entry in
the family's reduction ledger (alongside ``R=0`` -> bare body, ``M_a=inf`` -> ``RadiatedBody``,
``sigma_1=0`` -> model #2, ``nonlinear=False`` -> #5):

    ``Z = inf``  =>  ``beta = 0``    =>  exactly the rigid update (bit-identical) ;
    ``Z = 0``    =>  ``beta = inf``  =>  ``p^{n+1} = 0``, the open (pressure-release) face.

The same telescoping then leaves exactly the wall flux,
``E^{n+1} - E^n = -k sum_wallnodes A_node pbar^2 / Z <= 0``, so ``E`` alone is monotone and
``E + dissipated`` is flat. Note that a locally-reacting ``Z = rho0 c0`` wall is matched at **normal
incidence only** — obliquely it reflects. It is a passive, closed-form-checkable absorber, not an
anechoic chamber; the free-field read-out is protected by *windowing*, never by absorption.

**Source.** A **soft** (transparent) point source of volume velocity ``q`` (m^3/s), the continuity
equation's own source term: ``p_src += k rho0 c0^2 q / W_src`` folded into the pressure update. It
injects ``integral p q dt``, booked on :meth:`injected_energy`, so the conserved statement is
``E + dissipated - injected = const`` — what :meth:`energy` returns. A *hard* source (assigning
``p``) would not be passive and is not offered.

**The room pushes back (batch 2): the port.** :meth:`inject` is a *read-out* source — it drives the
room and the room does not load it, exactly the position
:class:`~physsynth.core.radiation.AirRadiation` occupied in radiation batch 1. :class:`RoomPort` and
:class:`RoomLoadedBody` close that loop. Within one step an injection changes the pressure at
its own nodes and nowhere else (propagation waits for the next momentum sub-step), so the room
seen from a port is a **Thevenin source**, ``pbar = pbar_free + R_room q``: a known open-circuit
pressure carrying the room's whole history *including every reflection*, in series with one
positive constant. The body's rank-1 solve then closes in a **single division**, and because
``1 + G R_room >= 1`` it is unconditionally passive — no CFL of its own, no stability guard. The
payoff is the thing no lumped ``R(omega)`` can produce at any order: a **delayed echo** loading the
body, rather than a decaying exponential. Still deferred: PML, HRTF/ambisonics, scattering objects
and non-rectangular geometry, viscothermal air absorption, moving ports, overlapping ports, and the
distributed *area* coupling (a body radiating from every node rather than through one port).

**Cost is a design constraint here, for the first time in this repo.** At ``fs = 44.1 kHz`` the CFL
forces ``h >= sqrt(3) c0 / fs ~ 1.35 cm``, so a 1 m^3 room is ~74^3 nodes and one second of audio is
44100 steps — minutes, not seconds. The modal oracles are rate-independent and run on ~10x8x6 grids
where they are free; genuinely audio-rate rooms belong in a diagnose script, not a test suite.

Headless: NumPy only.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

import numpy as np
import physsynth_rs as _rs
from numpy.typing import NDArray
from scipy import sparse

if TYPE_CHECKING:  # type-only: the room stays free of any dependency on the resonator modules
    from scipy.sparse.linalg import SuperLU

    from physsynth.core.body import ModalBody
    from physsynth.core.membrane import Membrane
    from physsynth.core.plate import Plate, VKPlate

__all__ = [
    "AirBox",
    "FACES",
    "PLANES",
    "RHO0_AIR",
    "C0_AIR",
    "impedance_from_zeta",
    "RoomPort",
    "RoomLoadedBody",
    "SurfacePort",
    "RoomLoadedPlate",
    "InteriorSurfacePort",
    "RoomSuspendedPlate",
    "RoomLoadedMembrane",
    "RoomSuspendedMembrane",
    "RoomLoadedVKPlate",
    "RoomSuspendedVKPlate",
]

# Ambient air (matches physsynth.core.radiation and .bore so every tier of the air node agrees).
RHO0_AIR = 1.2041  # kg/m^3
C0_AIR = 343.0     # m/s

# The six faces of the box, named <axis><end>: "x0" is the x = 0 wall, "x1" the x = Lx wall.
FACES = ("x0", "x1", "y0", "y1", "z0", "z1")

# The three interior plane orientations, named by their normal axis (see AirBox.add_cut). A wall is
# a FACE and has an end; an interior plane has neither, which is why the two vocabularies differ.
PLANES = ("x", "y", "z")

# How a surface node's area is handed to the air grid (see `SurfacePort`).
Spreading = Literal["bilinear", "nearest"]

# The room and the three ports are Rust, on every path. Imported here rather than reached through
# `physsynth_rs` at each use site because they are this module's public names -- `__all__` above
# lists four of them, `web/serialize.py` imports two, and `crates/physsynth-py/src/airbox_wrap.rs`
# reads `RoomPort`, `SurfacePort` and `InteriorSurfacePort` off this module's namespace at call
# time (plan section 46.7, findings ledger #39: the binding depends on Python).
#
# `_face_axes` and `_free_pressure_nodes` are NOT re-exported. They were module-level helpers the
# Python ports shared, and the only things that reached them through this module were the parity
# tests that go with this deletion -- so they fail section 41.2's third category ("reached through
# this module") rather than satisfying it. `physsynth_rs` still exports both.
from physsynth_rs import (  # noqa: E402
    AirBox,
    InteriorSurfacePort,
    RoomPort,
    SurfacePort,
    impedance_from_zeta,
)


class RoomLoadedBody:
    """A :class:`~physsynth.core.body.ModalBody` **loaded by a room** — the two-way port (batch 2).

    :class:`~physsynth.core.radiation.RadiatedBody` loads a body with a constant resistance and
    :class:`~physsynth.core.radiation.RationalAirLoad` with an exact first-order impedance. Both are
    *one-ports with no memory of geometry*, so their impulse response is a decaying exponential and
    they structurally cannot represent the thing a room does: **give energy back, at a delay, from a
    direction**. A body in a small hard room feels its own reflected wave arriving ``2d/c0`` later
    and is loaded by it — not reverberation added to a dry signal, but a change to the body's own
    oscillation. That is what this class adds, and no ``R(omega)`` reproduces it at any order.

    **The solve is one division.** The body's volume velocity is ``U = sum_i a_i q_i'`` and the port
    pressure ``pbar`` enters mode ``i`` as ``-a_i pbar`` (the *same* weights, by reciprocity — the
    fact :class:`~physsynth.core.radiation.RadiatedBody` already relies on). Take the force-free
    step, read its free centered ``U_free``, ask the port for its two numbers (:class:`RoomPort`),
    and substitute::

        G = (k/2) sum_i a_i^2 / (m_i (1 + sigma_i k))
        U = (U_free - G pbar_free) / (1 + G R_room)
        pbar = pbar_free + R_room U
        q_i^{n+1} = q~_i^{n+1} - pbar k^2 a_i / (m_i (1 + sigma_i k))

    Since ``G >= 0`` and ``R_room >= 0``, ``1 + G R_room >= 1``: the solve can never be singular at
    any sample rate, any grid, any body. **The port is unconditionally passive — no CFL of its own,
    no stability guard**, exactly like ``RadiatedBody`` and unlike the bridge springs, whose rank-1
    block can go negative. The room's own CFL and the body's are unchanged; coupling them adds no
    third condition.

    **The ledgers cancel identically.** The corrected state's centered volume velocity is exactly
    the ``U`` solved for, so the body's energy decrement telescopes to precisely ``k pbar U`` —
    which is :attr:`radiated_energy`. The room books the *same* number into its ``injected``, from
    its own post-closure pressure, because the injection weights and the read-back weights are the
    same vector. So for a scene::

        sum_j inst_j.energy() + room.energy()

    is conserved to machine precision, and the coupling term **cancels out of the statement
    entirely** — which is why a drift in it is unambiguous evidence of a bug rather than of
    accounting. (Assert on that total, not on ``body.energy()``, which is not monotone: the port's
    near-field reactance hands energy back every cycle.)

    Being a drop-in for ``ModalBody`` (it delegates every read accessor) it slots straight into a
    :class:`~physsynth.core.connection.StringBodyBridge` as the body, giving the full
    ``string -> bridge -> body -> room`` chain with no edit to ``connection.py``.

    **This class does not step the room, and that is deliberate.** ::

        for n in range(n_steps):
            inst_a.step(force)     # or bridge_a.step(), which owns the body's step
            inst_b.step(force)
            room.step()            # one room, one step, after every port has solved

    Had ``step()`` advanced the room, two instruments would step it twice per sample and a
    string-driven instrument (where the *bridge* owns ``body.step``) could not be a member at all.
    The cost is one line in the caller's loop, and forgetting it raises (:meth:`RoomPort.inject`)
    rather than silently freezing the room.

    Parameters
    ----------
    body : ModalBody
        The body to load. Its radiation weights ``body.a`` set the volume-velocity coupling; ``a =
        0`` decouples the room and is **bit-identical** to the bare body. Use ``sigmas = 0`` to
        isolate the room channel in the energy identity.
    room : AirBox
        The room to radiate into. Its sample rate must match the body's.
    at, radius
        Passed to :class:`RoomPort` — the port centre (m) and ball radius (m), or ``radius=None``
        for a point port. See that class on why the choice is required rather than defaulted.

    Raises
    ------
    ValueError
        If ``room.fs`` does not match ``body.fs``, or on any of :class:`RoomPort`'s refusals.
    """

    def __init__(
        self,
        *,
        body: ModalBody,
        room: AirBox,
        at: Sequence[float],
        radius: float | None,
    ) -> None:
        self.body = body  # FIRST: any attribute miss before this makes __getattr__ recurse
        if not np.isclose(body.k, room.k, rtol=1e-12, atol=0.0):
            raise ValueError(
                f"sample-rate mismatch: body fs = {body.fs} but room fs = {room.fs}. The port's "
                "solve is a single timestep shared by both, so they must agree exactly."
            )
        self.room = room
        self.port = RoomPort(room=room, at=at, radius=radius)
        self.k = body.k
        # Rank-1 precomputes, from PUBLIC body attributes (RadiatedBody's _G is its own business).
        # The (1 + sigma_i k) carries the body's implicit damping denominator into the load.
        one_plus_sk = 1.0 + body.sigma * body.k
        self._G = 0.5 * body.k * float(np.sum(body.a * body.a / (body.m * one_plus_sk)))
        self._corr = body.k * body.k * body.a / (body.m * one_plus_sk)
        self.radiated_energy = 0.0  # integral pbar U dt: the work this body did on the room
        self.volume_velocity = 0.0  # last centered U (diagnostic)
        self.port_pressure = 0.0    # last centered pbar the body was loaded by
        self.n = 0

    def __getattr__(self, name: str):
        # Delegate read-only body accessors (phi, m, omega, M, q, q_prev, state, bridge_*, ...) so a
        # RoomLoadedBody is a drop-in wherever a bare ModalBody is expected (e.g. StringBodyBridge).
        # Only reached for names not set on the instance, so the overrides below always win.
        if name == "body":  # nothing to delegate through yet -- never recurse
            raise AttributeError(name)
        return getattr(self.body, name)

    # -- time stepping ---------------------------------------------------------------------

    def step(self, force: float = 0.0) -> None:
        """Advance one step: read the port, advance the body, solve the load, queue the injection.

        ``force`` is the optional external (bridge) force, forwarded to
        :meth:`~physsynth.core.body.ModalBody.step`; the room's load is applied on top of it. The
        room is **not** stepped here — see the class docstring.
        """
        b, port = self.body, self.port
        port.require_ready()                           # before mutating anything (§ the guard)
        pbar_free = port.free_pressure()               # read u^{n+1/2}, BEFORE room.step()
        q_nm1 = b.q_prev.copy()                        # q^{n-1}, before step() rolls history
        b.step(force)                                  # commit the force-free next state q~^{n+1}
        u_free = float(np.dot(b.a, b.q - q_nm1)) / (2.0 * self.k)
        u = (u_free - self._G * pbar_free) / (1.0 + self._G * port.R_room)
        pbar = pbar_free + port.R_room * u
        b.q = b.q - pbar * self._corr                  # rank-1 correction of q^{n+1}
        # Refresh q'' from the *corrected* second difference so pressure() carries the load (the
        # same reason ModalBody.pressure reads the true _accel rather than reconstructing it).
        b._accel = (b.q - 2.0 * b.q_prev + q_nm1) / (self.k * self.k)
        port.inject(u)
        self.radiated_energy += self.k * pbar * u
        self.volume_velocity = u
        self.port_pressure = pbar
        self.n += 1

    # -- diagnostics -----------------------------------------------------------------------

    def energy(self) -> float:
        """Total discrete energy ``E_body + integral pbar U dt`` (Joules).

        **An explicit override, not a delegation** — ``__getattr__`` would otherwise hand back the
        bare modal energy, i.e. the total *without* its coupling channel, which is exactly the
        number that looks fine and is not conserved. Unlike
        :meth:`~physsynth.core.radiation.RadiatedBody.energy` this one is **not** monotone: the room
        gives energy back, so :attr:`radiated_energy` can decrease. The conserved statement is the
        whole scene, ``sum_j inst_j.energy() + room.energy()``.
        """
        return self.body.energy() + self.radiated_energy

    def pressure(self) -> float:
        """Radiated pressure read-out ``sum_i a_i q_i''``, reflecting the room load.

        Delegates to :meth:`~physsynth.core.body.ModalBody.pressure`, whose ``_accel`` this class
        refreshes *after* the rank-1 correction. For the pressure **in the room**, read the field:
        :meth:`AirBox.pressure_at`.
        """
        return self.body.pressure()

    def set_state(
        self,
        q0: NDArray[np.float64] | float,
        v0: NDArray[np.float64] | float = 0.0,
    ) -> None:
        """Set the body's initial modal state and reset this port's coupling ledger to zero."""
        self.body.set_state(q0, v0)
        self._reset_books()

    def reset(self) -> None:
        """Zero the body state and the coupling ledger — reuse on a new run.

        The room is a separate object with its own :meth:`AirBox.set_state`.
        """
        self.body.set_state(0.0)
        self._reset_books()

    def _reset_books(self) -> None:
        self.radiated_energy = 0.0
        self.volume_velocity = 0.0
        self.port_pressure = 0.0
        self.port.reset()
        self.n = 0


def _require_same_rate(model, room: AirBox, label: str) -> None:
    """Refuse a resonator whose timestep is not the room's, naming it as the caller knows it.

    The port's Thevenin solve is a **single timestep shared by both** — there is no resampling
    anywhere in the coupling — so this is a refusal, not a warning. ``label`` is the word the
    message uses for the model ("plate", "membrane"): an error naming the wrong model is worse
    than none.
    """
    if not np.isclose(model.k, room.k, rtol=1e-12, atol=0.0):
        raise ValueError(
            f"sample-rate mismatch: {label} fs = {model.fs} but room fs = {room.fs}. The port's "
            "solve is a single timestep shared by both, so they must agree exactly."
        )


class _PlateSurface:
    """The seam between a room port and the grid resonator it loads — the :class:`Plate` side.

    Batches 3 and 4 each reassemble the plate's system matrix and its theta-scheme RHS, in two
    boundary branches: four copies of one piece of arithmetic, and every further loaded resonator
    doubles them again. This is that arithmetic, extracted once, as six members — everything a
    loaded surface has to provide and nothing about the room:

    ==================  ====================================================================
    ``surface()``       ``(coords, areas)`` — the live nodes the port spreads onto, in the
                        model's own C-order over ``mask``, so ``T``'s columns line up with
                        the state vector by construction
    ``denominator``     the per-node mass ``rho_s`` that the load and ``f_ext`` divide by
    ``a_bare()``        the **unloaded** system matrix, before ``(k/2 rho_s) T^T R T``
    ``u_prev``          ``u^{n-1}`` — read once per step, **before** :meth:`commit`
    ``rhs(f_ext)``      the force-free RHS plus the ``f_ext`` path: everything the room does
                        not touch
    ``commit(u_next)``  roll the history and refresh whatever the model caches
    ==================  ====================================================================

    The extraction lives **inside this module**, never in ``plate.py``: the room may depend on the
    resonators and not the reverse (the headless-core rule), and "``plate.py`` untouched" is a
    claim batches 3 and 4 both make and keep.

    **``u_prev`` is a live read, not a snapshot, and the caller must take it before**
    :meth:`commit`. The wrapper needs ``u^{n-1}`` in three places — inside :meth:`rhs`, in the
    load's ``(k/2 rho_s) (T^T R T) u^{n-1}`` term, and in the centered velocity
    ``(u^{n+1} - u^{n-1}) / 2k`` — and after :meth:`commit` this property returns ``u^n`` instead.
    Reading it late is the one way this extraction can go wrong and still look plausible: the
    scheme stays stable and each ledger still telescopes against the pressure it used, so only the
    physics is off by one time level and nothing green turns red.

    **Not a base class, deliberately.** There is one implementation today; a second arrives with
    its first caller, and what the two genuinely share is better measured then than guessed now.
    :meth:`commit`'s ``_accel`` write in particular is the *plate's*, not the seam's — a two-level
    roll that caches no acceleration would not want it.
    """

    def __init__(self, plate: Plate) -> None:
        self.model = plate
        self.k = plate.k
        if plate.boundary == "supported":
            self.areas = np.full(plate.n_live, plate.h * plate.h)
            # Per-node mass rho_s h^2: Plate.step divides f_ext by exactly this.
            self.denominator = plate.rho * plate.h * plate.h
        else:
            self.areas = plate.w.copy()  # lumped cell areas (h^2, h^2/2, h^2/4) -- no dead rim
            self.denominator = plate.rho  # W lives inside A and is divided out by the solve

    def surface(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """``(coords, areas)`` — live-node positions (m) and their areas (m^2), in model order."""
        p = self.model
        coords = np.column_stack((p.X[p.mask], p.Y[p.mask]))
        return coords, self.areas

    def a_bare(self) -> sparse.spmatrix:
        """The unloaded system matrix: ``(1 + sigma k) I + theta k^2 kappa^2 B``, or its ``W`` form.

        Reassembled here rather than reached into, because ``plate.py`` stays untouched — which is
        exactly why the coupled cross-check at two timesteps exists to pin it.
        """
        p = self.model
        sk = p.sigma * p.k
        coeff = p.theta * p.k * p.k * p.kappa * p.kappa
        if p.boundary == "supported":
            return (1.0 + sk) * sparse.identity(p.n_live, format="csc") + coeff * p.B
        return (1.0 + sk) * p.W + coeff * p.K

    @property
    def u_prev(self) -> NDArray[np.float64]:
        """``u^{n-1}`` — read once per step, before :meth:`commit` (see the class docstring)."""
        return self.model.u_prev

    def rhs(self, f_ext: NDArray[np.float64] | None) -> NDArray[np.float64]:
        """The force-free theta-scheme RHS plus the ``f_ext`` path — the room's terms are not here.

        Byte-for-byte :meth:`~physsynth.core.plate.Plate.step`'s own arithmetic, in its own operand
        order, so a zero air load is a clean reduction to the bare plate and a transcription slip
        would show up against the model itself.
        """
        p = self.model
        sk = p.sigma * self.k
        k2 = self.k * self.k
        kappa2 = p.kappa * p.kappa
        u_nm1 = p.u_prev
        if p.boundary == "supported":
            rhs = (
                2.0 * p.u
                + (1.0 - 2.0 * p.theta) * k2 * (-kappa2 * (p.B @ p.u))
                - u_nm1
                + p.theta * k2 * (-kappa2 * (p.B @ u_nm1))
                + sk * u_nm1
            )
            if f_ext is not None:
                rhs = rhs + k2 * f_ext / (p.rho * p.h * p.h)
        else:
            rhs = (
                p.W @ (2.0 * p.u - u_nm1)
                + (1.0 - 2.0 * p.theta) * k2 * (-kappa2 * (p.K @ p.u))
                + p.theta * k2 * (-kappa2 * (p.K @ u_nm1))
                + sk * (p.W @ u_nm1)
            )
            if f_ext is not None:
                rhs = rhs + k2 * f_ext / p.rho
        return rhs

    def commit(self, u_next: NDArray[np.float64]) -> None:
        """Roll ``u^{n-1} <- u^n <- u^{n+1}`` and refresh the plate's acceleration cache.

        ``_accel`` needs no post-solve correction, deliberately: the load was **inside** the solve,
        so the second difference already carries it and
        :meth:`~physsynth.core.plate.Plate.pressure` is right for free. (:class:`RoomLoadedBody`,
        whose rank-1 load lands *after* its solve, does need the refresh — the asymmetry is in the
        tier, not in the model.)
        """
        p = self.model
        p._accel = (u_next - 2.0 * p.u + p.u_prev) / (self.k * self.k)
        p.u_prev = p.u
        p.u = u_next
        p.n += 1


class RoomLoadedPlate:
    """A :class:`~physsynth.core.plate.Plate` mounted flush in a wall and loaded by the room over
    its **whole surface** — the distributed area coupling (batch 3).

    :class:`RoomLoadedBody` couples a lumped :class:`~physsynth.core.body.ModalBody` to a room
    through one scalar terminal. This couples a **grid** resonator through every node it has. The
    physical difference is :class:`SurfacePort`'s headline — the acoustic short circuit — and it is
    structural, not a refinement: an even-index mode of a simply supported plate has *exactly* zero
    net volume velocity, so every one-port in this repo (``AirRadiation``, ``RadiatedBody``,
    ``RationalAirLoad``, ``RoomPort``) reports exact silence from a mode this class radiates from
    definitely. This is also what makes the *existing* grid resonators radiate honestly: models #5
    and #5b have had a ``pressure()`` read-out since they were built, and it is
    ``sum_i area_i u_i''`` — a monopole, i.e. exactly the net volume acceleration, i.e. exactly the
    quantity that is insufficient. That read-out is not wrong; it is the compact ``a -> 0`` limit of
    this, and this is where the plate stops being a point source. (This sentence used to name model
    #4 as well. It does not have one: :class:`~physsynth.core.membrane.Membrane` exposes no
    ``pressure()``, and its two-level roll keeps no ``_accel`` to build one from. The point stands
    for the plates; the membrane simply never had the compact limit to be superseded.)

    **The load folds into the plate's own factorization — nothing new is solved.** The plate already
    back-substitutes a prefactored SPD system every step (``A = (1 + sigma k) I + theta k^2 kappa^2
    B`` supported, or the ``W``-weighted free form). Substituting the port's Thevenin relation into
    the plate's own linear response gives

        f_load = -T^T pbar_free  -  T^T R T (u^{n+1} - u^{n-1}) / (2k)

    whose unknown part is a constant, symmetric, **positive-semidefinite** sparse matrix times
    ``u^{n+1}``. A PSD addition to an SPD matrix is SPD, so ``A + (k/2rho_s) T^T R T`` is factored
    **once at construction** and the per-step cost of the whole room is two sparse matrix-vector
    products. Passivity is unconditional — it is a property of the matrix, not an inequality to
    check — so the room's CFL and the plate's unconditional stability are both untouched and
    coupling them adds no third condition.

    Per step, in this order (the ordering is load-bearing: ``pbar_free`` must be read from the
    stored ``u^{n+1/2}`` **before** the room advances)::

        1.  pbar_free = port.free_pressure()                     # vector, O(patch)
        2.  rhs = the plate's own force-free RHS  +  k^2 f_ext / rho_s
                  -  k^2 T^T pbar_free / rho_s
                  +  (k/2) (T^T R T) u^{n-1} / rho_s
        3.  u^{n+1} = LU_loaded.solve(rhs)
        4.  q    = T (u^{n+1} - u^{n-1}) / (2k)                  # per-node volume velocity, m^3/s
        5.  pbar = pbar_free + R q
        6.  radiated_energy += k pbar . q ;  port.inject(q)
        7.  (caller) room.step()

    where ``rho_s`` stands for ``rho h^2`` on the supported branch and ``rho`` on the free branch
    (the lumped mass ``W`` lives inside ``A`` and is divided out by the solve, exactly as
    :meth:`~physsynth.core.plate.Plate.step`'s own ``f_ext`` path already does).

    **The energy ledger telescopes exactly, and the conserved total is NOT the money test.** The
    theta-average applies to the *conservative* spatial operator, not to the load: ``f_load`` enters
    at time ``n`` as a plain force, so multiplying the update by the centered velocity gives
    ``dE_plate = -k pbar . q`` with **exactly** the ``q`` injected into the room. So

        plate.energy() + radiated_energy + room.energy()

    is flat for *any* ``T`` and *any* ``R`` — including a **wrong** ``R``. Each side's ledger
    telescopes against whatever pressure it used, and the sum of two internally-consistent
    identities is conserved even when the two disagree with each other. Measured by dropping the
    ``1 + beta`` wall factor from ``R_j`` on a lossy mounting wall, 300 steps: the conserved total
    drifts **4.9e-15 — smaller than the correct run's own 2.0e-14** — while ``radiated - injected``
    goes from *exactly zero* to **18% of the channel** (12.9%, free plate). So the conserved total
    ships as **necessary and not sufficient**, and the money test is the cross-ledger identity
    ``radiated_energy == room.injected`` plus a differential per-node measurement of ``R_j`` read
    straight off the room.

    **And every conservation claim must report how big the channel is**, because a conservation test
    on a channel worth 1e-14 of the total passes with the coupling disconnected. The channel is a
    property of the *motion*, not of the coupling: measured over 400 steps into an all-lossy room,
    a narrow plucked bump gives **0.0028** of ``E0`` on a supported plate and 0.0018 on a free one
    (fine spatial patterns radiate badly — that is the acoustic short circuit doing its job), while
    the free plate's uniform-velocity **piston** gives **0.9974**. So the piston configuration is
    the one that makes a conservation assertion non-vacuous, and it ships alongside the others.

    **No ``_accel`` correction is needed, deliberately.** :class:`RoomLoadedBody` has to refresh
    ``body._accel`` *after* its rank-1 correction because its load is applied post-solve. Here
    the load is **inside** the solve, so the second difference
    ``(u^{n+1} - 2u^n + u^{n-1})/k^2`` already carries it and
    :meth:`~physsynth.core.plate.Plate.pressure` is right for free. The one-line override
    batch 2 needed is absent here.

    **The string -> bridge -> plate -> room chain comes free — measured, not inferred.** Being a
    drop-in for ``Plate`` by ``__getattr__`` delegation, this slots straight into a
    :class:`~physsynth.core.connection.StringPlateBridge` as the body with **no edit to
    ``connection.py``**. That claim needed measuring rather than inheriting, because unlike batch
    2's ``ModalBody`` bridge, ``StringPlateBridge._stability_margin`` **reassembles the plate's
    ``G0`` block from scratch** out of ``theta, rho, h, kappa, B / W, K`` — every one of which a
    delegating wrapper hands over happily, so the guard would be computed against physics that is
    not happening and the delegation would hide it perfectly. The news is good on both counts:
    ``G0 = M + (theta - 1/4) k^2 S`` is a statement about **mass and theta-excess stiffness**, and
    the air load is **dissipative** — it enters ``A``, never ``G0`` — so the margin comes out
    **bit-identical** loaded or bare (measured ``0.2061806714931906`` supported and
    ``0.2061840079056186`` free, in both cases to the last digit); and adding the load block to
    ``G0`` anyway *reduces* ``(G0^-1)_dp`` — measured ratio 0.500 supported, 0.995 free, both below
    1 — i.e. the true margin is *smaller* than the guard reports. **The guard's blindness errs
    safe**, and it is the sign of that ratio, not its size, that is the claim. A test pins the
    bit-identity so a future change making the load non-dissipative fails loudly there instead of
    silently mis-guarding. It used to name batch 4's two-sided dipole plate, "whose face cut removes
    air mass", as that change; measured, it is not one — :class:`RoomSuspendedPlate` gives the same
    margin to the last digit, because the cut removes air inertia from the **room's** ledger and
    never from the plate's ``G0``.

    **The free plate's rigid-body translation now radiates.** Model #5b's stiffness nullspace is
    exactly ``{1, x, y}`` — bare, a uniform velocity translates forever at *constant* energy
    (measured ``E/E0 = 1.0000`` at 400 … 2000 steps). Mount the same plate flush in a baffle and
    that identical motion **is a piston**, the most efficient radiator the geometry has, so a lossy
    room takes all of it (``E_plate/E0 < 1e-3`` by 400 steps, ``radiated/E0 -> 1.0000``). This is
    also why the coupling channel is 0.35% of ``E0`` for a supported plate and **35%** for a free
    one in the same rig — the clamped rim is a poor radiator and the free plate's rigid body is a
    perfect one. Do **not** expect monotone decay in a *rigid* room: the plate drops to 4.5%, climbs
    back to 15.9% and wanders, because a closed box gives the piston's energy back. Only the total
    is monotone there (exactly flat, 2.0e-13).

    Parameters
    ----------
    plate : Plate
        The resonator — ``boundary="supported"`` (model #5) or ``"free"`` (model #5b). Only its
        public state and operators are used; ``plate.py`` is untouched. Its sample rate must match
        the room's.
    room : AirBox
        The room to radiate into. The plate lies flush in one of its walls, so the rigid wall *is*
        the textbook infinite baffle and the plate's back face is unloaded.
    face : str
        Which of the six walls (:data:`FACES`) the plate is mounted in. Positive plate displacement
        is along that face's **inward normal** (see :class:`SurfacePort` — the local convention is
        what makes the per-face sign that no energy report can catch not exist in the code).
    origin : (float, float), optional
        The plate's ``(0, 0)`` corner in the face's own two coordinates (m). **Defaults to
        centred**, and :class:`SurfacePort`'s spreading and scene-symmetry paragraphs are why: an
        off-centre plate in a mirror-symmetric room is legal and physical, but its even modes stop
        being exactly silent, *linearly* in the offset with no threshold (measured 2.3e-01 at an
        offset of ``h_air/3``, against 7.2e-14 centred).
    spreading : {"bilinear", "nearest"}
        Forwarded to :class:`SurfacePort`. ``"nearest"`` is that class's measured negative control,
        not a configuration.

    Raises
    ------
    ValueError
        A sample-rate mismatch, or any of :class:`SurfacePort`'s refusals.

    Notes
    -----
    **This class does not step the room**, exactly as :class:`RoomLoadedBody` does not::

        for n in range(n_steps):
            inst.step(f_ext)       # or bridge.step(), which owns plate.step
            room.step()            # one room, one step, after every port has solved

    Mixing tiers in one room is inherited unchanged: a ``RoomLoadedPlate`` and a
    ``RoomLoadedBody`` can share a room provided their node sets are disjoint, which is exactly what
    makes each port's solve exact.
    """

    def __init__(
        self,
        *,
        plate: Plate,
        room: AirBox,
        face: str,
        origin: Sequence[float] | None = None,
        spreading: Spreading = "bilinear",
    ) -> None:
        self.plate = plate  # FIRST: any attribute miss before this makes __getattr__ recurse
        _require_same_rate(plate, room, "plate")
        self.room = room
        self.k = plate.k

        # The surface protocol (see _PlateSurface): live-node coordinates and their areas. The live
        # ordering is C-order over `mask` (operators2d's contract), which is exactly what `X[mask]`
        # yields -- so the columns of T line up with the plate's own state vector by construction.
        self._surface = _PlateSurface(plate)
        coords, areas = self._surface.surface()
        self._denominator = self._surface.denominator
        self.port = SurfacePort(
            room=room,
            face=face,
            coords=coords,
            areas=areas,
            origin=origin,
            spreading=spreading,
        )

        # A_loaded = A + (k / 2 rho_s) T^T R T -- SPD (PSD added to SPD), factored ONCE. The plate's
        # own A is reassembled by the seam rather than reached into, because plate.py stays
        # untouched; the dense coupled cross-check at two timesteps pins that reassembly.
        a_bare = self._surface.a_bare()
        self._load_scale = 0.5 * plate.k / self._denominator
        a_loaded = (a_bare + self._load_scale * self.port.load_matrix).tocsc()
        # Drop the structural zeros the load's sparsity pattern contributes where its value is 0, so
        # a zero-area surface (T = 0) factors the plate's OWN matrix and reduces to the bare plate.
        a_loaded.eliminate_zeros()
        # The load's cost, reported rather than assumed -- and reported twice, because the two
        # numbers say different things. `nnz_growth` counts STORED entries; `lu_nnz` is what splu
        # actually pays, which is fill-in and is the one that matters. Measured against a bare-plate
        # factorization at h_plate/h_air = 0.45, 0.23, 0.15: stored 2.9x, 8.7x, 18.2x but fill only
        # 1.55x, 3.50x, 5.29x. Real growth either way -- the load couples every plate node sharing
        # an air node, so the block is ~(h_air/h_plate)^2 wide and the factorization DOES thicken.
        self.nnz_growth = a_loaded.nnz / a_bare.tocsc().nnz
        self._lu_loaded = splu(a_loaded)
        self.lu_nnz = int(self._lu_loaded.L.nnz + self._lu_loaded.U.nnz)

        self.radiated_energy = 0.0  # integral pbar . q dt: the work this plate did on the room
        self.nodal_volume_velocity = np.zeros(self.port.node_count)  # last q (m^3/s per node)
        self.surface_pressure = np.zeros(self.port.node_count)       # last pbar (Pa per node)
        self.volume_velocity = 0.0  # last sum_j q_j -- the LUMPED tier's coupling, i.e. the control
        self.n = 0

    def __getattr__(self, name: str):
        # Delegate read accessors (u, u_prev, X, Y, mask, B, K, W, w, theta, kappa, rho, h, n_live,
        # boundary, state, to_live, pickup_index_at, ...) so a RoomLoadedPlate is a drop-in wherever
        # a bare Plate is expected -- notably StringPlateBridge, which reassembles the plate's G0
        # block out of exactly those. Only reached for names not set on the instance, so the
        # overrides below always win. NOTHING here may shadow a name that bridge reads.
        if name == "plate":  # nothing to delegate through yet -- never recurse
            raise AttributeError(name)
        return getattr(self.plate, name)

    # -- time stepping ---------------------------------------------------------------------

    def step(self, f_ext: NDArray[np.float64] | None = None) -> None:
        """Advance one step: read the port, solve the **loaded** system, queue the injection.

        ``f_ext`` is the optional external nodal force (live vector) — the driving-point coupling a
        :class:`~physsynth.core.connection.StringPlateBridge` injects — and is added to the RHS
        with exactly :meth:`~physsynth.core.plate.Plate.step`'s own arithmetic, so a zero air load
        is a clean reduction to the bare plate. The room is **not** stepped here (class docstring).
        """
        port = self.port
        port.require_ready()               # before mutating anything
        pbar_free = port.free_pressure()   # read u^{n+1/2}, BEFORE room.step()

        k2 = self.k * self.k
        u_nm1 = self._surface.u_prev   # ONCE, and before commit() -- see _PlateSurface
        rhs = self._surface.rhs(f_ext)
        # The air load: the known open-circuit force, plus the u^{n-1} half of the centered
        # velocity (its u^{n+1} half is already inside the factorization).
        rhs = rhs - k2 * (port.T.T @ pbar_free) / self._denominator
        rhs = rhs + self._load_scale * (port.load_matrix @ u_nm1)

        u_next = self._lu_loaded.solve(rhs)
        # _accel already carries the load -- it was IN the solve, so no post-solve refresh.
        self._surface.commit(u_next)

        q = port.T @ ((u_next - u_nm1) / (2.0 * self.k))
        pbar = pbar_free + port.R * q
        port.inject(q)
        self.radiated_energy += self.k * float(np.dot(pbar, q))
        self.nodal_volume_velocity = q
        self.surface_pressure = pbar
        self.volume_velocity = float(np.sum(q))
        self.n += 1

    # -- diagnostics -----------------------------------------------------------------------

    def energy(self) -> float:
        """Total discrete energy ``E_plate + integral pbar . q dt`` (Joules).

        **An explicit override, not a delegation** — ``__getattr__`` would otherwise hand back the
        bare plate energy, i.e. the total *without* its coupling channel, which is exactly the
        number that looks fine and is not conserved. Like :meth:`RoomLoadedBody.energy` and unlike
        :meth:`~physsynth.core.radiation.RadiatedBody.energy` it is **not** monotone: a room gives
        energy back. The conserved statement is the whole scene, ``inst.energy() + room.energy()``
        — and see the class docstring on why that flatness is necessary and not sufficient.
        """
        return self.plate.energy() + self.radiated_energy

    def pressure(self) -> float:
        """The plate's **monopole** read-out ``sum_i area_i u_i''`` (Pa-ish), reflecting the load.

        Delegates to :meth:`~physsynth.core.plate.Plate.pressure`, which is right for free here
        (the load is inside the solve, so ``_accel`` carries it with no post-solve refresh). Keep in
        mind what it *is*: the net volume acceleration — the compact limit this class exists to go
        beyond. It is exactly zero for the modes whose radiation is this batch's headline. For the
        pressure **in the room**, read the field: :meth:`AirBox.pressure_at`.
        """
        return self.plate.pressure()

    def set_state(
        self,
        u0: NDArray[np.float64],
        v0: NDArray[np.float64] | float = 0.0,
    ) -> None:
        """Set the plate's initial state and reset this port's coupling ledger to zero.

        The plate's own consistent second-order start is used verbatim, i.e. ``u^{-1}`` is derived
        from the **unloaded** acceleration — the room has not acted yet at ``n = 0``.
        """
        self.plate.set_state(u0, v0)
        self._reset_books()

    def reset(self) -> None:
        """Zero the plate state and the coupling ledger — reuse on a new run.

        The room is a separate object with its own :meth:`AirBox.set_state`.
        """
        self.plate.set_state(np.zeros(self.plate.n_live))
        self._reset_books()

    def _reset_books(self) -> None:
        self.radiated_energy = 0.0
        self.nodal_volume_velocity = np.zeros(self.port.node_count)
        self.surface_pressure = np.zeros(self.port.node_count)
        self.volume_velocity = 0.0
        self.port.reset()
        self.n = 0


class RoomSuspendedPlate:
    """A :class:`~physsynth.core.plate.Plate` hanging **in** a room, loaded on both faces (batch 4).

    :class:`RoomLoadedPlate` mounts a plate flush in a wall: baffled, one-sided, a source. This
    suspends the same plate in the air — the free cymbal of model #5b, rather than a cone in a
    cabinet — so it radiates from both faces, is driven by the pressure *jump* across it, and
    **blocks a path through the room even at rest**. Three things follow that no amount of refining
    the one-sided model reaches:

    * **The blockage is not a refinement of the source.** Drop the cut and keep the ``-q``/``+q``
      pair and what is left is a legal, perfectly conservative dipole *source* with the plate's own
      motion and no obstacle — the "phantom". Its coupling *diverges* from this one under air-grid
      refinement (measured ``t50`` ratios 5.2, 19.3, 40.8 at 1x/2x/3x), because a transparent
      doublet at separation ``h`` has moment proportional to ``h``. There is no grid on which the
      source alone is a coarse version of the object.
    * **Unbaffling is not a factor — it changes sign.** Under prescribed uniform motion in a
      windowed free field the ratio of radiation resistances dipole/baffled measures 0.278, 0.569,
      1.339, 2.257, 2.314, 1.965 at ``ka = 0.8 .. 2.8`` — it **crosses 1**, and a ratio that crosses
      1 cannot be reproduced by any constant, hence not by any ``R(omega)`` fitted to the baffled
      case. That is the same structural argument as batch 2's delayed echo and batch 3's acoustic
      short circuit, a third time. The *crossing* is the claim: under air-grid refinement the
      baffled arm converges smoothly toward its plane-wave asymptote while the dipole arm tracks
      :attr:`~InteriorSurfacePort.blocked_area` instead, so its magnitude does not converge — only
      the ratio's sign relative to 1 does.
    * **It has a direction, and nothing else in this repo does.** Windowed free field, normalized
      peak ``|p|`` from on-axis to in-plane: 1.000, 0.928, 0.786, 0.565, 0.347, 0.164, **0.012** —
      an 85x null in the plate's own plane — against a baffled arc that only reaches 0.530,
      with no null anywhere. Every lumped one-port here (``AirRadiation``, ``RadiatedBody``,
      ``RationalAirLoad``, ``RoomPort``) is a monopole with no angular dependence at all.

    The last two are ``scripts/diagnose_airbox_dipole.py``'s, not the suite's — see
    :attr:`radiated_energy` on why a radiation figure needs a prescribed-velocity rig. **Do not read
    the first bullet against the ``t50`` numbers above and conclude they disagree.** They point
    opposite ways and both are right: ``t50`` counts the reactive near field as though it had left,
    so a free plate's piston sheds its energy ~20x *faster* suspended than baffled while
    **radiating** far less at that same low ``ka``. That pair is exactly why neither a decay time
    nor :attr:`radiated_energy` ships as a radiation measure.

    **The scheme**, per step, in this order (``pbar_free`` must be read from the stored
    ``u^{n+1/2}`` **before** the room advances — batch 2's contract, unchanged)::

        1.  (lo_free, hi_free) = port.free_pressure() ;  d_free = hi_free - lo_free
        2.  rhs = the plate's own force-free RHS  +  k^2 f_ext / rho_s
                  -  k^2 T^T d_free / rho_s
                  +  (k / 2 rho_s) (2 T^T R T) u^{n-1}
        3.  u^{n+1} = LU_loaded.solve(rhs)          # A + (k / 2 rho_s) 2 T^T R T, factored ONCE
        4.  q      = T (u^{n+1} - u^{n-1}) / (2k)   # per-FACE volume velocity, m^3/s
        5.  d_pbar = d_free + 2 R q
        6.  radiated_energy += k (d_pbar . q) ;  port.inject(q)
        7.  (caller) room.step()

    where ``rho_s`` is ``rho h^2`` on the supported branch and ``rho`` on the free branch, exactly
    as batch 3. **No** ``_accel`` correction, for batch 3's reason: the load is inside the solve, so
    the second difference already carries it.

    **Why the plate's centered velocity goes into a face slot that lives at** ``n+1/2``. The face
    velocity is a half-step quantity; the plate's natural velocity at time ``n`` is
    ``(u^{n+1} - u^{n-1})/(2k)``. Using the centered one is what makes both ledgers telescope
    against the **same** ``(q, pbar)`` pair, i.e. what makes ``radiated == injected`` an identity
    rather than an approximation. The alternative — the forward difference, which is the *formally*
    correct half-step object — puts a term in ``u^{n+1}`` alone into the load, i.e. an **added
    mass**, which would land in ``StringPlateBridge``'s stability guard and break the bit-identity
    below. So the choice is forced, and it is the same half-step mixing batch 3 already ships.
    Nothing in the ledgers can see it (both sides use the same pair), so it is bounded instead by a
    ``k``-only refinement — fix ``h`` and the geometry, raise ``fs`` — reported in the diagnose
    script.

    **The stability guard stays bit-identical, and batch 3's prediction that it would not is
    wrong.** ``tests/test_airbox_surface.py::test_string_bridge_plate_room_chain`` predicted that
    "the two-sided dipole plate of batch 4, whose face cut removes air mass" would make the load
    non-dissipative and fail there. It does not, and the reasoning does not survive: the face cut
    removes air inertia from the **room's** ledger, where it was never part of the plate's ``G0``,
    while the load itself stays proportional to ``u^{n+1} - u^{n-1}`` — dissipative, merely doubled
    — so it enters ``A`` and never ``G0``, exactly as in batch 3. The margin comes out
    bit-identical, and it is the *same* margin batch 3 measured, because the guard never saw
    either load.

    Parameters
    ----------
    plate : Plate
        The resonator — ``boundary="supported"`` (model #5) or ``"free"`` (model #5b). Only its
        public state and operators are used; ``plate.py`` is untouched. Its sample rate must match
        the room's.
    room : AirBox
        The room to hang it in.
    plane, index
        The plane of faces the plate lies on (see :class:`InteriorSurfacePort`). Positive plate
        displacement is along ``plane``'s own axis.
    origin : (float, float), optional
        The plate's ``(0, 0)`` corner in the plane's own two coordinates (m). **Defaults to
        centred**; :class:`SurfacePort`'s spreading and scene-symmetry paragraphs are why.
    spreading : {"bilinear", "nearest"}
        Forwarded to the port.

    Raises
    ------
    ValueError
        A sample-rate mismatch, or any of :class:`InteriorSurfacePort`'s refusals.

    Attributes
    ----------
    radiated_energy : float
        **The work done on the air — and about half of it comes back.** The name is batch 3's and
        must not keep its connotation: that channel was a one-way drain, so "radiated" was honest
        there. This one is *dominantly reactive* — 50.2% of its per-step increments are negative,
        against 1.2% for the same motion baffled — so a reader who takes ``radiated_energy/E0 =
        0.46`` to mean "46% radiated" is wrong by about a factor of two, in a direction the number
        cannot show. It keeps the name anyway because ``radiated_energy == room.injected`` is the
        money test and both sides must obviously name the same thing. For an actual radiation
        figure, prescribe the motion and integrate over whole cycles (the diagnose script), and
        never read a decay time or this fraction as one.
    pressure_jump : ndarray
        The last ``pbar_hi - pbar_lo`` (Pa), per face.
    nodal_volume_velocity : ndarray
        The last ``q`` (m^3/s per face). :attr:`volume_velocity` is its sum — the lumped tier's
        whole coupling, kept as the control.

    Notes
    -----
    **This class does not step the room**, exactly as batches 2 and 3 do not::

        for n in range(n_steps):
            inst.step(f_ext)       # or bridge.step(), which owns plate.step
            room.step()            # one room, one step, after every port has solved
    """

    def __init__(
        self,
        *,
        plate: Plate,
        room: AirBox,
        plane: str,
        index: int,
        origin: Sequence[float] | None = None,
        spreading: Spreading = "bilinear",
    ) -> None:
        self.plate = plate  # FIRST: any attribute miss before this makes __getattr__ recurse
        _require_same_rate(plate, room, "plate")
        self.room = room
        self.k = plate.k

        # The surface protocol (see _PlateSurface), batch 3's verbatim: live-node coordinates and
        # their areas, in the plate's own C-order over `mask`, so T's columns line up with the
        # plate's state vector.
        self._surface = _PlateSurface(plate)
        coords, areas = self._surface.surface()
        self._denominator = self._surface.denominator
        self.port = InteriorSurfacePort(
            room=room,
            plane=plane,
            index=index,
            coords=coords,
            areas=areas,
            origin=origin,
            spreading=spreading,
        )

        # A_loaded = A + (k / 2 rho_s) 2 T^T R T -- SPD (PSD added to SPD), factored ONCE. The
        # plate's own A is reassembled by the seam rather than reached into, because plate.py stays
        # untouched; the coupled residual at two timesteps is what pins that reassembly, and it is
        # the only guard that catches BOTH ways of getting the 2 wrong.
        a_bare = self._surface.a_bare()
        self._load_scale = 0.5 * plate.k / self._denominator
        a_loaded = (a_bare + self._load_scale * self.port.load_matrix).tocsc()
        # Drop the structural zeros the load's sparsity pattern contributes where its value is 0, so
        # a zero-area surface (T = 0) factors the plate's OWN matrix and reduces to the bare plate.
        a_loaded.eliminate_zeros()
        self.nnz_growth = a_loaded.nnz / a_bare.tocsc().nnz
        self._lu_loaded = splu(a_loaded)
        self.lu_nnz = int(self._lu_loaded.L.nnz + self._lu_loaded.U.nnz)

        self.radiated_energy = 0.0
        self.pressure_jump = np.zeros(self.port.face_count)          # last (pbar_hi - pbar_lo), Pa
        self.nodal_volume_velocity = np.zeros(self.port.face_count)  # last q (m^3/s per face)
        self.volume_velocity = 0.0  # last sum_j q_j -- the LUMPED tier's coupling, i.e. the control
        self.n = 0

    def __getattr__(self, name: str):
        # Delegate read accessors (u, u_prev, X, Y, mask, B, K, W, w, theta, kappa, rho, h, n_live,
        # boundary, state, to_live, pickup_index_at, ...) so this is a drop-in wherever a bare Plate
        # is expected -- notably StringPlateBridge, which reassembles the plate's G0 block out of
        # exactly those. Only reached for names not set on the instance, so the overrides below
        # always win. NOTHING here may shadow a name that bridge reads.
        if name == "plate":  # nothing to delegate through yet -- never recurse
            raise AttributeError(name)
        return getattr(self.plate, name)

    # -- time stepping ---------------------------------------------------------------------

    def step(self, f_ext: NDArray[np.float64] | None = None) -> None:
        """Advance one step: read both planes, solve the **loaded** system, queue the ``-q``/``+q``.

        ``f_ext`` is the optional external nodal force (live vector) — the driving-point coupling a
        :class:`~physsynth.core.connection.StringPlateBridge` injects — added to the RHS with
        exactly :meth:`~physsynth.core.plate.Plate.step`'s own arithmetic. The room is **not**
        stepped here (class docstring).
        """
        port = self.port
        port.require_ready()                      # before mutating anything
        lo_free, hi_free = port.free_pressure()   # read u^{n+1/2}, BEFORE room.step()
        d_free = hi_free - lo_free

        k2 = self.k * self.k
        u_nm1 = self._surface.u_prev   # ONCE, and before commit() -- see _PlateSurface
        rhs = self._surface.rhs(f_ext)
        # The air load: the known open-circuit pressure JUMP, plus the u^{n-1} half of the centered
        # velocity (its u^{n+1} half is already inside the factorization).
        rhs = rhs - k2 * (port.T.T @ d_free) / self._denominator
        rhs = rhs + self._load_scale * (port.load_matrix @ u_nm1)

        u_next = self._lu_loaded.solve(rhs)
        # _accel already carries the load -- it was IN the solve, so no post-solve refresh.
        self._surface.commit(u_next)

        q = port.T @ ((u_next - u_nm1) / (2.0 * self.k))
        d_pbar = d_free + 2.0 * port.R * q
        port.inject(q)
        self.radiated_energy += self.k * float(np.dot(d_pbar, q))
        self.nodal_volume_velocity = q
        self.pressure_jump = d_pbar
        self.volume_velocity = float(np.sum(q))
        self.n += 1

    # -- diagnostics -----------------------------------------------------------------------

    def energy(self) -> float:
        """Total discrete energy ``E_plate + radiated_energy`` (Joules).

        **An explicit override, not a delegation** — ``__getattr__`` would otherwise hand back the
        bare plate energy, i.e. the total *without* its coupling channel. Not monotone, and here for
        a stronger reason than batches 2 and 3: this channel is dominantly **reactive** (see
        :attr:`radiated_energy`), so it gives back roughly half of what it takes. The conserved
        statement is the whole scene, ``inst.energy() + inst.room.energy()`` — and that flatness is
        *necessary and not sufficient*; see :class:`InteriorSurfacePort` on the factor 2.
        """
        return self.plate.energy() + self.radiated_energy

    def pressure(self) -> float:
        """The plate's **monopole** read-out ``sum_i area_i u_i''``, reflecting the load.

        Delegates to :meth:`~physsynth.core.plate.Plate.pressure`, which is right for free here (the
        load is inside the solve). Keep in mind what it is: the net volume acceleration — the
        compact limit this class exists to go beyond, and doubly misleading for a dipole, which
        has no monopole moment in the far field at all. For the pressure **in the room**, read the
        field: :meth:`AirBox.pressure_at`.
        """
        return self.plate.pressure()

    def set_state(
        self,
        u0: NDArray[np.float64],
        v0: NDArray[np.float64] | float = 0.0,
    ) -> None:
        """Set the plate's initial state and reset this port's coupling ledger to zero.

        The plate's own consistent second-order start is used verbatim, i.e. ``u^{-1}`` is derived
        from the **unloaded** acceleration — the room has not acted yet at ``n = 0``.
        """
        self.plate.set_state(u0, v0)
        self._reset_books()

    def reset(self) -> None:
        """Zero the plate state and the coupling ledger — reuse on a new run.

        The room is a separate object with its own :meth:`AirBox.set_state`. Note the **cut is
        geometry, not state**: it survives both resets, as it must.
        """
        self.plate.set_state(np.zeros(self.plate.n_live))
        self._reset_books()

    def _reset_books(self) -> None:
        self.radiated_energy = 0.0
        self.pressure_jump = np.zeros(self.port.face_count)
        self.nodal_volume_velocity = np.zeros(self.port.face_count)
        self.volume_velocity = 0.0
        self.port.reset()
        self.n = 0


class _MembraneSurface:
    """The seam's membrane side (batch 5) — see :class:`_PlateSurface` for the six members.

    Three differences from the plate, and each one is a fact about model #4 rather than a
    convenience:

    * **The mass is uniform.** Every live node carries ``rho h^2``; there is no lumped ``W`` and no
      free-boundary branch, because a membrane's rim is clamped and dead. So ``denominator`` is a
      scalar and the load's ``(k / 2 rho h^2) T^T R T`` is a plain scaling.
    * **There is no ``_accel``** to refresh, so :meth:`commit` is the two-level roll and nothing
      else. This is the same gap that leaves ``Membrane`` without a ``pressure()`` read-out.
    * **``rhs``'s ``f_ext`` term has no counterpart in the model.**
      :meth:`~physsynth.core.plate.Plate.step` has its own ``f_ext`` path, so batch 3's copy of it
      could be checked against the original; ``Membrane.step()`` takes no force at all. This term
      is therefore *new* arithmetic and is pinned directly, by the static-deflection oracle
      ``u_ss = -L^-1 f / (T h^2)`` — not left to the energy ledger, which would stay green with the
      coefficient wrong.
    """

    def __init__(self, membrane: Membrane) -> None:
        self.model = membrane
        self.k = membrane.k
        self.areas = np.full(membrane.n_live, membrane.h * membrane.h)
        # Per-node mass rho h^2, uniform: the clamped rim is dead, not lightly weighted.
        self.denominator = membrane.rho * membrane.h * membrane.h

    def surface(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """``(coords, areas)`` — the **live** nodes only, which is the moving surface (batch 5).

        Unlike the free plate, where ``mask`` is all-ones, a membrane's rim is clamped and dead, so
        the radiating surface is one cell inside the nominal boundary: measured on a disk of
        ``R = 0.15 m`` at ``N = 56``, ``sum h^2 = 0.070284 m^2``, i.e. ``0.9943 pi R^2``. Read
        :attr:`InteriorSurfacePort.blocked_area` beside it — the obstacle the suspended tier puts
        in the room follows the *air* grid and is a different number again (1.228x there).
        """
        m = self.model
        coords = np.column_stack((m.X[m.mask], m.Y[m.mask]))
        return coords, self.areas

    def a_bare(self) -> sparse.spmatrix:
        """``(1 + sigma k) I`` — and its emptiness is the batch's main design fact.

        Model #4 is a **pure explicit** update: one matvec, no solve. The air load's unknown is
        ``u^{n+1}``, so putting it in ``A`` is what keeps ``radiated == injected`` an identity and
        passivity a property of the matrix rather than an inequality to check — at the price of a
        sparse factorization and a back-substitution per step, in a model whose whole character was
        that it had neither. The alternative (lag the load velocity at ``(u^n - u^{n-1})/k`` and
        stay explicit) is measured once as a negative control and does not ship.
        """
        m = self.model
        return (1.0 + m.sigma * m.k) * sparse.identity(m.n_live, format="csc")

    @property
    def u_prev(self) -> NDArray[np.float64]:
        """``u^{n-1}`` — read once per step, before :meth:`commit` (see :class:`_PlateSurface`)."""
        return self.model.u_prev

    def rhs(self, f_ext: NDArray[np.float64] | None) -> NDArray[np.float64]:
        """``2 u^n - (1 - sigma k) u^{n-1} + c^2 k^2 L u^n``, plus ``k^2 f_ext / rho h^2``.

        The first part is :meth:`~physsynth.core.membrane.Membrane.step`'s own numerator in its own
        operand order, so a zero air load and no force reduce to the bare membrane exactly. The
        ``f_ext`` term is the one piece of arithmetic in this batch with nothing to be
        bit-identical to (class docstring).
        """
        m = self.model
        sk = m.sigma * self.k
        c2k2 = m.c * m.c * self.k * self.k
        rhs = 2.0 * m.u - (1.0 - sk) * m.u_prev + c2k2 * (m.L @ m.u)
        if f_ext is not None:
            rhs = rhs + self.k * self.k * f_ext / self.denominator
        return rhs

    def commit(self, u_next: NDArray[np.float64]) -> None:
        """Roll ``u^{n-1} <- u^n <- u^{n+1}``. There is no acceleration cache to refresh."""
        m = self.model
        m.u_prev = m.u
        m.u = u_next
        m.n += 1


class _RoomLoadedMembraneMixin:
    """What :class:`RoomLoadedMembrane` and :class:`RoomSuspendedMembrane` share verbatim."""

    def __getattr__(self, name: str):
        # Delegate read accessors (u, u_prev, X, Y, mask, L, c, h, n_live, domain, state, to_live,
        # pickup_index_at, displacement_at, ...) so this is a drop-in wherever a bare Membrane is
        # expected. Only reached for names not set on the instance, so the overrides always win.
        # Note what is deliberately NOT here: `pressure()`, because model #4 has none.
        if name == "membrane":  # nothing to delegate through yet -- never recurse
            raise AttributeError(name)
        return getattr(self.membrane, name)

    def energy(self) -> float:
        """Total discrete energy ``E_membrane + integral pbar . q dt`` (Joules).

        **An explicit override, not a delegation** — ``__getattr__`` would otherwise hand back the
        bare membrane energy, i.e. the total *without* its coupling channel. Not monotone: a room
        gives energy back. The conserved statement is the whole scene,
        ``inst.energy() + inst.room.energy()`` — and that flatness is *necessary and not
        sufficient*, for the third batch running (see :class:`RoomLoadedPlate`).
        """
        return self.membrane.energy() + self.radiated_energy

    def set_state(
        self,
        u0: NDArray[np.float64],
        v0: NDArray[np.float64] | float = 0.0,
    ) -> None:
        """Set the membrane's initial state and reset this port's coupling ledger to zero.

        The membrane's own consistent second-order start is used verbatim, i.e. ``u^{-1}`` comes
        from the **unloaded** acceleration — the room has not acted yet at ``n = 0``.
        """
        self.membrane.set_state(u0, v0)
        self._reset_books()

    def reset(self) -> None:
        """Zero the membrane state and the coupling ledger — reuse on a new run."""
        self.membrane.set_state(np.zeros(self.membrane.n_live))
        self._reset_books()


class RoomLoadedMembrane(_RoomLoadedMembraneMixin):
    """A :class:`~physsynth.core.membrane.Membrane` mounted flush in a wall — the baffled drumhead.

    Batch 3's :class:`RoomLoadedPlate` with model #4 in place of model #5, and the physics that
    changes is not a detail. **A membrane has no coincidence frequency.** Kirchhoff bending gives
    ``c_b(omega) = sqrt(kappa omega)``, which grows without bound, so every plate in this repo
    *crosses* ``c0`` at one frequency and is a poor radiator below it and a good one above. A
    membrane's wave speed is the constant ``c = sqrt(T/rho)``, with no ``omega`` in it at all. So:

    * there is no frequency at which the character changes — the surface is subsonic at **every**
      mode or supersonic at every mode;
    * the control is the material and the tuning, not the mode number, through the single number
      ``c/c0``, which a player walks across the threshold by tightening the head;
    * real drumheads sit well below it (Mylar at ``T ~ 3000 N/m``, ``rho ~ 0.26 kg/m^2`` gives
      ``c/c0 ~ 0.31``), so the acoustic short circuit is not an edge case for a drum — it is the
      drum's normal operating point, and it is why a head with no shell is quiet.

    **The claim is a continuum one and the scheme smears it — read this before quoting it.** The
    5-point Laplacian is dispersive, so the discrete phase speed falls below ``c`` at high
    wavenumber and a *marginally* supersonic head drops back under ``c0`` somewhere on the grid:
    measured at ``lambda_mem = 1/sqrt(2)``, ``c/c0 = 1.05`` inverts at ``beta h = 1.445``, about
    2.2 nodes per wavelength — on the grid, not past it. The threshold therefore ships
    **bracketed**, with the measurement band held below the 1% dispersion knee
    (``beta h = 0.686``, ~4.6 nodes/wavelength). A sweep that silently crossed the inversion would
    measure the opposite of the claim and look clean doing it.

    A gift from the same table: at ``lambda_mem = 1/sqrt(2)`` the grid **diagonal** is exactly
    dispersionless (1.0000 across the band) while the axis degrades to 0.707 at Nyquist — the 2-D
    analogue of the 1-D ``lambda = 1`` exactness, surviving on one direction only. It is a reason
    to run the membrane *at* its CFL ceiling here, and it is also the anisotropy maximum.

    **The membrane stops being explicit.** See :meth:`_MembraneSurface.a_bare`: the load goes into
    ``A``, so model #4 acquires a factorization and a back-substitution it never had. That price
    is reported, not buried — :attr:`nnz_growth` and :attr:`lu_nnz` are on the instance.

    **Both domains work, and the round one needed a fix.** ``domain="circle"`` is the interesting
    head, and :meth:`_PatchPort._check_footprint` refused every disk at every resolution until this
    batch: its required set was a bounding box. The disk's own two staircases — the clamped rim on
    the membrane grid, and the port footprint on the air grid — are why every disk oracle here is a
    ratio or a rate and never a magnitude.

    Parameters
    ----------
    membrane : Membrane
        The resonator (model #4), ``domain="rectangle"`` or ``"circle"``. Only its public state and
        operators are used; ``membrane.py`` is untouched. Its sample rate must match the room's.
    room : AirBox
        The room to radiate into. The membrane lies flush in one of its walls, so the rigid wall
        *is* the textbook infinite baffle and the back face is unloaded.
    face : str
        Which of the six walls (:data:`FACES`). Positive displacement is along that face's inward
        normal (see :class:`SurfacePort`).
    origin : (float, float), optional
        The membrane's ``(0, 0)`` corner in the face's own two coordinates (m). Defaults to
        **centred**; an off-centre surface's even modes stop being exactly silent, linearly in the
        offset with no threshold.
    spreading : {"bilinear", "nearest"}
        Forwarded to :class:`SurfacePort`. ``"nearest"`` is that class's measured negative control.

    Raises
    ------
    ValueError
        A sample-rate mismatch, or any of :class:`SurfacePort`'s refusals.

    Notes
    -----
    **This class does not step the room**, exactly as batches 2, 3 and 4 do not::

        for n in range(n_steps):
            inst.step(f_ext)
            room.step()            # one room, one step, after every port has solved
    """

    def __init__(
        self,
        *,
        membrane: Membrane,
        room: AirBox,
        face: str,
        origin: Sequence[float] | None = None,
        spreading: Spreading = "bilinear",
    ) -> None:
        self.membrane = membrane  # FIRST: an attribute miss before this makes __getattr__ recurse
        _require_same_rate(membrane, room, "membrane")
        self.room = room
        self.k = membrane.k

        self._surface = _MembraneSurface(membrane)
        coords, areas = self._surface.surface()
        self._denominator = self._surface.denominator
        self.port = SurfacePort(
            room=room,
            face=face,
            coords=coords,
            areas=areas,
            origin=origin,
            spreading=spreading,
        )

        a_bare = self._surface.a_bare()
        self._load_scale = 0.5 * self.k / self._denominator
        a_loaded = (a_bare + self._load_scale * self.port.load_matrix).tocsc()
        # Drop the structural zeros the load contributes where its value is 0, so a zero-area
        # surface (T = 0) factors (1 + sigma k) I and reduces to the bare membrane.
        a_loaded.eliminate_zeros()
        self.nnz_growth = a_loaded.nnz / a_bare.tocsc().nnz
        self._lu_loaded = splu(a_loaded)
        self.lu_nnz = int(self._lu_loaded.L.nnz + self._lu_loaded.U.nnz)

        self.radiated_energy = 0.0  # integral pbar . q dt: the work this membrane did on the room
        self.nodal_volume_velocity = np.zeros(self.port.node_count)  # last q (m^3/s per node)
        self.surface_pressure = np.zeros(self.port.node_count)       # last pbar (Pa per node)
        self.volume_velocity = 0.0  # last sum_j q_j -- the LUMPED tier's coupling, i.e. the control
        self.n = 0

    # -- time stepping ---------------------------------------------------------------------

    def step(self, f_ext: NDArray[np.float64] | None = None) -> None:
        """Advance one step: read the port, solve the **loaded** system, queue the injection.

        ``f_ext`` is the optional external nodal force (live vector, newtons). Unlike batch 3's, it
        has no counterpart in the model to be checked against — ``Membrane.step()`` takes no force
        — so it is pinned by its own static-deflection oracle rather than by a reduction. The room
        is **not** stepped here (class docstring).
        """
        port = self.port
        port.require_ready()               # before mutating anything
        pbar_free = port.free_pressure()   # read u^{n+1/2}, BEFORE room.step()

        k2 = self.k * self.k
        u_nm1 = self._surface.u_prev   # ONCE, and before commit() -- see _PlateSurface
        rhs = self._surface.rhs(f_ext)
        # The air load: the known open-circuit force, plus the u^{n-1} half of the centered
        # velocity (its u^{n+1} half is already inside the factorization).
        rhs = rhs - k2 * (port.T.T @ pbar_free) / self._denominator
        rhs = rhs + self._load_scale * (port.load_matrix @ u_nm1)

        u_next = self._lu_loaded.solve(rhs)
        self._surface.commit(u_next)

        q = port.T @ ((u_next - u_nm1) / (2.0 * self.k))
        pbar = pbar_free + port.R * q
        port.inject(q)
        self.radiated_energy += self.k * float(np.dot(pbar, q))
        self.nodal_volume_velocity = q
        self.surface_pressure = pbar
        self.volume_velocity = float(np.sum(q))
        self.n += 1

    def _reset_books(self) -> None:
        self.radiated_energy = 0.0
        self.nodal_volume_velocity = np.zeros(self.port.node_count)
        self.surface_pressure = np.zeros(self.port.node_count)
        self.volume_velocity = 0.0
        self.port.reset()
        self.n = 0


class RoomSuspendedMembrane(_RoomLoadedMembraneMixin):
    """A :class:`~physsynth.core.membrane.Membrane` hanging **in** the room — the frame drum.

    :class:`RoomLoadedMembrane` with the wall taken away, i.e. batch 4's move applied to model #4:
    the head radiates from **both** faces, is driven by the pressure **jump** across it, and is an
    *object* rather than a source — it also removes paths through the room, which is the cut
    :class:`InteriorSurfacePort` registers. The load matrix and the ``pbar`` term both double.

    The two tiers ship together and completeness is not the reason: the ``c/c0`` claim of
    :class:`RoomLoadedMembrane` is a **comparison**. Batch 4's dipole-over-baffled resistance ratio
    is the instrument that turned "unbaffling changes sign" into a claim rather than a magnitude,
    and the baffled arm here is the reference this one is read against.

    **Two areas, and they are different numbers.** :attr:`InteriorSurfacePort.net_area` is the
    moving surface — the *live* nodes, so one cell inside the nominal rim, because a membrane's
    boundary is clamped and dead. :attr:`InteriorSurfacePort.blocked_area` is the obstacle, which
    follows the reached **air** nodes. Measured on a disk of ``R = 0.15 m`` at ``N = 56``: 0.070284
    and 0.086293 m^2, a ratio of 1.228. Batch 4 measured that the dipole's magnitude tracks
    ``blocked_area`` rather than the air spacing, so an area quietly taken as ``pi R^2`` gives a
    plausible, wrong, and green-ledgered result. Report both by name or neither.

    Parameters
    ----------
    membrane : Membrane
        The resonator (model #4). Its sample rate must match the room's.
    room : AirBox
        The room. The cut is registered on construction and is **geometry, not state** — it
        survives :meth:`reset`.
    plane, index : str, int
        Which interior plane (:data:`PLANES`) and which index along its normal axis.
    origin : (float, float), optional
        Defaults to **centred** — and this is the first surface for which that centres a *disk*.
    spreading : {"bilinear", "nearest"}
        Forwarded to :class:`InteriorSurfacePort`.

    Raises
    ------
    ValueError
        A sample-rate mismatch, or any of :class:`InteriorSurfacePort`'s refusals.
    """

    def __init__(
        self,
        *,
        membrane: Membrane,
        room: AirBox,
        plane: str,
        index: int,
        origin: Sequence[float] | None = None,
        spreading: Spreading = "bilinear",
    ) -> None:
        self.membrane = membrane  # FIRST: an attribute miss before this makes __getattr__ recurse
        _require_same_rate(membrane, room, "membrane")
        self.room = room
        self.k = membrane.k

        self._surface = _MembraneSurface(membrane)
        coords, areas = self._surface.surface()
        self._denominator = self._surface.denominator
        self.port = InteriorSurfacePort(
            room=room,
            plane=plane,
            index=index,
            coords=coords,
            areas=areas,
            origin=origin,
            spreading=spreading,
        )

        # A_loaded = (1 + sigma k) I + (k / 2 rho h^2) 2 T^T R T -- the 2 is the two loaded faces,
        # and the coupled residual at two timesteps is the only guard that catches BOTH ways of
        # getting it wrong (see InteriorSurfacePort).
        a_bare = self._surface.a_bare()
        self._load_scale = 0.5 * self.k / self._denominator
        a_loaded = (a_bare + self._load_scale * self.port.load_matrix).tocsc()
        a_loaded.eliminate_zeros()
        self.nnz_growth = a_loaded.nnz / a_bare.tocsc().nnz
        self._lu_loaded = splu(a_loaded)
        self.lu_nnz = int(self._lu_loaded.L.nnz + self._lu_loaded.U.nnz)

        self.radiated_energy = 0.0
        self.pressure_jump = np.zeros(self.port.face_count)          # last (pbar_hi - pbar_lo), Pa
        self.nodal_volume_velocity = np.zeros(self.port.face_count)  # last q (m^3/s per face)
        self.volume_velocity = 0.0  # last sum_j q_j -- the LUMPED tier's coupling, i.e. the control
        self.n = 0

    # -- time stepping ---------------------------------------------------------------------

    def step(self, f_ext: NDArray[np.float64] | None = None) -> None:
        """Advance one step: read both planes, solve the **loaded** system, queue the ``-q``/``+q``.

        ``f_ext`` is the optional external nodal force (live vector, newtons); see
        :meth:`RoomLoadedMembrane.step` on why it is pinned by an oracle and not by a reduction.
        The room is **not** stepped here (class docstring).
        """
        port = self.port
        port.require_ready()                      # before mutating anything
        lo_free, hi_free = port.free_pressure()   # read u^{n+1/2}, BEFORE room.step()
        d_free = hi_free - lo_free

        k2 = self.k * self.k
        u_nm1 = self._surface.u_prev   # ONCE, and before commit() -- see _PlateSurface
        rhs = self._surface.rhs(f_ext)
        # The air load: the known open-circuit pressure JUMP, plus the u^{n-1} half of the centered
        # velocity (its u^{n+1} half is already inside the factorization).
        rhs = rhs - k2 * (port.T.T @ d_free) / self._denominator
        rhs = rhs + self._load_scale * (port.load_matrix @ u_nm1)

        u_next = self._lu_loaded.solve(rhs)
        self._surface.commit(u_next)

        q = port.T @ ((u_next - u_nm1) / (2.0 * self.k))
        d_pbar = d_free + 2.0 * port.R * q
        port.inject(q)
        self.radiated_energy += self.k * float(np.dot(d_pbar, q))
        self.nodal_volume_velocity = q
        self.pressure_jump = d_pbar
        self.volume_velocity = float(np.sum(q))
        self.n += 1

    def _reset_books(self) -> None:
        self.radiated_energy = 0.0
        self.pressure_jump = np.zeros(self.port.face_count)
        self.nodal_volume_velocity = np.zeros(self.port.face_count)
        self.volume_velocity = 0.0
        self.port.reset()
        self.n = 0


class _VKPlateSurface:
    """The seam's von Kármán side (batch 6) — see :class:`_PlateSurface` for the six members.

    Model #6 carries :class:`~physsynth.core.plate.Plate`'s entire linear vocabulary under the same
    names (``theta``, ``kappa``, ``B``, ``K``, ``W``, ``h``, ``n_live``, ``mask``, ``index_map``,
    ``X``, ``Y``, ``boundary``, ``u``, ``u_prev``, ``sigma``, ``k``), so batch 3's load arithmetic
    carries over with **one substitution and three differences**:

    * **``rho`` becomes ``rho_s``, and the substitution is this batch's silent-failure trap.**
      :class:`~physsynth.core.plate.VKPlate` has no ``rho`` at all: it has ``rho_v`` (volumetric,
      kg/m^3) and ``rho_s`` (areal, kg/m^2), which differ by the thickness ``e`` — a factor of
      **1000** for a 1 mm plate. Writing ``rho_v`` leaves the air load 1000x too weak, and every
      ledger still telescopes against the pressure it used, so **nothing green turns red**. It is
      :class:`_PlateSurface`'s own documented failure class arriving through a different door, and
      the ``nonlinear=False`` bit-identical regression is what catches it.
    * **:meth:`rhs` delegates the linear half to the model** rather than transcribing it, which is
      the opposite of what :class:`_PlateSurface` does — deliberately, and for a reason that is a
      fact about the two models rather than a change of mind.
      :meth:`~physsynth.core.plate.Plate.step` *inlines* its theta-scheme RHS, so batch 3 had no
      choice but to copy it; :meth:`~physsynth.core.plate.VKPlate._linear_rhs` is already a method,
      because the Picard loop needs it hoisted out. Calling it keeps ``plate.py`` untouched (the
      rule batches 3–5 keep) *and* removes a whole class of transcription slip.
    * **A fourth member, :meth:`solve`** — the Picard loop hook. The room's load folds into ``A``
      once and is linear in ``w^{n+1}``, but the loaded back-substitution sits *inside* the
      fixed-point iteration, so the seam has to own the loop rather than one solve.
    * **:meth:`commit` takes both fields.** There is no ``_accel`` to refresh (``VKPlate`` has
      none, and no ``pressure()`` to read it), but there *is* a second cached history to roll:
      ``F_prev <- F <- F^{n+1}``. That is why :meth:`solve` hands ``F^{n+1}`` back instead of
      stashing it — hidden ordered state is what this seam's docstring warns about, and a loop
      between the read and the commit makes it worse.

    ``u_prev`` remains a live read that the caller must take **before** :meth:`commit`, now with an
    entire fixed-point iteration in between.
    """

    def __init__(self, plate: VKPlate) -> None:
        self.model = plate
        self.k = plate.k
        if plate.boundary == "supported":
            self.areas = np.full(plate.n_live, plate.h * plate.h)
        else:
            self.areas = plate.wdiag.copy()  # lumped cell areas (h^2, h^2/2, h^2/4) -- no dead rim
        # Per-node mass: rho_s h^2 (supported) / rho_s (free, W divided out by the solve). Read off
        # the model rather than recomputed, so this and VKPlate.step()'s own f_ext path are the same
        # expression by construction -- AREAL density, never rho_v (see the class docstring).
        self.denominator = plate.force_denominator

    def surface(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """``(coords, areas)`` — live-node positions (m) and their areas (m^2), in model order."""
        p = self.model
        coords = np.column_stack((p.X[p.mask], p.Y[p.mask]))
        return coords, self.areas

    def a_bare(self) -> sparse.spmatrix:
        """The unloaded system matrix: ``(1 + sigma k) I + theta k^2 kappa^2 B``, or its ``W`` form.

        Batch 3's assembly verbatim — ``kappa`` already carries the areal density
        (``kappa = sqrt(D/rho_s)``), so this is the one place the ``rho_v``/``rho_s`` trap cannot
        reach. It reaches :attr:`denominator` and :attr:`RoomLoadedVKPlate._load_scale` instead.
        """
        p = self.model
        sk = p.sigma * p.k
        coeff = p.theta * p.k * p.k * p.kappa * p.kappa
        if p.boundary == "supported":
            return (1.0 + sk) * sparse.identity(p.n_live, format="csc") + coeff * p.B
        return (1.0 + sk) * p.W + coeff * p.K

    @property
    def u_prev(self) -> NDArray[np.float64]:
        """``w^{n-1}`` — read once per step, before :meth:`commit` (see :class:`_PlateSurface`)."""
        return self.model.u_prev

    def rhs(self, f_ext: NDArray[np.float64] | None) -> NDArray[np.float64]:
        """The model's own linear theta-scheme RHS, plus the ``f_ext`` path.

        The linear half is :meth:`~physsynth.core.plate.VKPlate._linear_rhs` itself (class
        docstring), so it is bit-identical to the model by construction rather than by inspection —
        and, because that method and :meth:`_PlateSurface.rhs` are the same expression in the same
        operand order, bit-identical to batch 3's as well, which is what makes the
        ``nonlinear=False`` regression a *byte-exact* claim.

        The ``f_ext`` term is the new arithmetic, and unlike batch 5's it has a counterpart:
        ``VKPlate.step()`` takes no force either, but :meth:`RoomLoadedPlate.step` does, so the
        regression pins this coefficient exactly rather than leaving it to an oracle.
        """
        rhs = self.model._linear_rhs()
        if f_ext is not None:
            rhs = rhs + self.k * self.k * f_ext / self.denominator
        return rhs

    def solve(
        self,
        lu: SuperLU,
        rhs_fixed: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """One step's solve — ``(w^{n+1}, F^{n+1})``, Picard-iterated on the **loaded** operator.

        ``rhs_fixed`` is the whole sweep-invariant right-hand side: the linear RHS, the ``f_ext``
        path, and both of the room's terms. That the room's terms are sweep-invariant is what makes
        this hook cheap — ``pbar_free`` and ``w^{n-1}`` are fixed for the step, so only the
        ``mu``-averaged coupling changes between sweeps and the loop is
        :meth:`~physsynth.core.plate.VKPlate.step`'s arithmetic otherwise verbatim.

        Three rules, each a failure this seam has already documented:

        * **``lu`` is an argument and is never assigned to the model.** The loaded factorization
          must not become ``vk._lu``, or bare-vs-loaded stops being observable and the regression
          stops meaning anything.
        * **``F^{n+1}`` comes back as a return value**, not as state on the adapter.
        * The model's ``n_iters`` / ``converged`` / ``last_residual`` diagnostics are written here,
          exactly as the model's own step writes them, so a caller reading them through the
          wrapper's ``__getattr__`` sees the loaded loop's truth. Read them **per step**: a plate
          can sit at the sweep cap throughout a run and still converge on its final step.
        """
        p = self.model
        if not p.nonlinear:
            w_next = lu.solve(rhs_fixed)
            p.n_iters = 1
            p.converged = True
            p.last_residual = 0.0
            # F and F_prev are both zeros on this path and nothing writes them, so handing F back
            # makes commit()'s roll a structural no-op rather than a coincidental one.
            return w_next, p.F

        k2 = self.k * self.k
        # Coupling force -> RHS factor, the model's own: "supported" k^2 l/rho_s (scalar mass);
        # "free" k^2 h^2 l/rho_s (A carries W's h^2, and the /W is the solve's, not ours).
        couple_factor = k2 / p.rho_s
        if p.boundary == "free":
            couple_factor *= p.h * p.h
        w_prev_full = p._to_full(p.u_prev)  # w^{n-1}
        f_prev_full = p.F_prev              # F^{n-1} (cached)
        w_j = 2.0 * p.u - p.u_prev          # predictor w^{n+1}_(0)
        f_new_full = p.F                    # fallback (unused once the loop runs)
        p.n_iters = 0
        p.converged = False
        for sweep in range(1, p.couple_max_iter + 1):
            p.n_iters = sweep
            w_j_full = p._to_full(w_j)
            f_new_full = p._airy_F(w_j_full)          # F^{n+1}_(j)
            w_avg = 0.5 * (w_j_full + w_prev_full)    # mu_{t.} w
            f_avg = 0.5 * (f_new_full + f_prev_full)  # mu_{t.} F
            coupling = p._to_live(p.bracket(w_avg, f_avg))
            rhs = rhs_fixed + couple_factor * coupling
            w_next = lu.solve(rhs)
            incr = float(np.linalg.norm(w_next - w_j))
            scale = float(np.linalg.norm(w_next))
            w_j = w_next
            p.last_residual = incr / max(scale, 1e-30)
            if p.last_residual <= p.couple_tol:
                p.converged = True
                break
        return w_j, f_new_full

    def commit(
        self,
        u_next: NDArray[np.float64],
        f_next: NDArray[np.float64],
    ) -> None:
        """Roll **both** histories: ``w^{n-1} <- w^n <- w^{n+1}`` and ``F^{n-1} <- F^n <- F^{n+1}``.

        There is no acceleration cache to refresh (model #6 has none, hence no ``pressure()``), and
        the ``F`` roll is the member neither predecessor has. Its order matters: ``F_prev`` must be
        taken from the *old* ``F`` before ``F`` is overwritten, which is the model's own roll.
        """
        p = self.model
        p.F_prev = p.F
        p.F = f_next
        p.u_prev = p.u
        p.u = u_next
        p.n += 1


class _RoomLoadedVKPlateMixin:
    """What :class:`RoomLoadedVKPlate` and :class:`RoomSuspendedVKPlate` share verbatim."""

    def __getattr__(self, name: str):
        # Delegate read accessors (u, u_prev, X, Y, mask, B, K, W, wdiag, theta, kappa, rho_s, h,
        # n_live, boundary, state, stress_field, to_live, pickup_index_at, n_iters, converged,
        # last_residual, ...) so this is a drop-in wherever a bare VKPlate is expected. Only reached
        # for names not set on the instance, so the overrides always win. Note what is deliberately
        # NOT here: `pressure()`, because model #6 has none. That absence is now load-bearing
        # rather than incidental: `StringVKPlateBridge` DOES compose with this class (the three-way
        # chain, `docs/dev/string-vk-plate-room-plan.md`), and it composes precisely because the
        # bridge batch supplied the two things model #6 lacked -- it reads `plate.rho_s` and calls
        # `plate.step(f_ext=...)`, both of which delegation and the overrides below hand over -- and
        # *refused* the third. A `pressure()` added here would silently give that chain a compact
        # monopole read-out measured at 3e-7 of the truth, pointing the wrong way for a cymbal.
        if name == "plate":  # nothing to delegate through yet -- never recurse
            raise AttributeError(name)
        return getattr(self.plate, name)

    def energy(self) -> float:
        """Total discrete energy ``E_plate + integral pbar . q dt`` (Joules).

        **An explicit override, not a delegation** — ``__getattr__`` would otherwise hand back the
        bare plate energy, i.e. the total *without* its coupling channel. Not monotone: a room gives
        energy back. The conserved statement is the whole scene,
        ``inst.energy() + inst.room.energy()`` — necessary and not sufficient for the fourth batch
        running, and here it carries a **second** caveat that is new: model #6 conserves only at the
        Picard fixed point, so a run that stops short of ``couple_tol`` has an error source the air
        load knows nothing about. Read ``converged`` and ``n_iters`` per step beside this number.
        """
        return self.plate.energy() + self.radiated_energy

    def set_state(
        self,
        u0: NDArray[np.float64],
        v0: NDArray[np.float64] | float = 0.0,
    ) -> None:
        """Set the plate's initial state and reset this port's coupling ledger to zero.

        The plate's own consistent second-order start is used verbatim, i.e. ``w^{-1}`` comes from
        the **unloaded** acceleration — coupling included, room not, because the room has not acted
        yet at ``n = 0``. It also seeds the cached ``F(w^0)`` and ``F(w^{-1})``.
        """
        self.plate.set_state(u0, v0)
        self._reset_books()

    def reset(self) -> None:
        """Zero the plate state and the coupling ledger — reuse on a new run."""
        self.plate.set_state(np.zeros(self.plate.n_live))
        self._reset_books()


class RoomLoadedVKPlate(_RoomLoadedVKPlateMixin):
    """A :class:`~physsynth.core.plate.VKPlate` mounted flush in a wall — the **baffled gong**.

    Batch 3's :class:`RoomLoadedPlate` with model #6 in place of model #5, and what changes is not
    a refinement of the radiation — it is the first radiating object in this repo whose acoustic
    character is a function of **how hard it was hit**.

    Every other radiator here is linear in its excitation: strike it twice as hard and every
    acoustic observable doubles, so radiated fraction, directivity and dipole-over-baffled are
    amplitude-**invariant** by construction. The von Kármán coupling is quadratic, so the *shape* of
    the motion evolves during a single strike — measured on the bare plate, 41% of the modal energy
    moves to different modes within half a second at ``w = 3e``, against 0.26% at ``w -> 0``, same
    geometry, same excitation, one flag apart. Shape is exactly what :class:`SurfacePort` was built
    to make audible (batch 3: a surface radiates by the shape of its motion, not by its net volume
    displacement), so:

        **A loud plate's radiation is time-varying at fixed geometry, and a quiet one's is not.**

    No ``R(omega)`` can state that — not :class:`~physsynth.core.radiation.AirRadiation`, not
    :class:`~physsynth.core.radiation.RadiatedBody`, not
    :class:`~physsynth.core.radiation.RationalAirLoad`: a scalar-per-frequency load has *one*
    pattern per frequency and cannot change it mid-strike. The lumped read-out is measured blind to
    it — net volume displacement runs very nearly linear in the strike amplitude (1 : 20 : 60
    against amplitudes 1 : 20 : 60) while the shape content is not.

    **Two things this class does not claim**, both costed and deferred rather than omitted. The
    stiffening does *not* move coincidence usefully: the uniform-strain ansatz predicts 5.06x at
    ``w = 3e`` and the measured free plate gives **1.18–1.41x**, because a free edge relieves the
    stretching (the repo's own ``test_vk_free.py`` already asserted only ``>1.15x``). And resolving
    coincidence at all needs ~89 M air nodes and ~5.25 h for half a second, which a 14% shift does
    not touch. A cascade claim dies with it, and takes the *observable* with it: at ``w/e = 3`` a
    spectral peak tracker reads a mode's frequency as 0.53x its own linear value, because the field
    has gone broadband and "the" frequency has stopped existing. **Identify modes by projection
    under the mass matrix here, never by an FFT peak.**

    **The nonlinearity is also a control, in the same code path.** ``VKPlate(nonlinear=False)`` is
    bit-identical to :class:`~physsynth.core.plate.Plate`, so this class with the flag off is
    bit-identical to :class:`RoomLoadedPlate` — "frozen versus drifting" is one class and one flag,
    not two rigs.

    **The price is per sweep, not per step.** The loaded back-substitution runs once per Picard
    sweep, and the sweep count is a strong function of the configuration: measured 5, 10, 29 and 50
    (the cap, i.e. NaN) across plate geometries at ``w/e = 3``. A *narrow* strike on a small plate
    does not converge at all — the model's own "the strong-cascade regime may not converge" warning
    arriving as a hard limit on experiment design. Assert ``converged`` **per step**.

    Parameters
    ----------
    plate : VKPlate
        The resonator (model #6), ``boundary="supported"`` (the gong) or ``"free"`` (the cymbal).
        Only its public state, its operators and its own two step helpers are used; ``plate.py`` is
        untouched. Its sample rate must match the room's.
    room : AirBox
        The room to radiate into. The plate lies flush in one of its walls, so the rigid wall *is*
        the textbook infinite baffle and the plate's back face is unloaded.
    face : str
        Which of the six walls (:data:`FACES`). Positive plate displacement is along that face's
        inward normal (see :class:`SurfacePort`).
    origin : (float, float), optional
        The plate's ``(0, 0)`` corner in the face's own two coordinates (m). Defaults to
        **centred**; an off-centre surface's even modes stop being exactly silent, linearly in the
        offset with no threshold.
    spreading : {"bilinear", "nearest"}
        Forwarded to :class:`SurfacePort`. ``"nearest"`` is that class's measured negative control.

    Raises
    ------
    ValueError
        A sample-rate mismatch, or any of :class:`SurfacePort`'s refusals.

    Notes
    -----
    **This class does not step the room**, exactly as batches 2–5 do not::

        for n in range(n_steps):
            inst.step(f_ext)
            room.step()            # one room, one step, after every port has solved
    """

    def __init__(
        self,
        *,
        plate: VKPlate,
        room: AirBox,
        face: str,
        origin: Sequence[float] | None = None,
        spreading: Spreading = "bilinear",
    ) -> None:
        self.plate = plate  # FIRST: any attribute miss before this makes __getattr__ recurse
        _require_same_rate(plate, room, "plate")
        self.room = room
        self.k = plate.k

        self._surface = _VKPlateSurface(plate)
        coords, areas = self._surface.surface()
        self._denominator = self._surface.denominator
        self.port = SurfacePort(
            room=room,
            face=face,
            coords=coords,
            areas=areas,
            origin=origin,
            spreading=spreading,
        )

        # A_loaded = A + (k / 2 rho_s) T^T R T -- SPD, factored ONCE. The load is linear in w^{n+1}
        # and independent of the Airy stress F, which is the orthogonality that makes this batch
        # tractable: it folds into A exactly once and the Picard loop is otherwise untouched.
        a_bare = self._surface.a_bare()
        self._load_scale = 0.5 * plate.k / self._denominator
        a_loaded = (a_bare + self._load_scale * self.port.load_matrix).tocsc()
        # Drop the structural zeros the load's sparsity pattern contributes where its value is 0, so
        # a zero-area surface (T = 0) factors the plate's OWN matrix and reduces to the bare plate.
        a_loaded.eliminate_zeros()
        self.nnz_growth = a_loaded.nnz / a_bare.tocsc().nnz
        self._lu_loaded = splu(a_loaded)
        self.lu_nnz = int(self._lu_loaded.L.nnz + self._lu_loaded.U.nnz)

        self.radiated_energy = 0.0  # integral pbar . q dt: the work this plate did on the room
        self.nodal_volume_velocity = np.zeros(self.port.node_count)  # last q (m^3/s per node)
        self.surface_pressure = np.zeros(self.port.node_count)       # last pbar (Pa per node)
        self.volume_velocity = 0.0  # last sum_j q_j -- the LUMPED tier's coupling, i.e. the control
        self.n = 0

    # -- time stepping ---------------------------------------------------------------------

    def step(self, f_ext: NDArray[np.float64] | None = None) -> None:
        """Advance one step: read the port, Picard-solve the **loaded** system, queue the injection.

        ``f_ext`` is the optional external nodal force (live vector, newtons), added to the RHS with
        exactly :meth:`RoomLoadedPlate.step`'s arithmetic — which is what gives it a byte-exact
        counterpart, the first time this seam's force path has had one. The room's two terms are
        sweep-invariant and go into the fixed RHS the loop hook iterates around
        (:meth:`_VKPlateSurface.solve`). The room is **not** stepped here (class docstring).
        """
        port = self.port
        port.require_ready()               # before mutating anything
        pbar_free = port.free_pressure()   # read u^{n+1/2}, BEFORE room.step()

        k2 = self.k * self.k
        u_nm1 = self._surface.u_prev   # ONCE, and before commit() -- see _PlateSurface
        rhs = self._surface.rhs(f_ext)
        # The air load: the known open-circuit force, plus the w^{n-1} half of the centered
        # velocity (its w^{n+1} half is already inside the factorization).
        rhs = rhs - k2 * (port.T.T @ pbar_free) / self._denominator
        rhs = rhs + self._load_scale * (port.load_matrix @ u_nm1)

        u_next, f_next = self._surface.solve(self._lu_loaded, rhs)
        self._surface.commit(u_next, f_next)

        q = port.T @ ((u_next - u_nm1) / (2.0 * self.k))
        pbar = pbar_free + port.R * q
        port.inject(q)
        self.radiated_energy += self.k * float(np.dot(pbar, q))
        self.nodal_volume_velocity = q
        self.surface_pressure = pbar
        self.volume_velocity = float(np.sum(q))
        self.n += 1

    def _reset_books(self) -> None:
        self.radiated_energy = 0.0
        self.nodal_volume_velocity = np.zeros(self.port.node_count)
        self.surface_pressure = np.zeros(self.port.node_count)
        self.volume_velocity = 0.0
        self.port.reset()
        self.n = 0


class RoomSuspendedVKPlate(_RoomLoadedVKPlateMixin):
    """A :class:`~physsynth.core.plate.VKPlate` hanging **in** the room — the **cymbal on a stand**.

    :class:`RoomLoadedVKPlate` with the wall taken away, i.e. batch 4's move applied to model #6:
    the plate radiates from **both** faces, is driven by the pressure **jump** across it, and is an
    *object* rather than a source — it also removes paths through the room, which is the cut
    :class:`InteriorSurfacePort` registers. The load matrix and the ``pbar`` term both double, and
    the ``2`` is the coefficient no single ledger catches (see :class:`InteriorSurfacePort`).

    The two tiers ship together because this batch's claim is a **comparison**: the pattern the
    loud plate changes during a strike is read against the baffled arm and against its own quiet
    control, and batch 4's dipole-over-baffled ratio is the instrument that makes it a ratio rather
    than a magnitude. ``boundary="free"`` here is the physically honest cymbal — free edge, hung in
    air, radiating both sides — and it is the configuration batch 6's headline is measured on.

    Parameters
    ----------
    plate : VKPlate
        The resonator (model #6). Its sample rate must match the room's.
    room : AirBox
        The room. The cut is registered on construction and is **geometry, not state** — it
        survives :meth:`reset`.
    plane, index : str, int
        Which interior plane (:data:`PLANES`) and which index along its normal axis.
    origin : (float, float), optional
        Defaults to **centred**.
    spreading : {"bilinear", "nearest"}
        Forwarded to :class:`InteriorSurfacePort`.

    Raises
    ------
    ValueError
        A sample-rate mismatch, or any of :class:`InteriorSurfacePort`'s refusals.
    """

    def __init__(
        self,
        *,
        plate: VKPlate,
        room: AirBox,
        plane: str,
        index: int,
        origin: Sequence[float] | None = None,
        spreading: Spreading = "bilinear",
    ) -> None:
        self.plate = plate  # FIRST: any attribute miss before this makes __getattr__ recurse
        _require_same_rate(plate, room, "plate")
        self.room = room
        self.k = plate.k

        self._surface = _VKPlateSurface(plate)
        coords, areas = self._surface.surface()
        self._denominator = self._surface.denominator
        self.port = InteriorSurfacePort(
            room=room,
            plane=plane,
            index=index,
            coords=coords,
            areas=areas,
            origin=origin,
            spreading=spreading,
        )

        # A_loaded = A + (k / 2 rho_s) 2 T^T R T -- the 2 is the two loaded faces, and the coupled
        # residual at two timesteps is the only guard that catches BOTH ways of getting it wrong.
        a_bare = self._surface.a_bare()
        self._load_scale = 0.5 * plate.k / self._denominator
        a_loaded = (a_bare + self._load_scale * self.port.load_matrix).tocsc()
        a_loaded.eliminate_zeros()
        self.nnz_growth = a_loaded.nnz / a_bare.tocsc().nnz
        self._lu_loaded = splu(a_loaded)
        self.lu_nnz = int(self._lu_loaded.L.nnz + self._lu_loaded.U.nnz)

        self.radiated_energy = 0.0
        self.pressure_jump = np.zeros(self.port.face_count)          # last (pbar_hi - pbar_lo), Pa
        self.nodal_volume_velocity = np.zeros(self.port.face_count)  # last q (m^3/s per face)
        self.volume_velocity = 0.0  # last sum_j q_j -- the LUMPED tier's coupling, i.e. the control
        self.n = 0

    # -- time stepping ---------------------------------------------------------------------

    def step(self, f_ext: NDArray[np.float64] | None = None) -> None:
        """Advance one step: read both planes, Picard-solve, queue the ``-q``/``+q`` pair.

        ``f_ext`` is the optional external nodal force (live vector, newtons); see
        :meth:`RoomLoadedVKPlate.step`. The room is **not** stepped here (class docstring).
        """
        port = self.port
        port.require_ready()                      # before mutating anything
        lo_free, hi_free = port.free_pressure()   # read u^{n+1/2}, BEFORE room.step()
        d_free = hi_free - lo_free

        k2 = self.k * self.k
        u_nm1 = self._surface.u_prev   # ONCE, and before commit() -- see _PlateSurface
        rhs = self._surface.rhs(f_ext)
        # The air load: the known open-circuit pressure JUMP, plus the w^{n-1} half of the centered
        # velocity (its w^{n+1} half is already inside the factorization).
        rhs = rhs - k2 * (port.T.T @ d_free) / self._denominator
        rhs = rhs + self._load_scale * (port.load_matrix @ u_nm1)

        u_next, f_next = self._surface.solve(self._lu_loaded, rhs)
        self._surface.commit(u_next, f_next)

        q = port.T @ ((u_next - u_nm1) / (2.0 * self.k))
        d_pbar = d_free + 2.0 * port.R * q
        port.inject(q)
        self.radiated_energy += self.k * float(np.dot(d_pbar, q))
        self.nodal_volume_velocity = q
        self.pressure_jump = d_pbar
        self.volume_velocity = float(np.sum(q))
        self.n += 1

    def _reset_books(self) -> None:
        self.radiated_energy = 0.0
        self.pressure_jump = np.zeros(self.port.face_count)
        self.nodal_volume_velocity = np.zeros(self.port.face_count)
        self.volume_velocity = 0.0
        self.port.reset()
        self.n = 0


# -- the Rust swap (plan sections 28, 30 and 31) --------------------------------------------------
#
# Three things happen here, one batch apart each.
#
# **Section 31 (this batch): the port tier is ported** -- `RoomPort`, `SurfacePort` and
# `InteriorSurfacePort`, plus the module-level `_free_pressure_nodes` they share. The six
# `RoomLoaded*` / `RoomSuspended*` wrappers below still stay Python, and they keep working because
# a Rust port carries `dict` (two tests replace `free_pressure` and `inject` with lambdas ON THE
# INSTANCE) and exposes `T`, `load_matrix`, `R`, `areas` and `_queued_at` as settable slots holding
# whatever SciPy or NumPy object a caller assigns. `_PatchPort` is not swapped: it is a private base
# that exists to hold shared code, nothing outside this file names it, and the Rust classes share
# that code as free functions instead.
#
# The ports go before the wrappers for a reason worth keeping: a wrapper's `step` calls
# `port.free_pressure()`, solves, then calls `port.inject(q)`, so a Rust wrapper over a Python port
# would be a `&mut self` pymethod handing control back to Python twice mid-step -- section 13.2's
# refusal, which `bore` and `reed` already paid for. The other direction needs no contortion. And
# this tier owns no `splu` at all: every factorization in the file is in the wrapper tier.
#
# **Section 30 (this batch): `AirBox` itself is ported.** Only the class -- the ports and the six
# `RoomLoaded*` / `RoomSuspended*` wrappers below it stay Python, and they keep working because the
# binding exposes the whole private surface they reach through (`_w`, `_W`, `_beta`, `_open`,
# `_has_walls`, `_pending`, `_pending_ports`, `_ports`, `_cut_mask`, `_cut_index`, `_cuts`,
# `_register_cut`, `_plane_axis`, `_divergence`) as settable attributes rather than mirroring any
# of it. Six of those are containers a client WRITES -- every port appends to `_pending_ports`, and
# `tests/test_airbox_dipole.py::_uncut` assigns a fresh `[None, None, None]` over the three cut
# fields -- so they are Python objects the Rust class merely holds and reads back each step.
#
# `AirBoxPy` below is the reference implementation and stays the name every parity check reaches
# for. `_free_pressure_nodes`, the module-level helper the ports share, is unchanged and reads the
# Rust room's arrays directly; its docstring's bit-identity claim against `_divergence`-then-closure
# is therefore now a CROSS-LANGUAGE claim, and `tests/test_rust_parity_airbox.py` asserts it.
#
# **Section 28 (the previous batch): one imported name.**
#
# Every room-loaded class **reassembles** the plate's system matrix and factors it here, rather
# than reaching into `plate.py` -- and four of the family's reduction tests turn that into a
# bit-identity claim: with the load switched off, `a_loaded` IS the plate's own `A`, so a loaded
# plate must reproduce a bare one byte for byte. That claim is what pins this module's
# transcription of the theta-scheme right-hand side, and it survives only while both sides
# factor with the same solver. Porting `plate.py` moved the model's solver to Rust, so this
# module's has to move with it or the anchor breaks for a reason having nothing to do with
# `airbox.py` -- section 15.2's finding reaching a model and its *client* rather than two models.
#
# The shim is the whole of the change: `splu` is looked up as a module global at call time, so
# rebinding it here is enough, and nothing else in the file knows.

RoomLoadedBodyPy = RoomLoadedBody
RoomLoadedPlatePy = RoomLoadedPlate
RoomSuspendedPlatePy = RoomSuspendedPlate
RoomLoadedVKPlatePy = RoomLoadedVKPlate
RoomSuspendedVKPlatePy = RoomSuspendedVKPlate
RoomLoadedMembranePy = RoomLoadedMembrane
RoomSuspendedMembranePy = RoomSuspendedMembrane
_PlateSurfacePy = _PlateSurface
_VKPlateSurfacePy = _VKPlateSurface
_MembraneSurfacePy = _MembraneSurface
"""The pure-Python wrapper tier and its three seams, under names the swap never rebinds.

The seams keep their leading underscore in the alias (`_PlateSurfacePy`), unlike
`free_pressure_nodes_py`: `tests/test_stability.py` skips a leading underscore when it collects
`<name>_py` FUNCTION aliases and does not when it collects `<Name>Py` CLASS aliases, so this is the
spelling that keeps the guard covering them.
"""
"""The pure-Python port tier and its shared helper, under names the swap never rebinds.

`free_pressure_nodes_py` matters more than it looks. Its docstring's claim -- that a local
divergence read reproduces `_divergence()`-then-closure exactly, at wall, edge and corner nodes
alike -- is asserted in the parity file, and if the only spelling of the function were the Rust one
that assertion would compare Rust with Rust and pass having compared nothing (plan section 23.6,
which has now arrived through five different doors). Both spellings therefore stay live, and the
parity file compares them to each other.

The alias drops the leading underscore on purpose: `tests/test_stability.py`'s swap derive collects
`<name>_py` aliases and skips any that begin with one, which is the shape that dropped `collision`
out of that table for a whole batch (plan section 17.6). Following `collision`'s spelling is what
keeps this guard covering something.
parity file compares them to each other.
"""

_USE_RUST = os.environ.get("PHYSSYNTH_RS", "").strip() not in ("", "0", "false", "False")

# -- the factorization, and why it is NOT behind the flag (plan §43) -----------------------------
#
# Every room-loaded class in the wrapper tier **reassembles** the plate's (or the membrane's, or
# the body's) system matrix and factors it here rather than reaching into the model, and four of
# the family's reduction tests turn that into a bit-identity claim: with the load switched off,
# `a_loaded` IS the model's own `A`, so a loaded plate must reproduce a bare one byte for byte.
# That claim pins this module's transcription of the theta-scheme right-hand side, and it holds
# only while both sides factor with the same solver.
#
# This shim was written for that reason when the plate was ported (§28) and it sat behind
# `PHYSSYNTH_RS`, because *unflagged* the model still factored with SuperLU and so did this. Unit
# 5's deletion ended that: `plate.Plate` is the Rust class on every path now, so a SuperLU
# factorization here would break all four anchors on the DEFAULT path, for a reason having nothing
# to do with `airbox.py`. Measured before the fix: 15 tests, `array_equal` failing at ~1e-16 —
# §24.2's SuperLU-versus-crate gap, arriving through a client that re-derives the matrix.
#
# So the rebinding is unconditional. `splu` is looked up as a module global at call time, so this
# is the whole of it and nothing else in the file knows. What stays behind the flag below is the
# swapping of this module's own classes, which is a separate question and unit 6's.

class _Fill:
    """Just enough of a SuperLU factor for `lu_nnz` to read `.nnz` off it."""

    __slots__ = ("nnz",)

    def __init__(self, nnz: int) -> None:
        self.nnz = nnz


class _RustSuperLU:
    """`splu(A)`'s object, from the crate: `solve`, and the two fill counts."""

    __slots__ = ("_lu", "L", "U")

    def __init__(self, a) -> None:
        m = sparse.csr_matrix(a, copy=True)
        m.sort_indices()
        self._lu = _rs.SparseLu(
            m.data, m.indices.astype(np.int32), m.indptr.astype(np.int32), m.shape[0]
        )
        l_nnz, u_nnz = self._lu.nnz
        self.L = _Fill(l_nnz)
        self.U = _Fill(u_nnz)

    def solve(self, b):
        return np.asarray(self._lu.solve(np.ascontiguousarray(b, dtype=float)))


splu = _RustSuperLU

if _USE_RUST:  # pragma: no cover - exercised by the dedicated CI job, not the default gate
    _PlateSurface = _rs._PlateSurface  # type: ignore[assignment,misc]  # noqa: F811
    _VKPlateSurface = _rs._VKPlateSurface  # type: ignore[assignment,misc]  # noqa: F811
    RoomLoadedBody = _rs.RoomLoadedBody  # type: ignore[assignment,misc]  # noqa: F811
    RoomLoadedPlate = _rs.RoomLoadedPlate  # type: ignore[assignment,misc]  # noqa: F811
    RoomSuspendedPlate = _rs.RoomSuspendedPlate  # type: ignore[assignment,misc]  # noqa: F811
    RoomLoadedVKPlate = _rs.RoomLoadedVKPlate  # type: ignore[assignment,misc]  # noqa: F811
    RoomSuspendedVKPlate = _rs.RoomSuspendedVKPlate  # type: ignore[assignment,misc]  # noqa: F811
    _MembraneSurface = _rs._MembraneSurface  # type: ignore[assignment,misc]  # noqa: F811
    RoomLoadedMembrane = _rs.RoomLoadedMembrane  # type: ignore[assignment,misc]  # noqa: F811
    RoomSuspendedMembrane = _rs.RoomSuspendedMembrane  # type: ignore[assignment,misc]  # noqa: F811
