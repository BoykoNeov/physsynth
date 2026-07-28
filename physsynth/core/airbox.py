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

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:  # type-only: the room stays free of any dependency on the body module
    from physsynth.core.body import ModalBody

__all__ = [
    "AirBox",
    "FACES",
    "RHO0_AIR",
    "C0_AIR",
    "impedance_from_zeta",
    "RoomPort",
    "RoomLoadedBody",
]

# Ambient air (matches physsynth.core.radiation and .bore so every tier of the air node agrees).
RHO0_AIR = 1.2041  # kg/m^3
C0_AIR = 343.0     # m/s

# The six faces of the box, named <axis><end>: "x0" is the x = 0 wall, "x1" the x = Lx wall.
FACES = ("x0", "x1", "y0", "y1", "z0", "z1")

_AXES = "xyz"
_LAMBDA_MAX = 1.0 / np.sqrt(3.0)  # 3-D CFL ceiling
_LAMBDA_TOL = 1e-12


def impedance_from_zeta(
    zeta: float, *, rho0: float = RHO0_AIR, c0: float = C0_AIR
) -> float:
    """Specific acoustic impedance ``Z = zeta rho0 c0`` (Pa*s/m) from the **normalized** ``zeta``.

    A convenience for talking about walls in the dimensionless units the reflection oracle uses:
    a normal-incidence pulse hitting a locally-reacting wall reflects with
    ``R = (zeta - 1) / (zeta + 1)``, so ``zeta = 1`` is matched (at normal incidence), big ``zeta``
    is rigid and ``zeta = 0`` is pressure-release. Per the standing "unphysical parameters are a
    feature" rule the primary wall parameter stays the **effective** ``Z``; this is a helper for
    choosing one, never a materials table.
    """
    return float(zeta) * rho0 * c0


class AirBox:
    """A rectangular room of air on a 3-D Yee grid — staggered p/u leapfrog, energy-exact.

    Parameters
    ----------
    L : (float, float, float)
        Requested room dimensions ``(Lx, Ly, Lz)`` in metres, all positive.
    fs : float
        Sample rate (Hz); timestep ``k = 1/fs``.
    h : float
        Requested grid spacing (m), **uniform on all three axes**. The grid takes
        ``N_d = round(L_d / h)`` cells per axis, so the *actual* room is
        :attr:`L_actual` ``= (Nx h, Ny h, Nz h)``. Grid snap is the resolution and it is reported,
        never silently resampled away (the juari batch's precedent).
    walls : str, float, or mapping
        Wall termination. A single token/number applies to all six faces; a mapping over
        :data:`FACES` (``"x0" "x1" "y0" "y1" "z0" "z1"``) sets them independently, with unlisted
        faces left rigid. Each value is ``"rigid"`` (``Z = inf``), ``"open"`` (``Z = 0``, a
        pressure-release face), or a **float** specific acoustic impedance ``Z`` in Pa*s/m.
        :func:`impedance_from_zeta` turns a normalized ``zeta`` into one.
    source : (float, float, float), optional
        Default injection point (m), snapped to the nearest node. Defaults to the box centre.
    rho0, c0 : float
        Air density (kg/m^3) and sound speed (m/s). Default ambient air.

    Raises
    ------
    ValueError
        Non-physical parameters, a room thinner than one cell on any axis, an unknown face name or
        wall token, a negative impedance, or CFL ``lambda = c0 k / h > 1/sqrt(3)``.
    """

    def __init__(
        self,
        *,
        L: Sequence[float],
        fs: float,
        h: float,
        walls: str | float | Mapping[str, str | float] = "rigid",
        source: Sequence[float] | None = None,
        rho0: float = RHO0_AIR,
        c0: float = C0_AIR,
    ) -> None:
        L = tuple(float(v) for v in L)
        if len(L) != 3:
            raise ValueError(f"L must be a (Lx, Ly, Lz) triple, got {L!r}.")
        if min(*L, fs, h, rho0, c0) <= 0:
            raise ValueError("L, fs, h, rho0, c0 must all be positive.")

        self.L: tuple[float, float, float] = L  # type: ignore[assignment]
        self.fs = float(fs)
        self.h = float(h)
        self.rho0 = float(rho0)
        self.c0 = float(c0)
        self.k = 1.0 / self.fs

        self.N: tuple[int, int, int] = tuple(  # type: ignore[assignment]
            int(round(v / self.h)) for v in L
        )
        if min(self.N) < 1:
            raise ValueError(
                f"h = {self.h} is coarser than the room {L}: N = {self.N} has an axis with no "
                "cells. Refine h (or enlarge L) so every axis takes at least one cell."
            )
        self.L_actual: tuple[float, float, float] = tuple(  # type: ignore[assignment]
            n * self.h for n in self.N
        )

        self.lam = self.c0 * self.k / self.h
        if self.lam > _LAMBDA_MAX + _LAMBDA_TOL:
            raise ValueError(
                f"CFL violated: lambda = c0*k/h = {self.lam:.6f} > 1/sqrt(3) = "
                f"{_LAMBDA_MAX:.6f}. Raise fs, or coarsen the grid (increase h). Note 3-D has no "
                "dispersionless lambda — do not tune toward the ceiling as if it were the string."
            )

        Nx, Ny, Nz = self.N
        # Trapezoidal node weights per axis: h interior, h/2 at each wall. The half-cell IS the
        # rigid-wall closure (the free-beam end-mass lesson, twice generalized).
        self._w = tuple(self._trapezoid(n) for n in self.N)
        wx, wy, wz = self._w
        self._W = wx[:, None, None] * wy[None, :, None] * wz[None, None, :]
        # Face weights: full h along the face normal, trapezoid transverse.
        self._Wx = self.h * np.broadcast_to(
            wy[None, :, None] * wz[None, None, :], (Nx, Ny + 1, Nz + 1)
        ).copy()
        self._Wy = self.h * np.broadcast_to(
            wx[:, None, None] * wz[None, None, :], (Nx + 1, Ny, Nz + 1)
        ).copy()
        self._Wz = self.h * np.broadcast_to(
            wx[:, None, None] * wy[None, :, None], (Nx + 1, Ny + 1, Nz)
        ).copy()

        self.walls: dict[str, float] = self._normalize_walls(walls)
        self._build_wall_closure()

        self.source_index = self.node_index(
            source if source is not None else tuple(0.5 * v for v in self.L_actual)
        )
        self._pending: list[tuple[tuple[int, int, int], float]] = []
        # Port injections are queued separately from the scalar `inject()` path: a spread port is
        # thousands of nodes and the scalar loop would dominate the step (measured). Each entry is
        # (fancy-index triple, normalized volume weights, volume velocity U).
        self._pending_ports: list[
            tuple[tuple[NDArray[np.intp], ...], NDArray[np.float64], float]
        ] = []
        self._ports: list[RoomPort] = []  # live ports, for the disjointness check and reset

        self.p: NDArray[np.float64] = np.zeros((Nx + 1, Ny + 1, Nz + 1))
        self.ux: NDArray[np.float64] = np.zeros((Nx, Ny + 1, Nz + 1))
        self.uy: NDArray[np.float64] = np.zeros((Nx + 1, Ny, Nz + 1))
        self.uz: NDArray[np.float64] = np.zeros((Nx + 1, Ny + 1, Nz))
        self.ux_prev: NDArray[np.float64] = np.zeros_like(self.ux)
        self.uy_prev: NDArray[np.float64] = np.zeros_like(self.uy)
        self.uz_prev: NDArray[np.float64] = np.zeros_like(self.uz)

        self.dissipated = 0.0  # cumulative energy absorbed by the walls (>= 0)
        self.injected = 0.0    # cumulative work done by the soft source
        self.n: int = 0

    # -- construction helpers -------------------------------------------------------------

    def _trapezoid(self, n: int) -> NDArray[np.float64]:
        w = np.full(n + 1, self.h)
        w[0] = w[-1] = 0.5 * self.h
        return w

    def _normalize_walls(
        self, walls: str | float | Mapping[str, str | float]
    ) -> dict[str, float]:
        """Turn any accepted ``walls`` spelling into a plain ``{face: Z}`` dict of floats."""
        if isinstance(walls, Mapping):
            unknown = set(walls) - set(FACES)
            if unknown:
                raise ValueError(f"unknown face name(s) {sorted(unknown)}; expected {FACES}.")
            spec = {f: walls.get(f, "rigid") for f in FACES}
        else:
            spec = dict.fromkeys(FACES, walls)
        out: dict[str, float] = {}
        for face, value in spec.items():
            if isinstance(value, str):
                if value == "rigid":
                    out[face] = np.inf
                elif value == "open":
                    out[face] = 0.0
                else:
                    raise ValueError(
                        f"wall {face!r}: unknown token {value!r}; expected 'rigid', 'open', or a "
                        "float specific acoustic impedance Z (Pa*s/m)."
                    )
            else:
                z = float(value)
                if z < 0.0 or np.isnan(z):
                    raise ValueError(f"wall {face!r}: impedance Z must be >= 0, got {z!r}.")
                out[face] = z
        return out

    def _build_wall_closure(self) -> None:
        """Assemble the per-node ``beta`` field, the open-face mask, and the wall node areas.

        A node on an edge or corner touches several walls; their **admittances add**, so ``beta``
        is a plain sum over the faces that node belongs to and the solve stays 1x1. An ``open``
        face (``Z = 0``) is infinite admittance — carried as a boolean mask rather than an ``inf``
        in ``beta``, so no NaN can appear in the update, and it wins over any finite wall it shares
        a node with (a pressure-release face pins ``p = 0`` regardless).
        """
        shape = tuple(n + 1 for n in self.N)
        self._beta = np.zeros(shape)
        self._open = np.zeros(shape, dtype=bool)
        self._lossy: list[tuple[tuple[slice | int, ...], NDArray[np.float64], float]] = []
        for face in FACES:
            Z = self.walls[face]
            if np.isinf(Z):
                continue  # rigid: zero admittance, contributes nothing anywhere
            axis, end = _AXES.index(face[0]), int(face[1])
            sl: list[slice | int] = [slice(None)] * 3
            sl[axis] = 0 if end == 0 else -1
            index = tuple(sl)
            if Z == 0.0:
                self._open[index] = True
                continue  # an ideal pressure-release face carries no energy and sheds none
            w_wall = self._w[axis][0 if end == 0 else -1]
            self._beta[index] += self.k * self.rho0 * self.c0**2 / (2.0 * Z * w_wall)
            # Wall area carried by each node of this face: the transverse trapezoid weights.
            t0, t1 = (a for a in range(3) if a != axis)
            area = self._w[t0][:, None] * self._w[t1][None, :]
            self._lossy.append((index, area, Z))
        self._has_walls = bool(np.any(self._beta) or np.any(self._open))

    # -- initial conditions ---------------------------------------------------------------

    def set_state(
        self,
        p0: NDArray[np.float64],
        u0: Sequence[NDArray[np.float64]] | float = 0.0,
    ) -> None:
        """Set the initial pressure field ``p^0`` (and optional half-step-back velocity).

        Mirrors :meth:`physsynth.core.bore.Bore.set_state` exactly, and the contract matters: ``u0``
        is ``u^{-1/2}`` (default: rest), and the constructor derives ``u^{1/2}`` by one consistent
        momentum half-step from ``p^0``. Get that half-step wrong and a cross-model comparison
        against the bore is off by half a step — which reads as a *scheme* error, not a
        bookkeeping one. Pass ``u0`` as a ``(ux, uy, uz)`` triple of face arrays to seed motion;
        :meth:`set_mode` uses it for the exact discrete eigenmode.

        Open faces are pinned to ``p = 0`` here, and the wall/source energy books are reset, so a
        fresh run starts from a clean ledger.
        """
        p0 = np.asarray(p0, dtype=float).copy()
        if p0.shape != self.p.shape:
            raise ValueError(f"p0 must have shape {self.p.shape}, got {p0.shape}.")
        p0[self._open] = 0.0

        if np.isscalar(u0):
            prev = tuple(np.full_like(a, float(u0)) for a in (self.ux, self.uy, self.uz))
        else:
            prev = tuple(np.asarray(a, dtype=float).copy() for a in u0)  # type: ignore[union-attr]
            for got, want in zip(prev, (self.ux, self.uy, self.uz), strict=True):
                if got.shape != want.shape:
                    raise ValueError(
                        f"u0 components must have shapes "
                        f"{[a.shape for a in (self.ux, self.uy, self.uz)]}, got "
                        f"{[a.shape for a in prev]}."
                    )

        self.p = p0
        self.ux_prev, self.uy_prev, self.uz_prev = prev
        self.ux, self.uy, self.uz = self._momentum(p0, prev)
        self.dissipated = 0.0
        self.injected = 0.0
        self._pending.clear()
        self._pending_ports.clear()
        for port in self._ports:
            port._queued_at = -1  # a fresh run: no port owes this room an injection
        self.n = 0

    # -- time stepping --------------------------------------------------------------------

    def _momentum(
        self, p: NDArray[np.float64], u_prev: Sequence[NDArray[np.float64]]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """One momentum half-step ``u^{+1/2} = u^{-1/2} - (k / (rho0 h)) grad p``, per axis."""
        c = self.k / (self.rho0 * self.h)
        return (
            u_prev[0] - c * np.diff(p, axis=0),
            u_prev[1] - c * np.diff(p, axis=1),
            u_prev[2] - c * np.diff(p, axis=2),
        )

    def _divergence(self) -> NDArray[np.float64]:
        """Discrete divergence at every node — the **transpose** of the momentum gradient.

        Each node accumulates the face velocities on either side of it (a wall node sees only the
        one face it has, which *is* the rigid closure: no ghost velocity), then divides by the
        **per-direction** weight ``w_d``, not the volume weight ``W``.
        """
        dx = np.zeros_like(self.p)
        dx[:-1, :, :] += self.ux
        dx[1:, :, :] -= self.ux
        dy = np.zeros_like(self.p)
        dy[:, :-1, :] += self.uy
        dy[:, 1:, :] -= self.uy
        dz = np.zeros_like(self.p)
        dz[:, :, :-1] += self.uz
        dz[:, :, 1:] -= self.uz
        wx, wy, wz = self._w
        return dx / wx[:, None, None] + dy / wy[None, :, None] + dz / wz[None, None, :]

    def inject(self, q: float, at: Sequence[float] | None = None) -> None:
        """Queue a soft point injection of volume velocity ``q`` (m^3/s) for the next :meth:`step`.

        ``at`` overrides the constructor's source point for this one injection (snapped to the
        nearest node); repeated calls before a ``step`` accumulate. Volume velocity is the
        primitive because it is what the continuity equation's source term *is*. The lumped tier's
        ``_VolumeAccelerationSource`` protocol hands out ``Q''`` instead, so drive the box from
        :attr:`~physsynth.core.radiation.ReactiveRadiatedBody.volume_velocity` — the exact quantity,
        already public. Do **not** integrate ``Q''`` here to fake the protocol: an accumulating
        integrator has a DC drift mode with nothing to restore it, and it would be a silent one.
        """
        index = self.source_index if at is None else self.node_index(at)
        self._pending.append((index, float(q)))

    def step(self) -> None:
        """Advance one timestep: pressure (plus source and walls) first, then velocity.

        The ordering is :class:`~physsynth.core.bore.Bore`'s, and it is load-bearing: afterwards the
        object holds ``p^{n+1}`` alongside ``u^{n+3/2}`` and ``u^{n+1/2}``, so :meth:`energy` is a
        pure function of the stored state rather than a snapshot taken mid-step. The wall closure
        must be applied to the new pressure **before** the momentum sub-step reads it — the same
        reason the bore's radiating drain lives inside ``step()``.
        """
        p_old = self.p
        p_next = p_old - self.k * self.rho0 * self.c0**2 * self._divergence()

        if self._pending or self._pending_ports:
            gain = self.k * self.rho0 * self.c0**2
            for index, q in self._pending:
                p_next[index] += gain * q / self._W[index]
            # Ports inject the same soft source term, vectorized over their node set: node n takes
            # the share w_n of the volume velocity U. Disjointness makes the plain += exact (no two
            # ports touch a node, and a port's own node list has no repeats).
            for nodes, w, q in self._pending_ports:
                p_next[nodes] += gain * q * w / self._W[nodes]

        if self._has_walls:
            p_next = (p_next - self._beta * p_old) / (1.0 + self._beta)
            p_next[self._open] = 0.0
            # Book the wall flux at the centered pressure: k * A * pbar^2 / Z >= 0, per face. An
            # edge node pays into every lossy face it belongs to, which is exactly what summing
            # admittances into beta already charged it.
            pbar = 0.5 * (p_next + p_old)
            for index, area, Z in self._lossy:
                self.dissipated += self.k * float(np.sum(area * pbar[index] ** 2)) / Z

        if self._pending or self._pending_ports:
            for index, q in self._pending:
                self.injected += self.k * 0.5 * (p_next[index] + p_old[index]) * q
            # Booked from the room's OWN post-closure pressure, never from a number the port hands
            # back. The port books k pbar_predicted U; these two agree only if R_room is exactly
            # right, so their difference is what the conserved total is watching (§6.1).
            for nodes, w, q in self._pending_ports:
                pbar_port = float(np.sum(w * 0.5 * (p_next[nodes] + p_old[nodes])))
                self.injected += self.k * pbar_port * q
            self._pending.clear()
            self._pending_ports.clear()

        u_next = self._momentum(p_next, (self.ux, self.uy, self.uz))
        self.ux_prev, self.uy_prev, self.uz_prev = self.ux, self.uy, self.uz
        self.ux, self.uy, self.uz = u_next
        self.p = p_next
        self.n += 1

    # -- geometry / read-out --------------------------------------------------------------

    def node_index(self, point: Sequence[float]) -> tuple[int, int, int]:
        """Index of the grid node nearest ``point`` (m).

        Rounding to the nearest node is the resolution and is reported (see :meth:`snapped`), but a
        point **outside** the room is a caller bug, not a resolution question — relocating a source
        or a listener onto a wall silently would be the worst kind of quiet wrong answer, so it
        raises instead.
        """
        point = tuple(float(v) for v in point)
        if len(point) != 3:
            raise ValueError(f"point must be an (x, y, z) triple, got {point!r}.")
        index = tuple(int(round(v / self.h)) for v in point)
        if any(not 0 <= i <= n for i, n in zip(index, self.N, strict=True)):
            raise ValueError(
                f"point {point} lies outside the room (0, 0, 0)..{self.L_actual} m. Nearest node "
                f"index would be {index}, valid range 0..{self.N} per axis."
            )
        return index  # type: ignore[return-value]

    def snapped(self, point: Sequence[float]) -> tuple[float, float, float]:
        """The node coordinate ``point`` actually lands on (m) — the snap, made visible.

        **This is the radius a closed form must be evaluated at.** Two nearby listeners can collapse
        onto the same node, and the snap error is up to ``h/2`` (~10% of ``r`` at ``r = 5h``). An
        oracle that compares against the *requested* ``r`` silently charges that error to the
        physics.
        """
        return tuple(i * self.h for i in self.node_index(point))  # type: ignore[return-value]

    def pressure_at(self, point: Sequence[float]) -> float:
        """Pressure (Pa) at the node nearest ``point`` — a microphone. See :meth:`snapped`."""
        return float(self.p[self.node_index(point)])

    @property
    def state(self) -> NDArray[np.float64]:
        """Current pressure field ``p^n`` (a copy, safe to store for plotting)."""
        return self.p.copy()

    # -- energy ---------------------------------------------------------------------------

    def acoustic_energy(self) -> float:
        """Energy **stored in the air** (Joules): compliance ``p^2`` plus the **cross-time**
        inductive ``u^{n+1/2} u^{n-1/2}`` term. This is the quantity that *decreases* as absorbing
        walls take sound out of the room; it is not the conserved one (see :meth:`energy`)."""
        pot = 0.5 * float(np.sum(self._W * self.p * self.p)) / (self.rho0 * self.c0**2)
        kin = 0.5 * self.rho0 * (
            float(np.sum(self._Wx * self.ux * self.ux_prev))
            + float(np.sum(self._Wy * self.uy * self.uy_prev))
            + float(np.sum(self._Wz * self.uz * self.uz_prev))
        )
        return pot + kin

    def dissipated_energy(self) -> float:
        """Cumulative energy absorbed by the impedance walls (Joules, monotone non-decreasing)."""
        return self.dissipated

    def injected_energy(self) -> float:
        """Cumulative work ``integral p q dt`` done on the room by the soft source (Joules)."""
        return self.injected

    def energy(self) -> float:
        """The **conserved** total ``acoustic + dissipated - injected`` (Joules).

        Flat to machine precision for any walls and any source: the wall channel captures exactly
        what the room sheds and the source channel exactly what it is handed, so a drift here is a
        real bug in the scheme (or in the boundary handling — suspect that first). With rigid walls
        and no source this is bit-for-bit :meth:`acoustic_energy`.
        """
        return self.acoustic_energy() + self.dissipated - self.injected

    # -- the modal oracle (closed-form knowledge, not I/O) ---------------------------------

    def mode_shape(self, l: int, m: int, n: int) -> NDArray[np.float64]:
        """The rigid-room mode ``cos(l pi i/Nx) cos(m pi j/Ny) cos(n pi k/Nz)`` on the grid.

        This tensor cosine is an **exact** eigenvector of the discrete Neumann Laplacian *including
        at the h/2 wall nodes* — nothing is staircased in a grid-aligned rectangle — which puts the
        modal oracle a tier above the membrane's Bessel test (convergence-rate only). Valid for
        ``0 <= l <= Nx`` etc.; meaningful only for rigid walls.
        """
        idx = self._mode_indices(l, m, n)
        cos = [np.cos(q * np.pi * np.arange(N + 1) / N) for q, N in zip(idx, self.N, strict=True)]
        return cos[0][:, None, None] * cos[1][None, :, None] * cos[2][None, None, :]

    def mode_frequency(self, l: int, m: int, n: int) -> float:
        """The **exact discrete** frequency (Hz) of mode ``(l, m, n)`` — what the scheme *will* do.

        The spatial eigenvalue is ``-mu^2`` with
        ``mu^2 = (4/h^2) sum_d sin^2(l_d pi / (2 N_d))``, and the leapfrog turns that into
        ``omega_d = (2/k) arcsin(c0 k mu / 2)``. The arcsin argument is ``lambda mu h / 2 <= lambda
        sqrt(3) <= 1``, so the CFL is exactly the condition that every mode stays real (oscillatory)
        rather than growing — the ceiling is reached by the corner mode, and reached *exactly* at
        ``lambda = 1/sqrt(3)``, where that mode turns defective (see the module docstring).
        """
        mu = np.sqrt(self._mu_squared(l, m, n))
        omega = (2.0 / self.k) * np.arcsin(np.clip(self.c0 * self.k * mu / 2.0, -1.0, 1.0))
        return float(omega / (2.0 * np.pi))

    def continuum_mode_frequency(self, l: int, m: int, n: int) -> float:
        """The **textbook** rigid rectangular-room frequency (Hz),
        ``f = (c0/2) sqrt((l/Lx)^2 + (m/Ly)^2 + (n/Lz)^2)``, evaluated on :attr:`L_actual`.

        :meth:`mode_frequency` converges to this at second order in ``h``; the gap between them is
        the scheme's numerical dispersion, which in 3-D never vanishes at any ``lambda``.
        """
        idx = self._mode_indices(l, m, n)
        s = sum((q / Ld) ** 2 for q, Ld in zip(idx, self.L_actual, strict=True))
        return float(0.5 * self.c0 * np.sqrt(s))

    def set_mode(self, l: int, m: int, n: int, amplitude: float = 1.0) -> float:
        """Initialise the room in the exact discrete mode ``(l, m, n)``; return its frequency (Hz).

        The field then evolves as ``amplitude * cos(2 pi f n k) * mode_shape(l, m, n)`` to machine
        precision, with ``f`` the :meth:`mode_frequency`. The half-step-back velocity that makes
        that exact is ``u^{-1/2} = (k / (2 rho0 h)) diff(p^0)`` — note it is **omega-free**. The
        plausible-looking continuum form ``sin(omega k/2)/(rho0 omega h)`` is *nearly* right and
        leaves a shape error that masquerades as scheme inaccuracy; deriving the exact one is the
        derivation of the dispersion relation above.
        """
        p0 = amplitude * self.mode_shape(l, m, n)
        s = self.k / (2.0 * self.rho0 * self.h)
        u0 = (s * np.diff(p0, axis=0), s * np.diff(p0, axis=1), s * np.diff(p0, axis=2))
        self.set_state(p0, u0)
        return self.mode_frequency(l, m, n)

    def _mode_indices(self, l: int, m: int, n: int) -> tuple[int, int, int]:
        idx = (int(l), int(m), int(n))
        for q, N, axis in zip(idx, self.N, _AXES, strict=True):
            if not 0 <= q <= N:
                raise ValueError(f"mode index {axis}={q} out of range 0..{N}.")
        return idx

    def _mu_squared(self, l: int, m: int, n: int) -> float:
        idx = self._mode_indices(l, m, n)
        s = sum(np.sin(q * np.pi / (2 * N)) ** 2 for q, N in zip(idx, self.N, strict=True))
        return float(4.0 * s / (self.h * self.h))


class RoomPort:
    """One two-way terminal into an :class:`AirBox` — the room's **Thevenin equivalent** at a spot.

    A port is a set of grid nodes with normalized volume weights ``w_n`` (summing to 1). It answers
    exactly one question, and answers it as a *linear scalar relation*: if a volume velocity ``q``
    is injected here this step, what centered pressure ``pbar`` will the injector feel?

        ``pbar = pbar_free + R_room q``

    :meth:`free_pressure` is ``pbar_free``, the open-circuit term — what the port would feel with
    ``q = 0``, carrying the room's entire history *including every reflection off every wall*.
    :attr:`R_room` is the constant internal resistance. Together they are everything a coupled
    solver needs, which is why the body's load collapses to one division
    (:class:`RoomLoadedBody`) instead of a field solve.

    **Why it is only two numbers.** Within a single step an injection changes the pressure at its
    own nodes and nowhere else: :meth:`AirBox.step` adds the source term node-locally to
    ``p^{n+1}``, and propagation to the neighbours waits for the *next* momentum sub-step. So the
    room is linear-in-``q`` with a diagonal instantaneous response, and

        ``R_room = sum_n w_n^2 k rho0 c0^2 / (2 W_n (1 + beta_n))``

    where the ``2`` is the centered (trapezoidal) pressure, ``W_n`` the tensor-trapezoid node
    weight, and ``(1 + beta_n)`` the **wall closure denominator**. That last factor is easy to omit
    — the naive ``k rho0 c0^2 / (2 W)`` is right at every interior node — and omitting it costs
    nothing until someone mounts a port on a lossy wall, where it leaks ~2% of the run's energy
    (measured 1.9e-2 against 8.4e-15 with the factor). ``step()`` injects *before* it closes the
    wall, so the injection is divided by ``1 + beta`` along with everything else, and ``R_room``
    must say so.

    **Point port versus spread port — a measured non-convergence.** ``radius=None`` puts the whole
    volume velocity on one node. That is exact and perfectly conservative, and its *magnitude* is a
    property of the grid rather than of the physics: the port behaves as a pulsating sphere of
    equivalent radius ``~ h / 3.1`` (measured ``a_eff / h`` = 0.324, 0.320, 0.317 on three grids),
    which at low frequency is an added **mass** loading the body. Refining ``h`` therefore makes the
    artifact *worse*, not better — it halves the equivalent radius and doubles the load (measured
    ratios 0.493 and 0.496 across two halvings). A ball of fixed ``radius`` does not move: 1.045 and
    1.038 over the same halvings, a factor of twenty less grid-sensitive. **So use a point port for
    structural tests, where only exactness matters, and a spread port whenever the magnitude of the
    coupling is meant to mean something.** ``radius`` has no default so that choice is always made,
    never inherited.

    A spread port's absolute size is the **uniformly injecting ball**, whose equivalent shell radius
    is ``5a/6`` — the classic **6/5** shape factor, the same one as the mean potential of a
    uniformly charged sphere — and *not* a pulsating shell of radius ``a``. Confirmed to 0.3% in a
    room large enough for the port to be compact. Read that caveat as a real one: the same
    measurement in a room only 10x the port reads 8.6% high, and that excess is the **room's** own
    reactance rather than the port's (measured ratios to ``5a/6`` of 1.086, 1.040, 1.003, 0.977 for
    rooms of 0.5, 0.7, 1.0 and 1.4 m). The number is insensitive to the Courant number to five
    significant figures, so it is a static near-field quantity and not a dispersion artifact.

    Parameters
    ----------
    room : AirBox
        The room to open a port into. The port registers itself with the room for the life of the
        room (used for the disjointness check), so build a fresh room rather than reusing one whose
        ports you have discarded.
    at : (float, float, float)
        Port centre (m), snapped to the nearest node (see :meth:`AirBox.snapped`).
    radius : float or None
        Ball radius (m) around ``at``, or ``None`` for a single-node point port. **Required.**
        Nodes outside the room are clipped, so a port near a wall is smaller and one-sided —
        physical, and reported by :attr:`node_count`.

    Raises
    ------
    ValueError
        Centre outside the room; a non-positive ``radius``; a ``radius`` too small for the grid to
        resolve (it would silently be a point port); a port touching an ``"open"`` face (which can
        do no work at all — see below); or a port sharing a node with an existing port.

    Notes
    -----
    **An open face is refused, and the energy report cannot catch why.** An ``"open"`` (``Z = 0``)
    face pins ``p = 0``, so a port on it has ``pbar_free = 0`` and ``R_room = 0``: the body radiates
    into a short circuit. Measured over 400 steps, ``injected`` and ``acoustic`` are *exactly* zero
    and the conserved total drifts 8.6e-15. The run is perfectly conservative and completely silent
    — the physics is exactly right and exactly useless — so this project's primary bug detector is
    structurally blind to it and a construction-time refusal is the only place to catch it.

    **Overlap is refused for a reason that is also the reason N instruments work.** Two ports
    sharing a node are not independent within a step: A's injection changes B's ``pbar``, so B
    solved against a pressure that never occurred and the two ledgers stop matching (measured 3.6e-2
    drift, against 7.1e-15 for disjoint ports in the same room). Disjointness is exactly the
    condition that makes the cheap per-port scalar solve *exact*. A simultaneous ``(I + G R) U``
    solve with the cross-resistance matrix would handle overlap and is passive too, but it needs one
    central object owning every port — which is precisely what makes a per-instrument
    :class:`~physsynth.core.connection.StringBodyBridge` chain impossible. The hazard is real
    because of **snapping**: two ports 5 mm apart on a 13.5 mm grid are one port.

    **Do not mix :meth:`AirBox.inject` with ports** in a run whose energy you intend to assert on.
    That source's work lands in the room's ``injected`` book with no port ledger to cancel it, so
    ``sum_j inst_j.energy() + room.energy()`` is no longer the conserved total (it picks up the
    source's contribution). Ports and the read-out source are both fine, just not in one ledger.
    """

    def __init__(self, *, room: AirBox, at: Sequence[float], radius: float | None) -> None:
        self.room = room
        self.index = room.node_index(at)
        self.radius = None if radius is None else float(radius)

        if self.radius is None:
            nodes = tuple(np.array([i], dtype=np.intp) for i in self.index)
        else:
            if self.radius <= 0.0 or not np.isfinite(self.radius):
                raise ValueError(f"port radius must be a positive length, got {radius!r}.")
            offs = [
                room.h * (np.arange(n + 1) - c)
                for n, c in zip(room.N, self.index, strict=True)
            ]
            d2 = (
                offs[0][:, None, None] ** 2
                + offs[1][None, :, None] ** 2
                + offs[2][None, None, :] ** 2
            )
            nodes = np.nonzero(d2 <= self.radius * self.radius)
            if nodes[0].size == 1:
                raise ValueError(
                    f"port radius {self.radius} m is smaller than the grid can resolve (h = "
                    f"{room.h}): the ball contains only the centre node {self.index}, so this "
                    "would silently be a point port with a grid-dependent load magnitude. Coarsen "
                    "the request, refine h, or pass radius=None to ask for a point port on purpose."
                )
        self.nodes: tuple[NDArray[np.intp], ...] = tuple(nodes)  # type: ignore[assignment]

        self._check_open_faces()
        self._flat = np.ravel_multi_index(self.nodes, room.p.shape)
        self._check_disjoint()

        W = room._W[self.nodes]
        self.w: NDArray[np.float64] = W / W.sum()
        self.R_room = float(
            np.sum(
                self.w * self.w * room.k * room.rho0 * room.c0**2
                / (2.0 * W * (1.0 + room._beta[self.nodes]))
            )
        )
        self._queued_at = -1  # room.n at which this port last queued an injection
        room._ports.append(self)

    # -- construction refusals -------------------------------------------------------------

    def _check_open_faces(self) -> None:
        """Refuse a port that touches a pressure-release face — it can do no work (see Notes)."""
        room = self.room
        if not np.any(room._open):
            return
        touched = [
            face
            for face in FACES
            if room.walls[face] == 0.0
            and np.any(
                self.nodes[_AXES.index(face[0])]
                == (0 if face[1] == "0" else room.N[_AXES.index(face[0])])
            )
        ]
        if touched:
            raise ValueError(
                f"port at {self.index} touches the open (pressure-release) face(s) {touched}, "
                "where p is pinned to 0: pbar_free and R_room are both exactly zero, so the body "
                "would radiate into a short circuit — perfectly conservative, perfectly silent, "
                "and invisible to the energy report. Move the port off that face, or give the face "
                "a finite impedance."
            )

    def _check_disjoint(self) -> None:
        """Refuse a port sharing any node with an existing one (see Notes)."""
        for other in self.room._ports:
            shared = np.intersect1d(self._flat, other._flat)
            if shared.size:
                node = np.unravel_index(int(shared[0]), self.room.p.shape)
                raise ValueError(
                    f"port at {self.index} shares node {node} with the existing port at "
                    f"{other.index} ({shared.size} node(s) in common). Overlapping ports are not "
                    "independent within a step, so each one's solve uses a pressure that never "
                    "occurred and the energy ledgers stop matching. Note grid snapping: two nearby "
                    f"centres collapse onto one node at h = {self.room.h}."
                )

    # -- the two numbers -------------------------------------------------------------------

    @property
    def node_count(self) -> int:
        """How many grid nodes the port actually covers (clipping at walls included)."""
        return int(self.nodes[0].size)

    @property
    def volume(self) -> float:
        """The port's discrete volume ``sum_n W_n`` (m^3) — the staircased ball, made visible.

        Compare against ``4 pi a^3 / 3``: the wobble as whole nodes fall in or out of the ball is
        the membrane batch's staircase and it is this port's accuracy floor (measured -4.6%, +1.5%
        on two grids), not a defect of the scheme.
        """
        return float(np.sum(self.room._W[self.nodes]))

    def free_pressure(self) -> float:
        """The open-circuit centered pressure ``pbar_free`` this port would feel with ``q = 0``.

        ``O(port nodes)``: the divergence at a node needs only the six faces touching it, so this
        never touches the whole field. It replicates :meth:`AirBox.step`'s order exactly —
        divergence, **then** the wall closure — and must be read *before* ``room.step()``, from the
        stored ``u^{n+1/2}``.

        It deliberately ignores other ports' queued injections. For **disjoint** ports that is not
        an approximation: a queued injection at another node cannot reach this node within the step,
        so including it would change nothing — while for overlapping ports (refused at construction)
        reading it would make the solve *asymmetric* rather than merely wrong.
        """
        room = self.room
        ix, iy, iz = self.nodes
        div = np.zeros(ix.size)
        for axis, (u, idx) in enumerate(
            ((room.ux, ix), (room.uy, iy), (room.uz, iz))
        ):
            n_face = room.N[axis]
            pick = [ix, iy, iz]
            pick[axis] = np.minimum(idx, n_face - 1)
            plus = np.where(idx < n_face, u[tuple(pick)], 0.0)
            pick[axis] = np.maximum(idx - 1, 0)
            minus = np.where(idx > 0, u[tuple(pick)], 0.0)
            div += (plus - minus) / room._w[axis][idx]

        p_node = room.p[self.nodes]
        p_free = p_node - room.k * room.rho0 * room.c0**2 * div
        if room._has_walls:
            beta = room._beta[self.nodes]
            p_free = (p_free - beta * p_node) / (1.0 + beta)
        return float(np.sum(self.w * 0.5 * (p_free + p_node)))

    # -- driving ---------------------------------------------------------------------------

    def require_ready(self) -> None:
        """Raise if this port's previous injection is still pending — i.e. no ``room.step()``.

        The check is **per-port**, keyed on the room's step counter, not on whether the room has
        anything queued: with several instruments in one room, every port after the first solves
        while earlier ports' injections sit queued, and a global "is anything pending" test would
        fire on all of them.
        """
        if self._queued_at == self.room.n:
            raise RuntimeError(
                f"port at {self.index} was asked to solve twice within one room step (room.n = "
                f"{self.room.n}). A port does not step its room — the caller does, once, after "
                "every port has solved:  for inst in instruments: inst.step(...)  then  "
                "room.step(). Without it the room is frozen and the body is loaded by a stale "
                "field, silently."
            )

    def inject(self, q: float) -> None:
        """Queue this port's volume velocity ``q`` (m^3/s) for the room's next :meth:`AirBox.step`.

        Node ``n`` receives the share ``w_n q``. The room books the work at its own post-closure
        centered pressure — never at a number handed back from here, which is what keeps the two
        ledgers an independent check on each other rather than a tautology.
        """
        self.require_ready()
        self.room._pending_ports.append((self.nodes, self.w, float(q)))
        self._queued_at = self.room.n

    def reset(self) -> None:
        """Forget any pending-injection mark — for reusing the port on a fresh run."""
        self._queued_at = -1


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
