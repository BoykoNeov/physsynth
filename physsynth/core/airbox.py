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
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.sparse.linalg import splu

if TYPE_CHECKING:  # type-only: the room stays free of any dependency on the resonator modules
    from physsynth.core.body import ModalBody
    from physsynth.core.plate import Plate

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
]

# Ambient air (matches physsynth.core.radiation and .bore so every tier of the air node agrees).
RHO0_AIR = 1.2041  # kg/m^3
C0_AIR = 343.0     # m/s

# The six faces of the box, named <axis><end>: "x0" is the x = 0 wall, "x1" the x = Lx wall.
FACES = ("x0", "x1", "y0", "y1", "z0", "z1")

# The three interior plane orientations, named by their normal axis (see AirBox.add_cut). A wall is
# a FACE and has an end; an interior plane has neither, which is why the two vocabularies differ.
PLANES = ("x", "y", "z")

_AXES = "xyz"
_LAMBDA_MAX = 1.0 / np.sqrt(3.0)  # 3-D CFL ceiling
_LAMBDA_TOL = 1e-12

# How a surface node's area is handed to the air grid (see SurfacePort).
Spreading = Literal["bilinear", "nearest"]
_SPREADINGS = ("bilinear", "nearest")


def _face_axes(face: str) -> tuple[int, int, int, int]:
    """``(normal axis, end, in-plane axis 0, in-plane axis 1)`` for a face name.

    The two in-plane axes are taken in increasing order — ``x0`` spans ``(y, z)``, ``y0`` spans
    ``(x, z)``, ``z0`` spans ``(x, y)`` — the same ordering :meth:`AirBox._build_wall_closure` uses
    for its transverse area weights. **No axis is mirrored on a high face**: the inward normal is
    carried by the sign convention (see :class:`SurfacePort`), never by flipping a coordinate.
    """
    if face not in FACES:
        raise ValueError(f"unknown face {face!r}; expected one of {FACES}.")
    axis = _AXES.index(face[0])
    t0, t1 = (a for a in range(3) if a != axis)
    return axis, int(face[1]), t0, t1


def _free_pressure_nodes(
    room: AirBox, nodes: tuple[NDArray[np.intp], ...]
) -> NDArray[np.float64]:
    """Per-node open-circuit centered pressure ``pbar_free`` at ``nodes`` — ``O(patch)``.

    Shared verbatim by :class:`RoomPort` (which contracts it with its weights) and
    :class:`SurfacePort` (which returns the vector). It replicates :meth:`AirBox.step`'s order
    exactly — divergence, **then** the wall closure — and must be read *before* ``room.step()``,
    from the stored ``u^{n+1/2}``.

    The local divergence read is bit-identical to the full-array :meth:`AirBox._divergence`
    followed by the closure, on low **and** high faces alike (the ``idx < N`` / ``idx > 0`` branches
    are different code paths and both are exercised), for every wall type *except* an ``open``
    face — where :meth:`AirBox.step` pins ``p = 0`` and this read does not. Ports touching an open
    face are refused at construction, so the identity holds by construction rather than by luck.
    """
    ix, iy, iz = nodes
    div = np.zeros(ix.size)
    for axis, (u, idx) in enumerate(((room.ux, ix), (room.uy, iy), (room.uz, iz))):
        n_face = room.N[axis]
        pick = [ix, iy, iz]
        pick[axis] = np.minimum(idx, n_face - 1)
        plus = np.where(idx < n_face, u[tuple(pick)], 0.0)
        pick[axis] = np.maximum(idx - 1, 0)
        minus = np.where(idx > 0, u[tuple(pick)], 0.0)
        div += (plus - minus) / room._w[axis][idx]

    p_node = room.p[nodes]
    p_free = p_node - room.k * room.rho0 * room.c0**2 * div
    if room._has_walls:
        beta = room._beta[nodes]
        p_free = (p_free - beta * p_node) / (1.0 + beta)
    return 0.5 * (p_free + p_node)


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
        # (fancy-index triple, per-node volume weights, volume velocity U), and every use of the
        # pair is LINEAR in `w` -- the injection is `w * U` and the read-back `w . pbar`. The
        # weights need not be normalized: a RoomPort passes a unit-sum `w` with the scalar `U`, and
        # a SurfacePort passes the per-node volume-velocity VECTOR itself with `U = 1.0`. Both come
        # out exactly right, which is why the distributed tier needs no AirBox edit at all.
        self._pending_ports: list[
            tuple[tuple[NDArray[np.intp], ...], NDArray[np.float64], float]
        ] = []
        # Live ports, for the disjointness check and the set_state reset. Every port tier registers
        # here (they share the `_flat` / `index` / `_queued_at` protocol the checks read).
        self._ports: list[RoomPort | SurfacePort | InteriorSurfacePort] = []
        # Internal cuts (batch 4). `_cut_mask[axis]` is the accumulated boolean face mask and
        # `_cut_index[axis]` the fancy index it holds -- the mask is the bookkeeping, the index is
        # the hot path (an O(patch) assignment per step instead of an O(faces) mask write).
        # `_cuts` records who owns which faces, which is what makes the cut ADDITIVE rather than
        # single-slot: a second plate must not be able to un-block the first (see add_cut).
        self._cut_mask: list[NDArray[np.bool_] | None] = [None, None, None]
        self._cut_index: list[tuple[NDArray[np.intp], ...] | None] = [None, None, None]
        self._cuts: list[tuple[object, int, NDArray[np.intp]]] = []

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
        """One momentum half-step ``u^{+1/2} = u^{-1/2} - (k / (rho0 h)) grad p``, per axis.

        **Cut faces are zeroed here**, which is the whole implementation of :meth:`add_cut`. This is
        the single place both :meth:`step` and :meth:`set_state` produce velocities, so a cut room
        can never hold a live velocity on a cut face at *any* half-step — including the consistent
        start, where getting it wrong would seed a one-off transient that reads as a scheme error.
        """
        c = self.k / (self.rho0 * self.h)
        u = (
            u_prev[0] - c * np.diff(p, axis=0),
            u_prev[1] - c * np.diff(p, axis=1),
            u_prev[2] - c * np.diff(p, axis=2),
        )
        for axis, idx in enumerate(self._cut_index):
            if idx is not None:
                u[axis][idx] = 0.0  # O(cut faces), not O(faces): a precomputed fancy index
        return u

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

    # -- internal boundaries: the cut (batch 4) ---------------------------------------------

    def add_cut(
        self,
        plane: str,
        index: int,
        extent: Sequence[Sequence[int]] | None = None,
    ) -> None:
        """Add a rigid, zero-thickness internal partition on a plane of velocity **faces**.

        The plate hanging *in* a room (:class:`RoomSuspendedPlate`) is a source **and** an obstacle,
        and the obstacle half is this: zero ``u`` on a set of faces and the air on either side is
        disconnected there. It is the only new primitive batch 4 needs — prescribing a face velocity
        and injecting a ``-q``/``+q`` pair on the two node planes that straddle it are the *same
        arithmetic* (see :class:`InteriorSurfacePort`), so the port, the injection weights and the
        ``injected`` ledger are all batch 3's, unchanged.

        Public in its own right, because a rigid partition is a physical object — a room divider, a
        duct termination — and because the sub-room modal oracle needs one without a port (a legal
        port can never span a full cross-section; see :class:`InteriorSurfacePort`'s rim refusal).

        Parameters
        ----------
        plane : {"x", "y", "z"}
            The cut's **normal** axis. ``plane="z", index=m`` cuts ``uz[:, :, m]`` — the faces
            between node planes ``m`` and ``m+1``.
        index : int
            Face index along ``plane``, ``0 .. N-1``. The room's own walls live at *node* planes
            ``0`` and ``N`` and are already rigid, so they are not cut positions.
        extent : ((lo0, hi0), (lo1, hi1)), optional
            **Inclusive** node-index ranges restricting the cut on the two in-plane axes, taken in
            increasing axis order (a ``"z"`` cut spans ``(x, y)``). Default: the full cross-section,
            which splits the room in two.

        Notes
        -----
        **It costs nothing to book.** A cut face's ``u`` is identically zero at every half-step, so
        its contribution to the kinetic sum ``1/2 rho0 sum_f W_f u^{n+1/2} u^{n-1/2}`` is
        identically zero: the energy identity needs no new term and no exclusion list — the faces
        remove themselves. The potential/kinetic telescoping is untouched too, because it only ever
        used the divergence's being the transpose of the gradient, which zeroing does not disturb.

        **The cut is additive, and that is a design decision rather than an implementation detail.**
        A single-slot ``room._cut = ...`` is overwritten by the next port, and the measured failure
        is silent: the first plate keeps injecting its ``-q``/``+q`` pair and stops blocking, i.e.
        it degrades to a transparent doublet whose moment vanishes under grid refinement — a 40x
        error at 3x refinement, with every ledger green. So cuts accumulate, and a patch that would
        share faces with a **port**'s cut is refused.

        **A full cut has an exact modal oracle, and it is a new one.** Cutting the whole
        cross-section at face ``m`` makes two independent rooms of length ``(m + 1/2) h`` and
        ``(N - m - 1/2) h`` — summing to ``N h`` exactly, since the cut lies half a cell past the
        last node on each side. That end is **face-centered** (the mirror plane sits *between*
        nodes, ghost condition ``p_{m+1} = p_m``), so the exact discrete eigenvector along the cut
        axis is ``cos(n pi i / (m + 1/2))`` and **not** the room's own ``cos(n pi i / N)``.
        """
        axis = self._plane_axis(plane)
        n_face = self.N[axis]
        if not 0 <= int(index) <= n_face - 1:
            raise ValueError(
                f"cut index {index} is out of range for plane {plane!r}: the room has {n_face} "
                f"face(s) there, so a cut sits at index 0..{n_face - 1} (face i lies between node "
                f"planes i and i+1). The room's own walls are at NODE planes 0 and "
                f"{n_face} and are already rigid — they are not cut positions."
            )
        t0, t1 = (a for a in range(3) if a != axis)
        sel = []
        for d, ax in enumerate((t0, t1)):
            n_node = self.N[ax]
            if extent is None:
                sel.append(np.arange(n_node + 1, dtype=np.intp))
                continue
            try:
                lo, hi = (int(v) for v in extent[d])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "extent must be a ((lo0, hi0), (lo1, hi1)) pair of inclusive node-index ranges "
                    f"on the in-plane axes {_AXES[t0]!r} and {_AXES[t1]!r}, got {extent!r}."
                ) from exc
            if not 0 <= lo <= hi <= n_node:
                raise ValueError(
                    f"cut extent {lo}..{hi} on axis {_AXES[ax]!r} is not an inclusive node-index "
                    f"range inside 0..{n_node}."
                )
            sel.append(np.arange(lo, hi + 1, dtype=np.intp))
        i0, i1 = (a.ravel() for a in np.meshgrid(sel[0], sel[1], indexing="ij"))
        self._register_cut(None, axis, int(index), i0, i1)

    @property
    def cut_faces(self) -> int:
        """How many velocity faces are currently cut — **reported, not tuned**.

        The blocked area is ``cut_faces * h^2`` for an interior cut (every cut face carries the full
        transverse weight, which is what the in-plane rim refusal guarantees for a port).
        """
        return sum(0 if m is None else int(np.count_nonzero(m)) for m in self._cut_mask)

    def _plane_axis(self, plane: str) -> int:
        if plane not in PLANES:
            raise ValueError(
                f"unknown plane {plane!r}; expected one of {PLANES}. An interior plane is named by "
                f"its normal axis alone — it has no end, unlike a wall face ({FACES})."
            )
        return _AXES.index(plane)

    def _register_cut(
        self,
        owner: object,
        axis: int,
        index: int,
        i0: NDArray[np.intp],
        i1: NDArray[np.intp],
    ) -> None:
        """Cut the faces ``(i0, i1)`` on plane ``axis`` at ``index``, additively — the one writer.

        ``owner`` is the port that owns these faces, or ``None`` for a hand-placed cut. Two
        hand-placed cuts may overlap (the mask is a boolean union, so that is idempotent); anything
        sharing faces with a **port's** cut is refused, because there the cut and the ``-q``/``+q``
        pair are two halves of one object and a shared face makes the pairing ambiguous.
        """
        t0, t1 = (a for a in range(3) if a != axis)
        shape = tuple(n + (0 if a == axis else 1) for a, n in enumerate(self.N))
        full: list[NDArray[np.intp]] = [i0, i0, i0]
        full[axis] = np.full(i0.size, index, dtype=np.intp)
        full[t0], full[t1] = i0, i1
        faces = tuple(full)
        flat = np.ravel_multi_index(faces, shape)

        for other_owner, other_axis, other_flat in self._cuts:
            if other_axis != axis or (owner is None and other_owner is None):
                continue
            shared = np.intersect1d(flat, other_flat)
            if shared.size:
                face = np.unravel_index(int(shared[0]), shape)
                raise ValueError(
                    f"the cut on plane {PLANES[axis]!r} at index {index} shares face {face} "
                    f"with an existing cut ({shared.size} face(s) in common). A port's cut and "
                    "its -q/+q pair are two halves of one object, so sharing faces makes the "
                    "pairing ambiguous: the blocked path belongs to one plate and the injection "
                    "to another, and every ledger stays green while one of them silently stops "
                    "blocking."
                )

        mask = self._cut_mask[axis]
        if mask is None:
            mask = np.zeros(shape, dtype=bool)
            self._cut_mask[axis] = mask
        mask[faces] = True
        self._cut_index[axis] = tuple(a.astype(np.intp) for a in np.nonzero(mask))
        self._cuts.append((owner, axis, flat))
        # A cut face carries no velocity at ANY half-step, so clear the stored pair too: a cut added
        # to a room that is already moving must not leave a live velocity behind on it.
        (self.ux, self.uy, self.uz)[axis][faces] = 0.0
        (self.ux_prev, self.uy_prev, self.uz_prev)[axis][faces] = 0.0

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
    — the naive ``k rho0 c0^2 / (2 W)`` is right at every interior node — and it matters as soon as
    a port is mounted on a lossy wall: ``step()`` injects *before* it closes the wall, so the
    injection is divided by ``1 + beta`` along with everything else, and ``R_room`` must say so.

    **What omitting it costs is NOT an energy leak, and this repo's primary bug detector cannot
    see it.** Corrected on measurement (the area-coupling batch §6.1,
    ``docs/dev/air-box-area-coupling-plan.md``): the body's energy identity telescopes
    to ``-k pbar U`` for *whatever* ``pbar`` was used in the force, and the room's
    identity is exact for *whatever* injection it received, so the
    scene total is the sum of two separately-exact identities and stays flat even when the pressure
    the body was pushed by and the pressure the room developed are different numbers. Measured here
    on batch 3's distributed port, whose ``R_j`` is this same constant per node — the factor dropped
    on a lossy mounting wall (``zeta = 3``, mean ``beta = 0.173``), 300 steps: the scene total
    drifts **4.9e-15, which is SMALLER than the correct run's 2.0e-14** — green either way, and not
    even in the suspicious direction — while ``|radiated - injected|`` goes from **exactly
    0.00e+00** to **18% of the channel** (12.9% on a free plate). A wrong ``R`` does not leak
    energy; it creates it on one side and destroys the same amount on the other. The tests that
    catch it are the *cross-ledger* identity ``radiated == injected`` and the differential
    measurement ``test_R_room_is_what_the_room_does``, never the conserved total.

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
        return float(np.sum(self.w * _free_pressure_nodes(self.room, self.nodes)))

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


class _PatchPort:
    """Shared machinery for the two distributed (surface) port tiers — **not public**.

    :class:`SurfacePort` mounts a surface flush in a **wall**; :class:`InteriorSurfacePort` hangs
    one on an **interior plane of faces**. Everything between the *surface protocol* (nodal
    coordinates plus nodal areas, in the surface's own frame) and the air grid is identical for both
    — the bilinear spreading operator, the footprint / rim / open-face / disjointness refusals, and
    the one-injection-per-room-step guard — so it lives here once. What differs is only *where the
    nodes are*: one node plane on a wall, or two interior node planes straddling a cut.

    Subclasses set :attr:`_where`, a short phrase naming the mounting in error messages (``"face
    'z0'"``, ``"the plane 'z' cross-section at index 4"``), and own their node placement.
    """

    room: AirBox
    spreading: Spreading
    in_plane_axes: tuple[int, int]
    coords: NDArray[np.float64]
    areas: NDArray[np.float64]
    n_surface: int
    origin: tuple[float, float]
    nodes: tuple[NDArray[np.intp], ...]
    index: tuple[int, int, int]
    _where: str
    _flat: NDArray[np.intp]
    _queued_at: int

    # -- construction: the surface protocol -------------------------------------------------

    def _accept_surface(
        self,
        *,
        room: AirBox,
        coords: NDArray[np.float64],
        areas: NDArray[np.float64],
        origin: Sequence[float] | None,
        spreading: Spreading,
        in_plane_axes: tuple[int, int],
        where: str,
    ) -> NDArray[np.float64]:
        """Validate the surface protocol, place it in the plane, and return its ``face_coords``."""
        if spreading not in _SPREADINGS:
            raise ValueError(
                f"unknown spreading {spreading!r}; expected one of {_SPREADINGS}. 'nearest' is the "
                "measured negative control of the symmetry argument, not a configuration."
            )
        self.room = room
        self.spreading = spreading
        self.in_plane_axes = in_plane_axes
        self._where = where
        t0, t1 = in_plane_axes

        coords = np.asarray(coords, dtype=float)
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError(
                "coords must be an (n_surface, 2) array of in-plane node positions (m), got shape "
                f"{coords.shape}."
            )
        areas = np.asarray(areas, dtype=float)
        if areas.shape != (coords.shape[0],):
            raise ValueError(
                f"areas must have shape {(coords.shape[0],)} (one per surface node), got "
                f"{areas.shape}."
            )
        if np.any(areas < 0.0) or not np.all(np.isfinite(areas)):
            raise ValueError("surface node areas must be finite and >= 0 (m^2).")
        self.coords = coords.copy()
        self.areas = areas.copy()
        self.n_surface = int(coords.shape[0])

        extent = (room.N[t0] * room.h, room.N[t1] * room.h)  # the plane's own size (m)
        if origin is None:
            # Centred: the footprint's midpoint lands on the plane's midpoint, so the grid's own
            # mirror maps the surface to itself and S = 2*centre/h is an integer. Not an aesthetic
            # default -- see SurfacePort: it is what makes the LOAD equivariant (measured 1.0e-15 at
            # integral S, 1.6e-01..3.8e-01 elsewhere) and what lets the SCENE be symmetric, and the
            # scene leak is LINEAR in the offset with no tolerance band.
            origin = tuple(
                0.5 * (extent[d] - (coords[:, d].min() + coords[:, d].max())) for d in (0, 1)
            )
        origin = tuple(float(v) for v in origin)
        if len(origin) != 2:
            raise ValueError(
                f"origin must be an (o0, o1) pair in the plane's own axes, got {origin!r}."
            )
        self.origin = origin  # type: ignore[assignment]

        face_coords = coords + np.asarray(origin)  # node positions in the plane's own coordinates
        self._face_coords = face_coords
        tol = 1e-9 * room.h
        for d, (size, ax) in enumerate(zip(extent, (t0, t1), strict=True)):
            lo, hi = float(face_coords[:, d].min()), float(face_coords[:, d].max())
            if lo < -tol or hi > size + tol:
                raise ValueError(
                    f"the surface's footprint spans {lo:.6g}..{hi:.6g} m along {_AXES[ax]}, "
                    f"outside {where}, which is 0..{size:.6g} m there. Move it with origin= "
                    f"(it currently sits at {origin[d]:.6g} m on that axis), or enlarge the room."
                )
        return face_coords

    # -- construction: the spreading operator ----------------------------------------------

    def _spread(
        self, face_coords: NDArray[np.float64]
    ) -> tuple[NDArray[np.intp], NDArray[np.intp], NDArray[np.float64]]:
        """Distribute each surface node's area over the air nodes — the ``T`` entries, unassembled.

        Returns ``(row, col, value)`` in the plane's flat ``i0 * (N1 + 1) + i1`` indexing. Entries
        whose *geometric* weight is exactly zero are dropped (an on-grid surface node genuinely does
        not reach its outboard neighbour); entries whose weight is nonzero are kept even when the
        node's **area** is zero, so a zero-area surface still names the nodes it covers and the
        ``T = 0`` reduction to the bare resonator stays exercisable.
        """
        room = self.room
        t0, t1 = self.in_plane_axes
        n_axis = (room.N[t0], room.N[t1])
        stencil: list[tuple[tuple[NDArray[np.intp], ...], tuple[NDArray[np.float64], ...]]] = []
        for d in (0, 1):
            t = face_coords[:, d] / room.h
            if self.spreading == "nearest":
                i = np.clip(np.rint(t).astype(np.intp), 0, n_axis[d])
                stencil.append(((i,), (np.ones(self.n_surface),)))
            else:
                # floor, with the top edge folded down one cell so the stencil is always {i0, i0+1}
                # and the outboard node carries weight exactly 0 there (never a clipped fold).
                i0 = np.minimum(np.floor(t).astype(np.intp), n_axis[d] - 1)
                f = t - i0
                stencil.append(((i0, i0 + 1), (1.0 - f, f)))

        surf = np.arange(self.n_surface, dtype=np.intp)
        n1 = n_axis[1]
        rows_l, cols_l, vals_l, geo_l = [], [], [], []
        for a0, w0 in zip(stencil[0][0], stencil[0][1], strict=True):
            for a1, w1 in zip(stencil[1][0], stencil[1][1], strict=True):
                w = w0 * w1
                rows_l.append(a0 * (n1 + 1) + a1)
                cols_l.append(surf)
                vals_l.append(self.areas * w)
                geo_l.append(w)
        keep = np.concatenate(geo_l) != 0.0
        return (
            np.concatenate(rows_l)[keep],
            np.concatenate(cols_l)[keep],
            np.concatenate(vals_l)[keep],
        )

    def _plane_nodes(
        self, rows: NDArray[np.intp]
    ) -> tuple[NDArray[np.intp], NDArray[np.intp], NDArray[np.intp]]:
        """Unique in-plane node indices ``(i0, i1)`` of the spread stencil, plus the sorted rows."""
        t1 = self.in_plane_axes[1]
        n1 = self.room.N[t1]
        plane_nodes = np.unique(rows)
        return (
            (plane_nodes // (n1 + 1)).astype(np.intp),
            (plane_nodes % (n1 + 1)).astype(np.intp),
            plane_nodes,
        )

    def _build_T(
        self,
        rows: NDArray[np.intp],
        cols: NDArray[np.intp],
        vals: NDArray[np.float64],
        plane_nodes: NDArray[np.intp],
    ) -> sparse.csr_matrix:
        pos = np.searchsorted(plane_nodes, rows)
        return sparse.coo_matrix(
            (vals, (pos, cols)), shape=(plane_nodes.size, self.n_surface)
        ).tocsr()

    def _check_in_plane_rim(self, i0: NDArray[np.intp], i1: NDArray[np.intp]) -> None:
        """Refuse a stencil reaching the mounting plane's own rim (see :class:`SurfacePort`)."""
        room = self.room
        t0, t1 = self.in_plane_axes
        for idx, ax in ((i0, t0), (i1, t1)):
            lo, hi = int(idx.min()), int(idx.max())
            if lo < 1 or hi > room.N[ax] - 1:
                raise ValueError(
                    f"the surface's spread stencil reaches air node index {lo}..{hi} along "
                    f"{_AXES[ax]} on {self._where}, but a node on the plane's own rim (0 or "
                    f"{room.N[ax]}) touches a SECOND wall: it carries half the node weight W and "
                    "the sum of two wall admittances, so R_j stops being uniform across the patch "
                    "and the spreading operator's reflection equivariance stops holding. Keep the "
                    "footprint plus one air cell strictly inside the plane -- move it with "
                    "origin=, enlarge the room, or shrink the surface."
                )

    # -- construction refusals -------------------------------------------------------------

    def _check_footprint(self, face_coords: NDArray[np.float64]) -> None:
        """Refuse a surface too coarse to feed every air node under it (a comb at the grid scale).

        Counts air nodes inside the footprint's bounding box that no surface node reaches. With
        bilinear spreading this is 0 at every grid ratio measured, so the refusal rarely fires — and
        it still ships, because a surface coarse enough to skip whole air cells is a real thing to
        refuse. Note the condition is a **count of unfed nodes**, not an inequality on
        ``h_surface/h_air``: at ratio 0.909, comfortably inside the naive inequality, nearest-node
        still left half the footprint unfed.
        """
        room = self.room
        t0, t1 = self.in_plane_axes
        tol = 1e-9 * room.h
        inside = []
        for d, ax in enumerate((t0, t1)):
            grid = np.arange(room.N[ax] + 1) * room.h
            lo, hi = face_coords[:, d].min() - tol, face_coords[:, d].max() + tol
            inside.append(np.nonzero((grid >= lo) & (grid <= hi))[0])
        foot = (inside[0][:, None] * (room.N[t1] + 1) + inside[1][None, :]).ravel()
        reached = np.ravel_multi_index(
            (self.nodes[t0], self.nodes[t1]), (room.N[t0] + 1, room.N[t1] + 1)
        )
        # Recorded for the message and for the plan's API surface, but note it can only ever
        # read 0 on a LIVE port: a nonzero count is refused right below. It is a construction-time
        # diagnostic, not a state a caller can observe and act on.
        self.footprint_empty = int(np.setdiff1d(foot, reached).size)
        if self.footprint_empty:
            raise ValueError(
                f"{self.footprint_empty} of {foot.size} air node(s) inside the surface's "
                f"footprint on {self._where} are fed by no surface node, so the acoustic "
                f"source would be a comb at the grid scale. The surface's spacing is too coarse "
                f"for h_air = {room.h:.6g} m (spreading={self.spreading!r}). Refine the surface, "
                "or coarsen the air grid."
            )

    def _check_open_faces(self) -> None:
        """Refuse a surface mounted on a pressure-release face — it can do no work (batch 2's rule).

        An ``open`` (``Z = 0``) face pins ``p = 0``, so ``pbar_free`` and every ``R_j`` are exactly
        zero: the surface radiates into a short circuit, perfectly conservative and completely
        silent, and the energy report is structurally blind to it. There is a **second,
        independent** reason here: :func:`_free_pressure_nodes`' local read does not pin
        ``p = 0`` where :meth:`AirBox.step` does, so a patch touching an open face is the
        one case where the local
        read stops being bit-identical to the full-array closure (measured 4.3e-04, against exactly
        0.0 everywhere else). Refusing makes that identity hold by construction rather than by luck.
        """
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
                f"the surface on {self._where} touches the open (pressure-release) face(s) "
                f"{touched}, where p is pinned to 0: pbar_free and every R_j are exactly zero, so "
                "the surface would radiate into a short circuit -- perfectly conservative, "
                "perfectly silent, and invisible to the energy report. Give that face a finite "
                "impedance, or mount the surface elsewhere."
            )

    def _check_disjoint(self) -> None:
        """Refuse a patch sharing any node with an existing port (:class:`RoomPort`'s rule)."""
        for other in self.room._ports:
            shared = np.intersect1d(self._flat, other._flat)
            if shared.size:
                node = np.unravel_index(int(shared[0]), self.room.p.shape)
                raise ValueError(
                    f"the surface on {self._where} ({self.node_count} nodes) shares node "
                    f"{node} with the existing port at {other.index} ({shared.size} node(s) in "
                    "common). Overlapping ports are not independent within a step, so each one's "
                    "solve uses a pressure that never occurred and the energy ledgers stop "
                    "matching. Note the acoustic source is up to one air cell LARGER than the "
                    "surface itself (bilinear spreads outboard), so footprints that merely look "
                    "separate can still collide."
                )

    # -- geometry / read-out ---------------------------------------------------------------

    @property
    def node_count(self) -> int:
        """How many air nodes the surface's spread source actually covers."""
        return int(self.nodes[0].size)

    @property
    def net_area(self) -> float:
        """The **radiating** area ``sum_n area_n`` (m^2) — which is *not* the bounding rectangle.

        A simply-supported plate's rim nodes are dead: they do not move, so they displace no volume,
        and the total comes out at exactly ``((N-1)/N)^2 Lx Ly`` (measured 0.5625, 0.7656, 0.8403,
        0.8789, 0.9184 at ``N = 4, 8, 12, 16, 24``). That shortfall is **physics, not a defect** —
        but it means any comparison against a closed form for a piston of area ``Lx Ly`` is wrong by
        that factor at coarse ``N``, so the number is reported (:attr:`RoomPort.volume`'s
        precedent). A free plate has no dead rim and comes out at exactly ``Lx Ly``.
        """
        return float(np.sum(self.areas))

    # -- driving ---------------------------------------------------------------------------

    def require_ready(self) -> None:
        """Raise if this port's previous injection is still pending — i.e. no ``room.step()``.

        Per-port and keyed on the room's step counter, exactly as :meth:`RoomPort.require_ready`:
        with several instruments in one room every port after the first solves while earlier ports'
        injections sit queued, so a global "is anything pending" test would fire on all of them.
        """
        if self._queued_at == self.room.n:
            raise RuntimeError(
                f"the surface port on {self._where} was asked to solve twice within one room "
                f"step (room.n = {self.room.n}). A port does not step its room -- the caller does, "
                "once, after every port has solved:  for inst in instruments: inst.step(...)  then "
                " room.step(). Without it the room is frozen and the surface is loaded by a stale "
                "field, silently."
            )

    def reset(self) -> None:
        """Forget any pending-injection mark — for reusing the port on a fresh run."""
        self._queued_at = -1


class SurfacePort(_PatchPort):
    """A **whole moving surface** mounted flush in one wall of an :class:`AirBox` (batch 3).

    :class:`RoomPort` is a *one-port*: it couples through exactly one scalar, the net volume
    velocity ``U = sum_i a_i q_i'``, and hands back exactly one scalar pressure. That is the right
    shape for a :class:`~physsynth.core.body.ModalBody`, whose radiation weights *are* its surface
    collapsed to a point. It is the wrong shape for a **grid** resonator, which has a real surface
    with a real shape on it — and the difference is not a refinement but a whole physical effect
    that every lumped tier in this repo predicts to be **exactly zero**:

        a surface radiates according to the **shape** of its motion, not only its net volume
        displacement — so a mode with zero net volume velocity, which every one-port here calls
        silent, is not.

    That is the **acoustic short circuit**. An even-index mode of a simply supported plate —
    ``(2,1)``, ``(2,2)``, ``(4,2)`` — has ``+`` and ``-`` regions displacing *exactly* cancelling
    volumes (``sum_i sin(m pi i/N) = 0`` identically for even ``m``, so ``U = 0`` to rounding and
    not merely small). What actually happens is that each patch of surface pushes on the air
    *locally*, and the cancellation is only as complete as the acoustic wavelength's ability to
    bridge a ``+`` region and its ``-`` neighbour. Below coincidence the cancellation is nearly
    complete and the radiation comes from the edges; above it the surface pattern is supersonic and
    radiates freely. A one-port has no length scale on its surface, so no ``R(omega)`` reproduces
    this at any order — the same structural argument as batch 2's delayed echo, one tier further in.

    **The room's instantaneous response is diagonal, so the load is ``T^T R T``.** Batch 2 read
    "an injection changes the pressure at its own node and nowhere else within a step" as a Thevenin
    source at *a* node. The stronger reading is that the room's instantaneous response over *any*
    set of nodes is **diagonal** — no cross-resistance at any separation, including one cell
    (measured: off-diagonal exactly ``0.00e+00`` at interior, lossy-wall and lossy-edge patches
    alike). So with

    * ``T`` the **volume-weight matrix** ``(n_patch x n_surface)``, ``q = T v`` in m^3/s,
    * ``R = diag(R_j)``, ``R_j = k rho0 c0^2 / (2 W_j (1 + beta_j))`` — :attr:`RoomPort.R_room`'s
      constant per node, wall-closure denominator included,
    * ``pbar = pbar_free + R q`` a **vector** Thevenin relation,

    reciprocity carries the pressure back as ``f = -T^T pbar`` (the same structural fact
    :class:`~physsynth.core.radiation.RadiatedBody` leans on when it reuses its read-out weights as
    coupling weights), and the unknown part of the load is ``-(1/2k) T^T R T u^{n+1}``: a constant,
    symmetric, positive-semidefinite, sparse matrix times the unknown. It **folds into the
    resonator's own factorization** — nothing new is solved (see :class:`RoomLoadedPlate`). The
    family's ladder, complete: ``RadiatedBody`` scalar ``R`` -> one division; ``RationalAirLoad``
    ``R_eff`` + one aux state -> one division; ``RoomPort`` scalar ``R_room`` with a rich
    ``pbar_free`` -> one division; **this** diagonal ``R`` with a matrix ``T`` -> absorbed into the
    resonator's LU. Batch 2 *is* the rank-1 case: put every surface node on one air node and ``T``
    becomes ``w 1^T``, ``T^T R T`` the rank-1 ``R_room w w^T``, and Sherman-Morrison is what
    inverting ``A + rank-1`` gives instead of refactoring. Passivity is unconditional and for a
    better reason than before — ``T^T R T`` is PSD because ``R_j >= 0``, so the loaded matrix is SPD
    for every timestep, every grid, every patch, and can never be singular.

    **The sign convention is part of the API, and it is local.** Positive surface displacement is
    along the room's **inward normal at** :attr:`face` — ``+z`` for ``z0``, ``-z`` for ``z1``, and
    so on — so an outward-moving surface *compresses* the air at its own face. Hence ``T`` is
    entrywise non-negative on all six walls and **no inward normal appears in the code at all**.
    This is not cosmetic: a consistent sign flip in ``T``/``T^T`` is invisible to every energetic
    quantity in the batch — ``T^T R T`` is sign-invariant, so the load matrix, the solve and
    ``radiated_energy`` all come out **bit-identical** while the room's field is inverted
    (measured). The alternative convention, "positive displacement along the global axis", needs an
    explicit per-face sign, and getting it wrong on three of the six faces is a perfectly passive,
    perfectly conservative, perfectly green simulation of a plate pushing the wrong way. The local
    convention disarms the trap instead of testing around it. (Its only detector is the sign of the
    surface pressure on the **first** step: a six-step read gave the *wrong* answer in the prototype
    because the plate's own half period was five steps.)

    **Bilinear spreading, and the argument that decides it is COVERAGE — the symmetry argument does
    not survive measurement.** ``T`` must (a) conserve volume exactly, (b) be usable as ``T^T`` for
    the pressure, and (c) leave no air node in the footprint unfed. Nearest-node assignment
    satisfies (a) and (b) and fails (c): at ``h_surface/h_air = 1.21`` it leaves footprint nodes
    unfed, which this class refuses. Where it does construct, its interior assigned area per air
    node is lumpy at the grid scale and **does not converge** — measured spread 0.83, 1.03, 0.64,
    0.46 ``h_air^2`` at ``N_plate = 8, 16, 24, 32``, wandering rather than shrinking. Bilinear
    spreading (each surface node's area distributed over the four surrounding air nodes with
    weights summing to 1) leaves **0** unfed at every ratio and is 10x to 100x flatter at every
    one: 0.082, 0.062, 0.051, 0.031 over the same refinements, decreasing. That is what ships, and
    ``spreading="nearest"`` exists **only** as its measured negative control.

    **How flat, exactly — because "partition of unity" promises more than it delivers.** Bilinear's
    interior assignment is ``h_air^2`` *exactly* (measured 5e-16 … 2e-15) when ``h_air/h_surface``
    is an **integer**, and only then. Off an integer it ripples, and not monotonically in the ratio:
    2.93 gives 0.0077 while the finer 4.40 gives 0.0207. Poisson summation says why — the
    periodised hat sum has Fourier coefficients ``sinc^2(pi k h_air/h_surface)``, whose ``k``-th
    term vanishes exactly when ``k h_air/h_surface`` is a nonzero integer, so which harmonics
    cancel depends on the ratio's arithmetic rather than on its size. The residual is a small
    ripple in the source's *amplitude* across the surface, it shrinks under refinement, and it is
    the accuracy floor of this coupling — worth knowing before reading a radiated magnitude as
    physics.

    **What was expected to decide it, and what was actually found.** The plan for this batch argued
    from *reflection equivariance* — that bilinear's load matrix is mirror-symmetric at every offset
    while nearest-node's is not. Measured here (two independent routes: the defect of ``T^T R T``
    under the surface's own mirror permutation, and the defect of ``T`` itself under that mirror
    composed with the air grid's), **bilinear's equivariance is not offset-independent.** It holds
    exactly when ``S = 2 (surface centre) / h_air`` is an **integer** — defect 1.0e-15 there — and
    fails smoothly otherwise, 1.6e-01 … 3.8e-01 across sixteen offsets spanning one air cell,
    peaking at the half-cell offset. The algebra says the same thing: the surface's mirror sends
    node ``i`` to a cell fraction ``frac(S - t_i)``, which equals the ``1 - f_i`` that reverses a
    bilinear weight pair *only* for integral ``S``. Nearest-node's equivariance is patchier still,
    and the way it fails is the tell: it is exact at an **even** ``S`` (measured ``0.00e+00``) and
    breaks at an **odd** one (5.2e-01), because there the surface's own centre node lands exactly on
    a rounding tie, which round-half-to-even resolves the same way from both directions. So
    nearest-node's symmetry is an accident of alignment and of the rounding rule; bilinear's is a
    property of the geometry.

    **Two consequences, and neither weakens the design.** First, centring is load-bearing for *two*
    independent reasons rather than one — it is what makes ``S`` integral and the load matrix
    equivariant at all, on top of the scene-symmetry argument below. Second, the coupled monopole
    leak does **not** discriminate the two operators: with the surface centred, an even plate mode's
    ``|U|/A`` stays at rounding (1.3e-14 … 3.3e-13 over 200 steps) under *both* spreadings at
    ``N_plate = 8, 16, 24``. What breaks the zero is the **scene**, not the spreading operator —
    measured 2.3e-01 for a plate off-centre by ``h_air/3`` and 7.4e-02 for a centred plate in a room
    made asymmetric in the mode's own axis, against 7.2e-14 centred and symmetric.

    **The zero is a property of the whole SCENE, not of the surface or the spreading operator.**
    Equivariance of the load is necessary and *not* sufficient, because the load is only half the
    coupling: the incoming ``T^T pbar_free`` is the **room's** field, so an antisymmetric surface
    mode stays silent only if the room is mirror-symmetric about that mode's own antisymmetry
    plane. A **perfectly centred** plate in a room made asymmetric *in x* (lossy ``x0``, rigid
    ``x1``) leaks 3.5e-02 — the largest figure in the study — while the same asymmetry
    in *y* leaves a ``(2,1)`` mode at 3.0e-15. Six different room widths with the plate
    centred, spanning two grid alignments,
    all sit at rounding: **grid commensurability is not the criterion, centring is.** And there is
    no tolerance band — the leak is *linear* in the offset over four decades (``delta/h_air = 1e-6``
    gives 1.0e-07), so "approximately centred" is not approximately silent. Hence :attr:`origin`
    **defaults to centred in the face**. The asymmetric case is not a defect to hide: a room that
    re-excites a surface's *shape*, converting an acoustically silent mode into a radiating one at
    the 1-3% level, is one more thing no one-port can represent at all.

    **A baffled surface needs no new boundary machinery, and that is algebra rather than a
    measurement.** A soft injection of ``q = A_n v`` at a **wall** node *is* the moving-wall
    (piston) condition, not an approximation of it: the wall node's divergence is ``u_face/w_z``
    with ``w_z = h/2``, so a wall moving at ``v`` adds ``k rho0 c0^2 v/(h/2)``, while a soft source
    ``q = h^2 v`` into ``W = h^3/2`` adds ``k rho0 c0^2 (h^2 v)/(h^3/2)`` — and
    ``h^2/(h^3/2) = 1/(h/2)``, so these are the same number identically, on every wall type. Note
    the two expressions divide by ``W`` and by ``w_z`` in different orders, so an implementation
    written the other way would agree to machine precision and **not** bit-for-bit; this repo's
    bit-identity claims are load-bearing and that one could not be cashed.

    Parameters
    ----------
    room : AirBox
        The room to mount the surface in. The port registers itself with the room for the room's
        lifetime (used by the disjointness check), so build a fresh room rather than reusing one
        whose ports you have discarded.
    face : str
        Which of the six walls (:data:`FACES`) the surface lies flush in.
    coords : array, shape (n_surface, 2)
        Surface node positions (m) in the face's own two in-plane coordinates, in the surface's own
        frame — ``x0`` spans ``(y, z)``, ``y0`` spans ``(x, z)``, ``z0`` spans ``(x, y)``, with **no
        axis mirrored on a high face**. Together with ``areas`` this is the whole *surface
        protocol*: any grid resonator that can name its nodes and their areas can be mounted, with
        no port edit.
    areas : array, shape (n_surface,)
        Nodal areas (m^2) — the volume a unit nodal velocity displaces per second. Their sum is
        :attr:`net_area`, which is **not** the surface's bounding rectangle whenever the resonator
        has dead rim nodes (see that attribute).
    origin : (float, float), optional
        Where the surface frame's ``(0, 0)`` lands in the face's own coordinates (m). **Defaults to
        centred**, for the two independent reasons above: centring is what makes ``S`` integral and
        the load matrix mirror-equivariant, *and* what lets the scene be symmetric about a mode's
        own antisymmetry plane. An off-centre surface is legal and physical; it simply is not a
        configuration in which an antisymmetric mode is silent.
    spreading : {"bilinear", "nearest"}
        The area-spreading operator. ``"bilinear"`` ships. ``"nearest"`` is retained solely as the
        measured negative control of the symmetry argument above — **do not use it for physics.**

    Raises
    ------
    ValueError
        An unknown face or spreading name; malformed or negative ``coords``/``areas``; a footprint
        lying outside the face; a footprint that (with its one-cell bilinear halo) reaches the
        face's own rim; air nodes inside the footprint that no surface node feeds; a surface mounted
        on an ``"open"`` face; or a patch sharing a node with an existing port.

    Notes
    -----
    **Why the rim of the face is refused rather than clipped.** Clipping the stencil at the face
    boundary (``np.clip(i0 + di, 0, N)``) folds a node's outboard weight back onto the boundary
    node. The weights still sum to 1, so volume is still conserved and **every ledger stays green
    while the source geometry is quietly wrong** — the same failure shape as the sign flip above.
    :meth:`AirBox.node_index` already refuses to relocate an out-of-room point rather than snapping
    it; this matches. The refusal is stated on the *stencil*: every air node that receives weight
    must be strictly interior to the face. That has a second, load-bearing consequence — a node on
    the face's rim also touches a second wall, so it carries half the node weight ``W`` and the
    **sum** of two wall admittances in ``beta``. ``R_j`` would stop being uniform across the patch,
    and with it the ``T^T R T = R (T^T T)`` identity the offset-independent equivariance rests on.
    The refusal is therefore what keeps the symmetry argument true, not merely what keeps the
    geometry tidy. It also means a legal patch can only ever touch its own mounting face.

    **The acoustic source is up to one air cell larger than the surface**, because bilinear
    legitimately spreads onto the nodes just outboard of the surface's rectangle. The disjointness
    check therefore uses the **actual node set**, never the footprint.

    **A lossy mounting wall is a test configuration, not a recommended one.** Patch nodes keep the
    mounting wall's impedance: :meth:`AirBox.step` divides the injection by ``1 + beta`` whether one
    likes it or not, so ``R_j`` must say so, and the result is physically a piston in an *absorbing*
    baffle — consistent, passive, and the only way to pin the ``(1 + beta)`` factor (which the
    radiation leg left unpinned twice in its history). But such a wall absorbs **through the
    surface's own footprint**, i.e. it models a plate porous to the wall behind it. A **rigid**
    mounting wall is the true infinite baffle and is the sane default.
    """

    def __init__(
        self,
        *,
        room: AirBox,
        face: str,
        coords: NDArray[np.float64],
        areas: NDArray[np.float64],
        origin: Sequence[float] | None = None,
        spreading: Spreading = "bilinear",
    ) -> None:
        axis, end, t0, t1 = _face_axes(face)  # validates `face`
        self.face = face
        self.axis = axis
        face_coords = self._accept_surface(
            room=room,
            coords=coords,
            areas=areas,
            origin=origin,
            spreading=spreading,
            in_plane_axes=(t0, t1),
            where=f"face {face!r}",
        )

        rows, cols, vals = self._spread(face_coords)
        self._assemble(rows, cols, vals, axis, end, t0, t1)

        self._check_footprint(face_coords)
        self._check_open_faces()
        self._check_disjoint()
        room._ports.append(self)

    # -- construction: node placement -------------------------------------------------------

    def _assemble(
        self,
        rows: NDArray[np.intp],
        cols: NDArray[np.intp],
        vals: NDArray[np.float64],
        axis: int,
        end: int,
        t0: int,
        t1: int,
    ) -> None:
        """Build the patch node set, ``T``, ``R`` and the load matrix from the spread entries."""
        room = self.room
        i0, i1, face_nodes = self._plane_nodes(rows)
        self._check_in_plane_rim(i0, i1)
        self.T = self._build_T(rows, cols, vals, face_nodes)

        node_idx: list[NDArray[np.intp]] = [i0, i0, i0]  # placeholders; all three are overwritten
        node_idx[axis] = np.full(face_nodes.size, 0 if end == 0 else room.N[axis], dtype=np.intp)
        node_idx[t0] = i0
        node_idx[t1] = i1
        self.nodes: tuple[NDArray[np.intp], ...] = tuple(node_idx)
        self.index: tuple[int, int, int] = tuple(  # type: ignore[assignment]
            int(a[0]) for a in self.nodes
        )
        self._flat = np.ravel_multi_index(self.nodes, room.p.shape)

        W = room._W[self.nodes]
        self.R: NDArray[np.float64] = (
            room.k * room.rho0 * room.c0**2 / (2.0 * W * (1.0 + room._beta[self.nodes]))
        )
        # T^T R T: constant, symmetric, PSD, sparse -- the whole load, assembled once. Left as the
        # RAW triple product (never symmetrized): its ~1e-15 asymmetry is the sparse product's
        # summation order, and the symmetry-defect oracle is only meaningful on the raw matrix.
        self.load_matrix = (self.T.T @ sparse.diags(self.R) @ self.T).tocsr()
        self._queued_at = -1  # room.n at which this port last queued an injection

    # -- geometry / read-out ---------------------------------------------------------------

    def free_pressure(self) -> NDArray[np.float64]:
        """The open-circuit centered pressure **vector** ``pbar_free`` over the patch, ``O(patch)``.

        The distributed analogue of :meth:`RoomPort.free_pressure`, and the same contract: read it
        *before* ``room.step()``, from the stored ``u^{n+1/2}``. See :func:`_free_pressure_nodes`.
        """
        return _free_pressure_nodes(self.room, self.nodes)

    # -- driving ---------------------------------------------------------------------------

    def inject(self, q: NDArray[np.float64]) -> None:
        """Queue the **per-node** volume-velocity vector ``q`` (m^3/s) for the room's next step.

        Handed to the room as ``(nodes, weights=q, U=1.0)``: :meth:`AirBox.step` is linear in the
        weights for both the injection (``w U``) and the read-back (``w . pbar``), so a vector goes
        in and both come out exactly right — which is why the distributed tier needs **zero edits**
        to ``AirBox``. The room books the work at its own post-closure centered pressure, never at a
        number handed back from here, so the two ledgers stay an independent check on each other.
        """
        q = np.asarray(q, dtype=float)
        if q.shape != (self.node_count,):
            raise ValueError(
                f"q must be the per-node volume-velocity vector, shape {(self.node_count,)}, "
                f"got {q.shape}. (Pass q = port.T @ v, not the scalar sum -- the scalar is exactly "
                "what the lumped tier would have coupled through, i.e. the negative control.)"
            )
        self.require_ready()
        self.room._pending_ports.append((self.nodes, q, 1.0))
        self._queued_at = self.room.n


class RoomLoadedPlate:
    """A :class:`~physsynth.core.plate.Plate` mounted flush in a wall and loaded by the room over
    its **whole surface** — the distributed area coupling (batch 3).

    :class:`RoomLoadedBody` couples a lumped :class:`~physsynth.core.body.ModalBody` to a room
    through one scalar terminal. This couples a **grid** resonator through every node it has. The
    physical difference is :class:`SurfacePort`'s headline — the acoustic short circuit — and it is
    structural, not a refinement: an even-index mode of a simply supported plate has *exactly* zero
    net volume velocity, so every one-port in this repo (``AirRadiation``, ``RadiatedBody``,
    ``RationalAirLoad``, ``RoomPort``) reports exact silence from a mode this class radiates from
    definitely. This is also what makes the *existing* grid resonators radiate honestly: models #4,
    #5 and #5b have had a ``pressure()`` read-out since they were built, and it is
    ``sum_i area_i u_i''`` — a monopole, i.e. exactly the net volume acceleration, i.e. exactly the
    quantity that is insufficient. That read-out is not wrong; it is the compact ``a -> 0`` limit of
    this, and this is where the plate stops being a point source.

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
    bit-identity so a future change making the load non-dissipative (the two-sided dipole plate,
    batch 4, whose face cut removes air mass) fails loudly there instead of silently mis-guarding.

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
        if not np.isclose(plate.k, room.k, rtol=1e-12, atol=0.0):
            raise ValueError(
                f"sample-rate mismatch: plate fs = {plate.fs} but room fs = {room.fs}. The port's "
                "solve is a single timestep shared by both, so they must agree exactly."
            )
        self.room = room
        self.k = plate.k

        # The surface protocol: live-node coordinates and their areas. The live ordering is C-order
        # over `mask` (operators2d's contract), which is exactly what `X[mask]` yields -- so the
        # columns of T line up with the plate's own state vector by construction.
        coords = np.column_stack((plate.X[plate.mask], plate.Y[plate.mask]))
        if plate.boundary == "supported":
            areas = np.full(plate.n_live, plate.h * plate.h)
            # Per-node mass rho_s h^2: Plate.step divides f_ext by exactly this.
            self._denominator = plate.rho * plate.h * plate.h
        else:
            areas = plate.w.copy()  # lumped cell areas (h^2, h^2/2, h^2/4) -- no dead rim
            self._denominator = plate.rho  # W lives inside A and is divided out by the solve
        self.port = SurfacePort(
            room=room,
            face=face,
            coords=coords,
            areas=areas,
            origin=origin,
            spreading=spreading,
        )

        # A_loaded = A + (k / 2 rho_s) T^T R T -- SPD (PSD added to SPD), factored ONCE. The plate's
        # own A is reassembled here rather than reached into, because plate.py stays untouched; the
        # dense coupled cross-check at two timesteps is what pins that this reassembly is right.
        sk = plate.sigma * plate.k
        coeff = plate.theta * plate.k * plate.k * plate.kappa * plate.kappa
        if plate.boundary == "supported":
            a_bare = (1.0 + sk) * sparse.identity(plate.n_live, format="csc") + coeff * plate.B
        else:
            a_bare = (1.0 + sk) * plate.W + coeff * plate.K
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
        p, port = self.plate, self.port
        port.require_ready()               # before mutating anything
        pbar_free = port.free_pressure()   # read u^{n+1/2}, BEFORE room.step()

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
        # The air load: the known open-circuit force, plus the u^{n-1} half of the centered
        # velocity (its u^{n+1} half is already inside the factorization).
        rhs = rhs - k2 * (port.T.T @ pbar_free) / self._denominator
        rhs = rhs + self._load_scale * (port.load_matrix @ u_nm1)

        u_next = self._lu_loaded.solve(rhs)
        # _accel already carries the load -- it was IN the solve, so no post-solve refresh.
        p._accel = (u_next - 2.0 * p.u + u_nm1) / k2
        p.u_prev = p.u
        p.u = u_next
        p.n += 1

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


class InteriorSurfacePort(_PatchPort):
    """A moving surface **inside** the room, on a plane of faces — the two-sided port (batch 4).

    :class:`SurfacePort` mounts a surface flush in a wall. The wall then does the rest: it is the
    textbook infinite baffle, the surface's back face is unloaded, and the surface is — for all the
    room can tell — a **source**, a patch of wall that moves. This class takes the wall away. The
    surface hangs *in* the room, radiates from **both** faces, and is driven by the pressure
    **jump** across it.

    That is not batch 3 with a sign. A source adds sound to a room; an **object** also *removes*
    paths through it, and does so whether or not it is moving. Every instrument in the free-plate
    family — the suspended cymbal of model #5b, the gong of #6 — is physically an object hanging in
    air, so :class:`RoomLoadedPlate` models a loudspeaker cone or a soundboard in a cabinet honestly
    and a cymbal dishonestly.

    **The implementation news is better than the area-coupling plan feared.** That plan budgeted for
    new boundary machinery — "putting the plate on a plane of velocity faces and replacing their
    momentum update with the plate's motion". Measured, prescribing the face velocity and injecting
    a ``-q``/``+q`` pair on the two node planes straddling the plane are **the same
    arithmetic**::

        node (i,j,m)  divergence term  +u_face / w_z   ->  dp = -k rho0 c0^2 u_face / w_z
        a soft source q at that node                   ->  dp = +k rho0 c0^2 q / W,  W = A_face w_z

    and ``A_face * w_z = W`` identically, so with ``q = A_face u_face`` the two agree term for term,
    with the sign flipping between the planes. **The only new machinery is the cut** —
    :meth:`AirBox.add_cut`, zeroing ``u`` on the faces under the surface. The port protocol, the
    injection weights and the ``injected`` ledger are batch 3's, unchanged. (Batch 1's step ordering
    paid for batch 3 the same way — "a soft injection at a wall node *is* the moving-wall
    condition"; this is that lesson one geometry further.)

    **The load is ``2 T^T R T``, and the room's response is diagonal on both planes.** With ``T``
    the volume-weight matrix (surface -> **face**), ``q = T v`` the per-face volume velocity, and
    the injection weights ``(-q, +q)`` on planes ``m`` and ``m+1``::

        pbar_lo = pbar_free_lo  -  R q
        pbar_hi = pbar_free_hi  +  R q
        f       = -T^T (pbar_hi - pbar_lo)  =  -T^T d_free  -  2 T^T R T v

    with the **same** ``R_j = k rho0 c0^2 / (2 W_j (1 + beta_j))`` on both planes — which is what
    the rim refusals below are for. So the load is exactly *twice* the one-sided load on the same
    ``T`` and ``R``, still constant, symmetric, PSD and sparse, so it still folds into the
    resonator's own ``splu`` with nothing new solved. The family's ladder gains its last rung:
    :class:`~physsynth.core.radiation.RadiatedBody` scalar ``R`` -> one division;
    :class:`~physsynth.core.radiation.RationalAirLoad` ``R_eff`` + one aux state -> one division;
    :class:`RoomPort` scalar ``R_room`` -> one division; :class:`SurfacePort` diagonal ``R`` with a
    matrix ``T`` -> absorbed into the LU; **this**, the same, doubled.

    **The factor 2 must be earned by measurement, not by construction.** ``assert load_matrix ==
    2 * one_sided`` is not a test — it re-checks arithmetic this file just wrote. Two realistic ways
    to get it wrong are blind to *different* ledgers: a ``1x`` inside the factorization only leaves
    ``|radiated - injected|`` at rounding (4.5e-16) while the scene drifts (1.8e-02), and a
    consistent ``1x`` ("I forgot the plate has two faces") leaves the scene flat to 2.4e-14 while
    the ledgers disagree by 1.8x. **So the money test is not sufficient either** — the guard that
    catches both is the coupled residual at two timesteps, against the room's own
    post-closure pressure jump.

    **The sign convention is different from batch 3's, and it is load-bearing.**
    :class:`SurfacePort` can make the inward normal disappear by defining positive displacement
    *along* it, which is why no inward normal appears in that code at all. An interior plane has two
    sides and no inward normal, so this class must choose: **positive surface displacement is along
    the plane's own axis** (``+x``/``+y``/``+z``), so the ``index+1`` plane is the one a positive
    velocity compresses. Flip the ``-q``/``+q`` order **and** the jump direction together and every
    energetic quantity is *bit-identical* — the load matrix, the solve, the pressure jump and
    ``radiated_energy`` — while the room's field is exactly inverted and the plate is anti-driven by
    it: passive, conservative, green, and pushing the wrong way. (Flip only one of the two and the
    conserved total does catch it.) The detector is the sign of the room's own pressure on the
    **first** step.

    **The obstacle is up to one air cell larger than the surface, and that is the honest choice.**
    The cut is the **support of** ``T``, not the surface's footprint. It has to be: the identity
    above holds only where a face carrying ``q`` is also cut, so a face with ``q`` and no cut is a
    phantom patch — and clipping ``T`` instead is exactly the "volume conserved, ledgers green,
    geometry quietly wrong" failure batch 3 refuses. :attr:`blocked_area` reports the cost.

    **A legal interior surface always has a diffraction path around it**, and that is a corollary
    rather than a limitation to hide: the rim refusal means a port can never span a full
    cross-section, so it can never seal the room. :meth:`AirBox.add_cut` provides the sealing
    version, which is what the sub-room modal oracle uses.

    Parameters
    ----------
    room : AirBox
        The room to hang the surface in. The port registers itself with the room for the room's
        lifetime (used by the disjointness checks), so build a fresh room rather than reusing one
        whose ports you have discarded.
    plane : {"x", "y", "z"}
        The surface's **normal** axis.
    index : int
        Face index along ``plane``. Both straddling node planes must be strictly interior, so
        ``1 <= index <= N - 2``.
    coords : array, shape (n_surface, 2)
        Surface node positions (m) in the plane's own two in-plane coordinates, in the surface's own
        frame — a ``"z"`` plane spans ``(x, y)``. Together with ``areas`` this is batch 3's *surface
        protocol* unchanged, so any grid resonator that can name its nodes and their areas mounts
        here with no port edit.
    areas : array, shape (n_surface,)
        Nodal areas (m^2). See :attr:`net_area`.
    origin : (float, float), optional
        Where the surface frame's ``(0, 0)`` lands in the plane's own coordinates (m). **Defaults to
        centred**, for :class:`SurfacePort`'s two independent reasons.
    spreading : {"bilinear", "nearest"}
        The area-spreading operator; ``"nearest"`` is that class's measured negative control.

    Raises
    ------
    ValueError
        An unknown plane or spreading name; an ``index`` whose straddling node planes are not both
        interior; malformed or negative ``coords``/``areas``; a footprint outside the plane; a
        footprint that (with its one-cell bilinear halo) reaches the room's rim; air nodes inside
        the footprint that no surface node feeds; a patch sharing pressure nodes with an existing
        port; or a patch sharing **cut faces** with an existing cut.

    Notes
    -----
    **There is no open-face refusal here, and that is provable rather than an omission.** The
    in-plane rim refusal keeps every patch node strictly interior on both in-plane axes, and the
    ``index`` refusal does the same along the normal — so an interior patch cannot touch any wall,
    open or otherwise, and :class:`SurfacePort`'s two reasons for that refusal cannot arise.

    **Cut-face disjointness is a genuinely new failure mode, not an inherited one.** Two ports can
    have disjoint *node* sets and overlapping *cuts*, because the node sets live on different planes
    while the cuts can share one. That is why :meth:`AirBox.add_cut` is additive and why a port
    records its own face set.
    """

    def __init__(
        self,
        *,
        room: AirBox,
        plane: str,
        index: int,
        coords: NDArray[np.float64],
        areas: NDArray[np.float64],
        origin: Sequence[float] | None = None,
        spreading: Spreading = "bilinear",
    ) -> None:
        axis = room._plane_axis(plane)  # validates `plane`
        n_face = room.N[axis]
        index = int(index)
        if not 1 <= index <= n_face - 2:
            raise ValueError(
                f"interior surface index {index} on plane {plane!r} is out of range 1..{n_face - 2}"
                f" (the room has {n_face} face(s) there). The two node planes straddling the "
                "surface must BOTH be strictly interior: a node plane on a wall carries half the "
                "node weight W and the wall's admittance in beta, so R_j would differ between the "
                "two sides and the load would stop being 2 T^T R T with a single R."
            )
        self.plane = plane
        self.axis = axis
        self.face_index = index
        t0, t1 = (a for a in range(3) if a != axis)
        face_coords = self._accept_surface(
            room=room,
            coords=coords,
            areas=areas,
            origin=origin,
            spreading=spreading,
            in_plane_axes=(t0, t1),
            where=f"the plane {plane!r} cross-section at index {index}",
        )

        rows, cols, vals = self._spread(face_coords)
        self._assemble(rows, cols, vals, axis, t0, t1)

        self._check_footprint(face_coords)
        self._check_disjoint()
        # The cut LAST, because it is the only registration that mutates the room: if any refusal
        # above fires, the room is left exactly as it was found.
        room._register_cut(self, axis, index, self._in_plane[0], self._in_plane[1])
        room._ports.append(self)

    # -- construction: node placement -------------------------------------------------------

    def _assemble(
        self,
        rows: NDArray[np.intp],
        cols: NDArray[np.intp],
        vals: NDArray[np.float64],
        axis: int,
        t0: int,
        t1: int,
    ) -> None:
        """Build the two node planes, ``T``, ``R`` and the doubled load from the spread entries."""
        room = self.room
        i0, i1, plane_nodes = self._plane_nodes(rows)
        self._check_in_plane_rim(i0, i1)
        self.T = self._build_T(rows, cols, vals, plane_nodes)
        self._in_plane = (i0, i1)

        planes = []
        for offset in (0, 1):
            node_idx: list[NDArray[np.intp]] = [i0, i0, i0]  # placeholders; all three overwritten
            node_idx[axis] = np.full(i0.size, self.face_index + offset, dtype=np.intp)
            node_idx[t0] = i0
            node_idx[t1] = i1
            planes.append(tuple(node_idx))
        self.nodes_lo, self.nodes_hi = planes
        # The combined set is what the disjointness check and the room's node bookkeeping see.
        self.nodes: tuple[NDArray[np.intp], ...] = tuple(
            np.concatenate((lo, hi)) for lo, hi in zip(self.nodes_lo, self.nodes_hi, strict=True)
        )
        self.index: tuple[int, int, int] = tuple(  # type: ignore[assignment]
            int(a[0]) for a in self.nodes_lo
        )
        self._flat = np.ravel_multi_index(self.nodes, room.p.shape)

        # One R for both planes -- the refusals above are exactly what makes that true, and the
        # differential oracle measures it off the room rather than trusting this line.
        W = room._W[self.nodes_lo]
        self.R: NDArray[np.float64] = (
            room.k * room.rho0 * room.c0**2 / (2.0 * W * (1.0 + room._beta[self.nodes_lo]))
        )
        # 2 T^T R T: two faces, two resistances. Left as the RAW triple product (never symmetrized)
        # for batch 3's reason -- the symmetry-defect oracle is only meaningful on the raw matrix.
        self.load_matrix = 2.0 * (self.T.T @ sparse.diags(self.R) @ self.T).tocsr()
        self._queued_at = -1  # room.n at which this port last queued an injection

    # -- geometry / read-out ---------------------------------------------------------------

    @property
    def face_count(self) -> int:
        """How many velocity faces the surface cuts — half of :attr:`node_count`."""
        return int(self.nodes_lo[0].size)

    @property
    def blocked_area(self) -> float:
        """The **cut** area (m^2) — ``face_count * h^2``, and *not* :attr:`net_area`.

        The obstacle is the support of ``T``, so it is up to one air cell larger than the surface in
        each direction. Reported rather than discovered. Two things to know before blaming a
        radiated number on it: under air-grid refinement the ratio to the moving surface's own
        bounding rectangle is non-monotone (the node count is a rounding of the footprint) and
        trends to 1; and on a **supported** plate the moving surface is the *live* footprint, so the
        clamped rim is not part of the obstacle at all — the free plate is the configuration where
        the obstacle and the physical plate coincide, which is also the one a suspended object is.
        """
        return float(self.face_count) * self.room.h**2

    def free_pressure(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """The open-circuit centered pressure on the **low** and **high** node planes, ``O(patch)``.

        Batch 3's contract, twice: read *before* ``room.step()``, from the stored ``u^{n+1/2}``. The
        shared local read needs no cut-awareness — on the two straddling planes the face between
        them is cut, i.e. ``u = 0``, which the read picks up from the stored array like any other
        value. See :func:`_free_pressure_nodes`.
        """
        return (
            _free_pressure_nodes(self.room, self.nodes_lo),
            _free_pressure_nodes(self.room, self.nodes_hi),
        )

    # -- driving ---------------------------------------------------------------------------

    def inject(self, q: NDArray[np.float64]) -> None:
        """Queue the **per-face** volume-velocity vector ``q`` (m^3/s) as a ``-q``/``+q`` pair.

        ``-q`` goes to the node plane at :attr:`face_index` and ``+q`` to the one above it, which is
        the sign convention of the class docstring: a positive velocity along the plane's own axis
        compresses the air it moves toward. Both are ordinary soft injections
        (:meth:`AirBox.inject`'s own source term, vectorized), so the room books the work at its own
        post-closure pressure from both planes and the two ledgers stay an independent check on each
        other: their sum is ``k q . (pbar_hi - pbar_lo)``, exactly the port's own channel.
        """
        q = np.asarray(q, dtype=float)
        if q.shape != (self.face_count,):
            raise ValueError(
                f"q must be the per-FACE volume-velocity vector, shape {(self.face_count,)}, "
                f"got {q.shape}. Note this is HALF the node count ({self.node_count}): the two "
                "node planes share one q, with opposite signs."
            )
        self.require_ready()
        self.room._pending_ports.append((self.nodes_lo, -q, 1.0))
        self.room._pending_ports.append((self.nodes_hi, q, 1.0))
        self._queued_at = self.room.n


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
        if not np.isclose(plate.k, room.k, rtol=1e-12, atol=0.0):
            raise ValueError(
                f"sample-rate mismatch: plate fs = {plate.fs} but room fs = {room.fs}. The port's "
                "solve is a single timestep shared by both, so they must agree exactly."
            )
        self.room = room
        self.k = plate.k

        # The surface protocol, batch 3's verbatim: live-node coordinates and their areas, in the
        # plate's own C-order over `mask`, so T's columns line up with the plate's state vector.
        coords = np.column_stack((plate.X[plate.mask], plate.Y[plate.mask]))
        if plate.boundary == "supported":
            areas = np.full(plate.n_live, plate.h * plate.h)
            self._denominator = plate.rho * plate.h * plate.h
        else:
            areas = plate.w.copy()  # lumped cell areas (h^2, h^2/2, h^2/4) -- no dead rim
            self._denominator = plate.rho  # W lives inside A and is divided out by the solve
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
        # plate's own A is reassembled here rather than reached into, because plate.py stays
        # untouched; the coupled residual at two timesteps is what pins that this reassembly is
        # right, and it is the only guard that catches BOTH ways of getting the 2 wrong.
        sk = plate.sigma * plate.k
        coeff = plate.theta * plate.k * plate.k * plate.kappa * plate.kappa
        if plate.boundary == "supported":
            a_bare = (1.0 + sk) * sparse.identity(plate.n_live, format="csc") + coeff * plate.B
        else:
            a_bare = (1.0 + sk) * plate.W + coeff * plate.K
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
        p, port = self.plate, self.port
        port.require_ready()                      # before mutating anything
        lo_free, hi_free = port.free_pressure()   # read u^{n+1/2}, BEFORE room.step()
        d_free = hi_free - lo_free

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
        # The air load: the known open-circuit pressure JUMP, plus the u^{n-1} half of the centered
        # velocity (its u^{n+1} half is already inside the factorization).
        rhs = rhs - k2 * (port.T.T @ d_free) / self._denominator
        rhs = rhs + self._load_scale * (port.load_matrix @ u_nm1)

        u_next = self._lu_loaded.solve(rhs)
        # _accel already carries the load -- it was IN the solve, so no post-solve refresh.
        p._accel = (u_next - 2.0 * p.u + u_nm1) / k2
        p.u_prev = p.u
        p.u = u_next
        p.n += 1

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
