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

**Read-out only — no back-reaction.** The source injects into the room and the room does not push
back on it. That is the exact position :class:`~physsynth.core.radiation.AirRadiation` occupied in
radiation batch 1, and the module's own history says build read-out first; a two-way, provably
passive room-body port is the natural next batch. Likewise deferred: PML, HRTF/ambisonics,
scattering objects and non-rectangular geometry, and viscothermal air absorption.

**Cost is a design constraint here, for the first time in this repo.** At ``fs = 44.1 kHz`` the CFL
forces ``h >= sqrt(3) c0 / fs ~ 1.35 cm``, so a 1 m^3 room is ~74^3 nodes and one second of audio is
44100 steps — minutes, not seconds. The modal oracles are rate-independent and run on ~10x8x6 grids
where they are free; genuinely audio-rate rooms belong in a diagnose script, not a test suite.

Headless: NumPy only.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

__all__ = ["AirBox", "FACES", "RHO0_AIR", "C0_AIR", "impedance_from_zeta"]

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

        if self._pending:
            gain = self.k * self.rho0 * self.c0**2
            for index, q in self._pending:
                p_next[index] += gain * q / self._W[index]

        if self._has_walls:
            p_next = (p_next - self._beta * p_old) / (1.0 + self._beta)
            p_next[self._open] = 0.0
            # Book the wall flux at the centered pressure: k * A * pbar^2 / Z >= 0, per face. An
            # edge node pays into every lossy face it belongs to, which is exactly what summing
            # admittances into beta already charged it.
            pbar = 0.5 * (p_next + p_old)
            for index, area, Z in self._lossy:
                self.dissipated += self.k * float(np.sum(area * pbar[index] ** 2)) / Z

        if self._pending:
            for index, q in self._pending:
                self.injected += self.k * 0.5 * (p_next[index] + p_old[index]) * q
            self._pending.clear()

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
