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

from typing import Literal

import numpy as np
import physsynth_rs as _rs
from scipy import sparse

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

# Every class in this module is Rust, on every path -- the room, the three ports, the seven
# wrappers and the three seams. They are imported into this namespace rather than reached through
# `physsynth_rs` at each use site for two reasons, and the second is not optional:
#
#   * they are this module's public names -- `__all__` above lists ten of them, `tests/helpers.py`
#     imports eight and `web/serialize.py` five;
#   * `crates/physsynth-py/src/airbox_wrap.rs` calls `py.import("physsynth.core.airbox")` and reads
#     `RoomPort`, `splu` and the seam and port class names **off this namespace at call time** --
#     the reference's own behaviour, transcribed. That is findings ledger #39, Rust depending on
#     Python, and this import is what keeps it resolving.
#
# The three seams carry the `noqa` because they are in neither `__all__` nor any Python caller --
# `airbox_wrap.rs` is the only thing that reads them, by name, and ruff cannot see that use.
#
# `_face_axes` and `_free_pressure_nodes` are NOT re-exported. They were module-level helpers the
# Python ports shared, and the only things that reached them through this module were the parity
# tests deleted alongside -- so they fail section 41.2's third category ("reached through this
# module") rather than satisfying it. `physsynth_rs` still exports both.
from physsynth_rs import (  # noqa: E402
    AirBox,
    InteriorSurfacePort,
    RoomLoadedBody,
    RoomLoadedMembrane,
    RoomLoadedPlate,
    RoomLoadedVKPlate,
    RoomPort,
    RoomSuspendedMembrane,
    RoomSuspendedPlate,
    RoomSuspendedVKPlate,
    SurfacePort,
    _MembraneSurface,  # noqa: F401
    _PlateSurface,  # noqa: F401
    _VKPlateSurface,  # noqa: F401
    impedance_from_zeta,
)

# -- the factorization, and why it is NOT behind a flag (plan section 43) -------------------------
#
# Every room-loaded class **reassembles** the model's system matrix and factors it here rather than
# reaching into the model, and four of the family's reduction tests turn that into a bit-identity
# claim: with the load switched off, `a_loaded` IS the model's own `A`, so a loaded plate must
# reproduce a bare one byte for byte. That claim holds only while both sides factor with the same
# solver, and unit 5's deletion made the model's solver Rust on every path -- so a SuperLU
# factorization here would break all four anchors for a reason having nothing to do with this
# module (measured before the fix: 15 tests, `array_equal` failing at ~1e-16).
#
# `splu` is looked up as a **module global at call time**, by the Rust wrappers in
# `crates/physsynth-py/src/airbox_wrap.rs` and by three of the surviving physics test files
# (`test_airbox_dipole.py`, `test_airbox_surface.py`, `test_airbox_vk.py`), which re-derive a
# wrapper's factorization and must use the same factorizer it did. So this name is load-bearing in
# both directions and stays exactly where it is.


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
