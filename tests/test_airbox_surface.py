"""What is left of the baffled plate's file: the **bridge** chain, which is not ported yet.

This file has now been split twice, and the line both times was the same one (plan
``docs/dev/python-retirement-plan.md`` §14.3): **a test retires when its referent is a class that
has been re-homed into a surviving crate, and stays when it drives a caller that has not been.**

* §14 re-homed :class:`SurfacePort` and :class:`InteriorSurfacePort` and took the *port* half —
  the spreading operator, the span-wise footprint criterion, ``net_area``, the construction
  refusals — to ``crates/physsynth-core/tests/airbox_surface.rs``.
* §16 re-homed the six ``RoomLoaded*`` / ``RoomSuspended*`` wrappers and the two linear seams, and
  took the *wrapper* half to ``crates/physsynth-core/tests/airbox_grid.rs``: the two ledgers, the
  differentially measured ``R_j``, the volume identity, the coupled residual, the sign convention
  over six faces, the sign-flip negative control, the scene total, the free plate's fully radiated
  piston, the factorization's cost, and the batch's headline — that an even mode with **exactly**
  zero net volume velocity is silent to every one-port in this repo and still radiates.

One test stays, and only one. :func:`test_string_bridge_plate_room_chain` drives
:class:`~physsynth.core.connection.StringPlateBridge`, which lives in ``crates/physsynth-py`` and
has no home outside the binding yet; it is the *bridge* tier, and it goes when that tier does.
"""

import numpy as np
import pytest
from helpers import make_room_loaded_plate

from physsynth.core.airbox import impedance_from_zeta
from physsynth.core.connection import StringPlateBridge
from physsynth.core.plate import Plate
from physsynth.core.string_ideal import IdealString

ALL_LOSSY = impedance_from_zeta(4.0)
BOUNDARIES = ("supported", "free")


@pytest.mark.parametrize("boundary", BOUNDARIES)
def test_string_bridge_plate_room_chain(boundary):
    """``string -> bridge -> plate -> room``, with **no edit to** ``connection.py``, guard safe.

    The margin assertion is the load-bearing half. ``StringPlateBridge._stability_margin``
    reassembles the plate's ``G0`` block from scratch out of ``theta, rho, h, kappa, B / W, K`` —
    every one of which ``__getattr__`` delegation hands over happily — so the guard is computed
    against physics that is not happening, and the delegation would hide that perfectly. It is safe
    because ``G0 = M + (theta - 1/4) k^2 S`` is a statement about mass and theta-excess stiffness
    while the air load is **dissipative**: it enters ``A``, never ``G0``. Pinning the bit-identity
    here means a future change making the load non-dissipative fails loudly instead of silently
    mis-guarding.

    **This docstring used to name batch 4's two-sided dipole plate, "whose face cut removes air
    mass", as that future change. It is not one**, and the reasoning did not survive: the face cut
    removes air inertia from the *room's* ledger, where it was never part of the plate's ``G0``,
    while the load stays proportional to ``u^{n+1} - u^{n-1}`` and enters ``A``. Measured in
    ``tests/test_airbox_dipole.py::test_string_bridge_plate_room_chain``, a ``RoomSuspendedPlate``
    gives the **same** margin this test pins, to the last digit.
    """
    inst = make_room_loaded_plate(boundary=boundary, walls=ALL_LOSSY)
    loaded = inst.plate
    bare = Plate(
        Lx=loaded.Lx, Ly=loaded.Ly, kappa=loaded.kappa, rho=loaded.rho, fs=loaded.fs,
        N=loaded.N, boundary=boundary,
    )
    strings = [
        IdealString(L=0.6, T=60.0, rho=0.005, fs=loaded.fs, N=40, boundary=("fixed", "free"))
        for _ in range(2)
    ]
    bridge_bare = StringPlateBridge(string=strings[0], plate=bare, K=800.0)
    bridge = StringPlateBridge(string=strings[1], plate=inst, K=800.0)
    assert bridge.stability_margin == bridge_bare.stability_margin

    xs = np.linspace(0.0, strings[1].L, strings[1].N + 1)
    strings[1].set_state(1e-3 * np.sin(np.pi * xs / strings[1].L))
    e0 = bridge.energy() + inst.room.energy()
    worst = 0.0
    for _ in range(600):
        bridge.step()
        inst.room.step()
        worst = max(worst, abs(bridge.energy() + inst.room.energy() - e0))
    assert worst <= 1e-11 * abs(e0)
    assert abs(inst.radiated_energy) > 1e-4 * abs(e0)
