"""What is left of the suspended plate's file: the **bridge** chain, which is not ported yet.

Plan ``docs/dev/python-retirement-plan.md`` §16 re-homed the six ``RoomLoaded*`` /
``RoomSuspended*`` wrappers and took this file's dipole bars to
``crates/physsynth-core/tests/airbox_grid.rs`` — the two ledgers, the piston's non-vacuous channel,
the coupled residual that catches **both** wrong twos, the measurement that each ledger is blind to
a *different* one of them, ``R_j`` read off both planes with opposite signs, the reservoir-not-a-
drain channel, the first-step sign, the mirror-symmetric oracle, the phantom's bit-identity to two
monopoles, and the headline: **drop the cut and the coupling converges to silence under refinement,
which is what proves the cut is load-bearing.**

Two things did not come over, and both are recorded in the plan rather than lost:

* ``test_a_sign_flip_is_invisible_to_every_energy_quantity`` worked by **replacing two of the
  port's methods on the instance** — a seam the binding deliberately kept alive by giving every
  port class a ``dict`` (``docs/dev/rust-migration-plan.md`` §31.6). A value-typed native port has
  no such seam, so the suspended arm of that claim has no analogue; the *baffled* arm survives as
  ``airbox_grid.rs::a_sign_flip_is_invisible_to_every_energy_quantity``, and the detector the
  claim exists to justify survives as ``the_sign_is_readable_on_the_first_step``.
* ``test_surface_port_is_unchanged_across_the_shared_spreading_refactor`` is two thirds carried and
  one third spent. Its **construction** digests are now
  ``airbox_grid.rs::the_baffled_port_construction_digests_are_unchanged``, asserted against the
  reference's own numbers; its 200-step run-end values are not re-frozen, because the refactor they
  guarded shipped long ago and re-recording them from one machine would promote a pile of
  incidental digits to a cross-machine claim (§15).

One test stays. :func:`test_string_bridge_plate_room_chain` drives
:class:`~physsynth.core.connection.StringPlateBridge`, which has no home outside the binding yet.
"""

import numpy as np
import pytest
from helpers import make_suspended_plate

from physsynth.core.airbox import impedance_from_zeta
from physsynth.core.connection import StringPlateBridge
from physsynth.core.plate import Plate
from physsynth.core.string_ideal import IdealString

ALL_LOSSY = impedance_from_zeta(4.0)
BOUNDARIES = ("supported", "free")


@pytest.mark.parametrize("boundary", BOUNDARIES)
def test_string_bridge_plate_room_chain(boundary):
    """``string -> bridge -> suspended plate -> room``, no ``connection.py`` edit, guard **safe**.

    And this retires batch 3's own prediction. ``test_string_bridge_plate_room_chain`` there says
    the two-sided dipole plate, "whose face cut removes air mass", would make the load
    non-dissipative and fail loudly here. It does not, and the reasoning does not survive: the face
    cut removes air inertia from the **room's** ledger, where it never was part of ``G0``,
    while the load itself stays proportional to ``u^{n+1} - u^{n-1}`` — dissipative, merely doubled
    — so it enters ``A`` and never ``G0``. The margin comes out bit-identical, and it is the *same*
    margin batch 3 measured (0.2061806714931906 supported, 0.2061840079056186 free) because the
    guard never saw either load.
    """
    inst = make_suspended_plate(boundary=boundary, walls=ALL_LOSSY)
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
