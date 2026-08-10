"""Diagnostics for the interior two-sided (dipole) plate: air-box batch 4.

Batch 3's plate was mounted flush in a wall — baffled, one-sided, and for all the room could tell a
*source*. Batch 4 hangs it *in* the room. Three figures, one per claim, and every one of them
carries a number the suite deliberately does not assert, because each needs either a
**prescribed-velocity rig** (the coupled ``radiated_energy`` is half reactive and cannot be read
as radiation) or a room far too large for a test suite.

  1. **Unbaffling is not a factor — it changes sign.** Prescribed uniform piston motion on the same
     surface, mounted two ways in the same 5 m rigid room, read as the radiation resistance
     ``R = dE_injected / integral v^2 dt`` over whole cycles inside a **windowed** reflection-free
     interval (batch 1's doctrine: windowing, never absorption). The ratio dipole/baffled measures
     **0.278, 0.569, 1.339, 2.257, 2.314, 1.965** at ``ka = 0.8 .. 2.8``: far below 1 at the bottom
     and above 1 by ``ka ~ 1.2``. A ratio that **crosses 1** cannot be reproduced by any constant,
     and therefore not by any ``R(omega)`` fitted to the baffled case — the same structural argument
     as batch 2's delayed echo and batch 3's acoustic short circuit, a third time.

     **The crossing is the claim; the magnitudes are not**, and this script measures which is which
     rather than asserting it. Under air-grid refinement at fixed room and plate the *baffled* arm
     converges smoothly (``R/rho0 c0 A`` = 0.901, 0.915, 0.941, 0.966 at ``ka = 2.2``, approaching
     its plane-wave asymptote of 1 from below — the shape a sane arm has, and the rig's own sanity
     check). The *dipole* arm does not: at ``ka = 1.0`` it reads 0.279, 0.231, 0.360, 0.225, and it
     tracks :attr:`~physsynth.core.airbox.InteriorSurfacePort.blocked_area` (1.44, 1.36, **1.78**,
     1.44 plate areas) rather than ``h``. **At low ``ka`` the dipole's magnitude is set by the
     obstacle, and the obstacle is a rounding of the footprint onto the air grid, so it does not
     converge.** The ratio's *sign relative to 1* survives all of it — every refinement has
     ``ka = 1.0`` below and ``ka = 1.7`` above.

     **The plane-wave ceiling, and why "``ratio <= 2``" is the wrong pass criterion.** A baffled
     piston tends to ``R -> rho0 c0 A`` and a two-sided one radiates from *each* face, so
     ``R -> 2 rho0 c0 A`` and the asymptotic ratio's ceiling is 2. The plan proposed that as a pass
     criterion. Measured, the *dipole's own* ``R/rho0 c0 A`` tops out at 2.12 - 2.30 — a few per
     cent over — while the *ratio* reaches 2.31, because the baffled arm has not saturated yet (0.92
     at ``ka = 2.2``). A piston's resistance legitimately overshoots its own asymptote near its
     first maximum, so a mid-band value slightly above the asymptote is physics; the prototype's
     3.75 and 4.69 were not. **The criterion that survives is on each arm separately,
     asymptotically** — and the baffled arm's textbook shape is what makes the dipole arm
     believable at all.

  2. **It has a direction, and nothing else in this repo does.** Windowed free field, normalized
     peak ``|p|`` on a quarter arc from on-axis to in-plane. The dipole reads **1.000, 0.928,
     0.786, 0.565, 0.347, 0.164, 0.012** against ``cos theta`` = 1.000, 0.966, 0.866, 0.707, 0.500,
     0.259, 0.000 — an **85x null in the plate's own plane** — while the baffled plate over the
     same arc reads 1.000 … 0.530 with no null anywhere. Every lumped one-port here
     (``AirRadiation``, ``RadiatedBody``, ``RationalAirLoad``, ``RoomPort``) is a monopole with no
     angular dependence at all, so an angular null is not a magnitude and survives every caveat the
     magnitudes carry.

     Three caveats, reported rather than hidden. The probe **radii and angles are the snapped ones**
     (:meth:`~physsynth.core.airbox.AirBox.snapped`'s own docstring forbids charging the snap to the
     physics): requested 0/15/30/45/60/75/90 land on 0.0/14.3/30.3/45.9/59.2/74.2/88.8 degrees at
     radii 1.175 - 1.222 m. The source is not compact at this radius (``a = 0.17 m``,
     ``r = 1.2 m``), which is why the mid-arc values sit below the cosine. And the baffled arm's
     ``theta = 90`` probe sits **on the ``z0`` wall** — a half-weight node on the baffle surface,
     not the free-space counterpart of the dipole's in-plane point; the two rows' end columns are
     not the same measurement.

     And what directivity does *not* distinguish: the **phantom** (the same ``-q``/``+q`` pair with
     the cut dropped) has the same pattern to within a per cent — in-plane 0.010 — at **5.2x** less
     amplitude. *Directivity identifies the dipole; the blockage sets its strength.* Both figures
     are needed and they are testing different things.

  3. **The source alone converges to silence.** ``t50`` (steps to lose half the plate's energy)
     for the same plate and the same motion, three mountings, under air-grid refinement. The phantom
     is a legal, perfectly conservative dipole *source* carrying the plate's own motion and no
     obstacle, and its ``t50`` ratio against the real plate **diverges**: 5.2, 19.3, 40.8 at
     1x/2x/3x for a supported ``(1,1)`` mode, while the free plate's piston phantom never reaches
     ``t50`` at all. A doublet at separation ``h`` has moment proportional to ``h`` by construction,
     so this is an **implementation** control rather than a claim about source-only tiers: it is
     exactly what batch 4 degrades to if the cut is omitted or clobbered, so the divergence is what
     proves the cut is load-bearing and cannot be quietly dropped.

     Read the *dipole/baffled* row on that same figure with §7.7's warning in hand, because it
     points the other way from figure 1 and both are right. ``t50`` counts the reactive near field
     as though it had left, and batch 4's channel is dominantly reactive: the free piston's dipole
     ``t50`` is **~1/20th** of the baffled one, i.e. the plate sheds energy far *faster* — into a
     near field that hands most of it back. Figure 1, which integrates over whole cycles at a
     prescribed velocity, says the dipole is the far weaker **radiator** at that same low ``ka``.
     A decay time is not a radiation measure, and this is the pair that proves it.

Two smaller measurements print without a figure:

  * **The blockage overshoot** (§6.4), against the **moving** surface's own bounding rectangle
    rather than the plate's, because the cut is the support of ``T`` and a supported plate's
    clamped rim is not in ``T``. Re-measured with a mode-shaped motion as well as a piston, since
    the overshoot extends the rectangle asymmetrically relative to a mode's nodal lines.
  * **The ``k``-only refinement** (§5), bounding the half-step placement of the plate's centered
    velocity in a face slot that lives at ``n+1/2``. Nothing in either ledger can see that choice,
    so it is bounded by fixing ``h`` and raising ``fs`` through the Courant fraction.

Two things it costs, owned rather than discovered: **~7 min** (figure 1's refinement is 24 runs in
a 5 m room and is 85% of it), and **~1 GB peak** at the finest grid, where a 5 m room at
``h = 3 cm`` is 167^3 nodes. Both are why this is a script and not a test.

Run:  python scripts/diagnose_airbox_dipole.py
Figures land in ./out/.
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from physsynth.core.airbox import (  # noqa: E402
    C0_AIR,
    RHO0_AIR,
    AirBox,
    InteriorSurfacePort,
    RoomLoadedPlate,
    RoomSuspendedPlate,
    SurfacePort,
    impedance_from_zeta,
)
from physsynth.core.plate import Plate  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out")
os.makedirs(OUT, exist_ok=True)

CFL = 0.9  # fraction of the 3-D ceiling 1/sqrt(3), the helpers' default everywhere

# -- the prescribed-velocity rig (figures 1 and 2) ---------------------------------------
PLATE_L = 0.30          # m, square surface
N_SURFACE = 13          # surface nodes per side
FREE_H = 0.05           # m, air spacing of the free-field rooms
FREE_ROOM = 5.0         # m cube -- big enough to window a 258 Hz burst reflection-free
PISTON_V = 1e-3         # m/s
KA_SWEEP = (0.8, 1.0, 1.3, 1.7, 2.2, 2.8)
KA_REFINE = (1.0, 1.7, 2.2)
REFINE_GRIDS = ((0.0600, 13), (0.0500, 13), (0.0400, 17), (0.0300, 21))
ARC_KA = 1.7
ARC_R = 1.20            # m
ARC_THETAS = np.array([0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0])

AREA = PLATE_L * PLATE_L
A_EFF = np.sqrt(AREA / np.pi)          # equal-area circular radius, the ka the tables use
PLANE_WAVE = RHO0_AIR * C0_AIR * AREA  # the baffled arm's high-ka asymptote


def air_fs(h: float) -> float:
    """The sample rate a room must use to sit at ``CFL`` of the 3-D ceiling at this ``h``."""
    return C0_AIR * np.sqrt(3.0) / (CFL * h)


def ka_frequency(ka: float) -> float:
    return ka * C0_AIR / (2.0 * np.pi * A_EFF)


def flat_surface(n_side: int):
    """A square grid of surface nodes with uniform areas summing to :data:`AREA` — a rigid piston.

    Deliberately *not* a :class:`Plate`: figures 1 and 2 prescribe the motion, so there is no
    resonator here at all. That is the whole point of the rig — ``radiated_energy`` on a coupled
    plate counts the reactive near field as though it had left (see figure 3), and a radiation
    figure has to come from a motion that is *held*, integrated over whole cycles.
    """
    t = np.linspace(0.0, PLATE_L, n_side)
    X, Y = np.meshgrid(t, t, indexing="ij")
    coords = np.column_stack((X.ravel(), Y.ravel()))
    return coords, np.full(coords.shape[0], AREA / coords.shape[0])


def free_room(arm: str, h: float, n_side: int, room_m: float = FREE_ROOM):
    """A rigid room with the piston mounted ``arm``-wise, plus its port and its source origin."""
    n = int(round(room_m / h))
    box = AirBox(L=(n * h,) * 3, fs=air_fs(h), h=h, walls="rigid")
    coords, areas = flat_surface(n_side)
    if arm == "baffled":
        port = SurfacePort(room=box, face="z0", coords=coords, areas=areas)
        origin = np.array([(n // 2) * h, (n // 2) * h, 0.0])
    else:
        port = InteriorSurfacePort(
            room=box, plane="z", index=n // 2, coords=coords, areas=areas
        )
        origin = np.array([(n // 2) * h, (n // 2) * h, (n // 2 + 0.5) * h])
        if arm == "phantom":  # the cut dropped: a legal dipole SOURCE with no obstacle
            box._cut_mask = [None, None, None]
            box._cut_index = [None, None, None]
            box._cuts = []
            assert box.cut_faces == 0
    return box, port, origin


def radiation_resistance(arm: str, f: float, h: float, n_side: int, steady: float = 2.0):
    """``R = dE_injected / integral v^2 dt`` (kg/s) over whole cycles, inside the window.

    One cosine ramp cycle so the near field is established, then ``steady`` **whole** cycles over
    which the reactive part integrates out. The whole run is truncated before the first reflection
    reaches the source, so this is a free-field number read out of a finite rigid room.
    """
    box, port, _ = free_room(arm, h, n_side)
    k, period = box.k, 1.0 / f
    n_ramp = int(round(period / k))
    n_steady = int(round(steady * period / k))
    ones = np.ones(port.n_surface)
    e_start, v2 = 0.0, 0.0
    for step in range(n_ramp + n_steady):
        t = step * k
        env = 0.5 * (1.0 - np.cos(np.pi * min(1.0, t / period)))
        v = PISTON_V * env * np.sin(2.0 * np.pi * f * t)
        port.inject(port.T @ (v * ones))
        box.step()
        if step == n_ramp - 1:
            e_start = box.injected
        if step >= n_ramp:
            v2 += k * v * v
    window = (n_ramp + n_steady) * k
    reflection = box.L_actual[2] / box.c0  # source at the centre: round trip to the nearest wall
    return (box.injected - e_start) / v2, port, window, reflection


def fig_resistance():
    """Figure 1: the resistance sweep, its refinement, and the plane-wave ceiling."""
    sweep = {}
    for ka in KA_SWEEP:
        f = ka_frequency(ka)
        r_b, _, window, reflection = radiation_resistance("baffled", f, FREE_H, N_SURFACE)
        r_d, port, _, _ = radiation_resistance("dipole", f, FREE_H, N_SURFACE)
        sweep[ka] = (f, r_b, r_d, port.blocked_area / AREA, window, reflection)

    refine = {}
    for ka in KA_REFINE:
        f = ka_frequency(ka)
        rows = []
        for h, n_side in REFINE_GRIDS:
            r_b, _, _, _ = radiation_resistance("baffled", f, h, n_side)
            r_d, port, _, _ = radiation_resistance("dipole", f, h, n_side)
            rows.append((h, r_b / PLANE_WAVE, r_d / PLANE_WAVE, r_d / r_b,
                         port.blocked_area / AREA))
        refine[ka] = rows

    ka = np.array(KA_SWEEP)
    baf = np.array([sweep[q][1] / PLANE_WAVE for q in KA_SWEEP])
    dip = np.array([sweep[q][2] / PLANE_WAVE for q in KA_SWEEP])
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))
    ax[0].plot(ka, baf, "o-", label="baffled (batch 3)")
    ax[0].plot(ka, dip, "s-", label="interior dipole (batch 4)")
    ax[0].axhline(1.0, color="0.6", ls=":", lw=1)
    ax[0].axhline(2.0, color="0.6", ls="--", lw=1)
    ax[0].text(ka[0], 2.05, r"$2\rho_0 c_0 A$ — each face radiates plane waves", fontsize=8)
    ax[0].text(ka[0], 1.05, r"$\rho_0 c_0 A$", fontsize=8)
    ax[0].set_xlabel("ka"), ax[0].set_ylabel(r"$R\,/\,\rho_0 c_0 A$")
    ax[0].set_title("radiation resistance, prescribed piston")
    ax[0].legend(fontsize=8), ax[0].grid(alpha=0.3)

    ax[1].plot(ka, dip / baf, "d-", color="C3")
    ax[1].axhline(1.0, color="0.4", ls="--", lw=1)
    ax[1].set_xlabel("ka"), ax[1].set_ylabel("dipole / baffled")
    ax[1].set_title("the ratio CROSSES 1 — no constant can do that")
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(OUT, "airbox_dipole_resistance.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path, sweep, refine


def directivity(arm: str, f: float, cycles: float = 2.5):
    """Peak ``|p|`` on a quarter arc, windowed before the first image — SNAPPED geometry."""
    box, port, origin = free_room(arm, FREE_H, N_SURFACE)
    probes, actual = [], []
    for theta in ARC_THETAS:
        want = origin + ARC_R * np.array(
            [np.sin(np.radians(theta)), 0.0, np.cos(np.radians(theta))]
        )
        want = np.clip(want, 0.0, np.array(box.L_actual) - 1e-9)
        probes.append(box.node_index(want))
        got = np.array(box.snapped(want)) - origin
        actual.append((float(np.hypot(got[0], got[2])),
                       float(np.degrees(np.arctan2(got[0], got[2])))))
    k, period = box.k, 1.0 / f
    steps = int(round((cycles * period + (ARC_R + 0.6) / box.c0) / k))
    ones = np.ones(port.n_surface)
    peak = np.zeros(len(ARC_THETAS))
    for step in range(steps):
        t = step * k
        env = np.sin(np.pi * min(1.0, t / (cycles * period))) ** 2 if t <= cycles * period else 0.0
        v = PISTON_V * env * np.sin(2.0 * np.pi * f * t)
        port.inject(port.T @ (v * ones))
        box.step()
        for j, idx in enumerate(probes):
            peak[j] = max(peak[j], abs(box.p[idx]))
    image = (FREE_ROOM - ARC_R) / box.c0
    return peak, actual, steps * k, image


def fig_directivity():
    """Figure 2: the angular null — the claim that survives every magnitude caveat."""
    f = ka_frequency(ARC_KA)
    rows, geometry, window, image = {}, None, None, None
    for arm in ("dipole", "baffled", "phantom"):
        peak, actual, window, image = directivity(arm, f)
        rows[arm] = peak
        if arm == "dipole":
            geometry = actual

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    angles = np.array([g[1] for g in geometry])
    for arm, style in (("dipole", "s-"), ("baffled", "o-"), ("phantom", "^--")):
        ax.plot(angles, rows[arm] / rows[arm][0], style, label=arm)
    ax.plot(angles, np.cos(np.radians(angles)), "k:", lw=1, label=r"$\cos\theta$")
    ax.set_xlabel(r"$\theta$ from the plate's normal (degrees, SNAPPED)")
    ax.set_ylabel("normalized peak |p|")
    ax.set_title("the two-sided plate has a direction; the baffled one does not")
    ax.legend(fontsize=8), ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(OUT, "airbox_dipole_directivity.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path, rows, geometry, window, image


# -- the coupled rig (figure 3 and the two small measurements) ---------------------------
COUPLED_FS = 8000.0
COUPLED_N = (12, 11, 9)
COUPLED_INDEX = 4
COUPLED_WALLS = impedance_from_zeta(1.0)


def coupled_room(refine: int, walls=COUPLED_WALLS) -> AirBox:
    h = C0_AIR * np.sqrt(3.0) / (CFL * COUPLED_FS * refine)
    n = tuple(v * refine for v in COUPLED_N)
    return AirBox(L=tuple(v * h for v in n), fs=COUPLED_FS * refine, h=h, walls=walls)


def coupled_plate(room: AirBox, boundary: str, n_plate: int = 8) -> Plate:
    return Plate(Lx=PLATE_L, Ly=PLATE_L, kappa=20.0, rho=0.5, fs=room.fs, N=n_plate,
                 boundary=boundary)


def mode_state(plate: Plate, m: int, n: int) -> np.ndarray:
    x, y = plate.X[plate.mask], plate.Y[plate.mask]
    shape = np.sin(m * np.pi * x / plate.Lx) * np.sin(n * np.pi * y / plate.Ly)
    return 1e-3 * shape / np.sqrt(np.mean(shape * shape))


def seed(inst, ic: str) -> float:
    """Set the plate's initial condition and return ``E0``. ``piston`` / ``mode`` / ``mode21``."""
    if ic == "piston":
        inst.set_state(np.zeros(inst.plate.n_live), 1e-3 * np.ones(inst.plate.n_live))
    elif ic == "mode21":
        inst.set_state(mode_state(inst.plate, 2, 1))  # has interior nodal lines for the halo to
    else:                                             # extend across, unlike (1,1)
        inst.set_state(mode_state(inst.plate, 1, 1))
    return inst.plate.energy()


def t50(inst, ic: str, limit: int = 6000):
    """Steps for the plate to lose half its energy."""
    e0 = seed(inst, ic)
    for step in range(1, limit + 1):
        inst.step()
        inst.room.step()
        if inst.plate.energy() <= 0.5 * e0:
            return step
    return None


def fig_t50():
    """Figure 3: the headline — the source alone converges to silence."""
    table = {}
    for label, boundary, ic in (("supported (1,1)", "supported", "mode"),
                                ("free piston", "free", "piston")):
        rows = []
        for refine in (1, 2, 3):
            room = coupled_room(refine)
            dip = t50(RoomSuspendedPlate(plate=coupled_plate(room, boundary), room=room,
                                         plane="z", index=COUPLED_INDEX * refine), ic)
            room = coupled_room(refine)
            baf = t50(RoomLoadedPlate(plate=coupled_plate(room, boundary), room=room,
                                      face="z0"), ic)
            room = coupled_room(refine)
            ghost = RoomSuspendedPlate(plate=coupled_plate(room, boundary), room=room,
                                       plane="z", index=COUPLED_INDEX * refine)
            room._cut_mask = [None, None, None]
            room._cut_index = [None, None, None]
            room._cuts = []
            assert room.cut_faces == 0
            pha = t50(ghost, ic)
            rows.append((refine, dip, baf, pha))
        table[label] = rows

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    for label, rows in table.items():
        ratios = [(r, p / d) for r, d, _, p in rows if p is not None and d]
        if ratios:
            ax.plot([r for r, _ in ratios], [v for _, v in ratios], "o-", label=label)
            continue
        # A series with no finite ratio is the STRONGER result, not a missing one: say so on the
        # figure rather than letting it vanish into the legend's absence.
        ax.annotate(
            f"{label}: the phantom never reaches t50 at all\n(off the top of this axis at every "
            "refinement)",
            xy=(0.5, 0.06), xycoords="axes fraction", ha="center", fontsize=8, color="C1",
            bbox={"boxstyle": "round", "fc": "0.95", "ec": "C1"},
        )
    ax.set_xlabel("air-grid refinement (fs and room cells scale together)")
    ax.set_ylabel("t50 phantom / t50 dipole")
    ax.set_title("drop the cut and the source DIVERGES from the object")
    ax.set_xticks([1, 2, 3])
    ax.legend(fontsize=8), ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(OUT, "airbox_dipole_t50.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path, table


def blockage_overshoot():
    """§6.4: the obstacle against the **moving** surface's own rectangle, piston and mode alike.

    Normalized against the live footprint rather than ``Lx Ly``, because the cut is the support of
    ``T`` and a supported plate's clamped rim is not in ``T`` — so its obstacle is the *live*
    rectangle plus the halo, and dividing by the plate would show it converging to 0.5625 rather
    than to 1. The free plate's live rectangle IS the plate, which is also the configuration a
    physically suspended object is.
    """
    out = {}
    for boundary in ("supported", "free"):
        rows = []
        for refine in (1, 2, 3, 4):
            room = coupled_room(refine)
            plate = coupled_plate(room, boundary)
            inst = RoomSuspendedPlate(plate=plate, room=room, plane="z",
                                      index=COUPLED_INDEX * refine)
            x, y = plate.X[plate.mask], plate.Y[plate.mask]
            live = (x.max() - x.min()) * (y.max() - y.min())
            rows.append((room.h, inst.port.blocked_area / live,
                         inst.port.blocked_area / (plate.Lx * plate.Ly)))
        out[boundary] = rows

    # ... and what the overshoot is WORTH, on TWO probes and for a mode-shaped motion as well as a
    # piston (the rectangle grows asymmetrically relative to a mode's nodal lines, and only the
    # piston had ever been checked). The two probes disagree, which is the finding: the FRACTION
    # radiated saturates and is insensitive, while the RATE is not. The mode's fraction saturates
    # at exactly 1.0000 at every level, which is not the same thing as insensitive, so it is
    # reported as saturated rather than as a result.
    worth = {}
    for boundary, ic in (("free", "piston"), ("supported", "mode21")):
        rows = []
        for refine in (1, 2, 3, 4):
            room = coupled_room(refine)
            inst = RoomSuspendedPlate(plate=coupled_plate(room, boundary), room=room,
                                      plane="z", index=COUPLED_INDEX * refine)
            blocked = inst.port.blocked_area / (inst.plate.Lx * inst.plate.Ly)
            steps = t50(inst, ic)
            room = coupled_room(refine)
            inst = RoomSuspendedPlate(plate=coupled_plate(room, boundary), room=room,
                                      plane="z", index=COUPLED_INDEX * refine)
            e0 = seed(inst, ic)
            for _ in range(400 * refine):
                inst.step()
                inst.room.step()
            rows.append((blocked, None if steps is None else 1e3 * steps * inst.k,
                         inst.radiated_energy / e0))
        worth[f"{boundary}/{ic}"] = rows
    return out, worth


def k_only_refinement():
    """§5: fix ``h`` and the geometry, raise ``fs`` through the Courant fraction.

    The plate's *centered* velocity goes into a face slot that lives at ``n+1/2``. That choice is
    forced (the forward difference would put an added mass into the stability guard) and invisible
    to both ledgers, since each uses the same ``(q, pbar)`` pair. So it is bounded here instead —
    isolated from the blockage overshoot, which is what confounds a space refinement.
    """
    h = C0_AIR * np.sqrt(3.0) / (CFL * COUPLED_FS)
    rows = []
    for factor in (1, 2, 4, 8):
        fs = COUPLED_FS * factor
        room = AirBox(L=tuple(v * h for v in COUPLED_N), fs=fs, h=h, walls=COUPLED_WALLS)
        inst = RoomSuspendedPlate(plate=coupled_plate(room, "free"), room=room, plane="z",
                                  index=COUPLED_INDEX)
        inst.set_state(np.zeros(inst.plate.n_live), 1e-3 * np.ones(inst.plate.n_live))
        e0 = inst.plate.energy()
        for _ in range(400 * factor):
            inst.step()
            inst.room.step()
        rows.append((room.lam, inst.radiated_energy / e0))
    return rows


def main():
    print("air-box batch 4 — the plate that stops being a source\n")
    print(f"the prescribed-velocity rig: {PLATE_L} m square piston, A = {AREA:.4f} m^2, "
          f"a_eff = {A_EFF:.4f} m")
    print(f"   rho0 c0 A = {PLANE_WAVE:.4f} kg/s   (the baffled arm's high-ka asymptote; the "
          "dipole's is twice it)")
    print(f"   room {FREE_ROOM} m cube, h_air = {FREE_H} m, fs = {air_fs(FREE_H):.0f} Hz\n")

    t0 = time.perf_counter()
    p1, sweep, refine = fig_resistance()
    print(f"1. radiation resistance -> {p1}   [{time.perf_counter() - t0:.0f}s]")
    first = sweep[KA_SWEEP[0]]
    print(f"   window {first[4] * 1e3:.2f} ms, first reflection at the source "
          f"{first[5] * 1e3:.2f} ms — read free-field out of a finite rigid room")
    print(f"   {'ka':>5} {'f (Hz)':>8} {'baf/pwA':>9} {'dip/pwA':>9} {'dip/baf':>9}")
    for ka in KA_SWEEP:
        f, r_b, r_d, _, _, _ = sweep[ka]
        print(f"   {ka:5.2f} {f:8.1f} {r_b / PLANE_WAVE:9.3f} {r_d / PLANE_WAVE:9.3f} "
              f"{r_d / r_b:9.3f}")
    crossing = [ka for ka in KA_SWEEP if sweep[ka][2] / sweep[ka][1] >= 1.0]
    print(f"   the ratio crosses 1 between ka = {KA_SWEEP[KA_SWEEP.index(crossing[0]) - 1]:.2f} "
          f"and {crossing[0]:.2f} — a crossing no constant R can reproduce.")
    print(f"   blocked/plate = {sweep[KA_SWEEP[0]][3]:.3f} on this grid.\n")
    print("   the same sweep under AIR-GRID refinement — which arm converges, and which does not:")
    print(f"   {'ka':>5} {'h_air':>8} {'baf/pwA':>9} {'dip/pwA':>9} {'dip/baf':>9} "
          f"{'blocked':>8}")
    for ka in KA_REFINE:
        for h, b, d, r, blocked in refine[ka]:
            print(f"   {ka:5.2f} {h:8.4f} {b:9.3f} {d:9.3f} {r:9.3f} {blocked:8.2f}")
    print("   the BAFFLED arm rises monotonically toward its asymptote of 1 — the rig's own")
    print("   sanity check. The DIPOLE arm tracks the BLOCKED AREA, not h: at low ka its")
    print("   magnitude is set by the obstacle, and the obstacle is a rounding of the footprint")
    print("   onto the air grid. What survives every refinement is the ratio's SIGN vs 1.")
    peak_dip = max(d for ka in KA_REFINE for _, _, d, _, _ in refine[ka])
    print(f"   peak dipole R/(rho0 c0 A) over the refinement = {peak_dip:.2f} — a few per cent")
    print("   above the plane-wave 2, which a piston legitimately does near its first maximum.")
    print("   'ratio <= 2' is the WRONG criterion: the ratio reaches "
          f"{max(r for ka in KA_REFINE for _, _, _, r, _ in refine[ka]):.2f} because the baffled")
    print("   arm has not saturated yet. The criterion belongs on each arm, asymptotically.\n")

    t0 = time.perf_counter()
    p2, arc, geometry, window, image = fig_directivity()
    print(f"2. directivity -> {p2}   [{time.perf_counter() - t0:.0f}s]")
    print(f"   ka = {ARC_KA}, f = {ka_frequency(ARC_KA):.1f} Hz, r = {ARC_R} m; window "
          f"{window * 1e3:.2f} ms vs first image {image * 1e3:.2f} ms")
    print("   theta REQUESTED " + "".join(f"{t:8.0f}" for t in ARC_THETAS))
    print("   theta SNAPPED   " + "".join(f"{g[1]:8.1f}" for g in geometry))
    print("   r     SNAPPED   " + "".join(f"{g[0]:8.3f}" for g in geometry))
    for arm in ("dipole", "baffled", "phantom"):
        peak = arc[arm]
        print(f"   {arm:15s} " + "".join(f"{v / peak[0]:8.3f}" for v in peak))
    print("   cos theta       " + "".join(f"{np.cos(np.radians(t)):8.3f}" for t in ARC_THETAS))
    null = arc["dipole"][0] / arc["dipole"][-1]
    print(f"   -> a {null:.0f}x null in the plate's own plane, against a baffled plate whose arc "
          f"only reaches {arc['baffled'][-1] / arc['baffled'][0]:.3f}")
    print("      (and whose theta=90 probe sits ON the z0 wall — a half-weight baffle node, not")
    print("      the free-space counterpart of the dipole's in-plane point).")
    in_plane = arc["phantom"][-1] / arc["phantom"][0]
    print(f"   the PHANTOM has the same pattern (in-plane {in_plane:.3f}) at "
          f"{arc['dipole'][0] / arc['phantom'][0]:.1f}x less amplitude:")
    print("      directivity identifies the dipole; the BLOCKAGE sets its strength.\n")

    t0 = time.perf_counter()
    p3, table = fig_t50()
    print(f"3. t50 by mounting -> {p3}   [{time.perf_counter() - t0:.0f}s]")
    for label, rows in table.items():
        print(f"   {label}")
        for refine, dip, baf, pha in rows:
            ratio = "never reaches t50" if pha is None else f"{pha / dip:.1f}x the dipole"
            print(f"     {refine}x: dipole {dip}  baffled {baf}  phantom {pha}   "
                  f"dip/baf {dip / baf:.3f}   phantom {ratio}")
    print("   the phantom/dipole ratio GROWS with refinement — the assertion is the divergence,")
    print("   not a value. A doublet at separation h has moment proportional to h, so this is an")
    print("   IMPLEMENTATION control: it is what batch 4 degrades to if the cut is dropped.")
    print("   And read dip/baf against figure 1, which points the other way: t50 counts the")
    print("   reactive near field as though it had left, so the dipole sheds energy FASTER while")
    print("   RADIATING less. A decay time is not a radiation measure.\n")

    t0 = time.perf_counter()
    overshoot, worth = blockage_overshoot()
    print(f"4. the blockage overshoot (§6.4)   [{time.perf_counter() - t0:.0f}s]")
    print("   blocked_area / LIVE rectangle (and / the plate, which is the misleading one):")
    for boundary, rows in overshoot.items():
        print(f"     {boundary:10s} " + "  ".join(f"{v:.3f} ({w:.3f})" for _, v, w in rows))
    print("     the supported plate's clamped rim is not in T, so its obstacle is the LIVE")
    print("     rectangle plus the halo — dividing by the plate would show it heading for 0.5625.")
    print("   and what it is WORTH — on TWO probes, which disagree, and that is the finding:")
    for name, rows in worth.items():
        blocked = [b for b, _, _ in rows]
        times = [t for _, t, _ in rows if t is not None]
        frac = [f for _, _, f in rows]
        saturated = all(abs(f - 1.0) < 1e-6 for f in frac)
        print(f"     {name:18s} blocked " + "  ".join(f"{b:.2f}" for b in blocked)
              + f" ({max(blocked) / min(blocked):.2f}x)")
        print(f"     {'':18s} t50/ms  " + "  ".join(f"{v:.3f}" for v in times)
              + f" ({max(times) / min(times):.2f}x)")
        tail = " — SATURATED, not insensitive" if saturated else ""
        print(f"     {'':18s} rad/E0  " + "  ".join(f"{v:.4f}" for v in frac)
              + f" ({max(frac) / min(frac):.2f}x){tail}")
    print("     the FRACTION radiated is insensitive (the plate gives up the same share however")
    print("     big the obstacle is); the RATE is not, and figure 1's prescribed-velocity R agrees")
    print("     with the rate. So the plan's '+-3% while the area moves 33%' is true of the")
    print("     fraction and false of everything else.")

    t0 = time.perf_counter()
    rows = k_only_refinement()
    print(f"\n5. the k-only refinement (§5) — h and the geometry FIXED   "
          f"[{time.perf_counter() - t0:.0f}s]")
    print("   lambda  " + "  ".join(f"{lam:8.4f}" for lam, _ in rows))
    print("   rad/E0  " + "  ".join(f"{v:8.4f}" for _, v in rows))
    vals = [v for _, v in rows]
    print(f"   spread over a 8x drop in lambda: {max(vals) / min(vals):.4f}x — the half-step")
    print("   placement is worth less than that, isolated from the blockage overshoot.")


if __name__ == "__main__":
    main()
