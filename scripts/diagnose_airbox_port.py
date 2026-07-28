"""Diagnostics for the two-way body<->room port: :class:`physsynth.core.airbox.RoomLoadedBody`.

Air-box batch 1 made a room *audible*; this batch makes it **push back**. Three figures, one per
claim, and the third is the one that gets printed as a refusal rather than a result.

  1. **The ledger cancels.** The body's ``radiated_energy`` and the room's ``injected`` are the same
     number seen from the two sides of one terminal, so summing the two ledgers removes the coupling
     term from the conserved statement entirely. Drawn as four channels plus their flat total —
     note that the coupling channel is *not* monotone, unlike
     :class:`~physsynth.core.radiation.RadiatedBody`'s ``k R U^2``: the room hands energy back.
     That is the difference between a load and a room.
  2. **The reflection arrives, late, and moves when the wall moves.** Two rooms differing *only* in
     ``Lz``: the body's trajectory is **bit-identical** until the round trip ``2d`` completes and
     different immediately after. A lumped ``R(omega)`` cannot produce this at any order — a causal
     one-port's impulse response is a decaying exponential, never a delayed echo. The second panel
     is the scene version: two instruments in one room, the second at rest until the first's wave
     reaches it, and the arrival index is **Manhattan** (the 7-point stencil spreads one node per
     step), not ``r/c0``.
  3. **A point port's load is a grid quantity — the measured non-convergence.** Driving a port and
     reading its reactance gives the equivalent radius ``a_eff = rho0 / (4 pi M_a)``: what the port
     *is*, as a sphere. For a point port that is ``~ h/3.1``, so refining the grid **halves** it and
     doubles the added mass hung on the body — refinement makes the artifact worse. A fixed-radius
     ball barely moves over the same halvings: a factor of twenty in grid sensitivity. The expensive
     fine level lives here rather than in the suite.
  4. **And what a spread port's size actually is — once the room is out of the way.** A uniformly
     injecting ball's equivalent shell radius is ``5a/6``, the classic 6/5 shape factor (the mean
     potential of a uniformly charged sphere), *not* a pulsating shell of radius ``a``. But the
     room's own reactance is the larger term until the port is genuinely compact: the same port
     reads 8.6% high in a 0.5 m room and 0.3% high in a 1 m one. The right-hand panel sweeps the
     room and lets the curve cross the closed form, which is the attribution — not an error bar.

Run:  python scripts/diagnose_airbox_port.py
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
    RoomLoadedBody,
    RoomPort,
    impedance_from_zeta,
)
from physsynth.core.body import ModalBody  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out")
os.makedirs(OUT, exist_ok=True)

H = 0.05                       # m — 5 cm nodes, so a travel time is a countable number of steps
CFL = 0.9                      # fraction of the 3-D ceiling 1/sqrt(3)
FS = C0_AIR * np.sqrt(3.0) / (CFL * H)
BODY_FREQS = np.array([220.0, 337.0])
BODY_A = np.array([2e-3, 1.3e-3])   # m^2, radiation weights
BODY_M = 0.05                       # kg, modal mass
Q0 = np.array([1e-3, 5e-4])         # m


def instrument(room, at, q0=Q0, radius=None):
    body = ModalBody(
        freqs=BODY_FREQS, fs=room.fs, sigmas=0.0, masses=BODY_M, phi=1.0, radiation=BODY_A
    )
    inst = RoomLoadedBody(body=body, room=room, at=at, radius=radius)
    inst.set_state(q0)
    return inst


# -- 1. The ledger ------------------------------------------------------------------------------
def fig_ledger(steps=1400):
    room = AirBox(L=(0.5, 0.4, 0.3), fs=FS, h=H, walls=impedance_from_zeta(6.0))
    inst = instrument(room, (0.15, 0.15, 0.15))

    body, coupling, acoustic, dissipated = [], [], [], []
    for _ in range(steps):
        inst.step()
        room.step()
        body.append(inst.body.energy())
        coupling.append(inst.radiated_energy)
        acoustic.append(room.acoustic_energy())
        dissipated.append(room.dissipated)
    body = np.array(body)
    coupling = np.array(coupling)
    acoustic = np.array(acoustic)
    dissipated = np.array(dissipated)
    total = body + acoustic + dissipated          # the coupling term cancels IDENTICALLY
    t = np.arange(steps) / room.fs * 1e3

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.4))
    ax.plot(t, body, label="body  $E_{body}$")
    ax.plot(t, coupling, label=r"coupling  $\int \bar p\,U\,dt$")
    ax.plot(t, acoustic, label="air  (stored)")
    ax.plot(t, dissipated, label="walls  (absorbed)")
    ax.plot(t, total, "k--", lw=2, label="total  = body + air + walls")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("energy (J)")
    ax.set_title("the two ledgers are one number, twice")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # The coupling channel, magnified: it goes DOWN as well as up. RadiatedBody's cannot.
    ax2.plot(t, coupling, lw=1.0)
    ax2.axhline(0.0, color="k", lw=0.6)
    drift = (total.max() - total.min()) / abs(total[0])
    ax2.set_xlabel("time (ms)")
    ax2.set_ylabel(r"$\int \bar p\,U\,dt$  (J)")
    ax2.set_title("the coupling runs both ways\n(a lumped load's channel is monotone)")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    path = os.path.join(OUT, "airbox_port_ledger.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    back = float(np.min(np.diff(coupling)))
    return path, drift, back


# -- 2. The delayed echo -------------------------------------------------------------------------
def _history(nz, steps=44):
    room = AirBox(L=(0.5, 0.4, nz * H), fs=FS, h=H)
    inst = instrument(room, (0.15, 0.15, 0.15))
    hist = []
    for _ in range(steps):
        inst.step()
        room.step()
        hist.append(inst.body.q[0])
    return np.array(hist)


def fig_echo():
    reference = _history(12)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.4))
    arrivals = {}
    for nz, colour in ((6, "C0"), (8, "C1"), (10, "C2")):
        hist = _history(nz)
        differs = np.any(hist[:, None] != reference[:, None], axis=1)
        t = int(np.argmax(differs))
        arrivals[nz] = t
        ax.plot(hist - reference, color=colour, label=f"$L_z$ = {nz} nodes  (d = {nz - 3})")
        ax.axvline(t, color=colour, ls=":", lw=1)
    ax.set_xlabel("step")
    ax.set_ylabel(r"$q_1$ difference vs the deep room (m)")
    ax.set_title("exactly zero until the wave comes back\n"
                 "(bit-identical, then not: $2d + 1$ steps)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # The scene version: B at rest until A's wave arrives, and the index is Manhattan.
    room = AirBox(L=(0.5, 0.4, 0.3), fs=FS, h=H)
    a = instrument(room, (0.10, 0.20, 0.15))
    b = instrument(room, (0.40, 0.20, 0.15), q0=0.0)
    manhattan = sum(abs(x - y) for x, y in zip(a.port.index, b.port.index, strict=True))
    euclid = H * np.sqrt(
        sum((x - y) ** 2 for x, y in zip(a.port.index, b.port.index, strict=True))
    )
    euclid_steps = euclid / C0_AIR * room.fs
    trace, first = [], None
    for n in range(60):
        a.step()
        b.step()
        room.step()
        trace.append(b.body.q[0])
        if first is None and b.body.q[0] != 0.0:
            first = n
    ax2.plot(np.abs(trace) + 1e-30, lw=1.2)
    ax2.set_yscale("log")
    ax2.axvline(manhattan, color="C3", ls="--", label=f"Manhattan = {manhattan} nodes")
    ax2.axvline(euclid_steps, color="C4", ls="--", label=f"$r/c_0$ = {euclid_steps:.1f} steps")
    ax2.set_xlabel("step")
    ax2.set_ylabel(r"$|q_1|$ of the SECOND body (m)")
    ax2.set_title("two instruments, one room\n(the grid's precursor beats the wavefront)")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    path = os.path.join(OUT, "airbox_port_echo.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path, arrivals, first, manhattan, euclid_steps


# -- 3. What a port IS ---------------------------------------------------------------------------
# The same three low-ka frequencies the suite's cross-tier test averages, so the batch quotes
# ONE set of digits. a_eff falls slowly with frequency (ka is no longer small), so widening
# this list shifts every number below by a couple of percent -- worth knowing before
# comparing a figure here against a test comment.
SWEEP_FREQS = np.array([50.0, 75.0, 100.0])


def equivalent_radius(side, h, radius, cfl=CFL):
    """``a_eff = rho0 / (4 pi M_a)`` with ``M_a = Im Z / omega`` — the port's near-field size.

    Driven through the port's own two numbers, so this measures ``R_room`` as well as the geometry.
    """
    fs = C0_AIR * np.sqrt(3.0) / (cfl * h)
    room = AirBox(L=(side,) * 3, fs=fs, h=h, walls=impedance_from_zeta(1.0))
    port = RoomPort(room=room, at=tuple(0.5 * v for v in room.L_actual), radius=radius)
    n_steps = int(round(0.04 * fs))
    t = np.arange(n_steps) / fs
    q = np.exp(-0.5 * ((t - 4.0e-3) / 6.0e-4) ** 2)
    pbar = np.empty(n_steps)
    t0 = time.perf_counter()
    for n in range(n_steps):
        pbar[n] = port.free_pressure() + port.R_room * q[n]
        port.inject(q[n])
        room.step()
    wall = time.perf_counter() - t0
    P, Q = np.fft.rfft(pbar), np.fft.rfft(q)
    f = np.fft.rfftfreq(n_steps, 1.0 / fs)
    Z = np.interp(SWEEP_FREQS, f, (P / Q).real) + 1j * np.interp(SWEEP_FREQS, f, (P / Q).imag)
    m_a = Z.imag / (2.0 * np.pi * SWEEP_FREQS)
    return float(np.mean(RHO0_AIR / (4.0 * np.pi * m_a))), port.node_count, room.N, wall


# The fine level is why this figure lives in a diagnose script and not in the suite. cfl is fixed
# at 0.9 (the helpers' default), so fs is solved for -- and a_eff turns out to be Courant-invariant
# to five significant figures anyway, which is how we know it is a static near-field quantity.
LEVELS_H = (0.027, 0.0135, 0.00675)
BALL = 0.05
CONTRAST_ROOM = 0.5   # cheap; adequate for a ratio, wrong for a magnitude
ROOM_SWEEP = (0.4, 0.5, 0.7, 1.0, 1.4)


def fig_port_size():
    point, spread, sizes = [], [], []
    for h in LEVELS_H:
        a_pt, _, N, t_pt = equivalent_radius(CONTRAST_ROOM, h, None)
        a_sp, n_sp, _, t_sp = equivalent_radius(CONTRAST_ROOM, h, BALL)
        point.append(a_pt)
        spread.append(a_sp)
        sizes.append((N, n_sp, t_pt + t_sp))

    swept = [equivalent_radius(side, LEVELS_H[0], BALL)[0] for side in ROOM_SWEEP]
    ball = 5.0 * BALL / 6.0

    hs = np.array(LEVELS_H)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.6))
    ax.loglog(hs * 1e3, np.array(point) * 1e3, "o-",
              label="point port  ($a_{eff}$ ~ $h/3.1$)")
    ax.loglog(hs * 1e3, np.array(spread) * 1e3, "s-", label=f"ball, a = {BALL * 100:.0f} cm")
    ax.loglog(hs * 1e3, hs * 1e3 / 3.1, "k:", lw=1, label="$h/3.1$")
    ax.set_xlabel("grid spacing $h$ (mm)")
    ax.set_ylabel(r"equivalent radius $a_{eff}$ (mm)")
    ax.set_title("what a port IS, measured from its reactance\n"
                 "refining the grid makes a POINT port worse, not better")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    ax2.plot(np.array(ROOM_SWEEP), np.array(swept) / ball, "o-")
    ax2.axhline(1.0, color="C3", ls="--", label=r"the ball's $5a/6$ (the 6/5 shape factor)")
    ax2.set_xlabel("room side (m)   —   port radius fixed at 5 cm")
    ax2.set_ylabel(r"$a_{eff}\;/\;(5a/6)$")
    ax2.set_title("the excess is the ROOM's reactance, not the port's\n"
                  "(it crosses the closed form as the walls recede)")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    path = os.path.join(OUT, "airbox_port_size.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path, np.array(point), np.array(spread), sizes, np.array(swept) / ball


def main():
    p1, drift, back = fig_ledger()
    print(f"energy ledger           -> {p1}")
    print(f"  max drift of body + air + walls, relative     : {drift:.3e}")
    print(f"  most negative coupling increment (J)          : {back:.3e}"
          "   <- the room pushing BACK")

    p2, arrivals, first, manhattan, euclid_steps = fig_echo()
    print(f"the delayed echo        -> {p2}")
    for nz, t in arrivals.items():
        print(f"  Lz = {nz} nodes, port {nz - 3} away: first difference at step {t} "
              f"(2d + 1 = {2 * (nz - 3) + 1})")
    print(f"  second body first moves at step {first}: Manhattan = {manhattan}, "
          f"r/c0 = {euclid_steps:.1f}")

    p3, point, spread, sizes, swept = fig_port_size()
    print(f"what a port IS          -> {p3}")
    for h, a_pt, a_sp, (N, n_sp, wall) in zip(LEVELS_H, point, spread, sizes, strict=True):
        print(f"  h = {h * 1e3:6.3f} mm  N = {N}  ball = {n_sp:6d} nodes  [{wall:5.1f}s]")
        print(f"      point  a_eff = {a_pt * 1e3:7.3f} mm   (a_eff/h = {a_pt / h:.3f})")
        print(f"      ball   a_eff = {a_sp * 1e3:7.3f} mm")
    print(f"  point  a_eff ratios across halvings: {point[1] / point[0]:.3f}, "
          f"{point[2] / point[1]:.3f}   <- it HALVES: a grid quantity")
    print(f"  ball   a_eff ratios across halvings: {spread[1] / spread[0]:.3f}, "
          f"{spread[2] / spread[1]:.3f}   <- 20x less grid-sensitive")
    print(f"  a_eff / (5a/6) as the walls recede ({', '.join(f'{s:.1f} m' for s in ROOM_SWEEP)}):")
    print("      " + ", ".join(f"{v:.4f}" for v in swept)
          + "   <- the excess is the ROOM")


if __name__ == "__main__":
    main()
