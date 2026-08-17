"""Diagnostics for the gong on a string, in the room: the three-way chain.

The suite pins the coupling — the guard bit-identity, the ledgers, the conserved total, passivity,
the ``nonlinear=False`` reduction, and the rigid share on a cheap rig. What it deliberately does
not assert is the batch's **physical headline**, for the reason air-box batches 4–6 did not
either: a radiation figure has to come from a motion that is *held*, over a window long enough for
the effect to develop. That is this script.

**What died first.** The obvious batch was air-box batch 6's headline with a string in place of
its strike: *a loud plate's radiation is time-varying at fixed geometry* — measured there as
``sigma_shape`` moving 46.0% at ``w/e = 3`` against 1.4% at ``w/e = 0.05``, 33x the control.
Section 1 reruns exactly that, on batch 6's rig, with batch 6's observable, at the same peak
``w/e = 3`` reached by a *string*. It is **flat to four digits**, and the loud arm's spread lands
*below its own quiet control*. There is no effect here with a sign, let alone a size.

**Why, and what replaced it.** ``w/e`` is not an amplitude when the drive is a **point force**.
A point force on a free plate feeds the ``{1, x, y}`` rigid nullspace, and rigid motion stretches
nothing — the von Karman coupling is a functional of *stretching*, so it is asleep in exactly that
fraction of the motion. Section 2 measures it: at matched peak displacement the string-driven
plate carries **~190x less energy** than a struck one (normalised by ``(w/e)^2``, since strain
energy is quadratic), and 95.5% of its motion is rigid.

**The claim.** Section 3. Hold the plate at batch 6's validated 1 mm and move the *string's*
fundamental across the plate's first flexural mode by changing the string's **length** only —
which holds the wave impedance ``sqrt(T rho)`` fixed, unlike changing tension — with the pluck
amplitude set to hold the string's initial **energy** constant:

    **whether a string can play a gong nonlinearly is decided by band overlap, not by how hard
    you pluck.**

Both arms are run, and they disagree about *where* the optimum sits, which is itself the point:
at a defensible pluck the plate's energy share peaks at ``f1/f_el = 1``, at a hard one at ``2``.
An optimum that moves with drive amplitude is something no linear chain can do. Neither peak
*value* is claimed as a magnitude — only the location moving, and the span between the worst and
best overlap.

Run:  python scripts/diagnose_string_vk_room.py      (about 12 minutes)
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.linalg import eigh  # noqa: E402

from physsynth.core.airbox import (  # noqa: E402
    C0_AIR,
    RHO0_AIR,
    AirBox,
    RoomSuspendedVKPlate,
    impedance_from_zeta,
)
from physsynth.core.connection import StringVKPlateBridge  # noqa: E402
from physsynth.core.plate import VKPlate  # noqa: E402
from physsynth.core.string_ideal import IdealString  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out")
os.makedirs(OUT, exist_ok=True)

# -- air-box batch 6's rig, verbatim, so every number here is comparable to its own -----------
CFL = 0.9
MAT = dict(E=2.0e11, e=1.0e-3, nu=0.3, rho=7800.0)   # test_vk_free.py's canonical steel sheet
PLATE_L = 0.10             # m, square
PLATE_N = 16
H_AIR = 0.0114             # m -- >=5 air cells per structural wave at the strike's p90 mode
ROOM_M = 0.6               # m cube. A RATIO survives a small room; a magnitude does not
FS = C0_AIR * np.sqrt(3.0) / (CFL * H_AIR)           # the ROOM sets the rate: 57.9 kHz
N_ROOM = int(round(ROOM_M / H_AIR))
AREA = PLATE_L * PLATE_L
WALLS = impedance_from_zeta(4.0)
MAX_SWEEPS = 200
CELLS_PER_WAVE = 5.0       # batch 5's resolution floor, applied per MODE

# -- the string. Tension and linear density fixed, so sqrt(T rho) is fixed and only the BAND
# moves; N scales with L so lambda stays at 0.9 (the bridge requires lambda < 1).
T_STR, RHO_STR = 60.0, 0.005
C_STR = np.sqrt(T_STR / RHO_STR)   # 109.5 m/s
LAM = 0.9
K_BRIDGE = 3.0e4           # N/m -- just inside the exact guard (margin 0.975) on this rig
DRIVE_INDEX = None         # set once in main(); NEVER left to default across compared runs

SECONDS_SIGMA = 0.12       # section 1: batch 6's window, so the comparison is like for like
WINDOWS = 4
SECONDS_SWEEP = 0.03       # sections 2-3
E_HARD = 0.4649            # J -- what a 43.4 mm pluck on the 0.6 m string carries (w/e ~ 3)
E_SOFT = 0.0025            # J -- amplitude 0.5-2.0% of string length across the whole sweep
RATIOS = (0.28, 0.50, 1.00, 2.00, 4.00)   # f1 / (the plate's first flexural mode)


def make_plate(nonlinear=True):
    return VKPlate(
        Lx=PLATE_L, Ly=PLATE_L, fs=FS, N=PLATE_N, boundary="free",
        nonlinear=nonlinear, couple_max_iter=MAX_SWEEPS, **MAT,
    )


def make_room():
    return AirBox(L=(N_ROOM * H_AIR,) * 3, fs=FS, h=H_AIR, walls=WALLS)


def first_flexural_hz(plate):
    """The free plate's lowest **elastic** mode — index 3, past the ``{1, x, y}`` nullspace."""
    vals, _ = eigh(plate.K.toarray() * plate.kappa**2, np.diag(plate.wdiag))
    return float(np.sqrt(np.clip(vals, 0.0, None))[3] / (2.0 * np.pi))


def rigid_basis(plate):
    """``W``-orthonormal ``span{1, x, y}`` — built, not diagonalised (model #5b's own result)."""
    x, y = plate.X[plate.mask], plate.Y[plate.mask]
    w = plate.wdiag
    cols = []
    for raw in (np.ones_like(x), x, y):
        v = np.asarray(raw, dtype=float).copy()
        for q in cols:
            v -= (q @ (w * v)) * q
        cols.append(v / np.sqrt(v @ (w * v)))
    return np.column_stack(cols)


def free_modes(plate):
    """Mass-orthonormal modes, frequencies, and which ones the AIR grid resolves (batch 5).

    Projection under ``W``, **never** a spectral peak: at ``w/e = 3`` a peak tracker reads a
    mode's frequency as 0.53x its own linear value, because the field has gone broadband.
    """
    vals, vecs = eigh(plate.K.toarray() * plate.kappa**2, np.diag(plate.wdiag))
    f = np.sqrt(np.clip(vals, 0.0, None)) / (2.0 * np.pi)
    beta = np.sqrt(2.0 * np.pi * np.maximum(f, 1e-9) / plate.kappa)
    return vecs, f, (2.0 * np.pi / beta) / H_AIR >= CELLS_PER_WAVE


def build(f1, energy, nonlinear=True):
    """The chain at a given string fundamental and pluck energy. Plate starts at rest."""
    plate = make_plate(nonlinear)
    room = make_room()
    inst = RoomSuspendedVKPlate(plate=plate, room=room, plane="z", index=N_ROOM // 2)
    L = C_STR / (2.0 * f1)
    s = IdealString(
        L=L, T=T_STR, rho=RHO_STR, fs=FS, N=int(np.floor(LAM * L * FS / C_STR)),
        boundary=("fixed", "free"),
    )
    bridge = StringVKPlateBridge(string=s, plate=inst, K=K_BRIDGE, drive_index=DRIVE_INDEX)
    amp = np.sqrt(4.0 * L * energy / (T_STR * np.pi**2))
    xs = np.linspace(0.0, L, s.N + 1)
    s.set_state(amp * np.sin(np.pi * xs / L))
    return bridge, inst, plate, room, amp, L


# -- 1. the dead headline: batch 6's sigma_shape, string-driven ------------------------------


def sigma_case(nonlinear, energy):
    """``sigma_shape`` per window, batch 6's expression verbatim, on a string-driven plate."""
    bridge, inst, plate, room, _, _ = build(F1_SIGMA, energy, nonlinear)
    vecs, _, resolved = free_modes(plate)
    vecs_r = vecs[:, resolved]
    load = inst.port.load_matrix     # T^T R T -- the room's own resistive operator
    e0 = bridge.energy() + room.energy()
    per = int(SECONDS_SIGMA * FS) // WINDOWS
    shape, shape_r, shares, in_band = [], [], [], []
    sweeps, conv, w_peak = 0, True, 0.0
    for _ in range(WINDOWS):
        acc = np.zeros(vecs.shape[1])
        p_shape = p_res = v2 = v2_res = 0.0
        for _ in range(per):
            v = (plate.u - plate.u_prev) / plate.k
            c = vecs.T @ (plate.wdiag * v)
            v_res = vecs_r @ c[resolved]
            p_shape += 2.0 * float(v @ (load @ v))       # 2 faces: the suspended tier
            p_res += 2.0 * float(v_res @ (load @ v_res))
            v2 += float(np.dot(plate.wdiag * v, v)) / AREA
            v2_res += float(np.dot(plate.wdiag * v_res, v_res)) / AREA
            acc += c * c
            bridge.step()
            room.step()
            sweeps = max(sweeps, plate.n_iters)
            conv &= plate.converged
            w_peak = max(w_peak, float(np.max(np.abs(plate.u))))
        shape.append((p_shape / per) / (RHO0_AIR * C0_AIR * AREA * (v2 / per)))
        shape_r.append((p_res / per) / (RHO0_AIR * C0_AIR * AREA * (v2_res / per)))
        shares.append(acc / acc.sum())
        in_band.append(v2_res / v2)
    shares = np.array(shares)
    return dict(
        shape=np.array(shape), shape_r=np.array(shape_r), sweeps=sweeps, conv=conv,
        we=w_peak / plate.e, in_band=np.array(in_band), n_res=int(resolved.sum()),
        n_modes=resolved.size, radiated=inst.radiated_energy / abs(e0),
        drift_modal=0.5 * float(np.abs(shares[-1] - shares[0]).sum()),
    )


# -- 2 & 3. the mechanism and the claim ------------------------------------------------------


def sweep_case(f1, energy, nonlinear=True):
    """One point of the band sweep: rigid share, plate energy share, peak w/e, the history."""
    bridge, inst, plate, room, amp, L = build(f1, energy, nonlinear)
    basis, w = rigid_basis(plate), plate.wdiag
    steps = int(SECONDS_SWEEP * FS)
    hist = np.empty((steps, plate.n_live))
    e0 = bridge.energy() + room.energy()
    rigid = total = 0.0
    w_peak = e_plate = worst = 0.0
    sweeps, conv = 0, True
    for n in range(steps):
        v = (plate.u - plate.u_prev) / plate.k
        c = basis.T @ (w * v)
        rigid += float(c @ c)
        total += float(v @ (w * v))
        bridge.step()
        room.step()
        hist[n] = plate.u
        worst = max(worst, abs(bridge.energy() + room.energy() - e0))
        w_peak = max(w_peak, float(np.max(np.abs(plate.u))))
        e_plate = max(e_plate, plate.energy())
        sweeps = max(sweeps, plate.n_iters)
        conv &= plate.converged
    return dict(rigid=rigid / total, we=w_peak / plate.e, share=e_plate / e0, hist=hist,
                e_plate=e_plate, sweeps=sweeps, conv=conv, drift=worst / abs(e0),
                amp=amp, L=L)


def struck_case(we0):
    """Batch 6's own excitation for the same plate — a centred raised Gaussian, no string."""
    plate = make_plate(True)
    room = make_room()
    inst = RoomSuspendedVKPlate(plate=plate, room=room, plane="z", index=N_ROOM // 2)
    width = 0.20 * plate.Lx
    dx, dy = plate.X - 0.5 * plate.Lx, plate.Y - 0.5 * plate.Ly
    inst.set_state((we0 * plate.e * np.exp(-((dx * dx + dy * dy) / (width * width))))[plate.mask])
    w_peak = e_plate = 0.0
    for _ in range(int(SECONDS_SWEEP * FS)):
        inst.step()
        room.step()
        w_peak = max(w_peak, float(np.max(np.abs(plate.u))))
        e_plate = max(e_plate, plate.energy())
    return dict(we=w_peak / plate.e, e_plate=e_plate)


def figure(sigma_cases, soft, hard, path):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    x = np.arange(1, WINDOWS + 1)
    for (label, nl), c in sigma_cases.items():
        colour = "0.45" if label == "quiet" else "C3"
        style = "o-" if nl else "s--"
        axes[0].plot(x, c["shape"] / c["shape"][0], style, color=colour, ms=4,
                     label=f"{label}, {'nonlinear' if nl else 'linear'}")
    axes[0].axhline(1.0, color="0.3", ls=":", lw=1.0)
    axes[0].set_xticks(x)
    axes[0].set_xlabel(f"window ({1e3 * SECONDS_SIGMA / WINDOWS:.0f} ms each)")
    axes[0].set_ylabel("efficiency, relative to window 1")
    axes[0].set_title(r"1. DEAD: $\sigma_{shape}$ does not move with the pluck", fontsize=10)
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.3)

    for arm, cases, colour, mark in (("defensible", soft, "C0", "o"), ("hard", hard, "C3", "s")):
        axes[1].plot(RATIOS, [100 * c["rigid"] for c in cases], mark + "-", color=colour,
                     label=f"{arm} pluck", ms=5)
        axes[2].plot(RATIOS, [100 * c["share"] for c in cases], mark + "-", color=colour,
                     label=f"{arm} pluck", ms=5)
    for ax, ylab, title in (
        (axes[1], "rigid share of plate motion (%)", "2. the nullspace empties as the bands meet"),
        (axes[2], "plate's peak energy share (%)", "3. and the optimum MOVES with the pluck"),
    ):
        ax.set_xscale("log")
        ax.set_xticks(RATIOS)
        ax.set_xticklabels([f"{r:g}" for r in RATIOS])
        ax.axvline(1.0, color="0.3", ls=":", lw=1.0)
        ax.set_xlabel(r"string $f_1$ / plate's first flexural mode")
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.suptitle(
        "whether a string can play a gong nonlinearly is decided by band overlap, "
        "not by how hard you pluck",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def main():
    global DRIVE_INDEX, F1_SIGMA
    print(__doc__.strip().splitlines()[0])
    print("=" * 96)

    probe = make_plate()
    f_el = first_flexural_hz(probe)
    DRIVE_INDEX = probe.pickup_index_at(0.3 * probe.Lx, 0.4 * probe.Ly)
    F1_SIGMA = 0.28 * f_el     # the 0.6 m string: batch 6's rig as a musician would string it
    print(f"\nthe cell: {PLATE_L * 1e3:.0f} mm FREE steel plate, e = {MAT['e'] * 1e3:.0f} mm, "
          f"N = {PLATE_N}, SUSPENDED in a {ROOM_M} m cube at h = {H_AIR * 1e3:.1f} mm "
          f"({N_ROOM}^3), fs = {FS / 1e3:.1f} kHz")
    print(f"the plate's first FLEXURAL mode is {f_el:.1f} Hz (its {{1, x, y}} nullspace is at 0)")
    print(f"drive index {DRIVE_INDEX}, pinned once and passed to EVERY run: two runs that derive")
    print("it separately can disagree, and that moves a comparison with no ledger turning red")

    # -- 1 -----------------------------------------------------------------------------------
    print(f"\n1. THE DEAD HEADLINE -- batch 6's sigma_shape, string-driven ({F1_SIGMA:.1f} Hz)")
    sigma_cases, t0 = {}, time.perf_counter()
    for label, energy in (("quiet", E_HARD / 3600.0), ("loud", E_HARD)):
        for nl in (True, False):
            sigma_cases[(label, nl)] = sigma_case(nl, energy)
    ref = sigma_cases[("quiet", True)]
    print(f"   {ref['n_res']} of {ref['n_modes']} modes keep >= {CELLS_PER_WAVE:.0f} air cells "
          f"per structural wave, carrying {100 * ref['in_band'][0]:.1f}% of the kinetic energy")
    print(f"   {'':22s} {'w1':>7s} {'w2':>7s} {'w3':>7s} {'w4':>7s} {'spread':>8s} "
          f"{'modal drift':>12s} {'peak w/e':>9s}")
    for (label, nl), c in sigma_cases.items():
        rel = c["shape"] / c["shape"][0]
        tag = f"{label}, {'nonlinear' if nl else 'LINEAR   '}"
        print(f"   {tag:22s} " + " ".join(f"{v:7.4f}" for v in rel) +
              f" {max(c['shape']) / min(c['shape']):8.4f} {c['drift_modal']:12.4f} "
              f"{c['we']:9.4f}")
    loud, quiet = sigma_cases[("loud", True)], sigma_cases[("quiet", True)]
    s_loud = max(loud["shape"]) / min(loud["shape"])
    s_quiet = max(quiet["shape"]) / min(quiet["shape"])
    print("\n   Batch 6's STRIKE at this same peak w/e gives 46.0% against 1.4%, i.e. 33x the")
    print(f"   control. Here: {100 * (s_loud - 1):.2f}% against {100 * (s_quiet - 1):.2f}%, and "
          f"the loud arm is {'BELOW' if s_loud < s_quiet else 'above'} its own control.")
    print(f"   Radiated over the run: {100 * loud['radiated']:.4f}% of the scene total -- in this")
    print("   chain the room is a READ-OUT and not a variable, which is why the replacement claim")
    print("   had to come from the plate's MOTION rather than from anybody's books.")
    print(f"   ({time.perf_counter() - t0:.0f}s)")

    # -- 2 -----------------------------------------------------------------------------------
    print("\n2. WHY -- w/e is not an amplitude when the drive is a POINT FORCE")
    driven = sweep_case(F1_SIGMA, E_HARD)
    hit = struck_case(1.727)
    print(f"   string-driven, {driven['amp'] * 1e3:.1f} mm pluck: peak w/e = {driven['we']:.3f}, "
          f"plate energy {driven['share'] * 100:.2f}% of the scene, rigid share "
          f"{100 * driven['rigid']:.2f}%")
    print(f"   struck, Gaussian w0/e = 1.727:            peak w/e = {hit['we']:.3f}, "
          f"plate energy {hit['e_plate']:.4e} J")
    raw = hit["e_plate"] / driven["e_plate"]
    norm = (hit["e_plate"] / hit["we"] ** 2) / (driven["e_plate"] / driven["we"] ** 2)
    print(f"   -> {raw:.0f}x more energy at matched peak, {norm:.0f}x normalised by (w/e)^2")
    print("      (strain energy is quadratic, so the normalised figure is the one to quote --")
    print("       an unnormalised one measured at 2.997 against a 5.213 that RANG UP read 781x,")
    print("       and that number was wrong. Batch 6's own warning, inside the batch citing it.)")
    print("   Rigid motion stretches NOTHING (l(w,w) = 0), so the von Karman coupling is asleep")
    print(f"   in {100 * driven['rigid']:.1f}% of the string-driven motion. Same displacement,")
    print("   different physics -- which is why section 3 measures band overlap, not amplitude.")
    print("   (the struck arm's own rigid share is 0.00% by IDENTITY, not by measurement: a")
    print("    displacement start gives the nullspace no velocity, so it proves nothing alone.)")

    # -- 3 -----------------------------------------------------------------------------------
    print("\n3. THE CLAIM -- band overlap, at two pluck energies")
    soft, hard = [], []
    for arm, energy, into in (("defensible", E_SOFT, soft), ("hard", E_HARD, hard)):
        print(f"\n   {arm} pluck ({energy:.4f} J)")
        print(f"   {'f1/f_el':>8s} {'f1(Hz)':>8s} {'L(mm)':>7s} {'amp/L':>7s} {'rigid':>8s} "
              f"{'plate E':>8s} {'peak w/e':>9s} {'sweeps':>7s} {'departure':>11s} {'drift':>9s}")
        for r in RATIOS:
            nl = sweep_case(r * f_el, energy, True)
            li = sweep_case(r * f_el, energy, False)
            dep = np.linalg.norm(nl["hist"] - li["hist"]) / np.linalg.norm(li["hist"])
            nl["departure"] = dep
            into.append(nl)
            print(f"   {r:8.2f} {r * f_el:8.1f} {nl['L'] * 1e3:7.1f} "
                  f"{nl['amp'] / nl['L']:7.1%} {nl['rigid']:7.2%} {nl['share']:7.2%} "
                  f"{nl['we']:9.3f} {nl['sweeps']:7d} {dep:11.4e} {nl['drift']:9.1e}")

    for arm, cases in (("defensible", soft), ("hard", hard)):
        deps = [c["departure"] for c in cases]
        best = RATIOS[int(np.argmax([c["share"] for c in cases]))]
        print(f"\n   {arm}: departure spans {max(deps) / min(deps):.0f}x between the WORST and "
              f"BEST overlap (a range,")
        print(f"   not a trend -- the column is non-monotone), and the plate's energy share peaks "
              f"at f1/f_el = {best:g}.")
    print("\n   The optimum MOVES with the pluck, and that is the second claim: no linear chain")
    print("   can do it. Neither peak VALUE is claimed as a magnitude -- only the location")
    print("   moving, and the span. (Batch 6's rule: the crossing is the claim, the size is not.)")

    path = figure(sigma_cases, soft, hard, os.path.join(OUT, "string_vk_room.png"))
    print(f"\nfigure -> {path}")


if __name__ == "__main__":
    main()
