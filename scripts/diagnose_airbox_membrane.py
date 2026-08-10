"""Diagnostics for the drumhead in the room: air-box batch 5.

The suite pins the coupling — the ledgers, the reduction, the ``f_ext`` path, the two areas, the
negative control. What it deliberately does not assert is the batch's **physical headline**, for
the same reason batch 4's did not: a radiation figure has to come from a motion that is *held*, in
a room large enough to window a reflection-free interval. That is this script.

**The claim.** A membrane has no coincidence frequency. Bending gives ``c_b(omega) = sqrt(kappa
omega)``, so a plate's structural wavelength shrinks *slower* than the acoustic one as frequency
rises and it crosses ``c0`` exactly once. A membrane's ``c = sqrt(T/rho)`` has no ``omega`` in it,
so for **every** mode

    k0 / beta  =  (omega / c0) / (omega / c)  =  c / c0

— one number, the same at every mode, set by the player tightening the head. Whether a drumhead
short-circuits is therefore not a property of the note; it is a property of the instrument.

**How that is measured here, and why not with a piston.** Prescribed *uniform* motion cannot show
it: a piston has no structural wavenumber at all, so ``c`` never enters. What does is the classic
corrugated-surface problem — prescribe ``v(x, y, t) = V sin(beta_x x) sin(beta_y y) sin(omega t)``
at **fixed** ``omega`` and sweep ``beta``. Then ``ka`` is constant, the surface, the room and the
grid are constant, and the only thing moving is the ratio ``k0/beta``. The efficiency curve that
comes out is *universal*: every resonator in this repo is a point on it, and what distinguishes the
families is only **where they sit and whether they move**.

  1. **The universal curve, and the two families on it.** Radiation efficiency
     ``sigma = R / (rho0 c0 A)`` against ``k0/beta``, both mountings. A membrane's modes pile up at
     a *single* abscissa (``c/c0``); a plate's walk along it as ``sqrt(kappa omega)/c0``. That is
     the whole claim in one picture, and it is a claim no ``R(omega)`` fitted to one mode can make.

  2. **The scheme's own coincidence, which the continuum claim does not have.** The 5-point
     Laplacian is dispersive, so the *discrete* phase speed falls below ``c`` at high wavenumber and
     a marginally supersonic head falls back under ``c0`` somewhere on the grid. Figure 2 is
     ``k0/beta`` against ``beta h`` for the shipped operator: flat at ``c/c0`` in the continuum,
     bending down and **crossing 1** on the grid. This is why the claim ships bracketed, with the
     measurement band held below the 1% dispersion knee.

Run:  python scripts/diagnose_airbox_membrane.py     (a few minutes)
"""

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
    SurfacePort,
)

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out")
os.makedirs(OUT, exist_ok=True)

CFL = 0.9               # fraction of the 3-D ceiling 1/sqrt(3), as everywhere else
SURFACE_L = 0.30        # m, square surface
DRIVE_V = 1e-3          # m/s
RATIO_SWEEP = (0.30, 0.40, 0.60, 0.80, 1.00, 1.25, 1.80, 2.60)

# TWO acoustic sizes, because the answer depends on which one you ask at -- see figure 1.
# (label, ka, h_air, room cube, surface nodes per side). Each room is sized so the burst is
# windowed before the first reflection AND the shortest wave swept keeps >3 air cells.
SCALES = (
    ("drum-sized  ka=1.2", 1.2, 0.050, 5.0, 21),
    ("large  ka=8",        8.0, 0.015, 1.8, 31),
)

# The band of k0/beta over which EVERY point of the ka=8 sweep keeps >=5 air cells per
# structural wave. Multipliers are quoted over this band only; outside it the low-sigma end is an
# aliasing floor as much as a cancellation floor, which is batch 4's "the crossing is the claim,
# the magnitudes are not" arriving in a new costume.
RESOLVED = (0.60, 1.25)

AREA = SURFACE_L * SURFACE_L
A_EFF = np.sqrt(AREA / np.pi)

# Where the two families sit on the curve. A Mylar head and a hard-tuned one; a plate's modes.
MYLAR_C = np.sqrt(3000.0 / 0.26)     # 107.4 m/s -> c/c0 = 0.313
TIGHT_C = 1.4 * C0_AIR
PLATE_KAPPA = 20.0                   # m^2/s, the batch-3/4 plate


def air_fs(h: float) -> float:
    return C0_AIR * np.sqrt(3.0) / (CFL * h)


def corrugated_surface(beta: float, n_surface: int):
    """A square patch carrying ``sin(beta_x x) sin(beta_y y)``, ``beta_x = beta_y = beta/sqrt(2)``.

    Deliberately not a :class:`~physsynth.core.membrane.Membrane`: the motion is *prescribed*, so
    there is no resonator here and no coupled ``radiated_energy`` to misread as radiation. What the
    patch borrows from model #4 is only its geometry — a square of live nodes with uniform areas.
    """
    t = np.linspace(0.0, SURFACE_L, n_surface)
    X, Y = np.meshgrid(t, t, indexing="ij")
    coords = np.column_stack((X.ravel(), Y.ravel()))
    areas = np.full(coords.shape[0], AREA / coords.shape[0])
    b = beta / np.sqrt(2.0)
    shape = (np.sin(b * X) * np.sin(b * Y)).ravel()
    return coords, areas, shape


def free_room(arm: str, beta: float, h: float, room_m: float, n_surface: int):
    n = int(round(room_m / h))
    box = AirBox(L=(n * h,) * 3, fs=air_fs(h), h=h, walls="rigid")
    coords, areas, shape = corrugated_surface(beta, n_surface)
    if arm == "baffled":
        port = SurfacePort(room=box, face="z0", coords=coords, areas=areas)
    else:
        port = InteriorSurfacePort(room=box, plane="z", index=n // 2, coords=coords, areas=areas)
    return box, port, shape, areas


def efficiency(arm, f, beta, h, room_m, n_surface, steady: float = 2.0):
    """``sigma = dE_injected / (rho0 c0 integral sum_i A_i v_i^2 dt)`` over whole cycles.

    One cosine-ramp cycle to establish the near field, then ``steady`` **whole** cycles over which
    the reactive part integrates out — batch 4's rig with a shaped velocity in place of a uniform
    one. The whole run is truncated before the first reflection reaches the source.
    """
    box, port, shape, areas = free_room(arm, beta, h, room_m, n_surface)
    k, period = box.k, 1.0 / f
    n_ramp = int(round(period / k))
    n_steady = int(round(steady * period / k))
    weighted = float(np.dot(areas, shape * shape))
    e_start, v2 = 0.0, 0.0
    for step in range(n_ramp + n_steady):
        t = step * k
        env = 0.5 * (1.0 - np.cos(np.pi * min(1.0, t / period)))
        amp = DRIVE_V * env * np.sin(2.0 * np.pi * f * t)
        port.inject(port.T @ (amp * shape))
        box.step()
        if step == n_ramp - 1:
            e_start = box.injected
        if step >= n_ramp:
            v2 += k * amp * amp * weighted
    window = (n_ramp + n_steady) * k
    reflection = box.L_actual[2] / box.c0
    return (box.injected - e_start) / (RHO0_AIR * C0_AIR * v2), window, reflection


def sweep_scale(label, ka, h, room_m, n_surface):
    """One acoustic size: efficiency against ``k0/beta`` at fixed ``omega``, both mountings."""
    f = ka * C0_AIR / (2.0 * np.pi * A_EFF)
    k0 = 2.0 * np.pi * f / C0_AIR
    rows, window, reflection = [], None, None
    print(f"   {label}:  f = {f:.0f} Hz, h_air = {h} m, room {room_m} m, {n_surface} nodes/side")
    for ratio in RATIO_SWEEP:
        beta = k0 / ratio
        sig_b, window, reflection = efficiency("baffled", f, beta, h, room_m, n_surface)
        sig_d, _, _ = efficiency("dipole", f, beta, h, room_m, n_surface)
        rows.append((ratio, sig_b, sig_d))
        print(f"     k0/beta = {ratio:4.2f}  baffled {sig_b:8.4f}   dipole {sig_d:8.4f}   "
              f"({2 * np.pi / beta / h:4.1f} air cells/wave, "
              f"{SURFACE_L * beta / (2 * np.pi):4.1f} periods on the patch)")
    assert window < reflection, "the window must close before the first reflection"
    return rows, f, window, reflection


def fig_universal_curve():
    """Figure 1: efficiency against ``k0/beta``, at two acoustic sizes, and where the families sit.

    **The size is not a detail and this is the batch's correction to its own plan.** The
    corrugated-surface threshold at ``k0/beta = 1`` is a statement about a surface large compared
    with *both* wavelengths. At ``ka = 8`` it is textbook — 900x across the sweep, each arm
    saturating at its own plane-wave asymptote (1 baffled, 2 suspended, which is batch 4's ceiling
    reached honestly). At ``ka = 1.2`` — a real drumhead's first few modes — **there is no
    threshold at all**: the surface is a compact source, and what suppression there is comes from
    fitting a structural period onto the patch, not from crossing 1.
    """
    scales = {}
    for label, ka, h, room_m, n_surface in SCALES:
        scales[label] = sweep_scale(label, ka, h, room_m, n_surface)

    ratios = np.array(RATIO_SWEEP)
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))
    for (label, (rows, _, _, _)), marker in zip(scales.items(), ("o", "s"), strict=True):
        ax[0].loglog(ratios, [r[1] for r in rows], marker + "-", label=f"baffled, {label}")
        ax[0].loglog(ratios, [r[2] for r in rows], marker + "--", label=f"suspended, {label}")
    ax[0].axvline(1.0, color="0.4", ls="--", lw=1)
    ax[0].axhline(1.0, color="0.7", ls=":", lw=1)
    ax[0].axhline(2.0, color="0.7", ls=":", lw=1)
    for label, c, style in (("Mylar head  $c/c_0=0.31$", MYLAR_C, ":"),
                            ("tight head  $c/c_0=1.40$", TIGHT_C, "-.")):
        ax[0].axvline(c / C0_AIR, color="C2", ls=style, lw=1)
        ax[0].text(c / C0_AIR * 1.03, 3e-3, label, fontsize=7, rotation=90)
    ax[0].set_xlabel(r"$k_0/\beta$   ( $=c/c_0$ for EVERY membrane mode )")
    ax[0].set_ylabel(r"$\sigma = R\,/\,\rho_0 c_0 A$")
    ax[0].set_title("the threshold is sharp — once the surface is big enough to have one")
    ax[0].legend(fontsize=7), ax[0].grid(alpha=0.3, which="both")

    # Where each family sits: kinematics, no simulation -- that IS the claim.
    modes = [(m, n) for m in range(1, 5) for n in range(1, 5)]
    beta_m = np.array([np.pi * np.hypot(m / SURFACE_L, n / SURFACE_L) for m, n in modes])
    for c, name, colour in ((MYLAR_C, "membrane $c/c_0=0.31$", "C2"),
                            (TIGHT_C, "membrane $c/c_0=1.40$", "C3")):
        ax[1].plot(beta_m, np.full_like(beta_m, c / C0_AIR), "o", color=colour, label=name)
    omega_plate = PLATE_KAPPA * beta_m**2
    ax[1].plot(beta_m, np.sqrt(PLATE_KAPPA * omega_plate) / C0_AIR, "^", color="C0",
               label=r"plate #5 $\sqrt{\kappa\omega}/c_0$")
    ax[1].axhline(1.0, color="0.4", ls="--", lw=1)
    ax[1].set_xlabel(r"structural wavenumber $\beta$ of the mode  [1/m]")
    ax[1].set_ylabel(r"$k_0/\beta$")
    ax[1].set_yscale("log")
    ax[1].set_title("a membrane's modes do not move; a plate's cross")
    ax[1].legend(fontsize=8), ax[1].grid(alpha=0.3, which="both")
    fig.tight_layout()
    path = os.path.join(OUT, "airbox_membrane_curve.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path, scales


def discrete_ratio(c_over_c0: float, lam: float, bh: np.ndarray) -> np.ndarray:
    """``k0/beta`` on the **grid**: the 5-point scheme's phase speed, worse (axis) direction.

    ``omega_d = (2/k) arcsin(lambda sin(beta h / 2))`` for an axis-aligned wave, so the discrete
    phase speed is ``omega_d/beta`` and ``k0/beta = (omega_d/beta)/c0`` — which the continuum claim
    says is the constant ``c/c0`` and which the scheme bends downward.
    """
    arg = np.clip(lam * np.sin(bh / 2.0), -1.0, 1.0)
    return c_over_c0 * (2.0 * np.arcsin(arg) / (lam * bh))


def fig_the_schemes_own_coincidence():
    """Figure 2: the trap — a marginally supersonic head goes subsonic on the grid."""
    bh = np.linspace(1e-3, np.pi, 400)
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    crossings = {}
    for c_ratio, colour in ((1.05, "C3"), (1.10, "C1"), (1.2247, "C0")):
        for lam, style in ((1.0 / np.sqrt(2.0), "-"), (0.5, "--")):
            y = discrete_ratio(c_ratio, lam, bh)
            ax.plot(bh, y, style, color=colour, lw=1.4,
                    label=f"$c/c_0$={c_ratio:.2f}, $\\lambda$={lam:.3f}")
            if lam > 0.7 and y[-1] < 1.0:
                idx = int(np.argmax(y < 1.0))
                crossings[c_ratio] = float(bh[idx])
    knee = float(bh[int(np.argmax(discrete_ratio(1.0, 1.0 / np.sqrt(2.0), bh) < 0.99))])
    ax.axhline(1.0, color="0.3", ls=":", lw=1.2)
    ax.axvline(knee, color="0.6", lw=1)
    ax.text(knee * 1.05, 1.15, f"1% knee\n$\\beta h$={knee:.3f}", fontsize=7)
    ax.set_xlabel(r"$\beta h$   (grid Nyquist at $\pi$)")
    ax.set_ylabel(r"$k_0/\beta$ on the grid")
    ax.set_title("the scheme manufactures the coincidence the continuum has not")
    ax.legend(fontsize=7), ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(OUT, "airbox_membrane_inversion.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path, crossings, knee


def main() -> None:
    print(__doc__.strip().splitlines()[0])
    print("=" * 96)

    t0 = time.perf_counter()
    print("\n1. the universal curve (figure 1) — fixed omega, only k0/beta moves")
    path1, scales = fig_universal_curve()
    print(f"   [{time.perf_counter() - t0:.0f}s]")
    for label, (rows, _f, window, reflection) in scales.items():
        baf = [r[1] for r in rows]
        dip = [r[2] for r in rows]
        print(f"   {label}: baffled {baf[0]:.4f} -> {max(baf):.4f} ({max(baf) / baf[0]:6.1f}x), "
              f"suspended {dip[0]:.4f} -> {max(dip):.4f} ({max(dip) / dip[0]:6.1f}x); "
              f"window {window * 1e3:.2f} ms < reflection {reflection * 1e3:.2f} ms")
    big = scales["large  ka=8"][0]
    small = scales["drum-sized  ka=1.2"][0]
    # The RESOLVED span, not the full sweep. Batch 4's doctrine applied to this batch's own
    # number: the sweep's bottom point sits at 2.7 air cells/wave, so its sigma is an aliasing
    # floor as much as a cancellation floor and the full-sweep multiplier is an UPPER BOUND.
    res = [r for r in big if RESOLVED[0] <= r[0] <= RESOLVED[1]]
    print("   THE CORRECTION THIS BATCH OWES ITS OWN PLAN: the threshold is a large-surface")
    print("   statement. At ka=8 it is textbook and each arm saturates at its own plane-wave")
    print(f"   asymptote ({max(r[1] for r in big):.2f} baffled -> 1, "
          f"{max(r[2] for r in big):.2f} suspended -> 2, batch 4's ceiling reached honestly).")
    print(f"   the rise across the knee, between RESOLVED points only (k0/beta "
          f"{RESOLVED[0]} .. {RESOLVED[1]}, >=5 air cells/wave): "
          f"{max(r[1] for r in res) / min(r[1] for r in res):5.1f}x baffled, "
          f"{max(r[2] for r in res) / min(r[2] for r in res):5.1f}x suspended.")
    full = max(r[1] for r in big) / min(r[1] for r in big)
    print(f"   the FULL sweep reads {full:.0f}x, but its bottom point is at 2.7 air cells/wave"
          " -- an aliasing floor as much as a")
    print("   cancellation one, so that is an UPPER BOUND. The KNEE at 1 is the claim and it is")
    print("   robust across 5.3 / 7.1 / 8.9 cells per wave; the multiplier is not.")
    print(f"   At ka=1.2 -- a real head's first modes -- the same sweep spans only "
          f"{max(r[1] for r in small) / min(r[1] for r in small):.0f}x and has NO knee at 1 (its")
    print("   shortest wave is 5.3 cells, so that one is resolved throughout): the head is quiet")
    print("   because it is COMPACT, not because it is subsonic.")
    print(f"   -> {path1}")

    print("\n2. where the families sit (figure 1, right) — kinematics, no simulation")
    print(f"   a membrane is ONE abscissa: c/c0 = {MYLAR_C / C0_AIR:.3f} (Mylar) at every mode.")
    modes = [(1, 1), (2, 2), (3, 3), (4, 4)]
    for m, n in modes:
        beta = np.pi * np.hypot(m / SURFACE_L, n / SURFACE_L)
        f_mem = MYLAR_C * beta / (2.0 * np.pi)
        f_pl = PLATE_KAPPA * beta * beta / (2.0 * np.pi)
        print(f"     mode ({m},{n})  beta = {beta:6.2f}   membrane {f_mem:7.1f} Hz "
              f"k0/beta = {MYLAR_C / C0_AIR:.3f}   plate {f_pl:8.1f} Hz "
              f"k0/beta = {np.sqrt(PLATE_KAPPA * 2 * np.pi * f_pl) / C0_AIR:.3f}")

    t0 = time.perf_counter()
    path2, crossings, knee = fig_the_schemes_own_coincidence()
    print(f"\n3. the scheme's own coincidence (figure 2)   [{time.perf_counter() - t0:.0f}s]")
    print(f"   1% dispersion knee at beta h = {knee:.3f} "
          f"({2 * np.pi / knee:.1f} nodes/wavelength) -- the band any claim must stay below")
    for c_ratio, bh in sorted(crossings.items()):
        print(f"   c/c0 = {c_ratio:.4f} falls back through 1 at beta h = {bh:.3f} "
              f"({2 * np.pi / bh:.1f} nodes/wavelength)")
    print("   every one of those is ON the grid, so a broadband strike on a marginally supersonic")
    print("   head genuinely contains a subsonic upper spectrum. The claim ships bracketed.")
    print(f"   -> {path2}")


if __name__ == "__main__":
    main()
