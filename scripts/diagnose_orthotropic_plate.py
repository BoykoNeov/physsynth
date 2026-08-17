"""Orthotropic-plate (model #5o) diagnostic harness — the plate with a **grain**.

Model #5 assumes a material equally stiff in every direction. This runs the same plate made of
wood, and prints the numbers the batch's claims rest on so the model can be judged from the console
alone:

1. **Energy first, and why it proves less than it looks like it proves.** Three *deliberately
   mis-wired* grains — swapped end for end, the factor of 2 dropped, the cross term taken as `D_1`
   alone — all conserve to machine precision. Only the modal oracle separates them.
2. **Detected vs the closed-form orthotropic oracle**, the tight tier, in cents.
3. **The cross-term sweep**: hold the along/across stiffness ratio fixed and move only the cross
   term. It detunes the modes by 1.3% to 29% — and does *not* reorder them anywhere between solid
   spruce and isotropic material.
4. **Where the grain actually shows up in a coupled instrument**: the partial series carries it,
   the level at a single node does not.
5. Mode shapes, isotropic against spruce, so the elongation is visible.

Run:  python scripts/diagnose_orthotropic_plate.py
Figures land in ./out/.
"""

from __future__ import annotations

import os
import sys

# Make `physsynth` importable when run directly (no editable install): repo root on sys.path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from physsynth.analysis import modal  # noqa: E402
from physsynth.core.engine import simulate  # noqa: E402
from physsynth.core.exciter import raised_cosine_2d  # noqa: E402
from physsynth.core.operators2d import embed  # noqa: E402
from physsynth.core.plate import Plate, grain_ratios_from_material  # noqa: E402
from physsynth.viz import plots  # noqa: E402

KAPPA, RHO, LX, LY = 20.0, 0.005, 1.0, 1.0  # square, so shape asymmetry cannot fake a material one
OUT = "out"

# Sitka spruce, roughly. 11 GPa along the grain against 0.8 GPa across it.
SPRUCE = dict(E_x=11.0e9, E_y=0.8e9, nu_xy=0.37, G_xy=0.7e9, thickness=3.0e-3, rho=420.0)


def build(N, mu, *, sigma=0.0, grain=(1.0, 1.0, 1.0), Lx=LX, Ly=LY):
    h = Lx / N
    fs = KAPPA / (mu * h * h)  # invert mu = kappa k / h^2
    gx, gh, gy = grain
    return Plate(
        Lx=Lx, Ly=Ly, kappa=KAPPA, rho=RHO, fs=fs, N=N, sigma=sigma,
        grain_x=gx, grain_cross=gh, grain_y=gy,
    )


def pluck(p, center, width, amp=1e-3):
    f = raised_cosine_2d(p.X, p.Y, center, width, amplitude=amp)
    f[~p.mask] = 0.0
    return f


def sine(p, m, n):
    jj, ii = np.nonzero(p.mask)
    return np.sin(m * np.pi * ii * p.h / p.Lx) * np.sin(n * np.pi * jj * p.h / p.Ly)


def mode_freqs(p, modes):
    mn = np.asarray(modes, dtype=int)
    return np.asarray(
        modal.discrete_orthotropic_plate_eigenfrequency(
            modal.dirichlet_axis_eigenvalue(mn[:, 0], p.Lx, p.h),
            modal.dirichlet_axis_eigenvalue(mn[:, 1], p.Ly, p.h),
            p.kappa, p.k, p.theta, p.grain_x, p.grain_cross, p.grain_y,
        )
    )


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    _spec = grain_ratios_from_material(**SPRUCE)
    gx, gh, gy = _spec.grain_x, _spec.grain_cross, _spec.grain_y
    stretched = np.sqrt(gx * gy)

    print("Spruce, as three bending-stiffness ratios:")
    print(f"  along grain g_x = {gx:.4f}   across g_y = {gy:.4f}   ratio = {gx / gy:.2f}")
    print(f"  cross term g_h  = {gh:.4f}   = {gh / stretched:.3f} x sqrt(g_x g_y)")
    print("  a 'stretched isotropic' plate would be FORCED to 1.000 x -- the cross term is a")
    print("  genuinely independent axis, which is what the sweep in section 3 tests.")

    # --- 1. Energy conserves for the RIGHT grain and for three WRONG ones alike ---
    print("\n1. Energy drift, correct grain and three deliberate mis-wirings (~0.3 s):")
    probes = [(1, 1), (2, 1), (1, 2), (2, 2)]
    variants = {
        "correct        ": (gx, gh, gy),
        "swapped x<->y  ": (gy, gh, gx),
        "factor 2 dropped": (gx, gh / 2.0, gy),
        "H = D_1 only   ": (gx, gh * 0.30, gy),
    }
    f_ref = None
    for name, g in variants.items():
        p = build(32, 2.0, grain=g)
        f = mode_freqs(p, probes)
        if f_ref is None:
            f_ref = f
        p.set_state(pluck(p, (0.4 * LX, 0.55 * LY), 0.25 * LX))
        drift = simulate(p, num_steps=int(0.3 * p.fs)).energy_drift
        off = np.max(np.abs(f - f_ref) / f_ref) * 100.0
        print(f"  {name}  drift = {drift:.2e}   f_11 = {f[0]:7.2f} Hz   off truth = {off:5.1f} %")
    print("  -> every one of them conserves. The ledger is the WRONG detector for a coefficient;")
    print("     the oracle is the right one. (Same shape as the volume-vs-areal density slip.)")
    print("     NOTE f_11 is IDENTICAL for the swapped grain -- on a SQUARE plate every diagonal")
    print("     mode (m,m) is exactly blind to the grain running 90 degrees wrong. Check an")
    print("     off-diagonal mode, or use a non-square plate:")
    swap_probes = [("square   ", (1.0, 1.0), (2, 1)), ("rectangle", (0.62, 0.43), (1, 1))]
    for tag, (lx, ly), mode in swap_probes:
        a = build(32, 2.0, grain=(gx, gh, gy), Lx=lx, Ly=ly)
        b = build(32, 2.0, grain=(gy, gh, gx), Lx=lx, Ly=ly)
        ra = mode_freqs(a, [mode])[0]
        rb = mode_freqs(b, [mode])[0]
        print(f"       {tag} mode {mode[0]},{mode[1]}: {ra:8.2f} vs {rb:8.2f} Hz "
              f"({abs(rb / ra - 1) * 100:5.1f} % apart)")

    # --- 2. Detected vs the closed-form orthotropic oracle ---
    print("\n2. Assembled operator vs closed form (N=96, the tight tier):")
    p = build(96, 0.5, grain=(gx, gh, gy))
    worst_res = 0.0
    for m, n in [(1, 1), (2, 1), (1, 2), (2, 2), (3, 2), (2, 3)]:
        v = sine(p, m, n)
        lx = float(modal.dirichlet_axis_eigenvalue(m, p.Lx, p.h))
        ly = float(modal.dirichlet_axis_eigenvalue(n, p.Ly, p.h))
        q = gx * lx * lx + 2.0 * gh * lx * ly + gy * ly * ly
        worst_res = max(worst_res, np.linalg.norm(p.B @ v - q * v) / (q * np.linalg.norm(v)))
    f_disc = mode_freqs(p, [(1, 1), (2, 1), (1, 2), (2, 2), (3, 2), (2, 3)])
    f_cont = modal.orthotropic_plate_freqs(
        KAPPA, p.Lx, p.Ly, [(1, 1), (2, 1), (1, 2), (2, 2), (3, 2), (2, 3)], gx, gh, gy
    )
    labels = ["1,1", "2,1", "1,2", "2,2", "3,2", "2,3"]
    print(f"  sine is an exact discrete eigenvector: worst residual {worst_res:.2e}")
    for lab, fd, fc in zip(labels, f_disc, f_cont, strict=True):
        print(f"  mode {lab}  disc={fd:8.3f}  oracle={fc:8.3f}  err={modal.cents(fd, fc):+7.3f} c")

    fig, ax = plt.subplots(figsize=(7, 4))
    plots.plot_membrane_partials(ax, f_disc, f_cont, labels=labels)
    worst_c = float(np.nanmax(np.abs(modal.cents(f_disc, f_cont))))
    ax.set_title(f"Spruce plate vs orthotropic oracle (worst |err| = {worst_c:.3f} cents)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "orthotropic_partials.png"), dpi=110)
    plt.close(fig)

    # --- 3. The cross-term sweep: selective detuning, and NO reordering ---
    print("\n3. Cross-term sweep at FIXED stiffness ratio (g_x/g_y held at 13.75):")
    modes = [(m, n) for m in range(1, 5) for n in range(1, 5)]
    factors = np.array([0.2, 0.4, 0.567, 0.8, 1.0, 1.5, 2.0])
    gyy = gx / 13.75
    curves, orders = [], []
    for fac in factors:
        pf = build(48, 1.0, grain=(gx, fac * np.sqrt(gx * gyy), gyy))
        f = mode_freqs(pf, modes)
        curves.append(f)
        orders.append(tuple(np.argsort(f)))
    curves = np.array(curves)

    physical = [i for i, fac in enumerate(factors) if fac <= 1.0]
    same = len(set(orders[i] for i in physical)) == 1
    print(f"  ordering identical from 0.2x to 1.0x (wood 0.567x .. isotropic 1.0x): {same}")
    print(f"  ordering at 2.0x differs from 1.0x: {orders[-1] != orders[physical[-1]]}")
    lo, hi = curves[0], curves[physical[-1]]
    shift = hi / lo - 1.0
    by_mode = dict(zip(modes, shift, strict=True))
    print("  per-mode shift across 0.2x -> 1.0x (the SELECTIVITY):")
    for m in [(3, 1), (3, 2), (2, 2), (1, 3), (2, 3), (2, 4)]:
        print(f"    mode {m[0]},{m[1]}  {by_mode[m] * 100:+6.1f} %")
    print(f"  spread in leverage: {shift.max() / shift.min():.1f}x  "
          f"(largest where the DIRECT stiffness is weakest -- across the grain)")

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for j, m in enumerate(modes[:8]):
        ax.plot(factors, curves[:, j], marker="o", ms=3, label=f"{m[0]},{m[1]}")
    ax.axvline(0.567, ls="--", c="0.4", lw=1)
    ax.axvline(1.0, ls=":", c="0.4", lw=1)
    ax.set_xlabel(
        "cross term / sqrt(g_x g_y)   [dashed = spruce 0.567, dotted = isotropic 1.0]"
    )
    ax.set_ylabel("mode frequency (Hz)")
    ax.set_title("Cross term detunes selectively; the curves do not cross below 1.0x")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "orthotropic_cross_sweep.png"), dpi=110)
    plt.close(fig)

    # --- 4. Partial series vs level: where a bridge can and cannot hear the grain ---
    print("\n4. Grain against an isotropic plate MATCHED ON THE FUNDAMENTAL:")
    probe = [(m, n) for m in range(1, 4) for n in range(1, 4)]
    pg = build(32, 1.0, grain=(gx, gh, gy))
    fg = np.sort(mode_freqs(pg, probe))
    pi_ = build(32, 1.0)
    fi = np.sort(mode_freqs(pi_, probe))
    scale = fg[0] / fi[0]
    reshape = float(np.max(np.abs(fg / fg[0] - fi / fi[0]) / (fi / fi[0])))
    print(f"  partial series (pitch removed): moves {reshape * 100:.1f} %")

    geometries = [
        ((0.40, 0.55), (0.31, 0.27)), ((0.50, 0.50), (0.50, 0.50)),
        ((0.25, 0.70), (0.60, 0.35)), ((0.60, 0.30), (0.20, 0.80)),
        ((0.45, 0.45), (0.70, 0.70)),
    ]

    def ring(p, pl, pk):
        p.set_state(pluck(p, (pl[0] * p.Lx, pl[1] * p.Ly), 0.25 * min(p.Lx, p.Ly)))
        idx = p.pickup_index_at(pk[0] * p.Lx, pk[1] * p.Ly)
        out = simulate(p, num_steps=int(0.15 * p.fs), pickup_index=idx).output
        return float(np.sqrt(np.mean(np.square(out))))

    ratios = []
    for pl, pk in geometries:
        a = build(32, 1.0, grain=(gx, gh, gy))
        b = Plate(Lx=LX, Ly=LY, kappa=KAPPA * scale, rho=RHO, fs=a.fs, N=32)
        ratios.append(ring(a, pl, pk) / ring(b, pl, pk))
    ratios = np.array(ratios)
    print(f"  single-node RMS ratio over 5 pluck/pickup geometries: "
          f"{np.array2string(ratios, precision=3)}")
    print(f"  min {ratios.min():.3f}, max {ratios.max():.3f} -- STRADDLES 1, so that spread is the")
    print("  geometry talking, not the grain. Judge a body by how loud its terminus rings and you")
    print("  cannot tell wood from metal; the grain is audible as TUNING, not as OUTPUT.")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(len(ratios)), ratios - 1.0, bottom=1.0, color="0.6")
    ax.axhline(1.0, c="k", lw=1)
    ax.axhline(1.0 + reshape, c="C3", ls="--", lw=1.5,
               label=f"partial-series reshaping ({reshape * 100:.0f}%)")
    ax.set_xticks(range(len(ratios)))
    ax.set_xticklabels([f"g{j + 1}" for j in range(len(ratios))])
    ax.set_ylabel("single-node RMS, grained / isotropic (pitch matched)")
    ax.set_title("The grain is in the partial series, not in the level")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "orthotropic_level_vs_partials.png"), dpi=110)
    plt.close(fig)

    # --- 5. Mode shapes, isotropic against spruce ---
    show = [(1, 1), (2, 1), (1, 2), (2, 2)]
    fig, axes = plt.subplots(2, 4, figsize=(13, 6))
    for row, (tag, g) in enumerate([("isotropic", (1.0, 1.0, 1.0)), ("spruce", (gx, gh, gy))]):
        pp = build(64, 1.0, grain=g)
        fr = mode_freqs(pp, show)
        for col, (m, n) in enumerate(show):
            field = embed(sine(pp, m, n), pp.index_map)
            plots.plot_membrane_field(
                axes[row, col], pp.X, pp.Y, field, mask=pp.mask,
                title=f"{tag}  {m},{n}   {fr[col]:.1f} Hz",
            )
    fig.suptitle("Same mode shapes, very different frequencies — the grain runs along x")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "orthotropic_modes.png"), dpi=110)
    plt.close(fig)
    print("\nFigures: out/orthotropic_{partials,cross_sweep,level_vs_partials,modes}.png")


if __name__ == "__main__":
    main()
