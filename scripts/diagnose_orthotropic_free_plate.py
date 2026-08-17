"""Orthotropic **free** plate (model #5of) diagnostics — a wooden plate before it is glued in.

Model #5o gave the pinned-rim plate a grain and needed three bending numbers. A free rim needs
**four**, because it can tell the coupling rigidity from the torsional one — and this script prints
the numbers that claim rests on, so the model can be judged from the console alone:

1. **The premise, both halves in one place.** Hold the cross term `H` fixed and move only the split
   between its two halves: the *supported* operator is bit-identical (a printed `0.0`), while the
   *free* plate's fundamental spans a factor.
2. **One detector per constant**, since there is no closed-form spectrum: the saddle's Rayleigh
   quotient (torsion, blind to the rest), the exact reduction to the shipped 1-D free beam
   (along/across stiffness, and only at zero coupling), and the `(x^2, y^2)` probe (coupling, the
   only detector that sees it).
3. **The mode race the grain can win.** The fundamental is the twist mode for isotropic material and
   is nearly tied with cross-grain bending for spruce; below `g_y/g_xy ~ 1.03` the bender wins.
   The supported plate never reorders anything, so this is a free-edge phenomenon.
4. **Anticlastic curvature**, which is why a plate is not a wide beam: with a coupling rigidity a
   beam-like field keeps *exactly* the beam's energy while ceasing to be an eigenvector.
5. Chladni patterns, isotropic against spruce, so the grain is visible.

Run:  python scripts/diagnose_orthotropic_free_plate.py
Figures land in ./out/.
"""

from __future__ import annotations

import os
import sys

# Make `physsynth` importable when run directly (no editable install): repo root on sys.path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.linalg import eigh  # noqa: E402

from physsynth.analysis import modal  # noqa: E402
from physsynth.core.engine import simulate  # noqa: E402
from physsynth.core.exciter import raised_cosine_2d  # noqa: E402
from physsynth.core.operators import free_beam_stiffness  # noqa: E402
from physsynth.core.operators2d import free_plate_stiffness  # noqa: E402
from physsynth.core.plate import Plate, grain_ratios_from_material  # noqa: E402
from physsynth.viz import plots  # noqa: E402

KAPPA, RHO, SIDE = 20.0, 0.005, 1.0  # square, so a shape asymmetry cannot fake a material one
OUT = "out"

SPRUCE = dict(E_x=11.0e9, E_y=0.8e9, nu_xy=0.37, G_xy=0.7e9, thickness=3.0e-3, rho=420.0)


def build(N, mu, *, grain=None, sigma=0.0, boundary="free", side=SIDE):
    """A plate at plate-Courant `mu`; `grain` is `(g_x, g_y, g_1, g_xy)` or None for isotropic."""
    h = side / N
    fs = KAPPA / (mu * h * h)
    kw = dict(Lx=side, Ly=side, kappa=KAPPA, rho=RHO, fs=fs, N=N, sigma=sigma, boundary=boundary)
    if grain is None:
        return Plate(**kw, nu=0.3)
    g_x, g_y, g_1, g_xy = grain
    return Plate(**kw, grain_x=g_x, grain_y=g_y, grain_coupling=g_1, grain_torsion=g_xy)


def elastic_spectrum(p, n=6):
    """The `n` lowest elastic `(lambda, phi)` of `K phi = mu W phi`, lambda = omega a^2 / kappa."""
    vals, vecs = eigh(p.K.toarray(), np.diag(p.W.diagonal()))
    lam = p.Lx * p.Lx * np.sqrt(np.clip(vals[3:3 + n], 0.0, None))
    return lam, vecs[:, 3:3 + n]


def centred(p):
    return p.X - 0.5 * p.Lx, p.Y - 0.5 * p.Ly


def classify(p, phi):
    """Label a mode `twist` (correlates with the saddle) or `x`/`y`-bend (flat along one axis)."""
    Xc, Yc = centred(p)
    f = phi.reshape(p.X.shape)
    corr = abs(np.corrcoef((Xc * Yc).ravel(), phi)[0, 1])
    if corr > 0.9:
        return "twist "
    y_const = np.linalg.norm(np.repeat(f.mean(axis=0, keepdims=True), f.shape[0], 0))
    x_const = np.linalg.norm(np.repeat(f.mean(axis=1, keepdims=True), f.shape[1], 1))
    if y_const / np.linalg.norm(f) > 0.9:
        return "x-bend"
    if x_const / np.linalg.norm(f) > 0.9:
        return "y-bend"
    return "mixed "


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    spec = grain_ratios_from_material(**SPRUCE)
    g_spruce = (spec.grain_x, spec.grain_y, spec.grain_coupling, spec.grain_torsion)

    print("=" * 78)
    print("Sitka-spruce-ish soundboard, as the four bending ratios the free branch needs")
    print("=" * 78)
    print(f"  g_x  (D_x/D_ref)  = {spec.grain_x:.6f}")
    print(f"  g_y  (D_y/D_ref)  = {spec.grain_y:.6f}      -> {1 / spec.grain_y:.1f}x stiffer along")
    print(f"  g_1  (D_1/D_ref)  = {spec.grain_coupling:.6f}   coupling  (implied nu_yx)")
    print(f"  g_xy (D_xy/D_ref) = {spec.grain_torsion:.6f}   torsional")
    print(f"  H = g_1 + 2 g_xy  = {spec.grain_cross:.6f}   <- ALL the supported plate can see")
    print(f"  torsional share of H = {2 * spec.grain_torsion / spec.grain_cross:.1%}"
          "   (the split is lopsided, which is why the free edge matters)")

    # --- 1. The premise: same H, two boundaries -------------------------------------------------
    print("\n" + "=" * 78)
    print("1. Fixed H = 0.153, only the SPLIT varies. Two boundaries, two answers.")
    print("=" * 78)
    g_h, g_y = 0.153, 0.073
    base_B = None
    print(f"  {'g_1':>8} {'g_xy':>8} | {'supported |B - B_0|':>21} | {'free lambda_1':>14} "
          f"| fundamental")
    lams = []
    for g_1 in (-0.10, 0.0, 0.05, 0.10, 0.15):
        g_xy = 0.5 * (g_h - g_1)
        sup = build(24, 1.0, grain=(1.0, g_y, g_1, g_xy), boundary="supported")
        data = sup.B.tocsr()
        data.sort_indices()
        if base_B is None:
            base_B, dB = data, 0.0
        else:
            dB = float(abs(base_B.data - data.data).max())
        fr = build(24, 1.0, grain=(1.0, g_y, g_1, g_xy))
        lam, vecs = elastic_spectrum(fr, 2)
        lams.append(lam[0])
        print(f"  {g_1:8.3f} {g_xy:8.4f} | {dB:21.1f} | {lam[0]:14.3f} "
              f"| {classify(fr, vecs[:, 0])}")
    print("\n  supported: bit-identical at every split (a printed 0.0, not a small number)")
    print(f"  free:      lambda_1 spans {min(lams):.3f} .. {max(lams):.3f} = "
          f"{max(lams) / min(lams):.2f}x")

    # --- 2. One detector per constant ------------------------------------------------------------
    print("\n" + "=" * 78)
    print("2. The four constants, one probe each (there is no closed-form spectrum)")
    print("=" * 78)
    N, h = 16, 0.05
    Xc = (np.arange(N + 1) - 0.5 * N) * h
    X, Y = np.meshgrid(Xc, Xc)
    xy = (X * Y).ravel()
    print("  a) the saddle's Rayleigh quotient must be BLIND to g_x, g_y, g_1:")
    for g_x, g_y_, g_1 in [(1.0, 1.0, 0.3), (1.0, 0.073, 0.02), (3.1, 0.5, -0.4)]:
        K, W, _ = free_plate_stiffness(
            N, N, h, 0.3, grain_x=g_x, grain_y=g_y_, grain_coupling=g_1, grain_torsion=0.37
        )
        num = float(xy @ (K @ xy))
        R = num / float(xy @ (W.diagonal() * xy))
        print(f"     g=({g_x:5.2f},{g_y_:6.3f},{g_1:+5.2f}): numerator {num:.12f} "
              f"(4 g_xy ab = {4 * 0.37 * (N * h) ** 2:.12f})   R = {R:.9f}")
    print("  b) the beam reduction is EXACT at zero coupling, and breaks with it (anticlastic):")
    S, M = free_beam_stiffness(20, h)
    vals_b, vecs_b = eigh(S.toarray(), np.diag(M.diagonal()))
    w = np.kron(np.ones(15), vecs_b[:, 2])
    for g_1 in (0.0, 0.1, 0.3):
        K, W, _ = free_plate_stiffness(
            20, 14, h, 0.3, grain_x=2.3, grain_y=0.6, grain_coupling=g_1, grain_torsion=0.25
        )
        mu_pred = 2.3 * vals_b[2]
        Ww = W.diagonal() * w
        resid = np.linalg.norm(K @ w - mu_pred * Ww) / np.linalg.norm(mu_pred * Ww)
        rayleigh = float(w @ (K @ w)) / float(w @ Ww)
        print(f"     g_1 = {g_1:.1f}: eigenvector residual {resid:.3e}   "
              f"energy/beam = {rayleigh / mu_pred:.12f}  <- energy UNCHANGED, shape not")
    print("  c) the (x^2, y^2) probe is the only detector that sees the coupling rigidity:")
    for g_1 in (0.05, 0.153):
        K, _, _ = free_plate_stiffness(
            N, N, h, 0.3, grain_x=0.0, grain_y=0.0, grain_coupling=g_1, grain_torsion=0.0
        )
        val = float((X**2).ravel() @ (K @ (Y**2).ravel()))
        exact = modal.free_plate_coupling_form(g_1, h, N, N)
        cont = 4.0 * g_1 * (N * h) ** 2
        print(f"     g_1 = {g_1:.3f}: probe {val:.12f}  exact {exact:.12f}  "
              f"(continuum {cont:.6f}, gap {abs(exact - cont) / cont:.2%} = one boundary strip)")

    # --- 3. The mode race ------------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("3. The grain REORDERS the free plate's modes (the supported plate's never move)")
    print("=" * 78)
    print(f"  {'material':>16} | {'lambda_1':>9} {'lambda_2':>9} {'lambda_3':>9} | low modes")
    for tag, g in [
        ("isotropic", (1.0, 1.0, 0.3, 0.35)),
        ("spruce", g_spruce),
        ("softer across", (1.0, 0.04, 0.0, 0.063)),
    ]:
        p = build(32, 1.0, grain=g)
        lam, vecs = elastic_spectrum(p, 3)
        kinds = " ".join(classify(p, vecs[:, j]) for j in range(3))
        print(f"  {tag:>16} | {lam[0]:9.3f} {lam[1]:9.3f} {lam[2]:9.3f} | {kinds}")
    print(f"\n  twist bound 24*sqrt(g_xy):  isotropic {24 * np.sqrt(0.35):.3f}, "
          f"spruce {24 * np.sqrt(g_spruce[3]):.3f}")
    print(f"  free-beam landmark 22.373*sqrt(g_y): isotropic 22.373, "
          f"spruce {22.3733 * np.sqrt(g_spruce[1]):.3f}  <- nearly TIED with the twist")
    print(f"  spruce sits at g_y/g_xy = {g_spruce[1] / g_spruce[3]:.3f}; the crossing is at ~1.03, "
          f"so it is {100 * (g_spruce[1] / g_spruce[3] / 1.025 - 1):.0f}% onto the twist side")

    # --- 4. Energy is blind, and the ledger says so ----------------------------------------------
    print("\n" + "=" * 78)
    print("4. The ledger cannot referee any of this: three DIFFERENT plates, all exact")
    print("=" * 78)
    for tag, g in [
        ("spruce", g_spruce),
        ("split swapped (g_1 <-> g_xy)", (1.0, g_spruce[1], g_spruce[3], g_spruce[2])),
        ("all of H taken as torsion", (1.0, g_spruce[1], 0.0, 0.5 * spec.grain_cross)),
    ]:
        p = build(24, 1.0, grain=g)
        p.set_state(raised_cosine_2d(p.X, p.Y, (0.4 * SIDE, 0.55 * SIDE), 0.25 * SIDE, 1e-3))
        res = simulate(p, num_steps=int(0.2 * p.fs))
        lam, _ = elastic_spectrum(p, 1)
        print(f"  {tag:>30}: drift {res.energy_drift:.2e}   lambda_1 = {lam[0]:.3f}")
    print("  -> all conserve to machine precision; only the probes of section 2 separate them.")

    # --- 5. Chladni patterns ---------------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.5))
    for row, (tag, g) in enumerate([("isotropic", (1.0, 1.0, 0.3, 0.35)), ("spruce", g_spruce)]):
        p = build(48, 1.0, grain=g)
        lam, vecs = elastic_spectrum(p, 3)
        for col in range(3):
            field = vecs[:, col].reshape(p.X.shape)
            plots.plot_chladni(
                axes[row, col], p.X, p.Y, field / np.abs(field).max(),
                title=f"{tag}  mode {col + 1}  lambda={lam[col]:.2f}  {classify(p, vecs[:, col])}",
            )
    fig.suptitle(
        "Free-edge plate, isotropic vs spruce: the grain stretches the patterns AND reorders them"
    )
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "orthotropic_free_chladni.png"), dpi=110)
    plt.close(fig)

    # --- 6. The split sweep, plotted -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    g1s = np.linspace(-0.20, 0.145, 14)
    lam1, kinds = [], []
    for g_1 in g1s:
        p = build(24, 1.0, grain=(1.0, g_y, g_1, 0.5 * (g_h - g_1)))
        lam, vecs = elastic_spectrum(p, 1)
        lam1.append(lam[0])
        kinds.append(classify(p, vecs[:, 0]).strip())
    ax.plot(g1s, lam1, "o-", label=r"free plate, $\lambda_1$")
    ax.axhline(lam1[list(g1s).index(min(g1s, key=lambda v: abs(v)))], ls=":", c="gray")
    ax.plot(g1s, [lam1[0]] * len(g1s), alpha=0.0)  # keep the y-range honest
    ax.set_xlabel(r"coupling rigidity $g_1 = D_1/D_x$   (with $g_{xy} = (H - g_1)/2$, $H$ fixed)")
    ax.set_ylabel(r"$\lambda_1 = \omega_1 a^2/\kappa$")
    ax.set_title("Same H, same $D_x$, same $D_y$ — the supported plate cannot tell these apart")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "orthotropic_free_split_sweep.png"), dpi=110)
    plt.close(fig)
    print("\nFigures: out/orthotropic_free_{chladni,split_sweep}.png")


if __name__ == "__main__":
    main()
