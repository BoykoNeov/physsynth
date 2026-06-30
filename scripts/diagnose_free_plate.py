"""Free-edge (FFFF) Kirchhoff-plate (model #5b) diagnostic harness — the curved Chladni figures.

Runs the energy-first, W-weighted theta-scheme free plate and renders diagnostics-as-visuals
(HANDOFF §7), the free-edge counterpart of ``diagnose_plate.py``:

- energy-conservation drift **at a timestep the explicit plate could not run** (mu = kappa k / h²
  past the explicit bound 1/4) — the unconditional-stability win, here on the W-weighted scheme;
- the iconic **curved Chladni nodal-line figures**: the lowest free-plate eigenmodes (from ``eigsh``
  on the energy-first stiffness, so independent of the time-stepper), heatmap + ``u = 0`` contour.
  Unlike the SS plate's plain rectangular nodal grid, these are the genuinely *curved* sand figures;
- detected-vs-Leissa frequency parameters (the percent-level absolute anchor — no closed form), with
  the fundamental being the **saddle/twist**, not a drum bulge;
- passive energy decay under loss, and a struck-plate displacement animation.

Prints the real validation numbers so the model can be judged from the console alone.

Run:  python scripts/diagnose_free_plate.py
Figures land in ./out/.
"""

from __future__ import annotations

import os
import sys

# Make `physsynth` importable when run directly (no editable install): repo root on sys.path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.sparse.linalg import eigsh  # noqa: E402

from physsynth.analysis import modal, spectrum  # noqa: E402
from physsynth.core.engine import simulate  # noqa: E402
from physsynth.core.exciter import raised_cosine_2d  # noqa: E402
from physsynth.core.operators2d import embed  # noqa: E402
from physsynth.core.plate import Plate  # noqa: E402
from physsynth.viz import plots  # noqa: E402

# Canonical free square plate: kappa = sqrt(D/rho_s) = 20 m^2/s, rho_s = 0.005 kg/m^2, side a = 1 m,
# Poisson's ratio nu = 0.3 (the Leissa-table value).
KAPPA, RHO, A, NU = 20.0, 0.005, 1.0, 0.3
OUT = "out"


def build(N, mu, *, sigma=0.0):
    h = A / N
    fs = KAPPA / (mu * h * h)  # invert mu = kappa k / h^2
    return Plate(Lx=A, Ly=A, kappa=KAPPA, rho=RHO, fs=fs, N=N, sigma=sigma, boundary="free", nu=NU)


def low_eigenpairs(p, n_elastic):
    """The 3 rigid + ``n_elastic`` lowest elastic eigenpairs of ``K φ = μ W φ`` (ascending)."""
    mu1 = (13.0 / (A * A)) ** 2  # lambda_1 ~ 13.47 -> a safe negative shift (K is only PSD)
    vals, vecs = eigsh(p.K, k=n_elastic + 3, M=p.W, sigma=-1e-3 * mu1, which="LM")
    order = np.argsort(vals)
    return vals[order], vecs[:, order]


def main() -> None:
    os.makedirs(OUT, exist_ok=True)

    # --- 1. Energy conservation at a timestep the explicit plate could not run (W-weighted) -------
    print("Energy drift (lossless, ~1 s, free edges):")
    for mu in (2.0, 16.0):
        p = build(40, mu)
        p.set_state(raised_cosine_2d(p.X, p.Y, (0.4 * A, 0.55 * A), 0.25 * A, amplitude=1e-3))
        res = simulate(p, num_steps=int(1.0 * p.fs))
        flag = "  (explicit plate bound is mu <= 1/4)" if mu > 0.25 else ""
        print(f"  mu = {mu:5.1f}   drift = {res.energy_drift:.2e}   nlive={p.n_live}{flag}")

    # --- 2. The curved Chladni figures + the Leissa absolute anchor -------------------------------
    p = build(80, 1.0)
    n_show = 6
    vals, vecs = low_eigenpairs(p, n_show)
    elastic = vals[3:]  # drop the 3 rigid-body modes {1, x, y}
    lam_meas = A * A * np.sqrt(np.clip(elastic, 0.0, None))  # Leissa lambda = a^2 sqrt(mu)
    f_meas = p.kappa * np.sqrt(np.clip(elastic, 0.0, None)) / (2.0 * np.pi)
    lam_ref = modal.free_plate_ffff_square_lambdas()

    fig, axes = plt.subplots(2, 3, figsize=(11, 7.5))
    for j, ax in enumerate(axes.flat):
        field = embed(vecs[:, 3 + j], p.index_map)
        plots.plot_chladni(
            ax, p.X, p.Y, field, title=f"mode {j + 1}:  λ = {lam_meas[j]:.2f}   {f_meas[j]:.1f} Hz"
        )
    fig.suptitle(
        "Free-edge plate (FFFF) — curved Chladni nodal lines (mode 1 = the saddle/twist)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "free_plate_chladni.png"), dpi=120)
    plt.close(fig)

    print("\nLeissa FFFF-square anchor (N=80, nu=0.3) — detected lambda vs tabulated:")
    for j in range(len(lam_ref)):
        err = 100.0 * (lam_meas[j] - lam_ref[j]) / lam_ref[j]
        print(f"  mode {j + 1}:  lambda_meas = {lam_meas[j]:7.3f}   ref = {lam_ref[j]:7.3f}"
              f"   err = {err:+5.2f}%")

    fig, ax = plt.subplots(figsize=(7, 4))
    idx = np.arange(len(lam_ref))
    ax.bar(idx, 100.0 * (lam_meas[: len(lam_ref)] - lam_ref) / lam_ref, color="C0", width=0.6)
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xlabel("elastic mode")
    ax.set_ylabel("λ error vs Leissa (%)")
    worst = float(np.max(np.abs((lam_meas[: len(lam_ref)] - lam_ref) / lam_ref))) * 100.0
    ax.set_title(f"Free plate vs Leissa FFFF-square (worst |err| = {worst:.2f}%) — percent tier")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "free_plate_leissa.png"), dpi=120)
    plt.close(fig)

    # --- 3. Passivity under loss ------------------------------------------------------------------
    pd = build(40, 2.0, sigma=8.0)
    pd.set_state(raised_cosine_2d(pd.X, pd.Y, (0.3 * A, 0.4 * A), 0.4 * A, amplitude=1e-3))
    resd = simulate(pd, num_steps=int(0.8 * pd.fs))
    fig, ax = plt.subplots(figsize=(7, 4))
    plots.plot_energy(ax, resd.time, resd.energy, drift=None)
    ax.set_yscale("log")
    mono = bool(np.all(np.diff(resd.energy) <= 0))
    ax.set_title(f"Free plate passive decay (sigma=8): monotone = {mono}")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "free_plate_passivity.png"), dpi=110)
    plt.close(fig)
    print(f"\nPassivity (sigma=8): energy monotone non-increasing = "
          f"{bool(np.all(np.diff(resd.energy) <= 0))}")

    # --- 4. Struck-plate animation ----------------------------------------------------------------
    pa = build(64, 1.0)
    pa.set_state(raised_cosine_2d(pa.X, pa.Y, (0.32 * A, 0.4 * A), 0.35 * A, amplitude=1e-3))
    n_anim = int(0.04 * pa.fs)
    resa = simulate(pa, num_steps=n_anim, snapshot_stride=max(1, n_anim // 60))
    snaps = [(s, embed(v[pa.mask] if v.shape == pa.mask.shape else v, pa.index_map))
             for s, v in resa.snapshots]
    ok = plots.save_membrane_animation(
        os.path.join(OUT, "free_plate_strike.gif"), pa.X, pa.Y, snaps, pa.fs, mask=pa.mask
    )
    print(f"\nAnimation written: {ok}  ->  out/free_plate_strike.gif")

    # --- 5. FFT sanity: the time-stepper rings at the discrete fundamental ------------------------
    pf = build(40, 1.0)
    vals_f, _ = low_eigenpairs(pf, 1)
    mu_fund = vals_f[3]  # first elastic generalized eigenvalue (= omega^2 / kappa^2)
    f_disc = modal.discrete_beam_eigenfrequency(mu_fund, pf.kappa, pf.k, pf.theta)
    pf.set_state(raised_cosine_2d(pf.X, pf.Y, (0.3 * A, 0.62 * A), 0.3 * A, amplitude=1e-3))
    pk = pf.pickup_index_at(0.18 * A, 0.22 * A)
    resf = simulate(pf, num_steps=int(0.5 * pf.fs), pickup_index=pk)
    found = spectrum.measure_partials_near(
        resf.output, resf.fs, np.array([f_disc]), search_hz=20.0
    )[0]
    print(f"FFT fundamental: found {found:.2f} Hz vs discrete {float(f_disc):.2f} Hz "
          f"({modal.cents(found, f_disc):+.3f} cents)")


if __name__ == "__main__":
    main()
