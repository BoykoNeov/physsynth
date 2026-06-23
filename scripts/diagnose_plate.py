"""Simply-supported Kirchhoff-plate (model #5) diagnostic harness.

Runs the implicit theta-scheme plate and renders diagnostics-as-visuals (HANDOFF §7), reusing the
generic 2D field/animation plotters built for the membrane:

- energy-conservation drift **at a timestep the explicit plate could not run** (mu = kappa k / h²
  well past the explicit bound 1/4) — the unconditional-stability win;
- low simply-supported mode shapes as heatmaps. NOTE these nodal lines form a plain rectangular
  GRID — the iconic *curved* Chladni figures need free edges (a deferred follow-on), not the
  validated SS plate shown here;
- detected-vs-analytic partials in cents — the **tight** SS tier (< ~1 cent at a fine grid), in
  contrast to the membrane disk's looser ~O(h) staircase tier;
- passive energy decay under loss, and a struck-plate displacement animation.

Prints the real validation numbers so the model can be judged from the console alone.

Run:  python scripts/diagnose_plate.py
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

# Canonical plate: kappa = sqrt(D/rho_s) = 20 m^2/s, rho_s = 0.005 kg/m^2. A slightly NON-square
# rectangle (Lx != Ly) lifts the (m,n)<->(n,m) degeneracy so modes carry clean (m,n) labels.
KAPPA, RHO, LX, LY = 20.0, 0.005, 1.0, 0.8
OUT = "out"


def build(N, mu, *, sigma=0.0):
    h = LX / N
    fs = KAPPA / (mu * h * h)  # invert mu = kappa k / h^2
    return Plate(Lx=LX, Ly=LY, kappa=KAPPA, rho=RHO, fs=fs, N=N, sigma=sigma)


def pluck(p, center, width, amp=1e-3):
    f = raised_cosine_2d(p.X, p.Y, center, width, amplitude=amp)
    f[~p.mask] = 0.0
    return f


def sorted_modes(Lx, Ly, n_modes, m_max=6, n_max=6):
    """Lowest ``n_modes`` continuum SS modes as ``(m, n, f_cont)``, sorted by frequency.

    Uses the plate's *actual* side lengths — ``Ly`` is snapped to an integer number of cells at
    construction, so the oracle must use the snapped value, not the requested one (a 0.26% mismatch
    here would otherwise read as a spurious ~10-cent error and hide the true sub-0.1-cent SS tier).
    """
    modes = [(m, n) for m in range(1, m_max + 1) for n in range(1, n_max + 1)]
    fc = modal.rectangular_plate_freqs(KAPPA, Lx, Ly, modes)
    order = np.argsort(fc)
    return [(modes[i][0], modes[i][1], float(fc[i])) for i in order[:n_modes]]


def main() -> None:
    os.makedirs(OUT, exist_ok=True)

    # --- 1. Energy conservation at a timestep the explicit plate could not run ---
    print("Energy drift (lossless, ~1 s):")
    for mu in (2.0, 16.0):
        p = build(48, mu)
        p.set_state(pluck(p, (0.4 * LX, 0.55 * LY), 0.25 * LX))
        res = simulate(p, num_steps=int(1.0 * p.fs))
        flag = "  (explicit plate bound is mu <= 1/4)" if mu > 0.25 else ""
        print(f"  mu = {mu:5.1f}   drift = {res.energy_drift:.2e}   nlive={p.n_live}{flag}")

    # --- 2. Low SS mode shapes (a nodal GRID, not Chladni curves) + tight detected-vs-oracle ---
    p = build(96, 1.0)
    K = 6
    Lam, V = eigsh(-p.L, k=K, sigma=0.0, which="LM")
    order = np.argsort(Lam)
    Lam, V = Lam[order], V[:, order]
    f_disc = modal.discrete_plate_eigenfrequency(Lam, p.kappa, p.k, p.theta)
    oracle = sorted_modes(p.Lx, p.Ly, K)
    labels = [f"{m},{n}" for m, n, _ in oracle]
    f_cont = np.array([f for _, _, f in oracle])

    fig, axes = plt.subplots(2, 3, figsize=(11, 7))
    for j, ax in enumerate(axes.flat):
        field = embed(V[:, j], p.index_map)
        plots.plot_membrane_field(
            ax, p.X, p.Y, field, mask=p.mask, title=f"mode {labels[j]}  {f_disc[j]:.1f} Hz"
        )
    fig.suptitle("Simply-supported plate low modes — nodal GRID (free-edge Chladni curves deferred)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "plate_modes.png"), dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    plots.plot_membrane_partials(ax, f_disc, f_cont, labels=labels)
    worst = float(np.nanmax(np.abs(modal.cents(f_disc, f_cont))))
    ax.set_title(f"Plate partials vs SS oracle — TIGHT tier (worst |err| = {worst:.3f} cents)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "plate_partials.png"), dpi=110)
    plt.close(fig)

    print("\nDetected vs continuum SS oracle (N=96, the tight tier):")
    for (m, n, _), fd, fc in zip(oracle, f_disc, f_cont, strict=True):
        print(f"  mode {m},{n}  disc={fd:8.3f}  oracle={fc:8.3f}  err={modal.cents(fd, fc):+6.3f} c")

    # --- 3. Passivity under loss ---
    pd = build(48, 2.0, sigma=8.0)
    pd.set_state(pluck(pd, (0.3 * LX, 0.4 * LY), 0.4 * LX))
    resd = simulate(pd, num_steps=int(0.8 * pd.fs))
    fig, ax = plt.subplots(figsize=(7, 4))
    plots.plot_energy(ax, resd.time, resd.energy, drift=None)
    ax.set_yscale("log")
    ax.set_title(f"Passive decay (sigma=8): monotone = {np.all(np.diff(resd.energy) <= 0)}")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "plate_passivity.png"), dpi=110)
    plt.close(fig)
    print(f"\nPassivity (sigma=8): energy monotone non-increasing = "
          f"{bool(np.all(np.diff(resd.energy) <= 0))}")

    # --- 4. Struck-plate animation ---
    pa = build(64, 1.0)
    pa.set_state(pluck(pa, (0.35 * LX, 0.42 * LY), 0.4 * LX))
    n_anim = int(0.04 * pa.fs)
    resa = simulate(pa, num_steps=n_anim, snapshot_stride=max(1, n_anim // 60))
    snaps = [(s, embed(v[pa.mask] if v.shape == pa.mask.shape else v, pa.index_map))
             for s, v in resa.snapshots]
    ok = plots.save_membrane_animation(
        os.path.join(OUT, "plate_strike.gif"), pa.X, pa.Y, snaps, pa.fs, mask=pa.mask
    )
    print(f"\nAnimation written: {ok}  ->  out/plate_strike.gif")

    # --- 5. FFT sanity: the time-stepper rings at the discrete fundamental ---
    pf = build(48, 1.0)
    f_fund = modal.discrete_plate_eigenfrequency(
        eigsh(-pf.L, k=1, sigma=0.0, which="LM", return_eigenvectors=False),
        pf.kappa, pf.k, pf.theta,
    )
    pf.set_state(pluck(pf, (0.35 * LX, 0.42 * LY), 0.45 * LX))
    pk = pf.pickup_index_at(0.3 * LX, 0.28 * LY)
    resf = simulate(pf, num_steps=int(0.5 * pf.fs), pickup_index=pk)
    found = spectrum.measure_partials_near(resf.output, resf.fs, f_fund, search_hz=20.0)[0]
    print(f"FFT fundamental: found {found:.2f} Hz vs discrete {f_fund[0]:.2f} Hz "
          f"({modal.cents(found, f_fund[0]):+.3f} cents)")


if __name__ == "__main__":
    main()
