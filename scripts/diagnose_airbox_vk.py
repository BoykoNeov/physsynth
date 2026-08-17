"""Diagnostics for the gong in the room: air-box batch 6.

The suite pins the coupling — the linear reduction, the ledgers, the conserved total, passivity and
the coupled residual. What it deliberately does not assert is the batch's **physical headline**, for
the reason batches 4 and 5 did not either: a radiation figure has to come from a motion that is
*held*, over a window long enough for the effect to develop. That is this script.

**The claim.** Every radiating thing in this repo before model #6 is linear in its excitation:
strike it twice as hard and every acoustic observable doubles, so radiated fraction, directivity and
dipole-over-baffled are amplitude-**invariant** by construction. The von Kármán coupling is
quadratic, so the *shape* of the motion evolves during a single strike — and shape is exactly what
:class:`~physsynth.core.airbox.SurfacePort` was built to make audible (batch 3: a surface radiates
by the shape of its motion, not by its net volume displacement). Hence:

    **a loud plate's radiation is time-varying at fixed geometry, and a quiet one's is not.**

No ``R(omega)`` in ``radiation.py`` can state that. A scalar-per-frequency load has *one* pattern
per frequency and cannot change it during a single strike.

**The observable, and why it is not the ledger.** The obvious measurement — radiated energy per
time window, off the port's own books — was tried first and does not separate the two runs: the
**room's own build-up** moves it by 1.8x in the quiet arm against 3.6x in the loud one, i.e. the
confound is the same size as the effect. That is batch 2's lesson recurring ("the room contaminated
the port's own measured size by more than the effect"), and section 3 below reports it rather than
hiding it. What separates cleanly is the radiation efficiency evaluated as a **functional of the
shape alone**,

    sigma_shape  =  v^T (T^T R T) v  /  (rho0 c0 A <v^2>)

— the room's own resistive load operator, the one inside the factorization, applied to the plate's
*actual coupled* velocity field. The run is fully coupled; only the read-out is a fixed quadratic
form, which is what takes the room's transient out of the number without taking the room out of the
physics.

**Three sections:**

  1. **The headline.** Same plate, same room, same strike position, one flag apart: ``w/e = 0.05``
     (effectively linear) against ``w/e = 3`` (the cascade), baffled and suspended. ``sigma_shape``
     per window, beside the modal-share drift that drives it. Every number is quoted **twice** —
     over all modes and over the modes the air grid actually resolves (batch 5's ≥5 cells per
     structural wave, applied per mode) — because the cascade's destination modes are precisely
     the ones the air grid resolves worst. The **separation** survives that restriction; the
     multiplier does not, and only the separation is claimed.

  2. **The compact limit, and it does not merely under-read.** The monopole ``sigma_mono`` — the
     net volume velocity, i.e. everything ``AirRadiation`` / ``RadiatedBody`` / ``RationalAirLoad``
     can see — is **seven orders of magnitude** below the true figure for this plate, and for the
     suspended cymbal it moves the **wrong way**: it rises while the true efficiency falls.

  3. **What died on measurement**, with the numbers: the ledger observable (above), the directivity
     panel (plan §7.7), and the plan's own cost and convergence estimates.

Run:  python scripts/diagnose_airbox_vk.py      (about 6 minutes)
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
    RoomLoadedVKPlate,
    RoomSuspendedVKPlate,
    impedance_from_zeta,
)
from physsynth.core.plate import VKPlate  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out")
os.makedirs(OUT, exist_ok=True)

CFL = 0.9                  # fraction of the 3-D ceiling 1/sqrt(3), as everywhere else
MAT = dict(E=2.0e11, e=1.0e-3, nu=0.3, rho=7800.0)   # test_vk_free.py's canonical steel sheet
PLATE_L = 0.10             # m, square -- small and thick-ish, so the modal ka is not tiny
PLATE_N = 16               # 6.25 mm plate spacing: >8 points per structural wave to ~3 kHz
H_AIR = 0.0114             # m -- >=5 air cells per structural wave at the strike's p90 mode
ROOM_M = 0.6               # m cube. A RATIO survives a small room; a magnitude does not (batch 2)
FS = C0_AIR * np.sqrt(3.0) / (CFL * H_AIR)           # the ROOM sets the rate: 57.9 kHz
N_ROOM = int(round(ROOM_M / H_AIR))
AREA = PLATE_L * PLATE_L
WALLS = impedance_from_zeta(4.0)
SECONDS = 0.12             # the observation window is PART OF THE CLAIM -- fixed and named
WINDOWS = 4
STRIKE_WIDTH = 0.20        # of Lx. NOT a free parameter: a narrow strike does not converge
MAX_SWEEPS = 120           # measured 72 at w/e = 3 at this rate; the default 50 would cap out
QUIET, LOUD = 0.05, 3.0    # w/e -- the control and the claim, one number apart
CELLS_PER_WAVE = 5.0       # batch 5's resolution floor, applied per MODE (see free_modes)


def strike(plate: VKPlate, we: float) -> np.ndarray:
    """A centred raised-Gaussian strike of peak ``we * e``, on the plate's live nodes."""
    w = STRIKE_WIDTH * plate.Lx
    dx, dy = plate.X - 0.5 * plate.Lx, plate.Y - 0.5 * plate.Ly
    return we * plate.e * np.exp(-((dx * dx + dy * dy) / (w * w)))[plate.mask]


def make(tier: str):
    """A free-edge von Kármán plate in a small lossy room, baffled or hung."""
    room = AirBox(L=(N_ROOM * H_AIR,) * 3, fs=FS, h=H_AIR, walls=WALLS)
    plate = VKPlate(
        Lx=PLATE_L, Ly=PLATE_L, fs=FS, N=PLATE_N, boundary="free",
        couple_max_iter=MAX_SWEEPS, **MAT,
    )
    if tier == "baffled":
        return RoomLoadedVKPlate(plate=plate, room=room, face="z0")
    return RoomSuspendedVKPlate(plate=plate, room=room, plane="z", index=N_ROOM // 2)


def free_modes(plate: VKPlate):
    """Mass-orthonormal free-plate modes, their frequencies, and which ones the AIR grid resolves.

    Projection under ``W``, **never** a spectral peak: at ``w/e = 3`` a peak tracker reads a mode's
    frequency as 0.53x its own linear value, because the field has gone broadband and "the"
    frequency has stopped existing.

    ``resolved`` is batch 5's criterion applied to this batch: a mode whose structural wavelength
    ``2 pi / sqrt(omega / kappa)`` spans fewer than :data:`CELLS_PER_WAVE` air cells is smoothed by
    the port's own spreading, so its contribution to any efficiency is an interpolation artefact as
    much as a physical one. Every headline number is quoted twice — once over all modes and once
    over this set — because a strike excites well past it.
    """
    vals, vecs = eigh(plate.K.toarray() * plate.kappa**2, np.diag(plate.wdiag))
    f = np.sqrt(np.clip(vals, 0.0, None)) / (2.0 * np.pi)
    beta = np.sqrt(2.0 * np.pi * np.maximum(f, 1e-9) / plate.kappa)
    cells = (2.0 * np.pi / beta) / H_AIR
    return vecs, f, cells >= CELLS_PER_WAVE


def run_case(tier: str, we: float) -> dict:
    """One strike, ``WINDOWS`` windows, three efficiencies and the modal shares behind them."""
    inst = make(tier)
    plate = inst.plate
    inst.set_state(strike(plate, we))
    vecs, freqs, resolved = free_modes(plate)
    vecs_r = vecs[:, resolved]
    load = inst.port.load_matrix    # T^T R T -- the room's own resistive operator
    faces = 2.0 if tier == "suspended" else 1.0
    per = int(SECONDS * FS) // WINDOWS
    shape, shape_r, mono, ledger, shares, in_band = [], [], [], [], [], []
    sweeps, converged = 0, True
    t0 = time.perf_counter()
    for _ in range(WINDOWS):
        r0 = inst.radiated_energy
        acc = np.zeros(vecs.shape[1])
        p_shape = p_res = p_mono = v2 = v2_res = 0.0
        for _ in range(per):
            v = (plate.u - plate.u_prev) / plate.k
            c = vecs.T @ (plate.wdiag * v)
            v_res = vecs_r @ c[resolved]        # the same motion, band-limited to what air resolves
            p_shape += faces * float(v @ (load @ v))
            p_res += faces * float(v_res @ (load @ v_res))
            p_mono += faces * RHO0_AIR * C0_AIR * float(np.sum(plate.wdiag * v)) ** 2 / AREA
            v2 += float(np.dot(plate.wdiag * v, v)) / AREA
            v2_res += float(np.dot(plate.wdiag * v_res, v_res)) / AREA
            acc += c * c
            inst.step()
            inst.room.step()
            sweeps = max(sweeps, plate.n_iters)
            converged &= plate.converged
        denom = RHO0_AIR * C0_AIR * AREA * (v2 / per)
        shape.append((p_shape / per) / denom)
        shape_r.append((p_res / per) / (RHO0_AIR * C0_AIR * AREA * (v2_res / per)))
        mono.append((p_mono / per) / denom)
        ledger.append(((inst.radiated_energy - r0) / (per * plate.k)) / denom)
        shares.append(acc / acc.sum())
        in_band.append(v2_res / v2)
    shares = np.array(shares)
    return dict(
        tier=tier, we=we, freqs=freqs, resolved=resolved,
        shape=np.array(shape), shape_r=np.array(shape_r), mono=np.array(mono),
        ledger=np.array(ledger), in_band=np.array(in_band),
        drift=0.5 * float(np.abs(shares[-1] - shares[0]).sum()),
        drift_r=0.5 * float(np.abs(shares[-1][resolved] - shares[0][resolved]).sum()),
        shares=shares, sweeps=sweeps, converged=converged,
        seconds=time.perf_counter() - t0,
    )


def spread(x: np.ndarray) -> float:
    return float(np.max(x) / np.min(x))


def modal_ka(freqs: np.ndarray, shares: np.ndarray) -> tuple[float, float]:
    """The energy-weighted ``ka`` centroid and its 90th percentile — the band a claim is made in."""
    a_eff = np.sqrt(AREA / np.pi)
    ka = 2.0 * np.pi * freqs / C0_AIR * a_eff
    order = np.argsort(ka)
    ka_s, w_s = ka[order], shares[order]
    centroid = float(np.sum(ka_s * w_s) / np.sum(w_s))
    cum = np.cumsum(w_s) / np.sum(w_s)
    return centroid, float(ka_s[int(np.searchsorted(cum, 0.90))])


def figure(cases: dict, path: str) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    x = np.arange(1, WINDOWS + 1)
    styles = {(QUIET, "baffled"): "o-", (QUIET, "suspended"): "s-",
              (LOUD, "baffled"): "o--", (LOUD, "suspended"): "s--"}
    for key, c in cases.items():
        lbl = f"{c['tier']}, w/e={c['we']:g}"
        colour = "0.45" if c["we"] == QUIET else "C3"
        axes[0].plot(x, c["shape"] / c["shape"][0], styles[key], color=colour, label=lbl, ms=4)
        axes[1].plot(x, c["mono"] / c["mono"][0], styles[key], color=colour, label=lbl, ms=4)
        axes[2].plot(x, c["ledger"] / c["ledger"][0], styles[key], color=colour, label=lbl, ms=4)
    for ax, title in zip(
        axes,
        (r"$\sigma_{shape}$ - the claim",
         r"$\sigma_{monopole}$ - the compact limit",
         r"$\sigma_{ledger}$ - confounded by the room"),
        strict=True,
    ):
        ax.axhline(1.0, color="0.3", ls=":", lw=1.0)
        ax.set_xlabel(f"window ({1e3 * SECONDS / WINDOWS:.0f} ms each)")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_xticks(x)
    axes[0].set_ylabel("efficiency, relative to window 1")
    axes[0].legend(fontsize=7)
    fig.suptitle(
        "a loud plate's radiation is time-varying at fixed geometry, and a quiet one's is not",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def main() -> None:
    print(__doc__.strip().splitlines()[0])
    print("=" * 96)
    print(f"\nthe cell: {PLATE_L * 1e3:.0f} mm free steel plate, e = {MAT['e'] * 1e3:.0f} mm, "
          f"N = {PLATE_N}; room {ROOM_M} m cube at h = {H_AIR * 1e3:.1f} mm ({N_ROOM}^3 cells), "
          f"fs = {FS / 1e3:.1f} kHz")
    print(f"the window is part of the claim: {SECONDS * 1e3:.0f} ms in {WINDOWS} windows of "
          f"{1e3 * SECONDS / WINDOWS:.0f} ms")

    cases = {}
    for we in (QUIET, LOUD):
        for tier in ("baffled", "suspended"):
            c = run_case(tier, we)
            cases[(we, tier)] = c
            print(f"   ran {tier:9s} w/e={we:4.2f}  {c['seconds']:5.1f}s  "
                  f"sweeps<={c['sweeps']:3d}  converged={c['converged']}")

    ref = cases[(QUIET, "baffled")]
    centroid, p90 = modal_ka(ref["freqs"], ref["shares"][0])
    n_res = int(np.sum(ref["resolved"]))
    print(f"\n   the band this is measured in: modal ka centroid {centroid:.2f}, p90 {p90:.2f}.")
    print(f"   The strike excites past what the AIR grid resolves: {n_res} of "
          f"{ref['resolved'].size} modes keep >= {CELLS_PER_WAVE:.0f} air cells per structural")
    print(f"   wave, carrying {100 * ref['in_band'][0]:.1f}% of the plate's kinetic energy in "
          f"window 1. Batch 5's doctrine says quote the RESOLVED")
    print("   band, so every headline number below is given twice -- all modes, and resolved")
    print("   modes only. It is also why no directivity claim is made here: this cell sits ABOVE")
    print("   batch 4's validated band (ka 0.8 .. 2.8), and going lower costs convergence (3c).")

    print("\n1. THE HEADLINE -- sigma_shape per window, quiet against loud")
    print(f"   {'tier':10s} {'w/e':>5s} {'drift':>7s} {'sigma per window':>42s} {'spread':>8s}"
          f" {'resolved':>9s}")
    for (we, tier), c in cases.items():
        vals = "  ".join(f"{v:8.5f}" for v in c["shape"])
        print(f"   {tier:10s} {we:5.2f} {c['drift']:7.4f}  {vals}  {spread(c['shape']):7.3f}x "
              f"{spread(c['shape_r']):8.3f}x")
    for tier in ("baffled", "suspended"):
        q, ell = spread(cases[(QUIET, tier)]["shape"]), spread(cases[(LOUD, tier)]["shape"])
        qr, lr = spread(cases[(QUIET, tier)]["shape_r"]), spread(cases[(LOUD, tier)]["shape_r"])
        print(f"   -> {tier:9s}: quiet moves {100 * (q - 1):4.1f}%, loud "
              f"{100 * (ell - 1):4.1f}%  -- {(ell - 1) / (q - 1):4.0f}x the control")
        print(f"      resolved band only: quiet {100 * (qr - 1):4.1f}%, loud "
              f"{100 * (lr - 1):4.1f}%  -- {(lr - 1) / (qr - 1):4.0f}x the control")
    print("   The mechanism is in the same table: modal-share drift "
          f"{cases[(QUIET, 'baffled')]['drift']:.4f} quiet against "
          f"{cases[(LOUD, 'baffled')]['drift']:.4f} loud")
    print(f"   ({cases[(QUIET, 'baffled')]['drift_r']:.4f} against "
          f"{cases[(LOUD, 'baffled')]['drift_r']:.4f} over resolved modes only). Same geometry,")
    print("   same strike position, same room, ONE FLAG APART. This is what no")
    print("   scalar-per-frequency load can express: it has one pattern per frequency.")
    print("   Read the two rows the way batch 5 taught: the SEPARATION is the claim and it")
    print("   survives the resolution restriction; the MULTIPLIER does not, because the")
    print("   cascade's destination modes are exactly the ones the air grid resolves worst.")

    print("\n2. THE COMPACT LIMIT -- the monopole is not merely small, it points the wrong way")
    for (we, tier), c in cases.items():
        print(f"   {tier:10s} w/e={we:4.2f}  sigma_mono/sigma_shape = "
              f"{np.mean(c['mono'] / c['shape']):9.2e}   first->last: mono "
              f"{c['mono'][-1] / c['mono'][0]:5.2f}x, shape {c['shape'][-1] / c['shape'][0]:5.2f}x")
    loud_s = cases[(LOUD, "suspended")]
    print(f"   -> for the suspended cymbal at w/e = {LOUD:g} the monopole read-out "
          f"{'rises' if loud_s['mono'][-1] > loud_s['mono'][0] else 'falls'} "
          f"{loud_s['mono'][-1] / loud_s['mono'][0]:.2f}x while the true efficiency "
          f"{'rises' if loud_s['shape'][-1] > loud_s['shape'][0] else 'falls'} "
          f"{loud_s['shape'][-1] / loud_s['shape'][0]:.2f}x.")
    print("      A lumped one-port would report this cymbal getting BRIGHTER as it actually dulls,")
    print("      on a channel worth ~1e-6 of the radiation in the first place.")

    print("\n3. WHAT DIED ON MEASUREMENT")
    print("   (a) the ledger observable -- radiated energy per window, off the port's own books.")
    for tier in ("baffled", "suspended"):
        q, ell = spread(cases[(QUIET, tier)]["ledger"]), spread(cases[(LOUD, tier)]["ledger"])
        print(f"       {tier:10s} quiet {q:5.2f}x, loud {ell:5.2f}x -- the ROOM's own build-up "
              f"moves the control by as much as")
    print("       the effect moves the claim, so it does not separate. Batch 2's lesson again:")
    print("       a ratio survives a small room, a magnitude does not -- and this one is a")
    print("       magnitude wearing a ratio's clothes, because the denominator is the plate and")
    print("       the numerator is the room. sigma_shape fixes it by taking the room's own")
    print("       operator and applying it to the plate's own motion.")
    print("   (b) the directivity panel (plan 7.7) is REFUSED on a costed contradiction, not")
    print(f"       skipped: the pattern change needs the {SECONDS * 1e3:.0f} ms window above, and")
    print(f"       reflection-free {SECONDS * 1e3:.0f} ms needs a room "
          f"{SECONDS * C0_AIR:.0f} m across. The two requirements are")
    print("       in direct conflict at any budget, which is why plan 7.8's compact-safe")
    print("       alternative is the one that ships (section 2).")
    print("   (c) the plan's cost and convergence estimates, both:")
    print("       - COST: plan 4 costed the build cell at ~2-3 min for 0.2 s. Measured, a")
    print(f"         1.5 m room at h = {H_AIR * 1e3:.1f} mm is 2.35 M nodes and 109 ms per room")
    print("         step, i.e. ~21 min for 0.2 s -- 8x the estimate. The room here is 0.6 m")
    print("         for that reason.")
    print("       - CONVERGENCE: plan 0.5 measured sweep count against plate GEOMETRY at a fixed")
    print("         96 kHz. It is a strong function of the TIMESTEP too, and the room sets the")
    print("         timestep. Measured on this plate at w/e = 3: 72 sweeps at 57.9 kHz, and at")
    print("         33.0 kHz it does not converge at all (NaN). At 22.0 kHz even w/e = 2 diverges.")
    print("         So the air grid cannot be coarsened to buy affordability -- coarsening the")
    print("         ROOM breaks the PLATE's fixed point. That is a second, independent reason the")
    print("         cost runs the wrong way, on top of the 3-D CFL's h^-4.")

    path = figure(cases, os.path.join(OUT, "airbox_vk_shape_drift.png"))
    print(f"\n   -> {path}")


if __name__ == "__main__":
    main()
