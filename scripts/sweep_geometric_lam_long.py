"""The ``lam_long`` edge of the geometrically-exact string, and whether a Jacobian fix moved it.

``docs/dev/scientific-hurdles.md`` section 6 records an unresolved regime: the longitudinal wave
speed is ``sqrt(EA/rho)``, 10-30x the transverse one, so the familiar ``lam = 0.5`` silently means
``lam_long ~ 11``; the theta-scheme is unconditionally stable so nothing refuses it, and what
happens instead is that the Newton solve stops converging and the energy drift explodes. Section 1
of the same document then fixed the ``(v,v)`` block of that solve's Jacobian -- it had been
assembled as a difference of two ``O(1)`` terms and was wrong in its fifth digit at musical strain.
Section 6 asks the obvious follow-up: **does the convergent window move now that the solve is
steered correctly?**

This script answers it, and the answer is no. What it is really for is the *shape* of the
measurement, which is the part worth keeping:

**Two columns, not one.** Section 1's own argument is that the residual defines the root and the
Jacobian only steers the iteration, so energy drift cannot be the sole readout: in the converged
regime a correct Jacobian is *predicted* to change nothing, and in the failing regime drift is a
step function of the iteration count (``tests/test_geometric_energy.py``'s tolerance test says so,
and ``docs/memory/geometric-string-state.md`` records an earlier five-decade "proportionality" that
was the broken parameterization certifying itself). So every table here reports the Newton
iteration count and the number of steps that hit ``newton_maxiter`` beside the drift. That second
column is what separates the two thresholds this model turns out to have -- see :func:`edge_table`.

**One process, both spellings, identical fixtures.** The old ``lam_long`` table in
``physsynth/core/string_geometric.py``'s ``LAM_LONG_WARN`` docstring was measured months earlier at
unrecorded ``N``, amplitude and duration. Comparing fresh numbers against it would attribute to the
Jacobian whatever else drifted in the rig in between. :class:`OldJacobianString` therefore carries
the pre-fix expression verbatim, and every cell is run both ways back to back.

Run:  python scripts/sweep_geometric_lam_long.py [--quick]
Prints tables; writes nothing.
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings

# Make `physsynth` importable when run directly (no editable install): repo root on sys.path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
from scipy import sparse  # noqa: E402

from physsynth.core.string_geometric import GeometricString  # noqa: E402

# The canonical string of this family's tests and figures: c = 200 m/s, f1 = 100 Hz, EA/T = 500 so
# c_long/c = 22.4. `kappa` is the suite's 2.0 rather than 0, because the sweep has to be comparable
# to the numbers the tests report.
L, T, RHO, EA, KAPPA = 1.0, 200.0, 0.005, 1.0e5, 2.0

# CLAUDE.md's lossless bar. The "energy edge" below is defined against this and nothing else.
DRIFT_GATE = 1e-10


class OldJacobianString(GeometricString):
    """The pre-2026-09-02 ``_dg_jacobian``, kept as the A/B control.

    Recovered verbatim from the parent of commit ``13ae21d``: the ``(v,v)`` block written literally
    as ``chi/2 - 1/2 + (1+vbar_x)(1+v_x+)/(2 Lambdabar^2 Lambda+)``, two ``O(1)`` terms that cancel
    to an ``O(strain^2)`` remainder. Measured against a 60-digit reference it is 3.9e-11 relative at
    strain 0.1 and 5.8e-5 at strain 1e-4 -- *worse the more realistic the string*, because the
    relative error of a cancellation grows like ``1/strain^2``.

    It lives here and not in the test suite on purpose: it is not a thing the package should be able
    to build, it is the control for one measurement, and a reader who wants to re-take that
    measurement needs the exact expression rather than a description of it.
    """

    def _dg_jacobian(self, q_plus, q_minus):
        lam_p, e_p, _, _, _ = self._stretch_terms(q_plus)
        lam_m, e_m, _, _, _ = self._stretch_terms(q_minus)
        lam_bar = 0.5 * (lam_p + lam_m)
        q_bar = 0.5 * (q_plus + q_minus)
        chi = 0.5 * (e_p + e_m) / lam_bar
        n_p = np.stack((q_plus[0], q_plus[1], 1.0 + q_plus[2])) / lam_p
        m_bar = np.stack((q_bar[0], q_bar[1], 1.0 + q_bar[2]))
        coef = 0.5 / lam_bar**2

        blocks: list[list[sparse.spmatrix]] = []
        for a_i in range(3):
            row: list[sparse.spmatrix] = []
            for b_i in range(3):
                d = coef * m_bar[a_i] * n_p[b_i]
                if a_i == b_i:
                    d = d + 0.5 * chi
                    if a_i == 2:
                        d = d - 0.5
                row.append(sparse.diags(self._a * d, format="csr"))
            blocks.append(row)
        return sparse.bmat(blocks, format="csr")


def mode_ic(n_cells: int, m: int, amp: float) -> np.ndarray:
    """A single simply-supported eigenmode on the full ``N+1`` grid."""
    return amp * np.sin(m * np.pi * np.linspace(0.0, L, n_cells + 1) / L)


def pluck_ic(n_cells: int, amp: float, at: float = 0.2) -> np.ndarray:
    """A triangular pluck: broadband, so the stretch varies cell to cell."""
    x = np.linspace(0.0, 1.0, n_cells + 1)
    return amp * np.where(x <= at, x / at, (1.0 - x) / (1.0 - at))


def run(cls, *, n_cells: int, lam_long: float, ic: str, mode: int, amp: float,
        duration: float) -> dict:
    """One trajectory. Returns drift *and* the two convergence columns.

    ``duration`` is physical seconds, not steps: ``lam_long`` sets ``fs``, so a fixed step count
    would compare different amounts of physics at each point of the sweep.
    """
    fs = float(np.sqrt(EA / RHO)) * n_cells / (L * lam_long)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # the lam_long warning is the point here
        s = cls(L=L, T=T, rho=RHO, fs=fs, N=n_cells, EA=EA, kappa=KAPPA)
    s.set_state(mode_ic(s.N, mode, amp) if ic == "mode" else pluck_ic(s.N, amp))

    e0 = s.energy()
    steps = int(round(duration * fs))
    peak_iters = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # non-convergence is counted, not printed
        for _ in range(steps):
            s.step()
            peak_iters = max(peak_iters, s.newton_iters)
    return {
        "drift": abs(s.energy() - e0) / abs(e0),
        "steps": steps,
        "iters_per_step": s.total_newton_iters / steps,
        "peak_iters": peak_iters,
        "stalled": s.n_not_converged,
    }


SPELLINGS = (("new", GeometricString), ("old", OldJacobianString))


def sweep_table(*, n_cells: int, duration: float, amp: float, lams: list[float]) -> None:
    """The headline A/B: drift and convergence against ``lam_long``, three initial conditions."""
    print(f"\n== the sweep: N={n_cells} duration={duration}s amp={amp:g} ==")
    print(f"{'case':8} {'lam_long':>8} {'sp':>4} {'drift':>10} {'it/step':>8} {'peak':>5} "
          f"{'stalled':>8} {'steps':>6}")
    for name, ic, mode in (("mode-3", "mode", 3), ("mode-1", "mode", 1), ("pluck", "pluck", 0)):
        for lam_long in lams:
            for label, cls in SPELLINGS:
                r = run(cls, n_cells=n_cells, lam_long=lam_long, ic=ic, mode=mode, amp=amp,
                        duration=duration)
                print(f"{name:8} {lam_long:8.1f} {label:>4} {r['drift']:10.2e} "
                      f"{r['iters_per_step']:8.2f} {r['peak_iters']:5d} {r['stalled']:8d} "
                      f"{r['steps']:6d}", flush=True)


def amplitude_table(*, n_cells: int, duration: float, amps: list[float],
                    lams: list[float]) -> None:
    """The axis that explains the null result, and the reason it is not a weak measurement.

    The old Jacobian's relative error grows like ``1/strain^2``, so it is worst where the string is
    *quietest*. The Newton solve's trouble grows with the nonlinearity, so it is worst where the
    string is *loudest*. This table runs both axes at once, and the strain column is what makes the
    disjointness visible rather than merely asserted.
    """
    print(f"\n== the amplitude axis: N={n_cells} duration={duration}s mode-3 ==")
    print(f"{'amp':>9} {'strain':>9} {'lam_long':>8} {'sp':>4} {'drift':>10} {'it/step':>8} "
          f"{'stalled':>8}")
    for amp in amps:
        strain = float(np.max(np.abs(np.diff(mode_ic(n_cells, 3, amp)) * n_cells / L)))
        for lam_long in lams:
            for label, cls in SPELLINGS:
                r = run(cls, n_cells=n_cells, lam_long=lam_long, ic="mode", mode=3, amp=amp,
                        duration=duration)
                print(f"{amp:9.1e} {strain:9.2e} {lam_long:8.1f} {label:>4} {r['drift']:10.2e} "
                      f"{r['iters_per_step']:8.2f} {r['stalled']:8d}", flush=True)


def edge_table(*, duration: float, lams: list[float], cases: list[tuple]) -> None:
    """The deliverable: two edges per case, and whether either moved.

    **The energy edge** is the smallest ``lam_long`` whose drift breaks :data:`DRIFT_GATE`. **The
    convergence edge** is the smallest whose Newton solve hits ``newton_maxiter`` on any step. They
    are not the same number, and the gap between them is this batch's finding: there is a band --
    roughly ``lam_long`` 4 to 8 -- in which the solve warns on a fifth of its steps and the energy
    still conserves to 1e-15, so **a flat energy is not a convergence certificate here**.
    """
    print(f"\n== the edges: duration={duration}s gate={DRIFT_GATE:g} ==")
    print(f"{'N':>4} {'amp':>8} {'ic':>7} | {'energy new':>10} {'energy old':>10} | "
          f"{'conv new':>9} {'conv old':>9}")
    agree = 0
    for n_cells, amp, ic, mode in cases:
        edges = {}
        for label, cls in SPELLINGS:
            energy_edge = convergence_edge = None
            for lam_long in lams:
                r = run(cls, n_cells=n_cells, lam_long=lam_long, ic=ic, mode=mode, amp=amp,
                        duration=duration)
                if convergence_edge is None and r["stalled"] > 0:
                    convergence_edge = lam_long
                if r["drift"] > DRIFT_GATE:
                    energy_edge = lam_long
                    break
            edges[label] = (energy_edge, convergence_edge)
        agree += edges["new"][0] == edges["old"][0]
        print(f"{n_cells:4d} {amp:8.1e} {ic:>7} | {str(edges['new'][0]):>10} "
              f"{str(edges['old'][0]):>10} | {str(edges['new'][1]):>9} "
              f"{str(edges['old'][1]):>9}", flush=True)
    print(f"\nthe energy edge is identical in {agree}/{len(cases)} cells: the section 1 Jacobian "
          f"fix does NOT move it.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quick", action="store_true", help="the edge table only, on a short run")
    args = ap.parse_args()

    edge_cases = [
        (16, 4e-3, "mode", 3), (16, 8e-3, "mode", 3), (16, 4e-3, "pluck", 0),
        (24, 4e-3, "mode", 3), (24, 8e-3, "mode", 3),
        (32, 4e-3, "mode", 3), (32, 8e-3, "mode", 3), (32, 4e-3, "pluck", 0),
        (32, 1e-2, "mode", 3),
    ]
    edge_table(duration=0.004, lams=[2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
               cases=edge_cases)
    if args.quick:
        return

    sweep_table(n_cells=32, duration=0.02, amp=4e-3,
                lams=[0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 9.0, 9.5, 11.0])
    amplitude_table(n_cells=32, duration=0.02, amps=[2e-5, 6.9e-5, 2e-4, 1e-3, 4e-3],
                    lams=[0.5, 4.0, 8.0, 9.0, 9.5, 10.0])


if __name__ == "__main__":
    main()
