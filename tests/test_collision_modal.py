"""Quantitative validation for the string <-> distributed-barrier collision (model #8).

Two independent oracles that energy conservation (``test_collision_energy.py``) cannot supply:

1. **Static-equilibrium magnitude oracle.** Energy conservation proves only *internal consistency* —
   the force injection and the ``h*sum phi`` PE telescope together, but *both could carry a
   compensating scale factor* and still conserve. This test pins the **absolute** coupling magnitude
   against known ``K/rho``. With ``alpha = 1`` and a full-interior linear-spring bed (barrier a hair
   above rest so ``eta > 0`` everywhere), at rest ``eta+ = eta-`` so the discrete gradient hits its
   Taylor branch (no time-averaging warp) and the scheme's discrete fixed point equals the
   *continuous* augmented equilibrium ``S u* = (K/rho) b`` with ``S = -L + (K/rho) I`` — exactly.
   Start both history levels at that closed-form ``u*`` with zero velocity and the sim holds it to
   ~1e-14. A **negative control** (double the coupling) moves the true fixed point, so the sim
   drifts off ``u*`` by orders of magnitude — the gate has teeth.

   (A *frequency* eigenvalue oracle — partials vs ``eig(-L + (K/rho) I)`` — is **not** tight here:
   the ``alpha = 1`` discrete gradient is ``K(eta+ + eta-)/2``, whose time-weights ``(1/2,0,1/2)``
   differ from the elastic theta-term's ``(theta,1-2theta,theta)``, warping the discrete partials
   away from the continuous augmented eig at finite ``k``. The static test covers magnitude instead;
   the frequency shift is a diagnostic signature, in ``test_collision_signature.py``.)

2. **Single-active-node -> scalar collapse.** When exactly one node contacts, the dense vector solve
   degenerates to the mallet's *scalar* ``solve_contact`` with admittance ``g = G_jj = (k^2/rho)
   (A^-1)_jj``. The two solvers differ (vector = damped Newton + Armijo; scalar = Newton + brentq),
   so agreement to ~newton_tol checks that both converge to the same unique root — validating the
   degenerate path against already-green code.
"""

import numpy as np
from helpers import make_barrier_string

from physsynth.core.collision import solve_contact


def _augmented_equilibrium(bar, eps):
    """Closed-form continuous augmented equilibrium ``u*`` (interior) for a full-interior linear
    spring bed at height ``eps``: ``S u* = (K/rho) eps``, ``S = -L + (K/rho) I``."""
    s = bar.string
    L = s._L.toarray()
    m = s.N - 1
    S = -L + (bar.K / s.rho) * np.eye(m)
    rhs = (bar.K / s.rho) * eps * np.ones(m)
    return np.linalg.solve(S, rhs), S


def _seat_at_equilibrium(bar, ustar):
    """Force both string history levels to the interior equilibrium ``ustar`` (a true static start —
    ``set_state``'s consistent ``u^{-1}`` is the *free*-string one, which is off-equilibrium)."""
    uf = np.zeros(bar.string.N + 1)
    uf[1:-1] = ustar
    bar.set_state(uf, 0.0)
    bar.string.u = uf.copy()
    bar.string.u_prev = uf.copy()
    bar.penetration = bar._b - bar.string.u[bar._support]


# -- Oracle 1: static-equilibrium magnitude gate --------------------------------------------------

def test_static_equilibrium_matches_closed_form():
    # alpha=1 full-interior spring bed. Seated at the closed-form continuous equilibrium, the sim
    # must hold it to ~1e-14 -> the discrete coupling magnitude equals the intended K/rho exactly.
    eps = 2.0e-3
    bar = make_barrier_string(N=60, lam=0.5, K=3000.0, alpha=1.0, barrier=eps)
    ustar, _ = _augmented_equilibrium(bar, eps)
    _seat_at_equilibrium(bar, ustar)
    assert np.all(bar.penetration > 0.0), "not in permanent contact — pick a larger eps"
    drift = 0.0
    for _ in range(2000):
        bar.step()
        drift = max(drift, np.max(np.abs(bar.string.u[1:-1] - ustar)))
    scale = np.max(np.abs(ustar))
    assert drift < 1e-13 * max(scale, 1.0), (
        f"held equilibrium to only {drift:.2e} (deflection {scale:.2e}) — coupling magnitude off"
    )


def test_static_equilibrium_negative_control():
    # The gate has teeth: doubling the coupling (both the admittance block G and the applied-force
    # prefactor) moves the true fixed point, so seating at the SINGLE-K equilibrium drifts by orders
    # of magnitude. A wrong force_pref would fail the test above exactly this way.
    eps = 2.0e-3
    good = make_barrier_string(N=60, lam=0.5, K=3000.0, alpha=1.0, barrier=eps)
    ustar, _ = _augmented_equilibrium(good, eps)
    _seat_at_equilibrium(good, ustar)
    good_drift = 0.0
    for _ in range(400):
        good.step()
        good_drift = max(good_drift, np.max(np.abs(good.string.u[1:-1] - ustar)))

    bad = make_barrier_string(N=60, lam=0.5, K=3000.0, alpha=1.0, barrier=eps)
    _seat_at_equilibrium(bad, ustar)
    bad._G = bad._G * 2.0            # double the whole contact coupling (G and the applied force)
    bad._force_pref = bad._force_pref * 2.0
    bad_drift = 0.0
    for _ in range(400):
        bad.step()
        bad_drift = max(bad_drift, np.max(np.abs(bad.string.u[1:-1] - ustar)))

    assert bad_drift > 1e4 * good_drift, (
        f"negative control too weak: good drift {good_drift:.2e}, doubled-coupling {bad_drift:.2e}"
    )


# -- Oracle 2: single-active-node -> scalar collapse ----------------------------------------------

def test_single_node_collapses_to_scalar_solver():
    # A point barrier at one node; the string's antinode swings into it. Each step, cross-check the
    # vector solver's penetration against the imported scalar solve_contact at g = G_jj.
    N, node = 80, 40
    b = np.full(N + 1, -np.inf)
    b[node] = -1.0e-3
    bar = make_barrier_string(N=N, lam=0.4, K=8.0e5, alpha=1.5, barrier=b)
    x = bar.string.x
    bar.set_state(5.0e-3 * np.sin(np.pi * x / bar.string.L))
    g = float(bar._G[0, 0])  # = (k^2/rho)(A^-1)_jj (scalar admittance of the one contact node)

    contact_steps = 0
    max_force = 0.0
    max_mismatch = 0.0
    for _ in range(800):
        eta_prev = (bar._b - bar.string.u_prev[bar._support])[0]
        seed = bar.penetration[0]
        bar.step()
        eta_v = bar.penetration[0]
        f_v = bar.contact_force[0]
        # reconstruct the free penetration the vector solver saw: eta_free = eta + g*f (applied).
        eta_free = eta_v + g * f_v
        eta_s, _, _ = solve_contact(
            eta_free, eta_prev, g, bar.K, bar.alpha, 0.0, bar.k,
            tol=bar.eta_tol, seed=seed, newton_tol=1e-14,
        )
        max_mismatch = max(max_mismatch, abs(eta_v - eta_s))
        if f_v > 0.0:
            contact_steps += 1
        max_force = max(max_force, f_v)

    assert contact_steps > 100, f"point barrier barely touched ({contact_steps} steps)"
    assert max_force > 1.0, f"contact force {max_force:.2e} N too small to be a real test"
    assert max_mismatch < 1e-13, (
        f"vector (m=1) and scalar solvers disagree by {max_mismatch:.2e} — not the same root"
    )
