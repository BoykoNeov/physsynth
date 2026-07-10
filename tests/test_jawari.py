"""Validation for the **jawari / buzzing bridge** — the sitar & tanpura curved-bridge timbre.

The jawari is **not new core physics**: it composes model #8 (:class:`BarrierString`, the string
against a one-sided distributed nonlinear barrier) with a *curved* barrier profile hugging the
termination (:func:`jawari_barrier`). The string wraps onto the parabolic bridge on each downswing
and its departure point travels along the curve — the mechanism behind the shimmering, sustained,
high-partial-rich "life" (jawari) of these instruments.

So the correctness gates are model #8's, re-exercised in the *curved, persistently-wrapping* regime
its earlier tests never touched (they used a flat rail / a point fret):

* **Money gate — lossless energy conservation** through the sustained curved contact (drift ~1e-12).
* **Money gate — static-equilibrium magnitude oracle on a curved ``alpha = 1`` profile**: the
  discrete fixed point equals the closed-form ``S u* = (K/rho) b`` to machine precision, with a
  negative control that has teeth. This pins the coupling magnitude for the *curved* profile, not a
  flat one.

and the *jawari-specific* signatures (diagnostic tier), which must separate the curve from the
already-validated flat rail / clean string:

* **Sustained brightness (shimmer)**: the curved contact keeps regenerating high partials, so the
  spectral centroid stays elevated deep into the decay — unlike a clean string, whose tone darkens.
* **Travelling contact point (wrap precession)**: the departure/wrap edge sweeps a wide span on the
  curve, versus a fixed cluster on a flat rail at the same minimum clearance.
"""

import numpy as np
import pytest
from helpers import jawari_barrier, make_jawari_string

from physsynth.core.collision import BarrierString

CONSERVE_TOL = 1e-10  # lossless, elastic: relative drift of total energy; observed ~7e-13
AMP = 8.0e-3          # first-mode pluck amplitude (m); swings down onto the bridge


# -- shared drivers -------------------------------------------------------------------------------

def _pluck(bar, amp=AMP):
    x = bar.string.x
    bar.set_state(amp * np.sin(np.pi * x / bar.string.L))


def _spectral_centroid(sig, fs):
    """Amplitude-weighted mean frequency (Hz) of ``sig`` — a brightness proxy."""
    w = np.hanning(len(sig))
    mag = np.abs(np.fft.rfft(sig * w))
    freqs = np.fft.rfftfreq(len(sig), 1.0 / fs)
    return float(np.sum(freqs * mag) / np.sum(mag))


def _run_pickup_wrap(bar, steps, node_frac=0.5):
    """Run ``steps``, return (pickup series at a mid node, wrap-edge series, per-step contact flag).

    The wrap edge is the *furthest-in-contact* support-node index (``-1`` when clear) — its travel
    is the jawari's precessing departure point."""
    _pluck(bar)
    node = max(1, int(node_frac * bar.string.N))
    pickup = np.empty(steps)
    wrap = np.full(steps, -1.0)
    for i in range(steps):
        bar.step()
        m = bar.contact_mask()
        if np.any(m):
            wrap[i] = float(np.max(np.where(m)[0]))
        pickup[i] = bar.string.displacement_at(node)
    return pickup, wrap, wrap >= 0.0


# =================================================================================================
# Money gate 1 — lossless energy conservation through the curved wrap
# =================================================================================================

@pytest.mark.parametrize("alpha", [1.0, 1.5, 2.0])
def test_lossless_energy_conserved_through_curved_wrap(alpha):
    # No string loss, elastic bridge: E_string + bridge PE is conserved to machine precision even
    # through the many-node persistent wrap the flat-rail tests never exercised.
    bar = make_jawari_string(alpha=alpha, sigma0=0.0, sigma1=0.0, hysteresis=0.0)
    _pluck(bar)
    e = np.empty(6001)
    e[0] = bar.energy()
    for i in range(1, 6001):
        bar.step()
        assert bar.newton_iters < bar.newton_maxiter, f"contact solve stalled at step {i}"
        e[i] = bar.energy()
    drift = np.max(np.abs(e - e[0])) / abs(e[0])
    assert drift < CONSERVE_TOL, f"jawari energy drift {drift:.2e} (alpha={alpha})"


def test_string_actually_wraps_the_bridge():
    # Sanity: the bridge must be wrapped over a real span for a real fraction of the run, else the
    # conservation gate is a trivial free-string re-test. Demand multi-node contact and real force.
    bar = make_jawari_string(sigma0=0.3)
    _pluck(bar)
    max_span = 0
    max_force = 0.0
    contact_steps = 0
    for _ in range(6000):
        bar.step()
        m = bar.contact_mask()
        if np.any(m):
            contact_steps += 1
            max_span = max(max_span, int(np.sum(m)))
        max_force = max(max_force, float(np.max(np.abs(bar.contact_force))))
    assert contact_steps > 300, f"bridge barely touched ({contact_steps} steps)"
    assert max_span >= 3, f"string never wrapped multiple nodes (max span {max_span})"
    assert max_force > 1.0, f"peak contact force {max_force:.2e} N too small to test the coupling"


# =================================================================================================
# Money gate 2 — static-equilibrium magnitude oracle on a CURVED alpha=1 profile
# =================================================================================================

def _curved_equilibrium(bar):
    """Closed-form continuous augmented equilibrium ``u*`` for the (partial-support) curved linear
    spring bed: ``S u* = (K/rho) b`` with ``S = -L + (K/rho)·diag(mask)``, mask over the bridge
    support, and ``b`` the barrier height there (0 off support)."""
    s = bar.string
    Lop = s._L.toarray()
    m = s.N - 1
    mask = np.zeros(m)
    mask[bar._support - 1] = 1.0
    S = -Lop + (bar.K / s.rho) * np.diag(mask)
    rhs = np.zeros(m)
    rhs[bar._support - 1] = (bar.K / s.rho) * bar._b
    return np.linalg.solve(S, rhs), S


def _seat_at(bar, ustar):
    uf = np.zeros(bar.string.N + 1)
    uf[1:-1] = ustar
    bar.set_state(uf, 0.0)
    bar.string.u = uf.copy()
    bar.string.u_prev = uf.copy()
    bar.penetration = bar._b - bar.string.u[bar._support]


def test_curved_static_equilibrium_matches_closed_form():
    # alpha=1, crest preloaded above rest (clearance<0) so the WHOLE curved span is in gentle
    # contact at rest -> the discrete gradient hits its no-warp Taylor branch and the sim must hold
    # the closed-form curved equilibrium to ~1e-14. Proves the *curved* profile is wired at the
    # intended K/rho, not just a flat rail.
    bar = make_jawari_string(K=3000.0, alpha=1.0, depth=1.5e-3, clearance=-2.0e-3, lam=0.5)
    ustar, _ = _curved_equilibrium(bar)
    _seat_at(bar, ustar)
    assert np.all(bar.penetration > 0.0), "not all support in contact — raise the preload"
    drift = 0.0
    for _ in range(2000):
        bar.step()
        drift = max(drift, np.max(np.abs(bar.string.u[1:-1] - ustar)))
    scale = np.max(np.abs(ustar))
    assert drift < 1e-13 * max(scale, 1.0), (
        f"held curved equilibrium to only {drift:.2e} (deflection {scale:.2e}) — magnitude off"
    )


def test_curved_static_equilibrium_negative_control():
    # Teeth: doubling the coupling moves the true fixed point, so seating at the single-K curved
    # equilibrium drifts by orders of magnitude (a wrong force_pref would fail the gate this way).
    good = make_jawari_string(K=3000.0, alpha=1.0, depth=1.5e-3, clearance=-2.0e-3, lam=0.5)
    ustar, _ = _curved_equilibrium(good)
    _seat_at(good, ustar)
    good_drift = 0.0
    for _ in range(400):
        good.step()
        good_drift = max(good_drift, np.max(np.abs(good.string.u[1:-1] - ustar)))

    bad = make_jawari_string(K=3000.0, alpha=1.0, depth=1.5e-3, clearance=-2.0e-3, lam=0.5)
    _seat_at(bad, ustar)
    bad._G = bad._G * 2.0
    bad._force_pref = bad._force_pref * 2.0
    bad_drift = 0.0
    for _ in range(400):
        bad.step()
        bad_drift = max(bad_drift, np.max(np.abs(bad.string.u[1:-1] - ustar)))
    assert bad_drift > 1e4 * good_drift, (
        f"negative control too weak: good {good_drift:.2e}, doubled-coupling {bad_drift:.2e}"
    )


# =================================================================================================
# Signature 1 — the bridge buzzes: elevated brightness vs a clean string
# =================================================================================================

def test_jawari_brightens_the_tone():
    steps = 12000
    jaw = make_jawari_string(sigma0=0.5)
    clean = make_jawari_string(sigma0=0.5, clearance=1.0)  # crest 1 m below rest -> never contacts
    pj, _, cflag = _run_pickup_wrap(jaw, steps)
    pc, _, _ = _run_pickup_wrap(clean, steps)
    assert np.any(cflag), "jawari never contacted the bridge"
    fs = 1.0 / jaw.k
    cj = _spectral_centroid(pj, fs)
    cc = _spectral_centroid(pc, fs)
    assert cj > 2.0 * cc, f"bridge did not brighten the tone: clean={cc:.0f} jawari={cj:.0f}"


# =================================================================================================
# Signature 2 — shimmer: the brightness is SUSTAINED (curved contact keeps regenerating highs)
# =================================================================================================

def test_jawari_brightness_is_sustained():
    # The defining perceptual jawari feature. A clean string's midpoint pickup sits near the
    # fundamental throughout (centroid ratio ~1, and *low*); the jawari's curved contact re-injects
    # highs every downswing, so late into the decay it is still ~3x brighter than the clean string
    # (the robust claim) and has not collapsed back toward its own fundamental.
    steps = 12000
    half = steps // 2
    jaw = make_jawari_string(sigma0=0.5)
    clean = make_jawari_string(sigma0=0.5, clearance=1.0)
    pj, _, _ = _run_pickup_wrap(jaw, steps)
    pc, _, _ = _run_pickup_wrap(clean, steps)
    fs = 1.0 / jaw.k
    j_e, j_l = _spectral_centroid(pj[:half], fs), _spectral_centroid(pj[half:], fs)
    c_e, c_l = _spectral_centroid(pc[:half], fs), _spectral_centroid(pc[half:], fs)
    j_ratio, c_ratio = j_l / j_e, c_l / c_e
    assert j_l > 2.5 * c_l, (  # the robust claim: highs persist deep into the decay
        f"late-window brightness not dominated by the bridge (jawari {j_l:.0f} vs clean {c_l:.0f})"
    )
    assert j_l > 0.7 * j_e, (  # the jawari has not collapsed back toward its own fundamental
        f"jawari brightness collapsed (early {j_e:.0f} -> late {j_l:.0f})"
    )
    assert j_ratio > c_ratio, (
        f"jawari not more sustained than clean string (jawari {j_ratio:.2f} vs {c_ratio:.2f})"
    )


# =================================================================================================
# Signature 3 — travelling contact: the wrap edge precesses (curve vs flat rail, matched clearance)
# =================================================================================================

def test_wrap_edge_travels_more_than_a_flat_rail():
    # On the curve the departure point sweeps along the bridge; on a flat rail at the same minimum
    # clearance the contact is a cluster pinned near the antinode crossing. The wrap-edge spread is
    # the discriminator (both buzz; only the curve *travels*).
    steps = 12000
    jaw = make_jawari_string(sigma0=0.5)
    crest = float(jaw._b[0])  # highest (nearest-string) point of the curve = matched min clearance
    flat = make_jawari_string(sigma0=0.5)
    flat._b = np.full_like(flat._b, crest)  # flatten the same support to a rail at the crest level
    _, wj, _ = _run_pickup_wrap(jaw, steps)
    _, wf, _ = _run_pickup_wrap(flat, steps)
    sj = float(np.std(wj[wj >= 0.0]))
    sf = float(np.std(wf[wf >= 0.0]))
    assert sj > 1.5 * sf, (
        f"wrap edge does not travel more on the curve (curve {sj:.2f} vs flat {sf:.2f})"
    )


# =================================================================================================
# Sanity — profile shape & protocol
# =================================================================================================

def test_jawari_profile_is_a_curved_ramp():
    x = np.linspace(0.0, 1.0, 101)
    b = jawari_barrier(x, 1.0, width_frac=0.15, depth=1.0e-3, clearance=0.0)
    on = np.isfinite(b)
    assert not np.isfinite(b[0]), "the x=0 termination must be off-support (-inf)"
    vals = b[on]
    assert np.all(np.diff(vals) < 0.0), "bridge must curve monotonically away from the crest"
    assert not np.any(np.isfinite(b[x > 0.15])), "no barrier beyond the bridge span"


def test_jawari_is_a_barrier_string():
    bar = make_jawari_string()
    assert isinstance(bar, BarrierString)
    assert bar._support.size >= 8, "too few bridge nodes to resolve the wrap"
    assert bar._support.size < 100, "support over the dense-solve BLAS cliff"
    assert callable(bar.step) and callable(bar.energy)
