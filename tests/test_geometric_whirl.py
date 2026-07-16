"""Model #10 — **the whirling threshold** (test 12), and the payoff of decision #1's knob.

Batch 2's last claim, and the one that needed machinery built for it two batches earlier. An
isotropic string **cannot** whirl: ``w -> -w`` is a reflection symmetry (Tier A/2), so a planar IC
stays planar bit-exactly, and the rotation generator ``dw = q_u(t)`` is an *exact* solution of the
out-of-plane variational equation — pinning one Floquet multiplier at ``+1``, and the flow being
Hamiltonian (Wronskian conserved), the other with it. **Both ``+1`` => marginal, never
exponential.** That is *why* ``kappa_u != kappa_w`` was built (decision #1): whirling is a threshold
instability, and a threshold needs a degeneracy to break.

**The mechanism is a Mathieu equation**, and every parameter in this file is chosen from it. Reduce
to one transverse mode pair — the shape ``phi`` is a *sine*, exact for both ``d_xx`` and ``d_xxxx``
under simply-supported ends, so the two polarizations **share it** even when detuned::

    q_u'' + wu^2 q_u + eps (q_u^2 + q_w^2) q_u = 0      eps = (a/4 rho) p^4,   a = EA - T0
    q_w'' + ww^2 q_w + eps (q_u^2 + q_w^2) q_w = 0      w^2 = c^2 p^2 + kappa^2 p^4

``eps`` is model #9's own Duffing coefficient (:func:`kc_mode_coefficients` under ``EA -> a``): the
quartic is isotropic, so the planar reduction is untouched by the second polarization. Drive the
``u`` plane (``q_u = A cos(Om t)``, ``Om^2 = wu^2 + (3/4) eps A^2``) and linearise out of plane::

    dq_w'' + [ ww^2 + eps A^2/2 + (eps A^2/2) cos(2 Om t) ] dq_w = 0

A **parametric pump at ``2 Om``**, sitting on the principal resonance. Writing
``delta = ww^2 - wu^2``, the tongue is ``|delta - eps A^2/4| < eps A^2/4``::

    unstable  <=>  0 < delta < eps A^2 / 2,     most unstable at  delta = eps A^2 / 4

Three consequences, and this file measures all three:

* **Gough's (1984) threshold** ``eps A^2 ~ dw0^2`` is that upper edge read in the amplitude
  direction: ``A_c = sqrt(2 delta / eps)``, so the threshold **moves as ``sqrt(delta)``**.
* **``delta = 0`` sits exactly *on* the tongue edge** — marginal. The leading-order Mathieu analysis
  reproduces the exact Floquet result it knows nothing about, which is the cheapest evidence the
  reduction is the right one.
* **Only the *soft* plane whirls.** ``delta > 0`` says the driven polarization must be the
  *lower*-frequency one; drive the stiff plane instead and ``delta < 0``: stable at every
  amplitude.

**The dimensionless coordinate is ``delta / (eps A^2)``**, and the whole file lives on it: the
tongue is ``(0, 0.5)`` with its peak at ``0.25``, whatever ``kappa_w``, ``A`` or ``N`` are
individually.
Measured (t = 0.06 s), growth of the out-of-plane seed across that coordinate::

    delta/eps A^2   0.00    0.07    0.25    0.41    0.50    0.80
    growth          1.00x  14.7x   76.3x   37.4x    8.4x    1.63x
                    ~~~~           ~~~~~                   ~~~~~
                    edge           peak                    outside

**Why the exaggeration is a microscope and not a thumb on the scale** — the same argument
``KAPPA_PHANTOM`` makes, and it has to be made again because the numbers here look wilder. A real
whirling string has a sub-Hz detuning and whirls over seconds; this one is detuned by ~17 Hz
(``kappa_w ~ 39``) at a 35 mm amplitude and whirls in ~12 ms. Both exaggerations are the *same*
exaggeration: the tongue is dimensionless, so holding ``delta/(eps A^2)`` fixed while scaling
``delta`` and ``eps A^2`` together preserves the physics exactly and only compresses the wall-clock
(the growth rate ``~ eps A^2 / 8 Om`` is what buys the speed). Nothing here changes *whether* the
string whirls — only how long you would have to wait to see it.

**What is exact and what is a limit.** The tongue is a **KC-limit oracle** (Tier C): ``eps`` comes
from the quasi-static reduction and ``Om`` from leading-order Duffing, so the *edges* are
approximate — measured, the upper one is soft (8.4x growth still at ``0.50``, dead by ``0.80``).
The **peak location** is the sharp part and it lands on ``0.25`` exactly. So this file asserts the
tongue's *shape* and its *scaling*, and reports the edge's softness rather than pinning it.

**A finding the plan did not have: its discarded precession rate is the whirl growth rate.** The
same Mathieu algebra predicts *how fast* the seed grows, ``(Om/2) sqrt(qM^2 - sigma^2)``, and it
holds across the tongue to 5-11 % (systematically low — leading-order, not a bug). At the centre it
collapses to ``eps A^2/(8 Om)``, which is — under ``w0 -> Om`` — exactly the ``Om_prec =
eps A^2/(8 w0)`` this plan derived by naive averaging and then **correctly rejected**: a degenerate
string has no precession rate, because rotational symmetry forces ``ww = wu`` at any amplitude. The
plan even named its own error ("the averaging drops the ``2w`` pump") without following it home:
the dropped pump's *strength* is that very quantity, and the pump is the whole instability. The
formula was never spurious — it was misattributed. It does not describe a degenerate string
precessing; it describes how fast a **detuned** one whirls.

**The honesty gate.** Every claim below is about a *seeded* perturbation. An unseeded planar run at
these same extreme parameters gives ``max|w| == 0.0`` exactly (Tier A/2 survives detuning), which is
what makes a growth ratio mean anything: there is no leak for it to be measuring.

**Runtime ~2.8 min**, nearly all of it in the four module-scoped fixtures (15 runs). The cost is the
Newton solve (2.0 ms/step vs 0.08 ms for :meth:`energy`, so the drift gate is free and is taken
every step). ``N = 16`` is deliberate and sufficient: mode 1 carries the entire claim, its discrete
``p^2`` is within 0.3 % of the continuum, and the tongue is refinement-invariant (centre at
``kappa_w = 39.05 / 39.01 / 39.00`` for ``N = 16 / 24 / 32``) — a test below spends the wall-clock
to hold that at 2x the grid, because an instability that moves with the mesh is an artifact, which
is exactly the standard model #9's parametric discovery had to meet.

References
----------
Gough (1984), "The nonlinear free vibration of a damped elastic string", JASA 75(6).
"""

import numpy as np
import pytest
from helpers import (
    EA_DEFAULT,
    L_DEFAULT,
    RHO_DEFAULT,
    T_DEFAULT,
    make_geometric_string,
    wave_speed,
)

from physsynth.analysis.damping import spatial_eigenvalue_p2
from physsynth.analysis.duffing import kc_mode_coefficients

# -- parameters, and why each is what it is --------------------------------------------------------

N = 16
LAM_LONG = 0.9
T_TOTAL = 0.06
MODE = 1
SEED_REL = 1e-3
"""Out-of-plane seed, as a fraction of the driven amplitude. Small enough that the run starts deep
in the linear (Mathieu) regime, large enough that ~6 e-foldings saturate it — which is what keeps
``T_TOTAL`` at 0.06 s rather than the 0.1 s a 1e-6 seed would need for the same conclusion."""

TENSION_RISE = 1.5
"""The nonlinearity strength, as the single-mode tension rise ``dT/T0`` — and it is the **one
number** the whole design is tuned against, from both sides.

**Up**, because it buys the wall-clock: the Mathieu growth rate is ``eps A^2/(8 Om)``, so the
e-folding time is ``~ 12 ms`` here and ``~ 650 ms`` at the 4 mm amplitude the rest of the batch-2
suite uses. A whirl test at a *realistic* amplitude is a ~5 s render per point, and there are 15.

**Down**, because of a genuine confound: model #9 measured a **planar** parametric instability
(in-plane modal exchange) above ``dT/T0 ~ 3``. That one also conserves energy, so the drift gate
cannot separate it from whirling — only staying away from it can. At ``kappa_u = 0`` the two are
directly comparable, because ``eps A^2 / wu^2 == dT/T0`` **exactly** (both are
``(a p^2 A^2/4)/T0``), which is the reason the driven plane is the flexible one. 1.5 is half of 3,
and the tests assert the driven field stays single-mode rather than trusting the margin: measured,
mode-3 content is 0.09 % here and 0.2 % at the 2.16 the ``sqrt(delta)`` test reaches."""

TONGUE_FRACS = (0.0, 0.07, 0.25, 0.41, 0.5, 0.8)
"""The sweep, in the dimensionless coordinate ``delta/(eps A^2)``: the edge, the rising flank, the
predicted peak, the falling flank, the predicted upper edge, and outside."""


def _kc(n=N, EA=EA_DEFAULT):
    """``(p2, wu^2, eps)`` — the KC modal reduction of mode 1 at ``kappa_u = 0``.

    The **discrete** ``p^2``, never the continuum ``(m pi/L)^2``: the tongue is compared against a
    string on a grid, and the batch-2 lesson (the phantom ladder) is that the oracle has to carry
    the scheme's own eigenvalue or it is measuring the formula.
    """
    p2 = spatial_eigenvalue_p2(n, L_DEFAULT / n, MODE)
    w0sq, eps = kc_mode_coefficients(
        c=wave_speed(T_DEFAULT, RHO_DEFAULT), kappa=0.0, EA=EA - T_DEFAULT,
        rho=RHO_DEFAULT, p2=p2, L=L_DEFAULT,
    )
    return p2, w0sq, eps


def _amp_for(tension_rise, n=N):
    """The amplitude at which ``eps A^2/wu^2 == tension_rise == dT/T0`` (exact at ``kappa_u=0``)."""
    _, w0sq, eps = _kc(n)
    return float(np.sqrt(tension_rise * w0sq / eps))


def _delta(kappa_w, n=N):
    """``ww^2 - wu^2 = (kappa_w^2 - kappa_u^2) p^4`` at ``kappa_u = 0``; the biharmonic eig is
    ``(p^2)^2`` because the scheme's ``d_xxxx`` is literally ``D2 @ D2`` (model #2's choice)."""
    p2, _, _ = _kc(n)
    return kappa_w**2 * p2**2


def _kappa_w_at(frac, amp, n=N):
    """The ``kappa_w`` placing the string at ``delta/(eps A^2) == frac`` — the sweep's inverse."""
    p2, _, eps = _kc(n)
    return float(np.sqrt(frac * eps * amp**2 / p2**2))


def _a_crit(kappa_w, n=N):
    """Gough's threshold amplitude ``A_c = sqrt(2 delta / eps)`` — the tongue's upper edge, read in
    the amplitude direction. A **limit** oracle (KC + leading-order Duffing), used to place runs and
    to state the ``sqrt(delta)`` scaling; never asserted to cents."""
    _, _, eps = _kc(n)
    return float(np.sqrt(2.0 * _delta(kappa_w, n) / eps))


def _mathieu_rate(kappa_w, amp, n=N):
    """The Mathieu growth rate ``(Om/2) sqrt(qM^2 - sigma^2)`` — zero outside the tongue.

    ``qM = eps A^2/(4 Om^2)`` is the pump's strength and ``sigma = (delta - eps A^2/4)/Om^2`` the
    detuning from the principal resonance, so the rate peaks at ``delta = eps A^2/4`` and vanishes
    at both edges — the tongue and its rate profile are the same formula, read twice.
    """
    _, w0sq, eps = _kc(n)
    ea2 = eps * amp**2
    om = np.sqrt(w0sq + 0.75 * ea2)
    qm = ea2 / (4 * om**2)
    sigma = (_delta(kappa_w, n) - ea2 / 4) / om**2
    return float((om / 2) * np.sqrt(max(qm**2 - sigma**2, 0.0)))


def _growth_rate(quarters, t_total=T_TOTAL):
    """Exponential rate from the **last two** quarter-envelopes, past the seed's transient.

    The seed is not the growing Floquet mode, so the first quarter is contaminated by the decaying
    partner; the last two are clean. ``max|q|`` in quarter ``i`` lands at its end, so consecutive
    envelopes are ``e^(r t_total/4)`` apart.
    """
    return float(4.0 * np.log(quarters[3] / quarters[2]) / t_total)


def _whirl_run(*, kappa_w, amp, drive="u", seed="disp", t_total=T_TOTAL, n=N):
    """One run: drive one polarization in a plane, seed the other, watch it grow (or not).

    ``drive="u"`` drives the **soft** plane (``kappa_u = 0``) and seeds the stiff one; ``drive="w"``
    is the same string with the roles swapped, which is the ``delta < 0`` half of Gough's asymmetry.

    ``seed="disp"`` displaces the perturbed field (``dw = s A phi``, at rest); ``seed="vel"`` gives
    it a velocity instead (``dw' = s A wu phi``, from zero), which **injects angular momentum**.
    The difference is load-bearing on a degenerate string and is why both exist — see
    :func:`test_the_degenerate_string_is_marginal_not_exponential`.
    """
    s = make_geometric_string(N=n, kappa=0.0, kappa_w=kappa_w, EA=EA_DEFAULT, lam_long=LAM_LONG)
    phi = np.sin(MODE * np.pi * s.x / s.L)
    norm = float(np.dot(phi, phi))
    driven_ic = amp * phi

    if seed == "disp":
        seeded = {"w0" if drive == "u" else "u0": SEED_REL * amp * phi}
    else:
        wu = float(np.sqrt(_kc(n)[1]))
        seeded = {"w_dot" if drive == "u" else "u_dot": SEED_REL * amp * wu * phi}
    s.set_state(**{"u0" if drive == "u" else "w0": driven_ic}, **seeded)

    n_steps = int(t_total * s.fs)
    e0 = s.energy()
    e_lo = e_hi = e0
    q_perp = np.empty(n_steps)
    for i in range(n_steps):
        s.step()
        perp = s.w if drive == "u" else s.u
        q_perp[i] = np.dot(perp, phi) / norm
        e = s.energy()
        e_lo, e_hi = min(e_lo, e), max(e_hi, e)

    driven = s.u if drive == "u" else s.w
    modes = np.array([abs(np.dot(driven, np.sin(m * np.pi * s.x / s.L)) / norm) for m in (1, 3, 5)])
    seed_amp = SEED_REL * amp
    return dict(
        growth=float(np.abs(q_perp).max()) / seed_amp,
        saturation=float(np.abs(q_perp).max()) / amp,
        drift=(e_hi - e_lo) / abs(e0),
        n_not_converged=s.n_not_converged,
        off_mode=float(modes[1:].max() / modes[0]),
        quarters=_envelope_quarters(q_perp, seed_amp),
        amp=amp,
    )


def _envelope_quarters(q, seed_amp):
    """Peak ``|q|`` in each quarter of the run, in units of the seed.

    The shape of this 4-vector *is* the marginal-vs-exponential discriminator: secular growth
    (``q ~ c t``) gives ``1 : 2 : 3 : 4``, exponential gives a constant ratio between neighbours.
    Four numbers separate them without fitting anything.
    """
    n = len(q)
    return np.array([float(np.abs(q[i * n // 4:(i + 1) * n // 4]).max()) / seed_amp
                     for i in range(4)])


# -- fixtures: 15 runs, ~2.8 min, shared as hard as they can be ------------------------------------


@pytest.fixture(scope="module")
def tongue():
    """The Mathieu tongue: growth vs ``delta/(eps A^2)`` at fixed amplitude. Six runs, ~1 min."""
    amp = _amp_for(TENSION_RISE)
    return {f: _whirl_run(kappa_w=_kappa_w_at(f, amp), amp=amp) for f in TONGUE_FRACS}


@pytest.fixture(scope="module")
def threshold_moves():
    """Gough's threshold in the *amplitude* direction, at two detunings a factor 2 apart in
    ``delta`` — so ``A_c`` must move by exactly ``sqrt(2)``. Three runs, ~30 s."""
    amp_ref = _amp_for(TENSION_RISE)
    k1 = _kappa_w_at(0.25, amp_ref)   # delta_1
    k2 = _kappa_w_at(0.5, amp_ref)    # delta_2 = 2 delta_1  =>  A_c2 = sqrt(2) A_c1
    a1, a2 = 1.2 * _a_crit(k1), 1.2 * _a_crit(k2)
    return dict(
        k1=k1, k2=k2, a1=a1, a2=a2,
        below_k1=_whirl_run(kappa_w=k1, amp=a1),   # above A_c(k1) -> whirls
        below_k2=_whirl_run(kappa_w=k2, amp=a1),   # same A, now below A_c(k2) -> stable
        above_k2=_whirl_run(kappa_w=k2, amp=a2),   # sqrt(2) more amplitude -> whirls again
    )


@pytest.fixture(scope="module")
def angular_momentum_seed():
    """Degenerate vs detuned under the seed that actually tests marginality. Two runs, ~20 s."""
    amp = _amp_for(TENSION_RISE)
    return dict(
        degenerate=_whirl_run(kappa_w=0.0, amp=amp, seed="vel"),
        detuned=_whirl_run(kappa_w=_kappa_w_at(0.25, amp), amp=amp, seed="vel"),
    )


# -- test 12: the whirling threshold ---------------------------------------------------------------


def test_the_whirling_growth_maps_the_mathieu_tongue(tongue):
    """**The headline.** Whirling is not a thing that happens above an amplitude — it is a
    *parametric resonance*, and the growth traces the tongue the Mathieu equation predicts.

    Sweeping the detuning at fixed amplitude, the out-of-plane seed's growth over 0.06 s::

        delta/eps A^2   0.00    0.07    0.25    0.41    0.50    0.80
        growth          1.00x  14.7x   76.3x   37.4x    8.4x    1.63x

    Every feature is predicted with no free parameter. It is **unimodal**, it peaks at exactly the
    predicted ``0.25`` (where the pump's ``2 Om`` sits dead on twice the perturbed polarization's
    effective frequency), and it is dead outside ``(0, 0.5)``. A model that merely *had* an
    instability would not put its maximum there, and a growth that rose monotonically with detuning
    — the naive "more detuning, more whirl" reading — would be a different physics entirely.

    The ``0.00`` end is marginal, as the exact Floquet argument requires, **but it earns nothing
    here**: with a displacement seed a degenerate string is just a planar mode in a *rotated* plane
    (Tier A/1), so ``1.00x`` is rotational invariance restated, not evidence about marginality.
    :func:`test_the_degenerate_string_is_marginal_not_exponential` is where that is tested.

    The upper edge is **soft** — 8.4x growth is still visible at ``0.50``, because ``A_c`` comes
    from a leading-order Duffing ``Om`` and a quasi-static ``eps``. That is a Tier C residual and it
    is reported, not pinned: the assertion is that the tongue is dead by ``0.80``.
    """
    growth = {f: r["growth"] for f, r in tongue.items()}
    peak = max(growth, key=growth.__getitem__)
    assert peak == 0.25, (
        f"the tongue must peak at delta/(eps A^2) = 0.25, where the 2 Om pump sits on the "
        f"principal parametric resonance; measured peak at {peak} — {growth}"
    )
    assert growth[0.25] > 40.0, f"the tongue centre must whirl; got {growth[0.25]:.2f}x"
    assert growth[0.8] < 2.0, (
        f"past the tongue's upper edge the perturbation must not grow; got {growth[0.8]:.2f}x at "
        f"delta/(eps A^2) = 0.8 — if this whirls, the instability is not the parametric one"
    )
    assert growth[0.25] / growth[0.8] > 20.0

    # Unimodal, with both flanks resolved: the shape is the claim, not just the peak's height.
    assert growth[0.0] < growth[0.07] < growth[0.25], "the tongue's rising flank is not monotone"
    assert growth[0.25] > growth[0.41] > growth[0.5] > growth[0.8], "the falling flank is not"

    # It genuinely whirls rather than merely growing: the seed reaches a macroscopic fraction of the
    # driven amplitude, i.e. the (u, w) orbit opens out. (Measured 0.69 A at 0.1 s, 0.076 A here.)
    assert tongue[0.25]["saturation"] > 20.0 * SEED_REL


def test_the_growth_rate_is_the_mathieu_rate_the_plan_discarded_as_a_precession_rate(tongue):
    """The tongue's **rate profile**, not just its shape — and the resurrection of a formula this
    plan wrote down, called wrong, and threw away. It was wrong. It was also this.

    ``(Om/2) sqrt(qM^2 - sigma^2)`` predicts how fast the seed grows at *every* point of the tongue,
    with nothing fitted. Measured against the three unstable points::

        delta/eps A^2    0.07     0.25     0.41
        predicted       56.0/s   80.7/s   62.0/s
        measured        53.2/s   74.2/s   55.3/s
        ratio            0.951    0.920    0.892

    5-11 % low and **systematically** so, which is the signature of a leading-order oracle rather
    than a bug: ``eps`` is quasi-static, ``Om`` is first-order Duffing, and the measured rate is
    diluted by the seed's non-growing component. Tier C — the residual is reported, not pinned.

    **The finding.** At the tongue centre ``sigma = 0`` and the rate collapses to
    ``eps A^2/(8 Om)``. That is, to the substitution ``w0 -> Om``, the ``Om_prec = eps A^2/(8 w0)``
    the plan derived by naive averaging and correctly discarded: rotational symmetry forces
    ``ww = wu`` exactly, so a degenerate string has **no** precession rate to be the leading order
    of. The plan diagnosed its own error precisely — "the averaging drops the ``2w`` pump" — without
    noticing the consequence: the discarded quantity *is the pump's strength*, and the pump is the
    entire instability. So the number was never spurious, only misattributed. It does not describe a
    degenerate string precessing (nothing does); it describes how fast a **detuned** one whirls.
    Measured, ``eps A^2/(8 w0) = 117.6/s`` against the true ``eps A^2/(8 Om) = 80.7/s`` — so the
    ``w0 -> Om`` hardening is 46 % of it and not a detail, which is presumably why averaging at
    ``w0`` looked plausible enough to write down.
    """
    amp = _amp_for(TENSION_RISE)
    for frac in (0.07, 0.25, 0.41):
        pred = _mathieu_rate(_kappa_w_at(frac, amp), amp)
        meas = _growth_rate(tongue[frac]["quarters"])
        assert meas / pred == pytest.approx(1.0, abs=0.2), (
            f"at delta/(eps A^2) = {frac} the seed grows at {meas:.1f}/s against the Mathieu "
            f"prediction {pred:.1f}/s ({meas/pred:.3f}x): the tongue is in the right place but "
            f"growing at the wrong speed, so the mechanism is probably not the 2 Om pump"
        )
        assert meas < pred, (
            f"the measured rate ({meas:.1f}/s) exceeds the leading-order prediction "
            f"({pred:.1f}/s) at {frac} — the residual has changed sign, so this is no longer the "
            f"dilution-plus-leading-order story the tolerance above is calibrated for"
        )

    # The centre's rate IS the plan's discarded precession rate, at the hardened Om.
    _, w0sq, eps = _kc()
    ea2 = eps * _amp_for(TENSION_RISE) ** 2
    om = np.sqrt(w0sq + 0.75 * ea2)
    assert _mathieu_rate(_kappa_w_at(0.25, amp), amp) == pytest.approx(ea2 / (8 * om), rel=1e-9)


def test_the_whirl_conserves_energy_and_stays_converged(tongue):
    """**The gate that makes the tongue mean anything**, and the family's contract at the one place
    it feels backwards: something grows by 76x and the energy is *flat to 1e-12*.

    A parametric instability is energy **redistribution**, ``u -> w``, not energy creation. The
    lossless model therefore conserves straight through the blow-up, and that is precisely how a
    physical whirl is told apart from numerical divergence — which is the other thing that makes
    ``max|w|`` grow by orders of magnitude. Without this assertion the whole file could false-pass
    on a diverging solve. Measured: drift ~1e-12 at **every** point of the sweep, including the
    centre, and Newton never fails to converge.

    Energy alone is not sufficient, though, because model #9's **in-plane** modal-exchange
    instability (``dT/T0 ~ 3``) conserves too, and would also dump mode-1 energy somewhere. So the
    driven field is checked to stay single-mode: measured, 0.09 % off-mode content, flat across the
    sweep — which is the Duffing self-interaction, not an exchange. See ``TENSION_RISE``.
    """
    for frac, r in tongue.items():
        assert r["drift"] < 1e-10, (
            f"at delta/(eps A^2) = {frac} the energy drifted by {r['drift']:.2e}: a parametric "
            f"instability redistributes energy, it does not create it — this is divergence, and "
            f"the {r['growth']:.1f}x 'whirl' at this point means nothing"
        )
        assert r["n_not_converged"] == 0, f"non-converged solve at {frac}; its trajectory is void"
        assert r["off_mode"] < 0.01, (
            f"at delta/(eps A^2) = {frac} the driven field is {100*r['off_mode']:.2f} % off-mode: "
            f"the energy may be leaving mode 1 in-plane (model #9's instability) rather than "
            f"whirling out of plane — lower TENSION_RISE"
        )


def test_whirling_must_be_seeded_and_never_leaks_from_a_planar_start(tongue):
    """**The honesty gate.** At the tongue centre, at 35 mm, with a 17 Hz detuning — the most
    unstable configuration in this file — an **unseeded** planar run stays planar ``== 0.0``,
    bit-exactly.

    This is what licenses reading every growth ratio in the file as a growth ratio. ``w -> -w`` is a
    reflection symmetry of the detuned model too (the knob splits only the *linear* operator), so
    the planar subspace is invariant and an identically-zero ``w`` can never acquire a nonzero bit.
    If it could, the tongue's 76x would be partly a measurement of that leak, and the ``1.63x``
    outside the tongue would be a floor rather than a null.

    ``test_geometric_polarization.py`` makes this claim structurally; this is the same claim at the
    parameters that would actually break it, which is the version worth having here.
    """
    amp = _amp_for(TENSION_RISE)
    s = make_geometric_string(
        N=N, kappa=0.0, kappa_w=_kappa_w_at(0.25, amp), EA=EA_DEFAULT, lam_long=LAM_LONG
    )
    assert not s.is_degenerate
    s.set_state(amp * np.sin(MODE * np.pi * s.x / s.L))
    w_max = u_max = 0.0
    for _ in range(int(T_TOTAL * s.fs)):
        s.step()
        # Tracked, never sampled at the last step: a mode-1 string passes through u ~ 0 twice a
        # period, so the final state is at an arbitrary phase (measured 0.42 A) and "is it still
        # ringing?" asked there is a coin flip. The same trap `mode_off_fraction` warns about.
        w_max = max(w_max, float(np.abs(s.w).max()))
        u_max = max(u_max, float(np.abs(s.u).max()))

    assert w_max == 0.0, (
        f"an unseeded planar run leaked {w_max:.3e} into w at the tongue centre — every growth "
        f"ratio in this file is then partly measuring the leak"
    )
    assert u_max > 0.9 * amp, "the driven plane must ring at full amplitude throughout"
    # ...and the seeded run at the identical parameters whirls. Same string, one seed apart.
    assert tongue[0.25]["growth"] > 40.0


def test_only_the_plane_of_the_lower_mode_whirls(tongue):
    """**Gough's asymmetry, and the sharpest statement in the file**: same string, same amplitude,
    same seed — drive the *soft* polarization and it whirls 76x; drive the *stiff* one and the seed
    does not move. Not "grows less": **1.00x**.

    The tongue is ``0 < delta < eps A^2/2`` with ``delta = ww^2 - wu^2`` *measured from the driven
    plane*, so driving the stiff polarization flips the sign of ``delta`` and puts the string
    outside the tongue at **every** amplitude — there is no threshold to cross. This is why a real
    string bowed or plucked in the plane of its lower mode whirls while the other plane does not,
    and it is a claim no energy, spectrum, or amplitude measurement can make: the two runs are the
    same string at the same energy, distinguished only by *which* polarization carries the motion.

    It is also a free test of the ``u``/``w`` symmetry of the implementation — the two runs traverse
    different code paths through :meth:`set_state` and the DG, and must disagree *only* through the
    linear operator's ``kappa``.
    """
    amp = _amp_for(TENSION_RISE)
    stiff = _whirl_run(kappa_w=_kappa_w_at(0.25, amp), amp=amp, drive="w")
    soft = tongue[0.25]

    assert soft["growth"] > 40.0
    assert stiff["growth"] < 1.5, (
        f"driving the stiff plane put delta < 0, which is outside the tongue at every amplitude — "
        f"it must not whirl; got {stiff['growth']:.2f}x"
    )
    assert soft["growth"] / stiff["growth"] > 30.0, (
        f"the soft and stiff planes of the same string whirl comparably ({soft['growth']:.1f}x vs "
        f"{stiff['growth']:.1f}x) — Gough's asymmetry is not holding"
    )
    assert stiff["drift"] < 1e-10 and stiff["n_not_converged"] == 0


def test_the_threshold_moves_as_the_square_root_of_the_detuning(threshold_moves):
    """**Gough's threshold, read in the amplitude direction** — ``A_c = sqrt(2 delta / eps)``, so
    doubling ``delta`` moves it by ``sqrt(2)``. Three runs pin that it *moves*, and that it moves by
    the right factor:

    * ``A1`` at ``delta_1`` — above ``A_c(delta_1)``, **whirls** (25.9x).
    * ``A1`` at ``delta_2 = 2 delta_1`` — the same amplitude, now **below** the moved threshold:
      **stable** (1.88x). The threshold really moved; the amplitude did not.
    * ``A2 = sqrt(2) A1`` at ``delta_2`` — **whirls again** (215x). Restoring the *dimensionless*
      position recovers the instability, which is the ``sqrt(delta)`` law rather than merely a
      monotone trend.

    The third run is what makes this a scaling law and not a stability boundary sighted twice: run 2
    alone is consistent with "more detuning is more stable", which is false — run 3 is the same
    detuning as run 2 and unstable. Only ``delta/(eps A^2)`` predicts all three.

    Both ``delta_2/delta_1 = 2`` and ``A_c2/A_c1 = sqrt(2)`` are exact **by construction** here
    (``delta ~ kappa_w^2``), so the assertion on them is a guard on the sweep's own arithmetic; the
    physics is in the three growths.
    """
    tm = threshold_moves
    assert _delta(tm["k2"]) / _delta(tm["k1"]) == pytest.approx(2.0, rel=1e-9)
    assert _a_crit(tm["k2"]) / _a_crit(tm["k1"]) == pytest.approx(np.sqrt(2.0), rel=1e-9)

    assert tm["below_k1"]["growth"] > 10.0, (
        f"A1 sits 1.2x above A_c(delta_1) and must whirl; got {tm['below_k1']['growth']:.2f}x"
    )
    assert tm["below_k2"]["growth"] < 2.5, (
        f"at delta_2 the threshold has moved up to sqrt(2) A_c(delta_1) > A1, so the SAME "
        f"amplitude must now be stable; got {tm['below_k2']['growth']:.2f}x — the threshold is "
        f"not moving with the detuning"
    )
    assert tm["above_k2"]["growth"] > 10.0, (
        f"sqrt(2) more amplitude restores delta/(eps A^2) and must whirl again; got "
        f"{tm['above_k2']['growth']:.2f}x — the threshold moves, but not as sqrt(delta)"
    )
    assert tm["below_k1"]["growth"] / tm["below_k2"]["growth"] > 5.0
    assert tm["above_k2"]["growth"] / tm["below_k2"]["growth"] > 5.0
    for r in (tm["below_k1"], tm["below_k2"], tm["above_k2"]):
        assert r["drift"] < 1e-10 and r["n_not_converged"] == 0
        assert r["off_mode"] < 0.01, "the sqrt(delta) test reaches dT/T0 = 2.16 — check mode purity"


def test_the_degenerate_string_is_marginal_not_exponential(angular_momentum_seed):
    """**The negative control**, and the one that needs the *velocity* seed to say anything.

    A degenerate string's out-of-plane perturbation is marginal — both Floquet multipliers ``+1`` —
    which is a strictly weaker statement than "does not grow": the second solution grows
    **secularly**, linearly in ``t``, at a rate set by the injected angular momentum. Seeding with a
    *displacement* misses this entirely, because ``dw = s A phi`` at rest is exactly the rotation
    generator: the string just oscillates planar in a rotated plane and the ratio pins at 1.00x
    forever. That is Tier A/1, not marginality. Seeding with a *velocity* injects angular momentum
    and excites the secular solution, which is the thing to measure.

    The discriminator is the envelope's **shape** over four quarters of the run, no fit required —
    ``q ~ c t`` gives ``1 : 2 : 3 : 4``, exponential gives a constant ratio. Measured::

        degenerate  [1.88 3.60 5.79 7.55]  ->  1 : 1.91 : 3.08 : 4.01   (secular: 1:2:3:4 to 1.3 %)
        detuned     [1.39 5.27 17.1 52.3]  ->  1 : 3.80 : 12.3 : 37.6   (exponential)

    Same amplitude, same seed, same everything — one knob. The degenerate string reaches 7.6x and
    the detuned one 52x, but the *ratio* is the weaker half of this: 7.6x would look like growth on
    its own. The shape is what says one is marginal and the other is not, and it is what would catch
    a slow exponential masquerading as secular.
    """
    deg = angular_momentum_seed["degenerate"]
    det = angular_momentum_seed["detuned"]
    q_deg = deg["quarters"] / deg["quarters"][0]
    q_det = det["quarters"] / det["quarters"][0]

    # Secular: the envelope is linear in t, so the quarters are 1 : 2 : 3 : 4.
    assert q_deg == pytest.approx(np.array([1.0, 2.0, 3.0, 4.0]), rel=0.12), (
        f"a degenerate string's perturbation must grow SECULARLY (both Floquet multipliers +1): "
        f"the envelope should be linear in t (1:2:3:4), measured {np.round(q_deg, 3)}"
    )
    # Exponential: constant ratio between neighbouring quarters, and far past what linear reaches.
    assert q_det[-1] > 5.0 * q_deg[-1], (
        f"the detuned string must grow exponentially past the degenerate one's secular envelope: "
        f"{q_det[-1]:.1f}x vs {q_deg[-1]:.1f}x"
    )
    ratios = q_det[1:] / q_det[:-1]
    assert ratios.min() > 2.0, (
        f"the detuned envelope's quarter-on-quarter ratios {np.round(ratios, 2)} are not those of "
        f"an exponential — this control only means something against a real exponential"
    )
    assert deg["drift"] < 1e-10 and det["drift"] < 1e-10
    assert deg["n_not_converged"] == 0 and det["n_not_converged"] == 0


def test_the_tongue_does_not_move_with_the_grid():
    """**Physics, not artifact** — the standard model #9's parametric discovery had to meet, and a
    whirl claim has to meet it too: an instability that lives on the mesh is a bug.

    The tongue is a *dimensionless* statement, so doubling ``N`` at the same ``delta/(eps A^2)``
    must reproduce it. Measured at ``N = 32`` (t = 0.05 s): **29.5x** at ``0.25`` and **1.61x** at
    ``0.80`` — whirls and doesn't, in the same places. The tongue's centre sits at
    ``kappa_w = 39.05 / 39.01 / 39.00`` for ``N = 16 / 24 / 32``: it moves by 0.13 % while the grid
    halves, converging as the discrete ``p^2`` does.

    The growth *rate* is mildly ``N``-dependent (76x vs 59x at t = 0.06 s, i.e. ~6 % in the
    exponent) because ``eps`` is a continuum coefficient and the discrete quartic is not exactly it.
    That is an O(h^2) residual on the rate, not a shift of the tongue — the position is the claim,
    which is why this test asserts where the tongue *is* and not how fast it grows there.
    """
    n_fine = 2 * N
    amp = _amp_for(TENSION_RISE, n_fine)
    inside = _whirl_run(kappa_w=_kappa_w_at(0.25, amp, n_fine), amp=amp, t_total=0.05, n=n_fine)
    outside = _whirl_run(kappa_w=_kappa_w_at(0.8, amp, n_fine), amp=amp, t_total=0.05, n=n_fine)

    assert inside["growth"] > 10.0, (
        f"the tongue centre must still whirl at N = {n_fine}; got {inside['growth']:.2f}x — an "
        f"instability that weakens under refinement is a numerical artifact, not whirling"
    )
    assert outside["growth"] < 2.0, (
        f"outside the tongue must still be stable at N = {n_fine}; got {outside['growth']:.2f}x"
    )
    assert inside["growth"] / outside["growth"] > 5.0
    for r in (inside, outside):
        assert r["drift"] < 1e-10 and r["n_not_converged"] == 0
