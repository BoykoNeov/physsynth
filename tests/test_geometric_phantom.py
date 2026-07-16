"""Model #10 — **phantom partials** (test 10) and the **polarization discriminator** (test 11).

Batch 2's headline, and the reason model #10 exists: these are the two claims model #9 refuses.
Model #9's tension is a spatial *scalar*, so it has no longitudinal field and nowhere to put a
combination tone. Model #10's tension is a **field**, and its nonlinear excess carries the term KC
structurally lacks::

    V_nl = (EA - T0) [ r^2 v_x / 2  +  r^4 / 8  + ... ]
                       ~~~~~~~~~~~    ~~~~~~~
                       PHANTOMS       model #9's KC quartic, recovered *locally*

``r^2 v_x / 2`` is **quadratic in the transverse fields and linear in the longitudinal one**, so two
transverse partials at ``f_i``, ``f_j`` drive ``v`` at ``f_i ± f_j``. Those are Conklin's (1999)
phantom partials. ``test_geometric_limits.py`` measures where model #10 *agrees* with model #9; this
file measures the physics that makes them different models.

**Readout: the bridge force ``EA v_x(0)``** — what actually radiates in a real piano, and the honest
place to look. Below the first longitudinal resonance the ``v`` response is quasi-static
(``-EA v_xx ~ -(a/2) (r^2)_x``), so ``EA v_x`` carries ``r^2`` almost directly and the combination
tones are in it by construction rather than by luck.

**Why the free longitudinal modes do not contaminate any of this.** ``v = 0`` is *not* the
longitudinal equilibrium (a held string has already settled), so every run here radiates a
longitudinal transient — measured, it is the **largest** feature in the bridge spectrum. It is also
entirely out of the way: the free longitudinal modes sit at ``n c_long/(2L) = n * 2236 Hz`` while
the phantoms live below 500 Hz, ~4.5x lower. So the phantom band is *purely forced response*, which
is both the piano physics (phantoms are forced, not free longitudinal modes) and the reason a band
limit at the first longitudinal mode is a physical cut and not a fudge. Every peak search here is
band-limited for that reason.

**The tests here run for ~2.5 minutes**, essentially all of it in the three shared fixtures. Model
#10's timestep is set by the *longitudinal* wave
(``lam_long``, ~22x faster than the transverse one), so ~0.1 s of audio costs ~16k Newton solves.
The window is not padding, and halving it does not merely cost precision: at 0.05 s the raw bins are
20 Hz wide, the ``2 f1`` phantom (the weakest of the four) sits inside its neighbours' leakage
skirts, and it is mislocated by **0.52 Hz** — against 0.013-0.16 Hz for the other three, and 0.039
Hz for the worst of the four at 0.1 s. Its margin to ``f2`` collapses from 170x to 8x with it.
"""

import numpy as np
import pytest
from helpers import (
    EA_DEFAULT,
    T_DEFAULT,
    geometric_mode_ic,
    make_geometric_string,
)

from physsynth.analysis.damping import spatial_eigenvalue_p2
from physsynth.analysis.dispersion import stiff_dispersion_frequencies
from physsynth.analysis.duffing import kc_mode_coefficients
from physsynth.analysis.spectrum import detect_peaks, magnitude_spectrum

# -- parameters, and why each is what it is --------------------------------------------------------

N = 32
LAM_LONG = 0.9
T_TOTAL = 0.10  # seconds of audio; see the module docstring on why not 0.05
AMP = 1.5e-3

KAPPA_PHANTOM = 8.0
"""Stiffness for the phantom tests — **4x the piano-ish default**, and the choice is load-bearing.

The Conklin signature needs the transverse partials *stretched*: a phantom is discriminating only
because it lands where no partial is. On a harmonic string (``B = 0``) every phantom coincides with
a partial *exactly* — ``f2 - f1 = f1``, ``2 f1 = f2``, ``f1 + f2 = f3`` — so the test would have
nothing to see. Inharmonicity ``B = pi^2 kappa^2/(c^2 L^2)`` is what opens the gaps, and they open
in proportion to it: ``f3 - (f1 + f2) ~ 9 B f1``.

**This exaggerates the *contrast*, not the *effect*.** The phantom mechanism is
``r^2 = u_x^2 + w_x^2`` pumping ``v`` — completely ``kappa``-independent. ``kappa`` only decides
*where the transverse partials sit*, i.e. whether the gap a phantom lands in is visible. Turning it
up is a microscope, not a thumb on the scale.

At the ``kappa = 2`` default the gap is **0.89 Hz**, and that is not merely too small to resolve —
it makes the test **wrong**. The partials ``f1``, ``f2`` are measured from the phantom run, so they
are the *hardened* ones, and hardening drives both **up**: the phantom at ``f1 + f2`` climbs by
``d1 + d2``, measured here at **1.29 Hz**. That exceeds the 0.89 Hz gap, so at ``kappa = 2`` the
phantom does not just blur into ``f3`` — it **crosses** it, and the test would confidently report a
phantom landing *on* a transverse partial. No run length fixes a physical confound. At
``kappa = 8`` the gap is 11.4 Hz and the same 1.29 Hz of hardening is 11 % of it.
"""


def _bridge_force(s):
    """The longitudinal end force ``EA v_x(0) = EA v[1]/h`` (N) — the piano's radiating channel."""
    return s.EA * s.v[1] / (s.L / s.N)


def _two_mode_run(amp):
    """Excite transverse modes 1 **and** 2; record the bridge force and both modal amplitudes.

    Two modes, not one: a single mode gives only ``2 f1``, and the *sum and difference* tones are
    the whole point — they are what cannot be mistaken for a harmonic of anything.
    """
    s = make_geometric_string(N=N, kappa=KAPPA_PHANTOM, EA=EA_DEFAULT, lam_long=LAM_LONG)
    sin1, sin2 = np.sin(np.pi * s.x / s.L), np.sin(2 * np.pi * s.x / s.L)
    s.set_state(amp * (sin1 + sin2))

    n_steps = int(T_TOTAL * s.fs)
    d1, d2 = float(np.dot(sin1, sin1)), float(np.dot(sin2, sin2))
    bridge = np.empty(n_steps)
    q1, q2 = np.empty(n_steps), np.empty(n_steps)
    for i in range(n_steps):
        s.step()
        bridge[i] = _bridge_force(s)
        q1[i] = np.dot(s.u, sin1) / d1
        q2[i] = np.dot(s.u, sin2) / d2

    assert s.converged and s.n_not_converged == 0, "a non-converged run's spectrum means nothing"
    # Modal projections, not a point probe: they isolate f1 and f2 from each other, so each is a
    # single clean peak and the parabolic refinement is not fighting a neighbour's skirt.
    return dict(
        s=s,
        bridge=bridge,
        f1=float(detect_peaks(q1, s.fs, 1, f_min=10.0)[0]),
        f2=float(detect_peaks(q2, s.fs, 1, f_min=10.0)[0]),
        f_long1=s.c_long / (2 * s.L),
        ladder=stiff_dispersion_frequencies(
            s.c, s.L, s.N, s.kappa_u, s.k, s.theta, np.arange(1, 6)
        ),
    )


@pytest.fixture(scope="module")
def phantom():
    """The phantom run, shared by every test that reads it (each run is ~30 s)."""
    return _two_mode_run(AMP)


@pytest.fixture(scope="module")
def linear():
    """The same string at ``amp -> 0``. Earns two things the headline test cannot assert without it:
    that the ladder oracle is right at this ``kappa``/``N``, and *how far* hardening moved the
    partials at ``AMP``."""
    return _two_mode_run(1e-6)


def _combinations(run):
    """The four quadratic combination tones of the **measured** transverse partials.

    Measured, never predicted: ``f1`` and ``f2`` carry the theta-scheme's temporal dispersion *and*
    the hardening shift, and the phantom rides on whatever the partials actually are. Predicting
    them from ``n c/(2L) sqrt(1 + B n^2)`` would put both errors into the oracle and measure the
    formula instead of the string.
    """
    f1, f2 = run["f1"], run["f2"]
    return {"f2-f1": f2 - f1, "2f1": 2 * f1, "f1+f2": f1 + f2, "2f2": 2 * f2}


def _in_band_peaks(run):
    """Blindly detected peaks below the first free longitudinal mode, and their magnitudes.

    ``detect_peaks``, not ``measure_partials_near``: anchoring a search window on the frequency
    under test is how a spectrum test passes by construction. Nothing here tells the detector where
    a phantom should be.
    """
    bridge, fs = run["bridge"], run["s"].fs
    peaks = detect_peaks(bridge, fs, 40, f_min=10.0)
    peaks = peaks[peaks < 0.9 * run["f_long1"]]
    freqs, mag, _ = magnitude_spectrum(bridge, fs)
    mags = np.array([mag[int(np.argmin(np.abs(freqs - p)))] for p in peaks])
    return peaks, mags


# -- test 10: the phantom oracle -------------------------------------------------------------------


def test_longitudinal_peaks_are_quadratic_combinations_of_the_transverse_partials(phantom):
    """**The mechanism.** Every peak the longitudinal field carries is a *quadratic* combination of
    the transverse partials — and the partials themselves are absent from it.

    This is the ``r^2 v_x/2`` term made audible. ``r^2`` is quadratic in ``u``, so a transverse
    spectrum ``{f1, f2}`` drives ``v`` at ``{f2-f1, 2f1, f1+f2, 2f2}`` and at nothing else. The
    absence of ``f1`` and ``f2`` from the longitudinal spectrum is half the claim and the more
    discriminating half: a *linear* coupling between the fields — the bug this excludes — would
    reproduce the transverse partials in ``v`` directly.

    Measured: the four strongest in-band peaks are exactly the four combinations, to **0.039 Hz**,
    with the strongest non-combination peak **5.4x** weaker. (That runner-up, and its friends near
    130 and 389 Hz, are the broadband ``v = 0`` transient, not physics.)
    """
    combos = _combinations(phantom)
    peaks, mags = _in_band_peaks(phantom)
    assert len(peaks) >= 4

    strongest = np.sort(peaks[np.argsort(mags)[::-1][:4]])
    expected = np.sort(np.array(list(combos.values())))
    err = np.abs(strongest - expected)
    assert err.max() < 0.15, (
        f"the 4 strongest in-band peaks {np.round(strongest, 3)} are not the 4 quadratic "
        f"combinations {np.round(expected, 3)} (max err {err.max():.4f} Hz)"
    )

    is_combo = np.array([any(abs(p - v) < 1.0 for v in combos.values()) for p in peaks])
    assert mags[is_combo].min() > 3.0 * mags[~is_combo].max(), (
        f"the combinations do not dominate the spectrum: weakest combo {mags[is_combo].min():.3e} "
        f"vs strongest non-combo {mags[~is_combo].max():.3e}"
    )

    # The other half: no longitudinal peak sits on a transverse partial. `f1` and `f2` are the only
    # partials actually excited, so these are the only two that can be checked without an oracle.
    for name, f_partial in (("f1", phantom["f1"]), ("f2", phantom["f2"])):
        nearest = peaks[int(np.argmin(np.abs(peaks - f_partial)))]
        assert abs(nearest - f_partial) > 3.0, (
            f"a longitudinal peak sits on the transverse partial {name} = {f_partial:.3f}: "
            f"nearest peak {nearest:.3f}. The coupling is supposed to be quadratic."
        )


def test_phantoms_are_displaced_from_the_partials_by_the_inharmonicity_defect(phantom):
    """**The Conklin signature — the primary form, with no oracle and no confound.** The
    displacement that puts a phantom in a *gap* is ``f2 - 2 f1``: a single number, measured in this
    one run.

    The plan states the signature as ``f1 + f2`` sitting ``~9 B f1`` below ``f3``, which needs an
    oracle for ``f3`` (mode 3 is not excited) and inherits the hardening confound with it. That
    version is worth having and is the test below — but as the *secondary*. This one is strictly
    better: it needs neither, and it is hiding in the same run.

    For a **harmonic** string the low phantoms coincide with partials *exactly*: ``f2 = 2 f1`` gives
    ``f2 - f1 = f1`` and ``2 f1 = f2``. So the distance from the difference tone to ``f1``, and the
    distance from ``2 f1`` to ``f2``, are **both** exactly ``|f2 - 2 f1|`` — the inharmonicity
    defect, nonzero only because ``kappa > 0``. Every term is measured here, at one amplitude, in
    one run: no ladder, no continuum formula, and **no hardening confound** — hardening cannot
    smuggle in a false positive because it moves the phantoms and the partials *together*. Measured,
    it slightly *widens* the defect (4.416 -> 4.574 Hz), so it works against the claim, not for it.

    Measured: defect **4.57 Hz**, both displacements **118x** the 0.039 Hz peak-location error.
    """
    f1, f2 = phantom["f1"], phantom["f2"]
    defect = f2 - 2 * f1
    assert defect > 3.0, f"kappa > 0 must stretch the partials: f2 - 2 f1 = {defect:.4f} Hz"

    peaks, _ = _in_band_peaks(phantom)
    combos = _combinations(phantom)
    peak_diff = peaks[int(np.argmin(np.abs(peaks - combos["f2-f1"])))]
    peak_2f1 = peaks[int(np.argmin(np.abs(peaks - combos["2f1"])))]

    # Both displacements are the same physical number, approached from opposite sides.
    assert abs(peak_diff - f1) == pytest.approx(defect, abs=0.3)
    assert abs(peak_2f1 - f2) == pytest.approx(defect, abs=0.3)
    assert min(abs(peak_diff - f1), abs(peak_2f1 - f2)) > 3.0, (
        "the phantoms must land in the gaps, not on the partials"
    )


def test_the_phantom_lands_below_where_the_third_partial_would_be(phantom, linear):
    """**The plan's headline, kept as the secondary form** (Conklin 1999): ``f1 + f2`` lands
    ``~9 B f1`` **below** ``f3``, where no transverse partial exists. Mode 3 is not excited, so this
    one needs an oracle — and the oracle has to be earned, which is what ``linear`` is for. The
    test above makes the same point without either, which is why this is the second statement of it
    and not the first.

    Two things make it honest rather than a formula-vs-formula check:

    * **The ladder is the discrete oracle**, not ``n f1 sqrt(1 + B n^2)``. It carries the scheme's
      own spatial eigenvalue and theta-scheme dispersion, so ``f3`` is where *this string on this
      grid* puts it. Earned, not assumed: at ``amp -> 0`` the same measurement lands on it to
      **1.5e-4** relative, which is what licenses using it for the ``f3`` that is not excited.
    * **The hardening is measured, not waved at.** ``f1`` and ``f2`` are the hardened partials, so
      ``f1 + f2`` has climbed by ``d1 + d2 = 1.29 Hz`` against a *linear* ladder. That is the whole
      confound, it is 11 % of the 11.4 Hz gap, and it is asserted below rather than hoped about.
      (It is also pessimistic: an excited mode 3 would harden *too*, and in the same direction.)

    Measured: gap **11.41 Hz** = 295x the peak-location error, against 1.29 Hz of hardening.
    """
    f1, f2 = phantom["f1"], phantom["f2"]
    f3 = float(phantom["ladder"][2])

    # The oracle earns its keep: at amp -> 0 the measurement lands on the ladder.
    for i, name in ((0, "f1"), (1, "f2")):
        assert linear[f"f{i + 1}"] == pytest.approx(float(linear["ladder"][i]), rel=1e-3), (
            f"the discrete ladder oracle is wrong at kappa={KAPPA_PHANTOM}, N={N} for {name} — "
            f"so it cannot be trusted for the f3 that is not excited"
        )

    hardening = (f1 - linear["f1"]) + (f2 - linear["f2"])
    gap = f3 - (f1 + f2)
    assert gap > 0.0, f"the phantom must land BELOW f3; gap = {gap:.3f} Hz"
    assert hardening < 0.4 * gap, (
        f"hardening ({hardening:.3f} Hz) is no longer small against the gap ({gap:.3f} Hz): at "
        f"this amplitude the phantom could be crossing f3 rather than sitting below it"
    )

    peaks, _ = _in_band_peaks(phantom)
    peak_sum = peaks[int(np.argmin(np.abs(peaks - (f1 + f2))))]
    assert peak_sum == pytest.approx(f1 + f2, abs=0.15)
    assert abs(peak_sum - f3) > 5.0, (
        f"the phantom at {peak_sum:.3f} Hz is supposed to land where no partial is, but f3 = "
        f"{f3:.3f} Hz is only {abs(peak_sum - f3):.3f} Hz away"
    )


def test_a_linear_string_has_no_longitudinal_motion_to_put_a_phantom_in():
    """The harness control: at ``EA = T0`` the phantom channel **does not exist**, bit-exactly.

    ``a = EA - T0 = 0`` kills the whole nonlinear excess, so the three fields decouple and ``v``,
    started at rest, never leaves it. ``v == 0.0`` *identically* — not "small".

    This is what rules out the readout manufacturing its own result. If ``_bridge_force`` indexed
    the wrong field, or the peak detector invented structure out of a windowed transient, this run
    would show it: the same IC, the same readout, the same detector, and the only change is the one
    coefficient the phantom is supposed to come from. It is also why the phantom tests can trust a
    *blind* detector — the detector has been shown to report nothing when there is nothing.
    """
    s = make_geometric_string(N=N, kappa=KAPPA_PHANTOM, EA=T_DEFAULT, lam_long=LAM_LONG)
    sin1, sin2 = np.sin(np.pi * s.x / s.L), np.sin(2 * np.pi * s.x / s.L)
    s.set_state(AMP * (sin1 + sin2))
    assert s._a == 0.0

    bridge = np.array([(s.step(), _bridge_force(s))[1] for _ in range(int(T_TOTAL * s.fs))])
    assert np.max(np.abs(s.u)) > 0.0, "the transverse field must actually be moving"
    assert np.all(bridge == 0.0), (
        f"EA = T0 must leave v identically zero; got max |EA v_x(0)| = {np.abs(bridge).max():.3e}"
    )


# -- test 11 (Tier A/3): the polarization discriminator --------------------------------------------


def _circular_omega(s, amp, mode=1):
    """The angular frequency of the KC **circular relative equilibrium** at amplitude ``amp``.

    A circular mode holds ``r^2`` *static*, so the tension rise needs no time-averaging and the KC
    reduction gives ``Omega^2 = omega0^2 + eps A^2`` exactly — with ``eps`` from the existing model
    #9 oracle under the ``EA -> a`` identification.

    Used to *construct* an IC, never asserted: whether that formula is exact for model #10 is Tier
    C/8, which is batch 3's business. All that is claimed here is that it is a better circle than
    the linear ``Omega``, and the test below measures how much better.
    """
    p2 = spatial_eigenvalue_p2(s.N, s.L / s.N, mode)
    w0sq, eps = kc_mode_coefficients(c=s.c, kappa=s.kappa_u, EA=s._a, rho=s.rho, p2=p2, L=s.L)
    return float(np.sqrt(w0sq + eps * amp**2))


def _polarization_run(omega=None, amp=AMP):
    """One single-mode run. ``omega=None`` is planar; otherwise circular at that ``Omega``.

    Circular means ``u = A phi cos(Omega t)``, ``w = A phi sin(Omega t)``, i.e. ``w_dot = Omega A
    phi`` at ``t = 0`` — batch 1's convention, unchanged.

    Reports the bridge force's **AC at ``2 f1``** (the pump) and its **DC** (the static stretch)
    separately, because the whole discriminator is that polarization moves those two in opposite
    directions.
    """
    s = make_geometric_string(N=N, kappa=KAPPA_PHANTOM, EA=EA_DEFAULT, lam_long=LAM_LONG)
    shape = geometric_mode_ic(N, 1, amp)
    f1 = float(stiff_dispersion_frequencies(
        s.c, s.L, s.N, s.kappa_u, s.k, s.theta, np.array([1])
    )[0])
    if omega is None:
        s.set_state(shape)
    else:
        s.set_state(shape, 0.0, w_dot=omega * shape)

    n_steps = int(T_TOTAL * s.fs)
    bridge = np.empty(n_steps)
    w_max = 0.0
    for i in range(n_steps):
        s.step()
        bridge[i] = _bridge_force(s)
        w_max = max(w_max, float(np.abs(s.w).max()))
    assert s.n_not_converged == 0

    freqs, mag, _ = magnitude_spectrum(bridge, s.fs)
    i2 = int(np.argmin(np.abs(freqs - 2 * f1)))
    return dict(
        pump=float(mag[max(1, i2 - 2) : i2 + 3].max()),
        dc=float(np.mean(bridge)),
        # Tracked across the run, never sampled at the last step: `w = A phi sin(Omega t)` is at a
        # essentially random phase when the loop happens to stop.
        w_max=w_max,
        f1=f1,
        s=s,
    )


@pytest.fixture(scope="module")
def polarization():
    """Planar vs circular, same string, same mode, same amplitude — the three runs test 11 reads."""
    planar = _polarization_run()
    naive = _polarization_run(omega=2 * np.pi * planar["f1"])
    tuned = _polarization_run(omega=_circular_omega(planar["s"], AMP))
    return dict(planar=planar, naive=naive, tuned=tuned)


def test_a_circular_mode_does_not_pump_the_longitudinal_field(polarization):
    """**Tier A/3, the discriminator.** Same string, same mode, same amplitude — and the
    longitudinal spectrum five orders of magnitude apart, decided by **polarization alone**.

    The cleanest statement of what it means for the tension to be a **field**. A planar mode has
    ``r^2 = A^2 phi'^2 cos^2(Omega t)``, which oscillates at ``2 Omega`` and pumps ``v`` — that is
    the phantom. A circular mode ``u = A phi cos(Omega t)``, ``w = A phi sin(Omega t)`` has
    ``r^2 = A^2 phi'^2`` **time-independent**: identical nonlinearity, but the longitudinal forcing
    is *static* and there is nothing to radiate. Model #9 cannot say this in either direction — no
    ``v`` to pump, and only one polarization to pump it with.

    **Why this null is not just a quiet string**, which is the only way it could cheat — and note
    the circular run is emphatically *not* the quieter of the two. At equal amplitude it carries
    **twice the energy** (both polarizations at full amplitude, not one), and the *static* stretch
    says the nonlinearity is engaged **harder**: a planar ``r^2`` averages to ``A^2 phi'^2 / 2``
    while a circular one *is* ``A^2 phi'^2``, so the DC bridge force must come out at exactly **2x**
    the planar one. Measured **1.987x**. So the circular string is twice as energetic and twice as
    stretched, and radiates **113,000x less**: the nonlinearity is not off, it is on and *silent*.
    Energy cannot see any of this — both runs conserve, and the 2x is why "same energy" would be the
    wrong way to say it.

    Measured: pump at ``2 f1`` of **1.07e+03** planar vs **9.4e-03** circular — **113,000x**.
    """
    planar, tuned = polarization["planar"], polarization["tuned"]
    ratio = planar["pump"] / tuned["pump"]
    assert ratio > 1e4, (
        f"planar and circular pump the longitudinal field comparably ({ratio:.1f}x) — the "
        f"discriminator is this model's headline claim and it is not holding"
    )
    assert tuned["dc"] / planar["dc"] == pytest.approx(2.0, rel=0.05), (
        f"a circular mode's static stretch must be exactly 2x a planar one's (its r^2 is the "
        f"planar peak, not the planar average); got {tuned['dc'] / planar['dc']:.4f}x — so the "
        f"quiet longitudinal AC is not the nonlinearity being engaged twice as hard in silence"
    )
    assert tuned["w_max"] == pytest.approx(AMP, rel=0.05), (
        f"the circular run must trace a full circle: max|w| = {tuned['w_max']:.3e} vs A = {AMP:.3e}"
    )
    assert planar["w_max"] == 0.0, "the planar run must stay bit-exactly planar (Tier A/2)"


def test_the_circular_residual_is_ellipticity_not_a_defect_of_the_scheme(polarization):
    """*Why* the circular null is not bit-zero — and the answer is not the one the plan expected.

    The plan attributes the residual to mode shape: a sine-circular IC is not the true relative
    equilibrium, so the helix relaxes to the (non-sine) exact one and radiates. Measured, that is
    the *small* half. Drive the same **sine** helix at the KC circular
    ``Omega = sqrt(omega0^2 + eps A^2)`` instead of the linear ``Omega``, and the residual pump
    falls by **300x** (2.90 -> 9.4e-03), taking the discriminator from 367x to 113,000x.

    So the residual is **ellipticity**: at the linear ``Omega`` the two polarizations run at the
    wrong rate for their amplitude, the "circle" is an ellipse, ``r^2`` is no longer static, and it
    pumps ``v`` at ``2 Omega`` exactly like a planar mode does — in miniature.

    **This bounds ``phi``, it does not merely observe that ``Omega`` helps** — which is what makes
    it worth acting on. ``phi`` is untouched across the two runs (both are sines), so *everything*
    the shape error contributes is still present in the tuned run: its 9.4e-03 is an **upper bound**
    on ``phi``'s share. Against the 2.90 the ``Omega`` correction removed, that is ``phi <~ 9.4e-03
    << 2.9`` — the shape error is at most ~0.3 % of the effect. **So batch 3's BVP is mostly solving
    for ``Omega``, not ``phi``**, and a converged ``Omega`` with a sine ``phi`` should already get
    most of the way to Tier B's bit-zero.
    """
    naive, tuned, planar = polarization["naive"], polarization["tuned"], polarization["planar"]
    assert naive["pump"] > 50.0 * tuned["pump"], (
        f"tuning Omega to the circular relative equilibrium should collapse the residual (naive "
        f"{naive['pump']:.3e} vs tuned {tuned['pump']:.3e}) — if it does not, the residual is "
        f"something other than ellipticity and is worth understanding before batch 3 builds on it"
    )
    # Ellipticity is a *frequency* error, not an amplitude one: both circles stretch the string
    # identically (same DC), and only the AC pump moves. That is what makes it ellipticity.
    assert naive["dc"] == pytest.approx(tuned["dc"], rel=1e-2)
    # Even the naive circle is decisive; the sharpening is a bonus, not the claim's foundation.
    assert planar["pump"] / naive["pump"] > 50.0
