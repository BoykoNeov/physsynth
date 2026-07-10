"""Acoustic bore (wind leg, batch 2): the radiating bell — passivity, reflection, far field.

The open end of batch 1 reflected perfectly and radiated nothing. A **radiating** end presents a
resistance ``R_bell`` (acoustic, Pa·s/m^3): a rank-1, unconditionally-passive dashpot on the
terminating half-cell node — the :class:`~physsynth.core.radiation.RadiatedBody` pattern moved onto
a boundary. This is what makes the tube passively lossy and how sound leaves. The gates:

- **Passivity money test**: ``E_bore + radiated_energy`` is flat to machine precision for a lossless
  air column (any ``R``), across every placement of the radiating end (left / right / both — the
  both-ends case is what catches a left-end sign error), and ``radiated_energy > 0``.
- **Unconditional stability**: no CFL/guard beyond the interior one — even an absurd ``R`` (``-> 0``
  or ``-> inf``) stays bounded (``a + b > 0`` always).
- **Independent reflection oracle (the teeth)**: a pulse sheds the fraction
  ``1 - r^2 = 4 R Z0 / (R + Z0)^2`` of its energy, ``r = (R - Z0)/(R + Z0)``; a matched ``R = Z0``
  is anechoic (absorbs everything). This verifies ``R`` is *physical* radiation, which the
  self-consistent energy identity alone cannot.
- **Signature / limits**: ``R -> 0`` keeps the odd-harmonic clarinet (open-ish bell); ``R -> inf``
  recovers the rigid **closed** wall (even harmonics reappear -> closed-closed full series).
- **Clean observables**: the *interior* spectrum and the far-field read-out ``pressure()`` are clean
  even though the heavily-overdamped raw end node carries a cosmetic Nyquist ripple.
"""

import numpy as np
import pytest
from helpers import (
    BORE_LENGTH_DEFAULT,
    BORE_RADIUS_DEFAULT,
    make_bore,
    make_radiating_bore,
)

from physsynth.analysis import modal, spectrum
from physsynth.core.bore import C0_AIR, Bore
from physsynth.core.engine import simulate
from physsynth.core.radiation import piston_radiation_resistance

DRIFT_TOL = 1e-10  # same bar as every other resonator


def _bump(bore, center_frac=0.4, width_frac=0.06, amplitude=1e-3):
    c = center_frac * bore.L
    w = width_frac * bore.L
    return amplitude * np.exp(-((bore.x - c) ** 2) / (2.0 * w * w))


# -- Passivity: E_bore + radiated_energy is flat, for every placement of the radiating end. -------
@pytest.mark.parametrize(
    "boundary",
    [("closed", "radiating"), ("radiating", "closed"), ("radiating", "radiating")],
)
def test_radiation_energy_identity(boundary):
    # A big R (near-closed) so the tube keeps ringing and radiates a little every reflection -- the
    # identity must hold for ANY R, and both-ends is the only test that catches a left-end sign bug.
    bore = make_radiating_bore(N=200, lam=0.9, boundary=boundary, R_bell=1e5)
    bore.set_state(_bump(bore))
    res = simulate(bore, num_steps=int(0.3 * bore.fs), pickup_index=1)
    assert res.energy_drift < DRIFT_TOL, f"{boundary} drift {res.energy_drift:.2e}"
    assert bore.radiated_energy > 0.0, "a radiating end must shed some energy"


def test_acoustic_energy_decreases_as_bell_radiates():
    # The field energy alone falls (the bell sheds it); the total is what is conserved.
    bore = make_radiating_bore(N=200, lam=0.9, R_bell=5e4)
    bore.set_state(_bump(bore))
    e_field, e_total = [], []
    for _ in range(int(0.2 * bore.fs)):
        bore.step()
        e_field.append(bore.acoustic_energy())
        e_total.append(bore.energy())
    e_field, e_total = np.array(e_field), np.array(e_total)
    assert e_field[-1] < 0.99 * e_field[0], "air-column energy should fall as the bell radiates"
    assert np.all(np.diff(e_field) <= 1e-12 * e_field[0]), "field energy must be monotone (passive)"
    assert e_total.max() - e_total.min() < DRIFT_TOL * e_total[0], "total must stay conserved"


# -- Unconditional stability: no guard beyond the interior CFL, at any R. -------------------------
@pytest.mark.parametrize("R_bell", [1e-2, 1e12])
def test_unconditional_stability_absurd_R(R_bell):
    bore = make_radiating_bore(N=200, lam=0.9, R_bell=R_bell)
    bore.set_state(_bump(bore))
    res = simulate(bore, num_steps=int(0.3 * bore.fs), pickup_index=1)
    assert np.all(np.isfinite(res.output)), f"blew up at R={R_bell:.0e}"
    assert np.max(np.abs(res.output)) <= 2.0e-3, f"amplitude grew at R={R_bell:.0e} (should not)"


# -- The teeth: energy shed by one reflection == 1 - r^2 (an oracle independent of the coupling). --
@pytest.mark.parametrize("frac", [0.1, 0.3, 1.0, 3.0, 10.0])
def test_reflection_energy_matches_1_minus_r2(frac):
    # A stationary Gaussian at the centre splits (lambda = 1, exact) into two equal E0/2 halves.
    # The right half hits the radiating end once and sheds (1 - r^2) E0/2; the left half reflects
    # off the closed end losslessly. Measure the radiated energy in the window after the first
    # right-bounce and before the reflected-left half returns -- a single, clean reflection.
    N = 400
    R = frac * Bore(L=BORE_LENGTH_DEFAULT, fs=1e6, N=N, radius=BORE_RADIUS_DEFAULT).Z0
    bore = make_radiating_bore(N=N, lam=1.0, boundary=("closed", "radiating"), R_bell=R)
    c, w = 0.5 * bore.L, 0.03 * bore.L
    bore.set_state(1e-3 * np.exp(-((bore.x - c) ** 2) / (2.0 * w * w)))
    e0 = bore.energy()
    simulate(bore, num_steps=N, pickup_index=1)  # ~1.0 N: one right-bounce done, no second yet
    r = (R - bore.Z0) / (R + bore.Z0)
    oracle = 0.5 * (1.0 - r * r)  # single right-half reflection -> half the energy in play
    measured = bore.radiated_energy / e0
    assert abs(measured - oracle) < 0.02 * oracle, (
        f"R/Z0={frac}: shed {measured:.4f} of E0, oracle {oracle:.4f}"
    )


def test_matched_load_is_anechoic():
    # R = Z0 (r = 0): the bell absorbs everything. After long enough for both split halves to reach
    # the radiating end, essentially all the energy has left and the tube is quiet.
    N = 400
    Z0 = Bore(L=BORE_LENGTH_DEFAULT, fs=1e6, N=N, radius=BORE_RADIUS_DEFAULT).Z0
    bore = make_radiating_bore(N=N, lam=1.0, boundary=("closed", "radiating"), R_bell=Z0)
    c, w = 0.5 * bore.L, 0.03 * bore.L
    bore.set_state(1e-3 * np.exp(-((bore.x - c) ** 2) / (2.0 * w * w)))
    e0 = bore.energy()
    simulate(bore, num_steps=3 * N, pickup_index=1)
    assert bore.radiated_energy > 0.97 * e0, (
        f"anechoic load absorbed only {bore.radiated_energy / e0:.3f} of E0"
    )
    assert bore.acoustic_energy() < 0.03 * e0, "matched load should leave the tube nearly silent"


# -- Signature: R interpolates the two ideal ends (open-harmonic <-> closed-harmonic). ------------
def _odd_even_ratio(bore, center_frac, f1_co):
    bore.set_state(_bump(bore, center_frac=center_frac))
    res = simulate(bore, num_steps=int(0.5 * bore.fs), pickup_index=1)  # INTERIOR pickup
    freqs, mag, _ = spectrum.magnitude_spectrum(res.output, res.fs)
    df = freqs[1] - freqs[0]

    def peak(f):
        i = int(round(f / df))
        return mag[max(1, i - 2) : i + 3].max()

    odd = [peak((2 * n - 1) * f1_co) for n in range(1, 5)]
    even = [peak(2 * n * f1_co) for n in range(1, 5)]
    return min(odd) / max(even)


def test_small_R_bell_is_odd_harmonic_clarinet():
    # A realistic bell (R << Z0) stays the odd-only clarinet -- the interior spectrum is clean (the
    # end-node Nyquist ripple does not leak into the harmonics).
    f1_co = modal.bore_resonance_frequencies(C0_AIR, BORE_LENGTH_DEFAULT, 1, "closed-open")[0]
    bore = make_radiating_bore(N=256, lam=1.0, R_bell=650.0)
    ratio = _odd_even_ratio(bore, center_frac=0.1, f1_co=f1_co)
    assert ratio > 1e3, f"open-ish bell not odd-harmonic: min-odd/max-even = {ratio:.2e}"


def test_large_R_bell_recovers_closed_wall():
    # A rigid bell (R -> inf) is a closed wall -> closed-closed -> the FULL series: the even
    # harmonics (absent for closed-open) are now present, so the odd-only ratio collapses.
    f1_co = modal.bore_resonance_frequencies(C0_AIR, BORE_LENGTH_DEFAULT, 1, "closed-open")[0]
    bore = make_radiating_bore(N=256, lam=1.0, R_bell=1e10)
    ratio = _odd_even_ratio(bore, center_frac=0.12, f1_co=f1_co)
    assert ratio < 1.0, f"rigid bell still suppresses even harmonics: ratio = {ratio:.2e}"


# -- Clean observables: the far-field read-out is smoother than the raw (overdamped) end node. ----
def test_far_field_readout_is_clean():
    bore = make_radiating_bore(N=256, lam=1.0, R_bell=650.0)
    bore.set_state(_bump(bore, center_frac=0.1))
    p_far, p_raw = [], []
    for _ in range(int(0.3 * bore.fs)):
        bore.step()
        p_far.append(bore.pressure())  # volume acceleration dU_out/dt (Nyquist cancels)
        p_raw.append(bore.p[-1])       # raw terminating node (carries the cosmetic ripple)
    p_far, p_raw = np.array(p_far), np.array(p_raw)

    def nyquist_fraction(x):
        mag = np.abs(np.fft.rfft(x - x.mean()))
        return (mag[int(0.95 * len(mag)) :] ** 2).sum() / (mag**2).sum()

    ff = nyquist_fraction(p_far)
    assert ff < 1e-2, f"far-field read-out not clean: Nyquist fraction {ff:.2e}"
    # And it is markedly cleaner than reading the raw overdamped end node.
    assert ff < 0.5 * nyquist_fraction(p_raw), "pressure() should be cleaner than the raw end node"


def test_pressure_readout_composes_with_air_radiation():
    # The bell's pressure() is exactly what batch-1 AirRadiation reads: the chain produces a
    # nonzero, delayed far-field pressure once the wavefront reaches the listener.
    from physsynth.core.radiation import AirRadiation

    bore = make_radiating_bore(N=200, lam=1.0, R_bell=650.0)
    bore.set_state(_bump(bore, center_frac=0.3))
    air = AirRadiation(fs=bore.fs, distance=1.0)
    out = []
    for _ in range(int(0.05 * bore.fs)):
        bore.step()
        out.append(air.radiate(bore))
    out = np.array(out)
    assert np.all(np.isfinite(out))
    assert np.max(np.abs(out)) > 0.0, "the radiating bell should drive the far-field mic"


# -- The piston radiation-resistance oracle sizes a physical bell (R << Z0, lightly radiating). ---
def test_physical_bell_resistance_is_lightly_radiating():
    bore = make_bore(N=200, lam=1.0)  # geometry reference (Z0, f1)
    f1 = modal.bore_resonance_frequencies(C0_AIR, bore.L, 1, "closed-open")[0]
    omega1 = 2.0 * np.pi * f1
    R = piston_radiation_resistance(omega1, bore.radius)
    assert 0.0 < R < 1e-3 * bore.Z0, (
        f"a small bore end should radiate weakly: R/Z0 = {R / bore.Z0:.2e}"
    )


# -- Validation: radiating needs R_bell > 0; unknown tokens rejected. ----------------------------
def test_radiating_requires_positive_R():
    with pytest.raises(ValueError, match="radiating"):
        Bore(L=0.5, fs=1e6, N=200, boundary=("closed", "radiating"), R_bell=0.0)


def test_negative_R_rejected():
    with pytest.raises(ValueError, match="R_bell"):
        Bore(L=0.5, fs=1e6, N=200, boundary=("closed", "open"), R_bell=-1.0)


def test_unknown_boundary_token_rejected():
    with pytest.raises(ValueError, match="each boundary end"):
        Bore(L=0.5, fs=1e6, N=200, boundary=("closed", "flared"))


def test_no_radiating_end_is_bit_identical_to_plain_bore():
    # R_bell defaults do not touch a closed/open bore: radiated_energy stays 0, energy() unchanged.
    a = make_bore(N=200, lam=0.9, boundary=("closed", "open"))
    b = Bore(L=a.L, fs=a.fs, N=a.N, radius=a.radius, boundary=("closed", "open"), R_bell=0.0)
    for bore in (a, b):
        bore.set_state(_bump(bore))
    ea = simulate(a, num_steps=2000, pickup_index=1)
    eb = simulate(b, num_steps=2000, pickup_index=1)
    assert np.array_equal(ea.output, eb.output)
    assert b.radiated_energy == 0.0
