"""The closed-form oracles, Python against Rust -- where almost everything agrees exactly.

Phase 7 batch 2 (``docs/dev/rust-migration-plan.md`` §37). ``modal``, ``damping``, ``dispersion``
and ``duffing`` are the *instrument* that every modal test in this project reads, so they go behind
``PHYSSYNTH_RS_ANALYSIS`` rather than ``PHYSSYNTH_RS`` -- the second flag, whose whole purpose is
that the acceptance run measures a Rust model with a Python ruler against an unmoved oracle (§36.4).

**This file runs unflagged and builds both sides itself.** Every ``*_py`` name is the Python
implementation the module keeps under an alias the swap never rebinds, and every ``rs.*`` name is
the binding. Setting ``PHYSSYNTH_RS_ANALYSIS=1`` here would compare Rust against Rust.

What is asserted, and why the line falls where it does
-----------------------------------------------------

Measured over the fixtures below, **almost every function in this batch is bit-identical**,
including all of the ones with a ``sin``, ``arcsin``, ``arccos`` or ``log2`` on the path. That is a
pleasant number and it is *not* what this file requires, for the reason ledger #28 was written
down: NumPy computes transcendentals with its own CPU-dispatched kernels rather than the platform
libm (§22.1), so "these agree to the bit" is a claim about which machine ran CI -- and this project
has already watched one such assertion turn into eighteen failures on unchanged code. So:

* **Equality is required only where IEEE-754 requires it** -- expressions built from ``+ - * /``
  and ``sqrt``, which every conforming machine rounds identically. Eighteen functions qualify.
* **Everything with a transcendental on the path gets a tolerance**, and the observed exactness is
  reported in a comment rather than asserted.
* **The two root-finds have a structural claim as well as a numeric one**: the *number* of roots
  found must match exactly. A tolerance cannot describe a search that returns a different answer
  count, and for the free circular plate a missing root is the dangerous direction -- all three of
  its docstring's self-checks catch a spurious root and none catches an absent one.
* **Oscillatory outputs are normalised by amplitude, never pointwise.** ``duffing_displacement``
  agrees to 1.4e-17 absolute on a waveform of amplitude 0.07 -- two parts in 1e16 -- but at the
  sample nearest a zero crossing the *relative* disagreement is 2.7e-4, because ``|q|`` there is
  1.7e-17. Same for ``duffing_frequency_shift``, which is a difference of two nearly equal
  frequencies: 3.5e-15 of the shift, 1.6e-16 of the frequency. Both are recorded scars
  (``docs/memory/rust-phase2-radiation-state.md``), and this file is where they would otherwise
  have been rediscovered.
"""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest

import physsynth.analysis.damping as D
import physsynth.analysis.dispersion as P
import physsynth.analysis.duffing as F
import physsynth.analysis.modal as M
import physsynth.core.radiation as R

rs = pytest.importorskip("physsynth_rs")

C, L, N, KAPPA, K, THETA = 200.0, 0.65, 128, 0.7, 1e-5, 0.5
MODES = [(1, 1), (2, 1), (1, 2), (3, 2), (4, 5)]
W0SQ = (2.0 * np.pi * 220.0) ** 2


def _flat(a) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(a, dtype=float).ravel())


# -- tier 1: the arithmetic IEEE-754 pins, asserted as equality ---------------------------------


def test_the_pure_arithmetic_is_bit_identical():
    """``+ - * / sqrt`` only -- every conforming machine rounds these the same way.

    This is the tier that can be *required* rather than observed, and it is the reason the batch
    can claim anything portable at all. Eighteen functions, no transcendental among them.
    """
    assert np.array_equal(
        M.harmonic_frequencies_py(C, L, 12), rs.modal_harmonic_frequencies(C, L, 12)
    )
    assert M.inharmonicity_B_py(C, L, KAPPA) == rs.modal_inharmonicity_b(C, L, KAPPA)
    assert np.array_equal(
        M.stiff_harmonic_frequencies_py(C, L, KAPPA, 12),
        rs.modal_stiff_harmonic_frequencies(C, L, KAPPA, 12),
    )
    assert np.array_equal(
        M.rectangular_membrane_freqs_py(C, 0.4, 0.3, MODES),
        rs.modal_rectangular_membrane_freqs(C, 0.4, 0.3, MODES),
    )
    assert np.array_equal(
        M.rectangular_plate_freqs_py(KAPPA, 0.4, 0.3, MODES),
        rs.modal_rectangular_plate_freqs(KAPPA, 0.4, 0.3, MODES),
    )
    assert np.array_equal(
        M.orthotropic_plate_freqs_py(KAPPA, 0.4, 0.3, MODES, 1.2, 0.9, 0.7),
        rs.modal_orthotropic_plate_freqs(KAPPA, 0.4, 0.3, MODES, 1.2, 0.9, 0.7),
    )
    assert np.array_equal(
        M.free_plate_ffff_square_lambdas_py(), rs.modal_free_plate_ffff_square_lambdas()
    )
    lam = np.array([13.468, 19.596, 24.270])
    assert np.array_equal(
        M.free_plate_freq_from_lambda_py(lam, KAPPA, 0.25),
        rs.modal_free_plate_freq_from_lambda(_flat(lam), KAPPA, 0.25),
    )
    assert M.free_plate_twist_bound_py(KAPPA, 0.3, 0.2, 0.5) == rs.modal_free_plate_twist_bound(
        KAPPA, 0.3, 0.2, 0.5
    )
    assert M.free_circular_plate_saddle_bound_py(
        0.3
    ) == rs.modal_free_circular_plate_saddle_bound(0.3)
    assert M.free_plate_coupling_form_py(0.8, 0.01, 40, 30) == rs.modal_free_plate_coupling_form(
        0.8, 0.01, 40, 30
    )
    for boundary in ("closed-open", "open-open", "closed-closed"):
        assert np.array_equal(
            M.bore_resonance_frequencies_py(343.0, 0.6, 8, boundary),
            rs.modal_bore_resonance_frequencies(343.0, 0.6, 8, boundary),
        )
    for m in range(1, 12):
        assert D.modal_loss_rate_continuum_py(
            C, L, KAPPA, 0.5, 1e-5, m
        ) == rs.damping_modal_loss_rate_continuum(C, L, KAPPA, 0.5, 1e-5, m)
    assert D.loss_coefficients_from_T60_py(
        C, L, 0.0, 100.0, 4.0, 1000.0, 1.0
    ) == rs.damping_loss_coefficients_from_t60(C, L, 0.0, 100.0, 4.0, 1000.0, 1.0)
    assert F.kc_mode_coefficients_py(
        c=C, kappa=0.0, EA=1.2e4, rho=6.3e-3, p2=400.0, L=L
    ) == rs.duffing_kc_mode_coefficients(C, 0.0, 1.2e4, 6.3e-3, 400.0, L)
    assert F.kc_mode_stretch_py(0.07, p2=400.0, L=L) == rs.duffing_kc_mode_stretch(0.07, 400.0, L)
    assert F.duffing_elliptic_parameter_py(0.07, W0SQ, 5e7) == rs.duffing_elliptic_parameter(
        0.07, W0SQ, 5e7
    )


# -- tier 2: a transcendental on the path, asserted as a tolerance ------------------------------


def test_the_frequency_formulas_agree_to_a_last_bit():
    """Every one of these has a ``sin``, ``arcsin``, ``arccos`` or ``log2`` in it.

    All observed **bit-identical** on the machine that wrote this, and none of that is required:
    NumPy's transcendentals are its own CPU-dispatched kernels, so requiring equality here would be
    requiring a property of the runner (ledger #28).
    """
    x = np.linspace(0.0, L, N + 1)
    assert rs.modal_mode_shape(_flat(x), L, 3) == pytest.approx(
        M.mode_shape_py(x, L, 3), rel=1e-15, abs=1e-16
    )
    for m in range(1, 12):
        assert rs.modal_discrete_mode_frequency(C, L, N, 0.9, m) == pytest.approx(
            M.discrete_mode_frequency_py(C, L, N, 0.9, m), rel=1e-15
        )
        assert rs.modal_discrete_stiff_mode_frequency(
            C, L, N, KAPPA, K, m, THETA
        ) == pytest.approx(
            M.discrete_stiff_mode_frequency_py(C, L, N, KAPPA, K, m, THETA), rel=1e-15
        )
        assert rs.damping_spatial_eigenvalue_p2(N, L / N, m) == pytest.approx(
            D.spatial_eigenvalue_p2_py(N, L / N, m), rel=1e-15
        )
    f = M.harmonic_frequencies_py(C, L, 12)
    assert rs.modal_cents(_flat(f), _flat(f * 1.001)) == pytest.approx(
        M.cents_py(f, f * 1.001), rel=1e-14, abs=1e-14
    )
    xx, yy = np.meshgrid(np.linspace(0, 0.4, 17), np.linspace(0, 0.3, 13))
    assert rs.modal_rectangular_mode_field(_flat(xx), _flat(yy), 0.4, 0.3, 2, 3) == pytest.approx(
        M.rectangular_mode_field_py(xx, yy, 0.4, 0.3, 2, 3).ravel(), rel=1e-14, abs=1e-16
    )
    assert rs.modal_rectangular_discrete_eigenvalues(0.01, 40, 30, MODES) == pytest.approx(
        M.rectangular_discrete_eigenvalues_py(0.01, 40, 30, MODES), rel=1e-15
    )
    lam = np.array([100.0, 4000.0, 1e5])
    assert rs.modal_discrete_membrane_eigenfrequency(_flat(lam), C, K) == pytest.approx(
        M.discrete_membrane_eigenfrequency_py(lam, C, K), rel=1e-15
    )
    assert rs.modal_discrete_plate_eigenfrequency(_flat(lam), KAPPA, K, THETA) == pytest.approx(
        M.discrete_plate_eigenfrequency_py(lam, KAPPA, K, THETA), rel=1e-15
    )
    lx, ly = np.array([900.0, 1600.0, 2500.0]), np.array([400.0, 1200.0, 3000.0])
    assert rs.modal_discrete_orthotropic_plate_eigenfrequency(
        _flat(lx), _flat(ly), KAPPA, K, THETA, 1.2, 0.9, 0.7
    ) == pytest.approx(
        M.discrete_orthotropic_plate_eigenfrequency_py(lx, ly, KAPPA, K, THETA, 1.2, 0.9, 0.7),
        rel=1e-15,
    )
    mm = np.arange(1, 9, dtype=float)
    assert rs.modal_dirichlet_axis_eigenvalue(_flat(mm), L, L / 128) == pytest.approx(
        M.dirichlet_axis_eigenvalue_py(mm, L, L / 128), rel=1e-15
    )
    w2 = np.array([1e6, 4e6, 9e6])
    assert rs.modal_discrete_bore_eigenfrequency(_flat(w2), K) == pytest.approx(
        M.discrete_bore_eigenfrequency_py(w2, K), rel=1e-15
    )
    mu = np.array([1e6, 4e7, 9e8])
    assert rs.modal_discrete_beam_eigenfrequency(_flat(mu), KAPPA, K, THETA) == pytest.approx(
        M.discrete_beam_eigenfrequency_py(mu, KAPPA, K, THETA), rel=1e-15
    )


def test_the_damped_decay_and_rate_agree():
    """The decay factor and the rate, which are the same numbers with a logarithm between them."""
    for m in (1, 3, 7, 17):
        g_py = D.discrete_damped_mode_decay_py(C, L, N, KAPPA, K, THETA, 0.5, 1e-5, m)
        g_rs = rs.damping_discrete_damped_mode_decay(C, L, N, KAPPA, K, THETA, 0.5, 1e-5, m)
        assert g_rs == pytest.approx(g_py, rel=1e-15)
        assert rs.damping_discrete_damped_mode_rate(
            C, L, N, KAPPA, K, THETA, 0.5, 1e-5, m
        ) == pytest.approx(
            D.discrete_damped_mode_rate_py(C, L, N, KAPPA, K, THETA, 0.5, 1e-5, m), rel=1e-15
        )
        # The lossless case IS exact and is required as such: both sides build `a` and `c` from the
        # same expression with the sigma term at zero, so `c/a` is 1.0 on the nose. A string with no
        # loss must not decay, and "1.0 within a tolerance" would not say that.
        assert (
            rs.damping_discrete_damped_mode_decay(C, L, N, KAPPA, K, THETA, 0.0, 0.0, m) == 1.0
        )
        assert D.discrete_damped_mode_decay_py(C, L, N, KAPPA, K, THETA, 0.0, 0.0, m) == 1.0


@pytest.mark.parametrize(
    "sigma0,sigma1", [(0.0, 0.0), (0.5, 1e-5), (50.0, 1e-3), (1e6, 0.0), (1e5, 1.0)]
)
def test_the_underdamped_predicate_agrees_and_its_margin_is_reported(sigma0, sigma1):
    """A bool is a discrete output (§25), so the margin is measured rather than assumed.

    Observed over these five configurations and modes 1..40: the discriminant ``b^2 - 4ac`` is never
    within eleven orders of magnitude of zero relative to ``|b^2| + |4ac|``, so no rounding can flip
    the predicate. The agreement is required; the margin is reported, because a future fixture that
    narrows it should fail *here* rather than silently answer differently.
    """
    worst = np.inf
    for m in range(1, 41):
        a = D.discrete_damped_mode_is_underdamped_py(C, L, N, KAPPA, K, THETA, sigma0, sigma1, m)
        b = rs.damping_discrete_damped_mode_is_underdamped(
            C, L, N, KAPPA, K, THETA, sigma0, sigma1, m
        )
        assert a == b, f"the two sides disagree about mode {m}"
        p2 = D.spatial_eigenvalue_p2_py(N, L / N, m)
        q = C * C * p2 + KAPPA * KAPPA * p2 * p2
        base = 1.0 + THETA * K * K * q
        aa, bb = base + (sigma0 + sigma1 * p2) * K, -2.0 + (1.0 - 2.0 * THETA) * K * K * q
        cc = base - (sigma0 + sigma1 * p2) * K
        disc, scale = bb * bb - 4.0 * aa * cc, abs(bb * bb) + abs(4.0 * aa * cc)
        worst = min(worst, abs(disc) / scale)
    assert worst > 1e-11, f"the underdamped margin has narrowed to {worst:.3e}"


# -- tier 3: the two searches, where the count is the claim -------------------------------------


def test_the_beam_roots_agree_and_there_are_the_same_number_of_them():
    py, rust = M.free_free_beam_betaL_py(12), rs.modal_free_free_beam_beta_l(12)
    assert len(py) == len(rust)
    assert rust == pytest.approx(py, rel=1e-14)
    assert rs.modal_free_free_beam_freqs(KAPPA, L, 12) == pytest.approx(
        M.free_free_beam_freqs_py(KAPPA, L, 12), rel=1e-14
    )
    with pytest.raises(ValueError):
        rs.modal_free_free_beam_beta_l(0)


@pytest.mark.parametrize("nu", [0.0, 0.3, 0.45, -0.2])
@pytest.mark.parametrize("n", [0, 1, 2, 3, 4, 5, 6])
def test_the_free_disk_finds_the_same_roots_and_the_same_number_of_them(nu, n):
    """The batch's one decision, and its margin was measured before the port.

    The scan keeps a bracket wherever ``vs[i]*vs[i+1] < 0`` across 20,000 points, so a last-bit sign
    flip adds or drops a root. Worst ``|det|`` against the cancellation that produced it, away from
    a genuine crossing: 4.6e-6, versus the ~1e-15 a Bessel routine can move it -- a margin of ~5e9x.
    Asserting the *count* is what makes that measurement mean something: a missing root is the
    dangerous direction, and the docstring's three self-checks all catch a spurious one instead.
    """
    py = M.free_circular_plate_lambda_roots_py(nu, n)
    rust = rs.modal_free_circular_plate_lambda_roots(nu, n, 14.0, 20000)
    assert len(py) == len(rust), f"root COUNT differs at nu={nu}, n={n}: {py} vs {rust}"
    if len(py):
        assert rust == pytest.approx(py, rel=1e-12)


def test_the_disk_spectrum_agrees_including_its_multiplicities():
    lam_py, n_py = M.free_circular_plate_lambdas_py(0.3, 7)
    lam_rs, n_rs = rs.modal_free_circular_plate_lambdas(0.3, 7, 8)
    # The nodal-diameter tags are integers: exact, and the degenerate pairs must be paired the same
    # way on both sides or every comparison past the first entry misaligns.
    assert np.array_equal(n_py, n_rs)
    assert lam_rs == pytest.approx(lam_py, rel=1e-12)


def test_the_circular_membrane_spectrum_agrees_and_its_bessel_zeros_with_it():
    """``jn_zeros`` by AMOS on one side and by interlacing plus Brent on the other.

    13 of 20 frequencies come back bit-identical and the rest agree to 3.5e-16 relative. The
    ordering and the degeneracy tags are integers and must match exactly -- a spectrum that sorted
    differently would misalign every downstream comparison while looking entirely plausible.
    """
    py = M.circular_membrane_freqs_py(100.0, 0.15, 20)
    rust = rs.modal_circular_membrane_freqs(100.0, 0.15, 20, 12, 12)
    assert len(py) == len(rust)
    for (m_a, n_a, f_a, d_a), (m_b, n_b, f_b, d_b) in zip(py, rust, strict=True):
        assert (m_a, n_a, d_a) == (m_b, n_b, d_b)
        assert f_b == pytest.approx(f_a, rel=1e-14)


# -- tier 4: the oscillatory outputs, normalised by amplitude ------------------------------------


@pytest.mark.parametrize("eps", [0.0, 1e6, 5e7, 5e8])
def test_the_duffing_waveform_agrees_relative_to_its_own_amplitude(eps):
    """Normalised by amplitude, never pointwise -- the scar from Phase 2 batch 4.

    Observed: 1.4e-17 absolute on an amplitude of 0.07, which is two parts in 1e16, with 484 of 501
    samples bit-identical. Pointwise the same data reads as **2.7e-4 relative**, entirely because
    the worst sample sits on a zero crossing where ``|q|`` is 1.7e-17. The second number describes
    the fixture's luck, not the port.
    """
    amp, ts = 0.07, np.linspace(0.0, 0.05, 501)
    py = F.duffing_displacement_py(ts, amp, W0SQ, eps)
    rust = rs.duffing_displacement(_flat(ts), amp, W0SQ, eps)
    assert np.max(np.abs(rust - py)) < 1e-14 * amp
    assert rust[0] == amp and py[0] == amp, "both must start exactly at the amplitude"


def test_the_duffing_frequency_agrees_and_its_shift_is_measured_against_the_frequency():
    """``duffing_frequency_shift`` is ``omega(A) - omega_0``: a difference of two close numbers.

    3.5e-15 of the shift, 1.6e-16 of the frequency it came from. The second is the honest bar; the
    first is a statement about how much cancellation the subtraction did.
    """
    for amp in (0.001, 0.01, 0.07):
        w_py = F.duffing_frequency_py(amp, W0SQ, 5e7)
        w_rs = rs.duffing_frequency(amp, W0SQ, 5e7)
        assert w_rs == pytest.approx(w_py, rel=1e-15)
        s_py = F.duffing_frequency_shift_py(amp, W0SQ, 5e7)
        s_rs = rs.duffing_frequency_shift(amp, W0SQ, 5e7)
        assert abs(s_rs - s_py) < 1e-14 * w_py
        assert rs.duffing_frequency_expansion(amp, W0SQ, 5e7) == pytest.approx(
            F.duffing_frequency_expansion_py(amp, W0SQ, 5e7), rel=1e-15
        )


# -- dispersion ---------------------------------------------------------------------------------


def test_dispersion_delegates_to_the_same_numbers_on_both_sides():
    modes = np.arange(1, 13)
    assert rs.dispersion_frequencies(C, L, N, 0.9, [int(m) for m in modes]) == pytest.approx(
        P.dispersion_frequencies_py(C, L, N, 0.9, modes), rel=1e-15
    )
    f = P.stiff_dispersion_frequencies_py(C, L, N, KAPPA, K, THETA, modes)
    assert rs.dispersion_stiff_frequencies(
        C, L, N, KAPPA, K, THETA, [int(m) for m in modes]
    ) == pytest.approx(f, rel=1e-15)
    # Phase velocity is pure arithmetic on top: 2*L*f/m, so it is exact given the same f.
    assert np.array_equal(
        rs.dispersion_phase_velocity(_flat(f), L, [int(m) for m in modes]),
        P.phase_velocity_py(f, L, modes),
    )


# -- the piston, plan §14's parked call --------------------------------------------------------


def test_the_piston_resistance_agrees_at_the_call_sites_the_suite_reaches():
    """The Bessel debt §14 parked in Phase 2 batch 4, paid -- and it swaps on the MODEL flag.

    `piston_radiation_resistance` lives in `physsynth/core/radiation.py`, so it goes behind
    ``PHYSSYNTH_RS`` like the rest of its file even though its Rust implementation lives in the
    analysis crate. See that module's footer for why those are separate questions.
    """
    for omega, radius in ((2 * np.pi * 2000.0, 0.05), (2 * np.pi * 440.0, 0.0075), (1e4, 0.02)):
        assert rs.piston_radiation_resistance(omega, radius) == pytest.approx(
            R.piston_radiation_resistance_py(omega, radius), rel=1e-13
        )


def test_the_pistons_series_branch_is_bit_identical_and_the_direct_one_is_not():
    """The two branches have different agreement stories, and the split is the whole design.

    This test asserted the *cancellation band* until 2026-09-03 — the shipped `ka < 1e-8` cutoff sat
    three decades below where `1 - J1(2ka)/ka` becomes computable, so the two sides disagreed by
    300% just above it and the test's job was to record that no caller was in the band. Hurdles §14
    is now fixed: three Taylor terms below `ka = 3e-2`.

    What replaces it is the *reason* the fix works. Measured over 3,000 values per branch:

    * **below the cutoff: 0 differ.** The series is `+ - * /` only, which IEEE-754 pins, so this is
      required as equality rather than as a tolerance.
    * **above it: 1,444 of 3,000 differ, worst 9.8e-13.** The direct form runs through two different
      `J1` implementations — Cephes on the Python side, a Miller recurrence on the Rust one — and
      that can never be exact. The bar is a tolerance and the count is reported, not required.

    The 9.8e-13 is much larger than the ~1e-16 the two `J1`s differ by, and that factor is why the
    threshold is where it is: at `ka = 3e-2` the bracket is 4.5e-4, so the subtraction still
    amplifies a last bit about 2,200 times. A lower cutoff hands more of the domain to a branch that
    magnifies disagreement; a higher one hands more to a truncated series.
    """
    radius, c0 = 0.05, 343.0
    below = np.logspace(-10, np.log10(R.PISTON_SERIES_CUTOFF_KA * 0.999), 400)
    for ka in below:
        omega = ka * c0 / radius
        assert rs.piston_radiation_resistance(omega, radius) == R.piston_radiation_resistance_py(
            omega, radius
        ), f"the series branch must be bit-identical, and is not at ka = {ka}"
    above = np.logspace(np.log10(R.PISTON_SERIES_CUTOFF_KA * 1.001), 1.0, 400)
    worst = 0.0
    for ka in above:
        omega = ka * c0 / radius
        a = rs.piston_radiation_resistance(omega, radius)
        b = R.piston_radiation_resistance_py(omega, radius)
        assert a == pytest.approx(b, rel=1e-11)
        worst = max(worst, abs(a / b - 1.0))
    assert worst < 1e-11, f"the direct branch has drifted to {worst:.3e}"


def test_the_pistons_two_branches_meet_so_the_function_has_no_step():
    """The property the threshold has to keep, on the Python side as well as the Rust one.

    A future edit that moved the cutoff back into the cancellation would not break any physics bar
    in this project — they are all percentage-level and the branches differ by parts in 1e13 — so
    it is asserted directly, on both implementations, at the seam.
    """
    radius, c0 = 0.05, 343.0
    cut = R.PISTON_SERIES_CUTOFF_KA
    for d in (0.999, 0.9999, 1.0001, 1.001):
        ka = cut * d
        omega = ka * c0 / radius
        for f in (R.piston_radiation_resistance_py, rs.piston_radiation_resistance):
            here = f(omega, radius)
            across = f(omega * (1.002 if d < 1.0 else 0.998), radius)
            assert here == pytest.approx(across, rel=5e-3), (
                f"a step at the seam: {here} vs {across} at ka = {ka}"
            )


# -- the seam: what the flag actually swaps -----------------------------------------------------

_FLAG_PROBE = """
import numpy as np
import physsynth.analysis.dispersion as P
import physsynth.analysis.modal as M

# 1. The early binding. `dispersion.py` does `from .modal import discrete_mode_frequency` at import
#    time, and an early binding is exactly what defeats a swap done by rebinding module globals.
#    It is safe because modal's footer runs inside modal's own body -- assert that, don't argue it.
assert P.discrete_mode_frequency is not M.discrete_mode_frequency_py, "dispersion is stale"
assert M.discrete_mode_frequency is not M.discrete_mode_frequency_py, "modal did not swap"

# 2. The 0-d shape convention. NumPy returns a SCALAR for scalar input, and the binding's flat
#    arrays would return shape (1,) if the wrapper got this wrong -- which would keep working.
one = M.discrete_membrane_eigenfrequency(4000.0, 200.0, 1e-5)
assert np.isscalar(one) or np.ndim(one) == 0, f"0-d input gave {type(one)} of shape {np.shape(one)}"
many = M.discrete_membrane_eigenfrequency(np.array([4000.0, 9000.0]), 200.0, 1e-5)
assert many.shape == (2,), many.shape
mesh = np.array([[1.0, 2.0], [3.0, 4.0]])
assert M.dirichlet_axis_eigenvalue(mesh, 0.65, 0.005).shape == (2, 2)

# 3. The other flag must not have moved: a core model is Python here.
import physsynth.core.string_ideal as S
assert S.IdealString is S.IdealStringPy, "PHYSSYNTH_RS_ANALYSIS swapped a MODEL"
print("PROBE OK")
"""


def test_the_analysis_flag_swaps_the_instrument_and_only_the_instrument():
    """Run the probe in a subprocess, because the flag is read at import time.

    Three things it pins, none of which the unflagged tests above can reach: that the early
    ``from .modal import`` in ``dispersion.py`` picks up the swapped names, that a 0-d input still
    comes back as a scalar rather than a one-element array, and that the analysis flag leaves the
    *models* alone -- which is the whole point of there being two flags (§36.4).
    """
    env = dict(os.environ, PHYSSYNTH_RS_ANALYSIS="1")
    env.pop("PHYSSYNTH_RS", None)
    out = subprocess.run(
        [sys.executable, "-c", _FLAG_PROBE], env=env, capture_output=True, text=True
    )
    assert out.returncode == 0, f"stdout:\n{out.stdout}\nstderr:\n{out.stderr}"
    assert "PROBE OK" in out.stdout
