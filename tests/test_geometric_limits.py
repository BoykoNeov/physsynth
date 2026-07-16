"""Model #10 — convergence order, and the Tier C **limit** oracles (KC / Duffing / model #9).

Two things live here, and the plan bundles them on purpose (acceptance criterion 13: "amplitude
shift vs Duffing + Richardson O(h²), **both short-time**").

**Convergence order — the gate batch 1 does not have.** CLAUDE.md lists order beside conservation,
passivity and modal frequency, and the family carries one Richardson number per model (#5b: 5.66,
#6: 4.40/5.66). Batch 1 has three of the four. It was deferred to here because the oracle it would
naturally be measured against lands in this file — and then, on arrival, that oracle turned out to
be the wrong instrument (see below), so the number is a **self**-convergence one after all.

**Tier C is a set of LIMIT oracles, and that is the model's whole thesis.** Model #9's elliptic
Duffing solution is exact *for model #9* **because** #9's tension is a spatial scalar. Promote the
tension to a field and the closed form degrades to a limit. That degradation is not a regression;
it *is* model #10. So these tests **measure the residual and its scaling** — they do not promise
cents. The two limit oracles break by *different* mechanisms, which is what makes them worth having
separately:

* **Planar single-mode Duffing** breaks via **phantom leakage** — the ``a r^2 v_x/2`` term bleeds
  energy into the longitudinal field that KC has nowhere to put.
* **Model #9 itself** is the quasi-static-longitudinal (``c_long >> c``) + small-slope limit, so it
  is a genuine cross-model gate — mind ``EA_#9 <-> (EA - T0)_#10``.

The circular-mode limit oracle (Tier C/8) and the exact rotating-wave BVP (Tier B) are batch 3.
"""

import numpy as np
import pytest
from helpers import (
    EA_DEFAULT,
    KAPPA_DEFAULT,
    L_DEFAULT,
    RHO_DEFAULT,
    make_geometric_string,
    make_tension_string,
    wave_speed,
)

from physsynth.analysis.damping import spatial_eigenvalue_p2
from physsynth.analysis.duffing import (
    duffing_frequency_shift,
    kc_mode_coefficients,
)

# -- convergence order: Richardson SELF-convergence ------------------------------------------------


def _mode(s, m):
    """The ``m``-th simply-supported eigenmode on ``s``'s grid (interior + clamped ends)."""
    return np.sin(m * np.pi * s.x / s.L)


def test_richardson_second_order_self_convergence():
    """O(h²), refining ``h`` and ``k`` **together** — the order gate batch 1 deferred.

    **Why self-convergence and not "error against Duffing"** (model #9's convergence test, which
    would be the obvious thing to copy): for model #9 the Duffing reduction is *exact*, so its error
    goes to zero and a rate can be read off it. For model #10 Duffing is a **limit** — the
    ``a r^2 v_x/2`` coupling is real physics that KC structurally lacks — so that error converges to
    a **nonzero constant** (the phantom leakage), and the measured "order" would decay to 0 as the
    residual plateaus. It would be measuring the *oracle's* error, not the *scheme's*. The plan's
    own thesis ("the exact oracle degrades to a limit oracle the moment tension becomes a field")
    is exactly the reason the instrument has to change. Models #5b/#6 already do it this way.

    **Refine ``lam_long``-fixed, i.e. ``k`` with ``h``.** Refining ``h`` at fixed ``fs`` is the trap
    batch 1 documented: ``lam_long ∝ 1/h``, so spatial refinement walks a *converging* scheme into
    the unresolved regime and the error grows. Holding ``lam_long`` keeps ``k ∝ h``, which is the
    honest joint O(h²)+O(k²) claim anyway. ``n_steps = M (N/32)`` then lands every grid on the
    **identical** physical time, exactly, with no rounding.

    Smooth low-mode IC (models #6's lesson): a plucked corner is O(h) and would swamp the rate.
    Two modes, not one — a single mode is secretly a scalar Duffing test and never exercises the
    *local* tension field that distinguishes this model.
    """
    M, amp = 1500, 3e-3
    probe_x = 0.25  # a fixed PHYSICAL position; index N//4 on every grid below

    def state_after(N):
        s = make_geometric_string(N=N, EA=EA_DEFAULT, lam_long=0.5)
        ic = amp * (np.sin(3 * np.pi * s.x / s.L) + 0.5 * np.sin(4 * np.pi * s.x / s.L))
        s.set_state(ic)
        for _ in range(M * (N // 32)):
            s.step()
        assert s.converged and s.n_not_converged == 0
        nl_frac = abs(s.nonlinear_energy() / s.energy())
        return s.u[int(round(probe_x * N))], nl_frac

    res = {N: state_after(N) for N in (32, 64, 128)}
    u = {N: r[0] for N, r in res.items()}

    # The nonlinearity must actually be engaged, or this is a linear convergence test wearing a
    # nonlinear model's clothes (models #6/#9's lesson).
    assert min(r[1] for r in res.values()) > 1e-3, (
        f"nonlinear fraction too small to be measuring the nonlinear scheme: "
        f"{[f'{r[1]:.2e}' for r in res.values()]}"
    )

    e1, e2 = abs(u[32] - u[64]), abs(u[64] - u[128])
    ratio = e1 / e2
    order = np.log2(ratio)
    assert ratio > 3.4, f"ratio {ratio:.2f} (order {order:.2f}) — not second order"
    assert ratio < 5.2, (
        f"ratio {ratio:.2f} (order {order:.2f}) — suspiciously fast, check the setup"
    )


# -- Tier C/7: the planar single-mode Duffing LIMIT ------------------------------------------------


@pytest.mark.parametrize("amp", [0.004, 0.008])
def test_amplitude_shift_tracks_the_duffing_limit(amp):
    """The hardening shift lands on KC's Duffing oracle — to a **few percent**, not to cents.

    **Lead with the amplitude shift, never an absolute frequency** (model #9's lesson, inherited):
    a measured ``omega(A)`` carries the linear θ-scheme's temporal dispersion error, which would
    dominate the nonlinear shift being measured. ``omega(A→0)`` carries the *same* error, so the
    difference cancels it and isolates the nonlinear physics.

    The oracle is model #9's, with **``EA -> a = EA - T0``** — the identification, because model
    #10's excess expands to ``a[r^2 v_x/2 + r^4/8 + ...]`` and it is that ``r^4/8`` that *is* KC's
    quartic, recovered locally. Feeding this model's ``EA`` straight into #9's formula instead is a
    ``T0/EA ~ 0.2 %`` error — the identification, not a discrepancy; do not chase it.

    The bar is deliberately loose (``rel=0.05``). This is a **limit** oracle: model #10 leaks a real
    fraction of its motion into the longitudinal field through the phantom term, which KC has no
    ``v`` to hold, so exact agreement would mean the physics under test was *missing*. What is
    asserted is that the residual is small and that the shift is a real, audible number of Hz.

    **Measured, the residual is 0.06 % — ~80x inside that bar, at both amplitudes.** The loose bar
    stays anyway: it is a *limit* oracle and the honest claim is "small", not a number this model is
    obliged to hit. But the smallness is worth reading, because it says *where* the leak is not: the
    phantom term costs ``v_x = O(a r^2 / EA)``, so its back-reaction on the transverse frequency is
    higher-order in the amplitude, not first-order. The leakage is real (that is what
    ``test_geometric_phantom.py`` measures directly, in the field it leaks *into*) — it is simply
    not visible in this projection at these amplitudes. Do not tighten this bar to 0.001 and call
    it an exact oracle; the residual grows with amplitude and the plan's point stands.
    """
    N, m = 32, 1
    s0 = make_geometric_string(N=N, EA=EA_DEFAULT, lam_long=0.5)
    p2 = spatial_eigenvalue_p2(N, L_DEFAULT / N, m)
    w0sq, eps = kc_mode_coefficients(
        c=wave_speed(), kappa=KAPPA_DEFAULT, EA=s0._a, rho=RHO_DEFAULT, p2=p2, L=L_DEFAULT
    )

    def measure(a):
        s = make_geometric_string(N=N, EA=EA_DEFAULT, lam_long=0.5)
        shape = _mode(s, m)
        s.set_state(a * shape)
        return _mode_frequency(s, shape)

    shift_meas = measure(amp) - measure(1e-6)
    shift_oracle = duffing_frequency_shift(amp, w0sq, eps) / (2 * np.pi)
    # Not vacuity theatre: `0 == approx(0, rel=0.05)` PASSES, so a linear string (a = 0) would sail
    # through the assertion below. This is what makes agreement a claim. The bar is 0.5 Hz rather
    # than the 1.0 Hz first written, which excluded amp=0.004 (oracle 0.73 Hz) for no reason:
    # 0.73 Hz at f0 = 100 Hz is ~13 cents -- audible twice over -- and ~700x the zero-crossing
    # estimator's precision. Keeping both shows the shift scaling ~A^2 (0.73 -> 2.90 Hz).
    assert shift_oracle > 0.5, "the shift should be a real, audible number of Hz"
    assert shift_meas == pytest.approx(shift_oracle, rel=0.05)


def _mode_frequency(s, shape, *, n_crossings=6, max_steps=2_000_000):
    """A mode's **nonlinear** frequency (Hz) from descending zero crossings of ``q(t)``.

    Zero crossings, **not** ``spectrum.measure_partials_near``: its window is anchored on the
    *linear* frequency and misses a peak shifted by hardening (models #6/#9's trap, third outing).
    """
    denom = float(np.dot(shape, shape))
    prev = float(np.dot(s.u, shape)) / denom
    times = []
    for n in range(1, max_steps + 1):
        s.step()
        cur = float(np.dot(s.u, shape)) / denom
        if prev > 0.0 >= cur:
            times.append((n - 1 + prev / (prev - cur)) * s.k)
            if len(times) >= n_crossings:
                break
        prev = cur
    if len(times) < 2:
        raise AssertionError("too few zero crossings to measure a frequency")
    return 1.0 / float(np.mean(np.diff(times)))


# -- Tier C/9: model #9 IS the KC limit of model #10 -----------------------------------------------


def test_model_9_is_the_kc_limit_of_model_10():
    """**A genuine cross-model gate.** Quasi-static longitudinal + small slopes ⟹ Kirchhoff-Carrier,
    so model #9 becomes an *oracle* for model #10's transverse limit.

    The identification is ``EA_#9 = (EA - T0)_#10``: model #9's tension is ``T0 + (EA_#9/2L) I``,
    and model #10's excess expands to ``a[r^2 v_x/2 + r^4/8 + ...]`` whose quartic is KC's with
    ``EA -> a``. Feeding model #10's ``EA`` into model #9 instead is a ``T0/EA ~ 0.2 %`` offset.

    Small amplitude and a single mode, deliberately: that is precisely the corner where the two
    models must agree, because it is where the phantom term (quadratic in the transverse field,
    linear in ``v``) is negligible. This is the *agreement* half of the thesis; the phantom tests
    are the *disagreement* half, and the model is only interesting because it has both.
    """
    N, m, amp = 32, 1, 2e-3
    geo = make_geometric_string(N=N, EA=EA_DEFAULT, lam_long=0.5)
    kc = make_tension_string(N=N, EA=geo._a, lam=0.5)  # the identification

    shape_geo, shape_kc = _mode(geo, m), np.sin(m * np.pi * kc.x / kc.L)
    geo.set_state(amp * shape_geo)
    kc.set_state(amp * shape_kc)

    f_geo = _mode_frequency(geo, shape_geo)
    f_kc = _kc_mode_frequency(kc, shape_kc)
    assert f_geo == pytest.approx(f_kc, rel=0.02), (
        f"model #10 ({f_geo:.3f} Hz) should sit on model #9 ({f_kc:.3f} Hz) in the KC corner"
    )


def _kc_mode_frequency(s, shape, *, n_crossings=6, max_steps=2_000_000):
    """``_mode_frequency`` for model #9, whose state is the single field ``s.state``."""
    denom = float(np.dot(shape, shape))
    prev = float(np.dot(s.state, shape)) / denom
    times = []
    for n in range(1, max_steps + 1):
        s.step()
        cur = float(np.dot(s.state, shape)) / denom
        if prev > 0.0 >= cur:
            times.append((n - 1 + prev / (prev - cur)) * s.k)
            if len(times) >= n_crossings:
                break
        prev = cur
    if len(times) < 2:
        raise AssertionError("too few zero crossings to measure a frequency")
    return 1.0 / float(np.mean(np.diff(times)))
