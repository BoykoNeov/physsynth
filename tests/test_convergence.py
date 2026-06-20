"""Grid-convergence order (criterion 3).

At lambda = 1 the explicit scheme is *exact* for any h, so refinement would show nothing. The
convergence rate of the scheme is therefore measured at a fixed lambda < 1, where numerical
dispersion produces an O(h^2) frequency error that must shrink by ~4x each time h is halved. A
single spatial mode is used as the initial condition so the FDTD solution is a clean single tone,
and a higher mode (m = 8) is chosen so the dispersion error stays well above the spectral
measurement floor across the whole refinement.
"""

import numpy as np
import pytest
from helpers import convergence_orders, make_string, wave_speed

from physsynth.analysis import modal, spectrum
from physsynth.core.engine import simulate

LAMBDA = 0.9
MODE = 8
GRIDS = (64, 128, 256)


@pytest.mark.slow
def test_second_order_convergence_at_fixed_lambda():
    c, L = wave_speed(), 1.0
    f_cont = MODE * c / (2 * L)

    errors, step_sizes = [], []
    for N in GRIDS:
        string = make_string(N=N, lam=LAMBDA)
        string.set_state(modal.mode_shape(string.x, L, MODE) * 1e-3)
        res = simulate(
            string, num_steps=int(1.5 * string.fs), pickup_index=int(round(0.413 * N))
        )
        f_det = spectrum.measure_partials_near(res.output, res.fs, np.array([f_cont]))[0]
        errors.append(abs(f_det - f_cont))
        step_sizes.append(L / N)

    errors = np.array(errors)
    orders = convergence_orders(errors, np.array(step_sizes))

    # Error must shrink monotonically, at (close to) second order.
    assert np.all(np.diff(errors) < 0.0), f"errors did not shrink: {errors}"
    assert np.all(orders > 1.7), f"a refinement step fell below 2nd order: {orders}"
    assert 1.85 < orders.mean() < 2.15, f"mean order {orders.mean():.3f} not ~2"


@pytest.mark.slow
def test_detected_frequency_tracks_dispersion_oracle():
    # The simulated frequency should match the closed-form discrete-dispersion oracle (which lies
    # below the continuum value for lambda < 1), confirming the error is genuine dispersion.
    c, L, N = wave_speed(), 1.0, 128
    f_cont = MODE * c / (2 * L)
    string = make_string(N=N, lam=LAMBDA)
    string.set_state(modal.mode_shape(string.x, L, MODE) * 1e-3)
    res = simulate(string, num_steps=int(1.5 * string.fs), pickup_index=int(round(0.413 * N)))
    f_det = spectrum.measure_partials_near(res.output, res.fs, np.array([f_cont]))[0]
    f_oracle = modal.discrete_mode_frequency(c, L, N, LAMBDA, MODE)

    assert f_oracle < f_cont  # dispersion lowers the frequency for lambda < 1
    assert abs(f_det - f_oracle) < 0.01 * (f_cont - f_oracle) + 1e-3
