"""Difference operators and the discrete inner product (HANDOFF Appendix A)."""

import numpy as np

from physsynth.core import operators as ops


def test_inner_and_norm_match_definition():
    h = 0.01
    f = np.array([1.0, -2.0, 3.0, 0.5])
    g = np.array([0.5, 1.0, -1.0, 2.0])
    assert np.isclose(ops.inner(f, g, h), h * float(np.dot(f, g)))
    assert np.isclose(ops.norm2(f, h), h * float(np.dot(f, f)))
    assert ops.norm2(f, h) >= 0.0


def test_forward_difference_exact_on_linear_ramp():
    # delta_x+ of a linear function is its (constant) slope, exactly.
    h = 0.25
    x = np.arange(0, 5) * h
    slope = 3.0
    u = slope * x + 1.0
    d = ops.delta_x_forward(u, h)
    assert np.allclose(d, slope)
    assert d.shape == (len(u) - 1,)


def test_second_difference_exact_eigenvector():
    # sin(m pi l / N) is an exact eigenvector of delta_xx with eigenvalue
    # -(4/h^2) sin^2(m pi / 2N). This pins the operator down without any continuum approximation.
    N = 64
    L = 1.0
    h = L / N
    x = np.linspace(0.0, L, N + 1)
    m = 5
    v = np.sin(m * np.pi * x / L)
    lhs = ops.delta_xx(v, h)  # interior nodes l = 1 .. N-1
    eig = -(4.0 / h**2) * np.sin(m * np.pi / (2 * N)) ** 2
    assert np.allclose(lhs, eig * v[1:-1], atol=1e-10)


def test_second_difference_exact_on_quadratic():
    # delta_xx of a quadratic is its constant curvature, exactly.
    h = 0.1
    x = np.arange(0, 7) * h
    u = 2.0 * x**2 - x + 5.0
    assert np.allclose(ops.delta_xx(u, h), 4.0)
