"""Frequency-dependent damping oracles for the damped stiff string (model #3, HANDOFF §5 row 3).

The damped stiff string adds a frequency-dependent loss term ``+2 sigma1 u_txx`` to model #2:

    u_tt = c^2 u_xx - kappa^2 u_xxxx - 2 sigma0 u_t + 2 sigma1 u_txx.

In the modal domain ``delta_xx -> -p^2`` so the loss becomes a velocity term with a mode-dependent
coefficient: a mode decays in **energy** as ``exp(-2 sigma_eff t)`` with ``sigma_eff = sigma0 +
sigma1 * (wavenumber)^2`` -- high modes die faster, the physically-correct ordering real strings
have (and model #2's frequency-*independent* loss got backwards).

This module is the single source of truth for that decay relation, in three layers:

- the **continuum** rate ``2 sigma_eff`` from the true wavenumber ``beta = m pi / L`` -- the
  physics the scheme should approach as the grid refines;
- the **discrete** per-step energy factor ``g_m`` of the implicit theta-scheme -- the *exact*
  thing a simulated single mode decays by, the tight test oracle. It includes the theta-scheme's
  ``1/(1+theta Q k^2)`` rate-suppression (the model #2 artifact), so the per-mode decay test
  stays tight without pretending the artifact is gone;
- the **T60 -> (sigma0, sigma1)** inversion -- a luthier specifies decay times, this returns the
  loss coefficients (continuum mapping; the round-trip lands a few % off, a documented demo).

Pure NumPy; independent of the core (an oracle must not depend on the code it validates).
See ``docs/dev/damped-string-plan.md``.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "spatial_eigenvalue_p2",
    "modal_loss_rate_continuum",
    "discrete_damped_mode_decay",
    "discrete_damped_mode_rate",
    "discrete_damped_mode_is_underdamped",
    "loss_coefficients_from_T60",
    "T60_SECONDS_PER_RATE",
]

# A 60 dB amplitude drop is a factor 10^-3; amplitude ~ exp(-sigma_eff t), so
# exp(-sigma_eff T60) = 10^-3  =>  T60 = 3 ln(10) / sigma_eff. (Energy, being amplitude^2, drops
# 60 dB in T60/2, but T60 is conventionally an amplitude/level figure -- keep it amplitude-based.)
T60_SECONDS_PER_RATE = 3.0 * np.log(10.0)  # ~= 6.9078; T60 = this / sigma_eff


def spatial_eigenvalue_p2(N: int, h: float, m: int) -> float:
    """Discrete eigenvalue ``p^2 = (4/h^2) sin^2(m pi / 2N)`` of ``-delta_xx`` for mode ``m``.

    ``sin(m pi x / L)`` is an exact eigenvector of the second-difference matrix with eigenvalue
    ``-p^2`` (and of the biharmonic ``(delta_xx)^2`` with ``+p^4``). Tends to the continuum
    ``beta^2 = (m pi / L)^2`` as ``h -> 0``.
    """
    return float((4.0 / (h * h)) * np.sin(m * np.pi / (2 * N)) ** 2)


def modal_loss_rate_continuum(
    c: float, L: float, kappa: float, sigma0: float, sigma1: float, m: int
) -> float:
    """Continuum **energy** decay rate ``2 sigma_eff = 2(sigma0 + sigma1 beta_m^2)`` (1/s).

    ``E_m(t) ~ E_m(0) exp(-rate * t)`` with ``beta_m = m pi / L`` the true wavenumber. ``kappa`` is
    unused here (the loss rate depends on the wavenumber, not the stiffness; stiffness enters only
    via the oscillation frequency) -- kept in the signature so callers pass the full parameter set
    symmetrically with the discrete oracle. The discrete scheme approaches this as the grid refines.
    """
    beta2 = (m * np.pi / L) ** 2
    return float(2.0 * (sigma0 + sigma1 * beta2))


def _decay_roots_ac(
    c: float, L: float, N: int, kappa: float, k: float, theta: float,
    sigma0: float, sigma1: float, m: int,
) -> tuple[float, float, float, float]:
    """Return ``(a, b, cc, Q)`` of the per-mode characteristic ``a z^2 + b z + cc = 0``.

    From ``u^n = z^n sin(m pi x / L)`` in the implicit theta-scheme with both loss terms:
    ``a = 1 + theta k^2 Q + sigma_eff k``, ``b = -2 + (1-2theta) k^2 Q``,
    ``cc = 1 + theta k^2 Q - sigma_eff k``, with ``Q = c^2 p^2 + kappa^2 p^4`` and
    ``sigma_eff = sigma0 + sigma1 p^2``.
    """
    h = L / N
    p2 = spatial_eigenvalue_p2(N, h, m)
    Q = c * c * p2 + kappa * kappa * p2 * p2
    sigma_eff = sigma0 + sigma1 * p2
    base = 1.0 + theta * k * k * Q
    a = base + sigma_eff * k
    b = -2.0 + (1.0 - 2.0 * theta) * k * k * Q
    cc = base - sigma_eff * k
    return a, b, cc, Q


def discrete_damped_mode_decay(
    c: float, L: float, N: int, kappa: float, k: float, theta: float,
    sigma0: float, sigma1: float, m: int,
) -> float:
    """Exact per-step **energy** decay factor ``g_m = cc/a`` of mode ``m`` (implicit theta-scheme).

    For an underdamped mode the two roots are a complex-conjugate pair with ``|z|^2 = cc/a``, so the
    energy (amplitude squared) is multiplied by

        g_m = (1 + theta k^2 Q - sigma_eff k) / (1 + theta k^2 Q + sigma_eff k)  in (0, 1]

    every step (``E_m(t) = E_m(0) g_m^(t/k)``). This is the *exact* thing a simulated single mode
    decays by -- the tight test oracle. It keeps the theta-scheme's ``1/(1+theta Q k^2)`` factor
    (the model #2 high-mode rate-suppression), so the test is tight without claiming that
    artifact is gone. ``sigma0 = sigma1 = 0`` gives ``g_m = 1`` (lossless).
    """
    a, _b, cc, _Q = _decay_roots_ac(c, L, N, kappa, k, theta, sigma0, sigma1, m)
    return float(cc / a)


def discrete_damped_mode_rate(
    c: float, L: float, N: int, kappa: float, k: float, theta: float,
    sigma0: float, sigma1: float, m: int,
) -> float:
    """Per-second **energy** decay rate ``Gamma_m = -(1/k) ln g_m`` (1/s); 0 when lossless.

    Tends to the continuum ``2(sigma0 + sigma1 beta^2)`` as ``k, h -> 0``. Not monotone in ``m``
    forever: the numerator grows like ``p^2`` but the ``1/(1+theta Q k^2)`` denominator like ``p^4``
    with stiffness, so the rate turns over at very high modes (see the plan).
    """
    g = discrete_damped_mode_decay(c, L, N, kappa, k, theta, sigma0, sigma1, m)
    if g <= 0.0:
        return float("inf")  # overdamped to a sign flip -- not a clean exponential; guard caller
    return float(-np.log(g) / k)


def discrete_damped_mode_is_underdamped(
    c: float, L: float, N: int, kappa: float, k: float, theta: float,
    sigma0: float, sigma1: float, m: int,
) -> bool:
    """True if mode ``m`` is underdamped (complex-conjugate roots, ``b^2 - 4 a cc < 0``).

    The ``g_m = cc/a`` energy oracle assumes a clean oscillatory decay; a heavily overdamped mode
    has two distinct real roots and is not a single exponential. Tests pick modes where this holds.
    """
    a, b, cc, _Q = _decay_roots_ac(c, L, N, kappa, k, theta, sigma0, sigma1, m)
    return bool(b * b - 4.0 * a * cc < 0.0)


def loss_coefficients_from_T60(
    c: float, L: float, kappa: float,
    f1: float, T60_1: float, f2: float, T60_2: float,
) -> tuple[float, float]:
    """Invert two ``(frequency, T60)`` targets to ``(sigma0, sigma1)`` (continuum mapping).

    A luthier specifies decay times; this returns the loss coefficients. With amplitude decay
    ``exp(-sigma_eff t)`` a 60 dB drop takes ``T60 = T60_SECONDS_PER_RATE / sigma_eff``, so
    ``sigma_eff(f) = T60_SECONDS_PER_RATE / T60``. Using ``beta^2(omega)`` from the continuum
    dispersion ``omega^2 = c^2 beta^2 + kappa^2 beta^4`` gives the linear system

        [1, beta1^2; 1, beta2^2] @ [sigma0; sigma1] = [sigma_eff1; sigma_eff2].

    **Pick targets with T60 decreasing in frequency** (``f2 > f1`` and ``T60_2 < T60_1``) so
    ``sigma0, sigma1 >= 0``. This is a *continuum* inversion: a simulation decays at the discrete
    rate (with the theta-factor), so the round-trip lands a few % off -- a physics demo, not a
    tight test. Raises ``ValueError`` if the targets are degenerate (equal frequencies) or imply a
    negative coefficient.
    """
    if f1 <= 0 or f2 <= 0 or T60_1 <= 0 or T60_2 <= 0:
        raise ValueError("frequencies and T60s must be positive.")
    if f1 == f2:
        raise ValueError("need two distinct frequencies to separate sigma0 from sigma1.")

    def beta2_of_f(f: float) -> float:
        omega2 = (2.0 * np.pi * f) ** 2
        if kappa == 0.0:
            return omega2 / (c * c)
        # solve kappa^2 beta^4 + c^2 beta^2 - omega^2 = 0 for beta^2 (positive root).
        disc = c**4 + 4.0 * kappa**2 * omega2
        return (-(c**2) + np.sqrt(disc)) / (2.0 * kappa**2)

    b1, b2 = beta2_of_f(f1), beta2_of_f(f2)
    s1 = T60_SECONDS_PER_RATE / T60_1
    s2 = T60_SECONDS_PER_RATE / T60_2
    # Solve the 2x2 system for (sigma0, sigma1).
    sigma1 = (s2 - s1) / (b2 - b1)
    sigma0 = s1 - sigma1 * b1
    if sigma0 < 0.0 or sigma1 < 0.0:
        raise ValueError(
            f"targets imply negative loss (sigma0={sigma0:.4g}, sigma1={sigma1:.4g}); "
            "pick T60 decreasing with frequency."
        )
    return float(sigma0), float(sigma1)
