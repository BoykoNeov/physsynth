"""The fixtures the analysis oracles are frozen at — imported by the generator AND the test.

One module, because the whole point is that the *inputs* are identical on both sides of a
recording made months apart. Anything random here is drawn from a written-down seed; nothing is
read off the clock, the filesystem or the environment.

See ``scripts/freeze_analysis.py`` for the generator and ``tests/test_analysis_frozen.py`` for what
is done with the result. ``docs/dev/rust-migration-plan.md`` §44 is why this exists at all.
"""

from __future__ import annotations

import numpy as np

# -- inputs that are arrays, built deterministically ----------------------------------------------

X8 = np.linspace(0.0, 0.65, 8)
GRID_X, GRID_Y = np.meshgrid(np.linspace(0.0, 0.4, 6), np.linspace(0.0, 0.3, 5))
MODES_2D = [(1, 1), (2, 1), (1, 2), (3, 2), (4, 5)]
MODES_1D = np.array([1, 2, 3, 5, 8, 13], dtype=np.int64)


def _tone(fs: float, n: int, partials, amps, seed: int) -> np.ndarray:
    """A deterministic sum of decaying sinusoids plus a pinch of seeded noise."""
    t = np.arange(n) / fs
    sig = np.zeros(n)
    for f, a in zip(partials, amps, strict=True):
        sig += a * np.sin(2.0 * np.pi * f * t) * np.exp(-3.0 * t)
    sig += 1e-4 * np.random.default_rng(seed).standard_normal(n)
    return sig


SIGNAL_SHORT = _tone(8000.0, 512, (220.0, 441.0, 663.0), (1.0, 0.4, 0.15), 20260903)
SIGNAL_LONG = _tone(16000.0, 4096, (110.0, 221.5, 333.0, 447.0), (1.0, 0.5, 0.2, 0.1), 7)

HESS_P = 0.01 * np.random.default_rng(3).standard_normal(12)
HESS_Z = 1e-3 * np.random.default_rng(4).standard_normal(12)

# -- the cases: key -> (module, function, args, kwargs) -------------------------------------------
#
# Two or three per function wherever a second one asks a different question -- a branch, a sign, a
# degenerate end. The key is what appears in the frozen file and in a failure message, so it is
# spelled `<module>.<function>[#n]` and never renumbered: a key is a permanent name for one
# measurement.

CASES: dict[str, tuple[str, str, tuple, dict]] = {
    # ---- modal: the closed-form oracles ---------------------------------------------------------
    "modal.harmonic_frequencies": ("modal", "harmonic_frequencies", (200.0, 0.65, 12), {}),
    "modal.mode_shape": ("modal", "mode_shape", (X8, 0.65, 3), {}),
    "modal.discrete_mode_frequency": (
        "modal",
        "discrete_mode_frequency",
        (200.0, 0.65, 128, 0.9, 5),
        {},
    ),
    "modal.discrete_mode_frequency#lam1": (
        "modal",
        "discrete_mode_frequency",
        (200.0, 0.65, 128, 1.0, 5),
        {},
    ),
    "modal.inharmonicity_B": ("modal", "inharmonicity_B", (200.0, 0.65, 0.7), {}),
    "modal.stiff_harmonic_frequencies": (
        "modal",
        "stiff_harmonic_frequencies",
        (200.0, 0.65, 0.7, 10),
        {},
    ),
    "modal.discrete_stiff_mode_frequency": (
        "modal",
        "discrete_stiff_mode_frequency",
        (200.0, 0.65, 128, 0.7, 1e-5, 4, 0.5),
        {},
    ),
    "modal.cents": (
        "modal",
        "cents",
        (np.array([220.0, 440.1, 219.9]), np.array([220.0, 440.0, 220.0])),
        {},
    ),
    "modal.rectangular_membrane_freqs": (
        "modal",
        "rectangular_membrane_freqs",
        (200.0, 0.4, 0.3, MODES_2D),
        {},
    ),
    "modal.rectangular_mode_field": (
        "modal",
        "rectangular_mode_field",
        (GRID_X, GRID_Y, 0.4, 0.3, 2, 3),
        {},
    ),
    "modal.rectangular_discrete_eigenvalues": (
        "modal",
        "rectangular_discrete_eigenvalues",
        (0.0125, 32, 24, MODES_2D),
        {},
    ),
    "modal.circular_membrane_freqs": ("modal", "circular_membrane_freqs", (200.0, 0.3, 12), {}),
    "modal.discrete_membrane_eigenfrequency": (
        "modal",
        "discrete_membrane_eigenfrequency",
        (np.array([120.0, 3400.0, 91000.0]), 200.0, 1e-5),
        {},
    ),
    "modal.rectangular_plate_freqs": (
        "modal",
        "rectangular_plate_freqs",
        (1.0, 1.0, 1.0, MODES_2D),
        {},
    ),
    "modal.discrete_plate_eigenfrequency": (
        "modal",
        "discrete_plate_eigenfrequency",
        (np.array([90.0, 1500.0, 40000.0]), 1.0, 1e-5, 0.28),
        {},
    ),
    "modal.orthotropic_plate_freqs": (
        "modal",
        "orthotropic_plate_freqs",
        (1.0, 0.4, 0.3, MODES_2D),
        dict(grain_x=1.0, grain_cross=0.153, grain_y=0.073),
    ),
    "modal.discrete_orthotropic_plate_eigenfrequency": (
        "modal",
        "discrete_orthotropic_plate_eigenfrequency",
        (np.array([40.0, 900.0]), np.array([30.0, 700.0]), 1.0, 1e-5, 0.28),
        dict(grain_x=1.0, grain_cross=0.153, grain_y=0.073),
    ),
    "modal.dirichlet_axis_eigenvalue": (
        "modal",
        "dirichlet_axis_eigenvalue",
        (np.array([1, 2, 5, 9], dtype=np.int64), 0.4, 0.0125),
        {},
    ),
    "modal.free_free_beam_betaL": ("modal", "free_free_beam_betaL", (8,), {}),
    "modal.free_free_beam_freqs": ("modal", "free_free_beam_freqs", (20.0, 1.0, 8), {}),
    "modal.free_plate_ffff_square_lambdas": ("modal", "free_plate_ffff_square_lambdas", (), {}),
    "modal.free_plate_freq_from_lambda": (
        "modal",
        "free_plate_freq_from_lambda",
        (np.array([13.49, 19.79, 24.43]), 1.0, 0.5),
        {},
    ),
    "modal.free_plate_twist_bound": (
        "modal",
        "free_plate_twist_bound",
        (1.0, 0.4, 0.3),
        dict(grain_torsion=0.063),
    ),
    "modal.free_circular_plate_lambda_roots#n0": (
        "modal",
        "free_circular_plate_lambda_roots",
        (0.3, 0),
        {},
    ),
    "modal.free_circular_plate_lambda_roots#n2": (
        "modal",
        "free_circular_plate_lambda_roots",
        (0.3, 2),
        {},
    ),
    "modal.free_circular_plate_lambda_roots#nu0": (
        "modal",
        "free_circular_plate_lambda_roots",
        (0.0, 1),
        {},
    ),
    "modal.free_circular_plate_lambdas": (
        "modal",
        "free_circular_plate_lambdas",
        (),
        dict(nu=0.3, n_modes=7),
    ),
    "modal.free_circular_plate_saddle_bound": (
        "modal",
        "free_circular_plate_saddle_bound",
        (0.3,),
        {},
    ),
    "modal.free_plate_coupling_form": (
        "modal",
        "free_plate_coupling_form",
        (0.0269, 0.025, 16, 12),
        {},
    ),
    "modal.bore_resonance_frequencies": (
        "modal",
        "bore_resonance_frequencies",
        (343.0, 0.5, 8),
        {},
    ),
    "modal.bore_resonance_frequencies#open": (
        "modal",
        "bore_resonance_frequencies",
        (343.0, 0.5, 8),
        dict(boundary="open-open"),
    ),
    "modal.discrete_bore_eigenfrequency": (
        "modal",
        "discrete_bore_eigenfrequency",
        (np.array([4.0e6, 3.6e7]), 1.0 / 48000.0),
        {},
    ),
    "modal.discrete_beam_eigenfrequency": (
        "modal",
        "discrete_beam_eigenfrequency",
        (np.array([500.0, 30000.0]), 20.0, 1e-5, 0.28),
        {},
    ),
    # ---- damping --------------------------------------------------------------------------------
    "damping.spatial_eigenvalue_p2": ("damping", "spatial_eigenvalue_p2", (128, 0.65 / 128, 7), {}),
    "damping.modal_loss_rate_continuum": (
        "damping",
        "modal_loss_rate_continuum",
        (200.0, 0.65, 0.7, 1.2, 5e-5, 4),
        {},
    ),
    "damping.discrete_damped_mode_decay": (
        "damping",
        "discrete_damped_mode_decay",
        (200.0, 0.65, 128, 0.7, 1e-5, 0.5, 1.2, 5e-5, 4),
        {},
    ),
    "damping.discrete_damped_mode_rate": (
        "damping",
        "discrete_damped_mode_rate",
        (200.0, 0.65, 128, 0.7, 1e-5, 0.5, 1.2, 5e-5, 4),
        {},
    ),
    "damping.discrete_damped_mode_is_underdamped": (
        "damping",
        "discrete_damped_mode_is_underdamped",
        (200.0, 0.65, 128, 0.7, 1e-5, 0.5, 1.2, 5e-5, 4),
        {},
    ),
    # ... and one that really is OVERDAMPED. The first draft of this row used
    # `sigma1 = 5e-2, m = 60` and came back `True` like its sibling, so the pair covered one branch
    # twice. A frozen bool is a *decision*: a case that never takes the other arm asserts nothing
    # about the arm it is there for.
    "damping.discrete_damped_mode_is_underdamped#over": (
        "damping",
        "discrete_damped_mode_is_underdamped",
        (200.0, 0.65, 128, 0.7, 1e-5, 0.5, 5000.0, 0.5, 1),
        {},
    ),
    "damping.loss_coefficients_from_T60": (
        "damping",
        "loss_coefficients_from_T60",
        (200.0, 0.65, 0.7, 200.0, 6.0, 2000.0, 1.5),
        {},
    ),
    # ---- dispersion -----------------------------------------------------------------------------
    "dispersion.dispersion_frequencies": (
        "dispersion",
        "dispersion_frequencies",
        (200.0, 0.65, 128, 0.9, MODES_1D),
        {},
    ),
    "dispersion.stiff_dispersion_frequencies": (
        "dispersion",
        "stiff_dispersion_frequencies",
        (200.0, 0.65, 128, 0.7, 1e-5, 0.5, MODES_1D),
        {},
    ),
    "dispersion.phase_velocity": (
        "dispersion",
        "phase_velocity",
        (np.array([153.8, 307.7, 461.5, 769.2, 1230.7, 2000.0]), 0.65, MODES_1D),
        {},
    ),
    # ---- duffing --------------------------------------------------------------------------------
    "duffing.kc_mode_coefficients": (
        "duffing",
        "kc_mode_coefficients",
        (),
        dict(c=200.0, kappa=0.7, EA=1.0e5, rho=0.005, p2=233.4, L=0.65),
    ),
    "duffing.kc_mode_stretch": ("duffing", "kc_mode_stretch", (1e-3,), dict(p2=233.4, L=0.65)),
    "duffing.duffing_elliptic_parameter": (
        "duffing",
        "duffing_elliptic_parameter",
        (0.07, (2.0 * np.pi * 220.0) ** 2, 5e7),
        {},
    ),
    "duffing.duffing_frequency": (
        "duffing",
        "duffing_frequency",
        (0.07, (2.0 * np.pi * 220.0) ** 2, 5e7),
        {},
    ),
    "duffing.duffing_frequency#eps0": (
        "duffing",
        "duffing_frequency",
        (0.07, (2.0 * np.pi * 220.0) ** 2, 0.0),
        {},
    ),
    "duffing.duffing_frequency_shift": (
        "duffing",
        "duffing_frequency_shift",
        (0.07, (2.0 * np.pi * 220.0) ** 2, 5e7),
        {},
    ),
    "duffing.duffing_frequency_expansion": (
        "duffing",
        "duffing_frequency_expansion",
        (0.07, (2.0 * np.pi * 220.0) ** 2, 5e7),
        {},
    ),
    "duffing.duffing_displacement": (
        "duffing",
        "duffing_displacement",
        (np.linspace(0.0, 0.01, 17), 0.07, (2.0 * np.pi * 220.0) ** 2, 5e7),
        {},
    ),
    # ---- spectrum (unit 11) ---------------------------------------------------------------------
    "spectrum.magnitude_spectrum": ("spectrum", "magnitude_spectrum", (SIGNAL_SHORT, 8000.0), {}),
    "spectrum.magnitude_spectrum#pad4": (
        "spectrum",
        "magnitude_spectrum",
        (SIGNAL_SHORT, 8000.0),
        dict(zero_pad_factor=4),
    ),
    "spectrum.detect_peaks": ("spectrum", "detect_peaks", (SIGNAL_LONG, 16000.0, 4), {}),
    "spectrum.detect_peaks#sep": (
        "spectrum",
        "detect_peaks",
        (SIGNAL_LONG, 16000.0, 3),
        dict(f_min=50.0, min_separation_hz=80.0),
    ),
    "spectrum.measure_partials_near": (
        "spectrum",
        "measure_partials_near",
        (SIGNAL_LONG, 16000.0, np.array([110.0, 220.0, 330.0, 440.0])),
        {},
    ),
    "spectrum.measure_partials_near#window": (
        "spectrum",
        "measure_partials_near",
        (SIGNAL_LONG, 16000.0, np.array([110.0, 221.0, 333.0])),
        dict(search_hz=12.0),
    ),
    # ---- rotating_wave --------------------------------------------------------------------------
    "rotating_wave.planar_hessian_cells": (
        "rotating_wave",
        "planar_hessian_cells",
        (HESS_P, HESS_Z, 1.0e5 - 200.0),
        {},
    ),
    "rotating_wave.kc_circular_frequency": (
        "rotating_wave",
        "kc_circular_frequency",
        (),
        dict(omega0_sq=(2.0 * np.pi * 220.0) ** 2, eps=5e7, amplitude=1e-3),
    ),
    "rotating_wave.solve_rotating_wave": (
        "rotating_wave",
        "solve_rotating_wave",
        (),
        dict(
            L=1.0,
            T=200.0,
            rho=0.005,
            EA=1.0e5,
            fs=48000.0,
            N=24,
            theta=0.28,
            amplitude=2e-3,
            mode=1,
            kappa=0.0,
        ),
    ),
    "rotating_wave.solve_rotating_wave#stiff": (
        "rotating_wave",
        "solve_rotating_wave",
        (),
        dict(
            L=1.0,
            T=200.0,
            rho=0.005,
            EA=1.0e5,
            fs=48000.0,
            N=24,
            theta=0.28,
            amplitude=3e-3,
            mode=2,
            kappa=2.0,
        ),
    ),
}


# The one case whose input is another function's OUTPUT, so it cannot be a literal argument tuple:
# `rotating_wave_history` takes a solved `RotatingWave`. It is built on the same side of the
# comparison as the call, which is the point -- freezing the history against a wave solved by the
# *other* implementation would be measuring the solver twice.
HISTORY_KEY = "rotating_wave.rotating_wave_history"
HISTORY_WAVE_KWARGS = dict(
    L=1.0,
    T=200.0,
    rho=0.005,
    EA=1.0e5,
    fs=48000.0,
    N=24,
    theta=0.28,
    amplitude=2e-3,
    mode=1,
    kappa=0.0,
)
HISTORY_FS = 4000.0

KEYS = [*CASES, HISTORY_KEY]


# The `RotatingWave` field this freeze must NOT record, and why it takes a deliberate line of code
# to avoid it.
#
# `flatten` collects every integer in a returned tuple and `test_analysis_frozen.py` compares them
# EXACTLY -- which is right for a mode label, a multiplicity, an FFT length or a root count, and
# wrong for exactly one field here. `iterations` is how many Newton steps the continuation took,
# and ledger #33 settled that it is **not a comparable quantity**: when only the root survives, two
# implementations may take visibly different paths to the same answer, and 17 of 108 fixtures did.
# `tests/test_rust_parity_rotating_wave.py` carried a test whose whole job was to say so --
# `test_the_iteration_count_is_not_compared_and_this_is_the_witness`, "kept as an executable
# statement of what this file declines to assert" -- and that file was deleted in the same batch
# that created this one. Freezing the count would quietly reinstate the assertion its witness
# existed to forbid, and it would have gone green, because neither fixture here happens to be one
# of the ones that differ.
#
# Blanked in `call` rather than in the test, so the generator and the test see the same object and
# a future regeneration cannot reintroduce it.
NOT_COMPARABLE = "iterations"


def call(key, resolve):
    """Run one case. ``resolve(module, name)`` hands back the implementation to use."""
    if key == HISTORY_KEY:
        wave = resolve("rotating_wave", "solve_rotating_wave")(**HISTORY_WAVE_KWARGS)
        return resolve("rotating_wave", "rotating_wave_history")(wave, fs=HISTORY_FS)
    module, func, args, kwargs = CASES[key]
    out = resolve(module, func)(*args, **kwargs)
    if func == "solve_rotating_wave":
        out = out._replace(**{NOT_COMPARABLE: 0})
    return out


def flatten(value):
    """A case's return value as ``(structure, floats, ints)``.

    Structure is compared **exactly** and the floats to a tolerance -- which is the split that
    matters, because a search that returns a different number of roots is not a small error and no
    tolerance can describe it.
    """
    floats: list[float] = []
    ints: list[int] = []

    def walk(x) -> str:
        if isinstance(x, (bool, np.bool_)):
            ints.append(int(x))
            return "b"
        if isinstance(x, (int, np.integer)):
            ints.append(int(x))
            return "i"
        if isinstance(x, (float, np.floating)):
            floats.append(float(x))
            return "f"
        if isinstance(x, np.ndarray):
            if x.dtype.kind in "iub":
                ints.extend(int(t) for t in x.ravel())
            else:
                floats.extend(float(t) for t in x.ravel())
            return f"a{x.dtype.kind}{list(x.shape)}"
        if isinstance(x, (tuple, list)):
            return "(" + ",".join(walk(e) for e in x) + ")"
        raise TypeError(f"the frozen cases do not know how to flatten {type(x).__name__}")

    return walk(value), floats, ints
