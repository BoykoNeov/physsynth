"""Shared test fixtures and physical parameters for the validation harness.

A single canonical string (L=1 m, T=200 N, rho=0.005 kg/m -> c=200 m/s, f1=100 Hz) is reused so the
expected frequencies are easy to reason about.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from physsynth.analysis import modal
from physsynth.analysis.horizon import pitch_horizon
from physsynth.core.airbox import (
    AirBox,
    RoomLoadedBody,
)
from physsynth.core.body import ModalBody
from physsynth.core.bore import C0_AIR, RHO0_AIR
from physsynth.core.membrane import Domain, Membrane
from physsynth.core.plate import THETA_DEFAULT as PLATE_THETA_DEFAULT  # noqa: F401 (re-export)
from physsynth.core.plate import Plate
from physsynth.core.radiation import (
    AirRadiation,
    RadiatedBody,
    RationalAirLoad,
    ReactiveRadiatedBody,
)
from physsynth.core.string_damped import DampedStiffString
from physsynth.core.string_stiff import THETA_DEFAULT

L_DEFAULT = 1.0
T_DEFAULT = 200.0
RHO_DEFAULT = 0.005  # -> c = sqrt(T/rho) = 200 m/s, fundamental f1 = c/(2L) = 100 Hz

# Membrane (model #4): areal density chosen so c = sqrt(T/rho) = 200 m/s again (same nice numbers).
RHO_AREAL_DEFAULT = 0.005  # kg/m^2  -> c = 200 m/s with T_DEFAULT = 200 N/m
RADIUS_DEFAULT = 0.5  # m  -> circular fundamental f_01 = c*j_{0,1}/(2*pi*a) ~ 153.1 Hz
KAPPA_DEFAULT = 2.0  # -> B = pi^2 kappa^2 / (c^2 L^2) ~ 9.87e-4, a piano-ish inharmonicity

# Plate (model #5): stiffness kappa = sqrt(D/rho_s) (m^2/s). On the 1x1 m plate the fundamental is
# f_11 = (pi/2) kappa [1 + 1] = pi kappa ~ 62.8 Hz at kappa = 20, with modes spreading
# quadratically.
KAPPA_PLATE_DEFAULT = 20.0
# Plate "Courant" number mu = kappa k / h^2: the EXPLICIT-scheme stability parameter (explicit needs
# mu <= 1/4). The implicit theta-scheme has no limit, so the default sits well past 1/4 -- a regime
# the explicit plate could not run. make_plate solves fs from mu: fs = kappa / (mu h^2).
MU_PLATE_DEFAULT = 2.0


# ARPACK start vector. `eigsh` without `v0` draws a RANDOM one, so every eigensolve in the suite is
# non-reproducible run to run -- measured on the free-free beam: the elastic eigenvalues wobble by
# ~1e-12 relative and their eigenvectors by ~5e-11, and the two RIGID-BODY modes (mu ~ 0, exactly
# degenerate) come back as an arbitrary basis of the {1, x} nullspace, differing by ~1e-1. That is
# harmless against a 5-cent bar, but an oracle that is not bit-reproducible reads as a port bug the
# first time a comparison is tightened, and an eigenvector fed to `set_state` makes a genuinely
# different trajectory (rust-migration-plan.md Sec 7). Pinned here so it cannot.
ARPACK_SEED = 20260828


def arpack_v0(op) -> NDArray[np.float64]:
    """A fixed ARPACK start vector sized for ``op`` (a square matrix, or its dimension).

    Uniform doubles from a seeded PCG64 stream: deterministic across platforms and NumPy versions,
    generic (no symmetry, so it is orthogonal to no eigenvector -- unlike the obvious ``arange``,
    which is antisymmetric about the centre of a symmetric operator and would starve every
    symmetric mode), and free of any transcendental, whose CPU-dispatched NumPy loop would put the
    start vector itself back in the class of things that vary by machine (Sec 22.1).
    """
    n = int(op) if isinstance(op, (int, np.integer)) else int(op.shape[0])
    return np.random.default_rng(ARPACK_SEED).random(n)


def wave_speed(T: float = T_DEFAULT, rho: float = RHO_DEFAULT) -> float:
    return float(np.sqrt(T / rho))


def make_damped_string(
    *,
    N: int = 100,
    lam: float = 1.0,
    kappa: float = KAPPA_DEFAULT,
    sigma0: float = 0.0,
    sigma1: float = 0.0,
    theta: float = THETA_DEFAULT,
    L: float = L_DEFAULT,
    T: float = T_DEFAULT,
    rho: float = RHO_DEFAULT,
) -> DampedStiffString:
    """Build a damped stiff string (model #3) at Courant number ``lam`` via ``fs = c N / (L lam)``.

    ``sigma0`` is the frequency-independent loss (model #2's ``sigma``), ``sigma1`` the
    frequency-dependent loss (the new term). ``lam > 1`` is allowed (unconditionally stable).
    """
    c = wave_speed(T, rho)
    fs = c * N / (L * lam)
    return DampedStiffString(
        L=L, T=T, rho=rho, fs=fs, N=N, kappa=kappa, sigma0=sigma0, sigma1=sigma1, theta=theta
    )


# =====================================================================================
# The resolution horizon: what stayed here, and why only this
# =====================================================================================
#
# The seven primitives that used to live here -- `pitch_error_cents`, `pitch_horizon`,
# `sinc_horizon_fraction`, `mode_family`, `mode_block`, `cancellation_courant`, `block_weight` --
# were promoted into `physsynth/analysis/horizon.py` and are now Rust, per
# `docs/dev/resolution-horizon-plan.md` section 11. Callers import them from the package
# directly;
# there is deliberately NO re-export here. A re-export would be an unused import that only
# `__all__` keeps alive (a `ruff --fix` hazard the migration hit as ledger #66), and worse, it
# would leave two routes to one name with nothing saying which is canonical.
#
# `spatial_operator_horizon` did NOT go with them, and section 7.7 is the reason. It takes `(N,
# kappa)` and hands back "the horizon", which reads like a general answer and is not one: it
# hardcodes `L_DEFAULT`, `wave_speed()` and a 1-D Dirichlet second difference, so it silently
# answers about the default string whatever the caller had in mind. That is exactly the shape
# `mode_family` was designed against -- index-side only, geometry left to the caller. A fixture
# that assumes the canonical string is honest in the test folder and would be a lie in the library,
# so it stays on this side of the line and builds its frequencies from the promoted `pitch_horizon`
# like any other caller.


def spatial_operator_horizon(N: int, kappa: float, cents: float = 5.0) -> tuple[int, bool]:
    """The horizon of the *spatial* operator alone, with no timestep in it at all.

    ``k`` appears nowhere here: this is the eigenvalue error of the second difference (and of the
    biharmonic built from it), so it is what remains when the timestep is refined to nothing. For
    the implicit theta-scheme family that makes it a genuine **floor** -- the time error and the
    space error both flatten the pitch, so they compound and no sample rate passes this line. For
    the explicit family it is not a floor at all: there the time error is *sharp* and cancels the
    space droop exactly at the magic Courant number.

    About the canonical string, always. See the note above for why that keeps it here.
    """
    h = L_DEFAULT / N
    modes = np.arange(1, N)
    p2_disc = np.array([modal.dirichlet_axis_eigenvalue(int(m), L_DEFAULT, h) for m in modes])
    p2_cont = (modes * np.pi / L_DEFAULT) ** 2
    c = wave_speed()
    w_disc = np.sqrt(c * c * p2_disc + kappa * kappa * p2_disc * p2_disc)
    w_cont = np.sqrt(c * c * p2_cont + kappa * kappa * p2_cont * p2_cont)
    return pitch_horizon(w_disc, w_cont, cents)


def make_membrane(
    *,
    domain: Domain,
    N: int,
    lam: float = 0.7,
    sigma: float = 0.0,
    T: float = T_DEFAULT,
    rho: float = RHO_AREAL_DEFAULT,
    Lx: float = 1.0,
    Ly: float = 1.0,
    radius: float = RADIUS_DEFAULT,
) -> Membrane:
    """Build a membrane whose Courant number is exactly ``lam`` by choosing ``fs = c / (lam h)``.

    ``h`` is fixed by the geometry and ``N`` (``Lx/N`` for a rectangle, ``2 radius/N`` for a disk),
    so the sample rate is solved for to hit the target ``lam``. The 2D CFL ceiling is ``1/sqrt(2)``.
    """
    c = float(np.sqrt(T / rho))
    h = Lx / N if domain == "rectangle" else 2.0 * radius / N
    fs = c / (lam * h)
    if domain == "rectangle":
        return Membrane(domain=domain, T=T, rho=rho, fs=fs, N=N, Lx=Lx, Ly=Ly, sigma=sigma)
    return Membrane(domain=domain, T=T, rho=rho, fs=fs, N=N, radius=radius, sigma=sigma)


def make_plate(
    *,
    N: int,
    mu: float = MU_PLATE_DEFAULT,
    kappa: float = KAPPA_PLATE_DEFAULT,
    sigma: float = 0.0,
    theta: float = THETA_DEFAULT,
    Lx: float = 1.0,
    Ly: float = 1.0,
    rho: float = RHO_AREAL_DEFAULT,
) -> Plate:
    """Build a simply-supported plate at plate-Courant number ``mu = kappa k / h^2``.

    ``h = Lx/N`` is fixed by the geometry, so the sample rate is solved for to hit the target
    ``mu``: ``fs = kappa / (mu h^2)``. There is no CFL ceiling (the implicit theta-scheme is
    unconditionally stable for ``theta >= 1/4``), so ``mu`` above the explicit bound ``1/4`` is a
    feature -- a coarse-grid / large-timestep regime the explicit plate could not run. Smaller
    ``mu`` means a finer timestep (less numerical dispersion), used for the tight modal tests.
    """
    h = Lx / N
    fs = kappa / (mu * h * h)
    return Plate(Lx=Lx, Ly=Ly, kappa=kappa, rho=rho, fs=fs, N=N, sigma=sigma, theta=theta)


# Modal body (body/radiation node): a few guitar-top-ish modes. fs is high (audio rate) so every
# mode sits well under the modal CFL omega*k < 2.
BODY_FREQS_DEFAULT = np.array([110.0, 196.0, 261.0, 440.0])  # Hz


def make_body(
    *,
    freqs: np.ndarray = BODY_FREQS_DEFAULT,
    fs: float = 48000.0,
    sigmas: np.ndarray | float = 0.0,
    masses: np.ndarray | float = 1.0,
    phi: np.ndarray | float = 1.0,
) -> ModalBody:
    """Build a modal body (soundboard) at sample rate ``fs`` with the given modal set."""
    return ModalBody(freqs=freqs, fs=fs, sigmas=sigmas, masses=masses, phi=phi)


# Air radiation (the "air" node): a listener 1 m away in ambient air. fs matches the body defaults.
RADIATION_DISTANCE_DEFAULT = 1.0  # m


def make_radiation(
    *,
    fs: float = 48000.0,
    distance: float = RADIATION_DISTANCE_DEFAULT,
    retarded: bool = True,
) -> AirRadiation:
    """Build a monopole far-field radiation node at sample rate ``fs`` for a listener ``distance`` m
    away (ambient-air ``rho0``/``c0`` defaults)."""
    return AirRadiation(fs=fs, distance=distance, retarded=retarded)


# Radiation load (batch 2): a modal body loaded by its own far-field radiation resistance. R here is
# an ACOUSTIC resistance (Pa·s/m^3), sized so the body audibly sheds energy over a couple thousand
# steps (moderate per-step R*G ~ 0.05) — the exact energy identity holds for any R, this just makes
# the decay easy to see. The stability test overrides it with a deliberately enormous value.
R_RADIATION_DEFAULT = 2000.0  # Pa·s/m^3


def make_radiated_body(
    *,
    freqs: np.ndarray = BODY_FREQS_DEFAULT,
    fs: float = 48000.0,
    sigmas: np.ndarray | float = 0.0,
    masses: np.ndarray | float = 1.0,
    phi: np.ndarray | float = 1.0,
    radiation: np.ndarray | float | None = None,
    R: float = R_RADIATION_DEFAULT,
) -> RadiatedBody:
    """Build a modal body loaded by its own radiation resistance ``R`` (the back-reaction).

    ``sigmas = 0`` (default) keeps the modes lossless so the *only* energy sink is the radiation
    channel — then ``body.energy() + radiated_energy`` is conserved and ``radiated_energy`` accounts
    for all the shed energy. ``R = 0`` decouples the air (bit-identical to :func:`make_body`)."""
    body = ModalBody(
        freqs=freqs, fs=fs, sigmas=sigmas, masses=masses, phi=phi, radiation=radiation
    )
    return RadiatedBody(body=body, R=R)


# Frequency-dependent radiation load (batch 3): resistance R in PARALLEL with the radiation mass
# M_a, the exact first-order (pulsating-sphere) impedance. The default M_a puts the trapezoid's
# k R / (2 M_a) around 0.1 — the reactance is genuinely in play without being stiff. M_a = inf is
# the constant-R load (batch 2, bit-identical); RationalAirLoad.from_sphere gives the physically
# consistent pair for a given radius.
M_A_RADIATION_DEFAULT = 0.2  # kg/m^4
SPHERE_RADIUS_DEFAULT = 0.05  # m — a 5 cm pulsating sphere (ka = 1 at ~1.1 kHz)


def make_reactive_body(
    *,
    freqs: np.ndarray = BODY_FREQS_DEFAULT,
    fs: float = 48000.0,
    sigmas: np.ndarray | float = 0.0,
    masses: np.ndarray | float = 1.0,
    phi: np.ndarray | float = 1.0,
    radiation: np.ndarray | float | None = None,
    R: float = R_RADIATION_DEFAULT,
    M_a: float = M_A_RADIATION_DEFAULT,
) -> ReactiveRadiatedBody:
    """Build a modal body loaded by the rational (frequency-dependent) radiation impedance.

    ``sigmas = 0`` (default) keeps the modes lossless so the air is the only channel and
    ``body.energy() + stored + radiated`` is conserved. ``R = 0`` decouples the air entirely
    (bit-identical to :func:`make_body`); ``M_a = inf`` collapses to :func:`make_radiated_body`."""
    body = ModalBody(
        freqs=freqs, fs=fs, sigmas=sigmas, masses=masses, phi=phi, radiation=radiation
    )
    return ReactiveRadiatedBody(body=body, load=RationalAirLoad(fs=fs, R=R, M_a=M_a))


def discrete_sho_frequency(f: float, k: float) -> float:
    """Exact discrete oscillation frequency (Hz) of the leapfrog SHO for a mode of ``f`` Hz.

    The scheme ``q^{n+1} - 2q^n + q^{n-1} = -k^2 omega^2 q^n`` has solutions ``cos(Omega n k)`` with
    ``sin(Omega k / 2) = omega k / 2``, i.e. ``Omega = (2/k) arcsin(omega k / 2)``. Approaches the
    continuum ``f`` as ``omega k -> 0``; used as the modal oracle.
    """
    omega = 2.0 * np.pi * f
    return float(np.arcsin(0.5 * omega * k) / (np.pi * k))


# -- the 3-D air box (HANDOFF §12.H): the distributed tier of the air node ---------------
#
# A small, ordinary room. The default grid is deliberately tiny (0.9 x 0.7 x 0.6 m at h = 10 cm,
# i.e. 10 x 8 x 7 = 560 nodes): 3-D is the first model here where grid cost is a design constraint,
# and every structural/modal oracle is grid-size-independent, so they run where they are free.
# The sample rate is *solved for* from the requested Courant number, as the bore's helper did --
# but the 3-D ceiling is lambda <= 1/sqrt(3) ~ 0.577, and unlike the 1-D string NO lambda is
# dispersionless, so 0.9 of the ceiling is a default, never a sweet spot.
AIRBOX_ROOM_DEFAULT = (0.9, 0.7, 0.6)  # m
AIRBOX_H_DEFAULT = 0.1                 # m
AIRBOX_CFL_FRACTION = 0.9              # lambda = fraction / sqrt(3)


def make_airbox(
    *,
    L: tuple[float, float, float] = AIRBOX_ROOM_DEFAULT,
    h: float = AIRBOX_H_DEFAULT,
    cfl: float = AIRBOX_CFL_FRACTION,
    walls="rigid",
    source: tuple[float, float, float] | None = None,
    rho0: float = RHO0_AIR,
    c0: float = C0_AIR,
) -> AirBox:
    """Build an :class:`AirBox` whose Courant number is exactly ``cfl / sqrt(3)``.

    ``h`` fixes the grid, so the sample rate is solved for: ``fs = c0 sqrt(3) / (cfl h)``. ``cfl``
    is the *fraction of the 3-D CFL ceiling*, not lambda itself — pass ``cfl=1.0`` to sit exactly
    on ``lambda = 1/sqrt(3)`` (where the corner mode reaches the arcsin's argument of 1) and
    ``cfl > 1`` to check that construction is refused. ``walls`` takes the same token / float /
    per-face mapping the class does.
    """
    fs = c0 * np.sqrt(3.0) / (cfl * h)
    return AirBox(L=L, fs=fs, h=h, walls=walls, source=source, rho0=rho0, c0=c0)


def airbox_noise(box: AirBox, seed: int = 0, amplitude: float = 1.0) -> AirBox:
    """Seed the room with a random pressure field (and rest velocity) — the structural-test IC.

    Broadband noise excites **every** mode at once, including the ones a smooth pulse would miss,
    so an energy identity that survives it has no direction left to hide in.
    """
    rng = np.random.default_rng(seed)
    box.set_state(amplitude * rng.standard_normal(box.p.shape))
    return box


# The two-way port (batch 2): a modal body loaded BY the room rather than merely heard in it.
# A slightly larger default room than make_airbox's, at a coarser h, so a port can sit at an
# interior node, on a wall and in a corner of the same geometry, and so the two-instrument scene
# has room for a measurable travel time -- 11 x 9 x 7 = 693 nodes, still free.
AIRBOX_PORT_ROOM_DEFAULT = (0.5, 0.4, 0.3)  # m -> N = (10, 8, 6) at h = 5 cm
AIRBOX_PORT_H_DEFAULT = 0.05                # m
AIRBOX_PORT_AT_DEFAULT = (0.15, 0.15, 0.15)  # m -> node (3, 3, 3), interior on every axis
# Body: inharmonic pair, a soundboard-scale modal mass, and radiation weights (m^2) big enough
# that the room genuinely loads it within a few hundred steps.
AIRBOX_PORT_FREQS = np.array([220.0, 337.0])
AIRBOX_PORT_MASS = 0.05          # kg
AIRBOX_PORT_RADIATION = 2e-3     # m^2 (a_1; a_2 = 0.65 a_1)
AIRBOX_PORT_Q0 = np.array([1e-3, 5e-4])  # m


def make_room_loaded_body(
    *,
    room: AirBox | None = None,
    L: tuple[float, float, float] = AIRBOX_PORT_ROOM_DEFAULT,
    h: float = AIRBOX_PORT_H_DEFAULT,
    cfl: float = AIRBOX_CFL_FRACTION,
    walls="rigid",
    at: tuple[float, float, float] = AIRBOX_PORT_AT_DEFAULT,
    radius: float | None = None,
    freqs: np.ndarray = AIRBOX_PORT_FREQS,
    sigmas: np.ndarray | float = 0.0,
    masses: np.ndarray | float = AIRBOX_PORT_MASS,
    phi: np.ndarray | float = 1.0,
    radiation: np.ndarray | float | None = None,
    q0: np.ndarray | float | None = AIRBOX_PORT_Q0,
) -> RoomLoadedBody:
    """A modal body loaded by an :class:`AirBox` through a port — the two-way coupling fixture.

    Returns the :class:`RoomLoadedBody`; its room is on ``.room`` and its port on ``.port``. Pass
    ``room=`` to put a **second** instrument in an existing room (their ports must be disjoint) —
    the sample rate then comes from that room, so ``h`` and ``cfl`` are ignored.

    ``radius=None`` (the default) is a point port: exact, cheap, and with a *grid-dependent* load
    magnitude, which is right for every structural oracle and wrong for a physical one. ``sigmas=0``
    keeps the modes lossless so the **only** energy channel is the room, and the conserved statement
    is ``inst.energy() + inst.room.energy()``.
    """
    if room is None:
        room = make_airbox(L=L, h=h, cfl=cfl, walls=walls)
    if radiation is None:
        radiation = AIRBOX_PORT_RADIATION * 0.65 ** np.arange(np.size(freqs))
    body = ModalBody(
        freqs=freqs, fs=room.fs, sigmas=sigmas, masses=masses, phi=phi, radiation=radiation
    )
    inst = RoomLoadedBody(body=body, room=room, at=at, radius=radius)
    if q0 is not None:
        inst.set_state(q0)
    return inst


def room_scene_energy(*instruments: RoomLoadedBody) -> float:
    """The conserved total of a whole scene: ``sum_j inst_j.energy() + room.energy()``.

    The coupling term cancels identically — each port's ``radiated_energy`` *is* the room's
    ``injected``, seen from the other side of the same terminal — so this statement contains no
    coupling term at all, which is why a drift in it is unambiguously a bug rather than accounting.
    Every instrument must share one room.
    """
    room = instruments[0].room
    if any(inst.room is not room for inst in instruments):
        raise ValueError("room_scene_energy expects instruments sharing a single room.")
    return sum(inst.energy() for inst in instruments) + room.energy()


def gaussian_pulse(fs: float, f0: float, *, amplitude: float = 1e-3, widths: float = 4.0):
    """A Gaussian volume-velocity pulse ``q(t)`` and its derivative ``qdot(t)``, centred at
    ``widths`` standard deviations in so it starts (and ends) at numerical zero.

    ``sigma = 1/(2 pi f0)`` puts the spectral peak near ``f0``. Returned as a pair of callables of
    *time in seconds*, plus the effective duration ``2 * widths * sigma`` — which is what sizes the
    reflection-free window in the free-field oracle.
    """
    sigma = 1.0 / (2.0 * np.pi * f0)
    t0 = widths * sigma

    def q(t):
        return amplitude * np.exp(-((t - t0) ** 2) / (2.0 * sigma * sigma))

    def qdot(t):
        return -amplitude * (t - t0) / (sigma * sigma) * np.exp(
            -((t - t0) ** 2) / (2.0 * sigma * sigma)
        )

    return q, qdot, 2.0 * widths * sigma

