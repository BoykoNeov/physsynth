//! What a room port *is* — carried from `tests/test_airbox_scene.py`'s cross-tier tests
//! (retirement plan §44). Split from `airbox_scene.rs` because these three cost 98 s unoptimised
//! against 3 s optimised, and CI runs them optimised only (the human's call, §44.6). They pin no
//! spelling: every bar is a ratio or a magnitude with percent-level headroom or more.
//!
//! At low `ka` a port's load is an added mass, `Z ~ j omega M_a` with `M_a = rho0 / (4 pi a_eff)`,
//! so `a_eff = rho0 / (4 pi M_a)` is one number characterising the port independently of the
//! room's modal wiggle. A **point** port's is `~ h/3.1`, so it *halves* every time the grid is
//! refined; a **spread** port's barely moves — that contrast, a factor of twenty in grid
//! sensitivity, is the assertion. The spread port's absolute size has a separate trap, the room:
//! the same port reads 8.6% above the ball's `5a/6` in a room ten times its radius and 0.3% above
//! it in one twenty times its radius, so the closed form is asserted where the port is compact.

use physsynth_core::airbox::{impedance_from_zeta, AirBox, Params, Wall, C0_AIR, RHO0_AIR};
use physsynth_core::airbox_port::RoomPort;

// -- the estimator -------------------------------------------------------------------
//    A Gaussian volume-velocity pulse driven straight through the port's own two numbers, so this
//    exercises `R_room` as well as the geometry. Two rooms, on purpose: a RATIO survives a small
//    cheap room, a MAGNITUDE does not (the small room's own reactance is the bigger term).

const SWEEP_H: [f64; 2] = [0.027, 0.0135]; // one halving
const SWEEP_FREQS: [f64; 3] = [50.0, 75.0, 100.0];
const BALL_RADIUS: f64 = 0.05;
const CONTRAST_ROOM: f64 = 0.5; // cheap, and adequate for a ratio
const COMPACT_ROOM: f64 = 1.0; // a / L = 0.05: compact, which is what the closed form wants

/// Bin `k` of the length-`x.len()` DFT, `sum_n x[n] e^{-2 pi i k n / N}`, summed directly — the
/// estimator needs six bins, not a transform.
fn dft_bin(x: &[f64], k: usize) -> (f64, f64) {
    let n = x.len();
    let (mut re, mut im) = (0.0, 0.0);
    for (j, &v) in x.iter().enumerate() {
        // `k j mod N` keeps the phase argument small and exact.
        let phase = 2.0 * std::f64::consts::PI * ((k * j) % n) as f64 / n as f64;
        re += v * phase.cos();
        im -= v * phase.sin();
    }
    (re, im)
}

/// Drive a port with a pulse, read `Z = pbar/q`, and turn its reactance into a radius:
/// `M_a = Im Z / omega`, `a_eff = rho0 / (4 pi M_a)`. Reactance rather than `|Z|` on purpose: `|Z|`
/// carries the room's modal wiggle and the near-field mass does not.
///
/// `Z` at each sweep frequency is NumPy's `np.interp` of the real and imaginary parts of the bin
/// ratio `P/Q` between the two bins that bracket it, on the axis `rfftfreq(n, 1/fs)`.
fn equivalent_radius(side: f64, h: f64, radius: Option<f64>, cfl: f64) -> f64 {
    let fs = C0_AIR * 3.0f64.sqrt() / (cfl * h);
    let z = impedance_from_zeta(1.0, RHO0_AIR, C0_AIR);
    let mut room = AirBox::new(
        Params::new(
            [side; 3],
            fs,
            h,
            [Wall::Impedance(z); 6],
            None,
            RHO0_AIR,
            C0_AIR,
        )
        .unwrap(),
    );
    let centre = room.p.l_actual.map(|v| 0.5 * v);
    let mut port = RoomPort::new(&mut room, centre, radius).unwrap();
    let n_steps = (0.04 * fs).round() as usize;
    let q: Vec<f64> = (0..n_steps)
        .map(|n| {
            let t = n as f64 / fs;
            (-0.5 * ((t - 4.0e-3) / 6.0e-4).powi(2)).exp()
        })
        .collect();
    let mut pbar = vec![0.0; n_steps];
    for n in 0..n_steps {
        pbar[n] = port.free_pressure(&room) + port.r_room() * q[n];
        port.inject(&mut room, q[n]).unwrap();
        room.step();
    }

    let d = 1.0 / fs;
    let df = 1.0 / (n_steps as f64 * d);
    let ratio = |k: usize| {
        let ((pr, pi), (qr, qi)) = (dft_bin(&pbar, k), dft_bin(&q, k));
        let den = qr * qr + qi * qi;
        ((pr * qr + pi * qi) / den, (pi * qr - pr * qi) / den)
    };
    let mut total = 0.0;
    for f in SWEEP_FREQS {
        let k = (f / df).floor() as usize;
        let (f0, f1) = (k as f64 * df, (k + 1) as f64 * df);
        let ((_, i0), (_, i1)) = (ratio(k), ratio(k + 1));
        let reactance = i0 + (f - f0) * (i1 - i0) / (f1 - f0);
        let m_a = reactance / (2.0 * std::f64::consts::PI * f);
        total += RHO0_AIR / (4.0 * std::f64::consts::PI * m_a);
    }
    total / SWEEP_FREQS.len() as f64
}

/// `test_point_port_load_is_a_grid_quantity_and_the_spread_port_is_not` — the measured
/// non-convergence, shipped as one. A point port behaves as a sphere of radius `~ h/3.1`, so
/// refining the grid **halves** it and doubles the added mass it hangs on the body; a fixed-radius
/// ball does not move. That contrast — a factor of twenty in grid sensitivity — is the assertion.
#[test]
fn a_point_ports_load_is_a_grid_quantity_and_a_spread_ports_is_not() {
    let point: Vec<f64> = SWEEP_H
        .iter()
        .map(|&h| equivalent_radius(CONTRAST_ROOM, h, None, 0.9))
        .collect();
    let spread: Vec<f64> = SWEEP_H
        .iter()
        .map(|&h| equivalent_radius(CONTRAST_ROOM, h, Some(BALL_RADIUS), 0.9))
        .collect();
    for (a_eff, h) in point.iter().zip(SWEEP_H) {
        assert!(
            (0.30..0.34).contains(&(a_eff / h)),
            "a_eff / h = {}",
            a_eff / h
        );
    }
    let shrink = point[1] / point[0];
    assert!((0.45..0.55).contains(&shrink), "point port: {shrink}");
    let hold = spread[1] / spread[0];
    assert!((0.95..1.10).contains(&hold), "spread port: {hold}");
    assert!(
        spread[0] > 4.0 * point[0],
        "the two tiers are nowhere near each other"
    );
}

/// `test_spread_port_matches_the_uniformly_injecting_ball` — a uniformly injecting **ball** carries
/// the classic 6/5 factor in its volume-averaged self-pressure, so its equivalent shell radius is
/// `5a/6`. The room matters more than the shape factor does, so the closed form is asserted where
/// the port is compact and the small room's excess is asserted too — as the attribution.
#[test]
fn a_spread_port_is_the_uniformly_injecting_ball() {
    let ball = 5.0 * BALL_RADIUS / 6.0;
    let compact = equivalent_radius(COMPACT_ROOM, 0.027, Some(BALL_RADIUS), 0.9);
    let small = equivalent_radius(CONTRAST_ROOM, 0.027, Some(BALL_RADIUS), 0.9);
    assert!((compact / ball - 1.0).abs() < 0.05, "{}", compact / ball);
    assert!(
        small > 1.05 * compact,
        "the small room should read high: {small} vs {compact}"
    );
}

/// `test_the_reactance_is_not_a_dispersion_artifact` — `a_eff` is a static near-field quantity, so
/// it must not depend on the Courant number; 3-D has no dispersionless `lambda`, so this is
/// asserted rather than assumed.
#[test]
fn the_reactance_is_not_a_dispersion_artifact() {
    let slow = equivalent_radius(CONTRAST_ROOM, 0.027, Some(BALL_RADIUS), 0.5);
    let fast = equivalent_radius(CONTRAST_ROOM, 0.027, Some(BALL_RADIUS), 0.9);
    assert!((fast / slow - 1.0).abs() < 1e-4, "{}", fast / slow - 1.0);
}
