//! Native acceptance bars for `connection::StringBodyBridge` — a string terminated on a modal
//! body through a linear bridge spring, the coupling node of `exciter -> resonator -> body`.
//!
//! These replace `tests/test_connection.py`, retired whole in the same commit, and the four chain
//! tests that put the bridge in front of a loaded body — three in `tests/test_radiation.py` and
//! one in `tests/test_airbox_freefield.py`. Each bar names the Python test it carries over; the
//! new ones say they are new (retirement plan §19).
//!
//! Conservation and passivity follow from the linear-leapfrog structure and are necessary but
//! not discriminating — a flipped coupling sign conserves energy too. The sharp bars are the
//! coupling's **shape** (one step is the free parts plus exactly one spring force), `K = 0`
//! bit-identity, the pressure read-out carrying the spring's term, and the guard's frozen value.

use physsynth_core::airbox::{flat, impedance_from_zeta, AirBox, Params as RoomParams, Wall};
use physsynth_core::airbox_wrap::RoomLoadedBody;
use physsynth_core::body::{ModalBody, Params as BodyParams};
use physsynth_core::connection::{BridgeBody, BridgeError, StringBodyBridge};
use physsynth_core::exciter::triangular_pluck;
use physsynth_core::radiation::{
    AirParams, AirRadiation, LoadParams, RadiatedBody, RationalAirLoad, ReactiveRadiatedBody,
    C0_AIR, RHO0_AIR,
};
use physsynth_core::string_ideal::{Boundary, IdealString, Params as StringParams};

/// The project's lossless acceptance bar (`CLAUDE.md`), deliberately not tightened.
const DRIFT_TOL: f64 = 1e-10;

// -- the suite's fixture, `tests/helpers.py::make_bridge` --------------------------------------

const L_DEFAULT: f64 = 1.0;
const T_DEFAULT: f64 = 200.0;
/// `c = sqrt(T / rho) = 200 m/s`.
const RHO_DEFAULT: f64 = 0.005;
const BODY_FREQS_DEFAULT: [f64; 4] = [110.0, 196.0, 261.0, 440.0];
/// Modal masses comparable to the string's `rho L = 0.005 kg`, so the body genuinely loads it.
const BODY_MASS_DEFAULT: f64 = 0.02;
const K_BRIDGE_DEFAULT: f64 = 8000.0;

/// How a bridge is built, mirroring the Python helper's keyword defaults.
struct Fixture {
    n: i64,
    lam: f64,
    k: f64,
    sigma_string: f64,
    sigma_body: f64,
}

impl Default for Fixture {
    fn default() -> Fixture {
        Fixture {
            n: 100,
            lam: 0.9,
            k: K_BRIDGE_DEFAULT,
            sigma_string: 0.0,
            sigma_body: 0.0,
        }
    }
}

impl Fixture {
    fn lam(mut self, lam: f64) -> Fixture {
        self.lam = lam;
        self
    }

    fn stiffness(mut self, k: f64) -> Fixture {
        self.k = k;
        self
    }

    fn string_loss(mut self, sigma: f64) -> Fixture {
        self.sigma_string = sigma;
        self
    }

    fn body_loss(mut self, sigma: f64) -> Fixture {
        self.sigma_body = sigma;
        self
    }

    /// `fs = c N / (L lambda)` — `lambda < 1` gives the coupled system headroom below the
    /// string's Nyquist mode.
    fn fs(&self) -> f64 {
        let c = (T_DEFAULT / RHO_DEFAULT).sqrt();
        c * (self.n as f64) / (L_DEFAULT * self.lam)
    }

    fn string(&self) -> IdealString {
        free_string(self.fs(), self.n, self.sigma_string)
    }

    fn body(&self) -> ModalBody {
        body_at(self.fs(), self.sigma_body)
    }

    fn try_build(&self) -> Result<StringBodyBridge<ModalBody>, BridgeError> {
        StringBodyBridge::new(self.string(), self.body(), self.k)
    }

    fn build(&self) -> StringBodyBridge<ModalBody> {
        self.try_build().expect("the fixture sits inside the guard")
    }
}

fn free_string(fs: f64, n: i64, sigma: f64) -> IdealString {
    IdealString::new(
        StringParams::new(
            L_DEFAULT,
            T_DEFAULT,
            RHO_DEFAULT,
            fs,
            n,
            sigma,
            Some((Boundary::Fixed, Boundary::Free)),
        )
        .expect("valid string"),
    )
}

fn body_at(fs: f64, sigma: f64) -> ModalBody {
    let m = BODY_FREQS_DEFAULT.len();
    ModalBody::new(
        BodyParams::new(
            BODY_FREQS_DEFAULT.to_vec(),
            fs,
            vec![sigma; m],
            vec![BODY_MASS_DEFAULT; m],
            vec![1.0; m],
            None,
        )
        .expect("valid body"),
    )
}

/// A triangular pluck at `at * L`, amplitude `1e-3`, from rest.
fn pluck(s: &mut IdealString, at: f64) {
    let p = s.params().clone();
    let u0 = triangular_pluck(&p.grid(), p.l, at * p.l, 1e-3).expect("inside the string");
    s.set_state(&u0, &vec![0.0; u0.len()]);
}

/// `max |E^n - E^0| / |E^0|` over `steps` steps, `E^0` included — the engine's `energy_drift`.
fn drift<B: BridgeBody<Room = ()>>(br: &mut StringBodyBridge<B>, steps: usize) -> f64
where
    B::Error: std::fmt::Debug,
{
    let e0 = br.energy();
    let mut worst: f64 = 0.0;
    for _ in 0..steps {
        br.step(&mut ()).unwrap();
        worst = worst.max((br.energy() - e0).abs());
    }
    worst / e0.abs()
}

/// The energy trace over `steps` steps, `E^0` first.
fn trace(br: &mut StringBodyBridge<ModalBody>, steps: usize) -> Vec<f64> {
    let mut e = vec![br.energy()];
    for _ in 0..steps {
        br.step(&mut ()).unwrap();
        e.push(br.energy());
    }
    e
}

fn seconds(br: &StringBodyBridge<impl BridgeBody>, s: f64) -> usize {
    (s * br.string().params().fs) as usize
}

// -- the total energy is conserved (the string alone is not) -----------------------------------

/// Carries `test_total_energy_conserved_across_lambda`.
#[test]
fn the_total_energy_is_conserved_across_lambda() {
    for lam in [0.9, 0.7, 0.5] {
        let mut br = Fixture::default().lam(lam).build();
        pluck(br.string_mut(), 0.137);
        let steps = seconds(&br, 2.0);
        let d = drift(&mut br, steps);
        assert!(d < DRIFT_TOL, "total drift {d:.2e} at lam={lam}");
    }
}

/// Carries `test_total_energy_conserved_across_stiffness`.
#[test]
fn the_total_energy_is_conserved_across_stiffness() {
    for k in [500.0, 4000.0, 8000.0, 15000.0] {
        let mut br = Fixture::default().stiffness(k).build();
        pluck(br.string_mut(), 0.137);
        let steps = seconds(&br, 1.5);
        let d = drift(&mut br, steps);
        assert!(d < DRIFT_TOL, "total drift {d:.2e} at K={k}");
    }
}

/// Carries `test_string_energy_alone_is_not_conserved`: the string exchanges real energy with
/// the body, so its own energy wanders while the total is pinned.
#[test]
fn the_string_energy_alone_is_not_conserved() {
    let mut br = Fixture::default().build();
    pluck(br.string_mut(), 0.137);
    let e0 = br.string().energy();
    let (mut lo, mut hi) = (e0, e0);
    for _ in 0..seconds(&br, 0.3) {
        br.step(&mut ()).unwrap();
        let e = br.string().energy();
        lo = lo.min(e);
        hi = hi.max(e);
    }
    let spread = (hi - lo) / e0;
    assert!(
        spread > 1e-3,
        "string energy barely moved ({spread:.2e}); coupling not engaged"
    );
}

/// Carries `test_energy_flows_string_to_body`.
#[test]
fn energy_flows_from_the_string_into_the_body() {
    let mut br = Fixture::default().build();
    pluck(br.string_mut(), 0.137);
    let mut max_frac: f64 = 0.0;
    for _ in 0..seconds(&br, 0.5) {
        br.step(&mut ()).unwrap();
        let (es, eb) = (br.string().energy(), br.body().energy());
        max_frac = max_frac.max(eb / (es + eb));
    }
    assert!(
        max_frac > 0.1,
        "the body only ever held {:.1}%",
        100.0 * max_frac
    );
}

// -- passivity -----------------------------------------------------------------------------------

fn assert_monotone(e: &[f64]) {
    let worst = e.windows(2).map(|w| w[1] - w[0]).fold(f64::MIN, f64::max);
    assert!(worst <= 1e-12 * e[0], "max positive step {worst:.2e}");
}

/// Carries `test_passivity_with_body_damping`.
#[test]
fn a_lossy_body_makes_the_total_decrease_monotonically() {
    let mut br = Fixture::default().body_loss(10.0).build();
    pluck(br.string_mut(), 0.137);
    let steps = seconds(&br, 2.0);
    assert_monotone(&trace(&mut br, steps));
}

/// Carries `test_passivity_with_string_damping`.
#[test]
fn a_lossy_string_makes_the_total_decrease_monotonically() {
    let mut br = Fixture::default().string_loss(5.0).build();
    pluck(br.string_mut(), 0.137);
    let steps = seconds(&br, 2.0);
    assert_monotone(&trace(&mut br, steps));
}

// -- the coupling's shape --------------------------------------------------------------------------

/// Carries `test_K0_bit_identical_to_uncoupled_parts`, the body excited too so its path is not
/// trivially at rest.
#[test]
fn at_zero_stiffness_the_bridge_is_bit_identical_to_the_uncoupled_parts() {
    let fx = Fixture::default().stiffness(0.0);
    let q0 = [1e-3, -0.5e-3, 0.7e-3, 0.3e-3];
    let mut br = fx.build();
    pluck(br.string_mut(), 0.137);
    br.body_mut().set_state(&q0, &[0.0; 4]);
    let mut s = fx.string();
    pluck(&mut s, 0.137);
    let mut b = fx.body();
    b.set_state(&q0, &[0.0; 4]);
    for _ in 0..500 {
        br.step(&mut ()).unwrap();
        s.step();
        b.step(0.0);
    }
    assert_eq!(
        br.string().u,
        s.u,
        "string not bit-identical to uncoupled at K=0"
    );
    assert_eq!(
        br.body().q(),
        b.q(),
        "body not bit-identical to uncoupled at K=0"
    );
}

/// **New.** One coupled step is one free step of each part plus exactly one spring force: the
/// string corrected at its end node by `2 k^2 F / (rho h)`, the body handed `+F` unscaled — bit
/// for bit, since every difference is elementwise arithmetic in the same order. §12's restated
/// sympathetic bar, applied to the bridge it was restated *away* from.
#[test]
fn one_step_is_the_free_parts_plus_exactly_one_spring_force() {
    let fx = Fixture::default();
    let mut br = fx.build();
    pluck(br.string_mut(), 0.137);
    for _ in 0..300 {
        br.step(&mut ()).unwrap();
    }
    let f = br.connection_force();
    assert!(f.abs() > 0.0, "the wave must have reached the bridge");
    let mut s = br.string().clone();
    let mut b = br.body().clone();
    br.step(&mut ()).unwrap();
    s.step();
    let p = s.params().clone();
    let last = s.u.len() - 1;
    s.u[last] -= (2.0 * p.k * p.k / (p.rho * p.h)) * f;
    b.step(f);
    assert_eq!(br.string().u, s.u);
    assert_eq!(br.body().q(), b.q());
    assert_eq!(br.beta_s(), 2.0 * p.k * p.k / (p.rho * p.h));
}

/// Carries `test_pressure_includes_coupling_term`: after a step with `F != 0` the stored modal
/// acceleration is the full forced equation of motion, so the radiated pressure carries the
/// spring and is not the unforced `-omega^2 q` reconstruction.
#[test]
fn the_pressure_read_out_carries_the_coupling_term() {
    let mut br = Fixture::default().build();
    pluck(br.string_mut(), 0.137);
    for _ in 0..400 {
        br.step(&mut ()).unwrap();
    }
    let q_before = br.body().q().to_vec();
    let f = br.connection_force();
    assert!(f.abs() > 0.0, "no bridge force to test against");
    br.step(&mut ()).unwrap();
    let p = br.body().params().clone();
    let expected: Vec<f64> = (0..q_before.len())
        .map(|i| -p.omega[i] * p.omega[i] * q_before[i] + p.phi[i] * f / p.m[i])
        .collect();
    let scale = expected.iter().fold(0.0f64, |a, v| a.max(v.abs()));
    for (i, (&got, &want)) in br.body().accel().iter().zip(&expected).enumerate() {
        assert!(
            (got - want).abs() <= 1e-9 * scale,
            "mode {i}: accel {got:e} vs the forced EoM {want:e}"
        );
    }
    let naive: f64 = (0..q_before.len())
        .map(|i| p.a[i] * (-p.omega[i] * p.omega[i] * q_before[i]))
        .sum();
    assert!(
        (br.pressure() - naive).abs() > 1e-6 * naive.abs(),
        "pressure() equals the unforced reconstruction -- the coupling term is missing"
    );
}

// -- the stability guard ----------------------------------------------------------------------------

/// Carries `test_unstable_stiffness_rejected`.
#[test]
fn an_over_stiff_spring_is_rejected() {
    let err = Fixture::default().stiffness(1e6).try_build().unwrap_err();
    assert!(matches!(err, BridgeError::Unstable(_)), "{err:?}");
    assert!(err.to_string().contains("unstable"), "{err}");
}

/// Carries `test_guard_holds_at_its_boundary`: just inside the exact guard (the ceiling is near
/// 21.5k at `lambda = 0.9`) the run still conserves to machine precision — the guard is the true
/// energy-preserving bound, not merely "does not overflow".
#[test]
fn the_guard_holds_just_inside_its_boundary() {
    let mut br = Fixture::default().stiffness(20000.0).build();
    assert!(br.timestep() * br.timestep() * br.spectral_radius() < 4.0);
    pluck(br.string_mut(), 0.137);
    let steps = seconds(&br, 1.5);
    let d = drift(&mut br, steps);
    assert!(d < DRIFT_TOL, "drift {d:.2e} near the guard boundary");
}

/// **New.** The guard's number, frozen from the implementation it replaces: `lambda_max` of the
/// default fixture as LAPACK's general `dgeev` computed it through the binding. The native guard
/// is a symmetric routine on the mass-symmetrized operator, so this pins the **mass derivation**
/// too — a wrong half cell, or the spring's factor of 2 on the wrong side, still symmetrizes and
/// silently rescales the spectrum. Measured spread 4.3e-16; a value and a tolerance rather than
/// a bit pattern, as §12 froze the sympathetic set's.
#[test]
fn the_guard_reproduces_the_binding_measured_spectral_radius() {
    let br = Fixture::default().build();
    let frozen = 1.661_741_169_893_860_3e9;
    let rel = (br.spectral_radius() - frozen).abs() / frozen;
    assert!(
        rel < 1e-12,
        "lambda_max {:.17e}, rel {rel:.2e}",
        br.spectral_radius()
    );
}

/// **New.** `M^1/2 A M^-1/2` is symmetric, which is what licenses the symmetric eigenroutine.
#[test]
fn the_coupled_operator_is_self_adjoint_in_the_energy_inner_product() {
    let br = Fixture::default().build();
    let n = br.dofs();
    let a = br.symmetrized_operator();
    let scale = a.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    for i in 0..n {
        for j in 0..i {
            let d = (a[i * n + j] - a[j * n + i]).abs();
            assert!(d <= 1e-14 * scale, "asymmetry {d:e} at ({i}, {j})");
        }
    }
}

/// **New.** No `lambda < 1` refusal. The sympathetic set added one; this bridge never had it, and
/// at `lambda = 1` a soft spring is accepted (`k^2 lambda_max = 3.99986`, measured through the
/// binding) while a stiffer one is refused by the **spectral** guard, not by a lambda check.
#[test]
fn a_string_at_the_courant_limit_is_refused_only_by_the_spectral_guard() {
    let br = Fixture::default().lam(1.0).stiffness(100.0).build();
    assert!(br.timestep() * br.timestep() * br.spectral_radius() < 4.0);
    let err = Fixture::default()
        .lam(1.0)
        .stiffness(1000.0)
        .try_build()
        .unwrap_err();
    assert!(matches!(err, BridgeError::Unstable(_)), "{err:?}");
}

// -- construction refusals --------------------------------------------------------------------------

/// Carries `test_mismatched_timestep_rejected`.
#[test]
fn a_mismatched_timestep_is_rejected() {
    let s = free_string(22000.0, 100, 0.0);
    let b = body_at(24000.0, 0.0);
    let err = StringBodyBridge::new(s, b, K_BRIDGE_DEFAULT).unwrap_err();
    assert!(
        matches!(err, BridgeError::TimestepMismatch("body", _, _)),
        "{err:?}"
    );
    assert!(err.to_string().contains("timestep"), "{err}");
}

/// Carries `test_right_end_must_be_free`.
#[test]
fn a_clamped_right_end_is_rejected() {
    let fs = 22000.0;
    let s = IdealString::new(
        StringParams::new(
            1.0,
            200.0,
            0.005,
            fs,
            100,
            0.0,
            Some((Boundary::Fixed, Boundary::Fixed)),
        )
        .unwrap(),
    );
    let err = StringBodyBridge::new(s, body_at(fs, 0.0), K_BRIDGE_DEFAULT).unwrap_err();
    assert_eq!(err, BridgeError::RightEndNotFree("body"));
    assert!(err.to_string().contains("free"), "{err}");
}

/// **New.** A negative stiffness is refused, after the two structural refusals.
#[test]
fn a_negative_stiffness_is_rejected() {
    let err = Fixture::default().stiffness(-1.0).try_build().unwrap_err();
    assert_eq!(err, BridgeError::NegativeStiffness);
}

// -- composition: the bridge is blind to what its body is -------------------------------------------

/// Carries `tests/test_radiation.py::test_full_chain_with_radiation_conserves_the_total`: a
/// lossless string and body with the radiation channel as the only sink, so the bridge's own
/// energy — which books the loaded body's **override** — is conserved.
#[test]
fn the_full_chain_with_radiation_conserves_the_total() {
    let fx = Fixture::default();
    let loaded = RadiatedBody::new(fx.body(), 1500.0).unwrap();
    let mut br = StringBodyBridge::new(fx.string(), loaded, 8000.0).unwrap();
    pluck(br.string_mut(), 0.3);
    let e0 = br.energy();
    let d = drift(&mut br, 6000);
    assert!(d < 1e-9, "the four-way energy identity drifted {d:.2e}");
    assert!(
        br.body().radiated_energy > 0.05 * e0,
        "the chain must ring down into the air"
    );
}

/// Carries `tests/test_radiation.py::test_full_chain_with_reactive_radiation_conserves_the_total`.
#[test]
fn the_full_chain_with_reactive_radiation_conserves_the_total() {
    let fx = Fixture::default();
    let load =
        RationalAirLoad::new(LoadParams::new(fx.fs(), 1500.0, 0.05, RHO0_AIR, C0_AIR).unwrap());
    let loaded = ReactiveRadiatedBody::new(fx.body(), load).unwrap();
    let mut br = StringBodyBridge::new(fx.string(), loaded, 8000.0).unwrap();
    pluck(br.string_mut(), 0.3);
    let e0 = br.energy();
    let d = drift(&mut br, 6000);
    assert!(
        d < 1e-9,
        "E_string + E_body + E_conn + E_air drifted {d:.2e}"
    );
    assert!(br.body().load().radiated_energy > 0.05 * e0);
}

/// Carries `tests/test_radiation.py::test_full_chain_radiates_with_retardation`: string ->
/// bridge -> body -> far field, silent until the wavefront arrives.
#[test]
fn the_full_chain_radiates_with_retardation() {
    let mut br = Fixture::default().build();
    pluck(br.string_mut(), 0.3);
    let fs = 1.0 / br.timestep();
    let mut rad = AirRadiation::new(AirParams::new(fs, 1.0, RHO0_AIR, C0_AIR, true).unwrap());
    let latency = rad.params().latency_samples;
    assert!(latency > 0);
    let mut p = Vec::with_capacity(4000);
    for _ in 0..4000 {
        br.step(&mut ()).unwrap();
        p.push(rad.process(br.pressure()));
    }
    assert!(p.iter().all(|v| v.is_finite()));
    assert!(
        p[..latency].iter().all(|&v| v == 0.0),
        "sound before the wavefront arrived"
    );
    assert!(p[latency..].iter().any(|&v| v != 0.0), "nothing radiated");
}

/// Carries `tests/test_connection.py::test_string_bridge_body_room_chain_conserves`, which had
/// been moved there from `tests/test_airbox_port.py` to wait for this port: the full
/// `string -> bridge -> body -> room` chain. The bridge hands its room straight to the body and
/// never steps it; the caller steps the room after the bridge, as every Python chain did.
#[test]
fn the_string_bridge_body_room_chain_conserves() {
    let fs = 200.0 * 100.0 / (1.0 * 0.9);
    let h = 343.0 * 3.0f64.sqrt() / (0.9 * fs);
    let mut room = AirBox::new(
        RoomParams::new(
            [10.0 * h, 8.0 * h, 6.0 * h],
            fs,
            h,
            [Wall::Impedance(impedance_from_zeta(1.0, RHO0_AIR, C0_AIR)); 6],
            None,
            RHO0_AIR,
            C0_AIR,
        )
        .unwrap(),
    );
    let body = ModalBody::new(
        BodyParams::new(
            vec![180.0, 291.0],
            fs,
            vec![0.0; 2],
            vec![0.02; 2],
            vec![1.0; 2],
            Some(vec![3e-3, 2e-3]),
        )
        .unwrap(),
    );
    let inst = RoomLoadedBody::new(body, &mut room, [3.0 * h, 3.0 * h, 3.0 * h], None).unwrap();
    let mut br = StringBodyBridge::new(free_string(fs, 100, 0.0), inst, 8000.0).unwrap();
    pluck(br.string_mut(), 0.3);

    // bridge.energy() already carries the body's override (its radiated channel); adding
    // radiated_energy again would double-count.
    let total = |br: &StringBodyBridge<RoomLoadedBody>, room: &AirBox| br.energy() + room.energy();
    let e0 = total(&br, &room);
    let (mut lo, mut hi) = (e0, e0);
    for _ in 0..600 {
        br.step(&mut room).unwrap();
        room.step();
        let e = total(&br, &room);
        lo = lo.min(e);
        hi = hi.max(e);
    }
    assert!(
        (hi - lo) / e0.abs() < DRIFT_TOL,
        "chain drift {:.2e}",
        (hi - lo) / e0.abs()
    );
    assert!(
        br.body().radiated_energy() != 0.0,
        "the room was actually driven"
    );
}

/// Carries `tests/test_airbox_freefield.py::test_string_bridge_body_air_load_drives_the_room`:
/// `string -> bridge -> ReactiveRadiatedBody -> AirBox`, the room driven by the load's public
/// volume velocity — the exact continuity source — with no edit to any of the three models.
#[test]
fn the_string_bridge_body_air_load_drives_the_room() {
    let fx = Fixture::default();
    let fs = fx.fs();
    let load = RationalAirLoad::new(LoadParams::new(fs, 1500.0, 0.05, RHO0_AIR, C0_AIR).unwrap());
    let loaded = ReactiveRadiatedBody::new(fx.body(), load).unwrap();
    let mut br = StringBodyBridge::new(fx.string(), loaded, 8000.0).unwrap();
    pluck(br.string_mut(), 0.3);

    // The room runs at the string's rate for a sample-aligned drive, so the CFL fixes h.
    let h = 3.0f64.sqrt() * C0_AIR / fs;
    let mut room = AirBox::new(
        RoomParams::new(
            [8.0 * h, 6.0 * h, 5.0 * h],
            fs,
            h,
            [Wall::Impedance(RHO0_AIR * C0_AIR); 6],
            None,
            RHO0_AIR,
            C0_AIR,
        )
        .unwrap(),
    );
    let shape = room.p.p_shape();
    let at = flat(shape, 6, 3, 2);
    let mut heard: f64 = 0.0;
    for _ in 0..400 {
        br.step(&mut ()).unwrap();
        room.inject(br.body().load().volume_velocity, None);
        room.step();
        heard = heard.max(room.pressure[at].abs());
    }
    assert!(heard > 0.0, "the room heard nothing");
    assert!(room.injected != 0.0);
    // The room's own three-channel identity still closes while an external chain drives it.
    assert!(room.energy().abs() / room.injected.abs() < 1e-9);
}
