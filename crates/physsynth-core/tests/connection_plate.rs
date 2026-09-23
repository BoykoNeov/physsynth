//! Native acceptance bars for `connection::StringPlateBridge` — a string terminated on a grid
//! plate at one driving-point node through a linear spring: the soundboard, the suspended cymbal,
//! the gong, and each of them mounted in a room.
//!
//! One generic type replaces the binding's two classes, so these bars replace four Python files'
//! worth of tests at once, retired in the same commit (retirement plan §20):
//! `tests/test_plate_connection.py` and `tests/test_free_plate_connection.py` (the linear plate,
//! supported and free — every bar below that loops over both boundaries carries one test from
//! each), `tests/test_vk_connection.py` (the von Karman plate), and the chain tests of
//! `tests/test_airbox_vk.py`, `tests/test_airbox_surface.py` and `tests/test_airbox_dipole.py`.
//! Each bar names what it carries; the new ones say so.
//!
//! Two anchors the Python asserted **across classes** are now one code path, and each got a
//! verdict rather than a tautology: the von Karman margin equalling the linear margin is kept,
//! because the linear twin is an independently built `Plate` and the equality is the claim that
//! `VkParams::lin` carries the *areal* density (and a bar now shows the volume density would move
//! it); the room-loaded margin equalling the bare one is kept as the statement that the guard reads
//! the plate's parameters, never the loaded factorization.

use physsynth_core::airbox::{impedance_from_zeta, AirBox, Params as RoomParams, Wall};
use physsynth_core::airbox_port::Spreading;
use physsynth_core::airbox_wrap::{GridSeam, PlateSeam, RoomGrid, VkSeam};
use physsynth_core::connection::{BridgeError, BridgePlate, StringPlateBridge};
use physsynth_core::exciter::triangular_pluck;
use physsynth_core::plate::{
    self, Boundary as PB, CoupleMethod, Plate, PlateSpec, VkParams, VkPlate, VkSpec,
};
use physsynth_core::radiation::{C0_AIR, RHO0_AIR};
use physsynth_core::string_ideal::{Boundary, IdealString, Params as StringParams};

/// The project's lossless acceptance bar (`CLAUDE.md`), deliberately not tightened.
const DRIFT_TOL: f64 = 1e-10;

const BOUNDARIES: [PB; 2] = [PB::Supported, PB::Free];

// -- shared pieces -------------------------------------------------------------------------------

fn free_string(l: f64, t: f64, n: i64, fs: f64, sigma: f64) -> IdealString {
    IdealString::new(
        StringParams::new(
            l,
            t,
            0.005,
            fs,
            n,
            sigma,
            Some((Boundary::Fixed, Boundary::Free)),
        )
        .expect("valid string"),
    )
}

/// A triangular pluck at `0.137 L`, from rest.
fn pluck(s: &mut IdealString, amplitude: f64) {
    let p = s.params().clone();
    let u0 = triangular_pluck(&p.grid(), p.l, 0.137 * p.l, amplitude).expect("inside");
    s.set_state(&u0, &vec![0.0; u0.len()]);
}

/// A half-sine pluck — the room chains' excitation (`vk_room_pluck`).
fn sine_pluck(s: &mut IdealString, amplitude: f64) {
    let p = s.params().clone();
    let u0: Vec<f64> = p
        .grid()
        .iter()
        .map(|x| amplitude * (std::f64::consts::PI * x / p.l).sin())
        .collect();
    s.set_state(&u0, &vec![0.0; u0.len()]);
}

/// Live-node `(x, y)` in the plate's own C-order.
fn live_xy(p: &plate::Params) -> Vec<(f64, f64)> {
    let mut out = Vec::with_capacity(p.n_live);
    for (idx, &alive) in p.mask.flags().iter().enumerate() {
        if alive {
            out.push((p.x[idx], p.y[idx]));
        }
    }
    out
}

/// `max |E^n - E^0| / |E^0|` — the engine's `energy_drift`.
fn drift<P: BridgePlate<Room = ()>>(br: &mut StringPlateBridge<P>, steps: usize) -> f64
where
    P::Error: std::fmt::Debug,
{
    let e0 = br.energy();
    let mut worst: f64 = 0.0;
    for _ in 0..steps {
        br.step(&mut ()).unwrap();
        worst = worst.max((br.energy() - e0).abs());
    }
    worst / e0.abs()
}

fn assert_monotone<P: BridgePlate<Room = ()>>(br: &mut StringPlateBridge<P>, steps: usize)
where
    P::Error: std::fmt::Debug,
{
    let e0 = br.energy();
    let mut prev = e0;
    for n in 0..steps {
        br.step(&mut ()).unwrap();
        let e = br.energy();
        assert!(
            e - prev <= 1e-12 * e0,
            "energy rose at step {n}: {prev:.12e} -> {e:.12e}"
        );
        prev = e;
    }
    assert!(prev < e0, "the lossy run never lost anything");
}

fn seconds<P: BridgePlate>(br: &StringPlateBridge<P>, s: f64) -> usize {
    (s * br.string().params().fs) as usize
}

// =================================================================================================
// The linear plate — `tests/helpers.py::make_plate_bridge` / `make_free_plate_bridge`
// =================================================================================================

/// A coarse plate whose node mass `rho h^2` is comparable to the string's end half-cell.
const KAPPA_PLATE_BRIDGE: f64 = 20.0;
const K_PLATE_BRIDGE_DEFAULT: f64 = 3000.0;
const RHO_AREAL_DEFAULT: f64 = 0.005;

#[derive(Clone, Copy)]
struct Grid {
    boundary: PB,
    lam: f64,
    k: f64,
    sigma_string: f64,
    sigma_plate: f64,
    drive: Option<usize>,
}

impl Grid {
    fn new(boundary: PB) -> Grid {
        Grid {
            boundary,
            lam: 0.9,
            k: K_PLATE_BRIDGE_DEFAULT,
            sigma_string: 0.0,
            sigma_plate: 0.0,
            drive: None,
        }
    }

    fn lam(mut self, lam: f64) -> Grid {
        self.lam = lam;
        self
    }

    fn stiffness(mut self, k: f64) -> Grid {
        self.k = k;
        self
    }

    fn string_loss(mut self, s: f64) -> Grid {
        self.sigma_string = s;
        self
    }

    fn plate_loss(mut self, s: f64) -> Grid {
        self.sigma_plate = s;
        self
    }

    fn fs(&self) -> f64 {
        200.0 * 100.0 / (1.0 * self.lam)
    }

    fn string(&self) -> IdealString {
        free_string(1.0, 200.0, 100, self.fs(), self.sigma_string)
    }

    fn plate(&self) -> Plate {
        grid_plate(
            1.0,
            KAPPA_PLATE_BRIDGE,
            RHO_AREAL_DEFAULT,
            16,
            self.fs(),
            self.sigma_plate,
            self.boundary,
        )
    }

    fn try_build(&self) -> Result<StringPlateBridge<Plate>, BridgeError> {
        StringPlateBridge::new(self.string(), self.plate(), self.k, self.drive)
    }

    fn build(&self) -> StringPlateBridge<Plate> {
        self.try_build().expect("inside the guard")
    }

    fn plucked(&self) -> StringPlateBridge<Plate> {
        let mut br = self.build();
        pluck(br.string_mut(), 1e-3);
        br
    }

    /// The exact ceiling: the margin is linear in `K`, so scale a probe to margin 1.
    fn ceiling(&self) -> f64 {
        1000.0 / self.stiffness(1000.0).build().stability_margin()
    }
}

fn grid_plate(l: f64, kappa: f64, rho: f64, n: i64, fs: f64, sigma: f64, b: PB) -> Plate {
    Plate::new(
        plate::Params::new(&PlateSpec {
            lx: l,
            ly: l,
            kappa,
            rho,
            fs,
            n,
            sigma,
            boundary: Some(b),
            nu: if b == PB::Free { Some(0.3) } else { None },
            ..PlateSpec::default()
        })
        .expect("valid plate"),
    )
}

/// Carries `test_total_energy_conserved_across_lambda` from both linear files.
#[test]
fn the_total_energy_is_conserved_across_lambda() {
    for b in BOUNDARIES {
        for lam in [0.9, 0.7, 0.5] {
            let mut br = Grid::new(b).lam(lam).plucked();
            let steps = seconds(&br, 1.5);
            let d = drift(&mut br, steps);
            assert!(d < DRIFT_TOL, "{b:?}: drift {d:.2e} at lam={lam}");
        }
    }
}

/// Carries `test_total_energy_conserved_across_stiffness` from both linear files.
#[test]
fn the_total_energy_is_conserved_across_stiffness() {
    for b in BOUNDARIES {
        for k in [500.0, 3000.0, 6000.0, 10000.0] {
            let mut br = Grid::new(b).stiffness(k).plucked();
            let steps = seconds(&br, 1.2);
            let d = drift(&mut br, steps);
            assert!(d < DRIFT_TOL, "{b:?}: drift {d:.2e} at K={k}");
        }
    }
}

/// Carries `test_string_energy_alone_is_not_conserved` from both linear files.
#[test]
fn the_string_energy_alone_is_not_conserved() {
    for b in BOUNDARIES {
        let mut br = Grid::new(b).plucked();
        let e0 = br.string().energy();
        let (mut lo, mut hi) = (e0, e0);
        for _ in 0..seconds(&br, 0.3) {
            br.step(&mut ()).unwrap();
            lo = lo.min(br.string().energy());
            hi = hi.max(br.string().energy());
        }
        assert!((hi - lo) / e0 > 1e-3, "{b:?}: string energy barely moved");
    }
}

/// Carries `test_energy_flows_string_to_plate` from both linear files.
#[test]
fn energy_flows_from_the_string_into_the_plate() {
    for b in BOUNDARIES {
        let mut br = Grid::new(b).plucked();
        let mut frac: f64 = 0.0;
        for _ in 0..seconds(&br, 0.5) {
            br.step(&mut ()).unwrap();
            let (es, ep) = (br.string().energy(), br.plate().energy());
            frac = frac.max(ep / (es + ep));
        }
        assert!(
            frac > 0.1,
            "{b:?}: the plate only ever held {:.1}%",
            100.0 * frac
        );
    }
}

/// Carries `test_passivity_with_plate_damping` from both linear files.
#[test]
fn a_lossy_plate_makes_the_total_decrease_monotonically() {
    for b in BOUNDARIES {
        let mut br = Grid::new(b).plate_loss(10.0).plucked();
        let steps = seconds(&br, 1.5);
        assert_monotone(&mut br, steps);
    }
}

/// Carries `test_passivity_with_string_damping` from both linear files.
#[test]
fn a_lossy_string_makes_the_total_decrease_monotonically() {
    for b in BOUNDARIES {
        let mut br = Grid::new(b).string_loss(5.0).plucked();
        let steps = seconds(&br, 1.5);
        assert_monotone(&mut br, steps);
    }
}

/// Carries `test_K0_bit_identical_to_uncoupled_parts` from both linear files, the plate excited
/// too — `sin sin` supported, `cos cos` free, as each file did.
#[test]
fn at_zero_stiffness_the_bridge_is_bit_identical_to_the_uncoupled_parts() {
    for b in BOUNDARIES {
        let fx = Grid::new(b).stiffness(0.0);
        let mut br = fx.build();
        let lin = br.plate().p.clone();
        let pi = std::f64::consts::PI;
        let u0: Vec<f64> = live_xy(&lin)
            .into_iter()
            .map(|(x, y)| match b {
                PB::Supported => 1e-3 * (pi * x / lin.lx).sin() * (pi * y / lin.ly).sin(),
                PB::Free => 1e-3 * (pi * x / lin.lx).cos() * (pi * y / lin.ly).cos(),
            })
            .collect();
        let zeros = vec![0.0; u0.len()];
        pluck(br.string_mut(), 1e-3);
        br.plate_mut().set_state(&u0, &zeros);
        let mut s = fx.string();
        pluck(&mut s, 1e-3);
        let mut p = fx.plate();
        p.set_state(&u0, &zeros);
        for _ in 0..400 {
            br.step(&mut ()).unwrap();
            s.step();
            p.step(None);
        }
        assert_eq!(br.string().u, s.u, "{b:?}: string not bit-identical at K=0");
        assert_eq!(br.plate().u, p.u, "{b:?}: plate not bit-identical at K=0");
    }
}

/// **New.** One step is one free string step, the end-node impulse `beta_s F`, and one plate step
/// under a force vector that is `F` at the drive node and zero elsewhere — bit for bit, on two
/// consecutive steps.
#[test]
fn one_step_is_the_free_parts_plus_exactly_one_spring_force() {
    for b in BOUNDARIES {
        let mut br = Grid::new(b).plucked();
        for _ in 0..400 {
            br.step(&mut ()).unwrap();
        }
        let f = br.connection_force();
        assert!(f.abs() > 0.0);
        let mut s = br.string().clone();
        let mut p = br.plate().clone();
        br.step(&mut ()).unwrap();
        s.step();
        let last = s.u.len() - 1;
        s.u[last] -= br.beta_s() * f;
        let mut fe = vec![0.0; p.p.n_live];
        fe[br.drive_index()] = f;
        p.step(Some(&fe));
        assert_eq!(br.string().u, s.u, "{b:?}");
        assert_eq!(br.plate().u, p.u, "{b:?}");
        let f2 = br.connection_force();
        br.step(&mut ()).unwrap();
        s.step();
        s.u[last] -= br.beta_s() * f2;
        fe[br.drive_index()] = f2;
        p.step(Some(&fe));
        assert_eq!(br.plate().u, p.u, "{b:?}: second step");
    }
}

/// Carries `test_pressure_includes_coupling_term` from both linear files: after a forced step the
/// plate's stored acceleration differs from an unforced step's by exactly `A^-1 (F / den)` at the
/// drive node, and the pressure read-out is not the unforced one.
#[test]
fn the_pressure_read_out_carries_the_coupling_term() {
    for b in BOUNDARIES {
        let mut br = Grid::new(b).stiffness(6000.0).plucked();
        for _ in 0..400 {
            br.step(&mut ()).unwrap();
        }
        let f = br.connection_force();
        assert!(f.abs() > 0.0, "no bridge force to test against");
        let mut free = br.plate().clone();
        free.step(None);
        br.step(&mut ()).unwrap();
        let p = &br.plate().p;
        let mut s_f = vec![0.0; p.n_live];
        s_f[br.drive_index()] = f / p.force_denominator();
        let expected = p.lu.solve(&s_f).unwrap();
        for (i, e) in expected.iter().enumerate() {
            let got = br.plate().accel[i] - free.accel[i];
            assert!((got - e).abs() <= 1e-9, "{b:?}: node {i}: {got:e} vs {e:e}");
        }
        let naive = free.pressure();
        assert!(
            (br.pressure() - naive).abs() > 1e-9 * naive.abs(),
            "{b:?}: pressure() equals the unforced read-out"
        );
    }
}

/// Carries `test_free_plate_connection.py::test_no_rigid_body_drift`: the single-point spring
/// cannot pump the free plate's `{1, x, y}` nullspace into a ramp.
#[test]
fn a_free_plate_does_not_drift_rigidly() {
    let mut br = Grid::new(PB::Free).stiffness(6000.0).plucked();
    let w = br.plate().p.w.clone();
    let area: f64 = w.iter().sum();
    let mut piston = Vec::new();
    for n in 0..seconds(&br, 1.5) {
        br.step(&mut ()).unwrap();
        if n % 500 == 0 {
            let m: f64 = w.iter().zip(br.plate().u.iter()).map(|(a, b)| a * b).sum();
            piston.push(m / area);
        }
    }
    let max = piston.iter().fold(0.0f64, |a, v| a.max(v.abs()));
    assert!(max < 1e-2, "piston mode grew to {max:.2e}");
    let half = piston.len() / 2;
    let early = piston[..half].iter().fold(0.0f64, |a, v| a.max(v.abs()));
    let late = piston[half..].iter().fold(0.0f64, |a, v| a.max(v.abs()));
    assert!(
        late <= 3.0 * early + 1e-9,
        "piston ramps: late {late:.2e} >> early {early:.2e}"
    );
}

// -- the guard ------------------------------------------------------------------------------------

/// Carries `test_unstable_stiffness_rejected` from both linear files.
#[test]
fn an_over_stiff_spring_is_rejected() {
    for b in BOUNDARIES {
        let err = Grid::new(b).stiffness(1e7).try_build().unwrap_err();
        assert!(
            matches!(err, BridgeError::UnstableMargin(_)),
            "{b:?}: {err:?}"
        );
        assert!(err.to_string().contains("unstable"));
    }
}

/// Carries `test_margin_is_linear_in_stiffness` from both linear files.
#[test]
fn the_margin_is_linear_in_the_stiffness() {
    for b in BOUNDARIES {
        let m1 = Grid::new(b).stiffness(1000.0).build().stability_margin();
        let m3 = Grid::new(b).stiffness(3000.0).build().stability_margin();
        assert!(
            ((m3 - 3.0 * m1) / (3.0 * m1)).abs() < 1e-9,
            "{b:?}: {m3:e} vs {:e}",
            3.0 * m1
        );
    }
}

/// Carries `test_guard_holds_at_its_boundary` from both linear files.
#[test]
fn the_guard_holds_just_inside_its_boundary() {
    for b in BOUNDARIES {
        let k_in = 0.93 * Grid::new(b).ceiling();
        let mut br = Grid::new(b).stiffness(k_in).plucked();
        assert!(br.stability_margin() < 1.0);
        let steps = seconds(&br, 1.2);
        let d = drift(&mut br, steps);
        assert!(d < DRIFT_TOL, "{b:?}: drift {d:.2e} near the boundary");
    }
}

/// Carries `test_just_over_the_ceiling_is_rejected` from both linear files.
#[test]
fn just_over_the_ceiling_is_rejected() {
    for b in BOUNDARIES {
        let kc = Grid::new(b).ceiling();
        Grid::new(b).stiffness(0.95 * kc).build();
        let err = Grid::new(b).stiffness(1.05 * kc).try_build().unwrap_err();
        assert!(
            matches!(err, BridgeError::UnstableMargin(_)),
            "{b:?}: {err:?}"
        );
    }
}

/// Carries `test_ceiling_is_the_true_instability_onset` from both linear files: just inside
/// conserves, and a spring raised just past the ceiling *after* construction actually diverges —
/// the guard is the physical onset, not merely self-consistent.
#[test]
fn the_ceiling_is_the_true_instability_onset() {
    for b in BOUNDARIES {
        let kc = Grid::new(b).ceiling();
        let mut inside = Grid::new(b).stiffness(0.99 * kc).plucked();
        let steps = seconds(&inside, 1.0);
        let d = drift(&mut inside, steps);
        assert!(d < DRIFT_TOL, "{b:?}: 0.99x should conserve, drift {d:.2e}");

        let mut over = Grid::new(b).stiffness(0.5 * kc).plucked();
        over.set_stiffness_unguarded(1.05 * kc);
        for _ in 0..seconds(&over, 0.3) {
            over.step(&mut ()).unwrap();
        }
        assert!(
            !over.string().u.iter().all(|v| v.is_finite()),
            "{b:?}: did not diverge just past the ceiling"
        );
    }
}

/// **New.** The guard's value on the suite's fixture, frozen from the binding (SciPy's `spsolve`
/// and `splu`): the supported margin at `K = 3000` and the free one at `K = 6000`. Both agreed to
/// the bit when measured; a tolerance rather than a bit pattern, because the solves are different
/// implementations. Without it the `G0` assembly would only ever be checked against itself.
#[test]
fn the_guard_reproduces_the_binding_measured_margins() {
    for (b, k, frozen) in [
        (PB::Supported, 3000.0, 0.214_778_354_483_558_45),
        (PB::Free, 6000.0, 0.429_556_708_967_118_3),
    ] {
        let m = Grid::new(b).stiffness(k).build().stability_margin();
        assert!(((m - frozen) / frozen).abs() < 1e-12, "{b:?}: {m:.17e}");
    }
}

// -- construction refusals ------------------------------------------------------------------------

/// Carries `test_string_at_lambda_one_rejected`.
#[test]
fn a_string_at_the_courant_limit_is_rejected() {
    let err = Grid::new(PB::Supported).lam(1.0).try_build().unwrap_err();
    assert_eq!(err, BridgeError::LambdaAtLimit);
    assert!(err.to_string().contains("lambda < 1"));
}

/// Carries `test_mismatched_timestep_rejected` from both linear files.
#[test]
fn a_mismatched_timestep_is_rejected() {
    let s = free_string(1.0, 200.0, 100, 22000.0, 0.0);
    let p = grid_plate(1.0, 20.0, 0.005, 16, 24000.0, 0.0, PB::Supported);
    let err = StringPlateBridge::new(s, p, 1000.0, None).unwrap_err();
    assert!(
        matches!(err, BridgeError::TimestepMismatch("plate", _, _)),
        "{err:?}"
    );
    assert!(err.to_string().contains("timestep"));
}

/// Carries `test_right_end_must_be_free` from both linear files.
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
    let p = grid_plate(1.0, 20.0, 0.005, 16, fs, 0.0, PB::Supported);
    let err = StringPlateBridge::new(s, p, 1000.0, None).unwrap_err();
    assert_eq!(err, BridgeError::RightEndNotFree("plate"));
}

/// Carries `test_drive_index_out_of_range_rejected` from both linear files.
#[test]
fn a_drive_index_off_the_plate_is_rejected() {
    let fs = 22000.0;
    let p = grid_plate(1.0, 20.0, 0.005, 16, fs, 0.0, PB::Supported);
    let n_live = p.p.n_live;
    let s = free_string(1.0, 200.0, 100, fs, 0.0);
    let err = StringPlateBridge::new(s, p, 1000.0, Some(n_live)).unwrap_err();
    assert_eq!(err, BridgeError::DriveIndexOutOfRange(n_live, n_live));
    assert!(err.to_string().contains("drive_index"));
}

/// **New.** A negative stiffness is refused.
#[test]
fn a_negative_stiffness_is_rejected() {
    let err = Grid::new(PB::Free).stiffness(-1.0).try_build().unwrap_err();
    assert_eq!(err, BridgeError::NegativeStiffness);
}

/// **New.** The default drive node is the live node nearest `(0.3 Lx, 0.4 Ly)`. Measured through
/// the binding on the suite's fixtures: 79 supported, 107 free (a free plate keeps its rim).
#[test]
fn the_default_drive_node_is_the_corner_offset_point() {
    assert_eq!(Grid::new(PB::Supported).build().drive_index(), 79);
    assert_eq!(Grid::new(PB::Free).build().drive_index(), 107);
}

// =================================================================================================
// The von Karman plate — `tests/helpers.py::make_vk_plate_bridge`
// =================================================================================================

/// A thin steel sheet, 40 cm square, 0.1 mm thick; `rho` is **volumetric**.
const VK_SIDE: f64 = 0.4;
const VK_E: f64 = 1.0e-4;
const VK_K_DEFAULT: f64 = 3000.0;

#[derive(Clone, Copy)]
struct Vk {
    boundary: PB,
    nonlinear: bool,
    k: f64,
    sigma_string: f64,
    sigma_plate: f64,
    max_iter: i64,
    method: CoupleMethod,
}

impl Vk {
    fn new(boundary: PB) -> Vk {
        Vk {
            boundary,
            nonlinear: true,
            k: VK_K_DEFAULT,
            sigma_string: 0.0,
            sigma_plate: 0.0,
            max_iter: VkSpec::default().couple_max_iter,
            method: VkSpec::default().couple_method.expect("a default method"),
        }
    }

    fn linear(mut self) -> Vk {
        self.nonlinear = false;
        self
    }

    fn stiffness(mut self, k: f64) -> Vk {
        self.k = k;
        self
    }

    fn fs(&self) -> f64 {
        200.0 * 100.0 / 0.9
    }

    fn plate(&self) -> VkPlate {
        VkPlate::new(
            VkParams::new(&VkSpec {
                lx: VK_SIDE,
                ly: VK_SIDE,
                young: 2.0e11,
                thickness: VK_E,
                nu: 0.3,
                rho: 7800.0,
                fs: self.fs(),
                n: 16,
                sigma: self.sigma_plate,
                boundary: Some(self.boundary),
                nonlinear: self.nonlinear,
                couple_tol: 1e-13,
                couple_max_iter: self.max_iter,
                couple_method: Some(self.method),
                ..VkSpec::default()
            })
            .expect("valid von Karman plate"),
        )
    }

    fn build(&self) -> StringPlateBridge<VkPlate> {
        let s = free_string(1.0, 200.0, 100, self.fs(), self.sigma_string);
        StringPlateBridge::new(s, self.plate(), self.k, None).expect("inside the guard")
    }

    fn plucked(&self, amplitude: f64) -> StringPlateBridge<VkPlate> {
        let mut br = self.build();
        pluck(br.string_mut(), amplitude);
        br
    }

    fn ceiling(&self) -> f64 {
        1000.0 / self.stiffness(1000.0).build().stability_margin()
    }
}

/// `vk_linear_twin`: the `Plate` a linear `VkPlate` must equal, built from the von Karman plate's
/// own numbers — the snapped `Ly`, `kappa = sqrt(D / rho_s)`, `nu`, and `rho = rho_s`, the areal
/// density. `rho` is a parameter so a bar can put the volume density there instead.
fn linear_twin(vk: &VkPlate, rho: f64) -> Plate {
    let l = &vk.p.lin;
    Plate::new(
        plate::Params::new(&PlateSpec {
            lx: l.lx,
            ly: l.ly,
            kappa: l.kappa,
            rho,
            fs: l.fs,
            n: l.n as i64,
            sigma: l.sigma,
            theta: l.theta,
            boundary: Some(l.boundary),
            nu: Some(l.nu),
            ..PlateSpec::default()
        })
        .expect("valid twin"),
    )
}

fn twin_bridge(br: &StringPlateBridge<VkPlate>, rho: f64) -> StringPlateBridge<Plate> {
    let s = br.string().params();
    let string = free_string(s.l, s.t, s.n as i64, s.fs, s.sigma);
    StringPlateBridge::new(
        string,
        linear_twin(br.plate(), rho),
        br.stiffness(),
        Some(br.drive_index()),
    )
    .expect("inside the guard")
}

/// Carries `test_nonlinear_false_is_the_linear_bridge_bit_identical` (both boundaries, lossless
/// and lossy): state, energy, force and margin, against an independently built linear `Plate`.
#[test]
fn nonlinear_false_is_the_linear_bridge_bit_identical() {
    for b in BOUNDARIES {
        for sigma in [0.0, 2.0] {
            let mut vk = Vk::new(b).linear();
            vk.sigma_plate = sigma;
            let mut nl = vk.plucked(3e-4);
            let rho_s = nl.plate().p.rho_s;
            let mut lin = twin_bridge(&nl, rho_s);
            pluck(lin.string_mut(), 3e-4);
            assert_eq!(nl.stability_margin(), lin.stability_margin());
            for n in 0..300 {
                nl.step(&mut ()).unwrap();
                lin.step(&mut ()).unwrap();
                assert_eq!(nl.string().u, lin.string().u, "{b:?} string, step {n}");
                assert_eq!(nl.plate().u, lin.plate().u, "{b:?} plate, step {n}");
            }
            assert_eq!(nl.energy(), lin.energy());
            assert_eq!(nl.connection_force(), lin.connection_force());
        }
    }
}

/// Carries `test_the_margin_ignores_the_nonlinearity_and_uses_the_areal_density`, **sharpened**:
/// the Python asserted the nonlinear margin equals the areal-density twin's, which on its own
/// could not rule out a margin that ignores the density altogether. The twin built with the
/// *volume* density must now come out different (0.1394 against 0.1425 through the binding).
///
/// The supported and free margins happen to agree to the last digit on this rig: the drive node
/// is interior, where both lumped masses are `rho_s h^2`, and the theta-excess stiffness is below
/// the string term's rounding at this timestep.
#[test]
fn the_margin_ignores_the_nonlinearity_and_uses_the_areal_density() {
    for b in BOUNDARIES {
        let nl = Vk::new(b).build();
        let p = &nl.plate().p;
        assert!(p.nonlinear);
        assert_eq!(p.rho_s, p.rho_v * p.thickness);
        assert_eq!(
            nl.stability_margin(),
            twin_bridge(&nl, p.rho_s).stability_margin()
        );
        let trap = twin_bridge(&nl, p.rho_v).stability_margin();
        let rel = (trap - nl.stability_margin()).abs() / nl.stability_margin();
        assert!(
            rel > 1e-3,
            "{b:?}: the margin cannot see the density (rel {rel:.1e})"
        );
    }
}

/// Carries `test_total_energy_conserved_lossless`, asserted with the convergence flag.
#[test]
fn the_von_karman_total_is_conserved_lossless() {
    for b in BOUNDARIES {
        let mut br = Vk::new(b).plucked(1e-3);
        let e0 = br.energy();
        let (mut lo, mut hi, mut worst_iters) = (e0, e0, 0usize);
        for _ in 0..2500 {
            br.step(&mut ()).unwrap();
            assert!(
                br.plate().converged,
                "{b:?}: the drift is not a valid measurement"
            );
            worst_iters = worst_iters.max(br.plate().n_iters);
            lo = lo.min(br.energy());
            hi = hi.max(br.energy());
        }
        assert!(worst_iters > 1, "{b:?}: the nonlinearity never engaged");
        let d = (hi - lo) / e0.abs();
        assert!(d < DRIFT_TOL, "{b:?}: drift {d:.2e}");
    }
}

/// Carries `test_nonlinearity_is_genuinely_engaged`: the string alone drives the plate past its
/// own thickness.
#[test]
fn the_string_drives_the_plate_past_its_thickness() {
    for b in BOUNDARIES {
        let mut br = Vk::new(b).plucked(1e-3);
        let mut peak: f64 = 0.0;
        for _ in 0..2500 {
            br.step(&mut ()).unwrap();
            peak = br.plate().u.iter().fold(peak, |a, v| a.max(v.abs()));
        }
        assert!(peak > VK_E, "{b:?}: peak {peak:.2e} never reached e");
        assert!(br.plate().membrane_energy() > 0.0);
    }
}

/// Carries `test_passivity_with_loss` (both boundaries, loss on either part).
#[test]
fn a_lossy_von_karman_chain_decreases_monotonically() {
    for b in BOUNDARIES {
        for lossy_plate in [true, false] {
            let mut vk = Vk::new(b);
            if lossy_plate {
                vk.sigma_plate = 3.0;
            } else {
                vk.sigma_string = 1.0;
            }
            let mut br = vk.plucked(1e-3);
            assert_monotone(&mut br, 1200);
        }
    }
}

/// Carries `test_zero_stiffness_decouples_bit_identically`.
#[test]
fn at_zero_stiffness_the_von_karman_bridge_decouples_bit_identically() {
    for b in BOUNDARIES {
        let vk = Vk::new(b).stiffness(0.0);
        let mut br = vk.plucked(1e-3);
        let mut alone = free_string(1.0, 200.0, 100, vk.fs(), 0.0);
        pluck(&mut alone, 1e-3);
        for _ in 0..200 {
            br.step(&mut ()).unwrap();
            alone.step();
            assert_eq!(br.string().u, alone.u);
        }
        assert_eq!(br.stability_margin(), 0.0);
        assert!(
            br.plate().u.iter().all(|&v| v == 0.0),
            "{b:?}: the plate was driven"
        );
    }
}

/// Carries `test_vk_connection.py::test_string_energy_alone_is_not_conserved`.
#[test]
fn the_string_energy_alone_is_not_conserved_on_the_gong() {
    let mut br = Vk::new(PB::Supported).plucked(1e-3);
    let e0 = br.string().energy();
    let (mut lo, mut hi) = (e0, e0);
    for _ in 0..1500 {
        br.step(&mut ()).unwrap();
        lo = lo.min(br.string().energy());
        hi = hi.max(br.string().energy());
    }
    assert!((hi - lo) / e0 > 1e-3);
}

/// Carries `test_a_linear_body_scales_bit_exactly_with_the_pluck`: doubling the pluck doubles
/// every quantity exactly, a machine-precision zero for the nonlinear departure to be measured
/// against.
#[test]
fn a_linear_body_scales_bit_exactly_with_the_pluck() {
    for b in BOUNDARIES {
        let mut a = Vk::new(b).linear().plucked(5e-5);
        let mut c = Vk::new(b).linear().plucked(1e-4);
        for n in 0..600 {
            a.step(&mut ()).unwrap();
            c.step(&mut ()).unwrap();
            let s2: Vec<f64> = a.string().u.iter().map(|v| 2.0 * v).collect();
            let p2: Vec<f64> = a.plate().u.iter().map(|v| 2.0 * v).collect();
            assert_eq!(s2, c.string().u, "{b:?}: string not exact at step {n}");
            assert_eq!(p2, c.plate().u, "{b:?}: plate not exact at step {n}");
        }
    }
}

/// Carries `test_departure_from_a_linear_body_is_second_order_in_the_pluck`: measured against the
/// same plate with the coupling off, so the linear control above makes the zero exact.
#[test]
fn the_departure_from_a_linear_body_is_second_order_in_the_pluck() {
    for b in BOUNDARIES {
        let amps = [1e-5, 2e-5, 5e-5];
        let mut dists = Vec::new();
        for amp in amps {
            let mut lin = Vk::new(b).linear().plucked(amp);
            let mut nl = Vk::new(b).plucked(amp);
            let (mut num, mut den): (f64, f64) = (0.0, 0.0);
            for _ in 0..500 {
                lin.step(&mut ()).unwrap();
                nl.step(&mut ()).unwrap();
                for (x, y) in nl.string().u.iter().zip(lin.string().u.iter()) {
                    num = num.max((x - y).abs());
                    den = den.max(y.abs());
                }
            }
            assert!(nl.plate().converged);
            dists.push(num / den);
        }
        assert!(dists[0] > 0.0, "{b:?}: no departure at all");
        for i in 0..2 {
            let order = (dists[i + 1] / dists[i]).ln() / (amps[i + 1] / amps[i]).ln();
            assert!(
                1.6 < order && order < 2.2,
                "{b:?}: departure order {order:.2}"
            );
        }
    }
}

/// Carries `test_linear_energy_share_is_amplitude_invariant_and_the_gong_s_is_not`.
#[test]
fn the_linear_energy_share_is_amplitude_invariant_and_the_gongs_is_not() {
    fn peak_share(mut br: StringPlateBridge<VkPlate>) -> f64 {
        let mut best: f64 = 0.0;
        for _ in 0..800 {
            br.step(&mut ()).unwrap();
            let total = br.string().energy() + br.plate().energy();
            if total > 0.0 {
                best = best.max(br.plate().energy() / total);
            }
        }
        best
    }
    for b in BOUNDARIES {
        let amps = [1e-5, 1e-4, 1e-3];
        let lin: Vec<f64> = amps
            .iter()
            .map(|&a| peak_share(Vk::new(b).linear().plucked(a)))
            .collect();
        let nl: Vec<f64> = amps
            .iter()
            .map(|&a| peak_share(Vk::new(b).plucked(a)))
            .collect();
        for l in &lin[1..] {
            assert!(
                ((l - lin[0]) / lin[0]).abs() <= 1e-12,
                "{b:?}: linear share moved"
            );
        }
        assert!(
            (nl[2] - nl[0]).abs() / nl[0] > 1e-3,
            "{b:?}: the gong's share did not move"
        );
    }
}

/// Carries `test_rigid_modes_are_immune_to_the_nonlinearity` — a claim about the von Karman plate
/// itself, which retired with this file and has no other native bar: started in a pure tilt, the
/// free plate stays tilted, because the bracket of second derivatives is zero on `{1, x, y}`.
#[test]
fn rigid_modes_are_immune_to_the_nonlinearity() {
    let br = Vk::new(PB::Free).stiffness(0.0).build();
    let mut p = br.plate().clone();
    let lx = p.p.lin.lx;
    let tilt: Vec<f64> = live_xy(&p.p.lin)
        .iter()
        .map(|(x, _)| (x - 0.5 * lx) * 2.5e-2)
        .collect();
    p.set_state(&tilt, &vec![0.0; tilt.len()]).unwrap();
    assert_eq!(
        p.f.iter().fold(0.0f64, |a, v| a.max(v.abs())),
        0.0,
        "F(w^0) must be exactly zero"
    );
    for _ in 0..400 {
        p.step(None).unwrap();
    }
    assert!(
        p.f.iter().fold(0.0f64, |a, v| a.max(v.abs())) < 1e-18,
        "the nonlinearity woke up"
    );
    let dev =
        p.u.iter()
            .zip(&tilt)
            .fold(0.0f64, |a, (u, t)| a.max((u - t).abs()));
    assert!(dev < 1e-12 * VK_SIDE, "tilt moved by {dev:e}");
    assert!(p.energy().abs() < 1e-15);
}

/// Carries `test_guard_rejects_an_overstiff_spring` (both boundaries).
#[test]
fn the_von_karman_guard_rejects_an_over_stiff_spring() {
    for b in BOUNDARIES {
        let ceiling = Vk::new(b).ceiling();
        let ok = Vk::new(b).stiffness(0.95 * ceiling).build();
        assert!((ok.stability_margin() - 0.95).abs() / 0.95 < 1e-9);
        let s = free_string(1.0, 200.0, 100, Vk::new(b).fs(), 0.0);
        let err = StringPlateBridge::new(s, Vk::new(b).plate(), 1.05 * ceiling, None).unwrap_err();
        assert!(err.to_string().contains("connection unstable"), "{err}");
    }
}

/// Carries `test_the_linear_margin_survives_a_strongly_nonlinear_run`: at 95% of the ceiling the
/// total still conserves at large amplitude, because the membrane energy is a sum of squares.
#[test]
fn the_linear_margin_survives_a_strongly_nonlinear_run() {
    let ceiling = Vk::new(PB::Supported).ceiling();
    let mut br = Vk::new(PB::Supported)
        .stiffness(0.95 * ceiling)
        .plucked(1e-3);
    let e0 = br.energy();
    let (mut lo, mut hi) = (e0, e0);
    for _ in 0..1500 {
        br.step(&mut ()).unwrap();
        assert!(br.plate().converged);
        lo = lo.min(br.energy());
        hi = hi.max(br.energy());
    }
    assert!((hi - lo) / e0.abs() < DRIFT_TOL);
}

/// Carries `test_the_failure_mode_migrates_to_non_convergence`: a configuration the guard passes
/// still hits the sweep cap, and the guard — a statement about a quadratic form — cannot see it.
#[test]
fn the_failure_mode_migrates_to_non_convergence() {
    let ceiling = Vk::new(PB::Supported).ceiling();
    let mut vk = Vk::new(PB::Supported).stiffness(0.9 * ceiling);
    vk.max_iter = 12;
    vk.method = CoupleMethod::Picard;
    let mut br = vk.plucked(3e-2);
    assert!(br.stability_margin() < 1.0, "the guard passed it");
    let mut failed = false;
    for _ in 0..400 {
        br.step(&mut ()).unwrap();
        if !br.plate().converged {
            failed = true;
            break;
        }
    }
    assert!(failed, "expected the sweep cap to be reached");
    assert_eq!(br.plate().n_iters, 12);
    assert!(br.plate().last_residual > br.plate().p.couple_tol);
}

/// Carries `test_construction_rejects_mismatched_and_malformed_inputs`, in the original's order.
#[test]
fn the_von_karman_bridge_refuses_malformed_inputs() {
    let vk = Vk::new(PB::Supported);
    let fs = vk.fs();
    let p = vk.plate();
    let n_live = p.p.lin.n_live;
    let other = free_string(1.0, 200.0, 100, 2.0 * fs, 0.0);
    let err = StringPlateBridge::new(other, p.clone(), 1.0, None).unwrap_err();
    assert!(err.to_string().contains("share a timestep"), "{err}");
    let clamped = IdealString::new(
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
    let err = StringPlateBridge::new(clamped, p.clone(), 1.0, None).unwrap_err();
    assert!(
        err.to_string().contains("right end must be 'free'"),
        "{err}"
    );
    let s = || free_string(1.0, 200.0, 100, fs, 0.0);
    let err = StringPlateBridge::new(s(), p.clone(), -1.0, None).unwrap_err();
    assert!(err.to_string().contains("must be >= 0"), "{err}");
    let err = StringPlateBridge::new(s(), p, 1.0, Some(n_live)).unwrap_err();
    assert!(err.to_string().contains("out of range"), "{err}");
}

// =================================================================================================
// The chains: string -> bridge -> room-loaded plate -> room
// =================================================================================================

const ROOM_FS: f64 = 8000.0;
const CHAIN_K: f64 = 800.0;
const DIPOLE_INDEX: i64 = 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Tier {
    Baffled,
    Suspended,
}

const TIERS: [Tier; 2] = [Tier::Baffled, Tier::Suspended];

fn all_lossy() -> [Wall; 6] {
    [Wall::Impedance(impedance_from_zeta(4.0, RHO0_AIR, C0_AIR)); 6]
}

fn surface_room(walls: [Wall; 6]) -> AirBox {
    let h = C0_AIR * 3.0f64.sqrt() / (0.9 * ROOM_FS);
    AirBox::new(
        RoomParams::new(
            [12.0 * h, 11.0 * h, 9.0 * h],
            ROOM_FS,
            h,
            walls,
            None,
            RHO0_AIR,
            C0_AIR,
        )
        .expect("a legal room"),
    )
}

fn mount<S: GridSeam>(tier: Tier, seam: S, room: &mut AirBox) -> RoomGrid<S> {
    match tier {
        Tier::Baffled => RoomGrid::baffled(seam, room, "z0", None, Spreading::Bilinear),
        Tier::Suspended => {
            RoomGrid::suspended(seam, room, "z", DIPOLE_INDEX, None, Spreading::Bilinear)
        }
    }
    .expect("a legal mounting")
}

/// `make_vk_room_string`: `L = 0.6`, `T = 60`, `N = 40`, with `N` scaled to hold `lambda` when
/// `L` moves the string's band.
fn room_string(l: f64) -> IdealString {
    let n = (l * 40.0 / 0.6).round() as i64;
    free_string(l, 60.0, n, ROOM_FS, 0.0)
}

/// `make_air_vk_plate`: 30 cm, 1 mm steel, `N = 8`, at the room's rate.
fn air_vk_plate(b: PB, nonlinear: bool, sigma: f64) -> VkPlate {
    VkPlate::new(
        VkParams::new(&VkSpec {
            lx: 0.30,
            ly: 0.30,
            young: 2.0e11,
            thickness: 1.0e-3,
            nu: 0.3,
            rho: 7800.0,
            fs: ROOM_FS,
            n: 8,
            sigma,
            boundary: Some(b),
            nonlinear,
            couple_tol: 1e-13,
            ..VkSpec::default()
        })
        .expect("valid gong"),
    )
}

#[derive(Clone, Copy)]
struct Chain {
    tier: Tier,
    boundary: PB,
    k: f64,
    walls: [Wall; 6],
    nonlinear: bool,
    sigma: f64,
    string_l: f64,
    drive: Option<usize>,
}

impl Chain {
    fn new(tier: Tier, boundary: PB) -> Chain {
        Chain {
            tier,
            boundary,
            k: CHAIN_K,
            walls: [Wall::Rigid; 6],
            nonlinear: true,
            sigma: 0.0,
            string_l: 0.6,
            drive: None,
        }
    }

    fn lossy(mut self) -> Chain {
        self.walls = all_lossy();
        self
    }

    fn build(&self) -> (AirBox, StringPlateBridge<RoomGrid<VkSeam>>) {
        let mut room = surface_room(self.walls);
        let seam = VkSeam::new(air_vk_plate(self.boundary, self.nonlinear, self.sigma));
        let inst = mount(self.tier, seam, &mut room);
        let br = StringPlateBridge::new(room_string(self.string_l), inst, self.k, self.drive)
            .expect("inside the guard");
        (room, br)
    }

    fn plucked(&self, amplitude: f64) -> (AirBox, StringPlateBridge<RoomGrid<VkSeam>>) {
        let (room, mut br) = self.build();
        sine_pluck(br.string_mut(), amplitude);
        (room, br)
    }
}

/// `make_vk_room_bare_twin`: the unloaded bridge, from the loaded one's own numbers, drive copied.
fn bare_twin(br: &StringPlateBridge<RoomGrid<VkSeam>>) -> StringPlateBridge<VkPlate> {
    let s = br.string().params();
    let string = free_string(s.l, s.t, s.n as i64, s.fs, s.sigma);
    let plate = VkPlate::new(br.plate().seam.plate.p.clone());
    StringPlateBridge::new(string, plate, br.stiffness(), Some(br.drive_index())).unwrap()
}

fn scene_total<P: BridgePlate>(br: &StringPlateBridge<P>, room: &AirBox) -> f64 {
    br.energy() + room.energy()
}

/// Carries `test_airbox_vk.py::test_the_chain_composes_and_the_guard_is_bit_identical` (both
/// boundaries, both tiers): the loaded margin equals the bare twin's — the guard reads the plate's
/// parameters, never the loaded factorization — and the scene total is flat with the convergence
/// flag read every step.
#[test]
fn the_gong_chain_composes_and_the_guard_ignores_the_room() {
    for tier in TIERS {
        for b in BOUNDARIES {
            let (mut room, mut br) = Chain::new(tier, b).lossy().plucked(1e-3);
            assert_eq!(br.stability_margin(), bare_twin(&br).stability_margin());
            let e0 = scene_total(&br, &room);
            let mut worst: f64 = 0.0;
            for _ in 0..400 {
                br.step(&mut room).unwrap();
                room.step();
                assert!(br.plate().seam.plate.converged, "{tier:?} {b:?}");
                worst = worst.max((scene_total(&br, &room) - e0).abs());
            }
            assert!(
                worst <= 1e-12 * e0.abs(),
                "{tier:?} {b:?}: drift {:.2e}",
                worst / e0.abs()
            );
            assert!(
                br.plate().radiated_energy().abs() > 1e-7 * e0.abs(),
                "the channel is vacuous"
            );
        }
    }
}

/// Carries `test_the_money_test_holds_with_the_string_as_the_only_excitation`.
#[test]
fn the_money_test_holds_with_the_string_as_the_only_excitation() {
    for tier in TIERS {
        for b in BOUNDARIES {
            let (mut room, mut br) = Chain::new(tier, b).lossy().plucked(1e-3);
            let e0 = scene_total(&br, &room);
            for _ in 0..400 {
                br.step(&mut room).unwrap();
                room.step();
            }
            let radiated = br.plate().radiated_energy();
            assert!(
                (radiated - room.injected).abs() <= 1e-12 * radiated.abs(),
                "{tier:?} {b:?}"
            );
            assert!(radiated.abs() > 1e-7 * e0.abs());
        }
    }
}

/// Carries `test_the_room_adds_no_outer_iteration`: the loaded chain takes the same worst sweep
/// count as the bare one — `F = K eta^n` enters outside the loop and the load is in `A`.
#[test]
fn the_room_adds_no_outer_iteration() {
    for tier in TIERS {
        for b in BOUNDARIES {
            let (mut room, mut loaded) = Chain::new(tier, b).lossy().plucked(5e-2);
            let mut bare = bare_twin(&loaded);
            sine_pluck(bare.string_mut(), 5e-2);
            let (mut wl, mut wb) = (0usize, 0usize);
            for _ in 0..300 {
                loaded.step(&mut room).unwrap();
                room.step();
                bare.step(&mut ()).unwrap();
                assert!(loaded.plate().seam.plate.converged && bare.plate().converged);
                wl = wl.max(loaded.plate().seam.plate.n_iters);
                wb = wb.max(bare.plate().n_iters);
            }
            assert_eq!(wl, wb, "{tier:?} {b:?}: loaded {wl} sweeps, bare {wb}");
            assert!(
                wl > 1,
                "the nonlinear path must iterate for this to mean anything"
            );
        }
    }
}

/// Carries `test_zero_bridge_stiffness_decouples_the_chain`: with no spring the plate is exactly
/// at rest, nothing is injected, and the string is a bare string to the bit.
#[test]
fn zero_bridge_stiffness_decouples_the_chain() {
    for tier in TIERS {
        for b in BOUNDARIES {
            let mut c = Chain::new(tier, b);
            c.k = 0.0;
            let (mut room, mut br) = c.plucked(1e-3);
            let mut alone = room_string(0.6);
            let u0 = br.string().u.clone();
            alone.set_state(&u0, &vec![0.0; u0.len()]);
            for _ in 0..200 {
                br.step(&mut room).unwrap();
                room.step();
                alone.step();
            }
            assert!(
                br.plate().seam.plate.u.iter().all(|&v| v == 0.0),
                "{tier:?} {b:?}"
            );
            assert_eq!(br.plate().radiated_energy(), 0.0);
            assert_eq!(br.string().u, alone.u);
        }
    }
}

/// Carries `test_the_nonlinear_false_chain_is_the_linear_bridge_bit_identical`: the whole chain
/// with the coupling off is the linear plate's chain in an identical room, to the bit — the areal
/// density, the drive node and the force path through two different seams at once.
#[test]
fn the_nonlinear_false_chain_is_the_linear_chain_bit_identical() {
    for tier in TIERS {
        for b in BOUNDARIES {
            let mut c = Chain::new(tier, b);
            c.nonlinear = false;
            let (mut room, mut br) = c.plucked(1e-3);
            let vk = &br.plate().seam.plate;
            let mut room2 = surface_room(c.walls);
            let inst = mount(
                tier,
                PlateSeam::new(linear_twin(vk, vk.p.rho_s)),
                &mut room2,
            );
            let mut twin = StringPlateBridge::new(
                room_string(0.6),
                inst,
                br.stiffness(),
                Some(br.drive_index()),
            )
            .unwrap();
            sine_pluck(twin.string_mut(), 1e-3);
            assert_eq!(br.stability_margin(), twin.stability_margin());
            for _ in 0..200 {
                br.step(&mut room).unwrap();
                room.step();
                twin.step(&mut room2).unwrap();
                room2.step();
            }
            assert_eq!(br.string().u, twin.string().u, "{tier:?} {b:?}");
            assert_eq!(
                br.plate().seam.plate.u,
                twin.plate().seam.plate.u,
                "{tier:?} {b:?}"
            );
            assert_eq!(br.plate().radiated_energy(), twin.plate().radiated_energy());
        }
    }
}

/// Carries `test_the_lossy_chain_is_monotone`: lossy walls and a lossy plate, the scene total
/// never rises.
#[test]
fn the_lossy_chain_is_monotone() {
    for tier in TIERS {
        for b in BOUNDARIES {
            let mut c = Chain::new(tier, b).lossy();
            c.sigma = 2.0;
            let (mut room, mut br) = c.plucked(1e-3);
            let mut prev = scene_total(&br, &room);
            for _ in 0..400 {
                br.step(&mut room).unwrap();
                room.step();
                let now = scene_total(&br, &room);
                assert!(
                    now <= prev + 1e-14 * prev.abs(),
                    "{tier:?} {b:?}: {prev:e} -> {now:e}"
                );
                prev = now;
            }
        }
    }
}

/// The time-integrated share of the free plate's kinetic energy in its `{1, x, y}` nullspace,
/// and the peak `w / e` — `vk_room_rigid_share`.
fn rigid_share(
    room: &mut AirBox,
    br: &mut StringPlateBridge<RoomGrid<VkSeam>>,
    steps: usize,
) -> (f64, f64) {
    let lin = br.plate().seam.plate.p.lin.clone();
    let w = lin.w.clone();
    let xy = live_xy(&lin);
    // A W-orthonormal basis of span{1, x, y}, by Gram-Schmidt under the lumped mass.
    let mut basis: Vec<Vec<f64>> = Vec::new();
    for raw in [
        vec![1.0; xy.len()],
        xy.iter().map(|p| p.0).collect::<Vec<f64>>(),
        xy.iter().map(|p| p.1).collect::<Vec<f64>>(),
    ] {
        let mut v = raw;
        for q in &basis {
            let c: f64 = q.iter().zip(&v).zip(&w).map(|((a, b), m)| a * m * b).sum();
            for (vi, qi) in v.iter_mut().zip(q) {
                *vi -= c * qi;
            }
        }
        let norm: f64 = v.iter().zip(&w).map(|(a, m)| a * m * a).sum::<f64>().sqrt();
        v.iter_mut().for_each(|x| *x /= norm);
        basis.push(v);
    }
    let (mut rigid, mut total, mut peak) = (0.0, 0.0, 0.0f64);
    for _ in 0..steps {
        let p = &br.plate().seam.plate;
        let vel: Vec<f64> =
            p.u.iter()
                .zip(&p.u_prev)
                .map(|(a, b)| (a - b) / lin.k)
                .collect();
        for q in &basis {
            let c: f64 = q
                .iter()
                .zip(&vel)
                .zip(&w)
                .map(|((a, b), m)| a * m * b)
                .sum();
            rigid += c * c;
        }
        total += vel.iter().zip(&w).map(|(a, m)| a * m * a).sum::<f64>();
        br.step(room).unwrap();
        room.step();
        peak = br
            .plate()
            .seam
            .plate
            .u
            .iter()
            .fold(peak, |a, v| a.max(v.abs()));
    }
    (rigid / total, peak / 1.0e-3)
}

/// Carries `test_band_overlap_decides_the_rigid_share_not_the_pluck`: move only the string's
/// length — its band, at fixed wave impedance — and as its fundamental falls below the free
/// plate's first flexural mode the plate stops flexing and merely bounces; the peak `w / e` barely
/// moves while the rigid share moves twentyfold.
#[test]
fn band_overlap_decides_the_rigid_share_not_the_pluck() {
    for tier in TIERS {
        let mut shares = Vec::new();
        let mut wes = Vec::new();
        for l in [0.6, 2.4, 4.8] {
            let mut c = Chain::new(tier, PB::Free);
            c.string_l = l;
            let (mut room, mut br) = c.plucked(1e-3);
            let (share, we) = rigid_share(&mut room, &mut br, 400);
            shares.push(share);
            wes.push(we);
        }
        assert!(shares[0] < 0.10, "{tier:?}: {:.3}", shares[0]);
        assert!(shares[2] > 0.85, "{tier:?}: {:.3}", shares[2]);
        assert!(
            shares.windows(2).all(|p| p[0] <= p[1]),
            "{tier:?}: {shares:?}"
        );
        assert!(shares[2] / shares[0] > 10.0);
        let (lo, hi) = wes
            .iter()
            .fold((f64::MAX, 0.0f64), |(a, b), &v| (a.min(v), b.max(v)));
        assert!(hi / lo < 3.0, "{tier:?}: peak w/e moved {:.2}x", hi / lo);
    }
}

/// Carries `test_a_wrong_string_reaction_is_seen_by_the_total_and_not_the_money_test`: a doubled
/// `beta_s` — Newton's third law broken at the spring, upstream of every port — moves the scene
/// total and leaves the room's money test green.
#[test]
fn a_wrong_string_reaction_is_seen_by_the_total_and_not_the_money_test() {
    for tier in TIERS {
        let mut drifts = Vec::new();
        let mut gaps = Vec::new();
        for scale in [1.0, 2.0] {
            let (mut room, mut br) = Chain::new(tier, PB::Free).lossy().plucked(1e-3);
            let beta = br.beta_s();
            br.set_beta_s(beta * scale);
            let e0 = scene_total(&br, &room);
            let mut worst: f64 = 0.0;
            for _ in 0..400 {
                br.step(&mut room).unwrap();
                room.step();
                worst = worst.max((scene_total(&br, &room) - e0).abs());
            }
            drifts.push(worst / e0.abs());
            let r = br.plate().radiated_energy();
            gaps.push((r - room.injected).abs() / r.abs());
        }
        assert!(
            drifts[0] <= 1e-12 && gaps[0] <= 1e-12,
            "{tier:?}: the correct chain must be green"
        );
        assert!(
            drifts[1] > 1e-4,
            "{tier:?}: the total missed it ({:.2e})",
            drifts[1]
        );
        assert!(
            gaps[1] <= 1e-12,
            "{tier:?}: the money test saw it ({:.2e})",
            gaps[1]
        );
    }
}

/// Carries `test_every_detector_is_blind_to_a_drive_index_that_differs_between_two_runs`: two
/// individually green runs whose drive nodes differ move the nonlinear departure figure, and no
/// ledger can see it — which is why a comparison passes `drive_index` explicitly.
#[test]
fn every_detector_is_blind_to_a_drive_index_that_differs_between_two_runs() {
    for tier in TIERS {
        let (_, reference) = Chain::new(tier, PB::Free).build();
        let lin = &reference.plate().seam.plate.p.lin;
        let other = plate::pickup_index_at(0.35 * lin.lx, 0.45 * lin.ly, lin);
        let drive = reference.drive_index();
        assert_ne!(other, drive);

        let history = |nonlinear: bool, d: usize| {
            let mut c = Chain::new(tier, PB::Free);
            c.nonlinear = nonlinear;
            c.drive = Some(d);
            let (mut room, mut br) = c.plucked(5e-2);
            let e0 = scene_total(&br, &room);
            let mut hist = Vec::new();
            let mut worst: f64 = 0.0;
            for _ in 0..300 {
                br.step(&mut room).unwrap();
                room.step();
                hist.extend_from_slice(&br.plate().seam.plate.u);
                worst = worst.max((scene_total(&br, &room) - e0).abs());
            }
            let r = br.plate().radiated_energy();
            (hist, worst / e0.abs(), (r - room.injected).abs() / r.abs())
        };
        let (nl, dn, gn) = history(true, drive);
        let (matched, da, ga) = history(false, drive);
        let (mismatched, db, gb) = history(false, other);
        for (d, g) in [(dn, gn), (da, ga), (db, gb)] {
            assert!(
                d <= 1e-12 && g <= 1e-12,
                "{tier:?}: every run is individually green"
            );
        }
        let norm = |v: &[f64]| v.iter().map(|x| x * x).sum::<f64>().sqrt();
        let diff = |a: &[f64], b: &[f64]| {
            a.iter()
                .zip(b)
                .map(|(x, y)| (x - y) * (x - y))
                .sum::<f64>()
                .sqrt()
        };
        let honest = diff(&nl, &matched) / norm(&matched);
        let corrupt = diff(&nl, &mismatched) / norm(&mismatched);
        assert!(honest > 1e-3, "{tier:?}: the departure must be live");
        assert!(
            corrupt / honest > 1.3,
            "{tier:?}: moved only {:.2}x",
            corrupt / honest
        );
    }
}

/// Carries `test_airbox_surface.py::test_string_bridge_plate_room_chain` and
/// `test_airbox_dipole.py::test_string_bridge_plate_room_chain` — the **linear** plate in a room,
/// flush in a wall and hung on an interior plane, both boundaries. The margin equals the bare
/// plate's to the bit (the load is dissipative and enters `A`, never `G0` — and the dipole's face
/// cut removes air inertia from the *room's* ledger, not the plate's), and the scene total is
/// flat.
#[test]
fn the_linear_plate_chain_conserves_and_the_guard_ignores_the_room() {
    for tier in TIERS {
        for b in BOUNDARIES {
            let mut room = surface_room(all_lossy());
            let pl = grid_plate(0.30, 20.0, 0.5, 8, ROOM_FS, 0.0, b);
            let bare_plate = pl.clone();
            let inst = mount(tier, PlateSeam::new(pl), &mut room);
            let bare = StringPlateBridge::new(room_string(0.6), bare_plate, CHAIN_K, None).unwrap();
            let mut br = StringPlateBridge::new(room_string(0.6), inst, CHAIN_K, None).unwrap();
            assert_eq!(
                br.stability_margin(),
                bare.stability_margin(),
                "{tier:?} {b:?}"
            );
            sine_pluck(br.string_mut(), 1e-3);
            let e0 = scene_total(&br, &room);
            let mut worst: f64 = 0.0;
            for _ in 0..600 {
                br.step(&mut room).unwrap();
                room.step();
                worst = worst.max((scene_total(&br, &room) - e0).abs());
            }
            assert!(
                worst <= 1e-11 * e0.abs(),
                "{tier:?} {b:?}: drift {:.2e}",
                worst / e0.abs()
            );
            assert!(br.plate().radiated_energy().abs() > 1e-4 * e0.abs());
        }
    }
}

/// **New.** The four chain margins, frozen from the binding: `K = 800`, the linear plate
/// (`0.2061806714931906` supported, `0.2061840079056186` free — the numbers the Python files'
/// docstrings quoted) and the gong (`0.20440593233341828`, `0.20440593249574418`).
#[test]
fn the_chain_margins_reproduce_the_binding() {
    for (b, lin_frozen, vk_frozen) in [
        (
            PB::Supported,
            0.206_180_671_493_190_6,
            0.204_405_932_333_418_28,
        ),
        (PB::Free, 0.206_184_007_905_618_6, 0.204_405_932_495_744_18),
    ] {
        let pl = grid_plate(0.30, 20.0, 0.5, 8, ROOM_FS, 0.0, b);
        let m = StringPlateBridge::new(room_string(0.6), pl, CHAIN_K, None)
            .unwrap()
            .stability_margin();
        assert!(
            ((m - lin_frozen) / lin_frozen).abs() < 1e-12,
            "{b:?} linear: {m:.17e}"
        );
        let (_, br) = Chain::new(Tier::Suspended, b).build();
        let m = br.stability_margin();
        assert!(
            ((m - vk_frozen) / vk_frozen).abs() < 1e-12,
            "{b:?} gong: {m:.17e}"
        );
    }
}
