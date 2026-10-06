//! The von Karman plate's validation harness — model #6, the gong and the cymbal.
//!
//! Carried from `tests/test_vk_{energy,modal,stability,free}.py` (phase C, retirement plan §40),
//! at the Python's own fixture: a 40 cm square of 1 mm steel (`E = 2e11`, `nu = 0.3`,
//! `rho = 7800`) at `N = 20`, `fs = 48 kHz`, struck with a centred 8 cm Gaussian. The plate has
//! **no analytic modal oracle**, so energy is the correctness test, and the bars below are the
//! three things that stand in for a closed form: conservation at an amplitude where the membrane
//! holds a real share of the energy, a drift that falls with the coupling tolerance (a scheme bug
//! would floor it), and second-order self-convergence. Two physics limits tie the model back to
//! the validated linear plate — the collapse at vanishing amplitude and the hardening glide —
//! and both boundaries run every bar the Python ran on both.
//!
//! What was already native and is not repeated here: the `nonlinear = false` anchor
//! (`plate.rs::a_linear_von_karman_plate_is_bit_identical_to_the_linear_plate`, structural rather
//! than transcribed), the four refusals the linear plate does not have, and the operators' own
//! identities (`ops2d.rs`).

use physsynth_analysis::spectrum::{hann, rfft_mag, rfftfreq};
use physsynth_core::eig::generalized_eigen_diag;
use physsynth_core::ops2d::{biharmonic_from_mask, rectangle_mask, AiryStressSolver};
use physsynth_core::plate::{
    linspace0, pickup_index_at, Boundary, Params, Plate, PlateSpec, VkParamError, VkParams,
    VkPlate, VkSpec,
};
use physsynth_core::reduce;
use physsynth_core::sparse_lu::SparseLu;
use std::f64::consts::PI;

const YOUNG: f64 = 2.0e11;
const THICK: f64 = 1.0e-3;
const NU: f64 = 0.3;
const RHO: f64 = 7800.0;
const FS: f64 = 48_000.0;
const SIDE: f64 = 0.4;
const BOTH: [Boundary; 2] = [Boundary::Supported, Boundary::Free];

fn spec(boundary: Boundary) -> VkSpec {
    VkSpec {
        lx: SIDE,
        ly: SIDE,
        young: YOUNG,
        thickness: THICK,
        nu: NU,
        rho: RHO,
        fs: FS,
        n: 20,
        boundary: Some(boundary),
        ..VkSpec::default()
    }
}

fn vk(s: &VkSpec) -> VkPlate {
    VkPlate::new(VkParams::new(s).expect("a valid plate"))
}

/// The linear plate a `VkPlate` reduces to: same `kappa`, the areal density as `rho`.
fn linear_twin(v: &VkPlate) -> Plate {
    let lin = &v.p.lin;
    Plate::new(
        Params::new(&PlateSpec {
            lx: SIDE,
            ly: lin.ly,
            kappa: lin.kappa,
            rho: v.p.rho_s,
            fs: lin.fs,
            n: lin.n as i64,
            boundary: Some(lin.boundary),
            nu: Some(NU),
            ..PlateSpec::default()
        })
        .expect("a valid plate"),
    )
}

/// A field over the live nodes, from `f(x, y)` at each.
fn live_field(p: &Params, f: impl Fn(f64, f64) -> f64) -> Vec<f64> {
    p.mask
        .flags()
        .iter()
        .enumerate()
        .filter(|(_, &alive)| alive)
        .map(|(idx, _)| f(p.x[idx], p.y[idx]))
        .collect()
}

/// The Python's `_strike`: a centred Gaussian of width 8 cm, peak 1, times `amp`.
///
/// Spelled as the original associates it: `amp` is formed first (`3.0 * MAT["e"]`), then
/// multiplies the shape. On the supported plate the rim is zeroed, which here means it is simply
/// not a live node; on the free plate every node is live.
fn strike(p: &Params, amp: f64) -> Vec<f64> {
    let width = 0.08;
    live_field(p, |x, y| {
        let (dx, dy) = (x - 0.5 * SIDE, y - 0.5 * p.ly);
        amp * (-((dx * dx + dy * dy) / (width * width))).exp()
    })
}

/// The (1,1) simply-supported mode `sin(pi X / Lx) sin(pi Y / Ly)`, times `amp`.
fn mode11(p: &Params, lx: f64, amp: f64) -> Vec<f64> {
    live_field(p, |x, y| {
        amp * ((PI * x / lx).sin() * (PI * y / p.ly).sin())
    })
}

fn start(v: &mut VkPlate, u0: &[f64]) {
    let zero = vec![0.0; u0.len()];
    v.set_state(u0, &zero).expect("the Airy solve factors");
}

fn step(v: &mut VkPlate) {
    v.step(None).expect("the solves succeed");
}

/// NaN-propagating max, as `np.max` and Python's `max` over a comparison chain behave.
fn nan_max(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else {
        a.max(b)
    }
}

/// `(emax - emin) / |e0|` over `steps` steps.
fn drift(v: &mut VkPlate, steps: usize) -> f64 {
    let e0 = v.energy();
    let (mut lo, mut hi) = (e0, e0);
    for _ in 0..steps {
        step(v);
        let e = v.energy();
        lo = if e.is_nan() { f64::NAN } else { lo.min(e) };
        hi = nan_max(hi, e);
    }
    (hi - lo) / e0.abs()
}

// -- energy: the headline and its two certificates -----------------------------------------------

/// The headline: lossless drift at a peak of three thicknesses, where the membrane holds a real
/// share of the energy (a bracket or averaging bug hides at small amplitude, where the run only
/// re-tests the linear theta-scheme). Carries the Python's drift, non-negativity and finiteness
/// bars on both boundaries, which ran the same 600 steps of the same strike.
///
/// The membrane share is read where each Python file read it: at the moment of peak deflection on
/// the supported plate (0.5686), and at the start on the free one (0.5684 — at its peak, which
/// sits on a free rim, it is 0.089). Drifts: 1.71e-13 supported, 2.13e-13 free.
#[test]
fn the_lossless_drift_at_three_thicknesses_meets_the_contract_on_both_boundaries() {
    for boundary in BOTH {
        let mut v = vk(&spec(boundary));
        let u0 = strike(&v.p.lin, 3.0 * THICK);
        start(&mut v, &u0);
        let e0 = v.energy();
        let share0 = v.membrane_energy() / e0;
        let (mut lo, mut hi) = (e0, e0);
        let (mut peak, mut share_at_peak) = (0.0f64, 0.0);
        for n in 0..600 {
            step(&mut v);
            let e = v.energy();
            assert!(
                e.is_finite() && e >= 0.0,
                "{boundary:?} step {n}: energy {e}"
            );
            assert!(
                v.u.iter().all(|x| x.is_finite()),
                "{boundary:?} step {n}: a non-finite node"
            );
            lo = lo.min(e);
            hi = hi.max(e);
            let w = v.state().iter().fold(0.0f64, |m, x| m.max(x.abs()));
            if w > peak {
                peak = w;
                share_at_peak = v.membrane_energy() / e;
            }
        }
        let d = (hi - lo) / e0.abs();
        eprintln!(
            "{boundary:?}: drift {d:e}, share at start {share0}, at peak {share_at_peak}, peak {} e",
            peak / THICK
        );
        match boundary {
            Boundary::Supported => assert!(share_at_peak > 0.1, "share at peak {share_at_peak}"),
            Boundary::Free => assert!(share0 > 0.1, "share at the start {share0}"),
        }
        assert!(d < 1e-10, "{boundary:?}: drift {d:.3e}");
    }
}

/// The self-certifying gate: exact conservation holds only AT the coupled step's fixed point, so
/// the drift is bounded by the tolerance and must fall with it. A scheme bug would floor it
/// whatever the tolerance; this is the machine-precision certificate standing in for the absent
/// closed form. Recorded from the retired Python (supported / free): 3.48e-5 / 3.48e-5,
/// 1.75e-8 / 1.73e-8, 9.46e-13 / 1.06e-12.
#[test]
fn the_drift_falls_with_the_coupling_tolerance_on_both_boundaries() {
    for boundary in BOTH {
        let drifts: Vec<f64> = [1e-4, 1e-8, 1e-12]
            .iter()
            .map(|&tol| {
                let mut v = vk(&VkSpec {
                    couple_tol: tol,
                    ..spec(boundary)
                });
                let u0 = strike(&v.p.lin, 3.0 * THICK);
                start(&mut v, &u0);
                drift(&mut v, 400)
            })
            .collect();
        eprintln!("{boundary:?}: drifts {drifts:?}");
        assert!(
            drifts[0] > drifts[1] && drifts[1] > drifts[2],
            "{boundary:?}: not monotone in the tolerance: {drifts:?}"
        );
        assert!(
            drifts[2] < 1e-10,
            "{boundary:?}: tightest drift {:.3e}",
            drifts[2]
        );
    }
}

/// Passivity: `sigma = 8`, and the total never rises over 800 steps (beyond the Python's
/// `1e-9 e0` allowance) and does fall. Measured: every step FALLS, by at least 4.0e-11 of the
/// start on either boundary, and 800 steps keep 0.753 / 0.751 of it.
#[test]
fn a_lossy_plate_is_passive_on_both_boundaries() {
    for boundary in BOTH {
        let mut v = vk(&VkSpec {
            sigma: 8.0,
            ..spec(boundary)
        });
        let u0 = strike(&v.p.lin, 3.0 * THICK);
        start(&mut v, &u0);
        let e0 = v.energy();
        let mut prev = e0;
        for n in 0..800 {
            step(&mut v);
            let cur = v.energy();
            assert!(
                cur <= prev + 1e-9 * e0,
                "{boundary:?} step {n}: rose by {:.3e}",
                cur - prev
            );
            prev = cur;
        }
        eprintln!("{boundary:?}: kept {} of the start", prev / e0);
        assert!(prev < e0, "{boundary:?}: no decay");
    }
}

/// The two halves of the ledger: `energy() = linear_energy() + membrane_energy()` (exactly — it
/// is defined as that sum, so the Python's `approx` is sharpened), with a real membrane store at
/// two thicknesses; and with the coupling off the membrane half is exactly zero.
#[test]
fn the_energy_is_the_linear_plus_the_membrane_part_and_the_linear_plate_has_no_membrane() {
    for boundary in BOTH {
        for nonlinear in [true, false] {
            let mut v = vk(&VkSpec {
                nonlinear,
                ..spec(boundary)
            });
            let u0 = strike(&v.p.lin, 2.0 * THICK);
            start(&mut v, &u0);
            for _ in 0..20 {
                step(&mut v);
            }
            assert_eq!(v.energy(), v.linear_energy() + v.membrane_energy());
            if nonlinear {
                assert!(v.membrane_energy() > 0.0, "{boundary:?}: no membrane store");
            } else {
                assert_eq!(v.membrane_energy(), 0.0);
                assert_eq!(v.energy(), v.linear_energy());
            }
        }
    }
}

/// In the gate regime (`w ~ e`) the coupled step reaches its tolerance on every one of 300 steps,
/// far from the cap. Sharpened: the step must get there by its fixed-point sweeps ALONE — under
/// the default `Auto`, a step whose sweeps failed and Newton rescued also reads `converged`, so
/// without the fallback count this bar would not be about the sweeps at all. Measured: at most 5
/// sweeps (cap 50), worst residual 9.8e-14 / 8.7e-14 against 1e-13, no fallback.
#[test]
fn the_coupled_step_converges_by_its_sweeps_in_the_gate_regime_on_both_boundaries() {
    for boundary in BOTH {
        let mut v = vk(&spec(boundary));
        let u0 = strike(&v.p.lin, THICK);
        start(&mut v, &u0);
        let (mut most, mut worst) = (0usize, 0.0f64);
        for n in 0..300 {
            step(&mut v);
            most = most.max(v.n_iters);
            worst = worst.max(v.last_residual);
            assert!(v.converged, "{boundary:?} step {n}");
            assert!(v.last_residual <= v.p.couple_tol, "{boundary:?} step {n}");
            assert!(
                v.n_iters < v.p.couple_max_iter,
                "{boundary:?} step {n}: {}",
                v.n_iters
            );
            assert_eq!(v.n_fallbacks, 0, "{boundary:?} step {n}: Newton rescued it");
        }
        eprintln!("{boundary:?}: at most {most} sweeps, worst residual {worst:e}");
    }
}

// -- the two physics limits -------------------------------------------------------------------

/// At `w / e ~ 1e-6` the coupling (which vanishes like `w^3`) is negligible and the plate must
/// reproduce the linear one it reduces to: the supported (1,1) mode read at the centre, and the
/// free plate's strike read off-centre, for 2,000 steps each. Measured: the supported pickup is
/// BIT-identical to the linear plate's over all 2,000 steps (the coupling's contribution stays
/// under half an ulp of every node), the free one 3.4e-13 of its scale; the bar is the Python's.
#[test]
fn at_vanishing_amplitude_the_plate_is_the_linear_one_on_both_boundaries() {
    for boundary in BOTH {
        let mut v = vk(&spec(boundary));
        let mut lin = linear_twin(&v);
        let (u0, pk) = match boundary {
            Boundary::Supported => (
                mode11(&v.p.lin, SIDE, 1e-6 * THICK),
                pickup_index_at(0.5 * SIDE, 0.5 * v.p.lin.ly, &v.p.lin),
            ),
            Boundary::Free => (
                strike(&v.p.lin, 1e-6 * THICK),
                pickup_index_at(0.47 * SIDE, 0.53 * v.p.lin.ly, &v.p.lin),
            ),
        };
        start(&mut v, &u0);
        lin.set_state(&u0, &vec![0.0; u0.len()]);
        let (mut worst, mut scale) = (0.0f64, 0.0f64);
        for _ in 0..2000 {
            step(&mut v);
            lin.step(None);
            worst = nan_max(worst, (v.u[pk] - lin.u[pk]).abs());
            scale = nan_max(scale, lin.u[pk].abs());
        }
        eprintln!("{boundary:?}: pickup {pk}, worst {worst:e}, scale {scale:e}");
        assert!(worst / scale < 1e-4, "{boundary:?}: {:.3e}", worst / scale);
    }
}

/// NOT carried from Python: planted in §40.4, the human's call. Dropping the coupling force from
/// the second-order start (`w^{-1}` built from the bending acceleration alone) was seen by nothing
/// in the workspace but the viewer freeze, which is exact only on Windows. The supported half and
/// the free half are separate lines of the start, and each was planted.
///
/// §35's start-up bar, on the plate: released from rest the motion is even in time and the scheme
/// is time-reversible, so a start on the even solution makes `w^1 = w^{-1}` up to the Taylor
/// start's own `O(k^4)` error. Read at the headline's three thicknesses, where the coupling is
/// 0.76 of the first step's whole displacement (asserted above half, so the bar cannot go vacuous
/// by the coupling becoming too small to matter). Measured: 2.0e-2 on both boundaries correct;
/// planted, 0.855 on the planted boundary, caught by the symmetry assertion (the share guard reads
/// 0 then too). The 0.1 bound is this fixture's.
#[test]
fn released_from_rest_the_first_step_is_time_symmetric_on_both_boundaries() {
    let gap = |a: &[f64], b: &[f64]| {
        a.iter()
            .zip(b)
            .fold(0.0f64, |m, (x, y)| nan_max(m, (x - y).abs()))
    };
    for boundary in BOTH {
        let mut v = vk(&spec(boundary));
        let u0 = strike(&v.p.lin, 3.0 * THICK);
        start(&mut v, &u0);
        let back = v.u_prev.clone();
        let mut bending_only = vk(&VkSpec {
            nonlinear: false,
            ..spec(boundary)
        });
        start(&mut bending_only, &u0);
        step(&mut v);
        let size = gap(&v.u, &u0);
        let coupling_share = gap(&back, &bending_only.u_prev) / size;
        let asym = gap(&v.u, &back) / size;
        eprintln!("{boundary:?}: coupling share {coupling_share:.3e}, asymmetry {asym:.3e}");
        // The physics claim first, so a dropped coupling is caught BY it; the share is the guard
        // that keeps the claim from going vacuous, and a dropped coupling trips it too.
        assert!(asym < 0.1, "{boundary:?}: asymmetry {asym:.3e}");
        assert!(
            coupling_share > 0.5,
            "{boundary:?}: coupling {coupling_share:.3e}"
        );
    }
}

/// `fs / (2 mean(diff(zero crossings)))`, the Python's crude fundamental: crossings are where
/// `np.signbit` of the de-meaned record changes, and the mean is NumPy's pairwise one.
fn zero_cross_fundamental(sig: &[f64], fs: f64) -> f64 {
    let mean = reduce::sum(sig) / sig.len() as f64;
    let s: Vec<bool> = sig.iter().map(|v| (v - mean).is_sign_negative()).collect();
    let zc: Vec<usize> = (0..s.len() - 1).filter(|&i| s[i] != s[i + 1]).collect();
    let gaps: Vec<f64> = zc.windows(2).map(|w| (w[1] - w[0]) as f64).collect();
    fs / (2.0 * (reduce::sum(&gaps) / gaps.len() as f64))
}

/// A record of `steps` displacements at `pk`.
fn record(v: &mut VkPlate, pk: usize, steps: usize) -> Vec<f64> {
    (0..steps)
        .map(|_| {
            step(v);
            v.u[pk]
        })
        .collect()
}

/// The small-amplitude fundamental is the Navier law `f = (pi/2) kappa ((1/Lx)^2 + (1/Ly)^2)`
/// to grid dispersion, on a NON-square plate (0.4 x 0.32, snapped to 0.3167) so an axis swap
/// would show. The law is written here, not read from the model. The retired Python: 38.97158 Hz
/// against 39.04709, 0.19%; the bar is its 2%.
#[test]
fn the_small_amplitude_fundamental_is_the_navier_law() {
    let (lx, ly) = (0.4, 0.32);
    let mut v = vk(&VkSpec {
        lx,
        ly,
        n: 24,
        ..spec(Boundary::Supported)
    });
    let u0 = mode11(&v.p.lin, lx, 1e-4 * THICK);
    start(&mut v, &u0);
    let pk = pickup_index_at(0.47 * lx, 0.53 * v.p.lin.ly, &v.p.lin);
    let f = zero_cross_fundamental(&record(&mut v, pk, 4000), FS);
    let kappa = (YOUNG * THICK.powi(3) / (12.0 * (1.0 - NU * NU)) / (RHO * THICK)).sqrt();
    let snapped = v.p.lin.ly;
    let f_an = (PI / 2.0) * kappa * ((1.0 / lx).powi(2) + (1.0 / snapped).powi(2));
    eprintln!("pickup {pk}, Ly {snapped}, {f} Hz against {f_an} Hz");
    assert!((f - f_an).abs() / f_an < 0.02, "{f} Hz against {f_an} Hz");
}

/// Hardening on the supported plate: the (1,1) mode on a 15 cm plate struck at 0.01, 1.5, 3 and
/// 5 thicknesses rises monotonically, by over 20% at five. The retired Python: 213.36, 239.80,
/// 290.40, 372.96 Hz (a factor of 1.748). The sign is the claim, not a cents value: the law is
/// amplitude-dependent and has no closed form.
#[test]
fn the_supported_gong_glides_up_with_amplitude() {
    let side = 0.15;
    let freqs: Vec<f64> = [0.01, 1.5, 3.0, 5.0]
        .iter()
        .map(|&amp| {
            let mut v = vk(&VkSpec {
                lx: side,
                ly: side,
                n: 18,
                ..spec(Boundary::Supported)
            });
            let u0 = mode11(&v.p.lin, side, amp * THICK);
            start(&mut v, &u0);
            let pk = pickup_index_at(0.47 * side, 0.53 * v.p.lin.ly, &v.p.lin);
            zero_cross_fundamental(&record(&mut v, pk, 5000), FS)
        })
        .collect();
    assert!(
        freqs.windows(2).all(|w| w[0] < w[1]),
        "not rising: {freqs:?}"
    );
    eprintln!("supported glide {freqs:?}");
    assert!(
        freqs[3] > 1.2 * freqs[0],
        "glide {:.4}",
        freqs[3] / freqs[0]
    );
}

/// The FFT peak above `fmin`: Hann-windowed, de-meaned, first maximum on a tie (`np.argmax`).
fn fft_peak(sig: &[f64], fs: f64, fmin: f64) -> f64 {
    let mean = reduce::sum(sig) / sig.len() as f64;
    let w: Vec<f64> = sig
        .iter()
        .zip(hann(sig.len()))
        .map(|(v, h)| (v - mean) * h)
        .collect();
    let mag = rfft_mag(&w);
    let freqs = rfftfreq(sig.len(), 1.0 / fs);
    let (mut best, mut at) = (f64::NEG_INFINITY, 0usize);
    for (i, (&m, &f)) in mag.iter().zip(freqs.iter()).enumerate() {
        let m = if f < fmin { 0.0 } else { m };
        if m > best {
            best = m;
            at = i;
        }
    }
    freqs[at]
}

/// Hardening on the free plate, driven by its lowest ELASTIC mode (the fourth eigenvector of
/// `K x = mu W x`; the first three are the rigid-body `{1, x, y}`) and read by the FFT peak at
/// 0.01, 1, 2 and 3 thicknesses (beyond three the cascade smears "the" fundamental).
///
/// **The mode is not the retired Python's, on purpose.** It took the mode from ARPACK at
/// `sigma = -1e-3` beside the three-dimensional rigid nullspace, which is §28.2's failure: the
/// vector it got back has a residual `|K x - mu W x| / |K x|` of 0.45 — a mix of modes, not the
/// mode — and an eigenvalue 3.4% low (342,704 against LAPACK's 354,913). Here the mode is the
/// dense generalized solve, which LAPACK agrees with. The bar held either way; the figures moved:
/// the Python read 144, 156, 180, 252 Hz off its mixture, and 144, 156, 180, 204 Hz (a factor of
/// 1.42) when re-run with LAPACK's vector — the run this one is compared with.
#[test]
fn the_free_cymbal_glides_up_with_amplitude_from_its_lowest_elastic_mode() {
    let (side, fs) = (0.15, 96_000.0);
    let s = VkSpec {
        lx: side,
        ly: side,
        fs,
        n: 18,
        ..spec(Boundary::Free)
    };
    let base = VkParams::new(&s).expect("a valid plate");
    let k = &base.lin.stiffness;
    let n = k.nrows();
    let mut dense = vec![0.0; n * n];
    for i in 0..n {
        for p in k.indptr()[i]..k.indptr()[i + 1] {
            dense[i * n + k.indices()[p]] = k.data()[p];
        }
    }
    let (values, vectors) = generalized_eigen_diag(&dense, &base.lin.w, n).expect("SPD mass");
    // Three rigid-body zeros, then the first elastic mode, well separated from them.
    assert!(values[2].abs() < 1e-6 * values[3] && values[3] > 0.0);
    let col: Vec<f64> = (0..n).map(|i| vectors[i * n + 3]).collect();
    let peak = col.iter().fold(0.0f64, |m, x| m.max(x.abs()));
    let mode: Vec<f64> = col.iter().map(|x| x / peak).collect();
    let freqs: Vec<f64> = [0.01, 1.0, 2.0, 3.0]
        .iter()
        .map(|&amp| {
            let mut v = vk(&s);
            let a = amp * THICK;
            let u0: Vec<f64> = mode.iter().map(|m| a * m).collect();
            start(&mut v, &u0);
            let pk = pickup_index_at(0.30 * side, 0.62 * v.p.lin.ly, &v.p.lin);
            fft_peak(&record(&mut v, pk, 8000), fs, 100.0)
        })
        .collect();
    assert!(
        freqs.windows(2).all(|w| w[0] < w[1]),
        "not rising: {freqs:?}"
    );
    eprintln!("free glide {freqs:?}, mu {:?}", &values[..6]);
    assert!(
        freqs[3] > 1.15 * freqs[0],
        "glide {:.4}",
        freqs[3] / freqs[0]
    );
}

// -- convergence ------------------------------------------------------------------------------

/// Richardson self-convergence: one node's displacement after 80 steps at 96 kHz on N = 24, 48,
/// 96, from a smooth two-mode start of one thickness (a narrow strike sits pre-asymptotic: the
/// nonlinear source doubles the wavenumber content). Supported reads the centre, free the node
/// at (0.1, 0.1), which every grid has. The retired Python: ratios 4.398 and 5.656.
#[test]
fn the_plate_self_converges_at_second_order_on_both_boundaries() {
    for boundary in BOTH {
        let after = |n: i64| -> f64 {
            let mut v = vk(&VkSpec {
                fs: 96_000.0,
                n,
                ..spec(boundary)
            });
            let ly = v.p.lin.ly;
            let (u0, pk) = match boundary {
                Boundary::Supported => (
                    live_field(&v.p.lin, |x, y| {
                        let ic = (PI * x / SIDE).sin() * (PI * y / ly).sin()
                            + 0.4 * (2.0 * PI * x / SIDE).sin() * (PI * y / ly).sin();
                        THICK * ic
                    }),
                    pickup_index_at(0.5 * SIDE, 0.5 * ly, &v.p.lin),
                ),
                Boundary::Free => (
                    live_field(&v.p.lin, |x, y| {
                        let ic = (PI * x / SIDE).cos() * (PI * y / ly).cos()
                            + 0.4 * (2.0 * PI * x / SIDE).cos() * (PI * y / ly).cos();
                        THICK * ic
                    }),
                    pickup_index_at(0.1, 0.1, &v.p.lin),
                ),
            };
            start(&mut v, &u0);
            for _ in 0..80 {
                step(&mut v);
            }
            v.u[pk]
        };
        let (w24, w48, w96) = (after(24), after(48), after(96));
        let ratio = (w24 - w48).abs() / (w48 - w96).abs();
        eprintln!("{boundary:?}: {w24:e} {w48:e} {w96:e}, ratio {ratio}");
        assert!(ratio > 3.4, "{boundary:?}: ratio {ratio:.4}");
    }
}

// -- construction -----------------------------------------------------------------------------

/// The derived constants, against the formulas written here (a bar about a constant must not
/// import it): `rho_s = rho e`, `D = E e^3 / (12 (1 - nu^2))`, `kappa = sqrt(D / rho_s)`,
/// `Y = E e`. To a few ulp rather than the Python's `rel=1e-6`, which would pass a 1e-7 slip.
#[test]
fn the_derived_material_constants_are_the_textbook_ones() {
    let p = VkParams::new(&VkSpec {
        n: 12,
        ..spec(Boundary::Supported)
    })
    .expect("a valid plate");
    let close = |got: f64, want: f64| (got - want).abs() <= 4.0 * f64::EPSILON * want.abs();
    let d = YOUNG * THICK.powi(3) / (12.0 * (1.0 - NU * NU));
    assert!(close(p.rho_s, RHO * THICK), "rho_s {}", p.rho_s);
    assert!(close(p.d, d), "D {}", p.d);
    assert!(
        close(p.lin.kappa, (d / (RHO * THICK)).sqrt()),
        "kappa {}",
        p.lin.kappa
    );
    assert!(close(p.y_mem, YOUNG * THICK), "Y {}", p.y_mem);
    assert_eq!(
        p.lin.rho, p.rho_s,
        "the linear half carries the AREAL density"
    );
}

/// `Ly` snaps to a whole number of `h = Lx / N` cells: 0.83 on a 1 m, N = 10 plate becomes 0.8.
#[test]
fn the_side_snaps_to_square_cells() {
    let p = VkParams::new(&VkSpec {
        lx: 1.0,
        ly: 0.83,
        n: 10,
        ..spec(Boundary::Supported)
    })
    .expect("a valid plate");
    assert_eq!(p.lin.h, 0.1);
    assert_eq!(p.lin.ny, 8);
    assert_eq!(p.lin.ly, p.lin.ny as f64 * p.lin.h);
    assert!((p.lin.ly - 0.83).abs() <= p.lin.h);
}

/// Every refusal the Python constructor raised, as its variant and its message (the boundary's
/// message is the binding's: natively an unrecognised spelling arrives as `None`, and the text
/// that quotes it back leaves with the binding). The four of these that only a nonlinear plate
/// has are also in `plate.rs`; this is the whole table at the Python's fixture.
#[test]
fn the_constructor_refuses_what_the_python_refused() {
    let base = VkSpec {
        n: 12,
        ..spec(Boundary::Supported)
    };
    let cases: Vec<(VkSpec, VkParamError, &str)> = vec![
        (
            VkSpec {
                lx: 0.0,
                ..base.clone()
            },
            VkParamError::NonPositive,
            "Lx, Ly, fs must all be positive.",
        ),
        (
            VkSpec {
                ly: -1.0,
                ..base.clone()
            },
            VkParamError::NonPositive,
            "Lx, Ly, fs must all be positive.",
        ),
        (
            VkSpec {
                fs: 0.0,
                ..base.clone()
            },
            VkParamError::NonPositive,
            "Lx, Ly, fs must all be positive.",
        ),
        (
            VkSpec {
                young: 0.0,
                ..base.clone()
            },
            VkParamError::NonPositiveYoung,
            "E (Young's modulus) must be positive.",
        ),
        (
            VkSpec {
                thickness: -1e-3,
                ..base.clone()
            },
            VkParamError::NonPositiveThickness,
            "e (thickness) must be positive.",
        ),
        (
            VkSpec {
                rho: 0.0,
                ..base.clone()
            },
            VkParamError::NonPositiveDensity,
            "rho (density) must be positive.",
        ),
        (
            VkSpec {
                n: 1,
                ..base.clone()
            },
            VkParamError::TooFewSegments,
            "N must be >= 2 (need at least one interior node).",
        ),
        (
            VkSpec {
                sigma: -1.0,
                ..base.clone()
            },
            VkParamError::NegativeSigma,
            "sigma (loss) must be >= 0.",
        ),
        (
            VkSpec {
                theta: 0.0,
                ..base.clone()
            },
            VkParamError::BadTheta(0.0),
            "theta must be in (0, 1], got 0.0.",
        ),
        (
            VkSpec {
                theta: 1.5,
                ..base.clone()
            },
            VkParamError::BadTheta(1.5),
            "theta must be in (0, 1], got 1.5.",
        ),
        (
            VkSpec {
                nu: 0.5,
                ..base.clone()
            },
            VkParamError::BadNu(0.5),
            "nu (Poisson's ratio) must be in (-1, 1/2), got 0.5.",
        ),
        (
            VkSpec {
                nu: -1.0,
                ..base.clone()
            },
            VkParamError::BadNu(-1.0),
            "nu (Poisson's ratio) must be in (-1, 1/2), got -1.0.",
        ),
        (
            VkSpec {
                couple_tol: 0.0,
                ..base.clone()
            },
            VkParamError::NonPositiveTol,
            "couple_tol must be positive.",
        ),
        (
            VkSpec {
                couple_max_iter: 0,
                ..base.clone()
            },
            VkParamError::TooFewSweeps,
            "couple_max_iter must be >= 1.",
        ),
        (
            VkSpec {
                boundary: None,
                ..base.clone()
            },
            VkParamError::BadBoundary,
            "",
        ),
    ];
    for (s, want, text) in cases {
        let got = VkParams::new(&s).err();
        assert_eq!(got, Some(want.clone()), "{s:?}");
        assert_eq!(want.to_string(), text);
    }
}

/// A free plate makes every node an unknown (no Dirichlet rim) and carries the lumped mass.
#[test]
fn a_free_plate_has_every_node_live_and_a_lumped_mass() {
    let p = VkParams::new(&VkSpec {
        n: 8,
        ..spec(Boundary::Free)
    })
    .expect("a valid plate");
    let nodes = (p.lin.n + 1) * (p.lin.ny + 1);
    assert_eq!(p.lin.n_live, nodes);
    assert_eq!(p.lin.mask.flags().len(), nodes);
    assert!(p.lin.mask.flags().iter().all(|&alive| alive));
    assert!(p.lin.mass.is_some(), "the free plate's mass matrix");
    assert_eq!(p.lin.w.len(), p.lin.n_live);
    assert_eq!(p.lin.stiffness.nrows(), p.lin.n_live);
}

// -- the Airy solve's two grid-refinement bars (moved from ops2d.rs, §40.8) -----------------------
//
// Operator bars, not plate bars: they sit here only because this file runs optimised-only. Both
// factor the stress operator on grids up to 160 x 128, which takes ~60 s unoptimised against ~4 s
// optimised, and both read rates and error sizes, never an exact spelling (the human's call).

/// `np.meshgrid(np.linspace(0, lx, nx + 1), np.linspace(0, ly, ny + 1))`, flattened row-major.
fn mesh(nx: usize, ny: usize, lx: f64, ly: f64) -> (Vec<f64>, Vec<f64>) {
    let xs = linspace0(lx, nx + 1);
    let ys = linspace0(ly, ny + 1);
    let mut x = Vec::with_capacity((nx + 1) * (ny + 1));
    let mut y = Vec::with_capacity((nx + 1) * (ny + 1));
    for &yv in &ys {
        for &xv in &xs {
            x.push(xv);
            y.push(yv);
        }
    }
    (x, y)
}

fn max_abs(v: impl IntoIterator<Item = f64>) -> f64 {
    // NaN-propagating, as `np.max` is: a NaN must fail the bar it is compared against.
    v.into_iter().fold(0.0f64, |m, x| {
        if x.is_nan() || m.is_nan() {
            f64::NAN
        } else {
            m.max(x.abs())
        }
    })
}

/// The clamped manufactured stress function on the `1 x 0.8` rectangle and its biharmonic.
///
/// `F = (1 - cos(2 pi x / Lx)) (1 - cos(2 pi y / Ly))` has `F = F,n = 0` on every edge, so it IS a
/// clamped solution, and `lap F != 0` there, so it is NOT a Navier one. `g = 1 - cos(a x)` has
/// `g'''' = -a^4 cos(a x)` -- note the sign.
fn airy_manufactured(nx: usize, ny: usize) -> (Vec<f64>, Vec<f64>) {
    let (lx, ly) = (1.0, 0.8);
    let (a, b) = (2.0 * PI / lx, 2.0 * PI / ly);
    let (x, y) = mesh(nx, ny, lx, ly);
    let mut f = Vec::with_capacity(x.len());
    let mut lap4 = Vec::with_capacity(x.len());
    for (&xv, &yv) in x.iter().zip(y.iter()) {
        let (cx, cy) = ((a * xv).cos(), (b * yv).cos());
        let (g, q) = (1.0 - cx, 1.0 - cy);
        f.push(g * q);
        lap4.push(-(a.powi(4)) * cx * q + 2.0 * a * a * b * b * cx * cy - g * b.powi(4) * cy);
    }
    (f, lap4)
}

/// The Python's grid ladder: `h = 1 / Nx`, `Ny = round(0.8 / h)`.
fn ladder(nx: usize) -> (usize, usize, f64) {
    let h = 1.0 / nx as f64;
    (nx, (0.8 / h).round() as usize, h)
}

#[test]
fn the_airy_solve_recovers_a_clamped_manufactured_field_at_second_order() {
    // THE Part-2 gate. The retired Python read 0.02258 / 0.005631 / 0.001407 and rates 2.0033 /
    // 2.0008; the error is the discretization's, three orders above any solver's rounding.
    let mut errs = Vec::new();
    for nx in [40usize, 80, 160] {
        let (nx, ny, h) = ladder(nx);
        let (exact, lap4) = airy_manufactured(nx, ny);
        let f = AiryStressSolver::new(nx, ny, h)
            .expect("SPD")
            .solve(&lap4)
            .expect("solve");
        errs.push(max_abs(f.iter().zip(exact.iter()).map(|(a, b)| a - b)));
    }
    eprintln!("Airy errors {errs:?}");
    for w in errs.windows(2) {
        let rate = (w[0] / w[1]).ln() / 2.0f64.ln();
        assert!(
            rate > 1.9,
            "Airy solve converges at rate {rate:.4}, errors {errs:?}"
        );
    }
    assert!(errs[2] < 5e-3, "finest error {:.3e}", errs[2]);
}

#[test]
fn the_clamped_airy_operator_is_not_the_navier_biharmonic() {
    // The discriminator: both operators are SPD, so only a field that is clamped but not Navier can
    // tell them apart. Solving the same source through `B = L^2` saturates at O(1) (the retired
    // Python: 4.144 against 0.005631, a factor of 736) while the clamped solve converges.
    let (nx, ny, h) = ladder(80);
    let (exact, lap4) = airy_manufactured(nx, ny);
    let clamped = AiryStressSolver::new(nx, ny, h)
        .expect("SPD")
        .solve(&lap4)
        .expect("solve");
    let err_clamped = max_abs(clamped.iter().zip(exact.iter()).map(|(a, b)| a - b));
    let mask = rectangle_mask(nx, ny);
    let (b, _) = biharmonic_from_mask(&mask, h);
    let live: Vec<usize> = (0..mask.flags().len())
        .filter(|&p| mask.flags()[p])
        .collect();
    let rhs: Vec<f64> = live.iter().map(|&p| lap4[p]).collect();
    let f_ss = SparseLu::factor(&b)
        .expect("SPD")
        .solve(&rhs)
        .expect("solve");
    let err_ss = max_abs(live.iter().zip(f_ss.iter()).map(|(&p, v)| v - exact[p]));
    eprintln!("Navier {err_ss:e}, clamped {err_clamped:e}");
    assert!(
        err_ss > 50.0 * err_clamped,
        "Navier {err_ss:.4e} against clamped {err_clamped:.4e}"
    );
}
