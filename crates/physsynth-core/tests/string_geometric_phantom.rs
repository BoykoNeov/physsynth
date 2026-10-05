//! The geometrically exact string's **phantom partials** and **polarization discriminator** —
//! carried from `tests/test_geometric_phantom.py` (retirement plan §36).
//!
//! The two claims model #9 refuses. Its tension is a spatial scalar, so it has no longitudinal
//! field and nowhere to put a combination tone. Model #10's tension is a field, and its nonlinear
//! excess carries the term Kirchhoff–Carrier lacks:
//!
//! ```text
//! V_nl = (EA - T0) [ r^2 v_x / 2  +  r^4 / 8  + ... ]
//!                    ~~~~~~~~~~~    ~~~~~~~
//!                    PHANTOMS       model #9's quartic, recovered locally
//! ```
//!
//! `r^2 v_x / 2` is quadratic in the transverse fields and linear in the longitudinal one, so two
//! transverse partials at `f_i`, `f_j` drive `v` at `f_i ± f_j` — Conklin's (1999) phantom
//! partials. `string_geometric_harness.rs` measures where model #10 agrees with model #9; this file
//! measures the physics that makes them different models.
//!
//! **The read-out is the bridge force `EA v_x(0)`**, what radiates in a real piano. Below the
//! first longitudinal resonance `v` is quasi-static, so `EA v_x` carries `r^2` almost directly.
//! `v = 0` is not the longitudinal equilibrium, so every run also radiates a longitudinal
//! transient — the largest feature in the bridge spectrum — but the free longitudinal modes sit at
//! `n * 2236 Hz` while the phantoms live below 500 Hz. Every peak search is band-limited below the
//! first longitudinal mode for that reason: a physical cut, not a fudge.
//!
//! **Why `kappa = 8`, four times the default** ([`KAPPA_PHANTOM`]). A phantom is discriminating
//! only because it lands where no partial is; on a harmonic string every phantom coincides with a
//! partial exactly (`f2 - f1 = f1`, `2 f1 = f2`, `f1 + f2 = f3`). Inharmonicity opens the gaps
//! (`f3 - (f1 + f2) ~ 9 B f1`). This exaggerates the CONTRAST, not the effect: the mechanism is
//! `kappa`-independent. At the default `kappa = 2` the gap is 0.89 Hz, and hardening moves `f1 + f2`
//! up by 1.29 Hz — so the phantom would CROSS `f3`, and no run length fixes that. At `kappa = 8`
//! the gap is 11.4 Hz.
//!
//! **Why 0.1 s and not 0.05 s.** At 0.05 s the raw bins are 20 Hz wide, the `2 f1` phantom (the
//! weakest) sits in its neighbours' leakage skirts and is mislocated by 0.52 Hz, against 0.039 Hz
//! for the worst of the four at 0.1 s.
//!
//! Nothing here is NumPy's or SciPy's own number: the spectrum the Python read was this crate's
//! `physsynth-analysis::spectrum` through the binding, so nothing is frozen, as in §35.

mod geometric_fixture;

use geometric_fixture::*;
use physsynth_analysis::damping::spatial_eigenvalue_p2;
use physsynth_analysis::dispersion::stiff_dispersion_frequencies;
use physsynth_analysis::duffing::kc_mode_coefficients;
use physsynth_analysis::spectrum::{detect_peaks, magnitude_spectrum};
use physsynth_core::string_geometric::GeometricString;
use std::f64::consts::PI;
use std::sync::OnceLock;

const N: i64 = 32;
const LAM_LONG: f64 = 0.9;
/// Seconds of audio; see the header on why not 0.05.
const T_TOTAL: f64 = 0.10;
const AMP: f64 = 1.5e-3;
/// Stiffness for the phantom bars, 4x the default; see the header.
const KAPPA_PHANTOM: f64 = 8.0;

fn phantom_string(ea: f64) -> GeometricString {
    Geo {
        kappa: KAPPA_PHANTOM,
        ea,
        lam_long: Some(LAM_LONG),
        ..Geo::new(N)
    }
    .build()
}

/// The longitudinal end force `EA v_x(0) = EA v[1] / h` (N) — the piano's radiating channel.
fn bridge_force(s: &GeometricString) -> f64 {
    s.p.ea * s.v[1] / (s.p.l / s.p.n as f64)
}

fn steps(s: &GeometricString) -> usize {
    (T_TOTAL * s.p.fs) as usize
}

/// `np.argmin(np.abs(xs - x))`: the first index on a tie, as NumPy's.
fn nearest(xs: &[f64], x: f64) -> usize {
    let mut best = 0;
    for (i, v) in xs.iter().enumerate() {
        if (v - x).abs() < (xs[best] - x).abs() {
            best = i;
        }
    }
    best
}

struct TwoMode {
    fs: f64,
    bridge: Vec<f64>,
    f1: f64,
    f2: f64,
    f_long1: f64,
    ladder: Vec<f64>,
}

/// `_two_mode_run`: transverse modes 1 AND 2 — a single mode gives only `2 f1`, and the sum and
/// difference tones are what cannot be mistaken for a harmonic of anything. `f1` and `f2` are read
/// off modal projections, which isolate them from each other.
fn two_mode_run(amp: f64) -> TwoMode {
    let mut s = phantom_string(EA);
    let x = s.p.grid();
    let sin1: Vec<f64> = x.iter().map(|&xi| (PI * xi / L).sin()).collect();
    let sin2: Vec<f64> = x.iter().map(|&xi| (2.0 * PI * xi / L).sin()).collect();
    let u0: Vec<f64> = sin1.iter().zip(&sin2).map(|(a, b)| amp * (a + b)).collect();
    start_u(&mut s, &u0);

    let (d1, d2) = (dot(&sin1, &sin1), dot(&sin2, &sin2));
    let n = steps(&s);
    let (mut bridge, mut q1, mut q2) = (Vec::new(), Vec::new(), Vec::new());
    for _ in 0..n {
        step(&mut s);
        bridge.push(bridge_force(&s));
        q1.push(dot(&s.u, &sin1) / d1);
        q2.push(dot(&s.u, &sin2) / d2);
    }
    assert!(
        s.converged && s.n_not_converged == 0,
        "a non-converged run's spectrum means nothing"
    );
    let p = &s.p;
    TwoMode {
        fs: p.fs,
        f1: detect_peaks(&q1, p.fs, 1, 10.0, None)[0],
        f2: detect_peaks(&q2, p.fs, 1, 10.0, None)[0],
        f_long1: p.c_long / (2.0 * p.l),
        ladder: stiff_dispersion_frequencies(
            p.c,
            p.l,
            N,
            p.kappa_u,
            p.k,
            p.theta,
            &[1, 2, 3, 4, 5],
        ),
        bridge,
    }
}

/// The phantom run, shared by every bar that reads it.
fn phantom() -> &'static TwoMode {
    static RUN: OnceLock<TwoMode> = OnceLock::new();
    RUN.get_or_init(|| two_mode_run(AMP))
}

/// The same string at `amp -> 0`: it earns the ladder oracle at this `kappa`/`N`, and measures
/// how far hardening moved the partials at `AMP`.
fn linear() -> &'static TwoMode {
    static RUN: OnceLock<TwoMode> = OnceLock::new();
    RUN.get_or_init(|| two_mode_run(1e-6))
}

/// The four quadratic combination tones of the MEASURED partials — never predicted, since `f1`
/// and `f2` carry the theta-scheme's dispersion and the hardening shift, and the phantom rides on
/// whatever the partials actually are. Order: `f2-f1`, `2f1`, `f1+f2`, `2f2`.
fn combinations(r: &TwoMode) -> [f64; 4] {
    [r.f2 - r.f1, 2.0 * r.f1, r.f1 + r.f2, 2.0 * r.f2]
}

/// Blindly detected peaks below the first free longitudinal mode, and their magnitudes. Nothing
/// tells the detector where a phantom should be.
fn in_band_peaks(r: &TwoMode) -> (Vec<f64>, Vec<f64>) {
    let peaks: Vec<f64> = detect_peaks(&r.bridge, r.fs, 40, 10.0, None)
        .into_iter()
        .filter(|p| *p < 0.9 * r.f_long1)
        .collect();
    let spec = magnitude_spectrum(&r.bridge, r.fs, 2);
    let mags = peaks
        .iter()
        .map(|p| spec.mag[nearest(&spec.freqs, *p)])
        .collect();
    (peaks, mags)
}

fn nearest_peak(peaks: &[f64], f: f64) -> f64 {
    peaks[nearest(peaks, f)]
}

#[test]
fn longitudinal_peaks_are_quadratic_combinations_of_the_transverse_partials() {
    // Carried from the test of the same name: the `r^2 v_x / 2` term made audible. A transverse
    // spectrum {f1, f2} drives `v` at {f2-f1, 2f1, f1+f2, 2f2} and nothing else, and the ABSENCE of
    // f1 and f2 is the more discriminating half: a LINEAR coupling between the fields would put
    // the transverse partials into `v` directly.
    let r = phantom();
    let combos = combinations(r);
    let (peaks, mags) = in_band_peaks(r);
    assert!(peaks.len() >= 4);

    let mut order: Vec<usize> = (0..peaks.len()).collect();
    order.sort_by(|a, b| mags[*b].total_cmp(&mags[*a]));
    let mut strongest: Vec<f64> = order[..4].iter().map(|i| peaks[*i]).collect();
    strongest.sort_by(f64::total_cmp);
    let mut expected = combos.to_vec();
    expected.sort_by(f64::total_cmp);
    let err: Vec<f64> = strongest
        .iter()
        .zip(&expected)
        .map(|(a, b)| a - b)
        .collect();
    println!("strongest {strongest:.3?} vs combinations {expected:.3?}: {err:.4?}");
    assert!(
        max_abs(&err) < 0.15,
        "the 4 strongest in-band peaks are not the 4 combinations: {err:.4?}"
    );

    let is_combo: Vec<bool> = peaks
        .iter()
        .map(|p| combos.iter().any(|v| (p - v).abs() < 1.0))
        .collect();
    let weakest_combo = (0..peaks.len())
        .filter(|i| is_combo[*i])
        .map(|i| mags[i])
        .fold(f64::INFINITY, f64::min);
    let strongest_other = max_abs(
        &(0..peaks.len())
            .filter(|i| !is_combo[*i])
            .map(|i| mags[i])
            .collect::<Vec<_>>(),
    );
    println!("dominance {:.3}", weakest_combo / strongest_other);
    assert!(
        weakest_combo > 3.0 * strongest_other,
        "the combinations do not dominate: {weakest_combo:.3e} vs {strongest_other:.3e}"
    );

    // The other half: no longitudinal peak sits on an excited transverse partial.
    for (name, f) in [("f1", r.f1), ("f2", r.f2)] {
        let near = nearest_peak(&peaks, f);
        assert!(
            (near - f).abs() > 3.0,
            "a longitudinal peak sits on {name} = {f:.3}: {near:.3}. The coupling is quadratic."
        );
    }
}

#[test]
fn phantoms_are_displaced_from_the_partials_by_the_inharmonicity_defect() {
    // Carried from the test of the same name: the Conklin signature in its primary form, no oracle
    // and no confound. On a harmonic string `f2 - f1 = f1` and `2 f1 = f2`, so both distances are
    // exactly `|f2 - 2 f1|`, the inharmonicity defect, nonzero only because `kappa > 0`. Hardening
    // moves phantoms and partials TOGETHER, so it cannot smuggle in a false positive (it slightly
    // widens the defect, against the claim).
    let r = phantom();
    let defect = r.f2 - 2.0 * r.f1;
    assert!(
        defect > 3.0,
        "kappa > 0 must stretch the partials: {defect:.4} Hz"
    );
    let (peaks, _) = in_band_peaks(r);
    let combos = combinations(r);
    let peak_diff = nearest_peak(&peaks, combos[0]);
    let peak_2f1 = nearest_peak(&peaks, combos[1]);
    let (d1, d2) = ((peak_diff - r.f1).abs(), (peak_2f1 - r.f2).abs());
    println!("defect {defect:.4} Hz; displacements {d1:.4}, {d2:.4}");
    // The same physical number, approached from opposite sides.
    assert!((d1 - defect).abs() <= 0.3, "{d1} vs {defect}");
    assert!((d2 - defect).abs() <= 0.3, "{d2} vs {defect}");
    assert!(
        d1.min(d2) > 3.0,
        "the phantoms must land in the gaps, not on the partials"
    );
}

#[test]
fn the_phantom_lands_below_where_the_third_partial_would_be() {
    // Carried from the test of the same name: Conklin's headline, the secondary form. `f1 + f2`
    // lands ~9 B f1 BELOW `f3`, which is not excited, so `f3` needs an oracle -- the DISCRETE
    // ladder (the scheme's own eigenvalue and theta dispersion), earned by landing on it at
    // `amp -> 0`. The hardening confound is measured, not waved at.
    let (r, lin) = (phantom(), linear());
    let f3 = r.ladder[2];
    for (i, got) in [lin.f1, lin.f2].into_iter().enumerate() {
        let want = lin.ladder[i];
        println!("ladder f{}: {:.3e} relative", i + 1, got / want - 1.0);
        assert!(
            (got - want).abs() <= 1e-3 * want,
            "the ladder oracle is wrong for f{} at kappa = {KAPPA_PHANTOM}: {got} vs {want}",
            i + 1
        );
    }
    let hardening = (r.f1 - lin.f1) + (r.f2 - lin.f2);
    let gap = f3 - (r.f1 + r.f2);
    println!("gap {gap:.4} Hz, hardening {hardening:.4} Hz");
    assert!(gap > 0.0, "the phantom must land BELOW f3: {gap:.3}");
    assert!(
        hardening < 0.4 * gap,
        "hardening {hardening:.3} Hz is no longer small against the gap {gap:.3} Hz"
    );
    let (peaks, _) = in_band_peaks(r);
    let peak_sum = nearest_peak(&peaks, r.f1 + r.f2);
    assert!(
        (peak_sum - (r.f1 + r.f2)).abs() <= 0.15,
        "{peak_sum} vs {}",
        r.f1 + r.f2
    );
    assert!(
        (peak_sum - f3).abs() > 5.0,
        "the phantom at {peak_sum:.3} is only {:.3} Hz from f3",
        (peak_sum - f3).abs()
    );
}

#[test]
fn a_linear_string_has_no_longitudinal_motion_to_put_a_phantom_in() {
    // Carried from the test of the same name: the harness control. At `EA = T0` the whole
    // nonlinear excess is gone, the fields decouple, and `v` started at rest never leaves it --
    // identically, not "small". Same start, same read-out; the only change is the coefficient the
    // phantom is supposed to come from. It is also what licenses a BLIND detector.
    let mut s = phantom_string(T);
    assert_eq!(s.p.a, 0.0);
    let x = s.p.grid();
    let u0: Vec<f64> = x
        .iter()
        .map(|&xi| AMP * ((PI * xi / L).sin() + (2.0 * PI * xi / L).sin()))
        .collect();
    start_u(&mut s, &u0);
    let mut bridge = Vec::new();
    for _ in 0..steps(&s) {
        step(&mut s);
        bridge.push(bridge_force(&s));
    }
    assert!(max_abs(&s.u) > 0.0, "the transverse field must be moving");
    assert!(
        bridge.iter().all(|b| *b == 0.0),
        "EA = T0 must leave v identically zero; got {:.3e}",
        max_abs(&bridge)
    );
}

// -- the polarization discriminator --------------------------------------------------------------

struct Polar {
    pump: f64,
    dc: f64,
    w_max: f64,
}

/// The angular frequency of the Kirchhoff–Carrier CIRCULAR relative equilibrium, used to BUILD a
/// start, never asserted: a circular mode holds `r^2` static, so the shift is the full `eps A^2`.
fn circular_omega(s: &GeometricString, amp: f64) -> f64 {
    let p = &s.p;
    let p2 = spatial_eigenvalue_p2(p.n as i64, p.l / p.n as f64, 1);
    let (w0sq, eps) = kc_mode_coefficients(p.c, p.kappa_u, p.a, p.rho, p2, p.l).unwrap();
    (w0sq + eps * amp.powi(2)).sqrt()
}

/// `_polarization_run`: one single-mode run, planar when `omega` is `None`, else circular at that
/// rate (`w_dot = Omega A phi` at t = 0). Reports the bridge force's AC at `2 f1` (the pump) and
/// its DC (the static stretch) separately: polarization moves the two in opposite directions.
fn polarization_run(omega: Option<f64>) -> Polar {
    let mut s = phantom_string(EA);
    let shape = mode_ic(&s, 1, AMP);
    let p = &s.p;
    let f1 = stiff_dispersion_frequencies(p.c, p.l, N, p.kappa_u, p.k, p.theta, &[1])[0];
    let z = zeros(&s);
    let w_dot = match omega {
        Some(om) => shape.iter().map(|x| om * x).collect(),
        None => z.clone(),
    };
    s.set_state(&shape, &z, &z, &[z.clone(), w_dot, z.clone()]);

    let mut bridge = Vec::new();
    let mut w_max = 0.0f64;
    for _ in 0..steps(&s) {
        step(&mut s);
        bridge.push(bridge_force(&s));
        // Tracked, never sampled at the last step: the circle is at an arbitrary phase there.
        w_max = nan_max(w_max, max_abs(&s.w));
    }
    assert_eq!(s.n_not_converged, 0);
    let spec = magnitude_spectrum(&bridge, s.p.fs, 2);
    let i2 = nearest(&spec.freqs, 2.0 * f1);
    Polar {
        pump: max_abs(&spec.mag[i2.saturating_sub(2).max(1)..i2 + 3]),
        dc: bridge.iter().sum::<f64>() / bridge.len() as f64,
        w_max,
    }
}

/// Planar, circular at the LINEAR rate, and circular at the KC circular rate.
fn polarization() -> &'static [Polar; 3] {
    static RUNS: OnceLock<[Polar; 3]> = OnceLock::new();
    RUNS.get_or_init(|| {
        let s = phantom_string(EA);
        let p = &s.p;
        let f1 = stiff_dispersion_frequencies(p.c, p.l, N, p.kappa_u, p.k, p.theta, &[1])[0];
        let runs = [
            polarization_run(None),
            polarization_run(Some(2.0 * PI * f1)),
            polarization_run(Some(circular_omega(&s, AMP))),
        ];
        for (name, r) in ["planar", "naive", "tuned"].iter().zip(&runs) {
            println!(
                "{name}: pump {:.4e}, dc {:.6e}, w_max {:.6e}",
                r.pump, r.dc, r.w_max
            );
        }
        runs
    })
}

#[test]
fn a_circular_mode_does_not_pump_the_longitudinal_field() {
    // Carried from the test of the same name: same string, mode and amplitude, the longitudinal
    // pump orders of magnitude apart, decided by polarization alone. A planar `r^2` oscillates at
    // 2 Omega and pumps `v`; a circular one is time-independent. Not a quiet string: the circular
    // run carries twice the energy and its static stretch is exactly twice the planar average, so
    // the nonlinearity is on, harder, and silent. Energy cannot see any of this.
    let [planar, _, tuned] = polarization();
    let ratio = planar.pump / tuned.pump;
    let dc = tuned.dc / planar.dc;
    println!("pump ratio {ratio:.4e}, dc ratio {dc:.5}");
    assert!(
        ratio > 1e4,
        "planar and circular pump comparably: {ratio:.1}x"
    );
    assert!(
        (dc - 2.0).abs() <= 0.05 * 2.0,
        "a circular static stretch must be 2x a planar one: {dc:.4}x"
    );
    assert!(
        (tuned.w_max - AMP).abs() <= 0.05 * AMP,
        "the circular run must trace a full circle: {:.3e}",
        tuned.w_max
    );
    assert_eq!(
        planar.w_max, 0.0,
        "the planar run must stay planar to the bit"
    );
}

#[test]
fn the_circular_residual_is_ellipticity_not_a_defect_of_the_scheme() {
    // Carried from the test of the same name. Driving the same SINE helix at the KC circular rate
    // rather than the linear one collapses the residual pump ~300x: at the linear rate the circle
    // is an ellipse, `r^2` is no longer static, and it pumps `v` like a planar mode in miniature.
    // Both are sines, so the tuned run's residual BOUNDS the shape error's share.
    let [planar, naive, tuned] = polarization();
    assert!(
        naive.pump > 50.0 * tuned.pump,
        "tuning Omega should collapse the residual: naive {:.3e} vs tuned {:.3e}",
        naive.pump,
        tuned.pump
    );
    // Ellipticity is a FREQUENCY error, not an amplitude one: same DC, only the AC moves.
    assert!(
        (naive.dc - tuned.dc).abs() <= 1e-2 * tuned.dc.abs(),
        "{} vs {}",
        naive.dc,
        tuned.dc
    );
    // Even the naive circle is decisive; the sharpening is a bonus.
    assert!(planar.pump / naive.pump > 50.0);
}
