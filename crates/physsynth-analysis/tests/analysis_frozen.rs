//! The analysis oracles against numbers a **different implementation** produced.
//!
//! `tests/reference/analysis_frozen.json` is the record `tests/analysis_frozen_values.py` held: what
//! the Python oracles in `physsynth/analysis/` said at 74 fixtures, recorded to the last digit
//! before they were deleted (`docs/dev/rust-migration-plan.md` §44; the twelve `horizon.` rows were
//! recorded from `tests/helpers.py`'s bodies in the batch that promoted them,
//! `docs/dev/resolution-horizon-plan.md` §6). It moved here at retirement plan §47 so that it can
//! outlive Python, and this file is the Python test that read it, carried whole except for one
//! guard (below).
//!
//! **What it can and cannot see.** After the deletion there is one implementation of these
//! oracles, so nothing can re-derive what they should say. This keeps the next best thing: a
//! transcription error, a wrong branch, a changed convention and a regression all move a frozen
//! number. An error the Python made too moves nothing — that is what the other files in this
//! directory are for, where the oracles are checked against their mathematical definitions.
//!
//! **The inputs are frozen too, not only the answers.** Several fixtures were built by NumPy — two
//! signals and two Hessian strains draw on its seeded normal generator, the tones go through its
//! own `sin` and `exp`, the grids through `linspace` and `meshgrid` — and nothing native can
//! rebuild those doubles. So every argument is in the record as the doubles the cases module built
//! on the converting machine, with every default of the deleted Python wrappers written out (the
//! circular membrane's `m_max`, the root scan's `scan`, the spectrum's `zero_pad_factor`, the
//! rotating wave's continuation steps, tolerance and iteration cap, ...). Those doubles may differ in
//! a last bit from the ones the generating machine built, which is one reason the bar is a
//! tolerance.
//!
//! **The bar is one number, the plan's Group A target**, 1e-13, on the gap normalised by the
//! answer's own scale. At generation 51 of the first 62 rows were exactly zero and the worst was
//! 3.5e-15 (`duffing_frequency_shift`, a difference of two nearly equal frequencies), about 28x
//! headroom. It is deliberately not an equality: ledger #28 is the standing warning that a
//! cross-machine bit-identity claim is a claim about the runner. **Structure and integers are
//! compared exactly** — a root search that finds a different number of roots, a spectrum of a
//! different length, a `converged` that flips, a shifted mode count: none of those is a small error
//! and no tolerance describes one.
//!
//! **The one guard not carried** is the Python file's derived coverage check — every public name
//! in `physsynth.analysis`'s `__all__` must have a frozen case. Its population was the Python
//! package, which is going, and its purpose had already lapsed: a freeze needs a second
//! implementation to record, none is left (retirement plan §22 deleted the generators), and its own
//! failure message told a new oracle to get a native bar instead. A native "every `pub fn` has a
//! row" would be the same dead end at a larger number.

use std::cell::RefCell;
use std::collections::BTreeSet;
use std::sync::OnceLock;

use physsynth_analysis::{damping, dispersion, duffing, horizon, modal, rotating_wave, spectrum};
use serde_json::{Map, Value};

/// The plan's Group A agreement target, on the amplitude-normalised gap. One bar for every case.
const BAR: f64 = 1e-13;

/// Rows in the record. Counted from the data at the move (§47): 33 modal, 12 horizon, 8 duffing,
/// 7 damping, 6 spectrum, 5 rotating wave, 3 dispersion.
const ROWS: usize = 74;

/// What the horizon freeze wrote in the gap column where an answer has no float in it at all. Not
/// "the comparison failed" — "the comparison was entirely the exact one", the stronger arm.
const NO_FLOATS: &str = "no floats in this answer -- the comparison is the exact int/structure one";

fn record() -> &'static Value {
    static RECORD: OnceLock<Value> = OnceLock::new();
    RECORD.get_or_init(|| {
        serde_json::from_str(include_str!("reference/analysis_frozen.json"))
            .expect("the frozen record parses")
    })
}

fn cases() -> &'static Map<String, Value> {
    record()["cases"].as_object().expect("`cases` is an object")
}

// -- reading a row's arguments -------------------------------------------------------------------

/// One row's arguments, by the deleted Python wrapper's parameter names. Every read is recorded,
/// and [`Args::all_read`] checks that the dispatch consumed every argument the record holds — so a
/// recorded input (a default above all) cannot be silently replaced by a literal in the call.
struct Args<'a> {
    map: &'a Map<String, Value>,
    read: RefCell<BTreeSet<&'a str>>,
}

impl<'a> Args<'a> {
    fn new(map: &'a Map<String, Value>) -> Self {
        Args {
            map,
            read: RefCell::new(BTreeSet::new()),
        }
    }

    fn get(&self, name: &str) -> &'a Value {
        let (k, v) = self
            .map
            .get_key_value(name)
            .unwrap_or_else(|| panic!("the record has no argument `{name}`"));
        self.read.borrow_mut().insert(k.as_str());
        v
    }

    fn f(&self, name: &str) -> f64 {
        self.get(name)
            .as_f64()
            .unwrap_or_else(|| panic!("`{name}` is not a number"))
    }

    fn i(&self, name: &str) -> i64 {
        self.get(name)
            .as_i64()
            .unwrap_or_else(|| panic!("`{name}` is not an integer"))
    }

    fn u(&self, name: &str) -> usize {
        usize::try_from(self.i(name)).unwrap_or_else(|_| panic!("`{name}` is negative"))
    }

    fn b(&self, name: &str) -> bool {
        self.get(name)
            .as_bool()
            .unwrap_or_else(|| panic!("`{name}` is not a bool"))
    }

    fn s(&self, name: &str) -> &'a str {
        self.get(name)
            .as_str()
            .unwrap_or_else(|| panic!("`{name}` is not a string"))
    }

    fn opt_f(&self, name: &str) -> Option<f64> {
        let v = self.get(name);
        if v.is_null() {
            None
        } else {
            Some(
                v.as_f64()
                    .unwrap_or_else(|| panic!("`{name}` is neither null nor a number")),
            )
        }
    }

    /// An array argument as doubles, and its NumPy shape. An integer array (`dirichlet_axis_eigenvalue`'s
    /// mode numbers) converts exactly as the wrapper's `np.asarray(m, dtype=float)` did.
    fn arr(&self, name: &str) -> (Vec<f64>, Vec<usize>) {
        let v = self.get(name);
        let shape = v["shape"]
            .as_array()
            .unwrap_or_else(|| panic!("`{name}` has no shape"))
            .iter()
            .map(|d| d.as_u64().expect("a dimension") as usize)
            .collect();
        let vals = if let Some(fs) = v.get("floats") {
            fs.as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().expect("a float"))
                .collect()
        } else {
            v["ints"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_i64().expect("an int") as f64)
                .collect()
        };
        (vals, shape)
    }

    fn ints(&self, name: &str) -> Vec<i64> {
        self.get(name)["ints"]
            .as_array()
            .unwrap_or_else(|| panic!("`{name}` is not an integer array"))
            .iter()
            .map(|x| x.as_i64().expect("an int"))
            .collect()
    }

    fn pairs(&self, name: &str) -> Vec<(i64, i64)> {
        self.get(name)["pairs"]
            .as_array()
            .unwrap_or_else(|| panic!("`{name}` is not a mode list"))
            .iter()
            .map(|p| (p[0].as_i64().unwrap(), p[1].as_i64().unwrap()))
            .collect()
    }

    fn nested(&self, name: &str) -> Args<'a> {
        Args::new(
            self.get(name)
                .as_object()
                .unwrap_or_else(|| panic!("`{name}` is not an object")),
        )
    }

    fn all_read(&self) -> Result<(), Vec<&'a str>> {
        let read = self.read.borrow();
        let unread: Vec<&str> = self
            .map
            .keys()
            .map(String::as_str)
            .filter(|k| !read.contains(k))
            .collect();
        if unread.is_empty() {
            Ok(())
        } else {
            Err(unread)
        }
    }
}

// -- flattening an answer the way the Python freeze did -------------------------------------------

/// An answer as `(structure, floats, ints)` — a transcription of `analysis_frozen_cases.flatten`.
/// Each method appends its values and returns its piece of the structure string, so a tuple built
/// left to right appends in the order Python's walk did: `f`, `i` and `b` for a scalar (a bool goes
/// into `ints` as 0 or 1), `af[..]` / `ai[..]` for an array with NumPy's `list(shape)` spelling.
#[derive(Default)]
struct Flat {
    floats: Vec<f64>,
    ints: Vec<i64>,
}

impl Flat {
    fn f(&mut self, x: f64) -> String {
        self.floats.push(x);
        "f".into()
    }

    fn i(&mut self, x: i64) -> String {
        self.ints.push(x);
        "i".into()
    }

    fn b(&mut self, x: bool) -> String {
        self.ints.push(i64::from(x));
        "b".into()
    }

    fn af(&mut self, xs: &[f64], shape: &[usize]) -> String {
        assert_eq!(
            xs.len(),
            shape.iter().product::<usize>(),
            "an array does not fill its shape"
        );
        self.floats.extend_from_slice(xs);
        format!("af{shape:?}")
    }

    fn af1(&mut self, xs: &[f64]) -> String {
        self.af(xs, &[xs.len()])
    }

    fn ai1(&mut self, xs: &[i64]) -> String {
        self.ints.extend_from_slice(xs);
        format!("ai[{}]", xs.len())
    }
}

fn tuple(parts: Vec<String>) -> String {
    format!("({})", parts.join(","))
}

fn ok<T>(r: Result<T, String>, function: &str) -> T {
    r.unwrap_or_else(|e| panic!("{function} refused its frozen case: {e}"))
}

/// Run one case through the native implementation. `function` is `<module>.<python name>`; the
/// argument names are the Python wrapper's, defaults included.
fn answer(function: &str, a: &Args, fl: &mut Flat) -> String {
    // An element-wise oracle: the wrapper flattened, mapped, and reshaped to the input's shape.
    fn each(fl: &mut Flat, (xs, shape): (Vec<f64>, Vec<usize>), f: impl Fn(f64) -> f64) -> String {
        let out: Vec<f64> = xs.into_iter().map(f).collect();
        fl.af(&out, &shape)
    }

    match function {
        // ---- modal ----------------------------------------------------------------------------
        "modal.harmonic_frequencies" => fl.af1(&modal::harmonic_frequencies(
            a.f("c"),
            a.f("L"),
            a.u("n_partials"),
        )),
        "modal.mode_shape" => {
            let (x, shape) = a.arr("x");
            fl.af(&modal::mode_shape(&x, a.f("L"), a.i("m")), &shape)
        }
        "modal.discrete_mode_frequency" => fl.f(modal::discrete_mode_frequency(
            a.f("c"),
            a.f("L"),
            a.i("N"),
            a.f("lam"),
            a.i("m"),
        )),
        "modal.inharmonicity_B" => fl.f(modal::inharmonicity_b(a.f("c"), a.f("L"), a.f("kappa"))),
        "modal.stiff_harmonic_frequencies" => fl.af1(&modal::stiff_harmonic_frequencies(
            a.f("c"),
            a.f("L"),
            a.f("kappa"),
            a.u("n_partials"),
        )),
        "modal.discrete_stiff_mode_frequency" => fl.f(modal::discrete_stiff_mode_frequency(
            a.f("c"),
            a.f("L"),
            a.i("N"),
            a.f("kappa"),
            a.f("k"),
            a.i("m"),
            a.f("theta"),
        )),
        "modal.cents" => {
            let (f, shape) = a.arr("f");
            let (f_ref, ref_shape) = a.arr("f_ref");
            assert_eq!(
                shape, ref_shape,
                "the frozen `cents` case needs no broadcast"
            );
            let out: Vec<f64> = f
                .iter()
                .zip(&f_ref)
                .map(|(&x, &y)| modal::cents(x, y))
                .collect();
            fl.af(&out, &shape)
        }
        "modal.rectangular_membrane_freqs" => fl.af1(&modal::rectangular_membrane_freqs(
            a.f("c"),
            a.f("Lx"),
            a.f("Ly"),
            &a.pairs("modes"),
        )),
        "modal.rectangular_mode_field" => {
            let (x, shape) = a.arr("X");
            let (y, y_shape) = a.arr("Y");
            assert_eq!(shape, y_shape);
            let out =
                modal::rectangular_mode_field(&x, &y, a.f("Lx"), a.f("Ly"), a.i("m"), a.i("n"));
            fl.af(&out, &shape)
        }
        "modal.rectangular_discrete_eigenvalues" => {
            fl.af1(&modal::rectangular_discrete_eigenvalues(
                a.f("h"),
                a.i("Nx"),
                a.i("Ny"),
                &a.pairs("modes"),
            ))
        }
        "modal.circular_membrane_freqs" => {
            let m_max = u32::try_from(a.i("m_max")).unwrap();
            let modes = modal::circular_membrane_freqs(
                a.f("c"),
                a.f("a"),
                a.u("n_modes"),
                m_max,
                a.u("n_max"),
            );
            let parts = modes
                .iter()
                .map(|e| {
                    tuple(vec![
                        fl.i(i64::from(e.m)),
                        fl.i(e.n as i64),
                        fl.f(e.freq),
                        fl.i(i64::from(e.degeneracy)),
                    ])
                })
                .collect();
            tuple(parts)
        }
        "modal.discrete_membrane_eigenfrequency" => {
            let (c, k) = (a.f("c"), a.f("k"));
            each(fl, a.arr("Lambda"), |x| {
                modal::discrete_membrane_eigenfrequency(x, c, k)
            })
        }
        "modal.rectangular_plate_freqs" => fl.af1(&modal::rectangular_plate_freqs(
            a.f("kappa"),
            a.f("Lx"),
            a.f("Ly"),
            &a.pairs("modes"),
        )),
        "modal.discrete_plate_eigenfrequency" => {
            let (kappa, k, theta) = (a.f("kappa"), a.f("k"), a.f("theta"));
            each(fl, a.arr("Lambda_lap"), |x| {
                modal::discrete_plate_eigenfrequency(x, kappa, k, theta)
            })
        }
        "modal.orthotropic_plate_freqs" => fl.af1(&ok(
            modal::orthotropic_plate_freqs(
                a.f("kappa"),
                a.f("Lx"),
                a.f("Ly"),
                &a.pairs("modes"),
                a.f("grain_x"),
                a.f("grain_cross"),
                a.f("grain_y"),
            ),
            function,
        )),
        "modal.discrete_orthotropic_plate_eigenfrequency" => {
            let (lx, shape) = a.arr("lam_x");
            let (ly, y_shape) = a.arr("lam_y");
            assert_eq!(shape, y_shape, "the frozen case needs no broadcast");
            let (kappa, k, theta) = (a.f("kappa"), a.f("k"), a.f("theta"));
            let (gx, gc, gy) = (a.f("grain_x"), a.f("grain_cross"), a.f("grain_y"));
            let out: Vec<f64> = lx
                .iter()
                .zip(&ly)
                .map(|(&x, &y)| {
                    ok(
                        modal::discrete_orthotropic_plate_eigenfrequency(
                            x, y, kappa, k, theta, gx, gc, gy,
                        ),
                        function,
                    )
                })
                .collect();
            fl.af(&out, &shape)
        }
        "modal.dirichlet_axis_eigenvalue" => {
            let (l, h) = (a.f("L"), a.f("h"));
            each(fl, a.arr("m"), |m| {
                modal::dirichlet_axis_eigenvalue(m, l, h)
            })
        }
        "modal.free_free_beam_betaL" => {
            fl.af1(&ok(modal::free_free_beam_beta_l(a.u("n_modes")), function))
        }
        "modal.free_free_beam_freqs" => fl.af1(&ok(
            modal::free_free_beam_freqs(a.f("kappa"), a.f("L"), a.u("n_modes")),
            function,
        )),
        "modal.free_plate_ffff_square_lambdas" => fl.af1(&modal::free_plate_ffff_square_lambdas()),
        "modal.free_plate_freq_from_lambda" => {
            let (kappa, side) = (a.f("kappa"), a.f("a"));
            each(fl, a.arr("lam"), |x| {
                modal::free_plate_freq_from_lambda(x, kappa, side)
            })
        }
        "modal.free_plate_twist_bound" => fl.f(ok(
            modal::free_plate_twist_bound(a.f("kappa"), a.f("a"), a.f("b"), a.f("grain_torsion")),
            function,
        )),
        "modal.free_circular_plate_lambda_roots" => {
            let n = i32::try_from(a.i("n")).unwrap();
            fl.af1(&ok(
                modal::free_circular_plate_lambda_roots(a.f("nu"), n, a.f("lam_max"), a.u("scan")),
                function,
            ))
        }
        "modal.free_circular_plate_lambdas" => {
            let n_max = u32::try_from(a.i("n_max")).unwrap();
            let (lam, nodal) = ok(
                modal::free_circular_plate_lambdas(a.f("nu"), a.u("n_modes"), n_max),
                function,
            );
            tuple(vec![fl.af1(&lam), fl.ai1(&nodal)])
        }
        "modal.free_circular_plate_saddle_bound" => fl.f(ok(
            modal::free_circular_plate_saddle_bound(a.f("nu")),
            function,
        )),
        "modal.free_plate_coupling_form" => fl.f(ok(
            modal::free_plate_coupling_form(a.f("grain_coupling"), a.f("h"), a.i("Nx"), a.i("Ny")),
            function,
        )),
        "modal.bore_resonance_frequencies" => fl.af1(&ok(
            modal::bore_resonance_frequencies(
                a.f("c0"),
                a.f("L"),
                a.u("n_partials"),
                a.s("boundary"),
            ),
            function,
        )),
        "modal.discrete_bore_eigenfrequency" => {
            let k = a.f("k");
            each(fl, a.arr("omega2"), |x| {
                modal::discrete_bore_eigenfrequency(x, k)
            })
        }
        "modal.discrete_beam_eigenfrequency" => {
            let (kappa, k, theta) = (a.f("kappa"), a.f("k"), a.f("theta"));
            each(fl, a.arr("mu"), |x| {
                modal::discrete_beam_eigenfrequency(x, kappa, k, theta)
            })
        }

        // ---- damping --------------------------------------------------------------------------
        "damping.spatial_eigenvalue_p2" => {
            fl.f(damping::spatial_eigenvalue_p2(a.i("N"), a.f("h"), a.i("m")))
        }
        "damping.modal_loss_rate_continuum" => fl.f(damping::modal_loss_rate_continuum(
            a.f("c"),
            a.f("L"),
            a.f("kappa"),
            a.f("sigma0"),
            a.f("sigma1"),
            a.i("m"),
        )),
        "damping.discrete_damped_mode_decay"
        | "damping.discrete_damped_mode_rate"
        | "damping.discrete_damped_mode_is_underdamped" => {
            let args = (
                a.f("c"),
                a.f("L"),
                a.i("N"),
                a.f("kappa"),
                a.f("k"),
                a.f("theta"),
                a.f("sigma0"),
                a.f("sigma1"),
                a.i("m"),
            );
            let (c, l, n, kappa, k, theta, s0, s1, m) = args;
            match function {
                "damping.discrete_damped_mode_decay" => fl.f(damping::discrete_damped_mode_decay(
                    c, l, n, kappa, k, theta, s0, s1, m,
                )),
                "damping.discrete_damped_mode_rate" => fl.f(damping::discrete_damped_mode_rate(
                    c, l, n, kappa, k, theta, s0, s1, m,
                )),
                _ => fl.b(damping::discrete_damped_mode_is_underdamped(
                    c, l, n, kappa, k, theta, s0, s1, m,
                )),
            }
        }
        "damping.loss_coefficients_from_T60" => {
            let (s0, s1) = ok(
                damping::loss_coefficients_from_t60(
                    a.f("c"),
                    a.f("L"),
                    a.f("kappa"),
                    a.f("f1"),
                    a.f("T60_1"),
                    a.f("f2"),
                    a.f("T60_2"),
                ),
                function,
            );
            tuple(vec![fl.f(s0), fl.f(s1)])
        }

        // ---- dispersion -----------------------------------------------------------------------
        "dispersion.dispersion_frequencies" => fl.af1(&dispersion::dispersion_frequencies(
            a.f("c"),
            a.f("L"),
            a.i("N"),
            a.f("lam"),
            &a.ints("modes"),
        )),
        "dispersion.stiff_dispersion_frequencies" => {
            fl.af1(&dispersion::stiff_dispersion_frequencies(
                a.f("c"),
                a.f("L"),
                a.i("N"),
                a.f("kappa"),
                a.f("k"),
                a.f("theta"),
                &a.ints("modes"),
            ))
        }
        "dispersion.phase_velocity" => {
            let (f, _) = a.arr("f");
            fl.af1(&dispersion::phase_velocity(&f, a.f("L"), &a.ints("modes")))
        }

        // ---- duffing --------------------------------------------------------------------------
        "duffing.kc_mode_coefficients" => {
            let (omega0_sq, eps) = ok(
                duffing::kc_mode_coefficients(
                    a.f("c"),
                    a.f("kappa"),
                    a.f("EA"),
                    a.f("rho"),
                    a.f("p2"),
                    a.f("L"),
                ),
                function,
            );
            tuple(vec![fl.f(omega0_sq), fl.f(eps)])
        }
        "duffing.kc_mode_stretch" => fl.f(duffing::kc_mode_stretch(
            a.f("amplitude"),
            a.f("p2"),
            a.f("L"),
        )),
        "duffing.duffing_elliptic_parameter"
        | "duffing.duffing_frequency"
        | "duffing.duffing_frequency_shift"
        | "duffing.duffing_frequency_expansion" => {
            let (amp, w2, eps) = (a.f("amplitude"), a.f("omega0_sq"), a.f("eps"));
            let oracle: fn(f64, f64, f64) -> Result<f64, String> = match function {
                "duffing.duffing_elliptic_parameter" => duffing::duffing_elliptic_parameter,
                "duffing.duffing_frequency" => duffing::duffing_frequency,
                "duffing.duffing_frequency_shift" => duffing::duffing_frequency_shift,
                _ => duffing::duffing_frequency_expansion,
            };
            fl.f(ok(oracle(amp, w2, eps), function))
        }
        "duffing.duffing_displacement" => {
            let (t, shape) = a.arr("t");
            let out = ok(
                duffing::duffing_displacement(&t, a.f("amplitude"), a.f("omega0_sq"), a.f("eps")),
                function,
            );
            fl.af(&out, &shape)
        }

        // ---- spectrum -------------------------------------------------------------------------
        "spectrum.magnitude_spectrum" => {
            let (sig, _) = a.arr("signal");
            let s = spectrum::magnitude_spectrum(&sig, a.f("fs"), a.u("zero_pad_factor"));
            tuple(vec![fl.af1(&s.freqs), fl.af1(&s.mag), fl.i(s.nfft as i64)])
        }
        "spectrum.detect_peaks" => {
            let (sig, _) = a.arr("signal");
            fl.af1(&spectrum::detect_peaks(
                &sig,
                a.f("fs"),
                a.u("n_peaks"),
                a.f("f_min"),
                a.opt_f("min_separation_hz"),
            ))
        }
        "spectrum.measure_partials_near" => {
            let (sig, _) = a.arr("signal");
            let (expected, _) = a.arr("expected");
            fl.af1(&spectrum::measure_partials_near(
                &sig,
                a.f("fs"),
                &expected,
                a.opt_f("search_hz"),
            ))
        }

        // ---- rotating wave --------------------------------------------------------------------
        "rotating_wave.planar_hessian_cells" => {
            let (p, _) = a.arr("p");
            let (z, _) = a.arr("z");
            let (h_pp, h_pz, h_zz) = rotating_wave::planar_hessian_cells(&p, &z, a.f("a"));
            tuple(vec![fl.af1(&h_pp), fl.af1(&h_pz), fl.af1(&h_zz)])
        }
        "rotating_wave.kc_circular_frequency" => fl.f(ok(
            rotating_wave::kc_circular_frequency(a.f("omega0_sq"), a.f("eps"), a.f("amplitude")),
            function,
        )),
        "rotating_wave.solve_rotating_wave" => {
            let w = solve(a);
            // The Python `RotatingWave`'s field order. `iterations` is written as 0, exactly as the
            // freeze blanked it: ledger #33 settled that a Newton count is not comparable across
            // implementations (17 of 108 fixtures took different paths to the same root), and
            // freezing it would reinstate the assertion `test_rust_parity_rotating_wave.py`'s
            // witness existed to forbid.
            tuple(vec![
                fl.af1(&w.phi),
                fl.af1(&w.psi),
                fl.f(w.omega),
                fl.f(w.frequency),
                fl.f(w.s),
                fl.f(w.amplitude),
                fl.i(w.mode as i64),
                fl.af1(&w.stretch_ratio),
                fl.af1(&w.tension),
                fl.f(w.shape_residual),
                fl.i(0),
                fl.b(w.converged),
                fl.b(w.time_discrete),
            ])
        }
        "rotating_wave.rotating_wave_history" => {
            // Its input is another function's output: the wave is solved on this side, at the
            // recorded parameters, and handed straight over — freezing the history against a wave
            // solved by the other implementation would measure the solver twice.
            let wave_args = a.nested("wave");
            let wave = solve(&wave_args);
            wave_args
                .all_read()
                .unwrap_or_else(|u| panic!("unread wave arguments {u:?}"));
            let (u0, w0, v0, up, wp, vp) = ok(
                rotating_wave::rotating_wave_history(&wave, a.f("fs")),
                function,
            );
            tuple([u0, w0, v0, up, wp, vp].iter().map(|v| fl.af1(v)).collect())
        }

        // ---- horizon --------------------------------------------------------------------------
        "horizon.pitch_error_cents" => {
            let (fd, shape) = a.arr("f_discrete");
            let (fc, _) = a.arr("f_continuum");
            fl.af(&ok(horizon::pitch_error_cents(&fd, &fc), function), &shape)
        }
        "horizon.pitch_horizon" => {
            let (fd, _) = a.arr("f_discrete");
            let (fc, _) = a.arr("f_continuum");
            let (h, monotone) = ok(horizon::pitch_horizon(&fd, &fc, a.f("cents")), function);
            tuple(vec![fl.i(h as i64), fl.b(monotone)])
        }
        "horizon.sinc_horizon_fraction" => fl.f(ok(
            horizon::sinc_horizon_fraction(a.f("cents"), a.i("power")),
            function,
        )),
        "horizon.mode_family" | "horizon.mode_block" => {
            let modes = if function == "horizon.mode_family" {
                ok(horizon::mode_family(a.s("kind"), a.i("count")), function)
            } else {
                ok(horizon::mode_block(a.i("m_max")), function)
            };
            tuple(
                modes
                    .iter()
                    .map(|&(m, n)| tuple(vec![fl.i(m), fl.i(n)]))
                    .collect(),
            )
        }
        "horizon.cancellation_courant" => fl.f(ok(
            horizon::cancellation_courant(a.i("m"), a.i("n")),
            function,
        )),
        "horizon.block_weight" => fl.f(ok(horizon::block_weight(a.i("m"), a.i("n")), function)),

        other => panic!("the record names `{other}`, which this file does not know how to call"),
    }
}

fn solve(a: &Args) -> rotating_wave::RotatingWave {
    let p = rotating_wave::BvpParams {
        l: a.f("L"),
        t: a.f("T"),
        rho: a.f("rho"),
        ea: a.f("EA"),
        fs: a.f("fs"),
        n_cells: a.u("N"),
        theta: a.f("theta"),
        amplitude: a.f("amplitude"),
        mode: a.u("mode"),
        kappa: a.f("kappa"),
        time_discrete: a.b("time_discrete"),
        continuation_steps: a.u("continuation_steps"),
        tol: a.f("tol"),
        maxiter: a.u("maxiter"),
    };
    ok(
        rotating_wave::solve_rotating_wave(&p),
        "rotating_wave.solve_rotating_wave",
    )
}

fn floats_of(v: &Value) -> Vec<f64> {
    v.as_array()
        .expect("an array")
        .iter()
        .map(|x| x.as_f64().expect("a number"))
        .collect()
}

fn ints_of(v: &Value) -> Vec<i64> {
    v.as_array()
        .expect("an array")
        .iter()
        .map(|x| x.as_i64().expect("an integer"))
        .collect()
}

// -- the tests -----------------------------------------------------------------------------------

#[test]
fn every_frozen_case_still_reads_the_same() {
    let mut failures = Vec::new();
    for (key, row) in cases() {
        let function = row["function"].as_str().expect("a function name");
        let args = Args::new(row["args"].as_object().expect("an argument object"));
        let mut fl = Flat::default();
        let structure = answer(function, &args, &mut fl);
        if let Err(unread) = args.all_read() {
            failures.push(format!(
                "{key}: the call never read the recorded arguments {unread:?}"
            ));
            continue;
        }

        let want_structure = row["structure"].as_str().expect("a structure string");
        if structure != want_structure {
            failures.push(format!(
                "{key}: the shape of the answer changed, {structure} against the frozen \
                 {want_structure}. That is a different number of roots, partials or fields -- not \
                 a numerical drift, and no tolerance describes it"
            ));
            continue;
        }
        let want_ints = ints_of(&row["ints"]);
        if fl.ints != want_ints {
            failures.push(format!(
                "{key}: an integer or boolean in the answer changed -- {:?} against {want_ints:?}. \
                 Mode counts, multiplicities, iteration flags and `converged` live here",
                fl.ints
            ));
            continue;
        }
        let want = floats_of(&row["floats"]);
        assert_eq!(
            fl.floats.len(),
            want.len(),
            "{key}: same structure, different float count"
        );
        // Every recorded float is finite (checked below), so a NaN or infinity here is a failure in
        // its own right. Said explicitly, because a `max` fold drops a NaN wherever it sits.
        if let Some(bad) = fl.floats.iter().find(|x| !x.is_finite()) {
            failures.push(format!("{key}: the answer holds a non-finite value, {bad}"));
            continue;
        }
        let scale = want.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let worst = fl
            .floats
            .iter()
            .zip(&want)
            .fold(0.0f64, |m, (g, w)| m.max((g - w).abs()));
        let rel = if scale == 0.0 { worst } else { worst / scale };
        // Machine-readable, for comparing this gap with the one the Python path measured (§47).
        println!("GAP {key} {rel:?}");
        // NaN named explicitly, so the bar cannot be passed by one.
        if rel.is_nan() || rel >= BAR {
            failures.push(format!(
                "{key}: {rel:.3e} of the answer's own scale away from what the Python oracle said \
                 (recorded gap at generation: {}). The bar is {BAR:.0e}",
                row["gap_at_generation"]
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {ROWS} frozen cases moved:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn the_record_is_the_one_that_was_frozen() {
    // A record that lost a row, or had a digit edited, would still be read and compared — against
    // the wrong numbers. The count and a checksum over every double's bit pattern pin it.
    assert_eq!(
        cases().len(),
        ROWS,
        "the record has a different number of rows"
    );

    // FNV-1a over the little-endian bytes of every double: each row in key order, its arguments in
    // key order (nested objects too), then its recorded floats. Only JSON numbers written as floats
    // count — integers and the shapes are compared exactly elsewhere.
    fn walk(v: &Value, out: &mut Vec<f64>) {
        match v {
            Value::Number(n) if n.is_f64() => out.push(n.as_f64().unwrap()),
            Value::Array(xs) => xs.iter().for_each(|x| walk(x, out)),
            Value::Object(m) => {
                let mut keys: Vec<&String> = m.keys().collect();
                keys.sort();
                keys.into_iter().for_each(|k| walk(&m[k], out));
            }
            _ => {}
        }
    }
    let mut keys: Vec<&String> = cases().keys().collect();
    keys.sort();
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    let mut count = 0usize;
    for key in keys {
        let row = &cases()[key];
        let mut doubles = Vec::new();
        walk(&row["args"], &mut doubles);
        doubles.extend(floats_of(&row["floats"]));
        count += doubles.len();
        for d in doubles {
            for byte in d.to_le_bytes() {
                hash ^= u64::from(byte);
                hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
    }
    let want = record()["float_checksum_fnv1a64"]
        .as_str()
        .expect("a checksum");
    assert_eq!(
        format!("{hash:016x}"),
        want,
        "the record's doubles are not the ones converted at §47 ({count} read). If serde_json \
         lost its `float_roundtrip` feature, every decimal parses to a neighbouring double"
    );

    let recorded: usize = cases()
        .values()
        .map(|r| r["floats"].as_array().unwrap().len())
        .sum();
    assert_eq!(
        recorded, 3754,
        "the recorded answers hold a different number of floats"
    );
    assert!(
        cases().values().flat_map(|r| floats_of(&r["floats"])).all(f64::is_finite),
        "a recorded answer is not finite, which the NaN-blind gap fold above relies on never seeing"
    );
}

#[test]
fn every_case_carries_a_measured_gap_rather_than_a_reason_it_could_not_be_measured() {
    // The canary on the record itself. A row's gap column is the Python-versus-Rust gap measured
    // at generation; a STRING there means the two sides could not be compared. The one string
    // allowed is the horizon freeze's "nothing float in this answer", and only on a row that really
    // has no floats and does have integers — otherwise the exemption is a row that asserts nothing.
    for (key, row) in cases() {
        let gap = &row["gap_at_generation"];
        if gap.is_f64() {
            continue;
        }
        assert_eq!(
            gap.as_str(),
            Some(NO_FLOATS),
            "{key}: frozen without a comparison behind it"
        );
        assert!(
            row["floats"].as_array().unwrap().is_empty(),
            "{key}: claims no floats, froze some"
        );
        assert!(
            !row["ints"].as_array().unwrap().is_empty(),
            "{key}: no floats and no integers either -- a frozen row with nothing in it to compare"
        );
    }
}

#[test]
fn the_underdamped_predicate_has_room_to_spare() {
    // A bool is a discrete output, so freezing one is a claim about how far it is from flipping.
    // Carried from `test_analysis_frozen.py`, which inherited it from the deleted parity file: the
    // discriminant `b^2 - 4ac` over five loss configurations and modes 1..40 is never within eleven
    // orders of magnitude of zero relative to `|b^2| + |4ac|`, so no rounding can flip a frozen
    // verdict. A future fixture that narrows it fails here rather than silently answering
    // differently. The arithmetic below is a separate spelling of the predicate's, on purpose.
    let (c, ell, n, kappa, k, theta) = (200.0, 0.65, 128i64, 0.7, 1e-5, 0.5);
    let mut worst = f64::INFINITY;
    let mut saw = BTreeSet::new();
    for (sigma0, sigma1) in [
        (0.0, 0.0),
        (0.5, 1e-5),
        (50.0, 1e-3),
        (1e6, 0.0),
        (1e5, 1.0),
    ] {
        for m in 1..=40i64 {
            saw.insert(damping::discrete_damped_mode_is_underdamped(
                c, ell, n, kappa, k, theta, sigma0, sigma1, m,
            ));
            let p2 = damping::spatial_eigenvalue_p2(n, ell / n as f64, m);
            let q_mode = c * c * p2 + kappa * kappa * p2 * p2;
            let base = 1.0 + theta * k * k * q_mode;
            let aa = base + (sigma0 + sigma1 * p2) * k;
            let bb = -2.0 + (1.0 - 2.0 * theta) * k * k * q_mode;
            let cc = base - (sigma0 + sigma1 * p2) * k;
            let disc = bb * bb - 4.0 * aa * cc;
            let scale = (bb * bb).abs() + (4.0 * aa * cc).abs();
            worst = worst.min(disc.abs() / scale);
        }
    }
    println!("underdamped margin {worst:e}");
    assert!(
        worst > 1e-11,
        "the underdamped margin has narrowed to {worst:.3e}"
    );
    assert_eq!(
        saw,
        BTreeSet::from([false, true]),
        "this sweep no longer exercises both arms of the predicate, so the margin it measures is a \
         margin on one of them"
    );
}
