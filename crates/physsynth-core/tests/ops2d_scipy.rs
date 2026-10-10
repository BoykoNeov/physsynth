//! The von Kármán half of the 2-D builders held against what SciPy actually said — the outside
//! referee for the Airy operator, the corner average and the Airy solve, frozen before the binding
//! that could ask SciPy goes (retirement plan §49).
//!
//! The record is `reference/scipy_ops2d.json`, three parts:
//!
//! * **`airy`** — 16 grids (the ones `AiryStressSolver` is built on across the von Kármán suite,
//!   plus three that are not). For each, the crate's own clamped second differences, then SciPy's
//!   **right-associated** Gram product `Lc_rᵀ (Wa Lc_r)` assembled from them, in canonical CSR.
//!   And where SciPy's LEFT bracketing `(Lc_rᵀ Wa) Lc_r` came out different, those entries'
//!   positions and values — which is what keeps the comparison from going vacuous: on most grids
//!   the two bracketings agree, and a bar that only ever met agreeing grids could not tell which
//!   one the crate builds.
//! * **`lemma`** — the corner average `Acell` on two grids, and SciPy's `Acell.T @ v` for eight
//!   vectors each. In SciPy `csr.T` is a CSC and its matvec is a *scatter*; the crate transposes
//!   to CSR and *gathers*. Both accumulate each output over increasing index, so for a canonically
//!   stored matrix they are the same sum — and this is what checks the premise.
//! * **`superlu`** — SuperLU's solve of the crate's own `B_F` on the 16 grids, three sources each.
//!   The five large grids the binding test also ran (up to 160 × 128) were not recorded — 2.7 MB
//!   for a comparison the human judged not worth carrying (§49): there the crate's solve is held to
//!   its backward error instead, which is what certifies a solve.
//! * **`builders`** — not from the retired test: fingerprints of SciPy's own product for the masked
//!   biharmonic `L @ L` (a rectangle and two staircased guitar outlines) and the free guitar plate's
//!   stiffness, each asserted equal to the crate's to the bit when recorded. §49's plant F (a
//!   product contracting in descending order) moves exactly these, and before this record only the
//!   viewer freeze saw it — which compares values on Windows alone. A 64-bit FNV-1a of the
//!   canonical CSR rather than the matrices, at the human's call: a failure says which matrix, not
//!   which entry.
//!
//! The inputs are not stored. They are the LCG `tests/reductions.rs` uses, rebuilt bit for bit,
//! with each vector's first and last element kept as a canary: a failure there is a broken
//! generator, not a broken operator, and the test says which.

use physsynth_core::ops2d::{
    biharmonic_from_mask, clamped_d2_1d, free_plate_stiffness_from_mask, guitar_mask,
    prune_to_area_carrying, rectangle_mask, AiryStressSolver, Mask, VonKarmanBracket,
};
use physsynth_core::plate::linspace0;
use physsynth_core::radiation::py_round;
use physsynth_core::sparse::Csr;
use serde_json::Value;

mod airy_fixture;
use airy_fixture::*;

fn record() -> Value {
    serde_json::from_str(include_str!("reference/scipy_ops2d.json")).expect("the record")
}

fn f(v: &Value, key: &str) -> f64 {
    v[key]
        .as_f64()
        .unwrap_or_else(|| panic!("`{key}` is a number"))
}

fn u(v: &Value, key: &str) -> usize {
    v[key]
        .as_u64()
        .unwrap_or_else(|| panic!("`{key}` is an integer")) as usize
}

fn doubles(v: &Value) -> Vec<f64> {
    v.as_array()
        .expect("an array")
        .iter()
        .map(|x| x.as_f64().expect("a double"))
        .collect()
}

fn usizes(v: &Value) -> Vec<usize> {
    v.as_array()
        .expect("an array")
        .iter()
        .map(|x| x.as_u64().expect("an index") as usize)
        .collect()
}

fn grid(case: &Value) -> (usize, usize, f64) {
    (u(case, "nx"), u(case, "ny"), f(case, "h"))
}

/// Bit for bit, structure included: a CSR matvec sums a row in stored order, so a different
/// column order is a different operator on the update path, not a cosmetic difference.
fn assert_is_recorded(m: &Csr, rec: &Value, what: &str) {
    let shape = usizes(&rec["shape"]);
    assert_eq!(
        (m.nrows(), m.ncols()),
        (shape[0], shape[1]),
        "{what}: shape"
    );
    assert_eq!(
        m.indptr(),
        usizes(&rec["indptr"]).as_slice(),
        "{what}: indptr"
    );
    assert_eq!(
        m.indices(),
        usizes(&rec["indices"]).as_slice(),
        "{what}: stored column order"
    );
    let want = doubles(&rec["data"]);
    let differing = m.data().iter().zip(&want).filter(|(a, b)| a != b).count();
    assert_eq!(
        differing,
        0,
        "{what}: {differing} of {} entries differ",
        want.len()
    );
}

/// The inputs this run rebuilt, checked against the canaries before anything else is.
fn rebuilt(case: &Value, n: usize) -> Vec<f64> {
    let v = lcg(u(case, "seed") as u64, n);
    assert_eq!(
        (v[0], v[n - 1]),
        (f(case, "first"), f(case, "last")),
        "seed {}: the generator no longer reproduces the recorder's inputs -- fix the LCG, \
         nothing has been compared yet",
        case["seed"]
    );
    v
}

#[test]
fn the_airy_operator_is_scipys_right_associated_gram_product() {
    // The strongest statement this project has about the Airy assembly: not the crate against a
    // second spelling of its own recipe, but against SciPy's kernels doing the arithmetic
    // themselves, on every grid the von Kármán suite builds. The recorded second differences are
    // checked first, so a moved difference and a moved product are told apart.
    let rec = record();
    let cases = rec["airy"].as_array().expect("airy cases");
    assert_eq!(cases.len(), 16);
    let mut witness_grids = 0;
    for case in cases {
        let (nx, ny, h) = grid(case);
        let at = format!("{nx}x{ny} h={h}");
        assert_is_recorded(&clamped_d2_1d(nx, h), &case["d2x"], &format!("D2x {at}"));
        assert_is_recorded(&clamped_d2_1d(ny, h), &case["d2y"], &format!("D2y {at}"));
        let airy = AiryStressSolver::new(nx, ny, h).expect("SPD");
        assert_is_recorded(airy.bf(), &case["bf_right"], &format!("B_F {at}"));

        // Where SciPy's two bracketings part company, the crate must be on the right-hand side of
        // every split, which is a claim only the witnesses can make. What separates them in SciPy
        // is the ORDER of the contraction -- its left-associated intermediate comes back with
        // descending rows -- so this is the half that sees a product summing in the wrong order
        // (§49's plant F). The crate's OWN two bracketings both contract ascending and, on these
        // values, round the same: swapping its parentheses changes no bit anywhere in the
        // workspace (plant L, an equivalent mutant), and nothing here pretends to see it.
        let positions = usizes(&case["left_differs_at"]);
        let left = doubles(&case["left_values_there"]);
        for (&p, &l) in positions.iter().zip(&left) {
            assert_ne!(
                airy.bf().data()[p],
                l,
                "{at}: entry {p} is the LEFT-associated product's"
            );
        }
        if !positions.is_empty() {
            witness_grids += 1;
        }
    }
    // Recorded: (8, 8, 0.0375) with 2 differing entries and (16, 12, 0.06) with 46. SciPy handed
    // back an unsorted left intermediate on all 16 grids; that is a fact about SciPy's kernel with
    // no native analogue (`Csr::from_rows` sorts), so it is provenance here, not a bar.
    assert!(
        witness_grids > 0,
        "no recorded grid distinguishes SciPy's two bracketings -- the bar above can no longer \
         tell an ascending contraction from a descending one"
    );
}

#[test]
fn the_corner_averages_transpose_is_scipys_scatter() {
    // `VonKarmanBracket` stores `acell.transpose()` and gathers through it; SciPy scattered
    // through a CSC. Same order, same doubles -- for a canonically stored matrix. The recorded
    // `Acell` is checked first, so the premise and the lemma fail separately.
    let rec = record();
    let cases = rec["lemma"].as_array().expect("lemma cases");
    assert_eq!(cases.len(), 2);
    for case in cases {
        let (nx, ny, h) = grid(case);
        let bracket = VonKarmanBracket::new(nx, ny, h);
        assert_is_recorded(bracket.acell(), &case["acell"], &format!("Acell {nx}x{ny}"));
        let gather = bracket.acell().transpose();
        let products = case["products"].as_array().expect("products");
        assert_eq!(products.len(), 8);
        for prod in products {
            let v = rebuilt(prod, bracket.acell().nrows());
            assert_eq!(
                gather.matvec(&v),
                doubles(&prod["acell_t_v"]),
                "{nx}x{ny} seed {}: the gather is not SciPy's scatter",
                prod["seed"]
            );
        }
    }
}

/// Group D's bar for the Airy solve, a SCALING LAW rather than a constant. SuperLU is supernodal,
/// so matching it to the bit would be a claim about how SciPy was built (plan §24.2); both solves
/// are backward stable to machine precision, so their forward difference is the condition number
/// times epsilon, and a clamped biharmonic's condition number grows like `N⁴`. The retired test
/// measured the gap at 3.1e-16 on 4 × 4, 1.3e-13 on 24 × 19 and 5.2e-10 on 160 × 128, and fitted
/// this with ~8x slack; on the recorded LCG sources the worst ratio to it is 0.19.
fn airy_solve_tol(nx: usize, ny: usize) -> f64 {
    1e-17 * ((nx * ny) as f64).powi(2)
}

#[test]
fn the_airy_solve_is_superlus_to_the_measured_tolerance() {
    // The crate's sparse LU against SuperLU on the same operator: two SOLVERS, not two copies of
    // this project's code, and the operator is the crate's (pinned equal to SciPy's product above).
    let rec = record();
    let cases = rec["superlu"].as_array().expect("superlu cases");
    assert_eq!(cases.len(), 16);
    let mut worst_ratio = 0.0f64;
    for case in cases {
        let (nx, ny, h) = grid(case);
        let airy = AiryStressSolver::new(nx, ny, h).expect("SPD");
        for solve in case["solves"].as_array().expect("solves") {
            let raw = rebuilt(solve, airy.n_nodes());
            let f_lu = doubles(&solve["x"]);
            assert_eq!(f_lu.len(), airy.n_interior());
            let f_rs = airy.solve(&source(&airy, &raw)).expect("solve");
            let gap: Vec<f64> = live(&airy, &f_rs)
                .iter()
                .zip(&f_lu)
                .map(|(a, b)| a - b)
                .collect();
            let ratio = max_abs(&gap) / (airy_solve_tol(nx, ny) * max_abs(&f_lu));
            assert!(
                ratio <= 1.0,
                "{nx}x{ny} seed {}: the crate's solve is {ratio:.3} of the SuperLU tolerance",
                solve["seed"]
            );
            worst_ratio = worst_ratio.max(ratio);
            // The rim comes back exactly zero -- structural, not a tolerance, and what makes `F`
            // a legal bracket argument.
            for (&v, &m) in f_rs.iter().zip(airy.index_map()) {
                if m < 0 {
                    assert_eq!(v, 0.0, "{nx}x{ny}: a rim node came back nonzero");
                }
            }
            // And SuperLU's own answer is backward stable against the crate's operator, which is
            // what makes the gap above conditioning and not an assembly error on either side.
            let rhs = live_and_load(&airy, nx, ny, h, &raw);
            let be = backward_error(airy.bf(), &f_lu, &rhs);
            assert!(be <= 1e-13, "{nx}x{ny}: SuperLU's backward error {be:.3e}");
        }
    }
    eprintln!("worst gap / tolerance over the record: {worst_ratio:.3}");
}

#[test]
fn the_crates_airy_solve_is_backward_stable_on_the_recorded_grids() {
    // The forward gap above grows with the grid; the BACKWARD error does not, which is the whole
    // explanation for why its tolerance cannot be one small constant. Without this, the growing
    // bar would read as the solver getting worse on finer grids -- the opposite of the truth. The
    // load is assembled here, not by the crate, so a wrong `Wa` or a wrong live ordering inside
    // `solve` fails here too. The five large grids are `ops2d_airy_large.rs`, optimised only.
    let rec = record();
    for case in rec["superlu"].as_array().expect("superlu cases") {
        let (nx, ny, h) = grid(case);
        let airy = AiryStressSolver::new(nx, ny, h).expect("SPD");
        let raw = lcg(11, airy.n_nodes());
        let f_rs = airy.solve(&source(&airy, &raw)).expect("solve");
        let rhs = live_and_load(&airy, nx, ny, h, &raw);
        let be = backward_error(airy.bf(), &live(&airy, &f_rs), &rhs);
        assert!(
            be <= 1e-13,
            "the crate's solve is not backward stable at {nx}x{ny}: {be:.3e}"
        );
    }
}

/// 64-bit FNV-1a over little-endian words: the recorder's fingerprint of a canonical CSR.
fn fnv_words(words: impl IntoIterator<Item = u64>) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for w in words {
        for b in w.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x0100_0000_01b3);
        }
    }
    h
}

/// Shape, `indptr`, `indices`, then each double's bit pattern.
fn csr_fingerprint(m: &Csr) -> u64 {
    let words = [m.nrows() as u64, m.ncols() as u64]
        .into_iter()
        .chain(m.indptr().iter().map(|&v| v as u64))
        .chain(m.indices().iter().map(|&v| v as u64))
        .chain(m.data().iter().map(|v| v.to_bits()));
    fnv_words(words)
}

/// The mask's flags, one byte each, row-major — FNV-1a byte by byte.
fn mask_fingerprint(mask: &Mask) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &live in mask.flags() {
        h ^= live as u64;
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}

/// The pruned guitar mask on `Plate::new`'s grid (`py_round`, `linspace0`, `v - 0.5 * lx`).
fn guitar(lx: f64, ly_asked: f64, n: usize, waist: f64, asym: f64) -> (Mask, f64) {
    let h = lx / (n as f64);
    let ny = (py_round(ly_asked / h) as i64).max(1) as usize;
    let ly = (ny as f64) * h;
    let (xs, ys) = (linspace0(lx, n + 1), linspace0(ly, ny + 1));
    let (mut x, mut y) = (Vec::new(), Vec::new());
    for &yj in &ys {
        for &xi in &xs {
            x.push(xi - 0.5 * lx);
            y.push(yj);
        }
    }
    let raw = guitar_mask(&x, &y, ly, lx, waist, asym, ny + 1, n + 1);
    (prune_to_area_carrying(&raw).0, h)
}

fn hex(v: &Value, key: &str) -> u64 {
    u64::from_str_radix(v[key].as_str().expect("a hex fingerprint"), 16).expect("hex")
}

#[test]
fn the_masked_biharmonic_and_the_free_plate_stiffness_are_scipys_products() {
    // The builders a descending contraction moves that nothing else native saw off Windows. The
    // mask is checked first, so a moved outline and a moved product fail separately.
    let rec = record();
    let cases = rec["builders"].as_array().expect("builder cases");
    assert_eq!(cases.len(), 6);
    for case in cases {
        let (mask, h) = if case["domain"] == "rectangle" {
            (rectangle_mask(u(case, "nx"), u(case, "ny")), f(case, "h"))
        } else {
            guitar(
                f(case, "lx"),
                f(case, "ly"),
                u(case, "n"),
                f(case, "waist"),
                f(case, "asym"),
            )
        };
        assert_eq!(mask.n_live(), u(case, "live"), "{case}: live nodes");
        assert_eq!(
            mask_fingerprint(&mask),
            hex(case, "mask_fnv"),
            "{case}: the mask moved"
        );
        let m = if case["kind"] == "biharmonic" {
            biharmonic_from_mask(&mask, h).0
        } else {
            free_plate_stiffness_from_mask(&mask, h, f(case, "nu"), 1.0, 1.0, None, None).0
        };
        assert_eq!(m.nnz(), u(case, "nnz"), "{case}: nnz");
        assert_eq!(
            csr_fingerprint(&m),
            hex(case, "fnv"),
            "{case}: not SciPy's product (its structure or a last bit)"
        );
    }
}
