"use strict";
/* Web viewer front-end (architecture B). Talks to the local Python backend: POST /simulate -> the
 * core recomputes offline -> we animate the string (slow-mo, decoupled from audio per catch #2),
 * play the sound at its true rate (48 kHz, catch #1), and draw the energy + partials diagnostics
 * gated by loss (catch #4). Vanilla JS + Canvas2D, no framework. */

// ── element handles ─────────────────────────────────────────────────────────────────────────
const $ = (id) => document.getElementById(id);
const modelSel = $("model");
const domainSel = $("domain");
const renderBtn = $("render");
const autoRender = $("autorender");
const statusEl = $("status");
const stringCv = $("string");
const energyCv = $("energy");
const partialsCv = $("partials");
const partialsTitle = $("partials-title");
const partialsSub = $("partials-sub");
const scrub = $("scrub");
const speedInput = $("speed");
const speedVal = $("speed-val");
const animToggle = $("anim-toggle");
const playAudioBtn = $("play-audio");
const loopAudio = $("loop-audio");
const canvasMsg = $("canvas-msg");

const LABELS = {
  T: "tension T", rho: "density ρ", L: "length L", N: "grid N", lambda: "Courant λ",
  kappa: "stiffness κ", theta: "θ (time-avg)", sigma: "loss σ", sigma0: "loss σ₀ (flat)",
  sigma1: "loss σ₁ (HF)", pluck_position: "pluck pos", pickup_position: "pickup pos",
  audio_duration: "audio length", animation_window: "anim window",
  radius: "radius a", Lx: "width Lx", Ly: "height Ly",
  pluck_x: "strike x", pluck_y: "strike y", pluck_width: "strike width",
  pickup_x: "pickup x", pickup_y: "pickup y",
  mu: "plate Courant μ", fs: "sample rate fs", nu: "Poisson ν",
  E: "Young's E", e: "thickness e", w_over_e: "strike w/e",
  amplitude: "amplitude A", EA: "axial EA",
  lam_long: "longitudinal λ", dt_over_t0: "tension ΔT/T₀", tongue_position: "tongue δ/(εA²)",
  mass: "mallet mass M", stiffness: "felt stiffness K", alpha: "felt exponent α",
  strike_velocity: "strike speed v₀", hysteresis: "felt loss λ_h",
  K: "bridge K", detune: "detune (semis)",
  bell_ratio_exp: "bell log₁₀(R/Z₀)",
};

// Per-model slider re-ranging (min/max/step/fixed/val) applied on model switch. The backend mirrors
// these caps. Two params carry DIFFERENT physical meaning per model and must be re-ranged + reset:
// κ (string stiffness ~1 vs plate bending speed ~20) and ρ (areal kg/m² for string/membrane/plate
// vs volumetric kg/m³ for the von Kármán plate). _default restores the string family's values so a
// switch back off plate/vk resets them. A spec's `val` is (re)applied on every model switch; params
// with no `val` in the spec persist and are only clamped into range.
const MODEL_RANGES = {
  // Amplitude IS this model's independent variable (the shift scales as A²), so it leads the panel.
  // The string path's inherited 1e-3 would render a 0.8-cent near-null; 0.02 gives ~270 cents. The
  // caps are cost- and honesty-driven: every step runs a tension root-find (~2x a 2D membrane step),
  // and dT/T0 is bounded server-side to keep the mode below its parametric-breakup threshold.
  tension: { N: { max: 256, val: 128 }, kappa: { val: 1.0 }, audio_duration: { max: 3, val: 1 },
             amplitude: { val: 0.02 }, sigma0: { val: 0 }, sigma1: { val: 0 } },
  // Loss is ON by default here — the OPPOSITE of the tension string above, and load-bearing:
  // sigma0 > 0 lets the note settle to a steady Helmholtz limit cycle instead of growing without
  // bound, and sigma1 > 0 damps the high partials so the corner stays clean (one slip per period)
  // rather than raucous (~18). sigma1's string-path max of 0.01 is far too small for the 0.05 the
  // bow wants, so it is re-ranged. kappa = 0 (a flexible string, f1 = c/2L) isolates the bow
  // physics; lambda = 0.9 keeps a hair of headroom below the Nyquist mode for the coupled solve.
  bow: { N: { max: 256, val: 100 }, lambda: { max: 1.0, val: 0.9 }, kappa: { val: 0.0 },
         sigma0: { val: 0.5 }, sigma1: { max: 0.2, step: 0.005, fixed: 3, val: 0.05 },
         pickup_position: { val: 0.33 }, audio_duration: { max: 3, val: 2 } },
  membrane: { N: { max: 100, val: 80 }, lambda: { max: 0.7, val: 0.6 },
              audio_duration: { max: 2, val: 1.5 } },
  // Mallet → drum (model #7): the membrane heatmap path, but the head starts at REST and a lumped
  // mass strikes it. N is capped at 80 (each step also runs a contact root-find) and defaults to
  // 60 for a quick render; loss is OFF by default so the out-of-box verdict is CONSERVATION (drift
  // < 1e-10), the mallet's money test. The felt defaults (K = 5e4, M = 0.02, α = 2.3, v₀ = 3) are
  // the core's canonical rig, where the strike is well-resolved (~48 steps/contact) and the
  // "inefficient point-mass exciter" story reads cleanly: restitution ≈ 1, the head keeps ~0.05 %.
  mallet: { N: { max: 80, val: 60 }, lambda: { max: 0.7, val: 0.5 }, sigma: { val: 0 },
            audio_duration: { max: 2, val: 1 },
            mass: { val: 0.02 }, stiffness: { val: 50000 }, alpha: { val: 2.3 },
            strike_velocity: { val: 3.0 }, hysteresis: { val: 0 } },
  plate: { N: { max: 80, val: 60 }, kappa: { min: 2, max: 80, step: 0.5, fixed: 1, val: 20 },
           rho: { min: 0.001, max: 0.02, step: 0.0005, fixed: 4, val: 0.005, unit: "kg/m²" },
           Lx: { val: 1.0 }, Ly: { val: 1.0 }, audio_duration: { max: 2, val: 1 } },
  vk: { N: { min: 8, max: 32, val: 20 },
        rho: { min: 2000, max: 12000, step: 100, fixed: 0, val: 7800, unit: "kg/m³" },
        Lx: { val: 0.3 }, Ly: { val: 0.3 }, audio_duration: { max: 1, val: 0.5 } },
  // The slowest model in the viewer, and irreducibly so: fs is forced ~22x a normal string's by the
  // longitudinal wave (lam_long <= 1) and every step is a 3-field vector Newton solve, so N is capped
  // at 32 and defaults to 16 — the grid the whirl rig uses, where the tongue is refinement-invariant.
  // kappa = 0 (the soft plane; the whirl regime forces it anyway) and loss is OFF: this model's
  // claim is that it CONSERVES straight through a 60x parametric blow-up, which needs sigma = 0.
  // amplitude is capped well below model #9's mode-1 breakup (dT/T0 ~ 4.4 at A = 0.06 here).
  geometric: { N: { min: 8, max: 32, val: 16 }, kappa: { val: 0.0 },
               EA: { min: 5, val: 100 },
               amplitude: { min: 0.0005, max: 0.02, step: 0.0005, fixed: 4, val: 0.004 },
               sigma0: { val: 0 }, sigma1: { val: 0 }, pickup_position: { val: 0.25 } },
  // Sympathetic / coupled strings (J = 2). A closed undriven system, lossless this batch, so it
  // rides the ordinary conservation-drift panel; the CLAIM is the bridge trace in the second panel.
  // K is the shared bridge stiffness (the core's exact dense guard rejects an over-stiff one). N ~
  // 100 matches the diagnose rig; audio is the plucked string's pickup over the slosh.
  sympathetic: { N: { min: 16, max: 160, val: 100 }, lambda: { max: 0.99, val: 0.9 },
                 K: { val: 8000 }, detune: { val: 0 },
                 pluck_position: { val: 0.3 }, pickup_position: { val: 0.1 },
                 audio_duration: { max: 3, val: 2 } },
  // Transfer regime: a SOFTER bridge is frequency-selective (the resonant transfer is the point), so
  // K resets to 1500 and the detune slider appears (gated OUT of `normal`, where the bit-exact
  // w_b == 0 needs two identical strings).
  "sympathetic:transfer": { K: { val: 1500 }, detune: { val: 0 } },
  // Weinreich two-stage decay: a LOSSY bridge (the body-loss slider, weinreich-only) + a FINE detune.
  // A piano unison is mistuned by a few cents, not semitones, so detune is re-ranged to 0..0.4 semis
  // in 0.01-semi (~1-cent) steps — distinct from transfer's 0..12 semitones. K resets to 6000.
  "sympathetic:weinreich": {
    K: { val: 6000 }, sigma_body: { val: 20 },
    detune: { min: 0, max: 0.4, step: 0.01, fixed: 2, val: 0 },
  },
  // Jawari / buzzing bridge (model #8 curved). Loss is ON by default and load-bearing: "SUSTAINED
  // brightness" is meaningless on a lossless string, where every mode sustains by definition — the
  // signal only exists because sigma0 would darken a clean string while the bridge keeps re-injecting
  // highs. sigma0 also GATES the verdict (0 -> conservation drift through the wrap, > 0 -> passivity
  // + the 2·sigma0 oracle, which survives here because the barrier is elastic and dissipates nothing).
  // amplitude defaults to the test suite's 8 mm, which puts downswing/depth at 3.8 — comfortably
  // above the ~1.5 floor below which the string only grazes the crest. lambda = 0.4 (not the string
  // path's 1.0) gives the coupled contact solve headroom; N = 100 puts ~15 nodes under the bridge,
  // enough to resolve the travelling wrap and well under the dense-solve BLAS cliff.
  jawari: { N: { min: 32, max: 128, val: 100 }, lambda: { min: 0.2, max: 0.9, val: 0.4 },
            amplitude: { min: 0.001, max: 0.04, step: 0.001, fixed: 3, val: 0.008 },
            sigma0: { val: 0.5 }, sigma1: { val: 0 },
            // step MUST be re-ranged with the val: index.html ships step = 0.1 for the long runs of
            // every other model, and a range input SNAPS an off-grid value to its step. Without this
            // the shipped default silently became 0.2 — the browser could not express the 0.24 the
            // tests pin, so the tested config and the rendered one had drifted apart (elevation
            // 3.44x tested vs 2.75x rendered, a hair over the 2.5x gate).
            pickup_position: { val: 0.5 },
            audio_duration: { min: 0.1, max: 1.5, step: 0.01, fixed: 2, val: 0.24 } },
  // Acoustic bore + bell (the wind leg). The first model here with NO loss slider at all: the
  // bell's radiation is the only loss, and it is BOOKED into Bore.energy(), which is what lets a
  // radiating tube keep the conservation verdict. λ is absent for a different reason — at λ = 1,
  // fs = c₀N/(λL), so steps scale as 1/λ and the budget dies at λ = 0.878 before the reflection
  // run, while the payoff (0.07–0.67 cents) is inaudible. The λ claim is served instead by an
  // eigenvalue panel that needs no time-stepping. N buys the SAMPLE RATE, not just the grid, so
  // cost per second of audio goes as N²; 256/1.5 s is the worst passing render at ~3 s.
  // audio_duration's min is 0.25 because that is where the odd/even gate is measured (the ratio is
  // set by the FFT window, not by physics). animation_window is re-ranged HARD — the shared 0.3 max
  // is a cost hole: at N = 256 even 0.3 s is 52k animation steps on top of the render.
  bore: { N: { min: 32, max: 256, val: 128 }, L: { val: 0.5 },
          pickup_position: { val: 0.1 },
          bell_ratio_exp: { min: -4, max: 1.4, step: 0.1, fixed: 1, val: -3.5 },
          audio_duration: { min: 0.25, max: 1.5, step: 0.05, fixed: 2, val: 0.5 },
          animation_window: { min: 0.005, max: 0.1, step: 0.005, fixed: 3, val: 0.03 } },
  // Regime-level ranges, keyed "model:domain" and merged AFTER the model spec (see
  // applyModelRanges). The phantom regime is the first customer and needs both: κ = 8 is its
  // microscope (the geometric model defaults κ = 0, which is a HARMONIC string — every phantom
  // would coincide with a partial exactly and the panel would render its own opposite), and N = 32
  // is the test rig's grid, which the viewer copies so it inherits the suite's validation.
  // The same "every param a regime re-ranges must reset one level up" discipline as _default:
  // MODEL_RANGES.geometric already resets κ → 0 and N → 16, so switching regime away restores them.
  "geometric:phantom": { kappa: { val: 8.0 }, N: { val: 32 }, amplitude: { val: 0.0015 } },
  // `amplitude` is shown only for the tension string, but gatherParams sends every slider — so it
  // must reset to the linear string path's historical 1e-3 on switch, or those models would silently
  // re-render at the tension default (a pure scale for a linear model, but not bit-for-bit).
  // Every param a model above re-ranges must be reset here to its index.html default, or it leaks
  // into the next model on switch (gatherParams sends every slider, hidden ones included). sigma0/
  // sigma1/pickup_position joined `amplitude` when the bow arrived: the bow needs sigma1 = 0.05,
  // which is 25x the damped string's default AND outside its own slider max of 0.01, so without
  // the reset a bow → damped switch would silently render a wildly over-damped string on a stale
  // range. (This also fixes the same leak tension's sigma0 = 0 already had.)
  // amplitude and EA are re-ranged by BOTH nonlinear string models, so both reset here.
  // The jawari re-ranges the LOWER bound of lambda (0.2 — the contact solve wants headroom) and of
  // audio_duration (0.1 s — its runs are short and expensive), and applyModelRanges only rewrites a
  // bound the spec actually names. So both mins are reset here to index.html's values; without them
  // a jawari → anything switch leaves the next model able to select a lambda or duration its own
  // path never intended. The `val`-only leak, one field over. audio_duration also resets step and
  // fixed, because the jawari has to narrow BOTH (see its spec) and a stale 0.01 step would leave
  // every other model's multi-second slider crawling in hundredths.
  _default: { N: { min: 16, max: 512 }, lambda: { min: 0.5, max: 2.0, val: 1.0 },
              kappa: { min: 0, max: 8, step: 0.05, fixed: 2, val: 1.0 },
              rho: { min: 0.001, max: 0.02, step: 0.0005, fixed: 4, val: 0.005, unit: "kg/m²" },
              amplitude: { min: 0.001, max: 0.06, step: 0.001, fixed: 3, val: 0.001 },
              EA: { min: 0, val: 100 },
              sigma0: { min: 0, max: 20, step: 0.1, val: 1.0 },
              sigma1: { min: 0, max: 0.01, step: 0.0001, fixed: 4, val: 0.002 },
              pickup_position: { val: 0.1 },
              K: { min: 500, max: 10000, step: 100, fixed: 0, val: 8000 },
              detune: { min: 0, max: 12, step: 0.1, fixed: 1, val: 0 },
              sigma_body: { min: 0, max: 80, step: 1, fixed: 0, val: 0, unit: "s⁻¹" },
              audio_duration: { min: 0.2, max: 6, step: 0.1, fixed: 1, val: 2 },
              // L and animation_window joined when the bore arrived: it is the first model to
              // re-range EITHER (L → 0.5 m, and the animation window down to a 0.1 s max because
              // its own budget cannot cover the shared 0.3). Without these resets a bore → string
              // switch would leave a 0.5 m string on a window slider that could no longer reach
              // 0.06 s. Same `val`-only leak the jawari's mins had, two fields over.
              L: { min: 0.25, max: 2.0, step: 0.05, val: 1.0 },
              animation_window: { min: 0.01, max: 0.3, step: 0.005, fixed: 3, val: 0.06 } },
};

// Secondary select repurposed per model: geometry (membrane), boundary (plate / von Kármán) or
// REGIME (the geometric string — three claims, one string, cheapest first).
const DOMAIN_MODELS = ["membrane", "mallet", "plate", "vk", "geometric", "sympathetic", "bore"];
const DOMAIN_OPTS = {
  membrane: [["circle", "Circle (drumhead)"], ["rectangle", "Rectangle"]],
  mallet: [["circle", "Circle (drumhead)"], ["rectangle", "Rectangle"]],
  plate: [["supported", "Simply-supported (#5)"], ["free", "Free edge — Chladni (#5b)"]],
  vk: [["supported", "Supported gong (#6)"], ["free", "Free-edge cymbal (#6)"]],
  geometric: [["rotating", "Rotating wave — exact circle"],
              ["planar", "Planar — max|w| = 0 exactly"],
              ["whirl", "Whirling — the Mathieu tongue"],
              ["phantom", "Phantom partials — the bridge force"]],
  sympathetic: [["normal", "Normal modes — the bridge oracle"],
                ["transfer", "Sympathetic transfer"],
                ["weinreich", "Weinreich two-stage decay"]],
  // The bore's secondary select is the far END of the tube, and it is what drawBore switches on.
  // Radiating is the default: a σ = 0 closed-open tube with no bell rings forever at constant
  // amplitude, a sustained buzz that never decays (loss-default-ON, the bow's and jawari's rule).
  // The ideal open end is kept as the lossless contrast — a perfect mirror, nothing radiates.
  bore: [["radiating", "Radiating bell — sound leaves"],
         ["open", "Ideal open end (r = −1)"]],
};
const DOMAIN_LABELS = { membrane: "Domain", mallet: "Drum shape", geometric: "Regime",
                        sympathetic: "Regime", bore: "Far end" };

const sliders = {};      // param -> <input>
const updaters = {};     // param -> fn() that refreshes its value label
const fixedOf = {};      // param -> decimal places for the value label (re-rangeable)
const scaleOf = {};      // param -> multiplier applied in gatherParams (E in GPa, e in mm)
const unitOf = {};       // param -> value-label unit suffix (re-rangeable: ρ is areal vs volumetric)
let payload = null;
let dims = 1;            // 1 = string polyline, 2 = membrane heatmap
let frames = null, nFrames = 0, width = 0, fieldAmp = 1, animDt = 1e-3;
let gridNx = 0, gridNy = 0, maskData = null, gridMeta = null, heatCv = null;
// Geometric string (model #10): three stacked fields per frame + the (u, w) orbit trail.
let isGeom = false, orbitU = null, orbitW = null, orbitPerFrame = 1, uwAmp = 1, vAmp = 1;
// Stacked-strip `drawFields` state (geometric u/w/v AND sympathetic string A/B): the field count,
// per-field vertical scales, display names and a shared colour palette. Set once at load.
let isSymp = false, nFields = 1, fieldAmps = [1], fieldLabels = [];
// Jawari (model #8, curved): the bridge profile under the string + the travelling wrap edge.
let isJawari = false, barrierProfile = null, wrapFrames = null;
// Acoustic bore: a PRESSURE field, not a displacement one — and the two ends are not alike, which
// is the whole physics (see drawBore).
let isBore = false, boreEnv = null, boreRad = null, boreEnds = ["closed", "open"];
const FIELD_COLORS = ["#4cc2ff", "#ff8f4c", "#9d7bff"];
let audioSamples = null, audioFs = 48000, audioBuf = null, audioCtx = null, audioSrc = null;
let speed = 0.02, animPlaying = true, scrubbing = false, currentFrame = 0, animStart = 0;
let autoTimer = null;

// ── slider construction ─────────────────────────────────────────────────────────────────────
function guessFixed(step) {
  if (step >= 1) return 0;
  if (step >= 0.1) return 1;
  if (step >= 0.01) return 2;
  if (step >= 0.001) return 3;
  return 4;
}

function buildSliders() {
  document.querySelectorAll(".slider").forEach((el) => {
    const d = el.dataset;
    fixedOf[d.param] = d.fixed !== undefined ? +d.fixed : guessFixed(+d.step);
    scaleOf[d.param] = d.scale !== undefined ? +d.scale : 1;
    unitOf[d.param] = d.unit || "";
    el.innerHTML =
      `<div class="row"><span class="name">${LABELS[d.param] || d.param}</span>` +
      `<span class="value" id="v-${d.param}"></span></div>` +
      `<input type="range" id="s-${d.param}" min="${d.min}" max="${d.max}" ` +
      `step="${d.step}" value="${d.val}">`;
    const input = el.querySelector("input");
    const valEl = el.querySelector(".value");
    const update = () => {
      const u = unitOf[d.param] ? " " + unitOf[d.param] : "";
      valEl.textContent = (+input.value).toFixed(fixedOf[d.param]) + u;
    };
    input.addEventListener("input", () => { update(); onControlChange(d.param); });
    update();
    sliders[d.param] = input;
    updaters[d.param] = update;
  });
}

function param(name) { return sliders[name] ? +sliders[name].value : undefined; }

function setSlider(name, val) {
  if (!sliders[name]) return;
  sliders[name].value = val;
  if (updaters[name]) updaters[name]();
}

// ── model-dependent visibility, ranges + hints ───────────────────────────────────────────────
// Repopulate the secondary select for the current model (geometry for the membrane, boundary for
// the plate family). Preserves the current value if still valid; sets the label text.
function populateDomain(model) {
  const opts = DOMAIN_OPTS[model];
  if (!opts) return;                       // hidden for string models; leave as-is
  const prev = domainSel.value;
  domainSel.innerHTML = opts.map(([v, l]) => `<option value="${v}">${l}</option>`).join("");
  if (opts.some(([v]) => v === prev)) domainSel.value = prev;
  const lbl = $("domain-label");
  if (lbl) lbl.textContent = DOMAIN_LABELS[model] || "Boundary";
}

function updateVisibility() {
  const m = modelSel.value;
  const usesDomain = DOMAIN_MODELS.includes(m);
  const d = usesDomain ? domainSel.value : "";
  document.querySelectorAll("[data-show]").forEach((el) => {
    el.hidden = !el.dataset.show.split(" ").includes(m);
  });
  // data-domain sliders (radius / Lx / Ly) are gated by the secondary select; the multi-value list
  // lets Lx/Ly show for both the membrane rectangle AND either plate boundary. Model gating is by
  // the parent fieldset's data-show, so here we only test the domain membership.
  document.querySelectorAll("[data-domain]").forEach((el) => {
    el.hidden = !usesDomain || !el.dataset.domain.split(" ").includes(d);
  });
  // The inverse gate, for sliders SHARED with a model that has no domains: data-domain would hide
  // them everywhere else (a param whose element is data-domain-gated is hidden whenever the model
  // has no secondary select at all), so "hide only in these regimes" needs its own attribute.
  // Used by κ and amplitude (which the whirl derives rather than reads) and by the animation window
  // (which the phantom regime fixes at 0.10 s of physics).
  //
  // Recomputed from scratch, never read off el.hidden. The first cut early-returned when el.hidden
  // was already true, meaning "model gating hid it, leave it" — but that also LATCHED any element a
  // previous regime had hidden. It only worked because every element using this attribute also had
  // data-show, whose pass above rewrites el.hidden unconditionally each time. The animation window
  // has no data-show (it is shown for almost every model), so once you visited the phantom regime
  // its slider stayed hidden forever — through every other regime, until a reload.
  document.querySelectorAll("[data-hide-domain]").forEach((el) => {
    const modelHides = el.hasAttribute("data-show")
      && !el.dataset.show.split(" ").includes(m);
    const domainHides = el.hasAttribute("data-domain")
      && (!usesDomain || !el.dataset.domain.split(" ").includes(d));
    const regimeHides = usesDomain && el.dataset.hideDomain.split(" ").includes(d);
    el.hidden = modelHides || domainHides || regimeHides;
  });
}

// Re-range sliders to the current model's caps/defaults (see MODEL_RANGES). Merges _default with the
// model spec (model wins); applies min/max/step/fixed, (re)sets `val` when the spec gives one, and
// always clamps the current value into range. Run on model switch only, so resets are intentional.
// True when some regime of `model` re-ranges sliders, i.e. a "model:domain" key exists. Gating on
// this keeps the secondary select's existing behaviour intact everywhere else: a membrane
// circle→rectangle or a plate supported→free switch must NOT reset the user's sliders, and only
// models that declare regime ranges get re-ranged on a domain change.
function hasRegimeRanges(model) {
  return Object.keys(MODEL_RANGES).some((k) => k.startsWith(model + ":"));
}

// Merge range specs PER PARAM, not per layer. Object.assign is shallow, so a later layer's
// {val: 0.0015} would REPLACE the earlier {min, max, step, fixed, val} outright rather than override
// one field of it — the slider would then keep whatever min/max/step index.html last left on it.
// That is not theoretical: it snapped the phantom regime's amplitude from 0.0015 to 0.002 on a stale
// step="0.001" and quietly rendered the wrong physics. The _default-leak trap, one level down.
function mergeSpecs(...layers) {
  const out = {};
  for (const layer of layers) {
    for (const k in (layer || {})) out[k] = Object.assign({}, out[k], layer[k]);
  }
  return out;
}

function applyModelRanges() {
  const regimeKey = domainSel ? modelSel.value + ":" + domainSel.value : "";
  const spec = mergeSpecs(MODEL_RANGES._default, MODEL_RANGES[modelSel.value],
                          MODEL_RANGES[regimeKey]);
  for (const pkey in spec) {
    const inp = sliders[pkey];
    if (!inp) continue;
    const s = spec[pkey];
    if (s.min !== undefined) inp.min = String(s.min);
    if (s.max !== undefined) inp.max = String(s.max);
    if (s.step !== undefined) inp.step = String(s.step);
    if (s.fixed !== undefined) fixedOf[pkey] = s.fixed;
    if (s.unit !== undefined) unitOf[pkey] = s.unit;
    if (s.val !== undefined) inp.value = String(s.val);
    const lo = +inp.min, hi = +inp.max;
    if (+inp.value > hi) inp.value = String(hi);
    if (+inp.value < lo) inp.value = String(lo);
    if (updaters[pkey]) updaters[pkey]();
  }
}

function updateLambdaHint() {
  const m = modelSel.value;
  const hint = $("lambda-hint");
  if (m === "plate") {
    const mu = param("mu");
    hint.textContent = `μ = κ·k/h² = ${mu.toFixed(2)}  (implicit → no CFL; large μ is coarse-but-`
      + `stable, cost rises at LOW μ)`;
    hint.style.color = "var(--muted)";
  } else if (m === "vk") {
    const fs = param("fs");
    hint.textContent = `fs = ${Math.round(fs)} Hz  (oversample the nonlinearity: higher = truer, `
      + `more cost; κ is derived from E, e, ν, ρ)`;
    hint.style.color = "var(--muted)";
  } else if (m === "membrane" || m === "mallet") {
    const lam = param("lambda");
    hint.textContent = `λ = c·k/h = ${lam.toFixed(2)}  (2D CFL: λ ≤ 1/√2 ≈ 0.71; no λ is `
      + `dispersionless)`;
    hint.style.color = lam > 0.708 ? "var(--bad)" : "var(--muted)";
  } else if (m === "geometric") {
    // The inversion that IS this model: λ_long is the knob, λ is whatever falls out. Showing both
    // is the point — the familiar λ reads a reassuring ~0.04 in exactly the regime that works.
    const lamL = param("lam_long"), EA = param("EA") * 1e3, T = param("T"), rho = param("rho");
    const cLong = Math.sqrt(EA / rho), c = Math.sqrt(T / rho);
    const fs = cLong * param("N") / (param("L") * lamL);
    hint.textContent = `λ_long = c_long·k/h = ${lamL.toFixed(2)} → fs = ${fmt(fs)} Hz, λ = `
      + `${(c * param("N") / (param("L") * fs)).toFixed(3)}. c_long/c = ${(cLong / c).toFixed(0)}× — `
      + `that ratio is the whole cost of this model.`;
    hint.style.color = "var(--muted)";
  } else if (m === "bore") {
    // λ is PINNED at 1 here and has no slider, so the hint reports the derived sample rate instead
    // — which is the number that actually bites: fs = c₀N/L, so N buys the sample rate as well as
    // the grid and cost per second of audio goes as N².
    const N = param("N"), L = param("L");
    hint.textContent = `λ = 1 (pinned, dispersionless) → fs = c₀N/L = ${fmt(343 * N / L)} Hz. N `
      + `buys the SAMPLE RATE too, so cost per second of audio ~ N². One transit = L/c₀ = `
      + `${(1e3 * L / 343).toFixed(2)} ms, four times shorter than the period of f₁.`;
    hint.style.color = "var(--muted)";
  } else if (m === "sympathetic") {
    const lam = param("lambda");
    hint.textContent = `λ = c·k/h = ${lam.toFixed(2)}  (must be < 1: the bridge spring pushes the `
      + `string's Nyquist mode unstable at λ = 1)`;
    hint.style.color = lam >= 1 ? "var(--bad)" : "var(--muted)";
  } else if (m === "ideal" && param("lambda") > 1.0) {
    hint.textContent = "λ>1 breaks the explicit ideal string's CFL (will error). Stiff/damped allow"
      + " it.";
    hint.style.color = "var(--bad)";
  } else {
    const lam = param("lambda");
    hint.textContent = m === "ideal"
      ? `λ = c·k/h = ${lam.toFixed(2)}  (1.0 = exact, dispersionless)`
      : `λ = ${lam.toFixed(2)}  (implicit scheme — no CFL limit)`;
    hint.style.color = "var(--muted)";
  }
  // von Kármán: on the supported gong the strike is the (1,1) eigenmode (a clean glide), so the
  // strike-position sliders are ignored; the free cymbal takes a positioned crash.
  const vkHint = $("vk-strike-hint");
  if (vkHint) {
    vkHint.textContent = (m === "vk" && domainSel.value === "supported")
      ? "supported gong: struck in its (1,1) mode — strike x/y/width are ignored (pickup still used)"
      : (m === "vk" ? "free cymbal: a positioned crash (multi-mode wash)" : "");
  }
  // Tension string: dT/T0 = EA·A²·p₁²/(4T) is the load-bearing quantity — amplitude alone is a
  // proxy, since EA and T move it just as hard. The server rejects above ~4.45, where the mode
  // parametrically breaks up. Show it live so the ceiling is visible while dragging.
  const tHint = $("tension-hint");
  if (tHint) {
    if (m === "tension") {
      const A = param("amplitude"), EA = param("EA") * 1e3, T = param("T"), L = param("L");
      const dt = EA * A * A * Math.pow(Math.PI / L, 2) / (4 * T);
      tHint.textContent = `ΔT/T₀ = ${dt.toFixed(2)}  ·  the mode breaks up above ~4.45 (real, `
        + `energy-conserving physics — but the Duffing shift stops applying there)`;
      tHint.style.color = dt > 4.45 ? "var(--bad)" : "var(--muted)";
    } else {
      tHint.textContent = "";
    }
  }
  // Geometric string: EA is the AXIAL stiffness here, and the nonlinearity coefficient is EA − T₀
  // (so EA = T₀ is exactly linear — the regression anchor that reduces u bit-for-bit to model #3).
  // The same slider means the *coefficient itself* on the tension string: mind the swap.
  const gHint = $("geom-hint");
  if (gHint) {
    if (m === "geometric") {
      const EA = param("EA") * 1e3, T = param("T");
      gHint.textContent = `EA is the AXIAL stiffness (model #9's EA is the coefficient — here that `
        + `is EA − T₀ = ${fmt(EA - T)} N). EA = T₀ ⇒ exactly linear. EA/T₀ = `
        + `${(EA / T).toFixed(0)} (real strings: 150–600).`;
      gHint.style.color = "var(--muted)";
    } else {
      gHint.textContent = "";
    }
  }
  // Whirl: the tongue is dimensionless, so the hint speaks in frac and never in κ_w (which is
  // derived, and moves with N). Unstable ⟺ 0 < frac < ½, peak at ¼; the upper edge is SOFT.
  const wHint = $("whirl-hint");
  if (wHint) {
    if (m === "geometric" && domainSel.value === "whirl") {
      const f = param("tongue_position");
      const where = f === 0 ? "degenerate: κ_w = κ_u ⇒ no tongue. A displaced seed just rotates the "
                            + "plane (1.00×); a velocity seed grows SECULARLY, not exponentially."
        : f < 0.5 ? (Math.abs(f - 0.25) < 0.06 ? "at the tongue's peak — fastest growth"
                                               : "inside the tongue — it whirls")
        : "outside the tongue — the growth should die (the upper edge is soft, so it fades)";
      wHint.textContent = `δ/(εA²) = ${f.toFixed(2)} · ${where}`;
      wHint.style.color = f > 0 && f < 0.5 ? "var(--muted)" : "var(--warn, var(--muted))";
    } else {
      wHint.textContent = "";
    }
  }
  // Phantom: κ is a MICROSCOPE, not a thumb on the scale — the r² → v mechanism is completely
  // κ-independent; κ only decides whether the gap a phantom lands in is visible. The defect the
  // panel gates on is f₂−2f₁ ≈ 3B·f₁ MINUS the θ-scheme's own dispersion, which is why the hint
  // speaks in the measured B and warns well above κ=0 (at κ≈2 the two cancel).
  const pHint = $("phantom-hint");
  if (pHint) {
    if (m === "geometric" && domainSel.value === "phantom") {
      const c = Math.sqrt(param("T") / param("rho")), L = param("L"), kap = param("kappa");
      const B = (Math.PI ** 2 * kap * kap) / (c * c * L * L);
      const defect = 3 * B * (c / (2 * L));         // continuum estimate; the run measures the truth
      const fs = Math.sqrt(param("EA") * 1e3 / param("rho")) * param("N") / (L * param("lam_long"));
      const secs = Math.round(0.1 * fs * 2.8e-3);   // ~2.8 ms/step incl. panel telemetry (measured)
      pHint.textContent =
        `B = ${B.toExponential(2)} ⇒ defect ≈ ${defect.toFixed(1)} Hz before dispersion. `
        + (defect < 4 ? "TOO HARMONIC — the phantoms collapse onto the partials; raise κ. " : "")
        + `Two modes plucked, 0.10 s of bridge force measured — about ${secs} s to render.`;
      pHint.style.color = defect < 4 ? "var(--warn, var(--muted))" : "var(--muted)";
    } else {
      pHint.textContent = "";
    }
  }
  // Bow: Schelleng's playable force window is real but has NO closed form in the core — the tests
  // just pick known-good points — so this reports the empirical rule they use rather than inventing
  // an Fmin/Fmax. force ~ 4·v_bow holds the window across bow speeds; force <= 0.4 stays clean out
  // to beta = 0.25. Outside the window the note crushes or goes raucous: real physics, and the
  // stick-slip panel says so rather than failing.
  const bHint = $("bow-hint");
  if (bHint) {
    if (m === "bow") {
      const beta = param("bow_position") / param("L");
      bHint.textContent = `β = ${beta.toFixed(2)} (slip ≈ β of each period)  ·  Schelleng: `
        + `force ≈ 4·v_bow ≈ ${(4 * param("v_bow")).toFixed(2)} N keeps the window across speeds`;
      bHint.style.color = "var(--muted)";
    } else {
      bHint.textContent = "";
    }
  }
  // Loss nudges for the two nonlinear string models. They default OPPOSITE ways, each for a good
  // reason, so each default hides something different and each needs its own nudge. The tension
  // string renders lossless (its shift panel measures a lossless run), which hides the audible
  // glide. The bow renders lossy (σ₀ settles the note to a steady Helmholtz cycle, σ₁ keeps the
  // corner clean), which hides the exact balance closure.
  const lossHint = $("loss-hint");
  if (lossHint) {
    const quiet = param("sigma0") === 0 && param("sigma1") === 0;
    if (m === "tension") {
      lossHint.textContent = quiet
        ? "σ = 0 → steady pitch. Add loss (σ₀ ≈ 1) to hear the tone glide down as the amplitude "
          + "decays."
        : "the tone glides down as it decays — the shift panel measures its own lossless run, so "
          + "it stays put.";
    } else if (m === "bow") {
      lossHint.textContent = quiet
        ? "σ = 0 → every joule the bow puts in stays in the string: the two curves become one. The "
          + "note grows without bound (not musical) — that is the price of exact closure."
        : "σ > 0 → a steady Helmholtz note. Set σ₀ = σ₁ = 0 to watch E−E₀ and the bow work close "
          + "to machine precision.";
    } else if (m === "geometric") {
      lossHint.textContent = quiet
        ? "σ = 0 → the claim: energy conserves to ~1e-12 straight THROUGH a 60× whirl blow-up. "
          + "That is what separates redistribution from a diverging solve."
        : "σ > 0 → the drift verdict becomes a passivity check, and loss competes with the whirl's "
          + "growth: the tongue's threshold moves. Set σ = 0 to see the conservation claim.";
    } else {
      lossHint.textContent = "";
    }
    lossHint.style.color = "var(--muted)";
  }
  // Sympathetic: the coupling story per regime — the bridge oracle vs the tuned transfer.
  const syHint = $("symp-hint");
  if (syHint) {
    if (m === "sympathetic") {
      if (domainSel.value === "transfer") {
        const d = param("detune");
        syHint.textContent = d < 0.05
          ? "unison: pluck A and the tuned neighbour B rings up in sympathy. Detune B and watch the "
            + "transfer fall away."
          : `B is ${d.toFixed(1)} semis flat of A: off its partial the transfer weakens. `
            + `K = ${Math.round(param("K"))} N/m sets how selective the coupling is.`;
      } else if (domainSel.value === "weinreich") {
        const d = param("detune"), sb = param("sigma_body");
        syHint.textContent = sb === 0
          ? "body loss = 0 → no bridge damping, nothing decays (the energy verdict is the drift "
            + "check). Raise it to load the symmetric mode and split the decay in two."
          : d < 0.005
            ? `unison: strike ONE string over a lossy bridge — the symmetric mode dies fast (prompt), `
              + `the antisymmetric mode is bit-exactly bridge-decoupled so it rings on LOSSLESS `
              + `(a dead-flat aftersound: the normal-mode oracle again). Detune → a finite aftersound.`
            : `~${(d * 100).toFixed(0)} cents mistuned: the antisymmetric mode now loads the bridge a `
              + `little, so the aftersound decays slowly (the realistic piano unison) instead of `
              + `flat. Body loss ${Math.round(sb)} s⁻¹ sets the prompt rate.`;
      } else {
        syHint.textContent = "normal modes: A & B plucked in ± antiphase keep the bridge exactly "
          + "still (w_b ≡ 0 bit-exact); in phase, it swings and loads the body. The zero is the "
          + "oracle energy cannot see.";
      }
      syHint.style.color = "var(--muted)";
    } else {
      syHint.textContent = "";
    }
  }
  // Jawari: surface the geometry gate LIVE, BEFORE the render. downswing/depth decides whether the
  // string wraps the curve or merely grazes its crest, and it is the one setting that can turn this
  // model into a different (legitimate, but not jawari) timbre without any error appearing. Showing
  // it as a live dimensionless number beats letting the user discover it in the readout 30 s later.
  const jHint = $("jawari-hint");
  if (jHint) {
    if (m === "jawari") {
      const ratio = (param("amplitude") * Math.PI * param("width_frac")) / param("depth");
      const s0 = param("sigma0");
      jHint.textContent = ratio < 1.5
        ? `downswing/depth ≈ ${ratio.toFixed(2)} — TOO DEEP: the string will only graze the crest `
          + `(a stiff point contact, no travelling wrap, no shimmer). Raise amplitude or lower depth.`
        : `downswing/depth ≈ ${ratio.toFixed(2)} — the swing clears the curve, so the string wraps `
          + `and its departure point travels. `
          + (s0 === 0
            ? `σ₀ = 0: lossless, so the verdict is conservation drift THROUGH the wrap — but nothing `
              + `decays, so "sustained" brightness has no meaning. Raise σ₀ for the shimmer.`
            : `σ₀ = ${s0.toFixed(1)}: a clean string would darken; the bridge re-injects highs.`);
      jHint.style.color = ratio < 1.5 ? "var(--warn, #ffcf5c)" : "var(--muted)";
    } else {
      jHint.textContent = "";
    }
  }
  // Bore: R/Z₀ is a log slider, so the hint carries the LINEAR ratio — the number the physics is
  // actually stated in — plus what that ratio means at the two landmarks (a real clarinet bell
  // barely leaks; the matched load absorbs everything and leaves no standing wave at all).
  const boreHint = $("bore-hint");
  if (boreHint) {
    if (m !== "bore") {
      boreHint.textContent = "";
    } else if (domainSel.value === "open") {
      boreHint.textContent = "ideal open end: pressure-release, r = −1. A perfect mirror — nothing "
        + "radiates, so the tube rings forever and the bell panel has nothing to score.";
      boreHint.style.color = "var(--muted)";
    } else {
      const ratio = Math.pow(10, param("bell_ratio_exp"));
      const shed = 0.5 * (1 - Math.pow((ratio - 1) / (ratio + 1), 2));
      boreHint.textContent = `R/Z₀ = ${ratio.toExponential(2)} → one bounce sheds `
        + `${(100 * shed).toFixed(1)} % of the incident energy. A real clarinet bell is ~3e-4 (a `
        + `slow leak, sharp resonances); R = Z₀ is ANECHOIC — the pulse simply vanishes at the `
        + `mouth and no standing wave forms, so the odd-harmonic claim stops applying.`;
      boreHint.style.color = ratio > 0.05 ? "var(--warn, #ffcf5c)" : "var(--muted)";
    }
  }
  // Mallet: surface the felt-resolution guard LIVE (the ctor's warning never reaches the browser).
  // The rigid-wall estimate π√(M/K)·fs must span several steps or the stiff contact aliases — a
  // NOTE, not an error, since the energy method conserves even under-resolved. Harder felt (↑K, ↓M)
  // shortens the contact and raises the pitch, but also eats the resolution, so the two trade off.
  const mHint = $("mallet-hint");
  if (mHint) {
    if (m === "mallet") {
      const c = Math.sqrt(param("T") / param("rho"));
      const h = domainSel.value === "circle" ? 2 * param("radius") / param("N")
                                             : param("Lx") / param("N");
      const fs = c / (param("lambda") * h);
      const spc = Math.PI * Math.sqrt(param("mass") / param("stiffness")) * fs;
      const under = spc < 8;
      mHint.textContent = under
        ? `~${spc.toFixed(0)} steps/contact (< 8) — the stiff felt aliases; raise λ→lower or soften `
          + `K. Energy still conserves.`
        : `~${spc.toFixed(0)} steps/contact (rigid-wall estimate; the yielding head stretches it ~20×). `
          + `Harder felt (↑K) → shorter, brighter strike.`;
      mHint.style.color = under ? "var(--bad)" : "var(--muted)";
    } else {
      mHint.textContent = "";
    }
  }
}

function onControlChange(name) {
  if (name === "lambda" || name === "mu" || name === "fs") updateLambdaHint();
  if (name === "amplitude" || name === "EA" || name === "T" || name === "L") updateLambdaHint();
  if (name === "sigma0" || name === "sigma1") updateLambdaHint();
  if (name === "bow_position" || name === "v_bow") updateLambdaHint();
  if (name === "lam_long" || name === "N" || name === "rho") updateLambdaHint();
  if (name === "tongue_position" || name === "dt_over_t0") updateLambdaHint();
  if (name === "kappa") updateLambdaHint();   // the phantom hint's microscope + cost estimate
  if (name === "K" || name === "detune" || name === "sigma_body") updateLambdaHint();  // symp hint
  scheduleAuto();
}

modelSel.addEventListener("change", () => {
  populateDomain(modelSel.value);
  applyModelRanges();
  updateVisibility();
  updateLambdaHint();
  scheduleAuto();
});
if (domainSel) domainSel.addEventListener("change", () => {
  if (hasRegimeRanges(modelSel.value)) applyModelRanges();
  updateVisibility();
  updateLambdaHint();
  scheduleAuto();
});
const nonlinearChk = $("nonlinear");
if (nonlinearChk) nonlinearChk.addEventListener("change", scheduleAuto);
const seedVelChk = $("seed-velocity");
if (seedVelChk) seedVelChk.addEventListener("change", () => { updateLambdaHint(); scheduleAuto(); });

function scheduleAuto() {
  if (!autoRender.checked) return;
  clearTimeout(autoTimer);
  setStatus("queued…", "busy");
  autoTimer = setTimeout(render, 400);
}

// ── networking ──────────────────────────────────────────────────────────────────────────────
function gatherParams() {
  const p = { model: modelSel.value };
  if (domainSel) p.domain = domainSel.value;
  for (const k in sliders) p[k] = +sliders[k].value * (scaleOf[k] || 1);
  if (nonlinearChk) p.nonlinear = nonlinearChk.checked;
  if (seedVelChk) p.seed_velocity = seedVelChk.checked;
  return p;
}

function setStatus(text, cls) {
  statusEl.textContent = text;
  statusEl.className = "status" + (cls ? " " + cls : "");
}

async function render() {
  clearTimeout(autoTimer);
  setStatus("computing…", "busy");
  renderBtn.disabled = true;
  try {
    const resp = await fetch("/simulate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(gatherParams()),
    });
    const data = await resp.json();
    if (data.error) {
      setStatus("error: " + data.error.message, "error");
      showOverlay("⚠ " + data.error.message);
      return;
    }
    applyPayload(data);
    const scheme = data.lam_long !== undefined ? `λ_long ${data.lam_long}`
      : data.lambda !== undefined ? `λ ${data.lambda}`
      : data.mu !== undefined ? `μ ${data.mu}`
      : (data.nonlinear === false ? "linear" : "nonlinear");
    // The geometric string ships no audio at all (audio: null) — see its audio_note. Reporting
    // "0 audio samples" would read as a bug; say what it is instead.
    const tail = data.audio ? `${data.audio.n} audio samples` : "viz-only (no audio)";
    setStatus(
      `ok — fs_sim ${fmt(data.fs_sim)} Hz · ${scheme} · ` +
      `${data.frames.n_frames} frames · ${tail}`, "");
  } catch (err) {
    setStatus("network error: " + err, "error");
  } finally {
    renderBtn.disabled = false;
  }
}

// ── payload handling ────────────────────────────────────────────────────────────────────────
function b64ToFloat32(b64) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Float32Array(bytes.buffer);
}

function b64ToUint8(b64) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return bytes;
}

function applyPayload(data) {
  payload = data;
  dims = (data.frames && data.frames.dims) || 1;
  frames = b64ToFloat32(data.frames.b64);
  nFrames = data.frames.n_frames;
  width = data.frames.width;
  fieldAmp = data.field_amp || 1;
  animDt = data.anim_dt || 1e-3;
  if (dims === 2) {
    gridNx = data.frames.nx; gridNy = data.frames.ny;
    maskData = data.mask ? b64ToUint8(data.mask.b64) : null;
    gridMeta = data.grid;
    heatCv = document.createElement("canvas");   // offscreen field-resolution buffer
    heatCv.width = gridNx; heatCv.height = gridNy;
  }
  isGeom = data.model === "geometric";
  isSymp = data.model === "sympathetic";
  isJawari = data.model === "jawari";
  isBore = data.model === "bore";
  if (isBore) {
    // The envelope is a running max|p(x)| over the FULL audio run, not a frame — it is what makes
    // the formed node/antinode structure legible, which no single instantaneous frame can show.
    // It shares fieldAmp with the polyline deliberately: drawn on its own scale it would stop
    // being an envelope OF the trace and become a second, unrelated curve.
    boreEnv = data.grid.envelope || null;
    boreEnds = (data.meta && data.meta.ends) || ["closed", "open"];
    // The mouth glow tracks the RATE energy leaves, not the cumulative total. The payload ships the
    // cumulative fraction (monotone, and the honest quantity for the energy panel), but a glow
    // driven by it would only ever brighten — a steady ramp that says nothing about WHEN sound
    // leaves. Differencing makes it pulse as each wavefront reaches the mouth, which is the thing
    // worth seeing. Normalized to the run's own peak rate so a 3e-4 bell is still visible.
    const rad = data.radiated_frames || [];
    boreRad = new Float32Array(rad.length);
    let peak = 0;
    for (let i = 1; i < rad.length; i++) {
      boreRad[i] = Math.max(0, rad[i] - rad[i - 1]);
      if (boreRad[i] > peak) peak = boreRad[i];
    }
    if (peak > 0) for (let i = 0; i < boreRad.length; i++) boreRad[i] /= peak;
  }
  if (isJawari) {
    // The bridge profile (NaN off its support) and the per-frame wrap edge, which is the marker
    // whose travel IS the second claim. Both are in the same units as the field, so they share
    // fieldAmp — drawing the barrier on its own scale would destroy the one thing the picture is
    // for: how far the string has swung PAST the curve.
    barrierProfile = data.grid.barrier || null;
    wrapFrames = data.wrap_frames || null;
  }
  if (isGeom) {
    orbitU = b64ToFloat32(data.orbit.u);
    orbitW = b64ToFloat32(data.orbit.w);
    orbitPerFrame = data.orbit.per_frame;
    // Scales are computed ONCE over the whole run, never per frame: a per-frame autoscale would
    // renormalize the whirl's growth away — the picture would look identical at every instant while
    // max|w| climbed 60×. u and w SHARE a scale (both transverse, and their ratio is exactly the
    // claim: equal ⇒ a circle, 1e-4 ⇒ a line); v gets its own, because it is a different physical
    // quantity (longitudinal stretch, orders smaller) and would otherwise be a flat line at 0.
    uwAmp = 0; vAmp = 0;
    for (let f = 0; f < nFrames; f++) {
      for (let i = 0; i < width; i++) {
        const u = Math.abs(frames[(f * 3 + 0) * width + i]);
        const w = Math.abs(frames[(f * 3 + 1) * width + i]);
        const v = Math.abs(frames[(f * 3 + 2) * width + i]);
        if (u > uwAmp) uwAmp = u;
        if (w > uwAmp) uwAmp = w;
        if (v > vAmp) vAmp = v;
      }
    }
    nFields = 3;
    fieldAmps = [uwAmp, uwAmp, vAmp];
    fieldLabels = ["u — transverse", "w — out of plane", "v — longitudinal"];
  } else if (isSymp) {
    // J strings, all the SAME quantity (transverse displacement), so they SHARE one scale computed
    // over the whole run — one string ringing up while the other rings down is the picture, and a
    // per-strip autoscale would flatten it away. Field names/count come from the payload.
    nFields = data.frames.fields.length;
    fieldLabels = data.frames.field_labels || data.frames.fields;
    let amp = 0;
    for (let k = 0; k < frames.length; k++) { const a = Math.abs(frames[k]); if (a > amp) amp = a; }
    fieldAmps = new Array(nFields).fill(amp || 1);
  }
  audioSamples = data.audio ? b64ToFloat32(data.audio.b64) : null;
  audioFs = data.audio ? data.audio.fs : 48000;
  audioBuf = null;                       // rebuilt lazily on first Play
  updateAudioTransport(data);
  currentFrame = 0; animStart = 0;
  scrub.max = Math.max(0, nFrames - 1);
  scrub.value = 0;
  hideOverlay();
  drawEnergy();
  drawDiagnostics();
}

// A model with no audio gets no player, and the note says WHY rather than leaving a dead button.
function updateAudioTransport(data) {
  const has = !!data.audio;
  playAudioBtn.hidden = !has;
  const lbl = $("loop-audio-label");
  if (lbl) lbl.hidden = !has;
  const note = $("audio-note");
  if (note) {
    note.hidden = has;
    note.textContent = data.audio_note || "";
  }
}

// ── string animation ────────────────────────────────────────────────────────────────────────
function drawString(idx) {
  const g = stringCv.getContext("2d");
  const W = stringCv.width, H = stringCv.height, midY = H / 2, margin = 22;
  g.clearRect(0, 0, W, H);
  g.strokeStyle = "#2a3340"; g.lineWidth = 1;
  g.beginPath(); g.moveTo(0, midY); g.lineTo(W, midY); g.stroke();
  if (!frames || nFrames === 0) return;

  const amp = fieldAmp > 0 ? fieldAmp : 1;
  const sx = (W - 2 * margin) / (width - 1);
  const sy = (H / 2 - margin) / amp * 0.92;
  const base = idx * width;

  // pickup marker
  if (payload) {
    const px = margin + Math.round(param("pickup_position") * (width - 1)) * sx;
    g.strokeStyle = "rgba(255,207,92,.35)"; g.setLineDash([4, 4]); g.lineWidth = 1;
    g.beginPath(); g.moveTo(px, 8); g.lineTo(px, H - 8); g.stroke(); g.setLineDash([]);
  }

  g.strokeStyle = "#4cc2ff"; g.lineWidth = 2.5; g.lineJoin = "round";
  g.beginPath();
  for (let i = 0; i < width; i++) {
    const x = margin + i * sx;
    const y = midY - frames[base + i] * sy;
    if (i === 0) g.moveTo(x, y); else g.lineTo(x, y);
  }
  g.stroke();

  // endpoints
  g.fillStyle = "#8b98a8";
  g.beginPath(); g.arc(margin, midY, 3.5, 0, 7); g.fill();
  g.beginPath(); g.arc(W - margin, midY, 3.5, 0, 7); g.fill();
}

// ── jawari: the string over its bridge, plus a zoom on the wrap ──────────────────────────────
// Two views because one cannot carry both halves. The bridge spans ~15 % of the string and its
// curve drops ~1 mm against an 8 mm swing, so in the full view it is a few pixels of detail near
// the termination — you can see THAT the string is held off the rest line but not that it is lying
// along a curve. The zoom (right) is the money picture: the string conforming to the parabola over
// a span that grows and shrinks, with the departure point sliding. Same trap as the phantom
// batch's two spectra, one level over.
function drawJawariViz(idx) {
  const g = stringCv.getContext("2d");
  const W = stringCv.width, H = stringCv.height;
  g.clearRect(0, 0, W, H);
  const fullW = Math.round(W * 0.58);
  drawJawariPane(g, idx, 0, 0, fullW, H, false);
  drawJawariPane(g, idx, fullW, 0, W - fullW, H, true);
}

// One pane. `zoom` restricts the x-range to the bridge span (plus a margin) and rescales y to the
// bridge depth, so the curve fills the box instead of hugging the axis.
function drawJawariPane(g, idx, x0, y0, w, h, zoom) {
  const sp = payload && payload.meta && payload.meta.spectrum;
  const xs = (payload && payload.grid && payload.grid.x) || [];
  if (!frames || nFrames === 0 || !xs.length) return;
  const span = (payload.meta && payload.meta.bridge_span) || 0.15;
  const Lx = xs[xs.length - 1] || 1;
  const margin = 20;

  // Node window. Zoomed: the bridge plus half again, so the string is seen ARRIVING at the curve.
  const iMax = zoom ? Math.min(width - 1, Math.ceil((span * 1.6 / Lx) * (width - 1))) : width - 1;
  const amp = fieldAmp > 0 ? fieldAmp : 1;
  // Zoomed, the vertical scale follows the BRIDGE depth, not the pluck amplitude: at the full
  // scale a 1 mm curve under an 8 mm swing is ~4 px of the box and the wrap is invisible.
  let vAmpLocal = amp;
  if (zoom && sp && sp.depth) vAmpLocal = Math.max(sp.depth * 3.0, amp * 0.22);
  const midY = y0 + h * (zoom ? 0.42 : 0.5);
  const sx = (w - 2 * margin) / iMax;
  const sy = (Math.min(midY - y0, y0 + h - midY) - margin) / vAmpLocal * 0.92;
  const px = (i) => x0 + margin + i * sx;
  const py = (v) => midY - v * sy;

  g.save();
  g.beginPath(); g.rect(x0, y0, w, h); g.clip();

  g.strokeStyle = "#2a3340"; g.lineWidth = 1;
  g.beginPath(); g.moveTo(x0, midY); g.lineTo(x0 + w, midY); g.stroke();

  // The bridge itself: a filled body below the curve reads as a solid the string rests ON, where a
  // bare line reads as just another trace.
  if (barrierProfile) {
    const pts = [];
    for (let i = 0; i <= iMax; i++) {
      const b = barrierProfile[i];
      if (b !== null && b !== undefined && isFinite(b)) pts.push([px(i), py(b)]);
    }
    if (pts.length > 1) {
      g.beginPath();
      g.moveTo(pts[0][0], pts[0][1]);
      for (const [X, Y] of pts) g.lineTo(X, Y);
      g.lineTo(pts[pts.length - 1][0], y0 + h);
      g.lineTo(pts[0][0], y0 + h);
      g.closePath();
      g.fillStyle = "rgba(198,142,86,.20)"; g.fill();
      g.strokeStyle = "#c68e56"; g.lineWidth = 2;
      g.beginPath();
      g.moveTo(pts[0][0], pts[0][1]);
      for (const [X, Y] of pts) g.lineTo(X, Y);
      g.stroke();
    }
  }

  // The string.
  g.strokeStyle = "#4cc2ff"; g.lineWidth = zoom ? 2.5 : 2; g.lineJoin = "round";
  g.beginPath();
  const base = idx * width;
  for (let i = 0; i <= iMax; i++) {
    const Y = py(frames[base + i]);
    if (i === 0) g.moveTo(px(i), Y); else g.lineTo(px(i), Y);
  }
  g.stroke();

  // The departure point — the furthest-in-contact node. -1 means the string is clear of the bridge
  // this frame, and drawing nothing then is deliberate: the marker's absence is the intermittency.
  const we = wrapFrames && idx < wrapFrames.length ? wrapFrames[idx] : -1;
  if (we >= 0 && we <= iMax) {
    const bx = px(we), by = py(barrierProfile ? barrierProfile[we] : 0);
    g.fillStyle = "#ff6b6b";
    g.beginPath(); g.arc(bx, by, zoom ? 5 : 3.5, 0, 7); g.fill();
    if (zoom) {
      g.strokeStyle = "rgba(255,107,107,.45)"; g.lineWidth = 1; g.setLineDash([3, 3]);
      g.beginPath(); g.moveTo(bx, y0 + 6); g.lineTo(bx, y0 + h - 6); g.stroke(); g.setLineDash([]);
    }
  }

  if (!zoom) {
    const pk = margin + Math.round(param("pickup_position") * (width - 1)) * sx;
    g.strokeStyle = "rgba(255,207,92,.35)"; g.setLineDash([4, 4]); g.lineWidth = 1;
    g.beginPath(); g.moveTo(x0 + pk, y0 + 8); g.lineTo(x0 + pk, y0 + h - 8); g.stroke();
    g.setLineDash([]);
    g.fillStyle = "#8b98a8";
    g.beginPath(); g.arc(px(0), midY, 3.5, 0, 7); g.fill();
    g.beginPath(); g.arc(px(iMax), midY, 3.5, 0, 7); g.fill();
  }

  g.fillStyle = "#8b98a8"; g.font = "11px system-ui, sans-serif";
  g.fillText(zoom ? `zoom — the bridge (0 … ${(span * 1.6).toFixed(2)} m)`
                  : "the whole string", x0 + margin, y0 + 14);
  if (zoom && we >= 0) {
    g.fillStyle = "#ff6b6b";
    g.fillText(`departure node ${we}`, x0 + margin, y0 + 28);
  }
  g.restore();
}

// ── acoustic bore: pressure down the tube, and two ends that are NOT alike ────────────────────
// Its own path rather than drawString, and the forcing reason is CORRECTNESS, not precedent.
// drawString pins BOTH endpoints to the rest line. The bore's closed end is a pressure ANTINODE
// (p free and large); only the open end is a node (p = 0). That asymmetry IS the odd-harmonic
// claim, so drawString would not merely look wrong — it would draw the batch's own physics
// backwards. The polyline/margin/scale arithmetic they share is ~8 lines; duplicating it is far
// clearer than parameterizing drawString with five flags.
function drawBore(idx) {
  const g = stringCv.getContext("2d");
  const W = stringCv.width, H = stringCv.height, midY = H / 2;
  g.clearRect(0, 0, W, H);
  if (!frames || nFrames === 0) return;

  const amp = fieldAmp > 0 ? fieldAmp : 1;
  // Margins are ASYMMETRIC, per end. The bell's flare and its glow are drawn OUTWARD past the
  // mouth, so an end that radiates needs room beyond the tube or the one element that shows energy
  // leaving is clipped by the canvas. Walls likewise sit well inside the panel: pressure is scaled
  // to fill the bore, not the whole canvas.
  const room = (kind) => (kind === "radiating" ? 76 : 34);
  const mL = room(boreEnds[0]), mR = room(boreEnds[1]);
  const wall = Math.round(H * 0.32);
  const sx = (W - mL - mR) / (width - 1);
  const sy = (wall * 0.92) / amp;
  const px = (i) => mL + i * sx;
  const py = (p) => midY - p * sy;
  const base = idx * width;
  const xL = px(0), xR = px(width - 1);

  g.strokeStyle = "#2a3340"; g.lineWidth = 1;
  g.beginPath(); g.moveTo(0, midY); g.lineTo(W, midY); g.stroke();
  g.strokeStyle = "#55647a"; g.lineWidth = 2.5;
  for (const s of [-1, 1]) {
    g.beginPath(); g.moveTo(xL, midY + s * wall); g.lineTo(xR, midY + s * wall); g.stroke();
  }

  // The envelope: max|p(x)| over the WHOLE run, drawn as a static band. This is the only thing on
  // screen that shows the FORMED standing wave — its nodes and antinodes are a property of the
  // run, not of any instant, so no single frame can carry them. Honest caveat, printed below: at
  // the anechoic R = Z₀ there IS no standing wave and the band correctly degrades to the trace of
  // the pulse's single pass.
  if (boreEnv) {
    g.fillStyle = "rgba(76,194,255,.07)";
    g.beginPath();
    for (let i = 0; i < boreEnv.length; i++) g.lineTo(px(i), py(boreEnv[i]));
    for (let i = boreEnv.length - 1; i >= 0; i--) g.lineTo(px(i), py(-boreEnv[i]));
    g.closePath(); g.fill();
    // Outlined as well as filled, and dashed: a bare fill at this alpha is indistinguishable from
    // the tube's interior, so the envelope stops reading as a measured curve and starts reading as
    // decoration — which is the opposite of its job.
    g.strokeStyle = "rgba(76,194,255,.35)"; g.lineWidth = 1; g.setLineDash([5, 4]);
    for (const s of [-1, 1]) {
      g.beginPath();
      for (let i = 0; i < boreEnv.length; i++) g.lineTo(px(i), py(s * boreEnv[i]));
      g.stroke();
    }
    g.setLineDash([]);
  }

  // pickup marker
  if (payload) {
    const pk = px(Math.round(param("pickup_position") * (width - 1)));
    g.strokeStyle = "rgba(255,207,92,.35)"; g.setLineDash([4, 4]); g.lineWidth = 1;
    g.beginPath(); g.moveTo(pk, midY - wall); g.lineTo(pk, midY + wall); g.stroke();
    g.setLineDash([]);
  }

  g.strokeStyle = "#4cc2ff"; g.lineWidth = 2.5; g.lineJoin = "round";
  g.beginPath();
  for (let i = 0; i < width; i++) {
    const x = px(i), y = py(frames[base + i]);
    if (i === 0) g.moveTo(x, y); else g.lineTo(x, y);
  }
  g.stroke();

  const glow = boreRad && idx < boreRad.length ? boreRad[idx] : 0;
  drawBoreEnd(g, boreEnds[0], xL, midY, wall, -1, glow);
  drawBoreEnd(g, boreEnds[1], xR, midY, wall, +1, glow);

  g.fillStyle = "#8b98a8"; g.font = "11px system-ui, sans-serif";
  g.fillText(`p(x) — pressure along the tube (${boreEnds[0]} → ${boreEnds[1]})`, mL, 14);
  g.fillText("dashed: max|p| over the whole run — the standing wave", mL, H - 8);
}

// One end of the tube. Switching on the END TYPE, never hardcoding it, is what makes this the viz
// batch 10 reuses: the reed IS a bore with a new end type at the mouth, so "reed" becomes another
// case here rather than a rewrite. `dir` is +1 for the end that faces right (outward = +x).
function drawBoreEnd(g, kind, x, midY, wall, dir, glow) {
  if (kind === "closed") {
    // A rigid wall: U = 0, so pressure is FREE and largest here. Drawn as a solid hatched block —
    // you cannot pass — and deliberately not as a pinned node.
    g.fillStyle = "#2a3340";
    g.fillRect(x - (dir > 0 ? 0 : 9), midY - wall, 9, 2 * wall);
    g.strokeStyle = "#4a5768"; g.lineWidth = 1;
    for (let y = midY - wall; y < midY + wall; y += 6) {
      g.beginPath(); g.moveTo(x - (dir > 0 ? 0 : 9), y + 6); g.lineTo(x + (dir > 0 ? 9 : 0), y);
      g.stroke();
    }
    return;
  }
  if (kind === "open") {
    // Pressure-release: p = 0 exactly. The tube simply stops and a dashed vertical marks the node.
    g.strokeStyle = "rgba(139,152,168,.7)"; g.lineWidth = 1.5; g.setLineDash([3, 3]);
    g.beginPath(); g.moveTo(x, midY - wall); g.lineTo(x, midY + wall); g.stroke();
    g.setLineDash([]);
    return;
  }
  // Radiating bell: a flared mouth plus an outward glow whose brightness tracks the BOOKED
  // radiated power — the field-side dual of the energy panel's radiated curve, so energy leaving
  // is SEEN leaving. At the matched R = Z₀ the pulse reaches the mouth and simply vanishes: the
  // anechoic null, visible.
  const flare = 16 * dir;
  g.strokeStyle = "#7fd4a0"; g.lineWidth = 2.5;
  for (const s of [-1, 1]) {
    g.beginPath();
    g.moveTo(x, midY + s * wall);
    g.quadraticCurveTo(x + flare * 0.6, midY + s * wall, x + flare, midY + s * (wall + 12));
    g.stroke();
  }
  const a = Math.min(1, Math.max(0, glow));
  if (a > 0.01) {
    const cx = x + flare, R = wall * 1.15;
    const gr = g.createRadialGradient(cx, midY, 2, cx, midY, R);
    gr.addColorStop(0, `rgba(127,212,160,${(0.45 * a).toFixed(3)})`);
    gr.addColorStop(1, "rgba(127,212,160,0)");
    g.save();
    // Clipped to the OUTWARD side of the mouth: a full disc would bleed back down the tube and
    // read as pressure inside it, which is exactly what the glow is not.
    g.beginPath();
    g.rect(dir > 0 ? cx : cx - R, midY - R, R, 2 * R);
    g.clip();
    g.fillStyle = gr;
    g.beginPath(); g.arc(cx, midY, R, 0, 7); g.fill();
    g.restore();
  }
}

function tick(ts) {
  if (frames && nFrames > 0) {
    if (animPlaying && !scrubbing) {
      if (animStart === 0) animStart = ts - (currentFrame * animDt / speed) * 1000;
      const physElapsed = ((ts - animStart) / 1000) * speed;
      currentFrame = Math.floor(physElapsed / animDt) % nFrames;
      scrub.value = currentFrame;
    }
    (dims === 2 ? drawHeatmap : isGeom ? drawGeometric
      : isSymp ? drawSympatheticViz : isJawari ? drawJawariViz
        : isBore ? drawBore : drawString)(currentFrame);
  }
  requestAnimationFrame(tick);
}

// ── geometrically-exact string: the orbit + the three fields ─────────────────────────────────
// The plot model #9 structurally cannot draw. It has ONE polarization, so its only orbit is a point
// on a line; here the cross-section traces a real curve — a line (planar), a circle (the rotating
// wave) or a slowly-opening sliver (whirling). Left panel: the (u, w) orbit at the probe node, on
// EQUAL axes. Right: u(x), w(x), v(x) down the string.
function drawGeometric(idx) {
  const g = stringCv.getContext("2d");
  const W = stringCv.width, H = stringCv.height;
  g.clearRect(0, 0, W, H);
  const orbW = Math.min(H, Math.round(W * 0.42));
  drawOrbit(g, idx, 0, 0, orbW, H);
  drawFields(g, idx, orbW, 0, W - orbW, H);
}

function drawOrbit(g, idx, x0, y0, w, h) {
  const cx = x0 + w / 2, cy = y0 + h / 2, margin = 26;
  const R = Math.min(w, h) / 2 - margin;
  // EQUAL axes, always. Scaling w up to fill the box would manufacture a circle out of a whirl's
  // 1e-4 sliver — the orbit's whole job is to show the true aspect, and the envelope panel carries
  // the growth story instead.
  const s = R / (uwAmp > 0 ? uwAmp : 1);

  g.strokeStyle = "#2a3340"; g.lineWidth = 1;
  g.beginPath(); g.moveTo(cx - R, cy); g.lineTo(cx + R, cy); g.stroke();
  g.beginPath(); g.moveTo(cx, cy - R); g.lineTo(cx, cy + R); g.stroke();

  g.fillStyle = "#8b98a8"; g.font = "11px ui-monospace, monospace";
  g.fillText("u", cx + R + 4, cy + 4);
  g.fillText("w", cx - 4, cy - R - 6);

  if (!orbitU || orbitU.length === 0) return;
  const upto = Math.max(2, Math.min(orbitU.length, Math.round((idx + 1) * orbitPerFrame)));
  // The trail ACCUMULATES as the animation plays: the curve draws itself, which is what makes a
  // circle read as a circle (a single moving dot reads as noise).
  g.strokeStyle = "rgba(76,194,255,.55)"; g.lineWidth = 1.4; g.lineJoin = "round";
  g.beginPath();
  for (let i = 0; i < upto; i++) {
    const px = cx + orbitU[i] * s, py = cy - orbitW[i] * s;
    if (i === 0) g.moveTo(px, py); else g.lineTo(px, py);
  }
  g.stroke();
  g.fillStyle = "#ffcf5c";
  g.beginPath(); g.arc(cx + orbitU[upto - 1] * s, cy - orbitW[upto - 1] * s, 3.5, 0, 7); g.fill();

  const sp = payload && payload.meta ? payload.meta.spectrum : null;
  g.fillStyle = "#8b98a8"; g.font = "11px ui-monospace, monospace";
  g.fillText(`orbit at x = ${payload.meta.probe_x} m · equal axes`, x0 + 10, y0 + h - 10);
  if (sp && sp.kind === "whirl") {
    g.fillText(`max|w|/max|u| = ${sp.w_over_u.toExponential(1)}`, x0 + 10, y0 + h - 24);
  }
}

// Stacked strips, one per field. Generalized from the geometric string's fixed u/w/v to nFields:
// geometric ships 3 (u/w share a scale, v its own — see fieldAmps at load); sympathetic ships J = 2
// strings all sharing one scale (they are the same physical quantity, and one string ringing up
// while the other rings down is the whole picture, so a per-strip autoscale would hide it). Names,
// colours and per-field amps are module state set once in applyPayload, so the stride and layout
// are the only per-frame work here.
function drawFields(g, idx, x0, y0, w, h) {
  const stripH = h / nFields, margin = 20;
  for (let f = 0; f < nFields; f++) {
    const midY = y0 + stripH * f + stripH / 2;
    const amp = fieldAmps[f];
    const sy = (stripH / 2 - 12) / (amp > 0 ? amp : 1);
    const sx = (w - 2 * margin) / (width - 1);
    g.strokeStyle = "#2a3340"; g.lineWidth = 1;
    g.beginPath(); g.moveTo(x0 + margin, midY); g.lineTo(x0 + w - margin, midY); g.stroke();

    g.strokeStyle = FIELD_COLORS[f % FIELD_COLORS.length]; g.lineWidth = 2; g.lineJoin = "round";
    g.beginPath();
    for (let i = 0; i < width; i++) {
      const px = x0 + margin + i * sx;
      const py = midY - frames[(idx * nFields + f) * width + i] * sy;
      if (i === 0) g.moveTo(px, py); else g.lineTo(px, py);
    }
    g.stroke();
    g.fillStyle = "#8b98a8"; g.font = "11px ui-monospace, monospace";
    // Each strip prints its OWN scale (the geometric strips do not all share one).
    g.fillText(`${fieldLabels[f]}  ±${amp.toExponential(1)} m`, x0 + margin, y0 + stripH * f + 13);
  }
}

// ── sympathetic / coupled strings: the J string fields, stacked ──────────────────────────────
// No orbit (unlike the geometric string): the shared bridge displacement w_b goes in the second
// panel, not a cross-section trail, so the strings get the full canvas width. In the normal-mode
// animation the two strips are exact mirror images (string B = −string A over a dead bridge); in
// transfer, string A rings down as string B rings up.
function drawSympatheticViz(idx) {
  const g = stringCv.getContext("2d");
  const W = stringCv.width, H = stringCv.height;
  g.clearRect(0, 0, W, H);
  if (!frames || nFrames === 0) return;
  drawFields(g, idx, 0, 0, W, H);
}

// ── membrane heatmap ─────────────────────────────────────────────────────────────────────────
// Diverging colormap centred at 0: cool (cyan/blue) for displacement < 0, warm (orange/red) for
// > 0, near-black at rest. t is signed displacement / field_amp, clamped to [-1, 1].
function divColor(t) {
  const a = Math.min(1, Math.abs(t));
  if (t >= 0) return [20 + a * 235, 24 + a * 96, 30 + a * 6];     // dark → orange/red
  return [20 + a * 24, 24 + a * 162, 30 + a * 225];               // dark → cyan/blue
}

function drawHeatmap(idx) {
  const g = stringCv.getContext("2d");
  const W = stringCv.width, H = stringCv.height;
  g.clearRect(0, 0, W, H);
  if (!frames || nFrames === 0 || !heatCv) return;

  // Paint the decimated field into the offscreen buffer (one device pixel per node), masking
  // the exterior to the panel background so the domain shape (incl. the staircased rim) reads.
  const hctx = heatCv.getContext("2d");
  const img = hctx.createImageData(gridNx, gridNy);
  const amp = fieldAmp > 0 ? fieldAmp : 1;
  const base = idx * gridNx * gridNy;
  for (let p = 0; p < gridNx * gridNy; p++) {
    const o = p * 4;
    if (maskData && maskData[p] === 0) {            // outside the membrane
      img.data[o] = 22; img.data[o + 1] = 27; img.data[o + 2] = 34; img.data[o + 3] = 255;
      continue;
    }
    const c = divColor(frames[base + p] / amp);
    img.data[o] = c[0]; img.data[o + 1] = c[1]; img.data[o + 2] = c[2]; img.data[o + 3] = 255;
  }
  hctx.putImageData(img, 0, 0);

  // Blit to the main canvas preserving the physical aspect ratio (snapped Ly for a rectangle).
  const extX = (gridMeta && gridMeta.extent_x) || 1, extY = (gridMeta && gridMeta.extent_y) || 1;
  const pad = 14, availW = W - 2 * pad, availH = H - 2 * pad;
  const scale = Math.min(availW / extX, availH / extY);
  const dw = extX * scale, dh = extY * scale;
  const dx = (W - dw) / 2, dy = (H - dh) / 2;
  g.imageSmoothingEnabled = true;
  g.drawImage(heatCv, dx, dy, dw, dh);

  // Strike marker (mallet only): the SNAPPED felt-contact node, drawn as a filled dot so it reads
  // distinctly from the hollow pickup cross. Coords come from the payload (the ctor snaps to the
  // nearest live node), not the raw slider, so the dot sits exactly where the felt actually landed.
  const sp = payload && payload.meta && payload.meta.spectrum;
  if (sp && sp.kind === "mallet" && sp.strike_fx !== undefined) {
    const sx = dx + sp.strike_fx * dw, sy = dy + sp.strike_fy * dh;
    g.fillStyle = "rgba(255,95,109,.95)";
    g.beginPath(); g.arc(sx, sy, 5.5, 0, 7); g.fill();
    g.strokeStyle = "rgba(255,95,109,.55)"; g.lineWidth = 1.5;
    g.beginPath(); g.arc(sx, sy, 9, 0, 7); g.stroke();
  }

  // Pickup marker (x, y) in domain fractions → screen.
  const pkx = param("pickup_x"), pky = param("pickup_y");
  if (pkx !== undefined && pky !== undefined) {
    const mx = dx + pkx * dw, my = dy + pky * dh;
    g.strokeStyle = "rgba(255,207,92,.9)"; g.lineWidth = 1.5;
    g.beginPath(); g.arc(mx, my, 5, 0, 7); g.stroke();
    g.beginPath(); g.moveTo(mx - 8, my); g.lineTo(mx + 8, my);
    g.moveTo(mx, my - 8); g.lineTo(mx, my + 8); g.stroke();
  }
}

// ── energy diagnostic ───────────────────────────────────────────────────────────────────────
// An actively DRIVEN model (the bow) reports an energy BALANCE — a third verdict type, and it
// replaces both branches below rather than joining them, because for a driven model both are
// actively wrong, not merely weaker: at σ=0 the bow pumps energy in, so "drift" is enormous by
// design and would read as a catastrophic FAIL; at σ>0 the energy RISES from rest to the Helmholtz
// limit cycle, so "monotone decrease" fails too. Either would paint a red badge on a correct run.
// Dispatched before both, like the von Kármán convergence override.
function drawBalance() {
  const g = energyCv.getContext("2d");
  const W = energyCv.width, H = energyCv.height, pad = 24;
  const e = payload.energy, b = e.balance;
  const t = e.time;
  const num = (a) => a.map((x) => (x == null ? 0 : x));
  const dE = num(b.delta_energy), work = num(b.work), diss = num(b.dissipation);
  const tmax = t[t.length - 1] || 1;
  const vmax = Math.max(1e-30, ...dE.map(Math.abs), ...work.map(Math.abs), ...diss.map(Math.abs));

  g.strokeStyle = "#2a3340"; g.lineWidth = 1;
  g.strokeRect(pad, 8, W - pad - 8, H - pad - 8);
  const plot = (arr, colour, width) => {
    g.strokeStyle = colour; g.lineWidth = width; g.beginPath();
    for (let i = 0; i < arr.length; i++) {
      const x = pad + (t[i] / tmax) * (W - pad - 8);
      const y = (H - pad) - (arr[i] / vmax) * (H - pad - 8);
      if (i === 0) g.moveTo(x, y); else g.lineTo(x, y);
    }
    g.stroke();
  };
  // Bow work drawn thick underneath, ΔE thin on top: at σ=0 the money test IS the visual — the two
  // are one curve, every joule the bow put in is sitting in the string.
  plot(work, "#e0b050", 3.5);
  plot(diss, "#6aa9e0", 1.5);
  plot(dE, "#5ad17a", 1.5);

  g.font = "10px ui-monospace, monospace";
  const key = [["bow work", "#e0b050"], ["E−E₀", "#5ad17a"], ["loss", "#6aa9e0"]];
  key.forEach(([label, colour], i) => {
    g.fillStyle = colour;
    g.fillRect(pad + 6 + i * 68, 14, 8, 3);
    g.fillText(label, pad + 18 + i * 68, 18);
  });
  g.fillStyle = "#8b98a8";
  g.fillText(`${tmax.toFixed(2)} s`, W - 48, H - 8);

  const badge = $("energy-verdict"), out = $("energy-readout");
  if (e.sigma_is_zero) {
    const ok = b.lossless.pass;
    badge.textContent = ok ? "balanced" : "IMBALANCE";
    badge.className = "badge " + (ok ? "good" : "bad");
    out.textContent =
      `lossless · E−E⁰ == bow work · residual ${b.lossless.residual.toExponential(2)}\n` +
      `tol ${b.lossless.tol.toExponential(0)}  →  ${ok ? "PASS ✓" : "FAIL ✗"}` +
      `  (the tone grows without bound — that is the price of exact closure)`;
  } else {
    // NOT a residual here, deliberately: with loss on, dissipation is never measured — it is
    // INFERRED as bow_work − ΔE, so a "balance residual" would be identically zero by construction,
    // a green tick that cannot fail. What is real is that the inferred loss only ever removes.
    const ok = b.lossy.pass;
    badge.textContent = ok ? "passive" : "NON-PASSIVE";
    badge.className = "badge " + (ok ? "good" : "bad");
    out.textContent =
      `lossy · bow work ${b.work_total.toExponential(2)} J = stored + loss ` +
      `${b.lossy.dissipation_total.toExponential(2)} J\n` +
      `inferred loss ≥ 0: ${b.lossy.non_negative ? "yes ✓" : "NO ✗"}  ·  ` +
      `never adds energy: ${b.lossy.monotone ? "yes ✓" : "NO ✗"}  →  ${ok ? "PASS ✓" : "FAIL ✗"}`;
  }
}

function drawEnergy() {
  const g = energyCv.getContext("2d");
  const W = energyCv.width, H = energyCv.height, pad = 24;
  g.clearRect(0, 0, W, H);
  if (!payload) return;
  const e = payload.energy;
  if (e.kind === "balance") { drawBalance(); return; }
  const t = e.time, v = e.value.map((x) => (x == null ? 0 : x));
  const tmax = t[t.length - 1] || 1;
  // Headroom only when the split is drawn: the total is flat AT the maximum, so without it the
  // conserving curve lies exactly on the frame's top edge and reads as part of the box rather than
  // as the result. Left untouched otherwise so every other model's panel stays pixel-identical.
  const vmax = (Math.max(...v) || 1) * (payload.energy.split ? 1.12 : 1);

  g.strokeStyle = "#2a3340"; g.lineWidth = 1;
  g.strokeRect(pad, 8, W - pad - 8, H - pad - 8);
  const curve = (arr, colour, wdt) => {
    g.strokeStyle = colour; g.lineWidth = wdt; g.beginPath();
    for (let i = 0; i < arr.length; i++) {
      const x = pad + (t[i] / tmax) * (W - pad - 8);
      const y = (H - pad) - ((arr[i] == null ? 0 : arr[i]) / vmax) * (H - pad - 8);
      if (i === 0) g.moveTo(x, y); else g.lineTo(x, y);
    }
    g.stroke();
  };
  // The SPLIT (the bore). Without it a flat green "conserved, drift 1e-15 ✓" sits beside a pickup
  // that audibly decays — because acoustic energy is leaving to the radiated channel — and the
  // panel reads as self-contradictory, hiding the exact physics the model is about. With it: the
  // acoustic curve falls, the radiated curve rises, and the total stays flat. That is "watch the
  // sound leave the tube," and it is what makes R/Z₀ legible as a control.
  if (e.split) {
    curve(e.split.acoustic, "#4cc2ff", 1.5);
    curve(e.split.radiated, "#7fd4a0", 1.5);
  }
  curve(v, "#5ad17a", 2);
  g.fillStyle = "#8b98a8"; g.font = "10px ui-monospace, monospace";
  g.fillText("E", 6, 16); g.fillText(`${tmax.toFixed(2)} s`, W - 48, H - 8);
  if (e.split) {
    [["total", "#5ad17a"], ["acoustic", "#4cc2ff"], ["radiated", "#7fd4a0"]]
      .forEach(([label, colour], i) => {
        g.fillStyle = colour;
        g.fillRect(pad + 6 + i * 66, 14, 8, 3);
        g.fillText(label, pad + 18 + i * 66, 18);
      });
  }

  const badge = $("energy-verdict"), out = $("energy-readout");
  // von Kármán convergence gate (catch): the energy identity telescopes ONLY at the Picard fixed
  // point, so a run with any non-converged step has a drift number that is iteration noise, not
  // physics. Overrides the pass/fail verdict in BOTH the lossless and lossy branches.
  // The same gate serves the tension string, whose per-step scalar root-find is a different solver
  // with different telemetry — so a model may supply its own `detail`/`note` wording. Absent those,
  // the von Kármán Picard wording stands unchanged.
  const conv = e.convergence;
  if (conv && !conv.all_converged) {
    badge.textContent = "NOT CONVERGED";
    badge.className = "badge bad";
    out.textContent = conv.detail ||
      (`Picard did not converge: ${conv.n_not_converged} step(s), worst residual ` +
       `${conv.worst_residual.toExponential(1)} > tol ${conv.couple_tol.toExponential(0)}\n` +
       `energy verdict N/A — lower the strike amplitude (w/e) or raise fs`);
    return;
  }
  const convNote = conv ? (conv.note || `  ·  Picard converged (≤ ${conv.max_iters} sweeps)`) : "";
  if (e.sigma_is_zero) {
    const ok = e.lossless.pass;
    badge.textContent = ok ? "conserved" : "DRIFT";
    badge.className = "badge " + (ok ? "good" : "bad");
    // The whirl needs NO new verdict type (unlike the bow's balance) — it is the ordinary lossless
    // drift check. What makes it a claim is WHAT it survives: a parametric instability is energy
    // REDISTRIBUTION between polarizations, so a correct scheme conserves straight through the
    // blow-up. That is exactly what separates a whirl from a diverging solve, the other thing that
    // makes |w| grow orders of magnitude — so the growth belongs next to the drift, not elsewhere.
    const sp = payload && payload.meta ? payload.meta.spectrum : null;
    const through = (sp && sp.kind === "whirl" && sp.seeded && sp.growth > 2)
      ? `  ·  THROUGH a ${sp.growth.toFixed(0)}× |w| blow-up` : "";
    out.textContent =
      `lossless · drift max|Eⁿ−E⁰|/E⁰ = ${e.lossless.drift.toExponential(2)}${convNote}${through}\n` +
      `tol ${e.lossless.tol.toExponential(0)}  →  ${ok ? "PASS ✓" : "FAIL ✗"}`;
  } else {
    const mono = e.lossy.monotone;
    badge.textContent = mono ? "passive" : "NON-MONOTONE";
    badge.className = "badge " + (mono ? "good" : "bad");
    // The 2σ decay oracle is dropped for a CLOSED driven-mass model (the mallet): its total energy
    // sits on a near-constant ½M·v₀² floor once the felt separates, so a fitted "2σ" would be a
    // meaningless ~0 against a nonzero oracle. Passivity (monotone non-increasing) is the honest
    // verdict there; the 2σ comparison only means something for a resonator decaying toward rest.
    const hasOracle = e.lossy.measured_2sigma !== undefined;
    const meas = e.lossy.measured_2sigma;
    // The decay_oracle=False readout is model-specific: the mallet decays from a ½M·v₀² floor, the
    // weinreich unison is a TWO-rate decay to a nonzero aftersound floor. Neither has a single
    // flat-loss oracle to compare against, so both report pure passivity — but for different reasons.
    const spec2 = payload && payload.meta ? payload.meta.spectrum : null;
    const isWein = spec2 && spec2.kind === "sympathetic" && spec2.regime === "weinreich";
    out.textContent = hasOracle
      ? `lossy · energy monotone decrease: ${mono ? "yes ✓" : "NO ✗"}${convNote}\n` +
        `measured 2σ = ${meas == null ? "—" : meas.toFixed(3)} s⁻¹` +
        `  (flat-loss oracle ${e.lossy.oracle_2sigma.toFixed(3)})`
      : isWein
        ? `lossy · passive: total energy monotone non-increasing: ${mono ? "yes ✓" : "NO ✗"}` +
          `${convNote}\na two-rate decay to a nonzero aftersound floor — no single-exponential ` +
          `oracle; the two slopes are the claim (see the two-stage-decay panel)`
        : `lossy · passive: energy monotone non-increasing: ${mono ? "yes ✓" : "NO ✗"}${convNote}\n` +
          `felt/membrane loss removes energy from a ½M·v₀² floor — no decay-rate oracle for a ` +
          `closed struck system`;
  }
}

// ── second diagnostic panel: partials (string) | mode spectrum (2D) ──────────────────────────
function drawDiagnostics() {
  const spec = payload && payload.meta && payload.meta.spectrum;
  // Checked before the dims gate: the tension string is a 1-D model that still wants a spectrum
  // panel, not the linear per-partial cents bars (its peak moves tens of percent with amplitude).
  if (spec && spec.kind === "tension") {
    partialsTitle.firstChild.textContent = "Amplitude shift ";
    partialsSub.textContent = "measured vs exact Duffing";
    drawTensionSpectrum();
    return;
  }
  // Also before the dims gate: another 1-D model that wants its own panel rather than cents bars.
  if (spec && spec.kind === "bow") {
    partialsTitle.firstChild.textContent = "Stick-slip ";
    partialsSub.textContent = "slip fraction vs β";
    drawStickSlip();
    return;
  }
  // The jawari: a 1-D model whose claim is a spectral CONTRAST, not its own partials.
  if (spec && spec.kind === "jawari") {
    partialsTitle.firstChild.textContent = "Sustained shimmer ";
    partialsSub.textContent = "late-window brightness vs a clean string";
    drawJawari();
    return;
  }
  // The bore: a 1-D model whose headline is the BELL, not its partials. Odd harmonics are
  // table-stakes here (the boundary condition guarantees them), so the panel spends itself on the
  // reflection oracle instead and reports the partials as text.
  if (spec && spec.kind === "bore") {
    partialsTitle.firstChild.textContent = "Bell reflection ";
    partialsSub.textContent = "measured vs r = (R−Z₀)/(R+Z₀)";
    drawBoreOracle();
    return;
  }
  // The geometric string's four regimes, four panels — all 1-D, none of them cents bars.
  if (spec && spec.kind === "phantom") {
    partialsTitle.firstChild.textContent = "Phantom partials ";
    partialsSub.textContent = "combination tones in the bridge force";
    drawPhantom();
    return;
  }
  if (spec && spec.kind === "whirl") {
    partialsTitle.firstChild.textContent = "Whirl growth ";
    partialsSub.textContent = "envelope of max|w|, log scale";
    drawWhirl();
    return;
  }
  if (spec && (spec.kind === "rotating" || spec.kind === "planar")) {
    partialsTitle.firstChild.textContent = spec.kind === "rotating" ? "Rotating wave " : "Planar ";
    partialsSub.textContent = spec.kind === "rotating" ? "roundness + longitudinal rest"
      : "the reflection symmetry";
    drawGeomVerdict(spec);
    return;
  }
  // Sympathetic / coupled strings: the money panel energy CANNOT see. Normal → both bridge traces
  // (antisym ≡ 0 vs symmetric swinging); transfer → the per-string energy exchange.
  if (spec && spec.kind === "sympathetic") {
    if (spec.regime === "normal") {
      partialsTitle.firstChild.textContent = "Bridge motion ";
      partialsSub.textContent = "antisymmetric ≡ 0 vs symmetric";
    } else if (spec.regime === "weinreich") {
      partialsTitle.firstChild.textContent = "Two-stage decay ";
      partialsSub.textContent = "prompt + aftersound (log)";
    } else {
      partialsTitle.firstChild.textContent = "Sympathetic transfer ";
      partialsSub.textContent = "energy reaching the neighbour";
    }
    drawSympathetic(spec);
    return;
  }
  // The mallet is a 2D heatmap model, but its second panel is NOT a mode spectrum: a soft felt
  // low-passes the strike, so per-mode partial-locking would lock onto noise. The headline is the
  // CONTACT — a point mass is an inefficient membrane exciter (it bounces off with restitution ≈ 1,
  // the head keeps ~0.05 %) — so the panel shows the contact episode, not a tone.
  if (spec && spec.kind === "mallet") {
    partialsTitle.firstChild.textContent = "Contact ";
    partialsSub.textContent = "mallet bounce + felt force";
    drawMallet(spec);
    return;
  }
  if (dims !== 2) {
    partialsTitle.firstChild.textContent = "Partials ";
    partialsSub.textContent = "detected vs analytic";
    drawPartials();
    return;
  }
  const sp = payload && payload.meta && payload.meta.spectrum;
  const kind = sp && sp.kind;
  if (kind === "vk") {
    partialsTitle.firstChild.textContent = "Spectrum (nonlinear) ";
    partialsSub.textContent = "FFT vs linear modes + hardening";
    drawVkSpectrum();
  } else {
    partialsTitle.firstChild.textContent = "Mode spectrum ";
    partialsSub.textContent = kind === "plate" ? "FFT vs plate eigenmodes"
      : "FFT vs discrete eigenmodes";
    drawSpectrum();
  }
}

// Membrane: pickup magnitude spectrum with vertical markers at the discrete eigenfreqs (where the
// time-stepper actually rings — peaks landing on these = self-consistency) and fainter markers at
// the continuum oracle (the O(h) staircase offset; shown, NOT scored). Cf. advisor review 3.
function drawSpectrum() {
  const g = partialsCv.getContext("2d");
  const W = partialsCv.width, H = partialsCv.height, padL = 26, padB = 16, top = 8;
  g.clearRect(0, 0, W, H);
  const out = $("partials-readout");
  const sp = payload && payload.meta && payload.meta.spectrum;
  if (!sp) { out.textContent = "no spectrum"; return; }

  const plotW = W - padL - 8, plotH = H - padB - top;
  const x0 = padL, y0 = top + plotH;
  const fmax = sp.fmax || (sp.freq[sp.freq.length - 1] || 1);
  const fx = (f) => x0 + (f / fmax) * plotW;

  g.strokeStyle = "#2a3340"; g.lineWidth = 1; g.strokeRect(x0, top, plotW, plotH);

  // continuum (Bessel / rectangular) markers — faint, the geometry-tier reference
  g.strokeStyle = "rgba(139,152,168,.35)"; g.lineWidth = 1; g.setLineDash([3, 3]);
  (sp.modes_continuum || []).forEach((f) => {
    if (f == null || f > fmax) return;
    g.beginPath(); g.moveTo(fx(f), top); g.lineTo(fx(f), y0); g.stroke();
  });
  g.setLineDash([]);
  // discrete eigenmode markers — the honest lines the FFT peaks should land on
  g.strokeStyle = "rgba(76,194,255,.55)"; g.lineWidth = 1;
  (sp.modes_discrete || []).forEach((f) => {
    if (f == null || f > fmax) return;
    g.beginPath(); g.moveTo(fx(f), top); g.lineTo(fx(f), y0); g.stroke();
  });

  // FFT magnitude (normalized 0..1)
  g.strokeStyle = "#5ad17a"; g.lineWidth = 1.5; g.beginPath();
  for (let i = 0; i < sp.freq.length; i++) {
    const m = sp.mag[i] == null ? 0 : sp.mag[i];
    const x = fx(sp.freq[i]), y = y0 - m * (plotH - 4);
    if (i === 0) g.moveTo(x, y); else g.lineTo(x, y);
  }
  g.stroke();

  g.fillStyle = "#8b98a8"; g.font = "10px ui-monospace, monospace";
  g.fillText("|X(f)|", 3, top + 10);
  g.fillText(`${Math.round(fmax)} Hz`, W - 54, H - 4);

  const cf = sp.cents_fundamental, cg = sp.cents_geometry;
  // The geometry-tier gap means different things per model: the membrane's O(h) staircase (~9
  // cents by design), the SS plate's tight Navier tier (~1 cent), the free plate's Leissa anchor.
  const tierLabel = sp.kind === "plate" ? "geometry tier (continuum/Leissa)"
    : "geometry tier (O(h) staircase)";
  out.textContent =
    `f₁ = ${sp.f1_discrete.toFixed(2)} Hz (discrete)   peaks on blue lines = self-consistent\n` +
    `fundamental detected vs discrete: ${cf == null ? "—" : cf.toFixed(3) + " cents"}` +
    `   ·   ${tierLabel}: ${cg == null ? "—" : cg.toFixed(2) + " cents"}`;
}

// von Kármán: the marker lines are the LINEAR (w→0) modes; the real peaks sit ABOVE them by the
// amplitude hardening. So this reads the gap as a *hardening shift*, never a cents error. On the
// free cymbal the fundamental is a mode wash (no clean f0) — report the cascade, not a percentage.
function drawVkSpectrum() {
  const g = partialsCv.getContext("2d");
  const W = partialsCv.width, H = partialsCv.height, padL = 26, padB = 16, top = 8;
  g.clearRect(0, 0, W, H);
  const out = $("partials-readout");
  const sp = payload && payload.meta && payload.meta.spectrum;
  if (!sp) { out.textContent = "no spectrum"; return; }

  const plotW = W - padL - 8, plotH = H - padB - top;
  const x0 = padL, y0 = top + plotH;
  const fmax = sp.fmax || (sp.freq[sp.freq.length - 1] || 1);
  const fx = (f) => x0 + (f / fmax) * plotW;

  g.strokeStyle = "#2a3340"; g.lineWidth = 1; g.strokeRect(x0, top, plotW, plotH);

  // linear (w→0) eigenmode markers — the peaks harden ABOVE these, so they are a reference floor
  g.strokeStyle = "rgba(139,152,168,.5)"; g.lineWidth = 1; g.setLineDash([3, 3]);
  (sp.modes_linear || []).forEach((f) => {
    if (f == null || f > fmax) return;
    g.beginPath(); g.moveTo(fx(f), top); g.lineTo(fx(f), y0); g.stroke();
  });
  g.setLineDash([]);
  // the detected hardened fundamental (supported gong only)
  if (sp.f0_detected != null && sp.f0_detected <= fmax) {
    g.strokeStyle = "#ffcf5c"; g.lineWidth = 1.5;
    g.beginPath(); g.moveTo(fx(sp.f0_detected), top); g.lineTo(fx(sp.f0_detected), y0); g.stroke();
  }

  // FFT magnitude (normalized 0..1)
  g.strokeStyle = "#5ad17a"; g.lineWidth = 1.5; g.beginPath();
  for (let i = 0; i < sp.freq.length; i++) {
    const m = sp.mag[i] == null ? 0 : sp.mag[i];
    const x = fx(sp.freq[i]), y = y0 - m * (plotH - 4);
    if (i === 0) g.moveTo(x, y); else g.lineTo(x, y);
  }
  g.stroke();

  g.fillStyle = "#8b98a8"; g.font = "10px ui-monospace, monospace";
  g.fillText("|X(f)|", 3, top + 10);
  g.fillText(`${Math.round(fmax)} Hz`, W - 54, H - 4);

  if (sp.shift_pct != null) {
    out.textContent =
      `linear f₁ = ${sp.f1_linear.toFixed(1)} Hz (grey)   ·   hardened f₀ = `
      + `${sp.f0_detected.toFixed(1)} Hz (yellow)\n`
      + `amplitude hardening: ${sp.shift_pct >= 0 ? "+" : ""}${sp.shift_pct.toFixed(1)}% — the peak `
      + `rides ABOVE the linear mode (not a cents error)`;
  } else {
    out.textContent =
      `linear modes (grey) at ${sp.f1_linear.toFixed(1)} Hz and up\n`
      + `free-edge crash: a multi-mode wash — no single hardened fundamental to report`;
  }
}

// Tension-modulated string: the headline is the *amplitude shift* ω(A) − ω(A→0), not an absolute
// frequency — a measured ω(A) carries the θ-scheme's linear dispersion error and so does ω(A→0), so
// their difference cancels it and isolates the nonlinear physics. Both come from a short LOSSLESS
// pair of runs, never from the (deliberately lossy, deliberately gliding) audio pickup.
function drawTensionSpectrum() {
  const g = partialsCv.getContext("2d");
  const W = partialsCv.width, H = partialsCv.height, padL = 26, padB = 16, top = 8;
  g.clearRect(0, 0, W, H);
  const out = $("partials-readout");
  const sp = payload && payload.meta && payload.meta.spectrum;
  if (!sp) { out.textContent = "no spectrum"; return; }

  const plotW = W - padL - 8, plotH = H - padB - top;
  const x0 = padL, y0 = top + plotH;
  const fmax = sp.fmax || 1;
  const fx = (f) => x0 + (f / fmax) * plotW;
  g.strokeStyle = "#2a3340"; g.lineWidth = 1; g.strokeRect(x0, top, plotW, plotH);

  // linear mode-1 reference (grey): the peak rides ABOVE it by the hardening
  g.strokeStyle = "rgba(139,152,168,.6)"; g.lineWidth = 1; g.setLineDash([3, 3]);
  if (sp.f_linear <= fmax) {
    g.beginPath(); g.moveTo(fx(sp.f_linear), top); g.lineTo(fx(sp.f_linear), y0); g.stroke();
  }
  g.setLineDash([]);
  if (sp.f_hardened != null && sp.f_hardened <= fmax) {
    g.strokeStyle = "#ffcf5c"; g.lineWidth = 1.5;
    g.beginPath(); g.moveTo(fx(sp.f_hardened), top); g.lineTo(fx(sp.f_hardened), y0); g.stroke();
  }

  g.strokeStyle = "#5ad17a"; g.lineWidth = 1.5; g.beginPath();
  for (let i = 0; i < sp.freq.length; i++) {
    const m = sp.mag[i] == null ? 0 : sp.mag[i];
    const x = fx(sp.freq[i]), y = y0 - m * (plotH - 4);
    if (i === 0) g.moveTo(x, y); else g.lineTo(x, y);
  }
  g.stroke();
  g.fillStyle = "#8b98a8"; g.font = "10px ui-monospace, monospace";
  g.fillText("|X(f)|", 3, top + 10);
  g.fillText(`${Math.round(fmax)} Hz`, W - 54, H - 4);

  const dt = `ΔT/T₀ = ${sp.dT_over_T.toFixed(2)}`;
  if (sp.shift_measured == null) {
    // The mode broke up (parametric instability): the Duffing reduction no longer describes the
    // motion, so report nothing rather than a lying number.
    out.textContent =
      `linear f₁ = ${sp.f_linear.toFixed(2)} Hz (grey)   ·   ${dt}\n`
      + `mode broke up (off-mode ${sp.purity.off_mode.toExponential(1)}) — shift N/A`;
    return;
  }
  const err = sp.shift_rel_error == null ? "" :
    `   ·   error ${(sp.shift_rel_error * 100).toFixed(2)}%`;
  out.textContent =
    `linear f₁ ${sp.f_linear.toFixed(2)} Hz (grey)  →  hardened ${sp.f_hardened.toFixed(2)} Hz `
    + `(yellow)   ·   ${dt}\n`
    + `shift ${sp.shift_measured.toFixed(2)} Hz vs exact Duffing ${sp.shift_oracle.toFixed(2)} Hz`
    + `${err}   ·   ${sp.shift_cents == null ? "" : `+${sp.shift_cents.toFixed(0)} cents`}`;
}

// Bowed string: the bow-point relative velocity over ~3 settled periods. Helmholtz motion is a
// two-state cycle — the string STICKS to the bow (v_rel ≈ 0, inside the shaded band) for 1−β of
// the period, then SLIPS back once, for a fraction β. The oracle is that slip fraction == β, with
// the bow-position slider sitting directly on its free parameter.
//
// It is scored ONLY when the motion really is one-slip-per-period. Outside Schelleng's force
// window (which narrows as the bow moves off the bridge) the note crushes or goes raucous and the
// match legitimately breaks — that is real physics faithfully reproduced, not a solver failure, so
// the panel labels it and scores nothing rather than painting a red FAIL on a correct run.
function drawStickSlip() {
  const g = partialsCv.getContext("2d");
  const W = partialsCv.width, H = partialsCv.height, padL = 26, padB = 16, top = 8;
  g.clearRect(0, 0, W, H);
  const out = $("partials-readout");
  const sp = payload && payload.meta && payload.meta.spectrum;
  if (!sp) { out.textContent = "no stick-slip trace"; return; }

  const v = sp.v_rel.map((x) => (x == null ? 0 : x));
  const plotW = W - padL - 8, plotH = H - padB - top;
  const x0 = padL, yMid = top + plotH / 2;
  const vmax = Math.max(1e-12, ...v.map(Math.abs)) * 1.1;
  const vy = (val) => yMid - (val / vmax) * (plotH / 2);

  // the stick band: |v_rel| < half the bow speed, the detector's own threshold
  g.fillStyle = "rgba(90,209,122,.10)";
  g.fillRect(x0, vy(sp.stick_threshold), plotW, vy(-sp.stick_threshold) - vy(sp.stick_threshold));
  g.strokeStyle = "#2a3340"; g.lineWidth = 1; g.strokeRect(x0, top, plotW, plotH);
  g.strokeStyle = "rgba(139,152,168,.5)"; g.setLineDash([3, 3]);
  g.beginPath(); g.moveTo(x0, vy(0)); g.lineTo(x0 + plotW, vy(0)); g.stroke();
  g.setLineDash([]);

  g.strokeStyle = "#e0b050"; g.lineWidth = 1.5; g.beginPath();
  for (let i = 0; i < v.length; i++) {
    const x = x0 + (i / Math.max(1, v.length - 1)) * plotW;
    if (i === 0) g.moveTo(x, vy(v[i])); else g.lineTo(x, vy(v[i]));
  }
  g.stroke();
  g.fillStyle = "#8b98a8"; g.font = "10px ui-monospace, monospace";
  g.fillText("v_rel", 4, top + 10);
  g.fillText("stick band", x0 + 4, vy(0) - 3);

  if (!sp.helmholtz) {
    out.textContent =
      `${sp.slips_per_period.toFixed(1)} slips/period — outside the Helmholtz window (clean `
      + `bowing is 1)\nreal physics, not a solver failure: Schelleng's force window narrows off `
      + `the bridge — slip = β not scored`;
    return;
  }
  const ok = sp.slip_matches_beta;
  const pitch = sp.pitch_cents == null ? "" :
    `   ·   pitch ${sp.f_detected.toFixed(1)} Hz = f₁ ${sp.f1.toFixed(0)} Hz `
    + `${sp.pitch_cents >= 0 ? "+" : ""}${sp.pitch_cents.toFixed(0)} c`;
  out.textContent =
    `Helmholtz: ${sp.slips_per_period.toFixed(2)} slips/period ✓${pitch}\n`
    + `slip fraction ${sp.slip_fraction.toFixed(3)} vs β = ${sp.beta.toFixed(3)}  (Δ `
    + `${sp.slip_error >= 0 ? "+" : ""}${sp.slip_error.toFixed(3)}, tol ${sp.slip_tol})  →  `
    + `${ok ? "PASS ✓" : "FAIL ✗"}`;
}

// ── mallet: the contact episode ──────────────────────────────────────────────────────────────
// The mallet's headline, not a tone spectrum. Two curves over the (auto-zoomed) contact window:
// the mallet velocity — which crosses zero and comes out POSITIVE, the visible signature of a
// bounce — over a signed centred axis, and the felt force as a filled pulse underneath (its own
// scale, since force ≥ 0). A dashed line marks separation. The readout carries the physics the
// picture can't: restitution ≈ 1 and the head keeping ~0.05 % — a point mass is an inefficient
// membrane exciter (the local reactive dimple returns almost all the energy to the mallet).
function drawMallet(sp) {
  const g = partialsCv.getContext("2d");
  const W = partialsCv.width, H = partialsCv.height, padL = 26, padB = 16, top = 8;
  g.clearRect(0, 0, W, H);
  const out = $("partials-readout");
  if (!sp || !sp.vel) { out.textContent = "no contact trace"; return; }

  const vel = sp.vel.map((x) => (x == null ? 0 : x));
  const force = sp.force.map((x) => (x == null ? 0 : x));
  const t = sp.t.map((x) => (x == null ? 0 : x));
  const plotW = W - padL - 8, plotH = H - padB - top;
  const x0 = padL, yMid = top + plotH / 2;
  const tmax = t[t.length - 1] || 1;
  const vmax = Math.max(1e-9, ...vel.map(Math.abs)) * 1.1;
  const fmax = Math.max(1e-9, ...force);
  const tx = (i) => x0 + (t[i] / tmax) * plotW;

  g.strokeStyle = "#2a3340"; g.lineWidth = 1; g.strokeRect(x0, top, plotW, plotH);
  // zero-velocity line: crossing it upward IS the bounce.
  g.strokeStyle = "rgba(139,152,168,.5)"; g.setLineDash([3, 3]);
  g.beginPath(); g.moveTo(x0, yMid); g.lineTo(x0 + plotW, yMid); g.stroke();
  g.setLineDash([]);

  // felt force as a filled pulse from the baseline (its own scale, drawn first/underneath).
  g.fillStyle = "rgba(224,176,80,.18)"; g.beginPath(); g.moveTo(x0, top + plotH);
  for (let i = 0; i < force.length; i++) g.lineTo(tx(i), top + plotH - (force[i] / fmax) * plotH);
  g.lineTo(x0 + plotW, top + plotH); g.closePath(); g.fill();

  // separation marker.
  if (sp.separated && sp.contact_ms != null) {
    const xs = x0 + (sp.contact_ms / 1e3 / tmax) * plotW;
    g.strokeStyle = "rgba(255,95,109,.6)"; g.lineWidth = 1; g.setLineDash([2, 3]);
    g.beginPath(); g.moveTo(xs, top); g.lineTo(xs, top + plotH); g.stroke(); g.setLineDash([]);
  }

  // velocity trace on the signed axis.
  g.strokeStyle = "#4cc2ff"; g.lineWidth = 1.8; g.beginPath();
  for (let i = 0; i < vel.length; i++) {
    const y = yMid - (vel[i] / vmax) * (plotH / 2);
    if (i === 0) g.moveTo(tx(i), y); else g.lineTo(tx(i), y);
  }
  g.stroke();

  g.fillStyle = "#8b98a8"; g.font = "10px ui-monospace, monospace";
  g.fillText("mallet v", 4, top + 10);
  g.fillText("felt force", x0 + 4, top + plotH - 4);
  g.fillText(`${(tmax * 1e3).toFixed(0)} ms`, W - 40, H - 4);

  const dur = sp.separated ? `${sp.contact_ms.toFixed(1)} ms`
    : (sp.contact_ms == null && sp.restitution === 1 && sp.peak_force === 0
      ? "no contact" : "still in contact at window end");
  const note = sp.resolved ? "" :
    `\n⚠ felt under-resolved (~${sp.steps_per_contact.toFixed(0)} steps/contact < 8) — raise fs `
    + `(lower λ) or soften K; energy still conserves, the strike just aliases`;
  out.textContent =
    `restitution ${sp.restitution.toFixed(3)}  ·  contact ${dur}  ·  peak felt force `
    + `${sp.peak_force.toFixed(2)} N\n`
    + `head keeps ${sp.final_head_pct.toFixed(3)} % of the strike (peak ${sp.peak_head_pct.toFixed(0)} `
    + `% mid-contact) — a point mass barely rings a drum${note}`;
}

// ── geometric string: the whirl envelope (log-y) ─────────────────────────────────────────────
// A straight line on log axes IS the Mathieu instability — nothing else makes max|w| climb like
// that, and it is legible after ~2 e-foldings, long before the orbit visibly opens (at the
// affordable 0.06 s max|w|/max|u| has only reached ~0.08, which still reads as a line on equal
// axes). So this panel, not the orbit, is where the whirl is proven. What is plotted is the ENVELOPE
// (a sliding
// ~1-period max), never the raw max|w|: every node crosses zero twice a period, so the instantaneous
// value oscillates non-monotonically and the line underneath is lost in the spikes.
// One strip of the phantom panel: a dB spectrum over [f_lo, f_hi] with two marker families.
// dB, not linear: the four combination tones span orders of magnitude (the difference tone towers
// over 2f₁), and on a linear axis the weakest of them is a flat line on the floor.
function phantomStrip(g, sp, x0, y0, w, h, fLo, fHi, freq, mag, opts) {
  const fx = (f) => x0 + ((f - fLo) / Math.max(fHi - fLo, 1e-9)) * w;
  const DB_FLOOR = -70;
  const dy = (m) => {
    const db = 20 * Math.log10((m == null ? 0 : m) + 1e-12);
    return y0 + h - Math.max(0, (db - DB_FLOOR) / -DB_FLOOR) * (h - 3);
  };
  g.strokeStyle = "#2a3340"; g.lineWidth = 1; g.strokeRect(x0, y0, w, h);

  // Where transverse partials ARE (shown, never scored — the discrete ladder). The phantoms land
  // in the GAPS between these; a longitudinal peak sitting ON one would mean the fields are coupled
  // linearly, which is the bug this excludes.
  g.strokeStyle = "rgba(255,110,110,.65)"; g.lineWidth = 1; g.setLineDash([3, 3]);
  (opts.partials || []).forEach((f) => {
    if (f == null || f < fLo || f > fHi) return;
    g.beginPath(); g.moveTo(fx(f), y0); g.lineTo(fx(f), y0 + h); g.stroke();
  });
  g.setLineDash([]);
  // Where the phantoms ARE: the quadratic combinations of the MEASURED partials.
  g.strokeStyle = "#5ad17a"; g.lineWidth = 1.2;
  (opts.combos || []).forEach((f) => {
    if (f == null || f < fLo || f > fHi) return;
    g.beginPath(); g.moveTo(fx(f), y0); g.lineTo(fx(f), y0 + h); g.stroke();
  });

  g.strokeStyle = "#cdd6e2"; g.lineWidth = 1; g.beginPath();
  let started = false;
  for (let i = 0; i < freq.length; i++) {
    if (freq[i] < fLo || freq[i] > fHi) continue;
    const x = fx(freq[i]), y = dy(mag[i]);
    if (!started) { g.moveTo(x, y); started = true; } else g.lineTo(x, y);
  }
  g.stroke();

  // Labels go over a knocked-out background: the trace's peaks reach the top of the strip, and
  // plain text there is unreadable (the first cut was drawn straight through by the spectrum).
  g.font = "9px ui-monospace, monospace";
  if (opts.label) {
    const tw = g.measureText(opts.label).width;
    g.fillStyle = "rgba(16,20,28,.82)";
    g.fillRect(x0 + 2, y0 + 1, tw + 5, 11);
    g.fillStyle = "#8b98a8";
    g.fillText(opts.label, x0 + 4, y0 + 9);
  }
  const hz = `${Math.round(fHi)} Hz`;
  g.fillStyle = "rgba(16,20,28,.82)";
  g.fillRect(x0 + w - g.measureText(hz).width - 6, y0 + h - 11, g.measureText(hz).width + 5, 10);
  g.fillStyle = "#8b98a8";
  g.fillText(hz, x0 + w - g.measureText(hz).width - 4, y0 + h - 3);
  return fx;
}

// The phantom panel — model #9's first refusal, discharged. TWO strips, and that is not decoration:
// the claim has two halves and no single axis carries both. Wide (0 → 4.8 f₁) shows the four
// combination tones, which is the mechanism; but there the 4.6 Hz defect is ~4 px, so the half that
// says "and NOT on a partial" renders as its own opposite — exactly the trap the diagnose figure hit
// on a 2 kHz axis. The zoom strip frames the f₁ / (f₂−f₁) pair where 4.6 Hz is ~31 px and the two
// lines are plainly separate.
function drawPhantom() {
  const g = partialsCv.getContext("2d");
  const W = partialsCv.width, H = partialsCv.height, padL = 6, gap = 8;
  g.clearRect(0, 0, W, H);
  const out = $("partials-readout");
  const sp = payload && payload.meta && payload.meta.spectrum;
  if (!sp || !sp.wide_freq || !sp.wide_freq.length) { out.textContent = "no bridge spectrum"; return; }

  const combos = Object.values(sp.combos);
  const stripH = (H - gap - 4) / 2, plotW = W - padL - 8;
  phantomStrip(g, sp, padL, 2, plotW, stripH, sp.band[0], sp.band[1], sp.wide_freq, sp.wide_mag,
               { partials: sp.ladder, combos, label: "bridge force EA·v_x(0)" });
  const fx = phantomStrip(g, sp, padL, 2 + stripH + gap, plotW, stripH, sp.zoom[0], sp.zoom[1],
                          sp.zoom_freq, sp.zoom_mag,
                          { partials: [sp.f1], combos: [sp.combos["f2-f1"]],
                            label: "zoom: f₂−f₁ vs f₁" });

  // The defect, drawn: a bracket between the partial and the phantom that is supposed to miss it.
  if (sp.f1 >= sp.zoom[0] && sp.f1 <= sp.zoom[1]) {
    const yb = 2 + stripH + gap + stripH - 12;
    const xa = fx(sp.f1), xb = fx(sp.combos["f2-f1"]);
    g.strokeStyle = "#ffcf5c"; g.lineWidth = 1;
    g.beginPath(); g.moveTo(xa, yb); g.lineTo(xb, yb); g.stroke();
    g.fillStyle = "#ffcf5c"; g.font = "9px ui-monospace, monospace";
    g.fillText(`${Math.abs(sp.defect).toFixed(2)} Hz`, Math.min(xa, xb) + 3, yb - 3);
  }

  const err = sp.combo_err == null ? "—" : sp.combo_err.toFixed(3);
  const dom = sp.dominance == null ? "—" : sp.dominance.toFixed(1) + "×";
  const d1 = sp.displacements[0], d2 = sp.displacements[1];
  // The harness control, and it comes FIRST: on a linear string (EA = T₀ ⇒ a = 0) the phantom
  // channel does not exist at all and the bridge force is bit-exactly zero. That is a result, not a
  // failure — and checking it before `resolved` is what stops an empty spectrum reading as a claim.
  if (sp.linear) {
    out.textContent =
      `EA = T₀ ⇒ a = 0 · the bridge force is ${sp.bridge_max === 0 ? "BIT-EXACTLY zero" : "zero"} — `
      + `there is no phantom channel to look in\n`
      + `the nonlinear excess a(r²v_x/2 + r⁴/8) vanishes outright, so the three fields decouple and `
      + `v, started at rest, never leaves it. The control that proves the readout is not `
      + `manufacturing its own result.`;
    return;
  }
  // Label, never FAIL (the bow's Schelleng window, the free cymbal's null): a too-harmonic string
  // genuinely has no phantom signature. Nothing is broken — there is just nothing to discriminate.
  if (sp.defect < sp.defect_min) {
    out.textContent =
      `not discriminating · defect f₂−2f₁ = ${sp.defect.toFixed(2)} Hz (need ≥ ${sp.defect_min}), `
      + `at κ = ${sp.kappa}, N = ${payload.frames.width - 1}\n`
      + `the phantoms have collapsed onto the partials: on a harmonic string f₂−f₁ = f₁ and 2f₁ = f₂ `
      + `EXACTLY, so there is no gap to land in. TWO knobs open it — κ is the microscope, but the `
      + `θ-scheme's own dispersion drags f₂ flat and eats the stiffness on a coarse grid `
      + `(at κ=8 the defect is +0.4 Hz at N=8 and +4.4 at N=32). Raise κ, or N, or both.`;
    return;
  }
  if (!sp.resolved) {
    out.textContent =
      `only ${sp.n_peaks} in-band peak${sp.n_peaks === 1 ? "" : "s"} detected — fewer than the 4 `
      + `quadratic combinations the claim is about, so nothing is scored here.`;
    return;
  }
  // The lobe caveat belongs here, not on the strip: the zoom's peak is ~40 Hz wide (a 0.1 s Hann
  // main lobe, 4/T) while the defect is ~4.6 Hz, so a reader who knows their DSP will immediately
  // ask how a 4.6 Hz gap is claimed inside one lobe. The answer is that peak POSITION is not
  // Rayleigh-limited — f₁ is absent from v, so the phantom has no neighbour to be resolved from,
  // and parabolic refinement locates it to ~0.04 Hz. Answering it unasked is the difference between
  // a panel that is trusted and one that is caught out.
  const lobe = Math.round(4 / (payload.meta.num_steps / payload.fs_sim));
  out.textContent =
    `the 4 strongest in-band peaks ARE the 4 quadratic combinations (max err ${err} Hz), `
    + `${dom} over the strongest non-combo\n`
    + `defect f₂−2f₁ = ${sp.defect.toFixed(2)} Hz · the difference tone misses f₁ by `
    + `${d1 == null ? "—" : d1.toFixed(2)} Hz and 2f₁ misses f₂ by ${d2 == null ? "—" : d2.toFixed(2)} `
    + `Hz — the same number, from both sides, with no oracle\n`
    + `(the zoom's peak is ~${lobe} Hz wide — a 0.1 s Hann lobe. Its POSITION is the claim, and `
    + `position is not resolution: f₁ is absent from v, so nothing neighbours the phantom to blur it.)`;
}

// ── jawari: the shimmer, as a spectral CONTRAST ──────────────────────────────────────────────
// Two late-window spectra on ONE shared vertical scale — the jawari's and the same string with the
// bridge dropped out of reach. Shared is the whole point: the claim is that the curved contact
// keeps re-injecting high partials, so it is the RELATIVE height of the two traces up the band
// that carries it. Normalizing each to its own peak would show two similar-looking curves and
// silently delete the result.
function drawJawari() {
  const g = partialsCv.getContext("2d");
  const W = partialsCv.width, H = partialsCv.height, padL = 34, padB = 16, top = 10;
  g.clearRect(0, 0, W, H);
  const out = $("partials-readout");
  const sp = payload && payload.meta && payload.meta.spectrum;
  if (!sp || !sp.spectra) { out.textContent = "no jawari spectra"; return; }

  const plotW = W - padL - 8, plotH = H - padB - top;
  g.strokeStyle = "#2a3340"; g.lineWidth = 1; g.strokeRect(padL, top, plotW, plotH);

  const fMax = sp.spectra.f_max || 2000;
  const px = (f) => padL + (f / fMax) * plotW;
  // Log magnitude over 4 decades: the re-injected partials are 10⁻²–10⁻³ of the fundamental, which
  // on a linear axis is a flat line along the bottom for BOTH traces — the contrast would vanish
  // into the axis exactly where it lives.
  const floor = 1e-4;
  const py = (m) => {
    const v = Math.max(m || floor, floor);
    return top + plotH - ((Math.log10(v) - Math.log10(floor)) / 4) * plotH;
  };
  for (let d = 0; d >= -4; d--) {
    const y = py(Math.pow(10, d));
    g.strokeStyle = "rgba(139,152,168,.15)"; g.lineWidth = 1;
    g.beginPath(); g.moveTo(padL, y); g.lineTo(padL + plotW, y); g.stroke();
    g.fillStyle = "#8b98a8"; g.font = "9px ui-monospace, monospace";
    g.fillText(`1e${d}`, 3, y + 3);
  }
  // f1 marker: the clean string collapses onto it, the jawari does not.
  if (sp.f1) {
    g.strokeStyle = "rgba(255,207,92,.35)"; g.setLineDash([4, 4]);
    g.beginPath(); g.moveTo(px(sp.f1), top); g.lineTo(px(sp.f1), top + plotH); g.stroke();
    g.setLineDash([]);
  }

  const trace = (data, colour, wdt) => {
    if (!data || !data.f) return;
    g.strokeStyle = colour; g.lineWidth = wdt; g.beginPath();
    let started = false;
    for (let i = 0; i < data.f.length; i++) {
      const X = px(data.f[i]), Y = py(data.mag[i]);
      if (!started) { g.moveTo(X, Y); started = true; } else g.lineTo(X, Y);
    }
    g.stroke();
  };
  trace(sp.spectra.clean, "#8b98a8", 1.5);
  trace(sp.spectra.jawari, "#ff8f4c", 2);

  g.font = "10px ui-monospace, monospace";
  g.fillStyle = "#ff8f4c"; g.fillText("jawari", padL + 6, top + 12);
  g.fillStyle = "#8b98a8"; g.fillText("clean string", padL + 6, top + 24);
  g.fillText(`${Math.round(fMax)} Hz`, W - 52, H - 5);

  const c = sp.centroid;
  const wrap = sp.wrap || {};
  // The elevation is the gate; the sustain ratio is printed but deliberately NOT gated — it wobbles
  // 0.9–1.3 with the decay rate and window placement, so a render could flip on nothing physical.
  const verdict = sp.shimmering
    ? `SHIMMER ✓  late brightness ${sp.elevation.toFixed(2)}× the clean string (gate ${sp.elevation_gate}×)`
    : sp.grazing
      ? `GRAZING — not a jawari at these settings (${sp.elevation.toFixed(2)}× < ${sp.elevation_gate}×)`
      : `weak — ${sp.elevation.toFixed(2)}× < ${sp.elevation_gate}×`;
  const geometry = sp.grazing
    ? `\ndownswing/depth = ${sp.ratio} < ${sp.ratio_floor}: the string only grazes the crest, so the `
      + `contact is a stiff POINT, not a wrap — raise amplitude or reduce depth. A legitimate `
      + `config, just not this timbre.`
    : `\ndownswing/depth = ${sp.ratio} (floor ~${sp.ratio_floor}) — the swing clears the curve, so `
      + `the string wraps rather than grazing`;
  out.textContent =
    `${verdict}\n`
    + `centroid late: jawari ${c.jawari_late} Hz vs clean ${c.clean_late} Hz  `
    + `(early ${c.jawari_early} / ${c.clean_early})\n`
    + `sustain jawari ${sp.sustain_ratio == null ? "—" : sp.sustain_ratio.toFixed(2)}× vs clean `
    + `${sp.clean_sustain_ratio == null ? "—" : sp.clean_sustain_ratio.toFixed(2)}× (late/early — reported, not gated)`
    + geometry
    + `\nwrap edge sweeps nodes ${wrap.min_node}–${wrap.max_node} of ${wrap.support}, std `
    + `${wrap.std}, in contact ${(wrap.duty * 100).toFixed(0)} % of the run — the suite's flat rail `
    + `at matched clearance pins at std ${wrap.flat_rail_std} (tests/test_jawari.py)`;
}

// ── the bell: one bounce against a closed form, on a log R/Z₀ axis ────────────────────────────
// The curve is FREE — r = (ratio−1)/(ratio+1) is geometry-invariant, so the analytic sweep costs no
// simulation. Only one point is measured (the user's own R/Z₀), from a centred Gaussian that splits
// into two halves and bounces once. Log axis because the interesting range is five decades wide and
// the two landmarks — a physical clarinet at 3e-4 and the anechoic match at 1 — are 3.5 decades
// apart; on a linear axis every real bell would pile onto the left edge.
function drawBoreOracle() {
  const g = partialsCv.getContext("2d");
  const W = partialsCv.width, H = partialsCv.height, padL = 38, padB = 18, top = 12;
  g.clearRect(0, 0, W, H);
  const out = $("partials-readout");
  const m = payload && payload.meta;
  const rb = m && m.reflection;
  if (!rb) { out.textContent = "no bell data"; return; }

  const plotW = W - padL - 10, plotH = H - padB - top;
  g.strokeStyle = "#2a3340"; g.lineWidth = 1; g.strokeRect(padL, top, plotW, plotH);

  const lo = -4, hi = 1.5;                       // log₁₀(R/Z₀) span of the shipped curve
  const px = (r) => padL + ((Math.log10(r) - lo) / (hi - lo)) * plotW;
  const py = (s) => top + plotH - (s / 0.55) * plotH;
  for (let d = lo; d <= hi + 1e-9; d++) {
    const x = px(Math.pow(10, d));
    g.strokeStyle = "rgba(139,152,168,.15)"; g.lineWidth = 1;
    g.beginPath(); g.moveTo(x, top); g.lineTo(x, top + plotH); g.stroke();
    g.fillStyle = "#8b98a8"; g.font = "9px ui-monospace, monospace";
    g.fillText(`1e${d}`, x - 8, H - 6);
  }
  // The 0.5 ceiling: a centred pulse sends only HALF its energy at the bell, so even a perfect
  // absorber cannot shed more. Marking it stops the anechoic point reading as "only 50 % — a leak".
  g.strokeStyle = "rgba(139,152,168,.25)"; g.setLineDash([4, 4]);
  g.beginPath(); g.moveTo(padL, py(0.5)); g.lineTo(padL + plotW, py(0.5)); g.stroke();
  g.setLineDash([]);
  g.fillStyle = "#8b98a8";
  g.fillText("½ — the whole right-going half", padL + plotW - 158, py(0.5) - 5);

  g.strokeStyle = "#7fd4a0"; g.lineWidth = 2; g.beginPath();
  rb.curve.ratio.forEach((r, i) => {
    const x = px(r), y = py(rb.curve.shed[i]);
    if (i === 0) g.moveTo(x, y); else g.lineTo(x, y);
  });
  g.stroke();

  if (rb.radiating) {
    const x = px(rb.ratio), y = py(rb.measured);
    g.strokeStyle = "rgba(255,207,92,.4)"; g.lineWidth = 1; g.setLineDash([3, 3]);
    g.beginPath(); g.moveTo(x, top); g.lineTo(x, top + plotH); g.stroke(); g.setLineDash([]);
    g.fillStyle = "#ffcf5c";
    g.beginPath(); g.arc(x, y, 4.5, 0, 7); g.fill();
  }
  g.font = "10px ui-monospace, monospace";
  g.fillStyle = "#7fd4a0"; g.fillText("oracle ½(1−r²)", padL + 6, top + 12);
  if (rb.radiating) { g.fillStyle = "#ffcf5c"; g.fillText("measured", padL + 6, top + 24); }

  const sp = m.spectrum, dp = m.dispersion, pa = sp && sp.partials;
  const verdict = !rb.radiating
    ? "NO BELL — the ideal open end is a perfect mirror (r = −1); nothing radiates."
    : rb.pass
      ? `MATCH ✓  shed ${rb.measured.toFixed(6)} vs oracle ${rb.oracle.toFixed(6)} `
        + `(|err| ${rb.abs_error.toExponential(1)}${rb.anechoic ? ", the ANECHOIC null" : ""})`
      : `MISMATCH — shed ${rb.measured.toFixed(6)} vs oracle ${rb.oracle.toFixed(6)}`;
  // The signature claims need a standing wave. At a heavily-absorbing bell there is none — that is
  // a correct render with nothing to measure, so it is LABELLED, never failed.
  const sig = !sp ? "" : sp.applies
    ? `\nclarinet signature ✓  odd/even = ${sp.odd_even.ratio.toExponential(2)} `
      + `(gate ${sp.odd_even.gate.toExponential(0)}) — only odd resonances exist for a closed-open `
      + `tube, so this is the boundary condition made audible`
      + `\npartials vs the EIGENVALUE oracle: worst `
      + `${Math.max(...pa.cents_vs_eigen.map(Math.abs)).toFixed(4)} cents; the oracle itself sits `
      + `${Math.max(...pa.eigen_vs_continuum.map(Math.abs)).toFixed(4)} cents off the continuum `
      + `(λ = 1 is dispersionless, at every N)`
    : `\nno standing wave at R/Z₀ = ${m.r_ratio.toExponential(2)}: the bell absorbs the pulse `
      + `before it can return, so the odd-harmonic and partial claims do not apply here. Not a `
      + `failure — there is nothing to measure. The envelope degrades to the pulse's single pass.`;
  const disp = !dp ? "" : `\nλ (from the OPERATOR, no stepping): worst departure `
    + `${dp.coarse[0].toFixed(3)} cents at λ = ${dp.lambda[0]} on N = ${dp.n_coarse}, `
    + `${dp.fine[0].toFixed(3)} on N = ${dp.n_fine} — a ratio of ${dp.order[0]}, i.e. O(h²) — `
    + `collapsing to ${dp.fine[dp.fine.length - 1].toFixed(4)} at λ = 1`;
  // The "and it STILL conserves" line only means something when something actually left. With no
  // bell it would boast about booking a channel that carried nothing.
  const book = rb.radiating
    ? `\nradiated ${(100 * m.radiated_frac).toFixed(2)} % of E₀ over the run — and the total still `
      + `conserves, because the bell's loss is BOOKED (see the energy panel's split)`
    : `\nnothing radiated: the split in the energy panel is the whole total, and conservation here `
      + `is the ordinary lossless kind`;
  out.textContent = verdict + book + sig + disp;
}

function drawWhirl() {
  const g = partialsCv.getContext("2d");
  const W = partialsCv.width, H = partialsCv.height, padL = 34, padB = 16, top = 10;
  g.clearRect(0, 0, W, H);
  const out = $("partials-readout");
  const sp = payload && payload.meta && payload.meta.spectrum;
  if (!sp) { out.textContent = "no whirl trace"; return; }

  const plotW = W - padL - 8, plotH = H - padB - top;
  g.strokeStyle = "#2a3340"; g.lineWidth = 1; g.strokeRect(padL, top, plotW, plotH);

  // The honesty gate: unseeded, max|w| is bit-exact 0.0 — there is nothing to plot, and that is the
  // result. Every growth ratio on this panel rests on it (without it, growth partly measures a leak).
  if (!sp.seeded) {
    g.fillStyle = "#5ad17a"; g.font = "12px ui-monospace, monospace";
    g.fillText("max|w| = 0.0   (bit-exact)", padL + 14, top + plotH / 2);
    out.textContent =
      "honesty gate · seed = 0 → max|w| is bit-exact zero, at the tongue's centre\n"
      + "the out-of-plane field cannot be excited by the in-plane one: nothing leaks, so every "
      + "growth ratio here is real";
    return;
  }

  const t = sp.time, e = sp.envelope.map((x) => (x == null || x <= 0 ? 1e-300 : x));
  const lo = Math.log10(Math.min(...e)), hi = Math.log10(Math.max(...e));
  const span = Math.max(hi - lo, 1e-9);
  const tmax = Math.max(t[t.length - 1], 1e-12);
  const px = (i) => padL + (t[i] / tmax) * plotW;
  const py = (val) => top + plotH - ((Math.log10(val) - lo) / span) * plotH;

  // decade gridlines — the eye reads "straight over N decades" off these
  g.strokeStyle = "rgba(139,152,168,.18)"; g.lineWidth = 1;
  for (let d = Math.ceil(lo); d <= Math.floor(hi); d++) {
    const y = py(Math.pow(10, d));
    g.beginPath(); g.moveTo(padL, y); g.lineTo(padL + plotW, y); g.stroke();
    g.fillStyle = "#8b98a8"; g.font = "9px ui-monospace, monospace";
    g.fillText(`1e${d}`, 3, y + 3);
  }

  g.strokeStyle = "#ff8f4c"; g.lineWidth = 2; g.beginPath();
  for (let i = 0; i < e.length; i++) {
    if (i === 0) g.moveTo(px(i), py(e[i])); else g.lineTo(px(i), py(e[i]));
  }
  g.stroke();
  g.fillStyle = "#8b98a8"; g.font = "10px ui-monospace, monospace";
  g.fillText("max|w| (log)", padL + 4, top + 10);
  g.fillText(`${tmax.toFixed(3)} s`, W - 46, H - 5);

  const rate = sp.measured_rate == null ? "—" : sp.measured_rate.toFixed(1);
  if (sp.degenerate) {
    // A degenerate string has no tongue — rotational symmetry forces ω_w = ω_u at any amplitude.
    // Which seed you use decides what you learn, and the curve's SHAPE is the discriminator: a
    // velocity kick injects angular momentum and grows SECULARLY (linear in t ⇒ bends on log axes);
    // a displaced w is the rotation generator, so the run is just planar motion in a rotated plane.
    out.textContent = sp.seed_velocity
      ? `degenerate (δ = 0) · |w| grows ${sp.growth.toFixed(2)}× — but LINEARLY, not exponentially\n`
        + `a velocity seed injects angular momentum ⇒ secular growth (it BENDS on log axes). `
        + `Marginal, not unstable — no tongue exists at δ = 0.`
      : `degenerate (δ = 0) · |w| grows ${sp.growth.toFixed(2)}× — it cannot whirl\n`
        + `a displaced w IS the rotation generator: this is the same planar motion in a rotated `
        + `plane. Tick "velocity seed" to see the marginal (secular) growth instead.`;
    return;
  }
  if (!sp.in_tongue) {
    out.textContent =
      `outside the tongue · δ/(εA²) = ${sp.tongue_position} > ½ → |w| grows ${sp.growth.toFixed(2)}×\n`
      + `no parametric resonance: the pump cannot reach the detuned polarization. The upper edge is `
      + `SOFT (leading-order ε), so growth fades rather than switching off.`;
    return;
  }
  // Tier C, reported and never scored: the measured rate runs 5–11% BELOW the closed form, and
  // systematically so (leading-order ε, plus the seed's non-growing component). Dressing that as a
  // pass/fail would invent a bar the physics does not support.
  out.textContent =
    `inside the tongue · δ/(εA²) = ${sp.tongue_position} (peak ${sp.peak_at}) → |w| grows `
    + `${sp.growth.toFixed(1)}×, κ_w = ${sp.kappa_w}\n`
    + `rate ${rate} s⁻¹ vs Mathieu (Ω/2)√(q²−σ²) = ${sp.predicted_rate.toFixed(1)} s⁻¹  `
    + `(${sp.rate_ratio == null ? "—" : (sp.rate_ratio * 100).toFixed(0) + "%"} — runs `
    + `systematically low; reported, not scored)`;
}

// ── geometric string: the planar + rotating verdicts ─────────────────────────────────────────
// Both are exact statements rather than plots, so the panel is the number. Planar: max|w| == 0.0 is
// the w → −w reflection symmetry, not a small number. Rotating: the helix is an EXACT solution of
// the scheme, so its own roundness is the oracle, and the longitudinal field holds still.
function drawGeomVerdict(sp) {
  const g = partialsCv.getContext("2d");
  const W = partialsCv.width, H = partialsCv.height;
  g.clearRect(0, 0, W, H);
  const out = $("partials-readout");
  const rows = sp.kind === "planar"
    ? [["max|w|", sp.max_w === 0 ? "0.0" : sp.max_w.toExponential(2),
        sp.exact_zero ? "bit-exact ✓" : "NOT zero ✗"]]
    : [["roundness (r_max−r_min)/r̄", sp.roundness.toExponential(2),
        sp.bvp_converged ? "a true circle ✓" : "BVP DID NOT CONVERGE ✗"],
       ["longitudinal KE / E", sp.long_kin_over_e.toExponential(2), "it does not move ✓"],
       ["BVP frequency", sp.bvp_frequency.toFixed(3) + " Hz", `${sp.bvp_iterations} Newton iters`]];

  g.font = "12px ui-monospace, monospace";
  rows.forEach(([k, v, note], i) => {
    const y = 26 + i * 34;
    g.fillStyle = "#8b98a8"; g.fillText(k, 12, y);
    g.fillStyle = "#e8edf3"; g.font = "15px ui-monospace, monospace";
    g.fillText(v, 12, y + 18);
    g.font = "12px ui-monospace, monospace";
    g.fillStyle = "#5ad17a"; g.fillText(note, W - 12 - g.measureText(note).width, y + 18);
  });

  out.textContent = sp.kind === "planar"
    ? "max|w| is bit-exact zero — the reflection symmetry w → −w, not a small number.\n"
      + "This is the orbit model #9 can draw. The other two are what it structurally cannot."
    : "The helix is an EXACT solution of the scheme, so it is round from the first frame — no\n"
      + "growth needed. ψ (the static stretch) is NONZERO and held: v does not move, it leans.";
}

// ── sympathetic / coupled strings: the panel energy cannot see ───────────────────────────────
// Normal: BOTH bridge traces — the antisymmetric mode pins w_b at zero (the discriminating oracle),
// the symmetric mode swings it. A flat zero alone reads as "broken", so the contrast is the point.
// Transfer: the per-string energy fractions — string A (plucked) drains into a tuned neighbour B.
function drawSympathetic(sp) {
  const g = partialsCv.getContext("2d");
  const W = partialsCv.width, H = partialsCv.height, pad = 26;
  g.clearRect(0, 0, W, H);
  const out = $("partials-readout");
  const t = sp.time, tmax = t[t.length - 1] || 1;
  const num = (a) => a.map((x) => (x == null ? 0 : x));
  const xat = (i) => pad + (t[i] / tmax) * (W - pad - 8);
  g.strokeStyle = "#2a3340"; g.lineWidth = 1;
  g.strokeRect(pad, 8, W - pad - 8, H - 24);
  g.font = "10px ui-monospace, monospace";

  if (sp.regime === "normal") {
    const anti = num(sp.wb_anti), sym = num(sp.wb_sym);
    const vmax = Math.max(1e-30, ...sym.map(Math.abs), ...anti.map(Math.abs));
    const mid = 8 + (H - 24) / 2;                    // signed trace: 0 at the vertical centre
    g.strokeStyle = "#3a4452"; g.setLineDash([3, 3]);
    g.beginPath(); g.moveTo(pad, mid); g.lineTo(W - 8, mid); g.stroke(); g.setLineDash([]);
    const plot = (arr, colour, wdt) => {
      g.strokeStyle = colour; g.lineWidth = wdt; g.beginPath();
      for (let i = 0; i < arr.length; i++) {
        const y = mid - (arr[i] / vmax) * ((H - 24) / 2 - 4);
        if (i === 0) g.moveTo(xat(i), y); else g.lineTo(xat(i), y);
      }
      g.stroke();
    };
    plot(sym, "#ff8f4c", 1.5);                       // symmetric: swings
    plot(anti, "#4cc2ff", 2.5);                      // antisymmetric: dead flat on the axis
    g.fillStyle = "#4cc2ff"; g.fillText("antisym w_b", pad + 6, 20);
    g.fillStyle = "#ff8f4c"; g.fillText("symmetric w_b", pad + 92, 20);
    g.fillStyle = "#8b98a8"; g.fillText(`${tmax.toFixed(2)} s`, W - 46, H - 6);
    out.textContent =
      `antisym max|w_b| = ${sp.anti_max === 0 ? "0.0 (bit-exact ✓)" : sp.anti_max.toExponential(2)}` +
      `  ·  symmetric max|w_b| = ${sp.sym_max.toExponential(2)} m\n` +
      `the antiphase pair rings on a mode the bridge cannot feel (E_body → ` +
      `${sp.body_frac_anti.toExponential(1)}); the symmetric pair loads the body to ` +
      `${(sp.body_frac_sym * 100).toFixed(0)}%. Energy conservation passes BOTH — this is the claim ` +
      `it cannot see.`;
    return;
  }

  if (sp.regime === "weinreich") {
    // Log-y string-energy envelope: strike-ONE shows the fast prompt → slow/flat aftersound knee;
    // strike-BOTH (the pure symmetric mode) decays away single-slope with no aftersound — the
    // contrast that proves the strike-one plateau is the un-decaying antisymmetric mode, not a
    // floor. drawWhirl's log-axis is the precedent.
    const padL = 30, padB = 16, top = 10;
    const one = num(sp.env_one).map((x) => (x <= 0 ? 1e-300 : x));
    const both = num(sp.env_both).map((x) => (x <= 0 ? 1e-300 : x));
    const plotW = W - padL - 8, plotH = H - padB - top;
    g.clearRect(0, 0, W, H);
    const lo = -3, hi = Math.log10(1.5);            // fixed 1e-3 .. ~1.5 window (matches the sweep)
    const span = hi - lo;
    const px = (i) => padL + (t[i] / tmax) * plotW;
    const py = (v) => top + plotH - ((Math.log10(v) - lo) / span) * plotH;
    g.strokeStyle = "#2a3340"; g.lineWidth = 1; g.strokeRect(padL, top, plotW, plotH);
    g.strokeStyle = "rgba(139,152,168,.18)";
    for (let d = -3; d <= 0; d++) {
      const y = py(Math.pow(10, d));
      g.beginPath(); g.moveTo(padL, y); g.lineTo(padL + plotW, y); g.stroke();
      g.fillStyle = "#8b98a8"; g.font = "9px ui-monospace, monospace";
      g.fillText(`1e${d}`, 2, y + 3);
    }
    const line = (arr, colour, wdt) => {
      g.strokeStyle = colour; g.lineWidth = wdt; g.beginPath();
      for (let i = 0; i < arr.length; i++) {
        const y = Math.max(top, Math.min(top + plotH, py(arr[i])));
        if (i === 0) g.moveTo(px(i), y); else g.lineTo(px(i), y);
      }
      g.stroke();
    };
    line(both, "#ff8f4c", 1.5);                     // strike-both: decays away (no aftersound)
    line(one, "#4cc2ff", 2.5);                      // strike-one: prompt → aftersound plateau
    g.font = "10px ui-monospace, monospace";
    g.fillStyle = "#4cc2ff"; g.fillText("strike one", padL + 6, top + 12);
    g.fillStyle = "#ff8f4c"; g.fillText("strike both", padL + 78, top + 12);
    g.fillStyle = "#8b98a8"; g.fillText(`${tmax.toFixed(2)} s`, W - 46, H - 5);
    const flat = sp.sigma_zero;
    const ratio = sp.aftersound_rate > 1e-6 ? (sp.prompt_rate / sp.aftersound_rate) : Infinity;
    out.textContent = flat
      ? `body loss = 0 → nothing decays: both curves ring on flat (the energy verdict is the ` +
        `conservation-drift check, not passivity).\nRaise body loss to load the symmetric mode and ` +
        `split the decay into a fast prompt + a lingering aftersound.`
      : `prompt ${sp.prompt_rate.toFixed(1)} s⁻¹ → aftersound ${sp.aftersound_rate.toFixed(2)} s⁻¹` +
        `${ratio === Infinity ? "" : ` (${ratio.toFixed(0)}× slower)`}: strike-one keeps ` +
        `${(sp.floor_one * 100).toFixed(0)}% of its energy in the un-decaying mode, while strike-both ` +
        `falls to ${(sp.both_final * 100).toFixed(0)}%.\n` +
        (sp.detune < 0.005
          ? `at unison the aftersound is EXACTLY lossless (the antisymmetric mode is bit-exactly ` +
            `bridge-decoupled — the normal-mode oracle). Dial detune up for a finite piano aftersound.`
          : `~${(sp.detune * 100).toFixed(0)} cents mistuned: the antisymmetric mode loads the bridge ` +
            `a little, so the aftersound decays slowly instead of flat — the real piano unison.`);
    return;
  }

  const f0 = num(sp.frac0), f1 = num(sp.frac1);
  const base = H - 18, top = 22;                      // fraction axis: 0 at bottom, 1 at top
  const plot = (arr, colour, wdt) => {
    g.strokeStyle = colour; g.lineWidth = wdt; g.beginPath();
    for (let i = 0; i < arr.length; i++) {
      const y = base - Math.max(0, Math.min(1, arr[i])) * (base - top);
      if (i === 0) g.moveTo(xat(i), y); else g.lineTo(xat(i), y);
    }
    g.stroke();
  };
  plot(f0, "#4cc2ff", 1.5);                          // string A (plucked): drains
  plot(f1, "#ff8f4c", 2.5);                          // string B (neighbour): rings up
  g.fillStyle = "#4cc2ff"; g.fillText("A (plucked)", pad + 6, 18);
  g.fillStyle = "#ff8f4c"; g.fillText("B (neighbour)", pad + 86, 18);
  g.fillStyle = "#8b98a8"; g.fillText(`${tmax.toFixed(2)} s`, W - 46, H - 6);
  out.textContent = sp.tuned
    ? `unison: the neighbour drains ${(sp.peak_neighbour * 100).toFixed(0)}% of the total energy — ` +
      `near-complete coupled-oscillator exchange.\nDetune it and watch the transfer collapse ` +
      `(the coupling is frequency-selective at K = ${sp.K} N/m).`
    : `Δ ${sp.detune.toFixed(1)} semitones off unison: the neighbour peaks at only ` +
      `${(sp.peak_neighbour * 100).toFixed(0)}%.\noff the partial the bridge barely couples the two ` +
      `— why a sympathetic string lights up for the right note and no other.`;
}

// ── partials diagnostic ─────────────────────────────────────────────────────────────────────
function drawPartials() {
  const g = partialsCv.getContext("2d");
  const W = partialsCv.width, H = partialsCv.height, pad = 24, mid = H / 2;
  g.clearRect(0, 0, W, H);
  const out = $("partials-readout");
  const pa = payload && payload.meta.partials;
  if (!pa) { out.textContent = "no partials in band"; return; }

  const cents = pa.cents;
  let worst = 0;
  cents.forEach((c) => { if (c != null && Math.abs(c) > worst) worst = Math.abs(c); });
  const scale = Math.max(worst, 1);

  g.strokeStyle = "#2a3340"; g.lineWidth = 1;
  g.beginPath(); g.moveTo(pad, mid); g.lineTo(W - 8, mid); g.stroke();   // 0-cent line
  const n = cents.length, bw = (W - pad - 16) / n;
  for (let i = 0; i < n; i++) {
    const c = cents[i];
    if (c == null) continue;
    const h = (c / scale) * (H / 2 - 14);
    const x = pad + i * bw + 2;
    g.fillStyle = Math.abs(c) < 1 ? "#4cc2ff" : "#ffcf5c";
    g.fillRect(x, mid - Math.max(h, 0), bw - 4, Math.abs(h) || 1);
    if (h < 0) g.fillRect(x, mid, bw - 4, -h);
  }
  g.fillStyle = "#8b98a8"; g.font = "10px ui-monospace, monospace";
  g.fillText("+cents", 2, 14); g.fillText("−cents", 2, H - 6);

  out.textContent =
    `f₁ = ${payload.meta.f1.toFixed(2)} Hz   ·   partials shown: ${n}\n` +
    `worst error = ${worst.toFixed(3)} cents`;
}

// ── audio ───────────────────────────────────────────────────────────────────────────────────
function playAudio() {
  if (!audioSamples) return;
  if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  if (audioCtx.state === "suspended") audioCtx.resume();
  if (!audioBuf) {
    audioBuf = audioCtx.createBuffer(1, audioSamples.length, audioFs);
    audioBuf.copyToChannel(audioSamples, 0);
  }
  if (audioSrc) { try { audioSrc.stop(); } catch (e) { /* already stopped */ } }
  audioSrc = audioCtx.createBufferSource();
  audioSrc.buffer = audioBuf;
  audioSrc.loop = loopAudio.checked;
  audioSrc.connect(audioCtx.destination);
  audioSrc.start();
}

// ── overlay + misc ──────────────────────────────────────────────────────────────────────────
function showOverlay(msg) { canvasMsg.textContent = msg; canvasMsg.hidden = false; }
function hideOverlay() { canvasMsg.hidden = true; }
function fmt(x) { return Math.round(x).toLocaleString(); }

// ── transport wiring ────────────────────────────────────────────────────────────────────────
animToggle.addEventListener("click", () => {
  animPlaying = !animPlaying;
  animToggle.textContent = animPlaying ? "⏸ Pause" : "▶ Play";
  if (animPlaying) animStart = 0;     // re-anchor on resume
});
speedInput.addEventListener("input", () => {
  speed = +speedInput.value;
  speedVal.textContent = speed.toFixed(3) + "×";
  animStart = 0;                       // re-anchor so the new speed is continuous
});
scrub.addEventListener("input", () => { scrubbing = true; currentFrame = +scrub.value; });
scrub.addEventListener("change", () => { scrubbing = false; animStart = 0; });
playAudioBtn.addEventListener("click", playAudio);
renderBtn.addEventListener("click", render);

// ── boot ────────────────────────────────────────────────────────────────────────────────────
// Optional deep-link: ?model=membrane&domain=circle preselects the model/domain before the first
// render (also what the headless-browser verification drives).
function applyUrlParams() {
  const q = new URLSearchParams(location.search);
  const m = q.get("model");
  if (m && [...modelSel.options].some((o) => o.value === m)) modelSel.value = m;
  populateDomain(modelSel.value);          // options depend on the (possibly just-set) model
  const d = q.get("domain");
  if (d && domainSel && [...domainSel.options].some((o) => o.value === d)) domainSel.value = d;
}

buildSliders();
applyUrlParams();
applyModelRanges();
updateVisibility();
updateLambdaHint();
speedVal.textContent = speed.toFixed(3) + "×";
requestAnimationFrame(tick);
render();                              // auto-render the defaults on load
