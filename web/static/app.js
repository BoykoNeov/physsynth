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
  plate: { N: { max: 80, val: 60 }, kappa: { min: 2, max: 80, step: 0.5, fixed: 1, val: 20 },
           rho: { min: 0.001, max: 0.02, step: 0.0005, fixed: 4, val: 0.005, unit: "kg/m²" },
           Lx: { val: 1.0 }, Ly: { val: 1.0 }, audio_duration: { max: 2, val: 1 } },
  vk: { N: { min: 8, max: 32, val: 20 },
        rho: { min: 2000, max: 12000, step: 100, fixed: 0, val: 7800, unit: "kg/m³" },
        Lx: { val: 0.3 }, Ly: { val: 0.3 }, audio_duration: { max: 1, val: 0.5 } },
  // `amplitude` is shown only for the tension string, but gatherParams sends every slider — so it
  // must reset to the linear string path's historical 1e-3 on switch, or those models would silently
  // re-render at the tension default (a pure scale for a linear model, but not bit-for-bit).
  // Every param a model above re-ranges must be reset here to its index.html default, or it leaks
  // into the next model on switch (gatherParams sends every slider, hidden ones included). sigma0/
  // sigma1/pickup_position joined `amplitude` when the bow arrived: the bow needs sigma1 = 0.05,
  // which is 25x the damped string's default AND outside its own slider max of 0.01, so without
  // the reset a bow → damped switch would silently render a wildly over-damped string on a stale
  // range. (This also fixes the same leak tension's sigma0 = 0 already had.)
  _default: { N: { min: 16, max: 512 }, lambda: { max: 2.0, val: 1.0 },
              kappa: { min: 0, max: 8, step: 0.05, fixed: 2, val: 1.0 },
              rho: { min: 0.001, max: 0.02, step: 0.0005, fixed: 4, val: 0.005, unit: "kg/m²" },
              amplitude: { val: 0.001 },
              sigma0: { min: 0, max: 20, step: 0.1, val: 1.0 },
              sigma1: { min: 0, max: 0.01, step: 0.0001, fixed: 4, val: 0.002 },
              pickup_position: { val: 0.1 },
              audio_duration: { max: 6, val: 2 } },
};

// Secondary select repurposed per model: geometry (membrane) vs boundary (plate / von Kármán).
const DOMAIN_MODELS = ["membrane", "plate", "vk"];
const DOMAIN_OPTS = {
  membrane: [["circle", "Circle (drumhead)"], ["rectangle", "Rectangle"]],
  plate: [["supported", "Simply-supported (#5)"], ["free", "Free edge — Chladni (#5b)"]],
  vk: [["supported", "Supported gong (#6)"], ["free", "Free-edge cymbal (#6)"]],
};

const sliders = {};      // param -> <input>
const updaters = {};     // param -> fn() that refreshes its value label
const fixedOf = {};      // param -> decimal places for the value label (re-rangeable)
const scaleOf = {};      // param -> multiplier applied in gatherParams (E in GPa, e in mm)
const unitOf = {};       // param -> value-label unit suffix (re-rangeable: ρ is areal vs volumetric)
let payload = null;
let dims = 1;            // 1 = string polyline, 2 = membrane heatmap
let frames = null, nFrames = 0, width = 0, fieldAmp = 1, animDt = 1e-3;
let gridNx = 0, gridNy = 0, maskData = null, gridMeta = null, heatCv = null;
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
  if (lbl) lbl.textContent = model === "membrane" ? "Domain" : "Boundary";
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
}

// Re-range sliders to the current model's caps/defaults (see MODEL_RANGES). Merges _default with the
// model spec (model wins); applies min/max/step/fixed, (re)sets `val` when the spec gives one, and
// always clamps the current value into range. Run on model switch only, so resets are intentional.
function applyModelRanges() {
  const spec = Object.assign({}, MODEL_RANGES._default, MODEL_RANGES[modelSel.value] || {});
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
  } else if (m === "membrane") {
    const lam = param("lambda");
    hint.textContent = `λ = c·k/h = ${lam.toFixed(2)}  (2D CFL: λ ≤ 1/√2 ≈ 0.71; no λ is `
      + `dispersionless)`;
    hint.style.color = lam > 0.708 ? "var(--bad)" : "var(--muted)";
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
    } else {
      lossHint.textContent = "";
    }
    lossHint.style.color = "var(--muted)";
  }
}

function onControlChange(name) {
  if (name === "lambda" || name === "mu" || name === "fs") updateLambdaHint();
  if (name === "amplitude" || name === "EA" || name === "T" || name === "L") updateLambdaHint();
  if (name === "sigma0" || name === "sigma1") updateLambdaHint();
  if (name === "bow_position" || name === "v_bow") updateLambdaHint();
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
  updateVisibility();
  updateLambdaHint();
  scheduleAuto();
});
const nonlinearChk = $("nonlinear");
if (nonlinearChk) nonlinearChk.addEventListener("change", scheduleAuto);

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
    const scheme = data.lambda !== undefined ? `λ ${data.lambda}`
      : data.mu !== undefined ? `μ ${data.mu}`
      : (data.nonlinear === false ? "linear" : "nonlinear");
    setStatus(
      `ok — fs_sim ${fmt(data.fs_sim)} Hz · ${scheme} · ` +
      `${data.frames.n_frames} frames · ${data.audio.n} audio samples`, "");
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
  audioSamples = b64ToFloat32(data.audio.b64);
  audioFs = data.audio.fs;
  audioBuf = null;                       // rebuilt lazily on first Play
  currentFrame = 0; animStart = 0;
  scrub.max = Math.max(0, nFrames - 1);
  scrub.value = 0;
  hideOverlay();
  drawEnergy();
  drawDiagnostics();
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

function tick(ts) {
  if (frames && nFrames > 0) {
    if (animPlaying && !scrubbing) {
      if (animStart === 0) animStart = ts - (currentFrame * animDt / speed) * 1000;
      const physElapsed = ((ts - animStart) / 1000) * speed;
      currentFrame = Math.floor(physElapsed / animDt) % nFrames;
      scrub.value = currentFrame;
    }
    (dims === 2 ? drawHeatmap : drawString)(currentFrame);
  }
  requestAnimationFrame(tick);
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
  const vmax = Math.max(...v) || 1;

  g.strokeStyle = "#2a3340"; g.lineWidth = 1;
  g.strokeRect(pad, 8, W - pad - 8, H - pad - 8);
  g.strokeStyle = "#5ad17a"; g.lineWidth = 2; g.beginPath();
  for (let i = 0; i < v.length; i++) {
    const x = pad + (t[i] / tmax) * (W - pad - 8);
    const y = (H - pad) - (v[i] / vmax) * (H - pad - 8);
    if (i === 0) g.moveTo(x, y); else g.lineTo(x, y);
  }
  g.stroke();
  g.fillStyle = "#8b98a8"; g.font = "10px ui-monospace, monospace";
  g.fillText("E", 6, 16); g.fillText(`${tmax.toFixed(2)} s`, W - 48, H - 8);

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
    out.textContent =
      `lossless · drift max|Eⁿ−E⁰|/E⁰ = ${e.lossless.drift.toExponential(2)}${convNote}\n` +
      `tol ${e.lossless.tol.toExponential(0)}  →  ${ok ? "PASS ✓" : "FAIL ✗"}`;
  } else {
    const mono = e.lossy.monotone;
    badge.textContent = mono ? "passive" : "NON-MONOTONE";
    badge.className = "badge " + (mono ? "good" : "bad");
    const meas = e.lossy.measured_2sigma;
    out.textContent =
      `lossy · energy monotone decrease: ${mono ? "yes ✓" : "NO ✗"}${convNote}\n` +
      `measured 2σ = ${meas == null ? "—" : meas.toFixed(3)} s⁻¹` +
      `  (flat-loss oracle ${e.lossy.oracle_2sigma.toFixed(3)})`;
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
