"use strict";
/* Web viewer front-end (architecture B). Talks to the local Python backend: POST /simulate -> the
 * core recomputes offline -> we animate the string (slow-mo, decoupled from audio per catch #2),
 * play the sound at its true rate (48 kHz, catch #1), and draw the energy + partials diagnostics
 * gated by loss (catch #4). Vanilla JS + Canvas2D, no framework. */

// ── element handles ─────────────────────────────────────────────────────────────────────────
const $ = (id) => document.getElementById(id);
const modelSel = $("model");
const renderBtn = $("render");
const autoRender = $("autorender");
const statusEl = $("status");
const stringCv = $("string");
const energyCv = $("energy");
const partialsCv = $("partials");
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
};

const sliders = {};      // param -> <input>
let payload = null;
let frames = null, nFrames = 0, width = 0, fieldAmp = 1, animDt = 1e-3;
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
    const fixed = d.fixed !== undefined ? +d.fixed : guessFixed(+d.step);
    const unit = d.unit ? " " + d.unit : "";
    el.innerHTML =
      `<div class="row"><span class="name">${LABELS[d.param] || d.param}</span>` +
      `<span class="value" id="v-${d.param}"></span></div>` +
      `<input type="range" id="s-${d.param}" min="${d.min}" max="${d.max}" ` +
      `step="${d.step}" value="${d.val}">`;
    const input = el.querySelector("input");
    const valEl = el.querySelector(".value");
    const update = () => { valEl.textContent = (+input.value).toFixed(fixed) + unit; };
    input.addEventListener("input", () => { update(); onControlChange(d.param); });
    update();
    sliders[d.param] = input;
  });
}

function param(name) { return sliders[name] ? +sliders[name].value : undefined; }

// ── model-dependent visibility + hints ──────────────────────────────────────────────────────
function updateVisibility() {
  const m = modelSel.value;
  document.querySelectorAll("[data-show]").forEach((el) => {
    el.hidden = !el.dataset.show.split(" ").includes(m);
  });
}

function updateLambdaHint() {
  const m = modelSel.value, lam = param("lambda");
  const hint = $("lambda-hint");
  if (m === "ideal" && lam > 1.0) {
    hint.textContent = "λ>1 breaks the explicit ideal string's CFL (will error). Stiff/damped allow it.";
    hint.style.color = "var(--bad)";
  } else {
    hint.textContent = m === "ideal"
      ? `λ = c·k/h = ${lam.toFixed(2)}  (1.0 = exact, dispersionless)`
      : `λ = ${lam.toFixed(2)}  (implicit scheme — no CFL limit)`;
    hint.style.color = "var(--muted)";
  }
}

function onControlChange(name) {
  if (name === "lambda") updateLambdaHint();
  scheduleAuto();
}

modelSel.addEventListener("change", () => {
  updateVisibility();
  updateLambdaHint();
  scheduleAuto();
});

function scheduleAuto() {
  if (!autoRender.checked) return;
  clearTimeout(autoTimer);
  setStatus("queued…", "busy");
  autoTimer = setTimeout(render, 400);
}

// ── networking ──────────────────────────────────────────────────────────────────────────────
function gatherParams() {
  const p = { model: modelSel.value };
  for (const k in sliders) p[k] = +sliders[k].value;
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
    setStatus(
      `ok — fs_sim ${fmt(data.fs_sim)} Hz · λ ${data.lambda} · ` +
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

function applyPayload(data) {
  payload = data;
  frames = b64ToFloat32(data.frames.b64);
  nFrames = data.frames.n_frames;
  width = data.frames.width;
  fieldAmp = data.field_amp || 1;
  animDt = data.anim_dt || 1e-3;
  audioSamples = b64ToFloat32(data.audio.b64);
  audioFs = data.audio.fs;
  audioBuf = null;                       // rebuilt lazily on first Play
  currentFrame = 0; animStart = 0;
  scrub.max = Math.max(0, nFrames - 1);
  scrub.value = 0;
  hideOverlay();
  drawEnergy();
  drawPartials();
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
    drawString(currentFrame);
  }
  requestAnimationFrame(tick);
}

// ── energy diagnostic ───────────────────────────────────────────────────────────────────────
function drawEnergy() {
  const g = energyCv.getContext("2d");
  const W = energyCv.width, H = energyCv.height, pad = 24;
  g.clearRect(0, 0, W, H);
  if (!payload) return;
  const e = payload.energy;
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
  if (e.sigma_is_zero) {
    const ok = e.lossless.pass;
    badge.textContent = ok ? "conserved" : "DRIFT";
    badge.className = "badge " + (ok ? "good" : "bad");
    out.textContent =
      `lossless · drift max|Eⁿ−E⁰|/E⁰ = ${e.lossless.drift.toExponential(2)}\n` +
      `tol ${e.lossless.tol.toExponential(0)}  →  ${ok ? "PASS ✓" : "FAIL ✗"}`;
  } else {
    const mono = e.lossy.monotone;
    badge.textContent = mono ? "passive" : "NON-MONOTONE";
    badge.className = "badge " + (mono ? "good" : "bad");
    const meas = e.lossy.measured_2sigma;
    out.textContent =
      `lossy · energy monotone decrease: ${mono ? "yes ✓" : "NO ✗"}\n` +
      `measured 2σ = ${meas == null ? "—" : meas.toFixed(3)} s⁻¹` +
      `  (flat-loss oracle ${e.lossy.oracle_2sigma.toFixed(3)})`;
  }
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
buildSliders();
updateVisibility();
updateLambdaHint();
speedVal.textContent = speed.toFixed(3) + "×";
requestAnimationFrame(tick);
render();                              // auto-render the defaults on load
