// ── JUCE 8 bridge ────────────────────────────────────────────────────
const promises = new Map();
let nextPromiseId = 0;
window.__JUCE__.backend.addEventListener("__juce__complete", ({ promiseId, result }) => {
  const h = promises.get(promiseId);
  if (!h) return;
  promises.delete(promiseId);
  h.resolve(result);
});
function nf(name) {
  return (...args) => new Promise((resolve, reject) => {
    const id = nextPromiseId++;
    promises.set(id, { resolve, reject });
    window.__JUCE__.backend.emitEvent("__juce__invoke", { name, params: args, resultId: id });
  });
}
const uploadAudio = nf("uploadAudio");
const setStyleFn  = nf("setStyle");
const setPrompt   = nf("setPrompt");
const setDenoise  = nf("setDenoise");
const setChar     = nf("setCharacter");
const setEvolve   = nf("setEvolve");
const reconfigure = nf("reconfigure");
const setModel    = nf("setModel");
const setMetas    = nf("setMetas");
const play        = nf("play");
const stop        = nf("stop");
const openFile    = nf("openFile");

// ── DOM ──────────────────────────────────────────────────────────────
const $ = (id) => document.getElementById(id);
const statusEl = $("status"), statusText = $("status-text");
const sourceEl = $("source"), sourceEmpty = $("source-empty"), sourceLoaded = $("source-loaded");
const srcName = $("srcname-text"), srcDur = $("srcdur-text"), bpmkey = $("bpmkey"), srcWave = $("srcwave");
const meterEl = $("meter"), mBuf = $("m-buf"), mRegens = $("m-regens"), mLat = $("m-lat");
const promptEl = $("prompt"), promptClear = $("prompt-clear");
const denoiseEl = $("denoise"), denoiseVal = $("denoise-value");
const charEl = $("character"), charVal = $("character-value");
const stepsEl = $("steps"), stepsVal = $("steps-value");
const windowEl = $("window"), evolveEl = $("evolve-toggle");
const playBtn = $("play"), playLabel = $("play-label");
const dl = $("dl"), dlFill = $("dl-fill");
const errBanner = $("error-banner"), errText = $("error-text"), errDismiss = $("error-dismiss");
const filePicker = $("file-picker");

let loaded = false, styled = false, playing = false, evolve = false;
let playheadEl = null;

function setStatus(t, cls) { statusText.textContent = t; statusEl.className = "status" + (cls ? " " + cls : ""); }
function refresh() { playBtn.disabled = !(loaded && styled); }

// ── waveform (adapted from plugin_morph) ─────────────────────────────
function mountWaveform(el, peaks) {
  const h = 64, bw = 2, gap = 1, n = peaks.length || 1;
  const W = n * (bw + gap) - gap, cy = h / 2;
  let d = `M 0 ${cy}`;
  for (let i = 0; i < n; i++) { const x = i * (bw + gap); d += ` L ${x} ${cy - peaks[i] * h * 0.46}`; }
  for (let i = n - 1; i >= 0; i--) { const x = i * (bw + gap); d += ` L ${x} ${cy + peaks[i] * h * 0.46}`; }
  el.innerHTML =
    `<svg viewBox="0 0 ${W} ${h}" preserveAspectRatio="none" style="width:100%;height:${h}px;display:block">
       <path d="${d}" fill="currentColor" fill-opacity="0.22"/>
       <path d="${d}" fill="currentColor" fill-opacity="0.7"/>
     </svg><div class="playhead" style="position:absolute;top:0;bottom:0;width:1px;background:var(--accent);left:0%"></div>`;
  el.style.position = "relative";
  playheadEl = el.querySelector(".playhead");
}

// ── load (drag-drop / picker) ────────────────────────────────────────
function fileToBase64(file) {
  return new Promise((res, rej) => {
    const r = new FileReader();
    r.onload = () => {
      const bytes = new Uint8Array(r.result); let s = "";
      const CH = 0x8000;
      for (let i = 0; i < bytes.length; i += CH) s += String.fromCharCode.apply(null, bytes.subarray(i, i + CH));
      res(btoa(s));
    };
    r.onerror = rej; r.readAsArrayBuffer(file);
  });
}
async function loadFile(file) {
  setStatus("loading track…", "accent");
  srcName.textContent = file.name;
  const ext = "." + (file.name.split(".").pop() || "wav").toLowerCase();
  const b64 = await fileToBase64(file);
  await uploadAudio(b64, ext);
}
sourceEmpty.addEventListener("click", () => filePicker.click());
filePicker.addEventListener("change", () => { if (filePicker.files[0]) loadFile(filePicker.files[0]); });
document.addEventListener("dragover", (e) => { e.preventDefault(); sourceEl.classList.add("drag"); });
document.addEventListener("dragleave", (e) => { if (e.relatedTarget === null) sourceEl.classList.remove("drag"); });
document.addEventListener("drop", (e) => {
  e.preventDefault(); sourceEl.classList.remove("drag");
  const f = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
  if (f) loadFile(f);
});

// ── controls ─────────────────────────────────────────────────────────
// Live edits use the QUEUED setters (applied in the engine's producer thread).
// setStyle re-encodes directly, so only call it when NOT playing.
function applyStyle() { setStyleFn(promptEl.value, parseFloat(denoiseEl.value), parseFloat(charEl.value)); }

let promptTimer = null;
promptEl.addEventListener("input", () => {
  promptClear.hidden = !promptEl.value;
  clearTimeout(promptTimer);
  promptTimer = setTimeout(() => {
    if (!styled) { applyStyle(); return; }
    if (playing) setPrompt(promptEl.value); else applyStyle();
  }, 450);
});
promptClear.addEventListener("click", () => { promptEl.value = ""; promptClear.hidden = true; if (playing) setPrompt(""); else applyStyle(); });

denoiseEl.addEventListener("input", () => denoiseVal.textContent = parseFloat(denoiseEl.value).toFixed(2));
denoiseEl.addEventListener("change", () => { if (playing) setDenoise(parseFloat(denoiseEl.value)); else if (styled) applyStyle(); });

charEl.addEventListener("input", () => charVal.textContent = parseFloat(charEl.value).toFixed(2));
charEl.addEventListener("change", () => { if (playing) setChar(parseFloat(charEl.value)); else if (styled) applyStyle(); });

stepsEl.addEventListener("input", () => stepsVal.textContent = stepsEl.value);
stepsEl.addEventListener("change", () => { if (playing) reconfigure(parseInt(stepsEl.value), parseFloat(windowEl.value)); });
windowEl.addEventListener("change", () => { if (playing) reconfigure(parseInt(stepsEl.value), parseFloat(windowEl.value)); });

evolveEl.addEventListener("click", () => {
  evolve = !evolve; evolveEl.textContent = evolve ? "Evolve" : "Coherent";
  evolveEl.classList.toggle("one-shot", evolve); setEvolve(evolve);
});

// Match tempo/key toggles (inject detected bpm/key into the prompt Metas)
let sendBpm = true, sendKey = true;
const bpmToggle = $("bpm-toggle"), keyToggle = $("key-toggle");
function pushMetas() {
  setMetas(sendBpm, sendKey);            // syncs native state + live re-encode if playing
  if (!playing && styled) applyStyle();  // not playing: re-encode now with the new flags
}
bpmToggle.addEventListener("click", () => { sendBpm = !sendBpm; bpmToggle.classList.toggle("one-shot", sendBpm); pushMetas(); });
keyToggle.addEventListener("click", () => { sendKey = !sendKey; keyToggle.classList.toggle("one-shot", sendKey); pushMetas(); });

let model = "quality";   // default = XL (best sound)
const modelEl = $("model-toggle");
modelEl.addEventListener("click", () => {
  model = model === "fast" ? "quality" : "fast";
  modelEl.textContent = model === "quality" ? "Quality" : "Fast";
  modelEl.classList.toggle("one-shot", model === "quality");
  styled = false; refresh();
  setStatus(model === "quality" ? "loading Quality (XL)…" : "loading Fast (2B)…", "accent");
  setModel(model);   // C++ reloads the current track on the new model (downloads XL on first use)
});

playBtn.addEventListener("click", () => {
  if (!playing) { play(); playing = true; playLabel.textContent = "Stop"; playBtn.classList.add("playing"); setStatus("playing", "accent"); }
  else { stop(); playing = false; playLabel.textContent = "Play"; playBtn.classList.remove("playing"); setStatus("stopped"); }
});
$("reset").addEventListener("click", () => {
  denoiseEl.value = 0.7; denoiseVal.textContent = "0.70";
  charEl.value = 0; charVal.textContent = "0.00";
  if (evolve) evolveEl.click();
  if (playing) { setDenoise(0.7); setChar(0); } else if (styled) applyStyle();
});
document.addEventListener("keydown", (e) => {
  if (e.code === "Space" && document.activeElement !== promptEl && document.activeElement !== windowEl) {
    e.preventDefault(); if (!playBtn.disabled) playBtn.click();
  }
});

// ── engine events ────────────────────────────────────────────────────
window.__JUCE__.backend.addEventListener("engineEvent", (ev) => {
  if (!ev || typeof ev !== "object") return;
  switch (ev.event) {
    case "loaded":
      loaded = true;
      sourceEl.classList.remove("empty"); sourceEmpty.hidden = true; sourceLoaded.hidden = false; meterEl.hidden = false;
      srcDur.textContent = ev.duration ? ev.duration.toFixed(1) + "s" : "";
      mountWaveform(srcWave, (ev.peaks || []).map(Number));
      setStatus("applying style…", "accent");
      applyStyle();                       // create the handle -> "styled"
      break;
    case "styled":
      styled = true; bpmkey.textContent = `${ev.bpm} BPM · ${ev.key}`;
      refresh();
      if (playing) play();   // resume after a model/track reload (new engine starts stopped)
      setStatus(playing ? "playing" : "ready — press Play");
      break;
    case "playing": setStatus("playing", "accent"); break;
    case "stats":
      mBuf.textContent = (ev.buffered_s ?? 0).toFixed(1) + "s";
      mRegens.textContent = ev.regens ?? "–";
      mLat.textContent = (ev.worst_regen_ms ?? 0) + "ms";
      if (playheadEl && typeof ev.progress === "number") playheadEl.style.left = (ev.progress * 100).toFixed(2) + "%";
      break;
    case "download_progress":
      dl.hidden = false; dlFill.style.width = (ev.pct || 0) + "%";
      setStatus(`downloading models ${Math.round(ev.pct)}%`, "accent");
      if (ev.pct >= 100) setTimeout(() => { dl.hidden = true; }, 800);
      break;
    case "error": errText.textContent = ev.msg || "error"; errBanner.hidden = false; setStatus("error", "error"); break;
  }
});
errDismiss.addEventListener("click", () => errBanner.hidden = true);

setStatus("drop a track");
