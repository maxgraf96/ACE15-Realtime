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
const enhanceFn   = nf("enhance");
const setDenoise  = nf("setDenoise");
const setChar     = nf("setCharacter");
const setEvolve   = nf("setEvolve");
const setDcw      = nf("setDcw");
const seekFn      = nf("seek");
const reconfigure = nf("reconfigure");
const setModel    = nf("setModel");
const setMetas    = nf("setMetas");
const play        = nf("play");
const pauseFn     = nf("pause");
const stop        = nf("stop");
const setBypassFn = nf("setBypass");
const setInputGainFn = nf("setInputGain");
const setMakeupFn = nf("setMakeup");
const startRealtimeFn = nf("startRealtime");
const stopRealtimeFn = nf("stopRealtime");
const openFile    = nf("openFile");

// ── DOM ──────────────────────────────────────────────────────────────
const $ = (id) => document.getElementById(id);
const statusEl = $("status"), statusText = $("status-text");
const sourceEl = $("source"), sourceEmpty = $("source-empty"), sourceLoaded = $("source-loaded");
const sourceLoading = $("source-loading"), loadName = $("loadname-text"), loadStage = $("loadstage-text");
const srcName = $("srcname-text"), srcDur = $("srcdur-text"), srcWave = $("srcwave");
const metaEdit = $("meta-edit"), bpmField = $("bpm-field"), keyField = $("key-field");
const meterEl = $("meter"), mBuf = $("m-buf"), mRegens = $("m-regens"), mLat = $("m-lat");
const promptEl = $("prompt"), promptClear = $("prompt-clear"), enhanceBtn = $("enhance-btn");
const denoiseEl = $("denoise"), denoiseVal = $("denoise-value");
const charEl = $("character"), charVal = $("character-value");
const stepsEl = $("steps"), stepsVal = $("steps-value");
const windowEl = $("window"), evolveEl = $("evolve-toggle");
const playBtn = $("play"), stopBtn = $("stop"), abEl = $("ab-toggle");
const srcModeBtn = $("srcmode-toggle"), sourceLive = $("source-live");
const inMeterFill = $("inmeter-fill"), liveBpm = $("live-bpm"), liveKey = $("live-key");
const dl = $("dl"), dlFill = $("dl-fill");
const errBanner = $("error-banner"), errText = $("error-text"), errDismiss = $("error-dismiss");
const filePicker = $("file-picker");

let loaded = false, styled = false, evolve = false, liveMode = false, bypass = false;
let playState = "stopped";   // "stopped" | "playing" | "paused"
const started = () => playState !== "stopped";   // producer running (play OR pause) -> use live setters
let playheadEl = null;
let trackDur = 0;   // seconds, for the m:ss position readout

function setStatus(t, cls) { statusText.textContent = t; statusEl.className = "status" + (cls ? " " + cls : ""); }
function refresh() {
  const ready = liveMode ? true : (loaded && styled);   // live needs no file, just the engine
  playBtn.disabled = !ready;
  stopBtn.disabled = !ready || playState === "stopped";
  abEl.disabled = !ready;
}
function setTransport(state) {   // drive the transport buttons + status
  playState = state;
  playBtn.classList.toggle("playing", state === "playing");
  if (state === "stopped") { if (playheadEl) playheadEl.style.left = "0%"; showPos(0); }
  refresh();
}
function fmtTime(s) { s = Math.max(0, Math.floor(s)); return Math.floor(s / 60) + ":" + String(s % 60).padStart(2, "0"); }
function showPos(frac) { if (trackDur) srcDur.textContent = fmtTime(frac * trackDur) + " / " + fmtTime(trackDur); }

// ── loading state (drop/model-switch → ready) ────────────────────────
function setStage(t) { loadStage.textContent = t; }
function showLoading(name) {
  loadName.textContent = name; setStage("loading…");
  sourceEl.classList.remove("empty");
  sourceEmpty.hidden = true; sourceLoaded.hidden = true; meterEl.hidden = true;
  sourceLoading.hidden = false;
}
function hideLoading() { sourceLoading.hidden = true; }

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
  el.style.cursor = "pointer";
  playheadEl = el.querySelector(".playhead");
}

// ── scrub: click/drag the waveform to jump through the track ──────────
// Drag moves the playhead visually; we seek on release (each seek triggers a
// regen, so we don't flood the engine while dragging). After a seek we briefly
// ignore engine playhead updates so the head doesn't snap back before the
// producer catches up at the new position.
let scrubbing = false, scrubHoldUntil = 0;
function scrubFrac(e) {
  const r = srcWave.getBoundingClientRect();
  return Math.max(0, Math.min(1, (e.clientX - r.left) / r.width));
}
function showHead(frac) { if (playheadEl) playheadEl.style.left = (frac * 100).toFixed(2) + "%"; showPos(frac); }
srcWave.addEventListener("pointerdown", (e) => {
  if (!loaded) return;
  scrubbing = true; srcWave.setPointerCapture(e.pointerId); showHead(scrubFrac(e)); e.preventDefault();
});
srcWave.addEventListener("pointermove", (e) => { if (scrubbing) showHead(scrubFrac(e)); });
srcWave.addEventListener("pointerup", (e) => {
  if (!scrubbing) return;
  const f = scrubFrac(e); showHead(f); seekFn(f);
  scrubbing = false; scrubHoldUntil = performance.now() + 600;   // let the engine catch up
});

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
  showLoading(file.name);
  srcName.textContent = file.name;
  setStatus("loading track…", "accent");
  const ext = "." + (file.name.split(".").pop() || "wav").toLowerCase();
  const b64 = await fileToBase64(file);
  await uploadAudio(b64, ext);
}
sourceEmpty.addEventListener("click", () => { if (!liveMode) filePicker.click(); });
filePicker.addEventListener("change", () => { if (filePicker.files[0]) loadFile(filePicker.files[0]); });
document.addEventListener("dragover", (e) => { if (liveMode) return; e.preventDefault(); sourceEl.classList.add("drag"); });
document.addEventListener("dragleave", (e) => { if (e.relatedTarget === null) sourceEl.classList.remove("drag"); });
document.addEventListener("drop", (e) => {
  if (liveMode) return;                 // no file drops in live mode
  e.preventDefault(); sourceEl.classList.remove("drag");
  const f = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
  if (f) loadFile(f);
});

// ── controls ─────────────────────────────────────────────────────────
// Live edits use the QUEUED setters (applied in the engine's producer thread).
// setStyle re-encodes directly, so only call it when NOT playing.
function applyStyle() { setStyleFn(promptEl.value, parseFloat(denoiseEl.value), parseFloat(charEl.value)); }
function autogrow() { promptEl.style.height = "auto"; promptEl.style.height = Math.min(promptEl.scrollHeight, 118) + "px"; }

let promptTimer = null;
promptEl.addEventListener("input", () => {
  promptClear.hidden = !promptEl.value; autogrow();
  clearTimeout(promptTimer);
  promptTimer = setTimeout(() => {
    if (!styled) { applyStyle(); return; }
    if (started()) setPrompt(promptEl.value); else applyStyle();
  }, 450);
});
promptClear.addEventListener("click", () => { promptEl.value = ""; promptClear.hidden = true; autogrow(); if (started()) setPrompt(""); else applyStyle(); });
promptClear.hidden = !promptEl.value; autogrow();   // reflect the default prompt

// ✨ Enhance: send the short style to the 5Hz LM; it replies with a rich caption
// (engineEvent "enhanced"). Works with no track loaded. First use downloads the LM.
let enhancing = false;
enhanceBtn.addEventListener("click", () => {
  const v = promptEl.value.trim();
  if (!v || enhancing) return;
  enhancing = true; enhanceBtn.disabled = true; enhanceBtn.textContent = "Expanding…";
  setStatus("expanding prompt…", "accent");
  enhanceFn(v);
});
function endEnhance() { enhancing = false; enhanceBtn.disabled = false; enhanceBtn.textContent = "Expand with LM"; }

denoiseEl.addEventListener("input", () => denoiseVal.textContent = parseFloat(denoiseEl.value).toFixed(2));
denoiseEl.addEventListener("change", () => { if (started()) setDenoise(parseFloat(denoiseEl.value)); else if (styled) applyStyle(); });

charEl.addEventListener("input", () => charVal.textContent = parseFloat(charEl.value).toFixed(2));
charEl.addEventListener("change", () => { if (started()) setChar(parseFloat(charEl.value)); else if (styled) applyStyle(); });

stepsEl.addEventListener("input", () => stepsVal.textContent = stepsEl.value);
stepsEl.addEventListener("change", () => { if (started()) reconfigure(parseInt(stepsEl.value), parseFloat(windowEl.value)); });
windowEl.addEventListener("change", () => { if (started()) reconfigure(parseInt(stepsEl.value), parseFloat(windowEl.value)); });

evolveEl.addEventListener("click", () => {
  evolve = !evolve; evolveEl.textContent = evolve ? "Evolve" : "Coherent";
  evolveEl.classList.toggle("one-shot", evolve); setEvolve(evolve);
});

// DCW: opt-in wavelet-domain correction. Live setter (queued -> regen) when
// playing; otherwise re-style so the next handle is built in the new DCW state.
let dcw = false;  // DCW off by default (it runs hot in our regime; opt-in via the toggle)
const dcwEl = $("dcw-toggle");
dcwEl.addEventListener("click", () => {
  dcw = !dcw; dcwEl.textContent = dcw ? "DCW On" : "DCW Off";
  dcwEl.classList.toggle("one-shot", dcw);
  setDcw(dcw);                              // updates native state (+ live regen if running)
  if (!started() && styled) applyStyle();  // not running: rebuild the handle in the new DCW state
});

// Match tempo/key toggles (inject the bpm/key from the editable fields into the
// prompt Metas) + the editable BPM/Key fields themselves (auto-filled from
// analysis on "styled"; edits correct a wrong detection).
let sendBpm = true, sendKey = true;
const bpmToggle = $("bpm-toggle"), keyToggle = $("key-toggle");
function pushMetas() {
  setMetas(sendBpm, sendKey, bpmField.value.trim(), keyField.value.trim());  // flags + (edited) values
  if (!started() && styled) applyStyle();  // not running: re-encode now with the new state
}
bpmToggle.addEventListener("click", () => { sendBpm = !sendBpm; bpmToggle.classList.toggle("one-shot", sendBpm); pushMetas(); });
keyToggle.addEventListener("click", () => { sendKey = !sendKey; keyToggle.classList.toggle("one-shot", sendKey); pushMetas(); });
bpmField.addEventListener("change", pushMetas);
keyField.addEventListener("change", pushMetas);
// Live tempo/key: while running, update the cover Metas. (The bar grid is fixed at
// the BPM when Play was pressed; change tempo before Play to move the grid.)
const pushLiveMetas = () => { if (started()) setMetas(sendBpm, sendKey, liveBpm.value.trim(), liveKey.value.trim()); };
liveBpm.addEventListener("change", pushLiveMetas);
liveKey.addEventListener("change", pushLiveMetas);

let model = "quality";   // default = XL (best sound)
const modelEl = $("model-toggle");
modelEl.addEventListener("click", () => {
  model = model === "fast" ? "quality" : "fast";
  modelEl.textContent = model === "quality" ? "Quality" : "Fast";
  modelEl.classList.toggle("one-shot", model === "quality");
  styled = false; refresh();
  if (loaded) { showLoading(srcName.textContent || "track"); setStage("loading model…"); }  // reload = same wait as a drop
  setStatus(model === "quality" ? "loading Quality (XL)…" : "loading Fast (2B)…", "accent");
  setModel(model);   // C++ reloads the current track on the new model (downloads XL on first use)
});

// Transport. FILE mode: Play/Pause (pause keeps position) + Stop (reset). LIVE mode:
// Play = start listening+generating, Stop = stop (no pause concept live).
playBtn.addEventListener("click", () => {
  if (liveMode) { if (playState === "playing") stopLive(); else startLive(); return; }
  if (playState === "playing") { pauseFn(); setTransport("paused"); setStatus("paused"); }
  else { play(); setTransport("playing"); setStatus("playing", "accent"); }
});
stopBtn.addEventListener("click", () => {
  if (playState === "stopped") return;
  if (liveMode) stopLive(); else { stop(); setTransport("stopped"); setStatus("ready — press Play"); }
});

// ── Live mode (real-time input accompaniment) ──────────────────────────
function startLive() {
  setStatus("loading model…", "accent");
  startRealtimeFn(promptEl.value, parseFloat(denoiseEl.value), parseFloat(charEl.value),
                  liveBpm.value.trim(), liveKey.value.trim());
  setTransport("playing");
}
function stopLive() {
  stopRealtimeFn();
  setTransport("stopped"); setStatus("live mode — press Play");
  inMeterFill.style.width = "0%";
}
// Source mode toggle: File (cover a track) <-> Live (accompany the input bus).
function setSourceMode(live) {
  if (live === liveMode) return;
  if (started()) { if (liveMode) stopLive(); else { stop(); setTransport("stopped"); } }
  liveMode = live;
  document.body.classList.toggle("live-mode", live);
  srcModeBtn.textContent = live ? "Live" : "File";
  srcModeBtn.classList.toggle("ab-on", live);
  bypass = false; abEl.textContent = "Cover"; abEl.classList.remove("ab-on");
  stepsEl.disabled = live; windowEl.disabled = live;   // live window is bar-derived
  modelEl.disabled = live;   // pick the model in File mode before going Live (avoids a mid-live file reload)
  refresh();
  setStatus(live ? "live mode — press Play" : (loaded ? "ready — press Play" : "drop a track to begin"));
}
srcModeBtn.addEventListener("click", () => setSourceMode(!liveMode));

// A/B: hear the original (file) / your live input (live) vs the cover. Instant —
// the engine streams both pairs frame-aligned; this just flips which C++ outputs.
abEl.addEventListener("click", () => {
  bypass = !bypass;
  const src = liveMode ? "Input" : "Original";
  abEl.textContent = bypass ? src : "Cover";
  abEl.classList.toggle("ab-on", bypass);
  abEl.title = `A/B — instantly hear the ${src.toLowerCase()} instead of the cover (no regeneration). Currently: ${bypass ? src : "Cover"}.`;
  setBypassFn(bypass);
});

// VST-style gain knob: vertical click-drag (up = louder), double-click resets,
// wheel = fine step. `live` commits on every move (cheap controls); otherwise it
// commits only on release (the input-gain re-encode is heavy).
function makeKnob(id, min, max, def, fmt, onCommit, live) {
  const el = $(id);
  const ind = el.querySelector(".knob-ind");
  const valEl = el.querySelector(".knob-val");
  let val = def;
  const render = () => {
    const t = (val - min) / (max - min);
    ind.style.transform = `rotate(${(-135 + t * 270).toFixed(1)}deg)`;
    valEl.textContent = fmt(val);
  };
  const setVal = (v, commit) => {
    val = Math.max(min, Math.min(max, v)); render();
    if (commit) onCommit(val);
  };
  let dragging = false, startY = 0, startVal = 0;
  const sens = (max - min) / 160;   // px of vertical drag for the full range
  el.addEventListener("pointerdown", (e) => {
    dragging = true; startY = e.clientY; startVal = val;
    el.setPointerCapture(e.pointerId); el.classList.add("dragging"); e.preventDefault();
  });
  el.addEventListener("pointermove", (e) => {
    if (dragging) setVal(startVal + (startY - e.clientY) * sens, live);
  });
  const up = () => { if (dragging) { dragging = false; el.classList.remove("dragging"); onCommit(val); } };
  el.addEventListener("pointerup", up);
  el.addEventListener("pointercancel", up);
  el.addEventListener("dblclick", () => setVal(def, true));
  el.addEventListener("wheel", (e) => { e.preventDefault(); setVal(val + (e.deltaY < 0 ? 1 : -1) * (max - min) / 40, true); }, { passive: false });
  render();
  return { get: () => val, set: (v) => setVal(v, false) };
}
const dbFmt = (v) => `${v > 0 ? "+" : ""}${v.toFixed(1)} dB`;
// Input trim feeding the model (max 0 dB) — heavy re-encode, so commit on release only.
const inGain = makeKnob("ingain-knob", -20, 0, 0, dbFmt, (v) => setInputGainFn(v), false);
// Make-up gain just before audio-out (-20..+20 dB) — cheap, commit live while dragging.
const makeup = makeKnob("makeup-knob", -20, 20, 0, dbFmt, (v) => setMakeupFn(v), true);
$("reset").addEventListener("click", () => {
  denoiseEl.value = 0.7; denoiseVal.textContent = "0.70";
  charEl.value = 0; charVal.textContent = "0.00";
  if (evolve) evolveEl.click();
  if (started()) { setDenoise(0.7); setChar(0); } else if (styled) applyStyle();
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
    case "loading":   // engine stage updates while a track loads (drop → ready)
      if (ev.stage === "model") setStage("loading model…");
      else if (ev.stage === "analyze") setStage("analyzing track…");
      break;
    case "enhancing":
      setStatus("expanding prompt…", "accent");   // (first run also loads the LM)
      break;
    case "enhanced":
      endEnhance();
      promptEl.value = ev.caption || promptEl.value;
      promptClear.hidden = !promptEl.value; autogrow();
      if (started()) setPrompt(promptEl.value); else if (styled) applyStyle();   // use it now
      setStatus("prompt expanded", "accent");
      break;
    case "loaded":
      loaded = true;
      sourceEl.classList.remove("empty");
      trackDur = ev.duration || 0; showPos(0);   // "0:00 / m:ss"
      mountWaveform(srcWave, (ev.peaks || []).map(Number));   // into (still-hidden) sourceLoaded
      setStage("preparing…");
      setStatus("applying style…", "accent");
      applyStyle();                       // create the handle -> "styled"
      break;
    case "styled":
      styled = true;
      if (!sourceLoading.hidden) {         // first styled after a (re)load -> reveal + auto-fill detection
        bpmField.value = ev.bpm; keyField.value = ev.key;   // editable; override corrects a wrong detection
        metaEdit.hidden = false;
        hideLoading(); sourceEmpty.hidden = true; sourceLoaded.hidden = false; meterEl.hidden = false;
        if (inGain.get() !== 0) setInputGainFn(inGain.get());   // re-apply a non-default input trim (engine source is fresh)
      }
      if (playState === "playing") { play(); setTransport("playing"); setStatus("playing", "accent"); }   // resume after a (re)load (new engine starts stopped)
      else { setTransport("stopped"); setStatus("ready — press Play"); }
      break;
    case "playing": setStatus("playing", "accent"); break;
    case "paused":  setStatus("paused"); break;
    case "stopped": setStatus("ready — press Play"); break;
    case "live_started":
      meterEl.hidden = false;
      setTransport("playing"); setStatus(`live · ${ev.bpm} BPM · ${ev.key}`, "accent");
      break;
    case "live_stopped":
      setTransport("stopped"); setStatus("live mode — press Play");
      inMeterFill.style.width = "0%";
      break;
    case "input_level": {       // live input meter (sqrt for low-level visibility)
      const w = Math.min(100, Math.sqrt(Math.max(0, ev.peak || 0)) * 100);
      inMeterFill.style.width = w.toFixed(0) + "%";
      break;
    }
    case "stats":
      mBuf.textContent = (ev.buffered_s ?? 0).toFixed(1) + "s";
      mRegens.textContent = ev.regens ?? "–";
      mLat.textContent = (ev.worst_regen_ms ?? 0) + "ms";
      if (!scrubbing && performance.now() > scrubHoldUntil && typeof ev.progress === "number") {
        if (playheadEl) playheadEl.style.left = (ev.progress * 100).toFixed(2) + "%";
        showPos(ev.progress);   // m:ss readout follows the playhead
      }
      break;
    case "download_progress":
      dl.hidden = false; dlFill.style.width = (ev.pct || 0) + "%";
      setStage(`downloading models ${Math.round(ev.pct)}%`);
      setStatus(`downloading models ${Math.round(ev.pct)}%`, "accent");
      if (ev.pct >= 100) setTimeout(() => { dl.hidden = true; }, 800);
      break;
    case "error":
      errText.textContent = ev.msg || "error"; errBanner.hidden = false; setStatus("error", "error");
      endEnhance();                                   // re-enable Enhance if it failed
      hideLoading();                                  // don't strand the user on the spinner
      if (!loaded) { sourceEmpty.hidden = false; sourceEl.classList.add("empty"); }
      break;
  }
});
errDismiss.addEventListener("click", () => errBanner.hidden = true);

setStatus("drop a track");
