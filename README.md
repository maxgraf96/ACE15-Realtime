# ACE15-Realtime

Real-time, text-prompt **style-transfer ("cover")** for Apple Silicon: drop a track,
type a Style, and hear a **structure-preserving remix** stream out live — the source's
melody/rhythm/harmony kept, the timbre/genre swapped to match the prompt, with the
Style restylable on the fly (~1–2 s).

It ports **ACE-Step 1.5**'s cover capability + **DEMON**'s streaming engine off
CUDA/TensorRT to **Metal (MPS)**, and ships as a single **standalone macOS app** (JUCE
WebView UI + a bundled Python engine spawned internally — one download, no separate
install).

Target/dev machine: Apple **M4 Max, 48 GB**. macOS / Apple Silicon only. No CUDA/TensorRT.

---

## Status

**Working end-to-end:** a standalone app that drops a track → live restylable cover,
real-time on the M4 Max, with selectable Fast (2B) / Quality (XL) models.

| Phase | What | Status |
|---|---|---|
| 0 | Gating benchmark (eager DiT forward vs real-time budget) | ✅ **GO** |
| 1 | Headless streaming cover engine (JIT, real-time) | ✅ done + tuned |
| 2a/2b | Real-time producer ring + sidecar IPC server | ✅ done |
| 2c | JUCE standalone app (ported UI, IPC client, model toggle) | ✅ working |
| — | XL model evaluation + integration (default Quality) | ✅ done |
| 3 | Distributable packaging (bundle Python, code-sign/notarize) | ⏳ next |

Per-phase findings: [`notes/`](notes/) (`phase0_findings.md`, `phase1_findings.md`,
`phase2_app.md`, `prompting.md`).

---

## How it works

The shipped app is one process that **spawns the engine internally** and talks to it
over a local socket:

```
ACE15 Realtime.app (JUCE / C++)                 bundled Python engine (sidecar)
┌───────────────────────────────┐   framed TCP  ┌──────────────────────────────┐
│ WebView UI (Resources/)        │  127.0.0.1    │ sidecar/server.py            │
│ PluginEditor  ── native fns ───┼── CONTROL ──▶ │  RealtimeCover (producer ring)│
│ PluginProcessor (audio cb)     │ ◀── AUDIO ────┤   JITCover (lazy tile decode) │
│   IpcClient (jitter ring)──────┤ ◀── EVENT ────┤    Session → ACE-Step / MPS   │
└───────────────────────────────┘               └──────────────────────────────┘
```

**The cover mechanism (why this works where plain img2img doesn't):** ACE-Step is a
*cover* model with a first-class structure input — `context_latents` (a lossy ~5 Hz
FSQ tokenize→detokenize of the source latent) concatenated to the noisy latent every
denoise step. Structure (from `context_latents`) and style (from text cross-attention)
are independent dials. Per-tick hot path = **one DiT forward + windowed VAE decode**;
everything else (encode source, extract hints, encode text) is per-load.

**Real-time architecture (pinned-prefix rolling):** a background producer keeps one
**continuous latent** and extends it in chunks — each chunk's prefix is *pinned*
(latent-domain inpainting) to the already-generated tail, so the model denoises the
new frames to *continue* the real past. The overlap is identical → **no window seams**
(this replaced an earlier windowed+crossfade scheme). The committed latent is decoded
lazily (overlap-discard tiles, full VAE context) into a PCM ring that the audio
callback drains. Control changes apply to new chunks (smooth, pinned transitions).

---

## Quick start

```bash
# 1. Python engine env (3.11; arm64 torch, no CUDA)
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python torch torchaudio torchvision \
  "transformers>=4.51,<4.58" diffusers accelerate safetensors huggingface_hub \
  einops scipy soundfile numpy loguru toml vector-quantize-pytorch torchao \
  soxr laion-clap \          # soxr: key detection · laion-clap: dev style metric (optional)
  pytorch_wavelets PyWavelets  # DCW wavelet-domain correction (optional; toggle is off without it)

# 2. Build the app (JUCE 8 fetched at configure; first build ~minutes)
cmake -S app -B app/build -DCMAKE_BUILD_TYPE=Release && cmake --build app/build -j8

# 3. Run (app auto-spawns the sidecar via the project .venv)
./scripts/run_app.sh
```

On first track load it downloads the models (Fast ~6 GB; Quality/XL +20 GB) with a
progress bar, cached at `~/.daydream-scope/models/demon/checkpoints`. (The 5 Hz LM and
MelBandRoFormer are intentionally **not** downloaded.)

**Headless CLIs** (no app): `cli/jit_cover.py` (low-latency JIT + live control),
`cli/stream_cover.py`, `cli/offline_cover.py` (cover + structure metrics),
`cli/realtime_test.py` (1× real-time drain test), `cli/make_audition.py`.

---

## Using the app

Drop a track (or browse) → set a **Style** → **Play**. While a track loads (and on a
model switch) the source panel shows an animated bar with the stage (loading model →
analyzing → preparing) until it's ready. Edit Style live and it restyles in ~1–2 s. Detected **BPM · Key** show next to the filename; a live meter shows
buffer / regens / worst-regen. **Click or drag the waveform to scrub** — playback
jumps to that position (the playhead follows; audio re-arrives within ~lookahead).

| Control | Effect |
|---|---|
| **Style** | Prompt (live). Style/instruments/timbre/era — comma-separated keywords. |
| **Expand with LM** | Rewrites a short style ("tech house") into a rich, model-aligned caption via ACE-Step's 5Hz LM (`acestep-5Hz-lm-0.6B`), which lands far better than terse prompts. Runs in a separate process (the LM and DEMON can't share an interpreter); first use downloads ~1.2 GB. Result fills the Style box — edit freely. |
| **Amount** | Structure ↔ style (denoise). **0 = the original source** (no restyle — handy for A/B); real songs usually want **~0.5–0.65**; tight loops tolerate higher. |
| **Character** | 0 = full restyle · 1 = keep the source's own instrument/voice character. |
| **Steps** | Diffusion steps (4–12). Fewer = snappier control; more = cleaner. Restarts on change. |
| **Window** | Rolling generation window (s). The engine keeps a continuous latent and extends it in chunks whose prefix is *pinned* to the past (seamless — no window seams). Smaller (~8–10s) = snappier control + cheaper; larger (20s) = more context per chunk but laggier control. Restarts on change. |
| **Match: Tempo / Key** | Inject the detected bpm/key into the prompt Metas. Turn Tempo off if BPM detection looks wrong. |
| **Mode** | Coherent (fixed seed) ↔ Evolve (new variations as it loops). |
| **Correction (DCW)** | Per-step wavelet-domain sampler correction ([`DCW`](https://arxiv.org/abs/2604.16044)). **Off by default** — in this turbo/few-step regime it runs the output hot (limiter saturation, harsh on dense material) for a marginal structure gain. Opt-in to experiment. ~1.3 ms/step on MPS, hot-applies on the next regen. |
| **Model** | **Quality (XL)** = richer style, ~2 s control latency (default) · **Fast (2B)** = ~1 s. |

### Models

| | Fast (2B) | Quality (XL) — default |
|---|---|---|
| DiT forward (T=250, depth-1) | ~58 ms | ~135 ms |
| Control latency | ~1 s | ~2 s |
| RAM | ~9 GB (fp32) | ~15 GB (bf16) |
| Download | ~6 GB | +20 GB |
| Hard/real-song style transfer | weaker | clearly stronger |

Both run comfortably real-time on the M4 Max. XL must load in **bf16** (fp32 would
~40 GB OOM); 2B stays fp32.

### Prompting (short version — see [`notes/prompting.md`](notes/prompting.md))

ACE-Step is instruction-tuned, so prompts are *structured*: Caption = style only
(genre/instruments/timbre/era), **never** tempo/key — those are separate Metas (the
Match toggles). It's a *cover*: it keeps the source's tempo/structure, so a slow ballad
won't become "upbeat" — it gets the genre's timbre. Pick a source with the energy you
want; combine 4–8 descriptors; ride Amount.

---

## Project layout

```
engine/          MPS-first cover engine (Python)
  mps_compat.py    runtime shims: no-op cuda fences, fp32→fp64 apg fallback, global no_grad,
                   bf16-on-MPS for XL, periodic reclaim() — keeps DEMON read-only
  loader.py        Session/model loader + audio I/O
  jit.py           JITCover: small-lookahead JIT cover (window walk, lazy tile decode,
                   custom cover prompt, Character/Evolve/Steps, peaks)
  realtime.py      RealtimeCover: producer thread + PCM ring + live control + reconfigure + stats
  metrics.py       timbre-robust structure metrics (chroma / onset correlation)
  style_metric.py  CLAP audio↔text style-adherence metric (dev tuning)
sidecar/
  server.py        local TCP server: framed CONTROL/AUDIO/EVENT, model select,
                   first-run model download with progress
  enhancer.py      prompt-enhancer subprocess (ACE-Step-1.5 5Hz LM, format_sample) —
                   separate process so its `acestep` pkg doesn't clash with DEMON's
  test_client.py   Python IPC client (protocol validation)
app/             JUCE 8 standalone app
  CMakeLists.txt   FetchContent JUCE 8.0.4, Standalone target, BinaryData resources
  Source/          IpcClient (framed TCP→AbstractFifo ring), PluginProcessor (audio cb +
                   resample + soft-clip limiter + sidecar lifecycle), PluginEditor (WebView bridge)
  Resources/       index.html / app.js / styles.css (ported "Hairline" UI)
bench/           Phase 0/1 benchmarks, batch/XL probes, sweeps
cli/             headless CLIs (jit_cover, stream_cover, offline_cover, realtime_test, make_audition)
scripts/run_app.sh   dev launcher (app auto-spawns the venv sidecar)
notes/           per-phase findings + prompting strategy
```

---

## Key engineering findings (the non-obvious stuff)

- **Engine base = DEMON** (vendors the ACE-Step turbo-2B + XL models, node graph, and
  the `StreamPipeline`). Kept **read-only**; all MPS adaptation is runtime monkeypatches
  in `engine/mps_compat.py`.
- **MPS gotchas:** force fp32 (or bf16 for XL); `apg_project` fp64 → fp32; **disable
  DCW** (needs `pytorch_wavelets`, MPS-fragile); `torch.set_grad_enabled(False)` is
  **thread-local** (set it in every worker thread); reclaim MPS memory periodically (not
  per-op — `torch.mps.empty_cache` is expensive). fp16/bf16 ≈ fp32 *speed* on MPS.
- **Decode:** full-clip VAE decode is pathologically slow on MPS (~3.9 s/12 s); the hot
  path decodes only the playhead window. JIT caches a full-window decode and emits it in
  small slices (control latency without per-slice transient smearing).
- **Source latent length must be a multiple of 5** (the 5 Hz semantic tokenizer) — the
  engine truncates on load.
- **Clipping → "metallic":** raw cover output peaks >1.0; the app has a soft-clip
  limiter (threshold 0.92).
- **Amount is song-dependent:** tight beats tolerate 0.8; real songs lose their groove
  above ~0.7 — default 0.7, recommend 0.5–0.65 for songs.

---

## Reference repos (READ-ONLY)

`/Users/max/Code/DEMON` (streaming engine + vendored models), `…/ACE-Step-1.5` (model),
`…/ace-step-apple-silicon` (MPS gotcha reference), `…/stable-audio-3` (the JUCE
`plugin_morph` shell whose UI we ported).

## Remaining (Phase 3 — packaging)

Bundle a private Python + torch into the `.app` (so it runs with zero dev setup),
on-demand model download, code-sign + notarize → a distributable download. Stretch:
in-process MLX/CoreML port to drop Python entirely.
