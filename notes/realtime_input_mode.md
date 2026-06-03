# Real-time live-input mode (Phase 3)

Take **live audio input** (a DAW track / line-in / loopback) and generate a
tempo-synced AI **accompaniment** that follows it, in real time.

## Decisions (user, 2026-06-03)
- **Task = cover re-band** (turbo, real-time): the input is reimagined as a full
  arrangement in the Style prompt that follows its groove/harmony. Output is the AI
  band (contains a restyled version of the input). True `lego` accompaniment is
  base-model-only (slow) → not the real-time path.
- **Timing = bar-quantized**: AI output trails the input by ~1–2 bars, locked to the
  tempo grid (musical; like an AI loop/canon).
- **Build standalone first**; a VST3 build later just swaps BPM source to the host.
- **Output = AI mix only** (balance the dry track in the DAW).
- **Tempo**: host `AudioPlayHead` (VST) or the manual BPM field (standalone). **Key**:
  manual field. Both already feed the cover Metas.

## The hard constraint
The DiT is **non-causal** (`is_causal=False` everywhere in ACE-Step-1.5 *and* DEMON —
confirmed). It denoises each frame from context before *and* after it. So live output
must trail by ~one window; good quality wants the target in the window interior →
~1–2 bars latency. No sub-100ms interplay without retraining (out of scope). DEMON has
no live-input or causal path to reuse; only its windowed-decode/ring patterns (already
in our engine).

## Data flow (target)
```
[DAW track / line-in / loopback]
 -> JUCE input bus (host SR, stereo)
 -> processBlock: capture -> lock-free FIFO -> sender thread -> IPC 0x04 (host SR)
 -> sidecar: resample host->48k -> append to rolling input waveform
 -> rolling VAE encode (overlap-discard) -> grow source latent + hints
 -> pinned-rolling producer (cover, turbo, full-band Style, mid-high Amount),
      frontier gen_f <= available input frames; pin to styled past for continuity
 -> tiled decode -> ring -> IPC 0x02 -> C++ -> output (AI mix only)
 bar-grid: output scheduled on the tempo grid, trailing input by whole bars
```
The existing pinned-rolling producer already extends a growing `committed` latent
bounded by `full_T`; live mode = feed it a source latent that grows from the stream +
gate the frontier. Extension, not rewrite.

## Phases
- **A — Audio input path (no model). [DONE 2026-06-03]**
  Input bus; capture in `processBlock`; lock-free FIFO + dedicated sender thread (never
  block the RT thread); new IPC `0x04` (client->server, host-SR stereo); write mutex so
  control+audio share the socket. Sidecar: `input_config/start/stop` controls, `0x04`
  ingest, resample host->48k (torchaudio, one pass at finalize), ~10 Hz `input_level`
  events, capture WAV. Temporary "Live in" header toggle (Phase D replaces).
  *Gate PASSED:* `bench/rt_input_test.py` streams a 1 kHz tone @44.1k → captured 48k WAV
  is 2.00s, dom freq 1000.0 Hz, peak 0.500 (pitch/level/duration preserved). App builds.
  Live mic test (speak in, watch input_level, inspect WAV) = user's manual check.
- **B — Live cover (continuous), headless-validated. [DONE 2026-06-03]**
  `JITCover.begin_live/feed_live/encode_pending` rolling overlap-discard encode (margin
  12 frames ≈0.48s; rel-err vs full encode **6e-4** — near-lossless, small VAE receptive
  field). Lazy `_ensure_handle` (live source appears only after the first chunk).
  `RealtimeCover.begin_live/feed_input/_produce_live` — same pinned rolling, source grows
  from the stream, frontier GATED by available input, decode only frames with full
  context, no loop/seek. Sidecar `live_start`/`live_stop` (initial style set BEFORE start
  so no MPS races the producer; live edits use the queued setters), `0x04`→`feed_input`
  (resample host→48k). *Gates PASSED:* `bench/rt_encode_test.py` (encode accuracy),
  `bench/rt_live_test.py` (0 underruns, seamless — max-jump 77x >> p99.9 19x = transients
  not a seam; **latency 9.5s@win8 / 5.3s@win4** — scales with window), `bench/rt_live_socket_test.py`
  (full sidecar path: cover streams non-silent from a live 0x04 stream). A/B WAVs:
  `test_output/diag/rt_live_cover.wav` + `rt_live_input.wav`. NOT YET: C++ Live-in toggle
  still does Phase-A capture; wiring it to `live_start` + the model-load UI is Phase D.
  Latency floor (~window + ~1.5s headroom/buffer) is the Phase C lever.
- **C — Bar-quantization + tempo grid. [DONE 2026-06-03]**
  `_produce_live` derives the bar from `jit.bpm` (manual field / host later), generates in whole-bar
  hops (`_pin_f=_hop_f=bar_frames` → 2-bar window), and at the FIRST cover push pads the output up to
  the next whole bar (`n_sil = ceil(L/bar)*bar - L`) so the cover trails the input by an integer # of
  bars, grid-locked. Live `_TILE=16` (smaller decode tiles → less headroom). *Gate PASSED*
  `bench/rt_bar_test.py`: trail = **exactly 3 bars** @120bpm (6.00s) AND @90bpm (8.04s), 0 drift.
  Metric note: measure the trail by `first_sound` (first non-silent cover) — NOT `fed-real`, which
  counts the prepended silence as real. Trail floor ≈ window + ~1 bar; smaller window-in-bars = lower
  latency, less context. bpm fixed at producer start (mid-stream tempo change = Phase E concern).
- **D — UI mode + polish. [DONE 2026-06-03]**
  Header **File/Live** toggle (`#srcmode-toggle`) → `body.live-mode` CSS hides the file panels +
  shows `#source-live` (tempo/key inputs + input level meter `#inmeter-fill` + hint). Transport is
  live-aware: Play = `startRealtime` (C++ `live_start` w/ model+style+bpm+key+dcw, `captureInput=true`),
  Stop = `stopRealtime` (`live_stop` + flush). Input meter from `input_level` events; `live_started`/
  `live_stopped` drive status/transport. A/B label = "Input" in live (cover vs your delayed input).
  Steps/Window/Model disabled in live (window is bar-derived; pick model in File mode first — avoids a
  mid-live file reload). File drops ignored in live. Output = AI mix only. C++ `startRealtime`/
  `stopRealtime` native fns. Build clean; full live socket path re-verified PASS (bar-quantized). Live
  audition (pick input in Options›Audio, Play, jam) = user's manual test. Known: model toggle disabled
  in live; mid-live BPM change updates Metas not the grid.
- **E — VST3 (later).** VST3 target; BPM/transport from `AudioPlayHead`; AI-only bus.
  Deferred: bundled-Python-per-instance-in-a-DAW is fragile.

- **OUTPUT stem mixer (bonus). [DONE 2026-06-03]** Separate the GENERATED accompaniment into 4 stems
  with per-stem **mute/solo** ("I only want the drums the model made"). `engine/separation.py`
  `StemSeparator` = torchaudio **Hybrid Demucs** (`HDEMUCS_HIGH_MUSDB_PLUS`, drums/bass/other/vocals,
  44.1k; resample 48k↔44.1k; per-chunk normalize; **warms the MPS graph at load** — first forward is
  ~3.8s). Applied at the OUTPUT: `JITCover.decode_out(lat,cache,a,b)` decodes the cover slice, and if a
  stem subset is selected, separates a margin window (`OUT_SEP_MARGIN_F=20` ≈0.8s, fits the live decode
  headroom 24 → no extra latency; rolling-vs-whole corr 0.976) and keeps only those stems. Both
  producers push via `decode_out`. `set_stems` = active stem set (UI computes from mute/solo: solos if
  any, else non-muted). `RealtimeCover.set_stems` (queued live; pre-warms Demucs when set before start);
  sidecar `stems` control + `live_start` pre-warms `jit._ensure_sep()` so mute/solo is responsive any
  time; C++ `setStems`/`liveStems`; UI 4-group M/S mixer in the live panel. **CRITICAL:** in live the
  producer is input-gated, so any mid-stream stall (Demucs first-forward ~3.8s) depletes the buffer →
  underruns; FIX = pre-warm before the producer runs (`_ensure_sep` warmup + live_start pre-warm).
  *Gates PASSED:* `bench/out_sep_test.py` (rolling-vs-whole corr 0.976 @margin20), `rt_live_test.py
  ACE15_LIVE_STEMS=drums` (**0 underruns**), `rt_live_socket_test.py ACE15_LIVE_STEMS=drums` (full path).
  (Earlier INPUT-separation attempt was reverted — user wanted output.) A/B "Input" = the full input.

## Risks
- Latency floor ~1–2 bars (inherent; managed by bar-quant).
- Rolling-encode added to the producer RT budget (encode+gen+decode) — keep RTF<1
  (validate in B).
- Capturing other apps' audio in standalone needs a loopback device (BlackHole/Loopback);
  line/mic works out of the box.
- Long-run MPS/bf16 stability for continuous encode+gen.

## Phase A artifacts
- IPC `0x04` + input FIFO + `inLoop` sender + `writeFrame`/`writeMutex` (IpcClient).
- Input bus + `captureInput` + `setRealtimeInput` + processBlock capture (PluginProcessor).
- `setRealtimeInput` native fn (PluginEditor); "Live in" toggle (index.html/app.js).
- Sidecar `input_config/start/stop` + `_on_audio_in` + `_finalize_input_capture`.
- `bench/rt_input_test.py` (validation). Capture WAV: `test_output/diag/rt_input_capture.wav`.

Related: [[phase2-realtime]].
