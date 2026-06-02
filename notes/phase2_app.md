# Phase 2 — Standalone app (status + how to run)

**Working end-to-end native app skeleton.** One command launches a JUCE
standalone app that auto-spawns the Python cover engine internally, connects over
a local socket, and streams a live, restylable cover. This is the shippable
architecture (Path A): in the distributed `.app` the sidecar + a private Python
live inside the bundle — the user opens one app.

## Architecture (all built + proven)

```
ACE15 Realtime.app (JUCE)                     bundled Python sidecar
┌─────────────────────────────┐   framed TCP  ┌──────────────────────────┐
│ WebView UI (Resources/)      │  127.0.0.1    │ server.py                │
│ PluginEditor  ── native fns ─┼──CONTROL────▶ │  RealtimeCover (ring)    │
│ PluginProcessor (audio cb)   │ ◀──AUDIO──────┤   JITCover (lazy decode) │
│   IpcClient (ring) ──────────┤ ◀──EVENT──────┤    Session (ACE-Step/MPS)│
└─────────────────────────────┘               └──────────────────────────┘
```

- IPC: `[4B len][1B type]` frames — 0x01 CONTROL(JSON c→s), 0x02 AUDIO(f32
  stereo 48k c←s), 0x03 EVENT(JSON c←s). C++ `IpcClient` ↔ `sidecar/server.py`.
- Audio thread drains a lock-free `AbstractFifo` ring filled by the net thread,
  resamples 48k→host SR. Controls from the WebView → CONTROL frames; EVENTs
  (loaded/styled/playing/download_progress/error) → UI.
- App auto-spawns the sidecar via `juce::ChildProcess` (env-driven in dev;
  embedded interpreter in the shipped bundle).

## Validated
- App builds (JUCE 8.0.4, CMake), launches, no crash.
- IPC client connects to the sidecar (handshake confirmed in logs).
- One-command launch auto-spawns the sidecar.
- Underlying audio+control path proven via the Python client (`sidecar_stream.wav`)
  and the real-time ring (0 underruns at 1×).

## Run it (dev)
```
# 1. build once (JUCE fetched at configure)
cmake -S app -B app/build -DCMAKE_BUILD_TYPE=Release && cmake --build app/build -j8
# 2. launch (auto-spawns the sidecar; uses the project .venv)
./scripts/run_app.sh
```
Then: **Load track…** → set **Style** → **Play**. Edit Style live → restyles in
~1 s. **Amount** = structure↔style. First run downloads ~6 GB models with a
progress bar (models cached at ~/.daydream-scope/models/demon/checkpoints).

## DCW — Differential Correction in Wavelet domain (toggle)

Opt-in sampler-side per-step correction (ACE-Step `acestep.engine.dcw`, paper
arXiv:2604.16044): after each integration step, decompose `x_next` and the
predicted clean `denoised = x - v·t` with a 1-D DWT along time and push `x_next`'s
bands away from `denoised` (`double` mode: low band `t·0.05`, high band
`(1−t)·0.02`, haar). Wired as a UI **Correction** toggle.

- **Backend works on MPS.** The original `dcw_enabled=False` was *precaution*, not
  a tested failure — `pytorch_wavelets`/`PyWavelets` were simply never installed.
  Measured on MPS: DWT1D perfect-reconstructs (err 4.8e-7), and a full
  `DCWCorrector.apply` is **~1.3 ms/step** (double/haar, T=500) steady-state — the
  33 ms first call is one-time graph build, cached. Negligible vs the regen.
- **Integration = mutate `handle.base_kwargs`.** `StreamDenoise.execute` re-reads
  `dcw_*` from kwargs and calls `pipe.set_dcw()` *every tick*, so toggling needs no
  pipeline rebuild and no compiled-graph invalidation. `JITCover.set_dcw()` updates
  `base_kwargs`; `RealtimeCover.set_dcw()` queues it → producer applies + forces one
  regen so the change is audible within ~lookahead. On Steps change (handle rebuild)
  the dcw state is re-passed via `_dcw_kwargs()`.
- **Verified (all PASS):** DCW off is bit-deterministic; on changes the latent
  (maxdiff ~2.9); toggling back off restores **byte-identical** output (clean
  identity, no residual state). Producer toggle mid-play: regen fired, worst step
  723 ms < 1 s lookahead, 0 underruns. Socket path: `style{dcw}` + live `dcw` cmd,
  no errors, audio flows.
- **Default OFF** (reverted 2026-06-02 after the regression below). Was briefly
  default-on per user request, but **DCW runs the output hot** in this turbo/few-step
  regime: raw peak 0.93–1.06 vs 0.84 off → slams the soft-clip limiter (post-limiter
  0.995 vs 0.871) + ~2× HF → harsh/distorted on dense material (Dubstep), for only a
  marginal structure gain (onset 0.611 vs 0.570). Diagnosis: engine output otherwise
  fine (realtime matches render; 0 underruns even with tweaks) — it's a LEVEL issue
  the onset/chroma metrics miss; check output peak vs the 0.92 limiter threshold.
  `JITCover.__init__ self.dcw_enabled=False`, C++ `dcwEnabled{false}`, JS `dcw=false`,
  HTML toggle "DCW Off". Toggle kept for opt-in; A/B WAVs in test_output/diag/.
- **Robustness:** control commands sent before a track is loaded used to crash
  (`self.rc` is None → `'NoneType' has no attribute ...`); the sidecar now no-ops
  any non-`load` command while `rc is None` (the UI re-sends state via `style` on
  load anyway). Fixed for ALL controls, not just DCW.
- **Graceful degrade:** `mps_compat.dcw_available()` probes `pytorch_wavelets`; if
  missing, enabling logs a warning and stays off (no mid-tick crash). Deps are
  optional in the quick-start.

## Adaptive buffer — fix for glitchy/"metallic" playback at large windows

Symptom (user, 2026-06-02): load song → Fast → back to Quality → play at **window
30 / XL / DCW on / Amount 0.90** → "super metallic and terrible"; changing the
window (which restarts) mostly fixed it.

- **NOT the model switch.** Headless repro (`bench/model_switch_repro.py`): after a
  2B→XL round-trip in one process the XL DiT loads correctly in **bf16** and the
  generated audio is stable (peak 1.305, identical across 1st/2nd/3rd gen, no NaN).
  So dtype/generation is clean — the switch was incidental.
- **It's UNDERRUNS.** A window regen blocks production for ~`max_step_ms`; at
  window 30 + XL + DCW that's **~3.1 s**, but the buffer was a FIXED
  `lookahead` (2.0 s for Quality). When a regen fires mid-playback the consumer
  drains the 2 s buffer and then reads zeros for ~1 s → periodic glitch/dropout
  that reads as "metallic". `bench/underrun_test.py` at 1×: window30/LA2.0 →
  **23 underruns**, worst regen 3155 ms. Shrinking the window shrank the regen →
  fewer gaps (why resizing "fixed" it). The screenshot tell: worst 2321 ms > the
  2000 ms lookahead.
- **Fix = adaptive buffer** (`engine/realtime.py` `_produce`): the fill target is
  `max(lookahead, min(8s, max_step_ms/1000 + 0.4)) + slice` — it auto-grows to
  cover the MEASURED worst regen, so a big window / slow model / DCW can't outrun
  the buffer. Verified: window 30 → **0 steady-state underruns** at both LA 2.0 and
  LA 1.0 floors (buffer settles ~5 s). The static `lookahead` is now just a floor.
- **Tradeoff:** control latency = buffer depth, so a 30 s window costs ~5 s
  latency; window 20 (~2 s regen) costs ~3 s. Documented in the Window tooltip +
  README. (The first regen still gives ~3 s of startup silence while priming —
  acceptable; a future polish could gate the sender until primed.)

## Window-seam crossfade — fix for occasional "continuity break"

Symptom (user, 2026-06-02): while vibing, every so often a "break in the latents"
— not silence/hang, content stays *somewhat* connected (structure) but the
texture/voicing jumps. Hunch: related to lookahead/window. Correct area.

- **Cause:** the realtime producer **hard-cut** window seams. On each regen it did
  `win_start = committed_f`, decoded from there, `_push` with NO crossfade. Each
  window is an INDEPENDENT generation (same source → same structure, but different
  diffusion realization → different timbre/phase), so the boundary is a texture
  discontinuity. The offline `render()` (whose auditions the user loved) avoids
  this: it backs the new window up by `XF` frames and equal-power blends the
  overlap (`back` / `append(seg, XF)`). The realtime path simply never ported that.
  It recurs ~every window length after a tweak (a forced regen restarts the clock),
  which is why it felt sporadic and window-tied.
- **Fix** (`engine/realtime.py`): port the overlapping-window crossfade. Streaming
  can't blend already-pushed audio, so the producer **holds back the last `_xf_samp`
  samples as a tail** (`_emit`), and on a regen generates the new window backed up
  by `_xf_f` frames and equal-power blends its head into that tail (`_emit_seam`).
  Position-aligned (seg[:xf] = same timeline as the tail) so the rhythm isn't
  compressed — the exact failure the Phase-1 streaming notes warned about with
  contiguous crossfades. `xfade_s=0.12` (matches render). Tail reset on producer
  start and on seek (clean cut at the seek point); the loop point blends
  track-end→track-start. Also smooths forced regens (control tweaks now morph over
  0.12s instead of hard-cutting).
- **Verified:** synthetic — a phase-jumped tone seam drops from a **0.79** sample
  jump (audible click) to **0.044** crossfaded (~18×), peak unchanged, length
  drift-free (no rhythm compression). Real-model **parity with `render()`**: same
  clip/style, max sample jump 1.28 (realtime) vs 1.22 (render) and 99.99pct 0.548
  vs 0.547 — identical (the big jumps are musical transients, present in both),
  i.e. the realtime seam now behaves exactly like the proven offline render.
- Engine-only (Python) — no app rebuild; live on the next sidecar respawn.

## Window-boundary continuity — interior overlap-add hand-off + DCW gain-comp

User (2026-06-02): "at the end of every window things turn terrible." Asked to do
it "properly like DEMON" (rolling buffer), no hacks.

- **Finding: DEMON's cover path is ALSO windowed.** Its `StreamPipeline` continuous
  mode (depth=steps, submit-per-tick) is for the *morph* case; for a sliding cover
  it'd regenerate a full window every tick to emit a sliver (~tens of × compute) —
  not real-time. DEMON's actual cover driver is **walk-window** (fixed window,
  advance on a boundary) — same shape as ours. So boundaries are inherent for long
  tracks; the lever is HOW you hand off.
- **Root cause of OUR harsh seam:** the JIT spliced at the window EDGES (the DiT's
  weakest region — least temporal context) with a tiny crossfade. Each window is an
  independent generation (fresh noise), so the edge↔edge splice is a real
  content/quality jump. (A global position-indexed noise field to make overlapping
  windows agree was tried and *measured worse* — at the edges the missing-context
  error dominates the noise.)
- **Fix = interior overlap-add hand-off** (`engine/realtime.py _produce`): transition
  `_margin_f` (1 s) BEFORE the window edge; generate the next window backed up by the
  margin so the playhead lands in its INTERIOR; equal-power crossfade `_xf_f` (0.12 s)
  where BOTH windows have full context; discard the degraded edges. Position-aligned
  (no timeline compression). Hop becomes W−2·margin.
- **Crossfade length matters — keep it SHORT.** Controlled psig test (onset_corr,
  higher=better rhythm): edge/short(≈old) **0.316**, interior/short **0.366** (best),
  interior/long-0.4s **0.144** (a long crossfade smears rhythm across the seam). So:
  interior margin 1.0 s + crossfade 0.12 s. Both tunable in `RealtimeCover.__init__`.
- **Verified:** synthetic timeline test = 0 drift/repeat/gap (frames 1→450 all +1);
  interior/short beats old edge/short on onset; runs clean. Engine-only Python → no
  app rebuild; live on sidecar respawn. Still inherent: a residual texture change at
  seams (independent-window generations) — interior+short minimizes it; user auditions.
- **DCW gain-comp:** when DCW is on, output ×0.85 (~−1.4 dB) so it no longer slams the
  soft-clip limiter (it ran hot — see the DCW-default-OFF note). `_dcw_gain` in the
  producer, applied to each emitted slice.

## Pinned-prefix ROLLING engine — the real continuity fix (replaces windowed crossfade)

User confirmed the windowed crossfade still had audible seams near song segments
(verse→drop). Replaced the windowed producer with a **continuous rolling latent**
(time-axis inpainting) — the proper DEMON-style rolling buffer.

- **Mechanism:** keep ONE continuous `committed` latent; extend it in chunks where
  each chunk's first `_pin_f` frames are PINNED (inpainting `LatentNoiseMask`,
  mask=0 preserve / mask=1 generate) to the committed tail. The model denoises the
  new frames to CONTINUE the fixed past → the overlap is identical, **no crossfade,
  no seam by construction**. `jit._gen(..., pin=clean_tail)` builds the mask;
  `RealtimeCover._produce` rolls + decodes the committed latent with the existing
  overlap-discard tiler (lazy, trailing the frontier by `_TILE+_TOV` so every
  decoded tile has full VAE context). window_s = generation window; pin = 4s fixed
  context; hop = window − pin = new frames/chunk.
- **Probe first** (`bench/rolling_probe.py`): pin honored exactly (err 0.0000),
  continuity at pin→new = 0.87× a typical frame step (no jump), pinned-new quality
  == independent window (onset 0.542 vs 0.554), flat RMS over a 4-window chain (no
  drift). User A/B'd `roll_continuous.wav` vs `roll_windowed.wav` → "way way better".
- **Live producer verified** (psig/Dubstep, W20, dn0.70, char0 == probe): peak 1.26,
  onset_corr 0.579, max-jump 0.40, **0 underruns** at 1×, RMS rising with the song
  (no drift) — matches the probe. Seamless max-jump on smooth material ~0.01–0.08
  (vs windowed's seam spikes).
- **Cost ≈ same** as windowed: per chunk it's still N forwards over the window;
  pinning adds no forwards. Window/latency tradeoff: SMALL window (~8–10s) = snappy
  control + cheap + still seamless (pin); LARGE window (20s) = more new-content
  context per chunk but laggier control (committed runs ~hop ahead). Both seamless.
  Recommend ~8–10s for live tweaking. (Follow-up for instant control: force a
  re-roll from the playhead on control change.)
- **Removed:** the interior-overlap-add + equal-power crossfade (`_emit`/`_emit_seam`,
  `_tail`, `_xf_*`, `_margin_f`) — superseded by rolling. DCW gain-comp + adaptive
  buffer + lazy tiler retained. Engine-only Python (no app rebuild; sidecar respawn).
- **Note:** away.mp3 covers low-energy at most styles (song-dependent, pre-existing)
  — test rolling on a song that covers well (psig) to hear the seamless benefit.

## First-window style — 2-pass primed first chunk

User: "the second window always sounds way better than the first 20s — the style
doesn't take hold yet." Cause (CLAP, same source region on XL): a COLD independent
generation styles ~2× weaker than a PINNED continuation (region[W:W+H] indep 0.042
vs pinned 0.096). The rolling first chunk is cold (no styled predecessor); later
chunks pin to styled output → stronger. Fix (uses the play-time prime budget, as
the user suggested): the first chunk (and post-seek/loop) is **2-pass** — a quick
cold pass yields a short prefix, then regenerate the chunk PINNED to it so the body
gets the same continuation style-boost. `prime_s=2.0`. Verified: first 16s dubstep
0.042→0.099 (steady 0.140; body ≈ steady), 0 underruns. Only the literal first ~2s
stays cold (nothing before frame 0 to pin) — trimmable if desired. Engine-only.

## Control latency in the rolling engine (Amount/Character take effect fast)

User: big Amount/Character jumps sometimes took a long time. Cause: the producer
generates a whole hop ahead (up to ~16s @ win20) with the OLD settings; a change
only affects future chunks → that old-setting latent plays first (variable with the
gen cycle = "sometimes"). Fix (`_produce`): on a re-style control, discard the
un-decoded generated-ahead and regenerate from the decode point with the new
settings (pinned to the decoded past → smooth, no gap; invalidate decoded-ahead
tiles too). Latency then = the inherent buffer depth: ~5s @ win20, ~1.5s @ win10
(was up to ~16s); 0 underruns. **Use a smaller Window for snappier control.**
Future option for near-instant: also flush the ring + re-roll from the playhead
(brief gap per tweak) — deferred.

## Prompt enhancer (✨) — short style → rich caption via the 5Hz LM

User: "my text prompts almost never land." Integrated ACE-Step's prompt LM
(`acestep-5Hz-lm-0.6B`, ~1.2 GB, MPS via transformers) using `format_sample` —
"tech house" → "An energetic, driving electronic track... punchy four-on-the-floor
kick... deep resonant synth bassline...". ACE-Step-1.5 and DEMON both ship an
`acestep` package, so the LM runs in a SEPARATE process: `sidecar/enhancer.py`
(sys.path=ACE-Step-1.5, lazy LM load, JSON-line stdin/stdout, stdout kept clean).
Sidecar spawns it lazily, `_ensure_lm()` downloads w/ progress, `_do_enhance()`
runs off-thread → "enhancing"/"enhanced" events (handled BEFORE the rc-None guard
so it works with no track loaded). C++ `enhance(tags)` + native fn; JS ✨ button
next to Style fills the box + applies. Verified end-to-end over the socket.
Runtime dep: ACE-Step-1.5 repo (env ACE15_ACESTEP15) — bundle for distribution.

## Remaining to ship (packaging/polish — no model/DSP risk left)
1. **Bundle a private Python + torch into the .app** (python-build-standalone +
   vendored wheels) so it runs with no dev venv; spawn the embedded interpreter.
   Reference: stable-audio-3 `plugin/cmake/vendor_bundle.sh`, `scripts/build_release.sh`.
2. **First-run model download** — wired (`_ensure_models` + UI progress bar);
   verify on a clean machine (models already present here).
3. **Code-sign + notarize** (entitlements for JIT/network/microphone-none).
   Reference: `plugin/scripts/notarize.sh`, `entitlements.plist`.
4. Polish: drag-drop (file picker works now), waveform/playhead, presets,
   sidecar health/restart, graceful "engine starting…" state.
5. Later: VST3/AU (same engine; per-instance sidecar caveats), and the Phase 3
   native MLX port to drop Python entirely.
