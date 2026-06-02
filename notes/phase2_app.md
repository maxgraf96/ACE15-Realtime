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
