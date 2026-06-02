# Phase 1 — Headless streaming cover engine on Mac

**Status: ✅ working.** A headless walk-window streaming cover runs on Apple
Silicon (MPS) and produces a coherent structure-preserving remix **far inside
real time (RTF ≈ 0.09–0.12, i.e. 8–12× headroom)**, with live prompt control.
Built on DEMON's `Session.stream`/`StreamPipeline`, kept read-only and adapted
to Metal entirely via `engine/mps_compat.py`.

Date: 2026-06-02 · M4 Max, 48 GB · torch 2.12.0 (MPS) · turbo-2B, depth-1
drain, RCFG/CFG off, DCW off, fp32.

## What was built

- `engine/streaming.py` — `StreamingCover`: load track → `prepare_source`
  (VAE-encode + semantic hints) → walk **overlapping** windows through the track,
  generate each window latent once (depth-1 drain), full-decode it (tiled), and
  forward-write with an equal-power crossfade over the generated overlap.
- `cli/stream_cover.py` — `WAV + style → streamed remix WAV`, prints RTF +
  latencies + structure (chroma/onset) self-validation; `--prompt-at S:tags`
  for live prompt swaps.
- `bench/phase1_sweep.py` — operating-point sweep (`notes/phase1_results.md`).
- `engine/metrics.py` — timbre-robust chroma/onset structure metrics.

## Operating points (30 s source, M4 Max, MPS)

| window | steps | denoise | RTF | max-chunk (lookahead) | chroma | onset |
|---|---|---|---|---|---|---|
| 10 s | 8 | 0.6 | 0.11 | 0.95 s | **0.82** | **0.77** |
| 10 s | 8 | 0.8 | 0.12 | 0.98 s | 0.54 | 0.16 |
| 10 s | 4 | 0.8 | **0.085** | 0.72 s | 0.53 | 0.17 |
| 15 s | 8 | 0.6 | 0.13 | 1.4 s | **0.88** | **0.90** |
| 20 s | 8 | 0.8 | 0.10 | 1.8 s | 0.69 | 0.47 |
| 30 s | 8 | 0.6 (1 chunk) | 0.09 | 2.6 s | **0.91** | **0.92** |

- **RTF ≈ 0.09–0.12 across every config** — real time is comfortably met; the
  per-chunk DiT drain (~0.5 s for a 10 s window) + tiled decode (~0.4 s) are the
  costs, amortized over the window.
- **`max-chunk` = worst single gen+decode** = the minimum producer lookahead
  buffer for gapless playback (0.7 s at win10/steps4 → 2.6 s at win30). Smaller
  window/steps → smaller lookahead.
- **steps=4 ≈ steps=8** in quality here at ~25 % less compute — a cheap lever.
- **Live prompt swap**: re-encode ~120 ms; applies at the next chunk boundary
  (v0 control granularity = window size, ~10–15 s). Sub-0.5 s knob latency needs
  the continuous depth pipeline (see "next").

## Two bugs found and fixed (both general MPS lessons)

1. **OOM from autograd graphs.** Model params load `requires_grad=True`; the VAE
   decode path is not wrapped in `no_grad`, so retained graphs ballooned MPS
   memory into a hard OOM (61 GiB into swap) over a streaming session. Fix:
   `torch.set_grad_enabled(False)` globally in `mps_compat` (inference-only).
2. **Timeline-compression from naive crossfade → wrecked rhythm.** Crossfading
   *contiguous* (non-overlapping) chunks removes real audio per seam, drifting
   the timeline and tanking onset corr (0.92 → 0.37 multi-chunk). Fix: generate
   **overlapping** windows (hop = win − overlap) and crossfade the *generated
   overlap* — exact timeline, smooth seam. Onset recovered to ~0.90.

   Related decode finding: the **0.36 s windowed decode** (`StreamVAEDecode`,
   DEMON's live-morph hot path) smears transients when tiled into a track (56
   independent 0.36 s decodes → onset 0.27). For a track cover we already hold
   the whole window latent, so **full tiled decode per chunk is both faster on
   MPS overall and far higher quality** (onset 0.88+). The windowed decode is
   reserved for the continuous live-morph mode where the latent changes per tick.

## Quality / control knobs (for the plugin)

- **Amount → denoise.** Clear tradeoff, quantified: dn≈0.6 maximises structure
  (chroma 0.82–0.91, onset 0.77–0.92); dn≈0.8 maximises style (chroma 0.54–0.69,
  onset lower/variable). User auditioned dn0.7–0.8 as best-sounding, so perceived
  quality tolerates lower onset corr than the metric implies (an aggressive
  restyle legitimately diverges from the source's transients).
- **Window** ≥ 15 s gives best coherence; 10 s is fine with more headroom and a
  smaller lookahead buffer. **Steps** 4–8.

## Open / next

1. **Continuous depth-pipeline + shared-curve control** for sub-0.5 s live knob
   latency (true 1-tick control). v0's per-chunk granularity is fine for
   "render/stream a cover", not for live knob-twiddling — that's the Phase 2
   plugin UX, and needs the StreamPipeline depth>1 + `set_shared_curve`.
2. **Producer/consumer threading** (generator thread fills a ring; audio thread
   drains) — the real gapless-playback architecture; this is the Phase 2 plugin
   shape. RTF 0.1 + a 1–3 s lookahead buffer makes it trivially feasible.
3. MPS hygiene under long sessions: pin one dtype + bounded shapes; `reclaim()`
   per chunk (done).

## Phase 1 v1 — small-lookahead JIT control (the plugin engine)

`engine/jit.py` (`JITCover`) + `cli/jit_cover.py`. Default dn0.8 (user-preferred).

The audio you hear was generated `lookahead` ago, so control latency ≥ lookahead.
With RTF ~0.1 we keep lookahead tiny (~1 s) and regenerate the window the instant a
control changes (fixed seed → coherent variation), crossfaded into the committed
output. Quality stays full: we FULL-decode each window once and cache it, emitting
committed audio in small slices from that cache (free copies) — so control applies
at lookahead granularity without the per-slice transient smearing.

Measured (30 s, M4 Max, dn0.8, win10, lookahead1.0):
- **control latency = 1.00 s** (was >15 s with per-chunk v0; "metal" swap now gets a
  full 21–30 s segment instead of never appearing).
- steady quality = full-window decode (chroma 0.56 / onset 0.13 at dn0.8 — same as
  the chunk-render; the regens add no degradation), exact 30.00 s timeline (no drift).
- RTF 0.14–0.18; worst single step (regen + full decode) ~1.0 s, covered by the
  lookahead buffer. This is literally the producer/consumer ring the Phase 2 plugin
  needs (generator fills ~1 s ahead; audio callback drains).

Knob mapping for the plugin: prompt → re-encode + regen (~1 s); Amount → denoise
+ regen (~1 s). Lower lookahead → lower latency at the cost of a thinner safety
buffer against the worst-case regen spike.

## Reproduce

```
.venv/bin/python cli/stream_cover.py --seconds 30 --window 15 --denoise 0.7 \
    --style "lo-fi hip hop, jazzy piano, boom-bap" --prompt-at 20:"8-bit chiptune"
.venv/bin/python bench/phase1_sweep.py   # operating-point table
```
