# Phase 0 — Setup + the gating measurement

**Verdict: 🟢 GO.** The eager turbo-2B DiT forward on this Mac runs at **58–190 ms**
in every config the real-time cover path actually uses — far below the
**360 ms** per-tick budget (and nowhere near the "≫0.36 s everywhere" stop
condition). A real-time Apple-Silicon cover VST is reachable **on the MPS
backend alone** — we do not even need the MLX DiT port to clear the gate.

Date: 2026-06-02 · Machine: Apple **M4 Max**, 40 GPU cores, 48 GB unified RAM ·
torch 2.12.0 (arm64, MPS) · Python 3.11.

---

## TL;DR for the product decision

- **One offline cover works on Mas, unmodified DEMON engine** (runtime no-op of the
  CUDA fences + DCW off + fp32). Output is a genuine structure-preserving remix:
  chroma corr **0.81**, onset corr **0.74** vs source at denoise 0.6.
- **Per-tick cost in the prescribed config** (turbo-2B, depth-1, RCFG-self,
  10 s window, bf16): **~58 ms forward + ~18 ms windowed decode ≈ 77 ms** →
  **~4.7× real-time headroom** against the 360 ms budget.
- **fp16/bf16 give ~zero speedup over fp32 on MPS** (they only halve memory).
  The "use fp16" real-time lever does **not** buy latency here.
- **Short window is the dominant speed lever**, not precision: T=250 (10 s) is
  ~1.8× faster than T=500 (20 s) per forward, and is all the cover needs (local
  structure only).
- **MPS stability caveat (important for Phase 1/2):** hammering many different
  dtype+batch shapes in one long-lived process triggers a hard MPSGraph abort
  (`mps_matmul ... invalid shape` → `LLVM ERROR`). Every shape works fine in a
  *fresh* process. The sidecar must run a **single fixed dtype + a bounded set
  of shapes** and must not re-cast the model at runtime.

---

## Environment

- Base engine = **DEMON** (`/Users/max/Code/DEMON`), kept read-only. It vendors
  the ACE-Step v1.5 turbo-2B DiT (`acestep.models.modeling_acestep_v15_turbo`),
  the node graph, and the streaming engine — so Phase 0 directly exercises the
  Phase 1 core. The apple-silicon port is kept only as an MPS-gotcha reference.
- Models (cover path only, ~6 GB) at `~/.daydream-scope/models/demon/checkpoints`:
  `acestep-v15-turbo` 4.5 GB · `vae` (Oobleck) 322 MB · `Qwen3-Embedding-0.6B`
  1.1 GB. The 5 Hz LM and MelBandRoFormer were **not** downloaded; an empty
  `acestep-5Hz-lm-1.7B/` dir satisfies DEMON's existence check without loading it.
- MPS adaptation is **runtime-only** (`engine/mps_compat.py`): no-op
  `torch.cuda.synchronize/empty_cache`, fp32 fallback for `ode_steps.apg_project`
  (MPS has no float64), prepend DEMON to `sys.path`. DEMON's tree is untouched.
- `Session`/nodes run with `device="mps"`, `decoder_backend="eager"`,
  `vae_backend="eager"`, attn=SDPA, **DCW disabled**, **no CFG / RCFG-self**.
- Surprisingly little else was needed: DEMON already auto-selects `mps`, already
  forces fp32 on non-CUDA, and its offline path's CUDA calls are either guarded
  by `is_available()` or harmless no-ops without CUDA.

## (A) Eager DiT decoder forward — ms/forward vs 360 ms budget

`bench/phase0_gate.py`, eager, MPS, real Qwen-encoded conditioning (enc seq≈70,
hidden=2048). `single(B=1)` = RCFG-self/depth-1 per-tick; `cfg(B=2)` = standard CFG.

| dtype | T=250 (10 s) B=1 | T=250 B=2 | T=500 (20 s) B=1 | T=500 B=2 |
|---|---|---|---|---|
| fp32 | 57 ms ✅ | 106 ms ✅ | 102 ms ✅ | 190 ms ✅ |
| fp16 | 58 ms ✅ | 104 ms ✅ | 100 ms ✅ | 185 ms ✅ |
| bf16 | 58 ms ✅ | 103 ms ✅ | 100 ms ✅ | 185 ms ✅ |

Every cell is under 360 ms. **fp16/bf16 ≈ fp32** (MPS isn't precision-bound on
this matmul mix).

## (B) Batch / pipeline-depth scaling (bf16, isolated process per point)

Per-tick batch = pipeline depth in StreamDiffusion mode (`bench/probe_batch.py`).

| T (window) | B=1 | B=2 | B=4 | B=8 | ms/slot @B=8 |
|---|---|---|---|---|---|
| 250 (10 s) | 57.8 | 102.6 | 186.3 | **341.5** | 42.7 |
| 500 (20 s) | 98.5 | 184.9 | 343.1 | 665.9 | 83.2 |

Batch scales **sublinearly** (58→43 ms/slot). Implication for streaming modes:

- **depth-1 + RCFG-self (B=1)** — the prescribed mode: 58 ms/forward, ~77 ms/tick
  incl. decode. Lowest latency (control changes land in 1 tick), lowest memory.
- **depth-8 StreamDiffusion pipelining (B=8)**: 341 ms/tick at T=250 — *just*
  fits 360 ms; at T=500 (666 ms) it does **not** fit. So if we ever pipeline,
  the window must stay short. depth-1 is the safer default and clears the bar
  with room to spare.

## (C) Windowed VAE decode (PyTorch/MPS)

| frames | audio | ms |
|---|---|---|
| 9  | 0.36 s (kept slice) | **18.5** |
| 25 | 1.00 s (engine window) | 38.1 |

But **full-window decode is slow on MPS**: decoding a whole 12 s (300-frame)
latent via `tiled_decode` took **3.94 s**. The Oobleck decoder does not scale
linearly on MPS — large single decodes are pathological. This is exactly why the
streaming design decodes only the ~0.36 s slice at the playhead. Phase 1 must use
the windowed decode, never the full decode, in the hot loop.

## (D) Offline cover correctness — structure preservation

`cli/offline_cover.py`, source = a 60 s music loop (DEMON fixture), trimmed to
12 s, style = "aggressive heavy metal, distorted guitars, double kick drums".
Timbre-robust metrics (amplitude envelope is meaningless across a timbre swap):

| variant | chroma corr (harmony) | onset corr (rhythm) |
|---|---|---|
| cover, denoise = 0.60 | **0.806** | **0.735** |
| cover, denoise = 1.00 | 0.728 | −0.057 |
| control, no context (silence), denoise 1.0 | 0.665 | −0.046 |

- `context_latents` **does** carry structure: chroma 0.73–0.81 with context vs
  0.67 without.
- **denoise is the dominant rhythm dial.** At 0.6 (blend source latent at init)
  both harmony *and* rhythm are preserved (onset 0.74). At 1.0 (pure noise init)
  rhythm is lost (onset ≈ 0) while harmony survives via the 5 Hz semantic hint.
- ⇒ Map the plugin's **"Amount" knob → denoise**, sweet spot ≈ 0.5–0.7 for a
  recognizable but restyled cover.
- WAVs saved under `test_output/` for auditory confirmation (not yet listened to).

## What this means for Phase 1 (and the open risks it closes/opens)

CLOSED: "Is the eager 2B forward fast enough on this Mac?" — **yes, decisively.**
The make-or-break unknown from the brief is resolved in favor of GO, on MPS,
without MLX.

STILL OPEN / to measure in Phase 1:
1. **Sustained end-to-end 1×** through the real walk-window runner (not just the
   forward in isolation): scheduling, crossfades, control-change latency.
2. **MPS process stability** under a long-lived streaming session — must pin one
   dtype and a fixed shape set; add a watchdog/restart path. (The dtype-churn
   abort is the single scariest finding.)
3. The **silence-timbre encoder path** (`refer [1,750,64]`) was implicated in one
   crash; verify it's churn (likely) vs a genuine shape bug, since EncodeConditioning
   defaults to it when no timbre ref is given.
4. Whether to default **depth-1** (latency/safety) vs **short-window depth-8**
   (throughput) — both clear the bar at T=250.

MLX status: there is **no MLX DiT** in DEMON or the apple-silicon port (MLX is
used only for the optional 5 Hz LM, which we don't load). The "move DiT to MLX"
lever requires a real port — defer to Phase 3, and only if MPS headroom proves
insufficient under load. Phase 0 says MPS headroom is large, so this is optional.

## Reproduce

```
.venv/bin/python bench/phase0_gate.py --cover-seconds 12   # cover + dtype sweep + decode
for B in 1 2 4 8; do .venv/bin/python bench/probe_batch.py --batch $B --T 250; done  # batch scaling
.venv/bin/python cli/offline_cover.py --denoise 0.6 1.0    # structure validation + WAVs
```
