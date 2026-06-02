# ACE15-Realtime

Real-time text-prompt **style-transfer ("cover")** audio engine for Apple Silicon:
input track + text Style prompt → streamed structure-preserving remix (melody/
rhythm kept, timbre/genre swapped). Ports ACE-Step 1.5's cover capability + DEMON's
streaming engine off CUDA/TensorRT to Metal (MPS, later MLX).

Target machine: Apple M4 Max, 48 GB. macOS / Apple Silicon only. No CUDA/TensorRT.

## Status

- **Phase 0 — gating benchmark: ✅ GO.** Eager turbo-2B DiT forward = 58–190 ms
  on M4 Max (MPS), vs a 360 ms per-tick budget → ~4.7× headroom in the prescribed
  config. Offline cover works and preserves structure (chroma 0.81, onset 0.74).
  See [`notes/phase0_findings.md`](notes/phase0_findings.md).
- Phase 1 — headless streaming cover engine: next.

## Layout

- `engine/` — MPS compat shims (`mps_compat.py`) + model loader (`loader.py`).
- `cli/offline_cover.py` — offline cover + structure validation (chroma/onset).
- `bench/phase0_gate.py` — offline cover + DiT dtype sweep + windowed VAE decode.
- `bench/probe_batch.py` — isolated per-process DiT batch-scaling timings.
- `notes/` — per-phase findings.
- Reference repos (READ-ONLY): `/Users/max/Code/{DEMON, ACE-Step-1.5, ace-step-apple-silicon, stable-audio-3}`.

## Setup

```
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python torch torchaudio torchvision \
  transformers diffusers accelerate safetensors huggingface_hub einops scipy \
  soundfile numpy loguru toml vector-quantize-pytorch torchao mlx mlx-lm
```

Models (~6 GB) auto-resolve from `~/.daydream-scope/models/demon/checkpoints`
(turbo-2B DiT + Oobleck VAE + Qwen3-Embedding-0.6B). The 5 Hz LM and
MelBandRoFormer are intentionally not downloaded.

## Engine base

DEMON (`/Users/max/Code/DEMON`) is the engine base — it vendors the ACE-Step
turbo-2B model, the node graph, and the streaming `StreamPipeline`. We keep it
read-only and adapt to MPS via runtime monkeypatches in `engine/mps_compat.py`.
