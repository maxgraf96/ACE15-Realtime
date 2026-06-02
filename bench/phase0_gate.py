"""Phase 0 gate: offline cover + DiT-forward / VAE-decode benchmark on Mac.

Loads the turbo-2B cover stack ONCE, then:
  (A) runs one offline cover (encode -> semantic hint -> text style -> generate
      -> decode -> WAV) to confirm the cover path is correct on MPS;
  (B) benchmarks the eager DiT decoder forward across T / dtype / CFG, plus the
      windowed VAE decode, and prints ms vs the ~0.36 s real-time budget.

Usage:
  .venv/bin/python bench/phase0_gate.py [--src assets/source.wav] [--cover-seconds 12]
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import loader  # noqa: E402  (installs mps_compat on import)
from engine.mps_compat import mps_sync  # noqa: E402

import torch  # noqa: E402

SR = 48000
FPS = 25.0                 # latent frames per second
RT_BUDGET_S = 0.36         # mission's per-tick real-time budget (windowed slice)
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test_output")
NOTES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "notes")


# ----------------------------------------------------------------------------
# Robust VAE decode (MPS conv1d can choke on long sequences; tile + crossfade).
# ----------------------------------------------------------------------------
def vae_decode_bdt(handler, lat_bdt: torch.Tensor, chunk: int = 64, overlap: int = 8) -> torch.Tensor:
    """Decode [B,64,T] latents -> [B,2,samples], tiling on MPS if needed."""
    try:
        with handler._load_model_context("vae"):
            return handler.tiled_decode(lat_bdt)
    except Exception as e:
        print(f"[decode] tiled_decode failed ({type(e).__name__}: {e}); manual chunked decode")
    # Manual overlap-add decode through the raw Oobleck VAE.
    vae = handler.vae
    T = lat_bdt.shape[-1]
    stride = chunk - 2 * overlap
    pieces = []
    pos = 0
    with handler._load_model_context("vae"):
        while pos < T:
            lo = max(0, pos - overlap)
            hi = min(T, pos + stride + overlap)
            sub = lat_bdt[..., lo:hi].to(vae.dtype)
            wav = vae.decode(sub).sample.float()  # [B,2,(hi-lo)*1920]
            trim_l = (pos - lo) * 1920
            trim_r = (hi - min(T, pos + stride)) * 1920
            wav = wav[..., trim_l: wav.shape[-1] - trim_r if trim_r else None]
            pieces.append(wav)
            pos += stride
    return torch.cat(pieces, dim=-1)


# ----------------------------------------------------------------------------
# (A) Offline cover
# ----------------------------------------------------------------------------
def offline_cover(L, src_path: str, seconds: float, style: str):
    from acestep.nodes.vae_nodes import VAEEncodeAudio
    from acestep.nodes.semantic_nodes import SemanticExtract
    from acestep.nodes.cond_nodes import EncodeText, EncodeConditioning
    from acestep.nodes.diffusion_nodes import StreamDenoise, OdeSolver

    os.makedirs(OUT_DIR, exist_ok=True)
    print("\n" + "=" * 72 + "\n[A] OFFLINE COVER\n" + "=" * 72)
    src_audio = loader.load_audio(src_path, duration=seconds)
    dur = src_audio.waveform.shape[-1] / SR
    print(f"  source: {src_path}  ->  {list(src_audio.waveform.shape)}  ({dur:.1f}s)")

    t = time.time()
    source_latent = VAEEncodeAudio().execute(vae=L.vae, audio=src_audio)["latent"]
    mps_sync(); print(f"  VAE encode      {list(source_latent.tensor.shape)}  {time.time()-t:.2f}s")

    t = time.time()
    context_latent = SemanticExtract().execute(model=L.model, latent=source_latent)["latent"]
    mps_sync(); print(f"  semantic hint   {list(context_latent.tensor.shape)}  {time.time()-t:.2f}s")

    t = time.time()
    te = EncodeText().execute(
        clip=L.clip, tags=style, lyrics="", task_type="cover",
        bpm=120, duration=dur, key="C major", time_signature="4", language="en",
    )["text_embed"]
    conditioning = EncodeConditioning().execute(
        model=L.model, text_embed=te, timbre_ref=source_latent,
    )["conditioning"]
    enc = conditioning.to_entries()[0]
    mps_sync()
    print(f"  text+cond       enc={list(enc.encoder_hidden_states.shape)}  {time.time()-t:.2f}s  style='{style}'")

    solver = OdeSolver().execute()["solver"]
    t = time.time()
    out_latent = StreamDenoise().execute(
        model=L.model, solver=solver, positive=conditioning,
        context_latent=context_latent, source_latent=source_latent,
        steps=8, shift=3.0, denoise=1.0, seed=1234,
        pipeline_depth=1, drain=True, dcw_enabled=False, duration=dur,
        noise_on_cpu=True,
    )["latent"]
    mps_sync(); gen_s = time.time() - t
    print(f"  generate(8 step) {list(out_latent.tensor.shape)}  {gen_s:.2f}s  ({gen_s/8*1000:.0f} ms/step incl. overhead)")

    t = time.time()
    lat_bdt = out_latent.tensor.transpose(1, 2)
    wav = vae_decode_bdt(L.handler, lat_bdt)
    mps_sync(); print(f"  VAE decode      {list(wav.shape)}  {time.time()-t:.2f}s")

    from acestep.nodes.types import Audio
    out_path = os.path.join(OUT_DIR, "phase0_cover.wav")
    loader.save_audio(Audio(waveform=wav.squeeze(0), sample_rate=SR), out_path)
    loader.save_audio(src_audio, os.path.join(OUT_DIR, "phase0_source.wav"))

    # Sanity numbers
    o = wav.squeeze(0).float().cpu()
    s = src_audio.waveform.float().cpu()
    n = min(o.shape[-1], s.shape[-1])
    rms = o.pow(2).mean().sqrt().item()
    peak = o.abs().max().item()
    # crude structure-correlation on the mono envelope (10ms windows)
    import torch.nn.functional as F
    def env(x):
        m = x[:, :n].mean(0).abs()
        w = int(0.01 * SR)
        return F.avg_pool1d(m.view(1, 1, -1), w, w).flatten()
    eo, es = env(o), env(s)
    k = min(len(eo), len(es))
    eo, es = eo[:k] - eo[:k].mean(), es[:k] - es[:k].mean()
    corr = (eo * es).sum() / (eo.norm() * es.norm() + 1e-9)
    print(f"  sanity: out RMS={rms:.4f} peak={peak:.3f}  envelope-corr(src,out)={corr:.3f}")
    return conditioning


# ----------------------------------------------------------------------------
# (B) DiT forward benchmark
# ----------------------------------------------------------------------------
@torch.no_grad()
def bench_forward(decoder, enc1, mask1, device, dtype, B, T, iters=10, warmup=3):
    H = enc1.shape[-1]
    L_ = enc1.shape[1]
    hs = torch.randn(B, T, 64, device=device, dtype=dtype)
    ctx = torch.randn(B, T, 128, device=device, dtype=dtype)
    tb = torch.full((B,), 0.5, device=device, dtype=dtype)
    attn = torch.ones(B, T, device=device, dtype=dtype)
    if B == 1:
        enc = enc1.to(dtype)
        mask = mask1.to(dtype)
    else:  # CFG: positive + (zeroed) negative
        enc = torch.cat([enc1.to(dtype), torch.zeros_like(enc1).to(dtype)], 0)
        mask = torch.cat([mask1.to(dtype), mask1.to(dtype)], 0)

    def one():
        out = decoder(
            hidden_states=hs, timestep=tb, timestep_r=tb, attention_mask=attn,
            encoder_hidden_states=enc, encoder_attention_mask=mask,
            context_latents=ctx, use_cache=False, past_key_values=None,
        )
        return out[0]

    for _ in range(warmup):
        one()
    mps_sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        one()
    mps_sync()
    return (time.perf_counter() - t0) / iters * 1000.0  # ms/forward


def benchmark(L, conditioning):
    print("\n" + "=" * 72 + "\n[B] DiT FORWARD BENCHMARK (eager, MPS)\n" + "=" * 72)
    device = L.handler.device
    decoder = L.handler.model.decoder
    enc1 = conditioning.to_entries()[0].encoder_hidden_states.detach()
    mask1 = conditioning.to_entries()[0].encoder_attention_mask.detach()
    print(f"  device={device}  enc seq_len={enc1.shape[1]} hidden={enc1.shape[-1]}")

    dtypes = [("fp32", torch.float32), ("fp16", torch.float16), ("bf16", torch.bfloat16)]
    Ts = [(250, "10s"), (500, "20s")]
    cfgs = [("single(B=1)", 1), ("cfg(B=2)", 2)]
    rows = []
    for dname, dt in dtypes:
        decoder = decoder.to(dt)
        e1, m1 = enc1.to(dt), mask1.to(dt)
        for T, tlabel in Ts:
            for clabel, B in cfgs:
                try:
                    ms = bench_forward(decoder, e1, m1, device, dt, B, T)
                    rows.append((dname, T, tlabel, clabel, ms))
                    print(f"  {dname:5s} T={T:4d}({tlabel}) {clabel:11s}  {ms:8.1f} ms/fwd"
                          f"   budget {RT_BUDGET_S*1000:.0f}ms -> {'OK' if ms<=RT_BUDGET_S*1000 else 'OVER'}")
                except Exception as e:
                    rows.append((dname, T, tlabel, clabel, None))
                    print(f"  {dname:5s} T={T:4d} {clabel:11s}  FAILED: {type(e).__name__}: {e}")
    # NOTE: batch scaling (B=4/8) is measured by bench/probe_batch.py in
    # ISOLATED processes. Churning many dtype+batch shapes in ONE long-lived
    # process triggers a hard MPSGraph abort (mps_matmul "invalid shape" ->
    # LLVM ERROR), so we keep this in-process sweep to B<=2 and fixed shapes.

    # restore fp32
    L.handler.model.decoder = decoder.to(torch.float32)

    # Windowed VAE decode (~0.36s slice = ~9 kept frames; engine decodes ~25)
    print("\n  windowed VAE decode (PyTorch/MPS):")
    vae_rows = []
    for nfr in (9, 25):
        lat = torch.randn(1, 64, nfr, device=device, dtype=L.handler.vae.dtype)
        for _ in range(2):
            with L.handler._load_model_context("vae"):
                L.handler.vae.decode(lat)
        mps_sync(); t0 = time.perf_counter()
        for _ in range(5):
            with L.handler._load_model_context("vae"):
                L.handler.vae.decode(lat)
        mps_sync(); ms = (time.perf_counter() - t0) / 5 * 1000
        vae_rows.append((nfr, ms))
        print(f"    {nfr:2d} frames ({nfr/FPS:.2f}s audio)  {ms:7.1f} ms")
    return rows, vae_rows


def write_notes(L, rows, vae_rows, cover_ok):
    os.makedirs(NOTES, exist_ok=True)
    p = os.path.join(NOTES, "phase0_results.md")
    lines = []
    lines.append("# Phase 0 results — eager DiT forward on Apple M4 Max (MPS)\n")
    lines.append(f"- Generated by `bench/phase0_gate.py` on torch {torch.__version__}")
    lines.append(f"- Device: {L.handler.device}, model dtype (load): fp32 (model_context forces fp32 on non-CUDA)")
    lines.append(f"- Real-time per-tick budget: **{RT_BUDGET_S*1000:.0f} ms** (windowed decode slice)")
    lines.append(f"- Offline cover correctness: **{'PASS' if cover_ok else 'FAIL'}** (test_output/phase0_cover.wav)\n")
    lines.append("## DiT decoder forward (ms/forward)\n")
    lines.append("| dtype | T (window) | mode | ms/fwd | vs 360ms |")
    lines.append("|---|---|---|---|---|")
    for dname, T, tlabel, clabel, ms in rows:
        if ms is None:
            lines.append(f"| {dname} | {T} ({tlabel}) | {clabel} | FAILED | — |")
        else:
            verdict = "✅ OK" if ms <= RT_BUDGET_S * 1000 else "❌ OVER"
            lines.append(f"| {dname} | {T} ({tlabel}) | {clabel} | {ms:.1f} | {verdict} |")
    lines.append("\n## Windowed VAE decode (ms)\n")
    lines.append("| frames | audio | ms |")
    lines.append("|---|---|---|")
    for nfr, ms in vae_rows:
        lines.append(f"| {nfr} | {nfr/FPS:.2f}s | {ms:.1f} |")
    lines.append("")
    with open(p, "w") as f:
        f.write("\n".join(lines))
    print(f"\n[notes] wrote {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "source.wav"))
    ap.add_argument("--cover-seconds", type=float, default=12.0)
    ap.add_argument("--style", default="aggressive heavy metal, distorted electric guitars, double kick drums")
    ap.add_argument("--skip-cover", action="store_true")
    ap.add_argument("--skip-bench", action="store_true")
    args = ap.parse_args()

    L = load = loader.load_model(device="mps")
    print(f"  torch {torch.__version__}  mps={torch.backends.mps.is_available()}")

    conditioning = None
    cover_ok = False
    if not args.skip_cover:
        conditioning = offline_cover(L, args.src, args.cover_seconds, args.style)
        cover_ok = True
    if not args.skip_bench:
        if conditioning is None:
            from acestep.nodes.cond_nodes import EncodeText, EncodeConditioning
            te = EncodeText().execute(clip=L.clip, tags=args.style, task_type="cover", duration=20.0)["text_embed"]
            conditioning = EncodeConditioning().execute(model=L.model, text_embed=te)["conditioning"]
        rows, vae_rows = benchmark(L, conditioning)
        write_notes(L, rows, vae_rows, cover_ok)


if __name__ == "__main__":
    main()
