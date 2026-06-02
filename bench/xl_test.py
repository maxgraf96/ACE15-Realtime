"""Can ACE-Step 1.5 XL-turbo (4 GB→~20 GB bf16) run the cover forward in real
time on this Mac? Loads XL in bf16 (mps), reports RAM, benchmarks the eager
decoder forward vs the 2B baseline + the 360 ms budget. Quality A/B is a second
step only if the forward clears.
"""
from __future__ import annotations
import os, sys, time, resource
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import loader, mps_compat
mps_compat.force_bf16_on_mps()   # opt-in: keep XL in bf16 on MPS (else ~40GB OOM)
import torch  # noqa: E402

BUDGET_MS = 360.0
B2B = {  # Phase 0 baseline (2B turbo, bf16, MPS)
    (250, 1): 57.8, (250, 2): 102.6, (500, 1): 98.5, (500, 2): 184.9,
}


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9  # macOS: bytes


def mps_gb():
    try:
        return torch.mps.current_allocated_memory() / 1e9
    except Exception:
        return float("nan")


@torch.no_grad()
def bench(decoder, enc1, mask1, B, T, dtype, iters=8, warmup=3):
    H = enc1.shape[-1]
    hs = torch.randn(B, T, 64, device="mps", dtype=dtype)
    ctx = torch.randn(B, T, 128, device="mps", dtype=dtype)
    tb = torch.full((B,), 0.5, device="mps", dtype=dtype)
    attn = torch.ones(B, T, device="mps", dtype=dtype)
    enc = enc1.to(dtype) if B == 1 else torch.cat([enc1, torch.zeros_like(enc1)], 0).to(dtype)
    mask = mask1.to(dtype) if B == 1 else torch.cat([mask1, mask1], 0).to(dtype)

    def one():
        return decoder(hidden_states=hs, timestep=tb, timestep_r=tb, attention_mask=attn,
                       encoder_hidden_states=enc, encoder_attention_mask=mask,
                       context_latents=ctx, use_cache=False, past_key_values=None)[0]
    for _ in range(warmup):
        one()
    torch.mps.synchronize(); t0 = time.perf_counter()
    for _ in range(iters):
        one()
    torch.mps.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0


def main():
    t0 = time.time()
    L = loader.load_model(device="mps", config_path="acestep-v15-xl-turbo")
    print(f"[xl] loaded XL-turbo dtype={L.handler.dtype} in {time.time()-t0:.1f}s | "
          f"RSS={rss_gb():.1f}GB mps_alloc={mps_gb():.1f}GB")

    # real conditioning shape (timbre=source path; avoid the silence-timbre MPS quirk)
    from acestep.engine.session import Session
    from acestep.nodes.vae_nodes import VAEEncodeAudio
    from acestep.nodes.semantic_nodes import SemanticExtract
    from acestep.nodes.cond_nodes import EncodeText, EncodeConditioning
    audio = loader.load_audio(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "source.wav"), duration=8.0)
    srclat = VAEEncodeAudio().execute(vae=L.vae, audio=audio)["latent"]
    from acestep.constants import TASK_INSTRUCTIONS
    te = EncodeText().execute(clip=L.clip, tags="lo-fi hip hop", task_type="cover", duration=8.0)["text_embed"]
    cond = EncodeConditioning().execute(model=L.model, text_embed=te, timbre_ref=srclat)["conditioning"]
    e = cond.to_entries()[0]
    enc1 = e.encoder_hidden_states.detach(); mask1 = e.encoder_attention_mask.detach()
    print(f"[xl] enc seq={enc1.shape[1]} hidden={enc1.shape[-1]}  RSS={rss_gb():.1f}GB mps_alloc={mps_gb():.1f}GB")

    dec = L.handler.model.decoder
    print(f"\n  {'T':>4} {'B':>2} {'XL ms':>8} {'2B ms':>7} {'XL/2B':>6} {'vs 360ms':>9}")
    for T in (250, 500):
        for B in (1, 2):
            ms = bench(dec, enc1, mask1, B, T, torch.bfloat16)
            base = B2B[(T, B)]
            ok = "OK" if ms <= BUDGET_MS else "OVER"
            print(f"  {T:>4} {B:>2} {ms:8.1f} {base:7.1f} {ms/base:5.1f}x {('  '+ok):>9}")
    print(f"\n[xl] peak RSS={rss_gb():.1f}GB mps_alloc={mps_gb():.1f}GB  (48GB total)")
    print(f"[xl] budget {BUDGET_MS:.0f}ms/tick. Prescribed: depth-1 RCFG-self T=250 B=1 (lowest).")


if __name__ == "__main__":
    main()
