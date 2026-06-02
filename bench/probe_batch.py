"""Isolate the MPS DiT-forward batch threshold. One batch size per process
(the failure is an LLVM abort, not a catchable exception)."""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import loader
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--batch", type=int, required=True)
ap.add_argument("--T", type=int, default=250)
ap.add_argument("--dtype", default="bf16")
a = ap.parse_args()
dt = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[a.dtype]

L = loader.load_model(device="mps")
dec = L.handler.model.decoder.to(dt)
B, T = a.batch, a.T
H = 2048
Lt = 68
hs = torch.randn(B, T, 64, device="mps", dtype=dt)
ctx = torch.randn(B, T, 128, device="mps", dtype=dt)
tb = torch.full((B,), 0.5, device="mps", dtype=dt)
attn = torch.ones(B, T, device="mps", dtype=dt)
enc = torch.randn(B, Lt, H, device="mps", dtype=dt)
mask = torch.ones(B, Lt, device="mps", dtype=dt)
import time
print(f"PROBE batch={B} T={T} dtype={a.dtype} ...", flush=True)
def one():
    with torch.no_grad():
        return dec(hidden_states=hs, timestep=tb, timestep_r=tb, attention_mask=attn,
                   encoder_hidden_states=enc, encoder_attention_mask=mask,
                   context_latents=ctx, use_cache=False, past_key_values=None)[0]
for _ in range(3):
    one()
torch.mps.synchronize()
t0 = time.perf_counter()
N = 10
for _ in range(N):
    one()
torch.mps.synchronize()
ms = (time.perf_counter() - t0) / N * 1000
print(f"RESULT batch={B} T={T} dtype={a.dtype} {ms:.1f} ms/fwd {ms/B:.1f} ms/slot", flush=True)
