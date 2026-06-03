"""Phase B.1: does the rolling (overlap-discard) live encode match a full encode?

Encodes a file two ways on ONE model: (a) full pass via load_track, (b) streaming via
begin_live + feed_live + encode_pending in small chunks. Compares the resulting source
latents frame-by-frame, sweeping the discard margin to pick the smallest that's accurate.
If the interior frames match closely, the live source latent is sound to generate from.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from engine import mps_compat  # noqa
from engine.jit import JITCover, SPF

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "assets", "source.wav")
SECS = 20

jit = JITCover(device="mps", steps=8)            # 2B Fast; VAE (what we measure) shared with XL
jit.load_track(SRC, seconds=SECS, detect=False)  # full encode = reference
full = jit.source.latent.tensor.float().clone()
wav = jit.source_wav.clone()                     # [2, N] @ 48k
print(f"[probe] full latent: {tuple(full.shape)}  ({full.shape[1]/25:.1f}s)", flush=True)


def rel_err(a, b):
    return (a - b).norm(dim=-1) / (b.norm(dim=-1) + 1e-6)   # per-frame relative L2


for margin in (12, 25, 50):
    jit.handle = None
    jit.begin_live()
    jit.LIVE_MARGIN_F = margin
    # feed in 0.25 s chunks, encoding as we go
    step = SPF * 6   # ~0.25 s
    for i in range(0, wav.shape[1], step):
        jit.feed_live(wav[:, i:i + step])
        jit.encode_pending()
    roll = jit.source.latent.tensor.float()
    L = min(roll.shape[1], full.shape[1])
    if L < 80:
        print(f"  margin={margin}: only {L} frames encoded — skip"); continue
    e = rel_err(roll[0, :L], full[0, :L]).cpu().numpy()
    interior = e[60:L - 10]
    print(f"  margin={margin:2d} ({margin/25:.2f}s): frames={L}  "
          f"interior rel-err  median={np.median(interior):.4f}  p90={np.percentile(interior,90):.4f}  "
          f"edge(first 60) median={np.median(e[:60]):.4f}", flush=True)

print("VERDICT: pick the smallest margin whose interior median rel-err is small (<~0.05).", flush=True)
jit.close()
