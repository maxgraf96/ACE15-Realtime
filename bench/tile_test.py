"""Does the overlap-discard tiler bit-match a full decode at small _TILE? If error grows
as _TILE shrinks, the live _TILE=16 introduces periodic tile-edge artifacts (warble).
VAE-only (no generation) — fast.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from engine import mps_compat  # noqa
from engine.jit import JITCover, SPF
from acestep.nodes.types import Latent

PIANO = "/Users/max/Music/Ableton/Noiiz/SynapticJourney_Noiiz/Loops/Keys/90_C_BriarPatchVintagePiano_01_695.wav"
jit = JITCover(device="mps", steps=8)   # 2B; VAE is shared with XL
jit.load_track(PIANO)
lat = jit.source.latent
N = min(lat.tensor.shape[1], 240)
print(f"[probe] latent {N} frames ({N/25:.1f}s)", flush=True)

# full reference: one VAE decode
h = jit.session.handler
with h._load_model_context("vae"):
    lb = lat.tensor[:, :N, :].transpose(1, 2).to(h.vae.dtype)
    full = h.vae.decode(lb).sample.float().squeeze(0).cpu().numpy()   # [2, N*SPF]

for TILE in (48, 24, 16, 8):
    jit._TILE = TILE
    out = jit._ensure_tiles(Latent(tensor=lat.tensor[:, :N, :]), {}, 0, N).float().numpy()
    L = min(out.shape[-1], full.shape[-1])
    err = np.abs(out[:, :L] - full[:, :L]).mean() / (np.abs(full[:, :L]).mean() + 1e-9)
    # discontinuity AT tile boundaries (k*TILE*SPF) vs interior
    d = np.abs(np.diff(out[0, :L]))
    tb = TILE * SPF
    bj = np.mean([d[k] for k in range(tb, len(d), tb)]) if len(d) > tb else 0
    print(f"  _TILE={TILE:2d}: rel-err vs full={err:.4f}  tile-boundary jump/typ={bj/(d.mean()+1e-9):.1f}x", flush=True)
print("VERDICT: if rel-err/boundary-jump grow as _TILE shrinks -> _TOV=8 margin too small -> warble", flush=True)
