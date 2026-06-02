"""Probe: is the A/B (bypass) ORIGINAL time-aligned with the COVER it replaces?

The engine pushes cover[dec_f:dec_target] and source_slice(dec_f, len) together in
one ring chunk, so the buffer/lookahead delays BOTH equally — there is no relative
lookahead offset. What CAN misalign A/B:

  (A) source_slice's assumption "latent frame f <-> samples [f*SPF:(f+1)*SPF]".
      If the VAE encode/decode has a constant framing/group delay, the reconstruction
      is shifted vs the raw source by a FIXED amount -> a fixed shift would fix it.
      Measured at denoise=0 (decode(source_latent) == VAE roundtrip, no model
      transient movement), raw cross-correlation (timbre preserved -> sample-exact).
      The VAE is shared across Fast/Quality, so this lag is model-independent.

  (B) the model moving transients at denoise>0 (a generative restyle isn't sample-
      locked to the source groove). CONTENT-dependent -> NOT fixable by any fixed
      shift. Measured via onset-envelope cross-correlation of a real cover window.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from engine import mps_compat, metrics  # noqa
from engine.jit import JITCover, FPS, SPF, SR
from acestep.nodes.types import Latent

SONG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "source.wav")
DENOISE = 0.70; SEED = 1234


def best_lag_raw(a, b, maxlag):
    """Lag L (samples) maximizing corr(a, shift(b, L)); L>0 => b lags a (b is later)."""
    a = a - a.mean(); b = b - b.mean()
    best, bl = -1e9, 0
    for L in range(-maxlag, maxlag + 1):
        if L >= 0:  x, y = a[L:], b[:len(b) - L] if L else b
        else:       x, y = a[:L], b[-L:]
        k = min(len(x), len(y))
        if k < 2000: continue
        d = float(np.dot(x[:k], y[:k]) / (np.linalg.norm(x[:k]) * np.linalg.norm(y[:k]) + 1e-9))
        if d > best: best, bl = d, L
    return bl, best


def best_lag_onset(cover_mono, src_mono, max_frames=24):
    """Onset-envelope cross-correlation; lag in onset frames (hop=512) -> ms."""
    hop = 512
    ec = metrics.onset_env(torch.tensor(cover_mono), hop=hop).numpy()
    es = metrics.onset_env(torch.tensor(src_mono), hop=hop).numpy()
    n = min(len(ec), len(es)); ec, es = ec[:n], es[:n]
    ec = ec - ec.mean(); es = es - es.mean()
    best, bl = -1e9, 0
    for L in range(-max_frames, max_frames + 1):
        if L >= 0:  x, y = ec[L:], es[:len(es) - L] if L else es
        else:       x, y = ec[:L], es[-L:]
        k = min(len(x), len(y))
        if k < 8: continue
        d = float(np.dot(x[:k], y[:k]) / (np.linalg.norm(x[:k]) * np.linalg.norm(y[:k]) + 1e-9))
        if d > best: best, bl = d, L
    return bl * hop, best   # samples, corr


jit = JITCover(device="mps", steps=8)   # 2B Fast; VAE (the thing measured in A) is shared with XL
jit.load_track(SONG)
src_mono = jit.source_wav.float().mean(0).numpy()
T = jit.source.latent.tensor.shape[1]
print(f"[probe] song={os.path.basename(SONG)}  latent T={T} ({T/FPS:.1f}s)  samples={len(src_mono)}", flush=True)

# ---- (A) VAE roundtrip alignment (decode(source_latent) vs raw source) ----
nrec = min(T, 250)                                   # up to 10 s
recon = jit._ensure_tiles(Latent(tensor=jit.source.latent.tensor), {}, 0, nrec).float().mean(0).numpy()
src_seg = src_mono[:len(recon)]
lag_a, corr_a = best_lag_raw(recon, src_seg, maxlag=2000)   # +-42 ms, sample-exact
print(f"[A] VAE roundtrip lag = {lag_a:+d} samples ({lag_a/SR*1000:+.1f} ms)  corr={corr_a:.3f}", flush=True)
print(f"    => source_slice frame<->sample mapping is {'CORRECT (no fixed offset)' if abs(lag_a) <= 48 else 'OFF by a constant -> compensate source_slice'}", flush=True)

# ---- (B) real cover transient movement (onset env), a mid-track window ----
jit.set_style("warm lo-fi hip hop, dusty drums", denoise=DENOISE, character=0.0)
W = min(200, T); w0 = min(T // 4, max(0, T - W))     # avoid the cold first frames
cover = jit._gen(w0, w0 + W, SEED).tensor
cov_mono = jit._ensure_tiles(Latent(tensor=cover), {}, 0, W).float().mean(0).numpy()
src_win = src_mono[w0 * SPF: w0 * SPF + len(cov_mono)]
lag_b, corr_b = best_lag_onset(cov_mono, src_win)
print(f"[B] cover vs source onset lag = {lag_b:+d} samples ({lag_b/SR*1000:+.1f} ms)  corr={corr_b:.3f}", flush=True)
print(f"    => {'tight (<=~10ms): A/B will feel locked' if abs(lag_b) <= 512 else 'model moves transients (content-dependent; NOT fixable by a fixed shift)'}", flush=True)
jit.close()
