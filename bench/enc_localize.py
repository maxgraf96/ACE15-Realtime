"""Localize the live-specific EXTRA drift: is it the chunked LATENT encode or the
frozen HINTS? Run live (anchor off, 32s) at the 4 combinations of
enc_lat_mode in {chunked, whole} x enc_hint_mode in {frozen, whole}.

Reference: chunked+frozen ~0.53 (bad), whole+whole ~0.35 (file-level, good).
Whichever axis flipping to 'whole' recovers the quality is the culprit to fix cheaply.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from engine import mps_compat, loader
mps_compat.force_bf16_on_mps()
from engine.realtime import RealtimeCover, SR

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "test_output", "diag"); os.makedirs(OUT, exist_ok=True)
PIANO = "/Users/max/Music/Ableton/Noiiz/SynapticJourney_Noiiz/Loops/Keys/90_C_BriarPatchVintagePiano_01_695.wav"
STYLE = "Instrumental jazz trio, warm upright bass, brushed drums, piano comping"
import soundfile as sf

def instab(x):
    m = x.mean(1).astype(np.float64); nfft, hop = 2048, 512; w = np.hanning(nfft)
    F = np.array([np.abs(np.fft.rfft(m[i:i+nfft]*w)) for i in range(0, len(m)-nfft, hop)])
    if len(F) < 2: return 0.0
    e = F.sum(1); thr = max(1e-6, 0.05*np.median(e))
    d = np.abs(np.diff(F, axis=0)).sum(1)/(F[1:].sum(1)+1e-9); keep = e[1:] > thr
    return float(np.mean(d[keep])) if keep.any() else 0.0

def quarters(cov):
    n = len(cov)//4; return [instab(cov[i*n:(i+1)*n]) for i in range(4)]

rc = RealtimeCover(device="mps", steps=8, window_s=20.0, lookahead_s=2.0, config_path="acestep-v15-xl-turbo")
src = loader.load_audio(PIANO, duration=11).waveform.float(); N = src.shape[1]

def capture(secs):
    block = 2048; period = block/SR; out=[]; cap=False; got=0; need=int(secs*SR); fed=0
    t0=time.perf_counter(); nxt=t0; u0=rc.underruns
    while got<need and time.perf_counter()<t0+secs+90:
        if rc.live:
            s=fed%N; ch=src[:, s:s+block]
            if ch.shape[1]<block: ch=torch.cat([ch, src[:, :block-ch.shape[1]]],1)
            rc.feed_input(ch); fed+=block
        o=rc.read(block); cov=o[:,:2]
        if not cap and np.abs(cov).max()>1e-3: cap=True; u0=rc.underruns
        if cap: out.append(cov.copy()); got+=block
        nxt+=period; dt=nxt-time.perf_counter()
        if dt>0: time.sleep(dt)
        else: nxt=time.perf_counter()
    return np.concatenate(out,0), rc.underruns-u0

for lat_m, hint_m in [("chunked","frozen"), ("whole","whole"), ("whole","frozen"), ("chunked","whole")]:
    rc.reset(); rc.begin_live(); rc.live_pin_bars=1; rc.live_hop_bars=1; rc.live_anchor_bars=0
    rc.jit.enc_lat_mode = lat_m; rc.jit.enc_hint_mode = hint_m
    rc.set_style(STYLE, denoise=0.7, bpm=90, key="C minor"); rc.start()
    cov, u = capture(32); rc.stop()
    q = quarters(cov)
    print(f"[lat={lat_m:>7} hint={hint_m:>6}] quarters = [{q[0]:.3f} {q[1]:.3f} {q[2]:.3f} {q[3]:.3f}]  "
          f"drift={q[3]-q[0]:+.3f}  underruns={u}", flush=True)
rc.close()
print("Culprit = the axis whose flip to 'whole' drops the final quarter toward ~0.35.", flush=True)
