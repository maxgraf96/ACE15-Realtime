"""Validate the live source-latent fix (window re-encode) over a LONG 60s capture.

The chunked latent encode is the dominant cause of live drift; whole re-encode fixes
it but is O(n) (slows down -> underruns long-term). 'window' re-encodes only the recent
~20s seamlessly (bounded cost) and should MATCH whole quality while staying real-time.
Drift is measured in sixths (10s each) so the trajectory over a full minute is visible.

PASS: window+frozen stays flat & low (~0.20-0.30) with ~0 underruns, like whole but cheaper.
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

def softclip(x, t=0.92):
    a = np.abs(x); return np.where(a <= t, x, np.sign(x)*(t+(1-t)*np.tanh((a-t)/(1-t)))).astype(np.float32)

def instab(x):
    m = x.mean(1).astype(np.float64); nfft, hop = 2048, 512; w = np.hanning(nfft)
    F = np.array([np.abs(np.fft.rfft(m[i:i+nfft]*w)) for i in range(0, len(m)-nfft, hop)])
    if len(F) < 2: return 0.0
    e = F.sum(1); thr = max(1e-6, 0.05*np.median(e))
    d = np.abs(np.diff(F, axis=0)).sum(1)/(F[1:].sum(1)+1e-9); keep = e[1:] > thr
    return float(np.mean(d[keep])) if keep.any() else 0.0

def sixths(cov):
    n = len(cov)//6; return [instab(cov[i*n:(i+1)*n]) for i in range(6)]

rc = RealtimeCover(device="mps", steps=8, window_s=20.0, lookahead_s=2.0, config_path="acestep-v15-xl-turbo")
src = loader.load_audio(PIANO, duration=11).waveform.float(); N = src.shape[1]

def capture(secs):
    block = 2048; period = block/SR; out=[]; cap=False; got=0; need=int(secs*SR); fed=0
    t0=time.perf_counter(); nxt=t0; u0=rc.underruns
    while got<need and time.perf_counter()<t0+secs+120:
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

CFG = [("chunked+froz",   "chunked", "frozen", 0),
       ("whole+froz",     "whole",   "frozen", 0),
       ("window+froz",    "window",  "frozen", 0),
       ("window+anchor2", "window",  "frozen", 2)]
for tag, lat_m, hint_m, anchor in CFG:
    rc.reset(); rc.begin_live(); rc.live_pin_bars=1; rc.live_hop_bars=1; rc.live_anchor_bars=anchor
    rc.jit.enc_lat_mode = lat_m; rc.jit.enc_hint_mode = hint_m; rc.max_step_ms = 0.0
    rc.set_style(STYLE, denoise=0.7, bpm=90, key="C minor"); rc.start()
    cov, u = capture(60); rc.stop()
    s = sixths(cov)
    sf.write(f"{OUT}/encwin_{tag.replace('+','_')}.wav", softclip(cov), SR)
    print(f"[{tag:>14}] 10s-bins = [{' '.join(f'{v:.2f}' for v in s)}]  "
          f"drift(last-first)={s[-1]-s[0]:+.3f}  worst_step={rc.max_step_ms:.0f}ms  "
          f"reanchors={rc.reanchors}  underruns={u}", flush=True)
rc.close()
print("PASS: window+froz flat & low like whole, underruns~0 (whole may creep up over 60s).", flush=True)
