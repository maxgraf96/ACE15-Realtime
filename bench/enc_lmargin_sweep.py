"""Find the minimum LEFT-context margin that makes the cheap CHUNKED latent encode match
the seamless WHOLE encode. Hypothesis: chunked drifts because its encode sees only ~0.5s
of left context (starved VAE receptive field) -> per-chunk boundary artifacts the roll
amplifies. A bigger left margin (past audio, always available) should heal it at bounded
cost (encode lm+cf+mf per chunk, NOT O(n)). Live, anchor off, frozen hints, 40s.

PASS: some chunked lm matches whole's drift (~0.18) with underruns~0 and low worst_step.
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

def fifths(cov):
    n = len(cov)//5; return [instab(cov[i*n:(i+1)*n]) for i in range(5)]

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

# chunked latent at increasing left margins, plus whole as the quality reference
CFG = [("chunk lm12",  "chunked", 12), ("chunk lm38",  "chunked", 38),
       ("chunk lm75",  "chunked", 75), ("chunk lm150", "chunked", 150),
       ("whole",       "whole",   0)]
for tag, lat_m, lm in CFG:
    rc.reset(); rc.begin_live(); rc.live_pin_bars=1; rc.live_hop_bars=1; rc.live_anchor_bars=0
    rc.jit.enc_lat_mode = lat_m; rc.jit.enc_hint_mode = "frozen"; rc.jit.LIVE_LMARGIN_F = lm
    rc.max_step_ms = 0.0
    rc.set_style(STYLE, denoise=0.7, bpm=90, key="C minor"); rc.start()
    cov, u = capture(40); rc.stop()
    s = fifths(cov)
    sf.write(f"{OUT}/lmargin_{tag.replace(' ','_')}.wav", cov.astype(np.float32), SR)
    print(f"[{tag:>10}] 8s-bins=[{' '.join(f'{v:.2f}' for v in s)}]  drift={s[-1]-s[0]:+.3f}  "
          f"worst_step={rc.max_step_ms:.0f}ms  underruns={u}", flush=True)
rc.close()
print("PASS: chunked at some lm matches whole drift (~0.18) with underruns~0.", flush=True)
