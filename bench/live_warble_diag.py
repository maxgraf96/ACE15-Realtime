"""Diagnose live warble: does loop-cover LOCK, and is the COVER clean (instab)? Tests the onset
path (no Link) with the 10.67s piano loop. instab ~0.22 = clean (file quality); ~0.5 = warbly.
If it never locks (give_up/continuous), that's the warble source, not the render."""
import os, sys, time
sys.path.insert(0, "/Users/max/Code/ACE15-Realtime")
import numpy as np, torch
from engine import mps_compat, loader
mps_compat.force_bf16_on_mps()
from engine.realtime import RealtimeCover, SR
PIANO = "/Users/max/Music/Ableton/Noiiz/SynapticJourney_Noiiz/Loops/Keys/90_C_BriarPatchVintagePiano_01_695.wav"
STYLE = "Instrumental jazz trio, warm upright bass, brushed drums, piano comping"
def instab(x):
    m = x.mean(1).astype(np.float64); nfft, hop = 2048, 512; w = np.hanning(nfft)
    F = np.array([np.abs(np.fft.rfft(m[i:i+nfft]*w)) for i in range(0, len(m)-nfft, hop)])
    if len(F) < 2: return 0.0
    e = F.sum(1); thr = max(1e-6, 0.05*np.median(e)); d = np.abs(np.diff(F, axis=0)).sum(1)/(F[1:].sum(1)+1e-9); k = e[1:] > thr
    return float(np.mean(d[k])) if k.any() else 0.0
full = loader.load_audio(PIANO).waveform.float()
N = int(round(4*60.0/90*4*SR)); base = full[:, :N]
rc = RealtimeCover(device="mps", steps=8, lookahead_s=6.0, config_path="acestep-v15-turbo")
rc.begin_live(); rc.link = None; rc.loop_bars_hint = 4
rc.set_style(STYLE, denoise=0.7, bpm=90, key="C minor"); rc.start()
block=2048; period=block/SR; t0=time.perf_counter(); nxt=t0; fed=0
cov_o=[]; locked_at=None; got=0; peak=0.0; need=55*SR
while got<need and time.perf_counter()<t0+need/SR+90:
    s=fed%N; ch=base[:,s:s+block]
    if ch.shape[1]<block: ch=torch.cat([ch,base[:,:block-ch.shape[1]]],1)
    rc.feed_input(ch); fed+=block
    o=rc.read(block); cov_o.append(o[:,:2].copy())
    if o[:,:2].size: peak=max(peak,float(np.abs(o[:,:2]).max()))
    if locked_at is None and rc.loop_locked: locked_at=got/SR; print(f"  LOCKED@{locked_at:.1f}s",flush=True)
    got+=block; nxt+=period; dt=nxt-time.perf_counter()
    if dt>0: time.sleep(dt)
    else: nxt=time.perf_counter()
rc.stop()
cov=np.concatenate(cov_o,0); la=locked_at or 25
post=cov[int((la+3)*SR):]
print(f"[diag] locked={rc.loop_locked} give_up(continuous)={not rc.loop_locked} link_active={rc.link_active} peak={peak:.3f}")
print(f"  post-lock cover instab = {instab(post):.3f}  (clean ~0.22; warbly ~0.5)")
rc.close()
