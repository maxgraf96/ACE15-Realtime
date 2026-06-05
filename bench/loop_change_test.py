"""Re-lock on loop change: feed loop A, lock, then SWAP to loop B. Verify the engine detects the
change, drops the stale cover, and RE-LOCKS on B (no warble from a stale/wrong cover). Onset path
(rc.link=None) so it runs reliably headless. Tracks loop_locked transitions + final cover match."""
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
N = int(round(4*60.0/90*4*SR))
A = full[:, :N]
# loop B: a clearly different loop (later region of the track if available, else pitch/var-shifted A)
B = full[:, N:2*N] if full.shape[1] >= 2*N else None
if B is None or B.shape[1] < N:
    import torchaudio.functional as AF
    B = AF.resample(A, N, int(N*1.18))[:, :N]   # ~+3 semitone shift -> clearly different content
B = B[:, :N].contiguous()
rc = RealtimeCover(device="mps", steps=8, lookahead_s=6.0, config_path="acestep-v15-turbo")
rc.begin_live(); rc.link = None; rc.loop_bars_hint = 4
rc.set_style(STYLE, denoise=0.7, bpm=90, key="C minor"); rc.start()
block=2048; period=block/SR; t0=time.perf_counter(); nxt=t0; fed=0
locks=[]; prev=False; got=0; swap_at=28*SR; total=62*SR; cov_o=[]
while got<total and time.perf_counter()<t0+total/SR+120:
    cur_loop = A if got < swap_at else B           # SWAP loop at 28s
    s=fed%N; ch=cur_loop[:,s:s+block]
    if ch.shape[1]<block: ch=torch.cat([ch,cur_loop[:,:block-ch.shape[1]]],1)
    rc.feed_input(ch); fed+=block
    o=rc.read(block); cov_o.append(o[:,:2].copy())
    if rc.loop_locked != prev:
        locks.append((round(got/SR,1), rc.loop_locked)); prev=rc.loop_locked
    got+=block; nxt+=period; dt=nxt-time.perf_counter()
    if dt>0: time.sleep(dt)
    else: nxt=time.perf_counter()
rc.stop()
cov=np.concatenate(cov_o,0)
# instab on A (pre-swap, post initial lock) and on B (well after swap+re-lock)
ia = instab(cov[int(20*SR):int(27*SR)]); ib = instab(cov[int(52*SR):int(60*SR)])
print(f"[loop-change] loop_locked transitions (s, locked): {locks}", flush=True)
print(f"  instab on A (pre-swap) = {ia:.3f}   instab on B (post re-lock) = {ib:.3f}  (clean ~0.22)", flush=True)
# PASS: locked on A, dropped after swap (a False transition after 28s), re-locked (True after that), both clean
relocked = any(not lk for t,lk in locks if t>=28) and any(lk for t,lk in locks if t>=29)
print("PASS: re-locked on the new loop, both covers clean" if (relocked and ia<0.3 and ib<0.3) else "FAIL", flush=True)
rc.close()
