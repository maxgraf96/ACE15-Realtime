import os, sys, time
sys.path.insert(0, "/Users/max/Code/ACE15-Realtime")
import numpy as np, torch, soundfile as sf
from engine import mps_compat, loader
mps_compat.force_bf16_on_mps()
from engine.realtime import RealtimeCover, SR
OUT="/Users/max/Code/ACE15-Realtime/test_output/diag"
PIANO="/Users/max/Music/Ableton/Noiiz/SynapticJourney_Noiiz/Loops/Keys/90_C_BriarPatchVintagePiano_01_695.wav"
STYLE="Instrumental jazz trio, warm upright bass, brushed drums, piano comping"
def sc(x,t=0.92):
    a=np.abs(x); return np.where(a<=t,x,np.sign(x)*(t+(1-t)*np.tanh((a-t)/(1-t)))).astype(np.float32)
def instab(x):
    if np.abs(x).max()<1e-4: return 0.0
    m=x.mean(1).astype(np.float64); nfft,hop=2048,512; w=np.hanning(nfft)
    F=np.array([np.abs(np.fft.rfft(m[i:i+nfft]*w)) for i in range(0,len(m)-nfft,hop)])
    if len(F)<2: return 0.0
    e=F.sum(1); thr=max(1e-6,0.05*np.median(e)); d=np.abs(np.diff(F,axis=0)).sum(1)/(F[1:].sum(1)+1e-9); k=e[1:]>thr
    return float(np.mean(d[k])) if k.any() else 0.0
full=loader.load_audio(PIANO).waveform.float(); bar=60.0/90*4*SR; loop4=full[:, :int(round(4*bar))]; N=loop4.shape[1]
rc=RealtimeCover(device="mps",steps=8,lookahead_s=2.0,config_path="acestep-v15-xl-turbo")
rc.begin_live(); rc.set_style(STYLE,denoise=0.7,bpm=90,key="C minor"); rc.start()
block=2048; period=block/SR; t0=time.perf_counter(); nxt=t0; fed=0; got=0; need=45*SR
out=[]; lock_t=None
while got<need and time.perf_counter()<t0+need/SR+120:
    s=fed%N; ch=loop4[:,s:s+block]
    if ch.shape[1]<block: ch=torch.cat([ch,loop4[:,:block-ch.shape[1]]],1)
    rc.feed_input(ch); fed+=block
    o=rc.read(block); out.append(o[:,:2].copy()); got+=block
    if lock_t is None and rc.loop_locked: lock_t=got/SR
    nxt+=period; dt=nxt-time.perf_counter()
    if dt>0: time.sleep(dt)
    else: nxt=time.perf_counter()
rc.stop(); rc.close()
cov=np.concatenate(out,0)
lt=lock_t or 99
pre=cov[:int(lt*SR)] if lt<99 else cov
post=cov[int(lt*SR)+SR:] if lt<99 else cov[-10*SR:]
print(f"locked_at={lt:.1f}s",flush=True)
print(f"PRE-lock  AI RMS = {np.sqrt((pre**2).mean()):.5f}  peak={np.abs(pre).max():.4f}  (want ~0 = SILENT, no warble)",flush=True)
print(f"POST-lock instab = {instab(post):.3f}  peak={np.abs(post).max():.3f}  (want clean ~0.22)",flush=True)
sf.write(f"{OUT}/silentwarm_cover.wav", sc(cov), SR)
print("wrote silentwarm_cover.wav (silent intro -> clean loop)",flush=True)
