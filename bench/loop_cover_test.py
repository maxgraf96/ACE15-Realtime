import os, sys, time
sys.path.insert(0, "/Users/max/Code/ACE15-Realtime")
import numpy as np, torch, soundfile as sf
from engine import mps_compat, loader
mps_compat.force_bf16_on_mps()
from engine.realtime import RealtimeCover, SR
OUT="/Users/max/Code/ACE15-Realtime/test_output/diag"; os.makedirs(OUT,exist_ok=True)
PIANO="/Users/max/Music/Ableton/Noiiz/SynapticJourney_Noiiz/Loops/Keys/90_C_BriarPatchVintagePiano_01_695.wav"
STYLE="Instrumental jazz trio, warm upright bass, brushed drums, piano comping"
def sc(x,t=0.92):
    a=np.abs(x); return np.where(a<=t,x,np.sign(x)*(t+(1-t)*np.tanh((a-t)/(1-t)))).astype(np.float32)
def instab(x):
    m=x.mean(1).astype(np.float64); nfft,hop=2048,512; w=np.hanning(nfft)
    F=np.array([np.abs(np.fft.rfft(m[i:i+nfft]*w)) for i in range(0,len(m)-nfft,hop)])
    if len(F)<2: return 0.0
    e=F.sum(1); thr=max(1e-6,0.05*np.median(e)); d=np.abs(np.diff(F,axis=0)).sum(1)/(F[1:].sum(1)+1e-9); k=e[1:]>thr
    return float(np.mean(d[k])) if k.any() else 0.0
full=loader.load_audio(PIANO).waveform.float()
bar=60.0/90*4*SR; loop4=full[:, :int(round(4*bar))]; N=loop4.shape[1]
rc=RealtimeCover(device="mps",steps=8,lookahead_s=6.0,config_path="acestep-v15-xl-turbo")
rc.begin_live(); rc.set_style(STYLE,denoise=0.7,bpm=90,key="C minor"); rc.start()
block=2048; period=block/SR; t0=time.perf_counter(); nxt=t0; fed=0
cov_out=[]; org_out=[]; in_fed=[]; locked_at=None
need=55*SR; got=0
while got<need and time.perf_counter()<t0+need/SR+120:
    s=fed%N; ch=loop4[:,s:s+block]
    if ch.shape[1]<block: ch=torch.cat([ch,loop4[:,:block-ch.shape[1]]],1)
    rc.feed_input(ch); fed+=block
    o=rc.read(block)
    cov_out.append(o[:,:2].copy()); org_out.append(o[:,2:].copy()); in_fed.append(ch.t().numpy().copy())
    if locked_at is None and rc.loop_locked: locked_at=got/SR
    got+=block
    nxt+=period; dt=nxt-time.perf_counter()
    if dt>0: time.sleep(dt)
    else: nxt=time.perf_counter()
rc.stop()
cov=np.concatenate(cov_out,0); org=np.concatenate(org_out,0); inp=np.concatenate(in_fed,0)
print(f"loop_locked={rc.loop_locked} bars={rc.loop_bars} locked_at~{locked_at}s underruns={rc.underruns}",flush=True)
# steady (post-lock) cleanliness
post=cov[int((locked_at or 30)*SR)+SR:] if locked_at else cov[-20*SR:]
print(f"post-lock cover instab = {instab(post):.3f}  (clean ~0.22; warbly ~0.5)",flush=True)
# SYNC: cross-correlate the consumed orig (cover's source) vs the concurrently-fed input
seg=slice(int((locked_at or 30)*SR)+SR, int((locked_at or 30)*SR)+SR+int(8*SR))
a=org[seg].mean(1).astype(np.float64); b=inp[seg].mean(1).astype(np.float64)
n=min(len(a),len(b)); a=a[:n]-a[:n].mean(); b=b[:n]-b[:n].mean()
corr=np.fft.irfft(np.fft.rfft(a)*np.conj(np.fft.rfft(b)),n=n)
lag=int(np.argmax(corr)); lag=lag if lag<n//2 else lag-n
print(f"sync lag (consumed-orig vs fed-input) = {lag} samples ({lag/SR*1000:+.0f} ms)  (want ~0)",flush=True)
sf.write(f"{OUT}/loopmode_cover.wav", sc(cov), SR)
print("wrote loopmode_cover.wav",flush=True)
rc.close()
