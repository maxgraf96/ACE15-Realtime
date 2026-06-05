"""Diagnose the LINK-path cover quality. Conductor peer = fake Ableton; feed the piano loop
LINK-LOCKED; measure the COVER instab. If high (~0.5) the Link slice/render warbles; clean ~0.22."""
import os, sys, asyncio, threading, time
sys.path.insert(0, "/Users/max/Code/ACE15-Realtime")
import numpy as np, torch
from engine import mps_compat, loader
mps_compat.force_bf16_on_mps()
from engine.realtime import RealtimeCover, SR
from sidecar.link_sync import get_link
PIANO = "/Users/max/Music/Ableton/Noiiz/SynapticJourney_Noiiz/Loops/Keys/90_C_BriarPatchVintagePiano_01_695.wav"
STYLE = "Instrumental jazz trio, warm upright bass, brushed drums, piano comping"
def instab(x):
    m = x.mean(1).astype(np.float64); nfft, hop = 2048, 512; w = np.hanning(nfft)
    F = np.array([np.abs(np.fft.rfft(m[i:i+nfft]*w)) for i in range(0, len(m)-nfft, hop)])
    if len(F) < 2: return 0.0
    e = F.sum(1); thr = max(1e-6, 0.05*np.median(e)); d = np.abs(np.diff(F, axis=0)).sum(1)/(F[1:].sum(1)+1e-9); k = e[1:] > thr
    return float(np.mean(d[k])) if k.any() else 0.0
# Use the REAL Ableton as the Link peer (no in-process conductor — a 2nd aalink scheduler thread
# stalls the model load under GIL contention). Requires Ableton open with Link enabled.
link = get_link()
for _ in range(40):
    if link.connected: break
    time.sleep(0.1)
tempo = link.tempo; bars = 4; bpb = 4; loop_beats = bars*bpb
print(f"connected={link.connected} peers={link.peers} tempo={tempo:.3f}", flush=True)
print("stage: loading audio + model…", flush=True)
N = int(round(bars*60.0/tempo*bpb*SR)); base = loader.load_audio(PIANO).waveform.float()[:, :int(round(4*60.0/90*4*SR))]
if base.shape[1] != N:   # linear interpolate (fast; AF.resample builds a giant kernel for coprime ratios -> hang)
    base = torch.nn.functional.interpolate(base.unsqueeze(0), size=N, mode="linear", align_corners=False).squeeze(0)
base = base[:, :N].contiguous()
def cread(buf, s, blk):
    L = buf.shape[1]; s %= L
    return buf[:, s:s+blk] if s+blk <= L else torch.cat([buf[:, s:], buf[:, :blk-(L-s)]], 1)
rc = RealtimeCover(device="mps", steps=8, lookahead_s=6.0, config_path="acestep-v15-turbo")
rc.begin_live(); rc.link = link; rc.loop_bars_hint = bars
rc.set_style(STYLE, denoise=0.7, bpm=round(tempo, 2), key="C minor"); rc.start()
print("stage: model up, feeding…", flush=True)
cov_o=[]; locked_at=None; got=0; peak=0.0; need=55*SR; last_beat=link.beat; spb=60.0/tempo
t0=time.perf_counter()
while got<need and time.perf_counter()-t0 < need/SR + 60:
    time.sleep(0.01); cur=link.beat; nf=int(round((cur-last_beat)*spb*SR))
    if nf<=0: continue
    s=int((last_beat%loop_beats)/loop_beats*N); rc.feed_input(cread(base,s,nf)); last_beat=cur
    o=rc.read(nf); cov_o.append(o[:,:2].copy())
    if o[:,:2].size: peak=max(peak,float(np.abs(o[:,:2]).max()))
    if locked_at is None and rc.loop_locked: locked_at=got/SR; print(f"  LOCKED@{locked_at:.1f}s link_active={rc.link_active}",flush=True)
    got+=nf
rc.stop()
cov=np.concatenate(cov_o,0); la=locked_at or 25; post=cov[int((la+3)*SR):]
print(f"[diag] locked={rc.loop_locked} link_active={rc.link_active} peak={peak:.3f}")
print(f"  LINK-path post-lock cover instab = {instab(post):.3f}  (clean ~0.22; warbly ~0.5)")
rc.close()
