"""Is the drift INHERENT to long pinned rolling, or LIVE-specific?

The piano loops at 11s, so the FILE producer re-anchors (loop restart -> cold prime)
every ~11s, before drift sets in. Concatenate the piano x4 (~44s) so a 32s capture
stays within ONE loop = a long pinned roll with NO restart. If the file producer
drifts too (quarters ramp like live), drift is INHERENT -> the re-anchor fix is the
right idea. If it stays flat, the bug is live-specific (something else differs).
"""
import os, sys, time, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from engine import mps_compat, loader
mps_compat.force_bf16_on_mps()
from engine.realtime import RealtimeCover, SR, FPS

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

def quarters(cov):
    n = len(cov)//4; return [instab(cov[i*n:(i+1)*n]) for i in range(4)]

# build a ~44s track = piano x4 so a 32s capture never hits the loop restart
src = loader.load_audio(PIANO, duration=11).waveform.float()
long = torch.cat([src]*4, dim=1)
tmp = os.path.join(tempfile.gettempdir(), "piano_x4.wav")
sf.write(tmp, long.t().numpy(), SR)
print(f"long track = {long.shape[1]/SR:.1f}s", flush=True)

rc = RealtimeCover(device="mps", steps=8, window_s=20.0, lookahead_s=2.0, config_path="acestep-v15-xl-turbo")
rc.load_track(tmp)

def capture(secs):
    block = 2048; period = block/SR; out=[]; cap=False; got=0; need=int(secs*SR)
    t0=time.perf_counter(); nxt=t0
    while got<need and time.perf_counter()<t0+secs+90:
        o = rc.read(block); cov=o[:,:2]
        if not cap and np.abs(cov).max()>1e-3: cap=True
        if cap: out.append(cov.copy()); got+=block
        nxt+=period; dt=nxt-time.perf_counter()
        if dt>0: time.sleep(dt)
        else: nxt=time.perf_counter()
    return np.concatenate(out,0)

# FILE producer at the SAME small bar-window as live (90 BPM, pin=hop=1 bar)
bar = 60.0/90*4
rc.reset(); rc._pin_f = round(bar*FPS); rc._hop_f = round(bar*FPS)
rc.set_style(STYLE, denoise=0.7, bpm=90, key="C minor"); rc.start()
cov = capture(32); rc.stop(); rc.close()
q = quarters(cov)
sf.write(f"{OUT}/drift_file_long.wav", softclip(cov), SR)
print(f"[file long] quarters instab = [{q[0]:.3f} {q[1]:.3f} {q[2]:.3f} {q[3]:.3f}]  drift(q4-q1)={q[3]-q[0]:+.3f}", flush=True)
print("INHERENT if file-long drifts like live (ramps up); LIVE-SPECIFIC if flat.", flush=True)
