"""Is the warble small-window ROLLING in general, or LIVE-specific (rolling encode/hints)?
Run the FILE producer (preloaded source) at the SAME small window/pin/hop as live, on the
piano. If file-small-window is clean (low instab, no warble after the first window) -> the
bug is live-specific. If it warbles too -> small-window rolling is the issue.
"""
import os, sys, time
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
    return float(np.mean(np.abs(np.diff(F, axis=0)).sum(1) / (F[1:].sum(1)+1e-9)))

rc = RealtimeCover(device="mps", steps=8, window_s=20.0, pin_s=4.0, lookahead_s=2.0, config_path="acestep-v15-xl-turbo")
rc.load_track(PIANO)

def capture(secs):
    block = 2048; period = block / SR; out = []; cap = False; got = 0; need = int(secs*SR)
    t0 = time.perf_counter(); nxt = t0
    while got < need and time.perf_counter() < t0 + secs + 60:
        o = rc.read(block); cov = o[:, :2]
        if not cap and np.abs(cov).max() > 1e-3: cap = True
        if cap: out.append(cov.copy()); got += block
        nxt += period; dt = nxt - time.perf_counter()
        if dt > 0: time.sleep(dt)
        else: nxt = time.perf_counter()
    return np.concatenate(out, 0)

# FILE producer at small windows (pin = hop, matching the live bar split at 90 BPM)
for W, P in [(5.33, 2.67), (10.67, 5.33), (20.0, 10.0)]:
    rc.reset()
    rc._pin_f = round(P * FPS); rc._hop_f = round(W * FPS) - rc._pin_f
    rc.set_style(STYLE, denoise=0.7, bpm=90, key="C minor"); rc.start()
    cov = capture(16); rc.stop()
    sf.write(f"{OUT}/warble_file_win{W:.0f}.wav", softclip(cov), SR)
    print(f"[file] window={W:.1f}s pin={P:.1f}s hop={W-P:.1f}s  instab={instab(cov):.3f}", flush=True)
rc.close()
print("VERDICT: file-small-window clean => LIVE-specific (encode/hints); warbly => small-window rolling.", flush=True)
