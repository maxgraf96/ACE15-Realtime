"""Validate the frontier-crossfade RE-ANCHOR fix for live-mode autoregressive drift.

Live cover of the looped piano, captured long enough for drift to grow, with the
spectral-instability metric measured per QUARTER so we can see the drift trajectory
(clean baseline stays flat; drift ramps up). Compare re-anchor OFF vs ON.

PASS = with anchor ON the last quarter's instab is close to the first quarter's
(drift reset), instead of ramping ~0.30 -> ~0.50 like anchor OFF.
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

def instab(x):                       # mean energy-normalised frame-to-frame spectral change
    m = x.mean(1).astype(np.float64); nfft, hop = 2048, 512; w = np.hanning(nfft)
    F = np.array([np.abs(np.fft.rfft(m[i:i+nfft]*w)) for i in range(0, len(m)-nfft, hop)])
    if len(F) < 2: return 0.0
    e = F.sum(1); thr = max(1e-6, 0.05 * np.median(e))     # ignore near-silent frames (blow up the ratio)
    d = np.abs(np.diff(F, axis=0)).sum(1) / (F[1:].sum(1) + 1e-9)
    keep = e[1:] > thr
    return float(np.mean(d[keep])) if keep.any() else 0.0

def quarters(cov):
    n = len(cov) // 4
    return [instab(cov[i*n:(i+1)*n]) for i in range(4)]

rc = RealtimeCover(device="mps", steps=8, window_s=20.0, lookahead_s=2.0, config_path="acestep-v15-xl-turbo")
src = loader.load_audio(PIANO, duration=11).waveform.float(); N = src.shape[1]

def capture(secs):                   # prime past startup silence, then collect `secs` of steady cover
    block = 2048; period = block / SR; out = []; cap = False; got = 0; need = int(secs*SR); fed = 0
    t0 = time.perf_counter(); nxt = t0; u0 = rc.underruns
    while got < need and time.perf_counter() < t0 + secs + 90:
        if rc.live:
            s = fed % N; ch = src[:, s:s+block]
            if ch.shape[1] < block: ch = torch.cat([ch, src[:, :block-ch.shape[1]]], 1)
            rc.feed_input(ch); fed += block
        o = rc.read(block); cov = o[:, :2]
        if not cap and np.abs(cov).max() > 1e-3: cap = True; u0 = rc.underruns
        if cap: out.append(cov.copy()); got += block
        nxt += period; dt = nxt - time.perf_counter()
        if dt > 0: time.sleep(dt)
        else: nxt = time.perf_counter()
    return np.concatenate(out, 0), rc.underruns - u0

# (tag, anchor_bars, dbg_whole): baseline / isolate live source pipeline / the re-anchor fix
for tag, anchor, whole in [("off", 0, False), ("whole", 0, True), ("2bar", 2, False)]:
    rc.reset(); rc.begin_live(); rc.live_pin_bars = 1; rc.live_hop_bars = 1; rc.live_anchor_bars = anchor
    rc.jit._dbg_whole = whole
    rc.set_style(STYLE, denoise=0.7, bpm=90, key="C minor"); rc.start()
    cov, u = capture(32); rc.stop()
    q = quarters(cov)
    sf.write(f"{OUT}/warble_anchor_{tag}.wav", softclip(cov), SR)
    print(f"[{tag:>5}] quarters instab = [{q[0]:.3f} {q[1]:.3f} {q[2]:.3f} {q[3]:.3f}]  "
          f"drift(q4-q1)={q[3]-q[0]:+.3f}  reanchors={rc.reanchors} skips={rc.anchor_skips}  "
          f"regens={rc.regens}  underruns={u}", flush=True)
rc.close()
print("baseline file-long ~0.38 plateau; PASS if 2bar holds near q1 (~0.20-0.28) vs off ramping to ~0.52.", flush=True)
