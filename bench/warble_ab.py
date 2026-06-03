"""Localize the live warble cleanly: PRIMED (skip startup/silence) steady captures of a
known-good FILE cover vs LIVE covers at increasing window sizes, same piano + XL + Amount
0.7 + looped input. Writes WAVs + an objective spectral-instability metric (warble => high
frame-to-frame spectral change in sustained material).
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
    return float(np.mean(np.abs(np.diff(F, axis=0)).sum(1) / (F[1:].sum(1)+1e-9)))

rc = RealtimeCover(device="mps", steps=8, window_s=20.0, lookahead_s=2.0, config_path="acestep-v15-xl-turbo")
src = loader.load_audio(PIANO, duration=11).waveform.float(); N = src.shape[1]

def capture(secs):                   # prime past the startup silence, then collect `secs` of steady cover
    block = 2048; period = block / SR; out = []; cap = False; got = 0; need = int(secs*SR); fed = 0
    t0 = time.perf_counter(); nxt = t0; u0 = rc.underruns
    while got < need and time.perf_counter() < t0 + secs + 60:
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

rc.load_track(PIANO); rc.set_style(STYLE, denoise=0.7, bpm=90, key="C minor"); rc.start()
cov, u = capture(14); rc.stop()
sf.write(f"{OUT}/warble_file.wav", softclip(cov), SR)
print(f"[file ] window20  instab={instab(cov):.3f}  steady_underruns={u}", flush=True)

for dn in (0.7, 0.9):                # does more source-anchoring (higher Amount) reduce drift?
    rc.reset(); rc.begin_live(); rc.live_pin_bars = 1; rc.live_hop_bars = 1; rc.live_anchor_bars = 0
    rc.set_style(STYLE, denoise=dn, bpm=90, key="C minor"); rc.start()
    cov, u = capture(24); rc.stop()
    h = len(cov) // 2
    sf.write(f"{OUT}/warble_live_dn{int(dn*100)}.wav", softclip(cov), SR)
    print(f"[live ] denoise={dn}  1st-half={instab(cov[:h]):.3f}  2nd-half={instab(cov[h:]):.3f}  "
          f"underruns={u}", flush=True)
rc.close()
print("VERDICT: compare instab (file=clean baseline) + listen. Higher live instab => window-starved warble.", flush=True)
