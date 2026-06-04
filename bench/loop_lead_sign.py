"""Verify SIGNED AI-sync (loop_lead): +ms => cover plays EARLIER (negative lag / ahead),
-ms => cover plays LATER (positive lag / behind). Exact loop, fast model."""
import os, sys, time
sys.path.insert(0, "/Users/max/Code/ACE15-Realtime")
import numpy as np, torch
from engine import mps_compat, loader
mps_compat.force_bf16_on_mps()
from engine.realtime import RealtimeCover, SR
PIANO = "/Users/max/Music/Ableton/Noiiz/SynapticJourney_Noiiz/Loops/Keys/90_C_BriarPatchVintagePiano_01_695.wav"
STYLE = "Instrumental jazz trio, warm upright bass, brushed drums, piano comping"

def lag_ms(org, inp, c0, c1):
    a = org[c0:c1].mean(1).astype(np.float64); b = inp[c0:c1].mean(1).astype(np.float64)
    n = min(len(a), len(b)); a = a[:n]-a[:n].mean(); b = b[:n]-b[:n].mean()
    corr = np.fft.irfft(np.fft.rfft(a)*np.conj(np.fft.rfft(b)), n=n)
    lag = int(np.argmax(corr)); lag = lag if lag < n//2 else lag-n
    return lag/SR*1000.0

def run(lead_ms, bpm=90, bars=4):
    full = loader.load_audio(PIANO).waveform.float()
    N = int(round(bars*60.0/bpm*4*SR)); base = full[:, :N]
    rc = RealtimeCover(device="mps", steps=8, lookahead_s=6.0, config_path="acestep-v15-turbo")
    rc.begin_live(); rc.loop_bars_hint = bars; rc.loop_lead_s = lead_ms/1000.0
    rc.set_style(STYLE, denoise=0.7, bpm=bpm, key="C minor"); rc.start()
    block = 2048; period = block/SR; t0 = time.perf_counter(); nxt = t0; fed = 0
    org_o=[]; in_o=[]; locked_at=None; need=45*SR; got=0
    while got < need and time.perf_counter() < t0 + need/SR + 90:
        s = fed % N; ch = base[:, s:s+block]
        if ch.shape[1] < block: ch = torch.cat([ch, base[:, :block-ch.shape[1]]], 1)
        rc.feed_input(ch); fed += block
        o = rc.read(block); org_o.append(o[:, 2:].copy()); in_o.append(ch.t().numpy().copy())
        if locked_at is None and rc.loop_locked: locked_at = got/SR
        got += block; nxt += period; dt = nxt-time.perf_counter()
        if dt > 0: time.sleep(dt)
        else: nxt = time.perf_counter()
    rc.stop()
    org = np.concatenate(org_o, 0); inp = np.concatenate(in_o, 0); la = (locked_at or 25)
    ms = lag_ms(org, inp, int((la+1)*SR), int((la+1)*SR)+int(6*SR))
    sign = "AHEAD" if ms < -5 else ("BEHIND" if ms > 5 else "ALIGNED")
    print(f"lead={lead_ms:+5.0f}ms -> measured lag {ms:+6.0f}ms  ({sign})", flush=True)
    rc.close()

if __name__ == "__main__":
    run(0)
    run(+200)
    run(-200)
