"""Measure loop-cover SYNC at lead=0, EARLY vs LATE, to diagnose the 'AI comes in
BEFORE the loop start' (ahead) report. Feeds a loop whose TRUE period is a hair off
the nominal bars*60/bpm (like a real DAW region) and a perfectly-periodic one.

Reports the cross-correlation lag between the consumed ORG (pushed at the same phase
as the cover) and the concurrently-fed INPUT, early and late. lag>0 = cover BEHIND the
input; lag<0 = cover AHEAD. Want ~0 and stable (no drift)."""
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
    return lag, lag/SR*1000.0

def run(true_period, label, bars=4, bpm=90):
    full = loader.load_audio(PIANO).waveform.float()
    # build a loop of EXACTLY true_period samples (resample the nominal 4-bar slice)
    nominal = int(round(bars*60.0/bpm*4*SR))
    base = full[:, :nominal]
    if true_period != nominal:
        import torchaudio.functional as AF
        base = AF.resample(base, nominal, true_period)[:, :true_period]
    N = base.shape[1]
    rc = RealtimeCover(device="mps", steps=8, lookahead_s=6.0, config_path="acestep-v15-turbo")
    rc.begin_live(); rc.loop_bars_hint = bars
    rc.set_style(STYLE, denoise=0.7, bpm=bpm, key="C minor"); rc.start()
    block = 2048; period = block/SR; t0 = time.perf_counter(); nxt = t0; fed = 0
    cov_o=[]; org_o=[]; in_o=[]; locked_at=None
    need = 70*SR; got = 0
    while got < need and time.perf_counter() < t0 + need/SR + 120:
        s = fed % N; ch = base[:, s:s+block]
        if ch.shape[1] < block: ch = torch.cat([ch, base[:, :block-ch.shape[1]]], 1)
        rc.feed_input(ch); fed += block
        o = rc.read(block)
        cov_o.append(o[:, :2].copy()); org_o.append(o[:, 2:].copy()); in_o.append(ch.t().numpy().copy())
        if locked_at is None and rc.loop_locked: locked_at = got/SR
        got += block
        nxt += period; dt = nxt-time.perf_counter()
        if dt > 0: time.sleep(dt)
        else: nxt = time.perf_counter()
    rc.stop()
    org = np.concatenate(org_o, 0); inp = np.concatenate(in_o, 0)
    la = (locked_at or 30)
    e_l, e_ms = lag_ms(org, inp, int((la+1)*SR), int((la+1)*SR)+int(6*SR))
    l_l, l_ms = lag_ms(org, inp, int((la+1)*SR)+int(40*SR), int((la+1)*SR)+int(46*SR))
    print(f"[{label}] true_P={N} nominal={nominal} locked@{la:.1f}s "
          f"EARLY lag={e_l}({e_ms:+.0f}ms) LATE lag={l_l}({l_ms:+.0f}ms) "
          f"DRIFT={l_ms-e_ms:+.0f}ms underruns={rc.underruns}", flush=True)
    rc.close()

if __name__ == "__main__":
    nominal = int(round(4*60.0/90*4*SR))
    run(nominal, "exact")
    run(int(nominal*1.03), "+3% region (longer)")
    run(int(nominal*0.97), "-3% region (shorter)")
