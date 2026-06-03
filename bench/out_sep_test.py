"""Validate OUTPUT stem separation: separate a generated COVER into stems, rolling per
1s slice with a small margin (as the producer would), vs whole-file. If the rolling drum
stem matches the whole-file one, doing it per-push slice is sound. Picks the margin.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from engine import loader
from engine.separation import StemSeparator, SR

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAND = [os.path.join(ROOT, "test_output", "diag", "rt_live_cover.wav"),
        os.path.join(ROOT, "assets", "source.wav")]
SRC = next(p for p in CAND if os.path.exists(p))
SPF, SL = 1920, 25
sep = StemSeparator(device="mps")
wav = loader.load_audio(SRC, duration=16).waveform.float()
print(f"[probe] {os.path.basename(SRC)} {wav.shape[1]/SR:.1f}s", flush=True)

ref = sep.separate(wav, ["drums"]).numpy()[0]   # whole-file drum stem (reference)
for m in (12, 20, 36):
    roll = np.zeros_like(ref); a = 0
    while (a + SL + m) * SPF <= wav.shape[1]:
        lo = max(0, a - m); hi = a + SL + m
        win = sep.separate(wav[:, lo * SPF:hi * SPF], ["drums"]).numpy()[0]
        take = a - lo
        roll[a * SPF:(a + SL) * SPF] = win[take * SPF:(take + SL) * SPF]
        a += SL
    L = a * SPF
    corr = float(np.corrcoef(roll[m * SPF:L], ref[m * SPF:L])[0, 1])
    print(f"  margin={m:2d} ({m/25:.2f}s): rolling-vs-whole drum corr={corr:.3f}", flush=True)
print("VERDICT: pick smallest margin with corr>~0.97 (fits live decode headroom=24 frames)", flush=True)
