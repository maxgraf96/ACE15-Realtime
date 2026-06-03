"""Validate live source separation: (1) whole-file 4-stem split (write WAVs + check the
sum reconstructs the mix), (2) ROLLING per-chunk separation with a discard margin matches
the whole-file drum stem (so live, chunk-by-chunk separation is sound), (3) cost.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from engine import mps_compat, loader  # noqa
from engine.separation import StemSeparator, STEMS, SR

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "assets", "source.wav")
OUT = os.path.join(ROOT, "test_output", "diag"); os.makedirs(OUT, exist_ok=True)
SPF = 1920
sep = StemSeparator(device="mps")
wav = loader.load_audio(SRC, duration=18).waveform.float()   # [2, N] @48k
print(f"[probe] {os.path.basename(SRC)} {wav.shape[1]/SR:.1f}s  stems={STEMS}", flush=True)

# (1) whole-file split: each stem + reconstruction check
import soundfile as sf
full = {}
for st in STEMS:
    full[st] = sep.separate(wav, [st]).numpy()         # [2, N]
    sf.write(f"{OUT}/sep_{st}.wav", full[st].T, SR)
recon = sum(full.values())
err = np.abs(recon - wav.numpy()).mean() / (np.abs(wav.numpy()).mean() + 1e-9)
print(f"[1] wrote 4 stems; sum-vs-mix rel-err={err:.3f} (low => clean 4-way split)", flush=True)
for st in STEMS:
    e = np.sqrt((full[st] ** 2).mean())
    print(f"      {st:7s} rms={e:.4f}", flush=True)

# (2) rolling per-chunk drum separation vs whole-file drum stem (margin check)
ref_drums = full["drums"][0]
CF, MF = 25, 36                                          # 1.0s chunk, ~1.44s margin
roll = np.zeros_like(ref_drums); enc = 0
while (enc + CF + MF) * SPF <= wav.shape[1]:
    lo = max(0, enc - MF); hi = enc + CF + MF
    sub = wav[:, lo * SPF:hi * SPF]
    stem = sep.separate(sub, ["drums"]).numpy()[0]      # mono drum stem of the window
    take = enc - lo
    roll[enc * SPF:(enc + CF) * SPF] = stem[take * SPF:(take + CF) * SPF]
    enc += CF
L = enc * SPF
a, b = roll[MF * SPF:L], ref_drums[MF * SPF:L]          # skip the very start
corr = float(np.corrcoef(a, b)[0, 1])
print(f"[2] rolling vs whole-file drum stem: corr={corr:.3f} (want >~0.9 -> margin OK)", flush=True)

# (3) cost: per ~1s chunk
n = int(SR * (CF + 2 * MF) / 25)
x = wav[:, :n]
_ = sep.separate(x, ["drums"]); ts = []
for _ in range(3):
    t = time.perf_counter(); sep.separate(x, ["drums"]); ts.append(time.perf_counter() - t)
dt = min(ts)
print(f"[3] separate {n/SR:.1f}s window = {dt*1000:.0f}ms  RTF={dt/(CF/25):.3f} per 1s chunk", flush=True)
print("VERDICT: clean split + rolling corr high + RTF tiny => live 'only drums' is viable", flush=True)
