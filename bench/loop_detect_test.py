import os, sys
sys.path.insert(0, "/Users/max/Code/ACE15-Realtime")
import torch
from engine import mps_compat, loader
mps_compat.force_bf16_on_mps()
from engine.jit import JITCover, SR
PIANO="/Users/max/Music/Ableton/Noiiz/SynapticJourney_Noiiz/Loops/Keys/90_C_BriarPatchVintagePiano_01_695.wav"
jit=JITCover(device="mps",steps=8,config_path="acestep-v15-xl-turbo")
jit.begin_live(); jit.bpm=90
full=loader.load_audio(PIANO).waveform.float()
print(f"file length: {full.shape[1]/SR:.2f}s",flush=True)
bar=60.0/90*4*SR
loop4=full[:, :int(round(4*bar))]   # exact 4-bar loop (10.67s)
# feed 2.5 loops of the exact 4-bar loop
jit._live_raw = torch.cat([loop4, loop4, loop4[:, :loop4.shape[1]//2]], 1)
P,c,B = jit.detect_loop()
print(f"[exact 4-bar loop fed] -> bars={B} period={P/SR:.2f}s conf={c:.3f}  (expect bars=4)",flush=True)
# test at wrong bpm (120) -> should still find the period in bars-of-120 or fail gracefully
jit.bpm=120; P,c,B=jit.detect_loop()
print(f"[same audio, bpm=120]  -> bars={B} period={P/SR if P else 0:.2f}s conf={c:.3f}",flush=True)
# non-looping: white noise -> should NOT detect
jit.bpm=90; jit._live_raw=torch.randn(2, int(3*4*bar))*0.1
P,c,B=jit.detect_loop()
print(f"[white noise]          -> bars={B} conf={c:.3f}  (expect None/low)",flush=True)
# 2-bar loop
loop2=full[:, :int(round(2*bar))]
jit._live_raw=torch.cat([loop2]*4,1)
P,c,B=jit.detect_loop()
print(f"[exact 2-bar loop fed] -> bars={B} period={P/SR if P else 0:.2f}s conf={c:.3f}  (expect bars=2)",flush=True)
