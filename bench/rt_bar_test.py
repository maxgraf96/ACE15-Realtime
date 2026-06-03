"""Phase C gate: is the live output trailing the input by a WHOLE number of bars,
locked to the tempo grid? Runs the live engine at two tempos (one model load), feeds a
file as a 1x stream, and checks the achieved trailing == N * bar, for integer N.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from engine import mps_compat, loader
mps_compat.force_bf16_on_mps()
from engine.realtime import RealtimeCover, SR

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "assets", "source.wav")
MODEL = os.environ.get("ACE15_LIVE_MODEL", "acestep-v15-turbo")
src = loader.load_audio(SRC, duration=30).waveform.float()
rc = RealtimeCover(device="mps", steps=8, lookahead_s=1.0, config_path=MODEL)
print(f"[probe] model={MODEL}", flush=True)


def run(bpm, secs=26):
    rc.reset(); rc.begin_live()
    rc.set_style("warm lo-fi hip hop, dusty drums", denoise=0.8, bpm=bpm, key="C minor")
    rc.start()
    block = 2048; period = block / SR; fed = 0; first = None; und0 = None
    t0 = time.perf_counter(); nxt = t0; end = t0 + secs
    while time.perf_counter() < end:
        if fed < src.shape[1]:
            rc.feed_input(src[:, fed:fed + block]); fed += block
        o = rc.read(block)
        if first is None and np.abs(o[:, :2]).max() > 1e-4:
            first = time.perf_counter() - t0     # output-time of cover-of-frame-0 = the trail
            und0 = rc.underruns                   # ignore startup underruns
        nxt += period; dt = nxt - time.perf_counter()
        if dt > 0: time.sleep(dt)
        else: nxt = time.perf_counter()
    bar = 60.0 / bpm * rc.beats_per_bar
    trail = first or 0.0                          # cover-of-frame-0 plays trail s after it was fed
    target = rc._bar_trail_s                      # engine's bar-rounded target
    steady_under = rc.underruns - (und0 or 0)     # drift after the offset is set
    rc.stop()
    n = trail / bar
    aligned = (abs(target / bar - round(target / bar)) < 0.02   # target is a whole # of bars
               and abs(trail - target) < 0.30                    # cover actually appears at the target
               and steady_under == 0)                            # no drift (offset stays N bars)
    print(f"  bpm={bpm:3d}  bar={bar:.2f}s  trail={trail:.2f}s = {n:.2f} bars  "
          f"target={target:.2f}s ({target/bar:.0f} bars)  steady_underruns={steady_under}  "
          f"{'OK' if aligned else 'OFF-GRID'}", flush=True)
    return aligned


ok = True
for bpm in (120, 90):
    ok = run(bpm) & ok
print("PASS: live output trails by a whole number of bars, grid-locked" if ok
      else "FAIL: output not bar-aligned", flush=True)
rc.close()
sys.exit(0 if ok else 1)
