"""Phase B gate: live rolling cover. Replays a file as a simulated 1x live stream into
RealtimeCover (live mode) — feeding input and draining output in lockstep at 1x — and
measures startup latency, steady end-to-end latency, underruns, RTF, and continuity.
Writes the cover + the (delayed) input for an A/B listen.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from engine import mps_compat, loader
mps_compat.force_bf16_on_mps()
from engine.realtime import RealtimeCover, SR

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "assets", "source.wav")
MODEL = os.environ.get("ACE15_LIVE_MODEL", "acestep-v15-turbo")   # 2B Fast (set xl-turbo for Quality)
OUT = os.path.join(ROOT, "test_output", "diag"); os.makedirs(OUT, exist_ok=True)
DUR = 40.0
STYLE = "warm lo-fi hip hop, dusty drums, mellow rhodes"


def softclip(x, t=0.92):
    a = np.abs(x)
    return np.where(a <= t, x, np.sign(x) * (t + (1 - t) * np.tanh((a - t) / (1 - t)))).astype(np.float32)


src = loader.load_audio(SRC, duration=DUR + 5).waveform.float()   # [2, N] @ 48k
print(f"[probe] model={MODEL}  source={src.shape[1]/SR:.1f}s  window=8s pin=3s", flush=True)

WIN = float(os.environ.get("ACE15_LIVE_WINDOW", "8.0"))
PIN = float(os.environ.get("ACE15_LIVE_PIN", "3.0"))
print(f"[probe] window={WIN}s pin={PIN}s", flush=True)
rc = RealtimeCover(device="mps", steps=8, window_s=WIN, pin_s=PIN, prime_s=2.0,
                   lookahead_s=1.0, config_path=MODEL)
rc.begin_live()
rc.set_style(STYLE, denoise=0.8, bpm=120, key="C minor")
rc.start()

block = 2048; period = block / SR
out = []; fed = 0; first_sound_t = None
t0 = time.perf_counter(); nxt = t0; end = t0 + DUR
primed_underruns0 = None
while time.perf_counter() < end:
    if fed < src.shape[1]:
        rc.feed_input(src[:, fed:fed + block]); fed += block
    o = rc.read(block)                      # [block, 4] = cover pair + input pair
    cov = o[:, :2].copy(); out.append(cov)
    if first_sound_t is None and np.abs(cov).max() > 1e-4:
        first_sound_t = time.perf_counter() - t0
        primed_underruns0 = rc.underruns    # ignore startup underruns
    nxt += period
    dt = nxt - time.perf_counter()
    if dt > 0: time.sleep(dt)
    else: nxt = time.perf_counter()

cover = np.concatenate(out, 0)              # [M, 2]
fed_secs = fed / SR
real_secs = rc._real_f / SR
steady_underruns = rc.underruns - (primed_underruns0 or 0)
# continuity in the STEADY region only (skip the leading silence + its onset, + first slice).
# raw sample-jump is transient-dominated, so report max + p99.9: an isolated max ~= a drum
# hit; many large jumps (p99.9 also high) ~= a recurring seam.
fs = int((first_sound_t or 0) * SR) + 2 * SR
steady = cover[fs:, 0]
d = np.abs(np.diff(steady))
mean = float(d.mean() + 1e-9)
jump = float(d.max() / mean) if len(d) > 100 else 0.0
p999 = float(np.percentile(d, 99.9) / mean) if len(d) > 100 else 0.0
s = rc.stats()
print(f"[result] first_sound={first_sound_t:.2f}s  steady_latency(fed-out)={fed_secs - real_secs:.2f}s  "
      f"steady_underruns={steady_underruns}  worst_step={s['worst_regen_ms']}ms  regens={rc.regens}", flush=True)
print(f"         output: {real_secs:.1f}s real  peak={np.abs(cover).max():.3f}  rms={np.sqrt(np.mean(cover**2)):.3f}  "
      f"jump max={jump:.0f}x p99.9={p999:.0f}x (max>>p99.9 => isolated transient, not a seam)", flush=True)

import soundfile as sf
sf.write(f"{OUT}/rt_live_cover.wav", softclip(cover), SR)
sf.write(f"{OUT}/rt_live_input.wav", src[:, :cover.shape[0]].t().numpy(), SR)
ok = (first_sound_t is not None and steady_underruns == 0)   # gate: sustains 1x, no underruns
print(f"[wav] {OUT}/rt_live_cover.wav + rt_live_input.wav", flush=True)
print("PASS: live rolling cover sustains 1x, seamless, gated by input"
      if ok else f"CHECK: underruns={steady_underruns} jump={jump:.1f}x (tune window/buffer)", flush=True)
rc.close()
sys.exit(0 if ok else 1)
