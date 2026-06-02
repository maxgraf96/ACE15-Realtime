"""Does the user's setting (XL, window 30, DCW on, Amount 0.90) underrun at 1x?

Drives RealtimeCover with the sidecar's exact Quality config and a TRUE 1x
consumer, then reports underruns + worst regen vs lookahead. Underruns => the
glitchy/'metallic' playback the user heard; the lookahead is too small for a
30s-window XL regen.
"""
import sys, os, time, threading
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import mps_compat
mps_compat.force_bf16_on_mps()
from engine.realtime import RealtimeCover, SR

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "source.wav")


def trial(window_s, lookahead_s, label, secs=35):
    rc = RealtimeCover(device="mps", steps=8, window_s=window_s, lookahead_s=lookahead_s,
                       config_path="acestep-v15-xl-turbo")
    rc.load_track(SRC, seconds=60)
    rc.jit.set_dcw(enabled=True)
    rc.set_style("future bass", denoise=0.90)
    rc.start()
    # prime: wait until the (adaptive) buffer is filled, then measure STEADY-STATE
    t_prime = time.perf_counter()
    while rc.buffered_s() < 4.0 and time.perf_counter() - t_prime < 30:
        time.sleep(0.05)
    prime_s = time.perf_counter() - t_prime
    rc.underruns = 0   # ignore startup; count only steady-state gaps
    # TRUE 1x consumer: pull `block` frames every block/SR seconds.
    block = 2048; period = block / SR
    nxt = time.perf_counter(); end = time.perf_counter() + secs
    while time.perf_counter() < end:
        rc.read(block)
        nxt += period
        dt = nxt - time.perf_counter()
        if dt > 0: time.sleep(dt)
        else: nxt = time.perf_counter()
    s = rc.stats()
    print(f"[{label}] window={window_s}s lookahead={lookahead_s}s primed_in={prime_s:.1f}s -> "
          f"STEADY underruns={rc.underruns} worst_regen={s['worst_regen_ms']}ms "
          f"buffered={s['buffered_s']}s regens={s['regens']}", flush=True)
    rc.close(); mps_compat.reclaim()
    return rc.underruns, s['worst_regen_ms']


# With adaptive buffering the static lookahead is just a floor; the buffer should
# auto-grow to cover the ~3.1s regen, so BOTH should be ~0 steady-state underruns.
print("=== user's setting: window 30, lookahead 2.0 (now adaptive) ===", flush=True)
u1, w1 = trial(30.0, 2.0, "window30/LA2.0")
print("=== window 30, lookahead 1.0 floor (adaptive should still cover) ===", flush=True)
u2, w2 = trial(30.0, 1.0, "window30/LA1.0")
print("\n--- VERDICT ---")
print(f"window30/LA2.0: steady underruns={u1} (worst regen {w1}ms)")
print(f"window30/LA1.0: steady underruns={u2} (worst regen {w2}ms)")
print("FIX WORKS: adaptive buffer covers the regen" if u1 == 0 and u2 == 0
      else "STILL UNDERRUNNING — adaptive buffer insufficient")
