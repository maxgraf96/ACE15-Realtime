"""Reproduce the 'metallic after Fast->Quality round-trip' bug headlessly.

Simulates the app's model-switch path: build Fast (2B fp32), then close it and
build Quality (XL bf16) in the SAME process, then measure the audible peak +
NaN-ness of the FIRST vs SECOND XL window generation. If the first XL gen after
the switch is pathological (huge peak / NaN) and the second is normal, the fix
is a post-load warmup inference.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from engine import mps_compat
mps_compat.force_bf16_on_mps()
from engine.realtime import RealtimeCover, FPS

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "source.wav")
TAGS = "future bass"


def dit_dtype(rc):
    for path in ("handler.model", "model.model", "model"):
        try:
            obj = rc.jit.session
            for a in path.split("."):
                obj = getattr(obj, a)
            return str(next(obj.parameters()).dtype)
        except Exception:
            continue
    return "?"


def gen_peak(rc, w0, w1, seed=1234):
    """Generate window [w0,w1) and decode the first ~2s slice; return (peak, nan?)."""
    lat = rc.jit._gen(w0, w1, seed)
    cache = {}
    nf = lat.tensor.shape[1]
    seg = rc.jit._ensure_tiles(lat, cache, 0, min(50, nf))   # ~2s
    mps_compat.mps_sync()
    t = seg.float()
    return float(t.abs().max()), bool(torch.isnan(t).any())


def run(model, label):
    cfg = {"fast": "acestep-v15-turbo", "quality": "acestep-v15-xl-turbo"}[model]
    rc = RealtimeCover(device="mps", steps=8, window_s=30.0,
                       lookahead_s=2.0 if model == "quality" else 1.0, config_path=cfg)
    rc.load_track(SRC, seconds=30)
    rc.set_style(TAGS, denoise=0.90)
    print(f"[{label}] dit dtype = {dit_dtype(rc)}", flush=True)
    W = min(int(30 * FPS), rc.full_T)
    p1, n1 = gen_peak(rc, 0, W)
    p2, n2 = gen_peak(rc, 0, W)          # same window again (2nd inference)
    p3, n3 = gen_peak(rc, 0, W)          # 3rd
    print(f"[{label}] gen peaks: 1st={p1:.3f}(nan={n1})  2nd={p2:.3f}(nan={n2})  3rd={p3:.3f}(nan={n3})", flush=True)
    return rc, (p1, p2, p3)


print("=== building Fast (2B) ===", flush=True)
rc_fast, _ = run("fast", "fast")
print("=== closing Fast, reclaiming ===", flush=True)
rc_fast.close(); mps_compat.reclaim()

print("=== building Quality (XL) AFTER the 2B session (the bug path) ===", flush=True)
rc_q, (p1, p2, p3) = run("quality", "quality-after-switch")
rc_q.close()

print("\n--- VERDICT ---")
print(f"XL 1st-gen peak {p1:.3f} vs steady {p2:.3f}/{p3:.3f}")
if p1 > 1.8 * max(p2, p3) or p1 != p1:   # 1st >> steady, or NaN
    print("REPRO: first XL gen after switch is PATHOLOGICAL -> warmup fixes it")
else:
    print("NO REPRO via peak: first XL gen looks normal; metallic cause is elsewhere")
