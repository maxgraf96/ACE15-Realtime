"""Headless streaming cover CLI (Phase 1).

  .venv/bin/python cli/stream_cover.py --src assets/source.wav --seconds 30 \
      --style "8-bit chiptune, square wave synth" --window 10 --denoise 0.8

Streams a structure-preserving remix window-by-window with windowed decode at
the playhead, writes the output WAV, and reports the real-time factor + latencies.
Use --prompt-at "8.0:lo-fi hip hop" (repeatable) to test live prompt swaps.
"""
from __future__ import annotations
import argparse, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import loader, mps_compat  # noqa: E402
from engine.streaming import StreamingCover, SR  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test_output")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "source.wav"))
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--style", default="8-bit chiptune, retro video game, square-wave synth lead")
    ap.add_argument("--window", type=float, default=10.0)
    ap.add_argument("--denoise", type=float, default=0.8)  # user-preferred default
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--prompt-at", action="append", default=[], help="SECONDS:tags (repeatable)")
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "stream_cover.wav"))
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    sched = []
    for spec in args.prompt_at:
        at, tags = spec.split(":", 1)
        sched.append((float(at), tags))

    t0 = time.time()
    sc = StreamingCover(device="mps", steps=args.steps)
    sc.load_track(args.src, seconds=args.seconds)
    sc.set_style(args.style, denoise=args.denoise)
    print(f"[stream_cover] loaded+styled in {time.time()-t0:.1f}s  track={sc.track_dur:.1f}s "
          f"window={args.window}s denoise={args.denoise} steps={args.steps}")
    if sched:
        print(f"[stream_cover] live prompt swaps: {sched}")

    wav, st, plat = sc.render(win_s=args.window, seed=args.seed, prompt_schedule=sched)

    # peak-normalize for fair listening
    m = wav.abs().max()
    if m > 1e-6:
        wav = wav * (0.97 / m)

    from acestep.nodes.types import Audio
    loader.save_audio(Audio(waveform=wav, sample_rate=SR), args.out)
    print("\n" + st.summary())
    if plat:
        print(f"  live prompt swap latency: {[f'{x:.0f}ms' for x in plat]} (re-encode; applies next chunk)")

    # structure validation vs source (timbre-robust)
    if not sched:  # only meaningful when style is constant across the clip
        from engine.metrics import chroma_corr, onset_corr
        src = loader.load_audio(args.src, duration=args.seconds).waveform.float().cpu()
        cc, oc = chroma_corr(src, wav.cpu()), onset_corr(src, wav.cpu())
        print(f"  structure vs source: chroma={cc:.3f} onset={oc:.3f}  (higher = more preserved)")
    sc.close()


if __name__ == "__main__":
    main()
