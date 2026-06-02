"""Real-time drain test: consume PCM at wall-clock 1x for the whole track with
mid-stream prompt swaps; report underruns + buffer level. Proves the producer
sustains real time (the Phase 2 plugin's audio-callback contract)."""
import argparse, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import loader
from engine.realtime import RealtimeCover, SR, FPS
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test_output")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "source.wav"))
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--style", default="8-bit chiptune, square wave synth lead")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--lookahead", type=float, default=1.5)
    ap.add_argument("--block", type=int, default=2048)   # audio callback size (frames)
    args = ap.parse_args()

    rc = RealtimeCover(device="mps", steps=args.steps, window_s=20.0, lookahead_s=args.lookahead)
    rc.load_track(args.src, seconds=args.seconds)
    rc.set_style(args.style, timbre="none")
    full_s = rc.full_T / FPS
    print(f"[rt] track={full_s:.1f}s steps={args.steps} lookahead={args.lookahead}s block={args.block}f")

    rc.start()
    # prime the buffer (like a player buffering before play)
    t0 = time.time()
    while rc.buffered_s() < args.lookahead and time.time() - t0 < 60:
        time.sleep(0.02)
    print(f"[rt] primed {rc.buffered_s():.2f}s in {time.time()-t0:.1f}s; starting 1x playback")

    swaps = [(10.0, "lo-fi hip hop, jazzy piano, vinyl, boom-bap drums"),
             (20.0, "aggressive heavy metal, distorted guitars, double kick")]
    drained = []
    period = args.block / SR
    played = 0.0
    next_t = time.perf_counter()
    min_buf = 1e9
    n_blocks = int(full_s / period)
    for i in range(n_blocks):
        for st, tags in list(swaps):
            if played >= st:
                rc.set_prompt(tags); swaps.remove((st, tags))
                print(f"[rt] t={played:5.1f}s -> prompt swap (buffer={rc.buffered_s():.2f}s)")
        drained.append(rc.read(args.block))
        played += period
        min_buf = min(min_buf, rc.buffered_s())
        next_t += period
        dt = next_t - time.perf_counter()
        if dt > 0:
            time.sleep(dt)
    rc.stop()

    pcm = np.concatenate(drained, axis=0)
    import soundfile as sf
    m = np.abs(pcm).max(); pcm = pcm * (0.97 / m) if m > 1e-6 else pcm
    p = os.path.join(OUT, "audition", "realtime_stream.wav")
    sf.write(p, pcm, SR)
    print(f"\n[rt] RESULT underruns={rc.underruns}  min_buffer={min_buf:.2f}s  "
          f"regens={rc.regens}  worst_regen={rc.max_step_ms:.0f}ms")
    print(f"[rt] {'REAL-TIME OK (0 underruns)' if rc.underruns == 0 else 'UNDERRUNS - raise lookahead or lower steps'}")
    print(f"[rt] wrote {p} ({len(pcm)/SR:.1f}s)")
    rc.close()


if __name__ == "__main__":
    main()
