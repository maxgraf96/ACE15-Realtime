"""Small-lookahead JIT streaming cover CLI (Phase 1 v1).

  .venv/bin/python cli/jit_cover.py --seconds 30 --style "8-bit chiptune" \
      --lookahead 1.0 --window 10 \
      --prompt-at 10:"lo-fi hip hop, jazzy piano" --prompt-at 20:"heavy metal" \
      --denoise-at 25:0.6

Renders the cover with a live control schedule, writes the WAV, reports RTF +
control latency, and self-validates structure (chroma/onset) when no controls.
"""
from __future__ import annotations
import argparse, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import loader  # noqa: E402
from engine.jit import JITCover, SR  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test_output")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "source.wav"))
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--style", default="8-bit chiptune, square wave synth lead")
    ap.add_argument("--window", type=float, default=10.0)
    ap.add_argument("--lookahead", type=float, default=1.0)
    ap.add_argument("--slice", type=float, default=1.0)
    ap.add_argument("--denoise", type=float, default=0.8)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--prompt-at", action="append", default=[], help="T:tags")
    ap.add_argument("--denoise-at", action="append", default=[], help="T:value")
    ap.add_argument("--out", default=os.path.join(OUT, "jit_cover.wav"))
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    controls = []
    for s in args.prompt_at:
        t, v = s.split(":", 1); controls.append((float(t), "prompt", v))
    for s in args.denoise_at:
        t, v = s.split(":", 1); controls.append((float(t), "denoise", float(v)))

    t0 = time.time()
    jc = JITCover(device="mps", steps=args.steps)
    jc.load_track(args.src, seconds=args.seconds)
    jc.set_style(args.style, denoise=args.denoise)
    print(f"[jit] loaded+styled {time.time()-t0:.1f}s  track={jc.track_dur:.1f}s window={args.window}s "
          f"lookahead={args.lookahead}s denoise={args.denoise}")
    if controls:
        print(f"[jit] controls: {controls}")

    wav, st = jc.render(controls=controls, window_s=args.window, lookahead_s=args.lookahead,
                        slice_s=args.slice, seed=1234)
    m = wav.abs().max()
    if m > 1e-6:
        wav = wav * (0.97 / m)
    from acestep.nodes.types import Audio
    loader.save_audio(Audio(waveform=wav, sample_rate=SR), args.out)
    print("\n" + st.summary())

    if not controls:
        from engine.metrics import chroma_corr, onset_corr
        src = loader.load_audio(args.src, duration=args.seconds).waveform.float().cpu()
        print(f"  structure vs source: chroma={chroma_corr(src, wav.cpu()):.3f} onset={onset_corr(src, wav.cpu()):.3f}")
    jc.close()


if __name__ == "__main__":
    main()
