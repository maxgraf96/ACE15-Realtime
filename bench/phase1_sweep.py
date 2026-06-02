"""Phase 1 operating-point sweep: RTF + structure across window/steps/denoise,
plus a live prompt-swap test. Fresh Session per config (avoids MPS state churn)."""
from __future__ import annotations
import os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import loader, mps_compat  # noqa: E402
from engine.streaming import StreamingCover, SR  # noqa: E402
from engine.metrics import chroma_corr, onset_corr  # noqa: E402

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "source.wav")
SECONDS = 30.0
STYLE = "8-bit chiptune, retro video game, square-wave synth lead"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test_output")
NOTES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "notes")

# (window_s, steps, denoise)
CONFIGS = [
    (10.0, 8, 0.8),
    (20.0, 8, 0.8),
    (30.0, 8, 0.8),   # single chunk (no seams)
    (10.0, 4, 0.8),
    (10.0, 8, 0.6),
]


def run_one(win, steps, denoise):
    sc = StreamingCover(device="mps", steps=steps)
    sc.load_track(SRC, seconds=SECONDS)
    sc.set_style(STYLE, denoise=denoise)
    wav, st, _ = sc.render(win_s=win, seed=1234)
    src = loader.load_audio(SRC, duration=SECONDS).waveform.float().cpu()
    cc, oc = chroma_corr(src, wav.cpu()), onset_corr(src, wav.cpu())
    sc.close()
    mps_compat.reclaim()
    return st, cc, oc


def main():
    rows = []
    print("=" * 78)
    for win, steps, dn in CONFIGS:
        t0 = time.time()
        st, cc, oc = run_one(win, steps, dn)
        rows.append((win, steps, dn, st.rtf, st.chunks,
                     (sum(st.gen_ms)/len(st.gen_ms)), (sum(st.dec_ms)/len(st.dec_ms)),
                     st.max_chunk_ms, cc, oc))
        print(f"win={win:4.0f}s steps={steps} dn={dn}  RTF={st.rtf:.3f}  "
              f"gen~{sum(st.gen_ms)/len(st.gen_ms):.0f}ms dec~{sum(st.dec_ms)/len(st.dec_ms):.0f}ms "
              f"maxchunk={st.max_chunk_ms:.0f}ms  chroma={cc:.3f} onset={oc:.3f}  ({time.time()-t0:.0f}s)")

    # live prompt swap test (control latency + audition)
    print("-" * 78)
    sc = StreamingCover(device="mps", steps=8)
    sc.load_track(SRC, seconds=SECONDS)
    sc.set_style(STYLE, denoise=0.8)
    wav, st, plat = sc.render(win_s=10.0, seed=1234,
                              prompt_schedule=[(10.0, "lo-fi hip hop, jazzy piano, boom-bap"),
                                               (20.0, "aggressive heavy metal, distorted guitars")])
    m = wav.abs().max(); wav = wav * (0.97/m) if m > 1e-6 else wav
    from acestep.nodes.types import Audio
    loader.save_audio(Audio(waveform=wav, sample_rate=SR), os.path.join(OUT, "stream_promptswap.wav"))
    print(f"prompt-swap render RTF={st.rtf:.3f}  re-encode latency={[f'{x:.0f}ms' for x in plat]}")
    sc.close(); mps_compat.reclaim()

    # write notes
    os.makedirs(NOTES, exist_ok=True)
    L = ["# Phase 1 — headless streaming cover, operating points (M4 Max, MPS)\n",
         f"- 30 s source, style='{STYLE}', depth-1 drain, RCFG/CFG off, DCW off, fp32.",
         "- RTF = compute wall / audio seconds (<1 = real-time). max-chunk = worst gen+decode for one chunk (= min producer lookahead). Overlapping windows (0.5 s) for exact timeline.\n",
         "| window | steps | denoise | RTF | chunks | gen ms | dec ms | max-chunk ms | chroma | onset |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for win, steps, dn, rtf, ch, gm, dm, mc, cc, oc in rows:
        L.append(f"| {win:.0f}s | {steps} | {dn} | {rtf:.3f} | {ch} | {gm:.0f} | {dm:.0f} | {mc:.0f} | {cc:.3f} | {oc:.3f} |")
    L.append(f"\n- Live prompt swap re-encode ~{plat[0]:.0f} ms; applies at next chunk boundary "
             f"(v0 control granularity = window size). 1-tick control needs the continuous depth pipeline (next).")
    L.append("- Worst per-chunk gen+decode sets the minimum producer lookahead buffer for gapless playback.\n")
    with open(os.path.join(NOTES, "phase1_results.md"), "w") as f:
        f.write("\n".join(L))
    print(f"[notes] wrote {os.path.join(NOTES, 'phase1_results.md')}")


if __name__ == "__main__":
    main()
