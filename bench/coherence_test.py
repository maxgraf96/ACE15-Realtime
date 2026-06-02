"""Coherence experiment: does correct bpm/key + larger window fix intra-window
incoherence? Renders steady covers (no swaps) for audition. Load once."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import loader, mps_compat
from engine.jit import JITCover, SR
from engine.metrics import chroma_corr, onset_corr
from acestep.nodes.types import Audio

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "source.wav")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test_output", "audition")
STYLE = "8-bit chiptune, retro video game, square-wave synth lead"
SECONDS = 24.0

os.makedirs(OUT, exist_ok=True)
jc = JITCover(device="mps", steps=8)
jc.load_track(SRC, seconds=SECONDS)           # auto-detects bpm/key
src = loader.load_audio(SRC, duration=SECONDS).waveform.float().cpu()

def run(tag, denoise, win, bpm=None, key=None):
    jc.set_style(STYLE, denoise=denoise, bpm=bpm, key=key)
    wav, st = jc.render(window_s=win, lookahead_s=1.0, slice_s=1.0, seed=1234)
    m = wav.abs().max(); wav = wav*(0.97/m) if m > 1e-6 else wav
    loader.save_audio(Audio(waveform=wav, sample_rate=SR), os.path.join(OUT, f"coh_{tag}.wav"))
    cc, oc = chroma_corr(src, wav.cpu()), onset_corr(src, wav.cpu())
    print(f"  {tag:22s} dn={denoise} win={win:.0f}s bpm/key={'WRONG(120/C)' if bpm else 'auto'}  "
          f"RTF={st.rtf:.3f} chroma={cc:.3f} onset={oc:.3f}")
    mps_compat.reclaim()

print("=== coherence experiment (chiptune, 24s) ===")
run("A_wrongmeta_w10", 0.8, 10, bpm=120, key="C major")   # the current baseline
run("B_autometa_w10",  0.8, 10)                            # + correct bpm/key
run("C_autometa_w20",  0.8, 20)                            # + larger window
run("D_dn70_w20",      0.7, 20)                            # + slightly lower denoise
run("E_dn60_w20",      0.6, 20)                            # structure-leaning
jc.close()
print("Done. Audition test_output/audition/coh_*.wav (A=baseline -> E).")
