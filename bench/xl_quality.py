"""XL-turbo vs 2B-turbo cover quality A/B (run once per --model; separate
processes so XL=bf16 and 2B=fp32 don't share the global dtype patch / RAM)."""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings; warnings.filterwarnings("ignore")
from engine import mps_compat
ap = argparse.ArgumentParser(); ap.add_argument("--model", choices=["xl", "2b"], required=True)
args = ap.parse_args()
if args.model == "xl":
    mps_compat.force_bf16_on_mps()
from engine import loader, style_metric as sm
from engine.jit import JITCover, SR
from engine.metrics import chroma_corr, onset_corr
from acestep.nodes.types import Audio

CFG = "acestep-v15-xl-turbo" if args.model == "xl" else "acestep-v15-turbo"
CASES = [
    ("loop", "assets/source.wav", "lo-fi hip hop, jazzy electric piano, vinyl crackle, boom-bap drums", 0.6),
    ("away", "/Users/max/Desktop/Testsongs/away.mp3", "instrumental 70s funk, slap bass, syncopated groove, wah guitar, punchy drums", 0.55),
]
jc = JITCover(device="mps", steps=8, config_path=CFG)
print(f"=== {args.model.upper()} ({CFG}) dtype={jc.session.handler.dtype} ===")
for name, path, style, dn in CASES:
    jc.load_track(path, seconds=20)
    src = loader.load_audio(path, duration=20).waveform.float().cpu()
    jc.set_style(style, denoise=dn, character=0.0)
    w, st = jc.render(window_s=20, lookahead_s=2.0, seed=1234)
    m = w.abs().max(); wn = w * (0.97 / m) if m > 1e-6 else w
    p = f"test_output/audition/{name}_{args.model}.wav"
    loader.save_audio(Audio(waveform=wn, sample_rate=SR), p)
    clap = sm.score(p, style.split(",")[0], ["acoustic piano", "ambient", "rock"])["sim"]
    print(f"  {name:5s} dn={dn} RTF={st.rtf:.3f}  chroma={chroma_corr(src,w):.3f} onset={onset_corr(src,w):.3f} styleCLAP={clap:.3f} -> {os.path.basename(p)}")
    mps_compat.reclaim()
jc.close()
