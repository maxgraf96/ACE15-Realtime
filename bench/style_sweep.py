"""CLAP-guided style/structure frontier: timbre=none, sweep denoise.
Finds the config that maximizes style adherence while keeping coherence."""
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import loader, mps_compat, style_metric as sm
from engine.jit import JITCover, SR
from engine.metrics import chroma_corr, onset_corr
from acestep.nodes.types import Audio

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "source.wav")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test_output", "audition")
SEC = 20.0
GENRES = {
    "lofi":  ("lo-fi hip hop, jazzy electric piano, vinyl crackle, boom-bap drums",
              ["heavy metal", "8-bit chiptune", "acoustic folk", "orchestral"]),
    "metal": ("aggressive heavy metal, distorted electric guitars, double kick",
              ["lo-fi hip hop", "8-bit chiptune", "acoustic folk", "orchestral"]),
}

jc = JITCover(device="mps", steps=8)
jc.load_track(SRC, seconds=SEC)
src = loader.load_audio(SRC, duration=SEC).waveform.float().cpu()
print(f"=== style/structure frontier (timbre=none, win20) — bpm/key auto ===")
print(f"{'config':26s} {'CLAPsim':>8s} {'margin':>7s} {'chroma':>7s} {'onset':>7s}")
best = {}
for name, (target, distract) in GENRES.items():
    for dn in [0.7, 0.8, 0.9]:
        for timbre in (["none"] if dn != 0.8 else ["none", "source"]):  # include src@0.8 as ref
            jc.set_style(target, denoise=dn, timbre=timbre)
            wav, st = jc.render(window_s=20, lookahead_s=2.0, seed=1234)
            m = wav.abs().max(); wav = wav*(0.97/m) if m > 1e-6 else wav
            p = os.path.join(OUT, f"style_{name}_dn{int(dn*100)}_{timbre}.wav")
            loader.save_audio(Audio(waveform=wav, sample_rate=SR), p)
            sc = sm.score(p, target, distract)
            cc, oc = chroma_corr(src, wav.cpu()), onset_corr(src, wav.cpu())
            tag = f"{name} dn{dn} {timbre}"
            print(f"{tag:26s} {sc['sim']:8.3f} {sc['margin']:7.3f} {cc:7.3f} {oc:7.3f}")
            mps_compat.reclaim()
jc.close()
print("Audition test_output/audition/style_*.wav")
