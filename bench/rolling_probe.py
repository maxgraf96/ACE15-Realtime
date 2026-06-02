"""Probe: pinned-prefix rolling cover (time-axis inpainting) vs independent windows.

Each window's prefix (the overlap with the previous window) is PINNED to the
previous window's clean latent via LatentNoiseMask (mask=0 preserve, mask=1
generate). The new region is denoised from the source, conditioned (via attention)
on the pinned past -> by construction the overlap is identical, so there's no seam.

Answers:
  1. Does the pin hold?           (pin region == previous clean latent)
  2. Does the new content continue smoothly from the pin? (no latent jump at the seam)
  3. Is the new content good quality vs an INDEPENDENT window? (onset_corr to source)
  4. Does it drift over a chain?  (latent RMS per window)
Writes rolling-continuous vs windowed(current) WAVs for an A/B listen.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from engine import mps_compat, metrics, loader
mps_compat.force_bf16_on_mps()
from engine.jit import JITCover, FPS, SPF, SR
from acestep.nodes.types import Latent
from acestep.engine.masking import LatentNoiseMask
from acestep.engine.session import PreparedSource

SONG = "/Users/max/Desktop/Testsongs/psig.mp3"
STYLE = "Dubstep"; DENOISE = 0.70; SEED = 1234
W, C = 500, 100              # 20 s window, 4 s pinned context
H = W - C                    # 16 s hop (new frames per window)
NWIN = 4                     # ~ W + 3*H = 68 s
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test_output", "diag")
os.makedirs(OUT, exist_ok=True)

jit = JITCover(device="mps", steps=8, config_path="acestep-v15-xl-turbo")
jit.load_track(SONG, seconds=72)
jit.set_style(STYLE, denoise=DENOISE, character=0.0)   # builds the handle
src_lat_full = jit.source.latent.tensor
src_audio = loader.load_audio(SONG, duration=72).waveform.float().mean(0).numpy()


def gen_window(w0, pin=None, seed=SEED):
    """Generate window source[w0:w0+W]; if pin [1,C,D] given, pin the prefix to it."""
    src = src_lat_full[:, w0:w0 + W, :].clone()
    ctx = jit.source.context_latent.tensor[:, w0:w0 + W, :]
    sl = Latent(tensor=src)
    if pin is not None:
        c = pin.shape[1]
        orig = src.clone(); orig[:, :c, :] = pin.to(src.dtype)
        m = torch.ones(1, W, 1, device=src.device, dtype=src.dtype); m[:, :c, :] = 0.0
        sl.mask = LatentNoiseMask(mask=m, original_latents=orig)
    jit.handle.context_latent = Latent(tensor=ctx)
    jit.handle.source = PreparedSource(latent=sl, context_latent=Latent(tensor=ctx))
    lat = jit.handle.tick(drain=True, denoise=DENOISE, seed=seed)
    mps_compat.mps_sync()
    return lat.tensor.float()                              # [1, W, D]


def decode(lat_bTD):
    """Full tiled decode of a [1,T,D] latent -> [2, T*SPF] float."""
    return jit._ensure_tiles(Latent(tensor=lat_bTD.to(jit.source.latent.tensor.dtype)),
                             {}, 0, lat_bTD.shape[1]).float()


def onset_to_src(audio_2xN, src_off_frames):
    m = audio_2xN.mean(0).numpy(); n = audio_2xN.shape[-1]
    s = src_audio[src_off_frames * SPF: src_off_frames * SPF + n]
    k = min(len(m), len(s))
    return float(metrics.onset_corr(torch.tensor(s[:k]), torch.tensor(m[:k])))


print(f"[probe] W={W}({W/FPS:.0f}s) C={C}({C/FPS:.0f}s) hop={H}({H/FPS:.0f}s) windows={NWIN}", flush=True)

# ---- chain with pinning, collect committed continuous latent + per-window stats ----
committed = gen_window(0)                       # window 0, no pin
rms = [float(committed.pow(2).mean().sqrt())]
seam_ratios = []; pin_errs = []
for k in range(1, NWIN):
    pin = committed[:, -C:, :]                  # last C committed frames (clean past)
    wk = gen_window(k * H, pin=pin)
    pin_errs.append(float((wk[:, :C] - pin).abs().mean() / (pin.abs().mean() + 1e-9)))
    fd = (wk[:, 1:] - wk[:, :-1]).abs().mean(dim=(0, 2))   # per-frame latent delta
    seam_ratios.append(float(fd[C - 1] / fd.mean()))       # jump at pin->new vs typical
    rms.append(float(wk[:, C:].pow(2).mean().sqrt()))
    committed = torch.cat([committed, wk[:, C:, :]], dim=1)

print(f"[1] pin honored:   mean |pinned - target| / |target| = {np.mean(pin_errs):.4f}  (want ~0)", flush=True)
print(f"[2] continuity:    latent jump at pin->new / typical  = {np.mean(seam_ratios):.2f}x  (want ~1, >>1 = seam)", flush=True)
print(f"[4] drift (RMS/win): {[round(r,3) for r in rms]}  (want flat, no blow-up/decay)", flush=True)

# ---- quality: window-1 pinned-new vs INDEPENDENT window-1 (onset to source) ----
w1_pin = committed[:, W:W + H, :]               # window-1 new region, source[H+? ..] -> committed[W:W+H]
w1_indep = gen_window(H)[:, C:, :]              # same source region, independent gen
src_off = W                                      # committed[W:] starts at source frame W
oc_pin = onset_to_src(decode(w1_pin), src_off)
oc_indep = onset_to_src(decode(w1_indep), H + C)
print(f"[3] quality (onset_corr to source): pinned-new={oc_pin:.3f}  independent={oc_indep:.3f}  (pinned >= indep = good)", flush=True)

# ---- A/B WAVs: rolling continuous vs current windowed (render crossfade) ----
import soundfile as sf
def softclip(x, t=0.92):
    a = np.abs(x); return np.where(a <= t, x, np.sign(x) * (t + (1 - t) * np.tanh((a - t) / (1 - t)))).astype(np.float32)
roll = softclip(decode(committed).numpy().T)
sf.write(f"{OUT}/roll_continuous.wav", roll, SR)
print(f"[wav] wrote roll_continuous.wav  ({roll.shape[0]/SR:.1f}s, peak {np.abs(roll).max():.3f})", flush=True)
win, _ = jit.render(window_s=20.0, slice_s=1.0, lookahead_s=2.0, xfade_s=0.12, seed=SEED)
wv = softclip(win.float().numpy().T)[:roll.shape[0]]
sf.write(f"{OUT}/roll_windowed.wav", wv, SR)
print(f"[wav] wrote roll_windowed.wav   ({wv.shape[0]/SR:.1f}s)", flush=True)
print("VERDICT: pin OK + continuity~1x + pinned quality>=indep + flat RMS => rolling viable", flush=True)
jit.close()
