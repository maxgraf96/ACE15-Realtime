"""Deep probe: where does the residual 'metallic' come from, and what fixes it?

Chain distortion suspects after the earlier fixes (hard-clip, underruns, DCW):
  (8) the C++ instantaneous tanh SOFT-CLIP @0.92 — a per-sample waveshaper. When the
      cover runs HOT (high Amount), it engages on a large fraction of samples and the
      harmonics it generates above 24 kHz ALIAS back down = inharmonic 'metallic'.
  (9) the C++ LINEAR resampler (only when the output device SR != 48k) — poor stopband
      => imaging/aliasing.
  + transient underruns / decode seams.

This:
  1. Generates a REAL cover at the user's hot setting (Amount 0.95) and a tamer 0.70,
     captures the RAW pre-limiter output, counts 1x underruns + decode-seam jumps.
  2. Quantifies clipper engagement (peak, %>0.92) and the HF/alias energy the current
     tanh soft-clip ADDS, vs two candidate fixes: an OVERSAMPLED soft-clip (4x) and a
     LOOKAHEAD LIMITER (gain-reduction, no waveshaping). Writes WAVs to A/B by ear.
  3. Model-free resampler test: 19 kHz tone 48k->44.1k, linear vs polyphase-sinc,
     measures aliased (out-of-band) energy.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
import torchaudio.functional as AF
from engine import mps_compat
mps_compat.force_bf16_on_mps()
from engine.realtime import RealtimeCover, SR
from engine.jit import SPF

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "assets", "source.wav")
OUT = os.path.join(ROOT, "test_output", "diag"); os.makedirs(OUT, exist_ok=True)
STYLE = "warm lo-fi hip hop, dusty drums"
MODEL = "acestep-v15-xl-turbo"


def softclip(x, t=0.92):   # exact copy of the C++ instantaneous clipper
    a = np.abs(x)
    return np.where(a <= t, x, np.sign(x) * (t + (1 - t) * np.tanh((a - t) / (1 - t)))).astype(np.float32)


def os_softclip(x_n2, os=4, t=0.92):   # candidate A: oversample -> clip -> downsample (kills alias)
    xt = torch.tensor(x_n2.T)
    up = AF.resample(xt, SR, SR * os)
    a = up.abs(); cl = torch.where(a <= t, up, up.sign() * (t + (1 - t) * torch.tanh((a - t) / (1 - t))))
    return AF.resample(cl, SR * os, SR).T.numpy().astype(np.float32)


def limiter(x_n2, ceil=0.97, la_ms=2.0, rel_ms=80.0):   # candidate B: lookahead peak limiter
    pk = np.abs(x_n2).max(1)
    g = np.minimum(1.0, ceil / np.maximum(pk, 1e-9))
    la = max(1, int(SR * la_ms / 1000))
    try:
        from scipy.ndimage import minimum_filter1d
        g = minimum_filter1d(g, size=la, origin=-(la // 2))   # gain dips BEFORE the peak
    except Exception:
        g = np.minimum(g, np.concatenate([g[la:], np.ones(la)]))  # crude forward-min fallback
    rel = np.exp(-1.0 / (SR * rel_ms / 1000)); cur = 1.0; out = np.empty_like(g)
    for i in range(len(g)):                       # instant attack (lookahead), slow release
        cur = g[i] if g[i] < cur else cur * rel + g[i] * (1 - rel)
        out[i] = cur
    return (x_n2 * out[:, None]).astype(np.float32)


def hf_ratio(x_n2, fmin=12000):   # fraction of energy above fmin (saturation/alias adds HF)
    m = x_n2.mean(1).astype(np.float64); w = np.hanning(len(m))
    X = np.abs(np.fft.rfft(m * w)) ** 2; f = np.fft.rfftfreq(len(m), 1 / SR)
    return float(X[f >= fmin].sum() / (X.sum() + 1e-12))


def stats(x_n2):
    pk = float(np.abs(x_n2).max()); rms = float(np.sqrt(np.mean(x_n2 ** 2)))
    return pk, rms, pk / (rms + 1e-9)


def capture(denoise, secs=18, window_s=20.0):
    rc = RealtimeCover(device="mps", steps=8, window_s=window_s, lookahead_s=2.0, config_path=MODEL)
    rc.load_track(SRC, seconds=60)
    rc.set_style(STYLE, denoise=denoise)
    rc.start()
    t0 = time.perf_counter()
    while rc.buffered_s() < 4.0 and time.perf_counter() - t0 < 40:
        time.sleep(0.05)
    rc.underruns = 0
    block = 2048; period = block / SR; nxt = time.perf_counter(); end = nxt + secs; buf = []
    while time.perf_counter() < end:
        buf.append(rc.read(block)[:, :2].copy())   # cover pair = RAW pre-limiter output
        nxt += period; dt = nxt - time.perf_counter()
        if dt > 0: time.sleep(dt)
        else: nxt = time.perf_counter()
    u = rc.underruns; rc.close(); mps_compat.reclaim()
    return np.concatenate(buf, 0), u


print(f"[probe] src={os.path.basename(SRC)} model={MODEL}", flush=True)
for dn in (0.95, 0.70):
    cov, under = capture(dn)
    pk, rms, crest = stats(cov)
    over = float(np.mean(np.abs(cov) > 0.92))
    # decode-seam discontinuity: sample jump at tile boundaries (every 48 latent frames)
    d = np.abs(np.diff(cov[:, 0])); tile = 48 * SPF
    seam = float(np.mean([d[k] for k in range(tile, len(d), tile)]) / (d.mean() + 1e-9)) if len(d) > tile else 0
    print(f"\n=== Amount {dn:.2f} ===  underruns(1x)={under}  peak={pk:.3f} rms={rms:.3f} "
          f"crest={crest:.1f}  %>0.92={100*over:.1f}%  seam/typ={seam:.1f}x", flush=True)
    variants = {"raw": cov, "softclip@0.92(current)": softclip(cov),
                "os-softclip-4x": os_softclip(cov), "limiter": limiter(cov)}
    base_hf = hf_ratio(cov)
    for name, sig in variants.items():
        p, r, c = stats(sig); hf = hf_ratio(sig)
        print(f"   {name:24s} peak={p:.3f} crest={c:4.1f} HF>12k={100*hf:.2f}%  "
              f"(+{100*(hf-base_hf):+.2f}% vs raw)", flush=True)
        import soundfile as sf
        tag = name.split("@")[0].split("(")[0].replace(" ", "")
        sf.write(f"{OUT}/metallic_a{int(dn*100)}_{tag}.wav", sig, SR)

# ---- model-free resampler aliasing: 19 kHz tone, 48k -> 44.1k ----
print("\n=== resampler aliasing (19 kHz tone, 48000 -> 44100) ===", flush=True)
t = np.arange(SR)[:, None] / SR; tone = np.sin(2 * np.pi * 19000 * t).astype(np.float32).repeat(2, 1)
ratio = SR / 44100.0; n_out = int(len(tone) / ratio); pos = np.arange(n_out) * ratio
i0 = np.floor(pos).astype(int); i1 = np.minimum(i0 + 1, len(tone) - 1); fr = (pos - i0)[:, None]
lin = (tone[i0] * (1 - fr) + tone[i1] * fr).astype(np.float32)            # current C++ linear interp
sinc = AF.resample(torch.tensor(tone.T), SR, 44100).T.numpy()             # polyphase sinc (good)
for name, sig in [("linear(current)", lin), ("polyphase-sinc", sinc)]:
    m = sig[:, 0].astype(np.float64); w = np.hanning(len(m))
    X = np.abs(np.fft.rfft(m * w)) ** 2; f = np.fft.rfftfreq(len(m), 1 / 44100.0)
    band = (f > 18500) & (f < 19500); spur = X[~band].sum() / (X.sum() + 1e-12)
    print(f"   {name:18s} aliased(out-of-band) energy = {100*spur:.3f}%", flush=True)
print(f"\n[wav] wrote A/B variants to {OUT}/metallic_*.wav", flush=True)
