"""Offline cover CLI + structure validation (Phase 0 deliverable 1).

Runs the DEMON cover node graph on MPS and validates that the output is a
*structure-preserving* remix using timbre-robust metrics:

  - chroma correlation  : harmonic/melodic structure (pitch classes over time)
  - onset correlation   : rhythmic structure (spectral-flux envelope)

A raw amplitude-envelope correlation is misleading here: a metal cover of a
mellow loop has a completely different envelope even with identical chords and
beats. Chroma + onset stay meaningful across a timbre swap.

To prove the structure comes from ``context_latents`` (not coincidence), it also
generates a control with the context replaced by silence and compares: a real
cover should score much higher chroma/onset corr WITH context than without.

Usage:
  .venv/bin/python cli/offline_cover.py --src assets/source.wav --seconds 12 \
      --style "aggressive heavy metal, distorted guitars" --denoise 0.6 1.0
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import loader  # noqa: E402  (installs mps_compat)
from engine.mps_compat import mps_sync  # noqa: E402

import torch  # noqa: E402

SR = 48000
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test_output")


# ---- timbre-robust structure metrics ---------------------------------------
def _stft_mag(wav, n_fft=2048, hop=512):
    x = wav.mean(0).float()
    win = torch.hann_window(n_fft)
    return torch.stft(x, n_fft=n_fft, hop_length=hop, window=win, return_complex=True).abs()


def chromagram(wav, n_fft=2048, hop=512):
    spec = _stft_mag(wav, n_fft, hop)                      # [F, T]
    freqs = torch.linspace(0, SR / 2, spec.shape[0])
    pc = torch.full((spec.shape[0],), -1, dtype=torch.long)
    valid = freqs > 20
    midi = (69 + 12 * torch.log2(freqs[valid] / 440.0)).round().long()
    pc[valid] = midi % 12
    C = torch.zeros(12, spec.shape[1])
    for k in range(12):
        m = pc == k
        if m.any():
            C[k] = spec[m].sum(0)
    return C / (C.norm(dim=0, keepdim=True) + 1e-9)


def onset_env(wav, n_fft=2048, hop=512):
    spec = _stft_mag(wav, n_fft, hop)
    return (spec[:, 1:] - spec[:, :-1]).clamp(min=0).sum(0)   # spectral flux [T-1]


def _corr(a, b):
    n = min(a.shape[-1], b.shape[-1])
    a, b = a[..., :n].flatten(), b[..., :n].flatten()
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-9))


def chroma_corr(a, b):
    return _corr(chromagram(a), chromagram(b))


def onset_corr(a, b):
    return _corr(onset_env(a), onset_env(b))


def vae_decode(handler, latent_btd):
    lat_bdt = latent_btd.transpose(1, 2)
    try:
        with handler._load_model_context("vae"):
            return handler.tiled_decode(lat_bdt)
    except Exception:
        vae = handler.vae
        T = lat_bdt.shape[-1]
        chunk, overlap, stride = 64, 8, 48
        out, pos = [], 0
        with handler._load_model_context("vae"):
            while pos < T:
                lo, hi = max(0, pos - overlap), min(T, pos + stride + overlap)
                wav = vae.decode(lat_bdt[..., lo:hi].to(vae.dtype)).sample.float()
                tl = (pos - lo) * 1920
                tr = (hi - min(T, pos + stride)) * 1920
                out.append(wav[..., tl: wav.shape[-1] - tr if tr else None])
                pos += stride
        return torch.cat(out, dim=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "source.wav"))
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--style", default="aggressive heavy metal, distorted electric guitars, double kick drums")
    ap.add_argument("--denoise", type=float, nargs="+", default=[0.6, 1.0])
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--timbre", choices=["source", "silence"], default="source")
    ap.add_argument("--steps", type=int, default=8)
    args = ap.parse_args()

    from acestep.nodes.vae_nodes import VAEEncodeAudio
    from acestep.nodes.semantic_nodes import SemanticExtract
    from acestep.nodes.cond_nodes import EncodeText, EncodeConditioning
    from acestep.nodes.diffusion_nodes import StreamDenoise, OdeSolver
    from acestep.nodes.types import Audio, Latent

    os.makedirs(OUT_DIR, exist_ok=True)
    L = loader.load_model(device="mps")

    src_audio = loader.load_audio(args.src, duration=args.seconds)
    dur = src_audio.waveform.shape[-1] / SR
    src_wav = src_audio.waveform.float().cpu()
    loader.save_audio(src_audio, os.path.join(OUT_DIR, "cover_source.wav"))

    source_latent = VAEEncodeAudio().execute(vae=L.vae, audio=src_audio)["latent"]
    context_latent = SemanticExtract().execute(model=L.model, latent=source_latent)["latent"]
    # zero/silence context control of identical shape
    silence_ctx = Latent(tensor=torch.zeros_like(context_latent.tensor))

    te = EncodeText().execute(
        clip=L.clip, tags=args.style, lyrics="", task_type="cover",
        bpm=120, duration=dur, key="C major", time_signature="4", language="en",
    )["text_embed"]
    timbre_ref = source_latent if args.timbre == "source" else None
    conditioning = EncodeConditioning().execute(model=L.model, text_embed=te, timbre_ref=timbre_ref)["conditioning"]
    solver = OdeSolver().execute()["solver"]

    def generate(ctx, denoise):
        return StreamDenoise().execute(
            model=L.model, solver=solver, positive=conditioning,
            context_latent=ctx, source_latent=source_latent,
            steps=args.steps, shift=3.0, denoise=denoise, seed=args.seed,
            pipeline_depth=1, drain=True, dcw_enabled=False, duration=dur,
            noise_on_cpu=True,
        )["latent"]

    print("\n" + "=" * 76)
    print(f"COVER structure validation — style='{args.style}'  timbre={args.timbre}")
    print("=" * 76)
    print(f"{'variant':28s} {'chroma':>8s} {'onset':>8s}   (higher = more source structure preserved)")

    results = []
    for dn in args.denoise:
        # real cover (with semantic context)
        t = time.time()
        lat = generate(context_latent, dn)
        wav = vae_decode(L.handler, lat.tensor); mps_sync()
        p = os.path.join(OUT_DIR, f"cover_denoise_{int(dn*100):03d}.wav")
        loader.save_audio(Audio(waveform=wav.squeeze(0), sample_rate=SR), p)
        cc, oc = chroma_corr(src_wav, wav.squeeze(0).cpu()), onset_corr(src_wav, wav.squeeze(0).cpu())
        results.append((f"cover denoise={dn:.2f}", cc, oc))
        print(f"{'cover denoise='+format(dn,'.2f'):28s} {cc:8.3f} {oc:8.3f}   {time.time()-t:.1f}s -> {os.path.basename(p)}")

    # control: NO structure context (silence), denoise=1.0, same prompt/seed
    lat = generate(silence_ctx, 1.0)
    wav = vae_decode(L.handler, lat.tensor); mps_sync()
    loader.save_audio(Audio(waveform=wav.squeeze(0), sample_rate=SR), os.path.join(OUT_DIR, "cover_NOCTX_control.wav"))
    cc, oc = chroma_corr(src_wav, wav.squeeze(0).cpu()), onset_corr(src_wav, wav.squeeze(0).cpu())
    results.append(("control (no context)", cc, oc))
    print(f"{'control (no context)':28s} {cc:8.3f} {oc:8.3f}   <- structure should be MUCH lower here")

    # verdict
    best_cover = max(r[1] for r in results[:-1])
    ctrl = results[-1][1]
    print("\nVERDICT:", "context_latents PRESERVES structure"
          if best_cover > ctrl + 0.05 else "structure preservation WEAK/unclear — investigate")
    print(f"  best cover chroma-corr {best_cover:.3f}  vs  no-context control {ctrl:.3f}")


if __name__ == "__main__":
    main()
