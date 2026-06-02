"""Render a clean, peak-normalized audition set for human listening.

Fixes the two things that made the raw Phase-0 WAVs an unfair audition:
  - peak-normalizes (raw cover output clips at ~1.35),
  - uses fast chunked VAE decode (full-clip MPS decode is ~10x slower).

Renders source + (styles x denoise) into test_output/audition/.
"""
from __future__ import annotations
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import loader
from engine.mps_compat import mps_sync
import torch

SR = 48000
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test_output", "audition")

STYLES = {
    "metal":   "aggressive heavy metal, distorted electric guitars, double kick drums",
    "lofi":    "lo-fi hip hop, mellow jazzy electric piano, vinyl crackle, boom-bap drums",
    "chiptune":"8-bit chiptune, retro video game, square-wave synth lead, arpeggios",
}
DENOISE = [0.5, 0.7]
SECONDS = 20.0
SEED = 1234


def chunked_decode(handler, latent_btd, chunk=64, overlap=8):
    vae = handler.vae
    lat = latent_btd.transpose(1, 2)
    T = lat.shape[-1]
    stride = chunk - 2 * overlap
    out, pos = [], 0
    with handler._load_model_context("vae"):
        while pos < T:
            lo, hi = max(0, pos - overlap), min(T, pos + stride + overlap)
            wav = vae.decode(lat[..., lo:hi].to(vae.dtype)).sample.float()
            tl = (pos - lo) * 1920
            tr = (hi - min(T, pos + stride)) * 1920
            out.append(wav[..., tl: wav.shape[-1] - tr if tr else None])
            pos += stride
    return torch.cat(out, dim=-1)


def peak_norm(wav, peak=0.97):
    m = wav.abs().max()
    return wav * (peak / m) if m > 1e-6 else wav


def main():
    from acestep.nodes.vae_nodes import VAEEncodeAudio
    from acestep.nodes.semantic_nodes import SemanticExtract
    from acestep.nodes.cond_nodes import EncodeText, EncodeConditioning
    from acestep.nodes.diffusion_nodes import StreamDenoise, OdeSolver
    from acestep.nodes.types import Audio

    os.makedirs(OUT, exist_ok=True)
    L = loader.load_model(device="mps")
    src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "source.wav")
    src_audio = loader.load_audio(src_path, duration=SECONDS)
    dur = src_audio.waveform.shape[-1] / SR
    loader.save_audio(Audio(waveform=peak_norm(src_audio.waveform), sample_rate=SR),
                      os.path.join(OUT, "00_source.wav"))

    source_latent = VAEEncodeAudio().execute(vae=L.vae, audio=src_audio)["latent"]
    context_latent = SemanticExtract().execute(model=L.model, latent=source_latent)["latent"]
    solver = OdeSolver().execute()["solver"]

    print("\nRendering audition set ->", OUT)
    for sname, style in STYLES.items():
        te = EncodeText().execute(clip=L.clip, tags=style, lyrics="", task_type="cover",
                                  bpm=120, duration=dur, key="C major", time_signature="4",
                                  language="en")["text_embed"]
        cond = EncodeConditioning().execute(model=L.model, text_embed=te, timbre_ref=source_latent)["conditioning"]
        for dn in DENOISE:
            t = time.time()
            lat = StreamDenoise().execute(
                model=L.model, solver=solver, positive=cond,
                context_latent=context_latent, source_latent=source_latent,
                steps=8, shift=3.0, denoise=dn, seed=SEED,
                pipeline_depth=1, drain=True, dcw_enabled=False, duration=dur, noise_on_cpu=True,
            )["latent"]
            wav = chunked_decode(L.handler, lat.tensor); mps_sync()
            wav = peak_norm(wav.squeeze(0))
            p = os.path.join(OUT, f"{sname}_dn{int(dn*100):02d}.wav")
            loader.save_audio(Audio(waveform=wav, sample_rate=SR), p)
            print(f"  {sname:9s} denoise={dn:.1f}  {time.time()-t:.1f}s")
            del lat, wav
            from engine.mps_compat import reclaim
            reclaim()
    print("\nDone. Listen to 00_source.wav first, then each style at dn50 (more source) vs dn70 (more style).")


if __name__ == "__main__":
    main()
