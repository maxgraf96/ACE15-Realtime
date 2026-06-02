"""Thin loader around DEMON's node-based ACE-Step engine, MPS-first.

Keeps the reference repos read-only. ``load_model`` returns the three node
handles (model/clip/vae) that the DEMON node graph consumes, plus the
underlying ``ModelContext`` (``handler``) for direct access to the DiT, VAE,
text encoder, and timing helpers.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from . import mps_compat

mps_compat.install()  # add DEMON to path + no-op cuda + apg fp32 BEFORE acestep imports

import torch  # noqa: E402
import soundfile as sf  # noqa: E402


@dataclass
class Loaded:
    handler: object   # acestep.engine.model_context.ModelContext
    model: object     # ModelHandle (node wire)
    clip: object      # CLIPHandle
    vae: object       # VAEHandle
    device: str
    dtype: object


def load_model(
    device: str = "mps",
    config_path: str = "acestep-v15-turbo",
    use_flash_attention: bool = False,
) -> Loaded:
    """Load the turbo-2B cover stack (DiT + Oobleck VAE + Qwen3 text encoder)."""
    from acestep.nodes.model_nodes import LoadModel
    from acestep.paths import checkpoints_dir

    t0 = time.time()
    handles = LoadModel().execute(
        project_root=str(checkpoints_dir()),
        config_path=config_path,
        device=device,
        use_flash_attention=use_flash_attention,
        decoder_backend="eager",
        vae_backend="eager",
    )
    model, clip, vae = handles["model"], handles["clip"], handles["vae"]
    handler = model.handler
    print(
        f"[loader] loaded {config_path} on {handler.device} "
        f"dtype={handler.dtype} in {time.time() - t0:.1f}s"
    )
    return Loaded(
        handler=handler, model=model, clip=clip, vae=vae,
        device=str(handler.device), dtype=handler.dtype,
    )


def load_audio(path: str, duration: Optional[float] = None, sr_target: int = 48000):
    """Load a wav into an acestep ``Audio`` payload (stereo, 48 kHz)."""
    from acestep.nodes.types import Audio

    data, sr = sf.read(path, dtype="float32")
    wav = torch.from_numpy(data.T if data.ndim > 1 else data.reshape(1, -1))
    if sr != sr_target:
        from torchaudio.transforms import Resample
        wav = Resample(sr, sr_target)(wav)
    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)  # mono -> stereo
    wav = wav[:2]
    if duration is not None:
        wav = wav[:, : int(duration * sr_target)]
    return Audio(waveform=wav, sample_rate=sr_target)


def save_audio(audio, path: str) -> None:
    from acestep.nodes.types import Audio  # noqa: F401

    wav = audio.waveform
    if wav.dim() == 3:
        wav = wav.squeeze(0)
    sf.write(path, wav.detach().cpu().float().numpy().T, audio.sample_rate)
    print(f"[loader] saved {path}")
