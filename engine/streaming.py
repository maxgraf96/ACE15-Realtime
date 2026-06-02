"""Headless walk-window streaming cover engine (Phase 1).

Built on DEMON's Session.stream / StreamPipeline (running on MPS via
engine.mps_compat). DCW OFF, RCFG/CFG off, depth-1 drain per window, fp32.

Two decode modes:
  - "chunk" (default): each walk-window chunk's latent is generated once (drain)
    and FULL-decoded (tiled, fast on MPS), then forward-written with a short
    equal-power crossfade at chunk seams. Best quality (preserves transients);
    still real-time (gen ~0.5s + decode ~0.5s per 10s chunk -> RTF ~0.1).
  - "window": models the latency-critical hot path used for live-morph — decode
    only the ~0.36 s slice at the playhead per tick (StreamVAEDecode). Lowest
    per-tick latency but smears transients across the many slice joins (onset
    corr drops a lot); reserved for the live-knob UX, measured for its RTF.

Why chunk-decode for a track cover: we already hold the whole chunk latent, so
decoding it in one (tiled) pass avoids the per-0.36s-slice transient smearing
that the windowed decode incurs. The windowed decode only wins when the latent
itself changes every tick (continuous live morph), which is a separate mode.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional

from . import mps_compat  # installs shims on import

import torch  # noqa: E402

FPS = 25.0
SR = 48000
SAMPLES_PER_FRAME = 1920


@dataclass
class StreamStats:
    audio_s: float = 0.0
    compute_wall_s: float = 0.0
    gen_ms: list = field(default_factory=list)
    dec_ms: list = field(default_factory=list)
    max_chunk_ms: float = 0.0        # worst gen+decode for one chunk
    chunks: int = 0

    @property
    def rtf(self) -> float:
        return self.compute_wall_s / max(1e-9, self.audio_s)

    def summary(self) -> str:
        import statistics as st
        g = self.gen_ms or [0.0]
        d = self.dec_ms or [0.0]
        return (
            f"audio={self.audio_s:.1f}s compute={self.compute_wall_s:.2f}s "
            f"RTF={self.rtf:.3f} ({'REAL-TIME OK' if self.rtf < 1 else 'TOO SLOW'})\n"
            f"  chunks={self.chunks}  gen/chunk med={st.median(g):.0f} max={max(g):.0f} ms"
            f"  decode/chunk med={st.median(d):.0f} max={max(d):.0f} ms\n"
            f"  worst chunk (gen+decode)={self.max_chunk_ms:.0f} ms "
            f"(min producer lookahead for gapless playback)"
        )


def _equal_power(n, device, dtype):
    t = torch.linspace(0, 1, n, device=device, dtype=dtype)
    return torch.sin(t * math.pi / 2), torch.cos(t * math.pi / 2)


class StreamingCover:
    def __init__(self, device: str = "mps", vae_window: float = 0.36,
                 steps: int = 8, shift: float = 3.0):
        from acestep.engine.session import Session
        self.session = Session(
            device=device, decoder_backend="eager", vae_backend="eager",
            use_flash_attention=False, vae_window=vae_window,
        )
        self.steps = steps
        self.shift = shift
        self.vae_window = vae_window
        self.source = None
        self.cond = None
        self.handle = None
        self.track_dur = 0.0
        self.denoise = 0.8

    # ---- load + style ----
    def load_track(self, path: str, seconds: Optional[float] = None):
        from . import loader
        audio = loader.load_audio(path, duration=seconds)
        self.track_dur = audio.waveform.shape[-1] / SR
        self.source = self.session.prepare_source(audio)
        return self

    def set_style(self, tags: str, denoise: float = 0.8, bpm: int = 120, key: str = "C major"):
        from acestep.constants import TASK_INSTRUCTIONS
        self.denoise = denoise
        self.cond = self.session.encode_text(
            tags=tags, instruction=TASK_INSTRUCTIONS["cover"],
            refer_latent=self.source.latent, duration=self.track_dur, bpm=bpm, key=key,
        )
        if self.handle is None:
            self.handle = self.session.stream(
                source=self.source, conditioning=self.cond,
                steps=self.steps, shift=self.shift,
                pipeline_depth=1, dcw_enabled=False,
            )
        else:
            self.handle.conditioning = self.cond
        return self

    def set_prompt_live(self, tags: str, bpm: int = 120, key: str = "C major"):
        from acestep.constants import TASK_INSTRUCTIONS
        t = time.perf_counter()
        self.cond = self.session.encode_text(
            tags=tags, instruction=TASK_INSTRUCTIONS["cover"],
            refer_latent=self.source.latent, duration=self.track_dur, bpm=bpm, key=key,
        )
        self.handle.conditioning = self.cond
        mps_compat.mps_sync()
        return (time.perf_counter() - t) * 1000

    # ---- generation + decode ----
    def _gen(self, w0: int, w1: int, seed: int):
        from acestep.nodes.types import Latent
        from acestep.engine.session import PreparedSource
        src = self.source.latent.tensor[:, w0:w1, :]
        ctx = self.source.context_latent.tensor[:, w0:w1, :]
        self.handle.context_latent = Latent(tensor=ctx)
        self.handle.source = PreparedSource(
            latent=Latent(tensor=src), context_latent=Latent(tensor=ctx))
        lat = self.handle.tick(drain=True, denoise=self.denoise, seed=seed)
        mps_compat.mps_sync()
        return lat

    def _chunk_decode(self, latent, chunk_frames: int = 64, overlap: int = 8):
        """Full tiled decode of a chunk latent -> [2, samples] (fast on MPS)."""
        vae = self.session.handler.vae
        lb = latent.tensor.transpose(1, 2)  # [1,64,T]
        T = lb.shape[-1]
        if T <= chunk_frames:
            with self.session.handler._load_model_context("vae"):
                return vae.decode(lb.to(vae.dtype)).sample.float().squeeze(0)
        stride = chunk_frames - 2 * overlap
        out, pos = [], 0
        with self.session.handler._load_model_context("vae"):
            while pos < T:
                lo, hi = max(0, pos - overlap), min(T, pos + stride + overlap)
                w = vae.decode(lb[..., lo:hi].to(vae.dtype)).sample.float()
                tl = (pos - lo) * SAMPLES_PER_FRAME
                tr = (hi - min(T, pos + stride)) * SAMPLES_PER_FRAME
                out.append(w[..., tl: w.shape[-1] - tr if tr else None])
                pos += stride
        return torch.cat(out, dim=-1).squeeze(0)

    # ---- the walk-window render ----
    def render(self, win_s: float = 10.0, seed: int = 1234,
               prompt_schedule: Optional[list] = None, overlap_s: float = 0.5) -> tuple:
        """Walk the track in OVERLAPPING windows; full-decode each chunk.

        Consecutive windows overlap by ``overlap_s`` (hop = win - overlap) and
        the generated overlap region is equal-power crossfaded — this keeps the
        output timeline EXACT (no per-seam compression) while smoothing the join
        between independent generations, so rhythm/transients stay aligned.

        prompt_schedule: list of (at_seconds, tags) live prompt swaps.
        Returns (waveform [2, samples] float32, StreamStats, prompt_latency_ms).
        """
        assert self.handle is not None, "call set_style() first"
        W = int(round(win_s * FPS))
        full_T = self.source.latent.tensor.shape[1]
        full_s = full_T / FPS
        OV = 0 if W >= full_T else int(round(overlap_s * FPS))
        hop = max(1, W - OV)

        st = StreamStats(audio_s=full_s)
        chunk_audios = []
        plat = []
        sched = sorted(prompt_schedule or [], key=lambda x: x[0])

        wall0 = time.perf_counter()
        w0 = 0
        while w0 < full_T:
            w1 = min(w0 + W, full_T)
            if w1 - w0 < 4:
                break
            chunk_start_s = w0 / FPS
            while sched and chunk_start_s >= sched[0][0]:
                _, tags = sched.pop(0)
                plat.append(self.set_prompt_live(tags))

            g0 = time.perf_counter()
            lat = self._gen(w0, w1, seed)
            gen_ms = (time.perf_counter() - g0) * 1000
            d0 = time.perf_counter()
            wav = self._chunk_decode(lat).detach().float().cpu()
            mps_compat.mps_sync()
            dec_ms = (time.perf_counter() - d0) * 1000

            st.gen_ms.append(gen_ms)
            st.dec_ms.append(dec_ms)
            st.max_chunk_ms = max(st.max_chunk_ms, gen_ms + dec_ms)
            st.chunks += 1
            chunk_audios.append(wav)
            mps_compat.reclaim()
            if w1 >= full_T:
                break
            w0 += hop

        st.compute_wall_s = time.perf_counter() - wall0
        # crossfade exactly the generated overlap region (OV frames) -> exact timeline
        out = self._stitch(chunk_audios, OV * SAMPLES_PER_FRAME / SR)
        return out, st, plat

    @staticmethod
    def _stitch(chunks, xfade_s: float):
        if not chunks:
            return torch.zeros(2, 1)
        xf = int(xfade_s * SR)
        out = chunks[0]
        for nxt in chunks[1:]:
            if xf > 0 and out.shape[-1] >= xf and nxt.shape[-1] >= xf:
                fin, fout = _equal_power(xf, out.device, out.dtype)
                out = torch.cat(
                    [out[:, :-xf], out[:, -xf:] * fout + nxt[:, :xf] * fin, nxt[:, xf:]],
                    dim=-1)
            else:
                out = torch.cat([out, nxt], dim=-1)
        return out

    # ---- windowed-decode RTF probe (models the live-morph hot path) ----
    def measure_windowed_rtf(self, win_s: float = 10.0, seed: int = 1234) -> StreamStats:
        """Generate chunks, then decode 0.36 s slices at an advancing playhead.
        Reports the per-tick (forward-amortized + windowed decode) RTF only."""
        W = int(round(win_s * FPS))
        full_T = self.source.latent.tensor.shape[1]
        full_s = full_T / FPS
        n_chunks = max(1, math.ceil(full_T / W))
        st = StreamStats(audio_s=full_s)
        wall0 = time.perf_counter()
        cur = -1
        lat = None
        ph = 0.0
        while ph < full_s - 1e-6:
            c = min(int(ph // win_s), n_chunks - 1)
            w0 = max(0, min(c * W, full_T - W)); w1 = w0 + W
            if c != cur:
                g0 = time.perf_counter(); lat = self._gen(w0, w1, seed)
                st.gen_ms.append((time.perf_counter() - g0) * 1000); cur = c; st.chunks += 1
            local = max(0.0, min(ph - w0 / FPS, win_s - self.vae_window))
            d0 = time.perf_counter(); self.handle.decode(lat, t_start=local); mps_compat.mps_sync()
            st.dec_ms.append((time.perf_counter() - d0) * 1000)
            ph += self.vae_window
            if len(st.dec_ms) % 50 == 0:
                mps_compat.reclaim()
        st.compute_wall_s = time.perf_counter() - wall0
        return st

    def close(self):
        try:
            if self.handle is not None:
                self.handle.close()
            self.session.close()
        except Exception:
            pass
