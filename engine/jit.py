"""Small-lookahead just-in-time streaming cover (Phase 1 v1 / plugin engine).

The audio you hear was generated `lookahead` seconds ago, so control latency >=
lookahead. With RTF ~0.1 we have the headroom to keep lookahead TINY (~1 s) and
regenerate the current window the instant a knob/prompt changes -> ~1 s control
latency, while windows stay large (good structure) and decode stays full-quality
(decode the latent in ~1 s sub-regions with receptive-field margin, not 0.36 s
slices). This is also exactly the producer/consumer shape the Phase 2 plugin
needs (a generator filling a ring a little ahead of the audio callback).

Model (per output frame, decoded in order):
  - controls issued at playback time t take effect at output time t+lookahead
    (we never rewrite already-committed audio) -> measured latency = lookahead;
  - a control change (or the playhead leaving the window) forces a window regen
    with current settings (fixed seed -> coherent variation), crossfaded into the
    committed output over a short equal-power region (no timeline compression).
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
SPF = 1920  # samples per latent frame


@dataclass
class JITStats:
    audio_s: float = 0.0
    compute_wall_s: float = 0.0
    regens: int = 0
    gen_ms: list = field(default_factory=list)
    dec_ms: list = field(default_factory=list)
    lookahead_s: float = 0.0
    control_latency_s: list = field(default_factory=list)
    max_step_ms: float = 0.0

    @property
    def rtf(self):
        return self.compute_wall_s / max(1e-9, self.audio_s)

    def summary(self):
        import statistics as st
        g = self.gen_ms or [0.0]; d = self.dec_ms or [0.0]
        return (
            f"audio={self.audio_s:.1f}s compute={self.compute_wall_s:.2f}s RTF={self.rtf:.3f} "
            f"({'REAL-TIME OK' if self.rtf < 1 else 'TOO SLOW'})\n"
            f"  regens={self.regens} gen med={st.median(g):.0f}ms  decode/slice med={st.median(d):.0f}ms\n"
            f"  lookahead={self.lookahead_s:.2f}s -> control latency ~{self.lookahead_s:.2f}s"
            + (f"  (measured: {[f'{x:.2f}s' for x in self.control_latency_s]})" if self.control_latency_s else "")
            + f"\n  worst single step (regen+decode)={self.max_step_ms:.0f}ms"
        )


def _eq(n, device, dtype):
    t = torch.linspace(0, 1, n, device=device, dtype=dtype)
    return torch.sin(t * math.pi / 2), torch.cos(t * math.pi / 2)


class JITCover:
    def __init__(self, device="mps", steps=8, shift=3.0):
        from acestep.engine.session import Session
        self.session = Session(device=device, decoder_backend="eager", vae_backend="eager",
                               use_flash_attention=False, vae_window=0.0)
        self.steps = steps; self.shift = shift
        self.source = None; self.cond = None; self.handle = None
        self.track_dur = 0.0; self.denoise = 0.8

    def load_track(self, path, seconds=None):
        from . import loader
        a = loader.load_audio(path, duration=seconds)
        self.track_dur = a.waveform.shape[-1] / SR
        self.source = self.session.prepare_source(a)
        return self

    def set_style(self, tags, denoise=0.8, bpm=120, key="C major"):
        from acestep.constants import TASK_INSTRUCTIONS
        self.denoise = denoise
        self.cond = self.session.encode_text(tags=tags, instruction=TASK_INSTRUCTIONS["cover"],
                                             refer_latent=self.source.latent, duration=self.track_dur, bpm=bpm, key=key)
        if self.handle is None:
            self.handle = self.session.stream(source=self.source, conditioning=self.cond,
                                              steps=self.steps, shift=self.shift, pipeline_depth=1, dcw_enabled=False)
        else:
            self.handle.conditioning = self.cond
        return self

    def _set_prompt(self, tags, bpm=120, key="C major"):
        from acestep.constants import TASK_INSTRUCTIONS
        self.cond = self.session.encode_text(tags=tags, instruction=TASK_INSTRUCTIONS["cover"],
                                             refer_latent=self.source.latent, duration=self.track_dur, bpm=bpm, key=key)
        self.handle.conditioning = self.cond

    def _gen(self, w0, w1, seed):
        from acestep.nodes.types import Latent
        from acestep.engine.session import PreparedSource
        src = self.source.latent.tensor[:, w0:w1, :]
        ctx = self.source.context_latent.tensor[:, w0:w1, :]
        self.handle.context_latent = Latent(tensor=ctx)
        self.handle.source = PreparedSource(latent=Latent(tensor=src), context_latent=Latent(tensor=ctx))
        lat = self.handle.tick(drain=True, denoise=self.denoise, seed=seed)
        mps_compat.mps_sync()
        return lat

    def _full_decode(self, lat, chunk=64, overlap=8):
        """Full tiled decode of a window latent -> [2, T*SPF] on CPU (cached)."""
        vae = self.session.handler.vae
        lb = lat.tensor.transpose(1, 2)
        T = lb.shape[-1]
        with self.session.handler._load_model_context("vae"):
            if T <= chunk:
                return vae.decode(lb.to(vae.dtype)).sample.float().squeeze(0).cpu()
            stride = chunk - 2 * overlap
            out, pos = [], 0
            while pos < T:
                lo, hi = max(0, pos - overlap), min(T, pos + stride + overlap)
                w = vae.decode(lb[..., lo:hi].to(vae.dtype)).sample.float()
                tl = (pos - lo) * SPF
                tr = (hi - min(T, pos + stride)) * SPF
                out.append(w[..., tl: w.shape[-1] - tr if tr else None])
                pos += stride
        return torch.cat(out, dim=-1).squeeze(0).cpu()

    def render(self, controls=None, window_s=10.0, lookahead_s=1.0, slice_s=1.0,
               xfade_s=0.12, seed=1234) -> tuple:
        """Headless JIT render with a control schedule.

        controls: list of (issue_time_s, kind, value); kind in {'prompt','denoise'}.
        Returns (waveform [2, samples] float32, JITStats).
        """
        assert self.handle is not None, "call set_style() first"
        W = int(round(window_s * FPS))
        SL = max(1, int(round(slice_s * FPS)))
        XF = int(round(xfade_s * SR))
        full_T = self.source.latent.tensor.shape[1]
        full_s = full_T / FPS
        W = min(W, full_T)

        XF_f = max(1, XF // SPF)          # crossfade in whole latent frames
        XF = XF_f * SPF                    # ...and exact samples (no drift)
        st = JITStats(audio_s=full_s, lookahead_s=lookahead_s)
        pending = sorted(controls or [], key=lambda x: x[0])
        out = None
        committed_f = 0                   # output frames committed (playback trails by lookahead)
        win_start = -10 ** 9
        win_audio = None                  # full-decoded current window [2, Wlat*SPF] (cached on CPU)
        force = False

        def append(wav, xf):
            nonlocal out
            if out is None:
                out = wav
            elif xf > 0 and out.shape[-1] >= xf and wav.shape[-1] >= xf:
                fin, fout = _eq(xf, wav.device, wav.dtype)
                out = torch.cat([out[:, :-xf], out[:, -xf:] * fout + wav[:, :xf] * fin, wav[:, xf:]], dim=-1)
            else:
                out = torch.cat([out, wav], dim=-1)

        wall0 = time.perf_counter()
        while committed_f < full_T:
            # controls issued >= lookahead ago take effect from this frame on
            while pending and committed_f / FPS >= pending[0][0] + lookahead_s:
                t, kind, val = pending.pop(0)
                if kind == "prompt":
                    self._set_prompt(val)
                elif kind == "denoise":
                    self.denoise = float(val)
                force = True
                st.control_latency_s.append(lookahead_s)

            # (re)generate + FULL-decode the window when we leave it or a control fired
            win_end = win_start + (0 if win_audio is None else win_audio.shape[-1] // SPF)
            if win_audio is None or committed_f >= win_end or force:
                back = XF_f if out is not None else 0
                win_start = max(0, min(committed_f - back, full_T - W))
                g0 = time.perf_counter()
                lat = self._gen(win_start, min(win_start + W, full_T), seed)
                st.gen_ms.append((time.perf_counter() - g0) * 1000)
                d0 = time.perf_counter()
                win_audio = self._full_decode(lat)
                st.dec_ms.append((time.perf_counter() - d0) * 1000)
                st.regens += 1
                st.max_step_ms = max(st.max_step_ms, st.gen_ms[-1] + st.dec_ms[-1])
                force = False
                # emit the crossfade-overlap slice straight away (from back-of-committed)
                from_f = committed_f - back
                seg = win_audio[:, (from_f - win_start) * SPF: (min(committed_f + SL, win_start + win_audio.shape[-1] // SPF, full_T) - win_start) * SPF]
                append(seg, XF if back else 0)
                committed_f = from_f + seg.shape[-1] // SPF
                mps_compat.reclaim()
                continue

            # steady: copy the next slice from the cached window decode (free, full quality)
            f1 = min(committed_f + SL, win_end, full_T)
            seg = win_audio[:, (committed_f - win_start) * SPF: (f1 - win_start) * SPF]
            append(seg, 0)
            committed_f = f1

        st.compute_wall_s = time.perf_counter() - wall0
        return out, st

    def close(self):
        try:
            if self.handle is not None:
                self.handle.close()
            self.session.close()
        except Exception:
            pass
