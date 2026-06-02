"""Real-time producer/consumer cover engine (Phase 2 core).

Wraps the JIT cover logic in a background PRODUCER thread that keeps a PCM ring
filled ~lookahead seconds ahead of a CONSUMER (the audio callback / sidecar
socket) draining at 1x. This is the architecture the plugin needs: the audio
thread never blocks on the model; it just pulls ready PCM. Live controls are
queued and applied on the next produced slice -> latency ~= the buffered amount
(~lookahead). Validates sustained real-time headlessly before any JUCE work.
"""
from __future__ import annotations

import threading
import time
from collections import deque

import numpy as np

from . import mps_compat
from .jit import JITCover, FPS, SR, SPF

import torch  # noqa: E402


class RealtimeCover:
    def __init__(self, device="mps", steps=8, window_s=20.0, lookahead_s=1.0, slice_s=1.0, denoise=0.8, seed=1234,
                 config_path="acestep-v15-turbo"):
        self.jit = JITCover(device=device, steps=steps, config_path=config_path)
        self.window_s = window_s
        self.lookahead_samp = int(round(lookahead_s * SR))   # ring counts AUDIO SAMPLES
        self.SL = max(1, int(round(slice_s * FPS)))          # slice length in LATENT frames
        self.denoise = denoise
        self.seed = seed
        # ring of produced PCM chunks (np.float32 [m,2]) + counters in frames
        self._ring = deque()
        self._ring_frames = 0
        self._lock = threading.Lock()
        self._ctrl = deque()           # pending (kind, value)
        self._produced_f = 0
        self._consumed_f = 0
        self.underruns = 0
        self.full_T = 0
        self._running = False
        self._done = False
        self._thread = None
        # telemetry
        self.regens = 0
        self.max_step_ms = 0.0

    # ---- setup ----
    def load_track(self, path, seconds=None):
        self.jit.load_track(path, seconds=seconds)
        self.full_T = self.jit.source.latent.tensor.shape[1]
        return self

    def set_style(self, tags, denoise=None, character=None, timbre=None, send_bpm=None, send_key=None):
        self.jit.set_style(tags, denoise=denoise if denoise is not None else self.denoise,
                           character=character, timbre=timbre, send_bpm=send_bpm, send_key=send_key)
        return self

    def set_metas(self, send_bpm=None, send_key=None):
        with self._lock:
            self._ctrl.append(("metas", (send_bpm, send_key)))

    def peaks(self):
        return self.jit.peaks

    # ---- live controls (any thread) ----
    def set_prompt(self, tags):
        with self._lock:
            self._ctrl.append(("prompt", tags))

    def set_denoise(self, v):
        with self._lock:
            self._ctrl.append(("denoise", float(v)))

    def set_character(self, c):
        with self._lock:
            self._ctrl.append(("character", float(c)))

    def set_evolve(self, on):
        self.jit.evolve = bool(on)   # read live in jit._gen; plain bool, thread-safe

    def reconfigure(self, steps=None, window_s=None):
        """Apply Steps/Window changes, reusing the loaded source (no re-encode).
        Restarts playback from the top (like plugin_morph's reload-on-change)."""
        self.stop()
        if window_s is not None:
            self.window_s = float(window_s)
        if steps is not None and int(steps) != self.jit.steps:
            self.jit.steps = int(steps)
            # rebuild the StreamPipeline handle at the new depth, reusing src+cond
            self.jit.handle = self.jit.session.stream(
                source=self.jit.source, conditioning=self.jit.cond,
                steps=self.jit.steps, shift=self.jit.shift, pipeline_depth=1, dcw_enabled=False)
        with self._lock:
            self._ring.clear(); self._ring_frames = 0
            self._produced_f = 0; self._consumed_f = 0
        self.start()

    def stats(self):
        dur = max(1e-6, self.full_T / FPS)
        with self._lock:
            buffered = self._ring_frames / SR
            played = self._consumed_f / SR
        return {"buffered_s": round(buffered, 2), "regens": self.regens,
                "worst_regen_ms": round(self.max_step_ms), "underruns": self.underruns,
                "progress": (played % dur) / dur, "duration_s": round(dur, 1)}

    # ---- consumer (audio thread) ----
    def read(self, n):
        """Pull n frames -> np.float32 [n,2]; zero-fill (count underrun) if short."""
        out = np.zeros((n, 2), dtype=np.float32)
        got = 0
        with self._lock:
            while got < n and self._ring:
                chunk = self._ring[0]
                take = min(n - got, chunk.shape[0])
                out[got:got + take] = chunk[:take]
                if take == chunk.shape[0]:
                    self._ring.popleft()
                else:
                    self._ring[0] = chunk[take:]
                got += take
                self._ring_frames -= take
            self._consumed_f += n           # playback time advances even on underrun
        if got < n:
            self.underruns += 1
        return out

    def buffered_s(self):
        with self._lock:
            return self._ring_frames / SR

    # ---- producer thread ----
    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._produce, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    def _push(self, pcm_2xN):
        chunk = pcm_2xN.detach().t().contiguous().numpy().astype(np.float32)  # [N,2]
        with self._lock:
            self._ring.append(chunk)
            self._ring_frames += chunk.shape[0]
            self._produced_f += chunk.shape[0]

    def _produce(self):
        torch.set_grad_enabled(False)   # grad flag is THREAD-LOCAL; disable in this worker
        jit = self.jit
        W = int(round(self.window_s * FPS))
        W = min(W, self.full_T)
        committed_f = 0
        win_start = -10 ** 9
        win_lat = None
        win_tiles: dict = {}
        win_nf = 0
        force = False
        while self._running:
            if committed_f >= self.full_T:     # loop the track for continuous live play
                committed_f = 0; win_lat = None
            # drain control queue
            with self._lock:
                ctrls = list(self._ctrl); self._ctrl.clear()
                ahead = self._produced_f - self._consumed_f
            for kind, val in ctrls:
                if kind == "prompt":
                    jit._set_prompt(val)
                elif kind == "denoise":
                    jit.denoise = val
                elif kind == "character":
                    jit.set_character(val)
                elif kind == "metas":
                    jit.set_metas(send_bpm=val[0], send_key=val[1])
                force = True
            # keep the ring ~lookahead ahead; if full enough, wait (ahead in samples)
            if ahead >= self.lookahead_samp + self.SL * SPF and not force:
                time.sleep(0.003)
                continue
            if win_lat is None or committed_f >= win_start + win_nf or force:
                win_start = max(0, min(committed_f, self.full_T - W))
                t0 = time.perf_counter()
                win_lat = jit._gen(win_start, min(win_start + W, self.full_T), self.seed)
                win_nf = win_lat.tensor.shape[1]
                win_tiles = {}
                f1 = min(committed_f + self.SL, win_start + win_nf, self.full_T)
                seg = jit._ensure_tiles(win_lat, win_tiles, committed_f - win_start, f1 - win_start)
                mps_compat.mps_sync()
                self.max_step_ms = max(self.max_step_ms, (time.perf_counter() - t0) * 1000)
                self.regens += 1
                force = False
                self._push(seg)
                committed_f = f1
                mps_compat.reclaim()
                continue
            f1 = min(committed_f + self.SL, win_start + win_nf, self.full_T)
            seg = jit._ensure_tiles(win_lat, win_tiles, committed_f - win_start, f1 - win_start)
            mps_compat.mps_sync()
            self._push(seg)
            committed_f = f1
        self._done = True

    @property
    def done(self):
        return self._done

    def close(self):
        self.stop()
        self.jit.close()
