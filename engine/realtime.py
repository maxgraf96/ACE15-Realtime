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
from .jit import JITCover, FPS, SR, SPF, _eq

import torch  # noqa: E402


class RealtimeCover:
    def __init__(self, device="mps", steps=8, window_s=20.0, lookahead_s=1.0, slice_s=1.0, denoise=0.8, seed=1234,
                 config_path="acestep-v15-turbo", xfade_s=0.12, margin_s=1.0):
        self.jit = JITCover(device=device, steps=steps, config_path=config_path)
        self.window_s = window_s
        self.lookahead_samp = int(round(lookahead_s * SR))   # ring counts AUDIO SAMPLES
        self.SL = max(1, int(round(slice_s * FPS)))          # slice length in LATENT frames
        # Window seams = INTERIOR overlap-add hand-off. Each window is an independent
        # generation; its EDGES are the model's weakest region (least DiT context), so
        # we never splice there. Instead we transition _margin_f frames BEFORE the
        # window edge, generate the next window backed up so the playhead lands in its
        # INTERIOR, and equal-power crossfade _xf_f frames where BOTH windows have full
        # context (the degraded edges are discarded). Position-aligned → no timeline
        # compression. A held tail of _xf_samp lets the next window blend into it.
        self._xf_f = max(1, int(round(xfade_s * FPS)))       # crossfade length, latent frames
        self._margin_f = max(self._xf_f, int(round(margin_s * FPS)))  # edge margin (>= crossfade)
        self._xf_samp = self._xf_f * SPF                     # crossfade in samples
        self._tail = None                                    # held-back audio [2,_xf_samp]
        # DCW raises the output level ~1.4 dB (it slammed the limiter); make-up gain
        # to bring DCW-on output back in line with DCW-off so it stays clean.
        self._dcw_gain = 0.85
        self.denoise = denoise
        self.seed = seed
        # ring of produced PCM chunks (np.float32 [m,2]) + counters in frames
        self._ring = deque()
        self._ring_frames = 0
        self._lock = threading.Lock()
        self._ctrl = deque()           # pending (kind, value)
        self._produced_f = 0
        self._consumed_f = 0           # frames PULLED by the consumer (incl. underrun zeros) — paces the buffer gate
        self._real_f = 0               # frames of REAL audio consumed — drives the playhead (no startup-silence drift)
        self._seek_pos_s = 0.0         # track-seconds at the last seek (playhead origin)
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

    def set_dcw(self, enabled):
        with self._lock:
            self._ctrl.append(("dcw", bool(enabled)))

    def seek(self, fraction):
        """Jump playback to a fractional position (0..1) of the track. Queued so the
        producer thread (which owns the playback cursor) performs the jump: it moves
        the cursor, flushes the stale ring, and realigns the playhead. If stopped,
        the seek is applied when playback (re)starts."""
        f = max(0.0, min(1.0, float(fraction)))
        seek_f = max(0, min(self.full_T - 1, int(f * self.full_T))) if self.full_T else 0
        with self._lock:
            self._ctrl.append(("seek", seek_f))

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
                steps=self.jit.steps, shift=self.jit.shift, pipeline_depth=1,
                **self.jit._dcw_kwargs())
        with self._lock:
            self._ring.clear(); self._ring_frames = 0
            self._produced_f = 0; self._consumed_f = 0; self._real_f = 0
            self._seek_pos_s = 0.0; self._ctrl.clear()   # restart from the top
        self.start()

    def stats(self):
        dur = max(1e-6, self.full_T / FPS)
        with self._lock:
            buffered = self._ring_frames / SR
            played = self._real_f / SR                 # real audio heard, not pull-pace (no startup drift)
            pos = (self._seek_pos_s + played) % dur   # playhead = seek origin + elapsed
        return {"buffered_s": round(buffered, 2), "regens": self.regens,
                "worst_regen_ms": round(self.max_step_ms), "underruns": self.underruns,
                "progress": pos / dur, "duration_s": round(dur, 1)}

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
            self._consumed_f += n           # pull pace (incl. underrun zeros) — for the buffer gate
            self._real_f += got             # REAL audio only — for the playhead (silence doesn't advance it)
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

    def _emit(self, seg):
        """Push contiguous audio, holding back the last _xf_samp samples so the next
        window seam can crossfade into them. `seg` is torch [2,N], time-contiguous
        with the held tail."""
        if self._tail is not None:
            seg = torch.cat([self._tail, seg], dim=-1)
            self._tail = None
        xf = self._xf_samp
        if seg.shape[-1] > xf:
            self._push(seg[:, :-xf])
            self._tail = seg[:, -xf:].clone()
        else:
            self._tail = seg.clone()          # shorter than a crossfade; keep accumulating

    def _emit_seam(self, seg):
        """Equal-power crossfade a freshly generated window's head into the held tail
        (the previous window's last _xf_samp samples), then emit the remainder. The
        producer backs the new window up by _xf_f frames so seg[:xf] covers the SAME
        timeline as the tail (no compression); at the loop point it blends track-end
        into track-start."""
        if self._tail is None:
            self._emit(seg); return
        ov = min(self._xf_samp, self._tail.shape[-1], seg.shape[-1])
        if ov <= 0:
            self._emit(seg); return
        fin, fout = _eq(ov, seg.device, seg.dtype)
        head = self._tail[:, :-ov] if self._tail.shape[-1] > ov else None
        blend = self._tail[:, -ov:] * fout + seg[:, :ov] * fin
        self._tail = None
        parts = ([head] if head is not None else []) + [blend, seg[:, ov:]]
        self._emit(torch.cat(parts, dim=-1))

    def _produce(self):
        torch.set_grad_enabled(False)   # grad flag is THREAD-LOCAL; disable in this worker
        self._tail = None               # fresh stream: no carry-over crossfade tail
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
                elif kind == "dcw":
                    jit.set_dcw(enabled=val)
                elif kind == "seek":
                    committed_f = val          # jump the playback cursor
                    win_lat = None             # force a regen at the new position
                    self._tail = None          # clean cut at the seek point (no crossfade)
                    with self._lock:           # flush the now-stale buffered audio
                        self._ring.clear(); self._ring_frames = 0
                        self._produced_f = 0; self._consumed_f = 0; self._real_f = 0
                        self._seek_pos_s = val / FPS   # playhead origin = seek point
                force = True
            # Keep the ring ahead by at least one worst-case regen. A regen blocks
            # production for ~max_step_ms; if the buffer is smaller than that, the
            # consumer underruns mid-regen -> glitchy/"metallic" gaps. So the target
            # auto-grows to cover the MEASURED worst regen (+0.4s margin), which is
            # what makes a big window / slow model / DCW safe. Costs control latency
            # (= buffer depth); capped so a one-off stall can't balloon it.
            target = max(self.lookahead_samp, int(min(8.0, self.max_step_ms / 1000 + 0.4) * SR))
            if ahead >= target + self.SL * SPF and not force:
                time.sleep(0.003)
                continue
            # Hand off to the next window a margin BEFORE the current one's edge (the
            # last window plays fully to the track end). usable_end = where this window
            # stops being trusted; the [usable_end, win_end] tail is the discarded edge.
            win_end = win_start + win_nf
            last_win = win_end >= self.full_T
            usable_end = win_end if last_win else win_end - self._margin_f
            if win_lat is None or committed_f >= usable_end or force:
                # Seam: back the new window up by _margin_f so the playhead lands in its
                # INTERIOR; decode only from _xf_f before the playhead (the crossfade
                # region, aligned with the held tail). back=0 at stream start / loop.
                seam = self._tail is not None and committed_f >= self._margin_f
                back = self._margin_f if seam else 0
                win_start = max(0, min(committed_f - back, self.full_T - W))
                from_f = max(win_start, committed_f - self._xf_f) if seam else committed_f
                t0 = time.perf_counter()
                win_lat = jit._gen(win_start, min(win_start + W, self.full_T), self.seed)
                win_nf = win_lat.tensor.shape[1]
                win_tiles = {}
                f1 = min(committed_f + self.SL, win_start + win_nf, self.full_T)
                seg = jit._ensure_tiles(win_lat, win_tiles, from_f - win_start, f1 - win_start)
                mps_compat.mps_sync()
                if jit.dcw_enabled: seg = seg * self._dcw_gain   # make-up for DCW's level boost
                self.max_step_ms = max(self.max_step_ms, (time.perf_counter() - t0) * 1000)
                self.regens += 1
                force = False
                # Crossfade whenever a tail exists (interior seam, or loop end→start);
                # plain emit only at the very first window / after a seek (tail cleared).
                self._emit_seam(seg) if self._tail is not None else self._emit(seg)
                committed_f = from_f + seg.shape[-1] // SPF
                mps_compat.reclaim()
                continue
            f1 = min(committed_f + self.SL, usable_end, self.full_T)
            seg = jit._ensure_tiles(win_lat, win_tiles, committed_f - win_start, f1 - win_start)
            mps_compat.mps_sync()
            if jit.dcw_enabled: seg = seg * self._dcw_gain   # make-up for DCW's level boost
            self._emit(seg)
            committed_f = f1
        self._done = True

    @property
    def done(self):
        return self._done

    def close(self):
        self.stop()
        self.jit.close()
