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
                 config_path="acestep-v15-turbo", pin_s=4.0, prime_s=2.0):
        self.jit = JITCover(device=device, steps=steps, config_path=config_path)
        self.window_s = window_s
        self.lookahead_samp = int(round(lookahead_s * SR))   # ring counts AUDIO SAMPLES
        self.SL = max(1, int(round(slice_s * FPS)))          # slice length in LATENT frames
        # PINNED-PREFIX ROLLING (true continuity, no seams). We keep one continuous
        # "committed" latent and extend it in chunks: each new chunk's first _pin_f
        # frames are PINNED (inpainting) to the committed tail, so the model denoises
        # the new frames to CONTINUE the real past — the overlap is identical, no
        # crossfade. window_s = the generation window; _pin_f = pinned context (fixed),
        # _hop_f = new frames per chunk (= window − pin). Smaller window ⇒ snappier
        # control + cheaper; larger ⇒ more new-content context per chunk.
        self._pin_f = max(1, int(round(pin_s * FPS)))                       # pinned context frames
        self._hop_f = max(1, int(round(window_s * FPS)) - self._pin_f)      # new frames per chunk
        self._prime_f = max(1, int(round(prime_s * FPS)))                   # cold-prefix length for the 2-pass first chunk
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

    def set_style(self, tags, denoise=None, character=None, timbre=None, send_bpm=None, send_key=None,
                  bpm=None, key=None):
        self.jit.set_style(tags, denoise=denoise if denoise is not None else self.denoise,
                           character=character, timbre=timbre, send_bpm=send_bpm, send_key=send_key,
                           bpm=bpm, key=key)
        return self

    def set_metas(self, send_bpm=None, send_key=None, bpm=None, key=None):
        with self._lock:
            self._ctrl.append(("metas", (send_bpm, send_key, bpm, key)))

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
            self._hop_f = max(1, int(round(self.window_s * FPS)) - self._pin_f)  # new frames/chunk
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
        """Pull n frames -> np.float32 [n,4] = [coverL,coverR, origL,origR]; zero-fill
        (count underrun) if short. The original pair is the raw source for the same
        frames, so the client can A/B cover vs source instantly (it just picks a pair)."""
        out = np.zeros((n, 4), dtype=np.float32)
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
    @property
    def running(self):
        """True while the producer thread is active (play OR pause — pause stops the
        sender/consumer, NOT the producer, so playback position is kept)."""
        return self._running

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._produce, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    def reset(self):
        """Full stop: stop the producer AND reset playback to the start (clears the
        ring, counters, and playhead origin) so the next play() begins from 0."""
        self.stop()
        with self._lock:
            self._ring.clear(); self._ring_frames = 0
            self._produced_f = 0; self._consumed_f = 0; self._real_f = 0
            self._seek_pos_s = 0.0

    def _push(self, cover_2xN, orig_2xN):
        cov = cover_2xN.detach().t().contiguous().numpy().astype(np.float32)  # [N,2]
        org = orig_2xN.detach().t().contiguous().numpy().astype(np.float32)   # [N,2]
        chunk = np.concatenate([cov, org], axis=1)                           # [N,4] cover+original
        with self._lock:
            self._ring.append(chunk)
            self._ring_frames += chunk.shape[0]
            self._produced_f += chunk.shape[0]

    def _produce(self):
        """Pinned-prefix ROLLING producer. Maintain one continuous `committed` latent;
        extend it in chunks where each chunk's first _pin_f frames are pinned to the
        committed tail (jit._gen(pin=...)) so the new frames continue the real past —
        seamless. Decode the committed latent with the existing overlap-discard tiler
        (lazily, trailing the frontier by the VAE margin so every decoded tile has full
        context) and push. No crossfade — continuity is by construction."""
        torch.set_grad_enabled(False)   # grad flag is THREAD-LOCAL; disable in this worker
        from acestep.nodes.types import Latent
        jit = self.jit
        C = max(1, min(self._pin_f, self.full_T // 2))   # pinned context frames
        H = max(1, self._hop_f)                           # new frames per chunk
        headroom = jit._TILE + jit._TOV                   # extra committed needed for full-context decode tiles
        committed = None                                  # [1,n,D] continuous clean latent (model dtype)
        base_f = 0                                        # source frame of committed[0]
        gen_f = 0; dec_f = 0                              # source frames: generated-to / decoded-pushed-to
        tiles: dict = {}                                  # decode cache (tile idx local to base_f -> audio)
        while self._running:
            if dec_f >= self.full_T:                      # loop the track (hard restart at the loop point)
                committed = None; base_f = gen_f = dec_f = 0; tiles = {}
            with self._lock:
                ctrls = list(self._ctrl); self._ctrl.clear()
                ahead = self._produced_f - self._consumed_f
            restyled = False
            for kind, val in ctrls:
                if kind == "prompt":      jit._set_prompt(val); restyled = True
                elif kind == "denoise":   jit.denoise = val; restyled = True
                elif kind == "character": jit.set_character(val); restyled = True
                elif kind == "metas":     jit.set_metas(send_bpm=val[0], send_key=val[1], bpm=val[2], key=val[3]); restyled = True
                elif kind == "dcw":       jit.set_dcw(enabled=val); restyled = True
                elif kind == "seek":
                    committed = None; base_f = gen_f = dec_f = int(val); tiles = {}
                    with self._lock:      # flush stale buffered audio; re-prime from the seek point
                        self._ring.clear(); self._ring_frames = 0
                        self._produced_f = 0; self._consumed_f = 0; self._real_f = 0
                        self._seek_pos_s = val / FPS
            # A re-style change only affects FUTURE chunks; the latent already generated
            # ahead (up to one hop, old settings) would otherwise play out first -> the
            # change "takes a long time", variably (depends where in the gen cycle it lands).
            # Discard that un-decoded ahead and regenerate from the decode point with the new
            # settings, PINNED to the decoded past (smooth morph). Latency then = the decoded
            # ring depth (~buffer), not a hop; no gap (the ring covers the regen).
            if restyled and committed is not None and gen_f > dec_f and (dec_f - base_f) >= C:
                committed = committed[:, :dec_f - base_f, :].contiguous()
                gen_f = dec_f
                tcut = (dec_f - base_f) // jit._TILE          # drop decoded-ahead tiles at/after dec_f
                tiles = {t: v for t, v in tiles.items() if t < tcut}
            # adaptive buffer: keep the ring ahead by >= one worst-case chunk (auto-grows
            # to the measured worst step). committed is None right after seek/start/loop
            # -> never wait, produce immediately.
            target = max(self.lookahead_samp, int(min(8.0, self.max_step_ms / 1000 + 0.4) * SR))
            if committed is not None and ahead >= target + self.SL * SPF:
                time.sleep(0.003)
                continue
            t_iter = time.perf_counter()
            dec_target = min(dec_f + self.SL, self.full_T)
            # 1) extend the committed latent (pinned roll) until we can decode the slice
            #    with full VAE context (or we hit the track end).
            while committed is None or (gen_f < dec_target + headroom and gen_f < self.full_T):
                if committed is None:
                    # First chunk (also after seek/loop) has no styled predecessor, so a
                    # cold cover leans weakly on the source ("style hasn't taken hold yet").
                    # 2-pass fix: a quick cold pass yields a short prefix to PIN, then we
                    # regenerate in continuation mode so the body gets the same style boost
                    # every rolling chunk gets. Only the first ~prime_s stays cold.
                    w1 = min(base_f + C + H, self.full_T)
                    cold = jit._gen(base_f, min(base_f + self._prime_f + C, w1), self.seed).tensor
                    cp = min(self._prime_f, cold.shape[1] - 1)
                    committed = jit._gen(base_f, w1, self.seed, pin=cold[:, :cp, :]).tensor
                    gen_f = base_f + committed.shape[1]
                else:
                    pin = committed[:, -C:, :]
                    new = jit._gen(gen_f - C, min(gen_f + H, self.full_T), self.seed, pin=pin).tensor[:, C:, :]
                    committed = torch.cat([committed, new], dim=1)
                    gen_f += new.shape[1]
                self.regens += 1
            # 2) decode + push the next slice from the continuous committed latent
            seg = jit._ensure_tiles(Latent(tensor=committed), tiles, dec_f - base_f, dec_target - base_f)
            mps_compat.mps_sync()
            if jit.dcw_enabled: seg = seg * self._dcw_gain   # make-up for DCW's level boost
            orig = jit.source_slice(dec_f, seg.shape[-1])    # raw source, same frames -> instant A/B
            self._push(seg, orig)
            dec_f = dec_target
            self.max_step_ms = max(self.max_step_ms, (time.perf_counter() - t_iter) * 1000)
            keep = (dec_f - base_f) // jit._TILE - 1          # prune decoded tiles behind the playhead
            for t in [t for t in tiles if t < keep]:
                del tiles[t]
            mps_compat.reclaim()
        self._done = True

    @property
    def done(self):
        return self._done

    def close(self):
        self.stop()
        self.jit.close()
