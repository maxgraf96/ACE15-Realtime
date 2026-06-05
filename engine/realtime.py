"""Real-time producer/consumer cover engine (Phase 2 core).

Wraps the JIT cover logic in a background PRODUCER thread that keeps a PCM ring
filled ~lookahead seconds ahead of a CONSUMER (the audio callback / sidecar
socket) draining at 1x. This is the architecture the plugin needs: the audio
thread never blocks on the model; it just pulls ready PCM. Live controls are
queued and applied on the next produced slice -> latency ~= the buffered amount
(~lookahead). Validates sustained real-time headlessly before any JUCE work.
"""
from __future__ import annotations

import os
import threading
import time
from collections import deque

import numpy as np

from . import mps_compat
from .jit import JITCover, FPS, SR, SPF, SEM_PATCH, _eq

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
        self.live = False              # real-time live-input mode (source grows from a stream)
        self.beats_per_bar = 4         # tempo grid (4/4); bar = 60/bpm * beats_per_bar
        self.live_pin_bars = 1         # live window = (pin + hop) bars; bigger = more context, more latency
        self.live_hop_bars = 1
        # Live warble fix = TWO layers: (1) seamless windowed source re-encode
        # (jit.enc_lat_mode="window") kills the chunk-seam perturbation; (2) the pinned roll
        # is BISTABLE (bf16 nondeterminism tips it into runaway warble ~half the time), so we
        # FULLY re-anchor every N bars: cold-prime a complete fresh window from source so the
        # pin resets to 100% source-derived (like file mode's per-loop restart). Enabled by
        # holding the output a full window (~2 bars) behind the input. 4 bars resets often
        # enough that drift never accumulates between resets, at half the re-anchor cost of 2.
        self.live_anchor_bars = 4      # 0 = off
        # LOOP-COVER mode: if the live input is a loop, auto-detect it and cover it like a
        # FILE (clean) streamed phase-locked (low latency) — the non-causal model's missing
        # "future" is supplied by the loop itself. Falls back to continuous mode when no loop.
        self.loop_enable = True
        self.loop_min_conf = 0.90      # autocorrelation confidence to accept a loop
        self.loop_listen_s = 28.0      # stay SILENT (AI "listens", you hear your dry input) up to this
                                       # long while detecting+rendering the loop -> no warble before lock;
                                       # if no loop appears by then, fall back to continuous (non-loop input)
        self.loop_bars_hint = 0        # >0 = MANUAL loop length (bars): skip auto-detect, lock after 1 loop
        self.loop_locked = False       # telemetry: currently streaming a locked loop cover
        self.loop_bars = 0
        self.loop_lead_s = 0.0         # play the cover this far AHEAD of the input phase (compensate the
                                       # input+output round-trip latency so the AI isn't audibly behind)
        self.restyling = False         # telemetry: re-rendering the loop cover (Style/loop change) in the bg
        self._pending_cover = None     # bg re-render result handed back to the producer
        self._rendering = False        # a bg re-render is in flight
        self._restyle_pending = False  # a new Style/loop change arrived during a bg re-render
        self._restem = False           # stem mute/solo changed -> re-separate the locked loop cover
        self._pending_stems = None     # bg stem re-separation result handed back to the producer
        # Ableton Link (set by the sidecar to a LinkSync; None = Link off). When a peer is present we
        # place the loop cover on Link's BEAT GRID (the shared Ableton clock) instead of the input
        # onset/fed-sample phase -> drift-free against the DAW. Falls back to onset phase if no peer.
        self.link = None
        self.link_active = False       # telemetry: currently placing the cover on the Link grid
        self._link_anchor = None       # (link_beat0, cover_phase0, loop_P, loop_bars) alignment snapshot
        self._bar_trail_s = 0.0        # achieved output trailing (whole bars) in live mode
        self._running = False
        self._done = False
        self._thread = None
        # telemetry
        self.regens = 0
        self.reanchors = 0             # live drift-reset re-anchors performed
        self.anchor_skips = 0          # re-anchor was due but source/zone too small (gate blocked)
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

    def set_input_gain(self, db):
        with self._lock:
            self._ctrl.append(("input_gain", float(db)))

    def set_stems(self, stems):
        """OUTPUT stem mixer: keep only these stems (e.g. ['drums']) of the generated cover."""
        if self._running:
            with self._lock:
                self._ctrl.append(("stems", stems))
        else:
            self.jit.set_stems(stems)
            if stems:
                self.jit._ensure_sep()   # pre-load Demucs now (producer not running) — avoid a mid-stream stall

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
                "progress": pos / dur, "duration_s": round(dur, 1),
                "loop_locked": bool(self.loop_locked), "loop_bars": int(self.loop_bars),
                "restyling": bool(self.restyling), "link_active": bool(self.link_active)}

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

    # ---- live / streaming input (Phase 3) ----
    def begin_live(self):
        """Switch to real-time live-input mode: the source latent grows from feed_input()
        instead of a preloaded track. Call before set_style()/start()."""
        self.live = True
        self.loop_locked = False; self.loop_bars = 0; self.restyling = False
        self._pending_cover = None; self._rendering = False; self._restyle_pending = False
        self._restem = False; self._pending_stems = None
        with self._lock:
            self._ctrl.clear()       # drop stale queued ctrls so a kept-alive engine restarts clean
        self.jit.begin_live()
        return self

    def _refine_period(self, P0, frac=0.08):
        """Measure the loop period to sample precision near P0 (the bars*60/bpm estimate). The DAW's
        real loop length almost never equals bars*60/bpm exactly (file/region length, tempo a hair
        off), and that tiny mismatch makes the cover DRIFT against the input over many loops. Returns
        the normalized-autocorrelation peak lag within +/-frac of P0 (falls back to P0 if too little
        captured). Run from the producer thread (cheap CPU FFT)."""
        md = max(1, int(P0 * frac))
        with self.jit._live_lock:
            n = self.jit._live_raw.shape[1]
            take = min(n, 2 * P0 + md)
            if take < P0 + md + P0 // 4:                  # need ~1.3 loops for a reliable peak
                return P0
            mono = self.jit._live_raw[:, -take:].mean(0).cpu().numpy().astype(np.float64)
        mono -= mono.mean()
        L = 1
        while L < 2 * len(mono):
            L *= 2
        f = np.fft.rfft(mono, L)
        ac = np.fft.irfft(f * np.conj(f), L)[:len(mono)]
        ac = ac / np.maximum(np.arange(len(mono), 0, -1), 1)   # normalize by overlap (de-bias the decay)
        lo, hi = max(1, P0 - md), min(len(mono) - 1, P0 + md)
        if hi <= lo:
            return P0
        return int(np.argmax(ac[lo:hi + 1])) + lo

    def _apply_stems(self, full_np):
        """OUTPUT stem mixer for the locked loop cover [P,2]: if a stem SUBSET is selected, separate
        (Demucs) and keep only those, else return the full mix. Tiled x3 (take the middle) so the
        loop boundary keeps full context -> seamless. Cheap (~RTF 0.01); MPS, so the caller runs it
        only when no background render is in flight (avoid concurrent GPU use)."""
        from .separation import STEMS
        st = self.jit.stems
        if not st or set(st) >= set(STEMS):
            return full_np
        P = full_np.shape[0]
        t = torch.from_numpy(np.concatenate([full_np, full_np, full_np], 0).T).contiguous()  # [2,3P]
        mix = self.jit._ensure_sep().separate(t, st)
        return mix[:, P:2 * P].detach().t().contiguous().cpu().float().numpy()                 # [P,2]

    def set_loop_bars(self, n):
        """Manual loop length in bars (0 = auto-detect). Queued; applied by the producer."""
        with self._lock:
            self._ctrl.append(("loop_bars", float(n)))

    def set_loop_lead(self, ms):
        """Nudge the loop cover by this many ms (SIGNED) against the input phase: +ms plays it
        EARLIER (cancel the input+output round-trip lag so the AI isn't behind), -ms plays it
        LATER (pull it back if it lands ahead). Queued; applied by the producer."""
        with self._lock:
            self._ctrl.append(("loop_lead", float(ms)))

    def feed_input(self, pcm_48k):
        """Append live 48 kHz input (any thread)."""
        self.jit.feed_live(pcm_48k)

    def _render_loop_cover(self, start_a, period_s, seed):
        """Render a clean cover of the loop [start_a, start_a+period_s), for LOOP-COVER mode.
        `start_a` is ONSET-ALIGNED (onset + k*period), so cover frame i maps to the loop's
        onset-phase i -> the cover plays phase-locked to the input, no cross-correlation. The
        non-causal model needs future context the live edge lacks, AND it would generate a song-
        ENDING (decrescendo) at the source's end. Both are solved by TILING the loop x3 and taking
        the MIDDLE copy: it has loop content before AND after, so the model neither winds down nor
        starts weak. RESAMPLE to EXACTLY period_s so it tiles forever without drift. One-shot.
        Returns (cover_pcm [period_s,2], orig_pcm [period_s,2]) or None."""
        jit = self.jit
        self._n_renders = getattr(self, "_n_renders", 0) + 1   # telemetry: total loop-cover renders
        def _rlog(msg):                                        # step-by-step render memory probe (opt-in)
            if os.environ.get("ACE15_LIVE_LOG") != "1":
                return
            try:
                _m = getattr(torch, "mps", None)
                _c = getattr(_m, "current_allocated_memory", None)
                _d = getattr(_m, "driver_allocated_memory", None)
                with open("/tmp/ace15_live.log", "a") as _f:
                    _f.write(f"  RENDER[{self._n_renders}] {msg} "
                             f"cur={(_c()/1e6 if _c else -1):.0f}MB drv={(_d()/1e6 if _d else -1):.0f}MB\n")
            except Exception:
                pass
        import torchaudio.functional as AF
        from acestep.nodes.types import Audio, Latent
        from acestep.engine.session import PreparedSource
        with jit._live_lock:
            if start_a + period_s > jit._live_raw.shape[1]:
                return None
            loop = jit._live_raw[:, start_a: start_a + period_s].contiguous()   # one onset-aligned loop
        _rlog(f"START period_s={period_s} ({period_s/SR:.2f}s) bpm={jit.bpm} bpb={self.beats_per_bar} steps={jit.steps}")
        raw3 = torch.cat([loop, loop, loop], 1)                # tile x3 (the loop repeats -> seamless joins)
        lat = jit.session.encode_audio(Audio(waveform=raw3, sample_rate=SR)).tensor
        Pf = (lat.shape[1] // 3 // SEM_PATCH) * SEM_PATCH      # encoded frames per loop (multiple of 5)
        _rlog(f"after encode raw3={tuple(raw3.shape)} lat={tuple(lat.shape)} Pf={Pf}")
        if Pf < 5:
            return None
        lat = lat[:, :3 * Pf, :].contiguous()
        ctx = jit.session.extract_hints(Latent(tensor=lat))
        _rlog("after extract_hints")
        saved_src, saved_handle, saved_tile = jit.source, jit.handle, jit._TILE
        try:
            jit.source = PreparedSource(latent=Latent(tensor=lat), context_latent=ctx)
            jit._encode()                                      # rebuild cond w/ the latest Style AND a valid
            #            source for _refer() (Character timbre ref) — must run with jit.source set, on THIS
            #            (background) thread so it's serialized with the render's other GPU work.
            _rlog("after _encode")
            jit.handle = jit.session.stream(source=jit.source, conditioning=jit.cond, steps=jit.steps,
                                            shift=jit.shift, pipeline_depth=1, **jit._dcw_kwargs())
            jit._TILE = 48                                      # full-quality decode for the one-shot render
            _win = min(20.0, 3 * Pf / FPS)
            _rlog(f"before render window_s={_win:.2f}")
            cover, _ = jit.render(window_s=_win, lookahead_s=1.0,
                                  slice_s=1.0, xfade_s=0.12, seed=seed)
            _rlog(f"after render cover={tuple(cover.shape)}")
        finally:
            # CLOSE the streaming handle this render created — else every render leaks its handle's
            # MPS buffers (KV cache etc.: ~hundreds of MB on 2B, GBs on XL). Repeated locks/re-renders
            # then climb to tens of GB -> swap -> beachball -> OOM crash. THIS was the runaway leak.
            if jit.handle is not None and jit.handle is not saved_handle:
                try: jit.handle.close()
                except Exception: pass
            jit.source, jit.handle, jit._TILE = saved_src, saved_handle, saved_tile
        if jit.dcw_enabled:
            cover = cover * self._dcw_gain
        cov = cover[:, Pf * SPF: 2 * Pf * SPF].contiguous()    # the MIDDLE loop (full context, no ending/weak start)
        if cov.shape[1] != period_s:                           # lock the loop length to the EXACT period.
            # NB: do NOT use torchaudio.functional.resample here — its sinc kernel is ~orig/gcd(orig,new)
            # taps, and (cov_len=Pf*SPF) vs (period_s from _refine_period) are large + ~coprime, so the
            # kernel explodes to ~350k taps -> tens of GB -> the producer thread hangs -> 96 GB -> crash.
            # (Synthetic test loops are exact multiples so gcd is huge and the kernel was ~1 tap, hiding
            # this.) The ratio is ~1.0 (length-locking, not pitch), so kernel-free linear interpolation is
            # sonically identical and O(n). This is the loop-cover twin of the streaming "metallic" fix.
            cov = torch.nn.functional.interpolate(
                cov.float().unsqueeze(0), size=int(period_s), mode="linear", align_corners=False).squeeze(0)
        cov = cov[:, :period_s]
        if cov.shape[1] < period_s:                            # pad a hair if resample rounded short
            cov = torch.cat([cov, cov[:, :period_s - cov.shape[1]]], 1)
        cov = cov.detach().t().contiguous().cpu().float().numpy()                  # [period_s, 2]
        org = loop[:, :period_s].detach().t().contiguous().cpu().float().numpy()   # the onset-aligned input loop
        return cov, org

    def _cover_phase(self, fed_s, onset, loop_P, ahead_samp):
        """Sample-phase into the loop cover for the NEXT push. `ahead_samp` = ring + lead frames, i.e.
        look ahead to the moment this audio will actually PLAY.

        LINK GRID (an Ableton Link peer is present): the cover is `loop_bars` bars long, so one beat =
        loop_P / (loop_bars * beats_per_bar) samples. We snapshot the current cover phase against the
        current Link beat ONCE (so the cover stays where the input had it), then advance phase purely
        from Link's beat -> the cover rides Ableton's shared clock and never drifts against the DAW,
        regardless of input-stream jitter.

        FALLBACK (no peer): the original onset/fed-sample phase."""
        if loop_P <= 0:
            self.link_active = False
            return 0
        link = self.link
        if link is not None and getattr(link, "connected", False):
            b = link.beat
            if b is not None:
                bars = max(1, int(round(self.loop_bars))) if self.loop_bars else 1
                spb = loop_P / (bars * self.beats_per_bar)             # samples per beat
                tempo = link.tempo or 120.0
                if self._link_anchor is None or self._link_anchor[2] != loop_P:
                    ph0 = ((fed_s - onset) % loop_P) if onset >= 0 else 0   # align cover-now to Link-now
                    self._link_anchor = (b, ph0, loop_P, bars)
                b0, ph0, _lp, _bars = self._link_anchor
                fut = b + (ahead_samp / SR) * (tempo / 60.0)          # Link beat at the PLAY moment
                self.link_active = True
                return int(round(ph0 + (fut - b0) * spb)) % loop_P
        self.link_active = False
        self._link_anchor = None
        return ((fed_s - onset + ahead_samp) % loop_P) if onset >= 0 else 0

    def start(self):
        self._running = True
        target = self._produce_live if self.live else self._produce
        self._thread = threading.Thread(target=target, daemon=True)
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

    def _push_np(self, cov, org):
        """Append pre-rendered loop PCM: cov/org are [n,2] float32 (cover, original)."""
        chunk = np.concatenate([cov, org], axis=1).astype(np.float32)   # [n,4]
        with self._lock:
            self._ring.append(chunk)
            self._ring_frames += chunk.shape[0]
            self._produced_f += chunk.shape[0]

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
                elif kind == "input_gain": jit.set_input_gain(val); restyled = True   # re-encode source (heavy)
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
            seg = jit.decode_out(Latent(tensor=committed), tiles, dec_f - base_f, dec_target - base_f)
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

    def _produce_live(self):
        """Live producer: same pinned-prefix rolling cover, but the source latent GROWS
        from the incoming stream (jit.encode_pending) and the frontier is GATED by the
        available input — we never generate/decode past what's been captured+encoded. No
        loop, no seek. Output trails the live input by ~the window + decode headroom."""
        torch.set_grad_enabled(False)
        import math
        from acestep.nodes.types import Latent
        jit = self.jit
        # Bar grid from the (manual/host) tempo: generate in whole-bar hops, and below we
        # round the output up to a whole bar so the cover trails the input by N bars exactly.
        bpm = float(getattr(jit, "bpm", 120) or 120); bpm = min(220.0, max(40.0, bpm))
        bar_frames = max(1, round(60.0 / bpm * self.beats_per_bar * FPS))
        self._pin_f = bar_frames * self.live_pin_bars   # bar-aligned window = pin + hop bars
        self._hop_f = bar_frames * self.live_hop_bars
        bar_s = bar_frames / FPS
        C = max(1, self._pin_f); H = max(1, self._hop_f)
        headroom = jit._TILE + jit._TOV
        anchor_period = bar_frames * self.live_anchor_bars   # 0 = never re-anchor
        reanchor_win = C + H   # full re-anchor window = the source lead we hold ahead of the decode
                               # point so a whole fresh window always fits -> full (pure-fresh) reset
        committed = None; base_f = 0; gen_f = 0; dec_f = 0; tiles = {}
        anchor_f = 0
        aligned = False
        # loop-cover state. Phase is anchored to jit._onset (the loop's hard start = first audible
        # input sample), so it's deterministic + stable — no cross-correlation, no re-align jitter.
        loop_cov = None; loop_full = None; loop_org = None; loop_P = 0   # loop_full = full mix; loop_cov = stem-mixed
        last_detect = -10 ** 9
        last_lock_try = -10 ** 9                 # wall-clock of the last lock ATTEMPT (rate limit, see below)
        detect_every = int(bar_s * SR)          # re-check for a loop ~once per bar
        give_up = not self.loop_enable          # no loop found in time -> continuous fallback
        cont_started = False                    # one-time start-point reset when continuous takes over
        self.loop_locked = False

        def _exact_period(bars):
            return int(round(bars * (60.0 / bpm) * self.beats_per_bar * SR))   # exact sample period

        def _aligned_start(onset, P):
            """Start of the most recent COMPLETE onset-aligned loop [a, a+P] within the buffer."""
            m = (jit._live_raw.shape[1] - onset) // P - 1
            return onset + max(0, m) * P

        def _lock_loop(start_a, P, B):
            """Render the onset-aligned loop cover once (blocking; the ring covers it). Returns
            (cov, org, P) or None. Cover frame i maps to onset-phase i -> deterministic playback."""
            res = self._render_loop_cover(start_a, P, self.seed)
            if not res:
                return None
            self.loop_bars = B
            return res[0], res[1], P

        def _spawn_rerender(start_a, P, B):
            """Re-render the (onset-aligned) loop cover in a BACKGROUND thread (Style/loop change)
            while the producer keeps streaming the OLD cover -> no audio dropout, just a 'restyling'
            status. Safe: in loop mode the producer does no MPS, so the render owns the GPU."""
            if self._rendering:
                self._restyle_pending = True       # coalesce: render again once this one finishes
                return
            self._rendering = True; self.restyling = True
            def work():
                torch.set_grad_enabled(False)          # grad flag is THREAD-LOCAL; disable in this worker
                try:
                    res = self._render_loop_cover(start_a, P, self.seed)   # re-encodes (Style) internally
                    if res:
                        cov = self._apply_stems(res[0])      # apply current stems in the BG (no main stall)
                        self._pending_cover = (res[0], cov, res[1], P, B)
                except Exception:
                    pass
                finally:
                    self._rendering = False
            threading.Thread(target=work, daemon=True).start()

        def _spawn_restem(full):
            """Re-separate the locked cover for new stem mute/solo in a BACKGROUND thread (no producer
            stall) while the old cover keeps streaming. Shares `_rendering` as the GPU mutex so it never
            runs concurrently with a re-render. On completion -> _pending_stems (swapped + flushed)."""
            if self._rendering or full is None:
                return                                 # GPU busy / not locked yet -> retry next tick
            self._restem = False
            self._rendering = True
            def work():
                torch.set_grad_enabled(False)
                try:
                    self._pending_stems = self._apply_stems(full)
                except Exception:
                    pass
                finally:
                    self._rendering = False
            threading.Thread(target=work, daemon=True).start()

        def _loop_loud(period_s):
            """Is the recent loop (period_s) loud overall? Guards the lock against silence / not-yet-
            started audio. Uses the WHOLE-window RMS (not per-bar) so a quiet bar within a real loop
            doesn't block the lock forever — sitting unlocked leaks memory until it OOM-crashes."""
            bs = bar_frames * SPF
            n = min(jit._live_raw.shape[1], max(period_s, bs))
            if n < bs:
                return False
            with jit._live_lock:
                seg = jit._live_raw[:, -n:].float()
            return float(seg.pow(2).mean().sqrt()) > 1.5e-3       # ~ -56 dB overall
        _mps_empty = getattr(getattr(torch, "mps", None), "empty_cache", None)   # release cached MPS blocks
        _last_gc = time.time()
        while self._running:
            # Periodically release the MPS allocator cache. Live generation (esp. the continuous
            # fallback) allocates large transient tensors every tick; MPS caches the freed blocks and
            # the footprint climbs UNBOUNDED (observed ~25 MB/s on 2B, GB/s on XL -> 80+ GB -> the OS
            # swaps -> beachball -> OOM crash). empty_cache() every ~2 s caps it. Cheap relative to a
            # model step; the ~2 s cadence keeps any latency blip well under the buffer.
            if _mps_empty is not None and time.time() - _last_gc > 2.0:
                _last_gc = time.time()
                try: _mps_empty()
                except Exception: pass
                # lightweight live-mode telemetry (one line / 2 s; negligible). Reveals whether the
                # engine actually LOCKS and whether mps_cur is flat — set ACE15_LIVE_LOG=0 to silence.
                if os.environ.get("ACE15_LIVE_LOG") == "1":      # opt-in live-mode diagnostics
                    try:
                        _mps = getattr(torch, "mps", None)
                        _cf = getattr(_mps, "current_allocated_memory", None)
                        _df = getattr(_mps, "driver_allocated_memory", None)
                        _cur = (_cf() / 1e6) if _cf else -1
                        _drv = (_df() / 1e6) if _df else -1
                        import resource
                        _rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6   # bytes on macOS -> MB
                        with open("/tmp/ace15_live.log", "a") as _lf:
                            _lf.write(f"mps_cur={_cur:.0f}MB mps_drv={_drv:.0f}MB rss={_rss:.0f}MB "
                                      f"locked={self.loop_locked} onset={self.jit._onset} "
                                      f"cond={'Y' if self.jit.cond is not None else 'N'} "
                                      f"renders={getattr(self, '_n_renders', 0)} "
                                      f"bars_hint={self.loop_bars_hint} raw_s={self.jit._live_raw.shape[1]/SR:.1f}\n")
                    except Exception:
                        pass
            with self._lock:
                ctrls = list(self._ctrl); self._ctrl.clear()
                ahead = self._produced_f - self._consumed_f
            restyled = False; need_encode = False
            # In LOOP mode, DEFER the (MPS) conditioning re-encode to the background re-render
            # thread. Running it here on the producer thread races the bg render's MPS (two
            # threads submitting GPU kernels) -> the engine hangs/crashes (this is what stopped
            # playback when the Amount/Character sliders were dragged). Continuous/file mode has
            # no bg render, so encode immediately. dcw/stems/gain don't re-encode.
            in_loop = loop_cov is not None
            for kind, val in ctrls:
                if kind == "prompt":      jit.tags = val; restyled = need_encode = True
                elif kind == "denoise":   jit.denoise = float(val); restyled = True
                elif kind == "character": jit.character = float(val); restyled = need_encode = True
                elif kind == "metas":
                    jit._set_bpm_key(val[2], val[3])
                    if val[0] is not None: jit.send_bpm = bool(val[0])
                    if val[1] is not None: jit.send_key = bool(val[1])
                    restyled = need_encode = True
                elif kind == "dcw":       jit.set_dcw(enabled=val); restyled = True
                elif kind == "input_gain": jit.input_gain_db = float(val)   # live: applied on feed, no re-encode
                elif kind == "stems":     jit.set_stems(val); self._restem = True   # re-separate the locked cover
                elif kind == "loop_bars": self.loop_bars_hint = max(0.0, float(val)); restyled = True
                elif kind == "loop_lead": self.loop_lead_s = float(val) / 1000.0   # ms -> s, signed (no re-render);
                #                          +ms = play the cover EARLIER (cancel round-trip lag), -ms = LATER
            if need_encode and not in_loop:
                jit._encode()                                  # continuous/file: apply the new style now
            fed_s = jit._live_raw.shape[1]
            # ===== LOOP-COVER MODE: stream the locked loop cover, phase-synced (low latency).
            # Re-render (brief stall, ring covers it) on a Style change or a loop change. =====
            if self.loop_enable and loop_cov is not None:
                onset = jit._onset
                # swap in a finished background re-render (glitch-free: old cover streamed until now).
                # full mix + stem-applied cover were both computed in the bg -> no producer-thread Demucs.
                if self._pending_cover is not None:
                    full, cov, org, P, B = self._pending_cover; self._pending_cover = None
                    loop_full, loop_cov, loop_org, loop_P, self.loop_bars = full, cov, org, P, B
                    self.restyling = False
                # stem mute/solo: when the BACKGROUND re-separation finishes, swap it in and DROP the
                # buffered old-stem audio so the toggle is heard near-instantly (phase stays aligned).
                if self._pending_stems is not None:
                    loop_cov = self._pending_stems; self._pending_stems = None
                    with self._lock:
                        self._ring.clear(); self._ring_frames = 0
                        self._produced_f = self._consumed_f                # ahead->0, ring empty (phase: rd=0)
                    ahead = 0                        # refill with the new stems THIS iteration (no gap)
                # kick off the bg re-separation on a stem change (re-uses the SAME cover; no re-gen)
                if self._restem:
                    _spawn_restem(loop_full)
                # re-render ONLY on an explicit Style / loop-bars change. The old auto-detect re-render
                # (detect_loop every bar -> re-render when the period drifted >4%) fired SPURIOUSLY on
                # any noisy/varying input, and each re-render retains ~300 MB (GBs on XL) -> unbounded
                # growth -> OOM crash. A locked loop now stays locked; if you change the loop, re-lock
                # with Stop/Play. Runs in the BACKGROUND; the old cover keeps streaming meanwhile.
                redo = restyled
                if redo or (self._restyle_pending and not self._rendering):
                    if not redo:
                        self._restyle_pending = False                     # consume the queued change
                    Pn, Bn = loop_P, self.loop_bars                        # SAME loop, reuse the locked period
                    if onset >= 0 and (jit._live_raw.shape[1] - onset) >= Pn:
                        _spawn_rerender(_aligned_start(onset, Pn), Pn, Bn)
                # stream PHASE-LOCKED. Default: anchor to the input ONSET (cover frame i == onset-phase i).
                # When Ableton Link has a peer, anchor to LINK'S BEAT GRID instead: snapshot the current
                # (input) cover phase against the current Link beat once, then advance phase purely from
                # Link's beat -> the cover rides the DAW's shared clock with no input-vs-DAW drift. Both
                # compensate the ring (rd) + round-trip latency (lead) by looking ahead to the PLAY moment.
                tgt = int(2.0 * SR); lead = int(self.loop_lead_s * SR)
                if ahead < tgt + self.SL * SPF:
                    with self._lock:
                        rd = self._ring_frames
                    ph = self._cover_phase(fed_s, onset, loop_P, rd + lead)
                    n = min(self.SL * SPF, loop_P - ph)
                    self._push_np(loop_cov[ph:ph + n], loop_org[ph:ph + n])
                else:
                    time.sleep(0.003)
                continue
            # ===== NOT LOCKED: try to lock a loop; stay SILENT ("listening", you hear your dry
            # input) meanwhile so there's NO warble before the lock. Fall back to continuous only
            # if no loop ever appears (non-looped live input). =====
            if not give_up:
                locked = None; onset = jit._onset
                # Anchor to the ONSET (loop hard start). Lock once >= 2 onset-aligned loops are
                # captured AND loud (clean, onset-aligned cover; never onto silence/partial silence).
                #
                # RATE-LIMIT the lock ATTEMPT. A normal input locks on the FIRST attempt (then loop_cov
                # is set and we never re-enter this branch), so this never delays a good lock. It only
                # bounds the COST when a render keeps FAILING on some input: without it the bars-hint
                # path rendered EVERY tick (20-40x/s), each render retaining MPS (GBs on XL) -> 60+ GB
                # in seconds -> beachball -> OOM (this is the user's "stuck listening + memory blows up"
                # symptom). With the gate a failing lock retries at most ~once/bar and the transient is
                # reclaimed by empty_cache(). The try/except keeps a throwing render from killing the
                # producer thread (which would silently stop all audio).
                # The gate only blocks the expensive RENDER (_lock_loop); the cheap condition checks run
                # every tick, and `last_lock_try` is consumed only when we actually fire a render — so a
                # good input still locks the instant it's ready (no added latency).
                now = time.time()
                gate_ok = (now - last_lock_try) >= max(1.0, bar_s)
                if onset >= 0 and jit.cond is not None and gate_ok:
                    try:
                        if self.loop_bars_hint > 0:                 # known length -> lock after ~1.3 loops
                            Bn = int(round(self.loop_bars_hint)); P0 = _exact_period(Bn)
                            if P0 > 0 and (fed_s - onset) >= (P0 * 4) // 3 and _loop_loud(P0):
                                last_lock_try = now                 # consumed only on a real render attempt
                                P = self._refine_period(P0)         # MEASURE the true period (no drift)
                                locked = _lock_loop(_aligned_start(onset, P), P, Bn)
                        elif fed_s - last_detect >= detect_every:   # auto: detect_loop already needs ~2 loops
                            last_detect = fed_s
                            Pd, cd, Bd = jit.detect_loop(thresh=self.loop_min_conf)
                            if Pd and (fed_s - onset) >= (Pd * 4) // 3 and _loop_loud(Pd):
                                last_lock_try = now
                                P = self._refine_period(Pd)
                                locked = _lock_loop(_aligned_start(onset, P), P, Bd)
                    except Exception as _e:
                        import traceback
                        try:
                            with open("/tmp/ace15_live.log", "a") as _lf:
                                _lf.write(f"LOCK FAIL: {_e}\n{traceback.format_exc()}\n")
                        except Exception:
                            pass
                        if _mps_empty is not None:
                            try: _mps_empty()             # reclaim the failed render's transient at once
                            except Exception: pass
                if locked:
                    loop_full, loop_org, loop_P = locked       # full mix
                    loop_cov = self._apply_stems(loop_full)    # apply the current stem mute/solo
                    self.loop_locked = True
                    self._link_anchor = None                   # re-align to Link on the new loop's first push
                    continue
                # NOT locked: stay SILENT (listening) and keep trying to lock — we NEVER fall into the
                # old continuous-generation fallback. That fallback ran the model every tick and its
                # committed/source latent grew UNBOUNDED (held refs, not cache: ~25 MB/s on 2B, GB/s on
                # XL -> 80+ GB -> the OS swaps -> beachball -> OOM crash) and it warbled. Staying silent
                # until a loop locks is bounded + safe; set the bar-loop field for a fast, reliable lock.
                if ahead < max(self.lookahead_samp, int(1.0 * SR)):
                    z = np.zeros((self.SL * SPF, 2), dtype=np.float32)
                    self._push_np(z, z)
                else:
                    time.sleep(0.005)
                continue
            # ===== CONTINUOUS MODE (non-looped input): rolling-encode + window+re-anchor cover =====
            avail = jit.encode_pending()            # rolling-encode new live input (MPS, this thread)
            if avail > self.full_T: self.full_T = avail
            jit._ensure_handle()
            if not cont_started:                    # cover the RECENT input, not the whole listen buffer
                cont_started = True
                base_f = gen_f = dec_f = max(0, avail - reanchor_win - headroom)
                committed = None; tiles = {}; anchor_f = base_f; aligned = False
            if restyled and committed is not None and gen_f > dec_f and (dec_f - base_f) >= C:
                committed = committed[:, :dec_f - base_f, :].contiguous(); gen_f = dec_f
                tcut = (dec_f - base_f) // jit._TILE
                tiles = {t: v for t, v in tiles.items() if t < tcut}
            # RE-ANCHOR (full drift reset): the pinned roll is BISTABLE — pinning each hop to
            # its OWN prior output, it can tip into runaway warble (bf16-sensitive, ~half the
            # time; not a timing/buffer artifact). File mode escapes this by cold-priming a
            # FRESH window from source at every loop restart; we do the same. Because we hold
            # the output a full window (reanchor_win) behind the input (see the decode gate),
            # there is always a whole window of source ahead of the decode point, so every
            # `anchor_period` we regenerate a FULL 2-pass cold-prime window [dec_f, dec_f+win]
            # (from source, NOT the drifted committed) and equal-power latent-crossfade it over
            # the small UNDECODED zone. The new pin (committed[-C:]) then lands entirely in the
            # fresh region -> the roll's autoregressive memory is fully reset; no seam (the fade
            # starts at 100% old, continuous with the already-decoded past).
            anchor_due = (anchor_period and committed is not None and (gen_f - anchor_f) >= anchor_period)
            if anchor_due and (avail - dec_f) >= reanchor_win and gen_f > dec_f:
                a0 = dec_f; w1 = a0 + reanchor_win
                cold = jit._gen(a0, a0 + self._prime_f, self.seed).tensor
                cp = min(self._prime_f, cold.shape[1] - 1)
                fresh = jit._gen(a0, w1, self.seed, pin=cold[:, :cp, :]).tensor
                XO = min(gen_f - dec_f, fresh.shape[1] - 1)      # crossfade ONLY the small undecoded zone
                fin, fout = _eq(XO, committed.device, committed.dtype)
                ov = a0 - base_f
                tail = committed[:, ov:ov + XO, :]
                blended = tail * fout.view(1, -1, 1) + fresh[:, :XO, :].to(tail.dtype) * fin.view(1, -1, 1)
                committed = torch.cat([committed[:, :ov, :], blended, fresh[:, XO:, :].to(tail.dtype)], dim=1)
                gen_f = a0 + fresh.shape[1]; anchor_f = gen_f
                tcut = (dec_f - base_f) // jit._TILE             # invalidate stale tiles at/after the decode frontier
                tiles = {t: v for t, v in tiles.items() if t < tcut}
                self.regens += 1; self.reanchors += 1
            elif anchor_due:
                self.anchor_skips += 1
            # wait until the handle exists and the first window's worth of source is captured
            if jit.handle is None or avail < C + H:
                time.sleep(0.02); continue
            # adaptive buffer gate — don't run too far ahead of the consumer
            target = max(self.lookahead_samp, int(min(8.0, self.max_step_ms / 1000 + 0.4) * SR))
            if committed is not None and ahead >= target + self.SL * SPF:
                time.sleep(0.003); continue
            t_iter = time.perf_counter()
            # extend the committed (pinned) latent, bounded by available source
            while True:
                cur_end = base_f + (committed.shape[1] if committed is not None else 0)
                if committed is not None and (cur_end >= dec_f + self.SL + headroom or cur_end >= avail):
                    break
                if committed is None:
                    w1 = min(base_f + C + H, avail)
                    cold = jit._gen(base_f, min(base_f + self._prime_f + C, w1), self.seed).tensor
                    cp = min(self._prime_f, cold.shape[1] - 1)
                    committed = jit._gen(base_f, w1, self.seed, pin=cold[:, :cp, :]).tensor
                    gen_f = base_f + committed.shape[1]
                else:
                    w1 = min(gen_f + H, avail)
                    if w1 - (gen_f - C) < C + 1:    # < 1 new frame of source -> wait for more input
                        break
                    new = jit._gen(gen_f - C, w1, self.seed, pin=committed[:, -C:, :]).tensor[:, C:, :]
                    committed = torch.cat([committed, new], dim=1); gen_f += new.shape[1]
                self.regens += 1
            # decode only frames with full context (headroom of generated frames after them),
            # AND (when re-anchoring is on) hold the output a full window behind the input
            # frontier so the re-anchor always has a whole window of source ahead for a
            # pure-fresh reset. This trail is the ~2-bar latency cost of the drift fix; with
            # re-anchor off there's no trail (lower latency, but the roll can warble).
            dec_limit = base_f + committed.shape[1] - headroom
            dec_ceiling = (avail - reanchor_win) if anchor_period else dec_limit
            dec_target = min(dec_f + self.SL, dec_limit, dec_ceiling)
            if dec_target <= dec_f:
                time.sleep(0.005); continue
            # BAR-QUANTIZE: before the very first cover push, pad the output up to the next
            # whole bar (relative to what the consumer has already drained) so the cover
            # trails the live input by an integer number of bars, locked to the grid.
            if not aligned:
                L = self._consumed_f / SR
                target = math.ceil(L / bar_s) * bar_s if L > 1e-3 else bar_s
                n_sil = int(round((target - L) * SR))
                if n_sil > 0:
                    z = torch.zeros(2, n_sil); self._push(z, z)
                aligned = True
                self._bar_trail_s = target
            seg = jit.decode_out(Latent(tensor=committed), tiles, dec_f - base_f, dec_target - base_f)
            mps_compat.mps_sync()
            if jit.dcw_enabled: seg = seg * self._dcw_gain
            orig = jit.source_slice(dec_f, seg.shape[-1])    # A/B = your live input (delayed), same frames
            self._push(seg, orig)
            dec_f = dec_target
            self.max_step_ms = max(self.max_step_ms, (time.perf_counter() - t_iter) * 1000)
            keep = (dec_f - base_f) // jit._TILE - 1
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
