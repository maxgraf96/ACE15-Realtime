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

from . import mps_compat
mps_compat.install()  # add DEMON to sys.path + cuda/grad shims, for ANY importer (sidecar, tests, app)

import torch  # noqa: E402

FPS = 25.0
SR = 48000
SPF = 1920  # samples per latent frame
SEM_PATCH = 5  # semantic tokenizer patch (25 fps -> 5 Hz); source latent T must be a multiple


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
    def __init__(self, device="mps", steps=8, shift=3.0, config_path="acestep-v15-turbo"):
        from acestep.engine.session import Session
        self.session = Session(device=device, decoder_backend="eager", vae_backend="eager",
                               use_flash_attention=False, vae_window=0.0, config_path=config_path)
        self.device = device
        self.stems = None      # OUTPUT stem mixer: stems to KEEP in the cover output (e.g. ['drums']); None/all = full mix
        self._sep = None       # lazy StemSeparator
        self.steps = steps; self.shift = shift
        self.source = None; self.cond = None; self.handle = None
        self.track_dur = 0.0; self.denoise = 0.8
        # character 0..1: 0 = full style (no source timbre), 1 = keep source character
        self.character = 0.0
        self.evolve = False; self._regen = 0
        self.tags = ""
        self.send_bpm = True; self.send_key = True   # inject detected tempo/key into Metas
        # DCW (Differential Correction in Wavelet domain) — sampler-side per-step
        # correction (acestep.engine.dcw). OFF by default: in our turbo/few-step
        # regime it runs the output hot (pushes peaks into the soft-clip limiter →
        # harsh on dense material) for only a marginal structure gain. Opt-in via the
        # UI toggle. Recommended ACE-Step values kept (double/0.05/0.02/haar).
        self.dcw_enabled = False
        self.dcw_mode = "double"; self.dcw_scaler = 0.05
        self.dcw_high_scaler = 0.02; self.dcw_wavelet = "haar"
        # Lyrics: always inject [Instrumental] (user choice) to keep covers instrumental
        # / suppress sung-lyric artifacts. (An earlier probe found EMPTY gave marginally
        # stronger style — funk CLAP 0.22 vs 0.14 — but instrumental-locking wins here.)
        self.lyrics = "[Instrumental]"
        self.peaks: list = []
        self.source_wav = None                 # [C, samples] @ SR — kept for the instant A/B bypass
        self.input_gain_db = 0.0               # input trim feeding the model (<=0 dB); re-encodes the source
        self.bpm = 120; self.key = "C major"   # auto-detected in load_track

    def load_track(self, path, seconds=None, detect=True):
        from . import loader
        a = loader.load_audio(path, duration=seconds)
        self.track_dur = a.waveform.shape[-1] / SR
        if detect:
            # Match the source's real tempo/key so the text metadata agrees with
            # the structure conditioning (a mismatch makes the cover incoherent).
            try:
                from acestep.nodes.audio_nodes import AudioInfo
                info = AudioInfo().execute(audio=a)
                self.bpm, self.key = int(info["bpm"]), str(info["key"])
                print(f"[jit] detected bpm={self.bpm} key={self.key}")
            except Exception as e:
                print(f"[jit] bpm/key detect failed ({e}); using {self.bpm}/{self.key}")
        # The semantic tokenizer patchifies T into groups of SEM_PATCH (5), so the
        # source latent length MUST be a multiple of 5 or extract_hints rearrange
        # fails. Encode, truncate to a multiple, then extract hints.
        self.source_wav = a.waveform.detach().float().cpu()   # raw source @ SR (A/B ref + re-encode src)
        self._encode_source()
        self.peaks = self._compute_peaks(a.waveform, 220)
        return self

    def _encode_source(self):
        """Encode the source waveform (scaled by the input-gain trim) into the latent
        + hints used as the generation source. Re-callable so the input gain can change
        live (re-encode). source_wav stays the TRUE original (the A/B bypass reference)."""
        from acestep.nodes.types import Latent, Audio
        from acestep.engine.session import PreparedSource
        g = 10.0 ** (self.input_gain_db / 20.0)
        wav = self.source_wav if g == 1.0 else self.source_wav * g
        lat = self.session.encode_audio(Audio(waveform=wav, sample_rate=SR))
        T = lat.tensor.shape[1]
        T5 = max(SEM_PATCH, (T // SEM_PATCH) * SEM_PATCH)
        if T5 != T:
            lat = Latent(tensor=lat.tensor[:, :T5, :].contiguous())
        ctx = self.session.extract_hints(lat)
        self.source = PreparedSource(latent=lat, context_latent=ctx)

    def set_input_gain(self, db):
        """Input trim feeding the model (max 0 dB): re-encode the source from the
        gain-scaled waveform. Heavy (full re-encode) — run from the producer thread;
        the ring buffer covers the stall, like one large regen."""
        self.input_gain_db = float(db)
        if getattr(self, "_live", False):
            return   # live source has no single waveform to re-encode; gain is applied on feed
        self._encode_source()

    # ---- live / streaming source (Phase 3: real-time input) ----
    LIVE_CHUNK_F = 25     # latent frames appended per encode step (1.0 s; multiple of SEM_PATCH)
    LIVE_MARGIN_F = 12    # right (future) overlap discarded — bounded by how much input we can wait for
    LIVE_LMARGIN_F = 12   # LEFT (past) context fed to the chunk encode; too small starves the VAE
                          # receptive field -> boundary artifacts that the roll amplifies into drift
    HINT_MARGIN_F = 25    # left-context margin for incremental (frozen) hint extraction (multiple of 5)
    OUT_SEP_MARGIN_F = 20  # discard margin for OUTPUT separation (~0.8s; rolling-vs-whole corr 0.976, fits headroom)

    def set_stems(self, stems):
        """OUTPUT stem mixer: keep only these stems (e.g. ['drums']) of the generated cover.
        None / all-4 = full mix (no separation). Applied to subsequent decoded slices."""
        self.stems = list(stems) if stems else None

    def _ensure_sep(self):
        if self._sep is None:
            from .separation import StemSeparator
            self._sep = StemSeparator(device=self.device)
        return self._sep

    def decode_out(self, lat, cache, a, b):
        """Decode cover local frames [a,b) -> audio [2,(b-a)*SPF], then (if a stem subset is
        selected) separate the decoded cover and keep only those stems. Separates a margin
        window (overlap-discard) so per-slice separation is seamless; reuses the tile cache."""
        full = self._ensure_tiles(lat, cache, a, b)
        from .separation import STEMS
        st = self.stems
        if not st or set(st) >= set(STEMS):          # full mix -> no separation
            return full
        m = self.OUT_SEP_MARGIN_F
        Wl = lat.tensor.shape[1]
        a2, b2 = max(0, a - m), min(Wl, b + m)
        wide = self._ensure_tiles(lat, cache, a2, b2)   # cover audio incl. margin (cached/redecoded tiles)
        mix = self._ensure_sep().separate(wide, st)     # one Demucs pass -> summed selected stems [2, ...]
        lo = (a - a2) * SPF
        return mix[:, lo: lo + (b - a) * SPF]

    def begin_live(self):
        """Start an EMPTY streaming source, grown by feed_live()+encode_pending(). For
        real-time live input — bpm/key come from the UI (manual/host), not detection."""
        import threading as _th
        self._live = True
        self._dbg_whole = False
        self._onset = -1                 # sample index of the first audible input (loop hard start)
        self.enc_lat_mode = "window"; self.enc_hint_mode = "frozen"   # seamless recent source + cheap stable hints
        self._TILE = 16   # smaller decode tiles for live -> less decode headroom -> lower latency
        self._live_lock = _th.Lock()
        self._live_raw = torch.zeros(2, 0)   # raw 48k stereo (cpu); also the A/B reference
        self._enc_f = 0                      # source-latent frames committed so far
        self._ctx = None                     # frozen (append-only) hints for the streaming source
        self.source = None
        self.source_wav = self._live_raw
        self.peaks = []
        self.track_dur = 30.0                # nominal duration meta (open-ended live)
        return self

    def feed_live(self, pcm):
        """Append raw 48 kHz input (stereo [2,n] or [n,2], or mono [n]). Thread-safe."""
        x = pcm if isinstance(pcm, torch.Tensor) else torch.as_tensor(pcm)
        x = x.float()
        if x.dim() == 1:
            x = x.unsqueeze(0).repeat(2, 1)
        elif x.shape[0] != 2 and x.shape[-1] == 2:
            x = x.t()
        x = x.cpu().contiguous()
        g = 10.0 ** (self.input_gain_db / 20.0)
        if g != 1.0:
            x = x * g                        # input-gain trim applied as the stream arrives
        with self._live_lock:
            start = self._live_raw.shape[1]
            self._live_raw = torch.cat([self._live_raw, x], dim=1)
            self.source_wav = self._live_raw
            if self._onset < 0:                  # ONSET = first audible sample = the loop's hard start
                hot = (x.abs().mean(0) > 0.03).nonzero()
                if hot.numel() > 0:
                    self._onset = start + int(hot[0, 0])

    # Live source pipeline (two independent axes; see encode_pending). enc_lat_mode:
    #   "chunked" = append a small overlap-discard chunk each step (cheap, O(1)) but the
    #               seams ACCUMULATE -> the dominant cause of live drift/warble;
    #   "whole"   = re-encode ALL raw each step (seamless, but O(n) -> too slow long-term);
    #   "window"  = re-encode only the recent W_ENC_F frames seamlessly and splice onto the
    #               frozen older latent (seamless where the generator reads, O(W) cost) <-
    #               the real fix. enc_hint_mode: "frozen" = extract hints for the new region
    #   only + freeze (cheap, stable); "whole" = re-extract all each step (O(n), slow, no
    #   quality gain — measured). Live default = window+frozen (seamless source, real-time).
    enc_lat_mode = "window"
    enc_hint_mode = "frozen"
    W_ENC_F = 224   # ~9 s recent re-encode window; the generator only reads the recent ~5 s,
                    # so the splice seam at the window's left edge sits safely behind it
    GEN_MARGIN_F = 12  # hold the GENERATABLE frontier this far behind the encode frontier: the
                       # newest source frames lack future context (encoder future RF), so letting
                       # the roll consume them bakes in instability. (chunked already excludes mf.)

    def encode_pending(self):
        """Encode any newly-available live audio and (re)build the streaming source latent
        + hints. Run from the PRODUCER thread (MPS). Returns the number of source frames now
        available to generate from. Source latent and hints are built independently per
        enc_lat_mode / enc_hint_mode so the two contributors to live drift can be isolated."""
        from acestep.nodes.types import Audio, Latent
        from acestep.engine.session import PreparedSource
        whole = getattr(self, "_dbg_whole", False)   # back-compat shorthand for whole+whole
        lat_mode = "whole" if whole else self.enc_lat_mode
        hint_mode = "whole" if whole else self.enc_hint_mode
        cf, mf, hm = self.LIVE_CHUNK_F, self.LIVE_MARGIN_F, self.HINT_MARGIN_F
        with self._live_lock:
            raw = self._live_raw
        n_avail = (raw.shape[1] // SPF // SEM_PATCH) * SEM_PATCH   # total encodable frames (multiple of 5)
        # ---- 1) source latent up to `target` ----
        if lat_mode in ("whole", "window"):
            # re-encode is seamless but costs O(window); only fire once >= cf new frames
            # arrived (not every ~0.2 s) so the load stays ~encode(window)/cf-seconds.
            if n_avail - self._enc_f < (cf if self._enc_f else 1):
                return max(0, self._enc_f - self.GEN_MARGIN_F)   # safe generatable frontier (consistent)
            target = n_avail
            prev = None if self.source is None else self.source.latent.tensor
            if lat_mode == "whole" or prev is None:
                lo = 0                               # prev=None: must encode from 0 to keep absolute indexing
            else:
                lo = max(0, ((target - self.W_ENC_F) // SEM_PATCH) * SEM_PATCH)
                lo = min(lo, prev.shape[1])          # never leave a gap before the re-encoded window
            enc = self.session.encode_audio(
                Audio(waveform=raw[:, lo * SPF: target * SPF], sample_rate=SR)).tensor[:, :target - lo, :].contiguous()
            lat_full = enc if (lo == 0 or prev is None) else torch.cat([prev[:, :lo, :], enc], dim=1)
        else:  # chunked overlap-discard append of [enc_f, enc_f+cf) with a LEFT-context margin
            lm = self.LIVE_LMARGIN_F
            if raw.shape[1] < (self._enc_f + cf + mf) * SPF:
                return self._enc_f
            lat_full = None if self.source is None else self.source.latent.tensor
            cur = self._enc_f
            while raw.shape[1] >= (cur + cf + mf) * SPF:     # chunkall drains ALL pending chunks; chunked does one
                lo = max(0, cur - lm); hi = cur + cf + mf
                lat = self.session.encode_audio(Audio(waveform=raw[:, lo * SPF: hi * SPF], sample_rate=SR)).tensor
                new = lat[:, cur - lo: cur - lo + cf, :]
                if new.shape[1] < cf:
                    break
                lat_full = new.contiguous() if lat_full is None else torch.cat([lat_full, new.contiguous()], dim=1)
                cur += cf
                if lat_mode != "chunkall":
                    break
            if cur == self._enc_f:
                return self._enc_f
            target = cur
        # ---- 2) hints up to `target` ----
        if hint_mode == "whole":
            ctx_full = self.session.extract_hints(Latent(tensor=lat_full)).tensor
        else:  # frozen: extract only the NEW region [old, target) (left-context margin) and append
            old = self._enc_f
            lo2 = max(0, old - hm)                    # multiple of 5
            hint = self.session.extract_hints(Latent(tensor=lat_full[:, lo2:target, :])).tensor
            new_ctx = hint[:, old - lo2:, :].contiguous()
            ctx_full = new_ctx if self._ctx is None else torch.cat([self._ctx, new_ctx], dim=1)
            self._ctx = ctx_full
        self._enc_f = target
        self.source = PreparedSource(latent=Latent(tensor=lat_full), context_latent=Latent(tensor=ctx_full))
        # whole/window expose every encoded frame incl. the under-contexted frontier; hold the
        # generatable frontier GEN_MARGIN_F behind it. chunked already committed only mf-future frames.
        margin = self.GEN_MARGIN_F if lat_mode in ("whole", "window") else 0
        return max(0, self._enc_f - margin)

    def detect_loop(self, min_bars=1, max_bars=16, thresh=0.90):
        """Find the loop period in the live raw buffer via normalized autocorrelation at
        integer-bar lags (from self.bpm). The input is a non-causal model's nightmare —
        no future context at the live edge — but if it LOOPS, the future == the loop, so we
        can cover it like a file (clean, low-latency). Returns (period_samples, confidence,
        bars) for the SMALLEST bar-count that self-matches above `thresh`, else (None,c,0).
        Needs >= 2 loops captured. Run from the producer thread (reads raw, cheap CPU)."""
        with self._live_lock:
            raw = self._live_raw
        n = raw.shape[1]
        bar = 60.0 / max(1e-6, float(self.bpm)) * 4.0 * SR     # samples per bar (4/4)
        if n < int(2 * bar):
            return None, 0.0, 0
        mono = raw.float().mean(0)
        win = max(int(bar), int(2 * bar))    # compare a 2-bar window (not a whole loop) -> detect sooner
        cands = []
        for B in range(min_bars, max_bars + 1):
            P = int(round(B * bar))
            w = min(P, win)
            if n < P + w:
                break
            a = mono[n - w:n]; b = mono[n - w - P:n - P]   # last w samples vs w samples one period earlier
            denom = (a.norm() * b.norm()).item() + 1e-9
            cands.append((B, P, (a * b).sum().item() / denom))   # cosine similarity at lag = P
        if not cands:
            return None, 0.0, 0
        mx = max(c[2] for c in cands)
        for B, P, c in cands:                                   # fundamental = smallest bar-count that matches
            if c >= thresh and c >= mx - 0.04:
                return P, c, B
        return None, mx, 0

    def source_slice(self, f0, n_samples):
        """Original source audio for the slice starting at latent frame f0, exactly
        n_samples long (zero-padded past the end), as stereo [2, n_samples]. Frame f
        maps to samples [f*SPF:(f+1)*SPF], so this is sample-aligned with the decoded
        cover for the same frames -> the client can A/B cover vs source with no drift."""
        if self.source_wav is None:
            return torch.zeros(2, n_samples)
        w = self.source_wav
        if w.shape[0] == 1:
            w = w.expand(2, -1)
        elif w.shape[0] > 2:
            w = w[:2]
        s0 = f0 * SPF
        seg = w[:, s0:s0 + n_samples]
        if seg.shape[-1] < n_samples:
            seg = torch.cat([seg, torch.zeros(2, n_samples - seg.shape[-1], dtype=seg.dtype)], dim=-1)
        return seg

    @staticmethod
    def _compute_peaks(wav, n):
        """Downsample |mono| to n peak bars in [0,1] for the UI waveform."""
        import torch as _t
        x = wav.float().abs().mean(0)
        if x.numel() == 0:
            return []
        step = max(1, x.numel() // n)
        p = _t.nn.functional.max_pool1d(x.view(1, 1, -1), step, step).flatten()
        m = float(p.max()) or 1.0
        return [round(float(v) / m, 4) for v in p[:n]]

    def _refer(self):
        """Timbre reference latent for the current Character (0=style, 1=source)."""
        c = self.character
        if c <= 0.01 or self.source is None:
            return None                       # full style (CLAP-tuned default), or no source yet (live)
        if c >= 0.99:
            return self.source.latent         # keep source character
        from acestep.nodes.types import Latent
        h = self.session.handler
        h._ensure_silence_latent_on_device()
        T = self.source.latent.tensor.shape[1]
        sil = h.silence_latent
        if sil.shape[1] < T:
            sil = sil.repeat(1, (T + sil.shape[1] - 1) // sil.shape[1], 1)
        sil = Latent(tensor=sil[:, :T, :].to(self.source.latent.tensor))
        return self.session.blend_latents(sil, self.source.latent, alpha=c)  # 0=silence,1=source

    def _encode(self):
        # Build the ACE-Step cover prompt ourselves so we can OMIT bpm/key
        # (they're optional soft guidance) and signal an instrumental cover via
        # [Instrumental] lyrics — per the ACE-Step prompting guide. Caption =
        # style/instruments/timbre only (NEVER tempo/key — those go in Metas).
        from acestep.constants import TASK_INSTRUCTIONS
        from acestep.nodes.types import TextEmbed
        from acestep.nodes.cond_nodes import EncodeConditioning
        h = self.session.handler
        device = h.device
        metas = []
        if self.send_bpm:
            metas.append(f"- bpm: {self.bpm}")
        metas.append("- timesignature: 4")
        if self.send_key:
            metas.append(f"- keyscale: {self.key}")
        metas.append(f"- duration: {self.track_dur}")
        text_prompt = (f"# Instruction\n{TASK_INSTRUCTIONS['cover']}\n\n"
                       f"# Caption\n{self.tags}\n\n# Metas\n" + "\n".join(metas) + "\n<|endoftext|>\n")
        lyrics_prompt = f"# Languages\nen\n\n# Lyric\n{self.lyrics}<|endoftext|><|endoftext|>"
        with h._load_model_context("text_encoder"):
            t = h.text_tokenizer(text_prompt, return_tensors="pt", add_special_tokens=False)
            text_hidden = h.infer_text_embeddings(t["input_ids"].to(device))
            text_mask = t["attention_mask"].to(device).bool()
            lt = h.text_tokenizer(lyrics_prompt, return_tensors="pt", add_special_tokens=False)
            lyric_hidden = h.infer_lyric_embeddings(lt["input_ids"].to(device))
            lyric_mask = torch.ones(lyric_hidden.shape[:2], device=device, dtype=torch.bool)
        te = TextEmbed(text_hidden_states=text_hidden, text_attention_mask=text_mask,
                       lyric_hidden_states=lyric_hidden, lyric_attention_mask=lyric_mask)
        self.cond = EncodeConditioning().execute(
            model=self.session.model, text_embed=te, timbre_ref=self._refer())["conditioning"]
        if self.handle is not None:
            self.handle.conditioning = self.cond

    def _set_bpm_key(self, bpm=None, key=None):
        """Override the auto-detected tempo/key from user input (best-effort parse).
        Empty/garbage is ignored so a stray keystroke can't blank the conditioning."""
        if bpm is not None:
            try: self.bpm = int(float(str(bpm).strip()))
            except (ValueError, TypeError): pass
        if key is not None:
            k = str(key).strip()
            if k: self.key = k

    def set_metas(self, send_bpm=None, send_key=None, bpm=None, key=None):
        if send_bpm is not None: self.send_bpm = bool(send_bpm)
        if send_key is not None: self.send_key = bool(send_key)
        self._set_bpm_key(bpm, key)
        self._encode()

    def set_style(self, tags, denoise=0.8, character=None, bpm=None, key=None, timbre=None,
                  send_bpm=None, send_key=None):
        self.denoise = denoise
        self.tags = tags
        if character is not None:
            self.character = float(character)
        elif timbre is not None:                          # back-compat: "source"/"none"
            self.character = 1.0 if timbre == "source" else 0.0
        self._set_bpm_key(bpm, key)
        if send_bpm is not None: self.send_bpm = bool(send_bpm)
        if send_key is not None: self.send_key = bool(send_key)
        self._encode()
        self._ensure_handle()   # no-op in live mode until the first chunk is encoded
        return self

    def _ensure_handle(self):
        """Build the streaming pipeline handle once a source exists (deferred in live
        mode, where the source latent only appears after the first encode_pending)."""
        if self.handle is None and self.source is not None:
            self.handle = self.session.stream(source=self.source, conditioning=self.cond,
                                              steps=self.steps, shift=self.shift, pipeline_depth=1,
                                              **self._dcw_kwargs())

    def _dcw_kwargs(self):
        """DCW params for session.stream() / handle.base_kwargs (read every tick)."""
        return {"dcw_enabled": self.dcw_enabled, "dcw_mode": self.dcw_mode,
                "dcw_scaler": self.dcw_scaler, "dcw_high_scaler": self.dcw_high_scaler,
                "dcw_wavelet": self.dcw_wavelet}

    def set_dcw(self, enabled=None, mode=None, scaler=None, high_scaler=None, wavelet=None):
        """Toggle/tune DCW. StreamDenoise.execute re-reads dcw_* from base_kwargs
        every tick and calls pipe.set_dcw(), so mutating base_kwargs hot-applies on
        the next regen — no pipeline rebuild. Guards against a missing wavelet dep."""
        if enabled:
            from . import mps_compat
            if not mps_compat.dcw_available():
                print("[jit] DCW requested but pytorch_wavelets is missing; staying off")
                enabled = False
        if enabled is not None: self.dcw_enabled = bool(enabled)
        if mode is not None: self.dcw_mode = mode
        if scaler is not None: self.dcw_scaler = float(scaler)
        if high_scaler is not None: self.dcw_high_scaler = float(high_scaler)
        if wavelet is not None: self.dcw_wavelet = wavelet
        if self.handle is not None:
            self.handle.base_kwargs.update(self._dcw_kwargs())

    def _set_prompt(self, tags):
        self.tags = tags
        self._encode()

    def set_character(self, c):
        self.character = float(c)
        self._encode()

    def _gen(self, w0, w1, seed, pin=None):
        """Generate the cover latent for source[w0:w1]. If `pin` ([1,C,D] clean
        latent) is given, the window's first C frames are PINNED to it via an
        inpainting mask (mask=0 preserve, mask=1 generate): the new frames are
        denoised to CONTINUE that fixed past -> seamless rolling (no per-window seam)."""
        from acestep.nodes.types import Latent
        from acestep.engine.session import PreparedSource
        src = self.source.latent.tensor[:, w0:w1, :]
        ctx = self.source.context_latent.tensor[:, w0:w1, :]
        self.handle.context_latent = Latent(tensor=ctx)
        src_lat = Latent(tensor=src)
        if pin is not None:
            from acestep.engine.masking import LatentNoiseMask
            C = pin.shape[1]
            orig = src.clone(); orig[:, :C, :] = pin.to(src.dtype)
            m = torch.ones(1, src.shape[1], 1, device=src.device, dtype=src.dtype); m[:, :C, :] = 0.0
            src_lat.mask = LatentNoiseMask(mask=m, original_latents=orig)
        self.handle.source = PreparedSource(latent=src_lat, context_latent=Latent(tensor=ctx))
        eff_seed = seed + (self._regen if self.evolve else 0)   # Evolve: browse new variations
        self._regen += 1
        lat = self.handle.tick(drain=True, denoise=self.denoise, seed=eff_seed)
        mps_compat.mps_sync()
        return lat

    # Lazy tiled decode: decode only the latent tiles needed for the frames the
    # playhead is reaching, cached per window. A regen then costs gen + ONE tile
    # (~0.15 s) instead of a full-window decode (~0.9 s), so control latency
    # (>= regen spike) drops accordingly. Overlap-discard tiling is seamless and
    # bit-matches a full decode (same as the old _full_decode).
    _TILE = 48          # kept frames per tile (1.92 s)
    _TOV = 8            # receptive-field margin frames each side

    def _ensure_tiles(self, lat, cache, a, b):
        """Decode any tiles covering local frames [a,b); return audio [2,(b-a)*SPF]."""
        vae = self.session.handler.vae
        lb = lat.tensor.transpose(1, 2)
        Wl = lb.shape[-1]
        a = max(0, a); b = min(Wl, b)
        t0, t1 = a // self._TILE, (b - 1) // self._TILE
        with self.session.handler._load_model_context("vae"):
            for t in range(t0, t1 + 1):
                if t in cache:
                    continue
                ks, ke = t * self._TILE, min((t + 1) * self._TILE, Wl)
                ds, de = max(0, ks - self._TOV), min(Wl, ke + self._TOV)
                w = vae.decode(lb[..., ds:de].to(vae.dtype)).sample.float()
                tl, tr = (ks - ds) * SPF, (de - ke) * SPF
                cache[t] = w[..., tl: w.shape[-1] - tr if tr else None].squeeze(0).cpu()
        parts = []
        for t in range(t0, t1 + 1):
            ks, ke = t * self._TILE, min((t + 1) * self._TILE, Wl)
            lo, hi = max(a, ks) - ks, min(b, ke) - ks
            parts.append(cache[t][:, lo * SPF: hi * SPF])
        return torch.cat(parts, dim=-1)

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
        win_lat = None                    # current window latent
        win_tiles: dict = {}              # lazily-decoded tiles for win_lat (CPU)
        win_nf = 0                        # window length in latent frames
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

            # (re)generate the window when we leave it or a control fired. The
            # window is decoded LAZILY tile-by-tile, so a regen blocks only on
            # gen + the first tile, not a full-window decode.
            if win_lat is None or committed_f >= win_start + win_nf or force:
                back = XF_f if out is not None else 0
                win_start = max(0, min(committed_f - back, full_T - W))
                g0 = time.perf_counter()
                win_lat = self._gen(win_start, min(win_start + W, full_T), seed)
                win_nf = win_lat.tensor.shape[1]
                win_tiles = {}
                gen_ms = (time.perf_counter() - g0) * 1000
                st.gen_ms.append(gen_ms)
                st.regens += 1
                force = False
                from_f = committed_f - back
                f1 = min(committed_f + SL, win_start + win_nf, full_T)
                d0 = time.perf_counter()
                seg = self._ensure_tiles(win_lat, win_tiles, from_f - win_start, f1 - win_start)
                mps_compat.mps_sync()
                dec_ms = (time.perf_counter() - d0) * 1000
                st.dec_ms.append(dec_ms)
                st.max_step_ms = max(st.max_step_ms, gen_ms + dec_ms)  # the regen spike that sets min lookahead
                append(seg, XF if back else 0)
                committed_f = from_f + seg.shape[-1] // SPF
                mps_compat.reclaim()
                continue

            # steady: decode (or reuse cached) the next slice's tiles, full quality
            f1 = min(committed_f + SL, win_start + win_nf, full_T)
            seg = self._ensure_tiles(win_lat, win_tiles, committed_f - win_start, f1 - win_start)
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
