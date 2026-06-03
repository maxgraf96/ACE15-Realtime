"""Real-time source separation (torchaudio Hybrid Demucs, 4 stems) for live mode.

Lets the OUTPUT stem mixer isolate stems of the generated accompaniment — e.g. "only the
drums the model made" (mute/solo per stem).
Demucs runs at 44.1 kHz; we resample 48k<->44.1k around it. It's a non-causal segment
model, so live use runs it per encode-chunk WITH a discard margin (overlap-discard).
Measured RTF on MPS ~0.012, so the cost is negligible next to the cover pipeline.
"""
from __future__ import annotations
import torch

SR = 48000
SR_D = 44100   # Demucs operates at 44.1 kHz
STEMS = ("drums", "bass", "other", "vocals")


class StemSeparator:
    def __init__(self, device="mps"):
        self.device = device
        self.model = None
        self.sources = None

    def _ensure(self):
        if self.model is None:
            from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB_PLUS as B
            self.model = B.get_model().to(self.device).eval()
            self.sources = list(self.model.sources)   # ['drums','bass','other','vocals']
            with torch.no_grad():                     # warm the MPS graph (first forward is slow ~3-4s)
                self.model(torch.zeros(1, 2, int(SR_D * 2.0), device=self.device))
            if str(self.device).startswith("mps"):
                torch.mps.synchronize()

    def separate(self, wav48, stems):
        """wav48: [2,N] @48k. `stems`: source names to KEEP (summed). Returns [2,N] @48k.
        Empty/None stems -> the full mix (passthrough of the resampled signal)."""
        self._ensure()
        import torchaudio.functional as AF
        n0 = wav48.shape[-1]
        x = wav48.to(self.device).float()
        if x.dim() == 1:
            x = x.unsqueeze(0).repeat(2, 1)
        xr = AF.resample(x, SR, SR_D)
        m, s = xr.mean(), xr.std() + 1e-8            # Demucs expects normalised input
        with torch.no_grad():
            out = self.model(((xr - m) / s).unsqueeze(0))[0] * s + m   # [src, 2, n]
        idx = [self.sources.index(k) for k in (stems or []) if k in self.sources]
        sel = out[idx].sum(0) if idx else out.sum(0)  # [2, n]
        sel = AF.resample(sel, SR_D, SR)
        if sel.shape[-1] >= n0:
            sel = sel[:, :n0]
        else:
            sel = torch.nn.functional.pad(sel, (0, n0 - sel.shape[-1]))
        return sel.detach().cpu()

    def close(self):
        self.model = None
