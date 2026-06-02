"""Timbre-robust structure metrics for cover validation.

chroma_corr -> harmonic/melodic structure; onset_corr -> rhythmic structure.
Both stay meaningful across a timbre/genre swap, unlike amplitude-envelope corr.
"""
from __future__ import annotations
import torch

SR = 48000


def _stft_mag(wav, n_fft=2048, hop=512):
    x = wav.mean(0).float() if wav.dim() > 1 else wav.float()
    win = torch.hann_window(n_fft)
    return torch.stft(x, n_fft=n_fft, hop_length=hop, window=win, return_complex=True).abs()


def chromagram(wav, n_fft=2048, hop=512):
    spec = _stft_mag(wav, n_fft, hop)
    freqs = torch.linspace(0, SR / 2, spec.shape[0])
    pc = torch.full((spec.shape[0],), -1, dtype=torch.long)
    valid = freqs > 20
    pc[valid] = (69 + 12 * torch.log2(freqs[valid] / 440.0)).round().long() % 12
    C = torch.zeros(12, spec.shape[1])
    for k in range(12):
        m = pc == k
        if m.any():
            C[k] = spec[m].sum(0)
    return C / (C.norm(dim=0, keepdim=True) + 1e-9)


def onset_env(wav, n_fft=2048, hop=512):
    spec = _stft_mag(wav, n_fft, hop)
    return (spec[:, 1:] - spec[:, :-1]).clamp(min=0).sum(0)


def _corr(a, b):
    n = min(a.shape[-1], b.shape[-1])
    a, b = a[..., :n].flatten(), b[..., :n].flatten()
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-9))


def chroma_corr(a, b):
    return _corr(chromagram(a), chromagram(b))


def onset_corr(a, b):
    return _corr(onset_env(a), onset_env(b))
