"""Objective style-adherence metric via CLAP (audio<->text similarity).

clap_score(wav_path, target_text, distractors) returns:
  - sim: cosine similarity audio vs target text (higher = more "sounds like it"),
  - margin: target sim minus best distractor sim (>0 = target genre wins),
  - rank/softmax confidence of target among the panel.
Lets us tune style strength without round-tripping a human for every config.
"""
from __future__ import annotations
import numpy as np

_MODEL = None


def _model():
    global _MODEL
    if _MODEL is None:
        import io
        import contextlib
        import laion_clap
        m = laion_clap.CLAP_Module(enable_fusion=False)
        # load_ckpt prints every weight name ("... Loaded") — silence it.
        with contextlib.redirect_stdout(io.StringIO()):
            m.load_ckpt()  # default 630k-audioset checkpoint (downloaded once)
        _MODEL = m
    return _MODEL


def _audio_embed(path):
    return _model().get_audio_embedding_from_filelist(x=[path], use_tensor=False)[0]


def text_embeds(texts):
    return _model().get_text_embedding(texts, use_tensor=False)


def clap_scores(path, panel):
    """panel: list of genre/style texts. Returns dict text->cosine sim (normalized embeds)."""
    a = _audio_embed(path)
    a = a / (np.linalg.norm(a) + 1e-9)
    T = text_embeds(panel)
    T = T / (np.linalg.norm(T, axis=1, keepdims=True) + 1e-9)
    sims = T @ a
    return {t: float(s) for t, s in zip(panel, sims)}


def score(path, target, distractors):
    panel = [target] + list(distractors)
    s = clap_scores(path, panel)
    tgt = s[target]
    best_d = max(s[d] for d in distractors)
    return {"sim": tgt, "margin": tgt - best_d, "wins": tgt >= best_d, "all": s}
