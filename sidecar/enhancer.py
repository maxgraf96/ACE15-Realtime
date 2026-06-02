"""Prompt-enhancer subprocess: ACE-Step-1.5's 5Hz LM (format_sample) on MPS.

Runs in its OWN process because ACE-Step-1.5 and DEMON both ship an `acestep`
package — they can't co-exist in one interpreter. The sidecar spawns this and
speaks one JSON object per line:

    stdin  ->  {"caption": "tech house"}
    stdout <-  {"ok": true, "caption": "<rich, model-aligned description>"}

`format_sample` turns the user's short style into a detailed caption the cover
model actually responds to. The 5Hz LM loads lazily on the first request. All
incidental output (model/loguru logs, progress bars) is forced to stderr so
stdout carries ONLY the JSON protocol.

Env: ACE15_CKPT (checkpoints dir), ACE15_ACESTEP15 (ACE-Step-1.5 repo root),
ACE15_LM (model dir name, default acestep-5Hz-lm-0.6B).
"""
import sys
import os
import json

ACESTEP15 = os.environ.get("ACE15_ACESTEP15", "/Users/max/Code/ACE-Step-1.5")
CKPT = os.environ.get("ACE15_CKPT", "")
LM = os.environ.get("ACE15_LM", "acestep-5Hz-lm-0.6B")
sys.path.insert(0, ACESTEP15)   # ACE-Step-1.5's `acestep` wins (not DEMON's)

# Keep stdout pristine for the JSON line protocol; everything else -> stderr.
_real_stdout = sys.stdout
sys.stdout = sys.stderr

_handler = None


def _enhance(caption: str) -> dict:
    global _handler
    if _handler is None:
        from acestep.llm_inference import LLMHandler
        h = LLMHandler()
        h.initialize(checkpoint_dir=CKPT, lm_model_path=LM, backend="pt", device="mps")
        _handler = h
    from acestep.inference import format_sample
    r = format_sample(_handler, caption, "", use_constrained_decoding=True)
    if getattr(r, "success", False) and getattr(r, "caption", ""):
        return {"ok": True, "caption": r.caption}
    return {"ok": False, "caption": "", "error": getattr(r, "error", "") or "enhance failed"}


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            out = _enhance(json.loads(line).get("caption", ""))
        except Exception as e:
            out = {"ok": False, "caption": "", "error": str(e)}
        _real_stdout.write(json.dumps(out) + "\n")
        _real_stdout.flush()


if __name__ == "__main__":
    main()
