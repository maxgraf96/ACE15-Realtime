"""Runtime MPS/Apple-Silicon compatibility shims for the DEMON ACE-Step engine.

DEMON is CUDA/TensorRT-oriented. We keep its source tree READ-ONLY and instead
apply the handful of patches it needs to run on Metal (MPS) as *runtime*
monkeypatches from our own project. The three things that bite on a Mac:

  1. ``torch.cuda.synchronize`` / ``torch.cuda.empty_cache`` — sprinkled through
     the engine as timing fences / cache drains. On a CUDA-less box the
     ``empty_cache`` calls are already harmless no-ops and the ``synchronize``
     calls in the offline path are guarded by ``is_available()``, but we hard
     no-op both so nothing can ever raise.
  2. ``ode_steps.apg_project`` runs in fp64 (``.double()``) for numerical
     stability. MPS has no float64 — it raises. We swap in an fp32 version.
     Only matters when CFG/guidance is engaged.
  3. DEMON must be importable: we prepend its repo root to ``sys.path``.

Call :func:`install` once, before importing anything from ``acestep``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

DEMON_ROOT = "/Users/max/Code/DEMON"


def _add_demon_to_path() -> None:
    if DEMON_ROOT not in sys.path:
        sys.path.insert(0, DEMON_ROOT)


def _noop_cuda() -> None:
    """Neutralize the engine's CUDA timing/cleanup calls on a Mac.

    ``synchronize`` becomes a no-op (we fence with ``torch.mps.synchronize``
    explicitly where we time). ``empty_cache`` is REDIRECTED to
    ``torch.mps.empty_cache`` — the engine sprinkles ``torch.cuda.empty_cache()``
    as its "return freed memory to the pool" hook; if we no-op it, MPS memory
    accumulates across generate/decode calls and OOMs (it allocated 61 GiB into
    swap before this fix). Mapping it to the MPS equivalent keeps it bounded.
    """
    import torch

    if not torch.cuda.is_available():
        torch.cuda.synchronize = lambda *a, **k: None  # type: ignore[assignment]
        # NO-OP, not redirect-to-mps: the engine calls empty_cache very
        # frequently (per _load_model_context, per VAE op). torch.mps.empty_cache
        # synchronizes and is expensive — routing every internal call to it tanks
        # throughput (~10x). Instead callers manage MPS memory explicitly via
        # reclaim() once per outer iteration (bounded memory, fast hot path).
        torch.cuda.empty_cache = lambda *a, **k: None  # type: ignore[assignment]


def reclaim() -> None:
    """Return freed MPS memory to the OS. Call once per outer loop iteration
    (per generate/decode), NOT inside the hot path."""
    import gc
    import torch

    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def patch_apg_fp32() -> None:
    """Replace ode_steps.apg_project's fp64 path with fp32 (MPS has no float64).

    Byte-for-byte the same projection, only the working precision changes from
    double to float. Idempotent.
    """
    import torch
    from acestep.engine import ode_steps

    if getattr(ode_steps.apg_project, "_mps_fp32", False):
        return

    def apg_project(v0, v1, dims=(-1,)):
        dtype = v0.dtype
        v0f, v1f = v0.float(), v1.float()
        v1f = torch.nn.functional.normalize(v1f, dim=dims)
        v0_parallel = (v0f * v1f).sum(dim=dims, keepdim=True) * v1f
        v0_orthogonal = v0f - v0_parallel
        return v0_parallel.to(dtype), v0_orthogonal.to(dtype)

    apg_project._mps_fp32 = True  # type: ignore[attr-defined]
    ode_steps.apg_project = apg_project


def mps_sync() -> None:
    """Flush the MPS command queue so timings measure real GPU work."""
    import torch

    if torch.backends.mps.is_available():
        torch.mps.synchronize()


def _disable_grad() -> None:
    """Inference-only engine: disable autograd globally.

    The DiT/VAE/text-encoder params load with requires_grad=True, so any forward
    NOT wrapped in no_grad builds + retains an autograd graph. The StreamPipeline
    tick has its own @torch.no_grad, but the VAE decode path does not — under a
    long streaming session that retained-graph memory balloons into a hard MPS
    OOM (observed: 61 GiB into swap). We never train, so kill grad process-wide.
    """
    import torch

    torch.set_grad_enabled(False)


def install(*, patch_apg: bool = True) -> None:
    """Apply all shims. Safe to call multiple times."""
    _add_demon_to_path()
    _noop_cuda()
    _disable_grad()
    if patch_apg:
        # ode_steps import is cheap and pulls in no heavy deps.
        try:
            patch_apg_fp32()
        except Exception as e:  # pragma: no cover - surfaced, not fatal
            print(f"[mps_compat] apg fp32 patch skipped: {e}")


__all__ = ["install", "patch_apg_fp32", "mps_sync", "DEMON_ROOT"]
