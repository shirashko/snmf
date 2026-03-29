"""
PyTorch device selection: 'auto' (cuda → mps → cpu), and CUDA capability checks
for wheels that omit older GPU architectures (e.g. sm_61).
"""
from __future__ import annotations

from typing import Callable

import torch


def default_device() -> str:
    """Default CLI value for model/fitting device: pick best backend at runtime."""
    return "auto"


def expand_auto_device(spec: str) -> str:
    """If spec is 'auto', prefer cuda, then mps, else cpu. Otherwise return spec unchanged."""
    if spec.strip().lower() != "auto":
        return spec
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def ensure_cuda_usable(
    device_str: str,
    label: str,
    log_fn: Callable[[str], None],
) -> str:
    """
    If device is cuda, ensure this PyTorch build has kernels for the GPU (e.g. sm_70+).
    Otherwise fall back to cpu (e.g. Pascal sm_61 while wheels only ship sm_70+).
    """
    if not device_str.startswith("cuda"):
        return device_str
    if not torch.cuda.is_available():
        log_fn(f"{label}: CUDA requested but not available; using cpu.")
        return "cpu"
    try:
        idx = int(device_str.split(":", 1)[1]) if ":" in device_str else 0
        major, minor = torch.cuda.get_device_capability(idx)
    except Exception as e:
        log_fn(f"{label}: could not read CUDA capability ({e}); using cpu.")
        return "cpu"
    if major < 7:
        log_fn(
            f"{label}: GPU compute capability {major}.{minor} is below sm_70; "
            f"this PyTorch install has no kernels for it. Using cpu."
        )
        return "cpu"
    return device_str


def resolve_device(spec: str, label: str, log_fn: Callable[[str], None]) -> str:
    """Apply auto selection, then CUDA capability checks."""
    return ensure_cuda_usable(expand_auto_device(spec), label, log_fn)
