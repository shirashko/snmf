import os
import re
import pickle
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Basic Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _cuda_usable_for_compute() -> bool:
    """
    torch.cuda.is_available() can be True on GPUs whose architecture is not compiled
    into this PyTorch binary (e.g. Pascal sm_61 vs wheels that only ship sm_70+).
    """
    if not torch.cuda.is_available():
        return False
    try:
        t = torch.zeros(1, device="cuda", dtype=torch.float32)
        _ = (t + 1.0).item()
        torch.cuda.synchronize()
        return True
    except Exception:
        return False


def resolve_device(spec: str) -> str:
    s = spec.lower().strip()
    if s == "auto":
        if _cuda_usable_for_compute():
            return "cuda"
        if torch.cuda.is_available():
            warnings.warn(
                "CUDA is visible but kernels fail on this GPU with the installed PyTorch "
                "(e.g. architecture too old for this wheel, such as sm_61 vs a sm_70+ build). "
                "Using CPU.",
                UserWarning,
                stacklevel=2,
            )
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if s == "cuda":
        if _cuda_usable_for_compute():
            return "cuda"
        warnings.warn(
            "CUDA requested but this GPU is not usable with the installed PyTorch build; using CPU.",
            UserWarning,
            stacklevel=2,
        )
        return "cpu"
    return spec


def _safe_model_name(model_name: str) -> str:
    return model_name.replace("/", "_")


def _safe_concept(name: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^\w\-]+", "_", name.strip().replace(" ", "_"))).strip("_")


def _safe_tokens(tokens: Sequence[str]) -> List[str]:
    out: List[str] = []
    for t in tokens:
        if t is None: continue
        if not isinstance(t, str): t = str(t)
        out.append(t.replace("\r", "\\r").replace("\n", "\\n"))
    return out


# ---------------------------------------------------------------------------
# Path & IO Management
# ---------------------------------------------------------------------------

_LAYER_DIR_RE = re.compile(r"^layer_(\d+)$")


def sorted_numeric_layer_dirs(results_dir: Path) -> List[Tuple[int, Path]]:
    """
    Subdirectories of ``results_dir`` named ``layer_<integer>`` only.

    Skips files (e.g. ``layer_concept_trends.png``) and non-numeric names
    (e.g. ``layer_concept``) that would otherwise match a naive ``layer_*`` glob.
    """
    found: List[Tuple[int, Path]] = []
    for p in results_dir.iterdir():
        if not p.is_dir():
            continue
        m = _LAYER_DIR_RE.match(p.name)
        if m:
            found.append((int(m.group(1)), p))
    found.sort(key=lambda t: t[0])
    return found
