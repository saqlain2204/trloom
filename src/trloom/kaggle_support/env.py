"""Detect and adapt to the Kaggle runtime environment."""

from __future__ import annotations

import os
from pathlib import Path


def is_kaggle_environment() -> bool:
    """Return True when running inside a Kaggle Notebook / kernel."""
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        return True
    if os.environ.get("KAGGLE_URL_BASE"):
        return True
    return Path("/kaggle/working").is_dir() and Path("/kaggle/input").is_dir()


def kaggle_working_dir() -> Path:
    """Return the writable Kaggle working directory (or a local fallback)."""
    candidate = Path("/kaggle/working")
    if candidate.is_dir():
        return candidate
    return Path.cwd()


def kaggle_gpu_available() -> bool:
    """Best-effort check for a CUDA GPU (common on Kaggle GPU sessions)."""
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False
