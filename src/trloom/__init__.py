"""TRLoom: YAML-driven end-to-end fine-tuning on top of Hugging Face TRL."""

from __future__ import annotations

from trloom.config import FineTuneConfig, load_config
from trloom.job import FineTuneJob, available_methods, run_from_yaml
from trloom.trainers import list_trainers

__all__ = [
    "FineTuneConfig",
    "FineTuneJob",
    "__version__",
    "available_methods",
    "list_trainers",
    "load_config",
    "run_from_yaml",
]


def _resolve_version() -> str:
    try:
        from trloom._version import __version__ as built

        return built
    except ImportError:
        pass
    try:
        from importlib.metadata import PackageNotFoundError, version

        return version("trloom")
    except PackageNotFoundError:
        return "0.0.0"


__version__ = _resolve_version()
