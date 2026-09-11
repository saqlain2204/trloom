"""Load and validate TRLoom YAML configuration files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from trloom.config.schema import FineTuneConfig


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if data is None:
        raise ValueError(f"Config file is empty: {path}")
    if not isinstance(data, dict):
        raise TypeError(f"Config root must be a mapping, got {type(data).__name__}: {path}")
    return data


def load_config_dict(source: str | Path | dict[str, Any]) -> dict[str, Any]:
    """Return a raw configuration dictionary from a path or mapping."""
    if isinstance(source, dict):
        return dict(source)
    path = Path(source).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    return _read_yaml(path)


def load_config(source: str | Path | dict[str, Any]) -> FineTuneConfig:
    """Load and validate a :class:`FineTuneConfig` from YAML or a dict."""
    data = load_config_dict(source)
    return FineTuneConfig.model_validate(data)
