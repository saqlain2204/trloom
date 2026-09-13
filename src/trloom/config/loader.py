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


def _resolve_user_code_paths(config: FineTuneConfig, base_dir: Path) -> FineTuneConfig:
    """Resolve relative ``user_code`` entries against the YAML file directory."""
    if not config.user_code:
        return config
    resolved: list[str] = []
    for raw in config.user_code:
        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            candidate = (base_dir / candidate).resolve()
        else:
            candidate = candidate.resolve()
        resolved.append(str(candidate))
    return config.model_copy(update={"user_code": resolved})


def load_config(source: str | Path | dict[str, Any]) -> FineTuneConfig:
    """Load and validate a :class:`FineTuneConfig` from YAML or a dict.

    When loading from a file, relative ``user_code`` paths are resolved against
    the YAML file's directory so Modal / local runs find the same modules.
    """
    data = load_config_dict(source)
    config = FineTuneConfig.model_validate(data)
    if not isinstance(source, dict):
        base_dir = Path(source).expanduser().resolve().parent
        config = _resolve_user_code_paths(config, base_dir)
    return config
