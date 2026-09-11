"""Dynamic registry of TRL trainers and their config classes."""

from __future__ import annotations

import importlib
import inspect
import logging
import pkgutil
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)

# Canonical method aliases → Trainer class name (without the Trainer suffix key)
_METHOD_ALIASES: dict[str, str] = {
    "sft": "sft",
    "supervised": "sft",
    "supervised_fine_tuning": "sft",
    "dpo": "dpo",
    "direct_preference_optimization": "dpo",
    "grpo": "grpo",
    "group_relative_policy_optimization": "grpo",
    "kto": "kto",
    "reward": "reward",
    "rm": "reward",
    "reward_model": "reward",
    "rloo": "rloo",
    "reinforce_leave_one_out": "rloo",
    "orpo": "orpo",
    "cpo": "cpo",
    "bco": "bco",
    "xpo": "xpo",
    "ppo": "ppo",
    "online_dpo": "online_dpo",
    "onlinedpo": "online_dpo",
    "nash_md": "nash_md",
    "nashmd": "nash_md",
    "prm": "prm",
    "gkd": "gkd",
    "minillm": "minillm",
    "distillation": "distillation",
    "tpo": "tpo",
    "a2po": "a2po",
    "async_grpo": "async_grpo",
    "gmpo": "gmpo",
    "gold": "gold",
    "sdft": "sdft",
    "sdpo": "sdpo",
    "ssd": "ssd",
    "iw_opd": "iw_opd",
    "iwopd": "iw_opd",
    "async_distillation": "async_distillation",
}


@dataclass(frozen=True)
class TrainerSpec:
    """Resolved TRL Trainer + Config pair for a training method."""

    method: str
    trainer_cls: type
    config_cls: type
    experimental: bool = False

    @property
    def trainer_name(self) -> str:
        return self.trainer_cls.__name__

    @property
    def config_name(self) -> str:
        return self.config_cls.__name__


def _camel_to_snake(name: str) -> str:
    import re

    # Standard CamelCase / acronym splitting (DPO -> dpo, MiniLLM -> mini_llm)
    stepped = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    stepped = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", stepped)
    return stepped.replace("__", "_").strip("_").lower()


def _method_key_from_trainer_name(trainer_name: str) -> str | None:
    if trainer_name.startswith("_") or not trainer_name.endswith("Trainer"):
        return None
    base = trainer_name[: -len("Trainer")]
    if not base or base.startswith("_"):
        return None
    return _camel_to_snake(base)


def _iter_trl_modules() -> list[tuple[Any, bool]]:
    """Import TRL modules that may expose Trainer/Config classes."""
    modules: list[tuple[Any, bool]] = []
    import trl

    modules.append((trl, False))
    try:
        trainer_pkg = importlib.import_module("trl.trainer")
        modules.append((trainer_pkg, False))
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Could not import trl.trainer: %s", exc)

    try:
        import os

        os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")
        experimental = importlib.import_module("trl.experimental")
        modules.append((experimental, True))
        if hasattr(experimental, "__path__"):
            for module_info in pkgutil.walk_packages(
                experimental.__path__,
                prefix=experimental.__name__ + ".",
            ):
                try:
                    mod = importlib.import_module(module_info.name)
                    modules.append((mod, True))
                except Exception as exc:  # pragma: no cover
                    logger.debug("Skipping experimental module %s: %s", module_info.name, exc)
    except Exception as exc:
        logger.debug("trl.experimental unavailable: %s", exc)

    return modules


def _collect_classes(modules: list[tuple[Any, bool]]) -> dict[str, tuple[type, bool]]:
    found: dict[str, tuple[type, bool]] = {}
    for module, experimental in modules:
        # TRL uses lazy modules where exports appear in dir() but not vars().
        for name in dir(module):
            if name.startswith("_"):
                continue
            try:
                obj = getattr(module, name)
            except Exception:  # pragma: no cover - defensive
                continue
            if not inspect.isclass(obj):
                continue
            # Prefer first discovery; stable trl package wins over experimental
            if name in found and found[name][1] and not experimental:
                found[name] = (obj, experimental)
            elif name not in found:
                found[name] = (obj, experimental)
    return found


@lru_cache(maxsize=1)
def _build_registry() -> dict[str, TrainerSpec]:
    classes = _collect_classes(_iter_trl_modules())
    registry: dict[str, TrainerSpec] = {}

    for class_name, (cls, experimental) in classes.items():
        method = _method_key_from_trainer_name(class_name)
        if method is None:
            continue
        config_name = f"{class_name[: -len('Trainer')]}Config"
        config_entry = classes.get(config_name)
        if config_entry is None:
            logger.debug("No config class found for %s (expected %s)", class_name, config_name)
            continue
        config_cls, config_experimental = config_entry
        spec = TrainerSpec(
            method=method,
            trainer_cls=cls,
            config_cls=config_cls,
            experimental=experimental or config_experimental,
        )
        # Prefer non-experimental when both exist
        existing = registry.get(method)
        if existing is None or (existing.experimental and not spec.experimental):
            registry[method] = spec

    if not registry:
        raise RuntimeError(
            "No TRL trainers discovered. Ensure the `trl` package is installed and importable."
        )
    return registry


def list_trainers(*, include_experimental: bool = True) -> list[str]:
    """Return sorted method names available from the installed TRL version."""
    registry = _build_registry()
    methods = [
        method
        for method, spec in registry.items()
        if include_experimental or not spec.experimental
    ]
    return sorted(methods)


def resolve_method_name(method: str) -> str:
    normalized = method.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized.endswith("_trainer"):
        normalized = normalized[: -len("_trainer")]
    return _METHOD_ALIASES.get(normalized, normalized)


def get_trainer_spec(method: str) -> TrainerSpec:
    """Resolve a training method string to a :class:`TrainerSpec`."""
    key = resolve_method_name(method)
    registry = _build_registry()
    if key in registry:
        return registry[key]

    # Fallback: compare compact forms (a2po == a2_po)
    compact = key.replace("_", "")
    for name, spec in registry.items():
        if name.replace("_", "") == compact:
            return spec

    available = ", ".join(list_trainers())
    raise KeyError(
        f"Unknown training method '{method}' (resolved to '{key}'). "
        f"Available methods for this TRL install: {available}"
    )


def clear_registry_cache() -> None:
    """Clear the cached trainer registry (useful in tests)."""
    _build_registry.cache_clear()
