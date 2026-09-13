"""Resolve TRL reward callables by name for online methods."""

from __future__ import annotations

import importlib
import logging
from collections.abc import Callable, Sequence
from typing import Any

from trloom.imports import resolve_callable

logger = logging.getLogger(__name__)


def _load_trl_rewards_module() -> Any | None:
    for module_name in ("trl.rewards", "trl.trainer.utils"):
        try:
            return importlib.import_module(module_name)
        except ImportError:
            continue
    return None


def resolve_reward_funcs(names: Sequence[str] | None) -> list[Callable[..., Any]] | None:
    """Resolve reward function names to callables from ``trl.rewards`` when available.

    Names may be:
    - an attribute on ``trl.rewards`` (e.g. ``accuracy_reward``)
    - a fully-qualified import path (``package.module:func`` or ``package.module.func``)
    """
    if not names:
        return None

    rewards_module = _load_trl_rewards_module()
    resolved: list[Callable[..., Any]] = []

    for name in names:
        func = _resolve_one(name, rewards_module)
        resolved.append(func)
        logger.info("Resolved reward function '%s' -> %s", name, getattr(func, "__name__", func))

    return resolved


def _resolve_one(name: str, rewards_module: Any | None) -> Callable[..., Any]:
    # Prefer explicit import paths (colon form or dotted path not found on trl.rewards).
    if ":" in name:
        return resolve_callable(name, label="reward function")

    if "." in name and rewards_module is not None and not hasattr(rewards_module, name):
        try:
            return resolve_callable(name, label="reward function")
        except (ImportError, AttributeError, TypeError, ValueError):
            pass

    if rewards_module is not None and hasattr(rewards_module, name):
        func = getattr(rewards_module, name)
        if not callable(func):
            raise TypeError(f"trl.rewards.{name} is not callable.")
        return func

    # Final attempt: treat as module.attr
    if "." in name:
        return resolve_callable(name, label="reward function")

    raise ValueError(
        f"Could not resolve reward function '{name}'. "
        "Provide a trl.rewards attribute name or a fully-qualified path like "
        "'my_package.rewards:my_reward'."
    )
