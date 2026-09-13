"""Resolve import-path references to Python callables."""

from __future__ import annotations

import importlib
import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


def parse_import_spec(spec: str) -> tuple[str, str]:
    """Split ``pkg.mod:attr`` or ``pkg.mod.attr`` into ``(module, attr)``."""
    value = spec.strip()
    if not value:
        raise ValueError("Import spec must be a non-empty string.")

    if ":" in value:
        module_name, attr = value.split(":", 1)
        module_name, attr = module_name.strip(), attr.strip()
        if not module_name or not attr:
            raise ValueError(
                f"Invalid import spec '{spec}'. Expected 'package.module:attribute'."
            )
        return module_name, attr

    if "." not in value:
        raise ValueError(
            f"Invalid import spec '{spec}'. Expected 'package.module:attribute' "
            "or 'package.module.attribute'."
        )

    module_name, attr = value.rsplit(".", 1)
    return module_name, attr


def resolve_callable(spec: str, *, label: str = "callable") -> Callable[..., Any]:
    """Import and return a callable from an import-path string.

    Accepted forms:
    - ``package.module:function``
    - ``package.module.function``
    """
    module_name, attr = parse_import_spec(spec)
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ImportError(
            f"Could not import module '{module_name}' while resolving {label} '{spec}'. "
            "If you are running on Modal / a remote provider, ensure the module is listed "
            "under `user_code` or referenced so TRLoom can bundle it with the job."
        ) from exc

    try:
        func = getattr(module, attr)
    except AttributeError as exc:
        raise AttributeError(
            f"Module '{module_name}' has no attribute '{attr}' "
            f"(while resolving {label} '{spec}')."
        ) from exc

    if not callable(func):
        raise TypeError(f"{label.capitalize()} '{spec}' resolved to a non-callable object.")

    logger.info("Resolved %s '%s' -> %s", label, spec, getattr(func, "__name__", func))
    return func
