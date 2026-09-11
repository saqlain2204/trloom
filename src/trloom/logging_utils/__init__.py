"""Weights & Biases helpers."""

from __future__ import annotations

import logging
import os
from typing import Any

from trloom.config.schema import FineTuneConfig, WandbConfig

logger = logging.getLogger(__name__)


def apply_wandb_settings(config: FineTuneConfig) -> dict[str, Any]:
    """Apply W&B settings from config to the environment / training report_to.

    Returns a dict of kwargs suitable for ``wandb.init`` (may be empty).
    Mutates ``config.training`` in-place to set ``report_to`` when enabled.
    """
    wandb_cfg = config.wandb
    training = config.training

    if not wandb_cfg.enabled:
        # Do not force-disable if user already set report_to explicitly
        return {}

    if wandb_cfg.mode:
        os.environ["WANDB_MODE"] = wandb_cfg.mode
    if wandb_cfg.project:
        os.environ.setdefault("WANDB_PROJECT", wandb_cfg.project)
    if wandb_cfg.entity:
        os.environ.setdefault("WANDB_ENTITY", wandb_cfg.entity)
    if wandb_cfg.dir:
        os.environ.setdefault("WANDB_DIR", wandb_cfg.dir)

    report_to = training.get("report_to")
    if report_to is None:
        training["report_to"] = "wandb"
    elif isinstance(report_to, str) and report_to != "wandb":
        if report_to in {"none", "None", ""}:
            training["report_to"] = "wandb"
        else:
            training["report_to"] = [report_to, "wandb"]
    elif isinstance(report_to, list) and "wandb" not in report_to:
        training["report_to"] = [*report_to, "wandb"]

    init_kwargs: dict[str, Any] = dict(wandb_cfg.init_kwargs)
    if wandb_cfg.project and "project" not in init_kwargs:
        init_kwargs["project"] = wandb_cfg.project
    if wandb_cfg.entity and "entity" not in init_kwargs:
        init_kwargs["entity"] = wandb_cfg.entity
    if wandb_cfg.run_name and "name" not in init_kwargs:
        init_kwargs["name"] = wandb_cfg.run_name
    if wandb_cfg.group and "group" not in init_kwargs:
        init_kwargs["group"] = wandb_cfg.group
    if wandb_cfg.tags and "tags" not in init_kwargs:
        init_kwargs["tags"] = list(wandb_cfg.tags)
    if wandb_cfg.notes and "notes" not in init_kwargs:
        init_kwargs["notes"] = wandb_cfg.notes
    if wandb_cfg.job_type and "job_type" not in init_kwargs:
        init_kwargs["job_type"] = wandb_cfg.job_type
    if wandb_cfg.dir and "dir" not in init_kwargs:
        init_kwargs["dir"] = wandb_cfg.dir
    if wandb_cfg.mode and "mode" not in init_kwargs:
        init_kwargs["mode"] = wandb_cfg.mode

    init_kwargs.setdefault("config", config.model_dump())
    return init_kwargs


def maybe_init_wandb(config: FineTuneConfig) -> Any | None:
    """Optionally call ``wandb.init`` when enabled. Returns the run or ``None``."""
    init_kwargs = apply_wandb_settings(config)
    if not config.wandb.enabled:
        return None

    try:
        import wandb
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "wandb is enabled in the config but the package is not installed. "
            "Install with: pip install 'trloom[wandb]'"
        ) from exc

    logger.info("Initializing Weights & Biases (project=%s)", init_kwargs.get("project"))
    return wandb.init(**init_kwargs)


def finish_wandb(run: Any | None) -> None:
    if run is None:
        return
    try:
        run.finish()
    except Exception as exc:  # pragma: no cover
        logger.warning("Failed to finish W&B run cleanly: %s", exc)


def wandb_config_summary(wandb_cfg: WandbConfig) -> dict[str, Any]:
    return {
        "enabled": wandb_cfg.enabled,
        "project": wandb_cfg.project,
        "entity": wandb_cfg.entity,
        "run_name": wandb_cfg.run_name,
        "mode": wandb_cfg.mode,
        "tags": list(wandb_cfg.tags),
    }
