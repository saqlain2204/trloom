"""Construct TRL trainers from a :class:`~trloom.config.schema.FineTuneConfig`."""

from __future__ import annotations

import inspect
import logging
from typing import Any

from trloom.config.schema import FineTuneConfig, ModelConfig
from trloom.data.dataset import load_train_eval_datasets
from trloom.rewards import resolve_reward_funcs
from trloom.trainers.registry import TrainerSpec, get_trainer_spec

logger = logging.getLogger(__name__)


def _to_trl_model_config(model: ModelConfig) -> Any:
    import dataclasses

    from trl import ModelConfig as TrlModelConfig

    fields = {f.name for f in dataclasses.fields(TrlModelConfig)}
    payload = model.model_dump(exclude_none=False)
    # trust_remote_code is often on training args in newer TRL, not ModelConfig
    filtered = {k: v for k, v in payload.items() if k in fields}
    return TrlModelConfig(**filtered)


def _filter_kwargs_for_callable(fn: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return kwargs
    params = signature.parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kwargs
    allowed = set(params)
    return {k: v for k, v in kwargs.items() if k in allowed}


def _build_training_args(spec: TrainerSpec, config: FineTuneConfig) -> Any:
    training_kwargs = dict(config.training)
    training_kwargs.setdefault("output_dir", "./outputs")
    if config.seed is not None:
        training_kwargs.setdefault("seed", config.seed)
    if config.hub_model_id:
        training_kwargs.setdefault("hub_model_id", config.hub_model_id)
    if config.push_to_hub:
        training_kwargs.setdefault("push_to_hub", True)

    # Newer TRL configs accept trust_remote_code
    try:
        signature = inspect.signature(spec.config_cls)
        if "trust_remote_code" in signature.parameters:
            training_kwargs.setdefault("trust_remote_code", config.model.trust_remote_code)
    except (TypeError, ValueError):
        pass

    filtered = _filter_kwargs_for_callable(spec.config_cls, training_kwargs)
    dropped = sorted(set(training_kwargs) - set(filtered))
    if dropped:
        logger.warning(
            "Ignoring training keys unsupported by %s: %s",
            spec.config_name,
            ", ".join(dropped),
        )
    return spec.config_cls(**filtered)


def _prepare_model_init_kwargs(training_args: Any, model_args: Any, config: FineTuneConfig) -> None:
    init_kwargs = {
        "revision": getattr(model_args, "model_revision", "main"),
        "trust_remote_code": config.model.trust_remote_code,
        "attn_implementation": getattr(model_args, "attn_implementation", None),
        "dtype": getattr(model_args, "dtype", None),
    }
    # Drop Nones
    init_kwargs = {k: v for k, v in init_kwargs.items() if v is not None}
    if hasattr(training_args, "model_init_kwargs"):
        existing = getattr(training_args, "model_init_kwargs") or {}
        if isinstance(existing, dict):
            training_args.model_init_kwargs = {**init_kwargs, **existing}
        else:
            training_args.model_init_kwargs = init_kwargs


def build_trainer(config: FineTuneConfig) -> Any:
    """Build a configured TRL trainer instance (without running training)."""
    from trl import get_peft_config, get_quantization_config

    spec = get_trainer_spec(config.method)
    if spec.experimental:
        logger.warning(
            "Method '%s' uses experimental TRL API (%s). APIs may change without notice.",
            config.method,
            spec.trainer_name,
        )

    model_args = _to_trl_model_config(config.model)
    training_args = _build_training_args(spec, config)
    _prepare_model_init_kwargs(training_args, model_args, config)

    train_dataset, eval_dataset = load_train_eval_datasets(config.dataset)

    # Disable eval dataset when strategy is "no"
    eval_strategy = getattr(training_args, "eval_strategy", None) or getattr(
        training_args, "evaluation_strategy", None
    )
    if eval_strategy in {None, "no"}:
        eval_dataset = None

    trainer_kwargs: dict[str, Any] = {
        "model": model_args.model_name_or_path,
        "args": training_args,
        "train_dataset": train_dataset,
        **dict(config.trainer_kwargs),
    }
    if eval_dataset is not None:
        trainer_kwargs.setdefault("eval_dataset", eval_dataset)

    peft_config = get_peft_config(model_args)
    if peft_config is not None:
        trainer_kwargs.setdefault("peft_config", peft_config)

    quantization_config = get_quantization_config(model_args)
    if quantization_config is not None:
        # Supported by newer TRL trainers; filtered below if unsupported
        trainer_kwargs.setdefault("quantization_config", quantization_config)

    reward_funcs = resolve_reward_funcs(config.reward_funcs)
    if reward_funcs is not None:
        trainer_kwargs.setdefault("reward_funcs", reward_funcs)

    filtered_kwargs = _filter_kwargs_for_callable(spec.trainer_cls.__init__, trainer_kwargs)
    dropped = sorted(set(trainer_kwargs) - set(filtered_kwargs))
    if dropped:
        logger.warning(
            "Ignoring trainer kwargs unsupported by %s: %s",
            spec.trainer_name,
            ", ".join(dropped),
        )

    logger.info(
        "Building %s with %s (method=%s)",
        spec.trainer_name,
        spec.config_name,
        spec.method,
    )
    return spec.trainer_cls(**filtered_kwargs)
