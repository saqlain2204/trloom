"""End-to-end fine-tuning job orchestration."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

from trloom.config.loader import load_config
from trloom.config.schema import FineTuneConfig
from trloom.logging_utils import finish_wandb, maybe_init_wandb
from trloom.trainers.builder import build_trainer
from trloom.trainers.registry import list_trainers
from trloom.user_code import prepare_local_user_code

logger = logging.getLogger(__name__)

BackendName = Literal["local", "modal", "kaggle"]


class FineTuneJob:
    """Load a YAML config, build a TRL trainer, and run training end-to-end."""

    def __init__(self, config: FineTuneConfig) -> None:
        self.config = config
        self.trainer: Any | None = None
        self._wandb_run: Any | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> FineTuneJob:
        """Create a job from a YAML configuration file."""
        return cls(load_config(path))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FineTuneJob:
        """Create a job from an in-memory configuration dictionary."""
        return cls(load_config(data))

    def build(self) -> Any:
        """Construct (but do not train) the underlying TRL trainer."""
        prepare_local_user_code(self.config)
        self._wandb_run = maybe_init_wandb(self.config)
        self.trainer = build_trainer(self.config)
        return self.trainer

    def train(self) -> Any:
        """Build the trainer if needed and run ``trainer.train()``."""
        if self.trainer is None:
            self.build()
        assert self.trainer is not None

        logger.info("Starting training (method=%s)", self.config.method)
        train_result = self.trainer.train()
        self._save_and_push()
        return train_result

    def run(self) -> Any:
        """Alias for :meth:`train` — run the fine-tuning job end-to-end."""
        try:
            return self.train()
        finally:
            finish_wandb(self._wandb_run)

    def _save_and_push(self) -> None:
        assert self.trainer is not None
        output_dir = self.config.resolved_output_dir()
        output_dir.mkdir(parents=True, exist_ok=True)

        save_fn = getattr(self.trainer, "save_model", None)
        if callable(save_fn):
            save_fn(str(output_dir))
            logger.info("Saved model to %s", output_dir)
        else:  # pragma: no cover
            logger.warning("Trainer has no save_model(); skipping save.")

        should_push = self.config.push_to_hub or bool(
            self.config.training.get("push_to_hub")
        )
        if should_push and hasattr(self.trainer, "push_to_hub"):
            kwargs: dict[str, Any] = {}
            if self.config.hub_model_id:
                kwargs["model_id"] = self.config.hub_model_id
            try:
                self.trainer.push_to_hub(**kwargs)
            except TypeError:
                self.trainer.push_to_hub()
            logger.info("Pushed model to the Hugging Face Hub.")


def resolve_backend(
    config: FineTuneConfig,
    *,
    use_modal: bool | None = None,
    use_kaggle: bool | None = None,
) -> BackendName:
    """Resolve the execution backend from CLI overrides and YAML flags.

    Explicit CLI flags win. ``use_modal`` / ``use_kaggle`` cannot both be True.
    """
    if use_modal and use_kaggle:
        raise ValueError("Choose at most one of use_modal / use_kaggle.")

    if use_modal is True:
        return "modal"
    if use_kaggle is True:
        return "kaggle"
    if use_modal is False and use_kaggle is False:
        return "local"
    if use_modal is False:
        # Explicitly disable Modal; still allow Kaggle from YAML.
        return "kaggle" if config.kaggle.enabled else "local"
    if use_kaggle is False:
        return "modal" if config.modal.enabled else "local"

    return config.resolved_backend()


def run_from_yaml(
    path: str | Path,
    *,
    use_modal: bool | None = None,
    use_kaggle: bool | None = None,
) -> Any:
    """Load a YAML config and run the fine-tuning job.

    Parameters
    ----------
    path:
        Path to a TRLoom YAML configuration file.
    use_modal:
        If ``True``, force Modal execution. If ``False``, never use Modal.
        If ``None``, follow ``modal.enabled`` in the config (subject to Kaggle).
    use_kaggle:
        If ``True``, force Kaggle execution. If ``False``, never use Kaggle.
        If ``None``, follow ``kaggle.enabled`` in the config (subject to Modal).
    """
    config = load_config(path)
    backend = resolve_backend(config, use_modal=use_modal, use_kaggle=use_kaggle)

    if backend == "modal":
        from trloom.modal_support import run_on_modal as _run_on_modal

        return _run_on_modal(path)

    if backend == "kaggle":
        from trloom.kaggle_support import run_on_kaggle as _run_on_kaggle

        return _run_on_kaggle(path)

    return FineTuneJob(config).run()


def available_methods(*, include_experimental: bool = True) -> list[str]:
    """List training methods supported by the installed TRL version."""
    return list_trainers(include_experimental=include_experimental)
