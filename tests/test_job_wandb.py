"""Tests for W&B helpers and job utilities."""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import MagicMock, patch

from trloom.config import load_config
from trloom.job import FineTuneJob
from trloom.logging_utils import apply_wandb_settings, maybe_init_wandb


def _base_config(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "method": "sft",
        "model": {"model_name_or_path": "sshleifer/tiny-gpt2"},
        "dataset": {"path": "trl-lib/Capybara"},
        "training": {"output_dir": "./outputs/test", "report_to": "none"},
    }
    data.update(overrides)
    return data


def test_apply_wandb_sets_report_to() -> None:
    config = load_config(
        _base_config(wandb={"enabled": True, "project": "proj", "mode": "offline"})
    )
    kwargs = apply_wandb_settings(config)
    assert config.training["report_to"] == "wandb"
    assert kwargs["project"] == "proj"
    assert os.environ.get("WANDB_MODE") == "offline"


def test_apply_wandb_disabled_noop() -> None:
    config = load_config(_base_config(wandb={"enabled": False}))
    kwargs = apply_wandb_settings(config)
    assert kwargs == {}
    assert config.training["report_to"] == "none"


def test_maybe_init_wandb() -> None:
    mock_wandb = MagicMock()
    mock_wandb.init.return_value = MagicMock(name="run")
    config = load_config(_base_config(wandb={"enabled": True, "project": "proj"}))
    with patch.dict("sys.modules", {"wandb": mock_wandb}):
        run = maybe_init_wandb(config)
    assert run is not None
    mock_wandb.init.assert_called_once()


def test_job_from_dict_and_yaml(tmp_path: Any) -> None:
    import yaml

    data = _base_config()
    job = FineTuneJob.from_dict(data)
    assert job.config.method == "sft"

    path = tmp_path / "job.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    job2 = FineTuneJob.from_yaml(path)
    assert job2.config.model.model_name_or_path == "sshleifer/tiny-gpt2"
