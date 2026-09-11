"""Tests for YAML config loading and validation."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from trloom.config import FineTuneConfig, load_config
from trloom.config.loader import load_config_dict


def test_load_config_from_dict() -> None:
    config = load_config(
        {
            "method": "SFT",
            "model": {"model_name_or_path": "sshleifer/tiny-gpt2"},
            "dataset": {"path": "trl-lib/Capybara"},
            "training": {"output_dir": "./out"},
        }
    )
    assert config.method == "sft"
    assert config.model.model_name_or_path == "sshleifer/tiny-gpt2"
    assert config.dataset.path == "trl-lib/Capybara"


def test_load_config_from_yaml_file(tmp_path: Path) -> None:
    path = tmp_path / "cfg.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "method": "dpo",
                "model": {"model_name_or_path": "sshleifer/tiny-gpt2"},
                "dataset": {"path": "Anthropic/hh-rlhf"},
                "wandb": {"enabled": True, "project": "demo"},
            }
        ),
        encoding="utf-8",
    )
    config = load_config(path)
    assert config.method == "dpo"
    assert config.wandb.enabled is True
    assert config.wandb.project == "demo"


def test_missing_dataset_raises() -> None:
    with pytest.raises(Exception):
        FineTuneConfig.model_validate(
            {
                "method": "sft",
                "model": {"model_name_or_path": "sshleifer/tiny-gpt2"},
                "dataset": {},
            }
        )


def test_quantization_conflict_raises() -> None:
    with pytest.raises(Exception):
        load_config(
            {
                "method": "sft",
                "model": {
                    "model_name_or_path": "sshleifer/tiny-gpt2",
                    "load_in_4bit": True,
                    "load_in_8bit": True,
                },
                "dataset": {"path": "trl-lib/Capybara"},
            }
        )


def test_reward_funcs_string_coerced() -> None:
    config = load_config(
        {
            "method": "grpo",
            "model": {"model_name_or_path": "sshleifer/tiny-gpt2"},
            "dataset": {"path": "trl-lib/Capybara"},
            "reward_funcs": "accuracy_reward",
        }
    )
    assert config.reward_funcs == ["accuracy_reward"]


def test_load_config_dict_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config_dict(tmp_path / "missing.yaml")


def test_example_configs_validate() -> None:
    root = Path(__file__).resolve().parents[1] / "examples"
    for path in root.glob("*.yaml"):
        config = load_config(path)
        assert config.method
        assert config.model.model_name_or_path
