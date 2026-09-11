"""Tests for Modal helpers (no remote calls)."""

from __future__ import annotations

from pathlib import Path

from trloom.config import load_config
from trloom.modal_support.runner import write_modal_entrypoint


def test_write_modal_entrypoint(tmp_path: Path) -> None:
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        """
method: sft
model:
  model_name_or_path: sshleifer/tiny-gpt2
dataset:
  path: trl-lib/Capybara
modal:
  enabled: true
  app_name: demo
""",
        encoding="utf-8",
    )
    dest = tmp_path / "entry.py"
    out = write_modal_entrypoint(cfg, destination=dest)
    assert out == dest
    text = dest.read_text(encoding="utf-8")
    assert "run_on_modal" in text
    assert "trloom" in text


def test_modal_config_roundtrip() -> None:
    config = load_config(
        {
            "method": "sft",
            "model": {"model_name_or_path": "sshleifer/tiny-gpt2"},
            "dataset": {"path": "trl-lib/Capybara"},
            "modal": {"enabled": True, "gpu": "A10G", "timeout": 3600},
        }
    )
    assert config.modal.enabled is True
    assert config.modal.gpu == "A10G"
    assert config.modal.timeout == 3600
