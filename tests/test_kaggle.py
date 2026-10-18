"""Tests for Kaggle backend detection, notebook generation, and dispatch."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from trloom.cli import main
from trloom.config import load_config
from trloom.job import resolve_backend
from trloom.kaggle_support.env import is_kaggle_environment, kaggle_working_dir
from trloom.kaggle_support.runner import (
    build_kaggle_notebook,
    prepare_kaggle_config,
    write_kaggle_notebook,
)


def _base_cfg(**kaggle: Any) -> dict[str, Any]:
    return {
        "method": "sft",
        "model": {"model_name_or_path": "sshleifer/tiny-gpt2"},
        "dataset": {"path": "trl-lib/Capybara", "eval_split": None},
        "kaggle": kaggle,
    }


def test_is_kaggle_environment_false_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KAGGLE_KERNEL_RUN_TYPE", raising=False)
    monkeypatch.delenv("KAGGLE_URL_BASE", raising=False)
    # Do not rely on /kaggle paths existing on Windows CI.
    assert is_kaggle_environment() in {True, False}


def test_is_kaggle_environment_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Interactive")
    assert is_kaggle_environment() is True


def test_kaggle_working_dir_fallback() -> None:
    path = kaggle_working_dir()
    assert path.exists() or path == Path.cwd()


def test_modal_and_kaggle_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="only one remote backend"):
        load_config(
            {
                **_base_cfg(enabled=True),
                "modal": {"enabled": True},
            }
        )


def test_resolved_backend_kaggle() -> None:
    config = load_config(_base_cfg(enabled=True))
    assert config.resolved_backend() == "kaggle"


def test_resolve_backend_cli_overrides() -> None:
    config = load_config(_base_cfg(enabled=True, mode="notebook"))
    assert resolve_backend(config, use_modal=True, use_kaggle=False) == "modal"
    assert resolve_backend(config, use_modal=False, use_kaggle=True) == "kaggle"
    assert resolve_backend(config, use_modal=False, use_kaggle=False) == "local"
    with pytest.raises(ValueError, match="at most one"):
        resolve_backend(config, use_modal=True, use_kaggle=True)


def test_prepare_kaggle_config_disables_remote() -> None:
    config = load_config(_base_cfg(enabled=True, output_dir="/kaggle/working/out"))
    prepared = prepare_kaggle_config(config)
    assert prepared.kaggle.enabled is False
    assert prepared.modal.enabled is False
    assert prepared.training["output_dir"] == "/kaggle/working/out"


def test_write_kaggle_notebook_embeds_user_code(tmp_path: Path) -> None:
    fmt = tmp_path / "formatters.py"
    fmt.write_text("def to_text(example):\n    return example\n", encoding="utf-8")
    data = tmp_path / "train.jsonl"
    data.write_text(json.dumps({"text": "hi"}) + "\n", encoding="utf-8")
    cfg_path = tmp_path / "job.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "method": "sft",
                "model": {"model_name_or_path": "sshleifer/tiny-gpt2"},
                "dataset": {
                    "path": str(data),
                    "train_split": "train",
                    "eval_split": None,
                    "map_fn": "formatters:to_text",
                },
                "user_code": [str(fmt)],
                "kaggle": {
                    "enabled": True,
                    "mode": "notebook",
                    "push": False,
                    "username": "demo_user",
                    "kernel_slug": "trloom-demo",
                    "accelerator": "gpu",
                },
            }
        ),
        encoding="utf-8",
    )

    out_dir = tmp_path / "kaggle_out"
    notebook_path = out_dir / "job.ipynb"
    dest = write_kaggle_notebook(cfg_path, destination=notebook_path)
    assert dest == notebook_path
    assert notebook_path.is_file()

    nb = json.loads(notebook_path.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    assert len(nb["cells"]) >= 2
    blob = json.dumps(nb)
    assert "formatters.py" in blob
    assert "def to_text" in blob
    assert "FineTuneJob" in blob

    meta_path = out_dir / "kernel-metadata.json"
    assert meta_path.is_file()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["id"] == "demo_user/trloom-demo"
    assert meta["enable_gpu"] is True
    assert meta["code_file"] == "job.ipynb"


def test_build_kaggle_notebook_git_install() -> None:
    config = load_config(
        _base_cfg(
            enabled=True,
            install_source="git",
            git_url="git+https://github.com/saqlain2204/trloom.git@main",
            accelerator="tpu",
        )
    )
    nb = build_kaggle_notebook(config, config_yaml="method: sft\n", user_code_bundle={})
    source = "".join(nb["cells"][1]["source"])
    assert "git+https://github.com/saqlain2204/trloom.git@main" in source


def test_example_sft_kaggle_validates() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "examples" / "sft_kaggle.yaml")
    assert config.kaggle.enabled is True
    assert config.kaggle.accelerator == "gpu"
    assert config.resolved_backend() == "kaggle"


def test_cli_kaggle_notebook(tmp_path: Path, capsys: Any) -> None:
    root = Path(__file__).resolve().parents[1]
    example = root / "examples" / "sft_kaggle.yaml"
    out = tmp_path / "generated.ipynb"
    assert main(["kaggle-notebook", str(example), "-o", str(out)]) == 0
    printed = capsys.readouterr().out.strip()
    assert Path(printed) == out
    assert out.is_file()
    assert (tmp_path / "kernel-metadata.json").is_file()


def test_cli_validate_reports_kaggle(capsys: Any) -> None:
    root = Path(__file__).resolve().parents[1]
    assert main(["validate", str(root / "examples" / "sft_kaggle.yaml")]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["kaggle_enabled"] is True
    assert payload["backend"] == "kaggle"


def test_cli_run_rejects_multiple_backend_flags() -> None:
    root = Path(__file__).resolve().parents[1]
    example = root / "examples" / "sft_local.yaml"
    with pytest.raises(SystemExit) as excinfo:
        main(["run", str(example), "--modal", "--kaggle"])
    assert excinfo.value.code == 2
