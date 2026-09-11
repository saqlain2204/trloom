"""CLI smoke tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from trloom.cli import main


def test_cli_version(capsys: Any) -> None:
    assert main(["--version"]) == 0
    assert capsys.readouterr().out.strip()


def test_cli_methods(capsys: Any) -> None:
    assert main(["methods"]) == 0
    out = capsys.readouterr().out
    assert "sft" in out


def test_cli_validate_example() -> None:
    root = Path(__file__).resolve().parents[1]
    config = root / "examples" / "sft_hub.yaml"
    assert main(["validate", str(config)]) == 0
