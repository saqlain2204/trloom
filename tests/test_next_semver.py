"""Tests for SemVer tag bump helper used by the release workflow."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "next_semver.py"

sys.path.insert(0, str(ROOT / "scripts"))
from next_semver import format_tag, latest_version, next_version  # noqa: E402


def test_latest_version_from_tags() -> None:
    assert latest_version(["v0.1.0", "v0.1.1", "v0.2.0"]) == (0, 2, 0)
    assert latest_version([]) is None
    assert latest_version(["notes", "v1.0.0-rc1"]) is None


def test_next_version_patch_minor_major() -> None:
    assert next_version((0, 1, 1), "patch") == (0, 1, 2)
    assert next_version((0, 1, 1), "minor") == (0, 2, 0)
    assert next_version((0, 1, 1), "major") == (1, 0, 0)
    assert next_version(None, "patch") == (0, 1, 0)


def test_format_tag() -> None:
    assert format_tag((1, 2, 3)) == "v1.2.3"


def test_cli_patch_bump() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--bump", "patch", "--tags", "v0.1.1"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "v0.1.2"


def test_cli_minor() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--bump", "minor", "--tags", "v0.1.1"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "v0.2.0"


def test_cli_major() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--bump", "major", "--tags", "v0.1.1"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "v1.0.0"
