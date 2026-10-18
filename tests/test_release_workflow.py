"""Smoke checks for the single Release publishing pipeline."""

from __future__ import annotations

from pathlib import Path

import yaml


def test_release_workflow_is_single_dispatch_pipeline() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / ".github" / "workflows" / "release.yml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert data["name"] == "Release"
    # PyYAML 1.1 treats bare key `on` as boolean True.
    on = data.get("on", data.get(True))
    assert on is not None
    assert "workflow_dispatch" in on
    # Avoid double publishes: no tag-push trigger alongside dispatch.
    assert "push" not in on

    jobs = data["jobs"]
    assert list(jobs) == ["prepare", "build", "publish"]
    assert jobs["build"]["needs"] == "prepare"
    assert jobs["publish"]["needs"] == ["prepare", "build"]


def test_cut_release_workflow_removed() -> None:
    root = Path(__file__).resolve().parents[1]
    assert not (root / ".github" / "workflows" / "cut-release.yml").exists()
