"""Tests for dataset loading helpers."""

from __future__ import annotations

import json
from pathlib import Path

from trloom.config.schema import DatasetConfig, DatasetSourceConfig
from trloom.data.dataset import load_train_eval_datasets


def test_load_local_jsonl(tmp_path: Path) -> None:
    data_path = tmp_path / "train.jsonl"
    rows = [
        {"text": "hello"},
        {"text": "world"},
    ]
    data_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    config = DatasetConfig(path=str(data_path), train_split="train", eval_split=None)
    train, eval_ds = load_train_eval_datasets(config)
    assert eval_ds is None
    assert len(train) == 2
    assert train[0]["text"] == "hello"


def test_column_rename(tmp_path: Path) -> None:
    data_path = tmp_path / "train.jsonl"
    data_path.write_text(json.dumps({"prompt": "hi"}) + "\n", encoding="utf-8")
    config = DatasetConfig(
        path=str(data_path),
        train_split="train",
        eval_split=None,
        columns={"prompt": "text"},
    )
    train, _ = load_train_eval_datasets(config)
    assert "text" in train.column_names
    assert "prompt" not in train.column_names


def test_mixture_concatenate(tmp_path: Path) -> None:
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    a.write_text(json.dumps({"text": "a"}) + "\n", encoding="utf-8")
    b.write_text(json.dumps({"text": "b"}) + "\n", encoding="utf-8")

    config = DatasetConfig(
        datasets=[
            DatasetSourceConfig(path=str(a)),
            DatasetSourceConfig(path=str(b)),
        ],
        train_split="train",
        eval_split=None,
    )
    train, _ = load_train_eval_datasets(config)
    assert len(train) == 2
