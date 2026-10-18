"""Load training datasets from the Hugging Face Hub or local files."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from datasets import (
    Dataset,
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
    concatenate_datasets,
    load_dataset,
)

from trloom.config.schema import DatasetConfig, DatasetSourceConfig
from trloom.data.formatting import format_dataset

logger = logging.getLogger(__name__)

DatasetLike = Dataset | DatasetDict | IterableDataset | IterableDatasetDict


def _is_local_path(path: str) -> bool:
    candidate = Path(path).expanduser()
    return candidate.exists() or path.startswith((".", "/", "~", "\\")) or ":\\" in path


def _infer_builder(path: str) -> str | None:
    """Infer a datasets builder name from a local file extension."""
    suffix = Path(path).suffix.lower()
    mapping = {
        ".json": "json",
        ".jsonl": "json",
        ".csv": "csv",
        ".tsv": "csv",
        ".parquet": "parquet",
        ".txt": "text",
        ".arrow": "arrow",
    }
    return mapping.get(suffix)


def _rename_columns(dataset: Any, columns: dict[str, str] | None) -> Any:
    if not columns:
        return dataset
    names = getattr(dataset, "column_names", None)
    if not names:
        return dataset
    rename_map = {src: dst for src, dst in columns.items() if src in names}
    if not rename_map:
        missing = sorted(set(columns) - set(names))
        if missing:
            logger.warning(
                "Column rename skipped missing source columns %s (have %s)",
                missing,
                list(names),
            )
        return dataset
    overlap = [dst for src, dst in rename_map.items() if dst in names and dst != src]
    if overlap:
        raise ValueError(
            f"Column rename would overwrite existing columns: {overlap}. "
            f"Rename map: {rename_map}"
        )
    return dataset.rename_columns(rename_map)


def _apply_columns(dataset: DatasetLike, columns: dict[str, str] | None) -> DatasetLike:
    if not columns:
        return dataset
    if isinstance(dataset, (DatasetDict, IterableDatasetDict)):
        return type(dataset)({k: _rename_columns(v, columns) for k, v in dataset.items()})
    return _rename_columns(dataset, columns)


def _load_single_source(
    source: DatasetSourceConfig | DatasetConfig,
    *,
    default_split: str | None = None,
    apply_columns: bool = True,
) -> DatasetLike:
    path = source.path
    if path is None:
        raise ValueError("Dataset source is missing 'path'.")

    kwargs: dict[str, Any] = {}
    if isinstance(source, DatasetConfig):
        kwargs.update(source.kwargs)
    # Allow extras from DatasetSourceConfig / DatasetConfig
    extras = source.model_extra or {}
    for key, value in extras.items():
        if key not in {"columns", "weight"}:
            kwargs.setdefault(key, value)

    name = source.name
    data_files = source.data_files
    data_dir = source.data_dir
    streaming = bool(source.streaming)
    split = source.split if getattr(source, "split", None) else default_split

    if data_files is not None:
        kwargs["data_files"] = data_files
    if data_dir is not None:
        kwargs["data_dir"] = data_dir

    if _is_local_path(path):
        local = Path(path).expanduser().resolve()
        if local.is_dir():
            # load_from_disk for saved DatasetDict / Dataset
            try:
                from datasets import load_from_disk

                logger.info("Loading local dataset from disk: %s", local)
                dataset = load_from_disk(str(local))
            except Exception:
                logger.info("Falling back to load_dataset for directory: %s", local)
                dataset = load_dataset(
                    str(local), name=name, split=split, streaming=streaming, **kwargs
                )
        else:
            builder = _infer_builder(str(local))
            if builder is None and data_files is None:
                raise ValueError(
                    f"Unsupported local dataset file type for '{local}'. "
                    "Use json/jsonl/csv/tsv/parquet/txt/arrow or provide data_files."
                )
            logger.info("Loading local dataset file via builder=%s path=%s", builder, local)
            load_path = builder or str(local)
            file_kwargs = dict(kwargs)
            if builder and "data_files" not in file_kwargs:
                file_kwargs["data_files"] = str(local)
            dataset = load_dataset(
                load_path, name=name, split=split, streaming=streaming, **file_kwargs
            )
    else:
        logger.info("Loading Hub dataset: %s (name=%s, split=%s)", path, name, split)
        dataset = load_dataset(path, name=name, split=split, streaming=streaming, **kwargs)

    if apply_columns:
        columns = getattr(source, "columns", None)
        dataset = _apply_columns(dataset, columns)

    return dataset


def _as_dataset_dict(dataset: DatasetLike, train_split: str) -> DatasetDict | IterableDatasetDict:
    if isinstance(dataset, (DatasetDict, IterableDatasetDict)):
        return dataset
    # Single split dataset — wrap under train_split key
    if isinstance(dataset, IterableDataset):
        return IterableDatasetDict({train_split: dataset})
    return DatasetDict({train_split: dataset})


def _mix_datasets(
    sources: list[DatasetSourceConfig],
    *,
    train_split: str,
) -> DatasetDict | IterableDatasetDict:
    loaded: list[Dataset | IterableDataset] = []
    weights: list[float] = []
    streaming = False

    for source in sources:
        ds = _load_single_source(source, default_split=source.split or train_split)
        if isinstance(ds, (DatasetDict, IterableDatasetDict)):
            if train_split not in ds:
                raise KeyError(
                    f"Mixture source '{source.path}' has no split '{train_split}'. "
                    f"Available: {list(ds.keys())}"
                )
            part = ds[train_split]
        else:
            part = ds
        if isinstance(part, IterableDataset):
            streaming = True
        loaded.append(part)
        weights.append(float(source.weight) if source.weight is not None else 1.0)

    if streaming:
        # Interleave for streaming mixtures; fall back to concatenate when weights are equal.
        from datasets import interleave_datasets

        mixed = interleave_datasets(loaded, probabilities=_normalize(weights), seed=42)
        return IterableDatasetDict({train_split: mixed})

    if len(set(weights)) == 1:
        mixed = concatenate_datasets(loaded)  # type: ignore[arg-type]
    else:
        from datasets import interleave_datasets

        mixed = interleave_datasets(loaded, probabilities=_normalize(weights), seed=42)
    return DatasetDict({train_split: mixed})


def _normalize(weights: list[float]) -> list[float]:
    total = sum(weights)
    if total <= 0:
        raise ValueError("Dataset mixture weights must sum to a positive value.")
    return [w / total for w in weights]


def _apply_formatting(split: Any, config: DatasetConfig) -> Any:
    if not (config.map_fn or config.prompt_template):
        return split
    return format_dataset(
        split,
        map_fn=config.map_fn,
        map_kwargs=config.map_kwargs,
        prompt_template=config.prompt_template,
        prompt_output_column=config.prompt_output_column,
        prompt_remove_columns=config.prompt_remove_columns,
    )


def _ensure_text_column(dataset: Any, text_column: str | None, *, split_label: str) -> None:
    if not text_column or not hasattr(dataset, "column_names"):
        return
    names = dataset.column_names
    if names is None:
        return
    if text_column not in names:
        raise KeyError(
            f"text_column '{text_column}' not found in {split_label} split. "
            f"Columns: {names}"
        )


def load_train_eval_datasets(
    config: DatasetConfig,
) -> tuple[Dataset | IterableDataset, Dataset | IterableDataset | None]:
    """Load train (and optional eval) datasets from a :class:`DatasetConfig`.

    After load / column rename, applies optional ``map_fn`` and ``prompt_template``.
    """
    if config.datasets:
        dataset_dict = _mix_datasets(config.datasets, train_split=config.train_split)
        # Top-level columns apply after mixture (per-source columns already applied).
        dataset_dict = _apply_columns(dataset_dict, config.columns)
    else:
        assert config.path is not None
        # Rename once at the DatasetDict level to avoid double-rename no-ops / conflicts.
        raw = _load_single_source(config, apply_columns=False)
        dataset_dict = _as_dataset_dict(raw, config.train_split)
        dataset_dict = _apply_columns(dataset_dict, config.columns)

    if config.train_split not in dataset_dict:
        available = list(dataset_dict.keys())
        raise KeyError(
            f"Train split '{config.train_split}' not found in dataset. Available splits: {available}"
        )

    train_dataset = _apply_formatting(dataset_dict[config.train_split], config)
    eval_dataset: Dataset | IterableDataset | None = None
    if config.eval_split and config.eval_split in dataset_dict:
        eval_dataset = _apply_formatting(dataset_dict[config.eval_split], config)

    _ensure_text_column(train_dataset, config.text_column, split_label="train")
    if eval_dataset is not None:
        _ensure_text_column(eval_dataset, config.text_column, split_label="eval")

    logger.info(
        "Loaded train dataset (%s)%s",
        type(train_dataset).__name__,
        f" and eval split '{config.eval_split}'" if eval_dataset is not None else "",
    )
    return train_dataset, eval_dataset
