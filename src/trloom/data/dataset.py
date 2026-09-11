"""Load training datasets from the Hugging Face Hub or local files."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict, concatenate_datasets, load_dataset

from trloom.config.schema import DatasetConfig, DatasetSourceConfig

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
    rename_map = {src: dst for src, dst in columns.items() if src in getattr(dataset, "column_names", [])}
    if not rename_map:
        return dataset
    return dataset.rename_columns(rename_map)


def _load_single_source(
    source: DatasetSourceConfig | DatasetConfig,
    *,
    default_split: str | None = None,
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
                dataset = load_dataset(str(local), name=name, split=split, streaming=streaming, **kwargs)
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
            dataset = load_dataset(load_path, name=name, split=split, streaming=streaming, **file_kwargs)
    else:
        logger.info("Loading Hub dataset: %s (name=%s, split=%s)", path, name, split)
        dataset = load_dataset(path, name=name, split=split, streaming=streaming, **kwargs)

    columns = getattr(source, "columns", None)
    if columns and hasattr(dataset, "column_names"):
        dataset = _rename_columns(dataset, columns)
    elif columns and isinstance(dataset, (DatasetDict, IterableDatasetDict)):
        dataset = type(dataset)({k: _rename_columns(v, columns) for k, v in dataset.items()})

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


def load_train_eval_datasets(
    config: DatasetConfig,
) -> tuple[Dataset | IterableDataset, Dataset | IterableDataset | None]:
    """Load train (and optional eval) datasets from a :class:`DatasetConfig`."""
    if config.datasets:
        dataset_dict = _mix_datasets(config.datasets, train_split=config.train_split)
    else:
        assert config.path is not None
        raw = _load_single_source(config)
        dataset_dict = _as_dataset_dict(raw, config.train_split)
        if config.columns:
            if isinstance(dataset_dict, (DatasetDict, IterableDatasetDict)):
                dataset_dict = type(dataset_dict)(
                    {k: _rename_columns(v, config.columns) for k, v in dataset_dict.items()}
                )

    if config.train_split not in dataset_dict:
        available = list(dataset_dict.keys())
        raise KeyError(
            f"Train split '{config.train_split}' not found in dataset. Available splits: {available}"
        )

    train_dataset = dataset_dict[config.train_split]
    eval_dataset: Dataset | IterableDataset | None = None
    if config.eval_split and config.eval_split in dataset_dict:
        eval_dataset = dataset_dict[config.eval_split]

    if config.text_column and hasattr(train_dataset, "column_names"):
        if config.text_column not in train_dataset.column_names:
            raise KeyError(
                f"text_column '{config.text_column}' not found. "
                f"Columns: {train_dataset.column_names}"
            )

    logger.info(
        "Loaded train dataset (%s)%s",
        type(train_dataset).__name__,
        f" and eval split '{config.eval_split}'" if eval_dataset is not None else "",
    )
    return train_dataset, eval_dataset
