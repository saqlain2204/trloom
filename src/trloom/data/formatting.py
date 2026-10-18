"""Dataset formatting helpers: prompt templates and map callables."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from trloom.imports import resolve_callable

logger = logging.getLogger(__name__)


def _render_prompt_template(template: str, example: dict[str, Any]) -> str:
    try:
        from jinja2 import BaseLoader, Environment, StrictUndefined, TemplateError
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Prompt templates require Jinja2. Install with: pip install jinja2"
        ) from exc

    env = Environment(
        loader=BaseLoader(),
        undefined=StrictUndefined,
        autoescape=False,
        keep_trailing_newline=True,
    )
    compiled = env.from_string(template)
    # Row values only — avoid leaking internals via **example if non-str keys appear
    context = {str(k): v for k, v in example.items()}
    try:
        return compiled.render(**context)
    except TemplateError as exc:
        columns = sorted(context.keys())
        raise ValueError(
            f"Failed to render prompt_template: {exc}. "
            f"Available columns: {columns}"
        ) from exc


def apply_prompt_template(
    dataset: Any,
    template: str,
    *,
    output_column: str = "text",
    remove_columns: list[str] | bool | None = None,
) -> Any:
    """Map a Jinja2 prompt template over a dataset, writing ``output_column``.

    Raises:
        ValueError: If ``template`` is empty/whitespace, or rendering fails
            (e.g. missing column referenced by the template).
        TypeError: If ``dataset`` looks like a DatasetDict (format splits individually).
    """
    if not isinstance(template, str) or not template.strip():
        raise ValueError("prompt_template must be a non-empty string.")
    if not isinstance(output_column, str) or not output_column.strip():
        raise ValueError("prompt_output_column must be a non-empty string.")

    # Guard against accidentally formatting a DatasetDict as a whole.
    column_names = getattr(dataset, "column_names", None)
    if isinstance(column_names, dict):
        raise TypeError(
            "prompt_template expects a single split (Dataset / IterableDataset), "
            "not a DatasetDict. Format each split individually."
        )

    def _map_one(example: dict[str, Any]) -> dict[str, Any]:
        return {output_column: _render_prompt_template(template, example)}

    map_kwargs: dict[str, Any] = {}
    if remove_columns is True:
        if column_names:
            map_kwargs["remove_columns"] = list(column_names)
        else:
            logger.warning(
                "prompt_remove_columns=True but dataset has no column_names; "
                "keeping original columns."
            )
    elif isinstance(remove_columns, list):
        if not all(isinstance(c, str) and c for c in remove_columns):
            raise ValueError("prompt_remove_columns list entries must be non-empty strings.")
        map_kwargs["remove_columns"] = remove_columns

    logger.info(
        "Applying prompt_template -> column '%s'%s",
        output_column,
        f" (remove_columns={map_kwargs.get('remove_columns')})" if map_kwargs else "",
    )
    return dataset.map(_map_one, **map_kwargs)


def apply_map_fn(
    dataset: Any,
    map_fn: str | Callable[..., Any],
    *,
    map_kwargs: dict[str, Any] | None = None,
) -> Any:
    """Apply a ``dataset.map`` transform resolved from an import path or callable."""
    if isinstance(map_fn, str) and not map_fn.strip():
        raise ValueError("map_fn must be a non-empty import path string or a callable.")

    column_names = getattr(dataset, "column_names", None)
    if isinstance(column_names, dict):
        raise TypeError(
            "map_fn expects a single split (Dataset / IterableDataset), "
            "not a DatasetDict. Format each split individually."
        )

    func = resolve_callable(map_fn, label="map_fn") if isinstance(map_fn, str) else map_fn
    if not callable(func):
        raise TypeError(f"map_fn resolved to a non-callable object: {func!r}")

    kwargs = dict(map_kwargs or {})
    logger.info(
        "Applying dataset map_fn '%s' with kwargs=%s",
        getattr(func, "__name__", func),
        kwargs,
    )
    try:
        return dataset.map(func, **kwargs)
    except Exception as exc:
        raise RuntimeError(
            f"dataset.map failed while applying map_fn "
            f"'{getattr(func, '__name__', func)}': {exc}"
        ) from exc


def format_dataset(
    dataset: Any,
    *,
    map_fn: str | Callable[..., Any] | None = None,
    map_kwargs: dict[str, Any] | None = None,
    prompt_template: str | None = None,
    prompt_output_column: str = "text",
    prompt_remove_columns: list[str] | bool | None = None,
) -> Any:
    """Run configured formatting steps on a single split.

    Order:
    1. ``map_fn`` (custom preprocess)
    2. ``prompt_template`` (Jinja2 → ``prompt_output_column``)
    """
    result = dataset
    if map_fn is not None:
        result = apply_map_fn(result, map_fn, map_kwargs=map_kwargs)
    if prompt_template is not None:
        result = apply_prompt_template(
            result,
            prompt_template,
            output_column=prompt_output_column,
            remove_columns=prompt_remove_columns,
        )
    return result
