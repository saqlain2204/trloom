"""Tests for import-path resolution, formatting, and user_code bundling."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from trloom.config import load_config
from trloom.config.schema import DatasetConfig, FineTuneConfig
from trloom.data.dataset import load_train_eval_datasets
from trloom.data.formatting import apply_prompt_template, format_dataset
from trloom.imports import parse_import_spec, resolve_callable
from trloom.user_code import (
    build_user_code_bundle,
    collect_callable_specs,
    install_user_code_bundle,
    prepare_local_user_code,
)


def test_parse_import_spec_colon() -> None:
    assert parse_import_spec("pkg.mod:fn") == ("pkg.mod", "fn")


def test_parse_import_spec_dotted() -> None:
    assert parse_import_spec("pkg.mod.fn") == ("pkg.mod", "fn")


def test_resolve_callable_operator_add() -> None:
    fn = resolve_callable("operator:add")
    assert fn(2, 3) == 5


def test_prompt_template_on_dataset(tmp_path: Path) -> None:
    data_path = tmp_path / "train.jsonl"
    rows = [
        {"instruction": "Hi", "response": "Hello"},
        {"instruction": "Bye", "response": "Goodbye"},
    ]
    data_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    config = DatasetConfig(
        path=str(data_path),
        train_split="train",
        eval_split=None,
        prompt_template="Q: {{ instruction }}\nA: {{ response }}",
        prompt_output_column="text",
        prompt_remove_columns=True,
        text_column="text",
    )
    train, _ = load_train_eval_datasets(config)
    assert train.column_names == ["text"]
    assert train[0]["text"] == "Q: Hi\nA: Hello"


def test_map_fn_from_user_module(tmp_path: Path) -> None:
    module_path = tmp_path / "my_formatters.py"
    module_path.write_text(
        "def to_text(example):\n"
        "    return {'text': example['instruction'] + ' => ' + example['response']}\n",
        encoding="utf-8",
    )
    data_path = tmp_path / "train.jsonl"
    data_path.write_text(
        json.dumps({"instruction": "x", "response": "y"}) + "\n",
        encoding="utf-8",
    )

    config = FineTuneConfig.model_validate(
        {
            "method": "sft",
            "model": {"model_name_or_path": "sshleifer/tiny-gpt2"},
            "dataset": {
                "path": str(data_path),
                "train_split": "train",
                "eval_split": None,
                "map_fn": "my_formatters:to_text",
                "text_column": "text",
            },
            "user_code": [str(module_path)],
        }
    )
    prepare_local_user_code(config)
    train, _ = load_train_eval_datasets(config.dataset)
    assert train[0]["text"] == "x => y"


def test_collect_and_bundle_user_code(tmp_path: Path) -> None:
    module_path = tmp_path / "ship_me.py"
    module_path.write_text("def fmt(x):\n    return x\n", encoding="utf-8")

    config = load_config(
        {
            "method": "sft",
            "model": {"model_name_or_path": "sshleifer/tiny-gpt2"},
            "dataset": {
                "path": "trl-lib/Capybara",
                "eval_split": None,
                "map_fn": "ship_me:fmt",
            },
            "user_code": [str(module_path)],
            "formatting_func": "ship_me:fmt",
        }
    )
    specs = collect_callable_specs(config)
    assert "ship_me:fmt" in specs

    bundle = build_user_code_bundle(config)
    assert "ship_me.py" in bundle
    assert "def fmt" in bundle["ship_me.py"]

    # Simulate remote: wipe module from path/cache, install bundle, resolve again.
    sys.modules.pop("ship_me", None)
    if str(tmp_path) in sys.path:
        sys.path.remove(str(tmp_path))

    install_dir = install_user_code_bundle(bundle)
    assert install_dir is not None
    fn = resolve_callable("ship_me:fmt")
    assert fn("ok") == "ok"


def test_stdlib_not_bundled() -> None:
    config = load_config(
        {
            "method": "grpo",
            "model": {"model_name_or_path": "sshleifer/tiny-gpt2"},
            "dataset": {"path": "trl-lib/Capybara", "eval_split": None},
            "reward_funcs": ["operator:add"],
        }
    )
    bundle = build_user_code_bundle(config)
    assert "operator.py" not in bundle


def test_formatting_func_root_overrides_dataset() -> None:
    config = load_config(
        {
            "method": "sft",
            "model": {"model_name_or_path": "sshleifer/tiny-gpt2"},
            "dataset": {
                "path": "trl-lib/Capybara",
                "eval_split": None,
                "formatting_func": "dataset_side:fn",
            },
            "formatting_func": "root_side:fn",
        }
    )
    assert config.resolved_formatting_func() == "root_side:fn"


def test_user_code_string_coerced() -> None:
    config = load_config(
        {
            "method": "sft",
            "model": {"model_name_or_path": "sshleifer/tiny-gpt2"},
            "dataset": {"path": "trl-lib/Capybara", "eval_split": None},
            "user_code": "formatters.py",
        }
    )
    assert config.user_code == ["formatters.py"]


def test_user_code_paths_resolve_relative_to_yaml(tmp_path: Path) -> None:
    fmt = tmp_path / "helpers.py"
    fmt.write_text("def f(x):\n    return x\n", encoding="utf-8")
    cfg = tmp_path / "job.yaml"
    cfg.write_text(
        """
method: sft
model:
  model_name_or_path: sshleifer/tiny-gpt2
dataset:
  path: trl-lib/Capybara
  eval_split: null
user_code:
  - ./helpers.py
""",
        encoding="utf-8",
    )
    config = load_config(cfg)
    assert Path(config.user_code[0]) == fmt.resolve()


def test_apply_prompt_template_helper() -> None:
    from datasets import Dataset

    ds = Dataset.from_list([{"a": 1, "b": 2}])
    out = apply_prompt_template(ds, "{{ a }}+{{ b }}", output_column="text", remove_columns=True)
    assert out[0]["text"] == "1+2"
    assert out.column_names == ["text"]


def test_format_dataset_order(tmp_path: Path) -> None:
    module_path = tmp_path / "order_fmt.py"
    module_path.write_text(
        "def double_x(example):\n    return {'x': example['x'] * 2}\n",
        encoding="utf-8",
    )
    sys.path.insert(0, str(tmp_path))
    try:
        from datasets import Dataset

        ds = Dataset.from_list([{"x": 3, "y": "z"}])
        out = format_dataset(
            ds,
            map_fn="order_fmt:double_x",
            prompt_template="val={{ x }}",
            prompt_output_column="text",
            prompt_remove_columns=True,
        )
        assert out[0]["text"] == "val=6"
    finally:
        sys.modules.pop("order_fmt", None)
        if str(tmp_path) in sys.path:
            sys.path.remove(str(tmp_path))


def test_example_sft_formatted_validates() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / "examples" / "sft_formatted.yaml"
    config = load_config(path)
    assert config.dataset.map_fn == "formatters:instruction_to_text"
    assert config.user_code


def test_train_remote_signature_accepts_user_code_bundle() -> None:
    import inspect

    from trloom.modal_support.runner import train_remote

    params = inspect.signature(train_remote).parameters
    assert "user_code_bundle" in params
