"""Tests for trainer registry and reward resolution."""

from __future__ import annotations

import pytest

from trloom.rewards import resolve_reward_funcs
from trloom.trainers.registry import (
    clear_registry_cache,
    get_trainer_spec,
    list_trainers,
    resolve_method_name,
)


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    clear_registry_cache()
    yield
    clear_registry_cache()


def test_list_trainers_includes_core_methods() -> None:
    methods = list_trainers()
    for required in ("sft", "dpo", "grpo", "kto", "reward", "rloo"):
        assert required in methods, f"missing {required} in {methods}"


def test_method_aliases() -> None:
    assert resolve_method_name("SFT") == "sft"
    assert resolve_method_name("direct-preference-optimization") == "dpo"
    assert resolve_method_name("rm") == "reward"


def test_get_trainer_spec_sft() -> None:
    spec = get_trainer_spec("sft")
    assert spec.trainer_name.endswith("Trainer")
    assert spec.config_name.endswith("Config")
    assert spec.method == "sft"


def test_unknown_method_raises() -> None:
    with pytest.raises(KeyError, match="Unknown training method"):
        get_trainer_spec("not_a_real_method_xyz")


def test_resolve_reward_import_path() -> None:
    funcs = resolve_reward_funcs(["operator:add"])
    assert funcs is not None
    assert funcs[0](2, 3) == 5


def test_resolve_reward_missing() -> None:
    with pytest.raises(ValueError):
        resolve_reward_funcs(["definitely_missing_reward_fn"])
