"""TRL trainer discovery and construction."""

from __future__ import annotations

from trloom.trainers.builder import build_trainer
from trloom.trainers.registry import TrainerSpec, get_trainer_spec, list_trainers

__all__ = [
    "TrainerSpec",
    "build_trainer",
    "get_trainer_spec",
    "list_trainers",
]
