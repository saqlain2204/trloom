"""Example dataset formatters for TRLoom YAML configs.

Referenced from YAML via import paths such as:

    dataset:
      map_fn: formatters:instruction_to_text
      formatting_func: formatters:messages_to_text

When jobs run on Modal, TRLoom bundles this module automatically so the remote
worker can import the same callables.
"""

from __future__ import annotations

from typing import Any


def instruction_to_text(example: dict[str, Any]) -> dict[str, str]:
    """Map instruction/response columns into a single ``text`` field."""
    instruction = example.get("instruction") or example.get("prompt") or ""
    response = example.get("response") or example.get("completion") or ""
    text = f"### Instruction:\n{instruction}\n\n### Response:\n{response}".strip()
    return {"text": text}


def messages_to_text(example: dict[str, Any]) -> str:
    """SFTTrainer-style formatting_func over a chat ``messages`` column."""
    messages = example.get("messages") or []
    parts: list[str] = []
    for turn in messages:
        role = turn.get("role", "user")
        content = turn.get("content", "")
        parts.append(f"{role}: {content}")
    return "\n".join(parts)
