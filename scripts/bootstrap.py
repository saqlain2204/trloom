#!/usr/bin/env python3
"""Cross-platform environment bootstrap for TRLoom.

Works on Windows, macOS, and Linux with one command:

    python scripts/bootstrap.py              # local install (no Modal auth)
    python scripts/bootstrap.py --modal      # local install + Modal extra + modal setup

What it does:
  1. Creates ``.venv`` in the repo root if it does not exist
  2. Upgrades pip inside that venv
  3. Editable-installs TRLoom with the chosen extras
  4. With ``--modal``, runs ``python -m modal setup`` (interactive login)
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENV_DIR = ROOT / ".venv"


def _venv_python() -> Path:
    if os.name == "nt":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def _venv_activate_hint() -> str:
    if os.name == "nt":
        return r".venv\Scripts\Activate.ps1"
    return "source .venv/bin/activate"


def _run(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    print(f"\n>> {' '.join(cmd)}")
    subprocess.check_call(cmd, cwd=ROOT, env=env)


def _ensure_venv() -> Path:
    py = _venv_python()
    if py.is_file():
        print(f"Using existing virtualenv: {VENV_DIR}")
        return py

    print(f"Creating virtualenv: {VENV_DIR}")
    _run([sys.executable, "-m", "venv", str(VENV_DIR)])
    if not py.is_file():
        raise RuntimeError(f"Virtualenv was created but Python was not found at {py}")
    return py


def _install(py: Path, *, extras: list[str]) -> None:
    _run([str(py), "-m", "pip", "install", "--upgrade", "pip"])
    if extras:
        # Editable install with extras, e.g. .[modal,dev]
        spec = ".[{}]".format(",".join(extras))
    else:
        spec = "."
    _run([str(py), "-m", "pip", "install", "-e", spec])


def _modal_setup(py: Path) -> None:
    print("\nRunning Modal setup (browser / token login may open)...")
    # UTF-8 helps avoid Windows console encoding issues with the Modal CLI.
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUTF8", "1")
    _run([str(py), "-m", "modal", "setup"], env=env)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create .venv (if needed) and install TRLoom.",
    )
    parser.add_argument(
        "--modal",
        action="store_true",
        help="Also install the Modal extra and run `python -m modal setup`.",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Also install the Weights & Biases extra.",
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Also install the dev extra (pytest, ruff).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Install wandb + modal + bitsandbytes extras and run Modal setup.",
    )
    parser.add_argument(
        "--skip-modal-setup",
        action="store_true",
        help="With --modal / --all, install Modal but skip `modal setup`.",
    )
    args = parser.parse_args(argv)

    if sys.version_info < (3, 10):
        print("TRLoom requires Python 3.10+.", file=sys.stderr)
        return 1

    extras: list[str] = []
    if args.all:
        extras.append("all")
    else:
        if args.modal:
            extras.append("modal")
        if args.wandb:
            extras.append("wandb")
    if args.dev:
        extras.append("dev")
    # Keep order stable / unique
    extras = list(dict.fromkeys(extras))

    py = _ensure_venv()
    _install(py, extras=extras)

    want_modal_setup = (args.modal or args.all) and not args.skip_modal_setup
    if want_modal_setup:
        _modal_setup(py)

    print("\nDone.")
    print(f"Activate the venv:  {_venv_activate_hint()}")
    print("Then try:            trloom --version")
    print("                     trloom methods")
    if args.modal or args.all:
        print("Modal smoke:         trloom run examples/modal_smoke/config.yaml --modal")
    else:
        print("Local example:       trloom run examples/sft_local.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
