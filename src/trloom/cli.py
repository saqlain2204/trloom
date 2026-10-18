"""Command-line interface for TRLoom."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trloom",
        description="YAML-driven fine-tuning on top of Hugging Face TRL.",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Print version and exit.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )

    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run a fine-tuning job from a YAML config.")
    run_parser.add_argument("config", type=str, help="Path to the YAML configuration file.")
    run_parser.add_argument(
        "--modal",
        action="store_true",
        help="Force execution on Modal (overrides modal.enabled in YAML).",
    )
    run_parser.add_argument(
        "--kaggle",
        action="store_true",
        help="Force execution on Kaggle (overrides kaggle.enabled in YAML).",
    )
    run_parser.add_argument(
        "--local",
        action="store_true",
        help="Force local execution (overrides remote backends in YAML).",
    )

    subparsers.add_parser("methods", help="List TRL training methods available in this environment.")

    validate_parser = subparsers.add_parser("validate", help="Validate a YAML config without training.")
    validate_parser.add_argument("config", type=str, help="Path to the YAML configuration file.")

    modal_parser = subparsers.add_parser(
        "modal-script",
        help="Write a standalone Modal entrypoint script for a YAML config.",
    )
    modal_parser.add_argument("config", type=str, help="Path to the YAML configuration file.")
    modal_parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Destination path for the generated script.",
    )

    kaggle_parser = subparsers.add_parser(
        "kaggle-notebook",
        help="Write a Kaggle notebook (+ kernel-metadata.json) for a YAML config.",
    )
    kaggle_parser.add_argument("config", type=str, help="Path to the YAML configuration file.")
    kaggle_parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Destination path for the generated .ipynb notebook.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(getattr(args, "verbose", False))

    if args.version:
        from trloom import __version__

        print(__version__)
        return 0

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "methods":
        from trloom import available_methods

        methods = available_methods(include_experimental=True)
        for method in methods:
            print(method)
        return 0

    if args.command == "validate":
        from trloom import load_config
        from trloom.trainers import get_trainer_spec

        config = load_config(args.config)
        spec = get_trainer_spec(config.method)
        payload = {
            "ok": True,
            "method": config.method,
            "trainer": spec.trainer_name,
            "config_class": spec.config_name,
            "experimental": spec.experimental,
            "output_dir": str(config.resolved_output_dir()),
            "wandb_enabled": config.wandb.enabled,
            "modal_enabled": config.modal.enabled,
            "kaggle_enabled": config.kaggle.enabled,
            "backend": config.resolved_backend(),
        }
        print(json.dumps(payload, indent=2))
        return 0

    if args.command == "modal-script":
        from trloom.modal_support.runner import write_modal_entrypoint

        dest = write_modal_entrypoint(args.config, destination=args.output)
        print(dest)
        return 0

    if args.command == "kaggle-notebook":
        from trloom.kaggle_support import write_kaggle_notebook

        dest = write_kaggle_notebook(args.config, destination=args.output)
        print(dest)
        return 0

    if args.command == "run":
        from trloom import run_from_yaml

        flags = [args.modal, args.kaggle, args.local]
        if sum(bool(f) for f in flags) > 1:
            parser.error("Use only one of --modal / --kaggle / --local.")

        use_modal: bool | None
        use_kaggle: bool | None
        if args.local:
            use_modal = False
            use_kaggle = False
        elif args.modal:
            use_modal = True
            use_kaggle = False
        elif args.kaggle:
            use_modal = False
            use_kaggle = True
        else:
            use_modal = None
            use_kaggle = None

        config_path = Path(args.config)
        if not config_path.is_file():
            print(f"Config not found: {config_path}", file=sys.stderr)
            return 1

        run_from_yaml(config_path, use_modal=use_modal, use_kaggle=use_kaggle)
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
