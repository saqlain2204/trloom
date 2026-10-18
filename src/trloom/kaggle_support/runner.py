"""Run TRLoom fine-tuning jobs on Kaggle GPUs / notebooks."""

from __future__ import annotations

import json
import logging
import os
import tempfile
import textwrap
import time
from pathlib import Path
from typing import Any

from trloom.config.loader import load_config
from trloom.config.schema import FineTuneConfig, KaggleConfig
from trloom.kaggle_support.env import is_kaggle_environment, kaggle_working_dir
from trloom.user_code import build_user_code_bundle

logger = logging.getLogger(__name__)

_DEFAULT_PIP = [
    "torch",
    "trl",
    "transformers",
    "datasets",
    "accelerate",
    "peft",
    "pyyaml",
    "pydantic",
    "huggingface-hub",
    "jinja2",
    "wandb",
]


def _ensure_kaggle_api() -> Any:
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Kaggle API support requires the `kaggle` package. "
            "Install with: pip install 'trloom[kaggle]'"
        ) from exc
    api = KaggleApi()
    api.authenticate()
    return api


def _dedupe(packages: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for package in packages:
        key = package.lower()
        if key not in seen:
            seen.add(key)
            unique.append(package)
    return unique


def _resolve_username(kaggle_cfg: KaggleConfig) -> str:
    if kaggle_cfg.username:
        return kaggle_cfg.username
    for key in ("KAGGLE_USERNAME", "KAGGLE_USER"):
        value = os.environ.get(key)
        if value:
            return value
    raise ValueError(
        "Kaggle username is required when pushing a notebook. "
        "Set kaggle.username or the KAGGLE_USERNAME environment variable."
    )


def _resolve_kernel_slug(kaggle_cfg: KaggleConfig, config_path: Path) -> str:
    if kaggle_cfg.kernel_slug:
        return kaggle_cfg.kernel_slug
    # Kaggle kernel ids are lowercase slug-ish; keep it simple and stable.
    stem = "".join(ch if ch.isalnum() else "-" for ch in config_path.stem.lower())
    stem = stem.strip("-") or "trloom-job"
    return f"trloom-{stem}"[:90]


def prepare_kaggle_config(config: FineTuneConfig) -> FineTuneConfig:
    """Return a config adjusted for in-kernel Kaggle execution (no recursion)."""
    data = config.model_dump(mode="python")
    data.setdefault("modal", {})
    data["modal"]["enabled"] = False
    data.setdefault("kaggle", {})
    data["kaggle"]["enabled"] = False
    data.setdefault("training", {})
    output_dir = config.kaggle.output_dir or str(kaggle_working_dir() / "outputs")
    data["training"]["output_dir"] = output_dir
    return FineTuneConfig.model_validate(data)


def _install_snippet(kaggle_cfg: KaggleConfig) -> str:
    packages = _dedupe([*_DEFAULT_PIP, *list(kaggle_cfg.pip_packages)])
    if kaggle_cfg.install_source == "git":
        install_line = f'"{kaggle_cfg.git_url}"'
    else:
        install_line = '"trloom"'
    pkg_literal = ", ".join(json.dumps(p) for p in packages)
    return textwrap.dedent(
        f"""\
        import subprocess, sys
        pkgs = [{install_line}, {pkg_literal}]
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *pkgs])
        """
    )


def _notebook_cells(
    *,
    config_yaml: str,
    user_code_bundle: dict[str, str],
    kaggle_cfg: KaggleConfig,
) -> list[dict[str, Any]]:
    install_code = _install_snippet(kaggle_cfg)
    bundle_json = json.dumps(user_code_bundle, indent=2)
    config_literal = json.dumps(config_yaml)
    run_code = textwrap.dedent(
        f"""\
        import json
        from pathlib import Path
        import yaml

        from trloom.config.loader import load_config
        from trloom.job import FineTuneJob
        from trloom.user_code import install_user_code_bundle

        USER_CODE_BUNDLE = json.loads({json.dumps(bundle_json)})
        CONFIG_YAML = json.loads({config_literal})

        install_user_code_bundle(USER_CODE_BUNDLE or None)

        data = yaml.safe_load(CONFIG_YAML)
        data.setdefault("modal", {{}})
        data["modal"]["enabled"] = False
        data.setdefault("kaggle", {{}})
        data["kaggle"]["enabled"] = False
        data.setdefault("training", {{}})
        data["training"]["output_dir"] = {json.dumps(kaggle_cfg.output_dir)}

        cfg = load_config(data)
        result = FineTuneJob(cfg).run()
        print({{"status": "completed", "output_dir": str(cfg.resolved_output_dir())}})
        print(result)
        """
    )

    def _code_cell(source: str) -> dict[str, Any]:
        return {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": source.splitlines(keepends=True),
        }

    def _md_cell(source: str) -> dict[str, Any]:
        return {
            "cell_type": "markdown",
            "metadata": {},
            "source": source.splitlines(keepends=True),
        }

    return [
        _md_cell(
            "# TRLoom Kaggle job\n\n"
            "Auto-generated notebook. Enable a **GPU** accelerator in the session "
            "settings if needed, then Run All.\n"
        ),
        _code_cell(install_code),
        _code_cell(run_code),
    ]


def build_kaggle_notebook(
    config: FineTuneConfig,
    *,
    config_yaml: str,
    user_code_bundle: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build an in-memory Jupyter notebook dict for a TRLoom job."""
    cells = _notebook_cells(
        config_yaml=config_yaml,
        user_code_bundle=user_code_bundle or {},
        kaggle_cfg=config.kaggle,
    )
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
            "accelerator": "GPU" if config.kaggle.accelerator == "gpu" else None,
        },
        "cells": cells,
    }


def write_kaggle_notebook(
    config_path: str | Path,
    destination: str | Path | None = None,
    *,
    write_metadata: bool = True,
) -> Path:
    """Write a Kaggle notebook (and optional kernel-metadata.json) for a YAML job."""
    path = Path(config_path).expanduser().resolve()
    config = load_config(path)
    bundle = build_user_code_bundle(config)
    notebook = build_kaggle_notebook(
        config,
        config_yaml=path.read_text(encoding="utf-8"),
        user_code_bundle=bundle,
    )

    if destination is None:
        if config.kaggle.notebook_path:
            dest = Path(config.kaggle.notebook_path).expanduser().resolve()
        else:
            dest = Path(tempfile.gettempdir()) / f"trloom_kaggle_{path.stem}.ipynb"
    else:
        dest = Path(destination).expanduser().resolve()

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
    logger.info("Wrote Kaggle notebook to %s", dest)

    if write_metadata:
        _write_kernel_metadata(dest.parent, config, path, code_file=dest.name)

    return dest


def _write_kernel_metadata(
    folder: Path,
    config: FineTuneConfig,
    config_path: Path,
    *,
    code_file: str,
) -> Path:
    kaggle_cfg = config.kaggle
    username = kaggle_cfg.username or os.environ.get("KAGGLE_USERNAME") or "YOUR_USERNAME"
    slug = _resolve_kernel_slug(kaggle_cfg, config_path)
    enable_gpu = kaggle_cfg.accelerator == "gpu"
    enable_tpu = kaggle_cfg.accelerator == "tpu"
    meta = {
        "id": f"{username}/{slug}",
        "id_no": 0,
        "title": kaggle_cfg.title,
        "code_file": code_file,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": kaggle_cfg.is_private,
        "enable_gpu": enable_gpu,
        "enable_tpu": enable_tpu,
        "enable_internet": kaggle_cfg.enable_internet,
        "dataset_sources": list(kaggle_cfg.dataset_sources),
        "competition_sources": list(kaggle_cfg.competition_sources),
        "kernel_sources": list(kaggle_cfg.kernel_sources),
        "model_sources": [],
    }
    meta_path = folder / "kernel-metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote Kaggle kernel metadata to %s", meta_path)
    return meta_path


def _push_kernel(folder: Path, kaggle_cfg: KaggleConfig) -> dict[str, Any]:
    api = _ensure_kaggle_api()
    logger.info("Pushing Kaggle kernel from %s", folder)
    # kernels_push_with_http_info is version-dependent; prefer kernels_push.
    push = getattr(api, "kernels_push", None) or getattr(api, "kernel_push", None)
    if push is None:  # pragma: no cover
        raise RuntimeError("Installed kaggle package does not expose kernels_push.")
    result = push(str(folder))
    payload: dict[str, Any] = {"status": "pushed", "folder": str(folder)}
    if result is not None:
        payload["result"] = str(result)

    if kaggle_cfg.wait:
        username = _resolve_username(kaggle_cfg)
        slug = Path(folder).name
        # Prefer slug from metadata when present.
        meta_path = folder / "kernel-metadata.json"
        if meta_path.is_file():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            kernel_id = str(meta.get("id") or f"{username}/{slug}")
        else:
            kernel_id = f"{username}/{slug}"
        payload.update(_wait_for_kernel(api, kernel_id, timeout=kaggle_cfg.timeout_seconds))

    return payload


def _wait_for_kernel(api: Any, kernel_id: str, *, timeout: int) -> dict[str, Any]:
    """Poll kernel status until complete or timeout."""
    status_fn = getattr(api, "kernels_status", None)
    if status_fn is None:
        logger.warning("kaggle API has no kernels_status; skipping wait for %s", kernel_id)
        return {"wait": "skipped"}

    deadline = time.time() + max(timeout, 1)
    last_status = None
    while time.time() < deadline:
        try:
            status = status_fn(kernel_id)
        except Exception as exc:
            logger.warning("Could not poll Kaggle kernel status (%s)", exc)
            return {"wait": "error", "error": str(exc)}
        status_str = str(getattr(status, "status", status))
        if status_str != last_status:
            logger.info("Kaggle kernel %s status: %s", kernel_id, status_str)
            last_status = status_str
        lowered = status_str.lower()
        if any(token in lowered for token in ("complete", "error", "cancel", "fail")):
            return {"wait": "done", "kernel_status": status_str}
        time.sleep(15)
    return {"wait": "timeout", "kernel_status": last_status}


def run_inline_on_kaggle(config: FineTuneConfig) -> Any:
    """Run a fine-tuning job in the current process with Kaggle-friendly paths."""
    from trloom.job import FineTuneJob

    prepared = prepare_kaggle_config(config)
    logger.info(
        "Running TRLoom inline on Kaggle (output_dir=%s)",
        prepared.training.get("output_dir"),
    )
    return FineTuneJob(prepared).run()


def run_on_kaggle(config_path: str | Path, *, output_subdir: str | None = None) -> dict[str, Any]:
    """Execute a YAML job on Kaggle (inline when in-kernel, else notebook / push)."""
    path = Path(config_path).expanduser().resolve()
    config = load_config(path)
    if not config.kaggle.enabled:
        logger.info("Enabling Kaggle for this run (kaggle.enabled was false in YAML).")

    mode = config.kaggle.mode
    if mode == "auto":
        mode = "inline" if is_kaggle_environment() else "notebook"

    if mode == "inline":
        result = run_inline_on_kaggle(config)
        metrics = getattr(result, "metrics", None)
        return {
            "status": "completed",
            "mode": "inline",
            "output_dir": str(prepare_kaggle_config(config).resolved_output_dir()),
            "metrics": dict(metrics) if isinstance(metrics, dict) else None,
            "in_kaggle": is_kaggle_environment(),
        }

    # notebook mode
    work = Path(tempfile.mkdtemp(prefix="trloom_kaggle_"))
    notebook_name = f"{(output_subdir or path.stem)}.ipynb"
    notebook_path = work / notebook_name
    dest = write_kaggle_notebook(path, destination=notebook_path, write_metadata=True)

    payload: dict[str, Any] = {
        "status": "notebook_ready",
        "mode": "notebook",
        "notebook_path": str(dest),
        "folder": str(work),
        "pushed": False,
    }

    kernel_id: str | None = None
    if config.kaggle.push:
        # Ensure metadata username is concrete before push.
        username = _resolve_username(config.kaggle)
        meta_path = work / "kernel-metadata.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        slug = _resolve_kernel_slug(config.kaggle, path)
        kernel_id = f"{username}/{slug}"
        meta["id"] = kernel_id
        meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        push_result = _push_kernel(work, config.kaggle)
        payload.update(push_result)
        payload["pushed"] = True
        payload["status"] = "pushed"
        payload["kernel_id"] = kernel_id
    else:
        logger.info(
            "Kaggle notebook ready at %s. Upload to Kaggle or set kaggle.push=true.",
            dest,
        )

    download_dir = config.kaggle.download_dir
    if download_dir and kernel_id:
        logger.info(
            "download_dir=%s — fetch kernel outputs from the Kaggle UI or "
            "`kaggle kernels output %s -p %s` after the run finishes.",
            download_dir,
            kernel_id,
            download_dir,
        )

    return payload
