"""Run TRLoom fine-tuning jobs on Modal."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any

from trloom.config.loader import load_config
from trloom.config.schema import FineTuneConfig, ModalConfig

logger = logging.getLogger(__name__)


def _ensure_modal() -> Any:
    try:
        import modal
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Modal support requires the `modal` package. Install with: pip install 'trloom[modal]'"
        ) from exc
    return modal


def _build_image(modal: Any, modal_cfg: ModalConfig) -> Any:
    packages = [
        "trloom",
        "trl",
        "transformers",
        "datasets",
        "accelerate",
        "peft",
        "pyyaml",
        "pydantic",
        "huggingface-hub",
        "wandb",
        *list(modal_cfg.pip_packages),
    ]
    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_packages: list[str] = []
    for package in packages:
        key = package.lower()
        if key not in seen:
            seen.add(key)
            unique_packages.append(package)

    image = (
        modal.Image.debian_slim(python_version=modal_cfg.python_version)
        .pip_install(*unique_packages)
    )
    return image


def create_modal_app(config: FineTuneConfig, config_path: str | Path | None = None) -> Any:
    """Create a Modal App wired to run a TRLoom job from YAML content."""
    modal = _ensure_modal()
    modal_cfg = config.modal

    app = modal.App(modal_cfg.app_name)
    volume = modal.Volume.from_name(modal_cfg.volume_name, create_if_missing=True)
    image = _build_image(modal, modal_cfg)

    secret_objects = [modal.Secret.from_name(name) for name in modal_cfg.secrets]

    function_kwargs: dict[str, Any] = {
        "image": image,
        "gpu": modal_cfg.gpu,
        "timeout": modal_cfg.timeout,
        "volumes": {modal_cfg.volume_mount: volume},
        "secrets": secret_objects,
    }
    if modal_cfg.cpu is not None:
        function_kwargs["cpu"] = modal_cfg.cpu
    if modal_cfg.memory is not None:
        function_kwargs["memory"] = modal_cfg.memory
    if modal_cfg.region is not None:
        function_kwargs["region"] = modal_cfg.region

    yaml_text: str | None = None
    if config_path is not None:
        yaml_text = Path(config_path).expanduser().resolve().read_text(encoding="utf-8")

    @app.function(**function_kwargs)
    def train_remote(config_yaml: str, output_subdir: str = "run") -> dict[str, Any]:
        """Remote entrypoint executed inside Modal."""
        from pathlib import Path as _Path

        import yaml

        from trloom.config.loader import load_config as _load_config
        from trloom.job import FineTuneJob

        mount = modal_cfg.volume_mount
        run_dir = _Path(mount) / output_subdir
        run_dir.mkdir(parents=True, exist_ok=True)

        data = yaml.safe_load(config_yaml)
        if not isinstance(data, dict):
            raise TypeError("Config YAML must deserialize to a mapping.")

        # Force local execution inside the container and redirect outputs to the volume
        data.setdefault("modal", {})
        data["modal"]["enabled"] = False
        data.setdefault("training", {})
        data["training"]["output_dir"] = str(run_dir)

        cfg = _load_config(data)
        result = FineTuneJob(cfg).run()
        volume.commit()

        metrics = getattr(result, "metrics", None)
        return {
            "output_dir": str(run_dir),
            "metrics": dict(metrics) if isinstance(metrics, dict) else None,
            "status": "completed",
        }

    # Attach helpers for callers
    app.trloom_train_remote = train_remote  # type: ignore[attr-defined]
    app.trloom_config_yaml = yaml_text  # type: ignore[attr-defined]
    app.trloom_volume = volume  # type: ignore[attr-defined]
    return app


def run_on_modal(config_path: str | Path, *, output_subdir: str | None = None) -> dict[str, Any]:
    """Submit a TRLoom YAML job to Modal and wait for completion."""
    modal = _ensure_modal()
    path = Path(config_path).expanduser().resolve()
    config = load_config(path)
    if not config.modal.enabled:
        logger.info("Enabling Modal for this run (modal.enabled was false in YAML).")

    app = create_modal_app(config, config_path=path)
    yaml_text = path.read_text(encoding="utf-8")
    subdir = output_subdir or path.stem

    logger.info(
        "Launching Modal job app=%s gpu=%s timeout=%ss",
        config.modal.app_name,
        config.modal.gpu,
        config.modal.timeout,
    )

    with app.run():
        remote = app.trloom_train_remote  # type: ignore[attr-defined]
        result = remote.remote(yaml_text, subdir)

    download_dir = config.modal.download_dir
    if download_dir:
        _download_volume_subdir(
            modal,
            volume_name=config.modal.volume_name,
            remote_subdir=subdir,
            local_dir=Path(download_dir).expanduser().resolve(),
        )

    return result


def _download_volume_subdir(
    modal: Any,
    *,
    volume_name: str,
    remote_subdir: str,
    local_dir: Path,
) -> None:
    """Best-effort download of volume outputs to a local directory."""
    local_dir.mkdir(parents=True, exist_ok=True)
    volume = modal.Volume.from_name(volume_name)
    # Modal Volume API varies by version; support common patterns.
    if hasattr(volume, "get"):
        # Newer APIs may support path-based get; fall back to read-only mount via temporary copy.
        logger.info(
            "Download requested to %s — use `modal volume get %s %s %s` if automatic copy is unavailable.",
            local_dir,
            volume_name,
            remote_subdir,
            local_dir,
        )
    try:
        entries = list(volume.listdir(remote_subdir))  # type: ignore[attr-defined]
        for entry in entries:
            name = getattr(entry, "path", None) or getattr(entry, "filename", None) or str(entry)
            basename = Path(str(name)).name
            target = local_dir / basename
            if hasattr(volume, "read_file"):
                data = volume.read_file(f"{remote_subdir}/{basename}")  # type: ignore[attr-defined]
                if isinstance(data, (bytes, bytearray)):
                    target.write_bytes(data)
                else:
                    chunks = list(data)
                    target.write_bytes(b"".join(chunks) if chunks and isinstance(chunks[0], (bytes, bytearray)) else b"".join(c.encode() if isinstance(c, str) else c for c in chunks))
        logger.info("Downloaded Modal volume outputs to %s", local_dir)
    except Exception as exc:
        logger.warning(
            "Could not auto-download Modal outputs (%s). "
            "Fetch manually with: modal volume get %s /%s %s",
            exc,
            volume_name,
            remote_subdir,
            local_dir,
        )
        # Keep a marker file so callers know the remote location
        marker = local_dir / "REMOTE_OUTPUT.txt"
        marker.write_text(
            f"Remote volume={volume_name} subdir={remote_subdir}\n",
            encoding="utf-8",
        )


def write_modal_entrypoint(config_path: str | Path, destination: str | Path | None = None) -> Path:
    """Write a standalone Modal script for the given YAML config.

    Useful when users prefer ``modal run script.py`` over the Python API.
    """
    path = Path(config_path).expanduser().resolve()
    config = load_config(path)
    dest = (
        Path(destination).expanduser().resolve()
        if destination
        else Path(tempfile.gettempdir()) / f"trloom_modal_{path.stem}.py"
    )

    script = f'''"""Auto-generated Modal entrypoint for TRLoom."""
from pathlib import Path

from trloom.modal_support import run_on_modal

CONFIG = Path(r"{path}")

if __name__ == "__main__":
    result = run_on_modal(CONFIG)
    print(result)
'''
    dest.write_text(script, encoding="utf-8")
    logger.info("Wrote Modal entrypoint to %s", dest)
    return dest
