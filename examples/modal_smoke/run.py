"""End-to-end Modal smoke job for TRLoom.

Run from the repository root (with the package installed):

    pip install -e ".[modal]"
    python -m modal run examples/modal_smoke/run.py

This launches a 3-step SFT job on a T4 using ``sshleifer/tiny-gpt2`` and
64 rows of ``stanfordnlp/imdb`` from the Hugging Face Hub.
"""

from __future__ import annotations

from pathlib import Path

import modal


def _resolve_package_dir() -> Path:
    """Local checkout path on the client; mounted path inside the Modal container."""
    here = Path(__file__).resolve()
    try:
        local = here.parents[2] / "src" / "trloom"
        if local.is_dir():
            return local
    except IndexError:
        pass
    remote = Path("/root/trloom")
    if remote.is_dir():
        return remote
    raise FileNotFoundError(
        "Could not locate the trloom package. Run from the repo root so "
        "examples/modal_smoke/run.py can find src/trloom."
    )


PACKAGE_DIR = _resolve_package_dir()

# Dependencies installed inside the Modal container.
# ``trloom`` itself is mounted from the local checkout.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "trl",
        "transformers",
        "datasets",
        "accelerate",
        "peft",
        "pyyaml",
        "pydantic",
        "huggingface-hub",
    )
    .env({"PYTHONPATH": "/root", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir(str(PACKAGE_DIR), remote_path="/root/trloom")
)

app = modal.App("trloom-modal-smoke", image=image)
volume = modal.Volume.from_name("trloom-smoke-outputs", create_if_missing=True)


@app.function(
    gpu="T4",
    timeout=30 * 60,
    volumes={"/outputs": volume},
)
def train_smoke(config_yaml: str) -> dict:
    """Run the YAML job inside Modal and write checkpoints to the volume."""
    from pathlib import Path as _Path

    import yaml

    from trloom.config.loader import load_config
    from trloom.job import FineTuneJob

    run_dir = _Path("/outputs/modal-smoke")
    run_dir.mkdir(parents=True, exist_ok=True)

    data = yaml.safe_load(config_yaml)
    data.setdefault("modal", {})
    data["modal"]["enabled"] = False  # already inside Modal
    data.setdefault("training", {})
    data["training"]["output_dir"] = str(run_dir)

    config = load_config(data)
    result = FineTuneJob(config).run()
    volume.commit()

    metrics = getattr(result, "metrics", None)
    return {
        "status": "completed",
        "output_dir": str(run_dir),
        "metrics": dict(metrics) if isinstance(metrics, dict) else None,
        "checkpoint_files": sorted(p.name for p in run_dir.iterdir()) if run_dir.exists() else [],
    }


@app.local_entrypoint()
def main() -> None:
    """Local entry: read YAML, submit remote train, print the result."""
    config_path = Path(__file__).resolve().parent / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing config: {config_path}")

    print(f"Submitting smoke job from {config_path}")
    print("  model: sshleifer/tiny-gpt2")
    print("  dataset: stanfordnlp/imdb train[:64]")
    print("  gpu: T4 | max_steps: 3")

    result = train_smoke.remote(config_path.read_text(encoding="utf-8"))
    print("Smoke job finished:")
    for key, value in result.items():
        print(f"  {key}: {value}")
