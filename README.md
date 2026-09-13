# TRLoom

<p align="center">
  <img src="https://raw.githubusercontent.com/saqlain2204/trloom/main/assets/trloom.png" alt="TRLoom" width="280" />
</p>

<p align="center">
  <a href="https://pypi.org/project/trloom/"><img src="https://img.shields.io/pypi/v/trloom.svg" alt="PyPI"></a>
  <a href="https://pypi.org/project/trloom/"><img src="https://img.shields.io/pypi/pyversions/trloom.svg" alt="Python"></a>
  <a href="https://github.com/saqlain2204/trloom/actions/workflows/ci.yml"><img src="https://github.com/saqlain2204/trloom/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://saqlain2204.github.io/trloom/"><img src="https://img.shields.io/badge/docs-mkdocs-blue" alt="Docs"></a>
  <a href="https://github.com/saqlain2204/trloom/stargazers"><img src="https://img.shields.io/github/stars/saqlain2204/trloom?style=social" alt="GitHub stars"></a>
  <a href="https://github.com/saqlain2204/trloom/blob/main/LICENSE"><img src="https://img.shields.io/github/license/saqlain2204/trloom" alt="License"></a>
</p>

**TRLoom** weaves a single YAML config into an end-to-end [Hugging Face TRL](https://huggingface.co/docs/trl) fine-tuning job.

Configure the model, dataset, trainer, Weights & Biases, and optional [Modal](https://modal.com) GPU execution — then run one command.

**Docs:** [https://saqlain2204.github.io/trloom/](https://saqlain2204.github.io/trloom/)

## Features

- **TRL-native** — discovers trainers/configs from your installed TRL version (SFT, DPO, GRPO, KTO, Reward, RLOO, and experimental methods)
- **YAML-first** — model, dataset, training args, W&B, and Modal all live in one file
- **Datasets** — Hugging Face Hub, local files (`json`/`jsonl`/`csv`/`parquet`/…), saved `datasets` directories, and mixtures
- **Formatting** — Jinja2 prompt templates, YAML-referenced `map_fn` / `formatting_func`, bundled to Modal with the job
- **W&B** — enable and configure logging entirely from YAML
- **Modal** — launch the same YAML on remote GPUs with volume-backed outputs
- **CLI + Python API** — `trloom run config.yaml` or `FineTuneJob.from_yaml(...)`

## Installation

Requires Python 3.10+ and a working TRL / PyTorch environment for actual training.

### From PyPI (recommended)

```bash
pip install trloom

# Optional extras
pip install "trloom[wandb]"
pip install "trloom[modal]"
pip install "trloom[bitsandbytes]"
pip install "trloom[all]"      # wandb + modal + bitsandbytes
pip install "trloom[dev]"      # pytest, ruff
pip install "trloom[docs]"     # mkdocs
```

### From Git

```bash
pip install git+https://github.com/saqlain2204/trloom.git

# With extras
pip install "trloom[modal] @ git+https://github.com/saqlain2204/trloom.git"
pip install "trloom[all] @ git+https://github.com/saqlain2204/trloom.git"
```

### From source (editable)

Clone the repo, then either bootstrap or install manually:

```bash
git clone https://github.com/saqlain2204/trloom.git
cd trloom

# One-command bootstrap (creates .venv if needed)
python scripts/bootstrap.py
# Optional: --modal, --wandb, --dev, --all, --skip-modal-setup

# Or editable install
pip install -e .
pip install -e ".[wandb]"
pip install -e ".[modal]"
pip install -e ".[docs]"
pip install -e ".[all]"
pip install -e ".[dev]"
```

Activate the venv after bootstrap:

```bash
# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

## Quickstart

### 1. Write a config

```yaml
# sft.yaml
method: sft

model:
  model_name_or_path: Qwen/Qwen2.5-0.5B-Instruct
  use_peft: true
  lora_r: 16
  lora_alpha: 32

dataset:
  path: trl-lib/Capybara
  train_split: train

training:
  output_dir: ./outputs/sft
  learning_rate: 2.0e-4
  num_train_epochs: 1
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 4
  report_to: none

wandb:
  enabled: false

modal:
  enabled: false
```

### 2. Run

```bash
trloom validate sft.yaml
trloom run sft.yaml
```

Or from Python:

```python
from trloom import FineTuneJob, run_from_yaml

# One-liner
run_from_yaml("sft.yaml")

# Or step through the API
job = FineTuneJob.from_yaml("sft.yaml")
job.run()
```

## Documentation

Full guides and API reference:

- Online: [saqlain2204.github.io/trloom](https://saqlain2204.github.io/trloom/)
- Local preview:

```bash
pip install -e ".[docs]"
mkdocs serve
```

Coverage includes configuration reference, datasets, W&B, Modal, CLI, and the Python API.

## Configuration reference

| Section | Purpose |
|--------|---------|
| `method` | TRL method key: `sft`, `dpo`, `grpo`, `kto`, `reward`, `rloo`, … |
| `model` | Model id + PEFT/quantization (aligned with TRL `ModelConfig`) |
| `dataset` | Hub repo, local path, or `datasets:` mixture |
| `training` | Forwarded to the TRL `*Config` class (`SFTConfig`, `DPOConfig`, …) |
| `wandb` | Weights & Biases project/entity/tags/mode |
| `modal` | Remote GPU execution on Modal |
| `reward_funcs` | Names or import paths for GRPO/RLOO-style rewards |
| `trainer_kwargs` | Extra kwargs passed to the Trainer constructor |
| `push_to_hub` / `hub_model_id` | Optional Hub upload after training |

List methods available in your environment:

```bash
trloom methods
```

### Dataset examples

**Hub**

```yaml
dataset:
  path: trl-lib/Capybara
  train_split: train
```

**Local JSONL**

```yaml
dataset:
  path: ./data/train.jsonl
  train_split: train
```

**Mixture**

```yaml
dataset:
  train_split: train
  datasets:
    - path: stanfordnlp/imdb
      split: train
      weight: 0.5
    - path: ./data/extra.jsonl
      weight: 0.5
```

### Weights & Biases

```yaml
training:
  report_to: wandb   # optional; set automatically when wandb.enabled is true

wandb:
  enabled: true
  project: my-project
  entity: my-team
  run_name: qwen-sft-01
  tags: [sft, lora]
  mode: online       # online | offline | disabled
```

### Modal

1. Install and authenticate: `pip install 'trloom[modal]' && modal setup`
2. Create secrets named in the config (default: `huggingface`, `wandb`)
3. Set `modal.enabled: true` (or pass `--modal`)

```bash
trloom run examples/grpo_modal.yaml --modal
# or generate a standalone script
trloom modal-script examples/grpo_modal.yaml -o run_modal.py
modal run run_modal.py
```

## Python API

```python
from trloom import FineTuneJob, available_methods, load_config

print(available_methods())

config = load_config("sft.yaml")
job = FineTuneJob(config)
trainer = job.build()   # construct TRL trainer
job.run()               # train + save (+ optional Hub push)
```

| Method | Description |
|--------|-------------|
| `FineTuneJob.from_yaml(path)` | Load YAML into a job |
| `FineTuneJob.from_dict(data)` | Load an in-memory config |
| `job.build()` | Construct the TRL trainer |
| `job.train()` / `job.run()` | Run training end-to-end |
| `run_from_yaml(path)` | Load + run (honors `modal.enabled`) |
| `load_config(path)` | Validate and return `FineTuneConfig` |
| `available_methods()` | List TRL methods for this install |

## Examples

See the [`examples/`](examples/) directory:

- `modal_smoke/` — **complete end-to-end Modal walkthrough** (tiny GPT-2, 3 steps, T4)
- `sft_hub.yaml` — SFT from the Hub
- `sft_local.yaml` — SFT from local JSONL
- `dpo_wandb.yaml` — DPO with W&B
- `grpo_modal.yaml` — GRPO on Modal

## Development

```bash
pip install -e ".[dev]"
pytest

# Documentation
pip install -e ".[docs]"
mkdocs serve
mkdocs build --strict
```

## License

Apache-2.0
