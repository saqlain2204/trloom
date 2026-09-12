# Getting started

## Requirements

- Python 3.10+
- A working PyTorch + TRL environment for real training

## Installation

### From PyPI (recommended)

```bash
pip install trloom

# Optional extras
pip install "trloom[wandb]"         # Weights & Biases
pip install "trloom[modal]"         # Modal remote GPUs
pip install "trloom[bitsandbytes]"  # 4-bit / 8-bit loading
pip install "trloom[all]"           # wandb + modal + bitsandbytes
pip install "trloom[dev]"           # pytest, ruff
pip install "trloom[docs]"          # mkdocs
```

### From Git

```bash
pip install git+https://github.com/saqlain2204/trloom.git

# With extras
pip install "trloom[modal] @ git+https://github.com/saqlain2204/trloom.git"
pip install "trloom[all] @ git+https://github.com/saqlain2204/trloom.git"
```

### From source (editable)

Clone the repository:

```bash
git clone https://github.com/saqlain2204/trloom.git
cd trloom
```

#### One-command bootstrap

Works on Windows, macOS, and Linux. Creates `.venv` if it does not exist.

```bash
# Local install only
python scripts/bootstrap.py

# Install Modal extra and run Modal auth setup
python scripts/bootstrap.py --modal
```

Activate the venv afterward:

```bash
# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Optional flags: `--wandb`, `--dev`, `--all`, `--skip-modal-setup`.

#### Manual editable install

```bash
pip install -e .
pip install -e ".[wandb]"         # Weights & Biases
pip install -e ".[modal]"         # Modal remote GPUs
pip install -e ".[bitsandbytes]"  # 4-bit / 8-bit loading
pip install -e ".[all]"           # wandb + modal + bitsandbytes
pip install -e ".[dev]"           # pytest, ruff
```

Check the CLI:

```bash
trloom --version
trloom methods
```

`trloom methods` lists trainers discovered from your installed TRL version.

## First job

Create `sft.yaml`:

```yaml
method: sft

model:
  model_name_or_path: Qwen/Qwen2.5-0.5B-Instruct
  use_peft: true
  lora_r: 16
  lora_alpha: 32

dataset:
  path: trl-lib/Capybara
  train_split: train
  eval_split: null

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

Validate, then run:

```bash
trloom validate sft.yaml
trloom run sft.yaml
```

Or from Python:

```python
from trloom import FineTuneJob, run_from_yaml

run_from_yaml("sft.yaml")

# Or step through the API
job = FineTuneJob.from_yaml("sft.yaml")
job.run()
```

## What happens on `run`

1. Load and validate the YAML
2. Optionally start Weights & Biases
3. Build the matching TRL trainer and config
4. Load the dataset
5. Train, save to `training.output_dir`, optionally push to the Hub
6. Finish the W&B run if one was started

## Next steps

- [Configuration reference](guides/configuration.md)
- [Dataset loading](guides/datasets.md)
- [Modal smoke example](guides/modal.md#smoke-test)
- [Example configs](guides/examples.md)
