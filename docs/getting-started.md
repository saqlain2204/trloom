# Getting started

## Requirements

- Python 3.10+
- A working PyTorch + TRL environment for real training

## Installation

Clone the repository and install in editable mode:

```bash
git clone https://github.com/saqlain2204/trloom.git
cd trloom
pip install -e .
```

### Optional extras

```bash
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
