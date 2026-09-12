# TRLoom

<p align="center">
  <img src="assets/trloom.png" alt="TRLoom" width="240" />
</p>

**TRLoom** turns a single YAML file into an end-to-end
[Hugging Face TRL](https://huggingface.co/docs/trl) fine-tuning job.

Configure the model, dataset, trainer, Weights & Biases, and optional
[Modal](https://modal.com) GPU execution, then run one command.

## Why TRLoom?

TRL is powerful, but wiring model loading, datasets, trainer configs, logging,
and remote compute usually means a custom script every time. TRLoom keeps that
in one config:

```yaml
method: sft
model:
  model_name_or_path: Qwen/Qwen2.5-0.5B-Instruct
  use_peft: true
dataset:
  path: trl-lib/Capybara
  train_split: train
training:
  output_dir: ./outputs/sft
  learning_rate: 2.0e-4
  num_train_epochs: 1
```

```bash
trloom run sft.yaml
```

## Install

```bash
pip install trloom
# or
pip install git+https://github.com/saqlain2204/trloom.git
```

See [Getting started](getting-started.md) for extras and editable installs.

## Features

- **TRL-native** — discovers trainers from your installed TRL version (SFT, DPO,
  GRPO, KTO, Reward, RLOO, and experimental methods)
- **YAML-first** — model, dataset, training args, W&B, and Modal in one file
- **Flexible datasets** — Hub repos, local files, saved datasets dirs, mixtures
- **W&B** — enable and configure logging from YAML
- **Modal** — run the same YAML on remote GPUs with volume-backed outputs
- **CLI + Python API** — `trloom run` or `FineTuneJob.from_yaml(...)`

## Quick links

| Page | What you get |
|------|----------------|
| [Getting started](getting-started.md) | Install, first job, validate + run |
| [Configuration](guides/configuration.md) | Full YAML reference |
| [Datasets](guides/datasets.md) | Hub, local, and mixture loading |
| [Weights & Biases](guides/wandb.md) | Experiment tracking |
| [Modal](guides/modal.md) | Remote GPU execution |
| [CLI](cli.md) | Command reference |
| [API reference](api.md) | Python API |

## License

Apache-2.0
