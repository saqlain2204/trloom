# Fine Tune LLMs With One YAML File: A Practical Guide to TRLoom

Fine tuning with Hugging Face TRL is powerful, but the setup can get messy. You wire up a model, load a dataset, pick a trainer, set training args, maybe add Weights and Biases, and sometimes send the job to a remote GPU. That is a lot of Python before you even start training.

**TRLoom** is a small open source library that turns that whole flow into one YAML file. You describe the job. TRLoom builds the TRL trainer and runs it, either on your machine or on [Modal](https://modal.com).

This article is a clear tutorial. You will learn:

- What TRLoom is and when it helps
- How to install it
- How every config section works
- How to run SFT, DPO, and GRPO style jobs
- How to use Hugging Face Hub data or your own files
- How to plug in Weights and Biases
- How to run the same YAML on Modal GPUs
- How to use the Python API if you prefer code over the CLI

Repo: [https://github.com/saqlain2204/trloom](https://github.com/saqlain2204/trloom)

---

## What is TRLoom?

TRLoom sits on top of [TRL](https://huggingface.co/docs/trl) (Transformers Reinforcement Learning). It does not invent a new training algorithm. It reads your config, maps it to the right TRL trainer and config class, loads the dataset, and runs the job.

In short:

1. You write a YAML file
2. You run `trloom run config.yaml`
3. TRLoom loads the model, data, and trainer, then trains and saves the result

It also supports:

- **Weights and Biases** logging from YAML
- **Modal** remote GPU runs from the same YAML
- **Local or Hub** datasets
- **PEFT / LoRA** and optional bitsandbytes quantization
- A **CLI** and a **Python API**

Method support follows your installed TRL version. Common ones include `sft`, `dpo`, `grpo`, `kto`, `reward`, and `rloo`, plus experimental methods when TRL exposes them.

---

## Install

You need Python 3.10+, PyTorch, and a working TRL stack.

### From PyPI

```bash
pip install trloom
pip install "trloom[wandb]"        # Weights and Biases
pip install "trloom[modal]"        # Modal remote GPUs
pip install "trloom[bitsandbytes]" # 4-bit / 8-bit loading
pip install "trloom[all]"          # wandb + modal + bitsandbytes
```

### From Git

```bash
pip install git+https://github.com/saqlain2204/trloom.git
pip install "trloom[modal] @ git+https://github.com/saqlain2204/trloom.git"
```

### From source (editable)

```bash
git clone https://github.com/saqlain2204/trloom.git
cd trloom
pip install -e .
pip install -e ".[wandb]"
pip install -e ".[modal]"
pip install -e ".[bitsandbytes]"
pip install -e ".[all]"
```
Check that the CLI works:

```bash
trloom --version
trloom methods
```

`trloom methods` prints the trainers TRLoom discovered from your TRL install. That list can grow when you upgrade TRL.

---

## The big idea: one YAML drives everything

A TRLoom config has a few top level sections:

| Section | What it does |
|--------|---------------|
| `method` | Which TRL method to use (`sft`, `dpo`, `grpo`, ...) |
| `model` | Base model, LoRA, dtype, quantization |
| `dataset` | Hub repo, local files, or a mixture |
| `training` | Args passed through to TRL (`SFTConfig`, `DPOConfig`, ...) |
| `wandb` | Optional Weights and Biases settings |
| `modal` | Optional remote GPU settings |
| `reward_funcs` | Reward functions for GRPO / RLOO style jobs |
| `trainer_kwargs` | Extra kwargs for the Trainer constructor |
| `push_to_hub` / `hub_model_id` | Optional upload to the Hugging Face Hub |

Most unknown keys under `model` and `training` are allowed so TRLoom can stay close to TRL as TRL evolves. If a training key is not valid for that method, TRLoom logs a warning and drops it.

---

## Your first job: supervised fine tuning from the Hub

Here is a solid starter config. It fine tunes a small instruct model with LoRA on a Hub dataset.

Save this as `sft.yaml` (or use `examples/sft_hub.yaml` from the repo):

```yaml
method: sft

model:
  model_name_or_path: Qwen/Qwen2.5-0.5B-Instruct
  dtype: bfloat16
  use_peft: true
  lora_r: 16
  lora_alpha: 32
  lora_target_modules:
    - q_proj
    - v_proj

dataset:
  path: trl-lib/Capybara
  train_split: train
  eval_split: null

training:
  output_dir: ./outputs/sft-capybara
  learning_rate: 2.0e-4
  num_train_epochs: 1
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 4
  logging_steps: 10
  save_steps: 100
  bf16: true
  report_to: none

wandb:
  enabled: false

modal:
  enabled: false
```

Validate first, then run:

```bash
trloom validate sft.yaml
trloom run sft.yaml
```

`validate` prints a small JSON summary: method, trainer class, output dir, and whether W&B or Modal is on. Use it before you burn GPU time.

---

## Configuration deep dive

### 1. `method`

This picks the TRL trainer. Examples:

- `sft` for supervised fine tuning
- `dpo` for direct preference optimization
- `grpo` for group relative policy optimization
- `kto`, `reward`, `rloo`, and more depending on your TRL version

Aliases work too. For example `supervised` maps to `sft`, and `rm` maps to `reward`.

Always run `trloom methods` if you are unsure what your environment supports.

### 2. `model`

Important fields:

| Field | Meaning |
|------|---------|
| `model_name_or_path` | Hub id or local model path (required) |
| `dtype` | `float32`, `float16`, `bfloat16`, or `auto` |
| `trust_remote_code` | Needed for some custom models |
| `use_peft` | Turn on LoRA / PEFT |
| `lora_r`, `lora_alpha`, `lora_dropout` | LoRA hyperparameters |
| `lora_target_modules` | Which layers get adapters |
| `load_in_4bit` / `load_in_8bit` | Quantized loading (do not enable both) |
| `bnb_4bit_quant_type` | Usually `nf4` or `fp4` |

Example LoRA block:

```yaml
model:
  model_name_or_path: Qwen/Qwen2.5-0.5B-Instruct
  use_peft: true
  lora_r: 16
  lora_alpha: 32
  lora_dropout: 0.05
  lora_target_modules:
    - q_proj
    - v_proj
```

For 4-bit work, install `trloom[bitsandbytes]` and set `load_in_4bit: true`.

### 3. `dataset`

You can load data in three main ways.

**From the Hugging Face Hub**

```yaml
dataset:
  path: trl-lib/Capybara
  train_split: train
  eval_split: null
```

**From a local file**

Supported file types include json, jsonl, csv, tsv, parquet, txt, and arrow. Local directories can also be loaded.

```yaml
dataset:
  path: ./data/train.jsonl
  train_split: train
  eval_split: null
```

**From a mixture**

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

Other useful fields:

- `split`: Hub subset at load time, for example `"train[:64]"` for a tiny smoke run
- `name`: dataset config name on the Hub
- `text_column`: column to treat as text
- `columns`: rename map from source names to target names
- `streaming`: stream large datasets
- `kwargs`: extra args passed to `load_dataset`

Tip: the default `eval_split` is `"test"`. If your dataset has no test split, set `eval_split: null`.

### 4. `training`

This section is mostly a pass through to TRL. Put the same knobs you would put in `SFTConfig`, `DPOConfig`, and so on.

Common fields:

```yaml
training:
  output_dir: ./outputs/my-run
  learning_rate: 2.0e-4
  num_train_epochs: 1
  max_steps: 100
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 4
  logging_steps: 10
  save_steps: 100
  bf16: true
  report_to: none
  max_length: 512
  dataset_text_field: text
```

Exact valid keys depend on the method. When in doubt, check the TRL docs for that trainer config.

### 5. Root extras

```yaml
seed: 42
push_to_hub: false
hub_model_id: null
reward_funcs:
  - accuracy_reward
trainer_kwargs: {}
```

`reward_funcs` can be a TRL built in name like `accuracy_reward`, or an import path like `my_package.rewards:my_fn`.

---

## Train on your own data

The repo ships a tiny local example: `examples/sft_local.yaml`.

```yaml
method: sft

model:
  model_name_or_path: Qwen/Qwen2.5-0.5B-Instruct
  use_peft: true
  lora_r: 8
  lora_alpha: 16

dataset:
  path: examples/data/sample_sft.jsonl
  train_split: train
  eval_split: null

training:
  output_dir: ./outputs/sft-local
  learning_rate: 2.0e-4
  num_train_epochs: 1
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8
  logging_steps: 5
  report_to: none

wandb:
  enabled: false

modal:
  enabled: false
```

Run it from the repo root:

```bash
trloom run examples/sft_local.yaml
```

For chat style SFT, your JSONL rows usually look like TRL expects, for example a `messages` field with role and content turns. Match the format to the trainer you chose.

---

## Preference tuning with DPO and Weights and Biases

Here is a DPO example with W&B enabled (`examples/dpo_wandb.yaml`):

```yaml
method: dpo

model:
  model_name_or_path: Qwen/Qwen2.5-0.5B-Instruct
  dtype: bfloat16
  use_peft: true
  lora_r: 16
  lora_alpha: 32

dataset:
  path: trl-lib/ultrafeedback_binarized
  train_split: train
  eval_split: test

training:
  output_dir: ./outputs/dpo-ultrafeedback
  learning_rate: 5.0e-6
  num_train_epochs: 1
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8
  logging_steps: 10
  bf16: true
  report_to: wandb

wandb:
  enabled: true
  project: trloom-dpo
  run_name: qwen05b-ultrafeedback
  tags:
    - dpo
    - example

modal:
  enabled: false
```

Install the W&B extra and log in once:

```bash
pip install -e ".[wandb]"
wandb login
```

Then:

```bash
trloom run examples/dpo_wandb.yaml
```

### W&B fields explained

| Field | Purpose |
|------|---------|
| `enabled` | Turn W&B on or off |
| `project` | W&B project name |
| `entity` | Team or user |
| `run_name` | Display name for the run |
| `group` | Group related runs |
| `tags` | Searchable tags |
| `notes` | Free text notes |
| `mode` | `online`, `offline`, or `disabled` |
| `dir` | Local W&B directory |
| `job_type` | Defaults to `train` |
| `init_kwargs` | Extra kwargs for `wandb.init` |

When `wandb.enabled` is true, TRLoom sets `training.report_to` to include `wandb` for you if needed. You still need the `wandb` package installed.

---

## Run on Modal GPUs

Modal lets you run the same YAML on a remote GPU without managing your own cloud setup.

### One time setup

```bash
pip install -e ".[modal]"
python -m modal setup
```

If you need private Hub models or W&B on Modal, create secrets once:

```bash
python -m modal secret create huggingface HF_TOKEN=hf_...
python -m modal secret create wandb WANDB_API_KEY=...
```

Then list those secret names under `modal.secrets` in YAML.

### Modal config fields

| Field | Purpose | Default idea |
|------|---------|--------------|
| `enabled` | Send the job to Modal | `false` |
| `app_name` | Modal app name | `trloom` |
| `gpu` | GPU type | `T4` |
| `timeout` | Max seconds | `14400` (4 hours) |
| `volume_name` | Persistent volume for outputs | `trloom-outputs` |
| `volume_mount` | Mount path inside the container | `/outputs` |
| `secrets` | Modal secret names to attach | `[]` |
| `pip_packages` | Extra pip packages in the image | `[]` |
| `python_version` | Image Python version | `3.11` |
| `install_source` | How to install TRLoom: `local`, `git`, or `pypi` | `local` |
| `download_dir` | Optional local folder to pull results into | unset |

`install_source: local` mounts your checkout. That is great while developing. For cleaner remote runs later, use `git` or `pypi` once the package is published the way you want.

### Tiny smoke test (recommended first Modal run)

The repo includes `examples/modal_smoke/`. It trains `sshleifer/tiny-gpt2` for 3 steps on 64 IMDB rows on a T4. No secrets needed because the model and data are public.

Config highlight:

```yaml
method: sft

model:
  model_name_or_path: sshleifer/tiny-gpt2
  dtype: float32
  use_peft: false

dataset:
  path: stanfordnlp/imdb
  split: "train[:64]"
  train_split: train
  eval_split: null
  text_column: text

training:
  output_dir: ./outputs/modal-smoke
  max_steps: 3
  per_device_train_batch_size: 2
  learning_rate: 5.0e-5
  logging_steps: 1
  report_to: none
  max_length: 128
  dataset_text_field: text

modal:
  enabled: true
  app_name: trloom-modal-smoke
  gpu: T4
  timeout: 1800
  volume_name: trloom-smoke-outputs
  volume_mount: /outputs
  secrets: []
  install_source: local
```

From the repo root:

```bash
python -m modal run examples/modal_smoke/run.py
```

Or with the CLI:

```bash
trloom validate examples/modal_smoke/config.yaml
trloom run examples/modal_smoke/config.yaml --modal
```

On Windows PowerShell, set UTF-8 before calling the Modal CLI if you hit encoding errors:

```powershell
$env:PYTHONIOENCODING='utf-8'; $env:PYTHONUTF8='1'
python -m modal run examples/modal_smoke/run.py
```

When it finishes, you should see something like `status: completed`, a short train loss, and checkpoint files on the Modal volume.

Download outputs:

```bash
python -m modal volume get trloom-smoke-outputs /modal-smoke ./outputs/modal-smoke-download
```

### Larger Modal example: GRPO

`examples/grpo_modal.yaml` shows a fuller setup: GRPO, rewards, W&B, A100, and secrets.

```yaml
method: grpo

model:
  model_name_or_path: Qwen/Qwen2.5-0.5B-Instruct
  dtype: bfloat16
  use_peft: true
  lora_r: 16
  lora_alpha: 32

dataset:
  path: HuggingFaceH4/Polaris-Dataset-53K
  train_split: train
  eval_split: null

reward_funcs:
  - accuracy_reward

training:
  output_dir: ./outputs/grpo-modal
  learning_rate: 1.0e-5
  num_train_epochs: 1
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 4
  logging_steps: 5
  bf16: true
  report_to: wandb

wandb:
  enabled: true
  project: trloom-grpo
  run_name: qwen05b-grpo-modal
  tags:
    - grpo
    - modal

modal:
  enabled: true
  app_name: trloom-grpo
  gpu: A100
  timeout: 14400
  volume_name: trloom-outputs
  volume_mount: /outputs
  secrets:
    - huggingface
    - wandb
  download_dir: ./outputs/grpo-modal-download
```

Run:

```bash
trloom run examples/grpo_modal.yaml --modal
```

You can also generate a standalone Modal script:

```bash
trloom modal-script examples/grpo_modal.yaml -o run_modal.py
python -m modal run run_modal.py
```

`--modal` and `--local` override the YAML. Do not pass both.

---

## CLI cheat sheet

```bash
trloom --version
trloom methods
trloom validate path/to/config.yaml
trloom run path/to/config.yaml
trloom run path/to/config.yaml --modal
trloom run path/to/config.yaml --local
trloom modal-script path/to/config.yaml -o run_modal.py
```

You can also use `python -m trloom ...`.

Add `-v` or `--verbose` when you want more logs.

---

## Python API

If you prefer code, TRLoom exposes a small API:

```python
from trloom import FineTuneJob, run_from_yaml, available_methods, load_config

print(available_methods())

# One liner
run_from_yaml("sft.yaml")

# Step by step
config = load_config("sft.yaml")
job = FineTuneJob(config)
job.build()   # build the TRL trainer
job.run()     # train, save, finish W&B if needed
```

Useful helpers:

| API | Role |
|-----|------|
| `FineTuneJob.from_yaml(path)` | Load YAML into a job |
| `FineTuneJob.from_dict(data)` | Build from a Python dict |
| `job.build()` | Construct the trainer |
| `job.train()` / `job.run()` | Train end to end |
| `run_from_yaml(path, use_modal=None)` | Load and run, honoring Modal unless overridden |
| `load_config(path)` | Validate and return `FineTuneConfig` |
| `available_methods()` | List discovered methods |

Typical lifecycle:

1. Load and validate config
2. Build trainer (and start W&B if enabled)
3. Train
4. Save to `output_dir`
5. Optionally push to the Hub
6. Finish the W&B run

---

## A simple workflow you can reuse

1. Pick a method with `trloom methods`
2. Copy the closest example YAML
3. Change `model.model_name_or_path` and `dataset.path`
4. Keep the first run tiny (`max_steps`, small split, small model)
5. Run `trloom validate ...`
6. Run local first if you can
7. Turn on W&B when the pipeline looks right
8. Flip `modal.enabled: true` (or pass `--modal`) for bigger GPUs
9. Pull checkpoints from the Modal volume when done

That pattern saves time and money. Prove the YAML, then scale the compute.

---

## Common gotchas

1. **Extras are required.** `wandb.enabled: true` needs `trloom[wandb]`. Modal needs `trloom[modal]`.
2. **Secrets are opt in.** Modal does not attach Hub or W&B secrets unless you list them under `modal.secrets`.
3. **Eval split defaults to `test`.** Set `eval_split: null` when there is no test split.
4. **Do not enable both 4-bit and 8-bit.** Pick one.
5. **Training keys must match the method.** Wrong keys are dropped with a warning.
6. **Method list depends on TRL.** Upgrade TRL, then check `trloom methods` again.
7. **Windows + Modal CLI.** Set `PYTHONIOENCODING` and `PYTHONUTF8` if you see encoding errors.
8. **Run Modal smoke from the repo root.** The smoke script expects the normal `src/trloom` layout.

---

## What you get at the end of a run

Locally, checkpoints land in `training.output_dir`.

On Modal, they land on the volume under the mount path, for example `/outputs/...`. You can download them with `modal volume get` or by setting `modal.download_dir`.

If `push_to_hub` is true and auth is set up, TRLoom can also upload the finished model.

---

## Wrap up

TRLoom is for people who want TRL power without rebuilding the same training script every time. One YAML can describe:

- the method
- the model and LoRA settings
- Hub or local data
- training hyperparameters
- W&B logging
- Modal GPU execution

Start with the Hub SFT example, try the local JSONL example, add W&B when you care about metrics, then use the Modal smoke job to prove remote execution before you spend money on a big model.

If you try it, the examples folder is the fastest place to begin:

- `examples/sft_hub.yaml`
- `examples/sft_local.yaml`
- `examples/dpo_wandb.yaml`
- `examples/grpo_modal.yaml`
- `examples/modal_smoke/`

Repo again: [https://github.com/saqlain2204/trloom](https://github.com/saqlain2204/trloom)

Happy fine tuning.
