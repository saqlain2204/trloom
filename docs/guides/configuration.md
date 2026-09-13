# Configuration

Every TRLoom job is a YAML file mapped to `FineTuneConfig`. Unknown keys under
several sections are allowed (`extra="allow"`) so the schema can stay close to
TRL as TRL evolves.

## Root fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `method` | string | required | TRL method key (`sft`, `dpo`, `grpo`, …). Normalized to lowercase with `_`. |
| `model` | object | required | Model / PEFT / quantization settings |
| `dataset` | object | required | Dataset source(s) |
| `training` | object | `{}` | Forwarded to the TRL `*Config` class |
| `wandb` | object | disabled | Weights & Biases settings |
| `modal` | object | disabled | Modal remote execution settings |
| `reward_funcs` | string or list | `null` | Reward names or import paths (GRPO / RLOO) |
| `formatting_func` | string | `null` | Import path for trainer formatting_func |
| `user_code` | string or list | `[]` | Local modules shipped to remote workers |
| `trainer_kwargs` | object | `{}` | Extra kwargs for the Trainer constructor |
| `push_to_hub` | bool | `false` | Upload after training |
| `hub_model_id` | string | `null` | Hub repo id when pushing |
| `seed` | int | `42` | Injected into training args if unset |

List methods available in your environment:

```bash
trloom methods
```

Common aliases: `supervised` → `sft`, `rm` → `reward`,
`direct_preference_optimization` → `dpo`.

## `model`

Aligned with TRL `ModelConfig`.

| Field | Default | Notes |
|-------|---------|-------|
| `model_name_or_path` | required | Hub id or local path |
| `model_revision` | `"main"` | |
| `dtype` | `"float32"` | `auto`, `bfloat16`, `float16`, `float32`, or null |
| `attn_implementation` | `null` | e.g. `flash_attention_2` |
| `trust_remote_code` | `false` | |
| `use_peft` | `false` | Enable LoRA / PEFT |
| `lora_r` | `16` | |
| `lora_alpha` | `32` | |
| `lora_dropout` | `0.05` | |
| `lora_target_modules` | `null` | list or string |
| `lora_target_parameters` | `null` | |
| `lora_modules_to_save` | `null` | |
| `lora_task_type` | `"CAUSAL_LM"` | |
| `use_rslora` | `false` | |
| `use_dora` | `false` | |
| `load_in_8bit` | `false` | Mutually exclusive with 4-bit |
| `load_in_4bit` | `false` | Needs `trloom[bitsandbytes]` |
| `bnb_4bit_quant_type` | `"nf4"` | `fp4` or `nf4` |
| `use_bnb_nested_quant` | `false` | |
| `bnb_4bit_quant_storage` | `null` | |

```yaml
model:
  model_name_or_path: Qwen/Qwen2.5-0.5B-Instruct
  dtype: bfloat16
  use_peft: true
  lora_r: 16
  lora_alpha: 32
  lora_target_modules:
    - q_proj
    - v_proj
```

!!! warning
    Do not set both `load_in_4bit` and `load_in_8bit`.

## `dataset`

Requires either `path` or `datasets`.

| Field | Default | Notes |
|-------|---------|-------|
| `path` | `null` | Hub id or local path |
| `name` | `null` | Hub config name |
| `split` | `null` | Load-time split, e.g. `"train[:64]"` |
| `data_files` / `data_dir` | `null` | |
| `streaming` | `false` | |
| `datasets` | `null` | Mixture of sources |
| `train_split` | `"train"` | Split key after load |
| `eval_split` | `"test"` | Set `null` to skip eval |
| `text_column` | `null` | Validated if set |
| `columns` | `null` | Rename map `{src: dst}` |
| `kwargs` | `{}` | Passthrough to `load_dataset` |
| `map_fn` | `null` | Import path for `Dataset.map` |
| `map_kwargs` | `{}` | Extra kwargs for `Dataset.map` |
| `prompt_template` | `null` | Jinja2 template → `prompt_output_column` |
| `prompt_output_column` | `"text"` | Column written by the template |
| `prompt_remove_columns` | `null` | List of columns to drop, or `true` for all |
| `formatting_func` | `null` | Import path for trainer formatting_func |

Per-source fields under `datasets[]`: `path`, `name`, `split`, `data_files`,
`data_dir`, `streaming`, `columns`, `weight`.

See [Datasets](datasets.md) for examples.

## `training`

Pass-through to the method's TRL config (`SFTConfig`, `DPOConfig`, …).

```yaml
training:
  output_dir: ./outputs/sft
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

Unsupported keys for the selected config class are logged and dropped.
Check the TRL docs for method-specific fields.

## `wandb`

See [Weights & Biases](wandb.md).

| Field | Default |
|-------|---------|
| `enabled` | `false` |
| `project` / `entity` / `run_name` / `group` / `notes` | `null` |
| `tags` | `[]` |
| `mode` | `null` (`online` / `offline` / `disabled`) |
| `dir` | `null` |
| `job_type` | `"train"` |
| `init_kwargs` | `{}` |

## `modal`

See [Modal](modal.md).

| Field | Default |
|-------|---------|
| `enabled` | `false` |
| `app_name` | `"trloom"` |
| `gpu` | `"T4"` |
| `timeout` | `14400` (4 hours) |
| `cpu` / `memory` / `region` | `null` |
| `volume_name` | `"trloom-outputs"` |
| `volume_mount` | `"/outputs"` |
| `secrets` | `[]` |
| `pip_packages` | `[]` |
| `python_version` | `"3.11"` |
| `install_source` | `"local"` (`local` / `git` / `pypi`) |
| `git_url` | repo git URL |
| `download_dir` | `null` |

## Rewards, formatting, and trainer extras

```yaml
reward_funcs:
  - accuracy_reward
  # or: my_package.rewards:my_fn

formatting_func: my_package.formatters:to_text  # trainer formatting_func

user_code:
  - ./my_package            # local modules shipped to Modal / remote workers

trainer_kwargs:
  # Extra kwargs for the Trainer constructor

push_to_hub: false
hub_model_id: null
seed: 42
```

`reward_funcs` and `formatting_func` / `dataset.map_fn` accept import paths
(`pkg.mod:func` / `pkg.mod.func`). Local modules listed in `user_code` (or
auto-detected from those import paths) are bundled when the job runs on Modal
so remote workers can import them.

## Full starter config

```yaml
method: sft

model:
  model_name_or_path: Qwen/Qwen2.5-0.5B-Instruct
  dtype: bfloat16
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
