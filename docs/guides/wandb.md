# Weights & Biases

Install the optional extra and authenticate once:

```bash
pip install "trloom[wandb]"
# or from source: pip install -e ".[wandb]"
wandb login
```

## Enable in YAML

```yaml
training:
  report_to: wandb   # optional; set automatically when wandb.enabled is true

wandb:
  enabled: true
  project: my-project
  entity: my-team
  run_name: qwen-sft-01
  group: experiments
  tags: [sft, lora]
  notes: First SFT baseline
  mode: online       # online | offline | disabled
  dir: null
  job_type: train
  init_kwargs: {}
```

When `wandb.enabled` is true, TRLoom:

1. Sets related environment variables when provided (`WANDB_MODE`,
   `WANDB_PROJECT`, `WANDB_ENTITY`, `WANDB_DIR`)
2. Ensures `training.report_to` includes `wandb`
3. Calls `wandb.init(...)` with your settings and the full config dump
4. Finishes the run when `job.run()` completes

## Example

See `examples/dpo_wandb.yaml` for DPO + W&B:

```bash
trloom run examples/dpo_wandb.yaml
```

## Modal + W&B

Create a Modal secret, then reference it:

```bash
python -m modal secret create wandb WANDB_API_KEY=...
```

```yaml
wandb:
  enabled: true
  project: trloom-grpo

modal:
  enabled: true
  secrets:
    - wandb
    - huggingface
```

## Troubleshooting

- Missing package → install `trloom[wandb]`
- Offline debugging → set `mode: offline`
- Disable without editing the rest of the block → `enabled: false` or
  `mode: disabled`
