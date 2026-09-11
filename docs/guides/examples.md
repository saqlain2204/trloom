# Examples

All example configs live in the repository [`examples/`](https://github.com/saqlain2204/trloom/tree/main/examples) folder.

| Path | What it shows |
|------|----------------|
| `examples/sft_hub.yaml` | SFT from a Hub dataset with LoRA |
| `examples/sft_local.yaml` | SFT from local JSONL |
| `examples/data/sample_sft.jsonl` | Tiny local chat-style sample |
| `examples/dpo_wandb.yaml` | DPO with Weights & Biases |
| `examples/grpo_modal.yaml` | GRPO on Modal with rewards and secrets |
| `examples/modal_smoke/` | Complete tiny Modal walkthrough |

## Run patterns

```bash
# Local SFT from the Hub
trloom validate examples/sft_hub.yaml
trloom run examples/sft_hub.yaml

# Local JSONL
trloom run examples/sft_local.yaml

# DPO + W&B (needs trloom[wandb] and wandb login)
trloom run examples/dpo_wandb.yaml

# Tiny Modal smoke (needs trloom[modal] and modal setup)
python -m modal run examples/modal_smoke/run.py
# or
trloom run examples/modal_smoke/config.yaml --modal
```

## Suggested learning path

1. Start with `sft_hub.yaml` or `sft_local.yaml` locally
2. Add W&B using `dpo_wandb.yaml` as a template
3. Prove remote execution with `modal_smoke/`
4. Scale up with `grpo_modal.yaml` (GPU, secrets, rewards)

Keep the first remote run tiny (`max_steps`, small `split`, small model) before
spending money on a full training job.
