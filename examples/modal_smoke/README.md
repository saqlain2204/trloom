# Modal smoke example (end-to-end)

This folder is a **complete tiny fine-tuning workflow** on [Modal](https://modal.com)
using TRLoom: public Hugging Face model + dataset, 3 training steps, T4 GPU.

| Piece | Choice |
|-------|--------|
| Method | SFT |
| Model | [`sshleifer/tiny-gpt2`](https://huggingface.co/sshleifer/tiny-gpt2) |
| Dataset | [`stanfordnlp/imdb`](https://huggingface.co/datasets/stanfordnlp/imdb) `train[:64]` |
| GPU | `T4` |
| Steps | `3` |

No Hugging Face or W&B secrets are required (everything used here is public).

## Prerequisites

1. A Modal account and CLI auth (one-time):

   ```bash
   pip install -e ".[modal]"
   python -m modal setup
   ```

2. Run commands from the **repository root**.

## Run the smoke job

From the repository root:

```bash
pip install -e ".[modal]"
python -m modal setup   # one-time auth, if needed

# On Windows PowerShell, set UTF-8 to avoid Modal CLI encoding issues:
#   $env:PYTHONIOENCODING='utf-8'; $env:PYTHONUTF8='1'

python -m modal run examples/modal_smoke/run.py
```

Expected result: `status: completed`, 3 training steps, and checkpoint files
(`model.safetensors`, tokenizer files, …) on the Modal volume.

What happens:

1. Modal builds an image with PyTorch + TRL + deps
2. Mounts local `src/trloom` into the container
3. Starts a T4 GPU function
4. Loads 64 IMDB rows from the Hub and runs 3 SFT steps on `tiny-gpt2`
5. Writes outputs to the Modal volume `trloom-smoke-outputs` at `/outputs/modal-smoke`
6. Prints status, metrics, and checkpoint file names

## Optional: run via the TRLoom API / CLI

Same YAML, library-managed Modal app:

```bash
# after: pip install -e ".[modal]"
trloom validate examples/modal_smoke/config.yaml
trloom run examples/modal_smoke/config.yaml --modal
```

Or:

```python
from trloom import run_from_yaml
print(run_from_yaml("examples/modal_smoke/config.yaml", use_modal=True))
```

## Fetch outputs from the volume

```bash
python -m modal volume get trloom-smoke-outputs /modal-smoke ./outputs/modal-smoke-download
```

## Add secrets later (private models / W&B)

```bash
python -m modal secret create huggingface HF_TOKEN=hf_...
python -m modal secret create wandb WANDB_API_KEY=...
```

Then in `config.yaml`:

```yaml
modal:
  secrets: [huggingface, wandb]
wandb:
  enabled: true
  project: trloom-smoke
```
