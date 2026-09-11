# Modal

Run the same YAML on remote GPUs with [Modal](https://modal.com). Outputs are
written to a Modal volume and can be downloaded locally.

## Setup

```bash
pip install -e ".[modal]"
python -m modal setup
```

### Secrets (when needed)

Public models and datasets need no secrets. For private Hub access or W&B:

```bash
python -m modal secret create huggingface HF_TOKEN=hf_...
python -m modal secret create wandb WANDB_API_KEY=...
```

Then list those names under `modal.secrets`. The default is an empty list.

## YAML settings

```yaml
modal:
  enabled: true
  app_name: trloom
  gpu: T4
  timeout: 14400
  volume_name: trloom-outputs
  volume_mount: /outputs
  secrets: []
  pip_packages: []
  python_version: "3.11"
  install_source: local   # local | git | pypi
  download_dir: ./outputs/remote-download
```

| Field | Purpose |
|-------|---------|
| `enabled` | Send the job to Modal |
| `gpu` | GPU type (`T4`, `A10G`, `A100`, …) |
| `timeout` | Max runtime in seconds |
| `volume_name` / `volume_mount` | Persistent output storage |
| `secrets` | Modal secret names to attach |
| `install_source` | How TRLoom is installed in the image |
| `download_dir` | Optional local path for pulled outputs |

### Install sources

| Value | Behavior |
|-------|----------|
| `local` | Mount your local package (best while developing) |
| `git` | `pip install` from `git_url` |
| `pypi` | `pip install trloom` (once published) |

Inside Modal, TRLoom forces `modal.enabled: false` and redirects
`training.output_dir` onto the volume mount so the remote process does not
recurse.

## Run

```bash
trloom run path/to/config.yaml --modal
```

Or generate a standalone script:

```bash
trloom modal-script path/to/config.yaml -o run_modal.py
python -m modal run run_modal.py
```

`--modal` and `--local` override YAML. Do not pass both.

## Smoke test

`examples/modal_smoke/` is a tiny end-to-end job:

| Piece | Choice |
|-------|--------|
| Model | `sshleifer/tiny-gpt2` |
| Dataset | `stanfordnlp/imdb` `train[:64]` |
| GPU | `T4` |
| Steps | `3` |

From the repository root:

```bash
python -m modal run examples/modal_smoke/run.py
```

Or:

```bash
trloom validate examples/modal_smoke/config.yaml
trloom run examples/modal_smoke/config.yaml --modal
```

!!! note "Windows PowerShell"
    If the Modal CLI hits encoding errors:

    ```powershell
    $env:PYTHONIOENCODING='utf-8'; $env:PYTHONUTF8='1'
    python -m modal run examples/modal_smoke/run.py
    ```

Expected result: `status: completed`, a short train loss, and checkpoint files
on volume `trloom-smoke-outputs`.

### Download outputs

```bash
python -m modal volume get trloom-smoke-outputs /modal-smoke ./outputs/modal-smoke-download
```

## Larger example

`examples/grpo_modal.yaml` shows GRPO + rewards + W&B + A100 + secrets:

```bash
trloom run examples/grpo_modal.yaml --modal
```
