# Kaggle

Run the same YAML on [Kaggle](https://www.kaggle.com/) GPUs. TRLoom can:

1. **Detect** that it is already inside a Kaggle kernel and train inline
2. **Generate** a notebook (+ `kernel-metadata.json`) for upload
3. **Push** that notebook via the Kaggle API (`kaggle.push: true`)

Outputs default to `/kaggle/working/outputs`.

## Setup

```bash
pip install "trloom[kaggle]"
# Configure API credentials (for push):
#   ~/.kaggle/kaggle.json  or  KAGGLE_USERNAME + KAGGLE_KEY
```

In a Kaggle notebook session, also enable a **GPU** accelerator when you need CUDA.

## YAML settings

```yaml
kaggle:
  enabled: true
  mode: auto              # auto | inline | notebook
  accelerator: gpu        # gpu | tpu | none
  push: false
  username: null          # or set KAGGLE_USERNAME
  kernel_slug: null       # defaults to trloom-<config-stem>
  title: TRLoom Fine-Tune
  is_private: true
  enable_internet: true
  dataset_sources: []
  competition_sources: []
  kernel_sources: []
  pip_packages: []
  install_source: pypi    # pypi | git
  git_url: git+https://github.com/saqlain2204/trloom.git
  output_dir: /kaggle/working/outputs
  notebook_path: null
  download_dir: null
  wait: true
  timeout_seconds: 32400
```

| Field | Purpose |
|-------|---------|
| `enabled` | Prefer the Kaggle backend |
| `mode` | `auto` picks inline vs notebook; `inline` runs now; `notebook` writes an `.ipynb` |
| `accelerator` | Written into kernel metadata (`gpu` / `tpu` / `none`) |
| `push` | Push the generated notebook with the Kaggle API |
| `install_source` | How TRLoom is installed inside the notebook |
| `output_dir` | Training output path inside the kernel |

`modal.enabled` and `kaggle.enabled` are mutually exclusive.

## User code and formatters

Local callables referenced from YAML (`map_fn`, `formatting_func`, rewards, …)
are **bundled into the notebook** the same way they are for Modal, so remote
Kaggle runs can import them.

```yaml
dataset:
  map_fn: formatters:instruction_to_text

user_code:
  - ./formatters.py

kaggle:
  enabled: true
  mode: notebook
```

## CLI

```bash
# Force Kaggle backend
trloom run examples/sft_kaggle.yaml --kaggle

# Write notebook + kernel-metadata.json
trloom kaggle-notebook examples/sft_kaggle.yaml -o ./kaggle_job/trloom.ipynb
```

## Python API

```python
from trloom.kaggle_support import run_on_kaggle, write_kaggle_notebook

write_kaggle_notebook("examples/sft_kaggle.yaml", destination="out/job.ipynb")
run_on_kaggle("examples/sft_kaggle.yaml")
```

Inside a Kaggle kernel, `mode: auto` (or `inline`) disables nested remote
dispatch and redirects `training.output_dir` to `kaggle.output_dir`.
