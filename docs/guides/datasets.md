# Datasets

TRLoom loads data through Hugging Face `datasets`. Configure everything under
the `dataset` section of your YAML.

## Hub dataset

```yaml
dataset:
  path: trl-lib/Capybara
  train_split: train
  eval_split: null
```

Optional Hub helpers:

```yaml
dataset:
  path: some/dataset
  name: subset_name
  split: "train[:1000]"
  streaming: false
  kwargs:
    revision: main
```

`split` is applied at load time. `train_split` / `eval_split` select keys from
the resulting dataset dict (or the single split name when you loaded one split).

!!! tip
    The default `eval_split` is `"test"`. Set `eval_split: null` when the
    dataset has no test split.

## Local files

Supported extensions: `.json`, `.jsonl`, `.csv`, `.tsv`, `.parquet`, `.txt`,
`.arrow`.

```yaml
dataset:
  path: ./data/train.jsonl
  train_split: train
  eval_split: null
```

Local directories prefer `load_from_disk`, then fall back to `load_dataset`.

Paths are treated as local when they exist on disk, or look like a path
(`.` / `/` / `~` / `\`, or a Windows drive like `C:\`).

## Column rename and text column

```yaml
dataset:
  path: ./data/train.jsonl
  train_split: train
  text_column: text
  columns:
    prompt: query
    completion: response
```

## Prompt templates and formatting functions

Raw datasets often need reshaping before TRL trainers see them. Configure this
in YAML — the same callables work locally and on Modal.

### Inline Jinja2 prompt template

```yaml
dataset:
  path: ./data/raw.jsonl
  train_split: train
  eval_split: null
  prompt_template: |
    ### Instruction:
    {{ instruction }}

    ### Response:
    {{ response }}
  prompt_output_column: text
  prompt_remove_columns: true   # drop original columns after render
  text_column: text
```

Template variables are dataset column names. Requires Jinja2 (pulled in by
`transformers`).

### Custom `map_fn` (import path)

```yaml
dataset:
  path: ./data/raw.jsonl
  train_split: train
  eval_split: null
  map_fn: formatters:instruction_to_text
  map_kwargs: {}          # forwarded to Dataset.map
  text_column: text

user_code:
  - ./formatters.py       # shipped to Modal automatically
```

`map_fn` runs before `prompt_template` when both are set.

### Trainer `formatting_func` (e.g. SFT)

```yaml
dataset:
  path: ./data/chat.jsonl
  formatting_func: formatters:messages_to_text

# Or at the root (overrides dataset.formatting_func):
formatting_func: formatters:messages_to_text
```

Resolved import paths are passed to trainers that accept `formatting_func`
(such as `SFTTrainer`).

### Remote providers (Modal)

YAML-referenced modules under `user_code` / callable import paths are
**bundled with the job** and installed on the remote worker before training.
You do not need a separate image build step for small formatter modules.

See [Modal](modal.md#user-code-and-formatters) and `examples/sft_formatted.yaml`.

## Mixtures

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

- Equal weights → concatenate
- Unequal weights or streaming → interleave

## Tiny smoke subset

Useful for proving a pipeline before a full run:

```yaml
dataset:
  path: stanfordnlp/imdb
  split: "train[:64]"
  train_split: train
  eval_split: null
  text_column: text
```

## Pair with training fields

For plain text SFT, set the TRL text field explicitly:

```yaml
training:
  dataset_text_field: text
  max_length: 128
  remove_unused_columns: false
```

For chat / preference datasets, follow the column layout expected by that TRL
trainer (for example `messages` for chat SFT, or chosen / rejected columns for
DPO).
