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
