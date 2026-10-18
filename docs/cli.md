# CLI

Install the package, then use the `trloom` entry point (or `python -m trloom`).

```bash
trloom --version
trloom -v|--verbose <command>
```

## `trloom methods`

List TRL training methods discovered in this environment, including experimental
ones when available.

```bash
trloom methods
```

## `trloom validate`

Validate a YAML config without training. Prints a JSON summary:

```bash
trloom validate path/to/config.yaml
```

Typical fields: `ok`, `method`, `trainer`, `config_class`, `experimental`,
`output_dir`, `wandb_enabled`, `modal_enabled`, `kaggle_enabled`, `backend`.

## `trloom run`

Run a fine-tuning job from YAML.

```bash
trloom run path/to/config.yaml
trloom run path/to/config.yaml --modal
trloom run path/to/config.yaml --kaggle
trloom run path/to/config.yaml --local
```

| Flag | Behavior |
|------|----------|
| `--modal` | Force Modal execution (overrides YAML) |
| `--kaggle` | Force Kaggle execution (overrides YAML) |
| `--local` | Force local execution (overrides YAML) |

`--modal`, `--kaggle`, and `--local` are mutually exclusive.

## `trloom modal-script`

Write a standalone Modal entrypoint for a config:

```bash
trloom modal-script path/to/config.yaml -o run_modal.py
python -m modal run run_modal.py
```

If `-o` / `--output` is omitted, a default script path is used.

## `trloom kaggle-notebook`

Write a Kaggle notebook and `kernel-metadata.json` for a config:

```bash
trloom kaggle-notebook path/to/config.yaml -o ./kaggle_job/job.ipynb
```

Upload the folder to Kaggle, or set `kaggle.push: true` with API credentials.
