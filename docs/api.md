# API reference

Public Python surface for TRLoom.

## Quick usage

```python
from trloom import FineTuneJob, load_config, run_from_yaml, available_methods

print(available_methods())

run_from_yaml("sft.yaml")

job = FineTuneJob.from_yaml("sft.yaml")
job.build()
job.run()
```

## Job orchestration

::: trloom.job.FineTuneJob

::: trloom.job.run_from_yaml

::: trloom.job.available_methods

## Configuration

::: trloom.config.schema.FineTuneConfig

::: trloom.config.schema.ModelConfig

::: trloom.config.schema.DatasetConfig

::: trloom.config.schema.DatasetSourceConfig

::: trloom.config.schema.WandbConfig

::: trloom.config.schema.ModalConfig

::: trloom.config.schema.KaggleConfig

::: trloom.config.loader.load_config

## Dataset formatting

::: trloom.data.formatting.format_dataset

::: trloom.imports.resolve_callable

::: trloom.user_code.build_user_code_bundle

## Kaggle

::: trloom.kaggle_support.run_on_kaggle

::: trloom.kaggle_support.write_kaggle_notebook

::: trloom.kaggle_support.is_kaggle_environment

## Trainers

::: trloom.trainers.registry.list_trainers
