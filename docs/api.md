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

::: trloom.config.loader.load_config

## Trainers

::: trloom.trainers.registry.list_trainers
