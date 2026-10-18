"""Pydantic models for TRLoom YAML configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ModelConfig(BaseModel):
    """Model / PEFT / quantization settings (mirrors TRL ``ModelConfig`` fields)."""

    model_config = ConfigDict(extra="allow")

    model_name_or_path: str
    model_revision: str = "main"
    dtype: Literal["auto", "bfloat16", "float16", "float32"] | None = "float32"
    attn_implementation: str | None = None
    trust_remote_code: bool = False

    use_peft: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] | str | None = None
    lora_target_parameters: list[str] | None = None
    lora_modules_to_save: list[str] | None = None
    lora_task_type: str = "CAUSAL_LM"
    use_rslora: bool = False
    use_dora: bool = False

    load_in_8bit: bool = False
    load_in_4bit: bool = False
    bnb_4bit_quant_type: Literal["fp4", "nf4"] = "nf4"
    use_bnb_nested_quant: bool = False
    bnb_4bit_quant_storage: str | None = None

    @model_validator(mode="after")
    def _validate_quantization(self) -> ModelConfig:
        if self.load_in_8bit and self.load_in_4bit:
            raise ValueError("Cannot enable both load_in_8bit and load_in_4bit.")
        return self


class DatasetSourceConfig(BaseModel):
    """A single dataset entry (Hub repo id or local path)."""

    model_config = ConfigDict(extra="allow")

    path: str
    name: str | None = None
    split: str | None = None
    data_files: str | list[str] | dict[str, Any] | None = None
    data_dir: str | None = None
    streaming: bool = False
    columns: dict[str, str] | None = None
    weight: float | None = None


class DatasetConfig(BaseModel):
    """Dataset loading configuration.

    Prefer ``path`` for a single Hub/local dataset, or ``datasets`` for a mixture.

    Formatting (applied after load / rename, before training):
    - ``map_fn``: import path applied via ``dataset.map``
    - ``prompt_template``: Jinja2 template rendered per row into ``prompt_output_column``
    - ``formatting_func``: import path forwarded to trainers that accept it (e.g. SFT)
    """

    model_config = ConfigDict(extra="allow")

    path: str | None = None
    name: str | None = None
    # Optional datasets split selector, e.g. "train" or "train[:64]"
    split: str | None = None
    data_files: str | list[str] | dict[str, Any] | None = None
    data_dir: str | None = None
    streaming: bool = False

    datasets: list[DatasetSourceConfig] | None = None

    train_split: str = "train"
    eval_split: str | None = "test"
    text_column: str | None = None
    columns: dict[str, str] | None = None

    # Optional Hugging Face datasets kwargs passthrough
    kwargs: dict[str, Any] = Field(default_factory=dict)

    # Pre-training dataset transforms (YAML-referenced callables / templates)
    map_fn: str | None = Field(
        default=None,
        description="Import path for a datasets.map callable, e.g. 'my_pkg.format:fn'.",
    )
    map_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description="Extra kwargs forwarded to Dataset.map when map_fn is set.",
    )
    prompt_template: str | None = Field(
        default=None,
        description="Jinja2 template rendered per row (keys = column names).",
    )
    prompt_output_column: str = Field(
        default="text",
        description="Column written by prompt_template.",
    )
    prompt_remove_columns: list[str] | bool | None = Field(
        default=None,
        description="Columns to drop after prompt_template. True drops all prior columns.",
    )
    formatting_func: str | None = Field(
        default=None,
        description="Import path for a trainer formatting_func (e.g. SFTTrainer).",
    )

    @field_validator("map_fn", "formatting_func", "prompt_template", mode="before")
    @classmethod
    def _empty_str_to_none(cls, value: Any) -> Any:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("prompt_output_column")
    @classmethod
    def _nonempty_output_column(cls, value: str) -> str:
        if not value or not str(value).strip():
            raise ValueError("prompt_output_column must be a non-empty string.")
        return value

    @field_validator("prompt_remove_columns", mode="before")
    @classmethod
    def _normalize_remove_columns(cls, value: Any) -> Any:
        if value is None or isinstance(value, bool):
            return value
        if isinstance(value, str):
            return [value]
        if isinstance(value, list):
            cleaned = [str(item) for item in value]
            if any(not item.strip() for item in cleaned):
                raise ValueError("prompt_remove_columns entries must be non-empty strings.")
            return cleaned
        raise TypeError(
            "prompt_remove_columns must be bool, string, list of strings, or null."
        )

    @model_validator(mode="after")
    def _require_source(self) -> DatasetConfig:
        if not self.path and not self.datasets:
            raise ValueError("Dataset config requires either 'path' or 'datasets'.")
        if self.path and self.datasets:
            raise ValueError(
                "Dataset config accepts either 'path' or 'datasets', not both."
            )
        return self


class WandbConfig(BaseModel):
    """Weights & Biases logging settings."""

    model_config = ConfigDict(extra="allow")

    enabled: bool = False
    project: str | None = None
    entity: str | None = None
    run_name: str | None = None
    group: str | None = None
    tags: list[str] = Field(default_factory=list)
    notes: str | None = None
    mode: Literal["online", "offline", "disabled"] | None = None
    dir: str | None = None
    job_type: str | None = "train"
    # Extra keys forwarded to wandb.init
    init_kwargs: dict[str, Any] = Field(default_factory=dict)


class ModalConfig(BaseModel):
    """Modal Labs remote execution settings."""

    model_config = ConfigDict(extra="allow")

    enabled: bool = False
    app_name: str = "trloom"
    gpu: str = "T4"
    timeout: int = 60 * 60 * 4
    cpu: float | None = None
    memory: int | None = None
    volume_name: str = "trloom-outputs"
    volume_mount: str = "/outputs"
    # Names of Modal Secrets to attach (create with `modal secret create ...`).
    # Leave empty for public Hub models/datasets that need no tokens.
    secrets: list[str] = Field(default_factory=list)
    pip_packages: list[str] = Field(default_factory=list)
    python_version: str = "3.11"
    region: str | None = None
    # How to get trloom into the Modal image:
    # - local: mount the local installed package (best for development)
    # - git: pip install from git_url (best for reproducibility)
    # - pypi: pip install trloom (once published)
    install_source: Literal["local", "git", "pypi"] = "local"
    git_url: str = "git+https://github.com/saqlain2204/trloom.git"
    # If set, copy training output from the Modal volume back to this local path
    download_dir: str | None = None


class KaggleConfig(BaseModel):
    """Kaggle Notebooks / GPU execution settings.

    Modes:
    - ``auto``: if already inside a Kaggle kernel, run inline; otherwise write a notebook
    - ``inline``: run in the current process (intended for Kaggle kernels)
    - ``notebook``: generate a Kaggle notebook (and optionally push via the API)
    """

    model_config = ConfigDict(extra="allow")

    enabled: bool = False
    mode: Literal["auto", "inline", "notebook"] = "auto"
    accelerator: Literal["gpu", "tpu", "none"] = "gpu"
    push: bool = False
    username: str | None = None
    kernel_slug: str | None = None
    title: str = "TRLoom Fine-Tune"
    is_private: bool = True
    enable_internet: bool = True
    dataset_sources: list[str] = Field(default_factory=list)
    competition_sources: list[str] = Field(default_factory=list)
    kernel_sources: list[str] = Field(default_factory=list)
    pip_packages: list[str] = Field(default_factory=list)
    install_source: Literal["pypi", "git"] = "pypi"
    git_url: str = "git+https://github.com/saqlain2204/trloom.git"
    output_dir: str = "/kaggle/working/outputs"
    notebook_path: str | None = None
    download_dir: str | None = None
    wait: bool = True
    timeout_seconds: int = 60 * 60 * 9

    @field_validator(
        "dataset_sources",
        "competition_sources",
        "kernel_sources",
        "pip_packages",
        mode="before",
    )
    @classmethod
    def _coerce_str_list(cls, value: Any) -> Any:
        if isinstance(value, str):
            return [value]
        return value


class FineTuneConfig(BaseModel):
    """Root configuration for a TRLoom fine-tuning job."""

    model_config = ConfigDict(extra="allow")

    method: str = Field(
        ...,
        description="TRL training method, e.g. sft, dpo, grpo, kto, reward, rloo.",
    )
    model: ModelConfig
    dataset: DatasetConfig
    training: dict[str, Any] = Field(
        default_factory=dict,
        description="Keyword arguments forwarded to the TRL *Config class "
        "(SFTConfig, DPOConfig, …).",
    )
    wandb: WandbConfig = Field(default_factory=WandbConfig)
    modal: ModalConfig = Field(default_factory=ModalConfig)
    kaggle: KaggleConfig = Field(default_factory=KaggleConfig)

    # Optional extras
    reward_funcs: list[str] | str | None = None
    formatting_func: str | None = Field(
        default=None,
        description=(
            "Import path for a trainer formatting_func. "
            "Overrides dataset.formatting_func when both are set."
        ),
    )
    user_code: list[str] = Field(
        default_factory=list,
        description=(
            "Local .py files or package directories to ship with remote jobs "
            "(Modal, Kaggle, etc.) so YAML-referenced callables remain importable."
        ),
    )
    trainer_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description="Extra kwargs passed directly to the TRL Trainer constructor.",
    )
    push_to_hub: bool = False
    hub_model_id: str | None = None
    seed: int | None = 42

    @field_validator("method")
    @classmethod
    def _normalize_method(cls, value: str) -> str:
        return value.strip().lower().replace("-", "_").replace(" ", "_")

    @field_validator("reward_funcs", mode="before")
    @classmethod
    def _coerce_reward_funcs(cls, value: Any) -> Any:
        if isinstance(value, str):
            return [value]
        return value

    @field_validator("user_code", mode="before")
    @classmethod
    def _coerce_user_code(cls, value: Any) -> Any:
        if isinstance(value, str):
            return [value]
        return value

    @model_validator(mode="after")
    def _exclusive_remote_backends(self) -> FineTuneConfig:
        if self.modal.enabled and self.kaggle.enabled:
            raise ValueError(
                "Enable only one remote backend: set either modal.enabled or "
                "kaggle.enabled, not both."
            )
        return self

    def resolved_formatting_func(self) -> str | None:
        """Trainer formatting_func import path (root overrides dataset)."""
        return self.formatting_func or self.dataset.formatting_func

    def resolved_output_dir(self) -> Path:
        output_dir = self.training.get("output_dir", "./outputs")
        return Path(output_dir).expanduser().resolve()

    def resolved_backend(self) -> Literal["local", "modal", "kaggle"]:
        """Return the execution backend implied by the config flags."""
        if self.modal.enabled:
            return "modal"
        if self.kaggle.enabled:
            return "kaggle"
        return "local"
