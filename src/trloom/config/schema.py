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
    """

    model_config = ConfigDict(extra="allow")

    path: str | None = None
    name: str | None = None
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

    @model_validator(mode="after")
    def _require_source(self) -> DatasetConfig:
        if not self.path and not self.datasets:
            raise ValueError("Dataset config requires either 'path' or 'datasets'.")
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
    gpu: str = "A100"
    timeout: int = 60 * 60 * 4
    cpu: float | None = None
    memory: int | None = None
    volume_name: str = "trloom-outputs"
    volume_mount: str = "/outputs"
    secrets: list[str] = Field(default_factory=lambda: ["huggingface", "wandb"])
    pip_packages: list[str] = Field(default_factory=list)
    python_version: str = "3.11"
    region: str | None = None
    # If set, copy training output from the Modal volume back to this local path
    download_dir: str | None = None


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

    # Optional extras
    reward_funcs: list[str] | str | None = None
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

    def resolved_output_dir(self) -> Path:
        output_dir = self.training.get("output_dir", "./outputs")
        return Path(output_dir).expanduser().resolve()
