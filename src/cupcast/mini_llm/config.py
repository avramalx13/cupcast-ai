from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, TypeVar

import yaml


T = TypeVar("T")


@dataclass
class ModelConfig:
    vocab_size: int = 8000
    context_length: int = 256
    n_layers: int = 6
    d_model: int = 384
    n_heads: int = 6
    d_ff: int = 1536
    dropout: float = 0.1

    def validate(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("model.vocab_size must be positive")
        if self.context_length <= 1:
            raise ValueError("model.context_length must be greater than 1")
        if self.n_layers <= 0:
            raise ValueError("model.n_layers must be positive")
        if self.d_model <= 0:
            raise ValueError("model.d_model must be positive")
        if self.n_heads <= 0:
            raise ValueError("model.n_heads must be positive")
        if self.d_model % self.n_heads != 0:
            raise ValueError("model.d_model must be divisible by model.n_heads")
        if self.d_ff <= 0:
            raise ValueError("model.d_ff must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("model.dropout must be in [0, 1)")


@dataclass
class TrainingConfig:
    batch_size: int = 32
    learning_rate: float = 3e-4
    max_steps: int = 3000
    eval_interval: int = 200
    checkpoint_interval: int = 500
    weight_decay: float = 0.01
    grad_clip: float = 1.0

    def validate(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("training.batch_size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("training.learning_rate must be positive")
        if self.max_steps <= 0:
            raise ValueError("training.max_steps must be positive")
        if self.eval_interval <= 0:
            raise ValueError("training.eval_interval must be positive")
        if self.checkpoint_interval <= 0:
            raise ValueError("training.checkpoint_interval must be positive")
        if self.weight_decay < 0:
            raise ValueError("training.weight_decay cannot be negative")
        if self.grad_clip <= 0:
            raise ValueError("training.grad_clip must be positive")


@dataclass
class TokenizerConfig:
    type: str = "byte_bpe"
    min_pair_freq: int = 2
    max_merges: int | None = None
    max_training_bytes: int | None = None

    def validate(self) -> None:
        if self.type not in {"byte_bpe", "word"}:
            raise ValueError("tokenizer.type must be 'byte_bpe' or 'word'")
        if self.min_pair_freq < 1:
            raise ValueError("tokenizer.min_pair_freq must be at least 1")
        if self.max_merges is not None and self.max_merges < 0:
            raise ValueError("tokenizer.max_merges cannot be negative")
        if self.max_training_bytes is not None and self.max_training_bytes <= 0:
            raise ValueError("tokenizer.max_training_bytes must be positive when set")


@dataclass
class DataConfig:
    validation_split: float = 0.1
    seed: int = 42

    def validate(self) -> None:
        if not 0.0 < self.validation_split < 0.5:
            raise ValueError("data.validation_split must be between 0 and 0.5")


@dataclass
class ExperimentConfig:
    model: ModelConfig
    training: TrainingConfig
    tokenizer: TokenizerConfig
    data: DataConfig

    def validate(self) -> None:
        self.model.validate()
        self.training.validate()
        self.tokenizer.validate()
        self.data.validate()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ExperimentConfig":
        config = cls(
            model=_dataclass_from_dict(ModelConfig, raw.get("model", {})),
            training=_dataclass_from_dict(TrainingConfig, raw.get("training", {})),
            tokenizer=_dataclass_from_dict(TokenizerConfig, raw.get("tokenizer", {})),
            data=_dataclass_from_dict(DataConfig, raw.get("data", {})),
        )
        config.validate()
        return config


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {config_path}")
    return ExperimentConfig.from_dict(raw)


def _dataclass_from_dict(cls: type[T], raw: dict[str, Any]) -> T:
    if not isinstance(raw, dict):
        raise ValueError(f"{cls.__name__} config section must be a mapping")
    valid_keys = {field.name for field in fields(cls)}
    unknown_keys = set(raw) - valid_keys
    if unknown_keys:
        keys = ", ".join(sorted(unknown_keys))
        raise ValueError(f"Unknown keys in {cls.__name__}: {keys}")
    return cls(**raw)
