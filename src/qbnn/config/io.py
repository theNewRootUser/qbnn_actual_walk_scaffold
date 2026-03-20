from __future__ import annotations
from pathlib import Path
from typing import Any
import yaml
from .schema import (
    DataConfig,
    ModelConfig,
    TrainingConfig,
    SamplingConfig,
    QuantumConfig,
    PartitionConfig,
    ExperimentConfig,
)


def load_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}
    return ExperimentConfig(
        name=raw.get("name", ExperimentConfig().name),
        data=DataConfig(**raw.get("data", {})),
        model=ModelConfig(**raw.get("model", {})),
        training=TrainingConfig(**raw.get("training", {})),
        sampling=SamplingConfig(**raw.get("sampling", {})),
        quantum=QuantumConfig(**raw.get("quantum", {})),
        partition=PartitionConfig(**raw.get("partition", {})),
    )
