from __future__ import annotations
from dataclasses import asdict
from src.qbnn.config import ExperimentConfig
from src.qbnn.data import load_zipcode_dataset
from src.qbnn.models import train_deterministic_lenet2


def run_classical_baseline(cfg: ExperimentConfig) -> dict:
    data = load_zipcode_dataset(cfg.data)
    result = train_deterministic_lenet2(cfg.model, cfg.training, data["x_train"], data["y_train"], data["x_test"], data["y_test"])
    return {"config": asdict(cfg), "result": result}
