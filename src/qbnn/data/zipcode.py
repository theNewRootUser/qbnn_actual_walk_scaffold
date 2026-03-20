from __future__ import annotations
from pathlib import Path
from typing import Dict
import numpy as np
from src.qbnn.config import DataConfig


def _reshape_x(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 2 and x.shape[1] == 256:
        return x.reshape(-1, 1, 16, 16)
    if x.ndim == 4:
        return x.astype(np.float64)
    raise ValueError(f"unexpected image shape: {x.shape}")


def _load_text_file(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    arr = np.loadtxt(path, dtype=np.float64)
    y = arr[:, 0].astype(np.int64)
    x = arr[:, 1:]
    return _reshape_x(x), y


def load_zipcode_dataset(cfg: DataConfig) -> Dict[str, np.ndarray]:
    if cfg.npz_path and Path(cfg.npz_path).exists():
        data = np.load(cfg.npz_path)
        x_train = _reshape_x(data["x_train"])
        y_train = np.asarray(data["y_train"], dtype=np.int64)
        x_test = _reshape_x(data["x_test"])
        y_test = np.asarray(data["y_test"], dtype=np.int64)
        return {"x_train": x_train, "y_train": y_train, "x_test": x_test, "y_test": y_test}
    if cfg.train_path and cfg.test_path:
        x_train, y_train = _load_text_file(cfg.train_path)
        x_test, y_test = _load_text_file(cfg.test_path)
        return {"x_train": x_train, "y_train": y_train, "x_test": x_test, "y_test": y_test}
    default_train = Path("data/zipcode/zip.train")
    default_test = Path("data/zipcode/zip.test")
    if default_train.exists() and default_test.exists():
        x_train, y_train = _load_text_file(default_train)
        x_test, y_test = _load_text_file(default_test)
        return {"x_train": x_train, "y_train": y_train, "x_test": x_test, "y_test": y_test}
    raise FileNotFoundError("Could not locate Zipcode dataset. See data/README.txt")
