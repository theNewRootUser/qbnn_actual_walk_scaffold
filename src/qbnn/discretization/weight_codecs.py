from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class FixedPointCodec:
    bits: int = 1
    step: float = 0.05

    @property
    def levels(self) -> np.ndarray:
        n = 2 ** self.bits
        half = (n - 1) / 2.0
        return (np.arange(n, dtype=np.float64) - half) * self.step

    def local_values_around(self, center: np.ndarray) -> np.ndarray:
        center = np.asarray(center, dtype=np.float64)
        return center[:, None] + self.levels[None, :]
