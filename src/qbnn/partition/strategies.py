from __future__ import annotations
from typing import Sequence
import numpy as np


def contiguous_weight_blocks(num_params: int, block_param_count: int) -> list[np.ndarray]:
    return [np.arange(i, min(num_params, i + block_param_count), dtype=np.int64) for i in range(0, num_params, block_param_count)]


def build_partition_blocks(num_params: int, strategy: str, *, block_param_count: int = 1, explicit: Sequence[Sequence[int]] | None = None) -> list[np.ndarray]:
    if strategy == "explicit_indices":
        if explicit is None:
            raise ValueError("explicit blocks required")
        return [np.asarray(block, dtype=np.int64) for block in explicit]
    if strategy != "contiguous_weight_blocks":
        raise ValueError(f"Unsupported partition strategy: {strategy}")
    return contiguous_weight_blocks(num_params, block_param_count)
