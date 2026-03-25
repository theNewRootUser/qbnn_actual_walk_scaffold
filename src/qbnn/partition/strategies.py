
from __future__ import annotations
from typing import Sequence
import numpy as np


def contiguous_weight_blocks(num_params: int, block_param_count: int) -> list[np.ndarray]:
    return [
        np.arange(i, min(num_params, i + block_param_count), dtype=np.int64)
        for i in range(0, num_params, block_param_count)
    ]


def region_aligned_weight_blocks(model, block_param_count: int) -> list[np.ndarray]:
    needed = ("s_fc1_w", "s_fc1_b", "s_fc2_w", "s_fc2_b")
    missing = [name for name in needed if not hasattr(model, name)]
    if missing:
        raise ValueError(
            f"region_aligned_weight_blocks requires model slices {needed}, missing={missing}"
        )

    regions = [model.s_fc1_w, model.s_fc1_b, model.s_fc2_w, model.s_fc2_b]
    out: list[np.ndarray] = []
    for s in regions:
        start, stop = int(s.start), int(s.stop)
        for i in range(start, stop, block_param_count):
            out.append(np.arange(i, min(stop, i + block_param_count), dtype=np.int64))
    return out


def hidden_pathway_blocks(model) -> list[np.ndarray]:
    needed = ("s_fc1_w", "s_fc1_b", "s_fc2_w", "s_fc2_b")
    missing = [name for name in needed if not hasattr(model, name)]
    if missing:
        raise ValueError(
            f"hidden_pathway_blocks requires model slices {needed}, missing={missing}"
        )

    hidden = np.concatenate(
        [
            np.arange(model.s_fc1_w.start, model.s_fc1_w.stop, dtype=np.int64),
            np.arange(model.s_fc1_b.start, model.s_fc1_b.stop, dtype=np.int64),
            np.arange(model.s_fc2_w.start, model.s_fc2_w.stop, dtype=np.int64),
        ]
    )
    out_bias = np.arange(model.s_fc2_b.start, model.s_fc2_b.stop, dtype=np.int64)
    return [hidden, out_bias]


def build_partition_blocks(
    num_params: int,
    strategy: str,
    *,
    block_param_count: int = 1,
    explicit: Sequence[Sequence[int]] | None = None,
    model=None,
) -> list[np.ndarray]:
    if strategy == "explicit_indices":
        if explicit is None:
            raise ValueError("explicit blocks required")
        return [np.asarray(block, dtype=np.int64) for block in explicit]

    if strategy == "contiguous_weight_blocks":
        return contiguous_weight_blocks(num_params, block_param_count)

    if strategy == "region_aligned_weight_blocks":
        if model is None:
            raise ValueError("model is required for region_aligned_weight_blocks")
        return region_aligned_weight_blocks(model, block_param_count)

    if strategy == "hidden_pathway_blocks":
        if model is None:
            raise ValueError("model is required for hidden_pathway_blocks")
        return hidden_pathway_blocks(model)

    raise ValueError(f"Unsupported partition strategy: {strategy}")
