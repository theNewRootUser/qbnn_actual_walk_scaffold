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


def _infer_hidden_and_output_sizes(model) -> tuple[int, int, int]:
    needed = ("s_fc1_w", "s_fc1_b", "s_fc2_w", "s_fc2_b")
    missing = [name for name in needed if not hasattr(model, name)]
    if missing:
        raise ValueError(
            f"hidden_pathway_blocks requires model slices {needed}, missing={missing}"
        )

    # infer hidden count from hidden-bias slice, not from model.fc_hidden
    n_hidden = int(model.s_fc1_b.stop - model.s_fc1_b.start)
    n_out = int(model.s_fc2_b.stop - model.s_fc2_b.start)
    fc1_w_size = int(model.s_fc1_w.stop - model.s_fc1_w.start)
    fc2_w_size = int(model.s_fc2_w.stop - model.s_fc2_w.start)

    if n_hidden <= 0:
        raise ValueError(f"inferred n_hidden must be positive, got {n_hidden}")
    if fc1_w_size % n_hidden != 0:
        raise ValueError(
            f"fc1 weight slice size {fc1_w_size} is not divisible by inferred n_hidden={n_hidden}"
        )
    if fc2_w_size % n_hidden != 0:
        raise ValueError(
            f"fc2 weight slice size {fc2_w_size} is not divisible by inferred n_hidden={n_hidden}"
        )

    n_in = fc1_w_size // n_hidden
    inferred_n_out_from_fc2 = fc2_w_size // n_hidden
    if inferred_n_out_from_fc2 != n_out:
        raise ValueError(
            f"inconsistent output size: s_fc2_b implies {n_out}, "
            f"but s_fc2_w implies {inferred_n_out_from_fc2}"
        )

    return n_hidden, n_out, n_in


def _single_hidden_pathway_blocks(model) -> list[np.ndarray]:
    n_hidden, n_out, n_in = _infer_hidden_and_output_sizes(model)

    fc1_w_start = int(model.s_fc1_w.start)
    fc1_b_start = int(model.s_fc1_b.start)
    fc2_w_start = int(model.s_fc2_w.start)

    pathways: list[np.ndarray] = []

    for h in range(n_hidden):
        # assumes hidden index is the fastest-changing axis in the flattened weights
        incoming = fc1_w_start + h + n_hidden * np.arange(n_in, dtype=np.int64)
        hidden_bias = np.asarray([fc1_b_start + h], dtype=np.int64)
        outgoing = fc2_w_start + h + n_hidden * np.arange(n_out, dtype=np.int64)

        pw = np.concatenate([incoming, hidden_bias, outgoing]).astype(np.int64)
        pathways.append(pw)

    return pathways


def hidden_pathway_blocks(model, block_param_count: int = 0) -> list[np.ndarray]:
    pathways = _single_hidden_pathway_blocks(model)
    if not pathways:
        out_bias = np.arange(model.s_fc2_b.start, model.s_fc2_b.stop, dtype=np.int64)
        return [out_bias]

    pathway_size = int(pathways[0].size)
    budget = int(block_param_count)

    # if budget is smaller than one pathway, still keep one whole pathway per block
    if budget <= 0 or budget < pathway_size:
        budget = pathway_size

    out: list[np.ndarray] = []
    current_parts: list[np.ndarray] = []
    current_size = 0

    for pw in pathways:
        pw_size = int(pw.size)
        if current_parts and (current_size + pw_size > budget):
            out.append(np.concatenate(current_parts).astype(np.int64))
            current_parts = [pw]
            current_size = pw_size
        else:
            current_parts.append(pw)
            current_size += pw_size

    if current_parts:
        out.append(np.concatenate(current_parts).astype(np.int64))

    out_bias = np.arange(model.s_fc2_b.start, model.s_fc2_b.stop, dtype=np.int64)
    out.append(out_bias)
    return out

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
        return hidden_pathway_blocks(model, block_param_count)

    raise ValueError(f"Unsupported partition strategy: {strategy}")
