from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class SignedDirectionMoves:
    active_indices: np.ndarray
    current_local: np.ndarray
    groups: list[np.ndarray]
    proposal_states: np.ndarray
    move_labels: list[tuple[int, int]]  # (direction_id, sign_bit), sign_bit: 0 -> -, 1 -> +
    direction_step: float

    @property
    def num_directions(self) -> int:
        return len(self.groups)

    @property
    def num_moves(self) -> int:
        return len(self.move_labels)


def build_signed_direction_moves(
    active_indices: np.ndarray,
    theta_ref: np.ndarray,
    *,
    direction_bank_size: int,
    direction_step: float,
) -> SignedDirectionMoves:
    active_indices = np.asarray(active_indices, dtype=np.int64)
    current_local = np.asarray(theta_ref[active_indices], dtype=np.float64)

    if active_indices.size == 0:
        raise ValueError("empty active block")

    actual_dirs = max(1, min(int(direction_bank_size), int(active_indices.size)))
    groups = [
        np.asarray(g, dtype=np.int64)
        for g in np.array_split(np.arange(active_indices.size, dtype=np.int64), actual_dirs)
        if len(g) > 0
    ]

    proposal_states = []
    move_labels = []
    for d, group in enumerate(groups):
        for sign_bit, sign in ((0, -1.0), (1, +1.0)):
            cand = np.array(current_local, copy=True)
            cand[group] += sign * float(direction_step)
            proposal_states.append(cand)
            move_labels.append((d, sign_bit))

    return SignedDirectionMoves(
        active_indices=active_indices,
        current_local=current_local,
        groups=groups,
        proposal_states=np.asarray(proposal_states, dtype=np.float64),
        move_labels=move_labels,
        direction_step=float(direction_step),
    )
