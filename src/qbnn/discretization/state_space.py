from __future__ import annotations
from dataclasses import dataclass
import itertools
import numpy as np
from .weight_codecs import FixedPointCodec

@dataclass
class LocalStateSpace:
    active_indices: np.ndarray
    centers: np.ndarray
    values_per_param: list[np.ndarray]
    states: np.ndarray
    state_ids: np.ndarray

    @property
    def num_states(self) -> int:
        return int(self.states.shape[0])

    @property
    def bits_per_state(self) -> int:
        return int(np.ceil(np.log2(max(1, self.num_states))))


def build_local_state_space(active_indices: np.ndarray, theta_ref: np.ndarray, codec: FixedPointCodec) -> LocalStateSpace:
    active_indices = np.asarray(active_indices, dtype=np.int64)
    centers = np.asarray(theta_ref[active_indices], dtype=np.float64)
    vals = [codec.local_values_around(np.array([c])).reshape(-1) for c in centers]
    products = list(itertools.product(*[range(len(v)) for v in vals]))
    states = np.array([[vals[i][choice[i]] for i in range(len(vals))] for choice in products], dtype=np.float64)
    state_ids = np.arange(states.shape[0], dtype=np.int64)
    return LocalStateSpace(active_indices=active_indices, centers=centers, values_per_param=vals, states=states, state_ids=state_ids)
