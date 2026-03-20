from __future__ import annotations
import numpy as np
from src.qbnn.models import BayesianLeNet2


def build_local_log_posterior(model: BayesianLeNet2, theta_ref: np.ndarray, active_indices: np.ndarray, local_states: np.ndarray, x_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    return model.local_block_log_posterior_table(theta_ref=theta_ref, active_indices=active_indices, local_states=local_states, x=x_train, y=y_train)
