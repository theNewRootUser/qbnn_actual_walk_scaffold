from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class LocalMoveTable:
    active_indices: np.ndarray
    current_local: np.ndarray
    proposal_states: np.ndarray
    move_labels: list[tuple[int, int]]
    log_pi_current: float
    log_pi_proposals: np.ndarray
    accept_probs: np.ndarray


def build_local_move_table(
    model,
    theta_ref: np.ndarray,
    active_indices: np.ndarray,
    proposal_states: np.ndarray,
    move_labels: list[tuple[int, int]],
    x_train: np.ndarray,
    y_train: np.ndarray,
) -> LocalMoveTable:
    active_indices = np.asarray(active_indices, dtype=np.int64)
    current_local = np.asarray(theta_ref[active_indices], dtype=np.float64)
    local_states = np.vstack([current_local[None, :], proposal_states])

    log_pi = model.local_block_log_posterior_table(
        theta_ref=theta_ref,
        active_indices=active_indices,
        local_states=local_states,
        x=x_train,
        y=y_train,
    )
    if not np.all(np.isfinite(log_pi)):
        raise ValueError(f"non-finite local move log_pi: {log_pi}")

    log_pi_current = float(log_pi[0])
    log_pi_proposals = np.asarray(log_pi[1:], dtype=np.float64)
    delta = log_pi_proposals - log_pi_current
    accept_probs = np.where(delta >= 0.0, 1.0, np.exp(np.clip(delta, -60.0, 0.0)))

    return LocalMoveTable(
        active_indices=active_indices,
        current_local=current_local,
        proposal_states=np.asarray(proposal_states, dtype=np.float64),
        move_labels=list(move_labels),
        log_pi_current=log_pi_current,
        log_pi_proposals=log_pi_proposals,
        accept_probs=np.asarray(accept_probs, dtype=np.float64),
    )
