from __future__ import annotations
import numpy as np
from scipy.special import logsumexp


def build_complete_graph_proposal(num_states: int, allow_self: bool = False) -> np.ndarray:
    q = np.ones((num_states, num_states), dtype=np.float64)
    if not allow_self:
        np.fill_diagonal(q, 0.0)
    q /= q.sum(axis=1, keepdims=True)
    return q


def build_hamming_graph_proposal(states: np.ndarray, max_hamming_distance: int = 1, allow_self: bool = False) -> np.ndarray:
    n = states.shape[0]
    q = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            if i == j and not allow_self:
                continue
            dist = int(np.sum(states[i] != states[j]))
            if dist <= max_hamming_distance and (allow_self or i != j):
                q[i, j] = 1.0
        if q[i].sum() == 0:
            q[i, i] = 1.0
        q[i] /= q[i].sum()
    return q


def build_mh_transition_matrix(log_pi: np.ndarray, proposal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    log_pi = np.asarray(log_pi, dtype=np.float64)
    log_pi = log_pi - logsumexp(log_pi)
    pi = np.exp(log_pi)
    n = log_pi.size
    p = np.zeros((n, n), dtype=np.float64)
    alpha = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        off_diag = 0.0
        for j in range(n):
            if i == j or proposal[i, j] <= 0:
                continue
            ratio = np.exp(log_pi[j] - log_pi[i]) * (proposal[j, i] / proposal[i, j])
            a = min(1.0, float(ratio))
            alpha[i, j] = a
            p[i, j] = proposal[i, j] * a
            off_diag += p[i, j]
        p[i, i] = max(0.0, 1.0 - off_diag)
    return p, pi


def stationary_distribution(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)

    vals, vecs = np.linalg.eig(p.T)
    idx = int(np.argmin(np.abs(vals - 1.0)))
    raw = np.real(vecs[:, idx])

    # Eigenvectors are only defined up to a global sign.
    if raw.sum() < 0:
        raw = -raw

    vec = np.maximum(raw, 0.0)
    s = vec.sum()

    if s <= 0 or not np.isfinite(s):
        vec = np.abs(raw)
        s = vec.sum()

    if s <= 0 or not np.isfinite(s):
        raise ValueError(f"Could not recover valid stationary eigenvector from p={p}")

    vec /= s
    return vec


def detailed_balance_error(p: np.ndarray, pi: np.ndarray) -> float:
    lhs = pi[:, None] * p
    rhs = pi[None, :] * p.T
    return float(np.max(np.abs(lhs - rhs)))
