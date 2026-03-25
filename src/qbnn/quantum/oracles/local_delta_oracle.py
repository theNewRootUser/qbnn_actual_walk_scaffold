
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.special import logsumexp


@dataclass
class LocalMoveTable:
    active_indices: np.ndarray
    current_local: np.ndarray
    proposal_states: np.ndarray
    move_labels: list[tuple[int, int]]
    log_pi_current: float
    log_pi_proposals: np.ndarray
    accept_probs: np.ndarray
    num_directions: int


def _infer_num_directions(move_labels: list[tuple[int, int]]) -> int:
    if not move_labels:
        return 0
    return len({int(label[0]) for label in move_labels})


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
        num_directions=int(_infer_num_directions(move_labels)),
    )


def build_hidden_pathway_move_table(
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
    proposal_states = np.asarray(proposal_states, dtype=np.float64)

    required = ("build_cache", "log_prior", "_prior_delta", "s_fc1_w", "s_fc1_b", "s_fc2_w", "h", "in_dim", "num_classes")
    missing = [name for name in required if not hasattr(model, name)]
    if missing:
        raise ValueError(f"build_hidden_pathway_move_table requires model attrs {required}, missing={missing}")

    cache = model.build_cache(theta_ref, x_train)
    p = cache.params
    x_flat = cache.x_flat
    u1 = cache.u1
    h1 = cache.h1
    logits0 = cache.logits

    log_probs0 = logits0 - logsumexp(logits0, axis=1, keepdims=True)
    base_ll = float(np.sum(log_probs0[np.arange(y_train.shape[0]), y_train.astype(np.int64)]))
    base_lp = float(model.log_prior(theta_ref))
    log_pi_current = base_lp + base_ll

    global_to_local = {int(g): i for i, g in enumerate(active_indices.tolist())}
    log_pi_proposals = np.empty(proposal_states.shape[0], dtype=np.float64)

    for i, (unit_id, _sign_bit) in enumerate(move_labels):
        r = int(unit_id)
        local = proposal_states[i]

        w_in_new = np.array(p["fc1_w"][:, r], copy=True)
        for j in range(int(model.in_dim)):
            g = int(model.s_fc1_w.start + j * int(model.h) + r)
            pos = global_to_local.get(g)
            if pos is not None:
                w_in_new[j] = local[pos]

        b_new = float(p["fc1_b"][r])
        g_bias = int(model.s_fc1_b.start + r)
        pos_bias = global_to_local.get(g_bias)
        if pos_bias is not None:
            b_new = float(local[pos_bias])

        w_out_new = np.array(p["fc2_w"][r, :], copy=True)
        for c in range(int(model.num_classes)):
            g = int(model.s_fc2_w.start + r * int(model.num_classes) + c)
            pos = global_to_local.get(g)
            if pos is not None:
                w_out_new[c] = local[pos]

        delta_u = x_flat @ (w_in_new - p["fc1_w"][:, r]) + (b_new - p["fc1_b"][r])
        u_new = u1[:, r] + delta_u
        h_new = np.tanh(u_new)

        logits = np.array(logits0, copy=True)
        logits += h_new[:, None] * w_out_new[None, :]
        logits -= h1[:, r][:, None] * p["fc2_w"][r, :][None, :]

        log_probs = logits - logsumexp(logits, axis=1, keepdims=True)
        ll = float(np.sum(log_probs[np.arange(y_train.shape[0]), y_train.astype(np.int64)]))
        lp = base_lp + float(model._prior_delta(theta_ref, local, active_indices))
        log_pi_proposals[i] = lp + ll

    if not np.all(np.isfinite(log_pi_proposals)):
        raise ValueError(f"non-finite hidden-pathway log_pi proposals: {log_pi_proposals}")

    delta = log_pi_proposals - log_pi_current
    accept_probs = np.where(delta >= 0.0, 1.0, np.exp(np.clip(delta, -60.0, 0.0)))

    return LocalMoveTable(
        active_indices=active_indices,
        current_local=current_local,
        proposal_states=np.asarray(proposal_states, dtype=np.float64),
        move_labels=list(move_labels),
        log_pi_current=float(log_pi_current),
        log_pi_proposals=np.asarray(log_pi_proposals, dtype=np.float64),
        accept_probs=np.asarray(accept_probs, dtype=np.float64),
        num_directions=int(_infer_num_directions(move_labels)),
    )
