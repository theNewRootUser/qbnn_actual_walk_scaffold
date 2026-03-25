
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class SignedDirectionMoves:
    active_indices: np.ndarray
    current_local: np.ndarray
    groups: list[np.ndarray]
    proposal_states: np.ndarray
    move_labels: list[tuple[int, int]]
    direction_step: float

    @property
    def num_directions(self) -> int:
        return len(self.groups)

    @property
    def num_moves(self) -> int:
        return len(self.move_labels)


@dataclass
class HiddenPathwayMoves:
    active_indices: np.ndarray
    current_local: np.ndarray
    proposal_states: np.ndarray
    move_labels: list[tuple[int, int]]
    step_in: float
    step_out: float
    bias_step: float

    @property
    def num_directions(self) -> int:
        return len({int(unit_id) for unit_id, _ in self.move_labels})

    @property
    def num_moves(self) -> int:
        return len(self.move_labels)


@dataclass
class OutputBiasMoves:
    active_indices: np.ndarray
    current_local: np.ndarray
    proposal_states: np.ndarray
    move_labels: list[tuple[int, int]]
    bias_step: float

    @property
    def num_directions(self) -> int:
        return len({int(class_id) for class_id, _ in self.move_labels})

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


def build_hidden_pathway_moves(
    model,
    active_indices: np.ndarray,
    theta_ref: np.ndarray,
    *,
    step_in: float,
    step_out: float,
    bias_step: float | None = None,
) -> HiddenPathwayMoves:
    active_indices = np.asarray(active_indices, dtype=np.int64)
    current_local = np.asarray(theta_ref[active_indices], dtype=np.float64)

    if bias_step is None:
        bias_step = float(step_in)

    required = ("s_fc1_w", "s_fc1_b", "s_fc2_w", "h", "in_dim", "num_classes")
    missing = [name for name in required if not hasattr(model, name)]
    if missing:
        raise ValueError(f"build_hidden_pathway_moves requires model attrs {required}, missing={missing}")

    global_to_local = {int(g): i for i, g in enumerate(active_indices.tolist())}

    proposal_states = []
    move_labels = []

    for r in range(int(model.h)):
        # Skip units that are not represented in this active block.
        touched = False
        for j in range(int(model.in_dim)):
            if int(model.s_fc1_w.start + j * int(model.h) + r) in global_to_local:
                touched = True
                break
        if not touched and int(model.s_fc1_b.start + r) in global_to_local:
            touched = True
        if not touched:
            for c in range(int(model.num_classes)):
                if int(model.s_fc2_w.start + r * int(model.num_classes) + c) in global_to_local:
                    touched = True
                    break
        if not touched:
            continue

        for sign_bit, sign in ((0, -1.0), (1, +1.0)):
            cand = np.array(current_local, copy=True)

            for j in range(int(model.in_dim)):
                g = int(model.s_fc1_w.start + j * int(model.h) + r)
                pos = global_to_local.get(g)
                if pos is not None:
                    cand[pos] += sign * float(step_in)

            g_bias = int(model.s_fc1_b.start + r)
            pos_bias = global_to_local.get(g_bias)
            if pos_bias is not None:
                cand[pos_bias] += sign * float(bias_step)

            for c in range(int(model.num_classes)):
                g = int(model.s_fc2_w.start + r * int(model.num_classes) + c)
                pos = global_to_local.get(g)
                if pos is not None:
                    cand[pos] += sign * float(step_out)

            proposal_states.append(cand)
            move_labels.append((int(r), int(sign_bit)))

    if not proposal_states:
        raise ValueError("No hidden-pathway proposals were generated for this block")

    return HiddenPathwayMoves(
        active_indices=active_indices,
        current_local=current_local,
        proposal_states=np.asarray(proposal_states, dtype=np.float64),
        move_labels=move_labels,
        step_in=float(step_in),
        step_out=float(step_out),
        bias_step=float(bias_step),
    )


def build_output_bias_moves(
    model,
    active_indices: np.ndarray,
    theta_ref: np.ndarray,
    *,
    bias_step: float,
) -> OutputBiasMoves:
    active_indices = np.asarray(active_indices, dtype=np.int64)
    current_local = np.asarray(theta_ref[active_indices], dtype=np.float64)

    if not hasattr(model, "s_fc2_b") or not hasattr(model, "num_classes"):
        raise ValueError("build_output_bias_moves requires model.s_fc2_b and model.num_classes")

    global_to_local = {int(g): i for i, g in enumerate(active_indices.tolist())}

    proposal_states = []
    move_labels = []

    for c in range(int(model.num_classes)):
        g = int(model.s_fc2_b.start + c)
        pos = global_to_local.get(g)
        if pos is None:
            continue
        for sign_bit, sign in ((0, -1.0), (1, +1.0)):
            cand = np.array(current_local, copy=True)
            cand[pos] += sign * float(bias_step)
            proposal_states.append(cand)
            move_labels.append((int(c), int(sign_bit)))

    if not proposal_states:
        raise ValueError("No output-bias proposals were generated for this block")

    return OutputBiasMoves(
        active_indices=active_indices,
        current_local=current_local,
        proposal_states=np.asarray(proposal_states, dtype=np.float64),
        move_labels=move_labels,
        bias_step=float(bias_step),
    )
