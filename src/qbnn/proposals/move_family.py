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
    output_mode: str
    num_output_patterns: int

    @property
    def num_directions(self) -> int:
        # Every logical direction contributes exactly two signed moves.
        return len(self.move_labels) // 2

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

    proposal_states: list[np.ndarray] = []
    move_labels: list[tuple[int, int]] = []
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


def _build_zero_sum_output_patterns(
    num_classes: int,
    *,
    output_mode: str,
    reference_class: int = 0,
    output_pattern_count: int | None = None,
) -> list[np.ndarray]:
    c = int(num_classes)
    if c < 2:
        raise ValueError("num_classes must be >= 2 for hidden-pathway proposals")

    if output_mode == "identical_all_classes":
        patterns = [np.ones(c, dtype=np.float64)]
    elif output_mode == "zero_sum_vs_reference":
        ref = int(reference_class) % c
        patterns = []
        for target in range(c):
            if target == ref:
                continue
            v = np.zeros(c, dtype=np.float64)
            v[target] = 1.0
            v[ref] = -1.0
            patterns.append(v)
    elif output_mode == "zero_sum_one_vs_rest":
        patterns = []
        for target in range(c):
            v = -np.ones(c, dtype=np.float64) / float(c - 1)
            v[target] = 1.0
            patterns.append(v)
    else:
        raise ValueError(
            "Unsupported hidden-pathway output_mode: "
            f"{output_mode!r}. Expected one of "
            "{'identical_all_classes', 'zero_sum_vs_reference', 'zero_sum_one_vs_rest'}."
        )

    if output_pattern_count is not None:
        k = max(1, min(int(output_pattern_count), len(patterns)))
        patterns = patterns[:k]

    return patterns


def build_hidden_pathway_moves(
    model,
    active_indices: np.ndarray,
    theta_ref: np.ndarray,
    *,
    step_in: float,
    step_out: float,
    bias_step: float | None = None,
    output_mode: str = "zero_sum_one_vs_rest",
    output_reference_class: int = 0,
    output_pattern_count: int | None = None,
) -> HiddenPathwayMoves:
    active_indices = np.asarray(active_indices, dtype=np.int64)
    current_local = np.asarray(theta_ref[active_indices], dtype=np.float64)
    if bias_step is None:
        bias_step = float(step_in)

    required = ("s_fc1_w", "s_fc1_b", "s_fc2_w", "h", "in_dim", "num_classes")
    missing = [name for name in required if not hasattr(model, name)]
    if missing:
        raise ValueError(
            f"build_hidden_pathway_moves requires model attrs {required}, missing={missing}"
        )

    global_to_local = {int(g): i for i, g in enumerate(active_indices.tolist())}
    output_patterns = _build_zero_sum_output_patterns(
        int(model.num_classes),
        output_mode=output_mode,
        reference_class=int(output_reference_class),
        output_pattern_count=output_pattern_count,
    )

    proposal_states: list[np.ndarray] = []
    move_labels: list[tuple[int, int]] = []

    for r in range(int(model.h)):
        touched = False
        for j in range(int(model.in_dim)):
            g = int(model.s_fc1_w.start + j * int(model.h) + r)
            if g in global_to_local:
                touched = True
                break

        if (not touched) and (int(model.s_fc1_b.start + r) in global_to_local):
            touched = True

        if not touched:
            for cls in range(int(model.num_classes)):
                g = int(model.s_fc2_w.start + r * int(model.num_classes) + cls)
                if g in global_to_local:
                    touched = True
                    break

        if not touched:
            continue

        for out_pattern in output_patterns:
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

                for cls in range(int(model.num_classes)):
                    g = int(model.s_fc2_w.start + r * int(model.num_classes) + cls)
                    pos = global_to_local.get(g)
                    if pos is not None:
                        cand[pos] += sign * float(step_out) * float(out_pattern[cls])

                proposal_states.append(cand)
                # Keep the label layout runner/oracle-compatible: (unit_id, sign_bit).
                # Multiple output patterns per unit are allowed; the proposal state fully
                # defines the move, and runner code should use moves.num_directions.
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
        output_mode=str(output_mode),
        num_output_patterns=len(output_patterns),
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
    proposal_states: list[np.ndarray] = []
    move_labels: list[tuple[int, int]] = []

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
