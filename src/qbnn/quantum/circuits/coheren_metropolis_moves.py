from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RYGate


@dataclass
class CoherentMoveProblem:
    sample_circuit: QuantumCircuit
    logical_info: dict


def build_coherent_metropolis_move_problem(
    accept_probs: np.ndarray,
    *,
    num_directions: int,
) -> CoherentMoveProblem:
    """
    Build a coherent move-sampling circuit for a signed direction bank.

    Register layout is kept backward-compatible with the existing runner:
      - d: direction register
      - s: sign qubit
      - a: accept qubit
      - c: classical bits measuring [d..., s, a]

    Key change versus the old version:
      - accept_probs may have ANY length from 1 up to 2 * num_directions
      - if a block produces fewer actual signed moves than the configured maximum,
        the remaining move slots are padded as dummy reject moves (accept prob = 0)

    This avoids crashes on small tail blocks like fc1_b / fc2_b / fc2_w, while
    preserving the existing measurement layout and downstream decode assumptions.
    """
    accept_probs = np.asarray(accept_probs, dtype=np.float64).reshape(-1)
    if accept_probs.size < 1:
        raise ValueError("accept_probs must be non-empty")

    num_directions = int(num_directions)
    if num_directions < 1:
        raise ValueError("num_directions must be >= 1")

    # Maximum signed moves supported by the configured direction bank.
    expected_moves = 2 * num_directions

    # Actual block may expose fewer valid moves; that is fine.
    if accept_probs.size > expected_moves:
        raise ValueError(
            f"expected at most {expected_moves} accept probabilities, got {accept_probs.size}"
        )

    # Clip to [0, 1] and pad missing move slots with zero-acceptance dummy moves.
    accept_probs = np.clip(accept_probs, 0.0, 1.0)
    padded_accept = np.zeros(expected_moves, dtype=np.float64)
    padded_accept[: accept_probs.size] = accept_probs

    dir_bits = int(np.ceil(np.log2(num_directions)))

    d = QuantumRegister(dir_bits, "d")
    s = QuantumRegister(1, "s")
    a = QuantumRegister(1, "a")
    c = ClassicalRegister(dir_bits + 2, "c")
    qc = QuantumCircuit(d, s, a, c, name="coherent_metropolis_move")

    # Uniform proposal over the configured move register.
    for q in d:
        qc.h(q)
    qc.h(s[0])

    controls = list(d) + [s[0]]
    num_controls = len(controls)

    # Encode acceptance amplitudes.
    # Move indices beyond the actual move count remain zero-acceptance dummies.
    for move_idx, p in enumerate(padded_accept):
        p = float(p)
        if p <= 0.0:
            continue

        theta = 2.0 * np.arcsin(np.sqrt(p))
        ctrl_state = format(move_idx, f"0{num_controls}b")
        gate = RYGate(theta).control(num_controls, ctrl_state=ctrl_state)
        qc.append(gate, controls + [a[0]])

    # Measurement layout preserved:
    #   c[0:dir_bits]          <- d
    #   c[dir_bits]            <- s
    #   c[dir_bits + 1]        <- a
    qc.measure(d, c[:dir_bits])
    qc.measure(s, c[dir_bits : dir_bits + 1])
    qc.measure(a, c[dir_bits + 1 : dir_bits + 2])

    return CoherentMoveProblem(
        sample_circuit=qc,
        logical_info={
            "family": "coherent_metropolis_move",
            "num_directions": int(num_directions),
            "actual_num_moves": int(accept_probs.size),
            "num_moves": int(expected_moves),          # backward-compatible max size
            "padded_num_moves": int(expected_moves),
            "dir_bits": int(dir_bits),
            "sample_total_qubits": int(qc.num_qubits),
            "accept_probs": accept_probs.tolist(),     # actual probabilities only
            "padded_accept_probs": padded_accept.tolist(),
        },
    )