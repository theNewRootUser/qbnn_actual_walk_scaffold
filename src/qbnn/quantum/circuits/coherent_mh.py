from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import StatePreparation

from src.qbnn.quantum.utils import amplitude_vector_from_probabilities, bitstring_of_int


@dataclass
class CoherentMHProblem:
    transition_matrix: np.ndarray
    current_state: int
    sample_circuit: QuantumCircuit
    diagnostic_qpe_circuit: QuantumCircuit | None
    logical_info: dict


def _classical_spectral_diagnostics_from_transition(p: np.ndarray) -> dict:
    eigvals = np.linalg.eigvals(p)
    phases = np.mod(np.angle(eigvals) / (2.0 * np.pi), 1.0)
    phases = np.sort(phases)
    return {
        "operator_dim": int(p.shape[0]),
        "unitarity_error": None,  # this is a stochastic matrix, not a unitary
        "eigenphases_sorted": [float(x) for x in phases.tolist()],
        "top_k_eigenphases": [float(x) for x in phases[: min(8, len(phases))].tolist()],
        "min_phase_gap": float(np.min(np.diff(phases)) if len(phases) > 1 else 0.0),
    }


def build_coherent_mh_problem(
    p: np.ndarray,
    current_state: int,
    num_eval_qubits: int = 4,
    build_qpe: bool = True,
    max_dense_diag_states: int = 32,
) -> CoherentMHProblem:
    """
    Safe coherent-MH local sampler.

    Current behavior:
    - sampling path: prepares the transition-row distribution p[current_state, :]
      on the destination/state register and measures it
    - diagnostic path: skips dense controlled-unitary QPE and stores only
      classical spectral diagnostics for local debugging

    This is intentional to avoid Qiskit's controlled-arbitrary-unitary synthesis
    crash during local diagnostics.
    """
    p = np.asarray(p, dtype=np.float64)
    if p.shape[0] != p.shape[1]:
        raise ValueError("transition matrix must be square")

    n_states = p.shape[0]
    n = int(np.ceil(np.log2(n_states)))
    if 2 ** n != n_states:
        raise ValueError("number of states must be a power of two")

    if not (0 <= current_state < n_states):
        raise ValueError(f"current_state={current_state} out of range for {n_states} states")

    row_probs = np.asarray(p[current_state], dtype=np.float64)
    row_probs = row_probs / np.sum(row_probs)
    amp = amplitude_vector_from_probabilities(row_probs)

    x = QuantumRegister(n, "x")
    y = QuantumRegister(n, "y")
    c = ClassicalRegister(n, "c")
    sample = QuantumCircuit(x, y, c, name="coherent_mh_sample")

    # encode current local state on x register
    bs = bitstring_of_int(current_state, n)[::-1]
    for k, b in enumerate(bs):
        if b == "1":
            sample.x(x[k])

    # prepare the row distribution on y
    sample.append(StatePreparation(amp), y)

    # measure the sampled next-state register
    sample.measure(y, c)

    qpe_circ = None
    logical_info = {
        "family": "coherent_mh",
        "num_states": int(n_states),
        "state_qubits": int(n),
        "sample_total_qubits": int(sample.num_qubits),
        "diagnostic_built": False,
        "diagnostic_skipped": bool(build_qpe),
        "diagnostic_skip_reason": (
            "Dense circuit-QPE disabled for coherent-MH local diagnostics; "
            "using classical spectral diagnostics instead."
            if build_qpe else None
        ),
        "classical_qpe_diagnostics": _classical_spectral_diagnostics_from_transition(p),
        "current_state": int(current_state),
        "current_row_distribution": row_probs.tolist(),
    }

    return CoherentMHProblem(
        transition_matrix=p,
        current_state=current_state,
        sample_circuit=sample,
        diagnostic_qpe_circuit=qpe_circ,
        logical_info=logical_info,
    )