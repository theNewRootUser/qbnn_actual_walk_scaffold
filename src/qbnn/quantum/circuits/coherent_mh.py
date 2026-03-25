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


def _build_sample_circuit_from_row(row_probs: np.ndarray, current_state: int) -> tuple[QuantumCircuit, int]:
    row_probs = np.asarray(row_probs, dtype=np.float64)
    n_states = int(row_probs.shape[0])
    n = int(np.ceil(np.log2(n_states)))
    if 2 ** n != n_states:
        raise ValueError("number of states must be a power of two")

    if not (0 <= current_state < n_states):
        raise ValueError(f"current_state={current_state} out of range for {n_states} states")

    row_probs = np.clip(row_probs, 0.0, None)
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
    return sample, n


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

    row_probs = np.asarray(p[current_state], dtype=np.float64)
    sample, n = _build_sample_circuit_from_row(row_probs=row_probs, current_state=current_state)

    qpe_circ = None
    logical_info = {
        "family": "coherent_mh",
        "num_states": int(p.shape[0]),
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
        "current_row_distribution": (row_probs / np.sum(row_probs)).tolist(),
    }

    return CoherentMHProblem(
        transition_matrix=p,
        current_state=current_state,
        sample_circuit=sample,
        diagnostic_qpe_circuit=qpe_circ,
        logical_info=logical_info,
    )


def build_coherent_mh_row_problem(
    row_probs: np.ndarray,
    current_state: int,
    num_eval_qubits: int = 4,
    build_qpe: bool = False,
    metadata: dict | None = None,
) -> CoherentMHProblem:
    """
    Row-only coherent-MH problem builder.

    This is for delayed-acceptance experiments where we only estimate the current
    transition row classically, then prepare that row directly on the sample register.
    It is coherent on the sampling circuit side, but it does not represent a full
    transition operator suitable for dense spectral diagnostics or Szegedy-style walks.
    """
    row_probs = np.asarray(row_probs, dtype=np.float64)
    if row_probs.ndim != 1:
        raise ValueError("row_probs must be one-dimensional")

    n_states = int(row_probs.shape[0])
    sample, n = _build_sample_circuit_from_row(row_probs=row_probs, current_state=current_state)

    placeholder_transition = np.eye(n_states, dtype=np.float64)
    placeholder_transition[current_state, :] = row_probs / np.sum(row_probs)

    qpe_circ = None
    logical_info = {
        "family": "coherent_mh",
        "num_states": int(n_states),
        "state_qubits": int(n),
        "sample_total_qubits": int(sample.num_qubits),
        "diagnostic_built": False,
        "diagnostic_skipped": True,
        "diagnostic_skip_reason": "Row-only delayed-acceptance estimate does not define a full transition matrix for coherent-MH diagnostics.",
        "classical_qpe_diagnostics": None,
        "current_state": int(current_state),
        "current_row_distribution": (row_probs / np.sum(row_probs)).tolist(),
        "row_only_estimate": True,
        "metadata": metadata or {},
    }

    return CoherentMHProblem(
        transition_matrix=placeholder_transition,
        current_state=current_state,
        sample_circuit=sample,
        diagnostic_qpe_circuit=qpe_circ,
        logical_info=logical_info,
    )

def build_coherent_row_sampler(row_probs: np.ndarray, family: str = "coherent_mh_sparse"):
    row_probs = np.asarray(row_probs, dtype=np.float64)
    row_probs = row_probs / row_probs.sum()

    support_size = int(row_probs.size)
    n = int(np.ceil(np.log2(max(1, support_size))))
    padded_size = 2 ** n

    padded = np.zeros(padded_size, dtype=np.float64)
    padded[:support_size] = row_probs
    padded /= padded.sum()

    amp = amplitude_vector_from_probabilities(padded)

    y = QuantumRegister(n, "y")
    c = ClassicalRegister(n, "c")
    sample = QuantumCircuit(y, c, name="coherent_sparse_row_sample")
    sample.append(StatePreparation(amp), y)
    sample.measure(y, c)

    return {
        "sample_circuit": sample,
        "logical_info": {
            "family": family,
            "support_size": support_size,
            "state_qubits": n,
            "sample_total_qubits": int(sample.num_qubits),
            "current_row_distribution": row_probs.tolist(),
            "padded_size": padded_size,
        },
    }