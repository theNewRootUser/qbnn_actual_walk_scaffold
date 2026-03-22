from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.linalg import block_diag
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import UnitaryGate
from src.qbnn.quantum.utils import amplitude_vector_from_probabilities, unitary_from_statevector, bitstring_of_int
from src.qbnn.quantum.circuits.qpe_textbook import build_qpe_diagnostic, build_qpe_circuit

@dataclass
class SzegedyProblem:
    transition_matrix: np.ndarray
    walk_unitary: np.ndarray | None
    sample_circuit: QuantumCircuit | None
    diagnostic_qpe_circuit: QuantumCircuit | None
    logical_info: dict


def _row_state_unitaries(p: np.ndarray) -> list[np.ndarray]:
    return [unitary_from_statevector(amplitude_vector_from_probabilities(row)) for row in p]


def _build_A_matrix(p: np.ndarray) -> np.ndarray:
    n_states = p.shape[0]
    n = int(np.ceil(np.log2(n_states)))
    d = 2 ** n
    if d != n_states:
        raise ValueError("states must be padded to a power of two")
    us = _row_state_unitaries(p)
    total_dim = d * d
    A = np.zeros((total_dim, total_dim), dtype=np.complex128)
    # Acts as sum_x |x><x| \otimes U_x
    for x in range(d):
        A[x*d:(x+1)*d, x*d:(x+1)*d] = us[x]
    return A


def _swap_matrix(d: int) -> np.ndarray:
    S = np.zeros((d*d, d*d), dtype=np.complex128)
    for x in range(d):
        for y in range(d):
            S[y*d + x, x*d + y] = 1.0
    return S


def _reflection_about_zero_on_second_register(d: int) -> np.ndarray:
    I = np.eye(d, dtype=np.complex128)
    z = np.zeros((d, d), dtype=np.complex128)
    z[0, 0] = 1.0
    return np.kron(I, 2.0 * z - np.eye(d, dtype=np.complex128))


def build_szegedy_walk_unitary(p: np.ndarray) -> np.ndarray:
    d = p.shape[0]
    A = _build_A_matrix(p)
    R0 = _reflection_about_zero_on_second_register(d)
    RA = A @ R0 @ A.conj().T
    S = _swap_matrix(d)
    RB = S @ RA @ S
    W = RB @ RA
    return W


def build_szegedy_qpe_problem(p: np.ndarray, num_eval_qubits: int = 4, max_dense_states: int = 16) -> SzegedyProblem:
    p = np.asarray(p, dtype=np.float64)
    n_states = p.shape[0]
    if p.shape[0] != p.shape[1]:
        raise ValueError("transition matrix must be square")

    if n_states > max_dense_states:
        logical = {
            "family": "szegedy",
            "num_states": int(n_states),
            "feasible": False,
            "reason": f"dense Szegedy walk disabled above {max_dense_states} states",
        }
        return SzegedyProblem(
            transition_matrix=p,
            walk_unitary=None,
            sample_circuit=None,
            diagnostic_qpe_circuit=None,
            logical_info=logical,
        )

    n = int(np.ceil(np.log2(n_states)))
    W = build_szegedy_walk_unitary(p)

    qpe_bundle = build_qpe_diagnostic(
        W,
        num_eval_qubits=num_eval_qubits,
        allow_dense_control=False,
        reason_if_skipped="Dense Szegedy circuit-QPE skipped locally; using classical spectral diagnostics instead.",
    )

    logical = {
        "family": "szegedy",
        "state_qubits": int(n),
        "eval_qubits": int(num_eval_qubits),
        "qpe_skipped": qpe_bundle.skipped,
        "qpe_skip_reason": qpe_bundle.reason,
        "classical_qpe_diagnostics": qpe_bundle.classical,
    }

    return SzegedyProblem(
        transition_matrix=p,
        walk_unitary=W,
        sample_circuit=None,
        diagnostic_qpe_circuit=None,
        logical_info=logical,
    )