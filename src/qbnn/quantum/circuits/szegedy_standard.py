from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import QFT

from src.qbnn.quantum.circuits.qpe_textbook import build_qpe_diagnostic


@dataclass
class SzegedyProblem:
    transition_matrix: np.ndarray
    walk_unitary: np.ndarray | None
    sample_circuit: QuantumCircuit | None
    diagnostic_qpe_circuit: QuantumCircuit | None
    logical_info: dict


def _row_angle_2state(row: np.ndarray) -> float:
    row = np.asarray(row, dtype=np.float64)
    row = np.clip(row, 0.0, None)
    s = float(row.sum())
    if s <= 0.0:
        raise ValueError(f"Invalid transition row for 2-state Szegedy gate: {row}")
    row = row / s
    # RY(theta)|0> = cos(theta/2)|0> + sin(theta/2)|1>
    return 2.0 * np.arctan2(np.sqrt(row[1]), np.sqrt(row[0]))


def _a_gate_2state(p: np.ndarray):
    """
    A = sum_x |x><x| ⊗ U_x
    with U_x |0> = sum_y sqrt(P_xy) |y>
    system qubits are ordered as [x, y]
    """
    qc = QuantumCircuit(2, name="A2")
    theta0 = _row_angle_2state(p[0])
    theta1 = _row_angle_2state(p[1])

    # Base prepare row 0 on y, then conditionally shift to row 1 when x=1
    qc.ry(theta0, 1)
    qc.cry(theta1 - theta0, 0, 1)
    return qc.to_gate(label="A2")


def _walk_gate_2state(p: np.ndarray):
    """
    W = R_B R_A with
    R_A = A (I ⊗ Z) A†
    R_B = S R_A S
    For d=2, the reflection about |0> on the second register is just Z on y.
    """
    A = _a_gate_2state(p)

    qc = QuantumCircuit(2, name="W2")
    # R_A
    qc.append(A, [0, 1])
    qc.z(1)
    qc.append(A.inverse(), [0, 1])

    # R_B = S R_A S
    qc.swap(0, 1)
    qc.append(A, [0, 1])
    qc.z(1)
    qc.append(A.inverse(), [0, 1])
    qc.swap(0, 1)

    return qc.to_gate(label="W2")


def _stationary_prep_gate_2state(p: np.ndarray, pi: np.ndarray):
    """
    Prepare the coherent stationary state:
        sum_x sqrt(pi_x) |x> |psi_x>
    where |psi_x> = sum_y sqrt(P_xy) |y>
    """
    pi = np.asarray(pi, dtype=np.float64)
    pi = np.clip(pi, 0.0, None)
    s = float(pi.sum())
    if s <= 0.0:
        raise ValueError(f"Invalid stationary distribution for 2-state Szegedy prep: {pi}")
    pi = pi / s

    theta_pi = 2.0 * np.arctan2(np.sqrt(pi[1]), np.sqrt(pi[0]))

    qc = QuantumCircuit(2, name="prep_pi")
    qc.ry(theta_pi, 0)               # prepare sqrt(pi) on x
    qc.append(_a_gate_2state(p), [0, 1])  # attach |psi_x> on y
    return qc.to_gate(label="prep_pi")


def _build_szegedy_qpe_circuit_2state(p: np.ndarray, pi: np.ndarray, num_eval_qubits: int) -> QuantumCircuit:
    eval_reg = QuantumRegister(num_eval_qubits, "eval")
    sys_reg = QuantumRegister(2, "sys")   # sys[0]=x, sys[1]=y

    c_eval = ClassicalRegister(num_eval_qubits, "c_eval")
    c_sys = ClassicalRegister(2, "c_sys")

    qc = QuantumCircuit(eval_reg, sys_reg, c_eval, c_sys, name="szegedy_qpe_2")

    # Prepare coherent stationary eigenstate
    qc.append(_stationary_prep_gate_2state(p, pi), sys_reg)

    # Standard QPE
    qc.h(eval_reg)

    W = _walk_gate_2state(p)

    for k in range(num_eval_qubits):
        power = 2 ** k
        sub = QuantumCircuit(2, name=f"W^{power}")
        for _ in range(power):
            sub.append(W, [0, 1])

        # IMPORTANT: control a structured custom gate, not a dense UnitaryGate(matrix)
        qc.append(sub.to_gate().control(1), [eval_reg[k], sys_reg[0], sys_reg[1]])

    qc.append(QFT(num_eval_qubits, inverse=True, do_swaps=True).to_gate(), eval_reg)

    # Keep register order compatible with szegedy_zero_phase_state_probs:
    # counts look like "sys_bits eval_bits" when split by spaces
    qc.measure(eval_reg, c_eval)
    qc.measure(sys_reg, c_sys)

    return qc


def build_szegedy_qpe_problem(
    p: np.ndarray,
    num_eval_qubits: int = 4,
    max_dense_states: int = 16,
    target_pi: np.ndarray | None = None,
) -> SzegedyProblem:
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

    # For your current noisy LeNet-2 config, this is the path that matters.
    if n_states == 2:
        if target_pi is None:
            raise ValueError("target_pi is required for faithful 2-state Szegedy QPE.")

        sample_circuit = _build_szegedy_qpe_circuit_2state(p, target_pi, num_eval_qubits)

        logical = {
            "family": "szegedy",
            "state_qubits": 1,
            "eval_qubits": int(num_eval_qubits),
            "qpe_skipped": False,
            "implementation": "structured_2state_qpe",
        }

        return SzegedyProblem(
            transition_matrix=p,
            walk_unitary=None,
            sample_circuit=sample_circuit,
            diagnostic_qpe_circuit=None,
            logical_info=logical,
        )

    # Keep the old dense-matrix path only as a diagnostic / ideal fallback for >2 states.
    from src.qbnn.quantum.circuits.szegedy_standard import build_szegedy_walk_unitary  # if kept in same file, remove this import
    W = build_szegedy_walk_unitary(p)
    qpe_bundle = build_qpe_diagnostic(
        W,
        num_eval_qubits=num_eval_qubits,
        allow_dense_control=False,
        reason_if_skipped="Dense Szegedy circuit-QPE skipped locally; using classical spectral diagnostics instead.",
    )

    logical = {
        "family": "szegedy",
        "state_qubits": int(np.ceil(np.log2(n_states))),
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