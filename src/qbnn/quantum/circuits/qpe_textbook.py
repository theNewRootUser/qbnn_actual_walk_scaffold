from __future__ import annotations
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import QFT, UnitaryGate
from dataclasses import dataclass
from typing import Any, Optional
from qiskit.quantum_info import Operator


def build_qpe_circuit(unitary: np.ndarray, init_prep: QuantumCircuit, num_eval_qubits: int, system_qubits: int, measure: bool = True) -> QuantumCircuit:
    eval_reg = QuantumRegister(num_eval_qubits, "eval")
    sys_reg = QuantumRegister(system_qubits, "sys")
    qc = QuantumCircuit(eval_reg, sys_reg, name="qpe")
    qc.compose(init_prep, qubits=sys_reg, inplace=True)
    qc.h(eval_reg)
    for k in range(num_eval_qubits):
        power = 2 ** k
        up = UnitaryGate(np.linalg.matrix_power(unitary, power), label=f"U^{power}")
        qc.append(up.control(1), [eval_reg[k], *sys_reg])
    qc.append(QFT(num_eval_qubits, inverse=True, do_swaps=True).to_gate(), eval_reg)
    if not measure:
        return qc
    c_eval = ClassicalRegister(num_eval_qubits, "c_eval")
    c_sys = ClassicalRegister(system_qubits, "c_sys")
    out = QuantumCircuit(eval_reg, sys_reg, c_eval, c_sys, name="qpe_meas")
    out.compose(qc, inplace=True)
    out.measure(eval_reg, c_eval)
    out.measure(sys_reg, c_sys)
    return out



@dataclass
class QPEDiagnosticBundle:
    circuit: Optional[QuantumCircuit]
    classical: dict[str, Any]
    skipped: bool
    reason: Optional[str] = None


def _to_unitary_matrix(op: Any) -> np.ndarray:
    if isinstance(op, np.ndarray):
        return np.asarray(op, dtype=complex)
    if hasattr(op, "to_matrix"):
        return np.asarray(op.to_matrix(), dtype=complex)
    return np.asarray(Operator(op).data, dtype=complex)


def classical_phase_diagnostics(op: Any, top_k: int = 8) -> dict[str, Any]:
    U = _to_unitary_matrix(op)
    eigvals = np.linalg.eigvals(U)
    phases = np.angle(eigvals) / (2.0 * np.pi)
    phases = np.mod(phases, 1.0)

    order = np.argsort(phases)
    phases_sorted = phases[order]

    out = {
        "operator_dim": int(U.shape[0]),
        "unitarity_error": float(np.linalg.norm(U.conj().T @ U - np.eye(U.shape[0]))),
        "eigenphases_sorted": [float(x) for x in phases_sorted.tolist()],
        "top_k_eigenphases": [float(x) for x in phases_sorted[: min(top_k, len(phases_sorted))].tolist()],
        "min_phase_gap": float(
            np.min(np.diff(phases_sorted)) if len(phases_sorted) > 1 else 0.0
        ),
    }
    return out


def build_qpe_diagnostic(
    op: Any,
    num_eval_qubits: int,
    *,
    allow_dense_control: bool = False,
    reason_if_skipped: str = "Skipped circuit-QPE for dense arbitrary unitary; using classical spectral diagnostics instead.",
) -> QPEDiagnosticBundle:
    """
    Safe diagnostic builder.

    For arbitrary dense unitaries, do NOT try to synthesize a controlled version.
    Return classical spectral diagnostics only.
    """

    classical = classical_phase_diagnostics(op)

    if not allow_dense_control:
        return QPEDiagnosticBundle(
            circuit=None,
            classical=classical,
            skipped=True,
            reason=reason_if_skipped,
        )

    # Only use the below branch if you later replace dense UnitaryGate objects
    # with structured, safely controllable circuits/gates.
    U = _to_unitary_matrix(op)
    n_sys = int(round(np.log2(U.shape[0])))

    qc = QuantumCircuit(num_eval_qubits + n_sys, num_eval_qubits)
    eval_reg = list(range(num_eval_qubits))
    sys_reg = list(range(num_eval_qubits, num_eval_qubits + n_sys))

    for q in eval_reg:
        qc.h(q)

    # WARNING:
    # this path is intentionally disabled by default because controlling a dense
    # arbitrary unitary is exactly what is crashing in your environment.
    from qiskit.circuit.library import UnitaryGate

    base = UnitaryGate(U)

    for k in range(num_eval_qubits):
        power = 2**k
        Up = np.linalg.matrix_power(U, power)
        up_gate = UnitaryGate(Up)
        qc.append(up_gate.control(1), [eval_reg[k], *sys_reg])

    # inverse QFT
    for j in range(num_eval_qubits // 2):
        qc.swap(j, num_eval_qubits - 1 - j)
    for j in range(num_eval_qubits):
        for m in range(j):
            qc.cp(-np.pi / (2 ** (j - m)), m, j)
        qc.h(j)

    qc.measure(eval_reg, range(num_eval_qubits))

    return QPEDiagnosticBundle(
        circuit=qc,
        classical=classical,
        skipped=False,
        reason=None,
    )
