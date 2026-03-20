from __future__ import annotations


def logical_resource_report(circuit) -> dict:
    ops = circuit.count_ops()
    two_q = 0
    for name, count in ops.items():
        if any(k in name.lower() for k in ["cx", "cz", "swap", "ecr", "cp", "ccx", "mc"]):
            two_q += int(count)
    return {
        "qubits": int(circuit.num_qubits),
        "clbits": int(circuit.num_clbits),
        "depth": int(circuit.depth() or 0),
        "size": int(circuit.size()),
        "ops": {str(k): int(v) for k, v in ops.items()},
        "two_qubit_like_ops": int(two_q),
    }
