from __future__ import annotations

from qiskit import transpile


def transpiled_resource_report(
    circuit,
    backend=None,
    seed: int | None = None,
    optimization_level: int = 0,
):
    try:
        tqc = transpile(
            circuit,
            backend=backend,
            seed_transpiler=seed,
            optimization_level=optimization_level,
            translation_method="translator",
        )
        out = {
            "ok": True,
            "failed": False,
            "num_qubits": int(tqc.num_qubits),
            "depth": int(tqc.depth() or 0),
            "size": int(tqc.size() or 0),
            "count_ops": {k: int(v) for k, v in tqc.count_ops().items()},
            "error": None,
        }
        try:
            out["estimated_duration"] = float(tqc.estimate_duration())
        except Exception:
            out["estimated_duration"] = None
        return out
    except Exception as e:
        return {
            "ok": False,
            "failed": True,
            "num_qubits": int(circuit.num_qubits),
            "depth": int(circuit.depth() or 0),
            "size": int(circuit.size() or 0),
            "count_ops": {k: int(v) for k, v in circuit.count_ops().items()},
            "estimated_duration": None,
            "error": repr(e),
        }