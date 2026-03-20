from __future__ import annotations
import os
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit.transpiler import generate_preset_pass_manager


def build_service(channel: str | None = None, instance: str | None = None, token: str | None = None):
    channel = channel or os.getenv("QISKIT_IBM_CHANNEL") or "ibm_quantum_platform"
    instance = instance or os.getenv("QISKIT_IBM_INSTANCE")
    token = token or os.getenv("QISKIT_IBM_TOKEN")
    if token is not None:
        kwargs = {"channel": channel, "token": token}
        if instance:
            kwargs["instance"] = instance
        return QiskitRuntimeService(**kwargs)
    kwargs = {"channel": channel}
    if instance:
        kwargs["instance"] = instance
    return QiskitRuntimeService(**kwargs)


def get_backend(service, backend_name: str | None = None):
    if backend_name is None:
        backends = service.backends(simulator=False, operational=True)
        if not backends:
            raise RuntimeError("No operational backends available")
        return sorted(backends, key=lambda b: b.num_qubits)[0]
    return service.backend(backend_name)


def run_ibm_sampler(circuit, backend, shots: int, optimization_level: int = 1):
    pm = generate_preset_pass_manager(backend=backend, optimization_level=optimization_level)
    isa = pm.run(circuit)
    sampler = Sampler(mode=backend)
    job = sampler.run([isa], shots=shots)
    res = job.result()[0]
    data = getattr(res, "data", None)
    if hasattr(data, "c"):
        try:
            return data.c.get_counts()
        except Exception:
            pass
    if hasattr(res, "join_data"):
        try:
            joined = res.join_data()
            if hasattr(joined, "get_counts"):
                return joined.get_counts()
        except Exception:
            pass
    raise RuntimeError("Could not extract counts from SamplerV2 result")
