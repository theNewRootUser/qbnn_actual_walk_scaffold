from __future__ import annotations
import os
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit.transpiler import generate_preset_pass_manager

import os
from qiskit_ibm_runtime import QiskitRuntimeService

def build_service(channel: str | None = None,
                  instance: str | None = None,
                  token: str | None = None,
                  verify: bool | None = None,
                  proxies: dict | None = None):
    kwargs = {}

    channel = channel or os.environ.get("QISKIT_IBM_CHANNEL")
    instance = instance or os.environ.get("QISKIT_IBM_INSTANCE")
    token = token or os.environ.get("QISKIT_IBM_TOKEN")

    if channel:
        kwargs["channel"] = channel
    if instance:
        kwargs["instance"] = instance
    if token:
        kwargs["token"] = token

    # Optional network overrides
    verify_env = os.environ.get("QISKIT_IBM_VERIFY")
    if verify is not None:
        kwargs["verify"] = verify
    elif verify_env is not None:
        kwargs["verify"] = verify_env.strip().lower() not in {"0", "false", "no"}

    if proxies is not None:
        kwargs["proxies"] = proxies

    return QiskitRuntimeService(**kwargs)


def get_backend(service, backend_name: str | None = None, instance: str | None = None):
    # First try direct named lookup in the already-initialized service context.
    if backend_name is not None:
        try:
            return service.backend(backend_name)
        except Exception:
            pass

    # Fall back to the unfiltered visible backend list.
    visible = service.backends()

    print('visible', visible)

    # Keep only real IBM hardware-style backends.
    visible = [b for b in visible if getattr(b, "name", "").startswith("ibm_")]

    if backend_name is not None:
        matches = [b for b in visible if b.name == backend_name]
        if not matches:
            raise RuntimeError(
                f"Backend {backend_name!r} not visible. Visible backends: {[b.name for b in visible]}"
            )
        return matches[0]

    # Auto-pick a real backend if none was specified.
    if not visible:
        raise RuntimeError("No visible IBM backends found")
    return sorted(visible, key=lambda b: getattr(b, "num_qubits", 10**9))[0]


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
