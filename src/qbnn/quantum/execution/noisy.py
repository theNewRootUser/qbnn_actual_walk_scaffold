from __future__ import annotations
from qiskit import transpile
from qiskit_aer import AerSimulator


def build_local_fake_backend(name: str | None = None):
    try:
        from qiskit_ibm_runtime.fake_provider import FakeManilaV2
        return FakeManilaV2()
    except Exception:
        return None


def run_noisy_sampler(circuit, shots: int, seed: int = 1234, backend=None) -> dict[str, int]:
    if backend is not None:
        sim = AerSimulator.from_backend(backend)
    else:
        sim = AerSimulator(method="density_matrix")
    tqc = transpile(circuit, backend=sim, seed_transpiler=seed)
    result = sim.run(tqc, shots=shots, seed_simulator=seed).result()
    return result.get_counts()
