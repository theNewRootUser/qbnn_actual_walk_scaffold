from __future__ import annotations
import numpy as np


def normalize_counts(counts):
    # Flat counts: {"0101": 12, ...}
    if not counts:
        return {}

    # Nested counts: {"eval": {...}, "state": {...}} -> unsupported here
    # because marginals are not enough to reconstruct the joint distribution.
    if any(isinstance(v, dict) for v in counts.values()):
        raise ValueError(
            "normalize_counts expected flat joint counts, but received nested "
            "register-wise counts. The sampler layer must combine classical "
            "register outputs into joint bitstrings first."
        )

    total = max(1, sum(int(v) for v in counts.values()))
    return {k: float(v) / float(total) for k, v in counts.items()}


def _parts(bitstr: str) -> list[str]:
    return bitstr.split() if " " in bitstr else [bitstr]


def state_probs_from_counts(counts: dict[str, int], state_qubits: int, measured_register: str = "tail") -> dict[int, float]:
    if not counts:
        return {}

    # tolerate wrapper shape {"counts": {...}, "metadata": {...}}
    if isinstance(counts, dict) and "counts" in counts and isinstance(counts["counts"], dict):
        counts = counts["counts"]

    norm = normalize_counts(counts)
    out: dict[int, float] = {}
    for bitstr, p in norm.items():
        raw = bitstr.replace(" ", "")
        if len(raw) < state_qubits:
            continue
        key = raw[-state_qubits:] if measured_register == "tail" else raw[:state_qubits]
        idx = int(key, 2)
        out[idx] = out.get(idx, 0.0) + p
    return out


def szegedy_zero_phase_state_probs(counts: dict[str, int], eval_qubits: int, state_qubits: int) -> dict[int, float]:
    out: dict[int, float] = {}
    kept = 0.0
    for bitstr, p in normalize_counts(counts).items():
        parts = _parts(bitstr)
        if len(parts) == 2:
            sys_bits, eval_bits = parts[0], parts[1]
        else:
            raw = parts[0]
            eval_bits = raw[-(eval_qubits):]
            sys_bits = raw[:-(eval_qubits)]
        if eval_bits == "0" * eval_qubits:
            left_bits = sys_bits[-state_qubits:]
            idx = int(left_bits, 2)
            out[idx] = out.get(idx, 0.0) + p
            kept += p
    if kept <= 0:
        return {}
    return {k: float(v / kept) for k, v in out.items()}


def sample_state_indices(state_probs: dict[int, float], num_samples: int, rng: np.random.Generator) -> np.ndarray:
    if not state_probs:
        raise ValueError("state_probs is empty")
    idx = np.array(sorted(state_probs.keys()), dtype=np.int64)
    p = np.array([state_probs[i] for i in idx], dtype=np.float64)
    p /= p.sum()
    return rng.choice(idx, size=num_samples, p=p)


def embed_local_samples(theta_ref: np.ndarray, active_indices: np.ndarray, local_states: np.ndarray, sampled_indices: np.ndarray) -> np.ndarray:
    out = np.repeat(np.asarray(theta_ref, dtype=np.float64)[None, :], repeats=sampled_indices.size, axis=0)
    out[:, active_indices] = local_states[sampled_indices]
    return out
