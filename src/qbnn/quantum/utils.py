from __future__ import annotations
import math
import numpy as np
from scipy.linalg import qr


def normalize_probabilities(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    p = np.maximum(p, 0.0)
    s = p.sum()
    if s <= 0:
        raise ValueError("probabilities sum to zero")
    return p / s


def amplitude_vector_from_probabilities(p: np.ndarray) -> np.ndarray:
    return np.sqrt(normalize_probabilities(p)).astype(np.complex128)


def unitary_from_statevector(psi: np.ndarray) -> np.ndarray:
    psi = np.asarray(psi, dtype=np.complex128)
    psi = psi / np.linalg.norm(psi)
    n = psi.size
    seed = np.arange(1, n + 1, dtype=np.float64)
    basis_guess = np.eye(n, dtype=np.complex128)
    basis_guess[:, 0] = psi
    basis_guess[:, -1] += 1e-6 * seed
    q, _ = qr(basis_guess, mode="economic")
    if np.abs(np.vdot(q[:, 0], psi)) < 0.99:
        # deterministic repair
        mat = np.column_stack([psi, np.eye(n, dtype=np.complex128)[:, 1:]])
        q, _ = qr(mat, mode="economic")
    phase = np.vdot(q[:, 0], psi)
    q[:, 0] *= np.conj(phase) / (np.abs(phase) + 1e-12)
    return q


def bitstring_of_int(x: int, width: int) -> str:
    return format(int(x), f"0{width}b")


def nearest_phase_zero_probability(counts: dict[str, int], eval_width: int) -> float:
    total = max(1, sum(counts.values()))
    target = "0" * eval_width
    s = 0
    for bitstr, c in counts.items():
        if bitstr.replace(" ", "").endswith(target):
            s += c
    return float(s / total)
