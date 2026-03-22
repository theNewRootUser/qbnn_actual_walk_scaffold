from __future__ import annotations

import numpy as np
from src.qbnn.models import BayesianLeNet2, predictive_metrics_from_samples


def distribution_diagnostics(empirical: dict[int, float] | np.ndarray, target: np.ndarray) -> dict:
    target = np.asarray(target, dtype=float).reshape(-1)
    if target.size == 0:
        raise ValueError("target distribution is empty")

    # Build a dense empirical vector aligned to target state indices
    emp = np.zeros_like(target, dtype=float)

    if isinstance(empirical, dict):
        for k, v in empirical.items():
            i = int(k)
            if 0 <= i < emp.size:
                emp[i] += float(v)
    else:
        arr = np.asarray(empirical, dtype=float).reshape(-1)
        n = min(emp.size, arr.size)
        emp[:n] = arr[:n]

    # Normalize defensively
    emp_sum = float(emp.sum())
    if emp_sum > 0.0:
        emp /= emp_sum

    target_sum = float(target.sum())
    if target_sum > 0.0:
        target = target / target_sum

    tv = 0.5 * float(np.abs(emp - target).sum())
    mask = (emp > 0) & (target > 0)
    kl = float(np.sum(emp[mask] * np.log(emp[mask] / target[mask])))

    return {
        "tv_distance": tv,
        "kl_divergence": kl,
        "empirical": emp.tolist(),
        "target": target.tolist(),
    }


def evaluate_theta_samples(
    model: BayesianLeNet2,
    theta_samples: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    return predictive_metrics_from_samples(model, theta_samples, x_test, y_test)