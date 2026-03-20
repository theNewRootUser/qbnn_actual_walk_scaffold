from __future__ import annotations
import numpy as np
from src.qbnn.models import BayesianLeNet2, predictive_metrics_from_samples


def distribution_diagnostics(empirical: dict[int, float], target: np.ndarray) -> dict:
    emp = np.asarray(emp, dtype=float)
    target = np.asarray(target, dtype=float)

    for k, v in empirical.items():
        if 0 <= int(k) < len(emp):
            emp[int(k)] += float(v)
    tv = 0.5 * float(np.abs(emp - target).sum())
    mask = (emp > 0) & (target > 0)
    kl = float(np.sum(emp[mask] * np.log(emp[mask] / target[mask])))
    return {"tv_distance": tv, "kl_divergence": kl, "empirical": emp.tolist(), "target": target.tolist()}


def evaluate_theta_samples(model: BayesianLeNet2, theta_samples: np.ndarray, x_test: np.ndarray, y_test: np.ndarray) -> dict:
    return predictive_metrics_from_samples(model, theta_samples, x_test, y_test)
