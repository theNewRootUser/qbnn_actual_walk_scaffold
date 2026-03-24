from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from scipy.special import logsumexp, softmax
from sklearn.metrics import accuracy_score, f1_score, log_loss

from src.qbnn.config import ModelConfig


@dataclass
class Net2Cache:
    x_flat: np.ndarray          # [n, 256]
    u1: np.ndarray              # [n, 12]
    h1: np.ndarray              # [n, 12]
    logits: np.ndarray          # [n, 10]
    params: Dict[str, np.ndarray]


class BayesianNet2:
    """
    Proper early Net-2 / LeNet-2 style model:
        16x16 -> 12 -> 10
    with tanh hidden layer and linear logits.
    """

    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        self.num_classes = cfg.num_classes
        self.in_dim = cfg.image_height * cfg.image_width * cfg.num_channels
        self.h = cfg.fc_hidden

        if self.in_dim != 256:
            raise ValueError(f"Net2 expects 16x16x1 = 256 inputs, got {self.in_dim}")
        if self.h != 12:
            raise ValueError(f"Proper Net2 expects fc_hidden=12, got {self.h}")

        self.s_fc1_w = slice(0, self.in_dim * self.h)
        self.s_fc1_b = slice(self.s_fc1_w.stop, self.s_fc1_w.stop + self.h)
        self.s_fc2_w = slice(self.s_fc1_b.stop, self.s_fc1_b.stop + self.h * self.num_classes)
        self.s_fc2_b = slice(self.s_fc2_w.stop, self.s_fc2_w.stop + self.num_classes)
        self.num_params = self.s_fc2_b.stop

    def unpack(self, theta: np.ndarray) -> Dict[str, np.ndarray]:
        theta = np.asarray(theta, dtype=np.float64)
        if theta.ndim != 1 or theta.size != self.num_params:
            raise ValueError(f"theta must have shape ({self.num_params},), got {theta.shape}")

        return {
            "fc1_w": theta[self.s_fc1_w].reshape(self.in_dim, self.h),
            "fc1_b": theta[self.s_fc1_b],
            "fc2_w": theta[self.s_fc2_w].reshape(self.h, self.num_classes),
            "fc2_b": theta[self.s_fc2_b],
        }

    def _flatten_x(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 4:
            return x.reshape(x.shape[0], -1)
        if x.ndim == 3:
            return x.reshape(x.shape[0], -1)
        if x.ndim == 2:
            return x
        raise ValueError(f"unexpected x shape: {x.shape}")

    def build_cache(self, theta: np.ndarray, x: np.ndarray) -> Net2Cache:
        p = self.unpack(theta)
        x_flat = self._flatten_x(x)
        u1 = x_flat @ p["fc1_w"] + p["fc1_b"]
        h1 = np.tanh(u1)
        logits = h1 @ p["fc2_w"] + p["fc2_b"]
        return Net2Cache(x_flat=x_flat, u1=u1, h1=h1, logits=logits, params=p)

    def forward_logits(self, theta: np.ndarray, x: np.ndarray) -> np.ndarray:
        return self.build_cache(theta, x).logits

    def predict_proba(self, theta: np.ndarray, x: np.ndarray) -> np.ndarray:
        return softmax(self.forward_logits(theta, x), axis=1)

    def predict(self, theta: np.ndarray, x: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(theta, x), axis=1)

    def log_prior(self, theta: np.ndarray) -> float:
        sigma2 = float(self.cfg.prior_std ** 2)
        return float(-0.5 * theta.size * np.log(2.0 * np.pi * sigma2) - 0.5 * np.sum(theta ** 2) / sigma2)

    def log_likelihood(self, theta: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
        logits = self.forward_logits(theta, x)
        log_probs = logits - logsumexp(logits, axis=1, keepdims=True)
        return float(np.sum(log_probs[np.arange(y.shape[0]), y.astype(np.int64)]))

    def log_posterior(self, theta: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
        return self.log_prior(theta) + self.log_likelihood(theta, x, y)

    def _prior_delta(self, theta_ref: np.ndarray, local: np.ndarray, active_indices: np.ndarray) -> float:
        sigma2 = float(self.cfg.prior_std ** 2)
        old = theta_ref[active_indices]
        return float(-0.5 * (np.sum(local ** 2) - np.sum(old ** 2)) / sigma2)

    def _active_region(self, idx: np.ndarray) -> str | None:
        idx = np.asarray(idx, dtype=np.int64)

        def inside(s: slice) -> bool:
            return np.all((idx >= s.start) & (idx < s.stop))

        if inside(self.s_fc1_w):
            return "fc1_w"
        if inside(self.s_fc1_b):
            return "fc1_b"
        if inside(self.s_fc2_w):
            return "fc2_w"
        if inside(self.s_fc2_b):
            return "fc2_b"
        return None

    def local_block_log_posterior_table(
        self,
        theta_ref: np.ndarray,
        active_indices: np.ndarray,
        local_states: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray:
        """
        Exact fast path for Net2 blocks.
        Falls back to generic full recomputation if a block crosses regions.
        """
        active_indices = np.asarray(active_indices, dtype=np.int64)
        cache = self.build_cache(theta_ref, x)
        base_ll = float(np.sum(
            (cache.logits - logsumexp(cache.logits, axis=1, keepdims=True))[np.arange(y.shape[0]), y.astype(np.int64)]
        ))
        base_lp = self.log_prior(theta_ref)

        region = self._active_region(active_indices)
        out = np.empty(local_states.shape[0], dtype=np.float64)

        # generic fallback for mixed blocks
        if region is None:
            for i, local in enumerate(local_states):
                theta = np.array(theta_ref, copy=True)
                theta[active_indices] = local
                out[i] = self.log_posterior(theta, x, y)
            return out

        p = cache.params
        x_flat = cache.x_flat
        u1 = cache.u1
        h1 = cache.h1
        logits0 = cache.logits

        for i, local in enumerate(local_states):
            logits = logits0

            if region == "fc2_w":
                logits = np.array(logits0, copy=True)
                flat0 = active_indices - self.s_fc2_w.start
                rows = flat0 // self.num_classes
                cols = flat0 % self.num_classes
                for k, val in enumerate(local):
                    r = int(rows[k])
                    c = int(cols[k])
                    delta = val - p["fc2_w"][r, c]
                    logits[:, c] += h1[:, r] * delta

            elif region == "fc2_b":
                logits = np.array(logits0, copy=True)
                cols = active_indices - self.s_fc2_b.start
                for k, val in enumerate(local):
                    c = int(cols[k])
                    delta = val - p["fc2_b"][c]
                    logits[:, c] += delta

            elif region == "fc1_b":
                logits = np.array(logits0, copy=True)
                cols = active_indices - self.s_fc1_b.start
                touched = np.unique(cols)
                for r in touched:
                    r = int(r)
                    delta_b = 0.0
                    for k, val in enumerate(local):
                        if cols[k] == r:
                            delta_b = val - p["fc1_b"][r]
                    u_new = u1[:, r] + delta_b
                    h_new = np.tanh(u_new)
                    logits += (h_new - h1[:, r])[:, None] * p["fc2_w"][r, :][None, :]

            elif region == "fc1_w":
                logits = np.array(logits0, copy=True)
                flat0 = active_indices - self.s_fc1_w.start
                in_rows = flat0 // self.h
                hid_cols = flat0 % self.h
                touched = np.unique(hid_cols)

                for r in touched:
                    r = int(r)
                    mask = (hid_cols == r)
                    delta_u = np.zeros(x_flat.shape[0], dtype=np.float64)
                    for k in np.where(mask)[0]:
                        j = int(in_rows[k])
                        delta = local[k] - p["fc1_w"][j, r]
                        delta_u += x_flat[:, j] * delta
                    u_new = u1[:, r] + delta_u
                    h_new = np.tanh(u_new)
                    logits += (h_new - h1[:, r])[:, None] * p["fc2_w"][r, :][None, :]

            log_probs = logits - logsumexp(logits, axis=1, keepdims=True)
            ll = float(np.sum(log_probs[np.arange(y.shape[0]), y.astype(np.int64)]))
            lp = base_lp + self._prior_delta(theta_ref, local, active_indices)
            out[i] = lp + ll

        return out


def predictive_metrics_from_samples(model, theta_samples: np.ndarray, x: np.ndarray, y: np.ndarray) -> dict:
    theta_samples = np.asarray(theta_samples, dtype=np.float64)
    if theta_samples.ndim != 2:
        raise ValueError("theta_samples must have shape [num_samples, num_params]")
    probs = np.mean([model.predict_proba(theta, x) for theta in theta_samples], axis=0)
    y_hat = np.argmax(probs, axis=1)
    return {
        "accuracy": float(accuracy_score(y, y_hat)),
        "macro_f1": float(f1_score(y, y_hat, average="macro")),
        "nll": float(log_loss(y, probs, labels=list(range(model.num_classes)))),
    }


def build_bayesian_model(cfg: ModelConfig):
    if cfg.architecture == "net2":
        return BayesianNet2(cfg)
    if cfg.architecture == "lenet_like_zipcnn":
        return BayesianLeNetLikeZipCNN(cfg)  # rename your current class to this
    raise ValueError(f"Unknown architecture: {cfg.architecture}")