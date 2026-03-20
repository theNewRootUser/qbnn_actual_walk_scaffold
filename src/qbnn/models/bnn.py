from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.special import logsumexp, softmax
from sklearn.metrics import accuracy_score, f1_score, log_loss
from src.qbnn.config import ModelConfig


def _avg_pool2x2(x: np.ndarray) -> np.ndarray:
    n, c, h, w = x.shape
    return x.reshape(n, c, h // 2, 2, w // 2, 2).mean(axis=(3, 5))


def _conv2d_valid_batch(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    n, cin, h, ww = x.shape
    cout, cin2, kh, kw = w.shape
    if cin != cin2:
        raise ValueError(f"conv channel mismatch: {cin} vs {cin2}")
    patches = sliding_window_view(x, (kh, kw), axis=(2, 3))
    y = np.einsum("ncxyij,fcij->nfxy", patches, w, optimize=True)
    y += b[None, :, None, None]
    return y


class BayesianLeNet2:
    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        self.num_classes = cfg.num_classes
        self.c1 = cfg.conv1_out
        self.c2 = cfg.conv2_out
        self.fc_hidden = cfg.fc_hidden
        self.num_params = (
            self.c1 * cfg.num_channels * 5 * 5 + self.c1 +
            2 * self.c1 +
            self.c2 * self.c1 * 5 * 5 + self.c2 +
            2 * self.c2 +
            (self.c2 * 2 * 2) * self.fc_hidden + self.fc_hidden +
            self.fc_hidden * cfg.num_classes + cfg.num_classes
        )

    def unpack(self, theta: np.ndarray) -> Dict[str, np.ndarray]:
        theta = np.asarray(theta, dtype=np.float64)
        if theta.ndim != 1 or theta.size != self.num_params:
            raise ValueError(f"theta must have shape ({self.num_params},), got {theta.shape}")
        i = 0
        s = {}
        s["conv1_w"] = theta[i:i + self.c1 * self.cfg.num_channels * 25].reshape(self.c1, self.cfg.num_channels, 5, 5); i += self.c1 * self.cfg.num_channels * 25
        s["conv1_b"] = theta[i:i + self.c1]; i += self.c1
        s["pool1_gamma"] = theta[i:i + self.c1]; i += self.c1
        s["pool1_beta"] = theta[i:i + self.c1]; i += self.c1
        s["conv2_w"] = theta[i:i + self.c2 * self.c1 * 25].reshape(self.c2, self.c1, 5, 5); i += self.c2 * self.c1 * 25
        s["conv2_b"] = theta[i:i + self.c2]; i += self.c2
        s["conv2_gamma"] = theta[i:i + self.c2]; i += self.c2
        s["conv2_beta"] = theta[i:i + self.c2]; i += self.c2
        flat_dim = self.c2 * 2 * 2
        s["fc1_w"] = theta[i:i + flat_dim * self.fc_hidden].reshape(flat_dim, self.fc_hidden); i += flat_dim * self.fc_hidden
        s["fc1_b"] = theta[i:i + self.fc_hidden]; i += self.fc_hidden
        s["fc2_w"] = theta[i:i + self.fc_hidden * self.cfg.num_classes].reshape(self.fc_hidden, self.cfg.num_classes); i += self.fc_hidden * self.cfg.num_classes
        s["fc2_b"] = theta[i:i + self.cfg.num_classes]
        return s

    def forward_logits(self, theta: np.ndarray, x: np.ndarray) -> np.ndarray:
        p = self.unpack(theta)
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 3:
            x = x[:, None, :, :]
        h = _conv2d_valid_batch(x, p["conv1_w"], p["conv1_b"])
        h = _avg_pool2x2(h)
        h = p["pool1_gamma"][None, :, None, None] * h + p["pool1_beta"][None, :, None, None]
        h = np.tanh(h)
        h = _conv2d_valid_batch(h, p["conv2_w"], p["conv2_b"])
        h = p["conv2_gamma"][None, :, None, None] * h + p["conv2_beta"][None, :, None, None]
        h = np.tanh(h)
        h = h.reshape(h.shape[0], -1)
        h = np.tanh(h @ p["fc1_w"] + p["fc1_b"])
        logits = h @ p["fc2_w"] + p["fc2_b"]
        return logits

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

    def local_block_log_posterior_table(self, theta_ref: np.ndarray, active_indices: np.ndarray, local_states: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        out = np.empty(local_states.shape[0], dtype=np.float64)
        for i, local in enumerate(local_states):
            theta = np.array(theta_ref, copy=True)
            theta[active_indices] = local
            out[i] = self.log_posterior(theta, x, y)
        return out


def predictive_metrics_from_samples(model: BayesianLeNet2, theta_samples: np.ndarray, x: np.ndarray, y: np.ndarray) -> dict:
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
    if cfg.architecture != "lenet2":
        raise ValueError(f"Unknown architecture: {cfg.architecture}")
    return BayesianLeNet2(cfg)
