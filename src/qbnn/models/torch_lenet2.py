from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
from src.qbnn.config import ModelConfig, TrainingConfig


class TorchLeNet2(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.conv1 = nn.Conv2d(cfg.num_channels, cfg.conv1_out, kernel_size=5)
        self.pool = nn.AvgPool2d(2)
        self.pool1_gamma = nn.Parameter(torch.ones(cfg.conv1_out))
        self.pool1_beta = nn.Parameter(torch.zeros(cfg.conv1_out))
        self.conv2 = nn.Conv2d(cfg.conv1_out, cfg.conv2_out, kernel_size=5)
        self.conv2_gamma = nn.Parameter(torch.ones(cfg.conv2_out))
        self.conv2_beta = nn.Parameter(torch.zeros(cfg.conv2_out))
        self.fc1 = nn.Linear(cfg.conv2_out * 2 * 2, cfg.fc_hidden)
        self.fc2 = nn.Linear(cfg.fc_hidden, cfg.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.pool(h)
        h = self.pool1_gamma[None, :, None, None] * h + self.pool1_beta[None, :, None, None]
        h = torch.tanh(h)
        h = self.conv2(h)
        h = self.conv2_gamma[None, :, None, None] * h + self.conv2_beta[None, :, None, None]
        h = torch.tanh(h)
        h = h.reshape(h.shape[0], -1)
        h = torch.tanh(self.fc1(h))
        return self.fc2(h)


def torch_model_to_theta(model: TorchLeNet2) -> np.ndarray:
    pieces = [
        model.conv1.weight.detach().cpu().numpy().reshape(-1),
        model.conv1.bias.detach().cpu().numpy().reshape(-1),
        model.pool1_gamma.detach().cpu().numpy().reshape(-1),
        model.pool1_beta.detach().cpu().numpy().reshape(-1),
        model.conv2.weight.detach().cpu().numpy().reshape(-1),
        model.conv2.bias.detach().cpu().numpy().reshape(-1),
        model.conv2_gamma.detach().cpu().numpy().reshape(-1),
        model.conv2_beta.detach().cpu().numpy().reshape(-1),
        model.fc1.weight.detach().cpu().numpy().T.reshape(-1),
        model.fc1.bias.detach().cpu().numpy().reshape(-1),
        model.fc2.weight.detach().cpu().numpy().T.reshape(-1),
        model.fc2.bias.detach().cpu().numpy().reshape(-1),
    ]
    return np.concatenate(pieces).astype(np.float64)


def train_deterministic_lenet2(cfg: ModelConfig, tr_cfg: TrainingConfig, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray) -> Dict[str, object]:
    torch.manual_seed(tr_cfg.random_seed)
    device = torch.device(tr_cfg.device)
    model = TorchLeNet2(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=tr_cfg.learning_rate, weight_decay=tr_cfg.weight_decay)
    ds = TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    dl = DataLoader(ds, batch_size=tr_cfg.batch_size, shuffle=True)
    for _ in range(tr_cfg.epochs):
        model.train()
        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = nn.CrossEntropyLoss()(model(xb), yb)
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(x_test, dtype=torch.float32, device=device)).cpu().numpy()
    pred = logits.argmax(axis=1)
    acc = float(accuracy_score(y_test, pred))
    theta = torch_model_to_theta(model)
    result = {"theta_map": theta.tolist(), "metrics": {"accuracy": acc}, "num_params": int(theta.size)}
    if tr_cfg.checkpoint_json_path:
        p = Path(tr_cfg.checkpoint_json_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            json.dump({"result": result}, f, indent=2)
    return result
