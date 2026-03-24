from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
from src.qbnn.config import ModelConfig, TrainingConfig


class TorchLeNet2(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(cfg.num_channels, cfg.fc_hidden)
        self.fc2 = nn.Linear(cfg.fc_hidden, cfg.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        h = self.fc1(x)
        h = self.fc2(h)
        return h


def torch_model_to_theta(model: TorchLeNet2) -> np.ndarray:
    pieces = [
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
            yb_predict = model(xb)
            loss = nn.CrossEntropyLoss()(yb_predict, yb)
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
