"""Temperature scaling."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemperatureScaler(nn.Module):
    def __init__(self, temperature: float = 1.0) -> None:
        super().__init__()
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(float(temperature))))

    @property
    def temperature(self) -> torch.Tensor:
        return torch.exp(self.log_temperature)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature.clamp_min(1e-6)

    def predict_proba(self, logits: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.forward(logits), dim=1)


def fit_temperature(logits_val, y_val, max_iter: int = 100) -> TemperatureScaler:
    logits = torch.as_tensor(logits_val, dtype=torch.float32)
    targets = torch.as_tensor(y_val, dtype=torch.long)
    scaler = TemperatureScaler()
    optimizer = torch.optim.LBFGS([scaler.log_temperature], lr=0.1, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(scaler(logits), targets)
        loss.backward()
        return loss

    optimizer.step(closure)
    return scaler.eval()
