"""Loss builders."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeneralizedCrossEntropyLoss(nn.Module):
    def __init__(self, q: float = 0.7) -> None:
        super().__init__()
        self.q = q

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1).clamp_min(1e-8)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        if self.q == 0:
            return (-torch.log(pt)).mean()
        loss = (1.0 - pt.pow(self.q)) / self.q
        return loss.mean()


class SymmetricCrossEntropyLoss(nn.Module):
    def __init__(self, alpha: float = 0.1, beta: float = 1.0, num_classes: int = 2) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets)
        probs = F.softmax(logits, dim=1).clamp_min(1e-7).clamp_max(1.0)
        one_hot = F.one_hot(targets, num_classes=self.num_classes).float().clamp_min(1e-4)
        rce = -(probs * torch.log(one_hot)).sum(dim=1).mean()
        return self.alpha * ce + self.beta * rce


def build_loss(name: str, label_smoothing: float = 0.0, num_classes: int = 2):
    name = name.lower()
    if name == "cross_entropy":
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    if name == "generalized_cross_entropy":
        return GeneralizedCrossEntropyLoss()
    if name == "symmetric_cross_entropy":
        return SymmetricCrossEntropyLoss(num_classes=num_classes)
    raise ValueError(f"Unsupported loss: {name}")
