"""Checkpoint helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch


def save_checkpoint(path: str | Path, model, optimizer, epoch: int, extra: Dict[str, Any] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "extra": extra or {},
    }
    torch.save(payload, path)


def load_checkpoint(path: str | Path, model, optimizer=None, *, map_location=None) -> Dict[str, Any]:
    path = Path(path)
    payload = torch.load(path, map_location=map_location)
    model.load_state_dict(payload["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    return payload


def checkpoint_name_for_policy(policy: str) -> str:
    policy = policy.lower()
    mapping = {
        "last": "last.ckpt",
        "best_val_loss": "best_val_loss.ckpt",
        "best_val_macro_f1": "best_val_macro_f1.ckpt",
        "best_val_ece": "best_val_ece.ckpt",
    }
    if policy not in mapping:
        raise ValueError(f"Unsupported checkpoint policy: {policy}")
    return mapping[policy]
