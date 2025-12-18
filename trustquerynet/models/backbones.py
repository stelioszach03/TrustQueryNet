"""Backbone creation through timm."""

from __future__ import annotations

import timm
import torch
import torch.nn as nn


def create_backbone(name: str, pretrained: bool, num_classes: int, img_size: int) -> nn.Module:
    try:
        return timm.create_model(name, pretrained=pretrained, num_classes=num_classes, img_size=img_size)
    except TypeError:
        return timm.create_model(name, pretrained=pretrained, num_classes=num_classes)


def _flatten_embeddings(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim <= 2:
        return tensor
    reduce_dims = tuple(range(2, tensor.ndim))
    return tensor.mean(dim=reduce_dims)


def forward_with_embeddings(model: nn.Module, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if hasattr(model, "forward_features") and hasattr(model, "forward_head"):
        features = model.forward_features(images)
        try:
            embeddings = model.forward_head(features, pre_logits=True)
        except TypeError:
            embeddings = features
        logits = model.forward_head(features)
        return _flatten_embeddings(logits), _flatten_embeddings(embeddings)

    logits = model(images)
    return logits, logits
