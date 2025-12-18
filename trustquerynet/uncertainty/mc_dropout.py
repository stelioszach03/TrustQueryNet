"""MC Dropout helpers."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn

from trustquerynet.models.backbones import forward_with_embeddings


def _enable_dropout(module: nn.Module) -> None:
    if isinstance(
        module,
        (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout, nn.FeatureAlphaDropout),
    ):
        module.train()


@torch.no_grad()
def predict_mc_dropout(model, loader, device: torch.device, num_samples: int) -> Dict[str, Any]:
    was_training = model.training
    model.eval()
    model.apply(_enable_dropout)

    sample_prob_list = []
    labels = None
    indices = None
    mean_embeddings = None

    for _ in range(num_samples):
        probs_pass = []
        embeddings_pass = []
        labels_pass = []
        indices_pass = []
        for batch in loader:
            images = batch["image"].to(device)
            logits, embeddings = forward_with_embeddings(model, images)
            probs = torch.softmax(logits, dim=1)
            probs_pass.append(probs.cpu())
            embeddings_pass.append(embeddings.cpu())
            labels_pass.append(batch["y_clean"].cpu())
            indices_pass.append(batch["index"].cpu())

        sample_prob_list.append(torch.cat(probs_pass).numpy())
        if mean_embeddings is None:
            mean_embeddings = torch.cat(embeddings_pass).numpy()
            labels = torch.cat(labels_pass).numpy()
            indices = torch.cat(indices_pass).numpy()

    if was_training:
        model.train()
    else:
        model.eval()

    stacked = np.stack(sample_prob_list, axis=0)
    return {
        "samples": stacked,
        "mean_probs": stacked.mean(axis=0),
        "labels": labels,
        "indices": indices,
        "embeddings": mean_embeddings,
    }
