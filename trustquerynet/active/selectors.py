"""Selector implementations."""

from __future__ import annotations

import numpy as np


def core_set_scores(embeddings: np.ndarray, selected_mask: np.ndarray | None = None) -> np.ndarray:
    embeddings = np.asarray(embeddings, dtype=np.float64)
    if selected_mask is None or not np.any(selected_mask):
        center = embeddings.mean(axis=0, keepdims=True)
        return np.linalg.norm(embeddings - center, axis=1)
    selected = embeddings[selected_mask]
    distances = np.linalg.norm(embeddings[:, None, :] - selected[None, :, :], axis=2)
    return distances.min(axis=1)


def kcenter_greedy_select(embeddings: np.ndarray, selected_mask: np.ndarray, budget: int) -> np.ndarray:
    selected_mask = selected_mask.astype(bool).copy()
    chosen = []
    for _ in range(budget):
        scores = core_set_scores(embeddings, selected_mask=selected_mask)
        scores[selected_mask] = -np.inf
        next_idx = int(np.argmax(scores))
        selected_mask[next_idx] = True
        chosen.append(next_idx)
    return np.asarray(chosen, dtype=np.int64)


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    min_score = np.min(scores)
    max_score = np.max(scores)
    if max_score - min_score < 1e-12:
        return np.zeros_like(scores)
    return (scores - min_score) / (max_score - min_score)
