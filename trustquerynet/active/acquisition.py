"""Acquisition score computation."""

from __future__ import annotations

import numpy as np

from trustquerynet.active.selectors import core_set_scores, kcenter_greedy_select, normalize_scores


def compute_scores(method, probs_or_samples, embeddings=None) -> np.ndarray:
    method = method.lower()
    if method in {"least_confidence", "uncertainty"}:
        probs = np.asarray(probs_or_samples, dtype=np.float64)
        return 1.0 - probs.max(axis=1)
    if method == "entropy":
        probs = np.asarray(probs_or_samples, dtype=np.float64).clip(1e-8, 1.0)
        return -(probs * np.log(probs)).sum(axis=1)
    if method == "margin":
        probs = np.sort(np.asarray(probs_or_samples, dtype=np.float64), axis=1)
        return 1.0 - (probs[:, -1] - probs[:, -2])
    if method == "bald":
        samples = np.asarray(probs_or_samples, dtype=np.float64).clip(1e-8, 1.0)
        mean_probs = samples.mean(axis=0)
        predictive_entropy = -(mean_probs * np.log(mean_probs)).sum(axis=1)
        expected_entropy = -(samples * np.log(samples)).sum(axis=2).mean(axis=0)
        return predictive_entropy - expected_entropy
    if method == "core_set":
        if embeddings is None:
            raise ValueError("Core-set scoring requires embeddings.")
        return core_set_scores(np.asarray(embeddings, dtype=np.float64))
    raise ValueError(f"Unsupported acquisition method: {method}")


def select_query_indices(
    method: str,
    budget: int,
    *,
    probs: np.ndarray | None = None,
    samples: np.ndarray | None = None,
    embeddings: np.ndarray | None = None,
    selected_mask: np.ndarray | None = None,
    shortlist_factor: int = 4,
) -> np.ndarray:
    method = method.lower()
    if budget <= 0:
        return np.asarray([], dtype=np.int64)

    total = None
    if probs is not None:
        total = np.asarray(probs).shape[0]
    elif samples is not None:
        total = np.asarray(samples).shape[1]
    elif embeddings is not None:
        total = np.asarray(embeddings).shape[0]
    elif selected_mask is not None:
        total = np.asarray(selected_mask).shape[0]
    if total is None:
        raise ValueError("Could not infer pool size for query selection.")

    if selected_mask is None:
        selected_mask = np.zeros(total, dtype=bool)
    else:
        selected_mask = np.asarray(selected_mask, dtype=bool)
    candidate_indices = np.where(~selected_mask)[0]
    if len(candidate_indices) == 0:
        return np.asarray([], dtype=np.int64)
    budget = min(budget, len(candidate_indices))

    if method == "core_set":
        if embeddings is None:
            raise ValueError("Core-set selection requires embeddings.")
        return kcenter_greedy_select(np.asarray(embeddings, dtype=np.float64), selected_mask=selected_mask, budget=budget)

    if method == "hybrid":
        if probs is None or embeddings is None:
            raise ValueError("Hybrid selection requires probabilities and embeddings.")
        uncertainty_scores = compute_scores("entropy", probs)
        shortlist_size = min(len(candidate_indices), max(budget, budget * max(shortlist_factor, 1)))
        shortlist_local = np.argsort(uncertainty_scores[candidate_indices])[-shortlist_size:]
        shortlist_indices = candidate_indices[shortlist_local]
        shortlist_embeddings = np.asarray(embeddings, dtype=np.float64)[shortlist_indices]
        chosen_local = kcenter_greedy_select(
            shortlist_embeddings,
            selected_mask=np.zeros(len(shortlist_indices), dtype=bool),
            budget=budget,
        )
        return shortlist_indices[chosen_local]

    if method == "bald":
        if samples is None:
            raise ValueError("BALD selection requires MC dropout or ensemble samples.")
        scores = compute_scores("bald", samples)
    else:
        if probs is None:
            raise ValueError(f"{method} selection requires class probabilities.")
        scores = compute_scores(method, probs, embeddings=embeddings)

    ranked_local = np.argsort(scores[candidate_indices])[-budget:]
    return candidate_indices[ranked_local]
