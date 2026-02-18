"""Statistical tests and lightweight utilities."""

from __future__ import annotations

from math import erf, sqrt
from typing import Callable, Dict

import numpy as np


def mcnemar_test(y_true, pred_a, pred_b) -> Dict[str, float]:
    y_true = np.asarray(y_true)
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)
    b = np.sum((pred_a == y_true) & (pred_b != y_true))
    c = np.sum((pred_a != y_true) & (pred_b == y_true))
    denom = b + c
    if denom == 0:
        return {"statistic": 0.0, "p_value": 1.0}
    statistic = ((abs(b - c) - 1) ** 2) / denom
    z = abs(b - c) / sqrt(denom)
    p_value = 2.0 * (1.0 - 0.5 * (1.0 + erf(z / sqrt(2.0))))
    return {"statistic": float(statistic), "p_value": float(p_value)}


def delong_auc_test(*args, **kwargs):
    raise NotImplementedError("DeLong AUC testing is intentionally left for a later slice.")


def bootstrap_metric_ci(
    y_true,
    probs,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    *,
    n_bootstrap: int = 200,
    seed: int = 42,
    confidence: float = 0.95,
) -> Dict[str, float]:
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)
    if len(y_true) == 0:
        raise ValueError("Cannot bootstrap an empty dataset.")
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_bootstrap):
        indices = rng.integers(0, len(y_true), size=len(y_true))
        values.append(float(metric_fn(y_true[indices], probs[indices])))
    alpha = (1.0 - confidence) / 2.0
    return {
        "mean": float(np.mean(values)),
        "lower": float(np.quantile(values, alpha)),
        "upper": float(np.quantile(values, 1.0 - alpha)),
    }


def bootstrap_metric_difference_ci(
    y_true,
    probs_a,
    probs_b,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    *,
    n_bootstrap: int = 200,
    seed: int = 42,
    confidence: float = 0.95,
) -> Dict[str, float]:
    y_true = np.asarray(y_true)
    probs_a = np.asarray(probs_a)
    probs_b = np.asarray(probs_b)
    if len(y_true) == 0:
        raise ValueError("Cannot bootstrap an empty dataset.")
    if len(probs_a) != len(y_true) or len(probs_b) != len(y_true):
        raise ValueError("Prediction arrays must match y_true length.")

    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_bootstrap):
        indices = rng.integers(0, len(y_true), size=len(y_true))
        diff = float(metric_fn(y_true[indices], probs_a[indices]) - metric_fn(y_true[indices], probs_b[indices]))
        values.append(diff)
    alpha = (1.0 - confidence) / 2.0
    return {
        "mean": float(np.mean(values)),
        "lower": float(np.quantile(values, alpha)),
        "upper": float(np.quantile(values, 1.0 - alpha)),
    }


def paired_permutation_test(
    values_a,
    values_b,
    *,
    n_permutations: int = 10000,
    seed: int = 42,
) -> Dict[str, float]:
    values_a = np.asarray(values_a, dtype=np.float64)
    values_b = np.asarray(values_b, dtype=np.float64)
    if values_a.shape != values_b.shape:
        raise ValueError("Paired samples must share the same shape.")
    if values_a.size == 0:
        raise ValueError("Cannot test empty paired samples.")

    differences = values_a - values_b
    observed = float(differences.mean())
    rng = np.random.default_rng(seed)
    permuted = np.empty(n_permutations, dtype=np.float64)
    for idx in range(n_permutations):
        signs = rng.choice(np.array([-1.0, 1.0]), size=differences.shape[0])
        permuted[idx] = float(np.mean(differences * signs))
    p_value = float((np.sum(np.abs(permuted) >= abs(observed)) + 1) / (n_permutations + 1))
    return {"statistic": observed, "p_value": p_value}
