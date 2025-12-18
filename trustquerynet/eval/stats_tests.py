"""Statistical test placeholders and lightweight utilities."""

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
