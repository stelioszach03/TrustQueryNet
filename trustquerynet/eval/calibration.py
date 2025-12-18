"""Calibration metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def expected_calibration_error(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 15) -> float:
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == y_true).astype(np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lower, upper in zip(bins[:-1], bins[1:]):
        mask = (confidences > lower) & (confidences <= upper)
        if not np.any(mask):
            continue
        bucket_confidence = confidences[mask].mean()
        bucket_accuracy = accuracies[mask].mean()
        ece += mask.mean() * abs(bucket_confidence - bucket_accuracy)
    return float(ece)


def multiclass_brier_score(y_true: np.ndarray, probs: np.ndarray) -> float:
    targets = np.eye(probs.shape[1])[y_true]
    return float(np.mean(np.sum((probs - targets) ** 2, axis=1)))


def reliability_bins(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 15) -> pd.DataFrame:
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == y_true).astype(np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for lower, upper in zip(bins[:-1], bins[1:]):
        mask = (confidences > lower) & (confidences <= upper)
        if np.any(mask):
            rows.append(
                {
                    "bin_start": lower,
                    "bin_end": upper,
                    "confidence": float(confidences[mask].mean()),
                    "accuracy": float(accuracies[mask].mean()),
                    "count": int(mask.sum()),
                }
            )
    return pd.DataFrame(rows)
