"""Selective prediction metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def risk_coverage_curve(y_true, probs, thresholds) -> pd.DataFrame:
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    rows = []
    for threshold in thresholds:
        keep_mask = confidences >= threshold
        coverage = float(keep_mask.mean())
        if keep_mask.sum() == 0:
            risk = 0.0
            accuracy = 0.0
        else:
            accuracy = float((predictions[keep_mask] == y_true[keep_mask]).mean())
            risk = float(1.0 - accuracy)
        rows.append({"threshold": float(threshold), "coverage": coverage, "selective_risk": risk, "accuracy": accuracy})
    return pd.DataFrame(rows)


def aurc_from_curve(rc_df: pd.DataFrame) -> float:
    if rc_df.empty:
        return 0.0
    curve = rc_df[["coverage", "selective_risk"]].sort_values("coverage")
    coverage = curve["coverage"].to_numpy(dtype=np.float64)
    risk = curve["selective_risk"].to_numpy(dtype=np.float64)
    if len(coverage) == 1:
        return float(risk[0] * coverage[0])
    return float(np.trapezoid(risk, coverage))


def default_threshold_grid(num: int = 101) -> np.ndarray:
    return np.linspace(0.0, 1.0, num=num)
