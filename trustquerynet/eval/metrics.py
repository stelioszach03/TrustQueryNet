"""Evaluation metric bundle."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from trustquerynet.eval.calibration import expected_calibration_error, multiclass_brier_score
from trustquerynet.eval.selective import aurc_from_curve, risk_coverage_curve


def compute_all(y_true, probs, y_pred=None, thresholds=None) -> Dict[str, Any]:
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)
    y_pred = probs.argmax(axis=1) if y_pred is None else np.asarray(y_pred)

    metrics: Dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "ece": float(expected_calibration_error(y_true, probs)),
        "brier": float(multiclass_brier_score(y_true, probs)),
    }
    try:
        metrics["macro_auroc"] = float(roc_auc_score(y_true, probs, multi_class="ovr", average="macro"))
    except ValueError:
        metrics["macro_auroc"] = float("nan")

    if thresholds is not None:
        rc_df = risk_coverage_curve(y_true, probs, thresholds)
        metrics["aurc"] = float(aurc_from_curve(rc_df))
        metrics["coverage_at_0.5"] = float(rc_df.loc[(rc_df["threshold"] - 0.5).abs().idxmin(), "coverage"])
        metrics["risk_at_0.5"] = float(rc_df.loc[(rc_df["threshold"] - 0.5).abs().idxmin(), "selective_risk"])
    return metrics


def macro_f1_from_probs(y_true, probs) -> float:
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)
    return float(f1_score(y_true, probs.argmax(axis=1), average="macro"))


def accuracy_from_probs(y_true, probs) -> float:
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)
    return float(accuracy_score(y_true, probs.argmax(axis=1)))
