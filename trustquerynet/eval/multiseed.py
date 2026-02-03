"""Helpers for aggregating multi-seed experiment results."""

from __future__ import annotations

from typing import Any, Dict, Iterable

import numpy as np


SUMMARY_METRIC_KEYS = [
    "realized_flip_rate",
    "best_val_accuracy",
    "best_val_macro_f1",
    "best_val_ece",
    "best_val_macro_auroc",
    "test_uncal_accuracy",
    "test_uncal_macro_f1",
    "test_uncal_ece",
    "test_uncal_macro_auroc",
    "test_cal_accuracy",
    "test_cal_macro_f1",
    "test_cal_ece",
    "test_cal_macro_auroc",
    "test_cal_coverage_at_0_5",
    "test_cal_risk_at_0_5",
]


def best_history_entry(history: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not history:
        return None
    return max(history, key=lambda item: item.get("val", {}).get("macro_f1", float("-inf")))


def make_seed_summary_row(
    *,
    run_name: str,
    seed: int,
    output_dir: str,
    final_metrics: dict[str, Any],
    run_type: str,
    rounds: int,
    initial_trusted_count: int = 0,
    queried_count: int = 0,
) -> dict[str, Any]:
    history = final_metrics.get("history", [])
    best_entry = best_history_entry(history)
    best_val = (best_entry or {}).get("val", {})
    test_uncal = final_metrics.get("test_uncalibrated", {})
    test_cal = final_metrics.get("test_calibrated", {})
    noise = final_metrics.get("noise", {})

    return {
        "run_name": run_name,
        "seed": int(seed),
        "output_dir": output_dir,
        "run_type": run_type,
        "rounds": int(rounds),
        "initial_trusted_count": int(initial_trusted_count),
        "queried_count": int(queried_count),
        "device": final_metrics.get("device"),
        "noise_type": noise.get("noise_type"),
        "realized_flip_rate": noise.get("realized_flip_rate"),
        "best_val_accuracy": best_val.get("accuracy"),
        "best_val_macro_f1": best_val.get("macro_f1"),
        "best_val_ece": best_val.get("ece"),
        "best_val_macro_auroc": best_val.get("macro_auroc"),
        "test_uncal_accuracy": test_uncal.get("accuracy"),
        "test_uncal_macro_f1": test_uncal.get("macro_f1"),
        "test_uncal_ece": test_uncal.get("ece"),
        "test_uncal_macro_auroc": test_uncal.get("macro_auroc"),
        "test_cal_accuracy": test_cal.get("accuracy"),
        "test_cal_macro_f1": test_cal.get("macro_f1"),
        "test_cal_ece": test_cal.get("ece"),
        "test_cal_macro_auroc": test_cal.get("macro_auroc"),
        "test_cal_coverage_at_0_5": test_cal.get("coverage_at_0.5"),
        "test_cal_risk_at_0_5": test_cal.get("risk_at_0.5"),
    }


def aggregate_summary_rows(
    rows: Iterable[dict[str, Any]],
    *,
    metric_keys: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    rows = list(rows)
    metric_keys = metric_keys or SUMMARY_METRIC_KEYS
    aggregates: dict[str, dict[str, float]] = {}

    for key in metric_keys:
        values = []
        for row in rows:
            value = row.get(key)
            if value is None:
                continue
            value = float(value)
            if not np.isfinite(value):
                continue
            values.append(value)
        if not values:
            continue
        array = np.asarray(values, dtype=np.float64)
        aggregates[key] = {
            "n": float(len(array)),
            "mean": float(array.mean()),
            "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
            "min": float(array.min()),
            "max": float(array.max()),
        }

    return aggregates
