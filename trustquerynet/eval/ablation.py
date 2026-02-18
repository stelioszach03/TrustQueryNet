"""Utilities for building ablation comparison tables."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def format_mean_std(metric: dict[str, float] | None, *, precision: int = 4) -> str:
    if not metric:
        return ""
    mean = metric.get("mean")
    std = metric.get("std")
    if mean is None or std is None:
        return ""
    if not math.isfinite(float(mean)) or not math.isfinite(float(std)):
        return ""
    return f"{float(mean):.{precision}f} ± {float(std):.{precision}f}"


def summarize_multiseed_run(run_dir: str | Path, *, label: str | None = None) -> dict[str, Any]:
    run_dir = Path(run_dir)
    aggregate_path = run_dir / "aggregate_results.json"
    if not aggregate_path.exists():
        raise FileNotFoundError(f"Expected aggregate_results.json in {run_dir}")

    aggregate = load_json(aggregate_path)
    manifest = load_json(run_dir / "multiseed_manifest.json") if (run_dir / "multiseed_manifest.json").exists() else {}

    uncal_ece = aggregate.get("test_uncal_ece")
    cal_ece = aggregate.get("test_cal_ece")
    ece_improvement = None
    if uncal_ece and cal_ece:
        ece_improvement = float(uncal_ece["mean"]) - float(cal_ece["mean"])

    row = {
        "label": label or run_dir.name,
        "run_dir": str(run_dir),
        "n_seeds": int(len(manifest.get("seeds", []))) if manifest.get("seeds") else None,
        "test_cal_accuracy": format_mean_std(aggregate.get("test_cal_accuracy")),
        "test_cal_macro_f1": format_mean_std(aggregate.get("test_cal_macro_f1")),
        "test_cal_macro_auroc": format_mean_std(aggregate.get("test_cal_macro_auroc")),
        "test_cal_aurc": format_mean_std(aggregate.get("test_cal_aurc")),
        "test_uncal_ece": format_mean_std(aggregate.get("test_uncal_ece")),
        "test_cal_ece": format_mean_std(aggregate.get("test_cal_ece")),
        "ece_improvement": f"{ece_improvement:.4f}" if ece_improvement is not None else "",
        "test_cal_coverage_at_0_5": format_mean_std(aggregate.get("test_cal_coverage_at_0_5")),
        "test_cal_risk_at_0_5": format_mean_std(aggregate.get("test_cal_risk_at_0_5")),
        "test_cal_accuracy_mean": aggregate.get("test_cal_accuracy", {}).get("mean"),
        "test_cal_macro_f1_mean": aggregate.get("test_cal_macro_f1", {}).get("mean"),
        "test_cal_macro_auroc_mean": aggregate.get("test_cal_macro_auroc", {}).get("mean"),
        "test_cal_aurc_mean": aggregate.get("test_cal_aurc", {}).get("mean"),
        "test_uncal_ece_mean": aggregate.get("test_uncal_ece", {}).get("mean"),
        "test_cal_ece_mean": aggregate.get("test_cal_ece", {}).get("mean"),
    }
    return row
