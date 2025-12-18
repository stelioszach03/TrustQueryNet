"""Plot helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from trustquerynet.eval.calibration import reliability_bins


def save_reliability_diagram(path: str | Path, y_true, probs, title: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bins = reliability_bins(y_true, probs)
    plt.figure(figsize=(6, 6))
    if not bins.empty:
        bin_centers = (bins["bin_start"] + bins["bin_end"]) / 2.0
        plt.bar(bin_centers, bins["accuracy"], width=0.06, alpha=0.7, label="Accuracy")
        plt.plot(bin_centers, bins["confidence"], marker="o", label="Confidence")
    plt.plot([0, 1], [0, 1], linestyle="--", color="black", label="Perfect calibration")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_risk_coverage_plot(path: str | Path, rc_df, title: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 5))
    plt.plot(rc_df["coverage"], rc_df["selective_risk"], marker="o")
    plt.xlabel("Coverage")
    plt.ylabel("Selective risk")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
