"""Collect final paper figures from corrected TrustQueryNet run directories.

This script does two things:
1. Copies the final calibrated reliability and risk-coverage plots for a
   representative internal repair seed and a representative external repair seed.
2. Builds a compact summary figure from aggregate internal/external results for
   repair, no-repair, and random-repair.

The representative seed is selected as the one whose calibrated macro-F1 is
closest to the multiseed mean for that run.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", default="artifacts/final_evidence")
    parser.add_argument("--repair-run", required=True)
    parser.add_argument("--no-repair-run", required=True)
    parser.add_argument("--random-repair-run", required=True)
    parser.add_argument("--repair-external-run", required=True)
    parser.add_argument("--no-repair-external-run", required=True)
    parser.add_argument("--random-repair-external-run", required=True)
    parser.add_argument("--overlap-report", default=None, help="Optional overlap audit JSON to copy into the figure bundle.")
    parser.add_argument("--output-dir", default="artifacts/paper_figures")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def final_round_dir(seed_dir: Path) -> Path:
    round_dirs = sorted(path for path in seed_dir.glob("active_round_*") if path.is_dir())
    return round_dirs[-1] if round_dirs else seed_dir


def representative_internal_seed(run_dir: Path) -> tuple[Path, float]:
    aggregate = load_json(run_dir / "aggregate_results.json")
    target = float(aggregate["test_cal_macro_f1"]["mean"])
    candidates: list[tuple[float, Path, float]] = []
    for seed_dir in sorted(path for path in run_dir.glob("seed-*") if path.is_dir()):
        report_path = seed_dir / "active_learning_report.json"
        if not report_path.exists():
            continue
        report = load_json(report_path)
        score = float(report["final_metrics"]["test_calibrated"]["macro_f1"])
        candidates.append((abs(score - target), seed_dir, score))
    if not candidates:
        raise FileNotFoundError(f"No active_learning_report.json files found in {run_dir}")
    _, seed_dir, score = min(candidates, key=lambda item: item[0])
    return seed_dir, score


def representative_external_seed(run_dir: Path) -> tuple[Path, float]:
    aggregate = load_json(run_dir / "aggregate_results.json")
    target = float(aggregate["test_cal_macro_f1"]["mean"])
    candidates: list[tuple[float, Path, float]] = []
    for seed_dir in sorted(path for path in run_dir.glob("seed-*") if path.is_dir()):
        metrics_path = seed_dir / "external_metrics.json"
        if not metrics_path.exists():
            continue
        metrics = load_json(metrics_path)
        score = float(metrics["test_calibrated"]["macro_f1"])
        candidates.append((abs(score - target), seed_dir, score))
    if not candidates:
        raise FileNotFoundError(f"No external_metrics.json files found in {run_dir}")
    _, seed_dir, score = min(candidates, key=lambda item: item[0])
    return seed_dir, score


def copy_if_exists(source: Path, destination: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(f"Expected figure not found: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def metric_mean(run_dir: Path, key: str) -> float:
    aggregate = load_json(run_dir / "aggregate_results.json")
    return float(aggregate[key]["mean"])


def make_summary_figure(
    repair_dir: Path,
    no_repair_dir: Path,
    random_dir: Path,
    repair_external_dir: Path,
    no_repair_external_dir: Path,
    random_external_dir: Path,
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    labels = ["Repair", "No repair", "Random repair"]
    internal_dirs = [repair_dir, no_repair_dir, random_dir]
    external_dirs = [repair_external_dir, no_repair_external_dir, random_external_dir]

    metric_specs = [
        ("test_cal_macro_f1", "Macro-F1"),
        ("test_cal_accuracy", "Accuracy"),
        ("test_cal_macro_auroc", "Macro-AUROC"),
        ("test_cal_ece", "ECE"),
    ]

    x = np.arange(len(labels))
    width = 0.35

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.flatten()

    for ax, (metric_key, title) in zip(axes, metric_specs):
        internal = [metric_mean(run_dir, metric_key) for run_dir in internal_dirs]
        external = [metric_mean(run_dir, metric_key) for run_dir in external_dirs]
        ax.bar(x - width / 2, internal, width, label="Internal", color="#2a6f97")
        ax.bar(x + width / 2, external, width, label="External", color="#c96c50")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=12)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        if metric_key == "test_cal_ece":
            ax.set_ylabel("Lower is better")
        else:
            ax.set_ylabel("Higher is better")

    axes[0].legend(frameon=False)
    fig.suptitle("TrustQueryNet Final Internal vs External Summary", fontsize=14)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()

    runs_root = Path(args.runs_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    repair_dir = (runs_root / args.repair_run).resolve()
    no_repair_dir = (runs_root / args.no_repair_run).resolve()
    random_dir = (runs_root / args.random_repair_run).resolve()
    repair_external_dir = (runs_root / args.repair_external_run).resolve()
    no_repair_external_dir = (runs_root / args.no_repair_external_run).resolve()
    random_external_dir = (runs_root / args.random_repair_external_run).resolve()

    internal_seed_dir, internal_score = representative_internal_seed(repair_dir)
    external_seed_dir, external_score = representative_external_seed(repair_external_dir)

    internal_final_dir = final_round_dir(internal_seed_dir)
    internal_plots_dir = internal_final_dir / "plots"
    external_plots_dir = external_seed_dir / "plots"

    copy_if_exists(internal_plots_dir / "reliability_calibrated.png", output_dir / "internal_reliability_calibrated.png")
    copy_if_exists(internal_plots_dir / "risk_coverage_calibrated.png", output_dir / "internal_risk_coverage_calibrated.png")
    copy_if_exists(external_plots_dir / "reliability_calibrated.png", output_dir / "external_reliability_calibrated.png")
    copy_if_exists(external_plots_dir / "risk_coverage_calibrated.png", output_dir / "external_risk_coverage_calibrated.png")

    make_summary_figure(
        repair_dir=repair_dir,
        no_repair_dir=no_repair_dir,
        random_dir=random_dir,
        repair_external_dir=repair_external_dir,
        no_repair_external_dir=no_repair_external_dir,
        random_external_dir=random_external_dir,
        output_path=output_dir / "internal_external_summary.png",
    )

    if args.overlap_report:
        overlap_path = Path(args.overlap_report).resolve()
        if overlap_path.exists():
            copy_if_exists(overlap_path, output_dir / "overlap_report.json")

    manifest = {
        "internal_representative_seed": internal_seed_dir.name,
        "internal_representative_macro_f1": internal_score,
        "internal_plot_source": str(internal_final_dir),
        "external_representative_seed": external_seed_dir.name,
        "external_representative_macro_f1": external_score,
        "external_plot_source": str(external_seed_dir),
        "files": sorted(path.name for path in output_dir.iterdir() if path.is_file()),
    }
    with (output_dir / "figure_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "internal_representative_seed": internal_seed_dir.name,
                "external_representative_seed": external_seed_dir.name,
                "files": manifest["files"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
