"""Package final TrustQueryNet artifacts into a small export bundle.

The bundle includes:
- summary.json
- results_table.csv
- results_table.md
- selected metrics/config/plot files from the requested run directories

By default, checkpoints are excluded to keep the ZIP small enough for a direct
Colab browser download.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-root",
        default="artifacts/runs",
        help="Directory containing run output folders.",
    )
    parser.add_argument(
        "--run",
        dest="runs",
        action="append",
        required=True,
        help="Run folder name relative to --runs-root. Pass multiple times.",
    )
    parser.add_argument(
        "--output-root",
        default="artifacts/exports",
        help="Directory where the export bundle will be created.",
    )
    parser.add_argument(
        "--bundle-name",
        default="trustquerynet-final-artifacts",
        help="Name for the export directory and ZIP file.",
    )
    parser.add_argument(
        "--include-checkpoints",
        action="store_true",
        help="Include .ckpt files in the bundle.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Attempt a direct browser download when running in Colab.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sanitize_value(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {key: sanitize_value(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [sanitize_value(inner) for inner in value]
    return value


def best_history_entry(history: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not history:
        return None
    return max(history, key=lambda item: item.get("val", {}).get("macro_f1", float("-inf")))


def final_round_dir(run_dir: Path) -> Path:
    round_dirs = sorted(path for path in run_dir.glob("active_round_*") if path.is_dir())
    return round_dirs[-1] if round_dirs else run_dir


def copy_file_if_exists(source: Path, destination: Path) -> None:
    if not source.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def copy_directory_if_exists(source: Path, destination: Path) -> None:
    if not source.exists():
        return
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)


def gather_run_summary(run_dir: Path) -> dict[str, Any]:
    active_report_path = run_dir / "active_learning_report.json"
    if active_report_path.exists():
        report = load_json(active_report_path)
        final_metrics = report["final_metrics"]
        rounds = len(report.get("round_metrics", []))
        initial_trusted = len(report.get("initial_trusted_indices", []))
        queried = sum(len(chunk) for chunk in report.get("selected_indices_by_round", []))
        run_type = "active_learning"
    else:
        report = None
        final_metrics = load_json(run_dir / "metrics.json")
        rounds = 1
        initial_trusted = 0
        queried = 0
        run_type = "single_run"

    history = final_metrics.get("history", [])
    best_entry = best_history_entry(history)
    best_val = (best_entry or {}).get("val", {})
    test_uncal = final_metrics.get("test_uncalibrated", {})
    test_cal = final_metrics.get("test_calibrated", {})
    noise = final_metrics.get("noise", {})

    summary_row = {
        "run_name": run_dir.name,
        "run_type": run_type,
        "rounds": rounds,
        "initial_trusted_count": initial_trusted,
        "queried_count": queried,
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

    return {
        "run_dir": str(run_dir),
        "run_name": run_dir.name,
        "run_type": run_type,
        "final_round_dir": str(final_round_dir(run_dir)),
        "summary_row": sanitize_value(summary_row),
        "final_metrics": sanitize_value(final_metrics),
        "active_learning_report": sanitize_value(report),
    }


def copy_run_artifacts(run_dir: Path, target_dir: Path, include_checkpoints: bool) -> None:
    active_report_path = run_dir / "active_learning_report.json"
    if active_report_path.exists():
        copy_file_if_exists(active_report_path, target_dir / "active_learning_report.json")
        source_dirs = sorted(path for path in run_dir.glob("active_round_*") if path.is_dir())
    else:
        source_dirs = [run_dir]

    keep_files = {
        "metrics.json",
        "noise_manifest.json",
        "run_config.yaml",
        "splits.csv",
    }

    for source_dir in source_dirs:
        relative = source_dir.relative_to(run_dir) if source_dir != run_dir else Path(".")
        destination_base = target_dir / relative
        for name in keep_files:
            copy_file_if_exists(source_dir / name, destination_base / name)
        copy_directory_if_exists(source_dir / "plots", destination_base / "plots")

        if include_checkpoints:
            for checkpoint_path in source_dir.glob("*.ckpt"):
                copy_file_if_exists(checkpoint_path, destination_base / checkpoint_path.name)


def write_results_table(rows: list[dict[str, Any]], output_dir: Path) -> None:
    if not rows:
        return

    fieldnames = list(rows[0].keys())
    csv_path = output_dir / "results_table.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    md_path = output_dir / "results_table.md"
    markdown_columns = [
        "run_name",
        "best_val_accuracy",
        "best_val_macro_f1",
        "best_val_ece",
        "test_cal_accuracy",
        "test_cal_macro_f1",
        "test_cal_ece",
        "test_cal_macro_auroc",
        "test_cal_coverage_at_0_5",
        "test_cal_risk_at_0_5",
    ]
    with md_path.open("w", encoding="utf-8") as handle:
        handle.write("| " + " | ".join(markdown_columns) + " |\n")
        handle.write("| " + " | ".join(["---"] * len(markdown_columns)) + " |\n")
        for row in rows:
            values = []
            for column in markdown_columns:
                value = row.get(column)
                if isinstance(value, float):
                    values.append(f"{value:.4f}")
                elif value is None:
                    values.append("")
                else:
                    values.append(str(value))
            handle.write("| " + " | ".join(values) + " |\n")


def maybe_trigger_download(path: Path) -> None:
    try:
        from google.colab import files  # type: ignore
    except Exception:
        return
    try:
        files.download(str(path))
    except Exception as exc:
        print(
            "Direct notebook download was skipped. "
            "Run this in a separate Colab cell instead:\n"
            "from google.colab import files\n"
            f"files.download({str(path)!r})"
        )
        print(f"download_error={exc}")


def main() -> None:
    args = parse_args()
    runs_root = Path(args.runs_root).resolve()
    output_root = Path(args.output_root).resolve()
    bundle_dir = output_root / args.bundle_name

    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    summary_rows = []

    for run_name in args.runs:
        run_dir = runs_root / run_name
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")
        summary = gather_run_summary(run_dir)
        summaries.append(summary)
        summary_rows.append(summary["summary_row"])
        copy_run_artifacts(run_dir, bundle_dir / "runs" / run_name, include_checkpoints=args.include_checkpoints)

    write_results_table(summary_rows, bundle_dir)
    with (bundle_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summaries, handle, indent=2)

    archive_base = output_root / args.bundle_name
    zip_path = Path(shutil.make_archive(str(archive_base), "zip", root_dir=bundle_dir))

    print(json.dumps(
        {
            "bundle_dir": str(bundle_dir),
            "zip_path": str(zip_path),
            "runs": args.runs,
            "included_checkpoints": args.include_checkpoints,
        },
        indent=2,
    ))

    if args.download:
        maybe_trigger_download(zip_path)


if __name__ == "__main__":
    main()
