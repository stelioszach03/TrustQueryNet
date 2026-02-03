"""Run a config across multiple seeds and aggregate the resulting metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
from copy import deepcopy
from pathlib import Path
import sys
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a base YAML config file.")
    parser.add_argument(
        "--seeds",
        required=True,
        nargs="+",
        type=int,
        help="Seed list to run, e.g. --seeds 42 52 62 72 82",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output root for the multi-seed run. Defaults to <config output_dir>-multiseed.",
    )
    parser.add_argument(
        "--resume-existing",
        action="store_true",
        help="Skip seeds that already have completed metrics in their per-seed output directory.",
    )
    return parser.parse_args()


def _default_multiseed_output_dir(cfg: dict[str, Any]) -> Path:
    base_output_dir = Path(cfg["output_dir"])
    return base_output_dir.parent / f"{base_output_dir.name}-multiseed"


def _seed_output_dir(root: Path, seed: int) -> Path:
    return root / f"seed-{seed}"


def _load_existing_result(seed_dir: Path, active_learning_enabled: bool) -> tuple[dict[str, Any], str, int, int, int] | None:
    if active_learning_enabled:
        report_path = seed_dir / "active_learning_report.json"
        if not report_path.exists():
            return None
        with report_path.open("r", encoding="utf-8") as handle:
            report = json.load(handle)
        final_metrics = report["final_metrics"]
        rounds = len(report.get("round_metrics", []))
        initial_trusted_count = len(report.get("initial_trusted_indices", []))
        queried_count = sum(len(chunk) for chunk in report.get("selected_indices_by_round", []))
        return final_metrics, "active_learning", rounds, initial_trusted_count, queried_count

    metrics_path = seed_dir / "metrics.json"
    if not metrics_path.exists():
        return None
    with metrics_path.open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    return metrics, "single_run", 1, 0, 0


def _run_one_seed(cfg: dict[str, Any]) -> tuple[dict[str, Any], str, int, int, int]:
    from trustquerynet.active.loop import run_active_learning
    from trustquerynet.training.trainer import train_one_run

    if cfg.get("active_learning", {}).get("enabled", False):
        report = run_active_learning(cfg)
        rounds = len(report.round_metrics)
        initial_trusted_count = len(report.initial_trusted_indices)
        queried_count = sum(len(chunk) for chunk in report.selected_indices_by_round)
        return report.final_metrics, "active_learning", rounds, initial_trusted_count, queried_count

    artifacts = train_one_run(cfg)
    return artifacts.metrics, "single_run", 1, 0, 0


def _write_seed_table(rows: list[dict[str, Any]], output_dir: Path) -> None:
    fieldnames = list(rows[0].keys())
    csv_path = output_dir / "seed_results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    md_columns = [
        "seed",
        "best_val_macro_f1",
        "test_cal_accuracy",
        "test_cal_macro_f1",
        "test_cal_ece",
        "test_cal_macro_auroc",
        "test_cal_coverage_at_0_5",
        "test_cal_risk_at_0_5",
    ]
    md_path = output_dir / "seed_results.md"
    with md_path.open("w", encoding="utf-8") as handle:
        handle.write("| " + " | ".join(md_columns) + " |\n")
        handle.write("| " + " | ".join(["---"] * len(md_columns)) + " |\n")
        for row in rows:
            values = []
            for column in md_columns:
                value = row.get(column)
                if isinstance(value, float) and math.isfinite(value):
                    values.append(f"{value:.4f}")
                elif isinstance(value, float):
                    values.append("")
                else:
                    values.append(str(value))
            handle.write("| " + " | ".join(values) + " |\n")


def _write_aggregate_table(aggregates: dict[str, dict[str, float]], output_dir: Path) -> None:
    from trustquerynet.eval.multiseed import SUMMARY_METRIC_KEYS

    csv_path = output_dir / "aggregate_results.csv"
    fieldnames = ["metric", "n", "mean", "std", "min", "max"]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for metric_name in SUMMARY_METRIC_KEYS:
            if metric_name not in aggregates:
                continue
            row = {"metric": metric_name, **aggregates[metric_name]}
            writer.writerow(row)

    focus_metrics = [
        "best_val_macro_f1",
        "test_cal_accuracy",
        "test_cal_macro_f1",
        "test_cal_ece",
        "test_cal_macro_auroc",
        "test_cal_coverage_at_0_5",
        "test_cal_risk_at_0_5",
    ]
    md_path = output_dir / "aggregate_results.md"
    with md_path.open("w", encoding="utf-8") as handle:
        handle.write("| Metric | Mean | Std | Min | Max |\n")
        handle.write("| --- | ---: | ---: | ---: | ---: |\n")
        for metric_name in focus_metrics:
            if metric_name not in aggregates:
                continue
            metric = aggregates[metric_name]
            handle.write(
                f"| `{metric_name}` | {metric['mean']:.4f} | {metric['std']:.4f} | "
                f"{metric['min']:.4f} | {metric['max']:.4f} |\n"
            )


def main() -> None:
    from trustquerynet.config.schema import load_config
    from trustquerynet.eval.multiseed import aggregate_summary_rows, make_seed_summary_row

    args = parse_args()
    base_cfg = load_config(args.config)
    output_root = Path(args.output_dir) if args.output_dir else _default_multiseed_output_dir(base_cfg)
    output_root.mkdir(parents=True, exist_ok=True)

    seed_rows = []
    for seed in args.seeds:
        seed_cfg = deepcopy(base_cfg)
        seed_cfg["seed"] = int(seed)
        seed_cfg["experiment_name"] = f"{base_cfg.get('experiment_name', 'experiment')}-seed-{seed}"
        seed_dir = _seed_output_dir(output_root, seed)
        seed_cfg["output_dir"] = str(seed_dir)

        existing = None
        if args.resume_existing:
            existing = _load_existing_result(
                seed_dir=seed_dir,
                active_learning_enabled=seed_cfg.get("active_learning", {}).get("enabled", False),
            )

        if existing is not None:
            final_metrics, run_type, rounds, initial_trusted_count, queried_count = existing
        else:
            final_metrics, run_type, rounds, initial_trusted_count, queried_count = _run_one_seed(seed_cfg)

        seed_rows.append(
            make_seed_summary_row(
                run_name=base_cfg.get("experiment_name", Path(args.config).stem),
                seed=seed,
                output_dir=str(seed_dir),
                final_metrics=final_metrics,
                run_type=run_type,
                rounds=rounds,
                initial_trusted_count=initial_trusted_count,
                queried_count=queried_count,
            )
        )

    aggregates = aggregate_summary_rows(seed_rows)
    _write_seed_table(seed_rows, output_root)
    _write_aggregate_table(aggregates, output_root)

    with (output_root / "aggregate_results.json").open("w", encoding="utf-8") as handle:
        json.dump(aggregates, handle, indent=2)
    with (output_root / "resolved_base_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(base_cfg, handle, sort_keys=False)
    with (output_root / "multiseed_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "base_config_path": str(Path(args.config).resolve()),
                "output_root": str(output_root.resolve()),
                "seeds": [int(seed) for seed in args.seeds],
                "resume_existing": bool(args.resume_existing),
                "aggregate_results_path": str((output_root / "aggregate_results.json").resolve()),
            },
            handle,
            indent=2,
        )

    print(
        json.dumps(
            {
                "output_dir": str(output_root.resolve()),
                "seeds": [int(seed) for seed in args.seeds],
                "aggregate_results": aggregates,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
