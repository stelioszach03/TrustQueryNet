"""Export compact paper-facing paired significance tables from multiseed runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from trustquerynet.eval.stats_tests import bootstrap_paired_mean_difference_ci, paired_permutation_test


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-root",
        default="artifacts/final_evidence",
        help="Root directory containing the selected multiseed run folders.",
    )
    parser.add_argument("--repair-run", required=True, help="Internal repair multiseed run directory name.")
    parser.add_argument("--no-repair-run", required=True, help="Internal no-repair multiseed run directory name.")
    parser.add_argument("--random-repair-run", required=True, help="Internal random-repair multiseed run directory name.")
    parser.add_argument("--gce-run", required=True, help="Internal GCE multiseed run directory name.")
    parser.add_argument("--repair-external-run", default=None, help="External repair multiseed run directory name.")
    parser.add_argument("--no-repair-external-run", default=None, help="External no-repair multiseed run directory name.")
    parser.add_argument("--random-repair-external-run", default=None, help="External random-repair multiseed run directory name.")
    parser.add_argument(
        "--output-dir",
        default="artifacts/paper_tables/significance",
        help="Directory where CSV/Markdown/JSON outputs will be written.",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=10000,
        help="Bootstrap resamples for paired mean-difference confidence intervals.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="Confidence level for paired mean-difference intervals.",
    )
    return parser.parse_args()


def load_seed_results(run_dir: Path) -> dict[int, dict[str, str]]:
    """Load the per-seed summary rows used for paper-facing paired comparisons."""
    seed_results_path = run_dir / "seed_results.csv"
    if not seed_results_path.exists():
        raise FileNotFoundError(f"Expected seed_results.csv in {run_dir}")
    with seed_results_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {int(row["seed"]): row for row in rows}


def build_row(
    *,
    split: str,
    comparison: str,
    metric_key: str,
    metric_label: str,
    lhs_label: str,
    rhs_label: str,
    lhs_rows: dict[int, dict[str, str]],
    rhs_rows: dict[int, dict[str, str]],
    n_bootstrap: int,
    confidence: float,
    seed: int,
) -> dict[str, Any]:
    """Build one paired-comparison row for the significance exports."""
    shared_seeds = sorted(set(lhs_rows) & set(rhs_rows))
    if not shared_seeds:
        raise ValueError(f"No shared seeds found for {comparison}")

    lhs_values = [float(lhs_rows[item][metric_key]) for item in shared_seeds]
    rhs_values = [float(rhs_rows[item][metric_key]) for item in shared_seeds]

    ci = bootstrap_paired_mean_difference_ci(
        lhs_values,
        rhs_values,
        n_bootstrap=n_bootstrap,
        confidence=confidence,
        seed=seed,
    )
    permutation = paired_permutation_test(lhs_values, rhs_values, exact=True, seed=seed)

    return {
        "split": split,
        "comparison": comparison,
        "metric": metric_label,
        "metric_key": metric_key,
        "lhs_label": lhs_label,
        "rhs_label": rhs_label,
        "n_pairs": len(shared_seeds),
        "shared_seeds": ",".join(str(item) for item in shared_seeds),
        "lhs_mean": sum(lhs_values) / len(lhs_values),
        "rhs_mean": sum(rhs_values) / len(rhs_values),
        "mean_difference": permutation["statistic"],
        "ci_lower": ci["lower"],
        "ci_upper": ci["upper"],
        "p_value": permutation["p_value"],
        "exact_permutation": permutation.get("exact", False),
    }


def write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    fieldnames = list(rows[0].keys())
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _format_decimal(value: float) -> str:
    return f"{value:.4f}"


def write_markdown(rows: list[dict[str, Any]], output_path: Path, confidence: float) -> None:
    """Write a compact reviewer-facing Markdown summary of paired comparisons."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["split"], []).append(row)

    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# TrustQueryNet Paired Statistical Comparisons\n\n")
        handle.write(
            "Seed-paired comparisons were computed on the exported multiseed summaries. "
            "Each row reports the mean delta (left minus right), a paired bootstrap confidence "
            f"interval with confidence {confidence:.0%}, and a two-sided exact sign-flip permutation p-value.\n\n"
        )
        handle.write(
            "Because only five shared seeds are available for the main comparisons and three for the GCE anchor, "
            "the p-value resolution is intentionally coarse. The table should therefore be read primarily as an "
            "effect-size and uncertainty summary rather than as a thresholded significance screen.\n"
        )
        for split in ("internal", "external"):
            split_rows = grouped.get(split, [])
            if not split_rows:
                continue
            handle.write(f"\n## {split.title()} Comparisons\n\n")
            handle.write(
                "| Comparison | Metric | Shared seeds | Delta (left - right) | "
                f"{int(confidence * 100)}% paired bootstrap CI | Exact sign-flip p |\n"
            )
            handle.write("| --- | --- | --- | ---: | --- | ---: |\n")
            for row in split_rows:
                handle.write(
                    f"| {row['comparison']} | {row['metric']} | `{row['shared_seeds']}` | "
                    f"{_format_decimal(row['mean_difference'])} | "
                    f"[{_format_decimal(row['ci_lower'])}, {_format_decimal(row['ci_upper'])}] | "
                    f"{_format_decimal(row['p_value'])} |\n"
                )

        handle.write("\n## Notes\n\n")
        handle.write(
            "- Internal primary comparisons use calibrated test metrics from the locked `e12` publication recipe.\n"
        )
        handle.write(
            "- Comparisons against `GCE no repair` are restricted to the shared three-seed subset (`42,52,62`), "
            "because that anchor was not run on five seeds.\n"
        )
        handle.write(
            "- The main manuscript uses calibrated macro-F1 as the primary model-selection and reporting endpoint; "
            "calibrated accuracy is retained here as a secondary summary.\n"
        )


def main() -> None:
    args = parse_args()
    runs_root = Path(args.runs_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    internal_runs = {
        "Repair": load_seed_results(runs_root / args.repair_run),
        "No repair": load_seed_results(runs_root / args.no_repair_run),
        "Random repair": load_seed_results(runs_root / args.random_repair_run),
        "GCE no repair": load_seed_results(runs_root / args.gce_run),
    }

    rows = []
    metric_specs = [
        ("test_cal_macro_f1", "Calibrated macro-F1"),
        ("test_cal_accuracy", "Calibrated accuracy"),
    ]
    internal_comparisons = [
        ("Repair vs no repair", "Repair", "No repair"),
        ("Repair vs random repair", "Repair", "Random repair"),
        ("Repair vs GCE no repair", "Repair", "GCE no repair"),
    ]
    for metric_key, metric_label in metric_specs:
        for comparison, lhs_label, rhs_label in internal_comparisons:
            rows.append(
                build_row(
                    split="internal",
                    comparison=comparison,
                    metric_key=metric_key,
                    metric_label=metric_label,
                    lhs_label=lhs_label,
                    rhs_label=rhs_label,
                    lhs_rows=internal_runs[lhs_label],
                    rhs_rows=internal_runs[rhs_label],
                    n_bootstrap=args.n_bootstrap,
                    confidence=args.confidence,
                    seed=42,
                )
            )

    if args.repair_external_run and args.no_repair_external_run and args.random_repair_external_run:
        external_runs = {
            "Repair": load_seed_results(runs_root / args.repair_external_run),
            "No repair": load_seed_results(runs_root / args.no_repair_external_run),
            "Random repair": load_seed_results(runs_root / args.random_repair_external_run),
        }
        external_comparisons = [
            ("Repair vs no repair", "Repair", "No repair"),
            ("Repair vs random repair", "Repair", "Random repair"),
        ]
        for metric_key, metric_label in metric_specs:
            for comparison, lhs_label, rhs_label in external_comparisons:
                rows.append(
                    build_row(
                        split="external",
                        comparison=comparison,
                        metric_key=metric_key,
                        metric_label=metric_label,
                        lhs_label=lhs_label,
                        rhs_label=rhs_label,
                        lhs_rows=external_runs[lhs_label],
                        rhs_rows=external_runs[rhs_label],
                        n_bootstrap=args.n_bootstrap,
                        confidence=args.confidence,
                        seed=84,
                    )
                )

    csv_path = output_dir / "paired_significance.csv"
    md_path = output_dir / "paired_significance.md"
    json_path = output_dir / "paired_significance.json"

    write_csv(rows, csv_path)
    write_markdown(rows, md_path, confidence=args.confidence)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "methodology": {
                    "paired_bootstrap": {
                        "n_bootstrap": args.n_bootstrap,
                        "confidence": args.confidence,
                        "target": "mean difference across shared seeds",
                    },
                    "paired_permutation_test": {
                        "type": "two-sided exact sign-flip test",
                    },
                },
                "rows": rows,
            },
            handle,
            indent=2,
        )

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "csv_path": str(csv_path),
                "md_path": str(md_path),
                "json_path": str(json_path),
                "rows": len(rows),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
