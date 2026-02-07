"""Export a paper-ready ablation comparison table from multi-seed run folders."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from trustquerynet.eval.ablation import summarize_multiseed_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-root",
        default="artifacts/runs",
        help="Root directory containing the selected multi-seed run folders.",
    )
    parser.add_argument(
        "--run-spec",
        action="append",
        required=True,
        help="Run folder and label, formatted as run_dir_name::Display label",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/ablations/ham10000-ablation-table",
        help="Directory where CSV/Markdown/JSON outputs will be written.",
    )
    return parser.parse_args()


def parse_run_spec(spec: str) -> tuple[str, str]:
    if "::" in spec:
        run_name, label = spec.split("::", 1)
        return run_name.strip(), label.strip()
    return spec.strip(), spec.strip()


def write_csv(rows: list[dict], output_path: Path) -> None:
    fieldnames = list(rows[0].keys())
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict], output_path: Path) -> None:
    columns = [
        "label",
        "n_seeds",
        "test_cal_accuracy",
        "test_cal_macro_f1",
        "test_cal_macro_auroc",
        "test_uncal_ece",
        "test_cal_ece",
        "ece_improvement",
        "test_cal_coverage_at_0_5",
        "test_cal_risk_at_0_5",
    ]
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("| " + " | ".join(columns) + " |\n")
        handle.write("| " + " | ".join(["---"] * len(columns)) + " |\n")
        for row in rows:
            values = [str(row.get(column, "")) for column in columns]
            handle.write("| " + " | ".join(values) + " |\n")


def main() -> None:
    args = parse_args()
    runs_root = Path(args.runs_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for spec in args.run_spec:
        run_name, label = parse_run_spec(spec)
        run_dir = runs_root / run_name
        rows.append(summarize_multiseed_run(run_dir, label=label))

    csv_path = output_dir / "ablation_table.csv"
    md_path = output_dir / "ablation_table.md"
    json_path = output_dir / "ablation_table.json"

    write_csv(rows, csv_path)
    write_markdown(rows, md_path)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)

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
