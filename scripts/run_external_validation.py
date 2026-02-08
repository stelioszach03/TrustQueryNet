"""Evaluate a completed multi-seed HAM10000 run on an external ISIC 2019 test set."""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
import json
import math
from pathlib import Path
import sys
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--multiseed-run-dir", required=True, help="Path to an existing multi-seed run directory.")
    parser.add_argument("--ground-truth-csv", required=True, help="ISIC 2019 test ground-truth CSV.")
    parser.add_argument("--image-dir", required=True, help="Directory containing ISIC 2019 external test images.")
    parser.add_argument("--metadata-csv", default=None, help="Optional ISIC 2019 test metadata CSV.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Where to write the external validation results. Defaults to a sibling of the multi-seed run dir.",
    )
    parser.add_argument("--device", default="auto", help="Device override. Defaults to auto.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers for evaluation.")
    parser.add_argument("--batch-size", type=int, default=None, help="Optional evaluation batch size override.")
    parser.add_argument(
        "--checkpoint-name",
        default="best_val_macro_f1.ckpt",
        help="Checkpoint filename to load from each seed run.",
    )
    parser.add_argument(
        "--keep-unk",
        action="store_true",
        help="Keep the ISIC 2019 UNK class instead of filtering it out. Defaults to filtering it out.",
    )
    return parser.parse_args()


def _default_output_dir(multiseed_run_dir: Path) -> Path:
    return multiseed_run_dir.parent / f"{multiseed_run_dir.name}-external-isic2019-test"


def _load_base_config(multiseed_run_dir: Path) -> dict[str, Any]:
    resolved_cfg_path = multiseed_run_dir / "resolved_base_config.yaml"
    if resolved_cfg_path.exists():
        with resolved_cfg_path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)

    manifest_path = multiseed_run_dir / "multiseed_manifest.json"
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        base_path = Path(manifest["base_config_path"])
        with base_path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)

    raise FileNotFoundError(f"Could not recover a base config from {multiseed_run_dir}")


def _load_manifest(multiseed_run_dir: Path) -> dict[str, Any]:
    manifest_path = multiseed_run_dir / "multiseed_manifest.json"
    with manifest_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _build_loader(dataset, batch_size: int, num_workers: int, device):
    from torch.utils.data import DataLoader

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )


def _load_checkpoint_path(seed_dir: Path, checkpoint_name: str) -> tuple[Path, int]:
    report_path = seed_dir / "active_learning_report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        rounds = len(report.get("round_metrics", []))
        checkpoint_path = seed_dir / f"active_round_{rounds}" / checkpoint_name
        return checkpoint_path, rounds

    return seed_dir / checkpoint_name, 1


def _write_seed_table(rows: list[dict[str, Any]], output_dir: Path) -> None:
    fieldnames = list(rows[0].keys())
    csv_path = output_dir / "seed_results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    columns = [
        "seed",
        "test_cal_accuracy",
        "test_cal_macro_f1",
        "test_cal_ece",
        "test_cal_macro_auroc",
        "test_cal_coverage_at_0_5",
        "test_cal_risk_at_0_5",
    ]
    md_path = output_dir / "seed_results.md"
    with md_path.open("w", encoding="utf-8") as handle:
        handle.write("| " + " | ".join(columns) + " |\n")
        handle.write("| " + " | ".join(["---"] * len(columns)) + " |\n")
        for row in rows:
            values = []
            for column in columns:
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
            writer.writerow({"metric": metric_name, **aggregates[metric_name]})

    focus_metrics = [
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
    args = parse_args()

    import torch

    from trustquerynet.data.isic2019 import build_isic2019_external_report, prepare_isic2019_external_test_dataset
    from trustquerynet.eval.metrics import compute_all
    from trustquerynet.eval.multiseed import aggregate_summary_rows, make_seed_summary_row
    from trustquerynet.eval.plots import save_reliability_diagram, save_risk_coverage_plot
    from trustquerynet.eval.selective import default_threshold_grid, risk_coverage_curve
    from trustquerynet.methods.losses import build_loss
    from trustquerynet.models.backbones import create_backbone
    from trustquerynet.training.reproducibility import choose_device
    from trustquerynet.training.trainer import _collect_predictions, build_dataset_bundle
    from trustquerynet.uncertainty.temperature_scaling import fit_temperature

    multiseed_run_dir = Path(args.multiseed_run_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else _default_output_dir(multiseed_run_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = _load_manifest(multiseed_run_dir)
    base_cfg = _load_base_config(multiseed_run_dir)
    device = choose_device(args.device)

    external_bundle = prepare_isic2019_external_test_dataset(
        ground_truth_csv=args.ground_truth_csv,
        image_dir=args.image_dir,
        metadata_csv=args.metadata_csv,
        img_size=int(base_cfg["dataset"]["img_size"]),
        exclude_unk=not args.keep_unk,
    )

    with (output_dir / "external_dataset_report.json").open("w", encoding="utf-8") as handle:
        json.dump(build_isic2019_external_report(external_bundle.manifest), handle, indent=2)

    seed_rows = []
    for seed in manifest["seeds"]:
        seed_cfg = deepcopy(base_cfg)
        seed_cfg["seed"] = int(seed)
        batch_size = int(args.batch_size or seed_cfg["training"]["batch_size"])
        criterion = build_loss(
            seed_cfg["training"]["loss"],
            label_smoothing=float(seed_cfg["training"].get("label_smoothing", 0.0)),
            num_classes=len(external_bundle.class_names),
        )

        internal_bundle = build_dataset_bundle(seed_cfg)
        val_loader = _build_loader(internal_bundle.val, batch_size=batch_size, num_workers=args.num_workers, device=device)
        external_loader = _build_loader(
            external_bundle.test,
            batch_size=batch_size,
            num_workers=args.num_workers,
            device=device,
        )

        model = create_backbone(
            name=seed_cfg["training"]["backbone"],
            pretrained=bool(seed_cfg["training"].get("pretrained", True)),
            num_classes=len(external_bundle.class_names),
            img_size=int(seed_cfg["dataset"]["img_size"]),
        ).to(device)

        seed_dir = multiseed_run_dir / f"seed-{seed}"
        checkpoint_path, rounds = _load_checkpoint_path(seed_dir, args.checkpoint_name)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

        val_outputs = _collect_predictions(model, val_loader, criterion, device, target_key="y_clean")
        external_outputs = _collect_predictions(model, external_loader, criterion, device, target_key="y_clean")

        scaler = fit_temperature(val_outputs["logits"], val_outputs["labels"])
        calibrated_external_probs = scaler.predict_proba(
            torch.tensor(external_outputs["logits"], dtype=torch.float32)
        ).detach().numpy()

        thresholds = seed_cfg.get("evaluation", {}).get("thresholds")
        if thresholds is None:
            thresholds = default_threshold_grid().tolist()

        metrics_uncal = compute_all(external_outputs["labels"], external_outputs["probs"], thresholds=thresholds)
        metrics_cal = compute_all(external_outputs["labels"], calibrated_external_probs, thresholds=thresholds)

        seed_output_dir = output_dir / f"seed-{seed}"
        seed_output_dir.mkdir(parents=True, exist_ok=True)
        (seed_output_dir / "plots").mkdir(parents=True, exist_ok=True)
        save_reliability_diagram(
            seed_output_dir / "plots" / "reliability_uncalibrated.png",
            external_outputs["labels"],
            external_outputs["probs"],
            "External Reliability (Uncalibrated)",
        )
        save_reliability_diagram(
            seed_output_dir / "plots" / "reliability_calibrated.png",
            external_outputs["labels"],
            calibrated_external_probs,
            "External Reliability (Temperature Scaled)",
        )
        save_risk_coverage_plot(
            seed_output_dir / "plots" / "risk_coverage_uncalibrated.png",
            risk_coverage_curve(external_outputs["labels"], external_outputs["probs"], thresholds),
            "External Risk-Coverage (Uncalibrated)",
        )
        save_risk_coverage_plot(
            seed_output_dir / "plots" / "risk_coverage_calibrated.png",
            risk_coverage_curve(external_outputs["labels"], calibrated_external_probs, thresholds),
            "External Risk-Coverage (Temperature Scaled)",
        )

        final_metrics = {
            "device": str(device),
            "history": [],
            "noise": {"noise_type": "external_validation", "seed": int(seed)},
            "test_uncalibrated": metrics_uncal,
            "test_calibrated": metrics_cal,
        }
        with (seed_output_dir / "external_metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(final_metrics, handle, indent=2)

        seed_rows.append(
            make_seed_summary_row(
                run_name=f"{multiseed_run_dir.name}-external-isic2019-test",
                seed=int(seed),
                output_dir=str(seed_output_dir),
                final_metrics=final_metrics,
                run_type="external_validation",
                rounds=rounds,
            )
        )

    aggregates = aggregate_summary_rows(seed_rows)
    _write_seed_table(seed_rows, output_dir)
    _write_aggregate_table(aggregates, output_dir)

    with (output_dir / "aggregate_results.json").open("w", encoding="utf-8") as handle:
        json.dump(aggregates, handle, indent=2)
    with (output_dir / "multiseed_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "source_multiseed_run_dir": str(multiseed_run_dir),
                "seeds": manifest["seeds"],
                "external_dataset": "isic2019_test",
                "ground_truth_csv": str(Path(args.ground_truth_csv).resolve()),
                "image_dir": str(Path(args.image_dir).resolve()),
                "metadata_csv": str(Path(args.metadata_csv).resolve()) if args.metadata_csv else None,
            },
            handle,
            indent=2,
        )

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "seeds": manifest["seeds"],
                "aggregate_results": aggregates,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
