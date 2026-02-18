"""Verify that Q1 smoke artifacts and configs satisfy the rerun gate conditions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from trustquerynet.eval.phase_gate import verify_config_gate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repair-config", required=True)
    parser.add_argument("--no-repair-config", required=True)
    parser.add_argument("--random-repair-config", required=True)
    parser.add_argument("--repair-run-dir", default=None)
    parser.add_argument("--random-repair-run-dir", default=None)
    parser.add_argument("--external-run-dir", default=None)
    parser.add_argument("--expected-checkpoint-policy", default="best_val_macro_f1")
    return parser.parse_args()


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _extract_metrics(run_dir: Path) -> dict[str, Any]:
    report_path = run_dir / "active_learning_report.json"
    if report_path.exists():
        report = load_json(report_path)
        return report["final_metrics"]
    return load_json(run_dir / "metrics.json")


def verify_run_gate(run_dir: Path, expected_checkpoint_policy: str) -> dict[str, bool]:
    metrics = _extract_metrics(run_dir)
    return {
        "checkpoint_policy_explicit": metrics.get("selected_checkpoint", {}).get("policy") == expected_checkpoint_policy,
        "selected_checkpoint_epoch_present": metrics.get("selected_checkpoint", {}).get("epoch") is not None,
        "dense_selective_metrics_present": "aurc" in metrics.get("test_calibrated", {}),
    }


def verify_external_gate(run_dir: Path, expected_checkpoint_policy: str) -> dict[str, bool]:
    seed_dirs = sorted(path for path in run_dir.glob("seed-*") if path.is_dir())
    if not seed_dirs:
        return {"external_checkpoint_policy_explicit": False, "external_dense_selective_metrics_present": False}
    metrics = load_json(seed_dirs[0] / "external_metrics.json")
    return {
        "external_checkpoint_policy_explicit": metrics.get("selected_checkpoint", {}).get("policy") == expected_checkpoint_policy,
        "external_dense_selective_metrics_present": "aurc" in metrics.get("test_calibrated", {}),
    }


def main() -> None:
    args = parse_args()
    repair_cfg = load_yaml(args.repair_config)
    no_repair_cfg = load_yaml(args.no_repair_config)
    random_cfg = load_yaml(args.random_repair_config)

    report = {
        "config_gate": verify_config_gate(repair_cfg, no_repair_cfg, random_cfg),
        "run_gate": {},
        "external_gate": {},
    }

    if args.repair_run_dir is not None:
        report["run_gate"].update(verify_run_gate(Path(args.repair_run_dir), args.expected_checkpoint_policy))
    if args.random_repair_run_dir is not None:
        random_gate = verify_run_gate(Path(args.random_repair_run_dir), args.expected_checkpoint_policy)
        report["run_gate"].update({f"random_{key}": value for key, value in random_gate.items()})
    if args.external_run_dir is not None:
        report["external_gate"].update(verify_external_gate(Path(args.external_run_dir), args.expected_checkpoint_policy))

    report["all_checks_passed"] = all(
        value
        for section in report.values()
        if isinstance(section, dict)
        for value in section.values()
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
