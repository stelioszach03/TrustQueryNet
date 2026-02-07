from pathlib import Path
import json

from trustquerynet.eval.ablation import format_mean_std, summarize_multiseed_run


def test_format_mean_std_handles_missing_and_numeric_values():
    assert format_mean_std(None) == ""
    assert format_mean_std({"mean": 0.81234, "std": 0.01234}) == "0.8123 ± 0.0123"


def test_summarize_multiseed_run_builds_ece_improvement_row(tmp_path: Path):
    run_dir = tmp_path / "example-run"
    run_dir.mkdir()

    with (run_dir / "aggregate_results.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "test_cal_accuracy": {"mean": 0.80, "std": 0.01},
                "test_cal_macro_f1": {"mean": 0.70, "std": 0.02},
                "test_cal_macro_auroc": {"mean": 0.94, "std": 0.01},
                "test_uncal_ece": {"mean": 0.08, "std": 0.01},
                "test_cal_ece": {"mean": 0.03, "std": 0.005},
                "test_cal_coverage_at_0_5": {"mean": 0.93, "std": 0.01},
                "test_cal_risk_at_0_5": {"mean": 0.16, "std": 0.01},
            },
            handle,
        )
    with (run_dir / "multiseed_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump({"seeds": [42, 52, 62, 72, 82]}, handle)

    row = summarize_multiseed_run(run_dir, label="Full model")

    assert row["label"] == "Full model"
    assert row["n_seeds"] == 5
    assert row["test_cal_accuracy"] == "0.8000 ± 0.0100"
    assert row["test_cal_ece"] == "0.0300 ± 0.0050"
    assert row["ece_improvement"] == "0.0500"
