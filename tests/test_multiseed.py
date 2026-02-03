import pytest

from trustquerynet.eval.multiseed import aggregate_summary_rows, make_seed_summary_row


def test_make_seed_summary_row_extracts_best_validation_and_calibrated_test_metrics():
    final_metrics = {
        "device": "cuda",
        "history": [
            {"epoch": 1, "val": {"accuracy": 0.70, "macro_f1": 0.45, "ece": 0.12, "macro_auroc": 0.81}},
            {"epoch": 2, "val": {"accuracy": 0.74, "macro_f1": 0.51, "ece": 0.09, "macro_auroc": 0.84}},
        ],
        "noise": {"noise_type": "pre_applied", "realized_flip_rate": 0.05},
        "test_uncalibrated": {"accuracy": 0.76, "macro_f1": 0.55, "ece": 0.10, "macro_auroc": 0.88},
        "test_calibrated": {
            "accuracy": 0.77,
            "macro_f1": 0.57,
            "ece": 0.04,
            "macro_auroc": 0.89,
            "coverage_at_0.5": 0.90,
            "risk_at_0.5": 0.13,
        },
    }

    row = make_seed_summary_row(
        run_name="full-ham10000-convnext-balanced",
        seed=52,
        output_dir="artifacts/runs/full-ham10000-convnext-balanced-multiseed/seed-52",
        final_metrics=final_metrics,
        run_type="active_learning",
        rounds=2,
        initial_trusted_count=120,
        queried_count=128,
    )

    assert row["seed"] == 52
    assert row["run_type"] == "active_learning"
    assert row["best_val_macro_f1"] == pytest.approx(0.51)
    assert row["test_cal_accuracy"] == pytest.approx(0.77)
    assert row["test_cal_macro_auroc"] == pytest.approx(0.89)
    assert row["test_cal_coverage_at_0_5"] == pytest.approx(0.90)
    assert row["realized_flip_rate"] == pytest.approx(0.05)


def test_aggregate_summary_rows_computes_mean_std_min_max():
    rows = [
        {"seed": 42, "test_cal_accuracy": 0.80, "test_cal_macro_f1": 0.70, "test_cal_ece": 0.03},
        {"seed": 52, "test_cal_accuracy": 0.84, "test_cal_macro_f1": 0.74, "test_cal_ece": 0.02},
        {"seed": 62, "test_cal_accuracy": 0.82, "test_cal_macro_f1": 0.72, "test_cal_ece": 0.04},
    ]

    aggregates = aggregate_summary_rows(rows, metric_keys=["test_cal_accuracy", "test_cal_macro_f1", "test_cal_ece"])

    assert aggregates["test_cal_accuracy"]["mean"] == pytest.approx(0.82)
    assert aggregates["test_cal_accuracy"]["std"] == pytest.approx(0.02)
    assert aggregates["test_cal_accuracy"]["min"] == pytest.approx(0.80)
    assert aggregates["test_cal_accuracy"]["max"] == pytest.approx(0.84)
    assert aggregates["test_cal_macro_f1"]["mean"] == pytest.approx(0.72)
    assert aggregates["test_cal_ece"]["mean"] == pytest.approx(0.03)
