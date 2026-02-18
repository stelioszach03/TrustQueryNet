import numpy as np
import torch

from trustquerynet.eval.metrics import compute_all
from trustquerynet.eval.stats_tests import bootstrap_metric_ci, bootstrap_metric_difference_ci, paired_permutation_test
from trustquerynet.uncertainty.temperature_scaling import fit_temperature


def test_temperature_scaling_keeps_argmax():
    logits = torch.tensor([[4.0, 1.0], [1.2, 2.8], [3.2, 0.3]], dtype=torch.float32)
    labels = torch.tensor([0, 1, 0], dtype=torch.long)
    scaler = fit_temperature(logits.numpy(), labels.numpy())
    scaled_logits = scaler(logits).detach()
    assert torch.equal(logits.argmax(dim=1), scaled_logits.argmax(dim=1))


def test_compute_all_returns_core_metrics():
    probs = np.array([[0.8, 0.2], [0.3, 0.7], [0.6, 0.4]])
    y_true = np.array([0, 1, 1])
    metrics = compute_all(y_true=y_true, probs=probs, thresholds=[0.5, 0.7])
    assert "accuracy" in metrics
    assert "macro_f1" in metrics
    assert "ece" in metrics
    assert "brier" in metrics
    assert "aurc" in metrics


def test_bootstrap_metric_ci_returns_bounds():
    probs = np.array([[0.8, 0.2], [0.3, 0.7], [0.6, 0.4], [0.1, 0.9]])
    y_true = np.array([0, 1, 1, 1])
    result = bootstrap_metric_ci(
        y_true,
        probs,
        metric_fn=lambda y, p: float((p.argmax(axis=1) == y).mean()),
        n_bootstrap=32,
        seed=11,
    )
    assert result["lower"] <= result["mean"] <= result["upper"]


def test_bootstrap_metric_difference_ci_returns_bounds():
    probs_a = np.array([[0.8, 0.2], [0.3, 0.7], [0.6, 0.4], [0.2, 0.8]])
    probs_b = np.array([[0.7, 0.3], [0.4, 0.6], [0.55, 0.45], [0.3, 0.7]])
    y_true = np.array([0, 1, 1, 1])
    result = bootstrap_metric_difference_ci(
        y_true,
        probs_a,
        probs_b,
        metric_fn=lambda y, p: float((p.argmax(axis=1) == y).mean()),
        n_bootstrap=32,
        seed=11,
    )
    assert result["lower"] <= result["mean"] <= result["upper"]


def test_paired_permutation_test_returns_probability():
    result = paired_permutation_test([0.8, 0.82, 0.81], [0.7, 0.72, 0.71], n_permutations=200, seed=7)
    assert 0.0 <= result["p_value"] <= 1.0
