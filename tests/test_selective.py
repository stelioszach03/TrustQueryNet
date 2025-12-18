import numpy as np

from trustquerynet.eval.selective import risk_coverage_curve


def test_risk_coverage_coverage_is_monotonic():
    probs = np.array(
        [
            [0.95, 0.05],
            [0.70, 0.30],
            [0.55, 0.45],
            [0.51, 0.49],
        ]
    )
    y_true = np.array([0, 0, 1, 1])
    curve = risk_coverage_curve(y_true, probs, thresholds=[0.0, 0.5, 0.7, 0.9])
    assert curve["coverage"].is_monotonic_decreasing
