import numpy as np

from trustquerynet.training.trainer import _build_weighted_sample_weights


def test_build_weighted_sample_weights_is_inverse_frequency():
    labels = np.array([0, 0, 0, 1, 2, 2], dtype=np.int64)
    weights = _build_weighted_sample_weights(labels)

    expected = np.array([1 / 3, 1 / 3, 1 / 3, 1 / 1, 1 / 2, 1 / 2], dtype=np.float64)
    np.testing.assert_allclose(weights, expected)
