import numpy as np

from trustquerynet.noise.symmetric import SymmetricNoise
from trustquerynet.noise.transition_matrix import TransitionMatrixNoise


def test_symmetric_noise_flip_rate_is_close():
    y = np.tile(np.arange(4), 250)
    noise = SymmetricNoise(rate=0.2, num_classes=4)
    y_noisy, info = noise.generate(y, seed=123)
    realized = np.mean(y != y_noisy)
    assert 0.15 <= realized <= 0.25
    assert abs(realized - info["realized_flip_rate"]) < 1e-9


def test_transition_matrix_noise_is_deterministic():
    y = np.array([0, 1, 2, 1, 0, 2])
    matrix = np.array(
        [
            [0.9, 0.1, 0.0],
            [0.0, 0.8, 0.2],
            [0.1, 0.0, 0.9],
        ]
    )
    noise = TransitionMatrixNoise(matrix)
    first, _ = noise.generate(y, seed=7)
    second, _ = noise.generate(y, seed=7)
    assert np.array_equal(first, second)
