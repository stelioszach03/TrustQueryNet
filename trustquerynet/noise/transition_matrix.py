"""Class-dependent label noise via transition matrices."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from trustquerynet.noise.base import NoiseModel


class TransitionMatrixNoise(NoiseModel):
    def __init__(self, matrix: np.ndarray) -> None:
        matrix = np.asarray(matrix, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Transition matrix must be square.")
        if not np.allclose(matrix.sum(axis=1), 1.0, atol=1e-6):
            raise ValueError("Each transition matrix row must sum to 1.")
        self.matrix = matrix

    def generate(self, y_clean, *, seed: int, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        rng = np.random.default_rng(seed)
        y_clean = np.asarray(y_clean, dtype=np.int64)
        y_observed = y_clean.copy()
        for idx, clean_label in enumerate(y_clean):
            y_observed[idx] = int(rng.choice(np.arange(self.matrix.shape[0]), p=self.matrix[clean_label]))
        info = {
            "noise_type": "transition_matrix",
            "seed": seed,
            "matrix": self.matrix.tolist(),
            "realized_flip_rate": float(np.mean(y_observed != y_clean)),
        }
        return y_observed, info
