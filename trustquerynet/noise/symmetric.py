"""Symmetric label noise."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from trustquerynet.noise.base import NoiseModel


class SymmetricNoise(NoiseModel):
    def __init__(self, rate: float, num_classes: int) -> None:
        if not 0.0 <= rate <= 1.0:
            raise ValueError("Noise rate must be within [0, 1].")
        self.rate = rate
        self.num_classes = num_classes

    def generate(self, y_clean, *, seed: int, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        rng = np.random.default_rng(seed)
        y_clean = np.asarray(y_clean, dtype=np.int64)
        y_observed = y_clean.copy()
        flip_mask = rng.random(len(y_clean)) < self.rate
        for idx in np.where(flip_mask)[0]:
            choices = np.delete(np.arange(self.num_classes), y_clean[idx])
            y_observed[idx] = int(rng.choice(choices))
        info = {
            "noise_type": "symmetric",
            "rate": self.rate,
            "seed": seed,
            "realized_flip_rate": float(np.mean(y_observed != y_clean)),
        }
        return y_observed, info
