"""Noise model interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import numpy as np


class NoiseModel(ABC):
    @abstractmethod
    def generate(self, y_clean, *, seed: int, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        raise NotImplementedError


def build_noise_model(noise_cfg: Dict[str, Any], num_classes: int) -> NoiseModel:
    noise_type = noise_cfg.get("type", "symmetric")
    if noise_type == "symmetric":
        from trustquerynet.noise.symmetric import SymmetricNoise

        return SymmetricNoise(rate=float(noise_cfg.get("rate", 0.0)), num_classes=num_classes)
    if noise_type == "transition_matrix":
        from trustquerynet.noise.transition_matrix import TransitionMatrixNoise

        matrix = np.asarray(noise_cfg["matrix"], dtype=np.float64)
        return TransitionMatrixNoise(matrix=matrix)
    raise ValueError(f"Unsupported noise type: {noise_type}")
