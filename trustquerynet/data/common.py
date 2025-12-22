"""Common dataset wrappers."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class BaseImageDataset(Dataset):
    """Dataset that serves manifest-backed samples with clean and observed labels."""

    def __init__(self, manifest: pd.DataFrame, transform=None) -> None:
        self.manifest = manifest.reset_index(drop=True).copy()
        if "y_observed" not in self.manifest.columns:
            self.manifest["y_observed"] = self.manifest["y_clean"]
        if "is_queried" not in self.manifest.columns:
            self.manifest["is_queried"] = False
        if "is_trusted" not in self.manifest.columns:
            self.manifest["is_trusted"] = False
        self.transform = transform

    def __len__(self) -> int:
        return len(self.manifest)

    def get_observed_labels(self) -> np.ndarray:
        return self.manifest["y_observed"].to_numpy(dtype=np.int64, copy=True)

    def get_clean_labels(self) -> np.ndarray:
        return self.manifest["y_clean"].to_numpy(dtype=np.int64, copy=True)

    def set_observed_labels(self, labels: np.ndarray) -> None:
        if len(labels) != len(self.manifest):
            raise ValueError("Label array length does not match dataset length.")
        self.manifest["y_observed"] = labels.astype(np.int64)

    def repair_labels(self, indices: np.ndarray) -> None:
        if len(indices) == 0:
            return
        indices = np.asarray(indices, dtype=np.int64)
        self.manifest.loc[indices, "y_observed"] = self.manifest.loc[indices, "y_clean"].to_numpy()
        self.manifest.loc[indices, "is_queried"] = True

    def mark_trusted(self, indices: np.ndarray) -> None:
        if len(indices) == 0:
            return
        indices = np.asarray(indices, dtype=np.int64)
        self.manifest.loc[indices, "y_observed"] = self.manifest.loc[indices, "y_clean"].to_numpy()
        self.manifest.loc[indices, "is_trusted"] = True

    def _load_pil_image(self, row: pd.Series) -> Image.Image:
        raise NotImplementedError

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.manifest.iloc[index]
        image = self._load_pil_image(row)
        if self.transform is not None:
            image = self.transform(image)
        return {
            "image": image,
            "y_clean": int(row["y_clean"]),
            "y_observed": int(row["y_observed"]),
            "sample_id": row["sample_id"],
            "group_id": row["group_id"],
            "index": index,
        }
