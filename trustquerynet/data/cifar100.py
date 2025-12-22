"""CIFAR-100 helpers for quick-mode experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision.datasets import CIFAR100

from trustquerynet.data.common import BaseImageDataset
from trustquerynet.data.transforms import build_eval_transform, build_train_transform


@dataclass
class DatasetBundle:
    train: BaseImageDataset
    val: BaseImageDataset
    test: BaseImageDataset
    class_names: list[str]
    manifests: Dict[str, pd.DataFrame]


class CIFAR100Dataset(BaseImageDataset):
    def __init__(self, manifest: pd.DataFrame, base_dataset: CIFAR100, transform=None) -> None:
        super().__init__(manifest=manifest, transform=transform)
        self.base_dataset = base_dataset

    def _load_pil_image(self, row: pd.Series) -> Image.Image:
        image, _ = self.base_dataset[int(row["data_idx"])]
        return image.convert("RGB")


def prepare_cifar100_splits(
    root: str | Path,
    seed: int,
    val_ratio: float,
    img_size: int,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
    max_test_samples: int | None = None,
) -> DatasetBundle:
    root = Path(root)
    train_base = CIFAR100(root=root, train=True, download=True)
    test_base = CIFAR100(root=root, train=False, download=True)

    train_indices = np.arange(len(train_base.targets))
    y_train = np.asarray(train_base.targets, dtype=np.int64)
    train_idx, val_idx = train_test_split(
        train_indices,
        test_size=val_ratio,
        random_state=seed,
        stratify=y_train,
    )

    train_manifest = pd.DataFrame(
        {
            "sample_id": [f"cifar100_train_{idx}" for idx in train_idx],
            "group_id": [f"cifar100_train_{idx}" for idx in train_idx],
            "data_idx": train_idx,
            "y_clean": y_train[train_idx],
            "split": "train",
            "is_queried": False,
            "is_trusted": False,
        }
    )
    val_manifest = pd.DataFrame(
        {
            "sample_id": [f"cifar100_train_{idx}" for idx in val_idx],
            "group_id": [f"cifar100_train_{idx}" for idx in val_idx],
            "data_idx": val_idx,
            "y_clean": y_train[val_idx],
            "split": "val",
            "is_queried": False,
            "is_trusted": False,
        }
    )
    test_indices = np.arange(len(test_base.targets))
    y_test = np.asarray(test_base.targets, dtype=np.int64)
    test_manifest = pd.DataFrame(
        {
            "sample_id": [f"cifar100_test_{idx}" for idx in test_indices],
            "group_id": [f"cifar100_test_{idx}" for idx in test_indices],
            "data_idx": test_indices,
            "y_clean": y_test,
            "split": "test",
            "is_queried": False,
            "is_trusted": False,
        }
    )

    rng = np.random.default_rng(seed)

    def _maybe_take_subset(frame: pd.DataFrame, max_samples: int | None) -> pd.DataFrame:
        if max_samples is None or len(frame) <= max_samples:
            return frame.reset_index(drop=True)
        chosen = rng.choice(frame.index.to_numpy(), size=max_samples, replace=False)
        return frame.loc[np.sort(chosen)].reset_index(drop=True)

    train_manifest = _maybe_take_subset(train_manifest, max_train_samples)
    val_manifest = _maybe_take_subset(val_manifest, max_val_samples)
    test_manifest = _maybe_take_subset(test_manifest, max_test_samples)

    return DatasetBundle(
        train=CIFAR100Dataset(train_manifest, train_base, transform=build_train_transform(img_size)),
        val=CIFAR100Dataset(val_manifest, train_base, transform=build_eval_transform(img_size)),
        test=CIFAR100Dataset(test_manifest, test_base, transform=build_eval_transform(img_size)),
        class_names=list(train_base.classes),
        manifests={"train": train_manifest, "val": val_manifest, "test": test_manifest},
    )
