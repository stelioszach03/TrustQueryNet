"""HAM10000 helpers."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict

import pandas as pd
from PIL import Image

from trustquerynet.data.common import BaseImageDataset
from trustquerynet.data.splits import make_group_stratified_split
from trustquerynet.data.transforms import build_eval_transform, build_train_transform

HAM10000_CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
HAM10000_CLASS_TO_INDEX = {label: idx for idx, label in enumerate(HAM10000_CLASSES)}
HAM10000_REQUIRED_COLUMNS = {"image_id", "lesion_id", "dx"}


@dataclass
class DatasetBundle:
    train: BaseImageDataset
    val: BaseImageDataset
    test: BaseImageDataset
    class_names: list[str]
    manifests: Dict[str, pd.DataFrame]


class HAM10000Dataset(BaseImageDataset):
    def _load_pil_image(self, row: pd.Series) -> Image.Image:
        return Image.open(row["image_path"]).convert("RGB")


def load_ham10000_metadata(metadata_csv: str | Path, image_dir: str | Path) -> pd.DataFrame:
    metadata_csv = Path(metadata_csv)
    image_dir = Path(image_dir)
    if not metadata_csv.exists():
        raise FileNotFoundError(f"HAM10000 metadata CSV not found: {metadata_csv}")
    if not image_dir.exists():
        raise FileNotFoundError(f"HAM10000 image directory not found: {image_dir}")
    df = pd.read_csv(metadata_csv)
    missing = HAM10000_REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"HAM10000 metadata is missing required columns: {sorted(missing)}")
    df = df.rename(columns={"image_id": "sample_id", "lesion_id": "group_id"})
    df["y_clean"] = df["dx"].map(HAM10000_CLASS_TO_INDEX)
    if "dx_type" in df.columns:
        df["ground_truth_type"] = df["dx_type"]
    else:
        df["ground_truth_type"] = "unknown"
    df["image_path"] = df["sample_id"].apply(lambda sample_id: str(image_dir / f"{sample_id}.jpg"))
    df["is_queried"] = False
    df["is_trusted"] = False
    if df["y_clean"].isna().any():
        raise ValueError("Found HAM10000 rows with unknown dx labels.")
    return df


def build_ham10000_dataset_report(df: pd.DataFrame) -> Dict[str, object]:
    class_counts = {
        HAM10000_CLASSES[int(class_idx)]: int(count)
        for class_idx, count in df["y_clean"].value_counts().sort_index().items()
    }
    gt_counts = {str(key): int(value) for key, value in df["ground_truth_type"].value_counts().to_dict().items()}
    duplicate_sample_ids = int(df["sample_id"].duplicated().sum())
    groups_with_multiple_images = int((df.groupby("group_id")["sample_id"].count() > 1).sum())
    return {
        "num_samples": int(len(df)),
        "num_groups": int(df["group_id"].nunique()),
        "num_classes": len(HAM10000_CLASSES),
        "class_counts": class_counts,
        "ground_truth_type_counts": gt_counts,
        "duplicate_sample_ids": duplicate_sample_ids,
        "groups_with_multiple_images": groups_with_multiple_images,
    }


def write_ham10000_dataset_report(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(build_ham10000_dataset_report(df), handle, indent=2)


def prepare_ham10000_splits(
    metadata_csv: str | Path,
    image_dir: str | Path,
    seed: int,
    ratios: Dict[str, float],
    img_size: int,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
    max_test_samples: int | None = None,
    split_csv: str | Path | None = None,
    save_split_csv: str | Path | None = None,
) -> DatasetBundle:
    df = load_ham10000_metadata(metadata_csv=metadata_csv, image_dir=image_dir)
    if split_csv is not None and Path(split_csv).exists():
        split_df = pd.read_csv(split_csv)
        df = df.merge(split_df[["sample_id", "split"]], on="sample_id", how="left")
    else:
        df = make_group_stratified_split(df, label_col="y_clean", group_col="group_id", seed=seed, ratios=ratios)
        if save_split_csv is not None:
            save_path = Path(save_split_csv)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            df[["sample_id", "group_id", "y_clean", "ground_truth_type", "split"]].to_csv(save_path, index=False)

    manifests = {}
    datasets = {}
    rng = pd.Series(range(len(df))).sample(frac=1.0, random_state=seed).to_numpy()
    df = df.iloc[rng].reset_index(drop=True)

    limits = {"train": max_train_samples, "val": max_val_samples, "test": max_test_samples}
    for split_name, transform in {
        "train": build_train_transform(img_size),
        "val": build_eval_transform(img_size),
        "test": build_eval_transform(img_size),
    }.items():
        manifest = df[df["split"] == split_name].copy().reset_index(drop=True)
        limit = limits[split_name]
        if limit is not None and len(manifest) > limit:
            manifest = manifest.iloc[:limit].reset_index(drop=True)
        manifests[split_name] = manifest
        datasets[split_name] = HAM10000Dataset(manifest=manifest, transform=transform)

    return DatasetBundle(
        train=datasets["train"],
        val=datasets["val"],
        test=datasets["test"],
        class_names=HAM10000_CLASSES,
        manifests=manifests,
    )
