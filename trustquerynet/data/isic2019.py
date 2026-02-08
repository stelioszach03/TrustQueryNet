"""ISIC 2019 dermoscopy challenge helpers for external validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import pandas as pd
from PIL import Image

from trustquerynet.data.common import BaseImageDataset
from trustquerynet.data.ham10000_isic import HAM10000_CLASSES, HAM10000_CLASS_TO_INDEX
from trustquerynet.data.transforms import build_eval_transform


ISIC2019_ONE_HOT_COLUMNS = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC", "UNK"]
ISIC2019_TO_HAM10000_LABEL = {
    "MEL": "mel",
    "NV": "nv",
    "BCC": "bcc",
    "AK": "akiec",
    "BKL": "bkl",
    "DF": "df",
    "VASC": "vasc",
    "SCC": "akiec",
    "UNK": None,
}


@dataclass
class ExternalDatasetBundle:
    test: BaseImageDataset
    class_names: list[str]
    manifest: pd.DataFrame


class ISIC2019Dataset(BaseImageDataset):
    def _load_pil_image(self, row: pd.Series) -> Image.Image:
        return Image.open(row["image_path"]).convert("RGB")


def _resolve_isic2019_label(row: pd.Series) -> str | None:
    active = [column for column in ISIC2019_ONE_HOT_COLUMNS if float(row[column]) == 1.0]
    if len(active) != 1:
        raise ValueError(f"Expected exactly one active ISIC 2019 label, got {active}")
    return ISIC2019_TO_HAM10000_LABEL[active[0]]


def load_isic2019_external_metadata(
    ground_truth_csv: str | Path,
    image_dir: str | Path,
    *,
    metadata_csv: str | Path | None = None,
    exclude_unk: bool = True,
) -> pd.DataFrame:
    ground_truth_csv = Path(ground_truth_csv)
    image_dir = Path(image_dir)
    if not ground_truth_csv.exists():
        raise FileNotFoundError(f"ISIC 2019 ground truth CSV not found: {ground_truth_csv}")
    if not image_dir.exists():
        raise FileNotFoundError(f"ISIC 2019 image directory not found: {image_dir}")

    df = pd.read_csv(ground_truth_csv)
    missing_columns = [column for column in ISIC2019_ONE_HOT_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"ISIC 2019 ground truth CSV is missing required columns: {missing_columns}")

    df = df.rename(columns={"image": "sample_id"})
    df["label_name"] = df.apply(_resolve_isic2019_label, axis=1)
    if exclude_unk:
        df = df[df["label_name"].notna()].copy()
    else:
        df["label_name"] = df["label_name"].fillna("unknown")

    if metadata_csv is not None:
        metadata_df = pd.read_csv(metadata_csv)
        if "image" in metadata_df.columns:
            metadata_df = metadata_df.rename(columns={"image": "sample_id"})
        df = df.merge(metadata_df, on="sample_id", how="left")

    if "lesion_id" in df.columns:
        df["group_id"] = df["lesion_id"].fillna(df["sample_id"]).astype(str)
    else:
        df["group_id"] = df["sample_id"].astype(str)

    df["y_clean"] = df["label_name"].map(HAM10000_CLASS_TO_INDEX)
    if df["y_clean"].isna().any():
        unknown = sorted(df.loc[df["y_clean"].isna(), "label_name"].fillna("NA").unique().tolist())
        raise ValueError(f"Found ISIC 2019 rows with unsupported labels: {unknown}")

    def _resolve_image_path(sample_id: str) -> str:
        for suffix in (".jpg", ".jpeg", ".png"):
            candidate = image_dir / f"{sample_id}{suffix}"
            if candidate.exists():
                return str(candidate)
        return str(image_dir / f"{sample_id}.jpg")

    df["image_path"] = df["sample_id"].map(_resolve_image_path)
    missing_images = [path for path in df["image_path"].tolist() if not Path(path).exists()]
    if missing_images:
        raise FileNotFoundError(f"Missing ISIC 2019 images. First missing path: {missing_images[0]}")

    df["split"] = "external_test"
    df["is_queried"] = False
    df["is_trusted"] = False
    return df.reset_index(drop=True)


def prepare_isic2019_external_test_dataset(
    ground_truth_csv: str | Path,
    image_dir: str | Path,
    *,
    metadata_csv: str | Path | None = None,
    img_size: int = 224,
    exclude_unk: bool = True,
) -> ExternalDatasetBundle:
    manifest = load_isic2019_external_metadata(
        ground_truth_csv=ground_truth_csv,
        image_dir=image_dir,
        metadata_csv=metadata_csv,
        exclude_unk=exclude_unk,
    )
    dataset = ISIC2019Dataset(manifest=manifest, transform=build_eval_transform(img_size))
    return ExternalDatasetBundle(test=dataset, class_names=HAM10000_CLASSES, manifest=manifest)


def build_isic2019_external_report(df: pd.DataFrame) -> Dict[str, object]:
    class_counts = {
        HAM10000_CLASSES[int(class_idx)]: int(count)
        for class_idx, count in df["y_clean"].value_counts().sort_index().items()
    }
    result = {
        "num_samples": int(len(df)),
        "num_groups": int(df["group_id"].nunique()),
        "class_counts": class_counts,
    }
    if "validation_weight" in df.columns:
        result["validation_weight_mean"] = float(df["validation_weight"].mean())
    if "score_weight" in df.columns:
        result["score_weight_mean"] = float(df["score_weight"].mean())
    return result
