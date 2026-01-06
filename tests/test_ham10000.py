from pathlib import Path

import pandas as pd
from PIL import Image

from trustquerynet.data.ham10000_isic import (
    build_ham10000_dataset_report,
    load_ham10000_metadata,
    prepare_ham10000_splits,
)


def test_prepare_ham10000_splits_and_report(tmp_path: Path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    rows = []
    labels = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
    sample_counter = 0
    for class_name in labels:
        for group_idx in range(3):
            lesion_id = f"{class_name}_lesion_{group_idx}"
            image_id = f"img_{sample_counter}"
            rows.append(
                {
                    "image_id": image_id,
                    "lesion_id": lesion_id,
                    "dx": class_name,
                    "dx_type": "histopathology",
                }
            )
            Image.new("RGB", (8, 8), color=(sample_counter % 255, 0, 0)).save(image_dir / f"{image_id}.jpg")
            sample_counter += 1

    metadata_csv = tmp_path / "HAM10000_metadata.csv"
    pd.DataFrame(rows).to_csv(metadata_csv, index=False)
    split_csv = tmp_path / "splits.csv"

    bundle = prepare_ham10000_splits(
        metadata_csv=metadata_csv,
        image_dir=image_dir,
        seed=42,
        ratios={"train": 0.7, "val": 0.15, "test": 0.15},
        img_size=32,
        split_csv=None,
        save_split_csv=split_csv,
    )
    assert split_csv.exists()
    total = len(bundle.train) + len(bundle.val) + len(bundle.test)
    assert total == len(rows)

    report = build_ham10000_dataset_report(pd.concat(bundle.manifests.values(), ignore_index=True))
    assert report["num_samples"] == len(rows)
    assert report["num_classes"] == 7


def test_load_ham10000_metadata_accepts_isic_collection_schema(tmp_path: Path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    rows = [
        {
            "isic_id": "ISIC_0000001",
            "lesion_id": "lesion_1",
            "diagnosis_2": "Malignant epidermal proliferations",
            "diagnosis_3": "Solar or actinic keratosis",
            "diagnosis_confirm_type": "histopathology",
        },
        {
            "isic_id": "ISIC_0000002",
            "lesion_id": "lesion_2",
            "diagnosis_2": "Malignant epidermal proliferations",
            "diagnosis_3": "Squamous cell carcinoma, NOS",
            "diagnosis_confirm_type": "histopathology",
        },
        {
            "isic_id": "ISIC_0000003",
            "lesion_id": "lesion_3",
            "diagnosis_2": "Benign soft tissue proliferations - Vascular",
            "diagnosis_3": None,
            "diagnosis_confirm_type": "single image expert consensus",
        },
    ]
    for row in rows:
        Image.new("RGB", (8, 8), color=(255, 0, 0)).save(image_dir / f"{row['isic_id']}.jpg")

    metadata_csv = tmp_path / "isic_collection_metadata.csv"
    pd.DataFrame(rows).to_csv(metadata_csv, index=False)

    df = load_ham10000_metadata(metadata_csv=metadata_csv, image_dir=image_dir)

    assert df["sample_id"].tolist() == ["ISIC_0000001", "ISIC_0000002", "ISIC_0000003"]
    assert df["group_id"].tolist() == ["lesion_1", "lesion_2", "lesion_3"]
    assert df["y_clean"].tolist() == [0, 0, 6]
    assert df["ground_truth_type"].tolist() == [
        "histopathology",
        "histopathology",
        "single image expert consensus",
    ]
