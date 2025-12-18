from pathlib import Path

import pandas as pd
from PIL import Image

from trustquerynet.data.ham10000_isic import build_ham10000_dataset_report, prepare_ham10000_splits


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
