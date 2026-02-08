from pathlib import Path

import pandas as pd

from trustquerynet.data.isic2019 import build_isic2019_external_report, load_isic2019_external_metadata


def test_load_isic2019_external_metadata_maps_supported_labels_and_filters_unk(tmp_path: Path):
    ground_truth = pd.DataFrame(
        [
            {"image": "ISIC_A", "MEL": 1.0, "NV": 0.0, "BCC": 0.0, "AK": 0.0, "BKL": 0.0, "DF": 0.0, "VASC": 0.0, "SCC": 0.0, "UNK": 0.0},
            {"image": "ISIC_B", "MEL": 0.0, "NV": 0.0, "BCC": 0.0, "AK": 1.0, "BKL": 0.0, "DF": 0.0, "VASC": 0.0, "SCC": 0.0, "UNK": 0.0},
            {"image": "ISIC_C", "MEL": 0.0, "NV": 0.0, "BCC": 0.0, "AK": 0.0, "BKL": 0.0, "DF": 0.0, "VASC": 0.0, "SCC": 1.0, "UNK": 0.0},
            {"image": "ISIC_D", "MEL": 0.0, "NV": 0.0, "BCC": 0.0, "AK": 0.0, "BKL": 0.0, "DF": 0.0, "VASC": 0.0, "SCC": 0.0, "UNK": 1.0},
        ]
    )
    metadata = pd.DataFrame(
        [
            {"image": "ISIC_A", "lesion_id": "lesion-1"},
            {"image": "ISIC_B", "lesion_id": "lesion-2"},
            {"image": "ISIC_C", "lesion_id": "lesion-3"},
            {"image": "ISIC_D", "lesion_id": "lesion-4"},
        ]
    )

    ground_truth_path = tmp_path / "gt.csv"
    metadata_path = tmp_path / "meta.csv"
    image_dir = tmp_path / "images"
    image_dir.mkdir()

    ground_truth.to_csv(ground_truth_path, index=False)
    metadata.to_csv(metadata_path, index=False)
    for sample_id in ["ISIC_A", "ISIC_B", "ISIC_C", "ISIC_D"]:
        (image_dir / f"{sample_id}.jpg").write_bytes(b"fake")

    df = load_isic2019_external_metadata(
        ground_truth_csv=ground_truth_path,
        image_dir=image_dir,
        metadata_csv=metadata_path,
        exclude_unk=True,
    )

    assert df["sample_id"].tolist() == ["ISIC_A", "ISIC_B", "ISIC_C"]
    assert df["label_name"].tolist() == ["mel", "akiec", "akiec"]
    assert df["group_id"].tolist() == ["lesion-1", "lesion-2", "lesion-3"]
    assert set(df["split"].unique()) == {"external_test"}

    report = build_isic2019_external_report(df)
    assert report["num_samples"] == 3
