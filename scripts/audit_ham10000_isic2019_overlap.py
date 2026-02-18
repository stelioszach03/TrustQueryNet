"""Audit potential overlap between HAM10000 and ISIC 2019 external images."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ham-metadata-csv", required=True)
    parser.add_argument("--ham-image-dir", required=True)
    parser.add_argument("--ham-split-csv", default=None)
    parser.add_argument("--isic-ground-truth-csv", required=True)
    parser.add_argument("--isic-image-dir", required=True)
    parser.add_argument("--isic-metadata-csv", default=None)
    parser.add_argument("--output-dir", default="artifacts/overlap/ham10000-isic2019")
    parser.add_argument("--dhash-threshold", type=int, default=4)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dhash(path: Path, *, size: int = 8) -> int:
    from PIL import Image

    image = Image.open(path).convert("L").resize((size + 1, size), Image.Resampling.BILINEAR)
    pixels = list(image.getdata())
    rows = [pixels[idx * (size + 1):(idx + 1) * (size + 1)] for idx in range(size)]
    value = 0
    for row in rows:
        for left, right in zip(row[:-1], row[1:]):
            value = (value << 1) | int(left > right)
    return value


def hamming_distance(left: int, right: int) -> int:
    return int((left ^ right).bit_count())


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as handle:
            handle.write("")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_record(row, *, dataset_name: str) -> dict:
    return {
        "dataset": dataset_name,
        "sample_id": str(row["sample_id"]),
        "group_id": str(row["group_id"]),
        "label_name": str(row["label_name"]),
        "image_path": str(row["image_path"]),
        "split": str(row.get("split", "")),
    }


def main() -> None:
    args = parse_args()

    import pandas as pd

    from trustquerynet.data.ham10000_isic import load_ham10000_metadata
    from trustquerynet.data.isic2019 import load_isic2019_external_metadata

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    ham_df = load_ham10000_metadata(args.ham_metadata_csv, args.ham_image_dir)
    if args.ham_split_csv is not None and Path(args.ham_split_csv).exists():
        split_frame = pd.read_csv(args.ham_split_csv)
        ham_df = ham_df.merge(split_frame[["sample_id", "split"]], on="sample_id", how="left")
    isic_df = load_isic2019_external_metadata(
        args.isic_ground_truth_csv,
        args.isic_image_dir,
        metadata_csv=args.isic_metadata_csv,
        exclude_unk=True,
    )

    ham_records = []
    for _, row in ham_df.iterrows():
        path = Path(row["image_path"])
        ham_records.append({**build_record(row, dataset_name="ham10000"), "sha256": sha256_file(path), "dhash": dhash(path)})

    isic_records = []
    for _, row in isic_df.iterrows():
        path = Path(row["image_path"])
        isic_records.append({**build_record(row, dataset_name="isic2019_external"), "sha256": sha256_file(path), "dhash": dhash(path)})

    ham_by_sha = {}
    for record in ham_records:
        ham_by_sha.setdefault(record["sha256"], []).append(record)

    exact_matches = []
    for record in isic_records:
        for ham_record in ham_by_sha.get(record["sha256"], []):
            exact_matches.append(
                {
                    "match_type": "exact_sha256",
                    "ham_sample_id": ham_record["sample_id"],
                    "ham_group_id": ham_record["group_id"],
                    "ham_label_name": ham_record["label_name"],
                    "ham_split": ham_record.get("split", ""),
                    "isic_sample_id": record["sample_id"],
                    "isic_group_id": record["group_id"],
                    "isic_label_name": record["label_name"],
                    "hamming_distance": 0,
                }
            )

    ham_prefix_buckets: dict[str, list[dict]] = {}
    for record in ham_records:
        prefix = format(record["dhash"], "016x")[:4]
        ham_prefix_buckets.setdefault(prefix, []).append(record)

    near_matches = []
    for record in isic_records:
        prefix = format(record["dhash"], "016x")[:4]
        for ham_record in ham_prefix_buckets.get(prefix, []):
            distance = hamming_distance(int(record["dhash"]), int(ham_record["dhash"]))
            if 0 < distance <= args.dhash_threshold:
                near_matches.append(
                    {
                        "match_type": "dhash_near_duplicate",
                        "ham_sample_id": ham_record["sample_id"],
                        "ham_group_id": ham_record["group_id"],
                        "ham_label_name": ham_record["label_name"],
                        "ham_split": ham_record.get("split", ""),
                        "isic_sample_id": record["sample_id"],
                        "isic_group_id": record["group_id"],
                        "isic_label_name": record["label_name"],
                        "hamming_distance": distance,
                    }
                )

    write_csv(output_dir / "exact_matches.csv", exact_matches)
    write_csv(output_dir / "near_duplicate_candidates.csv", near_matches)

    report = {
        "ham10000_samples": len(ham_records),
        "isic2019_external_samples": len(isic_records),
        "exact_match_count": len(exact_matches),
        "near_duplicate_candidate_count": len(near_matches),
        "dhash_threshold": int(args.dhash_threshold),
        "exact_matches_csv": str((output_dir / "exact_matches.csv").resolve()),
        "near_duplicate_candidates_csv": str((output_dir / "near_duplicate_candidates.csv").resolve()),
    }
    with (output_dir / "overlap_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
