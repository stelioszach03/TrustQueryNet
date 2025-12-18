"""Prepare HAM10000 split and dataset report artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from trustquerynet.data.ham10000_isic import (
    build_ham10000_dataset_report,
    load_ham10000_metadata,
    write_ham10000_dataset_report,
)
from trustquerynet.data.splits import make_group_stratified_split


def main() -> None:
    parser = argparse.ArgumentParser(description="Create HAM10000 split and report files.")
    parser.add_argument("--metadata-csv", required=True, help="Path to HAM10000_metadata.csv")
    parser.add_argument("--image-dir", required=True, help="Path to the HAM10000 images directory")
    parser.add_argument("--split-csv", required=True, help="Where to write the split manifest CSV")
    parser.add_argument("--report-json", required=True, help="Where to write the dataset report JSON")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split generation")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    args = parser.parse_args()

    df = load_ham10000_metadata(args.metadata_csv, args.image_dir)
    split_df = make_group_stratified_split(
        df,
        label_col="y_clean",
        group_col="group_id",
        seed=args.seed,
        ratios={"train": args.train_ratio, "val": args.val_ratio, "test": args.test_ratio},
    )

    split_path = Path(args.split_csv)
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_df[["sample_id", "group_id", "y_clean", "ground_truth_type", "split"]].to_csv(split_path, index=False)
    write_ham10000_dataset_report(split_df, args.report_json)

    report = build_ham10000_dataset_report(split_df)
    print(f"Prepared split file: {split_path}")
    print(f"Samples={report['num_samples']} Groups={report['num_groups']} GroupsWithMultipleImages={report['groups_with_multiple_images']}")


if __name__ == "__main__":
    main()
