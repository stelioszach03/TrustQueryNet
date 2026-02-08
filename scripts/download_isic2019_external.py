"""Download the official ISIC 2019 test set for external validation."""

from __future__ import annotations

import argparse
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path


ISIC2019_TEST_INPUT_URL = "https://isic-archive.s3.amazonaws.com/challenges/2019/ISIC_2019_Test_Input.zip"
ISIC2019_TEST_GROUND_TRUTH_URL = "https://isic-archive.s3.amazonaws.com/challenges/2019/ISIC_2019_Test_GroundTruth.csv"
ISIC2019_TEST_METADATA_URL = "https://isic-archive.s3.amazonaws.com/challenges/2019/ISIC_2019_Test_Metadata.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        default="data/isic2019_external_test",
        help="Directory that will receive images/ plus the metadata/ground-truth CSVs.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing files in the output root.",
    )
    return parser.parse_args()


def download_file(url: str, destination: Path) -> None:
    with urllib.request.urlopen(url) as response, destination.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def short_circuit_if_ready(output_root: Path, *, force: bool) -> bool:
    if force:
        return False
    expected = [
        output_root / "ISIC_2019_Test_GroundTruth.csv",
        output_root / "ISIC_2019_Test_Metadata.csv",
        output_root / "images",
    ]
    if all(path.exists() for path in expected):
        image_count = len(list((output_root / "images").glob("*.jpg")))
        if image_count:
            print(f"Reusing existing ISIC 2019 external test set at {output_root}")
            print(f"image_count={image_count}")
            return True
    return False


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).resolve()

    if short_circuit_if_ready(output_root, force=args.force):
        return

    if args.force and output_root.exists():
        shutil.rmtree(output_root)

    output_root.mkdir(parents=True, exist_ok=True)
    images_dir = output_root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    gt_path = output_root / "ISIC_2019_Test_GroundTruth.csv"
    metadata_path = output_root / "ISIC_2019_Test_Metadata.csv"

    with tempfile.TemporaryDirectory(prefix="isic2019-external-") as tmpdir:
        tmp_root = Path(tmpdir)
        archive_path = tmp_root / "ISIC_2019_Test_Input.zip"

        print("Downloading ISIC 2019 test images...")
        download_file(ISIC2019_TEST_INPUT_URL, archive_path)
        print("Downloading ISIC 2019 test ground truth...")
        download_file(ISIC2019_TEST_GROUND_TRUTH_URL, gt_path)
        print("Downloading ISIC 2019 test metadata...")
        download_file(ISIC2019_TEST_METADATA_URL, metadata_path)

        print("Extracting image archive...")
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(images_dir)

        extracted_images = [
            path
            for path in images_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]
        for image_path in extracted_images:
            if image_path.parent == images_dir:
                continue
            destination = images_dir / image_path.name
            if destination.exists():
                continue
            shutil.move(str(image_path), str(destination))

    image_count = len(list(images_dir.rglob("*.jpg")))
    print(f"output_root={output_root}")
    print(f"image_count={image_count}")
    print(f"ground_truth={gt_path}")
    print(f"metadata={metadata_path}")


if __name__ == "__main__":
    main()
