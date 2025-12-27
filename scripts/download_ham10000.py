"""Download the HAM10000 / ISIC 2018 Task 3 training collection to a target directory.

This script uses the public ISIC Archive ZIP download endpoint for collection 66,
which corresponds to the 10,015-image ISIC 2018 Challenge Task 3 training set.
It reshapes the extracted files into the layout expected by TrustQueryNet:

<output-root>/
  HAM10000_metadata.csv
  images/
    *.jpg
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path


ISIC_ZIP_URL_ENDPOINT = "https://api.isic-archive.com/api/v2/zip-download/url/"
DEFAULT_COLLECTION_ID = "66"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        default="data/ham10000",
        help="Directory that will receive HAM10000_metadata.csv and images/.",
    )
    parser.add_argument(
        "--collection-id",
        default=DEFAULT_COLLECTION_ID,
        help="ISIC collection id to download. Defaults to the 2018 Task 3 training collection (66).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing metadata/images if they already exist.",
    )
    parser.add_argument(
        "--keep-zip",
        action="store_true",
        help="Keep the downloaded ZIP file in the output root for debugging or reuse.",
    )
    return parser.parse_args()


def request_zip_url(collection_id: str) -> str:
    payload = json.dumps({"collections": str(collection_id)}).encode("utf-8")
    request = urllib.request.Request(
        ISIC_ZIP_URL_ENDPOINT,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request) as response:
        body = response.read().decode("utf-8")
    return json.loads(body)


def download_file(url: str, destination: Path) -> None:
    with urllib.request.urlopen(url) as response, destination.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def find_metadata_csv(root: Path) -> Path:
    candidates = sorted(root.rglob("*.csv"))
    if not candidates:
        raise FileNotFoundError("No CSV metadata file found in extracted archive.")

    preferred = [path for path in candidates if "metadata" in path.name.lower()]
    if preferred:
        return preferred[0]
    return max(candidates, key=lambda path: path.stat().st_size)


def collect_image_files(root: Path) -> list[Path]:
    image_paths = [
        *root.rglob("*.jpg"),
        *root.rglob("*.jpeg"),
        *root.rglob("*.png"),
    ]
    if not image_paths:
        raise FileNotFoundError("No image files found in extracted archive.")
    return sorted(image_paths)


def ensure_clean_target(output_root: Path, *, force: bool) -> tuple[Path, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    metadata_target = output_root / "HAM10000_metadata.csv"
    images_target = output_root / "images"

    if metadata_target.exists():
        if not force:
            raise FileExistsError(
                f"Target metadata already exists: {metadata_target}. Use --force to replace it."
            )
        metadata_target.unlink()

    if images_target.exists():
        if not force:
            raise FileExistsError(
                f"Target images directory already exists: {images_target}. Use --force to replace it."
            )
        shutil.rmtree(images_target)

    images_target.mkdir(parents=True, exist_ok=True)
    return metadata_target, images_target


def short_circuit_if_ready(output_root: Path, *, force: bool) -> bool:
    metadata_target = output_root / "HAM10000_metadata.csv"
    images_target = output_root / "images"
    if force:
        return False

    if metadata_target.exists() and images_target.exists():
        image_count = len(list(images_target.glob("*.jpg")))
        if image_count:
            print(f"Reusing existing dataset at {output_root}")
            print(f"metadata={metadata_target}")
            print(f"images={images_target}")
            print(f"image_count={image_count}")
            return True
    return False


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).resolve()

    if short_circuit_if_ready(output_root, force=args.force):
        return

    metadata_target, images_target = ensure_clean_target(output_root, force=args.force)

    with tempfile.TemporaryDirectory(prefix="ham10000-download-") as tmpdir:
        tmp_root = Path(tmpdir)
        archive_path = tmp_root / "ham10000_collection.zip"
        extracted_root = tmp_root / "extracted"
        extracted_root.mkdir(parents=True, exist_ok=True)

        print(f"Requesting ZIP url for collection {args.collection_id}...")
        zip_url = request_zip_url(args.collection_id)
        print(f"Downloading collection ZIP to {archive_path}...")
        download_file(zip_url, archive_path)

        print("Extracting archive...")
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(extracted_root)

        metadata_source = find_metadata_csv(extracted_root)
        image_sources = collect_image_files(extracted_root)

        shutil.move(str(metadata_source), str(metadata_target))
        for image_path in image_sources:
            shutil.move(str(image_path), str(images_target / image_path.name))

        if args.keep_zip:
            kept_zip_path = output_root / archive_path.name
            shutil.copy2(archive_path, kept_zip_path)
            print(f"Kept ZIP at {kept_zip_path}")

    image_count = len(list(images_target.glob("*.jpg")))
    if image_count == 0:
        image_count = len(list(images_target.glob("*.jpeg"))) + len(list(images_target.glob("*.png")))

    print("HAM10000 download complete.")
    print(f"metadata={metadata_target}")
    print(f"images={images_target}")
    print(f"image_count={image_count}")


if __name__ == "__main__":
    main()
