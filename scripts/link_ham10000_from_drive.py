"""Link a Drive-backed HAM10000 directory into the local project layout."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Link a HAM10000 folder into data/ham10000.")
    parser.add_argument(
        "--source",
        required=True,
        help="Path to a directory containing HAM10000_metadata.csv and an images/ folder.",
    )
    parser.add_argument(
        "--target",
        default="data/ham10000",
        help="Project-local target directory. Defaults to data/ham10000.",
    )
    args = parser.parse_args()

    source = Path(args.source).expanduser().resolve()
    target = Path(args.target).expanduser().resolve()

    metadata = source / "HAM10000_metadata.csv"
    images = source / "images"
    if not metadata.exists():
        raise FileNotFoundError(f"Missing metadata CSV: {metadata}")
    if not images.exists():
        raise FileNotFoundError(f"Missing image directory: {images}")

    target.mkdir(parents=True, exist_ok=True)
    metadata_link = target / "HAM10000_metadata.csv"
    images_link = target / "images"

    if metadata_link.exists() or metadata_link.is_symlink():
        metadata_link.unlink()
    if images_link.exists() or images_link.is_symlink():
        if images_link.is_symlink():
            images_link.unlink()
        else:
            raise FileExistsError(f"Target path exists and is not a symlink: {images_link}")

    metadata_link.symlink_to(metadata)
    images_link.symlink_to(images, target_is_directory=True)

    print(f"Linked metadata -> {metadata_link}")
    print(f"Linked images   -> {images_link}")


if __name__ == "__main__":
    main()
