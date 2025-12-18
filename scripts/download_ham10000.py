"""Validate a local HAM10000 layout."""

from __future__ import annotations

from pathlib import Path


def main() -> None:
    root = Path("data/ham10000")
    metadata = root / "HAM10000_metadata.csv"
    images = root / "images"
    print(f"metadata_exists={metadata.exists()}")
    print(f"images_exists={images.exists()}")
    if images.exists():
        jpg_count = len(list(images.glob("*.jpg")))
        print(f"image_count={jpg_count}")


if __name__ == "__main__":
    main()
