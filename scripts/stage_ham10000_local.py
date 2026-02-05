"""Stage HAM10000 from a persistent source directory into local runtime storage.

This is primarily intended for Colab runs where reading thousands of images
directly from Google Drive can cause DataLoader worker failures. The staged
layout matches the paths expected by TrustQueryNet:

<target-root>/
  HAM10000_metadata.csv
  splits.csv            # if present in the source root
  images/
    *.jpg
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        required=True,
        help="Directory containing HAM10000_metadata.csv and images/.",
    )
    parser.add_argument(
        "--target-root",
        default="/content/HAM10000-local",
        help="Local runtime directory that will receive the staged dataset.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete and recreate the target directory before staging.",
    )
    return parser.parse_args()


def count_images(image_dir: Path) -> int:
    count = 0
    for pattern in ("*.jpg", "*.jpeg", "*.png"):
        count += len(list(image_dir.glob(pattern)))
    return count


def copy_tree_python(source_dir: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for source_path in sorted(source_dir.iterdir()):
        destination = target_dir / source_path.name
        if destination.exists():
            continue
        shutil.copy2(source_path, destination)


def copy_tree(source_dir: Path, target_dir: Path) -> str:
    rsync_path = shutil.which("rsync")
    if rsync_path:
        subprocess.run(
            [rsync_path, "-a", "--ignore-existing", f"{source_dir}/", f"{target_dir}/"],
            check=True,
        )
        return "rsync"

    copy_tree_python(source_dir, target_dir)
    return "python-copy"


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_root).resolve()
    target_root = Path(args.target_root).resolve()

    if not source_root.exists():
        raise FileNotFoundError(f"Source root not found: {source_root}")

    source_metadata = source_root / "HAM10000_metadata.csv"
    source_images = source_root / "images"
    source_splits = source_root / "splits.csv"

    if not source_metadata.exists():
        raise FileNotFoundError(f"Missing source metadata: {source_metadata}")
    if not source_images.exists():
        raise FileNotFoundError(f"Missing source images directory: {source_images}")

    if args.force and target_root.exists():
        shutil.rmtree(target_root)

    target_root.mkdir(parents=True, exist_ok=True)
    target_metadata = target_root / "HAM10000_metadata.csv"
    target_images = target_root / "images"
    target_images.mkdir(parents=True, exist_ok=True)
    target_splits = target_root / "splits.csv"

    shutil.copy2(source_metadata, target_metadata)
    if source_splits.exists():
        shutil.copy2(source_splits, target_splits)

    copy_method = copy_tree(source_images, target_images)

    source_count = count_images(source_images)
    target_count = count_images(target_images)
    if target_count == 0:
        raise RuntimeError(f"No staged images found in {target_images}")

    print(f"source_root={source_root}")
    print(f"target_root={target_root}")
    print(f"copy_method={copy_method}")
    print(f"metadata={target_metadata}")
    if target_splits.exists():
        print(f"splits={target_splits}")
    print(f"source_image_count={source_count}")
    print(f"target_image_count={target_count}")


if __name__ == "__main__":
    main()
